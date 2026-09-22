"""The three things that decide *what* a speculation is: its address, its guess and its reach.

- the cache is content-addressed, so a result can only be handed to a call that would have
  computed exactly it;
- the predictor derives arguments from the previous output rather than memorising them, and drops
  a guess it cannot derive instead of inventing one;
- the taint rule bounds what a prediction may name, so a bad predictor cannot widen the agent's
  reach on its own.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import JsonValue

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.core.canonical import canonical_json
from speculative_tool_executor.spec.cache import (
    SpeculativeCache,
    SpeculativeResult,
    freshness_deadline,
)
from speculative_tool_executor.spec.overlay import BranchState, WriteOverlay
from speculative_tool_executor.spec.predictor import (
    ArgOrigin,
    MarkovPredictor,
    NullPredictor,
    ObservedCall,
    PredictionContext,
    call_signature,
    index_scalars,
    resolve_path,
)
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.spec.taint import TaintTracker, schema_constants

pytestmark = pytest.mark.unit


def result(
    key: str = "k",
    *,
    effect: Effect = Effect.READ,
    overlay: WriteOverlay | None = None,
    expires_at: float | None = None,
    tool: str = "read_file",
) -> SpeculativeResult:
    return SpeculativeResult(
        key=key,
        tool=tool,
        effect=effect,
        branch_id=f"spec:{key}",
        outcome="ok",
        output={"value": key},
        overlay=overlay,
        expires_at=expires_at,
    )


# -- cache --------------------------------------------------------------------------------------


def test_read_and_write_results_are_single_use_but_pure_results_are_not() -> None:
    cache = SpeculativeCache()
    cache.put(result("read", effect=Effect.READ))
    cache.put(result("pure", effect=Effect.PURE))

    assert cache.take("read", now=0.0) is not None
    assert cache.take("read", now=0.0) is None, "a read branch is consumed when claimed"

    assert cache.take("pure", now=0.0) is not None
    assert cache.take("pure", now=0.0) is not None, "a pure result depends on nothing else"


def test_a_miss_is_a_miss_and_is_counted() -> None:
    cache = SpeculativeCache()
    assert cache.take("absent", now=0.0) is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


def test_expiry_is_measured_when_the_authoritative_call_arrives() -> None:
    cache = SpeculativeCache()
    cache.put(result("k", expires_at=10.0))
    assert cache.take("k", now=9.99) is not None

    cache.put(result("k", expires_at=10.0))
    assert cache.take("k", now=10.0) is None
    assert cache.stats.expired == 1


def test_eviction_aborts_the_branch_it_drops() -> None:
    with VersionedStore(":memory:", synchronous="OFF") as store:
        overlay = WriteOverlay(store=store, branch_id="doomed")
        overlay.write("a", "1")
        cache = SpeculativeCache(max_entries=1)
        cache.put(result("first", overlay=overlay))
        cache.put(result("second"))

        assert cache.stats.evicted == 1
        assert overlay.state is BranchState.ABORTED
        assert store.get("a") is None


def test_replacing_an_entry_aborts_the_branch_it_displaces() -> None:
    with VersionedStore(":memory:", synchronous="OFF") as store:
        overlay = WriteOverlay(store=store, branch_id="old")
        overlay.write("a", "1")
        cache = SpeculativeCache()
        cache.put(result("k", overlay=overlay))
        cache.put(result("k"))
        assert overlay.state is BranchState.ABORTED


def test_clear_returns_what_it_dropped_so_waste_can_be_billed() -> None:
    cache = SpeculativeCache()
    cache.put(result("a"))
    cache.put(result("b"))
    dropped = cache.clear()
    assert sorted(item.key for item in dropped) == ["a", "b"]
    assert len(cache) == 0


def test_purge_expired_drops_only_stale_entries() -> None:
    cache = SpeculativeCache()
    cache.put(result("stale", expires_at=1.0))
    cache.put(result("fresh", expires_at=100.0))
    assert cache.purge_expired(now=50.0) == 1
    assert cache.keys == ("fresh",)


def test_per_tool_hit_ratio_tracks_attempts() -> None:
    cache = SpeculativeCache()
    cache.stats.record_attempt("read_file")
    cache.stats.record_attempt("read_file")
    cache.put(result("k", tool="read_file"))
    cache.take("k", now=0.0)
    assert cache.stats.hit_ratio("read_file") == 0.5
    assert cache.stats.hit_ratio("never_seen") == 0.0
    assert cache.stats.per_tool()["read_file"]["attempts"] == 2.0


def test_freshness_deadline_rejects_nonsense_windows() -> None:
    assert freshness_deadline(5.0, None) is None
    assert freshness_deadline(5.0, 2.0) == 7.0
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="positive, finite"):
            freshness_deadline(5.0, bad)


def test_cache_requires_a_positive_bound() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        SpeculativeCache(max_entries=0)


# -- predictor ----------------------------------------------------------------------------------

LISTING: JsonValue = {"path": "src/a", "entries": ["src/a/x.py", "src/a/y.py"], "count": 2}
READ: JsonValue = {"path": "src/a/x.py", "summary": "x does things", "note_key": "notes/x"}


def trajectory() -> tuple[ObservedCall, ...]:
    return (
        ObservedCall(tool="list_dir", arguments={"path": "src/a"}, output=LISTING),
        ObservedCall(tool="read_file", arguments={"path": "src/a/x.py"}, output=READ),
        ObservedCall(
            tool="save_note", arguments={"key": "notes/x", "text": "x does things"}, output=None
        ),
    )


async def test_arguments_are_derived_from_the_previous_output_not_memorised() -> None:
    """The template must generalise to a listing it has never seen."""
    predictor = MarkovPredictor(order=2).train([trajectory()])
    other: JsonValue = {"path": "src/b", "entries": ["src/b/z.py"], "count": 1}

    predictions = await predictor.predict(
        PredictionContext(history=("list_dir(path)",), last_output=other), top_k=3
    )
    read = next(p for p in predictions if p.tool == "read_file")
    assert read.arguments == {"path": "src/b/z.py"}
    assert read.origins["path"] is ArgOrigin.OBSERVED


async def test_a_template_whose_path_no_longer_resolves_is_dropped_not_guessed() -> None:
    predictor = MarkovPredictor(order=2).train([trajectory()])
    unrelated: JsonValue = {"totally": "different"}
    predictions = await predictor.predict(
        PredictionContext(history=("list_dir(path)",), last_output=unrelated), top_k=5
    )
    assert all(p.tool != "read_file" for p in predictions)


async def test_longer_contexts_outrank_shorter_ones() -> None:
    """Back-off: a specific match must beat a generic one of the same raw frequency."""
    specific = (
        ObservedCall(tool="search_code", arguments={"q": "x"}, output=READ),
        ObservedCall(tool="outline", arguments={"text": "x does things"}, output=None),
    )
    predictor = MarkovPredictor(order=2, backoff=0.4).train([trajectory(), specific])
    predictions = await predictor.predict(
        PredictionContext(history=("search_code(q)",), last_output=READ), top_k=5
    )
    assert predictions[0].tool == "outline"
    assert predictions[0].probability > 0.9


async def test_null_predictor_predicts_nothing() -> None:
    assert await NullPredictor().predict(PredictionContext(), top_k=5) == []


async def test_top_k_zero_or_negative_returns_nothing() -> None:
    predictor = MarkovPredictor().train([trajectory()])
    assert await predictor.predict(PredictionContext(history=("list_dir(path)",)), top_k=0) == []


async def test_training_is_additive_and_deterministic() -> None:
    first = MarkovPredictor(order=2).train([trajectory()])
    second = MarkovPredictor(order=2).train([trajectory()])
    context = PredictionContext(history=("list_dir(path)",), last_output=LISTING)
    left = await first.predict(context, top_k=5)
    right = await second.predict(context, top_k=5)
    assert [p.sort_key() for p in left] == [p.sort_key() for p in right]
    assert first.transitions == 2
    assert first.contexts > 0


def test_predictor_rejects_nonsense_configuration() -> None:
    for kwargs in ({"order": -1}, {"backoff": 0.0}, {"backoff": 1.5}, {"min_count": 0}):
        with pytest.raises(ValueError, match="must be"):
            MarkovPredictor(**kwargs)


def test_call_signature_ignores_argument_values() -> None:
    assert call_signature("read", {"path": "a"}) == call_signature("read", {"path": "b"})
    assert call_signature("read", {"path": "a"}) != call_signature("read", {"other": "a"})


def test_resolve_path_returns_one_sentinel_for_anything_that_no_longer_exists() -> None:
    """A derivation that cannot be resolved must be unusable, not merely falsy.

    Returning ``None`` would be indistinguishable from a genuine JSON null and could be passed
    to a tool as an argument; the sentinel cannot be.
    """
    index = index_scalars(LISTING)
    assert resolve_path(LISTING, index[canonical_json("src/a/y.py")]) == "src/a/y.py"

    out_of_range = resolve_path(LISTING, ("entries", 99))
    absent_key = resolve_path(LISTING, ("missing",))
    wrong_shape = resolve_path(LISTING, ("count", "nope"))

    assert out_of_range is absent_key is wrong_shape
    assert not isinstance(out_of_range, str | int | float | bool | list | dict)
    assert out_of_range is not None


def test_index_scalars_indexes_lists_as_well_as_scalars() -> None:
    index = index_scalars(LISTING)
    assert canonical_json(["src/a/x.py", "src/a/y.py"]) in index


# -- taint --------------------------------------------------------------------------------------


def test_only_values_the_model_has_seen_may_be_used() -> None:
    tracker = TaintTracker()
    tracker.observe({"paths": ["src/a/x.py"]})

    assert tracker.check({"path": "src/a/x.py"}).ok is True
    report = tracker.check({"path": "/etc/passwd"})
    assert report.ok is False
    assert report.fields == ("path",)
    assert "path" in report.reason()


def test_schema_constants_are_allowed_without_having_been_uttered() -> None:
    tracker = TaintTracker()
    constants = schema_constants({"properties": {"recipient": {"enum": ["security", "docs-team"]}}})
    assert tracker.check({"recipient": "security"}, constants=constants).ok is True
    assert tracker.check({"recipient": "attacker"}, constants=constants).ok is False


def test_numbers_and_booleans_are_exempt() -> None:
    """Paging arguments cannot name a file, a host or a recipient."""
    tracker = TaintTracker()
    assert tracker.check({"limit": 50, "recursive": True, "ratio": 0.5}).ok is True


def test_nested_values_are_checked() -> None:
    tracker = TaintTracker()
    tracker.observe("known")
    assert tracker.check({"a": {"b": ["known"]}}).ok is True
    assert tracker.check({"a": {"b": ["unknown"]}}).ok is False


def test_whitespace_tokens_of_observed_text_count_as_seen() -> None:
    tracker = TaintTracker()
    tracker.observe("please read src/a/x.py now")
    assert tracker.seen("src/a/x.py") is True
    assert tracker.seen("src/a/z.py") is False


def test_the_tracker_is_bounded_and_evicts_oldest_first() -> None:
    tracker = TaintTracker(max_values=3)
    for index in range(5):
        tracker.observe(f"v{index}")
    assert len(tracker) == 3
    assert tracker.seen("v0") is False
    assert tracker.seen("v4") is True
    with pytest.raises(ValueError, match="at least 1"):
        TaintTracker(max_values=0)


@given(values=st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=10))
def test_everything_observed_is_seen(values: list[str]) -> None:
    tracker = TaintTracker()
    tracker.observe_all(values)
    assert all(tracker.seen(value) for value in values)
    assert tracker.check({f"f{i}": v for i, v in enumerate(values)}).ok is True
