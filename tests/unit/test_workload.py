"""The workload: the corpus, the effect-typed tools over it, and the generated tasks.

The benchmark's credibility rests on this module being boring and honest: deterministic fixtures,
tools that behave the same whether or not they are being speculated, and reference trajectories
whose arguments a real run could actually have produced.
"""

from __future__ import annotations

import pytest

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.core.canonical import canonical_json
from speculative_tool_executor.spec.clock import VirtualClock
from speculative_tool_executor.spec.context import (
    ExecutionContext,
    current_execution,
    use_execution,
)
from speculative_tool_executor.spec.overlay import WriteOverlay
from speculative_tool_executor.spec.predictor import index_scalars
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.workload.corpus import (
    WorkspaceError,
    build_corpus,
    note_key_for,
    outline_text,
    tally_items,
)
from speculative_tool_executor.workload.tasks import (
    SHAPE_WEIGHTS,
    entries_of,
    generate_tasks,
    pick_entry,
    shape_counts,
    training_trajectories,
)
from speculative_tool_executor.workload.tools import (
    ALL_TOOLS,
    TOOL_PROFILES,
    build_canonicalizer,
    build_registry,
)

pytestmark = pytest.mark.unit

CORPUS = build_corpus()


# -- corpus -------------------------------------------------------------------------------------


def test_the_corpus_is_deterministic() -> None:
    assert build_corpus().files == build_corpus().files
    assert build_corpus(seed=1).files != CORPUS.files


def test_outputs_carry_the_arguments_of_the_calls_that_follow_them() -> None:
    """This is what lets the predictor derive a concrete next call instead of guessing one."""
    listing = CORPUS.list_dir("src/auth")
    entry = entries_of(listing)[0]
    read = CORPUS.read_file(entry)

    index = index_scalars(listing)
    assert canonical_json(entry) in index

    read_index = index_scalars(read)
    for field in ("note_key", "index_key", "summary"):
        assert canonical_json(read[field]) in read_index


def test_note_keys_are_derived_not_assembled_by_the_agent() -> None:
    assert note_key_for("src/auth/api.py") == "notes/src.auth.api.py"
    assert CORPUS.read_file("src/auth/api.py")["note_key"] == note_key_for("src/auth/api.py")


def test_missing_files_and_directories_are_refused() -> None:
    with pytest.raises(WorkspaceError, match="file not found"):
        CORPUS.read_file("src/auth/nope.py")
    with pytest.raises(WorkspaceError, match="directory not found"):
        CORPUS.list_dir("src/nope")
    with pytest.raises(WorkspaceError, match="must not be empty"):
        CORPUS.search_code("   ")
    with pytest.raises(WorkspaceError, match="document not found"):
        CORPUS.fetch_doc("https://docs.internal.test/guide/nope")
    with pytest.raises(WorkspaceError, match="1 and 200"):
        CORPUS.read_file("x" * 201)


def test_pure_helpers_depend_only_on_their_arguments() -> None:
    text = "One thing. Another thing; a third."
    assert outline_text(text) == outline_text(text)
    assert tally_items(["a", "b", "a"]) == {"distinct": 2, "total": 3, "top": ["a", "b"]}
    assert outline_text("")["summary"] == ""


def test_search_is_case_and_whitespace_insensitive_like_its_canonicalizer_claims() -> None:
    assert CORPUS.search_code("AUTH_API") == CORPUS.search_code("auth_api")


# -- tools --------------------------------------------------------------------------------------


def test_every_tool_declares_an_effect_and_the_lattice_is_covered() -> None:
    registry, _ = build_registry(CORPUS)
    assert set(registry.names) == set(ALL_TOOLS)
    effects = {registry.get(name).effect for name in registry.names}
    assert effects == {Effect.PURE, Effect.READ, Effect.WRITE, Effect.EXTERNAL}


def test_only_the_external_tool_needs_approval() -> None:
    registry, _ = build_registry(CORPUS)
    needing = {name for name in registry.names if registry.get(name).needs_approval}
    assert needing == {"send_report"}


def test_non_pure_tools_bound_the_life_of_a_speculative_result() -> None:
    registry, _ = build_registry(CORPUS)
    for name in registry.names:
        tool = registry.get(name)
        if tool.effect is Effect.PURE:
            assert tool.freshness_seconds is None
        else:
            assert tool.freshness_seconds is not None
            assert tool.freshness_seconds > 0


def test_every_tool_has_a_declared_latency_profile() -> None:
    assert set(TOOL_PROFILES) == set(ALL_TOOLS)
    assert all(profile.median_seconds > 0 for profile in TOOL_PROFILES.values())


def test_the_canonicalizer_declares_a_rule_only_for_tools_that_exist() -> None:
    canonicalizer = build_canonicalizer()
    assert set(canonicalizer.tools) <= set(ALL_TOOLS)
    # send_report is deliberately un-normalized: it is never matched against a speculation.
    assert "send_report" not in canonicalizer.tools


def test_write_tools_go_through_the_branch_overlay_not_the_store() -> None:
    """A write must be invisible until the branch commits, which is what makes it speculatable."""
    registry, _ = build_registry(CORPUS)
    with VersionedStore(":memory:", synchronous="OFF") as store:
        overlay = WriteOverlay(store=store, branch_id="spec")
        validated = registry.validate_call(
            ToolUseBlock(id="x", name="save_note", arguments={"key": "notes/a", "text": "hello"})
        )
        context = ExecutionContext(
            overlay=overlay, clock=VirtualClock(), branch_id="spec", speculative=True
        )
        with use_execution(context):
            output = validated.run()

        assert output == {"key": "notes/a", "stored": True, "replaced": False, "length": 5}
        assert store.get("notes/a") is None, "the write reached the store before committing"
        assert overlay.writes == {"notes/a": "hello"}
        assert overlay.commit().committed is True
        assert store.get("notes/a").value == "hello"  # type: ignore[union-attr]


def test_bump_index_records_the_version_it_read_so_the_increment_cannot_be_lost() -> None:
    registry, _ = build_registry(CORPUS)
    with VersionedStore(":memory:", synchronous="OFF") as store:
        store.put("index/auth", "4")
        overlay = WriteOverlay(store=store, branch_id="spec")
        validated = registry.validate_call(
            ToolUseBlock(id="x", name="bump_index", arguments={"key": "index/auth"})
        )
        with use_execution(
            ExecutionContext(
                overlay=overlay, clock=VirtualClock(), branch_id="spec", speculative=True
            )
        ):
            assert validated.run() == {"key": "index/auth", "count": 5}

        assert overlay.read_set == {"index/auth": store.get("index/auth").version}  # type: ignore[union-attr]

        # Another writer increments first; the branch must be refused rather than clobber it.
        store.put("index/auth", "5")
        assert overlay.commit().committed is False
        assert store.get("index/auth").value == "5"  # type: ignore[union-attr]


def test_a_handler_outside_an_execution_has_no_branch_to_write_to() -> None:
    with pytest.raises(RuntimeError, match="no execution context"):
        current_execution()


# -- tasks --------------------------------------------------------------------------------------


def test_generation_is_deterministic_in_its_seed() -> None:
    left = generate_tasks(count=30, corpus=CORPUS)
    right = generate_tasks(count=30, corpus=CORPUS)
    assert [task.task_id for task in left] == [task.task_id for task in right]
    assert [task.calls for task in left] == [task.calls for task in right]
    assert generate_tasks(count=30, corpus=CORPUS, seed=99) != left


def test_the_realised_shape_mix_matches_the_declared_weights() -> None:
    tasks = generate_tasks(count=300, corpus=CORPUS)
    counts = shape_counts(tasks)
    assert sum(counts.values()) == 300
    total = sum(SHAPE_WEIGHTS.values())
    for shape, weight in SHAPE_WEIGHTS.items():
        expected = 300 * weight / total
        assert abs(counts[shape] - expected) <= 1.0


def test_every_reference_call_validates_against_the_real_tool_schema() -> None:
    """The generated trajectories are calls the registry accepts, not a parallel vocabulary."""
    registry, _ = build_registry(CORPUS)
    checked = 0
    for task in generate_tasks(count=60, corpus=CORPUS):
        for index, call in enumerate(task.calls):
            validated = registry.validate_call(
                ToolUseBlock(id=f"c{index}", name=call.tool, arguments=dict(call.arguments))
            )
            assert validated.tool.name == call.tool
            assert len(validated.call_hash) == 64
            checked += 1
    assert checked > 100, "the generator must actually have produced calls to check"


def test_every_speculatable_step_is_fully_derivable_from_the_previous_output() -> None:
    """The predictor derives arguments; a step it cannot derive is one it can never get right.

    The property is scoped to speculatable tools on purpose. ``send_report`` composes a subject
    and a body out of what the model read -- ordinary model output, not a derivation -- and it is
    never speculated, so requiring it to be derivable would be requiring the wrong thing. The
    companion test below pins that exclusion down so it cannot quietly widen.
    """
    registry, _ = build_registry(CORPUS)
    for task in generate_tasks(count=60, corpus=CORPUS):
        for index, call in enumerate(task.calls[1:], start=1):
            if not registry.get(call.tool).speculatable:
                continue
            previous = task.calls[index - 1].output
            assert previous is not None, f"{task.task_id} step {index} follows a blind step"
            available = index_scalars(previous)
            undrivable = [
                name
                for name, value in call.arguments.items()
                if canonical_json(value) not in available
            ]
            assert not undrivable, (
                f"{task.task_id} step {index} ({call.tool}) cannot derive {undrivable}"
            )


def test_only_external_steps_are_allowed_to_be_underivable() -> None:
    registry, _ = build_registry(CORPUS)
    underivable_tools = set()
    for task in generate_tasks(count=120, corpus=CORPUS):
        for index, call in enumerate(task.calls[1:], start=1):
            previous = task.calls[index - 1].output
            if previous is None:
                continue
            available = index_scalars(previous)
            if any(canonical_json(v) not in available for v in call.arguments.values()):
                underivable_tools.add(call.tool)
    assert underivable_tools <= {"send_report"}
    assert all(not registry.get(name).speculatable for name in underivable_tools)


def test_tasks_are_multi_step_and_the_first_call_is_never_predictable() -> None:
    tasks = generate_tasks(count=60, corpus=CORPUS)
    assert all(len(task.calls) >= 2 for task in tasks)
    assert all(task.prompt for task in tasks)
    # Every task starts from an entry point that depends on nothing prior.
    starts = {task.calls[0].tool for task in tasks}
    assert starts <= {"list_dir", "search_code", "fetch_doc"}


def test_only_the_report_shape_reaches_outside_the_system() -> None:
    tasks = generate_tasks(count=120, corpus=CORPUS)
    for task in tasks:
        if task.external_calls:
            assert task.shape == "doc_report"
        assert ("send_report" in task.allowed_tools) == bool(task.external_calls)


def test_training_tasks_are_disjoint_from_the_evaluation_tasks() -> None:
    evaluation = {task.task_id for task in generate_tasks(count=300, corpus=CORPUS)}
    trajectories = training_trajectories(count=120, corpus=CORPUS)
    assert len(trajectories) == 120
    assert all(call for call in trajectories)
    training_ids = {
        task.task_id for task in generate_tasks(count=120, seed=777, corpus=CORPUS, prefix="train")
    }
    assert not (evaluation & training_ids)


def test_read_bias_concentrates_the_choice_without_eliminating_it() -> None:
    import random

    entries = [f"f{i}.py" for i in range(4)]
    uniform = [pick_entry(entries, random.Random(i), 0.0) for i in range(400)]
    biased = [pick_entry(entries, random.Random(i), 2.0) for i in range(400)]

    assert uniform.count(entries[0]) < biased.count(entries[0])
    assert len(set(biased)) == len(entries), "bias must not make the choice deterministic"
    assert pick_entry(["only.py"], random.Random(0), 2.0) == "only.py"


def test_a_negative_read_bias_or_count_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        generate_tasks(count=-1, corpus=CORPUS)
    with pytest.raises(ValueError, match="read_bias"):
        generate_tasks(count=1, corpus=CORPUS, read_bias=-1.0)
