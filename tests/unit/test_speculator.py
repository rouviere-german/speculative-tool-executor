"""The speculation session: what it launches, what it refuses, and how it bills the difference.

A speculator that silently drops most of its candidates looks exactly like one that is working,
right up until you look at the hit rate. So every refusal is named and counted, and these tests
pin the names down: they are the vocabulary the metrics, the audit log and the README all use.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from speculative_tool_executor.agentic.approval import ApprovalGate, ApprovalStore
from speculative_tool_executor.agentic.audit import AuditLog
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.core.tracing import build_tracer_provider
from speculative_tool_executor.spec.cache import SpeculativeCache
from speculative_tool_executor.spec.clock import VirtualClock
from speculative_tool_executor.spec.controller import (
    DecisionReason,
    SpeculationController,
    SpeculationPolicy,
)
from speculative_tool_executor.spec.executor import SpeculativeToolExecutor
from speculative_tool_executor.spec.latency import LatencyModel
from speculative_tool_executor.spec.metrics import create_speculation_metrics
from speculative_tool_executor.spec.predictor import Prediction, PredictionContext
from speculative_tool_executor.spec.speculator import SpeculationSession
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.spec.taint import TaintTracker
from speculative_tool_executor.workload.corpus import build_corpus
from speculative_tool_executor.workload.tasks import entries_of
from speculative_tool_executor.workload.tools import (
    DEFAULT_TOOL_PROFILE,
    MODEL_TURN_PROFILE,
    TOOL_PROFILES,
    build_registry,
)

pytestmark = pytest.mark.unit

CORPUS = build_corpus()
LISTING = CORPUS.list_dir("src/auth")
FIRST_ENTRY = entries_of(LISTING)[0]


class FixedPredictor:
    """Proposes exactly what the test asks for, so refusals can be provoked deliberately."""

    name = "fixed"

    def __init__(self, *predictions: Prediction) -> None:
        self._predictions = list(predictions)

    async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]:
        del context
        return self._predictions[:top_k]


def prediction(tool: str, arguments: dict[str, object], probability: float = 0.9) -> Prediction:
    return Prediction(
        tool=tool,
        arguments=dict(arguments),  # type: ignore[arg-type]
        probability=probability,
        source="fixed",
    )


class Harness:
    """An executor wired for one test, with everything it needs to inspect afterwards."""

    def __init__(
        self,
        *predictions: Prediction,
        policy: SpeculationPolicy | None = None,
        audit: AuditLog | None = None,
        seen: tuple[str, ...] = (),
        tracer: Tracer | None = None,
    ) -> None:
        self.store = VersionedStore(":memory:", synchronous="OFF")
        self.registry, self.sink = build_registry(CORPUS)
        self.clock = VirtualClock()
        self.controller = SpeculationController(policy or SpeculationPolicy(top_k=2))
        self.taint = TaintTracker()
        self.taint.observe_all([LISTING, *seen])
        self.metrics = create_speculation_metrics()
        self.executor = SpeculativeToolExecutor(
            registry=self.registry,
            store=self.store,
            clock=self.clock,
            latency=LatencyModel(
                tools=TOOL_PROFILES,
                default_tool=DEFAULT_TOOL_PROFILE,
                model_turn=MODEL_TURN_PROFILE,
                seed=3,
            ),
            gate=ApprovalGate(ApprovalStore(ttl_seconds=1000.0), principal="agent"),
            controller=self.controller,
            predictor=FixedPredictor(*predictions),
            cache=SpeculativeCache(max_entries=16),
            taint=self.taint,
            audit=audit,
            metrics=self.metrics,
            tracer=tracer,
        )

    def close(self) -> None:
        self.store.close()

    @property
    def refusals(self) -> dict[str, int]:
        return self.controller.stats.refusals


async def session_of(harness: Harness) -> AsyncIterator[SpeculationSession]:
    async with harness.executor.session() as session:
        yield session


async def run_turn(harness: Harness) -> dict[str, int]:
    """Start one turn's speculation and let it settle, returning the refusal counters."""
    with harness.clock.participant():
        async with harness.executor.session() as session:
            session.start_turn(PredictionContext(history=("list_dir(path)",), last_output=LISTING))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            session.settle_turn()
    return dict(harness.refusals)


# -- launching ----------------------------------------------------------------------------------


async def test_a_usable_prediction_is_launched_and_charged_to_the_budget() -> None:
    harness = Harness(prediction("read_file", {"path": FIRST_ENTRY}))
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await asyncio.sleep(0)
                assert session.stats.launched == 1
                assert harness.controller.stats.attempts == 1
                assert harness.controller.stats.speculative_spend_usd > 0
    finally:
        harness.close()


async def test_top_k_counts_launches_not_guesses() -> None:
    """A refused candidate must not consume a slot a usable one could have had."""
    harness = Harness(
        prediction("send_report", {"recipient": "security", "subject": "s", "body": "b"}, 0.99),
        prediction("read_file", {"path": FIRST_ENTRY}, 0.8),
        policy=SpeculationPolicy(top_k=1, candidate_multiplier=4),
    )
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await asyncio.sleep(0)
                assert session.stats.launched == 1
        assert harness.controller.tool_stats("read_file").attempts == 1
        assert harness.controller.tool_stats("send_report").attempts == 0
    finally:
        harness.close()


# -- refusals, each with its own name ------------------------------------------------------------


async def test_an_unknown_tool_is_refused() -> None:
    harness = Harness(prediction("no_such_tool", {"path": FIRST_ENTRY}))
    try:
        assert (await run_turn(harness)).get(DecisionReason.UNKNOWN_TOOL) == 1
    finally:
        harness.close()


async def test_invalid_arguments_are_refused_by_the_same_schema_a_real_call_faces() -> None:
    harness = Harness(prediction("read_file", {"path": 42}))
    try:
        assert (await run_turn(harness)).get(DecisionReason.INVALID_ARGUMENTS) == 1
    finally:
        harness.close()


async def test_an_external_tool_is_refused_by_effect() -> None:
    harness = Harness(
        prediction("send_report", {"recipient": "security", "subject": "s", "body": "b"}),
        seen=("s", "b"),
    )
    try:
        assert (await run_turn(harness)).get(DecisionReason.EFFECT_NOT_SPECULATABLE) == 1
    finally:
        harness.close()


async def test_an_argument_the_model_never_saw_is_refused_by_taint() -> None:
    harness = Harness(prediction("read_file", {"path": "src/secret/keys.py"}))
    try:
        assert (await run_turn(harness)).get(DecisionReason.TAINTED_ARGUMENTS) == 1
    finally:
        harness.close()


async def test_a_call_already_known_is_not_speculated_twice() -> None:
    harness = Harness(
        prediction("read_file", {"path": FIRST_ENTRY}, 0.9),
        prediction("read_file", {"path": FIRST_ENTRY}, 0.8),
        policy=SpeculationPolicy(top_k=2, candidate_multiplier=2),
    )
    try:
        assert (await run_turn(harness)).get(DecisionReason.ALREADY_KNOWN) == 1
    finally:
        harness.close()


async def test_max_in_flight_bounds_concurrent_branches() -> None:
    entries = entries_of(LISTING)
    harness = Harness(
        *[prediction("read_file", {"path": entry}, 0.9) for entry in entries],
        policy=SpeculationPolicy(top_k=4, max_in_flight=1, candidate_multiplier=2),
    )
    try:
        refusals = await run_turn(harness)
        assert refusals.get(DecisionReason.MAX_IN_FLIGHT, 0) >= 1
    finally:
        harness.close()


# -- settling ------------------------------------------------------------------------------------


async def test_a_claimed_branch_is_committed_and_the_rest_are_cancelled() -> None:
    entries = entries_of(LISTING)
    harness = Harness(
        prediction("read_file", {"path": entries[0]}, 0.9),
        prediction("read_file", {"path": entries[1]}, 0.8),
    )
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)  # let both branches finish
                outcome = await harness.executor.execute(
                    ToolUseBlock(id="x", name="read_file", arguments={"path": entries[0]}),
                    session=session,
                )
        assert outcome.mode == "speculative-committed"
        assert outcome.latency_saved_seconds > 0
        assert harness.controller.stats.hits == 1
        assert harness.controller.stats.misses == 1  # the other branch was waste
    finally:
        harness.close()


async def test_settle_turn_stops_a_predictor_that_is_still_thinking() -> None:
    """A slow drafter must not launch work for a turn the model has already answered."""

    class SlowPredictor:
        name = "slow"

        async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]:
            del context, top_k
            await asyncio.Event().wait()  # never finishes
            return []

    harness = Harness()
    harness.executor._predictor = SlowPredictor()
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(PredictionContext(history=("list_dir(path)",)))
                await asyncio.sleep(0)
                assert session.settle_turn() is True
                assert session.stats.turns_cancelled == 1
                assert session.settle_turn() is False
    finally:
        harness.close()


async def test_starting_a_new_turn_settles_the_previous_one() -> None:
    harness = Harness(prediction("read_file", {"path": FIRST_ENTRY}))
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await asyncio.sleep(0)
    finally:
        harness.close()


async def test_closing_the_session_bills_every_unclaimed_branch_exactly_once() -> None:
    harness = Harness(prediction("read_file", {"path": FIRST_ENTRY}))
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)
                assert len(harness.executor.cache) == 1
        assert len(harness.executor.cache) == 0
        assert harness.controller.stats.attempts == 1
        assert harness.controller.stats.misses == 1
        assert harness.controller.stats.wasted_usd > 0
    finally:
        harness.close()


async def test_refusals_and_launches_are_both_visible_in_metrics(tmp_path) -> None:  # type: ignore[no-untyped-def]
    audit = AuditLog(tmp_path / "audit.jsonl")
    harness = Harness(
        prediction("read_file", {"path": FIRST_ENTRY}),
        prediction("read_file", {"path": "src/unseen/x.py"}),
        policy=SpeculationPolicy(top_k=2, candidate_multiplier=2),
        audit=audit,
    )
    try:
        await run_turn(harness)
        rendered = {
            sample.name
            for family in harness.metrics.registry.collect()
            for sample in family.samples
        }
        assert any(name.startswith("app_speculation_attempts") for name in rendered)
        assert any(name.startswith("app_speculation_refusals") for name in rendered)

        actions = audit.path.read_text(encoding="utf-8").splitlines()
        assert any('"speculation.launched"' in line for line in actions)
    finally:
        harness.close()


# -- a reusable PURE result is claimed more than once ---------------------------------------------


async def test_a_pure_speculation_can_be_claimed_twice_without_exploding() -> None:
    """A PURE result stays in the cache after it is claimed, so the second claim must work.

    ``SpeculativeCache.take`` deliberately keeps a reusable (``PURE``) entry, so the *same*
    ``SpeculativeResult`` object is handed out again. If that object still carries the branch
    overlay from its speculation, the second claim commits an already-committed branch and the
    run dies with ``OverlayError``. A ``PURE`` handler touches no store state, so there is
    nothing to commit in the first place.
    """
    text = "Auth module rotates keys. It also logs failures."
    harness = Harness(prediction("outline_text", {"text": text}), seen=(text,))
    call = ToolUseBlock(id="t1", name="outline_text", arguments={"text": text})
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("read_file(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)

                first = await harness.executor.execute(call, session=session)
                assert first.mode == "speculative-committed"

                second = await harness.executor.execute(call, session=session)
                assert second.mode == "speculative-committed"
                assert second.output == first.output
    finally:
        harness.close()


async def test_a_pure_speculation_carries_no_branch_into_the_cache() -> None:
    """``spec/cache.py`` states that PURE results carry no branch; hold the code to it."""
    text = "Keys rotate nightly. Failures are logged."
    harness = Harness(prediction("outline_text", {"text": text}), seen=(text,))
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("read_file(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)
                stored = [harness.executor.cache.peek(key) for key in harness.executor.cache.keys]
                assert stored
                assert all(entry is not None for entry in stored)
                assert all(entry.overlay is None for entry in stored if entry is not None)
    finally:
        harness.close()


async def test_a_hit_is_credited_to_the_tool_that_produced_it() -> None:
    """A second claim has no live ``LaunchedSpeculation`` left; the credit must not go nowhere."""
    text = "Rotate the signing key. Alert on repeated failures."
    harness = Harness(prediction("outline_text", {"text": text}), seen=(text,))
    call = ToolUseBlock(id="t1", name="outline_text", arguments={"text": text})
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("read_file(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)
                await harness.executor.execute(call, session=session)
                await harness.executor.execute(call, session=session)
        assert harness.controller.tool_stats("outline_text").hits == 2
        assert harness.controller.tool_stats("unknown").hits == 0
    finally:
        harness.close()


# -- spans: which work the model asked for, and which it did not ---------------------------------


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = build_tracer_provider("spec-test", extra_exporter=exporter)
    return provider.get_tracer("test"), exporter


async def test_speculative_and_authoritative_work_are_separate_spans() -> None:
    """The spec asks for spans that mark speculative versus authoritative work; this is them."""
    tracer, exporter = _recording_tracer()
    harness = Harness(prediction("read_file", {"path": FIRST_ENTRY}), tracer=tracer)
    call = ToolUseBlock(id="t1", name="read_file", arguments={"path": FIRST_ENTRY})
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                await harness.clock.sleep(5.0)
                outcome = await harness.executor.execute(call, session=session)
        assert outcome.mode == "speculative-committed"

        spans = {span.name: dict(span.attributes or {}) for span in exporter.get_finished_spans()}
        assert set(spans) == {"tool.speculate", "tool.execute"}

        guess = spans["tool.speculate"]
        assert guess["speculative"] is True
        assert guess["tool.name"] == "read_file"
        assert guess["tool.effect"] == "read"
        assert guess["speculation.outcome"] == "ok"
        assert str(guess["speculation.branch"]).startswith("spec:read_file:")

        real = spans["tool.execute"]
        assert real["speculative"] is False
        assert real["speculation.mode"] == "speculative-committed"
        assert real["call.hash"] == guess["call.hash"]
        assert float(str(real["latency_saved_seconds"])) > 0
    finally:
        harness.close()


async def test_a_call_no_branch_matched_is_marked_direct_in_its_span() -> None:
    tracer, exporter = _recording_tracer()
    harness = Harness(tracer=tracer)
    call = ToolUseBlock(id="t1", name="read_file", arguments={"path": FIRST_ENTRY})
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                await harness.executor.execute(call, session=session)
        spans = [span for span in exporter.get_finished_spans() if span.name == "tool.execute"]
        assert len(spans) == 1
        attributes = dict(spans[0].attributes or {})
        assert attributes["speculative"] is False
        assert attributes["speculation.mode"] == "direct"
        assert attributes["latency_saved_seconds"] == 0.0
    finally:
        harness.close()


async def test_a_cancelled_branch_still_closes_its_span() -> None:
    """A mispredicted branch is the common case, so its span must not leak."""
    tracer, exporter = _recording_tracer()
    harness = Harness(prediction("read_file", {"path": FIRST_ENTRY}), tracer=tracer)
    other = ToolUseBlock(id="t1", name="list_dir", arguments={"path": "src/auth"})
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(
                    PredictionContext(history=("list_dir(path)",), last_output=LISTING)
                )
                for _ in range(5):
                    await asyncio.sleep(0)
                assert session.in_flight, "the branch must be running before it is cancelled"
                await harness.executor.execute(other, session=session)
        guesses = [span for span in exporter.get_finished_spans() if span.name == "tool.speculate"]
        assert len(guesses) == 1
        assert dict(guesses[0].attributes or {})["speculation.outcome"] == "cancelled"
    finally:
        harness.close()
