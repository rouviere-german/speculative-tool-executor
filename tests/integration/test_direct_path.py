"""The authoritative path when there is nothing to claim: retries, exhaustion, and a bad guess.

Speculation gets the attention, but every call speculation did not serve lands here -- including
every read-modify-write whose read set moved while it ran. The benchmark's contention injector
fires *before* a call, so it only ever exercises the speculative arm; these tests move the key
underneath a direct call instead, with a second writer scheduled on the same virtual clock.

What they pin is THREAT_MODEL row 6 ("a read-modify-write loses another party's update") on the
path that has no speculation to blame: the retry must recompute the increment from the value the
other party left, and when the retry budget runs out the call must fail loudly rather than write
a number it derived from a state that no longer exists.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from speculative_tool_executor.agentic.approval import ApprovalGate, ApprovalStore
from speculative_tool_executor.agentic.audit import AuditLog
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.spec.cache import SpeculativeCache, SpeculativeResult
from speculative_tool_executor.spec.clock import VirtualClock
from speculative_tool_executor.spec.controller import SpeculationController, SpeculationPolicy
from speculative_tool_executor.spec.executor import (
    ExecutionMode,
    SpeculativeToolExecutor,
    ToolOutcome,
)
from speculative_tool_executor.spec.latency import LatencyModel
from speculative_tool_executor.spec.metrics import SpeculationMetrics, create_speculation_metrics
from speculative_tool_executor.spec.overlay import WriteOverlay
from speculative_tool_executor.spec.predictor import Prediction, PredictionContext
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.spec.taint import TaintTracker
from speculative_tool_executor.workload.corpus import build_corpus
from speculative_tool_executor.workload.tools import (
    DEFAULT_TOOL_PROFILE,
    MODEL_TURN_PROFILE,
    TOOL_PROFILES,
    build_registry,
)

pytestmark = pytest.mark.integration

CORPUS = build_corpus()
KEY = "index/auth"
MISSING = "nope/missing.py"


class FixedPredictor:
    """Proposes exactly what the test asks for; nothing here is trying to be a good predictor."""

    name = "fixed"

    def __init__(self, *predictions: Prediction) -> None:
        self._predictions = list(predictions)

    async def predict(self, context: PredictionContext, *, top_k: int) -> list[Prediction]:
        del context
        return self._predictions[:top_k]


class Harness:
    """An executor with a real store and a virtual clock, and nothing predicted by default."""

    def __init__(
        self,
        *predictions: Prediction,
        max_commit_retries: int = 3,
        seen: tuple[str, ...] = (),
        audit: AuditLog | None = None,
    ) -> None:
        self.metrics: SpeculationMetrics = create_speculation_metrics()
        self.store = VersionedStore(":memory:", synchronous="OFF")
        self.registry, self.sink = build_registry(CORPUS)
        self.clock = VirtualClock()
        self.controller = SpeculationController(SpeculationPolicy(top_k=2))
        self.taint = TaintTracker()
        self.taint.observe_all(list(seen))
        self.executor = SpeculativeToolExecutor(
            registry=self.registry,
            store=self.store,
            clock=self.clock,
            latency=LatencyModel(
                tools=TOOL_PROFILES,
                default_tool=DEFAULT_TOOL_PROFILE,
                model_turn=MODEL_TURN_PROFILE,
                seed=11,
            ),
            gate=ApprovalGate(ApprovalStore(ttl_seconds=1000.0), principal="agent"),
            controller=self.controller,
            predictor=FixedPredictor(*predictions),
            cache=SpeculativeCache(max_entries=16),
            taint=self.taint,
            audit=audit,
            metrics=self.metrics,
            max_commit_retries=max_commit_retries,
        )

    def close(self) -> None:
        self.store.close()


async def _bump(harness: Harness, *, armed: asyncio.Event) -> ToolOutcome:
    """One authoritative ``bump_index``, with no speculation running for it."""
    with harness.clock.participant():
        armed.set()
        async with harness.executor.session() as session:
            return await harness.executor.execute(
                ToolUseBlock(id="call-1", name="bump_index", arguments={"key": KEY}),
                session=session,
            )


async def _write_while_it_runs(
    harness: Harness, value: str, *, armed: asyncio.Event, after: float = 0.001
) -> None:
    """Another party writes ``KEY`` inside the direct call's latency window.

    The interleaving is pinned, not raced. Both tasks hold a clock registration, so virtual time
    cannot move until both are asleep; ``armed`` makes sure the call is registered before this
    writer starts its own much shorter sleep. The write therefore always lands after the handler
    read the key and before its branch tries to commit -- the one ordering optimistic concurrency
    control exists to catch.
    """
    with harness.clock.participant():
        await armed.wait()
        await harness.clock.sleep(after)
        harness.store.put(KEY, value)


async def _bump_against_a_concurrent_writer(harness: Harness, value: str) -> ToolOutcome:
    armed = asyncio.Event()
    async with asyncio.TaskGroup() as group:
        group.create_task(_write_while_it_runs(harness, value, armed=armed))
        call = group.create_task(_bump(harness, armed=armed))
    return call.result()


async def test_a_direct_write_retries_on_top_of_the_other_party_s_value() -> None:
    harness = Harness()
    try:
        harness.store.put(KEY, "5")
        outcome = await _bump_against_a_concurrent_writer(harness, "100")

        assert outcome.ok is True
        assert outcome.mode == ExecutionMode.DIRECT
        assert outcome.conflicts == (KEY,), "the first attempt must be reported as conflicted"
        assert outcome.output == {"key": KEY, "count": 101}, "101, not 6: the retry re-read"
        assert outcome.committed_writes == (KEY,)

        entry = harness.store.get(KEY)
        assert entry is not None
        assert entry.value == "101"
        assert harness.controller.stats.occ_conflicts == 1
    finally:
        harness.close()


async def test_a_direct_write_that_runs_out_of_retries_fails_instead_of_losing_the_update() -> None:
    harness = Harness(max_commit_retries=1)
    try:
        harness.store.put(KEY, "5")
        outcome = await _bump_against_a_concurrent_writer(harness, "100")

        assert outcome.ok is False
        assert outcome.error == "CommitConflict"
        assert outcome.branch_id == "direct:exhausted"
        assert outcome.conflicts == (KEY,)
        assert outcome.committed_writes == ()

        entry = harness.store.get(KEY)
        assert entry is not None
        assert entry.value == "100", "the other party's write survives"
    finally:
        harness.close()


def test_the_retry_budget_must_allow_at_least_one_attempt() -> None:
    with VersionedStore(":memory:", synchronous="OFF") as store:
        registry, _ = build_registry(CORPUS)
        with pytest.raises(ValueError, match="at least 1"):
            SpeculativeToolExecutor(
                registry=registry,
                store=store,
                clock=VirtualClock(),
                latency=LatencyModel(
                    tools=TOOL_PROFILES,
                    default_tool=DEFAULT_TOOL_PROFILE,
                    model_turn=MODEL_TURN_PROFILE,
                    seed=1,
                ),
                gate=ApprovalGate(ApprovalStore(ttl_seconds=10.0), principal="agent"),
                max_commit_retries=0,
            )


async def test_a_speculation_that_failed_is_discarded_and_the_call_runs_for_real(
    tmp_path: Path,
) -> None:
    """A guess whose handler raised must never be served as the answer.

    The branch finished, so it sits in the cache like any other -- but with ``outcome == "error"``
    it is thrown away and billed as waste, and the authoritative call executes for itself. The
    error the caller sees therefore comes from the real call, not from the speculation.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    harness = Harness(
        Prediction(tool="read_file", arguments={"path": MISSING}, probability=0.9, source="fixed"),
        seen=(MISSING,),
        audit=audit,
    )
    try:
        with harness.clock.participant():
            async with harness.executor.session() as session:
                session.start_turn(PredictionContext(history=("list_dir(path)",)))
                await harness.clock.sleep(5.0)
                cached = harness.executor.cache.peek(harness.executor.cache.keys[0])
                assert cached is not None
                assert cached.outcome == "error"

                outcome = await harness.executor.execute(
                    ToolUseBlock(id="call-1", name="read_file", arguments={"path": MISSING}),
                    session=session,
                )

        assert outcome.mode == ExecutionMode.SPECULATIVE_ABORTED
        assert outcome.ok is False
        assert outcome.error == "WorkspaceError"
        assert harness.controller.stats.misses == 1
        assert harness.controller.stats.wasted_usd > 0
        assert _counter(harness, "app_speculation_occ_conflicts_total") == 0.0, (
            "a handler that raised is waste, not evidence of contention"
        )
        assert _counter(harness, "app_speculation_wasted_usd_total") > 0

        lines = audit.path.read_text(encoding="utf-8").splitlines()
        assert any('"reason":"speculative_error"' in line for line in lines)
        assert any('"action":"call.speculative_aborted"' in line for line in lines)
    finally:
        harness.close()


def _counter(harness: Harness, name: str) -> float:
    value = harness.metrics.registry.get_sample_value(name)
    return 0.0 if value is None else value


async def test_a_claimed_branch_that_was_already_settled_is_never_served(tmp_path: Path) -> None:
    """The fail-closed backstop behind the reusable-result fix.

    A branch that is no longer ``OPEN`` has been settled by something else. Committing it again
    raises, and treating it as an empty commit would report a result whose branch never reached
    the store. Neither is acceptable, so the entry is discarded and the call runs for real. The
    speculator never produces this state; the test builds it by hand to pin the behaviour.
    """
    audit = AuditLog(tmp_path / "audit.jsonl")
    harness = Harness(audit=audit)
    call = ToolUseBlock(id="call-1", name="bump_index", arguments={"key": KEY})
    try:
        harness.store.put(KEY, "5")
        key = harness.registry.validate_call(call).call_hash
        settled = WriteOverlay(store=harness.store, branch_id="spec:bump_index:settled")
        settled.write(KEY, "999")
        settled.abort()
        harness.executor.cache.put(
            SpeculativeResult(
                key=key,
                tool="bump_index",
                effect=harness.registry.get("bump_index").effect,
                branch_id=settled.branch_id,
                outcome="ok",
                output={"key": KEY, "count": 999},
                overlay=settled,
                cost_usd=0.001,
            )
        )

        with harness.clock.participant():
            async with harness.executor.session() as session:
                outcome = await harness.executor.execute(call, session=session)

        assert outcome.mode == ExecutionMode.SPECULATIVE_ABORTED
        assert outcome.ok is True
        assert outcome.output == {"key": KEY, "count": 6}, "computed for real, not the stale 999"
        assert outcome.committed_writes == (KEY,)
        entry = harness.store.get(KEY)
        assert entry is not None
        assert entry.value == "6"
        assert _counter(harness, "app_speculation_occ_conflicts_total") == 0.0

        lines = audit.path.read_text(encoding="utf-8").splitlines()
        assert any('"reason":"branch_settled"' in line for line in lines)
    finally:
        harness.close()
