"""The authoritative path: matching the model's real call against speculative work.

This is where speculation becomes visible or disappears. Every real tool call arrives here and
goes through the same four steps, in this order:

1. **Validate and canonicalize.** The allowlist, the strict schema and the canonicalizer decide
   what the call *is*. Its canonical hash is the only identity anything downstream uses.
2. **Authorize.** ``EXTERNAL`` tools need a human approval bound to that hash. This happens
   before any speculative result is looked at, so an approval can never be satisfied by work that
   was done before anyone approved it.
3. **Match.** If a branch with the same hash is running or finished, claim it and cancel every
   other one -- those are, by definition, mispredictions. A claimed branch is committed only if
   optimistic concurrency control still holds; otherwise it is discarded and the call runs for
   real. Either way the store ends in the state a non-speculative run would have reached.
4. **Record.** One audit entry per call, marked ``direct``, ``speculative-committed`` or
   ``speculative-aborted``, hash-chained with everything else the run did.

The latency saved by a hit is ``the tool latency the branch absorbed`` minus ``how long the
authoritative call had to wait for it to finish``. A branch that is still in flight when the
model asks for it therefore counts only the head start it actually had, which keeps the headline
number honest when speculation starts late.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from opentelemetry.trace import Tracer
from pydantic import JsonValue

from speculative_tool_executor.agentic.approval import ApprovalGate
from speculative_tool_executor.agentic.audit import AuditLog
from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.agentic.tools import ToolRegistry, ValidatedCall
from speculative_tool_executor.core.canonical import canonical_json, sha256_hex
from speculative_tool_executor.core.logging import get_logger
from speculative_tool_executor.core.tracing import get_tracer
from speculative_tool_executor.spec.cache import SpeculativeCache, SpeculativeResult
from speculative_tool_executor.spec.clock import Clock
from speculative_tool_executor.spec.controller import SpeculationController
from speculative_tool_executor.spec.latency import LatencyModel
from speculative_tool_executor.spec.metrics import SpeculationMetrics
from speculative_tool_executor.spec.overlay import BranchState
from speculative_tool_executor.spec.predictor import NullPredictor, Predictor
from speculative_tool_executor.spec.runner import ToolRunner
from speculative_tool_executor.spec.speculator import (
    LaunchedSpeculation,
    SpeculationDeps,
    SpeculationSession,
)
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.spec.taint import TaintTracker

__all__ = ["ExecutionMode", "SpeculativeToolExecutor", "ToolOutcome", "unusable_reason"]

_log = get_logger("speculative_tool_executor.executor")
_TRACER_NAME = "speculative_tool_executor.spec"


class ExecutionMode:
    """How an authoritative call was served. These strings appear in audit records and metrics."""

    DIRECT = "direct"
    SPECULATIVE_COMMITTED = "speculative-committed"
    SPECULATIVE_ABORTED = "speculative-aborted"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """The result of one authoritative call, plus how it was served."""

    tool: str
    call_hash: str
    effect: Effect
    mode: str
    ok: bool
    output: JsonValue
    error: str | None
    branch_id: str
    waited_seconds: float
    latency_saved_seconds: float
    committed_writes: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def output_sha256(self) -> str:
        return sha256_hex(canonical_json(self.output))


class SpeculativeToolExecutor:
    """Executes tool calls, using speculative work whenever it provably matches."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        store: VersionedStore,
        clock: Clock,
        latency: LatencyModel,
        gate: ApprovalGate,
        controller: SpeculationController | None = None,
        predictor: Predictor | None = None,
        cache: SpeculativeCache | None = None,
        taint: TaintTracker | None = None,
        audit: AuditLog | None = None,
        metrics: SpeculationMetrics | None = None,
        principal: str = "agent",
        max_commit_retries: int = 3,
        tracer: Tracer | None = None,
    ) -> None:
        if max_commit_retries < 1:
            raise ValueError("max_commit_retries must be at least 1")
        self._registry = registry
        self._store = store
        self._clock = clock
        self._gate = gate
        self._controller = controller or SpeculationController()
        self._predictor = predictor or NullPredictor()
        self._cache = cache or SpeculativeCache()
        self._taint = taint or TaintTracker()
        self._audit = audit
        self._metrics = metrics
        self._principal = principal
        self._max_commit_retries = max_commit_retries
        self._tracer = tracer or get_tracer(_TRACER_NAME)
        self._runner = ToolRunner(store=store, clock=clock, latency=latency)
        self._direct_branches = itertools.count()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    @property
    def controller(self) -> SpeculationController:
        return self._controller

    @property
    def cache(self) -> SpeculativeCache:
        return self._cache

    @property
    def taint(self) -> TaintTracker:
        return self._taint

    @property
    def store(self) -> VersionedStore:
        return self._store

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def runner(self) -> ToolRunner:
        return self._runner

    @asynccontextmanager
    async def session(self) -> AsyncIterator[SpeculationSession]:
        """One agent run's speculation scope. Leaving it cancels and settles every branch."""
        deps = SpeculationDeps(
            registry=self._registry,
            runner=self._runner,
            predictor=self._predictor,
            controller=self._controller,
            cache=self._cache,
            taint=self._taint,
            metrics=self._metrics,
            audit=self._audit,
            principal=self._principal,
            tracer=self._tracer,
        )
        async with asyncio.TaskGroup() as group:
            session = SpeculationSession(deps, group=group, clock=self._clock)
            try:
                yield session
            finally:
                session.close()
                if self._metrics is not None:
                    self._metrics.enabled.set(1.0 if self._controller.enabled else 0.0)

    async def execute(self, call: ToolUseBlock, *, session: SpeculationSession) -> ToolOutcome:
        """Execute one call the model actually made, using a matching branch when one exists.

        The whole call is one span marked ``speculative=False``: this is the authoritative path,
        whatever served it. ``speculation.mode`` then says which of the three ways it was served,
        so a trace shows at a glance which calls speculation actually removed the wait from.
        """
        with self._tracer.start_as_current_span("tool.execute") as span:
            span.set_attribute("speculative", False)
            span.set_attribute("tool.name", call.name)
            outcome = await self._execute(call, session=session)
            span.set_attribute("tool.effect", outcome.effect.value)
            span.set_attribute("speculation.mode", outcome.mode)
            span.set_attribute("call.hash", outcome.call_hash)
            span.set_attribute("latency_saved_seconds", outcome.latency_saved_seconds)
            return outcome

    async def _execute(self, call: ToolUseBlock, *, session: SpeculationSession) -> ToolOutcome:
        validated = self._registry.validate_call(call)
        # The model has answered this turn, so predicting for it can only create waste.
        session.settle_turn()
        key = validated.call_hash
        authorized = self._gate.authorize(validated)
        arrival = self._clock.now
        self._controller.record_baseline_spend(validated.tool.expected_cost_usd)
        self._cache.purge_expired(now=arrival)

        launched = session.claim(key)
        session.cancel_others(keep=key if launched is not None else None)
        if launched is not None:
            await session.wait_for(launched)

        mode = ExecutionMode.DIRECT
        conflicts: tuple[str, ...] = ()
        result = self._cache.take(key, now=self._clock.now)
        unusable = unusable_reason(result)
        if result is not None and unusable is None:
            waited = self._clock.now - arrival
            commit = (
                result.overlay.commit()
                if result.overlay is not None
                else self._store.commit({}, {})
            )
            if commit.committed:
                saved = max(0.0, result.latency_seconds - waited)
                session.record_hit(key, tool=result.tool, saved_seconds=saved)
                outcome = ToolOutcome(
                    tool=result.tool,
                    call_hash=key,
                    effect=result.effect,
                    mode=ExecutionMode.SPECULATIVE_COMMITTED,
                    ok=True,
                    output=result.output,
                    error=None,
                    branch_id=result.branch_id,
                    waited_seconds=waited,
                    latency_saved_seconds=saved,
                    committed_writes=commit.written,
                )
                self._record(outcome, extra={"p_hit": round(launched_probability(launched), 6)})
                return outcome
            conflicts = commit.conflicts
            self._controller.record_conflict(stale_read=result.effect is Effect.READ)
            session.record_discard(
                key, tool=result.tool, cost_usd=result.cost_usd, reason="occ_conflict"
            )
            mode = ExecutionMode.SPECULATIVE_ABORTED
        elif result is not None and unusable is not None:
            session.record_discard(key, tool=result.tool, cost_usd=result.cost_usd, reason=unusable)
            mode = ExecutionMode.SPECULATIVE_ABORTED

        outcome = await self._execute_directly(
            validated, authorized_hash=authorized, arrival=arrival, mode=mode, conflicts=conflicts
        )
        self._record(outcome)
        return outcome

    async def _execute_directly(
        self,
        validated: ValidatedCall,
        *,
        authorized_hash: str,
        arrival: float,
        mode: str,
        conflicts: tuple[str, ...],
    ) -> ToolOutcome:
        """Run the call for real, retrying a write whose read set moved under it."""
        tool = validated.tool
        for attempt in range(1, self._max_commit_retries + 1):
            branch_id = f"direct:{next(self._direct_branches):06d}"
            branch = await self._runner.run(
                validated,
                branch_id=branch_id,
                speculative=False,
                authorized_hash=authorized_hash,
            )
            if not branch.overlay.dirty:
                branch.overlay.abort()
                return ToolOutcome(
                    tool=tool.name,
                    call_hash=validated.call_hash,
                    effect=tool.effect,
                    mode=mode,
                    ok=branch.ok,
                    output=branch.output,
                    error=branch.error,
                    branch_id=branch_id,
                    waited_seconds=self._clock.now - arrival,
                    latency_saved_seconds=0.0,
                    conflicts=conflicts,
                )
            commit = branch.overlay.commit()
            if commit.committed:
                return ToolOutcome(
                    tool=tool.name,
                    call_hash=validated.call_hash,
                    effect=tool.effect,
                    mode=mode,
                    ok=branch.ok,
                    output=branch.output,
                    error=branch.error,
                    branch_id=branch_id,
                    waited_seconds=self._clock.now - arrival,
                    latency_saved_seconds=0.0,
                    committed_writes=commit.written,
                    conflicts=conflicts,
                )
            conflicts = commit.conflicts
            self._controller.record_conflict()
            _log.warning(
                "direct_commit_conflict",
                tool=tool.name,
                attempt=attempt,
                conflicts=list(conflicts),
            )
        return ToolOutcome(
            tool=tool.name,
            call_hash=validated.call_hash,
            effect=tool.effect,
            mode=mode,
            ok=False,
            output=None,
            error="CommitConflict",
            branch_id="direct:exhausted",
            waited_seconds=self._clock.now - arrival,
            latency_saved_seconds=0.0,
            conflicts=conflicts,
        )

    def _record(
        self, outcome: ToolOutcome, *, extra: Mapping[str, JsonValue] | None = None
    ) -> None:
        if self._metrics is not None:
            self._metrics.observe_call(outcome.tool, outcome.mode, outcome.waited_seconds)
        if self._audit is None:
            return
        action = {
            ExecutionMode.DIRECT: "call.direct",
            ExecutionMode.SPECULATIVE_COMMITTED: "call.speculative_committed",
            ExecutionMode.SPECULATIVE_ABORTED: "call.speculative_aborted",
        }[outcome.mode]
        data: dict[str, JsonValue] = {
            "tool": outcome.tool,
            "effect": outcome.effect.value,
            "call_hash": outcome.call_hash,
            "branch": outcome.branch_id,
            "ok": outcome.ok,
            "error": outcome.error,
            "output_sha256": outcome.output_sha256,
            "waited_seconds": round(outcome.waited_seconds, 6),
            "latency_saved_seconds": round(outcome.latency_saved_seconds, 6),
            "committed_writes": list(outcome.committed_writes),
            "conflicts": list(outcome.conflicts),
            "at": round(self._clock.now, 6),
        }
        if extra:
            data.update(extra)
        self._audit.append(self._principal, action, data)


def unusable_reason(result: SpeculativeResult | None) -> str | None:
    """Why a claimed result must not be served, or None when it may be.

    Two things disqualify it. A handler that raised produced no answer worth serving -- the
    authoritative call runs for itself and reports its own error. And a branch that is no longer
    ``OPEN`` has already been settled by something else: committing it again would raise, and
    treating it as an empty commit would report writes that never reached the store. Neither
    state is produced by the speculator (``PURE`` results carry no branch, and every other entry
    leaves the cache when its branch is settled), so this is the fail-closed backstop rather than
    a path the benchmark takes.
    """
    if result is None:
        return None
    if result.outcome != "ok":
        return "speculative_error"
    if result.overlay is not None and result.overlay.state is not BranchState.OPEN:
        return "branch_settled"
    return None


def launched_probability(launched: LaunchedSpeculation | None) -> float:
    """The prior the predictor gave a claimed branch, or 0 when the branch is unknown."""
    return launched.prediction.probability if launched is not None else 0.0
