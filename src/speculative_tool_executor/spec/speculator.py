"""The speculator: launching predicted calls, and cancelling the ones reality did not confirm.

A session owns one ``asyncio.TaskGroup`` for the whole agent run. Predicted calls become tasks in
that group, so they outlive the turn that started them -- which is the point, because a
speculation is only useful if it is still running (or already finished) when the model finally
says what it wants. Structured concurrency then guarantees the part that is easy to get wrong:
when the run ends, for any reason, every outstanding branch is cancelled and awaited before the
group exits, so no speculation escapes the run that started it.

A prediction goes through **the same** allowlist, schema validation and argument canonicalization
as a real call, plus two checks a real call does not need: the taint rule
(:mod:`speculative_tool_executor.spec.taint`) and the controller's cost/benefit test. Anything a
prediction cannot pass is refused with a named reason, and the reasons are counted -- a
speculator that silently drops most of its candidates looks identical to one that is working
until you can see why it refused.

Waste is billed at the moment a branch's fate is decided, and each branch is billed exactly once:
an in-flight branch the model did not ask for is cancelled and charged immediately; a finished
branch stays in the cache, costing nothing further, until it is claimed, expires, is evicted, or
the run ends without it.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import JsonValue

from speculative_tool_executor.agentic.audit import AuditLog
from speculative_tool_executor.agentic.effects import SPECULATABLE_EFFECTS
from speculative_tool_executor.agentic.errors import ToolArgumentError, ToolNotAllowedError
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.agentic.tools import ToolRegistry, ValidatedCall
from speculative_tool_executor.core.logging import get_logger
from speculative_tool_executor.spec.cache import (
    SpeculativeCache,
    SpeculativeResult,
    freshness_deadline,
)
from speculative_tool_executor.spec.clock import Clock
from speculative_tool_executor.spec.controller import DecisionReason, SpeculationController
from speculative_tool_executor.spec.metrics import SpeculationMetrics
from speculative_tool_executor.spec.predictor import Prediction, PredictionContext, Predictor
from speculative_tool_executor.spec.runner import ToolRunner
from speculative_tool_executor.spec.taint import TaintTracker, schema_constants

__all__ = ["LaunchedSpeculation", "SessionStats", "SpeculationDeps", "SpeculationSession"]

_log = get_logger("speculative_tool_executor.speculator")


@dataclass(slots=True)
class LaunchedSpeculation:
    """One speculative branch and everything needed to settle it later."""

    key: str
    tool: str
    branch_id: str
    prediction: Prediction
    done: asyncio.Event
    started_at: float
    cost_usd: float
    task: asyncio.Task[None] | None = None
    claimed: bool = False

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


@dataclass(frozen=True, slots=True)
class SpeculationDeps:
    """Everything a session needs, bundled so the executor can hand it over in one piece."""

    registry: ToolRegistry
    runner: ToolRunner
    predictor: Predictor
    controller: SpeculationController
    cache: SpeculativeCache
    taint: TaintTracker
    metrics: SpeculationMetrics | None = None
    audit: AuditLog | None = None
    principal: str = "agent"


@dataclass(slots=True)
class SessionStats:
    predictions: int = 0
    launched: int = 0
    cancelled: int = 0
    dropped: int = 0
    turns_cancelled: int = 0
    refused: dict[str, int] = field(default_factory=dict)


class SpeculationSession:
    """Speculation for one agent run, bound to one task group."""

    def __init__(
        self,
        deps: SpeculationDeps,
        *,
        group: asyncio.TaskGroup,
        clock: Clock,
    ) -> None:
        self._deps = deps
        self._group = group
        self._clock = clock
        self._in_flight: dict[str, LaunchedSpeculation] = {}
        self._completed: dict[str, LaunchedSpeculation] = {}
        self._branches = itertools.count()
        self._constants: dict[str, frozenset[str]] = {}
        self._turn_task: asyncio.Task[None] | None = None
        self.stats = SessionStats()

    @property
    def in_flight(self) -> tuple[str, ...]:
        return tuple(self._in_flight)

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)

    @property
    def controller(self) -> SpeculationController:
        return self._deps.controller

    # -- launching ---------------------------------------------------------------------------

    def start_turn(self, context: PredictionContext) -> asyncio.Task[None]:
        """Spawn prediction for the upcoming turn. Returns immediately; the task does the work."""
        self.settle_turn()
        task = self._group.create_task(self._predict_and_launch(context))
        self._clock.track(task)
        self._turn_task = task
        return task

    def settle_turn(self) -> bool:
        """Stop predicting for a turn the model has already answered.

        Once the authoritative call has arrived, a prediction for that same turn can only launch
        work nothing will ever claim. With the instant Markov predictor this finds the task
        already finished and does nothing; a predictor that takes real time (the optional LLM
        drafter) is cancelled instead of being left to spend money on a question already answered.
        """
        task = self._turn_task
        self._turn_task = None
        if task is None or task.done():
            return False
        task.cancel()
        self.stats.turns_cancelled += 1
        return True

    async def _predict_and_launch(self, context: PredictionContext) -> None:
        controller = self._deps.controller
        with self._clock.participant():
            if not controller.enabled:
                reason = (
                    DecisionReason.WASTE_RATIO_TRIPPED
                    if controller.stats.tripped
                    else DecisionReason.GLOBALLY_DISABLED
                )
                self._refuse(reason, tool="-")
                return
            policy = controller.policy
            predictions = await self._deps.predictor.predict(
                context, top_k=policy.candidates_to_rank
            )
            self.stats.predictions += len(predictions)
            launched = 0
            for prediction in predictions:
                if launched >= policy.top_k:
                    break
                if self._launch(prediction) is not None:
                    launched += 1

    def _launch(  # noqa: PLR0911 - each return is a distinct, named refusal
        self, prediction: Prediction
    ) -> LaunchedSpeculation | None:
        controller = self._deps.controller
        validated = self._validate(prediction)
        if validated is None:
            return None
        # The effect boundary is checked here, before anything else that could happen to refuse
        # the same call for an unrelated reason. An EXTERNAL tool must be turned away *because*
        # it is external, so the refusal counter says so and a test can assert it; relying on the
        # taint rule or the cost model to have caught it first would be an accident, not a
        # guarantee. ToolRunner.run re-checks the same invariant at launch.
        if validated.tool.effect not in SPECULATABLE_EFFECTS or not validated.tool.speculatable:
            reason = (
                DecisionReason.NEEDS_APPROVAL
                if validated.tool.effect in SPECULATABLE_EFFECTS
                else DecisionReason.EFFECT_NOT_SPECULATABLE
            )
            self._refuse(reason, validated.tool.name)
            return None
        report = self._deps.taint.check(
            validated.canonical_args, constants=self._constants_for(validated.tool.name)
        )
        if not report.ok:
            self._refuse(DecisionReason.TAINTED_ARGUMENTS, validated.tool.name)
            return None
        key = validated.call_hash
        known = key in self._in_flight or key in self._completed
        if known or self._deps.cache.peek(key) is not None:
            self._refuse(DecisionReason.ALREADY_KNOWN, validated.tool.name)
            return None
        if len(self._in_flight) >= controller.policy.max_in_flight:
            self._refuse(DecisionReason.MAX_IN_FLIGHT, validated.tool.name)
            return None
        decision = controller.decide(validated.tool, prediction.probability)
        if not decision.allowed:
            self._refuse(decision.reason, validated.tool.name, counted_by_controller=True)
            return None

        launched = LaunchedSpeculation(
            key=key,
            tool=validated.tool.name,
            branch_id=f"spec:{validated.tool.name}:{key[:12]}",
            prediction=prediction,
            done=asyncio.Event(),
            started_at=self._clock.now,
            cost_usd=validated.tool.expected_cost_usd,
        )
        self._in_flight[key] = launched
        task = self._group.create_task(self._run(launched, validated))
        self._clock.track(task)
        launched.task = task
        self.stats.launched += 1
        controller.record_launch(validated.tool.name, launched.cost_usd)
        self._deps.cache.stats.record_attempt(validated.tool.name)
        metrics = self._deps.metrics
        if metrics is not None:
            metrics.attempts_total.labels(tool=validated.tool.name).inc()
            metrics.speculative_spend_usd_total.inc(launched.cost_usd)
        self._audit(
            "speculation.launched",
            {
                "tool": validated.tool.name,
                "effect": validated.tool.effect.value,
                "call_hash": key,
                "branch": launched.branch_id,
                "probability": round(prediction.probability, 6),
                "p_hit": round(decision.p_hit, 6),
                "source": prediction.source,
                "at": round(self._clock.now, 6),
            },
        )
        return launched

    def _validate(self, prediction: Prediction) -> ValidatedCall | None:
        """Run a prediction through exactly the checks a model-proposed call goes through."""
        registry = self._deps.registry
        try:
            registry.get(prediction.tool)
        except ToolNotAllowedError:
            self._refuse(DecisionReason.UNKNOWN_TOOL, prediction.tool)
            return None
        block = ToolUseBlock(
            id=f"spec_{next(self._branches):06d}",
            name=prediction.tool,
            arguments=dict(prediction.arguments),
        )
        try:
            return registry.validate_call(block)
        except (ToolNotAllowedError, ToolArgumentError):
            self._refuse(DecisionReason.INVALID_ARGUMENTS, prediction.tool)
            return None

    async def _run(self, launched: LaunchedSpeculation, validated: ValidatedCall) -> None:
        """The speculative branch itself. Cancellation is expected here, not exceptional."""
        with self._clock.participant():
            try:
                outcome = await self._deps.runner.run(
                    validated, branch_id=launched.branch_id, speculative=True
                )
            except asyncio.CancelledError:
                launched.done.set()
                raise
            tool = validated.tool
            self._deps.cache.put(
                SpeculativeResult(
                    key=launched.key,
                    tool=tool.name,
                    effect=tool.effect,
                    branch_id=launched.branch_id,
                    outcome=outcome.outcome,
                    output=outcome.output,
                    error=outcome.error,
                    overlay=outcome.overlay,
                    created_at=outcome.finished_at,
                    expires_at=freshness_deadline(outcome.finished_at, tool.freshness_seconds),
                    latency_seconds=outcome.latency_seconds,
                    cost_usd=launched.cost_usd,
                )
            )
            if self._in_flight.pop(launched.key, None) is not None:
                self._completed[launched.key] = launched
            launched.done.set()

    # -- settling ----------------------------------------------------------------------------

    def claim(self, key: str) -> LaunchedSpeculation | None:
        """Mark the branch for ``key`` as the one reality confirmed, if there is one."""
        launched = self._in_flight.get(key) or self._completed.get(key)
        if launched is not None:
            launched.claimed = True
        return launched

    async def wait_for(self, launched: LaunchedSpeculation) -> None:
        """Wait for a claimed branch to finish, without holding the simulated clock back."""
        if launched.done.is_set():
            return
        with self._clock.parked():
            await launched.done.wait()

    def record_hit(self, key: str, *, saved_seconds: float) -> None:
        """A claimed branch committed: charge the saved latency to the tool's statistics."""
        launched = self._forget(key)
        tool = launched.tool if launched is not None else "unknown"
        controller = self._deps.controller
        controller.record_hit(tool, saved_seconds=saved_seconds)
        metrics = self._deps.metrics
        if metrics is not None:
            metrics.hits_total.labels(tool=tool).inc()
            metrics.latency_saved_seconds.inc(max(0.0, saved_seconds))
            metrics.hit_ratio.labels(tool=tool).set(controller.per_tool_hit_ratio().get(tool, 0.0))

    def record_discard(self, key: str, *, tool: str, cost_usd: float, reason: str) -> None:
        """A claimed branch failed validation (stale read or write conflict) and was thrown away."""
        self._forget(key)
        controller = self._deps.controller
        controller.record_miss(tool, cost_usd=cost_usd)
        metrics = self._deps.metrics
        if metrics is not None:
            metrics.wasted_usd_total.inc(cost_usd)
            metrics.occ_conflicts_total.inc()
            metrics.hit_ratio.labels(tool=tool).set(controller.per_tool_hit_ratio().get(tool, 0.0))
        self._audit(
            "speculation.discarded",
            {
                "tool": tool,
                "call_hash": key,
                "reason": reason,
                "wasted_usd": round(cost_usd, 9),
                "at": round(self._clock.now, 6),
            },
        )

    def cancel_others(self, *, keep: str | None = None, reason: str = "mispredicted") -> int:
        """Cancel every *running* branch except ``keep``; their cost becomes waste at once."""
        cancelled = 0
        for key, launched in list(self._in_flight.items()):
            if key == keep:
                continue
            self._in_flight.pop(key, None)
            if launched.task is not None and not launched.task.done():
                launched.task.cancel()
                cancelled += 1
            self._charge_waste(launched, cancelled=True, reason=reason)
        self.stats.cancelled += cancelled
        return cancelled

    def drop_unclaimed(self) -> int:
        """Bill finished-but-unclaimed speculations as waste and empty the cache."""
        dropped = self._completed
        self._completed = {}
        for launched in dropped.values():
            self._charge_waste(launched, cancelled=False, reason="unclaimed")
        self._deps.cache.clear()
        self.stats.dropped += len(dropped)
        return len(dropped)

    def close(self) -> None:
        """Settle everything still outstanding. Called from the session's own exit path."""
        self.settle_turn()
        self.cancel_others(keep=None, reason="run_finished")
        self.drop_unclaimed()

    # -- internals ---------------------------------------------------------------------------

    def _forget(self, key: str) -> LaunchedSpeculation | None:
        return self._completed.pop(key, None) or self._in_flight.pop(key, None)

    def _charge_waste(self, launched: LaunchedSpeculation, *, cancelled: bool, reason: str) -> None:
        self._deps.controller.record_miss(
            launched.tool, cost_usd=launched.cost_usd, cancelled=cancelled
        )
        metrics = self._deps.metrics
        if metrics is not None:
            metrics.wasted_usd_total.inc(launched.cost_usd)
            if cancelled:
                metrics.cancellations_total.inc()
            metrics.hit_ratio.labels(tool=launched.tool).set(
                self._deps.controller.per_tool_hit_ratio().get(launched.tool, 0.0)
            )
        self._deps.cache.discard(launched.key)
        self._audit(
            "speculation.aborted" if cancelled else "speculation.dropped",
            {
                "tool": launched.tool,
                "call_hash": launched.key,
                "branch": launched.branch_id,
                "reason": reason,
                "wasted_usd": round(launched.cost_usd, 9),
                "at": round(self._clock.now, 6),
            },
        )

    def _refuse(self, reason: str, tool: str, *, counted_by_controller: bool = False) -> None:
        """Record a refused prediction once, wherever in the chain it was refused."""
        self.stats.refused[reason] = self.stats.refused.get(reason, 0) + 1
        if not counted_by_controller:
            refusals = self._deps.controller.stats.refusals
            refusals[reason] = refusals.get(reason, 0) + 1
        if self._deps.metrics is not None:
            self._deps.metrics.refusals_total.labels(reason=reason).inc()
        _log.debug("speculation_refused", tool=tool, reason=reason)

    def _constants_for(self, tool_name: str) -> frozenset[str]:
        cached = self._constants.get(tool_name)
        if cached is None:
            schema = self._deps.registry.get(tool_name).args_model.model_json_schema()
            cached = schema_constants(schema)
            self._constants[tool_name] = cached
        return cached

    def _audit(self, action: str, data: Mapping[str, JsonValue]) -> None:
        if self._deps.audit is not None:
            self._deps.audit.append(self._deps.principal, action, dict(data))
