"""Running one tool call inside one branch. The single path both speculation and reality take.

Ordering here is a modelling decision worth stating: the handler runs **first**, against the
store as it is at the moment the call is issued, and the simulated latency is spent afterwards.
That is how a real call behaves -- a read observes the world when it is sent, not when its reply
arrives -- and it is what gives optimistic concurrency control something to catch: a write that
commits while a speculation is in flight invalidates the version that speculation read, exactly
as it would in a distributed system.

Because the handler has already run when a speculation is cancelled, cancellation safety cannot
come from "it never executed". It comes from the effect classes: only ``PURE``, ``READ`` and
``WRITE`` are ever speculated, and a ``WRITE`` speculates into an overlay that is discarded. The
runner re-checks that invariant itself rather than trusting its caller.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from speculative_tool_executor.agentic.effects import SPECULATABLE_EFFECTS, Effect
from speculative_tool_executor.agentic.tools import ValidatedCall
from speculative_tool_executor.core.logging import get_logger
from speculative_tool_executor.spec.clock import Clock
from speculative_tool_executor.spec.context import ExecutionContext, use_execution
from speculative_tool_executor.spec.latency import LatencyModel
from speculative_tool_executor.spec.overlay import WriteOverlay
from speculative_tool_executor.spec.store import VersionedStore

__all__ = ["BranchOutcome", "SpeculationNotAllowedError", "ToolRunner"]

_log = get_logger("speculative_tool_executor.runner")

Outcome = Literal["ok", "error"]
_READ_ONLY_EFFECTS = frozenset({Effect.PURE, Effect.READ})


class SpeculationNotAllowedError(RuntimeError):
    """A caller tried to speculate an effect class that may never be run ahead of the model."""


@dataclass(frozen=True, slots=True)
class BranchOutcome:
    """What one branch produced: its result, its buffered writes and the time it took."""

    call_hash: str
    tool: str
    effect: Effect
    branch_id: str
    outcome: Outcome
    output: JsonValue
    error: str | None
    overlay: WriteOverlay
    latency_seconds: float
    started_at: float
    finished_at: float

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"


class ToolRunner:
    """Executes a validated call against a fresh branch and charges it the simulated latency."""

    def __init__(
        self,
        *,
        store: VersionedStore,
        clock: Clock,
        latency: LatencyModel,
    ) -> None:
        self._store = store
        self._clock = clock
        self._latency = latency

    @property
    def clock(self) -> Clock:
        return self._clock

    @property
    def latency(self) -> LatencyModel:
        return self._latency

    def latency_for(self, validated: ValidatedCall) -> float:
        return self._latency.tool_latency(validated.tool.name, validated.call_hash)

    def new_overlay(self, validated: ValidatedCall, branch_id: str) -> WriteOverlay:
        return WriteOverlay(
            store=self._store,
            branch_id=branch_id,
            read_only=validated.tool.effect in _READ_ONLY_EFFECTS,
        )

    async def run(
        self,
        validated: ValidatedCall,
        *,
        branch_id: str,
        speculative: bool,
        authorized_hash: str | None = None,
    ) -> BranchOutcome:
        """Run one call in its own branch. Cancellation discards the branch and re-raises."""
        tool = validated.tool
        if speculative and tool.effect not in SPECULATABLE_EFFECTS:
            raise SpeculationNotAllowedError(
                f"tool {tool.name} has effect {tool.effect.value} and is never speculated"
            )
        if speculative and tool.needs_approval:
            raise SpeculationNotAllowedError(
                f"tool {tool.name} requires approval and is never speculated"
            )
        overlay = self.new_overlay(validated, branch_id)
        context = ExecutionContext(
            overlay=overlay, clock=self._clock, branch_id=branch_id, speculative=speculative
        )
        started_at = self._clock.now
        duration = self.latency_for(validated)
        outcome: Outcome = "ok"
        output: JsonValue = None
        error: str | None = None
        try:
            with use_execution(context):
                output = validated.run(authorized_hash=authorized_hash)
        except Exception as exc:  # noqa: BLE001 - handler failures are data, not crashes
            outcome, error = "error", type(exc).__name__
            _log.warning(
                "tool_failed",
                tool=tool.name,
                branch=branch_id,
                speculative=speculative,
                error_type=error,
            )
        try:
            await self._clock.sleep(duration)
        except asyncio.CancelledError:
            overlay.abort()
            raise
        return BranchOutcome(
            call_hash=validated.call_hash,
            tool=tool.name,
            effect=tool.effect,
            branch_id=branch_id,
            outcome=outcome,
            output=output,
            error=error,
            overlay=overlay,
            latency_seconds=duration,
            started_at=started_at,
            finished_at=self._clock.now,
        )
