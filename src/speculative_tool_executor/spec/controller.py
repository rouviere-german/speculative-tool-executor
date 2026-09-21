"""The cost-aware speculation controller: when speculating is worth it, and when to stop.

Speculation trades money for latency. Running a predicted call early buys ``p_hit * latency``
seconds and costs ``(1 - p_hit) * cost`` dollars in wasted work, so the decision needs both sides
in the same unit. ``latency_value_usd_per_second`` is that exchange rate, and it is configuration
rather than a constant, because what a saved second is worth is a product decision, not a
property of the system.

Three guards sit on top of the per-call arithmetic:

- a **speculative spend budget**, separate from the agent's own budget, so speculation cannot
  quietly consume the run's money;
- a **waste ratio trip**, which disables speculation for the rest of the run once wasted spend
  passes a share of the baseline spend -- the recovery path when the predictor is simply wrong
  about this workload;
- a **kill switch** and per-tool overrides from configuration, so speculation can be turned off
  in production without a deploy.

Per-tool hit rates are tracked with an exponentially weighted moving average, which follows a
workload that changes shape mid-run without keeping a window of history.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.tools import ToolDefinition

__all__ = [
    "Decision",
    "DecisionReason",
    "SpeculationController",
    "SpeculationPolicy",
    "disabled_tools_from",
    "policy_from_settings",
]


class DecisionReason:
    """Stable strings for why a speculation was or was not started (metrics and audit use them)."""

    OK = "ok"
    GLOBALLY_DISABLED = "globally_disabled"
    TOOL_DISABLED = "tool_disabled"
    EFFECT_NOT_SPECULATABLE = "effect_not_speculatable"
    NEEDS_APPROVAL = "needs_approval"
    LOW_PROBABILITY = "low_probability"
    NOT_WORTH_IT = "not_worth_it"
    SPECULATIVE_BUDGET = "speculative_budget"
    WASTE_RATIO_TRIPPED = "waste_ratio_tripped"
    # Refused before the controller is consulted, by the same checks the real call passes.
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    TAINTED_ARGUMENTS = "tainted_arguments"
    ALREADY_KNOWN = "already_known"
    MAX_IN_FLIGHT = "max_in_flight"


@dataclass(frozen=True, slots=True)
class SpeculationPolicy:
    """Everything the controller's decisions depend on, in one frozen object."""

    enabled: bool = True
    top_k: int = 2
    candidate_multiplier: int = 3
    max_in_flight: int = 4
    disabled_tools: frozenset[str] = frozenset()
    min_probability: float = 0.05
    latency_value_usd_per_second: float = 0.01
    overhead_usd: float = 0.0
    max_waste_ratio: float = 0.10
    speculative_budget_usd: float = 0.50
    ewma_alpha: float = 0.25
    prior_weight: float = 0.5
    warmup_samples: int = 3

    def __post_init__(self) -> None:
        if self.top_k < 0 or self.max_in_flight < 0:
            raise ValueError("top_k and max_in_flight must be non-negative")
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if not 0.0 < self.ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0.0 <= self.prior_weight <= 1.0:
            raise ValueError("prior_weight must be in [0, 1]")
        if self.min_probability < 0 or self.max_waste_ratio < 0:
            raise ValueError("min_probability and max_waste_ratio must be non-negative")
        if self.speculative_budget_usd < 0 or self.latency_value_usd_per_second < 0:
            raise ValueError("budget and latency valuation must be non-negative")

    @property
    def candidates_to_rank(self) -> int:
        """How many predictions to ask for in order to *launch* ``top_k`` of them.

        ``top_k`` counts speculations started, not guesses considered. Ranking exactly ``top_k``
        candidates would let one that can never run -- an external tool, a call already in the
        cache, one whose arguments fail the taint rule -- consume a slot and silently suppress a
        usable prediction behind it. Ranking more and stopping once ``top_k`` have launched costs
        only arithmetic, because refusing a prediction never executes anything.
        """
        return self.top_k * self.candidate_multiplier

    def with_overrides(self, **changes: object) -> SpeculationPolicy:
        current = {
            "enabled": self.enabled,
            "top_k": self.top_k,
            "candidate_multiplier": self.candidate_multiplier,
            "max_in_flight": self.max_in_flight,
            "disabled_tools": self.disabled_tools,
            "min_probability": self.min_probability,
            "latency_value_usd_per_second": self.latency_value_usd_per_second,
            "overhead_usd": self.overhead_usd,
            "max_waste_ratio": self.max_waste_ratio,
            "speculative_budget_usd": self.speculative_budget_usd,
            "ewma_alpha": self.ewma_alpha,
            "prior_weight": self.prior_weight,
            "warmup_samples": self.warmup_samples,
        }
        current.update(changes)
        return SpeculationPolicy(**current)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Decision:
    """The controller's answer for one candidate speculation."""

    allowed: bool
    reason: str
    p_hit: float = 0.0
    expected_benefit_usd: float = 0.0
    expected_waste_usd: float = 0.0

    @property
    def expected_value_usd(self) -> float:
        return self.expected_benefit_usd - self.expected_waste_usd


@dataclass(slots=True)
class _ToolStats:
    attempts: int = 0
    hits: int = 0
    ewma_hit_rate: float = 0.0
    wasted_usd: float = 0.0
    saved_seconds: float = 0.0


@dataclass(slots=True)
class ControllerStats:
    attempts: int = 0
    hits: int = 0
    misses: int = 0
    cancellations: int = 0
    occ_conflicts: int = 0
    stale_reads: int = 0
    speculative_spend_usd: float = 0.0
    wasted_usd: float = 0.0
    baseline_spend_usd: float = 0.0
    latency_saved_seconds: float = 0.0
    tripped: bool = False
    refusals: dict[str, int] = field(default_factory=dict)

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.attempts if self.attempts else 0.0

    @property
    def waste_ratio(self) -> float:
        return self.wasted_usd / self.baseline_spend_usd if self.baseline_spend_usd > 0 else 0.0


class SpeculationController:
    """Decides which predictions to run and keeps the ledgers those decisions depend on."""

    def __init__(self, policy: SpeculationPolicy | None = None) -> None:
        self.policy = policy or SpeculationPolicy()
        self.stats = ControllerStats()
        self._tools: dict[str, _ToolStats] = {}

    @property
    def enabled(self) -> bool:
        return self.policy.enabled and not self.stats.tripped

    def tool_stats(self, tool: str) -> _ToolStats:
        return self._tools.setdefault(tool, _ToolStats())

    def hit_rate(self, tool: str) -> float:
        stats = self._tools.get(tool)
        return stats.ewma_hit_rate if stats else 0.0

    def per_tool_hit_ratio(self) -> dict[str, float]:
        return {
            name: (stats.hits / stats.attempts if stats.attempts else 0.0)
            for name, stats in sorted(self._tools.items())
        }

    def estimate_p_hit(self, tool: str, prior: float) -> float:
        """Blend the predictor's per-call prior with the tool's realized hit rate.

        During warm-up there is no realized rate worth trusting, so the prior stands alone.
        """
        stats = self.tool_stats(tool)
        if stats.attempts < self.policy.warmup_samples:
            return max(0.0, min(1.0, prior))
        weight = self.policy.prior_weight
        blended = weight * prior + (1.0 - weight) * stats.ewma_hit_rate
        return max(0.0, min(1.0, blended))

    def _refuse(self, reason: str, *, p_hit: float = 0.0) -> Decision:
        self.stats.refusals[reason] = self.stats.refusals.get(reason, 0) + 1
        return Decision(allowed=False, reason=reason, p_hit=p_hit)

    def decide(  # noqa: PLR0911 - one return per named refusal reason is the point
        self, tool: ToolDefinition[Any], prior: float
    ) -> Decision:
        """Whether a predicted call to ``tool`` with prior probability ``prior`` may run early."""
        if not self.policy.enabled:
            return self._refuse(DecisionReason.GLOBALLY_DISABLED)
        if self.stats.tripped:
            return self._refuse(DecisionReason.WASTE_RATIO_TRIPPED)
        if tool.name in self.policy.disabled_tools:
            return self._refuse(DecisionReason.TOOL_DISABLED)
        if tool.effect is Effect.EXTERNAL:
            return self._refuse(DecisionReason.EFFECT_NOT_SPECULATABLE)
        if tool.needs_approval:
            return self._refuse(DecisionReason.NEEDS_APPROVAL)
        if prior < self.policy.min_probability:
            return self._refuse(DecisionReason.LOW_PROBABILITY, p_hit=prior)
        remaining = self.policy.speculative_budget_usd - self.stats.speculative_spend_usd
        if tool.expected_cost_usd > remaining:
            return self._refuse(DecisionReason.SPECULATIVE_BUDGET)
        p_hit = self.estimate_p_hit(tool.name, prior)
        benefit = p_hit * tool.expected_latency_seconds * self.policy.latency_value_usd_per_second
        waste = (1.0 - p_hit) * tool.expected_cost_usd + self.policy.overhead_usd
        if benefit <= waste:
            self.stats.refusals[DecisionReason.NOT_WORTH_IT] = (
                self.stats.refusals.get(DecisionReason.NOT_WORTH_IT, 0) + 1
            )
            return Decision(
                allowed=False,
                reason=DecisionReason.NOT_WORTH_IT,
                p_hit=p_hit,
                expected_benefit_usd=benefit,
                expected_waste_usd=waste,
            )
        return Decision(
            allowed=True,
            reason=DecisionReason.OK,
            p_hit=p_hit,
            expected_benefit_usd=benefit,
            expected_waste_usd=waste,
        )

    def record_launch(self, tool: str, cost_usd: float) -> None:
        """A speculation started: it is charged to the speculative budget immediately."""
        stats = self.tool_stats(tool)
        stats.attempts += 1
        self.stats.attempts += 1
        self.stats.speculative_spend_usd += cost_usd

    def record_hit(self, tool: str, *, saved_seconds: float) -> None:
        stats = self.tool_stats(tool)
        stats.hits += 1
        stats.saved_seconds += saved_seconds
        stats.ewma_hit_rate = self._ewma(stats.ewma_hit_rate, 1.0, stats.attempts)
        self.stats.hits += 1
        self.stats.latency_saved_seconds += saved_seconds

    def record_miss(self, tool: str, *, cost_usd: float, cancelled: bool = False) -> None:
        stats = self.tool_stats(tool)
        stats.wasted_usd += cost_usd
        stats.ewma_hit_rate = self._ewma(stats.ewma_hit_rate, 0.0, stats.attempts)
        self.stats.misses += 1
        self.stats.wasted_usd += cost_usd
        if cancelled:
            self.stats.cancellations += 1
        self._check_waste_ratio()

    def record_conflict(self, *, stale_read: bool = False) -> None:
        self.stats.occ_conflicts += 1
        if stale_read:
            self.stats.stale_reads += 1

    def record_baseline_spend(self, cost_usd: float) -> None:
        """Charge what the run would have cost without speculation.

        The waste ratio divides by this, so it must include the model turns, not only the tools.
        """
        self.stats.baseline_spend_usd += cost_usd
        self._check_waste_ratio()

    def _ewma(self, current: float, sample: float, attempts: int) -> float:
        if attempts <= 1:
            return sample
        alpha = self.policy.ewma_alpha
        return (1.0 - alpha) * current + alpha * sample

    def _check_waste_ratio(self) -> None:
        if self.stats.tripped or self.stats.baseline_spend_usd <= 0:
            return
        if self.stats.waste_ratio > self.policy.max_waste_ratio:
            self.stats.tripped = True

    def snapshot(self) -> dict[str, float | bool | dict[str, float] | dict[str, int]]:
        """A JSON-friendly view for ``/v1/speculation`` and the demo summary."""
        return {
            "enabled": self.enabled,
            "tripped": self.stats.tripped,
            "attempts": float(self.stats.attempts),
            "hits": float(self.stats.hits),
            "misses": float(self.stats.misses),
            "hit_ratio": self.stats.hit_ratio,
            "cancellations": float(self.stats.cancellations),
            "occ_conflicts": float(self.stats.occ_conflicts),
            "stale_reads": float(self.stats.stale_reads),
            "speculative_spend_usd": self.stats.speculative_spend_usd,
            "wasted_usd": self.stats.wasted_usd,
            "baseline_spend_usd": self.stats.baseline_spend_usd,
            "waste_ratio": self.stats.waste_ratio,
            "latency_saved_seconds": self.stats.latency_saved_seconds,
            "per_tool_hit_ratio": self.per_tool_hit_ratio(),
            "refusals": dict(sorted(self.stats.refusals.items())),
        }


class SettingsLike(Protocol):
    """The slice of :class:`~...core.config.Settings` the controller depends on.

    The members are declared as read-only properties rather than attributes: ``Settings`` is a
    frozen model, and a Protocol with settable attributes would not match it.
    """

    @property
    def speculation_enabled(self) -> bool: ...
    @property
    def speculation_top_k(self) -> int: ...
    @property
    def speculation_max_in_flight(self) -> int: ...
    @property
    def speculation_disabled_tools(self) -> list[str]: ...
    @property
    def speculation_min_probability(self) -> float: ...
    @property
    def speculation_latency_value_usd_per_second(self) -> float: ...
    @property
    def speculation_budget_usd(self) -> float: ...
    @property
    def speculation_max_waste_ratio(self) -> float: ...


def policy_from_settings(settings: SettingsLike) -> SpeculationPolicy:
    """Build the runtime policy from configuration.

    The kill switch and the per-tool overrides live here rather than in code so that speculation
    can be narrowed or turned off in an incident without a deploy. The controller consults the
    policy on every decision, so a restart with a new environment takes effect immediately and
    nothing has to be invalidated.
    """
    return SpeculationPolicy(
        enabled=settings.speculation_enabled,
        top_k=settings.speculation_top_k,
        max_in_flight=settings.speculation_max_in_flight,
        disabled_tools=disabled_tools_from(settings.speculation_disabled_tools),
        min_probability=settings.speculation_min_probability,
        latency_value_usd_per_second=settings.speculation_latency_value_usd_per_second,
        speculative_budget_usd=settings.speculation_budget_usd,
        max_waste_ratio=settings.speculation_max_waste_ratio,
    )


def disabled_tools_from(values: Iterable[str] | Mapping[str, bool] | None) -> frozenset[str]:
    """Normalize a per-tool override list from configuration into a frozen set of names."""
    if values is None:
        return frozenset()
    if isinstance(values, Mapping):
        return frozenset(name for name, allowed in values.items() if not allowed)
    return frozenset(values)
