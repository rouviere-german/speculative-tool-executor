"""The cost-aware controller: when speculating is worth it, and when to stop.

Speculation buys latency with money, so every decision here is an inequality between the two.
The tests below pin down the arithmetic, the three guards on top of it (speculative budget, waste
trip, kill switch) and the way per-tool hit rates are learned.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, JsonValue

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.agentic.tools import Capability, ToolDefinition
from speculative_tool_executor.spec.controller import (
    DecisionReason,
    SpeculationController,
    SpeculationPolicy,
    disabled_tools_from,
)

pytestmark = pytest.mark.unit


class Args(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = "x"


def tool(
    name: str = "read_file",
    *,
    effect: Effect = Effect.READ,
    latency: float = 1.0,
    cost: float = 0.001,
    requires_approval: bool | None = None,
) -> ToolDefinition[Args]:
    capabilities = {
        Effect.PURE: Capability.COMPUTE,
        Effect.READ: Capability.READ_DATA,
        Effect.WRITE: Capability.WRITE_DATA,
        Effect.EXTERNAL: Capability.SEND_MESSAGE,
    }[effect]
    return ToolDefinition(
        name=name,
        description="x",
        args_model=Args,
        effect=effect,
        capabilities=frozenset({capabilities}),
        handler=lambda args: args.value,
        requires_approval=requires_approval
        if requires_approval is not None
        else (False if effect is Effect.WRITE else None),
        freshness_seconds=None if effect is Effect.PURE else 30.0,
        expected_latency_seconds=latency,
        expected_cost_usd=cost,
    )


def test_speculation_is_allowed_when_expected_benefit_beats_expected_waste() -> None:
    """benefit = p_hit x latency x value-of-a-second; waste = (1 - p_hit) x cost."""
    controller = SpeculationController(
        SpeculationPolicy(latency_value_usd_per_second=0.01, min_probability=0.0)
    )
    decision = controller.decide(tool(latency=1.0, cost=0.001), prior=0.5)
    assert decision.allowed is True
    assert decision.reason == DecisionReason.OK
    assert decision.expected_benefit_usd == pytest.approx(0.5 * 1.0 * 0.01)
    assert decision.expected_waste_usd == pytest.approx(0.5 * 0.001)
    assert decision.expected_value_usd > 0


def test_a_call_that_is_not_worth_it_is_refused_with_the_arithmetic_attached() -> None:
    controller = SpeculationController(
        SpeculationPolicy(latency_value_usd_per_second=0.0001, min_probability=0.0)
    )
    # Cost stays inside the speculative budget so the budget guard is not what refuses this.
    decision = controller.decide(tool(latency=0.01, cost=0.1), prior=0.5)
    assert decision.allowed is False
    assert decision.reason == DecisionReason.NOT_WORTH_IT
    assert decision.expected_benefit_usd < decision.expected_waste_usd
    assert controller.stats.refusals[DecisionReason.NOT_WORTH_IT] == 1


def test_external_tools_are_refused_whatever_the_probability() -> None:
    controller = SpeculationController(SpeculationPolicy(latency_value_usd_per_second=1e6))
    decision = controller.decide(tool("send", effect=Effect.EXTERNAL, latency=100.0), prior=1.0)
    assert decision.allowed is False
    assert decision.reason == DecisionReason.EFFECT_NOT_SPECULATABLE


def test_tools_needing_approval_are_refused() -> None:
    controller = SpeculationController(SpeculationPolicy(latency_value_usd_per_second=1e6))
    decision = controller.decide(tool("reviewed", requires_approval=True), prior=1.0)
    assert decision.reason == DecisionReason.NEEDS_APPROVAL


def test_low_probability_predictions_are_refused_before_the_cost_model_runs() -> None:
    controller = SpeculationController(SpeculationPolicy(min_probability=0.2))
    decision = controller.decide(tool(), prior=0.19)
    assert decision.reason == DecisionReason.LOW_PROBABILITY
    assert decision.p_hit == pytest.approx(0.19)


def test_the_speculative_budget_is_separate_from_the_agent_budget() -> None:
    controller = SpeculationController(
        SpeculationPolicy(speculative_budget_usd=0.002, min_probability=0.0)
    )
    definition = tool(cost=0.001)
    assert controller.decide(definition, prior=1.0).allowed is True
    controller.record_launch(definition.name, definition.expected_cost_usd)
    controller.record_launch(definition.name, definition.expected_cost_usd)
    assert controller.decide(definition, prior=1.0).reason == DecisionReason.SPECULATIVE_BUDGET


def test_the_waste_trip_disables_speculation_for_the_rest_of_the_run() -> None:
    """The recovery path when the predictor is simply wrong about this workload."""
    controller = SpeculationController(SpeculationPolicy(max_waste_ratio=0.10, min_probability=0.0))
    controller.record_baseline_spend(1.0)
    before_trip = controller.enabled
    assert before_trip is True

    controller.record_launch("read_file", 0.2)
    controller.record_miss("read_file", cost_usd=0.2)

    assert controller.stats.tripped is True
    after_trip = controller.enabled
    assert after_trip is False
    assert controller.decide(tool(), prior=1.0).reason == DecisionReason.WASTE_RATIO_TRIPPED


def test_the_waste_ratio_needs_a_denominator_before_it_can_trip() -> None:
    controller = SpeculationController(SpeculationPolicy(max_waste_ratio=0.0))
    controller.record_miss("read_file", cost_usd=5.0)
    assert controller.stats.tripped is False


def test_the_kill_switch_refuses_everything() -> None:
    controller = SpeculationController(SpeculationPolicy(enabled=False))
    assert controller.enabled is False
    assert controller.decide(tool(), prior=1.0).reason == DecisionReason.GLOBALLY_DISABLED


def test_per_tool_disable_refuses_only_that_tool() -> None:
    controller = SpeculationController(
        SpeculationPolicy(disabled_tools=frozenset({"read_file"}), min_probability=0.0)
    )
    assert controller.decide(tool("read_file"), prior=1.0).reason == DecisionReason.TOOL_DISABLED
    assert controller.decide(tool("list_dir"), prior=1.0).allowed is True


def test_the_prior_stands_alone_until_warm_up_is_over() -> None:
    controller = SpeculationController(
        SpeculationPolicy(warmup_samples=2, prior_weight=0.5, ewma_alpha=1.0)
    )
    assert controller.estimate_p_hit("read_file", 0.8) == pytest.approx(0.8)

    # Two attempts, both missed: the realised rate is now trusted and drags the estimate down.
    for _ in range(2):
        controller.record_launch("read_file", 0.0)
        controller.record_miss("read_file", cost_usd=0.0)
    assert controller.estimate_p_hit("read_file", 0.8) == pytest.approx(0.4)


def test_hit_rates_are_tracked_per_tool() -> None:
    controller = SpeculationController(SpeculationPolicy(ewma_alpha=1.0))
    controller.record_launch("read_file", 0.0)
    controller.record_hit("read_file", saved_seconds=1.5)
    controller.record_launch("save_note", 0.0)
    controller.record_miss("save_note", cost_usd=0.001, cancelled=True)

    assert controller.per_tool_hit_ratio() == {"read_file": 1.0, "save_note": 0.0}
    assert controller.hit_rate("read_file") == pytest.approx(1.0)
    assert controller.hit_rate("never_seen") == 0.0
    assert controller.stats.latency_saved_seconds == pytest.approx(1.5)
    assert controller.stats.cancellations == 1


def test_estimates_stay_inside_zero_and_one() -> None:
    controller = SpeculationController(SpeculationPolicy(warmup_samples=0))
    assert controller.estimate_p_hit("t", -5.0) == 0.0
    assert controller.estimate_p_hit("t", 5.0) == 1.0


def test_conflicts_are_recorded_and_distinguish_stale_reads() -> None:
    controller = SpeculationController()
    controller.record_conflict()
    controller.record_conflict(stale_read=True)
    assert controller.stats.occ_conflicts == 2
    assert controller.stats.stale_reads == 1


def test_snapshot_is_json_friendly_and_complete() -> None:
    controller = SpeculationController()
    controller.record_baseline_spend(1.0)
    controller.record_launch("read_file", 0.001)
    controller.record_hit("read_file", saved_seconds=0.5)
    snapshot = controller.snapshot()
    for key in (
        "enabled",
        "tripped",
        "hit_ratio",
        "waste_ratio",
        "latency_saved_seconds",
        "per_tool_hit_ratio",
        "refusals",
        "occ_conflicts",
    ):
        assert key in snapshot
    assert snapshot["hit_ratio"] == 1.0


def test_candidates_to_rank_exceeds_top_k_so_refusals_do_not_steal_slots() -> None:
    policy = SpeculationPolicy(top_k=2, candidate_multiplier=3)
    assert policy.candidates_to_rank == 6


def test_policy_rejects_nonsense_configuration() -> None:
    for kwargs in (
        {"top_k": -1},
        {"max_in_flight": -1},
        {"candidate_multiplier": 0},
        {"ewma_alpha": 0.0},
        {"ewma_alpha": 1.5},
        {"prior_weight": -0.1},
        {"min_probability": -1.0},
        {"speculative_budget_usd": -1.0},
    ):
        with pytest.raises(ValueError, match="must be"):
            SpeculationPolicy(**kwargs)


def test_overrides_produce_a_new_policy_without_mutating_the_old_one() -> None:
    policy = SpeculationPolicy(top_k=2)
    changed = policy.with_overrides(top_k=5, enabled=False)
    assert policy.top_k == 2
    assert policy.enabled is True
    assert changed.top_k == 5
    assert changed.enabled is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, frozenset()),
        (["a", "b"], frozenset({"a", "b"})),
        ({"a": True, "b": False}, frozenset({"b"})),
    ],
)
def test_disabled_tools_from_configuration(value: JsonValue, expected: frozenset[str]) -> None:
    assert disabled_tools_from(value) == expected  # type: ignore[arg-type]
