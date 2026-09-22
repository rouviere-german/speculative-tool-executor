"""The latency model. Every simulated second in this repo comes from here.

Two properties matter more than realism, because every number the benchmark publishes depends on
them:

- **draws are deterministic in the call**, so the speculative and non-speculative arms of the A/B
  see the same latency for the same work and the comparison measures speculation rather than the
  random number generator;
- **the distribution is declared**, so a reported percentile can be traced back to an assumption
  a reader can disagree with.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from speculative_tool_executor.spec.latency import LatencyModel, LatencyProfile

pytestmark = pytest.mark.unit


def model(seed: int = 7) -> LatencyModel:
    return LatencyModel(
        tools={
            "fast": LatencyProfile(median_seconds=0.1, sigma=0.3),
            "slow": LatencyProfile(median_seconds=0.9, sigma=0.5),
        },
        default_tool=LatencyProfile(median_seconds=0.2),
        model_turn=LatencyProfile(median_seconds=1.0, sigma=0.4),
        seed=seed,
    )


def test_a_profile_is_described_by_its_median_and_sigma() -> None:
    profile = LatencyProfile(median_seconds=2.0, sigma=0.5, floor_seconds=0.1)
    assert profile.mu == pytest.approx(math.log(2.0))
    assert profile.mean_seconds == pytest.approx(math.exp(math.log(2.0) + 0.125) + 0.1)


def test_draws_follow_the_declared_median() -> None:
    """A sanity check that the profile means what it says: half the mass below the median."""
    profile = LatencyProfile(median_seconds=0.5, sigma=0.6)
    rng = random.Random(0)
    draws = [profile.draw(rng) for _ in range(4000)]
    assert statistics.median(draws) == pytest.approx(0.5, rel=0.08)
    assert min(draws) > 0


def test_a_floor_shifts_every_draw() -> None:
    profile = LatencyProfile(median_seconds=0.5, sigma=0.0, floor_seconds=2.0)
    assert profile.draw(random.Random(1)) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"median_seconds": 0.0},
        {"median_seconds": -1.0},
        {"median_seconds": 1.0, "sigma": -0.1},
        {"median_seconds": 1.0, "floor_seconds": -0.1},
    ],
)
def test_nonsense_profiles_are_refused(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must be"):
        LatencyProfile(**kwargs)


def test_the_same_call_draws_the_same_latency_in_every_run() -> None:
    """Without this the A/B would compare two different workloads."""
    left, right = model(), model()
    assert left.tool_latency("fast", "abc") == right.tool_latency("fast", "abc")
    assert left.model_latency("req") == right.model_latency("req")
    # Re-drawing the same call is stable, not a stream.
    assert left.tool_latency("fast", "abc") == left.tool_latency("fast", "abc")


def test_different_calls_and_different_seeds_draw_differently() -> None:
    assert model().tool_latency("fast", "a") != model().tool_latency("fast", "b")
    assert model(seed=1).tool_latency("fast", "a") != model(seed=2).tool_latency("fast", "a")
    assert model().tool_latency("fast", "a") != model().tool_latency("slow", "a")


def test_an_unknown_tool_falls_back_to_the_default_profile() -> None:
    assert model().profile_for("never_declared").median_seconds == 0.2


def test_the_latency_ratio_is_model_median_over_tool_median() -> None:
    assert model().median_tool_seconds() == pytest.approx(0.5)  # even count: (0.1 + 0.9) / 2
    assert model().latency_ratio == pytest.approx(2.0)

    odd = LatencyModel(
        tools={
            "a": LatencyProfile(median_seconds=0.1),
            "b": LatencyProfile(median_seconds=0.4),
            "c": LatencyProfile(median_seconds=0.9),
        },
        default_tool=LatencyProfile(median_seconds=0.2),
        model_turn=LatencyProfile(median_seconds=0.8),
    )
    assert odd.median_tool_seconds() == pytest.approx(0.4)
    assert odd.latency_ratio == pytest.approx(2.0)


def test_an_empty_tool_table_falls_back_to_the_default_median() -> None:
    empty = LatencyModel(
        tools={},
        default_tool=LatencyProfile(median_seconds=0.25),
        model_turn=LatencyProfile(median_seconds=1.0),
    )
    assert empty.median_tool_seconds() == pytest.approx(0.25)


def test_rescaling_moves_the_model_turn_and_leaves_the_tools_alone() -> None:
    """This is exactly what the sensitivity sweep does, so it has to be the only thing it does."""
    base = model()
    rescaled = base.with_ratio(5.0)
    assert rescaled.latency_ratio == pytest.approx(5.0)
    assert rescaled.model_turn.median_seconds == pytest.approx(2.5)
    assert rescaled.median_tool_seconds() == base.median_tool_seconds()
    assert rescaled.tool_latency("fast", "abc") == base.tool_latency("fast", "abc")
    assert rescaled.model_turn.sigma == base.model_turn.sigma


def test_scaling_a_profile_scales_its_median_and_floor() -> None:
    scaled = LatencyProfile(median_seconds=2.0, sigma=0.3, floor_seconds=0.5).scaled(3.0)
    assert scaled.median_seconds == pytest.approx(6.0)
    assert scaled.floor_seconds == pytest.approx(1.5)
    assert scaled.sigma == 0.3
    with pytest.raises(ValueError, match="factor must be positive"):
        LatencyProfile(median_seconds=1.0).scaled(0.0)


def test_reseeding_keeps_the_distributions_and_changes_the_draws() -> None:
    base = model(seed=1)
    reseeded = base.with_seed(2)
    assert reseeded.seed == 2
    assert reseeded.median_tool_seconds() == base.median_tool_seconds()
    assert reseeded.tool_latency("fast", "abc") != base.tool_latency("fast", "abc")


def test_a_non_positive_ratio_is_refused() -> None:
    with pytest.raises(ValueError, match="ratio must be positive"):
        model().with_ratio(0.0)
