"""The latency model the simulator draws from.

Tool and model latencies are lognormal. That is the usual shape for service response times -- a
floor, a dense body and a long right tail -- and it is the distribution whose parameters can be
stated honestly in the README, which matters more here than realism: every latency number this
repo reports is a *simulation*, and the only defensible way to publish one is to publish the
distribution it came from.

Draws are **deterministic in the call**, not in the order of execution: the seed for a draw is
``seed:tool:call_hash``. Two runs of the same workload -- with speculation on and with it off --
therefore see exactly the same latency for the same call, which is what makes the A/B comparison
a measurement of speculation rather than of the random number generator.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["LatencyModel", "LatencyProfile"]


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """A lognormal latency in seconds, given by its median and the sigma of ``log`` latency."""

    median_seconds: float
    sigma: float = 0.5
    floor_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.median_seconds <= 0:
            raise ValueError("median_seconds must be positive")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative")
        if self.floor_seconds < 0:
            raise ValueError("floor_seconds must be non-negative")

    @property
    def mu(self) -> float:
        return math.log(self.median_seconds)

    @property
    def mean_seconds(self) -> float:
        return math.exp(self.mu + self.sigma**2 / 2) + self.floor_seconds

    def scaled(self, factor: float) -> LatencyProfile:
        if factor <= 0:
            raise ValueError("factor must be positive")
        return LatencyProfile(
            median_seconds=self.median_seconds * factor,
            sigma=self.sigma,
            floor_seconds=self.floor_seconds * factor,
        )

    def draw(self, rng: random.Random) -> float:
        return self.floor_seconds + math.exp(rng.gauss(self.mu, self.sigma))


class LatencyModel:
    """Per-tool and per-model-turn latency draws, reproducible from the call itself."""

    def __init__(
        self,
        *,
        tools: Mapping[str, LatencyProfile],
        default_tool: LatencyProfile,
        model_turn: LatencyProfile,
        seed: int = 0,
    ) -> None:
        self._tools = dict(tools)
        self._default = default_tool
        self._model_turn = model_turn
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def model_turn(self) -> LatencyProfile:
        return self._model_turn

    def profile_for(self, tool: str) -> LatencyProfile:
        return self._tools.get(tool, self._default)

    def median_tool_seconds(self) -> float:
        """Median of the per-tool medians; the denominator of the model-to-tool latency ratio."""
        medians = sorted(
            [profile.median_seconds for profile in self._tools.values()]
            or [self._default.median_seconds]
        )
        middle = len(medians) // 2
        if len(medians) % 2:
            return medians[middle]
        return (medians[middle - 1] + medians[middle]) / 2

    @property
    def latency_ratio(self) -> float:
        """Median model-turn latency divided by median tool latency."""
        return self._model_turn.median_seconds / self.median_tool_seconds()

    def with_ratio(self, ratio: float) -> LatencyModel:
        """The same tool latencies with the model turn rescaled to the requested ratio."""
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        target = self.median_tool_seconds() * ratio
        model_turn = LatencyProfile(
            median_seconds=target,
            sigma=self._model_turn.sigma,
            floor_seconds=self._model_turn.floor_seconds,
        )
        return LatencyModel(
            tools=self._tools,
            default_tool=self._default,
            model_turn=model_turn,
            seed=self._seed,
        )

    def with_seed(self, seed: int) -> LatencyModel:
        return LatencyModel(
            tools=self._tools,
            default_tool=self._default,
            model_turn=self._model_turn,
            seed=seed,
        )

    def _rng(self, *parts: str) -> random.Random:
        return random.Random(f"{self._seed}:" + ":".join(parts))  # noqa: S311  # nosec B311

    def tool_latency(self, tool: str, call_hash: str) -> float:
        """The simulated duration of one tool call, identical in every run of this workload."""
        return self.profile_for(tool).draw(self._rng("tool", tool, call_hash))

    def model_latency(self, request_hash: str) -> float:
        """The simulated duration of one model turn."""
        return self._model_turn.draw(self._rng("model", request_hash))
