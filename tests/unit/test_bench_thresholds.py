"""The benchmark gate must cover the number the README leads with.

``bench/bench_speculation.py --check`` is what CI runs. These tests load its ``check_thresholds``
against the committed ``bench/speculation_thresholds.json`` with a synthetic headline, so a
regression of the end-to-end figures fails the gate even when tool-wait reduction holds up.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

BENCH = Path(__file__).resolve().parents[2] / "bench" / "bench_speculation.py"

#: The headline this repository reports (README Results), rounded as the benchmark prints it.
MEASURED: dict[str, Any] = {
    "p50_reduction": 0.0881,
    "p95_reduction": 0.0536,
    "tool_wait_reduction": 0.3456,
    "coverage": 0.7533,
    "hit_ratio": 0.4115,
    "waste_ratio": 0.0154,
    "external_speculations": 0,
    "divergent_tasks": 0,
}


def _bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_speculation_under_test", BENCH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(**headline: Any) -> dict[str, Any]:
    return {
        "results": {
            "headline": {**MEASURED, **headline},
            "contention": [],
            "latency_ratio_sweep": [],
            "ablations": [],
        }
    }


def test_the_measured_headline_passes_the_gate() -> None:
    assert _bench().check_thresholds(_payload()) == []


@pytest.mark.parametrize("metric", ["p50_reduction", "p95_reduction"])
def test_losing_the_end_to_end_gain_fails_the_gate_even_if_tool_wait_holds(metric: str) -> None:
    problems = _bench().check_thresholds(_payload(**{metric: 0.0}))
    assert len(problems) == 1
    assert problems[0].startswith(f"{metric} = 0.0 is below the floor")
