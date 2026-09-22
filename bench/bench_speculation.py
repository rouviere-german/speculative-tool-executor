"""The speculation benchmark: everything the README's Results section claims.

Usage:
  uv run python bench/bench_speculation.py            # run, write bench/results/speculation.json
  uv run python bench/bench_speculation.py --check     # enforce speculation_thresholds.json
  uv run python bench/bench_speculation.py --quick     # a smaller run for smoke tests
  uv run python bench/bench_speculation.py --markdown BENCHMARKS.md

Every number here comes from a discrete-event simulation on a virtual clock, not from measuring a
real agent. The latency distributions are assumptions declared in
``speculative_tool_executor.workload.tools``; what the simulation measures is what *this design*
does with those assumptions. Wall-clock runtime of the benchmark itself is reported separately so
that "simulated seconds" and "seconds this took" are never confused.

Thresholds are one-sided on purpose:

- ``min`` entries are floors on things that should stay good (tool-wait reduction, coverage);
- ``max`` entries are ceilings on things that should stay small (waste ratio, OCC conflicts) or
  that must stay zero (external speculations, tasks whose output diverged).

The floors sit below the measured values with room for noise, and the zero-valued ceilings are
exact: a single external speculation or a single divergent task is a correctness failure, not a
regression, so there is no tolerance to spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running the file directly
    sys.path.insert(0, str(ROOT / "src"))

from speculative_tool_executor.core.logging import configure_logging  # noqa: E402
from speculative_tool_executor.sim.experiment import (  # noqa: E402
    Comparison,
    run_ab,
    run_ablations,
)
from speculative_tool_executor.workload.corpus import build_corpus  # noqa: E402
from speculative_tool_executor.workload.tasks import generate_tasks  # noqa: E402
from speculative_tool_executor.workload.tools import (  # noqa: E402
    MODEL_TURN_PROFILE,
    TOOL_PROFILES,
)

RESULTS = ROOT / "bench" / "results" / "speculation.json"
THRESHOLDS = ROOT / "bench" / "speculation_thresholds.json"

HEADLINE_TASKS = 300
SWEEP_TASKS = 150
QUICK_TASKS = 40
RATIOS = (1.0, 2.0, 4.0, 7.0, 10.0)
BIASES = (0.0, 1.0, 2.0)
CONTENTIONS = (0.0, 0.5)


def machine() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
    }


def assumptions() -> dict[str, Any]:
    """The declared inputs every latency number below is a consequence of."""
    return {
        "model_turn_median_seconds": MODEL_TURN_PROFILE.median_seconds,
        "model_turn_sigma": MODEL_TURN_PROFILE.sigma,
        "tool_median_seconds": {
            name: profile.median_seconds for name, profile in sorted(TOOL_PROFILES.items())
        },
        "note": (
            "Lognormal latencies drawn deterministically from each call's hash. Simulated, "
            "not measured from a deployed agent."
        ),
    }


def _headline(comparison: Comparison) -> dict[str, Any]:
    summary = comparison.summary()
    summary["divergent_tasks"] = len(comparison.divergent_tasks)
    summary["equivalent_task_ratio"] = (
        1.0 - len(comparison.divergent_tasks) / comparison.baseline.tasks
        if comparison.baseline.tasks
        else 1.0
    )
    summary["speculative_spend_usd"] = round(comparison.speculative.speculative_spend_usd, 6)
    summary["baseline_spend_usd"] = round(comparison.baseline.baseline_spend_usd, 6)
    return summary


async def _run(tasks_count: int, sweep_count: int) -> dict[str, Any]:
    corpus = build_corpus()

    headline_tasks = generate_tasks(count=tasks_count, corpus=corpus)
    headline = _headline(await run_ab(headline_tasks, corpus=corpus))

    sweep_tasks = generate_tasks(count=sweep_count, corpus=corpus)
    ratios = []
    for ratio in RATIOS:
        comparison = await run_ab(sweep_tasks, corpus=corpus, latency_ratio=ratio)
        row = comparison.summary()
        ratios.append(
            {
                "ratio": ratio,
                "p50_reduction": row["p50_reduction"],
                "p95_reduction": row["p95_reduction"],
                "tool_wait_reduction": row["tool_wait_reduction"],
                "coverage": row["coverage"],
                "cancellations": row["cancellations"],
                "equivalent": row["equivalent"],
            }
        )

    biases = []
    for bias in BIASES:
        biased = generate_tasks(count=sweep_count, corpus=corpus, read_bias=bias)
        comparison = await run_ab(biased, corpus=corpus, read_bias=bias)
        row = comparison.summary()
        biases.append(
            {
                "read_bias": bias,
                "hit_ratio": row["hit_ratio"],
                "coverage": row["coverage"],
                "tool_wait_reduction": row["tool_wait_reduction"],
                "p50_reduction": row["p50_reduction"],
                "waste_ratio": row["waste_ratio"],
                "equivalent": row["equivalent"],
            }
        )

    contentions = []
    for contention in CONTENTIONS:
        comparison = await run_ab(sweep_tasks, corpus=corpus, contention=contention)
        row = comparison.summary()
        contentions.append(
            {
                "contention": contention,
                "occ_conflicts": row["occ_conflicts"],
                "hit_ratio": row["hit_ratio"],
                "tool_wait_reduction": row["tool_wait_reduction"],
                "store_identical": row["store_identical"],
                "equivalent": row["equivalent"],
            }
        )

    ablations = await run_ablations(sweep_tasks, corpus=corpus)
    return {
        "headline": headline,
        "latency_ratio_sweep": ratios,
        "predictability_sweep": biases,
        "contention": contentions,
        "ablations": ablations,
        "tasks": tasks_count,
        "sweep_tasks": sweep_count,
    }


def markdown(payload: dict[str, Any]) -> str:
    """Render BENCHMARKS.md from the measured payload. Nothing here is written by hand."""
    head = payload["results"]["headline"]
    lines = [
        "# Benchmarks",
        "",
        "Generated by `uv run python bench/bench_speculation.py --markdown BENCHMARKS.md`.",
        "Do not edit by hand.",
        "",
        "All latencies are **simulated** on a virtual clock from the lognormal distributions "
        "declared in `src/speculative_tool_executor/workload/tools.py`. They are not measurements "
        "of a deployed agent. The benchmark's own wall-clock runtime is reported at the end.",
        "",
        f"- Machine: `{payload['machine']['platform']}`, Python "
        f"{payload['machine']['python']}, {payload['machine']['cpu_count']} CPUs",
        f"- Tasks: {payload['results']['tasks']} for the headline, "
        f"{payload['results']['sweep_tasks']} per sweep point",
        f"- Model turn median {payload['assumptions']['model_turn_median_seconds']}s; "
        f"tool medians {payload['assumptions']['tool_median_seconds']}",
        "",
        "## Headline: speculation on versus off",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "tasks",
        "p50_baseline_seconds",
        "p50_speculative_seconds",
        "p50_reduction",
        "p95_baseline_seconds",
        "p95_speculative_seconds",
        "p95_reduction",
        "tool_wait_reduction",
        "hit_ratio",
        "coverage",
        "waste_ratio",
        "wasted_usd",
        "baseline_spend_usd",
        "occ_conflicts",
        "cancellations",
        "external_speculations",
        "divergent_tasks",
        "equivalent",
    ):
        lines.append(f"| {key} | {head[key]} |")

    lines += [
        "",
        "## Sensitivity to the model-to-tool latency ratio",
        "",
        "| ratio | p50 reduction | p95 reduction | tool-wait reduction | coverage | "
        "cancellations | equivalent |",
        "|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in payload["results"]["latency_ratio_sweep"]:
        lines.append(
            f"| {row['ratio']}:1 | {row['p50_reduction']} | {row['p95_reduction']} | "
            f"{row['tool_wait_reduction']} | {row['coverage']} | {row['cancellations']} | "
            f"{row['equivalent']} |"
        )

    lines += [
        "",
        "## Sensitivity to how predictable the workload is",
        "",
        "`read_bias` is the exponent of the rank-biased choice of which listed file gets read; "
        "0 is a uniform (unpredictable) choice and is the default everywhere else in this repo.",
        "",
        "| read_bias | hit ratio | coverage | tool-wait reduction | p50 reduction | waste ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["results"]["predictability_sweep"]:
        lines.append(
            f"| {row['read_bias']} | {row['hit_ratio']} | {row['coverage']} | "
            f"{row['tool_wait_reduction']} | {row['p50_reduction']} | {row['waste_ratio']} |"
        )

    lines += [
        "",
        "## Write contention (another party writing the keys speculations read)",
        "",
        "| contention | OCC conflicts | hit ratio | tool-wait reduction | store identical |",
        "|---:|---:|---:|---:|:--:|",
    ]
    for row in payload["results"]["contention"]:
        lines.append(
            f"| {row['contention']} | {row['occ_conflicts']} | {row['hit_ratio']} | "
            f"{row['tool_wait_reduction']} | {row['store_identical']} |"
        )

    lines += [
        "",
        "## Ablations",
        "",
        "| variant | p50 (s) | p50 reduction | hit ratio | attempts | waste ratio | equivalent |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for row in payload["results"]["ablations"]:
        lines.append(
            f"| {row['name']} | {row['p50_seconds']} | {row['p50_reduction']} | "
            f"{row['hit_ratio']} | {row['attempts']} | {row['waste_ratio']} | "
            f"{row['equivalent']} |"
        )

    lines += [
        "",
        f"Benchmark wall-clock runtime: {payload['runtime_seconds']}s.",
        "",
    ]
    return "\n".join(lines)


def check_thresholds(payload: dict[str, Any]) -> list[str]:
    """Compare the headline against the declared floors and ceilings."""
    limits: dict[str, dict[str, float]] = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    head = payload["results"]["headline"]
    problems: list[str] = []
    for metric, floor in limits.get("min", {}).items():
        value = float(head[metric])
        if value < floor:
            problems.append(f"{metric} = {value} is below the floor {floor}")
    for metric, ceiling in limits.get("max", {}).items():
        value = float(head[metric])
        if value > ceiling:
            problems.append(f"{metric} = {value} exceeds the ceiling {ceiling}")
    for row in payload["results"]["contention"]:
        if not row["store_identical"]:
            problems.append(f"contention {row['contention']}: store diverged")
    for row in payload["results"]["latency_ratio_sweep"]:
        if not row["equivalent"]:
            problems.append(f"ratio {row['ratio']}: run was not equivalent to the baseline")
    for row in payload["results"]["ablations"]:
        if not row["equivalent"]:
            problems.append(f"ablation {row['name']}: run was not equivalent to the baseline")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="speculation benchmark")
    parser.add_argument("--check", action="store_true", help="enforce the declared thresholds")
    parser.add_argument("--quick", action="store_true", help="small run for smoke tests")
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    # The benchmark's output is a table; the runtime's own debug stream would drown it.
    configure_logging("WARNING", json_output=False, stream=sys.stderr)
    tasks = args.tasks or (QUICK_TASKS if args.quick else HEADLINE_TASKS)
    sweep = min(tasks, QUICK_TASKS if args.quick else SWEEP_TASKS)

    started = time.perf_counter()
    results = asyncio.run(_run(tasks, sweep))
    runtime = round(time.perf_counter() - started, 1)

    payload = {
        "machine": machine(),
        "assumptions": assumptions(),
        "results": results,
        "runtime_seconds": runtime,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    head = results["headline"]
    print(f"speculation benchmark ({results['tasks']} tasks, {runtime}s wall clock)")
    print("| metric | value |")
    print("|---|---:|")
    for key in (
        "p50_baseline_seconds",
        "p50_speculative_seconds",
        "p50_reduction",
        "p95_reduction",
        "tool_wait_reduction",
        "hit_ratio",
        "coverage",
        "waste_ratio",
        "occ_conflicts",
        "cancellations",
        "external_speculations",
        "divergent_tasks",
        "equivalent",
    ):
        print(f"| {key} | {head[key]} |")

    if args.markdown is not None:
        target = args.markdown if args.markdown.is_absolute() else ROOT / args.markdown
        target.write_text(markdown(payload), encoding="utf-8")
        print(f"wrote {target.relative_to(ROOT)}")

    if args.check:
        problems = check_thresholds(payload)
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("speculation benchmark within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
