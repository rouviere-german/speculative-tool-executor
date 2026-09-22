"""Scenarios, arms and the A/B itself.

A *scenario* is a complete description of one run: whether speculation is on, which predictor and
top-k, whether the declared argument equivalences are applied, what the model-to-tool latency
ratio is, and how much write contention another party creates. An *arm* is a scenario
instantiated -- its own store, registry, clock and executor -- so two arms never share mutable
state and can be compared without one having warmed the other's caches.

Everything here is deliberately reproducible: the corpus, the tasks, the predictor's training
trajectories, the latency draws and the contention schedule are all functions of seeds. Running
the same scenario twice produces byte-identical results, which is what lets the benchmark be a
CI regression gate rather than a mood ring.

The contention injector deserves a note. It writes to the store between the model turn and the
authoritative call -- precisely the window in which a finished speculative branch is waiting to be
claimed. It fires at the same points in the call sequence in both arms, so the two arms do the
same work; what differs is that in the speculative arm those writes invalidate read sets, which is
how ``occ_conflicts`` becomes a number measured rather than asserted.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from speculative_tool_executor.agentic.approval import ApprovalGate, ApprovalStore
from speculative_tool_executor.sim.agent import (
    SimulatedAgent,
    TaskRun,
    preapprove_external_calls,
)
from speculative_tool_executor.spec.cache import SpeculativeCache
from speculative_tool_executor.spec.clock import VirtualClock
from speculative_tool_executor.spec.controller import SpeculationController, SpeculationPolicy
from speculative_tool_executor.spec.executor import SpeculativeToolExecutor
from speculative_tool_executor.spec.latency import LatencyModel
from speculative_tool_executor.spec.predictor import MarkovPredictor, NullPredictor, Predictor
from speculative_tool_executor.spec.store import VersionedStore
from speculative_tool_executor.spec.taint import TaintTracker
from speculative_tool_executor.workload.corpus import Corpus, build_corpus
from speculative_tool_executor.workload.tasks import Task, generate_tasks, training_trajectories
from speculative_tool_executor.workload.tools import (
    DEFAULT_TOOL_PROFILE,
    MODEL_TURN_PROFILE,
    TOOL_PROFILES,
    MessageSink,
    build_registry,
)

__all__ = [
    "Arm",
    "Comparison",
    "Scenario",
    "ScenarioResult",
    "ablation_scenarios",
    "compare",
    "run_ab",
    "run_scenario",
    "sweep_ratios",
]

PredictorName = Literal["markov", "none"]


@dataclass(frozen=True, slots=True)
class Scenario:
    """A complete, reproducible description of one run."""

    name: str
    speculate: bool = True
    predictor: PredictorName = "markov"
    order: int = 2
    top_k: int = 2
    canonicalize: bool = True
    latency_ratio: float | None = None
    contention: float = 0.0
    read_bias: float = 0.0
    max_in_flight: int = 4
    min_probability: float = 0.05
    latency_value_usd_per_second: float = 0.01
    speculative_budget_usd: float = 5.0
    max_waste_ratio: float = 10.0
    seed: int = 7

    def policy(self) -> SpeculationPolicy:
        return SpeculationPolicy(
            enabled=self.speculate,
            top_k=self.top_k,
            max_in_flight=self.max_in_flight,
            min_probability=self.min_probability,
            latency_value_usd_per_second=self.latency_value_usd_per_second,
            speculative_budget_usd=self.speculative_budget_usd,
            max_waste_ratio=self.max_waste_ratio,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "speculate": self.speculate,
            "predictor": self.predictor,
            "order": self.order,
            "top_k": self.top_k,
            "canonicalize": self.canonicalize,
            "latency_ratio": self.latency_ratio,
            "contention": self.contention,
            "read_bias": self.read_bias,
        }


def build_latency(scenario: Scenario) -> LatencyModel:
    """The latency model for a scenario, rescaled when the sweep asks for a different ratio."""
    model = LatencyModel(
        tools=TOOL_PROFILES,
        default_tool=DEFAULT_TOOL_PROFILE,
        model_turn=MODEL_TURN_PROFILE,
        seed=scenario.seed,
    )
    return model if scenario.latency_ratio is None else model.with_ratio(scenario.latency_ratio)


def build_predictor(scenario: Scenario, *, corpus: Corpus, training_count: int = 120) -> Predictor:
    """Train the scenario's predictor on trajectories from tasks it will never be evaluated on."""
    if scenario.predictor == "none" or not scenario.speculate:
        return NullPredictor()
    predictor = MarkovPredictor(order=scenario.order)
    # The predictor is trained on the same workload distribution it will meet, which is the
    # realistic case; what it must never see is the evaluation tasks themselves.
    predictor.train(
        training_trajectories(count=training_count, corpus=corpus, read_bias=scenario.read_bias)
    )
    return predictor


@dataclass(slots=True)
class Arm:
    """One instantiated scenario: everything it owns, so nothing leaks between runs."""

    scenario: Scenario
    corpus: Corpus
    store: VersionedStore
    clock: VirtualClock
    executor: SpeculativeToolExecutor
    agent: SimulatedAgent
    sink: MessageSink
    approvals: ApprovalStore
    controller: SpeculationController

    def close(self) -> None:
        self.store.close()


def _contention_injector(scenario: Scenario, store: VersionedStore, task: Task) -> Any:
    """A deterministic background writer, or None when the scenario has no contention."""
    if scenario.contention <= 0:
        return None
    rng = random.Random(f"{scenario.seed}:{task.task_id}")  # noqa: S311  # nosec B311
    counter = {"n": 0}

    def inject(index: int) -> None:
        if index >= len(task.calls):
            return
        key = task.calls[index].arguments.get("key")
        if isinstance(key, str) and rng.random() < scenario.contention:
            counter["n"] += 1
            store.put(key, f"written-by-another-party-{counter['n']}")

    return inject


def build_arm(
    scenario: Scenario,
    *,
    corpus: Corpus | None = None,
    training_count: int = 120,
) -> Arm:
    """Instantiate a scenario with its own store, clock, registry and executor."""
    workspace = corpus if corpus is not None else build_corpus()
    store = VersionedStore(":memory:", synchronous="OFF")
    registry, sink = build_registry(workspace, canonicalize=scenario.canonicalize)
    clock = VirtualClock()
    approvals = ApprovalStore(ttl_seconds=10_000.0)
    gate = ApprovalGate(approvals, principal="agent")
    controller = SpeculationController(scenario.policy())
    executor = SpeculativeToolExecutor(
        registry=registry,
        store=store,
        clock=clock,
        latency=build_latency(scenario),
        gate=gate,
        controller=controller,
        predictor=build_predictor(scenario, corpus=workspace, training_count=training_count),
        cache=SpeculativeCache(max_entries=64),
        taint=TaintTracker(),
    )
    agent = SimulatedAgent(
        executor=executor,
        registry=registry,
        gate=gate,
        speculate=scenario.speculate,
    )
    return Arm(
        scenario=scenario,
        corpus=workspace,
        store=store,
        clock=clock,
        executor=executor,
        agent=agent,
        sink=sink,
        approvals=approvals,
        controller=controller,
    )


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Aggregated outcome of running one scenario over a task set."""

    scenario: Scenario
    tasks: int
    completed: int
    calls_total: int
    calls_speculative: int
    speculatable_calls: int
    p50_seconds: float
    p95_seconds: float
    mean_seconds: float
    total_seconds: float
    tool_wait_total: float
    model_total: float
    hit_ratio: float
    attempts: int
    hits: int
    misses: int
    cancellations: int
    occ_conflicts: int
    stale_reads: int
    wasted_usd: float
    speculative_spend_usd: float
    baseline_spend_usd: float
    waste_ratio: float
    latency_saved_seconds: float
    messages_sent: int
    external_speculations: int
    store_digest: str
    per_tool_hit_ratio: dict[str, float] = field(default_factory=dict)
    refusals: dict[str, int] = field(default_factory=dict)
    equivalence_keys: tuple[str, ...] = ()
    write_sequences: tuple[tuple[tuple[str, ...], ...], ...] = ()
    elapsed: tuple[float, ...] = ()

    @property
    def coverage(self) -> float:
        """Share of the calls that *could* be speculated which actually were served that way.

        The denominator excludes each task's first call, which nothing has had a chance to
        predict. This is the metric that says how well the speculator is doing at its own job,
        separately from how much of a run's latency tool calls account for at all.
        """
        return self.calls_speculative / self.speculatable_calls if self.speculatable_calls else 0.0

    def summary(self) -> dict[str, Any]:
        """A JSON-friendly row; the tuples used only for equivalence are left out."""
        return {
            "scenario": self.scenario.describe(),
            "tasks": self.tasks,
            "completed": self.completed,
            "calls_total": self.calls_total,
            "calls_speculative": self.calls_speculative,
            "coverage": round(self.coverage, 4),
            "p50_seconds": round(self.p50_seconds, 4),
            "p95_seconds": round(self.p95_seconds, 4),
            "mean_seconds": round(self.mean_seconds, 4),
            "total_seconds": round(self.total_seconds, 3),
            "tool_wait_total": round(self.tool_wait_total, 3),
            "hit_ratio": round(self.hit_ratio, 4),
            "attempts": self.attempts,
            "hits": self.hits,
            "misses": self.misses,
            "cancellations": self.cancellations,
            "occ_conflicts": self.occ_conflicts,
            "wasted_usd": round(self.wasted_usd, 6),
            "speculative_spend_usd": round(self.speculative_spend_usd, 6),
            "baseline_spend_usd": round(self.baseline_spend_usd, 6),
            "waste_ratio": round(self.waste_ratio, 4),
            "latency_saved_seconds": round(self.latency_saved_seconds, 3),
            "messages_sent": self.messages_sent,
            "external_speculations": self.external_speculations,
            "per_tool_hit_ratio": {k: round(v, 4) for k, v in self.per_tool_hit_ratio.items()},
            "refusals": self.refusals,
        }


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


async def run_scenario(
    scenario: Scenario,
    tasks: Sequence[Task],
    *,
    corpus: Corpus | None = None,
    training_count: int = 120,
    timeout_seconds: float = 600.0,
) -> ScenarioResult:
    """Run every task once under ``scenario`` and aggregate what happened."""
    arm = build_arm(scenario, corpus=corpus, training_count=training_count)
    runs: list[TaskRun] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            for task in tasks:
                preapprove_external_calls(task, registry=arm.executor.registry, store=arm.approvals)
                agent = SimulatedAgent(
                    executor=arm.executor,
                    registry=arm.executor.registry,
                    gate=ApprovalGate(arm.approvals, principal="agent"),
                    speculate=scenario.speculate,
                    before_tool_call=_contention_injector(scenario, arm.store, task),
                )
                runs.append(await agent.run(task))
        stats = arm.controller.stats
        elapsed = [run.elapsed_seconds for run in runs]
        return ScenarioResult(
            scenario=scenario,
            tasks=len(runs),
            completed=sum(1 for run in runs if run.completed),
            calls_total=sum(len(run.calls) for run in runs),
            calls_speculative=sum(run.hits for run in runs),
            speculatable_calls=sum(run.speculatable_calls for run in runs),
            p50_seconds=_percentile(elapsed, 50),
            p95_seconds=_percentile(elapsed, 95),
            mean_seconds=float(np.mean(elapsed)) if elapsed else 0.0,
            total_seconds=float(sum(elapsed)),
            tool_wait_total=float(sum(run.tool_wait_seconds for run in runs)),
            model_total=float(sum(run.model_seconds for run in runs)),
            hit_ratio=stats.hit_ratio,
            attempts=stats.attempts,
            hits=stats.hits,
            misses=stats.misses,
            cancellations=stats.cancellations,
            occ_conflicts=stats.occ_conflicts,
            stale_reads=stats.stale_reads,
            wasted_usd=stats.wasted_usd,
            speculative_spend_usd=stats.speculative_spend_usd,
            baseline_spend_usd=stats.baseline_spend_usd,
            waste_ratio=stats.waste_ratio,
            latency_saved_seconds=stats.latency_saved_seconds,
            messages_sent=arm.sink.count,
            external_speculations=arm.controller.tool_stats("send_report").attempts,
            store_digest=arm.store.digest(),
            per_tool_hit_ratio=arm.controller.per_tool_hit_ratio(),
            refusals=dict(sorted(stats.refusals.items())),
            equivalence_keys=tuple(run.equivalence_key for run in runs),
            write_sequences=tuple(run.write_sequence for run in runs),
            elapsed=tuple(elapsed),
        )
    finally:
        arm.close()


@dataclass(frozen=True, slots=True)
class Comparison:
    """Speculative versus non-speculative, on the same tasks with the same latency draws."""

    baseline: ScenarioResult
    speculative: ScenarioResult

    @staticmethod
    def _reduction(before: float, after: float) -> float:
        return 0.0 if before <= 0 else (before - after) / before

    @property
    def p50_reduction(self) -> float:
        return self._reduction(self.baseline.p50_seconds, self.speculative.p50_seconds)

    @property
    def p95_reduction(self) -> float:
        return self._reduction(self.baseline.p95_seconds, self.speculative.p95_seconds)

    @property
    def mean_reduction(self) -> float:
        return self._reduction(self.baseline.mean_seconds, self.speculative.mean_seconds)

    @property
    def tool_wait_reduction(self) -> float:
        return self._reduction(self.baseline.tool_wait_total, self.speculative.tool_wait_total)

    @property
    def waste_ratio(self) -> float:
        """Speculative waste as a share of what the same work costs without speculation."""
        spend = self.baseline.baseline_spend_usd
        return self.speculative.wasted_usd / spend if spend > 0 else 0.0

    @property
    def outputs_equivalent(self) -> bool:
        return self.baseline.equivalence_keys == self.speculative.equivalence_keys

    @property
    def writes_equivalent(self) -> bool:
        return self.baseline.write_sequences == self.speculative.write_sequences

    @property
    def store_identical(self) -> bool:
        return self.baseline.store_digest == self.speculative.store_digest

    @property
    def equivalent(self) -> bool:
        """The safety claim: same answers, same writes in the same order, same bytes on disk."""
        return self.outputs_equivalent and self.writes_equivalent and self.store_identical

    @property
    def divergent_tasks(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, (left, right) in enumerate(
                zip(
                    self.baseline.equivalence_keys,
                    self.speculative.equivalence_keys,
                    strict=False,
                )
            )
            if left != right
        )

    def summary(self) -> dict[str, Any]:
        return {
            "tasks": self.baseline.tasks,
            "p50_baseline_seconds": round(self.baseline.p50_seconds, 4),
            "p50_speculative_seconds": round(self.speculative.p50_seconds, 4),
            "p50_reduction": round(self.p50_reduction, 4),
            "p95_baseline_seconds": round(self.baseline.p95_seconds, 4),
            "p95_speculative_seconds": round(self.speculative.p95_seconds, 4),
            "p95_reduction": round(self.p95_reduction, 4),
            "mean_reduction": round(self.mean_reduction, 4),
            "tool_wait_reduction": round(self.tool_wait_reduction, 4),
            "hit_ratio": round(self.speculative.hit_ratio, 4),
            "coverage": round(self.speculative.coverage, 4),
            "waste_ratio": round(self.waste_ratio, 4),
            "wasted_usd": round(self.speculative.wasted_usd, 6),
            "occ_conflicts": self.speculative.occ_conflicts,
            "cancellations": self.speculative.cancellations,
            "equivalent": self.equivalent,
            "outputs_equivalent": self.outputs_equivalent,
            "writes_equivalent": self.writes_equivalent,
            "store_identical": self.store_identical,
            "external_speculations": self.speculative.external_speculations,
            "messages_sent_baseline": self.baseline.messages_sent,
            "messages_sent_speculative": self.speculative.messages_sent,
        }


def compare(baseline: ScenarioResult, speculative: ScenarioResult) -> Comparison:
    return Comparison(baseline=baseline, speculative=speculative)


async def run_ab(
    tasks: Sequence[Task],
    *,
    corpus: Corpus | None = None,
    top_k: int = 2,
    order: int = 2,
    latency_ratio: float | None = None,
    contention: float = 0.0,
    read_bias: float = 0.0,
    seed: int = 7,
    training_count: int = 120,
) -> Comparison:
    """Run the same tasks with speculation off and on, and compare them."""
    workspace = corpus if corpus is not None else build_corpus()
    shared = {
        "latency_ratio": latency_ratio,
        "contention": contention,
        "read_bias": read_bias,
        "seed": seed,
        "top_k": top_k,
        "order": order,
    }
    baseline = await run_scenario(
        Scenario(name="baseline", speculate=False, predictor="none", **shared),  # type: ignore[arg-type]
        tasks,
        corpus=workspace,
        training_count=training_count,
    )
    speculative = await run_scenario(
        Scenario(name="speculative", speculate=True, predictor="markov", **shared),  # type: ignore[arg-type]
        tasks,
        corpus=workspace,
        training_count=training_count,
    )
    return compare(baseline, speculative)


async def sweep_ratios(
    tasks: Sequence[Task],
    ratios: Sequence[float],
    *,
    corpus: Corpus | None = None,
    training_count: int = 120,
) -> list[dict[str, Any]]:
    """How the gain depends on the model-to-tool latency ratio. The honest caveat, quantified."""
    workspace = corpus if corpus is not None else build_corpus()
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        comparison = await run_ab(
            tasks, corpus=workspace, latency_ratio=ratio, training_count=training_count
        )
        rows.append({"ratio": ratio, **comparison.summary()})
    return rows


def ablation_scenarios() -> tuple[Scenario, ...]:
    """The ablations: predictor, top-k and the controller's probability threshold.

    Canonicalization is deliberately *not* ablated here. The generated workload emits arguments
    that are already in canonical form, so switching the rules off changes nothing and the row
    would be a measurement of the generator rather than of the design. What canonicalization
    buys is shown where it can actually be established -- as an invariant, in
    ``tests/unit/test_canonicalize.py`` -- rather than as a benchmark row that reads as evidence
    and is not.
    """
    return (
        Scenario(name="no-speculation", speculate=False, predictor="none"),
        Scenario(name="markov-order1-k2", order=1, top_k=2),
        Scenario(name="markov-order2-k1", order=2, top_k=1),
        Scenario(name="markov-order2-k2", order=2, top_k=2),
        Scenario(name="markov-order2-k3", order=2, top_k=3),
        Scenario(name="threshold-0.00", order=2, top_k=2, min_probability=0.0),
        Scenario(name="threshold-0.35", order=2, top_k=2, min_probability=0.35),
    )


async def run_ablations(
    tasks: Sequence[Task],
    *,
    corpus: Corpus | None = None,
    training_count: int = 120,
) -> list[dict[str, Any]]:
    """Run each ablation and report it against the no-speculation arm."""
    workspace = corpus if corpus is not None else build_corpus()
    results = [
        await run_scenario(scenario, tasks, corpus=workspace, training_count=training_count)
        for scenario in ablation_scenarios()
    ]
    baseline = results[0]
    rows: list[dict[str, Any]] = []
    for result in results:
        comparison = compare(baseline, result)
        rows.append(
            {
                "name": result.scenario.name,
                "p50_seconds": round(result.p50_seconds, 4),
                "p50_reduction": round(comparison.p50_reduction, 4),
                "hit_ratio": round(result.hit_ratio, 4),
                "attempts": result.attempts,
                "waste_ratio": round(comparison.waste_ratio, 4),
                "equivalent": comparison.equivalent,
            }
        )
    return rows


def default_tasks(count: int = 300, seed: int = 20260101) -> tuple[Task, ...]:
    return generate_tasks(count=count, seed=seed)
