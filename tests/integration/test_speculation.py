"""End-to-end speculation: the hit, the miss, the conflict, and the equivalence that ties them.

These tests drive the real executor, the real speculator and the real ``TaskGroup`` on a virtual
clock. What they assert is the contract the README states: speculation changes *when* work
happens and never *what* the run did.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from speculative_tool_executor.agentic.errors import ToolNotAllowedError
from speculative_tool_executor.agentic.providers.base import ToolUseBlock
from speculative_tool_executor.sim.agent import preapprove_external_calls
from speculative_tool_executor.sim.experiment import (
    Scenario,
    build_arm,
    compare,
    run_ab,
    run_ablations,
    run_scenario,
)
from speculative_tool_executor.spec.executor import ExecutionMode
from speculative_tool_executor.workload.corpus import build_corpus
from speculative_tool_executor.workload.tasks import generate_tasks

pytestmark = pytest.mark.integration

CORPUS = build_corpus()


def tasks(count: int = 24, *, read_bias: float = 1.0, seed: int = 20260101):  # type: ignore[no-untyped-def]
    return generate_tasks(count=count, corpus=CORPUS, seed=seed, read_bias=read_bias)


# -- the headline contract ----------------------------------------------------------------------


async def test_speculation_removes_tool_wait_without_changing_what_the_run_did() -> None:
    comparison = await run_ab(tasks(40), corpus=CORPUS, read_bias=1.0, training_count=80)

    assert comparison.outputs_equivalent, "tool outputs diverged"
    assert comparison.writes_equivalent, "committed writes diverged"
    assert comparison.store_identical, "store bytes diverged"
    assert comparison.divergent_tasks == ()

    assert comparison.speculative.tool_wait_total < comparison.baseline.tool_wait_total
    assert comparison.tool_wait_reduction > 0.15
    assert comparison.speculative.hits > 0
    assert comparison.baseline.hits == 0


async def test_model_time_is_untouched_by_speculation() -> None:
    """Speculation can only remove tool wait; the model's own turns are irreducible.

    Pinning this keeps the README's framing honest: an end-to-end number is bounded by the share
    of the run that tool calls account for at all.
    """
    comparison = await run_ab(tasks(24), corpus=CORPUS, read_bias=1.0, training_count=60)
    assert comparison.speculative.model_total == pytest.approx(
        comparison.baseline.model_total, rel=1e-9
    )


async def test_running_the_same_scenario_twice_produces_identical_results() -> None:
    scenario = Scenario(name="repeat", read_bias=1.0)
    first = await run_scenario(scenario, tasks(16), corpus=CORPUS, training_count=60)
    second = await run_scenario(scenario, tasks(16), corpus=CORPUS, training_count=60)
    assert first.equivalence_keys == second.equivalence_keys
    assert first.elapsed == second.elapsed
    assert first.store_digest == second.store_digest
    assert first.hit_ratio == second.hit_ratio


# -- the individual paths -----------------------------------------------------------------------


async def test_a_matching_branch_is_committed_and_credited_with_the_latency_it_absorbed() -> None:
    arm = build_arm(Scenario(name="hit", read_bias=1.0), corpus=CORPUS, training_count=80)
    try:
        task = next(t for t in tasks(60) if t.shape == "audit_module")
        run = await arm.agent.run(task)
        served = [
            record for record in run.calls if record.mode == ExecutionMode.SPECULATIVE_COMMITTED
        ]
        assert served, "expected at least one speculative hit on a three-step task"
        for record in served:
            assert record.waited_seconds == pytest.approx(0.0, abs=1e-9)
            assert record.latency_saved_seconds > 0
        assert run.calls[0].mode == ExecutionMode.DIRECT, "the first call cannot be predicted"
    finally:
        arm.close()


async def test_an_unmatched_branch_is_cancelled_and_billed_as_waste() -> None:
    """At a 1:1 latency ratio the speculation is still running when the model answers."""
    comparison = await run_ab(
        tasks(40), corpus=CORPUS, read_bias=1.0, latency_ratio=1.0, training_count=80
    )
    assert comparison.speculative.cancellations > 0
    assert comparison.speculative.wasted_usd > 0
    assert comparison.equivalent


async def test_a_stale_speculative_read_is_discarded_and_re_executed() -> None:
    """Another writer moves the version a branch read; OCC must refuse to commit it."""
    contended = await run_ab(
        tasks(40), corpus=CORPUS, read_bias=1.0, contention=1.0, training_count=80
    )
    assert contended.speculative.occ_conflicts > 0
    assert contended.store_identical
    assert contended.outputs_equivalent
    assert contended.writes_equivalent


async def test_contention_lowers_the_hit_rate_but_never_the_correctness() -> None:
    clean = await run_ab(tasks(40), corpus=CORPUS, read_bias=1.0, training_count=80)
    contended = await run_ab(
        tasks(40), corpus=CORPUS, read_bias=1.0, contention=1.0, training_count=80
    )
    assert contended.speculative.hit_ratio < clean.speculative.hit_ratio
    assert clean.equivalent
    assert contended.equivalent


async def test_leaving_the_session_settles_every_outstanding_branch() -> None:
    """Structured concurrency: no speculation escapes the run that started it."""
    arm = build_arm(Scenario(name="settle", read_bias=1.0), corpus=CORPUS, training_count=80)
    try:
        before = len(asyncio.all_tasks())
        for task in tasks(6):
            preapprove_external_calls(task, registry=arm.executor.registry, store=arm.approvals)
            await arm.agent.run(task)
        assert len(asyncio.all_tasks()) == before
        assert len(arm.executor.cache) == 0
        assert arm.controller.stats.attempts == (
            arm.controller.stats.hits + arm.controller.stats.misses
        )
    finally:
        arm.close()


async def test_every_launched_speculation_is_billed_exactly_once() -> None:
    result = await run_scenario(
        Scenario(name="ledger", read_bias=1.0), tasks(30), corpus=CORPUS, training_count=80
    )
    assert result.attempts == result.hits + result.misses
    assert result.wasted_usd <= result.speculative_spend_usd + 1e-12


# -- the knobs ------------------------------------------------------------------------------------


async def test_disabling_speculation_reproduces_the_baseline_exactly() -> None:
    """The kill switch must be a true no-op, not merely "less speculation"."""
    off = await run_scenario(
        Scenario(name="off", speculate=False, predictor="none", read_bias=1.0),
        tasks(20),
        corpus=CORPUS,
        training_count=60,
    )
    null = await run_scenario(
        Scenario(name="null-predictor", predictor="none", read_bias=1.0),
        tasks(20),
        corpus=CORPUS,
        training_count=60,
    )
    assert off.attempts == null.attempts == 0
    assert off.elapsed == null.elapsed
    assert off.store_digest == null.store_digest


async def test_a_per_tool_override_stops_only_that_tool() -> None:
    arm = build_arm(Scenario(name="scoped", read_bias=1.0), corpus=CORPUS, training_count=80)
    try:
        arm.controller.policy = arm.controller.policy.with_overrides(
            disabled_tools=frozenset({"save_note"})
        )
        for task in tasks(20):
            await arm.agent.run(task)
        assert arm.controller.tool_stats("save_note").attempts == 0
        assert arm.controller.tool_stats("read_file").attempts > 0
    finally:
        arm.close()


async def test_higher_top_k_speculates_more_and_wastes_more() -> None:
    sample = tasks(30)
    narrow = await run_scenario(
        Scenario(name="k1", top_k=1, read_bias=1.0), sample, corpus=CORPUS, training_count=80
    )
    wide = await run_scenario(
        Scenario(name="k3", top_k=3, read_bias=1.0), sample, corpus=CORPUS, training_count=80
    )
    assert wide.attempts > narrow.attempts
    assert wide.wasted_usd > narrow.wasted_usd
    assert wide.calls_speculative >= narrow.calls_speculative


async def test_without_canonicalization_nothing_breaks_but_less_is_reused() -> None:
    """The ablation that says what the declared equivalences are worth."""
    sample = tasks(24)
    rows = {row["name"]: row for row in await run_ablations(sample, corpus=CORPUS)}
    assert all(row["equivalent"] for row in rows.values())
    assert rows["markov-order2-k2"]["attempts"] > 0
    assert rows["no-speculation"]["attempts"] == 0


async def test_every_ablation_stays_equivalent_to_the_non_speculative_run() -> None:
    rows = await run_ablations(tasks(20), corpus=CORPUS)
    assert next(row["name"] for row in rows) == "no-speculation"
    for row in rows:
        assert row["equivalent"] is True, f"{row['name']} diverged"


async def test_the_latency_ratio_governs_how_much_end_to_end_time_is_recoverable() -> None:
    """Lower model-to-tool ratio means tools are a bigger share, so more is recoverable."""
    sample = tasks(40)
    fast_model = await run_ab(
        sample, corpus=CORPUS, read_bias=1.0, latency_ratio=1.0, training_count=80
    )
    slow_model = await run_ab(
        sample, corpus=CORPUS, read_bias=1.0, latency_ratio=10.0, training_count=80
    )
    assert fast_model.p50_reduction > slow_model.p50_reduction
    # The mechanism itself is unchanged; only its share of the total moves.
    assert fast_model.speculative.coverage == pytest.approx(
        slow_model.speculative.coverage, abs=0.15
    )


# -- the executor's own guards --------------------------------------------------------------------


async def test_an_unknown_tool_is_refused_by_the_registry_not_executed() -> None:
    arm = build_arm(Scenario(name="guard", read_bias=1.0), corpus=CORPUS, training_count=20)
    try:
        async with arm.executor.session() as session:
            with pytest.raises(ToolNotAllowedError):
                await arm.executor.execute(
                    ToolUseBlock(id="x", name="rm_rf", arguments={}), session=session
                )
    finally:
        arm.close()


async def test_a_failing_tool_is_data_not_a_crash() -> None:
    arm = build_arm(Scenario(name="fail", read_bias=1.0), corpus=CORPUS, training_count=20)
    try:
        async with arm.executor.session() as session:
            outcome = await arm.executor.execute(
                ToolUseBlock(id="x", name="read_file", arguments={"path": "nope/missing.py"}),
                session=session,
            )
        assert outcome.ok is False
        assert outcome.error == "WorkspaceError"
        assert outcome.mode == ExecutionMode.DIRECT
    finally:
        arm.close()


async def test_compare_reports_divergence_when_two_arms_disagree() -> None:
    """The equivalence check must be able to fail, or passing it would prove nothing."""
    sample = tasks(8)
    left = await run_scenario(
        Scenario(name="a", read_bias=1.0), sample, corpus=CORPUS, training_count=40
    )
    right = await run_scenario(
        Scenario(name="b", read_bias=1.0), sample, corpus=CORPUS, training_count=40
    )
    assert compare(left, right).equivalent is True

    tampered = replace(right, equivalence_keys=("divergent", *right.equivalence_keys[1:]))
    comparison = compare(left, tampered)
    assert comparison.equivalent is False
    assert comparison.divergent_tasks == (0,)
