"""The virtual clock: a discrete-event simulation running inside a real asyncio loop.

The reason this matters is that the code under test is the production code -- real
``TaskGroup``s, real cancellation, real ordering. Only time is simulated. So the clock has to
guarantee one thing above all: **virtual time never advances while any participant could still do
work.** If it did, a speculation would appear to finish before it had run, and every latency number
the benchmark produces would be a fiction.
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from speculative_tool_executor.spec.clock import (
    RealClock,
    VirtualClock,
    run_with_timeout,
)

pytestmark = pytest.mark.unit


async def test_a_sleep_costs_no_wall_clock_time() -> None:
    clock = VirtualClock()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with clock.participant():
        await clock.sleep(3600.0)
    assert clock.now == pytest.approx(3600.0)
    assert loop.time() - started < 1.0


async def test_time_does_not_advance_while_a_participant_is_still_running() -> None:
    """The invariant. The sleeper must not be woken until the worker has parked or finished."""
    clock = VirtualClock()
    observed: list[float] = []

    async def worker() -> None:
        with clock.participant():
            for _ in range(5):
                await asyncio.sleep(0)  # yield without consuming virtual time
                observed.append(clock.now)

    async def sleeper() -> None:
        with clock.participant():
            await clock.sleep(10.0)
            observed.append(clock.now)

    async with asyncio.TaskGroup() as group:
        group.create_task(worker())
        group.create_task(sleeper())

    # Everything the worker saw happened at time 0; only then did the clock jump.
    assert observed[:-1] == [0.0] * 5
    assert observed[-1] == pytest.approx(10.0)


async def test_a_task_is_registered_from_creation_not_from_first_step() -> None:
    """Without ``track``, a spawned-but-not-started task would not hold time back."""
    clock = VirtualClock()
    order: list[str] = []

    async def spawned() -> None:
        with clock.participant():
            order.append(f"spawned-start@{clock.now}")
            await clock.sleep(1.0)
            order.append(f"spawned-end@{clock.now}")

    async def main() -> None:
        with clock.participant():
            async with asyncio.TaskGroup() as group:
                task = group.create_task(spawned())
                clock.track(task)
                await clock.sleep(5.0)
                order.append(f"main-end@{clock.now}")

    await main()
    assert order == ["spawned-start@0.0", "spawned-end@1.0", "main-end@5.0"]


async def test_parking_lets_time_advance_for_the_task_being_waited_on() -> None:
    """A task blocked on another participant must not hold the clock back."""
    clock = VirtualClock()
    done = asyncio.Event()
    seen: list[float] = []

    async def producer() -> None:
        with clock.participant():
            await clock.sleep(4.0)
            done.set()

    async def consumer() -> None:
        with clock.participant(), clock.parked():
            await done.wait()
            seen.append(clock.now)

    async with asyncio.TaskGroup() as group:
        task = group.create_task(producer())
        clock.track(task)
        group.create_task(consumer())

    assert seen == [pytest.approx(4.0)]


async def test_cancelling_a_sleep_returns_the_task_count_to_baseline() -> None:
    """A cancelled speculation must leave nothing running and must not stall the clock."""
    clock = VirtualClock()
    baseline = len(asyncio.all_tasks())
    finished: list[str] = []

    async def cancellable() -> None:
        with clock.participant():
            try:
                await clock.sleep(100.0)
            except asyncio.CancelledError:
                finished.append("cancelled")
                raise

    async def survivor() -> None:
        with clock.participant():
            await clock.sleep(1.0)
            finished.append("survived")

    async with asyncio.TaskGroup() as group:
        doomed = group.create_task(cancellable())
        clock.track(doomed)
        keeper = group.create_task(survivor())
        clock.track(keeper)
        await asyncio.sleep(0)
        doomed.cancel()

    # Both happened; the order in which a cancellation is delivered relative to an unrelated
    # wake-up is an event-loop detail, not part of the clock's contract.
    assert sorted(finished) == ["cancelled", "survived"]
    assert clock.now == pytest.approx(1.0)
    assert len(asyncio.all_tasks()) == baseline
    assert clock.sleeping == 0
    assert clock.registered == 0


async def test_a_cancelled_sleeper_does_not_keep_the_simulation_from_moving() -> None:
    clock = VirtualClock()

    async def doomed() -> None:
        with clock.participant():
            await clock.sleep(1_000.0)

    async def worker() -> None:
        with clock.participant():
            await clock.sleep(2.0)

    async with asyncio.TaskGroup() as group:
        first = group.create_task(doomed())
        clock.track(first)
        second = group.create_task(worker())
        clock.track(second)
        await asyncio.sleep(0)
        first.cancel()

    assert clock.now == pytest.approx(2.0)


@settings(max_examples=40, deadline=None)
@given(durations=st.lists(st.floats(min_value=0.01, max_value=50.0), min_size=1, max_size=6))
async def test_sleepers_wake_in_deadline_order(durations: list[float]) -> None:
    clock = VirtualClock()
    woken: list[float] = []

    async def sleeper(duration: float) -> None:
        with clock.participant():
            await clock.sleep(duration)
            woken.append(clock.now)

    async with asyncio.TaskGroup() as group:
        for duration in durations:
            task = group.create_task(sleeper(duration))
            clock.track(task)

    assert woken == sorted(woken)
    assert clock.now == pytest.approx(max(durations))
    assert woken[-1] == pytest.approx(max(durations))


async def test_zero_and_negative_durations() -> None:
    clock = VirtualClock()
    with clock.participant():
        await clock.sleep(0.0)
        assert clock.now == 0.0
        with pytest.raises(ValueError, match="non-negative"):
            await clock.sleep(-1.0)


async def test_advances_counts_only_real_jumps() -> None:
    clock = VirtualClock()
    with clock.participant():
        await clock.sleep(0.0)
        assert clock.advances == 0
        await clock.sleep(1.0)
        assert clock.advances == 1


async def test_using_the_clock_outside_a_task_is_refused() -> None:
    """A loop callback runs with a loop but no current task, which the clock must reject."""
    clock = VirtualClock()
    captured: list[str] = []

    def in_a_bare_callback() -> None:
        try:
            clock.participant().__enter__()
        except RuntimeError as exc:
            captured.append(str(exc))

    asyncio.get_running_loop().call_soon(in_a_bare_callback)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert captured
    assert "inside a running task" in captured[0]


async def test_real_clock_moves_forward_and_participation_is_a_no_op() -> None:
    clock = RealClock()
    with clock.participant(), clock.parked():
        before = clock.now
        await clock.sleep(0.01)
    assert clock.now >= before
    clock.track(asyncio.current_task())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        await clock.sleep(-1.0)


async def test_run_with_timeout_bounds_a_stalled_simulation() -> None:
    """A simulation that stops advancing must fail rather than hang forever."""

    async def never() -> None:
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await run_with_timeout(never(), seconds=0.05)


async def test_run_with_timeout_returns_the_value() -> None:
    async def answer() -> int:
        return 42

    assert await run_with_timeout(answer(), seconds=5.0) == 42
