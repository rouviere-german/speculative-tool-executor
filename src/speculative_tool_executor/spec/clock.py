"""Clocks. The simulator needs latency without spending it, so time is injected, never read.

``VirtualClock`` runs a discrete-event simulation *inside* a normal asyncio event loop, so the
code under test is the production code: real ``TaskGroup``s, real cancellation, real ordering.
Only the passage of time is simulated.

The invariant that makes this work is simple: **virtual time advances only when every task that
could still do work is waiting for it.** Each task that participates registers itself; a task
that blocks on another participant (awaiting a speculative result, for example) *parks*, because
it cannot make progress on its own and must not hold time back. When the number of sleeping
participants reaches the number of registered ones, the clock jumps straight to the earliest
deadline and wakes everything due at that instant. Nothing sleeps in wall-clock time, so a
simulated hour of agent latency costs milliseconds.

Two details matter for correctness:

- a sleeper is marked awake by the *waker*, before it is rescheduled, so a task that enters
  ``sleep`` while another has been woken but not yet resumed cannot trigger a second advance;
- cancelling a sleep marks its waiter dead and re-checks the advance condition, so a cancelled
  speculation never keeps the simulation from moving.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import time
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "RealClock", "VirtualClock", "run_with_timeout"]

_SLEEPING = "sleeping"
_WOKEN = "woken"
_CANCELLED = "cancelled"


@runtime_checkable
class Clock(Protocol):
    """Everything the runtime needs from time. Implementations must be usable from asyncio."""

    @property
    def now(self) -> float:
        """Seconds since this clock's origin."""

    async def sleep(self, duration: float) -> None:
        """Wait ``duration`` seconds of this clock's time."""

    def participant(self) -> contextlib.AbstractContextManager[None]:
        """Register the current task as able to consume time."""

    def parked(self) -> contextlib.AbstractContextManager[None]:
        """Temporarily deregister while blocked on another participant."""

    def track(self, task: asyncio.Task[object]) -> None:
        """Register a task from the moment it is created, before its first step runs."""


class RealClock:
    """Wall-clock implementation for production use. Participation is a no-op."""

    def __init__(self) -> None:
        self._origin = time.monotonic()

    @property
    def now(self) -> float:
        return time.monotonic() - self._origin

    async def sleep(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must be non-negative")
        await asyncio.sleep(duration)

    def participant(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def parked(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def track(self, task: asyncio.Task[object]) -> None:
        """Nothing to track: wall-clock time advances on its own."""


@dataclass(order=True, slots=True)
class _Waiter:
    deadline: float
    seq: int
    event: asyncio.Event = field(compare=False)
    state: str = field(default=_SLEEPING, compare=False)
    task: asyncio.Task[object] | None = field(default=None, compare=False)


class VirtualClock:
    """A deterministic clock whose time advances only when all participants are asleep."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start
        self._origin = start
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()
        self._registered: set[int] = set()
        self._sleeping = 0
        self._advances = 0

    @property
    def now(self) -> float:
        return self._now

    @property
    def elapsed(self) -> float:
        return self._now - self._origin

    @property
    def advances(self) -> int:
        """How many times time jumped forward; a cheap health signal for the simulation."""
        return self._advances

    @property
    def registered(self) -> int:
        return len(self._registered)

    @property
    def sleeping(self) -> int:
        return self._sleeping

    @staticmethod
    def _task_key() -> int:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("VirtualClock must be used from inside a running task")
        return id(task)

    def track(self, task: asyncio.Task[object]) -> None:
        """Register ``task`` now, so a task created but not yet started still holds time back.

        Without this, spawning a speculation and then sleeping would let the clock jump past the
        speculation's start: the new task is not running yet, so it would not be counted among
        the participants and the sleeper would look like the only one left.
        """
        self._registered.add(id(task))
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[object]) -> None:
        self._registered.discard(id(task))
        self._maybe_advance()

    @contextlib.contextmanager
    def participant(self) -> Iterator[None]:
        key = self._task_key()
        if key in self._registered:
            yield
            return
        self._registered.add(key)
        try:
            yield
        finally:
            self._registered.discard(key)
            self._maybe_advance()

    @contextlib.contextmanager
    def parked(self) -> Iterator[None]:
        """Stop holding time back while awaiting work another participant must finish."""
        key = self._task_key()
        was_registered = key in self._registered
        if was_registered:
            self._registered.discard(key)
            self._maybe_advance()
        try:
            yield
        finally:
            if was_registered:
                self._registered.add(key)

    async def sleep(self, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must be non-negative")
        if duration == 0:
            await asyncio.sleep(0)
            return
        key = self._task_key()
        implicit = key not in self._registered
        if implicit:
            self._registered.add(key)
        waiter = _Waiter(
            deadline=self._now + duration,
            seq=next(self._seq),
            event=asyncio.Event(),
            task=asyncio.current_task(),
        )
        heapq.heappush(self._waiters, waiter)
        self._sleeping += 1
        try:
            self._maybe_advance()
            await waiter.event.wait()
        except asyncio.CancelledError:
            if waiter.state == _SLEEPING:
                waiter.state = _CANCELLED
                self._sleeping -= 1
            raise
        finally:
            if implicit:
                self._registered.discard(key)
            self._maybe_advance()

    def _retire_cancelled(self) -> None:
        """Retire sleeps whose task has a cancellation pending but not yet delivered.

        ``Task.cancel()`` is asynchronous: the ``CancelledError`` is only raised the next time
        the task runs. In the window between the two the waiter still looks like an ordinary
        sleeper, so if every other participant finishes in that window the clock would jump to
        the deadline of a sleep that is already doomed -- charging the simulation time that
        nothing ever waited for. Cancelling a mispredicted speculation is the common case here,
        so this window is hit constantly rather than rarely.
        """
        for waiter in self._waiters:
            if waiter.state == _SLEEPING and waiter.task is not None and waiter.task.cancelling():
                waiter.state = _CANCELLED
                self._sleeping -= 1

    def _maybe_advance(self) -> None:
        """Jump to the earliest deadline when no participant can make progress without it."""
        self._retire_cancelled()
        if self._sleeping == 0 or self._sleeping < len(self._registered):
            return
        while self._waiters and self._waiters[0].state != _SLEEPING:
            heapq.heappop(self._waiters)
        if not self._waiters:
            return
        deadline = self._waiters[0].deadline
        if deadline > self._now:
            self._now = deadline
            self._advances += 1
        while self._waiters and self._waiters[0].deadline <= self._now:
            waiter = heapq.heappop(self._waiters)
            if waiter.state != _SLEEPING:
                continue
            waiter.state = _WOKEN
            self._sleeping -= 1
            waiter.event.set()


async def run_with_timeout[T](awaitable: Awaitable[T], *, seconds: float = 60.0) -> T:
    """Run ``awaitable`` under a wall-clock timeout.

    A virtual-clock simulation that stops advancing would otherwise hang forever rather than
    fail, because every task is waiting on an event no one will set. The timeout is wall-clock
    on purpose: it bounds the simulation itself, not the time it simulates.
    """
    async with asyncio.timeout(seconds):
        return await awaitable
