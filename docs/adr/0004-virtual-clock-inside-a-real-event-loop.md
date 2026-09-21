# 4. Simulate time inside a real asyncio event loop, not around it

- Status: accepted
- Date: 2026-09-21

## Context

The claim this project makes is about latency: that running a predicted call during the model's
turn removes wait from the critical path. Measuring that needs a workload long enough to be
meaningful — 300 multi-step tasks, a sweep over five latency ratios, three predictability points
and seven ablations. At real speed that is hours of sleeping, which rules out running it in CI, in
tests, or in a demo.

But the interesting behaviour lives in the concurrency, not the arithmetic: a branch that is still
in flight when the model answers, a cancellation that arrives mid-sleep, a task group unwinding at
the end of a run. A model of that behaviour would be measuring the model.

## Decision

`VirtualClock` runs a discrete-event simulation *inside* an ordinary asyncio event loop. The code
under test is the production code — real `TaskGroup`s, real `Task.cancel()`, real scheduling
order. Only the passage of time is simulated.

The invariant that makes it sound: **virtual time advances only when every task that could still
do work is waiting for it.** Tasks register as participants; a task blocked on another participant
*parks* (deregisters) because it cannot progress on its own and must not hold time back. When the
number of sleeping participants reaches the number registered, the clock jumps straight to the
earliest deadline and wakes everything due at that instant.

Three details, each of which was a bug before it was a rule:

- A task is registered at **creation**, via `clock.track(task)`, not at its first step. Otherwise
  spawning a speculation and then sleeping would let the clock jump past the speculation's start.
- A sleeper is marked awake by the **waker**, before being rescheduled, so a task entering `sleep`
  while another has been woken but not yet resumed cannot trigger a second advance.
- A task with a **pending but undelivered cancellation** is retired from the sleeping set before
  any advance. `Task.cancel()` is asynchronous; in the window before the `CancelledError` is
  raised, the waiter still looks like an ordinary sleeper, and the clock would otherwise jump to
  the deadline of a sleep that is already doomed — charging the simulation time nothing waited
  for. Cancelling mispredicted speculations is the common case here, so this window is hit
  constantly rather than rarely.

Latency draws are deterministic in the call (`seed:tool:call_hash`), not in execution order, so
the speculative and non-speculative arms see identical latencies for identical work.

## Consequences

- The full benchmark — 300 tasks, all sweeps and ablations — runs in about 64 seconds of wall
  clock on a laptop CPU, simulating far more than that. It is a CI gate rather than a nightly job.
- Cancellation, task-group unwinding and ordering are exercised for real, so the tests that assert
  "no speculation outlives its run" and "the task count returns to baseline" mean what they say.
- A/B comparisons are exact: the same call draws the same latency in both arms, so any difference
  is speculation.
- The failure mode of a bug in the advance condition is a **hang**, not a wrong number, because
  every task ends up waiting on an event nobody will set. `run_with_timeout` bounds the simulation
  in wall-clock time so this fails loudly instead of hanging forever.
- Any code that sleeps must sleep on the injected clock. A stray `asyncio.sleep` in a participant
  would deadlock the advance condition rather than silently skewing results — a good failure mode,
  but one contributors have to know about.
- Every latency number in the repository is a simulation output. README, `BENCHMARKS.md` and
  `CLAIMS.md` say so wherever such a number appears, and the benchmark reports its own wall-clock
  runtime separately so "simulated seconds" and "seconds this took" are never confused.

## Alternatives considered

**Real sleeps with tiny durations.** Scales everything down, but also scales down the thing being
measured relative to scheduler noise, and a 1 ms "tool call" on a loaded CI runner is mostly
jitter. It would also still take minutes for the full sweep.

**`asyncio` loop-time mocking (freezegun-style, or a custom loop).** Replacing the loop's clock
works for simple cases but interacts badly with `TaskGroup` and with anything that takes a real
lock; and a custom event loop would mean the tests no longer exercise the loop production uses.

**An analytical model instead of a simulation.** A closed-form expression for expected savings
would be faster and would not need any of this machinery — but it could not show that the
*implementation* preserves equivalence, cancels correctly, or handles a branch that is still in
flight when the model answers. Those are the properties worth having.
