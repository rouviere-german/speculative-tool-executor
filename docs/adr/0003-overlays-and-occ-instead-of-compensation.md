# 3. Speculative writes buffer in per-branch overlays, validated by optimistic concurrency control

- Status: accepted
- Date: 2026-09-21

## Context

A speculative write has two ways to be wrong. It can be *unwanted* — the model asked for something
else, so the write must never have happened. And it can be *stale* — the model did ask for it, but
the state the branch computed from has changed since, so applying it would clobber someone else's
update or produce a value derived from a world that no longer exists.

The obvious design is to write eagerly and undo on a miss: run the write against the real store,
and on a mispredict apply a compensating action. This is how a lot of saga-style systems work, and
it is what a first sketch of this project did.

## Decision

Speculative writes never reach the store. A branch owns a `WriteOverlay`: reads go through a
buffer and are recorded with the version they observed, writes accumulate in that buffer, and the
store is untouched until the branch commits.

Commit is a single `BEGIN IMMEDIATE` transaction that first re-checks every version in the read
set and then applies the writes. If any version moved, nothing is applied and the conflicting keys
are named. This is textbook optimistic concurrency control (Kung and Robinson, *On Optimistic
Methods for Concurrency Control*, 1981), with the validation phase doing the work that makes
speculation safe: a branch built on a stale read can never be committed.

Two details that are not incidental:

- **Deletes leave tombstones with fresh versions.** Without them, "created then deleted" would be
  indistinguishable from "never existed", and a branch that read a key as absent would commit over
  a value that had appeared and gone in between.
- **Read-only branches use the same machinery with an empty write set.** A speculated `READ` is
  validated by exactly the same check as a write, rather than by a second, weaker one.

Aborting is therefore not an operation on the store at all. It drops a buffer. There is nothing
to undo, which is why the "an aborted branch leaves the store byte-identical" property is provable
over arbitrary operation sequences rather than argued case by case.

## Consequences

- Correctness under contention is a throughput problem, never a correctness problem. The benchmark
  shows this directly: at 50% write contention, 39 branches were rejected and the hit ratio fell
  from 0.42 to 0.34, while the store stayed byte-identical to the non-speculative run.
- The differential equivalence claim becomes checkable end to end: same outputs, same committed
  writes in the same order, same store digest, on all 300 tasks.
- Every speculatable write tool must route through `current_execution().overlay`. That is a real
  constraint on tool authors, enforced by a semgrep rule that forbids touching the store directly
  from a handler.
- Handlers cannot see uncommitted writes from *other* branches. Within a branch, reads see the
  branch's own buffer, which is what a caller expects.
- A long-running branch under heavy contention will keep conflicting. The controller's waste trip
  is the backstop: sustained misses disable speculation for the rest of the run.

## Alternatives considered

**Write eagerly, compensate on a miss.** Rejected. Compensation is not undo: between the write and
the compensating action, the wrong value is visible to every other reader, and if the process dies
in that window the wrong value is permanent. It also requires every write tool to ship a correct
inverse, which is extra code on the path that must never be wrong.

**Snapshot isolation / MVCC over the whole store.** More powerful and much more machinery. The
read sets here are tiny (one or two keys per call), so full MVCC would buy generality this system
does not need, and it would make "why did this commit fail?" harder to answer than a list of
conflicting keys.

**Locking the keys a speculation will touch.** A speculation is a guess; taking locks on a guess
lets a mispredicted branch block the authoritative path. Optimistic is the right default precisely
because most branches are discarded.

**Per-key version counters instead of one global counter.** Equivalent for validation, but a
single monotonic counter makes a version a point in the store's history, which makes the store
digest and the equivalence comparison trivial to define.
