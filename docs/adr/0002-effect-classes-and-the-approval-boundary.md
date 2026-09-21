# 2. Effect classes decide what may be speculated, and only irreversible reach needs approval

- Status: accepted
- Date: 2026-09-21

## Context

Speculation runs a tool call before anyone asked for it. Whether that is acceptable depends
entirely on what the tool *does*, so the system needs a classification of tool effects that is
declared, checkable, and impossible to omit.

The shared portfolio baseline already had a rule: a tool declaring any side-effecting capability
(`WRITE_DATA`, `NETWORK_EGRESS`, `SEND_MESSAGE`, `EXECUTE_CODE`) could not opt out of human
approval. That rule is right for an ordinary agent, but applied here it forbids speculating any
write at all — every write tool would need a human before each call, and the design would collapse
to "speculate reads only", which is a much smaller and much less interesting claim.

At the same time, the opposite move — letting any side effect run on a guess — is obviously wrong.
A sent message cannot be unsent.

## Decision

Two orthogonal declarations per tool, both mandatory.

**Effect class** (`PURE` < `READ` < `WRITE` < `EXTERNAL`), a lattice ordered by how dangerous it
is to run ahead of the model. `SPECULATABLE_EFFECTS` is `{PURE, READ, WRITE}`; `EXTERNAL` is
outside it and is refused by the speculator, re-checked by the runner at launch, and asserted by
the handler itself. `ToolDefinition` has no default for `effect`, so a tool cannot be constructed
without one, and a semgrep rule catches a `ToolDefinition(...)` call that omits it.

**Capabilities**, constrained by the effect via `EFFECT_CAPABILITIES`, so a tool cannot declare
itself a harmless `READ` while holding `WRITE_DATA`.

We then split the baseline's side-effect set in two:

- `SIDE_EFFECT_CAPABILITIES` still drives the *default*: any of them means approval is required
  unless the tool explicitly says otherwise. A tool author has to say out loud that a side effect
  is safe to run without a human.
- `IRREVERSIBLE_CAPABILITIES` — `NETWORK_EGRESS`, `SEND_MESSAGE`, `EXECUTE_CODE` — cannot opt
  out. `requires_approval=False` combined with any of them, or with `EXTERNAL`, raises.

`WRITE_DATA` is deliberately in the first set and not the second. The justification is not that
writes are unimportant; it is that a write in this system is **structurally recoverable**: it
buffers in a branch overlay, it is validated against the versions it read, and it is applied in
one transaction. Up to the instant it commits, discarding it costs nothing and leaves no trace —
a property proved by a test over randomised operation sequences, not asserted. An effect that has
already reached a third party has no equivalent.

## Consequences

- Write tools are speculatable, which is what makes the repository's central claim non-trivial.
  Roughly half the speculative hits in the benchmark are `save_note` and `bump_index`.
- The approval requirement stays absolute exactly where it has to be, and is enforced in three
  independent places rather than one.
- This is a **deviation from the shared template's rule**, and a reviewer comparing repositories
  will notice. It is recorded here rather than left as a silent edit.
- A deployment that disagrees is not stuck: any tool can set `requires_approval=True`, and the
  effect lattice still confines it to an overlay.
- The residual risk — a write happening on a guess without a human — is listed explicitly in
  `docs/THREAT_MODEL.md` rather than argued away.

## Alternatives considered

**Keep the baseline rule, speculate reads only.** Safe and simple, but it removes the interesting
half of the problem: no write overlays, no OCC, no commit/abort semantics. The project would be a
read cache with a predictor.

**Per-call risk scoring instead of classes.** A continuous score is easy to tune and impossible to
audit. "Which tools can be run without asking?" must have an answer a reviewer can read off the
declaration, not compute.

**Speculate writes and compensate on a miss.** Rejected; see
[ADR-0003](0003-overlays-and-occ-instead-of-compensation.md).
