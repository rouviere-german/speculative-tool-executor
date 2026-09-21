# 5. Predictions derive their arguments from the previous output, and may only name what the model has seen

- Status: accepted
- Date: 2026-09-21

## Context

Predicting the *tool* is nearly worthless. "A read comes after a list" is true and useless: the
speculation only pays if the arguments are right too, and the arguments are where all the entropy
is. So the predictor has to produce concrete calls.

That immediately creates a security question. A component that produces concrete tool arguments is
a component that can name a file, a host or a recipient. The specification explicitly allows the
predictor to be an LLM (`claude-haiku-4-5` as an optional drafter), whose output is untrusted by
construction. A predictor that could invent `/etc/passwd` and have it executed ahead of the model
would be a new attack surface introduced by an optimisation.

## Decision

**Arguments are learned as a derivation scheme, not as constants.** For each training transition
the model records, per argument field, either `OBSERVED` — the value appeared in the previous
tool's output, stored as the JSON path where it was found — or `PRIOR` — it did not, and is
remembered as a literal. At prediction time an `OBSERVED` field is re-resolved against the
*actual* previous output, and a template whose path no longer resolves is **dropped, not guessed**.

This is what makes the predictor generalise: `read_file(path = previous_output["entries"][0])` is
learned once and works on a directory it has never seen. The n-gram context is a *call signature*
(tool name plus argument field names, values excluded), because including values would make every
directory its own context and no amount of training data would cover it.

**A speculative call may only use string values the main model has already seen, or constants the
tool's own schema declares** (`enum`, `const`, defaults). "Already seen" means a value from the
operator's task, from a tool result already returned to the model, or from an argument the model
itself already sent. Numbers and booleans are exempt: they cannot name a file, a host or a
recipient, and taint-checking them would reject ordinary paging arguments for nothing.

## Consequences

- The predictor cannot widen the agent's reach. It can pick badly among calls the conversation has
  already made available; it cannot introduce a new target. This holds for the n-gram model and
  would hold identically for an LLM drafter, which is the point.
- Schema constants are allowed without having been uttered, so `send_report(recipient="security")`
  is expressible — and is then refused anyway, by effect.
- **What this does not buy, stated plainly:** text inside a tool result *is* content the model has
  seen, so an injected string is a legal speculative argument. The taint rule bounds the blast
  radius of a bad predictor; it is not an injection defence. Containment of injected instructions
  is the job of the effect classes, the allowlist and the approval gate, and those are what the
  injection test exercises.
- Predictions pass through `ToolRegistry.validate_call` — the same object real calls use — so
  there is no second, weaker path to a handler.
- The n-gram model is weak in a specific, measurable way: it cannot see the operator's prompt, so
  when a task says "read `src/auth/api.py`" it can only guess which of a directory's files follows
  a `list_dir`. This is why the workload's `read_bias` is a declared, swept parameter rather than
  a constant, and why the default is the least flattering setting (uniform choice).
- `PRIOR`-only templates instantiate in every context, because a literal always "resolves". They
  rank low through back-off and are usually refused, but they do consume ranking slots — which is
  why `top_k` counts *launches* rather than guesses (see `SpeculationPolicy.candidates_to_rank`).

## Alternatives considered

**Memorise concrete argument values per context.** Simple, and it produces a predictor that scores
well on the training distribution and collapses on anything else. It would also make every learned
argument a literal the taint rule then has to reject, so the measured hit rate would be near zero
for the wrong reason.

**Let the predictor propose anything and rely on the allowlist alone.** The allowlist constrains
*which tool*, not *which file*. A `read_file` of an arbitrary path is inside the allowlist and
outside what anyone asked for.

**Taint-check numbers too.** Rejected: paging offsets, limits and line numbers are legitimately
computed rather than quoted, and rejecting them would cost hit rate for no security gain, since a
number cannot address a resource on its own.

**Filter predictions after execution instead of before.** Too late by definition — the effect has
already happened. Every check that can be made before launch is made before launch.
