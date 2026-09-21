# Threat model

## What this system is

A runtime that executes an agent's tool calls, and — while the model is still producing its turn
— speculatively executes calls it predicts the model is about to make. A speculative result is
handed to the authoritative call only when that call's canonical argument hash matches, and only
when every version the speculative branch read is still current.

## What makes it different from an ordinary agent runtime

Speculation **widens what the system does before anyone asked for it**. Three consequences drive
everything below:

1. Work runs that the model never requested. If that work has effects, the system has acted on a
   guess.
2. The *predictor* becomes an input that can influence execution. It is not a trusted component:
   it is trained on trajectories, and the spec permits it to be an LLM, whose output is untrusted
   by construction.
3. Results computed at time *t* are used at time *t + d*. Anything that changed in between makes
   them wrong.

## Trust boundaries

| Zone | Contents | Trust |
|---|---|---|
| Operator | The task prompt, configuration, approvals | Trusted |
| Runtime | Registry, executor, speculator, controller, store | Trusted (this codebase) |
| Model output | Tool calls the model proposes | **Untrusted** — validated, allowlisted, budgeted |
| Predictor output | Predicted `(tool, args)` | **Untrusted** — same gates as model output, plus taint |
| Tool output | Workspace files, documents, stored notes | **Untrusted data** — wrapped in envelopes, never instructions |
| Outside world | `send_report` recipients | Reached only via an approved, never-speculated call |

## Assets

- **Integrity of the versioned store.** A speculative branch must never change it except by a
  validated commit.
- **The one-shot-ness of external effects.** A message sent twice, or sent at all on a guess, is
  unrecoverable.
- **The run's money.** Speculation spends on work that may be thrown away.
- **The audit chain.** The record of what ran, and whether it ran on a guess.

## STRIDE

Each row names the control and the test that holds it. Every referenced test is real and runs in
CI.

| # | Category | Threat | Control | Test |
|---|---|---|---|---|
| 1 | Spoofing | An unauthenticated caller reads operational metrics or drives the service | API-key auth (SHA-256 digests only, constant-time compare) on `/metrics` and `/v1/*` | `tests/integration/test_speculation_config.py::test_the_snapshot_endpoint_requires_authentication` |
| 2 | Spoofing | A prediction reaches a handler by a route that skips the allowlist | Predictions go through `ToolRegistry.validate_call`, the same object real calls use | `tests/unit/test_speculator.py::test_an_unknown_tool_is_refused` |
| 3 | Spoofing | An approval granted for one call releases another | Approval binds tool + canonical args hash + principal; single use; requester cannot approve | `tests/integration/test_security_controls.py::test_an_approval_does_not_carry_over_to_different_arguments` |
| 4 | Tampering | A mispredicted write is visible in the store | Writes buffer in a branch overlay; an aborted branch is dropped, not rolled back | `tests/unit/test_store_overlay.py::test_aborting_a_branch_leaves_the_store_byte_identical` |
| 5 | Tampering | A branch built on a stale read overwrites a newer value | OCC: commit re-validates every read version inside the same write transaction | `tests/unit/test_store_overlay.py::test_occ_commits_exactly_when_the_read_set_is_unchanged` |
| 6 | Tampering | A read-modify-write loses another party's update | `bump_index` records the version it read; a moved version rejects the commit | `tests/unit/test_workload.py::test_bump_index_records_the_version_it_read_so_the_increment_cannot_be_lost` |
| 7 | Tampering | Speculation changes what a run produced | Differential equivalence: same outputs, same committed writes in order, same store digest | `tests/integration/test_speculation.py::test_speculation_removes_tool_wait_without_changing_what_the_run_did` |
| 8 | Tampering | Two different calls share a cache key and one gets the other's result | Content-addressed cache keyed by tool + canonical args; the registry executes the normalized args | `tests/unit/test_canonicalize.py::test_equal_keys_imply_equal_execution` |
| 9 | Tampering | Arguments are changed after approval via a field the hash does not cover | Args model must round-trip: a masking model (`SecretStr`, `exclude=True`) is refused | `tests/unit/test_tools.py::test_argument_models_that_do_not_round_trip_are_refused_without_echoing_values` |
| 10 | Tampering | A stale read is served after the world moved on | Per-tool freshness TTL, checked when the authoritative call arrives, plus OCC for store reads | `tests/unit/test_cache_predictor_taint.py::test_expiry_is_measured_when_the_authoritative_call_arrives` |
| 11 | Repudiation | It is unclear whether a call ran directly or on a guess | Hash-chained audit marks every call `direct` / `speculative-committed` / `speculative-aborted` | `tests/integration/test_security_controls.py::test_every_call_is_audited_with_how_it_was_served` |
| 12 | Repudiation | An audit entry is edited after the fact | SHA-256 chain over canonical JSON; verification reports the offending line | `tests/integration/test_security_controls.py::test_the_audit_chain_detects_an_edited_entry` |
| 13 | Information disclosure | A prediction reads a path nobody mentioned | Taint rule: every string in a speculative argument must already have been shown to the model, or be a schema constant | `tests/integration/test_security_controls.py::test_a_prediction_naming_something_the_model_never_saw_is_refused` |
| 14 | Information disclosure | Path traversal via an argument the canonicalizer "cleaned up" | `normalize_path` preserves leading `..` so the jail still sees an escape | `tests/integration/test_security_controls.py::test_canonicalization_does_not_launder_a_traversal_attempt` |
| 15 | Information disclosure | Outbound HTTP to an attacker-chosen host | `fetch_doc` accepts only `https://docs.internal.test` (scheme and host checked, not prefix-matched) | `tests/integration/test_security_controls.py::test_http_reads_only_reach_the_bundled_service` |
| 16 | Information disclosure | Secrets or PII in logs | Redacting structlog processor on every record, including tracebacks | `tests/unit/test_redaction.py::test_known_secret_and_pii_formats_are_redacted` |
| 17 | Denial of service | Speculation consumes the run's budget | Separate speculative spend budget; refusals counted by reason | `tests/integration/test_security_controls.py::test_the_speculative_budget_caps_speculative_spend_independently` |
| 18 | Denial of service | A wrong predictor burns money for the whole run | Waste-ratio trip disables speculation once waste passes a share of baseline spend | `tests/unit/test_controller.py::test_the_waste_trip_disables_speculation_for_the_rest_of_the_run` |
| 19 | Denial of service | Unbounded concurrent branches | `max_in_flight`, bounded `SpeculativeCache`, bounded `TaintTracker` | `tests/unit/test_speculator.py::test_max_in_flight_bounds_concurrent_branches` |
| 20 | Denial of service | A speculation outlives the run that started it | One `TaskGroup` per run; leaving the session cancels and awaits every branch | `tests/integration/test_speculation.py::test_leaving_the_session_settles_every_outstanding_branch` |
| 21 | Denial of service | HTTP flooding or oversized bodies | Token-bucket rate limit per key, body-size cap, bounded client table | `tests/integration/test_app.py::test_rate_limit_rejects_bursts_and_exempts_probes` |
| 22 | Elevation of privilege | An external effect happens on a guess | `EXTERNAL` is outside `SPECULATABLE_EFFECTS`; refused by the speculator, re-checked by the runner, asserted by the handler | `tests/integration/test_security_controls.py::test_a_predictor_that_insists_on_an_external_tool_is_refused_by_effect` |
| 23 | Elevation of privilege | A tool claims a harmless effect while holding a dangerous capability | `EFFECT_CAPABILITIES` constrains capabilities per effect; registration fails otherwise | `tests/unit/test_tools.py::test_effect_constrains_which_capabilities_a_tool_may_hold` |
| 24 | Elevation of privilege | A tool with irreversible reach opts out of approval | `IRREVERSIBLE_CAPABILITIES` cannot be combined with `requires_approval=False` | `tests/unit/test_tools.py::test_irreversible_capabilities_cannot_opt_out_of_approval` |
| 25 | Elevation of privilege | Injected text in tool output becomes an instruction | Untrusted-data envelopes (escaping prevents forging or escaping a container); enforcement is the registry and the gate, not the model's judgement | `tests/integration/test_security_controls.py::test_a_gullible_model_obeying_an_injection_still_sends_nothing` |
| 26 | Elevation of privilege | Speculation cannot be stopped during an incident | `APP_SPECULATION_ENABLED` kill switch and `APP_SPECULATION_DISABLED_TOOLS` per-tool overrides | `tests/integration/test_speculation_config.py::test_the_configured_kill_switch_stops_real_speculation` |

## Residual risk, stated plainly

- **Prompt injection is mitigated, not solved.** Text inside a tool result is content the model
  has seen, so an injected string *is* a legal speculative argument under the taint rule. What
  taint prevents is the predictor widening the agent's reach on its own. What stops an injected
  instruction from having an irreversible effect is the effect classes and the approval gate —
  and those are what the tests exercise.
- **A `WRITE` tool may run on a guess without a human.** That is deliberate
  ([ADR-0002](adr/0002-effect-classes-and-the-approval-boundary.md)): a contained, versioned,
  discardable write is not the same risk as a sent message. If a deployment disagrees, every tool
  can set `requires_approval=True`, and the effect lattice still applies.
- **Freshness for non-store reads is a timer, not a proof.** `read_file` and `fetch_doc` observe
  state the versioned store does not track, so a speculative result is bounded by a per-tool TTL.
  A change inside that window is not detected. The store-backed `get_note` has no such gap.
- **The predictor's training data is trusted.** Trajectories are generated locally. A deployment
  that trained on adversarially shaped history could bias *which* legal calls get run early; it
  could not make an illegal call legal, because the taint rule, the allowlist and the effect
  checks are applied at launch, not at training time.
- **The simulator is not the world.** Latency, cost and contention are declared assumptions. The
  safety properties (equivalence, no external effects, OCC) are invariants that hold regardless
  of those numbers; the performance properties are conditional on them.
- **Single process.** Cross-process safety rests on SQLite's `BEGIN IMMEDIATE` and file locks for
  the approval store, budget ledger and audit log. Multi-node coordination is out of scope.
