# Claims and evidence

Every claim this repository makes, and the thing that holds it up. A claim with no test, bench or
CI job behind it does not belong in the README; a row here that points at something that does not
exist fails `scripts/check_baseline.py`.

Three kinds of evidence appear:

- **Invariant** — a property that must hold on every input. Backed by a test, often a
  property-based one.
- **Measurement** — a number produced by `bench/`, on stated assumptions, reproducible from a
  seed. Never typed in by hand.
- **Gate** — a CI job that fails the build.

## Performance claims

All latency numbers are **simulated**. The distributions are declared in
`src/speculative_tool_executor/workload/tools.py` and printed into `BENCHMARKS.md` with the
results, so a reader can disagree with the assumptions rather than the arithmetic.

| # | Claim | Kind | Evidence |
|---|---|---|---|
| P1 | Speculation reduces end-to-end p50 by 8.8% and p95 by 5.4% on 300 tasks | Measurement + Gate | `bench/bench_speculation.py`, floors (0.05 / 0.02) enforced in `bench/speculation_thresholds.json` by `.github/workflows/ci.yml#bench`; `tests/unit/test_bench_thresholds.py::test_losing_the_end_to_end_gain_fails_the_gate_even_if_tool_wait_holds` |
| P2 | Speculation removes 34.6% of total tool wait | Measurement | `bench/bench_speculation.py`, floor enforced in `bench/speculation_thresholds.json` |
| P3 | 75% of speculatable calls are served from a speculative branch (coverage) | Measurement | `bench/bench_speculation.py`, floor enforced by `--check` |
| P4 | Hit ratio is 0.41 at top-k 2 | Measurement | `bench/bench_speculation.py` |
| P5 | The end-to-end gain depends on the model-to-tool latency ratio, from 14.3% at 1:1 to 3.9% at 10:1 | Measurement | `bench/bench_speculation.py` latency ratio sweep |
| P6 | Tool-wait reduction is roughly flat (~36%) across that range, so only the tool share of the run moves | Measurement + Invariant | `bench/bench_speculation.py`; `tests/integration/test_speculation.py::test_the_latency_ratio_governs_how_much_end_to_end_time_is_recoverable` |
| P7 | Speculation never changes model time | Invariant | `tests/integration/test_speculation.py::test_model_time_is_untouched_by_speculation` |
| P8 | The first call of a task is never served speculatively | Invariant | `tests/integration/test_speculation.py::test_a_matching_branch_is_committed_and_credited_with_the_latency_it_absorbed` |
| P9 | A committed speculation is credited only the wait it actually absorbed | Invariant | `tests/integration/test_speculation.py::test_a_matching_branch_is_committed_and_credited_with_the_latency_it_absorbed` |
| P10 | Raising the probability threshold trades reduction for precision (hit 0.76, waste 0.3%) | Measurement | `bench/bench_speculation.py` ablations |
| P11 | Higher top-k speculates and wastes more | Invariant | `tests/integration/test_speculation.py::test_higher_top_k_speculates_more_and_wastes_more` |
| P12 | A benchmark run is reproducible from its seeds | Invariant | `tests/integration/test_speculation.py::test_running_the_same_scenario_twice_produces_identical_results` |
| P13 | The whole benchmark finishes on CPU in about a minute | Measurement | `bench/bench_speculation.py` reports its own wall-clock runtime |

## Cost claims

| # | Claim | Kind | Evidence |
|---|---|---|---|
| C1 | Wasted speculative spend is 1.5% of baseline spend (target <10%) | Measurement | `bench/bench_speculation.py`, ceiling enforced in `bench/speculation_thresholds.json` |
| C2 | Every launched speculation is billed exactly once | Invariant | `tests/integration/test_speculation.py::test_every_launched_speculation_is_billed_exactly_once` |
| C3 | The cost model is benefit vs waste in one unit, with the exchange rate configurable | Invariant | `tests/unit/test_controller.py::test_speculation_is_allowed_when_expected_benefit_beats_expected_waste` |
| C4 | Speculation that is not worth it is refused with the arithmetic recorded | Invariant | `tests/unit/test_controller.py::test_a_call_that_is_not_worth_it_is_refused_with_the_arithmetic_attached` |
| C5 | top-k counts launches, not guesses, so a refused candidate cannot suppress a usable one | Invariant | `tests/unit/test_speculator.py::test_top_k_counts_launches_not_guesses` |
| C6 | A hit is credited to the tool that produced it, including a reused `PURE` result claimed a second time | Invariant | `tests/unit/test_speculator.py::test_a_hit_is_credited_to_the_tool_that_produced_it` |

## Correctness claims

| # | Claim | Kind | Evidence |
|---|---|---|---|
| X1 | A speculative run produces identical outputs, identical committed writes in the same order, and a byte-identical store | Invariant | `tests/integration/test_speculation.py::test_speculation_removes_tool_wait_without_changing_what_the_run_did` |
| X2 | 0 of 300 tasks diverged, and the benchmark fails if any does | Measurement + Gate | `bench/bench_speculation.py`, `.github/workflows/ci.yml#bench` |
| X3 | The equivalence check is capable of failing | Invariant | `tests/integration/test_speculation.py::test_compare_reports_divergence_when_two_arms_disagree` |
| X4 | An aborted branch leaves the store byte-identical, for any sequence of operations | Invariant (property) | `tests/unit/test_store_overlay.py::test_aborting_a_branch_leaves_the_store_byte_identical` |
| X5 | OCC commits if and only if the read set is unchanged, under randomised interleavings | Invariant (property) | `tests/unit/test_store_overlay.py::test_occ_commits_exactly_when_the_read_set_is_unchanged` |
| X6 | A rejected commit leaves the store exactly as the other writer left it | Invariant (property) | `tests/unit/test_store_overlay.py::test_occ_commits_exactly_when_the_read_set_is_unchanged` |
| X7 | Every normalizer is idempotent, so a canonical form is a fixed point | Invariant (property) | `tests/unit/test_canonicalize.py::test_every_normalizer_is_idempotent` |
| X8 | Semantically equal arguments get equal keys | Invariant (property) | `tests/unit/test_canonicalize.py::test_equivalent_path_spellings_normalize_together` |
| X9 | Equal keys imply equal execution | Invariant | `tests/unit/test_canonicalize.py::test_equal_keys_imply_equal_execution` |
| X10 | Normalization never merges two genuinely different calls | Invariant | `tests/unit/test_canonicalize.py::test_different_paths_keep_different_keys` |
| X11 | Without the declared rules an equivalent spelling is a different call, and fails | Invariant | `tests/unit/test_canonicalize.py::test_without_the_declared_rules_an_equivalent_spelling_is_a_different_call` |
| X12 | Virtual time advances only when every participant is asleep | Invariant | `tests/unit/test_clock.py::test_time_does_not_advance_while_a_participant_is_still_running` |
| X13 | Cancelling a speculation returns the task count to baseline and does not stall the clock | Invariant | `tests/unit/test_clock.py::test_cancelling_a_sleep_returns_the_task_count_to_baseline` |
| X14 | A task cancelled but not yet resumed cannot make the clock jump to its deadline | Invariant | `tests/unit/test_clock.py::test_a_cancelled_sleeper_does_not_keep_the_simulation_from_moving` |
| X15 | The same call draws the same latency in both arms of the A/B | Invariant | `tests/unit/test_latency.py::test_the_same_call_draws_the_same_latency_in_every_run` |
| X16 | Rescaling for the sweep moves the model turn and nothing else | Invariant | `tests/unit/test_latency.py::test_rescaling_moves_the_model_turn_and_leaves_the_tools_alone` |
| X17 | Predicted arguments are derived from the previous output, so they generalise to unseen data | Invariant | `tests/unit/test_cache_predictor_taint.py::test_arguments_are_derived_from_the_previous_output_not_memorised` |
| X18 | A derivation that no longer resolves is dropped, never guessed | Invariant | `tests/unit/test_cache_predictor_taint.py::test_a_template_whose_path_no_longer_resolves_is_dropped_not_guessed` |
| X19 | A speculative READ or WRITE result is single-use; a PURE one is reusable because it carries no branch, and can be claimed twice | Invariant | `tests/unit/test_cache_predictor_taint.py::test_read_and_write_results_are_single_use_but_pure_results_are_not`, `tests/unit/test_cache_predictor_taint.py::test_a_reusable_result_cannot_be_built_with_a_branch_attached`, `tests/unit/test_speculator.py::test_a_pure_speculation_can_be_claimed_twice_without_exploding` |
| X20 | Every evaluation task's reference calls validate against the real tool schemas | Invariant | `tests/unit/test_workload.py::test_every_reference_call_validates_against_the_real_tool_schema` |
| X21 | Every speculatable step is fully derivable from the previous output | Invariant | `tests/unit/test_workload.py::test_every_speculatable_step_is_fully_derivable_from_the_previous_output` |
| X22 | Predictor training tasks are disjoint from the evaluation tasks | Invariant | `tests/unit/test_workload.py::test_training_tasks_are_disjoint_from_the_evaluation_tasks` |
| X23 | Under write contention the hit ratio falls and the store stays identical | Invariant | `tests/integration/test_speculation.py::test_contention_lowers_the_hit_rate_but_never_the_correctness` |
| X24 | A stale speculative read is discarded and re-executed | Invariant | `tests/integration/test_speculation.py::test_a_stale_speculative_read_is_discarded_and_re_executed` |
| X25 | 39 OCC conflicts at 50% contention, with the store still byte-identical | Measurement | `bench/bench_speculation.py` contention section |
| X26 | Disabling speculation reproduces the baseline exactly | Invariant | `tests/integration/test_speculation.py::test_disabling_speculation_reproduces_the_baseline_exactly` |
| X27 | Every ablation stays equivalent to the non-speculative run | Invariant + Measurement | `tests/integration/test_speculation.py::test_every_ablation_stays_equivalent_to_the_non_speculative_run`; `bench/bench_speculation.py` |
| X28 | A failing tool is recorded as data, not raised as a crash | Invariant | `tests/integration/test_speculation.py::test_a_failing_tool_is_data_not_a_crash` |
| X29 | A direct read-modify-write that loses an OCC race retries on top of the other party's value | Invariant | `tests/integration/test_direct_path.py::test_a_direct_write_retries_on_top_of_the_other_party_s_value` |
| X30 | When the retries run out the call fails with `CommitConflict` and the other party's write survives | Invariant | `tests/integration/test_direct_path.py::test_a_direct_write_that_runs_out_of_retries_fails_instead_of_losing_the_update` |
| X31 | A speculation whose handler raised is discarded and the call runs for real | Invariant | `tests/integration/test_direct_path.py::test_a_speculation_that_failed_is_discarded_and_the_call_runs_for_real` |
| X32 | A claimed branch that was already settled is never committed again or served | Invariant | `tests/integration/test_direct_path.py::test_a_claimed_branch_that_was_already_settled_is_never_served` |
| X33 | A branch that writes a key and then reads it still depends on that key's version | Invariant | `tests/unit/test_store_overlay.py::test_reading_back_a_buffered_write_still_pins_the_underlying_version` |
| X34 | The registry refuses a canonicalizer whose output fails the schema or is not a fixed point of the args model | Invariant | `tests/unit/test_tools.py::test_a_canonicalizer_whose_output_fails_the_schema_is_refused`, `tests/unit/test_tools.py::test_a_canonicalizer_whose_output_is_not_a_fixed_point_is_refused` |
| X35 | `tally_items` ignores order but not repetition, so a repeated item is a different call | Invariant | `tests/unit/test_canonicalize.py::test_a_tally_keeps_multiplicity_out_of_its_equivalence_class` |

## Security claims

Each maps to a row of `docs/THREAT_MODEL.md`.

| # | Claim | Kind | Evidence |
|---|---|---|---|
| S1 | External tools are never speculated, refused by effect rather than incidentally | Invariant | `tests/integration/test_security_controls.py::test_a_predictor_that_insists_on_an_external_tool_is_refused_by_effect` |
| S2 | The runner re-checks the effect instead of trusting its caller | Invariant | `tests/integration/test_security_controls.py::test_the_runner_re_checks_the_effect_rather_than_trusting_its_caller` |
| S3 | The external handler itself refuses a speculative branch | Invariant | `tests/integration/test_security_controls.py::test_the_external_handler_itself_refuses_a_speculative_branch` |
| S4 | No external effect occurs from a mispredicted run | Invariant | `tests/integration/test_security_controls.py::test_no_external_effect_occurs_from_a_mispredicted_run` |
| S5 | 0 external speculations across the 300-task benchmark, enforced as a hard ceiling | Measurement + Gate | `bench/bench_speculation.py`, `.github/workflows/ci.yml#bench` |
| S6 | A tool cannot be registered without declaring an effect class | Invariant | `tests/unit/test_tools.py::test_a_tool_cannot_be_defined_without_an_effect_class`, `tests/unit/test_tools.py::test_an_effect_that_is_not_an_effect_class_is_refused_at_definition`, `tests/unit/test_tools.py::test_registration_re_checks_the_effect_of_a_tool_assembled_behind_the_constructor` |
| S7 | An effect constrains which capabilities a tool may hold | Invariant | `tests/unit/test_tools.py::test_effect_constrains_which_capabilities_a_tool_may_hold` |
| S8 | A capability with irreversible reach cannot opt out of approval | Invariant | `tests/unit/test_tools.py::test_irreversible_capabilities_cannot_opt_out_of_approval` |
| S9 | A non-pure tool must bound how long a speculative result stays usable | Invariant | `tests/unit/test_tools.py::test_non_pure_tools_must_bound_how_long_a_speculative_result_lives` |
| S10 | An external call without approval is refused, not executed | Invariant | `tests/integration/test_security_controls.py::test_an_external_call_without_approval_is_refused_not_executed` |
| S11 | An approval does not carry over to different arguments | Invariant | `tests/integration/test_security_controls.py::test_an_approval_does_not_carry_over_to_different_arguments` |
| S12 | An approved call runs exactly once | Invariant | `tests/integration/test_security_controls.py::test_an_approved_call_runs_exactly_once` |
| S13 | The requester cannot approve its own call | Invariant | `tests/integration/test_security_controls.py::test_the_requester_cannot_approve_its_own_call` |
| S14 | A gullible model obeying an injection still sends nothing | Invariant | `tests/integration/test_security_controls.py::test_a_gullible_model_obeying_an_injection_still_sends_nothing` |
| S15 | Hostile tool output cannot forge or escape an untrusted-data envelope | Invariant | `tests/integration/test_security_controls.py::test_tool_output_is_wrapped_as_untrusted_data_that_cannot_forge_an_envelope` |
| S16 | A prediction naming something the model never saw is refused, and one too large to finish checking is refused too (the bound fails closed) | Invariant | `tests/integration/test_security_controls.py::test_a_prediction_naming_something_the_model_never_saw_is_refused`, `tests/unit/test_cache_predictor_taint.py::test_an_argument_too_large_to_walk_is_refused_rather_than_waved_through` |
| S17 | Schema constants are usable without having been uttered; nothing else is | Invariant | `tests/unit/test_cache_predictor_taint.py::test_schema_constants_are_allowed_without_having_been_uttered` |
| S18 | Read tools are jailed to the workspace | Invariant | `tests/integration/test_security_controls.py::test_read_tools_are_jailed_to_the_workspace` |
| S19 | Canonicalization does not launder a traversal attempt | Invariant | `tests/integration/test_security_controls.py::test_canonicalization_does_not_launder_a_traversal_attempt` |
| S20 | HTTP reads reach only the bundled service | Invariant | `tests/integration/test_security_controls.py::test_http_reads_only_reach_the_bundled_service` |
| S21 | A per-task allowlist removes the tools a task does not need | Invariant | `tests/integration/test_security_controls.py::test_a_scoped_registry_removes_the_tools_a_task_does_not_need` |
| S22 | The speculative budget caps speculative spend independently of the agent budget | Invariant | `tests/integration/test_security_controls.py::test_the_speculative_budget_caps_speculative_spend_independently` |
| S23 | The step budget stops a run and speculation does not evade it | Invariant | `tests/integration/test_security_controls.py::test_the_step_budget_stops_a_run_and_speculation_does_not_evade_it` |
| S24 | The waste trip turns speculation off for the rest of the run | Invariant | `tests/integration/test_security_controls.py::test_the_waste_trip_turns_speculation_off_for_the_rest_of_the_run` |
| S25 | `APP_SPECULATION_ENABLED=false` stops real speculation end to end | Invariant | `tests/integration/test_speculation_config.py::test_the_configured_kill_switch_stops_real_speculation` |
| S26 | `APP_SPECULATION_DISABLED_TOOLS` stops only the named tools | Invariant | `tests/integration/test_speculation_config.py::test_a_configured_per_tool_override_stops_only_that_tool` |
| S27 | Every call is audited with how it was served, in a verifiable chain | Invariant | `tests/integration/test_security_controls.py::test_every_call_is_audited_with_how_it_was_served` |
| S28 | An edited audit entry is detected; a truncated log is detected by an open writer or an anchored head, not from the file alone | Invariant | `tests/integration/test_security_controls.py::test_the_audit_chain_detects_an_edited_entry`, `tests/unit/test_audit.py::test_truncation_while_open_is_detected_on_the_next_append`, `tests/unit/test_audit.py::test_tail_truncation_needs_an_anchored_head` |
| S29 | An args model that masks a field is refused, so nothing escapes the call hash | Invariant | `tests/unit/test_tools.py::test_argument_models_that_do_not_round_trip_are_refused_without_echoing_values` |
| S30 | The `/v1/speculation` snapshot requires authentication and reports benefit and cost together | Invariant | `tests/integration/test_speculation_config.py::test_the_snapshot_reports_both_benefit_and_cost` |
| S31 | Speculation is scoped to the run's task group and cannot outlive it | Invariant | `tests/integration/test_speculation.py::test_leaving_the_session_settles_every_outstanding_branch` |
| S32 | Every speculation refusal has a named, counted reason | Invariant | `tests/unit/test_speculator.py::test_an_argument_the_model_never_saw_is_refused_by_taint` |

## Observability claims

| # | Claim | Kind | Evidence |
|---|---|---|---|
| B1 | Spans mark speculative vs authoritative work: one `tool.speculate` span (`speculative=true`) per branch, one `tool.execute` span (`speculative=false`, with `speculation.mode`) per call, linked by `call.hash` | Invariant | `tests/unit/test_speculator.py::test_speculative_and_authoritative_work_are_separate_spans`, `tests/unit/test_speculator.py::test_a_call_no_branch_matched_is_marked_direct_in_its_span` |
| B2 | A cancelled branch still ends its span, marked `cancelled` | Invariant | `tests/unit/test_speculator.py::test_a_cancelled_branch_still_closes_its_span` |
| B3 | The demo checks its own trace live, and that check can fail | Invariant | `tests/integration/test_demos_cli.py::test_the_demo_span_check_passes_only_on_a_faithful_trace` |
| B4 | `speculation_occ_conflicts_total` counts a speculative branch whose read version moved, and nothing else a branch is discarded for | Invariant | `tests/integration/test_direct_path.py::test_a_stale_speculative_read_is_counted_as_exactly_one_occ_conflict`, `tests/integration/test_direct_path.py::test_a_speculation_that_failed_is_discarded_and_the_call_runs_for_real` |

## Offline and reproducibility claims

| # | Claim | Kind | Evidence |
|---|---|---|---|
| O1 | Tests, demo and benchmarks run with no API key and no network | Invariant | `tests/conftest.py` blocks non-loopback sockets; `tests/integration/test_security_controls.py::test_nothing_in_the_runtime_reads_an_api_key` |
| O2 | Importing the offline modules never imports the Anthropic SDK | Invariant | `tests/unit/test_anthropic_provider.py::test_offline_modules_do_not_import_the_sdk` |
| O3 | The demo runs end to end and all its safety checks pass | Invariant | `tests/integration/test_demos_cli.py::test_demo_is_offline_equivalent_and_contains_the_injection` |
| O4 | The demo is re-runnable from a dirty working directory | Invariant | `tests/integration/test_demos_cli.py::test_demo_is_rerunnable_from_a_dirty_workdir` |
| O5 | The corpus and task generator are deterministic in their seeds | Invariant | `tests/unit/test_workload.py::test_generation_is_deterministic_in_its_seed` |
| O6 | The realised task mix matches the declared shape weights | Invariant | `tests/unit/test_workload.py::test_the_realised_shape_mix_matches_the_declared_weights` |
| O7 | The `AnthropicProvider` request shape is tested against a mocked client, never the network | Invariant | `tests/unit/test_anthropic_provider.py::test_request_uses_adaptive_thinking_strict_tools_and_refusal_fallback` |

## Engineering gates

| # | Claim | Kind | Evidence |
|---|---|---|---|
| G1 | Lint and formatting are enforced | Gate | `.github/workflows/ci.yml#lint` |
| G2 | `mypy --strict` passes over src, tests, bench and scripts | Gate | `.github/workflows/ci.yml#typecheck` |
| G3 | Test coverage is at least 85% | Gate | `.github/workflows/ci.yml#test` |
| G4 | `bandit` and `pip-audit` pass | Gate | `.github/workflows/ci.yml#bandit`, `.github/workflows/ci.yml#pip-audit` |
| G5 | `semgrep` enforces the repo rules, including "every tool declares an effect" and "EXTERNAL is never speculatable" | Gate | `.github/workflows/ci.yml#semgrep`, `.semgrep/rules.yml` |
| G6 | Secrets scanning runs on every push | Gate | `.github/workflows/ci.yml#gitleaks` |
| G7 | The container is linted, built, smoke-tested, scanned, and an SBOM is produced | Gate | `.github/workflows/ci.yml#docker` |
| G8 | Benchmark thresholds are enforced, not merely reported, including floors on the end-to-end p50 and p95 reductions | Gate | `.github/workflows/ci.yml#bench`, `tests/unit/test_bench_thresholds.py::test_the_measured_headline_passes_the_gate` |
| G9 | The shared baseline (files, README sections, hardening, evidence links) is checked | Gate | `.github/workflows/ci.yml#baseline`, `scripts/check_baseline.py` |
| G10 | No pickle-family serialization anywhere in the package | Gate | `scripts/check_baseline.py`, `tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization` |

## Claims deliberately **not** made

- That these latencies would be observed in production. They are simulation outputs on declared
  assumptions.
- That the ≥25% p50 end-to-end target from the original specification is met. It is not; see
  README *Results* and *Non-goals* for the structural reason and the measured numbers.
- That prompt injection is solved. It is contained by the effect classes and the approval gate;
  see *Residual risk* in the threat model.
- That an LLM drafter works. It is not implemented; only the protocol and the n-gram predictor
  ship.
- That the container image is verified here. It is configured; scanning runs in CI.
