# Security policy

## Supported versions

Only the latest commit on `main` is supported. This is a portfolio project, not a product with
release branches.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private vulnerability
reporting ("Report a vulnerability" under the repository's Security tab). Include:

- the affected commit and component;
- steps to reproduce, ideally as a failing test;
- the impact you expect.

You can expect an acknowledgement within 7 days and a fix or a written assessment within 30 days.
Credit is given in the fix commit unless you ask otherwise.

## Scope

In scope: the code in this repository, its container image and its CI configuration.
Out of scope: denial of service by traffic volume, findings that need a compromised host, and
issues in third-party dependencies that are already publicly tracked (please report those
upstream).

## How this repository is hardened

- Threats, controls and the tests that prove them: [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).
- Every security claim is linked to evidence in [CLAIMS.md](CLAIMS.md).
- CI runs ruff, mypy --strict, pytest with a coverage floor, bandit, semgrep (registry packs plus
  `.semgrep/rules.yml`), pip-audit, gitleaks, a trivy image scan and a syft CycloneDX SBOM.
- Secrets come only from the environment; only `.env.example` is committed.
- Artifacts use pickle-free formats; pickle-family imports are banned by both
  `scripts/check_baseline.py` and a semgrep rule.

## Speculation-specific notes

This service runs tool calls *before* the model asks for them, so a few controls exist that an
ordinary agent runtime would not need:

- **`EXTERNAL` tools are never speculated.** Refused by the speculator by effect, re-checked by
  the runner at launch, and asserted by the handler itself.
- **Speculative writes never reach storage** until the authoritative call matches and optimistic
  concurrency control passes. An abandoned branch is discarded, not rolled back.
- **Kill switch.** `APP_SPECULATION_ENABLED=false` stops all speculation;
  `APP_SPECULATION_DISABLED_TOOLS` narrows it per tool. Both take effect on restart, with no
  rebuild, and both are tested end to end.
- **`GET /v1/speculation`** (authenticated) reports hit ratio, waste ratio, OCC conflicts,
  cancellations and whether the waste trip has already fired — the numbers you want during an
  incident, in one place.
- Residual risks, including what the taint rule does *not* protect against, are listed in
  [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md#residual-risk-stated-plainly).
