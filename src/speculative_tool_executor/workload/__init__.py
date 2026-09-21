"""The benchmark workload: a synthetic workspace, the tools over it, and the tasks.

- :mod:`corpus` -- the generated codebase and documentation service the tools read.
- :mod:`tools` -- the effect-typed tool registry, its canonicalization rules and the per-tool
  latency profiles.
- :mod:`tasks` -- the task generator: multi-step tasks with reference trajectories, produced by
  actually running the tools so every reference argument is one a real run would produce.
"""

from __future__ import annotations

__all__: list[str] = []
