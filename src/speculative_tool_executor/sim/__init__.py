"""The simulator: running the workload with and without speculation on a virtual clock.

- :mod:`agent` -- one agent run. The model is scripted to a task's reference trajectory, so the
  two arms of the A/B issue exactly the same calls and any latency difference is speculation.
- :mod:`experiment` -- the A/B itself, plus the sensitivity sweep over model-to-tool latency
  ratios and the ablations over predictor, top-k and canonicalization.
"""

from __future__ import annotations

__all__: list[str] = []
