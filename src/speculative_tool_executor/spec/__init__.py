"""Speculative execution of agent tool calls.

Modules, in the order they build on each other:

- :mod:`clock` -- injectable time; a virtual clock that runs discrete-event simulations inside a
  real asyncio loop.
- :mod:`latency` -- the lognormal latency model, drawn deterministically from the call itself.
- :mod:`store` and :mod:`overlay` -- the authoritative versioned store and the per-branch write
  overlays that speculative work is confined to, with optimistic concurrency control on commit.
- :mod:`canonicalize` -- one representative per equivalence class of tool arguments, so a
  speculative result can be matched to the call the model actually made.
- :mod:`cache` -- the content-addressed store of finished branches.
- :mod:`predictor` and :mod:`taint` -- what to speculate, and what a prediction may name.
- :mod:`controller` -- whether speculating is worth it, and when to stop.
- :mod:`runner`, :mod:`speculator` and :mod:`executor` -- running one branch, running the
  predicted ones, and matching the real call against them.
"""
