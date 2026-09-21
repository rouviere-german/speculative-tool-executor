"""Effect classes: the safety boundary that decides what may be executed speculatively.

Every tool declares exactly one effect class. The class is a property of the *handler*, not of
the call, and it is what the speculator consults before it is allowed to run anything ahead of
the model.

``PURE``
    No observable interaction with the outside world; the result is a function of the canonical
    arguments alone. Freely speculatable and cacheable without expiry.
``READ``
    Observes state the system owns (the versioned store, the workspace, a bundled service) but
    changes nothing. Speculatable; results are only usable while they are still current, which is
    enforced by a read-set version check and a per-tool freshness TTL.
``WRITE``
    Mutates state the system owns. Speculatable only against a per-branch overlay that buffers
    writes and records the versions it read; the overlay is applied to the authoritative store
    only when the real call matches and optimistic concurrency control passes.
``EXTERNAL``
    Causes an effect outside the system that cannot be rolled back (sending a message, calling a
    third party). **Never** speculated under any configuration, and always gated by an approval
    bound to the canonical argument hash.

The ordering below is a lattice from least to most dangerous, so policies can be expressed as
"at most this effect" rather than as a set of names.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["SPECULATABLE_EFFECTS", "Effect", "effect_rank"]


class Effect(StrEnum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


_RANK = {Effect.PURE: 0, Effect.READ: 1, Effect.WRITE: 2, Effect.EXTERNAL: 3}

#: The effect classes a speculator may ever run. ``EXTERNAL`` is deliberately absent, and
#: :func:`speculative_tool_executor.spec.speculator.Speculator` re-checks membership at launch
#: time rather than trusting a caller-supplied policy.
SPECULATABLE_EFFECTS = frozenset({Effect.PURE, Effect.READ, Effect.WRITE})


def effect_rank(effect: Effect) -> int:
    """Position in the lattice; higher means more dangerous to run ahead of the model."""
    return _RANK[effect]
