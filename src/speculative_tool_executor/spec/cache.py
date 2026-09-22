"""The content-addressed cache of speculative results.

The address of a result *is* the hash of the call that produced it: tool name plus canonical
arguments. A speculative result can therefore only be handed to an authoritative call that would
have executed exactly the same thing, which removes the usual "is this cache entry for my
request?" question entirely.

Entries are single-use for ``READ`` and ``WRITE`` results, because each carries a branch whose
read set is validated and then consumed. ``PURE`` results carry no branch and depend on nothing
outside their arguments, so they stay until they are evicted.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

from speculative_tool_executor.agentic.effects import Effect
from speculative_tool_executor.core.canonical import canonical_json, sha256_hex
from speculative_tool_executor.spec.overlay import WriteOverlay

__all__ = ["CacheStats", "SpeculativeCache", "SpeculativeResult"]

Outcome = Literal["ok", "error"]


@dataclass(frozen=True, slots=True)
class SpeculativeResult:
    """A completed speculative execution, waiting for the authoritative call to claim it."""

    key: str
    tool: str
    effect: Effect
    branch_id: str
    outcome: Outcome
    output: JsonValue = None
    error: str | None = None
    overlay: WriteOverlay | None = None
    created_at: float = 0.0
    expires_at: float | None = None
    latency_seconds: float = 0.0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        # A reusable entry is handed out to every matching call, so a branch attached to it
        # would be committed once per claim -- and a branch can only be settled once.
        if self.reusable and self.overlay is not None:
            raise ValueError(f"a reusable ({self.effect.value}) result must not carry a branch")

    @property
    def reusable(self) -> bool:
        """Pure results depend on nothing but their arguments, so they survive being claimed."""
        return self.effect is Effect.PURE

    @property
    def output_sha256(self) -> str:
        return sha256_hex(canonical_json(self.output))

    def expired_at(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


@dataclass(slots=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    evicted: int = 0
    stored: int = 0
    conflicts: int = 0
    _tool_hits: dict[str, int] = field(default_factory=dict, repr=False)
    _tool_attempts: dict[str, int] = field(default_factory=dict, repr=False)

    def record_attempt(self, tool: str) -> None:
        self._tool_attempts[tool] = self._tool_attempts.get(tool, 0) + 1

    def record_hit(self, tool: str) -> None:
        self._tool_hits[tool] = self._tool_hits.get(tool, 0) + 1

    def hit_ratio(self, tool: str | None = None) -> float:
        if tool is not None:
            attempts = self._tool_attempts.get(tool, 0)
            return self._tool_hits.get(tool, 0) / attempts if attempts else 0.0
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def per_tool(self) -> dict[str, dict[str, float]]:
        return {
            tool: {
                "attempts": float(attempts),
                "hits": float(self._tool_hits.get(tool, 0)),
                "hit_ratio": self.hit_ratio(tool),
            }
            for tool, attempts in sorted(self._tool_attempts.items())
        }


class SpeculativeCache:
    """A bounded, content-addressed store of finished speculative branches."""

    def __init__(self, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max = max_entries
        self._entries: OrderedDict[str, SpeculativeResult] = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def put(self, result: SpeculativeResult) -> None:
        """Store a finished speculation, evicting the least recently stored entry if needed."""
        existing = self._entries.pop(result.key, None)
        if existing is not None and existing.overlay is not None:
            existing.overlay.abort()
        self._entries[result.key] = result
        self.stats.stored += 1
        while len(self._entries) > self._max:
            _, evicted = self._entries.popitem(last=False)
            if evicted.overlay is not None:
                evicted.overlay.abort()
            self.stats.evicted += 1

    def peek(self, key: str) -> SpeculativeResult | None:
        return self._entries.get(key)

    def take(self, key: str, *, now: float) -> SpeculativeResult | None:
        """Claim the result for ``key``, or None on a miss or an expired entry.

        Expiry is checked here rather than at storage time because the freshness window is
        measured against the moment the authoritative call arrives, not the moment the
        speculation finished.
        """
        result = self._entries.get(key)
        if result is None:
            self.stats.misses += 1
            return None
        if result.expired_at(now):
            self._entries.pop(key, None)
            if result.overlay is not None:
                result.overlay.abort()
            self.stats.expired += 1
            self.stats.misses += 1
            return None
        if not result.reusable:
            self._entries.pop(key, None)
        self.stats.hits += 1
        self.stats.record_hit(result.tool)
        return result

    def purge_expired(self, *, now: float) -> int:
        """Drop entries whose freshness window has closed. Returns how many were dropped."""
        stale = [key for key, result in self._entries.items() if result.expired_at(now)]
        for key in stale:
            result = self._entries.pop(key)
            if result.overlay is not None:
                result.overlay.abort()
            self.stats.expired += 1
        return len(stale)

    def discard(self, key: str) -> SpeculativeResult | None:
        result = self._entries.pop(key, None)
        if result is not None and result.overlay is not None:
            result.overlay.abort()
        return result

    def clear(self) -> list[SpeculativeResult]:
        """Abort and remove every entry; returns what was dropped so the caller can bill waste."""
        dropped = list(self._entries.values())
        for result in dropped:
            if result.overlay is not None:
                result.overlay.abort()
        self._entries.clear()
        return dropped


def freshness_deadline(now: float, window: float | None) -> float | None:
    """The instant a result stops being usable, or None when it never expires."""
    if window is None:
        return None
    if not math.isfinite(window) or window <= 0:
        raise ValueError("freshness window must be a positive, finite number of seconds")
    return now + window
