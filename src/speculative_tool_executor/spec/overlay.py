"""Per-branch write overlays: where speculative writes live until they are proven correct.

A branch is a buffer in front of the store. Reads go through the buffer and are recorded with the
version they saw; writes never reach SQLite until the branch commits. An aborted branch is simply
dropped, which is why a mispredicted write can never be observed -- there is nothing to undo.

The recorded read set is what makes a commit safe rather than merely late. Whatever the branch
read while it ran must still hold at commit time; otherwise the work was computed from a state
that no longer exists and is thrown away. Read-only branches (used for speculated ``READ`` tools)
use the same machinery with an empty write set, so a stale read is caught by exactly the same
check instead of a second, weaker one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from speculative_tool_executor.spec.store import CommitResult, VersionedStore

__all__ = ["BranchState", "OverlayError", "WriteOverlay"]


class OverlayError(RuntimeError):
    """The overlay was used in a way its state does not allow."""


class BranchState(StrEnum):
    OPEN = "open"
    COMMITTED = "committed"
    ABORTED = "aborted"
    CONFLICTED = "conflicted"


@dataclass(slots=True)
class WriteOverlay:
    """A buffered, version-tracked view of the store for one (possibly speculative) branch."""

    store: VersionedStore
    branch_id: str
    read_only: bool = False
    _reads: dict[str, int | None] = field(default_factory=dict, init=False, repr=False)
    _writes: dict[str, str | None] = field(default_factory=dict, init=False, repr=False)
    _state: BranchState = field(default=BranchState.OPEN, init=False)

    @property
    def state(self) -> BranchState:
        return self._state

    @property
    def read_set(self) -> dict[str, int | None]:
        """Keys observed by this branch and the version each had when first observed."""
        return dict(self._reads)

    @property
    def writes(self) -> dict[str, str | None]:
        """Buffered writes; ``None`` marks a delete."""
        return dict(self._writes)

    @property
    def dirty(self) -> bool:
        return bool(self._writes)

    def _require_open(self) -> None:
        if self._state is not BranchState.OPEN:
            raise OverlayError(f"branch {self.branch_id} is {self._state.value}")

    def read(self, key: str) -> str | None:
        """Read through the buffer. The first read of a key pins the version commit re-checks.

        A key the branch already wrote is served from the buffer, but the dependency is still
        recorded: a branch that *looked* at a key must depend on it whichever order it touched
        it in. Without that, write-then-read would be a blind write and could silently clobber
        another party's value -- the exact failure optimistic concurrency control is here for.
        """
        self._require_open()
        entry = self.store.get(key)
        self.observe(key, None if entry is None else entry.version)
        if key in self._writes:
            return self._writes[key]
        return None if entry is None else entry.value

    def write(self, key: str, value: str) -> None:
        self._require_open()
        if self.read_only:
            raise OverlayError(f"branch {self.branch_id} is read-only")
        self._writes[key] = value

    def delete(self, key: str) -> None:
        self._require_open()
        if self.read_only:
            raise OverlayError(f"branch {self.branch_id} is read-only")
        self._writes[key] = None

    def observe(self, key: str, version: int | None) -> None:
        """Record that this branch depends on ``key`` being at ``version`` at commit time.

        :meth:`read` calls this for every key it touches. It is also the way a handler that
        learned a version by some other route pins it, which is why it is public.
        """
        self._require_open()
        self._reads.setdefault(key, version)

    def validate(self) -> tuple[str, ...]:
        """Keys whose version moved since this branch read them, without committing anything."""
        conflicts = []
        for key, seen in sorted(self._reads.items()):
            entry = self.store.get(key)
            current = None if entry is None else entry.version
            if current != seen:
                conflicts.append(key)
        return tuple(conflicts)

    def commit(self) -> CommitResult:
        """Apply the branch if every version it read is unchanged; otherwise change nothing."""
        self._require_open()
        result = self.store.commit(self._reads, self._writes)
        self._state = BranchState.COMMITTED if result.committed else BranchState.CONFLICTED
        return result

    def abort(self) -> None:
        """Drop the branch. The store is untouched, so there is nothing to roll back."""
        if self._state is BranchState.OPEN:
            self._state = BranchState.ABORTED
        self._writes.clear()
        self._reads.clear()
