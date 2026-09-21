"""The authoritative versioned key-value store (SQLite), and the commit rule for overlays.

Every key carries a version drawn from one monotonic counter, so a version is a global point in
the store's history rather than a per-key revision. Deletes leave a tombstone with a fresh
version instead of removing the row: without one, "created then deleted" would look identical to
"never existed", and an overlay that read a key as absent would commit over a value that had
appeared and gone in between.

Commit is the only place speculative work becomes real. It runs in a single ``BEGIN IMMEDIATE``
transaction that first re-checks every version the overlay observed and then applies the writes,
so an overlay is either applied whole against the state it actually read or not at all. That is
plain optimistic concurrency control (Kung and Robinson, "On Optimistic Methods for Concurrency
Control", 1981), with the read-set validation phase doing the work that makes speculation safe:
a branch built on a stale read can never be committed.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from speculative_tool_executor.core.canonical import canonical_json, sha256_hex

__all__ = ["CommitResult", "Entry", "StoreError", "VersionedStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key      TEXT    PRIMARY KEY,
    value    TEXT,
    version  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    next_version  INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta (id, next_version) VALUES (1, 1);
"""

Synchronous = Literal["OFF", "NORMAL", "FULL"]


class StoreError(RuntimeError):
    """The store could not serve a request (bad key, closed connection, corrupt file)."""


@dataclass(frozen=True, slots=True)
class Entry:
    """A key's current state. ``value is None`` is a tombstone: the key existed and was deleted."""

    key: str
    value: str | None
    version: int

    @property
    def present(self) -> bool:
        return self.value is not None


@dataclass(frozen=True, slots=True)
class CommitResult:
    """Outcome of validating and applying one overlay."""

    committed: bool
    version: int
    conflicts: tuple[str, ...] = ()
    written: tuple[str, ...] = field(default=())

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicts)


class VersionedStore:
    """A small versioned KV store. One instance owns one SQLite connection and a lock.

    The lock makes the store safe to share between threads; the ``BEGIN IMMEDIATE`` transaction
    makes validate-then-apply atomic against other processes holding the same file.
    """

    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        synchronous: Synchronous = "FULL",
    ) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self._path, isolation_level=None, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(f"PRAGMA synchronous={synchronous}")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.executescript(_SCHEMA)
        self._closed = False

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> VersionedStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise StoreError("store is closed")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        self._check_open()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("BEGIN IMMEDIATE")
            try:
                yield cursor
            except BaseException:
                cursor.execute("ROLLBACK")
                raise
            else:
                cursor.execute("COMMIT")

    @staticmethod
    def _check_key(key: str) -> str:
        if not key or len(key) > 512:
            raise StoreError("key must be between 1 and 512 characters")
        return key

    @property
    def version(self) -> int:
        """The version the next write will receive minus one: the store's current point in time."""
        self._check_open()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT next_version FROM meta WHERE id = 1")
            return int(cursor.fetchone()[0]) - 1

    def get(self, key: str) -> Entry | None:
        """The key's entry, or None if it never existed. A tombstone is returned, not hidden."""
        self._check_open()
        self._check_key(key)
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT value, version FROM kv WHERE key = ?", (key,))
            row = cursor.fetchone()
        return None if row is None else Entry(key=key, value=row[0], version=int(row[1]))

    def items(self) -> dict[str, str]:
        """Every live key and value, for snapshots and equivalence checks."""
        self._check_open()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT key, value FROM kv WHERE value IS NOT NULL ORDER BY key")
            return {str(key): str(value) for key, value in cursor.fetchall()}

    def snapshot(self) -> list[list[str | int | None]]:
        """Every row, tombstones and versions included, in key order."""
        self._check_open()
        with self._lock, closing(self._connection.cursor()) as cursor:
            cursor.execute("SELECT key, value, version FROM kv ORDER BY key")
            return [[str(key), value, int(version)] for key, value, version in cursor.fetchall()]

    def digest(self) -> str:
        """SHA-256 over the full snapshot: equal digests mean byte-identical store state."""
        return sha256_hex(canonical_json({"v": 1, "rows": self.snapshot()}))

    def put(self, key: str, value: str) -> int:
        """Write one key outside any overlay (fixtures and setup). Returns the new version."""
        return self._apply({self._check_key(key): value})

    def delete(self, key: str) -> int:
        """Tombstone one key outside any overlay. Returns the new version."""
        return self._apply({self._check_key(key): None})

    def _apply(self, writes: Mapping[str, str | None]) -> int:
        with self._transaction() as cursor:
            return self._apply_within(cursor, writes)

    @staticmethod
    def _apply_within(cursor: sqlite3.Cursor, writes: Mapping[str, str | None]) -> int:
        cursor.execute("SELECT next_version FROM meta WHERE id = 1")
        version = int(cursor.fetchone()[0])
        for key in sorted(writes):
            cursor.execute(
                "INSERT INTO kv (key, value, version) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, version = excluded.version",
                (key, writes[key], version),
            )
        if writes:
            cursor.execute("UPDATE meta SET next_version = ? WHERE id = 1", (version + 1,))
            return version
        return version - 1

    def commit(
        self, read_set: Mapping[str, int | None], writes: Mapping[str, str | None]
    ) -> CommitResult:
        """Validate ``read_set`` against current versions and, if it holds, apply ``writes``.

        Both phases run inside one write transaction, so no other writer can slip between the
        check and the apply. A conflict leaves the store untouched and names the offending keys.
        """
        for key in (*read_set, *writes):
            self._check_key(key)
        with self._transaction() as cursor:
            conflicts: list[str] = []
            for key in sorted(read_set):
                cursor.execute("SELECT version FROM kv WHERE key = ?", (key,))
                row = cursor.fetchone()
                current = None if row is None else int(row[0])
                if current != read_set[key]:
                    conflicts.append(key)
            if conflicts:
                cursor.execute("SELECT next_version FROM meta WHERE id = 1")
                current_version = int(cursor.fetchone()[0]) - 1
                return CommitResult(
                    committed=False, version=current_version, conflicts=tuple(conflicts)
                )
            version = self._apply_within(cursor, writes)
            return CommitResult(committed=True, version=version, written=tuple(sorted(writes)))
