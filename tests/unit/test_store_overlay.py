"""The versioned store and the write overlays speculative work is confined to.

The two properties this module exists to prove:

- **An aborted branch leaves no trace.** Whatever a speculation read or buffered, discarding it
  must leave the store byte-identical to what it was. This is what makes speculating a write safe
  at all, and it is checked over randomly generated operation sequences rather than one example.
- **OCC rejects every overlay whose read set moved.** A branch built on a version that no longer
  holds must never commit, under any interleaving. The test drives randomized interleavings of
  branch operations and external writes and asserts the rule exactly: commit succeeds *if and
  only if* every version the branch read is unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from speculative_tool_executor.spec.overlay import BranchState, OverlayError, WriteOverlay
from speculative_tool_executor.spec.store import StoreError, VersionedStore

pytestmark = pytest.mark.unit

KEYS = st.sampled_from(["a", "b", "c", "d"])
VALUES = st.text(min_size=0, max_size=6)


@pytest.fixture
def store() -> Iterator[VersionedStore]:
    with VersionedStore(":memory:", synchronous="OFF") as opened:
        yield opened


# -- the store itself ---------------------------------------------------------------------------


def test_versions_come_from_one_monotonic_counter(store: VersionedStore) -> None:
    first = store.put("a", "1")
    second = store.put("b", "2")
    third = store.put("a", "3")
    assert first < second < third
    entry = store.get("a")
    assert entry is not None
    assert entry.version == third
    assert entry.value == "3"


def test_delete_leaves_a_tombstone_rather_than_removing_the_row(store: VersionedStore) -> None:
    """ "Created then deleted" must not be indistinguishable from "never existed"."""
    assert store.get("a") is None
    store.put("a", "1")
    version = store.delete("a")

    entry = store.get("a")
    assert entry is not None
    assert entry.value is None
    assert entry.present is False
    assert entry.version == version
    assert "a" not in store.items()


def test_a_branch_that_read_a_key_as_absent_conflicts_after_it_appears_and_goes(
    store: VersionedStore,
) -> None:
    """The reason tombstones exist, stated as a test."""
    overlay = WriteOverlay(store=store, branch_id="b1")
    assert overlay.read("a") is None  # pins "absent"

    store.put("a", "value")
    store.delete("a")  # back to "not there", but at a new version

    overlay.write("b", "x")
    result = overlay.commit()
    assert result.committed is False
    assert result.conflicts == ("a",)
    assert store.get("b") is None


def test_digest_tracks_tombstones_and_versions(store: VersionedStore) -> None:
    empty = store.digest()
    store.put("a", "1")
    written = store.digest()
    store.delete("a")
    deleted = store.digest()
    assert len({empty, written, deleted}) == 3


def test_keys_are_bounded_and_a_closed_store_refuses_work(store: VersionedStore) -> None:
    with pytest.raises(StoreError, match="1 and 512"):
        store.put("", "x")
    with pytest.raises(StoreError, match="1 and 512"):
        store.put("k" * 513, "x")
    store.close()
    with pytest.raises(StoreError, match="closed"):
        store.get("a")


# -- overlay mechanics --------------------------------------------------------------------------


def test_reads_go_through_buffered_writes(store: VersionedStore) -> None:
    store.put("a", "old")
    overlay = WriteOverlay(store=store, branch_id="b1")
    overlay.write("a", "new")
    assert overlay.read("a") == "new"
    assert store.get("a").value == "old"  # type: ignore[union-attr]


def test_read_only_branches_refuse_writes(store: VersionedStore) -> None:
    overlay = WriteOverlay(store=store, branch_id="b1", read_only=True)
    with pytest.raises(OverlayError, match="read-only"):
        overlay.write("a", "x")
    with pytest.raises(OverlayError, match="read-only"):
        overlay.delete("a")


def test_a_settled_branch_cannot_be_used_again(store: VersionedStore) -> None:
    overlay = WriteOverlay(store=store, branch_id="b1")
    overlay.write("a", "1")
    assert overlay.commit().committed is True
    assert overlay.state is BranchState.COMMITTED
    with pytest.raises(OverlayError, match="committed"):
        overlay.write("a", "2")

    aborted = WriteOverlay(store=store, branch_id="b2")
    aborted.abort()
    assert aborted.state is BranchState.ABORTED
    with pytest.raises(OverlayError, match="aborted"):
        aborted.read("a")


def test_only_the_first_read_of_a_key_pins_its_version(store: VersionedStore) -> None:
    store.put("a", "1")
    overlay = WriteOverlay(store=store, branch_id="b1")
    overlay.read("a")
    pinned = overlay.read_set["a"]
    store.put("a", "2")
    overlay.read("a")
    assert overlay.read_set["a"] == pinned


# -- the properties -----------------------------------------------------------------------------


@settings(max_examples=60)
@given(
    operations=st.lists(
        st.tuples(st.sampled_from(["read", "write", "delete"]), KEYS, VALUES),
        min_size=1,
        max_size=12,
    ),
    seed_rows=st.lists(st.tuples(KEYS, VALUES), max_size=4),
)
def test_aborting_a_branch_leaves_the_store_byte_identical(
    operations: Sequence[tuple[str, str, str]], seed_rows: Sequence[tuple[str, str]]
) -> None:
    """No matter what a speculation did, discarding it changes nothing."""
    with VersionedStore(":memory:", synchronous="OFF") as store:
        for key, value in seed_rows:
            store.put(key, value)
        before = store.digest()

        overlay = WriteOverlay(store=store, branch_id="speculative")
        for kind, key, value in operations:
            if kind == "read":
                overlay.read(key)
            elif kind == "write":
                overlay.write(key, value)
            else:
                overlay.delete(key)
        overlay.abort()

        assert store.digest() == before
        assert overlay.state is BranchState.ABORTED


@settings(max_examples=60, deadline=None)
@given(
    branch_ops=st.lists(
        st.tuples(st.sampled_from(["read", "write"]), KEYS, VALUES), min_size=1, max_size=8
    ),
    interleaved=st.lists(st.tuples(KEYS, VALUES), max_size=5),
    seed_rows=st.lists(st.tuples(KEYS, VALUES), max_size=4),
)
def test_occ_commits_exactly_when_the_read_set_is_unchanged(
    branch_ops: Sequence[tuple[str, str, str]],
    interleaved: Sequence[tuple[str, str]],
    seed_rows: Sequence[tuple[str, str]],
) -> None:
    """Randomized interleavings: the commit rule holds in both directions, not just one."""
    with VersionedStore(":memory:", synchronous="OFF") as store:
        for key, value in seed_rows:
            store.put(key, value)

        before = store.digest()
        overlay = WriteOverlay(store=store, branch_id="speculative")
        for kind, key, value in branch_ops:
            if kind == "read":
                overlay.read(key)
            else:
                overlay.write(key, value)
        read_set = overlay.read_set
        writes = overlay.writes
        # Nothing the branch did has reached the store yet.
        assert store.digest() == before

        # Another party commits while the branch is in flight.
        for key, value in interleaved:
            store.put(key, value)

        moved = {
            key
            for key, seen in read_set.items()
            if (store.get(key).version if store.get(key) is not None else None) != seen  # type: ignore[union-attr]
        }
        after_interleave = store.digest()
        result = overlay.commit()

        assert result.committed is (not moved)
        assert set(result.conflicts) == moved
        if result.committed:
            for written_key, written_value in writes.items():
                entry = store.get(written_key)
                assert entry is not None
                assert entry.value == written_value
        else:
            # A rejected commit is a no-op: the store is exactly what the other party left.
            assert store.digest() == after_interleave


def test_validate_reports_conflicts_without_committing(store: VersionedStore) -> None:
    store.put("a", "1")
    overlay = WriteOverlay(store=store, branch_id="b1")
    overlay.read("a")
    overlay.write("b", "2")
    assert overlay.validate() == ()

    store.put("a", "2")
    assert overlay.validate() == ("a",)
    assert store.get("b") is None
    assert overlay.state is BranchState.OPEN


def test_observe_records_a_dependency_established_elsewhere(store: VersionedStore) -> None:
    """A handler that learned a version by another route can still pin it."""
    version = store.put("a", "1")
    overlay = WriteOverlay(store=store, branch_id="b1")
    overlay.observe("a", version)
    overlay.write("b", "x")
    store.put("a", "2")
    assert overlay.commit().conflicts == ("a",)


def test_an_empty_commit_does_not_advance_the_version(store: VersionedStore) -> None:
    store.put("a", "1")
    version = store.version
    assert store.commit({}, {}).committed is True
    assert store.version == version
