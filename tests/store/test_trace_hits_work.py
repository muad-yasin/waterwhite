"""The rows ``Store.trace_hits`` reads for one document, and the work it takes to read them.

The walk of a checkpoint asks for the trace rows of every document a step read. The statement
started at the segment's steps and looked the document up once per step, so one call cost the
number of steps and the whole walk cost the number of documents times the number of steps. The
work is counted here as the virtual-machine instructions SQLite runs for the call (a progress
handler counts one per ten), not as seconds and not as the text of a query plan, which no run
can fake and no SQLite version changes.

The stores are made small and by hand: a segment of S steps, step k reading document k, and the
first document read again in the last step.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from waterwhite.store import Store, views

SEGMENT = "probe/seg-1"


def make_store(path: Path, steps: int, *, drop_index: bool = False) -> Store:
    Store.create(path, normaliser="control-chars/1").close()
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = OFF")
    con.execute("INSERT INTO run (run_id, name) VALUES (1, 'probe')")
    con.execute("INSERT INTO run_segment (segment_id, run_id, name, position, start_kind, trace_route) "
                "VALUES (1, 1, ?, 0, 'fresh', 'cache files')", (SEGMENT,))
    for k in range(1, steps + 1):
        con.execute("INSERT INTO step_trace (trace_id, segment_id, step, consumed_samples) VALUES (?, 1, ?, 1)", (k, k))
        con.execute("INSERT INTO trace_sample (trace_id, sample_pos, global_index, rank, sample_tokens, covered_tokens) "
                    "VALUES (?, 0, ?, 0, 8, 8)", (k, k))
        con.execute("INSERT INTO trace_row VALUES (?, 0, 0, 0, ?, 0, 5)", (k, k))
    # the first document is read again in the last step, twice in one sample: two rows in one order
    con.execute("INSERT INTO trace_row VALUES (?, 0, 1, 0, 1, 5, 8)", (steps,))
    if drop_index:
        con.execute("DROP INDEX trace_row_document")
    con.commit()
    con.close()
    return Store.open(path, readonly=True)


def vm_work(store: Store, call) -> tuple[int, object]:
    """What ``call`` does, and how many instructions SQLite ran for it (counted by tens)."""
    counted = [0]
    store._conn.set_progress_handler(lambda: counted.__setitem__(0, counted[0] + 1), 10)
    try:
        result = call()
    finally:
        store._conn.set_progress_handler(None, 0)
    return counted[0], result


def hits(store: Store, ordinal: int, **more) -> list:
    return list(store.trace_hits(SEGMENT, 0, ordinal, **more))


def test_the_rows_of_a_document_come_in_step_order(tmp_path):
    with make_store(tmp_path / "s.db", 40) as store:
        assert hits(store, 1) == [views.TraceHit(1, 0, 0, 0, 5, 8, 8), views.TraceHit(40, 0, 1, 5, 8, 8, 8)]
        assert hits(store, 1, upto_step=39) == [views.TraceHit(1, 0, 0, 0, 5, 8, 8)]
        assert hits(store, 7) == [views.TraceHit(7, 0, 0, 0, 5, 8, 8)]
        assert hits(store, 99) == []


def test_the_work_of_one_call_does_not_grow_with_the_steps_of_the_segment(tmp_path):
    with make_store(tmp_path / "small.db", 100) as small, make_store(tmp_path / "big.db", 1600) as big:
        small_work, small_rows = vm_work(small, lambda: hits(small, 50))
        big_work, big_rows = vm_work(big, lambda: hits(big, 50))
    assert small_rows == big_rows == [views.TraceHit(50, 0, 0, 0, 5, 8, 8)]
    assert small_work > 0
    # sixteen times the steps: the work of the call stays about the same (it was sixteen times)
    assert big_work <= 2 * small_work + 5, (small_work, big_work)


def test_a_store_without_the_document_index_gives_the_same_rows_and_no_database_error(tmp_path):
    with make_store(tmp_path / "with.db", 40) as with_index, \
            make_store(tmp_path / "without.db", 40, drop_index=True) as without_index:
        for ordinal in (1, 7, 40, 99):
            assert hits(without_index, ordinal) == hits(with_index, ordinal)
