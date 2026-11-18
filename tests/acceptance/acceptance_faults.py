"""The faults of fixture/table/README.md section 7, put on a copy of a store.

One function per fault id, each written from the catalogue's "How it is applied"
column. "Outside the write API" is a direct edit of the SQLite file with the
standard library, against the tables of ``src/waterwhite/store/schema.sql``; the
faults the catalogue puts "through the write API" call the store, and one of the
scratch store's is a write through the tool followed by an edit of the store file.

A fault names its target in fixture terms (the edge of P, the log of the job
``paraphrase``). The target is found from the replica's records: the instance ids
the build returned and the line numbers of its manifests. Every edit checks that it
changed exactly the rows the catalogue names and raises ``FaultError`` otherwise,
so a fault that hit nothing cannot pass for an applied one.

A fault on a shard's map is found from the name the store keeps for the shard
(``artifact_name``), and a fault on a step trace from the name of its segment, so
the same function serves a store built with shards and registered runs and a small
store written by hand (``test_faults.py``). A fault on a record the store does not
hold is an error of the test (``FaultError``) and never a fault that was skipped.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import acceptance_scratch
import table


class FaultError(Exception):
    """A fault could not be applied as the catalogue says."""


# ``locate(RecordRef)`` gives (instance id, locator) of a record, from the
# replica's records.
Locate = Callable[[table.RecordRef], tuple[str, str]]

FILE_FAMILY_B = "file family: paraphrases of passages that source B supplies"
FAMILY_MEMBER_B = "B/r1"


def _one(db: sqlite3.Connection, sql: str, args: tuple, what: str) -> int:
    rows = db.execute(sql, args).fetchall()
    if len(rows) != 1:
        raise FaultError(f"{what}: expected one row in the store, found {len(rows)}")
    return rows[0][0]


def _change(db: sqlite3.Connection, sql: str, args: tuple, rows: int, what: str) -> None:
    changed = db.execute(sql, args).rowcount
    if changed != rows:
        raise FaultError(f"{what}: the edit changed {changed} row(s), the catalogue names {rows}")


def _occurrence(db: sqlite3.Connection, locate: Locate, ref: table.RecordRef) -> int:
    instance, locator = locate(ref)
    return _one(db, "SELECT o.occurrence_id FROM occurrence o JOIN artifact a USING (artifact_id) "
                    "WHERE a.instance_id = ? AND o.locator = ?", (instance, locator),
                f"the occurrence of {ref}")


def _edge(db: sqlite3.Connection, locate: Locate, child: table.RecordRef) -> int:
    return _one(db, "SELECT edge_id FROM edge WHERE child_id = ?",
                (_occurrence(db, locate, child),), f"the edge of {child}")


def _log(db: sqlite3.Connection, job: str) -> int:
    return _one(db, "SELECT log_id FROM harness_log WHERE job_name = ?", (job,),
                f"the stored harness log of the job {job}")


def _named_artifact(db: sqlite3.Connection, name: str) -> int:
    return _one(db, "SELECT artifact_id FROM artifact_name WHERE name = ?", (name,),
                f"the artifact {name}")


def _capture_map(db: sqlite3.Connection, shard: str) -> tuple[int, int, str]:
    """(binding id, declared count, declared digest) of the capture map of a shard,
    and the only map the shard holds: the table's faults are put on a store with no
    recovery map beside it."""
    artifact = _named_artifact(db, shard)
    held = db.execute("SELECT binding_id, namespace, declared_count, declared_digest, derivation "
                      "FROM binding WHERE artifact_id = ?", (artifact,)).fetchall()
    if len(held) != 1 or held[0][1] != "capture" or held[0][4] != "position":
        raise FaultError(f"the map of {shard}: expected the one capture map derived by "
                         f"position and no other, the store holds {[tuple(h[1:2] + h[4:]) for h in held]}")
    return held[0][0], held[0][2], held[0][3]


def _trace(db: sqlite3.Connection, segment: str, step: int) -> tuple[int, int]:
    """(trace id, consumed-sample count) of the trace of one step of a segment."""
    rows = db.execute("SELECT t.trace_id, t.consumed_samples FROM step_trace t "
                      "JOIN run_segment s USING (segment_id) WHERE s.name = ? AND t.step = ?",
                      (segment, step)).fetchall()
    if len(rows) != 1:
        raise FaultError(f"the trace of step {step} of {segment}: expected one row in the "
                         f"store, found {len(rows)}")
    return rows[0][0], rows[0][1]


P = table.RecordRef("gen/P", "P")
T = table.RecordRef("gen/T", "T")
S = table.RecordRef("gen/S", "S")
SC = table.RecordRef("src/C", "SC")
X_AT_A = table.RecordRef("src/A", "X@A")
X_AT_B = table.RecordRef("src/B", "X@B")


def edge_deleted(db: sqlite3.Connection, locate: Locate) -> None:
    """f.edge-deleted: the stated edge of P and its input rows are deleted. P's
    occurrence and the harness log stay."""
    edge = _edge(db, locate, P)
    _change(db, "DELETE FROM edge_input WHERE edge_id = ?", (edge,), 1, "the inputs of P's edge")
    _change(db, "DELETE FROM edge WHERE edge_id = ?", (edge,), 1, "P's edge")


def edge_file_level(db: sqlite3.Connection, locate: Locate) -> None:
    """f.edge-file-level: the label of T's stated edge is set to file-level and its
    derivation to file. Nothing else changes."""
    _change(db, "UPDATE edge SET label = 'file-level', derivation = 'file' "
                "WHERE edge_id = ? AND label = 'observed' AND derivation = 'stated'",
            (_edge(db, locate, T),), 1, "T's edge")


def bound_field(db: sqlite3.Connection, locate: Locate) -> None:
    """g.bound-field: the bound field of P's generation edge is set to not
    established. The log stays."""
    _change(db, "UPDATE edge SET bound_established = 0 "
                "WHERE edge_id = ? AND bound_established = 1",
            (_edge(db, locate, P),), 1, "the bound field of P's edge")


def log_outside_read(db: sqlite3.Connection, locate: Locate) -> None:
    """g.log-outside-read: one read is added to the stored log of the job
    ``paraphrase`` that names X@B, a record outside the hand-off, not refused."""
    log = _log(db, "paraphrase")
    instance, locator = locate(X_AT_B)
    last = db.execute("SELECT MAX(seq) FROM harness_read WHERE log_id = ?", (log,)).fetchone()[0]
    if last is None:
        raise FaultError("the stored log of the job paraphrase holds no read to add one after")
    _change(db, "INSERT INTO harness_read (log_id, seq, occurrence_id, outside_ref, refused) "
                "VALUES (?, ?, NULL, ?, 0)", (log, last + 1, f"{instance}:{locator}"), 1,
            "the added read")


def log_removed(db: sqlite3.Connection, locate: Locate) -> None:
    """g.log-removed: the link of P's edge to its harness log is removed and the
    log's rows are deleted. The bound field stays at established."""
    log = _log(db, "paraphrase")
    edge = _edge(db, locate, P)
    _change(db, "UPDATE edge SET log_id = NULL WHERE edge_id = ? AND log_id = ? "
                "AND bound_established = 1", (edge, log), 1, "the log link of P's edge")
    others = db.execute("SELECT COUNT(*) FROM edge WHERE log_id = ?", (log,)).fetchone()[0]
    if others:
        raise FaultError(f"{others} other edge(s) rest on the log of the job paraphrase")
    for tbl in ("harness_read", "harness_handed"):
        if db.execute(f"DELETE FROM {tbl} WHERE log_id = ?", (log,)).rowcount < 1:
            raise FaultError(f"the log of the job paraphrase held no row in {tbl}")
    _change(db, "DELETE FROM harness_log WHERE log_id = ?", (log,), 1, "the log's own row")


def model_entry_removed(db: sqlite3.Connection, locate: Locate) -> None:
    """g.model-entry-removed: the model entry ("none") is removed from the stored
    log of the job ``paraphrase``, so the log has no model entry."""
    _change(db, "UPDATE harness_log SET model = NULL WHERE log_id = ? AND model = 'none'",
            (_log(db, "paraphrase"),), 1, "the model entry")


def in_scope_only(db: sqlite3.Connection, locate: Locate) -> None:
    """s.in-scope-only: the input SC of S's edge is marked as handed and not stated
    as used."""
    _change(db, "UPDATE edge_input SET used = 0 "
                "WHERE edge_id = ? AND occurrence_id = ? AND used = 1",
            (_edge(db, locate, S), _occurrence(db, locate, SC)), 1, "the input SC of S's edge")


SEGMENT_1 = "E/seg-1"
SHARD_1, SHARD_2, SHARD_3 = "E/shard-1", "E/shard-2", "E/shard-3"
HALF_COVERED = 16  # tokens of the last sample that step 4 is marked as having read
SAMPLE_TOKENS = 33


def trace_deleted(db: sqlite3.Connection, locate: Locate) -> None:
    """f.trace-deleted: the trace of step 3 of E/seg-1 and its sample rows are
    deleted. The registered step stays."""
    trace, _ = _trace(db, SEGMENT_1, 3)
    _change(db, "DELETE FROM trace_row WHERE trace_id = ?", (trace,), _rows(db, "trace_row", trace),
            "the rows of the trace of step 3")
    _change(db, "DELETE FROM trace_sample WHERE trace_id = ?", (trace,),
            _rows(db, "trace_sample", trace), "the samples of the trace of step 3")
    _change(db, "DELETE FROM step_trace WHERE trace_id = ?", (trace,), 1, "the trace of step 3")
    _one(db, "SELECT ss.step FROM segment_step ss JOIN run_segment s USING (segment_id) "
             "WHERE s.name = ? AND ss.step = 3", (SEGMENT_1,), "the registered step 3")


def _rows(db: sqlite3.Connection, table_name: str, trace: int) -> int:
    count = db.execute(f"SELECT COUNT(*) FROM {table_name} WHERE trace_id = ?", (trace,)).fetchone()[0]
    if count < 1:
        raise FaultError(f"the trace {trace} holds no row in {table_name}")
    return count


def trace_half(db: sqlite3.Connection, locate: Locate) -> None:
    """f.trace-half: the coverage mark of the last sample (sample_pos 3) of the
    trace of step 4 of E/seg-1 is set to 16 of its 33 tokens. The trace still
    counts its four samples."""
    trace, counted = _trace(db, SEGMENT_1, 4)
    if counted != 4 or _rows(db, "trace_sample", trace) != 4:
        raise FaultError(f"the trace of step 4 counts {counted} samples, the fixture's step "
                         "holds four")
    _change(db, "UPDATE trace_sample SET covered_tokens = ? WHERE trace_id = ? AND sample_pos = 3 "
                "AND sample_tokens = ? AND covered_tokens = ?",
            (HALF_COVERED, trace, SAMPLE_TOKENS, SAMPLE_TOKENS), 1, "the last sample of step 4")


def map_deleted(db: sqlite3.Connection, locate: Locate) -> None:
    """f.map-deleted: the capture map of E/shard-1, its binding and its rows, is
    deleted. The shard's artifact row and the snapshots that name it stay."""
    binding, count, _ = _capture_map(db, SHARD_1)
    _change(db, "DELETE FROM map_candidate WHERE binding_id = ?", (binding,),
            db.execute("SELECT COUNT(*) FROM map_candidate WHERE binding_id = ?",
                       (binding,)).fetchone()[0], "the candidates of the map")
    _change(db, "DELETE FROM map_row WHERE binding_id = ?", (binding,), count, "the rows of the map")
    _change(db, "DELETE FROM binding WHERE binding_id = ?", (binding,), 1, "the map's binding")
    _one(db, "SELECT artifact_id FROM artifact WHERE artifact_id = ?",
         (_named_artifact(db, SHARD_1),), "the shard's artifact row")


def digest(db: sqlite3.Connection, locate: Locate) -> None:
    """f.digest: the first hex digit of the declared digest of the capture map of
    E/shard-3 is replaced by another. Nothing else changes."""
    binding, _, stored = _capture_map(db, SHARD_3)
    if not re.fullmatch(r"[0-9a-f]{64}", stored):
        raise FaultError(f"the stored digest is not 64 lower-case hex digits: {stored!r}")
    changed = ("1" if stored[0] == "0" else "0") + stored[1:]
    _change(db, "UPDATE binding SET declared_digest = ? WHERE binding_id = ? AND declared_digest = ?",
            (changed, binding, stored), 1, "the declared digest of the map of shard 3")


def label(db: sqlite3.Connection, locate: Locate) -> None:
    """f.label: the label field of the capture map of E/shard-2 is set to
    observed. Its derivation (position), its verification (passed) and the labels of
    its rows stay."""
    binding, _, _ = _capture_map(db, SHARD_2)
    _change(db, "UPDATE binding SET label = 'observed' WHERE binding_id = ? "
                "AND derivation = 'position' AND verification = 'passed' AND label <> 'observed'",
            (binding,), 1, "the label of the map of shard 2")


def edge_parent_missing(db: sqlite3.Connection, locate: Locate) -> None:
    """l.edge-parent: the parent reference of the stated edge of P (it names X@A) is
    changed to an occurrence id that no occurrence has. No occurrence row is
    deleted."""
    edge = _edge(db, locate, P)
    parent = _occurrence(db, locate, X_AT_A)
    absent = db.execute("SELECT COALESCE(MAX(occurrence_id), 0) + 1000 FROM occurrence").fetchone()[0]
    _change(db, "UPDATE edge_input SET occurrence_id = ? WHERE edge_id = ? AND occurrence_id = ? "
                "AND used = 1", (absent, edge, parent), 1, "the parent of P's edge")


def trace_one_short(db: sqlite3.Connection, locate: Locate) -> None:
    """l.trace-count: the last sample of the trace of step 5 of E/seg-1 and that
    sample's rows are deleted, and the trace's consumed-sample count is set to 3."""
    trace, counted = _trace(db, SEGMENT_1, 5)
    if counted != 4:
        raise FaultError(f"the trace of step 5 counts {counted} samples, the fixture's step "
                         "holds four")
    _change(db, "DELETE FROM trace_row WHERE trace_id = ? AND sample_pos = 3", (trace,),
            _rows_of_sample(db, trace, "trace_row"), "the rows of the last sample")
    _change(db, "DELETE FROM trace_sample WHERE trace_id = ? AND sample_pos = 3", (trace,), 1,
            "the last sample of step 5")
    _change(db, "UPDATE step_trace SET consumed_samples = 3 WHERE trace_id = ?", (trace,), 1,
            "the consumed-sample count of step 5")


def _rows_of_sample(db: sqlite3.Connection, trace: int, table_name: str) -> int:
    count = db.execute(f"SELECT COUNT(*) FROM {table_name} WHERE trace_id = ? AND sample_pos = 3",
                       (trace,)).fetchone()[0]
    if count < 1:
        raise FaultError(f"the last sample of the trace {trace} holds no row in {table_name}")
    return count


OUTSIDE_THE_WRITE_API: dict[str, Callable[[sqlite3.Connection, Locate], None]] = {
    "f.edge-deleted": edge_deleted,
    "f.edge-file-level": edge_file_level,
    "g.bound-field": bound_field,
    "g.log-outside-read": log_outside_read,
    "g.log-removed": log_removed,
    "g.model-entry-removed": model_entry_removed,
    "s.in-scope-only": in_scope_only,
    "f.trace-deleted": trace_deleted,
    "f.trace-half": trace_half,
    "f.map-deleted": map_deleted,
    "f.digest": digest,
    "f.label": label,
    "l.edge-parent": edge_parent_missing,
    "l.trace-count": trace_one_short,
}


def file_level_link_b(store_path: Path, locate: Locate) -> None:
    """s.file-level-link-B, through the write API: a file-level source link from
    P's occurrence to the member B/r1. P keeps its observed edge to X@A."""
    from waterwhite.store import Store

    instance, locator = locate(P)
    store = Store.open(store_path)
    try:
        store.add_source_link(instance, locator, FAMILY_MEMBER_B, "file-level", FILE_FAMILY_B,
                              derivation="file")
    finally:
        store.close()


def declare_499(store_path: Path, locate: Locate) -> None:
    """w.declare-499, through the write API: ``write_map`` is called declaring 499 documents
    for the shard of 500. The write must be refused with "declared count mismatch"; a write
    that is accepted, or refused another way, leaves no fault on the copy and is an error."""
    named = acceptance_scratch.declare_wrong_count(store_path, 499)
    if named != "declared count mismatch":
        raise FaultError(f"write_map refused 499 declared documents with {named!r}, not "
                         "'declared count mismatch'")


THROUGH_THE_WRITE_API: dict[str, Callable[[Path, Locate], None]] = {
    "s.file-level-link-B": file_level_link_b,
    "w.declare-499": declare_499,
}

R2_NAME = "R2"
_TEXT_VALUE = re.compile(rb'"text": "([^"\\])')


def file_byte_changed(store_path: Path, locate: Locate) -> None:
    """l.artifact-digest: one byte of one line's text of the file of R2 is changed
    on disk, in the copy of the fixture folder. The line count stays and the store
    is not edited. The file is the copy's: a path the store keeps that leaves the
    copy's folder is refused, so the original is never touched."""
    with closing(sqlite3.connect(store_path)) as db:
        stored = _one(db, "SELECT path FROM artifact WHERE artifact_id = ?",
                      (_named_artifact(db, R2_NAME),), "the path of R2")
    folder = store_path.parent.resolve()
    path = (folder / stored).resolve()
    if folder not in path.parents:
        raise FaultError(f"the file of R2 lies at {path}, outside the copy's folder {folder}")
    data = path.read_bytes()
    found = _TEXT_VALUE.search(data)
    if found is None:
        raise FaultError("no row of R2 holds a text value whose first character can be changed")
    at = found.start(1)
    changed = b"Q" if data[at:at + 1] != b"Q" else b"q"
    path.write_bytes(data[:at] + changed + data[at + 1:])


def count_499(store_path: Path, locate: Locate) -> None:
    """l.count-499: the complete 500-row map of the scratch store is written through the
    tool, then its stored count is set to 499 outside the write API (the store file is
    edited; ``acceptance_scratch.count_edit`` checks that exactly one binding changed)."""
    try:
        acceptance_scratch.count_edit(store_path)
    except acceptance_scratch.ScratchError as failed:
        raise FaultError(str(failed)) from failed


# A fault applied to the files of a copy and not to the store.
ON_DISK: dict[str, Callable[[Path, Locate], None]] = {
    "l.artifact-digest": file_byte_changed,
}

# A fault that writes through the tool first and then edits the store file outside the write
# API (the complete map of the scratch store, then its stored count).
WRITE_THEN_EDIT: dict[str, Callable[[Path, Locate], None]] = {
    "l.count-499": count_499,
}

FAULTS = (frozenset(OUTSIDE_THE_WRITE_API) | frozenset(THROUGH_THE_WRITE_API)
          | frozenset(ON_DISK) | frozenset(WRITE_THEN_EDIT))


def apply(state: str, store_path: Path, locate: Locate) -> None:
    """Put the fault ``state`` on the store file at ``store_path``, which is a
    fresh copy. One fault per copy."""
    if state in THROUGH_THE_WRITE_API:
        THROUGH_THE_WRITE_API[state](store_path, locate)
        return
    if state in ON_DISK:
        ON_DISK[state](store_path, locate)
        return
    if state in WRITE_THEN_EDIT:
        WRITE_THEN_EDIT[state](store_path, locate)
        return
    if state not in OUTSIDE_THE_WRITE_API:
        raise FaultError(f"no fault {state!r} is written here; written: {sorted(FAULTS)}")
    with closing(sqlite3.connect(store_path)) as db:
        with db:  # one transaction: a fault that fails half way changes nothing
            OUTSIDE_THE_WRITE_API[state](db, locate)
