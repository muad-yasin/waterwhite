"""Schema version 3 of the store (docs/store.md "Opening a store" and "The write methods").

Step 3 gives the hand-off of a harness log its position and gives one array of a blend
one digest. Every store here is made by the write methods; a store of an older version
is such a store with the step taken back by SQL (the two indexes and the column step 3
adds dropped, the version row set to 2), so that its rows were written by the real API.
A test compares one observed value with a value a document or a named constant gives.

The sentences the expectations rest on (docs/store.md):

- the schema is at version 3; a new store and a migrated one hold the same schema;
- the hand-off of a log is read back in the order it was handed, also after the store
  was closed and opened again; a record handed twice is stored once, at its first
  position; a second registration of a stored job compares the hand-off in order;
- a log written before version 3 has no position, reads in the order of its occurrence
  ids, and its job is compared as a set;
- a store that holds more than one row for one array of a segment's blend is refused at
  step 3 with "cache digest conflict" (exit 3), changes nothing in step 3 and keeps every
  row (a store from version 1 keeps the version 2 it reached), and the unique index
  refuses such a row in a store at version 3;
- the digest of a run record is the digest of the text the store holds.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from waterwhite import cli, errors
from waterwhite import vocabulary as v
from waterwhite.records import HarnessRead
from waterwhite.store.store import Store

REPO = Path(__file__).resolve().parents[2]


def _load_world():
    held = sys.modules.get("ww_world")
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location("ww_world", REPO / "tests" / "coverage" / "ww_world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ww_world"] = module
    spec.loader.exec_module(module)
    return module


ww = _load_world()

# --- The expected values ------------------------------------------------------------------

SCHEMA_NOW = 5  # step 3 is the subject of this file; steps 4 (the log's mark) and 5 follow it
SCHEMA_BEFORE = 2
NAME_CACHE = "cache digest conflict"
EXIT_LOUD_FAULT = 3
POSITION_COLUMN = "position"
INDEX_POSITION = "harness_handed_one_per_position"
INDEX_BLEND = "segment_cache_digest_one_blend"

# The records of the source, in the order of their occurrence ids, are L1, L2, L3.
HANDED_OUT_OF_ID_ORDER = (3, 1, 2)
HANDED_TWICE = (2, 1, 2)
HANDED_TWICE_STORED = (2, 1)
IN_ID_ORDER = (1, 2, 3)


# --- Helpers ---------------------------------------------------------------------------------


def lines(a: str, numbers) -> list:
    return [ww.ref(a, n) for n in numbers]


def keys(refs) -> list[str]:
    return [r.key for r in refs]


def build_store(base: Path, numbers=HANDED_OUT_OF_ID_ORDER, job: str = "order-job"):
    """A store with a source A of three lines and an output of one job that was handed
    A's lines in the given order and states all of them as used. Closed on return."""
    base.mkdir()
    store = Store.create(base / "s.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN])
    a = ww.ingest(store, base / "A.jsonl", ["alpha", "beta", "gamma"], "A", "src/A")
    handed = lines(a, numbers)
    ww.record(store, base / "Q.jsonl", ["summary"], "gen/Q", v.ROLE_INTERMEDIATE,
              ww.step("generate", "Q"), [a], [ww.generation_edge(1, job, handed, handed)])
    store.close()
    return base / "s.db", a


def register_again(store: Store, base: Path, a: str, numbers, job: str = "order-job", tag: str = "R"):
    """A second output of the same job, handed the same records in the given order."""
    handed = lines(a, numbers)
    reads = [HarnessRead(occurrence=ref) for ref in lines(a, HANDED_OUT_OF_ID_ORDER)]
    return ww.record(store, base / f"{tag}.jsonl", [f"second {tag}"], f"gen/{tag}", v.ROLE_INTERMEDIATE,
                     ww.step("generate", tag), [a], [ww.generation_edge(1, job, handed, handed, reads)])


def blend_record(*digests: str, extra=None) -> dict:
    record = ww.run_record("G", "G/seg-1", "fresh", 1, 1)
    record["cache_digests"] = [{"kind": v.CACHE_DATASET_INDEX, "shard": None, "digest": d} for d in digests]
    if extra is not None:
        record["notes"] = extra
    return record


def register_blend_segment(store: Store, *digests: str, extra=None) -> None:
    store.register_run_segment(
        run="G", segment="G/seg-1", position=0, mixture_instance_id=None, parent_checkpoint=None,
        start_kind="fresh", run_record=blend_record(*digests, extra=extra), steps=[(1, 1, None)],
        trace_route=v.ROUTE_NONE, trace_cause=v.NAMES_TRACE_MISSING,
        trace_cause_detail=v.TRACE_MISSING_NO_DIGEST,
    )


def build_blend_store(path: Path) -> Path:
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store, "a" * 64)
    return path


def take_step_3_back(path: Path) -> None:
    """Make the file the store it was at version 2: what step 3 added is dropped and the
    version row says 2."""
    raw = sqlite3.connect(path)
    try:
        raw.execute("DROP INDEX trace_row_segment_document")  # step 5 taken back first
        raw.execute("ALTER TABLE trace_row DROP COLUMN segment_id")
        raw.execute("DROP TABLE shard_files_check")
        raw.execute("ALTER TABLE harness_log DROP COLUMN hand_off_checked")  # step 4 taken back first
        raw.execute(f"DROP INDEX {INDEX_POSITION}")
        raw.execute(f"DROP INDEX {INDEX_BLEND}")
        raw.execute(f"ALTER TABLE harness_handed DROP COLUMN {POSITION_COLUMN}")
        raw.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_BEFORE),))
        raw.commit()
    finally:
        raw.close()


def take_step_2_back(path: Path) -> None:
    """After take_step_3_back: make the file a version 1 store, the harness column
    dropped and the one-binding index of schema.sql in place of the one step 2 made."""
    raw = sqlite3.connect(path)
    try:
        raw.execute("DROP INDEX binding_one_verified_per_mode")
        raw.execute("CREATE UNIQUE INDEX binding_one_per_mode "
                    "ON binding(artifact_id, namespace, COALESCE(derivation, ''))")
        raw.execute("ALTER TABLE harness_log DROP COLUMN harness")
        raw.execute("UPDATE meta SET value = '1' WHERE key = 'schema_version'")
        raw.commit()
    finally:
        raw.close()


def add_blend_row(path: Path, digest: str) -> None:
    raw = sqlite3.connect(path)
    try:
        raw.execute(
            "INSERT INTO segment_cache_digest (segment_id, kind, shard_position, digest) VALUES "
            "((SELECT segment_id FROM run_segment WHERE name = 'G/seg-1'), 'dataset_index', NULL, ?)",
            (digest,))
        raw.commit()
    finally:
        raw.close()


def scalar(path: Path, sql: str, params=()):
    raw = sqlite3.connect(path)
    try:
        return raw.execute(sql, params).fetchone()[0]
    finally:
        raw.close()


def schema_of(path: Path) -> set:
    raw = sqlite3.connect(path)
    try:
        return {tuple(r) for r in raw.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
    finally:
        raw.close()


def run_cli(*argv):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv], stdout=out, stderr=err)
    return code, json.loads(out.getvalue() or err.getvalue())


# --- A new store -----------------------------------------------------------------------------


def test_a_new_store_is_at_version_3_with_the_position_column_and_the_blend_index(tmp_path):
    path = tmp_path / "new.db"
    Store.create(path, normaliser=ww.NORMALISER).close()
    raw = sqlite3.connect(path)
    try:
        version = raw.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        columns = {r[1] for r in raw.execute("PRAGMA table_info(harness_handed)")}
        indexes = {r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    finally:
        raw.close()
    assert version == str(SCHEMA_NOW)
    assert POSITION_COLUMN in columns
    assert {INDEX_POSITION, INDEX_BLEND} <= indexes


def test_a_second_row_for_one_blend_array_with_a_null_position_is_refused_by_the_file(tmp_path):
    path = build_blend_store(tmp_path / "blend.db")
    with pytest.raises(sqlite3.IntegrityError):
        add_blend_row(path, "b" * 64)
    with pytest.raises(sqlite3.IntegrityError):
        add_blend_row(path, "a" * 64)
    assert scalar(path, "SELECT COUNT(*) FROM segment_cache_digest") == 1


# --- The hand-off keeps its order --------------------------------------------------------------


def test_a_hand_off_out_of_occurrence_order_is_read_back_in_the_order_handed_after_reopening(tmp_path):
    path, a = build_store(tmp_path / "order")
    with Store.open(path) as store:
        observed = keys(store.harness_log("order-job").handed)
    assert observed == keys(lines(a, HANDED_OUT_OF_ID_ORDER))


def test_a_record_handed_twice_is_stored_once_at_its_first_position(tmp_path):
    base = tmp_path / "twice"
    base.mkdir()
    with Store.create(base / "s.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        a = ww.ingest(store, base / "A.jsonl", ["alpha", "beta", "gamma"], "A", "src/A")
        stored = lines(a, HANDED_TWICE_STORED)
        edge = ww.generation_edge(1, "order-job", stored, stored)
        twice = dataclasses.replace(edge.harness_log, handed=tuple(lines(a, HANDED_TWICE)))
        ww.record(store, base / "Q.jsonl", ["summary"], "gen/Q", v.ROLE_INTERMEDIATE,
                  ww.step("generate", "Q"), [a], [dataclasses.replace(edge, harness_log=twice)])
    with Store.open(base / "s.db") as store:
        observed = keys(store.harness_log("order-job").handed)
    assert observed == keys(lines(a, HANDED_TWICE_STORED))


def test_the_positions_of_a_log_are_numbered_from_0_in_the_order_handed(tmp_path):
    path, a = build_store(tmp_path / "numbers")
    raw = sqlite3.connect(path)
    try:
        rows = raw.execute(
            "SELECT o.locator, h.position FROM harness_handed h "
            "JOIN occurrence o ON o.occurrence_id = h.occurrence_id ORDER BY h.position").fetchall()
    finally:
        raw.close()
    assert rows == [("L3", 0), ("L1", 1), ("L2", 2)]


def test_the_same_job_registered_again_with_the_same_hand_off_is_accepted(tmp_path):
    path, a = build_store(tmp_path / "same")
    with Store.open(path) as store:
        register_again(store, path.parent, a, HANDED_OUT_OF_ID_ORDER)
        assert keys(store.harness_log("order-job").handed) == keys(lines(a, HANDED_OUT_OF_ID_ORDER))


def test_the_same_job_registered_again_with_the_hand_off_in_another_order_is_refused(tmp_path):
    path, a = build_store(tmp_path / "other")
    with Store.open(path) as store:
        with pytest.raises(errors.InvalidRecord, match="already has a stored log"):
            register_again(store, path.parent, a, IN_ID_ORDER)


# --- A store of version 2 --------------------------------------------------------------------


def test_a_version_2_store_migrates_and_reads_its_old_hand_off_in_occurrence_order(tmp_path):
    path, a = build_store(tmp_path / "old")
    take_step_3_back(path)
    assert Store.migrate(path) == (SCHEMA_BEFORE, SCHEMA_NOW)
    assert scalar(path, "SELECT COUNT(*) FROM harness_handed WHERE position IS NOT NULL") == 0
    with Store.open(path) as store:
        assert store.schema_version == SCHEMA_NOW
        assert keys(store.harness_log("order-job").handed) == keys(lines(a, IN_ID_ORDER))
        store.check_integrity()


@pytest.mark.parametrize("numbers", [HANDED_OUT_OF_ID_ORDER, IN_ID_ORDER])
def test_the_job_of_a_log_without_positions_is_compared_as_a_set(tmp_path, numbers):
    path, a = build_store(tmp_path / "set")
    take_step_3_back(path)
    with Store.open(path) as store:  # migrates
        register_again(store, path.parent, a, numbers)
        assert keys(store.harness_log("order-job").handed) == keys(lines(a, IN_ID_ORDER))


def test_a_log_written_after_the_migration_has_its_positions(tmp_path):
    path, a = build_store(tmp_path / "after")
    take_step_3_back(path)
    with Store.open(path) as store:
        handed = lines(a, HANDED_OUT_OF_ID_ORDER)
        ww.record(store, path.parent / "N.jsonl", ["new"], "gen/N", v.ROLE_INTERMEDIATE,
                  ww.step("generate", "N"), [a], [ww.generation_edge(1, "new-job", handed, handed)])
    with Store.open(path) as store:
        assert keys(store.harness_log("new-job").handed) == keys(lines(a, HANDED_OUT_OF_ID_ORDER))


def test_a_migrated_store_holds_the_schema_of_a_new_one(tmp_path):
    path = build_blend_store(tmp_path / "old.db")
    take_step_3_back(path)
    Store.migrate(path)
    Store.create(tmp_path / "new.db", normaliser=ww.NORMALISER).close()
    assert schema_of(path) == schema_of(tmp_path / "new.db")


def test_a_version_2_store_with_one_digest_for_its_blend_migrates_and_then_refuses_a_second(tmp_path):
    path = build_blend_store(tmp_path / "blend.db")
    take_step_3_back(path)
    assert Store.migrate(path) == (SCHEMA_BEFORE, SCHEMA_NOW)
    with pytest.raises(sqlite3.IntegrityError):
        add_blend_row(path, "b" * 64)


@pytest.mark.parametrize("second", ["b" * 64, "a" * 64], ids=["another-digest", "the-same-digest"])
def test_a_version_2_store_with_two_rows_for_one_blend_array_is_refused_with_a_reason(tmp_path, second):
    path = build_blend_store(tmp_path / "blend.db")
    take_step_3_back(path)
    add_blend_row(path, second)
    before = schema_of(path)
    with pytest.raises(errors.CacheDigestConflict) as caught:
        Store.migrate(path)
    fault = caught.value
    assert (fault.name, fault.exit_code) == (NAME_CACHE, EXIT_LOUD_FAULT)
    assert (fault.context["segment"], fault.context["kind"], fault.context["held"]) == (
        "G/seg-1", v.CACHE_DATASET_INDEX, 2)
    assert "schema version 3" in fault.detail
    # The store is as it was: the version, the rows and the schema.
    assert scalar(path, "SELECT value FROM meta WHERE key = 'schema_version'") == str(SCHEMA_BEFORE)
    assert scalar(path, "SELECT COUNT(*) FROM segment_cache_digest") == 2
    assert schema_of(path) == before
    with pytest.raises(errors.CacheDigestConflict):
        Store.open(path)
    with pytest.raises(errors.Refused, match="migrated"):
        Store.open(path, readonly=True)


def test_a_version_1_store_with_two_rows_for_one_blend_array_keeps_the_version_2_it_reached(tmp_path):
    path = build_blend_store(tmp_path / "blend.db")
    take_step_3_back(path)
    take_step_2_back(path)
    add_blend_row(path, "b" * 64)
    with pytest.raises(errors.CacheDigestConflict):
        Store.migrate(path)
    assert scalar(path, "SELECT value FROM meta WHERE key = 'schema_version'") == str(SCHEMA_BEFORE)
    assert scalar(path, "SELECT COUNT(*) FROM segment_cache_digest") == 2


def test_a_version_1_store_with_one_digest_for_its_blend_migrates_by_two_steps(tmp_path):
    path = build_blend_store(tmp_path / "blend.db")
    take_step_3_back(path)
    take_step_2_back(path)
    assert Store.migrate(path) == (1, SCHEMA_NOW)


def test_the_command_line_refuses_such_a_store_with_the_named_fault(tmp_path):
    path = build_blend_store(tmp_path / "blend.db")
    take_step_3_back(path)
    add_blend_row(path, "b" * 64)
    code, printed = run_cli("migrate", "--store", str(path))
    assert (code, printed["error"]) == (EXIT_LOUD_FAULT, NAME_CACHE)
    assert printed["context"]["segment"] == "G/seg-1"


def test_the_command_line_migrates_a_version_2_store_by_two_steps(tmp_path):
    path, _ = build_store(tmp_path / "cli")
    take_step_3_back(path)
    code, said = run_cli("migrate", "--store", str(path))
    assert code == 0
    assert {k: said[k] for k in ("schema_version_found", "schema_version", "steps_run")} == {
        "schema_version_found": SCHEMA_BEFORE, "schema_version": SCHEMA_NOW, "steps_run": 3}


# --- The digest of a run record is the digest of the text the store holds -------------------


def test_the_digest_of_a_run_record_with_number_keys_is_the_digest_of_the_stored_text(tmp_path):
    path = tmp_path / "keys.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store, "a" * 64, extra={9: "nine", 10: "ten"})
        store.check_integrity()
        registered = store.segment("G/seg-1").run_record_digest
    text = scalar(path, "SELECT run_record FROM run_segment WHERE name = 'G/seg-1'")
    assert registered == hashlib.sha256(text.encode("utf-8")).hexdigest()
