"""The mark a stored harness log carries of whether its handed texts were compared
(docs/store.md "Opening a store" and "The write methods"; this file held the pins of the
hand-off unit that said no mark existed and the schema stayed at 3).

- The schema version is 4: a store that registered a compared hand-off is at version 4 and
  its log is marked compared; `Store.migrate` finds nothing to do.
- A confined log that was stored without any handed text (a hand-built log, a log read back
  from its file) is not marked: it establishes no bound and a source outside its hand-off
  answers "may contain"; opening it changes nothing.
- The handed texts are never stored: the log of a job whose output was registered,
  read from the store, carries none and equals the log read from the job's file.

The store of the second test is made by the write methods with a hand-built confined
log; the store of the first and third is made through the real harness and the real
capture wrappers.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

from waterwhite import capture, coverage, harness, requests
from waterwhite import vocabulary as v
from waterwhite.harness import Handed, JobOutput, run_job
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

SCHEMA = 5  # the version a store is at now; the hand-off mark is step 4
JOB = "job-1"
NO_TEXTS = "the log has no handed_texts field"
TEXT_A = "The repair workshop meets on the first Tuesday of the month."
OUTPUT_TEXT = "A short text that a job wrote."


def build_confined_store(base: Path) -> Path:
    """A store with sources A and B and one output of the job "job-1", handed A's first
    record, with a hand-built confined log that lists the one read of it. B is outside the
    hand-off. Closed on return."""
    base.mkdir()
    store = Store.create(base / "s.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN])
    a = ww.ingest(store, base / "A.jsonl", ["alpha", "beta"], "A", "src/A")
    ww.ingest(store, base / "B.jsonl", ["gamma"], "B", "src/B")
    handed = [ww.ref(a, 1)]
    edge = ww.generation_edge(1, JOB, handed, handed)
    edge = dataclasses.replace(edge, handed_texts=())  # a log whose texts were never shown
    ww.record(store, base / "Q.jsonl", ["summary"], "gen/Q", v.ROLE_INTERMEDIATE,
              ww.step("generate", "Q"), [a], [edge])
    store.close()
    return base / "s.db"


def answer(store: Store, source: str) -> dict:
    return coverage.answer(store, "gen/Q", requests.withdraw(store, source)).to_json()


def register_compared_hand_off(tmp_path: Path, store: Store):
    """One source, one job handed A's record with A's text, its output registered.
    Returns the reference, the result and the path of the job's log file."""
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources/A.jsonl").write_bytes((json.dumps({"id": "A-001", "text": TEXT_A}) + "\n").encode("utf-8"))
    src = capture.ingest_jsonl(
        store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    ref = capture.occurrence(src, 1)
    log_path = tmp_path / "logs/job-1.jsonl"
    result = run_job(
        lambda ctx: JobOutput(text=OUTPUT_TEXT, used=(ctx.handed()[0],)),
        job_name=JOB, handed=[Handed(ref, TEXT_A)], log_path=log_path)
    step = capture.step_ref("generate", "1", {"style": "plain"})
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen/O.jsonl").write_bytes((json.dumps({"id": "O-1", "text": OUTPUT_TEXT}) + "\n").encode("utf-8"))
    capture.record_jsonl(
        store, tmp_path / "gen/O.jsonl", name="gen/O", role=capture.ROLE_INTERMEDIATE, step=step,
        inputs=[src], edges=capture.generation_edges(result, [1], step))
    return ref, result, log_path


def test_a_store_that_registered_a_compared_hand_off_is_at_schema_4_and_its_log_is_marked(tmp_path):
    path = tmp_path / "evidence.db"
    store = Store.create(path, normaliser=ww.NORMALISER)
    try:
        register_compared_hand_off(tmp_path, store)
        assert store.meta()["schema_version"] == str(SCHEMA)
        assert store.schema_version == SCHEMA
        assert store.harness_log(JOB).hand_off_checked is True
    finally:
        store.close()
    assert Store.migrate(path) == (SCHEMA, SCHEMA)


def snapshot(path: Path) -> tuple[list, dict]:
    """The SQL of every object of the file and the number of rows of every table."""
    raw = sqlite3.connect(path)
    try:
        objects = raw.execute("SELECT type, name, sql FROM sqlite_master ORDER BY type, name").fetchall()
        tables = [name for kind, name, _ in objects if kind == "table" and not name.startswith("sqlite_")]
        return objects, {t: raw.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    finally:
        raw.close()


def test_a_stored_confined_log_without_texts_is_not_marked_establishes_no_bound_and_opening_changes_nothing(tmp_path):
    path = build_confined_store(tmp_path / "old")
    before = snapshot(path)
    assert Store.migrate(path) == (SCHEMA, SCHEMA)
    with Store.open(path) as store:
        assert store.schema_version == SCHEMA
        log = store.harness_log(JOB)
        assert log is not None and log.bound_stated() and not log.hand_off_checked
        assert not log.bound_established()
        outside = answer(store, "B")
        handed = answer(store, "A")
    assert (outside["answer"], outside["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    assert handed["answer"] == "contains"
    assert snapshot(path) == before


def test_the_stored_log_of_a_registered_job_carries_no_texts_and_equals_the_log_of_its_file(tmp_path):
    store = Store.create(tmp_path / "evidence.db", normaliser=ww.NORMALISER)
    try:
        ref, result, log_path = register_compared_hand_off(tmp_path, store)
        assert getattr(result.log, "handed_texts", NO_TEXTS) == ((ref, TEXT_A),)
        stored = store.harness_log(JOB)
        assert getattr(stored, "handed_texts", NO_TEXTS) == ()
        assert stored == harness.read_log(log_path)
    finally:
        store.close()
