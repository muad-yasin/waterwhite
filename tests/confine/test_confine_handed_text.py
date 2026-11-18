"""A reference handed with the text of another record, run in the real sandbox
(docs/confine.md section 5: the hand-off is compared with the record it names when the
output is registered, and a text that is not that record refuses the registration).

The same hand-off through the in-process harness is in
tests/capture/test_handed_and_reads_reproduction.py and
tests/capture/test_hand_off_text_compared_at_registration.py. Here the job runs in the
sandbox, which holds no store either, so it runs; the refusal is the registration's:
"invalid record", naming the job and the reference, and nothing is written."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, errors
from waterwhite.confine import python_program, run_confined_job
from waterwhite.harness import Handed
from waterwhite.store import Store

pytestmark = pytest.mark.sandbox

NORMALISER = "control-chars/1"
TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build(tmp_path: Path) -> SimpleNamespace:
    w = SimpleNamespace(folder=tmp_path, db=tmp_path / "evidence.db")
    w.store = Store.create(w.db, normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    return w


def row_counts(db: Path) -> dict[str, int]:
    """The number of rows of every table of the store file."""
    raw = sqlite3.connect(db)
    try:
        tables = [r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {table: raw.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
    finally:
        raw.close()


@pytest.fixture
def wrong_text_in_the_sandbox(sandbox, jobs, tmp_path):
    w = build(tmp_path)
    w.counts_before = row_counts(w.db)
    w.result = run_confined_job(
        python_program(jobs / "paraphrase_one.py"), job_name="paraphrase",
        handed=[Handed(w.ref_a, TEXT_B)], log_path=tmp_path / "logs/wrong.jsonl", timeout_seconds=30)
    step = capture.step_ref("generate", "1", {"style": "plain"})
    out = tmp_path / "gen/W.jsonl"
    write_rows(out, [{"id": "W", "text": w.result.outputs[0].text}])
    try:
        w.out = capture.record_jsonl(
            w.store, out, name="gen/W", role=capture.ROLE_INTERMEDIATE, step=step,
            inputs=[w.src_a, w.src_b], edges=capture.generation_edges(w.result, [1], step))
        w.refusal = None
    except errors.WaterwhiteError as error:
        w.out, w.refusal = None, error
    yield w
    w.store.close()


def test_the_sandbox_hands_the_job_the_text_it_was_given_under_the_reference(wrong_text_in_the_sandbox):
    assert wrong_text_in_the_sandbox.result.outputs[0].text == TEXT_B.upper()


def test_registering_the_output_of_a_sandbox_run_handed_the_wrong_text_is_refused(wrong_text_in_the_sandbox):
    w = wrong_text_in_the_sandbox
    assert w.refusal is not None, "the registration was accepted"
    assert w.refusal.name == "invalid record", str(w.refusal)
    assert "paraphrase" in str(w.refusal), str(w.refusal)
    assert w.ref_a.key in str(w.refusal), str(w.refusal)


def test_a_refused_sandbox_hand_off_leaves_no_artifact_no_edge_and_no_log(wrong_text_in_the_sandbox):
    w = wrong_text_in_the_sandbox
    assert w.store.artifact("gen/W") is None
    assert w.store.harness_log("paraphrase") is None
    assert row_counts(w.db) == w.counts_before
