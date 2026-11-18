"""A text a job was handed is compared with its record wherever it travels to the
registration, and a record that holds no text takes no text (docs/capture.md section 3,
docs/store.md "The write methods"), through the real harness, capture wrappers and store.

The sentences the expectations rest on:

- the texts a job was handed travel on its log (`HarnessLog.handed_texts`) and on the
  generation edge built from it (`EdgeRecord.handed_texts`); the store compares every text
  of both with the record its reference names, each once, so an edge built by hand that
  carries a log with texts is compared as one built by `generation_edge` is, and an edge
  and a log that differ are refused unless each text is the record after the normaliser;
- a text of the log that lies outside the log's hand-off, or is not a (reference, text)
  pair, is refused as a text of the edge is;
- a record that has no fingerprint (a line with no text field, or one empty after the
  normaliser) holds no text: a handed text that is empty after the normaliser is that
  record, and any other text is not, so it is refused with "invalid record", naming the job
  and the reference, and nothing is written.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, errors
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.store import Store

NORMALISER = "control-chars/1"
TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."
STEP = capture.step_ref("generate", "1", {"style": "plain"})
REFUSAL = "invalid record"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build(tmp_path: Path, rows_a: list[dict] | None = None) -> SimpleNamespace:
    w = SimpleNamespace(folder=tmp_path, db=tmp_path / "evidence.db")
    w.store = Store.create(w.db, normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", rows_a if rows_a is not None else [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    w.ref_b = capture.occurrence(w.src_b, 1)
    return w


def paraphrase(ctx):
    (only,) = ctx.handed()
    return JobOutput(text=ctx.read(only).upper(), used=(only,))


def run(w, handed, tag="run"):
    return run_job(paraphrase, job_name="paraphrase", handed=handed, log_path=w.folder / f"logs/{tag}.jsonl")


def edge_of(result):
    (edge,) = capture.generation_edges(result, [1], STEP)
    return edge


def register(w, edge, name="W"):
    """None when registered, otherwise the error the registration raised."""
    path = w.folder / f"gen/{name}.jsonl"
    write_rows(path, [{"id": f"{name}-1", "text": "An output."}])
    try:
        capture.record_jsonl(
            w.store, path, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=STEP,
            inputs=[w.src_a, w.src_b], edges=[edge])
    except errors.WaterwhiteError as error:
        return error
    return None


def row_counts(db: Path) -> dict[str, int]:
    raw = sqlite3.connect(db)
    try:
        tables = [r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: raw.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}
    finally:
        raw.close()


def assert_refused_naming(error, ref) -> None:
    assert error is not None, "the registration was accepted"
    assert error.name == REFUSAL, str(error)
    assert "paraphrase" in str(error), str(error)
    assert ref.key in str(error), str(error)


def log_only(edge):
    """The edge a person builds by hand around a returned log: the log carries the texts,
    the edge carries none of its own."""
    return dataclasses.replace(edge, handed_texts=())


# --- the texts on the log, with none on the edge ------------------------------------------------


def test_a_hand_built_edge_whose_log_holds_a_wrong_text_is_refused_and_nothing_is_written(tmp_path):
    w = build(tmp_path)
    try:
        edge = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_B)])))
        assert edge.handed_texts == () and edge.harness_log.handed_texts
        before = row_counts(w.db)
        assert_refused_naming(register(w, edge), w.ref_a)
        assert w.store.artifact("gen/W") is None
        assert row_counts(w.db) == before
    finally:
        w.store.close()


def test_a_hand_built_edge_whose_log_holds_the_right_text_registers(tmp_path):
    w = build(tmp_path)
    try:
        edge = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_A)])))
        refusal = register(w, edge)
        assert refusal is None, str(refusal)
        assert w.store.artifact("gen/W") is not None
        # the control: the other text handed under the same reference is refused here
        other = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_B)], tag="other")))
        other = dataclasses.replace(other, harness_log=dataclasses.replace(other.harness_log, job_name="paraphrase-2"),
                                    job_name="paraphrase-2")
        assert_refused_naming(register(w, other, name="C"), w.ref_a)
    finally:
        w.store.close()


# --- the edge and the log both carry texts and they differ -----------------------------------


@pytest.mark.parametrize("which", ["edge-right-log-wrong", "log-right-edge-wrong"])
def test_an_edge_and_a_log_that_differ_are_refused(tmp_path, which):
    w = build(tmp_path)
    try:
        edge = edge_of(run(w, [Handed(w.ref_a, TEXT_A)]))
        right, wrong = ((w.ref_a, TEXT_A),), ((w.ref_a, TEXT_B),)
        edge_texts, log_texts = (right, wrong) if which == "edge-right-log-wrong" else (wrong, right)
        edge = dataclasses.replace(
            edge, handed_texts=edge_texts, harness_log=dataclasses.replace(edge.harness_log, handed_texts=log_texts))
        assert_refused_naming(register(w, edge), w.ref_a)
        assert w.store.artifact("gen/W") is None
    finally:
        w.store.close()


def test_an_edge_and_a_log_that_differ_only_by_what_the_normaliser_removes_register(tmp_path):
    w = build(tmp_path)
    try:
        edge = edge_of(run(w, [Handed(w.ref_a, TEXT_A)]))
        edge = dataclasses.replace(
            edge, handed_texts=((w.ref_a, TEXT_A),),
            harness_log=dataclasses.replace(edge.harness_log, handed_texts=((w.ref_a, "\x07  " + TEXT_A + "\r\n"),)))
        refusal = register(w, edge)
        assert refusal is None, str(refusal)
    finally:
        w.store.close()


def test_a_text_of_the_log_outside_its_hand_off_is_refused(tmp_path):
    w = build(tmp_path)
    try:
        edge = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_A)])))
        edge = dataclasses.replace(
            edge, harness_log=dataclasses.replace(edge.harness_log, handed_texts=((w.ref_b, TEXT_B),)))
        error = register(w, edge)
        assert error is not None and error.name == REFUSAL, str(error)
        assert "outside the hand-off" in str(error), str(error)
    finally:
        w.store.close()


# --- a record that holds no text ----------------------------------------------------------------

NO_TEXT_RECORD = {
    "no-text-field": [{"id": "A-001", "text": TEXT_A}, {"id": "A-002"}],
    "text-empty-after-the-normaliser": [{"id": "A-001", "text": TEXT_A}, {"id": "A-002", "text": "\x07 \x00"}],
}


@pytest.mark.parametrize("carrier", ["edge-and-log", "log-only"])
@pytest.mark.parametrize("case", sorted(NO_TEXT_RECORD))
def test_a_text_that_is_not_empty_handed_under_a_record_with_no_text_is_refused(tmp_path, case, carrier):
    w = build(tmp_path, rows_a=NO_TEXT_RECORD[case])
    try:
        no_text = capture.occurrence(w.src_a, 2)
        edge = edge_of(run(w, [Handed(no_text, TEXT_B)]))
        edge = edge if carrier == "edge-and-log" else log_only(edge)
        before = row_counts(w.db)
        assert_refused_naming(register(w, edge), no_text)
        assert w.store.artifact("gen/W") is None
        assert row_counts(w.db) == before
    finally:
        w.store.close()


@pytest.mark.parametrize("handed_text", ["", "\x07 \x00", " \r\n"])
@pytest.mark.parametrize("case", sorted(NO_TEXT_RECORD))
def test_a_text_that_is_empty_after_the_normaliser_handed_under_a_record_with_no_text_registers(
        tmp_path, case, handed_text):
    w = build(tmp_path, rows_a=NO_TEXT_RECORD[case])
    try:
        no_text = capture.occurrence(w.src_a, 2)
        refusal = register(w, edge_of(run(w, [Handed(no_text, handed_text)])))
        assert refusal is None, str(refusal)
        assert w.store.artifact("gen/W") is not None
    finally:
        w.store.close()


# --- the shape of the texts of the log ------------------------------------------------------------


@pytest.mark.parametrize("shape", ["three-items", "text-is-not-text", "a-bare-reference", "not-a-reference"])
def test_a_text_of_the_log_that_is_not_a_reference_and_text_pair_is_refused(tmp_path, shape):
    w = build(tmp_path)
    try:
        edge = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_A)])))
        malformed = {
            "three-items": (w.ref_a, TEXT_A, "another item"),
            "text-is-not-text": (w.ref_a, b"bytes are not a text"),
            "a-bare-reference": w.ref_a,
            "not-a-reference": ("A-001", TEXT_A),
        }[shape]
        edge = dataclasses.replace(edge, harness_log=dataclasses.replace(edge.harness_log, handed_texts=(malformed,)))
        before = row_counts(w.db)
        error = register(w, edge)
        assert error is not None, "the registration was accepted"
        assert error.name == REFUSAL, str(error)
        assert row_counts(w.db) == before
    finally:
        w.store.close()


def test_the_texts_of_the_log_that_are_not_a_tuple_are_refused(tmp_path):
    w = build(tmp_path)
    try:
        edge = log_only(edge_of(run(w, [Handed(w.ref_a, TEXT_A)])))
        edge = dataclasses.replace(
            edge, harness_log=dataclasses.replace(edge.harness_log, handed_texts=[(w.ref_a, TEXT_A)]))
        error = register(w, edge)
        assert error is not None and error.name == REFUSAL, str(error)
        assert "tuple" in str(error), str(error)
    finally:
        w.store.close()


def test_a_lone_surrogate_handed_under_a_record_with_no_text_is_refused_not_a_crash(tmp_path):
    w = build(tmp_path, rows_a=NO_TEXT_RECORD["no-text-field"])
    try:
        no_text = capture.occurrence(w.src_a, 2)
        assert_refused_naming(register(w, edge_of(run(w, [Handed(no_text, "\ud800")]))), no_text)
        assert w.store.artifact("gen/W") is None
    finally:
        w.store.close()
