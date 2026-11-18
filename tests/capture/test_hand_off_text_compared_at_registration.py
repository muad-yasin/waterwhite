"""A hand-off whose text is not the record it names is refused when the output is
registered (docs/capture.md section 3 "Handed(ref, text)", docs/store.md "The write
methods"), driven through the real harness, the real capture wrappers, the real store
and the real coverage engine. Nothing is stood in for.

The sentences the expectations rest on:

- `run_job` holds no store, so it runs a job handed the reference of one record with
  the text of another; the log it returns keeps the handed texts in the hand-off order,
  outside the log's equality, its representation and its digest, and nothing of them is
  stored; the comparison is made where the output and its generation edge are
  registered, which compares the fingerprint of each handed text, after the store's
  normaliser, with the fingerprint of the record its reference names;
- a text that is not that record and a text that differs after the normaliser each
  refuse the registration with "invalid record", naming the job and the reference;
  nothing is written, the refusal is one transaction;
- a text that is the record after the normaliser (docs/capture.md "control-chars/1, in
  full": control characters removed, the ends stripped) is the record; a format
  character, white space inside the text or another text is not;
- a log that carries no texts (read back from its file, or built by hand) has nothing to
  compare: it registers and keeps its bound exactly as it did, the hand-off being the
  caller's statement; a record that has no fingerprint holds no text, so a text that is not
  empty after the normaliser handed under it is refused, and an empty one registers;
- an in-process log establishes no bound whatever it was handed: a source outside the
  hand-off answers "may contain" naming "upper bound not established".

The tests that hold a text that matches also register a text that does not, under
another job name, in the same store: a test that passed by accepting everything would
fail there. The log's field is read with getattr, so a build without it fails at the
comparison and not at a missing name.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, coverage, errors, harness, requests
from waterwhite import vocabulary as v
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.records import HarnessLog, HarnessRead
from waterwhite.store import Store

NORMALISER = "control-chars/1"

TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."
OUTPUT_TEXT = "A short text that a job wrote."

STEP = capture.step_ref("generate", "1", {"style": "plain"})
REFUSAL = "invalid record"
NO_TEXTS = "the log has no handed_texts field"

# Texts that are the record of TEXT_A after the normaliser: control characters are
# removed wherever they stand, then the tab, line feed, space and the other white space
# at the ends is stripped.
SAME_AFTER_THE_NORMALISER = {
    "controls-at-the-ends": "\x07  " + TEXT_A + "\r\n\x00",
    "control-inside": TEXT_A[:10] + "\x00" + TEXT_A[10:],
}
# Texts that are not: a format character is kept, white space inside is not changed,
# and another record's text is another text.
DIFFERENT_AFTER_THE_NORMALISER = {
    "format-character-at-the-end": TEXT_A + "‍",
    "byte-order-mark-at-the-start": "﻿" + TEXT_A,
    "white-space-inside": TEXT_A.replace("first Tuesday", "first  Tuesday"),
    "another-record": TEXT_B,
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build_sources(tmp_path: Path, rows_a: list[dict] | None = None) -> SimpleNamespace:
    """Two sources, ingested the way a team ingests them; A of one record unless rows
    are given."""
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
    text = ctx.read(only)
    return JobOutput(text=text.upper(), used=(only,))


def mix(ctx):
    """Reads both handed records and states only the first as used."""
    first, second = ctx.handed()
    ctx.read(first)
    ctx.read(second)
    return JobOutput(text="A bakery and a workshop share a street.", used=(first,))


def register_file(w: SimpleNamespace, name: str, edges, lines: int = 1) -> str:
    step_inputs = [w.src_a, w.src_b]
    path = w.folder / f"gen/{name}.jsonl"
    write_rows(path, [{"id": f"{name}-{n}", "text": OUTPUT_TEXT} for n in range(1, lines + 1)])
    return capture.record_jsonl(
        w.store, path, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=STEP,
        inputs=step_inputs, edges=edges)


def try_register(w: SimpleNamespace, result, name: str):
    """None when the output is registered, otherwise the error the registration raised."""
    try:
        register_file(w, name, capture.generation_edges(result, [1], STEP))
    except errors.WaterwhiteError as error:
        return error
    return None


def run(w: SimpleNamespace, handed, job=paraphrase, job_name: str = "paraphrase", tag: str = "run"):
    return run_job(job, job_name=job_name, handed=handed, log_path=w.folder / f"logs/{tag}.jsonl")


def ask(w: SimpleNamespace, source: str, artifact: str, label: str) -> dict:
    answer = coverage.answer(w.store, artifact, requests.withdraw(w.store, source)).to_json()
    folder = w.folder / "answers"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return answer


def row_counts(db: Path) -> dict[str, int]:
    """The number of rows of every table of the store file."""
    raw = sqlite3.connect(db)
    try:
        tables = [r[0] for r in raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {table: raw.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] for table in tables}
    finally:
        raw.close()


def assert_refused_naming(error, job: str, ref) -> None:
    assert error is not None, "the registration was accepted"
    assert error.name == REFUSAL, str(error)
    assert job in str(error), str(error)
    assert ref.key in str(error), str(error)


def assert_other_text_is_refused(w: SimpleNamespace, tag: str) -> None:
    """The control of a test that registered a text that matches: the same store refuses
    the text of the other source handed under A's reference, by another job."""
    job = f"control-{tag}"
    result = run(w, [Handed(w.ref_a, TEXT_B)], job_name=job, tag=job)
    assert_refused_naming(try_register(w, result, f"C-{tag}"), job, w.ref_a)


# --- (a) the reference of A with the text of B --------------------------------------------


@pytest.fixture
def wrong_text_handed(tmp_path):
    w = build_sources(tmp_path)
    w.counts_before = row_counts(w.db)
    w.result = run(w, [Handed(w.ref_a, TEXT_B)])
    w.refusal = try_register(w, w.result, "W")
    yield w
    w.store.close()


def test_registering_the_output_of_a_wrong_text_hand_off_is_refused_naming_job_and_reference(wrong_text_handed):
    w = wrong_text_handed
    assert_refused_naming(w.refusal, "paraphrase", w.ref_a)


def test_a_refused_wrong_text_hand_off_leaves_no_artifact_no_edge_and_no_log(wrong_text_handed):
    w = wrong_text_handed
    assert w.store.artifact("gen/W") is None
    assert w.store.harness_log("paraphrase") is None
    assert row_counts(w.db) == w.counts_before


def test_the_single_edge_builder_carries_the_texts_so_its_wrong_text_is_refused_too(tmp_path):
    """`capture.generation_edge` (one edge) is the form the replica's hooks call; it
    must carry what `generation_edges` carries. The right text registers first."""
    w = build_sources(tmp_path)
    try:
        right = run(w, [Handed(w.ref_a, TEXT_A)], tag="right")
        register_file(w, "O", [capture.generation_edge(1, right.outputs[0], right.log, STEP)])
        wrong = run(w, [Handed(w.ref_a, TEXT_B)], job_name="single-form", tag="wrong")
        try:
            register_file(w, "W", [capture.generation_edge(1, wrong.outputs[0], wrong.log, STEP)])
            refusal = None
        except errors.WaterwhiteError as error:
            refusal = error
        assert_refused_naming(refusal, "single-form", w.ref_a)
    finally:
        w.store.close()


# --- (c) the right text -------------------------------------------------------------------


def test_the_right_text_registers_and_an_in_process_log_still_gives_no_bound(tmp_path):
    w = build_sources(tmp_path)
    try:
        result = run(w, [Handed(w.ref_a, TEXT_A)])
        assert try_register(w, result, "O") is None
        outside = ask(w, "B", "gen/O", "right-withdraw-B")
        assert outside["answer"] == "may contain"
        assert outside["names"] == "upper bound not established"
        assert outside["failing"]["job"] == "paraphrase"
        handed = ask(w, "A", "gen/O", "right-withdraw-A")
        assert handed["answer"] == "contains"
        assert_other_text_is_refused(w, "right")
    finally:
        w.store.close()


def test_a_hand_off_of_two_records_with_the_right_texts_registers_and_a_wrong_first_text_does_not(tmp_path):
    w = build_sources(tmp_path)
    try:
        result = run(w, [Handed(w.ref_a, TEXT_A), Handed(w.ref_b, TEXT_B)], job=mix, job_name="mixer", tag="ok")
        assert try_register(w, result, "O") is None
        swapped = run(w, [Handed(w.ref_a, TEXT_B), Handed(w.ref_b, TEXT_B)], job=mix, job_name="mixer-2", tag="bad")
        assert_refused_naming(try_register(w, swapped, "W"), "mixer-2", w.ref_a)
    finally:
        w.store.close()


def test_each_handed_text_is_compared_so_a_wrong_text_for_the_record_not_stated_is_refused(tmp_path):
    """The output states A only; the text handed under B's reference is A's."""
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        result = run(w, [Handed(w.ref_a, TEXT_A), Handed(w.ref_b, TEXT_A)], job=mix, job_name="mixer")
        refusal = try_register(w, result, "O")
        assert_refused_naming(refusal, "mixer", w.ref_b)
        assert row_counts(w.db) == before
    finally:
        w.store.close()


# --- (d) the comparison is after the normaliser ---------------------------------------------


@pytest.mark.parametrize("case", sorted(SAME_AFTER_THE_NORMALISER))
def test_a_text_that_is_the_record_after_the_normaliser_matches(tmp_path, case):
    w = build_sources(tmp_path)
    try:
        result = run(w, [Handed(w.ref_a, SAME_AFTER_THE_NORMALISER[case])])
        refusal = try_register(w, result, "O")
        assert refusal is None, str(refusal)
        assert_other_text_is_refused(w, case)
    finally:
        w.store.close()


@pytest.mark.parametrize("case", sorted(DIFFERENT_AFTER_THE_NORMALISER))
def test_a_text_that_differs_from_the_record_after_the_normaliser_is_refused(tmp_path, case):
    w = build_sources(tmp_path)
    try:
        result = run(w, [Handed(w.ref_a, DIFFERENT_AFTER_THE_NORMALISER[case])])
        assert_refused_naming(try_register(w, result, "O"), "paraphrase", w.ref_a)
    finally:
        w.store.close()


# --- (e) a log that carries no texts is the caller's statement -------------------------------


@pytest.mark.parametrize("handed_text", [TEXT_A, TEXT_B], ids=["right-text", "wrong-text"])
def test_a_log_read_back_from_its_file_registers_and_keeps_its_bound_as_before(tmp_path, handed_text):
    """The log file holds references and never a text, so the result built from it has
    no texts to compare: it registers whichever text the job was handed, and its answers
    are the ones of an in-process log, which establishes no bound."""
    w = build_sources(tmp_path)
    try:
        ran = run(w, [Handed(w.ref_a, handed_text)])
        read_back = harness.JobResult(ran.outputs, harness.read_log(w.folder / "logs/run.jsonl"))
        refusal = try_register(w, read_back, "O")
        assert refusal is None, str(refusal)
        outside = ask(w, "B", "gen/O", "read-back-withdraw-B")
        assert (outside["answer"], outside["names"]) == ("may contain", "upper bound not established")
        handed = ask(w, "A", "gen/O", "read-back-withdraw-A")
        assert handed["answer"] == "contains"
    finally:
        w.store.close()


@pytest.mark.parametrize("texts_first", [True, False], ids=["texts-first", "read-back-first"])
def test_the_same_jobs_log_registered_once_with_texts_and_once_read_back_is_one_log(tmp_path, texts_first):
    """The texts are no part of the log's identity: the second registration of the job
    finds the log it already holds, whichever of the two forms came first."""
    w = build_sources(tmp_path)
    try:
        ran = run(w, [Handed(w.ref_a, TEXT_A)])
        read_back = harness.JobResult(ran.outputs, harness.read_log(w.folder / "logs/run.jsonl"))
        first, second = (ran, read_back) if texts_first else (read_back, ran)
        assert try_register(w, first, "O") is None
        refusal = try_register(w, second, "P")
        assert refusal is None, str(refusal)
        assert w.store.harness_log("paraphrase") == read_back.log
    finally:
        w.store.close()


def test_a_log_built_by_hand_registers_and_a_confined_one_establishes_no_bound_until_its_texts_are_shown(tmp_path):
    w = build_sources(tmp_path)
    try:
        log = HarnessLog(
            job_name="built-by-hand", log_digest="digest-of-no-file", handed=(w.ref_a,),
            reads=(HarnessRead(occurrence=w.ref_a),), model="none", harness=v.HARNESS_CONFINED)
        result = harness.JobResult(outputs=(JobOutput(text=OUTPUT_TEXT, used=(w.ref_a,)),), log=log)
        refusal = try_register(w, result, "H")
        assert refusal is None, str(refusal)
        outside = ask(w, "B", "gen/H", "hand-built-withdraw-B")
        assert outside["answer"] == "may contain" and outside["names"] == "upper bound not established"
        handed = ask(w, "A", "gen/H", "hand-built-withdraw-A")
        assert handed["answer"] == "contains"
        stored = w.store.harness_log("built-by-hand")
        assert stored.bound_stated() and not stored.hand_off_checked and not stored.bound_established()
        # the texts shown later: the clean answer returns (docs/store.md "The write methods")
        w.store.verify_hand_off("built-by-hand", [(w.ref_a, TEXT_A)])
        assert ask(w, "B", "gen/H", "hand-built-withdraw-B-after")["answer"] == "does not contain"
        assert w.store.harness_log("built-by-hand").bound_established()
    finally:
        w.store.close()


# --- the hand-off texts on the log (design: carried, never stored, never compared by equality) --


def test_run_job_gives_its_log_the_handed_texts_in_hand_off_order(tmp_path):
    w = build_sources(tmp_path)
    try:
        ran = run(w, [Handed(w.ref_b, TEXT_B), Handed(w.ref_a, TEXT_A)], job=mix, job_name="mixer")
        assert getattr(ran.log, "handed_texts", NO_TEXTS) == ((w.ref_b, TEXT_B), (w.ref_a, TEXT_A))
    finally:
        w.store.close()


def test_the_handed_texts_take_no_part_in_the_logs_equality_representation_or_digest(tmp_path):
    w = build_sources(tmp_path)
    try:
        ran = run(w, [Handed(w.ref_a, TEXT_A)])
        assert getattr(ran.log, "handed_texts", NO_TEXTS) == ((w.ref_a, TEXT_A),)
        from_file = harness.read_log(w.folder / "logs/run.jsonl")
        assert ran.log == from_file
        assert ran.log.log_digest == from_file.log_digest
        assert TEXT_A not in repr(ran.log)
    finally:
        w.store.close()


def test_a_log_read_from_its_file_carries_no_texts(tmp_path):
    w = build_sources(tmp_path)
    try:
        run(w, [Handed(w.ref_a, TEXT_A)])
        from_file = harness.read_log(w.folder / "logs/run.jsonl")
        assert getattr(from_file, "handed_texts", NO_TEXTS) == ()
    finally:
        w.store.close()


# --- (f) a handed reference whose record has no fingerprint ----------------------------------


NO_FINGERPRINT = {
    "no-text-field": ([{"id": "A-001", "text": TEXT_A}, {"id": "A-002"}], TEXT_A, "\x07 \x00"),
    "text-empty-after-the-normaliser": ([{"id": "A-001", "text": TEXT_A}, {"id": "A-002", "text": "\x07 \x00"}],
                                        TEXT_A, "\x07 \x00"),
}


@pytest.mark.parametrize("case", sorted(NO_FINGERPRINT))
def test_a_text_handed_under_a_record_with_no_fingerprint_is_refused_when_it_is_not_empty(tmp_path, case):
    """A record with no fingerprint holds no text, so a text that is not empty after the
    normaliser is not that record: refused naming the job and the reference (it was
    registered, uncompared, before)."""
    rows, handed_text, _ = NO_FINGERPRINT[case]
    w = build_sources(tmp_path, rows_a=rows)
    try:
        no_text = capture.occurrence(w.src_a, 2)
        result = run(w, [Handed(no_text, handed_text)])
        assert_refused_naming(try_register(w, result, "O"), "paraphrase", no_text)
        assert w.store.artifact("gen/O") is None
    finally:
        w.store.close()


@pytest.mark.parametrize("case", sorted(NO_FINGERPRINT))
def test_a_text_that_is_empty_after_the_normaliser_handed_under_a_record_with_no_fingerprint_registers(
        tmp_path, case):
    rows, _, empty_text = NO_FINGERPRINT[case]
    w = build_sources(tmp_path, rows_a=rows)
    try:
        no_text = capture.occurrence(w.src_a, 2)
        result = run(w, [Handed(no_text, empty_text)])
        refusal = try_register(w, result, "O")
        assert refusal is None, str(refusal)
        assert w.store.artifact("gen/O") is not None
    finally:
        w.store.close()


def test_a_handed_reference_naming_no_record_of_the_store_is_refused_as_before(tmp_path):
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        missing = capture.occurrence(w.src_a, 7)
        result = run(w, [Handed(missing, TEXT_A)])
        assert try_register(w, result, "O") is not None, "the registration was accepted"
        assert row_counts(w.db) == before
    finally:
        w.store.close()


# --- (g) a refusal is atomic ------------------------------------------------------------------


def test_a_refusal_for_the_second_job_of_a_file_leaves_the_first_job_unwritten_too(tmp_path):
    """One file of two lines, the first made by a job handed the right text and the second
    by a job handed the wrong text. No row of any table is added, so the first job's log,
    its edge and the artifact are not left behind."""
    w = build_sources(tmp_path)
    try:
        before = row_counts(w.db)
        good = run(w, [Handed(w.ref_a, TEXT_A)], job_name="good-job", tag="good")
        bad = run(w, [Handed(w.ref_b, TEXT_A)], job_name="bad-job", tag="bad")
        edges = capture.generation_edges(good, [1], STEP) + capture.generation_edges(bad, [2], STEP)
        try:
            register_file(w, "O", edges, lines=2)
            refusal = None
        except errors.WaterwhiteError as error:
            refusal = error
        assert row_counts(w.db) == before
        assert w.store.harness_log("good-job") is None
        assert w.store.artifact("gen/O") is None
        assert_refused_naming(refusal, "bad-job", w.ref_b)
    finally:
        w.store.close()
