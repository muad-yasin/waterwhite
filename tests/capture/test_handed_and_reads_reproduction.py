"""What the capture layer records, and what a query then answers, when a job's
statement and its reads disagree (docs/capture.md section 2 "Builders for what a
step states" and section 3, docs/coverage.md section 6, docs/confine.md section 5).

Two cases, both driven through the real code: the real harness (`run_job`), the
real capture wrappers, the real store and the real coverage engine
(`coverage.answer`). Nothing is stood in for.

- Case (a): a job is handed records of A and of B, reads both through the harness,
  and its output states that it used B only.
- Case (b): a job is handed `Handed(ref_A, text_B)`: the reference of a record of A
  with the text of a record of B.

Every expected value below is written from the documents: the edge of a job
handed two records keeps the one the output states as a parent and the other in
scope only (docs/capture.md, docs/coverage.md section 6); a record in scope only
that is in scope of the request gives "may contain" naming "in scope only", never
"does not contain"; a "contains" past a generation edge carries the
process-evidence note; a log the in-process harness wrote establishes no bound; the
log holds references and never a record's text; and `Handed(ref, text)` is the
caller's statement that the text is the record: the harness holds no store and does not
compare it, and the registration of the output compares it where the log it registers
still holds the texts (docs/confine.md section 5).

The expected value of each test alone decides red or green.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, coverage, errors, harness, requests
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.store import Store

NORMALISER = "control-chars/1"

TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."

# Case (b): what the documents say about a hand-off whose text is not the record's.
EXPECTED_HAND_OFF_OF_A_REFERENCE_WITH_ANOTHER_TEXT = "accepted"


def write_rows(path: Path, rows: list[dict]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(data)
    return data


def dump(folder: Path, name: str, value: object) -> None:
    """Keep the real output of a real call in the test's temp folder."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_sources(tmp_path: Path) -> SimpleNamespace:
    """Two sources of one record each, ingested the way a team ingests them."""
    w = SimpleNamespace(folder=tmp_path)
    w.store = Store.create(tmp_path / "evidence.db", normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    w.ref_b = capture.occurrence(w.src_b, 1)
    return w


def register_output(w: SimpleNamespace, result, name: str, output_text: str) -> str:
    """Write the job's one output as a file of one line and register it with the
    job's generation edge, as a pipeline does."""
    step = capture.step_ref("generate", "1", {"style": "plain"})
    path = w.folder / f"gen/{name}.jsonl"
    write_rows(path, [{"id": name, "text": output_text}])
    return capture.record_jsonl(
        w.store, path, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=step,
        inputs=[w.src_a, w.src_b], edges=capture.generation_edges(result, [1], step))


def ask(w: SimpleNamespace, source: str, artifact: str, label: str) -> dict:
    """Put a withdrawal of one source to one artifact through the real engine and
    keep the answer's JSON form."""
    request = requests.withdraw(w.store, source)
    answer = coverage.answer(w.store, artifact, request).to_json()
    dump(w.folder / "answers", f"{label}.json", answer)
    return answer


# --- case (a): reads A and B, the output states B only ---------------------------------


@pytest.fixture
def read_both_state_one(tmp_path):
    w = build_sources(tmp_path)

    def job(ctx):
        first, second = ctx.handed()
        ctx.read(first)
        ctx.read(second)
        return JobOutput(text="A bakery and a workshop share a street.", used=(second,))

    w.result = run_job(
        job, job_name="mixer",
        handed=[Handed(w.ref_a, TEXT_A), Handed(w.ref_b, TEXT_B)],
        log_path=tmp_path / "logs/mixer.jsonl")
    w.out = register_output(w, w.result, "O", w.result.outputs[0].text)
    yield w
    w.store.close()


def test_a_read_record_the_output_does_not_state_is_stored_as_in_scope_only(read_both_state_one):
    """What capture records for A: the log lists the reads of both records, and the
    stored edge keeps B as the parent and A as a record in scope only."""
    w = read_both_state_one
    log = w.result.log
    assert [(r.occurrence, r.refused) for r in log.reads] == [(w.ref_a, False), (w.ref_b, False)]
    edge = w.store.edge_of(capture.occurrence(w.out, 1))
    stored = {
        "parents": [ref.key for ref in edge.parents],
        "in_scope_only": [ref.key for ref in edge.in_scope_only],
    }
    dump(w.folder / "answers", "edge-a.json", stored)
    assert stored == {"parents": [w.ref_b.key], "in_scope_only": [w.ref_a.key]}


def test_withdrawing_the_record_the_output_did_not_state_gives_may_contain_in_scope_only(read_both_state_one):
    """Source A: the job was handed a record of A and its output does not state it.
    The answer is "may contain" naming "in scope only", with the job and the handed
    record (docs/coverage.md section 6). It is never "does not contain"."""
    w = read_both_state_one
    answer = ask(w, "A", "gen/O", "withdraw-A")
    assert answer["answer"] == "may contain"
    assert answer["names"] == "in scope only"
    assert answer["scope"] is None
    failing = answer["failing"]
    assert (failing["instance"], failing["record"], failing["job"]) == (w.src_a, "L1", "mixer")


def test_withdrawing_the_stated_record_gives_contains_with_the_process_evidence_note(read_both_state_one):
    """Source B: the output states B as used, so B's record is a parent. A "contains"
    past a generation edge carries the process-evidence note (docs/coverage.md
    section 6)."""
    w = read_both_state_one
    answer = ask(w, "B", "gen/O", "withdraw-B")
    assert answer["answer"] == "contains"
    assert w.src_b in [hop.get("instance") for hop in answer["chain"]]
    assert len(answer["notes"]) == 1 and "process evidence" in answer["notes"][0]


# --- case (b): Handed(ref_A, text_B) --------------------------------------------------


def paraphrase(ctx):
    (only,) = ctx.handed()
    text = ctx.read(only)
    return JobOutput(text=text.upper(), used=(only,))


def hand_off_outcome(tmp_path, handed, log_path) -> str:
    """"accepted" when the real harness runs the job, otherwise the name of the
    refusal."""
    try:
        run_job(paraphrase, job_name="paraphrase", handed=handed, log_path=log_path)
    except errors.WaterwhiteError as error:
        return error.name
    return "accepted"


def test_a_reference_handed_with_the_text_of_another_record_is_accepted(tmp_path):
    """docs/confine.md section 5: `Handed(ref, text)` is the caller's statement that
    the text is the record at `ref`; the harness holds no store and runs the job.
    The registration of its output is where the text is compared (tests below)."""
    w = build_sources(tmp_path)
    try:
        outcome = hand_off_outcome(tmp_path, [Handed(w.ref_a, TEXT_B)], tmp_path / "logs/wrong.jsonl")
    finally:
        w.store.close()
    assert outcome == EXPECTED_HAND_OFF_OF_A_REFERENCE_WITH_ANOTHER_TEXT


def test_the_log_of_a_wrong_text_hand_off_is_the_log_of_the_right_one(tmp_path):
    """A log holds references and never a record's text (docs/capture.md section 3),
    so the same job with the same references gives the same bytes whichever text it
    was handed."""
    w = build_sources(tmp_path)
    try:
        right = run_job(paraphrase, job_name="paraphrase", handed=[Handed(w.ref_a, TEXT_A)],
                        log_path=tmp_path / "logs/right.jsonl")
        wrong = run_job(paraphrase, job_name="paraphrase", handed=[Handed(w.ref_a, TEXT_B)],
                        log_path=tmp_path / "logs/wrong.jsonl")
    finally:
        w.store.close()
    assert right.outputs[0].text == TEXT_A.upper()
    assert wrong.outputs[0].text == TEXT_B.upper()
    assert (tmp_path / "logs/wrong.jsonl").read_bytes() == (tmp_path / "logs/right.jsonl").read_bytes()
    assert wrong.log == right.log


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
def wrong_text_registered(tmp_path):
    """The job runs; the registration of its output is attempted and its outcome kept."""
    w = build_sources(tmp_path)
    w.counts_before = row_counts(tmp_path / "evidence.db")
    w.result = run_job(paraphrase, job_name="paraphrase", handed=[Handed(w.ref_a, TEXT_B)],
                       log_path=tmp_path / "logs/wrong.jsonl")
    try:
        w.out = register_output(w, w.result, "W", w.result.outputs[0].text)
        w.refusal = None
    except errors.WaterwhiteError as error:
        w.out, w.refusal = None, error
    yield w
    w.store.close()


def test_a_wrong_text_hand_off_is_refused_at_registration_naming_the_job_and_the_reference(wrong_text_registered):
    """The store compares the text handed under each reference with the record the
    reference names: the output of a job handed A's reference with B's text is not
    registered with A as its parent."""
    w = wrong_text_registered
    assert w.refusal is not None, "the registration was accepted"
    assert w.refusal.name == "invalid record", str(w.refusal)
    assert "paraphrase" in str(w.refusal), str(w.refusal)
    assert w.ref_a.key in str(w.refusal), str(w.refusal)


def test_nothing_of_a_refused_wrong_text_hand_off_is_stored(wrong_text_registered):
    """No artifact, no edge and no log of the job are left, and no table of the store
    has a row more than before the attempt."""
    w = wrong_text_registered
    assert w.store.artifact("gen/W") is None
    assert w.store.harness_log("paraphrase") is None
    assert row_counts(w.folder / "evidence.db") == w.counts_before

