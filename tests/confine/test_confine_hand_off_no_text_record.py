"""A reference to a record that holds no text, handed with the text of another record, run
in the real sandbox (docs/confine.md section 5; docs/store.md "The write methods").

Before the store refused it, such an output registered with the text-less record as its
parent, and a request for the record whose text the job really read answered "does not
contain" past the confined log: a wrong negative in a new store. The registration refuses
it now, naming the job and the reference, and nothing is written. The control, in a store
built the same way: the right text handed under the right reference registers, and the answer for the
source outside the hand-off is the bound's "does not contain"."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, coverage, errors, requests
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
    w = SimpleNamespace(folder=tmp_path)
    w.store = Store.create(tmp_path / "evidence.db", normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", [{"id": "A-001", "text": TEXT_A}, {"id": "A-002"}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    w.no_text = capture.occurrence(w.src_a, 2)
    w.ref_b = capture.occurrence(w.src_b, 1)
    return w


def register(w, jobs, handed, job_name):
    result = run_confined_job(
        python_program(jobs / "paraphrase_one.py"), job_name=job_name, handed=handed,
        log_path=w.folder / f"logs/{job_name}.jsonl", timeout_seconds=30)
    step = capture.step_ref("generate", "1", {"style": "plain"})
    out = w.folder / f"gen/{job_name}.jsonl"
    write_rows(out, [{"id": job_name, "text": result.outputs[0].text}])
    try:
        capture.record_jsonl(
            w.store, out, name=f"gen/{job_name}", role=capture.ROLE_INTERMEDIATE, step=step,
            inputs=[w.src_a, w.src_b], edges=capture.generation_edges(result, [1], step))
    except errors.WaterwhiteError as error:
        return error
    return None


def answer_word(w, source, artifact):
    return coverage.answer(w.store, artifact, requests.withdraw(w.store, source)).to_json()["answer"]


@pytest.fixture
def world(tmp_path):
    w = build(tmp_path)
    yield w
    w.store.close()


def test_the_text_of_B_handed_under_a_record_with_no_text_is_refused_in_the_sandbox(world, sandbox, jobs):
    w = world
    refusal = register(w, jobs, [Handed(w.no_text, TEXT_B)], "wrong")
    assert refusal is not None, (
        "the registration was accepted: withdrawing B, whose text the job read, now answers "
        f"{answer_word(w, 'B', 'gen/wrong')!r}")
    assert refusal.name == "invalid record", str(refusal)
    assert "wrong" in str(refusal) and w.no_text.key in str(refusal), str(refusal)
    assert w.store.artifact("gen/wrong") is None
    assert w.store.harness_log("wrong") is None


def test_the_right_text_under_the_right_reference_registers_and_the_bound_answers(world, sandbox, jobs):
    w = world
    refusal = register(w, jobs, [Handed(w.ref_a, TEXT_A)], "right")
    assert refusal is None, str(refusal)
    assert answer_word(w, "B", "gen/right") == "does not contain"
