"""The sandbox's log of a hand-off, at registration (docs/confine.md section 5,
docs/store.md "The write methods").

- The right text handed in the sandbox registers and the confined log still establishes
  its bound: a source that is nowhere in the history answers "does not contain", the
  answer the bound exists for. A text that is not the record, handed in the sandbox in
  the same store, is refused.
- The result `run_confined_job` returns carries the handed texts on its log, in
  hand-off order, outside the log's equality; the log read back from the file carries
  none.
- A result built from the log file (`read_log`) has no texts to compare: it registers
  whichever text the job was handed, and its log is stored as not compared: it
  establishes no bound ("may contain") until the texts are shown and compared, when the
  text the job was handed is the record, and is refused when it is not.

The source whose withdrawal shows the bound is B, which is outside the hand-off; the
handed source A answers "contains". The log's field is read with getattr, so a build
without it fails at the comparison and not at a missing name."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import capture, coverage, errors, harness, requests
from waterwhite.confine import python_program, run_confined_job
from waterwhite.harness import Handed
from waterwhite.store import Store

pytestmark = pytest.mark.sandbox

NORMALISER = "control-chars/1"
TEXT_A = "The repair workshop meets on the first Tuesday of the month."
TEXT_B = "A bakery on the corner opens before the buses start to run."
NO_TEXTS = "the log has no handed_texts field"


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(row) + "\n" for row in rows).encode("utf-8"))


def build(tmp_path: Path) -> SimpleNamespace:
    w = SimpleNamespace(folder=tmp_path)
    w.store = Store.create(tmp_path / "evidence.db", normaliser=NORMALISER)
    write_rows(tmp_path / "sources/A.jsonl", [{"id": "A-001", "text": TEXT_A}])
    write_rows(tmp_path / "sources/B.jsonl", [{"id": "B-001", "text": TEXT_B}])
    w.src_a = capture.ingest_jsonl(
        w.store, tmp_path / "sources/A.jsonl", source="A", member="A/r1", revision="r1", name="src/A")
    w.src_b = capture.ingest_jsonl(
        w.store, tmp_path / "sources/B.jsonl", source="B", member="B/r1", revision="r1", name="src/B")
    w.ref_a = capture.occurrence(w.src_a, 1)
    return w


def sandbox_run(w: SimpleNamespace, jobs: Path, text: str, job_name: str):
    w.log_path = w.folder / f"logs/{job_name}.jsonl"
    return run_confined_job(
        python_program(jobs / "paraphrase_one.py"), job_name=job_name,
        handed=[Handed(w.ref_a, text)], log_path=w.log_path, timeout_seconds=30)


def register(w: SimpleNamespace, result, name: str) -> str:
    step = capture.step_ref("generate", "1", {"style": "plain"})
    out = w.folder / f"gen/{name}.jsonl"
    write_rows(out, [{"id": name, "text": result.outputs[0].text}])
    return capture.record_jsonl(
        w.store, out, name=f"gen/{name}", role=capture.ROLE_INTERMEDIATE, step=step,
        inputs=[w.src_a, w.src_b], edges=capture.generation_edges(result, [1], step))


def ask(w: SimpleNamespace, source: str, artifact: str, label: str) -> dict:
    answer = coverage.answer(w.store, artifact, requests.withdraw(w.store, source)).to_json()
    folder = w.folder / "answers"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{label}.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return answer


@pytest.fixture
def world(tmp_path):
    w = build(tmp_path)
    yield w
    w.store.close()


def test_the_right_text_in_the_sandbox_registers_with_its_bound_and_the_other_text_is_refused(sandbox, jobs, world):
    w = world
    register(w, sandbox_run(w, jobs, TEXT_A, "paraphrase"), "O")
    assert ask(w, "B", "gen/O", "sandbox-right-withdraw-B")["answer"] == "does not contain"
    answer = ask(w, "A", "gen/O", "sandbox-right-withdraw-A")
    assert answer["answer"] == "contains"
    assert w.src_a in [hop.get("instance") for hop in answer["chain"]]
    log = w.store.harness_log("paraphrase")
    assert log is not None and log.bound_established()
    wrong = sandbox_run(w, jobs, TEXT_B, "control")
    try:
        register(w, wrong, "W")
        refusal = None
    except errors.WaterwhiteError as error:
        refusal = error
    assert refusal is not None, "the registration was accepted"
    assert refusal.name == "invalid record", str(refusal)
    assert "control" in str(refusal), str(refusal)
    assert w.ref_a.key in str(refusal), str(refusal)


@pytest.mark.parametrize("handed_text", [TEXT_A, TEXT_B], ids=["right-text", "wrong-text"])
def test_a_confined_log_read_back_from_its_file_registers_and_establishes_no_bound_until_its_texts_are_shown(
        sandbox, jobs, world, handed_text):
    w = world
    ran = sandbox_run(w, jobs, handed_text, "paraphrase")
    read_back = harness.JobResult(ran.outputs, harness.read_log(w.log_path))
    register(w, read_back, "O")
    outside = ask(w, "B", "gen/O", "sandbox-read-back-withdraw-B")
    assert (outside["answer"], outside["names"]) == ("may contain", "upper bound not established")
    assert ask(w, "A", "gen/O", "sandbox-read-back-withdraw-A")["answer"] == "contains"
    log = w.store.harness_log("paraphrase")
    assert log is not None and log.bound_stated() and not log.bound_established()
    if handed_text == TEXT_A:
        w.store.verify_hand_off("paraphrase", [(w.ref_a, handed_text)])
        assert ask(w, "B", "gen/O", "sandbox-read-back-withdraw-B-after")["answer"] == "does not contain"
        assert w.store.harness_log("paraphrase").bound_established()
    else:
        with pytest.raises(errors.InvalidRecord):
            w.store.verify_hand_off("paraphrase", [(w.ref_a, handed_text)])
        assert not w.store.harness_log("paraphrase").hand_off_checked


def test_the_sandbox_result_carries_the_handed_texts_on_its_log_and_the_file_carries_none(sandbox, jobs, world):
    w = world
    ran = sandbox_run(w, jobs, TEXT_A, "paraphrase")
    assert getattr(ran.log, "handed_texts", NO_TEXTS) == ((w.ref_a, TEXT_A),)
    from_file = harness.read_log(w.log_path)
    assert getattr(from_file, "handed_texts", NO_TEXTS) == ()
    assert ran.log == from_file
