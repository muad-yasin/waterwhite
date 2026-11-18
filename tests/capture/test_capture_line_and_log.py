"""What capture and the harness gained when the tokenization maps, the recovery
and the confined harness became part of the tool: the public reader of a line's
text, the word a harness log carries for the harness that wrote it, and the
model a client reports (plan 3.1, 3.3 "Generation", F10).
"""

from __future__ import annotations

import json

import pytest

from waterwhite import capture, harness
from waterwhite import vocabulary as v
from waterwhite.errors import InvalidRecord
from waterwhite.harness import Handed, JobOutput, ModelReply, read_log, run_job
from waterwhite.records import OccurrenceRef

A1 = OccurrenceRef("a" * 64, "L1")
HAND_OFF = [Handed(A1, "first text")]


def entries(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


# --- The text a line holds --------------------------------------------------------------


def test_the_text_a_line_holds_has_one_public_definition():
    cases = [
        (b'{"id": "A-1", "text": "one"}', "one"),
        (b'{"text": ""}', ""),
        (b'{"body": "other field"}', None),
        (b'{"text": 5}', None),
        (b'["text"]', None),
        (b"not JSON", None),
        (b"", None),
        (b"\xff\xfe", None),
    ]
    for raw, text in cases:
        assert capture.line_text(raw, "text") == text, raw
        assert capture.line_text(raw) == text  # the field is "text" unless named
    assert "line_text" in capture.__all__


def test_a_line_is_read_by_the_named_field():
    assert capture.line_text(b'{"body": "other field", "text": "x"}', "body") == "other field"


# --- Which harness wrote a log (plan F10) ------------------------------------------------


def test_a_log_names_the_harness_that_wrote_it(tmp_path):
    def job(ctx):
        return JobOutput(ctx.read(A1), (A1,))

    result = run_job(job, job_name="plain", handed=HAND_OFF, log_path=tmp_path / "in.jsonl")
    assert result.log.harness == v.HARNESS_IN_PROCESS == "in-process"
    assert entries(tmp_path / "in.jsonl")[0]["harness"] == "in-process"
    assert read_log(tmp_path / "in.jsonl").harness == "in-process"
    assert v.HARNESS_KINDS == ("in-process", "confined")

    # The public function takes no word for the harness. The code behind it refuses a
    # word outside the vocabulary before a log file is made.
    with pytest.raises(TypeError):
        run_job(job, job_name="plain", handed=HAND_OFF, log_path=tmp_path / "y.jsonl",
                harness="confined")
    assert not (tmp_path / "y.jsonl").exists()
    with pytest.raises(InvalidRecord, match="a harness is one of"):
        harness._run_job(job, job_name="plain", handed=HAND_OFF, log_path=tmp_path / "x.jsonl",
                         model_client=None, harness="sandboxed")
    assert not (tmp_path / "x.jsonl").exists()

    # A log that does not say (one written before the word existed) reads as None,
    # and a log that names another word is not a harness log.
    lines = (tmp_path / "in.jsonl").read_text(encoding="ascii").splitlines()
    first = json.loads(lines[0])
    del first["harness"]
    (tmp_path / "old.jsonl").write_text(
        "\n".join([json.dumps(first, sort_keys=True, separators=(",", ":"))] + lines[1:]) + "\n",
        encoding="ascii")
    assert read_log(tmp_path / "old.jsonl").harness is None
    first["harness"] = "sandboxed"
    (tmp_path / "bad.jsonl").write_text(
        "\n".join([json.dumps(first, sort_keys=True, separators=(",", ":"))] + lines[1:]) + "\n",
        encoding="ascii")
    with pytest.raises(InvalidRecord, match="names the harness 'sandboxed'"):
        read_log(tmp_path / "bad.jsonl")


# --- The model a client reports -----------------------------------------------------------


def test_the_model_a_client_reports_is_logged_and_named_by_the_model_entry(tmp_path):
    """The job asks for a model by a text of its choosing. The client knows which
    model answered. Both are in the log, and the model entry names the reported
    one."""
    asked: list[str] = []

    def client(model: str, prompt: str):
        asked.append(model)
        if model == "plain":
            return "text only"  # a client that reports nothing: the asked name stands
        return ModelReply(text=f"answer to {prompt}", model="team-model-2026-10-01")

    def job(ctx):
        first = ctx.call_model("latest", "one")
        second = ctx.call_model("latest", "two")
        third = ctx.call_model("plain", "three")
        return JobOutput("|".join([first, second, third]), (A1,))

    log_path = tmp_path / "job.jsonl"
    result = run_job(job, job_name="reported", handed=HAND_OFF, log_path=log_path,
                     model_client=client)
    assert asked == ["latest", "latest", "plain"]
    assert result.outputs[0].text == "answer to one|answer to two|text only"
    assert entries(log_path)[1:-2] == [
        {"seq": 1, "kind": "model_call", "model": "latest"},
        {"seq": 2, "kind": "model_reply", "call": 1, "model": "team-model-2026-10-01"},
        {"seq": 3, "kind": "model_call", "model": "latest"},
        {"seq": 4, "kind": "model_reply", "call": 3, "model": "team-model-2026-10-01"},
        {"seq": 5, "kind": "model_call", "model": "plain"},
        {"seq": 6, "kind": "output", "index": 0, "used": [
            {"instance_id": A1.instance_id, "locator": A1.locator}]},
    ]
    assert entries(log_path)[-2] == {
        "seq": 7, "kind": "model", "model": '["team-model-2026-10-01","plain"]',
        "models": ["team-model-2026-10-01", "plain"]}
    assert result.log.model == '["team-model-2026-10-01","plain"]'
    assert "answer to" not in log_path.read_text(encoding="ascii")  # no reply text in the log


@pytest.mark.parametrize(("reply", "said"), [
    (ModelReply("text", "none"), "reported a model that cannot be logged"),
    (ModelReply("text", ""), "reported a model that cannot be logged"),
    (ModelReply(5, "m-1"), "returned a reply whose text is int"),
])
def test_a_reply_that_cannot_be_logged_ends_the_job(tmp_path, reply, said):
    def job(ctx):
        return JobOutput(ctx.call_model("asked", "prompt"), (A1,))

    with pytest.raises(InvalidRecord, match=said):
        run_job(job, job_name="bad-reply", handed=HAND_OFF, log_path=tmp_path / "job.jsonl",
                model_client=lambda model, prompt: reply)
    # The call is in the log, with the name the job asked for, and the log failed.
    log = entries(tmp_path / "job.jsonl")
    assert log[1] == {"seq": 1, "kind": "model_call", "model": "asked"}
    assert (log[-1]["status"], log[-2]["model"]) == ("failed", "asked")


def test_a_reply_entry_that_names_no_call_is_not_a_harness_log(tmp_path):
    def job(ctx):
        return JobOutput(ctx.call_model("asked", "prompt"), (A1,))

    run_job(job, job_name="ok", handed=HAND_OFF, log_path=tmp_path / "job.jsonl",
            model_client=lambda model, prompt: ModelReply("t", "answered"))
    lines = (tmp_path / "job.jsonl").read_text(encoding="ascii").splitlines()
    assert json.loads(lines[2]) == {"seq": 2, "kind": "model_reply", "call": 1, "model": "answered"}
    forged = lines[:2] + [lines[2].replace('"call":1', '"call":0')] + lines[3:]
    (tmp_path / "forged.jsonl").write_text("\n".join(forged) + "\n", encoding="ascii")
    with pytest.raises(InvalidRecord, match="names one call entry before it"):
        read_log(tmp_path / "forged.jsonl")
    # A model entry that keeps the asked name while a reply names another is refused.
    stale = [line.replace('"model":"answered","models":["answered"]',
                          '"model":"asked","models":["asked"]') for line in lines]
    assert stale != lines
    (tmp_path / "stale.jsonl").write_text("\n".join(stale) + "\n", encoding="ascii")
    with pytest.raises(InvalidRecord, match="the call and reply entries say 'answered'"):
        read_log(tmp_path / "stale.jsonl")
    assert harness.ENTRY_MODEL_REPLY == "model_reply"
