"""The confined harness as part of the tool: its errors are classes of
``waterwhite.errors``, its log says that a confined harness wrote it, the store
keeps that word, and the model its log names is the one the caller's client
reports (plan 3.3 "Generation", 2.2(g), F10).
"""

from __future__ import annotations

import json

import pytest

from waterwhite import capture, confine, errors
from waterwhite import vocabulary as v
from waterwhite.confine import errors as confine_errors
from waterwhite.confine import python_program, run_confined_job
from waterwhite.harness import Handed, ModelReply
from waterwhite.records import OccurrenceRef
from waterwhite.store import Store

pytestmark = pytest.mark.sandbox

LIMIT = 30


def entries(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def test_the_errors_of_the_confined_harness_are_classes_of_the_errors_module():
    for name, said in (("SandboxUnavailable", "sandbox unavailable"),
                       ("ConfinedJobFailed", "confined job failed"),
                       ("JobProtocolViolation", "job protocol violation")):
        cls = getattr(errors, name)
        assert cls.__module__ == "waterwhite.errors"
        assert getattr(confine_errors, name) is cls is getattr(confine, name)
        assert (cls.name, cls.exit_code) == (said, errors.EXIT_USAGE)
    # The module that held them defines no class of its own any more.
    assert [value for value in vars(confine_errors).values()
            if isinstance(value, type) and value.__module__ == confine_errors.__name__] == []


def test_a_confined_log_says_so_and_the_store_keeps_the_word(sandbox, jobs, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"text": "first text"}) + "\n", encoding="utf-8")
    with Store.create(tmp_path / "evidence.db", normaliser="control-chars/1") as store:
        instance = capture.ingest_jsonl(store, source, source="A", member="A/r1", revision="r1",
                                        name="src/A")
        only = OccurrenceRef(instance, "L1")
        result = run_confined_job(
            python_program(jobs / "echo.py"), job_name="echo", handed=[Handed(only, "first text")],
            log_path=tmp_path / "echo.jsonl", timeout_seconds=LIMIT)
        assert result.log.harness == v.HARNESS_CONFINED == "confined"
        assert entries(tmp_path / "echo.jsonl")[0]["harness"] == "confined"

        step = capture.step_ref("paraphrase", "1")
        out = tmp_path / "out.jsonl"
        out.write_text("".join(json.dumps({"text": o.text}) + "\n" for o in result.outputs),
                       encoding="utf-8")
        made = capture.record_jsonl(
            store, out, name="gen/echo", role=capture.ROLE_INTERMEDIATE, step=step,
            inputs=[instance],
            edges=capture.generation_edges(result, range(1, len(result.outputs) + 1), step))
        stored = store.edge_of(OccurrenceRef(made, "L1")).harness_log
        assert (stored.harness, stored.bound_established()) == ("confined", True)
        assert store.harness_log("echo").harness == "confined"


def test_the_model_the_callers_client_reports_is_what_the_log_names(sandbox, jobs, tmp_path):
    """The confined job chooses the text of its model request. The client runs
    outside the sandbox and reports which model answered; the log's model entry
    names that one, so a job cannot make the log name a model of its choosing."""
    log_path = tmp_path / "job.jsonl"

    def client(model: str, prompt: str) -> ModelReply:
        return ModelReply(text=f"<{model}>", model=f"resolved/{model}")

    result = run_confined_job(
        python_program(jobs / "call_model.py"), job_name="with-model",
        handed=[Handed(OccurrenceRef("a" * 64, "L1"), "first text")], log_path=log_path,
        timeout_seconds=LIMIT, model_client=client)
    assert result.outputs[0].text == "<stand-in-1>|<stand-in-2>|<stand-in-1>|invalid record"
    assert result.log.model == '["resolved/stand-in-1","resolved/stand-in-2"]'
    kinds = [(e["kind"], e.get("model"), e.get("call")) for e in entries(log_path)[2:8]]
    assert kinds == [
        ("model_call", "stand-in-1", None), ("model_reply", "resolved/stand-in-1", 2),
        ("model_call", "stand-in-2", None), ("model_reply", "resolved/stand-in-2", 4),
        ("model_call", "stand-in-1", None), ("model_reply", "resolved/stand-in-1", 6),
    ]


def test_a_reported_model_that_cannot_be_logged_ends_the_run(sandbox, jobs, tmp_path):
    with pytest.raises(errors.InvalidRecord, match="reported a model that cannot be logged"):
        run_confined_job(
            python_program(jobs / "call_model.py"), job_name="with-model",
            handed=[Handed(OccurrenceRef("a" * 64, "L1"), "first text")],
            log_path=tmp_path / "job.jsonl", timeout_seconds=LIMIT,
            model_client=lambda model, prompt: ModelReply("text", "none"))
    assert entries(tmp_path / "job.jsonl")[-1]["status"] == "failed"
