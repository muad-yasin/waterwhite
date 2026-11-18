"""run_job, the in-process closed-input harness (plan 3.3 "Generation", Check 5's
refusal sentence, Decisions 13 and 25, F10; DECISIONS.md D-07).
"""

from __future__ import annotations

import hashlib
import json

import pytest

from waterwhite import harness
from waterwhite.errors import EXIT_REFUSED, InvalidRecord, Refused, WaterwhiteError, WriteRefused
from waterwhite.harness import Handed, HarnessRefused, JobOutput, run_job
from waterwhite.records import HarnessLog, HarnessRead, OccurrenceRef

A1 = OccurrenceRef("a" * 64, "L1")
A2 = OccurrenceRef("a" * 64, "L2")
C1 = OccurrenceRef("c" * 64, "L1")
OUTSIDE = OccurrenceRef("b" * 64, "L1")

HAND_OFF = [Handed(A1, "first text"), Handed(A2, "second text"), Handed(C1, "third text")]


def entries(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def ref(r: OccurrenceRef) -> dict:
    return {"instance_id": r.instance_id, "locator": r.locator}


def test_reads_are_logged_in_order_and_the_digest_is_the_files(tmp_path):
    log_path = tmp_path / "logs" / "job.log.jsonl"

    def job(ctx):
        assert ctx.handed() == (A1, A2, C1)
        texts = [ctx.read(C1), ctx.read(A1), ctx.read(C1)]
        return JobOutput(text=" + ".join(texts), used=(C1, A1))

    result = run_job(job, job_name="order", handed=HAND_OFF, log_path=log_path)

    assert result.outputs == (JobOutput("third text + first text + third text", (C1, A1)),)
    assert result.log == HarnessLog(
        job_name="order",
        log_digest=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        handed=(A1, A2, C1),
        reads=(HarnessRead(C1), HarnessRead(A1), HarnessRead(C1)),
        model="none",
        harness="in-process",
    )
    # The log lists no read outside the hand-off; it is an in-process log, so it
    # establishes no bound (Decision 13: only a confined log does).
    assert result.log.reads_within_hand_off() and not result.log.bound_established()
    assert entries(log_path) == [
        {
            "seq": 0,
            "kind": "job",
            "format": "waterwhite.harness-log/1",
            "job_name": "order",
            "handed": [ref(A1), ref(A2), ref(C1)],
            "harness": "in-process",
        },
        {"seq": 1, "kind": "read", "refused": False, **ref(C1)},
        {"seq": 2, "kind": "read", "refused": False, **ref(A1)},
        {"seq": 3, "kind": "read", "refused": False, **ref(C1)},
        {"seq": 4, "kind": "output", "index": 0, "used": [ref(C1), ref(A1)]},
        {"seq": 5, "kind": "model", "model": "none", "models": []},
        {"seq": 6, "kind": "end", "status": "completed", "outputs": 1},
    ]


def test_the_log_holds_no_record_text(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    run_job(
        lambda ctx: JobOutput(text="made from " + ctx.read(A1), used=(A1,)),
        job_name="quiet",
        handed=HAND_OFF,
        log_path=log_path,
    )
    data = log_path.read_text(encoding="ascii")
    for text in ("first text", "second text", "third text", "made from"):
        assert text not in data


def test_an_entry_is_on_disk_before_the_read_returns(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    seen: list[list[str]] = []

    def job(ctx):
        seen.append([e["kind"] for e in entries(log_path)])
        ctx.read(A1)
        seen.append([e["kind"] for e in entries(log_path)])
        ctx.read(A2)
        seen.append([e["kind"] for e in entries(log_path)])
        return []

    run_job(job, job_name="durable", handed=HAND_OFF, log_path=log_path)
    assert seen == [["job"], ["job", "read"], ["job", "read", "read"]]


def test_the_same_job_gives_the_same_log_bytes(tmp_path):
    def job(ctx):
        return JobOutput(text=ctx.read(A2), used=(A2,))

    one = run_job(job, job_name="same", handed=HAND_OFF, log_path=tmp_path / "one.jsonl")
    two = run_job(job, job_name="same", handed=HAND_OFF, log_path=tmp_path / "two.jsonl")
    assert one.log == two.log
    other = run_job(job, job_name="other", handed=HAND_OFF, log_path=tmp_path / "three.jsonl")
    assert other.log.log_digest != one.log.log_digest


def test_a_read_outside_the_hand_off_is_refused_and_logged_and_counts_as_no_read(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    caught: list[HarnessRefused] = []

    def job(ctx):
        first = ctx.read(A1)
        try:
            ctx.read(OUTSIDE)
        except HarnessRefused as refusal:
            caught.append(refusal)
        return JobOutput(text=first, used=(A1,))

    result = run_job(job, job_name="stray", handed=HAND_OFF, log_path=log_path)

    assert len(caught) == 1
    assert caught[0].name == "read outside the hand-off"
    assert isinstance(caught[0], WaterwhiteError)
    # A refusal of the family of refused requests: nothing was delivered.
    assert isinstance(caught[0], Refused) and caught[0].exit_code == EXIT_REFUSED
    assert caught[0].context == {"job": "stray", "ref": OUTSIDE.key}
    assert result.log.reads == (
        HarnessRead(occurrence=A1, outside_ref=None, refused=False),
        HarnessRead(occurrence=None, outside_ref=OUTSIDE.key, refused=True),
    )
    assert entries(log_path)[2] == {
        "seq": 2,
        "kind": "read",
        "refused": True,
        "outside_ref": OUTSIDE.key,
    }
    # DECISIONS.md D-07: a refused attempt delivered nothing, so it does not count as
    # a read outside the hand-off. (An in-process log establishes no bound either
    # way: Decision 13.)
    assert result.log.reads_within_hand_off()
    assert result.outputs == (JobOutput("first text", (A1,)),)


def test_a_job_that_does_not_catch_the_refusal_fails_and_the_log_shows_the_attempt(tmp_path):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        ctx.read(A1)
        ctx.read(OUTSIDE)
        raise AssertionError("the read above is refused")

    with pytest.raises(HarnessRefused):
        run_job(job, job_name="stray", handed=HAND_OFF, log_path=log_path)

    log = harness.read_log(log_path)
    assert log.reads == (HarnessRead(A1), HarnessRead(None, OUTSIDE.key, True))
    assert log.reads_within_hand_off()
    assert log.log_digest == hashlib.sha256(log_path.read_bytes()).hexdigest()
    assert entries(log_path)[-1] == {
        "seq": 4,
        "kind": "end",
        "status": "failed",
        "error": "read outside the hand-off",
    }


@pytest.mark.parametrize("asked", ["sources/B.jsonl", ("b" * 64, "L1"), None, 7])
def test_anything_that_is_not_a_handed_reference_is_refused_and_logged(tmp_path, asked):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        with pytest.raises(HarnessRefused):
            ctx.read(asked)
        return []

    result = run_job(job, job_name="odd", handed=HAND_OFF, log_path=log_path)
    expected = asked if isinstance(asked, str) else repr(asked)
    assert result.log.reads == (HarnessRead(None, expected, True),)
    assert result.log.reads_within_hand_off()


def test_a_read_let_through_outside_the_hand_off_breaks_the_bound(tmp_path):
    # The harness never writes such an entry. A log that holds one (from another
    # harness, or edited) is read as it stands, and its bound is not established.
    log_path = tmp_path / "job.log.jsonl"
    run_job(lambda ctx: [], job_name="edited", handed=HAND_OFF, log_path=log_path)
    lines = log_path.read_bytes().splitlines(keepends=True)
    stray = {"seq": 1, "kind": "read", "refused": False, "outside_ref": OUTSIDE.key}
    moved = [json.loads(line) | {"seq": n} for n, line in enumerate(lines[1:], start=2)]
    edited = tmp_path / "edited.log.jsonl"
    edited.write_bytes(
        lines[0]
        + json.dumps(stray).encode() + b"\n"
        + b"".join(json.dumps(e).encode() + b"\n" for e in moved)
    )
    log = harness.read_log(edited)
    assert log.reads == (HarnessRead(None, OUTSIDE.key, False),)
    assert not log.bound_established()


def test_an_output_that_uses_a_record_that_was_not_handed_is_refused(tmp_path):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        return JobOutput(text=ctx.read(A1), used=(A1, OUTSIDE))

    with pytest.raises(InvalidRecord) as caught:
        run_job(job, job_name="overclaim", handed=HAND_OFF, log_path=log_path)
    assert caught.value.name == "invalid record"
    assert caught.value.context == {"job": "overclaim", "ref": OUTSIDE.key}
    last = entries(log_path)[-1]
    assert (last["kind"], last["status"], last["error"]) == ("end", "failed", "invalid record")
    assert [e["kind"] for e in entries(log_path)].count("output") == 0


@pytest.mark.parametrize(
    "returned",
    [
        None,
        "a bare text",
        [("text", ())],
        [JobOutput(text=7, used=())],  # type: ignore[arg-type]
        [JobOutput(text="x", used=[A1])],  # type: ignore[arg-type]
        [JobOutput(text="x", used=(A1, A1))],
        [JobOutput(text="x", used=("a" * 64 + ":L1",))],  # type: ignore[arg-type]
    ],
)
def test_a_job_returns_job_outputs_and_nothing_else(tmp_path, returned):
    with pytest.raises(InvalidRecord):
        run_job(lambda ctx: returned, job_name="shape", handed=HAND_OFF, log_path=tmp_path / "l")


def test_a_job_may_return_several_outputs_or_none(tmp_path):
    def two(ctx):
        return [JobOutput(ctx.read(A1), (A1,)), JobOutput("from nothing", ())]

    result = run_job(two, job_name="two", handed=HAND_OFF, log_path=tmp_path / "two.jsonl")
    assert [o.used for o in result.outputs] == [(A1,), ()]
    none = run_job(lambda ctx: [], job_name="none", handed=[], log_path=tmp_path / "none.jsonl")
    assert none.outputs == ()
    assert (none.log.handed, none.log.reads, none.log.model) == ((), (), "none")


def test_the_model_is_none_when_the_job_calls_no_model(tmp_path):
    result = run_job(
        lambda ctx: JobOutput(ctx.read(A1), (A1,)),
        job_name="fixed-output",
        handed=HAND_OFF,
        log_path=tmp_path / "job.log.jsonl",
    )
    assert result.log.model == "none"


def test_a_model_call_is_logged_before_the_client_is_called(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    seen: list[list[str]] = []

    def client(model, prompt):
        seen.append([e["kind"] for e in entries(log_path)])
        return f"{model} says: {prompt}"

    def job(ctx):
        return JobOutput(ctx.call_model("m-1", ctx.read(A1)), (A1,))

    result = run_job(
        job, job_name="called", handed=HAND_OFF, log_path=log_path, model_client=client
    )
    assert seen == [["job", "read", "model_call"]]
    assert result.outputs[0].text == "m-1 says: first text"
    assert result.log.model == "m-1"
    kinds = {e["kind"]: e for e in entries(log_path)}
    assert kinds["model_call"] == {"seq": 2, "kind": "model_call", "model": "m-1"}
    assert kinds["model"] == {"seq": 4, "kind": "model", "model": "m-1", "models": ["m-1"]}
    # The prompt holds a record's text and is not in the log.
    assert "first text" not in log_path.read_text(encoding="ascii")


def test_one_model_called_twice_is_one_model_and_two_models_are_a_list(tmp_path):
    def client(model, prompt):
        return prompt

    def once_each(ctx):
        ctx.call_model("m-2", "a")
        ctx.call_model("m-1", "b")
        ctx.call_model("m-2", "c")
        return []

    def same_twice(ctx):
        ctx.call_model("m-1", "a")
        ctx.call_model("m-1", "b")
        return []

    two = run_job(
        once_each, job_name="two", handed=[], log_path=tmp_path / "two.jsonl", model_client=client
    )
    one = run_job(
        same_twice, job_name="one", handed=[], log_path=tmp_path / "one.jsonl", model_client=client
    )
    assert one.log.model == "m-1"
    assert two.log.model == '["m-2","m-1"]'
    assert json.loads(two.log.model) == ["m-2", "m-1"]
    assert harness.model_field([]) == "none"


@pytest.mark.parametrize("model", ["none", "", "[m]", " m", None])
def test_a_model_identifier_cannot_be_read_as_none_or_as_a_list(tmp_path, model):
    def job(ctx):
        ctx.call_model(model, "prompt")
        return []

    log_path = tmp_path / "job.log.jsonl"
    with pytest.raises(InvalidRecord):
        run_job(job, job_name="bad", handed=[], log_path=log_path, model_client=lambda m, p: p)
    assert [e["kind"] for e in entries(log_path)] == ["job", "model", "end"]


def test_a_model_call_with_no_client_is_logged_and_fails(tmp_path):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        return JobOutput(ctx.call_model("m-1", "prompt"), ())

    with pytest.raises(harness.ModelClientMissing) as caught:
        run_job(job, job_name="no-client", handed=[], log_path=log_path)
    assert caught.value.name == "no model client"
    assert harness.read_log(log_path).model == "m-1"


def test_a_log_with_no_model_entry_reads_as_not_established(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    result = run_job(
        lambda ctx: JobOutput(ctx.read(A1), (A1,)),
        job_name="fixed-output",
        handed=HAND_OFF,
        log_path=log_path,
    )
    assert result.log.model == "none"
    kept = [
        line
        for line in log_path.read_bytes().splitlines(keepends=True)
        if json.loads(line)["kind"] != "model"
    ]
    stripped = tmp_path / "no-model-entry.jsonl"
    stripped.write_bytes(b"".join(kept))
    log = harness.read_log(stripped)
    assert log.model is None
    assert log.reads == result.log.reads
    assert log.log_digest != result.log.log_digest


def test_a_model_entry_that_disagrees_with_the_calls_is_refused(tmp_path):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        ctx.call_model("m-1", "prompt")
        return []

    run_job(job, job_name="called", handed=[], log_path=log_path, model_client=lambda m, p: p)
    edited = tmp_path / "edited.jsonl"
    edited.write_bytes(
        log_path.read_bytes().replace(
            b'"kind":"model","model":"m-1"', b'"kind":"model","model":"none"'
        )
    )
    assert b'"model":"none"' in edited.read_bytes()
    with pytest.raises(InvalidRecord):
        harness.read_log(edited)


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json\n",
        b'{"seq":0,"kind":"read","refused":false,"instance_id":"x","locator":"L1"}\n',
        b'{"seq":0,"kind":"job","format":"other/1","job_name":"j","handed":[]}\n',
        b'{"seq":0,"kind":"job","format":"waterwhite.harness-log/1","job_name":"j","handed":[]}\n'
        b'{"seq":0,"kind":"end","status":"completed","outputs":0}\n',
        b'{"seq":0,"kind":"job","format":"waterwhite.harness-log/1","job_name":"j","handed":[]}\n'
        b'{"seq":1,"kind":"surprise"}\n',
    ],
)
def test_a_file_that_is_not_a_harness_log_is_refused(tmp_path, content):
    path = tmp_path / "log.jsonl"
    path.write_bytes(content)
    with pytest.raises(InvalidRecord):
        harness.read_log(path)


def test_an_existing_log_is_never_overwritten(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    run_job(lambda ctx: [], job_name="first", handed=[], log_path=log_path)
    before = log_path.read_bytes()
    ran: list[str] = []
    with pytest.raises(WriteRefused):
        run_job(lambda ctx: ran.append("ran") or [], job_name="second", handed=[], log_path=log_path)
    assert ran == []
    assert log_path.read_bytes() == before


def test_a_hand_off_names_each_record_once(tmp_path):
    with pytest.raises(InvalidRecord):
        run_job(
            lambda ctx: [],
            job_name="twice",
            handed=[Handed(A1, "one"), Handed(A1, "another")],
            log_path=tmp_path / "a.jsonl",
        )
    with pytest.raises(InvalidRecord):
        run_job(lambda ctx: [], job_name="shape", handed=[(A1, "one")], log_path=tmp_path / "b.jsonl")  # type: ignore[list-item]
    with pytest.raises(InvalidRecord):
        run_job(lambda ctx: [], job_name="", handed=[], log_path=tmp_path / "c.jsonl")
    assert not any(tmp_path.iterdir())


def test_the_context_delivers_nothing_after_the_job_has_ended(tmp_path):
    kept = []

    def job(ctx):
        kept.append(ctx)
        return []

    result = run_job(job, job_name="late", handed=HAND_OFF, log_path=tmp_path / "job.log.jsonl")
    with pytest.raises(HarnessRefused):
        kept[0].read(A1)
    with pytest.raises(HarnessRefused):
        kept[0].call_model("m-1", "prompt")
    assert harness.read_log(tmp_path / "job.log.jsonl") == result.log


def test_a_job_that_raises_leaves_a_closed_log_and_its_error(tmp_path):
    log_path = tmp_path / "job.log.jsonl"

    def job(ctx):
        ctx.read(A1)
        raise RuntimeError("the job broke")

    with pytest.raises(RuntimeError, match="the job broke"):
        run_job(job, job_name="broken", handed=HAND_OFF, log_path=log_path)
    last = entries(log_path)[-1]
    assert (last["status"], last["error"]) == ("failed", "RuntimeError")
    assert harness.read_log(log_path).reads == (HarnessRead(A1),)
