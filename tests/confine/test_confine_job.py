"""run_confined_job: a generation job run out of process in the sandbox (plan 3.3
"Generation", Decision 13, F10, 5.8 B-1; DECISIONS.md D-07 and D-21).

Each test that takes the fixture ``sandbox`` starts a real bubblewrap sandbox and
is skipped, with the reason, where one cannot be built. The jobs are the small
scripts in ``jobs/``; each says in its first lines what it does.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import socket
from pathlib import Path

import pytest

from waterwhite import capture, confine
from waterwhite.confine import (
    CodeMount,
    ConfinedJobFailed,
    JobProtocolViolation,
    Program,
    SandboxUnavailable,
    python_program,
    run_confined_job,
)
from waterwhite.errors import (
    EXIT_USAGE,
    InvalidRecord,
    ModelClientMissing,
    WaterwhiteError,
    WriteRefused,
)
from waterwhite.harness import Handed, JobOutput, read_log, run_job
from waterwhite.records import HarnessLog, HarnessRead, OccurrenceRef

pytestmark = pytest.mark.sandbox

A1 = OccurrenceRef("a" * 64, "L1")
A2 = OccurrenceRef("a" * 64, "L2")
C1 = OccurrenceRef("c" * 64, "L1")
OUTSIDE = OccurrenceRef("b" * 64, "L1")

HAND_OFF = [Handed(A1, "first text"), Handed(A2, "second text"), Handed(C1, "third text")]

LIMIT = 30  # seconds; a job of these tests ends in well under one


def entries(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]


def ref(r: OccurrenceRef) -> dict:
    return {"instance_id": r.instance_id, "locator": r.locator}


# --- A job that reads its handed records and emits outputs ---------------------------


def test_reads_are_logged_in_order_and_the_digest_is_the_files(sandbox, jobs, tmp_path):
    log_path = tmp_path / "logs" / "job.log.jsonl"

    result = run_confined_job(
        python_program(jobs / "read_and_emit.py"),
        job_name="order",
        handed=HAND_OFF,
        log_path=log_path,
        timeout_seconds=LIMIT,
    )

    assert result.outputs == (
        JobOutput("third text + first text + third text", (C1, A1)),
        JobOutput("made by order", ()),
    )
    assert result.log == HarnessLog(
        job_name="order",
        log_digest=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        handed=(A1, A2, C1),
        reads=(HarnessRead(C1), HarnessRead(A1), HarnessRead(C1)),
        model="none",
        harness="confined",
    )
    assert result.log.bound_stated()
    assert entries(log_path) == [
        {
            "seq": 0,
            "kind": "job",
            "format": "waterwhite.harness-log/1",
            "job_name": "order",
            "handed": [ref(A1), ref(A2), ref(C1)],
            "harness": "confined",
        },
        {"seq": 1, "kind": "read", "refused": False, **ref(C1)},
        {"seq": 2, "kind": "read", "refused": False, **ref(A1)},
        {"seq": 3, "kind": "read", "refused": False, **ref(C1)},
        {"seq": 4, "kind": "output", "index": 0, "used": [ref(C1), ref(A1)]},
        {"seq": 5, "kind": "output", "index": 1, "used": []},
        {"seq": 6, "kind": "model", "model": "none", "models": []},
        {"seq": 7, "kind": "end", "status": "completed", "outputs": 2},
    ]
    data = log_path.read_text(encoding="ascii")
    for text in ("first text", "second text", "third text", "made by"):
        assert text not in data


def test_the_log_is_the_in_process_harness_log_but_for_the_harness_it_names(
    sandbox, jobs, tmp_path
):
    """The same reads and outputs through the in-process harness and through the
    sandbox give the same file, entry for entry, but for one word: the job entry
    names the harness that wrote the log. Everything downstream reads one format."""

    def in_process(ctx):
        first, _, third = ctx.handed()
        texts = [ctx.read(third), ctx.read(first), ctx.read(third)]
        return [
            JobOutput(" + ".join(texts), (third, first)),
            JobOutput("made by order", ()),
        ]

    inside = run_job(in_process, job_name="order", handed=HAND_OFF, log_path=tmp_path / "in.jsonl")
    confined = run_confined_job(
        python_program(jobs / "read_and_emit.py"),
        job_name="order",
        handed=HAND_OFF,
        log_path=tmp_path / "out.jsonl",
        timeout_seconds=LIMIT,
    )

    in_entries, out_entries = entries(tmp_path / "in.jsonl"), entries(tmp_path / "out.jsonl")
    assert (in_entries[0].pop("harness"), out_entries[0].pop("harness")) == ("in-process", "confined")
    assert out_entries == in_entries
    assert (tmp_path / "out.jsonl").read_bytes() == (tmp_path / "in.jsonl").read_bytes().replace(
        b'"harness":"in-process"', b'"harness":"confined"')
    assert confined.outputs == inside.outputs
    assert (inside.log.harness, confined.log.harness) == ("in-process", "confined")
    assert dataclasses.replace(confined.log, harness=None, log_digest="") == dataclasses.replace(
        inside.log, harness=None, log_digest="")


def test_edges_are_built_from_a_confined_result_as_from_any_other(sandbox, jobs, tmp_path):
    result = run_confined_job(
        python_program(jobs / "read_and_emit.py"),
        job_name="order",
        handed=HAND_OFF,
        log_path=tmp_path / "job.jsonl",
        timeout_seconds=LIMIT,
    )
    step = capture.step_ref("generate", "1")

    first, second = capture.generation_edges(result, [1, 2], step)

    assert [(i.parent, i.used) for i in first.inputs] == [(A1, True), (A2, False), (C1, True)]
    assert [(i.parent, i.used) for i in second.inputs] == [(A1, False), (A2, False), (C1, False)]
    assert first.is_generation and first.harness_log == result.log


def test_a_long_text_with_any_characters_passes_the_pipe_unchanged(sandbox, jobs, tmp_path):
    """A record far longer than a pipe's buffer, with characters outside ASCII, a
    line feed and a quote in it, goes in and comes back as it was."""
    text = 'Straße "quoted"\n\u00e9\u20ac\U0001f600 line two\t' * 20000
    assert len(text.encode("utf-8")) > 500_000

    result = run_confined_job(
        python_program(jobs / "echo.py"),
        job_name="echo",
        handed=[Handed(A1, text)],
        log_path=tmp_path / "job.jsonl",
        timeout_seconds=LIMIT,
    )

    assert result.outputs == (JobOutput(text, (A1,)),)
    assert (tmp_path / "job.jsonl").stat().st_size < 1000  # references, never the text


# --- A job that asks for a record outside its hand-off ------------------------------


def test_a_request_outside_the_hand_off_is_refused_and_logged(sandbox, jobs, tmp_path):
    log_path = tmp_path / "job.jsonl"

    result = run_confined_job(
        python_program(jobs / "ask_outside.py", [OUTSIDE.instance_id]),
        job_name="curious",
        handed=[Handed(A1, "first text")],
        log_path=log_path,
        timeout_seconds=LIMIT,
    )

    # Neither request delivered anything, and the job went on to its own record.
    assert result.outputs == (JobOutput("refused,refused|first text", (A1,)),)
    assert result.log.reads == (
        HarnessRead(occurrence=None, outside_ref=OUTSIDE.key, refused=True),
        HarnessRead(occurrence=None, outside_ref="/data/corpus/B.jsonl", refused=True),
        HarnessRead(A1),
    )
    assert entries(log_path)[1:4] == [
        {"seq": 1, "kind": "read", "refused": True, "outside_ref": OUTSIDE.key},
        {"seq": 2, "kind": "read", "refused": True, "outside_ref": "/data/corpus/B.jsonl"},
        {"seq": 3, "kind": "read", "refused": False, **ref(A1)},
    ]
    # A refused attempt delivered nothing, so the bound holds (DECISIONS.md D-07).
    assert result.log.bound_stated()
    # The sole handed record is the output's parent (DECISIONS.md D-21).
    edge = capture.generation_edge(1, result.outputs[0], result.log, capture.step_ref("g", "1"))
    assert [(i.parent, i.used) for i in edge.inputs] == [(A1, True)]


def test_a_text_the_job_chooses_is_bounded_in_the_log(sandbox, jobs, tmp_path):
    """What a job asked for and the model identifier it names are the two texts
    of its choosing that reach the log. Neither can carry a record at length."""
    log_path = tmp_path / "job.jsonl"

    result = run_confined_job(
        python_program(jobs / "ask_long.py"),
        job_name="long",
        handed=[Handed(A1, "first text ")],
        log_path=log_path,
        timeout_seconds=LIMIT,
        model_client=lambda model, prompt: "never called",
    )

    assert result.outputs == (JobOutput("refused|invalid record", (A1,)),)
    asked = "first text " * 500
    (refused,) = [read for read in result.log.reads if read.refused]
    assert refused.outside_ref == (
        asked[: confine.MAX_LOGGED_TEXT]
        + "...(sha256:"
        + hashlib.sha256(asked.encode("utf-8")).hexdigest()
        + ")"
    )
    assert result.log.model == "none"  # the long identifier was not logged or called
    assert result.log.bound_stated()
    assert log_path.stat().st_size < 2000


# --- A job that tries to reach a file and a network that exist outside ---------------


def look_around(jobs, tmp_path, *, code=()) -> tuple[dict, Path, socket.socket]:
    """Run the job that tries the routes a confined job must not have. Returns its
    report, the outside file and the outside listener."""
    data = tmp_path / "data"
    data.mkdir()
    outside_file = data / "corpus.jsonl"
    outside_file.write_text('{"text": "a record nobody handed the job"}\n', encoding="utf-8")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.setblocking(False)
    host, port = listener.getsockname()
    result = run_confined_job(
        python_program(jobs / "look_around.py", [str(outside_file), host, str(port)], code=code),
        job_name="look-around",
        handed=[],
        log_path=tmp_path / "job.jsonl",
        timeout_seconds=LIMIT,
    )
    (output,) = result.outputs
    assert result.log.reads == ()
    return json.loads(output.text), outside_file, listener


def test_a_file_outside_the_sandbox_is_not_there_for_the_job(sandbox, jobs, tmp_path):
    report, outside_file, listener = look_around(jobs, tmp_path)
    listener.close()

    # The file is there for this process, which is not confined.
    assert outside_file.read_text(encoding="utf-8").startswith('{"text"')
    # For the job it does not exist: not the file, and not the folder it lies in.
    assert report["file"] == {"ok": False, "error": "FileNotFoundError", "errno": 2}
    assert report["file_exists"] is False
    assert report["folder_exists"] is False
    # What the job sees at the top: devices, library folders, its scratch folder,
    # the client, and the folders on the way to its interpreter. No /etc, /proc,
    # /sys, /tmp, /run or /var, and nothing of this test's folder.
    assert {"dev", "scratch", "usr", "waterwhite"} <= set(report["root"])
    assert not {"etc", "proc", "sys", "tmp", "run", "var", "mnt", "media"} & set(report["root"])


def test_the_job_can_write_to_its_scratch_folder_and_nowhere_else(sandbox, jobs, tmp_path):
    report, _, listener = look_around(jobs, tmp_path)
    listener.close()

    assert report["write_scratch"] == {"ok": True, "value": "/scratch/scratch.txt"}
    assert report["write_root"] == {"ok": False, "error": "OSError", "errno": 30}  # read-only
    # The scratch folder lived in the sandbox only: nothing of it is on this machine.
    assert not Path("/scratch/scratch.txt").exists()
    # A user namespace of its own, from which it could mount, is denied, and it
    # holds no privilege over the namespaces it runs in.
    assert report["new_user_namespace"]["ok"] is False
    assert report["set_hostname"] == {"ok": False, "error": "PermissionError", "errno": 1}


def test_the_job_inherits_no_open_file_but_its_two_pipes(sandbox, jobs, tmp_path):
    """The harness holds its log file open while the job runs; the job must not
    get that descriptor or any other of the caller's."""
    report, _, listener = look_around(jobs, tmp_path)
    listener.close()

    assert len(report["pipes"]) == 2
    assert report["descriptors"] == sorted({0, 1, 2, *report["pipes"]})


def test_the_job_has_no_network(sandbox, jobs, tmp_path):
    report, _, listener = look_around(jobs, tmp_path)

    try:
        # The job's connection attempt failed, and nothing arrived at the listener.
        assert report["connect"]["ok"] is False
        with pytest.raises(BlockingIOError):
            listener.accept()
        assert report["connect_elsewhere"]["ok"] is False
        assert report["resolve"]["ok"] is False
        assert report["interfaces"] == ["lo"]  # its own loopback and no other device
        # The listener can be reached where there is a network: from this process.
        with socket.create_connection(listener.getsockname(), timeout=5):
            listener.setblocking(True)
            listener.settimeout(5)
            accepted, _ = listener.accept()
            accepted.close()
    finally:
        listener.close()


def test_the_job_gets_nothing_of_the_callers_environment(sandbox, jobs, tmp_path, monkeypatch):
    monkeypatch.setenv("WATERWHITE_TEST_SECRET", "a key of the caller")

    report, _, listener = look_around(jobs, tmp_path)
    listener.close()

    assert report["environment"] == [
        "HOME",
        "LANG",
        "PWD",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "TMPDIR",
        "WATERWHITE_JOB_FDS",
    ]


# --- A job that calls a model -------------------------------------------------------


def test_a_model_call_goes_to_the_callers_client_and_its_identifier_is_logged(
    sandbox, jobs, tmp_path
):
    log_path = tmp_path / "job.jsonl"
    calls: list[tuple[str, str]] = []

    def stand_in(model: str, prompt: str) -> str:
        # Runs in this process, outside the sandbox: the job needs no network.
        calls.append((model, prompt))
        return f"<{model} answers {len(calls)}>"

    result = run_confined_job(
        python_program(jobs / "call_model.py"),
        job_name="with-model",
        handed=[Handed(A1, "first text")],
        log_path=log_path,
        timeout_seconds=LIMIT,
        model_client=stand_in,
    )

    assert calls == [
        ("stand-in-1", "rewrite: first text"),
        ("stand-in-2", "shorten: <stand-in-1 answers 1>"),
        ("stand-in-1", "once more"),
    ]
    # The reserved identifier "none" was refused to the job, not logged, not called.
    assert result.outputs == (
        JobOutput(
            "<stand-in-1 answers 1>|<stand-in-2 answers 2>|<stand-in-1 answers 3>|invalid record",
            (A1,),
        ),
    )
    assert result.log.model == '["stand-in-1","stand-in-2"]'
    assert entries(log_path)[1:] == [
        {"seq": 1, "kind": "read", "refused": False, **ref(A1)},
        {"seq": 2, "kind": "model_call", "model": "stand-in-1"},
        {"seq": 3, "kind": "model_call", "model": "stand-in-2"},
        {"seq": 4, "kind": "model_call", "model": "stand-in-1"},
        {"seq": 5, "kind": "output", "index": 0, "used": [ref(A1)]},
        {
            "seq": 6,
            "kind": "model",
            "model": '["stand-in-1","stand-in-2"]',
            "models": ["stand-in-1", "stand-in-2"],
        },
        {"seq": 7, "kind": "end", "status": "completed", "outputs": 1},
    ]
    # No prompt and no reply is in the log.
    assert "rewrite" not in log_path.read_text(encoding="ascii")


def test_a_model_call_with_no_client_is_logged_and_ends_the_run(sandbox, jobs, tmp_path):
    log_path = tmp_path / "job.jsonl"

    with pytest.raises(ModelClientMissing):
        run_confined_job(
            python_program(jobs / "call_model.py"),
            job_name="with-model",
            handed=[Handed(A1, "first text")],
            log_path=log_path,
            timeout_seconds=LIMIT,
        )

    assert entries(log_path)[2:] == [
        {"seq": 2, "kind": "model_call", "model": "stand-in-1"},
        {"seq": 3, "kind": "model", "model": "stand-in-1", "models": ["stand-in-1"]},
        {"seq": 4, "kind": "end", "status": "failed", "error": "no model client"},
    ]


def test_an_error_of_the_callers_model_client_ends_the_run_and_is_passed_on(
    sandbox, jobs, tmp_path
):
    log_path = tmp_path / "job.jsonl"

    def broken(model: str, prompt: str) -> str:
        raise RuntimeError("the model service is down")

    with pytest.raises(RuntimeError, match="the model service is down"):
        run_confined_job(
            python_program(jobs / "call_model.py"),
            job_name="with-model",
            handed=[Handed(A1, "first text")],
            log_path=log_path,
            timeout_seconds=LIMIT,
            model_client=broken,
        )

    assert entries(log_path)[-1] == {
        "seq": 4,
        "kind": "end",
        "status": "failed",
        "error": "RuntimeError",
    }


def test_a_model_client_that_returns_no_text_is_refused(sandbox, jobs, tmp_path):
    with pytest.raises(InvalidRecord, match="a model client returns text"):
        run_confined_job(
            python_program(jobs / "call_model.py"),
            job_name="with-model",
            handed=[Handed(A1, "first text")],
            log_path=tmp_path / "job.jsonl",
            timeout_seconds=LIMIT,
            model_client=lambda model, prompt: 5,
        )


# --- A job that does not end as a job must: a named error and no outputs -------------


def failed_run(jobs, tmp_path, script, expected, *args, timeout=LIMIT):
    """Run a job that must fail. No JobResult exists afterwards, so there are no
    outputs to write and nothing to build an edge from. Returns the error and the
    entries of the log, which is closed as failed."""
    log_path = tmp_path / "job.jsonl"
    with pytest.raises(expected) as raised:
        run_confined_job(
            python_program(jobs / script, args),
            job_name="failing",
            handed=[Handed(A1, "first text")],
            log_path=log_path,
            timeout_seconds=timeout,
        )
    written = entries(log_path)
    assert written[-1] == {
        "seq": len(written) - 1,
        "kind": "end",
        "status": "failed",
        "error": raised.value.name,
    }
    assert not [entry for entry in written if entry["kind"] == "output"]
    assert isinstance(raised.value, WaterwhiteError)
    return raised.value, written


def test_a_job_that_exits_non_zero_gives_a_named_error_and_no_outputs(sandbox, jobs, tmp_path):
    error, written = failed_run(jobs, tmp_path, "exit_nonzero.py", ConfinedJobFailed)

    assert error.name == "confined job failed"
    assert error.exit_code == EXIT_USAGE
    assert error.context["status"] == 3
    assert "the job gives up" in error.detail  # what the job said last
    # The read it made before it failed is still in the log.
    assert written[1] == {"seq": 1, "kind": "read", "refused": False, **ref(A1)}
    assert read_log(tmp_path / "job.jsonl").reads == (HarnessRead(A1),)


def test_garbage_on_the_pipe_gives_a_named_error_and_no_outputs(sandbox, jobs, tmp_path):
    error, _ = failed_run(jobs, tmp_path, "garbage.py", JobProtocolViolation)

    assert error.name == "job protocol violation"
    assert "not JSON" in error.detail


def test_a_job_that_exits_without_the_end_message_gives_no_outputs(sandbox, jobs, tmp_path):
    error, _ = failed_run(jobs, tmp_path, "no_end.py", ConfinedJobFailed)

    assert "without the end message" in error.detail


def test_a_job_that_runs_past_its_time_limit_is_stopped(sandbox, jobs, tmp_path):
    error, _ = failed_run(jobs, tmp_path, "hang.py", ConfinedJobFailed, timeout=1.5)

    assert "time limit of 1.5 seconds" in error.detail


def test_an_output_that_states_a_record_not_handed_is_refused(sandbox, jobs, tmp_path):
    error, _ = failed_run(jobs, tmp_path, "bad_used.py", InvalidRecord, OUTSIDE.instance_id)

    assert OUTSIDE.key in error.detail


def test_a_program_that_is_no_job_fails_by_name(sandbox, tmp_path):
    """A program that never speaks the protocol: here, a command that is not in
    the sandbox at all."""
    with pytest.raises(ConfinedJobFailed):
        run_confined_job(
            Program(argv=("/usr/bin/no-such-command",)),
            job_name="absent",
            handed=[],
            log_path=tmp_path / "job.jsonl",
            timeout_seconds=LIMIT,
        )


# --- The harness refuses to run without its sandbox ---------------------------------


@pytest.mark.parametrize("path_value", ["", None], ids=["empty PATH", "no PATH"])
def test_without_the_sandbox_command_nothing_runs(jobs, tmp_path, monkeypatch, path_value):
    """No fixture ``sandbox``: this holds on a machine with no bubblewrap too."""
    log_path = tmp_path / "job.jsonl"
    if path_value is None:
        monkeypatch.delenv("PATH", raising=False)
    else:
        monkeypatch.setenv("PATH", path_value)

    with pytest.raises(SandboxUnavailable) as raised:
        run_confined_job(
            python_program(jobs / "read_and_emit.py"),
            job_name="order",
            handed=HAND_OFF,
            log_path=log_path,
            timeout_seconds=LIMIT,
        )

    assert raised.value.name == "sandbox unavailable"
    assert raised.value.exit_code == EXIT_USAGE
    assert "not on PATH" in raised.value.detail
    # Refused before anything was started or written: no log claims a run.
    assert not log_path.exists()


def test_a_sandbox_command_that_cannot_build_the_sandbox_is_unavailable(tmp_path, monkeypatch):
    """A command named bwrap that fails, as bubblewrap does where user namespaces
    are switched off, is reported and nothing runs."""
    fake = tmp_path / "bin" / "bwrap"
    fake.parent.mkdir()
    fake.write_text(
        "#!/bin/sh\necho 'bwrap: No permissions to create new namespace' >&2\nexit 1\n",
        encoding="ascii",
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake.parent))

    with pytest.raises(SandboxUnavailable) as raised:
        confine.find_sandbox()

    assert "No permissions to create new namespace" in raised.value.detail


# --- What the harness refuses before it starts anything ------------------------------


def test_an_existing_log_is_not_overwritten(sandbox, jobs, tmp_path):
    log_path = tmp_path / "job.jsonl"
    log_path.write_text("an earlier log\n", encoding="ascii")

    with pytest.raises(WriteRefused):
        run_confined_job(
            python_program(jobs / "read_and_emit.py"),
            job_name="order",
            handed=HAND_OFF,
            log_path=log_path,
            timeout_seconds=LIMIT,
        )

    assert log_path.read_text(encoding="ascii") == "an earlier log\n"


@pytest.mark.parametrize(
    "target",
    ["relative/path", "/", "/scratch/code", "/dev", "/waterwhite/client", "/waterwhite", "/a/../b"],
)
def test_a_mount_target_that_would_cover_the_sandboxs_own_folders_is_refused(
    jobs, tmp_path, target
):
    with pytest.raises(InvalidRecord):
        run_confined_job(
            python_program(jobs / "read_and_emit.py", code=[CodeMount(str(jobs), target)]),
            job_name="order",
            handed=HAND_OFF,
            log_path=tmp_path / "job.jsonl",
            timeout_seconds=LIMIT,
        )
    assert not (tmp_path / "job.jsonl").exists()


@pytest.mark.parametrize("limit", [0, -1, None, True, "10"])
def test_a_run_needs_a_time_limit(jobs, tmp_path, limit):
    with pytest.raises(InvalidRecord):
        run_confined_job(
            python_program(jobs / "read_and_emit.py"),
            job_name="order",
            handed=HAND_OFF,
            log_path=tmp_path / "job.jsonl",
            timeout_seconds=limit,
        )


# --- The command line and the client, with no sandbox started ------------------------


def test_the_command_line_mounts_code_only_and_shares_no_namespace(jobs):
    program = python_program(jobs / "read_and_emit.py")
    argv = confine.Sandbox(command="/usr/bin/bwrap").argv(program, request_fd=7, reply_fd=8)
    options, command = argv[: argv.index("--")], argv[argv.index("--") + 1 :]

    for namespace in ("user", "ipc", "pid", "net", "uts", "cgroup"):
        assert f"--unshare-{namespace}" in options
    for option in ("--clearenv", "--die-with-parent", "--new-session", "--disable-userns"):
        assert option in options
    # Nothing is mounted writable from this machine, and nothing shares a namespace.
    assert not {"--bind", "--dev-bind", "--bind-try", "--share-net", "--proc"} & set(options)
    assert not [option for option in options if option.endswith("-try")]
    # What is mounted read-only: library folders, the interpreter, the script, the client.
    sources = [options[i + 1] for i, option in enumerate(options) if option == "--ro-bind"]
    allowed = {"/usr/lib", "/usr/lib64", "/lib", "/lib64"}
    allowed |= {mount.source for mount in program.mounts}
    allowed.add(str(Path(confine.__file__).with_name("waterwhite_job.py")))
    assert set(sources) <= allowed
    assert str(jobs) not in sources  # the script's file, not its folder
    assert options[options.index("--tmpfs") + 1] == "/scratch"
    assert options[options.index("WATERWHITE_JOB_FDS") + 1] == "7,8"
    assert command[1:] == ["-s", "-B", "/waterwhite/job/read_and_emit.py"]


def test_the_job_side_client_needs_the_standard_library_only():
    source = Path(confine.__file__).with_name("waterwhite_job.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the client is mounted alone and has no package"
            imported.add(node.module)
    assert imported == {"__future__", "json", "os", "typing"}
    # The two sides name the protocol and the variable alike.
    assert f'PROTOCOL = "{confine.PROTOCOL}"' in source
    assert 'ENV_FDS = "WATERWHITE_JOB_FDS"' in source
