"""The runner of Check 10 (plan 5.5 Check 10; table README, section "Check 10", the six
subjects, "What a run hands a subject, and its conditions" and "A failed attempt, the two routes
after it, and when Check 10 is complete"): ``baselines.runner``, ``baselines.__main__`` and the
files one run leaves.

Nothing of a tool is started here. Every command the runner would start goes to a stand-in
starter that records it, every subject is a stand-in that reads hand-written values, and every
file lives under ``tmp_path``. The one exception is the group of tests of the lock prefix, near
the end: they start bash, flock and sleep through the real starter, behind a prefix script of
their own and on a scratch lock file inside their own folder, never the lock of the machine.
The values these tests expect come from the table, the README and the documents of the tools,
never from running the code under test.

The package module does not exist when these tests are written, so each test asks for it with
``need`` or ``need_attr`` in its own body: a missing module is a failed assertion.

The names this file asks of ``baselines.runner`` (a starter is whatever the caller hands in: it
has ``run(argv, timeout=None)`` giving an object with ``exit_code`` (None when the command did
not end by itself), ``output`` and ``seconds``, and ``start(argv)`` giving a handle):

- ``PreRunStop`` (an exception with ``stage``, one of "compare", "identity", "probe", "binary",
  "putter", "request-files", and a non-empty ``reason``) and ``GateClosed`` (a ``RuntimeError``).
- ``GatedStarter(inner)``: ``run``, ``start``, ``permit_probe(argv)``, ``release()`` and
  ``released``. Until released it starts nothing but an argv that was permitted as the probe.
- ``pre_run(subject, *, env_dir, env_record, identity_sha256, work, lock_prefix, starter,
  run_record, ob_binary=None, log=None)`` gives a ``PreRun`` with ``interpreter_sha256``,
  ``ob_binary_sha256`` and ``step``. ``starter`` is a ``GatedStarter``. In this order: the copy's
  ``compare.json`` in ``env_record`` says equal true and ``identity.json`` is there; the file's own
  sha256 is the one handed in; the copy's python alone, inside the sandbox, writes the path it
  resolves to and the sha256 of that binary into a file of the work folder; that sha256 is the
  ``interpreter_sha256`` of ``identity.json``; for ``originblame`` the program's own sha256 is the
  ``ob_binary_sha256`` of ``identity.json``. Only then is the gate released. The run record gets
  ``<subject>/compare.json`` (a copy) and ``<subject>/interpreter.txt`` (the two lines the probe
  wrote). ``identity.json`` is read and never written.
- ``check_putters(questions, putter)`` and ``check_request_files(requests, builds)``.
- ``StepLog(folder, roots=None)`` (``steps.jsonl`` and ``kept/`` inside ``folder``; ``add``,
  ``steps``, ``shared(target)``, ``reading(question)``, ``step_ids(question)``),
  ``RecordingRunner(starter, log, cap_seconds=...)`` (``run``, ``start``), ``ProcessHandle``,
  ``StopRecord`` and ``stop_record``, ``attempt_of(step, attempt_id)``.
- ``answer_question(question, subject, *, commit, log, grade_of, grade_layouts_of=None,
  failed_runs=None, notes=None)`` gives an ``Answered`` (``line``, ``attempt``, ``step_ids``);
  ``none_lines(questions, attempts, *, commit, step_ids_of=None)``.
- ``put_dolma_fault(fault_id, companion_file, chunk_file)`` (True when put, False for the three
  faults that are put elsewhere), ``ob_fault_putter(copy, fault_id)`` (the putter the OriginBlame
  subject calls; it gives the record of the binary index), ``binary_index_state(store)``,
  ``prepare_tokenizer_folders(build, work_dir=..., tmpdir=...)``,
  ``prepare_mixer_folders(build, work_dir=...)``, ``place_tokenizer(build, source, name=...)``,
  ``rows_after_stop(folder)``, ``sample_row_count(document, segment, step, sample_pos)`` and
  ``IdMap(build)`` (``file_of(id)``, ``files_read``).
- ``ProcessHandle(process, kill=os.kill, signal_number=15)`` (``pid``, ``readline()``, ``stop()``
  giving the signal number sent, ``wait()`` giving an object with ``exit_code`` and the output
  not yet read), ``stop_record(command=, sent=, ended=, attempt_id=, rows_showed_while_running=,
  output_before=)`` giving a ``StopRecord`` whose fields are the six ``stopped_*`` and
  ``rows_showed_while_running`` names of ``DolmaRun`` and ``StoppedWriter``.
- ``RunConfig(commit, work, run_record, lock_prefix, subjects, step_cap_seconds, setups,
  factories, questions, grade_of, grade_layouts_of=None, attempts=(), fault_putter=None,
  requests=None, builds=None)`` (a dataclass), ``ToolSetup(env_dir, env_record,
  identity_sha256, ob_binary=None)`` and ``run_pass(config, starter)`` giving an object with
  ``lines``; a factory is called with a ``SubjectContext`` once the gate of its subject is open.
  ``PreRunStop`` leaves ``run_pass`` as an exception. Before any start the pass checks the
  putters (``check_putters``) and, where ``requests`` and ``builds`` are given, the request files
  (``check_request_files``).
- A ``SubjectContext`` has ``runner`` (a ``RecordingRunner``), ``log`` (the ``StepLog``), ``work``,
  ``run_record``, ``ran_on`` (a set a factory adds the targets its subject ran on to) and
  ``failed_runs`` (a dict from a target to the failed ``Attempt`` of the run on it). The pass
  writes the one attempt of the subject on each of those targets (``store_attempts`` of its
  result) and from them the none lines of the subject's questions; a failed attempt is kept in
  ``attempts.jsonl`` with the ids of the questions that cite it.
- Every step the pass keeps has its work folder, each environment copy, each build and the
  interpreter bind folders replaced by placeholders: a step line that still holds a local path is
  refused when it is written (the folders of a test run sit under a home folder).
- ``retry_record(command=, ended=, attempt_id=)`` gives a ``RetryRecord`` whose fields are the
  ``retry_attempt_id``, ``retry_command``, ``retry_exit_code`` and ``retry_output`` names of
  ``DolmaRun``.
- ``seal_run_record``, ``start_line``, ``parse_arguments``, ``main``, ``RunConfig``,
  ``ToolSetup``, ``SubjectContext`` and ``run_pass``.

``baselines.record`` gains, for the steps, ``step_ids`` on ``answer_line`` and ``attempt_line``
(see ``test_record.py``).

The frozen program: ``pre_run`` takes ``program_sha256=None`` (the sha256 of the program that
the caller supplied; it is compared with the program's own sha256 at the stage ``binary``,
never read from ``identity.json``), ``ToolSetup`` has a field ``program_sha256`` and the
command has the option ``--program-sha256``.

The lock prefix (every command of a run starts behind a prefix that serialises it; the prefix
reads two environment values and writes one line to a note file): ``RecordingRunner`` and
``RunConfig`` take ``lock_wait_bound`` (seconds, ``None`` where no prefix is used) and the command
has the option ``--lock-wait-bound``. For each step the runner makes a note file inside its own
run record folder, and hands the starter ``lock_note`` (the file) and ``lock_wait_bound`` as
keywords of ``run(argv, timeout=None, ...)`` and ``start(argv, ...)``; ``GatedStarter`` hands them
on. The real starter sets ``WATERWHITE_LOCK_NOTE`` and ``WATERWHITE_LOCK_WAIT_BOUND`` on the
process it starts. The prefix writes ``acquired <seconds waited>`` to the note once it holds the
lock. The step's cap counts from that line, never before; the wait is the step's
``lock_wait_seconds`` (and ``StepResult.lock_wait_seconds``), a field of its line in
``steps.jsonl``, and is in neither the output nor the seconds of the step. A note with no such
line after the process ended is a wait bound that ran out: ``LockWaitStop`` is raised, no step is
kept for it and no answer line is written. A starter's result may carry ``lock_wait_seconds`` and
``lock_acquired``; a result without them (the stand-ins above) is a step with no lock.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import gzip
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"

# A commit is passed as a hand-written string; no git call is made.
COMMIT = "0123456789abcdef0123456789abcdef01234567"
LOCK = ["lock-prefix"]

# What stays out of a record, written in pieces so that this file holds none of it.
PRIVATE_PATH = "/" + "ho" + "me/" + "someone/work/build"

SCRATCH_500 = ("store", "scratch-500")
MAIN = ("store", "main")


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    found = getattr(need(module_name), attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _fn(name):
    return need_attr("baselines.runner", name)


# --- The table, as the record tests read it ------------------------------------------------------


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _read_toml(TABLE_DIR / "check10.toml")["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, tuple[str, int]]:
    found = {}
    for group in _read_toml(TABLE_DIR / "check10.toml")["group"]:
        for listed in group["rows"]:
            found[listed] = (group["id"], group["check"])
    return found


@lru_cache(maxsize=1)
def _group_rows() -> dict[str, list[str]]:
    return {group["id"]: list(group["rows"]) for group in _read_toml(TABLE_DIR / "check10.toml")["group"]}


def _question_id(listed_row: str, subject: str, size=None) -> str:
    base = f"{listed_row}|{subject}"
    return base if size is None else f"{base}@{size}"


def _make_question(listed_row: str, subject: str, size=None):
    """The question that puts ``listed_row`` to ``subject`` at ``size``, its fields taken from
    the subject's row of the group in ``check10.toml``."""
    question_type = need_attr("baselines.interfaces", "Question")
    group, check = _group_of_listed()[listed_row]
    row = _check10_rows()[f"c10.{group}.{subject}"]
    ordinal = row.get("ordinal")
    return question_type(
        id=_question_id(listed_row, subject, size),
        group=group,
        row_id=row["id"],
        listed_row=listed_row,
        subject=subject,
        check=check,
        size=size,
        excluded_from_tally=row["excluded_from_tally"],
        route=row.get("route"),
        missing=row.get("missing"),
        needs_ordinal=row["needs_ordinal"],
        marks=tuple(row["marks"]),
        tie_mark=ordinal["shown"]["mark"] if ordinal is not None else None,
    )


def _reading(question, raw):
    return need_attr("baselines.interfaces", "Reading")(question_id=question.id, raw=raw, route=question.route)


def _grade(status, evidence=None):
    return need_attr("baselines.interfaces", "Grade")(status=status, evidence=evidence)


def _attempt(attempt_id="attempt-1", exit_code=1, output="no companion row was written", command=("tool", "run")):
    return need_attr("baselines.interfaces", "Attempt")(
        id=attempt_id, command=tuple(command), exit_code=exit_code, output=output
    )


def _store_attempt(subject, target, attempt_id=None, run=None, layout=None):
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    fields = dict(subject=subject, target=target, attempt_id=attempt_id, recorded="the attempt's own record")
    if layout is not None:
        fields["layout"] = layout
    if run is not None:
        names = {field.name for field in dataclasses.fields(attempt_type)}
        assert "run" in names, "StoreAttempt has no run field"
        fields["run"] = run
    return attempt_type(**fields)


# --- Stand-ins: a starter, a process, a subject ----------------------------------------------------


def _result(code, output="", seconds=1.0):
    return SimpleNamespace(exit_code=code, output=output, seconds=seconds)


def _is_probe(argv) -> bool:
    return "-I" in argv and "-c" in argv


class Starter:
    """Starts nothing: it records every argv and every timeout, and answers from a script. The
    probe (the copy's python alone, with ``-I -c``) writes the two lines of its record file,
    which is the last argument, when it was given ``probe=(path, sha)``."""

    def __init__(self, *, probe=None, results=None, handles=None):
        self.argvs = []
        self.timeouts = []
        self.probe = probe
        self.results = list(results or [])
        self.handles = list(handles or [])

    def run(self, argv, timeout=None):
        self.argvs.append(list(argv))
        self.timeouts.append(timeout)
        if _is_probe(argv) and self.probe is not None:
            Path(argv[-1]).write_text(f"{self.probe[0]}\n{self.probe[1]}\n", encoding="utf-8")
            return _result(0, "", 0.25)
        if self.results:
            return self.results.pop(0)
        return _result(0, "", 1.0)

    def start(self, argv):
        self.argvs.append(list(argv))
        return self.handles.pop(0)


class _Process:
    """What a started process gives: a pid, a stream of its output and a wait."""

    def __init__(self, pid, text, code):
        self.pid = pid
        self.stdout = io.StringIO(text)
        self._code = code

    def wait(self):
        return self._code


class Subject:
    """A subject that reads one hand-written raw value, or calls ``steps`` first (each a command
    run through the recording runner), or raises."""

    def __init__(self, name="dolma", raw=None, error=None, outcome=None, steps=(), recorder=None, tied=False):
        self.name = name
        self.raw = raw if raw is not None else {"answer": "contains"}
        self.error = error
        self.outcome = outcome
        self.steps = list(steps)
        self.recorder = recorder
        self.tied = tied
        self.calls = 0

    def read(self, question):
        self.calls += 1
        for argv in self.steps:
            self.recorder.run(argv)
        if self.error is not None:
            raise self.error
        if self.outcome is not None:
            return self.outcome
        return _reading(question, self.raw)

    def used_tie(self, question):
        return self.tied


def _log(tmp_path, name="record", roots=None):
    return _fn("StepLog")(tmp_path / name, roots=roots)


def _recorder(tmp_path, starter=None, cap=600.0, roots=None):
    log = _log(tmp_path, roots=roots)
    starter = starter if starter is not None else Starter()
    return _fn("RecordingRunner")(starter, log, cap_seconds=cap), log, starter


# --- A stand-in setup: a copied environment, its record, a program ------------------------------------


def _world(tmp_path, *, name="dolma", compare="equal", identity=True, program=False, program_in_identity=True):
    """The folders a pre-run reads: a work folder with the copy of an environment (its python a
    link to a binary outside the work folder), the setup's record folder (``compare.json``,
    ``identity.json`` and another file), and a run record folder."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    work = tmp_path / "work"
    work.mkdir()
    uv = tmp_path / "uv" / "cpython-3.12.13" / "bin"
    uv.mkdir(parents=True)
    binary = uv / "python"
    binary.write_bytes(b"stand-in interpreter")
    env = work / f"env-{name}"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(binary)
    (env / "pyvenv.cfg").write_text(f"home = {uv}\nversion_info = 3.12.13.final.0\n", encoding="utf-8")
    sha = hashlib.sha256(b"stand-in interpreter").hexdigest()
    record = tmp_path / "setup-record" / name
    record.mkdir(parents=True)
    (record / "rows.tsv").write_text("package\tpath\tsha256\tverdict\n", encoding="utf-8")
    compare_file = record / "compare.json"
    if compare == "equal":
        compare_file.write_text(json.dumps({"equal": True, "differences": []}, indent=2) + "\n", encoding="utf-8")
    elif compare == "different":
        body = {"equal": False, "differences": [{"path": "bin/python", "reason": "the bytes differ"}]}
        compare_file.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    elif compare == "text":
        compare_file.write_text(json.dumps({"equal": "true", "differences": []}) + "\n", encoding="utf-8")
    program_file = None
    program_sha = None
    fields = {"env": f"venv-{name}", "interpreter_sha256": sha, "files_checked": 3}
    if program:
        program_file = tmp_path / "ob-program"
        program_file.write_bytes(b"stand-in program")
        program_sha = hashlib.sha256(b"stand-in program").hexdigest()
        if program_in_identity:
            fields["ob_binary_sha256"] = program_sha
    identity_file = record / "identity.json"
    if identity:
        identity_file.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
        identity_sha = hashlib.sha256(identity_file.read_bytes()).hexdigest()
    else:
        identity_sha = hashlib.sha256(b"no identity file was written").hexdigest()
    run_record = tmp_path / "run-record"
    run_record.mkdir()
    return SimpleNamespace(
        work=work, env=env, record=record, compare_file=compare_file, identity_file=identity_file,
        identity_sha=identity_sha, probe_path=str(binary), probe_sha=sha, program=program_file,
        program_sha=program_sha, run_record=run_record, name=name,
    )


def _gate(starter):
    return _fn("GatedStarter")(starter)


def _pre_run(w, gated, subject=None, **changes):
    arguments = dict(
        env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha, work=w.work,
        lock_prefix=LOCK, starter=gated, run_record=w.run_record,
    )
    if w.program is not None:
        arguments["ob_binary"] = w.program
    arguments.update(changes)
    return _fn("pre_run")(subject or w.name, **arguments)


# --- The pre-run condition --------------------------------------------------------------------------


def test_an_identity_file_whose_own_sha_is_not_the_one_handed_in_is_refused_before_any_start(tmp_path):
    w = _world(tmp_path)
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    gated = _gate(starter)
    before = w.identity_file.read_bytes()
    with pytest.raises(_fn("PreRunStop")) as caught:
        _pre_run(w, gated, identity_sha256=hashlib.sha256(b"another file").hexdigest())
    assert caught.value.stage == "identity"
    assert isinstance(caught.value.reason, str) and caught.value.reason
    assert starter.argvs == []
    assert gated.released is False
    assert w.identity_file.read_bytes() == before


def test_a_probe_that_reports_another_interpreter_stops_the_run_before_any_tool_command(tmp_path):
    w = _world(tmp_path)
    retargeted = hashlib.sha256(b"an interpreter the link was retargeted to").hexdigest()
    starter = Starter(probe=(w.probe_path, retargeted))
    gated = _gate(starter)
    with pytest.raises(_fn("PreRunStop")) as caught:
        _pre_run(w, gated)
    assert caught.value.stage == "probe" and caught.value.reason
    # Only the probe was started: no tool command was built or handed to the starter.
    assert len(starter.argvs) == 1 and _is_probe(starter.argvs[0])
    assert gated.released is False
    with pytest.raises(_fn("GateClosed")):
        gated.run(["a-tool-command"])
    assert len(starter.argvs) == 1
    # No answer line was written for a stopped run.
    assert list(w.run_record.rglob("answers*")) == []


def test_a_matching_probe_opens_the_gate_and_the_run_record_holds_the_compare_and_the_probe(tmp_path):
    w = _world(tmp_path)
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    gated = _gate(starter)
    got = _pre_run(w, gated)
    assert got.interpreter_sha256 == w.probe_sha
    assert gated.released is True
    assert len(starter.argvs) == 1
    argv = starter.argvs[0]
    # The probe runs behind the lock, inside the sandbox, as the copy's own python alone.
    assert argv[:1] == LOCK
    assert "--unshare-all" in argv
    assert argv[argv.index("-I") - 1] == str(w.env / "bin" / "python")
    assert argv[-1].startswith(str(w.work))
    # It is not the sandbox module's record-then-exec shell, which compares nothing before it starts a tool.
    assert need_attr("baselines.sandbox", "_RECORD_SCRIPT") not in argv
    # The run record keeps the compare result as a file, and the two lines of the probe.
    kept = w.run_record / "dolma"
    assert (kept / "compare.json").read_bytes() == w.compare_file.read_bytes()
    assert (kept / "interpreter.txt").read_text(encoding="utf-8") == f"{w.probe_path}\n{w.probe_sha}\n"
    # The setup's identity file is read, and nothing of it is written.
    assert w.identity_file.read_bytes() and json.loads(w.identity_file.read_text())["interpreter_sha256"] == w.probe_sha


@pytest.mark.parametrize(
    "case, stage",
    [
        ("absent", "compare"),
        ("different", "compare"),
        ("text", "compare"),
        ("no-identity", "identity"),
    ],
)
def test_no_tool_starts_unless_the_compare_says_equal_and_the_identity_file_is_there(tmp_path, case, stage):
    # The record folder and some of its files exist in every case; none of that starts a tool.
    changes = {"absent": dict(compare="absent"), "different": dict(compare="different"),
               "text": dict(compare="text"), "no-identity": dict(identity=False)}[case]
    w = _world(tmp_path, **changes)
    assert w.record.is_dir() and (w.record / "rows.tsv").is_file()
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    gated = _gate(starter)
    with pytest.raises(_fn("PreRunStop")) as caught:
        _pre_run(w, gated)
    assert caught.value.stage == stage and caught.value.reason
    assert starter.argvs == []
    assert gated.released is False


def test_the_runner_never_writes_or_rewrites_the_identity_file(tmp_path):
    w = _world(tmp_path)
    before = w.identity_file.read_bytes()
    stamp = w.identity_file.stat().st_mtime_ns
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    _pre_run(w, _gate(starter))
    assert w.identity_file.read_bytes() == before
    assert w.identity_file.stat().st_mtime_ns == stamp
    # A copy in the run record, if the runner keeps one, is the file as it was.
    for copy in w.run_record.rglob("identity.json"):
        assert copy.read_bytes() == before


def test_the_runner_holds_no_hash_literal_and_no_route_that_skips_the_order(tmp_path):
    module = need("baselines.runner")
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert re.search(r"[0-9a-fA-F]{32,}", source) is None, "runner.py holds a hash literal"
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    names = set()
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            strings.append(node.value)
    # The sandbox's own record-then-exec shell is not a route of the runner: it makes no compare.
    assert "recording_command" not in names and "_RECORD_SCRIPT" not in names
    # A process is stopped by its pid, never by a name or a pattern.
    for word in ("pkill", "killall"):
        assert not any(word in text for text in strings), word


def test_a_closed_gate_starts_nothing_but_the_one_probe_it_was_told_of(tmp_path):
    inner = SimpleNamespace(argvs=[])

    def run(argv, timeout=None):
        inner.argvs.append(list(argv))
        return _result(0, "", 0.1)

    def start(argv):
        inner.argvs.append(list(argv))
        return object()

    inner.run, inner.start = run, start
    gated = _gate(inner)
    closed = _fn("GateClosed")
    assert gated.released is False
    with pytest.raises(closed):
        gated.run(["tool", "--go"])
    with pytest.raises(closed):
        gated.start(["tool", "--go"])
    assert inner.argvs == []
    probe = ["probe", "-I", "-c", "text", "record-file"]
    gated.permit_probe(probe)
    gated.run(probe)
    assert inner.argvs == [probe]
    # Another command still waits for the release, and so does a second start of anything else.
    with pytest.raises(closed):
        gated.run(["probe", "-I", "-c", "text", "another-record-file"])
    assert inner.argvs == [probe]
    gated.release()
    assert gated.released is True
    gated.run(["tool", "--go"])
    assert inner.argvs == [probe, ["tool", "--go"]]


def test_the_program_of_the_tool_is_compared_with_the_sha_its_identity_file_names(tmp_path):
    stop_type = _fn("PreRunStop")
    # The program's sha256 is read by the runner from the file it was given.
    w = _world(tmp_path / "match", name="originblame", program=True)
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    got = _pre_run(w, _gate(starter))
    assert got.ob_binary_sha256 == w.program_sha
    # A program that is not the one identity.json names stops the run before any tool command.
    other = _world(tmp_path / "other", name="originblame", program=True)
    other.program.write_bytes(b"a program that was swapped")
    starter = Starter(probe=(other.probe_path, other.probe_sha))
    gated = _gate(starter)
    with pytest.raises(stop_type) as caught:
        _pre_run(other, gated)
    assert caught.value.stage == "binary" and caught.value.reason
    assert gated.released is False
    assert all(_is_probe(argv) for argv in starter.argvs)
    # An identity file that names no program leaves nothing to compare it with: stopped.
    bare = _world(tmp_path / "bare", name="originblame", program=True, program_in_identity=False)
    starter = Starter(probe=(bare.probe_path, bare.probe_sha))
    gated = _gate(starter)
    with pytest.raises(stop_type) as caught:
        _pre_run(bare, gated)
    assert caught.value.stage == "binary"
    assert gated.released is False


# --- Before any run: the fault putters and the request files ---------------------------------------------

FAULT_ROWS = (
    ("c05.f.map-deleted.E/shard-1", "f.map-deleted"),
    ("c05.f.digest.E/shard-3", "f.digest"),
    ("c04.count-499", "l.count-499"),
)


@pytest.mark.parametrize("listed, fault", FAULT_ROWS)
def test_a_faulted_row_of_the_originblame_subject_needs_a_putter_before_any_run(listed, fault):
    check = _fn("check_putters")
    stop_type = _fn("PreRunStop")
    question = _make_question(listed, "originblame")
    assert question.route is not None
    with pytest.raises(stop_type) as caught:
        check([question], None)
    assert caught.value.stage == "putter"
    assert fault in caught.value.reason
    # With a putter the run may start.
    assert check([question], lambda copy, fault_id: {}) is None


def test_a_row_with_no_fault_or_a_dolma_row_needs_no_putter():
    check = _fn("check_putters")
    plain = _make_question("c05.unmutated.R1", "originblame")
    assert check([plain], None) is None
    # The rows of Dolma are faulted by the runner on its own companion files.
    assert check([_make_question("c05.f.digest.E/shard-3", "dolma")], None) is None


def _build_with_records(folder: Path) -> Path:
    (folder / "sources").mkdir(parents=True)
    (folder / "gen" / "harness").mkdir(parents=True)
    (folder / "sources" / "A.jsonl").write_text('{"id": "a-1"}\n', encoding="utf-8")
    (folder / "sources" / "A.manifest.jsonl").write_text('{"id": "m-1"}\n', encoding="utf-8")
    (folder / "gen" / "T.jsonl").write_text('{"id": "t-1"}\n', encoding="utf-8")
    (folder / "gen" / "harness" / "log.jsonl").write_text('{"id": "h-1"}\n', encoding="utf-8")
    return folder


def test_every_request_file_is_a_record_file_of_the_build_its_store_registers_from(tmp_path):
    check = _fn("check_request_files")
    stop_type = _fn("PreRunStop")
    builds = {"main": _build_with_records(tmp_path / "main"), "rr": _build_with_records(tmp_path / "rr")}
    assert check({"withdraw A": ["sources/A.jsonl"], "withdraw C": ["gen/T.jsonl"]}, builds) is None
    # A file the build does not hold, named with the request and the file: stopped, never dropped.
    with pytest.raises(stop_type) as caught:
        check({"withdraw A": ["sources/A.jsonl", "sources/Z.jsonl"]}, builds)
    assert caught.value.stage == "request-files"
    assert "withdraw A" in caught.value.reason and "sources/Z.jsonl" in caught.value.reason
    # A file that is on disk but is no record file of the build (a manifest beside a record
    # file, a harness log) is not one the store registers from.
    for stranger in ("sources/A.manifest.jsonl", "gen/harness/log.jsonl"):
        with pytest.raises(stop_type) as caught:
            check({"withdraw B": [stranger]}, builds)
        assert caught.value.stage == "request-files"
        assert "withdraw B" in caught.value.reason and stranger in caught.value.reason
    # The build of one store lacking the file is enough to stop.
    (tmp_path / "rr" / "sources" / "A.jsonl").unlink()
    with pytest.raises(stop_type) as caught:
        check({"withdraw A": ["sources/A.jsonl"]}, builds)
    assert "rr" in caught.value.reason


# --- One pass --------------------------------------------------------------------------------------------


def _config(tmp_path, w, *, subjects, factories, questions, setups=None, cap=600.0, **changes):
    arguments = dict(
        commit=COMMIT, work=w.work, run_record=w.run_record, lock_prefix=LOCK, subjects=tuple(subjects),
        step_cap_seconds=cap, setups=setups if setups is not None else {}, factories=factories,
        questions=list(questions),
        grade_of=lambda question, reading: _grade("right"),
    )
    arguments.update(changes)
    return _fn("RunConfig")(**arguments)


def _setup(w):
    return _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha)


def _never(name):
    def factory(context):
        raise AssertionError(f"the subject {name} was built before its gate opened")

    return factory


def test_a_probe_that_differs_stops_the_pass_before_any_subject_is_built_and_writes_no_answer_line(tmp_path):
    w = _world(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    retargeted = hashlib.sha256(b"a retargeted interpreter").hexdigest()
    starter = Starter(probe=(w.probe_path, retargeted))
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": _never("dolma")},
                     questions=[question], setups={"dolma": _setup(w)})
    with pytest.raises(_fn("PreRunStop")) as caught:
        _fn("run_pass")(config, starter)
    assert caught.value.stage == "probe"
    assert len(starter.argvs) == 1 and _is_probe(starter.argvs[0])
    answers = w.run_record / "answers.jsonl"
    assert not answers.exists() or answers.read_text(encoding="utf-8") == ""


def test_every_pre_run_stop_leaves_a_stopped_json_that_names_the_stop_and_the_seal_lists_it(tmp_path):
    """17 Nov 2026: a stopped record is sealed AND names its
    stop, whichever pre-run stage stopped the pass (until now only the world-hash stop did)."""
    w = _world(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    retargeted = hashlib.sha256(b"a retargeted interpreter").hexdigest()
    starter = Starter(probe=(w.probe_path, retargeted))
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": _never("dolma")},
                     questions=[question], setups={"dolma": _setup(w)})
    with pytest.raises(_fn("PreRunStop")) as caught:
        _fn("run_pass")(config, starter)
    assert caught.value.stage == "probe"
    stopped = json.loads((w.run_record / "stopped.json").read_text(encoding="utf-8"))
    assert stopped["subject"] is None and stopped["question"] is None
    assert "PreRunStop" in stopped["error"] and "probe" in stopped["error"], stopped
    _fn("seal_run_record")(w.run_record)
    assert "  stopped.json" in (w.run_record / "sha256sums.txt").read_text(encoding="utf-8")
    # A stop that the pass never reached leaves no such file: a pass that ends by itself holds none.
    quiet = _world(tmp_path / "quiet")
    ended = _fn("run_pass")(_config(tmp_path, quiet, subjects=("dolma",), factories={"dolma": lambda c: Subject("dolma")},
                                    questions=[question], setups={"dolma": _setup(quiet)}),
                            Starter(probe=(quiet.probe_path, quiet.probe_sha)))
    assert ended is not None and not (quiet.run_record / "stopped.json").exists()


def test_a_pass_runs_the_gate_first_keeps_every_step_and_seals_the_record(tmp_path):
    w = _world(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    built = []

    def factory(context):
        built.append(context)
        return Subject("dolma")

    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": factory},
                     questions=[question], setups={"dolma": _setup(w)})
    result = _fn("run_pass")(config, starter)
    # The probe was the first thing started, and only after it the subject was built.
    assert _is_probe(starter.argvs[0]) and len(built) == 1
    assert [line["id"] for line in result.lines] == [question.id]
    assert result.lines[0]["status"] == "right" and result.lines[0]["commit"] == COMMIT
    read_answers = need_attr("baselines.record", "read_answers")
    assert read_answers(w.run_record / "answers.jsonl") == result.lines
    first_step = json.loads((w.run_record / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "-I" in first_step["command"] and first_step["exit_code"] == 0
    # The compare result and the probe record are files of the run record.
    assert (w.run_record / "dolma" / "compare.json").is_file()
    assert (w.run_record / "dolma" / "interpreter.txt").is_file()
    # Nothing of the run record is left out of the list of sha256 sums.
    sums = (w.run_record / "sha256sums.txt").read_text(encoding="utf-8")
    for name in ("answers.jsonl", "steps.jsonl", "dolma/compare.json", "dolma/interpreter.txt"):
        assert f"  {name}" in sums


def test_only_the_selected_subjects_pass_their_gate_and_are_built(tmp_path):
    w = _world(tmp_path)
    broken = _world(tmp_path / "second", name="originblame", compare="different")
    questions = [_make_question("c05.unmutated.E/shard-1", "dolma"), _make_question("c05.unmutated.E/shard-1", "originblame")]
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    config = _config(
        tmp_path, w, subjects=("dolma",),
        factories={"dolma": lambda context: Subject("dolma"), "originblame": _never("originblame")},
        questions=questions, setups={"dolma": _setup(w), "originblame": _setup(broken)},
    )
    result = _fn("run_pass")(config, starter)
    assert [line["subject"] for line in result.lines] == ["dolma"]
    assert len(starter.argvs) == 1


def test_a_pass_records_one_attempt_per_target_its_subject_ran_on_and_writes_the_none_lines_from_them(tmp_path):
    w = _world(tmp_path)
    ran = _make_question("c07.report.R1.X", "dolma")
    failed = _make_question("c04.declare-499.write", "dolma")
    idle = _make_question("c07.swap.report.counts", "dolma")
    for question in (ran, failed, idle):
        assert question.route is None, question.id
    on_ran, on_failed, on_idle = record_target(ran), record_target(failed), record_target(idle)
    assert len({on_ran, on_failed, on_idle}) == 3
    broken = _attempt("dolma-tokens-9", output="no companion row was written")
    made = []

    def factory(context):
        context.ran_on.add(on_ran)
        context.failed_runs[on_failed] = broken
        subject = Subject("dolma")
        made.append(subject)
        return subject

    starter = Starter(probe=(w.probe_path, w.probe_sha))
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": factory},
                     questions=[ran, failed, idle], setups={"dolma": _setup(w)})
    result = _fn("run_pass")(config, starter)
    # One attempt per target the subject ran on: the run that ran and the run that failed.
    seen = {(a.subject, tuple(a.target), a.attempt_id) for a in result.store_attempts}
    assert seen == {("dolma", on_ran, None), ("dolma", on_failed, "dolma-tokens-9")}
    by_id = {line["id"]: line for line in result.lines}
    # A store the tool ran on: cannot answer, with the table's own text of what is missing.
    assert by_id[ran.id]["status"] == "cannot answer" and by_id[ran.id]["evidence"] == ran.missing
    # A store whose run failed: the status of that attempt.
    assert by_id[failed.id]["status"] == "no answer: failed attempt dolma-tokens-9"
    assert by_id[failed.id]["attempt_id"] == "dolma-tokens-9"
    # A store the tool never ran on has no line, and nothing was read for any of these questions.
    assert idle.id not in by_id and set(by_id) == {ran.id, failed.id}
    assert made[0].calls == 0
    kept = need_attr("baselines.record", "read_attempts")(w.run_record / "attempts.jsonl")
    assert [line["id"] for line in kept] == ["dolma-tokens-9"]
    assert kept[0]["question_ids"] == [failed.id]


def test_a_pass_puts_the_none_rows_of_a_shardless_store_with_the_runs_the_tool_made_in_the_pass(tmp_path):
    """The none rows of a store with no chunk file and no shard (the store import): cannot answer with the row's own
    missing text and resting on the steps of the tool's completed runs; the failed attempt of the tool when it completed
    none; nothing when it did neither."""
    import dataclasses as dc
    on_import = ("store", "import")
    asked = _make_question("c06.import.file-family", "dolma")
    assert asked.route is None and record_target(asked) == on_import
    main = ("store", "main")

    def run_with(tmp, *, ran, failed):
        w = _world(tmp)
        broken = _attempt("dolma-tokens-9", output="no companion row was written")
        made = []

        def factory(context):
            with context.log.shared(main):
                context.runner.run(["the", "tool", "run"])
            if ran:
                context.ran_on.add(main)
            if failed:
                context.failed_runs[main] = broken
            subject = Subject("dolma")
            made.append(subject)
            return subject

        starter = Starter(probe=(w.probe_path, w.probe_sha))
        config = _config(tmp, w, subjects=("dolma",), factories={"dolma": factory}, questions=[asked],
                         setups={"dolma": _setup(w)})
        config = dc.replace(config, shardless=frozenset({on_import}))
        return _fn("run_pass")(config, starter), w, made

    result, w, made = run_with(tmp_path / "ran", ran=True, failed=False)
    (line,) = result.lines
    assert line["id"] == asked.id and line["status"] == "cannot answer" and line["evidence"] == asked.missing
    steps = [json.loads(l)["id"] for l in (w.run_record / "steps.jsonl").read_text().splitlines()]
    tool_step = next(step for step in steps if step not in ("step-1",) or len(steps) == 1)
    assert line["step_ids"] and set(line["step_ids"]) <= set(steps) and any(
        "tool" in json.loads(l).get("command", "") or "tool" in str(json.loads(l).get("command")) for l in
        (w.run_record / "steps.jsonl").read_text().splitlines() if json.loads(l)["id"] in line["step_ids"])
    assert made[0].calls == 0, "nothing was read for a none row"
    # The tool completed no run: the line is its failed attempt, which attempts.jsonl keeps with the question.
    result, w, made = run_with(tmp_path / "failed", ran=False, failed=True)
    (line,) = result.lines
    assert line["status"] == "no answer: failed attempt dolma-tokens-9" and line["attempt_id"] == "dolma-tokens-9"
    kept = need_attr("baselines.record", "read_attempts")(w.run_record / "attempts.jsonl")
    assert [k["id"] for k in kept] == ["dolma-tokens-9"] and kept[0]["question_ids"] == [asked.id]
    # No run and no failure: no line (not attempted).
    result, w, made = run_with(tmp_path / "none", ran=False, failed=False)
    assert result.lines == []


def test_a_pass_stops_before_any_start_when_a_faulted_row_has_no_putter_or_a_request_file_is_no_record_file(tmp_path):
    stop_type = _fn("PreRunStop")
    w = _world(tmp_path, name="originblame", program=True)
    faulted = _make_question("c05.f.digest.E/shard-3", "originblame")
    setups = {"originblame": _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha,
                                              ob_binary=w.program)}
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    config = _config(tmp_path, w, subjects=("originblame",), factories={"originblame": _never("originblame")},
                     questions=[faulted], setups=setups)
    # No putter for the fault the row names: stopped, and nothing was started, not even the probe.
    with pytest.raises(stop_type) as caught:
        _fn("run_pass")(dataclasses.replace(config, fault_putter=None), starter)
    assert caught.value.stage == "putter" and "f.digest" in caught.value.reason
    assert starter.argvs == []
    # A request file that is no record file of the build: stopped, nothing started.
    build = _build_with_records(tmp_path / "build-main")
    missing = dataclasses.replace(config, fault_putter=lambda copy, fault_id: {},
                                  requests={"withdraw A": ["sources/Z.jsonl"]}, builds={"main": build})
    with pytest.raises(stop_type) as caught:
        _fn("run_pass")(missing, starter)
    assert caught.value.stage == "request-files"
    assert "withdraw A" in caught.value.reason and "sources/Z.jsonl" in caught.value.reason
    assert starter.argvs == []


# --- The work path a tool's temporary folder lies under -----------------------------------------------------
#
# The first real pass (17 Nov 2026) ran the Dolma tokenizer with a work folder of 66 characters: its
# multiprocessing manager binds a unix socket at <work>/dolma-tmp-N/pymp-XXXXXXXX/listener-XXXXXXXX and
# stopped with "OSError: AF_UNIX path too long". Three steps were failed attempts that came from the length
# of a path. The runner refuses such a work path before any tool starts. The bound is shown here against
# the real mechanism: a real multiprocessing manager started with its temporary folder at the largest
# path the guard lets through, and at one character more.

_MANAGER_PROGRAM = (
    "import multiprocessing.managers as m\n"
    "manager = m.SyncManager()\n"
    "manager.start()\n"
    "manager.shutdown()\n"
)


@contextlib.contextmanager
def _short_base():
    """A scratch folder with a SHORT path (pytest's own folders are too long to build a 60 character
    work path under); it is removed when the test ends."""
    with tempfile.TemporaryDirectory(prefix="wp", dir="/tmp") as folder:
        yield Path(folder)


def _work_of_length(tmp_path: Path, length: int) -> Path:
    """A work folder (made) whose absolute path has exactly ``length`` characters."""
    base = tmp_path / "w"
    pad = length - len(str(base)) - 1
    assert pad >= 1, f"the test base path is too long for {length} characters"
    work = base / ("x" * pad)
    work.mkdir(parents=True)
    assert len(str(work)) == length
    return work


def _manager_outcome(work: Path) -> str:
    """Start a real multiprocessing manager whose temporary folder is the worst case the runner names:
    <work>/dolma-tmp-NNNN. Returns "started" or the text of the error."""
    tmp = work / "dolma-tmp-9999"
    tmp.mkdir()
    done = subprocess.run([sys.executable, "-c", _MANAGER_PROGRAM], env={**os.environ, "TMPDIR": str(tmp)},
                          capture_output=True, text=True, timeout=60)
    if done.returncode == 0:
        return "started"
    return done.stderr  # the whole traceback: the manager's server process names the OSError before the parent's EOFError


def test_the_work_path_the_guard_lets_through_is_the_longest_one_a_real_manager_can_bind_a_socket_under():
    check = _fn("check_work_path")
    stop_type = _fn("PreRunStop")
    allowed = _fn("WORK_PATH_LONGEST")
    assert isinstance(allowed, int)
    # At the longest length the guard names, the check passes and the real manager starts.
    with _short_base() as base:
        at_bound = _work_of_length(base / "at", allowed)
        check(at_bound)
        assert _manager_outcome(at_bound) == "started"
        # One character more: the guard refuses, naming the stage, and the real manager fails as step 12 of the first pass did.
        over = _work_of_length(base / "over", allowed + 1)
        with pytest.raises(stop_type) as caught:
            check(over)
        assert caught.value.stage == "work-path"
        assert str(allowed + 1) in caught.value.reason and str(allowed) in caught.value.reason


def test_a_real_manager_fails_one_character_over_the_work_path_bound():
    """The other half of the bound: with the work folder one character over, the real manager
    stops with "AF_UNIX path too long", as step 12 of the first pass did. That is a fact of the
    interpreter the manager runs under: Python 3.14 starts its manager's socket at a short
    path in /tmp, whatever TMPDIR is, so there the limit the guard names does not apply and the
    manager starts. The test is skipped, naming that, where this interpreter's manager starts."""
    allowed = _fn("WORK_PATH_LONGEST")
    with _short_base() as base:
        # Whether the limit applies to this interpreter is decided far over the bound (it fails
        # there on Python 3.12 and starts on 3.14); only where it applies is the bound itself held
        # to the manager, so a bound set too low is never hidden by the skip.
        far = _work_of_length(base / "far", allowed + 60)
        if _manager_outcome(far) == "started":
            pytest.skip(f"Python {sys.version.split()[0]}: its multiprocessing manager started with the "
                        f"work folder at {allowed + 60} characters, so the unix socket limit the guard "
                        "names does not apply to this interpreter")
        over = _work_of_length(base / "over", allowed + 1)
        assert "AF_UNIX path too long" in _manager_outcome(over)


def test_a_pass_of_the_dolma_tool_stops_before_any_start_when_its_work_path_is_too_long():
    stop_type = _fn("PreRunStop")
    allowed = _fn("WORK_PATH_LONGEST")
    assert "work-path" in stop_type.STAGES
    with _short_base() as base:
        w = _world(base / "p")
        question = _make_question("c05.unmutated.E/shard-1", "dolma")
        long_work = _work_of_length(base / "long", allowed + 1)
        config = _config(base, w, subjects=("dolma",), factories={"dolma": _never("dolma")},
                         questions=[question], setups={"dolma": _setup(w)}, work=long_work)
        # The real starter: had the guard let the pass through, its first command would have been started.
        with pytest.raises(stop_type) as caught:
            _fn("run_pass")(config, _fn("SubprocessStarter")())
        assert caught.value.stage == "work-path"
        assert not (w.run_record / "steps.jsonl").exists() or (w.run_record / "steps.jsonl").read_text() == ""
        # A stand-in starter, which never binds the manager's socket, is not held by the path.
        starter = Starter(probe=(w.probe_path, w.probe_sha))
        passing = dataclasses.replace(config, factories={"dolma": lambda context: Subject("dolma")})
        assert _fn("run_pass")(passing, starter) is not None


def test_a_pass_of_originblame_alone_is_not_held_by_the_length_of_its_work_path():
    """The socket belongs to the Dolma tokenizer's manager: a pass that never runs it is not refused for the path."""
    allowed = _fn("WORK_PATH_LONGEST")
    of_tools = _fn("check_work_path_of_tools")
    with _short_base() as base:
        long_work = _work_of_length(base / "long", allowed + 20)
        assert of_tools(("originblame",), long_work) is None
        with pytest.raises(_fn("PreRunStop")):
            of_tools(("dolma",), long_work)
        with pytest.raises(_fn("PreRunStop")):
            of_tools(("dolma", "originblame"), long_work)


# --- The size of a token file, for the second run of the tokenizer ---------------------------------------------------


def test_the_token_file_size_of_a_second_run_is_the_bytes_of_its_npy_files_over_the_dtype_the_run_printed(tmp_path):
    """The first real pass read the second run of the Dolma writer as a gap: the driver never gave the reader the
    number of tokens of the token file. It is the size of the destination's .npy files over the item size of the
    dtype the run itself printed in its configuration (``dtype: uint16``): no dtype is typed in the runner."""
    sizer = _fn("token_file_tokens_of")
    folder = tmp_path / "tokens"
    folder.mkdir()
    (folder / "part-0-00000.npy").write_bytes(b"\0" * 41220)
    (folder / "part-0-00001.npy").write_bytes(b"\0" * 10)
    (folder / "part-0-00000.csv.gz").write_bytes(b"\0" * 999)
    printed = "batch_size: 10000\ndebug: false\ndtype: uint16\nfields:\n  id_field_name: id\n"
    assert sizer(folder, printed) == (41220 + 10) // 2
    assert sizer(folder, printed.replace("uint16", "uint32")) == (41220 + 10) // 4
    # No dtype printed, or no token file: no number is made up.
    assert sizer(folder, "debug: false\n") is None
    assert sizer(tmp_path / "nothing", printed) is None


# --- The files of a member, from the world's own request entries ------------------------------------------------------


def test_the_files_of_a_member_come_from_the_requests_of_the_world_through_the_source_the_spec_gives_the_member():
    """A member of the table (``A/r1``) is the member of one source in the fixture spec; the world's request entries
    name the files of the sources their words withdraw. The grader of the import's file family compares the member
    through this file, derived by code from the world's own entry (no path shape is typed)."""
    of_world = _fn("member_files_of")
    world = {"requests": {"withdraw A": {"files": ["sources/A.jsonl"]},
                          "withdraw A, record X@A": {"files": ["sources/A.jsonl"], "record": "A-001"},
                          "withdraw C": {"files": ["sources/C.jsonl"]},
                          "withdraw W": {"files": ["sources/Z.jsonl"], "record": "Z-089"}}}
    # The request "withdraw W" names no source of its own (the spec's position gives it), so it makes no member's file.
    assert of_world(world) == {"A/r1": ["sources/A.jsonl"], "C/r1": ["sources/C.jsonl"]}
    assert of_world({"requests": {}}) == {}


# --- The generated sets made without capture: their own counting document and index -----------------------------


def test_a_question_of_the_set_made_without_capture_is_graded_against_the_counting_document_and_index_of_that_build(monkeypatch, tmp_path):
    """The rows of Check 6 that name a generated set and the store no-capture ask the build made without capture
    (the target ``no-capture-<size>``); its chunk files differ from the capture set's (unseeded shuffles), so its
    expected values come from ITS counting document and index (world keys ``counted_no_capture`` and
    ``index_no_capture``), never from the capture set's of the same size."""
    runner = importlib.import_module("baselines.runner")
    seen = []

    def expected_for(listed_row, size, *, counted, index):
        seen.append((listed_row, size, counted, index))
        raise RuntimeError("stop after the choice")

    monkeypatch.setattr(runner.expected, "expected_for", expected_for)
    monkeypatch.setattr(runner.expected, "index_of_shards", lambda files: {"index": list(files)})
    def write(name, body):
        path = tmp_path / name
        path.write_text(json.dumps(body), encoding="utf-8")
        return str(path)

    world = {"counted": {"small": write("capture-counted.json", {"build": "capture"})},
             "counted_no_capture": {"small": write("nc-counted.json", {"build": "no-capture"})},
             "index": {"small": ["capture.idx"]}, "index_no_capture": {"small": ["no-capture.idx"]}}
    grade_of, _ = runner._graders(world)
    generated = _make_question("c06.generated.candidates", "dolma", "small")
    capture_row = _make_question("c03.spans.per-ordinal", "dolma", "small")
    for question in (generated, capture_row):
        with pytest.raises(RuntimeError):
            grade_of(question, SimpleNamespace(question_id=question.id, raw={}, route=question.route))
    (g_row, g_size, g_counted, g_index), (c_row, c_size, c_counted, c_index) = seen
    assert g_counted == {"small": {"build": "no-capture"}} and g_index == {"small": {"index": ["no-capture.idx"]}}
    assert c_counted == {"small": {"build": "capture"}} and c_index == {"small": {"index": ["capture.idx"]}}


def test_a_sized_question_that_names_round_robin_is_graded_against_the_counting_document_of_its_own_size(monkeypatch, tmp_path):
    """Found by the first all-size pass on the cut tree (pass 15, 18 Nov 2026, not of record): the world holds
    ``counted_by_layout`` for the 10k set (built in two layouts), and the grader put that document in the place of the document of
    ``small`` and of ``100k`` for every row that names a layout, so 16 right readings of dolma and dolma+trace (c03.pairs.split and
    three more rows, at small and at 100k) were graded wrong. The document by layout stands only for the set built in two layouts,
    which is the 10k set; the 10k rows keep it."""
    runner = importlib.import_module("baselines.runner")
    seen = []

    def expected_for(listed_row, size, *, counted, index):
        seen.append((listed_row, size, counted))
        raise RuntimeError("stop after the choice")

    monkeypatch.setattr(runner.expected, "expected_for", expected_for)
    monkeypatch.setattr(runner.expected, "index_of_shards", lambda files: {"index": list(files)})

    def write(name, body):
        path = tmp_path / name
        path.write_text(json.dumps(body), encoding="utf-8")
        return str(path)

    # The 10k set's own document stands under both keys; a run record keeps the two as two files with the same bytes.
    round_robin = write("rr.json", {"set": "10k-round-robin"})
    world = {"requests": {}, "index": {},
             "counted": {"small": write("small.json", {"set": "small"}), "10k": write("10k.json", {"set": "10k-round-robin"}),
                         "100k": write("100k.json", {"set": "100k"})},
             "counted_by_layout": {"round-robin": round_robin, "partitioned": write("pt.json", {"set": "10k-partitioned"})},
             "shards": [{"name": "shard-01", "target": ["set", "10k"], "layout": "round-robin", "build": "/w/set-10k"},
                        {"name": "shard-A", "target": ["set", "10k"], "layout": "partitioned", "build": "/w/set-10k-partitioned"},
                        {"name": "shard-01", "target": ["set", "small"], "layout": "round-robin", "build": "/w/set-small"},
                        {"name": "shard-01", "target": ["set", "100k"], "layout": "round-robin", "build": "/w/set-100k"}]}
    grade_of, _ = runner._graders(world)
    for size in ("small", "100k"):
        question = _make_question("c03.pairs.split", "dolma", size)
        assert record_layout(question) == "round-robin", "the row must name the round-robin layout for this check to reach the line"
        with pytest.raises(RuntimeError):
            grade_of(question, SimpleNamespace(question_id=question.id, raw={}, route=question.route))
    (_, small_size, small_counted), (_, hundred_size, hundred_counted) = seen
    assert small_counted["small"] == {"set": "small"}, small_counted
    assert hundred_counted["100k"] == {"set": "100k"}, hundred_counted
    # The 10k rows of a layout keep the document of that layout (the existing test of W1 holds them).


def record_layout(question):
    return importlib.import_module("baselines.record").layout_of(question)


# --- A none row on a store with no chunk file and no shard ----------------------------------------------------------
#
# A none line is written after the attempt of its subject on the store or set its listed row names. For a store the world names
# for the manifest or the trace and that holds NO chunk file and NO shard of either tool (the table's stores `import` and
# `swap`), the none rows of the plain tools and of the combinations read "cannot answer" with the row's own missing text, citing
# the tool's attempts of the pass (the tool ran; its records hold nothing for that store); where the tool completed no run in the
# pass, the line is that failed attempt.


def _world_of_targets(dolma=(), shards=(), manifests=(), traces=()):
    return ({"stores": [{"target": list(t)} for t in dolma], "shards": [{"target": list(t)} for t in shards]},
            {"manifests": {key: [] for key in manifests}, "traces": {key: {} for key in traces}})


def test_a_store_the_world_names_for_the_manifest_and_the_trace_and_that_holds_no_chunk_file_or_shard_is_shardless():
    covered_targets = _fn("covered_targets")
    world, inputs = _world_of_targets(
        dolma=[("store", "main")], shards=[("store", "main")],
        manifests=["store-main", "store-import", "store-swap"], traces=["store-main", "store-import", "store-swap"])
    covered, shardless = covered_targets(world, inputs)
    assert shardless == frozenset({("store", "import"), ("store", "swap")})
    # What each subject is asked on is as before: the shardless stores are not a target of a tool's own run.
    assert covered["dolma"] == covered["dolma+trace"] == {("store", "main")}
    assert covered["originblame"] == covered["originblame+trace"] == {("store", "main")}
    assert covered["manifest"] == {("store", "main"), ("store", "import"), ("store", "swap")}
    # The plain tools' lines must not depend on which subjects are driven: with no subject inputs loaded (a pass of the two
    # plain tools) the shardless stores are read from the keys of the world's own subject_inputs, the same stores.
    named = dict(world, subject_inputs={"manifest": {key: {} for key in inputs["manifests"]},
                                        "trace": {key: {} for key in inputs["traces"]}})
    assert covered_targets(named, None)[1] == shardless
    assert covered_targets(world, None)[1] == frozenset(), "a world that names no subject inputs has no shardless store"
    # A store that has a Dolma entry or a shard is not shardless, and a pass with no such store has none.
    world2, inputs2 = _world_of_targets(dolma=[("store", "main"), ("store", "no-capture")],
                                        manifests=["store-main", "store-no-capture"], traces=["store-main"])
    assert covered_targets(world2, inputs2)[1] == frozenset()


def test_the_none_rows_of_a_shardless_store_cite_the_attempts_of_the_tool_in_the_pass():
    attempts_of = _fn("_store_attempts_of")
    attempt = _attempt()
    shardless = frozenset({("store", "import"), ("store", "swap")})
    none_questions = [_make_question("c06.import.file-family", "dolma"), _make_question("c07.swap.report.counts", "dolma")]
    assert all(q.route is None for q in none_questions)
    # The tool completed a run in the pass: each shardless store gets the attempt that ran.
    ran = SimpleNamespace(ran_on={("store", "main")}, failed_runs={}, second_runs_on=set(), failed_second_runs={}, layout_runs={})
    got = attempts_of("dolma", ran, none_questions, shardless)
    held = {a.target: a for a in got if a.target in shardless}
    assert set(held) == set(shardless) and all(a.attempt_id is None and a.subject == "dolma" for a in held.values())
    assert "no chunk file and no shard" in held[("store", "import")].recorded
    # The tool completed no run: its failed attempt is cited, so the lines are failed attempts (never cannot answer).
    failed = SimpleNamespace(ran_on=set(), failed_runs={("store", "main"): attempt}, second_runs_on=set(),
                             failed_second_runs={}, layout_runs={})
    got = attempts_of("dolma", failed, none_questions, shardless)
    held = {a.target: a for a in got if a.target in shardless}
    assert set(held) == set(shardless) and all(a.attempt_id == attempt.id for a in held.values())
    # No run and no failure at all: nothing is cited, so the lines stay not attempted.
    nothing = SimpleNamespace(ran_on=set(), failed_runs={}, second_runs_on=set(), failed_second_runs={}, layout_runs={})
    assert [a for a in attempts_of("dolma", nothing, none_questions, shardless) if a.target in shardless] == []
    # A subject that has its own run on the shardless store (the manifest and the trace read the store's own files) keeps that
    # attempt alone: no second attempt is made for it (two attempts of one subject on one store cannot stand).
    own = SimpleNamespace(ran_on={("store", "main"), ("store", "import")}, failed_runs={}, second_runs_on=set(),
                          failed_second_runs={}, layout_runs={})
    got = attempts_of("manifest", own, [_make_question("c06.import.file-family", "manifest")], shardless)
    assert [(a.target, a.run) for a in got if a.target == ("store", "import")] == [(("store", "import"), 1)]
    # A pass without shardless stores makes the attempts it made before.
    assert attempts_of("dolma", ran, none_questions) == attempts_of("dolma", ran, none_questions, frozenset())


def test_the_request_files_are_checked_in_each_build_of_a_target_built_in_two_layouts(tmp_path):
    """W8 (18 Nov 2026): the builds the request files are checked in are keyed by store or set; the 10k set built in two layouts
    holds two builds, and the shards of the second must not overwrite the first under ``set-10k``. A request file missing from
    ONE of the two builds stops the pass, whichever build it is, for the question that reads that build."""
    runner, check = _fn("builds_of_world"), _fn("check_request_files")
    stop_type = _fn("PreRunStop")
    own = _build_with_records(tmp_path / "own")
    added = _build_with_records(tmp_path / "added")
    world = {"stores": [{"target": ["set", "10k"], "build": str(own), "chunk_files": {}},
                        {"target": ["set", "10k"], "layout": "partitioned", "build": str(added), "chunk_files": {}}],
             "shards": [{"name": "shard-01", "target": ["set", "10k"], "layout": "round-robin", "build": str(own)},
                        {"name": "shard-A", "target": ["set", "10k"], "layout": "partitioned", "build": str(added)}]}
    builds = runner(world)
    assert builds == {"set-10k": str(own), "set-10k-partitioned": str(added)}, \
        "the build added in a layout took the key of the target's own build, or the target's own build was lost"
    questions = [_make_question("c09.10k.partitioned.kept-shards", "originblame"),
                 _make_question("c09.10k.round-robin.all-rewrite", "originblame")]
    check({"withdraw A": ["sources/A.jsonl"]}, builds, questions)
    (added / "sources" / "A.jsonl").unlink()
    with pytest.raises(stop_type) as caught:
        check({"withdraw A": ["sources/A.jsonl"]}, builds, questions)
    assert "set-10k-partitioned" in caught.value.reason
    # The question of the other layout reads the target's own build: the file missing from the added build is no stop for it.
    check({"withdraw A": ["sources/A.jsonl"]}, builds, questions[1:])
    (added / "sources").joinpath("A.jsonl").write_text('{"id": "a-1"}\n', encoding="utf-8")
    (own / "sources" / "A.jsonl").unlink()
    with pytest.raises(stop_type) as caught:
        check({"withdraw A": ["sources/A.jsonl"]}, builds, questions[1:])
    assert "set-10k" in caught.value.reason and "partitioned" not in caught.value.reason


def test_the_attempt_of_a_layout_is_that_of_the_build_in_that_layout_when_the_target_holds_a_build_for_each():
    """W8 (18 Nov 2026): the 10k set is built in two layouts. A none row of two layouts (c09.10k.multiset) needs an attempt per
    layout, and the attempt of the partitioned layout is the one of the build ADDED beside the round-robin build (``layout_runs``),
    not the target's own run: where that build failed, the line is its failed attempt, and no pair of attempts collides."""
    attempts_of, record = _fn("_store_attempts_of"), need("baselines.record")
    question = _make_question("c09.10k.multiset", "dolma")
    assert question.route is None and record.layouts_of(question) == ("partitioned", "round-robin")
    target = ("set", "10k")
    bad = _attempt("attempt-partitioned")
    both_ran = SimpleNamespace(ran_on={target}, failed_runs={}, second_runs_on=set(), failed_second_runs={},
                               layout_runs={(*target, "partitioned"): None})
    got = attempts_of("dolma", both_ran, [question])
    by_layout = {a.layout: a for a in got if a.target == target}
    assert set(by_layout) == {None, "partitioned", "round-robin"} and all(a.attempt_id is None for a in by_layout.values())
    assert record.none_line_status(question, got) == need("baselines.interfaces").CANNOT_ANSWER
    added_failed = SimpleNamespace(ran_on={target}, failed_runs={}, second_runs_on=set(), failed_second_runs={},
                                   layout_runs={(*target, "partitioned"): bad})
    got = attempts_of("dolma", added_failed, [question])
    by_layout = {a.layout: a for a in got if a.target == target}
    assert by_layout["partitioned"].attempt_id == bad.id and by_layout["round-robin"].attempt_id is None
    assert record.none_line_status(question, got) == need("baselines.interfaces").no_answer(bad.id), \
        "a failed build in one layout is the failed attempt of the row of two layouts"


def test_a_line_rests_on_the_steps_of_the_build_in_the_layout_its_question_names(tmp_path):
    """W8: the steps of the build ADDED beside the target's own are logged under (kind, name, layout); a question that names a
    layout rests on that build's steps where the target holds one, a row of two layouts on both, a question that names none on
    the target's own."""
    log = _log(tmp_path, "layout-log")
    log.owner = "dolma"
    with log.shared(("set", "10k")):
        first = log.add(["tool", "round-robin"], 0, "ok", 0.1).id
    with log.shared(("set", "10k", "partitioned")):
        second = log.add(["tool", "partitioned"], 0, "ok", 0.1).id
    only_partitioned = _make_question("c09.10k.partitioned.kept-shards", "dolma")
    only_round_robin = _make_question("c09.10k.round-robin.all-rewrite", "dolma")
    both = _make_question("c09.10k.multiset", "dolma")
    assert log.step_ids(only_partitioned) == (second,)
    assert log.step_ids(only_round_robin) == (first,), "a layout the world holds no extra build for rests on the target's own steps"
    assert sorted(log.step_ids(both)) == sorted([first, second])


def test_the_dolma_trace_driver_reads_each_layout_of_a_target_from_the_run_of_that_layout():
    """W8: the plain tool's runs of one target (its own build under the two-word key, a build ADDED beside it under (kind, name,
    layout)) are read by ONE subject, which holds the run of each layout for a row of two layouts."""
    subjects, dolma = need("baselines.subjects"), need("baselines.dolma")
    target = ("set", "10k")
    own = dolma.DolmaRun(build=Path("/w/set-10k"), rows=[], chunk_files={}, layouts=("round-robin",))
    added = dolma.DolmaRun(build=Path("/w/set-10k-partitioned"), rows=[], chunk_files={}, layouts=("partitioned",))
    plain = SimpleNamespace(_runs={target: own, (*target, "partitioned"): added}, _tie_path="tie.json", _requests={}, _split_pairs=())
    context = SimpleNamespace(ran_on={target}, failed_runs={}, keep_fact=lambda *args, **kwargs: None)
    driver = subjects.DolmaTraceDriver(plain, traces={}, state_traces={}, place_of=lambda row: None, context=context)
    subject = driver._by_target[target]
    assert subject._layout_runs == {(target, "round-robin"): own, (target, "partitioned"): added}
    assert set(subject._runs) == {target, (*target, "partitioned")}


def test_the_words_of_a_line_the_grader_could_not_grade_are_the_ones_the_summary_keeps_out_of_cannot_answer(monkeypatch, tmp_path):
    """The runner keeps such a line as cannot answer with these words as evidence; the summary reads them back and counts
    the line under its own word. One list of the words, so that a change of wording cannot turn the line back into the
    subject's cannot answer."""
    runner = importlib.import_module("baselines.runner")
    summary = importlib.import_module("baselines.summary")
    assert runner.NOT_DERIVED_WORDS in summary.NOT_DERIVED_EVIDENCE
    grade_of, _ = runner._graders({"requests": {}})
    q = _make_question("c06.import.file-family", "manifest")
    got = grade_of(q, SimpleNamespace(question_id=q.id, raw={"link": {"files": ["sources/A.jsonl"]}}, route=q.route))
    assert got.status == "cannot answer" and got.evidence.startswith(summary.NOT_DERIVED_EVIDENCE), got.evidence
    # And the summary does not count it as cannot answer.
    line = {"id": q.id, "size": q.size, "status": got.status, "evidence": got.evidence}
    assert summary.summarize([q], [line]).count("manifest", "once", "cannot answer") == 0


# --- The whole-run diff rows: the oracle's log of the run -----------------------------------------------------------


def test_the_whole_run_diff_rows_are_graded_against_the_oracles_log_of_the_world_and_never_from_the_subjects_diff(monkeypatch, tmp_path):
    """The rows c01.E.diff, c01.E-rr.diff and c01.K.diff name "the oracle's log of the run" as their source: the world keeps
    that log (key ``oracle_logs``, one file of lines for each run, hashed), and the runner grades the three rows by
    ``grade_whole_run`` with it. Where the world holds no log the line is "expected value not derived", which the summary
    keeps out of cannot answer."""
    runner = importlib.import_module("baselines.runner")
    summary = importlib.import_module("baselines.summary")
    grade = importlib.import_module("baselines.grade")
    interfaces = importlib.import_module("baselines.interfaces")
    lines = {"E": [{"segment": "E/seg-1", "step": 1}, {"segment": "E/seg-2", "step": 7}], "K": [{"segment": "K/seg-1", "step": 1}]}
    files = {}
    for run, rows in lines.items():
        path = tmp_path / f"{run}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        files[run] = str(path)
    seen = []

    def whole_run(question, outcome, oracle):
        seen.append((question.listed_row, [dict(line) for line in oracle]))
        return interfaces.Grade(status="right", evidence=None)

    monkeypatch.setattr(grade, "grade_whole_run", whole_run)
    grade_of, _ = runner._graders({"requests": {}, "oracle_logs": files})
    for listed in ("c01.E.diff", "c01.K.diff"):
        q = _make_question(listed, "trace")
        assert q.route is not None
        got = grade_of(q, SimpleNamespace(question_id=q.id, raw={"rows": []}, route=q.route))
        assert got.status == "right", listed
    assert [row for row, _ in seen] == ["c01.E.diff", "c01.K.diff"]
    assert all(log == lines["E"] + lines["K"] for _, log in seen), "the log handed in is the world's, every run"
    # No log in the world: not derived, never the subject's cannot answer.
    bare, _ = runner._graders({"requests": {}})
    q = _make_question("c01.E.diff", "trace")
    got = bare(q, SimpleNamespace(question_id=q.id, raw={"rows": []}, route=q.route))
    assert got.status == "cannot answer" and got.evidence.startswith(summary.NOT_DERIVED_EVIDENCE), got


# --- The expected-value sources of a record -----------------------------------------------------------------------------


def test_the_run_record_keeps_the_counting_documents_the_indexes_and_the_oracle_logs_the_grader_read(tmp_path):
    """A world built without capture cannot be built again (its shuffles are unseeded), so the expected values of its rows
    exist only in the world that was built. The record keeps what the grader read (the counting documents, the shard indexes,
    the oracle's logs), sealed beside the answers, so that a re-grade can read them after the machine and its world are gone."""
    runner = importlib.import_module("baselines.runner")
    world_folder = tmp_path / "world"
    world_folder.mkdir()
    files = {}
    for name, body in (("counted-small.json", b"{}"), ("counted-nc.json", b'{"nc": 1}'), ("small-1.idx", b"idx-1"),
                       ("nc-1.idx", b"idx-2"), ("E.jsonl", b'{"segment": "E/seg-1"}\n')):
        (world_folder / name).write_bytes(body)
        files[name] = str(world_folder / name)
    world = {"counted": {"small": files["counted-small.json"]}, "counted_no_capture": {"small": files["counted-nc.json"]},
             "index": {"small": {"shard-1": files["small-1.idx"]}}, "index_no_capture": {"small": {"shard-1": files["nc-1.idx"]}},
             "oracle_logs": {"E": files["E.jsonl"]}}
    record = tmp_path / "record"
    record.mkdir()
    runner.keep_expected_sources(world, record)
    kept = json.loads((record / "expected-sources" / "index.json").read_text(encoding="utf-8"))
    assert sorted(kept) == ["counted", "counted_no_capture", "index", "index_no_capture", "oracle_logs"]
    sources = {Path(path).read_bytes() for path in files.values()}
    copies = []
    for key, entry in kept.items():
        assert entry, key
        for relative, digest in entry.items():
            data = (record / "expected-sources" / relative).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, (key, relative)
            copies.append(data)
    assert set(copies) == sources and len(copies) == len(files), "a source of the grader was not kept, or one was kept twice"
    assert "/" not in "".join(str(v) for e in kept.values() for v in e.values()), "a local path is in the index"
    text = (record / "expected-sources" / "index.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "/home/" not in text
    seal = runner.seal_run_record(record)
    listed = seal.read_text(encoding="utf-8")
    assert "expected-sources/index.json" in listed and "expected-sources/counted_no_capture/small.json" in listed
    # The reader of the copies: the world keys a grader needs, pointing into the record, in the shape the world has them; a
    # grader built from it reads the same files (a re-grade after the machine and its world are gone).
    back = runner.world_from_record(record)
    assert sorted(back) == sorted(world)
    for key, value in world.items():
        for name, inner in value.items():
            got = back[key][name]
            if isinstance(inner, dict):
                assert sorted(got) == sorted(inner), (key, name)
                for shard, source in inner.items():
                    assert Path(got[shard]).read_bytes() == Path(source).read_bytes() and record in Path(got[shard]).parents
            else:
                assert Path(got).read_bytes() == Path(inner).read_bytes() and record in Path(got).parents, (key, name)
    # A copy that no longer matches its sha256 in the index is refused, never read.
    (record / "expected-sources" / "counted" / "small.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="counted/small.json"):
        runner.world_from_record(record)


def test_graders_built_from_the_record_grade_as_graders_built_from_the_world(monkeypatch, tmp_path):
    """A re-grade builds its graders from ``world_from_record``: for a file-family question (it needs the world's requests) and
    for a question of a set made without capture (it needs that build's counting document and index) the record gives the
    same grade, and the same documents, as the world."""
    runner = importlib.import_module("baselines.runner")
    interfaces = importlib.import_module("baselines.interfaces")
    world_folder = tmp_path / "world"
    world_folder.mkdir()
    counted = world_folder / "counted-nc.json"
    counted.write_text(json.dumps({"build": "nc"}), encoding="utf-8")
    index = world_folder / "nc-1.idx"
    index.write_bytes(b"idx")
    world = {"requests": {"withdraw A": {"files": ["sources/A.jsonl"]}},
             "counted_no_capture": {"small": str(counted)}, "index_no_capture": {"small": {"shard-1": str(index)}}}
    record = tmp_path / "record"
    record.mkdir()
    runner.keep_expected_sources(world, record)
    back = runner.world_from_record(record)
    assert back["requests"] == world["requests"]
    seen = []

    def expected_for(listed_row, size, *, counted, index):
        seen.append((counted, index))
        raise RuntimeError("stop after the choice")

    family = _make_question("c06.import.file-family", "manifest")
    generated = _make_question("c06.generated.candidates", "dolma", "small")
    grades, documents = [], []
    for source in (world, back):
        grade_of, _ = runner._graders(source)
        grades.append(grade_of(family, _reading(family, {"link": {"files": ["sources/A.jsonl"]}})))
        with monkeypatch.context() as patched:
            patched.setattr(runner.expected, "expected_for", expected_for)
            patched.setattr(runner.expected, "index_of_shards",
                            lambda files: {name: Path(path).read_bytes() for name, path in files.items()})
            grade_of, _ = runner._graders(source)
            with pytest.raises(RuntimeError):
                grade_of(generated, SimpleNamespace(question_id=generated.id, raw={}, route=generated.route))
        documents.append(seen[-1])
    assert grades[0] == grades[1] and grades[0].status == "right", grades
    assert documents[0] == documents[1] == ({"small": {"build": "nc"}}, {"small": {"shard-1": b"idx"}}), documents


# --- The steps the runner keeps ------------------------------------------------------------------------------


def test_a_step_keeps_its_command_exit_output_and_seconds_with_no_local_path(tmp_path):
    work = tmp_path / "work"
    log = _log(tmp_path, roots={str(work): "<work>"})
    step = log.add(("tool", str(work / "build" / "x.jsonl")), 0, f"wrote {work}/out", 1.5)
    assert step.id == "step-1"
    assert step.command == ("tool", "<work>/build/x.jsonl")
    assert step.output == "wrote <work>/out"
    assert step.exit_code == 0 and step.seconds == 1.5
    second = log.add(("tool2",), None, "", 3.0)
    assert second.id == "step-2" and second.exit_code is None
    assert [s.id for s in log.steps] == ["step-1", "step-2"]
    lines = (tmp_path / "record" / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert [p["id"] for p in parsed] == ["step-1", "step-2"]
    for key in ("command", "exit_code", "output", "seconds", "kept"):
        assert key in parsed[0]
    assert parsed[0]["command"] == ["tool", "<work>/build/x.jsonl"]
    assert str(tmp_path) not in "\n".join(lines)
    # A step that holds a local path, a host or a session name nothing placed is not kept.
    with pytest.raises(ValueError):
        log.add(("tool3",), 0, "failed in " + PRIVATE_PATH, 1.0)
    assert [s.id for s in log.steps] == ["step-1", "step-2"]
    assert PRIVATE_PATH not in (tmp_path / "record" / "steps.jsonl").read_text(encoding="utf-8")


def test_a_file_a_reading_parsed_is_kept_with_the_step_it_belongs_to(tmp_path):
    entries = tmp_path / "token-index-entry"
    entries.write_bytes(b'{"sources": ["h"], "token_count": 12}\n')
    log = _log(tmp_path)
    step = log.add(("ob", "clean"), 0, "done", 2.0, kept={"entries.jsonl": entries})
    assert step.kept == ("kept/step-1/entries.jsonl",)
    assert (tmp_path / "record" / "kept" / "step-1" / "entries.jsonl").read_bytes() == entries.read_bytes()
    parsed = json.loads((tmp_path / "record" / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert parsed["kept"] == ["kept/step-1/entries.jsonl"]
    assert entries.read_bytes() == b'{"sources": ["h"], "token_count": 12}\n'


def test_a_reading_that_rests_on_several_steps_names_every_one_of_them(tmp_path):
    rec, log, starter = _recorder(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    subject = Subject("dolma", steps=[["tokens", "--run"], ["mix", "--run"], ["status"]], recorder=rec)
    answered = _fn("answer_question")(question, subject, commit=COMMIT, log=log,
                                      grade_of=lambda q, r: _grade("right"))
    assert answered.line["step_ids"] == ["step-1", "step-2", "step-3"]
    assert tuple(answered.step_ids) == ("step-1", "step-2", "step-3")
    assert [argv[0] for argv in starter.argvs] == ["tokens", "mix", "status"]
    # The steps of a failed attempt are named by its line too.
    failing = Subject("dolma", steps=[["tokens", "--run"], ["mix", "--run"]], recorder=rec,
                      outcome=_attempt("dolma-mix-1"))
    failed = _fn("answer_question")(question, failing, commit=COMMIT, log=log,
                                    grade_of=lambda q, r: _grade("right"))
    assert failed.line["status"] == "no answer: failed attempt dolma-mix-1"
    assert failed.line["step_ids"] == ["step-4", "step-5"]
    assert failed.attempt.id == "dolma-mix-1"


def test_one_step_that_many_questions_share_is_named_by_each_of_them(tmp_path):
    rec, log, _starter = _recorder(tmp_path)
    on_main = [_make_question("c05.unmutated.E/shard-1", "originblame"),
               _make_question("c05.unmutated.E/shard-2", "originblame")]
    on_rr = _make_question("c05.unmutated.E-rr/shard-1", "originblame")
    assert all(q.route is not None for q in (*on_main, on_rr))
    assert record_target(on_main[0]) == MAIN and record_target(on_main[1]) == MAIN
    assert record_target(on_rr) == ("store", "rr")
    # The one track of the store main is made once, before any question is read.
    with log.shared(MAIN):
        rec.run(["track", "--store", "main"])
    answered = []
    for question in (*on_main, on_rr):
        subject = Subject("originblame", steps=[["revoke", question.id]], recorder=rec)
        answered.append(_fn("answer_question")(question, subject, commit=COMMIT, log=log,
                                               grade_of=lambda q, r: _grade("right")))
    # Both questions of the store name the track, then their own revoke; the other store's
    # question names only its own step.
    assert answered[0].line["step_ids"] == ["step-1", "step-2"]
    assert answered[1].line["step_ids"] == ["step-1", "step-3"]
    assert answered[2].line["step_ids"] == ["step-4"]
    assert len(log.steps) == 4


def record_target(question):
    return tuple(need_attr("baselines.record", "attempt_target")(question))


def test_the_steps_file_and_the_kept_files_are_sealed_with_the_sha256_of_each_file(tmp_path):
    folder = tmp_path / "record"
    log = _fn("StepLog")(folder)
    source = tmp_path / "parsed.csv"
    source.write_bytes(b"0,10,r-1,E/c.jsonl,1\n")
    log.add(("dolma", "tokens"), 0, "ok", 1.0, kept={"rows.csv": source})
    (folder / "answers.jsonl").write_text('{"id": "q"}\n', encoding="utf-8")
    seal = _fn("seal_run_record")(folder)
    assert seal.name == "sha256sums.txt" and seal.parent == folder
    listed = {}
    for line in seal.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    files = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p != seal}
    assert set(listed) == files
    assert {"steps.jsonl", "answers.jsonl", "kept/step-1/rows.csv"} <= files
    for name, digest in listed.items():
        assert digest == hashlib.sha256((folder / name).read_bytes()).hexdigest(), name


def test_a_step_past_the_cap_is_killed_and_is_a_failed_attempt_carrying_its_seconds(tmp_path):
    over = _result(None, "partial output", 612.0)
    rec, log, starter = _recorder(tmp_path, starter=Starter(results=[over, _result(0, "fine", 3.0)]), cap=600.0)
    got = rec.run(["tool", "tokens"])
    assert starter.timeouts == [600.0]
    assert got.exit_code is None
    assert "612" in got.output and "cap" in got.output
    step = log.steps[0]
    assert step.exit_code is None and step.seconds == 612.0 and step.capped is True
    attempt = _fn("attempt_of")(step, "dolma-tokens-1")
    assert attempt.id == "dolma-tokens-1"
    assert attempt.command == step.command and attempt.exit_code is None
    assert "612" in attempt.output
    # A step inside the cap is no attempt of that kind.
    ok = rec.run(["tool", "mix"])
    assert ok.exit_code == 0 and log.steps[1].capped is False


@pytest.mark.parametrize("cap", [None, 0, -5.0])
def test_a_run_has_a_cap_for_every_step(tmp_path, cap):
    with pytest.raises(ValueError):
        _fn("RecordingRunner")(Starter(), _log(tmp_path), cap_seconds=cap)


# --- The stop handle ------------------------------------------------------------------------------------------


def test_the_handle_stops_a_process_by_its_pid_and_hands_on_the_signal_and_the_exit(tmp_path):
    calls = []

    def fake_kill(pid, number):
        calls.append((pid, number))

    process = _Process(4242, "tracked 1\ntracked 2\nrest\n", -15)
    handle = _fn("ProcessHandle")(process, kill=fake_kill)
    assert handle.pid == 4242
    assert handle.readline() == "tracked 1\n"
    sent = handle.stop()
    # The number of the signal that was sent, and one kill call: by the pid of this process.
    assert sent == 15 and calls == [(4242, 15)]
    ended = handle.wait()
    assert ended.exit_code == -15
    # What readline did not give; the lines it gave are the caller's.
    assert ended.output == "tracked 2\nrest\n"
    other = _fn("ProcessHandle")(_Process(77, "", -9), kill=fake_kill, signal_number=9)
    assert other.stop() == 9 and calls[-1] == (77, 9)


def test_a_started_writer_is_kept_as_one_step_when_it_is_waited_for(tmp_path):
    process = _Process(4243, "tracked 1\ntracked 2\nrest\n", -15)
    handle = _fn("ProcessHandle")(process, kill=lambda pid, number: None)
    rec, log, _starter = _recorder(tmp_path, starter=Starter(handles=[handle]))
    started = rec.start(["writer", "--plan", "p"])
    assert started.readline() == "tracked 1\n"
    assert started.stop() == 15
    ended = started.wait()
    assert ended.exit_code == -15 and ended.output == "tracked 2\nrest\n"
    step = log.steps[-1]
    assert step.command == ("writer", "--plan", "p") and step.exit_code == -15
    assert "tracked 1" in step.output and "rest" in step.output
    assert isinstance(step.seconds, (int, float)) and step.seconds >= 0


def _stop_fields(code, sent=15, attempt_id="stop-1"):
    stop = _fn("stop_record")(
        command=("writer", "--plan", "p"), sent=sent, ended=_result(code, "rest\n", 2.0),
        attempt_id=attempt_id, rows_showed_while_running=True, output_before="tracked 100\n",
    )
    return stop


def test_one_stop_record_carries_the_names_both_subjects_read():
    stop = _stop_fields(-15)
    fields = dataclasses.asdict(stop)
    assert set(fields) == {
        "stopped_exit_code", "stopped_signal", "stopped_attempt_id", "stopped_command",
        "stopped_output", "rows_showed_while_running",
    }
    assert fields["stopped_exit_code"] == -15 and fields["stopped_signal"] == 15
    assert tuple(fields["stopped_command"]) == ("writer", "--plan", "p")
    assert fields["stopped_output"] == "tracked 100\nrest\n"
    assert fields["rows_showed_while_running"] is True
    dolma = need("baselines.dolma")
    originblame = need("baselines.originblame")
    # A stop that landed (the exit of the signal, as its negative or as 128 plus it) is read as
    # no failed attempt by both subjects.
    for code in (-15, 143):
        fields = dataclasses.asdict(_stop_fields(code))
        run = dolma.DolmaRun(build=Path("build"), rows=[], chunk_files={}, **fields)
        assert dolma.DolmaSubject._stop_that_did_not_land(run) is None
        assert originblame._stop_that_did_not_land(originblame.StoppedWriter(**fields)) is None
    # A stop that did not land is the failed attempt of the stopped writer in both.
    fields = dataclasses.asdict(_stop_fields(0))
    run = dolma.DolmaRun(build=Path("build"), rows=[], chunk_files={}, **fields)
    seen = dolma.DolmaSubject._stop_that_did_not_land(run)
    assert seen is not None and seen.id == "stop-1" and seen.exit_code == 0
    found = originblame._stop_that_did_not_land(originblame.StoppedWriter(**fields))
    assert found is not None and found[0] == "stop-1" and found[1] == 0


def test_the_second_run_of_the_tokenizer_is_handed_to_the_subject_with_its_id_and_command():
    retry = _fn("retry_record")(command=("dolma", "tokens", "--again"), ended=_result(1, "Traceback: boom", 5.0),
                                attempt_id="retry-1")
    fields = dataclasses.asdict(retry)
    assert set(fields) == {"retry_attempt_id", "retry_command", "retry_exit_code", "retry_output"}
    dolma = need("baselines.dolma")
    run = dolma.DolmaRun(build=Path("build"), rows=[], chunk_files={}, **fields)
    # An end that is no refusal of the toolkit is the failed attempt of the second run, with its id.
    attempt = dolma.DolmaSubject._failed_second_run(run)
    assert attempt.id == "retry-1" and attempt.exit_code == 1
    assert tuple(attempt.command) == ("dolma", "tokens", "--again")
    assert "boom" in attempt.output


# --- Faults put to the tool's own files ----------------------------------------------------------------------

COMPANION_ROWS = [
    "0,10,r-1,E/c.jsonl,1",
    "10,25,r-2,E/c.jsonl,2",
    "25,40,r-3,E/c.jsonl,3",
    "0,9,r-9,E/d.jsonl,1",
]


def _write_companion(path: Path, packed: bool) -> None:
    text = "\n".join(COMPANION_ROWS) + "\n"
    path.write_bytes(gzip.compress(text.encode("utf-8")) if packed else text.encode("utf-8"))


def _read_companion_lines(path: Path, packed: bool) -> list[str]:
    data = path.read_bytes()
    assert (data[:2] == b"\x1f\x8b") is packed, "the file did not keep its own format"
    return (gzip.decompress(data) if packed else data).decode("utf-8").splitlines()


@pytest.mark.parametrize("packed", [False, True], ids=["plain", "gzip"])
def test_a_companion_fault_is_written_back_in_the_files_own_format(tmp_path, packed):
    put = _fn("put_dolma_fault")
    read_rows = need_attr("baselines.companion", "read_rows")
    path = tmp_path / "part-0.csv"
    _write_companion(path, packed)
    assert put("f.digest", path, "E/c.jsonl") is True
    lines = _read_companion_lines(path, packed)
    assert len(lines) == 4 and lines[1:] == COMPANION_ROWS[1:]
    start, end, identifier, src, loc = lines[0].split(",")
    assert (start, identifier, src, loc) == ("0", "r-1", "E/c.jsonl", "1")
    # The first digit of end is another digit; the rest of the number and its length stay.
    assert len(end) == 2 and end[1] == "0" and end[0] not in ("1", "0")
    assert len(read_rows(path)) == 4
    # The last row of the chunk file goes (the row with the greatest loc), nothing else.
    _write_companion(path, packed)
    assert put("l.count-499", path, "E/c.jsonl") is True
    assert _read_companion_lines(path, packed) == COMPANION_ROWS[:2] + COMPANION_ROWS[3:]
    # Every row of the chunk file goes; the rows of the other chunk file stay.
    _write_companion(path, packed)
    assert put("f.map-deleted", path, "E/c.jsonl") is True
    assert _read_companion_lines(path, packed) == COMPANION_ROWS[3:]
    assert [row.id for row in read_rows(path)] == ["r-9"]


@pytest.mark.parametrize("fault", ["w.kill-after-100", "w.retry", "l.artifact-digest"])
def test_a_fault_that_is_put_elsewhere_changes_nothing_and_is_no_failed_attempt(tmp_path, fault):
    put = _fn("put_dolma_fault")
    path = tmp_path / "part-0.csv"
    _write_companion(path, False)
    before = path.read_bytes()
    assert put(fault, path, "E/c.jsonl") is False
    assert path.read_bytes() == before
    # A fault the table does not hold is an error of the caller, not a no-op.
    with pytest.raises(need_attr("baselines.faults", "FaultError")):
        put("f.no-such-fault", path, "E/c.jsonl")


ENTRY_FIELDS = ("sources", "token_count", "tokenizer", "revoked")


def _entry_line(source, count):
    return json.dumps({"sources": [source], "token_count": count, "tokenizer": "gpt2", "revoked": False})


def _entry_store(folder: Path) -> Path:
    """A store copy: two entry files (the integer names), a section record beside them."""
    entries = folder / ".ob" / "token-index.gpt2"
    entries.mkdir(parents=True)
    (entries / "0").write_text(_entry_line("h-a", 12) + "\n" + _entry_line("h-b", 30) + "\n", encoding="utf-8")
    (entries / "1").write_text(_entry_line("h-c", 41) + "\n", encoding="utf-8")
    (folder / ".ob" / "sections").write_text('{"sources/A.jsonl": "h-a"}\n', encoding="utf-8")
    return folder


def _entries_of(folder: Path):
    return need_attr("baselines.originblame", "read_store_entries")(folder)


def _entry(source, count):
    return need_attr("baselines.tokenindex", "Entry")((source,), count, "gpt2", False)


def test_a_digit_fault_changes_the_first_digit_of_the_first_entry_on_its_own_file(tmp_path):
    # The table: "in the first entry of the store that tracked E/shard-3 the first digit of
    # token_count is replaced by another digit".
    copy = _entry_store(tmp_path / "copy")
    got = _fn("ob_fault_putter")(copy, "f.digest")
    assert isinstance(got, dict)
    entries = _entries_of(copy)
    assert len(entries) == 3
    first = entries[0]
    text = str(first.token_count)
    assert len(text) == 2 and text[1] == "2" and text[0] not in ("1", "0")
    assert (first.sources, first.tokenizer, first.revoked) == (("h-a",), "gpt2", False)
    assert entries[1:] == [_entry("h-b", 30), _entry("h-c", 41)]
    # Each entry stays one JSON line with its four fields, in the file it was in.
    one = (copy / ".ob" / "token-index.gpt2" / "0").read_text(encoding="utf-8").splitlines()
    assert len(one) == 2 and set(json.loads(one[0])) == set(ENTRY_FIELDS)
    assert len((copy / ".ob" / "token-index.gpt2" / "1").read_text(encoding="utf-8").splitlines()) == 1


def test_a_count_fault_deletes_the_last_entry_and_a_map_fault_every_entry_and_keeps_the_sections(tmp_path):
    # The table: "the last of the 500 token-index entries of the store is deleted" and "the
    # token-index entries of the store that tracked E/shard-1 are deleted; its section records stay".
    put = _fn("ob_fault_putter")
    copy = _entry_store(tmp_path / "count")
    put(copy, "l.count-499")
    assert _entries_of(copy) == [_entry("h-a", 12), _entry("h-b", 30)]
    mapped = _entry_store(tmp_path / "map")
    sections = (mapped / ".ob" / "sections").read_bytes()
    put(mapped, "f.map-deleted")
    assert _entries_of(mapped) == []
    assert (mapped / ".ob" / "sections").read_bytes() == sections


def test_the_record_of_a_faulted_store_says_whether_the_binary_index_was_there_and_fresh(tmp_path):
    put = _fn("ob_fault_putter")
    state = _fn("binary_index_state")
    # Fresh: the binary index is not older than any entry file. Times are set by hand.
    fresh = _entry_store(tmp_path / "fresh")
    binary = fresh / ".ob" / "token-index.gpt2.bin"
    binary.write_bytes(b"binary index")
    for name in ("0", "1"):
        os.utime(fresh / ".ob" / "token-index.gpt2" / name, (1_000_000_000, 1_000_000_000))
    os.utime(binary, (1_000_000_100, 1_000_000_100))
    assert state(fresh) == {"existed": True, "fresh": True}
    got = put(fresh, "f.digest")
    # After the fault the entry file is newer than the binary index: the tool reads the entries.
    assert got["binary_index"] == {
        "before": {"existed": True, "fresh": True},
        "after": {"existed": True, "fresh": False},
    }
    # A store with no binary index says so, and an older one is not fresh.
    bare = _entry_store(tmp_path / "bare")
    assert state(bare) == {"existed": False, "fresh": False}
    assert put(bare, "l.count-499")["binary_index"]["before"] == {"existed": False, "fresh": False}
    stale = _entry_store(tmp_path / "stale")
    old = stale / ".ob" / "token-index.gpt2.bin"
    old.write_bytes(b"binary index")
    for name in ("0", "1"):
        os.utime(stale / ".ob" / "token-index.gpt2" / name, (1_000_000_200, 1_000_000_200))
    os.utime(old, (1_000_000_100, 1_000_000_100))
    assert state(stale) == {"existed": True, "fresh": False}


def test_a_line_carries_the_notes_the_runner_made_for_its_row(tmp_path):
    _rec, log, _starter = _recorder(tmp_path)
    question = _make_question("c05.f.digest.E/shard-3", "originblame")
    notes = {"binary_index": {"before": {"existed": True, "fresh": True},
                              "after": {"existed": True, "fresh": False}}}
    answered = _fn("answer_question")(question, Subject("originblame"), commit=COMMIT, log=log,
                                      grade_of=lambda q, r: _grade("right"), notes=notes)
    assert answered.line["notes"] == notes
    plain = _fn("answer_question")(question, Subject("originblame"), commit=COMMIT, log=log,
                                   grade_of=lambda q, r: _grade("right"))
    assert "notes" not in plain.line or plain.line["notes"] in (None, {})


# --- What a step needs on disk before it starts ----------------------------------------------------------------


def test_the_work_folders_of_the_tokenizer_and_the_mixer_and_the_temporary_folder_are_made(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    scratch = tmp_path / "work" / "tmp-dolma"
    scratch.parent.mkdir()
    _fn("prepare_tokenizer_folders")(build, work_dir="tokenizer-work", tmpdir=scratch)
    assert (build / "tokenizer-work" / "in").is_dir() and (build / "tokenizer-work" / "out").is_dir()
    assert scratch.is_dir()
    _fn("prepare_mixer_folders")(build, work_dir="mixer-work")
    assert (build / "mixer-work" / "in").is_dir() and (build / "mixer-work" / "out").is_dir()
    # A second start of the same step finds them there.
    _fn("prepare_mixer_folders")(build, work_dir="mixer-work")
    # The folders are relative to the build and stay inside it.
    for bad in ("/elsewhere", "../outside"):
        with pytest.raises(ValueError):
            _fn("prepare_mixer_folders")(build, work_dir=bad)


def test_the_tokenizer_file_is_copied_into_the_build_and_named_relative_to_it(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    source = tmp_path / "tokenizer-source.json"
    source.write_bytes(b'{"model": "stand-in"}')
    named = _fn("place_tokenizer")(build, source, name="gpt2-tokenizer.json")
    assert not os.path.isabs(named) and ".." not in Path(named).parts
    assert (build / named).read_bytes() == b'{"model": "stand-in"}'
    assert source.read_bytes() == b'{"model": "stand-in"}'


def test_the_rows_after_a_stop_are_read_from_a_file_the_stop_tore(tmp_path):
    folder = tmp_path / "out"
    folder.mkdir()
    whole = gzip.compress(b"0,10,r-1,E/c.jsonl,1\n10,25,r-2,E/c.jsonl,2\n")
    # The end of the file (its checksum and size) never came: the stop landed before the writer closed it.
    (folder / "part-0.csv.gz").write_bytes(whole[:-8])
    rows = _fn("rows_after_stop")(folder)
    assert [(row.id, row.loc) for row in rows] == [("r-1", 1), ("r-2", 2)]


# --- Readings the table gives no join for, and the reading of a sample ----------------------------------------


def _trace_document():
    def row(ordinal, start, end):
        return {"shard": "E/shard-1", "shard_position": 1, "ordinal": ordinal,
                "token_start": start, "token_end": end}

    return {"segments": [{
        "segment": "E/seg-2", "run": "E", "steps": [{
            "step": 8, "consumed_samples": 4, "samples": [
                {"sample_pos": 2, "rank": 0, "sample_tokens": 33, "covered_tokens": 33, "rows": [row(0, 0, 33)]},
                {"sample_pos": 3, "rank": 0, "sample_tokens": 33, "covered_tokens": 16,
                 "rows": [row(1, 0, 10), row(2, 0, 6)]},
            ],
        }],
    }]}


def test_the_sample_row_count_is_read_from_the_sample_itself():
    count = _fn("sample_row_count")
    document = _trace_document()
    assert count(document, "E/seg-2", 8, 3) == 2
    assert count(document, "E/seg-2", 8, 2) == 1
    for segment, step, position in (("E/seg-2", 8, 9), ("E/seg-2", 7, 3), ("E/seg-9", 8, 3)):
        with pytest.raises(ValueError):
            count(document, segment, step, position)


def test_the_id_map_of_a_build_is_made_once_and_gives_what_the_per_id_scan_gives(tmp_path):
    build = _build_with_records(tmp_path / "build")
    resolve_id = need_attr("baselines.companion", "resolve_id")
    ids = _fn("IdMap")(build)
    # The two record files the store registers from are read once, at the start.
    assert ids.files_read == 2
    for identifier in ("a-1", "t-1", "m-1", "h-1", "zz"):
        assert ids.file_of(identifier) == resolve_id(build, identifier), identifier
    assert ids.file_of("a-1") == "sources/A.jsonl" and ids.file_of("t-1") == "gen/T.jsonl"
    assert ids.file_of("m-1") is None and ids.file_of("zz") is None
    for _ in range(200):
        ids.file_of("a-1")
        ids.file_of("zz")
    assert ids.files_read == 2
    # An id two files hold is refused as the per-id scan refuses it.
    (build / "gen" / "U.jsonl").write_text('{"id": "a-1"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        _fn("IdMap")(build).file_of("a-1")


# --- Answering one question --------------------------------------------------------------------------------------


def test_an_expected_value_that_is_not_derived_never_costs_the_question_its_line(tmp_path):
    _rec, log, _starter = _recorder(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    said = "the counting document holds no documents.shards"
    error = need_attr("baselines.expected", "PartMissing")(said)

    def grade_of(q, reading):
        raise error

    answered = _fn("answer_question")(question, Subject("dolma", raw={"answer": "contains"}), commit=COMMIT,
                                      log=log, grade_of=grade_of)
    line = answered.line
    assert line is not None and line["id"] == question.id
    assert line["status"] == "cannot answer"
    assert "expected value not derived" in str(line["evidence"]) and said in str(line["evidence"])
    assert line["raw"] == {"answer": "contains"} and line["commit"] == COMMIT
    assert answered.attempt is None


@pytest.mark.parametrize("listed", ["c06.generated.downstream-of-translation", "c09.10k.successor-shards"])
def test_a_row_the_table_gives_no_join_for_is_a_cannot_answer_line_and_no_failed_attempt(tmp_path, listed):
    _rec, log, _starter = _recorder(tmp_path)
    question = _make_question(listed, "dolma", "small" if listed.startswith("c06") else None)
    assert question.route is not None
    error = ValueError(f"the table leaves the reading of {listed} open")

    class Raises(Subject):
        def read_by_layout(self, q):
            raise error

    subject = Raises("dolma", error=error)
    answered = _fn("answer_question")(question, subject, commit=COMMIT, log=log,
                                      grade_of=lambda q, r: _grade("right"),
                                      grade_layouts_of=lambda q, outcomes: _grade("right"))
    assert answered.line["status"] == "cannot answer"
    assert answered.line["evidence"] == "the table gives no join of the shards' answers"
    assert answered.line["attempt_id"] is None
    assert answered.attempt is None


def test_a_kind_row_read_after_a_failed_tokens_run_ends_as_a_line_citing_that_attempt(tmp_path):
    _rec, log, _starter = _recorder(tmp_path)
    # A source file asked in a routed group: the toolkit holds no record of it, and its reading
    # needs the successful tokenizer run of the store.
    listed = "c05.f.digest.src/A"
    question = _make_question(listed, "dolma")
    assert question.route is not None
    target = record_target(question)
    failed = _attempt("dolma-tokens-1", output="no companion row was written")
    error = ValueError("the run holds no companion row: it is no successful tokenizer run")
    answered = _fn("answer_question")(question, Subject("dolma", error=error), commit=COMMIT, log=log,
                                      grade_of=lambda q, r: _grade("right"), failed_runs={target: failed})
    assert answered.line["status"] == "no answer: failed attempt dolma-tokens-1"
    assert answered.line["attempt_id"] == "dolma-tokens-1"
    assert answered.attempt is failed
    # With no failed run of that store or set the error is the caller's, never a quiet line.
    with pytest.raises(ValueError):
        _fn("answer_question")(question, Subject("dolma", error=error), commit=COMMIT, log=log,
                               grade_of=lambda q, r: _grade("right"), failed_runs={})
    other = {("store", "rr"): failed}
    with pytest.raises(ValueError):
        _fn("answer_question")(question, Subject("dolma", error=error), commit=COMMIT, log=log,
                               grade_of=lambda q, r: _grade("right"), failed_runs=other)


def test_a_reading_that_raised_cites_the_failed_run_of_the_layout_it_read_not_the_run_of_the_other_layout(tmp_path):
    """W8 (18 Nov 2026): the 10k set holds a build in each layout. A partitioned question whose reading raised must not cite the
    round-robin build's failure (a failure that run never had for it): where the partitioned build ran the error is the caller's;
    where it failed, the line is its failed attempt."""
    _rec, log, _starter = _recorder(tmp_path)
    question = _make_question("c09.10k.partitioned.kept-shards", "dolma")
    assert question.route is not None
    target = record_target(question)
    round_robin_failed = _attempt("dolma-tokens-rr", output="no companion row was written")
    partitioned_failed = _attempt("dolma-tokens-part", output="no companion row was written")
    error = ValueError("the run holds no companion row: it is no successful tokenizer run")
    answer = _fn("answer_question")
    ran = {(*target, "partitioned"): None}
    with pytest.raises(ValueError):
        answer(question, Subject("dolma", error=error), commit=COMMIT, log=log, grade_of=lambda q, r: _grade("right"),
               failed_runs={target: round_robin_failed}, layout_runs=ran)
    failed = answer(question, Subject("dolma", error=error), commit=COMMIT, log=log, grade_of=lambda q, r: _grade("right"),
                    failed_runs={}, layout_runs={(*target, "partitioned"): partitioned_failed})
    assert failed.line["attempt_id"] == "dolma-tokens-part" and failed.attempt is partitioned_failed
    # A question that names the layout the target's own build is in still cites the target's failed run.
    other = _make_question("c09.10k.round-robin.all-rewrite", "dolma")
    cited = answer(other, Subject("dolma", error=error), commit=COMMIT, log=log, grade_of=lambda q, r: _grade("right"),
                   failed_runs={target: round_robin_failed}, layout_runs=ran)
    assert cited.line["attempt_id"] == "dolma-tokens-rr"


def test_an_answer_that_used_the_tie_carries_its_mark_and_one_that_did_not_does_not(tmp_path):
    _rec, log, _starter = _recorder(tmp_path)
    mark = need_attr("baselines.interfaces", "MARK_TIE")
    question = _make_question("c05.unmutated.E@9", "dolma+trace")
    assert question.tie_mark == mark
    used = _fn("answer_question")(question, Subject("dolma+trace", tied=True), commit=COMMIT, log=log,
                                  grade_of=lambda q, r: _grade("right"))
    assert mark in used.line["marks"]
    unused = _fn("answer_question")(question, Subject("dolma+trace", tied=False), commit=COMMIT, log=log,
                                    grade_of=lambda q, r: _grade("right"))
    assert mark not in unused.line["marks"]


class ByLayout:
    name = "originblame"

    def __init__(self, outcomes):
        self.outcomes = outcomes

    def read(self, question):
        raise AssertionError("a row that names two layouts is read by layout")

    def read_by_layout(self, question):
        return self.outcomes


def test_a_row_of_two_layouts_is_read_by_layout_and_its_line_follows_the_combined_status(tmp_path):
    _rec, log, _starter = _recorder(tmp_path)
    question = _make_question("c09.10k.successor-shards", "originblame")
    assert question.route is not None
    assert need_attr("baselines.record", "layouts_of")(question) == ("partitioned", "round-robin")
    good = _reading(question, {"answer": "contains"})
    failed = _attempt("ob-track-2")
    seen = {}

    def combined(status):
        def grade_layouts_of(q, outcomes):
            seen["outcomes"] = outcomes
            return _grade(status, {"partitioned": {"status": "right", "evidence": None}})

        return grade_layouts_of

    def boom(q, reading):
        raise AssertionError("the combined grade is the row's grade")

    answer = _fn("answer_question")
    # A failed attempt in one layout: the line is that attempt's status, and the attempt is kept.
    got = answer(question, ByLayout({"partitioned": good, "round-robin": failed}), commit=COMMIT, log=log,
                 grade_of=boom, grade_layouts_of=combined("no answer: failed attempt ob-track-2"))
    assert set(seen["outcomes"]) == {"partitioned", "round-robin"}
    assert got.line["status"] == "no answer: failed attempt ob-track-2"
    assert got.line["attempt_id"] == "ob-track-2" and got.attempt is failed
    # A layout that was not attempted: no line is written for the row.
    none_yet = answer(question, ByLayout({"partitioned": good, "round-robin": None}), commit=COMMIT, log=log,
                      grade_of=boom, grade_layouts_of=combined("not attempted"))
    assert none_yet.line is None and none_yet.attempt is None
    # Both layouts graded right: the line is right.
    right = answer(question, ByLayout({"partitioned": good, "round-robin": good}), commit=COMMIT, log=log,
                   grade_of=boom, grade_layouts_of=combined("right"))
    assert right.line["status"] == "right" and right.attempt is None


def test_the_none_lines_are_written_from_the_one_attempt_of_each_subject_on_its_store_or_set():
    none_lines = _fn("none_lines")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    no_answer = need_attr("baselines.interfaces", "no_answer")
    on_main = _make_question("c01.E.T", "manifest")
    on_scratch = _make_question("c04.count-499", "trace")
    uncovered = _make_question("c04.count-499", "manifest")
    routed = _make_question("c05.unmutated.R1", "manifest")
    for question in (on_main, on_scratch, uncovered):
        assert question.route is None
    assert routed.route is not None
    attempts = [_store_attempt("manifest", MAIN), _store_attempt("trace", SCRATCH_500, attempt_id="trace-export-1")]
    lines = none_lines([on_main, on_scratch, uncovered, routed], attempts, commit=COMMIT,
                       step_ids_of=lambda question: ("step-1",))
    by_id = {line["id"]: line for line in lines}
    # No attempt of the manifest on scratch-500 and a question with a route: no line.
    assert set(by_id) == {on_main.id, on_scratch.id}
    assert [line["id"] for line in lines] == [on_main.id, on_scratch.id]
    assert by_id[on_main.id]["status"] == cannot and by_id[on_main.id]["evidence"] == on_main.missing
    assert by_id[on_scratch.id]["status"] == no_answer("trace-export-1")
    assert by_id[on_scratch.id]["attempt_id"] == "trace-export-1"
    assert all(line["step_ids"] == ["step-1"] and line["commit"] == COMMIT for line in lines)


def test_the_none_lines_of_a_store_run_twice_read_each_the_run_they_ask_about():
    none_lines = _fn("none_lines")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    no_answer = need_attr("baselines.interfaces", "no_answer")
    kill = _make_question("c04.kill-after-100", "manifest")
    retry = _make_question("c04.retry", "manifest")
    declared = _make_question("c04.declare-499.write", "dolma")
    assert kill.route is None and retry.route is None and declared.route is None
    attempts = [
        _store_attempt("manifest", SCRATCH_500, run=1),
        _store_attempt("manifest", SCRATCH_500, attempt_id="retry-failed", run=2),
        _store_attempt("dolma", SCRATCH_500, run=1),
        _store_attempt("dolma", SCRATCH_500, attempt_id="retry-failed-2", run=2),
    ]
    lines = {line["id"]: line for line in none_lines([kill, retry, declared], attempts, commit=COMMIT)}
    assert lines[kill.id]["status"] == cannot
    assert lines[retry.id]["status"] == no_answer("retry-failed")
    assert lines[declared.id]["status"] == cannot


# --- The options and the start line -----------------------------------------------------------------------------


def test_the_subjects_and_the_step_cap_are_options_of_the_runner_itself():
    parse = _fn("parse_arguments")
    parsed = parse(["--subjects", "dolma", "originblame", "--step-cap", "600"])
    assert tuple(parsed.subjects) == ("dolma", "originblame")
    assert parsed.step_cap == 600.0
    # Without the option every subject of the table is selected.
    every = parse(["--step-cap", "600"])
    assert tuple(every.subjects) == need_attr("baselines.interfaces", "SUBJECTS")
    # A subject the table does not have, a cap that is missing and a cap that is no time: refused.
    for bad in (["--subjects", "no-such-subject", "--step-cap", "600"], ["--subjects", "dolma"],
                ["--subjects", "dolma", "--step-cap", "0"], ["--subjects", "dolma", "--step-cap", "-3"]):
        with pytest.raises(SystemExit):
            parse(bad)


def test_the_module_run_as_a_program_calls_the_runner_main():
    entry = need_attr("baselines.__main__", "main")
    assert callable(entry) and entry is _fn("main")


def test_a_pass_with_an_extension_on_is_refused_on_a_world_built_without_it(tmp_path):
    """A world records the extensions its rows were built for (``extensions`` of world.json; a world
    without the key was built with none). A pass that asks an extension the world was not built with
    stops before any run: the questions of that extension would stay without a line."""
    world_file = tmp_path / "world.json"
    world_file.write_text("{}", encoding="utf-8")
    arguments = ["--subjects", "manifest", "--work", str(tmp_path / "work"), "--run-record", str(tmp_path / "run"),
                 "--lock-prefix", "true", "--world", str(world_file), "--step-cap", "60",
                 "--extensions", "generating-model"]
    with pytest.raises(SystemExit) as caught:
        _fn("main")(arguments)
    assert "generating-model" in str(caught.value) and "built" in str(caught.value), str(caught.value)
    check = _fn("check_extensions_built")
    assert check({"extensions": ["generating-model"]}, ["generating-model"]) is None
    assert check({}, []) is None and check({"extensions": ["generating-model"]}, []) is None
    assert check({}, ["generating-model"]) is not None
    assert check({"extensions": []}, ["generating-model"]) is not None


def test_the_start_line_names_the_subjects_the_step_cap_and_the_sha_of_each_identity_file():
    first = hashlib.sha256(b"identity of the first environment").hexdigest()
    second = hashlib.sha256(b"identity of the second environment").hexdigest()
    line = _fn("start_line")(subjects=("dolma", "originblame"), step_cap_seconds=600.0,
                             identity_sha256s={"dolma": first, "originblame": second}, commit=COMMIT)
    assert isinstance(line, str) and "\n" not in line
    for word in ("dolma", "originblame", "600", first, second, COMMIT):
        assert word in line, word


# --- The documents of the part -----------------------------------------------------------------------------------

EM_DASH = chr(0x2014)


def _em_dash_in(text: str) -> list[str]:
    """The em dash is the one mark kept out of the text of the decision entries from D-170 on and of the
    baselines document. The mark of the src and loc tie is the plan's and the table's own text: it may stand whole."""
    text = text.replace(need_attr("baselines.interfaces", "MARK_TIE"), "")
    return [EM_DASH] if EM_DASH in text else []


def test_the_decisions_file_holds_entries_from_170_each_with_its_four_parts():
    text = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    entries = {}
    for block in re.split(r"(?m)^## (?=D-\d+ )", text)[1:]:
        entries[int(re.match(r"D-(\d+) ", block).group(1))] = block
    new = {number: body for number, body in entries.items() if number >= 170}
    assert new, "DECISIONS.md holds no entry from D-170 on"
    assert sorted(new) == list(range(170, 170 + len(new))), "the new entries are not numbered from D-170 without a gap"
    for number, body in new.items():
        for label in ("Built", "Other option", "Result of the other option", "What would change it"):
            assert label in body, f"D-{number} has no {label!r}"
        assert "traced" in body.lower() or "inferred" in body.lower(), f"D-{number} says neither traced nor inferred"
        assert _em_dash_in(body) == [], f"D-{number} holds an em dash"


def test_the_document_of_the_baselines_names_the_command_the_options_and_the_files_of_a_run():
    path = REPO / "docs" / "baselines.md"
    assert path.is_file(), "docs/baselines.md does not exist"
    text = path.read_text(encoding="utf-8")
    for word in ("python -m baselines", "--subjects", "--step-cap", "steps.jsonl", "sha256sums.txt",
                 "compare.json", "identity.json"):
        assert word in text, word
    assert _em_dash_in(text) == []


# --- The frozen program: the sha256 supplied from the command line -----------------------------------------------


def _needs_parameter(function, name):
    assert name in inspect.signature(function).parameters, f"{function.__name__} takes no {name}"


def _needs_field(type_, name):
    assert name in {item.name for item in dataclasses.fields(type_)}, f"{type_.__name__} has no field {name}"


def test_a_program_other_than_the_one_supplied_stops_the_run_even_when_identity_json_names_it(tmp_path):
    stop_type = _fn("PreRunStop")
    _needs_parameter(_fn("pre_run"), "program_sha256")
    # The setup and the program agree: identity.json names the program this file holds now.
    w = _world(tmp_path / "swapped", name="originblame", program=True)
    assert json.loads(w.identity_file.read_text(encoding="utf-8"))["ob_binary_sha256"] == w.program_sha
    supplied = hashlib.sha256(b"the program that was named on the command line").hexdigest()
    assert supplied != w.program_sha
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    gated = _gate(starter)
    with pytest.raises(stop_type) as caught:
        _pre_run(w, gated, program_sha256=supplied)
    assert caught.value.stage == "binary" and caught.value.reason
    assert gated.released is False
    assert all(_is_probe(argv) for argv in starter.argvs), "no tool command was started"
    # The supplied sha256 is the program's own: the run goes on.
    right = _world(tmp_path / "right", name="originblame", program=True)
    starter = Starter(probe=(right.probe_path, right.probe_sha))
    gated = _gate(starter)
    got = _pre_run(right, gated, program_sha256=right.program_sha)
    assert got.ob_binary_sha256 == right.program_sha and gated.released is True


def test_the_supplied_program_sha_of_a_setup_reaches_the_gate_of_a_pass(tmp_path):
    _needs_field(_fn("ToolSetup"), "program_sha256")
    w = _world(tmp_path, name="originblame", program=True)
    other = hashlib.sha256(b"another program than the one in the setup").hexdigest()
    setup = _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha,
                             ob_binary=w.program, program_sha256=other)
    question = _make_question("c05.unmutated.E/shard-1", "originblame")
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    config = _config(tmp_path, w, subjects=("originblame",), factories={"originblame": _never("originblame")},
                     questions=[question], setups={"originblame": setup})
    with pytest.raises(_fn("PreRunStop")) as caught:
        _fn("run_pass")(config, starter)
    assert caught.value.stage == "binary"
    assert all(_is_probe(argv) for argv in starter.argvs)
    answers = w.run_record / "answers.jsonl"
    assert not answers.exists() or answers.read_text(encoding="utf-8") == ""


def _main_code(arguments):
    try:
        return _fn("main")([str(word) for word in arguments])
    except SystemExit as leaving:
        return leaving.code


def test_the_program_sha_and_the_lock_wait_bound_are_options_of_the_runner_itself():
    def parse(arguments):
        try:
            return _fn("parse_arguments")(arguments)
        except SystemExit:
            return None

    sha = hashlib.sha256(b"a program").hexdigest()
    parsed = parse(["--step-cap", "600", "--program-sha256", sha, "--lock-wait-bound", "120"])
    assert parsed is not None, "the options are not known"
    assert parsed.program_sha256 == sha and parsed.lock_wait_bound == 120.0
    absent = parse(["--step-cap", "600"])
    assert absent is not None and absent.program_sha256 is None
    for bad in ("0", "-3", "soon"):
        assert parse(["--step-cap", "600", "--lock-wait-bound", bad]) is None, bad


def test_the_command_hands_the_supplied_program_sha_and_the_lock_wait_bound_to_the_pass(tmp_path, monkeypatch):
    runner = need("baselines.runner")
    caught = []

    def stand_in(config, starter):
        caught.append(config)
        raise runner.PreRunStop("compare", "a stand-in pass that starts nothing")

    monkeypatch.setattr(runner, "run_pass", stand_in)
    world = tmp_path / "world.json"
    world.write_text("{}", encoding="utf-8")
    # A world folder holds the list of its hashes beside world.json: a miniature one with the one row.
    (tmp_path / "WORLD.sha256").write_text(f"{hashlib.sha256(world.read_bytes()).hexdigest()}  world.json\n",
                                           encoding="utf-8")
    program_sha = hashlib.sha256(b"the program named on the command line").hexdigest()
    arguments = [
        "--subjects", "originblame", "--step-cap", "600", "--commit", COMMIT, "--work", tmp_path / "work",
        "--run-record", tmp_path / "run-record", "--lock-prefix", "a-prefix", "--world", world,
        "--program-sha256", program_sha, "--lock-wait-bound", "120",
        "--tool", "originblame", tmp_path / "env", tmp_path / "record", hashlib.sha256(b"identity").hexdigest(),
        tmp_path / "program",
    ]
    assert _main_code(arguments) == 3
    assert len(caught) == 1
    config = caught[0]
    assert config.setups["originblame"].program_sha256 == program_sha
    assert getattr(config, "lock_wait_bound", None) == 120.0


def test_the_start_line_names_the_sha256_of_the_list_of_hashes_beside_the_world_file(tmp_path, monkeypatch, capsys):
    runner = need("baselines.runner")

    def stand_in(config, starter):
        raise runner.PreRunStop("compare", "a stand-in pass that starts nothing")

    monkeypatch.setattr(runner, "run_pass", stand_in)
    folder = tmp_path.resolve()
    (folder / "tokenizer.json").write_text('{"stand_in": "a tokenizer"}\n', encoding="utf-8")
    (folder / "tie.json").write_text('{"chunk_files": {}}\n', encoding="utf-8")
    world = {"tokenizer": str(folder / "tokenizer.json"), "tie_record": str(folder / "tie.json"), "requests": {},
             "split_pairs": [], "stores": [], "shards": [], "subject_inputs": {}}
    (folder / "world.json").write_text(json.dumps(world), encoding="utf-8")
    listed = folder / "WORLD.sha256"
    listed.write_text("".join(f"{hashlib.sha256((folder / name).read_bytes()).hexdigest()}  {name}\n"
                              for name in ("tie.json", "tokenizer.json", "world.json")), encoding="utf-8")
    arguments = [
        "--subjects", "dolma", "--step-cap", "600", "--commit", COMMIT, "--work", folder / "work",
        "--run-record", folder / "run-record", "--lock-prefix", "a-prefix", "--world", folder / "world.json",
        "--tool", "dolma", folder / "env", folder / "record", hashlib.sha256(b"identity").hexdigest(),
    ]
    assert _main_code(arguments) == 3
    wanted = hashlib.sha256(listed.read_bytes()).hexdigest()
    start = (folder / "run-record" / "start.txt").read_text(encoding="utf-8")
    assert wanted in start, f"the start line does not name the sha256 of WORLD.sha256 ({wanted}): {start}"
    assert "WORLD.sha256" in start, f"the start line does not say which file the sha256 is of: {start}"
    assert wanted in capsys.readouterr().out, "the line the command prints does not name it"
    # What the line holds besides stays what it was: the commit, the subjects and the step cap.
    for word in (COMMIT, "dolma", "600"):
        assert word in start, word


def test_the_document_of_the_baselines_names_the_program_sha_and_the_lock_wait_options():
    path = REPO / "docs" / "baselines.md"
    assert path.is_file(), "docs/baselines.md does not exist"
    text = path.read_text(encoding="utf-8")
    for word in ("--program-sha256", "--lock-wait-bound", "lock_wait_seconds"):
        assert word in text, word
    assert _em_dash_in(text) == []


def test_the_gate_hands_the_lock_keywords_on_to_the_starter_it_guards():
    inner = SimpleNamespace(calls=[])

    def run(argv, timeout=None, **keywords):
        inner.calls.append(("run", list(argv), timeout, keywords))
        return _result(0, "", 0.1)

    def start(argv, **keywords):
        inner.calls.append(("start", list(argv), None, keywords))
        return object()

    inner.run, inner.start = run, start
    gated = _gate(inner)
    gated.release()
    try:
        gated.run(["tool"], timeout=7, lock_note="note-file", lock_wait_bound=20)
        gated.start(["writer"], lock_note="note-file-2", lock_wait_bound=20)
    except TypeError as error:
        pytest.fail(f"the gate does not hand the lock keywords on: {error}")
    assert inner.calls == [
        ("run", ["tool"], 7, {"lock_note": "note-file", "lock_wait_bound": 20}),
        ("start", ["writer"], None, {"lock_note": "note-file-2", "lock_wait_bound": 20}),
    ]


def test_the_probe_is_a_step_behind_the_lock_too_with_its_own_note_and_bound(tmp_path):
    _needs_parameter(_fn("pre_run"), "lock_wait_bound")
    w = _world(tmp_path)
    seen = []

    class Locked:
        def run(self, argv, timeout=None, **keywords):
            seen.append(keywords)
            Path(argv[-1]).write_text(f"{w.probe_path}\n{w.probe_sha}\n", encoding="utf-8")
            return SimpleNamespace(exit_code=0, output="", seconds=0.25, lock_wait_seconds=7.5, lock_acquired=True)

        def start(self, argv, **keywords):
            raise AssertionError("no tool command is started")

    log = _fn("StepLog")(w.run_record)
    got = _pre_run(w, _gate(Locked()), log=log, lock_wait_bound=30)
    assert len(seen) == 1
    note = Path(seen[0]["lock_note"])
    note.relative_to(w.run_record)
    assert note.parent.is_dir(), "the folder of the note is there before the prefix writes into it"
    assert seen[0]["lock_wait_bound"] == 30
    line = json.loads((w.run_record / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "-I" in line["command"] and line["lock_wait_seconds"] == 7.5
    assert got.step.lock_wait_seconds == 7.5


# --- The lock prefix: one hold for each step, the wait outside the cap ----------------------------------------------
#
# The prefix of these tests is a script of their own, started with bash: it takes the scratch lock
# file named by its first argument (never the lock of the machine), then writes ``acquired <seconds
# waited>`` to the file in ``WATERWHITE_LOCK_NOTE`` and starts the rest of its arguments. Where the
# wait bound (``WATERWHITE_LOCK_WAIT_BOUND``, seconds) runs out it starts nothing, writes nothing and
# exits 75. A second prefix does not lock: it writes what it was given to the file named by its
# first argument and a note line of a hand-written wait. Both tolerate an environment value that
# is not set, so that a runner that sets none is seen failing at the assertion, not at the script.

LOCK_PREFIX_FLOCK = r'''
lock=$1
shift
# The bound reaches the prefix as a float ("30.0") and flock reads it in the locale's number format:
# under de_DE, flock -w 30.0 stops with "invalid timeout value" (docs/baselines.md, "The lock prefix").
# The prefix pins the numeric format; the runner reads the acquire line with a decimal point.
export LC_ALL=C
bound=${WATERWHITE_LOCK_WAIT_BOUND:-60}
export NOTE_FILE=${WATERWHITE_LOCK_NOTE:-/dev/null}
begun=$(date +%s.%N)
exec flock -o -E 75 -w "$bound" "$lock" bash -c 'awk -v a="$0" -v b="$(date +%s.%N)" "BEGIN{printf \"acquired %.3f\\n\", b-a}" > "$NOTE_FILE"; exec "$@"' "$begun" "$@"
'''

LOCK_PREFIX_LIES = r'''
dump=$1
shift
printf '%s\n%s\n' "${WATERWHITE_LOCK_NOTE-unset}" "${WATERWHITE_LOCK_WAIT_BOUND-unset}" >> "$dump"
if [ -n "${WATERWHITE_LOCK_NOTE-}" ]; then printf 'acquired %s\n' "$STANDIN_WAIT" > "$WATERWHITE_LOCK_NOTE"; fi
exec "$@"
'''


def _lock_world(tmp_path, prefix=LOCK_PREFIX_FLOCK):
    folder = tmp_path / "lock-world"
    folder.mkdir()
    lock = folder / "scratch.lock"
    lock.write_text("", encoding="utf-8")
    script = folder / "prefix.sh"
    script.write_text(prefix, encoding="utf-8")

    def argv(first, command_text):
        return ["bash", str(script), str(first), "bash", "-c", command_text]

    return SimpleNamespace(folder=folder, lock=lock, script=script, argv=argv)


def _lock_recorder(tmp_path, *, cap, bound):
    """A recording runner on the real starter, with the lock wait bound where the runner takes one."""
    log = _fn("StepLog")(tmp_path / "lock-record", roots={str(tmp_path): "<tmp>"})
    extra = {"lock_wait_bound": bound} if "lock_wait_bound" in inspect.signature(_fn("RecordingRunner")).parameters else {}
    return _fn("RecordingRunner")(_fn("SubprocessStarter")(), log, cap_seconds=cap, **extra), log


def _hold(lock, seconds, folder):
    """Hold the scratch lock for ``seconds`` from another process; return it once it holds the lock."""
    ready = folder / f"held-{time.monotonic_ns()}"
    process = subprocess.Popen(
        ["flock", "-o", str(lock), "bash", "-c", 'touch "$1"; sleep "$2"', "holder", str(ready), str(seconds)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not ready.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            process.kill()
            process.wait()
            raise AssertionError("the holder did not take the scratch lock")
        time.sleep(0.02)
    return process


def _release(holder):
    try:
        holder.wait(timeout=120)
    except subprocess.TimeoutExpired:
        holder.kill()
        holder.wait()


def _alive(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return text.rsplit(")", 1)[1].split()[0] != "Z"


def _gone(pid, seconds=15.0):
    deadline = time.monotonic() + seconds
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _lock_is_free(lock, seconds=15.0):
    deadline = time.monotonic() + seconds
    while True:
        done = subprocess.run(["flock", "-n", "-o", str(lock), "true"], capture_output=True)
        if done.returncode == 0:
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


def test_a_wait_for_the_lock_longer_than_the_cap_is_no_cap_hit_and_is_recorded_apart_from_the_step(tmp_path):
    w = _lock_world(tmp_path)
    rec, log = _lock_recorder(tmp_path, cap=2.0, bound=30)
    holder = _hold(w.lock, 5.0, w.folder)
    try:
        got = rec.run(w.argv(w.lock, "printf 'tool text\\n'; sleep 0.2"))
    finally:
        _release(holder)
    step = log.steps[0]
    assert step.capped is False and step.exit_code == 0 and got.exit_code == 0
    # The raw output holds the tool's text and nothing of the lock.
    assert got.output == "tool text\n" and step.output == "tool text\n"
    waited = getattr(step, "lock_wait_seconds", None)
    assert isinstance(waited, (int, float)) and 3.0 < waited < 30, "the wait is a field of the step, longer than the cap"
    assert step.seconds < 2.0, "the seconds of the step do not hold the wait"
    assert getattr(got, "lock_wait_seconds", None) == waited
    line = json.loads((log.folder / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert line["lock_wait_seconds"] == waited and "acquired" not in line["output"]


def test_the_wait_a_step_records_is_the_number_of_its_acquire_line(tmp_path, monkeypatch):
    w = _lock_world(tmp_path, LOCK_PREFIX_LIES)
    monkeypatch.setenv("STANDIN_WAIT", "7.5")
    rec, log = _lock_recorder(tmp_path, cap=30, bound=30)
    got = rec.run(w.argv(w.folder / "dump.txt", "printf 'tool text\\n'"))
    step = log.steps[0]
    assert got.exit_code == 0 and step.exit_code == 0 and step.output == "tool text\n"
    assert getattr(step, "lock_wait_seconds", None) == 7.5
    assert step.seconds < 5.0, "the hand-written wait is not part of the seconds of the step"


def test_every_step_gets_its_own_note_file_in_the_run_record_and_the_bound_in_seconds(tmp_path, monkeypatch):
    w = _lock_world(tmp_path, LOCK_PREFIX_LIES)
    dump = w.folder / "dump.txt"
    monkeypatch.setenv("STANDIN_WAIT", "0.25")
    rec, log = _lock_recorder(tmp_path, cap=30, bound=45)
    rec.run(w.argv(dump, "true"))
    rec.start(w.argv(dump, "printf 'a writer\\n'")).wait()
    lines = dump.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    notes, bounds = lines[0::2], lines[1::2]
    assert "unset" not in notes + bounds, "the runner sets both values on the prefix process"
    assert len(set(notes)) == 2, "one note for each step"
    for note in notes:
        assert Path(note).is_absolute()
        Path(note).relative_to(log.folder)
        assert Path(note).read_text(encoding="utf-8") == "acquired 0.25\n"
    assert [float(found) for found in bounds] == [45.0, 45.0]
    assert log.steps[0].id in Path(notes[0]).relative_to(log.folder).parts, "the note is in the folder of its step"
    assert [getattr(step, "lock_wait_seconds", None) for step in log.steps] == [0.25, 0.25]


def test_a_wait_bound_that_runs_out_is_read_from_the_missing_acquire_line_and_stops_before_the_step(tmp_path):
    stop_type = _fn("LockWaitStop")
    w = _lock_world(tmp_path)
    marker = w.folder / "started"
    rec, log = _lock_recorder(tmp_path, cap=30, bound=1)
    holder = _hold(w.lock, 6.0, w.folder)
    try:
        begun = time.monotonic()
        with pytest.raises(stop_type) as caught:
            rec.run(w.argv(w.lock, f"touch '{marker}'"))
        elapsed = time.monotonic() - begun
    finally:
        _release(holder)
    assert elapsed < 5.0, "the run did not wait for the holder past its own bound"
    assert not marker.exists(), "the command was not started"
    assert "lock" in str(caught.value).lower()
    # No step is kept for it: no line of the steps file, none in the log.
    assert log.steps == []
    steps_file = log.folder / "steps.jsonl"
    assert not steps_file.exists() or steps_file.read_text(encoding="utf-8") == ""


def test_a_command_that_exits_75_after_the_acquire_line_is_an_ordinary_step(tmp_path, monkeypatch):
    # The exit code alone says nothing about the bound: the note does.
    w = _lock_world(tmp_path, LOCK_PREFIX_LIES)
    monkeypatch.setenv("STANDIN_WAIT", "0.25")
    rec, log = _lock_recorder(tmp_path, cap=30, bound=30)
    got = rec.run(w.argv(w.folder / "dump.txt", "exit 75"))
    assert got.exit_code == 75
    step = log.steps[0]
    assert step.exit_code == 75 and step.capped is False
    assert getattr(step, "lock_wait_seconds", None) == 0.25


def test_a_writer_that_waits_for_the_lock_longer_than_the_cap_is_not_stopped_at_the_cap(tmp_path):
    w = _lock_world(tmp_path)
    rec, log = _lock_recorder(tmp_path, cap=2.0, bound=30)
    holder = _hold(w.lock, 5.0, w.folder)
    try:
        ended = rec.start(w.argv(w.lock, "printf 'one\\n'; sleep 0.2")).wait()
    finally:
        _release(holder)
    step = log.steps[0]
    assert step.capped is False and ended.exit_code == 0 and "one" in step.output
    waited = getattr(step, "lock_wait_seconds", None)
    assert isinstance(waited, (int, float)) and waited > 3.0
    assert step.seconds < 2.0


def test_the_cap_ends_the_tool_itself_through_the_prefix_by_its_pid_and_leaves_the_lock_free(tmp_path):
    w = _lock_world(tmp_path)
    pidfile = w.folder / "tool.pid"
    rec, log = _lock_recorder(tmp_path, cap=1.5, bound=30)
    pid = None
    try:
        got = rec.run(w.argv(w.lock, f"echo $$ > '{pidfile}'; exec sleep 120"))
        pid = int(pidfile.read_text(encoding="utf-8"))
        step = log.steps[0]
        assert got.exit_code is None and step.capped is True
        assert 1.0 <= step.seconds < 30
        waited = getattr(step, "lock_wait_seconds", None)
        assert isinstance(waited, (int, float)) and waited < 10, "the step that was stopped still has its wait"
        assert _gone(pid), "the tool itself is gone, not only its prefix"
        assert _lock_is_free(w.lock)
    finally:
        if pid is not None and _alive(pid):
            os.kill(pid, 9)


def test_the_stop_handle_ends_the_tool_itself_through_the_prefix_by_its_pid_and_leaves_the_lock_free(tmp_path):
    w = _lock_world(tmp_path)
    pidfile = w.folder / "tool.pid"
    rec, log = _lock_recorder(tmp_path, cap=600, bound=30)
    handle = rec.start(w.argv(w.lock, f"echo $$ > '{pidfile}'; echo ready; exec sleep 120"))
    pid = None
    try:
        assert handle.readline() == "ready\n"
        pid = int(pidfile.read_text(encoding="utf-8"))
        assert handle.stop() == 15
        ended = handle.wait()
        assert ended.exit_code is not None and ended.exit_code != 0
        step = log.steps[-1]
        waited = getattr(step, "lock_wait_seconds", None)
        assert isinstance(waited, (int, float)) and waited < 20, "the stopped writer still has its wait"
        assert _gone(pid), "the tool itself is gone, not only its prefix"
        assert _lock_is_free(w.lock)
    finally:
        if not handle.done():
            handle.stop()
        if pid is not None and _alive(pid):
            os.kill(pid, 9)


def test_a_lock_wait_that_runs_out_ends_the_pass_with_no_line_for_the_question_and_no_failed_attempt(tmp_path):
    stop_type = _fn("LockWaitStop")
    w = _world(tmp_path)
    lw = _lock_world(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    target = record_target(question)
    marker = lw.folder / "started"
    probe_keywords = []

    class Probing(_fn("SubprocessStarter")):
        """The real starter, except that the copy's python alone is answered by hand."""

        def run(self, argv, timeout=None, **keywords):
            if _is_probe(argv):
                probe_keywords.append(keywords)
                Path(argv[-1]).write_text(f"{w.probe_path}\n{w.probe_sha}\n", encoding="utf-8")
                return _result(0, "", 0.1)
            return super().run(argv, timeout=timeout, **keywords)

    def factory(context):
        # A failed run of this target is on record: a stop that was taken for an error of the
        # reading would become the line of the question, and this test would not see it raised.
        context.failed_runs[target] = _attempt("dolma-tokens-9")
        return Subject("dolma", steps=[lw.argv(lw.lock, f"touch '{marker}'")], recorder=context.runner)

    extra = {"lock_wait_bound": 1.0} if "lock_wait_bound" in {item.name for item in dataclasses.fields(_fn("RunConfig"))} else {}
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": factory}, questions=[question],
                     setups={"dolma": _setup(w)}, **extra)
    holder = _hold(lw.lock, 6.0, lw.folder)
    try:
        with pytest.raises(stop_type):
            _fn("run_pass")(config, Probing())
    finally:
        _release(holder)
    assert not marker.exists()
    # The pass handed the bound and a note to the start of the copy's python too: the real prefix
    # does not start a step that has no note to write to.
    assert len(probe_keywords) == 1
    assert probe_keywords[0].get("lock_wait_bound") == 1.0 and probe_keywords[0].get("lock_note")
    answers = w.run_record / "answers.jsonl"
    assert not answers.exists() or question.id not in answers.read_text(encoding="utf-8")
    attempts = w.run_record / "attempts.jsonl"
    assert not attempts.exists() or attempts.read_text(encoding="utf-8") == ""
    steps = (w.run_record / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(steps) == 1 and "-I" in json.loads(steps[0])["command"], "only the probe was kept as a step"


# --- A set the plant generator wrote: its unparseable lines are no stop of a run ----------------------------

# The generator's own options for a small set (docs/plant.md section 3). "Cases per group" is 3, and
# there are three groups, the first of them lines that are not JSON; every case sits in Z (section 4.1
# and the table of the sanitizer cases in section 4.5).
PLANT_ARGUMENTS = ("--records", "800", "--seed", "7", "--length", "160", "--pairs", "24",
                   "--sanitizer-cases", "3", "--chunks", "20")
UNPARSEABLE_PLANTED = {"sources/Z.jsonl": 3}


@pytest.fixture(scope="module")
def planted_set(tmp_path_factory):
    """A set written by ``fixture/plant/generate.py`` in a folder of its own: its ``sources/`` hold
    the lines the sanitizer cases plant, byte for byte."""
    import sys

    folder = tmp_path_factory.mktemp("planted") / "set"
    done = subprocess.run(
        [sys.executable, "-I", str(REPO / "fixture" / "plant" / "generate.py"), "--out", str(folder),
         *PLANT_ARGUMENTS],
        capture_output=True, text=True, timeout=900, cwd=folder.parent, check=False)
    assert done.returncode == 0, done.stderr
    return folder


def _planted_rows(folder: Path) -> list[dict]:
    text = (folder / "plant_list.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def _is_unparseable(row: dict) -> bool:
    return bool(row["sanitizer"]) and row["sanitizer"]["case"] == "unparseable"


def _planted_unparseable(rows: list[dict]) -> dict[str, int]:
    """The unparseable lines per file, counted from the plant list."""
    found: dict[str, int] = {}
    for row in rows:
        if _is_unparseable(row):
            found[row["file"]] = found.get(row["file"], 0) + 1
    return found


def test_the_id_map_of_a_generated_set_passes_its_unparseable_lines_and_counts_them_per_file(planted_set):
    rows = _planted_rows(planted_set)
    # The plant list and the document of the plant script agree on what the set holds.
    assert _planted_unparseable(rows) == UNPARSEABLE_PLANTED
    id_map = _fn("IdMap")
    try:
        ids, failure = id_map(planted_set), None
    except ValueError as error:
        ids, failure = None, error
    assert failure is None, f"the id map stops on a line of the set: {failure!r}"
    # The four source files of the set are read, once.
    assert ids.files_read == 4
    # Never dropped silently: the count of each file is kept by the reader. A file with none has none.
    counts = getattr(ids, "unparseable_lines", None)
    assert counts is not None, "the id map keeps no count of the lines it could not parse"
    assert dict(counts) == UNPARSEABLE_PLANTED
    # A line that is not JSON holds no record (the repository's own reader of a line takes it for a line
    # with no text); every other line gives its id and its file.
    for row in rows:
        if not row["file"].startswith("sources/"):
            continue
        wanted = None if _is_unparseable(row) else row["file"]
        assert ids.file_of(row["id"]) == wanted, row["id"]
    assert set(ids.mapping().values()) == {"sources/A.jsonl", "sources/B.jsonl", "sources/C.jsonl",
                                           "sources/Z.jsonl"}


class _Counting(Subject):
    """A subject that reports the lines it could not parse, as the subjects of the runner do:
    ``unparseable_lines`` maps the name of a build (``<kind>-<name>`` of its store or set) to the count
    of each of its files."""

    def __init__(self, counts):
        super().__init__("dolma")
        self.unparseable_lines = counts


def test_a_pass_keeps_the_unparseable_line_counts_its_subjects_report_in_the_run_record(tmp_path):
    w = _world(tmp_path)
    question = _make_question("c05.unmutated.E/shard-1", "dolma")
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    counts = {"store-main": dict(UNPARSEABLE_PLANTED), "set-10k": dict(UNPARSEABLE_PLANTED)}
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": lambda context: _Counting(counts)},
                     questions=[question], setups={"dolma": _setup(w)})
    _fn("run_pass")(config, starter)
    kept = w.run_record / "unparseable.jsonl"
    assert kept.is_file(), "the run record holds no file of the unparseable line counts"
    lines = [json.loads(line) for line in kept.read_text(encoding="utf-8").splitlines() if line]
    assert sorted(lines, key=lambda line: line["build"]) == [
        {"build": "set-10k", "file": "sources/Z.jsonl", "lines": 3},
        {"build": "store-main", "file": "sources/Z.jsonl", "lines": 3}]
    assert "  unparseable.jsonl" in (w.run_record / "sha256sums.txt").read_text(encoding="utf-8")


# --- The request files are checked against the builds of the questions that carry the word -----------------

# The listed rows put to OriginBlame in this group of tests: the scratch store with "withdraw A", the
# two writer rows on the same store (they carry no request word), and shard 3 of the main store with
# "withdraw C" (fixture/table/check04.toml and check05.toml, field ``request`` of each row's ``given``).
CARRIERS = ("c04.count-499", "c04.kill-after-100", "c04.retry", "c05.withdraw-C.unmutated.E/shard-3")


def _carrier_questions():
    return [_make_question(listed, "originblame") for listed in CARRIERS]


def _scratch_like_builds(base: Path, *, main_files=("A", "C")) -> dict[str, Path]:
    """The builds of the main store (a record file for each name) and of the scratch store (the one
    file ``sources/A.jsonl``), named as the command names them: the kind of the target, a dash, its name."""
    builds = {}
    for key, names in (("store-main", main_files), ("store-scratch-500", ("A",))):
        folder = base / key
        (folder / "sources").mkdir(parents=True)
        for name in names:
            (folder / "sources" / f"{name}.jsonl").write_text(f'{{"id": "{name}-1"}}\n', encoding="utf-8")
        builds[key] = folder
    return builds


def test_a_request_word_is_checked_only_against_the_builds_of_the_questions_that_carry_it(tmp_path):
    check = _fn("check_request_files")
    _needs_parameter(check, "questions")
    builds = _scratch_like_builds(tmp_path)
    requests = {"withdraw A": ["sources/A.jsonl"], "withdraw C": ["sources/C.jsonl"], "ban X": ["sources/B.jsonl"]}
    # "withdraw C" is carried by a question of the main store alone, and the scratch store holds no file
    # of C. "ban X" names a file no build holds, and no question here carries that word.
    assert check(requests, builds, questions=_carrier_questions()) is None


def test_a_word_whose_question_has_a_build_without_the_file_still_stops_before_any_tool(tmp_path):
    check = _fn("check_request_files")
    stop_type = _fn("PreRunStop")
    _needs_parameter(check, "questions")
    requests = {"withdraw A": ["sources/A.jsonl"], "withdraw C": ["sources/C.jsonl"]}
    # The build of the question that carries "withdraw C" holds no file of C.
    builds = _scratch_like_builds(tmp_path / "no-c", main_files=("A",))
    with pytest.raises(stop_type) as caught:
        check(requests, builds, questions=_carrier_questions())
    assert caught.value.stage == "request-files"
    assert "withdraw C" in caught.value.reason and "sources/C.jsonl" in caught.value.reason
    assert "store-main" in caught.value.reason
    # The build of the question that carries "withdraw A" holds no file of A.
    builds = _scratch_like_builds(tmp_path / "no-a")
    (builds["store-scratch-500"] / "sources" / "A.jsonl").unlink()
    with pytest.raises(stop_type) as caught:
        check(requests, builds, questions=_carrier_questions())
    assert caught.value.stage == "request-files"
    assert "withdraw A" in caught.value.reason and "sources/A.jsonl" in caught.value.reason
    assert "store-scratch-500" in caught.value.reason


def test_a_pass_whose_scratch_build_holds_only_source_A_gets_past_the_request_check_to_its_subject(tmp_path):
    stop_type = _fn("PreRunStop")
    w = _world(tmp_path, name="originblame", program=True)
    builds = _scratch_like_builds(tmp_path / "builds")
    setups = {"originblame": _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha,
                                              ob_binary=w.program)}
    starter = Starter(probe=(w.probe_path, w.probe_sha))

    class Reached(Exception):
        """Raised by the factory: the pass got to build its subject."""

    def factory(context):
        raise Reached()

    config = _config(tmp_path, w, subjects=("originblame",), factories={"originblame": factory},
                     questions=_carrier_questions(), setups=setups, fault_putter=lambda copy, fault_id: {},
                     requests={"withdraw A": ["sources/A.jsonl"], "withdraw C": ["sources/C.jsonl"]},
                     builds=builds)
    try:
        _fn("run_pass")(config, starter)
        outcome = "ended without building its subject"
    except Reached:
        outcome = "built"
    except stop_type as stop:
        outcome = f"stopped at {stop.stage}: {stop.reason}"
    assert outcome == "built", outcome
    assert any(_is_probe(argv) for argv in starter.argvs), "the probe never started"


# 17 Nov 2026: a question whose listed row names ONE layout that the world does not build for its target is left out of the
# pass and named in the run record before the first tool step; it stays not attempted, never cannot answer and never a stop of
# the pass. Box pass 4 on a world with the 10k set in the round-robin layout only stopped at c09.10k.partitioned.kept-shards after
# the drivers had read the round-robin chunk files as the answer to a partitioned question.


def _world_with_layouts():
    return {"shards": [
        {"name": "set/shard-00", "target": ["set", "10k"], "layout": "round-robin"},
        {"name": "E/shard-1", "target": ["store", "main"], "layout": "partitioned"},
        {"name": "E-rr/shard-1", "target": ["store", "rr"], "layout": "round-robin"},
        {"name": "scratch/shard", "target": ["store", "scratch-500"], "layout": None}]}


def _expanded(listed_rows):
    table10 = need("baselines.table10")
    return [q for q in table10.expand(table10.load_table10()) if q.listed_row in listed_rows]


def test_a_question_naming_a_layout_the_world_does_not_build_is_left_out_but_a_routed_row_of_two_layouts_is_read_per_half():
    layout_not_built = _fn("layout_not_built")
    questions = _expanded({"c09.fixture.shards.counts", "c01.E.diff", "c01.E-rr.diff", "c09.10k.partitioned.kept-shards",
                           "c09.10k.round-robin.all-rewrite", "c09.10k.multiset", "c09.10k.successor-shards"})
    assert len(questions) == 7 * 6, "seven rows, six subjects each"
    # The world builds the main store in the round-robin layout only (not the partitioned layout its rows name) and the 10k
    # set in the round-robin layout only.
    world = _world_with_layouts()
    world["shards"][1]["layout"] = "round-robin"
    left_out = layout_not_built(world, questions)
    by_row = {}
    for q in left_out:
        by_row.setdefault(q.listed_row, set()).add(q.subject)
    everyone = {"dolma", "originblame", "manifest", "trace", "dolma+trace", "originblame+trace"}
    # A SINGLE-layout row of a PLAIN driver (c09.fixture.shards.counts and c01.E.diff name partitioned on main), the single-layout
    # row of the 10k set, and a none row of two layouts: left out for every subject.
    assert by_row["c09.fixture.shards.counts"] == everyone and by_row["c01.E.diff"] == everyone
    assert by_row["c09.10k.partitioned.kept-shards"] == everyone and by_row["c09.10k.multiset"] == everyone
    # A routed row of two layouts (c09.10k.successor-shards) is read per half by the subjects that read per layout
    # (the layout with no build reads None, its line follows the combine rule): not left out for them; left out for the
    # subjects with no per-layout reader that have a route for it (the none rows of the others are left out).
    per_half = {q.subject for q in questions if q.listed_row == "c09.10k.successor-shards" and q.route is not None
                and q.subject in {"originblame", "dolma+trace", "originblame+trace", "manifest"}}
    assert per_half and not (by_row.get("c09.10k.successor-shards", set()) & per_half)
    assert {q.subject for q in questions if q.listed_row == "c09.10k.successor-shards" and q.route is not None
            and q.subject not in {"originblame", "dolma+trace", "originblame+trace", "manifest"}} <= by_row["c09.10k.successor-shards"]
    # Rows on a target built in the layout they name are kept: round-robin on the round-robin store rr and on the 10k set.
    assert "c01.E-rr.diff" not in by_row and "c09.10k.round-robin.all-rewrite" not in by_row
    # With both layouts built nothing is left out.
    world = _world_with_layouts()
    world["shards"].append({"name": "set/shard-A", "target": ["set", "10k"], "layout": "partitioned"})
    assert layout_not_built(world, [q for q in questions if q.listed_row not in ("c09.fixture.shards.counts", "c01.E.diff")]) == []


def test_a_build_made_without_tokenization_counts_in_its_layout_from_the_chunk_files_of_the_dolma_entry():
    layouts_built_of = _fn("layouts_built_of")
    world = {"shards": [], "stores": [{"target": ["set", "10k"], "chunk_files": {
        "partitioned/shard-A": "pipeline/partitioned/tokenizer-input/shard-A.jsonl",
        "round-robin/shard-01": "pipeline/round-robin/tokenizer-input/shard-01.jsonl"}}]}
    assert layouts_built_of(world) == {("set", "10k"): {"partitioned", "round-robin"}}
    # The copies of a larger size (a folder ``set-10k``) and a successor (a folder ``successor``) are no layout: found on the box
    # probe pass 6, where the 10k set read as built in "set-10k".
    copies = {"shards": [], "stores": [{"target": ["set", "10k"], "chunk_files": {
        "shard-01": "set-10k/tokenizer-input/shard-01.jsonl", "shard-02": "pipeline/round-robin/tokenizer-input/shard-02.jsonl"}},
        {"target": ["store", "main"], "chunk_files": {"successor of E/shard-3": "successor/tokenizer-input/shard-3.jsonl"}}]}
    assert layouts_built_of(copies) == {("set", "10k"): {"round-robin"}}


def _every_target_world(*, with_partitioned_10k: bool) -> dict:
    """A world of the shape the builder writes, made from the table alone: one Dolma entry for every store and set the questions
    ask (the sets as the builder writes them: chunk files in a folder ``set-<size>`` with ``copy_of`` naming the round-robin file
    each copy was made from), the stores in the layout they are built in, and, with ``with_partitioned_10k``, the 10k set's build
    ADDED in the partitioned layout (W1), which names its layout in its entry."""
    record, table10, runner = need("baselines.record"), need("baselines.table10"), need("baselines.runner")
    targets = sorted({tuple(record.attempt_target(q)) for q in table10.expand(table10.load_table10())})
    store_layout = {"main": "partitioned", "rr": "round-robin", "no-capture": "partitioned"}
    entries = []
    for kind, name in targets:
        if kind == "set":
            shard = "shard-01"
            entries.append({"target": [kind, name], "build": f"/w/{kind}-{name}", "chunk_files": {shard: f"set-{name}/tokenizer-input/{shard}.jsonl"},
                            "copy_of": {shard: f"pipeline/round-robin/tokenizer-input/{shard}.jsonl"}})
        else:
            folder = f"pipeline/{store_layout[name]}/tokenizer-input" if name in store_layout else "scratch"
            entries.append({"target": [kind, name], "build": f"/w/{kind}-{name}", "chunk_files": {"shard": f"{folder}/shard.jsonl"}})
    if with_partitioned_10k:
        entries.append({"target": ["set", "10k"], "layout": "partitioned", "build": "/w/set-10k-partitioned",
                        "chunk_files": {"shard-A": "pipeline/partitioned/tokenizer-input/shard-A.jsonl"}})
    return {"stores": entries, "shards": []}


def test_no_two_builds_sit_under_one_key_and_each_question_resolves_to_exactly_one_build_of_the_layout_it_names():
    """W8 (18 Nov 2026), the RESOLUTION RULE over the WHOLE table (the drivers are wired to it in the next commit; this does not
    drive them): for every question, in every layout it names (or none), the world holds at most
    one Dolma entry, exactly one where the world builds that layout, and the builds the world holds for a target never share a
    key. With the 10k set built in both layouts the rows of the 10k set that name the partitioned layout find exactly one."""
    record, table10, runner = need("baselines.record"), need("baselines.table10"), need("baselines.runner")
    questions = table10.expand(table10.load_table10())
    for with_partitioned in (False, True):
        world = _every_target_world(with_partitioned_10k=with_partitioned)
        runner.check_distinct_builds(world)
        built = runner.layouts_built_of(world)
        checked = 0
        for q in questions:
            target = tuple(record.attempt_target(q))
            named = runner.layouts_named_by(q) or (None,)
            for layout in named:
                found = runner.dolma_entries_for(world, target, layout)
                assert len(found) <= 1, f"{q.id}: {len(found)} builds of {target} in {layout}"
                if layout is None:
                    assert len(found) == 1, f"{q.id}: {target} has no build that its name means"
                elif layout in built.get(target, set()):
                    assert len(found) == 1, f"{q.id}: {target} is built in {layout} and {len(found)} entries are read"
                else:
                    assert not found, f"{q.id}: {target} is not built in {layout} and an entry is read"
                checked += 1
        assert checked >= len(questions)
        partitioned_10k = [q for q in questions if tuple(record.attempt_target(q)) == ("set", "10k")
                           and "partitioned" in (runner.layouts_named_by(q) or ())]
        assert partitioned_10k, "the table has no row of the 10k set in the partitioned layout"
        for q in partitioned_10k:
            assert len(runner.dolma_entries_for(world, ("set", "10k"), "partitioned")) == (1 if with_partitioned else 0), q.id
    # The round-robin build of the 10k set keeps its entry: it is the build the name always meant (ruling A: no other row's
    # target moves), and the added build is not read for a question that names no layout.
    world = _every_target_world(with_partitioned_10k=True)
    assert [e["build"] for e in runner.dolma_entries_for(world, ("set", "10k"), None)] == ["/w/set-10k"]
    assert [e["build"] for e in runner.dolma_entries_for(world, ("set", "10k"), "round-robin")] == ["/w/set-10k"]
    assert [e["build"] for e in runner.dolma_entries_for(world, ("set", "10k"), "partitioned")] == ["/w/set-10k-partitioned"]


def test_a_row_that_names_one_layout_of_a_set_built_in_two_is_graded_against_the_document_counted_on_that_layout(tmp_path):
    """W1 (18 Nov 2026): the 10k set is built in two layouts; ``counted`` holds the set's own (round-robin) document and
    ``counted_by_layout`` the document of each layout. c09.10k.partitioned.kept-shards is derived from the partitioned document
    (the round-robin one would be refused: expected._shard_lists reads the layout the row names), all-rewrite from the other."""
    runner, interfaces, table10 = need("baselines.runner"), need("baselines.interfaces"), need("baselines.table10")
    round_robin = {"shards": {"layout": "round-robin", "keep": [], "rewrite": ["shard-01", "shard-02"]}}
    partitioned = {"shards": {"layout": "partitioned", "keep": ["shard-B"], "rewrite": ["shard-A", "shard-gen"]}}
    (tmp_path / "rr.json").write_text(json.dumps(round_robin), encoding="utf-8")
    (tmp_path / "pt.json").write_text(json.dumps(partitioned), encoding="utf-8")
    world = {"requests": {}, "counted": {"10k": str(tmp_path / "rr.json")}, "index": {},
             "counted_by_layout": {"round-robin": str(tmp_path / "rr.json"), "partitioned": str(tmp_path / "pt.json")}}
    grade_of, _ = runner._graders(world)
    questions = {q.id: q for q in table10.expand(table10.load_table10())}
    kept, every = questions["c09.10k.partitioned.kept-shards|manifest"], questions["c09.10k.round-robin.all-rewrite|manifest"]
    right = interfaces.Reading(kept.id, {"keep": ["shard-B"]}, kept.route)
    assert grade_of(kept, right).status == "right"
    assert grade_of(kept, interfaces.Reading(kept.id, {"keep": ["shard-A"]}, kept.route)).status == "wrong"
    assert grade_of(every, interfaces.Reading(every.id, {"keep": 0, "rewrite": 2}, every.route)).status == "right"
    # A world that holds no document by layout derives the partitioned row from the set's own document: refused (not derived).
    plain = dict(world)
    del plain["counted_by_layout"]
    with pytest.raises(ValueError, match="asked on the partitioned layout"):
        runner._graders(plain)[0](kept, right)


def test_the_questions_of_the_10k_set_in_the_partitioned_layout_are_left_out_only_while_the_world_holds_no_such_build():
    runner, table10, record = need("baselines.runner"), need("baselines.table10"), need("baselines.record")
    questions = [q for q in table10.expand(table10.load_table10()) if tuple(record.attempt_target(q)) == ("set", "10k")]
    named = [q for q in questions if "partitioned" in (runner.layouts_named_by(q) or ())]
    assert named
    plain = _every_target_world(with_partitioned_10k=False)
    plain["shards"] = [{"name": "shard-01", "target": ["set", "10k"], "layout": "round-robin", "build": "/w/set-10k"}]
    both = _every_target_world(with_partitioned_10k=True)
    both["shards"] = plain["shards"] + [{"name": "shard-A", "target": ["set", "10k"], "layout": "partitioned",
                                         "build": "/w/set-10k-partitioned"}]
    left_without = {q.id for q in runner.layout_not_built(plain, questions)}
    left_with = {q.id for q in runner.layout_not_built(both, questions)}
    assert left_without, "no question of the 10k set is left out of a world with the round-robin build only"
    assert not left_with, f"questions of the 10k set are left out of a world with both builds: {sorted(left_with)[:3]}"


def test_a_failed_track_of_the_build_added_in_a_layout_is_not_the_failure_of_the_targets_own_build():
    """W8 (18 Nov 2026), OriginBlame: the 10k set has shards in two layouts. A shard of the partitioned build that failed to track
    is the failed attempt of THAT build (``layout_runs``), never of the round-robin lines; its steps are attributed to
    (kind, name, layout); the round-robin shards that tracked are the target's own run."""
    runner, originblame = need("baselines.runner"), need("baselines.originblame")
    added = runner.added_layouts({"stores": [{"target": ["set", "10k"], "layout": "partitioned", "build": "/w/p"},
                                             {"target": ["set", "10k"], "build": "/w/rr"}]})
    assert added == {(("set", "10k"), "partitioned")}

    def spec(name, layout):
        return SimpleNamespace(name=name, target=("set", "10k"), layout=layout)

    shards = [spec("shard-01", "round-robin"), spec("shard-A", "partitioned"), spec("shard-B", "partitioned")]
    assert [runner.shard_key(s, added) for s in shards] == [("set", "10k"), ("set", "10k", "partitioned"), ("set", "10k", "partitioned")]
    attributing = runner._AttributingRunner(None, None, shards, added)
    assert attributing._target(["ob", "--store", "/x/tracked-0"]) == ("set", "10k")
    assert attributing._target(["ob", "--store", "/x/tracked-1"]) == ("set", "10k", "partitioned")
    bad, later = _attempt("track-A"), _attempt("track-B")
    context = SimpleNamespace(ran_on=set(), failed_runs={}, layout_runs={}, keep_fact=lambda *args, **kwargs: None)
    runner.record_shard_outcomes(context, shards, {1: bad, 2: later}, added)
    assert context.ran_on == {("set", "10k")} and context.failed_runs == {}, "the failed partitioned track became the target's"
    assert context.layout_runs == {("set", "10k", "partitioned"): bad}, "the first failed shard of the added build is its attempt"
    ok = SimpleNamespace(ran_on=set(), failed_runs={}, layout_runs={}, keep_fact=lambda *args, **kwargs: None)
    runner.record_shard_outcomes(ok, shards, {0: bad}, added)
    assert ok.failed_runs == {("set", "10k"): bad} and ok.layout_runs == {("set", "10k", "partitioned"): None}


def test_a_world_with_two_builds_under_one_key_is_refused():
    runner = need("baselines.runner")
    world = _every_target_world(with_partitioned_10k=True)
    runner.check_distinct_builds(world)
    twice = {"stores": world["stores"] + [dict(world["stores"][-1], build="/w/another")], "shards": []}
    with pytest.raises(ValueError, match="two builds of the world sit under the key"):
        runner.check_distinct_builds(twice)
    again = {"stores": world["stores"] + [{"target": ["set", "10k"], "build": "/w/other-round-robin", "chunk_files": {}}], "shards": []}
    with pytest.raises(ValueError, match="two builds of the world sit under the key"):
        runner.check_distinct_builds(again)
    shards = [{"name": "shard-A", "target": ["set", "10k"], "layout": "partitioned", "build": "/w/a", "successor": False},
              {"name": "shard-A", "target": ["set", "10k"], "layout": "round-robin", "build": "/w/b", "successor": False}]
    runner.check_distinct_builds({"stores": [], "shards": shards})
    with pytest.raises(ValueError, match="two shards of the world sit under the key"):
        runner.check_distinct_builds({"stores": [], "shards": shards + [dict(shards[0], build="/w/c")]})


def test_the_questions_left_out_for_a_layout_are_named_in_the_run_record_with_the_layouts_the_world_builds(tmp_path):
    layout_not_built, write = _fn("layout_not_built"), _fn("write_layouts_not_built")
    questions = _expanded({"c09.10k.multiset"})
    world = _world_with_layouts()
    left_out = layout_not_built(world, questions)
    assert len(left_out) == 6
    write(tmp_path, world, left_out)
    body = json.loads((tmp_path / "layouts-not-built.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in body["left_out"]] == [q.id for q in left_out]
    assert {(tuple(item["layouts"]), tuple(item["target"]), tuple(item["layouts_built"])) for item in body["left_out"]} == {
        (("partitioned", "round-robin"), ("set", "10k"), ("round-robin",))}


def test_a_request_file_that_is_no_record_file_of_a_build_stops_a_pass_of_a_subject_other_than_originblame_and_names_the_stop(tmp_path):
    """The request-file check went from OriginBlame alone to every subject (18 Nov 2026: a request file missing from a build
    the world's own entry says holds it is a contradiction in the world). It is a pre-run stop: nothing is started and
    ``stopped.json`` names it. Only the builds of the questions still in the pass are checked."""
    stop_type = _fn("PreRunStop")
    w = _world(tmp_path, name="dolma")
    question = _make_question("c05.unmutated.E/shard-3", "dolma")
    setups = {"dolma": _fn("ToolSetup")(env_dir=w.env, env_record=w.record, identity_sha256=w.identity_sha)}
    starter = Starter(probe=(w.probe_path, w.probe_sha))
    config = _config(tmp_path, w, subjects=("dolma",), factories={"dolma": _never("dolma")}, questions=[question], setups=setups)
    build = _build_with_records(tmp_path / "build-main")
    missing = dataclasses.replace(config, requests={"withdraw A": ["sources/Z.jsonl"]}, builds={"store-main": build})
    with pytest.raises(stop_type) as caught:
        _fn("run_pass")(missing, starter)
    assert caught.value.stage == "request-files" and "sources/Z.jsonl" in caught.value.reason
    assert starter.argvs == [], "nothing was started"
    stopped = json.loads((Path(missing.run_record) / "stopped.json").read_text(encoding="utf-8"))
    assert "request-files" in stopped["error"] or "sources/Z.jsonl" in stopped["error"], stopped
    # A build no question of the pass puts the word to is not checked: the pass goes on to its gate.
    elsewhere = dataclasses.replace(config, requests={"withdraw A": ["sources/Z.jsonl"]}, builds={"store-rr": build})
    with pytest.raises(Exception) as other:
        _fn("run_pass")(dataclasses.replace(elsewhere, run_record=tmp_path / "second-record"), starter)
    assert getattr(other.value, "stage", None) != "request-files"
