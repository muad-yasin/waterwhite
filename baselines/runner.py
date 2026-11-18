"""The runner of Check 10 (plan 5.5 Check 10; table README, section "Check 10": the six
subjects, "What a run hands a subject, and its conditions" and "A failed attempt, the two
routes after it, and when Check 10 is complete").

``python -m baselines`` runs one pass: for each selected subject that uses a tool it makes
the pre-run checks, builds the subject, puts each question of the table to it, grades what it
read, writes the none lines from the one attempt of the subject on each store or set it ran
on, and leaves one folder, the run record. Nothing is run by importing this module.

The order of a pass, and what each part guarantees:

- Before any command is started: every faulted OriginBlame row has a putter, and every
  request file that is handed to OriginBlame is a record file of the build of each store
  (the list the store registers from).
- For each tool: the copy compare of the environment says equal and its ``identity.json`` is
  there; the file's own sha256 is the one the caller handed in; the copy's python ALONE
  starts inside the sandbox through the built route and writes the path it resolves to and
  the sha256 of that binary into a file of the work folder; that sha256 is the one the
  identity file names; for OriginBlame the program's own sha256 is the one the identity file
  names; and for OriginBlame the program's sha256 is also the one supplied for the run
  (``--program-sha256``: an input of the command line, never read from ``identity.json``).
  Only then does the gate open and a tool command start. ``identity.json`` is read and
  never written, and the module holds no hash value.
- Every command is a step with an id: command, exit code, raw output and seconds are kept in
  ``steps.jsonl``, and the files a reading parsed are kept beside it in ``kept/``. A step has
  a cap in seconds; a step past it is stopped by the id of its process and is a failed
  attempt carrying its seconds. Every record line names the ids of the steps it rests on.
- Where the run is given a wait bound (``--lock-wait-bound``), every step starts behind the
  lock prefix, which takes the lock for that one step. The runner makes a note file for the
  step in its own folder of the run record (``locks/<step id>/note.txt``) and sets
  ``WATERWHITE_LOCK_NOTE`` (that file) and ``WATERWHITE_LOCK_WAIT_BOUND`` (seconds) on the
  prefix process. The prefix writes the one line ``acquired <seconds waited>`` to the note
  once it holds the lock. The cap of the step counts from that line and never before it; the
  wait is the step's own field ``lock_wait_seconds`` and is in neither the raw output nor the
  seconds. A note with no such line once the process has ended means the bound ran out: the
  step was never started, ``LockWaitStop`` leaves the pass, no line is written for the
  question and no step is kept.
- A step is kept with the folders of the run replaced by placeholders (``<work>``, an
  environment copy, a build, the interpreter folders); a step that still holds a local path,
  a host or a session name is not written as it is (see ``RecordingRunner``).
- One attempt of a subject on each store or set it ran on is recorded, and the none lines of
  the subject are written from it. A store run twice (the writer stopped, then run again)
  holds both attempts.
- Any of the six subjects may be named, in any order: the drivers are built and run in one
  order (``baselines.subjects``): plain ``dolma``, plain ``originblame``, then ``manifest``,
  ``trace``, ``dolma+trace`` and ``originblame+trace``. A combination reads the runs of its
  plain half and starts no tool of its own; named without its plain half it is a pre-run stop
  (stage ``subjects``). The plain subjects' lines, steps and attempts are the same whether two
  or six subjects are driven. A pass that ends with an error seals its record with
  ``stopped.json`` (the error, the driver and the question) and is never graded; the facts the
  subjects report (each export with its outcome, each half's and each shard's failure) are
  written as they are kept to ``subject-facts.jsonl``.

The folder of a run holds ``answers.jsonl``, ``attempts.jsonl`` (the failed attempts, each
with the ids of the questions that cite it), ``steps.jsonl``, ``kept/<step id>/<file>``, for
each tool ``<tool>/compare.json``, ``<tool>/identity.json`` and ``<tool>/interpreter.txt``
(the two lines the in-sandbox probe wrote, verbatim), and ``sha256sums.txt``, which lists
every other file of the folder with its sha256. The lines of the first three go through the
record writers, which refuse a local path, a host name and a session name; the probe record
is a result of the run and is kept as the probe wrote it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import tomllib
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baselines import (
    combo_dolma,
    companion,
    dolma,
    expected,
    faults,
    ob_driver,
    originblame,
    record,
    sandbox,
    subjects,
    table10,
)
from baselines.interfaces import (
    CANNOT_ANSWER,
    NO_ANSWER_PREFIX,
    SUBJECTS,
    Attempt,
    Grade,
    Question,
    Reading,
    Record,
)

COMPARE_FILE = "compare.json"
IDENTITY_FILE = "identity.json"
STEPS_FILE = "steps.jsonl"
KEPT_FOLDER = "kept"
ANSWERS_FILE = "answers.jsonl"
ATTEMPTS_FILE = "attempts.jsonl"
UNPARSEABLE_FILE = "unparseable.jsonl"
FACTS_FILE = "subject-facts.jsonl"
STOPPED_FILE = "stopped.json"
SEAL_FILE = "sha256sums.txt"
INTERPRETER_FILE = "interpreter.txt"

# The tool environment each subject that uses a tool runs through.
TOOL_OF = {
    "dolma": "dolma",
    "dolma+trace": "dolma",
    "originblame": "originblame",
    "originblame+trace": "originblame",
}

# The listed rows whose answer the table does not make by joining the answers of several
# shards (the subjects raise a ``ValueError`` for them).
NO_JOIN_ROWS = dolma.OPEN_ROWS
NO_JOIN_TEXT = dolma.NO_JOIN_TEXT
NOT_DERIVED_WORDS = "expected value not derived"

# The faults of the table that are put to a file of the tool or by a write, never to a
# companion row; they change nothing here and are no failed attempt.
PUT_ELSEWHERE = ("w.kill-after-100", "w.retry", "l.artifact-digest")

# The signal a stop sends.
STOP_SIGNAL = signal.SIGTERM.value

# What a cut local path is written as.
_LOCAL_PATH = re.compile(r"(?<![\w.])/[^\s\"'`):,]+")
SCRUBBED_NOTE = "(a local path in this step was cut to its last name)"

# The lock prefix: two environment values on its process, one line in the note it writes.
LOCK_NOTE_ENV = "WATERWHITE_LOCK_NOTE"
LOCK_BOUND_ENV = "WATERWHITE_LOCK_WAIT_BOUND"
LOCK_FOLDER = "locks"
LOCK_NOTE_FILE = "note.txt"
_ACQUIRED = re.compile(r"(?m)^acquired (\d+(?:\.\d+)?)[ \t]*$")
LOCK_POLL_SECONDS = 0.05
# A prefix that has neither written its acquire line nor ended this long after its own wait
# bound is not waiting for the lock any more; it is stopped by its id.
LOCK_GIVE_UP_SLACK = 30.0


class LockWaitStop(Exception):
    """The wait for the lock of one step ran out (the note of the step holds no acquire line
    once its process has ended): the step was never started. The pass stops before it, no line
    is written for the question and no step is kept. Not a ``ValueError``: a reader's error
    for a question must not turn it into a line."""


def read_acquire(note) -> float | None:
    """The seconds waited that the acquire line of ``note`` names, or None where the file is
    not there or holds no such line."""
    try:
        text = Path(note).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = _ACQUIRED.search(text)
    return float(found.group(1)) if found else None


def _lock_stop(note) -> LockWaitStop:
    return LockWaitStop(
        f"the wait for the lock ran out: the note {Path(note).parent.name}/{Path(note).name} holds no acquire line, "
        "so the step was never started")


# --- The pre-run condition ---------------------------------------------------------------------


class PreRunStop(Exception):
    """A pre-run check that stops the run before any tool command is started.

    stage: one of ``world-hash``, ``compare``, ``identity``, ``probe``, ``binary``, ``putter``,
        ``request-files``, ``subjects``, ``work-path``.
    reason: what was found, in words (never empty).
    """

    STAGES = ("world-hash", "compare", "identity", "probe", "binary", "putter", "request-files", "subjects",
              "work-path")

    def __init__(self, stage: str, reason: str):
        if stage not in self.STAGES:
            raise ValueError(f"{stage!r} is no pre-run stage: {self.STAGES}")
        if not isinstance(reason, str) or not reason:
            raise ValueError("a stop needs a reason")
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


# The Dolma tokenizer's multiprocessing manager binds a unix socket at
# <work>/dolma-tmp-N/pymp-XXXXXXXX/listener-XXXXXXXX (the sandbox sets its TMPDIR to <work>/dolma-tmp-N;
# multiprocessing adds a folder "pymp-" and a socket "listener-", each with eight random characters). A
# unix socket path may have at most 107 characters on Linux, and the first real pass failed three steps
# with "OSError: AF_UNIX path too long" because its work path had 66. The runner refuses such a work path
# before any tool starts (stage ``work-path``); a test shows the bound against a real manager.
UNIX_SOCKET_PATH_LONGEST = 107
DOLMA_TMP_FOLDER_DIGITS = 4  # the largest step counter the name dolma-tmp-N is made with, as digits
_SOCKET_TAIL = "/dolma-tmp-" + "9" * DOLMA_TMP_FOLDER_DIGITS + "/pymp-" + "x" * 8 + "/listener-" + "x" * 8
WORK_PATH_LONGEST = UNIX_SOCKET_PATH_LONGEST - len(_SOCKET_TAIL)


def check_work_path(work) -> None:
    """Refuse a work folder whose path leaves no room for the socket of the Dolma tokenizer's
    multiprocessing manager (``PreRunStop``, stage ``work-path``), before any tool starts."""
    length = len(str(Path(work).resolve()))
    if length > WORK_PATH_LONGEST:
        raise PreRunStop(
            "work-path",
            f"the work folder path has {length} characters and the longest the tool's temporary socket "
            f"allows is {WORK_PATH_LONGEST} ({UNIX_SOCKET_PATH_LONGEST} for a unix socket path, less "
            f"{len(_SOCKET_TAIL)} for {_SOCKET_TAIL}): the tokenizer would stop with "
            f"AF_UNIX path too long; a shorter work folder is needed")


def check_work_path_of_tools(tools, work) -> None:
    """The work path is checked when the Dolma tool is among the tools the pass runs; a pass of
    OriginBlame alone never starts the manager."""
    if "dolma" in tools:
        check_work_path(work)


class GateClosed(RuntimeError):
    """A command was handed to a starter whose gate is not open."""


class GatedStarter:
    """A starter that starts nothing until its gate is released, except the one command that
    was permitted as the probe (once). ``inner`` is the real starter: ``run(argv,
    timeout=None)`` and ``start(argv)``."""

    def __init__(self, inner):
        self._inner = inner
        self._probes: list[tuple[str, ...]] = []
        self.released = False

    def permit_probe(self, argv: Sequence[str]) -> None:
        self._probes.append(tuple(str(word) for word in argv))

    def release(self) -> None:
        self.released = True

    def _allow(self, argv: Sequence[str]) -> None:
        if self.released:
            return
        key = tuple(str(word) for word in argv)
        if key in self._probes:
            self._probes.remove(key)
            return
        raise GateClosed("the gate of this tool is closed: the pre-run checks did not pass yet")

    def run(self, argv, timeout=None, **keywords):
        self._allow(argv)
        return self._inner.run(list(argv), timeout=timeout, **keywords)

    def start(self, argv, **keywords):
        self._allow(argv)
        return self._inner.start(list(argv), **keywords)


@dataclass(frozen=True)
class PreRun:
    """What a pre-run found: the sha256 of the interpreter the copy's python resolved to, the
    sha256 of the OriginBlame program (None for another tool) and the step of the probe (None
    where no step log was given)."""

    interpreter_sha256: str
    ob_binary_sha256: str | None
    step: Any


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_file(path: Path, stage: str, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PreRunStop(stage, f"{what} cannot be read: {error}") from error


def pre_run(
    subject: str,
    *,
    env_dir,
    env_record,
    identity_sha256: str,
    work,
    lock_prefix,
    starter: GatedStarter,
    run_record,
    ob_binary=None,
    log=None,
    probe_timeout: float | None = None,
    program_sha256: str | None = None,
    lock_wait_bound: float | None = None,
) -> PreRun:
    """The checks of one tool before any command of it starts, in this order (a failed one is
    a ``PreRunStop`` that names its stage): the copy's ``compare.json`` in ``env_record`` says
    equal true; ``identity.json`` is there; its own sha256 is ``identity_sha256``; the copy's
    python alone, inside the sandbox, writes the path it resolves to and the sha256 of that
    binary into a file of the work folder, and that sha256 is the ``interpreter_sha256`` of
    the identity file; for ``originblame`` the program's own sha256 is the one supplied as
    ``program_sha256`` (a value of the caller, never read from the identity file; None where
    the caller supplies none) and the ``ob_binary_sha256`` of the identity file. Only then is
    the gate of ``starter`` released.

    ``lock_wait_bound`` (seconds, None where the prefix needs no note) makes the probe a step
    behind the lock prefix like every other: it is handed a note of its own and the bound, and
    ``LockWaitStop`` is raised where the bound ran out.

    The run record gets ``<subject>/compare.json`` and ``<subject>/identity.json`` (copies,
    byte for byte) and ``<subject>/interpreter.txt`` (the two lines the probe wrote).
    ``identity.json`` is read and never written."""
    env_record = Path(env_record)
    work = Path(work)
    compare_path = env_record / COMPARE_FILE
    if not compare_path.is_file():
        raise PreRunStop("compare", f"the record of the copy holds no {COMPARE_FILE}: it was not compared with its original")
    compare = _read_json_file(compare_path, "compare", COMPARE_FILE)
    if not isinstance(compare, Mapping) or compare.get("equal") is not True:
        found = compare.get("differences") if isinstance(compare, Mapping) else None
        raise PreRunStop("compare", f"the copy is not shown equal to its original in {COMPARE_FILE}"
                                    + (f" ({len(found)} differences)" if isinstance(found, list) else ""))
    identity_path = env_record / IDENTITY_FILE
    if not identity_path.is_file():
        raise PreRunStop("identity", f"the record of the copy holds no {IDENTITY_FILE}")
    identity_bytes = identity_path.read_bytes()
    if hashlib.sha256(identity_bytes).hexdigest() != str(identity_sha256).strip().lower():
        raise PreRunStop("identity", f"the sha256 of {IDENTITY_FILE} is not the one handed in for it")
    try:
        identity = json.loads(identity_bytes.decode("utf-8"))
    except ValueError as error:
        raise PreRunStop("identity", f"{IDENTITY_FILE} is not JSON: {error}") from error
    wanted = identity.get("interpreter_sha256") if isinstance(identity, Mapping) else None
    if not isinstance(wanted, str) or not wanted:
        raise PreRunStop("identity", f"{IDENTITY_FILE} names no interpreter_sha256")

    probe_file = work / f"interpreter-{subject}.txt"
    try:
        home = sandbox.prepare_home(work, f"probe-home-{subject}")
        argv = sandbox.sandboxed_step(
            lock_prefix=lock_prefix, work=work, home=home, chdir=work,
            command=sandbox.interpreter_probe_command(env_dir, probe_file, work=work),
            ro_binds=[*sandbox.interpreter_binds(env_dir), env_dir])
    except ValueError as error:
        raise PreRunStop("probe", f"the probe command cannot be built: {error}") from error
    starter.permit_probe(argv)
    keywords: dict[str, Any] = {}
    step_id = None
    if lock_wait_bound is not None:
        if log is not None:
            step_id = log.reserve()
            note = log.note_for(step_id)
        else:
            note = Path(run_record).absolute() / LOCK_FOLDER / f"probe-{subject}" / LOCK_NOTE_FILE
            note.parent.mkdir(parents=True, exist_ok=True)
        keywords = {"lock_note": str(note), "lock_wait_bound": float(lock_wait_bound)}
    result = starter.run(argv, timeout=probe_timeout, **keywords)
    if keywords and getattr(result, "lock_acquired", None) is False:
        raise _lock_stop(keywords["lock_note"])
    step = None
    if log is not None:
        step = keep_step(log, argv, result.exit_code, result.output, result.seconds, step_id=step_id,
                         lock_wait_seconds=getattr(result, "lock_wait_seconds", None))
    if result.exit_code != 0 or not probe_file.is_file():
        raise PreRunStop("probe", f"the copy's python did not write its record (exit {result.exit_code})")
    written = probe_file.read_text(encoding="utf-8")
    lines = written.splitlines()
    if len(lines) != 2:
        raise PreRunStop("probe", "the probe record does not hold the two lines it should")
    if lines[1] != wanted:
        raise PreRunStop(
            "probe", f"the interpreter the copy's python resolves to inside the sandbox has the sha256 {lines[1]}, "
                     f"and {IDENTITY_FILE} names {wanted}")

    program_sha = None
    if ob_binary is not None or subject == "originblame":
        if ob_binary is None:
            raise PreRunStop("binary", "no program of the tool was handed in")
        named = identity.get("ob_binary_sha256")
        if not isinstance(named, str) or not named:
            raise PreRunStop("binary", f"{IDENTITY_FILE} names no ob_binary_sha256: nothing to compare the program with")
        try:
            program_sha = _sha256_of(Path(ob_binary))
        except OSError as error:
            raise PreRunStop("binary", f"the program cannot be read: {error}") from error
        if program_sha256 is not None:
            supplied = str(program_sha256).strip().lower()
            if program_sha != supplied:
                raise PreRunStop(
                    "binary", f"the program has the sha256 {program_sha}, and the sha256 supplied for this run "
                              f"is {supplied}")
        if program_sha != named:
            raise PreRunStop("binary", f"the program has the sha256 {program_sha}, and {IDENTITY_FILE} names {named}")

    kept = Path(run_record) / subject
    kept.mkdir(parents=True, exist_ok=True)
    (kept / COMPARE_FILE).write_bytes(compare_path.read_bytes())
    (kept / IDENTITY_FILE).write_bytes(identity_bytes)
    (kept / INTERPRETER_FILE).write_text(written, encoding="utf-8")
    starter.release()
    return PreRun(interpreter_sha256=lines[1], ob_binary_sha256=program_sha, step=step)


def _faults_of_originblame(questions: Iterable[Question]) -> list[tuple[Question, str]]:
    """The questions of OriginBlame (alone or with the trace) whose row puts a fault to the
    shard it names: the fault id and the question."""
    found = []
    for question in questions:
        if question.subject not in ("originblame", "originblame+trace") or question.route is None:
            continue
        given = record._listed_givens()[question.listed_row]
        state = given.get("state")
        artifact = given.get("artifact")
        if state in originblame.FAULT_SHARDS and artifact in (None, originblame.FAULT_SHARDS[state]):
            found.append((question, state))
    return found


def check_putters(questions: Iterable[Question], putter) -> None:
    """``PreRunStop`` (stage ``putter``) naming the fault of the first OriginBlame row that
    needs one put to the entry files when no ``putter`` was given. The faults are put by the
    runner on the tool's own files, never through a tool call, so a row with a fault and no
    putter would read the unfaulted store."""
    if putter is not None:
        return
    for question, fault in _faults_of_originblame(questions):
        raise PreRunStop("putter", f"{question.listed_row} needs the fault {fault} put to the entry files, "
                                   "and no putter was given")


def _record_names(build: Path) -> set[str]:
    return {path.relative_to(build).as_posix() for path in companion._record_files(build)}


def _builds_of_words(questions: Iterable[Question], stores: Sequence[str]) -> dict[str, list[str]]:
    """The builds each request word is put to: for every question of any subject that has a route and a request in the ``given``
    of its listed row, the build named ``<kind>-<name>`` of the store or set the question is asked on (18 Nov 2026: this was the
    questions of OriginBlame alone; the check now covers every subject, and only the questions still in the pass count, so a build
    the layout filter left out, or a store no question puts the word to, is not checked). A question of OriginBlame whose build
    is not among ``stores`` makes its word count for every one of them (as before); a question of another subject whose build is
    not among ``stores`` (a store with no chunk file and no shard) checks nothing."""
    found: dict[str, list[str]] = {}
    for question in questions:
        if question.route is None:
            continue
        word = record._listed_givens()[question.listed_row].get("request")
        if word is None:
            continue
        own = "-".join(record.attempt_target(question))
        # A build ADDED beside the target's own is named ``<kind>-<name>-<layout>`` (``builds_of_world``): a question that names a
        # layout is put to the build of that layout where the target holds one, else to the target's own (W8, 18 Nov 2026).
        named = record.layouts_of(question) or ((record.layout_of(question),) if record.layout_of(question) else ())
        layered = [f"{own}-{layout}" for layout in named if f"{own}-{layout}" in stores]
        if layered or own in stores:
            chosen = layered + ([own] if own in stores and (not named or len(layered) < len(named)) else [])
        elif TOOL_OF.get(question.subject) == "originblame":
            chosen = list(stores)
        else:
            chosen = []
        for store in chosen:
            if store not in found.setdefault(word, []):
                found[word].append(store)
    return found


def builds_of_world(world: Mapping[str, Any]) -> dict[str, Any]:
    """The build folder of each store or set the world names, by ``<kind>-<name>``, and of each build ADDED beside a target's
    own (a Dolma entry that names its layout), by ``<kind>-<name>-<layout>``: the key of an added build never takes the key of
    the target's own (W8, 18 Nov 2026: the shards of the 10k set built in two layouts would otherwise overwrite one build with
    the other and the request files of only one would be checked)."""
    added = {entry["build"]: (_target_of(entry["target"]), entry["layout"]) for entry in world.get("stores", ()) if entry.get("layout")}
    found: dict[str, Any] = {}
    for shard in world.get("shards", ()):
        if shard["build"] in added:
            continue
        kind, name = _target_of(shard["target"])
        found[f"{kind}-{name}"] = shard["build"]
    for build, ((kind, name), layout) in added.items():
        found[f"{kind}-{name}-{layout}"] = build
    return found


def check_request_files(requests: Mapping[str, Sequence[str]], builds: Mapping[str, Any],
                        questions: Iterable[Question] | None = None) -> None:
    """``PreRunStop`` (stage ``request-files``) naming the request and the file when a file a
    request hands OriginBlame is no record file of a build the request is put to. A record file
    is one of the list ``baselines.companion`` reads (the list the store registers from), so
    the two cannot drift; a row is never dropped. Without ``questions`` a request is put to
    every build; with them, only to the builds of the questions whose row carries the word
    (the ``request`` of its ``given``), and a word no question carries is checked against no
    build."""
    held = {store: _record_names(Path(build)) for store, build in builds.items()}
    carried = None if questions is None else _builds_of_words(questions, list(held))
    for request, files in requests.items():
        stores = held if carried is None else {store: held[store] for store in carried.get(request, ())}
        for name in files:
            for store, names in stores.items():
                if name not in names:
                    raise PreRunStop(
                        "request-files",
                        f"the request {request!r} names {name!r}, which is no record file of the build of the store {store!r}")


def check_extensions_built(world: Mapping[str, Any], extensions: Collection[str]) -> str | None:
    """A sentence naming the extensions ``extensions`` asks that ``world`` was not built with, or
    None. A world records the extensions its rows were built for in ``extensions`` of world.json; a
    world without that key was built with none. A pass with an extension on against a world built
    without it is refused: the questions of that extension would stay without a line and read as
    questions that were not attempted."""
    built = set(world.get("extensions") or ())
    absent = sorted(set(extensions) - built)
    if not absent:
        return None
    return (f"the pass asks the extension(s) {', '.join(absent)}, which the world was not built with "
            f"(world.json names {sorted(built) or 'none'}): build the world with --extensions")


WORLD_LIST_FILE = "WORLD.sha256"
_WORLD_ROW = re.compile(r"([0-9a-f]{64})  (.+)")
# How many findings the stop names in full.
WORLD_HASH_REPORTED = 8


def world_list_sha256(world_file) -> str | None:
    """The sha256 of the ``WORLD.sha256`` beside ``world_file``, or None where there is none."""
    listed = Path(world_file).resolve().parent / WORLD_LIST_FILE
    return _sha256_of(listed) if listed.is_file() else None


def check_world_hashes(world_file) -> None:
    """``PreRunStop`` (stage ``world-hash``) unless the folder of ``world_file`` is the world the
    builder wrote: the ``WORLD.sha256`` beside the file is there; every row of it names a file that
    is there and has the listed sha256; no path is listed twice; ``world.json`` itself is a row; and
    every path ``world.json`` names has a row (the paths are those the world builder's own listing
    function gives, ``baselines.world._listed_paths``, from the builds the file names). A world
    folder with no list is no built world, so it is refused too. The stop names each file by its
    path relative to the folder."""
    from baselines import world as builder

    folder, own = Path(world_file).resolve().parent, Path(world_file).name
    listed = folder / WORLD_LIST_FILE
    if not listed.is_file():
        raise PreRunStop("world-hash", f"the world folder holds no {WORLD_LIST_FILE}: it is not a built world")
    problems: list[str] = []
    rows: dict[str, str] = {}
    try:
        text = listed.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise PreRunStop("world-hash", f"{WORLD_LIST_FILE} cannot be read: {error}") from error
    for number, line in enumerate(text.split("\n"), start=1):
        if not line:
            continue
        found = _WORLD_ROW.fullmatch(line)
        if found is None:
            problems.append(f"{WORLD_LIST_FILE} line {number} is no row (a sha256, two spaces and a path)")
            continue
        digest, path = found.groups()
        if path in rows:
            problems.append(f"listed twice: {path}")
            continue
        rows[path] = digest
    if own not in rows:
        problems.append(f"no row for {own} in {WORLD_LIST_FILE}")
    for path, digest in rows.items():
        parts = Path(path).parts
        if Path(path).is_absolute() or ".." in parts:
            problems.append(f"outside the world folder: {path}")
        elif not (folder / path).is_file():
            problems.append(f"missing: {path}")
        else:
            try:
                found_digest = _sha256_of(folder / path)
            except OSError as error:
                problems.append(f"cannot be read: {path}: {type(error).__name__}")
                continue
            if found_digest != digest:
                problems.append(f"changed: {path}")
    try:
        document = json.loads((folder / own).read_text(encoding="utf-8"))
        builds_listed = {"-".join([*entry["target"], *([entry["layout"]] if entry.get("layout") else [])]): Path(entry["build"])
                         for entry in document["stores"]}
        try:
            named = builder._listed_paths(document, builds_listed, folder)
        except builder.WorldError:
            # The paths the world file holds are written as the world command spelled its folder:
            # the folder of the file as given, or with its links followed.
            named = builder._listed_paths(document, builds_listed, Path(os.path.abspath(world_file)).parent)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, builder.WorldError) as error:
        problems.append(f"{own} does not give the paths it names: {type(error).__name__}: {error}")
    else:
        problems += [f"no row: {path}" for path in named if path not in rows]
    if problems:
        shown = "; ".join(problems[:WORLD_HASH_REPORTED])
        more = f"; and {len(problems) - WORLD_HASH_REPORTED} more" if len(problems) > WORLD_HASH_REPORTED else ""
        raise PreRunStop("world-hash", f"{WORLD_LIST_FILE} does not say what the world folder holds ({len(problems)} "
                                       f"findings): {shown}{more}")


# --- Steps -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One command the runner started: its id, command, exit code (None where it did not end
    by itself), raw output (standard output and error together), seconds, the files kept with
    it (relative to the run record) and whether it was stopped at the cap. ``lock_wait_seconds``
    is the seconds the step waited for the lock before it started (None for a step with no
    lock prefix); it is in neither the output nor ``seconds``."""

    id: str
    command: tuple[str, ...]
    exit_code: int | None
    output: str
    seconds: float
    kept: tuple[str, ...] = ()
    capped: bool = False
    lock_wait_seconds: float | None = None


class StepLog:
    """The steps of a run: ``steps.jsonl`` and ``kept/`` inside ``folder``. ``roots`` maps a
    folder of the run to the placeholder it is written as. A step whose text still holds a
    local path, a host name or a session name is refused with ``ValueError`` and nothing of it
    is written."""

    def __init__(self, folder, roots: Mapping[str, str] | None = None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._roots = sorted((roots or {}).items(), key=lambda pair: len(pair[0]), reverse=True)
        self.steps: list[Step] = []
        # The shared steps of a store or set, by the driver that logged them (``owner``; None
        # where no pass names one): a line rests on the steps of the drivers it is read by, so a
        # step a later driver logs for a store is a step of that driver's lines only.
        self._shared: dict[tuple[str | None, tuple[str, str]], list[str]] = {}
        self.owner: str | None = None
        self._shared_target: tuple[str, str] | None = None
        self._current: list[str] | None = None
        self._own: dict[str, list[str]] = {}
        self._issued = 0

    def _place(self, text: str) -> str:
        for root, name in self._roots:
            if root:
                text = text.replace(root, name)
        return text

    def reserve(self) -> str:
        """The id the next step takes, held for it: a step that starts now and is kept when it
        ends (a writer) has its note folder under its own id even where another step is kept
        in between. An id that is never used is a gap in the numbering."""
        self._issued = max(self._issued, len(self.steps)) + 1
        return f"step-{self._issued}"

    def note_for(self, step_id: str) -> Path:
        """The note file the lock prefix writes for the step ``step_id``, in the step's own
        folder of the run record; the folder is made, the file is not."""
        note = self.folder.absolute() / LOCK_FOLDER / step_id / LOCK_NOTE_FILE
        note.parent.mkdir(parents=True, exist_ok=True)
        return note

    def add(self, command: Sequence[str], exit_code: int | None, output: str, seconds: float, *,
            kept: Mapping[str, Any] | None = None, capped: bool | None = None,
            step_id: str | None = None, lock_wait_seconds: float | None = None) -> Step:
        """Keep one step and return it. ``kept`` maps a file name to the path of a file the
        reading parsed; each is copied to ``kept/<step id>/<name>``. ``step_id`` is an id
        ``reserve`` gave; ``lock_wait_seconds`` is the wait for the lock, a field of its own."""
        fresh = step_id is None
        if fresh:
            # The id is taken for good only once the step is accepted: a step refused for its
            # text leaves the numbering as it was.
            step_id = f"step-{max(self._issued, len(self.steps)) + 1}"
        names = dict(kept or {})
        for name, source in names.items():
            if not name or "/" in name or name in (".", "..") or not Path(source).is_file():
                raise ValueError(f"{name!r} is no name of a file to keep, or {source} is no file")
        placed = tuple(self._place(str(word)) for word in command)
        text = self._place(output if isinstance(output, str) else "")
        stored = tuple(f"{KEPT_FOLDER}/{step_id}/{name}" for name in names)
        flag = exit_code is None if capped is None else bool(capped)
        line = {"id": step_id, "command": list(placed), "exit_code": exit_code, "output": text,
                "seconds": seconds, "lock_wait_seconds": lock_wait_seconds, "kept": list(stored), "capped": flag}
        record.refuse_private(line, step_id)
        if fresh:
            self._issued = max(self._issued, len(self.steps)) + 1
        for name, source in names.items():
            target = self.folder / KEPT_FOLDER / step_id / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with open(self.folder / STEPS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=True) + "\n")
        step = Step(id=step_id, command=placed, exit_code=exit_code, output=text, seconds=seconds,
                    kept=stored, capped=flag, lock_wait_seconds=lock_wait_seconds)
        self.steps.append(step)
        if self._shared_target is not None:
            self._shared.setdefault((self.owner, self._shared_target), []).append(step_id)
        if self._current is not None:
            self._current.append(step_id)
        return step

    def add_scrubbed(self, command: Sequence[str], exit_code: int | None, output: str, seconds: float, *,
                     kept: Mapping[str, Any] | None = None, capped: bool | None = None,
                     step_id: str | None = None, lock_wait_seconds: float | None = None) -> Step:
        """The same for a step that ``add`` refused for a local path: each absolute path is
        cut to its last name, and the output says so. A step that is refused still (a host or
        a session name) raises ``ValueError``."""
        def cut(text: str) -> str:
            return _LOCAL_PATH.sub(lambda found: found.group(0).rstrip("/").rsplit("/", 1)[-1], self._place(text))

        words = [cut(str(word)) for word in command]
        return self.add(words, exit_code, cut(output if isinstance(output, str) else "") + "\n" + SCRUBBED_NOTE,
                        seconds, kept=kept, capped=capped, step_id=step_id, lock_wait_seconds=lock_wait_seconds)

    @contextlib.contextmanager
    def shared(self, target: Sequence[str]):
        """Steps added inside are the steps every question of ``target`` (a store or set) rests
        on: the one run of a tool on it."""
        before = self._shared_target
        self._shared_target = tuple(target)
        try:
            yield
        finally:
            self._shared_target = before

    @contextlib.contextmanager
    def reading(self, question: Question):
        """Steps added inside are the steps the reading of ``question`` made."""
        before = self._current
        self._current = []
        try:
            yield
        finally:
            self._own[question.id] = list(self._current)
            self._current = before

    def step_ids(self, question: Question) -> tuple[str, ...]:
        """The steps a line of ``question`` rests on: the shared steps of its store or set that
        the drivers of its subject logged (``subjects.owners_of``), then the steps of its own
        reading, in the order they were made."""
        target = tuple(record.attempt_target(question))
        named = record.layouts_of(question) or ((record.layout_of(question),) if record.layout_of(question) else ())
        held: list[str] = []
        for owner in subjects.owners_of(question.subject):
            # A build ADDED beside the target's own is logged under (kind, name, layout) (W8, 18 Nov 2026): a question that names
            # a layout rests on the steps of that build where the target holds one, else on the steps of the target.
            keys = [(*target, layout) if (owner, (*target, layout)) in self._shared else target for layout in named] or [target]
            for key in keys:
                held.extend(step for step in self._shared.get((owner, key), ()) if step not in held)
        return (*held, *self._own.get(question.id, ()))

    def tool_steps(self, subject: str, targets: Iterable[tuple[str, str]]) -> tuple[str, ...]:
        """The shared steps the drivers of ``subject`` logged on ``targets``, in the order they were made: the runs of a
        tool in the pass, which a none line on a store with no chunk file and no shard rests on."""
        found: list[str] = []
        for target in targets:
            for owner in subjects.owners_of(subject):
                found.extend(step for step in self._shared.get((owner, tuple(target)), ()) if step not in found)
        return tuple(sorted(found, key=lambda step: int(step.rsplit("-", 1)[-1])))

    def cite(self, question_id: str) -> None:
        """Add the steps the reading of another question made to the reading in progress: the
        line of a question served from another line rests on the same steps."""
        if self._current is None:
            return
        for step in self._own.get(question_id, ()):
            if step not in self._current:
                self._current.append(step)


def keep_step(log: StepLog, command, exit_code, output, seconds, *, kept=None, capped=None, step_id=None,
              lock_wait_seconds=None) -> Step:
    """Keep a step; one that holds a local path no root placed is kept with the path cut to its
    last name (and the output says so), so that a step is never lost to its own text."""
    try:
        return log.add(command, exit_code, output, seconds, kept=kept, capped=capped, step_id=step_id,
                       lock_wait_seconds=lock_wait_seconds)
    except ValueError:
        pass
    try:
        return log.add_scrubbed(command, exit_code, output, seconds, kept=kept, capped=capped, step_id=step_id,
                                lock_wait_seconds=lock_wait_seconds)
    except ValueError:
        # A host or session name is in the text: the step is kept without its text, and the raw
        # output goes whole to a file of the run record, so that nothing a re-read needs is lost.
        number = step_id.split("-", 1)[1] if step_id else max(log._issued, len(log.steps)) + 1
        raw = log.folder / f"withheld-step-{number}.txt"
        raw.write_text(json.dumps({"command": [str(w) for w in command], "output": output}, indent=1),
                       encoding="utf-8")
        return log.add(["(withheld)"], exit_code, f"the text of this step is in {raw.name}", seconds, capped=capped,
                       step_id=step_id, lock_wait_seconds=lock_wait_seconds)


@dataclass(frozen=True)
class StepResult:
    """What a finished command gave: its exit code (None where it did not end by itself), its
    output, its seconds, the id of its step and, for a step behind the lock prefix, the seconds
    it waited for the lock and whether it acquired it (None for a step with no lock prefix).
    The seconds count from the acquire line, never from the start of the process."""

    exit_code: int | None
    output: str
    seconds: float
    step_id: str | None = None
    lock_wait_seconds: float | None = None
    lock_acquired: bool | None = None


@dataclass(frozen=True)
class Ended:
    """What ``ProcessHandle.wait`` gives: the exit, the output not yet read and the seconds."""

    exit_code: int | None
    output: str
    seconds: float


def _seconds_since_acquire(total: float, waited: float | None) -> float:
    """The seconds of a step whose acquire line was first seen when it had ended: the whole
    time less the wait the note names, where that wait is shorter than the whole time."""
    if waited is not None and waited < total:
        return total - waited
    return total


class ProcessHandle:
    """A started process, stopped by its id. ``process`` has ``pid``, ``stdout`` and ``wait``;
    ``kill(pid, number)`` sends a signal (the group of the process in a real run, which is
    started as its own group, so that the wrappers go with it). ``lock_note`` and
    ``lock_wait_bound`` are those of a process started behind the lock prefix (None otherwise)."""

    def __init__(self, process, kill: Callable[[int, int], Any] = os.kill, signal_number: int = STOP_SIGNAL,
                 lock_note=None, lock_wait_bound: float | None = None):
        self._process = process
        self._kill = kill
        self._number = signal_number
        self._begun = time.monotonic()
        self._ended: Ended | None = None
        self.lock_note = lock_note
        self.lock_wait_bound = lock_wait_bound

    @property
    def pid(self) -> int:
        return self._process.pid

    def readline(self) -> str:
        return self._process.stdout.readline()

    def done(self) -> bool:
        """Whether the process has ended (it is not waited for)."""
        return self._process.poll() is not None

    def stop(self) -> int:
        """Send the signal to the process by its id; the number of the signal sent."""
        with contextlib.suppress(ProcessLookupError):
            self._kill(self._process.pid, self._number)
        return self._number

    def wait(self) -> Ended:
        if self._ended is None:
            rest = self._process.stdout.read()
            code = self._process.wait()
            self._ended = Ended(exit_code=code, output=rest or "", seconds=time.monotonic() - self._begun)
        return self._ended


class RecordedHandle:
    """A started writer whose end is kept as one step when it is waited for; it is stopped at
    the cap if it has not ended by then. A writer behind the lock prefix has its cap counted
    from the acquire line of its note, not from its start; its wait is the step's own field."""

    def __init__(self, handle, runner: "RecordingRunner", argv: Sequence[str], kept=None, step_id: str | None = None):
        self._handle = handle
        self._kept = kept
        self._runner = runner
        self._argv = tuple(str(word) for word in argv)
        self._step_id = step_id
        self._read: list[str] = []
        self._capped = False
        self._begun = time.monotonic()
        self._acquired_at: float | None = None
        self._ended: Ended | None = None
        self.step: Step | None = None
        self._note = getattr(handle, "lock_note", None)
        self._timer: threading.Timer | None = None
        self._watch: threading.Thread | None = None
        self._release = threading.Event()
        if self._note is None:
            self._timer = threading.Timer(runner.cap_seconds, self._expired)
            self._timer.daemon = True
            self._timer.start()
        else:
            self._watch = threading.Thread(target=self._watching, daemon=True)
            self._watch.start()

    def _expired(self) -> None:
        self._capped = True
        with contextlib.suppress(Exception):
            self._handle.stop()

    def _watching(self) -> None:
        """Wait for the acquire line, then count the cap from it. A prefix that wrote none
        and has not ended long after its own bound is stopped by its id."""
        bound = getattr(self._handle, "lock_wait_bound", None)
        give_up = self._begun + bound + LOCK_GIVE_UP_SLACK if bound is not None else None
        while not self._release.is_set():
            if read_acquire(self._note) is not None:
                self._acquired_at = time.monotonic()
                if not self._release.wait(self._runner.cap_seconds):
                    self._expired()
                return
            if self._handle.done():
                return
            if give_up is not None and time.monotonic() > give_up:
                with contextlib.suppress(Exception):
                    self._handle.stop()
                return
            self._release.wait(LOCK_POLL_SECONDS)

    @property
    def pid(self):
        return self._handle.pid

    def readline(self) -> str:
        line = self._handle.readline()
        self._read.append(line)
        return line

    def done(self) -> bool:
        return self._handle.done()

    def stop(self) -> int:
        return self._handle.stop()

    def wait(self):
        if self._ended is not None:
            return self._ended
        ended = self._handle.wait()
        self._release.set()
        if self._timer is not None:
            self._timer.cancel()
        if self._watch is not None:
            self._watch.join(timeout=5.0)
        now = time.monotonic()
        total = now - self._begun
        waited = None
        if self._note is not None:
            waited = read_acquire(self._note)
            if waited is None:
                # The note holds no acquire line: the wait bound ran out (or the process was
                # stopped before it held the lock) and nothing was started. A stop sent after the
                # process had ended changes nothing about that.
                raise _lock_stop(self._note)
            if self._acquired_at is not None:
                seconds = now - self._acquired_at
            else:
                seconds = _seconds_since_acquire(total, waited)
        else:
            seconds = total
        text = "".join(self._read) + (ended.output or "")
        if self._capped:
            text += f"\nthe step was stopped at the cap of {self._runner.cap_seconds} seconds after {seconds} seconds"
        held = self._kept() if callable(self._kept) else self._kept
        self.step = keep_step(self._runner.log, self._argv, ended.exit_code, text, seconds, kept=held,
                              capped=self._capped, step_id=self._step_id, lock_wait_seconds=waited)
        self._ended = ended
        return ended


class RecordingRunner:
    """Starts every command through ``starter`` and keeps it as a step of ``log``. Every step
    has a cap in seconds (a positive number): ``run`` hands it to the starter as the timeout,
    which stops the process by its id when it is reached; the step then has no exit code, and
    its output says the cap and the seconds.

    ``lock_wait_bound`` (seconds, a positive number, or None where the prefix of the commands
    takes no note) makes every step a step behind the lock prefix: the starter is handed a
    note file in the step's own folder of the run record (``lock_note``) and the bound
    (``lock_wait_bound``). The starter counts the cap from the acquire line of that note and
    gives the wait; a starter that says it did not acquire (the bound ran out) makes
    ``LockWaitStop``: no step is kept for it."""

    def __init__(self, starter, log: StepLog, *, cap_seconds: float, lock_wait_bound: float | None = None):
        if isinstance(cap_seconds, bool) or not isinstance(cap_seconds, (int, float)) or not cap_seconds > 0:
            raise ValueError(f"a step needs a cap in seconds that is a positive number, got {cap_seconds!r}")
        if lock_wait_bound is not None and (
                isinstance(lock_wait_bound, bool) or not isinstance(lock_wait_bound, (int, float))
                or not lock_wait_bound > 0):
            raise ValueError(f"a wait bound is a positive number of seconds, got {lock_wait_bound!r}")
        self._starter = starter
        self.log = log
        self.cap_seconds = float(cap_seconds)
        self.lock_wait_bound = None if lock_wait_bound is None else float(lock_wait_bound)

    def _lock_keywords(self) -> tuple[dict[str, Any], str | None]:
        """The note of the step about to start and the bound, as keywords of the starter, and
        the id the step will take. Nothing for a runner with no wait bound."""
        if self.lock_wait_bound is None:
            return {}, None
        step_id = self.log.reserve()
        return {"lock_note": str(self.log.note_for(step_id)), "lock_wait_bound": self.lock_wait_bound}, step_id

    def run(self, argv: Sequence[str], timeout=None, *, kept=None) -> StepResult:
        keywords, step_id = self._lock_keywords()
        result = self._starter.run(list(argv), timeout=self.cap_seconds, **keywords)
        if keywords and getattr(result, "lock_acquired", None) is False:
            raise _lock_stop(keywords["lock_note"])
        output = result.output or ""
        if result.exit_code is None:
            output += f"\nthe step was stopped at the cap of {self.cap_seconds} seconds after {result.seconds} seconds"
        # ``kept`` is a mapping of the files a reading parses, or a function that gives it once
        # the command has ended (the files are made by it).
        held = kept() if callable(kept) else kept
        step = keep_step(self.log, argv, result.exit_code, output, result.seconds, kept=held, step_id=step_id,
                         lock_wait_seconds=getattr(result, "lock_wait_seconds", None))
        return StepResult(exit_code=result.exit_code, output=output, seconds=result.seconds, step_id=step.id,
                          lock_wait_seconds=step.lock_wait_seconds,
                          lock_acquired=getattr(result, "lock_acquired", None))

    def start(self, argv: Sequence[str], *, kept=None) -> RecordedHandle:
        """Start a writer; ``kept`` (files, or a function that gives them) is read when its end
        is waited for, which is when the files it wrote are there."""
        keywords, step_id = self._lock_keywords()
        return RecordedHandle(self._starter.start(list(argv), **keywords), self, argv, kept, step_id=step_id)


def _lock_environment(note, bound: float | None) -> dict[str, str]:
    """The environment of a process started behind the lock prefix: the machine's own, and the
    two values the prefix reads."""
    environment = dict(os.environ)
    environment[LOCK_NOTE_ENV] = str(note)
    if bound is not None:
        environment[LOCK_BOUND_ENV] = repr(float(bound))
    return environment


class SubprocessStarter:
    """The real starter. A process is started as the leader of its own process group, so that
    the lock command, the sandbox and the tool end together when it is stopped; it is stopped
    by the id of that group, which is the id of the process started. Nothing is ever stopped
    by a name or a pattern.

    Given a ``lock_note`` (and the ``lock_wait_bound``), the process is started with the two
    environment values of the lock prefix and the cap (``timeout``) counts from the moment the
    note holds its acquire line; ``lock_wait_seconds`` is the number of that line and
    ``seconds`` count from the acquire line too. A note with no acquire line when the process
    has ended gives ``lock_acquired`` False: the step was not started."""

    def run(self, argv, timeout=None, lock_note=None, lock_wait_bound=None):
        if lock_note is not None:
            return self._run_behind_lock(argv, timeout, lock_note, lock_wait_bound)
        begun = time.monotonic()
        process = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   start_new_session=True)
        try:
            output, _ = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
            code = None
        return StepResult(exit_code=code, output=output or "", seconds=time.monotonic() - begun)

    def _run_behind_lock(self, argv, timeout, lock_note, lock_wait_bound):
        begun = time.monotonic()
        process = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   start_new_session=True, env=_lock_environment(lock_note, lock_wait_bound))
        give_up = begun + lock_wait_bound + LOCK_GIVE_UP_SLACK if lock_wait_bound is not None else None
        acquired_at: float | None = None
        waited: float | None = None
        killed = False
        while True:
            try:
                output, _ = process.communicate(timeout=LOCK_POLL_SECONDS)
                code = process.returncode
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if acquired_at is None:
                waited = read_acquire(lock_note)
                if waited is not None:
                    acquired_at = now
            if acquired_at is not None:
                over = timeout is not None and now - acquired_at >= timeout
            else:
                over = give_up is not None and now > give_up
            if over:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
                code = None
                killed = True
                break
        end = time.monotonic()
        if waited is None:
            waited = read_acquire(lock_note)
        if acquired_at is not None:
            seconds = end - acquired_at
        elif killed:
            seconds = end - begun
        else:
            seconds = _seconds_since_acquire(end - begun, waited)
        return StepResult(exit_code=code, output=output or "", seconds=seconds, lock_wait_seconds=waited,
                          lock_acquired=waited is not None)

    def start(self, argv, lock_note=None, lock_wait_bound=None):
        process = subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                                   start_new_session=True,
                                   env=None if lock_note is None else _lock_environment(lock_note, lock_wait_bound))
        return ProcessHandle(process, kill=os.killpg, lock_note=lock_note, lock_wait_bound=lock_wait_bound)


# The class itself, kept under a name no test replaces: the work path guard applies to a pass started by
# exactly this class (a stand-in starter, or a subclass of one, never binds the tokenizer's socket).
_REAL_STARTER = SubprocessStarter


def attempt_of(step: Step, attempt_id: str) -> Attempt:
    """The failed attempt a step is: its command, exit code and output."""
    return Attempt(id=attempt_id, command=tuple(step.command), exit_code=step.exit_code, output=step.output)


# --- The stop and the second run, as the subjects read them -------------------------------------------


@dataclass(frozen=True)
class StopRecord:
    """The end of a stopped writer, under the names ``DolmaRun`` and ``StoppedWriter`` read
    (both subjects take the same six)."""

    stopped_exit_code: int | None
    stopped_signal: int
    stopped_attempt_id: str
    stopped_command: tuple[str, ...]
    stopped_output: str
    rows_showed_while_running: bool


def stop_record(*, command: Sequence[str], sent: int, ended, attempt_id: str,
                rows_showed_while_running: bool, output_before: str = "") -> StopRecord:
    return StopRecord(
        stopped_exit_code=ended.exit_code, stopped_signal=sent, stopped_attempt_id=attempt_id,
        stopped_command=tuple(str(word) for word in command),
        stopped_output=output_before + (ended.output or ""),
        rows_showed_while_running=rows_showed_while_running)


@dataclass(frozen=True)
class RetryRecord:
    """The second run of the tokenizer, under the names ``DolmaRun`` reads."""

    retry_attempt_id: str
    retry_command: tuple[str, ...]
    retry_exit_code: int | None
    retry_output: str


def retry_record(*, command: Sequence[str], ended, attempt_id: str) -> RetryRecord:
    return RetryRecord(retry_attempt_id=attempt_id, retry_command=tuple(str(word) for word in command),
                       retry_exit_code=ended.exit_code, retry_output=ended.output or "")


# --- Faults put to the tool's own files ---------------------------------------------------------------


def _read_companion_text(path: Path) -> tuple[list[dict], bool]:
    packed = path.read_bytes()[:2] == b"\x1f\x8b"
    rows = [{"start": r.start, "end": r.end, "id": r.id, "src": r.src, "loc": r.loc}
            for r in companion.read_rows(path)]
    return rows, packed


def put_dolma_fault(fault_id: str, companion_file, chunk_file: str) -> bool:
    """Put a fault of the table to the companion rows of a chunk file, in the file's own
    format (gzip stays gzip, a plain file stays plain). True where it was put. False, and
    nothing changed, for the three faults that are put elsewhere (the stopped write, the second
    run and the changed file). ``baselines.faults.FaultError`` for a fault the table does not
    hold. The faulted file is a copy the caller made: the rows the run holds are never
    changed."""
    if fault_id in PUT_ELSEWHERE:
        return False
    path = Path(companion_file)
    rows, packed = _read_companion_text(path)
    changed = faults.apply_fault(fault_id, "dolma", rows, chunk_file)
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    for row in changed:
        writer.writerow([row["start"], row["end"], row["id"], row["src"], row["loc"]])
    data = out.getvalue().encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if packed else data)
    return True


ENTRY_FOLDER = originblame.ENTRY_FOLDER
BINARY_INDEX = "token-index.gpt2.bin"


def binary_index_state(store) -> dict:
    """Whether the binary index of a store (``.ob/token-index.gpt2.bin``) exists and is fresh,
    that is, not older than any entry file. The tool reads the binary index when it is fresh
    and the entry files otherwise."""
    base = Path(store) / ENTRY_FOLDER[0]
    binary = base / BINARY_INDEX
    if not binary.is_file():
        return {"existed": False, "fresh": False}
    folder = base / ENTRY_FOLDER[1]
    entry_files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
    stamp = binary.stat().st_mtime_ns
    return {"existed": True, "fresh": all(stamp >= p.stat().st_mtime_ns for p in entry_files)}


def ob_fault_putter(copy, fault_id: str) -> dict:
    """Put a fault of the table to the entry files of a store copy, with the faults module on
    the entries (never through a call of the tool): ``f.map-deleted`` empties every entry file
    and leaves the section records, ``f.digest`` changes the first digit of ``token_count`` of
    the first entry, ``l.count-499`` deletes the last entry. Each entry stays one JSON line in
    the file it was in; a line the fault does not change is kept as it was. The result says
    whether the binary index existed and was fresh before the fault and after it. ``ValueError``
    where the fault cannot be put."""
    folder = Path(copy).joinpath(*ENTRY_FOLDER)
    before = binary_index_state(copy)
    files: list[tuple[int, Path]] = []
    if folder.is_dir():
        for path in folder.iterdir():
            if not path.name.isdigit():
                raise ValueError(f"{path.name!r} in the entry folder is not the name of an entry file")
            files.append((int(path.name), path))
    files.sort()
    held: list[tuple[Path, str, dict]] = []
    for _, path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                held.append((path, line, json.loads(line)))
    try:
        changed = faults.apply_fault(fault_id, "originblame", {"entries": [entry for _, _, entry in held]})["entries"]
    except faults.FaultError as error:
        raise ValueError(str(error)) from error
    lines: dict[Path, list[str]] = {path: [] for _, path in files}
    for place, (path, text, entry) in enumerate(held):
        if place < len(changed):
            lines[path].append(text if changed[place] == entry else json.dumps(changed[place]))
    touched = []
    for _, path in files:
        body = "".join(line + "\n" for line in lines[path])
        if body != path.read_text(encoding="utf-8"):
            path.write_text(body, encoding="utf-8")
            touched.append(f"{ENTRY_FOLDER[0]}/{ENTRY_FOLDER[1]}/{path.name}")
    return {"fault": fault_id, "files": touched,
            "binary_index": {"before": before, "after": binary_index_state(copy)}}


# --- What a step needs on disk before it starts --------------------------------------------------------


def prepare_tokenizer_folders(build, *, work_dir, tmpdir) -> None:
    """Make the tokenizer's work folders ``<work_dir>/in`` and ``<work_dir>/out`` inside the
    build, and the folder for its temporary files. ``ValueError`` for a work folder that is
    not relative or leaves the build."""
    for relative in dolma.tokens_work_folders(work_dir):
        (Path(build) / relative).mkdir(parents=True, exist_ok=True)
    Path(tmpdir).mkdir(parents=True, exist_ok=True)


def prepare_mixer_folders(build, *, work_dir) -> None:
    """Make the mixer's work folders ``<work_dir>/in`` and ``<work_dir>/out`` inside the build
    (the toolkit makes a temporary folder only for a work folder it is not told)."""
    for relative in dolma.tokens_work_folders(work_dir):
        (Path(build) / relative).mkdir(parents=True, exist_ok=True)


def place_tokenizer(build, source, *, name: str) -> str:
    """Copy the tokenizer file into the build as ``name`` and return the name, relative to the
    build, as the command is given it."""
    if not name or os.path.isabs(name) or ".." in Path(name).parts:
        raise ValueError(f"the tokenizer is placed under a relative name inside the build, got {name!r}")
    target = Path(build) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return name


def rows_after_stop(folder) -> list[companion.Row]:
    """The companion rows a stop left, read as the rows that came out whole (a torn file is
    read to where it was torn)."""
    return dolma.read_stopped_rows(folder)


def sample_row_count(document: Mapping, segment: str, step: int, sample_pos: int) -> int:
    """The number of rows of one sample of a trace export, read from the sample itself (the
    history rows carry no sample position). ``ValueError`` where the export holds no such
    segment, step or sample."""
    for held in document.get("segments", ()):
        if held.get("segment") != segment:
            continue
        for traced in held.get("steps", ()):
            if traced.get("step") != step:
                continue
            for sample in traced.get("samples", ()):
                if sample.get("sample_pos") == sample_pos:
                    return len(sample.get("rows", ()))
    raise ValueError(f"the export holds no sample {sample_pos} of step {step} of {segment}")


class IdMap:
    """The file that holds each record id of a build, made once from the record files the
    store registers from (``companion.resolve_id`` scans every file for every id). ``file_of``
    gives what ``resolve_id`` gives: the relative path, None for an id no record file holds,
    and a ``ValueError`` for an id two files hold. A line that is not JSON holds no record
    (the line reader of the tool, ``waterwhite.capture.line_text``, gives such a line no text):
    it adds no id, and it is counted: ``unparseable_lines`` maps each file that has such lines
    to their number."""

    def __init__(self, build):
        self.build = Path(build)
        self.files_read = 0
        self.unparseable_lines: dict[str, int] = {}
        self._holders: dict[str, list[str]] = {}
        for path in companion._record_files(self.build):
            relative = path.relative_to(self.build).as_posix()
            self.files_read += 1
            with open(path, "rb") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        held = json.loads(line.decode("utf-8"))
                    except (ValueError, RecursionError):
                        self.unparseable_lines[relative] = self.unparseable_lines.get(relative, 0) + 1
                        continue
                    if not isinstance(held, dict):
                        raise ValueError(f"{path.name} line {number}: a record is not an object")
                    identifier = held.get("id")
                    if identifier is not None and relative not in self._holders.setdefault(identifier, []):
                        self._holders[identifier].append(relative)

    def mapping(self) -> dict[str, str]:
        """The file of every id, as ``baselines.originblame.track_plan`` takes it. An id two
        files hold is a ``ValueError``."""
        return {identifier: self.file_of(identifier) for identifier in self._holders}

    def file_of(self, document_id: str) -> str | None:
        if not document_id:
            return None
        holders = self._holders.get(document_id, [])
        if len(holders) > 1:
            raise ValueError(f"id {document_id!r} is held by {len(holders)} files: {holders}")
        return holders[0] if holders else None


# --- Answering one question ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Answered:
    """The result of one question: its line (None where the question stays not attempted), the
    failed attempt its line cites (else None) and the ids of the steps it rests on."""

    line: dict | None
    attempt: Attempt | None
    step_ids: tuple[str, ...]
    outcome: Any = None


# What the subjects raise for a store or set the tool did not run on: no line is written.
NOT_ATTEMPTED_ERRORS = (originblame.NotAttempted, combo_dolma.NotAttempted)


def _plain_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _grade_or_not_derived(make_grade: Callable[[], Grade]) -> Grade:
    """The grade, or cannot answer with the words of the expected value that could not be
    derived: the question keeps its line."""
    try:
        return make_grade()
    except expected.PartMissing as error:
        return Grade(status=CANNOT_ANSWER, evidence=f"{NOT_DERIVED_WORDS}: {error}")


def _failed_run_of(question: Question, target: tuple[str, str], failed_runs, layout_runs) -> Attempt | None:
    """The failed run a reading that raised ``ValueError`` rests on: where the target holds a build ADDED beside its own in the
    layout the question names (``layout_runs``), that build's failed attempt, or None where it ran (the error is then the
    caller's: the target's own run, which may have failed, is not the run this question read); else the failed run of the target."""
    layout = record.layout_of(question)
    if layout is not None and layout_runs and (*target, layout) in layout_runs:
        return layout_runs[(*target, layout)]
    return failed_runs.get(target) if failed_runs else None


def answer_question(
    question: Question,
    subject,
    *,
    commit: str,
    log: StepLog,
    grade_of: Callable[[Question, Reading], Grade],
    grade_layouts_of: Callable[[Question, Mapping], Grade] | None = None,
    failed_runs: Mapping[tuple[str, str], Attempt] | None = None,
    notes: Mapping[str, Any] | Callable[[], Mapping[str, Any] | None] | None = None,
    twin_of: Callable[[Question], str | None] | None = None,
    layout_runs: Mapping[tuple, Attempt | None] | None = None,
) -> Answered:
    """Put ``question`` to ``subject`` and give its line.

    A reading is graded by ``grade_of``; a row that names two layouts is read by the
    subject's ``read_by_layout`` and graded by ``grade_layouts_of``. A failed attempt is
    the line ``no answer: failed attempt <id>``. A reader that raises ``ValueError`` still
    gives a line: for a row the table gives no join for, cannot answer with the table's
    words; for another row after a failed tokenizer run of its store or set (``failed_runs``),
    the status of that attempt; otherwise the error is the caller's. An expected value that
    could not be derived (``expected.PartMissing``) is cannot answer with its words, never a
    lost line. A question the tool did not run on gives no line. ``notes`` (a mapping, or a
    function that gives one once the reading is done) is carried on the line."""
    if question.route is None:
        raise ValueError(f"{question.id} holds no record: its line is a none line")
    layouts = record.layouts_of(question)
    by_layout = layouts is not None and hasattr(subject, "read_by_layout")
    if by_layout and grade_layouts_of is None:
        raise ValueError(f"{question.listed_row} names two layouts: a grader of both is needed")
    target = tuple(record.attempt_target(question))
    failed_here: Attempt | None = None
    outcome: Any = None
    cannot_text: str | None = None
    skipped = False
    with log.reading(question):
        try:
            outcome = subject.read_by_layout(question) if by_layout else subject.read(question)
        except NOT_ATTEMPTED_ERRORS:
            skipped = True
        except ValueError:
            if question.listed_row in NO_JOIN_ROWS:
                cannot_text = NO_JOIN_TEXT
            elif _failed_run_of(question, target, failed_runs, layout_runs) is not None:
                failed_here = _failed_run_of(question, target, failed_runs, layout_runs)
            else:
                raise
        else:
            # A reading the plain line served rests on the steps of that line.
            twin = twin_of(question) if twin_of is not None else None
            served = getattr(subject, "was_served", None)
            if twin is not None and served is not None and served(question):
                log.cite(twin)
    ids = tuple(log.step_ids(question))
    if skipped:
        return Answered(None, None, ids)
    used = False
    tied = getattr(subject, "used_tie", None)
    if tied is not None and cannot_text is None and failed_here is None:
        used = bool(tied(question))
    attempt: Attempt | None = None
    if failed_here is not None:
        rec = record.record_of(question, failed_here)
        attempt = failed_here
    elif cannot_text is not None:
        rec = record.record_of(question, Reading(question.id, None, question.route),
                               Grade(status=CANNOT_ANSWER, evidence=cannot_text))
    elif by_layout:
        grade = _grade_or_not_derived(lambda: grade_layouts_of(question, outcome))
        if grade.status == record.NOT_ATTEMPTED:
            return Answered(None, None, ids)
        attempt_id = grade.status[len(NO_ANSWER_PREFIX):] if grade.status.startswith(NO_ANSWER_PREFIX) else None
        if attempt_id is not None:
            attempt = next((o for o in outcome.values() if isinstance(o, Attempt) and o.id == attempt_id), None)
        marks = tuple(question.marks)
        if used and question.tie_mark is not None and question.tie_mark not in marks:
            marks += (question.tie_mark,)
        raw = {name: (found.raw if isinstance(found, Reading) else None) for name, found in outcome.items()}
        rec = Record(question_id=question.id, status=grade.status, raw=raw, evidence=grade.evidence,
                     route=question.route, marks=marks, excluded_from_tally=question.excluded_from_tally,
                     attempt_id=attempt_id)
    elif isinstance(outcome, Attempt):
        rec = record.record_of(question, outcome)
        attempt = outcome
    else:
        grade = _grade_or_not_derived(lambda: grade_of(question, outcome))
        rec = record.record_of(question, outcome, grade, used_tie=used)
    line = record.answer_line(question, rec, commit=commit, step_ids=ids)
    held = notes() if callable(notes) else notes
    if held:
        line["notes"] = _plain_json(held)
    return Answered(line, attempt, ids, outcome)


def none_lines(
    questions: Iterable[Question],
    attempts: Sequence[record.StoreAttempt],
    *,
    commit: str,
    step_ids_of: Callable[[Question], Sequence[str]] | None = None,
) -> list[dict]:
    """The lines of the questions whose record is none, in question order, from the attempts
    of their subjects on their stores and sets: cannot answer (with the table's text of what
    the subject lacks) after an attempt that ran, ``no answer: failed attempt <id>`` after one
    that failed, and no line where the subject made none (the question stays not attempted)."""
    lines = []
    for question in questions:
        if question.route is not None:
            continue
        status = record.none_line_status(question, attempts)
        if status == record.NOT_ATTEMPTED:
            continue
        failed = status[len(NO_ANSWER_PREFIX):] if status.startswith(NO_ANSWER_PREFIX) else None
        rec = Record(question_id=question.id, status=status, raw=None,
                     evidence=question.missing if failed is None else None, route=None,
                     marks=tuple(question.marks), excluded_from_tally=question.excluded_from_tally,
                     attempt_id=failed)
        ids = list(step_ids_of(question)) if step_ids_of is not None else []
        lines.append(record.answer_line(question, rec, commit=commit, step_ids=ids))
    return lines


# --- One pass --------------------------------------------------------------------------------------------


@dataclass
class ToolSetup:
    """What a tool needs before its gate opens: its copied environment, the record of that copy
    (``compare.json`` and ``identity.json``), the sha256 of ``identity.json`` handed in by the
    caller, and for OriginBlame the program and ``program_sha256``, the sha256 the caller
    supplies for that program (from the command line, never from ``identity.json``; None
    where none is supplied)."""

    env_dir: Any
    env_record: Any
    identity_sha256: str
    ob_binary: Any = None
    program_sha256: str | None = None


@dataclass
class RunConfig:
    """One pass: the commit, the work folder, the run record folder, the lock prefix, the
    subjects, the cap of a step in seconds, the setup of each tool, a factory of each subject,
    the questions, the graders, the attempts made elsewhere (of the subjects that use no
    tool), the putter of OriginBlame's faults, and the request files and builds to check.
    ``lock_wait_bound`` is the seconds a step may wait for the lock behind the lock prefix
    (None where the prefix takes no note). ``world_file`` is the ``world.json`` the run was put
    to: where it is given, the pass checks the hashes of the world folder against the
    ``WORLD.sha256`` beside it before anything else (``check_world_hashes``)."""

    commit: str
    work: Any
    run_record: Any
    lock_prefix: Sequence[str]
    subjects: Sequence[str]
    step_cap_seconds: float
    setups: Mapping[str, ToolSetup]
    factories: Mapping[str, Callable[["SubjectContext"], Any]]
    questions: Sequence[Question]
    grade_of: Callable[[Question, Reading], Grade]
    grade_layouts_of: Callable[[Question, Mapping], Grade] | None = None
    attempts: Sequence[record.StoreAttempt] = ()
    fault_putter: Callable[[Any, str], Any] | None = None
    requests: Mapping[str, Sequence[str]] | None = None
    builds: Mapping[str, Any] | None = None
    lock_wait_bound: float | None = None
    world_file: Any = None
    shardless: frozenset = frozenset()


@dataclass
class SubjectContext:
    """What a factory is given once the gate of its tool is open.

    runner: a ``RecordingRunner`` that starts every command through the gate.
    log: the ``StepLog``; ``log.shared(target)`` around a run of the tool on a store or set.
    work, run_record: the folders of the run.
    ran_on: the targets the subject ran on (a factory adds each).
    failed_runs: for each target whose run failed, the failed ``Attempt`` of that run.
    layout_runs: for a build ADDED beside the target's own (an entry that names its layout), under (kind, name, layout): None
    where its run ended, the failed ``Attempt`` where it failed. ``ran_on`` and ``failed_runs`` keep two-word targets only.
    second_runs_on, failed_second_runs: the same for the second run on a store.
    put_fault: the putter of faults handed to OriginBlame, or None.
    """

    runner: RecordingRunner
    log: StepLog
    work: Path
    run_record: Path
    ran_on: set = field(default_factory=set)
    failed_runs: dict = field(default_factory=dict)
    layout_runs: dict = field(default_factory=dict)
    second_runs_on: set = field(default_factory=set)
    failed_second_runs: dict = field(default_factory=dict)
    put_fault: Callable[[Any, str], Any] | None = None
    _notes: Any = None
    name: str = ""
    plain: "SubjectContext | None" = None
    plain_subject: Any = None
    served: Callable[[Question], Any] | None = None

    def _putter(self, config_putter):
        def put(copy, fault_id):
            got = config_putter(copy, fault_id)
            self._notes = got
            return got

        return put

    def take_notes(self):
        found, self._notes = self._notes, None
        if found is None and self.plain is not None:
            # A combination calls the putter of its plain half: its notes are the plain context's.
            found, self.plain._notes = self.plain._notes, None
        return found

    def keep_fact(self, kind: str, target: Sequence[str], outcome: str, attempt_id: str | None = None) -> None:
        """Keep one fact of this subject, as one line of ``subject-facts.jsonl`` written now: ``kind``
        is ``export`` (a trace or a manifest taken for a store or set), ``half`` (one half of a
        combination, or a reason a question was not read) or ``shard`` (one shard of the tool),
        ``outcome`` says what came of it and ``attempt_id`` is the id of the failed attempt, if any.
        A local path is cut to its last name."""
        line = {"subject": self.name, "kind": kind, "target": list(target), "outcome": str(outcome),
                "attempt_id": attempt_id}
        try:
            record.refuse_private(line, "a fact")
        except ValueError:
            line["outcome"] = _LOCAL_PATH.sub(lambda found: found.group(0).rstrip("/").rsplit("/", 1)[-1], line["outcome"])
            try:
                record.refuse_private(line, "a fact")
            except ValueError:
                line["outcome"] = "(the text of this fact holds a private name and is withheld)"
        with open(self.run_record / FACTS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=True, sort_keys=True) + "\n")


@dataclass
class PassResult:
    """What a pass leaves: its lines (in question order), the attempts of each subject on each
    store or set, and the failed attempts kept in ``attempts.jsonl``."""

    lines: list[dict]
    store_attempts: list[record.StoreAttempt]
    failed_attempts: list[dict]


def _roots_of(config: RunConfig) -> dict[str, str]:
    roots = {str(Path(config.work)): "<work>"}
    for name, setup in config.setups.items():
        roots[str(setup.env_dir)] = f"<env-{name}>"
        with contextlib.suppress(ValueError, OSError):
            for bind in sandbox.interpreter_binds(setup.env_dir):
                roots[bind] = "<interpreters>"
        if setup.ob_binary is not None:
            roots[str(setup.ob_binary)] = "<program>"
    for name, build in (config.builds or {}).items():
        roots[str(build)] = f"<build-{name}>"
    return roots


SHARDLESS_RECORDED = ("the tool ran in this pass; its records hold nothing for a store with no chunk file and no shard")


def _store_attempts_of(subject: str, context: SubjectContext, none_questions: Sequence[Question],
                       shardless: Collection[tuple[str, str]] = ()) -> list[record.StoreAttempt]:
    out: list[record.StoreAttempt] = _own_store_attempts_of(subject, context, none_questions)
    if shardless and any(tuple(record.attempt_target(q)) in shardless for q in none_questions):
        # A store with no chunk file and no shard cannot be a target of a tool's own run (narrow amendment of 17 Nov 2026
        # to the round-2 rule that a none line needs the attempt of its subject on its store): the attempts of the
        # pass stand for it. The tool completed a run in the pass: it ran, and its records hold nothing for the store.
        # It completed none: its failed attempt is the attempt, and the line is that failed attempt. No run and no
        # failure at all: no attempt, and the line stays not attempted.
        failures = [bad for bad in (*context.failed_runs.values(), *context.failed_second_runs.values())]
        ran = bool(context.ran_on) or bool(context.second_runs_on)
        own = set(context.ran_on) | set(context.failed_runs)
        if ran or failures:
            # A subject with its own run on the store (the manifest and the trace read the store's own files) keeps that
            # attempt: no second one is made for it.
            for target in sorted(t for t in shardless if t not in own
                                 and any(tuple(record.attempt_target(q)) == t for q in none_questions)):
                layouts: list[str | None] = [None]
                for question in none_questions:
                    if tuple(record.attempt_target(question)) == target:
                        for name in record.layouts_of(question) or ():
                            if name not in layouts:
                                layouts.append(name)
                for layout in layouts:
                    out.append(record.StoreAttempt(
                        subject=subject, target=tuple(target), attempt_id=None if ran else failures[0].id,
                        recorded=SHARDLESS_RECORDED if ran else "the tool completed no run in this pass",
                        layout=layout, run=1))
    return out


def _own_store_attempts_of(subject: str, context: SubjectContext, none_questions: Sequence[Question]) -> list[record.StoreAttempt]:
    out: list[record.StoreAttempt] = []
    for run, ran, failed in ((1, context.ran_on, context.failed_runs), (2, context.second_runs_on, context.failed_second_runs)):
        for target in sorted(set(ran) | set(failed)):
            layouts: list[str | None] = [None]
            for question in none_questions:
                if tuple(record.attempt_target(question)) == tuple(target):
                    for name in record.layouts_of(question) or ():
                        if name not in layouts:
                            layouts.append(name)
            for layout in layouts:
                # The attempt of a layout is that of the build in that layout: a build ADDED beside the target's own has its
                # status in ``layout_runs`` (first run only); any other layout is the target's own run.
                if run == 1 and layout is not None and (*target, layout) in context.layout_runs:
                    bad = context.layout_runs[(*target, layout)]
                else:
                    bad = failed.get(target)
                out.append(record.StoreAttempt(
                    subject=subject, target=tuple(target), attempt_id=bad.id if bad is not None else None,
                    recorded="the run failed" if bad is not None else "the run ended and left its records",
                    layout=layout, run=run))
    return out


EXPECTED_SOURCES = "expected-sources"
_EXPECTED_SOURCE_KEYS = ("counted", "counted_by_layout", "counted_no_capture", "index", "index_no_capture", "oracle_logs")


def keep_expected_sources(world: Mapping[str, Any], run_record) -> None:
    """Keep, sealed beside the answers, the files the grader reads the expected values from: the counting documents, the shard
    indexes and the oracle's logs the world names, and the world's request entries. A build made without capture cannot be built again (its shuffles are
    unseeded), so its expected values exist only in the world that was built; the copies let a re-grade read them after the
    machine and its world are gone. ``expected-sources/index.json`` names each copy and its sha256 (no local path)."""
    base = Path(run_record) / EXPECTED_SOURCES
    index: dict[str, dict[str, str]] = {}
    for key in _EXPECTED_SOURCE_KEYS:
        entries = world.get(key) or {}
        found: dict[str, str] = {}
        for name, value in entries.items():
            sources = value if isinstance(value, Mapping) else {None: value}
            for inner, source in sources.items():
                source = Path(source)
                relative = "/".join(part for part in (key, str(name), str(inner) if inner is not None else None) if part)
                relative = f"{relative}{source.suffix}"
                target = base / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                found[relative] = _sha256_of(target)
        if found:
            index[key] = found
    base.mkdir(parents=True, exist_ok=True)
    if world.get("requests"):
        # The grader of the file family reads the files of each member through the world's request entries.
        (base / "requests.json").write_text(json.dumps(world["requests"], indent=1, sort_keys=True) + "\n", encoding="utf-8")
        index["requests"] = {"requests.json": _sha256_of(base / "requests.json")}
    (base / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (Path(run_record) / FROZEN_FILES).write_text(json.dumps(frozen_files_hashes(), indent=1, sort_keys=True) + "\n", encoding="utf-8")


FROZEN_FILES = "frozen-files.json"


def frozen_files_hashes() -> dict[str, str]:
    """The sha256 of the frozen files the graders read from the repository and not from the record: the fixture spec and the
    files of the table (``fixture/table/*.toml`` and its README) and the acceptance modules the graders load
    (``expected.ACCEPTANCE_MODULES``, 18 Nov 2026). A re-grade compares them with the files it finds (18 Nov 2026:
    the expected values of a row come from the spec and the table, which are not among the kept expected-value sources)."""
    root = Path(table10.REPO)
    names = [Path("fixture/spec/fixture_spec.toml"), Path("fixture/table/README.md"),
             *sorted(path.relative_to(root) for path in (root / "fixture" / "table").glob("*.toml")),
             *(Path("tests") / "acceptance" / f"{module}.py" for module in expected.ACCEPTANCE_MODULES)]
    return {name.as_posix(): _sha256_of(root / name) for name in names if (root / name).is_file()}


def world_from_record(run_record) -> dict[str, Any]:
    """The world keys a grader reads (the counting documents, the shard indexes, the oracle's logs), taken from the copies a
    run record keeps (``keep_expected_sources``) and in the shape the world has them, pointing into the record. A re-grade of the
    kept readings builds its graders from this (``_graders``), after the machine and its world are gone. ``ValueError`` for a
    copy that no longer has the sha256 the index names."""
    base = Path(run_record) / EXPECTED_SOURCES
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    world: dict[str, Any] = {}
    for key, entries in index.items():
        found: dict[str, Any] = {}
        for relative, digest in entries.items():
            path = base / relative
            if not path.is_file() or _sha256_of(path) != digest:
                raise ValueError(f"the kept copy {relative} is not the file the index names (sha256 {digest})")
            if key == "requests":
                found = json.loads(path.read_text(encoding="utf-8"))
                break
            parts = Path(relative).parts[1:]
            if len(parts) == 1:
                found[Path(parts[0]).stem] = str(path)
            else:
                found.setdefault(parts[0], {})[Path(parts[1]).stem] = str(path)
        world[key] = found
    return world


def seal_run_record(folder) -> Path:
    """Write ``sha256sums.txt`` in ``folder``: the sha256 of every other file below it, as
    ``<hex>  <relative name>``, in name order. Returns its path."""
    base = Path(folder)
    seal = base / SEAL_FILE
    names = sorted(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p != seal)
    seal.write_text("".join(f"{_sha256_of(base / name)}  {name}\n" for name in names), encoding="utf-8")
    return seal


def run_pass(config: RunConfig, starter) -> PassResult:
    """One pass (see ``_run_pass``). A ``PreRunStop`` of any stage leaves ``stopped.json`` in the run
    record, naming the stop (no driver, no question), so that the sealed record of a pass that stopped
    before any tool command names its stop as every other stopped record does."""
    try:
        return _run_pass(config, starter)
    except PreRunStop as stop:
        if not (Path(config.run_record) / STOPPED_FILE).exists():
            with contextlib.suppress(Exception):
                Path(config.run_record).mkdir(parents=True, exist_ok=True)
                _write_stopped(StepLog(config.run_record, roots=_roots_of(config)), Path(config.run_record), stop, None, None)
        raise


def _run_pass(config: RunConfig, starter) -> PassResult:
    """One pass over the questions of the selected subjects (see the module words). A
    ``PreRunStop`` leaves as an exception before any tool command, and then no answer line
    is written. The drivers are built and run in one order whatever order the subjects were
    named in (``subjects.ordered``); a combination whose plain half is not in the pass is a
    ``PreRunStop`` (stage ``subjects``) before anything is started. Where the configuration
    names the world file, the first check of all is ``check_world_hashes``. A stop of any stage
    leaves ``stopped.json`` in the run record (``run_pass`` writes it for every ``PreRunStop``)."""
    if config.world_file is not None:
        try:
            check_world_hashes(config.world_file)
        except PreRunStop as stop:
            with contextlib.suppress(Exception):
                _write_stopped(StepLog(config.run_record, roots=_roots_of(config)), Path(config.run_record), stop, None, None)
            raise
    selected = subjects.ordered(config.subjects)
    problem = subjects.missing_half(selected)
    if problem is not None:
        raise PreRunStop("subjects", problem)
    questions = [q for q in config.questions if q.subject in selected]
    check_putters(questions, config.fault_putter)
    if config.requests is not None and config.builds is not None:
        check_request_files(config.requests, config.builds, questions)
    if type(starter) is _REAL_STARTER:
        # Only the real starter runs the tokenizer's manager that binds the socket; a stand-in starter does not.
        check_work_path_of_tools({TOOL_OF.get(name) for name in selected}, config.work)
    run_record = Path(config.run_record)
    log = StepLog(run_record, roots=_roots_of(config))
    # The gates: every tool is checked before anything is built or started.
    gates: dict[str, GatedStarter] = {}
    for name in selected:
        tool = TOOL_OF.get(name)
        if tool is None or tool in gates:
            continue
        if tool not in config.setups:
            raise PreRunStop("compare", f"no environment was handed in for the tool {tool}")
        setup = config.setups[tool]
        gated = GatedStarter(starter)
        pre_run(tool, env_dir=setup.env_dir, env_record=setup.env_record, identity_sha256=setup.identity_sha256,
                work=config.work, lock_prefix=config.lock_prefix, starter=gated, run_record=run_record,
                ob_binary=setup.ob_binary, log=log, probe_timeout=config.step_cap_seconds,
                program_sha256=setup.program_sha256, lock_wait_bound=config.lock_wait_bound)
        gates[tool] = gated
    answered: dict[str, dict] = {}
    cited: dict[str, dict] = {}
    store_attempts: list[record.StoreAttempt] = list(config.attempts)
    contexts: dict[str, SubjectContext] = {}
    subjects_made: dict[str, Any] = {}
    # What the plain OriginBlame read for each question, for the combination served from it.
    plain_outcomes: dict[str, Any] = {}

    def served(question: Question):
        twin = subjects.twin_id(question)
        return plain_outcomes.get(twin) if twin is not None else None

    current: Question | None = None
    driver: str | None = None
    try:
        for name in selected:
            driver, current = name, None
            log.owner = name
            tool = TOOL_OF.get(name)
            gated = gates[tool] if tool is not None else GatedStarter(starter)
            context = SubjectContext(
                runner=RecordingRunner(gated, log, cap_seconds=config.step_cap_seconds,
                                       lock_wait_bound=config.lock_wait_bound), log=log,
                work=Path(config.work), run_record=run_record, name=name)
            if config.fault_putter is not None:
                context.put_fault = context._putter(config.fault_putter)
            plain_name = subjects.PLAIN_HALF.get(name)
            if plain_name is not None:
                # A combination starts from the runs of its plain half: the same targets and the
                # same failed attempts, so that its none lines stand on them.
                plain = contexts[plain_name]
                context.plain, context.plain_subject = plain, subjects_made[plain_name]
                context.ran_on.update(plain.ran_on)
                context.failed_runs.update(plain.failed_runs)
                context.layout_runs.update(plain.layout_runs)
                context.second_runs_on.update(plain.second_runs_on)
                context.failed_second_runs.update(plain.failed_second_runs)
                if name == "originblame+trace":
                    context.served = served
            contexts[name] = context
            subject = config.factories[name](context)
            subjects_made[name] = subject
            for question in (q for q in questions if q.subject == name and q.route is not None):
                context._notes = None
                if context.plain is not None:
                    context.plain._notes = None
                current = question
                result = answer_question(
                    question, subject, commit=config.commit, log=log, grade_of=config.grade_of,
                    grade_layouts_of=config.grade_layouts_of, failed_runs=context.failed_runs,
                    layout_runs=context.layout_runs, notes=context.take_notes, twin_of=subjects.twin_id if name == "originblame+trace" else None)
                if name == "originblame" and result.outcome is not None:
                    plain_outcomes[question.id] = result.outcome
                if result.line is not None:
                    answered[question.id] = result.line
                if result.attempt is not None:
                    _cite(cited, result.attempt, question.id, result.step_ids,
                          own_steps=name not in subjects.PLAIN_HALF)
            current = None
        driver = None
        none_questions = [q for q in questions if q.route is None]
        for name in selected:
            mine = [q for q in none_questions if q.subject == name]
            store_attempts += _store_attempts_of(name, contexts[name], mine, config.shardless)
        subject_of = {q.id: q.subject for q in none_questions}
        def step_ids_of(question: Question) -> Sequence[str]:
            held = list(log.step_ids(question))
            if tuple(record.attempt_target(question)) in config.shardless:
                # No run of the tool is on this store: the line rests on the runs the tool made in the pass.
                plain = contexts[subjects.PLAIN_HALF.get(question.subject, question.subject)]
                held += [step for step in log.tool_steps(
                    question.subject, [*plain.ran_on, *plain.failed_runs, *plain.second_runs_on, *plain.failed_second_runs])
                    if step not in held]
            return held

        lines = none_lines(none_questions, store_attempts, commit=config.commit, step_ids_of=step_ids_of)
        for line in lines:
            answered[line["id"]] = line
        # The plain tools cite their failed attempts first: a combination that cites the same
        # attempt adds its question and no step.
        for plain_first in (True, False):
            for line in lines:
                kind = subject_of[line["id"]]
                if line["attempt_id"] is None or (kind in subjects.PLAIN_TOOLS) is not plain_first:
                    continue
                for context in contexts.values():
                    bad = [a for a in (*context.failed_runs.values(), *context.failed_second_runs.values())
                           if a.id == line["attempt_id"]]
                    if bad:
                        _cite(cited, bad[0], line["id"], line["step_ids"], own_steps=kind not in subjects.PLAIN_HALF)
    except Exception as error:
        # A run that stops keeps what it has made: the lines and attempts so far are written and
        # sealed beside the steps, with the name of the stop (the error, the driver the pass was
        # in and the question it was answering). A stopped pass is never graded.
        if current is not None:
            error.add_note(f"while answering {current.id}")
        with contextlib.suppress(Exception):
            _write_stopped(log, run_record, error, driver, current)
        with contextlib.suppress(Exception):
            _write_record(config, questions, answered, cited, run_record, subjects_made)
        raise
    lines, attempt_lines = _write_record(config, questions, answered, cited, run_record, subjects_made)
    return PassResult(lines=lines, store_attempts=store_attempts, failed_attempts=attempt_lines)


def _write_stopped(log: StepLog, run_record: Path, error: BaseException, driver: str | None,
                   question: Question | None) -> None:
    """Write ``stopped.json`` of the record of a pass that stops: the text of the error (a local
    path cut to its last name), the driver the pass was in and the id of the question it was
    answering (None where the error came while a driver was being built or after the questions)."""
    text = "".join([f"{type(error).__name__}: {error}", *(f"\n{note}" for note in getattr(error, "__notes__", ()))])
    text = _LOCAL_PATH.sub(lambda found: found.group(0).rstrip("/").rsplit("/", 1)[-1], log._place(text))
    body = {"error": text, "subject": driver, "question": question.id if question is not None else None}
    (run_record / STOPPED_FILE).write_text(json.dumps(body, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_record(config: RunConfig, questions: Sequence[Question], answered: Mapping[str, dict],
                  cited: Mapping[str, dict], run_record: Path,
                  subjects: Mapping[str, Any] | None = None) -> tuple[list[dict], list[dict]]:
    lines = [answered[q.id] for q in questions if q.id in answered]
    attempt_lines = [record.attempt_line(info["attempt"], info["questions"], commit=config.commit,
                                         step_ids=info["steps"]) for info in cited.values()]
    record.write_answers(run_record / ANSWERS_FILE, lines)
    record.write_attempts(run_record / ATTEMPTS_FILE, attempt_lines)
    _write_unparseable(run_record, subjects or {})
    seal_run_record(run_record)
    return lines, attempt_lines


def _write_unparseable(run_record: Path, subjects: Mapping[str, Any]) -> None:
    """Keep the count of the lines no subject could parse, as ``unparseable.jsonl``: one line
    ``{"build", "file", "lines"}`` for each file of each build, in name order. A subject
    reports them as ``unparseable_lines``, a mapping of its build names to the count of each
    file; they are read here, once the questions are answered, because a subject counts a build
    when it first reads it. No subject that reports them, no file."""
    reported = [found for found in (getattr(subject, "unparseable_lines", None) for subject in subjects.values())
                if isinstance(found, Mapping)]
    if not reported:
        return
    counts: dict[tuple[str, str], int] = {}
    for found in reported:
        for build, files in found.items():
            for name, number in files.items():
                counts[(build, name)] = max(counts.get((build, name), 0), number)
    text = "".join(json.dumps({"build": build, "file": name, "lines": number}, sort_keys=True) + "\n"
                   for (build, name), number in sorted(counts.items()) if number)
    (run_record / UNPARSEABLE_FILE).write_text(text, encoding="utf-8")


def _cite(cited: dict, attempt: Attempt, question_id: str, step_ids: Sequence[str], *,
          own_steps: bool = True) -> None:
    """Cite ``attempt`` for a question. ``own_steps`` False (a combination's question) adds the
    question to an attempt a plain subject already cited and no step of its own: the attempt's
    steps are the plain subject's. An attempt no one cited yet takes the steps given."""
    known = attempt.id in cited
    info = cited.setdefault(attempt.id, {"attempt": attempt, "questions": [], "steps": []})
    if question_id not in info["questions"]:
        info["questions"].append(question_id)
    if known and not own_steps:
        return
    for found in step_ids:
        if found not in info["steps"]:
            info["steps"].append(found)


# --- The command -----------------------------------------------------------------------------------------


def start_line(*, subjects: Sequence[str], step_cap_seconds: float, identity_sha256s: Mapping[str, str],
               commit: str, program_sha256: str | None = None, lock_wait_bound: float | None = None,
               world_list: str | None = None) -> str:
    """The one line a run starts with: the commit, the subjects, the cap of a step, the sha256
    of the identity file of each tool and, where the run is given them, the sha256 supplied for
    the program, the wait bound of the lock and the sha256 of the ``WORLD.sha256`` of the world
    (``world_list``: the sha256, or the word ``none`` where the folder holds no such file)."""
    identities = " ".join(f"{name}={found}" for name, found in sorted(identity_sha256s.items()))
    line = (f"start: commit {commit}; subjects {','.join(subjects)}; step cap {step_cap_seconds} seconds; "
            f"identity.json sha256 {identities}")
    if program_sha256 is not None:
        line += f"; program sha256 supplied {program_sha256}"
    if lock_wait_bound is not None:
        line += f"; lock wait bound {lock_wait_bound} seconds"
    if world_list is not None:
        line += f"; {WORLD_LIST_FILE} sha256 {world_list}"
    return line


def _positive(text: str) -> float:
    try:
        found = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is no number of seconds") from error
    if not found > 0:
        raise argparse.ArgumentTypeError("a number of seconds above zero is needed")
    return found


def _sha256_text(text: str) -> str:
    found = text.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", found):
        raise argparse.ArgumentTypeError(f"{text!r} is no sha256 (64 hexadecimal digits)")
    return found


# The extensions a pass asked the rows of (one line, names separated by spaces, empty for none); the re-grade reads it.
EXTENSIONS_FILE = "extensions.txt"


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The options of ``python -m baselines``: ``--subjects`` (default: every subject of the
    table) and ``--step-cap`` (seconds, required) select what runs and bound each step; the
    others name the folders and the tool setups of a real run."""
    parser = argparse.ArgumentParser(prog="python -m baselines", description="Run Check 10.")
    parser.add_argument("--subjects", nargs="+", choices=SUBJECTS, default=list(SUBJECTS),
                        help="the subjects to run (default: all six)")
    parser.add_argument("--extensions", nargs="*", default=[],
                        help="the extensions whose rows are asked (default: none: a row of an extension that is off is not asked)")
    parser.add_argument("--step-cap", type=_positive, required=True,
                        help="the cap of every tool step, in seconds")
    parser.add_argument("--commit", help="the commit the run is on (default: the head of the repository)")
    parser.add_argument("--work", help="the work folder")
    parser.add_argument("--run-record", help="the new folder the run record goes to")
    parser.add_argument("--lock-prefix", nargs="+", help="the command every step is started behind")
    parser.add_argument("--world", help="the file that names the builds, chunk files and requests of the run")
    parser.add_argument("--program-sha256", type=_sha256_text, default=None,
                        help="the sha256 the program of OriginBlame must have, supplied for this run (never read "
                             "from identity.json); the run stops at the stage binary on a difference")
    parser.add_argument("--lock-wait-bound", type=_positive, default=None,
                        help="the seconds a step may wait for the lock behind the lock prefix; with it every step "
                             "is handed a note file and the bound (WATERWHITE_LOCK_NOTE, WATERWHITE_LOCK_WAIT_BOUND)")
    parser.add_argument("--tool", nargs="+", action="append", default=[], metavar="WORD",
                        help="one tool: its name (dolma or originblame), its copied environment, the record of the copy, "
                             "the sha256 of its identity.json and, for originblame, the program")
    return parser.parse_args(None if argv is None else list(argv))


# --- The world of a real run -------------------------------------------------------------------------------
#
# A run reads one JSON file, ``--world``, that names what the builds hold (docs/baselines.md):
#
#   tokenizer     the tokenizer file (an absolute path);
#   tie_record    the result of the tie show (an absolute path);
#   requests      the request words of the table, each with the ``files`` it names (relative to
#                 the build) and, where it narrows to one record, ``record``;
#   split_pairs   the plant list's split pairs;
#   stores        for Dolma, one entry per store or set the tool is run on: ``target`` (["store",
#                 NAME] or ["set", SIZE]), ``build`` (absolute), ``chunk_files`` (artifact name to
#                 the chunk file, relative to the build) and ``stop_and_run_again`` (true for the
#                 store whose writer is stopped and run again);
#   shards        for OriginBlame, one entry per shard it tracks: ``name``, ``target``, ``layout``,
#                 ``run``, ``successor``, ``build``, ``chunk_file`` and ``idx_prefix`` (absolute,
#                 the shard's own index and packed file without their suffix);
#   counted, index, counted_by_layout   the counting documents and shard index files the expected
#                 values are read from, by size (``once`` for a row asked once).

STOP_AFTER_ROWS = 100
POLL_SECONDS = 0.05
COMPANION_SUFFIXES = (".csv", ".csv.gz")


class WorldError(ValueError):
    """A world file that cannot drive a run."""


def _target_of(value: Any) -> tuple[str, str]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and value[0] in ("store", "set")):
        raise WorldError(f"a target is [\"store\", NAME] or [\"set\", SIZE], got {value!r}")
    return (str(value[0]), str(value[1]))


def _companion_files_in(folder: Path) -> dict[str, Path]:
    """The companion files a tokenizer run left in a folder, by a flat name (a name two files
    share is numbered), the files a reading parses."""
    found: dict[str, Path] = {}
    if not folder.is_dir():
        return found
    for path in sorted(p for p in folder.rglob("*") if p.is_file() and p.name.endswith(COMPANION_SUFFIXES)):
        name = path.name if path.name not in found else f"{len(found)}-{path.name}"
        found[name] = path
    return found


def _listing_of(folder: Path) -> list[dict]:
    return [{"name": path.relative_to(folder).as_posix(), "size": path.stat().st_size, "sha256": _sha256_of(path)}
            for path in sorted(folder.rglob("*")) if path.is_file()]


def _tokens_kept(context: "SubjectContext", build: Path, destination: str, number: int):
    """What a tokenizer step keeps, read once the step has ended: the companion files it wrote
    and a listing (name, size, sha256) of everything in its destination, the token files
    included, from which the size of a token file can be re-read."""
    def held() -> dict[str, Path]:
        folder = build / destination
        files = _companion_files_in(folder)
        listing = context.work / f"listing-{number}.json"
        listing.write_text(json.dumps(_listing_of(folder) if folder.is_dir() else [], indent=1), encoding="utf-8")
        files["listing.json"] = listing
        return files

    return held


class _Numbers:
    """Distinct numbers for the folders and ids of one run."""

    def __init__(self):
        self._next = 0

    def __call__(self) -> int:
        self._next += 1
        return self._next


def _tokens_argv(context: SubjectContext, setup: ToolSetup, lock_prefix, build: Path, chunk_files, destination: str,
                 tokenizer: str, number: int) -> list[str]:
    tmp = context.work / f"dolma-tmp-{number}"
    prepare_tokenizer_folders(build, work_dir="tokenizer-work", tmpdir=tmp)
    home = sandbox.prepare_home(context.work, f"dolma-home-{number}")
    return dolma.tokens_step(
        lock_prefix=lock_prefix, work=context.work, home=home, build=build, env_dir=setup.env_dir,
        chunk_files=chunk_files, destination=destination, tokenizer=tokenizer, work_dir="tokenizer-work",
        env={"TMPDIR": str(tmp)})


def _dolma_full_run(context, setup, lock_prefix, entry, build: Path, tokenizer: str, numbers: _Numbers):
    """One tokenizer run over every chunk file of a store or set, then the mixer once per
    layout (its steps are kept; the table reads nothing of its output). The rows, or the failed
    attempt."""
    chunk_files = list(entry["chunk_files"].values())
    destination = f"tokens-{numbers()}"
    number = numbers()
    argv = _tokens_argv(context, setup, lock_prefix, build, chunk_files, destination, tokenizer, number)

    def runner(words):
        got = context.runner.run(words, kept=_tokens_kept(context, build, destination, number))
        return got.exit_code, got.output

    rows = dolma.run_tokens(runner, argv, build=build, chunk_files=chunk_files, destination=destination,
                            attempt_id=f"dolma-tokens-{number}")
    if isinstance(rows, Attempt):
        return rows
    layouts: dict[str, list[str]] = {}
    for chunk in chunk_files:
        parts = Path(chunk).parts
        if dolma._CHUNK_FOLDER in parts and parts.index(dolma._CHUNK_FOLDER) > 0:
            layouts.setdefault(parts[parts.index(dolma._CHUNK_FOLDER) - 1], []).append(chunk)
    for layout, chunks in sorted(layouts.items()):
        count = numbers()
        config = f"mixer-{count}.yaml"
        dolma.write_mix_config(build / config, chunks, f"mixed-{count}", work_dir=f"mixer-work-{count}")
        prepare_mixer_folders(build, work_dir=f"mixer-work-{count}")
        tmp = context.work / f"dolma-tmp-{count}"
        tmp.mkdir(parents=True, exist_ok=True)
        home = sandbox.prepare_home(context.work, f"dolma-home-{count}")
        context.runner.run(dolma.mix_step(lock_prefix=lock_prefix, work=context.work, home=home, build=build,
                                          env_dir=setup.env_dir, config=config, env={"TMPDIR": str(tmp)}))
    return rows


_DTYPE_LINE = re.compile(r"^dtype:\s*(\w+)\s*$", re.MULTILINE)


def token_file_tokens_of(destination: Path, printed: str) -> int | None:
    """The number of tokens of the token file a run of the tokenizer wrote: the bytes of the ``.npy``
    files of ``destination`` over the item size of the dtype the run itself printed in its configuration
    (``dtype: uint16``). None where no dtype was printed or the destination holds no ``.npy`` file; no number
    is made up."""
    found = _DTYPE_LINE.search(printed or "")
    digits = re.search(r"(\d+)$", found.group(1)) if found else None
    files = sorted(Path(destination).glob("*.npy")) if Path(destination).is_dir() else []
    if digits is None or not files or int(digits.group(1)) % 8 != 0 or int(digits.group(1)) == 0:
        return None
    return sum(path.stat().st_size for path in files) // (int(digits.group(1)) // 8)


def _dolma_stop_and_again(context, setup, lock_prefix, entry, build: Path, tokenizer: str, numbers: _Numbers) -> dict:
    """The writer of the store of Check 4: started, stopped by its id as soon as its output
    holds the rows of ``STOP_AFTER_ROWS`` documents (or when it ends), then run again on the
    same output. The fields ``DolmaRun`` reads for the stopped writer and the second run."""
    chunk_files = list(entry["chunk_files"].values())
    destination = f"tokens-stop-{numbers()}"
    number = numbers()
    argv = _tokens_argv(context, setup, lock_prefix, build, chunk_files, destination, tokenizer, number)
    handle = context.runner.start(argv, kept=_tokens_kept(context, build, destination, number))
    showed = False
    while not handle.done():
        try:
            if len(rows_after_stop(build / destination)) >= STOP_AFTER_ROWS:
                showed = True
                break
        except (OSError, ValueError):
            pass
        time.sleep(POLL_SECONDS)
    sent = handle.stop()
    ended = handle.wait()
    fields: dict[str, Any] = dataclasses.asdict(stop_record(
        command=argv, sent=sent, ended=ended, attempt_id=f"dolma-stop-{number}", rows_showed_while_running=showed))
    fields["stopped_rows"] = rows_after_stop(build / destination)
    number = numbers()
    second = _tokens_argv(context, setup, lock_prefix, build, chunk_files, destination, tokenizer, number)
    result = context.runner.run(second, kept=_tokens_kept(context, build, destination, number))
    fields.update(dataclasses.asdict(retry_record(command=second, ended=result, attempt_id=f"dolma-retry-{number}")))
    fields["retry_rows"] = rows_after_stop(build / destination)
    fields["token_file_tokens"] = token_file_tokens_of(build / destination, result.output)
    fields["second_run_step"] = result.step_id
    return fields


def dolma_factory(world: Mapping[str, Any], setup: ToolSetup, lock_prefix):
    """The factory of the ``dolma`` subject: one run of the tokenizer on each store or set the
    world names (the steps shared by every question of it), and for the store whose writer is
    stopped, the stopped run and the second run beside it. A run that failed is the failed
    attempt of its store or set."""
    def factory(context: SubjectContext):
        numbers = _Numbers()
        runs: dict[tuple[str, str], Any] = {}
        for entry in world["stores"]:
            target = _target_of(entry["target"])
            # A build ADDED beside the target's own names its layout (W1): its run is kept under (kind, name, layout), its steps
            # are logged under that key, and its status goes to ``layout_runs``; ``ran_on`` and ``failed_runs`` keep targets.
            layout = entry.get("layout")
            key = (*target, layout) if layout else target
            build = Path(entry["build"])
            tokenizer = place_tokenizer(build, world["tokenizer"], name="gpt2-tokenizer.json")
            with context.log.shared(key):
                found = _dolma_full_run(context, setup, lock_prefix, entry, build, tokenizer, numbers)
                if layout:
                    context.layout_runs[key] = found if isinstance(found, Attempt) else None
                    runs[key] = found if isinstance(found, Attempt) else dolma.DolmaRun(
                        build=build, rows=found, chunk_files=dict(entry["chunk_files"]), layouts=(layout,))
                    continue
                if isinstance(found, Attempt):
                    context.failed_runs[target] = found
                    runs[target] = found
                    continue
                context.ran_on.add(target)
                extra: dict[str, Any] = {}
                if entry.get("stop_and_run_again"):
                    extra = _dolma_stop_and_again(context, setup, lock_prefix, entry, build, tokenizer, numbers)
                    context.second_runs_on.add(target)
                    step_id = extra.pop("second_run_step")
                    if extra["retry_exit_code"] is None:
                        # The second run did not end by itself (the cap): its attempt failed.
                        step = next(s for s in context.log.steps if s.id == step_id)
                        context.failed_second_runs[target] = attempt_of(step, extra["retry_attempt_id"])
                runs[target] = dolma.DolmaRun(build=build, rows=found, chunk_files=dict(entry["chunk_files"]),
                                              layouts=tuple(sorted(_entry_layouts(entry))), **extra)
        # The plant list's split pairs of each set (W8, 18 Nov 2026): the world names them in the entry of the set; a world that
        # holds one list for all (an older world file) hands that list on.
        per_set = {_target_of(e["target"]): e["split_pairs"] for e in world["stores"] if "split_pairs" in e}
        subject = dolma.DolmaSubject(runs, tie_record=world["tie_record"], requests=world["requests"],
                                     split_pairs=per_set if per_set else world.get("split_pairs", ()))
        # A fault put to the rows is noted on the line of the question (``notes``), so that the record shows it.
        subject.note_fault = lambda note: setattr(context, "_notes", note)
        return subject

    return factory


def shard_key(spec, added) -> tuple:
    """The key a shard's steps and status are kept under: the target of its store or set, or (kind, name, layout) where the shard
    belongs to a build ADDED beside the target's own (``added_layouts``)."""
    return (*spec.target, spec.layout) if (tuple(spec.target), spec.layout) in added else tuple(spec.target)


class _AttributingRunner:
    """A runner that names the store or set of the steps OriginBlame makes for each shard it
    tracks (it tracks them all in one call): the step whose ``--store`` is the tracked store of
    shard N is a step of the target of that shard."""

    TRACKED = re.compile(r"tracked-(\d+)$")

    def __init__(self, runner: RecordingRunner, log: StepLog, shards: Sequence[Any], added=frozenset()):
        self._runner, self._log, self._shards, self._added = runner, log, list(shards), frozenset(added)

    def _target(self, argv: Sequence[str]):
        if "--store" in argv:
            found = self.TRACKED.search(str(argv[list(argv).index("--store") + 1]))
            if found and int(found.group(1)) < len(self._shards):
                return shard_key(self._shards[int(found.group(1))], self._added)
        return None

    @staticmethod
    def _kept(argv: Sequence[str]):
        """The files a reading of the step parses, read once it has ended: the entry files of
        the store the step names (``--store``) and the file it was told to write (``--out``,
        the forget set), and with the entry files the ``driver_meta.json`` of the store (the section
        registered for each file, which a reading of ``gen/T`` is made from and which no entry
        names). Names are flat; the step's own folder tells the steps apart."""
        words = [str(word) for word in argv]

        def held() -> dict[str, Path]:
            files: dict[str, Path] = {}
            if "--store" in words:
                store = Path(words[words.index("--store") + 1])
                folder = store.joinpath(*ENTRY_FOLDER)
                if folder.is_dir():
                    files.update({f"entries-{p.name}": p for p in sorted(folder.iterdir()) if p.is_file()})
                meta = store / ob_driver.META_FILE
                if meta.is_file():
                    files[ob_driver.META_FILE] = meta
            if "--out" in words:
                out = Path(words[words.index("--out") + 1])
                if out.is_file():
                    files[f"out-{out.name}"] = out
            return files

        return held

    def run(self, argv, timeout=None, **keywords):
        target = self._target(argv)
        keywords.setdefault("kept", self._kept(argv))
        if target is None:
            return self._runner.run(argv, timeout, **keywords)
        with self._log.shared(target):
            return self._runner.run(argv, timeout, **keywords)

    def start(self, argv):
        return self._runner.start(argv, kept=self._kept(argv))


def originblame_shards(world: Mapping[str, Any], unparseable: dict[str, dict[str, int]] | None = None) -> list:
    """The shards of OriginBlame: each one's documents in ordinal order, from its chunk file and
    its own index. The id of each record is mapped to its file once for each build. The lines
    of a build that are not JSON are counted by ``IdMap``; when ``unparseable`` is handed in it
    is filled with them, by the name ``<kind>-<name>`` of the build's store or set."""
    maps: dict[Path, dict[str, str]] = {}
    counted: dict[Path, dict[str, int]] = {}
    shards = []
    added = added_layouts(world)
    for item in world["shards"]:
        build = Path(item["build"])
        if build not in maps:
            found = IdMap(build)
            maps[build], counted[build] = found.mapping(), dict(found.unparseable_lines)
        if unparseable is not None:
            # By the name ``<kind>-<name>`` of the build's store or set; a build ADDED beside the target's own (W1) by
            # ``<kind>-<name>-<layout>``, so that it never takes the key of the target's own build.
            kind, name = _target_of(item["target"])
            key = f"{kind}-{name}-{item['layout']}" if (( kind, name), item.get("layout")) in added else f"{kind}-{name}"
            unparseable[key] = dict(counted[build])
        documents = originblame.track_plan(build, item["chunk_file"], item["idx_prefix"], ids=maps[build])
        shards.append(originblame.ShardSpec(
            name=item["name"], target=_target_of(item["target"]), layout=item.get("layout"), run=item.get("run"),
            successor=bool(item.get("successor", False)), build=build, documents=documents))
    return shards


def record_shard_outcomes(context, shards: Sequence[Any], failed: Mapping[int, Attempt], added) -> None:
    """The status of each tracked shard in the context. The failed attempts of the shards are the subject's own record of them.
    A shard of a build ADDED beside the target's own is recorded under (kind, name, layout) in ``layout_runs`` (the first failed
    shard of that build, else None where it ran) and never as the target's own run or failure; a shard of the target's own build
    is ``ran_on`` or the first failure in ``failed_runs``, as before."""
    for number, spec in enumerate(shards):
        key = shard_key(spec, added)
        if len(key) == 3:
            if number in failed:
                if context.layout_runs.get(key) is None:
                    context.layout_runs[key] = failed[number]
                context.keep_fact("shard", key, "failed", attempt_id=failed[number].id)
            else:
                context.layout_runs.setdefault(key, None)
        elif number in failed:
            context.failed_runs.setdefault(spec.target, failed[number])
            # Each shard's failure is kept with its own id: the store's attempt is the first.
            context.keep_fact("shard", spec.target, "failed", attempt_id=failed[number].id)
        else:
            context.ran_on.add(spec.target)


def originblame_factory(world: Mapping[str, Any], setup: ToolSetup, lock_prefix, requests: Mapping[str, Sequence[str]]):
    """The factory of the ``originblame`` subject: every shard tracked once (one process, in
    packed order), the steps of each shard shared by the questions of its store or set. A shard
    whose track failed makes the failed attempt of its store or set."""
    def factory(context: SubjectContext):
        unparseable: dict[str, dict[str, int]] = {}
        shards = originblame_shards(world, unparseable)
        added = added_layouts(world)
        runner = _AttributingRunner(context.runner, context.log, shards, added)
        tool = originblame.OriginBlame(
            context.work, setup.env_dir, lock_prefix, runner, shards, requests,
            ob_binary=setup.ob_binary, fault_putter=context.put_fault)
        # The lines of the builds that are not JSON, counted when the shards were planned.
        tool.unparseable_lines = unparseable
        tool.track_all()
        record_shard_outcomes(context, shards, tool._failed, added)
        return tool

    return factory


def member_files_of(world: Mapping[str, Any]) -> dict[str, list[str]]:
    """The files of each member of the table, derived from the world's own request entries: a request word of the
    table withdraws one source (``baselines``' table reader), the fixture spec gives that source its member
    (``[sources.A] member = "A/r1"``), and the world's entry for the word names the source's files."""
    table = table10._acceptance_table()
    with open(Path(__file__).resolve().parents[1] / "fixture" / "spec" / "fixture_spec.toml", "rb") as handle:
        sources = tomllib.load(handle)["sources"]
    found: dict[str, list[str]] = {}
    for word, entry in (world.get("requests") or {}).items():
        request = table.REQUESTS.get(word)
        source = getattr(request, "source", None)
        member = (sources.get(source) or {}).get("member") if source is not None else None
        if member is None:
            continue
        for path in entry.get("files", ()):
            if path not in found.setdefault(member, []):
                found[member].append(path)
    return found


LAYOUTS_NOT_BUILT_FILE = "layouts-not-built.json"


PER_LAYOUT_READERS = frozenset({"originblame", "dolma+trace", "originblame+trace", "manifest"})
LAYOUT_NAMES = frozenset({"partitioned", "round-robin"})


def layouts_built_of(world: Mapping[str, Any]) -> dict[tuple[str, str], set[str]]:
    """The layouts the world builds for each target: those of its shard entries and the layout folder of every chunk file of
    its Dolma entries, so that a build made without tokenization (chunk files and no shard) counts as a build in its layout."""
    built: dict[tuple[str, str], set[str]] = {}
    for shard in world.get("shards", ()):
        if shard.get("layout"):
            built.setdefault(_target_of(shard["target"]), set()).add(shard["layout"])
    for entry in world.get("stores", ()):
        # Only a layout a question can name: the copies of the larger sizes lie in a folder ``set-<size>`` (a copy of the
        # round-robin chunk files, not a layout of its own) and a successor in ``successor``; a build ADDED beside the target's
        # own (W1) names its layout in its entry.
        for layout in _entry_layouts(entry):
            built.setdefault(_target_of(entry["target"]), set()).add(layout)
    return built


def _entry_layouts(entry: Mapping[str, Any]) -> set[str]:
    """The layouts one Dolma entry of the world is a build of: the layout its entry names (``layout``, written only for a build
    ADDED beside the one the target's name always meant), else the layout folder of each of its chunk files (of the file a copy
    was made from, where the entry says it)."""
    if entry.get("layout"):
        return {entry["layout"]}
    found: set[str] = set()
    copies = entry.get("copy_of") or {}
    for shard, chunk in (entry.get("chunk_files") or {}).items():
        # A copy (a larger size's chunk file lies in a folder ``set-<size>``) is in the layout of the file it was copied from
        # (W9: the world says where each copy came from).
        parts = Path(copies.get(shard, chunk)).parts
        if dolma._CHUNK_FOLDER in parts and parts.index(dolma._CHUNK_FOLDER) > 0:
            name = parts[parts.index(dolma._CHUNK_FOLDER) - 1]
            if name in LAYOUT_NAMES:
                found.add(name)
    return found


def added_layouts(world: Mapping[str, Any]) -> set[tuple[tuple[str, str], str]]:
    """The (target, layout) pairs of the builds ADDED beside a target's own: the Dolma entries of the world that name their layout
    (W1: the 10k set in the partitioned layout). A shard of such a build is read, logged and counted under (kind, name, layout)."""
    return {(_target_of(entry["target"]), entry["layout"]) for entry in world.get("stores", ()) if entry.get("layout")}


def dolma_entries_for(world: Mapping[str, Any], target: tuple[str, str], layout: str | None = None) -> list[Mapping[str, Any]]:
    """The Dolma entries of the world a question on ``target`` reads, in the layout it names (None: the question names none, so
    the build the target's name always meant). An entry that names its layout is read for that layout only; an entry that names
    none is the build of its target and is read for every layout its chunk files lie in, and for a question that names none."""
    found = []
    for entry in world.get("stores", ()):
        if _target_of(entry["target"]) != tuple(target):
            continue
        if layout is None:
            if not entry.get("layout"):
                found.append(entry)
        elif layout in _entry_layouts(entry):
            found.append(entry)
    return found


def check_distinct_builds(world: Mapping[str, Any]) -> None:
    """``ValueError`` when two builds of the world sit under one key (W8, 18 Nov 2026): the key of a Dolma entry is its target
    and its layout (the layout it names, else none); the key of a shard of OriginBlame is its target, its layout and its name. A
    world that holds two builds under one key would be read by one of them for the questions of both."""
    seen: dict[tuple, str] = {}
    for entry in world.get("stores", ()):
        key = ("entry", _target_of(entry["target"]), entry.get("layout") or None)
        if key in seen:
            raise ValueError(f"two builds of the world sit under the key {key[1:]}: {seen[key]} and {entry['build']}")
        seen[key] = entry["build"]
    named: dict[tuple, str] = {}
    for shard in world.get("shards", ()):
        key = ("shard", _target_of(shard["target"]), shard.get("layout"), shard["name"], bool(shard.get("successor")))
        if key in named:
            raise ValueError(f"two shards of the world sit under the key {key[1:]}: {named[key]} and {shard['build']}")
        named[key] = shard["build"]


def layouts_named_by(question: Question) -> tuple[str, ...]:
    """The layouts the listed row of ``question`` names: its one layout, its two, or none."""
    one = record.layout_of(question)
    return (one,) if one is not None else (record.layouts_of(question) or ())


def layout_not_built(world: Mapping[str, Any], questions: Iterable[Question]) -> list[Question]:
    """The questions that stay out of the pass because a layout they name has no build for their target. A reading is recorded
    only from a build in the layout the question asks. Plain Dolma and the Dolma combination (for a single-layout row) key a
    run by its target, so a question that names one layout would be answered from the chunk files of another (OriginBlame
    already selects its shards by the asked layout), and a row of two layouts with no run in one of them would put a none line ("cannot answer", which only a tool that ran can earn) for a layout its subject never ran on. Such a question
    stays not attempted (17 Nov 2026: never "cannot answer", never a stop of the pass, named before the first tool step).
    PER HALF (ruled the same day): a ROUTED row of two layouts that the subject reads per layout (``read_by_layout``) is NOT
    left out: its layout with no build reads None and the line follows ``expected.combine_layouts`` (wrong first, then a failed
    attempt, then not attempted). A none row of two layouts, and a routed one of a subject with no per-layout reader, are left
    out when either layout has no build."""
    built = layouts_built_of(world)
    left_out = []
    for question in questions:
        named = layouts_named_by(question)
        target = tuple(record.attempt_target(question))
        if not named or target not in built or all(layout in built[target] for layout in named):
            continue
        per_half = len(named) > 1 and question.route is not None and question.subject in PER_LAYOUT_READERS
        if not per_half:
            left_out.append(question)
    return left_out


def write_layouts_not_built(run_record: Path, world: Mapping[str, Any], left_out: Sequence[Question]) -> None:
    """Name the questions left out for a layout the world does not build, in the run record, before the first tool step."""
    built = layouts_built_of(world)
    body = {"left_out": [{"id": q.id, "layouts": list(layouts_named_by(q)), "target": list(record.attempt_target(q)),
                          "layouts_built": sorted(built[tuple(record.attempt_target(q))])} for q in left_out]}
    (Path(run_record) / LAYOUTS_NOT_BUILT_FILE).write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")


def covered_targets(world: Mapping[str, Any], inputs: Mapping[str, Any] | None):
    """The targets each subject is asked on, and the SHARDLESS targets: the stores the world names for the manifest or the
    trace that hold no Dolma entry and no shard of OriginBlame (no chunk file and no shard). A shardless store is no target
    of a tool's own run; the none rows of the tools and of the combinations on it are put with the attempts of the pass
    (``_store_attempts_of``)."""
    covered = {"dolma": {_target_of(e["target"]) for e in world.get("stores", ())},
               "originblame": {_target_of(s["target"]) for s in world.get("shards", ())}}
    covered["dolma+trace"], covered["originblame+trace"] = set(covered["dolma"]), set(covered["originblame"])
    named: set[tuple[str, str]] = set()
    if inputs is not None:
        covered["manifest"] = {subjects.target_of_key(key) for key in inputs["manifests"]}
        covered["trace"] = {subjects.target_of_key(key) for key in inputs["traces"]}
        named = covered["manifest"] | covered["trace"]
    else:
        # No subject inputs are loaded (a pass of the two plain tools): the stores the world names for the manifest and the
        # trace are read from the keys of its own subject_inputs, so that the plain tools' lines are the same whichever
        # subjects are driven.
        held = world.get("subject_inputs") or {}
        for where in ("manifest", "trace"):
            for key in held.get(where) or {}:
                if not key.startswith("repaired-"):
                    named.add(subjects.target_of_key(key))
    shardless = frozenset(t for t in named if t not in covered["dolma"] and t not in covered["originblame"])
    return covered, shardless


def _graders(world: Mapping[str, Any]):
    """The two graders of a real run: a reading against the expected value of its row, derived
    from the counting documents and shard indexes the world names."""
    from baselines import grade

    def held(key: str) -> dict:
        return {(None if size == "once" else size): json.loads(Path(path).read_text(encoding="utf-8"))
                for size, path in world.get(key, {}).items()}

    counted, by_layout = held("counted"), held("counted_by_layout")
    index = {(None if size == "once" else size): expected.index_of_shards(files)
             for size, files in world.get("index", {}).items()}
    # A set built in the way of a store (the rows that name a set and the store ``no-capture``) has its own counting
    # document and index (its chunk files differ from the capture set's of the same size): world keys
    # ``counted_<store>`` and ``index_<store>`` ("-" as "_"). A question of that target is graded against those.
    variants: dict[str, tuple[dict, dict]] = {}

    def variant_of(store: str) -> tuple[dict, dict]:
        if store not in variants:
            key = store.replace("-", "_")
            variants[store] = (held(f"counted_{key}"),
                               {(None if size == "once" else size): expected.index_of_shards(files)
                                for size, files in world.get(f"index_{key}", {}).items()})
        return variants[store]

    member_files = member_files_of(world)
    # The oracle's log of each run the world keeps (key ``oracle_logs``: run to a file of lines): the source of the three
    # whole-run diff rows. The log is the world's own, hashed with it.
    oracle = [json.loads(line) for path in (world.get("oracle_logs") or {}).values()
              for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def of(question, layouts=False):
        store = record.set_store_of(question)
        if store is not None:
            if layouts:
                raise ValueError(f"{question.listed_row} names layouts and a set of the store {store}: no counting "
                                 "document by layout is made for that build")
            counted_of, index_of = variant_of(store)
            return expected.expected_for(question.listed_row, question.size, counted=counted_of, index=index_of)
        counted_here = by_layout if layouts else counted
        layout = None if layouts else record.layout_of(question)
        if layout is not None and layout in by_layout:
            # A row that names ONE layout of a set the world built in two (W1) is derived from the counting document counted on
            # that layout, not from the set's own (round-robin) document: expected._shard_lists refuses a layout mismatch. The
            # document of the set's own layout is the same file under ``counted`` (the same document, whatever the path a record keeps it under). ONLY for that set: a row of another size that
            # names round-robin (the rows of Check 3 at small and 100k) keeps the document of its own size (pass 15 on the cut
            # tree, 18 Nov 2026: the 10k document stood in the place of both and 16 right readings were graded wrong).
            own = question.size if question.size is not None else record._listed_givens()[question.listed_row].get("set")
            if own in counted and counted[own] == by_layout.get("round-robin"):
                counted_here = {**counted, own: by_layout[layout]}
        return expected.expected_for(question.listed_row, question.size, counted=counted_here, index=index)

    def grade_of(question, reading):
        if question.listed_row in expected.WHOLE_RUN_ROWS:
            if not oracle:
                return Grade(status=CANNOT_ANSWER,
                             evidence=f"{NOT_DERIVED_WORDS}: the world holds no oracle log of the run {question.listed_row} reads")
            try:
                return grade.grade_whole_run(question, reading, oracle)
            except (grade.NotGradable, ValueError) as error:
                return Grade(status=CANNOT_ANSWER, evidence=f"{NOT_DERIVED_WORDS}: {error}")
        try:
            return grade.grade_concrete(question, expected.concrete_values(of(question)), reading,
                                        member_files=member_files)
        except grade.NotGradable as error:
            return Grade(status=CANNOT_ANSWER, evidence=f"not gradable: {error}")

    def grade_layouts_of(question, outcomes):
        try:
            return grade.grade_layouts(question, expected.concrete_values_by_layout(of(question, True)), outcomes)
        except grade.NotGradable as error:
            return Grade(status=CANNOT_ANSWER, evidence=f"not gradable: {error}")

    return grade_of, grade_layouts_of


def _setups_of(tools: Sequence[Sequence[str]], program_sha256: str | None = None) -> dict[str, ToolSetup]:
    """The setup of each ``--tool``; the sha256 supplied for the program is the setup's of
    OriginBlame (the one tool that has a program)."""
    setups: dict[str, ToolSetup] = {}
    for words in tools:
        if len(words) not in (4, 5) or words[0] not in ("dolma", "originblame"):
            raise SystemExit("--tool takes: dolma|originblame ENV_DIR ENV_RECORD IDENTITY_SHA256 [PROGRAM]")
        setups[words[0]] = ToolSetup(env_dir=Path(words[1]), env_record=Path(words[2]), identity_sha256=words[3],
                                     ob_binary=Path(words[4]) if len(words) == 5 else None,
                                     program_sha256=program_sha256 if words[0] == "originblame" else None)
    return setups


def _commit_of(arguments) -> str:
    if arguments.commit:
        return arguments.commit
    done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                          cwd=Path(table10.REPO))
    return done.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run Check 10 as ``python -m baselines`` does and print its summary. ``--subjects`` may
    name any of the six subjects; the drivers run in one fixed order (``subjects.ordered``). A
    pre-run stop leaves the run record with its steps and no answer line (exit 3); a pass that
    stops on an error leaves a sealed record that names the stop (``stopped.json``) and no
    summary: a stopped pass is never graded."""
    arguments = parse_arguments(argv)
    for name in ("work", "run_record", "lock_prefix", "world"):
        if not getattr(arguments, name):
            raise SystemExit(f"--{name.replace('_', '-')} is needed for a run")
    selected = subjects.ordered(arguments.subjects)
    problem = subjects.missing_half(selected)
    if problem is not None:
        raise SystemExit(problem)
    # The tools a pass needs: a combination runs the tool of its plain half.
    tools_needed = [tool for tool in ("dolma", "originblame") if any(TOOL_OF.get(name) == tool for name in selected)]
    if "originblame" in tools_needed and arguments.program_sha256 is None:
        raise SystemExit("--program-sha256 is needed for a run of originblame: the program is compared with it "
                         "before any tool starts")
    run_record = Path(arguments.run_record)
    if run_record.exists():
        raise SystemExit(f"{run_record} exists: a run record is a new folder")
    world = json.loads(Path(arguments.world).read_text(encoding="utf-8"))
    problem = check_extensions_built(world, arguments.extensions)
    if problem is not None:
        raise SystemExit(problem)
    own_inputs = [name for name in selected if name in ("manifest", "trace", "dolma+trace", "originblame+trace")]
    inputs = None
    if own_inputs:
        try:
            inputs = subjects.load_subject_inputs(world, Path(arguments.world).resolve().parent)
        except subjects.SubjectInputsError as error:
            raise SystemExit(f"{', '.join(own_inputs)} need the files the world names for them: {error}") from None
    work_folder = Path(arguments.work).resolve()
    for found in world.get("stores", ()):
        # The tokenizer writes into its build, so a build of Dolma lies inside the work folder.
        build = Path(found["build"]).resolve()
        if build == work_folder or work_folder not in build.parents:
            raise SystemExit(f"the build {build} lies outside the work folder: a tool writes only inside it")
    setups = _setups_of(arguments.tool, arguments.program_sha256)
    missing = [tool for tool in tools_needed if tool not in setups]
    if missing:
        raise SystemExit(f"--tool is needed for {', '.join(missing)}")
    commit = _commit_of(arguments)
    from baselines import summary

    every = [q for q in table10.expand(table10.load_table10(), arguments.extensions) if q.subject in selected]
    # A question is put to a subject only on a store or set the world names for it: any other
    # stays without a line, which the summary counts as not attempted. A combination is asked on
    # the targets of its plain half; the manifest and the trace on the targets whose files the
    # world names for them.
    try:
        check_distinct_builds(world)
    except ValueError as error:
        raise SystemExit(f"the world is not one build per key: {error}") from None
    covered, shardless = covered_targets(world, inputs)
    questions = [q for q in every if tuple(record.attempt_target(q)) in covered.get(q.subject, ())
                 or (q.route is None and (q.subject in subjects.PLAIN_TOOLS or q.subject in subjects.PLAIN_HALF)
                     and tuple(record.attempt_target(q)) in shardless)]
    left_out = layout_not_built(world, questions)
    left_ids = {q.id for q in left_out}
    questions = [q for q in questions if q.id not in left_ids]
    files_of = {word: list(found["files"]) for word, found in world.get("requests", {}).items()}
    builds = builds_of_world(world)
    ob_selected = "originblame" in selected
    factories = {"dolma": dolma_factory(world, setups.get("dolma"), arguments.lock_prefix),
                 "originblame": originblame_factory(world, setups.get("originblame"), arguments.lock_prefix, files_of)}
    if inputs is not None:
        place_of = subjects.table_place_of()
        factories["trace"] = subjects.trace_factory(inputs["traces"], place_of=place_of,
                                                    failed=inputs["failed_exports"])
        factories["manifest"] = subjects.manifest_factory(inputs["manifests"], requests=world.get("requests", {}),
                                                          traces=inputs["traces"], repaired=inputs["repaired"])
        factories["dolma+trace"] = subjects.dolma_trace_factory(
            traces=inputs["traces"], state_traces=inputs["state_traces"], place_of=place_of)
        factories["originblame+trace"] = subjects.originblame_trace_factory(
            traces=inputs["traces"], state_traces=inputs["state_traces"], place_of=place_of)
    grade_of, grade_layouts_of = _graders(world)
    config = RunConfig(
        commit=commit, work=Path(arguments.work), run_record=run_record, lock_prefix=arguments.lock_prefix,
        subjects=selected, step_cap_seconds=arguments.step_cap, setups=setups, factories=factories,
        questions=questions, grade_of=grade_of, grade_layouts_of=grade_layouts_of,
        fault_putter=ob_fault_putter if ob_selected else None,
        requests=files_of, builds=builds,
        lock_wait_bound=arguments.lock_wait_bound, world_file=Path(arguments.world), shardless=shardless)
    run_record.mkdir(parents=True)
    keep_expected_sources(world, run_record)
    line = start_line(subjects=selected, step_cap_seconds=arguments.step_cap,
                      identity_sha256s={name: setups[name].identity_sha256 for name in tools_needed}, commit=commit,
                      program_sha256=arguments.program_sha256 if ob_selected else None,
                      lock_wait_bound=arguments.lock_wait_bound,
                      world_list=world_list_sha256(arguments.world) or "none")
    print(line, flush=True)
    (run_record / "start.txt").write_text(line + "\n", encoding="utf-8")
    (run_record / EXTENSIONS_FILE).write_text(" ".join(arguments.extensions) + "\n", encoding="utf-8")   # a re-grade asks the same questions
    write_layouts_not_built(run_record, world, left_out)
    try:
        result = run_pass(config, SubprocessStarter())
    except PreRunStop as stop:
        print(f"stopped before any tool command: {stop}", file=sys.stderr)
        seal_run_record(run_record)
        return 3
    except LockWaitStop as stop:
        print(f"stopped: {stop}", file=sys.stderr)
        seal_run_record(run_record)
        return 4
    done = summary.summarize(every, result.lines, result.store_attempts)
    text = summary.render(done)
    (run_record / "summary.txt").write_text(text, encoding="utf-8")
    seal_run_record(run_record)
    print(text, end="")
    return 0
