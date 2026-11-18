"""The OriginBlame subject, host side (plan 5.5 Check 10; table README, section "Check 10":
the subject ``originblame``, "What a run hands a subject, and its conditions", "The
OriginBlame tie by token range" and "Readings chosen here" 3 and 4; the rows
``c10.<group>.originblame`` of ``check10.toml``).

The tool runs inside the sandbox, one step at a time, through ``baselines.ob_driver``. This
module builds the command of each step (the lock command outside, the sandbox inside, the
copy's own python by ``-m``, every path of the build relative to the build folder, the
tool started in that folder), hands it to a ``runner`` and reads what the driver printed
or wrote. It starts nothing itself and imports nothing of the tool. The three commands of
the tool's own program (``clean``, ``status``, ``generate-set``) are run by the driver; the
program's absolute path is ``ob_binary``, bound read-only in those steps and named to them
by the variable the package itself reads. Where it is not given, ``track_all`` refuses.

What each question is read from, all of it the tool's own records:

- A shard has a store of its own. The plan of a shard holds every record file under
  ``sources/`` and ``gen/`` of its build (one registered section per file) and, per
  document in ordinal order, the section's file, the record of its chunk-file line, the
  shard's packed file and the size the shard's index gives it. One process makes one native
  track call per document, then ``clean`` is run (``track_all``). The run's own statement
  (``run {"processes": 1, "order": "packed"}``) is kept per store. Before the store counts
  as tracked, a PRESENCE check is made: the section hash kept for every file a document is
  tracked under must be present among the sources of the entries (it compares no two hash
  functions: the only hash on the driver's path is the one the tool's own call returned).
  Where one is absent a revoke by it would find nothing and every shard would read "does not
  contain"; it is then a failed attempt.
- The entries are the files under ``.ob/token-index.gpt2`` of a store, read in the order of
  their integer names (``read_store_entries``); the tool has no call that exports them.
- A request about one source is put to a FRESH COPY of the tracked store, by the native
  by-sources call with the sections of the files of the members the request names, called
  once; then ``generate-set``. A fault of Check 4 or Check 5 is put to the entry files of the
  copy of the shard it names, before the revoke, by ``fault_putter`` (the runner's: the
  subject has no fault of its own, and a row that needs one where none is given is
  refused). The forget set is read least significant bit first. A shard reads ``contains``
  when a bit is 1, ``does not contain`` when the store holds at least one entry and no bit
  is 1, ``not covered`` when it holds no entry. The by-sources call or ``generate-set`` on a
  faulted copy that ends with an error of the tool is the reading ``stopped``; it is the
  tool's error only where the driver printed its line for it. Only this code knows that the
  copy holds a planted fault, so the two steps that read it are given the flag the driver
  decides by: on such a copy any RuntimeError of the native call (an OS error and an empty
  text included), the library's class of errors with or without a text, and the program's
  non-zero exit are the tool's line; on any other store an OS error, an empty text and the
  program's error are not. Any other end with a non-zero code (a lock that timed out, a
  sandbox that failed before the driver started, a signal, a panic, a traceback of another
  class, the driver's own failure line) is a failed attempt, never an error of the tool.
- A set (or one layout of it) reads ``contains`` when any shard does, ``does not contain``
  when every shard does not contain, and ``not covered`` otherwise.
- A question about a kind of artifact the tool holds no entry for (a source file, a
  generation output, a release, an import, a mixture, a checkpoint, the listing) reads
  ``cannot_answer`` with the missing text of the table row that names it, and only where
  the tool ran on the store the question is asked on: every shard of that store tracked.
  Where one of them failed, it is that failed ``Attempt``; where none was given or tracked,
  the question is not attempted (``NotAttempted``, a ``ValueError``: no line is written for
  it). Nothing is started for it, and nothing is started for a question whose row records
  none.
- ``c03.spans`` reads the ``token_count`` of the entries, entry k being the document with
  ordinal k, only where the store's run says one process in packed order; otherwise it is
  ``cannot_answer`` with the table's own text of the missing field.
- ``c04.kill-after-100`` and ``c04.retry`` make their own store: one process makes the
  native track calls for the shard and is stopped by its process id after its 100th call
  returns, then ``clean`` and ``status``; for the retry the calls are made again on the
  same store, then ``clean`` and ``status``. The numbers are the ``Entries`` line ``status``
  printed. A stop lands only where the writer ended with the exit of the signal that was sent
  (its negative, or 128 plus its number); any other exit means the fault was not put, and both
  rows are then the failed attempt of the stopped writer, with its exit. The first row's
  reading says whether its entries showed "while the writer ran" or "only at its end".
- A row that names two layouts reads a layout none of whose shards was given or tracked as
  None (not attempted): "cannot answer" comes only after the track of a store.

A step that cannot be read gives a failed ``Attempt`` (a command that did not end by
itself, a shard that did not track, an output of the wrong shape). Local paths are taken
out of its command and output before anything is written.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from replica.shard_index import read_shard_index

from . import companion, dolma, ob_driver, record, sandbox, table10, tokenindex
from .interfaces import Attempt, Question, Reading

SUBJECT = "originblame"
DRIVER_MODULE = ob_driver.MODULE_NAME

KILL_ROW = "c04.kill-after-100"
RETRY_ROW = "c04.retry"
SPANS_ROWS = ("c03.spans.per-ordinal", "c03.spans.total")
COUNTS_ROW = "c09.fixture.shards.counts"
KEPT_ROW = "c09.10k.partitioned.kept-shards"
ALL_REWRITE_ROW = "c09.10k.round-robin.all-rewrite"
ACTION_SUFFIX = ".action"

KILL_AFTER = 100

# The steps that run the tool's own command-line program, and the placeholder for its path.
PROGRAM_VERBS = ("clean", "status", "generate-set")
PROGRAM_NAME = "<program>"
# The flag of the two steps that read a store with a planted fault.
FAULTED_FLAG = ob_driver.FAULTED_FLAG

# What the earlier run's sandbox script set in the environment of its steps besides the folders:
# the language, and no hub reached for.
STEP_FIXED = {"LANG": "C.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# Where the merged entries of a store are (the PID files of the writers sit beside this
# folder, as ``token-index.gpt2.<pid>``) and the status line that gives their number.
ENTRY_FOLDER = (".ob", "token-index.gpt2")
PID_FILE = re.compile(r"token-index\.gpt2\.\d+")
STATUS_ENTRIES = re.compile(r"\bEntries:\s*(\d+)")

# A writer ended by a signal is given as a negative code or as 128 plus the signal's number; a
# stop lands only as the exit of the signal that was sent. What the first row of the stopped
# write says about when its entries showed.
SIGNAL_EXIT_BASE = 128
LAST_SIGNAL = 64
ROWS_SHOWED_WHILE_RUNNING = "while the writer ran"
ROWS_SHOWED_AT_END = "only at its end"

# The shard each fault of the table is put to (the table's own text of each fault).
FAULT_SHARDS = {
    "f.map-deleted": "E/shard-1",
    "f.digest": "E/shard-3",
    "l.count-499": "c04/shard-500",
}

# The row whose missing text a kind of artifact reads, by the table's own references.
MISSING_ROWS = {
    "files": "c10.c05.unmutated.files.originblame",
    "mixtures": "c10.c05.unmutated.mixtures.originblame",
    "checkpoints": "c10.c05.unmutated.checkpoints.originblame",
    "listing": "c10.c05.listing.map-deleted.originblame",
}
SHARD_NAME = re.compile(r"(successor of )?[A-Za-z0-9-]+/shard-\d+")

SPANS_NOTE = "the tool does not tokenize: each count is the one its caller passed"

CONTAINS = "contains"
DOES_NOT_CONTAIN = "does not contain"
NOT_COVERED = "not covered"

# Placeholders for the local roots taken out of a command and its output.
WORK_NAME = "<work>"
BUILD_NAME = "<build>"
ENV_NAME = "<env>"
BIND_NAME = "<interpreters>"


class NotAttempted(ValueError):
    """A question whose store the subject never tracked (none was given for it, or
    ``track_all`` was not called): no line is written for it."""


@dataclass(frozen=True)
class Document:
    """One tracked document: the relative path of the file whose section it is tracked
    under, its size in the shard's own index, the record its chunk-file line carries and the
    shard's packed file (a path relative to the build). A plan is made only of documents
    that carry both of the last two; ``track_plan`` gives them."""

    section: str
    tokens: int
    record: Any = field(default=None, hash=False)
    file: str | None = None


@dataclass(frozen=True)
class ShardSpec:
    """A shard the subject tracks.

    name: the artifact name as the table writes it.
    target: what ``baselines.record.attempt_target`` gives for a question asked of it.
    layout, run: None where the shard has none.
    successor: true for a shard of the rebuilt side.
    build: the folder its files are read from.
    documents: the documents to track, in ordinal order.
    """

    name: str
    target: tuple[str, str]
    layout: str | None
    run: str | None
    successor: bool
    build: Path
    documents: tuple[Document, ...]


# --- The plan of one shard ---------------------------------------------------------------


def id_files(build: Path | str, unparseable: dict[str, int] | None = None) -> dict[str, str]:
    """The file under ``sources/`` or ``gen/`` of a build that holds each record id, as a
    path relative to the build folder. An id two files hold is a ``ValueError``. A line that
    is not JSON holds no record (the line reader of the tool, ``waterwhite.capture.line_text``,
    gives such a line no text): it adds no id, and when ``unparseable`` is handed in it is
    counted there, for the file it is in."""
    base = Path(build)
    found: dict[str, str] = {}
    for path in companion._record_files(base):
        relative = path.relative_to(base).as_posix()
        with open(path, "rb") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                except (ValueError, RecursionError):
                    if unparseable is not None:
                        unparseable[relative] = unparseable.get(relative, 0) + 1
                    continue
                if not isinstance(row, dict):
                    raise ValueError(f"{relative} line {number}: a record is not an object")
                document_id = row.get("id")
                if not document_id:
                    continue
                if document_id in found and found[document_id] != relative:
                    raise ValueError(f"id {document_id!r} is held by {found[document_id]} and {relative}")
                found[document_id] = relative
    return found


class NoSourceFile(ValueError):
    """A packed line of a chunk file carries an id that no record file under ``sources/`` or ``gen/`` holds. A ``ValueError``."""


def track_plan(build, chunk_file: str, idx_prefix, *, ids: Mapping[str, str] | None = None) -> tuple[Document, ...]:
    """The documents of one shard, in ordinal order: line n + 1 of ``chunk_file`` (a path
    relative to ``build``) is the document with ordinal n. A document is tracked with the
    section of the file under ``sources/`` or ``gen/`` that holds the record its line
    carries (a generated record under its generation output, never under the sources it
    was made from) and with the size the shard's own index (``idx_prefix`` plus ``.idx``)
    gives it. A line the index holds as a document of no tokens was never packed and is
    left out of the plan. Each document carries the record its line holds and the shard's packed file
    (``idx_prefix`` plus ``.bin``) as a path relative to ``build``. ``ValueError`` where the
    index holds another number of documents than the file has lines, for an id no file holds
    and for a shard whose files lie outside ``build``. ``ids`` is ``id_files(build)``,
    handed in when many shards of one build are planned."""
    base = Path(build)
    prefix = Path(idx_prefix)
    if prefix.is_absolute():
        try:
            prefix = prefix.relative_to(base)
        except ValueError:
            raise ValueError("the files of the shard lie outside the build folder") from None
    packed_file = f"{prefix.as_posix()}.bin"
    content = (base / chunk_file).read_bytes().decode("utf-8")
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    index = read_shard_index(Path(f"{idx_prefix}.idx"))
    if index.documents != len(lines):
        raise ValueError(f"{chunk_file} has {len(lines)} lines and its shard index holds "
                         f"{index.documents} documents")
    known = id_files(base) if ids is None else ids
    found: list[Document] = []
    for number, line in enumerate(lines, start=1):
        if index.document_tokens[number - 1] == 0:
            # A line that gave no tokens was never packed: the shard's own index holds it as a document of no tokens, and the
            # tool is told of the documents that were packed only.
            continue
        row = json.loads(line)
        document_id = row.get("id") if isinstance(row, dict) else None
        section = known.get(document_id) if isinstance(document_id, str) else None
        if section is None:
            raise NoSourceFile(f"line {number} of {chunk_file}: no file under sources/ or gen/ "
                               f"holds the id {document_id!r}")
        found.append(Document(section=section, tokens=index.document_tokens[number - 1],
                              record=row, file=packed_file))
    return tuple(found)


def read_store_entries(store) -> list[tokenindex.Entry]:
    """The entries of a store: the files under ``.ob/token-index.gpt2``, read in the order
    of their integer names (``2`` before ``10``), each file in its own order. A store with
    no such folder holds no entry. The PID files of the writers (``token-index.gpt2.<pid>``)
    and the binary index beside the folder are not read: a writer's entries count once
    ``clean`` merged them. ``ValueError`` for a name that is no whole number and for an
    entry that is not in the form of ``baselines.tokenindex``."""
    folder = Path(store).joinpath(*ENTRY_FOLDER)
    if not folder.is_dir():
        return []
    files = []
    for path in folder.iterdir():
        if not path.name.isdigit():
            raise ValueError(f"{path.name!r} in the entry folder is not the name of an entry file")
        files.append((int(path.name), path))
    entries: list[tokenindex.Entry] = []
    for _, path in sorted(files):
        entries.extend(tokenindex.read_entries(path))
    return entries


# --- What the table says -----------------------------------------------------------------


@lru_cache(maxsize=1)
def _rows() -> Mapping[str, table10.Row]:
    return {row.id: row for row in table10.load_table10().rows}


def _missing_text(kind: str) -> str:
    return _rows()[MISSING_ROWS[kind]].missing


def _kind_of(artifact: str | None) -> str:
    """The kind of an artifact by the names of the table: a shard, or a kind of which the
    tool holds no entry; ``set`` for a row that names no artifact."""
    if artifact is None:
        return "set"
    if SHARD_NAME.fullmatch(artifact):
        return "shard"
    if artifact == "(listing)":
        return "listing"
    if "/mix-" in artifact:
        return "mixtures"
    if "@" in artifact:
        return "checkpoints"
    return "files"


# The rows for which the table gives no join of the shards' answers (the two rows of ``dolma.OPEN_ROWS``).
NO_JOIN_ROWS = dolma.OPEN_ROWS


def _given(question: Question) -> Mapping[str, Any]:
    return record._listed_givens()[question.listed_row]


def _combined(answers: Sequence[str]) -> str:
    """One word for several shards: ``contains`` when any shard does, ``does not contain``
    when every one does not, and ``not covered`` where a shard holds no entry."""
    if CONTAINS in answers:
        return CONTAINS
    if answers and all(word == DOES_NOT_CONTAIN for word in answers):
        return DOES_NOT_CONTAIN
    return NOT_COVERED


def _action(word: str) -> str:
    """The plan's own definition (4.1): contains is read as rewrite, does not contain as
    keep. A shard with no entry has no action."""
    return {CONTAINS: "rewrite", DOES_NOT_CONTAIN: "keep"}.get(word, NOT_COVERED)


def _ok(result) -> bool:
    code = result.exit_code
    return isinstance(code, int) and not isinstance(code, bool) and code == 0


def _ended(result) -> bool:
    """A command that ended by itself, with an exit code."""
    code = result.exit_code
    return isinstance(code, int) and not isinstance(code, bool)


def _tool_error(result) -> bool:
    """A command that ended with an error of the tool's own: an exit code that is not 0,
    and the driver's line saying a call of the tool raised its own class of errors, with a
    text after the marker. An end with neither that line nor a code of its own (a lock that
    timed out, a sandbox that failed before the driver started, a signal), a traceback of
    another class and a line with no text say nothing about the changed record."""
    return (_ended(result) and result.exit_code != 0
            and bool(_tool_text(result.output)))


def _tool_text(output: str) -> str:
    """What the tool's error said: the driver's line or lines, without the marker."""
    found = [line[len(ob_driver.TOOL_ERROR_MARKER):].strip() for line in output.splitlines()
             if line.startswith(ob_driver.TOOL_ERROR_MARKER)]
    return "\n".join(part for part in found if part)


def _is_whole_number(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class StoppedWriter:
    """What the subject knows of a writer it stopped.

    stopped_exit_code: the exit its ``wait()`` gave.
    stopped_signal: the number of the signal its ``stop()`` said it sent.
    stopped_attempt_id: the id of the failed ``Attempt`` the stop gives where it did not land.
    stopped_command, stopped_output: the command of the writer and what it printed.
    rows_showed_while_running: whether its entry files held the entries while it ran.
    """

    stopped_exit_code: Any
    stopped_signal: Any
    stopped_attempt_id: str
    stopped_command: tuple[str, ...] = ()
    stopped_output: str = ""
    rows_showed_while_running: bool = False


def _stop_that_did_not_land(stopped: StoppedWriter) -> tuple[str, Any, str] | None:
    """None where the stop landed: the exit of the writer is the negative of the signal sent,
    or 128 plus it. Else the id, the exit and the text of the failed attempt of the stopped
    writer, which records the exit and says the stop did not land. ``ValueError`` for an exit
    or a signal that is no whole number, a signal outside 1 to 64, and for a stop that did not
    land with no id for its attempt."""
    code, sent = stopped.stopped_exit_code, stopped.stopped_signal
    if not _is_whole_number(code):
        raise ValueError("the stopped writer gave no exit")
    if not _is_whole_number(sent) or not 1 <= sent <= LAST_SIGNAL:
        raise ValueError("the stop gave no number of the signal it sent")
    if code in (-sent, SIGNAL_EXIT_BASE + sent):
        return None
    if not stopped.stopped_attempt_id:
        raise ValueError("the stop did not land and holds no id for its attempt")
    note = "the stop did not land: the exit of the writer is not that of the signal sent to it"
    return stopped.stopped_attempt_id, code, (stopped.stopped_output + "\n" + note).strip()


def _status_entries(output: str) -> int | None:
    """The number the program's status line ``Entries`` gives, or None where it has none."""
    found = STATUS_ENTRIES.search(output)
    return int(found.group(1)) if found else None


def _statement(output: str) -> dict | None:
    """The run's own statement, the last ``run {...}`` line of a track's output."""
    found = None
    for line in output.splitlines():
        if line.startswith("run "):
            try:
                value = json.loads(line[4:])
            except ValueError:
                continue
            if isinstance(value, dict):
                found = value
    return found


def _one_process_in_packed_order(statement: dict | None) -> bool:
    return (statement is not None and statement.get("processes") == 1
            and statement.get("order") == "packed")


# --- The subject -------------------------------------------------------------------------


class OriginBlame:
    """The ``originblame`` subject: tracks the shards it is given and reads each question
    the table routes to it from fresh copies of the tracked stores.

    ``requests`` maps the request word of a row to the files of the members it names,
    relative to the build folder. ``runner.run(argv)`` gives something with ``exit_code``
    and ``output``; ``runner.start(argv)`` gives a handle with ``pid``, ``readline()``,
    ``stop()`` (it gives the number of the signal it sent) and ``wait()``; the exit code
    ``wait()`` gives for a writer the stop ended is the negative of that number or 128 plus it.

    ``ob_binary`` is the absolute path of the tool's command-line program. It is bound
    read-only in the steps that run it, named to them by the variable the package reads,
    and ``track_all`` refuses where it is None. ``fault_putter(copy, fault_id)`` puts a
    fault of the table to the entry files of a fresh copy (the folder ``copy`` holds
    ``.ob/token-index.gpt2``; ``read_store_entries`` reads it) and is called once per read
    of the faulted shard, before the revoke; a row that needs a fault where it is None is
    refused with a ``ValueError``.

    ``env`` holds the caller's environment variables of a step. Every step also gets
    ``LANG`` ``C.UTF-8``, ``HF_HUB_OFFLINE`` ``1`` and ``TRANSFORMERS_OFFLINE`` ``1``; a value of
    the caller's for one of these, or for a word the step sets itself, that differs from the
    step's is a ``ValueError`` (for the first three at construction, for the others at the
    step, before it is run).
    """

    name = SUBJECT

    def __init__(self, work, env_dir, lock_prefix, runner, shards, requests, *,
                 ob_binary=None, fault_putter=None, env=None):
        # The environment of every step: the caller's variables, and the language and the two
        # offline words the earlier run's sandbox script set. A value of the caller's for one of
        # these three that differs is refused before anything is started.
        self._variables = dict(env or {})
        for name, value in STEP_FIXED.items():
            if self._variables.setdefault(name, value) != value:
                raise ValueError(f"{name} is {value!r} in a step, not {self._variables[name]!r}")
        self.work = Path(work)
        self.env_dir = Path(env_dir)
        self.lock_prefix = list(lock_prefix)
        self.runner = runner
        self.shards = tuple(shards)
        self.requests = dict(requests)
        self.ob_binary = None if ob_binary is None else str(ob_binary)
        self.fault_putter = fault_putter
        self._steps = 0
        self._copies = 0
        self._attempts = 0
        self._tracked: set[int] = set()
        self._failed: dict[int, Attempt] = {}
        self._runs: dict[int, dict | None] = {}
        self._called = False
        self._binds: list[str] | None = None
        self._driver_folder = self.work / "ob-driver"
        self._driver_folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(ob_driver.__file__), self._driver_folder / "ob_driver.py")
        self._scratch = self.work / "ob-tmp"
        self._scratch.mkdir(parents=True, exist_ok=True)
        self._stores = self.work / "ob-stores"
        self._stores.mkdir(parents=True, exist_ok=True)
        self._plans = self.work / "ob-plans"
        self._plans.mkdir(parents=True, exist_ok=True)

    # -- the command of a step -------------------------------------------------------------

    def _roots(self) -> list[tuple[str, str]]:
        found = [(str(self.work), WORK_NAME), (str(self.env_dir), ENV_NAME)]
        found += [(str(spec.build), BUILD_NAME) for spec in self.shards]
        found += [(bind, BIND_NAME) for bind in self._interpreter_binds()]
        if self.ob_binary is not None:
            found.append((self.ob_binary, PROGRAM_NAME))
        return sorted(set(found), key=lambda pair: len(pair[0]), reverse=True)

    def _clean(self, text: str) -> str:
        for root, name in self._roots():
            text = text.replace(root, name)
        return text

    def _interpreter_binds(self) -> list[str]:
        if self._binds is None:
            self._binds = sandbox.interpreter_binds(self.env_dir)
        return self._binds

    def _program(self) -> str:
        if self.ob_binary is None:
            raise ValueError("the path of the tool's command-line program was not given: pass ob_binary")
        return self.ob_binary

    def _argv(self, spec: ShardSpec, verb: str, arguments: Sequence[str]) -> list[str]:
        binds = [self.env_dir, spec.build, *self._interpreter_binds()]
        own = {"PYTHONPATH": str(self._driver_folder), "TMPDIR": str(self._scratch),
               "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}
        if verb in PROGRAM_VERBS:
            program = self._program()
            binds.append(program)
            own[ob_driver.PROGRAM_ENV] = program
        env = dict(self._variables)
        for name, value in own.items():
            if env.setdefault(name, value) != value:
                raise ValueError(f"{name} is {value!r} in a step, not {env[name]!r}")
        self._steps += 1
        home = sandbox.prepare_home(self.work, f"ob-home-{self._steps}")
        command = sandbox.tool_command(self.env_dir, DRIVER_MODULE, [verb, *arguments])
        return sandbox.sandboxed_step(lock_prefix=self.lock_prefix, work=self.work, home=home,
                                      chdir=spec.build, command=command, ro_binds=binds, env=env)

    def _run(self, spec: ShardSpec, verb: str, arguments: Sequence[str]):
        argv = self._argv(spec, verb, arguments)
        return argv, self.runner.run(argv)

    def _attempt_id(self, verb: str) -> str:
        self._attempts += 1
        return f"originblame-{verb}-{self._attempts}"

    def _attempt(self, verb: str, argv: Sequence[str], exit_code, output: str, *,
                 attempt_id: str | None = None) -> Attempt:
        return Attempt(id=attempt_id or self._attempt_id(verb),
                       command=tuple(self._clean(str(word)) for word in argv),
                       exit_code=exit_code, output=self._clean(output))

    def _entries_showed(self, store: Path) -> bool:
        """True where the PID files of the writers held at least ``KILL_AFTER`` entries now:
        the entries showed while the writer ran. Read at the moment the 100th call returned."""
        found = 0
        for name in self._pid_files(store):
            try:
                with open(store / ENTRY_FOLDER[0] / name, "r", encoding="utf-8") as handle:
                    found += sum(1 for line in handle if line.strip())
            except (OSError, ValueError):
                continue
        return found >= KILL_AFTER

    # -- tracking --------------------------------------------------------------------------

    def _index_of(self, spec: ShardSpec) -> int:
        for number, held in enumerate(self.shards):
            if held is spec or held == spec:
                return number
        raise ValueError(f"the subject was not given the shard {spec.name!r}")

    def tracked_store(self, spec: ShardSpec) -> Path:
        """The folder of the tracked store of a shard."""
        return self._stores / f"tracked-{self._index_of(spec)}"

    def _plan_file(self, spec: ShardSpec, name: str) -> Path:
        """The plan of a shard: every record file of its build (one section is registered
        per file), and its documents in ordinal order."""
        documents = []
        for ordinal, item in enumerate(spec.documents):
            if item.record is None or not item.file:
                raise ValueError(f"document {ordinal} of {spec.name!r} has no record or no packed file: "
                                 f"plan the shard with track_plan")
            documents.append({"section": item.section, "tokens": item.tokens,
                              "record": item.record, "file": item.file})
        files = [path.relative_to(spec.build).as_posix() for path in companion._record_files(spec.build)]
        path = self._plans / name
        path.write_text(json.dumps({"files": files, "documents": documents}, sort_keys=True),
                        encoding="utf-8")
        return path

    def _hash_problem(self, spec: ShardSpec, store: Path) -> str | None:
        """A presence check only, no comparison of hash functions: None when the section hash
        the driver kept for every file a document is tracked under is present among the
        sources of the entries of the store; else the text of what is missing."""
        files = list(dict.fromkeys(item.section for item in spec.documents))
        if not files:
            return None
        try:
            meta = json.loads((store / ob_driver.META_FILE).read_text(encoding="utf-8"))
            sections = meta["sections"]
            held = {source for entry in read_store_entries(store) for source in entry.sources}
        except (OSError, ValueError, KeyError, TypeError) as error:
            return f"the section hashes of the store could not be checked: {type(error).__name__}: {error}"
        for path in files:
            if not isinstance(sections, dict) or sections.get(path) not in held:
                return (f"the section hash of {path} was not found among the sources of the "
                        f"tracked entries")
        return None

    def track_all(self) -> tuple[Attempt, ...]:
        """Track every shard once: one process, one call per document in ordinal order,
        then ``clean``, then the check that the section hash of each file a document is
        tracked under is found among the sources of the entries. The failed attempts, in
        shard order; an empty tuple when none."""
        if self._called:
            raise ValueError("the shards were tracked already")
        self._program()
        self._called = True
        failed: list[Attempt] = []
        for number, spec in enumerate(self.shards):
            store = self._stores / f"tracked-{number}"
            plan = self._plan_file(spec, f"track-{number}.json")
            argv, result = self._run(spec, "track", ["--store", str(store), "--plan", str(plan)])
            if not _ok(result):
                self._failed[number] = self._attempt("track", argv, result.exit_code, result.output)
                failed.append(self._failed[number])
                continue
            self._runs[number] = _statement(result.output)
            argv, result = self._run(spec, "clean", ["--store", str(store)])
            if not _ok(result):
                self._failed[number] = self._attempt("clean", argv, result.exit_code, result.output)
                failed.append(self._failed[number])
                continue
            problem = self._hash_problem(spec, store)
            if problem is not None:
                self._failed[number] = self._attempt("section-hash", argv, result.exit_code, problem)
                failed.append(self._failed[number])
                continue
            self._tracked.add(number)
        return tuple(failed)

    # -- a fresh copy of a tracked store ---------------------------------------------------------

    def _fresh_copy(self, number: int) -> Path:
        self._copies += 1
        copy = self._stores / f"copy-{self._copies}"
        shutil.copytree(self._stores / f"tracked-{number}", copy)
        return copy

    def _out_file(self, kind: str) -> Path:
        self._copies += 1
        return self._plans / f"{kind}-{self._copies}"

    def _entries_of(self, store: Path, label: str, argv: Sequence[str] = ()):
        """The entries of a store, read from its files: a list, or the failed attempt (the
        files do not hold entries in the form the tool writes them)."""
        try:
            return read_store_entries(store)
        except (OSError, ValueError) as error:
            return self._attempt(label, argv or [label, str(store)], 0,
                                 f"the entries could not be read: {error}")

    # -- the shard rule -------------------------------------------------------------------------

    def _read_shard(self, number: int, spec: ShardSpec, files: Sequence[str], fault: str | None):
        """``contains``, ``does not contain``, ``not covered``, ``stopped`` or a failed
        attempt, from one fresh copy of the tracked store of the shard."""
        if fault is not None and self.fault_putter is None:
            raise ValueError(f"the row needs the fault {fault!r} put to the entry files: "
                             f"pass fault_putter, nothing is read unfaulted")
        copy = self._fresh_copy(number)
        if fault is not None:
            try:
                self.fault_putter(copy, fault)
            except (OSError, ValueError) as error:
                return self._attempt("fault", ["fault", fault], None,
                                     f"the fault could not be put: {type(error).__name__}: {error}")
        # Only here is it known whether the store holds a planted fault; the two steps that read
        # it are told so, and the driver tells the tool's stop from a failure by that.
        told = [FAULTED_FLAG] if fault is not None else []
        argv, result = self._run(spec, "revoke", ["--store", str(copy), *told, *files])
        if not _ok(result):
            if fault is not None and _tool_error(result):
                return "stopped"
            return self._attempt("revoke", argv, result.exit_code, result.output)
        bits_file = self._out_file("bits")
        argv, result = self._run(spec, "generate-set",
                                 ["--store", str(copy), "--out", str(bits_file), *told])
        if not _ok(result):
            if fault is not None and _tool_error(result):
                return "stopped"
            return self._attempt("generate-set", argv, result.exit_code, result.output)
        entries = self._entries_of(copy, "entries", argv)
        if isinstance(entries, Attempt):
            return entries
        try:
            bits = tokenindex.read_forget_set(bits_file, len(entries))
        except (OSError, ValueError) as error:
            return self._attempt("generate-set", argv, 0, f"the forget set could not be read: {error}")
        if any(bits):
            return CONTAINS
        return DOES_NOT_CONTAIN if entries else NOT_COVERED

    # -- which shards a question names ----------------------------------------------------------

    def _select(self, question: Question, given: Mapping[str, Any], layout: str | None,
                kind: str) -> list[tuple[int, ShardSpec]]:
        target = record.attempt_target(question)
        if kind == "shard":
            found = [(n, s) for n, s in enumerate(self.shards)
                     if s.name == given["artifact"] and s.target == target]
            if len(found) > 1:
                raise ValueError(f"{len(found)} shards named {given['artifact']!r} on {target}")
            return found
        wanted_layout = layout if layout is not None else given.get("layout")
        wanted_successor = "successor" in question.listed_row
        found = []
        for number, spec in enumerate(self.shards):
            if spec.target != target or spec.successor != wanted_successor:
                continue
            if wanted_layout is not None and spec.layout != wanted_layout:
                continue
            if given.get("run") is not None and spec.run != given["run"]:
                continue
            found.append((number, spec))
        return found

    def _failed_of(self, picked: Sequence[tuple[int, ShardSpec]]) -> Attempt | None:
        for number, spec in picked:
            if number in self._failed:
                return self._failed[number]
            if number not in self._tracked:
                raise NotAttempted(f"the shard {spec.name!r} was not tracked: call track_all first")
        return None

    def _cannot_of_kind(self, question: Question, kind: str) -> Reading | Attempt:
        """The reading ``cannot_answer`` for a kind the tool holds no entry for, where the
        tool ran on the store the question is asked on: every shard of it tracked. Where one
        failed, its failed attempt; where none was given or tracked, ``NotAttempted``."""
        target = record.attempt_target(question)
        held = [(number, spec) for number, spec in enumerate(self.shards) if spec.target == target]
        if not held:
            raise NotAttempted(f"the subject was given no shard on {target}: the tool did not run "
                               f"for {question.listed_row}")
        failed = self._failed_of(held)
        if failed is not None:
            return failed
        return self._cannot(question, _missing_text(kind))

    # -- reading a question -----------------------------------------------------------------------

    def read(self, question: Question) -> Reading | Attempt:
        """A Reading, or the failed Attempt. A row that records none has the reading ``raw``
        None, and nothing is started for it. A row that names two layouts is read by
        ``read_by_layout``."""
        if question.route is None:
            return Reading(question_id=question.id, raw=None, route=None)
        if record.layouts_of(question) is not None:
            raise ValueError(f"{question.listed_row} names two layouts: use read_by_layout")
        return self._read_one(question, None)

    def read_by_layout(self, question: Question) -> dict[str, Reading | Attempt | None]:
        """The reading of a row that names two layouts, one per layout. A layout none of whose
        shards was given or tracked is None: not attempted, never "cannot answer" (a tool that
        did not run cannot earn it); a layout whose track failed is its failed attempt. The
        line of the row follows ``baselines.expected.combine_layouts``."""
        layouts = record.layouts_of(question)
        if layouts is None:
            raise ValueError(f"{question.listed_row} names no layouts")
        if question.route is None:
            return {layout: Reading(question_id=question.id, raw=None, route=None) for layout in layouts}
        found: dict[str, Reading | Attempt | None] = {}
        for layout in layouts:
            try:
                found[layout] = self._read_one(question, layout)
            except NotAttempted:
                found[layout] = None
            except ValueError:
                if question.listed_row not in NO_JOIN_ROWS:
                    raise
                # The layout was tracked and the table gives no join of its shards: ITS OWN reading is cannot answer, so that
                # the line of the row follows the layouts' slots (a layout with no run is None, not attempted; a failed attempt
                # comes first) and no cannot-answer line is put for a layout the tool never ran on.
                found[layout] = self._cannot(question, dolma.NO_JOIN_TEXT)
        return found

    def _reading(self, question: Question, raw: Any) -> Reading:
        return Reading(question_id=question.id, raw=raw, route=question.route)

    def _cannot(self, question: Question, text: str) -> Reading:
        return self._reading(question, {"cannot_answer": text})

    def _read_one(self, question: Question, layout: str | None) -> Reading | Attempt:
        given = _given(question)
        listed = question.listed_row
        if listed in SPANS_ROWS:
            return self._read_spans(question, given, layout)
        if listed in (KILL_ROW, RETRY_ROW):
            return self._read_interrupted(question, given)
        kind = "set" if layout is not None else _kind_of(given.get("artifact"))
        if kind not in ("shard", "set"):
            return self._cannot_of_kind(question, kind)
        picked = self._select(question, given, layout, kind)
        if not picked:
            raise NotAttempted(f"the subject was given no shard for {question.listed_row} on "
                               f"{record.attempt_target(question)}")
        failed = self._failed_of(picked)
        if failed is not None:
            return failed
        if listed in NO_JOIN_ROWS:
            # The table reads a shard (the forget set of the store that tracked THE shard) and gives no fold of several shards
            # into one answer; the driver's own "any shard contains" is no line of the table (ruled class 3, 17 Nov 2026,
            # by rule). The runner puts the line cannot answer with the table's words, as it does for Dolma.
            raise ValueError(f"{listed}: the table gives no join of the shards' answers")
        files = self.requests[given["request"]]
        state = given.get("state")
        answers: list[tuple[ShardSpec, str]] = []
        for number, spec in picked:
            fault = state if state in FAULT_SHARDS and FAULT_SHARDS[state] == spec.name else None
            got = self._read_shard(number, spec, list(files), fault)
            if isinstance(got, Attempt):
                return got
            answers.append((spec, got))
        return self._shaped(question, given, kind, answers)

    def _shaped(self, question: Question, given: Mapping[str, Any], kind: str,
                answers: list[tuple[ShardSpec, str]]) -> Reading:
        listed = question.listed_row
        words = [word for _, word in answers]
        if kind == "shard" and words == ["stopped"]:
            return self._reading(question, {"stopped": True})
        if listed.endswith(ACTION_SUFFIX):
            return self._reading(question, {"action": _action(words[0])})
        if listed in (COUNTS_ROW, KEPT_ROW):
            keep = [s.name for s, w in answers if _action(w) == "keep"]
            rewrite = [s.name for s, w in answers if _action(w) == "rewrite"]
            if listed == COUNTS_ROW:
                return self._reading(question, {"keep": keep, "rewrite": rewrite,
                                                "counts": {"keep": len(keep), "rewrite": len(rewrite)}})
            return self._reading(question, {"keep": keep, "rewrite": rewrite})
        if listed == ALL_REWRITE_ROW:
            keep = sum(1 for _, w in answers if _action(w) == "keep")
            rewrite = sum(1 for _, w in answers if _action(w) == "rewrite")
            return self._reading(question, {"keep": keep, "rewrite": rewrite})
        if "stopped" in words:
            return self._reading(question, {"stopped": True})
        return self._reading(question, {"answer": _combined(words)})

    # -- c03.spans --------------------------------------------------------------------------------

    def _read_spans(self, question: Question, given: Mapping[str, Any], layout: str | None) -> Reading | Attempt:
        picked = self._select(question, given, layout, "set")
        if not picked:
            raise NotAttempted(f"the subject was given no shard for {question.listed_row} on "
                               f"{record.attempt_target(question)}")
        failed = self._failed_of(picked)
        if failed is not None:
            return failed
        if not all(_one_process_in_packed_order(self._runs.get(number)) for number, _ in picked):
            return self._cannot(question, _rows()[question.row_id].tie["otherwise"])
        counts: dict[str, list[int]] = {}
        for number, spec in picked:
            # The entries are read where they lie; nothing is started and nothing is changed.
            entries = self._entries_of(self.tracked_store(spec), "entries")
            if isinstance(entries, Attempt):
                return entries
            counts[spec.name] = [entry.token_count for entry in entries]
        if question.listed_row == SPANS_ROWS[0]:
            return self._reading(question, {"per_ordinal": counts, "notes": [SPANS_NOTE]})
        return self._reading(question, {"per_shard": {name: sum(found) for name, found in counts.items()},
                                        "notes": [SPANS_NOTE]})

    # -- c04: the writer stopped, and run again -----------------------------------------------------

    @staticmethod
    def _pid_files(store: Path) -> list[str]:
        """The PID files the writers left in a store, by name, before ``clean`` merges them."""
        try:
            names = os.listdir(store / ENTRY_FOLDER[0])
        except OSError:
            return []
        return sorted(name for name in names if PID_FILE.fullmatch(name))

    def _status_of(self, spec: ShardSpec, store: Path):
        """The PID files of a store, then ``clean`` and ``status``: the number the status
        line ``Entries`` gives and the names of the PID files, or a failed attempt."""
        names = self._pid_files(store)
        argv, result = self._run(spec, "clean", ["--store", str(store)])
        if not _ok(result):
            return self._attempt("clean", argv, result.exit_code, result.output)
        argv, result = self._run(spec, "status", ["--store", str(store)])
        if not _ok(result):
            return self._attempt("status", argv, result.exit_code, result.output)
        entries = _status_entries(result.output)
        if entries is None:
            return self._attempt("status", argv, result.exit_code,
                                 f"the status has no Entries line: {result.output}")
        return {"entries": entries, "pid_files": names}

    def _read_interrupted(self, question: Question, given: Mapping[str, Any]) -> Reading | Attempt:
        picked = self._select(question, given, None, "shard")
        if not picked:
            raise NotAttempted(f"the subject was given no shard {given['artifact']!r} on "
                               f"{record.attempt_target(question)}")
        number, spec = picked[0]
        if number in self._failed:
            # The tool refused this shard's own track: a failed attempt to run it is no answer, and no writer is
            # started on a shard the tool refused (first real pass, finding 1).
            return self._failed[number]
        self._copies += 1
        store = self._stores / f"write-{self._copies}"
        plan = self._plan_file(spec, f"write-{self._copies}.json")
        argv = self._argv(spec, "track", ["--store", str(store), "--plan", str(plan)])
        handle = self.runner.start(argv)
        seen: list[str] = []
        for call in range(1, KILL_AFTER + 1):
            line = handle.readline()
            seen.append(line)
            if not line.startswith(f"tracked {call}"):
                ended = handle.wait()
                return self._attempt("track", argv, ended.exit_code, "".join(seen) + ended.output)
        showed = self._entries_showed(store)
        sent = handle.stop()
        ended = handle.wait()
        writer = StoppedWriter(
            stopped_exit_code=ended.exit_code, stopped_signal=sent,
            stopped_attempt_id=self._attempt_id("stop"), stopped_command=tuple(str(w) for w in argv),
            stopped_output="".join(seen) + (ended.output or ""), rows_showed_while_running=showed)
        not_landed = _stop_that_did_not_land(writer)
        if not_landed is not None:
            # The writer ended by itself, by an exit that is no signal's or by another signal where
            # the stop was meant to end it: the fault was not put, and the store says nothing
            # about an interrupted write. Both rows are the failed attempt of the stopped writer.
            attempt_id, code, text = not_landed
            return self._attempt("stop", writer.stopped_command, code, text, attempt_id=attempt_id)
        stopped = self._status_of(spec, store)
        if isinstance(stopped, Attempt):
            return stopped
        if question.listed_row == KILL_ROW:
            words = ROWS_SHOWED_WHILE_RUNNING if writer.rows_showed_while_running else ROWS_SHOWED_AT_END
            return self._reading(question, {"records": {
                "rows_after_stop": stopped["entries"], "pid_files": stopped["pid_files"],
                "rows_showed": words}})
        argv, result = self._run(spec, "track", ["--store", str(store), "--plan", str(plan)])
        if not (_ok(result) or _tool_error(result)):
            return self._attempt("track", argv, result.exit_code, result.output)
        refused = not _ok(result)
        again = self._status_of(spec, store)
        if isinstance(again, Attempt):
            return again
        mismatch = self._clean(_tool_text(result.output)) if refused else None
        return self._reading(question, {"records": {
            "rows_after_retry": again["entries"], "refused": refused, "matches_shard": None,
            "mismatch": mismatch, "pid_files": again["pid_files"]}})
