"""The OriginBlame subject (plan 5.5 Check 10; table README, section "Check 10": the
subject ``originblame``, its conditions, "The OriginBlame tie by token range" and
"Readings chosen here" 3 and 4; the rows ``c10.<group>.originblame`` of
``check10.toml``).

Nothing of the tool is started. The tool runs inside the sandbox through a driver
script; every test here feeds hand-written records and a stand-in for the driver
(``FakeTool`` below), which keeps its store as the files the tool leaves (the entry
files under ``.ob/token-index.gpt2``, the PID files beside them, a file of section
hashes), models the hazards the README names (a second revoke of one target un-revokes
it; two writers in one sandbox can name the same token-index file) and returns what the
real driver would print. The names of the tool's own calls are the ones of the installed
package (``register_section`` of the public module; ``init``, ``author_add``, ``track``
and ``token_index_revoke_by_sources`` of its native module) and of the baseline run's
report (sections 3.1 and 5); the stand-ins of the tool in the driver tests mirror their
signatures. ``clean``, ``status`` and ``generate-set`` are the commands of the tool's
own command-line program, which the driver starts.

What this file asks of ``baselines.originblame``:

- ``Document(section, tokens, record, file)``: one tracked document: the relative path of
  the file whose section it is tracked under, its size in the shard's own index, the
  record its chunk-file line carries and the shard's packed file (relative to the build).
  ``record`` and ``file`` default to None; a plan is refused for a document without them.
- ``read_store_entries(store)``: the entries of a store as the files under
  ``.ob/token-index.gpt2`` hold them, the files read in the order of their integer
  names. ``NotAttempted`` (a ``ValueError``): a question whose store was never tracked.
- ``track_plan(build, chunk_file, idx_prefix)``: the documents of one shard, in ordinal
  order. ``chunk_file`` is a path relative to ``build`` (its line n + 1 is the document
  with ordinal n), ``idx_prefix`` the shard's own files without suffix. A document is
  tracked with the section of the file under ``sources/`` or ``gen/`` that holds the
  record its line carries (the generated record under its generation output, never
  under the sources it was made from), with the size its index gives it. A shard whose
  index holds another number of documents than the file has lines, and an id no file
  holds, are a ``ValueError``: a gap is never read as an absence.
- ``ShardSpec(name, target, layout, run, successor, build, documents)``: a shard the
  subject tracks. ``name`` is the artifact name as the table writes it, ``target`` is
  what ``baselines.record.attempt_target`` gives for a question asked of it, ``layout``
  and ``run`` are None where the shard has none, ``successor`` is true for a shard of
  the rebuilt side, ``build`` the folder its files are read from.
- ``OriginBlame(work, env_dir, lock_prefix, runner, shards, requests, *, ob_binary,
  fault_putter)``: a ``Subject`` named ``originblame``. ``requests`` maps the request word
  of a row to the relative files of the members it names. ``ob_binary`` is the absolute
  path of the tool's command-line program (bound read-only in the steps that run it and
  named to them by the variable the package itself reads); ``track_all`` refuses with a
  ``ValueError`` naming it where it is not given. ``fault_putter(copy, fault_id)`` puts a
  fault of the table to the entry files of a fresh copy before the revoke; a row that
  needs a fault where none is given is refused with a ``ValueError`` naming it, never read
  unfaulted. ``runner.run(argv)`` gives something with ``exit_code`` and ``output``;
  ``runner.start(argv)`` gives a handle with ``pid``, ``readline()`` (a line, "" at the
  end), ``stop()`` (stops it by its process id and gives the number of the signal it sent)
  and ``wait()``. A stop lands only where ``wait()`` gives the negative of that number or
  128 plus it; any other exit means the fault was not put, and both rows of the stopped
  write are the failed ``Attempt`` of the stopped writer with its exit and "the stop did not
  land" in its output. An exit or a signal that is no whole number (a signal of 1 to 64) is a
  ``ValueError``. The reading of the first row carries ``records["rows_showed"]``: "while the
  writer ran" where its entry files held 100 entries when the 100th call had returned, else
  "only at its end".
  - ``track_all()``: tracks every shard once, one process, one call per document in
    ordinal order, then ``clean``, then checks that the section hash of every file a
    document was tracked under is present among the sources of the entries (a presence
    check, no comparison of two hash functions); returns the failed ``Attempt``s
    (an empty tuple when none). ``tracked_store(shard)`` is the folder of the tracked
    store of a shard.
  - ``read(question)``: a ``Reading`` (its ``route`` is the question's route) or a
    failed ``Attempt``; a question whose row records none has the reading ``raw`` None
    and nothing is started for it. ``read_by_layout(question)`` gives the same per
    layout for a row that names two, and None for a layout none of whose shards was given
    or tracked (not attempted; a layout whose track failed is its failed ``Attempt``)
    (``read`` refuses such a row with ``ValueError``
    when the row has a route; a row that records none is read by ``read`` whatever
    layouts it names).

What this file asks of ``baselines.ob_driver`` (run inside the sandbox; stdlib only at
module level; the tool's package is a parameter ``ob`` of ``main`` and its native module
a parameter ``native``, never imported there):

- ``MODULE_NAME`` is ``ob_driver`` (the dotted name ``sandbox.tool_command`` is given) and
  ``TOOL_MODULE`` is ``ob``.
- ``main(argv, *, ob=None, native=None, program=None) -> int``: ``argv[0]`` is the verb,
  then ``--store`` (the folder of the store) and the options of that verb: ``track --plan
  FILE`` (a JSON file ``{"files": [..], "documents": [{"section", "tokens", "record",
  "file"}]}``), ``revoke FILE...`` (the native by-sources call with a set of hashes, once
  per store), and the three commands of the tool's program, run in the folder of the
  store: ``clean``, ``status`` and ``generate-set --out FILE`` (``program(arguments, cwd)``
  gives the exit code and the output). ``track`` prints ``tracked K`` after the Kth call
  returns, then one line ``run {"processes": 1, "order": "packed"}``. There is no
  ``export`` and no ``fault`` verb: the entries are files, and a fault is put to them by
  the runner.
- ``ob``, the tool as the driver is handed it, has ``register_section(path, authors,
  license, year, contributors=None, ob_dir=None)``, ``ODError`` (the class of the
  library's own errors) and ``_ob_native`` with ``init(ob_dir)``, ``author_add(name,
  email, ob_dir)``, ``track(record, file, sections, ob_dir, token_count, tokenizer)`` and
  ``token_index_revoke_by_sources(ob_dir, tokenizer, sources)``.
- The native module raises RuntimeError, with the text of the error behind it, for every error
  of its own; the library's own class (``ODError``) derives from Exception, not from
  RuntimeError. A native track or by-sources call that raises a RuntimeError with a text, or
  the library's class with a text, ends the driver with a non-zero code after one line that
  starts with ``TOOL_ERROR_MARKER``: the tool's line. A text that carries an OS error
  ("(os error N)") is the tool's line only where the step is given ``--faulted`` (a store with
  a planted fault: ``revoke`` and ``generate-set`` take the flag); the program's non-zero
  exit is the tool's line on such a store too. Every other end (an error about the call
  itself, an error of another class, a subclass of RuntimeError, an error with no text, a
  signal, a program that ended with an error on a store with no fault) starts its line with
  ``FAILURE_MARKER`` and names the raw text; a panic of the native module ends the process
  with its traceback. The host reads a command of the subject that ends with an error as
  ``stopped`` (a faulted store) or ``refused`` (the write made again) only where the tool's
  line is there, with a text; any other end with a non-zero code (a lock that timed out, a
  sandbox that failed before the driver started, a signal, a stop of the driver's own) is a
  failed attempt.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import inspect
import io
import json
import os
import re
import shutil
import struct
import sys
from collections import namedtuple
from pathlib import Path

import pytest

MODULE = "baselines.originblame"
DRIVER = "baselines.ob_driver"
SUBJECT = "originblame"
DRIVER_WORD = "ob_driver"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
EM_DASH = chr(0x2014)
LOCAL_MARKERS = ("/ho" + "me/", "Proj" + "ects/")
SESSION_WORDS = ("C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor")
REVIEW_WORDS = ("R2" + "A", "ROUND" + "2", "rul" + "ing")

LOCK = ["stand-in-lock", "--held"]
REQUESTS = {"withdraw A": ("sources/A.jsonl",), "withdraw C": ("sources/C.jsonl",)}
BINARY = "/stand-in/bin/ob"
PROGRAM_VERBS = ("clean", "status", "generate-set")
DRIVER_VERBS = ("track", "revoke", *PROGRAM_VERBS)
PROGRAM_VARIABLE = "OB_RUST_BIN"
PACKED = "pipeline/partitioned/shards/shard-1.bin"
# The artifact the table's row of the successors of a set names: the successor shards of both layouts.
EACH_SUCCESSOR = "each successor shard"
# The record files every build of these tests holds, in the order the subject lists them:
# the files under sources/ first, then those under gen/.
BUILD_FILES = ("sources/A.jsonl", "sources/B.jsonl", "sources/C.jsonl", "sources/Z.jsonl", "gen/P.jsonl")
ENTRY_FOLDER = (".ob", "token-index.gpt2")
# What the tool's native calls raise, as RuntimeError with the text of the error behind it (the
# native module maps every error through one function that makes a RuntimeError of its text):
# the text of a serde_json parse error on the track path, of a serde_json write error on the
# by-sources path, and of an io::Error (the text ends in "(os error N)").
TRACK_TOOL_TEXT = "expected value at line 1 column 6"
REVOKE_TOOL_TEXT = "key must be a string"
OS_ERROR_TEXT = "No such file or directory (os error 2)"
# What the program prints when the folder it is run in is no store (its own source, the
# ``clean`` and ``generate-set`` commands), and the exit code that ends it.
PROGRAM_ERROR_TEXT = "Error: .ob/ not found. Run `ob init` first.\n"
PROGRAM_ERROR_EXIT = 1
Result = namedtuple("Result", "exit_code output")

# The routed questions of the subject, counted by hand from the groups of the table: the group
# and the number of its questions (its listed rows, each at the sizes it is asked at).
ROUTED_BY_HAND = (
    ("c03.spans", 4),
    ("c04.interrupted-write", 2),
    ("c04.count-499", 1),
    ("c05.unmutated.shards", 6),
    ("c05.f.map-deleted", 24),
    ("c05.f.digest", 25),
    ("c05.withdraw-C.unmutated", 2),
    ("c06.downstream", 6),
    ("c09.artifact-actions", 8),
    ("c09.successors", 3),
)
ROUTED_TOTAL = 81

# The text each rule of the table gives for a kind of artifact the tool holds no entry for.
MISSING_ROW = {
    "files": "c10.c05.unmutated.files.originblame",
    "mixtures": "c10.c05.unmutated.mixtures.originblame",
    "checkpoints": "c10.c05.unmutated.checkpoints.originblame",
    "listing": "c10.c05.listing.map-deleted.originblame",
}
SHARD_NAME = re.compile(r"(successor of )?[A-Za-z0-9-]+/shard-\d+")
WITHDRAW_A = "withdraw A"


# --- Hand-written records ------------------------------------------------------------


def doc(ob, section: str, tokens: int):
    """A tracked document. Where ``Document`` carries a record and a packed file, both are
    given: the record of the line and the shard's packed file."""
    held = {field.name for field in dataclasses.fields(ob.Document)}
    extra = {}
    if "record" in held:
        extra["record"] = {"id": f"{Path(section).stem}-{tokens}", "text": f"t {section}"}
    if "file" in held:
        extra["file"] = PACKED
    return ob.Document(section=section, tokens=tokens, **extra)


def plain_documents(ob, with_a: bool = False, with_c: bool = False):
    """Three documents of a shard that holds no member of A or C; ``with_a`` and
    ``with_c`` add one document tracked under the file of that member."""
    found = [doc(ob, "sources/B.jsonl", 4), doc(ob, "sources/Z.jsonl", 5), doc(ob, "sources/B.jsonl", 3)]
    if with_a:
        found.insert(1, doc(ob, "sources/A.jsonl", 6))
    if with_c:
        found.append(doc(ob, "sources/C.jsonl", 7))
    return tuple(found)


def shard(ob, tmp_path, name, documents, *, target=("store", "main"), layout=None, run=None,
          successor=False):
    build = tmp_path / "build"
    build.mkdir(parents=True, exist_ok=True)
    for relative in BUILD_FILES:
        if not (build / relative).exists():
            write_records(build / relative, [Path(relative).stem + "-001"])
    return ob.ShardSpec(name=name, target=target, layout=layout, run=run, successor=successor,
                        build=build, documents=tuple(documents))


def write_shard_files(prefix: Path, documents: list[list[int]]) -> None:
    """A shard in the layout the index format has: one sequence per document that holds
    tokens, a document boundary after each document."""
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for item in documents:
        if item:
            pointers.append(2 * len(tokens))
            lengths.append(len(item))
            tokens.extend(item)
        boundaries.append(len(lengths))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    Path(f"{prefix}.idx").write_bytes(
        b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 8, len(lengths), len(boundaries))
        + struct.pack(f"<{len(lengths)}i", *lengths)
        + struct.pack(f"<{len(pointers)}q", *pointers)
        + struct.pack(f"<{len(boundaries)}q", *boundaries))


def write_records(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "text": "t " + i}) + "\n" for i in ids),
                    encoding="utf-8")


def make_env(tmp_path: Path, work: Path) -> Path:
    """A copied environment inside the work folder shaped like the real ones: its
    ``bin/python`` is a link to a plain file in an interpreter folder outside the work
    folder. Nothing here starts it."""
    interpreters = tmp_path / "uv-python"
    real = interpreters / "cpython-3.12.13-linux-x86_64-gnu"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python3.12").write_bytes(b"a stand-in for an interpreter binary\n")
    (real / "lib").mkdir()
    os.symlink(real.name, interpreters / "cpython-3.12-linux-x86_64-gnu")
    copy = work / "venv-ob"
    (copy / "bin").mkdir(parents=True)
    os.symlink(str(interpreters / "cpython-3.12-linux-x86_64-gnu" / "bin" / "python3.12"),
               copy / "bin" / "python")
    return copy


def tree_digest(folder: Path) -> dict[str, bytes | str]:
    out: dict[str, bytes | str] = {}
    for current, dirs, files in os.walk(folder):
        for name in dirs:
            out[str(Path(current, name).relative_to(folder))] = "dir"
        for name in files:
            out[str(Path(current, name).relative_to(folder))] = Path(current, name).read_bytes()
    return out


# --- The stand-in for the sandboxed driver ---------------------------------------------


class Handle:
    """A started ``track``: one line per call that returned; ``stop`` ends it."""

    def __init__(self, tool, store: Path, documents: list[dict], pid_name: str):
        self.tool, self.store, self.documents, self.pid_name = tool, store, documents, pid_name
        self.pid = 4242
        self.sent = 0
        self.stopped = False
        self.lines_read = 0

    def readline(self) -> str:
        if self.stopped:
            return ""
        if self.sent < len(self.documents):
            item = self.documents[self.sent]
            if not self.tool.write_at_end:
                self.tool.append_entry(self.store, self.pid_name, item)
            self.sent += 1
            self.lines_read += 1
            return f"tracked {self.sent}\n"
        if self.sent == len(self.documents):
            self.sent += 1
            self.lines_read += 1
            return "run " + json.dumps(self.tool.run_statement()) + "\n"
        return ""

    def stop(self):
        """Stops the writer by its process id and gives the number of the signal sent."""
        self.stopped = True
        self.tool.stops.append(self.pid)
        if self.tool.write_at_end:
            # A writer that puts its entries to the file only when it ends.
            for item in self.documents[:self.sent]:
                self.tool.append_entry(self.store, self.pid_name, item)
        self.tool.mark_partial(self.store, self.pid_name)
        return self.tool.stop_signal

    def wait(self) -> Result:
        if self.stopped:
            return Result(self.tool.stop_exit, self.tool.stop_output)
        return Result(0, "")


class FakeTool:
    """The sandboxed driver, as the files the tool leaves: a store holds ``driver_meta.json``
    (the section hash of each registered file), one PID file ``.ob/token-index.gpt2.<name>``
    per writer until ``clean`` merges it into ``.ob/token-index.gpt2/<NNN>``, and
    ``state.json`` (the stand-in's own bookkeeping). Options set what the tool does:
    ``processes`` (the writers of one run), ``same_pid_file`` (a second run names the file of
    the first), ``drop_partial`` (``clean`` drops the entries of a stopped run),
    ``error_after`` (fault ids whose store makes ``revoke`` end with an error of the tool,
    which the driver reports on its own line), ``refuse_second_track`` ((code, text): the
    second track ends the same way), ``fail`` (verb to (exit code, output): an end the driver
    did not write, as a lock or a sandbox that fails before it starts), ``fail_track_in``
    (folder names of the stores whose track fails so), ``say_run``, ``pid_names`` (the names
    the writers give their files, in turn), ``stop_signal`` (the number of the signal the
    handle's ``stop`` says it sent), ``stop_exit`` and ``stop_output`` (the end of a writer
    that was stopped), ``write_at_end`` (a writer that puts its entries to its file only when
    it ends), ``entry_hash`` (the prefix of the section hash the entries carry; the file of
    hashes holds ``h:``) and ``status_format`` (what ``status`` prints; ``{n}`` the number of
    entries, ``{a}`` the active ones, ``{r}`` the revoked ones, ``{t}`` their tokens).

    It answers only the verbs and the forms of output the real thing has. The driver has the
    verbs ``track``, ``revoke``, ``clean``, ``status`` and ``generate-set``; any other verb is
    refused as the driver refuses it, with its failure line and the exit code 2. The program's
    ``status --tokenizer`` prints ``Tokenizer: <name>``, ``Entries: <n> (<a> active, <r>
    revoked)`` and ``Total tokens: <t>``; its ``clean`` and ``generate-set`` print the lines the
    program's own source gives them."""

    def __init__(self, *, processes=1, same_pid_file=False, drop_partial=False,
                 error_after=(), fail=None, say_run=True, pid_names=("2", "3", "4"),
                 refuse_second_track=None, stop_exit=-15, stop_signal=15, stop_output="",
                 write_at_end=False, entry_hash="h:", fail_track_in=(),
                 status_format="Tokenizer: gpt2\nEntries: {n} ({a} active, {r} revoked)\nTotal tokens: {t}\n"):
        self.processes, self.same_pid_file, self.drop_partial = processes, same_pid_file, drop_partial
        self.error_after, self.fail, self.say_run = set(error_after), dict(fail or {}), say_run
        self.pid_names = list(pid_names)
        self.refuse_second_track = refuse_second_track
        self.stop_exit, self.stop_signal, self.stop_output = stop_exit, stop_signal, stop_output
        self.write_at_end, self.entry_hash = write_at_end, entry_hash
        self.fail_track_in, self.status_format = set(fail_track_in), status_format
        self.calls: list[dict] = []
        self.stops: list[int] = []
        self.starts = 0
        self.local_text: list[str] = []

    # -- argv -----------------------------------------------------------------------

    def parse(self, argv):
        argv = list(argv)
        assert argv[: len(LOCK)] == LOCK, "a step must start with the lock prefix"
        index = argv.index(DRIVER_WORD)
        verb, rest = argv[index + 1], argv[index + 2:]
        options, positional, at = {}, [], 0
        while at < len(rest):
            if rest[at] in ("--store", "--plan", "--out"):
                options[rest[at][2:]] = rest[at + 1]
                at += 2
            elif rest[at] == "--faulted":
                options["faulted"] = True
                at += 1
            else:
                positional.append(rest[at])
                at += 1
        return verb, options, positional

    # -- the files of a store -------------------------------------------------------

    @staticmethod
    def load(store: Path) -> dict:
        path = store / "state.json"
        if not path.exists():
            return {"order": [], "merged": {}, "revokes": 0, "faulted": None, "partial": None,
                    "tracks": 0}
        return json.loads(path.read_text())

    @staticmethod
    def save(store: Path, state: dict) -> None:
        store.mkdir(parents=True, exist_ok=True)
        (store / "state.json").write_text(json.dumps(state, sort_keys=True))

    @staticmethod
    def pid_path(store: Path, name: str) -> Path:
        return store / ".ob" / f"token-index.gpt2.{name}"

    @staticmethod
    def read_rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def entry_files(store: Path) -> list[Path]:
        folder = store.joinpath(*ENTRY_FOLDER)
        if not folder.is_dir():
            return []
        return sorted(folder.iterdir(), key=lambda path: int(path.name))

    def entries(self, store: Path) -> list[dict]:
        return [row for path in self.entry_files(store) for row in self.read_rows(path)]

    def rewrite_entries(self, store: Path, rows: list[dict]) -> None:
        for path in self.entry_files(store):
            path.unlink()
        if rows:
            self.write_rows(store.joinpath(*ENTRY_FOLDER) / "001", rows)

    @staticmethod
    def ensure_meta(store: Path, plan: dict) -> None:
        path = store / "driver_meta.json"
        if not path.exists():
            store.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"phase": "track", "sections": {name: "h:" + name for name in plan.get("files", [])}},
                sort_keys=True))

    def run_statement(self) -> dict:
        return {"processes": self.processes, "order": "packed"}

    def next_pid_name(self, state: dict) -> str:
        if self.same_pid_file and state["order"]:
            return state["order"][0]
        return self.pid_names[min(state["tracks"], len(self.pid_names) - 1)]

    def entry(self, item: dict) -> dict:
        return {"sources": [self.entry_hash + item["section"]], "token_count": item["tokens"],
                "tokenizer": "gpt2", "revoked": False}

    def append_entry(self, store: Path, pid_name: str, item: dict) -> None:
        path = self.pid_path(store, pid_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.entry(item), sort_keys=True) + "\n")

    def mark_partial(self, store: Path, pid_name: str) -> None:
        state = self.load(store)
        state["partial"] = pid_name
        self.save(store, state)

    # -- verbs ----------------------------------------------------------------------

    def start(self, argv) -> Handle:
        verb, options, _ = self.parse(argv)
        assert verb == "track"
        store = Path(options["store"])
        self.starts += 1
        self.calls.append({"verb": verb, "store": store, "options": options, "positional": [],
                           "argv": list(argv), "started": True})
        plan = json.loads(Path(options["plan"]).read_text())
        self.ensure_meta(store, plan)
        state = self.load(store)
        name = self.next_pid_name(state)
        if name not in state["order"]:
            state["order"].append(name)
        state["tracks"] += 1
        self.save(store, state)
        path = self.pid_path(store, name)
        if self.same_pid_file or not path.exists():
            self.write_rows(path, [])
        return Handle(self, store, plan["documents"], name)

    def run(self, argv) -> Result:
        verb, options, positional = self.parse(argv)
        store = Path(options["store"])
        self.calls.append({"verb": verb, "store": store, "options": options,
                           "positional": positional, "argv": list(argv),
                           "plan": (json.loads(Path(options["plan"]).read_text())
                                    if "plan" in options else None)})
        if verb not in DRIVER_VERBS:
            # What the driver does with a verb it does not have: its failure line, exit code 2.
            return self.tool_failure(2, f"unknown verb {verb}")
        if verb in self.fail:
            code, text = self.fail[verb]
            return Result(code, text.replace("{store}", str(store)))
        state = self.load(store)
        if verb == "track":
            return self.do_track(store, state, options)
        if verb == "clean":
            return self.do_clean(store, state)
        if verb == "status":
            rows = self.entries(store)
            revoked = sum(1 for row in rows if row["revoked"])
            return Result(0, self.status_format.format(
                n=len(rows), a=len(rows) - revoked, r=revoked, t=sum(row["token_count"] for row in rows)))
        if verb == "revoke":
            return self.do_revoke(store, state, positional, bool(options.get("faulted")))
        return self.do_generate(store, options["out"])

    @staticmethod
    def tool_failure(code: int, text: str) -> Result:
        """An end the driver itself makes: its failure line, then the code."""
        import importlib

        marker = getattr(importlib.import_module(DRIVER), "FAILURE_MARKER", None)
        assert marker, f"{DRIVER} has no FAILURE_MARKER"
        return Result(code, f"{marker} {text}\n")

    def put_fault(self, store: Path, fault_id: str) -> None:
        """The fault of the table put to the entry files of a store, as the runner does it."""
        from baselines import faults

        record = {"sections": [], "documents": [], "entries": self.entries(store)}
        changed = faults.apply_fault(fault_id, "originblame", record)
        self.rewrite_entries(store, changed["entries"])
        state = self.load(store)
        state["faulted"] = fault_id
        self.save(store, state)

    def do_track(self, store, state, options) -> Result:
        plan = json.loads(Path(options["plan"]).read_text())
        if store.name in self.fail_track_in:
            return Result(1, "sandbox failed before the driver started\n")
        if self.refuse_second_track is not None and state["tracks"] >= 1:
            return self.tool_error(*self.refuse_second_track)
        self.ensure_meta(store, plan)
        name = self.next_pid_name(state)
        if name not in state["order"]:
            state["order"].append(name)
        state["tracks"] += 1
        self.save(store, state)
        path = self.pid_path(store, name)
        rows = [] if self.same_pid_file or not path.exists() else self.read_rows(path)
        lines = []
        for number, item in enumerate(plan["documents"], start=1):
            rows.append(self.entry(item))
            lines.append(f"tracked {number}")
        self.write_rows(path, rows)
        if self.say_run:
            lines.append("run " + json.dumps(self.run_statement()))
        return Result(0, "\n".join(lines) + "\n")

    def do_clean(self, store, state) -> Result:
        merged_files = merged_entries = 0
        for name in list(state["order"]):
            path = self.pid_path(store, name)
            if not path.exists():
                continue
            if self.drop_partial and state.get("partial") == name:
                path.unlink()
                state["partial"] = None
                continue
            merged = state["merged"].setdefault(name, f"{len(state['merged']) + 1:03d}")
            rows = self.read_rows(path)
            self.write_rows(store.joinpath(*ENTRY_FOLDER) / merged, rows)
            path.unlink()
            merged_files += 1
            merged_entries += len(rows)
        self.save(store, state)
        # The lines the program's own source prints for ``clean``.
        lines = ["Merged 0 document records from PID files", "Deleted 0 PID files"]
        if merged_entries:
            lines += [f"Merged {merged_entries} token-index entries",
                      f"Deleted {merged_files} token-index PID files"]
        lines.append("Archived 0 revoked records")
        return Result(0, "\n".join(lines) + "\n")

    @staticmethod
    def tool_error(code: int, text: str, call: str = "track") -> Result:
        """An end by an error of the tool's native call, as the driver prints it: its line for
        a RuntimeError of the call (``<call> raised RuntimeError: <text>``), then the code."""
        import importlib

        marker = getattr(importlib.import_module(DRIVER), "TOOL_ERROR_MARKER", None)
        assert marker, f"{DRIVER} has no TOOL_ERROR_MARKER"
        return Result(code, f"{marker} {call} raised RuntimeError: {text.rstrip()}\n")

    def do_revoke(self, store, state, files, faulted=False) -> Result:
        if state["faulted"] in self.error_after:
            # The by-sources call of the native module ends in an io error (its text ends in
            # "(os error N)"): the tool's stop on a store with a planted fault (the step is told
            # so), the driver's own line on any other.
            call = "token_index_revoke_by_sources"
            if faulted:
                return self.tool_error(5, OS_ERROR_TEXT, call)
            return self.tool_failure(6, f"{call} raised RuntimeError: {OS_ERROR_TEXT}")
        sections = json.loads((store / "driver_meta.json").read_text())["sections"]
        wanted = {sections.get(path, "h:" + path) for path in files}
        for path in self.entry_files(store):
            rows = self.read_rows(path)
            for row in rows:
                if wanted & set(row["sources"]):
                    row["revoked"] = not row["revoked"]
            self.write_rows(path, rows)
        state["revokes"] += 1
        self.save(store, state)
        return Result(0, "revoked\n")

    def do_generate(self, store, out) -> Result:
        flags = [row["revoked"] for row in self.entries(store)]
        data = bytearray((len(flags) + 7) // 8)
        for place, flag in enumerate(flags):
            if flag:
                data[place >> 3] |= 1 << (place & 7)
        Path(out).write_bytes(bytes(data))
        # The line the program's own source prints for ``generate-set``.
        return Result(0, f"Forget set written to {out} ({len(data)} bytes, {sum(flags)} revoked entries)\n")


class FaultPutter:
    """What the runner hands the subject: the fault put to the entry files of a fresh copy,
    with the table's own ``faults`` module, before the revoke."""

    def __init__(self, tool: FakeTool):
        self.tool = tool
        self.puts: list[tuple[Path, str]] = []

    def __call__(self, copy: Path, fault_id: str) -> None:
        self.puts.append((copy, fault_id))
        self.tool.calls.append({"verb": "fault", "store": copy, "options": {"fault": fault_id},
                                "positional": [], "argv": []})
        self.tool.put_fault(copy, fault_id)


class FakeRunner:
    def __init__(self, tool: FakeTool):
        self.tool = tool

    def run(self, argv, **_):
        return self.tool.run(argv)

    def start(self, argv, **_):
        return self.tool.start(argv)


class DriverRunner(FakeRunner):
    """A runner that sends some verbs through the real driver, in this process, over a tool whose
    native call raises what the native module raises (``revoke_error``, ``track_error``) or a
    program that ends as given (``program``: the exit code and the output of ``generate-set``).
    What the driver prints and its exit code are what the host reads. A panic of the native
    module ends the process with its traceback and the exit code 1. The other verbs go to the
    stand-in of the driver."""

    def __init__(self, tool, driver, *, revoke_error=None, track_error=None, program=None):
        super().__init__(tool)
        self.driver, self.program = driver, program
        self.revoke_error, self.track_error = revoke_error, track_error

    def run(self, argv, **_):
        verb, options, positional = self.tool.parse(argv)
        store = options["store"]
        flag = ["--faulted"] if options.get("faulted") else []
        if verb == "revoke" and self.revoke_error is not None:
            args, kept = ["revoke", "--store", store, *flag, *positional], {"ob": raising_ob(self.revoke_error)}
        elif verb == "track" and self.track_error is not None:
            args = ["track", "--store", store, "--plan", options["plan"]]
            kept = {"ob": raising_ob(self.track_error, call="track")}
        elif verb == "generate-set" and self.program is not None:
            args = ["generate-set", "--store", store, "--out", options["out"], *flag]
            kept = {"program": lambda arguments, cwd: self.program}
        else:
            return self.tool.run(argv)
        self.tool.calls.append({"verb": verb, "store": Path(store), "options": options,
                                "positional": positional, "argv": list(argv), "plan": None})
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                code = self.driver.main(args, **kept)
        except PanicException as error:
            return Result(1, out.getvalue() + "Traceback (most recent call last):\n"
                             f"pyo3_runtime.PanicException: {error}\n")
        return Result(code, out.getvalue())


def calls_of(tool: FakeTool, verb: str, store: Path | None = None) -> list[dict]:
    return [c for c in tool.calls if c["verb"] == verb and (store is None or c["store"] == store)]


def make_subject(need_module, tmp_path, shards, tool=None, requests=REQUESTS, *, binary=True, putter=True,
                 step_env=None, runner_of=None):
    """The subject over stand-ins. Where the subject takes the path of the tool's program
    and a fault putter they are given (``binary`` and ``putter`` False leave them out);
    ``step_env`` is the caller's environment variables of a step, given where it takes them;
    ``runner_of(tool)`` makes the runner (the stand-in's by default)."""
    ob = need_module(MODULE)
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    env = make_env(tmp_path, work)
    tool = tool or FakeTool()
    accepted = inspect.signature(ob.OriginBlame).parameters
    extra = {}
    if binary and "ob_binary" in accepted:
        extra["ob_binary"] = BINARY
    if putter and "fault_putter" in accepted:
        extra["fault_putter"] = FaultPutter(tool)
    if step_env is not None and "env" in accepted:
        extra["env"] = dict(step_env)
    runner = FakeRunner(tool) if runner_of is None else runner_of(tool)
    subject = ob.OriginBlame(work=work, env_dir=env, lock_prefix=LOCK, runner=runner,
                             shards=tuple(shards), requests=dict(requests), **extra)
    return subject, tool


# --- The questions of the table ---------------------------------------------------------


def ob_questions(need_module):
    """Every question put to the subject, by (listed row, size)."""
    table = need_module("baselines.table10")
    found = {}
    for question in table.expand(table.load_table10()):
        if question.subject == SUBJECT:
            found[(question.listed_row, question.size)] = question
    return found


def question_of(need_module, listed_row, size=None):
    return ob_questions(need_module)[(listed_row, size)]


def expectation(need_module, listed_row, size=None):
    expected = need_module("baselines.expected")
    return expected.expected_for(listed_row, size).expect


def graded(need_module, question, reading, *, expect=None):
    grade = need_module("baselines.grade")
    expect = expectation(need_module, question.listed_row, question.size) if expect is None else expect
    return grade.grade(question, expect, reading)


def artifact_kind(artifact) -> str:
    """The kind of an artifact, by the names of the table: a shard, or one of the kinds
    of which the tool holds no entry. A row of a set that names "each successor shard" is the
    successor shards of each of its layouts: a set."""
    if artifact is None or artifact == EACH_SUCCESSOR:
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


def missing_text(facts, kind: str) -> str:
    return facts.row_by_id[MISSING_ROW[kind]]["missing"]


def shards_for(ob, tmp_path, facts, question, *, with_a=False):
    """The shards a question needs, written from the table's own ``given`` of its listed
    row: one shard for an artifact that is a shard, three for a set or a run."""
    given = facts.core_given[question.listed_row]
    record = ob_module_record()
    target = record.attempt_target(question)
    successor = "successor" in question.listed_row
    artifact = given.get("artifact")
    if artifact is not None and artifact_kind(artifact) == "shard":
        return [shard(ob, tmp_path, artifact, plain_documents(ob, with_a=with_a), target=target,
                      layout=given.get("layout"), run=None, successor=successor)]
    if artifact_kind(artifact) != "set":
        return []
    layouts = list(given.get("layouts") or [given.get("layout")])
    found = []
    for layout in layouts:
        for number in (1, 2, 3):
            name = f"{layout or 'set'}/shard-{number}" if given.get("run") is None else f"{given['run']}/shard-{number}"
            found.append(shard(ob, tmp_path, name, plain_documents(ob, with_a=with_a and number == 3),
                               target=target, layout=layout, run=given.get("run"),
                               successor=successor))
    return found


def ob_module_record():
    import importlib

    return importlib.import_module("baselines.record")


# --- The stand-in holds what it says -------------------------------------------------------


def test_the_stand_in_refuses_a_step_that_is_not_behind_the_lock(tmp_path):
    # The known-bad input is the step itself: no lock prefix in front.
    tool = FakeTool()
    with pytest.raises(AssertionError):
        tool.run(["bwrap", "--", "python", "-m", DRIVER_WORD, "status", "--store", str(tmp_path)])


def test_the_stand_in_refuses_a_verb_the_driver_does_not_have_as_the_driver_does(tmp_path):
    # export and fault are no verbs of the driver: its failure line and the exit code 2, and
    # nothing is made of the store.
    tool = FakeTool()
    for verb in ("export", "fault", "format-everything"):
        got = tool.run(LOCK + [DRIVER_WORD, verb, "--store", str(tmp_path / "s"), "--out", str(tmp_path / "o")])
        assert got.exit_code == 2 and got.output.startswith("ob_driver failure:"), verb
    assert not (tmp_path / "s").exists()


class MostSignificantFirstTool(FakeTool):
    """A stand-in that writes the forget set with the wrong bit order: the known-bad input
    of the check below."""

    def do_generate(self, store, out) -> Result:
        flags = [row["revoked"] for row in self.entries(store)]
        data = bytearray((len(flags) + 7) // 8)
        for place, flag in enumerate(flags):
            if flag:
                data[place >> 3] |= 0x80 >> (place & 7)
        Path(out).write_bytes(bytes(data))
        return Result(0, "")


def stand_in_bits(tool, tmp_path) -> tuple[bytes, bytes]:
    """The forget set after one revoke of the file of A, then after a second one, of a
    store of ten entries whose last entry is the one of A."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    store = tmp_path / "s"
    plan = tmp_path / "plan.json"
    items = [{"section": "sources/B.jsonl", "tokens": 1, "record": {"id": "r"}, "file": PACKED}] * 9 + [
        {"section": "sources/A.jsonl", "tokens": 1, "record": {"id": "r"}, "file": PACKED}]
    plan.write_text(json.dumps({"files": ["sources/A.jsonl", "sources/B.jsonl"], "documents": items}))
    tool.run(LOCK + [DRIVER_WORD, "track", "--store", str(store), "--plan", str(plan)])
    tool.run(LOCK + [DRIVER_WORD, "clean", "--store", str(store)])
    out = tmp_path / "bits"
    found = []
    for _ in range(2):
        tool.run(LOCK + [DRIVER_WORD, "revoke", "--store", str(store), "sources/A.jsonl"])
        tool.run(LOCK + [DRIVER_WORD, "generate-set", "--store", str(store), "--out", str(out)])
        found.append(out.read_bytes())
    return found[0], found[1]


def test_the_stand_in_un_revokes_on_a_second_call_and_writes_the_bits_least_significant_first(tmp_path):
    assert stand_in_bits(FakeTool(), tmp_path / "a") == (bytes([0x00, 0x02]), bytes([0x00, 0x00]))
    # Entry 9 is bit 1 of byte 1; a stand-in that counts from the other end is told apart.
    assert stand_in_bits(MostSignificantFirstTool(), tmp_path / "b")[0] == bytes([0x00, 0x40])


# --- The two modules ------------------------------------------------------------------------


def test_both_modules_exist_and_hold_the_names_asked(need_module):
    ob = need_module(MODULE)
    driver = need_module(DRIVER)
    for name in ("Document", "ShardSpec", "OriginBlame", "track_plan", "read_store_entries", "NotAttempted"):
        assert hasattr(ob, name), f"{MODULE} has no {name}"
    assert hasattr(driver, "main") and callable(driver.main)
    assert driver.MODULE_NAME == DRIVER_WORD
    # The package is imported by its own name, and the program by the variable the package reads.
    assert driver.TOOL_MODULE == "ob" and driver.PROGRAM_ENV == PROGRAM_VARIABLE


def test_the_subject_is_a_subject_named_originblame(need_module, tmp_path):
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    subject, _ = make_subject(need_module, tmp_path, [shard(ob, tmp_path, "E/shard-1", plain_documents(ob))])
    assert isinstance(subject, interfaces.Subject)
    assert subject.name == SUBJECT == interfaces.SUBJECTS[3]


def _source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_the_modules_import_nothing_of_the_tool_or_the_world_helpers(need_module):
    ob = need_module(MODULE)
    driver = need_module(DRIVER)
    for module in (ob, driver):
        tree = ast.parse(_source_of(module))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert not roots & FORBIDDEN_IMPORT_ROOTS, f"{module.__name__} imports {roots & FORBIDDEN_IMPORT_ROOTS}"
    # The driver runs inside the sandbox with the copy's python and nothing of this repository
    # on its path: at module level it imports the standard library only, and not even a
    # package of this repository inside a function that the tests below reach.
    top = [n for n in ast.parse(_source_of(driver)).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names - set(sys.stdlib_module_names)
    assert "baselines" not in names and "replica" not in names and "ob" not in names


def test_the_modules_hold_no_dash_no_local_path_and_no_name_of_this_build(need_module):
    for module in (need_module(MODULE), need_module(DRIVER)):
        text = _source_of(module)
        assert EM_DASH not in text
        for word in LOCAL_MARKERS + SESSION_WORDS + REVIEW_WORDS:
            assert word not in text, f"{module.__name__} holds {word!r}"


# --- The plan of one shard -------------------------------------------------------------------


def _build_folder(tmp_path):
    build = tmp_path / "pb"
    write_records(build / "sources" / "A.jsonl", ["A-001", "A-002"])
    write_records(build / "sources" / "B.jsonl", ["B-001", "B-002"])
    write_records(build / "gen" / "P.jsonl", ["P-001"])
    return build


def test_a_plan_gives_the_file_of_each_line_and_the_size_its_index_gives_in_ordinal_order(need_module, tmp_path):
    ob = need_module(MODULE)
    build = _build_folder(tmp_path)
    chunk = build / "pipeline" / "partitioned" / "tokenizer-input" / "shard-1.jsonl"
    # Line 1 is a record of B, line 2 a generated record (made from A), line 3 a record of A.
    write_records(chunk, ["B-002", "P-001", "A-001"])
    write_shard_files(build / "pipeline" / "partitioned" / "shards" / "shard-1",
                      [[11, 12, 13, 99], [21, 99], [31, 32, 99]])
    found = ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-1.jsonl",
                          build / "pipeline" / "partitioned" / "shards" / "shard-1")
    assert [(d.section, d.tokens) for d in found] == [
        ("sources/B.jsonl", 4), ("gen/P.jsonl", 2), ("sources/A.jsonl", 3)]


def test_a_plan_refuses_a_shard_of_another_length_and_an_id_no_file_holds(need_module, tmp_path):
    ob = need_module(MODULE)
    build = _build_folder(tmp_path)
    chunk = build / "pipeline" / "partitioned" / "tokenizer-input" / "shard-2.jsonl"
    prefix = build / "pipeline" / "partitioned" / "shards" / "shard-2"
    write_records(chunk, ["B-001", "A-002"])
    write_shard_files(prefix, [[1, 99], [2, 99], [3, 99]])
    with pytest.raises(ValueError):
        ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-2.jsonl", prefix)
    write_records(chunk, ["B-001", "NOPE-1"])
    write_shard_files(prefix, [[1, 99], [2, 99]])
    with pytest.raises(ValueError):
        ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-2.jsonl", prefix)


def test_a_plan_names_the_packed_documents_only_as_the_shards_own_index_says(need_module, tmp_path):
    """A line that gave no tokens was never packed (the planted empty-text line of the store scratch-notokens): the shard's own
    index holds it as a document of no tokens, and the tool is told of the documents that were packed only. The packed
    documents are read from the index, never from the plant list. A document that holds tokens and whose id no file holds is
    still refused (the test above)."""
    ob = need_module(MODULE)
    build = _build_folder(tmp_path)
    chunk = build / "pipeline" / "partitioned" / "tokenizer-input" / "shard-4.jsonl"
    prefix = build / "pipeline" / "partitioned" / "shards" / "shard-4"
    write_records(chunk, ["B-001", "NOPE-1", "A-002"])
    write_shard_files(prefix, [[1, 99], [], [3, 99]])
    found = ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-4.jsonl", prefix)
    assert [getattr(d, "record", {}).get("id") for d in found] == ["B-001", "A-002"], "the line of no tokens was planned"
    assert [getattr(d, "tokens", None) for d in found] == [2, 2]
    # The tokens of each packed document are the index's own number for that line, so the ordinal of a line is kept.
    write_records(chunk, ["B-001", "B-002", "A-002"])
    write_shard_files(prefix, [[1, 99], [], [3, 99]])
    assert [d.record["id"] for d in ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-4.jsonl", prefix)] == [
        "B-001", "A-002"], "a known id with no tokens was planned"


def test_a_plan_carries_the_record_of_each_line_and_the_packed_file_relative_to_the_build(need_module, tmp_path):
    ob = need_module(MODULE)
    build = _build_folder(tmp_path)
    chunk = build / "pipeline" / "partitioned" / "tokenizer-input" / "shard-3.jsonl"
    prefix = build / "pipeline" / "partitioned" / "shards" / "shard-3"
    write_records(chunk, ["B-002", "A-001"])
    write_shard_files(prefix, [[11, 99], [21, 22, 99]])
    found = ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-3.jsonl", prefix)
    assert [getattr(d, "record", None) for d in found] == [
        {"id": "B-002", "text": "t B-002"}, {"id": "A-001", "text": "t A-001"}]
    assert [getattr(d, "file", None) for d in found] == ["pipeline/partitioned/shards/shard-3.bin"] * 2
    # A shard whose files lie outside the build has no path relative to it: refused.
    elsewhere = tmp_path / "elsewhere" / "shard-3"
    write_shard_files(elsewhere, [[11, 99], [21, 22, 99]])
    with pytest.raises(ValueError):
        ob.track_plan(build, "pipeline/partitioned/tokenizer-input/shard-3.jsonl", elsewhere)


# --- The driver, through a stand-in for the tool -----------------------------------------------


class ToolError(Exception):
    """The stand-in's class of the library's own errors (the package's ``ODError``)."""


class StandInNative:
    """The tool's native module, with the signatures the installed package gives. It
    records each call."""

    def __init__(self, log):
        self.log = log

    def init(self, ob_dir):
        self.log.append(("init", ob_dir))

    def author_add(self, name, email, ob_dir):
        self.log.append(("author_add", name, email, ob_dir))
        return "id-" + name

    def track(self, record, file, sections, ob_dir, token_count, tokenizer):
        self.log.append(("track", record, file, sections, ob_dir, token_count, tokenizer))
        return "line-hash"

    def token_index_revoke_by_sources(self, ob_dir, tokenizer, sources):
        self.log.append(("revoke_by_sources", ob_dir, tokenizer, sources))
        return len(sources)


class StandInOb:
    """The tool's public module: ``register_section`` as the installed package gives it, the
    class of its own errors and its native module."""

    ODError = ToolError

    def __init__(self):
        self.log: list[tuple] = []
        self._ob_native = StandInNative(self.log)

    def register_section(self, path, authors, license, year, contributors=None, ob_dir=None):
        self.log.append(("register_section", path, authors, license, year, ob_dir))
        return "h:" + path


PLAN_FILES = ["sources/A.jsonl", "sources/B.jsonl", "gen/P.jsonl"]


def write_plan(path: Path, files=tuple(PLAN_FILES), sections=("sources/B.jsonl", "sources/A.jsonl", "sources/B.jsonl"),
               tokens=(4, 6, 3)) -> list[dict]:
    """A plan file: the files to register and one document per section given."""
    items = [{"section": section, "tokens": count, "record": {"id": f"r{number}", "text": section},
              "file": PACKED}
             for number, (section, count) in enumerate(zip(sections, tokens), start=1)]
    path.write_text(json.dumps({"files": list(files), "documents": items}))
    return items


def need_options(driver, *names) -> None:
    held = inspect.signature(driver.main).parameters
    for name in names:
        assert name in held, f"{DRIVER}.main takes no {name}"


def tracked_by_driver(driver, tmp_path: Path, name: str) -> str:
    """A store made by the driver's own track verb over the stand-in; its folder."""
    plan = tmp_path / f"plan-{name}.json"
    write_plan(plan)
    store = str(tmp_path / name)
    assert driver.main(["track", "--store", store, "--plan", str(plan)], ob=StandInOb()) == 0
    return store


def test_the_driver_makes_the_calls_of_the_installed_package_one_registration_per_file_then_one_track_per_document(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    assert getattr(driver, "TOOL_MODULE", None) == "ob"
    plan = tmp_path / "plan.json"
    items = write_plan(plan)
    store = str(tmp_path / "s")
    standin = StandInOb()
    assert driver.main(["track", "--store", store, "--plan", str(plan)], ob=standin) == 0
    assert standin.log == [
        ("init", store),
        ("author_add", "authorA", "authorA@example.invalid", store),
        ("author_add", "authorB", "authorB@example.invalid", store),
        ("author_add", "authorP", "authorP@example.invalid", store),
        ("register_section", "sources/A.jsonl", ["authorA"], "CC-BY-4.0", "2026", store),
        ("register_section", "sources/B.jsonl", ["authorB"], "CC-BY-4.0", "2026", store),
        ("register_section", "gen/P.jsonl", ["authorP"], "CC-BY-4.0", "2026", store),
        ("track", items[0]["record"], PACKED, ["h:sources/B.jsonl"], store, 4, "gpt2"),
        ("track", items[1]["record"], PACKED, ["h:sources/A.jsonl"], store, 6, "gpt2"),
        ("track", items[2]["record"], PACKED, ["h:sources/B.jsonl"], store, 3, "gpt2")]
    saved = json.loads((tmp_path / "s" / "driver_meta.json").read_text())
    assert saved["sections"] == {name: "h:" + name for name in PLAN_FILES}
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == ["tracked 1", "tracked 2", "tracked 3"]
    assert lines[3].startswith("run ") and json.loads(lines[3][4:]) == {"processes": 1, "order": "packed"}


def test_the_write_made_again_on_the_same_store_makes_no_setup_call_and_uses_the_hashes_of_the_first(need_module, tmp_path):
    driver = need_module(DRIVER)
    store = tracked_by_driver(driver, tmp_path, "s")
    again = StandInOb()
    assert driver.main(["track", "--store", store, "--plan", str(tmp_path / "plan-s.json")], ob=again) == 0
    assert [entry[0] for entry in again.log] == ["track", "track", "track"]
    assert again.log[0][3] == ["h:sources/B.jsonl"] and again.log[1][3] == ["h:sources/A.jsonl"]


def test_a_document_under_a_file_that_was_not_registered_is_a_stop_of_the_drivers_own_and_no_track_is_made(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    plan = tmp_path / "plan.json"
    write_plan(plan, files=("sources/A.jsonl",))
    standin = StandInOb()
    code = driver.main(["track", "--store", str(tmp_path / "s"), "--plan", str(plan)], ob=standin)
    assert code != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) and "sources/B.jsonl" in line for line in lines)
    assert not any(line.startswith(driver.TOOL_ERROR_MARKER) for line in lines)
    assert not [entry for entry in standin.log if entry[0] == "track"]


def test_the_driver_revokes_with_a_set_of_the_hashes_of_the_files_once_per_store(need_module, tmp_path):
    driver = need_module(DRIVER)
    store = tracked_by_driver(driver, tmp_path, "s")
    standin = StandInOb()
    assert driver.main(["revoke", "--store", store, "sources/A.jsonl", "gen/P.jsonl"], ob=standin) == 0
    assert standin.log == [("revoke_by_sources", store, "gpt2", {"h:sources/A.jsonl", "h:gen/P.jsonl"})]
    assert type(standin.log[0][3]) is set
    again = StandInOb()
    assert driver.main(["revoke", "--store", store, "sources/A.jsonl"], ob=again) != 0
    assert again.log == []


def test_a_file_that_has_no_registered_section_is_a_stop_of_the_drivers_own_before_any_call(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    store = tracked_by_driver(driver, tmp_path, "s")
    capsys.readouterr()
    standin = StandInOb()
    assert driver.main(["revoke", "--store", store, "sources/NOPE.jsonl"], ob=standin) != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) and "sources/NOPE.jsonl" in line for line in lines)
    assert standin.log == []


def test_the_driver_runs_the_programs_commands_in_the_folder_of_the_store_and_has_no_other_verb(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    need_options(driver, "program")
    store = str(tmp_path / "s")
    out = str(tmp_path / "out")
    ran = []

    def program(arguments, cwd):
        ran.append((list(arguments), cwd))
        return 0, "Entries: 3\n"

    assert driver.main(["clean", "--store", store], program=program) == 0
    assert driver.main(["status", "--store", store], program=program) == 0
    assert driver.main(["generate-set", "--store", store, "--out", out], program=program) == 0
    assert ran == [(["clean"], store), (["status", "--tokenizer", "gpt2"], store),
                   (["generate-set", "--tokenizer", "gpt2", "-o", out], store)]
    assert "Entries: 3" in capsys.readouterr().out
    standin = StandInOb()
    for verb in ("export", "fault", "format-everything"):
        assert driver.main([verb, "--store", store, "--out", out], ob=standin) != 0
    assert standin.log == []


def test_a_program_that_ends_with_an_error_is_the_drivers_line_and_on_a_faulted_store_the_tools(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    need_options(driver, "program")
    out = str(tmp_path / "out")
    ends = lambda arguments, cwd: (PROGRAM_ERROR_EXIT, PROGRAM_ERROR_TEXT)
    # An end of the program with an error on a store with no planted fault: the driver's line.
    code = driver.main(["clean", "--store", str(tmp_path / "s")], program=ends)
    assert code != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) and "Error: .ob/ not found." in line for line in lines)
    assert not any(line.startswith(driver.TOOL_ERROR_MARKER) for line in lines)
    code = driver.main(["generate-set", "--store", str(tmp_path / "s"), "--out", out], program=ends)
    assert code != 0 and not any(
        line.startswith(driver.TOOL_ERROR_MARKER) for line in capsys.readouterr().out.splitlines())
    # On a store with a planted fault (the flag), the same end is the tool's stop: its line,
    # with the raw text. A signal (a negative code) and the driver's own paths stay the driver's.
    code = driver.main(["generate-set", "--store", str(tmp_path / "s"), "--out", out, "--faulted"], program=ends)
    assert code != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.TOOL_ERROR_MARKER) and "Error: .ob/ not found." in line for line in lines)
    assert not any(line.startswith(driver.FAILURE_MARKER) for line in lines)
    code = driver.main(["generate-set", "--store", str(tmp_path / "s"), "--out", out, "--faulted"],
                       program=lambda arguments, cwd: (-9, ""))
    lines = capsys.readouterr().out.splitlines()
    assert code != 0 and any(line.startswith(driver.FAILURE_MARKER) for line in lines)
    assert not any(line.startswith(driver.TOOL_ERROR_MARKER) for line in lines)
    # The flag is for the two steps that read a faulted store.
    assert driver.main(["clean", "--store", str(tmp_path / "s"), "--faulted"], program=ends) != 0
    assert any(line.startswith(driver.FAILURE_MARKER) for line in capsys.readouterr().out.splitlines())


def test_the_program_is_named_by_the_variable_the_package_reads_and_is_not_started_without_it(need_module, tmp_path, monkeypatch, capsys):
    driver = need_module(DRIVER)
    assert getattr(driver, "PROGRAM_ENV", None) == PROGRAM_VARIABLE
    monkeypatch.delenv(PROGRAM_VARIABLE, raising=False)
    assert driver.main(["clean", "--store", str(tmp_path / "s")]) != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) and PROGRAM_VARIABLE in line for line in lines)


class PanicException(BaseException):
    """The stand-in of the native module's panic: the exception the binding layer raises when
    Rust code panics derives from BaseException, so that it passes ``except Exception``."""


def raising_ob(error, *, call="token_index_revoke_by_sources") -> StandInOb:
    """A tool whose native ``call`` raises ``error``: the known-bad inputs of the error line.
    The native module raises RuntimeError for every error of its own."""
    standin = StandInOb()

    def raiser(*arguments):
        raise error

    setattr(standin._ob_native, call, raiser)
    return standin


def test_only_the_tools_own_class_with_a_text_is_the_line_for_a_tool_error_every_other_end_is_the_drivers_with_the_raw_text(need_module, tmp_path, capsys):
    # Three channels end a native call with the tool's line: a RuntimeError of the native call with
    # a text; on a store with a planted fault (``--faulted``) that includes one that carries an OS
    # error ("(os error N)"); and the library's own error class with a text. Every other end is
    # the driver's line with the raw text.
    driver = need_module(DRIVER)
    tool_marker = getattr(driver, "TOOL_ERROR_MARKER", None)
    failure_marker = getattr(driver, "FAILURE_MARKER", None)
    assert tool_marker and failure_marker, "the driver names its two lines"
    assert not tool_marker.startswith(failure_marker) and not failure_marker.startswith(tool_marker)
    count = []

    def end_of(error, call, faulted=False):
        name = f"n{len(count)}"
        count.append(name)
        store = tracked_by_driver(driver, tmp_path, name)
        capsys.readouterr()
        if call == "track":
            args = ["track", "--store", store, "--plan", str(tmp_path / f"plan-{name}.json")]
        else:
            args = ["revoke", "--store", store, *(["--faulted"] if faulted else []), "sources/A.jsonl"]
        code = driver.main(args, ob=raising_ob(error, call=call))
        assert isinstance(code, int) and code != 0
        return capsys.readouterr().out.splitlines()

    def is_tool_line(lines, text):
        return any(line.startswith(tool_marker) and text in line for line in lines)

    def is_driver_line(lines, text):
        return (any(line.startswith(failure_marker) and text in line for line in lines)
                and not any(line.startswith(tool_marker) for line in lines))

    revoke, track = "token_index_revoke_by_sources", "track"
    # (a) A RuntimeError of the native call with a text is the tool's line, on both calls.
    for text, call in ((REVOKE_TOOL_TEXT, revoke), (TRACK_TOOL_TEXT, track)):
        lines = end_of(RuntimeError(text), call)
        assert is_tool_line(lines, text), (call, lines)
        assert not any(failure_marker in line for line in lines)
    # (b) The library's own class with a text (the class derives from Exception, not from
    # RuntimeError) is the tool's line too.
    for call in (revoke, track):
        assert is_tool_line(end_of(ToolError("unresolvable author name(s): authorA"), call), "unresolvable author")
    # An OS error text is the tool's line only on a store with a planted fault; on any other it is
    # the driver's line with the raw text (the retry of the stopped write included).
    for call in (revoke, track):
        lines = end_of(RuntimeError(OS_ERROR_TEXT), call)
        assert is_driver_line(lines, OS_ERROR_TEXT), (call, lines)
        assert any("RuntimeError" in line for line in lines)
    assert is_tool_line(end_of(RuntimeError(OS_ERROR_TEXT), revoke, faulted=True), OS_ERROR_TEXT)
    # On a store with a planted fault a RuntimeError of the native call of any text, an empty one
    # included, and the library's class with or without a text are the tool's line; with no fault
    # they are the driver's line: the empty text says nothing.
    empty = [(RuntimeError(""), "RuntimeError"), (RuntimeError("   "), "RuntimeError"),
             (ToolError(""), "ToolError"), (ToolError("   "), "ToolError")]
    for error, name in empty:
        assert is_driver_line(end_of(error, revoke, False), name), (name, "no fault")
        faulted_lines = end_of(error, revoke, True)
        assert is_tool_line(faulted_lines, name), (name, "faulted", faulted_lines)
        assert not any(failure_marker in line for line in faulted_lines)
    # Every other end is the driver's line, faulted store or not: it names the class and the
    # text: another class, an error about the call itself (arity, type, a path), a subclass of
    # RuntimeError (the native module raises the class itself).
    others = [(NotImplementedError("not here"), "NotImplementedError"),
              (TypeError("token_index_revoke_by_sources() takes 3 positional arguments but 4 were given"), "TypeError"),
              (OSError("no such file"), "OSError")]
    for error, name in others:
        for faulted in (False, True):
            lines = end_of(error, revoke, faulted)
            assert is_driver_line(lines, name), (name, faulted, lines)
    # A panic of the native module passes through the driver: no line of the tool's, and the
    # process ends with the traceback.
    for faulted in (False, True):
        store = tracked_by_driver(driver, tmp_path, f"panic{faulted}")
        capsys.readouterr()
        with pytest.raises(PanicException):
            driver.main(["revoke", "--store", store, *(["--faulted"] if faulted else []), "sources/A.jsonl"],
                        ob=raising_ob(PanicException("panic from Rust code")))
        assert not any(line.startswith(tool_marker) for line in capsys.readouterr().out.splitlines())
    # A call that returns prints neither.
    assert driver.main(["status", "--store", str(tmp_path / "n0")], program=lambda arguments, cwd: (0, "")) == 0
    assert capsys.readouterr().out == ""




def test_an_error_of_the_tools_class_in_the_setup_or_a_path_the_driver_reads_is_the_drivers_own_stop(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    plan = tmp_path / "plan.json"
    write_plan(plan)
    standin = StandInOb()

    def refuse(path, authors, license, year, contributors=None, ob_dir=None):
        raise ToolError("unresolvable author name(s): authorA")

    standin.register_section = refuse
    assert driver.main(["track", "--store", str(tmp_path / "s"), "--plan", str(plan)], ob=standin) != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) and "unresolvable author" in line for line in lines)
    assert not any(line.startswith(driver.TOOL_ERROR_MARKER) for line in lines)
    # A plan file that is not there is a path of the driver's own.
    assert driver.main(["track", "--store", str(tmp_path / "t"), "--plan", str(tmp_path / "none.json")],
                       ob=StandInOb()) != 0
    lines = capsys.readouterr().out.splitlines()
    assert any(line.startswith(driver.FAILURE_MARKER) for line in lines)
    assert not any(line.startswith(driver.TOOL_ERROR_MARKER) for line in lines)


def test_a_track_call_that_the_tool_refuses_with_its_own_class_is_the_tools_line_after_the_calls_that_returned(need_module, tmp_path, capsys):
    driver = need_module(DRIVER)
    plan = tmp_path / "plan.json"
    write_plan(plan)
    standin = StandInOb()
    original = standin._ob_native.track
    calls = []

    def second_refused(record, file, sections, ob_dir, token_count, tokenizer):
        calls.append(token_count)
        if len(calls) == 2:
            raise RuntimeError(TRACK_TOOL_TEXT)
        return original(record, file, sections, ob_dir, token_count, tokenizer)

    standin._ob_native.track = second_refused
    code = driver.main(["track", "--store", str(tmp_path / "s"), "--plan", str(plan)], ob=standin)
    assert code != 0
    lines = capsys.readouterr().out.splitlines()
    assert "tracked 1" in lines and "tracked 2" not in lines
    assert any(line.startswith(driver.TOOL_ERROR_MARKER) and TRACK_TOOL_TEXT in line for line in lines)


# --- The host side: the command of every step --------------------------------------------------


def test_every_step_runs_behind_the_lock_in_the_sandbox_through_the_copy_by_name_and_relative_paths(need_module, tmp_path):
    ob = need_module(MODULE)
    sandbox = need_module("baselines.sandbox")
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [one])
    assert subject.track_all() == ()
    question = question_of(need_module, "c05.unmutated.E/shard-3")
    assert subject.read(question).raw == {"answer": "contains"}
    assert tool.calls, "the subject ran nothing"
    env = tmp_path / "work" / "venv-ob"
    expected_start = sandbox.tool_command(env, DRIVER_WORD, [])[:3]
    for call in tool.calls:
        argv = call["argv"]
        assert argv[: len(LOCK)] == LOCK
        assert argv[len(LOCK)] == "bwrap"
        assert "--unshare-all" in argv and "--die-with-parent" in argv and "--clearenv" in argv
        inner = argv[argv.index("--", len(LOCK)) + 1:]
        assert inner[:3] == expected_start, "the copy's own python by -m, never a launcher"
        home = argv[argv.index("HOME") + 1]
        assert home.startswith(str(tmp_path / "work") + "/") and home != str(tmp_path / "work")
        chdir = argv[argv.index("--chdir") + 1]
        assert chdir == str(one.build), "the tool starts in the build folder"
        assert all(not word.endswith(".sh") for word in inner)
    revoke = calls_of(tool, "revoke")[0]
    assert revoke["positional"] == ["sources/A.jsonl"], "files are handed relative to the build folder"


def ro_bind_problem(argv, work, copy) -> str | None:
    """Where the copied environment stands in the argv of a step: bound read-only (the three
    words ``--ro-bind COPY COPY``), after the writable bind of the work folder that holds it
    (``--bind WORK WORK``). The problem as text, or None."""
    argv = [str(word) for word in argv]
    writable = [at for at in range(len(argv) - 2)
                if argv[at] == "--bind" and argv[at + 1] == argv[at + 2] == str(work)]
    if not writable:
        return "the work folder is not bound writable"
    read_only = [at for at in range(len(argv) - 2)
                 if argv[at] == "--ro-bind" and argv[at + 1] == argv[at + 2] == str(copy)]
    if not read_only:
        return "the copy is not bound read-only"
    if min(read_only) < min(writable):
        return "the read-only bind of the copy comes before the writable bind of its parent"
    return None


def test_the_copy_is_bound_read_only_after_the_writable_bind_of_the_work_folder_that_holds_it(need_module, tmp_path):
    ob = need_module(MODULE)
    sandbox = need_module("baselines.sandbox")
    work, copy = tmp_path / "work", tmp_path / "work" / "venv-ob"
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [one])
    assert subject.track_all() == ()
    subject.read(question_of(need_module, "c05.unmutated.E/shard-3"))
    assert len(tool.calls) >= 4
    assert [ro_bind_problem(call["argv"], work, copy) for call in tool.calls] == [None] * len(tool.calls)
    # The step that is started and stopped is built the same way.
    scratch, started = _scratch(need_module, tmp_path / "kill", FakeTool())[:2]
    scratch.read(question_of(need_module, "c04.kill-after-100"))
    started_argv = [call["argv"] for call in started.calls if call.get("started")]
    assert started_argv and ro_bind_problem(
        started_argv[0], tmp_path / "kill" / "work", tmp_path / "kill" / "work" / "venv-ob") is None
    # Known-bad inputs: a step built with the interpreter folders alone leaves the copy writable
    # through its parent; and the same words in the wrong place.
    env = tmp_path / "work" / "venv-ob"
    alone = sandbox.sandboxed_step(
        lock_prefix=LOCK, work=work, home=sandbox.prepare_home(work, "home-known-bad"), chdir=one.build,
        command=sandbox.tool_command(env, DRIVER_WORD, ["status"]),
        ro_binds=[one.build, *sandbox.interpreter_binds(env)], env={})
    assert ro_bind_problem(alone, work, copy) == "the copy is not bound read-only"
    reordered = list(tool.calls[0]["argv"])
    at = reordered.index("--ro-bind", reordered.index("--bind"))
    pair = reordered[at: at + 3]
    del reordered[at: at + 3]
    reordered[1:1] = pair
    assert "before the writable bind" in (ro_bind_problem(reordered, work, copy) or "")


# --- Tracking: one process, packed order, clean before anything is read ------------------------------


def test_each_shard_is_tracked_once_in_one_process_in_ordinal_order_then_cleaned(need_module, tmp_path):
    ob = need_module(MODULE)
    first = shard(ob, tmp_path, "E/shard-1", plain_documents(ob))
    second = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [first, second])
    assert subject.track_all() == ()
    stores = {subject.tracked_store(first), subject.tracked_store(second)}
    assert len(stores) == 2, "each shard has a store of its own"
    for spec in (first, second):
        store = subject.tracked_store(spec)
        mine = [c["verb"] for c in tool.calls if c["store"] == store]
        assert mine.count("track") == 1 and mine.count("clean") == 1
        assert mine.index("track") < mine.index("clean")
        plan = calls_of(tool, "track", store)[0]["plan"]
        assert plan["documents"] == [{"section": d.section, "tokens": d.tokens, "record": getattr(d, "record", None),
                                      "file": getattr(d, "file", None)} for d in spec.documents]
        # One registration per file of the build, the files no document is tracked under included.
        assert plan.get("files") == list(BUILD_FILES)
    assert tool.starts == 0


def test_a_failed_track_is_a_failed_attempt_with_no_local_path_and_the_reads_of_that_shard_give_it(need_module, tmp_path):
    ob = need_module(MODULE)
    record = need_module("baselines.record")
    interfaces = need_module("baselines.interfaces")
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    other = shard(ob, tmp_path, "E/shard-1", plain_documents(ob))
    here = str(tmp_path)
    tool = FakeTool(fail={"track": (1, f"Traceback in {tmp_path}/work/x\nsandbox failed at {tmp_path}/build\n")})
    subject, tool = make_subject(need_module, tmp_path, [one, other], tool=tool)
    failed = subject.track_all()
    assert len(failed) == 2 and all(isinstance(a, interfaces.Attempt) for a in failed)
    assert len({a.id for a in failed}) == 2 and all(a.id for a in failed)
    for attempt in failed:
        assert attempt.exit_code == 1
        assert here not in attempt.output and all(here not in word for word in attempt.command)
        assert "sandbox failed" in attempt.output, "the failure stays readable"
    question = question_of(need_module, "c05.unmutated.E/shard-3")
    got = subject.read(question)
    assert isinstance(got, interfaces.Attempt) and got.id in {a.id for a in failed}
    line = record.attempt_line(got, [question.id], commit="abc1234")
    record.write_attempts(tmp_path / "attempts.jsonl", [line])
    assert "diff" not in json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])


def test_the_section_hash_of_each_file_a_document_is_tracked_under_is_present_among_the_sources_of_the_entries_a_presence_check_only(need_module, tmp_path):
    # A presence check, not a comparison of two hash functions: the only hash on the driver's path
    # is the one the tool's own call returned. Where the hash kept for a file is not among the
    # sources of the entries, a revoke by it would find nothing and every shard would read "does
    # not contain": such a store is a failed attempt before any answer is read from it.
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    question = question_of(need_module, "c05.unmutated.E/shard-3")
    one = shard(ob, tmp_path / "a", "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path / "a", [one], tool=FakeTool(entry_hash="x:"))
    failed = subject.track_all()
    assert len(failed) == 1 and isinstance(failed[0], interfaces.Attempt)
    assert "sources/" in failed[0].output and "hash" in failed[0].output
    assert str(tmp_path) not in failed[0].output
    got = subject.read(question)
    assert isinstance(got, interfaces.Attempt) and got.id == failed[0].id
    assert not calls_of(tool, "revoke") and not calls_of(tool, "generate-set"), "no answer was asked for"
    # Where the hashes agree the store is tracked, and a shard that holds no document has no hash to find.
    two = shard(ob, tmp_path / "b", "E/shard-3", plain_documents(ob, with_a=True))
    empty = shard(ob, tmp_path / "b", "E/shard-1", ())
    subject, _ = make_subject(need_module, tmp_path / "b", [two, empty])
    assert subject.track_all() == ()
    assert subject.read(question).raw == {"answer": "contains"}


def test_the_steps_that_run_the_tools_program_name_it_by_the_variable_the_package_reads_and_bind_it_read_only(need_module, tmp_path):
    ob = need_module(MODULE)
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [one])
    seen = set()

    def check_steps():
        for call in tool.calls:
            argv = [str(word) for word in call["argv"]]
            named = any(argv[at] == "--setenv" and argv[at + 1] == PROGRAM_VARIABLE and argv[at + 2] == BINARY
                        for at in range(len(argv) - 2))
            bound = any(argv[at] == "--ro-bind" and argv[at + 1] == argv[at + 2] == BINARY
                        for at in range(len(argv) - 2))
            runs_program = call["verb"] in PROGRAM_VERBS
            assert named == runs_program and bound == runs_program, call["verb"]
            seen.add(call["verb"])

    assert subject.track_all() == ()
    check_steps()
    subject.read(question_of(need_module, "c05.unmutated.E/shard-3"))
    check_steps()
    # The entries are files and the fault is the runner's: no export verb, no fault verb, and
    # nothing asks the program for its status of a revoked copy.
    assert seen == {"track", "clean", "revoke", "generate-set"}, seen


def test_every_step_sets_the_language_and_the_two_offline_words_and_a_differing_value_of_the_caller_is_refused(need_module, tmp_path):
    # As the earlier run's sandbox script set them: LANG C.UTF-8, HF_HUB_OFFLINE 1, TRANSFORMERS_OFFLINE 1.
    ob = need_module(MODULE)
    fixed = {"LANG": "C.UTF-8", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    one = shard(ob, tmp_path / "a", "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path / "a", [one])
    assert subject.track_all() == ()
    subject.read(question_of(need_module, "c05.unmutated.E/shard-3"))
    assert tool.calls, "the subject ran nothing"
    for call in tool.calls:
        argv = [str(word) for word in call["argv"]]
        words = {argv[at + 1]: argv[at + 2] for at in range(len(argv) - 2) if argv[at] == "--setenv"}
        assert {name: words.get(name) for name in fixed} == fixed, call["verb"]
    # The caller may give them as the step sets them, once each; never another way.
    same = shard(ob, tmp_path / "b", "E/shard-3", plain_documents(ob))
    subject, tool = make_subject(need_module, tmp_path / "b", [same], step_env=fixed)
    assert subject.track_all() == ()
    argv = [str(word) for word in tool.calls[0]["argv"]]
    assert [argv[at + 1] for at in range(len(argv) - 1) if argv[at] == "--setenv"].count("LANG") == 1
    for number, bad in enumerate(({"LANG": "de_DE.UTF-8"}, {"HF_HUB_OFFLINE": "0"}, {"TRANSFORMERS_OFFLINE": "0"},
                                  {**fixed, "LANG": "C"})):
        other = shard(ob, tmp_path / f"bad{number}", "E/shard-3", plain_documents(ob))
        with pytest.raises(ValueError):
            made, ran = make_subject(need_module, tmp_path / f"bad{number}", [other], step_env=bad)
            made.track_all()
        assert not (tmp_path / f"bad{number}" / "work" / "ob-home-1").exists(), "nothing was started"


def test_without_the_path_of_the_tools_program_the_shards_are_not_tracked(need_module, tmp_path):
    ob = need_module(MODULE)
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [one], binary=False)
    with pytest.raises(ValueError, match="ob_binary"):
        subject.track_all()
    assert tool.calls == []


def test_the_entries_of_a_store_are_read_from_its_files_in_the_order_of_their_integer_names(need_module, tmp_path):
    ob = need_module(MODULE)
    reader = getattr(ob, "read_store_entries", None)
    assert callable(reader), f"{MODULE} has no read_store_entries"
    store = tmp_path / "s"
    folder = store.joinpath(*ENTRY_FOLDER)
    folder.mkdir(parents=True)

    def line(count):
        return json.dumps({"sources": ["h"], "token_count": count, "tokenizer": "gpt2", "revoked": False}) + "\n"

    (folder / "10").write_text(line(9))
    (folder / "2").write_text(line(7) + line(8))
    # The PID files and the binary index beside the folder are not entry files.
    (store / ".ob" / "token-index.gpt2.4").write_text(line(1))
    (store / ".ob" / "token-index.gpt2.bin").write_bytes(b"\x00\x01")
    assert [entry.token_count for entry in reader(store)] == [7, 8, 9]
    assert reader(tmp_path / "none") == [], "a store with no entry folder holds no entry"
    (folder / "x").write_text(line(1))
    with pytest.raises(ValueError):
        reader(store)


def test_the_status_of_the_program_is_read_by_its_entries_line_and_the_other_forms_it_prints_are_no_count(need_module, tmp_path):
    # The program prints, with a tokenizer, "Tokenizer: <name>", "Entries: <n> (<a> active, <r>
    # revoked)" and "Total tokens: <t>"; without one "Authors: <n>" and "Path: <dir>"; for a
    # folder that is not a store "Not initialized. Run `ob init` first." (and exit code 0).
    interfaces = need_module("baselines.interfaces")
    question = question_of(need_module, "c04.kill-after-100")
    subject, _, _ = _scratch(need_module, tmp_path / "real", FakeTool())
    got = subject.read(question)
    assert isinstance(got, interfaces.Reading)
    assert got.raw["records"]["rows_after_stop"] == 100
    # Another form has no entries line, so it is no count: a failed attempt, never a zero.
    for number, form in enumerate(("Not initialized. Run `ob init` first.\n", "Authors: 3\nPath: .\n")):
        subject, _, _ = _scratch(need_module, tmp_path / f"none{number}", FakeTool(status_format=form))
        got = subject.read(question)
        assert isinstance(got, interfaces.Attempt), form
        assert form.splitlines()[0] in got.output


# --- The shard rule --------------------------------------------------------------------------------


def _read_answer(need_module, tmp_path, documents, listed="c05.unmutated.E/shard-3", name="E/shard-3",
                 tool=None, target=("store", "main")):
    ob = need_module(MODULE)
    one = shard(ob, tmp_path, name, documents, target=target)
    subject, tool = make_subject(need_module, tmp_path, [one], tool=tool)
    assert subject.track_all() == ()
    return subject, tool, one, subject.read(question_of(need_module, listed))


def test_a_shard_reads_contains_does_not_contain_or_not_covered_by_its_bits_and_entries(need_module, tmp_path):
    ob = need_module(MODULE)
    _, _, _, with_a = _read_answer(need_module, tmp_path / "a", plain_documents(ob, with_a=True))
    _, _, _, without = _read_answer(need_module, tmp_path / "b", plain_documents(ob))
    _, _, _, empty = _read_answer(need_module, tmp_path / "c", ())
    assert with_a.raw == {"answer": "contains"}
    assert without.raw == {"answer": "does not contain"}
    assert empty.raw == {"answer": "not covered"}, "a store that holds no entry is no record"
    assert with_a.route == question_of(need_module, "c05.unmutated.E/shard-3").route


def test_a_shard_linked_to_a_source_only_through_a_generated_record_does_not_contain_and_is_graded_wrong(need_module, tmp_path):
    ob = need_module(MODULE)
    # The caller adds no lineage: the generated record is tracked under its generation output.
    documents = (doc(ob, "sources/B.jsonl", 4), doc(ob, "gen/P.jsonl", 5), doc(ob, "sources/Z.jsonl", 3))
    _, _, _, got = _read_answer(need_module, tmp_path, documents)
    assert got.raw == {"answer": "does not contain"}
    question = question_of(need_module, "c05.unmutated.E/shard-3")
    expect = expectation(need_module, "c05.unmutated.E/shard-3")
    assert expect["answer"] == "contains"
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["answer"]


def test_each_request_is_put_to_a_fresh_copy_revoked_once_and_the_tracked_store_is_never_changed(need_module, tmp_path):
    ob = need_module(MODULE)
    subject, tool, one, first = _read_answer(need_module, tmp_path, plain_documents(ob, with_a=True))
    tracked = subject.tracked_store(one)
    before = tree_digest(tracked)
    question = question_of(need_module, "c05.unmutated.E/shard-3")
    again = subject.read(question)
    third = subject.read(question)
    assert first.raw == again.raw == third.raw == {"answer": "contains"}
    assert tree_digest(tracked) == before, "the tracked store holds the same bytes"
    revokes = calls_of(tool, "revoke")
    assert len(revokes) == 3
    copies = [c["store"] for c in revokes]
    assert len(set(copies)) == 3 and tracked not in copies
    for copy in copies:
        mine = [c["verb"] for c in tool.calls if c["store"] == copy]
        assert mine.count("revoke") == 1
        assert mine.index("revoke") < mine.index("generate-set")


def test_the_request_names_the_files_of_its_members_and_an_unknown_request_is_refused(need_module, tmp_path):
    ob = need_module(MODULE)
    subject, tool, _, got = _read_answer(
        need_module, tmp_path, plain_documents(ob, with_c=True), listed="c05.withdraw-C.unmutated.E/shard-3")
    assert got.raw == {"answer": "contains"}, "the request is withdraw C: its member is C/r1"
    assert calls_of(tool, "revoke")[0]["positional"] == ["sources/C.jsonl"]
    bare, _ = make_subject(need_module, tmp_path / "x", [shard(ob, tmp_path / "x", "E/shard-3", plain_documents(ob))],
                           requests={})
    bare.track_all()
    with pytest.raises((KeyError, ValueError)):
        bare.read(question_of(need_module, "c05.unmutated.E/shard-3"))


# --- Kinds the tool holds no entry for, and rows that record none --------------------------------------


def test_a_kind_the_tool_holds_no_entry_for_is_cannot_answer_with_the_missing_text_of_the_table(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    # The tool ran on the store each case is asked on: the main store and the one without capture.
    subject, tool = make_subject(need_module, tmp_path, [
        shard(ob, tmp_path, "E/shard-1", plain_documents(ob)),
        shard(ob, tmp_path, "E/shard-9", plain_documents(ob), target=("store", "no-capture"))])
    assert subject.track_all() == ()
    started = len(tool.calls)
    cases = {
        "c05.f.map-deleted.src/A": "files", "c05.f.map-deleted.gen/T": "files",
        "c05.f.map-deleted.R1": "files", "c05.f.map-deleted.E/mix-1": "mixtures",
        "c05.f.map-deleted.E@6": "checkpoints", "c05.f.digest.listing": "listing",
        "c09.fixture.successor.R1": "files", "c09.fixture.R1.action": "files",
        "c05.withdraw-C.unmutated.R1": "files", "c06.fixture.downstream.gen/T": "files",
    }
    for listed, kind in cases.items():
        assert artifact_kind(facts.core_given[listed].get("artifact")) == kind
        got = subject.read(question_of(need_module, listed))
        assert got.raw == {"cannot_answer": missing_text(facts, kind)}, listed
    assert len(tool.calls) == started, "nothing is started for a kind the tool has no entry for"


KIND_LISTED = ("c05.f.map-deleted.src/A", "c05.f.map-deleted.E/mix-1", "c05.f.map-deleted.E@6",
               "c05.f.digest.listing")


def test_a_kind_the_tool_holds_no_entry_for_is_cannot_answer_only_after_the_successful_track_of_its_store(need_module, tmp_path):
    # A tool that did not run cannot earn "cannot answer": the reading needs the track of every
    # shard of the store the question is asked on. Else it is the failed attempt of that track,
    # or no line at all (a ValueError: not attempted).
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    questions = [question_of(need_module, listed) for listed in KIND_LISTED]
    # The track of the store failed.
    one = shard(ob, tmp_path / "a", "E/shard-1", plain_documents(ob))
    subject, _ = make_subject(need_module, tmp_path / "a", [one], tool=FakeTool(fail={"track": (1, "sandbox failed\n")}))
    failed = subject.track_all()
    assert len(failed) == 1
    for question in questions:
        got = subject.read(question)
        assert isinstance(got, interfaces.Attempt) and got.id == failed[0].id, question.id
    # One of the two shards of the store failed to track: the store did not run.
    first = shard(ob, tmp_path / "b", "E/shard-1", plain_documents(ob))
    second = shard(ob, tmp_path / "b", "E/shard-3", plain_documents(ob, with_a=True))
    subject, _ = make_subject(need_module, tmp_path / "b", [first, second], tool=FakeTool(fail_track_in=("tracked-1",)))
    failed = subject.track_all()
    assert len(failed) == 1
    for question in questions:
        got = subject.read(question)
        assert isinstance(got, interfaces.Attempt) and got.id == failed[0].id, question.id
    # No shard was given for the store: nothing ran for it.
    elsewhere = shard(ob, tmp_path / "c", "E/shard-1", plain_documents(ob), target=("store", "no-capture"))
    subject, _ = make_subject(need_module, tmp_path / "c", [elsewhere])
    assert subject.track_all() == ()
    for question in questions:
        with pytest.raises(ValueError) as caught:
            subject.read(question)
        assert type(caught.value).__name__ == "NotAttempted", question.id
    # The shards were not tracked yet.
    fresh, _ = make_subject(need_module, tmp_path / "d", [shard(ob, tmp_path / "d", "E/shard-1", plain_documents(ob))])
    for question in questions:
        with pytest.raises(ValueError):
            fresh.read(question)


def test_a_question_whose_row_records_none_has_no_reading_and_starts_nothing(need_module, tmp_path):
    ob = need_module(MODULE)
    subject, tool = make_subject(need_module, tmp_path, [shard(ob, tmp_path, "E/shard-1", plain_documents(ob))])
    subject.track_all()
    started = len(tool.calls)
    none_questions = [q for q in ob_questions(need_module).values() if q.route is None]
    assert len(none_questions) == 391 - ROUTED_TOTAL
    for question in none_questions:
        got = subject.read(question)
        assert got.question_id == question.id and got.raw is None and got.route is None
    assert len(tool.calls) == started
    grade = need_module("baselines.grade")
    record = need_module("baselines.record")
    question = next(q for q in none_questions if record.layouts_of(q) is None)
    attempt = record.StoreAttempt(SUBJECT, record.attempt_target(question), None, "ran")
    verdict = grade.grade(question, {"answer": "x"}, subject.read(question), attempts=[attempt])
    assert verdict.status == "cannot answer" and verdict.evidence == question.missing


# --- c03.spans ---------------------------------------------------------------------------------------


def _set_shards(ob, tmp_path, size, sizes_by_shard, layout="round-robin"):
    return [shard(ob, tmp_path, name, [doc(ob, "sources/B.jsonl", n) for n in counts],
                  target=("set", size), layout=layout) for name, counts in sizes_by_shard.items()]


def test_spans_read_the_token_count_of_each_entry_and_their_sum_and_say_the_count_is_the_callers(need_module, tmp_path):
    ob = need_module(MODULE)
    index = {"chunk-1": [7, 8, 9], "chunk-2": [5, 6]}
    subject, tool = make_subject(need_module, tmp_path, _set_shards(ob, tmp_path, "small", index))
    assert subject.track_all() == ()
    per_ordinal = subject.read(question_of(need_module, "c03.spans.per-ordinal", "small"))
    total = subject.read(question_of(need_module, "c03.spans.total", "small"))
    assert per_ordinal.raw["per_ordinal"] == index
    assert total.raw["per_shard"] == {"chunk-1": 24, "chunk-2": 11}
    assert set(per_ordinal.raw) == {"per_ordinal", "notes"} and set(total.raw) == {"per_shard", "notes"}
    for got in (per_ordinal, total):
        notes = got.raw["notes"]
        assert "caller" in " ".join(notes if isinstance(notes, list) else [notes])
    # Graded against the sizes the shards' own indexes give (written here by hand).
    expected = need_module("baselines.expected")
    grade = need_module("baselines.grade")
    for listed, reading in (("c03.spans.per-ordinal", per_ordinal), ("c03.spans.total", total)):
        question = question_of(need_module, listed, "small")
        concrete = expected.concrete_values(expected.expected_for(listed, "small", index={"small": index}))
        assert grade.grade_concrete(question, concrete, reading).status == "right"
    changed = {"chunk-1": [7, 8, 9], "chunk-2": [5, 7]}
    other, _ = make_subject(need_module, tmp_path / "w", _set_shards(ob, tmp_path / "w", "small", changed))
    other.track_all()
    question = question_of(need_module, "c03.spans.per-ordinal", "small")
    concrete = expected.concrete_values(expected.expected_for("c03.spans.per-ordinal", "small", index={"small": index}))
    verdict = grade.grade_concrete(question, concrete, other.read(question))
    assert verdict.status == "wrong" and list(verdict.evidence) == ["per_ordinal"]


@pytest.mark.parametrize("tool", [FakeTool(processes=2), FakeTool(say_run=False)], ids=["two writers", "no statement"])
def test_spans_are_cannot_answer_where_the_run_does_not_say_one_process_in_packed_order(need_module, tmp_path, tool):
    ob = need_module(MODULE)
    index = {"chunk-1": [7, 8, 9]}
    subject, _ = make_subject(need_module, tmp_path, _set_shards(ob, tmp_path, "100k", index), tool=tool)
    subject.track_all()
    for listed in ("c03.spans.per-ordinal", "c03.spans.total"):
        got = subject.read(question_of(need_module, listed, "100k"))
        assert list(got.raw) == ["cannot_answer"]
        assert "an id or an offset" in got.raw["cannot_answer"]
        assert graded(need_module, question_of(need_module, listed, "100k"), got).status == "cannot answer"


# --- c04: the writer stopped, and run again ---------------------------------------------------------------


def _scratch(need_module, tmp_path, tool, documents=None, **more):
    ob = need_module(MODULE)
    count = 500
    documents = documents or tuple(doc(ob, "sources/B.jsonl", 3) for _ in range(count))
    one = shard(ob, tmp_path, "c04/shard-500", documents, target=("store", "scratch-500"))
    subject, tool = make_subject(need_module, tmp_path, [one], tool=tool, **more)
    return subject, tool, one


def test_a_shard_whose_track_was_refused_starts_no_writer_and_both_writer_rows_give_its_failed_attempt(need_module, tmp_path):
    """README "a failed attempt to run an existing tool is not an answer": the two writer rows of a shard whose
    own track failed are that failed attempt; no writer is started on it (first real pass, finding 1)."""
    interfaces = need_module("baselines.interfaces")
    tool = FakeTool(fail={"track": (1, "the tool refused the plan\n")})
    subject, tool, one = _scratch(need_module, tmp_path, tool)
    failed = subject.track_all()
    assert len(failed) == 1 and isinstance(failed[0], interfaces.Attempt)
    starts_before, tracks_before = tool.starts, len(calls_of(tool, "track"))
    for listed in ("c04.kill-after-100", "c04.retry"):
        got = subject.read(question_of(need_module, listed))
        assert isinstance(got, interfaces.Attempt) and got.id == failed[0].id, listed
    assert tool.starts == starts_before == 0, "a writer was started on a shard whose track was refused"
    assert len(calls_of(tool, "track")) == tracks_before, "a track was run again for a writer row"


def test_the_stopped_write_is_stopped_by_its_process_id_after_its_100th_call_and_the_entries_are_counted(need_module, tmp_path):
    tool = FakeTool(drop_partial=False)
    subject, tool, one = _scratch(need_module, tmp_path, tool)
    got = subject.read(question_of(need_module, "c04.kill-after-100"))
    assert isinstance(got, need_module("baselines.interfaces").Reading)
    assert list(got.raw) == ["records"]
    assert got.raw["records"]["rows_after_stop"] == 100
    assert got.raw["records"]["pid_files"] == ["token-index.gpt2.2"], "the file names are recorded beside the count"
    assert tool.stops == [4242] and tool.starts == 1
    verbs = [c["verb"] for c in tool.calls]
    assert verbs.index("track") < verbs.index("clean") < verbs.index("status")
    verdict = graded(need_module, question_of(need_module, "c04.kill-after-100"), got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["never"]


def test_the_file_names_are_recorded_beside_the_count_and_change_no_grade(need_module, tmp_path):
    question = question_of(need_module, "c04.kill-after-100")
    reading = need_module("baselines.interfaces").Reading
    found = []
    for names in (("2",), ("7",)):
        subject, tool, _ = _scratch(need_module, tmp_path / names[0], FakeTool(pid_names=names))
        got = subject.read(question)
        assert isinstance(got, reading)
        assert got.raw["records"]["rows_after_stop"] == 100
        verdict = graded(need_module, question, got)
        found.append((got.raw["records"]["pid_files"], verdict.status, list(verdict.evidence)))
    assert found == [(["token-index.gpt2.2"], "wrong", ["never"]), (["token-index.gpt2.7"], "wrong", ["never"])]
    # A stop that leaves no entry (the partial run dropped by clean) is allowed.
    subject, tool, _ = _scratch(need_module, tmp_path / "none", FakeTool(drop_partial=True))
    got = subject.read(question)
    assert isinstance(got, reading)
    assert got.raw["records"]["rows_after_stop"] == 0
    assert graded(need_module, question, got).status == "right"


def test_the_write_made_again_runs_on_the_same_store_and_is_graded_by_its_counts(need_module, tmp_path):
    question = question_of(need_module, "c04.retry")
    # Two writers that name one file: the second run replaces the first: 500 entries, not refused.
    subject, tool, _ = _scratch(need_module, tmp_path / "a", FakeTool(same_pid_file=True))
    got = subject.read(question)
    records = got.raw["records"]
    assert records["rows_after_retry"] == 500 and records["refused"] is False
    assert records.get("matches_shard") is None, "the tool's entries are counted and matched to no file"
    tracks = calls_of(tool, "track")
    assert len(tracks) == 2 and tracks[0]["store"] == tracks[1]["store"], "the same store, not a fresh copy"
    assert graded(need_module, question, got).status == "right"
    # Two files: the entries add up to 600, not refused: never allowed.
    subject, tool, _ = _scratch(need_module, tmp_path / "b", FakeTool(same_pid_file=False))
    got = subject.read(question)
    assert got.raw["records"]["rows_after_retry"] == 600 and got.raw["records"]["refused"] is False
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["never"]
    # Refused with the stopped 100 entries left: 1 to 499 entries remain, never allowed.
    subject, tool, _ = _scratch(need_module, tmp_path / "c", FakeTool(refuse_second_track=(5, TRACK_TOOL_TEXT)))
    got = subject.read(question)
    records = got.raw["records"]
    assert records["refused"] is True and records["rows_after_retry"] == 100
    assert isinstance(records["mismatch"], str) and records["mismatch"].strip()
    assert records["mismatch"] == f"track raised RuntimeError: {TRACK_TOOL_TEXT}", "the driver's line without its marker"
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["never"]
    # Refused, the partial entries dropped by clean, the mismatch named: allowed.
    subject, tool, _ = _scratch(need_module, tmp_path / "d", FakeTool(
        drop_partial=True, refuse_second_track=(5, TRACK_TOOL_TEXT)))
    got = subject.read(question)
    assert got.raw["records"]["rows_after_retry"] == 0 and got.raw["records"]["refused"] is True
    assert graded(need_module, question, got).status == "right"
    last = calls_of(tool, "clean")[-1], calls_of(tool, "status")[-1]
    assert tool.calls[-2] is last[0] and tool.calls[-1] is last[1], "clean then status come last"


def test_a_store_with_its_last_entry_deleted_is_read_as_stopped_only_where_the_tool_ends_with_an_error(need_module, tmp_path):
    question = question_of(need_module, "c04.count-499")
    subject, tool, _ = _scratch(need_module, tmp_path / "a", FakeTool(error_after=("l.count-499",)),
                                documents=plain_documents(need_module(MODULE), with_a=True))
    subject.track_all()
    got = subject.read(question)
    assert isinstance(got, need_module("baselines.interfaces").Reading)
    assert list(got.raw) == ["stopped"] and got.raw["stopped"] is True
    assert graded(need_module, question, got).status == "right"
    fault = calls_of(tool, "fault")[0]
    assert fault["options"]["fault"] == "l.count-499"
    verbs = [c["verb"] for c in tool.calls if c["store"] == fault["store"]]
    assert verbs.index("fault") < verbs.index("revoke")
    # An answer read from the 499 entries where a stop is expected is wrong, in its answer.
    subject, tool, _ = _scratch(need_module, tmp_path / "b", FakeTool(),
                                documents=plain_documents(need_module(MODULE), with_a=True))
    subject.track_all()
    got = subject.read(question)
    assert got.raw == {"answer": "contains"}
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["answer"]
    # An end with no exit code (the step was cut off) is a failed attempt, not a stop.
    interfaces = need_module("baselines.interfaces")
    subject, tool, _ = _scratch(need_module, tmp_path / "c", FakeTool(fail={"revoke": (None, "")}),
                                documents=plain_documents(need_module(MODULE), with_a=True))
    subject.track_all()
    assert isinstance(subject.read(question), interfaces.Attempt)


def test_an_end_with_an_error_and_no_line_of_the_tool_is_a_failed_attempt_never_a_stop_or_a_refusal(need_module, tmp_path):
    # Table, c04.count-499 and c05.f.digest: "stopped: a command of the subject that reads the
    # changed record ends with an error that reports it"; c04.retry: "whether the tool refused
    # instead". A lock that times out or a sandbox that fails before the driver starts reports
    # nothing about the record: the step has no answer, so no reading and a failed attempt.
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    driver = need_module(DRIVER)
    ends = {"no line": "flock: could not get the lock in time\n",
            "the driver's own line": f"{driver.FAILURE_MARKER} the tool has no call revoke_by_sources\n"}
    for name, text in ends.items():
        for number, code in enumerate((1, 2)):
            here = tmp_path / f"{name[:2]}-{number}"
            subject, tool, _ = _scratch(need_module, here / "a", FakeTool(fail={"revoke": (code, text)}),
                                        documents=plain_documents(ob, with_a=True))
            subject.track_all()
            got = subject.read(question_of(need_module, "c04.count-499"))
            assert isinstance(got, interfaces.Attempt), f"c04.count-499, {name}, exit {code}: {got}"
            assert got.exit_code == code
            subject, tool = make_subject(need_module, here / "b", _main_shards(ob, here / "b"),
                                         tool=FakeTool(fail={"revoke": (code, text)}))
            subject.track_all()
            got = subject.read(question_of(need_module, "c05.f.digest.E/shard-3"))
            assert isinstance(got, interfaces.Attempt), f"c05.f.digest, {name}, exit {code}: {got}"
            subject, tool, _ = _scratch(need_module, here / "c", FakeTool(fail={"track": (code, text)}))
            got = subject.read(question_of(need_module, "c04.retry"))
            assert isinstance(got, interfaces.Attempt), f"c04.retry, {name}, exit {code}: {got}"
            assert got.exit_code == code
    # The same end with the tool's own line is the stop and the refusal the rows read.
    subject, tool, _ = _scratch(need_module, tmp_path / "line-a", FakeTool(error_after=("l.count-499",)),
                                documents=plain_documents(ob, with_a=True))
    subject.track_all()
    got = subject.read(question_of(need_module, "c04.count-499"))
    assert isinstance(got, interfaces.Reading) and got.raw == {"stopped": True}
    subject, tool, _ = _scratch(need_module, tmp_path / "line-b", FakeTool(refuse_second_track=(5, TRACK_TOOL_TEXT)))
    got = subject.read(question_of(need_module, "c04.retry"))
    assert isinstance(got, interfaces.Reading) and got.raw["records"]["refused"] is True


def test_a_stop_and_a_refusal_are_the_tools_line_with_a_text_and_no_other_end_is_one(need_module, tmp_path):
    # The three ends that say nothing about the changed record: a traceback of another class, a
    # signal exit with no output, the tool's line with no text. Each is a failed attempt with
    # the exit code and the raw text kept; only the tool's line with a text is the tool's error.
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    driver = need_module(DRIVER)
    marker = driver.TOOL_ERROR_MARKER
    documents = plain_documents(ob, with_a=True)
    ends = {
        "a traceback of another class": (1, "Traceback (most recent call last):\n  File \"x.py\", line 3, in go\n"
                                            "pyo3_runtime.PanicException: panic from Rust code\n"),
        "a signal exit and no output": (137, ""),
        "a signal exit as a negative code": (-9, ""),
        "the tool's line with no text": (3, marker + "\n"),
        "the tool's line with a blank text": (3, marker + "   \n"),
    }
    for number, (name, (code, text)) in enumerate(ends.items()):
        subject, _, _ = _scratch(need_module, tmp_path / f"stop-{number}", FakeTool(fail={"revoke": (code, text)}),
                                 documents=documents)
        subject.track_all()
        got = subject.read(question_of(need_module, "c04.count-499"))
        assert isinstance(got, interfaces.Attempt), name
        assert got.exit_code == code, name
        if "Traceback" in text:
            assert "panic from Rust code" in got.output, "the raw text stays"
        subject, _, _ = _scratch(need_module, tmp_path / f"retry-{number}", FakeTool(fail={"track": (code, text)}))
        got = subject.read(question_of(need_module, "c04.retry"))
        assert isinstance(got, interfaces.Attempt), name
        assert got.exit_code == code, name
    # The tool's line with a text is the stop and the refusal.
    line = marker + f" token_index_revoke_by_sources raised RuntimeError: {REVOKE_TOOL_TEXT}\n"
    track_line = marker + f" track raised RuntimeError: {TRACK_TOOL_TEXT}\n"
    subject, _, _ = _scratch(need_module, tmp_path / "stop-ok", FakeTool(fail={"revoke": (5, line)}), documents=documents)
    subject.track_all()
    assert subject.read(question_of(need_module, "c04.count-499")).raw == {"stopped": True}
    subject, _, _ = _scratch(need_module, tmp_path / "retry-ok", FakeTool(fail={"track": (5, track_line)}))
    records = subject.read(question_of(need_module, "c04.retry")).raw["records"]
    assert records["refused"] is True and records["mismatch"] == f"track raised RuntimeError: {TRACK_TOOL_TEXT}"

# The stop of Check 4, by the signal that was sent. A runner gives the end of a process a signal
# stopped as a negative code or as 128 plus the signal's number, and the handle's ``stop`` gives
# the number of the signal it sent. (Signal sent, exit of the writer) of a stop that landed:
LANDED_STOPS = ((9, -9), (9, 137), (15, -15), (15, 143), (2, -2), (2, 130), (64, -64), (64, 192))
# Signal 9 sent to a writer that ended by itself (0, 1, 2, 127, 128, 255), by an exit that is no
# signal's (193), or by another signal (11, 15):
NOT_LANDED_EXITS = (0, 1, 2, 127, 128, 193, 255, -11, 139, -15, 143, -1, 129)
STOPPED_ROWS = ["c04.kill-after-100", "c04.retry"]


@pytest.mark.parametrize("listed", STOPPED_ROWS)
def test_a_stop_lands_only_as_the_exit_of_the_signal_sent_else_it_is_the_failed_attempt_of_the_stopped_writer(need_module, tmp_path, listed):
    # A writer that ended by itself, by an exit that is no signal's or by another signal where the
    # stop was meant to end it was not stopped: no fault was put, so the store says nothing about
    # an interrupted write. Both rows are the failed attempt of the stopped writer, with its exit.
    interfaces = need_module("baselines.interfaces")
    question = question_of(need_module, listed)
    for number, (sent, code) in enumerate(LANDED_STOPS):
        subject, _, _ = _scratch(need_module, tmp_path / f"l{number}", FakeTool(stop_signal=sent, stop_exit=code))
        got = subject.read(question)
        assert isinstance(got, interfaces.Reading), (listed, sent, code)
        assert list(got.raw) == ["records"]
    for number, code in enumerate(NOT_LANDED_EXITS):
        base = tmp_path / f"n{number}"
        printed = f'Traceback: File "{base}/work/x.py", line 3\ndone'
        subject, tool, _ = _scratch(need_module, base, FakeTool(stop_signal=9, stop_exit=code, stop_output=printed))
        got = subject.read(question)
        assert isinstance(got, interfaces.Attempt), (listed, code)
        assert got.exit_code == code
        assert got.id.startswith("originblame-stop-"), got.id
        assert "did not land" in got.output and "done" in got.output
        assert str(base) not in got.output and all(str(base) not in word for word in got.command)
        assert tool.stops == [4242], "the stop was made"
        assert not calls_of(tool, "clean") and not calls_of(tool, "status"), "nothing is read after it"
        assert len(calls_of(tool, "track")) == 1, "the write is not made again"


@pytest.mark.parametrize("listed", STOPPED_ROWS)
def test_a_stop_with_no_exit_or_no_signal_is_an_incomplete_run_not_a_guess(need_module, tmp_path, listed):
    question = question_of(need_module, listed)
    for number, bad in enumerate((None, True, "137")):
        subject, _, _ = _scratch(need_module, tmp_path / f"exit-{number}", FakeTool(stop_exit=bad))
        with pytest.raises(ValueError):
            subject.read(question)
    # The signal that was sent is part of the record: none, or one that is no signal's number,
    # leaves the exit with nothing to be the exit of.
    for number, bad in enumerate((None, True, "9", 0, -9, 65)):
        subject, _, _ = _scratch(need_module, tmp_path / f"signal-{number}", FakeTool(stop_signal=bad))
        with pytest.raises(ValueError):
            subject.read(question)


def test_the_rows_after_a_stop_say_whether_they_showed_while_the_writer_ran_or_only_at_its_end(need_module, tmp_path):
    interfaces = need_module("baselines.interfaces")
    question = question_of(need_module, "c04.kill-after-100")
    for number, (at_end, words) in enumerate(((False, "while the writer ran"), (True, "only at its end"))):
        subject, _, _ = _scratch(need_module, tmp_path / f"k{number}", FakeTool(write_at_end=at_end))
        got = subject.read(question)
        assert isinstance(got, interfaces.Reading)
        assert got.raw["records"].get("rows_showed") == words
        assert got.raw["records"]["rows_after_stop"] == 100
        # The grade reads the count alone, as before.
        verdict = graded(need_module, question, got)
        assert verdict.status == "wrong" and list(verdict.evidence) == ["never"]
    # The second row's reading does not carry it.
    subject, _, _ = _scratch(need_module, tmp_path / "retry", FakeTool())
    assert "rows_showed" not in subject.read(question_of(need_module, "c04.retry")).raw["records"]



def test_a_native_refusal_on_the_retry_reads_the_rows_right_and_an_os_error_is_a_failed_attempt_line(need_module, tmp_path):
    # c10.c04.interrupted-write.originblame: "after the retry there are 500 entries, or the tool
    # refused the write with an error". The real driver ends the retry; the host reads what it
    # printed; the grade is the grader's. A retry that no fault was put to ends in the tool's line
    # for a RuntimeError of the native call with a text and in the driver's own line for one that
    # carries an OS error.
    interfaces = need_module("baselines.interfaces")
    grade = need_module("baselines.grade")
    driver = need_module(DRIVER)
    question = question_of(need_module, "c04.retry")
    # A refusal with nothing left of the stopped write (its entries dropped) is the row's right.
    subject, tool, _ = _scratch(need_module, tmp_path / "refused", FakeTool(drop_partial=True),
                                runner_of=lambda tool: DriverRunner(tool, driver, track_error=RuntimeError(TRACK_TOOL_TEXT)))
    got = subject.read(question)
    assert isinstance(got, interfaces.Reading)
    records = got.raw["records"]
    assert records["refused"] is True and records["rows_after_retry"] == 0
    assert TRACK_TOOL_TEXT in records["mismatch"]
    assert graded(need_module, question, got).status == "right"
    # An OS error is no refusal: a failed attempt that keeps the raw text, and its line is the
    # failed attempt's, never a grade.
    subject, tool, _ = _scratch(need_module, tmp_path / "os", FakeTool(drop_partial=True),
                                runner_of=lambda tool: DriverRunner(tool, driver, track_error=RuntimeError(OS_ERROR_TEXT)))
    got = subject.read(question)
    assert isinstance(got, interfaces.Attempt)
    assert OS_ERROR_TEXT in got.output
    assert interfaces.no_answer(got.id).startswith("no answer: failed attempt originblame-")
    with pytest.raises(grade.NotGradable):
        grade.grade(question, expectation(need_module, "c04.retry"), got)


def test_a_faulted_store_whose_call_ends_with_an_error_reads_stopped_and_a_store_with_no_fault_is_a_failed_attempt(need_module, tmp_path):
    # The table's rights read "... stopped for a row that expects an error" (c10.c05.f.digest
    # .originblame, check10.toml 4157). On a store with a planted fault any error end of the
    # tool's own call is its stop: a RuntimeError of the native call with a text, an OS error
    # included, the library's class with a text, the program's non-zero exit. The driver's own
    # errors, a panic, a signal and an empty text are failed attempts.
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    driver = need_module(DRIVER)
    faulted = question_of(need_module, "c05.f.digest.E/shard-3")
    plain = question_of(need_module, "c05.unmutated.E/shard-3")
    count = []

    def read(question, **ends):
        base = tmp_path / f"r{len(count)}"
        count.append(base)
        subject, _ = make_subject(need_module, base, _main_shards(ob, base),
                                  runner_of=lambda tool: DriverRunner(tool, driver, **ends))
        assert subject.track_all() == ()
        return subject.read(question)

    stops = [{"revoke_error": RuntimeError(OS_ERROR_TEXT)}, {"revoke_error": RuntimeError(REVOKE_TOOL_TEXT)},
             {"revoke_error": RuntimeError("")}, {"revoke_error": ToolError("unresolvable author name(s): authorA")},
             {"revoke_error": ToolError("")}, {"program": (PROGRAM_ERROR_EXIT, PROGRAM_ERROR_TEXT)}]
    for ends in stops:
        got = read(faulted, **ends)
        assert isinstance(got, interfaces.Reading), ends
        assert got.raw == {"stopped": True}, ends
    attempts = [{"revoke_error": TypeError("takes 3 positional arguments")},
                {"revoke_error": PanicException("panic from Rust code")}, {"program": (-9, "")}]
    for ends in attempts:
        got = read(faulted, **ends)
        assert isinstance(got, interfaces.Attempt), ends
    # No planted fault: an OS error, an empty text and the program's error are failed attempts,
    # the raw text kept; so is a text of any other kind (a store that was not faulted has no stop).
    for ends in (stops[0], stops[1], stops[2], stops[4], stops[5]):
        got = read(plain, **ends)
        assert isinstance(got, interfaces.Attempt), ends
    assert OS_ERROR_TEXT in read(plain, **stops[0]).output


# --- The faults of Check 5 --------------------------------------------------------------------------------


def _main_shards(ob, tmp_path):
    return [shard(ob, tmp_path, "E/shard-1", plain_documents(ob)),
            shard(ob, tmp_path, "E/shard-2", plain_documents(ob)),
            shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))]


def test_the_faulted_shard_alone_carries_the_fault_and_a_deleted_map_reads_not_covered(need_module, tmp_path):
    ob = need_module(MODULE)
    subject, tool = make_subject(need_module, tmp_path, _main_shards(ob, tmp_path))
    subject.track_all()
    got = {name: subject.read(question_of(need_module, f"c05.f.map-deleted.E/shard-{name}"))
           for name in (1, 2, 3)}
    assert got[1].raw == {"answer": "not covered"}, "the entries of E/shard-1 are deleted: no record"
    assert got[2].raw == {"answer": "does not contain"} and got[3].raw == {"answer": "contains"}
    faulted = [c for c in tool.calls if c["verb"] == "fault"]
    assert [c["options"]["fault"] for c in faulted] == ["f.map-deleted"] * 1
    for name, right in ((1, "right"), (2, "right"), (3, "right")):
        question = question_of(need_module, f"c05.f.map-deleted.E/shard-{name}")
        assert graded(need_module, question, got[name]).status == right
    # One wrong reading of the group: the shard holds a member of A, an answer of "does not
    # contain" is not the outcome of the listed row.
    interfaces = need_module("baselines.interfaces")
    question = question_of(need_module, "c05.f.map-deleted.E/shard-3")
    other = interfaces.Reading(question_id=question.id, raw={"answer": "does not contain"}, route=question.route)
    assert graded(need_module, question, other).status == "wrong"
    # A reading of a kind the tool holds no entry for is cannot answer, and the row allows it.
    mix = question_of(need_module, "c05.f.map-deleted.E/mix-1")
    assert graded(need_module, mix, subject.read(mix)).status == "cannot answer"


def test_a_fault_is_put_by_the_runners_putter_once_on_the_fresh_copy_before_the_revoke_and_never_skipped(need_module, tmp_path):
    ob = need_module(MODULE)
    question = question_of(need_module, "c05.f.digest.E/shard-3")
    # Without a putter the row is refused, never read as if the fault had been put.
    subject, tool = make_subject(need_module, tmp_path / "a", _main_shards(ob, tmp_path / "a"), putter=False)
    subject.track_all()
    with pytest.raises(ValueError, match="fault_putter"):
        subject.read(question)
    assert not calls_of(tool, "revoke"), "the unfaulted store was not read"
    # With one, it is called with the copy and the fault of the table, before the revoke of that copy.
    subject, tool = make_subject(need_module, tmp_path / "b", _main_shards(ob, tmp_path / "b"))
    subject.track_all()
    subject.read(question)
    putter = getattr(subject, "fault_putter", None)
    assert putter is not None and [fault for _, fault in putter.puts] == ["f.digest"]
    copy = putter.puts[0][0]
    assert copy.name.startswith("copy-") and copy != subject.tracked_store(subject.shards[2])
    verbs = [c["verb"] for c in tool.calls if c["store"] == copy]
    assert verbs.index("fault") < verbs.index("revoke") < verbs.index("generate-set")
    # The two steps that read the faulted copy are told so (the driver tells a stop of the tool
    # from a failure by it); no other step is.
    told = [c["verb"] for c in tool.calls if c["store"] == copy and c.get("options", {}).get("faulted")]
    assert told == ["revoke", "generate-set"]
    # The shards the fault does not name are read with no putter call, and no flag.
    subject.read(question_of(need_module, "c05.f.digest.E/shard-1"))
    assert len(putter.puts) == 1
    assert not [c for c in tool.calls if c["store"] != copy and c.get("options", {}).get("faulted")]


def test_a_digest_fault_is_a_stop_only_if_the_tool_ends_with_an_error(need_module, tmp_path):
    ob = need_module(MODULE)
    # The tool notices nothing: the reading is an answer where a stop is expected: wrong.
    subject, tool = make_subject(need_module, tmp_path / "a", _main_shards(ob, tmp_path / "a"))
    subject.track_all()
    question = question_of(need_module, "c05.f.digest.E/shard-3")
    got = subject.read(question)
    assert got.raw == {"answer": "contains"}
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["answer"]
    assert [c["options"]["fault"] for c in calls_of(tool, "fault")] == ["f.digest"]
    # The tool ends with an error that reports it: a stop, right.
    subject, tool = make_subject(need_module, tmp_path / "b", _main_shards(ob, tmp_path / "b"),
                                 tool=FakeTool(error_after=("f.digest",)))
    subject.track_all()
    got = subject.read(question)
    assert isinstance(got, need_module("baselines.interfaces").Reading)
    assert list(got.raw) == ["stopped"]
    assert graded(need_module, question, got).status == "right"
    # The shards the fault does not touch are read as they stand.
    for name, word in ((1, "does not contain"), (2, "does not contain")):
        other = subject.read(question_of(need_module, f"c05.f.digest.E/shard-{name}"))
        assert other.raw == {"answer": word}
    # The listing is cannot answer, as in the row of the listing group.
    listing = question_of(need_module, "c05.f.digest.listing")
    assert graded(need_module, listing, subject.read(listing)).status == "cannot answer"


def test_withdraw_c_reads_the_shard_by_the_file_of_c_and_cannot_answer_for_r1(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    subject, tool = make_subject(need_module, tmp_path, [
        shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))])
    subject.track_all()
    question = question_of(need_module, "c05.withdraw-C.unmutated.E/shard-3")
    got = subject.read(question)
    assert got.raw == {"answer": "does not contain"}, "a member of A is not a member of C"
    assert graded(need_module, question, got).status == "right"
    wrong, _ = make_subject(need_module, tmp_path / "w", [
        shard(ob, tmp_path / "w", "E/shard-3", plain_documents(ob, with_c=True))])
    wrong.track_all()
    assert graded(need_module, question, wrong.read(question)).status == "wrong"
    r1 = question_of(need_module, "c05.withdraw-C.unmutated.R1")
    assert subject.read(r1).raw == {"cannot_answer": missing_text(facts, "files")}


def test_the_shards_of_the_fresh_stores_read_as_the_unmutated_group_asks(need_module, tmp_path):
    ob = need_module(MODULE)
    shards = [shard(ob, tmp_path, "E/shard-1", plain_documents(ob)),
              shard(ob, tmp_path, "E-rr/shard-3", plain_documents(ob, with_a=True), target=("store", "rr")),
              shard(ob, tmp_path, "E-rr/shard-1", plain_documents(ob, with_c=True), target=("store", "rr"))]
    subject, _ = make_subject(need_module, tmp_path, shards)
    subject.track_all()
    for listed, word in (("c05.unmutated.E/shard-1", "does not contain"),
                         ("c05.unmutated.E-rr/shard-3", "contains"),
                         ("c05.unmutated.E-rr/shard-1", "does not contain")):
        question = question_of(need_module, listed)
        got = subject.read(question)
        assert got.raw == {"answer": word}
        assert graded(need_module, question, got).status == "right"


# --- c06: everything downstream of the unrecorded translation ------------------------------------------------


def test_downstream_of_the_translation_is_an_answer_the_table_never_calls_right(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    one = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True), target=("store", "no-capture"))
    subject, tool = make_subject(need_module, tmp_path, [one])
    subject.track_all()
    question = question_of(need_module, "c06.fixture.downstream.E/shard-3")
    got = subject.read(question)
    assert got.raw == {"answer": "contains"}
    # Every expected reading of the group is "may contain": no rule of this subject gives it.
    for listed in ("c06.fixture.downstream.E/shard-3", "c06.generated.downstream-of-translation"):
        assert expectation(need_module, listed, "small" if "generated" in listed else None)["answer"] == "may contain"
    verdict = graded(need_module, question, got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["answer"]
    for listed in ("c06.fixture.downstream.gen/T", "c06.fixture.downstream.R1"):
        other = question_of(need_module, listed)
        assert subject.read(other).raw == {"cannot_answer": missing_text(facts, "files")}
        assert graded(need_module, other, subject.read(other)).status == "cannot answer"


def test_a_generated_set_is_not_read_into_one_answer_the_table_gives_no_join_of_its_shards(need_module, tmp_path):
    # Changed 17 Nov 2026 (old: "one shard holds a revoked entry" read "contains" and none "does not contain": the driver's
    # own fold of the shards of the target ("set", "no-capture-10k"); the fold is class 3, by rule: the table reads
    # a shard and gives no join, as for Dolma). The row raises, so that the runner puts cannot answer with the table's words.
    ob = need_module(MODULE)
    sets = [shard(ob, tmp_path, "round-robin/shard-1", plain_documents(ob), target=("set", "no-capture-10k"), layout="round-robin"),
            shard(ob, tmp_path, "round-robin/shard-2", plain_documents(ob, with_a=True), target=("set", "no-capture-10k"),
                  layout="round-robin")]
    subject, _ = make_subject(need_module, tmp_path, sets)
    subject.track_all()
    question = question_of(need_module, "c06.generated.downstream-of-translation", "10k")
    with pytest.raises(ValueError, match="no join of the shards"):
        subject.read(question)
    assert need_module(MODULE).NO_JOIN_ROWS == need_module("baselines.dolma").OPEN_ROWS


# --- c09: the action of an artifact, the lists, the successors -------------------------------------------------------


def _fixture_run(ob, tmp_path):
    return [shard(ob, tmp_path, "E/shard-1", plain_documents(ob), layout="partitioned", run="E"),
            shard(ob, tmp_path, "E/shard-2", plain_documents(ob), layout="partitioned", run="E"),
            shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True), layout="partitioned", run="E")]


def test_contains_is_rewrite_and_does_not_contain_is_keep_and_the_lists_are_read_from_the_same_answers(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    subject, _ = make_subject(need_module, tmp_path, _fixture_run(ob, tmp_path))
    subject.track_all()
    for listed, word in (("c09.fixture.E/shard-1.action", "keep"), ("c09.fixture.E/shard-2.action", "keep"),
                         ("c09.fixture.E/shard-3.action", "rewrite")):
        question = question_of(need_module, listed)
        got = subject.read(question)
        assert got.raw == {"action": word}, listed
        assert graded(need_module, question, got).status == "right"
    counts = question_of(need_module, "c09.fixture.shards.counts")
    got = subject.read(counts)
    assert got.raw == {"keep": ["E/shard-1", "E/shard-2"], "rewrite": ["E/shard-3"],
                       "counts": {"keep": 2, "rewrite": 1}}
    assert graded(need_module, counts, got).status == "right"
    swapped = [shard(ob, tmp_path / "s", "E/shard-1", plain_documents(ob, with_a=True), layout="partitioned", run="E"),
               shard(ob, tmp_path / "s", "E/shard-2", plain_documents(ob), layout="partitioned", run="E"),
               shard(ob, tmp_path / "s", "E/shard-3", plain_documents(ob), layout="partitioned", run="E")]
    other, _ = make_subject(need_module, tmp_path / "s", swapped)
    other.track_all()
    verdict = graded(need_module, counts, other.read(counts))
    assert verdict.status == "wrong"
    question = question_of(need_module, "c09.fixture.E/shard-3.action")
    assert graded(need_module, question, other.read(question)).status == "wrong"
    for listed in ("c09.fixture.R1.action", "c09.fixture.R2.action"):
        question = question_of(need_module, listed)
        assert subject.read(question).raw == {"cannot_answer": missing_text(facts, "files")}


def test_the_kept_shards_and_the_number_of_rewritten_chunks_of_the_10k_set_are_read_per_layout(need_module, tmp_path):
    ob = need_module(MODULE)
    partitioned = [shard(ob, tmp_path, f"partitioned/shard-{n}", plain_documents(ob, with_a=(n == 2)),
                         target=("set", "10k"), layout="partitioned") for n in (1, 2, 3)]
    chunks = [shard(ob, tmp_path, f"round-robin/shard-{n}", plain_documents(ob, with_a=True),
                    target=("set", "10k"), layout="round-robin") for n in (1, 2)]
    subject, _ = make_subject(need_module, tmp_path, partitioned + chunks)
    subject.track_all()
    kept = subject.read(question_of(need_module, "c09.10k.partitioned.kept-shards"))
    assert kept.raw == {"keep": ["partitioned/shard-1", "partitioned/shard-3"], "rewrite": ["partitioned/shard-2"]}
    rewritten = subject.read(question_of(need_module, "c09.10k.round-robin.all-rewrite"))
    assert rewritten.raw == {"keep": 0, "rewrite": 2}


def test_the_successors_are_read_by_the_shard_rule_and_the_two_layouts_are_two_readings(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    expected = need_module("baselines.expected")
    grade = need_module("baselines.grade")
    one = shard(ob, tmp_path, "successor of E/shard-3", plain_documents(ob), successor=True)
    old = shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))
    subject, tool = make_subject(need_module, tmp_path, [one, old])
    subject.track_all()
    question = question_of(need_module, "c09.fixture.successor.E/shard-3")
    got = subject.read(question)
    assert got.raw == {"answer": "does not contain"}, "the successor shard is read, not the shard it replaces"
    assert graded(need_module, question, got).status == "right"
    assert subject.read(question_of(need_module, "c09.fixture.successor.R1")).raw == {
        "cannot_answer": missing_text(facts, "files")}
    layered = [shard(ob, tmp_path / "l", f"{layout}/shard-1", plain_documents(ob), target=("set", "10k"),
                     layout=layout, successor=True) for layout in ("partitioned", "round-robin")]
    other, _ = make_subject(need_module, tmp_path / "l", layered)
    other.track_all()
    ten = question_of(need_module, "c09.10k.successor-shards")
    # Changed 17 Nov 2026 (old: each layout's successor shards were folded into one word and graded right or wrong): the table
    # gives no join of several shards into one answer, so the row raises for each layout and the runner puts cannot answer.
    with pytest.raises(ValueError):
        other.read(ten)
    # Each layout that was tracked reads its OWN cannot answer (the line of the row follows the layouts' slots), not a stop.
    by_layout = other.read_by_layout(ten)
    assert set(by_layout) == {"partitioned", "round-robin"}
    assert all(r.raw == {"cannot_answer": need_module("baselines.dolma").NO_JOIN_TEXT} for r in by_layout.values())
    held = expected.concrete_values_by_layout(expected.expected_for("c09.10k.successor-shards"))
    assert grade.grade_layouts(ten, held, by_layout).status == "cannot answer"


def test_a_layout_with_no_tracked_store_is_not_attempted_never_cannot_answer(need_module, tmp_path):
    # "A tool that did not run cannot earn it": a layout none of whose shards was given has no
    # reading (None), and the row's line follows the order of the layouts' slots: with the other
    # layout right, it is "not attempted". The tracked layout is read as before.
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    expected = need_module("baselines.expected")
    grade = need_module("baselines.grade")
    ten = question_of(need_module, "c09.10k.successor-shards")
    held = expected.concrete_values_by_layout(expected.expected_for("c09.10k.successor-shards"))
    only = [shard(ob, tmp_path, "partitioned/shard-1", plain_documents(ob), target=("set", "10k"),
                  layout="partitioned", successor=True)]
    subject, _ = make_subject(need_module, tmp_path, only)
    assert subject.track_all() == ()
    # Changed 17 Nov 2026 (old: the tracked layout read "does not contain", the other None, the row "not attempted"): the row is
    # one the table gives no join of the shards for, so the layout that was tracked raises (the runner puts cannot answer
    # with the table's words); a layout with no shard given is read None before that, as the cases below show.
    by_layout = subject.read_by_layout(ten)
    assert by_layout["round-robin"] is None, "the layout with no shard given is not attempted"
    assert by_layout["partitioned"].raw == {"cannot_answer": need_module("baselines.dolma").NO_JOIN_TEXT}
    assert grade.grade_layouts(ten, held, by_layout).status == "not attempted", "no cannot-answer line for a layout never run on"
    both = [shard(ob, tmp_path / "b", f"{layout}/shard-1", plain_documents(ob), target=("set", "10k"), layout=layout, successor=True)
            for layout in ("partitioned", "round-robin")]
    tracked, _ = make_subject(need_module, tmp_path / "b", both)
    assert tracked.track_all() == ()
    assert grade.grade_layouts(ten, held, tracked.read_by_layout(ten)).status == "cannot answer"
    # A layout whose shards failed to track is that failed attempt, not "not attempted".
    broken, _ = make_subject(need_module, tmp_path / "f", [shard(
        ob, tmp_path / "f", "partitioned/shard-1", plain_documents(ob), target=("set", "10k"),
        layout="partitioned", successor=True)], tool=FakeTool(fail={"track": (1, "sandbox failed\n")}))
    failed = broken.track_all()
    assert len(failed) == 1
    got = broken.read_by_layout(ten)
    assert got["round-robin"] is None and isinstance(got["partitioned"], interfaces.Attempt)
    # Shards that were given and not tracked yet are not attempted either.
    fresh, _ = make_subject(need_module, tmp_path / "g", [shard(
        ob, tmp_path / "g", "partitioned/shard-1", plain_documents(ob), target=("set", "10k"),
        layout="partitioned", successor=True)])
    assert fresh.read_by_layout(ten) == {"partitioned": None, "round-robin": None}


# --- The mechanical bar: every routed question, the fields its rule names ---------------------------------------------


def routed_questions(need_module):
    return [q for q in ob_questions(need_module).values() if q.route is not None]


def named_fields(facts, question) -> set[str]:
    """The fields the rule of the question's row names for a reading that answers, from
    the rule texts of ``check10.toml`` (written here by hand, per listed row)."""
    listed = question.listed_row
    given = facts.core_given[listed]
    kind = artifact_kind(given.get("artifact"))
    if listed == "c03.spans.per-ordinal":
        return {"per_ordinal", "notes"}
    if listed == "c03.spans.total":
        return {"per_shard", "notes"}
    if listed in ("c04.kill-after-100", "c04.retry"):
        return {"records"}
    if kind not in ("shard", "set"):
        return {"cannot_answer"}
    if listed == "c09.fixture.shards.counts":
        return {"keep", "rewrite", "counts"}
    if listed in ("c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"):
        return {"keep", "rewrite"}
    if listed.endswith(".action"):
        return {"action"}
    return {"answer"}


def fields_problem(got_fields, wanted) -> str | None:
    """None when a reading holds exactly the wanted fields, else what differs."""
    got_fields, wanted = set(got_fields), set(wanted)
    if got_fields == wanted:
        return None
    return f"missing {sorted(wanted - got_fields)}, extra {sorted(got_fields - wanted)}"


def routed_counts_problem(facts) -> str | None:
    """None when the table's routed questions are the 81 counted by hand, group by group,
    else what differs."""
    if sum(count for _, count in ROUTED_BY_HAND) != ROUTED_TOTAL:
        return "the hand count is not 81"
    routed_groups = {r["given"]["group"] for r in facts.rows
                     if r["given"]["subject"] == SUBJECT and r.get("route")}
    if routed_groups != {group for group, _ in ROUTED_BY_HAND}:
        return f"the routed groups differ: {sorted(routed_groups ^ {g for g, _ in ROUTED_BY_HAND})}"
    for group, count in ROUTED_BY_HAND:
        asked = sum(len(facts.sizes_by_listed[listed]) for listed in facts.group_by_id[group]["rows"])
        if asked != count:
            return f"{group}: {asked} questions in the table, {count} by hand"
    return None


def test_the_routed_questions_number_81_by_hand_and_in_the_table(facts):
    assert routed_counts_problem(facts) is None


def test_every_routed_question_reads_exactly_the_fields_its_rule_names(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    questions = routed_questions(need_module)
    assert len(questions) == ROUTED_TOTAL
    # One shard of a name on a target: where a row names the shard only and another names its
    # run and layout, the one that carries the run and layout stands.
    pool: dict = {}
    for question in questions:
        for spec in shards_for(ob, tmp_path, facts, question, with_a=True):
            key = (spec.name, spec.target, spec.successor)
            if key not in pool or (pool[key].run is None and spec.run is not None):
                pool[key] = spec
    scratch = shard(ob, tmp_path, "c04/shard-500", tuple(doc(ob, "sources/B.jsonl", 3) for _ in range(500)),
                    target=("store", "scratch-500"))
    pool[(scratch.name, scratch.target, scratch.successor)] = scratch
    assert len({(s.name, s.target, s.successor) for s in pool.values()}) == len(pool)
    subject, tool = make_subject(need_module, tmp_path, list(pool.values()))
    assert subject.track_all() == ()
    problems = []
    for question in questions:
        wanted = named_fields(facts, question)
        if question.listed_row in need_module(MODULE).NO_JOIN_ROWS:
            # The two rows the table gives no join of the shards for raise (17 Nov 2026); the runner puts cannot answer.
            if facts.core_given[question.listed_row].get("layouts"):
                assert all(r.raw == {"cannot_answer": need_module("baselines.dolma").NO_JOIN_TEXT}
                           for r in subject.read_by_layout(question).values() if r is not None)
            else:
                with pytest.raises(ValueError, match="no join of the shards"):
                    subject.read(question)
            continue
        if facts.core_given[question.listed_row].get("layouts"):
            results = list(subject.read_by_layout(question).values())
        else:
            results = [subject.read(question)]
        for got in results:
            assert isinstance(got, interfaces.Reading) and got.question_id == question.id
            assert got.route == question.route
            problem = fields_problem(set(got.raw), wanted)
            if problem:
                problems.append(f"{question.id}: {problem}")
    assert not problems, problems[:5]


def test_the_field_check_catches_a_reader_that_drops_one_named_field():
    # The known-bad input of the check above: the counts of a shard list with one field dropped.
    assert fields_problem({"keep", "rewrite", "counts"}, {"keep", "rewrite", "counts"}) is None
    assert fields_problem({"keep", "rewrite"}, {"keep", "rewrite", "counts"}) == "missing ['counts'], extra []"
    assert fields_problem({"per_ordinal"}, {"per_ordinal", "notes"}) == "missing ['notes'], extra []"
    assert fields_problem({"answer", "diff"}, {"answer"}) == "missing [], extra ['diff']"


def test_no_reading_holds_a_diff_or_a_local_path(need_module, facts, tmp_path):
    ob = need_module(MODULE)
    subject, tool = make_subject(need_module, tmp_path, [
        shard(ob, tmp_path, "E/shard-3", plain_documents(ob, with_a=True))])
    subject.track_all()
    for listed in ("c05.unmutated.E/shard-3", "c05.f.map-deleted.E/mix-1"):
        got = subject.read(question_of(need_module, listed))
        text = json.dumps(got.raw, sort_keys=True)
        assert "diff" not in got.raw and str(tmp_path) not in text


# --- A set the plant generator wrote: its unparseable lines are no stop of a plan --------------------------------

# The generator's own options for a small set (docs/plant.md section 3). "Cases per group" is 3, and
# there are three groups, the first of them lines that are not JSON; every case sits in Z (section 4.1
# and the table of the sanitizer cases in section 4.5).
PLANT_SCRIPT = Path(__file__).resolve().parents[2] / "fixture" / "plant" / "generate.py"
PLANT_ARGUMENTS = ("--records", "800", "--seed", "7", "--length", "160", "--pairs", "24",
                   "--sanitizer-cases", "3", "--chunks", "20")
UNPARSEABLE_PLANTED = {"sources/Z.jsonl": 3}


@pytest.fixture(scope="module")
def planted_set(tmp_path_factory):
    """A set written by ``fixture/plant/generate.py`` in a folder of its own."""
    import subprocess

    folder = tmp_path_factory.mktemp("planted") / "set"
    done = subprocess.run([sys.executable, "-I", str(PLANT_SCRIPT), "--out", str(folder), *PLANT_ARGUMENTS],
                          capture_output=True, text=True, timeout=900, cwd=folder.parent, check=False)
    assert done.returncode == 0, done.stderr
    return folder


def _planted_rows(folder: Path) -> list[dict]:
    text = (folder / "plant_list.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line]


def _is_unparseable(row: dict) -> bool:
    return bool(row["sanitizer"]) and row["sanitizer"]["case"] == "unparseable"


def test_the_id_files_of_a_generated_set_pass_its_unparseable_lines_and_count_them_per_file(need_module, planted_set):
    ob = need_module(MODULE)
    rows = _planted_rows(planted_set)
    assert sum(1 for row in rows if _is_unparseable(row)) == UNPARSEABLE_PLANTED["sources/Z.jsonl"]
    try:
        held, failure = ob.id_files(planted_set), None
    except ValueError as error:
        held, failure = None, error
    assert failure is None, f"the id files stop on a line of the set: {failure!r}"
    # A line that is not JSON holds no record (the repository's own reader of a line takes it for a line
    # with no text); every other line gives its id and its file.
    for row in rows:
        if row["file"].startswith("sources/"):
            wanted = None if _is_unparseable(row) else row["file"]
            assert held.get(row["id"]) == wanted, row["id"]
    # Never dropped silently: the reader hands the count of each file to the caller that asks for it.
    assert "unparseable" in inspect.signature(ob.id_files).parameters, "id_files takes no place for its counts"
    counted: dict = {}
    again = ob.id_files(planted_set, unparseable=counted)
    assert again == held
    assert dict(counted) == UNPARSEABLE_PLANTED


def test_a_plan_of_a_shard_of_a_generated_set_resolves_its_lines_past_the_unparseable_ones(need_module, tmp_path,
                                                                                         planted_set):
    ob = need_module(MODULE)
    rows = _planted_rows(planted_set)
    build = tmp_path / "build"
    shutil.copytree(planted_set / "sources", build / "sources")
    # A shard of three lines: the first line of A and the first two lines of Z that hold a record.
    a_ids = [row["id"] for row in rows if row["file"] == "sources/A.jsonl"][:1]
    z_ids = [row["id"] for row in rows if row["file"] == "sources/Z.jsonl" and not _is_unparseable(row)][:2]
    chunk = "pipeline/partitioned/tokenizer-input/shard-1.jsonl"
    write_records(build / chunk, [*a_ids, *z_ids])
    prefix = build / "pipeline" / "partitioned" / "shards" / "shard-1"
    write_shard_files(prefix, [[11, 99], [21, 99], [31, 99]])
    try:
        found, failure = ob.track_plan(build, chunk, prefix), None
    except ValueError as error:
        found, failure = None, error
    assert failure is None, f"the plan stops on a line of the set: {failure!r}"
    assert [(d.section, d.tokens) for d in found] == [
        ("sources/A.jsonl", 2), ("sources/Z.jsonl", 2), ("sources/Z.jsonl", 2)]
