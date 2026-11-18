"""The Dolma toolkit as a subject of Check 10 (plan 5.5 Check 10; table README, section
"Check 10": the subject ``dolma``, "What a run hands a subject, and its conditions", "The
Dolma tie to an ordinal"; the ``c10.<group>.dolma`` rows of ``check10.toml``).

Every input is written by hand: a build folder of a few chunk files and record files, the
companion rows the tokenizer would have written (in an order unlike the input's), a tie
record, and a runner that is a stub. Nothing is started: no bubblewrap, no Dolma, no
environment's python.

What the table says, and what these tests hold the subject to:

- The tokenizer is run inside the sandbox, through the copy's own interpreter with ``-m``
  and the module of the toolkit's command line, ``dolma.cli`` (the toolkit has no
  ``dolma.__main__``), started in the build folder and handed paths relative to it, among
  them its tokenizer file and its two work folders. It is given a scratch folder for
  temporary files and told to stay offline in the environment of its step. The mixer is run
  on the chunk files of one layout of one build, with a filter that excludes nothing (the
  toolkit refuses a stream with no filter); its step binds the copy read-only like the
  tokenizer's.
- After every tokenizer run the number of companion rows for each chunk file is compared
  with the number of its lines. A run that writes no row for a chunk file that has lines
  is a failed attempt, not an answer, even when it ends with exit code 0. A failed attempt
  keeps no local path (neither in its command nor in its output).
- A row of the companion file is tied to a document by its ``src`` and ``loc`` and never by
  its place in the file. The tie is used only for a chunk file where the recorded tie show
  says "shown"; otherwise the question that needs an ordinal is cannot answer, with the
  missing tie (the row's own ``ordinal.not_shown.evidence``) as the evidence. Every answer
  that used the tie carries the tie's mark; one that did not does not.
- The shard rule reads a chunk file's rows by ``src`` alone: contains / does not contain /
  no record (the answer "not covered") / cannot answer. An artifact that is not a shard
  (a source file, a generation output, a release, an import, a mixture, a checkpoint, a
  listing) is cannot answer, with the missing record of the row for that kind as evidence.
- The faults of Checks 4 and 5 are put to the companion rows (``baselines.faults``); the
  rows the run holds are never changed.
- The second run of c04.retry: exit 0 is no refusal. A non-zero exit is a refusal only when its
  output ends in an error the toolkit raised by one of its own error classes (the classes of
  its error module, as a traceback prints them with the module's dotted path, or bare); the error's text, with
  local paths cut, is the mismatch named, and an empty text names none. Any other end is a
  failed attempt and never a reading: an error of no class of the toolkit, a signal (a
  negative code, or 128 plus the signal's number), no output, and anything else the last line
  does not say.
- A stop that did not land is no stop: the stopped writer's exit is the exit of the signal that
  was sent to it (a negative code, or 128 plus the signal's number); any other exit means the
  writer ended by itself, or by another signal, and the question is the failed attempt of the
  stopped writer, which records that exit. That holds for the rows after the stop and for the second
  run, which is the write made again after a stop. The rows after the stop say whether the
  companion rows showed while the writer ran or only at its end.
- A companion file torn by a stop is read as the rows that came out whole before the tear
  (a row that ends with its newline); it is never a crash. After a run that ended with exit
  code 0 a torn file is a failed attempt.
- A reading that says what the toolkit does not hold (a source, a generation output, a
  release, an import, a mixture, a checkpoint, the listing) needs a successful tokenizer run
  of its store: a run with no row for a chunk file that has lines is not one.

The names asked of ``baselines.dolma``:

- ``DolmaRun(build, rows, chunk_files, stopped_rows=None, retry_rows=None,
  retry_exit_code=None, retry_output="", token_file_tokens=None, retry_attempt_id="",
  retry_command=(), stopped_exit_code=None, stopped_signal=None, stopped_attempt_id="",
  stopped_command=(), stopped_output="", rows_showed_while_running=None)``: what one
  tokenizer run on one store or set left. ``build`` is the build folder; ``rows`` the
  companion rows (``baselines.companion.Row``); ``chunk_files`` maps an artifact (or shard)
  name to the relative path of its chunk file; the other fields are the records of the two
  Check 4 writer states (the rows after the stopped write, the rows and the end of the second
  run, and the number of tokens the token file holds; ``retry_attempt_id`` and
  ``retry_command`` name the second run as an attempt, for the failed attempt it gives when
  it crashed; ``stopped_exit_code``, ``stopped_attempt_id``, ``stopped_command`` and
  ``stopped_output`` are the end of the stopped writer and ``stopped_signal`` the number of
  the signal sent to it, for the failed attempt it gives when the stop did not land;
  ``rows_showed_while_running`` says whether the companion rows showed while the writer ran
  (True) or only at its end (False)).
- ``DolmaSubject(runs, *, tie_record, requests, split_pairs=())``: a subject (``name`` is
  ``dolma``). ``runs`` maps the target of a question (``baselines.record.attempt_target``)
  to a ``DolmaRun`` or to the failed ``Attempt`` of that run; ``tie_record`` is the path of
  the tie record; ``requests`` maps a request's words (``withdraw A``) to ``{"files":
  [relative paths of the files of the members it names], "record": <id, optional>}``;
  ``split_pairs`` is the plant list's split pairs, ``[{"pair", "members": [{"shard",
  "ordinal"}]}]``. ``read(question)`` gives a ``Reading`` whose ``raw`` holds exactly the
  values the row's rule names (or ``{"cannot_answer": <evidence>}``), whose route is the
  question's own, or ``Reading(question.id, None, None)`` for a question whose row records
  none; a run that failed gives that ``Attempt``; a target with no run is a ``ValueError``.
  ``used_tie(question)`` says whether the last answer to the question used the tie.
- ``tokens_command(env_dir, chunk_files, destination, *, tokenizer, work_dir)``,
  ``tokens_work_folders(work_dir)``, ``tokens_step(*, lock_prefix, work, home, build,
  env_dir, chunk_files, destination, tokenizer, work_dir, env)``, ``mix_command(env_dir,
  config)``, ``mix_step(*, lock_prefix, work, home, build, env_dir, config, env)``,
  ``mix_config(chunk_files, destination, *, work_dir)`` and ``write_mix_config(path,
  chunk_files, destination, *, work_dir)``: the commands and the mixer's config file
  (written as yaml). ``tokenizer`` is the tokenizer file and ``work_dir`` the folder of the
  tool's two work folders (``in`` and ``out`` inside it), both relative to the build folder;
  the caller makes the work folders of either tool. ``env`` is the
  environment of the step: it holds ``TMPDIR``, a folder inside the work folder. The sandbox
  clears the environment, so the step sets ``PATH`` (the copy's ``bin`` first, then
  ``/usr/bin``), ``LANG`` (``C.UTF-8``) and ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE``
  (``1``) itself; a value of the caller's that differs from them is a ``ValueError``.
- ``run_tokens(runner, argv, *, build, chunk_files, destination, attempt_id)``: calls
  ``runner(argv)`` once (it gives ``(exit_code, output)``), reads the rows the run left in
  the folder ``destination``, a path relative to the build folder (as in the command), and
  gives them as a list, or the failed ``Attempt``.
- ``read_run_rows(folder)``: the rows of every ``.csv`` and ``.csv.gz`` file of a folder, in
  the order of the file names. ``read_stopped_rows(folder)``: the same for a folder a stop
  left behind, where a torn file gives the rows that came out whole.

Left open and not pinned here: the reading of ``c06.generated.downstream-of-translation``
(its row names no artifact) and of ``c09.10k.successor-shards`` (its row names two layouts
and one answer for each successor shard); the rule of the table says nothing about how
several shards' readings become one; a partial row count after a run that wrote some rows
for a chunk file.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import csv
import gzip
import importlib
import json
import os
import re
import tomllib
import zlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
MODULE = "baselines.dolma"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")
PRIVATE_PATH = "/ho" + "me/someone/work"
LOCK = ["lock-command", "lock-file", "--", "low-priority"]

# The tokenizer file and the folder of the two work folders, both relative to the build folder.
TOKENIZER = "tokenizer/tokenizer.json"
TOKENS_WORK = "scratch/tokens-work"
TOKENS_ARGS = {"tokenizer": TOKENIZER, "work_dir": TOKENS_WORK}

# The end of a stopped writer that was stopped (killed by signal 9), as a runner records it.
STOP_ID = "attempt-stop"
STOP_COMMAND = (PRIVATE_PATH + "/venv-dolma/bin/python", "-m", "dolma.cli", "tokens",
                "--documents", "shard.jsonl")
LANDED_STOP = {"stopped_exit_code": -9, "stopped_signal": 9, "stopped_attempt_id": STOP_ID,
               "stopped_command": STOP_COMMAND, "stopped_output": "", "rows_showed_while_running": True}

CANNOT = "cannot_answer"

# What a request names: the file of the member (the table README, section 2).
REQUESTS = {
    "withdraw A": {"files": ["sources/A.jsonl"]},
    "withdraw A, record X@A": {"files": ["sources/A.jsonl"], "record": "A-001"},
    "withdraw C": {"files": ["sources/C.jsonl"]},
}


# --- The table, read from its files ---------------------------------------------------


def _mod(name: str):
    return importlib.import_module(name)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _read_toml(TABLE_DIR / "check10.toml")["row"]}


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows


def _missing_of(group: str) -> str:
    """The record the toolkit lacks, as the table writes it for the group's dolma row."""
    return _check10_rows()[f"c10.{group}.dolma"]["missing"]


def _not_shown_evidence(q) -> str:
    """The evidence of the question's own row for a tie that is not shown."""
    return _check10_rows()[q.row_id]["ordinal"]["not_shown"]["evidence"]


def _tie_mark() -> str:
    return _check10_rows()["c10.c03.spans.dolma"]["ordinal"]["shown"]["mark"]


@lru_cache(maxsize=1)
def _questions() -> dict:
    table10 = _mod("baselines.table10")
    return {q.id: q for q in table10.expand(table10.load_table10())}


def question(listed: str, size: str | None = None):
    """The question that puts a listed row to the Dolma subject at a size."""
    table10 = _mod("baselines.table10")
    return _questions()[table10.question_id(listed, "dolma", size)]


def target_of(q):
    return _mod("baselines.record").attempt_target(q)


# --- A hand-written world ---------------------------------------------------------------

ALWAYS = {"A-001": "sources/A.jsonl", "A-002": "sources/A.jsonl", "C-001": "sources/C.jsonl"}


def _record_file(identifier: str) -> str:
    if len(identifier) > 2 and identifier[1] == "-" and identifier[0] in "ABCZ":
        return f"sources/{identifier[0]}.jsonl"
    return f"gen/{identifier}.jsonl"


class World:
    """A build folder with chunk files and record files, the companion rows of every chunk
    file in an order unlike the input's, and the tie record."""

    def __init__(self, root: Path, shards: Mapping[str, list], paths: Mapping[str, str] | None = None):
        self.root = Path(root)
        self.build = self.root / "build"
        self.chunk_files: dict[str, str] = {}
        paths = paths or {}
        holders: dict[str, str] = dict(ALWAYS)
        for name, ids in shards.items():
            relative = paths.get(name) or f"pipeline/partitioned/tokenizer-input/{name.replace('/', '-')}.jsonl"
            self.chunk_files[name] = relative
            target = self.build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for entry in ids:
                identifier, text = entry if isinstance(entry, tuple) else (entry, f"text of {entry}")
                if not identifier.startswith("N-"):
                    # An id of the form N-nnn is held by no record file.
                    holders.setdefault(identifier, _record_file(identifier))
                lines.append(json.dumps({"id": identifier, "text": text}))
            target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        by_file: dict[str, list[str]] = {}
        for identifier, relative in holders.items():
            by_file.setdefault(relative, []).append(identifier)
        for relative, identifiers in by_file.items():
            target = self.build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(json.dumps({"id": i, "text": f"text of {i}"}) + "\n" for i in identifiers),
                encoding="utf-8")
        self.rows = self.make_rows()
        self.tie_path = self.root / "tie.json"

    @staticmethod
    def size_of(ordinal: int) -> int:
        """The tokens of the document with this ordinal: 3, 4, 5, 3, 4, 5, ..."""
        return 3 + ordinal % 3

    def make_rows(self) -> list:
        row_type = _mod("baselines.companion").Row
        rows = []
        for relative in self.chunk_files.values():
            lines = (self.build / relative).read_text(encoding="utf-8").splitlines()
            position = 0
            for ordinal, line in enumerate(lines):
                size = self.size_of(ordinal)
                rows.append(row_type(start=position, end=position + size,
                                     id=json.loads(line)["id"], src=relative, loc=ordinal + 1))
                position += size
        return rows[::-1]

    def row(self, name: str, loc: int):
        (found,) = [r for r in self.rows if r.src == self.chunk_files[name] and r.loc == loc]
        return found

    def without(self, name: str, loc: int) -> None:
        dropped = self.row(name, loc)
        self.rows = [r for r in self.rows if r is not dropped]

    def write_tie(self, shown: Mapping[str, bool] | bool = True) -> Path:
        tie = _mod("baselines.tie")
        results = []
        for name, relative in self.chunk_files.items():
            held = shown if isinstance(shown, bool) else shown.get(name, True)
            lines = len((self.build / relative).read_text(encoding="utf-8").splitlines())
            results.append({"chunk_file": relative, "shown": held, "documents": lines,
                            "lines": lines, "first_mismatch": None if held else 0})
        tie.write_result(self.tie_path, results, {"note": "hand-written"})
        return self.tie_path

    def run(self, dolma, **more):
        if "stopped_rows" in more or "retry_rows" in more:
            # A run of the two Check 4 states holds the end of its stopped writer: a stop that
            # landed (the exit of a signal), and whether the rows showed while the writer ran.
            more = {**LANDED_STOP, **more}
        return dolma.DolmaRun(build=self.build, rows=tuple(self.rows),
                              chunk_files=dict(self.chunk_files), **more)

    def subject(self, dolma, questions, *, shown: Mapping[str, bool] | bool = True,
                split_pairs=(), tie_file: bool = True, **more):
        if tie_file:
            self.write_tie(shown)
        run = self.run(dolma, **more)
        runs = {target_of(q): run for q in questions}
        return dolma.DolmaSubject(runs, tie_record=self.tie_path, requests=REQUESTS,
                                  split_pairs=split_pairs)


def counted_document(shards: Mapping[str, list]) -> dict:
    """A counting document of the plant script for shards with no identical documents:
    every document is recovered."""
    return {"documents": {"total": sum(len(i) for i in shards.values()), "shards": [
        {"shard": name, "documents": len(ids), "ids": list(ids), "identical": [],
         "labels": {"recovered": len(ids), "candidates": 0}}
        for name, ids in shards.items()]}}


def concrete_of(listed: str, size: str | None, **sources):
    expected = _mod("baselines.expected")
    return expected.concrete_values(expected.expected_for(listed, size, **sources))


def listed_expect(listed: str) -> dict:
    return dict(_listed_rows()[listed]["expect"])


SET_SHARDS = {
    "shard-1": ["Z-001", "Z-002", "Z-003", "Z-004", "Z-005"],
    "shard-2": ["Z-006", "Z-007", "Z-008", "Z-009"],
}
SET_PATHS = {
    "shard-1": "pipeline/round-robin/tokenizer-input/shard-1.jsonl",
    "shard-2": "pipeline/round-robin/tokenizer-input/shard-2.jsonl",
}
# The spans of SET_SHARDS: the tokens of ordinal n are 3 + n % 3 (3, 4, 5, 3, 4, ...).
SET_SPANS = {"shard-1": [3, 4, 5, 3, 4], "shard-2": [3, 4, 5, 3]}
SET_TOTALS = {"shard-1": 19, "shard-2": 15}


def set_world(tmp_path) -> World:
    return World(tmp_path, SET_SHARDS, SET_PATHS)


# --- Names -----------------------------------------------------------------------------


def test_names_of_the_dolma_subject(need_attribute):
    for name in ("DolmaSubject", "DolmaRun", "tokens_command", "tokens_step", "mix_command",
                 "mix_config", "write_mix_config", "run_tokens", "read_run_rows", "read_stopped_rows",
                 "tokens_work_folders", "mix_step"):
        need_attribute(MODULE, name)


def test_the_subject_is_named_dolma_and_is_a_subject(need_module, tmp_path):
    dolma = need_module(MODULE)
    subject = dolma.DolmaSubject({}, tie_record=tmp_path / "tie.json", requests=REQUESTS)
    assert subject.name == "dolma"
    assert isinstance(subject, _mod("baselines.interfaces").Subject)


# --- The commands -----------------------------------------------------------------------

CHUNKS = ["pipeline/partitioned/tokenizer-input/shard-1.jsonl",
          "pipeline/partitioned/tokenizer-input/shard-2.jsonl"]


def make_environment(tmp_path):
    """A copy of an environment whose interpreter is a link to a binary outside the work
    folder, as an uv interpreter's is. Nothing is started."""
    work = tmp_path / "work"
    binary = tmp_path / "earlier" / "cpython-3.12.13" / "bin"
    binary.mkdir(parents=True)
    (binary / "python").write_bytes(b"not a program")
    link = tmp_path / "earlier" / "cpython-3.12-linux-x86_64-gnu"
    link.symlink_to("cpython-3.12.13")
    env = work / "venv-dolma"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(link / "bin" / "python")
    (env / "pyvenv.cfg").write_text(f"home = {link}/bin\nversion_info = 3.12.13\n", encoding="utf-8")
    return work, env


def test_the_tokenizer_runs_through_the_interpreter_of_the_copy_with_relative_paths(need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    _work, env = make_environment(tmp_path)
    command = dolma.tokens_command(env, CHUNKS, "out/tokens", **TOKENS_ARGS)
    # The module of the toolkit's command line: the toolkit has no module ``dolma.__main__``.
    assert command[:4] == [str(env / "bin" / "python"), "-m", "dolma.cli", "tokens"]
    assert command == sandbox.tool_command(env, "dolma.cli", command[3:])
    for chunk in CHUNKS:
        assert command.count(chunk) == 1, chunk
    assert "out/tokens" in command
    assert all(not os.path.isabs(word) for word in command[1:]), command
    assert not any(word.endswith("/dolma") for word in command), "never the launcher of the copy"


def flag_values(command, flag):
    """The words after ``flag`` up to the next flag; none where the command has no such flag."""
    if flag not in command:
        return []
    start = command.index(flag) + 1
    stop = next((i for i in range(start, len(command)) if command[i].startswith("--")), len(command))
    return command[start:stop]


def test_the_tokenizer_command_names_its_tokenizer_end_token_process_and_work_folders(need_module, tmp_path):
    dolma = need_module(MODULE)
    _work, env = make_environment(tmp_path)
    command = dolma.tokens_command(env, CHUNKS, "out/tokens", **TOKENS_ARGS)
    assert flag_values(command, "--documents") == CHUNKS
    assert flag_values(command, "--destination") == ["out/tokens"]
    assert flag_values(command, "--tokenizer.name_or_path") == [TOKENIZER]
    assert flag_values(command, "--tokenizer.eos_token_id") == ["50256"]
    assert flag_values(command, "--processes") == ["1"]
    assert flag_values(command, "--work_dir.input") == [f"{TOKENS_WORK}/in"]
    assert flag_values(command, "--work_dir.output") == [f"{TOKENS_WORK}/out"]
    assert dolma.tokens_work_folders(TOKENS_WORK) == (f"{TOKENS_WORK}/in", f"{TOKENS_WORK}/out")
    # The chunk rows hold an id and a text: no id field is switched off, and nothing is given
    # a value the toolkit would read as the text "null".
    assert not any(word.startswith("--fields") or word == "null" for word in command)


def test_a_tokenizer_file_or_work_folder_that_is_not_a_relative_path_inside_the_build_is_refused(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    _work, env = make_environment(tmp_path)
    for bad in ("/abs/tokenizer.json", "../tokenizer.json", "tokenizer/../../tokenizer.json", ""):
        with pytest.raises(ValueError):
            dolma.tokens_command(env, CHUNKS, "out/tokens", tokenizer=bad, work_dir=TOKENS_WORK)
        with pytest.raises(ValueError):
            dolma.tokens_command(env, CHUNKS, "out/tokens", tokenizer=TOKENIZER, work_dir=bad)
        with pytest.raises(ValueError):
            dolma.tokens_work_folders(bad)
    # Neither is a default: a command that does not name its tokenizer is not built.
    with pytest.raises(TypeError):
        dolma.tokens_command(env, CHUNKS, "out/tokens")


def test_a_chunk_file_that_is_not_a_relative_path_inside_the_build_is_refused(need_module, tmp_path):
    dolma = need_module(MODULE)
    _work, env = make_environment(tmp_path)
    for bad in ("/abs/shard-1.jsonl", "../shard-1.jsonl", "pipeline/../../shard-1.jsonl"):
        with pytest.raises(ValueError):
            dolma.tokens_command(env, [bad], "out/tokens", **TOKENS_ARGS)
    with pytest.raises(ValueError):
        dolma.tokens_command("venv-dolma", CHUNKS, "out/tokens", **TOKENS_ARGS)


def _triples(argv, flag):
    return [(argv[i + 1], argv[i + 2]) for i, word in enumerate(argv) if word == flag and i + 2 < len(argv)]


def step_env(work):
    """The environment a step is given: a scratch folder for temporary files inside the work
    folder."""
    return {"TMPDIR": str(Path(work) / "scratch" / "tmp")}


def step_fixed(env_dir):
    """The words a step sets itself: offline, the language, and the path with the copy's own
    ``bin`` first."""
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "LANG": "C.UTF-8",
            "PATH": f"{env_dir}/bin:/usr/bin"}


def test_the_step_is_the_command_behind_the_lock_in_a_sandbox_that_starts_in_the_build(need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    work, env = make_environment(tmp_path)
    build = work / "build"
    home = sandbox.prepare_home(work)
    step = dolma.tokens_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                             chunk_files=CHUNKS, destination="out/tokens", env=step_env(work),
                             **TOKENS_ARGS)
    assert step[:len(LOCK)] == LOCK
    command = dolma.tokens_command(env, CHUNKS, "out/tokens", **TOKENS_ARGS)
    assert step[-len(command):] == command
    assert "--unshare-all" in step and "--clearenv" in step
    i = step.index("--chdir")
    assert step[i + 1] == str(build)
    binds = _triples(step, "--ro-bind")
    for folder in [*sandbox.interpreter_binds(env), str(env)]:
        assert (str(folder), str(folder)) in binds, folder
    assert (str(work), str(work)) in _triples(step, "--bind")
    assert [b for b in _triples(step, "--bind")] == [(str(work), str(work))], "only the work folder is writable"


def _bind_places(step, flag, folder):
    """The places in an argv where ``flag folder folder`` stands."""
    return [i for i, word in enumerate(step)
            if word == flag and step[i + 1:i + 3] == [str(folder), str(folder)]]


def _copy_is_bound_read_only_after_the_work_folder(step, work, env_dir):
    """True where the copy has its read-only bind and it comes after the writable bind of
    the folder that holds it, so that it wins there."""
    writable = _bind_places(step, "--bind", work)
    read_only = _bind_places(step, "--ro-bind", env_dir)
    return bool(writable) and bool(read_only) and min(read_only) > max(writable)


def test_the_copy_is_bound_read_only_in_the_step_and_after_the_bind_of_the_work_folder(need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    work, env = make_environment(tmp_path)
    build = work / "build"
    home = sandbox.prepare_home(work)
    step = dolma.tokens_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                             chunk_files=CHUNKS, destination="out/tokens", env=step_env(work),
                             **TOKENS_ARGS)
    assert len(_bind_places(step, "--ro-bind", env)) == 1
    assert len(_bind_places(step, "--bind", work)) == 1
    assert _copy_is_bound_read_only_after_the_work_folder(step, work, env)
    # A step built with the interpreter's folders alone leaves the copy writable through the
    # bind of the work folder: the check must see that.
    bare = sandbox.sandboxed_step(lock_prefix=LOCK, work=work, home=home, chdir=build,
                                  command=dolma.tokens_command(env, CHUNKS, "out/tokens", **TOKENS_ARGS),
                                  ro_binds=sandbox.interpreter_binds(env))
    assert _bind_places(bare, "--ro-bind", env) == []
    assert not _copy_is_bound_read_only_after_the_work_folder(bare, work, env)


def _setenv_of(step):
    return {step[i + 1]: step[i + 2] for i, word in enumerate(step) if word == "--setenv"}


def test_the_tokenizer_step_binds_its_tokenizer_file_read_only_and_sets_its_environment(need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    work, env = make_environment(tmp_path)
    build = work / "build"
    home = sandbox.prepare_home(work)
    step = dolma.tokens_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                             chunk_files=CHUNKS, destination="out/tokens", env=step_env(work),
                             **TOKENS_ARGS)
    # The tokenizer file is bound read-only, at its own place in the build folder, after the
    # writable bind of the work folder that holds it.
    tokenizer = build / TOKENIZER
    read_only = _bind_places(step, "--ro-bind", tokenizer)
    assert len(read_only) == 1 and read_only[0] > max(_bind_places(step, "--bind", work))
    assert [b for b in _triples(step, "--bind")] == [(str(work), str(work))], "only the work folder is writable"
    # The sandbox clears the environment: the step sets the scratch folder of temporary files
    # and the words a tool run needs, after the clearing.
    variables = _setenv_of(step)
    assert variables.get("TMPDIR") == step_env(work)["TMPDIR"]
    for name, value in step_fixed(env).items():
        assert variables.get(name) == value, name
    assert step.index("--clearenv") < step.index("--setenv")


def test_the_environment_of_a_step_needs_a_scratch_folder_inside_the_work_folder_and_stays_offline(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    work, env = make_environment(tmp_path)
    build = work / "build"
    home = sandbox.prepare_home(work)

    def step_with(variables):
        return dolma.tokens_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                                 chunk_files=CHUNKS, destination="out/tokens", env=variables, **TOKENS_ARGS)

    good = step_env(work)
    # The words the step sets may be given as it sets them, and never another way.
    assert _setenv_of(step_with({**good, **step_fixed(env)})).get("HF_HUB_OFFLINE") == "1"
    for bad in ({}, {"TMPDIR": "scratch/tmp"}, {"TMPDIR": str(tmp_path / "elsewhere")},
                {"TMPDIR": str(work)}, {"TMPDIR": str(work / ".." / "elsewhere")},
                {**good, "HF_HUB_OFFLINE": "0"}, {**good, "TRANSFORMERS_OFFLINE": "0"},
                {**good, "LANG": "de_DE.UTF-8"}, {**good, "PATH": "/usr/bin"},
                {**good, "HOME": str(work / "home")}):
        with pytest.raises(ValueError):
            step_with(bad)


def test_the_mixer_step_binds_the_copy_read_only_and_sets_its_environment(need_module, tmp_path):
    dolma = need_module(MODULE)
    sandbox = _mod("baselines.sandbox")
    work, env = make_environment(tmp_path)
    build = work / "build"
    home = sandbox.prepare_home(work)
    step = dolma.mix_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                          config="out/mix-config.json", env=step_env(work))
    assert step[:len(LOCK)] == LOCK
    command = dolma.mix_command(env, "out/mix-config.json")
    assert step[-len(command):] == command
    assert step[step.index("--chdir") + 1] == str(build)
    binds = _triples(step, "--ro-bind")
    for folder in [*sandbox.interpreter_binds(env), str(env)]:
        assert (str(folder), str(folder)) in binds, folder
    assert len(_bind_places(step, "--ro-bind", env)) == 1 and len(_bind_places(step, "--bind", work)) == 1
    assert _copy_is_bound_read_only_after_the_work_folder(step, work, env)
    assert [b for b in _triples(step, "--bind")] == [(str(work), str(work))], "only the work folder is writable"
    variables = _setenv_of(step)
    assert variables.get("TMPDIR") == step_env(work)["TMPDIR"]
    for name, value in step_fixed(env).items():
        assert variables.get(name) == value, name
    for bad in ({}, {"TMPDIR": str(tmp_path / "elsewhere")}, {**step_env(work), "HF_HUB_OFFLINE": "0"},
                {**step_env(work), "PATH": "/usr/bin"}):
        with pytest.raises(ValueError):
            dolma.mix_step(lock_prefix=LOCK, work=work, home=home, build=build, env_dir=env,
                           config="out/mix-config.json", env=bad)


# The toolkit's mixer refuses a stream with neither a filter nor a span replacement, so the
# stream holds a filter whose one exclusion names a text no chunk holds.
NEVER_MATCHING_EXCLUDE = '.text == "this text never occurs"'
# The folder of the mixer's two work folders (in and out inside it), relative to the build
# folder, and the size at which the mixer starts a new output file.
MIX_WORK = "scratch/mix-work"
MIX_OUTPUT_MAX_BYTES = 1000000000


def test_the_mixer_takes_the_chunk_files_of_one_layout_through_a_filter_that_excludes_nothing(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    _work, env = make_environment(tmp_path)
    config = dolma.mix_config(CHUNKS, "out/mix", work_dir=MIX_WORK)
    text = json.dumps(config)
    for chunk in CHUNKS:
        assert text.count(chunk) == 1, chunk
    assert "out/mix" in text and "/ho" + "me/" not in text
    # One stream, with the filter the toolkit asks for: the jq syntax and one exclusion that no
    # document of a chunk file meets.
    (stream,) = config["streams"]
    assert stream.get("filter") == {"syntax": "jq", "exclude": [NEVER_MATCHING_EXCLUDE]}
    # Nothing else of the stream or of the config can leave a document out: no inclusion, no
    # span replacement, no minimum text length, no discarded field. The size at which an output
    # file is closed and a new one started leaves none out either.
    assert set(stream) == {"name", "documents", "output", "filter"}
    assert set(stream["output"]) == {"path", "max_size_in_bytes"}
    assert stream["output"]["max_size_in_bytes"] == MIX_OUTPUT_MAX_BYTES
    assert set(config) == {"streams", "work_dir", "processes"} and config["processes"] == 1
    other = "pipeline/round-robin/tokenizer-input/shard-1.jsonl"
    with pytest.raises(ValueError):
        dolma.mix_config([CHUNKS[0], other], "out/mix", work_dir=MIX_WORK)
    command = dolma.mix_command(env, "out/mix-config.yaml")
    assert command == [str(env / "bin" / "python"), "-m", "dolma.cli", "-c", "out/mix-config.yaml", "mix"]


# The config file the mixer is given is yaml, in the layout the toolkit's command line reads
# its config in: nested blocks, a dash for each list item, every text in single quotes.
MIX_CONFIG_YAML = """\
streams:
  - name: 'chunks'
    documents:
      - 'pipeline/partitioned/tokenizer-input/shard-1.jsonl'
      - 'pipeline/partitioned/tokenizer-input/shard-2.jsonl'
    output:
      path: 'out/mix'
      max_size_in_bytes: 1000000000
    filter:
      syntax: 'jq'
      exclude:
        - '.text == "this text never occurs"'
work_dir:
  input: 'scratch/mix-work/in'
  output: 'scratch/mix-work/out'
processes: 1
"""


def test_the_mixer_config_is_written_as_yaml(need_module, tmp_path):
    dolma = need_module(MODULE)
    written = tmp_path / "mix-config.yaml"
    dolma.write_mix_config(written, CHUNKS, "out/mix", work_dir=MIX_WORK)
    assert written.read_text(encoding="utf-8") == MIX_CONFIG_YAML
    # A single quote in a path is doubled inside single quotes, as yaml writes it.
    quoted = tmp_path / "quoted.yaml"
    dolma.write_mix_config(quoted, ["pipeline/a/tokenizer-input/it's.jsonl"], "out/mix", work_dir=MIX_WORK)
    assert "      - 'pipeline/a/tokenizer-input/it''s.jsonl'\n" in quoted.read_text(encoding="utf-8")
    # A path with a line break cannot be written on one line: refused, and no file is made.
    refused = tmp_path / "refused.yaml"
    with pytest.raises(ValueError):
        dolma.write_mix_config(refused, ["pipeline/a/tokenizer-input/two\nlines.jsonl"], "out/mix",
                               work_dir=MIX_WORK)
    assert not refused.exists()


def test_the_mixer_is_given_work_folders_inside_the_build_that_the_caller_makes(need_module, tmp_path):
    dolma = need_module(MODULE)
    config = dolma.mix_config(CHUNKS, "out/mix", work_dir=MIX_WORK)
    assert config.get("work_dir") == {"input": f"{MIX_WORK}/in", "output": f"{MIX_WORK}/out"}
    # The work folder is a relative path inside the build folder, as every other path is, and
    # it is not a default: a config that does not name it is not built.
    refused = tmp_path / "refused.yaml"
    for bad in ("/abs/work", "../work", "work/../../work", ""):
        with pytest.raises(ValueError):
            dolma.mix_config(CHUNKS, "out/mix", work_dir=bad)
        with pytest.raises(ValueError):
            dolma.write_mix_config(refused, CHUNKS, "out/mix", work_dir=bad)
    assert not refused.exists()
    with pytest.raises(TypeError):
        dolma.mix_config(CHUNKS, "out/mix")
    with pytest.raises(TypeError):
        dolma.write_mix_config(refused, CHUNKS, "out/mix")


# --- A run, and the row count check after it -------------------------------------------


def write_companion(path: Path, rows, *, packed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if packed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row.start, row.end, row.id, row.src, row.loc])


def stub_runner(calls, writer, exit_code=0, output=""):
    def run(argv):
        calls.append(list(argv))
        writer()
        return exit_code, output
    return run


def run_world(tmp_path):
    return World(tmp_path, {"E/shard-1": ["B-001", "Z-001", "Z-002"], "E/shard-2": ["B-002", "Z-003"]})


def test_the_rows_of_a_run_are_the_rows_of_the_files_it_left_in_name_order(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = run_world(tmp_path)
    first, second = world.rows[:2], world.rows[2:]
    folder = tmp_path / "out"
    write_companion(folder / "b.csv", second)
    write_companion(folder / "a.csv.gz", first, packed=True)
    (folder / "notes.txt").write_text("not a companion file", encoding="utf-8")
    assert list(dolma.read_run_rows(folder)) == [*first, *second]


def test_a_run_that_wrote_rows_for_every_chunk_file_gives_its_rows_and_runs_once(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = run_world(tmp_path)
    calls = []
    out = world.build / "out"
    runner = stub_runner(calls, lambda: write_companion(out / "rows.csv.gz", world.rows, packed=True))
    argv = ["step", "of", "the", "run"]
    got = dolma.run_tokens(runner, argv, build=world.build, chunk_files=list(world.chunk_files.values()),
                           destination="out", attempt_id="attempt-1")
    assert calls == [argv]
    assert got == world.rows


def test_the_destination_of_the_command_and_of_the_rows_is_one_place_relative_to_the_build(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    _work, env = make_environment(tmp_path)
    world = run_world(tmp_path)
    chunks = list(world.chunk_files.values())
    argv = dolma.tokens_command(env, chunks, "out/tokens", **TOKENS_ARGS)
    written = world.build / "out" / "tokens"
    calls = []
    runner = stub_runner(calls, lambda: write_companion(written / "rows.csv.gz", world.rows, packed=True))
    got = dolma.run_tokens(runner, argv, build=world.build, chunk_files=chunks,
                           destination="out/tokens", attempt_id="attempt-5")
    assert calls == [argv] and got == world.rows
    # The same run with its rows in another folder than the destination inside the build left
    # nothing the checker reads: a failed attempt.
    elsewhere = tmp_path / "elsewhere"
    runner = stub_runner([], lambda: write_companion(elsewhere / "out" / "other" / "rows.csv", world.rows))
    missed = dolma.run_tokens(runner, argv, build=world.build, chunk_files=chunks,
                              destination="out/other", attempt_id="attempt-6")
    assert isinstance(missed, interfaces.Attempt) and missed.exit_code == 0


def test_a_destination_that_is_not_a_relative_path_inside_the_build_is_refused_before_the_run(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    world = run_world(tmp_path)
    calls = []
    runner = stub_runner(calls, lambda: None)
    for bad in (str(world.build / "out"), "../out", "out/../../out", ""):
        with pytest.raises(ValueError):
            dolma.run_tokens(runner, ["step"], build=world.build,
                             chunk_files=list(world.chunk_files.values()),
                             destination=bad, attempt_id="attempt-7")
    assert calls == [], "the tool is not started with a destination that cannot be read"


def test_exit_zero_with_no_row_for_a_chunk_file_that_has_lines_is_a_failed_attempt(need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    world = run_world(tmp_path)
    second = world.chunk_files["E/shard-2"]
    kept = [r for r in world.rows if r.src != second]
    out = world.build / "out"
    argv = ["step", "of", "the", "run"]
    runner = stub_runner([], lambda: write_companion(out / "rows.csv", kept), exit_code=0, output="done")
    got = dolma.run_tokens(runner, argv, build=world.build, chunk_files=list(world.chunk_files.values()),
                           destination="out", attempt_id="attempt-2")
    assert isinstance(got, interfaces.Attempt)
    assert got.id == "attempt-2" and got.exit_code == 0
    assert got.command == tuple(argv)
    assert second in got.output
    # A chunk file with no lines needs no row: the same run on a build whose second chunk
    # file is empty is not a failure.
    (world.build / second).write_text("", encoding="utf-8")
    again_out = world.build / "again"
    runner = stub_runner([], lambda: write_companion(again_out / "rows.csv", kept))
    again = dolma.run_tokens(runner, argv, build=world.build, chunk_files=list(world.chunk_files.values()),
                             destination="again", attempt_id="attempt-3")
    assert not isinstance(again, interfaces.Attempt)


def test_a_failed_run_keeps_no_local_path_and_can_be_written_as_an_attempt(need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    record = _mod("baselines.record")
    world = run_world(tmp_path)
    argv = [PRIVATE_PATH + "/venv-dolma/bin/python", "-m", "dolma.cli", "tokens"]
    output = (f"Traceback: File \"{PRIVATE_PATH}/venv-dolma/lib/x.py\", line 3\n"
              f"OSError: cannot open {PRIVATE_PATH}/build/shard-1.jsonl")
    runner = stub_runner([], lambda: None, exit_code=1, output=output)
    got = dolma.run_tokens(runner, argv, build=world.build, chunk_files=list(world.chunk_files.values()),
                           destination="out", attempt_id="attempt-4")
    assert isinstance(got, interfaces.Attempt) and got.exit_code == 1
    assert "OSError" in got.output
    private = "/ho" + "me/"
    assert private not in got.output and not any(private in word for word in got.command)
    line = record.attempt_line(got, ["c03.spans.per-ordinal|dolma@small"], commit="abcdef0")
    record.write_attempts(tmp_path / "attempts.jsonl", [line])


# --- A companion file torn by a stop ---------------------------------------------------------


def companion_text(rows) -> str:
    return "".join(f"{r.start},{r.end},{r.id},{r.src},{r.loc}\n" for r in rows)


def write_torn_by_a_stop(path: Path, rows, tail: str = "") -> None:
    """A gzip companion file whose writer was stopped: the whole rows and then ``tail`` (a row
    that is not whole) came out, and the end-of-stream marker never did, so that reading it to
    its end raises an EOFError."""
    path.parent.mkdir(parents=True, exist_ok=True)
    packer = zlib.compressobj(wbits=31)
    path.write_bytes(packer.compress((companion_text(rows) + tail).encode("utf-8"))
                     + packer.flush(zlib.Z_SYNC_FLUSH))


def write_torn_by_junk(path: Path, rows) -> None:
    """A complete gzip companion file with bytes after its end that are no gzip member: reading
    it to its end raises a BadGzipFile."""
    write_companion(path, rows, packed=True)
    with open(path, "ab") as handle:
        handle.write(b"junk that is no member of a gzip file")


def test_a_companion_file_torn_by_a_stop_gives_the_rows_that_came_out_whole(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = run_world(tmp_path)
    rows = world.rows
    assert len(rows) == 5
    # The stop came in the middle of a row: the rows before it are read, the half row is not.
    folder = tmp_path / "eof"
    write_torn_by_a_stop(folder / "a.csv.gz", rows[:3], tail="10,12,Z-00")
    assert list(dolma.read_stopped_rows(folder)) == rows[:3]
    # A last row that is whole but lacks its newline is not known to be whole: it is left out.
    folder = tmp_path / "no-newline"
    write_torn_by_a_stop(folder / "a.csv.gz", rows[:3], tail=companion_text(rows[3:4]).rstrip("\n"))
    assert list(dolma.read_stopped_rows(folder)) == rows[:3]
    # The same torn file beside a file that is whole: the rows of both, in the order of the names.
    folder = tmp_path / "beside"
    write_torn_by_a_stop(folder / "a.csv.gz", rows[:3], tail="5,9,Z")
    write_companion(folder / "b.csv", rows[3:])
    assert list(dolma.read_stopped_rows(folder)) == [*rows[:3], *rows[3:]]
    # A file with bytes after its end that are no gzip member: its rows are whole.
    folder = tmp_path / "junk"
    write_torn_by_junk(folder / "a.csv.gz", rows[:4])
    assert list(dolma.read_stopped_rows(folder)) == rows[:4]
    # A torn file with no whole row, and one cut inside its header, give no row.
    folder = tmp_path / "none"
    write_torn_by_a_stop(folder / "a.csv.gz", [], tail="5,9,Z")
    (folder / "b.csv.gz").write_bytes(gzip.compress(b"1,2,x,y,1\n")[:5])
    assert list(dolma.read_stopped_rows(folder)) == []
    # A folder that does not exist holds no row, as for read_run_rows; a whole folder reads as
    # read_run_rows reads it.
    assert list(dolma.read_stopped_rows(tmp_path / "missing")) == []
    whole = tmp_path / "whole"
    write_companion(whole / "a.csv.gz", rows[:2], packed=True)
    write_companion(whole / "b.csv", rows[2:])
    assert list(dolma.read_stopped_rows(whole)) == list(dolma.read_run_rows(whole))


def test_the_rows_of_a_torn_file_read_the_stop_as_what_they_count(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = scratch_world(tmp_path)
    q = question("c04.kill-after-100")
    held = scratch_rows(world, 151)
    folder = tmp_path / "left-by-the-stop"
    write_torn_by_a_stop(folder / "part-0-00000.csv.gz", held[:150], tail=companion_text(held[150:]).rstrip("\n"))
    after_the_stop = tuple(dolma.read_stopped_rows(folder))
    got = world.subject(dolma, [q], stopped_rows=after_the_stop).read(q)
    assert got.raw["records"]["rows_after_stop"] == 150
    assert grade.grade(q, listed_expect(q.listed_row), got).status == "wrong"


def test_a_companion_file_torn_after_a_run_that_ended_with_exit_0_is_a_failed_attempt_never_a_crash(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    world = run_world(tmp_path)
    chunks = list(world.chunk_files.values())
    for number, write in enumerate((
            lambda path: write_torn_by_a_stop(path, world.rows, tail="10,12,Z-00"),
            lambda path: write_torn_by_junk(path, world.rows))):
        out = world.build / f"out-{number}"
        runner = stub_runner([], lambda: write(out / "rows.csv.gz"), exit_code=0, output="done")
        got = dolma.run_tokens(runner, ["step"], build=world.build, chunk_files=chunks,
                               destination=f"out-{number}", attempt_id=f"attempt-torn-{number}")
        assert isinstance(got, interfaces.Attempt), number
        assert got.id == f"attempt-torn-{number}" and got.exit_code == 0
        assert "cannot be read" in got.output and "done" in got.output


# --- The tie ----------------------------------------------------------------------------


def test_the_spans_of_a_shard_are_end_minus_start_of_the_row_tied_by_src_and_loc(need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    world = set_world(tmp_path)
    q = question("c03.spans.per-ordinal", "small")
    subject = world.subject(dolma, [q])
    assert world.rows != sorted(world.rows, key=lambda r: (r.src, r.loc)), "the file order is not the input order"
    reading = subject.read(q)
    assert isinstance(reading, interfaces.Reading)
    assert reading.question_id == q.id and reading.route == q.route
    assert set(reading.raw) == {"per_ordinal", "notes"}
    assert {name: list(held) for name, held in reading.raw["per_ordinal"].items()} == SET_SPANS
    assert any("end-of-sequence" in note for note in reading.raw["notes"])
    assert subject.used_tie(q) is True


def test_a_document_with_no_tied_row_is_a_gap_in_its_span_never_its_neighbour(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = set_world(tmp_path)
    world.without("shard-2", 3)
    q = question("c03.spans.per-ordinal", "small")
    reading = world.subject(dolma, [q]).read(q)
    assert list(reading.raw["per_ordinal"]["shard-2"]) == [3, 4, None, 3]
    assert list(reading.raw["per_ordinal"]["shard-1"]) == SET_SPANS["shard-1"]


def test_the_totals_select_rows_by_src_alone_and_need_no_tie(need_module, tmp_path):
    dolma = need_module(MODULE)
    row_type = _mod("baselines.companion").Row
    world = set_world(tmp_path)
    # A row of another layout's chunk file with the same base name must not be counted.
    world.rows.append(row_type(start=0, end=100, id="Z-001",
                               src="pipeline/partitioned/tokenizer-input/shard-1.jsonl", loc=1))
    q = question("c03.spans.total", "small")
    subject = world.subject(dolma, [q], shown={"shard-1": True, "shard-2": False})
    reading = subject.read(q)
    assert set(reading.raw) == {"per_shard", "notes"}
    assert dict(reading.raw["per_shard"]) == SET_TOTALS
    assert any("end-of-sequence" in note for note in reading.raw["notes"])
    assert subject.used_tie(q) is False


@pytest.mark.parametrize("listed", ["c03.spans.per-ordinal", "c03.records.capture",
                                    "c03.records.position-recovery", "c03.records.token-matching"])
def test_where_the_tie_is_not_shown_for_a_chunk_file_the_question_is_cannot_answer(need_module, tmp_path, listed):
    dolma = need_module(MODULE)
    world = set_world(tmp_path)
    q = question(listed, "small")
    subject = world.subject(dolma, [q], shown={"shard-1": True, "shard-2": False})
    reading = subject.read(q)
    assert dict(reading.raw) == {CANNOT: _not_shown_evidence(q)}
    assert reading.route == q.route
    assert subject.used_tie(q) is False
    # No result recorded at all reads "not shown" too.
    bare = world.subject(dolma, [q], tie_file=False)
    (world.root / "tie.json").unlink(missing_ok=True)
    assert dict(bare.read(q).raw) == {CANNOT: _not_shown_evidence(q)}


def test_the_documents_rows_and_the_split_pairs_are_cannot_answer_where_their_tie_is_not_shown(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    world = set_world(tmp_path / "set")
    pairs = [{"pair": "pair-2", "members": [{"shard": "shard-1", "ordinal": 4},
                                            {"shard": "shard-2", "ordinal": 1}]}]
    unshown = {"shard-1": True, "shard-2": False}
    for listed, extra in (("c06.generated.documents-mapped", {}), ("c03.pairs.split", {"split_pairs": pairs})):
        q = question(listed, "small")
        subject = world.subject(dolma, [q], shown=unshown, **extra)
        reading = subject.read(q)
        assert dict(reading.raw) == {CANNOT: _not_shown_evidence(q)}, listed
        assert subject.used_tie(q) is False, listed
    # A pair whose two members lie in shards where the tie is shown is read.
    inside = [{"pair": "pair-1", "members": [{"shard": "shard-1", "ordinal": 0},
                                             {"shard": "shard-1", "ordinal": 4}]}]
    q = question("c03.pairs.split", "small")
    reading = world.subject(dolma, [q], shown=unshown, split_pairs=inside).read(q)
    assert set(reading.raw) == {"record"}
    # A row of the fixture reads its own shard's tie alone.
    fixture = World(tmp_path / "fixture", FIXTURE_SHARDS)
    stopped = question("c06.fixture.documents-mapped.E/shard-2")
    held = question("c06.fixture.documents-mapped.E/shard-1")
    subject = fixture.subject(dolma, [stopped, held], shown={"E/shard-2": False})
    assert dict(subject.read(stopped).raw) == {CANNOT: _not_shown_evidence(stopped)}
    assert subject.used_tie(stopped) is False
    assert set(subject.read(held).raw) == {"documents", "recovered", "unknown"}
    assert subject.used_tie(held) is True


@pytest.mark.parametrize("listed", ["c03.records.capture", "c03.records.position-recovery",
                                    "c03.records.token-matching"])
def test_the_records_are_the_ids_of_the_tied_rows_in_ordinal_order(need_module, tmp_path, listed):
    dolma = need_module(MODULE)
    world = set_world(tmp_path)
    q = question(listed, "small")
    subject = world.subject(dolma, [q])
    reading = subject.read(q)
    assert set(reading.raw) == {"per_ordinal"}
    assert {name: list(ids) for name, ids in reading.raw["per_ordinal"].items()} == SET_SHARDS
    assert subject.used_tie(q) is True
    world.without("shard-1", 2)
    gap = world.subject(dolma, [q]).read(q)
    assert list(gap.raw["per_ordinal"]["shard-1"]) == ["Z-001", None, "Z-003", "Z-004", "Z-005"]


def test_a_split_pair_reads_the_chunk_file_and_the_line_the_tied_row_names_and_no_id(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = set_world(tmp_path)
    pairs = [{"pair": "pair-2", "members": [{"shard": "shard-1", "ordinal": 4},
                                            {"shard": "shard-2", "ordinal": 1}]}]
    q = question("c03.pairs.split", "small")
    subject = world.subject(dolma, [q], split_pairs=pairs)
    reading = subject.read(q)
    assert set(reading.raw) == {"record"}
    assert [dict(p) for p in reading.raw["record"]] == [{"pair": "pair-2", "members": [
        {"shard": "shard-1", "line": 5}, {"shard": "shard-2", "line": 2}]}]
    assert subject.used_tie(q) is True
    world.without("shard-2", 2)
    gap = world.subject(dolma, [q], split_pairs=pairs).read(q)
    assert [dict(m) for m in gap.raw["record"][0]["members"]] == [
        {"shard": "shard-1", "line": 5}, {"shard": "shard-2", "line": None}]


def test_an_answer_that_used_the_tie_carries_its_mark_and_one_that_did_not_does_not(need_module, tmp_path):
    dolma = need_module(MODULE)
    record = _mod("baselines.record")
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    used = question("c03.records.capture", "small")
    total = question("c03.spans.total", "small")
    subject = world.subject(dolma, [used, total])
    assert used.tie_mark == _tie_mark() == total.tie_mark
    concrete = concrete_of("c03.records.capture", "small", counted={"small": counted_document(SET_SHARDS)})
    reading = subject.read(used)
    verdict = grade.grade_concrete(used, concrete, reading)
    assert verdict.status == "right"
    marked = record.record_of(used, reading, verdict, used_tie=subject.used_tie(used))
    assert _tie_mark() in marked.marks
    total_concrete = concrete_of("c03.spans.total", "small", index={"small": SET_SPANS})
    total_reading = subject.read(total)
    total_verdict = grade.grade_concrete(total, total_concrete, total_reading)
    assert total_verdict.status == "right"
    plain = record.record_of(total, total_reading, total_verdict, used_tie=subject.used_tie(total))
    assert _tie_mark() not in plain.marks


# --- Grading the readings of the set rows ------------------------------------------------


def test_the_spans_and_the_totals_are_graded_against_the_index_sizes(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    per = question("c03.spans.per-ordinal", "small")
    total = question("c03.spans.total", "small")
    subject = world.subject(dolma, [per, total])
    for q in (per, total):
        concrete = concrete_of(q.listed_row, "small", index={"small": SET_SPANS})
        assert grade.grade_concrete(q, concrete, subject.read(q)).status == "right", q.id
    world.rows = [r for r in world.rows if not (r.src == world.chunk_files["shard-1"] and r.loc == 2)]
    gapped = world.subject(dolma, [per, total])
    got = grade.grade_concrete(per, concrete_of(per.listed_row, "small", index={"small": SET_SPANS}),
                               gapped.read(per))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]
    got = grade.grade_concrete(total, concrete_of(total.listed_row, "small", index={"small": SET_SPANS}),
                               gapped.read(total))
    assert got.status == "wrong" and list(got.evidence) == ["per_shard"]


def test_the_records_are_graded_against_the_planted_ids_and_another_id_is_wrong(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    q = question("c03.records.position-recovery", "small")
    concrete = concrete_of(q.listed_row, "small", counted={"small": counted_document(SET_SHARDS)})
    assert grade.grade_concrete(q, concrete, world.subject(dolma, [q]).read(q)).status == "right"
    row_type = _mod("baselines.companion").Row
    held = world.row("shard-1", 2)
    world.rows = [row_type(start=r.start, end=r.end, id="Z-999", src=r.src, loc=r.loc) if r is held else r
                  for r in world.rows]
    got = grade.grade_concrete(q, concrete, world.subject(dolma, [q]).read(q))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]


def test_the_split_pair_is_graded_on_its_chunk_file_and_line_alone(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    counted = counted_document(SET_SHARDS)
    counted["check3"] = {"pairs": {"list": [{
        "pair": "pair-2", "kind": "split", "placement": "split", "members": [
            {"id": "Z-005", "source": "Z", "shard": "shard-1", "line": 5, "ordinal": 4,
             "expected_label": "recovered"},
            {"id": "Z-007", "source": "Z", "shard": "shard-2", "line": 2, "ordinal": 1,
             "expected_label": "recovered"}]}]}}
    pairs = [{"pair": "pair-2", "members": [{"shard": "shard-1", "ordinal": 4},
                                            {"shard": "shard-2", "ordinal": 1}]}]
    q = question("c03.pairs.split", "small")
    concrete = concrete_of("c03.pairs.split", "small", counted={"small": counted})
    subject = world.subject(dolma, [q], split_pairs=pairs)
    assert grade.grade_concrete(q, concrete, subject.read(q)).status == "right"
    world.without("shard-2", 2)
    got = grade.grade_concrete(q, concrete, world.subject(dolma, [q], split_pairs=pairs).read(q))
    assert got.status == "wrong" and list(got.evidence) == ["record"]


# --- The three counts of the documents-mapped row ------------------------------------------

COUNT_KEYS = {"per_ordinal", "unmapped_documents", "documents_per_shard",
              "documents_with_one_tied_row_per_shard", "documents_with_no_tied_row_per_shard"}


def test_the_documents_mapped_reading_holds_the_three_counts_per_shard_and_the_ids(need_module, tmp_path):
    dolma = need_module(MODULE)
    row_type = _mod("baselines.companion").Row
    world = set_world(tmp_path)
    q = question("c06.generated.documents-mapped", "small")
    reading = world.subject(dolma, [q]).read(q)
    assert set(reading.raw) == COUNT_KEYS
    assert dict(reading.raw["documents_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(reading.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(reading.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 0, "shard-2": 0}
    assert reading.raw["unmapped_documents"] == 0
    assert {name: list(ids) for name, ids in reading.raw["per_ordinal"].items()} == SET_SHARDS
    # Shard 1: the document of ordinal 1 has no row, the document of ordinal 3 has two.
    world.without("shard-1", 2)
    world.rows.append(row_type(start=90, end=95, id="Z-004",
                               src=world.chunk_files["shard-1"], loc=4))
    changed = world.subject(dolma, [q]).read(q)
    assert dict(changed.raw["documents_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(changed.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 3, "shard-2": 4}
    assert dict(changed.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 1, "shard-2": 0}
    assert changed.raw["unmapped_documents"] == 1
    assert changed.raw["per_ordinal"]["shard-1"][1] is None


def test_a_document_of_an_identical_pair_has_its_own_row_and_is_counted_as_one_tied_row(need_module, tmp_path):
    """The companion rows hold no text: two lines with one text are two documents, each
    with the one row of its own line."""
    dolma = need_module(MODULE)
    shards = {"shard-1": [("Z-001", "the same passage"), ("Z-002", "the same passage"), "Z-003"]}
    world = World(tmp_path, shards, {"shard-1": SET_PATHS["shard-1"]})
    q = question("c06.generated.documents-mapped", "small")
    reading = world.subject(dolma, [q]).read(q)
    assert dict(reading.raw["documents_per_shard"]) == {"shard-1": 3}
    assert dict(reading.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 3}
    assert dict(reading.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 0}


def _drop_one_key(subject, key):
    """A known-bad reader: it leaves one named value out of every reading."""
    interfaces = _mod("baselines.interfaces")

    class Dropper:
        name = "dolma"

        def read(self, q):
            got = subject.read(q)
            if isinstance(got, interfaces.Reading) and isinstance(got.raw, Mapping) and key in got.raw:
                return interfaces.Reading(got.question_id, {k: v for k, v in got.raw.items() if k != key},
                                          got.route)
            return got

    return Dropper()


def test_the_documents_mapped_reading_is_graded_against_the_counts_and_a_missing_count_is_wrong(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    q = question("c06.generated.documents-mapped", "small")
    subject = world.subject(dolma, [q])
    concrete = concrete_of(q.listed_row, "small", counted={"small": counted_document(SET_SHARDS)})
    assert grade.grade_concrete(q, concrete, subject.read(q)).status == "right"
    # The same check on a reader that leaves one count out finds the fault, for each count.
    for key in ("documents_per_shard", "documents_with_one_tied_row_per_shard",
                "documents_with_no_tied_row_per_shard"):
        got = grade.grade_concrete(q, concrete, _drop_one_key(subject, key).read(q))
        assert got.status == "wrong" and list(got.evidence) == [key], key
    # One changed count (a document with no row) is wrong.
    world.without("shard-2", 4)
    got = grade.grade_concrete(q, concrete, world.subject(dolma, [q]).read(q))
    assert got.status == "wrong"
    assert {"documents_with_one_tied_row_per_shard", "documents_with_no_tied_row_per_shard",
            "unmapped_documents"} <= set(got.evidence)


FIXTURE_SHARDS = {
    "E/shard-1": [f"Z-{n:03d}" for n in range(1, 19)],
    "E/shard-2": [f"Z-{n:03d}" for n in range(19, 38)],
    "E/shard-3": ["T"] + [f"Z-{n:03d}" for n in range(38, 64)],
}


def test_the_documents_of_a_shard_of_the_fixture_are_read_as_documents_recovered_and_unknown(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = World(tmp_path, FIXTURE_SHARDS)
    listed = [f"c06.fixture.documents-mapped.{name}" for name in FIXTURE_SHARDS]
    questions = [question(name) for name in listed]
    subject = world.subject(dolma, questions)
    for q, name in zip(questions, FIXTURE_SHARDS):
        reading = subject.read(q)
        assert set(reading.raw) == {"documents", "recovered", "unknown"}, q.id
        assert reading.raw["documents"] == len(FIXTURE_SHARDS[name]) and reading.raw["unknown"] == 0
        assert grade.grade(q, listed_expect(q.listed_row), reading).status == "right", q.id
        assert subject.used_tie(q) is True
    world.without("E/shard-1", 5)
    first = questions[0]
    reading = world.subject(dolma, questions).read(first)
    assert (reading.raw["documents"], reading.raw["recovered"], reading.raw["unknown"]) == (18, 17, 1)
    got = grade.grade(first, listed_expect(first.listed_row), reading)
    assert got.status == "wrong" and list(got.evidence) == ["recovered", "unknown"]


# --- The shard rule ----------------------------------------------------------------------


def answer_of(world, dolma, listed, **more):
    q = question(listed)
    reading = world.subject(dolma, [q], **more).read(q)
    return q, reading


SHARD_RULE_CASES = [
    # (the ids of the lines of E/shard-1's chunk file, what the shard rule reads under withdraw A)
    (["B-001", "Z-001"], {"answer": "does not contain"}),    # every id resolves to a source file
    (["B-002", "A-002"], {"answer": "contains"}),            # a row's id resolves to A's file
    (["T", "Z-002"], ("T", "derived-from link")),            # a row's id resolves to a generation output
    (["Z-003", "N-404"], ("N-404", "file")),                 # a row's id resolves to no file
    (["T", "A-001", "Z-004"], {"answer": "contains"}),       # a member's row wins over a generation row
    ([], {"answer": "not covered"}),                         # no row has this src
]


@pytest.mark.parametrize("ids, wanted", SHARD_RULE_CASES, ids=[str(n) for n in range(len(SHARD_RULE_CASES))])
def test_the_shard_rule_gives_the_four_readings_of_the_table(need_module, tmp_path, ids, wanted):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"E/shard-1": ids})
    q, reading = answer_of(world, dolma, "c05.unmutated.E/shard-1")
    assert reading.route == q.route
    if isinstance(wanted, dict):
        assert dict(reading.raw) == wanted
    else:
        assert set(reading.raw) == {CANNOT}
        for word in wanted:
            assert word in reading.raw[CANNOT], word


def test_a_shard_reading_is_graded_against_the_answer_of_its_listed_row(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    clean = World(tmp_path / "clean", {"E/shard-1": ["B-001", "Z-001"], "E/shard-3": ["T", "Z-002"],
                                       "successor of E/shard-3": ["Z-003"]})
    q, reading = answer_of(clean, dolma, "c05.unmutated.E/shard-1")
    assert grade.grade(q, listed_expect(q.listed_row), reading).status == "right"
    s, successor = answer_of(clean, dolma, "c09.fixture.successor.E/shard-3")
    assert grade.grade(s, listed_expect(s.listed_row), successor).status == "right"
    # A shard that holds a row of A is read "contains", where the listed row expects "does
    # not contain": wrong, and the answer is the field that differs.
    holding = World(tmp_path / "holding", {"E/shard-1": ["A-002", "Z-001"],
                                           "successor of E/shard-3": ["A-001"]})
    for listed in ("c05.unmutated.E/shard-1", "c09.fixture.successor.E/shard-3"):
        q, reading = answer_of(holding, dolma, listed)
        got = grade.grade(q, listed_expect(q.listed_row), reading)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
    # A shard whose row names a generation output is cannot answer, whatever the row expects.
    q, reading = answer_of(clean, dolma, "c05.unmutated.E/shard-3")
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "cannot answer" and "derived-from link" in got.evidence


def _graded(world, dolma, grade, listed):
    q, reading = answer_of(world, dolma, listed)
    return grade.grade(q, listed_expect(q.listed_row), reading)


def test_the_shard_rows_of_the_faults_and_of_the_requests_are_graded_right_and_wrong(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    clean = {"E/shard-1": ["B-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"], "E/shard-3": ["B-003", "Z-003"]}
    right = World(tmp_path / "right", clean)
    # (listed row, the shard the changed world changes, its ids, the listed row graded there)
    cases = [
        ("c05.f.map-deleted.E/shard-1", "E/shard-2", ["A-002", "Z-002"], "c05.f.map-deleted.E/shard-2"),
        ("c05.f.digest.E/shard-1", "E/shard-1", ["A-002", "Z-001"], "c05.f.digest.E/shard-1"),
        ("c05.withdraw-C.unmutated.E/shard-3", "E/shard-3", ["C-001", "Z-003"],
         "c05.withdraw-C.unmutated.E/shard-3"),
        ("c06.fixture.downstream.E/shard-3", None, None, None),
        ("c09.fixture.E/shard-1.action", "E/shard-1", ["A-002", "Z-001"], "c09.fixture.E/shard-1.action"),
    ]
    for number, (listed, changed, ids, wrong_listed) in enumerate(cases):
        if wrong_listed is None:
            # The shard of the table's downstream row holds nothing a withdraw reaches here, so
            # the toolkit reads "does not contain" where the row expects "may contain": wrong.
            # No right reading exists for this row: the toolkit has no reading "may contain".
            got = _graded(right, dolma, grade, listed)
            assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
            continue
        assert _graded(right, dolma, grade, listed).status == "right", listed
        other = World(tmp_path / f"wrong-{number}", {**clean, changed: ids})
        got = _graded(other, dolma, grade, wrong_listed)
        field = "action" if listed.endswith(".action") else "answer"
        assert got.status == "wrong" and list(got.evidence) == [field], wrong_listed
    # A record-level request: right where the shard holds the named record, wrong where it holds
    # another record of the same file.
    named = World(tmp_path / "named", {**clean, "E/shard-3": ["A-001", "Z-003"]})
    assert _graded(named, dolma, grade, "c05.record-X@A.unmutated.E/shard-3").status == "right"
    file_only = World(tmp_path / "file-only", {**clean, "E/shard-3": ["A-002", "Z-003"]})
    got = _graded(file_only, dolma, grade, "c05.record-X@A.unmutated.E/shard-3")
    assert got.status == "wrong" and list(got.evidence) == ["answer"]


def test_the_request_decides_which_file_is_a_member_and_a_record_level_request_names_one_id(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"E/shard-3": ["A-002", "C-001", "Z-001"], "E/shard-1": ["A-001", "Z-002"]})
    # E/shard-3 holds a row of A that is not the record X@A, and a row of C.
    a, a_reading = answer_of(world, dolma, "c05.unmutated.E/shard-3")
    assert dict(a_reading.raw) == {"answer": "contains"}
    c, c_reading = answer_of(world, dolma, "c05.withdraw-C.unmutated.E/shard-3")
    assert dict(c_reading.raw) == {"answer": "contains"}
    record_level = "c05.record-X@A.unmutated.E/shard-3"
    q, reading = answer_of(world, dolma, record_level)
    assert dict(reading.raw) == {"answer": "does not contain"}
    clean = World(tmp_path / "other", {"E/shard-3": ["B-001", "Z-001"]})
    q, reading = answer_of(clean, dolma, "c05.withdraw-C.unmutated.E/shard-3")
    assert dict(reading.raw) == {"answer": "does not contain"}
    named = World(tmp_path / "named", {"E/shard-3": ["A-001", "Z-001"]})
    q, reading = answer_of(named, dolma, record_level)
    assert dict(reading.raw) == {"answer": "contains"}


def test_the_round_robin_chunk_files_share_base_names_and_are_told_apart_by_the_whole_src(need_module, tmp_path):
    dolma = need_module(MODULE)
    paths = {"E/shard-1": "pipeline/partitioned/tokenizer-input/shard-1.jsonl",
             "E-rr/shard-1": "pipeline/round-robin/tokenizer-input/shard-1.jsonl"}
    world = World(tmp_path, {"E/shard-1": ["B-001"], "E-rr/shard-1": ["A-001"]}, paths)
    q = question("c05.unmutated.E/shard-1")
    rr = question("c05.unmutated.E-rr/shard-1")
    world.write_tie(True)
    run = world.run(dolma)
    subject = dolma.DolmaSubject({target_of(q): run, target_of(rr): run}, tie_record=world.tie_path,
                                 requests=REQUESTS)
    assert dict(subject.read(q).raw) == {"answer": "does not contain"}
    assert dict(subject.read(rr).raw) == {"answer": "contains"}


@pytest.mark.parametrize("listed, group", [
    ("c05.f.map-deleted.src/A", "c05.unmutated.files"),
    ("c05.f.map-deleted.gen/P", "c05.unmutated.files"),
    ("c05.f.map-deleted.R1", "c05.unmutated.files"),
    ("c05.withdraw-C.unmutated.R1", "c05.unmutated.files"),
    ("c06.fixture.downstream.gen/T", "c05.unmutated.files"),
    ("c06.fixture.downstream.R1", "c05.unmutated.files"),
    ("c09.fixture.R1.action", "c05.unmutated.files"),
    ("c09.fixture.R2.action", "c05.unmutated.files"),
    ("c09.fixture.successor.R1", "c05.unmutated.files"),
    ("c05.f.map-deleted.E/mix-1", "c05.unmutated.mixtures"),
    ("c05.record-X@A.unmutated.K/mix-1", "c05.unmutated.mixtures"),
    ("c05.f.map-deleted.E@6", "c05.unmutated.checkpoints"),
    ("c05.record-X@A.unmutated.K@40", "c05.unmutated.checkpoints"),
    ("c05.f.digest.listing", "c05.listing.map-deleted"),
])
def test_an_artifact_that_is_no_shard_is_cannot_answer_with_the_missing_record_of_its_kind(
        need_module, tmp_path, listed, group):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"E/shard-1": ["B-001"]})
    q, reading = answer_of(world, dolma, listed)
    assert set(reading.raw) == {CANNOT}, listed
    assert _missing_of(group) in reading.raw[CANNOT]
    assert reading.route == q.route


@pytest.mark.parametrize("listed", [
    "c05.f.map-deleted.src/A", "c05.f.map-deleted.E/mix-1", "c05.f.map-deleted.E@6", "c05.f.digest.listing"])
def test_a_cannot_answer_of_a_kind_the_toolkit_holds_no_record_for_needs_a_successful_tokens_run(
        need_module, tmp_path, listed):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    world = World(tmp_path / "one", {"E/shard-1": ["B-001"], "E/shard-2": ["B-002"]})
    q = question(listed)

    def subject_of(entry):
        return dolma.DolmaSubject({target_of(q): entry}, tie_record=world.tie_path, requests=REQUESTS)

    # No tokenizer run of the store was made: the question is an error, not an absence and not
    # a cannot answer.
    with pytest.raises(ValueError):
        dolma.DolmaSubject({}, tie_record=world.tie_path, requests=REQUESTS).read(q)
    # The run failed: its attempt is the reading.
    attempt = interfaces.Attempt(id="attempt-9", command=("step",), exit_code=1, output="no row")
    assert subject_of(attempt).read(q) is attempt
    # A run that holds no row, or none for a chunk file that has lines, did not succeed: the
    # row count check of a run is the check of its reading.
    shard_1 = [r for r in world.rows if r.src == world.chunk_files["E/shard-1"]]
    for rows in ((), tuple(shard_1)):
        run = dolma.DolmaRun(build=world.build, rows=rows, chunk_files=dict(world.chunk_files))
        with pytest.raises(ValueError):
            subject_of(run).read(q)
    # The run that wrote a row for every chunk file that has lines reads cannot answer.
    reading = subject_of(world.run(dolma)).read(q)
    assert set(reading.raw) == {CANNOT} and reading.route == q.route
    # A chunk file with no lines needs no row.
    (world.build / world.chunk_files["E/shard-2"]).write_text("", encoding="utf-8")
    run = dolma.DolmaRun(build=world.build, rows=tuple(shard_1), chunk_files=dict(world.chunk_files))
    assert set(subject_of(run).read(q).raw) == {CANNOT}


# --- The faults on the companion rows ---------------------------------------------------


def test_the_map_deleted_fault_removes_the_rows_of_one_chunk_file_and_changes_no_row_of_the_run(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"E/shard-1": ["B-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"]})
    before = list(world.rows)
    faulted = question("c05.f.map-deleted.E/shard-1")
    other = question("c05.f.map-deleted.E/shard-2")
    subject = world.subject(dolma, [faulted, other])
    assert dict(subject.read(faulted).raw) == {"answer": "not covered"}
    assert dict(subject.read(other).raw) == {"answer": "does not contain"}
    assert world.rows == before
    again = question("c05.unmutated.E/shard-1")
    unfaulted = world.subject(dolma, [again])
    assert dict(unfaulted.read(again).raw) == {"answer": "does not contain"}


def test_a_fault_put_to_the_rows_is_noted_with_the_rows_before_and_after_it(need_module, tmp_path):
    """The record must show that a fault was applied (17 Nov 2026):
    the Dolma subject hands every fault it puts to a copy of the rows to ``note_fault``, naming the state, the
    chunk file, the number of rows and the sha256 of the rows before and after."""
    dolma = need_module(MODULE)
    world = World(tmp_path, {"E/shard-1": ["B-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"],
                             "E/shard-3": ["B-003", "Z-003"]})
    faulted = question("c05.f.map-deleted.E/shard-1")
    digest = question("c05.f.digest.E/shard-3")
    plain = question("c05.unmutated.E/shard-1")
    subject = world.subject(dolma, [faulted, digest, plain])
    notes = []
    subject.note_fault = notes.append
    subject.read(plain)
    assert notes == [], "a question with no fault was given a fault note"
    subject.read(faulted)
    assert len(notes) == 1
    note = notes[0]
    assert note["fault"] == "f.map-deleted" and note["chunk_file"] == world.chunk_files["E/shard-1"]
    assert note["rows_before"] == len(world.rows) and note["rows_after"] < note["rows_before"]
    assert note["rows_sha256_before"] != note["rows_sha256_after"]
    notes.clear()
    subject.read(digest)
    (note,) = notes
    assert note["fault"] == "f.digest" and note["rows_after"] == note["rows_before"]
    assert note["rows_sha256_before"] != note["rows_sha256_after"], "the digest fault changed no row"
    # No hook, no note: the subject reads as before.
    other = world.subject(dolma, [faulted])
    assert other.note_fault is None
    assert dict(other.read(faulted).raw) == {"answer": "not covered"}


def test_the_digest_fault_changes_the_end_of_one_row_and_the_answer_is_still_read_from_the_ids(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = World(tmp_path, {"E/shard-3": ["B-001", "Z-001"]})
    q = question("c05.f.digest.E/shard-3")
    reading = world.subject(dolma, [q]).read(q)
    assert dict(reading.raw) == {"answer": "does not contain"}
    assert "stopped" not in reading.raw
    # The row expects a stop: the toolkit has no command that reports the changed record.
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "wrong" and list(got.evidence) == ["answer"]


def test_the_count_fault_deletes_the_last_row_and_the_reading_is_an_answer_never_a_stop(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    ids = [f"Z-{n:03d}" for n in range(1, 500)] + ["A-001"]
    world = World(tmp_path, {"c04/shard-500": ids})
    q = question("c04.count-499")
    assert target_of(q) == ("store", "scratch-500")
    reading = world.subject(dolma, [q]).read(q)
    assert dict(reading.raw) == {"answer": "does not contain"}, "the row of the 500th line is gone"
    assert "stopped" not in reading.raw
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "wrong" and list(got.evidence) == ["answer"]
    held = [r for r in world.rows if r.src == world.chunk_files["c04/shard-500"]]
    assert len(held) == 500, "the run's own rows are never changed"


# --- The two writer states of Check 4 ----------------------------------------------------

SCRATCH_IDS = [f"Z-{n:03d}" for n in range(1, 501)]
SCRATCH_TOKENS = 1999  # 500 documents of 3, 4, 5, 3, 4, 5, ... tokens


def scratch_world(tmp_path):
    return World(tmp_path, {"c04/shard-500": SCRATCH_IDS})


def scratch_rows(world, count):
    held = sorted((r for r in world.rows if r.src == world.chunk_files["c04/shard-500"]),
                  key=lambda r: r.loc)
    return held[:count]


def test_the_rows_after_a_stopped_write_are_counted_by_src(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = scratch_world(tmp_path)
    q = question("c04.kill-after-100")
    assert target_of(q) == ("store", "scratch-500")
    for count, status, evidence in ((0, "right", None), (500, "right", None), (100, "wrong", ["never"])):
        subject = world.subject(dolma, [q], stopped_rows=tuple(scratch_rows(world, count)))
        reading = subject.read(q)
        assert dict(reading.raw) == {"records": {"rows_after_stop": count,
                                              "rows_showed": "while the writer ran"}}
        got = grade.grade(q, listed_expect(q.listed_row), reading)
        assert (got.status, got.evidence) == (status, evidence), count


def traceback_of(last_line):
    """A printed traceback of an uncaught exception, ending in ``last_line``."""
    return ("Traceback (most recent call last):\n"
            f"  File \"{PRIVATE_PATH}/venv-dolma/lib/dolma/cli/main.py\", line 98, in main\n"
            "    return cli.run_from_args(args=args, config=config)\n"
            f"  File \"{PRIVATE_PATH}/venv-dolma/lib/dolma/cli/tokenizer.py\", line 200, in run\n"
            "    raise SomeError\n"
            f"{last_line}\n")


def error_output(name, text):
    """The traceback of an uncaught error ``name`` with ``text``; Python prints no colon for an
    error with no text."""
    return traceback_of(f"{name}: {text}" if text else name)


# The classes of the toolkit's error module, in the order of the file, each with the name a
# traceback prints (the module's dotted path in front) and the bare name.
DOLMA_ERROR_CLASSES = ("DolmaError", "DolmaFatalError", "DolmaShardError", "DolmaRetryableFailure",
                       "DolmaRustPipelineError", "DolmaConfigError")
ERROR_MODULE = "dolma.core.errors"
ATTEMPT_ID = "attempt-7"
RETRY_COMMAND = (PRIVATE_PATH + "/venv-dolma/bin/python", "-m", "dolma.cli", "tokens", "--documents", "shard.jsonl")


def second_run(dolma, tmp_path, exit_code, output, **more):
    """Read c04.retry of a second run that ended with ``exit_code`` and wrote ``output``, with
    no rows left, given the id and the command of the attempt it was."""
    assert {"retry_attempt_id", "retry_command"} <= set(dolma.DolmaRun.__dataclass_fields__)
    world = scratch_world(tmp_path)
    q = question("c04.retry")
    settings = dict(retry_rows=(), retry_exit_code=exit_code, retry_output=output,
                    token_file_tokens=SCRATCH_TOKENS, retry_attempt_id=ATTEMPT_ID,
                    retry_command=RETRY_COMMAND)
    settings.update(more)
    return world.subject(dolma, [q], **settings).read(q)


def records_of(got, *keys):
    """The named values of a reading's records."""
    return {key: got.raw["records"][key] for key in keys}


def assert_failed_attempt(got, exit_code):
    interfaces = _mod("baselines.interfaces")
    assert isinstance(got, interfaces.Attempt), got
    assert not isinstance(got, interfaces.Reading)
    assert got.id == ATTEMPT_ID and got.exit_code == exit_code


def test_an_error_of_the_toolkit_in_the_second_run_is_a_refusal_and_its_text_is_the_mismatch(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    q = question("c04.retry")
    text = "declared count mismatch: the map holds 100 rows"
    got = second_run(dolma, tmp_path, 1, error_output(f"{ERROR_MODULE}.DolmaConfigError", text))
    assert records_of(got, "rows_after_retry", "refused", "mismatch") == {
        "rows_after_retry": 0, "refused": True, "mismatch": text}
    assert grade.grade(q, listed_expect("c04.retry"), got).status == "right"
    # The text is the error's own, with a local path in it cut to its last name.
    got = second_run(dolma, tmp_path / "path", 1, error_output(
        f"{ERROR_MODULE}.DolmaConfigError", f"No documents found for paths {PRIVATE_PATH}/build/shard.jsonl."))
    assert got.raw["records"]["mismatch"] == "No documents found for paths shard.jsonl."


def test_every_class_of_the_toolkit_s_error_module_is_a_refusal_in_either_printed_form(need_module, tmp_path):
    dolma = need_module(MODULE)
    for number, name in enumerate(DOLMA_ERROR_CLASSES):
        for form in (f"{ERROR_MODULE}.{name}", name):
            got = second_run(dolma, tmp_path / f"{number}-{form}", 1, error_output(form, "the text of it"))
            assert records_of(got, "rows_after_retry", "refused", "mismatch") == {
                "rows_after_retry": 0, "refused": True, "mismatch": "the text of it"}, form


def test_an_error_of_the_toolkit_with_no_text_is_a_refusal_with_no_mismatch_named(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    q = question("c04.retry")
    for number, last in enumerate((f"{ERROR_MODULE}.DolmaConfigError", f"{ERROR_MODULE}.DolmaConfigError: ",
                                   "DolmaFatalError")):
        got = second_run(dolma, tmp_path / str(number), 1, traceback_of(last))
        assert records_of(got, "rows_after_retry", "refused", "mismatch") == {
            "rows_after_retry": 0, "refused": True, "mismatch": None}, last
    verdict = grade.grade(q, listed_expect("c04.retry"), got)
    assert verdict.status == "wrong" and list(verdict.evidence) == ["records"]


def test_a_crash_of_no_class_of_the_toolkit_in_the_second_run_is_a_failed_attempt(need_module, tmp_path):
    dolma = need_module(MODULE)
    for number, last in enumerate(("KeyError: 'id'", "RuntimeError: Multiprocessing start method must be spawn",
                                   "ValueError: not enough values to unpack",
                                   "dolma.taggers.url.UrlNotParsedError: no url", "MemoryError")):
        got = second_run(dolma, tmp_path / str(number), 1, traceback_of(last))
        assert_failed_attempt(got, 1)
    # The last line of the traceback is the error that ended the run: an error of the toolkit
    # that another error followed is a crash, and so is a text that only names an error class.
    chained = (error_output(f"{ERROR_MODULE}.DolmaConfigError", "Invalid error while parsing key")
               + "\nThe above exception was the direct cause of the following exception:\n\n"
               + traceback_of("KeyError: 'id'"))
    assert_failed_attempt(second_run(dolma, tmp_path / "chained", 1, chained), 1)
    named = traceback_of(f"KeyError: '{ERROR_MODULE}.DolmaConfigError: the map is short'")
    assert_failed_attempt(second_run(dolma, tmp_path / "named", 1, named), 1)


def test_a_name_that_only_looks_like_a_class_of_the_toolkit_is_a_crash(need_module, tmp_path):
    dolma = need_module(MODULE)
    for number, last in enumerate(("MyDolmaConfigError: text", "other.module.DolmaConfigError: text",
                                   "DolmaConfigErrors: text", "DolmaConfigError text",
                                   "dolma.core.errors.DolmaWarning: text")):
        assert_failed_attempt(second_run(dolma, tmp_path / str(number), 1, traceback_of(last)), 1)


def test_a_signal_exit_of_the_second_run_is_a_failed_attempt_whatever_it_wrote(need_module, tmp_path):
    dolma = need_module(MODULE)
    printed = error_output(f"{ERROR_MODULE}.DolmaConfigError", "declared count mismatch: the map holds 100 rows")
    # A runner gives a signal as a negative code or as 128 plus the number of the signal.
    for exit_code in (-9, -11, -15, 129, 130, 137, 139, 143, 192):
        assert_failed_attempt(second_run(dolma, tmp_path / f"printed{exit_code}", exit_code, printed), exit_code)
        assert_failed_attempt(second_run(dolma, tmp_path / f"blank{exit_code}", exit_code, ""), exit_code)


def test_a_non_zero_exit_of_the_second_run_with_no_output_is_a_failed_attempt(need_module, tmp_path):
    dolma = need_module(MODULE)
    for exit_code in (1, 2, 3):
        for number, output in enumerate(("", "\n", "  \n \n")):
            got = second_run(dolma, tmp_path / f"{exit_code}-{number}", exit_code, output)
            assert_failed_attempt(got, exit_code)
    assert_failed_attempt(second_run(dolma, tmp_path / "none", 1, None), 1)


def test_a_non_zero_exit_of_the_second_run_the_reading_cannot_classify_is_a_failed_attempt(need_module, tmp_path):
    dolma = need_module(MODULE)
    plain = "declared count mismatch: the map holds 100 rows"
    cases = (
        (3, plain),
        (2, "usage: dolma {global options} [command] {command options}\n"
            "dolma: error: the following arguments are required: command\n"),
    )
    for number, (exit_code, output) in enumerate(cases):
        assert_failed_attempt(second_run(dolma, tmp_path / str(number), exit_code, output), exit_code)


def test_the_failed_attempt_of_the_second_run_holds_no_local_path_and_needs_an_id(need_module, tmp_path):
    dolma = need_module(MODULE)
    got = second_run(dolma, tmp_path, 1, traceback_of("KeyError: 'id'"))
    assert got.command == ("python", "-m", "dolma.cli", "tokens", "--documents", "shard.jsonl")
    assert PRIVATE_PATH not in got.output and "main.py" in got.output and "KeyError: 'id'" in got.output
    # The question of the stopped write on the same run still reads: the failed second run does
    # not change it.
    world = scratch_world(tmp_path / "stopped")
    stopped = question("c04.kill-after-100")
    subject = world.subject(dolma, [stopped], stopped_rows=tuple(scratch_rows(world, 100)),
                            retry_rows=(), retry_exit_code=-9, retry_output="",
                            retry_attempt_id=ATTEMPT_ID, retry_command=RETRY_COMMAND)
    assert dict(subject.read(stopped).raw) == {"records": {"rows_after_stop": 100,
                                                    "rows_showed": "while the writer ran"}}
    # A failed second run with no id for its attempt is not guessed: the run is incomplete.
    world = scratch_world(tmp_path / "noid")
    q = question("c04.retry")
    subject = world.subject(dolma, [q], retry_rows=(), retry_exit_code=1, retry_output="",
                            token_file_tokens=SCRATCH_TOKENS)
    with pytest.raises(ValueError):
        subject.read(q)


def test_a_second_run_that_ends_with_exit_0_is_read_as_before_whatever_it_wrote(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = scratch_world(tmp_path)
    q = question("c04.retry")
    whole = tuple(scratch_rows(world, 500))
    interfaces = _mod("baselines.interfaces")
    for number, output in enumerate(("", "done\n", error_output(f"{ERROR_MODULE}.DolmaConfigError", "text"))):
        subject = world.subject(dolma, [q], retry_rows=whole, retry_exit_code=0, retry_output=output,
                                token_file_tokens=SCRATCH_TOKENS)
        got = subject.read(q)
        assert isinstance(got, interfaces.Reading), number
        assert dict(got.raw["records"]) == {"rows_after_retry": 500, "matches_shard": True,
                                            "refused": False, "mismatch": None}, number


def test_the_base_class_of_the_toolkit_s_errors_with_no_text_is_a_refusal_naming_no_mismatch(need_module, tmp_path):
    dolma = need_module(MODULE)
    for number, last in enumerate(("DolmaError", "DolmaError:", f"{ERROR_MODULE}.DolmaError",
                                   f"{ERROR_MODULE}.DolmaError: ")):
        got = second_run(dolma, tmp_path / str(number), 1, traceback_of(last))
        assert records_of(got, "rows_after_retry", "refused", "mismatch") == {
            "rows_after_retry": 0, "refused": True, "mismatch": None}, last


# --- A stop that did not land -----------------------------------------------------------------

# A runner gives the end of a process that a signal stopped as a negative code or as 128 plus the
# signal's number; any other exit means the writer ended by itself or by another signal.
# (signal sent, exit of the writer) of a stop that landed:
LANDED_STOPS = ((9, -9), (9, 137), (15, -15), (15, 143), (2, -2), (2, 130), (64, -64), (64, 192))
# The same signal 9 sent to a writer that ended by itself (0, 1, 2, 127, 128, 255), by an exit
# that is no signal's (193), or by another signal (11, 15).
NOT_LANDED_EXITS = (0, 1, 2, 127, 128, 193, 255, -11, 139, -15, 143, -1, 129)


def stopped_world(tmp_path):
    return scratch_world(tmp_path)


def kill_and_retry(dolma, world, **more):
    """The readings of both questions of the stopped write on one run, with a second run that
    wrote the whole map; ``more`` replaces what the run holds."""
    stop_q, retry_q = question("c04.kill-after-100"), question("c04.retry")
    settings = {"stopped_rows": tuple(scratch_rows(world, 500)), "retry_rows": tuple(scratch_rows(world, 500)),
                "retry_exit_code": 0, "token_file_tokens": SCRATCH_TOKENS}
    settings.update(more)
    subject = world.subject(dolma, [stop_q, retry_q], **settings)
    return subject.read(stop_q), subject.read(retry_q)


def test_a_stop_that_landed_is_read_as_the_exit_of_the_signal_that_was_sent(need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    for sent, exit_code in LANDED_STOPS:
        world = stopped_world(tmp_path / f"{sent}-{exit_code}")
        stopped, retried = kill_and_retry(dolma, world, stopped_exit_code=exit_code, stopped_signal=sent)
        assert isinstance(stopped, interfaces.Reading) and isinstance(retried, interfaces.Reading)
        assert stopped.raw["records"].get("rows_showed") == "while the writer ran", (sent, exit_code)


def test_a_stop_that_did_not_land_is_the_failed_attempt_of_the_stopped_writer_with_its_exit(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    printed = f"Traceback: File \"{PRIVATE_PATH}/venv-dolma/lib/x.py\", line 3\ndone"
    for exit_code in NOT_LANDED_EXITS:
        world = stopped_world(tmp_path / str(exit_code))
        # The writer ended by itself with the whole map written, or by a signal other than the
        # one sent: the rows after it would read right, and so would the second run; neither is
        # a reading of a stop. (Signal 9 was sent.)
        for got in kill_and_retry(dolma, world, stopped_exit_code=exit_code, stopped_output=printed):
            assert isinstance(got, interfaces.Attempt), exit_code
            assert not isinstance(got, interfaces.Reading)
            assert got.id == STOP_ID and got.exit_code == exit_code
            assert got.command == ("python", "-m", "dolma.cli", "tokens", "--documents", "shard.jsonl")
            assert PRIVATE_PATH not in got.output and "done" in got.output and "did not land" in got.output
    # A second run that crashed too is not read before the stop: the stop's attempt is the one.
    world = stopped_world(tmp_path / "both")
    for got in kill_and_retry(dolma, world, stopped_exit_code=0, retry_rows=(), retry_exit_code=1,
                              retry_output=traceback_of("KeyError: 'id'"),
                              retry_attempt_id=ATTEMPT_ID, retry_command=RETRY_COMMAND):
        assert isinstance(got, interfaces.Attempt) and got.id == STOP_ID


def test_a_stop_with_no_exit_or_no_id_for_its_attempt_is_an_incomplete_run_not_a_guess(need_module, tmp_path):
    dolma = need_module(MODULE)
    for number, bad in enumerate((None, True, "137")):
        world = stopped_world(tmp_path / f"exit-{number}")
        stop_q, retry_q = question("c04.kill-after-100"), question("c04.retry")
        subject = world.subject(dolma, [stop_q, retry_q], stopped_rows=tuple(scratch_rows(world, 100)),
                                retry_rows=tuple(scratch_rows(world, 500)), retry_exit_code=0,
                                token_file_tokens=SCRATCH_TOKENS, stopped_exit_code=bad)
        for q in (stop_q, retry_q):
            with pytest.raises(ValueError):
                subject.read(q)
    # The signal that was sent is part of the record: none, or one that is no signal's number,
    # leaves the exit with nothing to be the exit of.
    for number, bad in enumerate((None, True, "9", 0, -9, 65)):
        world = stopped_world(tmp_path / f"signal-{number}")
        stop_q, retry_q = question("c04.kill-after-100"), question("c04.retry")
        subject = world.subject(dolma, [stop_q, retry_q], stopped_rows=tuple(scratch_rows(world, 100)),
                                retry_rows=tuple(scratch_rows(world, 500)), retry_exit_code=0,
                                token_file_tokens=SCRATCH_TOKENS, stopped_signal=bad)
        for q in (stop_q, retry_q):
            with pytest.raises(ValueError):
                subject.read(q)
    world = stopped_world(tmp_path / "no-id")
    stop_q, retry_q = question("c04.kill-after-100"), question("c04.retry")
    subject = world.subject(dolma, [stop_q, retry_q], stopped_rows=tuple(scratch_rows(world, 100)),
                            retry_rows=tuple(scratch_rows(world, 500)), retry_exit_code=0,
                            token_file_tokens=SCRATCH_TOKENS, stopped_exit_code=0, stopped_attempt_id="")
    for q in (stop_q, retry_q):
        with pytest.raises(ValueError):
            subject.read(q)


def test_the_rows_after_a_stop_say_whether_they_showed_while_the_writer_ran_or_only_at_its_end(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = stopped_world(tmp_path)
    q = question("c04.kill-after-100")
    for showed, words in ((True, "while the writer ran"), (False, "only at its end")):
        for count, status in ((100, "wrong"), (500, "right")):
            subject = world.subject(dolma, [q], stopped_rows=tuple(scratch_rows(world, count)),
                                    rows_showed_while_running=showed)
            reading = subject.read(q)
            assert dict(reading.raw) == {"records": {"rows_after_stop": count, "rows_showed": words}}
            # The grade reads the count alone, as before.
            assert grade.grade(q, listed_expect(q.listed_row), reading).status == status
    # A run that does not say is not read: the answer would hide which of the two it was.
    for bad in (None, 1, "while the writer ran"):
        subject = world.subject(dolma, [q], stopped_rows=tuple(scratch_rows(world, 100)),
                                rows_showed_while_running=bad)
        with pytest.raises(ValueError):
            subject.read(q)


def test_the_second_run_is_read_as_rows_ranges_refusal_and_the_mismatch_it_names(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    row_type = _mod("baselines.companion").Row
    world = scratch_world(tmp_path)
    q = question("c04.retry")
    whole = tuple(scratch_rows(world, 500))
    expect = listed_expect("c04.retry")

    def read(rows, exit_code, output="", tokens=SCRATCH_TOKENS):
        subject = world.subject(dolma, [q], retry_rows=rows, retry_exit_code=exit_code,
                                retry_output=output, token_file_tokens=tokens)
        return subject.read(q)

    # The whole map, its ranges indexing the documents of the token file: not refused.
    reading = read(whole, 0)
    assert set(reading.raw) == {"records"}
    assert set(reading.raw["records"]) == {"rows_after_retry", "matches_shard", "refused", "mismatch"}
    assert dict(reading.raw["records"]) == {"rows_after_retry": 500, "matches_shard": True,
                                            "refused": False, "mismatch": None}
    assert grade.grade(q, expect, reading).status == "right"
    # A range that leaves a gap does not index the documents of the token file.
    gapped = tuple(row_type(start=r.start + 1, end=r.end, id=r.id, src=r.src, loc=r.loc)
                   if r.loc == 7 else r for r in whole)
    assert read(gapped, 0).raw["records"]["matches_shard"] is False
    # A token file whose length the run does not give: the subject cannot say.
    assert read(whole, 0, tokens=None).raw["records"]["matches_shard"] is None
    # Refused with no row left, the mismatch named by the text of the error the tool raised.
    refused = read((), 1, error_output("dolma.core.errors.DolmaConfigError",
                                       "declared count mismatch: the map holds 100 rows"))
    assert refused.raw["records"]["rows_after_retry"] == 0
    assert refused.raw["records"]["refused"] is True
    assert refused.raw["records"]["mismatch"] == "declared count mismatch: the map holds 100 rows"
    assert grade.grade(q, expect, refused).status == "right"
    # Refused with the error's text empty is a gap in the records.
    unnamed = read((), 1, error_output("dolma.core.errors.DolmaConfigError", ""))
    assert unnamed.raw["records"]["refused"] is True
    assert unnamed.raw["records"]["mismatch"] is None
    got = grade.grade(q, expect, unnamed)
    assert got.status == "wrong" and list(got.evidence) == ["records"]


# --- The actions of Check 9 --------------------------------------------------------------


def test_the_action_of_a_shard_is_rewrite_where_the_shard_rule_says_contains_and_keep_where_it_does_not(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = World(tmp_path, {"E/shard-1": ["B-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"],
                             "E/shard-3": ["A-001", "Z-003"]})
    listed = ["c09.fixture.E/shard-1.action", "c09.fixture.E/shard-2.action", "c09.fixture.E/shard-3.action"]
    questions = [question(name) for name in listed]
    subject = world.subject(dolma, questions)
    for q, action in zip(questions, ("keep", "keep", "rewrite")):
        reading = subject.read(q)
        assert dict(reading.raw) == {"action": action}, q.id
    for q in questions[:2]:
        assert grade.grade(q, listed_expect(q.listed_row), subject.read(q)).status == "right"


def test_the_counts_of_the_fixture_shards_are_lists_read_keep_and_rewrite_and_one_cannot_answer_makes_the_row_so(
        need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = World(tmp_path, {"E/shard-1": ["B-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"],
                             "E/shard-3": ["A-001", "Z-003"]})
    q = question("c09.fixture.shards.counts")
    reading = world.subject(dolma, [q]).read(q)
    assert set(reading.raw) == {"counts", "keep", "rewrite"}
    assert dict(reading.raw["counts"]) == {"keep": 2, "rewrite": 1}
    assert list(reading.raw["keep"]) == ["E/shard-1", "E/shard-2"]
    assert list(reading.raw["rewrite"]) == ["E/shard-3"]
    assert grade.grade(q, listed_expect(q.listed_row), reading).status == "right"
    # A reading with one shard on the wrong side is not the row's lists and numbers.
    flipped = World(tmp_path / "flipped", {"E/shard-1": ["A-001", "Z-001"], "E/shard-2": ["B-002", "Z-002"],
                                           "E/shard-3": ["A-001", "Z-003"]})
    other = flipped.subject(dolma, [q]).read(q)
    assert dict(other.raw["counts"]) == {"keep": 1, "rewrite": 2}
    assert grade.grade(q, listed_expect(q.listed_row), other).status == "wrong"
    # One shard whose row names a generation output makes the row cannot answer.
    generated = World(tmp_path / "gen", {"E/shard-1": ["B-001"], "E/shard-2": ["B-002"],
                                         "E/shard-3": ["T", "Z-003"]})
    cannot = generated.subject(dolma, [q]).read(q)
    assert set(cannot.raw) == {CANNOT} and "E/shard-3" in cannot.raw[CANNOT]


def test_the_round_robin_chunks_are_read_rewrite_or_keep_one_by_one(need_module, tmp_path):
    dolma = need_module(MODULE)
    shards = {f"shard-{n:02d}": ["A-001", f"Z-{n:03d}"] for n in range(1, 21)}
    world = World(tmp_path, shards)
    q = question("c09.10k.round-robin.all-rewrite")
    assert target_of(q) == ("set", "10k")
    reading = world.subject(dolma, [q]).read(q)
    assert dict(reading.raw) == {"rewrite": 20, "keep": 0}
    shards["shard-07"] = ["B-001", "Z-107"]
    other = World(tmp_path / "other", shards).subject(dolma, [q]).read(q)
    assert dict(other.raw) == {"rewrite": 19, "keep": 1}


def test_the_kept_shards_of_a_partitioned_set_are_the_shards_read_keep(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"shard-1": ["B-001", "Z-001"], "shard-2": ["A-001", "Z-002"],
                             "shard-3": ["Z-003"]})
    q = question("c09.10k.partitioned.kept-shards")
    reading = world.subject(dolma, [q]).read(q)
    assert set(reading.raw) == {"keep"}
    assert list(reading.raw["keep"]) == ["shard-1", "shard-3"]


def test_a_successor_is_read_by_the_shard_rule_and_the_successor_of_a_release_is_cannot_answer(need_module, tmp_path):
    dolma = need_module(MODULE)
    world = World(tmp_path, {"successor of E/shard-3": ["B-001", "Z-001"]})
    q = question("c09.fixture.successor.E/shard-3")
    assert dict(world.subject(dolma, [q]).read(q).raw) == {"answer": "does not contain"}
    r = question("c09.fixture.successor.R1")
    reading = world.subject(dolma, [r]).read(r)
    assert set(reading.raw) == {CANNOT} and _missing_of("c05.unmutated.files") in reading.raw[CANNOT]


def test_the_shard_that_holds_a_generation_output_is_cannot_answer_downstream_of_the_translation(need_module, tmp_path):
    dolma = need_module(MODULE)
    grade = _mod("baselines.grade")
    world = World(tmp_path, {"E/shard-3": ["T", "Z-001"]})
    q = question("c06.fixture.downstream.E/shard-3")
    reading = world.subject(dolma, [q]).read(q)
    assert set(reading.raw) == {CANNOT}
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "cannot answer"


# --- A failed run, a missing run, a question that records none -----------------------------


def test_a_run_that_failed_gives_its_attempt_for_every_question_it_covers(need_module, tmp_path):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    attempt = interfaces.Attempt(id="attempt-9", command=("step",), exit_code=0, output="no row")
    q = question("c05.unmutated.E/shard-1")
    subject = dolma.DolmaSubject({target_of(q): attempt}, tie_record=tmp_path / "tie.json", requests=REQUESTS)
    assert subject.read(q) is attempt


def test_a_question_on_a_target_with_no_run_is_an_error_not_an_absence(need_module, tmp_path):
    dolma = need_module(MODULE)
    subject = dolma.DolmaSubject({}, tie_record=tmp_path / "tie.json", requests=REQUESTS)
    with pytest.raises(ValueError):
        subject.read(question("c05.unmutated.E/shard-1"))


# --- Every question the table routes to the subject -------------------------------------

# The number of questions of the Dolma subject and how many have a route, counted from the
# rows of check10.toml and the sizes of the listed rows by a script that reads the table: 391 questions (the same for every subject), 105 with a route. The
# groups: c03.spans 4, c03.records 6, c03.pairs.split 2, c04.interrupted-write 2,
# c04.count-499 1, c05.unmutated.shards 6, c05.f.map-deleted 24, c05.record-X@A.unmutated 10,
# c05.f.digest 25, c05.withdraw-C.unmutated 2, c06.documents-mapped 6, c06.downstream 6,
# c09.artifact-actions 8, c09.successors 3.
QUESTIONS_OF_THE_SUBJECT = 391
QUESTIONS_WITH_A_ROUTE = 105

# The listed rows whose reading the table leaves open (see the module words).
OPEN_ROWS = frozenset({"c06.generated.downstream-of-translation", "c09.10k.successor-shards"})

KEYS = {
    "c03.spans.per-ordinal": {"per_ordinal", "notes"},
    "c03.spans.total": {"per_shard", "notes"},
    "c03.records.capture": {"per_ordinal"},
    "c03.records.position-recovery": {"per_ordinal"},
    "c03.records.token-matching": {"per_ordinal"},
    "c03.pairs.split": {"record"},
    "c04.kill-after-100": {"records"},
    "c04.retry": {"records"},
    "c04.count-499": {"answer"},
    "c06.generated.documents-mapped": COUNT_KEYS,
    "c09.fixture.shards.counts": {"counts", "keep", "rewrite"},
    "c09.10k.partitioned.kept-shards": {"keep"},
    "c09.10k.round-robin.all-rewrite": {"keep", "rewrite"},
}


def kind_of(artifact: str | None) -> str:
    if artifact is None:
        return "set"
    if artifact.startswith(("src/", "gen/", "import/")) or re.fullmatch(r"(successor of )?R\d+", artifact):
        return "files"
    if "/mix-" in artifact:
        return "mixtures"
    if "@" in artifact:
        return "checkpoints"
    if artifact == "(listing)":
        return "listing"
    return "shard"


def expected_keys(listed: str, artifact: str | None) -> set[str] | None:
    """The keys of a reading that is not cannot answer, or None where the artifact is of a
    kind the toolkit holds no record for."""
    if kind_of(artifact) not in ("set", "shard"):
        return None
    if listed in KEYS:
        return KEYS[listed]
    if listed.startswith("c06.fixture.documents-mapped."):
        return {"documents", "recovered", "unknown"}
    if listed.endswith(".action"):
        return {"action"}
    return {"answer"}


UNIVERSAL = {
    "E/shard-1": ["B-001", "Z-001", "Z-002"],
    "E/shard-2": ["B-002", "Z-003"],
    "E/shard-3": ["T", "Z-004"],
    "E-rr/shard-1": ["B-001", "Z-001"],
    "E-rr/shard-2": ["B-002", "Z-003"],
    "E-rr/shard-3": ["T", "Z-004"],
    "c04/shard-500": SCRATCH_IDS,
    "successor of E/shard-3": ["Z-004"],
    "shard-1": ["Z-010", "Z-011"],
    "shard-2": ["Z-012"],
}
UNIVERSAL_PATHS = {"E-rr/shard-1": "pipeline/round-robin/tokenizer-input/rr-1.jsonl",
                   "E-rr/shard-2": "pipeline/round-robin/tokenizer-input/rr-2.jsonl",
                   "E-rr/shard-3": "pipeline/round-robin/tokenizer-input/rr-3.jsonl"}
UNIVERSAL_PAIRS = [{"pair": "pair-1", "members": [{"shard": "shard-1", "ordinal": 0},
                                                  {"shard": "shard-2", "ordinal": 0}]}]


def routed_questions() -> list:
    return [q for q in _questions().values() if q.subject == "dolma" and q.route is not None]


def key_problems(subject, questions) -> list[str]:
    interfaces = _mod("baselines.interfaces")
    problems = []
    for q in questions:
        listed = q.listed_row
        if listed in OPEN_ROWS:
            continue
        given = _listed_rows()[listed]["given"]
        reading = subject.read(q)
        if not isinstance(reading, interfaces.Reading) or reading.question_id != q.id:
            problems.append(f"{q.id}: not a reading of the question")
            continue
        if reading.route != q.route:
            problems.append(f"{q.id}: route {reading.route!r}")
        keys = set(reading.raw)
        wanted = expected_keys(listed, given.get("artifact"))
        if wanted is None:
            if keys != {CANNOT}:
                problems.append(f"{q.id}: {sorted(keys)} for a kind with no record")
        elif given.get("artifact") in ("E/shard-1", "E/shard-2", "E-rr/shard-1", "E-rr/shard-2",
                                       "successor of E/shard-3"):
            if keys != wanted:
                problems.append(f"{q.id}: {sorted(keys)}, wanted {sorted(wanted)}")
        elif keys not in (wanted, {CANNOT}):
            problems.append(f"{q.id}: {sorted(keys)}, wanted {sorted(wanted)} or cannot answer")
        if given.get("set") is not None and listed.startswith(("c03.", "c06.generated.documents-mapped")) \
                and keys != wanted:
            problems.append(f"{q.id}: a set row reads {sorted(keys)}")
    return problems


def test_every_question_the_table_routes_to_the_subject_reads_the_values_its_rule_names(
        need_module, tmp_path, facts):
    dolma = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    subject_questions = [q for q in _questions().values() if q.subject == "dolma"]
    assert len(subject_questions) == QUESTIONS_OF_THE_SUBJECT == facts.questions_per_subject
    routed = routed_questions()
    assert len(routed) == QUESTIONS_WITH_A_ROUTE
    # 101 are read and checked below; the 4 of the two open rows (three sizes of the first, one
    # of the second) are not.
    checked = [q for q in routed if q.listed_row not in OPEN_ROWS]
    assert len(checked) == 101 and len(routed) - len(checked) == 4
    # The same count from the table's own rows: every dolma row that records something, times
    # the sizes each of its group's listed rows is asked at.
    from_table = [
        (listed, size) for (listed, subject, size) in facts.question_triples()
        if subject == "dolma" and _check10_rows()[f"c10.{facts.group_of(listed)}.dolma"].get("route")
    ]
    assert sorted(from_table, key=str) == sorted(((q.listed_row, q.size) for q in routed), key=str)

    world = World(tmp_path, UNIVERSAL, UNIVERSAL_PATHS)
    subject = world.subject(dolma, routed, split_pairs=UNIVERSAL_PAIRS,
                            stopped_rows=tuple(scratch_rows(world, 100)),
                            retry_rows=tuple(scratch_rows(world, 500)), retry_exit_code=0,
                            token_file_tokens=SCRATCH_TOKENS)
    assert key_problems(subject, checked) == []
    # The same check on a reader that leaves one named value out finds the fault.
    for key in ("documents_with_no_tied_row_per_shard", "per_ordinal", "record", "answer"):
        assert key_problems(_drop_one_key(subject, key), checked) != [], key
    # Every question whose row records none is read as no reading: no raw, no route.
    none_questions = [q for q in subject_questions if q.route is None]
    assert len(none_questions) == QUESTIONS_OF_THE_SUBJECT - QUESTIONS_WITH_A_ROUTE
    bare = dolma.DolmaSubject({}, tie_record=tmp_path / "none-tie.json", requests=REQUESTS)
    for q in none_questions:
        reading = bare.read(q)
        assert isinstance(reading, interfaces.Reading) and reading.question_id == q.id, q.id
        assert reading.raw is None and reading.route is None, q.id
    # What the table leaves open is named, and each is a listed row of a routed group.
    assert OPEN_ROWS <= {q.listed_row for q in routed}


# --- The module -------------------------------------------------------------------------


def test_the_module_imports_nothing_forbidden_and_holds_no_local_path(need_module):
    module = need_module(MODULE)
    source = Path(module.__file__).read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, roots & FORBIDDEN_IMPORT_ROOTS
    for text in FORBIDDEN_TEXT:
        assert text not in source, text


# --- A set the plant generator wrote: its unparseable lines are no stop of a question -------------------------

# The generator's own options for a small set (docs/plant.md section 3). "Cases per group" is 3, and
# there are three groups, the first of them lines that are not JSON; every case sits in Z (section 4.1
# and the table of the sanitizer cases in section 4.5).
PLANT_ARGUMENTS = ("--records", "800", "--seed", "7", "--length", "160", "--pairs", "24",
                   "--sanitizer-cases", "3", "--chunks", "20")
UNPARSEABLE_PLANTED = {"sources/Z.jsonl": 3}


@pytest.fixture(scope="module")
def planted_set(tmp_path_factory):
    """A set written by ``fixture/plant/generate.py`` in a folder of its own."""
    import subprocess
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


def _readable_ids(rows: list[dict], file: str, count: int) -> list[str]:
    """The ids of the first ``count`` lines of a source file that hold a record (a line that is not JSON
    holds none)."""
    found = [row["id"] for row in rows if row["file"] == file and not _is_unparseable(row)]
    assert len(found) >= count, file
    return found[:count]


def test_a_question_on_a_generated_set_resolves_the_ids_past_its_unparseable_lines_and_counts_them(
        need_module, tmp_path, planted_set):
    import shutil

    dolma = need_module(MODULE)
    companion = _mod("baselines.companion")
    rows = _planted_rows(planted_set)
    assert sum(1 for row in rows if _is_unparseable(row)) == UNPARSEABLE_PLANTED["sources/Z.jsonl"]
    build = tmp_path / "build"
    shutil.copytree(planted_set / "sources", build / "sources")
    # Two chunks of the round-robin layout. The first holds a line of A and two of Z, the second a line of
    # B and two of Z; the ids are those of lines that parse.
    shards = {"shard-1": [*_readable_ids(rows, "sources/A.jsonl", 1), *_readable_ids(rows, "sources/Z.jsonl", 2)],
              "shard-2": [*_readable_ids(rows, "sources/B.jsonl", 1), *_readable_ids(rows, "sources/Z.jsonl", 4)[2:]]}
    chunk_files, companion_rows = {}, []
    for name, ids in shards.items():
        relative = f"pipeline/round-robin/tokenizer-input/{name}.jsonl"
        chunk_files[name] = relative
        (build / relative).parent.mkdir(parents=True, exist_ok=True)
        (build / relative).write_text("".join(json.dumps({"id": i, "text": "t"}) + "\n" for i in ids),
                                      encoding="utf-8")
        companion_rows += [companion.Row(start=3 * n, end=3 * n + 3, id=i, src=relative, loc=n + 1)
                           for n, i in enumerate(ids)]
    q = question("c09.10k.round-robin.all-rewrite")
    run = dolma.DolmaRun(build=build, rows=tuple(companion_rows), chunk_files=chunk_files)
    subject = dolma.DolmaSubject({target_of(q): run}, tie_record=tmp_path / "tie.json", requests=REQUESTS)
    try:
        reading, failure = subject.read(q), None
    except ValueError as error:
        reading, failure = None, error
    assert failure is None, f"the question stops on a line of the set: {failure!r}"
    # The rule of the row: the chunk with a line of A is "rewrite"; the chunk whose lines all resolve to
    # source files, none of them A's, is "keep".
    assert dict(reading.raw) == {"rewrite": 1, "keep": 1}
    # Never dropped silently: the subject keeps the count of each file, by the build's name.
    counts = getattr(subject, "unparseable_lines", None)
    assert counts is not None, "the subject keeps no count of the lines it could not parse"
    assert {name: dict(held) for name, held in counts.items()} == {"set-10k": UNPARSEABLE_PLANTED}


# 17 Nov 2026 (W8): a question reads exactly ONE build, picked by what its given names; no two builds share a key. A target
# that holds a build for each layout is keyed by the target and the layout; a single-layout row reads the run of its layout.


def test_a_single_layout_row_reads_the_run_of_its_layout_when_the_target_holds_a_build_for_each(need_module, tmp_path):
    dolma = need_module(MODULE)
    partitioned = World(tmp_path / "p", {"shard-1": ["B-001", "Z-001"], "shard-2": ["A-001", "Z-002"], "shard-3": ["Z-003"]})
    chunks = World(tmp_path / "r", {f"shard-{n:02d}": ["A-001", f"Z-{n:03d}"] for n in range(1, 21)})
    kept, all_rewrite = question("c09.10k.partitioned.kept-shards"), question("c09.10k.round-robin.all-rewrite")
    assert target_of(kept) == target_of(all_rewrite) == ("set", "10k")
    partitioned.write_tie(True)
    chunks.write_tie(True)
    subject = dolma.DolmaSubject(
        {("set", "10k", "partitioned"): partitioned.run(dolma), ("set", "10k", "round-robin"): chunks.run(dolma)},
        tie_record=partitioned.tie_path, requests=REQUESTS)
    # Each row is read from the build of its own layout, never from the other: the kept shards are the partitioned build's
    # (shard-1 and shard-3), the chunks are the round-robin build's twenty.
    assert list(subject.read(kept).raw["keep"]) == ["shard-1", "shard-3"]
    assert dict(subject.read(all_rewrite).raw) == {"rewrite": 20, "keep": 0}
    # A row that names no layout on a target that holds both builds is an error, not a reading from either.
    unlayered = question("c06.generated.drops", "10k")
    both = dolma.DolmaSubject(
        {(*target_of(unlayered), "partitioned"): partitioned.run(dolma), (*target_of(unlayered), "round-robin"): chunks.run(dolma)},
        tie_record=partitioned.tie_path, requests=REQUESTS)
    with pytest.raises(ValueError, match="holds a build for each"):
        both._run_of(unlayered)
    # A target with one build in another layout is not read for a layout it has no build in.
    only = dolma.DolmaSubject({("set", "10k", "round-robin"): chunks.run(dolma)}, tie_record=chunks.tie_path, requests=REQUESTS)
    with pytest.raises(ValueError, match="no build of set 10k in the layout partitioned"):
        only.read(kept)


# 18 Nov 2026 (W8): the split pairs of a set are the plant list of THAT set. Box pass 9 (the 100k world) read the 100k questions
# against the one list the world kept from the smallest set: the reading was the small one, shard and line for shard and line.


def test_a_question_reads_the_split_pairs_of_its_own_set_and_a_missing_list_is_no_empty_one(need_module, tmp_path):
    dolma = need_module(MODULE)
    small, big = question("c03.pairs.split", "small"), question("c03.pairs.split", "100k")
    assert target_of(small) != target_of(big)
    world = set_world(tmp_path)
    world.write_tie(True)
    run = world.run(dolma)
    pairs_small = [{"pair": "pair-s", "members": [{"shard": "shard-1", "ordinal": 4}, {"shard": "shard-2", "ordinal": 1}]}]
    pairs_big = [{"pair": "pair-b", "members": [{"shard": "shard-1", "ordinal": 0}, {"shard": "shard-2", "ordinal": 3}]}]
    subject = dolma.DolmaSubject({target_of(small): run, target_of(big): run}, tie_record=world.tie_path, requests=REQUESTS,
                                 split_pairs={target_of(small): pairs_small, target_of(big): pairs_big})
    got_small, got_big = subject.read(small), subject.read(big)
    assert [p["pair"] for p in got_small.raw["record"]] == ["pair-s"]
    assert [p["pair"] for p in got_big.raw["record"]] == ["pair-b"], "the big set is read against its own list, not the small one"
    assert [dict(m) for m in got_big.raw["record"][0]["members"]] == [{"shard": "shard-1", "line": 1}, {"shard": "shard-2", "line": 4}]
    # A set the mapping holds no list for is an error, never a reading of no pairs (which would be "right" by default).
    only = dolma.DolmaSubject({target_of(small): run, target_of(big): run}, tie_record=world.tie_path, requests=REQUESTS,
                              split_pairs={target_of(small): pairs_small})
    with pytest.raises(ValueError, match="no split pairs were given for set 100k"):
        only.read(big)


def test_the_ids_the_tools_own_rows_give_at_the_listed_positions_are_reported_beside_the_reading_and_not_graded(need_module, tmp_path):
    dolma, grade = need_module(MODULE), _mod("baselines.grade")
    world = set_world(tmp_path)
    q = question("c03.pairs.split", "small")
    pairs = [{"pair": "pair-1", "members": [{"shard": "shard-1", "ordinal": 4, "id": "Z-005"},       # the tool's row at this place is Z-005
                                            {"shard": "shard-2", "ordinal": 1, "id": "A-000001"}]}]   # the tool's row there is Z-007
    subject = world.subject(dolma, [q], split_pairs=pairs)
    notes: list = []
    subject.note_fault = notes.append
    reading = subject.read(q)
    assert set(reading.raw) == {"record"}, "the reading keeps ``record`` alone; the ids are a note on the line"
    note = notes[0]["tool_id_at_each_listed_position_is_the_listed_member"]
    assert note["members_with_a_listed_id"] == 2 and note["equal"] == 1
    assert note["first_differing"] == [{"pair": "pair-1", "shard": "shard-2", "ordinal": 1, "tool_id": "Z-007", "listed_id": "A-000001"}]
    # Beside the line and ungraded: the grade compares the record alone.
    expect = {"label": "recovered", "record": [{"pair": "pair-1", "members": [{"shard": "shard-1", "line": 5}, {"shard": "shard-2", "line": 2}]}]}
    assert grade.grade(q, expect, reading).status == "right"
