"""The Dolma toolkit as a subject of Check 10 (plan 5.5 Check 10; table README, section
"Check 10": the subject ``dolma``, "What a run hands a subject, and its conditions" and "The
Dolma tie to an ordinal"; the ``c10.<group>.dolma`` rows of ``check10.toml``).

Nothing here starts a tool. The module builds the commands of the tokenizer and the mixer
(through ``baselines.sandbox``), checks the companion rows a tokenizer run left, and reads
those rows, the plain files of a build and the recorded tie into the values the rows of the
table name. A runner outside this module starts the commands and hands the results in.

What is held to, by the table:

- The tokenizer and the mixer run through the copy's own interpreter with ``-m`` and the
  module of the toolkit's command line (``dolma.cli``; the toolkit has no ``dolma.__main__``)
  and are handed paths relative to the build folder, in which they start: the chunk files,
  the destination, the tokenizer file, the two work folders of the tokenizer and the mixer's
  config. The tokenizer is told its tokenizer file, the end-of-sequence token id, one process
  and its work folders; the chunk rows hold an id, so no id field is switched off. The step of
  either tool binds the copy read-only, and the tokenizer's file with it. The sandbox clears
  the environment, so the step gives the tool a scratch folder for temporary files inside the
  work folder, a path with the copy's ``bin`` first, a language, and the two offline words.
- The mixer's stream holds a filter whose one exclusion is a text no chunk holds: the toolkit
  refuses a stream with neither a filter nor a span replacement, so a stream that is to leave
  nothing out still needs one.
- The destination has one meaning: a path relative to the build folder, in the command and
  where the rows are read.
- After a tokenizer run the rows of each chunk file are counted against its lines. A run that
  writes no row for a chunk file that has lines is a failed attempt, not an answer, whatever
  its exit code. A failed attempt keeps no local path in its command or its output.
- A companion row is tied to a document by its ``src`` and ``loc``, never by its place in the
  file. The tie is used only for a chunk file the tie record shows; for any other the question
  that needs an ordinal is cannot answer, with the evidence its own row gives. Only an answer
  that used the tie is reported as having used it (``used_tie``); the mark is then carried by
  the record of the answer.
- The shard rule reads the rows of one chunk file by ``src`` alone. An artifact that is not a
  shard has no record in the toolkit and reads cannot answer, with the missing record of the
  row for its kind.
- The faults of Checks 4 and 5 are put to a copy of the rows (``baselines.faults``); the rows
  the run holds are never changed.

What the module leaves open, and does not decide:

- ``c06.generated.downstream-of-translation`` and ``c09.10k.successor-shards``: the rule of
  the table does not say how the readings of several shards become one answer, so ``read``
  raises ``ValueError`` for them (a failed run still gives its ``Attempt``).
- A run that wrote some rows, but not all, for a chunk file: the row count check only
  refuses a chunk file with no row at all.
- A shard with no row (the answer "not covered") asked for an action: cannot answer, with
  the missing row as evidence.
- The second run of ``c04.retry`` ends in one of three ways. Exit 0 is no refusal. A non-zero
  exit of 1 to 128 whose output ends in an error of one of the toolkit's own error classes (the
  classes of its error module, printed with the module's dotted path or bare) is a refusal; the
  text of that error, local paths cut, is the mismatch named, and an error with no text names
  none. Every other non-zero end is a failed attempt, never a reading: an error of no class of
  the toolkit, a signal (a negative code, or 128 plus the signal's number), no output, and an
  output whose last line is not an error of the toolkit (an error whose text runs over several
  lines is read from its last line only, so it is not recognised). The attempt takes its id and
  its command from the run; a failed second run with no id is a ``ValueError``.
- A second run whose exit code is not given: ``ValueError``. The ranges of its rows are said to
  index the token file when they chain from 0 without a gap or an overlap, end within the
  token count (at it, for a row for every line) and name each line once; a row list with a
  repeated line therefore reads "no", and whether that is what the table means is not decided.
- The runs are keyed by target alone, and a set row reads every chunk file its run holds. A
  set target is shared by the rows of Check 3, ``c06.generated.documents-mapped`` (a build made
  without capture) and, at 10k, both layouts of Check 9. A ``DolmaRun`` holds the chunk files
  of one build, so the caller hands one subject for each build and layout; a guard on the
  layout folder of the chunk files is not possible, because the readings are also asked of
  chunk files laid out in another folder.
- The flags of the tokenizer and the keys of the mixer's file are those the toolkit's own
  command line module defines; nothing here starts the toolkit to check them. The mixer's
  config is written as yaml, the form its command line read it in (``dolma -c <yaml> mix``).
  The folders ``in`` and ``out`` of the tokenizer's work folder are made by the caller
  (``tokens_work_folders`` names them), as the script of the earlier tool run made them (its
  line 5) before it started the tokenizer; that the toolkit makes them itself is not shown.
  The mixer's work folders are named in its config, and the runner makes them too: the
  toolkit makes a temporary folder only for a work folder that is not named, and the mixer
  line of that script made none, so whether the mixer makes them is not shown. The mixer's
  output files are closed at the size that earlier run set; the tokenizer's are left to the
  toolkit.
- A stop of the writer (Check 4) counts only when the stopped writer's exit is that of the
  signal sent to it (a negative code, or 128 plus the signal's number). Any other exit means
  the writer ended on its own or by another signal: both questions of the stopped write are
  then the failed attempt of the stopped writer, which records that exit. The rows after the
  stop say whether the companion rows showed while the writer ran or only at its end; a run
  that does not say, or does not give the stopped writer's exit and the signal sent, is a
  ``ValueError``.
- A companion file a stop tore is read by ``read_stopped_rows`` as the rows that came out
  whole: those whose line ends with its newline, in a gzip file whose end is missing
  (``EOFError``) or followed by bytes that are no gzip member (``BadGzipFile``). A plain file
  cut inside a row is still a ``ValueError``. After a run that ended with exit code 0 a torn
  file is a failed attempt.
- A reading of an artifact the toolkit holds no record for (cannot answer) is made only for a
  run that wrote a row for every chunk file that has lines, and for at least one chunk file;
  any other run is a ``ValueError``, as a target with no run is.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from baselines import companion, faults, record, sandbox, table10, tie
from baselines.interfaces import Attempt, Reading

CANNOT = "cannot_answer"

# The toolkit's tokenizer appends an end-of-sequence token of its own to each document.
SPAN_NOTE = ("the numbers are those of the toolkit's own tokenizer run, which appends its own "
             "end-of-sequence token")

# The chunk file each fault of the table is put to (the ``fault`` text of its row).
FAULT_SHARD = {
    "f.map-deleted": "E/shard-1",
    "f.digest": "E/shard-3",
    "l.count-499": "c04/shard-500",
}
# The states a question may be asked in that put no fault.
UNFAULTED = (None, "unmutated")

# The group of the row that names the missing record of an artifact that is not a shard.
MISSING_GROUP = {
    "files": "c05.unmutated.files",
    "mixtures": "c05.unmutated.mixtures",
    "checkpoints": "c05.unmutated.checkpoints",
    "listing": "c05.listing.map-deleted",
}

# The listed rows whose reading the table leaves open.
OPEN_ROWS = frozenset({"c06.generated.downstream-of-translation", "c09.10k.successor-shards"})
# The words of the line of a row the table gives no join of the shards for (the runner, the combinations and OriginBlame use them).
NO_JOIN_TEXT = "the table gives no join of the shards' answers"

_COMPANION_SUFFIXES = (".csv", ".csv.gz")
_MANIFEST_SUFFIX = ".manifest.jsonl"
_HARNESS_FOLDER = "harness"
_CHUNK_FOLDER = "tokenizer-input"

# The module of the toolkit's command line: the package has ``dolma.cli.__main__`` and no
# ``dolma.__main__``.
TOOLKIT_MODULE = "dolma.cli"
# The token id the tokenizer appends to each document (the GPT-2 tokenizer's end of text).
EOS_TOKEN_ID = 50256
# The size at which the mixer closes an output file and starts the next.
MIX_OUTPUT_MAX_BYTES = 1000000000
# The one exclusion of the mixer's filter: a text no chunk holds.
NEVER_MATCHING_EXCLUDE = '.text == "this text never occurs"'
# The environment a step hands the tool: a scratch folder for temporary files the caller names.
_TEMP_VARIABLE = "TMPDIR"
# The sandbox clears the environment; the step sets these itself (the tool's own interpreter
# and its folder of commands first on the path, the language, and no hub reached for).
_LANGUAGE = "C.UTF-8"
_SYSTEM_PATH = "/usr/bin"

# What the rows after a stop say about when the companion rows showed.
ROWS_SHOWED_WHILE_RUNNING = "while the writer ran"
ROWS_SHOWED_AT_END = "only at its end"


# --- Paths and commands ------------------------------------------------------------------


def _relative(path: Any, what: str) -> str:
    """A path that is relative and stays inside the folder the tool starts in, or ``ValueError``."""
    text = os.fspath(path)
    parts = PurePosixPath(text).parts
    if not text or os.path.isabs(text) or ".." in parts:
        raise ValueError(f"{what} must be a relative path inside the build folder, got {text!r}")
    return text


def _work_folders(work_dir, what: str) -> tuple[str, str]:
    base = PurePosixPath(_relative(work_dir, what))
    return str(base / "in"), str(base / "out")


def tokens_work_folders(work_dir) -> tuple[str, str]:
    """The tokenizer's two work folders, ``in`` and ``out`` inside the folder ``work_dir``,
    relative to the build folder. The tokenizer is told them; the caller makes them, as the
    script of the earlier tool run made both (its line 5) before it started the tokenizer."""
    return _work_folders(work_dir, "the tokenizer's work folder")


def tokens_command(env_dir, chunk_files: Sequence[str], destination, *, tokenizer, work_dir) -> list[str]:
    """The tokenizer's command: the copy's interpreter with ``-m`` and the module of the
    toolkit's command line, the chunk files, the folder it writes to, its tokenizer file, the
    end-of-sequence token id, one process and its work folders, all paths relative to the
    build folder it starts in. The chunk rows hold an id and a text, so the id field is not
    switched off."""
    chunks = [_relative(chunk, "a chunk file") for chunk in chunk_files]
    target = _relative(destination, "the tokenizer's destination")
    tokenizer_file = _relative(tokenizer, "the tokenizer file")
    work_input, work_output = tokens_work_folders(work_dir)
    return sandbox.tool_command(env_dir, TOOLKIT_MODULE, [
        "tokens", "--documents", *chunks, "--destination", target,
        "--tokenizer.name_or_path", tokenizer_file,
        "--tokenizer.eos_token_id", str(EOS_TOKEN_ID),
        "--processes", "1",
        "--work_dir.input", work_input, "--work_dir.output", work_output])


def _step_environment(env, work, env_dir) -> dict[str, str]:
    """The environment of a step. The sandbox clears the environment, so the step sets what the
    tool needs: ``PATH`` (the ``bin`` folder of the copy first, then ``/usr/bin``), ``LANG``
    (``C.UTF-8``), and ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` (``1``). ``env`` holds
    the caller's ``TMPDIR``, a folder inside the work folder (the one place a tool may write).
    A ``TMPDIR`` that is missing, relative, outside the work folder or the work folder itself,
    or a value of the caller's for a word the step sets that differs from the step's, is a
    ``ValueError``."""
    variables = dict(env or {})
    scratch = variables.get(_TEMP_VARIABLE)
    if not isinstance(scratch, str) or not os.path.isabs(scratch):
        raise ValueError(f"a step needs {_TEMP_VARIABLE}, an absolute path inside the work folder")
    folder = os.path.normpath(scratch)
    work_folder = os.path.normpath(os.fspath(work)).rstrip("/")
    if not folder.startswith(work_folder + "/"):
        raise ValueError(f"{_TEMP_VARIABLE} must lie inside the work folder")
    fixed = {
        "PATH": os.path.join(os.path.normpath(os.fspath(env_dir)), "bin") + ":" + _SYSTEM_PATH,
        "LANG": _LANGUAGE,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for name, value in fixed.items():
        if variables.setdefault(name, value) != value:
            raise ValueError(f"{name} is {value!r} in a step, not {variables[name]!r}")
    return variables


def _copy_binds(env_dir, *more) -> list:
    """The read-only binds of a step: the folders of the copy's interpreter, the copy itself,
    and ``more``. All of them are bound after the work folder, so that each wins where it lies
    inside it."""
    return [*sandbox.interpreter_binds(env_dir), env_dir, *more]


def tokens_step(*, lock_prefix, work, home, build, env_dir, chunk_files, destination, tokenizer,
                work_dir, env) -> list[str]:
    """The tokenizer's command behind the lock, in a sandbox that starts in the build folder
    and holds the copy and the tokenizer file read-only; ``env`` is its environment (see
    ``_step_environment``)."""
    command = tokens_command(env_dir, chunk_files, destination, tokenizer=tokenizer, work_dir=work_dir)
    tokenizer_in_build = os.path.join(os.fspath(build), _relative(tokenizer, "the tokenizer file"))
    return sandbox.sandboxed_step(
        lock_prefix=lock_prefix, work=work, home=home, chdir=build, command=command,
        ro_binds=_copy_binds(env_dir, os.path.normpath(tokenizer_in_build)),
        env=_step_environment(env, work, env_dir))


def mix_command(env_dir, config) -> list[str]:
    """The mixer's command: the copy's interpreter with ``-m`` and the module of the toolkit's
    command line, and a config file."""
    config_file = _relative(config, "the mixer's config")
    return sandbox.tool_command(env_dir, TOOLKIT_MODULE, ["-c", config_file, "mix"])


def mix_step(*, lock_prefix, work, home, build, env_dir, config, env) -> list[str]:
    """The mixer's command behind the lock, in a sandbox that starts in the build folder and
    holds the copy read-only like the tokenizer's step; ``env`` is its environment."""
    return sandbox.sandboxed_step(
        lock_prefix=lock_prefix, work=work, home=home, chdir=build,
        command=mix_command(env_dir, config), ro_binds=_copy_binds(env_dir),
        env=_step_environment(env, work, env_dir))


def _layout_of(chunk_file: str) -> str:
    """The layout folder a chunk file lies in: ``<layout>/tokenizer-input/<shard>.jsonl``."""
    parts = PurePosixPath(chunk_file).parts
    if _CHUNK_FOLDER not in parts or parts.index(_CHUNK_FOLDER) == 0:
        raise ValueError(f"{chunk_file!r} is not the chunk file of a layout")
    return parts[parts.index(_CHUNK_FOLDER) - 1]


def mix_config(chunk_files: Sequence[str], destination, *, work_dir) -> dict:
    """The mixer's config: one stream over the chunk files of ONE layout of one build, with a
    filter whose one exclusion matches no document, so that nothing is left out (the toolkit
    refuses a stream that has neither a filter nor a span replacement), the size at which an
    output file is closed, and the mixer's two work folders, ``in`` and ``out`` inside the
    folder ``work_dir`` (relative to the build folder). The toolkit makes a temporary folder
    only for a work folder that is not named, so the runner makes these two folders before the
    mixer starts; that the mixer makes them itself is not shown."""
    chunks = [_relative(chunk, "a chunk file") for chunk in chunk_files]
    if not chunks:
        raise ValueError("the mixer needs at least one chunk file")
    if len({_layout_of(chunk) for chunk in chunks}) > 1:
        raise ValueError("the chunk files of the mixer lie in more than one layout")
    work_input, work_output = _work_folders(work_dir, "the mixer's work folder")
    return {
        "streams": [{"name": "chunks", "documents": chunks,
                     "output": {"path": _relative(destination, "the mixer's destination"),
                                "max_size_in_bytes": MIX_OUTPUT_MAX_BYTES},
                     "filter": {"syntax": "jq", "exclude": [NEVER_MATCHING_EXCLUDE]}}],
        "work_dir": {"input": work_input, "output": work_output},
        "processes": 1,
    }


def _yaml_scalar(value: Any) -> str:
    """A whole number as it is, a text in single quotes (a single quote doubled); a text with a
    line break, or any other value, is a ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"a config value cannot be written as yaml: {value!r}")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and "\n" not in value and "\r" not in value:
        return "'" + value.replace("'", "''") + "'"
    raise ValueError(f"a config value cannot be written as yaml on one line: {value!r}")


def _yaml_lines(value: Any, level: int) -> list[str]:
    """The lines of a mapping or a list at a nesting ``level`` (two spaces each): a key on
    its own line over a nested block, a dash for each list item."""
    pad = "  " * level
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, (Mapping, list)):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_lines(item, level + 1))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
        return lines
    for item in value:
        if isinstance(item, Mapping):
            inner = _yaml_lines(item, level + 1)
            lines.append(pad + "- " + inner[0][len(pad) + 2:])
            lines.extend(inner[1:])
        else:
            lines.append(f"{pad}- {_yaml_scalar(item)}")
    return lines


def write_mix_config(path, chunk_files: Sequence[str], destination, *, work_dir) -> None:
    """Write the mixer's config to ``path`` as yaml, the form its command line is given its
    config in. The text is made before the file, so a config that cannot be written leaves no
    file."""
    text = "\n".join(_yaml_lines(mix_config(chunk_files, destination, work_dir=work_dir), 0)) + "\n"
    Path(path).write_text(text, encoding="utf-8")


# --- A run and the row count check ---------------------------------------------------------


def _companion_files(folder) -> list[Path]:
    """The companion files (``.csv``, ``.csv.gz``) under a folder, in the order of their
    names. A folder that does not exist holds none."""
    base = Path(folder)
    if not base.is_dir():
        return []
    return sorted((p for p in base.rglob("*") if p.is_file() and p.name.endswith(_COMPANION_SUFFIXES)),
                  key=lambda p: p.relative_to(base).as_posix())


def read_run_rows(folder) -> list[companion.Row]:
    """The rows of every companion file (``.csv``, ``.csv.gz``) under a folder, in the order
    of the file names; the rows of one file in its own order. A folder that does not exist
    holds no row."""
    rows: list[companion.Row] = []
    for path in _companion_files(folder):
        rows.extend(companion.read_rows(path))
    return rows


# The size of one read of a gzip file torn by a stop.
_READ_SIZE = 1 << 16


def _row_of(fields: Sequence[str], line: int) -> companion.Row:
    """One companion row from the fields of its line, as ``companion.read_rows`` makes it."""
    if len(fields) != len(companion.COLUMNS):
        raise ValueError(f"companion row {line}: {len(fields)} fields, expected {len(companion.COLUMNS)}")
    start, end, document_id, src, loc = fields
    for column, text in (("start", start), ("end", end), ("loc", loc)):
        if not (text.isascii() and text.isdigit()):
            raise ValueError(f"companion row {line}: {column} is not an integer: {text!r}")
    if int(loc) < 1:
        raise ValueError(f"companion row {line}: loc counts from 1, got {int(loc)}")
    return companion.Row(start=int(start), end=int(end), id=document_id, src=src, loc=int(loc))


def _rows_before_the_tear(path: Path) -> list[companion.Row]:
    """The rows of a gzip companion file that came out whole before its end went missing or
    bytes that are no gzip member followed it: those whose line ends with its newline."""
    pieces: list[bytes] = []
    try:
        with gzip.GzipFile(path, "rb") as packed:
            while True:
                piece = packed.read1(_READ_SIZE)
                if not piece:
                    break
                pieces.append(piece)
    except (EOFError, gzip.BadGzipFile):
        pass
    data = b"".join(pieces)
    whole = data[:data.rfind(b"\n") + 1].decode("utf-8")
    return [_row_of(fields, line)
            for line, fields in enumerate(csv.reader(io.StringIO(whole, newline="")), start=1)]


def read_stopped_rows(folder) -> list[companion.Row]:
    """The rows of every companion file under a folder a stop left behind, in the order of
    the file names. A gzip file the stop tore (its end is missing, or bytes that are no gzip
    member follow it) gives the rows that came out whole before the tear, never an error; a
    file that is whole is read as ``read_run_rows`` reads it."""
    rows: list[companion.Row] = []
    for path in _companion_files(folder):
        try:
            rows.extend(companion.read_rows(path))
        except (EOFError, gzip.BadGzipFile):
            rows.extend(_rows_before_the_tear(path))
    return rows


def _line_count(path: Path) -> int:
    """The lines of a file, counted as the tokenizer counts them: by newline only."""
    data = path.read_bytes()
    if not data:
        return 0
    return data.count(b"\n") + (0 if data.endswith(b"\n") else 1)


# An absolute path, up to the next character a path does not hold; it is kept by its last name.
_LOCAL_PATH = re.compile(r"(?<![\w.])/[^\s\"'`):,]+")


def _without_local_paths(text: str) -> str:
    return _LOCAL_PATH.sub(lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1], text)


def run_tokens(runner, argv: Sequence[str], *, build, chunk_files: Sequence[str], destination,
               attempt_id: str) -> list[companion.Row] | Attempt:
    """Run the tokenizer once and check what it left.

    ``runner(argv)`` gives ``(exit_code, output)``. The rows are those of the files in the
    folder ``destination``, a path relative to the build folder as it is in the command (an
    absolute path or one with ``..`` is a ``ValueError`` before the tool starts). The result is
    the rows, or the failed ``Attempt`` ``attempt_id``: an exit code other than 0, rows that
    cannot be read (a file torn by an end that never came is one), or a chunk file with lines
    and no row (said by name in the attempt's output). Neither the command nor the output of
    an attempt holds a local path."""
    target = _relative(destination, "the tokenizer's destination")
    exit_code, output = runner(list(argv))
    command = tuple(_without_local_paths(str(word)) for word in argv)
    text = _without_local_paths(output or "")

    def failed(note: str = "") -> Attempt:
        return Attempt(id=attempt_id, command=command, exit_code=exit_code,
                       output=(text + "\n" + _without_local_paths(note)).strip() if note else text)

    if exit_code != 0:
        return failed()
    try:
        rows = read_run_rows(Path(build) / target)
    except (ValueError, EOFError, gzip.BadGzipFile) as error:
        return failed(f"the companion rows cannot be read: {error}")
    empty = _chunk_files_without_a_row(build, chunk_files, rows)
    if empty:
        return failed("no companion row for the chunk file " + ", ".join(empty))
    return rows


def _chunk_files_without_a_row(build, chunk_files: Sequence[str], rows: Sequence[companion.Row]) -> list[str]:
    """The chunk files that have lines and no row among ``rows``."""
    written = {row.src for row in rows}
    return [chunk for chunk in chunk_files
            if _line_count(Path(build) / chunk) > 0 and chunk not in written]


# The classes of the toolkit's error module; a traceback prints the module's dotted path in front
# of the class, and the class alone is read the same.
_DOLMA_ERROR_CLASSES = ("DolmaError", "DolmaFatalError", "DolmaShardError", "DolmaRetryableFailure",
                        "DolmaRustPipelineError", "DolmaConfigError")
_DOLMA_ERROR_MODULE = "dolma.core.errors"
_DOLMA_ERROR_LINE = re.compile(
    r"(?:" + re.escape(_DOLMA_ERROR_MODULE) + r"\.)?(?:" + "|".join(_DOLMA_ERROR_CLASSES)
    + r")(?::(?:\s+(?P<text>.*))?)?")
# A run ended by a signal is given as a negative code or as 128 plus the signal's number.
_SIGNAL_EXIT_BASE = 128
_LAST_SIGNAL = 64
_LAST_EXIT_CODE_OF_A_REFUSAL = _SIGNAL_EXIT_BASE


def _is_whole_number(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _refusal_text(exit_code: int, output: str | None) -> str | None:
    """The text of the error the toolkit refused with, "" for an error with no text, or None
    when the run did not refuse. A refusal is a non-zero exit of 1 to 128 whose output ends in
    an error of one of the toolkit's own error classes (the last non-blank line is the error
    that ended the run). A signal, no output, an error of any other class and anything else
    the last line does not say are no refusal."""
    if not 1 <= exit_code <= _LAST_EXIT_CODE_OF_A_REFUSAL:
        return None
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return None
    matched = _DOLMA_ERROR_LINE.fullmatch(lines[-1])
    if matched is None:
        return None
    return _without_local_paths(matched.group("text") or "").strip()


@dataclass(frozen=True)
class DolmaRun:
    """What one tokenizer run on one store or set left.

    build: the build folder.
    rows: the companion rows of the run.
    chunk_files: the relative path of the chunk file of each artifact (or shard) name.
    stopped_rows: the rows after the stopped write of Check 4.
    retry_rows, retry_exit_code, retry_output: the rows, the exit code and the output of the
        second run on the same output folder.
    retry_attempt_id, retry_command: the id and the command of the second run as an attempt;
        the failed ``Attempt`` of a second run that crashed carries them.
    token_file_tokens: the number of tokens the token file holds.
    stopped_exit_code, stopped_signal, stopped_attempt_id, stopped_command, stopped_output:
        the end of the stopped writer and the number of the signal sent to it, as an attempt;
        the failed ``Attempt`` of a stop that did not land carries them (it lands when the
        exit is that of the signal sent: its negative, or 128 plus its number).
    rows_showed_while_running: True where the companion rows showed while the writer ran (it
        was stopped at the first moment its output held enough rows), False where they showed
        only at its end (it was stopped at a moment before it ended).
    """

    build: Path
    rows: Sequence[companion.Row]
    chunk_files: Mapping[str, str]
    stopped_rows: Sequence[companion.Row] | None = None
    retry_rows: Sequence[companion.Row] | None = None
    retry_exit_code: int | None = None
    retry_output: str = ""
    token_file_tokens: int | None = None
    retry_attempt_id: str = ""
    retry_command: Sequence[str] = ()
    stopped_exit_code: int | None = None
    stopped_signal: int | None = None
    stopped_attempt_id: str = ""
    stopped_command: Sequence[str] = ()
    stopped_output: str = ""
    rows_showed_while_running: bool | None = None
    layouts: Sequence[str] = ()


# --- The table the subject reads -------------------------------------------------------------


@lru_cache(maxsize=1)
def _table_rows() -> Mapping[str, Any]:
    return {row.id: row for row in table10.load_table10().rows}


@lru_cache(maxsize=1)
def _listed_givens() -> Mapping[str, Mapping[str, Any]]:
    """The ``given`` of every row of Checks 1 to 9 that a group lists, by id: the core rows and the rows
    of an extension (``record._listed_givens`` reads the same rows, in the same way)."""
    table = table10._acceptance_table()
    found = {}
    for number in table10.LISTED_CHECKS:
        for row in table.load_check(number, table10.DEFAULT_PATH.parent):
            found[row.id] = row.given
    return found


def _kind_of(artifact: str | None) -> str:
    """What an artifact is: ``set`` (none named), ``files`` (a source, a generation output, a
    release, an import), ``mixtures``, ``checkpoints``, ``listing`` or ``shard``."""
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


def _record_files(build: Path) -> list[Path]:
    """The record files of a build that ``companion.resolve_id`` reads: the ``.jsonl`` files
    under ``sources/`` and ``gen/``, without the manifests beside them and the harness logs."""
    files: list[Path] = []
    for top in (companion.SOURCES_FOLDER, companion.GEN_FOLDER):
        base = build / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.jsonl")):
            relative = path.relative_to(build)
            if path.name.endswith(_MANIFEST_SUFFIX):
                continue
            if top == companion.GEN_FOLDER and len(relative.parts) > 2 and relative.parts[1] == _HARNESS_FOLDER:
                continue
            if path.is_file():
                files.append(path)
    return files


def _holders_of_ids(build: Path, unparseable: dict[str, int] | None = None) -> dict[str, list[str]]:
    """The files of a build, by the id of each record they hold. A line that is not JSON holds
    no record (the line reader of the tool, ``waterwhite.capture.line_text``, gives such a line
    no text): it adds no id, and when ``unparseable`` is handed in it is counted there, for the
    file it is in."""
    holders: dict[str, list[str]] = {}
    for path in _record_files(build):
        relative = path.relative_to(build).as_posix()
        with open(path, "rb") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    held = json.loads(line.decode("utf-8"))
                except (ValueError, RecursionError):
                    if unparseable is not None:
                        unparseable[relative] = unparseable.get(relative, 0) + 1
                    continue
                if not isinstance(held, dict):
                    raise ValueError(f"{path.name} line {number}: a record is not an object")
                identifier = held.get("id")
                if identifier is not None and relative not in holders.setdefault(identifier, []):
                    holders[identifier].append(relative)
    return holders


# --- The subject -------------------------------------------------------------------------------


def _rows_sha256(rows) -> str:
    """The sha256 of rows (dicts) as canonical JSON: a fault note names the rows before and after it."""
    text = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class DolmaSubject:
    """The Dolma toolkit read for the questions the table routes to it.

    ``runs`` maps the target of a question (``baselines.record.attempt_target``) to the
    ``DolmaRun`` of that store or set, or to the failed ``Attempt`` of its run; a target that holds a build for each layout is
    keyed by the target and the layout (three words) and a question reads the run of the layout it names (``_run_of``). ``tie_record``
    is the path of the recorded tie show. ``requests`` maps the words of a request (as the
    table writes them) to ``{"files": [relative paths of the files of the members it names],
    "record": <an id, only where the request narrows to one record>}``. ``split_pairs`` is the
    plant list's split pairs, ``[{"pair": ..., "members": [{"shard": ..., "ordinal": ...}]}]``.
    """

    name = "dolma"

    def __init__(self, runs: Mapping[tuple[str, str], DolmaRun | Attempt], *, tie_record,
                 requests: Mapping[str, Mapping[str, Any]], split_pairs: Sequence[Mapping[str, Any]] = ()):
        self._runs = dict(runs)
        self._tie_path = Path(tie_record)
        self._requests = requests
        # The plant list's split pairs: one list, or (W8, 18 Nov 2026: a question reads exactly one build) a mapping from a target to
        # the list of THAT set; the world builds each set's own pairs and a reading from another size's list is no reading of this one.
        self._split_pairs = ({tuple(target): [dict(pair) for pair in pairs] for target, pairs in split_pairs.items()}
                             if isinstance(split_pairs, Mapping) else [dict(pair) for pair in split_pairs])
        self._tied: set[str] = set()
        self._holders: dict[str, dict[str, list[str]]] = {}
        self._unparseable: dict[str, dict[str, int]] = {}
        # Called with one note for every fault put to a copy of the rows (see ``_after_fault``); the
        # runner sets it so that the note reaches the line of the question.
        self.note_fault: Callable[[dict], None] | None = None

    @property
    def unparseable_lines(self) -> dict[str, dict[str, int]]:
        """The lines of the builds read so far that are not JSON: for each build, named
        ``<kind>-<name>`` of its store or set, the count of each file that has such lines.
        A build is counted when a question first needs its ids."""
        found: dict[str, dict[str, int]] = {}
        for target, run in self._runs.items():
            if isinstance(run, DolmaRun) and str(run.build) in self._unparseable:
                found["-".join(target)] = dict(self._unparseable[str(run.build)])
        return found

    # -- what a reading used --

    def used_tie(self, question) -> bool:
        """Whether the last answer to ``question`` used the tie to an ordinal."""
        return question.id in self._tied

    # -- the question --

    def read(self, question) -> Reading | Attempt:
        self._tied.discard(question.id)
        if question.subject != self.name:
            raise ValueError(f"{question.id} is a question of {question.subject}, not of {self.name}")
        if question.route is None:
            return Reading(question.id, None, None)
        run = self._run_of(question)
        if isinstance(run, Attempt):
            return run
        if question.listed_row in OPEN_ROWS:
            raise ValueError(f"the table leaves the reading of {question.listed_row} open")
        raw = self._raw(question, run)
        if isinstance(raw, Attempt):
            return raw
        return Reading(question.id, raw, question.route)

    def _run_of(self, question) -> DolmaRun | Attempt:
        """The ONE run a question reads (17 Nov 2026, W8: a question reads exactly one build, picked by what its ``given`` names).
        A run is kept under its target, or, where a target holds a build for each layout, under the target and the layout
        (a key of three words: kind, name, layout). A question that names a layout reads the run of that layout, or the run
        of the target when the target holds one build; a question that names none, on a target that holds a build for each
        layout, is an error, never a reading from either."""
        target = record.attempt_target(question)
        layout = record.layout_of(question)
        if layout is not None and (*target, layout) in self._runs:
            return self._runs[(*target, layout)]
        if target in self._runs:
            return self._runs[target]
        held = sorted(key[2] for key in self._runs if len(key) == 3 and key[:2] == tuple(target))
        if held and layout is None:
            raise ValueError(f"{target[0]} {target[1]} holds a build for each of the layouts {held} and "
                             f"{question.listed_row} names no layout")
        if held:
            raise ValueError(f"no build of {target[0]} {target[1]} in the layout {layout} (built: {held})")
        raise ValueError(f"no run of the toolkit for {target[0]} {target[1]}")

    def _row(self, question):
        return _table_rows()[question.row_id]

    def _chunk(self, run: DolmaRun, name: str) -> str:
        if name not in run.chunk_files:
            raise ValueError(f"the run holds no chunk file for {name}")
        return run.chunk_files[name]

    @staticmethod
    def _require_a_tokens_run(run: DolmaRun) -> None:
        """A run that did not write a row for every chunk file that has lines, or no row at
        all, is no successful tokenizer run of its store (``ValueError``): a store the tool
        did not run on cannot be said to hold no record."""
        if not run.rows:
            raise ValueError("the run holds no companion row: it is no successful tokenizer run")
        empty = _chunk_files_without_a_row(run.build, list(run.chunk_files.values()), run.rows)
        if empty:
            raise ValueError("the run holds no companion row for the chunk file " + ", ".join(empty))

    def _tie_or_cannot(self, question, run: DolmaRun, names: Sequence[str]) -> dict | None:
        """None, and the tie noted as used, where it is shown for the chunk file of every
        name; else the cannot answer of the question's own row."""
        shown = tie.read_result(self._tie_path)
        if all(tie.is_shown(shown, self._chunk(run, name)) for name in names):
            self._tied.add(question.id)
            return None
        return {CANNOT: self._row(question).ordinal["not_shown"]["evidence"]}

    # -- rows --

    @staticmethod
    def _by_src(rows: Sequence[companion.Row]) -> dict[str, dict[int, list[companion.Row]]]:
        """The rows grouped by ``src`` and then by ``loc``."""
        grouped: dict[str, dict[int, list[companion.Row]]] = {}
        for row in rows:
            grouped.setdefault(row.src, {}).setdefault(row.loc, []).append(row)
        return grouped

    @staticmethod
    def _tied_row(held: Mapping[int, list[companion.Row]], ordinal: int) -> companion.Row | None:
        """The one row of the document with this ordinal; None where it has none or two."""
        found = held.get(ordinal + 1, [])
        return found[0] if len(found) == 1 else None

    def _after_fault(self, run: DolmaRun, state: str | None) -> list[companion.Row]:
        """The rows of the run with the fault of ``state`` put to a copy."""
        if state in UNFAULTED:
            return list(run.rows)
        if state not in FAULT_SHARD:
            raise ValueError(f"the state {state!r} puts no fault to the companion rows")
        chunk = self._chunk(run, FAULT_SHARD[state])
        as_dicts = [{"start": r.start, "end": r.end, "id": r.id, "src": r.src, "loc": r.loc}
                    for r in run.rows]
        changed = faults.apply_fault(state, "dolma", as_dicts, chunk)
        if self.note_fault is not None:
            self.note_fault({"fault": state, "chunk_file": chunk, "rows_before": len(as_dicts),
                             "rows_after": len(changed), "rows_sha256_before": _rows_sha256(as_dicts),
                             "rows_sha256_after": _rows_sha256(changed)})
        return [companion.Row(**row) for row in changed]

    # -- the shard rule --

    def _ids_to_files(self, run: DolmaRun) -> dict[str, list[str]]:
        key = str(run.build)
        if key not in self._holders:
            counted: dict[str, int] = {}
            self._holders[key] = _holders_of_ids(Path(run.build), counted)
            self._unparseable[key] = counted
        return self._holders[key]

    def _resolve(self, run: DolmaRun, identifier: str) -> str | None:
        files = self._ids_to_files(run).get(identifier, [])
        if len(files) > 1:
            raise ValueError(f"id {identifier!r} is held by {len(files)} files: {files}")
        return files[0] if files else None

    def _shard_word(self, run: DolmaRun, rows: Sequence[companion.Row], request: str,
                    name: str) -> tuple[str | None, str | None]:
        """The shard rule for one shard: ``(word, None)`` with the word "contains", "does not
        contain" or "not covered", or ``(None, evidence)`` for cannot answer."""
        chunk = self._chunk(run, name)
        held = [row for row in rows if row.src == chunk]
        if not held:
            return "not covered", None
        named = self._requests[request]
        files = set(named["files"])
        resolved = [(row, self._resolve(run, row.id)) for row in held]
        if "record" in named:
            hit = any(row.id == named["record"] for row, _ in resolved)
        else:
            hit = any(found in files for _, found in resolved if found)
        if hit:
            return "contains", None
        for row, found in resolved:
            if found is None:
                return None, f"the id {row.id} resolves to no file"
            if companion.kind_of(found) == companion.KIND_GENERATION:
                return None, (f"the row of {row.id} resolves to {found}, a generation output; the "
                              "toolkit has no derived-from link that says what that record was made from")
        return "does not contain", None

    def _actions(self, run, rows, request, names) -> dict:
        """The words of the shard rule for every name, turned into keep and rewrite lists, or the
        cannot answer that names the first shard that has no word."""
        keep, rewrite = [], []
        for name in names:
            word, why = self._shard_word(run, rows, request, name)
            if word is None or word == "not covered":
                return {CANNOT: f"{name}: {why or 'no row has this src'}"}
            (rewrite if word == "contains" else keep).append(name)
        return {"keep": keep, "rewrite": rewrite}

    # -- the readings --

    def _raw(self, question, run: DolmaRun) -> dict | Attempt:
        listed = question.listed_row
        given = _listed_givens()[listed]
        artifact = given.get("artifact")
        kind = _kind_of(artifact)
        if kind in MISSING_GROUP:
            self._require_a_tokens_run(run)
            return {CANNOT: _table_rows()[f"c10.{MISSING_GROUP[kind]}.dolma"].missing}
        if listed.startswith(("c03.spans", "c03.records")) or listed == "c03.pairs.split":
            return self._set_row(question, run, listed)
        if listed == "c06.generated.documents-mapped":
            return self._generated_documents(question, run)
        if listed.startswith("c06.fixture.documents-mapped."):
            return self._fixture_documents(question, run, artifact)
        if listed in ("c04.kill-after-100", "c04.retry"):
            return self._writer_state(run, listed, artifact)
        request = given.get("request")
        rows = self._after_fault(run, given.get("state"))
        if listed == "c09.fixture.shards.counts":
            names = sorted(n for n in run.chunk_files if n.startswith(given["run"] + "/"))
            held = self._actions(run, rows, request, names)
            if CANNOT in held:
                return held
            return {"counts": {"keep": len(held["keep"]), "rewrite": len(held["rewrite"])},
                    "keep": held["keep"], "rewrite": held["rewrite"]}
        if listed in ("c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"):
            held = self._actions(run, rows, request, sorted(run.chunk_files))
            if CANNOT in held:
                return held
            if listed.endswith("kept-shards"):
                return {"keep": held["keep"]}
            return {"keep": len(held["keep"]), "rewrite": len(held["rewrite"])}
        word, why = self._shard_word(run, rows, request, artifact)
        if listed.startswith("c09.") and listed.endswith(".action"):
            if word is None or word == "not covered":
                return {CANNOT: why or "no row has this src"}
            return {"action": "rewrite" if word == "contains" else "keep"}
        if word is None:
            return {CANNOT: why}
        return {"answer": word}

    def _per_ordinal(self, run: DolmaRun, by_src, name: str, value) -> list:
        """One value for each line of the chunk file of ``name``: that of its tied row, None
        where the document has none (or two), never that of a neighbour."""
        chunk = self._chunk(run, name)
        held = by_src.get(chunk, {})
        lines = _line_count(Path(run.build) / chunk)
        tied = (self._tied_row(held, ordinal) for ordinal in range(lines))
        return [None if row is None else value(row) for row in tied]

    def _set_row(self, question, run: DolmaRun, listed: str) -> dict:
        names = list(run.chunk_files)
        if listed == "c03.spans.total":
            totals = {name: sum(row.end - row.start for row in run.rows if row.src == run.chunk_files[name])
                      for name in names}
            return {"per_shard": totals, "notes": [SPAN_NOTE]}
        if listed == "c03.pairs.split":
            return self._split_pairs_reading(question, run)
        blocked = self._tie_or_cannot(question, run, names)
        if blocked:
            return blocked
        by_src = self._by_src(run.rows)
        if listed == "c03.spans.per-ordinal":
            return {"per_ordinal": {name: self._per_ordinal(run, by_src, name, lambda r: r.end - r.start)
                                    for name in names},
                    "notes": [SPAN_NOTE]}
        return {"per_ordinal": {name: self._per_ordinal(run, by_src, name, lambda r: r.id) for name in names}}

    def _split_pairs_of(self, question) -> list[dict]:
        """The split pairs of the set the question is asked of: the list itself where the subject was given one list, else the
        list of the question's target (``ValueError`` where the mapping holds none: a missing list is not an empty one)."""
        if not isinstance(self._split_pairs, Mapping):
            return self._split_pairs
        target = tuple(record.attempt_target(question))
        if target not in self._split_pairs:
            raise ValueError(f"no split pairs were given for {target[0]} {target[1]}")
        return self._split_pairs[target]

    def _split_pairs_reading(self, question, run: DolmaRun) -> dict:
        """For each member of each split pair, the chunk file and the line of the tied row. Beside the reading, ungraded (the table
        compares ``record`` alone; the note goes to ``note_fault``, the hook of the line's ``notes``), the note says whether the id the tool's own row gives at each listed position is the id the
        plant list names for that member: by the table's tie the line is the listed ordinal plus one whenever a row exists, so
        this is what the tool itself says about the record at that place."""
        listed = self._split_pairs_of(question)
        used = sorted({member["shard"] for pair in listed for member in pair["members"]})
        blocked = self._tie_or_cannot(question, run, used)
        if blocked:
            return blocked
        by_src = self._by_src(run.rows)
        shard_of = {chunk: name for name, chunk in run.chunk_files.items()}
        pairs, differing, same, total = [], [], 0, 0
        for pair in listed:
            members = []
            for member in pair["members"]:
                found = self._tied_row(by_src.get(self._chunk(run, member["shard"]), {}), member["ordinal"])
                members.append({"shard": member["shard"], "line": None} if found is None
                               else {"shard": shard_of[found.src], "line": found.loc})
                if member.get("id") is not None:
                    total += 1
                    if found is not None and found.id == member["id"]:
                        same += 1
                    elif len(differing) < 5:
                        differing.append({"pair": pair["pair"], "shard": member["shard"], "ordinal": member["ordinal"],
                                          "tool_id": None if found is None else found.id, "listed_id": member["id"]})
            pairs.append({"pair": pair["pair"], "members": members})
        if self.note_fault is not None:
            # The hook that puts a note on the line of the question (the runner sets it); the reading itself stays ``record`` alone.
            self.note_fault({"tool_id_at_each_listed_position_is_the_listed_member": {
                "members_with_a_listed_id": total, "equal": same, "first_differing": differing}})
        return {"record": pairs}

    def _counts(self, run: DolmaRun, by_src, name: str) -> tuple[int, int, int]:
        """The documents of a shard, those with exactly one tied row and those with none."""
        chunk = self._chunk(run, name)
        held = by_src.get(chunk, {})
        lines = _line_count(Path(run.build) / chunk)
        one = sum(1 for ordinal in range(lines) if len(held.get(ordinal + 1, [])) == 1)
        none = sum(1 for ordinal in range(lines) if not held.get(ordinal + 1))
        return lines, one, none

    def _generated_documents(self, question, run: DolmaRun) -> dict:
        names = list(run.chunk_files)
        blocked = self._tie_or_cannot(question, run, names)
        if blocked:
            return blocked
        by_src = self._by_src(run.rows)
        counts = {name: self._counts(run, by_src, name) for name in names}
        return {
            "per_ordinal": {name: self._per_ordinal(run, by_src, name, lambda r: r.id) for name in names},
            "unmapped_documents": sum(none for _, _, none in counts.values()),
            "documents_per_shard": {name: documents for name, (documents, _, _) in counts.items()},
            "documents_with_one_tied_row_per_shard": {name: one for name, (_, one, _) in counts.items()},
            "documents_with_no_tied_row_per_shard": {name: none for name, (_, _, none) in counts.items()},
        }

    def _fixture_documents(self, question, run: DolmaRun, artifact: str) -> dict:
        blocked = self._tie_or_cannot(question, run, [artifact])
        if blocked:
            return blocked
        documents, one, none = self._counts(run, self._by_src(run.rows), artifact)
        return {"documents": documents, "recovered": one, "unknown": none}

    # -- the two writer states of Check 4 --

    @staticmethod
    def _failed_second_run(run: DolmaRun) -> Attempt:
        """The failed attempt of a second run that did not refuse: a crash, a signal, no
        output, or an end the last line of its output does not say. Neither its command nor
        its output holds a local path."""
        if not run.retry_attempt_id:
            raise ValueError("the second run failed and the run holds no id for its attempt")
        return Attempt(id=run.retry_attempt_id,
                       command=tuple(_without_local_paths(str(word)) for word in run.retry_command),
                       exit_code=run.retry_exit_code,
                       output=_without_local_paths(run.retry_output or ""))

    @staticmethod
    def _stop_that_did_not_land(run: DolmaRun) -> Attempt | None:
        """The failed attempt of the stopped writer when its exit is not the exit of the signal
        that was sent to it (a negative code, or 128 plus the signal's number); the writer
        then ended on its own or by another signal, so no fault was put. Else None. The
        attempt records the writer's exit, and holds no local path. A run that gives no exit
        or no signal (or a value that is no whole number, or a signal outside 1 to 64), or no
        id for a failed attempt, is incomplete: ``ValueError``."""
        exit_code, sent = run.stopped_exit_code, run.stopped_signal
        if not _is_whole_number(exit_code):
            raise ValueError("the run holds no exit of the stopped writer")
        if not _is_whole_number(sent) or not 1 <= sent <= _LAST_SIGNAL:
            raise ValueError("the run holds no signal that stopped the writer")
        if exit_code in (-sent, _SIGNAL_EXIT_BASE + sent):
            return None
        if not run.stopped_attempt_id:
            raise ValueError("the stop did not land and the run holds no id for its attempt")
        text = _without_local_paths(run.stopped_output or "")
        note = "the stop did not land: the exit of the writer is not that of the signal sent to it"
        return Attempt(id=run.stopped_attempt_id,
                       command=tuple(_without_local_paths(str(word)) for word in run.stopped_command),
                       exit_code=exit_code, output=(text + "\n" + note).strip())

    def _writer_state(self, run: DolmaRun, listed: str, artifact: str) -> dict | Attempt:
        chunk = self._chunk(run, artifact)
        not_landed = self._stop_that_did_not_land(run)
        if not_landed is not None:
            return not_landed
        if listed == "c04.kill-after-100":
            if run.stopped_rows is None:
                raise ValueError("the run holds no rows after the stopped write")
            showed = run.rows_showed_while_running
            if not isinstance(showed, bool):
                raise ValueError("the run does not say whether the rows showed while the writer ran")
            return {"records": {"rows_after_stop": sum(1 for row in run.stopped_rows if row.src == chunk),
                                "rows_showed": ROWS_SHOWED_WHILE_RUNNING if showed else ROWS_SHOWED_AT_END}}
        if run.retry_rows is None or run.retry_exit_code is None:
            raise ValueError("the run holds no second run with an exit code")
        rows = [row for row in run.retry_rows if row.src == chunk]
        refused, words = False, ""
        if run.retry_exit_code != 0:
            said = _refusal_text(run.retry_exit_code, run.retry_output)
            if said is None:
                return self._failed_second_run(run)
            refused, words = True, said
        matches = None
        if rows and run.token_file_tokens is not None:
            # The ranges index the documents of the token file when they follow one another in
            # the token file (the first at 0, no gap, no overlap) and end within it, each line
            # has at most one row and none names a line the file does not have; a map of every
            # document ends at the last token. The order of the rows by line is not asked.
            documents = _line_count(Path(run.build) / chunk)
            position, chained = 0, True
            for row in sorted(rows, key=lambda r: (r.start, r.end)):
                if row.start != position:
                    chained = False
                position = row.end
            locs = [row.loc for row in rows]
            once = len(set(locs)) == len(locs) and all(1 <= loc <= documents for loc in locs)
            whole = len(rows) == documents
            matches = (chained and once and position <= run.token_file_tokens
                       and (position == run.token_file_tokens or not whole))
        return {"records": {"rows_after_retry": len(rows), "matches_shard": matches, "refused": refused,
                            "mismatch": words if refused and words else None}}
