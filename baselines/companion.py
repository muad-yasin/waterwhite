"""The Dolma companion rows (plan 5.5 Check 10): the records the tokenizer writes beside
its output, read from a file, and the two lookups the baseline needs from them.

A companion file has no header row. Each line is one row of five columns in this
order: ``start``, ``end``, ``id``, ``src``, ``loc``. ``src`` is the input path as it
was given to the tokenizer and ``loc`` is the line number in that file, counted from
1. The file is plain or gzip. The order of the rows of a run is not the order of the
input, so a row is tied to a document by its ``src`` and ``loc`` and never by its
place in the file.

The reader never reads a gap as an absence: a row of the wrong shape is an error, and
a document with no row has no tie (None), never the row next to it. The module holds
words, files and plain data only; it imports nothing of the tool.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

COLUMNS = ("start", "end", "id", "src", "loc")

# The first two bytes of a gzip file.
_GZIP_MAGIC = b"\x1f\x8b"

# The folders of a build that hold the record files of its documents, and the kind of
# file each one holds.
SOURCES_FOLDER = "sources"
GEN_FOLDER = "gen"
KIND_SOURCE = "source"
KIND_GENERATION = "generation"

# Files in those folders that are not a build file's own records: the line-by-line
# manifest beside each record file, and the harness logs.
_MANIFEST_SUFFIX = ".manifest.jsonl"
_HARNESS_FOLDER = "harness"


@dataclass(frozen=True)
class Row:
    """One companion row: the token range of a document in the tokenizer's output,
    its id, the input file it came from and its line in that file (from 1)."""

    start: int
    end: int
    id: str
    src: str
    loc: int


def _whole_number(text: str, column: str, line: int) -> int:
    # int() would also take a sign, spaces and underscores; the file holds digits only.
    if not (text.isascii() and text.isdigit()):
        raise ValueError(f"companion row {line}: {column} is not an integer: {text!r}")
    return int(text)


def _open_text(path: Path) -> io.TextIOBase:
    with open(path, "rb") as f:
        packed = f.read(2) == _GZIP_MAGIC
    if packed:
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def read_rows(path: str | Path) -> list[Row]:
    """The rows of one companion file in file order."""
    rows: list[Row] = []
    with _open_text(Path(path)) as f:
        for line, fields in enumerate(csv.reader(f), start=1):
            if len(fields) != len(COLUMNS):
                raise ValueError(
                    f"companion row {line}: {len(fields)} fields, expected {len(COLUMNS)}")
            start, end, document_id, src, loc = fields
            loc_number = _whole_number(loc, "loc", line)
            if loc_number < 1:
                raise ValueError(f"companion row {line}: loc counts from 1, got {loc_number}")
            rows.append(Row(
                start=_whole_number(start, "start", line),
                end=_whole_number(end, "end", line),
                id=document_id,
                src=src,
                loc=loc_number,
            ))
    return rows


def rows_for_src(rows: Iterable[Row], src: str) -> list[Row]:
    """The rows whose ``src`` equals ``src`` whole, in the order given."""
    return [r for r in rows if r.src == src]


def tie_row(rows: Sequence[Row], src: str, ordinal: int) -> Row | None:
    """The row of the document with this ordinal in the chunk file ``src``: the row
    whose ``src`` is that file and whose ``loc`` is ``ordinal + 1``. None where no row
    has them; an error where two do."""
    found = [r for r in rows if r.src == src and r.loc == ordinal + 1]
    if len(found) > 1:
        raise ValueError(f"{len(found)} companion rows hold src {src!r} and loc {ordinal + 1}")
    return found[0] if found else None


def _record_files(folder: Path) -> list[Path]:
    """The record files of a build, in a fixed order: the ``.jsonl`` files under
    ``sources/`` and ``gen/``, without the manifests beside them and the harness logs."""
    files: list[Path] = []
    for top in (SOURCES_FOLDER, GEN_FOLDER):
        base = folder / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.jsonl")):
            relative = path.relative_to(folder)
            if path.name.endswith(_MANIFEST_SUFFIX):
                continue
            if top == GEN_FOLDER and len(relative.parts) > 2 and relative.parts[1] == _HARNESS_FOLDER:
                continue
            if path.is_file():
                files.append(path)
    return files


def _holds_id(path: Path, document_id: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path.name} line {number}: a record is not an object")
            if record.get("id") == document_id:
                return True
    return False


def resolve_id(folder: str | Path, document_id: str) -> str | None:
    """The path, relative to the build folder and written with ``/``, of the file
    under ``sources/`` or ``gen/`` that holds a row with this id; None where no file
    does. An error where two files do."""
    if not document_id:
        return None
    base = Path(folder)
    holders = [p.relative_to(base).as_posix() for p in _record_files(base)
               if _holds_id(p, document_id)]
    if len(holders) > 1:
        raise ValueError(f"id {document_id!r} is held by {len(holders)} files: {holders}")
    return holders[0] if holders else None


def kind_of(relative_path: str) -> str:
    """"source" for a file under ``sources/`` (the file of a member), "generation"
    for a file under ``gen/``."""
    top = relative_path.split("/", 1)[0]
    if top == SOURCES_FOLDER:
        return KIND_SOURCE
    if top == GEN_FOLDER:
        return KIND_GENERATION
    raise ValueError(f"{relative_path!r} is under neither {SOURCES_FOLDER}/ nor {GEN_FOLDER}/")
