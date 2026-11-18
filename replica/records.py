"""One line of a JSONL file as the replica carries it from step to step.

A record keeps the id of its row, its text, the line as it was read, and the file
and line the next step reads it from. A line that is not a readable record (blank,
not JSON, no text, no id) is carried too: the copy steps pass it on byte for byte
and the sanitizer skips and logs it, as the published pipeline does.

Two ways give a line its id. In the hand-checked fixture the id is the row's own
``id`` field. In a generated set an id names a line: the set lists the id of every
line of a file, also of a line that does not parse (docs/plant.md 4.1), and a row
that carries another id than its line's is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .errors import InputError

FAULT_BLANK = "blank line"
FAULT_NOT_JSON = "not JSON"
FAULT_NOT_OBJECT = "not a JSON object"
FAULT_NO_TEXT = "no text field"
FAULT_NO_ID = "no id field"


@dataclass(frozen=True)
class Record:
    """A line of a JSONL file.

    ``ref`` is the line's id; a line with no id gets ``<file>#L<line>`` of the file
    it was first read from. ``text`` is None when the line is not a readable
    record, and ``fault`` then says why. ``artifact`` and ``line`` name the place
    the next step reads the record from: the path of a JSONL file relative to the
    build folder and the line there, counted from 1; ``artifact`` is None for a
    line that came from no file (the fixture's skipped sanitizer line, which the
    spec writes into one chunk). ``source`` is the name of the source the record's
    file belongs to, None for a generation output. ``has_id`` is False when ``ref``
    is the made-up key of a line with no id."""

    ref: str
    text: str | None
    raw: str
    artifact: str | None
    line: int
    source: str | None
    fault: str | None = None
    has_id: bool = True

    @property
    def readable(self) -> bool:
        return self.fault is None

    def at(self, artifact: str, line: int) -> "Record":
        """The same record at the place a step has just written it to."""
        return replace(self, artifact=artifact, line=line)

    def with_text(self, text: str) -> "Record":
        return replace(self, text=text)


def parse_line(raw: str, *, artifact: str | None, line: int, source: str | None,
               ref: str | None = None) -> Record:
    """Read one line. A readable record is a JSON object with a string ``id`` and a
    string ``text``; anything else is carried as an unreadable line.

    ``ref`` is the id the line has by its place, when the input names its lines
    (a generated set). The line then has that id whatever it holds; a row without
    an ``id`` field is readable, and a row whose ``id`` differs is refused."""
    key = f"{artifact}#L{line}"

    def unreadable(fault: str, row_id: str | None = None) -> Record:
        known = ref or row_id
        return Record(ref=known or key, text=None, raw=raw, artifact=artifact, line=line,
                      source=source, fault=fault, has_id=known is not None)

    if raw.strip() == "":
        return unreadable(FAULT_BLANK)
    try:
        row = json.loads(raw)
    except ValueError:
        return unreadable(FAULT_NOT_JSON)
    if not isinstance(row, dict):
        return unreadable(FAULT_NOT_OBJECT)
    row_id = row.get("id")
    text = row.get("text")
    has_id = isinstance(row_id, str) and row_id != ""
    if ref is not None and "id" in row and row_id != ref:
        raise InputError(f"{artifact}: line {line} carries the id {row_id!r}; the id of "
                         f"this line is {ref!r}")
    if not isinstance(text, str):
        return unreadable(FAULT_NO_TEXT, row_id if has_id else None)
    if not has_id and ref is None:
        return unreadable(FAULT_NO_ID)
    return Record(ref=ref or row_id, text=text, raw=raw, artifact=artifact, line=line,
                  source=source)


def split_lines(data: str) -> list[str]:
    """The lines of a JSONL file. Only a line feed ends a line, so a text that holds
    another line-breaking character stays on its line."""
    if data == "":
        return []
    lines = data.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def read_records(path: Path, *, artifact: str, source: str | None, strict: bool,
                 refs: tuple[str, ...] | None = None) -> list[Record]:
    """Read a JSONL file into records, in line order.

    ``strict`` is for the hand-checked fixture, where every line must be a readable
    record. Without it an unreadable line is carried as it is. ``refs`` gives the
    id of every line by its place (a generated set); the file must then hold
    exactly that many lines."""
    try:
        data = Path(path).read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(f"{path}: not UTF-8 ({exc})") from exc
    lines = split_lines(data)
    if refs is not None and len(refs) != len(lines):
        raise InputError(f"{path}: {len(lines)} lines, and {len(refs)} ids are listed for it")
    records = []
    for number, raw in enumerate(lines, 1):
        record = parse_line(raw, artifact=artifact, line=number, source=source,
                            ref=None if refs is None else refs[number - 1])
        if strict and not record.readable:
            raise InputError(f"{path}: line {number}: {record.fault}")
        records.append(record)
    return records
