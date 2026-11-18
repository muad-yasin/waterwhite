"""Reading a sanitizer's own log of the lines it skipped (plan 3.3 "Deterministic
rewrite").

The log is JSON lines, one row per skipped line of a sanitizer-input file:

    {"input": "pipeline/partitioned/sanitizer-input/shard-2.jsonl", "line": 6,
     "reason": "sanitizer skip", ...}

- ``line``: the line of the sanitizer's input file, counted from 1;
- ``reason``: "empty line" or "sanitizer skip", spelled as the tool's drop reasons;
- ``input``: which input file the row is about, when one log covers several
  files. A log of one file may leave the field out of every row.

Other fields are not read. A row the reader cannot use is refused with the line of
the log it stands on; nothing is skipped and nothing is guessed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .. import ids
from .. import vocabulary as v
from ..errors import InvalidRecord

# The reasons a sanitizer logs (plan 3.1 "Drop", 3.3): the vocabulary's, the same
# tuple a map recovered through a sanitizer's log reads its reasons by.
SANITIZER_REASONS = v.SANITIZER_DROP_REASONS


@dataclass(frozen=True)
class LoggedSkip:
    """One line a sanitizer skipped: its line in the sanitizer's input, and why."""

    line: int
    reason: str


def read_sanitizer_log(
    path: str | os.PathLike, *, input: str | None = None
) -> tuple[LoggedSkip, ...]:
    """The skipped lines of one sanitizer-input file, in line order.

    ``input`` is the value the log's rows carry in their ``input`` field for the
    file in question; rows about other files are passed over. With ``input`` None
    the log must be the log of one file: a row that carries an ``input`` field is
    then refused, because the reader will not guess which file a row is about.
    A line logged twice with one reason is one skip; with two reasons it is
    refused."""
    if input is not None and (not isinstance(input, str) or not input):
        raise InvalidRecord("the name of the log's input file is a text")
    where = os.fspath(path)
    if not os.path.isfile(where):
        raise InvalidRecord(f"no sanitizer log at {where}")
    found: dict[int, str] = {}
    for number, raw in enumerate(ids.read_lines(where), start=1):
        at = f"{where}, line {number}"
        try:
            row = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            raise InvalidRecord(f"{at}: not a row of a sanitizer log (not JSON)") from None
        if not isinstance(row, dict):
            raise InvalidRecord(f"{at}: not a row of a sanitizer log (not a JSON object)")
        named = row.get("input")
        if input is None:
            if named is not None:
                raise InvalidRecord(
                    f"{at}: the row is about the input file {named!r}; name the input "
                    "file this recovery is about, a row is not assigned to a file by guess"
                )
        else:
            if not isinstance(named, str):
                raise InvalidRecord(
                    f"{at}: the row names no input file, and this recovery asks for the "
                    f"rows about {input!r}"
                )
            if named != input:
                continue
        line = row.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise InvalidRecord(f"{at}: the skipped line is a whole number from 1, got {line!r}")
        reason = row.get("reason")
        if reason not in SANITIZER_REASONS:
            raise InvalidRecord(
                f"{at}: the reason of a skipped line is one of "
                f"{', '.join(repr(r) for r in SANITIZER_REASONS)}, got {reason!r}"
            )
        held = found.get(line)
        if held is not None and held != reason:
            raise InvalidRecord(
                f"{at}: line {line} is logged twice, as {held!r} and as {reason!r}"
            )
        found[line] = reason
    return tuple(LoggedSkip(line, found[line]) for line in sorted(found))
