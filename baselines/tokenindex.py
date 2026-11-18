"""OriginBlame's token-index entries and forget set (plan 5.5 Check 10), read from files,
and the tie from a document's token range to an entry.

An entry has ``sources`` (section hashes), ``token_count``, ``tokenizer`` and
``revoked``. It has no id, no offset and no line hash. The entries of a shard are kept
in the order the documents were tracked.

The forget set is ``ceil(N / 8)`` bytes, one bit per entry in entry order, least
significant bit first, 1 for revoked: entry 0 is bit 0 of byte 0, entry 8 is bit 0 of
byte 1.

The token range of an entry is the running sum of the counts of the entries before it,
up to that sum plus its own count. A document is tied to the entry whose range equals
its own. That holds only where one process tracked the documents of the shard in
packed order and the run says so; without that the tie is refused, and what is missing
is an id or an offset on the entry.

The reader never reads a gap as an absence: a forget set of the wrong length and an
entry without a count are errors. The module imports nothing of the tool.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class TieNotValid(ValueError):
    """The tie by token range was asked for where the run does not give one process in
    packed order."""


@dataclass(frozen=True)
class Entry:
    sources: tuple[str, ...]
    token_count: int
    tokenizer: str
    revoked: bool


def _entry(record: object, line: int) -> Entry:
    if not isinstance(record, dict):
        raise ValueError(f"entry {line}: not a JSON object")
    for name in ("sources", "token_count", "tokenizer", "revoked"):
        if name not in record:
            raise ValueError(f"entry {line}: no {name}")
    sources, count = record["sources"], record["token_count"]
    tokenizer, revoked = record["tokenizer"], record["revoked"]
    if not isinstance(sources, list) or not all(isinstance(s, str) for s in sources):
        raise ValueError(f"entry {line}: sources is not a list of text")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(f"entry {line}: token_count is not a whole number: {count!r}")
    if not isinstance(tokenizer, str):
        raise ValueError(f"entry {line}: tokenizer is not text")
    if not isinstance(revoked, bool):
        raise ValueError(f"entry {line}: revoked is not true or false")
    return Entry(tuple(sources), count, tokenizer, revoked)


def read_entries(path: str | Path) -> list[Entry]:
    """The entries of one JSON-lines file, in file order."""
    entries: list[Entry] = []
    with open(path, "r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            entries.append(_entry(json.loads(line), number))
    return entries


def forget_bits(data: bytes, count: int) -> list[bool]:
    """``count`` booleans from the bytes of a forget set, least significant bit first.
    An error where ``len(data)`` is not ``ceil(count / 8)``."""
    if count < 0:
        raise ValueError(f"count is negative: {count}")
    need = (count + 7) // 8
    if len(data) != need:
        raise ValueError(f"a forget set of {count} entries is {need} bytes, got {len(data)}")
    return [bool((data[i >> 3] >> (i & 7)) & 1) for i in range(count)]


def read_forget_set(path: str | Path, count: int) -> list[bool]:
    """The same, from a file."""
    return forget_bits(Path(path).read_bytes(), count)


def token_ranges(entries: Sequence[Entry]) -> list[tuple[int, int]]:
    """``(start, end)`` of every entry by running sum, end excluded, from 0."""
    ranges: list[tuple[int, int]] = []
    total = 0
    for entry in entries:
        ranges.append((total, total + entry.token_count))
        total += entry.token_count
    return ranges


def tie_by_range(entries: Sequence[Entry], start: int, end: int, *,
                 one_process_packed: bool) -> int | None:
    """The place of the entry whose range is ``(start, end)``, or None where none is.
    Refused (TieNotValid) unless the run says one process tracked the shard in packed
    order."""
    if not one_process_packed:
        raise TieNotValid(
            "the token-range tie needs one process in packed order; without it the "
            "entry lacks an id or an offset to tie by")
    found = [i for i, r in enumerate(token_ranges(entries)) if r == (start, end)]
    if len(found) > 1:
        raise ValueError(f"{len(found)} entries hold the range ({start}, {end})")
    return found[0] if found else None
