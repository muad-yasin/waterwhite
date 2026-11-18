"""The pure steps of the replica: no files, no clock, no subprocess.

Each function takes lists and returns lists, so a test can check one step alone.
The steps follow the published pipeline the plan describes (plan Section 13, the
heading on the published pipeline; plan 5.2): shuffle each listed file, deal it
round-robin over N chunks, merge, shuffle each chunk, sanitize with logged skips.
The exact dedup is the fixture's addition (plan 5.1).
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .errors import SpecError
from .records import FAULT_BLANK, Record

try:
    # The one import of the tool that every mode makes (DECISIONS.md D-17): the
    # normaliser is defined once, in the tool, and the sanitizer calls it by name.
    from waterwhite import normalise as tool_normalise
except ModuleNotFoundError as exc:  # a wrong set-up, said in words
    raise ModuleNotFoundError(
        "the replica's sanitizer takes its normaliser from waterwhite.normalise "
        "(DECISIONS.md D-17) and the package waterwhite is not importable: put the "
        "repository's src folder on the Python path (PYTHONPATH=src) or install the package"
    ) from exc

T = TypeVar("T")

# --- Id ranges ---------------------------------------------------------------------

_ID_WITH_NUMBER = re.compile(r"^(?P<prefix>.*?)(?P<number>\d+)$")


def expand_ids(items: Iterable[str]) -> list[str]:
    """Expand ``"Z-001..Z-020"`` into every id from the first to the last.

    Both ends carry the same prefix and the same number of digits, and the first is
    not after the last. An item with no ``..`` is one id and is passed through."""
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or item == "":
            raise SpecError(f"not a record id: {item!r}")
        if ".." not in item:
            out.append(item)
            continue
        first, sep, last = item.partition("..")
        a = _ID_WITH_NUMBER.match(first)
        b = _ID_WITH_NUMBER.match(last)
        if not sep or a is None or b is None:
            raise SpecError(f"not an id range: {item!r}")
        if a["prefix"] != b["prefix"] or len(a["number"]) != len(b["number"]):
            raise SpecError(f"the two ends of a range must share prefix and width: {item!r}")
        lo, hi = int(a["number"]), int(b["number"])
        if lo > hi:
            raise SpecError(f"a range must run upwards: {item!r}")
        width = len(a["number"])
        out.extend(f"{a['prefix']}{n:0{width}d}" for n in range(lo, hi + 1))
    return out


# --- Shuffles ----------------------------------------------------------------------


def _fisher_yates(items: Sequence[T], below: Callable[[int], int]) -> list[T]:
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = below(i + 1)
        out[i], out[j] = out[j], out[i]
    return out


class SeededShuffler:
    """Seeded mode: the same seed and the same label give the same order, on any
    machine and under any Python version.

    The draws come from SHA-256 over ``(seed, label, counter)``, taken eight bytes at
    a time, with rejection so that no position is favoured. Each shuffle of a build
    has its own label, so adding a shuffle elsewhere never moves this one."""

    DOMAIN = "waterwhite.replica.shuffle/1"
    seeded = True

    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SpecError(f"a shuffle seed is an integer, got {seed!r}")
        self.seed = seed

    def _draws(self, label: str):
        counter = 0
        while True:
            block = hashlib.sha256(
                f"{self.DOMAIN}|{self.seed}|{label}|{counter}".encode("utf-8")
            ).digest()
            for offset in range(0, 32, 8):
                yield int.from_bytes(block[offset:offset + 8], "big")
            counter += 1

    def shuffle(self, items: Sequence[T], label: str) -> list[T]:
        draws = self._draws(label)

        def below(n: int) -> int:
            limit = (1 << 64) - ((1 << 64) % n)
            while True:
                value = next(draws)
                if value < limit:
                    return value % n

        return _fisher_yates(items, below)


class UnseededShuffler:
    """Unseeded mode, on purpose: the order comes from the operating system's
    randomness, as the published pipeline's ``shuf`` with no seed does. Two builds
    differ, and nothing about the order is recorded."""

    seeded = False
    seed = None

    def __init__(self) -> None:
        self._random = random.SystemRandom()

    def shuffle(self, items: Sequence[T], label: str) -> list[T]:
        return _fisher_yates(items, self._random.randrange)


# --- Round-robin chunks and partitioned shards ---------------------------------------


def deal_round_robin(items: Sequence[T], chunks: int) -> list[list[T]]:
    """Deal items over ``chunks`` piles: item i goes to pile i modulo ``chunks``."""
    if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks < 1:
        raise SpecError(f"the chunk count is a positive integer, got {chunks!r}")
    piles: list[list[T]] = [[] for _ in range(chunks)]
    for position, item in enumerate(items):
        piles[position % chunks].append(item)
    return piles


def chunk_names(chunks: int) -> list[str]:
    """``shard-1`` to ``shard-N``, zero-padded to the width of N so that a sorted
    listing is in chunk order."""
    width = len(str(chunks))
    return [f"shard-{n:0{width}d}" for n in range(1, chunks + 1)]


def round_robin_chunks(
    files: Sequence[tuple[str, Sequence[T]]], chunks: int, shuffler
) -> list[tuple[str, list[T]]]:
    """The round-robin layout: shuffle each listed file, deal it over the chunks
    starting at the first chunk, merge the piles of all files chunk by chunk in
    listing order, then shuffle each chunk."""
    merged: list[list[T]] = [[] for _ in range(chunks)]
    for name, items in files:
        shuffled = shuffler.shuffle(items, f"file/{name}")
        for pile, part in zip(merged, deal_round_robin(shuffled, chunks)):
            pile.extend(part)
    names = chunk_names(chunks)
    return [(name, shuffler.shuffle(pile, f"chunk/{name}")) for name, pile in zip(names, merged)]


def partitioned_shards(
    shards: Sequence[tuple[str, Sequence[T]]], shuffler
) -> list[tuple[str, list[T]]]:
    """The partitioned layout: each shard holds the records it is given and nothing
    else; the records of one shard are merged and shuffled once."""
    return [(name, shuffler.shuffle(items, f"shard/{name}")) for name, items in shards]


# --- The normaliser ----------------------------------------------------------------


def normaliser(name: str) -> Callable[[str], str]:
    """The function registered under ``name`` in ``waterwhite.normalise``.

    The replica holds no normaliser of its own. Its sanitizer and its dedup
    comparison call the tool's function, so a sanitized record and the fingerprint
    the tool takes of the record it was made from agree (plan 3.1, F8;
    DECISIONS.md D-17)."""
    try:
        return tool_normalise.get(name)
    except tool_normalise.UnknownNormaliser:
        raise SpecError(f"unknown normaliser: {name!r}; registered: "
                        f"{', '.join(tool_normalise.names())}") from None


# --- The sanitizer -----------------------------------------------------------------

# The two reasons a sanitizer skip carries. They are spelled as the tool's drop
# reasons are (a test compares them), so a reader of the log needs no second table.
REASON_SANITIZER_SKIP = "sanitizer skip"
REASON_EMPTY_LINE = "empty line"
DETAIL_EMPTY_AFTER = "empty after sanitizing"


@dataclass(frozen=True)
class Skip:
    """A line the sanitizer skipped. ``position`` is its place in the chunk as the
    sanitizer read it, counted from 1."""

    position: int
    record: Record
    reason: str
    detail: str


def sanitize(
    records: Sequence[Record], normalise: Callable[[str], str]
) -> tuple[list[Record], list[Skip]]:
    """Sanitize one chunk. A surviving record carries its sanitized text; a line
    that is unreadable or empty after sanitizing is skipped and returned as a skip.
    The order of the survivors is the order they came in."""
    kept: list[Record] = []
    skips: list[Skip] = []
    for position, record in enumerate(records, 1):
        if not record.readable:
            reason = REASON_EMPTY_LINE if record.fault == FAULT_BLANK else REASON_SANITIZER_SKIP
            skips.append(Skip(position, record, reason, record.fault or ""))
            continue
        clean = normalise(record.text or "")
        if clean == "":
            skips.append(Skip(position, record, REASON_EMPTY_LINE, DETAIL_EMPTY_AFTER))
            continue
        kept.append(record.with_text(clean))
    return kept, skips


# --- Exact dedup ---------------------------------------------------------------------

RULE_OFF = "dedup off"
RULE_SINGLE = "no duplicate"
RULE_WINNER = "winner source"
RULE_FIRST = "first copy"


@dataclass(frozen=True)
class Kept:
    """One line a release keeps. ``copies`` holds every equivalent copy in input
    order, the kept one among them; a record with no duplicate has one copy."""

    record: Record
    copies: tuple[Record, ...]
    rule: str

    @property
    def losers(self) -> tuple[Record, ...]:
        return tuple(c for c in self.copies if c is not self.record)


def dedup(
    records: Sequence[Record],
    normalise: Callable[[str], str],
    *,
    enabled: bool,
    priority: Sequence[str] = (),
) -> list[Kept]:
    """Exact dedup on the normalised text.

    Records whose normalised texts are equal form a group. The group keeps one line,
    at the place of its first copy, so the release has the same lines whichever copy
    wins. ``priority`` lists source names: the kept copy is the first copy from the
    source that stands first in that list, and a group with no copy from a listed
    source keeps its first copy. A line that is unreadable or empty after
    normalising has no text to compare and is never grouped: it stays a line of its
    own until the sanitizer drops it.

    With ``enabled`` False every record is kept as it is."""
    if not enabled:
        return [Kept(r, (r,), RULE_OFF) for r in records]
    rank = {source: position for position, source in enumerate(priority)}
    order: list[object] = []
    groups: dict[object, list[Record]] = {}
    for position, record in enumerate(records):
        text = normalise(record.text) if record.readable and record.text is not None else ""
        key: object = ("text", text) if text != "" else ("alone", position)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(record)
    kept: list[Kept] = []
    for key in order:
        copies = tuple(groups[key])
        if len(copies) == 1:
            kept.append(Kept(copies[0], copies, RULE_SINGLE))
            continue
        ranked = [c for c in copies if c.source in rank]
        if ranked:
            # min returns the first of equals: two copies from one source keep the
            # earlier one.
            kept.append(Kept(min(ranked, key=lambda c: rank[c.source]), copies, RULE_WINNER))
        else:
            kept.append(Kept(copies[0], copies, RULE_FIRST))
    return kept
