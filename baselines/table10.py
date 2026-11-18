"""The loader of ``fixture/table/check10.toml`` and the expander from its groups to
the questions of Check 10 (plan 5.5 Check 10).

A question is a listed row, a subject and the size at which the tool's own row is
asked. The sizes are the tool's own: for a listed row of Checks 1 to 9 that the
acceptance reader (``tests/acceptance/table.py``) reads as sized, ``small`` and then
the sizes of its set; every other listed row is asked once (size None), among them
a row that names a set but is not read as sized, and a row the reader cannot read.
The loader works the size list out once, from the check files that sit beside
``check10.toml``.

The file is read as it stands: every text is kept word for word, a field a row does
not hold is None, and ``marks`` and ``conditions`` are tuples. A key every row must
hold that is missing raises ``KeyError``; nothing is defaulted. The mapping fields
(``expect``, ``ordinal``, ``tie``) are read-only mappings.

The mark of the src/loc tie is held in ``tie_mark`` and never merged into
``marks``: an answer carries it only when it used the tie.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from baselines.interfaces import SIZE_SMALL, Question

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO / "fixture" / "table" / "check10.toml"
ACCEPTANCE_DIR = REPO / "tests" / "acceptance"

# The checks whose rows are listed by the groups.
LISTED_CHECKS = range(1, 10)

# The fallbacks of the table README that are in force when the sizes are read: none.
NO_FALLBACKS: tuple[str, ...] = ()

# One question is one listed row put to one subject at one size; its id joins the
# three: the row and the subject, and the size after the second separator when the
# row is asked at a size.
QUESTION_ID_SEPARATOR = "|"
SIZE_SEPARATOR = "@"


def _acceptance_table():
    """The table reader of the acceptance tests, the module the tool's own rows are
    read with. It imports nothing of the tool; its folder goes on the path for it."""
    folder = str(ACCEPTANCE_DIR)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import table  # noqa: PLC0415

    return table


def _frozen(value: Any) -> Any:
    """A loaded value with its tables read-only and its arrays as tuples."""
    if isinstance(value, dict):
        return MappingProxyType({k: _frozen(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    return value


@dataclass(frozen=True)
class Group:
    """A set of rows of Checks 1 to 9 that ask the same question."""

    id: str
    check: int
    question: str
    rows: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class Row:
    """One subject's reading rule for one group.

    A row holds read-only mappings and so is not hashable; rows are keyed by id.
    """

    id: str
    group: str
    subject: str
    question: str
    plan: str
    core: bool
    record: str
    route: str | None
    missing: str | None
    rule: str
    marks: tuple[str, ...]
    tie_mark: str | None
    ordinal: Mapping[str, Any] | None
    tie: Mapping[str, Any] | None
    needs_ordinal: bool
    excluded_from_tally: bool
    fault: str | None
    conditions: tuple[str, ...]
    expect: Mapping[str, Any]
    source: str
    extension: str | None = None   # the extension a row belongs to (core = false), else None


@dataclass(frozen=True)
class Table10:
    """The table: groups and rows, each in file order, and the sizes at which each
    listed row of Checks 1 to 9 is asked (``small`` first, then the plan sizes; the
    single entry None for a row asked once)."""

    groups: tuple[Group, ...]
    rows: tuple[Row, ...]
    sizes: Mapping[str, tuple[str | None, ...]]


def _group(raw: Mapping[str, Any]) -> Group:
    return Group(
        id=raw["id"],
        check=raw["check"],
        question=raw["question"],
        rows=tuple(raw["rows"]),
        source=raw["source"],
    )


def _row(raw: Mapping[str, Any]) -> Row:
    given = raw["given"]
    ordinal = raw.get("ordinal")
    tie = raw.get("tie")
    return Row(
        id=raw["id"],
        group=given["group"],
        subject=given["subject"],
        question=raw["question"],
        plan=raw["plan"],
        core=raw["core"],
        record=raw["record"],
        route=raw.get("route"),
        missing=raw.get("missing"),
        rule=raw["rule"],
        marks=tuple(raw["marks"]),
        tie_mark=ordinal["shown"]["mark"] if ordinal is not None else None,
        ordinal=_frozen(ordinal) if ordinal is not None else None,
        tie=_frozen(tie) if tie is not None else None,
        needs_ordinal=raw["needs_ordinal"],
        excluded_from_tally=raw["excluded_from_tally"],
        fault=raw.get("fault"),
        conditions=tuple(given["conditions"]),
        expect=_frozen(raw["expect"]),
        source=raw["source"],
        extension=raw.get("extension"),
    )


def sizes_of_listed_rows(
    folder: Path, fallbacks: Sequence[str] = NO_FALLBACKS
) -> Mapping[str, tuple[str | None, ...]]:
    """The sizes at which each core row of Checks 1 to 9 in ``folder`` is asked, read
    the way the tool's own rows are: a row the reader reads as sized is asked at
    ``small`` and at each size its set stands for; any other row, one the reader
    cannot read included, is asked once. A set the reader does not know raises."""
    table = _acceptance_table()
    sizes: dict[str, tuple[str | None, ...]] = {}
    for number in LISTED_CHECKS:
        for row in table.load_check(number, folder):
            # The extension rows are listed in groups of their own since 18 Nov 2026 (owner's answer Q1): their sizes are read the
            # same way (asked once where the reader reads them so).
            try:
                sized = table.interpret(row, fallbacks).sized
            except table.TableError:
                sized = False
            sizes[row.id] = (SIZE_SMALL, *table.sizes_of(row)) if sized else (None,)
    return MappingProxyType(sizes)


def load_table10(path: str | Path | None = None) -> Table10:
    """Read the table from ``path``, or from the table folder of this repository.
    The check files that give the sizes are read from the folder of ``path``."""
    where = Path(path) if path is not None else DEFAULT_PATH
    with open(where, "rb") as handle:
        document = tomllib.load(handle)
    return Table10(
        groups=tuple(_group(g) for g in document["group"]),
        rows=tuple(_row(r) for r in document["row"]),
        sizes=sizes_of_listed_rows(where.parent),
    )


def question_id(listed_row: str, subject: str, size: str | None = None) -> str:
    """The id of the question that puts ``listed_row`` to ``subject`` at ``size``."""
    base = listed_row + QUESTION_ID_SEPARATOR + subject
    return base if size is None else base + SIZE_SEPARATOR + size


def expand(table: Table10, extensions: Collection[str] = ()) -> tuple[Question, ...]:
    """One question for every listed row of every group, every size at which the
    listed row is asked, and every subject. A row of an extension (``extension`` is named on it)
    is asked only while that extension is among ``extensions``: the table's README, "An extension
    row is put only while its extension is on" (plan :305); the default is none.

    Order: groups in file order, then the group's listed rows in order, then the
    row's sizes in order, then the group's rows (one per subject) in file order. A
    listed row the table of Checks 1 to 9 does not hold raises ``KeyError``.
    """
    rows_of_group: dict[str, list[Row]] = {}
    for row in table.rows:
        if row.extension is not None and row.extension not in extensions:
            continue
        rows_of_group.setdefault(row.group, []).append(row)
    questions = []
    for group in table.groups:
        if group.id not in rows_of_group:     # every row of the group belongs to an extension that is off
            continue
        for listed in group.rows:
            for size in table.sizes[listed]:
                for row in rows_of_group[group.id]:
                    questions.append(
                        Question(
                            id=question_id(listed, row.subject, size),
                            group=group.id,
                            row_id=row.id,
                            listed_row=listed,
                            subject=row.subject,
                            check=group.check,
                            size=size,
                            excluded_from_tally=row.excluded_from_tally,
                            route=row.route,
                            missing=row.missing,
                            needs_ordinal=row.needs_ordinal,
                            marks=row.marks,
                            tie_mark=row.tie_mark,
                        )
                    )
    return tuple(questions)
