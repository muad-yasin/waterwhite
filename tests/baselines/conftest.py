"""Fixtures of the tests of the baselines' interfaces and table loader.

What these tests expect comes from the table files themselves: ``fixture/table/
check01.toml`` to ``check10.toml`` are read here with ``tomllib``, and the counts
(groups, core rows, questions per size, rows of the tally, rows with a route) are
taken from them by the code of this file. The size at which a listed row is asked
is worked out here from ``tests/acceptance/table.py`` (``interpret(...).sized`` and
``sizes_of``), the tool's own reading of its rows. Nothing here reads a value out
of the ``baselines`` package; the loader is compared with the file.

The package does not exist when these tests are written, so a test asks for a
module with ``need``: a missing module is an assertion that fails, not an import
error. ``need`` is called in the body of a test, never in a fixture, so that a
missing module is reported as a failed test and not as an error at setup.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
CHECK10_FILE = TABLE_DIR / "check10.toml"

# README, "The six subjects and their own records": the order of its table.
SUBJECTS_IN_README_ORDER = (
    "manifest",
    "trace",
    "dolma",
    "originblame",
    "dolma+trace",
    "originblame+trace",
)

# README, "Marks and the tally": the three marks, as the table writes them.
MARK_OWN_OUTPUT = "tool's own output; not an independent baseline"
MARK_STEP_TO_ORDINAL = "step-to-ordinal part is the tool's own output"
MARK_TIE = "ordinal derived from src and loc (owner's ruling after the debate)"

# README, "Marks and the tally": the rows left out of the tally are those of the
# trace and of both combinations for a question of Check 1 or Check 2.
TALLY_EXCLUDED_CHECKS = (1, 2)
TALLY_EXCLUDED_SUBJECTS = ("trace", "dolma+trace", "originblame+trace")

# The sizes. A question is one (listed row, subject, size): the size is None for a
# listed row asked once, "small" for the 800-document set, and "10k" or "100k" for
# a plan size.
SIZE_ONCE = None
SIZE_SMALL = "small"
PLAN_SIZES = ("10k", "100k")

# The numbers are the counts taken from the table at 5b76911 and frozen before the run.
GROUPS = 66
CORE_ROWS = 370
SUBJECT_COUNT = 6
SIZED_LISTED_ROWS = 17
QUESTIONS = 2346
QUESTIONS_BY_SIZE = {SIZE_ONCE: 2118, SIZE_SMALL: 102, "10k": 24, "100k": 102}
QUESTIONS_PER_SUBJECT = 391
EXCLUDED_QUESTIONS = 147
QUESTIONS_WITH_A_ROUTE = 737
QUESTIONS_WITH_NO_RECORD = 1609

# The one listed row that names the 100k set and is asked once.
EOD_CONVENTION = "c03.precondition.eod-convention"


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    module = need(module_name)
    found = getattr(module, attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def acceptance_module(name: str):
    """A module of ``tests/acceptance`` (they import each other by bare name, so
    that folder goes on the path first)."""
    folder = str(REPO / "tests" / "acceptance")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module(name)


class TableFacts:
    """What the table files say, read straight from them."""

    # The words and numbers above, reachable from a test through the fixture.
    subjects_in_readme_order = SUBJECTS_IN_README_ORDER
    mark_own_output = MARK_OWN_OUTPUT
    mark_step_to_ordinal = MARK_STEP_TO_ORDINAL
    mark_tie = MARK_TIE
    groups_count = GROUPS
    core_rows_count = CORE_ROWS
    subject_count = SUBJECT_COUNT
    sized_listed_rows_count = SIZED_LISTED_ROWS
    questions_count = QUESTIONS
    questions_by_size = QUESTIONS_BY_SIZE
    questions_per_subject = QUESTIONS_PER_SUBJECT
    excluded_count = EXCLUDED_QUESTIONS
    route_count = QUESTIONS_WITH_A_ROUTE
    no_record_count = QUESTIONS_WITH_NO_RECORD
    eod_convention = EOD_CONVENTION
    table_dir = TABLE_DIR
    check10_file = CHECK10_FILE
    repo = REPO

    def __init__(self) -> None:
        self.check10 = _read(CHECK10_FILE)
        # The file as it stands (every group and row of it) and the table as a pass with every extension OFF reads it (a row of an
        # extension is not asked while its extension is off: plan :305): ``groups`` and ``rows`` are the second, ``all_groups`` and
        # ``all_rows`` the first.
        self.all_groups = tuple(self.check10["group"])
        self.all_rows = tuple(self.check10["row"])
        extension_groups = {r["given"]["group"] for r in self.all_rows if "extension" in r}
        self.groups = tuple(g for g in self.all_groups if g["id"] not in extension_groups)
        self.rows = tuple(r for r in self.all_rows if "extension" not in r)
        self.group_by_id = {g["id"]: g for g in self.all_groups}
        self.row_by_id = {r["id"]: r for r in self.all_rows}
        # The core rows of Checks 1 to 9, in file order: id -> check number.
        self.core_check: dict[str, int] = {}
        self.core_given: dict[str, dict] = {}
        self.extension_ids: list[str] = []
        self.extension_check: dict[str, int] = {}
        self.extension_of: dict[str, str] = {}
        for number in range(1, 10):
            document = _read(TABLE_DIR / f"check{number:02d}.toml")
            for row in document["row"]:
                if row["core"]:
                    self.core_check[row["id"]] = number
                    self.core_given[row["id"]] = row["given"]
                else:
                    self.extension_ids.append(row["id"])
                    self.extension_check[row["id"]] = number
                    self.extension_of[row["id"]] = row["extension"]
        self.subjects = tuple(dict.fromkeys(r["given"]["subject"] for r in self.rows))
        self.sizes_by_listed = self._sizes_by_listed()

    def _sizes_by_listed(self) -> dict[str, tuple]:
        """The sizes at which each listed row is asked, in the tool's own reading:
        for a row that ``interpret`` reads as sized, "small" and then the sizes
        ``sizes_of`` gives; for any other row (one read as not sized, or one the
        reader cannot read, as the ``c09.10k`` rows are) the single entry None.
        Worked out here from ``tests/acceptance/table.py`` and the core rows of
        Checks 1 to 9, never from the code under test."""
        table = acceptance_module("table")
        sizes: dict[str, tuple] = {}
        for number in range(1, 10):
            for row in table.load_check(number):
                if row.id not in self.core_check:
                    continue
                try:
                    sized = table.interpret(row, ()).sized
                except table.TableError:
                    sized = False
                sizes[row.id] = (
                    (SIZE_SMALL, *table.sizes_of(row)) if sized else (SIZE_ONCE,)
                )
        return sizes

    def row_by_listed_given(self, listed_row: str) -> dict:
        """The ``given`` of a listed row's own row in Checks 1 to 9, as the file
        writes it."""
        return self.core_given[listed_row]

    def rows_of_group(self, group_id: str) -> list[dict]:
        return [r for r in self.rows if r["given"]["group"] == group_id]

    def group_of(self, listed_row: str) -> str:
        """The id of the group that lists a core row."""
        for group in self.groups:
            if listed_row in group["rows"]:
                return group["id"]
        raise KeyError(listed_row)

    def group_check(self, group_id: str) -> int:
        return self.group_by_id[group_id]["check"]

    def tally_excluded(self, group_id: str, subject: str) -> bool:
        """README's rule for the tally, applied by this file and not read from a
        row's own flag."""
        return (
            self.group_check(group_id) in TALLY_EXCLUDED_CHECKS
            and subject in TALLY_EXCLUDED_SUBJECTS
        )

    def question_triples(self) -> list[tuple[str, str, str | None]]:
        """(listed row id, subject, size) for every listed row of every group, once
        for each subject and each size the listed row is asked at."""
        return [
            (listed, row["given"]["subject"], size)
            for group in self.groups
            for listed in group["rows"]
            for size in self.sizes_by_listed[listed]
            for row in self.rows_of_group(group["id"])
        ]


@pytest.fixture(scope="session")
def facts() -> TableFacts:
    return TableFacts()


@pytest.fixture
def acceptance_modules():
    return acceptance_module


@pytest.fixture
def need_module():
    return need


@pytest.fixture
def need_attribute():
    return need_attr
