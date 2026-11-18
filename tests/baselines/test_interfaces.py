"""The interfaces of the baselines (plan 5.5 Check 10; table README, section
"Check 10"): the status words, the three marks, the six subjects and the types a
subject's reading, a grade, a failed attempt, a question and a record are made of.

The words come from the table: the three grade words are the union of every
``expect.one_of`` of ``check10.toml``, the three marks are the strings its rows
carry, the six subjects are the values of ``given.subject``. The README writes the
fourth status, for a failed attempt, as "no answer: failed attempt <id>".

The names this file asks of ``baselines.interfaces``:

- ``RIGHT``, ``WRONG``, ``CANNOT_ANSWER``, ``GRADES`` (the three words, in that
  order), ``NO_ANSWER_PREFIX`` and ``no_answer(attempt_id)``;
- ``MARK_OWN_OUTPUT``, ``MARK_STEP_TO_ORDINAL``, ``MARK_TIE`` and ``MARKS``;
- ``SUBJECTS``: the six subject names, in the order of the README's table;
- frozen dataclasses ``Question``, ``Reading``, ``Grade``, ``Attempt`` and
  ``Record``, with the fields listed in ``FIELDS`` below (a type may hold more); a
  ``Question`` carries its ``size``: None for a listed row asked once, else
  "small", "10k" or "100k";
- ``Subject``: a runtime-checkable protocol with ``name`` and ``read(question)``.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

FIELDS = {
    "Question": {
        "id", "group", "row_id", "listed_row", "subject", "check", "size",
        "excluded_from_tally", "route", "missing", "needs_ordinal", "marks", "tie_mark",
    },
    "Reading": {"question_id", "raw", "route"},
    "Grade": {"status", "evidence"},
    "Attempt": {"id", "command", "exit_code", "output"},
    "Record": {
        "question_id", "status", "raw", "evidence", "route", "marks",
        "excluded_from_tally", "attempt_id",
    },
}

# The modules these tests cover.
MODULES = ("baselines", "baselines.interfaces", "baselines.table10")

# Check 10 reuses the pure functions of some acceptance helpers and never imports
# the others (they import the replica's world); nothing here imports the tool.
FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}

# What stays out of a tracked file: an em-dash and a local path (written in two
# pieces so that this file holds neither).
FORBIDDEN_TEXT = (chr(0x2014),"/ho" + "me/", "Proj" + "ects/")


# --- The status words --------------------------------------------------------------------


def test_the_grade_words_are_the_three_the_table_uses(need_module, facts):
    interfaces = need_module("baselines.interfaces")
    in_table = set()
    for row in facts.rows:
        in_table |= set(row["expect"]["one_of"])
    assert in_table == {"right", "wrong", "cannot answer"}
    assert interfaces.RIGHT == "right"
    assert interfaces.WRONG == "wrong"
    assert interfaces.CANNOT_ANSWER == "cannot answer"
    assert tuple(interfaces.GRADES) == ("right", "wrong", "cannot answer")
    assert set(interfaces.GRADES) == in_table


def test_a_failed_attempt_has_a_status_that_is_no_grade(need_module):
    interfaces = need_module("baselines.interfaces")
    assert interfaces.NO_ANSWER_PREFIX == "no answer: failed attempt "
    assert interfaces.no_answer("A3") == "no answer: failed attempt A3"
    assert interfaces.no_answer("A3") not in interfaces.GRADES
    # The id is kept whole.
    assert interfaces.no_answer("dolma-7").endswith("dolma-7")


# --- The marks ---------------------------------------------------------------------------


def test_the_three_marks_equal_the_strings_of_the_table(need_module, facts):
    interfaces = need_module("baselines.interfaces")
    in_rows = set()
    for row in facts.rows:
        in_rows |= set(row["marks"])
        if "ordinal" in row:
            in_rows.add(row["ordinal"]["shown"]["mark"])
    assert in_rows == {facts.mark_own_output, facts.mark_step_to_ordinal, facts.mark_tie}
    assert interfaces.MARK_OWN_OUTPUT == facts.mark_own_output
    assert interfaces.MARK_STEP_TO_ORDINAL == facts.mark_step_to_ordinal
    assert interfaces.MARK_TIE == facts.mark_tie
    assert set(interfaces.MARKS) == in_rows
    assert len(interfaces.MARKS) == 3


# --- The subjects ------------------------------------------------------------------------


def test_the_six_subjects_are_the_ones_of_the_table_in_the_readmes_order(need_module, facts):
    interfaces = need_module("baselines.interfaces")
    assert tuple(interfaces.SUBJECTS) == facts.subjects_in_readme_order
    assert set(interfaces.SUBJECTS) == {r["given"]["subject"] for r in facts.rows}
    assert len(interfaces.SUBJECTS) == facts.subject_count == 6


def test_a_subject_is_anything_with_a_name_and_a_read_of_a_question(need_module):
    interfaces = need_module("baselines.interfaces")

    class Stub:
        name = "dolma"

        def read(self, question):
            return None

    class NoRead:
        name = "dolma"

    assert isinstance(Stub(), interfaces.Subject)
    assert not isinstance(NoRead(), interfaces.Subject)
    assert not isinstance(object(), interfaces.Subject)


# --- The types ---------------------------------------------------------------------------


def test_the_types_exist_as_frozen_dataclasses_with_the_fields_the_record_needs(need_module):
    interfaces = need_module("baselines.interfaces")
    for name, wanted in FIELDS.items():
        kind = getattr(interfaces, name, None)
        assert kind is not None, f"baselines.interfaces has no {name}"
        assert dataclasses.is_dataclass(kind), name
        assert kind.__dataclass_params__.frozen, f"{name} is not frozen"
        held = {f.name for f in dataclasses.fields(kind)}
        assert wanted <= held, f"{name} lacks {sorted(wanted - held)}"


# --- What these modules may hold --------------------------------------------------------


def _sources(need_module) -> dict[str, Path]:
    found = {}
    for name in MODULES:
        module = need_module(name)
        path = getattr(module, "__file__", None)
        assert path, f"{name} has no file"
        found[name] = Path(path)
    return found


def test_the_modules_import_nothing_of_the_tool_nor_of_the_world_helpers(need_module):
    for name, path in _sources(need_module).items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        assert not roots & FORBIDDEN_IMPORT_ROOTS, f"{name} imports {sorted(roots & FORBIDDEN_IMPORT_ROOTS)}"


def test_the_modules_hold_no_local_path_and_no_em_dash(need_module):
    for name, path in _sources(need_module).items():
        text = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_TEXT:
            assert bad not in text, f"{name} holds {bad!r}"
