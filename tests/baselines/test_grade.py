"""The grader (plan 5.5 Check 10; table README, section "Grades, and what is
compared"): ``baselines.grade``.

The expected values are read from the table files (``check01.toml`` to ``check09.toml``
and the fixture spec) by the helpers below, or are written by hand below. The readings are
written by hand; no tool ran.

The names this file asks of ``baselines.grade``:

- ``grade(question, expect, outcome)`` returns a ``baselines.interfaces.Grade``.
  ``question`` is a ``Question``; ``expect`` is the mapping of the listed row that is
  in force (an ``Expected``'s ``expect``); ``outcome`` is what the subject gave for
  that question: a ``Reading`` whose ``raw`` is a mapping of the fields it read, or a
  failed ``Attempt``.
- ``NotGradable``, a ``ValueError``: raised for an ``Attempt`` (a failed attempt is no
  answer and is never graded), for None (the subject did not run), for a reading
  whose ``question_id`` is not the question's, and for an ``expect`` that says the row
  does not apply (``{"applies": False}``: such a row is not put).
- ``CANNOT_KEY``, ``STOPPED_KEY``: the keys of ``raw`` that say the route could not be
  walked (its value is the evidence) and that a command of the subject stopped with an
  error.
- ``grade(question, expect, outcome, *, attempts=())``: ``attempts`` is the list of the
  ``baselines.record.StoreAttempt`` of the run (the none-line rule, below).
- ``grade_concrete(question, concrete, outcome, *, attempts=())``: the same for an
  ``baselines.expected.Concrete`` (the expectation with each sentence-valued field
  replaced by its concrete value). A ``Concrete`` that still holds a field with no
  concrete value (its ``open``) is not graded: ``NotGradable`` naming the field, so that
  no prose of the table is compared as a string.
- ``grade_whole_run(question, outcome, oracle)``: the three whole-run diff rows of Check 1.
  The grader takes the oracle's log (a list of its lines as mappings), computes the rows
  the join must report for the run, and compares them with the reading's ``rows`` (the
  subject's rows of the whole run, one mapping per row). The subject's own ``diff`` is
  never taken.
- ``grade_from_records(question, expect, outcome)``: the two rows that allow a set of
  outcomes (``one_of``, ``never``) read right only from the subject's own records: the
  reading's ``raw["records"]`` holds the counts (``rows_after_stop`` for
  c04.kill-after-100; ``rows_after_retry``, ``refused``, a named ``mismatch`` (a
  non-empty text) and, where the subject can say, ``matches_shard`` for c04.retry) and
  the outcome is worked out from them. An ``outcome`` the reading names beside them is
  never taken. ``grade`` reads the same two rows through the same path: it reads no
  bare outcome, and a reading with no records is wrong, evidence ``["records"]``.
- ``grade_layouts(question, by_layout, outcomes)``: a line whose listed row names two
  layouts is asked in each; ``by_layout`` is what
  ``baselines.expected.concrete_values_by_layout`` returns and ``outcomes`` maps each
  layout to the reading (or None: that layout was not attempted), or to a failed
  ``Attempt`` (the line then has the status of that attempt). One line comes back, put
  together by ``baselines.expected.combine_layouts`` in this order: wrong if
  any layout is wrong; otherwise ``no answer: failed attempt <id>`` if any layout's
  attempt failed; otherwise ``not attempted`` if any layout was not attempted; otherwise
  right if all are right; otherwise cannot answer, naming the layout.
- ``grade`` itself also refuses (``NotGradable`` naming the field) an expectation whose
  field still holds the table's sentence where a value stands (the rows of
  ``SENTENCE_FIELDS`` below): the sentence is never compared as a string.

The rules, from the README:

- A question whose row holds no record (``question.route`` is None) is graded cannot
  answer only after the subject's attempt on the store or set its listed row names is
  recorded as having run (``attempts``; ``baselines.record.none_line_status``); its
  evidence is the question's ``missing`` text word for word. It is never right or wrong.
  With no such attempt it is not graded (``NotGradable``: the line is "not attempted"),
  and a failed attempt is not graded either. A row that names two layouts needs an attempt
  per layout: a ``StoreAttempt`` carries its ``layout`` (None for a row that names no
  layout), and one layout's attempt alone leaves the line not attempted.
- A reading whose ``raw`` holds ``CANNOT_KEY`` is cannot answer; the evidence is that
  value. An empty or None value says nothing and is read as no key. On a row that allows
  a set of outcomes none of which is "cannot answer" (the two rows of ``grade_from_records``)
  the key is no outcome of the row: wrong, evidence ``["records"]``.
- Otherwise the reading is compared with the expected answer on the fact the question
  asks for. For an expected ``answer`` the field compared is ``answer`` among the
  fields tested here (the rows of the combinations say that only the answer is
  compared; whether ``first_exposure``, ``untraced_steps`` or ``candidates`` are
  compared for the other subjects is left open and is not pinned). For an expected stop (the expectation holds ``error``) the
  reading must be a stop (``raw[STOPPED_KEY]`` true) and nothing about the error is
  compared; an answer is wrong there, and a stop is wrong where an answer is expected.
  For an expected ``rows`` (the filled table of the spec, with its ``log_lines``) the
  reading's ``rows`` must equal the oracle's lines on (segment, step, sample_pos, shard
  position, ordinal, token start, token end), row for row, in order. The values the rule
  of the check10 row names are compared, every one of them, and a field the rule does not
  name is not compared: the rule of a position row
  names the trace's rows, so ``row``, ``rows_in_sample``, ``documents_in_shard``,
  ``covered_tokens`` and the like beside them are not read. Further readings of the table
  drop fields, each pinned below with the text of the table it rests on: a row that asks
  the values of a launch (``consumed_at_start``, ``first_step``, ``last_step``) is graded
  on those values and not on its rows; ``diff`` is never read from a reading, since it
  only says that the rows equal the oracle's; and the rule of an action row leaves out the
  byte identity afterwards, the records removed and the size of a successor.
- Not compared: ``names``, ``names_detail``, ``failing``, ``chain``, ``scope``,
  ``reason``, ``error``, ``exit_code``, ``notes``, ``basis``: the fields the table README
  (section "Grades, and what is compared", and its reading 7) leaves out as this
  tool's way of saying why.
- A field the expectation asks for that the reading lacks is wrong, never read as
  absence. A wrong grade's evidence lists the names of the fields that differ, sorted.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import tomllib
from functools import lru_cache
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    found = getattr(need(module_name), attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _spec() -> dict:
    """The fixture spec as the file writes it."""
    return _read_toml(SPEC_FILE)


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    """Every core row of Checks 1 to 9 as the file writes it, by id."""
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _read_toml(TABLE_DIR / "check10.toml")["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, tuple[str, int]]:
    """(group id, check number) of every listed row."""
    found = {}
    for group in _read_toml(TABLE_DIR / "check10.toml")["group"]:
        for listed in group["rows"]:
            found[listed] = (group["id"], group["check"])
    return found


def _question_id(listed_row: str, subject: str, size: str | None = None) -> str:
    base = f"{listed_row}|{subject}"
    return base if size is None else f"{base}@{size}"


def _make_question(listed_row: str, subject: str, size: str | None = None, **changes):
    """The question that puts ``listed_row`` to ``subject`` at ``size``, its fields
    taken from the subject's row of the group in ``check10.toml``, by the reading the
    table README gives (so these tests do not depend on the expander of
    ``baselines.table10``)."""
    question_type = need_attr("baselines.interfaces", "Question")
    group, check = _group_of_listed()[listed_row]
    row = _check10_rows()[f"c10.{group}.{subject}"]
    ordinal = row.get("ordinal")
    fields = dict(
        id=_question_id(listed_row, subject, size),
        group=group,
        row_id=row["id"],
        listed_row=listed_row,
        subject=subject,
        check=check,
        size=size,
        excluded_from_tally=row["excluded_from_tally"],
        route=row.get("route"),
        missing=row.get("missing"),
        needs_ordinal=row["needs_ordinal"],
        marks=tuple(row["marks"]),
        tie_mark=ordinal["shown"]["mark"] if ordinal is not None else None,
    )
    fields.update(changes)
    return question_type(**fields)


def _make_reading(question, raw, route="the route named before the run"):
    reading_type = need_attr("baselines.interfaces", "Reading")
    return reading_type(question_id=question.id, raw=raw, route=route)


def _make_attempt(attempt_id="attempt-1", exit_code=1, output="no companion row was written"):
    attempt_type = need_attr("baselines.interfaces", "Attempt")
    return attempt_type(
        id=attempt_id, command=("tool", "run", "input.jsonl"), exit_code=exit_code, output=output
    )


def _make_grade(status, evidence=None):
    grade_type = need_attr("baselines.interfaces", "Grade")
    return grade_type(status=status, evidence=evidence)


STOP_ROW = "c04.count-499"  # expects an error, an exit code and an empty stdout
ANSWER_ROW = "c05.f.edge-deleted.E@9"  # answer, names, failing
NOTES_ROW = "c05.unmutated.R1"  # answer and a process-evidence note
POSITIONS_ROW = "c01.E.T"


def _expect(listed_row: str) -> dict:
    """The expectation in force for a row with no fallback: the table's ``expect``,
    its pending values filled from the spec (the table README, section 5)."""

    def fill(value):
        if isinstance(value, str) and value.startswith("pending:"):
            found = _spec()
            for part in value[len("pending:"):].split("."):
                found = found[part]
            return found
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        if isinstance(value, list):
            return [fill(v) for v in value]
        return value

    return fill(_listed_rows()[listed_row]["expect"])


def _grade(question, expect, outcome):
    return need_attr("baselines.grade", "grade")(question, expect, outcome)


# --- The answer and the words that are not compared --------------------------------------


def test_the_same_answer_is_right_and_another_answer_is_wrong(facts):
    expect = _expect(ANSWER_ROW)
    assert expect["answer"] == "may contain"
    q = _make_question(ANSWER_ROW, "manifest")
    right = _grade(q, expect, _make_reading(q, {"answer": "may contain"}))
    assert right.status == "right"
    for other in ("contains", "does not contain", "not covered"):
        wrong = _grade(q, expect, _make_reading(q, {"answer": other}))
        assert wrong.status == "wrong", other
        assert list(wrong.evidence) == ["answer"], other


def test_the_cause_the_link_the_chain_the_scope_and_the_notes_are_not_compared():
    expect = _expect(ANSWER_ROW)
    assert {"names", "failing"} <= set(expect)
    q = _make_question(ANSWER_ROW, "manifest")
    other_words = {
        "answer": "may contain",
        "names": "a different cause in this tool's words",
        "names_detail": "something else",
        "failing": {"condition": "z", "artifact": "gen/Q", "record": "Q"},
        "chain": ["not", "the", "listed", "chain"],
        "scope": {"source": "B"},
        "reason": "another reason",
        "notes": ["another note"],
        "basis": "another basis",
        "error": "another error name",
        "exit_code": 99,
    }
    assert _grade(q, expect, _make_reading(q, other_words)).status == "right"
    # The same words with another answer are still wrong.
    other_words["answer"] = "contains"
    assert _grade(q, expect, _make_reading(q, other_words)).status == "wrong"
    # The note of a row whose expectation holds one is not asked of the reading.
    notes_expect = _expect(NOTES_ROW)
    assert "notes" in notes_expect
    nq = _make_question(NOTES_ROW, "manifest")
    assert _grade(nq, notes_expect, _make_reading(nq, {"answer": "contains"})).status == "right"


# --- A gap is never an absence ---------------------------------------------------------------


def _gap_grades_right(grade_function) -> bool:
    """True when a reading that holds no answer is graded right against a row that
    expects "does not contain" (the shape of a gap read as absence)."""
    expect = {"answer": "does not contain"}
    q = _make_question(ANSWER_ROW, "manifest")
    for raw in ({}, {"answer": None}, {"names": "a cause and no answer"}):
        outcome = _make_reading(q, raw)
        try:
            graded = grade_function(q, expect, outcome)
        except ValueError:
            continue
        if graded.status == "right":
            return True
    return False


def _gap_read_as_absence(real_grade):
    """The known-bad grader: a reading without an answer is read as "does not contain"
    before it is compared."""

    def bad(question, expect, outcome):
        raw = dict(outcome.raw)
        if raw.get("answer") is None:
            raw["answer"] = "does not contain"
        return real_grade(question, expect, _make_reading(question, raw))

    return bad


def test_a_reading_with_a_gap_is_not_graded_as_an_absence():
    real = need_attr("baselines.grade", "grade")
    # The same check run on the known-bad grader finds the fault, so it can fail.
    assert _gap_grades_right(_gap_read_as_absence(real)) is True
    assert _gap_grades_right(real) is False
    # The real grader says wrong for the gap, and names the field that is missing.
    q = _make_question(ANSWER_ROW, "manifest")
    got = real(q, {"answer": "does not contain"}, _make_reading(q, {}))
    assert got.status == "wrong"
    assert list(got.evidence) == ["answer"]


def test_a_does_not_contain_read_off_a_gap_is_wrong_against_a_row_that_expects_a_stop():
    expect = _expect(STOP_ROW)
    assert "error" in expect and "answer" not in expect
    q = _make_question(STOP_ROW, "dolma")
    assert _grade(q, expect, _make_reading(q, {"answer": "does not contain"})).status == "wrong"
    assert _grade(q, expect, _make_reading(q, {})).status == "wrong"


# --- A stop ---------------------------------------------------------------------------------


def test_a_stop_is_right_where_a_stop_is_expected_and_the_error_is_not_compared():
    key = need_attr("baselines.grade", "STOPPED_KEY")
    expect = _expect(STOP_ROW)
    q = _make_question(STOP_ROW, "dolma")
    got = _grade(q, expect, _make_reading(q, {key: True, "error": "another name", "exit_code": 1}))
    assert got.status == "right"
    # A stop where an answer is expected is wrong.
    answer_expect = _expect(ANSWER_ROW)
    aq = _make_question(ANSWER_ROW, "manifest")
    assert _grade(aq, answer_expect, _make_reading(aq, {key: True})).status == "wrong"


# --- The positions --------------------------------------------------------------------------


def _lines(expect):
    return [json.loads(line) for line in expect["rows"]["log_lines"]]


def _values_beside_rows(expect):
    """The value fields of a rows-form expectation other than its rows: every field but
    the rows, the diff (it only says that the rows equal the oracle's) and the words that
    describe the check."""
    return {
        key: value for key, value in expect.items()
        if key not in ("rows", "diff", "rule", "compared")
    }


def _changed(value):
    """A value that is not ``value``, of the same kind."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + " (another)"
    if isinstance(value, dict):
        first = next(iter(value))
        return {**value, first: _changed(value[first])}
    return list(value) + ["an extra item"]


def test_positions_are_compared_row_for_row_with_the_oracles_lines():
    # Changed bar: the rule of a position row names the trace's rows and nothing else, so
    # the value fields beside the rows (row, rows_in_sample, rows_of_T_in_run_E) are not
    # compared: the rows alone make a reading right or wrong.
    for subject in ("trace", "dolma+trace", "originblame+trace"):
        rule = _check10_rows()[f"c10.c01.positions.{subject}"]["rule"]
        assert rule == (
            "the reading is the trace's rows for the run, segment, step or position of the "
            "listed row."
        ), subject
    expect = _expect(POSITIONS_ROW)
    lines = _lines(expect)
    assert len(lines) == 3
    values = _values_beside_rows(expect)
    assert set(values) == {"row", "rows_in_sample", "rows_of_T_in_run_E"}
    q = _make_question(POSITIONS_ROW, "trace")
    assert q.route is not None

    def read(**changes):
        return _grade(q, expect, _make_reading(q, {**values, "rows": lines, **changes}))

    assert read().status == "right"
    # Fields the oracle logs and the join does not report (rank, global index) do not matter.
    stripped = [{k: v for k, v in r.items() if k not in ("rank", "global_index")} for r in lines]
    assert read(rows=stripped).status == "right"
    # A row that differs, a row that is missing, a row that is extra, rows out of order.
    changed = [dict(r) for r in lines]
    changed[1]["token_end"] += 1
    assert read(rows=changed).status == "wrong"
    assert read(rows=lines[:2]).status == "wrong"
    assert read(rows=lines + [lines[0]]).status == "wrong"
    assert read(rows=lines[::-1]).status == "wrong"
    gap = _grade(q, expect, _make_reading(q, dict(values)))
    assert gap.status == "wrong" and list(gap.evidence) == ["rows"]
    # The other values of the row are not named by the rule: a value that is changed, and a
    # value the reading lacks, leave the grade to the rows.
    for key, value in values.items():
        got = read(**{key: _changed(value)})
        assert got.status == "right" and got.evidence is None, key
        lacking = {k: v for k, v in values.items() if k != key}
        got = _grade(q, expect, _make_reading(q, {**lacking, "rows": lines}))
        assert got.status == "right" and got.evidence is None, key
    # Rows that differ are wrong whatever those values say, and only the rows are named.
    every_value_changed = {key: _changed(value) for key, value in values.items()}
    got = read(rows=changed, **every_value_changed)
    assert got.status == "wrong" and list(got.evidence) == ["rows"]


# The rows of group c01.positions that hold rows (c01.E.diff and its two siblings are the
# whole-run rows, graded from the oracle's log below). Each is read with every value field
# of its expectation; the rows are the oracle's lines, or for the cut of step 8 the rows of
# the covered part (the rule of that row in check01.toml).
POSITION_ROWS = (
    "c01.E.T",
    "c01.E.skipped-line",
    "c01.E.repeated-draw",
    "c01.E.shared-ordinal",
    "c01.E.split-document",
    "c01.E.shared-token",
    "c01.E.step8.first-half",
    "c01.K.epoch-boundary",
)


def _position_reading(expect):
    """A reading that holds exactly what the row holds: the rows (the covered part where
    the row names the tokens covered) and every other value field."""
    rows = expect["rows"]
    if "covered_tokens" in expect:
        lines = [dict(row) for row in rows["rows_of_the_covered_part"]]
    else:
        lines = _lines(expect)
    return {**_values_beside_rows(expect), "rows": lines}


def test_only_the_rows_of_a_position_row_are_compared_and_the_diff_is_never_taken():
    # Changed bar (the name and the value loop): the value fields beside the rows are not
    # named by the rule of the group, so they are not compared; the rows are.
    grade_fn = need_attr("baselines.grade", "grade")
    fields_seen = set()
    # The text of the table this rests on: for the rows of c01.positions right is "the rows
    # equal the oracle's rows the listed row names", the rule names the trace's rows, and the
    # diff the rows hold says no more than that.
    for subject in ("trace", "dolma+trace", "originblame+trace"):
        text = _check10_rows()[f"c10.c01.positions.{subject}"]["expect"]
        assert text["right"] == "the rows equal the oracle's rows the listed row names", subject
        assert _check10_rows()[f"c10.c01.positions.{subject}"]["rule"].startswith(
            "the reading is the trace's rows for the run, segment, step or position"
        ), subject
    for listed in POSITION_ROWS:
        expect = _expect(listed)
        q = _make_question(listed, "trace")
        assert q.route is not None and q.group == "c01.positions", listed
        raw = _position_reading(expect)
        assert grade_fn(q, expect, _make_reading(q, raw)).status == "right", listed
        # The diff only says that the rows equal the oracle's: its words are not read.
        said = dict(raw, diff="3 rows differ")
        assert grade_fn(q, expect, _make_reading(q, said)).status == "right", listed
        # A row that differs is wrong and named, whatever the reading's own diff says.
        rows = [dict(r) for r in raw["rows"]]
        rows[-1]["token_end"] += 1
        got = grade_fn(q, expect, _make_reading(q, dict(raw, rows=rows, diff="empty")))
        assert got.status == "wrong" and list(got.evidence) == ["rows"], listed
        for key in _values_beside_rows(expect):
            fields_seen.add(key)
            got = grade_fn(q, expect, _make_reading(q, dict(raw, **{key: _changed(raw[key])})))
            assert got.status == "right" and got.evidence is None, (listed, key)
            lacking = {k: v for k, v in raw.items() if k != key}
            got = grade_fn(q, expect, _make_reading(q, lacking))
            assert got.status == "right" and got.evidence is None, (listed, key)
    # The values the table holds beside the rows of these eight rows (none is compared).
    assert fields_seen == {
        "row", "rows_in_sample", "rows_of_T_in_run_E", "documents_in_shard", "skipped_line",
        "sample_tokens", "rows_of_W_in_steps_1_to_9", "covered_tokens", "boundary_after_step",
    }


def test_the_rows_of_the_cut_of_step_8_are_the_covered_part_and_the_text_it_rests_on():
    grade_fn = need_attr("baselines.grade", "grade")
    expect = _expect("c01.E.step8.first-half")
    # The text of the table this reading rests on (check01.toml, the row's own rule).
    assert expect["rule"] == (
        "rows that end within the first 16 tokens are equal; the row that holds the 17th "
        "token ends at the cut; no later row is reported"
    )
    assert expect["covered_tokens"] == 16
    covered = [dict(row) for row in expect["rows"]["rows_of_the_covered_part"]]
    assert covered == [{"shard_position": 2, "ordinal": 2, "token_start": 19, "token_end": 35}]
    q = _make_question("c01.E.step8.first-half", "trace")
    raw = {"rows": covered, "covered_tokens": 16}
    assert grade_fn(q, expect, _make_reading(q, raw)).status == "right"
    # The row of the oracle's log, uncut (it ends at token 44), is not the covered part.
    uncut = [dict(r) for r in _lines(expect)]
    assert uncut[0]["token_end"] == 44
    got = grade_fn(q, expect, _make_reading(q, {"rows": uncut[:1], "covered_tokens": 16}))
    assert got.status == "wrong" and list(got.evidence) == ["rows"]
    # A later row is reported where none may be.
    got = grade_fn(q, expect, _make_reading(q, {"rows": covered + uncut[1:], "covered_tokens": 16}))
    assert got.status == "wrong" and list(got.evidence) == ["rows"]


def test_a_launch_row_is_graded_on_the_launch_values_and_not_on_its_rows():
    grade_fn = need_attr("baselines.grade", "grade")
    # The text of the table this reading rests on: the step-facts group reads "the first and
    # last step of a launch and the consumed-sample counter its first sample starts from",
    # and its rows are the places those are read from (the row's source).
    trace_row = _check10_rows()["c10.c01.step-facts.trace"]
    assert "the first and last step of a launch and the consumed-sample counter its first sample starts from" in trace_row["rule"]
    assert trace_row["expect"]["right"] == "every value read equals the listed row's"
    for listed in ("c01.K.warm-start", "c01.K.resume"):
        row = _listed_rows()[listed]
        assert "the rows themselves are the oracle's log" in row["source"], listed
        expect = _expect(listed)
        assert "rows" in expect
        launch = {key: expect[key] for key in ("consumed_at_start", "first_step", "last_step")}
        q = _make_question(listed, "trace")
        assert q.group == "c01.step-facts" and q.route is not None
        assert grade_fn(q, expect, _make_reading(q, dict(launch))).status == "right", listed
        # Rows the reading holds beside the values, and a diff, are not asked.
        beside = dict(launch, rows=[{"segment": "K/seg-1", "step": 99}], diff="3 rows differ")
        assert grade_fn(q, expect, _make_reading(q, beside)).status == "right", listed
        for key, value in launch.items():
            got = grade_fn(q, expect, _make_reading(q, dict(launch, **{key: _changed(value)})))
            assert got.status == "wrong" and list(got.evidence) == [key], (listed, key)
            lacking = {k: v for k, v in launch.items() if k != key}
            got = grade_fn(q, expect, _make_reading(q, lacking))
            assert got.status == "wrong" and list(got.evidence) == [key], (listed, key)


# --- The diff of a whole run is computed from the oracle's log --------------------------------

# Hand-written from the log_lines of the spec (positions.skipped_line, positions.T,
# positions.step8_cut and positions.K): the rows the join must report, by sample.
RUN_E_ROWS = {
    ("E/seg-1", 1, 0): [(1, 17, 21, 50)],
    ("E/seg-2", 7, 0): [(2, 24, 32, 39), (2, 7, 0, 22), (2, 26, 0, 4)],
    # The last sample of step 8 covers its first 16 tokens: the first row is cut at token 35
    # and the second row is not reported.
    ("E/seg-2", 8, 3): [(2, 2, 19, 35)],
}
RUN_K_ROWS = {
    ("K/seg-1", 15, 0): [(1, 0, 14, 22)],
    ("K/seg-1", 1, 0): [(0, 0, 11, 31)],
    ("K/seg-1", 8, 3): [(0, 25, 45, 51), (0, 13, 0, 18), (0, 7, 0, 9)],
}
RUN_RR_LOG = [
    {"segment": "E-rr/seg-1", "step": 2, "sample_pos": 1, "global_index": 5, "rank": 0,
     "shard_position": 1, "ordinal": 4, "token_start": 0, "token_end": 33},
    {"segment": "E-rr/seg-2", "step": 7, "sample_pos": 0, "global_index": 24, "rank": 0,
     "shard_position": 2, "ordinal": 9, "token_start": 3, "token_end": 12},
]
RUN_RR_ROWS = {
    ("E-rr/seg-1", 2, 1): [(1, 4, 0, 33)],
    ("E-rr/seg-2", 7, 0): [(2, 9, 3, 12)],
}


def _oracle_lines(*names):
    out = []
    for name in names:
        out += [json.loads(line) for line in _spec()["positions"][name]["log_lines"]]
    return out


def _reported(rows_by_sample):
    """The rows a subject reports for a run: one mapping per row, as the join writes them."""
    return [
        {"segment": segment, "step": step, "sample_pos": pos, "shard_position": shard,
         "ordinal": ordinal, "token_start": start, "token_end": end}
        for (segment, step, pos), rows in rows_by_sample.items()
        for (shard, ordinal, start, end) in rows
    ]


def test_the_diff_of_a_whole_run_is_computed_from_the_oracles_log_and_never_taken_from_the_subject():
    # Changed bar: the old test took the diff a reading states ("empty" or "3 rows differ");
    # the grader now computes it from the oracle's log.
    grade_run = need_attr("baselines.grade", "grade_whole_run")
    run_e_log = _oracle_lines("skipped_line", "T", "step8_cut")
    cases = (
        ("c01.E.diff", run_e_log + _oracle_lines("K"), RUN_E_ROWS),
        ("c01.K.diff", _oracle_lines("K", "T"), RUN_K_ROWS),
        ("c01.E-rr.diff", RUN_RR_LOG + _oracle_lines("K"), RUN_RR_ROWS),
    )
    for listed, oracle, by_sample in cases:
        expect = _expect(listed)
        assert expect["diff"] == "empty"
        q = _make_question(listed, "trace")
        assert q.route is not None
        rows = _reported(by_sample)
        got = grade_run(q, _make_reading(q, {"rows": rows}), oracle)
        assert got.status == "right" and got.evidence is None, listed
        # The subject's own diff is never read: the same rows with any diff are right,
        # and wrong rows with the diff "empty" are wrong.
        said = {"rows": rows, "diff": "3 rows differ"}
        assert grade_run(q, _make_reading(q, said), oracle).status == "right", listed
        changed = [dict(row) for row in rows]
        changed[0]["token_end"] += 1
        wrong = grade_run(q, _make_reading(q, {"rows": changed, "diff": "empty"}), oracle)
        assert wrong.status == "wrong" and list(wrong.evidence) == ["diff"], listed
        for bad in (rows[1:], rows + [rows[0]]):
            got = grade_run(q, _make_reading(q, {"rows": bad, "diff": "empty"}), oracle)
            assert got.status == "wrong" and list(got.evidence) == ["diff"], listed
        # A subject that gives a diff and no rows has said nothing the grader can compare.
        got = grade_run(q, _make_reading(q, {"diff": "empty"}), oracle)
        assert got.status == "wrong" and list(got.evidence) == ["diff"], listed


def test_a_whole_run_reading_holds_the_cut_of_step_8_and_the_row_of_the_oracle_uncut_is_wrong():
    grade_run = need_attr("baselines.grade", "grade_whole_run")
    oracle = _oracle_lines("skipped_line", "T", "step8_cut")
    q = _make_question("c01.E.diff", "trace")
    uncut = dict(RUN_E_ROWS)
    uncut[("E/seg-2", 8, 3)] = [(2, 2, 19, 44), (2, 23, 0, 8)]
    got = grade_run(q, _make_reading(q, {"rows": _reported(uncut)}), oracle)
    assert got.status == "wrong" and list(got.evidence) == ["diff"]
    # A row of another run's log is not asked: the log of two runs is split by run.
    both = oracle + _oracle_lines("K")
    assert grade_run(q, _make_reading(q, {"rows": _reported(RUN_E_ROWS)}), both).status == "right"


def test_the_whole_run_grader_refuses_what_it_cannot_grade_and_grade_does_not_take_the_subjects_diff():
    grade_run = need_attr("baselines.grade", "grade_whole_run")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    oracle = _oracle_lines("K")
    q = _make_question("c01.K.diff", "trace")
    rows = _reported(RUN_K_ROWS)
    with pytest.raises(not_gradable):
        grade_run(q, None, oracle)
    with pytest.raises(not_gradable):
        grade_run(q, _make_attempt(), oracle)
    other = _make_question("c01.K.diff", "dolma+trace")
    with pytest.raises(not_gradable):
        grade_run(q, _make_reading(other, {"rows": rows}), oracle)
    # A row that is not a whole-run diff row, and a question that holds no record.
    positions = _make_question("c01.E.T", "trace")
    with pytest.raises(ValueError):
        grade_run(positions, _make_reading(positions, {"rows": rows}), oracle)
    none_q = _make_question("c01.K.diff", "manifest")
    assert none_q.route is None
    with pytest.raises(ValueError):
        grade_run(none_q, _make_reading(none_q, {"rows": rows}), oracle)
    # grade() has no oracle's log: it does not take a diff the reading states for these rows.
    expect = _expect("c01.K.diff")
    with pytest.raises(not_gradable):
        _grade(q, expect, _make_reading(q, {"diff": "empty"}))


# --- The other forms of an expectation that a subject has a route to ---------------------
# counts, actions, labels of documents and the diff of a whole run. For each, a reading
# that holds what the row holds is right, one field that differs is wrong and named, and
# a field the reading lacks is wrong (never an absence). The rows are those a subject
# has a route to; the values are the table's, and the readings are written by hand.

COUNTS_ROW = "c09.fixture.shards.counts"
ACTION_ROW = "c09.fixture.R1.action"
ACTION_PLUS_ROW = "c09.fixture.E/shard-3.action"
KEPT_ROW = "c09.fixture.R2.action"
LABELS_ROW = "c06.fixture.documents-mapped.E/shard-1"


def _reads_right_and_wrong(grade_fn, question, expect, right_raw, wrong_raw, field):
    """True when ``grade_fn`` gives right to ``right_raw`` and wrong to ``wrong_raw``,
    naming ``field``. The checker used for the real grader and for a known-bad one."""
    right = grade_fn(question, expect, _make_reading(question, right_raw))
    wrong = grade_fn(question, expect, _make_reading(question, wrong_raw))
    return (
        right.status == "right"
        and wrong.status == "wrong"
        and list(wrong.evidence) == [field]
    )


def _answer_only_grader(question, expect, outcome):
    """A known-bad grader: it looks at the field ``answer`` and at nothing else, so a
    row that holds other fields is right whatever they say."""
    grade_type = need_attr("baselines.interfaces", "Grade")
    same = expect.get("answer") == outcome.raw.get("answer")
    return grade_type(status="right" if same else "wrong", evidence=None if same else ["answer"])


def test_counts_are_compared_with_what_the_row_holds_and_a_missing_count_is_wrong():
    expect = _expect(COUNTS_ROW)
    assert expect["counts"] == {"keep": 2, "rewrite": 1}
    q = _make_question(COUNTS_ROW, "manifest")
    right = {"counts": {"keep": 2, "rewrite": 1}, "keep": ["E/shard-1", "E/shard-2"],
             "rewrite": ["E/shard-3"]}
    assert _grade(q, expect, _make_reading(q, right)).status == "right"
    wrong = dict(right, counts={"keep": 3, "rewrite": 0})
    got = _grade(q, expect, _make_reading(q, wrong))
    assert got.status == "wrong" and list(got.evidence) == ["counts"]
    wrong_list = dict(right, rewrite=["E/shard-1"])
    got = _grade(q, expect, _make_reading(q, wrong_list))
    assert got.status == "wrong" and list(got.evidence) == ["rewrite"]
    gap = {key: value for key, value in right.items() if key != "counts"}
    got = _grade(q, expect, _make_reading(q, gap))
    assert got.status == "wrong" and list(got.evidence) == ["counts"]


def test_an_action_is_compared_and_what_goes_with_it_is_not_read():
    # Changed bar (the name and the field loop): the rule of the manifest's action rows reads
    # "Byte identity afterwards, the records removed and the size of a successor are not
    # read", so only the action (the list or the number the row names) is compared.
    grade_fn = need_attr("baselines.grade", "grade")
    rule = _check10_rows()["c10.c09.artifact-actions.manifest"]["rule"]
    assert "Byte identity afterwards, the records removed and the size of a successor are not read" in rule
    expect = _expect(ACTION_ROW)
    q = _make_question(ACTION_ROW, "manifest")
    assert _reads_right_and_wrong(
        grade_fn, q, expect, {"action": "rewrite"}, {"action": "keep"}, "action"
    )
    plus = _expect(ACTION_PLUS_ROW)
    pq = _make_question(ACTION_PLUS_ROW, "manifest")
    full = {"action": "rewrite", "records_removed": ["T"], "documents_in_successor": 26}
    assert _grade(pq, plus, _make_reading(pq, full)).status == "right"
    # The size of the successor and the records removed are not read: any value, or none.
    for other in (dict(full, documents_in_successor=27), dict(full, records_removed=[]),
                  {"action": "rewrite"}):
        got = _grade(pq, plus, _make_reading(pq, other))
        assert got.status == "right" and got.evidence is None, other
    # The action is compared.
    assert _reads_right_and_wrong(grade_fn, pq, plus, full, dict(full, action="keep"), "action")
    # Byte identity afterwards is not read either, said or left out.
    kept = _expect(KEPT_ROW)
    assert kept["byte_identical_afterwards"] is True
    kq = _make_question(KEPT_ROW, "manifest")
    for other in ({"action": "keep", "byte_identical_afterwards": True},
                  {"action": "keep", "byte_identical_afterwards": False}, {"action": "keep"}):
        got = _grade(kq, kept, _make_reading(kq, other))
        assert got.status == "right" and got.evidence is None, other
    got = _grade(kq, kept, _make_reading(kq, {"action": "rewrite", "byte_identical_afterwards": True}))
    assert got.status == "wrong" and list(got.evidence) == ["action"]


def test_the_documents_of_a_shard_are_compared_on_the_counts_the_rule_names():
    # Changed bar (the name, the derivation case and the gap case): the rule of the row
    # names, per shard, the number of documents, the number with exactly one tied row and the
    # number with none; the derivation and the verification are not in that reading and are
    # not compared. The key ``documents`` holds the first, ``recovered`` the second (the
    # listed row holds recovered = documents here) and ``unknown`` the third.
    grade_fn = need_attr("baselines.grade", "grade")
    rule = _check10_rows()["c10.c06.documents-mapped.dolma"]["rule"]
    assert rule.startswith(
        "the reading is, per shard, the number of documents, the number with exactly one "
        "tied row and the number with none"
    )
    assert "the derivation mode of the listed rows are not compared" in rule
    expect = _expect(LABELS_ROW)
    assert expect["recovered"] == expect["documents"] and expect["unknown"] == 0
    q = _make_question(LABELS_ROW, "dolma")
    right = {"documents": 18, "recovered": 18, "candidates": 0, "unknown": 0,
             "derivation": "position", "verification": "passed"}
    assert _grade(q, expect, _make_reading(q, right)).status == "right"
    assert _reads_right_and_wrong(grade_fn, q, expect, right, dict(right, unknown=1), "unknown")
    # The derivation and the verification are not compared: another word, or none, is right.
    # The number of candidates is no count the rule names (only the documents, the ones with
    # exactly one tied row and the ones with none): another number, or none, is right too.
    for other in (dict(right, derivation="token-matching"), dict(right, verification="failed"),
                  dict(right, candidates=5), {k: v for k, v in right.items() if k != "candidates"},
                  {k: v for k, v in right.items() if k not in ("derivation", "verification")}):
        got = _grade(q, expect, _make_reading(q, other))
        assert got.status == "right" and got.evidence is None, other
    # The three counts the rule names are compared, and a count the reading lacks is a gap.
    for key in ("documents", "recovered", "unknown"):
        assert _reads_right_and_wrong(grade_fn, q, expect, right, dict(right, **{key: right[key] + 1}), key)
        gap = {k: v for k, v in right.items() if k != key}
        got = _grade(q, expect, _make_reading(q, gap))
        assert got.status == "wrong" and list(got.evidence) == [key], key


def test_a_plan_that_allows_a_set_of_outcomes_reads_no_bare_outcome_from_a_reading():
    # Changed bar (the name and every assertion): hand-read from the row, the plan allows two
    # outcomes and forbids two others, and the grade is right ONLY from the subject's own
    # records. A reading that names an outcome under the key "outcome" and holds no records
    # is wrong, whatever outcome it names: grade() reads no bare outcome.
    for listed in ("c04.kill-after-100", "c04.retry"):
        expect = _expect(listed)
        allowed, forbidden = list(expect["one_of"]), list(expect["never"])
        assert len(allowed) == 2 and len(forbidden) == 2
        q = _make_question(listed, "dolma")
        for outcome in (*allowed, *forbidden, "a third outcome"):
            got = _grade(q, expect, _make_reading(q, {"outcome": outcome}))
            assert got.status == "wrong" and list(got.evidence) == ["records"], (listed, outcome)
        # A reading that names no outcome and holds no records is the same gap.
        gap = _grade(q, expect, _make_reading(q, {"note": "no outcome named"}))
        assert gap.status == "wrong" and list(gap.evidence) == ["records"]


# The cases of the two rows that allow a set of outcomes, each (records, status, evidence):
# hand-read from the plan's words in the row (check04.toml) and the rule of the group (the
# number of rows or entries after the stop; after the second run, whether the ranges index the
# documents of the token file and whether the tool refused instead).
KILL_CASES = (
    ({"rows_after_stop": 0}, "right", None),
    ({"rows_after_stop": 500}, "right", None),
    ({"rows_after_stop": 1}, "wrong", ["never"]),
    ({"rows_after_stop": 100}, "wrong", ["never"]),
    ({"rows_after_stop": 499}, "wrong", ["never"]),
    ({"rows_after_stop": 501}, "wrong", ["one_of"]),
    ({}, "wrong", ["records"]),
    ({"rows_after_stop": "500"}, "wrong", ["records"]),
)
RETRY_CASES = (
    ({"rows_after_retry": 500, "refused": False, "matches_shard": True}, "right", None),
    ({"rows_after_retry": 500, "refused": False, "matches_shard": False}, "wrong", ["never"]),
    ({"rows_after_retry": 300, "refused": False, "matches_shard": False}, "wrong", ["never"]),
    ({"rows_after_retry": 0, "refused": True, "mismatch": "declared count mismatch"}, "right", None),
    ({"rows_after_retry": 0, "refused": True, "mismatch": "declared digest mismatch"}, "right", None),
    ({"rows_after_retry": 300, "refused": True, "matches_shard": True,
      "mismatch": "declared count mismatch"}, "wrong", ["never"]),
    ({"rows_after_retry": 1, "refused": True, "matches_shard": True}, "wrong", ["never"]),
    ({"rows_after_retry": 499, "refused": True, "matches_shard": True,
      "mismatch": "declared digest mismatch"}, "wrong", ["never"]),
    ({"refused": True}, "wrong", ["records"]),
    ({"rows_after_retry": 500}, "wrong", ["records"]),
)


def test_grade_reads_the_two_plan_rows_from_their_records_as_grade_from_records_does():
    # Changed bar (new): grade() reads raw["records"] for c04.kill-after-100 and c04.retry
    # through the one path of grade_from_records, so the two entry points give the same
    # grade for the same reading, and an outcome the reading names beside the records is
    # never taken.
    grade_fn = need_attr("baselines.grade", "grade")
    grade_records = need_attr("baselines.grade", "grade_from_records")
    for listed, cases in (("c04.kill-after-100", KILL_CASES), ("c04.retry", RETRY_CASES)):
        expect = _expect(listed)
        q = _make_question(listed, "dolma")
        for records, status, evidence in cases:
            reading = _make_reading(q, {"records": records})
            for entry in (grade_fn, grade_records):
                got = entry(q, expect, reading)
                assert (got.status, got.evidence) == (status, evidence), (listed, records)
        # An outcome named beside the records is not taken: the records decide.
        said = _make_reading(q, {"outcome": expect["one_of"][0], "records": cases[2][0]})
        for entry in (grade_fn, grade_records):
            assert entry(q, expect, said).status == cases[2][1], listed


def test_a_grader_that_looks_at_the_answer_alone_is_caught_by_the_same_checks():
    grade_fn = need_attr("baselines.grade", "grade")
    expect = _expect(ACTION_PLUS_ROW)
    q = _make_question(ACTION_PLUS_ROW, "manifest")
    full = {"action": "rewrite", "records_removed": ["T"], "documents_in_successor": 26}
    # Changed bar: the successor's size is not read (the rule of the manifest's action rows), so
    # the known-bad check can no longer use it: a reading that differs in the successor's size
    # is right, and the reading that differs in a compared field is the action.
    got = _grade(q, expect, _make_reading(q, dict(full, documents_in_successor=27)))
    assert got.status == "right" and got.evidence is None
    wrong = dict(full, action="keep")
    assert _reads_right_and_wrong(grade_fn, q, expect, full, wrong, "action") is True
    assert _reads_right_and_wrong(_answer_only_grader, q, expect, full, wrong, "action") is False
    # The same known-bad grader on the counts and the labels (the diff of a whole run is
    # no longer a field of a reading: the grader computes it, see above).
    counts = {"counts": {"keep": 2, "rewrite": 1}, "keep": ["E/shard-1", "E/shard-2"],
              "rewrite": ["E/shard-3"]}
    cq = _make_question(COUNTS_ROW, "manifest")
    labels = {"documents": 18, "recovered": 18, "candidates": 0, "unknown": 0,
              "derivation": "position", "verification": "passed"}
    lq = _make_question(LABELS_ROW, "dolma")
    cases = (
        (cq, _expect(COUNTS_ROW), counts, dict(counts, counts={"keep": 3, "rewrite": 0}), "counts"),
        (lq, _expect(LABELS_ROW), labels, dict(labels, unknown=1), "unknown"),
    )
    for question, form_expect, right_raw, wrong_raw, field in cases:
        assert _reads_right_and_wrong(grade_fn, question, form_expect, right_raw, wrong_raw, field) is True
        assert _reads_right_and_wrong(
            _answer_only_grader, question, form_expect, right_raw, wrong_raw, field
        ) is False, field


# --- The other forms of the rows of Checks 4, 6 and 3 -----------------------------------------
# Where a subject has a route to the row (the routes are those of check10.toml), the reading
# is compared on the values the row's own text names.

STOP_ROWS = (
    ("c04.count-499", "dolma"),
    ("c04.edge-parent.R1", "manifest"),
    ("c04.edge-parent.gen/P", "manifest"),
    ("c04.trace-count.E@6", "dolma+trace"),
    ("c04.trace-count.E@9", "dolma+trace"),
    ("c04.artifact-digest.R2", "manifest"),
)
NOT_A_STOP_ROWS = (
    ("c04.not-a-stop.trace-deleted", "dolma+trace"),
    ("c04.not-a-stop.trace-half", "dolma+trace"),
)


def test_the_rows_of_check_4_that_expect_a_stop_are_right_only_as_a_stop_and_the_others_as_an_answer():
    grade_fn = need_attr("baselines.grade", "grade")
    key = need_attr("baselines.grade", "STOPPED_KEY")
    for listed, subject in STOP_ROWS:
        expect = _expect(listed)
        assert "error" in expect and "answer" not in expect, listed
        q = _make_question(listed, subject)
        assert q.route is not None, listed
        # The text of the table this rests on: the row's own wrong text says the reading is an
        # answer, so an answer word read where a stop is expected is wrong, and the field that
        # differs is the answer.
        group, _check = _group_of_listed()[listed]
        wrong_text = _check10_rows()[f"c10.{group}.{subject}"]["expect"]["wrong"]
        assert wrong_text.startswith("the reading is an answer"), listed
        assert grade_fn(q, expect, _make_reading(q, {key: True})).status == "right", listed
        got = grade_fn(q, expect, _make_reading(q, {"answer": "contains"}))
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
    for listed, subject in NOT_A_STOP_ROWS:
        expect = _expect(listed)
        assert expect["answer"] == "may contain" and expect["loud_stop"] is False, listed
        q = _make_question(listed, subject)
        assert q.route is not None, listed
        assert grade_fn(q, expect, _make_reading(q, {"answer": "may contain"})).status == "right"
        got = grade_fn(q, expect, _make_reading(q, {"answer": "contains"}))
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
        got = grade_fn(q, expect, _make_reading(q, {key: True}))
        assert got.status == "wrong" and "stopped" in list(got.evidence), listed


MANIFEST_STOP_ROWS = tuple(row for row in STOP_ROWS if row[1] == "manifest")
ANSWER_WORDS = ("contains", "does not contain", "no record")


def _every_manifest_answer_word_is_wrong(grade_function) -> bool:
    """True when ``grade_function`` gives wrong to each of the three answer words of the
    manifest, for every row of the manifest that expects "stopped", and names the answer as
    the field that differs."""
    for listed, subject in MANIFEST_STOP_ROWS:
        expect = _expect(listed)
        q = _make_question(listed, subject)
        for word in ANSWER_WORDS:
            got = grade_function(q, expect, _make_reading(q, {"answer": word}))
            if got.status != "wrong" or list(got.evidence or ()) != ["answer"]:
                return False
    return True


def _reads_no_record_as_a_stop(real_grade):
    """The known-bad grader: the manifest's answer "no record" is taken for a stop."""

    def bad(question, expect, outcome):
        raw = dict(outcome.raw)
        if raw.get("answer") == "no record":
            raw[need_attr("baselines.grade", "STOPPED_KEY")] = True
        return real_grade(question, expect, _make_reading(question, raw))

    return bad


def test_a_manifest_reading_is_wrong_where_the_row_expects_stopped():
    # The text of the table this rests on: the three rows expect "stopped" (an expectation
    # that holds an error), the rule of the group defines a stop as "a command of the subject
    # that reads the changed record ends with an error that reports it", and the rule of the
    # manifest says that its reading "has no check that reports it". So the reading of the manifest
    # (contains, does not contain, no record) is compared with the expected word like any
    # other answer word: it differs, so wrong, and the row's own wrong text ("the reading is an
    # answer") makes the answer the field that differs.
    real = need_attr("baselines.grade", "grade")
    assert len(MANIFEST_STOP_ROWS) == 3
    for listed, subject in MANIFEST_STOP_ROWS:
        group, _check = _group_of_listed()[listed]
        rule = _check10_rows()[f"c10.{group}.{subject}"]["rule"]
        assert "has no check that reports it" in rule, listed
        assert "stopped: a command of the subject that reads the changed record ends with an error that reports it" in rule, listed
        assert "error" in _expect(listed), listed
        written = _check10_rows()[f"c10.{group}.{subject}"]["expect"]["wrong"]
        assert written == "the reading is an answer", listed
    # The same check on the known-bad grader finds the fault, so it can fail.
    assert _every_manifest_answer_word_is_wrong(_reads_no_record_as_a_stop(real)) is False
    assert _every_manifest_answer_word_is_wrong(real) is True


def test_the_file_an_imports_list_names_is_compared_with_the_file_of_the_member_the_listed_row_gives():
    """The right text of the manifest's row: "the list names the file of the member the listed row gives, and no other
    file". The member is compared THROUGH ITS FILE: the file comes from the world's own entry (``member_files``, the
    files of the request whose source has that member), so no path shape is typed in the grader. A member with no
    file derived is not graded (the expected value is not derived). (This test replaces one that compared the member
    id: old, a reading {"link": {"member": "A/r1"}} was right and "B/r1" wrong; new, the files the list names are
    compared with the files of the member.)"""
    grade_fn = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    expect = _expect("c06.import.file-family")
    assert expect["link"]["member"] == "A/r1"
    q = _make_question("c06.import.file-family", "manifest")
    assert q.route is not None
    rule = _check10_rows()["c10.c06.import.file-family.manifest"]["rule"]
    assert "The label word and the evidence sentence of the listed row are not compared" in rule
    files = {"A/r1": ["sources/A.jsonl"]}

    def grade_with(raw, member_files=files):
        return grade_fn(q, expect, _make_reading(q, raw), member_files=member_files)

    assert grade_with({"link": {"files": ["sources/A.jsonl"]}}).status == "right"
    other_words = {"link": {"files": ["sources/A.jsonl"], "label": "another label", "evidence": "x"}}
    assert grade_with(other_words).status == "right", "the label and the evidence are not compared"
    for named in (["sources/B.jsonl"], ["sources/A.jsonl", "sources/B.jsonl"], []):
        got = grade_with({"link": {"files": named}})
        assert got.status == "wrong" and list(got.evidence) == ["link"], named
    gap = grade_with({})
    assert gap.status == "wrong" and list(gap.evidence) == ["link"]
    # The world derives the file of another member: the same reading is graded against that file.
    assert grade_with({"link": {"files": ["sources/B.jsonl"]}}, {"A/r1": ["sources/B.jsonl"]}).status == "right"
    # No file derived for the member: the row is not graded, never compared as a typed shape.
    for none_derived in (None, {}, {"B/r1": ["sources/B.jsonl"]}):
        with pytest.raises(not_gradable, match="A/r1"):
            grade_with({"link": {"files": ["sources/A.jsonl"]}}, none_derived)
    # The other five subjects have no record for this row: their line is cannot answer once the subject ran on the
    # store, as before.
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    for subject in ("trace", "dolma", "originblame", "dolma+trace", "originblame+trace"):
        other = _make_question("c06.import.file-family", subject)
        ran = attempt_type(subject=subject, target=("store", "import"), attempt_id=None, recorded="the tool ran")
        got = grade_fn(other, expect, _make_reading(other, None), attempts=[ran])
        assert other.route is None and got.status == "cannot answer", subject


def test_the_listing_of_a_manifest_is_compared_by_membership_with_no_order_and_no_reason():
    """The right text of the manifest's row: "E/shard-1 is listed as no record, and every run, mixture, shard,
    checkpoint and release of the listed row's covers is listed"; the wrong text: "E/shard-1 is left out or listed
    with an answer, or an artifact of the covers is left out". It names no order and no reason. Old: ``covers`` and
    ``not_covered`` were compared as written, in order, with the nested reason; new: membership."""
    grade_fn = need_attr("baselines.grade", "grade")
    expect = _expect("c05.listing.map-deleted")
    q = _make_question("c05.listing.map-deleted", "manifest")
    assert q.route is not None
    text = _check10_rows()["c10.c05.listing.map-deleted.manifest"]["expect"]
    assert "listed as no record" in text["right"] and "left out" in text["wrong"]
    covers = {kind: list(names) for kind, names in expect["covers"].items()}
    assert expect["not_covered"][0]["artifact"] == "E/shard-1"

    def reading(covers_read, not_covered):
        return _make_reading(q, {"covers": covers_read, "not_covered": not_covered})

    right = reading(covers, [{"artifact": "E/shard-1"}])
    assert grade_fn(q, expect, right).status == "right"
    # No order, no reason, more kinds and more names are all the same listing.
    shuffled = {kind: list(reversed(names)) for kind, names in covers.items()}
    shuffled["sources"] = ["src/A", "src/B", "src/Z"]
    assert grade_fn(q, expect, reading(shuffled, [{"artifact": "E/shard-1", "reason": "no record"}])).status == "right"
    assert grade_fn(q, expect, reading(covers, ["E/shard-1"])).status == "right"
    # An artifact of the covers left out, or none of a kind: the covers are wrong.
    for kind in covers:
        cut = {k: (v[:-1] if k == kind else v) for k, v in covers.items()}
        got = grade_fn(q, expect, reading(cut, [{"artifact": "E/shard-1"}]))
        assert got.status == "wrong" and list(got.evidence) == ["covers"], kind
    # E/shard-1 left out of the no-record list, or listed with an answer (as a shard of the covers): wrong.
    got = grade_fn(q, expect, reading(covers, []))
    assert got.status == "wrong" and list(got.evidence) == ["not_covered"]
    answered = dict(covers, shards=[*covers["shards"], "E/shard-1"])
    got = grade_fn(q, expect, reading(answered, [{"artifact": "E/shard-1"}]))
    assert got.status == "wrong" and list(got.evidence) == ["not_covered"]
    # The lines of the five other subjects, which hold no record of this row, are cannot answer once the subject ran
    # on the store, as before.
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    for subject in ("trace", "dolma", "originblame", "dolma+trace", "originblame+trace"):
        other = _make_question("c05.listing.map-deleted", subject)
        ran = attempt_type(subject=subject, target=("store", "main"), attempt_id=None, recorded="the tool ran")
        got = grade_fn(other, expect, _make_reading(other, None), attempts=[ran])
        assert other.route is None and got.status == "cannot answer", subject


def test_a_listing_of_the_rebuild_and_a_restart_are_compared_value_by_value():
    # The report forms a subject has a route to: the listing of the mixtures that need a
    # successor and the exposure-free restart (the report rows of Check 7 and the counts of
    # Check 9 have no route for any subject, so they are not asked here).
    grade_fn = need_attr("baselines.grade", "grade")
    expect = _expect("c09.fixture.mixtures")
    q = _make_question("c09.fixture.mixtures", "manifest")
    assert q.route is not None
    right = {"needs_successor": list(expect["needs_successor"]), "kept": list(expect["kept"])}
    assert grade_fn(q, expect, _make_reading(q, right)).status == "right"
    for key in ("needs_successor", "kept"):
        got = grade_fn(q, expect, _make_reading(q, dict(right, **{key: right[key][:-1]})))
        assert got.status == "wrong" and list(got.evidence) == [key], key
        lacking = {k: v for k, v in right.items() if k != key}
        got = grade_fn(q, expect, _make_reading(q, lacking))
        assert got.status == "wrong" and list(got.evidence) == [key], key
    restart = _expect("c09.E.exposure-free-restart")
    rq = _make_question("c09.E.exposure-free-restart", "manifest")
    assert rq.route is not None and restart["exposure_free_restart"] == "E@6"
    assert grade_fn(rq, restart, _make_reading(rq, {"exposure_free_restart": "E@6"})).status == "right"
    got = grade_fn(rq, restart, _make_reading(rq, {"exposure_free_restart": "E@9"}))
    assert got.status == "wrong" and list(got.evidence) == ["exposure_free_restart"]


def test_the_two_lists_of_the_mixtures_are_each_compared_as_a_set():
    # The text of the table this rests on: the right text of the manifest's row reads "the two
    # lists read equal the listed row's, each compared as a set", and the source of the listed
    # row says "Each list is compared as a set".
    grade_fn = need_attr("baselines.grade", "grade")
    row = _check10_rows()["c10.c09.mixture-successors.manifest"]
    assert "each compared as a set" in row["expect"]["right"]
    assert "Each list is compared as a set" in _listed_rows()["c09.fixture.mixtures"]["source"]
    expect = _expect("c09.fixture.mixtures")
    q = _make_question("c09.fixture.mixtures", "manifest")
    assert q.route is not None
    right = {"needs_successor": list(expect["needs_successor"]), "kept": list(expect["kept"])}
    assert len(right["needs_successor"]) == 3 and len(right["kept"]) == 2
    assert grade_fn(q, expect, _make_reading(q, right)).status == "right"
    # The order of a list is not read: each list, read in another order, is right.
    for key in ("needs_successor", "kept"):
        for other in (list(reversed(right[key])), right[key][1:] + right[key][:1]):
            got = grade_fn(q, expect, _make_reading(q, dict(right, **{key: other})))
            assert got.status == "right" and got.evidence is None, (key, other)
    both = {key: list(reversed(items)) for key, items in right.items()}
    assert grade_fn(q, expect, _make_reading(q, both)).status == "right"
    # A different set is wrong, and the list that differs is the one named: a member missing,
    # a member of the other list, a member that is no mixture of the row.
    for key, other_key in (("needs_successor", "kept"), ("kept", "needs_successor")):
        for items in (right[key][:-1], right[key] + [right[other_key][0]], right[key][:-1] + ["Z/mix-9"]):
            got = grade_fn(q, expect, _make_reading(q, dict(right, **{key: items})))
            assert got.status == "wrong" and list(got.evidence) == [key], (key, items)


def test_a_query_naming_a_checkpoint_never_registered_is_graded_on_its_answer_and_not_its_reason():
    # The registration form: the listed row expects the answer "not covered" with the reason
    # "not registered". The text of the table this rests on: the rule of the manifest and of
    # the trace says "The reason beside the answer of the listed row (not registered) is not
    # compared."
    grade_fn = need_attr("baselines.grade", "grade")
    listed = "c05.unregistered-checkpoint"
    expect = _expect(listed)
    assert expect == {"answer": "not covered", "reason": "not registered"}
    for subject in ("manifest", "trace"):
        rule = _check10_rows()[f"c10.c05.unregistered-checkpoint.{subject}"]["rule"]
        assert "The reason beside the answer of the listed row (not registered) is not compared." in rule
    for subject in ("manifest", "trace", "dolma+trace", "originblame+trace"):
        q = _make_question(listed, subject)
        assert q.route is not None, subject
        right = {"answer": "not covered"}
        assert grade_fn(q, expect, _make_reading(q, right)).status == "right", subject
        other_reason = {"answer": "not covered", "reason": "record missing"}
        assert grade_fn(q, expect, _make_reading(q, other_reason)).status == "right", subject
        # An answer where none is expected is wrong, and it is the answer that is named.
        got = grade_fn(q, expect, _make_reading(q, {"answer": "may contain", "reason": "not registered"}))
        assert got.status == "wrong" and list(got.evidence) == ["answer"], subject
        # A reason with no answer is a gap.
        got = grade_fn(q, expect, _make_reading(q, {"reason": "not registered"}))
        assert got.status == "wrong" and list(got.evidence) == ["answer"], subject


# The rows of the rebuild report and of registering again. No subject has a route to them (the
# whole answer is a field of the rebuild report, or of an identity the subjects lack): the rule
# of every subject's row of their group is "cannot answer; the evidence is the missing record or
# field". A question given a route for a test is graded by what the rules say, so it is graded
# cannot answer, whatever the reading holds.
SUBJECTS_OF_THE_TABLE = (
    "manifest", "trace", "dolma", "originblame", "dolma+trace", "originblame+trace",
)


def _every_subjects_rule_is_cannot_answer(listed_row) -> bool:
    """True when each of the six subjects' rows of the group of ``listed_row`` holds no
    record and its rule starts "cannot answer" (read from the table's own rows)."""
    group, _check = _group_of_listed()[listed_row]
    for subject in SUBJECTS_OF_THE_TABLE:
        row = _check10_rows()[f"c10.{group}.{subject}"]
        if row.get("route") is not None or not row["rule"].startswith("cannot answer"):
            return False
    return True


def _given_route_problems(call, listed_row, raws, subject="manifest", size=None) -> list[str]:
    """What is wrong where a question that is given a route (nothing else of it changed) is
    graded by ``call(question, reading, attempts)``: each reading in ``raws`` must be graded
    cannot answer, the evidence the question's own missing text. The subject's attempt on the
    store or set of the row is recorded, as it is for a line that holds no record."""
    question = _make_question(listed_row, subject, size, route="a route given for this test")
    assert question.missing, listed_row
    attempts = _attempts_for(listed_row, subject, size)
    found = []
    for raw in raws:
        try:
            got = call(question, _make_reading(question, raw), attempts)
        except ValueError as problem:
            found.append(f"{listed_row} {subject}: {type(problem).__name__}: {problem}")
            continue
        if got.status != "cannot answer" or got.evidence != question.missing:
            found.append(f"{listed_row} {subject}: {got.status}, {got.evidence!r}")
    return found


REPORT_FORM_ROWS = (
    "c07.report.R1.X",
    "c07.report.R1.P",
    "c07.report.R1.U",
    "c07.swap.report.counts",
    "c07.reregister.same-id",
)


def test_a_report_row_and_registering_again_are_graded_cannot_answer_on_any_route():
    # Changed bar (the name and every assertion): the fields of these rows were compared one
    # by one on a question given a route. No subject's rule names a value of them, so such a
    # question is graded cannot answer, with the missing text as its evidence, whatever the
    # reading holds: the right values, one value changed, one value left out, or nothing.
    grade_fn = need_attr("baselines.grade", "grade")
    problems = []
    for listed in REPORT_FORM_ROWS:
        assert _every_subjects_rule_is_cannot_answer(listed), listed
        expect = _expect(listed)
        assert _make_question(listed, "manifest").route is None, listed
        raws = [dict(expect), {}]
        for key, value in expect.items():
            raws.append(dict(expect, **{key: _changed(value)}))
            raws.append({k: v for k, v in expect.items() if k != key})

        def call(question, reading, attempts, expect=expect):
            return grade_fn(question, expect, reading, attempts=attempts)

        for subject in ("manifest", "dolma"):
            problems += _given_route_problems(call, listed, raws, subject)
    assert not problems, "\n".join(problems)


# Hand-written in the shape of the counting script's document (docs/plant.md, section 7).
def _pair_of(pair_id, placement, members):
    return {"pair": pair_id, "kind": "pair-member", "placement": placement, "members": members}


def _member_of(record, shard, line):
    return {"id": record, "source": "Z", "shard": shard, "line": line, "ordinal": line - 1,
            "expected_label": "candidates"}


COUNTED = {
    "check3": {"pairs": {"list": [
        _pair_of("pair-1", "same-shard", [_member_of("Z-102", "shard-1", 2),
                                          _member_of("Z-101", "shard-1", 4)]),
        _pair_of("pair-2", "split", [_member_of("Z-201", "shard-1", 5),
                                     _member_of("Z-202", "shard-2", 2)]),
    ]}},
    "documents": {"total": 9, "shards": [
        {"shard": "shard-1", "documents": 5,
         "ids": ["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"],
         "identical": [{"pair": "pair-1", "text_sha256": "aa" * 32, "ordinals": [1, 3],
                        "lines": [2, 4], "ids": ["Z-102", "Z-101"]}],
         "labels": {"recovered": 3, "candidates": 2}},
        {"shard": "shard-2", "documents": 4,
         "ids": ["Z-301", "Z-302", "Z-006", "Z-303"],
         "identical": [{"pair": "pair-9", "text_sha256": "bb" * 32, "ordinals": [0, 1, 3],
                        "lines": [1, 2, 4], "ids": ["Z-301", "Z-302", "Z-303"]}],
         "labels": {"recovered": 1, "candidates": 3}},
    ]},
}
SPAN_SIZES = {"shard-1": [7, 22, 9], "shard-2": [30, 4]}


def _concrete_of(listed, size, **sources):
    """The concrete expectation of a row, from the expected module: the sentence-valued
    fields of the table replaced by the values their sources give."""
    concrete_values = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    return concrete_values(expected_for(listed, size, **sources))


def test_a_label_word_is_not_compared_where_the_question_also_asks_for_the_record():
    grade_fn = need_attr("baselines.grade", "grade")
    concrete = _concrete_of("c03.pairs.split", "small", counted={"small": COUNTED})
    expect = concrete.expect
    assert expect["label"] == "recovered"
    assert expect["record"] == [{"pair": "pair-2", "members": [
        {"id": "Z-201", "shard": "shard-1", "line": 5},
        {"id": "Z-202", "shard": "shard-2", "line": 2},
    ]}]
    q = _make_question("c03.pairs.split", "dolma", "small")
    # The text of the table this rests on: the row's rule for this subject.
    assert "The label word of the listed row is not compared" in _check10_rows()["c10.c03.pairs.split.dolma"]["rule"]
    reading = {"record": [dict(item) for item in expect["record"]]}
    assert grade_fn(q, expect, _make_reading(q, reading)).status == "right"
    assert grade_fn(q, expect, _make_reading(q, dict(reading, label="observed"))).status == "right"
    moved = [{"pair": "pair-2", "members": [
        {"id": "Z-201", "shard": "shard-1", "line": 6},
        {"id": "Z-202", "shard": "shard-2", "line": 2},
    ]}]
    got = grade_fn(q, expect, _make_reading(q, {"record": moved}))
    assert got.status == "wrong" and list(got.evidence) == ["record"]
    gap = grade_fn(q, expect, _make_reading(q, {"label": "recovered"}))
    assert gap.status == "wrong" and list(gap.evidence) == ["record"]


def test_the_member_id_of_a_split_pair_is_not_compared_only_its_chunk_file_and_line():
    # The text of the table this rests on: the rule of the toolkit's row reads "for each of the
    # two ordinals of a split pair, the chunk file and the line the tied row names", which names
    # no member id.
    grade_fn = need_attr("baselines.grade", "grade")
    rule = _check10_rows()["c10.c03.pairs.split.dolma"]["rule"]
    assert "the chunk file and the line the tied row names" in rule
    concrete = _concrete_of("c03.pairs.split", "small", counted={"small": COUNTED})
    expect = concrete.expect
    q = _make_question("c03.pairs.split", "dolma", "small")
    assert q.route is not None

    def reading_with(members):
        return {"record": [{"pair": "pair-2", "members": members}]}

    def verdict(members):
        return grade_fn(q, expect, _make_reading(q, reading_with(members)))

    right = [{"id": "Z-201", "shard": "shard-1", "line": 5},
             {"id": "Z-202", "shard": "shard-2", "line": 2}]
    assert verdict(right).status == "right"
    # The id of a member is not read: another id, or none, at the right chunk file and line.
    other_ids = [dict(member, id="Z-999") for member in right]
    no_ids = [{"shard": member["shard"], "line": member["line"]} for member in right]
    swapped = [dict(right[0], id="Z-202"), dict(right[1], id="Z-201")]
    for members in (other_ids, no_ids, swapped):
        got = verdict(members)
        assert got.status == "right" and got.evidence is None, members
    # The chunk file and the line are read, for each of the two ordinals.
    for index in (0, 1):
        for change in ({"shard": "shard-3"}, {"line": right[index]["line"] + 1}):
            members = [dict(member) for member in right]
            members[index].update(change)
            got = verdict(members)
            assert got.status == "wrong" and list(got.evidence) == ["record"], (index, change)


def test_the_map_built_by_word_of_the_fallback_is_not_compared_on_the_records_rows():
    # The text of the table this rests on: under F1 the capture row's expectation holds
    # map_built_by; no rule of the group c03.records names it.
    grade_fn = need_attr("baselines.grade", "grade")
    listed = "c03.records.capture"
    written = _listed_rows()[listed]["expect_under"]["F1"]
    assert written["map_built_by"] == "token-matching"
    for subject in ("dolma", "dolma+trace"):
        group, _check = _group_of_listed()[listed]
        assert "map_built_by" not in _check10_rows()[f"c10.{group}.{subject}"]["rule"]
    concrete = _concrete_of(listed, "small", counted={"small": COUNTED}, fallbacks=("F1",))
    assert concrete.expect["map_built_by"] == "token-matching"
    q = _make_question(listed, "dolma", "small")
    assert q.route is not None
    ids = {name: list(held) for name, held in GROUPS_READING.items()}
    for raw in ({"per_ordinal": ids},
                {"per_ordinal": ids, "map_built_by": "token-matching"},
                {"per_ordinal": ids, "map_built_by": "capture"}):
        got = grade_fn(q, concrete.expect, _make_reading(q, raw))
        assert got.status == "right" and got.evidence is None, raw
    # What the rule names is still compared: the ids.
    off = {name: list(held) for name, held in GROUPS_READING.items()}
    off["shard-2"][2] = "Z-007"
    got = grade_fn(q, concrete.expect, _make_reading(q, {"per_ordinal": off, "map_built_by": "capture"}))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]


DOCUMENT_COUNTS = {
    # Hand-counted from the counting document COUNTED: shard-1 has 5 documents and shard-2 has
    # 4; every document of a shard has exactly one tied row (a member of a group of identical
    # documents has its own companion row, tied by its own src and loc), so 5 and 4 have
    # exactly one tied row and none is left over.
    "documents_per_shard": {"shard-1": 5, "shard-2": 4},
    "documents_with_one_tied_row_per_shard": {"shard-1": 5, "shard-2": 4},
    "documents_with_no_tied_row_per_shard": {"shard-1": 0, "shard-2": 0},
}


def test_the_documents_mapped_row_compares_each_per_shard_count_beside_the_ids():
    # The text of the table this rests on: the rule names, per shard, the number of documents,
    # the number with exactly one tied row and the number with none, beside the line each tied
    # row names.
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    listed = "c06.generated.documents-mapped"
    rule = _check10_rows()["c10.c06.documents-mapped.dolma"]["rule"]
    assert "the number of documents, the number with exactly one tied row and the number with none" in rule
    concrete = _concrete_of(listed, "small", counted={"small": COUNTED})
    assert dict(concrete.counted_values) == DOCUMENT_COUNTS
    q = _make_question(listed, "dolma", "small")
    assert q.route is not None
    ids = {name: list(held) for name, held in GROUPS_READING.items()}
    right = dict(DOCUMENT_COUNTS, per_ordinal=ids, unmapped_documents=0)
    assert grade_concrete(q, concrete, _make_reading(q, right)).status == "right"
    for key, held in DOCUMENT_COUNTS.items():
        # One shard's count changed: wrong, and the count that differs is the one named.
        changed = {**held, "shard-2": held["shard-2"] + 1}
        got = grade_concrete(q, concrete, _make_reading(q, dict(right, **{key: changed})))
        assert got.status == "wrong" and list(got.evidence) == [key], key
        # A count the reading lacks is a gap, and wrong.
        lacking = {k: v for k, v in right.items() if k != key}
        got = grade_concrete(q, concrete, _make_reading(q, lacking))
        assert got.status == "wrong" and list(got.evidence) == [key], key
    # The ids are read as before, and name only themselves.
    off = {name: list(held) for name, held in GROUPS_READING.items()}
    off["shard-2"][2] = "Z-007"
    got = grade_concrete(q, concrete, _make_reading(q, dict(right, per_ordinal=off)))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]


# --- The planted values and the index feed the grade ----------------------------------------


def test_the_concrete_value_of_a_sentence_valued_field_is_what_the_reading_is_compared_with():
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    concrete = _concrete_of("c03.spans.per-ordinal", "small", index={"small": SPAN_SIZES})
    assert dict(concrete.expect["per_ordinal"]) == SPAN_SIZES
    q = _make_question("c03.spans.per-ordinal", "dolma", "small")
    assert q.route is not None
    # The text of the table this rests on: right is "the two numbers are equal for every
    # ordinal of every shard", so the count of mismatches the row holds is the number of
    # ordinals that differ, worked out from the spans and not a value the reading holds.
    right_text = _check10_rows()["c10.c03.spans.dolma"]["expect"]["right"]
    assert right_text.startswith("the two numbers are equal for every ordinal of every shard")
    sentence = _listed_rows()["c03.spans.per-ordinal"]["expect"]["per_ordinal"]
    assert isinstance(sentence, str)
    right = {"per_ordinal": {shard: list(sizes) for shard, sizes in SPAN_SIZES.items()}}
    assert grade_concrete(q, concrete, _make_reading(q, right)).status == "right"
    # The count of mismatches is the grader's to work out from the spans, not the reading's.
    said = dict(right, mismatches=0)
    assert grade_concrete(q, concrete, _make_reading(q, said)).status == "right"
    off = {"per_ordinal": {"shard-1": [7, 22, 10], "shard-2": [30, 4]}, "mismatches": 0}
    got = grade_concrete(q, concrete, _make_reading(q, off))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]
    # The table's sentence is not a value: a reading that holds it is wrong.
    got = grade_concrete(q, concrete, _make_reading(q, {"per_ordinal": sentence}))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]


def test_the_count_of_mismatches_beside_the_totals_of_the_shards_is_the_graders_to_work_out():
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    concrete = _concrete_of("c03.spans.total", "small", index={"small": SPAN_SIZES})
    # Hand-added from SPAN_SIZES: 7 + 22 + 9 and 30 + 4.
    assert dict(concrete.expect["per_shard"]) == {"shard-1": 38, "shard-2": 34}
    q = _make_question("c03.spans.total", "dolma", "small")
    assert q.route is not None
    # The text of the table this rests on: right is "the two totals are equal for every
    # shard", so the count of mismatches the row holds is the number of shards whose totals
    # differ, worked out from the totals and not a value the reading holds.
    right_text = _check10_rows()["c10.c03.spans.dolma"]["expect"]["right"]
    assert right_text.endswith("the two totals are equal for every shard")
    totals = {"per_shard": {"shard-1": 38, "shard-2": 34}}
    assert grade_concrete(q, concrete, _make_reading(q, totals)).status == "right"
    said = dict(totals, mismatches=0)
    assert grade_concrete(q, concrete, _make_reading(q, said)).status == "right"
    off = {"per_shard": {"shard-1": 38, "shard-2": 35}, "mismatches": 0}
    got = grade_concrete(q, concrete, _make_reading(q, off))
    assert got.status == "wrong" and list(got.evidence) == ["per_shard"]
    gap = grade_concrete(q, concrete, _make_reading(q, {"mismatches": 0}))
    assert gap.status == "wrong" and list(gap.evidence) == ["per_shard"]


def _takes_the_mismatches_the_reading_states(real_grade_concrete):
    """The known-bad grader: the count of mismatches the reading states is taken as the
    result (zero is right, anything else is wrong) in place of the totals."""

    def bad(question, concrete, outcome, **kwargs):
        raw = outcome.raw
        if "mismatches" in raw:
            if raw["mismatches"] == 0:
                return _make_grade("right")
            return _make_grade("wrong", ["mismatches"])
        return real_grade_concrete(question, concrete, outcome, **kwargs)

    return bad


def _mismatches_are_derived_from_the_totals(grade_function, concrete, question) -> bool:
    """True when the grade follows the per-shard totals alone: a count of mismatches the
    reading states, whatever it says, changes nothing."""
    good = {"per_shard": {"shard-1": 38, "shard-2": 34}}
    off = {"per_shard": {"shard-1": 38, "shard-2": 35}}
    for stated in (0, 1, 7):
        right = grade_function(question, concrete, _make_reading(question, dict(good, mismatches=stated)))
        wrong = grade_function(question, concrete, _make_reading(question, dict(off, mismatches=stated)))
        if right.status != "right" or wrong.status != "wrong" or list(wrong.evidence) != ["per_shard"]:
            return False
    return True


def test_the_totals_of_the_shards_are_compared_and_the_mismatches_are_derived_never_supplied():
    # The rule of c10.c03.spans.dolma: "the sum of end minus start over the rows whose src is
    # the chunk file, against the index total", per shard. The count of shards where that
    # comparison fails is derived from the same readings; it is not a field the subject
    # supplies, so a count the reading states, of any value, is not read.
    real = need_attr("baselines.grade", "grade_concrete")
    rule = _check10_rows()["c10.c03.spans.dolma"]["rule"]
    assert "c03.spans.total: the sum of end minus start over the rows whose src is the chunk file, against the index total." in rule
    concrete = _concrete_of("c03.spans.total", "small", index={"small": SPAN_SIZES})
    q = _make_question("c03.spans.total", "dolma", "small")
    # The same check on the known-bad grader finds the fault, so it can fail.
    assert _mismatches_are_derived_from_the_totals(
        _takes_the_mismatches_the_reading_states(real), concrete, q
    ) is False
    assert _mismatches_are_derived_from_the_totals(real, concrete, q) is True


def test_a_field_with_no_concrete_value_is_never_compared_as_a_string():
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    # A row with a route reaches the guard: the documents-mapped row asked with no counting
    # document holds the table's sentence for per_ordinal and no concrete value for the counts.
    unfilled = _concrete_of("c06.generated.documents-mapped", "small")
    assert "per_ordinal" in unfilled.open
    routed = _make_question("c06.generated.documents-mapped", "dolma", "small")
    assert routed.route is not None
    sentence_raw = {"per_ordinal": unfilled.expect["per_ordinal"], "unmapped_documents": 0}
    with pytest.raises(not_gradable, match="per_ordinal"):
        grade_concrete(routed, unfilled, _make_reading(routed, sentence_raw))
    # Changed bar (the row and the route): the declared-count row has a field with no concrete
    # value and no subject has a route to it (the rule of each is "cannot answer"). A question
    # given a route for a test is graded cannot answer, the table's sentence never compared
    # and never a refusal.
    concrete = _concrete_of("c04.declare-499.write", None)
    assert "map_after" in concrete.open
    assert _every_subjects_rule_is_cannot_answer("c04.declare-499.write")
    raw = {"stopped": True, "map_after": concrete.expect["map_after"], "rows_after": 0}

    def call(question, reading, attempts):
        return grade_concrete(question, concrete, reading, attempts=attempts)

    problems = _given_route_problems(call, "c04.declare-499.write", [raw, {}], "dolma")
    assert not problems, "\n".join(problems)
    # A line that holds no record needs no expected value: it is cannot answer once the
    # subject's attempt on its store is recorded, whatever the row holds.
    none_q = _make_question("c04.declare-499.write", "dolma")
    assert none_q.route is None
    ran = attempt_type(subject="dolma", target=("store", "scratch-500"), attempt_id=None,
                       recorded="the tool ran on the store")
    got = grade_concrete(none_q, concrete, _make_reading(none_q, None), attempts=[ran])
    assert got.status == "cannot answer" and got.evidence == none_q.missing


# The fields of the table that hold a sentence where a value stands (hand-read from the
# check files; each is a string in its row). A reading that holds the sentence itself must not
# make a row right, whichever way the expectation reaches the grader.
SENTENCE_FIELDS = (
    ("c02.ii.replay-rejected", "digests"),
    ("c03.spans.per-ordinal", "per_ordinal"),
    ("c03.spans.total", "per_shard"),
    ("c03.drops.sanitizer", "drops"),
    ("c03.drops.sanitizer", "documents_per_shard"),
    ("c03.records.capture", "per_ordinal"),
    ("c03.records.position-recovery", "per_ordinal"),
    ("c03.records.token-matching", "per_ordinal"),
    ("c03.pairs.same-shard", "candidate_set"),
    ("c03.pairs.split", "record"),
    ("c04.declare-499.write", "map_after"),
    ("c06.generated.documents-mapped", "per_ordinal"),
    ("c06.generated.drops", "drops"),
    ("c06.generated.candidates", "candidate_sources"),
    ("c06.import.review-scope", "missing_link"),
    ("c08.K.digests-match", "shuffle_digests"),
    ("c08.K.digests-match", "digest_covers"),
    ("c08.K.digests-match", "epoch_0_digests_of_launch_0_and_launch_1"),
    ("c09.10k.action-counts", "counts"),
    ("c09.10k.partitioned.kept-shards", "keep"),
    ("c09.E.stale-cache.E/seg-2", "rebuilt_from"),
)


# The listed rows of SENTENCE_FIELDS that a subject has a route to, with that subject (read from
# the table's own rows below). Every other listed row of the tuple has no subject with a route:
# the rule of each of the six subjects' rows is "cannot answer". The two fields that stay as the
# table wrote them because they are not compared are read apart, in the test of the not-compared
# fields.
ROUTED_SENTENCE_ROWS = {
    "c03.spans.per-ordinal": "dolma",
    "c03.spans.total": "dolma",
    "c03.records.capture": "dolma",
    "c03.records.position-recovery": "dolma",
    "c03.records.token-matching": "dolma",
    "c03.pairs.split": "dolma",
    "c06.generated.documents-mapped": "dolma",
    "c09.10k.partitioned.kept-shards": "manifest",
}
NOT_COMPARED_FIELDS = (
    ("c06.import.review-scope", "missing_link"),
    ("c08.K.digests-match", "digest_covers"),
)


def test_grade_does_not_compare_a_sentence_of_the_table_as_a_string_either():
    # The refusal of grade_concrete is not the only one: an expectation that reaches grade()
    # as the table wrote it, with the sentence still in a field, is not graded (NotGradable
    # naming the field), so that a reading that holds the sentence cannot be right. That is
    # asked of a row a subject has a route to, with its own route.
    not_gradable = need_attr("baselines.grade", "NotGradable")
    grade_fn = need_attr("baselines.grade", "grade")
    for listed, field in SENTENCE_FIELDS:
        if listed not in ROUTED_SENTENCE_ROWS:
            continue
        subject = ROUTED_SENTENCE_ROWS[listed]
        group, _check = _group_of_listed()[listed]
        assert _check10_rows()[f"c10.{group}.{subject}"].get("route") is not None, listed
        expect = _expect(listed)
        assert isinstance(expect[field], str), (listed, field)
        q = _make_question(listed, subject)
        raw = {key: value for key, value in expect.items()}
        with pytest.raises(not_gradable, match=field):
            _grade(q, expect, _make_reading(q, raw))
    # Changed bar (the rows and the route): every other row of the tuple has no subject with a
    # route, and a question given a route for a test used to be refused for its sentence. It is
    # graded by what the rules say: cannot answer, the missing text as its evidence, whatever
    # the reading holds, the sentence itself included.
    problems = []
    for listed, field in SENTENCE_FIELDS:
        if listed in ROUTED_SENTENCE_ROWS or (listed, field) in NOT_COMPARED_FIELDS:
            continue
        assert _every_subjects_rule_is_cannot_answer(listed), listed
        expect = _expect(listed)
        assert isinstance(expect[field], str), (listed, field)

        def call(question, reading, attempts, expect=expect):
            return grade_fn(question, expect, reading, attempts=attempts)

        raws = [{key: value for key, value in expect.items()}, {}]
        problems += _given_route_problems(call, listed, raws, "dolma")
    assert not problems, "\n".join(problems)


def test_a_field_the_concrete_lists_as_not_compared_raises_no_refusal_and_is_not_compared():
    # The two fields that stay as the table wrote them because they are not compared (the
    # failing link of the import row, the scope of a shuffle digest) were refused as a sentence
    # in the loop of the test above. A Concrete names them in not_compared, and a field named
    # there raises no NotGradable. No subject has a route to either row (the rule of each is
    # "cannot answer"), so a question given a route is graded cannot answer; under the cyclic
    # wrapper route the expectation of the digests row is the one word "none recorded", also
    # graded so.
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    problems = []
    for listed, field in NOT_COMPARED_FIELDS:
        assert _every_subjects_rule_is_cannot_answer(listed), listed
        concrete = _concrete_of(listed, None)
        assert field in concrete.not_compared, (listed, field)
        assert field not in concrete.open, (listed, field)
        assert isinstance(concrete.expect[field], str), (listed, field)

        def call(question, reading, attempts, concrete=concrete):
            return grade_concrete(question, concrete, reading, attempts=attempts)

        raws = [{key: value for key, value in concrete.expect.items()}, {}]
        problems += _given_route_problems(call, listed, raws, "dolma")
    cyclic = _concrete_of("c08.K.digests-match", None, fallbacks=("cyclic-wrapper-route",))
    assert dict(cyclic.expect) == {"shuffle_digests": "none recorded"}

    def call_cyclic(question, reading, attempts):
        return grade_concrete(question, cyclic, reading, attempts=attempts)

    raws = [{"shuffle_digests": "none recorded"}, {"shuffle_digests": "equal"}, {}]
    problems += _given_route_problems(call_cyclic, "c08.K.digests-match", raws, "dolma")
    assert not problems, "\n".join(problems)


def test_a_candidate_set_is_graded_cannot_answer_on_any_route_where_no_subject_has_one():
    # Changed bar (the second half of the test above): the candidate set of a same-shard pair
    # was compared on a question given a route. The rule of every subject's row is "cannot
    # answer", so a question given a route is graded cannot answer whatever the candidate set
    # or the label word it reads.
    grade_fn = need_attr("baselines.grade", "grade")
    listed = "c03.pairs.same-shard"
    assert _every_subjects_rule_is_cannot_answer(listed)
    same_shard = _expect(listed)
    value = dict(same_shard, candidate_set=["Z-102", "Z-101"])

    def call(question, reading, attempts):
        return grade_fn(question, value, reading, attempts=attempts)

    reading = {key: val for key, val in value.items()}
    raws = [reading, dict(reading, candidate_set=["Z-102", "Z-999"]), dict(reading, label="recovered")]
    problems = _given_route_problems(call, listed, raws, "dolma", "small")
    assert not problems, "\n".join(problems)


def _private_names_read_from(module_file, module_path):
    """The names with a leading underscore that the source of ``module_file`` takes from
    the module ``module_path``: ``from <module_path> import _x``, and ``_x`` read as an
    attribute of a name bound to the module (``import a.b as m``, ``from a import b as m``)."""
    import ast

    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    package, _dot, leaf = module_path.rpartition(".")
    bound = set()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == module_path:
                found += [alias.name for alias in node.names if alias.name.startswith("_")]
            elif node.module == package:
                bound |= {alias.asname or alias.name for alias in node.names if alias.name == leaf}
        elif isinstance(node, ast.Import):
            bound |= {alias.asname for alias in node.names if alias.name == module_path and alias.asname}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in bound and node.attr.startswith("_")):
            found.append(node.attr)
    return found


def test_the_grader_takes_the_sentence_fields_and_the_one_id_rows_from_the_one_public_source():
    # Changed bar (new): the fields that hold a sentence where a value stands, and the rows
    # that ask one id per ordinal, have one source in the module of the expectations, under
    # public names; the grader reads those and no private name of that module.
    grade_module = need("baselines.grade")
    expectations = need("baselines.expected")
    assert _private_names_read_from(grade_module.__file__, "baselines.expected") == []
    for name in ("SENTENCE_FIELDS", "ONE_ID_ROWS"):
        public = getattr(expectations, name, None)
        assert public is not None, f"baselines.expected has no {name}"
        held = getattr(grade_module, name, public)
        assert held is public, f"baselines.grade holds a copy of {name}"


def test_the_scan_for_private_names_finds_one_where_one_is_read(tmp_path):
    # The scan above can fail: a source that reads a private name of the module through an
    # alias, and one that imports it by name, are both found; a public name is not.
    probe = tmp_path / "probe_source.py"
    probe.write_text(
        "from baselines import expected as e\n"
        "from baselines.expected import _hidden, shown\n"
        "x = e._other\n"
        "y = e.public\n",
        encoding="utf-8",
    )
    assert sorted(_private_names_read_from(probe, "baselines.expected")) == ["_hidden", "_other"]


# The two fields of the table that hold a word and not a sentence beside the sentence-valued
# ones: the word stays as written and is compared as the word it is.
WORD_ROWS = (
    ("c02.iii.no-replay", "replay", "not attempted", "attempted"),
)


def _refuses_every_field_of_the_public_source(real_grade):
    """The known-bad grader: a field listed in the public source of the sentence fields is
    refused as a sentence whatever it holds, a word included."""
    not_gradable = need_attr("baselines.grade", "NotGradable")

    def bad(question, expect, outcome, **kwargs):
        sentence_fields = need_attr("baselines.expected", "SENTENCE_FIELDS")
        for name in sentence_fields.get(question.listed_row, ()):
            if isinstance(expect.get(name), str):
                raise not_gradable(f"{question.listed_row} holds a sentence for {name}")
        return real_grade(question, expect, outcome, **kwargs)

    return bad


def _word_row_is_graded_by_what_the_rules_say(grade_function) -> bool:
    """True when a question given a route, on a row whose word the public source lists, is
    graded cannot answer whatever the reading holds (the word, another word, nothing) and is
    never refused."""
    for listed, field, word, other in WORD_ROWS:
        expect = _expect(listed)
        assert expect[field] == word
        assert _every_subjects_rule_is_cannot_answer(listed)

        def call(question, reading, attempts, expect=expect):
            return grade_function(question, expect, reading, attempts=attempts)

        raws = [{field: word}, {field: other}, {}]
        if _given_route_problems(call, listed, raws, "manifest"):
            return False
    return True


def test_a_word_the_public_source_of_the_sentence_fields_lists_is_graded_by_what_the_rules_say():
    # The public source lists, beside the sentence-valued fields, two fields that hold a word
    # of the table (a status or an answer word): the replay of c02.iii.no-replay is one. A
    # word is never refused as a sentence. Changed bar (the name and the checks): no subject
    # has a route to the row, the rule of each is "cannot answer", so a question given a route
    # is graded cannot answer, not right or wrong on the word it reads.
    real = need_attr("baselines.grade", "grade")
    assert "replay" in need_attr("baselines.expected", "SENTENCE_FIELDS")["c02.iii.no-replay"]
    assert _word_row_is_graded_by_what_the_rules_say(_refuses_every_field_of_the_public_source(real)) is False
    assert _word_row_is_graded_by_what_the_rules_say(real) is True


GROUPS_READING = {
    "shard-1": ["Z-001", "Z-101", "Z-003", "Z-102", "Z-005"],
    "shard-2": ["Z-303", "Z-301", "Z-006", "Z-302"],
}


def _one_id_question(listed):
    return _make_question(listed, "dolma", "small")


def _check_one_id_rows(listed, extra):
    """The one-id reading of ``listed``: ``extra`` is what the reading holds beside the ids
    (the value fields the row's text names). Returns the reader for further cases."""
    grade_fn = need_attr("baselines.grade", "grade")
    concrete = _concrete_of(listed, "small", counted={"small": COUNTED})
    expect = concrete.expect
    assert expect["per_ordinal"]["shard-1"][1] == ["Z-102", "Z-101"], listed
    q = _one_id_question(listed)
    assert q.route is not None, listed

    def read(ids, **more):
        raw = dict(extra, **more)
        if ids is not None:
            raw["per_ordinal"] = ids
        return grade_fn(q, expect, _make_reading(q, raw))

    assert read(GROUPS_READING).status == "right", listed
    # The other member of the pair is as right as the first.
    swapped = dict(GROUPS_READING, **{"shard-1": ["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"]})
    assert read(swapped).status == "right", listed
    # An id outside the group, at an ordinal of the group or outside every group, is wrong.
    for shard, position, other in (("shard-1", 1, "Z-999"), ("shard-1", 0, "Z-102"),
                                   ("shard-2", 2, "Z-007"), ("shard-2", 0, "Z-101")):
        ids = {name: list(held) for name, held in GROUPS_READING.items()}
        ids[shard][position] = other
        got = read(ids)
        assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"], (listed, shard, position)
    # A gap is wrong: a missing shard, no ids at all.
    short = {"shard-1": GROUPS_READING["shard-1"]}
    assert read(short).status == "wrong", listed
    got = read(None)
    assert got.status == "wrong" and "per_ordinal" in list(got.evidence), listed
    return read


def test_one_id_at_an_ordinal_of_a_group_of_identical_documents_is_right_for_any_member():
    # The text of the table this rests on: for the records rows the reading is the id of the
    # tied row at every ordinal, and right is "for every ordinal the id is the id of the
    # planted record ...; for an identical pair placed in one shard, either member of the
    # pair". The count of mismatches the row holds is the number of ordinals that differ: it
    # is worked out from the ids, and the reading holds no count of its own.
    row = _check10_rows()["c10.c03.records.dolma"]
    assert row["rule"].startswith("the reading is, for every ordinal of every shard, the id of the tied row.")
    assert "for an identical pair placed in one shard, either member of the pair" in row["expect"]["right"]
    for listed in ("c03.records.capture", "c03.records.position-recovery"):
        read = _check_one_id_rows(listed, {})
        ids = {name: list(held) for name, held in GROUPS_READING.items()}
        ids["shard-2"][2] = "Z-007"
        assert read(ids, mismatches=0).status == "wrong", listed


def test_the_documents_mapped_row_reads_the_ids_and_the_number_of_documents_with_no_tied_row():
    # The text of the table this rests on: the reading of c06.generated.documents-mapped is,
    # per shard, the number of documents with no tied row, beside the occurrence each tied
    # row names (or a member of the candidate set of an identical pair).
    row = _check10_rows()["c10.c06.documents-mapped.dolma"]
    assert "the number with none" in row["rule"]
    assert "a member of the candidate set it gives" in row["expect"]["right"]
    listed = "c06.generated.documents-mapped"
    read = _check_one_id_rows(listed, {"unmapped_documents": 0})
    got = read(GROUPS_READING, unmapped_documents=1)
    assert got.status == "wrong" and list(got.evidence) == ["unmapped_documents"]
    # The number of documents with no tied row is a value of the reading, and a gap is wrong.
    got = read(GROUPS_READING, unmapped_documents=None)
    assert got.status == "wrong" and list(got.evidence) == ["unmapped_documents"]


def test_the_ordinals_of_a_same_shard_pair_are_not_compared_for_the_token_matching_row():
    grade_fn = need_attr("baselines.grade", "grade")
    # The text of the table this rests on: the row's own expectation.
    written = _listed_rows()["c03.records.token-matching"]["expect"]["per_ordinal"]
    assert written == "the same record as the capture map, for every document outside a same-shard pair"
    concrete = _concrete_of("c03.records.token-matching", "small", counted={"small": COUNTED})
    q = _one_id_question("c03.records.token-matching")
    in_groups = {
        "shard-1": ["Z-001", "Z-999", "Z-003", None, "Z-005"],
        "shard-2": ["Z-998", "Z-997", "Z-006", "Z-996"],
    }
    assert grade_fn(q, concrete.expect, _make_reading(q, {"per_ordinal": in_groups})).status == "right"
    # An ordinal outside every group is compared for this row too.
    outside = {name: list(held) for name, held in in_groups.items()}
    outside["shard-2"][2] = "Z-007"
    got = grade_fn(q, concrete.expect, _make_reading(q, {"per_ordinal": outside}))
    assert got.status == "wrong" and list(got.evidence) == ["per_ordinal"]
    # The capture row, asked of the same ids, does compare the ordinals of a group.
    capture = _concrete_of("c03.records.capture", "small", counted={"small": COUNTED})
    cq = _one_id_question("c03.records.capture")
    assert grade_fn(cq, capture.expect, _make_reading(cq, {"per_ordinal": in_groups})).status == "wrong"


# --- A plan that allows a set of outcomes reads right only from the subject's records -------


def test_a_plan_that_allows_a_set_of_outcomes_reads_right_only_from_the_subjects_own_records():
    grade_records = need_attr("baselines.grade", "grade_from_records")
    kill = _expect("c04.kill-after-100")
    q = _make_question("c04.kill-after-100", "dolma")
    assert q.route is not None

    def read(raw, question=q, expect=kill):
        return grade_records(question, expect, _make_reading(question, raw))

    # After the stop: no row, or 500, is allowed; 1 to 499 is forbidden; anything else is
    # outside the allowed outcomes.
    for rows in (0, 500):
        assert read({"records": {"rows_after_stop": rows}}).status == "right", rows
    for rows in (1, 100, 499):
        got = read({"records": {"rows_after_stop": rows}})
        assert got.status == "wrong" and list(got.evidence) == ["never"], rows
    got = read({"records": {"rows_after_stop": 501}})
    assert got.status == "wrong" and list(got.evidence) == ["one_of"]
    # An outcome the subject names is never taken: the records decide.
    claimed = read({"outcome": kill["one_of"][1], "records": {"rows_after_stop": 100}})
    assert claimed.status == "wrong" and list(claimed.evidence) == ["never"]
    denied = read({"outcome": kill["never"][0], "records": {"rows_after_stop": 500}})
    assert denied.status == "right"
    # No records, or a count that is not a number, is a gap and wrong.
    for raw in ({"outcome": kill["one_of"][1]}, {"records": {}}, {"records": None},
                {"records": {"rows_after_stop": "500"}}, {"records": {"rows_after_stop": True}}):
        got = read(raw)
        assert got.status == "wrong" and list(got.evidence) == ["records"], raw


def test_the_retry_reads_the_rows_after_it_whether_they_match_the_shard_and_whether_it_refused():
    grade_records = need_attr("baselines.grade", "grade_from_records")
    retry = _expect("c04.retry")
    for subject in ("dolma", "originblame"):
        q = _make_question("c04.retry", subject)
        assert q.route is not None, subject

        def read(raw, question=q):
            return grade_records(question, retry, _make_reading(question, raw))

        whole = {"rows_after_retry": 500, "refused": False}
        assert read({"records": dict(whole, matches_shard=True)}).status == "right", subject
        # Changed bar: a refusal is no longer right whatever else the records hold. A refusal
        # after which no row remains, with a named mismatch (a non-empty text), is right.
        named = {"rows_after_retry": 0, "refused": True, "mismatch": "declared count mismatch"}
        assert read({"records": named}).status == "right", subject
        # The refusal must name its mismatch: with no mismatch, or an empty text, it is not the
        # allowed "rejection with a named mismatch", so it is not right: the records are a gap,
        # and the evidence is "records".
        for unnamed in ({"rows_after_retry": 0, "refused": True},
                        {"rows_after_retry": 0, "refused": True, "mismatch": ""}):
            got = read({"records": unnamed})
            assert got.status == "wrong" and list(got.evidence) == ["records"], (subject, unnamed)
        # A refusal after which 1 to 499 rows remain (for OriginBlame: entries) is "a partial
        # map", which the plan never allows: wrong, whatever the message of the refusal.
        for rows in (1, 300, 499):
            for message in ("declared count mismatch", "declared digest mismatch", "any other text", None):
                held = {"rows_after_retry": rows, "refused": True}
                if subject == "dolma":
                    # The rows' ranges index the documents of the token file.
                    held["matches_shard"] = True
                if message is not None:
                    held["mismatch"] = message
                got = read({"records": held})
                assert got.status == "wrong" and list(got.evidence) == ["never"], (subject, held)
        # A partial map, and 500 rows that do not match the shard, are forbidden. The second is
        # asked of the subject that matches rows to a file only: for the subject that counts
        # entries, 500 entries are right (see the cells of the counting subjects below).
        forbidden = [{"rows_after_retry": 300, "refused": False, "matches_shard": False}]
        if subject == "dolma":
            forbidden.append({"rows_after_retry": 500, "refused": False, "matches_shard": False})
        for records in forbidden:
            got = read({"records": records})
            assert got.status == "wrong" and list(got.evidence) == ["never"], (subject, records)
        claimed = {"outcome": retry["one_of"][0],
                   "records": {"rows_after_retry": 300, "refused": False, "matches_shard": False}}
        assert read(claimed).status == "wrong"
        # The count and the refusal are both read; a record that lacks one is a gap.
        for records in ({"refused": False}, {"rows_after_retry": 500}):
            got = read({"records": records})
            assert got.status == "wrong" and list(got.evidence) == ["records"], (subject, records)
    # OriginBlame's entries are counted and not matched to a file, so it has no matches_shard.
    ob = _make_question("c04.retry", "originblame")
    got = grade_records(ob, retry, _make_reading(ob, {"records": dict(whole)}))
    assert got.status == "right"


# The cells of the write made again, by the three values the records give: the rows after it
# (n), whether their ranges index the documents of the token file (m; the one subject that
# counts entries and matches none to a file, OriginBlame, has none), and whether the tool
# refused, with the mismatch it names (a non-empty text). Each cell is (subject, records,
# status, evidence). Only the cells the plan's words decide are written; a cell they do not
# decide (a refusal that names nothing at 500 rows or over, a Dolma record with no m over 500
# rows, one refused over 500 rows with m false and no mismatch named, a write that was not
# refused and left no row, or left over 500 rows with m false) is left out and not pinned.
NAMED = "declared count mismatch"
OVER = (501, 600)


def _retry_records(rows, refused, mismatch=None, matches=None) -> dict:
    found = {"rows_after_retry": rows, "refused": refused}
    if mismatch is not None:
        found["mismatch"] = mismatch
    if matches is not None:
        found["matches_shard"] = matches
    return found


def _retry_cells_that_stand():
    """The cells the plan's words decide and the grader already read so."""
    cells = []
    for matches in (None, True, False):
        # A refusal after which no row remains: right when it names its mismatch (Dolma with m
        # of either value or none, and OriginBlame, which has none), else a gap in the records.
        cells.append(("dolma", _retry_records(0, True, NAMED, matches), "right", None))
        for unnamed in (None, ""):
            cells.append(("dolma", _retry_records(0, True, unnamed, matches), "wrong", ["records"]))
    cells.append(("originblame", _retry_records(0, True, NAMED), "right", None))
    for unnamed in (None, ""):
        cells.append(("originblame", _retry_records(0, True, unnamed), "wrong", ["records"]))
    for rows in (1, 300, 499):
        # A refusal after which 1 to 499 rows remain that do not index the documents of the
        # token file: right when it names its mismatch, else a gap.
        cells.append(("dolma", _retry_records(rows, True, NAMED, False), "right", None))
        cells.append(("dolma", _retry_records(rows, True, None, False), "wrong", ["records"]))
        cells.append(("dolma", _retry_records(rows, True, "", False), "wrong", ["records"]))
        # ... and the same rows that do index them (OriginBlame: its entries) are a partial map.
        for mismatch in (NAMED, "", None):
            cells.append(("dolma", _retry_records(rows, True, mismatch, True), "wrong", ["never"]))
            cells.append(("originblame", _retry_records(rows, True, mismatch), "wrong", ["never"]))
    for matches in (True, False):
        # A refusal after which the 500 rows remain, with its mismatch named.
        cells.append(("dolma", _retry_records(500, True, NAMED, matches), "right", None))
    for rows in OVER:
        cells.append(("dolma", _retry_records(rows, True, NAMED, False), "right", None))
    return cells


def _retry_cells_that_change():
    """The cells where more than 500 rows (or entries) remain, which the plan never allows
    when they index the documents of the token file, or for a subject that only counts them:
    the count is not the shard's 500, whether the tool refused or not."""
    cells = []
    for rows in OVER:
        for mismatch in (NAMED, "", None):
            cells.append(("dolma", _retry_records(rows, True, mismatch, True), "wrong", ["never"]))
            cells.append(("originblame", _retry_records(rows, True, mismatch), "wrong", ["never"]))
        cells.append(("dolma", _retry_records(rows, False, None, True), "wrong", ["never"]))
        cells.append(("originblame", _retry_records(rows, False), "wrong", ["never"]))
    return cells


def _retry_cell_problems(grade_functions, cells) -> list[str]:
    """Each cell read by every entry point that reads the retry: what differs from the cell."""
    expect = _expect("c04.retry")
    found = []
    for subject, records, status, evidence in cells:
        question = _make_question("c04.retry", subject)
        for name, grade_function in grade_functions.items():
            try:
                got = grade_function(question, expect, _make_reading(question, {"records": records}))
            except ValueError as problem:
                found.append(f"{name} {subject} {records}: {type(problem).__name__}: {problem}")
                continue
            if (got.status, got.evidence) != (status, evidence):
                found.append(f"{name} {subject} {records}: {got.status} {got.evidence!r}, "
                             f"not {status} {evidence!r}")
    return found


def _a_refusal_with_no_row_left_is_right_whatever_it_names(real_grade):
    """The known-bad grader: a refusal after which no row remains is right, named or not."""

    def bad(question, expect, outcome, **kwargs):
        records = (outcome.raw or {}).get("records") or {}
        if records.get("refused") is True and records.get("rows_after_retry") == 0:
            return _make_grade("right")
        return real_grade(question, expect, outcome, **kwargs)

    return bad


def _an_unnamed_refusal_is_wrong_with_another_word(real_grade):
    """The known-bad grader: a refusal that names no mismatch is wrong, as it should be, but
    the field it names is another word than "records"."""

    def bad(question, expect, outcome, **kwargs):
        got = real_grade(question, expect, outcome, **kwargs)
        records = (outcome.raw or {}).get("records") or {}
        if got.status == "wrong" and list(got.evidence) == ["records"] and records.get("refused") is True:
            return _make_grade("wrong", ["one_of"])
        return got

    return bad


def test_the_retry_cells_the_plans_words_decide_and_the_grader_already_reads_so_stand():
    # The cells where a refusal is named or not, with no row left or rows that do not index the
    # token file's documents: the status stands as the plan words it, and the evidence word of
    # a refusal that names no mismatch is "records" (the records are a gap).
    both = {name: need_attr("baselines.grade", name) for name in ("grade", "grade_from_records")}
    cells = _with_trace_subjects(_retry_cells_that_stand())
    assert len(cells) > 40
    problems = _retry_cell_problems(both, cells)
    assert not problems, "\n".join(problems)
    # The same cells on the known-bad graders find the faults, so they can fail: a refusal
    # that names nothing read as right, and one read as wrong with another word than "records".
    bad = {"grade": _a_refusal_with_no_row_left_is_right_whatever_it_names(both["grade"])}
    assert _retry_cell_problems(bad, cells) != []
    other_word = {"grade": _an_unnamed_refusal_is_wrong_with_another_word(both["grade"])}
    caught = _retry_cell_problems(other_word, cells)
    assert caught and all("one_of" in problem for problem in caught)


def test_the_retry_over_500_rows_is_never_allowed_whether_it_was_refused_or_not():
    # Changed bar (new cells): what the plan never allows for a write made again is "a map whose
    # count or digest does not match the shard", and 600 is not the shard's 500. Rows that index the token
    # file's documents, or entries counted for the subject that matches none, are never
    # allowed over 500, refused or not and whatever the refusal names; before this the grader
    # allowed a refusal that named its mismatch and called the others outside the plan.
    both = {name: need_attr("baselines.grade", name) for name in ("grade", "grade_from_records")}
    problems = _retry_cell_problems(both, _with_trace_subjects(_retry_cells_that_change()))
    assert not problems, "\n".join(problems)


# --- The write made again: the subjects beside the trace, the subjects that count entries ---------

COUNTS_ONLY_SUBJECTS = ("originblame", "originblame+trace")
MATCHING_SUBJECTS = ("dolma", "dolma+trace")
TRACE_TWIN = {"dolma": "dolma+trace", "originblame": "originblame+trace"}


def _with_trace_subjects(cells):
    """The cells, and each cell of a tool alone asked again of the subject that reads as the
    tool does beside the trace."""
    twins = [(TRACE_TWIN[subject], records, status, evidence)
             for subject, records, status, evidence in cells if subject in TRACE_TWIN]
    return [*cells, *twins]


def test_the_subjects_beside_the_trace_read_the_write_made_again_by_the_rule_of_the_tool_alone():
    # The text of the table this rests on: the row beside the trace is "as the row of the tool
    # alone" and holds its rule and its right and wrong text word for word.
    rows = _check10_rows()
    for alone, beside in (("dolma", "dolma+trace"), ("originblame", "originblame+trace")):
        one = rows[f"c10.c04.interrupted-write.{alone}"]
        other = rows[f"c10.c04.interrupted-write.{beside}"]
        assert other["route"].startswith("as the row of the "), beside
        assert other["rule"] == one["rule"], beside
        assert other["expect"]["right"] == one["expect"]["right"], beside
        assert other["expect"]["wrong"] == one["expect"]["wrong"], beside
    # The subject that counts entries names no ranges that index a token file, and its right
    # text names 500 entries; the one that reads rows names the ranges.
    for subject in COUNTS_ONLY_SUBJECTS:
        row = rows[f"c10.c04.interrupted-write.{subject}"]
        assert "index the documents" not in row["rule"], subject
        assert "after the retry there are 500 entries, or the tool refused the write with an error" \
            in row["expect"]["right"], subject
    for subject in MATCHING_SUBJECTS:
        row = rows[f"c10.c04.interrupted-write.{subject}"]
        assert "whether their ranges index the documents of the token file" in row["rule"], subject


REFUSALS = ((False, None), (True, NAMED), (True, ""), (True, None))


def _counting_subject_cells():
    """The cells of the two subjects that count entries and match none to a file: the rule
    names the number of entries and whether the tool refused. A matches_shard their reader
    emits is not part of the rule, so it changes no grade."""
    cells = []
    for subject in COUNTS_ONLY_SUBJECTS:
        for emitted in (None, True, False):
            # 500 entries are right whatever the refusal says (the right text: "there are 500
            # entries, or the tool refused the write with an error").
            for refused, mismatch in REFUSALS:
                cells.append((subject, _retry_records(500, refused, mismatch, emitted), "right", None))
            # Any other number of entries is not the shard's 500: 1 to 499 are a partial map,
            # and more than 500 are a count that is not the shard's; both are never allowed.
            for rows in (1, 300, 499, *OVER):
                for refused, mismatch in REFUSALS:
                    cells.append((subject, _retry_records(rows, refused, mismatch, emitted),
                                  "wrong", ["never"]))
        # The count and the refusal are both read; a record that lacks the refusal is a gap.
        cells.append((subject, {"rows_after_retry": 500}, "wrong", ["records"]))
        cells.append((subject, {"rows_after_retry": 500, "matches_shard": True}, "wrong", ["records"]))
    return cells


def test_the_subjects_that_count_entries_read_500_entries_as_right_and_ignore_a_matches_shard():
    both = {name: need_attr("baselines.grade", name) for name in ("grade", "grade_from_records")}
    problems = _retry_cell_problems(both, _counting_subject_cells())
    assert not problems, "\n".join(problems)


def _silent_cells():
    """The cells of the write made again that the plan's words leave to the rule of the row:
    Dolma and Dolma beside the trace, by the rows after it, whether their ranges index the
    documents of the token file (matches_shard, when the subject says) and the refusal."""
    cells = []
    for subject in MATCHING_SUBJECTS:
        for rows in OVER:
            # More than 500 rows and no word on whether they index the token file's documents:
            # a gap in the records, refused or not.
            for refused, mismatch in REFUSALS:
                cells.append((subject, _retry_records(rows, refused, mismatch), "wrong", ["records"]))
        # 500 rows whose ranges index the documents of the token file: right whatever the
        # refusal says (the right text: "500 rows that index the 500 documents ..., or the tool
        # refused").
        for refused, mismatch in REFUSALS:
            cells.append((subject, _retry_records(500, refused, mismatch, True), "right", None))
        # 500 rows whose ranges do not index them: right only with a refusal that names its
        # mismatch.
        cells.append((subject, _retry_records(500, True, NAMED, False), "right", None))
        for unnamed in (None, ""):
            cells.append((subject, _retry_records(500, True, unnamed, False), "wrong", ["records"]))
        cells.append((subject, _retry_records(500, False, None, False), "wrong", ["never"]))
    return cells


def test_the_retry_cells_the_ruled_grid_left_silent_are_read_by_the_rule_of_the_row():
    both = {name: need_attr("baselines.grade", name) for name in ("grade", "grade_from_records")}
    problems = _retry_cell_problems(both, _silent_cells())
    assert not problems, "\n".join(problems)


def _reads_the_trace_subject_as_one_that_counts(real_grade):
    """The known-bad grader: the subject beside the toolkit is read as the counting subject
    (a matches_shard its reader gives is not looked at)."""

    def bad(question, expect, outcome, **kwargs):
        if question.subject == "dolma+trace":
            question = dataclasses.replace(question, subject="originblame")
        return real_grade(question, expect, outcome, **kwargs)

    return bad


def _reads_the_counting_subject_by_its_matches_shard(real_grade):
    """The known-bad grader: the counting subject beside the trace is read as the toolkit
    that matches rows to a file, so a missing matches_shard is a gap."""

    def bad(question, expect, outcome, **kwargs):
        if question.subject == "originblame+trace":
            question = dataclasses.replace(question, subject="dolma+trace")
        return real_grade(question, expect, outcome, **kwargs)

    return bad


def test_the_cells_of_the_subjects_beside_the_trace_fail_a_grader_that_reads_them_as_the_other_kind():
    # The subject beside the toolkit passes these cells today, so its red is a known-bad input:
    # a grader that reads it as the subject that counts entries, and the reverse.
    real = need_attr("baselines.grade", "grade")
    cells = _with_trace_subjects(_retry_cells_that_stand())
    matching = [cell for cell in cells if cell[0] == "dolma+trace"]
    assert matching
    found = _retry_cell_problems({"grade": _reads_the_trace_subject_as_one_that_counts(real)}, matching)
    assert found, "a grader that reads the subject beside the toolkit as a counting one was not caught"
    counting = [cell for cell in _counting_subject_cells() if cell[0] == "originblame+trace"]
    assert counting
    found = _retry_cell_problems({"grade": _reads_the_counting_subject_by_its_matches_shard(real)}, counting)
    assert found, "a grader that reads the counting subject by its matches_shard was not caught"


# --- A reading that says it cannot answer, on a row that allows no such outcome ----------------


def test_a_cannot_answer_key_on_a_row_that_allows_a_set_of_outcomes_reads_wrong_in_its_records():
    # The text of the table this rests on: the one_of of the two rows names two outcomes, and
    # none is "cannot answer".
    grade_fn = need_attr("baselines.grade", "grade")
    grade_records = need_attr("baselines.grade", "grade_from_records")
    key = need_attr("baselines.grade", "CANNOT_KEY")
    held = {"c04.kill-after-100": {"rows_after_stop": 0},
            "c04.retry": {"rows_after_retry": 0, "refused": True, "mismatch": NAMED}}
    problems = []
    for listed, records in held.items():
        expect = _expect(listed)
        assert "cannot answer" not in expect["one_of"], listed
        q = _make_question(listed, "dolma")
        for said in ("the route could not be walked", "", None):
            for raw in ({key: said}, {key: said, "records": records}):
                for name, entry in (("grade", grade_fn), ("grade_from_records", grade_records)):
                    got = entry(q, expect, _make_reading(q, raw))
                    if (got.status, got.evidence) != ("wrong", ["records"]):
                        problems.append(f"{name} {listed} {raw}: {got.status} {got.evidence!r}")
        # A row whose one_of does name "cannot answer" keeps the route that could not be walked.
        allowing = dict(expect, one_of=[*expect["one_of"], "cannot answer"])
        got = grade_fn(q, allowing, _make_reading(q, {key: "the route could not be walked"}))
        if (got.status, got.evidence) != ("cannot answer", "the route could not be walked"):
            problems.append(f"{listed} one_of with cannot answer: {got.status} {got.evidence!r}")
    assert not problems, "\n".join(problems)


def test_an_empty_or_none_cannot_answer_key_is_never_a_cannot_answer_grade():
    grade_fn = need_attr("baselines.grade", "grade")
    key = need_attr("baselines.grade", "CANNOT_KEY")
    expect = _expect(ANSWER_ROW)
    q = _make_question(ANSWER_ROW, "manifest")
    problems = []
    for said in (None, "", []):
        got = grade_fn(q, expect, _make_reading(q, {key: said, "answer": "may contain"}))
        if got.status == "cannot answer":
            problems.append(f"{said!r}: cannot answer")
    # A key that names the missing route stays a cannot answer, with that evidence.
    got = grade_fn(q, expect, _make_reading(q, {key: "no tie", "answer": "may contain"}))
    if (got.status, got.evidence) != ("cannot answer", "no tie"):
        problems.append(f"a named key: {got.status} {got.evidence!r}")
    assert not problems, "\n".join(problems)


# --- The two rows of the 10k set that read one value each ------------------------------------

# Hand-written in the shape of the counting document (the part "shards"), one for each layout.
PARTITIONED_SHARDS_DOC = {"shards": {
    "layout": "partitioned", "keep": ["shard-02", "shard-07", "shard-11"],
    "rewrite": ["shard-01", "shard-05"],
}}
_ALL_CHUNKS = [f"chunk-{n:02d}" for n in range(1, 21)]
ROUND_ROBIN_SHARDS_DOC = {"shards": {"layout": "round-robin", "keep": [], "rewrite": _ALL_CHUNKS}}
# A count whose derived number is not the table's 20: the derived value is the one compared.
ROUND_ROBIN_ONE_KEPT_DOC = {"shards": {
    "layout": "round-robin", "keep": [_ALL_CHUNKS[0]], "rewrite": _ALL_CHUNKS[1:],
}}
ROUTED_ARTIFACT_SUBJECTS = ("manifest", "dolma", "originblame", "dolma+trace", "originblame+trace")


def test_the_kept_shards_row_and_the_all_rewrite_row_compare_only_the_value_their_rule_names():
    grade_concrete = need_attr("baselines.grade", "grade_concrete")
    grade_fn = need_attr("baselines.grade", "grade")
    kept_row, rewrite_row = "c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"
    # The text of the table this rests on: the rule names the list of shards read keep and the
    # number of chunks read rewrite, and the listed rows hold more than that.
    for subject in ROUTED_ARTIFACT_SUBJECTS:
        rule = _check10_rows()[f"c10.c09.artifact-actions.{subject}"]["rule"]
        assert "the reading is the list of shards read keep" in rule, subject
        assert "it is the number of chunks read rewrite" in rule, subject
    assert set(_listed_rows()[kept_row]["expect"]) == {"keep", "kept_shards_byte_identical", "every_other_shard"}
    assert set(_listed_rows()[rewrite_row]["expect"]) == {"rewrite", "keep"}
    kept = _concrete_of(kept_row, None, counted={None: PARTITIONED_SHARDS_DOC})
    assert kept.expect["keep"] == ["shard-02", "shard-07", "shard-11"]
    whole = _concrete_of(rewrite_row, None, counted={None: ROUND_ROBIN_SHARDS_DOC})
    one_kept = _concrete_of(rewrite_row, None, counted={None: ROUND_ROBIN_ONE_KEPT_DOC})
    problems = []

    def note(label, got, status, evidence):
        if (got.status, got.evidence) != (status, evidence):
            problems.append(f"{label}: {got.status} {got.evidence!r}, not {status} {evidence!r}")

    for subject in ROUTED_ARTIFACT_SUBJECTS:
        q = _make_question(kept_row, subject)
        assert q.route is not None, subject
        keep = ["shard-02", "shard-07", "shard-11"]
        # The list of shards read keep, alone, is the reading; the rest of the row is not read.
        note(f"{subject} kept", grade_concrete(q, kept, _make_reading(q, {"keep": keep})), "right", None)
        beside = {"keep": keep, "every_other_shard": "keep", "kept_shards_byte_identical": False}
        note(f"{subject} kept, fields beside",
             grade_concrete(q, kept, _make_reading(q, beside)), "right", None)
        note(f"{subject} kept, a shard missing",
             grade_concrete(q, kept, _make_reading(q, {"keep": keep[:-1]})), "wrong", ["keep"])
        note(f"{subject} kept, a shard too many",
             grade_concrete(q, kept, _make_reading(q, {"keep": [*keep, "shard-01"]})), "wrong", ["keep"])
        note(f"{subject} kept, no list",
             grade_concrete(q, kept, _make_reading(q, {"every_other_shard": "rewrite"})), "wrong", ["keep"])
        r = _make_question(rewrite_row, subject)
        assert r.route is not None, subject
        # The number of chunks read rewrite, alone, is the reading.
        note(f"{subject} rewrite", grade_concrete(r, whole, _make_reading(r, {"rewrite": 20})), "right", None)
        note(f"{subject} rewrite, fields beside",
             grade_concrete(r, whole, _make_reading(r, {"rewrite": 20, "keep": 3})), "right", None)
        note(f"{subject} rewrite, plain expectation",
             grade_fn(r, _expect(rewrite_row), _make_reading(r, {"rewrite": 20})), "right", None)
        note(f"{subject} rewrite, 19",
             grade_concrete(r, whole, _make_reading(r, {"rewrite": 19})), "wrong", ["rewrite"])
        note(f"{subject} rewrite, no number",
             grade_concrete(r, whole, _make_reading(r, {"keep": 0})), "wrong", ["rewrite"])
        # Where a number is derived, it is the one compared (the table's own 20 is not).
        note(f"{subject} rewrite, derived 19",
             grade_concrete(r, one_kept, _make_reading(r, {"rewrite": 19})), "right", None)
        note(f"{subject} rewrite, derived 19, read 20",
             grade_concrete(r, one_kept, _make_reading(r, {"rewrite": 20})), "wrong", ["rewrite"])
    assert not problems, "\n".join(problems)


# --- The count of a step reads one value ---------------------------------------------------------

STEP_FACTS_SUBJECTS = ("trace", "dolma+trace", "originblame+trace")


def test_the_count_of_step_8_compares_only_the_number_of_samples_its_rule_names():
    grade_fn = need_attr("baselines.grade", "grade")
    listed = "c01.E.step8.counted"
    # The text of the table this rests on: the rule names the number of samples a step counts,
    # and the listed row holds the loud stop beside it.
    for subject in STEP_FACTS_SUBJECTS:
        rule = _check10_rows()[f"c10.c01.step-facts.{subject}"]["rule"]
        assert "the number of samples a step counts" in rule, subject
        assert "loud_stop" not in rule and "stop" not in rule, subject
    expect = _expect(listed)
    assert set(expect) == {"consumed_samples", "loud_stop"}
    assert expect["consumed_samples"] == 4
    problems = []

    def note(label, got, status, evidence):
        if (got.status, got.evidence) != (status, evidence):
            problems.append(f"{label}: {got.status} {got.evidence!r}, not {status} {evidence!r}")

    for subject in STEP_FACTS_SUBJECTS:
        q = _make_question(listed, subject)
        assert q.group == "c01.step-facts" and q.route is not None, subject
        # The number of samples alone is the reading.
        note(f"{subject} count alone",
             grade_fn(q, expect, _make_reading(q, {"consumed_samples": 4})), "right", None)
        # The loud stop beside it is not read, whatever it says.
        for stop in (False, True):
            note(f"{subject} count, loud_stop {stop}",
                 grade_fn(q, expect, _make_reading(q, {"consumed_samples": 4, "loud_stop": stop})),
                 "right", None)
        note(f"{subject} count 5",
             grade_fn(q, expect, _make_reading(q, {"consumed_samples": 5})),
             "wrong", ["consumed_samples"])
        note(f"{subject} count 5, loud_stop beside",
             grade_fn(q, expect, _make_reading(q, {"consumed_samples": 5, "loud_stop": False})),
             "wrong", ["consumed_samples"])
        note(f"{subject} no count",
             grade_fn(q, expect, _make_reading(q, {"loud_stop": False})),
             "wrong", ["consumed_samples"])
        note(f"{subject} empty reading",
             grade_fn(q, expect, _make_reading(q, {})), "wrong", ["consumed_samples"])
    assert not problems, "\n".join(problems)


# --- The pair name of a split pair -----------------------------------------------------------


def test_the_name_of_a_split_pair_is_not_compared_only_the_chunk_file_and_line_of_its_two_ordinals():
    # The text of the table this rests on: the rule names, for each of the two ordinals of a
    # split pair, "the chunk file and the line the tied row names", which names no pair.
    grade_fn = need_attr("baselines.grade", "grade")
    rule = _check10_rows()["c10.c03.pairs.split.dolma"]["rule"]
    assert "the chunk file and the line the tied row names" in rule
    concrete = _concrete_of("c03.pairs.split", "small", counted={"small": COUNTED})
    expect = concrete.expect
    assert expect["record"][0]["pair"] == "pair-2"
    q = _make_question("c03.pairs.split", "dolma", "small")
    members = [{"shard": "shard-1", "line": 5}, {"shard": "shard-2", "line": 2}]
    problems = []
    for label, reading in (
        ("no pair name", [{"members": members}]),
        ("another pair name", [{"pair": "pair-9", "members": members}]),
        ("a field beside the member", [{"members": [dict(m, label="observed") for m in members]}]),
    ):
        got = grade_fn(q, expect, _make_reading(q, {"record": reading}))
        if (got.status, got.evidence) != ("right", None):
            problems.append(f"{label}: {got.status} {got.evidence!r}")
    # The chunk file and the line are still read with no pair name.
    moved = [{"members": [{"shard": "shard-1", "line": 6}, members[1]]}]
    got = grade_fn(q, expect, _make_reading(q, {"record": moved}))
    if (got.status, got.evidence) != ("wrong", ["record"]):
        problems.append(f"a moved line: {got.status} {got.evidence!r}")
    assert not problems, "\n".join(problems)


# --- The records rows of the fallback: the word that says how the map was built --------------


def test_the_map_built_by_word_of_the_fallback_is_not_compared_on_the_generated_documents_mapped_row():
    # The text of the table this rests on: under F1 the row's expectation holds map_built_by,
    # and no rule of its group names it.
    grade_fn = need_attr("baselines.grade", "grade")
    listed = "c06.generated.documents-mapped"
    assert _listed_rows()[listed]["expect_under"]["F1"]["map_built_by"] == "token-matching"
    group, _check = _group_of_listed()[listed]
    assert "map_built_by" not in _check10_rows()[f"c10.{group}.dolma"]["rule"]
    concrete = _concrete_of(listed, "small", counted={"small": COUNTED}, fallbacks=("F1",))
    assert concrete.expect["map_built_by"] == "token-matching"
    q = _make_question(listed, "dolma", "small")
    ids = {name: list(held) for name, held in GROUPS_READING.items()}
    problems = []
    for said in ({}, {"map_built_by": "token-matching"}, {"map_built_by": "capture"}):
        raw = dict({"per_ordinal": ids, "unmapped_documents": 0}, **said)
        got = grade_fn(q, concrete.expect, _make_reading(q, raw))
        if (got.status, got.evidence) != ("right", None):
            problems.append(f"{said}: {got.status} {got.evidence!r}")
    assert not problems, "\n".join(problems)


def test_the_rows_that_read_from_records_are_the_two_of_the_stop_and_the_retry_only():
    grade_records = need_attr("baselines.grade", "grade_from_records")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    stop = _make_question("c04.count-499", "dolma")
    with pytest.raises(ValueError):
        grade_records(stop, _expect("c04.count-499"), _make_reading(stop, {"records": {}}))
    q = _make_question("c04.kill-after-100", "dolma")
    expect = _expect("c04.kill-after-100")
    with pytest.raises(not_gradable):
        grade_records(q, expect, None)
    with pytest.raises(not_gradable):
        grade_records(q, expect, _make_attempt())


# --- A line asked in both layouts ------------------------------------------------------------

LAYOUTS = ["partitioned", "round-robin"]
LAYOUT_ROWS = (
    ("c09.10k.successor-shards", "does not contain"),
    ("c09.10k.R1-history", "contains"),
)


def _slots(listed):
    expected_for = need_attr("baselines.expected", "expected_for")
    by_layout = need_attr("baselines.expected", "concrete_values_by_layout")
    return by_layout(expected_for(listed, None))


def test_a_line_asked_in_both_layouts_is_one_grade_that_is_right_only_when_both_layouts_are():
    grade_layouts = need_attr("baselines.grade", "grade_layouts")
    interfaces_grade = need_attr("baselines.interfaces", "Grade")
    for listed, answer in LAYOUT_ROWS:
        assert _listed_rows()[listed]["given"]["layouts"] == LAYOUTS
        assert _listed_rows()[listed]["expect"]["answer"] == answer
        q = _make_question(listed, "manifest")
        assert q.route is not None, listed
        slots = _slots(listed)
        right = _make_reading(q, {"answer": answer})
        wrong = _make_reading(q, {"answer": "not covered"})
        got = grade_layouts(q, slots, {"partitioned": right, "round-robin": right})
        assert isinstance(got, interfaces_grade) and got.status == "right", listed
        assert list(got.evidence) == LAYOUTS
        assert all(slot == {"status": "right", "evidence": None} for slot in got.evidence.values())
        # Wrong in either layout, or in both, is wrong, and the evidence names each layout.
        for outcomes in ({"partitioned": wrong, "round-robin": right},
                         {"partitioned": right, "round-robin": wrong},
                         {"partitioned": wrong, "round-robin": wrong}):
            got = grade_layouts(q, slots, outcomes)
            assert got.status == "wrong", (listed, outcomes)
            assert list(got.evidence) == LAYOUTS
        got = grade_layouts(q, slots, {"partitioned": right, "round-robin": wrong})
        assert got.evidence["partitioned"] == {"status": "right", "evidence": None}
        assert got.evidence["round-robin"] == {"status": "wrong", "evidence": ["answer"]}


def test_a_layout_not_run_leaves_the_line_not_attempted_and_one_with_no_expected_value_cannot_answer():
    grade_layouts = need_attr("baselines.grade", "grade_layouts")
    listed, answer = LAYOUT_ROWS[0]
    q = _make_question(listed, "manifest")
    slots = _slots(listed)
    right = _make_reading(q, {"answer": answer})
    wrong = _make_reading(q, {"answer": "contains"})
    # Changed bar: one layout not run (no reading, or none given) is a layout not attempted,
    # and the line is "not attempted" (it was cannot answer); that layout is named, with the
    # slot "not attempted" (it read "not graded"). If the other layout were wrong the line
    # would be wrong first.
    for outcomes in ({"partitioned": right}, {"partitioned": right, "round-robin": None}):
        got = grade_layouts(q, slots, outcomes)
        assert got.status == "not attempted", outcomes
        assert got.evidence["round-robin"]["status"] == "not attempted"
        assert got.evidence["partitioned"]["status"] == "right"
    got = grade_layouts(q, slots, {"partitioned": wrong})
    assert got.status == "wrong" and got.evidence["round-robin"]["status"] == "not attempted"
    got = grade_layouts(q, slots, {"partitioned": None, "round-robin": None})
    assert got.status == "not attempted"
    # An expected value that could not be derived for one layout is named, never a refusal
    # of the whole line: the other layout keeps its grade.
    problem = ValueError("no counting document for the round-robin layout")
    underived = {"partitioned": slots["partitioned"], "round-robin": problem}
    got = grade_layouts(q, underived, {"partitioned": right, "round-robin": right})
    assert got.status == "cannot answer"
    assert got.evidence["round-robin"]["status"] == "expected value not derived"
    assert "round-robin" in got.evidence["round-robin"]["evidence"]
    assert got.evidence["partitioned"]["status"] == "right"
    # A wrong layout makes the line wrong whatever the other holds.
    got = grade_layouts(q, underived, {"partitioned": wrong, "round-robin": right})
    assert got.status == "wrong"
    assert got.evidence["round-robin"]["status"] == "expected value not derived"


def test_a_failed_attempt_or_a_stranger_in_a_layout_is_not_graded_and_a_layout_not_named_is_refused():
    grade_layouts = need_attr("baselines.grade", "grade_layouts")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    listed, answer = LAYOUT_ROWS[1]
    q = _make_question(listed, "manifest")
    slots = _slots(listed)
    right = _make_reading(q, {"answer": answer})
    wrong = _make_reading(q, {"answer": "does not contain"})
    # Changed bar: a failed attempt is never graded, and it no longer raises: the line is the
    # status of that attempt ("no answer: failed attempt <id>") when no layout is wrong.
    status = "no answer: failed attempt attempt-1"

    def line_of(outcomes):
        """The line, or a failed assertion that says the failed attempt was refused."""
        try:
            return grade_layouts(q, slots, outcomes)
        except not_gradable as problem:
            pytest.fail(f"a failed attempt in a layout is refused, not given its status: {problem}")

    got = line_of({"partitioned": right, "round-robin": _make_attempt()})
    assert got.status == status
    assert got.evidence["round-robin"] == {"status": status, "evidence": None}
    assert got.evidence["partitioned"]["status"] == "right"
    # A wrong layout comes first; a failed attempt comes before a layout not attempted; the
    # first failed attempt in the row's order is the line's status.
    assert line_of({"partitioned": wrong, "round-robin": _make_attempt()}).status == "wrong"
    got = line_of({"partitioned": _make_attempt("attempt-2"), "round-robin": None})
    assert got.status == "no answer: failed attempt attempt-2"
    both = {"partitioned": _make_attempt("attempt-3"), "round-robin": _make_attempt("attempt-4")}
    assert line_of(both).status == "no answer: failed attempt attempt-3"
    other = _make_question(listed, "dolma")
    with pytest.raises(not_gradable):
        grade_layouts(q, slots, {"partitioned": right, "round-robin": _make_reading(other, {"answer": answer})})
    with pytest.raises(ValueError, match="mirrored"):
        grade_layouts(q, slots, {"partitioned": right, "round-robin": right, "mirrored": right})
    # A row that names one layout only is graded by grade(), not here.
    one = _make_question("c09.10k.partitioned.kept-shards", "manifest")
    assert one.route is not None
    one_reading = _make_reading(one, {"keep": [], "every_other_shard": "rewrite"})
    with pytest.raises(ValueError, match="layouts"):
        grade_layouts(one, slots, {"partitioned": one_reading})


def test_a_none_line_of_a_row_that_names_two_layouts_is_not_graded_without_its_attempt():
    grade_layouts = need_attr("baselines.grade", "grade_layouts")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    listed = "c09.10k.R1-history"
    assert _listed_rows()[listed]["given"]["layouts"] == LAYOUTS
    q = _make_question(listed, "dolma")
    assert q.route is None and _table_target(listed, None) == ("set", "10k")
    slots = _slots(listed)
    # The subject holds no record in either layout, so no layout was read.
    none_run = {"partitioned": None, "round-robin": None}
    # No attempt at all: the line is "not attempted", never a status made up from the
    # layouts that were not read.
    with pytest.raises(not_gradable):
        grade_layouts(q, slots, none_run)
    # An attempt on the small set never covers the 10k line (README section 2; the none-line
    # rule), and an attempt of another subject covers nothing of it.
    for stranger in (_ran("dolma", ("set", "small")), _ran("trace", ("set", "10k"))):
        with pytest.raises(not_gradable):
            grade_layouts(q, slots, none_run, attempts=[stranger])
    # Changed bar (new cases): the line needs an attempt per layout, on the set that layout's
    # question is asked of. Both layouts attempted: cannot answer.
    both = [_ran("dolma", ("set", "10k"), layout=layout) for layout in LAYOUTS]
    got = grade_layouts(q, slots, none_run, attempts=both)
    assert got.status == "cannot answer"
    # One layout alone, or an attempt that names no layout, leaves the line not attempted.
    for only in ([both[0]], [both[1]], [_ran("dolma", ("set", "10k"))]):
        with pytest.raises(not_gradable):
            grade_layouts(q, slots, none_run, attempts=only)
    # A failed attempt of one layout is the status of that attempt, never a grade.
    failed = _ran("dolma", ("set", "10k"), layout="round-robin", attempt_id="attempt-3")
    with pytest.raises(not_gradable, match="attempt-3"):
        grade_layouts(q, slots, none_run, attempts=[both[0], failed])


# --- Cannot answer ----------------------------------------------------------------------------


def _table_target(listed_row, size):
    """The store or set the none line of a listed row is asked on, read from the row's
    ``given`` by README section 2 (and not from the code under test): a ``set`` is the set
    of the line's size (a row asked once names its own), else the ``store`` (left out:
    main); a row that names a run or a segment is the store that holds that run, which is
    main for every such row of Checks 1 to 9 that names no store."""
    given = _listed_rows()[listed_row]["given"]
    if "set" in given:
        name = size if size is not None else given["set"]
        # Changed 17 Nov 2026 (old: ("set", name) for every set row): a set row that names a store is the build made in
        # the way of that store, ``<store>-<name>``.
        return ("set", f"{given['store']}-{name}") if given.get("store") is not None else ("set", name)
    return ("store", given.get("store", "main"))


def _ran(subject, target, recorded="the subject's attempt on the store or set ran",
         layout=None, attempt_id=None):
    """One attempt of a subject on a store or set; it names a layout only where one is
    given (a row that names two layouts needs an attempt per layout)."""
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    fields = dict(subject=subject, target=target, attempt_id=attempt_id, recorded=recorded)
    if layout is not None:
        names = {field.name for field in dataclasses.fields(attempt_type)}
        assert "layout" in names, "StoreAttempt has no layout field"
        fields["layout"] = layout
    return attempt_type(**fields)


def _layouts_of(listed_row):
    """The layouts a listed row names in its ``given`` (None for a row that names none)."""
    return _listed_rows()[listed_row]["given"].get("layouts")


def _attempts_for(listed_row, subject, size):
    """The attempts that cover the none line of ``subject`` on its store or set: one, or one
    per layout for a row that names two layouts."""
    target = _table_target(listed_row, size)
    layouts = _layouts_of(listed_row)
    if not layouts:
        return [_ran(subject, target)]
    return [_ran(subject, target, layout=layout) for layout in layouts]


def test_a_row_with_no_record_is_cannot_answer_with_the_text_of_its_missing_field(facts):
    # Changed bar: a none line was cannot answer as soon as the subject had a reading; it is
    # cannot answer only after the subject's attempt on the store or set of its listed row
    # is recorded as having run. A second change: a row that names two layouts needs an
    # attempt per layout (on the set that layout's question is asked of), so one attempt no
    # longer closes its line.
    need_attr("baselines.record", "StoreAttempt")
    rows = [r for r in facts.rows if r["record"] == "none"]
    assert len(rows) > 100
    not_gradable = need_attr("baselines.grade", "NotGradable")
    two_layout_rows = 0
    for row in rows:
        group = row["given"]["group"]
        subject = row["given"]["subject"]
        listed = facts.group_by_id[group]["rows"][0]
        size = facts.sizes_by_listed[listed][0]
        q = _make_question(listed, subject, size)
        assert q.missing == row["missing"]
        expect = _expect(listed)
        attempts = _attempts_for(listed, subject, size)
        assert len(attempts) == len(_layouts_of(listed) or (None,)), listed
        # Whatever the subject's run read, a question the subject holds no record for
        # is never right and never wrong.
        for raw in (None, {}, {"answer": expect.get("answer", "contains")}):
            got = need_attr("baselines.grade", "grade")(q, expect, _make_reading(q, raw), attempts=attempts)
            assert got.status == "cannot answer", (row["id"], raw)
            assert got.evidence == row["missing"], row["id"]
        if _layouts_of(listed):
            # One layout's attempt alone leaves the line not attempted: it is not graded.
            two_layout_rows += 1
            for only in attempts:
                with pytest.raises(not_gradable):
                    need_attr("baselines.grade", "grade")(
                        q, expect, _make_reading(q, None), attempts=[only]
                    )
    # The none rows of the four groups whose first listed row names two layouts
    # (c09.10k.multiset, c09.10k.action-counts, c09.10k.successor-shards and
    # c09.10k.R1-history): 6 + 6 + 1 + 5 subjects, counted from the table's own rows.
    assert two_layout_rows == 18


def test_a_none_line_is_not_graded_without_the_attempt_of_its_subject_on_its_store_or_set():
    grade_fn = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    interfaces = need_attr("baselines.interfaces", "no_answer")
    failed_type = need_attr("baselines.record", "StoreAttempt")
    listed, subject = "c04.count-499", "manifest"
    q = _make_question(listed, subject)
    assert q.route is None and _table_target(listed, None) == ("store", "scratch-500")
    expect = _expect(listed)
    reading = _make_reading(q, None)
    target = ("store", "scratch-500")
    # No attempt at all: the line is "not attempted", and nothing is graded.
    with pytest.raises(not_gradable):
        grade_fn(q, expect, reading, attempts=[])
    # An attempt that ran gives cannot answer, with the question's own missing text.
    got = grade_fn(q, expect, reading, attempts=[_ran(subject, target)])
    assert got.status == "cannot answer" and got.evidence == q.missing
    # An attempt of another subject, or on another store or set, covers nothing of this line.
    for stranger in (_ran("trace", target), _ran(subject, ("store", "main")),
                     _ran(subject, ("set", "small"))):
        with pytest.raises(not_gradable):
            grade_fn(q, expect, reading, attempts=[stranger])
    # A failed attempt is not graded: the line takes the status of that attempt.
    failed = failed_type(subject=subject, target=target, attempt_id="attempt-9",
                         recorded="the store could not be opened")
    with pytest.raises(not_gradable, match="attempt-9"):
        grade_fn(q, expect, reading, attempts=[failed])
    assert interfaces("attempt-9") == "no answer: failed attempt attempt-9"


def test_a_route_that_cannot_be_walked_is_cannot_answer_with_the_evidence_it_names(facts):
    key = need_attr("baselines.grade", "CANNOT_KEY")
    listed = "c05.unmutated.E@9"
    q = _make_question(listed, "dolma+trace")
    assert q.route is not None and q.needs_ordinal
    written = facts.row_by_id["c10.c05.unmutated.checkpoints.dolma+trace"]
    evidence = written["ordinal"]["not_shown"]["evidence"]
    got = _grade(q, _expect(listed), _make_reading(q, {key: evidence}))
    assert got.status == "cannot answer"
    assert got.evidence == evidence
    # Even where an answer was also read, the route that could not be walked decides.
    both = _grade(q, _expect(listed), _make_reading(q, {key: evidence, "answer": "contains"}))
    assert both.status == "cannot answer"


# --- What is never graded -------------------------------------------------------------------------


def _turns_an_attempt_into_a_grade(grade_function) -> bool:
    q = _make_question(ANSWER_ROW, "manifest")
    interfaces_grade = need_attr("baselines.interfaces", "Grade")
    try:
        result = grade_function(q, _expect(ANSWER_ROW), _make_attempt())
    except ValueError:
        return False
    return isinstance(result, interfaces_grade)


def _failed_attempt_as_cannot_answer(real_grade):
    """The known-bad grader: a failed attempt is taken for a subject that ran and
    lacks the record."""

    def bad(question, expect, outcome):
        if isinstance(outcome, need_attr("baselines.interfaces", "Attempt")):
            return _make_grade("cannot answer", "the tool gave nothing")
        return real_grade(question, expect, outcome)

    return bad


def test_a_failed_attempt_is_never_turned_into_a_grade():
    real = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    assert issubclass(not_gradable, ValueError)
    # The same check on the known-bad grader finds the fault, so it can fail.
    assert _turns_an_attempt_into_a_grade(_failed_attempt_as_cannot_answer(real)) is True
    assert _turns_an_attempt_into_a_grade(real) is False
    q = _make_question(ANSWER_ROW, "manifest")
    with pytest.raises(not_gradable):
        real(q, _expect(ANSWER_ROW), _make_attempt())
    # A question that holds no record is no different: a failed attempt is no run.
    none_q = _make_question("c01.E.T", "manifest")
    assert none_q.route is None
    with pytest.raises(not_gradable):
        real(none_q, _expect("c01.E.T"), _make_attempt())


def test_a_subject_that_did_not_run_and_a_row_that_does_not_apply_are_not_graded():
    real = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    q = _make_question(ANSWER_ROW, "manifest")
    with pytest.raises(not_gradable):
        real(q, _expect(ANSWER_ROW), None)
    with pytest.raises(not_gradable):
        real(q, {"applies": False}, _make_reading(q, {"answer": "may contain"}))
    # No run, so no cannot answer either, for a question that holds no record.
    none_q = _make_question("c01.E.T", "manifest")
    with pytest.raises(not_gradable):
        real(none_q, _expect("c01.E.T"), None)


def test_a_reading_of_another_question_is_not_graded():
    real = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    small = _make_question("c03.pairs.same-shard", "dolma", "small")
    big = _make_question("c03.pairs.same-shard", "dolma", "100k")
    assert small.id != big.id
    expect = _expect("c03.pairs.same-shard")
    reading_of_small = _make_reading(small, {"label": "candidates"})
    with pytest.raises(not_gradable):
        real(big, expect, reading_of_small)


def test_a_grade_is_always_one_of_the_three_words(facts):
    interfaces = need_attr("baselines.interfaces", "GRADES")
    q = _make_question(ANSWER_ROW, "manifest")
    expect = _expect(ANSWER_ROW)
    seen = set()
    for raw in ({"answer": "may contain"}, {"answer": "contains"}, {}):
        seen.add(_grade(q, expect, _make_reading(q, raw)).status)
    # Changed bar: the none line is graded once the attempt on its store is recorded.
    none_q = _make_question("c01.E.T", "manifest")
    ran = _ran("manifest", ("store", "main"))
    grade_fn = need_attr("baselines.grade", "grade")
    seen.add(grade_fn(none_q, _expect("c01.E.T"), _make_reading(none_q, None), attempts=[ran]).status)
    assert seen == set(interfaces)
