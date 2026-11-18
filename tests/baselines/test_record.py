"""The record of Check 10 (plan 5.5 Check 10; table README, sections "Marks and the
tally" and "A failed attempt, the two routes after it, and when Check 10 is
complete"): ``baselines.record``.

One record line per (listed row, subject, size at which the tool's own row is asked);
2346 questions. The counts come from the table files by ``tests/baselines/conftest.py``
and the questions are built by the helpers below from ``check10.toml``; the readings, the
grades and the lines are written by hand. No git call is made: the commit is a
hand-written string.

The names this file asks of ``baselines.record``:

- ``NOT_ATTEMPTED``: the status word of a question with no line ("not attempted").
- ``record_of(question, outcome, grade=None, *, used_tie=False)`` returns a
  ``baselines.interfaces.Record``. A ``Reading`` needs a ``Grade`` and gives the
  grade's status, the reading's raw and route, the grade's evidence, the question's
  marks and its ``excluded_from_tally``. A failed ``Attempt`` takes no grade and gives
  the status ``no_answer(attempt.id)``, the attempt's id, no raw and no evidence. The
  mark of the src/loc tie (``question.tie_mark``) is added to the marks only when
  ``used_tie`` is true; a ``used_tie`` for a question that has no such mark, a grade
  with an attempt, a reading with no grade and an outcome whose ``question_id`` is not
  the question's raise ``ValueError``.
- ``answer_line(question, record, *, commit)`` returns the dict that is written for a
  question, JSON-ready: ``id``, ``group``, ``listed_row``, ``subject``, ``size``,
  ``status``, ``raw``, ``evidence``, ``route``, ``marks`` (a list),
  ``excluded_from_tally``, ``attempt_id`` and ``commit``. A commit is 7 to 40 lower
  case hex digits, else ``ValueError``.
- ``write_answers(path, lines)`` and ``read_answers(path)``: one JSON object per line.
  A line that holds a local path, a host name or a session name (anywhere in its
  values) is refused with ``ValueError`` and no file is made.
- ``attempt_line(attempt, question_ids, *, commit)``, ``write_attempts(path, lines)``
  and ``read_attempts(path)``: a failed attempt with its command, exit code, output and
  the ids of the questions it was for; never a status or a grade.
- ``answer_line`` and ``attempt_line`` take ``step_ids=()``: the ids of the steps the line
  rests on (the command, exit and output of each is kept by the runner's steps file). The
  line carries them as the list ``step_ids``, in the order given, ``[]`` where none is given;
  each is text, and one that holds a local path is refused when the line is written.
- The result of the tie show is written and read by ``baselines.tie`` alone (one writer,
  one reader; see ``test_tie_show.py``): this module holds no tie writer or reader of its
  own and reads the result through that one.
- ``attempt_target(question)``: the store or set the none line of a question is asked on,
  as ``("store", <name>)`` or ``("set", <size>)``, read from the listed row's ``given``
  (README section 2; the table's seven stores, the sets ``small``, ``10k`` and ``100k``).
- ``StoreAttempt(subject, target, attempt_id, recorded, layout=None)``: the one attempt of
  a subject on a store or set, in one layout where the listed row names two. ``attempt_id``
  is None when it ran (``recorded`` holds what it recorded: the documented refusal and its
  exit code, a manifest reported absent, a reading) and the id of the failed attempt when it
  failed. ``layout`` is None for a row that names no layout.
- ``none_line_status(question, attempts)``: the status of a line whose check10 row records
  "none" (a question with no route; ``ValueError`` for one that has a route). Cannot
  answer ONLY after an attempt of the question's subject on the question's store or set
  ran; ``no_answer(attempt_id)`` when that attempt failed; ``NOT_ATTEMPTED`` with no
  attempt. One attempt covers every none line of its subject on its store or set. For a row
  that names two layouts the line needs an attempt per layout, and the layouts are put
  together in this order: ``no_answer(attempt_id)`` of the first failed attempt (in the
  row's order), else ``NOT_ATTEMPTED`` if a layout has no attempt, else cannot answer. Two
  attempts for one subject, one target, one layout and one run are a ``ValueError``.
- ``StoreAttempt`` also carries ``run`` (1 for the first run of a store, the default, 2 for the
  second run on it). The two runs of one store are both in the record and neither stands for
  the other: the row ``c04.retry`` reads the attempt of the second run where there is one, every
  other row reads the first; a line citing a failed attempt stands only when it is the attempt
  its row reads.
- ``export_refusal_documented(exit_code, output)``: True for the one refusal docs/cli.md
  documents for a trace export that names no registered run segment (exit 2, "request
  refused"), the subject's own answer that the store holds no run; False for any other
  exit or output, which makes the attempt a failed one.
- ``completeness(questions, lines, attempts=())`` returns a ``Completeness`` with ``statuses`` (a
  mapping from the question id to its one status, ``NOT_ATTEMPTED`` where no line
  exists, and for a line of cannot answer of a question with no route: the status of its
  attempt, ``none_line_status``), ``open`` (the ids, in question order, whose status is no grade: not
  attempted or a failed attempt), ``by_size`` (a mapping from "once", "small", "10k",
  "100k" to a result with ``total``, ``graded``, ``open`` and ``complete``),
  ``complete`` (nothing open) and ``plan_sizes_complete`` (the "10k" and "100k" results
  are both complete). ``RecordError`` (a ``ValueError``) is raised for two lines of one
  question, a line of no question, a line whose ``size`` is not its question's, a status
  that is neither a grade nor a failed attempt's, and a line of right or wrong for a
  question whose record is none: such a question can only be cannot answer (table README,
  the rows with no record), so a right or wrong line is refused whatever attempts are given.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import tomllib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"

# A commit is passed to the writers as a hand-written string; no git call is made.
COMMIT = "0123456789abcdef0123456789abcdef01234567"

# What stays out of a record, written in pieces so that this file holds none of it.
PRIVATE_PATH = "/" + "ho" + "me/" + "someone/work/build"
PRIVATE_FOLDER = "Proj" + "ects/" + "thing"
PRIVATE_SESSION = "C" + "&" + "C"


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


ANSWERED = "c05.unmutated.R1"
CHECKPOINT = "c05.unmutated.E@9"
PAIRS = "c03.pairs.same-shard"


def _fn(name):
    return need_attr("baselines.record", name)


# --- One record of one question --------------------------------------------------------------


def test_a_graded_reading_is_recorded_with_its_grade_marks_route_and_flag():
    record_of = _fn("record_of")
    q = _make_question(CHECKPOINT, "dolma+trace")
    reading = _make_reading(q, {"answer": "contains"}, route="the route named first")
    grade = _make_grade("right", None)
    rec = record_of(q, reading, grade)
    interfaces = need_attr("baselines.interfaces", "Record")
    assert isinstance(rec, interfaces)
    assert rec.question_id == q.id
    assert rec.status == "right"
    assert rec.raw == {"answer": "contains"}
    assert rec.route == "the route named first"
    assert tuple(rec.marks) == q.marks == ("step-to-ordinal part is the tool's own output",)
    assert rec.excluded_from_tally is q.excluded_from_tally
    assert rec.attempt_id is None
    cannot = record_of(q, reading, _make_grade("cannot answer", "the missing tie"))
    assert cannot.status == "cannot answer" and cannot.evidence == "the missing tie"


def test_the_mark_of_the_tie_is_carried_only_by_an_answer_that_used_the_tie():
    record_of = _fn("record_of")
    mark = need_attr("baselines.interfaces", "MARK_TIE")
    q = _make_question(CHECKPOINT, "dolma+trace")
    assert q.tie_mark == mark and q.needs_ordinal
    reading = _make_reading(q, {"answer": "contains"})
    grade = _make_grade("right")
    unused = record_of(q, reading, grade)
    used = record_of(q, reading, grade, used_tie=True)
    assert mark not in unused.marks
    assert tuple(used.marks) == q.marks + (mark,)
    # The tie was not used where it was not shown: that answer carries no tie mark.
    not_shown = record_of(q, _make_reading(q, {"cannot_answer": "the missing tie"}),
                          _make_grade("cannot answer", "the missing tie"))
    assert mark not in not_shown.marks
    # A question with no tie to use cannot say it used one.
    plain = _make_question(ANSWERED, "manifest")
    assert plain.tie_mark is None
    with pytest.raises(ValueError):
        record_of(plain, _make_reading(plain, {"answer": "contains"}), grade, used_tie=True)


def test_a_failed_attempt_is_recorded_as_no_answer_and_never_as_a_grade():
    record_of = _fn("record_of")
    interfaces = need_attr("baselines.interfaces", "no_answer")
    grades = need_attr("baselines.interfaces", "GRADES")
    q = _make_question(CHECKPOINT, "dolma+trace")
    attempt = _make_attempt("attempt-7")
    rec = record_of(q, attempt)
    assert rec.status == interfaces("attempt-7") == "no answer: failed attempt attempt-7"
    assert rec.status not in grades
    assert rec.attempt_id == "attempt-7"
    assert rec.raw is None and rec.evidence is None
    assert tuple(rec.marks) == q.marks and rec.excluded_from_tally is q.excluded_from_tally
    # A grade cannot be put on a failed attempt, not even cannot answer.
    with pytest.raises(ValueError):
        record_of(q, attempt, _make_grade("cannot answer", "no record"))


def test_a_reading_needs_a_grade_and_belongs_to_its_own_question():
    record_of = _fn("record_of")
    small = _make_question(PAIRS, "dolma", "small")
    big = _make_question(PAIRS, "dolma", "100k")
    with pytest.raises(ValueError):
        record_of(small, _make_reading(small, {"label": "candidates"}))
    # The reading of the small set is never the record of the 100k question.
    with pytest.raises(ValueError):
        record_of(big, _make_reading(small, {"label": "candidates"}), _make_grade("right"))


# --- The line that is written ---------------------------------------------------------------------


def test_an_answer_line_names_the_question_its_size_and_the_commit_it_ran_on():
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    q = _make_question(PAIRS, "dolma", "100k")
    rec = record_of(q, _make_reading(q, {"label": "candidates"}), _make_grade("right"))
    line = answer_line(q, rec, commit=COMMIT)
    assert line["id"] == q.id
    assert (line["group"], line["listed_row"], line["subject"], line["size"]) == (
        q.group, PAIRS, "dolma", "100k",
    )
    assert line["status"] == "right"
    assert line["commit"] == COMMIT
    assert line["marks"] == list(q.marks)
    assert line["excluded_from_tally"] is q.excluded_from_tally
    assert line["attempt_id"] is None
    assert json.loads(json.dumps(line)) == line
    # A row asked once has size None; the line says so.
    once = _make_question(ANSWERED, "manifest")
    once_rec = record_of(once, _make_reading(once, {"answer": "contains"}), _make_grade("wrong"))
    assert answer_line(once, once_rec, commit=COMMIT)["size"] is None
    # Another question's record is not written under this question.
    with pytest.raises(ValueError):
        answer_line(once, rec, commit=COMMIT)


@pytest.mark.parametrize("commit", ["", "abc", "not-a-commit-at-all", "0123456789ABCDEF0123", "z" * 40, None])
def test_a_line_without_a_real_commit_is_refused(commit):
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    q = _make_question(ANSWERED, "manifest")
    rec = record_of(q, _make_reading(q, {"answer": "contains"}), _make_grade("right"))
    with pytest.raises((ValueError, TypeError)):
        answer_line(q, rec, commit=commit)


def test_answers_are_written_one_json_object_per_line_and_read_back(tmp_path):
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    write_answers, read_answers = _fn("write_answers"), _fn("read_answers")
    lines = []
    for size in ("small", "100k"):
        q = _make_question(PAIRS, "dolma", size)
        rec = record_of(q, _make_reading(q, {"label": "candidates", "n": 3}), _make_grade("right"))
        lines.append(answer_line(q, rec, commit=COMMIT))
    path = tmp_path / "answers.jsonl"
    write_answers(path, lines)
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) == 2
    assert [json.loads(t) for t in text.splitlines()] == lines
    assert read_answers(path) == lines


@pytest.mark.parametrize("private", [PRIVATE_PATH, PRIVATE_FOLDER, PRIVATE_SESSION])
def test_a_record_that_holds_a_local_path_or_a_session_name_is_not_written(tmp_path, private):
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    write_answers = _fn("write_answers")
    q = _make_question(ANSWERED, "manifest")
    raws = ({"answer": "contains", "where": private}, {"deep": [{"x": f"a {private} b"}]})
    for number, raw in enumerate(raws):
        rec = record_of(q, _make_reading(q, raw), _make_grade("right"))
        line = answer_line(q, rec, commit=COMMIT)
        path = tmp_path / f"answers-{number}.jsonl"
        with pytest.raises(ValueError):
            write_answers(path, [line])
        assert not path.exists()


# --- The attempts -----------------------------------------------------------------------------


def test_an_attempt_is_written_with_its_command_and_output_and_never_a_grade(tmp_path):
    attempt_line = _fn("attempt_line")
    write_attempts, read_attempts = _fn("write_attempts"), _fn("read_attempts")
    attempt = _make_attempt("attempt-3", exit_code=0, output="no row for chunk file 2")
    line = attempt_line(attempt, ["q1", "q2"], commit=COMMIT)
    assert line["id"] == "attempt-3"
    assert line["command"] == ["tool", "run", "input.jsonl"]
    assert line["exit_code"] == 0
    assert line["output"] == "no row for chunk file 2"
    assert line["question_ids"] == ["q1", "q2"]
    assert line["commit"] == COMMIT
    assert not {"status", "grade", "evidence", "marks"} & set(line)
    path = tmp_path / "attempts.jsonl"
    write_attempts(path, [line])
    assert read_attempts(path) == [line]
    # An attempt that did not end by itself has no exit code.
    killed = attempt_line(_make_attempt("attempt-4", exit_code=None), ["q3"], commit=COMMIT)
    assert killed["exit_code"] is None


def test_an_attempt_that_holds_a_local_path_is_not_written(tmp_path):
    attempt_line, write_attempts = _fn("attempt_line"), _fn("write_attempts")
    dirty = _make_attempt("attempt-5", output="failed in " + PRIVATE_PATH)
    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ValueError):
        write_attempts(path, [attempt_line(dirty, ["q1"], commit=COMMIT)])
    assert not path.exists()


def test_a_traceback_that_holds_a_local_path_is_not_written_and_the_same_traceback_without_it_is(tmp_path):
    attempt_line, write_attempts, read_attempts = (
        _fn("attempt_line"), _fn("write_attempts"), _fn("read_attempts")
    )
    traceback = (
        "Traceback (most recent call last):\n"
        f'  File "{PRIVATE_PATH}/tool.py", line 3, in run\n'
        "ValueError: no rows\n"
    )
    path = tmp_path / "attempts.jsonl"
    with pytest.raises(ValueError):
        write_attempts(path, [attempt_line(_make_attempt("attempt-8", output=traceback), ["q1"], commit=COMMIT)])
    assert not path.exists()
    stripped = traceback.replace(f"{PRIVATE_PATH}/", "")
    line = attempt_line(_make_attempt("attempt-8", output=stripped), ["q1"], commit=COMMIT)
    write_attempts(path, [line])
    assert read_attempts(path) == [line]


def test_a_record_that_holds_the_host_name_of_the_machine_is_not_written(tmp_path, monkeypatch):
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    write_answers, attempt_line, write_attempts = (
        _fn("write_answers"), _fn("attempt_line"), _fn("write_attempts")
    )
    # A made-up host name: the writers read the name of the machine they run on.
    host = "madeup" + "-box-7"
    monkeypatch.setattr("socket.gethostname", lambda: host)
    q = _make_question(ANSWERED, "manifest")
    rec = record_of(q, _make_reading(q, {"answer": "contains", "where": f"ran on {host}"}),
                    _make_grade("right"))
    answers = tmp_path / "answers.jsonl"
    with pytest.raises(ValueError):
        write_answers(answers, [answer_line(q, rec, commit=COMMIT)])
    assert not answers.exists()
    attempts = tmp_path / "attempts.jsonl"
    dirty = _make_attempt("attempt-6", output=f"failed on {host}")
    with pytest.raises(ValueError):
        write_attempts(attempts, [attempt_line(dirty, ["q1"], commit=COMMIT)])
    assert not attempts.exists()
    # The same line without the name is written.
    clean = record_of(q, _make_reading(q, {"answer": "contains"}), _make_grade("right"))
    write_answers(answers, [answer_line(q, clean, commit=COMMIT)])
    assert answers.exists()


# The result of the tie show has one writer and one reader, both in baselines/tie.py: the
# test of that pair, and of this module's use of it, is in test_tie_show.py (a changed bar:
# the test that stood here, of a second writer and reader in this module, is removed).


# --- Completeness ---------------------------------------------------------------------------------------


def _hand_line(question, status="right"):
    return {"id": question.id, "size": question.size, "status": status}


# A sized row whose Dolma row holds a record (a route) at the small set and at 100k. The
# questions of PAIRS (c03.pairs.same-shard) hold no record for Dolma, so a right or a wrong
# line on them is refused (see the test of that refusal below): the four questions used
# below are all questions that hold a record.
SIZED_ROUTED = "c03.pairs.split"


def _four(facts):
    """One question of each size: asked once, small, 10k and 100k. Each holds a record."""
    ten_k = next(l for l, s in facts.sizes_by_listed.items() if "10k" in s)
    return {
        "once": _make_question(ANSWERED, "manifest"),
        "small": _make_question(SIZED_ROUTED, "dolma", "small"),
        "100k": _make_question(SIZED_ROUTED, "dolma", "100k"),
        "10k": _make_question(ten_k, "dolma", "10k"),
    }


def _refuses_a_grade_on_a_none_question(completeness, record_error, size) -> bool:
    """True when ``completeness`` refuses a line of right, and one of wrong, for the
    question of PAIRS (a question whose record is none for Dolma) at ``size``: the old
    questions of ``_four``, which a right line used to close."""
    none_question = _make_question(PAIRS, "dolma", size)
    assert none_question.route is None
    for status in ("right", "wrong"):
        try:
            completeness([none_question], [_hand_line(none_question, status)])
        except record_error:
            continue
        return False
    return True


def test_a_question_with_no_line_is_not_attempted_and_leaves_check_10_open(facts):
    completeness = _fn("completeness")
    not_attempted = _fn("NOT_ATTEMPTED")
    assert not_attempted == "not attempted"
    qs = list(_four(facts).values())
    got = completeness(qs, [])
    assert dict(got.statuses) == {q.id: not_attempted for q in qs}
    assert tuple(got.open) == tuple(q.id for q in qs)
    assert got.complete is False and got.plan_sizes_complete is False
    assert set(got.by_size) == {"once", "small", "10k", "100k"}
    assert all(r.total == 1 and r.graded == 0 and r.complete is False for r in got.by_size.values())


def test_every_question_graded_is_complete(facts):
    completeness = _fn("completeness")
    qs = _four(facts)
    statuses = {"once": "right", "small": "wrong", "10k": "cannot answer", "100k": "right"}
    got = completeness(list(qs.values()), [_hand_line(qs[k], statuses[k]) for k in qs])
    assert got.complete is True and got.plan_sizes_complete is True
    assert tuple(got.open) == ()
    assert {q.id: got.statuses[q.id] for q in qs.values()} == {qs[k].id: statuses[k] for k in qs}
    assert all(r.complete and r.graded == 1 for r in got.by_size.values())
    # Changed bar (the questions): the small and the 100k questions used to be questions of
    # PAIRS, whose record is none for Dolma; a right or a wrong line on such a question is
    # refused, so the questions here hold a record.
    record_error = _fn("RecordError")
    for size in ("small", "100k"):
        assert _refuses_a_grade_on_a_none_question(completeness, record_error, size), size


def test_a_failed_attempt_has_a_status_and_leaves_the_question_open(facts):
    completeness = _fn("completeness")
    qs = _four(facts)
    no_answer = need_attr("baselines.interfaces", "no_answer")("attempt-2")
    lines = [_hand_line(qs[k]) for k in ("once", "small", "10k")] + [_hand_line(qs["100k"], no_answer)]
    got = completeness(list(qs.values()), lines)
    assert got.statuses[qs["100k"].id] == no_answer
    assert tuple(got.open) == (qs["100k"].id,)
    assert got.by_size["100k"].complete is False and got.by_size["100k"].graded == 0
    assert got.by_size["small"].complete is True
    assert got.complete is False and got.plan_sizes_complete is False
    # Changed bar (the small question): a question that holds a record, since a right line
    # on a question whose record is none is refused.
    assert _refuses_a_grade_on_a_none_question(completeness, _fn("RecordError"), "small")


def test_a_size_that_was_not_run_stays_open_while_the_small_set_is_complete(facts):
    completeness = _fn("completeness")
    qs = _four(facts)
    lines = [_hand_line(qs["once"]), _hand_line(qs["small"])]
    got = completeness(list(qs.values()), lines)
    assert got.by_size["small"].complete is True
    assert got.by_size["once"].complete is True
    assert got.by_size["100k"].complete is False
    assert tuple(got.by_size["100k"].open) == (qs["100k"].id,)
    assert got.statuses[qs["100k"].id] == _fn("NOT_ATTEMPTED")
    assert got.plan_sizes_complete is False and got.complete is False
    # The plan sizes are complete apart from the small set and the questions asked once.
    only_plan = completeness(list(qs.values()), [_hand_line(qs["10k"]), _hand_line(qs["100k"])])
    assert only_plan.plan_sizes_complete is True and only_plan.complete is False
    assert only_plan.by_size["small"].complete is False
    # Changed bar (the small and 100k questions): questions that hold a record; the same lines
    # on the questions of PAIRS (record none) are refused.
    record_error = _fn("RecordError")
    for size in ("small", "100k"):
        assert _refuses_a_grade_on_a_none_question(completeness, record_error, size), size


def _small_closes_the_100k_line(completeness_function, qs) -> bool:
    """True when the small line of a sized row is taken for the 100k line of the same
    row and subject: the plan sizes read as complete from the small line alone."""
    lines = [_hand_line(qs["small"])]
    got = completeness_function([qs["small"], qs["100k"]], lines)
    return bool(got.plan_sizes_complete)


def _completeness_keyed_without_the_size(questions, lines):
    """The known-bad checker: a question is closed by any line of its listed row and
    subject, whatever the size."""
    answered = {(l["id"].split("|")[0], l["id"].split("|")[1].split("@")[0]) for l in lines}
    plan = [q for q in questions if q.size in ("10k", "100k")]
    return SimpleNamespace(
        plan_sizes_complete=all((q.listed_row, q.subject) in answered for q in plan)
    )


def test_a_small_line_never_closes_a_100k_line(facts):
    completeness = _fn("completeness")
    qs = _four(facts)
    # The same check on the known-bad checker finds the fault, so it can fail.
    assert _small_closes_the_100k_line(_completeness_keyed_without_the_size, qs) is True
    assert _small_closes_the_100k_line(completeness, qs) is False
    # Changed bar (the questions): a small line of right on the question of PAIRS (record
    # none) is refused, not read as a status that leaves the plan sizes open, so the pair of
    # questions here holds a record.
    assert qs["small"].route is not None and qs["100k"].route is not None
    assert _refuses_a_grade_on_a_none_question(completeness, _fn("RecordError"), "small")


def test_the_ten_k_rows_of_check_9_are_asked_once_and_the_sized_rows_at_ten_k(facts):
    completeness = _fn("completeness")
    ten_k_once = [l for l in facts.sizes_by_listed if l.startswith("c09.10k.")]
    assert ten_k_once and all(facts.sizes_by_listed[l] == (None,) for l in ten_k_once)
    q = _make_question(ten_k_once[0], "manifest")
    assert q.size is None
    got = completeness([q], [])
    assert set(got.by_size) == {"once", "small", "10k", "100k"}
    assert {label: r.total for label, r in got.by_size.items()} == {
        "once": 1, "small": 0, "10k": 0, "100k": 0,
    }


def test_a_line_that_breaks_the_one_status_of_a_question_is_an_error(facts):
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    assert issubclass(record_error, ValueError)
    qs = _four(facts)
    questions = list(qs.values())
    # Two lines for one question.
    with pytest.raises(record_error):
        completeness(questions, [_hand_line(qs["once"]), _hand_line(qs["once"], "wrong")])
    # A line of no question.
    stranger = {"id": "c99.none|manifest", "size": None, "status": "right"}
    with pytest.raises(record_error):
        completeness(questions, [stranger])
    # A line that says a size other than its question's.
    wrong_size = dict(_hand_line(qs["100k"]), size="small")
    with pytest.raises(record_error):
        completeness(questions, [wrong_size])
    # Changed bar (the 100k question): it holds a record, so the size is the one reason this
    # line is refused; the same right line on the 100k question of PAIRS (record none) is
    # refused for its own reason, the record that is none.
    assert qs["100k"].route is not None
    assert _refuses_a_grade_on_a_none_question(completeness, record_error, "100k")
    # A status that is no grade and no failed attempt.
    with pytest.raises(record_error):
        completeness(questions, [_hand_line(qs["once"], "maybe")])
    with pytest.raises(record_error):
        completeness(questions, [_hand_line(qs["once"], "not attempted")])


def test_a_question_left_out_of_the_tally_still_needs_its_line(facts):
    completeness = _fn("completeness")
    excluded = _make_question("c01.E.T", "trace")
    assert excluded.excluded_from_tally is True
    got = completeness([excluded], [])
    assert tuple(got.open) == (excluded.id,)
    done = completeness([excluded], [_hand_line(excluded)])
    assert done.complete is True


def test_the_whole_table_asks_2346_questions_and_each_size_is_counted_by_itself(facts):
    completeness = _fn("completeness")
    questions = [_make_question(l, s, size) for l, s, size in facts.question_triples()]
    assert len(questions) == facts.questions_count == 2346
    assert len({q.id for q in questions}) == 2346
    # Changed bar: a right line on a question whose record is none is refused (1609 of the
    # questions); those lines are cannot answer, each closed by the attempt of its subject on
    # its store or set, and by one attempt per layout for the rows that name two.
    with pytest.raises(_fn("RecordError")):
        completeness(questions, [_hand_line(q) for q in questions])
    lines = _table_lines(questions)
    attempts = _attempts_of_the_table(questions)
    got = completeness(questions, lines, attempts=attempts)
    assert got.complete is True and got.plan_sizes_complete is True
    counted = {label: result.total for label, result in got.by_size.items()}
    assert counted == {
        "once": facts.questions_by_size[None],
        "small": facts.questions_by_size["small"],
        "10k": facts.questions_by_size["10k"],
        "100k": facts.questions_by_size["100k"],
    }
    assert sum(counted.values()) == 2346
    # The questions of the plan sizes are not run: 24 + 102 stay open, the small set
    # and the rows asked once are complete, and the plan sizes are open.
    partial = [l for l, q in zip(lines, questions) if q.size not in ("10k", "100k")]
    open_run = completeness(questions, partial, attempts=attempts)
    assert len(open_run.open) == 24 + 102
    assert open_run.by_size["small"].complete is True
    assert open_run.by_size["once"].complete is True
    assert open_run.by_size["10k"].complete is False and open_run.by_size["100k"].complete is False
    assert open_run.plan_sizes_complete is False and open_run.complete is False
    # One failed attempt among the rows asked once opens that size and nothing else. Changed
    # bar: the line is of the first question asked once that holds a record (a failed state of
    # a none question comes only from a failed attempt, which opens every none line of its
    # subject on that store).
    first_once = next(q for q in questions if q.size is None and q.route is not None)
    one_failed = [dict(l, status="no answer: failed attempt x") if l["id"] == first_once.id else l for l in lines]
    got = completeness(questions, one_failed, attempts=attempts)
    assert tuple(got.open) == (first_once.id,)
    assert got.by_size["once"].complete is False
    assert got.by_size["small"].complete and got.plan_sizes_complete


# --- The none-line rule ---------------------------------------------------------------------------
# A line whose check10 row records "none" is cannot answer ONLY after the subject's attempt on
# the store or set its listed row names is recorded as having run. One attempt on a store or
# set covers every none line of it, of its subject. With no attempt the line is "not attempted"
# and Check 10 stays open.

README_STORES = ("main", "rr", "swap", "no-capture", "import", "scratch-500", "scratch-notokens")
SET_NAMES = ("small", "10k", "100k")


@lru_cache(maxsize=1)
def _given_of_listed() -> dict[str, dict]:
    """The ``given`` of every core row of Checks 1 to 9, as the files write it."""
    found = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                found[row["id"]] = row["given"]
    return found


def _table_target(listed_row, size):
    """The store or set a none line is asked on, worked out here from the row's ``given``
    by README section 2 and never from the code under test: a ``set`` is the set of the
    line's size (a row asked once names its own set), else the ``store``, left out meaning
    main; a row that names a run or a segment is the store that holds it, main for every
    such row that names no store."""
    given = _given_of_listed()[listed_row]
    if "set" in given:
        name = size if size is not None else given["set"]
        if given.get("store") == "no-capture":
            # Changed 17 Nov 2026 (old: ("set", name) for every ``set`` row): a generated set made without capture is a
            # build of its own, so the rows that name it (set and store no-capture) ask the target of that build.
            return ("set", f"no-capture-{name}")
        return ("set", name)
    return ("store", given.get("store", "main"))


def _is_none(listed_row, subject):
    group, _check = _group_of_listed()[listed_row]
    return _check10_rows()[f"c10.{group}.{subject}"].get("route") is None


def _none_triples(facts):
    return [(l, s, size) for l, s, size in facts.question_triples() if _is_none(l, s)]


def _attempt(subject, target, attempt_id, recorded, layout):
    """A ``StoreAttempt``; it names a layout only where one is given (a row that names two
    layouts needs an attempt per layout)."""
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    fields = dict(subject=subject, target=target, attempt_id=attempt_id, recorded=recorded)
    if layout is not None:
        names = {field.name for field in dataclasses.fields(attempt_type)}
        assert "layout" in names, "StoreAttempt has no layout field"
        fields["layout"] = layout
    return attempt_type(**fields)


def _ran(subject, target, recorded="the attempt ran and its answer is recorded", layout=None):
    return _attempt(subject, target, None, recorded, layout)


def _failed(subject, target, attempt_id, layout=None):
    return _attempt(subject, target, attempt_id, "the command failed", layout)


def _layouts_of(listed_row):
    """The layouts a listed row names in its ``given``, in the row's order (None for a row
    that names none)."""
    return _given_of_listed()[listed_row].get("layouts")


def _attempts_of_the_table(questions):
    """One attempt for every subject on every store or set its none lines are asked on, and
    for a row that names two layouts one per layout."""
    keys = set()
    for question in questions:
        if question.route is None:
            target = _table_target(question.listed_row, question.size)
            for layout in _layouts_of(question.listed_row) or (None,):
                keys.add((question.subject, target, layout))
    return [_ran(subject, target, layout=layout)
            for subject, target, layout in sorted(keys, key=repr)]


def _table_lines(questions):
    """The line of every question as a hand-written status: cannot answer where the record is
    none, right where the question holds one."""
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    return [_hand_line(q, cannot if q.route is None else "right") for q in questions]


def test_the_store_or_set_of_every_none_line_is_the_one_the_table_names(facts):
    target_of = _fn("attempt_target")
    triples = _none_triples(facts)
    assert len(triples) == facts.no_record_count == 1609
    seen = set()
    for listed, subject, size in triples:
        question = _make_question(listed, subject, size)
        assert tuple(target_of(question)) == _table_target(listed, size), question.id
        seen.add(_table_target(listed, size))
    # The stores and sets the 1609 lines are asked on: every store of the README, and the
    # three sets; a small attempt is a set of its own, not the 100k set.
    # (old: the three sets; new: the three sets and the three generated sets made without capture)
    assert seen == ({("store", name) for name in README_STORES} | {("set", name) for name in SET_NAMES}
                    | {("set", f"no-capture-{name}") for name in SET_NAMES})
    # The edges by name, written by hand from the rows' own ``given``.
    edges = (
        ("c01.E-rr.diff", "manifest", None, ("store", "rr")),
        ("c01.K.diff", "manifest", None, ("store", "main")),
        ("c09.E.stale-cache.E/seg-2", "manifest", None, ("store", "main")),
        ("c03.precondition.eod-convention", "manifest", None, ("set", "100k")),
        ("c03.spans.per-ordinal", "manifest", "small", ("set", "small")),
        ("c03.spans.per-ordinal", "manifest", "100k", ("set", "100k")),
        ("c06.generated.candidates", "manifest", "small", ("set", "no-capture-small")),
        ("c06.generated.candidates", "manifest", "10k", ("set", "no-capture-10k")),
        ("c06.generated.candidates", "manifest", "100k", ("set", "no-capture-100k")),
        ("c06.fixture.drops", "manifest", None, ("store", "no-capture")),
        ("c09.10k.multiset", "manifest", None, ("set", "10k")),
        ("c04.count-499", "manifest", None, ("store", "scratch-500")),
        ("c03.drops.no-tokens.planted-line", "manifest", None, ("store", "scratch-notokens")),
        ("c06.import.record-request", "manifest", None, ("store", "import")),
        ("c07.swap.report.counts", "manifest", None, ("store", "swap")),
    )
    in_table = set(triples)
    for listed, subject, size, target in edges:
        assert (listed, subject, size) in in_table, (listed, subject, size)
        assert tuple(target_of(_make_question(listed, subject, size))) == target, (listed, size)
    # A question that has a route is not asked on a store for its none line.
    routed = _make_question(ANSWERED, "manifest")
    assert routed.route is not None


def test_only_the_rows_that_name_a_set_and_the_store_no_capture_have_a_new_target_and_every_other_row_keeps_its_old_one(facts):
    """The target of every question of the whole table, against the rule the README gave before the generated sets
    made without capture (a ``set`` is the set of the size, else the ``store``, main where it is left out): the
    rows whose ``given`` names a ``set`` AND the store ``no-capture`` are the only ones whose target changes, to the
    build of their own, taken from that ``given``; no other row moves, and the name cannot be a folder of a size."""
    target_of = _fn("attempt_target")
    moved, total = set(), 0
    for listed, subject, size in facts.question_triples():
        given = _given_of_listed()[listed]
        question = _make_question(listed, subject, size)
        old = (("set", size if size is not None else given["set"]) if "set" in given
               else ("store", given.get("store", "main")))
        got = tuple(target_of(question))
        total += 1
        if "set" in given and given.get("store") == "no-capture":
            assert got == ("set", f"no-capture-{size}"), question.id
            moved.add(listed)
        else:
            assert got == old, f"{question.id}: {got} moved from {old}"
    assert total > 2000 and moved == {"c06.generated.documents-mapped", "c06.generated.drops",
                                       "c06.generated.candidates", "c06.generated.downstream-of-translation"}, moved
    # A row that names a set and a store other than no-capture would be a new case: none exists.
    assert not [l for l, g in _given_of_listed().items() if "set" in g and g.get("store") not in (None, "no-capture")]
    # The key of that build in the world (``set-no-capture-<size>``) is no folder the larger sizes' copies use
    # (``set-<size>``).
    sizes = ("small", "10k", "100k")
    assert not ({f"set-no-capture-{name}" for name in sizes} & {f"set-{name}" for name in sizes})


def test_a_none_line_is_cannot_answer_only_after_an_attempt_of_its_subject_on_its_store_or_set():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    not_attempted = _fn("NOT_ATTEMPTED")
    q = _make_question("c04.count-499", "manifest")
    assert q.route is None
    target = ("store", "scratch-500")
    # No attempt: not attempted, the line is not written.
    assert status_of(q, []) == not_attempted
    # An attempt that ran: cannot answer.
    assert status_of(q, [_ran("manifest", target)]) == cannot
    # An attempt of another subject, on another store or on a set, covers nothing of this line.
    for stranger in (_ran("trace", target), _ran("manifest", ("store", "main")),
                     _ran("manifest", ("set", "small")), _ran("dolma", ("store", "scratch-500"))):
        assert status_of(q, [stranger]) == not_attempted, stranger
    # The attempt is found among others.
    held = [_ran("trace", target), _ran("manifest", ("store", "main")), _ran("manifest", target)]
    assert status_of(q, held) == cannot
    # A question that holds a record is no none line.
    routed = _make_question(ANSWERED, "manifest")
    with pytest.raises(ValueError):
        status_of(routed, [_ran("manifest", ("store", "main"))])
    # Two attempts of one subject on one store or set cannot both stand.
    with pytest.raises(ValueError):
        status_of(q, [_ran("manifest", target), _failed("manifest", target, "attempt-2")])


def test_one_attempt_covers_every_none_line_of_its_subject_on_its_store_or_set(facts):
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    not_attempted = _fn("NOT_ATTEMPTED")
    attempts = [_ran("manifest", ("store", "main"))]
    covered = uncovered = 0
    for listed, subject, size in _none_triples(facts):
        question = _make_question(listed, subject, size)
        covers = subject == "manifest" and _table_target(listed, size) == ("store", "main")
        got = status_of(question, attempts)
        assert got == (cannot if covers else not_attempted), question.id
        covered += covers
        uncovered += not covers
    # One attempt closes many lines (the none lines of the manifest on the main store) and
    # nothing else.
    assert covered > 100 and uncovered > covered


def test_a_small_attempt_never_covers_a_10k_or_a_100k_none_line():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    not_attempted = _fn("NOT_ATTEMPTED")
    small = _make_question("c03.spans.per-ordinal", "manifest", "small")
    big = _make_question("c03.spans.per-ordinal", "manifest", "100k")
    ten = _make_question("c06.generated.candidates", "manifest", "10k")
    assert small.route is None and big.route is None and ten.route is None
    attempts = [_ran("manifest", ("set", "small"))]
    assert status_of(small, attempts) == cannot
    assert status_of(big, attempts) == not_attempted
    assert status_of(ten, attempts) == not_attempted
    assert status_of(big, [_ran("manifest", ("set", "100k"))]) == cannot
    assert status_of(ten, [_ran("manifest", ("set", "100k"))]) == not_attempted


def test_a_failed_attempt_gives_its_none_lines_the_status_of_that_attempt_and_no_grade():
    status_of = _fn("none_line_status")
    interfaces = need_attr("baselines.interfaces", "no_answer")
    grades = need_attr("baselines.interfaces", "GRADES")
    q = _make_question("c01.E.T", "manifest")
    got = status_of(q, [_failed("manifest", ("store", "main"), "attempt-4")])
    assert got == interfaces("attempt-4") == "no answer: failed attempt attempt-4"
    assert got not in grades


def test_the_documented_refusal_of_a_trace_on_a_store_with_no_run_is_an_attempt_that_ran():
    refused = _fn("export_refusal_documented")
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    # The documented case, as docs/cli.md writes it: an export that names no registered run
    # segment exits 2 with "request refused".
    cli = (REPO / "docs" / "cli.md").read_text(encoding="utf-8")
    assert '| 2 | "request refused": the name is no registered run segment |' in cli
    assert refused(2, "waterwhite: request refused: the name is no registered run segment") is True
    assert refused(2, "request refused") is True
    # Any other exit or output is not the documented case: it is a failed attempt.
    for exit_code, output in (
        (0, ""), (1, "request refused"), (3, "request refused"), (2, "not a store"),
        (2, ""), (None, "request refused"), (2, "Traceback (most recent call last):"),
    ):
        assert refused(exit_code, output) is False, (exit_code, output)
    # A store with no registered run (scratch-500) answered the trace's export with that
    # refusal: the attempt ran, its exit and message are recorded, and the none lines of the
    # trace on that store read cannot answer.
    q = _make_question("c04.count-499", "trace")
    assert q.route is None and _table_target("c04.count-499", None) == ("store", "scratch-500")
    ran = _ran("trace", ("store", "scratch-500"), recorded={"exit_code": 2, "output": "request refused"})
    assert status_of(q, [ran]) == cannot


def test_a_manifest_reported_absent_on_a_build_made_without_capture_is_an_attempt_that_ran():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    # The build ran and the reader reports the manifest absent; that is recorded and the none
    # lines of the manifest on the no-capture store are cannot answer.
    q = _make_question("c06.fixture.drops", "manifest")
    assert q.route is None and _table_target("c06.fixture.drops", None) == ("store", "no-capture")
    absent = _ran("manifest", ("store", "no-capture"), recorded="the manifest is absent: the build was made with capture off")
    assert status_of(q, [absent]) == cannot


def test_completeness_does_not_count_a_none_line_of_cannot_answer_that_has_no_attempt(facts):
    completeness = _fn("completeness")
    not_attempted = _fn("NOT_ATTEMPTED")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    no_answer = need_attr("baselines.interfaces", "no_answer")("attempt-4")
    on_main = _make_question("c01.E.T", "manifest")
    on_scratch = _make_question("c04.count-499", "manifest")
    routed = _make_question(ANSWERED, "manifest")
    assert on_main.route is None and on_scratch.route is None and routed.route is not None
    questions = [on_main, on_scratch, routed]
    lines = [_hand_line(on_main, cannot), _hand_line(on_scratch, cannot), _hand_line(routed, "right")]
    # With no attempt a line of cannot answer is not a status the question holds.
    got = completeness(questions, lines)
    assert got.statuses[on_main.id] == not_attempted
    assert got.statuses[on_scratch.id] == not_attempted
    assert got.statuses[routed.id] == "right"
    assert tuple(got.open) == (on_main.id, on_scratch.id)
    assert got.complete is False
    assert got.by_size["once"].graded == 1 and got.by_size["once"].complete is False
    # The attempt on the main store closes the line of that store and no other.
    one = completeness(questions, lines, attempts=[_ran("manifest", ("store", "main"))])
    assert one.statuses[on_main.id] == cannot and one.statuses[on_scratch.id] == not_attempted
    assert tuple(one.open) == (on_scratch.id,)
    both = completeness(questions, lines, attempts=[_ran("manifest", ("store", "main")),
                                                    _ran("manifest", ("store", "scratch-500"))])
    assert both.complete is True and tuple(both.open) == ()
    # A failed attempt makes its lines "no answer: failed attempt <id>", open.
    failed = completeness(questions, lines, attempts=[_failed("manifest", ("store", "main"), "attempt-4"),
                                                      _ran("manifest", ("store", "scratch-500"))])
    assert failed.statuses[on_main.id] == no_answer
    assert tuple(failed.open) == (on_main.id,) and failed.complete is False
    # A line of cannot answer for a question that holds a record is a grade of its own and
    # needs no attempt (the route could not be walked).
    walked = _make_question("c05.unmutated.E@9", "dolma+trace")
    assert walked.route is not None
    done = completeness([walked], [_hand_line(walked, cannot)])
    assert done.complete is True and done.statuses[walked.id] == cannot


def test_a_10k_and_a_100k_none_line_stay_open_while_only_a_small_attempt_exists():
    completeness = _fn("completeness")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    small = _make_question("c03.spans.per-ordinal", "manifest", "small")
    big = _make_question("c03.spans.per-ordinal", "manifest", "100k")
    ten = _make_question("c06.generated.candidates", "manifest", "10k")
    questions = [small, big, ten]
    lines = [_hand_line(q, cannot) for q in questions]
    got = completeness(questions, lines, attempts=[_ran("manifest", ("set", "small"))])
    assert got.by_size["small"].complete is True
    assert got.by_size["100k"].complete is False and got.by_size["10k"].complete is False
    assert got.plan_sizes_complete is False and got.complete is False
    assert tuple(got.open) == (big.id, ten.id)
    # Changed 17 Nov 2026: the generated row `ten` is asked of the set made without capture (old: ("set", "10k")).
    full = completeness(questions, lines, attempts=[_ran("manifest", ("set", "small")),
                                                    _ran("manifest", ("set", "no-capture-10k")),
                                                    _ran("manifest", ("set", "100k"))])
    assert full.complete is True and full.plan_sizes_complete is True


def test_completeness_with_the_attempts_of_every_store_and_set_closes_the_2346_questions(facts):
    completeness = _fn("completeness")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    questions = [_make_question(l, s, size) for l, s, size in facts.question_triples()]
    lines = [_hand_line(q, cannot if q.route is None else "right") for q in questions]
    # No attempt: the 1609 none lines are open and the 737 others are graded.
    bare = completeness(questions, lines)
    assert len(bare.open) == facts.no_record_count
    # One attempt for every subject on every store and set its none lines are asked on, and
    # (changed bar) one per layout for the rows that name two layouts: one attempt on the set
    # no longer closes those lines.
    attempts = _attempts_of_the_table(questions)
    got = completeness(questions, lines, attempts=attempts)
    assert got.complete is True and tuple(got.open) == ()
    # Only one layout's attempts: the 42 none lines of the two-layout rows stay open, and
    # nothing else does.
    two_layout = tuple(
        q.id for q in questions if q.route is None and _layouts_of(q.listed_row)
    )
    assert len(two_layout) == 42
    first_layout_only = [a for a in attempts if a.layout != "round-robin"]
    half = completeness(questions, lines, attempts=first_layout_only)
    assert tuple(half.open) == two_layout
    # A failed attempt of one subject on one store opens exactly its none lines, with the
    # status of that attempt.
    main_target = ("store", "main")
    failed_one = [a for a in attempts if (a.subject, a.target) != ("manifest", main_target)]
    failed_one.append(_failed("manifest", main_target, "attempt-x"))
    opened = tuple(
        q.id for q in questions
        if q.route is None and q.subject == "manifest" and _table_target(q.listed_row, q.size) == main_target
    )
    assert len(opened) > 100
    failed = completeness(questions, lines, attempts=failed_one)
    assert tuple(failed.open) == opened
    assert {failed.statuses[name] for name in opened} == {"no answer: failed attempt attempt-x"}
    # Drop the attempts of the 100k set: those lines, and only those, are open again.
    kept = [a for a in attempts if a.target != ("set", "100k")]
    partial = completeness(questions, lines, attempts=kept)
    wanted = tuple(
        q.id for q in questions
        if q.route is None and _table_target(q.listed_row, q.size) == ("set", "100k")
    )
    assert wanted and tuple(partial.open) == wanted
    assert partial.by_size["100k"].complete is False and partial.by_size["10k"].complete is True


# --- A row that names two layouts needs an attempt per layout ---------------------------------
# The eight listed rows of c09.10k that name the layouts "partitioned" and "round-robin" are
# asked on the one set 10k, so the store or set of their none lines stays ("set", "10k") (the
# test of attempt_target above is unchanged); what tells the two attempts of one subject on
# that set apart is the layout the attempt carries. The line needs both, and the two are put
# together in the order: a failed attempt (the first in the row's order), else not attempted,
# else cannot answer.

TWO_LAYOUTS = ["partitioned", "round-robin"]
TWO_LAYOUT_NONE_ROW = "c09.10k.multiset"


def test_an_attempt_carries_its_layout_and_the_eight_two_layout_rows_give_42_none_questions(facts):
    attempt_type = _fn("StoreAttempt")
    names = [field.name for field in dataclasses.fields(attempt_type)]
    assert "layout" in names, "StoreAttempt has no layout field"
    plain = attempt_type(subject="manifest", target=("store", "main"), attempt_id=None, recorded="x")
    assert plain.layout is None
    named = attempt_type(subject="manifest", target=("set", "10k"), attempt_id=None, recorded="x",
                         layout="partitioned")
    assert named.layout == "partitioned"
    # Counted from the table's own rows: the none rows of the eight listed rows of c09.10k that
    # name both layouts (24 of the repair comparison, 12 of the record actions, 1 of the
    # successor shards, 5 of the history of R1), all asked on the set 10k.
    target_of = _fn("attempt_target")
    layout_rows = sorted(l for l, given in _given_of_listed().items() if "layouts" in given)
    assert len(layout_rows) == 8
    assert all(_layouts_of(l) == TWO_LAYOUTS for l in layout_rows)
    questions = [_make_question(l, s, size) for l, s, size in _none_triples(facts) if l in layout_rows]
    assert len(questions) == 42
    assert {tuple(target_of(q)) for q in questions} == {("set", "10k")}
    by_row = {}
    for q in questions:
        by_row[q.listed_row] = by_row.get(q.listed_row, 0) + 1
    assert sorted(by_row.values()) == [1, 5, 6, 6, 6, 6, 6, 6]


def test_two_attempts_of_one_subject_on_one_target_are_told_apart_by_their_layout():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    q = _make_question(TWO_LAYOUT_NONE_ROW, "manifest")
    assert q.route is None and _table_target(TWO_LAYOUT_NONE_ROW, None) == ("set", "10k")
    target = ("set", "10k")
    # An invariant widened from "one subject, one target" to "one subject, one target, one
    # layout": the two layouts of one subject on one set both stand.
    both = [_ran("manifest", target, layout=layout) for layout in TWO_LAYOUTS]
    assert status_of(q, both) == cannot
    # The same layout twice cannot stand, whether the second attempt ran or failed.
    with pytest.raises(ValueError):
        status_of(q, both + [_ran("manifest", target, layout="partitioned")])
    with pytest.raises(ValueError):
        status_of(q, both + [_failed("manifest", target, "attempt-2", layout="round-robin")])


def test_a_none_line_of_a_row_that_names_two_layouts_needs_an_attempt_per_layout():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    not_attempted = _fn("NOT_ATTEMPTED")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    q = _make_question(TWO_LAYOUT_NONE_ROW, "manifest")
    target = ("set", "10k")
    first, second = TWO_LAYOUTS
    # No attempt, or one layout's attempt alone, or an attempt that names no layout: the line
    # is not attempted.
    assert status_of(q, []) == not_attempted
    assert status_of(q, [_ran("manifest", target, layout=first)]) == not_attempted
    assert status_of(q, [_ran("manifest", target, layout=second)]) == not_attempted
    assert status_of(q, [_ran("manifest", target)]) == not_attempted
    # Both layouts attempted: cannot answer.
    both = [_ran("manifest", target, layout=first), _ran("manifest", target, layout=second)]
    assert status_of(q, both) == cannot
    # A failed attempt of either layout is the status of that attempt, also when the other
    # layout has no attempt at all (a failed attempt comes before not attempted).
    assert status_of(q, [_failed("manifest", target, "a-1", layout=first),
                         _ran("manifest", target, layout=second)]) == failed_status("a-1")
    assert status_of(q, [_ran("manifest", target, layout=first),
                         _failed("manifest", target, "a-2", layout=second)]) == failed_status("a-2")
    assert status_of(q, [_failed("manifest", target, "a-1", layout=first)]) == failed_status("a-1")
    # Both failed: the first in the row's order, whichever way the attempts are listed.
    two_failed = [_failed("manifest", target, "a-1", layout=first),
                  _failed("manifest", target, "a-2", layout=second)]
    assert status_of(q, two_failed) == failed_status("a-1")
    assert status_of(q, two_failed[::-1]) == failed_status("a-1")
    # An attempt of another subject, or on the small set, covers no layout of this line.
    strangers = [_ran("trace", target, layout=layout) for layout in TWO_LAYOUTS]
    strangers += [_ran("manifest", ("set", "small"), layout=layout) for layout in TWO_LAYOUTS]
    assert status_of(q, strangers) == not_attempted


def test_completeness_closes_a_none_line_of_two_layouts_only_with_both_layouts_attempts():
    completeness = _fn("completeness")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    not_attempted = _fn("NOT_ATTEMPTED")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    q = _make_question(TWO_LAYOUT_NONE_ROW, "manifest")
    target = ("set", "10k")
    first, second = TWO_LAYOUTS
    lines = [_hand_line(q, cannot)]
    one = completeness([q], lines, attempts=[_ran("manifest", target, layout=first)])
    assert one.statuses[q.id] == not_attempted and tuple(one.open) == (q.id,)
    both = completeness([q], lines, attempts=[_ran("manifest", target, layout=first),
                                              _ran("manifest", target, layout=second)])
    assert both.statuses[q.id] == cannot and tuple(both.open) == ()
    failed = completeness([q], lines, attempts=[_ran("manifest", target, layout=first),
                                                _failed("manifest", target, "a-9", layout=second)])
    assert failed.statuses[q.id] == failed_status("a-9") and tuple(failed.open) == (q.id,)


def test_a_right_or_wrong_line_on_a_question_whose_record_is_none_is_refused(facts):
    # The table README: a row whose record is none can only be graded cannot answer, so no
    # question that holds no record is ever right or wrong, with or without its attempts.
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    triples = _none_triples(facts)
    assert len(triples) == facts.no_record_count == 1609
    for listed, subject, size in triples:
        q = _make_question(listed, subject, size)
        attempts = _attempts_of_the_table([q])
        for status in ("right", "wrong"):
            with pytest.raises(record_error):
                completeness([q], [_hand_line(q, status)], attempts=attempts)
        # Cannot answer, after its attempts, is the one line such a question can hold.
        assert completeness([q], [_hand_line(q, cannot)], attempts=attempts).complete is True, q.id
    # A question that holds a record keeps right and wrong lines.
    routed = _make_question(ANSWERED, "manifest")
    for status in ("right", "wrong"):
        assert completeness([routed], [_hand_line(routed, status)]).complete is True


def test_a_failed_attempt_line_on_a_none_question_stands_only_with_that_failed_attempt():
    # A line "no answer: failed attempt <id>" is the status of a failed attempt the run made, and
    # of no other: on a question whose record is none it stands only when the attempts it is
    # given hold a failed attempt with that id, of that subject, on that store or set.
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    q = _make_question("c01.E.T", "manifest")
    assert q.route is None and _table_target("c01.E.T", None) == ("store", "main")
    status = failed_status("attempt-4")
    line = _hand_line(q, status)
    main = ("store", "main")
    # The failed attempt of that id, subject and store: the line stands, and the question is open.
    got = completeness([q], [line], attempts=[_failed("manifest", main, "attempt-4")])
    assert got.statuses[q.id] == status and tuple(got.open) == (q.id,)
    # It is found among attempts that are not it.
    among = [_ran("trace", main), _failed("manifest", ("store", "rr"), "attempt-9"),
             _failed("manifest", main, "attempt-4")]
    assert completeness([q], [line], attempts=among).statuses[q.id] == status
    # No attempt at all, or any attempt that is not that one, leaves the line refused.
    not_it = (
        [],
        [_failed("manifest", main, "attempt-5")],
        [_failed("trace", main, "attempt-4")],
        [_failed("manifest", ("store", "rr"), "attempt-4")],
        [_failed("manifest", ("set", "small"), "attempt-4")],
        [_ran("manifest", main)],
    )
    for attempts in not_it:
        with pytest.raises(record_error):
            completeness([q], [line], attempts=attempts)
    # A row that names two layouts: the failed attempt of one layout is the attempt of that id.
    layered = _make_question(TWO_LAYOUT_NONE_ROW, "manifest")
    assert layered.route is None
    target = ("set", "10k")
    first, second = TWO_LAYOUTS
    layered_line = _hand_line(layered, failed_status("a-9"))
    with pytest.raises(record_error):
        completeness([layered], [layered_line])
    attempts = [_ran("manifest", target, layout=first),
                _failed("manifest", target, "a-9", layout=second)]
    assert completeness([layered], [layered_line], attempts=attempts).statuses[layered.id] == failed_status("a-9")
    # A question that holds a record keeps a failed-attempt line with no attempts: the line is
    # its own status there (the attempt that failed was made for it).
    routed = _make_question(ANSWERED, "manifest")
    assert routed.route is not None
    assert completeness([routed], [_hand_line(routed, status)]).statuses[routed.id] == status


# --- A store that was run twice -----------------------------------------------------------------------
# A store that holds a stopped write and a second run on the same output (the scratch store of
# Check 4) has two attempts of one subject on one target. Both are in the record; each row reads
# the attempt of the run it asks about: c04.retry the second run (where there is one), every other
# row the first, so that a failed second run does not fail the rows gated on the first run, and a
# good second run does not hide a failed first.

SCRATCH_500 = ("store", "scratch-500")


def _run_attempt(subject, target, attempt_id, run, layout=None):
    attempt_type = need_attr("baselines.record", "StoreAttempt")
    names = {field.name for field in dataclasses.fields(attempt_type)}
    assert "run" in names, "StoreAttempt has no run field"
    fields = dict(subject=subject, target=target, attempt_id=attempt_id,
                  recorded="the run's own record", run=run)
    if layout is not None:
        fields["layout"] = layout
    return attempt_type(**fields)


def test_the_two_runs_of_one_store_are_both_in_the_record_and_raise_nothing():
    status_of = _fn("none_line_status")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    names = {field.name for field in dataclasses.fields(_fn("StoreAttempt"))}
    assert "run" in names, "StoreAttempt has no run field"
    plain = _fn("StoreAttempt")(subject="manifest", target=SCRATCH_500, attempt_id=None, recorded="x")
    assert plain.run == 1
    first = _run_attempt("manifest", SCRATCH_500, None, 1)
    second = _run_attempt("manifest", SCRATCH_500, None, 2)
    assert (first.run, second.run) == (1, 2)
    for listed in ("c04.kill-after-100", "c04.retry", "c04.count-499"):
        q = _make_question(listed, "manifest")
        assert q.route is None and _table_target(listed, None) == SCRATCH_500
        assert status_of(q, [first, second]) == cannot
        assert status_of(q, [second, first]) == cannot


def test_each_row_of_a_store_run_twice_reads_the_attempt_of_the_run_it_asks_about():
    status_of = _fn("none_line_status")
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    kill = _make_question("c04.kill-after-100", "manifest")
    retry = _make_question("c04.retry", "manifest")
    # The second run failed: c04.retry reads it, c04.kill-after-100 reads the first run.
    second_failed = [_run_attempt("manifest", SCRATCH_500, None, 1),
                     _run_attempt("manifest", SCRATCH_500, "attempt-12", 2)]
    assert status_of(kill, second_failed) == cannot
    assert status_of(retry, second_failed) == failed_status("attempt-12")
    # The first run failed and the second ran: c04.kill-after-100 reads the failed first run,
    # c04.retry reads the second run, which ran.
    first_failed = [_run_attempt("manifest", SCRATCH_500, "attempt-11", 1),
                    _run_attempt("manifest", SCRATCH_500, None, 2)]
    assert status_of(kill, first_failed) == failed_status("attempt-11")
    assert status_of(retry, first_failed) == cannot
    # Both failed: each row reads its own run, whatever the order the attempts are given in.
    both = [_run_attempt("manifest", SCRATCH_500, "attempt-11", 1),
            _run_attempt("manifest", SCRATCH_500, "attempt-12", 2)]
    for attempts in (both, both[::-1]):
        assert status_of(kill, attempts) == failed_status("attempt-11")
        assert status_of(retry, attempts) == failed_status("attempt-12")
    # A line that cites a failed attempt stands only with the attempt its row reads.
    got = completeness([kill, retry], [_hand_line(kill, failed_status("attempt-11")),
                                       _hand_line(retry, failed_status("attempt-12"))], attempts=both)
    assert got.statuses[kill.id] == failed_status("attempt-11")
    assert got.statuses[retry.id] == failed_status("attempt-12")
    assert tuple(got.open) == (kill.id, retry.id)
    with pytest.raises(record_error):
        completeness([retry], [_hand_line(retry, failed_status("attempt-11"))], attempts=both)
    with pytest.raises(record_error):
        completeness([kill], [_hand_line(kill, failed_status("attempt-12"))], attempts=both)


def test_a_row_gated_on_the_first_run_is_not_failed_by_a_failed_second_run_nor_cleared_by_a_good_one():
    status_of = _fn("none_line_status")
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    gated = _make_question("c04.declare-499.write", "dolma")
    assert gated.route is None and _table_target("c04.declare-499.write", None) == SCRATCH_500
    # A failed second run does not fail the row.
    second_failed = [_run_attempt("dolma", SCRATCH_500, None, 1),
                     _run_attempt("dolma", SCRATCH_500, "attempt-22", 2)]
    assert status_of(gated, second_failed) == cannot
    assert completeness([gated], [_hand_line(gated, cannot)], attempts=second_failed).complete is True
    with pytest.raises(record_error):
        completeness([gated], [_hand_line(gated, failed_status("attempt-22"))], attempts=second_failed)
    # A good second run does not hide a failed first run.
    first_failed = [_run_attempt("dolma", SCRATCH_500, "attempt-21", 1),
                    _run_attempt("dolma", SCRATCH_500, None, 2)]
    assert status_of(gated, first_failed) == failed_status("attempt-21")
    got = completeness([gated], [_hand_line(gated, failed_status("attempt-21"))], attempts=first_failed)
    assert got.statuses[gated.id] == failed_status("attempt-21") and tuple(got.open) == (gated.id,)


def test_a_stop_that_did_not_land_is_the_first_run_and_does_not_hide_behind_the_second():
    status_of = _fn("none_line_status")
    completeness = _fn("completeness")
    record_error = _fn("RecordError")
    failed_status = need_attr("baselines.interfaces", "no_answer")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    kill = _make_question("c04.kill-after-100", "manifest")
    retry = _make_question("c04.retry", "manifest")
    gated = _make_question("c04.declare-499.write", "manifest")
    assert gated.route is None
    # The stop did not land: its attempt is the only attempt of the store, and every row of it
    # reads that attempt (no second run was made, so c04.retry has none of its own).
    stopped = [_run_attempt("manifest", SCRATCH_500, "stop-1", 1)]
    for q in (kill, retry, gated):
        assert status_of(q, stopped) == failed_status("stop-1"), q.id
    # A second run that did run, or failed, beside it: the rows gated on the first run and the
    # stopped write keep the stop's attempt; c04.retry reads the second run.
    ran_after = stopped + [_run_attempt("manifest", SCRATCH_500, None, 2)]
    failed_after = stopped + [_run_attempt("manifest", SCRATCH_500, "attempt-13", 2)]
    assert status_of(kill, ran_after) == failed_status("stop-1")
    assert status_of(gated, ran_after) == failed_status("stop-1")
    assert status_of(retry, ran_after) == cannot
    assert status_of(retry, failed_after) == failed_status("attempt-13")
    assert status_of(gated, failed_after) == failed_status("stop-1")
    # The lines stand with exactly the attempt their row reads.
    got = completeness([kill, retry], [_hand_line(kill, failed_status("stop-1")),
                                       _hand_line(retry, failed_status("attempt-13"))], attempts=failed_after)
    assert tuple(got.open) == (kill.id, retry.id)
    with pytest.raises(record_error):
        completeness([retry], [_hand_line(retry, failed_status("stop-1"))], attempts=failed_after)


def test_two_attempts_that_are_not_two_runs_of_one_store_still_raise():
    status_of = _fn("none_line_status")
    q = _make_question("c04.declare-499.write", "dolma")
    # The same run twice, whether the second ran or failed.
    with pytest.raises(ValueError):
        status_of(q, [_run_attempt("dolma", SCRATCH_500, None, 1), _run_attempt("dolma", SCRATCH_500, "a-1", 1)])
    with pytest.raises(ValueError):
        status_of(q, [_run_attempt("dolma", SCRATCH_500, None, 2), _run_attempt("dolma", SCRATCH_500, None, 2)])
    # A layout twice in one run is no second run of the store.
    layered = _make_question(TWO_LAYOUT_NONE_ROW, "manifest")
    target = ("set", "10k")
    first, second = TWO_LAYOUTS
    fine = [_run_attempt("manifest", target, None, 1, layout=first),
            _run_attempt("manifest", target, None, 1, layout=second)]
    assert status_of(layered, fine) == need_attr("baselines.interfaces", "CANNOT_ANSWER")
    with pytest.raises(ValueError):
        status_of(layered, fine + [_run_attempt("manifest", target, None, 1, layout=first)])


# --- The steps a line rests on -------------------------------------------------------------------------


def test_a_line_names_the_steps_it_rests_on_in_the_order_given():
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    attempt_line = _fn("attempt_line")
    for function in (answer_line, attempt_line):
        assert "step_ids" in inspect.signature(function).parameters, function.__name__
    q = _make_question(ANSWERED, "manifest")
    rec = record_of(q, _make_reading(q, {"answer": "contains"}), _make_grade("right"))
    line = answer_line(q, rec, commit=COMMIT, step_ids=["step-3", "step-1"])
    assert line["step_ids"] == ["step-3", "step-1"]
    assert answer_line(q, rec, commit=COMMIT)["step_ids"] == []
    assert json.loads(json.dumps(line)) == line
    # A reading that failed: the line of its attempt names the steps too.
    attempt = _make_attempt("attempt-6")
    failed = attempt_line(attempt, ["q1", "q2"], commit=COMMIT, step_ids=("step-2",))
    assert failed["step_ids"] == ["step-2"]
    assert attempt_line(attempt, ["q1"], commit=COMMIT)["step_ids"] == []
    # The other keys of both lines are as they were.
    assert failed["id"] == "attempt-6" and failed["question_ids"] == ["q1", "q2"]
    assert line["id"] == q.id and line["status"] == "right"


def test_a_step_id_is_text_and_a_step_id_that_holds_a_local_path_is_not_written(tmp_path):
    record_of, answer_line = _fn("record_of"), _fn("answer_line")
    attempt_line, write_answers, write_attempts = _fn("attempt_line"), _fn("write_answers"), _fn("write_attempts")
    q = _make_question(ANSWERED, "manifest")
    rec = record_of(q, _make_reading(q, {"answer": "contains"}), _make_grade("right"))
    assert "step_ids" in inspect.signature(answer_line).parameters
    for bad in ("step-1", [3], [None], [["step-1"]]):
        with pytest.raises(ValueError):
            answer_line(q, rec, commit=COMMIT, step_ids=bad)
        with pytest.raises(ValueError):
            attempt_line(_make_attempt("attempt-7"), ["q1"], commit=COMMIT, step_ids=bad)
    dirty = answer_line(q, rec, commit=COMMIT, step_ids=[PRIVATE_PATH])
    path = tmp_path / "answers.jsonl"
    with pytest.raises(ValueError):
        write_answers(path, [dirty])
    assert not path.exists()
    dirty_attempt = attempt_line(_make_attempt("attempt-8"), ["q1"], commit=COMMIT, step_ids=[PRIVATE_PATH])
    attempts = tmp_path / "attempts.jsonl"
    with pytest.raises(ValueError):
        write_attempts(attempts, [dirty_attempt])
    assert not attempts.exists()
