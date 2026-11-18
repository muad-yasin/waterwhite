"""The summary of Check 10 (plan 5.5 Check 10; table README, section "Marks and the tally" and
"A failed attempt, the two routes after it, and when Check 10 is complete"): ``baselines.summary``.

The summary counts the questions of a run by subject, by size label and by status word, the tally
leaves out the questions the table excludes, and it says whether Check 10 is complete or open and
which questions are open. The small set is kept apart from the plan sizes. It holds no sentence
that sets one tool against another.

The names this file asks of ``baselines.summary``:

- ``WORDS``: ``right``, ``wrong``, ``cannot answer``, ``failed attempt`` and ``not attempted``.
- ``summarize(questions, lines, attempts=())`` gives a summary. ``lines`` are answer lines (they
  need hold only ``id``, ``size`` and ``status``); the one status of every question is the one
  ``baselines.record.completeness`` gives. The summary has ``count(subject, label, word)`` (every
  question), ``tally_count(subject, label, word)`` (the questions the table does not exclude),
  ``complete``, ``open`` (the ids, in question order), ``plan_sizes_complete`` and
  ``size_complete`` (a mapping from ``once``, ``small``, ``10k`` and ``100k`` to a flag).
- ``answered(subject, label)`` (a right or a wrong), ``right_among_answered(subject, label)`` and, in the tally,
  ``tally_answered`` and ``tally_right_among_answered``: the two counts of the owner's tally answer of 18 Nov 2026, "could it
  answer" and "right among those answered", counted apart.
- ``render(summary)`` gives the text of the summary. It holds the line ``Check 10: complete`` or
  ``Check 10: open``, for each label the line ``<label>: complete`` or ``<label>: open``, for each
  non-zero count the line ``<subject> <label> <word>: <count>`` (and the same behind ``tally`` for
  the tally), for each subject and label that answered anything the lines ``<subject> <label> answered: <n>`` and
  ``<subject> <label> right among answered: <n>`` (and behind ``tally``), and each open question's id on a line of its own.

The questions are built here from ``check10.toml`` by the reading the table README gives; the
statuses are written by hand.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from functools import lru_cache
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"

# Old: five words. New: a sixth, "expected value not derived", for a line the grader could not grade (a question that
# holds a record whose expected value is not derived): it is never counted as the subject's "cannot answer".
WORDS = ("right", "wrong", "cannot answer", "failed attempt", "expected value not derived", "not attempted")
NOT_DERIVED = "expected value not derived"
LABELS = ("once", "small", "10k", "100k")

# The words of a sentence that sets one tool against another.
COMPARISON_WORDS = re.compile(
    r"\b(better|worse|best|worst|outperform\w*|beats?|superior|inferior|than|versus|ahead|behind|"
    r"advantage|wins|loses|winner|loser)\b",
    re.IGNORECASE,
)


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


def _fn(name):
    return need_attr("baselines.summary", name)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _read_toml(TABLE_DIR / "check10.toml")["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, tuple[str, int]]:
    found = {}
    for group in _read_toml(TABLE_DIR / "check10.toml")["group"]:
        for listed in group["rows"]:
            found[listed] = (group["id"], group["check"])
    return found


def _make_question(listed_row: str, subject: str, size=None):
    question_type = need_attr("baselines.interfaces", "Question")
    group, check = _group_of_listed()[listed_row]
    row = _check10_rows()[f"c10.{group}.{subject}"]
    ordinal = row.get("ordinal")
    base = f"{listed_row}|{subject}"
    return question_type(
        id=base if size is None else f"{base}@{size}",
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


def _method(summary, name):
    """A method of a summary, or a failed assertion that says it has none."""
    found = getattr(summary, name, None)
    assert found is not None, f"the summary has no {name}"
    return found


def _line(question, status):
    return {"id": question.id, "size": question.size, "status": status}


# The questions of the small example: two asked once, a sized row at the small set and at 100k, and a
# question the table leaves out of the tally. Every one holds a record.
SIZED_ROUTED = "c03.pairs.split"


def _example():
    once_manifest = _make_question("c05.unmutated.R1", "manifest")
    once_dolma = _make_question("c05.unmutated.E/shard-1", "dolma")
    small = _make_question(SIZED_ROUTED, "dolma", "small")
    big = _make_question(SIZED_ROUTED, "dolma", "100k")
    excluded = _make_question("c01.E.T", "trace")
    for question in (once_manifest, once_dolma, small, big, excluded):
        assert question.route is not None, question.id
    assert excluded.excluded_from_tally is True
    assert not any(q.excluded_from_tally for q in (once_manifest, once_dolma, small, big))
    questions = [once_manifest, once_dolma, small, big, excluded]
    lines = [
        _line(once_manifest, "right"),
        _line(once_dolma, "wrong"),
        _line(small, "cannot answer"),
        _line(excluded, "no answer: failed attempt attempt-9"),
    ]
    return questions, lines


def test_the_questions_are_counted_by_subject_size_and_status_word():
    summarize = _fn("summarize")
    assert tuple(_fn("WORDS")) == WORDS
    questions, lines = _example()
    got = summarize(questions, lines)
    assert got.count("manifest", "once", "right") == 1
    assert got.count("dolma", "once", "wrong") == 1
    assert got.count("dolma", "small", "cannot answer") == 1
    assert got.count("trace", "once", "failed attempt") == 1
    # A question with no line is not attempted, and it is counted at its own size.
    assert got.count("dolma", "100k", "not attempted") == 1
    # Everything else is zero, and the counts add up to the questions.
    total = 0
    for subject in need_attr("baselines.interfaces", "SUBJECTS"):
        for label in LABELS:
            for word in WORDS:
                total += got.count(subject, label, word)
    assert total == len(questions) == 5
    assert got.count("dolma", "small", "right") == 0 and got.count("dolma", "100k", "right") == 0


def test_a_line_the_grader_could_not_grade_is_never_counted_as_the_subjects_cannot_answer():
    """The runner keeps the line of a question whose expected value is not derived (or that is not gradable) with the
    status "cannot answer" and the words as its evidence; the summary counts it under its own word, and the question
    stays open, so a graded summary never reads it as an answer the subject gave."""
    summarize = _fn("summarize")
    render = _fn("render")
    questions, lines = _example()
    small = next(q for q in questions if q.id.endswith("@small"))
    for evidence in ("expected value not derived: the part is missing", "not gradable: no file derived for A/r1"):
        changed = [dict(line, evidence=evidence) if line["id"] == small.id else line for line in lines]
        got = summarize(questions, changed)
        assert got.count("dolma", "small", NOT_DERIVED) == 1
        assert got.count("dolma", "small", "cannot answer") == 0, "a line the grader could not grade was counted as cannot answer"
        assert small.id in got.open and got.complete is False
        assert f"dolma small {NOT_DERIVED}: 1" in render(got) and "dolma small cannot answer" not in render(got)
    # A cannot answer line the subject gave (its own evidence) is counted as before.
    genuine = [dict(line, evidence="the toolkit holds no derived-from link") if line["id"] == small.id else line for line in lines]
    assert summarize(questions, genuine).count("dolma", "small", "cannot answer") == 1
    assert summarize(questions, lines).count("dolma", "small", "cannot answer") == 1


def test_the_tally_leaves_out_the_questions_the_table_excludes():
    summarize = _fn("summarize")
    questions, lines = _example()
    got = summarize(questions, lines)
    # The excluded question is counted and is not in the tally.
    assert got.count("trace", "once", "failed attempt") == 1
    assert got.tally_count("trace", "once", "failed attempt") == 0
    # Every other count is the same in the tally.
    for subject, label, word in (("manifest", "once", "right"), ("dolma", "once", "wrong"),
                                 ("dolma", "small", "cannot answer"), ("dolma", "100k", "not attempted")):
        assert got.tally_count(subject, label, word) == got.count(subject, label, word) == 1


def test_a_run_is_complete_only_when_no_question_is_open_and_the_open_ones_are_listed_in_order():
    summarize = _fn("summarize")
    questions, lines = _example()
    got = summarize(questions, lines)
    assert got.complete is False
    # The 100k question has no line; the excluded one holds a failed attempt: both are open.
    assert tuple(got.open) == (questions[3].id, questions[4].id)
    graded = [_line(q, "right") for q in questions]
    done = summarize(questions, graded)
    assert done.complete is True and tuple(done.open) == () and done.plan_sizes_complete is True


def test_the_small_set_is_kept_apart_from_the_plan_sizes():
    summarize = _fn("summarize")
    questions, lines = _example()
    got = summarize(questions, lines)
    # The small question is graded; the 100k question of the same row is not, and stays open.
    assert got.size_complete["small"] is True
    assert got.size_complete["100k"] is False
    assert got.size_complete["10k"] is True
    assert got.size_complete["once"] is False
    assert got.plan_sizes_complete is False
    assert got.count("dolma", "small", "cannot answer") == 1 and got.count("dolma", "100k", "cannot answer") == 0
    # A small line never closes a 100k question.
    only_small = summarize([questions[2], questions[3]], [lines[2]])
    assert only_small.size_complete["small"] is True and only_small.size_complete["100k"] is False
    assert only_small.plan_sizes_complete is False and only_small.complete is False


def test_a_none_line_is_counted_only_after_the_attempt_of_its_subject_closed_it():
    summarize = _fn("summarize")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    question = _make_question("c01.E.T", "manifest")
    assert question.route is None
    lines = [_line(question, cannot)]
    bare = summarize([question], lines)
    assert bare.count("manifest", "once", "not attempted") == 1 and bare.complete is False
    attempt = need_attr("baselines.record", "StoreAttempt")(
        subject="manifest", target=("store", "main"), attempt_id=None, recorded="the manifest was read")
    closed = summarize([question], lines, attempts=[attempt])
    assert closed.count("manifest", "once", "cannot answer") == 1 and closed.complete is True


def test_the_text_says_complete_or_open_for_the_run_and_for_each_size_and_lists_the_open_questions():
    summarize, render = _fn("summarize"), _fn("render")
    questions, lines = _example()
    got = summarize(questions, lines)
    text = render(got)
    shown = [line.strip() for line in text.splitlines()]
    assert "Check 10: open" in shown and "Check 10: complete" not in shown
    assert "once: open" in shown and "small: complete" in shown
    assert "10k: complete" in shown and "100k: open" in shown
    for question in (questions[3], questions[4]):
        assert question.id in shown
    for question in questions[:3]:
        assert question.id not in shown
    assert "manifest once right: 1" in shown and "dolma 100k not attempted: 1" in shown
    assert "tally trace once failed attempt: 1" not in shown
    assert "tally manifest once right: 1" in shown
    done = render(summarize(questions, [_line(q, "right") for q in questions]))
    assert "Check 10: complete" in [line.strip() for line in done.splitlines()]


def test_the_text_holds_no_sentence_that_sets_a_tool_against_another(facts):
    summarize, render = _fn("summarize"), _fn("render")
    record = need("baselines.record")
    cannot = need_attr("baselines.interfaces", "CANNOT_ANSWER")
    questions = [_make_question(l, s, size) for l, s, size in facts.question_triples()]
    assert len(questions) == facts.questions_count == 2346
    lines = [_line(q, cannot if q.route is None else "right") for q in questions]
    keys = set()
    for question in questions:
        if question.route is None:
            for layout in record.layouts_of(question) or (None,):
                keys.add((question.subject, tuple(record.attempt_target(question)), layout))
    attempts = [record.StoreAttempt(subject=s, target=t, attempt_id=None, recorded="the attempt ran", layout=layout)
                for s, t, layout in sorted(keys, key=repr)]
    got = summarize(questions, lines, attempts=attempts)
    assert got.complete is True and tuple(got.open) == ()
    # The 2346 questions are all counted once, and the tally is those the table does not exclude.
    every = sum(got.count(s, label, word) for s in need_attr("baselines.interfaces", "SUBJECTS")
                for label in LABELS for word in WORDS)
    tallied = sum(got.tally_count(s, label, word) for s in need_attr("baselines.interfaces", "SUBJECTS")
                  for label in LABELS for word in WORDS)
    assert every == 2346 and tallied == 2346 - facts.excluded_count == 2199
    text = render(got)
    assert COMPARISON_WORDS.search(text) is None, COMPARISON_WORDS.search(text)
    # An open run, with failed lines and lines not attempted, reads the same way.
    partial = summarize(questions, lines[:200])
    assert partial.complete is False
    assert COMPARISON_WORDS.search(render(partial)) is None


def test_could_it_answer_is_counted_apart_from_right_among_those_answered():
    """The owner's tally answer of 18 Nov 2026 ("two counts, no repeats"): how many questions a subject could answer (a right or a
    wrong; a cannot answer, a failed attempt, an expected value not derived and a question not attempted are not answers) is
    counted apart from how many of those answered are right, and both are given for every subject and label, in the whole
    summary and in the tally. One subject with one right, one wrong and one cannot answer: answered 2, right among answered 1,
    and not "1 of 3"."""
    summarize, render = _fn("summarize"), _fn("render")
    right = _make_question("c05.unmutated.E/shard-2", "dolma")
    wrong = _make_question("c05.unmutated.E/shard-1", "dolma")
    cannot = _make_question("c05.unmutated.E/shard-3", "dolma")
    excluded = _make_question("c01.E.T", "trace")
    questions = [right, wrong, cannot, excluded]
    lines = [_line(right, "right"), _line(wrong, "wrong"), _line(cannot, "cannot answer"),
             _line(excluded, "right")]
    got = summarize(questions, lines)
    shown = [line.strip() for line in render(got).splitlines()]
    assert "dolma once answered: 2" in shown and "dolma once right among answered: 1" in shown
    assert "tally dolma once answered: 2" in shown and "tally dolma once right among answered: 1" in shown
    # The excluded question is counted and is not in the tally, in both counts.
    assert "trace once answered: 1" in shown and "tally trace once answered: 1" not in shown
    assert "tally trace once right among answered: 1" not in shown
    answered, right_of = _method(got, "answered"), _method(got, "right_among_answered")
    assert (answered("dolma", "once"), right_of("dolma", "once")) == (2, 1)
    assert (answered("trace", "once"), right_of("trace", "once")) == (1, 1)
    tally_answered, tally_right = _method(got, "tally_answered"), _method(got, "tally_right_among_answered")
    assert (tally_answered("trace", "once"), tally_right("trace", "once")) == (0, 0)
    assert (tally_answered("dolma", "once"), tally_right("dolma", "once")) == (2, 1)
    # A failed attempt, a question not attempted and a line not derived are never answers.
    failed = [_line(right, "no answer: failed attempt attempt-9"), _line(wrong, "wrong"), _line(cannot, "cannot answer")]
    again = summarize(questions, failed)
    assert (_method(again, "answered")("dolma", "once"), _method(again, "right_among_answered")("dolma", "once")) == (1, 0)
    assert "dolma once right among answered: 0" in [line.strip() for line in render(again).splitlines()]
    assert "dolma once answered: 1" in [line.strip() for line in render(again).splitlines()]
    # No answer at all: neither line is printed.
    none = [_line(q, "cannot answer") for q in (right, wrong, cannot)]
    assert not any("answered" in line for line in render(summarize(questions, none)).splitlines()
                   if line.startswith("dolma"))
