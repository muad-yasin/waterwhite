"""The summary of a run of Check 10 (plan 5.5 Check 10; table README, sections "Marks and the
tally" and "A failed attempt, the two routes after it, and when Check 10 is complete").

The questions are counted by subject, by size label and by status word. The labels are
``once`` (a question asked once), ``small`` (the set of 800 documents) and the two plan
sizes, ``10k`` and ``100k``; the small set is counted and judged by itself and never stands
for a plan size. The tally leaves out the questions the table excludes from it (they are
counted and recorded). The summary says whether Check 10 is complete or open, for the whole
run and for each label, and lists the questions that are open, in the order the questions
were given.

The one status of every question is the one ``baselines.record.completeness`` gives: a
grade, the status of a failed attempt, or not attempted where no line exists (a none line
counts as cannot answer only with the attempt of its subject on its store or set).

The text states counts and states of completeness and sets no tool against another.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from baselines import record
from baselines.interfaces import (
    CANNOT_ANSWER,
    NO_ANSWER_PREFIX,
    RIGHT,
    SUBJECTS,
    WRONG,
    Question,
)

FAILED_ATTEMPT = "failed attempt"

# A line the grader could not grade: the runner keeps the line of a question whose expected value is not derived
# (or that is not gradable) with the status "cannot answer" and these words at the start of its evidence. It is
# counted under its own word and stays open: it is never the subject's cannot answer.
NOT_DERIVED = "expected value not derived"
NOT_DERIVED_EVIDENCE = (NOT_DERIVED, "not gradable")

# The words a status is counted under, in the order the text lists them.
WORDS = (RIGHT, WRONG, CANNOT_ANSWER, FAILED_ATTEMPT, NOT_DERIVED, record.NOT_ATTEMPTED)

LABELS = record.LABELS

TALLY_PREFIX = "tally "

# The two counts of the tally (the owner's answer of 18 Nov 2026, "two counts, no repeats"): how many questions a subject could
# answer, and how many of those it answered are right. They stand apart: the second is never read as a share of all the questions.
# A question is answered when its status is right or wrong; cannot answer, a failed attempt, a line not derived and a question not
# attempted are no answer. A subject that answered nothing at a label has neither line.
ANSWERED = "answered"
RIGHT_AMONG_ANSWERED = "right among answered"


def _label(question: Question) -> str:
    """The label a question is counted under: ``once`` for a question asked once, else its
    size."""
    return record.LABEL_ONCE if question.size is None else question.size


def _word_of(status: str) -> str:
    """The counting word of a status: a failed attempt's status is counted as one word."""
    if status.startswith(NO_ANSWER_PREFIX):
        return FAILED_ATTEMPT
    return status


@dataclass(frozen=True)
class Summary:
    """What a run holds.

    counts: how many questions each (subject, label, word) holds.
    tallied: the same, without the questions the table excludes from the tally.
    complete: no question is open.
    open: the ids of the questions that hold no grade, in question order.
    plan_sizes_complete: neither plan size has a question open.
    size_complete: for each of ``once``, ``small``, ``10k`` and ``100k``, whether nothing
        of that label is open (a label with no question has nothing open).
    """

    counts: Mapping[tuple[str, str, str], int]
    tallied: Mapping[tuple[str, str, str], int]
    complete: bool
    open: tuple[str, ...]
    plan_sizes_complete: bool
    size_complete: Mapping[str, bool]

    def count(self, subject: str, label: str, word: str) -> int:
        """The questions of ``subject`` at ``label`` whose status is ``word``."""
        return self.counts.get((subject, label, word), 0)

    def tally_count(self, subject: str, label: str, word: str) -> int:
        """The same, in the tally: the questions the table excludes are not counted."""
        return self.tallied.get((subject, label, word), 0)

    def answered(self, subject: str, label: str) -> int:
        """The questions of ``subject`` at ``label`` it could answer: a right or a wrong."""
        return self.count(subject, label, RIGHT) + self.count(subject, label, WRONG)

    def right_among_answered(self, subject: str, label: str) -> int:
        """Of the questions ``subject`` answered at ``label``, the ones that are right."""
        return self.count(subject, label, RIGHT)

    def tally_answered(self, subject: str, label: str) -> int:
        """``answered``, in the tally."""
        return self.tally_count(subject, label, RIGHT) + self.tally_count(subject, label, WRONG)

    def tally_right_among_answered(self, subject: str, label: str) -> int:
        """``right_among_answered``, in the tally."""
        return self.tally_count(subject, label, RIGHT)


def summarize(
    questions: Sequence[Question],
    lines: Sequence[Mapping[str, Any]],
    attempts: Sequence[record.StoreAttempt] = (),
) -> Summary:
    """The summary of ``questions`` given the answer ``lines`` (each needs ``id``, ``size``
    and ``status``) and the attempts of the subjects on their stores and sets.
    ``baselines.record.RecordError`` where the lines are not one record of the questions."""
    derived = {line["id"] for line in lines
               if line.get("status") == CANNOT_ANSWER and isinstance(line.get("evidence"), str)
               and line["evidence"].startswith(NOT_DERIVED_EVIDENCE)}
    done = record.completeness(questions, [line for line in lines if line["id"] not in derived], attempts)
    counts: Counter = Counter()
    tallied: Counter = Counter()
    for question in questions:
        word = NOT_DERIVED if question.id in derived else _word_of(done.statuses[question.id])
        key = (question.subject, _label(question), word)
        counts[key] += 1
        if not question.excluded_from_tally:
            tallied[key] += 1
    return Summary(
        counts=dict(counts),
        tallied=dict(tallied),
        complete=done.complete,
        open=done.open,
        plan_sizes_complete=done.plan_sizes_complete,
        size_complete={label: done.by_size[label].complete for label in LABELS},
    )


def _state(flag: bool) -> str:
    return "complete" if flag else "open"


def render(summary: Summary) -> str:
    """The text of a summary: ``Check 10: complete`` or ``Check 10: open``; for each label
    ``<label>: complete`` or ``<label>: open``; for each count above zero
    ``<subject> <label> <word>: <count>``, and the same behind ``tally`` for the tally; for each
    subject and label that answered anything, the two counts ``<subject> <label> answered: <n>`` and
    ``<subject> <label> right among answered: <n>`` (and behind ``tally``); then the id of each open question on a
    line of its own."""
    out = [f"Check 10: {_state(summary.complete)}"]
    for label in LABELS:
        out.append(f"{label}: {_state(summary.size_complete[label])}")
    out.append(f"plan sizes: {_state(summary.plan_sizes_complete)}")
    for prefix, table in (("", summary.counts), (TALLY_PREFIX, summary.tallied)):
        for subject in SUBJECTS:
            for label in LABELS:
                for word in WORDS:
                    found = table.get((subject, label, word), 0)
                    if found:
                        out.append(f"{prefix}{subject} {label} {word}: {found}")
                answered = summary.tally_answered(subject, label) if prefix else summary.answered(subject, label)
                if answered:
                    right = (summary.tally_right_among_answered(subject, label) if prefix
                             else summary.right_among_answered(subject, label))
                    out.append(f"{prefix}{subject} {label} {ANSWERED}: {answered}")
                    out.append(f"{prefix}{subject} {label} {RIGHT_AMONG_ANSWERED}: {right}")
    out.extend(summary.open)
    return "\n".join(out) + "\n"
