"""The interfaces of the baselines (plan 5.5 Check 10): the status words, the marks,
the six subjects and the types a question, a reading, a grade, a failed attempt and
a record are made of.

Nothing here imports the tool or any helper that builds the replica's world; the
module holds words and plain data only.

Every question gets exactly one status. Three of them are grades: ``right``,
``wrong`` and ``cannot answer``. The fourth is the status of a question whose
attempt failed, written ``no answer: failed attempt <id>``; it is no grade and is
never graded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

# --- The status words ---------------------------------------------------------

RIGHT = "right"
WRONG = "wrong"
CANNOT_ANSWER = "cannot answer"

# The three grades, in the order the table lists them.
GRADES = (RIGHT, WRONG, CANNOT_ANSWER)

# The status of a question whose attempt failed: this prefix and the attempt's id.
NO_ANSWER_PREFIX = "no answer: failed attempt "


def no_answer(attempt_id: str) -> str:
    """The status of a question whose attempt ``attempt_id`` failed."""
    return NO_ANSWER_PREFIX + attempt_id


# --- The marks ----------------------------------------------------------------

# A mark is carried by an answer that rests on something the subject made itself,
# or on a tie that was derived and not recorded. The strings are the table's.
MARK_OWN_OUTPUT = "tool's own output; not an independent baseline"
MARK_STEP_TO_ORDINAL = "step-to-ordinal part is the tool's own output"
MARK_TIE = "ordinal derived from src and loc (owner's ruling after the debate)"

MARKS = (MARK_OWN_OUTPUT, MARK_STEP_TO_ORDINAL, MARK_TIE)

# --- The sizes ----------------------------------------------------------------

# The set of 800 documents, and the sizes the plan names. A question asked once has
# the size None. The small set is kept apart from the plan sizes everywhere.
SIZE_SMALL = "small"
PLAN_SIZES = ("10k", "100k")

# --- The subjects -------------------------------------------------------------

# The six subjects, in the order of the table README's list.
SUBJECTS = (
    "manifest",
    "trace",
    "dolma",
    "originblame",
    "dolma+trace",
    "originblame+trace",
)


# --- The types ----------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    """One listed row put to one subject at one size.

    The unit is the listed row, the subject and the size at which the tool's own row
    is asked: a listed row asked once gives one question per subject, a sized one
    gives one per subject for each size.

    id: a string that is the same on every expansion and unique among questions,
        sizes included.
    group: the id of the group that lists the row.
    row_id: the id of the table row of that group and subject.
    listed_row: the id of the listed row of Checks 1 to 9.
    subject: one of ``SUBJECTS``.
    check: the number of the check the listed row belongs to.
    size: None for a listed row asked once, else ``small`` (the 800-document set)
        or a plan size (``10k``, ``100k``); the small set is never a plan size.
    excluded_from_tally: the table row's own flag; the question is recorded and
        left out of the tally.
    route: how the subject is read for this question, or None when the subject
        holds no record for it.
    missing: the table row's text naming the record the subject lacks, word for
        word, or None when the row has a route.
    needs_ordinal: the row ties a document to an ordinal.
    marks: the marks the table row carries, as a tuple.
    tie_mark: the mark of the src/loc tie that the row names, or None; it is kept
        apart from ``marks`` and is carried only by an answer that used the tie.
    """

    id: str
    group: str
    row_id: str
    listed_row: str
    subject: str
    check: int
    size: str | None
    excluded_from_tally: bool
    route: str | None
    missing: str | None
    needs_ordinal: bool
    marks: tuple[str, ...]
    tie_mark: str | None


@dataclass(frozen=True)
class Reading:
    """What a subject read for a question, before it is graded.

    question_id: the id of the question.
    raw: the answer as the subject gave it (any plain value).
    route: the route the reading followed, as it was named before the run.
    """

    question_id: str
    raw: Any
    route: str | None


@dataclass(frozen=True)
class Grade:
    """A grade and what it rests on.

    status: one of ``GRADES``.
    evidence: the text or value that shows why (for cannot answer, the missing
        field or the unmet condition).
    """

    status: str
    evidence: Any


@dataclass(frozen=True)
class Attempt:
    """One try at running a command for a question.

    id: the attempt's id.
    command: the command line as a tuple of words.
    exit_code: the exit code, or None when the command did not end by itself.
    output: what the command wrote.
    """

    id: str
    command: tuple[str, ...]
    exit_code: int | None
    output: str


@dataclass(frozen=True)
class Record:
    """The one record of a question.

    question_id: the id of the question.
    status: one of ``GRADES``, or ``no_answer(attempt_id)``.
    raw: the answer as the subject gave it, or None.
    evidence: what the status rests on, or None.
    route: the route that was followed, or None.
    marks: the marks the answer carries, as a tuple.
    excluded_from_tally: the question's flag.
    attempt_id: the id of the failed attempt when the status is a no answer,
        else None.
    """

    question_id: str
    status: str
    raw: Any
    evidence: Any
    route: str | None
    marks: tuple[str, ...]
    excluded_from_tally: bool
    attempt_id: str | None


@runtime_checkable
class Subject(Protocol):
    """Anything with a name and a way to read a question."""

    name: str

    def read(self, question: Question) -> Reading | Attempt:
        """Read one question; a reading is graded by someone else.

        The result is never None. A run that read the question gives a Reading.
        A run that failed gives the failed Attempt: that question is recorded
        with ``no_answer(attempt.id)`` and is never graded, and a gap is never
        taken for an absence.
        """
        ...
