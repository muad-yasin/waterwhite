"""The record of Check 10 (plan 5.5 Check 10; table README, sections "Marks and the
tally" and "A failed attempt, the two routes after it, and when Check 10 is
complete").

One record line per question, a question being a listed row, a subject and the size
at which the tool's own row is asked. Every question has exactly one status:

- a grade (right, wrong, cannot answer), from a reading that was graded;
- ``no answer: failed attempt <id>``, from a failed attempt (never graded);
- ``NOT_ATTEMPTED``, when no line exists for it.

Only the first two are written; the third is what the completeness check finds. A
size that was not run is therefore "not attempted" and leaves Check 10 open. The
small set (800 documents) is kept apart from the plan sizes (10k, 100k): a line of
one size never stands for a question of another, and the completeness check counts
each size by itself.

A line whose check10 row records "none" for its subject (a question with no route) is
cannot answer ONLY after the subject's attempt on the store or set its listed row
names is recorded as having run (``StoreAttempt``, ``none_line_status``). One attempt
covers every none line of its subject on that store or set; with no attempt the line
is not attempted, and with a failed attempt it takes the status of that attempt. An
attempt on the small set never covers a 10k or a 100k line. A listed row that names two
layouts needs an attempt per layout, each on the set that layout's question is asked of,
and the layouts are put together by ``baselines.expected.combine_layouts``: the status of
the first failed attempt, else not attempted when a layout has no attempt, else cannot
answer. Such a question is never right or wrong: ``completeness`` refuses a line of right
or wrong on a question whose record is none, with or without attempts, and a line of
``no answer: failed attempt <id>`` on such a question only when the attempts it is given
hold a failed attempt with that id, of that subject, on that store or set.

Every line names the commit it ran on. A line or an attempt that holds a local path, a
host name or a session name is refused when it is written, and no file is made. The
result of the tie show is written and read by ``baselines.tie`` alone;
``tie_shown_for`` reads it from there.
"""

from __future__ import annotations

import json
import re
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from baselines import table10, tie
from baselines.interfaces import (
    CANNOT_ANSWER,
    GRADES,
    NO_ANSWER_PREFIX,
    PLAN_SIZES,
    RIGHT,
    SIZE_SMALL,
    WRONG,
    Attempt,
    Grade,
    Question,
    Reading,
    Record,
    no_answer,
)

# The status of a question that has no line.
NOT_ATTEMPTED = "not attempted"

# A commit as the record names it: 7 to 40 lower case hex digits.
COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")

# The labels the completeness check counts by: a question asked once, the small set,
# and the plan sizes.
LABEL_ONCE = "once"
LABELS = (LABEL_ONCE, SIZE_SMALL, *PLAN_SIZES)

# What stays out of a record, written in pieces so that this file holds none of it.
PATH_MARKERS = ("/ho" + "me/", "/Us" + "ers/", "/ro" + "ot/", "Proj" + "ects/")
SESSION_MARKERS = ("C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor")
# A host name shorter than this is not searched for: it would match ordinary words.
SHORTEST_HOST_NAME = 4


class RecordError(ValueError):
    """A set of lines that cannot be one record of the questions."""


# --- One record of one question -------------------------------------------------


def record_of(
    question: Question,
    outcome: Reading | Attempt,
    grade: Grade | None = None,
    *,
    used_tie: bool = False,
) -> Record:
    """The record of ``question`` from what its subject gave.

    A ``Reading`` needs the ``Grade`` it was given; the record holds the grade's
    status and evidence, the reading's raw and route, the question's marks and its
    ``excluded_from_tally``. A failed ``Attempt`` takes no grade and gives the status
    ``no_answer(attempt.id)``, with no raw and no evidence. The mark of the src/loc
    tie is added only when ``used_tie`` is true; an answer that did not use the tie
    does not carry it. ``ValueError`` for a grade on an attempt, a reading with no
    grade, a reading of another question, a grade that is not one of the three, and a
    tie used where the question has no tie mark.
    """
    if used_tie and question.tie_mark is None:
        raise ValueError(f"{question.id} has no tie mark: it cannot say it used the tie")
    if isinstance(outcome, Attempt):
        if grade is not None:
            raise ValueError(f"attempt {outcome.id} failed: a failed attempt takes no grade")
        if used_tie:
            raise ValueError(f"attempt {outcome.id} failed: it used no tie")
        return Record(
            question_id=question.id,
            status=no_answer(outcome.id),
            raw=None,
            evidence=None,
            route=None,
            marks=tuple(question.marks),
            excluded_from_tally=question.excluded_from_tally,
            attempt_id=outcome.id,
        )
    if not isinstance(outcome, Reading):
        raise ValueError(f"not a reading or an attempt: {type(outcome).__name__}")
    if outcome.question_id != question.id:
        raise ValueError(f"the reading is of {outcome.question_id}, not of {question.id}")
    if grade is None:
        raise ValueError(f"the reading of {question.id} needs a grade")
    if grade.status not in GRADES:
        raise ValueError(f"{grade.status!r} is not a grade")
    marks = tuple(question.marks)
    if used_tie and question.tie_mark not in marks:
        marks += (question.tie_mark,)
    return Record(
        question_id=question.id,
        status=grade.status,
        raw=outcome.raw,
        evidence=grade.evidence,
        route=outcome.route,
        marks=marks,
        excluded_from_tally=question.excluded_from_tally,
        attempt_id=None,
    )


# --- The lines that are written --------------------------------------------------


def _plain(value: Any) -> Any:
    """A value ready for JSON: mappings as dicts, tuples as lists."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _checked_commit(commit: Any) -> str:
    if not isinstance(commit, str) or COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError(f"not a commit (7 to 40 lower case hex digits): {commit!r}")
    return commit


def _checked_step_ids(step_ids: Any) -> list[str]:
    """The ids of the steps a line rests on, as a list of text. ``ValueError`` for a text
    given whole, and for an entry that is not text."""
    if isinstance(step_ids, (str, bytes)) or not isinstance(step_ids, (list, tuple)):
        raise ValueError(f"the step ids are a list of text, not {type(step_ids).__name__}")
    for found in step_ids:
        if not isinstance(found, str):
            raise ValueError(f"a step id is text, got {found!r}")
    return list(step_ids)


def answer_line(question: Question, record: Record, *, commit: str, step_ids: Sequence[str] = ()) -> dict:
    """The line written for ``question``: JSON-ready, with its size and the commit it
    ran on, and the ids of the steps it rests on (``step_ids``, in the order given: the
    runner keeps the command, exit and output of each). ``ValueError`` for another
    question's record, a commit that is not 7 to 40 lower case hex digits, and step ids
    that are not a list of text."""
    if record.question_id != question.id:
        raise ValueError(f"the record is of {record.question_id}, not of {question.id}")
    return {
        "id": question.id,
        "group": question.group,
        "listed_row": question.listed_row,
        "subject": question.subject,
        "size": question.size,
        "status": record.status,
        "raw": _plain(record.raw),
        "evidence": _plain(record.evidence),
        "route": record.route,
        "marks": list(record.marks),
        "excluded_from_tally": record.excluded_from_tally,
        "attempt_id": record.attempt_id,
        "commit": _checked_commit(commit),
        "step_ids": _checked_step_ids(step_ids),
    }


def attempt_line(attempt: Attempt, question_ids: Sequence[str], *, commit: str,
                 step_ids: Sequence[str] = ()) -> dict:
    """The line written for a failed attempt: its command, exit code and output, the ids
    of the questions it was for and the ids of the steps it rests on. It holds no status
    and no grade."""
    return {
        "id": attempt.id,
        "command": list(attempt.command),
        "exit_code": attempt.exit_code,
        "output": attempt.output,
        "question_ids": list(question_ids),
        "commit": _checked_commit(commit),
        "step_ids": _checked_step_ids(step_ids),
    }


def _private_markers() -> tuple[str, ...]:
    markers = [*PATH_MARKERS, *SESSION_MARKERS]
    host = socket.gethostname().strip()
    if len(host) >= SHORTEST_HOST_NAME and host.lower() != "localhost":
        markers.append(host)
    return tuple(markers)


def refuse_private(value: Any, where: str) -> None:
    """``ValueError`` when anything inside ``value`` holds a local path, a session
    name or the host name of this machine. The one refusal of every writer of a result
    (``baselines.tie.write_result`` calls it too)."""
    text = json.dumps(_plain(value), ensure_ascii=False)
    for marker in _private_markers():
        if marker in text:
            raise ValueError(f"{where} holds {marker!r}: a record keeps no local path, host or session name")


def _write_lines(path: str | Path, lines: Sequence[Mapping[str, Any]], what: str) -> None:
    """Every line is checked before the file is opened, so a refusal makes no file."""
    for number, line in enumerate(lines, 1):
        if not isinstance(line, Mapping):
            raise ValueError(f"{what} {number} is not a mapping")
        refuse_private(line, f"{what} {number}")
    text = "".join(json.dumps(_plain(line), ensure_ascii=True) + "\n" for line in lines)
    Path(path).write_text(text, encoding="utf-8")


def _read_lines(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_answers(path: str | Path, lines: Sequence[Mapping[str, Any]]) -> None:
    """One JSON object per line. A line with a local path, host name or session name
    is refused with ``ValueError`` and no file is made."""
    _write_lines(path, lines, "answer line")


def read_answers(path: str | Path) -> list[dict]:
    return _read_lines(path)


def write_attempts(path: str | Path, lines: Sequence[Mapping[str, Any]]) -> None:
    """The failed attempts, one JSON object per line, refused as the answers are."""
    _write_lines(path, lines, "attempt line")


def read_attempts(path: str | Path) -> list[dict]:
    return _read_lines(path)


# --- The result of the tie show ----------------------------------------------------


def tie_shown_for(path: str | Path, chunk_file: str) -> bool:
    """Whether the tie show of the file at ``path`` holds ``chunk_file`` as shown. A chunk
    file the file does not hold, and a file that does not exist, read "not shown": the
    tie is never assumed. The result is read by ``baselines.tie`` and by nothing here."""
    return tie.is_shown(tie.read_result(Path(path)), chunk_file)


# --- A none line and the attempt that covers it --------------------------------------

# What the documented refusal of a trace export says (docs/cli.md, the export section).
REFUSAL_WORDS = "request refused"
REFUSAL_EXIT_CODE = 2


@dataclass(frozen=True)
class StoreAttempt:
    """The one attempt of a subject on a store or a set, in one layout where the listed
    row names two.

    subject: one of the six subjects.
    target: ``("store", <name>)`` or ``("set", <size>)``, as ``attempt_target`` gives it.
    attempt_id: None when the attempt ran; the id of the failed attempt when it failed.
    recorded: what the attempt recorded (a reading, the documented refusal with its exit
        code, a manifest reported absent).
    layout: the layout the attempt was made in; None for a row that names no layout.
    run: 1 for the first run of the subject on the store, 2 for the second run on it (a
        store whose writer was stopped and then run again). Two attempts are the same
        attempt only when subject, target, layout and run are equal; the two runs of one
        store are both kept, and ``none_line_status`` reads, for each row, the run it asks
        about.
    """

    subject: str
    target: tuple[str, str]
    attempt_id: str | None
    recorded: Any
    layout: str | None = None
    run: int = 1


# The one listed row that asks about the second run of a store; every other row reads the
# first run.
SECOND_RUN_ROW = "c04.retry"
FIRST_RUN = 1
SECOND_RUN = 2


def _run_read(listed_row: str, runs_held: set[int]) -> int:
    """The run a row reads: the second run for the row that asks about it where the store
    has one, else the first."""
    if listed_row == SECOND_RUN_ROW and SECOND_RUN in runs_held:
        return SECOND_RUN
    return FIRST_RUN


@lru_cache(maxsize=1)
def _listed_givens() -> Mapping[str, Mapping[str, Any]]:
    """The ``given`` of every core row of Checks 1 to 9, by id."""
    table = table10._acceptance_table()
    found = {}
    for number in table10.LISTED_CHECKS:
        for row in table.load_check(number, table10.DEFAULT_PATH.parent):
            # Every row of Checks 1 to 9 that a group lists: the core rows and, since 18 Nov 2026, the extension rows.
            found[row.id] = row.given
    return found


def set_store_of(question: Question) -> str | None:
    """The store a row names beside a set (``given``: ``set`` and ``store``), or None: a generated set made in the way of
    that store (the rows of Check 6 name the set and the store ``no-capture``) is a build of its own."""
    given = _listed_givens()[question.listed_row]
    return given.get("store") if "set" in given else None


def attempt_target(question: Question) -> tuple[str, str]:
    """The store or set a question is asked on, read from its listed row's ``given``
    (table README, section 2): a ``set`` is the set of the question's size (a row asked
    once names its own; a set the row names with a store is the build made in the way of
    that store: ``<store>-<size>``, for example ``no-capture-small``), otherwise the
    ``store``, left out meaning ``main``. A row that names a run, a segment or a checkpoint
    is the store that holds it. The name is taken from the row's ``given`` and is no folder
    of a size (``set-<size>``)."""
    given = _listed_givens()[question.listed_row]
    if "set" in given:
        name = question.size if question.size is not None else given["set"]
        if given.get("store") is not None:
            return ("set", f"{given['store']}-{name}")
        return ("set", name)
    return ("store", given.get("store", "main"))


def export_refusal_documented(exit_code: int | None, output: str) -> bool:
    """True for the one refusal docs/cli.md documents for a trace export that names no
    registered run segment (exit 2, "request refused"): the subject's own answer that
    the store holds no run. Any other exit or output is not documented for that case,
    which makes the attempt a failed one."""
    return (
        isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code == REFUSAL_EXIT_CODE
        and isinstance(output, str)
        and REFUSAL_WORDS in output
    )


def layouts_of(question: Question) -> tuple[str, ...] | None:
    """The layouts the listed row of ``question`` names in its ``given``, in the row's
    order, or None for a row that names none."""
    layouts = _listed_givens()[question.listed_row].get("layouts")
    return tuple(layouts) if layouts else None


def layout_of(question: Question) -> str | None:
    """The ONE layout the listed row of ``question`` names in its ``given`` (``layout``), or None for a row that names none
    or names two (``layouts_of``)."""
    return _listed_givens()[question.listed_row].get("layout")


def none_line_status(question: Question, attempts: Sequence[StoreAttempt]) -> str:
    """The status of a question that holds no record for its subject.

    ``CANNOT_ANSWER`` ONLY after the attempt of the question's subject on the question's
    store or set ran; ``no_answer(attempt_id)`` when that attempt failed; ``NOT_ATTEMPTED``
    with no attempt. One attempt covers every none line of its subject on its store or
    set. A listed row that names two layouts needs an attempt per layout (the attempt
    carries its ``layout``; one that names none, or another layout, covers neither):
    ``baselines.expected.combine_layouts`` puts the layouts together, the status of the
    first failed attempt in the row's order, else ``NOT_ATTEMPTED`` when a layout has no
    attempt, else ``CANNOT_ANSWER``. A store that was run twice holds both attempts: the
    row ``c04.retry`` reads the second run where there is one, every other row reads the
    first. ``ValueError`` for a question that has a route, and for two attempts of one
    subject on one store or set in one layout and one run.
    """
    if question.route is not None:
        raise ValueError(f"{question.id} holds a record: it is no none line")
    seen: dict[tuple, StoreAttempt] = {}
    target = attempt_target(question)
    runs_held: set[int] = set()
    for attempt in attempts:
        key = (attempt.subject, tuple(attempt.target), attempt.layout, attempt.run)
        if key in seen:
            where = f" in the {key[2]} layout" if key[2] is not None else ""
            raise ValueError(f"two attempts of {key[0]} on {key[1]}{where}, run {key[3]}: only one can stand")
        seen[key] = attempt
        if attempt.subject == question.subject and tuple(attempt.target) == target:
            runs_held.add(attempt.run)
    layouts = layouts_of(question)
    run = _run_read(question.listed_row, runs_held)

    def state(layout: str | None) -> str | Grade:
        covering = seen.get((question.subject, target, layout, run))
        if covering is None:
            return NOT_ATTEMPTED
        if covering.attempt_id is not None:
            return no_answer(covering.attempt_id)
        return Grade(status=CANNOT_ANSWER, evidence=None)

    if layouts is None:
        found = state(None)
        return found if isinstance(found, str) else found.status
    # Imported here: baselines.expected imports this module.
    from baselines import expected

    return expected.combine_layouts(question.listed_row, {name: state(name) for name in layouts}).status


# --- Completeness ----------------------------------------------------------------


@dataclass(frozen=True)
class SizeResult:
    """The questions of one label: how many there are, how many hold a grade, which
    are open (no grade: not attempted or a failed attempt) and whether none is."""

    total: int
    graded: int
    open: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class Completeness:
    """Where Check 10 stands for a list of questions.

    statuses: the one status of every question, ``NOT_ATTEMPTED`` where no line exists.
    open: the ids, in question order, whose status is no grade.
    by_size: a result for each of ``once``, ``small``, ``10k`` and ``100k``, each
        counted by itself.
    complete: nothing is open.
    plan_sizes_complete: neither plan size has anything open; a small line never
        closes a plan size. A size with no question in the list has nothing open, so
        the caller gives the whole list of questions.
    """

    statuses: Mapping[str, str]
    open: tuple[str, ...]
    by_size: Mapping[str, SizeResult]
    complete: bool
    plan_sizes_complete: bool


def _label_of(question: Question) -> str:
    if question.size is None:
        return LABEL_ONCE
    if question.size == SIZE_SMALL or question.size in PLAN_SIZES:
        return question.size
    raise RecordError(f"{question.id} asks at the size {question.size!r}, which is none of {LABELS}")


def _is_status(status: Any) -> bool:
    if status in GRADES:
        return True
    return isinstance(status, str) and status.startswith(NO_ANSWER_PREFIX) and len(status) > len(NO_ANSWER_PREFIX)


def _failed_attempt_stands(question: Question, status: str, attempts: Sequence[StoreAttempt]) -> bool:
    """Whether ``status`` is the status of a failed attempt the run made for ``question``:
    an attempt with that id, of the question's subject, on its store or set (in any layout
    of a row that names two), in the run the question's row reads."""
    if not status.startswith(NO_ANSWER_PREFIX):
        return False
    attempt_id = status[len(NO_ANSWER_PREFIX):]
    target = attempt_target(question)
    of_the_store = [a for a in attempts if a.subject == question.subject and tuple(a.target) == target]
    run = _run_read(question.listed_row, {a.run for a in of_the_store})
    return any(a.attempt_id == attempt_id and a.run == run for a in of_the_store)


def completeness(
    questions: Sequence[Question],
    lines: Sequence[Mapping[str, Any]],
    attempts: Sequence[StoreAttempt] = (),
) -> Completeness:
    """Every question gets exactly one status. ``lines`` need only hold ``id``, ``size``
    and ``status``. ``RecordError`` for a question listed twice, two lines of one
    question, a line of no question, a line whose size is not its question's, and a
    status that is neither a grade nor the status of a failed attempt, a line of right
    or wrong for a question that holds no record (such a question can only be cannot
    answer, so the refusal stands whatever attempts are given), and a line of
    ``no answer: failed attempt <id>`` for a question that holds no record unless
    ``attempts`` hold a failed attempt with that id, of that subject, on that store or set.

    A line of cannot answer for a question that holds no record (a none line) stands only
    with the attempt of its subject on its store or set (one per layout for a row that
    names two): its status is that of ``none_line_status`` (``NOT_ATTEMPTED`` with no
    attempt, the status of the attempt when it failed). A line of cannot answer for a
    question that holds a record is a grade of its own: the route could not be walked."""
    by_id: dict[str, Question] = {}
    for question in questions:
        if question.id in by_id:
            raise RecordError(f"{question.id} is listed twice")
        by_id[question.id] = question
    held: dict[str, str] = {}
    for line in lines:
        name = line.get("id")
        if name not in by_id:
            raise RecordError(f"a line of no question: {name!r}")
        if name in held:
            raise RecordError(f"two lines of {name}")
        if "size" not in line or line["size"] != by_id[name].size:
            raise RecordError(f"the line of {name} says the size {line.get('size')!r}, not {by_id[name].size!r}")
        if not _is_status(line.get("status")):
            raise RecordError(f"the line of {name} has the status {line.get('status')!r}, which is none a question can hold")
        if line["status"] in (RIGHT, WRONG) and by_id[name].route is None:
            raise RecordError(
                f"the line of {name} is {line['status']}, but its record is none: "
                "such a question can only be cannot answer"
            )
        if (
            by_id[name].route is None
            and line["status"] not in GRADES
            and not _failed_attempt_stands(by_id[name], line["status"], attempts)
        ):
            raise RecordError(
                f"the line of {name} says {line['status']!r}, but the attempts hold no failed "
                f"attempt with that id of {by_id[name].subject} on {attempt_target(by_id[name])}"
            )
        held[name] = line["status"]
    for name, status in held.items():
        if status == CANNOT_ANSWER and by_id[name].route is None:
            held[name] = none_line_status(by_id[name], attempts)
    statuses ={question.id: held.get(question.id, NOT_ATTEMPTED) for question in questions}
    open_ids = tuple(name for name, status in statuses.items() if status not in GRADES)
    by_size = {}
    for label in LABELS:
        ids = [q.id for q in questions if _label_of(q) == label]
        still_open = tuple(name for name in ids if statuses[name] not in GRADES)
        by_size[label] = SizeResult(
            total=len(ids),
            graded=len(ids) - len(still_open),
            open=still_open,
            complete=not still_open,
        )
    plan_complete = all(by_size[label].complete for label in PLAN_SIZES)
    return Completeness(
        statuses=statuses,
        open=open_ids,
        by_size=by_size,
        complete=not open_ids,
        plan_sizes_complete=plan_complete,
    )
