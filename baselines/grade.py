"""The grader of Check 10 (plan 5.5 Check 10; table README, section "Grades, and what
is compared").

``grade(question, expect, outcome, *, attempts=())`` gives one of three grades:

- right: the reading gives the expected answer of the row asked;
- wrong: it gives another answer, or none (a gap is never read as an absence);
- cannot answer: the subject ran and its own records lack the record or field the
  question needs; the evidence names it.

A failed attempt is not graded. Nor is a subject that did not run, a reading of
another question, a row that does not apply under the fallback in force, or an
expectation that still holds a pending value the spec did not fill: each raises
``NotGradable``, so that no such question can earn a grade by accident. The attempt
states ("not attempted" and "no answer: failed attempt <id>") are line statuses of
``baselines.record``, never a grade word; ``grade_layouts`` hands them to
``baselines.expected.combine_layouts``, which returns the line.

A reading whose ``raw`` holds a ``cannot_answer`` value that is not empty is cannot answer,
whatever else it holds; an empty or None value says nothing and is read as no key. On a row
that allows a set of outcomes (``one_of``) none of which is "cannot answer", a
``cannot_answer`` key is no outcome of the row: the reading is wrong, evidence ``["records"]``.

A question whose row records "none" for its subject (``question.route`` is None) is
cannot answer ONLY after the subject's attempt on the store or set its listed row names
is recorded as having run (``attempts``, ``baselines.record.StoreAttempt``; one per
layout for a row that names two); its evidence is the question's ``missing`` text word
for word. With no such attempt it is not attempted, and a failed attempt gives its own
status: neither is graded (``NotGradable``). The same holds on any route: where the rule
of every subject's row of the group is "cannot answer" (the table's own rows say so), no
value is compared whatever route the question carries or the reading holds, and the
question is graded as one that holds no record.

What is compared is the value each field of the listed row names: the values the rule of
the check10 row names, every one of them, and no field the rule does not name:

- a row whose expectation holds an ``answer``: the answer alone (the rows of the
  combinations say so, and the first exposure, the untraced steps and the candidates
  are left out with it);
- a row that expects a stop (it holds ``error``): the reading must be a stop, and
  nothing about the error is compared; an answer read where a stop is expected is wrong,
  and the field that differs is the ``answer``;
- a row that expects ``rows``: the reading's rows must equal the oracle's lines row
  for row, in order, on (segment, step, sample position, shard position, ordinal,
  token start, token end); where the row names the tokens covered, the rows of the
  covered part the spec states, on (shard position, ordinal, token start, token end).
  The rule of the position rows names the trace's rows, so the value fields beside them
  (``row``, ``rows_in_sample``, ``documents_in_shard`` and the like) are not read. A row
  that asks the values of a launch
  (``consumed_at_start``, ``first_step``, ``last_step``) is graded on those values; its
  rows are where they are read from. The ``diff`` such a row holds only says that the
  rows equal the oracle's and is never read from a reading;
- a row that allows a set of outcomes (``one_of``, ``never``: ``c04.kill-after-100`` and
  ``c04.retry``): read from the counts of the subject's own records alone, as
  ``grade_from_records`` reads them. ``grade`` reads no bare outcome; a reading without
  the subject's records is wrong, evidence ``["records"]``;
- any other row: every field it holds is compared for equality, except the fields the
  rule of its group leaves out (``NOT_READ``); the candidates and the untraced steps as
  sets, and the two lists of the mixtures each as a set. The listing of a manifest after a fault
  (``c05.listing.map-deleted``) is compared by membership, with no order and no reason: every artifact the
  row covers is listed and each artifact it names is listed as no record. The file family of an import
  (``c06.import.file-family``) is compared by the files its list names against the file of the member the listed row gives. Of a split pair only the chunk
  file and the line of each ordinal are read: neither the name of the pair nor the id of a
  member. A field the reading lacks is wrong.
- the rows whose rule names one value of the listed row's expectation
  (``c09.10k.partitioned.kept-shards``: the list of shards read keep;
  ``c09.10k.round-robin.all-rewrite``: the number of chunks read rewrite;
  ``c01.E.step8.counted``: the number of samples a step counts): that value alone is compared,
  from the derived value where the counting document gives one (``ONLY_READ``).

Things a row holds are not asked of the reading: the count of ``mismatches`` beside
``per_ordinal`` or ``per_shard`` and the ``diff`` of a whole run (the grader works both
out from what it compares), a ``label`` word beside the ``record`` the question also asks
for, the words of a ``link`` (its label and evidence; its member is compared through its file), and the fields ``NOT_READ`` names
(the byte identity afterwards, the records removed and the size of a successor of an
action row; the candidates, the derivation and the verification of a documents-mapped
row; the word that says how a map was built, on the records rows). The rows that ask one id per ordinal are read by
``baselines.expected.one_id_reading``, so that an ordinal of a group of identical
documents is right for any member of the group.

Entry points beside ``grade``:

- ``grade_concrete``: the same for a ``baselines.expected.Concrete``, the expectation
  with each sentence-valued field replaced by its concrete value; one that still holds a
  field with no concrete value is not graded, so that no prose of the table is compared
  as a string. A field the Concrete names as not compared is left out of what is compared
  and raises nothing. The per-shard counts the Concrete holds for a documents-mapped row
  are compared beside the ids;
- ``grade_whole_run``: the whole-run diff rows, from the oracle's log; the diff a
  subject states is never taken;
- ``grade_from_records``: the two rows that allow a set of outcomes (``one_of``,
  ``never``), read from the subject's own records and from no outcome the subject names
  (``grade`` reads them through it);
- ``grade_layouts``: a line whose listed row names two layouts, one line for the row.

Not compared, because it is this tool's way of saying why: ``names``,
``names_detail``, ``failing``, ``chain``, ``scope``, ``reason``, ``error``,
``exit_code``, ``stdout_empty``, ``notes``, ``basis``, and the words that describe the
check (``compared``, ``rule``, ``judged_against``, ``starts_at``, ``spec_key``).

A wrong grade's evidence is the sorted list of the names of the fields that differ.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from baselines import expected as expectations
from baselines import record, table10
from baselines.interfaces import (
    CANNOT_ANSWER,
    RIGHT,
    WRONG,
    Attempt,
    Grade,
    Question,
    Reading,
    no_answer,
)

# The keys of a reading's ``raw``: a route that could not be walked (its value is the
# evidence), and a command of the subject that stopped with an error.
CANNOT_KEY = "cannot_answer"
STOPPED_KEY = "stopped"

# The key of a reading's ``raw`` that holds the counts of the subject's own records.
RECORDS_KEY = "records"

APPLIES_KEY = "applies"
PENDING = "pending:"

# The fields that are this tool's way of saying why, and the words that describe a check.
NOT_COMPARED = frozenset({
    "names", "names_detail", "failing", "chain", "scope", "reason", "error", "exit_code",
    "stdout_empty", "notes", "basis", "compared", "rule", "judged_against",
    "starts_at", "spec_key",
})

# The key a row that expects ``rows`` also holds: it says only that the rows equal the
# oracle's, so a reading's own is never read.
DIFF_KEY = "diff"

# The values of a launch of a run: a row that asks them holds the whole run's rows only
# as the place they are read from, and is graded on the values.
LAUNCH_VALUES = frozenset({"consumed_at_start", "first_step", "last_step"})

# The fields of the table that hold a sentence where a value stands, by listed row: the one
# source is ``baselines.expected``. A field that still holds the sentence is never compared
# as a string, whichever way the expectation reaches ``grade``;
# ``baselines.expected.concrete_values`` puts a value in its place (a list, a mapping or a
# number) or leaves it open. The pairs of ``baselines.expected.WORD_FIELDS`` hold a word of
# the table, not a sentence, and are compared as the word they are.
SENTENCE_FIELDS = expectations.SENTENCE_FIELDS

# The fields compared as sets.
SET_FIELDS = frozenset({"candidates", "untraced_steps"})

# The fields compared as sets on one listed row only (a list read in another order is the
# same list; a member added or left out is not).
SET_FIELDS_OF_ROW: Mapping[str, frozenset[str]] = {
    "c09.fixture.mixtures": frozenset({"needs_successor", "kept"}),
}

# The listing of a manifest after the fault ``f.map-deleted``: the right text of the manifest's row names membership
# ("E/shard-1 is listed as no record, and every run, mixture, shard, checkpoint and release of the listed row's covers
# is listed") and neither an order nor a reason, so the two fields are compared by membership (see ``_listing_differs``).
LISTING_ROW = "c05.listing.map-deleted"

# The file family of an import: the rule of the manifest's row reads "the reading is the files that list names" and
# its right text "the list names the file of the member the listed row gives, and no other file"; it names no member
# id; the member is compared through its file, which comes from the world's own request entries (``member_files``).
IMPORT_FILE_ROW = "c06.import.file-family"

# The one row of which only the chunk file and the line of each ordinal are read: the rule of
# the toolkit's row names, for each of the two ordinals of a split pair, the chunk file and the
# line the tied row names. Neither the name of a pair nor the id of a member is in it.
MEMBER_ID_NOT_READ_ROW = "c03.pairs.split"
MEMBER_KEYS_READ = ("shard", "line")

# The per-shard counts a documents-mapped row compares beside the ids, which the Concrete
# holds as counted values.
COUNTED_COMPARED: Mapping[str, tuple[str, ...]] = {
    "c06.generated.documents-mapped": (
        expectations.COUNT_DOCUMENTS,
        expectations.COUNT_ONE_TIED_ROW,
        expectations.COUNT_NO_TIED_ROW,
    ),
    # The number of chunks read rewrite is the one value this row compares: the derived number
    # stands in place of the table's own.
    "c09.10k.round-robin.all-rewrite": ("rewrite",),
}

# The rows of which the rule of the check10 table names one value of the listed row's
# expectation (the listed row holds more): only that value is compared.
#   c09.10k.partitioned.kept-shards: "the reading is the list of shards read keep".
#   c09.10k.round-robin.all-rewrite: "the number of chunks read rewrite".
#   c01.E.step8.counted: "the number of samples a step counts" (the loud stop the listed row
#     holds beside it is not named).
ONLY_READ: Mapping[str, tuple[str, ...]] = {
    "c09.10k.partitioned.kept-shards": ("keep",),
    "c09.10k.round-robin.all-rewrite": ("rewrite",),
    "c01.E.step8.counted": ("consumed_samples",),
}

# The rows whose answer is one id per ordinal (see ``baselines.expected.one_id_reading``):
# the one source is ``baselines.expected``.
ONE_ID_ROWS = expectations.ONE_ID_ROWS

# The fields the rule of a group of the check10 table leaves out, by the group of the
# question (a field the rule of the check10 row does not name is not compared).
#   c09.artifact-actions: "Byte identity afterwards, the records removed and the size of a
#     successor are not read".
#   c06.documents-mapped: "The label words and the derivation mode of the listed rows are not
#     compared". The reading the rule names is, per shard, the number of documents, the
#     number with exactly one tied row and the number with none: the number of candidates and
#     the verification are not in it.
#   c03.records and c06.documents-mapped: no rule of either group names the word that says how
#     the map was built (the word the fallback adds to the listed row).
NOT_READ: Mapping[str, frozenset[str]] = {
    "c09.artifact-actions": frozenset({
        "byte_identical_afterwards", "records_removed", "documents_in_successor",
    }),
    "c06.documents-mapped": frozenset({
        "candidates", "derivation", "verification", "map_built_by",
    }),
    "c03.records": frozenset({"map_built_by"}),
}

# The fields of one row of the join, in full and for a covered part.
ROW_FIELDS = ("segment", "step", "sample_pos", "shard_position", "ordinal",
              "token_start", "token_end")
COVERED_FIELDS = ("shard_position", "ordinal", "token_start", "token_end")
SAMPLE_FIELDS = ("segment", "step", "sample_pos")

# The two rows that allow a set of outcomes, the rows of a 500-document shard they read.
KILL_ROW = "c04.kill-after-100"
RETRY_ROW = "c04.retry"
PLAN_ROWS = (KILL_ROW, RETRY_ROW)
SHARD_ROWS = 500
# The subjects whose entries are counted and not matched to a file: the tool alone, and the
# tool beside the trace (its row reads as the row of the tool alone). A matches_shard their
# reader gives is not part of their rule and is not looked at.
COUNTS_ONLY_SUBJECTS = frozenset({"originblame", "originblame+trace"})


class NotGradable(ValueError):
    """The outcome is not something a grade can be put on."""


def _plain(value: Any) -> Any:
    """A value with its mappings as dicts and its sequences as lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _unfilled(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(PENDING)
    if isinstance(value, Mapping):
        return any(_unfilled(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_unfilled(item) for item in value)
    return False


def _canonical(items: Any) -> Any:
    """A list as a sorted list of the canonical text of its items, for a set compare;
    anything else as it is."""
    if isinstance(items, list):
        return sorted(json.dumps(item, sort_keys=True) for item in items)
    return items


def _as_set(items: Any) -> Any:
    """A list as the set of the canonical text of its items (a member listed twice is one
    member); anything else as it is."""
    if isinstance(items, list):
        return {json.dumps(item, sort_keys=True) for item in items}
    return items


# The reading "no record" (the manifest's word: the artifact has no row) equals the answer "not covered" (table README, "Readings": a
# subject that keeps records for this kind of artifact and holds none for this one). Every other subject writes "not covered".
# Added 18 Nov 2026: the pass-3 lines `c05.f.map-deleted.E/shard-1` and `c05.unregistered-checkpoint` of the manifest read "no record"
# and graded wrong against "not covered".
READING_OF_ANSWER = {"no record": "not covered"}


def _same(name: str, expected: Any, got: Any, listed_row: str | None = None) -> bool:
    expected, got = _plain(expected), _plain(got)
    if name == "answer":
        expected, got = READING_OF_ANSWER.get(expected, expected), READING_OF_ANSWER.get(got, got)
    if isinstance(expected, bool) or isinstance(got, bool):
        return isinstance(expected, bool) and isinstance(got, bool) and expected == got
    if name in SET_FIELDS:
        return _canonical(expected) == _canonical(got)
    if name in SET_FIELDS_OF_ROW.get(listed_row, ()):
        return _as_set(expected) == _as_set(got)
    return expected == got


def _artifact_name(item: Any) -> str:
    return str(item["artifact"]) if isinstance(item, Mapping) and "artifact" in item else str(item)


def _listing_differs(expect_fields: Mapping[str, Any], raw: Mapping[str, Any]) -> list[str]:
    """The fields of a listing that differ by membership: ``covers`` (every artifact of each kind the listed row
    covers is listed under that kind; more kinds and names are no difference) and ``not_covered`` (each artifact the
    row names is listed as no record and is not also listed with a kind). Order and reason are not compared."""
    differing: list[str] = []
    got_covers = raw.get("covers")
    listed_names: set[str] = set()
    if isinstance(got_covers, Mapping):
        for names in got_covers.values():
            if isinstance(names, (list, tuple)):
                listed_names |= {_artifact_name(item) for item in names}
    wanted_covers = expect_fields.get("covers")
    if isinstance(wanted_covers, Mapping):
        ok = isinstance(got_covers, Mapping) and all(
            {_artifact_name(n) for n in names} <= {_artifact_name(n) for n in (got_covers.get(kind) or ())}
            for kind, names in wanted_covers.items())
        if not ok:
            differing.append("covers")
    wanted_not = expect_fields.get("not_covered")
    if wanted_not is not None:
        wanted = {_artifact_name(item) for item in wanted_not}
        got_not = raw.get("not_covered")
        listed = {_artifact_name(item) for item in got_not} if isinstance(got_not, (list, tuple)) else set()
        if not (wanted <= listed and not (wanted & listed_names)):
            differing.append("not_covered")
    return differing


def _import_file_differs(expected_link: Any, raw: Mapping[str, Any],
                         member_files: Mapping[str, Sequence[str]] | None) -> bool:
    """True where the files the reading's list names are not exactly the files of the member of the listed row. The
    files of the member come from ``member_files`` (the world's own entries); a member with none is not graded."""
    member = expected_link.get("member") if isinstance(expected_link, Mapping) else None
    wanted = (member_files or {}).get(member) if isinstance(member, str) else None
    if not wanted:
        raise NotGradable(f"{IMPORT_FILE_ROW}: the file of the member {member!r} is not derived: the world names no "
                          "request of its source, so the expected value is not derived")
    got = raw.get("link")
    files = got.get("files") if isinstance(got, Mapping) else None
    if not isinstance(files, (list, tuple)):
        return True
    return set(map(str, files)) != set(map(str, wanted))


def _chunk_file_and_line(pairs: Any) -> Any:
    """The records of the pairs as what is read of them: for each member its chunk file and
    its line, and neither the name of the pair nor the id of a member."""
    if not isinstance(pairs, list):
        return pairs
    found = []
    for pair in pairs:
        if isinstance(pair, Mapping) and isinstance(pair.get("members"), (list, tuple)):
            pair = {"members": [
                {key: member[key] for key in MEMBER_KEYS_READ if key in member}
                if isinstance(member, Mapping) else member
                for member in pair["members"]
            ]}
        found.append(pair)
    return found


def _lines(rows: Mapping[str, Any]) -> list[dict]:
    return [json.loads(line) if isinstance(line, str) else dict(line)
            for line in rows["log_lines"]]


def _rows_match(rows: Any, expect: Mapping[str, Any], got: Any) -> bool:
    """The reading's rows against the oracle's lines of the expectation."""
    if "log_lines" not in rows:
        raise NotGradable("the expected rows hold no lines of the oracle's log")
    if not isinstance(got, list) or not all(isinstance(row, Mapping) for row in got):
        return False
    if "covered_tokens" in expect and "rows_of_the_covered_part" in rows:
        fields, stated = COVERED_FIELDS, list(rows["rows_of_the_covered_part"])
    else:
        fields, stated = ROW_FIELDS, _lines(rows)
    wanted = [tuple(row.get(name) for name in fields) for row in stated]
    found = [tuple(row.get(name) for name in fields) for row in got]
    return wanted == found


def _one_id_matches(listed_row: str, expected: Any, got: Any) -> bool:
    """One id per ordinal against the expectation that carries a group of identical
    documents as the list of its ids."""
    if not isinstance(expected, Mapping):
        raise NotGradable(
            f"{listed_row} holds the table's sentence for per_ordinal, not the ids: "
            "it is graded from its concrete values"
        )
    if not isinstance(got, Mapping):
        return False
    if not all(isinstance(ids, (list, tuple)) for ids in got.values()):
        return False
    return expectations.one_id_reading(listed_row, expected, got).right


def _checked_reading(question: Question, outcome: Reading | Attempt | None) -> Reading:
    """The reading itself, or ``NotGradable`` for what no grade can be put on."""
    if outcome is None:
        raise NotGradable("the subject did not run: no outcome to grade")
    if isinstance(outcome, Attempt):
        raise NotGradable(f"attempt {outcome.id} failed: a failed attempt is not graded")
    if not isinstance(outcome, Reading):
        raise NotGradable(f"not a reading: {type(outcome).__name__}")
    if outcome.question_id != question.id:
        raise NotGradable(f"the reading is of {outcome.question_id}, not of {question.id}")
    return outcome


@lru_cache(maxsize=1)
def _groups_without_a_record() -> frozenset[str]:
    """The groups of the table in which no subject has a route and the rule of every
    subject's row is "cannot answer": nothing of such a group is ever compared."""
    table = table10.load_table10()
    seen: dict[str, bool] = {}
    for row in table.rows:
        none = row.route is None and row.rule.startswith("cannot answer")
        seen[row.group] = seen.get(row.group, True) and none
    return frozenset(group for group, none in seen.items() if none)


def _holds_no_record(question: Question) -> bool:
    """Whether the question is graded as a line that holds no record: its row records
    "none" for its subject, or the rule of every subject's row of its group is "cannot
    answer", whatever route the question carries."""
    return question.route is None or question.group in _groups_without_a_record()


def _none_line(question: Question, attempts: Sequence[record.StoreAttempt]) -> Grade:
    """The grade of a question whose row records "none" for its subject: cannot answer
    after the subject's attempt on its store or set ran, else not graded."""
    if question.route is not None:
        question = dataclasses.replace(question, route=None)
    status = record.none_line_status(question, attempts)
    if status == CANNOT_ANSWER:
        return Grade(status=CANNOT_ANSWER, evidence=question.missing)
    if status == record.NOT_ATTEMPTED:
        raise NotGradable(
            f"{question.id}: no attempt of {question.subject} on "
            f"{record.attempt_target(question)} is recorded: not attempted"
        )
    raise NotGradable(f"{question.id}: {status}")


def _is_said(raw: Mapping[str, Any]) -> bool:
    """Whether the reading says that the route could not be walked: its ``cannot_answer``
    holds a value (an empty or None value says nothing and is no key)."""
    held = raw.get(CANNOT_KEY)
    if held is None:
        return False
    return not (isinstance(held, (str, Sequence, Mapping)) and len(held) == 0)


def _cannot_answer(
    question: Question, expect: Mapping[str, Any], raw: Mapping[str, Any]
) -> Grade | None:
    """The grade of a reading that holds a ``cannot_answer`` key, or None where it holds
    none that says something. A row that allows a set of outcomes none of which is "cannot
    answer" has no such outcome: a key there is a reading with no records, wrong, evidence
    ``["records"]``."""
    if CANNOT_KEY not in raw:
        return None
    if question.listed_row in PLAN_ROWS and CANNOT_ANSWER not in (expect.get("one_of") or ()):
        return Grade(status=WRONG, evidence=[RECORDS_KEY])
    if _is_said(raw):
        return Grade(status=CANNOT_ANSWER, evidence=raw[CANNOT_KEY])
    return None


def grade(
    question: Question,
    expect: Mapping[str, Any],
    outcome: Reading | Attempt | None,
    *,
    attempts: Sequence[record.StoreAttempt] = (),
    member_files: Mapping[str, Sequence[str]] | None = None,
) -> Grade:
    """The grade of one question from what its subject gave. ``member_files`` maps a member of the table to the
    files of its source, derived from the world's own request entries: the file family of an import is compared
    through the file of the member the listed row gives.

    ``expect`` is the expectation in force for the listed row (an ``Expected``'s
    ``expect``). ``outcome`` is a ``Reading``; a failed ``Attempt`` or None raises
    ``NotGradable``, as does a reading of another question, an expectation that says
    the row does not apply, an expectation with a pending value left unfilled, a row
    that is a whole-run diff (``grade_whole_run``), an expectation whose field still holds
    the table's sentence where a value stands (``SENTENCE_FIELDS``: a sentence is never
    compared as a string), and a question that holds no record with no attempt of its
    subject on its store or set (see the module words).
    """
    reading = _checked_reading(question, outcome)
    if not expect or expect.get(APPLIES_KEY) is False:
        raise NotGradable(f"{question.listed_row} does not apply under the fallback in force")
    if _holds_no_record(question):
        return _none_line(question, attempts)
    raw = {} if reading.raw is None else reading.raw
    if not isinstance(raw, Mapping):
        raise NotGradable("the reading's raw is not a mapping of fields")
    said = _cannot_answer(question, expect, raw)
    if said is not None:
        return said
    if question.listed_row in PLAN_ROWS:
        # The two rows that allow a set of outcomes are read from the subject's records
        # alone, by the one path of ``grade_from_records``: no bare outcome is read.
        return _plan_row_grade(question, raw)

    stop_expected = "error" in expect
    fields = {key: value for key, value in expect.items() if key not in NOT_COMPARED}
    for name in NOT_READ.get(question.group, ()):
        fields.pop(name, None)
    if question.listed_row in ONLY_READ:
        # The rule of the row names one value of the expectation: no other field is compared.
        wanted = ONLY_READ[question.listed_row]
        fields = {name: value for name, value in fields.items() if name in wanted}
    if DIFF_KEY in fields and "rows" not in fields and "answer" not in fields:
        raise NotGradable(
            f"{question.listed_row} asks the diff of a whole run: the grader computes it "
            "from the oracle's log, see grade_whole_run"
        )
    fields.pop(DIFF_KEY, None)
    if "answer" in fields:
        fields = {"answer": fields["answer"]}
    elif "rows" in fields:
        # The rule of a position row names the trace's rows: the value fields beside them
        # are not compared. A row that asks the values of a launch is graded on those.
        launch = {key: value for key, value in fields.items() if key in LAUNCH_VALUES}
        fields = launch or {"rows": fields["rows"]}
    if "per_ordinal" in fields or "per_shard" in fields:
        # The number of ordinals (or of shards) that differ is worked out from what is
        # compared.
        fields.pop("mismatches", None)
    if "record" in fields:
        # A label word beside the record the question also asks for is not compared.
        fields.pop("label", None)
    # A set of outcomes is no field of a row that is read from a reading's own words.
    for key in ("one_of", "never"):
        fields.pop(key, None)
    if _unfilled(fields):
        raise NotGradable(f"{question.listed_row} expects a value the spec did not fill")
    if not fields and not stop_expected:
        raise NotGradable(f"{question.listed_row} expects nothing that is compared")
    sentences = sorted(
        name for name in SENTENCE_FIELDS.get(question.listed_row, ())
        if isinstance(fields.get(name), str)
        and (question.listed_row, name) not in expectations.WORD_FIELDS
    )
    if sentences:
        raise NotGradable(
            f"{question.listed_row} holds the table's sentence for {', '.join(sentences)}, "
            "not a value: it is not compared as a string (see grade_concrete)"
        )

    differing: list[str] = []
    if question.listed_row == LISTING_ROW:
        differing.extend(_listing_differs(fields, raw))
        fields = {key: value for key, value in fields.items() if key not in ("covers", "not_covered")}
    elif question.listed_row == IMPORT_FILE_ROW and "link" in fields:
        if _import_file_differs(fields["link"], raw, member_files):
            differing.append("link")
        fields = {key: value for key, value in fields.items() if key != "link"}
    stopped = bool(raw.get(STOPPED_KEY))
    if stop_expected and not stopped:
        # A reading that is an answer word where a stop is expected differs in its answer.
        differing.append("answer" if "answer" in raw else "error")
    if stopped and not stop_expected:
        differing.append(STOPPED_KEY)
    for name, value in fields.items():
        if name not in raw:
            differing.append(name)
        elif name == "rows" and isinstance(value, Mapping):
            if not _rows_match(value, expect, raw[name]):
                differing.append(name)
        elif name == "per_ordinal" and question.listed_row in ONE_ID_ROWS:
            if not _one_id_matches(question.listed_row, value, raw[name]):
                differing.append(name)
        elif name == "link" and isinstance(value, Mapping) and "member" in value:
            # The label word and the evidence sentence of the link are not compared.
            got = raw[name]
            if not isinstance(got, Mapping) or got.get("member") != value["member"]:
                differing.append(name)
        elif name == "record" and question.listed_row == MEMBER_ID_NOT_READ_ROW:
            if _chunk_file_and_line(_plain(value)) != _chunk_file_and_line(_plain(raw[name])):
                differing.append(name)
        elif not _same(name, value, raw[name], question.listed_row):
            differing.append(name)
    if differing:
        return Grade(status=WRONG, evidence=sorted(set(differing)))
    return Grade(status=RIGHT, evidence=None)


# --- A Concrete expectation -------------------------------------------------------


def grade_concrete(
    question: Question,
    concrete: expectations.Concrete,
    outcome: Reading | Attempt | None,
    *,
    attempts: Sequence[record.StoreAttempt] = (),
    member_files: Mapping[str, Sequence[str]] | None = None,
) -> Grade:
    """``grade`` for the expectation of a ``Concrete``. A question that holds a record
    is not graded while ``concrete.open`` names a field with no concrete value: its
    table text is a sentence, and a sentence is never compared as a string
    (``NotGradable`` naming the field). A question that holds no record needs no expected
    value."""
    _checked_reading(question, outcome)
    if concrete.listed_row != question.listed_row:
        raise ValueError(f"the expectation is of {concrete.listed_row}, not of {question.listed_row}")
    if not _holds_no_record(question) and concrete.open:
        raise NotGradable(
            f"{question.listed_row} has no concrete value for {', '.join(concrete.open)}: "
            "it is not compared as a string"
        )
    # A field the Concrete names as not compared is left out; the per-shard counts it holds
    # for the row are compared beside the table's own fields.
    expect = {key: value for key, value in concrete.expect.items() if key not in concrete.not_compared}
    for key in COUNTED_COMPARED.get(concrete.listed_row, ()):
        if key in concrete.counted_values:
            expect[key] = concrete.counted_values[key]
    return grade(question, expect, outcome, attempts=attempts, member_files=member_files)


# --- The diff of a whole run -------------------------------------------------------


def _rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[tuple, list[tuple]]:
    """The rows of each sample in the order they were given, as (shard position,
    ordinal, token start, token end)."""
    found: dict[tuple, list[tuple]] = {}
    for row in rows:
        sample = tuple(row.get(name) for name in SAMPLE_FIELDS)
        found.setdefault(sample, []).append(tuple(row.get(name) for name in COVERED_FIELDS))
    return found


def grade_whole_run(
    question: Question,
    outcome: Reading | Attempt | None,
    oracle: Sequence[Mapping[str, Any]],
) -> Grade:
    """The grade of one of the three whole-run diff rows (``c01.E.diff``,
    ``c01.E-rr.diff``, ``c01.K.diff``).

    ``oracle`` is the oracle's log, a list of its lines as mappings. The rows the join
    must report for the run are worked out from it (``baselines.expected``), and the
    reading's ``rows`` (the subject's rows of the whole run, one mapping per row) must
    equal them sample by sample, in the order the rows are given within a sample. The
    diff a subject states is never read. A reading without rows has said nothing to
    compare and is wrong; the evidence is ``["diff"]``.

    ``NotGradable`` for no outcome, a failed attempt and a reading of another question;
    ``ValueError`` for a row that is no whole-run diff row and for a question that holds
    no record.
    """
    reading = _checked_reading(question, outcome)
    if question.route is None:
        raise ValueError(f"{question.id} holds no record: a whole-run diff is not asked of it")
    wanted = expectations.whole_run_rows(
        expectations.expected_for(question.listed_row, question.size), oracle
    )
    raw = {} if reading.raw is None else reading.raw
    if not isinstance(raw, Mapping):
        raise NotGradable("the reading's raw is not a mapping of fields")
    if _is_said(raw):
        return Grade(status=CANNOT_ANSWER, evidence=raw[CANNOT_KEY])
    rows = raw.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return Grade(status=WRONG, evidence=[DIFF_KEY])
    if _rows_by_sample(rows) != {sample: list(spans) for sample, spans in wanted.items()}:
        return Grade(status=WRONG, evidence=[DIFF_KEY])
    return Grade(status=RIGHT, evidence=None)


# --- A plan that allows a set of outcomes -------------------------------------------


def _count(value: Any) -> int | None:
    """``value`` when it is a whole number (a boolean is none), else None."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _kill_outcome(records: Mapping[str, Any]) -> str | None:
    """"allowed", "never" or "outside" from the rows the stopped write left, or None
    where the records do not say."""
    rows = _count(records.get("rows_after_stop"))
    if rows is None:
        return None
    if rows in (0, SHARD_ROWS):
        return "allowed"
    return "never" if 0 < rows < SHARD_ROWS else "outside"


def _named(value: Any) -> bool:
    """Whether a mismatch is named: a non-empty text."""
    return isinstance(value, str) and bool(value.strip())


def _retry_outcome(records: Mapping[str, Any], subject: str) -> str | None:
    """The same for the write made again, from the rows after it, whether their ranges
    index the documents of the token file (``matches_shard``, where the subject can say),
    whether the tool refused and the mismatch the refusal names.

    A subject that counts entries and matches none to a file (``COUNTS_ONLY_SUBJECTS``)
    has no ``matches_shard``: whatever its reader gives is not looked at, and its entries
    are read as the rows of a map that is bound to its shard, which only the count tells.

    Rows after the second run, by their number n:

    - n = 0: allowed after a refusal that names its mismatch; a refusal that names none is
      a gap in the records; no refusal leaves nothing the plan allows (outside).
    - 1 to 499: "a partial map", never allowed, when the rows index the documents of the
      token file, refused or not and whatever the refusal's message; rows that do not index
      them are allowed after a refusal that names its mismatch; one that names none, or
      records that cannot tell whether the rows index the documents, are a gap.
    - 500: allowed when the rows index the documents of the token file, whatever the refusal
      says; rows that do not index them are allowed only after a refusal that names its
      mismatch (without a refusal: never); a refusal that names none, or records that cannot
      tell, are a gap, and so is a write that was not refused with no word on the rows.
    - more than 500: never allowed when the rows index the documents of the token file
      (and for a subject that only counts entries), refused or not and whatever the refusal
      names: the plan never allows a map whose count does not match the shard, and a count
      over 500 is not the shard's. Rows that do not index them are allowed after a refusal
      that names its mismatch, a gap after one that names none, and outside the plan with
      no refusal. Records that cannot tell whether the rows index the documents are a gap.
    """
    rows = _count(records.get("rows_after_retry"))
    refused = records.get("refused")
    if rows is None or not isinstance(refused, bool):
        return None
    if subject in COUNTS_ONLY_SUBJECTS:
        matches: bool | None = True
    else:
        said = records.get("matches_shard")
        matches = said if isinstance(said, bool) else None
    named = _named(records.get("mismatch"))
    if rows == 0:
        return ("allowed" if named else None) if refused else "outside"
    if rows < SHARD_ROWS:
        if matches is True or not refused:
            return "never"
        return "allowed" if matches is False and named else None
    if rows == SHARD_ROWS:
        if matches is True:
            return "allowed"
        if refused:
            return "allowed" if named else None
        return "never" if matches is False else None
    if matches is True:
        return "never"
    if matches is None:
        return None
    return ("allowed" if named else None) if refused else "outside"


def grade_from_records(
    question: Question,
    expect: Mapping[str, Any],
    outcome: Reading | Attempt | None,
    *,
    attempts: Sequence[record.StoreAttempt] = (),
) -> Grade:
    """The grade of the two rows that allow a set of outcomes (``c04.kill-after-100`` and
    ``c04.retry``), read from the counts of the subject's own records
    (``raw["records"]``) and never from an outcome the reading names beside them.

    A stopped write leaves no row or the 500, both allowed; 1 to 499 rows are a partial
    map, which the plan never allows; any other count is outside what the plan allows
    (evidence ``["one_of"]``, ``["never"]`` for the forbidden ones). A write made again
    is allowed when it was refused with no row left and a named mismatch (the records'
    ``mismatch``, a non-empty text) or when it left the whole map matching its shard
    (the entries of OriginBlame, alone or beside the trace, are counted and matched to no
    file: 500 entries are right whatever the refusal says); a refusal after which 1 to 499
    rows remain that index the token file holds "a partial map", whatever its message, and
    so do more than 500 rows that index it (and any count over 500 entries of OriginBlame),
    refused or not (see ``_retry_outcome``). Records that lack the count, or the refusal, or
    the mismatch a refusal must name, or (over 500 rows, for the toolkit that matches rows
    to a file) the word on whether the rows index the token file, are a gap and wrong
    (evidence ``["records"]``).

    ``ValueError`` for any other row. A question that holds no record is graded by
    ``grade`` from the attempts. ``grade`` itself reads these two rows through this
    function.
    """
    if question.listed_row not in PLAN_ROWS:
        raise ValueError(f"{question.listed_row} does not allow a set of outcomes")
    reading = _checked_reading(question, outcome)
    if not expect or expect.get(APPLIES_KEY) is False:
        raise NotGradable(f"{question.listed_row} does not apply under the fallback in force")
    if question.route is None:
        return _none_line(question, attempts)
    raw = {} if reading.raw is None else reading.raw
    if not isinstance(raw, Mapping):
        raise NotGradable("the reading's raw is not a mapping of fields")
    said = _cannot_answer(question, expect, raw)
    if said is not None:
        return said
    return _plan_row_grade(question, raw)


def _plan_row_grade(question: Question, raw: Mapping[str, Any]) -> Grade:
    """The grade of a plan row from the counts of the subject's own records alone; a
    reading with no records, or records that lack what the row reads, is wrong."""
    records = raw.get(RECORDS_KEY)
    if not isinstance(records, Mapping):
        return Grade(status=WRONG, evidence=[RECORDS_KEY])
    if question.listed_row == KILL_ROW:
        found = _kill_outcome(records)
    else:
        found = _retry_outcome(records, question.subject)
    if found is None:
        return Grade(status=WRONG, evidence=[RECORDS_KEY])
    if found == "allowed":
        return Grade(status=RIGHT, evidence=None)
    return Grade(status=WRONG, evidence=["never"] if found == "never" else ["one_of"])


# --- A line asked in both layouts ------------------------------------------------------


def grade_layouts(
    question: Question,
    by_layout: Mapping[str, expectations.Concrete | ValueError],
    outcomes: Mapping[str, Reading | Attempt | None],
    *,
    attempts: Sequence[record.StoreAttempt] = (),
) -> Grade:
    """The one line of a listed row that names two layouts.

    ``by_layout`` is what ``baselines.expected.concrete_values_by_layout`` returns (a
    ``Concrete`` per layout, or the ``ValueError`` its expected value could not be
    derived with) and ``outcomes`` maps each layout to the reading of the subject on it,
    to a failed ``Attempt`` (that layout's attempt failed), or to None (or leaves it out)
    where the layout was not attempted. Each reading is graded by ``grade_concrete``; the
    line is put together by ``baselines.expected.combine_layouts``, in this order: wrong if
    any layout is wrong; otherwise ``no answer: failed attempt <id>`` (the first in the
    row's order) if any layout's attempt failed; otherwise ``not attempted`` if any layout
    was not attempted; otherwise right if all are right; otherwise cannot answer, the
    evidence naming each layout. A layout whose expected value could not be derived is that
    ``ValueError``, named, and a failed attempt in it is not looked at.

    ``ValueError`` for a row that names no layouts and for a layout it does not name;
    ``NotGradable`` for a reading of another question in any layout, and for any reading
    that cannot be put on a grade at all (it is no layout's result).
    """
    # Raises for a row that names no layouts and for a layout it does not name.
    expectations.combine_layouts(question.listed_row, dict.fromkeys(outcomes))
    for outcome in outcomes.values():
        if outcome is not None and not isinstance(outcome, Attempt):
            _checked_reading(question, outcome)
    if _holds_no_record(question):
        # A none line holds no record in either layout: the attempt decides, as in grade().
        return _none_line(question, attempts)
    results: dict[str, Grade | ValueError | str | None] = {}
    for layout, slot in by_layout.items():
        reading = outcomes.get(layout)
        if isinstance(slot, ValueError):
            results[layout] = slot
        elif isinstance(reading, Attempt):
            results[layout] = no_answer(reading.id)
        elif reading is None:
            results[layout] = record.NOT_ATTEMPTED
        else:
            results[layout] = grade_concrete(question, slot, reading, attempts=attempts)
    return expectations.combine_layouts(question.listed_row, results)
