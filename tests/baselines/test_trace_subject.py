"""The trace as a subject of Check 10, on hand-written documents (plan 5.5 Check 10; table
README, section "Check 10": the subject ``trace``; the rows ``c10.<group>.trace`` of
``check10.toml``).

What the subject is. The trace is the exposure join's own position output with every source id
removed: per segment its run, its shards in the trainer's order, its parent checkpoint, the
checkpoints saved in it with their steps, its launches and its steps (per step the number of
samples it counts; per sample its place in the step, its rank, its tokens and its covered tokens;
per row the shard, its position, the ordinal and the token range). It holds no occurrence,
record, text, fingerprint, file or source. Only three groups of the table have a route to it:
``c01.positions`` (11 listed rows), ``c01.step-facts`` (7) and ``c05.unregistered-checkpoint``
(1). Every other question is a row that records none. Nothing here starts a tool: the documents
are written by hand from the fixture spec and the table; the one export of a store is the last
test, which builds what it needs with the replica's own calls.

The names asked of ``baselines.trace``:

- ``export_segment(shown, run_record, step_record, parent_shown=None)``: the segment record of
  the trace from what ``show SEGMENT --store F --trace`` printed (``shown``), the run record and
  the step record of the segment (docs/interfaces.md 8.1 and 8.2) and, for a segment that starts
  from a checkpoint, what ``show CHECKPOINT`` printed for it. It returns a dict with exactly the
  keys ``segment``, ``run`` (the run record's), ``parent_checkpoint`` (the run record's name, or
  None), ``shards`` (the run record's shard prefixes in position order, each without its
  ``shards/`` folder), ``checkpoints`` (the step record's ``{name, step}`` entries in step
  order), ``launches`` (as shown, with the seven keys docs/cli.md lists) and ``steps`` (as shown,
  with only ``step``, ``consumed_samples`` and ``samples``; a sample keeps ``sample_pos``,
  ``rank``, ``sample_tokens``, ``covered_tokens`` and ``rows``; a row keeps ``shard``,
  ``shard_position``, ``ordinal``, ``token_start`` and ``token_end``). Every other key of the
  printed document is dropped at every level, so no source id, record, file or text the join
  holds can reach the trace; ``untraced_steps`` is dropped too, because the trace's own reading
  works the untraced steps out from the rows. A segment record is the form
  ``baselines.faults.apply_fault`` takes, and ``{"segments": [...]}`` of them is the form
  ``baselines.history.history_rows`` takes. ``ExportError`` (a ``ValueError``) for documents of
  different segments, a run record that names one shard position twice, and a parent checkpoint
  that ``show CHECKPOINT`` does not show as a registered checkpoint of that name from which the
  segment was started.
- ``classify_show(segment, command, exit_code, output, attempt_id)``: what one start of the
  export command came to. Exit 0 with a JSON object whose ``segment`` is the one asked: that
  object. Exit 2 with the words "request refused" (the refusal docs/cli.md documents for a name
  that is no registered run segment): a ``Refusal`` with ``exit_code`` and ``output``, which is
  the subject's answer that the store holds no run, not a failure. Anything else (another exit,
  exit 2 without those words, an exit that did not come, output that is no JSON object, an
  object of another segment): a failed ``baselines.interfaces.Attempt`` with the id, the
  command, the exit code and the output, with every local path taken out of both the command and
  the output.
- ``checkpoint_record(document, name)``: ``{"segment", "step"}`` of the checkpoint the trace
  holds under ``name``, else None.
- ``untraced_steps(document, checkpoint)``: ``[segment, step]`` pairs (the order is free) of the
  steps of the checkpoint's history that have no row or an uncovered range. The history is the
  checkpoint's segment up to the checkpoint's step, then the history of the segment's parent
  checkpoint up to its step, and so on; the steps of a segment are those of its launches (each
  launch's ``first_step`` to ``last_step``). A step is untraced when it has no entry in
  ``steps`` or its samples hold no row (both forms of an absent trace), or a sample of it has
  fewer covered tokens than sample tokens. A step whose count differs from its samples is not
  untraced for that: no rule of the table turns it into doubt. None where the trace holds no
  such checkpoint.
- ``TraceSubject(document, *, place_of)``: a ``baselines.interfaces.Subject`` named ``trace``.
  ``document`` is ``{"segments": [...]}``. ``place_of(listed_row)`` gives the places the listed
  row reads, a list of mappings: for a row of ``c01.positions`` that names the lines of the
  oracle's log, one place per line (``segment``, ``step``, ``sample_pos``, ``shard_position``,
  ``ordinal``); for a whole-run row ``{"run": R}``; for a row of ``c01.step-facts`` and for
  ``c05.unregistered-checkpoint`` the listed row's own ``given``. ``read(question)`` gives a
  ``Reading`` (its ``question_id`` and its ``route`` are the question's). A question whose
  row records none gets a Reading whose ``raw`` is None and whose ``route`` is None, whatever
  the document holds: nothing is read, and the line is written from the subject's attempt on
  its store. A routed question gets a ``raw`` whose keys are exactly the values its rule names,
  and a value the trace holds nothing for is None (the grade is wrong, never a gap read as an
  absence):

  * ``c01.positions``: ``rows``, the rows of the places in the order of the places, each place
    giving its rows in the document's order (a row matches a place when every key the place
    names is equal; ``run`` is the run of the row's segment); each row is a mapping with
    ``segment``, ``step``, ``sample_pos``, ``shard_position``, ``ordinal``, ``token_start`` and
    ``token_end``. The oracle's lines name no sample or step alone: two of them lie in one
    sample, so a reader of whole samples reads rows no line names.
  * ``c01.E.step8.uncovered``: ``uncovered_in_sample`` (``[covered_tokens, sample_tokens]`` of
    the sample at the place, None for a whole sample), ``sample_tokens``, ``covered_tokens``.
  * ``c01.E.step8.counted``: ``consumed_samples``, the step's own count.
  * ``c01.E.step8.untraced``: ``untraced_steps``, as the function reads it for the checkpoint
    the place names.
  * ``c01.E-rr.fully-traced``: ``steps_with_uncovered_range`` and ``steps_without_trace``, each
    a list of ``[segment, step]`` over every step of every segment of the run the place names.
  * ``c01.K.warm-start`` and ``c01.K.resume``: ``consumed_at_start``, ``first_step`` and
    ``last_step`` of the launch (its ``position``) the place names.
  * ``c01.K.ranks``: ``samples_per_step``, ``samples_per_rank``, ``ranks`` and
    ``sample_pos_of_rank_<r>`` for each rank, the value every step of the segment agrees on
    (None where the steps differ or the segment holds none).
  * ``c05.unregistered-checkpoint``: ``answer``, "not covered" where the trace holds no
    checkpoint of the name the place names (the reading "no record" equals that answer), else
    an answer word that is not "not covered".

  The subject changes no document it is given and starts nothing.

Looked up in the body of a test, so that a module that does not exist yet fails a test on an
assertion and not on an import.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
MODULE = "baselines.trace"
MEGATRON_VARIABLE = "WATERWHITE_MEGATRON_ENV"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "replica",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")
LOCAL_PATH = "/ho" + "me/someone/work"


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


# --- The table and the spec, as the files write them ---------------------------------------


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _spec() -> dict:
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
def _check10() -> dict:
    return _read_toml(TABLE_DIR / "check10.toml")


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _check10()["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, tuple[str, int]]:
    found = {}
    for group in _check10()["group"]:
        for listed in group["rows"]:
            found[listed] = (group["id"], group["check"])
    return found


def _rows_of_group(group_id: str) -> tuple[str, ...]:
    (group,) = [g for g in _check10()["group"] if g["id"] == group_id]
    return tuple(group["rows"])


def _expect(listed_row: str) -> dict:
    """The expectation of a row with no fallback: the table's ``expect``, its pending values
    filled from the spec."""

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


def _make_question(listed_row: str, subject: str = "trace", size: str | None = None):
    """The question that puts a listed row to a subject, its fields taken from the subject's
    row of the group in ``check10.toml``."""
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


# --- What each routed row of the trace reads ------------------------------------------------

WHOLE_RUN_ROWS = ("c01.E.diff", "c01.E-rr.diff", "c01.K.diff")
# The rows of c01.positions whose expectation names lines of the oracle's log.
LINE_ROWS = (
    "c01.E.T",
    "c01.E.skipped-line",
    "c01.E.repeated-draw",
    "c01.E.shared-ordinal",
    "c01.E.split-document",
    "c01.E.shared-token",
    "c01.E.step8.first-half",
    "c01.K.epoch-boundary",
)
LAUNCH_VALUES = ("consumed_at_start", "first_step", "last_step")

# Read from the rule of each row of the trace in check10.toml ("the reading is the trace's rows
# for the run, segment, step or position of the listed row"; for the step facts "the reading is
# the value the listed row asks for, taken from the trace: ..."), and from the keys of the
# listed row's own expectation. The values the rule does not name are in LEFT_OUT below.
NAMED_VALUES = {
    **{listed: ("rows",) for listed in WHOLE_RUN_ROWS + LINE_ROWS},
    "c01.E.step8.uncovered": ("uncovered_in_sample", "sample_tokens", "covered_tokens"),
    "c01.E.step8.counted": ("consumed_samples",),
    "c01.E.step8.untraced": ("untraced_steps",),
    "c01.E-rr.fully-traced": ("steps_with_uncovered_range", "steps_without_trace"),
    "c01.K.warm-start": LAUNCH_VALUES,
    "c01.K.resume": LAUNCH_VALUES,
    "c01.K.ranks": ("samples_per_step", "samples_per_rank", "ranks", "sample_pos_of_rank_0",
                    "sample_pos_of_rank_1"),
    "c05.unregistered-checkpoint": ("answer",),
}
# The keys of a step-facts row's expectation that its rule does not name: the loud check is a
# command of the tool that the trace has none of, ``starts_at`` is the tool's own way of saying
# where the cut falls, and the rows of a launch row are the places its values are read from.
LEFT_OUT = {
    "c01.E.step8.counted": {"loud_stop"},
    "c01.E.step8.uncovered": {"starts_at"},
    "c01.K.warm-start": {"rows", "diff"},
    "c01.K.resume": {"rows", "diff"},
    "c05.unregistered-checkpoint": {"reason"},
}
ROW_FIELDS = ("segment", "step", "sample_pos", "shard_position", "ordinal", "token_start",
              "token_end")
PLACE_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal")


def place_of(listed_row: str) -> list[dict]:
    """The places a listed row reads, from the table and the spec."""
    row = _listed_rows()[listed_row]
    group = _group_of_listed()[listed_row][0]
    if group == "c01.positions":
        rows = row["expect"].get("rows")
        if isinstance(rows, str) and rows.startswith("pending:positions."):
            lines = _spec_lines(rows.split(".", 1)[1])
            return [{key: line[key] for key in PLACE_KEYS} for line in lines]
        return [{"run": row["given"]["run"]}]
    return [dict(row["given"])]


# --- The hand-written trace ---------------------------------------------------------------

SAMPLE_TOKENS = 33
BATCH_SIZE = 4
POSITION_KEYS_OF_RUN_E = ("skipped_line", "T", "step8_cut", "repeated_draw", "shared_ordinal",
                          "split_document", "shared_token", "seg1_shards")
CUT_SAMPLE = ("E/seg-2", 8, 3)


def _spec_lines(*keys: str) -> list[dict]:
    return [json.loads(text) for key in keys for text in _spec()["positions"][key]["log_lines"]]


def _launch(position, consumed, step_at_start, first, last, world=1):
    return {"position": position, "world_size": world, "data_parallel_size": world,
            "consumed_at_start": consumed, "step_at_start": step_at_start,
            "first_step": first, "last_step": last}


# The segments of the spec (runs.E, runs.E-rr and runs.K): shards in the trainer's order,
# parent checkpoint, checkpoints saved, launches.
SEGMENT_FACTS = {
    "E/seg-1": dict(run="E", parent=None, shards=["shard-1", "shard-2"],
                    checkpoints=[("E@6", 6)], launches=[_launch(0, 0, 0, 1, 6)]),
    "E/seg-2": dict(run="E", parent="E@6", shards=["shard-1", "shard-2", "shard-3"],
                    checkpoints=[("E@9", 9)], launches=[_launch(0, 24, 6, 7, 12)]),
    "E-rr/seg-1": dict(run="E-rr", parent=None, shards=["shard-1", "shard-2", "shard-3"],
                       checkpoints=[("E-rr@6", 6)], launches=[_launch(0, 0, 0, 1, 6)]),
    "E-rr/seg-2": dict(run="E-rr", parent="E-rr@6", shards=["shard-1", "shard-2", "shard-3"],
                       checkpoints=[], launches=[_launch(0, 24, 6, 7, 12)]),
    "K/seg-1": dict(run="K", parent="E@6", shards=["shard-3", "shard-2"],
                    checkpoints=[("K@3", 3), ("K@8", 8), ("K@40", 40)],
                    launches=[_launch(0, 0, 0, 1, 3, world=2), _launch(1, 12, 3, 4, 40, world=2)]),
}


def _same_row(line: dict):
    return tuple(line[key] for key in ROW_FIELDS)


def _oracle_lines() -> list[dict]:
    """The oracle's lines of the hand-written world: the lines the spec records for the rows of
    c01 (run E and run K), then one filler line for every sample of every step of every
    launch that no line names, so that every step is traced. A line is listed once."""
    lines: list[dict] = []
    seen = set()
    for line in _spec_lines(*POSITION_KEYS_OF_RUN_E) + _spec_lines("K"):
        if _same_row(line) not in seen:
            seen.add(_same_row(line))
            lines.append(line)
    named = {(line["segment"], line["step"], line["sample_pos"]) for line in lines}
    for segment, facts in SEGMENT_FACTS.items():
        for launch in facts["launches"]:
            for step in range(launch["first_step"], launch["last_step"] + 1):
                for pos in range(BATCH_SIZE):
                    if (segment, step, pos) in named:
                        continue
                    lines.append({
                        "segment": segment, "step": step, "sample_pos": pos,
                        "global_index": 0, "rank": pos // 2 if facts["run"] == "K" else 0,
                        "shard_position": 0, "ordinal": 1000 + 10 * step + pos,
                        "token_start": 0, "token_end": SAMPLE_TOKENS})
    return lines


def _document(lines: list[dict]) -> dict:
    """The trace a faithful join reports for the lines: every line a row, a sample that holds
    all of its 33 tokens, except the last sample of step 8 of E/seg-2, which the spec cuts to the
    rows of its covered part (16 tokens)."""
    cut = _spec()["positions"]["step8_cut"]
    segments = []
    for name, facts in SEGMENT_FACTS.items():
        own = [line for line in lines if line["segment"] == name]
        steps = []
        for step in sorted({line["step"] for line in own}):
            samples = []
            at_step = [line for line in own if line["step"] == step]
            for pos in sorted({line["sample_pos"] for line in at_step}):
                held = [line for line in at_step if line["sample_pos"] == pos]
                rows = [{"shard": facts["shards"][line["shard_position"]],
                         "shard_position": line["shard_position"], "ordinal": line["ordinal"],
                         "token_start": line["token_start"], "token_end": line["token_end"]}
                        for line in held]
                covered = SAMPLE_TOKENS
                if (name, step, pos) == CUT_SAMPLE:
                    covered = cut["fixed"]["covered_tokens"]
                    rows = [{"shard": facts["shards"][r["shard_position"]], **r}
                            for r in cut["rows_of_the_covered_part"]]
                samples.append({"sample_pos": pos, "rank": held[0]["rank"],
                                "sample_tokens": SAMPLE_TOKENS, "covered_tokens": covered,
                                "rows": rows})
            steps.append({"step": step, "consumed_samples": BATCH_SIZE, "samples": samples})
        segments.append({
            "segment": name, "run": facts["run"], "parent_checkpoint": facts["parent"],
            "shards": list(facts["shards"]),
            "checkpoints": [{"name": n, "step": s} for n, s in facts["checkpoints"]],
            "launches": copy.deepcopy(facts["launches"]), "steps": steps})
    return {"segments": segments}


ALL_LINES = _oracle_lines()


def make_document() -> dict:
    return _document(ALL_LINES)


def _segment(document: dict, name: str) -> dict:
    (found,) = [s for s in document["segments"] if s["segment"] == name]
    return found


def _step(document: dict, name: str, number: int) -> dict:
    (found,) = [s for s in _segment(document, name)["steps"] if s["step"] == number]
    return found


def _sample(document: dict, name: str, number: int, pos: int) -> dict:
    (found,) = [s for s in _step(document, name, number)["samples"] if s["sample_pos"] == pos]
    return found


def _one_segment(segment: dict) -> dict:
    return {"segments": [segment]}


def _row_of(document: dict, key: dict) -> dict:
    (found,) = [r for r in _sample(document, key["segment"], key["step"],
                                   key["sample_pos"])["rows"]
                if r["shard_position"] == key["shard_position"] and r["ordinal"] == key["ordinal"]]
    return found


def _subject(document: dict):
    return need_attr(MODULE, "TraceSubject")(document, place_of=place_of)


def _read(document: dict, listed: str):
    return _subject(document).read(_make_question(listed))


def _grade(listed: str, reading, **kwargs):
    grade = need_attr("baselines.grade", "grade")
    return grade(_make_question(listed), _expect(listed), reading, **kwargs)


def _grade_whole_run(listed: str, reading):
    run = need_attr("baselines.grade", "grade_whole_run")
    return run(_make_question(listed), reading, ALL_LINES)


def _pairs(items) -> list[tuple]:
    return sorted(tuple(item) for item in items)


# --- The questions of the table that the trace has a route for -------------------------------


def _routed_trace_triples(facts) -> list[tuple]:
    """(listed row, subject, size) of the questions whose row of the trace records something,
    counted from the table files and the reader of the tool's own rows."""
    routed_groups = {r["given"]["group"] for r in facts.rows
                     if r["given"]["subject"] == "trace" and r["record"] != "none"}
    return [t for t in facts.question_triples()
            if t[1] == "trace" and facts.group_of(t[0]) in routed_groups]


def _questions_of_trace():
    table10 = need("baselines.table10")
    return [q for q in table10.expand(table10.load_table10()) if q.subject == "trace"]


def _check_the_routed_questions(triples, named_values, expanded) -> None:
    """The questions the trace has a route for: counted from the table files, equal to the ones
    the expander gives, and named by exactly the rows the readings below cover."""
    assert len(triples) == 19
    assert {_group_of_listed()[t[0]][0] for t in triples} == {
        "c01.positions", "c01.step-facts", "c05.unregistered-checkpoint"}
    assert {t[2] for t in triples} == {None}
    assert set(named_values) == {t[0] for t in triples}
    assert sorted(q.id for q in expanded if q.route is not None) == sorted(
        f"{t[0]}|trace" for t in triples)


def test_the_trace_has_a_route_for_nineteen_questions_in_three_groups(facts):
    triples = _routed_trace_triples(facts)
    expanded = _questions_of_trace()
    assert [len(_rows_of_group(g)) for g in
            ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint")] == [11, 7, 1]
    assert len([t for t in facts.question_triples() if t[1] == "trace"]) == facts.questions_per_subject
    assert len(expanded) == 391
    _check_the_routed_questions(triples, NAMED_VALUES, expanded)
    # Known-bad inputs: one question counted short, one row with no reading named, one the
    # expander does not route.
    with pytest.raises(AssertionError):
        _check_the_routed_questions(triples[:-1], NAMED_VALUES, expanded)
    with pytest.raises(AssertionError):
        _check_the_routed_questions(
            triples, {k: v for k, v in NAMED_VALUES.items() if k != "c01.K.ranks"}, expanded)
    with pytest.raises(AssertionError):
        _check_the_routed_questions(
            triples, NAMED_VALUES,
            [dataclasses.replace(q, route=None) if q.listed_row == "c01.E.T" else q
             for q in expanded])


def _check_the_named_values(named_values) -> None:
    for listed, named in named_values.items():
        group = _group_of_listed()[listed][0]
        if group == "c01.positions":
            continue
        keys = set(_listed_rows()[listed]["expect"])
        assert set(named) == keys - LEFT_OUT.get(listed, set()), listed


def test_the_values_a_row_names_are_the_keys_of_its_expectation_less_what_its_rule_leaves_out():
    _check_the_named_values(NAMED_VALUES)
    # Known-bad inputs: a value of the expectation named by no reading, and a value named that
    # the expectation does not hold.
    with pytest.raises(AssertionError):
        _check_the_named_values({**NAMED_VALUES, "c01.K.ranks": ("samples_per_step",)})
    with pytest.raises(AssertionError):
        _check_the_named_values({**NAMED_VALUES, "c01.E.step8.counted": ("consumed_samples",
                                                                          "loud_stop")})
    # The rules these readings rest on, word for word from the table.
    step_rule = _check10_rows()["c10.c01.step-facts.trace"]["rule"]
    assert step_rule.startswith("the reading is the value the listed row asks for, taken from the trace")
    for text in ("the uncovered range of a sample (its covered tokens against its sample tokens)",
                 "the number of samples a step counts",
                 "the steps of a history that have no row or an uncovered range (read as untraced)",
                 "the first and last step of a launch and the consumed-sample counter its first "
                 "sample starts from",
                 "the samples of each rank in a step"):
        assert text in step_rule, text
    assert _check10_rows()["c10.c01.positions.trace"]["rule"] == (
        "the reading is the trace's rows for the run, segment, step or position of the listed row.")
    assert _check10_rows()["c10.c05.unregistered-checkpoint.trace"]["rule"].startswith(
        "no record: the trace holds no checkpoint of that name.")


# --- The checker of the mechanical bar, and the known-bad reader it must catch ---------------


def _holds_exactly_the_named_values(subject, questions) -> bool:
    """True when each routed question gets a reading whose keys are exactly the values its row
    names, and each question that records none gets no raw."""
    for question in questions:
        got = subject.read(question)
        if question.route is None:
            if got.raw is not None:
                return False
        elif set(got.raw) != set(NAMED_VALUES[question.listed_row]):
            return False
    return True


class _StandIn:
    """A reader that holds exactly the named values (hand-made, for the checker)."""

    name = "trace"

    def read(self, question):
        raw = None if question.route is None else {k: 0 for k in NAMED_VALUES[question.listed_row]}
        return _Got(raw)


class _Got:
    def __init__(self, raw):
        self.raw = raw


class _Dropping:
    """A known-bad reader: it drops one named field of every row that holds it."""

    name = "trace"

    def __init__(self, inner, field):
        self.inner, self.field = inner, field

    def read(self, question):
        got = self.inner.read(question)
        if got.raw is None:
            return got
        return _Got({k: v for k, v in got.raw.items() if k != self.field})


def test_the_checker_of_the_mechanical_bar_catches_a_reader_that_drops_a_named_field(facts):
    questions = [_make_question(listed) for listed in NAMED_VALUES] + [
        _make_question(_rows_of_group("c05.f.trace-deleted")[0])]
    assert _holds_exactly_the_named_values(_StandIn(), questions) is True
    for field in ("rows", "uncovered_in_sample", "consumed_samples", "untraced_steps",
                  "steps_without_trace", "first_step", "sample_pos_of_rank_1", "answer"):
        assert _holds_exactly_the_named_values(_Dropping(_StandIn(), field), questions) is False, field


def test_every_routed_question_gets_exactly_the_values_its_rule_names_and_every_other_none(facts):
    document = make_document()
    before = copy.deepcopy(document)
    subject = _subject(document)
    reading_type = need_attr("baselines.interfaces", "Reading")
    questions = _questions_of_trace()
    assert subject.name == "trace"
    assert _holds_exactly_the_named_values(subject, questions) is True
    # A reader that drops one named field is caught by the same check.
    assert _holds_exactly_the_named_values(_Dropping(subject, "rows"), questions) is False
    assert _holds_exactly_the_named_values(_Dropping(subject, "steps_with_uncovered_range"),
                                           questions) is False
    for question in questions:
        got = subject.read(question)
        assert isinstance(got, reading_type), question.id
        assert got.question_id == question.id
        assert got.route == question.route
    # Reading changes no document, and reading twice gives the same.
    assert document == before
    assert [subject.read(q) for q in questions] == [subject.read(q) for q in questions]
    # A question that records none is read as nothing whatever the document holds.
    empty = _subject({"segments": []})
    for question in questions:
        if question.route is None:
            assert empty.read(question).raw is None and empty.read(question).route is None


# --- The rows of c01.positions ----------------------------------------------------------------


def _check_the_neighbours(document) -> None:
    # The lines of skipped_line and split_document lie in one sample, which holds both rows;
    # the sample of T holds three rows and its step holds three samples more.
    assert len(_sample(document, "E/seg-1", 1, 0)["rows"]) >= 2
    assert len(_sample(document, "E/seg-2", 7, 0)["rows"]) == 3
    assert len(_step(document, "E/seg-2", 7)["samples"]) == BATCH_SIZE
    # The lines of the cut sample hold two rows; the trace holds the covered part only.
    assert len(_spec_lines("step8_cut")) == 2
    assert len(_sample(document, "E/seg-2", 8, 3)["rows"]) == 1


def test_the_hand_written_trace_has_neighbouring_rows_in_the_samples_the_lines_name():
    document = make_document()
    _check_the_neighbours(document)
    # Known-bad input: a trace made from the lines of one row alone has no neighbour, so a
    # reader of whole samples would not be told from a reader of the lines.
    alone = _document([line for line in ALL_LINES if line["ordinal"] == 17 and line["step"] == 1])
    with pytest.raises((AssertionError, ValueError)):
        _check_the_neighbours(alone)


def test_the_rows_of_the_places_a_line_names_equal_those_lines_and_not_the_whole_sample():
    document = make_document()
    for listed in LINE_ROWS:
        got = _read(document, listed)
        lines = _spec_lines(_listed_rows()[listed]["expect"]["rows"].split(".", 1)[1])
        wanted = lines
        if listed == "c01.E.step8.first-half":
            cut = _spec()["positions"]["step8_cut"]["rows_of_the_covered_part"]
            wanted = [{**lines[0], **cut[0]}]
        assert [_same_row(r) for r in got.raw["rows"]] == [_same_row(l) for l in wanted], listed
        assert set(got.raw) == {"rows"}, listed
    # The rows of one sample that two rows' lines share: each row's lines name its own.
    skipped = _read(document, "c01.E.skipped-line").raw["rows"]
    assert len(skipped) == 1 and skipped[0]["ordinal"] == 17
    split = _read(document, "c01.E.split-document").raw["rows"]
    assert [r["ordinal"] for r in split] == [8, 8] and split[0]["segment"] != split[1]["segment"]


def test_a_row_the_lines_name_is_graded_right_and_a_changed_row_or_a_missing_row_wrong():
    document = make_document()
    for listed in LINE_ROWS:
        right = _grade(listed, _read(document, listed))
        assert right.status == "right" and right.evidence is None, listed
        key = place_of(listed)[0]
        # One row of the trace changed: its token range no longer equals the line's.
        changed = copy.deepcopy(document)
        _row_of(changed, key)["token_end"] += 1
        wrong = _grade(listed, _read(changed, listed))
        assert wrong.status == "wrong" and list(wrong.evidence) == ["rows"], listed
        # One row of the trace deleted: the line names a row the trace does not hold.
        missing = copy.deepcopy(document)
        sample = _sample(missing, key["segment"], key["step"], key["sample_pos"])
        sample["rows"] = [r for r in sample["rows"] if r is not _row_of(missing, key)]
        wrong = _grade(listed, _read(missing, listed))
        assert wrong.status == "wrong" and list(wrong.evidence) == ["rows"], listed


def test_the_rows_of_a_whole_run_are_every_row_of_its_segments_and_graded_against_the_log():
    document = make_document()
    for listed, segments in (("c01.E.diff", {"E/seg-1", "E/seg-2"}),
                             ("c01.E-rr.diff", {"E-rr/seg-1", "E-rr/seg-2"}),
                             ("c01.K.diff", {"K/seg-1"})):
        got = _read(document, listed)
        assert set(got.raw) == {"rows"}
        assert {r["segment"] for r in got.raw["rows"]} == segments, listed
        held = [r for s in document["segments"] if s["segment"] in segments
                for st in s["steps"] for sm in st["samples"] for r in sm["rows"]]
        assert len(got.raw["rows"]) == len(held), listed
        right = _grade_whole_run(listed, got)
        assert right.status == "right" and right.evidence is None, listed
        # One row changed, and one row deleted: the diff is not empty.
        changed = copy.deepcopy(document)
        name = sorted(segments)[0]
        _step(changed, name, 1)["samples"][0]["rows"][0]["token_end"] += 1
        assert _grade_whole_run(listed, _read(changed, listed)).status == "wrong", listed
        deleted = copy.deepcopy(document)
        _step(deleted, name, 2)["samples"][1]["rows"].pop()
        assert _grade_whole_run(listed, _read(deleted, listed)).status == "wrong", listed
    # The cut sample of step 8 holds the covered part: the run's rows are the oracle's with the cut.
    e = _read(document, "c01.E.diff").raw["rows"]
    cut = [r for r in e if (r["segment"], r["step"], r["sample_pos"]) == CUT_SAMPLE]
    assert [(r["ordinal"], r["token_end"]) for r in cut] == [(2, 35)]


# --- The rows of c01.step-facts ----------------------------------------------------------------


def test_the_uncovered_range_of_the_cut_sample_is_its_covered_tokens_against_its_sample_tokens():
    document = make_document()
    expect = _expect("c01.E.step8.uncovered")
    assert expect["uncovered_in_sample"] == [16, 33]
    got = _read(document, "c01.E.step8.uncovered").raw
    assert got == {"uncovered_in_sample": [16, 33], "sample_tokens": 33, "covered_tokens": 16}
    assert _grade("c01.E.step8.uncovered", _read(document, "c01.E.step8.uncovered")).status == "right"
    # A sample cut at 17 tokens, and a sample that holds all its tokens.
    changed = copy.deepcopy(document)
    _sample(changed, *CUT_SAMPLE)["covered_tokens"] = 17
    assert _read(changed, "c01.E.step8.uncovered").raw["uncovered_in_sample"] == [17, 33]
    bad = _grade("c01.E.step8.uncovered", _read(changed, "c01.E.step8.uncovered"))
    assert bad.status == "wrong" and list(bad.evidence) == ["covered_tokens", "uncovered_in_sample"]
    whole = copy.deepcopy(document)
    _sample(whole, *CUT_SAMPLE)["covered_tokens"] = SAMPLE_TOKENS
    assert _read(whole, "c01.E.step8.uncovered").raw["uncovered_in_sample"] is None
    # A sample the trace does not hold reads None in every value.
    gone = copy.deepcopy(document)
    _step(gone, "E/seg-2", 8)["samples"] = [s for s in _step(gone, "E/seg-2", 8)["samples"]
                                           if s["sample_pos"] != 3]
    assert _read(gone, "c01.E.step8.uncovered").raw == {
        "uncovered_in_sample": None, "sample_tokens": None, "covered_tokens": None}


def test_the_count_of_a_step_is_the_number_of_samples_it_counts():
    document = make_document()
    assert _expect("c01.E.step8.counted")["consumed_samples"] == 4
    assert _read(document, "c01.E.step8.counted").raw == {"consumed_samples": 4}
    changed = copy.deepcopy(document)
    _step(changed, "E/seg-2", 8)["consumed_samples"] = 3
    assert _read(changed, "c01.E.step8.counted").raw == {"consumed_samples": 3}
    no_step = copy.deepcopy(document)
    _segment(no_step, "E/seg-2")["steps"] = [s for s in _segment(no_step, "E/seg-2")["steps"]
                                            if s["step"] != 8]
    assert _read(no_step, "c01.E.step8.counted").raw == {"consumed_samples": None}


def test_the_untraced_steps_of_a_history_are_those_with_no_row_or_an_uncovered_range():
    untraced = need_attr(MODULE, "untraced_steps")
    document = make_document()
    expect = _expect("c01.E.step8.untraced")
    assert expect["untraced_steps"] == [["E/seg-2", 8]]
    assert _pairs(untraced(document, "E@9")) == [("E/seg-2", 8)]
    assert untraced(document, "E@6") == []
    assert untraced(document, "E@12") is None
    got = _read(document, "c01.E.step8.untraced")
    assert set(got.raw) == {"untraced_steps"}
    assert _pairs(got.raw["untraced_steps"]) == [("E/seg-2", 8)]
    assert _grade("c01.E.step8.untraced", got).status == "right"
    # A step whose samples hold no row, and one that has no entry, are both untraced.
    empty = copy.deepcopy(document)
    for sample in _step(empty, "E/seg-2", 7)["samples"]:
        sample["rows"] = []
    absent = copy.deepcopy(document)
    _segment(absent, "E/seg-2")["steps"] = [s for s in _segment(absent, "E/seg-2")["steps"]
                                           if s["step"] != 7]
    for form in (empty, absent):
        assert _pairs(untraced(form, "E@9")) == [("E/seg-2", 7), ("E/seg-2", 8)]
        bad = _grade("c01.E.step8.untraced", _read(form, "c01.E.step8.untraced"))
        assert bad.status == "wrong" and list(bad.evidence) == ["untraced_steps"]
    # A step in a parent checkpoint's segment counts in the history of the child.
    parent = copy.deepcopy(document)
    _segment(parent, "E/seg-1")["steps"] = [s for s in _segment(parent, "E/seg-1")["steps"]
                                           if s["step"] != 2]
    assert _pairs(untraced(parent, "E@9")) == [("E/seg-1", 2), ("E/seg-2", 8)]


def test_a_step_with_a_fault_put_to_the_trace_is_untraced_in_either_form():
    untraced = need_attr(MODULE, "untraced_steps")
    apply_fault = need_attr("baselines.faults", "apply_fault")
    document = make_document()
    # The segment record is the form the fault applicator takes.
    for fault, step in (("f.trace-deleted", 3), ("f.trace-half", 4)):
        faulted = copy.deepcopy(document)
        segment = apply_fault(fault, "trace", _segment(faulted, "E/seg-1"))
        faulted["segments"] = [segment if s["segment"] == "E/seg-1" else s
                               for s in faulted["segments"]]
        assert _pairs(untraced(faulted, "E@6")) == [("E/seg-1", step)], fault
        assert _pairs(untraced(faulted, "E@9")) == [("E/seg-1", step), ("E/seg-2", 8)], fault
    # The deleted step as an entry with no samples reads the same as no entry.
    empty = copy.deepcopy(document)
    _step(empty, "E/seg-1", 3)["samples"] = []
    assert _pairs(untraced(empty, "E@6")) == [("E/seg-1", 3)]
    # The history reader takes the same document.
    history_rows = need_attr("baselines.history", "history_rows")
    rows = history_rows(document, "E@9")
    assert rows and {r.segment for r in rows} == {"E/seg-1", "E/seg-2"}
    assert history_rows(document, "E@12") is None


def test_every_step_of_the_round_robin_run_is_traced_and_a_missing_or_cut_step_is_named():
    document = make_document()
    expect = _expect("c01.E-rr.fully-traced")
    assert expect == {"steps_with_uncovered_range": [], "steps_without_trace": []}
    got = _read(document, "c01.E-rr.fully-traced")
    assert got.raw == expect
    assert _grade("c01.E-rr.fully-traced", got).status == "right"
    missing = copy.deepcopy(document)
    _segment(missing, "E-rr/seg-1")["steps"] = [s for s in _segment(missing, "E-rr/seg-1")["steps"]
                                               if s["step"] != 3]
    got = _read(missing, "c01.E-rr.fully-traced")
    assert got.raw["steps_without_trace"] == [["E-rr/seg-1", 3]]
    assert got.raw["steps_with_uncovered_range"] == []
    bad = _grade("c01.E-rr.fully-traced", got)
    assert bad.status == "wrong" and list(bad.evidence) == ["steps_without_trace"]
    cut = copy.deepcopy(document)
    _sample(cut, "E-rr/seg-2", 9, 1)["covered_tokens"] = 16
    got = _read(cut, "c01.E-rr.fully-traced")
    assert got.raw["steps_with_uncovered_range"] == [["E-rr/seg-2", 9]]
    assert got.raw["steps_without_trace"] == []
    bad = _grade("c01.E-rr.fully-traced", got)
    assert bad.status == "wrong" and list(bad.evidence) == ["steps_with_uncovered_range"]
    # A step of the run E is no step of the round-robin run.
    assert _read(make_document(), "c01.E-rr.fully-traced").raw == expect


def test_a_launch_is_read_from_the_launches_of_its_segment():
    document = make_document()
    for listed, launch in (("c01.K.warm-start", 0), ("c01.K.resume", 1)):
        expect = _expect(listed)
        got = _read(document, listed)
        assert got.raw == {k: expect[k] for k in LAUNCH_VALUES}, listed
        assert (got.raw["consumed_at_start"], got.raw["first_step"], got.raw["last_step"]) == {
            0: (0, 1, 3), 1: (12, 4, 40)}[launch]
        assert _grade(listed, got).status == "right", listed
        changed = copy.deepcopy(document)
        _segment(changed, "K/seg-1")["launches"][launch]["consumed_at_start"] += 24
        bad = _grade(listed, _read(changed, listed))
        assert bad.status == "wrong" and list(bad.evidence) == ["consumed_at_start"], listed
        no_launch = copy.deepcopy(document)
        _segment(no_launch, "K/seg-1")["launches"] = []
        assert _read(no_launch, listed).raw == {k: None for k in LAUNCH_VALUES}, listed


def test_the_samples_of_each_rank_are_what_every_step_of_the_segment_agrees_on():
    document = make_document()
    expect = _expect("c01.K.ranks")
    got = _read(document, "c01.K.ranks")
    assert got.raw == expect
    assert _grade("c01.K.ranks", got).status == "right"
    # One sample of rank 1 missing from one step: the steps no longer agree.
    changed = copy.deepcopy(document)
    step = _step(changed, "K/seg-1", 5)
    step["samples"] = [s for s in step["samples"] if s["sample_pos"] != 3]
    got = _read(changed, "c01.K.ranks")
    assert got.raw["samples_per_step"] is None and got.raw["samples_per_rank"] is None
    bad = _grade("c01.K.ranks", got)
    assert bad.status == "wrong" and "samples_per_step" in list(bad.evidence)
    # A sample moved to the other rank in one step.
    moved = copy.deepcopy(document)
    _sample(moved, "K/seg-1", 7, 1)["rank"] = 1
    got = _read(moved, "c01.K.ranks")
    assert got.raw["samples_per_rank"] is None
    assert _grade("c01.K.ranks", got).status == "wrong"


# --- c05.unregistered-checkpoint ---------------------------------------------------------------


def test_a_checkpoint_the_trace_does_not_hold_reads_no_record_which_is_the_answer_not_covered():
    checkpoint_record = need_attr(MODULE, "checkpoint_record")
    document = make_document()
    assert _expect("c05.unregistered-checkpoint")["answer"] == "not covered"
    assert checkpoint_record(document, "E@12") is None
    assert checkpoint_record(document, "E@9") == {"segment": "E/seg-2", "step": 9}
    assert checkpoint_record(document, "K@40") == {"segment": "K/seg-1", "step": 40}
    got = _read(document, "c05.unregistered-checkpoint")
    assert got.raw == {"answer": "not covered"}
    assert _grade("c05.unregistered-checkpoint", got).status == "right"
    # A trace that holds a checkpoint named E@12 reads an answer, which is wrong.
    held = copy.deepcopy(document)
    _segment(held, "E/seg-2")["checkpoints"].append({"name": "E@12", "step": 12})
    assert checkpoint_record(held, "E@12") == {"segment": "E/seg-2", "step": 12}
    got = _read(held, "c05.unregistered-checkpoint")
    assert set(got.raw) == {"answer"} and got.raw["answer"] != "not covered"
    bad = _grade("c05.unregistered-checkpoint", got)
    assert bad.status == "wrong" and list(bad.evidence) == ["answer"]


# --- A row that records none --------------------------------------------------------------------


def test_a_question_of_a_row_that_records_none_is_cannot_answer_only_after_the_attempt():
    interfaces = need("baselines.interfaces")
    record = need("baselines.record")
    grade = need_attr("baselines.grade", "grade")
    not_gradable = need_attr("baselines.grade", "NotGradable")
    listed = _rows_of_group("c05.f.trace-deleted")[0]
    question = _make_question(listed)
    assert question.route is None and question.missing
    reading = _subject(make_document()).read(question)
    expect = _expect(listed)
    ran = record.StoreAttempt(subject="trace", target=("store", "main"), attempt_id=None,
                              recorded="the export of the faulted store")
    got = grade(question, expect, reading, attempts=[ran])
    assert got.status == interfaces.CANNOT_ANSWER
    assert got.evidence == _check10_rows()["c10.c05.f.trace-deleted.trace"]["missing"]
    with pytest.raises(not_gradable):
        grade(question, expect, reading)
    failed = dataclasses.replace(ran, attempt_id="attempt-7")
    with pytest.raises(not_gradable):
        grade(question, expect, reading, attempts=[failed])


# --- The export ------------------------------------------------------------------------------


def _shown() -> dict:
    """What ``show E/seg-2 --store F --trace`` prints, in the form docs/cli.md gives, with
    the keys and ids the join holds that the trace must not keep added at every level."""
    return {
        "segment": "E/seg-2", "route": "cache files", "cause": None, "cause_detail": None,
        "store": "SRC-LEAK-store", "untraced_steps": [["E/seg-2", 8]],
        "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                      "consumed_at_start": 24, "step_at_start": 6, "first_step": 7,
                      "last_step": 8, "origin": "SRC-LEAK-launch"}],
        "steps": [
            {"step": 7, "consumed_samples": 4, "epochs": [0], "source": "SRC-LEAK-step",
             "samples": [{"sample_pos": 0, "global_index": 24, "rank": 0, "sample_tokens": 33,
                          "covered_tokens": 33, "record": "SRC-LEAK-sample",
                          "rows": [{"shard": "E/shard-3", "shard_position": 2, "ordinal": 7,
                                    "token_start": 0, "token_end": 22, "source": "SRC-LEAK-row",
                                    "record": "T", "occurrence": "SRC-LEAK-occurrence",
                                    "file": "SRC-LEAK-file", "text": "SRC-LEAK-text",
                                    "fingerprint": "SRC-LEAK-fingerprint",
                                    "artifact": "SRC-LEAK-artifact", "member": "SRC-LEAK-member"}]}]},
            {"step": 8, "consumed_samples": 4,
             "samples": [{"sample_pos": 3, "global_index": 31, "rank": 0, "sample_tokens": 33,
                          "covered_tokens": 16,
                          "rows": [{"shard": "E/shard-3", "shard_position": 2, "ordinal": 2,
                                    "token_start": 19, "token_end": 35}]}]},
        ],
    }


def _run_record(segment: str = "E/seg-2") -> dict:
    return {"format": "waterwhite.run-record/1", "run": "E", "segment": segment,
            "start_kind": "resume", "parent_checkpoint": "E@6",
            "settings": {"dataloader_type": "single"},
            "shards": [
                {"position": 2, "prefix": "shards/shard-3", "bin_sha256": "00", "idx_sha256": "00"},
                {"position": 0, "prefix": "shards/shard-1", "bin_sha256": "00", "idx_sha256": "00"},
                {"position": 1, "prefix": "shards/shard-2", "bin_sha256": "00", "idx_sha256": "00"}],
            "cache_digests": [], "shuffle_digests": []}


def _step_record(segment: str = "E/seg-2") -> dict:
    return {"segment": segment,
            "steps": [{"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}],
            "checkpoints": [{"name": "E@11", "step": 11}, {"name": "E@9", "step": 9}]}


def _parent_shown() -> dict:
    return {"artifact": "E@6", "registered": True, "kind": "checkpoint",
            "segment": "E/seg-1", "at_step": 6, "started_segments": ["E/seg-2"]}


# Every key a segment record of the trace may hold at any level.
EXPORT_KEYS = {
    "segment", "run", "parent_checkpoint", "shards", "checkpoints", "launches", "steps", "name",
    "step", "position", "world_size", "data_parallel_size", "consumed_at_start",
    "step_at_start", "first_step", "last_step", "consumed_samples", "samples", "sample_pos",
    "rank", "sample_tokens", "covered_tokens", "rows", "shard", "shard_position", "ordinal",
    "token_start", "token_end",
}


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_the_export_keeps_what_the_trace_holds_and_nothing_a_source_could_be_read_from():
    export_segment = need_attr(MODULE, "export_segment")
    shown, run_record, step_record, parent = _shown(), _run_record(), _step_record(), _parent_shown()
    before = copy.deepcopy((shown, run_record, step_record, parent))
    got = export_segment(shown, run_record, step_record, parent)
    assert (shown, run_record, step_record, parent) == before
    assert set(got) == {"segment", "run", "parent_checkpoint", "shards", "checkpoints",
                        "launches", "steps"}
    assert got["segment"] == "E/seg-2" and got["run"] == "E" and got["parent_checkpoint"] == "E@6"
    # The shards in the trainer's order (by position, not by the order the record lists them),
    # the checkpoints in step order, the launch with its seven keys.
    assert got["shards"] == ["shard-1", "shard-2", "shard-3"]
    assert got["checkpoints"] == [{"name": "E@9", "step": 9}, {"name": "E@11", "step": 11}]
    assert got["launches"] == [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                                "consumed_at_start": 24, "step_at_start": 6, "first_step": 7,
                                "last_step": 8}]
    assert [s["step"] for s in got["steps"]] == [7, 8]
    assert set(got["steps"][0]) == {"step", "consumed_samples", "samples"}
    assert set(got["steps"][0]["samples"][0]) == {"sample_pos", "rank", "sample_tokens",
                                                  "covered_tokens", "rows"}
    assert got["steps"][0]["samples"][0]["rows"] == [
        {"shard": "E/shard-3", "shard_position": 2, "ordinal": 7, "token_start": 0,
         "token_end": 22}]
    assert got["steps"][1]["samples"][0]["covered_tokens"] == 16
    # Nothing the join holds beyond positions: no key and no value of the extra ones.
    assert set(_walk_keys(got)) <= EXPORT_KEYS
    assert "SRC-LEAK" not in json.dumps(got)
    assert "untraced_steps" not in got
    # A segment with no parent needs no parent shown.
    first = _run_record("E/seg-1")
    first["parent_checkpoint"] = None
    plain = export_segment({**_shown(), "segment": "E/seg-1"}, first, _step_record("E/seg-1"))
    assert plain["parent_checkpoint"] is None


def test_the_export_is_the_form_the_fault_applicator_and_the_history_reader_take():
    export_segment = need_attr(MODULE, "export_segment")
    history_rows = need_attr("baselines.history", "history_rows")
    apply_fault = need_attr("baselines.faults", "apply_fault")
    got = export_segment(_shown(), _run_record(), _step_record(), _parent_shown())
    # E@9 is saved at step 9 of this segment; its parent E@6 is not in this document.
    rows = history_rows({"segments": [got]}, "E@9")
    assert [(r.segment, r.step, r.ordinal) for r in rows] == [
        ("E/seg-2", 7, 7), ("E/seg-2", 8, 2)]
    # The first segment of run E, exported from what the tool prints for it, takes the faults
    # that are put to the trace.
    first = _segment(make_document(), "E/seg-1")
    printed = {"segment": "E/seg-1", "route": "cache files", "cause": None,
               "cause_detail": None, "untraced_steps": [],
               "launches": copy.deepcopy(first["launches"]),
               "steps": [{**st, "samples": [{**sm, "global_index": 0} for sm in st["samples"]]}
                         for st in first["steps"]]}
    run_record = {**_run_record("E/seg-1"), "parent_checkpoint": None,
                  "shards": [{"position": 0, "prefix": "shards/shard-1"},
                             {"position": 1, "prefix": "shards/shard-2"}]}
    step_record = {"segment": "E/seg-1", "steps": [], "checkpoints": [{"name": "E@6", "step": 6}]}
    exported = export_segment(printed, run_record, step_record)
    assert exported == first
    deleted = apply_fault("f.trace-deleted", "trace", exported)
    assert [s["step"] for s in deleted["steps"]] == [1, 2, 4, 5, 6]
    half = apply_fault("f.trace-half", "trace", exported)
    assert _sample(_one_segment(half), "E/seg-1", 4, 3)["covered_tokens"] == 16


def test_an_export_of_documents_that_do_not_belong_together_is_refused():
    export_segment = need_attr(MODULE, "export_segment")
    export_error = need_attr(MODULE, "ExportError")
    assert issubclass(export_error, ValueError)
    with pytest.raises(export_error):
        export_segment(_shown(), _run_record("E/seg-1"), _step_record())
    with pytest.raises(export_error):
        export_segment(_shown(), _run_record(), _step_record("E/seg-1"))
    twice = _run_record()
    twice["shards"][1]["position"] = 2
    with pytest.raises(export_error):
        export_segment(_shown(), twice, _step_record(), _parent_shown())
    # The parent that show prints: another name, a name the store does not hold as a
    # checkpoint, and a checkpoint the segment was not started from.
    other = {**_parent_shown(), "artifact": "E@9"}
    unregistered = {"artifact": "E@6", "registered": False, "reason": "not registered",
                    "named_by": []}
    not_started = {**_parent_shown(), "started_segments": ["E/seg-9"]}
    for bad in (other, unregistered, not_started):
        with pytest.raises(export_error):
            export_segment(_shown(), _run_record(), _step_record(), bad)


def test_the_one_documented_refusal_is_an_answer_and_every_other_failure_is_a_failed_attempt():
    classify_show = need_attr(MODULE, "classify_show")
    refusal = need_attr(MODULE, "Refusal")
    attempt_type = need_attr("baselines.interfaces", "Attempt")
    record = need("baselines.record")
    command = ("waterwhite", "show", "E/seg-9", "--store", LOCAL_PATH + "/store.sqlite", "--trace")
    shown = json.dumps(_shown())

    def classify(exit_code, output, segment="E/seg-2"):
        return classify_show(segment, command, exit_code, output, "attempt-3")

    # Exit 0 with the document of the segment asked: the document.
    assert classify(0, shown) == json.loads(shown)
    # Exit 2 with the documented words: the subject's answer that the store holds no run.
    refused = classify(2, "request refused: no run segment 'E/seg-2' is registered", "E/seg-2")
    assert isinstance(refused, refusal) and not isinstance(refused, attempt_type)
    assert refused.exit_code == 2 and "request refused" in refused.output
    # Anything else is a failed attempt: another exit, the words under another exit, no exit,
    # no JSON, JSON that is no object, the document of another segment.
    failures = [
        (1, "usage: waterwhite show"),
        (3, "request refused"),
        (2, "internal error"),
        (None, ""),
        (0, "not json"),
        (0, "[1, 2]"),
        (0, json.dumps({**_shown(), "segment": "E/seg-1"})),
    ]
    for exit_code, output in failures:
        got = classify(exit_code, output)
        assert isinstance(got, attempt_type), (exit_code, output)
        assert (got.id, got.exit_code) == ("attempt-3", exit_code)
    # The failure the writer would refuse is made fit to write: no local path is left in the
    # command or the output, and what the output said is kept.
    trace = ("Traceback (most recent call last):\n  File \"" + LOCAL_PATH
             + "/src/waterwhite/cli.py\", line 9, in main\nValueError: boom")
    got = classify(1, trace)
    assert "ValueError: boom" in got.output
    record.refuse_private(got.output, "the output")
    record.refuse_private(list(got.command), "the command")
    assert got.command[:3] == command[:3] and "--trace" in got.command
    refused_with_path = classify(2, "request refused " + LOCAL_PATH + "/store.sqlite")
    assert isinstance(refused_with_path, refusal)
    record.refuse_private(refused_with_path.output, "the refusal")


def test_the_module_imports_nothing_of_the_tool_and_holds_no_private_text():
    module = need(MODULE)
    source = Path(module.__file__).read_text(encoding="utf-8")
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, roots & FORBIDDEN_IMPORT_ROOTS
    for text in FORBIDDEN_TEXT:
        assert text not in source, text


# --- The one export of a store with a fault, made by the tool itself ---------------------------


def _show(store: Path, *words: str) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO / "src"), str(REPO)]))
    done = subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", "show", *words, "--store", str(store)],
        capture_output=True, text=True, env=env, timeout=180)
    return done.returncode, done.stdout + done.stderr


@pytest.fixture(scope="module")
def registered_run_e(tmp_path_factory):
    """A capture-on fixture build with its shards tokenized, run E made by the replica's run
    wrapper and registered through the tool's command line (one run, two segments)."""
    if not os.environ.get(MEGATRON_VARIABLE):
        pytest.skip(f"{MEGATRON_VARIABLE} is not set")
    from replica.fixture import build_captured_fixture
    from replica.register_runs import register_runs
    from replica.run_plan import load_runs_plan
    from replica.runs import build_runs

    plan = load_runs_plan(SPEC_FILE)
    base = tmp_path_factory.mktemp("trace-export")
    built = build_captured_fixture(
        base / "store", tokenize=True, spec=SPEC_FILE, layouts=("partitioned",), releases=(),
        imports=(), capture_generation=False, parallel=3,
        verified_joins=plan.verified_joins(["E"]))
    runs = base / "runs"
    build_runs(runs, fixture_build=built.out, spec_path=SPEC_FILE, runs=["E"])
    register_runs(built.store, runs, plan, runs=["E"])
    return built, runs, plan


def _delete_the_trace_of_step_three(store: Path) -> None:
    """The fault f.trace-deleted of the table on a store file: the rows, the samples and the
    step trace of step 3 of E/seg-1 are deleted, and the registered step stays."""
    with closing(sqlite3.connect(store)) as db:
        (trace,) = db.execute(
            "SELECT t.trace_id FROM step_trace t JOIN run_segment s USING (segment_id) "
            "WHERE s.name = ? AND t.step = 3", ("E/seg-1",)).fetchone()
        rows = db.execute("DELETE FROM trace_row WHERE trace_id = ?", (trace,)).rowcount
        samples = db.execute("DELETE FROM trace_sample WHERE trace_id = ?", (trace,)).rowcount
        traces = db.execute("DELETE FROM step_trace WHERE trace_id = ?", (trace,)).rowcount
        db.commit()
        assert rows >= 1 and samples >= 1 and traces == 1, (rows, samples, traces)
        kept = db.execute(
            "SELECT ss.step FROM segment_step ss JOIN run_segment s USING (segment_id) "
            "WHERE s.name = ? AND ss.step = 3", ("E/seg-1",)).fetchall()
        assert len(kept) == 1, "the registered step 3 must stay"


@pytest.mark.megatron
@pytest.mark.slow
def test_the_export_of_the_store_with_the_trace_of_step_three_deleted(registered_run_e):
    built, runs, plan = registered_run_e
    spec = _spec()
    # --- What the tool prints, read against docs/cli.md (before the module is asked for) ---
    code, text = _show(built.store, "E/seg-1", "--trace")
    assert code == 0, text
    before = json.loads(text)
    assert before["segment"] == "E/seg-1" and before["route"] != "none", before["route"]
    assert [s["step"] for s in before["steps"]] == [1, 2, 3, 4, 5, 6]
    assert {len(s["samples"]) for s in before["steps"]} == {spec["runs"]["E"]["global_batch_size"]}
    # The documented refusal for a name that is no registered run segment.
    code, text = _show(built.store, "Q/seg-9", "--trace")
    assert code == 2 and "request refused" in text, (code, text)
    # The fault on a scratch copy of the whole store folder.
    scratch = built.store.parent.parent / "faulted"
    shutil.copytree(built.store.parent, scratch)
    faulted_store = scratch / built.store.name
    _delete_the_trace_of_step_three(faulted_store)
    code, text = _show(faulted_store, "E/seg-1", "--trace")
    assert code == 0, text
    after = json.loads(text)
    # docs/cli.md: `steps` holds one entry per traced step. A disagreement here is the tool's.
    assert [s["step"] for s in after["steps"]] == [1, 2, 4, 5, 6], (
        "docs/cli.md says steps holds one entry per traced step: the tool prints "
        f"{[s['step'] for s in after['steps']]} for the store with the trace of step 3 deleted")
    code, text = _show(faulted_store, "E/seg-2", "--trace")
    assert code == 0, text
    after_two = json.loads(text)
    code, text = _show(faulted_store, "E@6")
    assert code == 0, text
    parent = json.loads(text)
    assert parent["artifact"] == "E@6" and parent["kind"] == "checkpoint"
    assert parent["segment"] == "E/seg-1" and parent["at_step"] == 6
    assert parent["started_segments"] == ["E/seg-2"]

    # --- The export of both segments (the module is asked for here) ---
    export_segment = need_attr(MODULE, "export_segment")
    untraced_steps = need_attr(MODULE, "untraced_steps")
    classify_show = need_attr(MODULE, "classify_show")
    from replica.runs import segment_paths

    by_name = {s.name: s for s in plan.segments}

    def record(segment, key):
        path = segment_paths(runs, by_name[segment])[key]
        return json.loads(path.read_text(encoding="utf-8"))

    first = export_segment(after, record("E/seg-1", "run_record"), record("E/seg-1", "step_record"))
    second = export_segment(after_two, record("E/seg-2", "run_record"),
                            record("E/seg-2", "step_record"), parent)
    spec_segments = spec["runs"]["E"]["segments"]
    for exported, name in ((first, "E/seg-1"), (second, "E/seg-2")):
        assert exported["segment"] == name and exported["run"] == "E"
        assert exported["shards"] == spec_segments[name]["shards"]
        assert set(_walk_keys(exported)) <= EXPORT_KEYS, name
    assert first["parent_checkpoint"] is None and second["parent_checkpoint"] == "E@6"
    assert first["checkpoints"] == [{"name": "E@6", "step": 6}]
    assert second["checkpoints"] == [{"name": "E@9", "step": 9}]
    assert [s["step"] for s in first["steps"]] == [1, 2, 4, 5, 6]
    assert first["launches"][0]["first_step"] == 1 and first["launches"][0]["last_step"] == 6
    document = {"segments": [first, second]}
    # The deleted step is read as untraced; so is the form with an entry that holds no sample.
    assert _pairs(untraced_steps(document, "E@6")) == [("E/seg-1", 3)]
    as_entry = copy.deepcopy(document)
    as_entry["segments"][0]["steps"].insert(2, {"step": 3, "consumed_samples": 4, "samples": []})
    assert _pairs(untraced_steps(as_entry, "E@6")) == [("E/seg-1", 3)]
    # The subject reads the exported trace: the counted step reads the table's value, and step 8
    # and the deleted step 3 are the untraced ones. Nothing is graded here; the readings are
    # looked at raw.
    subject = need_attr(MODULE, "TraceSubject")(document, place_of=place_of)
    assert subject.read(_make_question("c01.E.step8.counted")).raw == {"consumed_samples": 4}
    assert _pairs(subject.read(_make_question("c01.E.step8.untraced")).raw["untraced_steps"]) == [
        ("E/seg-1", 3), ("E/seg-2", 8)]
    # The export of an unregistered segment is the documented refusal, an answer.
    code, text = _show(faulted_store, "Q/seg-9", "--trace")
    got = classify_show("Q/seg-9", ("waterwhite", "show", "Q/seg-9", "--trace"), code, text, "a-1")
    assert not isinstance(got, dict) and got.exit_code == 2
