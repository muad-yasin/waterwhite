"""The Dolma toolkit beside the trace as a subject of Check 10 (plan 5.5 Check 10; table README,
section "Check 10": the subject ``dolma+trace``, "The Dolma tie to an ordinal"; the
``c10.<group>.dolma+trace`` rows of ``check10.toml``).

Every input is written by hand: a build folder of chunk files and record files, the companion
rows the tokenizer would have written (in an order unlike the input's), a tie record, a trace
document in the form ``baselines.trace.export_segment`` returns, and stub run records. Nothing is
started: no bubblewrap, no Dolma, no environment's python, no tool.

How the combination reads, by the rows of the table:

- The trace half alone (``c01.positions``, ``c01.step-facts``, ``c05.unregistered-checkpoint``):
  the rows read as the trace's own row of the group reads them; no run of the toolkit and no tie
  is needed and none is used. A whole-run row holds ``rows`` only.
- The toolkit half alone (the set rows of Check 3, ``c04.interrupted-write``, ``c04.count-499``,
  ``c06.documents-mapped``, the shard rows of ``c05.unmutated.shards``, ``c05.withdraw-C.unmutated``,
  ``c09.artifact-actions`` and ``c09.successors``): read as the toolkit alone reads them.
- A checkpoint, by the checkpoint rule of the rows: the history is the checkpoint's segment up
  to its step and then the history of its parent checkpoint; each trace row is tied to the
  companion row of its document (the src is the chunk file of the shard, the loc is the ordinal
  plus 1), and the tied row's id is resolved to the file that holds it. ``contains``: a tied
  row's id resolves to the file of a member the request names (or is the named record).
  ``does not contain``: the history has trace rows, every tied row's id resolves to a source file
  and none is a member. ``not covered`` (no record): the trace does not hold the checkpoint or
  holds no row for any step of its history. ``cannot_answer``: no tied row is a member and a tied
  row's id resolves to a generation output or to no file. A step with no trace row, an uncovered
  range and a trace row with no companion row add nothing to the reading.
- A shard of a trace row is the artifact ``E/<name>`` (the shards of the table's runs) unless the
  name already holds a folder (``E-rr/shard-1``).
- A tie that is not shown for a chunk file the history reads is cannot answer with the row's own
  evidence. An answer that used the tie says so (``used_tie``); the record's writer then adds the
  mark of the tie, and every row of the checkpoint rule carries the mark of the step-to-ordinal
  part.
- The faults of Checks 4 and 5 that belong to the trace are put to the exported record with
  ``baselines.faults.apply_fault``; the faults that belong to the companion rows are put to a
  copy of the rows. Neither half has a check that reports them: the answer is read from what is
  left, never a stop.
- A kind of artifact neither half holds a record for (a source file, a generation output, a
  release, an import, a mixture, the listing) is cannot answer with the missing text of the
  combination's own row of that kind, only after a successful run of the toolkit on the store.
- A failed run, or a failed export, is that same ``Attempt`` for every question that needs the
  half that failed; it is never a reading and never cannot answer. A store with no run of the
  toolkit is not attempted (``NotAttempted``, a ``ValueError``). A layout with no run is not
  attempted, never cannot answer.

The names asked of ``baselines.combo_dolma``:

- ``DolmaTraceSubject(runs, *, trace, tie_record, requests, place_of, split_pairs=(),
  state_traces=None, layout_runs=None)``: a subject (``name`` is ``dolma+trace``). ``runs`` maps
  the target of a question (``baselines.record.attempt_target``) to a ``baselines.dolma.DolmaRun``
  or to the failed ``Attempt`` of that run; ``tie_record``, ``requests`` and ``split_pairs`` are
  those of ``baselines.dolma.DolmaSubject``. ``trace`` is the trace document, ``{"segments": [...]}``
  in the form of ``baselines.trace.export_segment`` (a ``baselines.trace.Refusal`` reads as a
  document with no segment), or the failed ``Attempt`` of the export. ``place_of(listed_row)``
  gives the places a row of the trace half reads (as for ``baselines.trace.TraceSubject``).
  ``state_traces`` maps a starting state of Check 2 (``c2.i`` to ``c2.iv``) to the trace written
  from it; a state it does not hold reads the main trace. ``layout_runs`` maps ``(target,
  layout)`` to the run, or the failed ``Attempt``, of one layout of a set.
  ``read(question)`` gives a ``Reading`` whose ``raw`` holds exactly the values the row's rule
  names (or ``{"cannot_answer": <evidence>}``) and whose route is the question's, or
  ``Reading(question.id, None, None)`` for a question whose row records none; a failed half gives
  its ``Attempt``. ``read_by_layout(question)`` gives, for a row that names layouts, one entry per
  layout: the reading, the failed ``Attempt``, or None where the layout has no run.
  ``used_tie(question)`` says whether the last answer to the question used the tie.
- ``NotAttempted``: a ``ValueError``, raised for a question whose store has no run of the toolkit.
- ``faulted_trace(document, state)``: a changed copy of the trace document with the fault of
  ``state`` put to the segment ``E/seg-1`` of it through ``baselines.faults.apply_fault``
  (``f.trace-deleted``, ``f.trace-half``, ``l.trace-count``); any other state gives an equal
  copy. The document it is given is never changed.
- The readings of the shapes below: a checkpoint ``{"answer": word}``; step 8 of E/seg-2 ``{"rows_that_name_T":
  n, "first_exposure": {"segment", "step"} or None}``; the restart ``{"exposure_free_restart":
  checkpoint or None}``.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import copy
import importlib
import json
import re
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
MODULE = "baselines.combo_dolma"
SUBJECT = "dolma+trace"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")
PRIVATE_PATH = "/ho" + "me/someone/work"

CANNOT = "cannot_answer"
SAMPLE_TOKENS = 33
CUT_COVERED = 16
EMPTY = {"segments": []}

# The end of a stopped writer that was stopped (killed by signal 9), as a runner records it.
STOP_ID = "attempt-stop"
STOP_COMMAND = (PRIVATE_PATH + "/venv-dolma/bin/python", "-m", "dolma.cli", "tokens",
                "--documents", "shard.jsonl")
LANDED_STOP = {"stopped_exit_code": -9, "stopped_signal": 9, "stopped_attempt_id": STOP_ID,
               "stopped_command": STOP_COMMAND, "stopped_output": "", "rows_showed_while_running": True}
NOT_LANDED_EXITS = (0, 1, 127, -11, 243)
RETRY_ID = "attempt-retry"

# What a request names: the file of the member and, for a record-level request, the record.
REQUESTS = {
    "withdraw A": {"files": ["sources/A.jsonl"]},
    "withdraw A, record X@A": {"files": ["sources/A.jsonl"], "record": "A-001"},
    "withdraw C": {"files": ["sources/C.jsonl"]},
    "withdraw W": {"files": ["sources/Z.jsonl"], "record": "Z-089"},
}

# The listed rows whose reading the table leaves open (the join of several shards).
OPEN_ROWS = frozenset({"c06.generated.downstream-of-translation", "c09.10k.successor-shards"})
# The number of questions of a subject and how many of this subject's have a route: 391 in all;
# by the groups of check10.toml (the listed rows of each group whose combination row has a
# route, counted once for each size they are asked at), 211 with a route.
QUESTIONS_OF_THE_SUBJECT = 391
QUESTIONS_WITH_A_ROUTE = 211


# --- The table, read from its files ---------------------------------------------------


def _mod(name: str):
    return importlib.import_module(name)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _check10() -> dict:
    return _read_toml(TABLE_DIR / "check10.toml")


@lru_cache(maxsize=1)
def _check10_rows() -> dict[str, dict]:
    return {row["id"]: row for row in _check10()["row"]}


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, str]:
    return {listed: group["id"] for group in _check10()["group"] for listed in group["rows"]}


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows


@lru_cache(maxsize=1)
def _spec() -> dict:
    return _read_toml(SPEC_FILE)


def _fill(value):
    """A value of the table with each ``pending:`` word replaced by the fixture spec's own."""
    if isinstance(value, str) and value.startswith("pending:"):
        found = _spec()
        for part in value[len("pending:"):].split("."):
            found = found[part]
        return found
    if isinstance(value, dict):
        return {k: _fill(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v) for v in value]
    return value


def listed_expect(listed: str) -> dict:
    """The expectation of a listed row as the table writes it, its pending values filled from
    the fixture spec."""
    return _fill(dict(_listed_rows()[listed]["expect"]))


def _missing_of(group: str, subject: str = SUBJECT) -> str:
    """The record a subject lacks, as the table writes it for the group's row of the subject."""
    return _check10_rows()[f"c10.{group}.{subject}"]["missing"]


def _not_shown_evidence(q) -> str:
    """The evidence of the question's own row for a tie that is not shown."""
    return _check10_rows()[q.row_id]["ordinal"]["not_shown"]["evidence"]


def _tie_mark() -> str:
    return _check10_rows()["c10.c03.spans.dolma"]["ordinal"]["shown"]["mark"]


@lru_cache(maxsize=1)
def _questions() -> dict:
    table10 = _mod("baselines.table10")
    return {q.id: q for q in table10.expand(table10.load_table10())}


def question(listed: str, size: str | None = None):
    """The question that puts a listed row to the combination at a size."""
    table10 = _mod("baselines.table10")
    return _questions()[table10.question_id(listed, SUBJECT, size)]


def target_of(q):
    return _mod("baselines.record").attempt_target(q)


def routed_questions() -> list:
    return [q for q in _questions().values() if q.subject == SUBJECT and q.route is not None]


# --- The places a row of the trace half reads --------------------------------------------

PLACE_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal")


def _spec_lines(*keys: str) -> list[dict]:
    return [json.loads(text) for key in keys for text in _spec()["positions"][key]["log_lines"]]


def place_of(listed_row: str) -> list[dict]:
    """The places a listed row reads, from the table and the spec."""
    row = _listed_rows()[listed_row]
    if _group_of_listed()[listed_row] == "c01.positions":
        rows = row["expect"].get("rows")
        if isinstance(rows, str) and rows.startswith("pending:positions."):
            lines = _spec_lines(rows.split(".", 1)[1])
            return [{key: line[key] for key in PLACE_KEYS} for line in lines]
        return [{"run": row["given"]["run"]}]
    return [dict(row["given"])]


# --- A hand-written trace ----------------------------------------------------------------


def launch(position, consumed, step_at_start, first, last, world=1):
    return {"position": position, "world_size": world, "data_parallel_size": world,
            "consumed_at_start": consumed, "step_at_start": step_at_start,
            "first_step": first, "last_step": last}


def seg(name, run, parent, shards, checkpoints, launches, steps):
    """One segment record in the form the export gives. ``steps`` maps a step to its samples,
    each ``(sample_pos, rank, covered_tokens, [(shard_position, ordinal, token_start,
    token_end), ...])``."""
    kept = []
    for number in sorted(steps):
        samples = [{"sample_pos": pos, "rank": rank, "sample_tokens": SAMPLE_TOKENS,
                    "covered_tokens": covered,
                    "rows": [{"shard": shards[p], "shard_position": p, "ordinal": o,
                              "token_start": s, "token_end": e} for p, o, s, e in rows]}
                   for pos, rank, covered, rows in steps[number]]
        kept.append({"step": number, "consumed_samples": len(samples), "samples": samples})
    return {"segment": name, "run": run, "parent_checkpoint": parent, "shards": list(shards),
            "checkpoints": [{"name": n, "step": s} for n, s in checkpoints],
            "launches": list(launches), "steps": kept}


def _filler(first, last, positions, modulo):
    steps = {}
    for step in range(first, last + 1):
        steps[step] = [(q, 0, SAMPLE_TOKENS,
                        [(positions[q % len(positions)], (2 * step + q) % modulo, 0, SAMPLE_TOKENS)])
                       for q in range(4)]
    return steps


def _put(steps, step, pos, rank, covered, rows):
    steps[step] = sorted([s for s in steps[step] if s[0] != pos] + [(pos, rank, covered, rows)],
                         key=lambda s: s[0])


def make_trace(over=None) -> dict:
    """The trace of runs E (two segments), D and K. The rows of the places the spec names are the
    spec's own lines: T in the first sample of step 7 of E/seg-2 (a document of shard 3), the cut
    sample of step 8, the shared token at step 9, the shared ordinal at step 4 of E/seg-1 (one
    row, tokens 6 to 39). Every other sample holds one row of a document that is no member of
    withdraw A. ``over`` maps ``(segment, step, sample_pos)`` to ``(rank, covered_tokens, rows)``
    and replaces that sample."""
    e1 = _filler(1, 6, [0, 1], 16)
    _put(e1, 1, 0, 0, SAMPLE_TOKENS, [(1, 17, 21, 50)])
    _put(e1, 1, 1, 0, SAMPLE_TOKENS, [(0, 8, 22, 44)])
    _put(e1, 4, 3, 0, SAMPLE_TOKENS, [(0, 1, 6, 39)])
    e2 = _filler(7, 12, [0, 1], 18)
    _put(e2, 7, 0, 0, SAMPLE_TOKENS, [(2, 24, 32, 39), (2, 7, 0, 22), (2, 26, 0, 4)])
    _put(e2, 8, 3, 0, CUT_COVERED, [(2, 2, 19, 35)])
    _put(e2, 9, 2, 0, SAMPLE_TOKENS, [(2, 6, 19, 51), (2, 18, 0, 1)])
    d1 = _filler(1, 3, [0, 1], 16)
    k1 = {step: [(q, q // 2, SAMPLE_TOKENS, [(1, (step + q) % 16, 0, SAMPLE_TOKENS)]) for q in range(4)]
          for step in range(1, 9)}
    _put(k1, 2, 3, 1, SAMPLE_TOKENS, [(0, 20, 0, SAMPLE_TOKENS)])
    parts = {"E/seg-1": e1, "E/seg-2": e2, "D/seg-1": d1, "K/seg-1": k1}
    for (segment, step, pos), (rank, covered, rows) in (over or {}).items():
        _put(parts[segment], step, pos, rank, covered, rows)
    return {"segments": [
        seg("E/seg-1", "E", None, ["shard-1", "shard-2"], [("E@6", 6)], [launch(0, 0, 0, 1, 6)], e1),
        seg("E/seg-2", "E", "E@6", ["shard-1", "shard-2", "shard-3"], [("E@9", 9)],
            [launch(0, 24, 6, 7, 12)], e2),
        seg("D/seg-1", "D", "E@9", ["shard-1", "shard-2"], [("D@3", 3)], [launch(0, 0, 0, 1, 3)], d1),
        seg("K/seg-1", "K", "E@6", ["shard-3", "shard-2"], [("K@3", 3), ("K@8", 8)],
            [launch(0, 0, 0, 1, 3, 2), launch(1, 12, 3, 4, 8, 2)], k1),
    ]}


def segment_of(document, name):
    (found,) = [s for s in document["segments"] if s["segment"] == name]
    return found


def sample_of(document, name, step, pos):
    (found_step,) = [s for s in segment_of(document, name)["steps"] if s["step"] == step]
    (found,) = [s for s in found_step["samples"] if s["sample_pos"] == pos]
    return found


def without_rows(document) -> dict:
    """The trace of a registration that wrote no trace: the segments and their checkpoints, no
    step."""
    found = copy.deepcopy(document)
    for segment in found["segments"]:
        segment["steps"] = []
    return found


def exported_document() -> dict:
    """The hand-written trace with E/seg-1 as ``baselines.trace.export_segment`` returns it from
    the printed documents."""
    trace = _mod("baselines.trace")
    document = make_trace()
    own = segment_of(document, "E/seg-1")
    shown = {"segment": "E/seg-1", "launches": copy.deepcopy(own["launches"]),
             "steps": copy.deepcopy(own["steps"]), "untraced_steps": []}
    run_record = {"segment": "E/seg-1", "run": "E", "parent_checkpoint": None,
                  "shards": [{"position": 0, "prefix": "shards/shard-1"},
                             {"position": 1, "prefix": "shards/shard-2"}]}
    step_record = {"segment": "E/seg-1", "checkpoints": [{"name": "E@6", "step": 6}]}
    exported = trace.export_segment(shown, run_record, step_record)
    document["segments"] = [exported if s["segment"] == "E/seg-1" else s for s in document["segments"]]
    return document


# --- A hand-written world -----------------------------------------------------------------

ALWAYS = {"A-001": "sources/A.jsonl", "A-002": "sources/A.jsonl", "C-001": "sources/C.jsonl"}


def _record_file(identifier: str) -> str:
    if len(identifier) > 2 and identifier[1] == "-" and identifier[0] in "ABCZ":
        return f"sources/{identifier[0]}.jsonl"
    return f"gen/{identifier}.jsonl"


class World:
    """A build folder with chunk files and record files, the companion rows of every chunk
    file in an order unlike the input's, and the tie record."""

    def __init__(self, root: Path, shards: Mapping[str, list], paths: Mapping[str, str] | None = None):
        self.root = Path(root)
        self.build = self.root / "build"
        self.chunk_files: dict[str, str] = {}
        paths = paths or {}
        holders: dict[str, str] = dict(ALWAYS)
        for name, ids in shards.items():
            relative = paths.get(name) or f"pipeline/partitioned/tokenizer-input/{name.replace('/', '-')}.jsonl"
            self.chunk_files[name] = relative
            target = self.build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for entry in ids:
                identifier, text = entry if isinstance(entry, tuple) else (entry, f"text of {entry}")
                if not identifier.startswith("N-"):
                    # An id of the form N-nnn is held by no record file.
                    holders.setdefault(identifier, _record_file(identifier))
                lines.append(json.dumps({"id": identifier, "text": text}))
            target.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        by_file: dict[str, list[str]] = {}
        for identifier, relative in holders.items():
            by_file.setdefault(relative, []).append(identifier)
        for relative, identifiers in by_file.items():
            target = self.build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "".join(json.dumps({"id": i, "text": f"text of {i}"}) + "\n" for i in identifiers),
                encoding="utf-8")
        self.rows = self.make_rows()
        self.tie_path = self.root / "tie.json"

    @staticmethod
    def size_of(ordinal: int) -> int:
        """The tokens of the document with this ordinal: 3, 4, 5, 3, 4, 5, ..."""
        return 3 + ordinal % 3

    def make_rows(self) -> list:
        row_type = _mod("baselines.companion").Row
        rows = []
        for relative in self.chunk_files.values():
            lines = (self.build / relative).read_text(encoding="utf-8").splitlines()
            position = 0
            for ordinal, line in enumerate(lines):
                size = self.size_of(ordinal)
                rows.append(row_type(start=position, end=position + size,
                                     id=json.loads(line)["id"], src=relative, loc=ordinal + 1))
                position += size
        return rows[::-1]

    def row(self, name: str, loc: int):
        (found,) = [r for r in self.rows if r.src == self.chunk_files[name] and r.loc == loc]
        return found

    def without(self, name: str, loc: int) -> None:
        dropped = self.row(name, loc)
        self.rows = [r for r in self.rows if r is not dropped]

    def write_tie(self, shown: Mapping[str, bool] | bool = True) -> Path:
        tie = _mod("baselines.tie")
        results = []
        for name, relative in self.chunk_files.items():
            held = shown if isinstance(shown, bool) else shown.get(name, True)
            lines = len((self.build / relative).read_text(encoding="utf-8").splitlines())
            results.append({"chunk_file": relative, "shown": held, "documents": lines,
                            "lines": lines, "first_mismatch": None if held else 0})
        tie.write_result(self.tie_path, results, {"note": "hand-written"})
        return self.tie_path

    def run(self, dolma, **more):
        if "stopped_rows" in more or "retry_rows" in more:
            # A run of the two Check 4 states holds the end of its stopped writer: a stop that
            # landed (the exit of a signal), and whether the rows showed while the writer ran.
            more = {**LANDED_STOP, **more}
        return dolma.DolmaRun(build=self.build, rows=tuple(self.rows),
                              chunk_files=dict(self.chunk_files), **more)

    def combo(self, combo, dolma, questions, *, trace, shown: Mapping[str, bool] | bool = True,
              split_pairs=(), state_traces=None, runs=None, layout_runs=None, **more):
        """The subject over one run of the toolkit that every question's store holds, or over the
        ``runs`` given."""
        self.write_tie(shown)
        if runs is None:
            run = self.run(dolma, **more)
            runs = {target_of(q): run for q in questions}
        return combo.DolmaTraceSubject(
            runs, trace=trace, tie_record=self.tie_path, requests=REQUESTS, split_pairs=split_pairs,
            place_of=place_of, state_traces=state_traces, layout_runs=layout_runs)


def _ids(base: int, count: int, overrides: Mapping[int, str]) -> list[str]:
    ids = [f"Z-{base + n}" for n in range(count)]
    for place, identifier in overrides.items():
        ids[place] = identifier
    return ids


# The shards of the table's run E: shard 1 holds T and a member of A late in its file, shard 2 one
# member of A at its last line, shard 3 holds T at ordinal 7, the shared-token record Z-089 at 18
# and a member of A at 20 (the places the spec's lines read).
E1 = _ids(600, 20, {18: "T", 19: "A-001"})
E2 = _ids(700, 20, {19: "A-002"})
E3 = _ids(800, 27, {2: "Z-080", 6: "Z-070", 7: "T", 18: "Z-089", 20: "A-001", 23: "Z-085",
                    24: "Z-088", 26: "Z-067"})
SCRATCH_IDS = [f"Z-{n:03d}" for n in range(1, 501)]
SCRATCH_TOKENS = 1999  # 500 documents of 3, 4, 5, 3, 4, 5, ... tokens
UNIVERSAL = {
    "E/shard-1": E1,
    "E/shard-2": E2,
    "E/shard-3": E3,
    "E-rr/shard-1": ["B-001", "Z-901"],
    "E-rr/shard-2": ["B-002", "Z-902"],
    "E-rr/shard-3": ["T", "Z-903"],
    "c04/shard-500": SCRATCH_IDS,
    "successor of E/shard-3": ["Z-904"],
    "shard-1": ["Z-910", "Z-911"],
    "shard-2": ["Z-912"],
}
UNIVERSAL_PATHS = {"E-rr/shard-1": "pipeline/round-robin/tokenizer-input/rr-1.jsonl",
                   "E-rr/shard-2": "pipeline/round-robin/tokenizer-input/rr-2.jsonl",
                   "E-rr/shard-3": "pipeline/round-robin/tokenizer-input/rr-3.jsonl"}
UNIVERSAL_PAIRS = [{"pair": "pair-1", "members": [{"shard": "shard-1", "ordinal": 0},
                                                  {"shard": "shard-2", "ordinal": 0}]}]


def universal_world(tmp_path) -> World:
    return World(tmp_path, UNIVERSAL, UNIVERSAL_PATHS)


def scratch_rows(world, count):
    held = sorted((r for r in world.rows if r.src == world.chunk_files["c04/shard-500"]),
                  key=lambda r: r.loc)
    return held[:count]


def read_in(world, combo, dolma, listed, *, trace, size=None, **more):
    """The question of a listed row and its reading on a subject built over the world."""
    q = question(listed, size)
    subject = world.combo(combo, dolma, [q], trace=trace, **more)
    return q, subject.read(q), subject


def tiny_world(tmp_path, ids, steps, *, checkpoint="E@6", covered=SAMPLE_TOKENS):
    """One shard of ``ids`` and a trace of one segment whose steps each hold one sample reading
    the ordinals of ``steps`` (one list per step) of that shard."""
    world = World(tmp_path, {"E/shard-1": ids})
    kept = {n + 1: [(0, 0, covered, [(0, o, 0, SAMPLE_TOKENS) for o in rows])]
            for n, rows in enumerate(steps)}
    trace = {"segments": [seg("E/seg-1", "E", None, ["shard-1"], [(checkpoint, max(len(steps), 1))],
                              [launch(0, 0, 0, 1, len(steps))], kept)]}
    return world, trace


# --- Names -----------------------------------------------------------------------------


def test_names_of_the_combination(need_attribute):
    for name in ("DolmaTraceSubject", "NotAttempted", "faulted_trace"):
        need_attribute(MODULE, name)


def test_the_subject_is_named_dolma_plus_trace_and_is_a_subject(need_module, tmp_path):
    combo = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    subject = combo.DolmaTraceSubject({}, trace=EMPTY, tie_record=tmp_path / "tie.json",
                                      requests=REQUESTS, place_of=place_of)
    assert subject.name == SUBJECT
    assert isinstance(subject, interfaces.Subject)
    assert issubclass(combo.NotAttempted, ValueError)


# --- The faults of the trace half, put to the exported record ---------------------------------


def test_the_half_fault_cuts_the_rows_of_the_exported_record_at_16_tokens(need_module):
    combo = need_module(MODULE)
    faults = _mod("baselines.faults")
    document = exported_document()
    before = copy.deepcopy(document)
    cut = combo.faulted_trace(document, "f.trace-half")
    assert document == before, "the document it is given is never changed"
    # The sample at position 3 of step 4 of E/seg-1 holds one row, tokens 6 to 39 of its document;
    # the cut is 16 tokens of the sample, so the row ends at 6 + 16.
    assert sample_of(before, "E/seg-1", 4, 3)["rows"] == [
        {"shard": "shard-1", "shard_position": 0, "ordinal": 1, "token_start": 6, "token_end": 39}]
    sample = sample_of(cut, "E/seg-1", 4, 3)
    assert sample["rows"] == [
        {"shard": "shard-1", "shard_position": 0, "ordinal": 1, "token_start": 6, "token_end": 22}]
    assert sample["covered_tokens"] == 16 and sample["sample_tokens"] == 33
    # Every other sample of the segment, and every other segment, is as it was.
    for step in segment_of(before, "E/seg-1")["steps"]:
        for other in step["samples"]:
            if (step["step"], other["sample_pos"]) != (4, 3):
                assert other == sample_of(cut, "E/seg-1", step["step"], other["sample_pos"])
    assert [s for s in cut["segments"] if s["segment"] != "E/seg-1"] == [
        s for s in before["segments"] if s["segment"] != "E/seg-1"]
    # It is the applicator's own change to the exported segment.
    assert segment_of(cut, "E/seg-1") == faults.apply_fault(
        "f.trace-half", "trace", segment_of(before, "E/seg-1"))


def test_the_half_fault_removes_the_rows_that_start_after_the_cut_and_ends_the_row_that_holds_the_17th_token(
        need_module):
    combo = need_module(MODULE)
    three = {("E/seg-1", 4, 3): (0, SAMPLE_TOKENS, [(0, 1, 0, 10), (0, 2, 0, 10), (1, 19, 0, 13)])}
    cut = combo.faulted_trace(make_trace(three), "f.trace-half")
    rows = sample_of(cut, "E/seg-1", 4, 3)["rows"]
    assert [(r["shard_position"], r["ordinal"], r["token_start"], r["token_end"]) for r in rows] == [
        (0, 1, 0, 10), (0, 2, 0, 6)]


def test_the_deleted_fault_takes_step_3_out_of_the_steps_and_the_count_fault_one_sample_out_of_step_5(
        need_module):
    combo = need_module(MODULE)
    document = exported_document()
    before = copy.deepcopy(document)
    deleted = combo.faulted_trace(document, "f.trace-deleted")
    assert [s["step"] for s in segment_of(deleted, "E/seg-1")["steps"]] == [1, 2, 4, 5, 6]
    assert [s["step"] for s in segment_of(before, "E/seg-1")["steps"]] == [1, 2, 3, 4, 5, 6]
    counted = combo.faulted_trace(document, "l.trace-count")
    (step,) = [s for s in segment_of(counted, "E/seg-1")["steps"] if s["step"] == 5]
    assert [s["sample_pos"] for s in step["samples"]] == [0, 1, 2] and step["consumed_samples"] == 3
    assert document == before


@pytest.mark.parametrize("state", ["unmutated", "f.map-deleted", "f.digest", "c2.ii", "q.unregistered",
                                   "l.count-499", None])
def test_a_state_that_is_no_fault_of_the_trace_gives_an_equal_copy(need_module, state):
    combo = need_module(MODULE)
    document = make_trace()
    got = combo.faulted_trace(document, state)
    assert got == document and got is not document


# --- The trace half alone ---------------------------------------------------------------------


def trace_only(combo, tmp_path, trace):
    return combo.DolmaTraceSubject({}, trace=trace, tie_record=tmp_path / "tie.json", requests=REQUESTS,
                                   place_of=place_of)


def test_the_positions_of_the_oracle_lines_are_the_trace_rows_read_with_no_run_and_no_tie(
        need_module, tmp_path):
    combo = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    grade = _mod("baselines.grade")
    q = question("c01.E.T")
    subject = trace_only(combo, tmp_path, make_trace())
    reading = subject.read(q)
    assert isinstance(reading, interfaces.Reading) and reading.question_id == q.id
    assert reading.route == q.route and set(reading.raw) == {"rows"}
    assert [(r["step"], r["shard_position"], r["ordinal"], r["token_start"], r["token_end"])
            for r in reading.raw["rows"]] == [(7, 2, 24, 32, 39), (7, 2, 7, 0, 22), (7, 2, 26, 0, 4)]
    assert grade.grade(q, listed_expect("c01.E.T"), reading).status == "right"
    assert subject.used_tie(q) is False
    changed = make_trace({("E/seg-2", 7, 0): (0, SAMPLE_TOKENS, [(2, 24, 32, 39), (2, 7, 0, 23), (2, 26, 0, 4)])})
    got = grade.grade(q, listed_expect("c01.E.T"), trace_only(combo, tmp_path, changed).read(q))
    assert got.status == "wrong" and list(got.evidence) == ["rows"]
    missing = make_trace({("E/seg-2", 7, 0): (0, SAMPLE_TOKENS, [(2, 24, 32, 39), (2, 26, 0, 4)])})
    got = grade.grade(q, listed_expect("c01.E.T"), trace_only(combo, tmp_path, missing).read(q))
    assert got.status == "wrong" and list(got.evidence) == ["rows"]


def test_a_whole_run_row_holds_the_rows_of_the_run_and_no_diff(need_module, tmp_path):
    combo = need_module(MODULE)
    document = make_trace()
    q = question("c01.E.diff")
    reading = trace_only(combo, tmp_path, document).read(q)
    assert set(reading.raw) == {"rows"}
    held = sum(len(sample["rows"]) for s in document["segments"] if s["run"] == "E"
               for step in s["steps"] for sample in step["samples"])
    assert len(reading.raw["rows"]) == held


def test_the_step_facts_are_read_from_the_trace(need_module, tmp_path):
    combo = need_module(MODULE)
    grade = _mod("baselines.grade")
    subject = trace_only(combo, tmp_path, make_trace())
    counted = question("c01.E.step8.counted")
    reading = subject.read(counted)
    assert dict(reading.raw) == {"consumed_samples": 4}
    assert grade.grade(counted, listed_expect("c01.E.step8.counted"), reading).status == "right"
    uncovered = subject.read(question("c01.E.step8.uncovered"))
    assert dict(uncovered.raw) == {"uncovered_in_sample": [16, 33], "sample_tokens": 33, "covered_tokens": 16}
    untraced = subject.read(question("c01.E.step8.untraced"))
    assert dict(untraced.raw) == {"untraced_steps": [["E/seg-2", 8]]}
    ranks = question("c01.K.ranks")
    reading = subject.read(ranks)
    assert dict(reading.raw) == {"samples_per_step": 4, "samples_per_rank": 2, "ranks": [0, 1],
                                 "sample_pos_of_rank_0": [0, 1], "sample_pos_of_rank_1": [2, 3]}
    assert grade.grade(ranks, listed_expect("c01.K.ranks"), reading).status == "right"
    short = make_trace({("E/seg-2", 8, 3): (0, SAMPLE_TOKENS, [(2, 2, 19, 35)])})
    plain = trace_only(combo, tmp_path, short).read(question("c01.E.step8.untraced"))
    assert dict(plain.raw) == {"untraced_steps": []}


def test_a_checkpoint_the_trace_does_not_hold_is_no_record_which_is_the_answer_not_covered(
        need_module, tmp_path):
    combo = need_module(MODULE)
    grade = _mod("baselines.grade")
    q = question("c05.unregistered-checkpoint")
    subject = trace_only(combo, tmp_path, make_trace())
    reading = subject.read(q)
    assert dict(reading.raw) == {"answer": "not covered"} and reading.route == q.route
    assert grade.grade(q, listed_expect(q.listed_row), reading).status == "right"
    assert subject.used_tie(q) is False
    held = make_trace()
    segment_of(held, "E/seg-2")["checkpoints"].append({"name": "E@12", "step": 12})
    got = trace_only(combo, tmp_path, held).read(q)
    assert got.raw["answer"] != "not covered"
    assert grade.grade(q, listed_expect(q.listed_row), got).status == "wrong"


# --- The checkpoint rule ---------------------------------------------------------------------

CHECKPOINT_CASES = [
    # (the ids of the lines of the shard, the ordinals each step reads, what the rule reads
    # under withdraw A)
    (["Z-001", "A-002", "Z-002"], [[0], [1]], {"answer": "contains"}),
    (["Z-001", "Z-002"], [[0], [1]], {"answer": "does not contain"}),
    (["Z-001", "Z-002"], [[0], [1, 5]], {"answer": "does not contain"}),   # ordinal 5 has no row
    (["Z-001", "T", "Z-002"], [[0], [1]], ("T", "gen/T.jsonl")),           # a generation output
    (["Z-001", "N-404"], [[0], [1]], ("N-404",)),                           # no file holds the id
    (["T", "A-001", "Z-001"], [[0, 1], [2]], {"answer": "contains"}),      # a member wins
    (["Z-001", "Z-002"], [], {"answer": "not covered"}),                    # no row for any step
]


@pytest.mark.parametrize("ids, steps, wanted", CHECKPOINT_CASES,
                         ids=[str(n) for n in range(len(CHECKPOINT_CASES))])
def test_the_checkpoint_rule_gives_the_four_readings_of_the_table(need_module, tmp_path, ids, steps, wanted):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world, trace = tiny_world(tmp_path, ids, steps)
    q, reading, _ = read_in(world, combo, dolma, "c05.unmutated.E@6", trace=trace)
    assert reading.route == q.route
    if isinstance(wanted, dict):
        assert dict(reading.raw) == wanted
    else:
        assert set(reading.raw) == {CANNOT}
        for word in wanted:
            assert word in reading.raw[CANNOT], word


def test_the_checkpoint_the_trace_does_not_hold_is_not_covered(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world, trace = tiny_world(tmp_path, ["Z-001", "A-002"], [[0], [1]], checkpoint="E@9")
    _, reading, _ = read_in(world, combo, dolma, "c05.unmutated.E@6", trace=trace)
    assert dict(reading.raw) == {"answer": "not covered"}
    _, refused, _ = read_in(world, combo, dolma, "c05.unmutated.E@6",
                            trace=_mod("baselines.trace").Refusal(exit_code=2, output="request refused"))
    assert dict(refused.raw) == {"answer": "not covered"}
    _, empty, _ = read_in(world, combo, dolma, "c05.unmutated.E@6", trace=EMPTY)
    assert dict(empty.raw) == {"answer": "not covered"}


def test_an_uncovered_range_adds_nothing_and_its_rows_still_count(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    plain, trace = tiny_world(tmp_path / "a", ["Z-001", "Z-002"], [[0], [1]], covered=16)
    _, reading, _ = read_in(plain, combo, dolma, "c05.unmutated.E@6", trace=trace)
    assert dict(reading.raw) == {"answer": "does not contain"}
    member, trace = tiny_world(tmp_path / "b", ["Z-001", "A-002"], [[0], [1]], covered=16)
    _, reading, _ = read_in(member, combo, dolma, "c05.unmutated.E@6", trace=trace)
    assert dict(reading.raw) == {"answer": "contains"}


def test_a_record_level_request_is_met_by_the_named_record_and_by_no_other_member_of_the_file(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    ids = ["Z-001", "A-002", "A-001"]
    # The file of A holds both A-002 and A-001; the request names the record A-001.
    other, trace = tiny_world(tmp_path / "other", ids, [[0], [1]], checkpoint="E@9")
    _, reading, _ = read_in(other, combo, dolma, "c05.record-X@A.unmutated.E@9", trace=trace)
    assert dict(reading.raw) == {"answer": "does not contain"}
    named, trace = tiny_world(tmp_path / "named", ids, [[0], [1, 2]], checkpoint="E@9")
    _, reading, _ = read_in(named, combo, dolma, "c05.record-X@A.unmutated.E@9", trace=trace)
    assert dict(reading.raw) == {"answer": "contains"}
    # The request that names the one record Z-089 of the file of Z.
    z_ids = ["Z-001", "Z-089"]
    other, trace = tiny_world(tmp_path / "z-other", z_ids, [[0], [0]], checkpoint="E@9")
    _, reading, _ = read_in(other, combo, dolma, "c08.shared-token.E@9", trace=trace)
    assert dict(reading.raw) == {"answer": "does not contain"}
    named, trace = tiny_world(tmp_path / "z-named", z_ids, [[0], [1]], checkpoint="E@9")
    _, reading, _ = read_in(named, combo, dolma, "c08.shared-token.E@9", trace=trace)
    assert dict(reading.raw) == {"answer": "contains"}


def test_the_history_of_a_checkpoint_goes_through_its_parent_and_the_trace_of_run_k_is_read_over_both_ranks(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world = universal_world(tmp_path)
    # D@3 reads shards 1 and 2 only; its parent E@9 read T, a generation output, at step 7.
    q, reading, _ = read_in(world, combo, dolma, "c05.unmutated.D@3", trace=make_trace())
    assert set(reading.raw) == {CANNOT} and "gen/T.jsonl" in reading.raw[CANNOT]
    member = make_trace({("E/seg-2", 9, 0): (0, SAMPLE_TOKENS, [(2, 20, 0, SAMPLE_TOKENS)])})
    for listed in ("c05.unmutated.D@3", "c05.unmutated.E@9"):
        assert dict(read_in(world, combo, dolma, listed, trace=member)[1].raw) == {"answer": "contains"}
    assert dict(read_in(world, combo, dolma, "c05.unmutated.E@6", trace=member)[1].raw) == {
        "answer": "does not contain"}
    # K@3 holds a member in the one sample of rank 1 at step 2: a reader of rank 0 alone misses it.
    assert dict(read_in(world, combo, dolma, "c05.unmutated.K@3", trace=make_trace())[1].raw) == {
        "answer": "contains"}
    no_rank_1 = make_trace({("K/seg-1", 2, 3): (1, SAMPLE_TOKENS, [(1, 3, 0, SAMPLE_TOKENS)])})
    assert dict(read_in(world, combo, dolma, "c05.unmutated.K@3", trace=no_rank_1)[1].raw) == {
        "answer": "does not contain"}


def test_the_trace_of_a_starting_state_of_check_2_is_the_one_that_state_wrote(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    states = {"c2.ii": without_rows(make_trace()), "c2.iii": without_rows(make_trace())}
    kept, ii = question("c02.i.E@6"), question("c02.ii.E@6")
    subject = world.combo(combo, dolma, [kept, ii], trace=make_trace(), state_traces=states)
    assert dict(subject.read(kept).raw) == {"answer": "does not contain"}
    assert grade.grade(kept, listed_expect(kept.listed_row), subject.read(kept)).status == "right"
    # The checkpoint is held, with no row for any step of its history: no record.
    assert dict(subject.read(ii).raw) == {"answer": "not covered"}
    got = grade.grade(ii, listed_expect(ii.listed_row), subject.read(ii))
    assert got.status == "wrong" and list(got.evidence) == ["answer"]
    # A state that holds no trace of its own reads the main one.
    iv = question("c02.iv.K@3")
    assert dict(subject.read(iv).raw) == {"answer": "contains"}


# --- The tie -----------------------------------------------------------------------------------


def test_where_the_tie_is_not_shown_the_checkpoint_is_cannot_answer_with_the_missing_tie(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world, trace = tiny_world(tmp_path, ["Z-001", "Z-002"], [[0], [1]])
    q, reading, subject = read_in(world, combo, dolma, "c05.unmutated.E@6", trace=trace,
                                  shown={"E/shard-1": False})
    assert dict(reading.raw) == {CANNOT: _not_shown_evidence(q)}
    assert subject.used_tie(q) is False
    # A shard that does show the tie does not answer for one that does not: the document read
    # through the shard that does not is unknown, so "does not contain" is no answer.
    two = World(tmp_path / "two", {"E/shard-1": ["Z-001", "Z-002"], "E/shard-2": ["Z-003", "Z-004"]})
    kept = {1: [(0, 0, SAMPLE_TOKENS, [(0, 0, 0, SAMPLE_TOKENS)]),
                (1, 0, SAMPLE_TOKENS, [(1, 1, 0, SAMPLE_TOKENS)])]}
    both = {"segments": [seg("E/seg-1", "E", None, ["shard-1", "shard-2"], [("E@6", 1)],
                             [launch(0, 0, 0, 1, 1)], kept)]}
    _, mixed, _ = read_in(two, combo, dolma, "c05.unmutated.E@6", trace=both, shown={"E/shard-2": False})
    assert dict(mixed.raw) == {CANNOT: _not_shown_evidence(q)}


def test_an_answer_that_used_the_tie_carries_its_mark_and_one_that_did_not_does_not(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    record = _mod("baselines.record")
    grade = _mod("baselines.grade")
    interfaces = _mod("baselines.interfaces")
    world, trace = tiny_world(tmp_path, ["Z-001", "Z-002"], [[0], [1]])
    q, reading, subject = read_in(world, combo, dolma, "c05.unmutated.E@6", trace=trace)
    assert q.tie_mark == _tie_mark()
    assert _check10_rows()[q.row_id]["marks"] == [interfaces.MARK_STEP_TO_ORDINAL]
    assert subject.used_tie(q) is True
    verdict = grade.grade(q, listed_expect(q.listed_row), reading)
    marked = record.record_of(q, reading, verdict, used_tie=subject.used_tie(q))
    assert _tie_mark() in marked.marks and interfaces.MARK_STEP_TO_ORDINAL in marked.marks
    alone = trace_only(combo, tmp_path / "alone", make_trace())
    first = question("c01.E.T")
    got = alone.read(first)
    plain = record.record_of(first, got, grade.grade(first, listed_expect("c01.E.T"), got),
                             used_tie=alone.used_tie(first))
    assert _tie_mark() not in plain.marks and interfaces.MARK_STEP_TO_ORDINAL in plain.marks
    # A checkpoint the trace does not hold used no tie either.
    absent = alone.read(question("c05.unregistered-checkpoint"))
    assert dict(absent.raw) == {"answer": "not covered"}
    assert alone.used_tie(question("c05.unregistered-checkpoint")) is False


def test_the_rows_of_the_checkpoint_rule_carry_the_mark_of_the_step_to_ordinal_part_and_the_tie(facts):
    interfaces = _mod("baselines.interfaces")
    groups = {"c01.E@9", "c02.checkpoints", "c04.trace-count", "c04.not-a-stop", "c05.unmutated.checkpoints",
              "c05.f.trace-deleted", "c05.f.trace-half", "c05.f.map-deleted", "c05.f.digest",
              "c05.record-X@A.unmutated", "c08.checkpoints", "c08.step-8-holds-no-T", "c08.shared-token",
              "c09.exposure-free-restart"}
    questions = [q for q in routed_questions() if q.group in groups]
    assert questions
    for q in questions:
        assert interfaces.MARK_STEP_TO_ORDINAL in q.marks, q.id
        assert q.tie_mark == _tie_mark() and q.needs_ordinal is True, q.id


# --- The step 8 of segment 2, and the restart ------------------------------------------------


STEP8 = "c08.withdraw-A.E@9.step-8-holds-no-T"


def test_the_cut_sample_of_step_8_holds_no_row_tied_to_t_and_the_first_exposure_is_step_7(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    q, reading, subject = read_in(world, combo, dolma, STEP8, trace=make_trace())
    assert dict(reading.raw) == {"rows_that_name_T": 0, "first_exposure": {"segment": "E/seg-2", "step": 7}}
    assert grade.grade(q, listed_expect(STEP8), reading).status == "right"
    assert subject.used_tie(q) is True
    # A row of the covered part of that sample ties to T.
    named = make_trace({("E/seg-2", 8, 3): (0, CUT_COVERED, [(2, 2, 19, 35), (2, 7, 0, 5)])})
    q, reading, _ = read_in(world, combo, dolma, STEP8, trace=named)
    assert dict(reading.raw) == {"rows_that_name_T": 1, "first_exposure": {"segment": "E/seg-2", "step": 7}}
    got = grade.grade(q, listed_expect(STEP8), reading)
    assert got.status == "wrong" and list(got.evidence) == ["rows_that_name_T"]
    # T read earlier, in the first segment, is the first exposure: the history is read oldest first.
    earlier = make_trace({("E/seg-1", 2, 0): (0, SAMPLE_TOKENS, [(0, 18, 0, SAMPLE_TOKENS)])})
    q, reading, _ = read_in(world, combo, dolma, STEP8, trace=earlier)
    assert dict(reading.raw) == {"rows_that_name_T": 0, "first_exposure": {"segment": "E/seg-1", "step": 2}}
    got = grade.grade(q, listed_expect(STEP8), reading)
    assert got.status == "wrong" and list(got.evidence) == ["first_exposure"]
    # No row of the history ties to T.
    none = make_trace({("E/seg-2", 7, 0): (0, SAMPLE_TOKENS, [(2, 24, 32, 39), (2, 26, 0, 4)])})
    q, reading, _ = read_in(world, combo, dolma, STEP8, trace=none)
    assert dict(reading.raw) == {"rows_that_name_T": 0, "first_exposure": None}
    got = grade.grade(q, listed_expect(STEP8), reading)
    assert got.status == "wrong" and list(got.evidence) == ["first_exposure"]
    # The tie not shown: cannot answer with the missing tie.
    q, blocked, _ = read_in(world, combo, dolma, STEP8, trace=make_trace(), shown=False)
    assert dict(blocked.raw) == {CANNOT: _not_shown_evidence(q)}


RESTART = "c09.E.exposure-free-restart"


def test_the_exposure_free_restart_is_the_latest_checkpoint_of_the_run_read_does_not_contain(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    # Checkpoint 6 does not contain, checkpoint 9 reads a member at step 9.
    member = make_trace({("E/seg-2", 9, 0): (0, SAMPLE_TOKENS, [(2, 20, 0, SAMPLE_TOKENS)])})
    q, reading, _ = read_in(world, combo, dolma, RESTART, trace=member)
    assert dict(reading.raw) == {"exposure_free_restart": "E@6"}
    assert grade.grade(q, listed_expect(RESTART), reading).status == "right"
    # Neither contains and no row reads a generation output: the latest is checkpoint 9.
    clean = make_trace({("E/seg-2", 7, 0): (0, SAMPLE_TOKENS, [(0, 3, 0, SAMPLE_TOKENS)])})
    q, reading, _ = read_in(world, combo, dolma, RESTART, trace=clean)
    assert dict(reading.raw) == {"exposure_free_restart": "E@9"}
    got = grade.grade(q, listed_expect(RESTART), reading)
    assert got.status == "wrong" and list(got.evidence) == ["exposure_free_restart"]
    # The first checkpoint contains: there is none.
    early = make_trace({("E/seg-1", 3, 1): (0, SAMPLE_TOKENS, [(1, 19, 0, SAMPLE_TOKENS)])})
    q, reading, _ = read_in(world, combo, dolma, RESTART, trace=early)
    assert dict(reading.raw) == {"exposure_free_restart": None}
    # One checkpoint of the run reads a generation output: the row is cannot answer.
    q, reading, _ = read_in(world, combo, dolma, RESTART, trace=make_trace())
    assert set(reading.raw) == {CANNOT}
    assert grade.grade(q, listed_expect(RESTART), reading).status == "cannot answer"


# --- The faults, read from what is left ------------------------------------------------------

A_ROW_IN_SHARD_2 = (1, 19, 0, SAMPLE_TOKENS)  # the member A-002, the last line of E/shard-2


def e6_answer(world, combo, dolma, listed, trace):
    q, reading, _ = read_in(world, combo, dolma, listed, trace=trace)
    return q, reading


def test_a_fault_of_the_trace_removes_what_it_removes_and_the_reading_is_an_answer_never_a_stop(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    cases = [
        # (where the one member is read, the listed row that puts the fault)
        (("E/seg-1", 3, 1, [A_ROW_IN_SHARD_2]), "c05.f.trace-deleted.E@6"),
        (("E/seg-1", 4, 3, [(0, 1, 0, 10), (0, 2, 0, 10), A_ROW_IN_SHARD_2]), "c05.f.trace-half.E@6"),
        (("E/seg-1", 5, 3, [A_ROW_IN_SHARD_2]), "c04.trace-count.E@6"),
    ]
    for number, ((segment, step, pos, rows), listed) in enumerate(cases):
        trace = make_trace({(segment, step, pos): (0, SAMPLE_TOKENS, rows)})
        before = copy.deepcopy(trace)
        _, unmutated = e6_answer(world, combo, dolma, "c05.unmutated.E@6", trace)
        assert dict(unmutated.raw) == {"answer": "contains"}, listed
        q, faulted = e6_answer(world, combo, dolma, listed, trace)
        assert dict(faulted.raw) == {"answer": "does not contain"}, listed
        assert "stopped" not in faulted.raw
        assert trace == before, "the trace the subject holds is never changed"
        got = grade.grade(q, listed_expect(listed), faulted)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
        # A hand-written stop is what the stop rows expect.
        if "error" in listed_expect(listed):
            stopped = _mod("baselines.interfaces").Reading(q.id, {"stopped": True}, q.route)
            assert grade.grade(q, listed_expect(listed), stopped).status == "right"


def test_a_trace_fault_that_does_not_remove_a_row_leaves_the_answer_as_it_was(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    trace = make_trace()
    _, unmutated = e6_answer(world, combo, dolma, "c05.unmutated.E@6", trace)
    assert dict(unmutated.raw) == {"answer": "does not contain"}
    for listed in ("c04.not-a-stop.trace-deleted", "c04.not-a-stop.trace-half", "c05.f.trace-half.E@6"):
        q, reading = e6_answer(world, combo, dolma, listed, trace)
        assert dict(reading.raw) == {"answer": "does not contain"}, listed
        got = grade.grade(q, listed_expect(listed), reading)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed


def test_the_faults_of_the_companion_rows_reach_the_trace_rows_tied_to_them(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world = universal_world(tmp_path)
    run = world.run(dolma)
    held = tuple(run.rows)

    def answer(listed, trace):
        q = question(listed)
        subject = world.combo(combo, dolma, [q], trace=trace, runs={target_of(q): run})
        return subject.read(q)

    # A member read through a line of shard 1, whose companion rows are the ones the fault deletes.
    trace = make_trace({("E/seg-1", 2, 0): (0, SAMPLE_TOKENS, [(0, 19, 0, SAMPLE_TOKENS)])})
    assert dict(answer("c05.unmutated.E@6", trace).raw) == {"answer": "contains"}
    # No row has that src and loc any more: the trace row has no companion row and adds nothing.
    assert dict(answer("c05.f.map-deleted.E@6", trace).raw) == {"answer": "does not contain"}
    # The digest fault changes an end, not an id: the row still resolves.
    assert dict(answer("c05.f.digest.E@6", trace).raw) == {"answer": "contains"}
    assert tuple(run.rows) == held, "the rows the run holds are never changed"
    # A shard of the toolkit is read as the toolkit alone reads it, whatever fault of the trace is put.
    alone = answer("c05.unmutated.E/shard-1", trace)
    for listed in ("c05.f.trace-deleted.E/shard-1", "c05.f.trace-half.E/shard-1"):
        assert dict(answer(listed, trace).raw) == dict(alone.raw), listed
    assert dict(alone.raw) == {"answer": "contains"}
    assert dict(answer("c05.f.map-deleted.E/shard-1", trace).raw) == {"answer": "not covered"}


# --- Artifacts neither half holds a record for -------------------------------------------------

KIND_ROWS = [
    ("c05.f.map-deleted.src/A", "c05.unmutated.files"),
    ("c05.f.map-deleted.gen/T", "c05.unmutated.files"),
    ("c05.f.map-deleted.R1", "c05.unmutated.files"),
    ("c05.f.digest.E/mix-1", "c05.unmutated.mixtures"),
    ("c05.f.digest.listing", "c05.listing.map-deleted"),
]


@pytest.mark.parametrize("listed, group", KIND_ROWS, ids=[row for row, _ in KIND_ROWS])
def test_a_kind_of_artifact_neither_half_holds_is_cannot_answer_with_the_missing_text_of_the_combination(
        need_module, tmp_path, listed, group):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    world = universal_world(tmp_path)
    _, reading, _ = read_in(world, combo, dolma, listed, trace=make_trace())
    assert dict(reading.raw) == {CANNOT: _missing_of(group)}
    assert _missing_of(group) != _missing_of(group, "dolma") or group == "c05.listing.map-deleted"


def test_cannot_answer_for_a_kind_needs_a_successful_run_of_the_store(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    world = universal_world(tmp_path)
    q = question("c05.f.map-deleted.src/A")
    failed = interfaces.Attempt(id="attempt-9", command=("step",), exit_code=0, output="no row")
    subject = world.combo(combo, dolma, [q], trace=make_trace(), runs={target_of(q): failed})
    assert subject.read(q) is failed
    absent = world.combo(combo, dolma, [q], trace=make_trace(), runs={})
    with pytest.raises(ValueError) as caught:
        absent.read(q)
    assert type(caught.value).__name__ == "NotAttempted" and isinstance(caught.value, combo.NotAttempted)
    # A run that wrote no row for a chunk file that has lines is no successful run.
    empty = combo.DolmaTraceSubject(
        {target_of(q): dolma.DolmaRun(build=world.build, rows=(), chunk_files=dict(world.chunk_files))},
        trace=make_trace(), tie_record=world.write_tie(), requests=REQUESTS, place_of=place_of)
    with pytest.raises(ValueError):
        empty.read(q)


# --- A failed half, and no run -----------------------------------------------------------------


def test_a_failed_run_of_the_toolkit_is_that_attempt_for_every_question_that_needs_it(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    world = universal_world(tmp_path)
    failed = interfaces.Attempt(id="attempt-9", command=("step",), exit_code=0, output="no row")
    for listed in ("c05.unmutated.E@6", "c05.unmutated.E/shard-1", "c05.f.map-deleted.R1", STEP8, RESTART):
        q = question(listed)
        subject = world.combo(combo, dolma, [q], trace=make_trace(), runs={target_of(q): failed})
        assert subject.read(q) is failed, listed
    # The trace half alone needs no run of the toolkit and reads whatever the run was.
    first = question("c01.E.T")
    subject = world.combo(combo, dolma, [first], trace=make_trace(), runs={target_of(first): failed})
    assert isinstance(subject.read(first), interfaces.Reading)


def test_a_failed_export_is_that_attempt_for_the_questions_that_need_the_trace_only(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    world = universal_world(tmp_path)
    failed = interfaces.Attempt(id="attempt-export", command=("show",), exit_code=1, output="boom")
    questions = [question(listed) for listed in
                 ("c01.E.T", "c01.E.step8.counted", "c05.unregistered-checkpoint", "c05.unmutated.E@6",
                  STEP8, RESTART, "c05.unmutated.E/shard-1", "c06.fixture.documents-mapped.E/shard-1")]
    subject = world.combo(combo, dolma, questions, trace=failed)
    for q in questions[:6]:
        assert subject.read(q) is failed, q.id
    # The toolkit half alone reads without the trace.
    for q in questions[6:]:
        assert isinstance(subject.read(q), interfaces.Reading), q.id


def test_a_store_with_no_run_of_the_toolkit_is_not_attempted_and_a_question_with_no_record_needs_none(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    world = universal_world(tmp_path)
    for listed in ("c05.unmutated.E@6", "c05.unmutated.E/shard-1", STEP8, RESTART, "c05.f.map-deleted.R1"):
        q = question(listed)
        subject = world.combo(combo, dolma, [q], trace=make_trace(), runs={})
        with pytest.raises(ValueError) as caught:
            subject.read(q)
        assert type(caught.value).__name__ == "NotAttempted", listed
    # The trace half alone is read where no run of the toolkit exists.
    none_q = question("c02.i.arrays.E/seg-1")
    assert none_q.route is None
    subject = world.combo(combo, dolma, [], trace=make_trace(), runs={})
    got = subject.read(none_q)
    assert isinstance(got, interfaces.Reading) and got.raw is None and got.route is None


# --- A layout with no run -----------------------------------------------------------------------


def test_a_layout_with_no_successful_run_is_not_attempted_never_cannot_answer(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    expected = _mod("baselines.expected")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    ten = question("c09.10k.successor-shards")
    held = expected.concrete_values_by_layout(expected.expected_for("c09.10k.successor-shards"))
    target = target_of(ten)
    subject = world.combo(combo, dolma, [ten], trace=EMPTY, runs={}, layout_runs={})
    by_layout = subject.read_by_layout(ten)
    assert by_layout == {"partitioned": None, "round-robin": None}
    assert grade.grade_layouts(ten, held, by_layout).status == "not attempted"
    # A layout whose run failed is that failed attempt; the other layout, with no run, is not
    # attempted, and the line is the failed attempt's status.
    failed = interfaces.Attempt(id="attempt-pt", command=("step",), exit_code=1, output="boom")
    subject = world.combo(combo, dolma, [ten], trace=EMPTY, runs={},
                          layout_runs={(target, "partitioned"): failed})
    by_layout = subject.read_by_layout(ten)
    assert by_layout["partitioned"] is failed and by_layout["round-robin"] is None
    assert grade.grade_layouts(ten, held, by_layout).status == "no answer: failed attempt attempt-pt"
    # The row of a layout, asked of a row that names none, is refused.
    with pytest.raises(ValueError):
        subject.read_by_layout(question("c05.unmutated.E@6"))


def test_a_layout_that_ran_reads_its_own_cannot_answer_for_a_row_the_table_gives_no_join_of_the_shards_for(need_module, tmp_path):
    # 17 Nov 2026 (ruled per half): the layout that ran on a no-join row reads cannot answer of its own, so the line follows the
    # layouts' slots: a layout with no run is None and the line is "not attempted" (no cannot-answer line for a layout the
    # toolkit never ran on); a failed run comes first; both layouts run: cannot answer.
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    expected = _mod("baselines.expected")
    grade = _mod("baselines.grade")
    world = universal_world(tmp_path)
    ten = question("c09.10k.successor-shards")
    held = expected.concrete_values_by_layout(expected.expected_for("c09.10k.successor-shards"))
    target = target_of(ten)
    run = world.run(dolma)
    subject = world.combo(combo, dolma, [ten], trace=EMPTY, runs={}, layout_runs={(target, "partitioned"): run})
    by_layout = subject.read_by_layout(ten)
    assert by_layout["round-robin"] is None
    assert by_layout["partitioned"].raw == {"cannot_answer": dolma.NO_JOIN_TEXT}
    assert grade.grade_layouts(ten, held, by_layout).status == "not attempted"
    both = world.combo(combo, dolma, [ten], trace=EMPTY, runs={},
                       layout_runs={(target, "partitioned"): run, (target, "round-robin"): run})
    assert grade.grade_layouts(ten, held, both.read_by_layout(ten)).status == "cannot answer"
    failed = interfaces.Attempt(id="attempt-rr", command=("step",), exit_code=1, output="boom")
    mixed = world.combo(combo, dolma, [ten], trace=EMPTY, runs={},
                        layout_runs={(target, "partitioned"): run, (target, "round-robin"): failed})
    assert grade.grade_layouts(ten, held, mixed.read_by_layout(ten)).status == "no answer: failed attempt attempt-rr"


def test_a_row_that_names_one_layout_is_read_from_the_run_of_that_layout_not_from_the_run_of_the_target(need_module, tmp_path):
    """Found by the first all-size pass on the cut tree (pass 15, 18 Nov 2026, not of record): c09.10k.partitioned.kept-shards is a
    row of the 10k set in the partitioned layout; the combination read it from the run of the target (the round-robin build) and
    answered {"keep": []} where the plain Dolma, which reads the run of the layout the row names (W8), reads the partitioned build.
    A question reads exactly one build, picked by what its row names: here the failed attempt of the partitioned layout comes
    first, whatever the run of the target holds."""
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    world = universal_world(tmp_path)
    kept = question("c09.10k.partitioned.kept-shards")
    assert _mod("baselines.record").layout_of(kept) == "partitioned"
    target = target_of(kept)
    run = world.run(dolma)
    failed = interfaces.Attempt(id="attempt-pt", command=("step",), exit_code=1, output="boom")
    subject = world.combo(combo, dolma, [kept], trace=EMPTY, runs={target: run},
                          layout_runs={(target, "partitioned"): failed, (target, "round-robin"): run})
    reading = subject.read(kept)
    assert reading is failed, reading


def test_a_question_that_records_none_names_no_run_in_any_layout(need_module, tmp_path):
    combo = need_module(MODULE)
    interfaces = _mod("baselines.interfaces")
    q = question("c09.10k.action-counts")
    assert q.route is None and _mod("baselines.record").layouts_of(q) == ("partitioned", "round-robin")
    subject = combo.DolmaTraceSubject({}, trace=EMPTY, tie_record=tmp_path / "tie.json", requests=REQUESTS,
                                      place_of=place_of)
    by_layout = subject.read_by_layout(q)
    assert set(by_layout) == {"partitioned", "round-robin"}
    for got in by_layout.values():
        assert isinstance(got, interfaces.Reading) and got.raw is None and got.route is None


# --- The writer states of Check 4 ---------------------------------------------------------------


def scratch_world(tmp_path):
    return World(tmp_path, {"c04/shard-500": SCRATCH_IDS})


def kill_and_retry(combo, dolma, world, **more):
    """The readings of both questions of the stopped write on one run, with a second run that
    wrote the whole map; ``more`` replaces what the run holds."""
    stop_q, retry_q = question("c04.kill-after-100"), question("c04.retry")
    settings = {"stopped_rows": tuple(scratch_rows(world, 500)), "retry_rows": tuple(scratch_rows(world, 500)),
                "retry_exit_code": 0, "token_file_tokens": SCRATCH_TOKENS}
    settings.update(more)
    subject = world.combo(combo, dolma, [stop_q, retry_q], trace=EMPTY, **settings)
    return subject.read(stop_q), subject.read(retry_q)


def test_the_rows_after_a_stop_say_whether_they_showed_while_the_writer_ran_or_only_at_its_end(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = scratch_world(tmp_path)
    q = question("c04.kill-after-100")
    assert target_of(q) == ("store", "scratch-500")
    for showed, words in ((True, "while the writer ran"), (False, "only at its end")):
        for count, status in ((100, "wrong"), (500, "right"), (0, "right")):
            subject = world.combo(combo, dolma, [q], trace=EMPTY, stopped_rows=tuple(scratch_rows(world, count)),
                                  rows_showed_while_running=showed)
            reading = subject.read(q)
            assert dict(reading.raw) == {"records": {"rows_after_stop": count, "rows_showed": words}}
            # The grade reads the count alone.
            assert grade.grade(q, listed_expect(q.listed_row), reading).status == status
    for bad in (None, 1, "while the writer ran"):
        subject = world.combo(combo, dolma, [q], trace=EMPTY, stopped_rows=tuple(scratch_rows(world, 100)),
                              rows_showed_while_running=bad)
        with pytest.raises(ValueError):
            subject.read(q)


def test_a_stop_that_did_not_land_is_the_failed_attempt_of_the_stopped_writer_with_its_exit(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    printed = f"Traceback: File \"{PRIVATE_PATH}/venv-dolma/lib/x.py\", line 3\ndone"
    for exit_code in NOT_LANDED_EXITS:
        world = scratch_world(tmp_path / str(exit_code))
        for got in kill_and_retry(combo, dolma, world, stopped_exit_code=exit_code, stopped_output=printed):
            assert isinstance(got, interfaces.Attempt), exit_code
            assert not isinstance(got, interfaces.Reading)
            assert got.id == STOP_ID and got.exit_code == exit_code
            assert PRIVATE_PATH not in got.output and "done" in got.output
    # A stop that landed, by the signal sent (its negative, or 128 plus its number), is read.
    for exit_code in (-9, 137):
        world = scratch_world(tmp_path / f"landed-{exit_code}")
        stopped, retried = kill_and_retry(combo, dolma, world, stopped_exit_code=exit_code)
        assert isinstance(stopped, interfaces.Reading) and isinstance(retried, interfaces.Reading)


def test_the_second_run_is_refused_by_the_error_class_of_the_toolkit_and_a_crash_is_a_failed_attempt(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    grade = _mod("baselines.grade")
    q = question("c04.retry")
    expect = listed_expect("c04.retry")

    def read(exit_code, output, rows=(), **more):
        world = scratch_world(tmp_path / f"r{len(list(tmp_path.iterdir()))}")
        subject = world.combo(combo, dolma, [q], trace=EMPTY, retry_rows=rows, retry_exit_code=exit_code,
                              retry_output=output, token_file_tokens=SCRATCH_TOKENS,
                              retry_attempt_id=RETRY_ID, retry_command=("python", "-m", "dolma.cli"), **more)
        return subject.read(q)

    refusal = ("Traceback (most recent call last):\n  File \"x.py\", line 3, in run\n"
               "dolma.core.errors.DolmaConfigError: declared count mismatch: the map holds 100 rows\n")
    got = read(1, refusal)
    assert dict(got.raw["records"]) == {"rows_after_retry": 0, "matches_shard": None, "refused": True,
                                        "mismatch": "declared count mismatch: the map holds 100 rows"}
    assert grade.grade(q, expect, got).status == "right"
    crash = "Traceback (most recent call last):\n  File \"x.py\", line 3, in run\nKeyError: 'id'\n"
    failed = read(1, crash)
    assert isinstance(failed, interfaces.Attempt) and failed.id == RETRY_ID and failed.exit_code == 1
    world = scratch_world(tmp_path / "whole")
    whole = tuple(scratch_rows(world, 500))
    subject = world.combo(combo, dolma, [q], trace=EMPTY, retry_rows=whole, retry_exit_code=0,
                          token_file_tokens=SCRATCH_TOKENS)
    reading = subject.read(q)
    assert dict(reading.raw["records"]) == {"rows_after_retry": 500, "matches_shard": True,
                                            "refused": False, "mismatch": None}
    assert grade.grade(q, expect, reading).status == "right"


# --- The rows of the toolkit alone ---------------------------------------------------------------

SET_SHARDS = {
    "shard-1": ["Z-001", "Z-002", "Z-003", "Z-004", "Z-005"],
    "shard-2": ["Z-006", "Z-007", "Z-008", "Z-009"],
}
SET_PATHS = {
    "shard-1": "pipeline/round-robin/tokenizer-input/shard-1.jsonl",
    "shard-2": "pipeline/round-robin/tokenizer-input/shard-2.jsonl",
}
# The spans of SET_SHARDS: the tokens of ordinal n are 3 + n % 3 (3, 4, 5, 3, 4, ...).
SET_SPANS = {"shard-1": [3, 4, 5, 3, 4], "shard-2": [3, 4, 5, 3]}
COUNT_KEYS = {"per_ordinal", "unmapped_documents", "documents_per_shard",
              "documents_with_one_tied_row_per_shard", "documents_with_no_tied_row_per_shard"}


def set_world(tmp_path) -> World:
    return World(tmp_path, SET_SHARDS, SET_PATHS)


def counted_document(shards: Mapping[str, list]) -> dict:
    """A counting document of the plant script for shards with no identical documents: every
    document is recovered."""
    return {"documents": {"total": sum(len(i) for i in shards.values()), "shards": [
        {"shard": name, "documents": len(ids), "ids": list(ids), "identical": [],
         "labels": {"recovered": len(ids), "candidates": 0}}
        for name, ids in shards.items()]}}


def concrete_of(listed: str, size: str | None, **sources):
    expected = _mod("baselines.expected")
    return expected.concrete_values(expected.expected_for(listed, size, **sources))


def test_the_documents_mapped_reading_holds_the_three_counts_per_shard_and_the_ids(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    row_type = _mod("baselines.companion").Row
    world = set_world(tmp_path)
    q = question("c06.generated.documents-mapped", "small")
    reading = world.combo(combo, dolma, [q], trace=EMPTY).read(q)
    assert set(reading.raw) == COUNT_KEYS and reading.route == q.route
    assert dict(reading.raw["documents_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(reading.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(reading.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 0, "shard-2": 0}
    assert reading.raw["unmapped_documents"] == 0
    assert {name: list(ids) for name, ids in reading.raw["per_ordinal"].items()} == SET_SHARDS
    # Shard 1: the document of ordinal 1 has no row, the document of ordinal 3 has two.
    world.without("shard-1", 2)
    world.rows.append(row_type(start=90, end=95, id="Z-004", src=world.chunk_files["shard-1"], loc=4))
    changed = world.combo(combo, dolma, [q], trace=EMPTY).read(q)
    assert dict(changed.raw["documents_per_shard"]) == {"shard-1": 5, "shard-2": 4}
    assert dict(changed.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 3, "shard-2": 4}
    assert dict(changed.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 1, "shard-2": 0}
    assert changed.raw["unmapped_documents"] == 1
    assert changed.raw["per_ordinal"]["shard-1"][1] is None


def _dropping(subject, key):
    """A known-bad reader: it leaves one named value out of every reading."""
    interfaces = _mod("baselines.interfaces")

    class Dropper:
        name = SUBJECT

        def read(self, q):
            got = subject.read(q)
            if isinstance(got, interfaces.Reading) and isinstance(got.raw, Mapping) and key in got.raw:
                return interfaces.Reading(got.question_id, {k: v for k, v in got.raw.items() if k != key},
                                          got.route)
            return got

    return Dropper()


def test_the_documents_mapped_reading_is_graded_against_the_counts_and_a_missing_count_is_wrong(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    q = question("c06.generated.documents-mapped", "small")
    subject = world.combo(combo, dolma, [q], trace=EMPTY)
    concrete = concrete_of(q.listed_row, "small", counted={"small": counted_document(SET_SHARDS)})
    assert grade.grade_concrete(q, concrete, subject.read(q)).status == "right"
    assert subject.used_tie(q) is True
    # A reader that leaves one count out is found, for each count.
    for key in ("documents_per_shard", "documents_with_one_tied_row_per_shard",
                "documents_with_no_tied_row_per_shard"):
        got = grade.grade_concrete(q, concrete, _dropping(subject, key).read(q))
        assert got.status == "wrong" and list(got.evidence) == [key], key
    # One changed count (a document with no row) is wrong.
    world.without("shard-2", 4)
    got = grade.grade_concrete(q, concrete, world.combo(combo, dolma, [q], trace=EMPTY).read(q))
    assert got.status == "wrong"
    assert {"documents_with_one_tied_row_per_shard", "documents_with_no_tied_row_per_shard",
            "unmapped_documents"} <= set(got.evidence)


def test_a_document_of_an_identical_pair_has_its_own_row_and_is_counted_as_one_tied_row(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    shards = {"shard-1": [("Z-001", "the same passage"), ("Z-002", "the same passage"), "Z-003"]}
    world = World(tmp_path, shards, {"shard-1": SET_PATHS["shard-1"]})
    q = question("c06.generated.documents-mapped", "small")
    reading = world.combo(combo, dolma, [q], trace=EMPTY).read(q)
    assert dict(reading.raw["documents_per_shard"]) == {"shard-1": 3}
    assert dict(reading.raw["documents_with_one_tied_row_per_shard"]) == {"shard-1": 3}
    assert dict(reading.raw["documents_with_no_tied_row_per_shard"]) == {"shard-1": 0}


def test_the_set_rows_are_read_as_the_toolkit_alone_reads_them_and_graded_against_the_index_and_the_plant(
        need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    world = set_world(tmp_path)
    per, total = question("c03.spans.per-ordinal", "small"), question("c03.spans.total", "small")
    records = question("c03.records.capture", "small")
    split = question("c03.pairs.split", "small")
    pairs = [{"pair": "pair-2", "members": [{"shard": "shard-1", "ordinal": 4}, {"shard": "shard-2", "ordinal": 1}]}]
    subject = world.combo(combo, dolma, [per, total, records, split], trace=EMPTY, split_pairs=pairs)
    for q in (per, total):
        concrete = concrete_of(q.listed_row, "small", index={"small": SET_SPANS})
        assert grade.grade_concrete(q, concrete, subject.read(q)).status == "right", q.id
    assert {n: list(v) for n, v in subject.read(per).raw["per_ordinal"].items()} == SET_SPANS
    counted = counted_document(SET_SHARDS)
    counted["check3"] = {"pairs": {"list": [{
        "pair": "pair-2", "kind": "split", "placement": "split", "members": [
            {"id": "Z-005", "source": "Z", "shard": "shard-1", "line": 5, "ordinal": 4,
             "expected_label": "recovered"},
            {"id": "Z-007", "source": "Z", "shard": "shard-2", "line": 2, "ordinal": 1,
             "expected_label": "recovered"}]}]}}
    by_plant = concrete_of(records.listed_row, "small", counted={"small": counted_document(SET_SHARDS)})
    assert grade.grade_concrete(records, by_plant, subject.read(records)).status == "right"
    by_pair = concrete_of(split.listed_row, "small", counted={"small": counted})
    assert grade.grade_concrete(split, by_pair, subject.read(split)).status == "right"
    for q in (per, total, records, split):
        assert subject.used_tie(q) is (q is not total), q.id
    world.without("shard-2", 2)
    got = grade.grade_concrete(split, by_pair, world.combo(combo, dolma, [split], trace=EMPTY,
                                                              split_pairs=pairs).read(split))
    assert got.status == "wrong" and list(got.evidence) == ["record"]


def test_a_shard_is_read_by_the_shard_rule_of_the_toolkit(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    clean = World(tmp_path / "clean", {"E/shard-1": ["B-001", "Z-001"], "E/shard-3": ["T", "Z-002"]})
    q, reading, _ = read_in(clean, combo, dolma, "c05.unmutated.E/shard-1", trace=EMPTY)
    assert dict(reading.raw) == {"answer": "does not contain"}
    assert grade.grade(q, listed_expect(q.listed_row), reading).status == "right"
    q, reading, _ = read_in(clean, combo, dolma, "c05.unmutated.E/shard-3", trace=EMPTY)
    assert set(reading.raw) == {CANNOT} and "derived-from link" in reading.raw[CANNOT]
    holding = World(tmp_path / "holding", {"E/shard-1": ["A-002", "Z-001"]})
    q, reading, _ = read_in(holding, combo, dolma, "c05.unmutated.E/shard-1", trace=EMPTY)
    assert dict(reading.raw) == {"answer": "contains"}
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "wrong" and list(got.evidence) == ["answer"]
    empty = World(tmp_path / "empty", {"E/shard-1": []})
    _, reading, _ = read_in(empty, combo, dolma, "c05.unmutated.E/shard-1", trace=EMPTY)
    assert dict(reading.raw) == {"answer": "not covered"}


def test_the_count_fault_of_a_shard_of_500_documents_reads_an_answer_never_a_stop(need_module, tmp_path):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    grade = _mod("baselines.grade")
    ids = [f"Z-{n:03d}" for n in range(1, 500)] + ["A-001"]
    world = World(tmp_path, {"c04/shard-500": ids})
    q, reading, _ = read_in(world, combo, dolma, "c04.count-499", trace=EMPTY)
    assert target_of(q) == ("store", "scratch-500")
    assert dict(reading.raw) == {"answer": "does not contain"}, "the row of the 500th line is gone"
    got = grade.grade(q, listed_expect(q.listed_row), reading)
    assert got.status == "wrong" and list(got.evidence) == ["answer"]
    assert len([r for r in world.rows if r.src == world.chunk_files["c04/shard-500"]]) == 500


# --- Every question the table routes to the subject -------------------------------------------

CHECKPOINT_RULE_GROUPS = frozenset({
    "c01.E@9", "c02.checkpoints", "c04.trace-count", "c04.not-a-stop", "c05.unmutated.checkpoints",
    "c05.f.trace-deleted", "c05.f.trace-half", "c05.f.map-deleted", "c05.f.digest",
    "c05.record-X@A.unmutated", "c08.checkpoints", "c08.shared-token"})
# The values each row of c01.step-facts names (its rule), and the one value of the other groups
# whose reading is not an answer word.
STEP_FACTS = {
    "c01.E.step8.uncovered": {"uncovered_in_sample", "sample_tokens", "covered_tokens"},
    "c01.E.step8.counted": {"consumed_samples"},
    "c01.E.step8.untraced": {"untraced_steps"},
    "c01.E-rr.fully-traced": {"steps_with_uncovered_range", "steps_without_trace"},
    "c01.K.warm-start": {"consumed_at_start", "first_step", "last_step"},
    "c01.K.resume": {"consumed_at_start", "first_step", "last_step"},
    "c01.K.ranks": {"samples_per_step", "samples_per_rank", "ranks", "sample_pos_of_rank_0",
                    "sample_pos_of_rank_1"},
}
KEYS = {
    "c03.spans.per-ordinal": {"per_ordinal", "notes"},
    "c03.spans.total": {"per_shard", "notes"},
    "c03.records.capture": {"per_ordinal"},
    "c03.records.position-recovery": {"per_ordinal"},
    "c03.records.token-matching": {"per_ordinal"},
    "c03.pairs.split": {"record"},
    "c04.kill-after-100": {"records"},
    "c04.retry": {"records"},
    "c04.count-499": {"answer"},
    "c06.generated.documents-mapped": COUNT_KEYS,
    "c09.fixture.shards.counts": {"counts", "keep", "rewrite"},
    "c09.10k.partitioned.kept-shards": {"keep"},
    "c09.10k.round-robin.all-rewrite": {"keep", "rewrite"},
}


def kind_of(artifact: str | None) -> str:
    if artifact is None:
        return "set"
    if artifact.startswith(("src/", "gen/", "import/")) or re.fullmatch(r"(successor of )?R\d+", artifact):
        return "files"
    if "/mix-" in artifact:
        return "mixtures"
    if "@" in artifact:
        return "checkpoints"
    if artifact == "(listing)":
        return "listing"
    return "shard"


def wanted_keys(listed: str, artifact: str | None) -> set[str] | None:
    """The keys of a reading that is not cannot answer, or None where the artifact is of a kind
    neither half holds a record for."""
    kind = kind_of(artifact)
    if kind in ("files", "mixtures", "listing"):
        return None
    if kind == "checkpoints":
        return {"answer"}
    if listed in KEYS:
        return KEYS[listed]
    if listed.startswith("c06.fixture.documents-mapped."):
        return {"documents", "recovered", "unknown"}
    if listed.endswith(".action"):
        return {"action"}
    return {"answer"}


def key_problems(subject, questions) -> list[str]:
    interfaces = _mod("baselines.interfaces")
    problems = []
    for q in questions:
        listed, group = q.listed_row, q.group
        given = _listed_rows()[listed]["given"]
        reading = subject.read(q)
        if not isinstance(reading, interfaces.Reading) or reading.question_id != q.id:
            problems.append(f"{q.id}: not a reading of the question")
            continue
        if reading.route != q.route:
            problems.append(f"{q.id}: route {reading.route!r}")
        keys = set(reading.raw)
        if group == "c01.positions":
            wanted = {"rows"}
        elif group == "c01.step-facts":
            wanted = STEP_FACTS[listed]
        elif group == "c08.step-8-holds-no-T":
            wanted = {"rows_that_name_T", "first_exposure"}
        elif group == "c09.exposure-free-restart":
            wanted = {"exposure_free_restart"}
        elif group == "c05.unregistered-checkpoint":
            wanted = {"answer"}
        else:
            wanted = wanted_keys(listed, given.get("artifact"))
        if wanted is None:
            if keys != {CANNOT}:
                problems.append(f"{q.id}: {sorted(keys)} for a kind with no record")
        elif group in ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint"):
            if keys != wanted:
                problems.append(f"{q.id}: {sorted(keys)}, wanted {sorted(wanted)}")
        elif given.get("artifact") in ("E/shard-1", "E/shard-2", "E-rr/shard-1", "E-rr/shard-2",
                                       "successor of E/shard-3"):
            if keys != wanted:
                problems.append(f"{q.id}: {sorted(keys)}, wanted {sorted(wanted)}")
        elif keys not in (wanted, {CANNOT}):
            problems.append(f"{q.id}: {sorted(keys)}, wanted {sorted(wanted)} or cannot answer")
        if given.get("set") is not None and listed.startswith(("c03.", "c06.generated.documents-mapped")) \
                and keys != wanted:
            problems.append(f"{q.id}: a set row reads {sorted(keys)}")
    return problems


def test_every_question_the_table_routes_to_the_subject_reads_the_values_its_rule_names(
        need_module, tmp_path, facts):
    combo, dolma = need_module(MODULE), need_module("baselines.dolma")
    interfaces = _mod("baselines.interfaces")
    subject_questions = [q for q in _questions().values() if q.subject == SUBJECT]
    assert len(subject_questions) == QUESTIONS_OF_THE_SUBJECT == facts.questions_per_subject
    routed = routed_questions()
    assert len(routed) == QUESTIONS_WITH_A_ROUTE
    # 207 are read and checked below; the 4 of the two open rows (three sizes of the first, one of
    # the second) are not.
    checked = [q for q in routed if q.listed_row not in OPEN_ROWS]
    assert len(checked) == 207 and len(routed) - len(checked) == 4
    # The same count from the table's own rows: every combination row that records something,
    # times the sizes each of its group's listed rows is asked at.
    from_table = [
        (listed, size) for (listed, subject, size) in facts.question_triples()
        if subject == SUBJECT and _check10_rows()[f"c10.{facts.group_of(listed)}.{SUBJECT}"].get("route")
    ]
    assert sorted(from_table, key=str) == sorted(((q.listed_row, q.size) for q in routed), key=str)

    world = universal_world(tmp_path)
    # A trace in which no row reads T: every checkpoint answers, so every reading holds its values.
    plain = make_trace({("E/seg-2", 7, 0): (0, SAMPLE_TOKENS, [(0, 3, 0, SAMPLE_TOKENS)])})
    subject = world.combo(combo, dolma, routed, trace=plain, split_pairs=UNIVERSAL_PAIRS,
                          stopped_rows=tuple(scratch_rows(world, 100)),
                          retry_rows=tuple(scratch_rows(world, 500)), retry_exit_code=0,
                          token_file_tokens=SCRATCH_TOKENS)
    assert key_problems(subject, checked) == []
    # The same check on a reader that leaves one named value out finds the fault.
    for key in ("documents_with_no_tied_row_per_shard", "per_ordinal", "record", "answer", "rows",
                "consumed_samples", "rows_that_name_T", "exposure_free_restart", "records"):
        assert key_problems(_dropping(subject, key), checked) != [], key
    # An answer that used the tie is one whose row has the tie's mark.
    for q in checked:
        if subject.used_tie(q):
            assert q.tie_mark == _tie_mark(), q.id
        if q.group in ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint"):
            assert subject.used_tie(q) is False, q.id
    # Every question whose row records none is read as no reading: no raw, no route.
    none_questions = [q for q in subject_questions if q.route is None]
    assert len(none_questions) == QUESTIONS_OF_THE_SUBJECT - QUESTIONS_WITH_A_ROUTE
    bare = combo.DolmaTraceSubject({}, trace=EMPTY, tie_record=tmp_path / "none-tie.json", requests=REQUESTS,
                                   place_of=place_of)
    for q in none_questions:
        reading = bare.read(q)
        assert isinstance(reading, interfaces.Reading) and reading.question_id == q.id, q.id
        assert reading.raw is None and reading.route is None, q.id
    # What the table leaves open is named, and each is a listed row of a routed group.
    assert OPEN_ROWS <= {q.listed_row for q in routed}


# One listed row of each routed group, graded once with a right and once with a wrong reading
# written by hand from the row's own expectation (the groups of the set rows, the writer states
# and the documents-mapped counts are graded by the tests above, and the positions by theirs).
GROUP_ROWS = {
    "c01.step-facts": "c01.E.step8.counted",
    "c01.E@9": "c01.E.E@9",
    "c02.checkpoints": "c02.i.E@6",
    "c04.count-499": "c04.count-499",
    "c04.trace-count": "c04.trace-count.E@6",
    "c04.not-a-stop": "c04.not-a-stop.trace-deleted",
    "c05.unmutated.shards": "c05.unmutated.E/shard-1",
    "c05.unmutated.checkpoints": "c05.unmutated.E@6",
    "c05.f.trace-deleted": "c05.f.trace-deleted.E@6",
    "c05.f.trace-half": "c05.f.trace-half.E@6",
    "c05.f.map-deleted": "c05.f.map-deleted.E/shard-2",
    "c05.f.digest": "c05.f.digest.E/shard-1",
    "c05.record-X@A.unmutated": "c05.record-X@A.unmutated.E/shard-3",
    "c05.withdraw-C.unmutated": "c05.withdraw-C.unmutated.E/shard-3",
    "c05.unregistered-checkpoint": "c05.unregistered-checkpoint",
    "c08.checkpoints": "c08.withdraw-A.E@6",
    "c08.step-8-holds-no-T": STEP8,
    "c08.shared-token": "c08.shared-token.E@6",
    "c09.artifact-actions": "c09.fixture.E/shard-1.action",
    "c09.successors": "c09.fixture.successor.E/shard-3",
    "c09.exposure-free-restart": RESTART,
}


def hand_readings(expect: Mapping):
    """A right and a wrong reading, written from the expectation of the listed row."""
    if "error" in expect:
        return {"stopped": True}, {"answer": "does not contain"}
    for key in ("answer", "action", "exposure_free_restart"):
        if key in expect:
            return {key: expect[key]}, {key: f"{expect[key]} changed"}
    if "consumed_samples" in expect:
        return ({"consumed_samples": expect["consumed_samples"]},
                {"consumed_samples": expect["consumed_samples"] + 1})
    if "rows_that_name_T" in expect:
        right = {"rows_that_name_T": expect["rows_that_name_T"], "first_exposure": expect["first_exposure"]}
        return right, {**right, "rows_that_name_T": expect["rows_that_name_T"] + 1}
    raise AssertionError(f"no hand-written reading for {sorted(expect)}")


def test_each_routed_group_takes_a_right_and_a_wrong_reading_through_the_grader(facts):
    grade = _mod("baselines.grade")
    interfaces = _mod("baselines.interfaces")
    routed_groups = {q.group for q in routed_questions()}
    covered = set(GROUP_ROWS) | {"c01.positions", "c03.spans", "c03.records", "c03.pairs.split",
                                 "c04.interrupted-write", "c06.documents-mapped", "c06.downstream"}
    assert covered == routed_groups, sorted(covered ^ routed_groups)
    for group, listed in GROUP_ROWS.items():
        q = question(listed)
        assert q.group == group, listed
        expect = listed_expect(listed)
        right, wrong = hand_readings(expect)
        got = grade.grade(q, expect, interfaces.Reading(q.id, right, q.route))
        assert got.status == "right", (group, got)
        got = grade.grade(q, expect, interfaces.Reading(q.id, wrong, q.route))
        assert got.status == "wrong", (group, got)


# --- The module -------------------------------------------------------------------------


def test_the_module_imports_nothing_forbidden_and_holds_no_local_path(need_module):
    module = need_module(MODULE)
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
