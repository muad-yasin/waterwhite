"""The combination of OriginBlame and the trace, on hand-written records and a stand-in for the
sandboxed driver (plan 5.5 Check 10; table README, section "Check 10": the subject
``originblame+trace``, its conditions, "The OriginBlame tie by token range" and "Readings chosen
here" 7 and 8; the rows ``c10.<group>.originblame+trace`` of ``check10.toml``).

Nothing of the tool is started. The tool half is ``baselines.originblame.OriginBlame`` over a
stand-in runner (``FakeTool`` below, a trimmed copy of the stand-in of the tool's own tests: it
keeps its stores as the files the tool leaves, models the second revoke that un-revokes and
returns what the real driver prints). The trace half is a document ``{"segments": [...]}`` in
the form ``baselines.trace.export_segment`` returns, written by hand from the fixture spec.

What this file asks of ``baselines.combo_originblame``:

- ``OriginBlameTrace(originblame, *, trace_of, place_of)``: a ``Subject`` named
  ``originblame+trace``. ``originblame`` is an ``OriginBlame`` as built (its shards, its
  requests, its stand-in runner, its fault putter); the caller has called its ``track_all``.
  ``trace_of(listed_row)`` gives the trace the question's starting state reads, as the
  document ``{"segments": [...]}``; a document with no step in a segment is a trace the
  registration wrote no rows for; where the export of the trace failed it gives the failed
  ``Attempt``. ``place_of(listed_row)`` is the one ``baselines.trace.TraceSubject`` takes.
  ``read(question)`` and ``read_by_layout(question)`` are those of ``OriginBlame``: ``read``
  refuses a row that names two layouts, ``read_by_layout`` gives one result per layout and None
  for a layout none of whose shards was given or tracked. Neither half is changed.
- A question whose row records none has the Reading with ``raw`` None and ``route`` None, and
  nothing is started for it.
- The trace half alone (``c01.positions``, ``c01.step-facts``, ``c05.unregistered-checkpoint``):
  the values ``baselines.trace.TraceSubject`` reads from the document ``trace_of`` gives.
  Nothing of the tool is started for them. A failed ``Attempt`` of the export is returned as
  that Attempt.
- The tool half alone (``c03.spans``, ``c04.interrupted-write``, ``c04.count-499``,
  ``c05.unmutated.shards``, ``c05.withdraw-C.unmutated``, ``c06.downstream``,
  ``c09.artifact-actions``, ``c09.successors``, and the shards of the four groups that put a
  fault: ``c05.f.trace-deleted``, ``c05.f.trace-half``, ``c05.f.map-deleted``,
  ``c05.f.digest``): the reading ``OriginBlame`` gives the same question, a failed Attempt of
  it (a track that failed, a stop that did not land) included, as that Attempt, and its
  ``NotAttempted`` before the store was tracked.
- A kind of artifact neither half holds (a source file, a generation output, a release, a
  mixture, the listing) is ``{"cannot_answer": <text>}``, the ``missing`` text of the
  combination's own row ``c10.c05.unmutated.files.originblame+trace``,
  ``c10.c05.unmutated.mixtures.originblame+trace`` or
  ``c10.c05.listing.map-deleted.originblame+trace``, and only where the tool ran on the store
  the question is asked on (every shard of it tracked): where one failed, that failed Attempt;
  where none was given or tracked, ``NotAttempted``.
- The walk (a checkpoint: ``c01.E@9``, ``c02.checkpoints``, ``c04.trace-count``,
  ``c04.not-a-stop``, ``c05.unmutated.checkpoints``, the checkpoints of the four fault groups,
  ``c08.checkpoints``): the history of the checkpoint is read from the document ``trace_of`` gives
  (``baselines.history.history_rows``); each row is tied to the token-index entry of its
  document by RANGE: entry k of the store that tracked the shard covers the tokens from the sum
  of the token counts of entries 0 to k-1 up to that sum plus its own count, the document with
  ordinal n covers the range the shard's own index gives it (the sizes of the documents of the
  ``ShardSpec``, never the entries' own counts), and a row is tied to the entry whose range
  equals its document's. The entry's bit is read from the forget set the tool makes for the
  files of the members the request names, on a fresh copy of the store of each shard the history
  reads, revoked once. The reading is ``{"answer": "contains"}`` where a tied entry's bit is 1,
  ``{"answer": "does not contain"}`` where the history has rows, at least one is tied and no
  tied entry's bit is 1, ``{"answer": "not covered"}`` (no record) where the trace holds no
  such checkpoint or holds no row for any step of its history. A step with no row, an uncovered
  range and a row with no entry add nothing: the reading is never "may contain" and never a stop
  of its own. The tie is valid only where the run of each shard it uses says one process in
  packed order; else ``{"cannot_answer": <the tie text of the question's row, ``otherwise``>}``.
  A fault of the tool half (``f.map-deleted``, ``f.digest``) is put by the putter of
  ``OriginBlame``, once, on the fresh copy of the shard the fault names, before the revoke, and
  the step that reads it is told the store is faulted; on such a store the tool's own end with
  an error is ``{"stopped": True}``, and any other end is the failed Attempt of the step. A
  fault of the trace half (``f.trace-deleted``, ``f.trace-half``, ``l.trace-count``) is in the
  document ``trace_of`` gives, put by ``baselines.faults.apply_fault`` on the exported record.
- ``c08.step-8-holds-no-T``: ``{"rows_that_name_T": <the rows of that sample tied to an entry
  whose sources name the section registered for the file of gen/T>, "first_exposure":
  {"segment", "step"} of the first step of the history of E@9 with such a row}``; the field is
  read as it stands and no revoke is made.
- ``c09.exposure-free-restart``: ``{"exposure_free_restart": <name>}``, the latest checkpoint
  of the run, in step order, read "does not contain" with every earlier one read so, and None
  where the first is not.

A module that does not exist yet fails a test on an assertion, not on an import: the module is
looked up in the body of a test.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import os
import shutil
import tomllib
from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"

MODULE = "baselines.combo_originblame"
OB_MODULE = "baselines.originblame"
DRIVER = "baselines.ob_driver"
SUBJECT = "originblame+trace"
DRIVER_WORD = "ob_driver"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
EM_DASH = chr(0x2014)
LOCAL_MARKERS = ("/ho" + "me/", "Proj" + "ects/")
SESSION_WORDS = ("C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor")
REVIEW_WORDS = ("R2" + "A", "ROUND" + "2", "rul" + "ing")

LOCK = ["stand-in-lock", "--held"]
REQUESTS = {"withdraw A": ("sources/A.jsonl",), "withdraw C": ("sources/C.jsonl",)}
BINARY = "/stand-in/bin/ob"
PACKED = "pipeline/partitioned/shards/shard-1.bin"
DRIVER_VERBS = ("track", "revoke", "clean", "status", "generate-set")
ENTRY_FOLDER = (".ob", "token-index.gpt2")
OS_ERROR_TEXT = "No such file or directory (os error 2)"
PROGRAM_ERROR_TEXT = "Error: forget set could not be written"
EACH_SUCCESSOR = "each successor shard"
SHARD_NAME = r"(successor of )?[A-Za-z0-9-]+/shard-\d+"
BUILD_FILES = ("sources/A.jsonl", "sources/B.jsonl", "sources/C.jsonl", "sources/Z.jsonl",
               "gen/P.jsonl", "gen/T.jsonl")
Result = namedtuple("Result", "exit_code output")

# The routed questions of the subject, counted by hand from the groups of the table: the group
# and the number of its questions (its listed rows, each at the sizes it is asked at).
ROUTED_BY_HAND = (
    ("c01.positions", 11),
    ("c01.step-facts", 7),
    ("c01.E@9", 1),
    ("c02.checkpoints", 15),
    ("c03.spans", 4),
    ("c04.interrupted-write", 2),
    ("c04.count-499", 1),
    ("c04.trace-count", 2),
    ("c04.not-a-stop", 2),
    ("c05.unmutated.shards", 6),
    ("c05.unmutated.checkpoints", 7),
    ("c05.f.trace-deleted", 24),
    ("c05.f.trace-half", 24),
    ("c05.f.map-deleted", 24),
    ("c05.f.digest", 25),
    ("c05.withdraw-C.unmutated", 2),
    ("c05.unregistered-checkpoint", 1),
    ("c06.downstream", 6),
    ("c08.checkpoints", 7),
    ("c08.step-8-holds-no-T", 1),
    ("c09.artifact-actions", 8),
    ("c09.successors", 3),
    ("c09.exposure-free-restart", 1),
)
NO_JOIN_ROWS = frozenset({"c06.generated.downstream-of-translation", "c09.10k.successor-shards"})
ROUTED_TOTAL = 184
TRACE_ONLY = ("c01.positions", "c01.step-facts", "c05.unregistered-checkpoint")
TOOL_ALONE = ("c03.spans", "c04.interrupted-write", "c04.count-499", "c05.unmutated.shards",
              "c05.withdraw-C.unmutated", "c06.downstream", "c09.artifact-actions",
              "c09.successors")
WALK = ("c01.E@9", "c02.checkpoints", "c04.trace-count", "c04.not-a-stop",
        "c05.unmutated.checkpoints", "c08.checkpoints", "c08.step-8-holds-no-T",
        "c09.exposure-free-restart")
BY_KIND = ("c05.f.trace-deleted", "c05.f.trace-half", "c05.f.map-deleted", "c05.f.digest")
TRACE_FAULTS = ("f.trace-deleted", "f.trace-half", "l.trace-count")
FAULT_SHARDS = {"f.map-deleted": "E/shard-1", "f.digest": "E/shard-3"}

MISSING_ROW = {
    "files": "c10.c05.unmutated.files.originblame+trace",
    "mixtures": "c10.c05.unmutated.mixtures.originblame+trace",
    "listing": "c10.c05.listing.map-deleted.originblame+trace",
}

# Read from the rule of each row of the trace half in check10.toml ("the reading is the trace's
# rows for the run, segment, step or position of the listed row"; for the step facts "the reading
# is the value the listed row asks for, taken from the trace: ...").
WHOLE_RUN_ROWS = ("c01.E.diff", "c01.E-rr.diff", "c01.K.diff")
LINE_ROWS = ("c01.E.T", "c01.E.skipped-line", "c01.E.repeated-draw", "c01.E.shared-ordinal",
             "c01.E.split-document", "c01.E.shared-token", "c01.E.step8.first-half",
             "c01.K.epoch-boundary")
LAUNCH_VALUES = ("consumed_at_start", "first_step", "last_step")
NAMED_TRACE_VALUES = {
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
ROW_FIELDS = ("segment", "step", "sample_pos", "shard_position", "ordinal", "token_start",
              "token_end")
PLACE_KEYS = ("segment", "step", "sample_pos", "shard_position", "ordinal")


# --- The table -----------------------------------------------------------------------------


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _spec() -> dict:
    return _read_toml(SPEC_FILE)


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows


@lru_cache(maxsize=1)
def _group_of_listed() -> dict[str, str]:
    found = {}
    for group in _read_toml(TABLE_DIR / "check10.toml")["group"]:
        for listed in group["rows"]:
            found[listed] = group["id"]
    return found


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


def artifact_kind(artifact) -> str:
    import re

    if artifact is None or artifact == EACH_SUCCESSOR:
        return "set"
    if re.fullmatch(SHARD_NAME, artifact):
        return "shard"
    if artifact == "(listing)":
        return "listing"
    if "/mix-" in artifact:
        return "mixtures"
    if "@" in artifact:
        return "checkpoints"
    return "files"


def named_fields(facts, question) -> set[str]:
    """The fields the rule of the question's row names for a reading that answers, written by
    hand per group from the rule texts of ``check10.toml``."""
    listed, group = question.listed_row, question.group
    given = facts.core_given[listed]
    kind = artifact_kind(given.get("artifact"))
    if group in TRACE_ONLY:
        return set(NAMED_TRACE_VALUES[listed])
    if group == "c08.step-8-holds-no-T":
        return {"rows_that_name_T", "first_exposure"}
    if group == "c09.exposure-free-restart":
        return {"exposure_free_restart"}
    if group in WALK or (group in BY_KIND and kind == "checkpoints"):
        return {"answer"}
    if group in BY_KIND:
        return {"answer"} if kind == "shard" else {"cannot_answer"}
    # The tool half alone.
    if listed == "c03.spans.per-ordinal":
        return {"per_ordinal", "notes"}
    if listed == "c03.spans.total":
        return {"per_shard", "notes"}
    if listed in ("c04.kill-after-100", "c04.retry"):
        return {"records"}
    if kind not in ("shard", "set"):
        return {"cannot_answer"}
    if listed == "c09.fixture.shards.counts":
        return {"keep", "rewrite", "counts"}
    if listed in ("c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"):
        return {"keep", "rewrite"}
    if listed.endswith(".action"):
        return {"action"}
    return {"answer"}


def fields_problem(got_fields, wanted) -> str | None:
    """None when a reading holds exactly the wanted fields, else what differs."""
    got_fields, wanted = set(got_fields), set(wanted)
    if got_fields == wanted:
        return None
    return f"missing {sorted(wanted - got_fields)}, extra {sorted(got_fields - wanted)}"


def routed_counts_problem(facts) -> str | None:
    """None when the table's routed questions of the subject are the 184 counted by hand,
    group by group, else what differs."""
    if sum(count for _, count in ROUTED_BY_HAND) != ROUTED_TOTAL:
        return "the hand count is not 184"
    routed_groups = {r["given"]["group"] for r in facts.rows
                     if r["given"]["subject"] == SUBJECT and r.get("route")}
    if routed_groups != {group for group, _ in ROUTED_BY_HAND}:
        return f"the routed groups differ: {sorted(routed_groups ^ {g for g, _ in ROUTED_BY_HAND})}"
    if set(TRACE_ONLY) | set(TOOL_ALONE) | set(WALK) | set(BY_KIND) != routed_groups:
        return "the four kinds of route do not cover the routed groups"
    for group, count in ROUTED_BY_HAND:
        asked = sum(len(facts.sizes_by_listed[listed]) for listed in facts.group_by_id[group]["rows"])
        if asked != count:
            return f"{group}: {asked} questions in the table, {count} by hand"
    return None


# --- Hand-written records ----------------------------------------------------------------


def doc(ob, section: str, tokens: int):
    return ob.Document(section=section, tokens=tokens,
                       record={"id": f"{Path(section).stem}-{tokens}", "text": f"t {section}"},
                       file=PACKED)


def plain_documents(ob):
    return (doc(ob, "sources/B.jsonl", 4), doc(ob, "sources/Z.jsonl", 5), doc(ob, "sources/B.jsonl", 3))


def write_records(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"id": i, "text": "t " + i}) + "\n" for i in ids),
                    encoding="utf-8")


def shard(ob, base, name, documents, *, target=("store", "main"), layout=None, run=None,
          successor=False):
    build = base / "build"
    build.mkdir(parents=True, exist_ok=True)
    for relative in BUILD_FILES:
        if not (build / relative).exists():
            write_records(build / relative, [Path(relative).stem + "-001"])
    return ob.ShardSpec(name=name, target=target, layout=layout, run=run, successor=successor,
                        build=build, documents=tuple(documents))


def scenario_shards(ob, base):
    """The three shards of run E on the main store. Documents of shard 3, in ordinal order: 0 a
    plain record (4 tokens), 1 a record of A (6), 2 a record of the generation output T (5),
    3 a plain record (3). Their ranges in the shard's own index: [0,4), [4,10), [10,15), [15,18)."""
    third = (doc(ob, "sources/Z.jsonl", 4), doc(ob, "sources/A.jsonl", 6),
             doc(ob, "gen/T.jsonl", 5), doc(ob, "sources/B.jsonl", 3))
    return [shard(ob, base, "E/shard-1", plain_documents(ob)),
            shard(ob, base, "E/shard-2", plain_documents(ob)),
            shard(ob, base, "E/shard-3", third)]


def make_env(base: Path, work: Path) -> Path:
    """A copied environment inside the work folder shaped like the real ones: its
    ``bin/python`` is a link to a plain file in an interpreter folder outside the work
    folder. Nothing here starts it."""
    interpreters = base / "uv-python"
    real = interpreters / "cpython-3.12.13-linux-x86_64-gnu"
    (real / "bin").mkdir(parents=True)
    (real / "bin" / "python3.12").write_bytes(b"a stand-in for an interpreter binary\n")
    (real / "lib").mkdir()
    os.symlink(real.name, interpreters / "cpython-3.12-linux-x86_64-gnu")
    copy_folder = work / "venv-ob"
    (copy_folder / "bin").mkdir(parents=True)
    os.symlink(str(interpreters / "cpython-3.12-linux-x86_64-gnu" / "bin" / "python3.12"),
               copy_folder / "bin" / "python")
    return copy_folder


def tree_digest(folder: Path) -> dict:
    out: dict = {}
    for current, dirs, files in os.walk(folder):
        for name in dirs:
            out[str(Path(current, name).relative_to(folder))] = "dir"
        for name in files:
            out[str(Path(current, name).relative_to(folder))] = Path(current, name).read_bytes()
    return out


# --- The hand-written trace --------------------------------------------------------------

SAMPLE_TOKENS = 33
BATCH_SIZE = 4


def _launch(position, consumed, step_at_start, first, last, world=1):
    return {"position": position, "world_size": world, "data_parallel_size": world,
            "consumed_at_start": consumed, "step_at_start": step_at_start,
            "first_step": first, "last_step": last}


def _row(shard_name, position, ordinal, start=0, end=SAMPLE_TOKENS):
    return {"shard": shard_name, "shard_position": position, "ordinal": ordinal,
            "token_start": start, "token_end": end}


def _sample(pos, rank, rows, covered=None):
    return {"sample_pos": pos, "rank": rank, "sample_tokens": SAMPLE_TOKENS,
            "covered_tokens": SAMPLE_TOKENS if covered is None else covered, "rows": rows}


def _plain_rows(step, pos, shards):
    """One plain row: a document of no member of A and not of T, from one of the shards."""
    name, position = shards[(step + pos) % len(shards)]
    return [_row(name, position, (step + pos) % 3)]


def _steps(first, last, shards, special=None, rank_of=lambda pos: 0):
    """Steps ``first`` to ``last`` of four samples, plain rows; ``special`` maps (step, pos) to
    ``(rows, covered)``."""
    special = special or {}
    steps = []
    for number in range(first, last + 1):
        samples = []
        for pos in range(BATCH_SIZE):
            rows, covered = special.get((number, pos), (_plain_rows(number, pos, shards), None))
            samples.append(_sample(pos, rank_of(pos), rows, covered))
        steps.append({"step": number, "consumed_samples": BATCH_SIZE, "samples": samples})
    return steps


def _segment(name, run, parent, shards, checkpoints, launches, steps):
    return {"segment": name, "run": run, "parent_checkpoint": parent, "shards": list(shards),
            "checkpoints": [{"name": n, "step": s} for n, s in checkpoints],
            "launches": launches, "steps": steps}


E_SHARDS = [("E/shard-1", 0), ("E/shard-2", 1)]
E_SHARDS_ONLY_2 = [("E/shard-2", 1)]
E_SHARDS_ONLY_1 = [("E/shard-1", 0)]


def world_trace() -> dict:
    """Run E, its descendant D, its branch Y and run K, as the fixture spec lays them out. The
    row of the last sample of step 4 of E/seg-1 is document 1 of shard 1, tokens 6 to 39 (the
    spec's shared ordinal). In E/seg-2, step 7 reads document 1 of shard 3 (a record of A) and
    document 2 of shard 3 (a record of T); the last sample of step 8 holds 16 of its 33 tokens
    (document 0 of shard 3, tokens 19 to 35). Run K reads the record of A through one sample of
    rank 1 at step 7. Every other row is a plain record of shard 1 or shard 2."""
    e1 = _segment(
        "E/seg-1", "E", None, ["shard-1", "shard-2"], [("E@6", 6)], [_launch(0, 0, 0, 1, 6)],
        _steps(1, 6, E_SHARDS, {(4, 3): ([_row("E/shard-1", 0, 1, 6, 39)], None)}))
    e2 = _segment(
        "E/seg-2", "E", "E@6", ["shard-1", "shard-2", "shard-3"], [("E@9", 9)],
        [_launch(0, 24, 6, 7, 12)],
        _steps(7, 12, E_SHARDS_ONLY_2, {
            (7, 0): ([_row("E/shard-3", 2, 1)], None),
            (7, 1): ([_row("E/shard-3", 2, 2)], None),
            (8, 3): ([_row("E/shard-3", 2, 0, 19, 35)], 16)}))
    d1 = _segment("D/seg-1", "D", "E@9", ["shard-1", "shard-2"], [("D@3", 3)],
                  [_launch(0, 36, 9, 1, 3)], _steps(1, 3, E_SHARDS_ONLY_1))
    y1 = _segment("Y/seg-1", "Y", "E@6", ["shard-1", "shard-2", "shard-3"], [("Y@6", 6)],
                  [_launch(0, 24, 6, 1, 6)], _steps(1, 6, E_SHARDS_ONLY_2))
    k1 = _segment(
        "K/seg-1", "K", "E@6", ["shard-3", "shard-2"],
        [("K@3", 3), ("K@8", 8), ("K@40", 40)],
        [_launch(0, 0, 0, 1, 3, world=2), _launch(1, 12, 3, 4, 40, world=2)],
        _steps(1, 40, E_SHARDS_ONLY_2, {(7, 2): ([_row("E/shard-3", 0, 1)], None)},
               rank_of=lambda pos: pos // 2))
    return {"segments": [e1, e2, d1, y1, k1]}


def segment_of(document: dict, name: str) -> dict:
    (found,) = [s for s in document["segments"] if s["segment"] == name]
    return found


def step_of(document: dict, name: str, number: int) -> dict:
    (found,) = [s for s in segment_of(document, name)["steps"] if s["step"] == number]
    return found


def trace_for(state, document: dict | None = None) -> dict:
    """The document a starting state reads: a registration that wrote no trace has segments
    with no step; a fault of the trace half is put on the record of E/seg-1."""
    from baselines.faults import apply_fault

    held = copy.deepcopy(world_trace() if document is None else document)
    if state in ("c2.ii", "c2.iii"):
        for seg in held["segments"]:
            seg["steps"] = []
    elif state == "c2.iv":
        for seg in held["segments"]:
            if seg["run"] == "K":
                seg["steps"] = []
    elif state in TRACE_FAULTS:
        held["segments"] = [apply_fault(state, "trace", seg) if seg["segment"] == "E/seg-1" else seg
                            for seg in held["segments"]]
    return held


def trace_of_factory(document: dict | None = None):
    def trace_of(listed_row: str):
        return trace_for(_listed_rows()[listed_row]["given"].get("state"), document)
    return trace_of


# --- The stand-in for the sandboxed driver (a trimmed copy of the tool's own tests') --------


class Handle:
    """A started ``track``: one line per call that returned; ``stop`` ends it."""

    def __init__(self, tool, store: Path, documents: list[dict], pid_name: str):
        self.tool, self.store, self.documents, self.pid_name = tool, store, documents, pid_name
        self.pid = 4242
        self.sent = 0
        self.stopped = False

    def readline(self) -> str:
        if self.stopped:
            return ""
        if self.sent < len(self.documents):
            item = self.documents[self.sent]
            if not self.tool.write_at_end:
                self.tool.append_entry(self.store, self.pid_name, item)
            self.sent += 1
            return f"tracked {self.sent}\n"
        if self.sent == len(self.documents):
            self.sent += 1
            return "run " + json.dumps(self.tool.run_statement()) + "\n"
        return ""

    def stop(self):
        self.stopped = True
        self.tool.stops.append(self.pid)
        if self.tool.write_at_end:
            for item in self.documents[:self.sent]:
                self.tool.append_entry(self.store, self.pid_name, item)
        self.tool.mark_partial(self.store, self.pid_name)
        return self.tool.stop_signal

    def wait(self) -> Result:
        if self.stopped:
            return Result(self.tool.stop_exit, self.tool.stop_output)
        return Result(0, "")


class FakeTool:
    """The sandboxed driver, as the files the tool leaves: a store holds ``driver_meta.json``
    (the section hash of each registered file), one PID file per writer until ``clean`` merges
    it into ``.ob/token-index.gpt2/<NNN>``, and ``state.json`` (the stand-in's bookkeeping).
    Options: ``processes`` and ``say_run`` (what the run says), ``error_after`` (fault ids whose
    store makes ``revoke`` end with an error of the tool: on a store the step is told is faulted
    the tool's own line, on any other the driver's failure line), ``faulted_program_error``
    (verbs of the tool's program that end with an error on a store the step is told is faulted),
    ``fail`` (verb to (exit code, output): an end the driver did not write), ``stop_exit``,
    ``stop_signal``, ``stop_output`` and ``write_at_end`` (the stopped writer)."""

    def __init__(self, *, processes=1, say_run=True, error_after=(), faulted_program_error=(),
                 fail=None, stop_exit=-15, stop_signal=15, stop_output="", write_at_end=False,
                 fail_track_in=()):
        self.processes, self.say_run = processes, say_run
        self.error_after, self.faulted_program_error = set(error_after), set(faulted_program_error)
        self.fail = dict(fail or {})
        self.stop_exit, self.stop_signal, self.stop_output = stop_exit, stop_signal, stop_output
        self.write_at_end = write_at_end
        self.fail_track_in = set(fail_track_in)
        self.pid_names = ["2", "3", "4"]
        self.calls: list[dict] = []
        self.stops: list[int] = []

    def parse(self, argv):
        argv = list(argv)
        assert argv[: len(LOCK)] == LOCK, "a step must start with the lock prefix"
        index = argv.index(DRIVER_WORD)
        verb, rest = argv[index + 1], argv[index + 2:]
        options, positional, at = {}, [], 0
        while at < len(rest):
            if rest[at] in ("--store", "--plan", "--out"):
                options[rest[at][2:]] = rest[at + 1]
                at += 2
            elif rest[at] == "--faulted":
                options["faulted"] = True
                at += 1
            else:
                positional.append(rest[at])
                at += 1
        return verb, options, positional

    @staticmethod
    def load(store: Path) -> dict:
        path = store / "state.json"
        if not path.exists():
            return {"order": [], "merged": {}, "revokes": 0, "faulted": None, "partial": None,
                    "tracks": 0}
        return json.loads(path.read_text())

    @staticmethod
    def save(store: Path, state: dict) -> None:
        store.mkdir(parents=True, exist_ok=True)
        (store / "state.json").write_text(json.dumps(state, sort_keys=True))

    @staticmethod
    def pid_path(store: Path, name: str) -> Path:
        return store / ".ob" / f"token-index.gpt2.{name}"

    @staticmethod
    def read_rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def entry_files(store: Path) -> list[Path]:
        folder = store.joinpath(*ENTRY_FOLDER)
        if not folder.is_dir():
            return []
        return sorted(folder.iterdir(), key=lambda path: int(path.name))

    def entries(self, store: Path) -> list[dict]:
        return [row for path in self.entry_files(store) for row in self.read_rows(path)]

    def rewrite_entries(self, store: Path, rows: list[dict]) -> None:
        for path in self.entry_files(store):
            path.unlink()
        if rows:
            self.write_rows(store.joinpath(*ENTRY_FOLDER) / "001", rows)

    @staticmethod
    def ensure_meta(store: Path, plan: dict) -> None:
        path = store / "driver_meta.json"
        if not path.exists():
            store.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(
                {"phase": "track", "sections": {name: "h:" + name for name in plan.get("files", [])}},
                sort_keys=True))

    def run_statement(self) -> dict:
        return {"processes": self.processes, "order": "packed"}

    def next_pid_name(self, state: dict) -> str:
        return self.pid_names[min(state["tracks"], len(self.pid_names) - 1)]

    @staticmethod
    def entry(item: dict) -> dict:
        return {"sources": ["h:" + item["section"]], "token_count": item["tokens"],
                "tokenizer": "gpt2", "revoked": False}

    def append_entry(self, store: Path, pid_name: str, item: dict) -> None:
        path = self.pid_path(store, pid_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.entry(item), sort_keys=True) + "\n")

    def mark_partial(self, store: Path, pid_name: str) -> None:
        state = self.load(store)
        state["partial"] = pid_name
        self.save(store, state)

    def start(self, argv) -> Handle:
        verb, options, _ = self.parse(argv)
        assert verb == "track"
        store = Path(options["store"])
        self.calls.append({"verb": verb, "store": store, "options": options, "positional": [],
                           "argv": list(argv), "started": True})
        plan = json.loads(Path(options["plan"]).read_text())
        self.ensure_meta(store, plan)
        state = self.load(store)
        name = self.next_pid_name(state)
        if name not in state["order"]:
            state["order"].append(name)
        state["tracks"] += 1
        self.save(store, state)
        path = self.pid_path(store, name)
        if not path.exists():
            self.write_rows(path, [])
        return Handle(self, store, plan["documents"], name)

    def run(self, argv) -> Result:
        verb, options, positional = self.parse(argv)
        store = Path(options["store"])
        self.calls.append({"verb": verb, "store": store, "options": options,
                           "positional": positional, "argv": list(argv)})
        if verb not in DRIVER_VERBS:
            return self.tool_failure(2, f"unknown verb {verb}")
        if verb in self.fail:
            code, text = self.fail[verb]
            return Result(code, text.replace("{store}", str(store)))
        state = self.load(store)
        if verb == "track":
            return self.do_track(store, state, options)
        if verb == "clean":
            return self.do_clean(store, state)
        if verb == "status":
            rows = self.entries(store)
            revoked = sum(1 for row in rows if row["revoked"])
            return Result(0, f"Tokenizer: gpt2\nEntries: {len(rows)} ({len(rows) - revoked} active, "
                             f"{revoked} revoked)\nTotal tokens: {sum(r['token_count'] for r in rows)}\n")
        if verb == "revoke":
            return self.do_revoke(store, state, positional, bool(options.get("faulted")))
        if verb in self.faulted_program_error and options.get("faulted"):
            return self.tool_error(1, PROGRAM_ERROR_TEXT, verb)
        return self.do_generate(store, options["out"])

    @staticmethod
    def tool_failure(code: int, text: str) -> Result:
        import importlib

        marker = getattr(importlib.import_module(DRIVER), "FAILURE_MARKER", None)
        assert marker, f"{DRIVER} has no FAILURE_MARKER"
        return Result(code, f"{marker} {text}\n")

    @staticmethod
    def tool_error(code: int, text: str, call: str = "track") -> Result:
        import importlib

        marker = getattr(importlib.import_module(DRIVER), "TOOL_ERROR_MARKER", None)
        assert marker, f"{DRIVER} has no TOOL_ERROR_MARKER"
        return Result(code, f"{marker} {call} raised RuntimeError: {text.rstrip()}\n")

    def put_fault(self, store: Path, fault_id: str) -> None:
        from baselines import faults

        record = {"sections": [], "documents": [], "entries": self.entries(store)}
        changed = faults.apply_fault(fault_id, "originblame", record)
        self.rewrite_entries(store, changed["entries"])
        state = self.load(store)
        state["faulted"] = fault_id
        self.save(store, state)

    def do_track(self, store, state, options) -> Result:
        plan = json.loads(Path(options["plan"]).read_text())
        if store.name in self.fail_track_in:
            return Result(1, "sandbox failed before the driver started\n")
        self.ensure_meta(store, plan)
        name = self.next_pid_name(state)
        if name not in state["order"]:
            state["order"].append(name)
        state["tracks"] += 1
        self.save(store, state)
        path = self.pid_path(store, name)
        rows = [] if not path.exists() else self.read_rows(path)
        lines = []
        for number, item in enumerate(plan["documents"], start=1):
            rows.append(self.entry(item))
            lines.append(f"tracked {number}")
        self.write_rows(path, rows)
        if self.say_run:
            lines.append("run " + json.dumps(self.run_statement()))
        return Result(0, "\n".join(lines) + "\n")

    def do_clean(self, store, state) -> Result:
        merged_files = merged_entries = 0
        for name in list(state["order"]):
            path = self.pid_path(store, name)
            if not path.exists():
                continue
            merged = state["merged"].setdefault(name, f"{len(state['merged']) + 1:03d}")
            rows = self.read_rows(path)
            self.write_rows(store.joinpath(*ENTRY_FOLDER) / merged, rows)
            path.unlink()
            merged_files += 1
            merged_entries += len(rows)
        self.save(store, state)
        lines = ["Merged 0 document records from PID files", "Deleted 0 PID files"]
        if merged_entries:
            lines += [f"Merged {merged_entries} token-index entries",
                      f"Deleted {merged_files} token-index PID files"]
        lines.append("Archived 0 revoked records")
        return Result(0, "\n".join(lines) + "\n")

    def do_revoke(self, store, state, files, faulted=False) -> Result:
        if state["faulted"] in self.error_after:
            call = "token_index_revoke_by_sources"
            if faulted:
                return self.tool_error(5, OS_ERROR_TEXT, call)
            return self.tool_failure(6, f"{call} raised RuntimeError: {OS_ERROR_TEXT}")
        sections = json.loads((store / "driver_meta.json").read_text())["sections"]
        wanted = {sections.get(path, "h:" + path) for path in files}
        for path in self.entry_files(store):
            rows = self.read_rows(path)
            for row in rows:
                if wanted & set(row["sources"]):
                    row["revoked"] = not row["revoked"]
            self.write_rows(path, rows)
        state["revokes"] += 1
        self.save(store, state)
        return Result(0, "revoked\n")

    def do_generate(self, store, out) -> Result:
        flags = [row["revoked"] for row in self.entries(store)]
        data = bytearray((len(flags) + 7) // 8)
        for place, flag in enumerate(flags):
            if flag:
                data[place >> 3] |= 1 << (place & 7)
        Path(out).write_bytes(bytes(data))
        return Result(0, f"Forget set written to {out} ({len(data)} bytes, {sum(flags)} revoked entries)\n")


class MostSignificantFirstTool(FakeTool):
    """A stand-in that writes the forget set with the wrong bit order: the known-bad input of the
    check of the stand-in below."""

    def do_generate(self, store, out) -> Result:
        flags = [row["revoked"] for row in self.entries(store)]
        data = bytearray((len(flags) + 7) // 8)
        for place, flag in enumerate(flags):
            if flag:
                data[place >> 3] |= 0x80 >> (place & 7)
        Path(out).write_bytes(bytes(data))
        return Result(0, "")


class FaultPutter:
    """What the runner hands the subject: the fault put to the entry files of a fresh copy, with
    the table's own ``faults`` module, before the revoke."""

    def __init__(self, tool: FakeTool):
        self.tool = tool
        self.puts: list[tuple[Path, str]] = []

    def __call__(self, copy_folder: Path, fault_id: str) -> None:
        self.puts.append((copy_folder, fault_id))
        self.tool.calls.append({"verb": "fault", "store": copy_folder, "options": {"fault": fault_id},
                                "positional": [], "argv": []})
        self.tool.put_fault(copy_folder, fault_id)


class FakeRunner:
    def __init__(self, tool: FakeTool):
        self.tool = tool

    def run(self, argv, **_):
        return self.tool.run(argv)

    def start(self, argv, **_):
        return self.tool.start(argv)


def calls_of(tool: FakeTool, verb: str) -> list[dict]:
    return [c for c in tool.calls if c["verb"] == verb]


# --- The two halves, built over the stand-in -----------------------------------------------


@dataclasses.dataclass
class World:
    ob: object
    tool: FakeTool
    putter: FaultPutter
    combo: object
    failed: tuple
    specs: list


def make_ob(need_module, base: Path, shards, tool=None):
    ob = need_module(OB_MODULE)
    work = base / "work"
    work.mkdir(parents=True, exist_ok=True)
    env = make_env(base, work)
    tool = tool or FakeTool()
    putter = FaultPutter(tool)
    subject = ob.OriginBlame(work=work, env_dir=env, lock_prefix=LOCK, runner=FakeRunner(tool),
                             shards=tuple(shards), requests=dict(REQUESTS), ob_binary=BINARY,
                             fault_putter=putter)
    return subject, tool, putter


def make_world(need_module, base: Path, *, shards_of=scenario_shards, tool=None, document=None,
               trace_of=None, tracked=True) -> World:
    """The tool half over the stand-in, tracked, and the combination over it and the trace."""
    ob = need_module(OB_MODULE)
    base.mkdir(parents=True, exist_ok=True)
    specs = shards_of(ob, base)
    subject, tool, putter = make_ob(need_module, base, specs, tool)
    failed = subject.track_all() if tracked else ()
    combo_module = need_module(MODULE)
    combo = combo_module.OriginBlameTrace(
        subject, trace_of=trace_of or trace_of_factory(document), place_of=place_of)
    return World(subject, tool, putter, combo, failed, list(specs))


# --- The questions of the table ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _all_combo_questions() -> dict:
    import importlib

    table = importlib.import_module("baselines.table10")
    found = {}
    for question in table.expand(table.load_table10()):
        if question.subject == SUBJECT:
            found[(question.listed_row, question.size)] = question
    return found


def combo_questions(need_module) -> dict:
    need_module("baselines.table10")
    return _all_combo_questions()


def question_of(need_module, listed_row, size=None):
    return combo_questions(need_module)[(listed_row, size)]


def expectation(need_module, listed_row, size=None):
    expected = need_module("baselines.expected")
    return expected.expected_for(listed_row, size).expect


def graded(need_module, question, raw, *, expect=None):
    grade = need_module("baselines.grade")
    interfaces = need_module("baselines.interfaces")
    expect = expectation(need_module, question.listed_row, question.size) if expect is None else expect
    reading = interfaces.Reading(question_id=question.id, raw=raw, route=question.route)
    return grade.grade(question, expect, reading)


def raw_of(world: World, need_module, listed_row, size=None):
    got = world.combo.read(question_of(need_module, listed_row, size))
    return got.raw


def ob_module_record():
    import importlib

    return importlib.import_module("baselines.record")


def shards_for(ob, base, facts, question, *, with_a=False):
    """The shards a question of the tool half needs, written from the table's own ``given`` of
    its listed row: one shard for an artifact that is a shard, three for a set or a run."""
    given = facts.core_given[question.listed_row]
    record = ob_module_record()
    target = record.attempt_target(question)
    successor = "successor" in question.listed_row
    artifact = given.get("artifact")
    if artifact is not None and artifact_kind(artifact) == "shard":
        documents = plain_documents(ob)
        if with_a:
            documents = (documents[0], doc(ob, "sources/A.jsonl", 6), *documents[1:])
        return [shard(ob, base, artifact, documents, target=target, layout=given.get("layout"),
                      run=None, successor=successor)]
    if artifact_kind(artifact) != "set":
        return []
    layouts = list(given.get("layouts") or [given.get("layout")])
    found = []
    for layout in layouts:
        for number in (1, 2, 3):
            name = (f"{layout or 'set'}/shard-{number}" if given.get("run") is None
                    else f"{given['run']}/shard-{number}")
            documents = plain_documents(ob)
            if with_a and number == 3:
                documents = (documents[0], doc(ob, "sources/A.jsonl", 6), *documents[1:])
            found.append(shard(ob, base, name, documents, target=target, layout=layout,
                               run=given.get("run"), successor=successor))
    return found


def pool_of(need_module, base: Path, facts, questions) -> list:
    """The shards every routed question needs: the three of the scenario first, then one shard
    of a name on a target (where one question names the run and layout of it, that one)."""
    ob = need_module(OB_MODULE)
    pool: dict = {}
    for spec in scenario_shards(ob, base):
        pool[(spec.name, spec.target, spec.successor)] = spec
    for question in questions:
        for spec in shards_for(ob, base, facts, question, with_a=True):
            key = (spec.name, spec.target, spec.successor)
            if key not in pool or (pool[key].run is None and spec.run is not None):
                pool[key] = spec
    scratch = shard(ob, base, "c04/shard-500", tuple(doc(ob, "sources/B.jsonl", 3) for _ in range(500)),
                    target=("store", "scratch-500"))
    pool[(scratch.name, scratch.target, scratch.successor)] = scratch
    return list(pool.values())


def routed_questions(need_module):
    return [q for q in combo_questions(need_module).values() if q.route is not None]


# --- The stand-in holds what it says ---------------------------------------------------------


def stand_in_bits(tool, base) -> tuple[bytes, bytes]:
    """The forget set after one revoke of the file of A, then after a second one, of a store of
    ten entries whose last entry is the one of A."""
    base.mkdir(parents=True, exist_ok=True)
    store = base / "s"
    plan = base / "plan.json"
    items = [{"section": "sources/B.jsonl", "tokens": 1, "record": {"id": "r"}, "file": PACKED}] * 9 + [
        {"section": "sources/A.jsonl", "tokens": 1, "record": {"id": "r"}, "file": PACKED}]
    plan.write_text(json.dumps({"files": ["sources/A.jsonl", "sources/B.jsonl"], "documents": items}))
    tool.run(LOCK + [DRIVER_WORD, "track", "--store", str(store), "--plan", str(plan)])
    tool.run(LOCK + [DRIVER_WORD, "clean", "--store", str(store)])
    out = base / "bits"
    found = []
    for _ in range(2):
        tool.run(LOCK + [DRIVER_WORD, "revoke", "--store", str(store), "sources/A.jsonl"])
        tool.run(LOCK + [DRIVER_WORD, "generate-set", "--store", str(store), "--out", str(out)])
        found.append(out.read_bytes())
    return found[0], found[1]


def test_the_stand_in_un_revokes_on_a_second_call_and_writes_the_bits_least_significant_first(tmp_path):
    assert stand_in_bits(FakeTool(), tmp_path / "a") == (bytes([0x00, 0x02]), bytes([0x00, 0x00]))
    assert stand_in_bits(MostSignificantFirstTool(), tmp_path / "b")[0] == bytes([0x00, 0x40])


def test_the_stand_in_refuses_a_step_that_is_not_behind_the_lock(tmp_path):
    with pytest.raises(AssertionError):
        FakeTool().run(["bwrap", "--", "python", "-m", DRIVER_WORD, "status", "--store", str(tmp_path)])


def _scratch_ob(need_module, base: Path, tool: FakeTool):
    ob = need_module(OB_MODULE)
    spec = shard(ob, base, "c04/shard-500", tuple(doc(ob, "sources/B.jsonl", 3) for _ in range(500)),
                 target=("store", "scratch-500"))
    subject, _, _ = make_ob(need_module, base, [spec], tool)
    assert subject.track_all() == ()
    return subject


def test_the_stand_in_gives_the_tool_half_the_ends_the_combinations_tests_rely_on(need_module, tmp_path):
    # The tool half alone, over the stand-in: the premises of the tests of the combination below.
    interfaces = need_module("baselines.interfaces")
    ob = need_module(OB_MODULE)
    for label, tool in (("revoke", FakeTool(error_after=("f.digest",))),
                        ("program", FakeTool(faulted_program_error=("generate-set",)))):
        subject, _, _ = make_ob(need_module, tmp_path / label, scenario_shards(ob, tmp_path / label), tool)
        assert subject.track_all() == ()
        assert subject.read(question_of(need_module, "c05.f.digest.E/shard-3")).raw == {"stopped": True}, label
    base = tmp_path / "fail"
    subject, _, _ = make_ob(need_module, base, scenario_shards(ob, base),
                            FakeTool(fail={"revoke": (7, "the lock timed out")}))
    assert subject.track_all() == ()
    got = subject.read(question_of(need_module, "c05.unmutated.E/shard-3"))
    assert isinstance(got, interfaces.Attempt) and got.exit_code == 7
    # A stop that lands and the rows that showed while the writer ran; a stop that did not land.
    got = _scratch_ob(need_module, tmp_path / "a", FakeTool()).read(
        question_of(need_module, "c04.kill-after-100"))
    assert got.raw["records"]["rows_showed"] == "while the writer ran"
    got = _scratch_ob(need_module, tmp_path / "b", FakeTool(write_at_end=True)).read(
        question_of(need_module, "c04.kill-after-100"))
    assert got.raw["records"]["rows_showed"] == "only at its end"
    for listed in ("c04.kill-after-100", "c04.retry"):
        got = _scratch_ob(need_module, tmp_path / listed, FakeTool(stop_exit=0)).read(
            question_of(need_module, listed))
        assert isinstance(got, interfaces.Attempt) and got.exit_code == 0, listed
        assert "did not land" in got.output, listed


def _ranges(counts) -> list[tuple[int, int]]:
    found, at = [], 0
    for count in counts:
        found.append((at, at + count))
        at += count
    return found


def test_the_hand_written_world_has_the_rows_ranges_and_documents_the_scenario_needs(need_module, tmp_path):
    history_rows = need_module("baselines.history").history_rows
    faults = need_module("baselines.faults")
    ob = need_module(OB_MODULE)
    document = world_trace()
    held = {name: history_rows(document, name) for name in ("E@6", "E@9", "D@3", "Y@6", "K@3", "K@8", "K@40")}
    assert {name: len(rows) for name, rows in held.items()} == {
        "E@6": 24, "E@9": 36, "D@3": 48, "Y@6": 48, "K@3": 36, "K@8": 56, "K@40": 184}
    in_shard_3 = {name: sorted((r.segment, r.step, r.ordinal) for r in rows if r.shard == "E/shard-3")
                  for name, rows in held.items()}
    assert in_shard_3["E@6"] == [] and in_shard_3["Y@6"] == [] and in_shard_3["K@3"] == []
    assert in_shard_3["E@9"] == [("E/seg-2", 7, 1), ("E/seg-2", 7, 2), ("E/seg-2", 8, 0)]
    assert in_shard_3["D@3"] == in_shard_3["E@9"]
    assert in_shard_3["K@8"] == [("K/seg-1", 7, 1)] == in_shard_3["K@40"]
    assert history_rows(document, "E@12") is None
    # The documents of shard 3 and their ranges in its own index, and the ranges of its entries
    # after the fault f.digest changed the first digit of the first count (4 to 5).
    third = scenario_shards(ob, tmp_path)[2]
    sizes = [d.tokens for d in third.documents]
    assert _ranges(sizes) == [(0, 4), (4, 10), (10, 15), (15, 18)]
    assert [d.section for d in third.documents] == [
        "sources/Z.jsonl", "sources/A.jsonl", "gen/T.jsonl", "sources/B.jsonl"]
    entries = [{"sources": ["h:" + d.section], "token_count": d.tokens, "tokenizer": "gpt2", "revoked": False}
               for d in third.documents]
    changed = faults.apply_fault("f.digest", "originblame", {"entries": entries})["entries"]
    after = _ranges([e["token_count"] for e in changed])
    assert after == [(0, 5), (5, 11), (11, 16), (16, 19)]
    assert not set(after) & set(_ranges(sizes))
    # The documents of the starting states and the faults of the trace half.
    assert all(seg["steps"] == [] for seg in trace_for("c2.iii")["segments"])
    assert [seg["segment"] for seg in trace_for("c2.iv")["segments"] if seg["steps"] == []] == ["K/seg-1"]
    assert [s["step"] for s in segment_of(trace_for("f.trace-deleted"), "E/seg-1")["steps"]] == [1, 2, 4, 5, 6]
    half = step_of(trace_for("f.trace-half"), "E/seg-1", 4)["samples"][3]
    assert half["covered_tokens"] == 16 and half["rows"] == [_row("E/shard-1", 0, 1, 6, 22)]
    short = step_of(trace_for("l.trace-count"), "E/seg-1", 5)
    assert len(short["samples"]) == 3 and short["consumed_samples"] == 3
    assert world_trace() == trace_for("unmutated") == trace_for(None)


# --- The module ----------------------------------------------------------------------------


def test_the_module_exists_and_holds_the_names_asked(need_module, tmp_path):
    combo_module = need_module(MODULE)
    interfaces = need_module("baselines.interfaces")
    assert hasattr(combo_module, "OriginBlameTrace")
    world = make_world(need_module, tmp_path)
    assert isinstance(world.combo, interfaces.Subject)
    assert world.combo.name == SUBJECT == interfaces.SUBJECTS[5]
    assert callable(world.combo.read) and callable(world.combo.read_by_layout)


def _source_of(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


def test_the_module_imports_nothing_of_the_tool_or_the_world_helpers(need_module):
    module = need_module(MODULE)
    tree = ast.parse(_source_of(module))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, roots & FORBIDDEN_IMPORT_ROOTS


def test_the_module_holds_no_dash_no_local_path_and_no_name_of_this_build(need_module):
    text = _source_of(need_module(MODULE))
    assert EM_DASH not in text
    for word in LOCAL_MARKERS + SESSION_WORDS + REVIEW_WORDS:
        assert word not in text, f"the module holds {word!r}"


# --- The mechanical bar: every routed question, the fields its rule names ---------------------


def test_the_routed_questions_number_184_by_hand_and_in_the_table(facts):
    assert routed_counts_problem(facts) is None
    assert facts.questions_per_subject == 391


def test_the_field_check_catches_a_reader_that_drops_one_named_field():
    # The known-bad input of the checks below: the counts of a shard list with one field dropped.
    assert fields_problem({"keep", "rewrite", "counts"}, {"keep", "rewrite", "counts"}) is None
    assert fields_problem({"keep", "rewrite"}, {"keep", "rewrite", "counts"}) == "missing ['counts'], extra []"
    assert fields_problem({"rows_that_name_T"}, {"rows_that_name_T", "first_exposure"}) == \
        "missing ['first_exposure'], extra []"
    assert fields_problem({"answer", "diff"}, {"answer"}) == "missing [], extra ['diff']"


def test_every_routed_group_of_the_combination_is_one_it_reads_the_rows_of_an_extension_included(need_module):
    """The first dry pass with the generating-model extension on (18 Nov 2026) stopped at the first routed row of
    ``c05.withdraw-C-on`` with "the combination has no reading": the groups the combination reads were the core groups
    only. Every group that holds a routed row of this subject, an extension row too, is one of the six kinds the
    combination reads, so that no row of the table the pass puts reaches the end of ``read`` unread."""
    combo = need_module("baselines.combo_originblame")
    table = need_module("baselines.table10").load_table10()
    routed = {row.group for row in table.rows if row.subject == SUBJECT and row.route}
    assert {row.group for row in table.rows if row.extension and row.subject == SUBJECT and row.route}, \
        "no row of an extension is routed: the check does not reach the extension rows"
    read = (set(combo.TRACE_GROUPS) | set(combo.TOOL_GROUPS) | set(combo.WALK_GROUPS) | set(combo.KIND_GROUPS)
            | {combo.STEP_GROUP, combo.RESTART_GROUP})
    assert routed <= read, f"routed groups the combination does not read: {sorted(routed - read)}"


def test_every_routed_question_reads_exactly_the_fields_its_rule_names(need_module, facts, tmp_path):
    interfaces = need_module("baselines.interfaces")
    questions = routed_questions(need_module)
    assert len(questions) == ROUTED_TOTAL
    world = make_world(need_module, tmp_path, shards_of=lambda ob, base: pool_of(
        need_module, base, facts, questions))
    assert world.failed == ()
    problems = []
    for question in questions:
        wanted = named_fields(facts, question)
        if question.listed_row in NO_JOIN_ROWS:
            # Changed 17 Nov 2026: no join of the shards is given by the table; the tool half raises.
            if facts.core_given[question.listed_row].get("layouts"):
                assert all(r.raw == {"cannot_answer": need_module("baselines.dolma").NO_JOIN_TEXT}
                           for r in world.combo.read_by_layout(question).values() if r is not None)
            else:
                with pytest.raises(ValueError, match="no join of the shards"):
                    world.combo.read(question)
            continue
        if facts.core_given[question.listed_row].get("layouts"):
            results = list(world.combo.read_by_layout(question).values())
        else:
            results = [world.combo.read(question)]
        for got in results:
            assert isinstance(got, interfaces.Reading) and got.question_id == question.id, question.id
            assert got.route == question.route
            problem = fields_problem(set(got.raw), wanted)
            if problem:
                problems.append(f"{question.id}: {problem}")
    assert not problems, problems[:5]


def test_every_other_question_records_none_and_starts_nothing(need_module, facts, tmp_path):
    interfaces = need_module("baselines.interfaces")
    questions = [q for q in combo_questions(need_module).values() if q.route is None]
    assert len(questions) == facts.questions_per_subject - ROUTED_TOTAL == 207
    world = make_world(need_module, tmp_path)
    before = len(world.tool.calls)
    for question in questions:
        got = world.combo.read(question)
        assert isinstance(got, interfaces.Reading)
        assert got.raw is None and got.route is None and got.question_id == question.id, question.id
    assert len(world.tool.calls) == before


def test_the_trace_half_alone_reads_the_trace_subjects_values_and_starts_nothing(need_module, facts, tmp_path):
    trace = need_module("baselines.trace")
    world = make_world(need_module, tmp_path, tracked=False)
    document = world_trace()
    subject = trace.TraceSubject(document, place_of=place_of)
    seen = 0
    for question in routed_questions(need_module):
        if question.group not in TRACE_ONLY:
            continue
        seen += 1
        got = world.combo.read(question)
        wanted = subject.read(question)
        assert got.raw == wanted.raw and got.route == question.route, question.id
        assert fields_problem(set(got.raw), named_fields(facts, question)) is None, question.id
    assert seen == 11 + 7 + 1
    assert world.tool.calls == []
    # A checkpoint the trace does not hold is the answer "not covered".
    assert raw_of(world, need_module, "c05.unregistered-checkpoint") == {"answer": "not covered"}


def test_the_tool_half_alone_reads_what_the_tool_subject_reads(need_module, facts, tmp_path):
    questions = [q for q in routed_questions(need_module)
                 if q.group in TOOL_ALONE or (q.group in BY_KIND and artifact_kind(
                     facts.core_given[q.listed_row].get("artifact")) == "shard")]
    assert len(questions) == 4 + 2 + 1 + 6 + 2 + 6 + 8 + 3 + 4 * 3
    world = make_world(need_module, tmp_path / "combo", shards_of=lambda ob, base: pool_of(
        need_module, base, facts, routed_questions(need_module)))
    alone, _, _ = make_ob(need_module, tmp_path / "alone", pool_of(
        need_module, tmp_path / "alone", facts, routed_questions(need_module)))
    assert alone.track_all() == ()
    for question in questions:
        layered = facts.core_given[question.listed_row].get("layouts")
        if question.listed_row in NO_JOIN_ROWS:
            # Changed 17 Nov 2026: the two rows the table gives no join of the shards for raise in the tool subject and so in
            # the combination's tool half; the runner puts cannot answer with the table's words.
            if layered:
                assert {k: v.raw for k, v in world.combo.read_by_layout(question).items()} == {
                    k: v.raw for k, v in alone.read_by_layout(question).items()}
            else:
                for reader in (world.combo, alone):
                    with pytest.raises(ValueError, match="no join of the shards"):
                        reader.read(question)
            continue
        if layered:
            got = {k: v.raw for k, v in world.combo.read_by_layout(question).items()}
            wanted = {k: v.raw for k, v in alone.read_by_layout(question).items()}
        else:
            got, wanted = world.combo.read(question).raw, alone.read(question).raw
        assert got == wanted, question.id


# --- One right and one wrong reading of each group, graded -------------------------------------

GRADED_ANSWER_ROWS = (
    ("c01.E@9", "c01.E.E@9"),
    ("c02.checkpoints", "c02.i.E@9"),
    ("c04.trace-count", "c04.trace-count.E@6"),
    ("c04.not-a-stop", "c04.not-a-stop.trace-deleted"),
    ("c04.count-499", "c04.count-499"),
    ("c05.unmutated.shards", "c05.unmutated.E/shard-3"),
    ("c05.unmutated.checkpoints", "c05.unmutated.E@9"),
    ("c05.f.trace-deleted", "c05.f.trace-deleted.E@9"),
    ("c05.f.trace-half", "c05.f.trace-half.E@9"),
    ("c05.f.map-deleted", "c05.f.map-deleted.E@9"),
    ("c05.f.digest", "c05.f.digest.E@9"),
    ("c05.withdraw-C.unmutated", "c05.withdraw-C.unmutated.E/shard-3"),
    ("c05.unregistered-checkpoint", "c05.unregistered-checkpoint"),
    ("c06.downstream", "c06.fixture.downstream.E/shard-3"),
    ("c08.checkpoints", "c08.withdraw-A.E@9"),
    ("c09.successors", "c09.fixture.successor.E/shard-3"),
)


def _answer_readings(expect: dict) -> tuple[dict, dict]:
    """The reading that gives the answer the listed row expects and one that gives another,
    written from the expectation: a stop where it expects an error."""
    if "error" in expect:
        return {"stopped": True}, {"answer": "does not contain"}
    wrong = "does not contain" if expect["answer"] == "contains" else "contains"
    return {"answer": expect["answer"]}, {"answer": wrong}


def test_a_right_and_a_wrong_reading_of_each_group_are_graded_through_the_grader(need_module):
    seen = set()
    for group, listed in GRADED_ANSWER_ROWS:
        question = question_of(need_module, listed)
        assert question.group == group and question.route is not None
        expect = expectation(need_module, listed)
        right, wrong = _answer_readings(expect)
        assert graded(need_module, question, right).status == "right", listed
        got = graded(need_module, question, wrong)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
        seen.add(group)
    # The groups whose values are not an answer word.
    question = question_of(need_module, "c01.K.warm-start")
    expect = expectation(need_module, "c01.K.warm-start")
    right = {key: expect[key] for key in LAUNCH_VALUES}
    assert graded(need_module, question, right).status == "right"
    got = graded(need_module, question, {**right, "first_step": right["first_step"] + 1})
    assert got.status == "wrong" and list(got.evidence) == ["first_step"]
    question = question_of(need_module, "c01.E.step8.counted")
    right = {"consumed_samples": expectation(need_module, "c01.E.step8.counted")["consumed_samples"]}
    assert graded(need_module, question, right).status == "right"
    assert graded(need_module, question, {"consumed_samples": right["consumed_samples"] - 1}).status == "wrong"
    question = question_of(need_module, "c08.withdraw-A.E@9.step-8-holds-no-T")
    expect = expectation(need_module, "c08.withdraw-A.E@9.step-8-holds-no-T")
    right = {"rows_that_name_T": expect["rows_that_name_T"], "first_exposure": expect["first_exposure"]}
    assert right == {"rows_that_name_T": 0, "first_exposure": {"segment": "E/seg-2", "step": 7}}
    assert graded(need_module, question, right).status == "right"
    got = graded(need_module, question, {**right, "rows_that_name_T": 1})
    assert got.status == "wrong" and list(got.evidence) == ["rows_that_name_T"]
    question = question_of(need_module, "c09.E.exposure-free-restart")
    right = {"exposure_free_restart": expectation(need_module, "c09.E.exposure-free-restart")["exposure_free_restart"]}
    assert right == {"exposure_free_restart": "E@6"}
    assert graded(need_module, question, right).status == "right"
    got = graded(need_module, question, {"exposure_free_restart": "E@9"})
    assert got.status == "wrong" and list(got.evidence) == ["exposure_free_restart"]
    seen |= {"c01.positions", "c01.step-facts", "c08.step-8-holds-no-T", "c09.exposure-free-restart"}
    # The rest are the groups of the tool half alone whose readings the tool's own tests grade.
    assert set(g for g, _ in ROUTED_BY_HAND) - seen == {"c03.spans", "c04.interrupted-write",
                                                          "c09.artifact-actions"}


# --- The walk ------------------------------------------------------------------------------------


def test_the_walk_reads_the_answers_the_table_expects_on_the_unmutated_store(need_module, tmp_path):
    world = make_world(need_module, tmp_path)
    listed_rows = (["c05.unmutated.E@6", "c05.unmutated.E@9", "c05.unmutated.D@3", "c05.unmutated.Y@6",
                    "c05.unmutated.K@3", "c05.unmutated.K@8", "c05.unmutated.K@40", "c01.E.E@9"]
                   + [f"c08.withdraw-A.{name}" for name in ("E@6", "E@9", "D@3", "Y@6", "K@3", "K@8", "K@40")]
                   + [f"c02.i.{name}" for name in ("E@6", "E@9", "D@3", "Y@6")])
    for listed in listed_rows:
        question = question_of(need_module, listed)
        raw = world.combo.read(question).raw
        assert raw == {"answer": expectation(need_module, listed)["answer"]}, listed
        assert graded(need_module, question, raw).status == "right", listed


def test_a_checkpoint_the_trace_does_not_hold_or_holds_no_row_for_reads_not_covered(need_module, tmp_path):
    # The trace holds no checkpoint at all.
    empty = make_world(need_module, tmp_path / "a", document={"segments": []})
    assert raw_of(empty, need_module, "c05.unmutated.E@9") == {"answer": "not covered"}
    # The registration wrote no row for any step of the history (the starting states ii and iii).
    for state, listed in (("c2.ii", "c02.ii.E@9"), ("c2.iii", "c02.iii.E@9")):
        world = make_world(need_module, tmp_path / state)
        question = question_of(need_module, listed)
        raw = world.combo.read(question).raw
        assert raw == {"answer": "not covered"}, listed
        got = graded(need_module, question, raw)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
    # The starting state in which only run K has no trace: the history of K@8 is that of its
    # parent checkpoint, whose rows are there and read no record of A.
    world = make_world(need_module, tmp_path / "iv")
    assert raw_of(world, need_module, "c02.iv.K@8") == {"answer": "does not contain"}
    assert raw_of(world, need_module, "c05.unmutated.E@9") == {"answer": "contains"}


def test_a_parents_history_stops_at_its_step_and_a_row_after_it_is_not_read(need_module, tmp_path):
    # A record of A at step 7 of E/seg-1, after its checkpoint at step 6: E@6 and Y@6, which
    # starts from E@6, do not read it.
    document = world_trace()
    e1 = segment_of(document, "E/seg-1")
    e1["launches"] = [_launch(0, 0, 0, 1, 7)]
    e1["steps"].append({"step": 7, "consumed_samples": 4, "samples": [
        _sample(0, 0, [_row("E/shard-3", 2, 1)])]})
    world = make_world(need_module, tmp_path, document=document)
    for listed in ("c05.unmutated.E@6", "c05.unmutated.Y@6", "c05.unmutated.K@3"):
        assert raw_of(world, need_module, listed) == {"answer": "does not contain"}, listed
    assert raw_of(world, need_module, "c05.unmutated.E@9") == {"answer": "contains"}


# --- The tie ---------------------------------------------------------------------------------------


def test_the_tie_is_valid_only_where_the_run_says_one_process_in_packed_order(need_module, facts, tmp_path):
    for label, tool in (("two", FakeTool(processes=2)), ("silent", FakeTool(say_run=False))):
        world = make_world(need_module, tmp_path / label, tool=tool)
        for listed in ("c05.unmutated.E@6", "c05.unmutated.E@9", "c08.withdraw-A.K@8"):
            question = question_of(need_module, listed)
            raw = world.combo.read(question).raw
            otherwise = facts.row_by_id[question.row_id]["tie"]["otherwise"]
            assert raw == {"cannot_answer": otherwise}, (label, listed)
            got = graded(need_module, question, raw)
            assert got.status == "cannot answer" and got.evidence == otherwise
    ok = make_world(need_module, tmp_path / "one")
    assert raw_of(ok, need_module, "c05.unmutated.E@9") == {"answer": "contains"}


def test_a_document_is_tied_by_the_range_its_own_index_gives_never_by_its_place_or_the_entries_counts(
        need_module, tmp_path):
    # Under f.digest the first digit of the first entry's count of shard 3 changes (4 to 5):
    # the entries' ranges are [0,5), [5,11), [11,16), [16,19) and the documents' [0,4), [4,10),
    # [10,15), [15,18). No document of shard 3 has an entry of its own range, so the record of
    # A at ordinal 1 is tied to nothing and its bit is not read. A tie by place (entry k is
    # ordinal k) reads the bit and answers "contains"; a range taken from the entries' own counts
    # ties everything and answers "contains" as well. The rows of shards 1 and 2 stay tied, so
    # the history has a tied row and the answer is "does not contain".
    world = make_world(need_module, tmp_path)
    question = question_of(need_module, "c05.f.digest.E@6")
    assert world.combo.read(question).raw == {"answer": "does not contain"}
    question = question_of(need_module, "c05.f.digest.E@9")
    raw = world.combo.read(question).raw
    assert raw == {"answer": "does not contain"}
    got = graded(need_module, question, raw)
    assert got.status == "wrong" and list(got.evidence) == ["answer"]
    # A shard whose first entry is deleted (the entries' list is one short) shifts the same way
    # for a later document: the fault f.map-deleted deletes every entry of shard 1 and leaves
    # the rows of shard 3 tied, so E@9 still reads "contains" and the table's answer is right.
    question = question_of(need_module, "c05.f.map-deleted.E@9")
    raw = world.combo.read(question).raw
    assert raw == {"answer": "contains"}
    assert graded(need_module, question, raw).status == "right"


def test_a_tool_fault_is_put_once_on_the_fresh_copy_before_its_revoke_and_only_for_the_shard_the_walk_reads(
        need_module, tmp_path):
    world = make_world(need_module, tmp_path)
    # E@6 reads shards 1 and 2 only: the fault of shard 3 is not put.
    world.combo.read(question_of(need_module, "c05.f.digest.E@6"))
    assert world.putter.puts == []
    world.combo.read(question_of(need_module, "c05.f.digest.E@9"))
    assert len(world.putter.puts) == 1
    copy_folder, fault = world.putter.puts[0]
    assert fault == "f.digest"
    assert copy_folder not in {world.ob.tracked_store(spec) for spec in world.specs}
    order = [(c["verb"], c["store"]) for c in world.tool.calls if c["store"] == copy_folder]
    assert order[0][0] == "fault" and order[1][0] == "revoke"
    assert [verb for verb, _ in order].count("revoke") == 1
    # A fault of the other shard: f.map-deleted is put for shard 1, which E@6 reads.
    world.combo.read(question_of(need_module, "c05.f.map-deleted.E@6"))
    assert [fault for _, fault in world.putter.puts] == ["f.digest", "f.map-deleted"]


def test_a_faulted_shard_the_tool_ends_with_an_error_on_reads_stopped_and_other_checkpoints_answer(
        need_module, tmp_path):
    for label, tool in (
            ("revoke", FakeTool(error_after=("f.digest",))),
            ("program", FakeTool(faulted_program_error=("generate-set",)))):
        world = make_world(need_module, tmp_path / label, tool=tool)
        question = question_of(need_module, "c05.f.digest.E@9")
        raw = world.combo.read(question).raw
        assert raw == {"stopped": True}, label
        assert graded(need_module, question, raw).status == "right", label
        # E@6 does not read the faulted shard: it answers, and the table's answer is right.
        question = question_of(need_module, "c05.f.digest.E@6")
        raw = world.combo.read(question).raw
        assert raw == {"answer": "does not contain"}, label
        assert graded(need_module, question, raw).status == "right", label


def test_an_end_that_is_not_the_tools_line_is_the_failed_attempt_itself(need_module, tmp_path):
    interfaces = need_module("baselines.interfaces")
    world = make_world(need_module, tmp_path, tool=FakeTool(fail={"revoke": (7, "the lock timed out")}))
    got = world.combo.read(question_of(need_module, "c05.unmutated.E@9"))
    assert isinstance(got, interfaces.Attempt) and got.exit_code == 7
    assert "the lock timed out" in got.output
    assert str(tmp_path) not in got.output and all(str(tmp_path) not in word for word in got.command)
    # The same end on a store the step is told is faulted is not the tool's line either.
    got = world.combo.read(question_of(need_module, "c05.f.digest.E@9"))
    assert isinstance(got, interfaces.Attempt) and got.exit_code == 7


def test_each_shard_is_revoked_on_a_fresh_copy_once_and_the_tracked_stores_never_change(need_module, tmp_path):
    world = make_world(need_module, tmp_path)
    before = {spec.name: tree_digest(world.ob.tracked_store(spec)) for spec in world.specs}
    for listed in ("c05.unmutated.E@9", "c05.unmutated.K@40", "c05.unmutated.E@6",
                   "c05.unmutated.E@9", "c05.unmutated.D@3"):
        world.combo.read(question_of(need_module, listed))
    revoked = [c["store"] for c in calls_of(world.tool, "revoke")]
    assert revoked, "the walk made no revoke"
    assert len(set(revoked)) == len(revoked), "a copy was revoked twice"
    tracked = {world.ob.tracked_store(spec) for spec in world.specs}
    assert not set(revoked) & tracked
    assert {spec.name: tree_digest(world.ob.tracked_store(spec)) for spec in world.specs} == before


# --- c08.step-8 and the restart ----------------------------------------------------------------------


def test_the_step_8_reading_counts_the_rows_tied_to_T_and_makes_no_revoke(need_module, tmp_path):
    listed = "c08.withdraw-A.E@9.step-8-holds-no-T"
    world = make_world(need_module, tmp_path / "a")
    question = question_of(need_module, listed)
    raw = world.combo.read(question).raw
    assert raw == {"rows_that_name_T": 0, "first_exposure": {"segment": "E/seg-2", "step": 7}}
    assert graded(need_module, question, raw).status == "right"
    assert calls_of(world.tool, "revoke") == []
    # The row of the cut sample is a record of T: one row names T, and the step is still 7.
    document = world_trace()
    step_of(document, "E/seg-2", 8)["samples"][3]["rows"] = [_row("E/shard-3", 2, 2, 19, 35)]
    world = make_world(need_module, tmp_path / "b", document=document)
    raw = world.combo.read(question).raw
    assert raw == {"rows_that_name_T": 1, "first_exposure": {"segment": "E/seg-2", "step": 7}}
    got = graded(need_module, question, raw)
    assert got.status == "wrong" and list(got.evidence) == ["rows_that_name_T"]
    # The record of A is not T: with the record of T first read at step 9 the first exposure is
    # step 9, though the record of A is read at step 7.
    document = world_trace()
    step_of(document, "E/seg-2", 7)["samples"][1]["rows"] = [_row("E/shard-2", 1, 0)]
    step_of(document, "E/seg-2", 9)["samples"][0]["rows"] = [_row("E/shard-3", 2, 2)]
    world = make_world(need_module, tmp_path / "c", document=document)
    raw = world.combo.read(question).raw
    assert raw == {"rows_that_name_T": 0, "first_exposure": {"segment": "E/seg-2", "step": 9}}
    got = graded(need_module, question, raw)
    assert got.status == "wrong" and list(got.evidence) == ["first_exposure"]


def test_the_exposure_free_restart_is_the_latest_checkpoint_read_does_not_contain_with_none_before(
        need_module, tmp_path):
    listed = "c09.E.exposure-free-restart"
    world = make_world(need_module, tmp_path / "a")
    question = question_of(need_module, listed)
    raw = world.combo.read(question).raw
    assert raw == {"exposure_free_restart": "E@6"}
    assert graded(need_module, question, raw).status == "right"
    # A record of A at step 2 of E/seg-1: the first checkpoint reads "contains", so there is none.
    document = world_trace()
    step_of(document, "E/seg-1", 2)["samples"][0]["rows"] = [_row("E/shard-3", 2, 1)]
    world = make_world(need_module, tmp_path / "b", document=document)
    raw = world.combo.read(question).raw
    assert raw == {"exposure_free_restart": None}
    got = graded(need_module, question, raw)
    assert got.status == "wrong" and list(got.evidence) == ["exposure_free_restart"]


# --- The faults of the trace half --------------------------------------------------------------------


def _printed_of(segment: dict) -> dict:
    """What ``show E/seg-1 --store F --trace`` prints for the hand-written segment: the same
    document with the keys the join holds besides."""
    return {"segment": segment["segment"], "route": "cache files", "cause": None,
            "cause_detail": None, "untraced_steps": [],
            "launches": copy.deepcopy(segment["launches"]),
            "steps": [{**st, "samples": [{**sm, "global_index": 0} for sm in st["samples"]]}
                      for st in segment["steps"]]}


def _exported_first_segment() -> dict:
    from baselines.trace import export_segment

    first = segment_of(world_trace(), "E/seg-1")
    run_record = {"format": "waterwhite.run-record/1", "run": "E", "segment": "E/seg-1",
                  "start_kind": "fresh", "parent_checkpoint": None,
                  "shards": [{"position": 0, "prefix": "shards/shard-1"},
                             {"position": 1, "prefix": "shards/shard-2"}]}
    step_record = {"segment": "E/seg-1", "steps": [], "checkpoints": [{"name": "E@6", "step": 6}]}
    exported = export_segment(_printed_of(first), run_record, step_record)
    assert exported == first
    return exported


def test_the_half_fault_cuts_the_row_of_an_exported_record_to_the_16_covered_tokens():
    from baselines.faults import apply_fault

    exported = _exported_first_segment()
    uncut = step_of({"segments": [exported]}, "E/seg-1", 4)["samples"][3]
    assert uncut["rows"] == [_row("E/shard-1", 0, 1, 6, 39)] and uncut["covered_tokens"] == 33
    half = apply_fault("f.trace-half", "trace", exported)
    cut = step_of({"segments": [half]}, "E/seg-1", 4)["samples"][3]
    # Document 1 of shard 1, tokens 6 to 39: its first 16 tokens stay, 6 to 22; the sample stays
    # counted with its 33 tokens and 16 covered.
    assert cut["rows"] == [_row("E/shard-1", 0, 1, 6, 22)]
    assert cut["covered_tokens"] == 16 and cut["sample_tokens"] == 33
    assert len(step_of({"segments": [half]}, "E/seg-1", 4)["samples"]) == 4
    assert exported == _exported_first_segment()


def test_the_combination_reads_a_cut_a_deleted_and_a_short_trace_as_they_stand(need_module, tmp_path):
    world = make_world(need_module, tmp_path)
    # f.trace-half and f.trace-deleted: no stop of the combination's own, no "may contain": the
    # answers are those of the whole trace, so the table's "may contain" of E@6 is read wrong and
    # the answers of E@9 are right.
    for listed in ("c05.f.trace-half.E@6", "c05.f.trace-deleted.E@6", "c04.not-a-stop.trace-half",
                   "c04.not-a-stop.trace-deleted"):
        question = question_of(need_module, listed)
        raw = world.combo.read(question).raw
        assert raw == {"answer": "does not contain"}, listed
        got = graded(need_module, question, raw)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
    for listed in ("c05.f.trace-half.E@9", "c05.f.trace-deleted.E@9"):
        question = question_of(need_module, listed)
        raw = world.combo.read(question).raw
        assert raw == {"answer": "contains"}, listed
        assert graded(need_module, question, raw).status == "right", listed
    # l.trace-count: a sample short of the batch size is read as it stands: an answer where the
    # table expects a stop.
    for listed in ("c04.trace-count.E@6", "c04.trace-count.E@9"):
        question = question_of(need_module, listed)
        raw = world.combo.read(question).raw
        assert "stopped" not in raw and set(raw) == {"answer"}, listed
        got = graded(need_module, question, raw)
        assert got.status == "wrong" and list(got.evidence) == ["answer"], listed
    # The document the question reads is the one trace_of gives: the cut is in it.
    document = trace_for("f.trace-half")
    assert step_of(document, "E/seg-1", 4)["samples"][3]["covered_tokens"] == 16
    assert [s["step"] for s in trace_for("f.trace-deleted")["segments"][0]["steps"]] == [1, 2, 4, 5, 6]
    assert len(step_of(trace_for("l.trace-count"), "E/seg-1", 5)["samples"]) == 3


def test_a_failed_export_of_the_trace_is_returned_as_that_attempt_and_the_tool_half_is_unaffected(
        need_module, facts, tmp_path):
    interfaces = need_module("baselines.interfaces")
    failed = interfaces.Attempt(id="trace-export-1", command=("export",), exit_code=1, output="no")
    world = make_world(need_module, tmp_path, trace_of=lambda listed: failed)
    for listed in ("c05.unmutated.E@9", "c01.E.step8.counted", "c08.withdraw-A.E@9.step-8-holds-no-T",
                   "c09.E.exposure-free-restart"):
        assert world.combo.read(question_of(need_module, listed)) == failed, listed
    got = world.combo.read(question_of(need_module, "c05.unmutated.E/shard-3"))
    assert isinstance(got, interfaces.Reading) and set(got.raw) == {"answer"}


# --- Kinds of artifact neither half holds ---------------------------------------------------------------


def test_a_kind_neither_half_holds_is_cannot_answer_with_the_missing_text_of_the_combinations_row(
        need_module, facts, tmp_path):
    world = make_world(need_module, tmp_path, shards_of=lambda ob, base: pool_of(
        need_module, base, facts, routed_questions(need_module)))
    seen = set()
    for question in routed_questions(need_module):
        if question.group not in BY_KIND:
            continue
        kind = artifact_kind(facts.core_given[question.listed_row].get("artifact"))
        if kind not in MISSING_ROW:
            continue
        seen.add(kind)
        wanted = facts.row_by_id[MISSING_ROW[kind]]["missing"]
        assert world.combo.read(question).raw == {"cannot_answer": wanted}, question.id
    assert seen == {"files", "mixtures", "listing"}


def test_cannot_answer_comes_only_after_the_successful_track_of_the_store(need_module, facts, tmp_path):
    interfaces = need_module("baselines.interfaces")
    ob = need_module(OB_MODULE)
    question = question_of(need_module, "c05.f.map-deleted.src/A")
    wanted = facts.row_by_id[MISSING_ROW["files"]]["missing"]
    # Not tracked yet: the tool did not run, so there is no line (the tool half's own refusal).
    before = make_world(need_module, tmp_path / "a", tracked=False)
    with pytest.raises(ob.NotAttempted):
        before.combo.read(question)
    # Given no shard of the store at all: not attempted either.
    none = make_world(need_module, tmp_path / "b", shards_of=lambda o, base: [])
    with pytest.raises(ob.NotAttempted):
        none.combo.read(question)
    # A shard whose track failed: that failed attempt, never "cannot answer".
    failing = make_world(need_module, tmp_path / "c", tool=FakeTool(fail_track_in=("tracked-1",)))
    assert len(failing.failed) == 1
    got = failing.combo.read(question)
    assert isinstance(got, interfaces.Attempt) and got == failing.failed[0]
    # Every shard tracked: cannot answer.
    good = make_world(need_module, tmp_path / "d")
    assert good.combo.read(question).raw == {"cannot_answer": wanted}


def test_a_walk_over_a_shard_whose_track_failed_is_that_failed_attempt(need_module, tmp_path):
    interfaces = need_module("baselines.interfaces")
    # Shard 3 is the third shard given: its store is tracked-2.
    world = make_world(need_module, tmp_path, tool=FakeTool(fail_track_in=("tracked-2",)))
    assert len(world.failed) == 1
    got = world.combo.read(question_of(need_module, "c05.unmutated.E@9"))
    assert isinstance(got, interfaces.Attempt) and got == world.failed[0]


# --- Two layouts and the stopped writer --------------------------------------------------------------------


def test_a_layout_with_no_tracked_store_is_not_attempted_never_cannot_answer(need_module, tmp_path):
    # "A tool that did not run cannot earn it": a layout none of whose shards was given has no
    # reading (None), and the line of the row follows the order of the layouts' slots: with the
    # other layout right, it is "not attempted". The tracked layout is read as before.
    interfaces = need_module("baselines.interfaces")
    expected = need_module("baselines.expected")
    grade = need_module("baselines.grade")
    ten = question_of(need_module, "c09.10k.successor-shards")
    held = expected.concrete_values_by_layout(expected.expected_for("c09.10k.successor-shards"))

    def partitioned(ob, base):
        return [shard(ob, base, "partitioned/shard-1", plain_documents(ob), target=("set", "10k"),
                      layout="partitioned", successor=True)]

    world = make_world(need_module, tmp_path / "a", shards_of=partitioned)
    assert world.failed == ()
    # Changed 17 Nov 2026 (old: the tracked layout read "does not contain", the other None, the row "not attempted"): the table
    # gives no join of the shards for this row, so the tracked layout raises; the cases below (failed track, not tracked)
    # come before the raise.
    by_layout = world.combo.read_by_layout(ten)
    assert by_layout["round-robin"] is None, "the layout with no shard given is not attempted"
    assert by_layout["partitioned"].raw == {"cannot_answer": need_module("baselines.dolma").NO_JOIN_TEXT}
    assert grade.grade_layouts(ten, held, by_layout).status == "not attempted", "no cannot-answer line for a layout never run on"
    # A row that names two layouts is read by read_by_layout only.
    with pytest.raises(ValueError):
        world.combo.read(ten)
    # A layout whose shards failed to track is that failed attempt, not "not attempted".
    broken = make_world(need_module, tmp_path / "f", shards_of=partitioned,
                        tool=FakeTool(fail={"track": (1, "sandbox failed\n")}))
    assert len(broken.failed) == 1
    got = broken.combo.read_by_layout(ten)
    assert got["round-robin"] is None and got["partitioned"] == broken.failed[0]
    # Shards that were given and not tracked yet are not attempted either.
    fresh = make_world(need_module, tmp_path / "g", shards_of=partitioned, tracked=False)
    assert fresh.combo.read_by_layout(ten) == {"partitioned": None, "round-robin": None}


def _scratch_world(need_module, base: Path, tool: FakeTool) -> World:
    def shards_of(ob, root):
        return [shard(ob, root, "c04/shard-500", tuple(doc(ob, "sources/B.jsonl", 3) for _ in range(500)),
                      target=("store", "scratch-500"))]
    return make_world(need_module, base, shards_of=shards_of, tool=tool)


def test_a_stop_that_did_not_land_is_the_failed_attempt_of_the_stopped_writer_in_both_rows(
        need_module, tmp_path):
    interfaces = need_module("baselines.interfaces")
    for listed in ("c04.kill-after-100", "c04.retry"):
        base = tmp_path / listed
        world = _scratch_world(need_module, base / "combo", FakeTool(stop_exit=0))
        alone, _, _ = make_ob(need_module, base / "alone", [shard(
            need_module(OB_MODULE), base / "alone", "c04/shard-500",
            tuple(doc(need_module(OB_MODULE), "sources/B.jsonl", 3) for _ in range(500)),
            target=("store", "scratch-500"))], FakeTool(stop_exit=0))
        assert alone.track_all() == ()
        question = question_of(need_module, listed)
        got, wanted = world.combo.read(question), alone.read(question)
        assert isinstance(got, interfaces.Attempt) and isinstance(wanted, interfaces.Attempt), listed
        assert got.exit_code == 0 and got.exit_code == wanted.exit_code
        assert got.output == wanted.output and "did not land" in got.output


def test_the_first_row_of_the_stopped_write_says_when_its_rows_showed(need_module, tmp_path):
    for label, tool, wanted in (("while", FakeTool(), "while the writer ran"),
                                ("end", FakeTool(write_at_end=True), "only at its end")):
        world = _scratch_world(need_module, tmp_path / label, tool)
        raw = world.combo.read(question_of(need_module, "c04.kill-after-100")).raw
        assert raw["records"]["rows_showed"] == wanted, label


def test_no_reading_holds_a_diff_or_a_local_path(need_module, facts, tmp_path):
    world = make_world(need_module, tmp_path / "w", shards_of=lambda ob, base: pool_of(
        need_module, base, facts, routed_questions(need_module)))
    for listed in ("c05.unmutated.E@9", "c05.f.map-deleted.E/mix-1", "c05.unmutated.E/shard-3",
                   "c08.withdraw-A.E@9.step-8-holds-no-T", "c01.E.step8.counted"):
        got = world.combo.read(question_of(need_module, listed))
        text = json.dumps(got.raw, sort_keys=True)
        assert "diff" not in got.raw and str(tmp_path) not in text, listed
