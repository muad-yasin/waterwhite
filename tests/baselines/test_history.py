"""A checkpoint's history over a trace document, on a hand-written document (plan 5.5
Check 10; table README, section "Check 10": the trace's own records and the
combinations' route).

The trace document used here is written by hand. It holds, per segment, what the
README lists for the trace: the segment's name and run, its parent checkpoint, its
shards in the trainer's order, the checkpoints saved in it with their steps, and its
steps, each with its samples and each sample with its rows (shard, shard position,
ordinal, token start, token end), in the form ``show SEGMENT --trace`` prints a
segment. The segments are the list under ``segments``.

The route of the combination rows reads: the checkpoint's segment up to its step,
then the history of its parent checkpoint. What these tests hold the reader to:

- The history of a checkpoint is the rows of the steps of its segment up to and
  including the checkpoint's step, in step order and, inside a step, in reading
  order (sample by sample, row by row); then the history of the checkpoint named as
  the segment's parent, the same way, and so on up the chain. The segment's own rows
  come first.
- A parent's history stops at the parent's own step: the rows its segment holds after
  that step are not part of the child's history.
- The reader adds no check of its own. A step with no row in the trace, a sample with
  no row and a parent the trace does not hold add nothing; there is no placeholder and
  no error. A checkpoint the trace does not hold is told apart from a checkpoint it
  holds with no row: None, and an empty list.

The names asked of ``baselines.history``:

- ``history_rows(trace, checkpoint)``: a list of rows, each with ``segment``,
  ``step``, ``shard``, ``ordinal``, ``token_start`` and ``token_end`` (an attribute or
  a key), in the order above; None where the trace holds no checkpoint of that name.
  The trace document is not changed.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Mapping
from pathlib import Path

import pytest

MODULE = "baselines.history"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")


def field(obj, name):
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def row(shard: str, ordinal: int, start: int = 0, end: int = 21) -> dict:
    return {"shard": shard, "shard_position": 0, "ordinal": ordinal,
            "token_start": start, "token_end": end}


def sample(pos: int, *rows: dict, covered: int | None = None) -> dict:
    covered_tokens = covered if covered is not None else sum(r["token_end"] - r["token_start"]
                                                              for r in rows)
    return {"sample_pos": pos, "rank": 0, "sample_tokens": 33,
            "covered_tokens": covered_tokens, "rows": list(rows)}


def step(number: int, *samples: dict) -> dict:
    return {"step": number, "consumed_samples": 4 * number, "samples": list(samples)}


def segment(name, run, parent, shards, checkpoints, steps) -> dict:
    return {"segment": name, "run": run, "parent_checkpoint": parent, "shards": list(shards),
            "checkpoints": [{"name": n, "step": s} for n, s in checkpoints],
            "steps": list(steps)}


def one(shard: str, ordinal: int):
    """A step's single sample with its single row."""
    return (sample(0, row(shard, ordinal)),)


# A hand-written trace of five segments.
#
# E/seg-1: steps 1 to 6 with step 3 untraced (no entry in "steps"); a checkpoint at
#   step 4 and one at step 6. Step 2 holds two rows in one sample and a sample with
#   no row; step 4 holds two samples.
# E/seg-2: resumes from E@4 and runs steps 5 to 9; a checkpoint at step 8.
# D/seg-1: resumes from E@8, steps 9 to 11; a checkpoint at step 10.
# Y/seg-1: a checkpoint at step 2 whose segment has rows from step 3 on only.
# K/seg-2: resumes from a checkpoint the trace does not hold; a checkpoint at step 5.
def make_trace() -> dict:
    e1 = segment(
        "E/seg-1", "E", None, ["E/shard-1", "E/shard-2"],
        [("E@4", 4), ("E@6", 6)],
        [
            step(1, *one("E/shard-1", 10)),
            step(2, sample(0, row("E/shard-1", 20), row("E/shard-1", 21)),
                 sample(1, covered=0)),
            step(4, sample(0, row("E/shard-2", 40)), sample(1, row("E/shard-2", 41))),
            step(5, *one("E/shard-2", 50)),
            step(6, *one("E/shard-2", 60)),
        ])
    e2 = segment(
        "E/seg-2", "E", "E@4", ["E/shard-3"], [("E@8", 8)],
        [step(n, *one("E/shard-3", 100 + n)) for n in range(5, 10)])
    d1 = segment(
        "D/seg-1", "D", "E@8", ["D/shard-1"], [("D@10", 10)],
        [step(n, *one("D/shard-1", 200 + n)) for n in range(9, 12)])
    y1 = segment(
        "Y/seg-1", "Y", None, ["Y/shard-1"], [("Y@2", 2)],
        [step(3, *one("Y/shard-1", 303))])
    k2 = segment(
        "K/seg-2", "K", "K@40", ["K/shard-1"], [("K@5", 5)],
        [step(4, *one("K/shard-1", 404)), step(5, *one("K/shard-1", 405))])
    return {"segments": [e1, e2, d1, y1, k2]}


def as_tuples(rows):
    return [(field(r, "segment"), field(r, "step"), field(r, "shard"), field(r, "ordinal"))
            for r in rows]


E1_UP_TO_4 = [
    ("E/seg-1", 1, "E/shard-1", 10),
    ("E/seg-1", 2, "E/shard-1", 20),
    ("E/seg-1", 2, "E/shard-1", 21),
    ("E/seg-1", 4, "E/shard-2", 40),
    ("E/seg-1", 4, "E/shard-2", 41),
]
E2_UP_TO_8 = [("E/seg-2", n, "E/shard-3", 100 + n) for n in (5, 6, 7, 8)]
D1_UP_TO_10 = [("D/seg-1", n, "D/shard-1", 200 + n) for n in (9, 10)]


# Looked up when a test calls it, never at setup: a module that does not exist yet is
# a failed assertion in the body of a test, not an error.
@pytest.fixture
def history_rows(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "history_rows")(*a, **k)


# --- The segment up to its step ----------------------------------------------------------


def test_a_checkpoint_in_a_segment_with_no_parent_has_the_rows_of_the_steps_up_to_its_step(history_rows):
    assert as_tuples(history_rows(make_trace(), "E@4")) == E1_UP_TO_4


def test_the_step_of_the_checkpoint_is_included_and_the_later_steps_are_not(history_rows):
    got = as_tuples(history_rows(make_trace(), "E@6"))
    assert got == E1_UP_TO_4 + [("E/seg-1", 5, "E/shard-2", 50), ("E/seg-1", 6, "E/shard-2", 60)]
    # Red example: a history that reads the whole segment ignores the step. E@4 holds
    # no row of step 5 or 6.
    assert not {r[1] for r in as_tuples(history_rows(make_trace(), "E@4"))} & {5, 6}


def test_a_step_with_no_entry_in_the_trace_adds_nothing(history_rows):
    """Step 3 of E/seg-1 is untraced: no placeholder stands for it."""
    steps = [r[1] for r in as_tuples(history_rows(make_trace(), "E@6"))]
    assert 3 not in steps
    assert steps == [1, 2, 2, 4, 4, 5, 6]


def test_a_step_absent_from_steps_and_a_step_with_no_samples_are_both_untraced(history_rows):
    """Two forms of an untraced step 3 of E/seg-1: no entry in `steps` (the form
    docs/cli.md documents: `steps` holds one entry per traced step), and an entry whose
    `samples` is []. The table reads a step with no row as untraced and adds nothing for
    it (check10.toml, the rules of the combination rows), so the history is the same in
    both forms and holds no row of step 3."""
    absent = make_trace()
    assert [s["step"] for s in absent["segments"][0]["steps"]] == [1, 2, 4, 5, 6]
    empty = make_trace()
    steps = empty["segments"][0]["steps"]
    steps.insert(2, step(3))
    assert steps[2]["samples"] == []
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5, 6]
    e1_up_to_6 = E1_UP_TO_4 + [("E/seg-1", 5, "E/shard-2", 50), ("E/seg-1", 6, "E/shard-2", 60)]
    expected = {
        "E@4": E1_UP_TO_4,
        "E@6": e1_up_to_6,
        "E@8": E2_UP_TO_8 + E1_UP_TO_4,
        "D@10": D1_UP_TO_10 + E2_UP_TO_8 + E1_UP_TO_4,
    }
    for name, rows in expected.items():
        for form, trace in (("absent", absent), ("samples []", empty)):
            got = as_tuples(history_rows(trace, name))
            assert got == rows, f"{name}, step 3 {form}"
            assert 3 not in {r[1] for r in got}, f"{name}, step 3 {form}"


def test_a_sample_with_no_row_adds_nothing_and_the_rows_keep_reading_order(history_rows):
    got = as_tuples(history_rows(make_trace(), "E@4"))
    # Step 2: sample 0 holds two rows, sample 1 holds none. Step 4: sample 0 then 1.
    assert got[1:3] == [("E/seg-1", 2, "E/shard-1", 20), ("E/seg-1", 2, "E/shard-1", 21)]
    assert got[3:5] == [("E/seg-1", 4, "E/shard-2", 40), ("E/seg-1", 4, "E/shard-2", 41)]


def test_a_row_keeps_its_token_range_as_the_trace_gives_it(history_rows):
    trace = make_trace()
    trace["segments"][0]["steps"][0]["samples"][0]["rows"][0].update(token_start=5, token_end=17)
    first = history_rows(trace, "E@4")[0]
    assert (field(first, "token_start"), field(first, "token_end")) == (5, 17)


# --- Then the parent checkpoint's history ------------------------------------------------


def test_the_segment_comes_first_then_the_history_of_its_parent_checkpoint(history_rows):
    assert as_tuples(history_rows(make_trace(), "E@8")) == E2_UP_TO_8 + E1_UP_TO_4


def test_the_parents_history_stops_at_the_parents_step(history_rows):
    """E/seg-2 resumed from E@4. E/seg-1 also holds steps 5 and 6, which E/seg-2
    replaces; they are not in the history of E@8."""
    got = as_tuples(history_rows(make_trace(), "E@8"))
    assert ("E/seg-1", 5, "E/shard-2", 50) not in got
    assert ("E/seg-1", 6, "E/shard-2", 60) not in got
    assert {r[0] for r in got} == {"E/seg-1", "E/seg-2"}


def test_the_chain_goes_up_through_every_parent(history_rows):
    assert as_tuples(history_rows(make_trace(), "D@10")) == D1_UP_TO_10 + E2_UP_TO_8 + E1_UP_TO_4


def test_a_step_after_the_checkpoint_in_its_own_segment_is_left_out_of_a_child_chain(history_rows):
    got = as_tuples(history_rows(make_trace(), "D@10"))
    assert ("D/seg-1", 11, "D/shard-1", 211) not in got
    assert ("E/seg-2", 9, "E/shard-3", 109) not in got


# --- What the trace does not hold --------------------------------------------------------


def test_a_checkpoint_the_trace_does_not_hold_is_none(history_rows):
    assert history_rows(make_trace(), "E@9") is None
    assert history_rows(make_trace(), "no-such-checkpoint") is None
    assert history_rows({"segments": []}, "E@4") is None


def test_a_checkpoint_the_trace_holds_with_no_row_is_an_empty_list_not_none(history_rows):
    """Y@2 is saved in Y/seg-1, whose first traced step is 3: nothing up to step 2."""
    got = history_rows(make_trace(), "Y@2")
    assert got is not None
    assert list(got) == []


def test_a_parent_the_trace_does_not_hold_adds_nothing_and_is_no_error(history_rows):
    """K/seg-2 names K@40 as its parent and the trace holds no such checkpoint: the
    reader reads what is there."""
    assert as_tuples(history_rows(make_trace(), "K@5")) == [
        ("K/seg-2", 4, "K/shard-1", 404),
        ("K/seg-2", 5, "K/shard-1", 405),
    ]


def test_a_segment_with_no_steps_at_all_holds_its_checkpoint_with_no_row(history_rows):
    trace = {"segments": [segment("Q/seg-1", "Q", None, ["Q/shard-1"], [("Q@3", 3)], [])]}
    got = history_rows(trace, "Q@3")
    assert got is not None and list(got) == []


# --- The document is only read -----------------------------------------------------------


def test_the_trace_document_is_not_changed(history_rows):
    trace = make_trace()
    before = copy.deepcopy(trace)
    history_rows(trace, "D@10")
    history_rows(trace, "K@5")
    history_rows(trace, "nothing")
    assert trace == before


def test_the_history_of_a_checkpoint_does_not_depend_on_the_order_of_the_segments(history_rows):
    trace = make_trace()
    shuffled = {"segments": list(reversed(trace["segments"]))}
    for name in ("E@4", "E@6", "E@8", "D@10", "K@5", "Y@2"):
        assert as_tuples(history_rows(shuffled, name)) == as_tuples(history_rows(trace, name))


# --- What the module may hold ------------------------------------------------------------


def _path(need_module) -> Path:
    module = need_module(MODULE)
    path = getattr(module, "__file__", None)
    assert path, f"{MODULE} has no file"
    return Path(path)


def test_the_module_imports_nothing_of_the_tool_nor_of_the_world_helpers(need_module):
    path = _path(need_module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, sorted(roots & FORBIDDEN_IMPORT_ROOTS)


def test_the_module_holds_no_local_path_and_no_em_dash(need_module):
    text = _path(need_module).read_text(encoding="utf-8")
    for bad in FORBIDDEN_TEXT:
        assert bad not in text, f"{MODULE} holds {bad!r}"
