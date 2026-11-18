"""The rows of Checks 1 and 8 that read ``show SEGMENT --trace``, put to the world with
the printed trace and the oracle's log replaced by documents written by hand.

The rows are put to the real command by ``test_check01.py`` and ``test_check08.py``. Here
``World._join_rows`` is run on traces built from the fixture spec's
positions (the rows the spec lists as lines of the oracle's log), once on a trace that is
right for the row and once on a trace with the one thing the row names changed: each of
those must be found. The values are the spec's and the table's, never the tool's.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import acceptance_join as join
import acceptance_runs as runs
import acceptance_trace as trace
import pytest
import table
from acceptance_world import NOT_BUILT, World

SPEC = runs.spec()
POSITIONS = SPEC["positions"]


def lines(name: str) -> list[dict]:
    return [json.loads(line) for line in POSITIONS[name]["log_lines"]]


def document(segment: str, rows: list[dict], cuts: dict[tuple[int, int], int] | None = None,
             untraced: list | None = None) -> dict:
    """A printed trace of ``segment`` from rows written like lines of the oracle's log. A
    sample named in ``cuts`` ((step, sample_pos) -> tokens read) is printed cut."""
    steps: dict[int, dict[int, dict]] = {}
    for r in rows:
        if r["segment"] != segment:
            continue
        held = steps.setdefault(r["step"], {}).setdefault(r["sample_pos"], {
            "sample_pos": r["sample_pos"], "global_index": r["global_index"], "rank": r["rank"],
            "sample_tokens": 33, "covered_tokens": 33, "rows": []})
        held["rows"].append({"shard": f"E/shard-{r['shard_position'] + 1}",
                             "shard_position": r["shard_position"], "ordinal": r["ordinal"],
                             "token_start": r["token_start"], "token_end": r["token_end"]})
    for (step, pos), tokens in (cuts or {}).items():
        held = steps[step][pos]
        spans = [(r["shard_position"], r["ordinal"], r["token_start"], r["token_end"])
                 for r in held["rows"]]
        held["covered_tokens"] = tokens
        held["rows"] = [{"shard": f"E/shard-{s + 1}", "shard_position": s, "ordinal": o,
                         "token_start": a, "token_end": b}
                        for s, o, a, b in join.cut_sample(spans, tokens)]
    return {"segment": segment, "route": "cache files", "cause": None, "cause_detail": None,
            "steps": [{"step": step, "consumed_samples": len(samples),
                       "samples": [samples[p] for p in sorted(samples)]}
                      for step, samples in sorted(steps.items())],
            "untraced_steps": untraced or []}


class Shown:
    """The world's store, with a command line that answers what the test says."""

    def __init__(self, tmp_path):
        self.store = tmp_path / "store.sqlite"
        self.out = tmp_path


@pytest.fixture
def joined(world, monkeypatch, tmp_path):
    """The world with its trace, its oracle's log, its store and its command line replaced."""
    state = {"traces": {}, "oracle": [], "check": 0, "answer": {}}
    monkeypatch.setattr(world, "trace", lambda store, segment: state["traces"][segment])
    monkeypatch.setattr(world, "trace_rows",
                        lambda store, segment: trace.flatten(state["traces"][segment]))
    monkeypatch.setattr(world, "oracle_runs", lambda store: tmp_path)
    monkeypatch.setattr(runs, "oracle_log", lambda root, segment: [
        r for r in state["oracle"] if r["segment"] == segment])
    monkeypatch.setattr(world, "store", lambda name: Shown(tmp_path))

    def cli(store, *arguments):
        if arguments[0] == "check":
            return table.Outcome(state["check"], "{}", "")
        return table.Outcome(0, json.dumps(state["answer"]), "")

    monkeypatch.setattr(world, "cli", cli)
    monkeypatch.setattr(world, "query_arguments", lambda question: ["query"])
    return state


def put(world: World, name: str, check: int) -> list[table.Mismatch]:
    """What ``grade`` does with a row, up to the question: interpret, drop the fields that
    wait, fill the spec's pending values, put it."""
    row = next(r for r in table.load_check(check) if r.id == name)
    question = table.interpret(row)
    question = question.without(question.waiting(NOT_BUILT))
    expect, empty = table.fill_pending(dict(question.expect), SPEC)
    assert not empty
    return world._join_rows(dataclasses.replace(question, expect=expect))


def fields(found: list[table.Mismatch]) -> set[str]:
    return {m.field.split(" ")[0] for m in found}


def e_rows() -> list[dict]:
    """A stand-in for run E's log, from the spec's positions: the sample of T at step 7,
    and the samples of steps 8 and 9 that hold the cut and the shared token."""
    rows = lines("T")
    rows += [{**r, "step": 8, "sample_pos": 3, "global_index": 31} for r in lines("step8_cut")]
    rows += [{"segment": "E/seg-2", "step": 8, "sample_pos": p, "global_index": 28 + p, "rank": 0,
              "shard_position": 0, "ordinal": p, "token_start": 0, "token_end": 33}
             for p in range(3)]
    rows += lines("shared_token")
    return rows


# --- c01.E.T and c01.E.shared-token ---------------------------------------------------------


def test_the_sample_of_t_is_found_whole_and_a_cut_row_is_found(world, joined):
    log = e_rows()
    joined["oracle"] = log
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts={(8, 3): 16})
    joined["traces"]["E/seg-1"] = document("E/seg-1", [])
    assert put(world, "c01.E.T", 1) == []
    broken = copy.deepcopy(joined["traces"]["E/seg-2"])
    broken["steps"][0]["samples"][0]["rows"][1]["token_end"] = 21
    joined["traces"]["E/seg-2"] = broken
    assert {"diff", "row"} <= fields(put(world, "c01.E.T", 1))


def test_t_in_a_second_row_of_the_run_is_found(world, joined):
    log = e_rows()
    joined["oracle"] = log
    t = POSITIONS["T"]
    extra = {**lines("T")[1], "segment": "E/seg-1", "step": 1, "sample_pos": 0, "global_index": 0}
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts={(8, 3): 16})
    joined["traces"]["E/seg-1"] = document("E/seg-1", [extra])
    assert t["rows_of_T_in_run_E"]
    assert "rows_of_T_in_run_E" in fields(put(world, "c01.E.T", 1))


def test_the_one_token_of_w_is_one_row_of_a_sample_of_33_tokens(world, joined):
    log = e_rows()
    joined["oracle"] = log
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts={(8, 3): 16})
    joined["traces"]["E/seg-1"] = document("E/seg-1", [])
    assert put(world, "c01.E.shared-token", 1) == []
    w = POSITIONS["shared_token"]
    second = {**lines("shared_token")[1], "segment": "E/seg-1", "step": 2, "sample_pos": 1,
              "global_index": 5}
    joined["traces"]["E/seg-1"] = document("E/seg-1", [second])
    assert "rows_of_W_in_steps_1_to_9" in fields(put(world, "c01.E.shared-token", 1))
    assert w["shard_position"] == second["shard_position"]


# --- The skipped line: ordinals stay in place --------------------------------------------------


def test_the_documents_of_a_shard_are_the_rows_show_prints_of_its_map(world, joined, monkeypatch):
    log = e_rows() + lines("skipped_line")
    joined["oracle"] = log
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts={(8, 3): 16})
    joined["traces"]["E/seg-1"] = document("E/seg-1", log)
    skip = {"record": "L6", "reason": "sanitizer skip"}   # the drop the row's other field reads
    monkeypatch.setattr(world, "shown", lambda store, *arguments, **kw: {
        "stored_rows": 19, "drops": [skip], "inputs": []})
    assert put(world, "c01.E.skipped-line", 1) == []
    monkeypatch.setattr(world, "shown", lambda store, *arguments, **kw: {
        "stored_rows": 20, "drops": [skip], "inputs": []})
    (found,) = put(world, "c01.E.skipped-line", 1)
    assert (found.field, found.expected, found.got) == ("documents_in_shard", 19, 20)


# --- The half-covered sample of step 8 -----------------------------------------------------


def cut_sample_of(document_: dict) -> dict:
    """The last sample of step 8 in a printed trace."""
    step = next(s for s in document_["steps"] if s["step"] == 8)
    return next(s for s in step["samples"] if s["sample_pos"] == 3)


def step8_world(joined, cuts):
    log = e_rows()
    joined["oracle"] = log
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts=cuts)
    joined["traces"]["E/seg-1"] = document("E/seg-1", [])


def test_the_cut_of_the_half_covered_sample_is_its_rows_and_its_mark(world, joined):
    step8_world(joined, {(8, 3): 16})
    for name in ("c01.E.step8.first-half", "c01.E.step8.uncovered", "c01.E.step8.counted"):
        assert put(world, name, 1) == [], name


def test_a_sample_that_is_whole_is_not_the_half_covered_one(world, joined):
    step8_world(joined, {})
    assert "covered_tokens" in fields(put(world, "c01.E.step8.first-half", 1))
    assert "uncovered_in_sample" in fields(put(world, "c01.E.step8.uncovered", 1))


def test_a_step_that_holds_three_samples_is_not_counted_as_four(world, joined):
    step8_world(joined, {(8, 3): 16})
    broken = copy.deepcopy(joined["traces"]["E/seg-2"])
    step = next(s for s in broken["steps"] if s["step"] == 8)
    step["consumed_samples"] = 3
    joined["traces"]["E/seg-2"] = broken
    assert "consumed_samples" in fields(put(world, "c01.E.step8.counted", 1))
    joined["check"] = 3
    assert "loud_stop" in fields(put(world, "c01.E.step8.counted", 1))


def test_the_row_that_holds_the_17th_token_ends_at_the_cut_and_no_later_row_is_reported(world, joined):
    step8_world(joined, {(8, 3): 16})
    kept = copy.deepcopy(joined["traces"]["E/seg-2"])
    sample = cut_sample_of(kept)
    sample["rows"] = [{**sample["rows"][0], "token_end": 44}]   # not cut: runs past 16 tokens
    joined["traces"]["E/seg-2"] = kept
    assert {"rows", "starts_at"} <= fields(put(world, "c01.E.step8.first-half", 1)) | fields(
        put(world, "c01.E.step8.uncovered", 1))


def test_checkpoint_9_rests_on_step_7_and_step_8_names_no_t(world, joined):
    step8_world(joined, {(8, 3): 16})
    joined["answer"] = {"first_exposure": {"segment": "E/seg-2", "step": 7}}
    assert put(world, "c08.withdraw-A.E@9.step-8-holds-no-T", 8) == []
    t = lines("T")[1]
    joined["oracle"] = e_rows() + [{**t, "step": 8, "sample_pos": 3, "global_index": 31}]
    leaky = copy.deepcopy(joined["traces"]["E/seg-2"])
    cut_sample_of(leaky)["rows"].append(
        {"shard": "E/shard-3", "shard_position": t["shard_position"], "ordinal": t["ordinal"],
         "token_start": 0, "token_end": 22})
    joined["traces"]["E/seg-2"] = leaky
    assert "rows_that_name_T" in fields(put(world, "c08.withdraw-A.E@9.step-8-holds-no-T", 8))
    joined["answer"] = {"first_exposure": {"segment": "E/seg-2", "step": 9}}
    assert "first_exposure" in fields(put(world, "c08.withdraw-A.E@9.step-8-holds-no-T", 8))


# --- Run K and the round-robin run ----------------------------------------------------------


def k_rows(steps: int = 3) -> list[dict]:
    rows = []
    for step in range(1, steps + 1):
        for pos in range(4):
            rows.append({"segment": "K/seg-1", "step": step, "sample_pos": pos,
                         "global_index": (step - 1) * 4 + pos, "rank": pos // 2,
                         "shard_position": 0, "ordinal": pos, "token_start": 0,
                         "token_end": 33})
    return rows


def test_every_step_of_run_k_holds_two_samples_of_each_rank(world, joined):
    joined["oracle"] = k_rows()
    joined["traces"]["K/seg-1"] = document("K/seg-1", k_rows())
    assert put(world, "c01.K.ranks", 1) == []
    broken = copy.deepcopy(joined["traces"]["K/seg-1"])
    broken["steps"][1]["samples"][1]["rank"] = 1
    joined["traces"]["K/seg-1"] = broken
    found = put(world, "c01.K.ranks", 1)
    assert found and "step 2" in found[0].field


def test_run_k_ranks_are_the_traces_own_and_a_one_rank_run_is_found(world, joined):
    joined["oracle"] = k_rows()
    rows = k_rows()
    for r in rows:
        r["rank"] = 0
    joined["traces"]["K/seg-1"] = document("K/seg-1", rows)
    assert fields(put(world, "c01.K.ranks", 1)) == {"K/seg-1"}


def test_a_trace_with_no_steps_or_with_fewer_steps_than_the_oracle_s_log_is_not_checked_out(
        world, joined):
    """The steps that must be there are the oracle's; what a trace prints of them is not
    the measure of what it left out."""
    joined["oracle"] = k_rows()
    joined["traces"]["K/seg-1"] = document("K/seg-1", [])
    found = put(world, "c01.K.ranks", 1)
    assert found and "K/seg-1 steps" in found[0].field and "[1, 2, 3]" in str(found[0].got)
    joined["traces"]["K/seg-1"] = document("K/seg-1", [r for r in k_rows() if r["step"] != 3])
    found = put(world, "c01.K.ranks", 1)
    assert found and "[3]" in str(found[0].got)


def k_log() -> list[dict]:
    """Run K's 40 steps, with the samples the spec records (positions.K) as they are."""
    recorded = lines("K")
    places = {(r["step"], r["sample_pos"]) for r in recorded}
    rows = [r for r in k_rows(40) if (r["step"], r["sample_pos"]) not in places]
    return sorted(rows + recorded, key=lambda r: (r["step"], r["sample_pos"]))


def k_document(rows: list[dict]) -> dict:
    """A printed trace of run K as the command prints it: the launches of the table, and the
    epoch of each step when an epoch holds 14 steps (the spec's positions.K.epoch_steps)."""
    made = document("K/seg-1", rows)
    made["launches"] = [
        {"position": 0, "world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0,
         "step_at_start": 0, "first_step": 1, "last_step": 3},
        {"position": 1, "world_size": 2, "data_parallel_size": 2, "consumed_at_start": 12,
         "step_at_start": 3, "first_step": 4, "last_step": 40}]
    for step in made["steps"]:
        step["epochs"] = [(step["step"] - 1) // 14]
    return made


def test_a_launch_is_the_steps_it_names_and_a_step_it_lacks_is_found(world, joined):
    log = k_log()
    joined["oracle"] = log
    joined["traces"]["K/seg-1"] = k_document(log)
    for name in ("c01.K.warm-start", "c01.K.resume", "c01.K.epoch-boundary"):
        assert put(world, name, 1) == [], name
    gap = k_document([r for r in log if r["step"] != 20])
    joined["traces"]["K/seg-1"] = gap
    assert {"first_step", "diff"} <= fields(put(world, "c01.K.resume", 1))
    assert put(world, "c01.K.warm-start", 1) == []   # step 20 is not in launch 0


def test_a_line_of_the_spec_that_the_trace_lacks_is_found_in_the_launch_that_holds_it(
        world, joined):
    log = k_log()
    joined["oracle"] = log
    changed = copy.deepcopy(k_document(log))
    sample = next(s for step in changed["steps"] if step["step"] == 15 for s in step["samples"]
                  if s["sample_pos"] == 0)
    sample["rows"][0]["token_end"] = 23
    joined["traces"]["K/seg-1"] = changed
    assert {"diff", "rows"} <= fields(put(world, "c01.K.resume", 1))
    assert put(world, "c01.K.warm-start", 1) == []


def rr_rows(segment: str, steps=(1, 2, 3)) -> list[dict]:
    return [{"segment": segment, "step": step, "sample_pos": 0, "global_index": step, "rank": 0,
             "shard_position": 0, "ordinal": step, "token_start": 0, "token_end": 33}
            for step in steps]


def test_a_run_with_a_step_that_has_no_trace_is_not_fully_traced(world, joined):
    segments = runs.segment_names(SPEC, "E-rr")
    joined["oracle"] = [row for name in segments for row in rr_rows(name)]
    for name in segments:
        joined["traces"][name] = document(name, rr_rows(name))
    assert put(world, "c01.E-rr.fully-traced", 1) == []
    joined["traces"][segments[0]]["untraced_steps"] = [[segments[0], 3]]
    assert fields(put(world, "c01.E-rr.fully-traced", 1)) == {"steps_without_trace"}
    joined["traces"][segments[0]]["untraced_steps"] = []
    joined["traces"][segments[0]] = document(
        segments[0], rr_rows(segments[0]), cuts={(1, 0): 16})
    assert fields(put(world, "c01.E-rr.fully-traced", 1)) == {"steps_with_uncovered_range"}


def test_a_trace_that_prints_no_step_is_not_fully_traced(world, joined):
    """The empty trace passes no row that asks what it covers: the steps are the oracle's."""
    segments = runs.segment_names(SPEC, "E-rr")
    joined["oracle"] = [row for name in segments for row in rr_rows(name)]
    for name in segments:
        joined["traces"][name] = document(name, [])
    found = put(world, "c01.E-rr.fully-traced", 1)
    assert [m.field for m in found] == ["steps_without_trace"]
    assert found[0].got == [1, 2, 3]


def test_a_step_8_trace_without_the_last_sample_is_a_finding_for_the_row_about_t(world, joined):
    step8_world(joined, {(8, 3): 16})
    gone = copy.deepcopy(joined["traces"]["E/seg-2"])
    step = next(s for s in gone["steps"] if s["step"] == 8)
    step["samples"] = [s for s in step["samples"] if s["sample_pos"] != 3]
    joined["traces"]["E/seg-2"] = gone
    joined["answer"] = {"first_exposure": {"segment": "E/seg-2", "step": 7}}
    found = put(world, "c08.withdraw-A.E@9.step-8-holds-no-T", 8)
    assert fields(found) == {"sample"}


# --- Every field of every row is read, or waits by name ------------------------------------


def join_questions():
    out = []
    for check in (1, 8):
        for row in table.load_check(check):
            question = table.interpret(row)
            if question.kind == table.KIND_JOIN_ROWS:
                out.append(pytest.param(question, id=row.id))
    return out


@pytest.mark.parametrize("question", join_questions())
def test_no_field_of_a_row_that_reads_the_trace_ends_in_an_error_of_the_world(question):
    """A field with no comparison would stop the row with an error once ``show --trace``
    exists; each is either compared (``JOIN_KEYS``), prose, or deferred by name."""
    unread = (set(question.expect) - World.JOIN_KEYS - World.PROSE_KEYS - set(question.deferred))
    assert unread == set(), f"{question.row.id}: {sorted(unread)}"


def test_no_field_of_a_row_that_reads_the_trace_is_left_out_of_the_row_for_good():
    """The launch's counter, the skipped line and the step after the boundary are read from
    the forms that print them (test_launches_epochs_shuffle.py); none waits as a field that
    no output gives."""
    for param in join_questions():
        question = param.values[0]
        assert question.deferred == {}, question.row.id
    by = {q.values[0].row.id: q.values[0] for q in join_questions()}
    assert "consumed_at_start" in by["c01.K.warm-start"].expect
    assert "consumed_at_start" in by["c01.K.resume"].expect
    assert "skipped_line" in by["c01.E.skipped-line"].expect
    assert "boundary_after_step" in by["c01.K.epoch-boundary"].expect
