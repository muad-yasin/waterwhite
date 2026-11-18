"""The rows of Checks 1 and 8 that read the run record's launches, the epoch of each step, the
skipped line of a shard and the shuffle ``register-run`` prints, put to the world with the
printed documents written by hand.

Each form is read by a pure function of ``acceptance_trace`` (tested first, on documents) and
put to a row by ``World`` (tested second, with the trace, the drops and the registration
replaced). A form the command does not print skips the row by name, after everything else the
row names was compared; a form it prints wrongly fails the row. The values are the table's
and the spec's (run K: launch 0 reads steps 1 to 3 from a counter of 0, launch 1 steps 4 to 40
from 12; an epoch is 14 steps; two ranks), never an output of the tool.
"""

from __future__ import annotations

import copy
import json

import acceptance_runs as runs
import acceptance_trace as trace
import acceptance_world
import pytest
import table
from test_join_row_fields import (  # noqa: F401
    document, e_rows, fields, joined, k_log, lines, put)

SPEC = runs.spec()
LAUNCHES = [
    {"position": 0, "world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0,
     "step_at_start": 0, "first_step": 1, "last_step": 3},
    {"position": 1, "world_size": 2, "data_parallel_size": 2, "consumed_at_start": 12,
     "step_at_start": 3, "first_step": 4, "last_step": 40},
]


def k_trace(*, launches: bool = True, epochs: bool = True, epoch_steps: int = 14,
            log: list[dict] | None = None) -> dict:
    """A printed trace of run K: its 40 steps, the launches of the table and, for each step,
    the epoch it is read in when an epoch holds ``epoch_steps`` steps."""
    made = document("K/seg-1", log or k_log())
    if launches:
        made["launches"] = copy.deepcopy(LAUNCHES)
    if epochs:
        for step in made["steps"]:
            step["epochs"] = [(step["step"] - 1) // epoch_steps]
    return made


# --- The pure half: launches, epochs, shuffle ------------------------------------------------


def test_a_trace_with_no_launches_field_prints_no_launches_and_an_empty_list_prints_none():
    assert trace.launches({"steps": []}) is None
    assert trace.launches({"steps": [], "launches": []}) == []
    assert trace.launch_at(LAUNCHES, 1)["consumed_at_start"] == 12
    assert trace.launch_at(LAUNCHES, 2) is None
    with pytest.raises(trace.TraceError):
        trace.launches({"steps": [], "launches": {"position": 0}})


def test_the_epoch_boundary_is_the_last_step_of_the_first_epoch():
    assert trace.epoch_boundary(k_trace(), 0, 1) == (14, [])


def test_an_epoch_that_is_one_step_short_or_long_moves_the_boundary():
    assert trace.epoch_boundary(k_trace(epoch_steps=13), 0, 1)[0] == 13
    assert trace.epoch_boundary(k_trace(epoch_steps=15), 0, 1)[0] == 15


def test_a_step_that_reads_two_epochs_is_a_finding():
    made = k_trace()
    made["steps"][14]["epochs"] = [0, 1]  # step 15
    boundary, found = trace.epoch_boundary(made, 0, 1)
    assert any("step 15 reads two epochs" in each for each in found)


def test_epochs_that_skip_one_start_late_or_are_no_list_are_findings():
    made = k_trace()
    for step in made["steps"][14:]:
        step["epochs"] = [step["epochs"][0] + 1]
    assert any("epoch 2 after epoch 0" in each for each in trace.epoch_boundary(made, 0, 1)[1])
    made = k_trace()
    for step in made["steps"]:
        step["epochs"] = [step["epochs"][0] + 1]
    assert any("not 0" in each for each in trace.epoch_boundary(made, 0, 1)[1])
    made = k_trace()
    made["steps"][3]["epochs"] = 0
    assert any("step 4 prints epochs 0" in each for each in trace.epoch_boundary(made, 0, 1)[1])


def test_an_epoch_that_never_changes_has_no_boundary():
    made = k_trace(epoch_steps=1000)
    assert trace.epoch_boundary(made, 0, 1) == (None, [])
    assert trace.epochs_printed(made)
    assert not trace.epochs_printed(k_trace(epochs=False))


CELLS = [(launch, epoch, rank) for launch, epochs in ((0, (0,)), (1, (0, 1, 2)))
         for epoch in epochs for rank in (0, 1)]


def shuffle(cells=CELLS) -> list[dict]:
    return [{"launch": launch, "epoch": epoch, "rank": rank,
             "digest": f"d{epoch}{rank}", "count": 56, "recorded_digest": f"d{epoch}{rank}",
             "recorded_count": 56, "equal": True} for launch, epoch, rank in cells]


# What the run record the replica wrote holds for each cell: the digest and the count.
RECORD = {(each["launch"], each["epoch"], each["rank"]): (each["digest"], each["count"])
          for each in shuffle()}


def test_a_shuffle_with_one_equal_entry_for_each_cell_has_no_finding():
    assert trace.shuffle_findings(shuffle(), CELLS) == []
    assert trace.shuffle_findings(shuffle(), CELLS, RECORD) == []


def test_a_tool_that_prints_its_own_digest_as_the_recorded_one_is_held_to_the_run_record():
    """``equal`` true and ``digest`` equal to ``recorded_digest`` are the tool's own words: the
    digest and the count must be the run record's, whatever the tool says it recorded."""
    echoing = shuffle()
    echoing[4].update({"digest": "own", "recorded_digest": "own"})
    assert trace.shuffle_findings(echoing, CELLS) == []   # the printed object alone agrees
    found = trace.shuffle_findings(echoing, CELLS, RECORD)
    assert any("the tool says it recorded 'own'" in each for each in found)
    assert any("the tool computed 'own'" in each for each in found)
    counted = shuffle()
    counted[0].update({"count": 55, "recorded_count": 55})
    assert trace.shuffle_findings(counted, CELLS, RECORD)


def test_a_missing_extra_or_misordered_entry_is_a_finding():
    assert trace.shuffle_findings(shuffle()[:-1], CELLS)
    assert trace.shuffle_findings(shuffle() + shuffle()[:1], CELLS)
    assert trace.shuffle_findings(list(reversed(shuffle())), CELLS)
    assert trace.shuffle_findings({}, CELLS)


def test_an_entry_that_is_not_equal_or_whose_digest_or_count_differs_is_a_finding():
    for change in ({"equal": False}, {"digest": "other"}, {"count": 55}, {"recorded_digest": ""},
                   {"recorded_digest": None, "digest": None}):
        made = shuffle()
        made[3].update(change)
        assert trace.shuffle_findings(made, CELLS), change


def test_epoch_0_digests_of_the_launches_must_agree_rank_by_rank():
    made = shuffle()
    cell = next(each for each in made if (each["launch"], each["epoch"], each["rank"]) == (1, 0, 1))
    cell["digest"] = cell["recorded_digest"] = "elsewhere"  # equal to its own record, not launch 0's
    found = trace.shuffle_findings(made, CELLS)
    assert any("rank 1: the epoch 0 digests of the launches differ" in each for each in found)


# --- The launches and the epochs, put to the rows of Check 1 ------------------------------------


def with_k(joined, **kwargs):  # noqa: F811
    log = k_log()
    joined["oracle"] = log
    joined["traces"]["K/seg-1"] = k_trace(**kwargs)


def test_run_k_launches_and_epochs_as_the_table_has_them_pass_every_row(world, joined):  # noqa: F811
    with_k(joined)
    for name in ("c01.K.warm-start", "c01.K.resume", "c01.K.epoch-boundary"):
        assert put(world, name, 1) == [], name


def test_a_counter_or_a_step_of_a_launch_that_differs_is_found(world, joined):  # noqa: F811
    with_k(joined)
    joined["traces"]["K/seg-1"]["launches"][1]["consumed_at_start"] = 11
    assert fields(put(world, "c01.K.resume", 1)) == {"consumed_at_start"}
    assert put(world, "c01.K.warm-start", 1) == []
    joined["traces"]["K/seg-1"]["launches"][0]["consumed_at_start"] = 24  # the counter of E@6
    assert fields(put(world, "c01.K.warm-start", 1)) == {"consumed_at_start"}
    joined["traces"]["K/seg-1"]["launches"][0]["last_step"] = 4
    assert fields(put(world, "c01.K.warm-start", 1)) == {"consumed_at_start", "last_step"}


def test_a_record_with_no_launch_at_the_position_is_found(world, joined):  # noqa: F811
    with_k(joined)
    joined["traces"]["K/seg-1"]["launches"] = LAUNCHES[:1]
    assert fields(put(world, "c01.K.resume", 1)) == {"launch"}
    joined["traces"]["K/seg-1"]["launches"] = []
    assert fields(put(world, "c01.K.warm-start", 1)) == {"launch"}


def test_a_trace_that_prints_no_launches_fails_the_row_by_name(  # noqa: F811
        world, joined):
    # The tool prints the launches (docs/cli.md): a trace without them is a failure of
    # the row, never a skip.
    with_k(joined, launches=False)
    for name in ("c01.K.warm-start", "c01.K.resume"):
        assert "show --trace launches" in {m.field for m in put(world, name, 1)}
    gap = k_trace(launches=False, log=[r for r in k_log() if r["step"] != 20])
    joined["traces"]["K/seg-1"] = gap
    assert {"first_step", "diff"} <= fields(put(world, "c01.K.resume", 1))  # a finding, no skip


def test_the_boundary_after_step_14_passes_and_any_other_is_found(world, joined):  # noqa: F811
    with_k(joined)
    assert put(world, "c01.K.epoch-boundary", 1) == []
    with_k(joined, epoch_steps=13)
    assert fields(put(world, "c01.K.epoch-boundary", 1)) == {"boundary_after_step"}
    made = k_trace()
    made["steps"][14]["epochs"] = [0, 1]
    joined["traces"]["K/seg-1"] = made
    assert "epochs" in fields(put(world, "c01.K.epoch-boundary", 1))


def test_a_trace_that_prints_no_epochs_fails_the_row_by_name(world, joined):  # noqa: F811
    with_k(joined, epochs=False)
    assert "show --trace epochs" in {m.field for m in put(world, "c01.K.epoch-boundary", 1)}
    assert put(world, "c01.K.warm-start", 1) == []


# --- The skipped line, put to the row of Check 1 ------------------------------------------------


def shard_documents(shard_drops, input_drops):
    def shown(store, name, *rest, **kwargs):
        if name == "E/shard-2":
            return {"stored_rows": 19, "drops": shard_drops,
                    "inputs": [{"name": "E/shard-2.input", "registered": True},
                               {"name": "E/shard-2.gone", "registered": False}]}
        assert name == "E/shard-2.input", name
        return {"drops": input_drops}
    return shown


def drop(record="L6", reason="sanitizer skip"):
    return {"artifact": "E/shard-2.sanitizer-input", "instance": "x", "record": record,
            "reason": reason, "step": "sanitize"}


def e_world(joined):  # noqa: F811
    log = e_rows() + lines("skipped_line")
    joined["oracle"] = log
    joined["traces"]["E/seg-2"] = document("E/seg-2", log, cuts={(8, 3): 16})
    joined["traces"]["E/seg-1"] = document("E/seg-1", log)


def test_the_skipped_line_is_the_one_drop_of_the_shard_s_input_that_names_line_6(
        world, joined, monkeypatch):  # noqa: F811
    e_world(joined)
    monkeypatch.setattr(world, "shown", shard_documents([], [drop()]))
    assert put(world, "c01.E.skipped-line", 1) == []
    monkeypatch.setattr(world, "shown", shard_documents([drop()], []))
    assert put(world, "c01.E.skipped-line", 1) == []  # a drop recorded under the shard itself


def test_no_drop_another_line_another_reason_or_two_drops_are_found(
        world, joined, monkeypatch):  # noqa: F811
    e_world(joined)
    for held in ([], [drop("L7")], [drop(reason="empty line")], [drop(), drop("L9")]):
        monkeypatch.setattr(world, "shown", shard_documents([], held))
        assert fields(put(world, "c01.E.skipped-line", 1)) == {"skipped_line"}, held


# --- The shuffle, put to the row of Check 8 --------------------------------------------------------


def digests_row() -> table.Question:
    return table.interpret(next(r for r in table.load_check(8) if r.id == "c08.K.digests-match"))


@pytest.fixture
def registered(world, monkeypatch, tmp_path):
    state = {"printed": {"K/seg-1": {"segment": "K/seg-1", "shuffle": shuffle()}}}
    monkeypatch.setattr(world, "registered", lambda store, st="unmutated": state["printed"])
    record = tmp_path / "K-seg-1.run.json"   # the run record the replica wrote
    record.write_text(json.dumps({"shuffle_digests": [
        {"launch": launch, "epoch": epoch, "rank": rank, "digest": digest, "count": count}
        for (launch, epoch, rank), (digest, count) in RECORD.items()]}), encoding="utf-8")
    monkeypatch.setattr(world, "_record_path", lambda store, state_name, name: record)
    return state


def test_the_row_of_the_shuffle_needs_only_the_runs():
    assert digests_row().needs == (table.NEED_RUN_CAPTURE,)


def test_eight_equal_digests_one_for_each_launch_epoch_and_rank_pass(world, registered):
    assert world._k_digests(digests_row()) == []


def test_a_cell_that_is_missing_a_digest_that_differs_or_a_launch_that_differs_is_found(
        world, registered):
    registered["printed"]["K/seg-1"]["shuffle"] = shuffle()[:-1]
    assert [m.field for m in world._k_digests(digests_row())] == ["shuffle_digests"]
    registered["printed"]["K/seg-1"]["shuffle"] = shuffle()
    registered["printed"]["K/seg-1"]["shuffle"][2]["equal"] = False
    assert world._k_digests(digests_row())
    registered["printed"]["K/seg-1"]["shuffle"] = shuffle()
    registered["printed"]["K/seg-1"]["shuffle"][3]["digest"] = "other"
    assert world._k_digests(digests_row())


def test_a_shuffle_that_agrees_with_itself_but_not_with_the_run_record_is_found(
        world, registered):
    printed = registered["printed"]["K/seg-1"]["shuffle"]
    printed[4].update({"digest": "own", "recorded_digest": "own"})
    printed[5].update({"count": 55, "recorded_count": 55})
    found = world._k_digests(digests_row())
    assert [m.field for m in found] == ["shuffle_digests"]
    assert "the tool says it recorded" in found[0].got


def test_a_run_record_whose_cells_are_not_the_launches_cells_is_an_error_of_the_world(
        world, registered, tmp_path, monkeypatch):
    record = tmp_path / "short.run.json"
    record.write_text(json.dumps({"shuffle_digests": [
        {"launch": 0, "epoch": 0, "rank": 0, "digest": "d00", "count": 56}]}), encoding="utf-8")
    monkeypatch.setattr(world, "_record_path", lambda store, state_name, name: record)
    with pytest.raises(acceptance_world.WorldError, match="the run record holds"):
        world._k_digests(digests_row())


def test_a_registration_that_prints_no_shuffle_fails_the_row_by_name(world, registered):
    del registered["printed"]["K/seg-1"]["shuffle"]
    assert [m.field for m in world._k_digests(digests_row())] == ["register-run's shuffle"]
