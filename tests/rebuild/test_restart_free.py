"""The exposure-free restart and the checkpoints that carry exposure (plan 4.3,
Decision 23; docs/interfaces.md section 7.1 "Restart points", docs/rebuild.md
section 4), on the hand-built world of ``rb_kit.py``.

The world's runs, read off its construction (``tests/coverage/ww_world.py``):

- run E: segment 1 reads shard-1 only (steps 1 to 3, checkpoint E@3); segment 2
  resumes from E@3 on mix-2 and reads T at step 4 (checkpoints E@4 and E@6);
- run D warm-starts from E@6 and has the one checkpoint D@2;
- run Y branches from E@3, trains on mix-2 and never reads T (checkpoint Y@2).

Under "withdraw A" E@3 and Y@2 answer "does not contain", E@4, E@6 and D@2 "contains".
Under "withdraw Z" the third step of segment 1 reads a row of Z, so no checkpoint of
any run is free of it.
"""

from __future__ import annotations

from waterwhite import coverage, requests
from waterwhite import vocabulary as v

from rb_kit import need_attr


def restarts(world, source: str = "A") -> dict[str, dict]:
    function = need_attr("waterwhite.rebuild.restart_free", "exposure_free_restarts")
    request = requests.withdraw(world.store, source)
    entries = function(world.store, request, coverage.Session(world.store, request))
    assert [e["run"] for e in entries] == ["E", "D", "Y"], "one entry per run, in registration order"
    return {e["run"]: e for e in entries}


def exposed(world, source: str = "A") -> list[dict]:
    function = need_attr("waterwhite.rebuild.restart_free", "exposed_checkpoints")
    request = requests.withdraw(world.store, source)
    return function(world.store, request, coverage.Session(world.store, request))


def test_the_last_checkpoint_free_of_exposure_is_the_restart_of_its_run(world):
    by_run = restarts(world)
    assert by_run["E"] == {
        "run": "E", "proven": True, "checkpoint": "E@3", "segment": "E/seg-1", "step": 3,
        "basis": v.RESTART_BASIS_LAST_CLEAR, "from_scratch": None,
    }
    assert by_run["Y"]["proven"] is True and by_run["Y"]["checkpoint"] == "Y@2"
    assert (by_run["Y"]["segment"], by_run["Y"]["step"]) == ("Y/seg-1", 2)


def test_a_run_none_of_whose_checkpoints_qualifies_names_the_first_step_of_its_lineage(world):
    # No row of the hand-written table asks this (plan 4.3 states it): D has one
    # checkpoint, which carries the exposure of E, so it names the first step of the
    # run it descends from, as the restart that is free of exposure by construction.
    entry = restarts(world)["D"]
    assert entry["proven"] is False
    assert entry["checkpoint"] is None and entry["segment"] is None and entry["step"] is None
    assert entry["basis"] == v.RESTART_NOT_PROVEN == "no exposure-free restart is proven"
    assert entry["from_scratch"] == {
        "run": "E", "segment": "E/seg-1", "step": 1, "basis": v.RESTART_BASIS_LINEAGE_START,
    }


def test_exposure_in_the_first_segment_leaves_no_run_with_a_proven_restart(world):
    # No row of the table asks this either. Step 3 of segment 1 reads the row of Z
    # that R2 and shard-1 hold, so every checkpoint of E and of the runs below it carries it.
    for run, entry in restarts(world, "Z").items():
        assert entry["proven"] is False, run
        assert entry["basis"] == v.RESTART_NOT_PROVEN and entry["checkpoint"] is None, run
        assert entry["from_scratch"] == {
            "run": "E", "segment": "E/seg-1", "step": 1, "basis": v.RESTART_BASIS_LINEAGE_START,
        }, run


def test_an_untraced_step_before_a_checkpoint_stops_it_from_qualifying(world):
    world.delete_trace("E/seg-1", 2)
    by_run = restarts(world)
    # E@3 cannot be shown clean: a step it consumed has no trace.
    assert by_run["E"]["proven"] is False and by_run["E"]["checkpoint"] is None
    assert by_run["E"]["from_scratch"]["step"] == 1
    assert by_run["Y"]["proven"] is False


def test_the_checkpoints_that_carry_exposure_are_listed_with_their_first_exposure(world):
    entries = {e["checkpoint"]: e for e in exposed(world)}
    assert set(entries) == {"E@4", "E@6", "D@2"}
    assert {name: (e["run"], e["segment"], e["step"], e["answer"]) for name, e in entries.items()} == {
        "E@4": ("E", "E/seg-2", 4, "contains"),
        "E@6": ("E", "E/seg-2", 6, "contains"),
        "D@2": ("D", "D/seg-1", 2, "contains"),
    }
    for name, entry in entries.items():
        # The record is read at step 4 of E's second segment, whichever checkpoint inherits it.
        assert entry["first_exposure"] == {"segment": "E/seg-2", "step": 4}, name
    # Step 5 of E's second segment was read for 3 of its 5 tokens only (the operator's note),
    # so the checkpoints after it name it as untraced.
    assert entries["E@4"]["untraced_steps"] == []
    assert entries["E@6"]["untraced_steps"] == [["E/seg-2", 5]]
    assert entries["D@2"]["untraced_steps"] == [["E/seg-2", 5]]


def test_a_checkpoint_that_may_contain_the_source_is_listed_with_its_untraced_steps(world):
    world.delete_trace("E/seg-1", 2)
    entries = {e["checkpoint"]: e for e in exposed(world)}
    assert entries["E@3"]["answer"] == "may contain"
    assert entries["E@3"]["untraced_steps"] == [["E/seg-1", 2]]
    assert entries["E@3"]["first_exposure"] is None


def test_no_count_of_reads_is_given(world):
    # Exposure counts belong to an extension that is off (plan 2.4).
    for entry in exposed(world):
        assert not [key for key in entry if "count" in key or "reads" in key], entry
