"""The exact-rerun restart in the report (plan 4.3; docs/interfaces.md sections 7.1 and
7.3), on the hand-built world of ``rb_kit.py``.

Every entry of the report's ``restarts`` gains the key ``exact_rerun``, placed last:
null for a run with no stale segment, otherwise the object
``{"run_record_replayable", "segment", "step", "basis", "step_comparison"}``.

- A run whose record holds every setting a replay needs (sequence length, random
  seed, split, add-extra-token, and the train iterations or samples with the global
  batch size) is replayable: the report names no step for it and leaves the restart
  to ``verify`` (plan 4.3; docs/interfaces.md 7.1), so ``step`` and
  ``step_comparison`` are null.
- A run whose record lacks any of them is not: the report gives the phase start, the
  first step of the earliest stale segment of the run, as an assumption, and says
  that no step comparison is possible.

A segment is stale where its mixture needs a successor (the report's ``mixtures``).
In the world, mix-2 needs one, and run E reads it from segment E/seg-2 (first step 4)
and run Y from segment Y/seg-1 (first step 1); run D reads mix-1 only.

``verify``'s ``--mode`` (the build mode of plan 4.2) has nothing to do with these
tests. Where a test needs a value of ``verify``, it runs the command on the report the
command line printed.

The wording of the replayable restart is the vocabulary constant
``RESTART_EXACT_BY_VERIFY`` (docs/interfaces.md sections 7.1 and 7.3), which a test
reads with a fallback so that a missing constant is a failed assertion.
"""

from __future__ import annotations

import json
import re

import pytest
from waterwhite import vocabulary as v

import rb_kit
from rb_kit import report_through_cli, run, write_json

BY_VERIFY = "stated by the verification of the repair, once the successor shards exist"
FORM = ["run_record_replayable", "segment", "step", "basis", "step_comparison"]
ENTRY_BEFORE = ["run", "proven", "checkpoint", "segment", "step", "basis", "from_scratch"]
SETTINGS = ["sequence_length", "split", "add_extra_token_to_sequence", "train_iters",
            "global_batch_size"]


def replayable(segment: str) -> dict:
    return {"run_record_replayable": True, "segment": segment, "step": None,
            "basis": BY_VERIFY, "step_comparison": None}


def phase_start(segment: str, step: int) -> dict:
    return {"run_record_replayable": False, "segment": segment, "step": step,
            "basis": v.RESTART_BASIS_PHASE_START,
            "step_comparison": v.STEP_COMPARISON_NOT_POSSIBLE}


def entries(document: dict) -> dict[str, dict]:
    return {entry["run"]: entry for entry in document["restarts"]}


def exact(document: dict, run: str):
    """The ``exact_rerun`` of a run's entry, after the assertion that the key is there."""
    found = entries(document)
    assert run in found, f"no restarts entry for run {run}: {list(found)}"
    assert "exact_rerun" in found[run], f"the entry of run {run} has no exact_rerun: {list(found[run])}"
    return found[run]["exact_rerun"]


def verified(world, tmp_path, document: dict) -> dict[str, dict]:
    """The ``restarts`` entries of ``verify`` on the report the command line printed,
    by run."""
    path = write_json(tmp_path / "printed-report.json", document)
    code, out, err = run(world, "verify", "--report", str(path))
    assert code == 0, err
    return {entry["run"]: entry for entry in json.loads(out)["restarts"]}


def build_missing(monkeypatch, tmp_path, setting: str, only_run: str | None = None):
    """The repair world whose run records lack one setting, for every run or for one."""
    saved = rb_kit.ww.run_record

    def without(run, *args, **kwargs):
        record = saved(run, *args, **kwargs)
        if only_run is None or run == only_run:
            record["settings"][setting] = None
        return record

    monkeypatch.setattr(rb_kit.ww, "run_record", without)
    return rb_kit.build(tmp_path / "w", successors=True)


# --- The form ----------------------------------------------------------------------------


def test_the_new_wording_is_a_constant_of_the_vocabulary():
    assert getattr(v, "RESTART_EXACT_BY_VERIFY", None) == BY_VERIFY


@pytest.mark.parametrize("fixture", ["world", "repair_world", "unseeded_repair_world"])
def test_every_restarts_entry_ends_with_exact_rerun_and_keeps_its_other_keys(fixture, request):
    document = report_through_cli(request.getfixturevalue(fixture))
    assert [e["run"] for e in document["restarts"]] == ["E", "D", "Y"]
    for entry in document["restarts"]:
        assert "exact_rerun" in entry, f"run {entry['run']}: {list(entry)}"
        assert list(entry) == ENTRY_BEFORE + ["exact_rerun"], entry["run"]


def test_a_run_with_no_stale_segment_has_null(world):
    # Run D reads mix-1, which the report keeps.
    assert exact(report_through_cli(world), "D") is None


# --- A run whose record holds every setting --------------------------------------------


@pytest.mark.parametrize("fixture", ["world", "repair_world"])
def test_a_replayable_run_names_the_verification_and_no_step(fixture, request):
    document = report_through_cli(request.getfixturevalue(fixture))
    for run_name, segment in (("E", "E/seg-2"), ("Y", "Y/seg-1")):
        found = exact(document, run_name)
        assert found == replayable(segment), run_name
        assert list(found) == FORM, run_name
        assert found["run_record_replayable"] is True


# --- A run whose record lacks the seed ---------------------------------------------------


def test_an_unseeded_run_gets_the_phase_start_as_an_assumption(unseeded_repair_world):
    document = report_through_cli(unseeded_repair_world)
    for run_name, segment, step in (("E", "E/seg-2", 4), ("Y", "Y/seg-1", 1)):
        found = exact(document, run_name)
        assert found == phase_start(segment, step), run_name
        assert list(found) == FORM, run_name
        assert found["run_record_replayable"] is False and isinstance(found["step"], int)
    assert exact(document, "D") is None


# --- A record that holds the seed and lacks another setting ------------------------------


@pytest.mark.parametrize("setting", SETTINGS)
def test_a_record_with_a_seed_and_without_another_setting_is_not_replayable(
        setting, tmp_path, monkeypatch):
    world = build_missing(monkeypatch, tmp_path, setting)
    try:
        document = report_through_cli(world)
        assert exact(document, "E") == phase_start("E/seg-2", 4), setting
        assert exact(document, "Y") == phase_start("Y/seg-1", 1), setting
        assert exact(document, "D") is None
        stated = verified(world, tmp_path, document)
        for run_name in ("E", "Y"):
            reported = exact(document, run_name)
            made = stated[run_name]
            assert (made["segment"], made["step"], made["basis"], made["step_comparison"]) == (
                reported["segment"], reported["step"], reported["basis"],
                reported["step_comparison"]), f"{setting}: {run_name}"
    finally:
        world.store.close()


def test_the_check_is_made_run_by_run(tmp_path, monkeypatch):
    # Run E lacks the split, run Y holds every setting.
    world = build_missing(monkeypatch, tmp_path, "split", only_run="E")
    try:
        document = report_through_cli(world)
        assert exact(document, "E") == phase_start("E/seg-2", 4)
        assert exact(document, "Y") == replayable("Y/seg-1")
        stated = verified(world, tmp_path, document)
        assert set(stated) == {"E", "Y"}
        assert (stated["E"]["segment"], stated["E"]["step"], stated["E"]["basis"]) == (
            "E/seg-2", 4, v.RESTART_BASIS_PHASE_START)
    finally:
        world.store.close()


# --- A run that has a stale segment and no checkpoint ------------------------------------


@pytest.mark.parametrize("seeded", [True, False])
def test_a_run_with_a_stale_segment_and_no_checkpoint_still_has_an_entry(
        seeded, tmp_path, monkeypatch):
    world = rb_kit.build(tmp_path / "w", successors=True)
    try:
        saved = rb_kit.ww.run_record

        def record(*args, **kwargs):
            made = saved(*args, **kwargs)
            if not seeded:
                made["settings"]["random_seed"] = None
            return made

        monkeypatch.setattr(rb_kit.ww, "run_record", record)
        # Run Q reads mix-2 in one segment and registers no checkpoint.
        rb_kit.ww.segment(world.store, "Q", "Q/seg-1", 0, world["mix-2"], None,
                          [(1, 1, None)], {1: (1, 1)})
        document = report_through_cli(world)
        want = replayable("Q/seg-1") if seeded else phase_start("Q/seg-1", 1)
        assert exact(document, "Q") == want
        assert exact(document, "E") is not None and exact(document, "D") is None
    finally:
        world.store.close()


# --- The text form -----------------------------------------------------------------------


def statement_lines(text: str, sentence: str) -> list[str]:
    return [line for line in text.splitlines() if sentence in line]


def test_the_text_form_names_the_verification_for_each_replayable_run(repair_world):
    code, out, err = run(repair_world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    lines = statement_lines(out, "stated by the verification of the repair")
    assert len(lines) == 2, f"one line per run with a stale segment, got {lines}"
    by_run = {name: [line for line in lines if re.search(rf"\b{name}\b", line)] for name in "EDY"}
    assert [len(by_run[name]) for name in "EDY"] == [1, 0, 1], by_run
    assert "E/seg-2" in by_run["E"][0] and "Y/seg-1" in by_run["Y"][0]


def test_the_text_form_gives_the_phase_start_for_each_unseeded_run(unseeded_repair_world):
    code, out, err = run(unseeded_repair_world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    lines = statement_lines(out, v.RESTART_BASIS_PHASE_START)
    assert len(lines) == 2, f"one line per run with a stale segment, got {lines}"
    by_run = {name: [line for line in lines if re.search(rf"\b{name}\b", line)] for name in "EDY"}
    assert [len(by_run[name]) for name in "EDY"] == [1, 0, 1], by_run
    assert "E/seg-2" in by_run["E"][0] and "4" in by_run["E"][0]
    assert "Y/seg-1" in by_run["Y"][0]
    for line in lines:
        assert v.STEP_COMPARISON_NOT_POSSIBLE in line


# --- The report and verify state the same restart ----------------------------------------


def test_the_runs_with_an_exact_rerun_are_the_runs_of_the_restarts_of_verify(repair_world, tmp_path):
    document = report_through_cli(repair_world)
    stated = verified(repair_world, tmp_path, document)
    with_one = {e["run"] for e in document["restarts"] if "exact_rerun" in e and e["exact_rerun"] is not None}
    assert with_one == set(stated) == {"E", "Y"}


def test_for_an_unseeded_run_the_report_and_verify_state_the_same_restart(
        unseeded_repair_world, tmp_path):
    document = report_through_cli(unseeded_repair_world)
    stated = verified(unseeded_repair_world, tmp_path, document)
    assert set(stated) == {"E", "Y"}
    for run_name in ("E", "Y"):
        reported = exact(document, run_name)
        assert reported is not None, run_name
        made = stated[run_name]
        assert (made["segment"], made["step"], made["basis"], made["step_comparison"]) == (
            reported["segment"], reported["step"], reported["basis"],
            reported["step_comparison"]), run_name
