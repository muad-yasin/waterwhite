"""The plan of a registration, the starting states of Check 2 and the probe for the
run commands (``acceptance_runs``), tested without running a command that takes time.

The expected values are the fixture spec's (the segments, mixtures and checkpoints
it states) and README section 7 of fixture/table (what each starting state changes).
"""

from __future__ import annotations

import acceptance_runs as runs
import pytest

SPEC = runs.spec()


def record(run: str = "E", **settings) -> dict:
    return {"run": run, "settings": {"random_seed": 578, "split": "1,0,0", "train_iters": 6,
                                      "train_samples": None, "micro_batch_size": 4, **settings},
            "cache_digests": [{"kind": "document_index", "shard": 0, "digest": "d"}],
            "shuffle_digests": [{"launch": 0, "epoch": 0, "rank": 0, "digest": "s", "count": 4}]}


# --- Which segments a store registers, and in which order ------------------------------------


def test_the_main_store_registers_the_fixtures_segments_in_the_order_a_checkpoint_needs():
    plan = runs.segments_of(SPEC, runs.RUNS_OF_STORE["main"])
    assert [(s.name, s.mixture, s.parent_checkpoint) for s in plan] == [
        ("E/seg-1", "E/mix-1", None), ("E/seg-2", "E/mix-2", "E@6"),
        ("D/seg-1", "D/mix-1", "E@9"), ("Y/seg-1", "Y/mix-1", "E@6"),
        ("K/seg-1", "K/mix-1", "E@6")]


def test_the_round_robin_store_registers_its_two_segments():
    plan = runs.segments_of(SPEC, runs.RUNS_OF_STORE["rr"])
    assert [(s.name, s.mixture, s.parent_checkpoint) for s in plan] == [
        ("E-rr/seg-1", "E-rr/mix-1", None), ("E-rr/seg-2", "E-rr/mix-2", "E-rr@6")]


def test_a_segment_whose_parent_checkpoint_is_not_registered_before_it_is_refused():
    with pytest.raises(runs.RunWorldError, match="E@9"):
        runs.segments_of(SPEC, ("D",))
    with pytest.raises(runs.RunWorldError, match="E@9"):
        runs.segments_of(SPEC, ("D", "E"))


def test_the_files_of_a_segment_are_a_folder_per_run_and_a_stem_per_segment(tmp_path):
    (second,) = [s for s in runs.segments_of(SPEC, ("E",)) if s.name == "E/seg-2"]
    files = runs.files_of(tmp_path, second)
    assert files.run_record == tmp_path / "E" / "E-seg-2.run.json"
    assert files.steps == tmp_path / "E" / "E-seg-2.steps.json"
    assert files.cache == tmp_path / "E" / "E-seg-2.cache"


def test_register_arguments_hand_over_the_cache_only_when_asked_and_the_parent_when_there_is_one(
        tmp_path):
    first, second = runs.segments_of(SPEC, ("E",))
    one = runs.register_arguments(tmp_path / "s.sqlite", first, runs.files_of(tmp_path, first),
                                  cache=True)
    assert one[0] == "register-run" and "--cache" in one and "--parent-checkpoint" not in one
    assert one[one.index("--mixture") + 1] == "E/mix-1"
    two = runs.register_arguments(tmp_path / "s.sqlite", second, runs.files_of(tmp_path, second),
                                  cache=False, run_record=tmp_path / "changed.json")
    assert "--cache" not in two and two[two.index("--parent-checkpoint") + 1] == "E@6"
    assert two[two.index("--run-record") + 1] == str(tmp_path / "changed.json")


def test_the_lowered_sample_of_run_e_is_the_last_sample_of_step_8_and_the_round_robin_run_has_none():
    assert runs.lowered_samples(SPEC, "E") == {("E/seg-2", 8, 3): 16}
    assert runs.lowered_samples(SPEC, "E-rr") == {}
    assert runs.lowered_samples(SPEC, "K") == {}


# --- The starting states of Check 2 (fixture/table/README.md section 7) ----------------------


def test_c2_i_withholds_the_cache_files_of_e_d_and_y_and_changes_nothing_else():
    for run in ("E", "D", "Y"):
        changed, cache = runs.record_variant("c2.i", run, record(run))
        assert cache is False and changed == record(run)
    changed, cache = runs.record_variant("c2.i", "K", record("K"))
    assert cache is True and changed == record("K")


def test_c2_ii_withholds_the_cache_and_adds_one_to_the_seed():
    original = record()
    changed, cache = runs.record_variant("c2.ii", "E", original)
    assert cache is False and changed["settings"]["random_seed"] == 579
    assert changed["cache_digests"] == original["cache_digests"]
    assert original["settings"]["random_seed"] == 578  # the record given is not changed
    assert runs.record_variant("c2.ii", "K", record("K")) == (record("K"), True)
    with pytest.raises(runs.RunWorldError, match="no seed"):
        runs.record_variant("c2.ii", "E", {"settings": {"random_seed": None}})


def test_c2_iii_withholds_the_cache_the_replay_settings_and_the_recorded_digests():
    changed, cache = runs.record_variant("c2.iii", "Y", record("Y"))
    assert cache is False and changed["cache_digests"] == []
    assert all(changed["settings"][key] is None for key in runs.REPLAY_SETTINGS)
    assert changed["settings"]["micro_batch_size"] == 4
    assert changed["shuffle_digests"] == record("Y")["shuffle_digests"]


def test_c2_iv_removes_the_shuffle_digests_of_run_k_and_keeps_its_cache():
    changed, cache = runs.record_variant("c2.iv", "K", record("K"))
    assert cache is True and changed["shuffle_digests"] == []
    assert changed["cache_digests"] == record("K")["cache_digests"]
    assert runs.record_variant("c2.iv", "E", record()) == (record(), True)
    with pytest.raises(runs.RunWorldError, match="no shuffle digests"):
        runs.record_variant("c2.iv", "K", {"shuffle_digests": []})


def test_an_unmutated_state_changes_nothing():
    assert runs.record_variant("unmutated", "E", record()) == (record(), True)


# --- Whether the run commands exist, found by asking ------------------------------------------


class Done:
    def __init__(self, returncode: int, stderr: str = ""):
        self.returncode, self.stderr = returncode, stderr


def refused(word: str) -> Done:
    return Done(2, f"argument command: invalid choice: '{word}' (choose from 'init', 'query')")


@pytest.mark.parametrize(("replica", "tool", "expected"), [
    (Done(0), Done(0), None),
    (refused("build-runs"), Done(0), "the replica has no command build-runs"),
    (Done(0), refused("register-run"), "the tool has no command register-run"),
    (refused("build-runs"), refused("register-run"),
     "the replica has no command build-runs; the tool has no command register-run"),
])
def test_a_run_command_is_missing_where_the_parser_does_not_know_its_word(
        monkeypatch, replica, tool, expected):
    answers = iter([replica, tool])
    monkeypatch.setattr(runs, "_ask", lambda command, env: next(answers))
    runs.capture_missing.cache_clear()
    try:
        assert runs.capture_missing() == expected
    finally:
        runs.capture_missing.cache_clear()


@pytest.mark.parametrize("broken", [
    Done(1, "ModuleNotFoundError: No module named 'torch'"),
    Done(2, "argument command: invalid choice: 'something-else'"),
])
def test_a_command_that_fails_for_another_reason_is_an_error_and_not_a_missing_command(
        monkeypatch, broken):
    monkeypatch.setattr(runs, "_ask", lambda command, env: broken)
    runs.capture_missing.cache_clear()
    try:
        with pytest.raises(runs.RunWorldError, match="build-runs"):
            runs.capture_missing()
    finally:
        runs.capture_missing.cache_clear()


def test_the_run_command_is_asked_for_the_runs_of_the_store_in_a_folder_of_its_own(tmp_path):
    arguments = runs.build_runs_arguments(tmp_path / "spec.toml", tmp_path / "build",
                                          tmp_path / "runs", ("E", "D", "Y", "K"))
    assert arguments[0] == "build-runs"
    assert arguments[arguments.index("--out") + 1] == str(tmp_path / "runs")
    assert arguments[arguments.index("--fixture-build") + 1] == str(tmp_path / "build")
    assert arguments[arguments.index("--runs") + 1] == "E,D,Y,K"


# --- The joins a store lists before its runs are registered -------------------------------------


def listed(path) -> list[tuple]:
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(path)) as db:
        return db.execute("SELECT release_tag, sampler_type, sampler_class FROM verified_join "
                          "ORDER BY sampler_type").fetchall()


def test_the_joins_a_store_lists_are_read_and_never_written(tmp_path):
    import mini_store

    none = mini_store.build_scratch(tmp_path / "none").store
    assert runs.verified_joins(none) == [] == listed(none)
    one = mini_store.build(tmp_path / "one").store
    assert runs.verified_joins(one) == listed(one) and len(listed(one)) == 1


def test_the_oracle_is_asked_for_the_runs_of_the_store_whose_build_it_reads():
    main = runs.oracle_arguments("build", "out", runs.RUNS_OF_STORE["main"])
    assert main[main.index("--runs") + 1] == "E,D,Y,K"
    assert runs.oracle_arguments("build", "out", runs.RUNS_OF_STORE["rr"])[-1] == "E-rr"
    assert main[main.index("--fixture") + 1] == "build" and main[main.index("--out") + 1] == "out"
