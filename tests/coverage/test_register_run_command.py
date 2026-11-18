"""``waterwhite register-run``, and the checkpoint answers it makes possible (plan
2.4, 3.5; docs/cli.md "register-run"; docs/coverage.md section 8).

The runs are the hand worlds of ``tests/join/join_kit.py``; the store is the hand
store of ``ww_world``. Everything goes through ``waterwhite.cli.main``, the way a
person would call it: the run record, the step record and the cache folder are
files, the store is a file, and the answers are the command line's. Nothing here
imports Megatron; a run read with the cyclic sampler is only registered with a
cause, so none needs torch.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "join"))
import join_kit as kit  # noqa: E402

from waterwhite import cli  # noqa: E402
from waterwhite import vocabulary as v  # noqa: E402

# One shard of three documents of 5 tokens, a sample of 5 tokens (sequence length 4):
# 15 tokens, (15 - 1) div 4 = 3 samples starting at tokens 0, 4 and 8.
#   S0 = D0[0:5]; S1 = D0[4:5] and D1[0:4]; S2 = D1[3:5] and D2[0:3]
# With the shuffle index [0, 1, 2], step 1 reads S0, step 2 reads S1, step 3 reads S2.
LENGTHS = ([5, 5, 5],)
ARRAYS = {
    ("document_index", 0): np.array([0, 1, 2], dtype=np.int32),
    ("sample_index", 0): np.array([[0, 0], [0, 4], [1, 3], [2, 2]], dtype=np.int32),
    ("shuffle_index", 0): np.array([0, 1, 2], dtype=np.uint32),
}
# shard-1 of the hand store holds R2's three records: D0 is B's "passage X", D1 is B's
# "b two" and D2 is Z's "z three". Source Z is therefore read at step 3 and no sooner.
CHECKPOINTS = [{"name": "H@1", "step": 1}, {"name": "H@2", "step": 2}, {"name": "H@3", "step": 3}]


def call(*argv: str) -> tuple[int, dict | None, dict | None]:
    """Run one command: (exit code, what it printed, what it printed as an error)."""
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(list(argv), stdout=out, stderr=err)
    return (code, json.loads(out.getvalue()) if out.getvalue() else None,
            json.loads(err.getvalue()) if err.getvalue() else None)


def write_run(folder: Path, *, record: dict | None = None, steps: dict | None = None,
              cache: bool = True) -> dict:
    """The files a training wrapper leaves for one segment, as paths."""
    record = kit.run_record(folder, shards=1, arrays=ARRAYS, lengths=LENGTHS, micro=1, batch=1,
                            launches=[kit.launch(0, 1, 0, 1, 3)]) if record is None else record
    if cache:
        kit.write_cache(folder / "cache", ARRAYS)
    return {
        "--run-record": str(kit.write_json(folder / "run_record.json", record)),
        "--steps": str(kit.write_json(folder / "step_record.json",
                                      steps or {"segment": "H/seg-1", "checkpoints": CHECKPOINTS})),
        **({"--cache": str(folder / "cache")} if cache else {}),
    }


def register(world, files: dict, *more: str):
    flags = [part for pair in files.items() for part in pair]
    return call("register-run", "--store", str(world.path), *flags, *more)


def withdraw(world, source: str, artifact: str) -> dict:
    code, answer, error = call("query", "withdraw", source, "--store", str(world.path),
                               "--artifact", artifact)
    assert (code, error) == (0, None)
    return answer


def test_register_run_prints_what_it_registered(world, tmp_path):
    code, out, error = register(world, write_run(tmp_path / "run"), "--mixture", "mix-1")
    assert (code, error) == (0, None)
    assert out == {
        "segment": "H/seg-1", "route": "cache files", "cause": None, "cause_detail": None,
        "steps": 3, "traced_steps": 3, "untraced_steps": [], "checkpoints": ["H@1", "H@2", "H@3"],
        "replay": {"attempted": False, "accepted": None, "reason": None},  # the cache files matched
        "shuffle": [],  # a sequential run computes no shuffle
    }
    held = world.store.segment("H/seg-1")
    assert (held.run, held.position, held.mixture_instance_id) == ("H", 0, world["mix-1"])
    assert [c.step for c in world.store.checkpoints_of("H/seg-1")] == [1, 2, 3]


def test_a_checkpoint_whose_history_read_the_source_contains_it_with_its_basis(world, tmp_path):
    register(world, write_run(tmp_path / "run"), "--mixture", "mix-1")
    answer = withdraw(world, "Z", "H@3")
    assert answer["answer"] == "contains"
    assert answer["basis"] == v.BASIS_WHOLE_SAMPLE
    assert answer["first_exposure"] == {"segment": "H/seg-1", "step": 3}
    assert answer["untraced_steps"] == []
    assert answer["chain"][0]["hop"] == "H/seg-1:step 3"


def test_a_checkpoint_before_the_read_does_not_contain_it_and_says_what_it_covers(world, tmp_path):
    register(world, write_run(tmp_path / "run"), "--mixture", "mix-1")
    answer = withdraw(world, "Z", "H@2")
    assert answer["answer"] == "does not contain"
    assert answer["scope"]["runs"] == ["H"]
    assert answer["scope"]["steps"] == [["H/seg-1", 1, 2]]


def test_the_operators_note_lowers_a_step_and_the_command_lists_it_as_untraced(world, tmp_path):
    # The operator says step 3 read 2 of the 5 tokens of its last sample: the read of
    # D2 lies in the part the note leaves uncovered, so the step is untraced and the
    # checkpoint may contain the source; it never counts as absent.
    steps = {"segment": "H/seg-1", "steps": [{"step": 3, "last_sample_tokens_read": 2}],
             "checkpoints": CHECKPOINTS}
    code, out, _ = register(world, write_run(tmp_path / "run", steps=steps), "--mixture", "mix-1")
    assert code == 0 and out["traced_steps"] == 3 and out["untraced_steps"] == [["H/seg-1", 3]]
    answer = withdraw(world, "Z", "H@3")
    assert answer["answer"] == "may contain" and answer["names"] == v.NAMES_STEP_UNTRACED
    assert answer["untraced_steps"] == [["H/seg-1", 3]]


# --- The route follows from the run record, and from nothing else (plan 3.5) --------------------


def test_the_cache_files_carry_the_route_only_through_their_recorded_digests(world, tmp_path):
    # Cache files whose arrays are not the ones whose digests the run record holds
    # give no route, whatever their names say; and with no cache folder the
    # replay is held against the same digests.
    files = write_run(tmp_path / "run")
    changed = ARRAYS | {("shuffle_index", 0): np.array([2, 1, 0], dtype=np.uint32)}
    kit.write_cache(tmp_path / "other", changed)
    files["--cache"] = str(tmp_path / "other")
    code, out, _ = register(world, files, "--mixture", "mix-1")
    assert code == 0
    assert (out["route"], out["cause"], out["cause_detail"]) == (
        "none", v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)
    assert out["untraced_steps"] == [["H/seg-1", 1], ["H/seg-1", 2], ["H/seg-1", 3]]
    assert out["traced_steps"] == 0
    # The replay was held against the recorded digests and refused for the difference.
    assert out["replay"] == {"attempted": True, "accepted": False, "reason": v.TRACE_MISSING_DIGEST_MISMATCH}
    answer = withdraw(world, "Z", "H@2")
    assert (answer["answer"], answer["names"], answer["names_detail"]) == (
        "may contain", v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)


def test_a_run_record_with_no_digest_gets_no_route_even_with_cache_files(world, tmp_path):
    record = kit.run_record(tmp_path / "run", shards=1, arrays=ARRAYS, lengths=LENGTHS, micro=1,
                            batch=1, launches=[kit.launch(0, 1, 0, 1, 3)])
    record["cache_digests"] = []
    code, out, _ = register(world, write_run(tmp_path / "run", record=record), "--mixture", "mix-1")
    assert code == 0
    assert (out["route"], out["cause"], out["cause_detail"]) == (
        "none", v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    # With no recorded digest no replay is tried, and the reason says so.
    assert out["replay"] == {"attempted": False, "accepted": None, "reason": v.TRACE_MISSING_NO_DIGEST}


def test_the_replay_route_when_no_cache_folder_is_given(world, tmp_path):
    # The golden segment of Megatron's own index build (``golden_megatron``) is the
    # one world a replay can rebuild from its settings: the hand arrays above are
    # not what any seed gives.
    record = kit.golden_record(tmp_path / "run")
    record["run"], record["segment"] = "G", "G/seg-1"
    files = write_run(tmp_path / "run", record=record, steps={"segment": "G/seg-1", "checkpoints": []},
                      cache=False)
    code, out, _ = register(world, files)
    assert code == 0
    assert (out["route"], out["cause"]) == ("replay", None) and out["traced_steps"] == 6
    assert out["replay"] == {"attempted": True, "accepted": True, "reason": None}


# --- A run read with the cyclic sampler and no shuffle digest (plan 2.4, 3.5) ------------------


def cyclic_files(folder: Path, *, shuffle: bool) -> dict:
    """The cyclic run of the hand world: 4 steps of batch 2 on 2 ranks, two epochs.
    Its orders are stand-ins, since only the digests are recorded and no test here
    reaches the comparison with a computed order."""
    record = kit.cyclic_record(folder, [kit.launch(0, 2, 0, 1, 4)], kit.stand_in_orders({0: [0, 1]}))
    record["run"], record["segment"] = "K", "K/seg-1"
    if not shuffle:
        record["shuffle_digests"] = []
    kit.write_cache(folder / "cache", kit.ARRAYS)
    return {
        "--run-record": str(kit.write_json(folder / "run_record.json", record)),
        "--steps": str(kit.write_json(folder / "step_record.json", {
            "segment": "K/seg-1", "checkpoints": [{"name": "K@2", "step": 2}]})),
        "--cache": str(folder / "cache"),
    }


@pytest.fixture
def cyclic_join_verified(world):
    """The store lists the cyclic sampler as verified, so a cyclic run is held up by
    nothing but its own record."""
    world.sql(("INSERT INTO verified_join (release_tag, sampler_type, sampler_class) VALUES (?, ?, ?)",
               (kit.TAG, "cyclic", kit.CYCLIC_CLASS)))


def test_a_cyclic_run_with_no_shuffle_digest_is_registered_without_torch(
        world, cyclic_join_verified, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # torch cannot be imported
    code, out, error = register(world, cyclic_files(tmp_path / "run", shuffle=False), "--mixture", "mix-2")
    assert (code, error) == (0, None)
    assert (out["route"], out["cause"], out["cause_detail"]) == ("none", v.NAMES_CYCLIC_NOT_RECORDED, None)
    assert out["traced_steps"] == 0
    assert out["replay"] == {"attempted": False, "accepted": None, "reason": None}  # decided before any digest
    assert out["untraced_steps"] == [["K/seg-1", step] for step in (1, 2, 3, 4)]


def test_that_checkpoint_may_contain_and_never_does_not_contain(
        world, cyclic_join_verified, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    register(world, cyclic_files(tmp_path / "run", shuffle=False), "--mixture", "mix-2")
    for source in ("A", "C", "Z"):
        answer = withdraw(world, source, "K@2")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_CYCLIC_NOT_RECORDED), source
        assert answer["untraced_steps"] == [["K/seg-1", 1], ["K/seg-1", 2]]  # up to K@2


def test_a_cyclic_run_whose_shuffle_must_be_computed_is_refused_when_torch_is_missing(
        world, cyclic_join_verified, tmp_path, monkeypatch):
    # Every digest is recorded, so the shuffle must be computed, and torch cannot be
    # imported: the registration is refused with a named error and nothing is written.
    monkeypatch.setitem(sys.modules, "torch", None)
    files = cyclic_files(tmp_path / "run", shuffle=True)
    before = world.store.segments()
    code, out, error = register(world, files, "--mixture", "mix-2")
    assert (code, out) == (4, None)
    assert error["error"] == "torch not available"
    assert world.store.segments() == before and world.store.segment("K/seg-1") is None
    assert world.store.checkpoint("K@2") is None


# --- What the command takes, and what it refuses ----------------------------------------------


def test_a_resume_names_its_parent_checkpoint_and_takes_the_next_position(world, tmp_path):
    register(world, write_run(tmp_path / "run1"), "--mixture", "mix-1")
    record = kit.run_record(tmp_path / "run2", shards=1, arrays=ARRAYS, lengths=LENGTHS, micro=1,
                            batch=1, launches=[kit.launch(0, 1, 3, 4, 4)])
    record["segment"], record["start_kind"] = "H/seg-2", "resume"
    steps = {"segment": "H/seg-2", "checkpoints": [{"name": "H@4", "step": 4}]}
    files = write_run(tmp_path / "run2", record=record, steps=steps)
    code, out, error = register(world, files, "--mixture", "mix-1", "--parent-checkpoint", "H@3")
    assert (code, error) == (0, None)
    held = world.store.segment("H/seg-2")
    assert (held.position, held.parent_checkpoint) == (1, world.store.checkpoint("H@3").instance_id)
    # Source Z was read at step 3, in the history of H@4.
    assert withdraw(world, "Z", "H@4")["answer"] == "contains"
    assert withdraw(world, "Z", "H@4")["first_exposure"] == {"segment": "H/seg-1", "step": 3}


def test_the_same_registration_again_changes_nothing(world, tmp_path):
    files = write_run(tmp_path / "run")
    first = register(world, files, "--mixture", "mix-1")
    before = world.store._conn.execute("SELECT COUNT(*) FROM trace_row").fetchone()[0]
    assert register(world, files, "--mixture", "mix-1") == first
    assert world.store._conn.execute("SELECT COUNT(*) FROM trace_row").fetchone()[0] == before


@pytest.mark.parametrize("more, name", [
    (["--mixture", "no-such-mixture"], "invalid record"),
    (["--mixture", "src/A"], "invalid record"),  # a name of the store that is no mixture
    (["--mixture", "mix-1", "--parent-checkpoint", "E@3"], "invalid record"),  # a fresh start
])
def test_a_registration_the_store_cannot_tie_together_is_refused_and_writes_nothing(
        world, tmp_path, more, name):
    code, out, error = register(world, write_run(tmp_path / "run"), *more)
    assert (code, out) == (4, None) and error["error"] == name
    assert world.store.segment("H/seg-1") is None


def test_a_run_record_that_is_no_run_record_exits_with_4_and_the_name_invalid_record(world, tmp_path):
    # A malformed record is named "invalid record" and exits with 4, as every
    # refusal of that name does.
    files = write_run(tmp_path / "run")
    Path(files["--run-record"]).write_text("[1, 2]", encoding="utf-8")
    code, out, error = register(world, files)
    assert (code, out) == (4, None) and error["error"] == "invalid record"
    Path(files["--run-record"]).write_text("{not json", encoding="utf-8")
    assert register(world, files)[0] == 4
    record = json.loads(Path(write_run(tmp_path / "run2")["--run-record"]).read_text(encoding="utf-8"))
    record["settings"]["micro_batch_size"] = "two"  # a field of the wrong type
    Path(files["--run-record"]).write_text(json.dumps(record), encoding="utf-8")
    code, _, error = register(world, files)
    assert (code, error["error"]) == (4, "invalid record")
    Path(files["--run-record"]).write_text(json.dumps({**record, "run": None}), encoding="utf-8")
    assert register(world, files)[0] == 4
    assert world.store.segment("H/seg-1") is None


def test_a_step_record_that_is_malformed_exits_with_4(world, tmp_path):
    files = write_run(tmp_path / "run")
    Path(files["--steps"]).write_text(json.dumps({"segment": "H/seg-9", "steps": []}), encoding="utf-8")
    code, _, error = register(world, files)
    assert (code, error["error"]) == (4, "invalid record")
    assert world.store.segment("H/seg-1") is None


def test_a_run_record_file_that_cannot_be_read_is_a_usage_error(world, tmp_path):
    files = write_run(tmp_path / "run")
    files["--run-record"] = str(tmp_path / "missing.json")
    code, out, error = register(world, files)
    assert (code, out) == (1, None) and error["error"] == "usage"


def test_a_checkpoint_the_segment_does_not_reach_is_refused_with_the_whole_segment(world, tmp_path):
    steps = {"segment": "H/seg-1", "checkpoints": [{"name": "H@9", "step": 9}]}
    code, out, error = register(world, write_run(tmp_path / "run", steps=steps), "--mixture", "mix-1")
    assert (code, out) == (4, None) and error["error"] == "invalid record"
    assert world.store.segment("H/seg-1") is None
