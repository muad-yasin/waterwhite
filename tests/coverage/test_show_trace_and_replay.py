"""``waterwhite show SEGMENT --trace``, the replay object of ``register-run`` and
``waterwhite replay``, through the command line only (plan 3.5, 5.5 Check 2; docs/cli.md;
docs/interfaces.md section 2; docs/join.md sections 4 and 5).

The data is the one of ``test_checkpoint_q2.py``: one shard of four documents under a run of
three steps of two samples, with the three index arrays of its build worked out by hand and
the digests of those arrays. The rows each step read are worked out here by hand as well,
from the layout in docs/join.md section 5 and nothing the tool computes:

    document lengths 6, 9, 5, 8 tokens; a sample is 5 tokens;
    sample number k spans sample_index[k] to sample_index[k + 1], the last token a label:
      0 = D0[0:5]            1 = D0[4:6] D1[0:3]     2 = D1[2:7]
      3 = D1[6:9] D2[0:2]    4 = D2[1:5] D3[0:1]     5 = D3[0:5]
    the shuffle index [3, 2, 1, 0, 5, 4] maps the place g of the run's order to a sample
    number, and a step of batch 2 takes the next two places:
      step 1 = places 0, 1 = samples 3, 2     step 2 = places 2, 3 = samples 1, 0
      step 3 = places 4, 5 = samples 5, 4

A test that passes before the commands exist is a control and says so in its name. Nothing in
this file imports Megatron or torch.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ww_world

from oracle.digests import file_sha256
from waterwhite import cli
from waterwhite import vocabulary as v
from waterwhite.records import MapRecord, SnapshotShard
from waterwhite.store.store import Store

RUN = "Q"
SEGMENT = "Q/seg-1"
CHECKPOINT = "Q@3"
MIXTURE = "mix-q"
SHARD = "shard-1"
STEPS = (1, 2, 3)
BATCH = 2
SEQUENCE_LENGTH = 4
SEED = 1234
DOCUMENTS = [
    [11, 12, 13, 14, 15, 16],
    [21, 22, 23, 24, 25, 26, 27, 28, 29],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45, 46, 47, 48],
]
RELEASE_TEXTS = ["a one", "a two", "b one", "b two"]

CACHE_HASH = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
CACHE_ARRAYS = {
    "document_index": ("<i4", [0, 1, 2, 3]),
    "sample_index": ("<i4", [[0, 0], [0, 4], [1, 2], [1, 6], [2, 1], [3, 0], [3, 4]]),
    "shuffle_index": ("<u4", [3, 2, 1, 0, 5, 4]),
}
CACHE_DIGESTS = {
    "document_index": "64c3e7b97e977bd554cc0d9e6d7ec4677da55b7c904fb04887c25ad89623b75e",
    "sample_index": "19f973d4f146bc93e5cea7012797e482388c04ea73a1682b588d9f8559ada21b",
    "shuffle_index": "8810e591b996e680142547b1190ca53ec508352b4a96e79a1fe4be84536a6668",
}
WRONG_DIGEST = "0" * 64
# The one launch of the run record below, as ``show --trace`` prints it.
LAUNCH = {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0,
          "step_at_start": 0, "first_step": STEPS[0], "last_step": STEPS[-1]}

# What each step read, by hand: (sample position, place in the run's order, [(ordinal, from, to)]).
READ = {
    1: [(0, 0, [(1, 6, 9), (2, 0, 2)]), (1, 1, [(1, 2, 7)])],
    2: [(0, 2, [(0, 4, 6), (1, 0, 3)]), (1, 3, [(0, 0, 5)])],
    3: [(0, 4, [(3, 0, 5)]), (1, 5, [(2, 1, 5), (3, 0, 1)])],
}


def expected_steps(shard: str | None = SHARD) -> list[dict]:
    return [
        {"step": step, "consumed_samples": BATCH, "samples": [
            {"sample_pos": pos, "global_index": place, "rank": 0, "sample_tokens": 5,
             "covered_tokens": 5,
             "rows": [{"shard": shard, "shard_position": 0, "ordinal": ordinal,
                       "token_start": start, "token_end": end} for ordinal, start, end in rows]}
            for pos, place, rows in samples]}
        for step, samples in READ.items()
    ]


def untraced(count: int) -> list[list]:
    return [[SEGMENT, step] for step in STEPS[:count]]


# --- The command line, in process -----------------------------------------------------


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(list(argv), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def command(*argv: str) -> dict:
    """Run one command that must succeed and return its JSON. The exit code is asserted
    (with the standard error as its message) before the output is read."""
    code, out, err = run_cli(*argv)
    assert code == 0, f"exit {code}, standard error: {err!r}"
    return json.loads(out)


def refusal(*argv: str) -> tuple[int, str]:
    """Run one command that must print nothing and fail: (exit code, error name)."""
    code, out, err = run_cli(*argv)
    assert out == "", f"printed {out!r}"
    assert err, f"exit {code} and no error was printed"
    return code, json.loads(err)["error"]


# --- The data -------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path) -> SimpleNamespace:
    folder = tmp_path
    store = Store.create(folder / "store.db", normaliser=ww_world.NORMALISER,
                         verified_joins=[ww_world.JOIN])
    a = ww_world.ingest(store, folder / "src/A.jsonl", ["a one", "a two"], "A", "src/A")
    b = ww_world.ingest(store, folder / "src/B.jsonl", ["b one", "b two"], "B", "src/B")
    release = ww_world.record(
        store, folder / "R.jsonl", RELEASE_TEXTS, "R", v.ROLE_RELEASE,
        ww_world.step("release", "R"), [a, b],
        [ww_world.copy_edge(1, ww_world.ref(a, 1)), ww_world.copy_edge(2, ww_world.ref(a, 2)),
         ww_world.copy_edge(3, ww_world.ref(b, 1)), ww_world.copy_edge(4, ww_world.ref(b, 2))],
    )
    prefix = folder / "runs" / RUN / "shards" / SHARD
    count, digest = ww_world.write_shard(prefix, DOCUMENTS)
    shard = store.register_shard(
        prefix, name=SHARD, step=ww_world.step("tokenize", SHARD), inputs=[release])
    rows = [ww_world.recovered(n, ww_world.ref(release, n + 1)) for n in range(count)]
    store.write_map(
        shard,
        MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED,
                  True, True, tuple(rows)),
        declared_count=count, declared_digest=digest,
    )
    store.register_mixture(
        name=MIXTURE, shards=[SnapshotShard(0, shard, f"runs/{RUN}/shards/{SHARD}", digest)],
        settings={"mix": "q"})
    store.close()
    return SimpleNamespace(folder=folder, path=folder / "store.db", runs=folder / "runs" / RUN)


def run_record(world, *, digests: bool = True) -> dict:
    """The run record of docs/interfaces.md 8.1: the sequential sampler, one launch of three
    steps from counter 0, one shard, the digests of its three index arrays (or none) and no
    shuffle digest."""
    record = ww_world.run_record(RUN, SEGMENT, "fresh", STEPS[0], STEPS[-1])
    record["settings"]["global_batch_size"] = BATCH
    record["settings"]["micro_batch_size"] = 1
    record["settings"]["sequence_length"] = SEQUENCE_LENGTH
    record["settings"]["random_seed"] = SEED
    record["launches"] = [dict(LAUNCH)]
    record["shards"] = [{
        "position": 0, "prefix": f"shards/{SHARD}",
        "bin_sha256": file_sha256(world.runs / "shards" / f"{SHARD}.bin"),
        "idx_sha256": file_sha256(world.runs / "shards" / f"{SHARD}.idx"),
    }]
    record["cache_digests"] = [
        {"kind": kind, "shard": 0, "digest": CACHE_DIGESTS[kind]} for kind in CACHE_ARRAYS
    ] if digests else []
    record["shuffle_digests"] = []
    return record


def write_cache(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for kind, (dtype, values) in CACHE_ARRAYS.items():
        np.save(folder / f"{CACHE_HASH}-GPTDataset-train-{kind}.npy", np.array(values, dtype=dtype))
    (folder / f"{CACHE_HASH}-GPTDataset-train-description.txt").write_text("{}", encoding="utf-8")
    return folder


def write_run_record(world, record: dict) -> Path:
    run_file = world.runs / "q.run.json"
    run_file.write_text(json.dumps(record), encoding="utf-8")
    return run_file


def register(world, record: dict, *, cache: str = "kept", mixture: bool = True) -> dict:
    """Write the run record and the step record and register the segment. ``cache`` says
    whether the run kept its cache folder ("kept"), kept one whose shuffle index is not the
    one of the run ("altered") or has none ("gone"). Returns what ``register-run`` printed."""
    run_file = write_run_record(world, record)
    steps_file = world.runs / "q.steps.json"
    steps_file.write_text(json.dumps({
        "segment": SEGMENT,
        "steps": [{"step": s, "batch_size": BATCH} for s in STEPS],
        "checkpoints": [{"name": CHECKPOINT, "step": STEPS[-1]}],
    }), encoding="utf-8")
    argv = ["register-run", "--store", str(world.path), "--run-record", str(run_file),
            "--steps", str(steps_file)]
    if mixture:
        argv += ["--mixture", MIXTURE]
    if cache != "gone":
        folder = write_cache(world.runs / "q.cache")
        if cache == "altered":
            np.save(folder / f"{CACHE_HASH}-GPTDataset-train-shuffle_index.npy",
                    np.array([0, 1, 2, 3, 4, 5], dtype="<u4"))
        argv += ["--cache", str(folder)]
    return command(*argv)


def show_trace(world, *more: str) -> dict:
    return command("show", SEGMENT, "--trace", "--store", str(world.path), *more)


# --- 1. show SEGMENT --trace ------------------------------------------------------------


def test_show_trace_prints_the_rows_of_every_step_in_reading_order(world):
    register(world, run_record(world))
    assert show_trace(world) == {
        "segment": SEGMENT, "route": "cache files", "cause": None, "cause_detail": None,
        "steps": expected_steps(), "untraced_steps": [], "launches": [LAUNCH],
    }


def test_the_replay_route_prints_the_same_rows(world):
    printed = register(world, run_record(world), cache="gone")
    assert printed["route"] == "replay"
    assert show_trace(world) == {
        "segment": SEGMENT, "route": "replay", "cause": None, "cause_detail": None,
        "steps": expected_steps(), "untraced_steps": [], "launches": [LAUNCH],
    }


@pytest.mark.parametrize("step", STEPS)
def test_step_keeps_that_step_only(world, step):
    register(world, run_record(world))
    shown = show_trace(world, "--step", str(step))
    assert shown["steps"] == [s for s in expected_steps() if s["step"] == step]
    assert (shown["segment"], shown["route"], shown["untraced_steps"]) == (SEGMENT, "cache files", [])


def test_a_segment_with_no_mixture_names_no_shard(world):
    register(world, run_record(world), mixture=False)
    shown = show_trace(world)
    assert shown["steps"] == expected_steps(shard=None)


def test_a_segment_with_route_none_prints_no_steps_and_its_cause(world):
    register(world, run_record(world, digests=False), cache="gone")
    assert show_trace(world) == {
        "segment": SEGMENT, "route": "none", "cause": "data-order trace missing",
        "cause_detail": "no recorded digest", "steps": [], "untraced_steps": untraced(3),
        "launches": [LAUNCH],
    }


def test_a_name_that_is_no_registered_segment_is_refused(world):
    register(world, run_record(world))
    assert show_trace(world)["route"] == "cache files"  # the form works for a segment
    for name in ("Q/seg-9", "no-such-thing", SHARD):  # the shard is registered, but is no segment
        assert refusal("show", name, "--trace", "--store", str(world.path)) == (
            2, "request refused"), name


@pytest.mark.parametrize("more", [["--record", "D0"], ["--map", "capture/position"]])
def test_trace_with_record_or_map_is_a_usage_error(world, more):
    register(world, run_record(world))
    # The form alone works: it is the pairing that is refused (a usage error is also what
    # a build with no such form gives, so the first line is what tells the two apart).
    assert show_trace(world)["route"] == "cache files"
    assert refusal("show", SEGMENT, "--trace", "--store", str(world.path), *more) == (1, "usage")


def test_control_show_without_trace_does_not_print_a_trace(world):
    register(world, run_record(world))
    code, out, _ = run_cli("show", SHARD, "--store", str(world.path))
    assert code == 0 and "steps" not in json.loads(out)


# --- 2. register-run prints the replay -----------------------------------------------------


def test_register_run_says_the_replay_was_not_needed_when_the_cache_files_matched(world):
    printed = register(world, run_record(world), cache="kept")
    assert (printed["route"], printed.get("replay")) == (
        "cache files", {"attempted": False, "accepted": None, "reason": None})


def test_register_run_says_the_replay_was_accepted_when_the_cache_folder_is_gone(world):
    printed = register(world, run_record(world), cache="gone")
    assert (printed["route"], printed.get("replay")) == (
        "replay", {"attempted": True, "accepted": True, "reason": None})


def test_a_cache_folder_whose_files_do_not_match_gives_an_accepted_replay(world):
    printed = register(world, run_record(world), cache="altered")
    assert (printed["route"], printed.get("replay")) == (
        "replay", {"attempted": True, "accepted": True, "reason": None})


def test_register_run_says_the_replay_was_refused_when_a_recorded_digest_differs(world):
    record = run_record(world)
    record["cache_digests"][0]["digest"] = WRONG_DIGEST
    printed = register(world, record, cache="gone")
    assert (printed["route"], printed["cause"], printed["cause_detail"], printed["traced_steps"]) == (
        "none", "data-order trace missing", "digest mismatch", 0)
    assert printed.get("replay") == {"attempted": True, "accepted": False, "reason": "digest mismatch"}


def test_register_run_says_no_replay_was_tried_for_want_of_a_recorded_digest(world):
    printed = register(world, run_record(world, digests=False), cache="gone")
    assert (printed["route"], printed["cause"], printed["cause_detail"]) == (
        "none", "data-order trace missing", "no recorded digest")
    assert printed.get("replay") == {"attempted": False, "accepted": None, "reason": "no recorded digest"}


# --- 3. waterwhite replay ------------------------------------------------------------------


def array_files(out: Path) -> list[str]:
    return sorted(p.name for p in out.glob("*.npy"))


def located(entry: dict, out: Path) -> Path:
    """The file an entry names, wherever the printed path is taken from."""
    path = Path(entry["file"])
    return path if path.is_absolute() else out / path.name


def replay(world, record: dict, out: Path) -> dict:
    return command("replay", "--run-record", str(write_run_record(world, record)), "--out", str(out))


def test_replay_writes_the_arrays_and_prints_each_with_both_digests(world, tmp_path):
    out = tmp_path / "replayed"
    before = file_sha256(world.path)
    printed = replay(world, run_record(world), out)
    assert set(printed) == {"segment", "arrays"} and printed["segment"] == SEGMENT
    by_kind = {e["kind"]: e for e in printed["arrays"]}
    assert len(printed["arrays"]) == 3 and set(by_kind) == set(CACHE_ARRAYS)
    for kind, (dtype, values) in CACHE_ARRAYS.items():
        entry = by_kind[kind]
        assert set(entry) == {"kind", "shard", "file", "digest", "recorded_digest", "equal"}
        assert (entry["shard"], entry["digest"], entry["recorded_digest"], entry["equal"]) == (
            0, CACHE_DIGESTS[kind], CACHE_DIGESTS[kind], True)
        # A per-shard kind is written under its kind and the shard's position.
        assert located(entry, out).name == f"{kind}-0.npy"
        assert np.array_equal(np.load(located(entry, out)), np.array(values, dtype=dtype)), kind
    assert array_files(out) == sorted(f"{kind}-0.npy" for kind in CACHE_ARRAYS)
    # The command writes no store: the one on disk is as it was.
    assert file_sha256(world.path) == before


def test_replay_prints_no_equality_where_no_digest_was_recorded(world, tmp_path):
    out = tmp_path / "replayed"
    printed = replay(world, run_record(world, digests=False), out)
    assert len(printed["arrays"]) == 3
    for entry in printed["arrays"]:
        assert (entry["recorded_digest"], entry["equal"], entry["digest"]) == (
            None, None, CACHE_DIGESTS[entry["kind"]])
    assert len(array_files(out)) == 3


def test_replay_says_which_array_differs_from_its_recorded_digest(world, tmp_path):
    record = run_record(world)
    record["cache_digests"][1]["digest"] = WRONG_DIGEST  # the sample index
    printed = replay(world, record, tmp_path / "replayed")
    by_kind = {e["kind"]: e for e in printed["arrays"]}
    assert by_kind["sample_index"]["equal"] is False
    assert (by_kind["sample_index"]["recorded_digest"], by_kind["sample_index"]["digest"]) == (
        WRONG_DIGEST, CACHE_DIGESTS["sample_index"])
    assert (by_kind["document_index"]["equal"], by_kind["shuffle_index"]["equal"]) == (True, True)


def test_a_record_that_lacks_a_setting_the_replay_needs_is_refused_and_writes_nothing(world, tmp_path):
    record = run_record(world)
    del record["settings"]["random_seed"]
    out = tmp_path / "replayed"
    assert refusal("replay", "--run-record", str(write_run_record(world, record)),
                   "--out", str(out)) == (1, "replay not possible")
    assert not out.exists() or array_files(out) == []
