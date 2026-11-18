"""What ``waterwhite show SEGMENT --trace`` and ``waterwhite register-run`` print about the
launches and the shuffle of a run, through the command line only (plan 3.5; docs/cli.md;
docs/interfaces.md sections 2 and 8; docs/join.md sections 4 to 6).

1. ``show --trace`` prints ``launches``: the run record's launches as registered, in position
   order, each with its ``position``, ``world_size``, ``data_parallel_size``,
   ``consumed_at_start``, ``step_at_start``, ``first_step`` and ``last_step``, and ``[]`` when
   the record holds none. For a segment read with the cyclic sampler each step also prints
   ``epochs``, the epochs its samples are drawn from, in order; a sequential segment prints no
   ``epochs``.
2. ``register-run`` prints ``shuffle``: ``[]`` for a sequential segment, and for a cyclic one an
   entry for each (launch, epoch, rank) the launches read, in that order, with the digest and
   the count of the order the join computed, the recorded digest and count, and ``equal``.

The data is the one shard of four documents of ``test_checkpoint_q2.py``, whose dataset holds
6 samples; its three index arrays and their digests are copied from there. The cyclic runs
below read it on 2 data-parallel ranks with a micro batch of 1, and the epochs and the shuffle
are worked out here by hand from the sizes, by the rule of plan 3.5 and docs/join.md section 6:

    a round is micro * ranks = 2 samples; an epoch is the 6 samples all ranks read, which is
    3 rounds; a rank owns a bucket of (6 div 2) * 1 = 3 indices, rank r from index 3 * r;
    a sample read at counter c belongs to epoch c div 6, and a round that starts at counter c
    puts its two samples in that epoch, so a step of batch b from counter c reads rounds that
    start at c, c + 2, and so on up to c + b - 2.

    straddling run: batch 4, three steps, launch 0 = step 1 from counter 0 and launch 1 =
    steps 2 and 3 from counter 4 (a resume inside the epoch):
      step 1 = counter 0 to 3, rounds at 0 and 2: epoch 0 only;
      step 2 = counter 4 to 7, rounds at 4, 6: 4 div 6 = 0 and 6 div 6 = 1, so epochs 0 and 1;
      step 3 = counter 8 to 11, rounds at 8, 10: epoch 1 only.
      The orders read are launch 0 epoch 0, launch 1 epoch 0 and launch 1 epoch 1, each for
      ranks 0 and 1: six entries.
    exact run: batch 2, four steps from counter 0 on one launch: rounds at 0 | 2 | 4 | 6 and
      so epochs [0], [0], [0], [1] (the epoch ends exactly at the end of step 3). The orders
      read are epoch 0 and epoch 1, each for ranks 0 and 1: four entries.

The order of one rank in one epoch is drawn here by the test's own call to torch as the sampler
draws it (a generator seeded with the epoch, ``torch.randperm`` over the bucket, the rank's
first index added) and its digest is taken with the test-time oracle's definition, never with
a function of ``src/waterwhite``. The tests that record such digests import torch when they
run; run this file with the interpreter of the Megatron environment.

A test that passes before the commands print these keys is a control and says so in its name.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import ww_world

from oracle.digests import file_sha256, order_digest
from waterwhite import cli
from waterwhite import vocabulary as v
from waterwhite.records import MapRecord, SnapshotShard
from waterwhite.store.store import Store

RUN = "Q"
SEGMENT = "Q/seg-1"
MIXTURE = "mix-q"
SHARD = "shard-1"
SEQUENCE_LENGTH = 4
SEED = 1234
DOCUMENTS = [
    [11, 12, 13, 14, 15, 16],
    [21, 22, 23, 24, 25, 26, 27, 28, 29],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45, 46, 47, 48],
]
RELEASE_TEXTS = ["a one", "a two", "b one", "b two"]
NOT_NEEDED = {"attempted": False, "accepted": None, "reason": None}

CYCLIC_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler"
CYCLIC_JOIN = (ww_world.JOIN[0], v.SAMPLER_CYCLIC, CYCLIC_CLASS)

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

RANKS = 2
BUCKET = 3  # (6 div (1 * 2)) * 1, worked out in the docstring


def launch(position: int, consumed: int, step_at_start: int, first: int, last: int, *,
           ranks: int = RANKS, world_size: int | None = None) -> dict:
    return {"position": position, "world_size": ranks if world_size is None else world_size,
            "data_parallel_size": ranks, "consumed_at_start": consumed,
            "step_at_start": step_at_start, "first_step": first, "last_step": last}


# The two cyclic runs of the docstring: the batch, the steps, the launches, the epochs each
# step reads and the (launch, epoch, rank) of every order the launches read, all by hand.
STRADDLE = SimpleNamespace(
    batch=4, steps=(1, 2, 3),
    launches=[launch(0, 0, 0, 1, 1), launch(1, 4, 1, 2, 3)],
    epochs={1: [0], 2: [0, 1], 3: [1]},
    keys=[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
)
EXACT = SimpleNamespace(
    batch=2, steps=(1, 2, 3, 4),
    launches=[launch(0, 0, 0, 1, 4)],
    epochs={1: [0], 2: [0], 3: [0], 4: [1]},
    keys=[(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)],
)


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


# --- The data -------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path) -> SimpleNamespace:
    """One release of four lines, one shard of four documents made from it with its whole
    map, one mixture of that shard. The store trusts the sequential and the cyclic sampler of
    the release."""
    folder = tmp_path
    store = Store.create(folder / "store.db", normaliser=ww_world.NORMALISER,
                         verified_joins=[ww_world.JOIN, CYCLIC_JOIN])
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


def run_record(world, *, steps: tuple, batch: int, launches: list[dict],
               cyclic: bool, shuffle: list[dict] | None = None) -> dict:
    """The run record of docs/interfaces.md 8.1: one shard, the digests of its three index
    arrays, the given launches (in the order given) and the given shuffle digests."""
    record = ww_world.run_record(
        RUN, SEGMENT, "fresh", steps[0], steps[-1],
        sampler=v.SAMPLER_CYCLIC if cyclic else "single",
        sampler_class=CYCLIC_CLASS if cyclic else ww_world.SAMPLER_CLASS)
    record["settings"]["global_batch_size"] = batch
    record["settings"]["micro_batch_size"] = 1
    record["settings"]["sequence_length"] = SEQUENCE_LENGTH
    record["settings"]["random_seed"] = SEED
    record["launches"] = launches
    record["shards"] = [{
        "position": 0, "prefix": f"shards/{SHARD}",
        "bin_sha256": file_sha256(world.runs / "shards" / f"{SHARD}.bin"),
        "idx_sha256": file_sha256(world.runs / "shards" / f"{SHARD}.idx"),
    }]
    record["cache_digests"] = [
        {"kind": kind, "shard": 0, "digest": CACHE_DIGESTS[kind]} for kind in CACHE_ARRAYS]
    record["shuffle_digests"] = shuffle or []
    return record


def drawn_order(epoch: int, rank: int) -> list[int]:
    """The whole order of one rank in one epoch, drawn as the cyclic sampler draws it."""
    import torch  # run with the interpreter of the Megatron environment

    generator = torch.Generator()
    generator.manual_seed(epoch)
    return [rank * BUCKET + place for place in torch.randperm(BUCKET, generator=generator).tolist()]


def recorded_rows(keys: list[tuple[int, int, int]]) -> list[dict]:
    """What a wrapper records: for each (launch, epoch, rank) the digest and the count of the
    rank's whole order of the epoch."""
    return [{"launch": launch_, "epoch": epoch, "rank": rank,
             "digest": order_digest(drawn_order(epoch, rank)), "count": BUCKET}
            for launch_, epoch, rank in keys]


def expected_shuffle(rows: list[dict]) -> list[dict]:
    """The entries ``register-run`` must print for a run record whose recorded rows are
    ``rows``: the order the join computes is the drawn one, so the digest and count printed
    are the drawn ones, and ``equal`` says whether the record agrees on both."""
    entries = []
    for row in rows:
        digest, count = order_digest(drawn_order(row["epoch"], row["rank"])), BUCKET
        entries.append({
            "launch": row["launch"], "epoch": row["epoch"], "rank": row["rank"],
            "digest": digest, "count": count,
            "recorded_digest": row["digest"], "recorded_count": row["count"],
            "equal": row["digest"] == digest and row["count"] == count})
    return entries


def write_cache(folder: Path) -> Path:
    folder.mkdir(parents=True)
    for kind, (dtype, values) in CACHE_ARRAYS.items():
        np.save(folder / f"{CACHE_HASH}-GPTDataset-train-{kind}.npy", np.array(values, dtype=dtype))
    (folder / f"{CACHE_HASH}-GPTDataset-train-description.txt").write_text("{}", encoding="utf-8")
    return folder


def register(world, record: dict, steps: tuple, batch: int) -> dict:
    """Write the run record and the step record, keep the cache folder and register the
    segment with its mixture. Returns what ``register-run`` printed."""
    run_file = world.runs / "q.run.json"
    run_file.write_text(json.dumps(record), encoding="utf-8")
    steps_file = world.runs / "q.steps.json"
    steps_file.write_text(json.dumps({
        "segment": SEGMENT,
        "steps": [{"step": s, "batch_size": batch} for s in steps],
        "checkpoints": [{"name": f"Q@{steps[-1]}", "step": steps[-1]}],
    }), encoding="utf-8")
    return command("register-run", "--store", str(world.path), "--run-record", str(run_file),
                   "--steps", str(steps_file), "--mixture", MIXTURE,
                   "--cache", str(write_cache(world.runs / "q.cache")))


def show_trace(world, *more: str) -> dict:
    return command("show", SEGMENT, "--trace", "--store", str(world.path), *more)


def register_cyclic(world, case, *, shuffle: list[dict] | None = None) -> dict:
    rows = recorded_rows(case.keys) if shuffle is None else shuffle
    record = run_record(world, steps=case.steps, batch=case.batch, launches=case.launches,
                        cyclic=True, shuffle=rows)
    return register(world, record, case.steps, case.batch)


# --- 1. show --trace prints the launches ---------------------------------------------------


def test_a_resume_inside_the_segment_prints_both_launches_in_position_order(world):
    first = launch(0, 0, 0, 1, 2)
    resumed = launch(1, 4, 2, 3, 3, ranks=1, world_size=4)  # tensor parallel: 4 ranks, 1 replica
    # The record lists the later launch first: it is the position that orders the print.
    record = run_record(world, steps=(1, 2, 3), batch=2, launches=[resumed, first], cyclic=False)
    printed = register(world, record, (1, 2, 3), 2)
    assert printed.get("shuffle") == []  # a sequential segment prints no shuffle entries
    shown = show_trace(world)
    assert shown.get("launches") == [first, resumed]
    # The launches are the record's and never derived: each is printed with exactly its seven
    # fields, as given.
    assert [sorted(item) for item in shown["launches"]] == [sorted(first)] * 2
    assert "epochs" not in shown["steps"][0]  # a sequential segment prints no epochs


def test_a_record_with_one_launch_prints_one(world):
    only = launch(0, 0, 0, 1, 3, ranks=1)
    record = run_record(world, steps=(1, 2, 3), batch=2, launches=[only], cyclic=False)
    register(world, record, (1, 2, 3), 2)
    assert show_trace(world).get("launches") == [only]


def test_a_record_with_no_launch_prints_an_empty_list(world):
    record = run_record(world, steps=(1, 2, 3), batch=2, launches=[], cyclic=False)
    register(world, record, (1, 2, 3), 2)
    assert show_trace(world).get("launches") == []


def test_step_filters_the_steps_and_leaves_the_launches_of_the_record(world):
    # docs/cli.md says that --step filters the steps as before; the launches are the record's
    # whatever the step. The test asserts the steps and that both launches are still printed.
    first, resumed = launch(0, 0, 0, 1, 2), launch(1, 4, 2, 3, 3)
    record = run_record(world, steps=(1, 2, 3), batch=2, launches=[first, resumed], cyclic=False)
    register(world, record, (1, 2, 3), 2)
    shown = show_trace(world, "--step", "3")
    assert [s["step"] for s in shown["steps"]] == [3]
    assert shown.get("launches") == [first, resumed]


def test_the_launches_of_a_cyclic_segment_are_printed_as_recorded(world):
    register_cyclic(world, STRADDLE)
    assert show_trace(world).get("launches") == STRADDLE.launches


# --- 1. show --trace prints the epochs of a cyclic segment ---------------------------------


def test_control_the_straddling_cyclic_run_registers_and_shows_its_steps(world):
    # What holds before the new keys exist: the registration traces three steps by the cache
    # files, so the epochs below are asked of a traced segment and not of a refused one.
    printed = register_cyclic(world, STRADDLE)
    assert (printed["route"], printed["cause"], printed["traced_steps"]) == ("cache files", None, 3)
    shown = show_trace(world)
    assert [s["step"] for s in shown["steps"]] == list(STRADDLE.steps)
    assert [s["consumed_samples"] for s in shown["steps"]] == [STRADDLE.batch] * 3


@pytest.mark.parametrize("case", [STRADDLE, EXACT], ids=["straddle", "exact"])
def test_each_step_of_a_cyclic_segment_prints_the_epochs_it_reads_in(world, case):
    register_cyclic(world, case)
    shown = show_trace(world)
    assert {s["step"]: s.get("epochs") for s in shown["steps"]} == case.epochs


def test_step_keeps_the_epochs_of_that_step(world):
    register_cyclic(world, STRADDLE)
    shown = show_trace(world, "--step", "2")
    assert [(s["step"], s.get("epochs")) for s in shown["steps"]] == [(2, [0, 1])]


def test_control_a_sequential_segment_prints_no_epochs(world):
    record = run_record(world, steps=(1, 2, 3), batch=2, launches=[launch(0, 0, 0, 1, 3, ranks=1)],
                        cyclic=False)
    register(world, record, (1, 2, 3), 2)
    shown = show_trace(world)
    assert [s["step"] for s in shown["steps"]] == [1, 2, 3]
    assert all("epochs" not in s for s in shown["steps"])


# --- 2. register-run prints the shuffle ------------------------------------------------------


@pytest.mark.parametrize("case", [STRADDLE, EXACT], ids=["straddle", "exact"])
def test_control_a_cyclic_run_with_every_digest_registers_by_the_cache_files(world, case):
    printed = register_cyclic(world, case)
    printed.pop("shuffle", None)  # the key under test: everything else is what the base prints
    assert printed == {
        "segment": SEGMENT, "route": "cache files", "cause": None, "cause_detail": None,
        "steps": len(case.steps), "traced_steps": len(case.steps), "untraced_steps": [],
        "checkpoints": [f"Q@{case.steps[-1]}"], "replay": NOT_NEEDED,
    }


@pytest.mark.parametrize("case", [STRADDLE, EXACT], ids=["straddle", "exact"])
def test_every_entry_of_a_cyclic_run_whose_orders_were_drawn_by_torch_is_equal(world, case):
    printed = register_cyclic(world, case)
    rows = recorded_rows(case.keys)
    entries = expected_shuffle(rows)
    assert all(entry["equal"] for entry in entries)  # the hand data: the record is the draw
    assert [(e["launch"], e["epoch"], e["rank"]) for e in entries] == case.keys
    assert printed.get("shuffle") == entries


def test_the_epoch_zero_orders_of_both_launches_are_equal_rank_by_rank(world):
    # The seed of an order is its epoch (docs/join.md section 6), so a resume inside epoch 0
    # draws the order launch 0 drew: launch 1's digest of epoch 0 is launch 0's, rank by rank.
    shuffle = register_cyclic(world, STRADDLE).get("shuffle")
    assert shuffle is not None and len(shuffle) == 6
    by_key = {(e["launch"], e["epoch"], e["rank"]): e for e in shuffle}
    for rank in range(RANKS):
        assert by_key[(0, 0, rank)]["digest"] == by_key[(1, 0, rank)]["digest"]
        assert by_key[(0, 0, rank)]["digest"] != by_key[(1, 1, rank)]["digest"]
    assert by_key[(0, 0, 0)]["digest"] != by_key[(0, 0, 1)]["digest"]


def test_a_changed_recorded_digest_is_the_one_entry_that_is_not_equal(world):
    rows = recorded_rows(STRADDLE.keys)
    rows[3] = {**rows[3], "digest": WRONG_DIGEST}  # launch 1, epoch 0, rank 1
    printed = register_cyclic(world, STRADDLE, shuffle=rows)
    assert (printed["route"], printed["cause"], printed["traced_steps"]) == (
        "none", "cyclic order digest mismatch", 0)
    entries = expected_shuffle(rows)
    assert [e["equal"] for e in entries] == [True, True, True, False, True, True]
    assert entries[3]["recorded_digest"] == WRONG_DIGEST
    assert entries[3]["digest"] != WRONG_DIGEST and entries[3]["count"] == entries[3]["recorded_count"]
    # Every entry is listed, the computed digest and the recorded one side by side.
    assert printed.get("shuffle") == entries


def test_a_changed_recorded_count_is_not_equal_either(world):
    rows = recorded_rows(EXACT.keys)
    rows[1] = {**rows[1], "count": BUCKET + 1}  # launch 0, epoch 0, rank 1: the digest is right
    printed = register_cyclic(world, EXACT, shuffle=rows)
    assert (printed["route"], printed["cause"]) == ("none", "cyclic order digest mismatch")
    entries = expected_shuffle(rows)
    assert [e["equal"] for e in entries] == [True, False, True, True]
    assert (entries[1]["count"], entries[1]["recorded_count"]) == (BUCKET, BUCKET + 1)
    assert printed.get("shuffle") == entries


def test_a_cyclic_record_with_no_shuffle_digest_prints_no_entries_and_its_cause(world):
    printed = register_cyclic(world, STRADDLE, shuffle=[])
    assert printed == {
        "segment": SEGMENT, "route": "none", "cause": "cyclic order not recorded",
        "cause_detail": None, "steps": 3, "traced_steps": 0,
        "untraced_steps": [[SEGMENT, step] for step in STRADDLE.steps],
        "checkpoints": ["Q@3"], "replay": NOT_NEEDED, "shuffle": [],
    }
