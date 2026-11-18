"""Developer tests of two checkpoint-path cases, through the command line only
(plan 3.5, 2.4, 2.2(e); docs/interfaces.md 7, 8.1, 8.2).

(a) A cyclic segment with no shuffle digest supports no "does not contain": a
    checkpoint of such a segment, asked about a record its history may have read,
    answers "may contain" naming "cyclic order not recorded", whatever cache files
    the run kept; the registration prints "shuffle": [] for it, as for every sequential
    segment (a cyclic one with recorded digests is in test_trace_launches_and_shuffle.py).
(b) A segment's trace route follows from its run record: the record with its cache
    folder and its recorded digests gives the route "cache files"; the same record
    with the cache folder gone, but its settings and digests recorded, gives
    "replay"; with the digests absent the route is "none" and the cause "data-order
    trace missing" with the reason "no recorded digest".

Each test makes a store in a temporary folder, writes the run record and the step
record by hand (docs/interfaces.md 8.1, 8.2), registers the segment with
``waterwhite register-run`` and puts the question with ``waterwhite query``. The
data is one shard of four documents under a run of three steps of two samples.
The arrays of the cache folder are those the index builder of the Megatron-LM
checkout of the test environment gives for that shard and those settings (seed 1234, sequence length
4, split 1,0,0, six samples, one epoch), worked out by hand from its source
(gpt_dataset.py: the document index shuffled by a RandomState of the seed, the
sample index walked over the document lengths, the shuffle index shuffled by the
same state) and not yet diffed against an index Megatron built; the digests are
those of oracle.digests over these arrays. Nothing in the test computes an order.

Nothing in this file imports Megatron or torch.
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
STEPS = (1, 2, 3)
BATCH = 2
# What register-run prints as "replay" when the cache files matched, or when no replay was
# tried because the order of a cyclic run was not recorded (docs/interfaces.md section 2).
NOT_NEEDED = {"attempted": False, "accepted": None, "reason": None}
SEQUENCE_LENGTH = 4
SEED = 1234
DOCUMENTS = [
    [11, 12, 13, 14, 15, 16],
    [21, 22, 23, 24, 25, 26, 27, 28, 29],
    [31, 32, 33, 34, 35],
    [41, 42, 43, 44, 45, 46, 47, 48],
]
# The release holds, in the order of the shard's documents, A's two lines and B's.
RELEASE_TEXTS = ["a one", "a two", "b one", "b two"]

CYCLIC_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler"
CYCLIC_JOIN = (ww_world.JOIN[0], v.SAMPLER_CYCLIC, CYCLIC_CLASS)

# The three arrays of the index build for the shard above under the settings above
# (a one-shard run builds no blend: three arrays and no blend rows). Unverified
# against a Megatron run: see the docstring.
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


def build_world(folder: Path) -> SimpleNamespace:
    """Sources A (two lines), B (two lines) and Z (one line, in no release); one
    release of A's and B's lines; one shard of four documents made from it, with
    its whole map; one mixture of that shard. The sampler types the store trusts are
    the sequential and the cyclic one of the same release."""
    store = Store.create(
        folder / "store.db", normaliser=ww_world.NORMALISER,
        verified_joins=[ww_world.JOIN, CYCLIC_JOIN],
    )
    a = ww_world.ingest(store, folder / "src/A.jsonl", ["a one", "a two"], "A", "src/A")
    b = ww_world.ingest(store, folder / "src/B.jsonl", ["b one", "b two"], "B", "src/B")
    ww_world.ingest(store, folder / "src/Z.jsonl", ["z one"], "Z", "src/Z")
    release = ww_world.record(
        store, folder / "R.jsonl", RELEASE_TEXTS, "R", v.ROLE_RELEASE,
        ww_world.step("release", "R"), [a, b],
        [ww_world.copy_edge(1, ww_world.ref(a, 1)), ww_world.copy_edge(2, ww_world.ref(a, 2)),
         ww_world.copy_edge(3, ww_world.ref(b, 1)), ww_world.copy_edge(4, ww_world.ref(b, 2))],
    )
    prefix = folder / "runs" / RUN / "shards" / "shard-1"
    count, digest = ww_world.write_shard(prefix, DOCUMENTS)
    shard = store.register_shard(
        prefix, name="shard-1", step=ww_world.step("tokenize", "shard-1"), inputs=[release])
    rows = [ww_world.recovered(n, ww_world.ref(release, n + 1)) for n in range(count)]
    store.write_map(
        shard,
        MapRecord(v.NAMESPACE_CAPTURE, v.DERIVED_POSITION, v.VERIFIED_PASSED, v.RECOVERED,
                  True, True, tuple(rows)),
        declared_count=count, declared_digest=digest,
    )
    mixture = store.register_mixture(
        name=MIXTURE, shards=[SnapshotShard(0, shard, f"runs/{RUN}/shards/shard-1", digest)],
        settings={"mix": "q"})
    store.close()
    return SimpleNamespace(
        folder=folder, path=folder / "store.db", runs=folder / "runs" / RUN, mixture=mixture)


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def run_record(world, *, sampler: str, digests: bool, shuffle: bool = False) -> dict:
    """The run record of docs/interfaces.md 8.1 for the one segment. ``digests`` says
    whether the wrapper recorded the digests of the cache arrays; ``shuffle`` whether
    it recorded a shuffle digest (never, in these tests)."""
    assert not shuffle
    cyclic = sampler == v.SAMPLER_CYCLIC
    record = ww_world.run_record(
        RUN, SEGMENT, "fresh", STEPS[0], STEPS[-1], sampler=sampler,
        sampler_class=CYCLIC_CLASS if cyclic else ww_world.SAMPLER_CLASS,
    )
    record["settings"]["global_batch_size"] = BATCH
    record["settings"]["micro_batch_size"] = 1
    record["settings"]["sequence_length"] = SEQUENCE_LENGTH
    record["settings"]["random_seed"] = SEED
    record["shards"] = [{
        "position": 0, "prefix": "shards/shard-1",
        "bin_sha256": file_sha256(world.runs / "shards" / "shard-1.bin"),
        "idx_sha256": file_sha256(world.runs / "shards" / "shard-1.idx"),
    }]
    record["cache_digests"] = [
        {"kind": kind, "shard": 0, "digest": CACHE_DIGESTS[kind]} for kind in CACHE_ARRAYS
    ] if digests else []
    record["shuffle_digests"] = []
    return record


def write_cache(folder: Path) -> Path:
    """The data-order cache folder, as Megatron wrote it: each array in a .npy file
    named by the dataset's description hash and its kind, and the description."""
    folder.mkdir(parents=True)
    for kind, (dtype, values) in CACHE_ARRAYS.items():
        np.save(folder / f"{CACHE_HASH}-GPTDataset-train-{kind}.npy", np.array(values, dtype=dtype))
    (folder / f"{CACHE_HASH}-GPTDataset-train-description.txt").write_text("{}", encoding="utf-8")
    return folder


def register(world, record: dict, *, cache: bool) -> dict:
    """Write the run record and the step record, keep the cache folder or not, and
    register the segment. Returns what ``register-run`` printed."""
    stem = world.runs / SEGMENT.replace("/", "-")
    run_file, steps_file = Path(f"{stem}.run.json"), Path(f"{stem}.steps.json")
    run_file.write_text(json.dumps(record), encoding="utf-8")
    steps_file.write_text(json.dumps({
        "segment": SEGMENT,
        "steps": [{"step": s, "batch_size": BATCH} for s in STEPS],
        "checkpoints": [{"name": CHECKPOINT, "step": STEPS[-1]}],
    }), encoding="utf-8")
    argv = ["register-run", "--store", str(world.path), "--run-record", str(run_file),
            "--steps", str(steps_file), "--mixture", MIXTURE]
    if cache:
        argv += ["--cache", str(write_cache(Path(f"{stem}.cache")))]
    return command(*argv)


def untraced(count: int) -> list[list]:
    return [[SEGMENT, step] for step in STEPS[:count]]


def ask(world, source: str) -> dict:
    return command("query", "withdraw", source, "--artifact", CHECKPOINT, "--store", str(world.path))


# --- (a) A cyclic segment with no shuffle digest supports no "does not contain" -------


def test_a_cyclic_segment_with_no_shuffle_digest_is_untraced_and_names_its_cause(world):
    printed = register(world, run_record(world, sampler=v.SAMPLER_CYCLIC, digests=True), cache=False)
    assert printed == {
        "segment": SEGMENT, "route": v.ROUTE_NONE, "cause": v.NAMES_CYCLIC_NOT_RECORDED,
        "cause_detail": None, "steps": 3, "traced_steps": 0, "untraced_steps": untraced(3),
        "checkpoints": [CHECKPOINT], "replay": NOT_NEEDED, "shuffle": [],
    }
    answer = ask(world, "A")
    assert (answer["answer"], answer["names"], answer["names_detail"]) == (
        "may contain", v.NAMES_CYCLIC_NOT_RECORDED, None)
    assert answer["untraced_steps"] == untraced(3)


def test_the_cache_files_a_cyclic_run_kept_do_not_replace_the_missing_shuffle_digest(world):
    printed = register(world, run_record(world, sampler=v.SAMPLER_CYCLIC, digests=True), cache=True)
    assert (printed["route"], printed["cause"], printed["traced_steps"]) == (
        v.ROUTE_NONE, v.NAMES_CYCLIC_NOT_RECORDED, 0)
    # A is read by the history; Z is read by none of it. Neither can be answered
    # "does not contain": the order of the samples is not shown to be the one trained.
    for source in ("A", "Z"):
        answer = ask(world, source)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_CYCLIC_NOT_RECORDED), source


def test_the_same_data_under_the_sequential_sampler_does_answer(world):
    # The control of (a): what differs is the sampler type and nothing else.
    printed = register(world, run_record(world, sampler=v.SAMPLER_SEQUENTIAL, digests=True), cache=True)
    assert printed["route"] == v.ROUTE_CACHE_FILES
    absent = ask(world, "Z")
    assert (absent["answer"], absent["names"], absent["untraced_steps"]) == ("does not contain", None, [])
    assert absent["scope"]["steps"] == [[SEGMENT, 1, 3]]


# --- (b) The trace route follows from the run record ----------------------------------


def test_with_its_cache_folder_and_recorded_digests_the_route_is_cache_files(world):
    printed = register(world, run_record(world, sampler=v.SAMPLER_SEQUENTIAL, digests=True), cache=True)
    assert printed == {
        "segment": SEGMENT, "route": "cache files", "cause": None, "cause_detail": None,
        "steps": 3, "traced_steps": 3, "untraced_steps": [], "checkpoints": [CHECKPOINT],
        "replay": NOT_NEEDED, "shuffle": [],
    }
    # The six samples of the run cover four documents: A's two lines are read.
    read = ask(world, "A")
    assert (read["answer"], read["level"], read["basis"], read["untraced_steps"]) == (
        "contains", "consumed", v.BASIS_WHOLE_SAMPLE, [])


def test_with_the_cache_folder_gone_but_digests_recorded_the_route_is_replay(world):
    printed = register(world, run_record(world, sampler=v.SAMPLER_SEQUENTIAL, digests=True), cache=False)
    assert printed == {
        "segment": SEGMENT, "route": "replay", "cause": None, "cause_detail": None,
        "steps": 3, "traced_steps": 3, "untraced_steps": [], "checkpoints": [CHECKPOINT],
        "replay": {"attempted": True, "accepted": True, "reason": None}, "shuffle": [],
    }
    # A replay that equals the recorded digests answers as the cache files do.
    read = ask(world, "A")
    assert (read["answer"], read["level"], read["untraced_steps"]) == ("contains", "consumed", [])
    absent = ask(world, "Z")
    assert (absent["answer"], absent["untraced_steps"]) == ("does not contain", [])


def test_with_no_recorded_digest_the_route_is_none_and_the_cause_is_data_order_trace_missing(world):
    printed = register(world, run_record(world, sampler=v.SAMPLER_SEQUENTIAL, digests=False), cache=False)
    assert printed == {
        "segment": SEGMENT, "route": "none", "cause": "data-order trace missing",
        "cause_detail": "no recorded digest", "steps": 3, "traced_steps": 0,
        "untraced_steps": untraced(3), "checkpoints": [CHECKPOINT],
        "replay": {"attempted": False, "accepted": None, "reason": "no recorded digest"},
        "shuffle": [],
    }
    for source in ("A", "Z"):
        answer = ask(world, source)
        assert (answer["answer"], answer["names"], answer["names_detail"]) == (
            "may contain", "data-order trace missing", "no recorded digest"), source
