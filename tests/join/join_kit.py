"""What the join's tests build by hand: tiny shards, index arrays written out as a
trainer's cache folder would hold them, and run records.

Nothing here comes from the join. A shard's index file is written from the layout
docs/store.md and the store's reader document. A recorded digest is taken with
``waterwhite.ids``, the definition the training wrapper uses. The index arrays of
the two hand worlds are written by hand; ``WORLD`` below derives every row a
sample of them reads.

The hand world. Sequence length 4, extra token on, so a sample is 5 tokens.

Shard 0 holds four documents of 6, 3, 4 and 5 tokens (D0 to D3). Its document
index, written by hand, is [2, 0, 3, 1]. Laid end to end in that order the 18
tokens are: D2 at 0 to 3, D0 at 4 to 9, D3 at 10 to 14, D1 at 15 to 17. There are
(18 - 1) div 4 = 4 samples, and sample k starts at token 4k of the stream and
reads 5 tokens. The sample index marks the starts 0, 4, 8, 12 and the end 16 as
(place in the document index, offset in that document):

    token  0 is D2 offset 0, place 0  -> [0, 0]
    token  4 is D0 offset 0, place 1  -> [1, 0]
    token  8 is D0 offset 4, place 1  -> [1, 4]
    token 12 is D3 offset 2, place 2  -> [2, 2]
    token 16 is D1 offset 1, place 3  -> [3, 1]

So, in sample-index order, with the extra token as each sample's last:

    S0 = D2[0:4] and D0[0:1]   D0 is read through the shared last token only
    S1 = D0[0:5]               one document, one row; its first token is S0's last
    S2 = D0[4:6] and D3[0:3]   D0 is split across S1 and S2; D0[4] is in both
    S3 = D3[2:5] and D1[0:2]   D3[2] is in S2 and S3

The shuffle index, written by hand, is [2, 0, 3, 1]: sample number 0 of the shard
is S2, number 1 is S0, number 2 is S3, number 3 is S1.

Shard 1 holds two documents of 5 tokens each. Its document index is [1, 0]: D1 at
0 to 4, D0 at 5 to 9. (10 - 1) div 4 = 2 samples; starts 0 and 4, end 8:

    token 0 is D1 offset 0, place 0  -> [0, 0]
    token 4 is D1 offset 4, place 0  -> [0, 4]
    token 8 is D0 offset 3, place 1  -> [1, 3]

    S0 = D1[0:5]
    S1 = D1[4:5] and D0[0:4]

Its shuffle index is [1, 0]: number 0 is S1, number 1 is S0. Ordinals 0 and 1
exist in both shards.

The blend of the two (6 samples), written by hand:

    dataset index        [0, 1, 0, 0, 1, 0]
    dataset sample index [0, 0, 1, 2, 1, 3]

So index g of the blend reads (shard position, ordinal, token start, token end):

    g0 = shard 0 number 0 = S2: (0, 0, 4, 6), (0, 3, 0, 3)
    g1 = shard 1 number 0 = S1: (1, 1, 4, 5), (1, 0, 0, 4)
    g2 = shard 0 number 1 = S0: (0, 2, 0, 4), (0, 0, 0, 1)
    g3 = shard 0 number 2 = S3: (0, 3, 2, 5), (0, 1, 0, 2)
    g4 = shard 1 number 1 = S0: (1, 1, 0, 5)
    g5 = shard 0 number 3 = S1: (0, 0, 0, 5)
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import golden_megatron as golden
import numpy as np
import pytest

from waterwhite import ids

TAG ="core_v0.20.0"
SEQUENTIAL_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
CYCLIC_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler"
JOINS = ((TAG, "single", SEQUENTIAL_CLASS), (TAG, "cyclic", CYCLIC_CLASS))

SEQUENCE_LENGTH = 4
SAMPLE_TOKENS = 5

LENGTHS = ([6, 3, 4, 5], [5, 5])
ARRAYS = {
    ("document_index", 0): np.array([2, 0, 3, 1], dtype=np.int32),
    ("sample_index", 0): np.array([[0, 0], [1, 0], [1, 4], [2, 2], [3, 1]], dtype=np.int32),
    ("shuffle_index", 0): np.array([2, 0, 3, 1], dtype=np.uint32),
    ("document_index", 1): np.array([1, 0], dtype=np.int32),
    ("sample_index", 1): np.array([[0, 0], [0, 4], [1, 3]], dtype=np.int32),
    ("shuffle_index", 1): np.array([1, 0], dtype=np.uint32),
    ("dataset_index", None): np.array([0, 1, 0, 0, 1, 0], dtype=np.int16),
    ("dataset_sample_index", None): np.array([0, 0, 1, 2, 1, 3], dtype=np.int64),
}

# The rows of each index of the blend, derived in the text above.
WORLD = {
    0: [(0, 0, 4, 6), (0, 3, 0, 3)],
    1: [(1, 1, 4, 5), (1, 0, 0, 4)],
    2: [(0, 2, 0, 4), (0, 0, 0, 1)],
    3: [(0, 3, 2, 5), (0, 1, 0, 2)],
    4: [(1, 1, 0, 5)],
    5: [(0, 0, 0, 5)],
}
# The rows of each sample number of shard 0 alone (a segment of one shard).
WORLD_SHARD_0 = {0: WORLD[0], 1: WORLD[2], 2: WORLD[3], 3: WORLD[5]}


def write_shard(prefix: Path, lengths: list[int]) -> str:
    """Write a shard's two files in the documented layout and return the SHA-256
    of its index file. Every token is 7: the join never reads a token."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    count = len(lengths)
    offsets = [4 * sum(lengths[:k]) for k in range(count)]
    index = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, count, count + 1)
    index += struct.pack(f"<{count}i", *lengths)
    index += struct.pack(f"<{count}q", *offsets)
    index += struct.pack(f"<{count + 1}q", *range(count + 1))
    Path(str(prefix) + ".idx").write_bytes(index)
    Path(str(prefix) + ".bin").write_bytes(struct.pack(f"<{sum(lengths)}i", *([7] * sum(lengths))))
    return hashlib.sha256(index).hexdigest()


def array_digest(array: np.ndarray) -> str:
    return ids.array_digest(array.dtype.str, array.shape, array.tobytes())


def write_cache(cache_dir: Path, arrays: dict) -> None:
    """Write index arrays under the kind of names a trainer's cache folder holds:
    a hash, the dataset class, the split and the kind."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for (kind, shard), array in arrays.items():
        owner = "BlendedDataset" if shard is None else "GPTDataset"
        stem = hashlib.md5(f"{owner}-{shard}".encode()).hexdigest()
        np.save(cache_dir / f"{stem}-{owner}-train-{kind}.npy", array)
        (cache_dir / f"{stem}-{owner}-train-description.txt").write_text("{}", encoding="utf-8")


def run_record(folder: Path, *, shards: int = 2, arrays: dict | None = None, sampler: str = "single",
               micro: int = 2, batch: int = 2, launches: list[dict] | None = None,
               shuffle: list[dict] | None = None, lengths=LENGTHS, **settings) -> dict:
    """A run record of the hand world, with its shard files written under
    ``folder``. ``shards`` 1 gives a segment of shard 0 alone, with no blend."""
    arrays = ARRAYS if arrays is None else arrays
    listed = []
    for position in range(shards):
        digest = write_shard(folder / "shards" / f"shard-{position}", list(lengths[position]))
        listed.append({"position": position, "prefix": f"shards/shard-{position}",
                       "bin_sha256": "0" * 64, "idx_sha256": digest})
    keys = [key for key in arrays if key[1] is not None and key[1] < shards]
    if shards >= 2:
        keys += [key for key in arrays if key[1] is None]
    record = {
        "format": "waterwhite.run-record/1",
        "run": "H",
        "segment": "H/seg-1",
        "start_kind": "fresh",
        "parent_checkpoint": None,
        "settings": {
            "dataloader_type": sampler,
            "sampler_class": SEQUENTIAL_CLASS if sampler == "single" else CYCLIC_CLASS,
            "data_sharding": True,
            "micro_batch_size": micro,
            "global_batch_size": batch,
            "sequence_length": SEQUENCE_LENGTH,
            "add_extra_token_to_sequence": True,
            "random_seed": 1234,
            "split": "1,0,0",
            "train_iters": 3,
            "train_samples": None,
            "batch_schedule": None,
        },
        "versions": {"release_tag": TAG, "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.14.1+cpu"},
        "shards": listed,
        "launches": launches if launches is not None else [launch(0, 1, 0, 1, 3)],
        "cache_digests": [{"kind": kind, "shard": shard, "digest": array_digest(arrays[(kind, shard)])}
                          for kind, shard in keys],
        "shuffle_digests": shuffle or [],
    }
    record["settings"].update(settings)
    return record


def launch(position: int, ranks: int, consumed: int, first: int, last: int) -> dict:
    return {"position": position, "world_size": ranks, "data_parallel_size": ranks,
            "consumed_at_start": consumed, "step_at_start": first - 1, "first_step": first,
            "last_step": last}


def shuffle_rows(orders: dict) -> list[dict]:
    """The shuffle digests a wrapper records for orders keyed by (launch, epoch,
    rank): the digest of each whole order, and its length."""
    return [{"launch": key[0], "epoch": key[1], "rank": key[2],
             "digest": ids.order_digest(order), "count": len(order)}
            for key, order in sorted(orders.items())]


# The cyclic run of the hand world. The blend holds 6 samples. Micro batch 1 on 2
# ranks: a round is 2 samples, an epoch is 6 samples and so 3 steps of batch 2,
# rank 0 owns indices 0, 1, 2 and rank 1 owns 3, 4, 5 (a bucket of 3).
BUCKET = 3
RANKS = 2


def torch_orders(launch_epochs: dict[int, list[int]]) -> dict:
    """The whole order of each (launch, epoch, rank), drawn here with this test's own
    call to torch, as the cyclic sampler draws it and as a training wrapper does it
    in the run's own environment: a generator seeded with the epoch number, a
    permutation of the bucket, the rank's first index added. Nothing of the tool is
    used, so a digest the tool's shuffle must equal is made by other code."""
    torch = pytest.importorskip("torch")
    out = {}
    for position, epochs in launch_epochs.items():
        for epoch in epochs:
            generator = torch.Generator()
            generator.manual_seed(epoch)
            place = torch.randperm(BUCKET, generator=generator).tolist()
            for rank in range(RANKS):
                out[(position, epoch, rank)] = [rank * BUCKET + p for p in place]
    return out


def stand_in_orders(launch_epochs: dict[int, list[int]]) -> dict:
    """Orders that no sampler drew, for a test whose digests are never held against
    a computed order (the join stops before it, or torch is not to be imported)."""
    return {(position, epoch, rank): [rank * BUCKET + p for p in range(BUCKET)]
            for position, epochs in launch_epochs.items() for epoch in epochs
            for rank in range(RANKS)}


def cyclic_places(orders: dict, launch: int, counter: int) -> list[tuple[int, int, int]]:
    """(sample_pos, global_index, rank) of the step that starts at ``counter``, derived
    from the orders alone: a round is one index of each rank's order, the epoch is
    the counter div 6 and the place in each order is the rest div 2."""
    epoch, inside = divmod(counter, BUCKET * RANKS)
    place = inside // RANKS
    return [(rank, orders[(launch, epoch, rank)][place], rank) for rank in range(RANKS)]


def cyclic_record(folder: Path, launches: list[dict], orders: dict) -> dict:
    """A run record of the cyclic run, with the digests of ``orders`` recorded."""
    return run_record(folder, sampler="cyclic", micro=1, batch=2, launches=launches,
                      shuffle=shuffle_rows(orders), train_iters=4)


def golden_record(folder: Path, **settings) -> dict:
    """A run record of the golden segment (``golden_megatron``): its two shards
    written from the golden lengths, the digests of the golden arrays recorded."""
    record = run_record(folder, arrays=golden.ARRAYS, lengths=golden.LENGTHS, micro=4, batch=4,
                        launches=[launch(0, 1, 0, 1, 6)], **golden.SETTINGS)
    record["settings"].update(settings)
    return record


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def rows(sample) -> list[tuple[int, int, int, int]]:
    return [(r.shard_position, r.ordinal, r.token_start, r.token_end) for r in sample.rows]


def placed(result) -> dict[int, list[tuple[int, int, int]]]:
    """Per step: (sample_pos, global_index, rank) of every sample of its trace."""
    return {trace.step: [(s.sample_pos, s.global_index, s.rank) for s in trace.samples]
            for trace in result.traces}
