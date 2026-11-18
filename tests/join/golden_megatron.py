"""Index arrays the trainer's own code built, kept as numbers, so that a test
with no trainer installed can hold the replay against them.

Where they come from: the data-order cache folder the oracle kept for segment
E/seg-1 of the hand-checked fixture (``oracle/fixture_runs.py`` over a fixture the
replica built from ``fixture/spec/fixture_spec.toml``), read with ``numpy.load``.
``LENGTHS`` are the token counts of the documents of that segment's two shards,
read from the shards' index files. Nothing here was computed by the join.

The file stands on its own: it holds the lengths the arrays were built from. If
the fixture's texts change, these numbers still say what the trainer's code
builds for these lengths and this seed. ``LOG_LINES`` are rows of the oracle's
log of the same segment that the fixture spec quotes under ``[positions]``; they
hold only while the fixture's shards have these lengths.
"""

from __future__ import annotations

import numpy as np

SETTINGS = {"sequence_length": 32, "random_seed": 578, "split": "1,0,0", "train_iters": 6,
            "global_batch_size": 4}

LENGTHS = (
    [49, 51, 52, 56, 18, 22, 45, 53, 44, 46, 19, 74, 37, 51, 59, 52, 16, 54],
    [22, 40, 45, 47, 53, 43, 43, 54, 59, 36, 56, 54, 17, 20, 35, 39, 53, 50, 61],
)

ARRAYS = {
    ("document_index", 0): np.array(
        [11, 8, 0, 17, 7, 4, 16, 2, 12, 6, 1, 10, 5, 3, 14, 9, 15, 13], dtype="<i4"),
    ("sample_index", 0): np.array(
        [[0, 0], [0, 32], [0, 64], [1, 22], [2, 10], [2, 42], [3, 25], [4, 3], [4, 35], [5, 14],
         [7, 12], [7, 44], [8, 24], [9, 19], [10, 6], [10, 38], [12, 0], [13, 10], [13, 42],
         [14, 18], [14, 50], [15, 23], [16, 9], [16, 41], [17, 21]], dtype="<i4"),
    ("shuffle_index", 0): np.array(
        [3, 20, 21, 10, 17, 5, 7, 14, 12, 23, 19, 4, 0, 6, 18, 22, 13, 1, 16, 9, 11, 8, 15, 2],
        dtype="<u4"),
    ("document_index", 1): np.array(
        [12, 9, 0, 17, 8, 4, 18, 7, 2, 16, 6, 11, 1, 10, 5, 3, 14, 15, 13], dtype="<i4"),
    ("sample_index", 1): np.array(
        [[0, 0], [1, 15], [2, 11], [3, 21], [4, 3], [4, 35], [5, 8], [5, 40], [6, 19], [6, 51],
         [7, 22], [8, 0], [8, 32], [9, 19], [9, 51], [10, 30], [11, 19], [11, 51], [12, 29],
         [13, 21], [13, 53], [14, 29], [15, 18], [16, 3], [17, 0], [17, 32]], dtype="<i4"),
    ("shuffle_index", 1): np.array(
        [3, 14, 12, 22, 21, 18, 5, 10, 17, 7, 19, 23, 24, 20, 4, 0, 6, 13, 1, 16, 9, 11, 8, 15, 2],
        dtype="<u4"),
    ("dataset_index", None): np.array([1, 0] * 12, dtype="<i2"),
    ("dataset_sample_index", None): np.array(
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11], dtype="<i8"),
}

# (step, sample_pos, global_index, shard_position, ordinal, token_start, token_end)
LOG_LINES = (
    (1, 0, 0, 1, 17, 21, 50),
    (1, 0, 0, 1, 8, 0, 4),
    (1, 1, 1, 0, 8, 22, 44),
    (3, 2, 10, 1, 1, 29, 40),
    (4, 3, 15, 0, 1, 6, 39),
    (6, 3, 23, 0, 0, 10, 43),
)
