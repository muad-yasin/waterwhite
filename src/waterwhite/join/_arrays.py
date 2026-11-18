"""The data-order index arrays of one run segment, and the cache-files route.

A segment has three arrays per shard (document index, sample index, shuffle
index) and, when it has two or more shards, two for the blend (dataset index,
dataset sample index). A segment of one shard builds no blend (plan 3.5;
docs/interfaces.md 8.1, 8.4).

An array counts only when its digest (``ids.array_digest``, DECISIONS.md D-08)
equals the digest the run record holds for its kind and shard position. That is
also how a cache file is tied to a shard here: the file whose array has the
digest recorded for (kind, shard position) is that shard's array. A file name
carries a hash of paths and settings and is read only to tell the five kinds
apart; nothing is read from ``description.txt``, whose paths change when a
folder moves. The run record's shard list gives the order that ties a dataset
number of the blend to a shard (Decision 27).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import ids
from .. import vocabulary as v

ArrayKey = tuple[str, int | None]  # (kind, shard position), None for the blend


def digest_of(array: np.ndarray) -> str:
    """The digest of one index array, as the run record holds it."""
    data = np.ascontiguousarray(array)
    return ids.array_digest(data.dtype.str, data.shape, data.tobytes())


def needed_keys(shard_count: int) -> list[ArrayKey]:
    """The arrays a segment of this many shards has."""
    return v.cache_keys(shard_count)


@dataclass(frozen=True)
class ShardArrays:
    document_index: np.ndarray
    sample_index: np.ndarray
    shuffle_index: np.ndarray

    @property
    def samples(self) -> int:
        """The number of samples the shard's dataset holds."""
        return max(int(self.sample_index.shape[0]) - 1, 0)


@dataclass(frozen=True)
class IndexArrays:
    """The arrays of one segment, each already held against its recorded digest."""

    shards: tuple[ShardArrays, ...]
    dataset_index: np.ndarray | None
    dataset_sample_index: np.ndarray | None

    @property
    def length(self) -> int:
        """The length of the dataset the sampler runs over: the blend's, or the
        one shard's when there is no blend."""
        if self.dataset_index is not None:
            return int(self.dataset_index.shape[0])
        return self.shards[0].samples

    @classmethod
    def build(cls, arrays: Mapping[ArrayKey, np.ndarray], shard_count: int) -> "IndexArrays | None":
        """Put verified arrays in their places. None when an array has a shape no
        index array of its kind can have: such a set is not readable."""
        shards = []
        for position in range(shard_count):
            document, sample, shuffle = (arrays[(kind, position)]
                                         for kind in v.CACHE_KINDS_PER_SHARD)
            if document.ndim != 1 or shuffle.ndim != 1 or sample.ndim != 2 or sample.shape[1] != 2:
                return None
            shards.append(ShardArrays(document, sample, shuffle))
        blend = blend_sample = None
        if shard_count >= 2:
            blend = arrays[(v.CACHE_DATASET_INDEX, None)]
            blend_sample = arrays[(v.CACHE_DATASET_SAMPLE_INDEX, None)]
            if blend.ndim != 1 or blend_sample.shape != blend.shape:
                return None
        return cls(tuple(shards), blend, blend_sample)


def _kind_of(name: str) -> str | None:
    """The kind of index array a cache file name stands for, by its ending."""
    for kind in v.CACHE_KINDS_PER_SHARD + v.CACHE_KINDS_PER_BLEND:
        if name == f"{kind}.npy" or name.endswith(f"-{kind}.npy"):
            return kind
    return None


def read_cache_files(
    cache_dir: str | os.PathLike | None, recorded: Mapping[ArrayKey, str], keys: list[ArrayKey]
) -> tuple[dict[ArrayKey, np.ndarray], list[ArrayKey], bool]:
    """The cache-files route of plan 3.5: for each needed array, the cache file of
    its kind whose array digest equals the recorded digest.

    Returns the arrays found, the keys with no matching file, and whether the
    folder held any index file at all. A file that cannot be read as a plain
    array matches nothing."""
    found: dict[ArrayKey, np.ndarray] = {}
    if cache_dir is None or not Path(cache_dir).is_dir():
        return found, list(keys), False
    by_kind: dict[str, dict[str, np.ndarray]] = {}
    any_file = False
    for path in sorted(Path(cache_dir).iterdir()):
        kind = _kind_of(path.name)
        if kind is None or not path.is_file():
            continue
        any_file = True
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            continue
        by_kind.setdefault(kind, {})[digest_of(array)] = array
    missing = []
    for key in keys:
        array = by_kind.get(key[0], {}).get(recorded[key])
        if array is None:
            missing.append(key)
        else:
            found[key] = array
    return found, missing, any_file
