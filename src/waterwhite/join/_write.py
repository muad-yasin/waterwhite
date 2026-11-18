"""The replay written to files, for a person or a check to hold against kept cache
files (plan 3.5 "replay", Check 2 (i); docs/cli.md "replay").

``write_replay`` derives the index arrays of one run segment again from its run
record and its shards, writes each to a folder as a ``.npy`` file and says, per
array, which digest it has and which the run record holds. It writes no store and
decides nothing: whether a replay counts for an answer is decided at registration
(docs/join.md section 4), and whether the written bytes equal a kept cache file is
the comparison of whoever holds the two.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ..errors import ReplayNotPossible, UsageError
from ._arrays import digest_of, needed_keys
from ._join import _shard_lengths
from ._record import parse_run_record
from ._replay import replay_arrays


def _file_name(kind: str, shard: int | None) -> str:
    return f"{kind}.npy" if shard is None else f"{kind}-{shard}.npy"


def write_replay(document: Mapping, record_dir: str | os.PathLike, out_dir: str | os.PathLike) -> dict:
    """Derive every index array of the segment a run record describes and write them
    to ``out_dir`` (made when it is not there), one file per array, named
    ``<kind>.npy`` for the blend's two and ``<kind>-<shard position>.npy`` for a
    shard's three. Returns the object ``waterwhite replay`` prints.

    ``record_dir`` is the folder the run record lies in: a shard prefix of the record
    is relative to it. Each array is written as the replay made it, dtype and shape
    untouched.

    Raises ``ReplayNotPossible`` when the replay cannot be made: the run record lacks
    a setting it needs, lists no shard, or names a shard whose index file is not
    there, cannot be read, or is not the file the record's digest names (the replay
    would derive from other lengths than the trainer saw). ``UsageError`` when a file
    the replay would write is already there: nothing is written then, and a written
    array is never overwritten. ``InvalidRecord`` for a document that is no run record."""
    record = parse_run_record(document)
    if not record.shards:
        raise ReplayNotPossible("the run record lists no shard, so there is nothing to derive "
                                "the arrays from")
    lengths, _, why = _shard_lengths(record, record_dir)
    if lengths is None:
        raise ReplayNotPossible(why)
    arrays = replay_arrays(record, lengths)
    keys = needed_keys(len(lengths))
    folder = Path(out_dir)
    taken = [name for name in (_file_name(*key) for key in keys) if (folder / name).exists()]
    if taken:
        raise UsageError(f"{taken[0]} is already in {folder}; a written array is not "
                         "overwritten, name another folder", folder=str(folder))
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        kind, shard = key
        target = folder / _file_name(kind, shard)
        array = arrays[key]
        np.save(target, array, allow_pickle=False)
        digest = digest_of(array)
        recorded = record.cache_digests.get(key)
        rows.append({
            "kind": kind,
            "shard": shard,
            "file": str(target),
            "digest": digest,
            "recorded_digest": recorded,
            "equal": None if recorded is None else digest == recorded,
        })
    return {"segment": record.segment, "arrays": rows}
