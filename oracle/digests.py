"""Digests the oracle writes into the run record.

The tool computes the same digests in ``src/waterwhite/ids.py``. The oracle must
share no code with the tool (plan W5), so each function here is written again from
the written definition in that file's docstrings, and a test compares the two on
the same inputs (tests/oracle/test_oracle_plain.py).

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from oracle.errors import OracleError

_READ_CHUNK = 1 << 20

# The five kinds of data-order index array (plan 3.5): three per shard, two for the
# blend. The names are the last part of the file names Megatron writes.
SHARD_KINDS = ("document_index", "sample_index", "shuffle_index")
BLEND_KINDS = ("dataset_index", "dataset_sample_index")


def _canonical(value: object) -> bytes:
    """Canonical JSON as ids.py defines it: sorted keys, no spaces, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes, lower-case hex (the content digest of ids.py)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_READ_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def array_digest(dtype: str, shape: Sequence[int], data: bytes) -> str:
    """Digest of one index array, as ids.array_digest defines it.

    SHA-256 over: the canonical JSON of the list ["ww.array.v1", dtype, shape], one
    zero byte, then the array's bytes in C order. ``dtype`` is numpy's ``dtype.str``
    (for example "<i4"). The digest is of the array and not of the ``.npy`` file."""
    h = hashlib.sha256()
    h.update(_canonical(["ww.array.v1", dtype, [int(n) for n in shape]]))
    h.update(b"\x00")
    h.update(data)
    return h.hexdigest()


def order_digest(indices: Iterable[int]) -> str:
    """Digest of an ordered list of sample indices, as ids.order_digest defines it.

    SHA-256 of the JSON list of the indices, in order, with no spaces."""
    text = json.dumps([int(i) for i in indices], separators=(",", ":"))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def compare_digests(replayed: list[dict], recorded: list[dict]) -> dict:
    """Compare replayed digests with recorded ones, array by array.

    Both are lists of rows with ``kind``, ``shard`` and ``digest`` (a run record's
    ``cache_digests`` or a digests document's). An array on one side only is
    listed and makes ``all_equal`` false."""

    def keyed(rows: list[dict], what: str) -> dict:
        out = {}
        for row in rows:
            key = (row["kind"], row["shard"])
            if key in out:
                raise OracleError(f"{what} holds {key} twice")
            out[key] = row["digest"]
        return out

    new, old = keyed(replayed, "the replay"), keyed(recorded, "the record")

    def order(key):
        return (key[1] is None, key[1] if key[1] is not None else -1, key[0])

    rows = [
        {
            "kind": kind,
            "shard": shard,
            "recorded": old[(kind, shard)],
            "replayed": new[(kind, shard)],
            "equal": old[(kind, shard)] == new[(kind, shard)],
        }
        for kind, shard in sorted(set(new) & set(old), key=order)
    ]
    only_recorded = [list(k) for k in sorted(set(old) - set(new), key=order)]
    only_replayed = [list(k) for k in sorted(set(new) - set(old), key=order)]
    return {
        "rows": rows,
        "only_recorded": only_recorded,
        "only_replayed": only_replayed,
        "all_equal": all(r["equal"] for r in rows) and not only_recorded and not only_replayed,
    }
