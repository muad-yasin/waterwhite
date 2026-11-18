"""A checkpoint's history over a trace document (plan 5.5 Check 10).

The trace document is the list ``segments``; each segment holds its name, its parent
checkpoint, the checkpoints saved in it with their steps, and its steps, each with
samples and each sample with rows. The history of a checkpoint is the rows of the steps
of its segment up to and including the checkpoint's step, in step order and, inside a
step, in reading order; then the history of the checkpoint the segment names as its
parent, the same way, up the chain. A parent's history stops at the parent's own step.

A step with no entry, a sample with no row and a parent the trace does not hold add
nothing; there is no placeholder and no error. A checkpoint the trace does not hold is
told apart from one it holds with no row: None, and an empty list. Two things the
trace could hold are errors and are never picked between: a checkpoint name held by
two segments, and a parent chain that loops. The document is only read. The module
imports nothing of the tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryRow:
    segment: str
    step: int
    shard: str
    ordinal: int
    token_start: int
    token_end: int


def _find(trace: Mapping, checkpoint: str):
    """The segment that holds the checkpoint and the checkpoint's step, or None."""
    found = []
    for seg in trace.get("segments", []):
        for ckpt in seg.get("checkpoints", []):
            if ckpt["name"] == checkpoint:
                found.append((seg, ckpt["step"]))
    if len(found) > 1:
        raise ValueError(f"{len(found)} segments hold a checkpoint named {checkpoint!r}")
    return found[0] if found else None


def _segment_rows(seg: Mapping, upto: int) -> list[HistoryRow]:
    rows: list[HistoryRow] = []
    for st in sorted(seg.get("steps", []), key=lambda s: s["step"]):
        if st["step"] > upto:
            continue
        for sample in st.get("samples", []):
            for r in sample.get("rows", []):
                rows.append(HistoryRow(
                    segment=seg["segment"], step=st["step"], shard=r["shard"],
                    ordinal=r["ordinal"], token_start=r["token_start"],
                    token_end=r["token_end"]))
    return rows


def history_rows(trace: Mapping, checkpoint: str) -> list[HistoryRow] | None:
    """The history of ``checkpoint``, or None where the trace holds no checkpoint of
    that name."""
    here = _find(trace, checkpoint)
    if here is None:
        return None
    rows: list[HistoryRow] = []
    seen = {checkpoint}
    while here is not None:
        seg, upto = here
        rows.extend(_segment_rows(seg, upto))
        parent = seg.get("parent_checkpoint")
        if parent is None:
            break
        if parent in seen:
            raise ValueError(f"the parent chain of {checkpoint!r} loops at {parent!r}")
        seen.add(parent)
        here = _find(trace, parent)
    return rows
