"""From a sample to the rows it read (plan 3.5 "step trace", Decision 18, F9).

The chain, for the index a sampler asks of the top dataset:

1. In a blend, the dataset index gives the shard (its position in the run
   record's shard list) and the dataset sample index gives the sample number
   inside that shard. A segment of one shard has no blend: the index is the
   sample number.
2. The shard's shuffle index turns the sample number into a place in its sample
   index.
3. Entries ``k`` and ``k + 1`` of the sample index give the sample's first and
   last place in the document index, each with a token offset.
4. The document index turns each place into a document ordinal of the shard.

A sample spans ``sequence_length + 1`` tokens with the extra token on: it ends
one token past entry ``k + 1``, on the token the next sample in sample-index
order starts with. That token is read as a label, and a document a step read
only through it is consumed (plan 3.5, Decision 18). So the rows of a sample are:
the first document from the first offset to its end, every document between
whole, and the last document from token 0 to the last offset plus the extra
token; or, when the sample lies in one document, that one range.

A sample whose rows do not come to exactly its token count, or that points
outside an array or a document, is not resolved: it is written with no row and a
coverage mark of 0, never filled in (plan 3.5).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..records import TraceRow
from ._arrays import IndexArrays


class SampleResolver:
    """Resolves samples of one segment. ``lengths`` holds, per shard position, the
    token count of each document, read from the shard's own index file."""

    def __init__(self, arrays: IndexArrays, lengths: Sequence[np.ndarray], sequence_length: int,
                 extra: int) -> None:
        self._arrays = arrays
        self._lengths = lengths
        self._extra = extra
        self.sample_tokens = sequence_length + extra

    def resolve(self, global_index: int) -> tuple[tuple[TraceRow, ...], int]:
        """The rows of one sample and how many of its tokens they cover."""
        rows = self._rows(global_index)
        if rows is None:
            return (), 0
        covered = sum(row.token_end - row.token_start for row in rows)
        if covered > self.sample_tokens:
            return (), 0
        return tuple(rows), covered

    def _rows(self, global_index: int) -> list[TraceRow] | None:
        arrays = self._arrays
        if not 0 <= global_index < arrays.length:
            return None
        if arrays.dataset_index is not None:
            position = int(arrays.dataset_index[global_index])
            number = int(arrays.dataset_sample_index[global_index])
        else:
            position, number = 0, global_index
        if not 0 <= position < len(arrays.shards):
            return None
        shard = arrays.shards[position]
        lengths = self._lengths[position]
        if not 0 <= number < int(shard.shuffle_index.shape[0]):
            return None
        place = int(shard.shuffle_index[number])
        if not 0 <= place < shard.samples:
            return None
        first, first_offset = (int(x) for x in shard.sample_index[place])
        last, last_offset = (int(x) for x in shard.sample_index[place + 1])
        if not 0 <= first <= last < int(shard.document_index.shape[0]):
            return None
        rows = []
        for at in range(first, last + 1):
            ordinal = int(shard.document_index[at])
            if not 0 <= ordinal < int(lengths.shape[0]):
                return None
            length = int(lengths[ordinal])
            start = first_offset if at == first else 0
            end = last_offset + self._extra if at == last else length
            if not 0 <= start <= end <= length:
                return None
            if end > start:
                rows.append(TraceRow(position, ordinal, start, end))
        return rows


def cut(rows: Sequence[TraceRow], tokens: int) -> tuple[TraceRow, ...]:
    """The rows of the first ``tokens`` tokens of a sample: the rows that end
    inside them stay, the row that holds the next token ends there, and no row
    after it is kept (plan 3.5; the step record lowers a step's last sample)."""
    kept: list[TraceRow] = []
    used = 0
    for row in rows:
        size = row.token_end - row.token_start
        if used + size <= tokens:
            kept.append(row)
            used += size
            continue
        if tokens > used:
            kept.append(TraceRow(row.shard_position, row.ordinal, row.token_start,
                                 row.token_start + tokens - used))
        break
    return tuple(kept)
