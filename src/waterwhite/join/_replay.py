"""The replay route: the index arrays of a segment, derived again from the run
record's settings and seed and the shards' document lengths, with numpy alone
(plan 3.5 "replay", F3; Check 2 of plan 5.5).

Nothing here imports or copies the trainer's code. The module states, in its own
words, the index semantics the release the join is verified for documents and
shows (plan 3.5), and it is believed by nobody: the join accepts a replayed array
only when its digest equals the digest the run record holds for it. A setting
stated wrongly, a different seed, another release or another machine's
arithmetic therefore shows as a digest that differs, and the replay then
supports no answer.

The semantics, per shard:

- The documents of the training split are the first ``round(share * count)``
  documents, ``share`` being the first number of the split after the three are
  scaled to sum to 1.
- One epoch is every one of those documents once. A shard of a blend is built
  for one epoch. A shard that is the whole dataset is built for as many epochs
  as the requested samples need, a sample taking ``sequence_length`` tokens and
  the last one extra token when the extra token is on.
- One ``numpy.random.RandomState(seed)`` is used in a fixed order: it shuffles
  the document index (the documents, once per epoch, in one array), then the
  shuffle index (the sample numbers). When the last epoch supplies fewer than
  0.80 of an epoch's samples, it is shuffled apart from the epochs before it,
  in both arrays.
- The sample index marks where each sample starts. The documents are laid end
  to end in document-index order; sample ``i`` starts at token
  ``i * sequence_length`` of that stream and reads ``sequence_length`` tokens,
  plus one when the extra token is on. Entry ``i`` holds the place in the
  document index and the token offset inside that document. With the extra
  token on, the entry points at the token two neighbouring samples share.

The blend: its weights are the shards' sample counts scaled to sum to 1, its
length is the smaller of the requested samples and the samples the shards hold,
and place ``i`` goes to the shard that is furthest behind its share, the first
such shard on a tie. No random number is used.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .. import vocabulary as v
from ..errors import ReplayNotPossible
from ._arrays import ArrayKey
from ._record import RunRecord

_FINAL_EPOCH_SHARE = 0.80


def training_documents(split: str, count: int) -> np.ndarray | None:
    """The documents of the training split, or None when the split gives it none."""
    try:
        shares = [float(part) for part in re.findall(r"[.0-9]+", split)]
    except ValueError:
        return None
    shares += [0.0] * (3 - len(shares))
    if len(shares) != 3:
        return None
    weights = np.array(shares, dtype=np.float64)
    total = np.sum(weights)
    if not total > 0:
        return None
    upper = 0 + (weights / total).tolist()[0]
    if upper <= 0:
        return None
    return np.arange(int(round(0 * float(count))), int(round(upper * float(count))), dtype=np.int32)


def _document_index(documents: np.ndarray, epochs: int, state: np.random.RandomState,
                    final_epoch_apart: bool) -> np.ndarray:
    if final_epoch_apart and epochs > 1:
        return np.concatenate((_document_index(documents, epochs - 1, state, False),
                               _document_index(documents, 1, state, False)))
    index = np.tile(documents, epochs).astype(np.int32)
    state.shuffle(index)
    return index


def _sample_index(lengths: np.ndarray, document_index: np.ndarray, sequence_length: int,
                  samples: int, extra: int) -> np.ndarray:
    largest = max(int(document_index.shape[0]), int(lengths.max()))
    dtype = np.int32 if largest <= np.iinfo(np.int32).max else np.int64
    index = np.zeros((samples + 1, 2), dtype=dtype)
    if samples == 0:
        return index
    ends = np.cumsum(lengths[document_index].astype(np.int64))
    # The token that closes sample i - 1: the shared token with the extra token
    # on, the last token of the sample with it off.
    tokens = np.arange(1, samples + 1, dtype=np.int64) * sequence_length - 1 + extra
    places = np.searchsorted(ends, tokens, side="right")
    starts = ends[places] - lengths[document_index[places]]
    index[1:, 0] = places
    index[1:, 1] = tokens - starts + (1 - extra)
    return index


def _shuffle_index(first: int, total: int, state: np.random.RandomState) -> np.ndarray:
    dtype = np.uint32 if total < np.iinfo(np.uint32).max - 1 else np.int64
    index = np.arange(0, first, dtype=dtype)
    state.shuffle(index)
    if first == total:
        return index
    last = np.arange(first, total, dtype=dtype)
    state.shuffle(last)
    return np.concatenate((index, last))


def replay_shard(lengths: np.ndarray, *, sequence_length: int, extra: int, seed: int, split: str,
                 requested: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three arrays of one shard. ``requested`` is None for a shard of a blend
    and the requested number of samples for a shard that is the whole dataset."""
    if sequence_length < 2:
        raise ReplayNotPossible("the sequence length is below 2")
    documents = training_documents(split, int(lengths.shape[0]))
    if documents is None or documents.size == 0:
        raise ReplayNotPossible(f"the split {split!r} gives the training split no document")
    epoch_tokens = int(np.sum(lengths[documents], dtype=np.int64))
    if epoch_tokens < 2:
        raise ReplayNotPossible("the training split holds fewer than two tokens")

    epochs = 1
    if requested is not None:
        while epochs * epoch_tokens < requested * sequence_length + extra:
            epochs += 1
    apart = False
    before_final = 0
    if epochs > 1:
        assert requested is not None
        before_final = ((epochs - 1) * epoch_tokens - extra) // sequence_length
        from_final = requested - before_final
        per_epoch = (epoch_tokens - extra) // sequence_length
        if not 0 <= from_final <= per_epoch + 1:
            raise ReplayNotPossible("the requested samples do not fit the epochs they need")
        apart = from_final < int(_FINAL_EPOCH_SHARE * per_epoch)

    state = np.random.RandomState(seed)
    document_index = _document_index(documents, epochs, state, apart)
    samples = (epochs * epoch_tokens - extra) // sequence_length
    sample_index = _sample_index(lengths, document_index, sequence_length, samples, extra)
    shuffle_index = _shuffle_index(before_final if apart else samples, samples, state)
    return document_index, sample_index, shuffle_index


def replay_blend(sample_counts: Sequence[int], requested: int) -> tuple[np.ndarray, np.ndarray]:
    """The two arrays of a blend with no weights given."""
    total = sum(sample_counts)
    if total <= 0:
        raise ReplayNotPossible("the shards hold no sample")
    size = min(requested, total)
    weights = (np.array(sample_counts, dtype=np.float64) / np.sum(
        np.array(sample_counts, dtype=np.float64))).tolist()
    dataset_index = np.zeros(size, dtype=np.int16)
    dataset_sample_index = np.zeros(size, dtype=np.int64)
    drawn = [0] * len(sample_counts)
    for place in range(size):
        scale = max(float(place), 1.0)
        best, behind = 0, weights[0] * scale - float(drawn[0])
        for shard in range(1, len(sample_counts)):
            lag = weights[shard] * scale - float(drawn[shard])
            if lag > behind:
                best, behind = shard, lag
        dataset_index[place] = best
        dataset_sample_index[place] = drawn[best]
        drawn[best] += 1
    if any(count > held for count, held in zip(drawn, sample_counts)):
        raise ReplayNotPossible("the blend asks a shard for more samples than it holds")
    return dataset_index, dataset_sample_index


def missing_replay_settings(record: RunRecord) -> list[str]:
    """The settings a replay needs that the run record does not hold, by name: the
    sequence length, the random seed, the split, the add-extra-token setting, and
    the train iterations or train samples with the global batch size. Empty for a
    record a replay can start from. This is the one check of what a run record holds
    for a replay: ``replay_arrays`` refuses by it, and the report and the
    verification of a rebuild call it to say whether a run can be replayed
    (docs/interfaces.md 7.3)."""
    needed = {
        "sequence_length": record.sequence_length,
        "random_seed": record.random_seed,
        "split": record.split,
        "add_extra_token_to_sequence": record.add_extra_token,
        "train_iters or train_samples, with the global batch size": record.requested_samples,
    }
    return [name for name, value in needed.items() if value is None]


def replay_arrays(record: RunRecord, lengths: Sequence[np.ndarray]) -> dict[ArrayKey, np.ndarray]:
    """Every index array of a segment, derived again from its run record and the
    token counts of its shards' documents, in the order of the shard list.

    Raises ``ReplayNotPossible`` when a setting the derivation needs was not
    recorded. The caller compares each array's digest with the recorded one."""
    absent = missing_replay_settings(record)
    if absent:
        raise ReplayNotPossible("the run record does not hold " + ", ".join(absent))
    if not 0 <= record.random_seed < 2**32:
        raise ReplayNotPossible("the seed is outside the range of the generator")
    extra = 1 if record.add_extra_token else 0
    blend = len(lengths) >= 2
    arrays: dict[ArrayKey, np.ndarray] = {}
    counts = []
    for position, shard_lengths in enumerate(lengths):
        built = replay_shard(
            shard_lengths, sequence_length=record.sequence_length, extra=extra,
            seed=record.random_seed, split=record.split,
            requested=None if blend else record.requested_samples,
        )
        for kind, array in zip(v.CACHE_KINDS_PER_SHARD, built):
            arrays[(kind, position)] = array
        counts.append(int(built[1].shape[0]) - 1)
    if blend:
        dataset_index, dataset_sample_index = replay_blend(counts, record.requested_samples)
        arrays[(v.CACHE_DATASET_INDEX, None)] = dataset_index
        arrays[(v.CACHE_DATASET_SAMPLE_INDEX, None)] = dataset_sample_index
    return arrays
