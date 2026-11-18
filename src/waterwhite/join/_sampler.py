"""The step-to-sample map, per sampler type (plan 3.5, F2, F4).

A step's samples are counted from the consumed-sample counter of its launch and
never from its step number: a resume carries the counter on, a warm start
restarts it at 0, and either way the step number says nothing about the place in
the data (plan F4).

Both samplers deal micro batches to the data-parallel ranks. With ``n`` ranks and
a micro batch of ``m`` samples, one round is ``m * n`` samples and a step of
batch size ``b`` is ``b / (m * n)`` rounds. The place of a sample inside its step
is ``round * m * n + rank * m + place in the micro batch`` (docs/interfaces.md
8.3).

- Sequential (``dataloader_type`` "single"): a step reads the next ``b`` indices
  in a row from the counter. Inside a round, rank ``r`` holds the indices at
  places ``r * m`` to ``(r + 1) * m``, so the place of a sample in its step is
  its distance from the step's first index.
- Cyclic (``dataloader_type`` "cyclic") with data sharding: the dataset of ``N``
  samples loses its last ``N mod (m * n)`` samples, the rest is one epoch, and
  each rank owns ``(N div (m * n)) * m`` consecutive indices of it, its bucket.
  The counter gives the epoch (counter div epoch length) and the place inside
  it; a rank at place ``p`` of the epoch has read ``p / n`` indices of its
  bucket's order. The order of a bucket in an epoch is a shuffle the trainer's
  library draws from a generator seeded with the epoch number. This module does
  not draw it: it is handed each epoch's order for each rank, taken from the
  run's own record and already held against the recorded digest (plan 3.5, the
  ruling on the cyclic order; DECISIONS.md D-16).

The cyclic sampler without data sharding follows another rule. The join is not
verified for it and does not walk it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Draw:
    """One sample of a step: its place in the step, the index asked of the top
    dataset, and the rank whose sampler drew it."""

    sample_pos: int
    global_index: int
    rank: int


class CannotPlace(Exception):
    """The map does not cover the step. The text says why."""


def rounds_of(batch: int, micro: int, ranks: int) -> int:
    """How many micro-batch rounds make a step of this batch size."""
    if micro < 1 or ranks < 1:
        raise CannotPlace("the micro batch size and the data-parallel size are at least 1")
    if batch < 1 or batch % (micro * ranks):
        raise CannotPlace(
            f"a batch of {batch} samples is not a whole number of rounds of {micro} samples "
            f"on {ranks} ranks"
        )
    return batch // (micro * ranks)


def check_sequential(consumed: int, batch: int, micro: int, ranks: int, length: int) -> None:
    """Raise ``CannotPlace`` unless the sequential map covers the step."""
    rounds_of(batch, micro, ranks)
    if consumed + batch > length:
        raise CannotPlace(
            f"the step would read samples {consumed} to {consumed + batch - 1} and the dataset "
            f"holds {length}; the sampler's pass ends before the step is whole, and plan 3.5 "
            "does not say what a new pass reads"
        )


def sequential_step(consumed: int, batch: int, micro: int, ranks: int, length: int) -> list[Draw]:
    """The samples of one step of the sequential sampler."""
    check_sequential(consumed, batch, micro, ranks, length)
    return [Draw(place, consumed + place, (place % (micro * ranks)) // micro)
            for place in range(batch)]


@dataclass(frozen=True)
class CyclicShape:
    """The sizes of a cyclic run with data sharding on one launch."""

    epoch_samples: int  # the samples all ranks read in one epoch
    bucket: int  # the indices one rank reads in one epoch

    @classmethod
    def of(cls, length: int, micro: int, ranks: int) -> "CyclicShape":
        if micro < 1 or ranks < 1:
            raise CannotPlace("the micro batch size and the data-parallel size are at least 1")
        round_size = micro * ranks
        epoch = length - length % round_size
        if epoch <= 0:
            raise CannotPlace(
                f"a dataset of {length} samples holds no whole round of {round_size} samples"
            )
        return cls(epoch, (length // round_size) * micro)

    def bucket_range(self, rank: int) -> range:
        """The indices a rank owns."""
        return range(rank * self.bucket, (rank + 1) * self.bucket)


def _round_start(at: int, micro: int, ranks: int, shape: CyclicShape) -> tuple[int, int]:
    """The epoch a round starts in and how many indices each rank has read of it."""
    epoch, inside = divmod(at, shape.epoch_samples)
    if inside % (micro * ranks):
        raise CannotPlace(
            f"the counter {at} lies {inside} samples into epoch {epoch}, which is not a whole "
            f"number of rounds of {micro * ranks}; the sampler refuses such a start"
        )
    return epoch, inside // ranks


def cyclic_epochs(consumed: int, batch: int, micro: int, ranks: int, shape: CyclicShape) -> list[int]:
    """The epochs one step reads in, in order. A step whose rounds lie on both
    sides of an epoch boundary reads in two. Raises ``CannotPlace`` unless the
    cyclic map covers the step."""
    epochs: list[int] = []
    for number in range(rounds_of(batch, micro, ranks)):
        epoch, _ = _round_start(consumed + number * micro * ranks, micro, ranks, shape)
        if epoch not in epochs:
            epochs.append(epoch)
    return epochs


def cyclic_step(consumed: int, batch: int, micro: int, ranks: int, shape: CyclicShape,
                order_of: Callable[[int, int], Sequence[int]]) -> list[Draw]:
    """The samples of one step of the cyclic sampler with data sharding.

    ``order_of(epoch, rank)`` gives the whole order of an epoch for a rank."""
    round_size = micro * ranks
    draws = []
    for number in range(rounds_of(batch, micro, ranks)):
        epoch, start = _round_start(consumed + number * round_size, micro, ranks, shape)
        for rank in range(ranks):
            order = order_of(epoch, rank)
            for place in range(micro):
                draws.append(Draw(number * round_size + rank * micro + place,
                                  int(order[start + place]), rank))
    return draws
