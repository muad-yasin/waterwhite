"""The epochs the steps of a registered cyclic segment read in (plan 3.5).

``show --trace`` prints, for a segment read with the cyclic sampler, the epochs each
step draws its samples from. This module places the steps of a stored segment the
way ``compute_join`` does when it walks them: a launch starts at its consumed-sample
counter, every step adds its batch size, and ``_sampler.cyclic_epochs`` says which
epochs the step's rounds lie in.

The dataset's length is not stored. The sizes of the cyclic map are taken from the
run record instead: the count of a launch's shuffle digests is the length of the
rank's whole order, which is the bucket, and an epoch holds ``bucket`` times the
data-parallel size samples (``_sampler.CyclicShape``). A launch whose digests give
no count has no epochs to print.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from ._record import RunRecord
from ._sampler import CannotPlace, CyclicShape, cyclic_epochs


def _shape_of(record: RunRecord, position: int, ranks: int) -> CyclicShape | None:
    counts = {count for (launch, _, _), (_, count) in record.shuffle_digests.items()
              if launch == position and count is not None}
    if len(counts) != 1:
        return None
    (bucket,) = counts
    if bucket < 1 or ranks < 1:
        return None
    return CyclicShape(bucket * ranks, bucket)


def step_epochs(record: RunRecord, batch_sizes: Mapping[int, int],
                steps: Collection[int] | None = None) -> dict[int, list[int] | None]:
    """The epochs of the steps of a cyclic segment, by step number.

    ``batch_sizes`` maps each registered step to its batch size. ``steps`` keeps
    some steps (the walk still starts at each launch's counter); None keeps all of
    them. A step of a launch whose sizes the run record does not give, or that the
    cyclic map does not cover (as ``compute_join`` reads it: a batch size other than
    the global one, a start inside a round) has None."""
    micro = record.micro_batch_size
    placed: dict[int, list[int] | None] = {}
    if micro is None:
        return placed
    for launch in record.launches:
        ranks = launch.data_parallel_size
        shape = _shape_of(record, launch.position, ranks)
        consumed = launch.consumed_at_start
        walking = shape is not None
        for step in range(launch.first_step, launch.last_step + 1):
            batch = batch_sizes.get(step)
            epochs = None
            if walking and batch is not None and batch == record.global_batch_size:
                try:
                    epochs = cyclic_epochs(consumed, batch, micro, ranks, shape)
                except CannotPlace:
                    walking = False
            else:
                walking = False
            if steps is None or step in steps:
                placed[step] = epochs
            if epochs is not None:
                consumed += batch
    return placed
