"""The step-to-sample map (plan 3.5, F2, F4; docs/interfaces.md 8.3).

Each expected list is worked out in the comment above it. The samplers are
checked here on their own; ``test_join_walk.py`` checks them inside a join.
"""

from __future__ import annotations

import pytest

from waterwhite.join._sampler import CannotPlace, CyclicShape, cyclic_step, sequential_step


def places(draws) -> list[tuple[int, int, int]]:
    return [(d.sample_pos, d.global_index, d.rank) for d in draws]


def test_sequential_one_rank_reads_the_next_indices_in_a_row():
    # Counter 8, batch 4, one rank with a micro batch of 4: indices 8 to 11.
    assert places(sequential_step(8, 4, 4, 1, 100)) == [
        (0, 8, 0), (1, 9, 0), (2, 10, 0), (3, 11, 0)]


def test_sequential_two_ranks_deal_each_round_in_rank_order():
    # Micro batch 2 on 2 ranks: a round is 4 indices, rank 0 holds the first two
    # and rank 1 the last two. Batch 8 is two rounds: 20..23, then 24..27.
    assert places(sequential_step(20, 8, 2, 2, 100)) == [
        (0, 20, 0), (1, 21, 0), (2, 22, 1), (3, 23, 1),
        (4, 24, 0), (5, 25, 0), (6, 26, 1), (7, 27, 1)]


def test_sequential_a_step_that_ends_at_the_dataset_end_is_read_and_one_past_it_is_not():
    assert len(sequential_step(96, 4, 4, 1, 100)) == 4
    with pytest.raises(CannotPlace):
        sequential_step(97, 4, 4, 1, 100)


def test_a_batch_that_is_no_whole_number_of_rounds_is_not_placed():
    with pytest.raises(CannotPlace):
        sequential_step(0, 6, 2, 2, 100)
    with pytest.raises(CannotPlace):
        sequential_step(0, 0, 2, 2, 100)


def test_cyclic_shape_drops_the_last_part_round_and_gives_each_rank_a_bucket():
    # 57 samples, micro batch 2 on 2 ranks: a round is 4, 57 mod 4 = 1 sample is in
    # no epoch, an epoch is 56 samples, and each rank owns (57 div 4) * 2 = 28.
    shape = CyclicShape.of(57, 2, 2)
    assert (shape.epoch_samples, shape.bucket) == (56, 28)
    assert shape.bucket_range(0) == range(0, 28) and shape.bucket_range(1) == range(28, 56)
    with pytest.raises(CannotPlace):
        CyclicShape.of(3, 2, 2)


# A dataset of 9 samples, micro batch 2 on 2 ranks: a round is 4, sample 8 is in
# no epoch, an epoch is 8 samples, rank 0 owns 0..3 and rank 1 owns 4..7. The
# orders below are written by hand.
ORDERS = {
    (0, 0): [3, 1, 0, 2], (0, 1): [6, 4, 7, 5],
    (1, 0): [0, 3, 2, 1], (1, 1): [5, 7, 4, 6],
}
SHAPE = CyclicShape.of(9, 2, 2)


def order_of(epoch: int, rank: int) -> list[int]:
    return ORDERS[(epoch, rank)]


def test_cyclic_a_step_takes_each_ranks_next_micro_batch_of_its_order():
    # Counter 0: epoch 0, nothing read. Rank 0 draws 3, 1 and rank 1 draws 6, 4.
    # sample_pos counts the rank first: rank 0 holds 0 and 1, rank 1 holds 2 and 3.
    assert places(cyclic_step(0, 4, 2, 2, SHAPE, order_of)) == [
        (0, 3, 0), (1, 1, 0), (2, 6, 1), (3, 4, 1)]


def test_cyclic_a_resume_inside_an_epoch_goes_on_in_the_same_order():
    # Counter 4: epoch 0, 4 samples in, so each rank has read 4 / 2 = 2 indices of
    # its order. Rank 0 draws 0, 2 and rank 1 draws 7, 5. A launch that starts
    # here reads what a launch from 0 reads at its second step.
    assert places(cyclic_step(4, 4, 2, 2, SHAPE, order_of)) == [
        (0, 0, 0), (1, 2, 0), (2, 7, 1), (3, 5, 1)]


def test_cyclic_the_counter_past_an_epoch_reads_the_next_epochs_order():
    # Counter 8 = one epoch: epoch 1, nothing read of it.
    assert places(cyclic_step(8, 4, 2, 2, SHAPE, order_of)) == [
        (0, 0, 0), (1, 3, 0), (2, 5, 1), (3, 7, 1)]


def test_cyclic_a_step_of_two_rounds_can_cross_the_epoch_boundary():
    # Batch 8 is two rounds. From counter 4: round 0 is the rest of epoch 0, round
    # 1 is the start of epoch 1. sample_pos: round, then rank, then place.
    assert places(cyclic_step(4, 8, 2, 2, SHAPE, order_of)) == [
        (0, 0, 0), (1, 2, 0), (2, 7, 1), (3, 5, 1),
        (4, 0, 0), (5, 3, 0), (6, 5, 1), (7, 7, 1)]


def test_cyclic_a_counter_inside_a_round_is_refused_as_the_sampler_refuses_it():
    with pytest.raises(CannotPlace):
        cyclic_step(2, 4, 2, 2, SHAPE, order_of)
