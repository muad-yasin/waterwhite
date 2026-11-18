"""The join of the hand world, end to end without a store (plan 3.5; Check 1).

The index arrays and the shards are written by hand (``join_kit``). The orders of
the cyclic tests are drawn there by the test's own call to torch, and the expected
places are derived from them by arithmetic written in the test. Every expected
sample and row is derived in ``join_kit`` or in the comment above the assertion. The cache folder is read back from disk, so the cache-files route
runs in every test here.
"""

from __future__ import annotations

import join_kit as kit
import pytest
from join_kit import cyclic_places, cyclic_record, torch_orders

from waterwhite import vocabulary as v
from waterwhite.join import compute_join


def join(tmp_path, record, **more):
    kit.write_cache(tmp_path / "cache", kit.ARRAYS)
    return compute_join(record, verified_joins=kit.JOINS, record_dir=tmp_path,
                        cache_dir=tmp_path / "cache", **more)


def sample_rows(result) -> dict[tuple[int, int], list]:
    return {(trace.step, s.sample_pos): kit.rows(s) for trace in result.traces for s in trace.samples}


def test_sequential_run_reads_the_blend_in_a_row(tmp_path):
    # Batch 2, three steps from counter 0: step 1 reads g0, g1; step 2 g2, g3;
    # step 3 g4, g5. The rows of each index are derived in join_kit.
    result = join(tmp_path, kit.run_record(tmp_path))
    assert (result.route, result.cause, result.dataset_length) == (v.ROUTE_CACHE_FILES, None, 6)
    assert result.steps == ((1, 2, None), (2, 2, None), (3, 2, None))
    assert kit.placed(result) == {1: [(0, 0, 0), (1, 1, 0)], 2: [(0, 2, 0), (1, 3, 0)],
                                  3: [(0, 4, 0), (1, 5, 0)]}
    assert sample_rows(result) == {(1, 0): kit.WORLD[0], (1, 1): kit.WORLD[1],
                                   (2, 0): kit.WORLD[2], (2, 1): kit.WORLD[3],
                                   (3, 0): kit.WORLD[4], (3, 1): kit.WORLD[5]}
    for trace in result.traces:
        assert trace.consumed_samples == 2
        for sample in trace.samples:
            assert (sample.sample_tokens, sample.covered_tokens, sample.uncovered) == (5, 5, None)
    assert result.untraced == () and result.steps_written == 3


def test_a_segment_of_one_shard_is_read_without_a_blend(tmp_path):
    # Shard 0 alone: 4 samples, three arrays, no blend rows in the run record.
    record = kit.run_record(tmp_path, shards=1, launches=[kit.launch(0, 1, 0, 1, 2)])
    assert len(record["cache_digests"]) == 3
    result = join(tmp_path, record)
    assert (result.route, result.dataset_length) == (v.ROUTE_CACHE_FILES, 4)
    assert sample_rows(result) == {(1, 0): kit.WORLD_SHARD_0[0], (1, 1): kit.WORLD_SHARD_0[1],
                                   (2, 0): kit.WORLD_SHARD_0[2], (2, 1): kit.WORLD_SHARD_0[3]}


def test_two_ranks_and_two_rounds_per_step(tmp_path):
    # Micro batch 1 on 2 ranks, batch 4: one step is two rounds of 2. Round 0:
    # rank 0 g0, rank 1 g1. Round 1: rank 0 g2, rank 1 g3.
    record = kit.run_record(tmp_path, micro=1, batch=4, launches=[kit.launch(0, 2, 0, 1, 1)])
    result = join(tmp_path, record)
    assert kit.placed(result) == {1: [(0, 0, 0), (1, 1, 1), (2, 2, 0), (3, 3, 1)]}
    assert sample_rows(result)[(1, 3)] == kit.WORLD[3]


def test_a_resume_is_counted_from_its_counter_and_not_from_its_step_number(tmp_path):
    # The launch starts at counter 2 and calls its steps 7 and 8 (plan F4). Step 7
    # reads g2, g3, not the samples "step 7 times batch 2" would name.
    record = kit.run_record(tmp_path, launches=[kit.launch(0, 1, 2, 7, 8)])
    result = join(tmp_path, record)
    assert kit.placed(result) == {7: [(0, 2, 0), (1, 3, 0)], 8: [(0, 4, 0), (1, 5, 0)]}


def test_a_warm_start_restarts_the_counter(tmp_path):
    # A warm start from a checkpoint saved at 24 samples: the launch's counter is
    # 0 and the join reads from 0, whatever the parent's counter was.
    record = kit.run_record(tmp_path, launches=[kit.launch(0, 1, 0, 1, 1)])
    record.update(start_kind="warm-start", parent_checkpoint="P@6")
    assert kit.placed(join(tmp_path, record)) == {1: [(0, 0, 0), (1, 1, 0)]}


def test_two_launches_of_one_segment_each_start_at_their_own_counter(tmp_path):
    record = kit.run_record(tmp_path, launches=[kit.launch(0, 1, 0, 1, 1), kit.launch(1, 1, 2, 2, 3)])
    assert kit.placed(join(tmp_path, record)) == {
        1: [(0, 0, 0), (1, 1, 0)], 2: [(0, 2, 0), (1, 3, 0)], 3: [(0, 4, 0), (1, 5, 0)]}


def test_the_step_record_gives_a_batch_schedule_for_the_sequential_sampler(tmp_path):
    # The operator's schedule (plan F2): step 1 has 2 samples, step 2 has 4. Step
    # 2 starts at counter 2 and reads g2 to g5. Step 3 would start at counter 6,
    # the dataset's end: the map does not cover it and it gets no trace.
    step_record = {"segment": "H/seg-1", "steps": [{"step": 1, "batch_size": 2},
                                                  {"step": 2, "batch_size": 4}]}
    result = join(tmp_path, kit.run_record(tmp_path), step_record=step_record)
    assert result.steps == ((1, 2, None), (2, 4, None), (3, 2, None))
    assert kit.placed(result) == {1: [(0, 0, 0), (1, 1, 0)],
                                  2: [(0, 2, 0), (1, 3, 0), (2, 4, 0), (3, 5, 0)]}
    assert [step for step, _ in result.untraced] == [3]
    assert "the dataset holds 6" in result.untraced[0][1]


def test_a_step_no_launch_holds_is_registered_and_gets_no_trace(tmp_path):
    step_record = {"steps": [{"step": 9, "batch_size": 2}]}
    result = join(tmp_path, kit.run_record(tmp_path), step_record=step_record)
    assert (9, 2, None) in result.steps and result.steps_written == 3
    assert result.untraced == ((9, "no launch of the run record holds the step"),)


def test_a_lowered_step_gives_a_sample_with_an_uncovered_range(tmp_path):
    # The step record says 4 of the 5 tokens of step 2's last sample were read.
    # That sample is g3: D3[2:5], 3 tokens, then D1[0:2]. The first 4 tokens are
    # D3[2:5] and D1[0:1]; the uncovered range is token 4 to 5 of the sample. The
    # sample is still counted: the trace holds 2 samples.
    step_record = {"segment": "H/seg-1",
                   "steps": [{"step": 2, "batch_size": 2, "last_sample_tokens_read": 4}],
                   "checkpoints": [{"name": "H@2", "step": 2}]}
    result = join(tmp_path, kit.run_record(tmp_path), step_record=step_record)
    assert result.steps == ((1, 2, None), (2, 2, 4), (3, 2, None))
    trace = result.traces[1]
    assert trace.step == 2 and trace.consumed_samples == 2 and len(trace.samples) == 2
    first, last = trace.samples
    assert (kit.rows(first), first.covered_tokens) == (kit.WORLD[2], 5)
    assert kit.rows(last) == [(0, 3, 2, 5), (0, 1, 0, 1)]
    assert (last.sample_tokens, last.covered_tokens, last.uncovered) == (5, 4, (4, 5))
    assert result.lowered_steps == (2,) and result.checkpoints == (("H@2", 2),)


def test_a_note_that_names_more_tokens_than_a_sample_holds_raises_nothing(tmp_path):
    # Silence never raises a mark and neither does a large number (plan 3.5).
    step_record = {"steps": [{"step": 2, "last_sample_tokens_read": 99}]}
    result = join(tmp_path, kit.run_record(tmp_path), step_record=step_record)
    last = result.traces[1].samples[-1]
    assert (kit.rows(last), last.covered_tokens) == (kit.WORLD[3], 5)
    assert result.lowered_steps == ()


# --- The cyclic run ------------------------------------------------------------------
# join_kit sets it up: 6 samples, micro batch 1 on 2 ranks, an epoch of 3 steps of
# batch 2, rank 0 owning indices 0, 1, 2 and rank 1 owning 3, 4, 5. The orders are
# drawn by torch in the test (``torch_orders``), so these tests need torch.


@pytest.mark.megatron
def test_cyclic_run_follows_the_computed_order_across_an_epoch_boundary(tmp_path):
    # Steps 1 to 3 are epoch 0: each rank draws its order of epoch 0 place by
    # place. Step 4 starts at counter 6, one epoch, and draws the first place of
    # epoch 1's orders.
    orders = torch_orders({0: [0, 1]})
    result = join(tmp_path, cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 4)], orders))
    assert (result.route, result.cause) == (v.ROUTE_CACHE_FILES, None)
    expected = {step: cyclic_places(orders, 0, 2 * (step - 1)) for step in (1, 2, 3, 4)}
    assert kit.placed(result) == expected
    assert sample_rows(result)[(4, 0)] == kit.WORLD[expected[4][0][1]]
    assert sample_rows(result)[(4, 1)] == kit.WORLD[expected[4][1][1]]


@pytest.mark.megatron
def test_cyclic_resume_inside_an_epoch_reads_on_where_the_first_launch_stopped(tmp_path):
    # Launch 0 reads step 1. Launch 1 resumes at counter 2, inside epoch 0, and
    # reads steps 2 to 4. Each launch records the whole order of every epoch it
    # reads in (DECISIONS.md D-16), so epoch 0 is recorded twice with one digest.
    orders = torch_orders({0: [0], 1: [0, 1]})
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 1), kit.launch(1, 2, 2, 2, 4)], orders)
    assert record["shuffle_digests"][0]["digest"] == record["shuffle_digests"][2]["digest"]
    result = join(tmp_path, record)
    expected = {1: cyclic_places(orders, 0, 0)} | {
        step: cyclic_places(orders, 1, 2 * (step - 1)) for step in (2, 3, 4)}
    assert kit.placed(result) == expected


@pytest.mark.megatron
def test_cyclic_step_with_another_batch_size_than_the_global_one_gets_no_trace(tmp_path):
    # Plan 3.5 names no batch schedule for the cyclic sampler. From the step whose
    # size differs on, the launch's counter is unknown.
    orders = torch_orders({0: [0, 1]})
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 4)], orders)
    result = join(tmp_path, record, step_record={"steps": [{"step": 2, "batch_size": 4}]})
    assert sorted(kit.placed(result)) == [1]
    assert [step for step, _ in result.untraced] == [2, 3, 4]
