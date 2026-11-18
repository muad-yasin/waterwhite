"""The replay: index arrays derived again with numpy alone (plan 3.5, F3; Check 2).

Two kinds of expected value. The sample index and the blend are derived by hand
(``join_kit``, and the comments here). The shuffles cannot be worked out by hand,
so they are held against arrays the trainer's own code built, kept as numbers in
``golden_megatron``. The comparison with the trainer's code itself, on every
segment of the fixture, is ``test_join_megatron.py``.
"""

from __future__ import annotations

import golden_megatron as golden
import join_kit as kit
import numpy as np
import pytest
from join_kit import golden_record

from waterwhite import vocabulary as v
from waterwhite.errors import ReplayNotPossible
from waterwhite.join import compute_join, parse_run_record, replay_arrays
from waterwhite.join._replay import _sample_index, replay_blend, replay_shard, training_documents


def lengths_of(world) -> list[np.ndarray]:
    return [np.array(x, dtype=np.int32) for x in world]


def test_the_replay_gives_the_arrays_the_trainers_code_built(tmp_path):
    replayed = replay_arrays(parse_run_record(golden_record(tmp_path)), lengths_of(golden.LENGTHS))
    assert set(replayed) == set(golden.ARRAYS)
    for key, expected in golden.ARRAYS.items():
        assert replayed[key].dtype == expected.dtype, key
        assert replayed[key].shape == expected.shape, key
        assert replayed[key].tobytes() == expected.tobytes(), key


def test_another_seed_changes_each_shards_arrays_and_leaves_the_blend(tmp_path):
    # The blend uses no random number: it follows from the shards' sample counts.
    record = parse_run_record(golden_record(tmp_path, random_seed=579))
    replayed = replay_arrays(record, lengths_of(golden.LENGTHS))
    for (kind, shard), expected in golden.ARRAYS.items():
        same = replayed[(kind, shard)].tobytes() == expected.tobytes()
        assert same == (shard is None), (kind, shard)


def test_the_sample_index_marks_the_hand_derived_starts():
    # join_kit derives both sample indexes by hand from its document indexes.
    for shard in (0, 1):
        built = _sample_index(np.array(kit.LENGTHS[shard], dtype=np.int32),
                              kit.ARRAYS[("document_index", shard)], kit.SEQUENCE_LENGTH,
                              len(kit.ARRAYS[("shuffle_index", shard)]), 1)
        expected = kit.ARRAYS[("sample_index", shard)]
        assert built.dtype == expected.dtype and built.tolist() == expected.tolist()


def test_the_blend_goes_to_the_shard_furthest_behind_its_share():
    # Shards of 4 and 2 samples: shares 2/3 and 1/3. At place i (scale max(i, 1))
    # each shard's lag is share * scale - drawn so far; the larger lag wins, the
    # first shard on a tie.
    #   i=0: 0.67, 0.33          -> shard 0 (its sample 0)
    #   i=1: 0.67-1, 0.33        -> shard 1 (0)
    #   i=2: 1.33-1, 0.67-1      -> shard 0 (1)
    #   i=3: 2-2, 1-1 (a tie)    -> shard 0 (2)
    #   i=4: 2.67-3, 1.33-1      -> shard 1 (1)
    #   i=5: 3.33-3, 1.67-2      -> shard 0 (3)
    # These are the arrays join_kit writes by hand for its blend.
    dataset_index, dataset_sample_index = replay_blend([4, 2], 6)
    assert dataset_index.tolist() == kit.ARRAYS[("dataset_index", None)].tolist()
    assert dataset_sample_index.tolist() == kit.ARRAYS[("dataset_sample_index", None)].tolist()
    assert (dataset_index.dtype, dataset_sample_index.dtype) == (np.int16, np.int64)
    # Asked for more than the shards hold, the blend is as long as what they hold;
    # asked for less, it is as long as what was asked.
    assert replay_blend([4, 2], 100)[0].shape == (6,)
    assert replay_blend([4, 2], 4)[0].tolist() == [0, 1, 0, 0]


def test_a_shard_of_a_blend_is_built_for_one_epoch():
    lengths = np.array(kit.LENGTHS[0], dtype=np.int32)
    document, sample, shuffle = replay_shard(lengths, sequence_length=4, extra=1, seed=7,
                                             split="1,0,0", requested=None)
    assert sorted(document.tolist()) == [0, 1, 2, 3] and document.dtype == np.int32
    assert sample.shape == (5, 2) and sorted(shuffle.tolist()) == [0, 1, 2, 3]
    assert shuffle.dtype == np.uint32


def test_a_shard_that_is_the_whole_dataset_is_built_for_the_epochs_asked_for():
    # 18 tokens per epoch. 10 samples of 4 tokens plus the extra token need 41
    # tokens: three epochs, 54 tokens, (54 - 1) div 4 = 13 samples.
    lengths = np.array(kit.LENGTHS[0], dtype=np.int32)
    document, sample, shuffle = replay_shard(lengths, sequence_length=4, extra=1, seed=7,
                                             split="1,0,0", requested=10)
    assert sorted(document.tolist()) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert sample.shape == (14, 2) and sorted(shuffle.tolist()) == list(range(13))
    # Two epochs give (36 - 1) div 4 = 8 samples, so the last epoch supplies
    # 10 - 8 = 2 of the 10. An epoch alone holds (18 - 1) div 4 = 4, and 2 is
    # below int(0.80 * 4) = 3: the last epoch is shuffled apart. The first 8
    # documents are then two whole epochs and the first 8 sample numbers stay
    # among themselves.
    assert sorted(document[:8].tolist()) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sorted(shuffle[:8].tolist()) == list(range(8))


def test_the_training_split_takes_the_first_share_of_the_documents():
    assert training_documents("1,0,0", 5).tolist() == [0, 1, 2, 3, 4]
    assert training_documents("3,1,0", 8).tolist() == [0, 1, 2, 3, 4, 5]
    assert training_documents("0,1,0", 8) is None


def test_a_replay_without_its_settings_is_not_possible(tmp_path):
    record = golden_record(tmp_path)
    del record["settings"]["random_seed"]
    with pytest.raises(ReplayNotPossible, match="random_seed"):
        replay_arrays(parse_run_record(record), lengths_of(golden.LENGTHS))


def test_with_the_cache_files_withheld_the_join_replays_and_reads_the_same_rows(tmp_path):
    # Check 2 (i) in small: no cache folder, settings and seed kept. The replay's
    # digests equal the recorded ones, the route is "replay", and the rows are the
    # ones the oracle logged for this segment.
    result = compute_join(golden_record(tmp_path), verified_joins=kit.JOINS, record_dir=tmp_path)
    assert (result.route, result.cause) == (v.ROUTE_REPLAY, None)
    assert (result.replay.attempted, result.replay.accepted) == (True, True)
    assert result.dataset_length == 24 and result.steps_written == 6
    mine = {(t.step, s.sample_pos, s.global_index, *row)
            for t in result.traces for s in t.samples for row in kit.rows(s)}
    assert set(golden.LOG_LINES) <= mine
    assert all(s.covered_tokens == 33 for t in result.traces for s in t.samples)


def test_the_cache_files_come_first_when_they_match(tmp_path):
    kit.write_cache(tmp_path / "cache", golden.ARRAYS)
    result = compute_join(golden_record(tmp_path), verified_joins=kit.JOINS, record_dir=tmp_path,
                          cache_dir=tmp_path / "cache")
    assert result.route == v.ROUTE_CACHE_FILES and not result.replay.attempted


def test_cache_files_that_do_not_match_give_way_to_a_replay_that_does(tmp_path):
    # A cache folder rebuilt later with another seed at the same place: its files
    # do not have the recorded digests. The replay from the recorded settings
    # does, so it counts, and the route says "replay".
    stale = dict(golden.ARRAYS)
    stale[("shuffle_index", 0)] = golden.ARRAYS[("shuffle_index", 0)][::-1].copy()
    kit.write_cache(tmp_path / "cache", stale)
    result = compute_join(golden_record(tmp_path), verified_joins=kit.JOINS, record_dir=tmp_path,
                          cache_dir=tmp_path / "cache")
    assert (result.route, result.replay.accepted) == (v.ROUTE_REPLAY, True)
