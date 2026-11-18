"""Where the join writes no trace, and which cause it names (plan 3.5, 2.4, F6;
Check 2 of plan 5.5).

Every test starts from a run that joins cleanly (the hand world of ``join_kit``
with its cache folder, or the golden segment) and takes one thing away or changes
one thing. A result with a cause has the route "none" and no trace.
"""

from __future__ import annotations

import sys

import join_kit as kit
import pytest
from join_kit import cyclic_record, golden_record, stand_in_orders, torch_orders

from waterwhite import vocabulary as v
from waterwhite.errors import InvalidRecord, TorchUnavailable
from waterwhite.join import compute_join

OTHER_CLASS = "megatron.bridge.data.samplers.MegatronPretrainingSampler"


def join(tmp_path, record, *, cache: bool = True, joins=kit.JOINS, arrays=kit.ARRAYS, **more):
    if cache:
        kit.write_cache(tmp_path / "cache", arrays)
    return compute_join(record, verified_joins=joins, record_dir=tmp_path,
                        cache_dir=tmp_path / "cache" if cache else None, **more)


def refused(result, cause: str, detail: str | None = None) -> bool:
    assert (result.route, result.cause, result.cause_detail) == (v.ROUTE_NONE, cause, detail), (
        result.explanation)
    assert result.traces == () and result.steps_written == 0 and result.explanation
    return True


def test_the_untouched_run_joins(tmp_path):
    result = join(tmp_path, kit.run_record(tmp_path))
    assert (result.route, result.cause, result.steps_written) == (v.ROUTE_CACHE_FILES, None, 3)


# --- The gate (plan F6, 3.5) -----------------------------------------------------------


def test_another_release_tag_gets_no_trace(tmp_path):
    record = kit.run_record(tmp_path)
    record["versions"]["release_tag"] = "core_v0.19.0"
    result = join(tmp_path, record)
    assert refused(result, v.NAMES_TRACE_UNVERIFIED_VERSION)
    # The completed steps are still handed on, so the segment registers them.
    assert result.steps == ((1, 2, None), (2, 2, None), (3, 2, None))


def test_another_release_tag_names_the_version_whatever_else_is_missing(tmp_path):
    # A release the join is not verified for comes before every other question:
    # here the run also recorded no digest and no shuffle.
    orders = stand_in_orders({0: [0, 1]})
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 4)], orders)
    record["versions"]["release_tag"] = "core_v0.19.0"
    record["cache_digests"] = []
    record["shuffle_digests"] = []
    assert refused(join(tmp_path, record, cache=False), v.NAMES_TRACE_UNVERIFIED_VERSION)


def test_a_run_record_that_states_no_release_gets_no_trace(tmp_path):
    record = kit.run_record(tmp_path)
    record["versions"] = {}
    assert refused(join(tmp_path, record), v.NAMES_TRACE_UNVERIFIED_VERSION)


def test_a_run_record_with_no_release_and_no_digest_names_the_missing_digest(tmp_path):
    # Nothing was recorded that a join could be held against. No verification of
    # a release can change that, so the missing digest is named.
    record = kit.run_record(tmp_path)
    record["versions"] = {}
    record["cache_digests"] = []
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)


def test_a_store_with_no_verified_join_gets_no_trace(tmp_path):
    assert refused(join(tmp_path, kit.run_record(tmp_path), joins=()),
                   v.NAMES_TRACE_UNVERIFIED_VERSION)


def test_a_sampler_class_the_release_is_not_verified_with_gets_no_trace(tmp_path):
    record = kit.run_record(tmp_path)
    record["settings"]["sampler_class"] = OTHER_CLASS
    assert refused(join(tmp_path, record), v.NAMES_TRACE_UNVERIFIED_SAMPLER)


def test_a_sampler_type_the_release_is_not_verified_with_gets_no_trace(tmp_path):
    # The store lists the release with the sequential sampler only.
    orders = stand_in_orders({0: [0, 1]})
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 4)], orders)
    assert refused(join(tmp_path, record, joins=kit.JOINS[:1]), v.NAMES_TRACE_UNVERIFIED_SAMPLER)


def test_the_cyclic_sampler_without_data_sharding_gets_no_trace(tmp_path):
    # Data sharding off picks the other of the sampler's two rules (plan 3.5,
    # input 4). The join is verified for sharding on.
    orders = stand_in_orders({0: [0, 1]})
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 4)], orders)
    record["settings"]["data_sharding"] = False
    assert refused(join(tmp_path, record), v.NAMES_TRACE_UNVERIFIED_SAMPLER)


# --- No recorded digest (plan 3.5; Check 2 (iii)) -----------------------------------------


def test_a_run_with_no_run_record_gets_no_trace_and_no_replay(tmp_path):
    step_record = {"steps": [{"step": 1, "batch_size": 2}, {"step": 2, "batch_size": 2}],
                   "checkpoints": [{"name": "H@2", "step": 2}]}
    result = compute_join(None, verified_joins=kit.JOINS, step_record=step_record)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert (result.replay.attempted, result.replay.accepted) == (False, False)
    assert result.steps == ((1, 2, None), (2, 2, None)) and result.checkpoints == (("H@2", 2),)


def test_a_run_record_with_no_cache_digest_gets_no_replay_even_with_its_settings(tmp_path):
    # Plan 3.5: a replay with no recorded digest supports no answer, so none is
    # tried, although the settings and the seed are all there.
    record = golden_record(tmp_path)
    record["cache_digests"] = []
    result = join(tmp_path, record, cache=False)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert not result.replay.attempted


def test_settings_and_digests_withheld_give_the_same_cause(tmp_path):
    # Check 2 (iii): the cache files, the settings and the recorded digests are
    # withheld. The sampler type is a setting, so it is gone as well.
    record = kit.run_record(tmp_path)
    record["settings"] = {}
    record["cache_digests"] = []
    result = join(tmp_path, record, cache=False)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert not result.replay.attempted
    assert result.steps == () and result.unsized == (1, 2, 3)


def test_one_missing_digest_is_no_recorded_digest(tmp_path):
    record = kit.run_record(tmp_path)
    record["cache_digests"] = [row for row in record["cache_digests"]
                               if (row["kind"], row["shard"]) != ("shuffle_index", 1)]
    result = join(tmp_path, record)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert "shuffle_index of shard 1" in result.explanation


# --- Digest mismatch (plan 3.5; Check 2 (ii)) ---------------------------------------------


def test_a_replay_from_a_changed_seed_is_rejected(tmp_path):
    # Check 2 (ii): cache files withheld, the seed in the settings changed. The
    # replayed arrays of both shards differ from the recorded digests; the blend's
    # two do not, since no seed enters them.
    result = join(tmp_path, golden_record(tmp_path, random_seed=579), cache=False)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)
    assert (result.replay.attempted, result.replay.accepted) == (True, False)
    assert set(result.replay.differing) == {
        (kind, shard) for shard in (0, 1) for kind in v.CACHE_KINDS_PER_SHARD}


def test_a_changed_cache_file_and_no_matching_replay_is_a_digest_mismatch(tmp_path):
    # The hand world's arrays are written by hand: no seed gives them, so the
    # replay cannot stand in when one cache file no longer has its digest.
    changed = dict(kit.ARRAYS)
    changed[("shuffle_index", 0)] = kit.ARRAYS[("shuffle_index", 0)][::-1].copy()
    result = join(tmp_path, kit.run_record(tmp_path), arrays=changed)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)
    assert (result.replay.attempted, result.replay.accepted) == (True, False)


def test_a_cache_file_is_tied_to_its_shard_by_its_digest_and_not_by_its_name(tmp_path):
    # The files of the two shards carry each other's names: the name of a cache
    # file is a hash of paths and settings and says nothing the join relies on.
    # Each array is found under the digest recorded for its kind and shard.
    renamed = dict(kit.ARRAYS)
    for kind in v.CACHE_KINDS_PER_SHARD:
        renamed[(kind, 0)], renamed[(kind, 1)] = kit.ARRAYS[(kind, 1)], kit.ARRAYS[(kind, 0)]
    result = join(tmp_path, kit.run_record(tmp_path), arrays=renamed)
    assert result.route == v.ROUTE_CACHE_FILES
    assert kit.rows(result.traces[0].samples[0]) == kit.WORLD[0]
    assert kit.rows(result.traces[0].samples[1]) == kit.WORLD[1]


def test_a_shard_index_file_that_is_not_the_one_the_trainer_read(tmp_path):
    record = kit.run_record(tmp_path)
    kit.write_shard(tmp_path / "shards" / "shard-1", [5, 6])
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)


def test_a_shard_index_file_with_no_recorded_digest(tmp_path):
    record = kit.run_record(tmp_path)
    del record["shards"][0]["idx_sha256"]
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)


# --- A run the join cannot read (plan 3.5, last fallback) -----------------------------------


def test_a_shard_index_file_that_is_gone(tmp_path):
    record = kit.run_record(tmp_path)
    record["shards"][1]["prefix"] = "shards/moved-away"
    result = join(tmp_path, record)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_SHARD_INDEX_UNREADABLE)
    assert "shard 1" in result.explanation


@pytest.mark.parametrize("field", ["micro_batch_size", "global_batch_size", "sequence_length",
                                   "add_extra_token_to_sequence"])
def test_a_setting_the_walk_needs_is_never_guessed(tmp_path, field):
    record = kit.run_record(tmp_path)
    record["settings"][field] = None
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_RECORD_INCOMPLETE)


def test_a_run_record_with_no_launch(tmp_path):
    record = kit.run_record(tmp_path, launches=[])
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_RECORD_INCOMPLETE)


def test_a_run_read_with_the_extra_token_off(tmp_path):
    record = kit.run_record(tmp_path, add_extra_token_to_sequence=False)
    assert refused(join(tmp_path, record), v.NAMES_TRACE_MISSING, v.TRACE_MISSING_EXTRA_TOKEN_OFF)


def test_no_cache_file_and_no_setting_to_replay_from(tmp_path):
    # Cache files and a setting of the replay both gone: nothing a replay could
    # be held against. docs/interfaces.md section 2 gives this the reason "no
    # recorded digest".
    record = kit.run_record(tmp_path)
    record["settings"]["random_seed"] = None
    result = join(tmp_path, record, cache=False)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST)
    assert not result.replay.attempted and "random_seed" in result.replay.why_not


def test_cache_files_that_differ_and_no_setting_to_replay_from(tmp_path):
    changed = dict(kit.ARRAYS)
    changed[("shuffle_index", 0)] = kit.ARRAYS[("shuffle_index", 0)][::-1].copy()
    record = kit.run_record(tmp_path)
    record["settings"]["random_seed"] = None
    result = join(tmp_path, record, arrays=changed)
    assert refused(result, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)
    assert not result.replay.attempted


# --- The order of a cyclic run (plan 3.5, the ruling; Check 2 (iv)) -------------------------
# The join computes each epoch's order by calling torch and accepts it only when its
# digest and count equal the recorded ones. The tests that reach that comparison draw
# the recorded orders with their own call to torch (``torch_orders``) and so need
# torch; the others stop before it, and a test below shows that they never load it.


def cyclic(tmp_path, orders=None, last_step: int = 4):
    orders = stand_in_orders({0: [0, 1]}) if orders is None else orders
    return cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, last_step)], orders)


@pytest.fixture
def no_torch(monkeypatch):
    """torch cannot be imported: ``import torch`` raises ImportError."""
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.mark.megatron
def test_the_cyclic_run_joins_when_its_computed_orders_match_the_recorded_digests(tmp_path):
    record = cyclic(tmp_path, torch_orders({0: [0, 1]}))
    assert join(tmp_path, record).steps_written == 4


def test_cyclic_with_no_shuffle_digest_whatever_cache_files_were_kept(tmp_path, no_torch):
    # Check 2 (iv): the cache files are all there and match; the shuffle is in
    # none of them. The join does not need torch to say so.
    record = cyclic(tmp_path)
    record["shuffle_digests"] = []
    assert refused(join(tmp_path, record), v.NAMES_CYCLIC_NOT_RECORDED)


def test_cyclic_with_no_shuffle_digest_and_no_cache_digest_names_the_cyclic_cause(tmp_path, no_torch):
    record = cyclic(tmp_path)
    record["shuffle_digests"] = []
    record["cache_digests"] = []
    assert refused(join(tmp_path, record), v.NAMES_CYCLIC_NOT_RECORDED)


def test_cyclic_with_no_digest_for_an_epoch_a_launch_reads_in(tmp_path, no_torch):
    # Step 4 reads in epoch 1, and the run record holds epoch 0 only. The join
    # asks for every digest before it draws anything, so it needs no torch to say so.
    record = cyclic(tmp_path, stand_in_orders({0: [0]}))
    result = join(tmp_path, record)
    assert refused(result, v.NAMES_CYCLIC_NOT_RECORDED)
    assert "epoch 1" in result.explanation


@pytest.mark.megatron
def test_the_same_record_joins_when_the_launch_stays_inside_the_recorded_epoch(tmp_path):
    record = cyclic(tmp_path, torch_orders({0: [0]}), last_step=3)
    assert join(tmp_path, record).steps_written == 3


@pytest.mark.megatron
def test_cyclic_with_a_wrong_digest(tmp_path):
    # The recorded order of epoch 1, rank 1 is not the order of the sampler: two of
    # its indices are swapped before the digest is taken.
    orders = torch_orders({0: [0, 1]})
    place = orders[(0, 1, 1)]
    orders[(0, 1, 1)] = [place[1], place[0], place[2]]
    result = join(tmp_path, cyclic(tmp_path, orders))
    assert refused(result, v.NAMES_CYCLIC_DIGEST_MISMATCH)
    assert "epoch 1, rank 1" in result.explanation


@pytest.mark.megatron
def test_cyclic_with_a_changed_recorded_digest(tmp_path):
    record = cyclic(tmp_path, torch_orders({0: [0, 1]}))
    record["shuffle_digests"][0]["digest"] = "0" * 64
    assert refused(join(tmp_path, record), v.NAMES_CYCLIC_DIGEST_MISMATCH)


@pytest.mark.megatron
def test_cyclic_with_a_changed_recorded_count(tmp_path):
    record = cyclic(tmp_path, torch_orders({0: [0, 1]}))
    record["shuffle_digests"][2]["count"] = 4
    assert refused(join(tmp_path, record), v.NAMES_CYCLIC_DIGEST_MISMATCH)


@pytest.mark.megatron
def test_cyclic_order_recorded_for_another_ranks_bucket(tmp_path):
    # Rank 1 owns indices 3, 4, 5. Rank 0's order recorded for it, with its own
    # true digest, is not the order the sizes of the run record give rank 1.
    orders = torch_orders({0: [0, 1]})
    orders[(0, 0, 1)] = orders[(0, 0, 0)]
    assert refused(join(tmp_path, cyclic(tmp_path, orders)), v.NAMES_CYCLIC_DIGEST_MISMATCH)


@pytest.mark.megatron
def test_cyclic_orders_that_differ_by_the_epoch_seed_are_a_mismatch(tmp_path):
    # An order drawn with the run's seed (or any seed but the epoch number) is
    # not the sampler's: epoch 0 recorded from a generator seeded 1.
    torch = pytest.importorskip("torch")
    orders = torch_orders({0: [0, 1]})
    own = orders[(0, 0, 0)]  # rank 0 owns the first indices, so this is the bare permutation

    def drawn(seed: int) -> list[int]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(kit.BUCKET, generator=generator).tolist()

    place = next(drawn(seed) for seed in range(1, 40) if drawn(seed) != own)
    for rank in range(kit.RANKS):
        orders[(0, 0, rank)] = [rank * kit.BUCKET + p for p in place]
    assert refused(join(tmp_path, cyclic(tmp_path, orders)), v.NAMES_CYCLIC_DIGEST_MISMATCH)


def test_a_cyclic_run_whose_shuffle_must_be_computed_needs_torch(tmp_path, no_torch):
    # Every digest is recorded, so the join must draw the shuffle, and torch is
    # not there: no answer is made up, the join raises its named error.
    with pytest.raises(TorchUnavailable, match="torch"):
        join(tmp_path, cyclic(tmp_path))


# --- Records that are not records ------------------------------------------------------------


def test_a_document_that_is_not_a_run_record_is_refused(tmp_path):
    record = kit.run_record(tmp_path)
    for change in ({"format": "something/1"}, {"launches": [{"position": 0}]},
                   {"shards": [{"position": 3}]}):
        with pytest.raises(InvalidRecord):
            join(tmp_path, {**record, **change})
    record["settings"]["micro_batch_size"] = "two"
    with pytest.raises(InvalidRecord):
        join(tmp_path, record)


def test_two_launches_that_share_a_step_are_refused(tmp_path):
    record = kit.run_record(tmp_path, launches=[kit.launch(0, 1, 0, 1, 2), kit.launch(1, 1, 2, 2, 3)])
    with pytest.raises(InvalidRecord, match="repeats a step"):
        join(tmp_path, record)


def test_a_step_record_of_another_segment_is_refused(tmp_path):
    with pytest.raises(InvalidRecord, match="written for segment"):
        join(tmp_path, kit.run_record(tmp_path), step_record={"segment": "H/seg-2", "steps": []})
