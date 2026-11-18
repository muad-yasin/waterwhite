"""One store call registers a segment with its step traces and its checkpoints
(plan 3.4, 3.5, Decision 19), and a segment read with the cyclic sampler whose run
record holds no shuffle digest has no trace route (plan 2.4, 3.5; DECISIONS.md D-26).

Nothing here imports Megatron or torch.
"""

from __future__ import annotations

import pytest

from waterwhite import errors
from waterwhite import vocabulary as v
from waterwhite.records import StepTrace, TraceRow, TraceSample
from waterwhite.store import Store

CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
JOIN = ("core_v0.20.0", "single", CLASS)


def record(sampler: str = "single", shuffle: list | None = None) -> dict:
    return {
        "format": "waterwhite.run-record/1", "run": "H", "segment": "H/seg-1", "start_kind": "fresh",
        "parent_checkpoint": None,
        "settings": {"dataloader_type": sampler, "sampler_class": CLASS, "data_sharding": True,
                     "micro_batch_size": 1, "global_batch_size": 1, "sequence_length": 4,
                     "add_extra_token_to_sequence": True, "random_seed": 1234, "split": "1,0,0",
                     "train_iters": 2, "train_samples": None, "batch_schedule": None},
        "versions": {"release_tag": JOIN[0], "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.0"},
        "shards": [],
        "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
                      "consumed_at_start": 0, "step_at_start": 0, "first_step": 1, "last_step": 2}],
        # The three arrays of one shard, which a route of "cache files" needs (plan 3.5).
        "cache_digests": [{"kind": kind, "shard": 0, "digest": f"{n}" * 64}
                          for n, kind in enumerate(("document_index", "sample_index", "shuffle_index"), 1)],
        "shuffle_digests": shuffle or [],
    }


def trace(step: int, ordinal: int) -> StepTrace:
    return StepTrace(step, 1, (TraceSample(0, step - 1, 0, 5, 5, (TraceRow(0, ordinal, 0, 5),)),))


def call(**changes) -> dict:
    args = dict(run="H", segment="H/seg-1", position=0, mixture_instance_id=None,
                parent_checkpoint=None, start_kind="fresh", run_record=record(),
                steps=[(1, 1, None), (2, 1, None)], trace_route=v.ROUTE_CACHE_FILES,
                traces=[trace(1, 0), trace(2, 1)], checkpoints=[("H@2", 2)])
    args.update(changes)
    return args


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser="control-chars/1", verified_joins=[JOIN]) as made:
        yield made


def counts(store: Store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def test_the_segment_its_traces_and_its_checkpoints_are_written_by_one_call(store):
    store.register_run_segment(**call())
    held = store.segment("H/seg-1")
    assert [s.step for s in held.steps] == [1, 2]
    assert [store.step_trace("H/seg-1", n) == trace(n, n - 1) for n in (1, 2)] == [True, True]
    assert [c.names for c in store.checkpoints_of("H/seg-1")] == [("H@2",)]


def test_a_trace_that_does_not_fit_leaves_nothing_of_the_segment_written(store):
    # Step 2's trace counts two samples and the step's batch size is 1: the store
    # refuses it, and the segment and the checkpoint registered with it are not there.
    before = counts(store)
    bad = StepTrace(2, 2, (trace(2, 1).samples[0], TraceSample(1, 9, 0, 5, 5, ())))
    with pytest.raises(errors.WriteRefused):
        store.register_run_segment(**call(traces=[trace(1, 0), bad]))
    assert store.segment("H/seg-1") is None and counts(store) == before


def test_a_checkpoint_at_a_step_the_segment_does_not_have_leaves_nothing_written(store):
    before = counts(store)
    with pytest.raises(errors.WriteRefused):
        store.register_run_segment(**call(checkpoints=[("H@7", 7)]))
    assert store.segment("H/seg-1") is None and counts(store) == before


def test_the_same_call_again_changes_nothing_and_another_trace_is_refused(store):
    store.register_run_segment(**call())
    before = counts(store)
    store.register_run_segment(**call())
    assert counts(store) == before
    with pytest.raises(errors.WriteRefused):
        store.register_run_segment(**call(traces=[trace(1, 2), trace(2, 1)]))
    assert counts(store) == before


def test_a_call_with_no_traces_and_no_checkpoints_is_what_it_was(store):
    # The arguments are new and optional: a caller of the earlier signature is unchanged.
    args = call()
    del args["traces"], args["checkpoints"]
    store.register_run_segment(**args)
    assert list(store.step_traces("H/seg-1")) == [] and store.checkpoints_of("H/seg-1") == ()


# --- A cyclic run with no shuffle digest has no trace route (plan 3.5; D-26) ------------------


def test_a_cyclic_segment_with_no_shuffle_digest_cannot_be_given_a_trace_route(store):
    before = counts(store)
    for route in (v.ROUTE_CACHE_FILES, v.ROUTE_REPLAY):
        with pytest.raises(errors.InvalidRecord, match="no shuffle digest"):
            store.register_run_segment(**call(run_record=record("cyclic"), trace_route=route))
    assert store.segment("H/seg-1") is None and counts(store) == before


def test_a_cyclic_segment_with_no_shuffle_digest_is_registered_with_its_cause(store):
    store.register_run_segment(**call(
        run_record=record("cyclic"), trace_route=v.ROUTE_NONE, trace_cause=v.NAMES_CYCLIC_NOT_RECORDED,
        traces=[], checkpoints=[]))
    held = store.segment("H/seg-1")
    assert (held.trace_route, held.trace_cause) == (v.ROUTE_NONE, v.NAMES_CYCLIC_NOT_RECORDED)


def test_a_cyclic_segment_with_a_shuffle_digest_may_have_a_route(store):
    digest = [{"launch": 0, "epoch": 0, "rank": 0, "digest": "0" * 64, "count": 1}]
    store.register_run_segment(**call(run_record=record("cyclic", digest), traces=[], checkpoints=[]))
    assert store.segment("H/seg-1").trace_route == v.ROUTE_CACHE_FILES
