"""The join's result in the store (plan 3.5, Decision 19, 2.2(e), 3.4).

``join_segment`` reads the files, joins, registers the segment with the route,
the cause and the completed steps, and writes the traces once. The last tests ask
the tool's own answer for a checkpoint of a segment that got no trace, to hold
the cause the join stored against the word the answer names.
"""

from __future__ import annotations

import join_kit as kit
import pytest
from join_kit import cyclic_record, golden_record, stand_in_orders, torch_orders

from waterwhite import capture, coverage, requests
from waterwhite import vocabulary as v
from waterwhite.errors import InvalidRecord, UsageError
from waterwhite.join import SegmentRegistration, join_segment
from waterwhite.store import Store

SEGMENT = SegmentRegistration(run="H", segment="H/seg-1", position=0, start_kind="fresh")
STEP_RECORD = {"segment": "H/seg-1",
               "steps": [{"step": 2, "batch_size": 2, "last_sample_tokens_read": 4}],
               "checkpoints": [{"name": "H@2", "step": 2}]}


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "store.db", normaliser="control-chars/1",
                      verified_joins=kit.JOINS) as made:
        yield made


def files(tmp_path, record: dict, *, cache: bool = True):
    """Write a run's files as a wrapper leaves them: the run record, the cache
    folder and the step record."""
    if cache:
        kit.write_cache(tmp_path / "cache", kit.ARRAYS)
    return (kit.write_json(tmp_path / "run_record.json", record),
            tmp_path / "cache" if cache else None,
            kit.write_json(tmp_path / "step_record.json", STEP_RECORD))


def test_join_segment_registers_the_segment_and_writes_its_traces(tmp_path, store):
    record_path, cache_dir, step_path = files(tmp_path, kit.run_record(tmp_path))
    result = join_segment(store, SEGMENT, record_path, cache_dir, step_path)
    assert (result.route, result.cause, result.steps_written) == (v.ROUTE_CACHE_FILES, None, 3)

    held = store.segment("H/seg-1")
    assert (held.trace_route, held.trace_cause, held.trace_cause_detail) == (
        v.ROUTE_CACHE_FILES, None, None)
    assert [(s.step, s.batch_size, s.last_sample_tokens_read) for s in held.steps] == [
        (1, 2, None), (2, 2, 4), (3, 2, None)]
    # The checkpoint of the step record is registered by the same call.
    mark = store.checkpoint("H@2")
    assert (mark.segment, mark.step) == ("H/seg-1", 2)
    # What the store reads back is what the join computed, step by step.
    for trace in result.traces:
        assert store.step_trace("H/seg-1", trace.step) == trace
    # Step 2 is lowered: its last sample covers 4 of 5 tokens and is still counted.
    last = store.step_trace("H/seg-1", 2).samples[-1]
    assert (last.covered_tokens, last.sample_tokens, last.uncovered) == (4, 5, (4, 5))
    assert kit.rows(last) == [(0, 3, 2, 5), (0, 1, 0, 1)]
    states = {state.step: state for state in store.trace_states("H/seg-1")}
    assert [states[step].traced for step in (1, 2, 3)] == [True, True, True]
    assert [states[step].uncovered_samples for step in (1, 2, 3)] == [0, 1, 0]
    # A lowered mark is an untraced step, not a count fault (plan 3.4).
    store.check_integrity()
    for name, step in result.checkpoints:
        store.register_checkpoint(name=name, segment="H/seg-1", step=step)


def test_the_same_join_again_changes_nothing(tmp_path, store):
    record_path, cache_dir, step_path = files(tmp_path, kit.run_record(tmp_path))
    first = join_segment(store, SEGMENT, record_path, cache_dir, step_path)
    again = join_segment(store, SEGMENT, record_path, cache_dir, step_path)
    assert again == first
    assert sum(1 for _ in store.step_traces("H/seg-1")) == 3


def test_a_segment_is_joined_once_and_another_join_of_it_is_refused(tmp_path, store):
    # The trace is written at registration and not again (Decision 19): a second
    # registration with other evidence does not replace the first.
    record_path, cache_dir, step_path = files(tmp_path, kit.run_record(tmp_path))
    join_segment(store, SEGMENT, record_path, cache_dir, step_path)
    with pytest.raises(InvalidRecord):
        join_segment(store, SEGMENT, record_path, cache_dir, None)


def test_a_run_record_file_that_cannot_be_read_registers_nothing(tmp_path, store):
    with pytest.raises(UsageError, match="cannot be read"):
        join_segment(store, SEGMENT, tmp_path / "no-such-file.json")
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(InvalidRecord, match="not JSON"):
        join_segment(store, SEGMENT, tmp_path / "bad.json")
    assert store.segment("H/seg-1") is None


def test_a_run_record_of_another_segment_registers_nothing(tmp_path, store):
    record_path, cache_dir, _ = files(tmp_path, kit.run_record(tmp_path))
    other = SegmentRegistration(run="H", segment="H/seg-9", position=0, start_kind="fresh")
    with pytest.raises(InvalidRecord):
        join_segment(store, other, record_path, cache_dir)
    assert store.segment("H/seg-9") is None and store.segment("H/seg-1") is None


# --- A segment with no trace, and what a checkpoint of it answers ---------------------------


def unverified(tmp_path):
    record = kit.run_record(tmp_path)
    record["versions"]["release_tag"] = "core_v0.19.0"
    return files(tmp_path, record), {}


def no_run_record(tmp_path):
    _, _, step_path = files(tmp_path, kit.run_record(tmp_path), cache=False)
    return (None, None, step_path), {}


def changed_seed(tmp_path):
    return files(tmp_path, golden_record(tmp_path, random_seed=579), cache=False), {}


def shard_index_gone(tmp_path):
    record = kit.run_record(tmp_path)
    record["shards"][0]["prefix"] = "shards/moved-away"
    return files(tmp_path, record), {}


def no_shuffle_digest(tmp_path):
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 3)], stand_in_orders({0: [0]}))
    record["shuffle_digests"] = []
    return files(tmp_path, record), {}


def wrong_shuffle_digest(tmp_path):
    record = cyclic_record(tmp_path, [kit.launch(0, 2, 0, 1, 3)], torch_orders({0: [0]}))
    record["shuffle_digests"][1]["digest"] = "f" * 64
    return files(tmp_path, record), {}


CASES = [
    (unverified, v.NAMES_TRACE_UNVERIFIED_VERSION, None),
    (no_run_record, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST),
    (changed_seed, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH),
    # A run the join cannot read: the plan's two reasons do not fit, so the cause
    # is stored with the reason the tool adds for it (docs/join.md, 11).
    (shard_index_gone, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_SHARD_INDEX_UNREADABLE),
    (no_shuffle_digest, v.NAMES_CYCLIC_NOT_RECORDED, None),
    pytest.param(wrong_shuffle_digest, v.NAMES_CYCLIC_DIGEST_MISMATCH, None,
                 marks=pytest.mark.megatron),
]


@pytest.mark.parametrize("case, cause, detail", CASES, ids=["unverified", "no_run_record", "changed_seed", "shard_index_gone",
                                      "no_shuffle_digest", "wrong_shuffle_digest"])
def test_a_segment_with_no_trace_is_registered_with_its_cause(tmp_path, store, case, cause, detail):
    (record_path, cache_dir, step_path), more = case(tmp_path)
    result = join_segment(store, SEGMENT, record_path, cache_dir, step_path, **more)
    assert (result.route, result.cause, result.cause_detail) == (v.ROUTE_NONE, cause, detail)
    held = store.segment("H/seg-1")
    assert (held.trace_route, held.trace_cause, held.trace_cause_detail) == (v.ROUTE_NONE, cause, detail)
    assert list(store.step_traces("H/seg-1")) == []
    assert 2 in [s.step for s in held.steps]

    # The checkpoint's own answer names that cause and never "does not contain".
    (tmp_path / "A.jsonl").write_text('{"text": "one"}\n', encoding="utf-8")
    capture.ingest_jsonl(store, tmp_path / "A.jsonl", source="A", member="A/r1", revision="r1")
    store.register_checkpoint(name="H@2", segment="H/seg-1", step=2)
    answer = coverage.answer(store, "H@2", requests.withdraw(store, "A")).to_json()
    assert (answer["answer"], answer["names"], answer["names_detail"]) == (v.MAY_CONTAIN, cause, detail)
