"""No trace route and no trace rest on a run record that holds no launch, and no trace
is stored under a segment whose route is "none", whoever calls the store (plan 3.5, 2.4;
docs/coverage.md section 8).

(a) ``register_run_segment`` refuses a trace route other than "none" for a run record that
holds no launch: nothing says which step comes first, so no step of the segment is known
to be whole. (b) ``write_step_traces``, and ``register_run_segment`` given traces, refuse
any trace for a segment whose route is "none". Both refusals are "invalid record" and leave
nothing written.

The tables are counted before and after, as in ``test_register_run_segment_in_one_call``.
Every refusal has a control: the same call with a launch (a), or with a route (b), is
accepted. Nothing here imports Megatron or torch.
"""

from __future__ import annotations

import pytest

from waterwhite import errors
from waterwhite.records import StepTrace, TraceRow, TraceSample
from waterwhite.store import Store

CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingSampler"
JOIN = ("core_v0.20.0", "single", CLASS)
LAUNCH = {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0,
          "step_at_start": 0, "first_step": 1, "last_step": 2}
CACHE_FILES, REPLAY, NONE = "cache files", "replay", "none"
MISSING = "data-order trace missing"


def record(*, launches: list[dict]) -> dict:
    """A run record that holds every digest of one shard's three arrays and the settings of
    a verified sequential run; ``launches`` is the only thing the tests change."""
    return {
        "format": "waterwhite.run-record/1", "run": "H", "segment": "H/seg-1", "start_kind": "fresh",
        "parent_checkpoint": None,
        "settings": {"dataloader_type": "single", "sampler_class": CLASS, "data_sharding": True,
                     "micro_batch_size": 1, "global_batch_size": 1, "sequence_length": 4,
                     "add_extra_token_to_sequence": True, "random_seed": 1234, "split": "1,0,0",
                     "train_iters": 2, "train_samples": None, "batch_schedule": None},
        "versions": {"release_tag": JOIN[0], "megatron_core": "0.20.0", "megatron_bridge": None,
                     "torch": "2.0"},
        "shards": [],
        "launches": launches,
        "cache_digests": [{"kind": kind, "shard": 0, "digest": f"{n}" * 64}
                          for n, kind in enumerate(("document_index", "sample_index", "shuffle_index"), 1)],
        "shuffle_digests": [],
    }


def trace(step: int, ordinal: int) -> StepTrace:
    return StepTrace(step, 1, (TraceSample(0, step - 1, 0, 5, 5, (TraceRow(0, ordinal, 0, 5),)),))


def call(*, launches: list[dict], route: str, traces: list[StepTrace] | None = None) -> dict:
    """The arguments of one registration. A route of "none" is given its stored cause, and
    no traces unless a test says so; the other routes are given the traces of both steps."""
    args = dict(run="H", segment="H/seg-1", position=0, mixture_instance_id=None,
                parent_checkpoint=None, start_kind="fresh", run_record=record(launches=launches),
                steps=[(1, 1, None), (2, 1, None)], trace_route=route,
                traces=[trace(1, 0), trace(2, 1)] if traces is None and route != NONE
                else (traces or []),
                checkpoints=[("H@2", 2)])
    if route == NONE:
        args["trace_cause"] = MISSING
    return args


@pytest.fixture
def store(tmp_path):
    with Store.create(tmp_path / "s.db", normaliser="control-chars/1", verified_joins=[JOIN]) as made:
        yield made


def counts(store: Store) -> dict[str, int]:
    names = [r[0] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {n: store._conn.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}


def refusal(action) -> errors.WriteRefused | None:
    """The write refusal the call raises, or None when it is accepted. A test asserts on the
    answer, so a call that is accepted fails on an assertion."""
    try:
        action()
    except errors.WriteRefused as raised:
        return raised
    return None


def assert_invalid_record(raised: errors.WriteRefused | None, what: str) -> None:
    assert raised is not None, f"{what}: the store accepted it"
    assert isinstance(raised, errors.InvalidRecord) and raised.name == "invalid record", (
        f"{what}: refused as {type(raised).__name__}, {raised}")


# --- (a) A route needs a launch ---------------------------------------------------------------


@pytest.mark.parametrize("route", [CACHE_FILES, REPLAY])
def test_a_run_record_with_no_launch_cannot_be_given_a_trace_route(store, route):
    before = counts(store)
    raised = refusal(lambda: store.register_run_segment(**call(launches=[], route=route)))
    assert_invalid_record(raised, f"route {route!r}, no launch")
    assert store.segment("H/seg-1") is None and counts(store) == before


@pytest.mark.parametrize("route", [CACHE_FILES, REPLAY])
def test_control_the_same_record_with_a_launch_is_given_that_route(store, route):
    assert refusal(lambda: store.register_run_segment(**call(launches=[LAUNCH], route=route))) is None
    held = store.segment("H/seg-1")
    assert held.trace_route == route and len(held.launches) == 1
    assert store.step_trace("H/seg-1", 1) == trace(1, 0)


@pytest.mark.parametrize("launches", [[], [LAUNCH]])
def test_control_route_none_with_no_traces_is_accepted_with_or_without_a_launch(store, launches):
    assert refusal(lambda: store.register_run_segment(**call(launches=launches, route=NONE))) is None
    held = store.segment("H/seg-1")
    assert (held.trace_route, held.trace_cause) == (NONE, MISSING)
    assert list(store.step_traces("H/seg-1")) == []


# --- (b) A route of "none" holds no trace ------------------------------------------------------


@pytest.mark.parametrize("launches", [[LAUNCH], []], ids=["with a launch", "with no launch"])
def test_a_trace_written_under_a_segment_whose_route_is_none_is_refused(store, launches):
    store.register_run_segment(**call(launches=launches, route=NONE))
    before = counts(store)
    raised = refusal(lambda: store.write_step_traces("H/seg-1", [trace(1, 0)]))
    assert_invalid_record(raised, "write_step_traces under route none")
    assert counts(store) == before and list(store.step_traces("H/seg-1")) == []


@pytest.mark.parametrize("launches", [[LAUNCH], []], ids=["with a launch", "with no launch"])
def test_a_segment_registered_with_route_none_and_traces_is_refused(store, launches):
    before = counts(store)
    raised = refusal(lambda: store.register_run_segment(
        **call(launches=launches, route=NONE, traces=[trace(1, 0), trace(2, 1)])))
    assert_invalid_record(raised, "register_run_segment(route none, traces)")
    assert store.segment("H/seg-1") is None and counts(store) == before


def test_traces_given_again_to_a_segment_already_held_with_route_none_are_refused(store):
    # The second call finds the segment held, a different path of the store than the first.
    store.register_run_segment(**call(launches=[LAUNCH], route=NONE))
    before = counts(store)
    raised = refusal(lambda: store.register_run_segment(
        **call(launches=[LAUNCH], route=NONE, traces=[trace(1, 0), trace(2, 1)])))
    assert_invalid_record(raised, "the same segment again, with traces")
    assert counts(store) == before and list(store.step_traces("H/seg-1")) == []


def test_control_a_segment_that_has_a_route_takes_traces_written_later(store):
    store.register_run_segment(**call(launches=[LAUNCH], route=CACHE_FILES, traces=[]))
    assert refusal(lambda: store.write_step_traces("H/seg-1", [trace(1, 0), trace(2, 1)])) is None
    assert [store.step_trace("H/seg-1", n) == trace(n, n - 1) for n in (1, 2)] == [True, True]
