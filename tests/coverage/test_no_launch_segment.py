"""A segment whose run record holds no launch supports no "contains" and no "does not
contain" (plan 2.2(e), 3.1, 3.5; docs/coverage.md section 8).

The step-to-sample map is counted from the consumed-sample counter of a launch (plan
3.5, F4): with no launch recorded there is no counter, so what a step read is not known
whatever else the record holds. The store therefore refuses such a segment any trace route
but "none" and refuses any trace under it, and a segment with route "none" answers as every
such segment does: its checkpoints answer "may contain" naming the cause and the reason the
segment holds. Through ``register-run`` that cause is the join's, taken in the order of its
questions (docs/join.md section 3): a release the store does not list, a cyclic run with no
shuffle digest, a digest not recorded, a sampler not verified, and otherwise, since no launch
is held, "data-order trace missing" with the reason "run record incomplete".

The hand world of ``ww_world`` supplies the sources, the release and the mixture. On the
direct store call the run record holds a shard list and every index-array digest, a verified
release and sampler class, so that the launch list is the only thing missing; the control
keeps one launch and nothing else different. Through ``register-run`` the same run record is
given with a step record and the mixture, and no cache folder; there the digests are
stand-ins, and the test says only what the command printed and what the checkpoint answers.
Source Z has one record in the mixture's shard that the traces below do not read; source C
has no record in it at all.

Nothing here imports Megatron or torch.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import ww_world

from waterwhite import cli, errors
from waterwhite.records import StepTrace

RUN = "N"
SEGMENT = "N/seg-1"
CHECKPOINT = "N@2"
DIGESTS = {"document_index": "11" * 32, "sample_index": "22" * 32, "shuffle_index": "33" * 32}
RELEASE = ww_world.JOIN[0]
CYCLIC_CLASS = "megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler"

# The words of the plan (3.5, 2.4, F6), written out here and not read from the tool.
MISSING = "data-order trace missing"
NO_DIGEST = "no recorded digest"
UNVERIFIED_VERSION = "data-order trace unverified for this version"
UNVERIFIED_SAMPLER = "data-order trace unverified for this sampler type"
CYCLIC_NOT_RECORDED = "cyclic order not recorded"

LAUNCH = {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0,
          "step_at_start": 0, "first_step": 1, "last_step": 2}


def run_record(*, launches: list[dict], digests: bool = True, **changes) -> dict:
    """The run record of the segment: one shard in its shard list, the digests of its three
    index arrays (unless ``digests`` is False), no shuffle digest. ``changes`` are the
    arguments of ``ww_world.run_record`` (the sampler, the release, the sampler class)."""
    record = ww_world.run_record(RUN, SEGMENT, "fresh", 1, 2, **changes)
    record["shards"] = [{"position": 0, "prefix": "shards/shard-1", "bin_sha256": "b" * 64,
                         "idx_sha256": "c" * 64}]
    record["cache_digests"] = ([{"kind": kind, "shard": 0, "digest": digest}
                                for kind, digest in DIGESTS.items()] if digests else [])
    record["shuffle_digests"] = []
    record["launches"] = launches
    return record


def traces() -> list[StepTrace]:
    """Step 1 reads document 0 of the shard (R2's "passage X") and step 2 document 1
    ("b two"). Document 2, Z's "z three", is read by neither."""
    return [StepTrace(1, 1, (ww_world.sample(0, 0, 0, 0),)),
            StepTrace(2, 1, (ww_world.sample(0, 1, 0, 1),))]


def register_directly(world, *, launches: list[dict], route: str = "cache files",
                      cause: str | None = None, detail: str | None = None) -> None:
    """A direct store call. A segment of route "none" is given its cause and no traces."""
    world.store.register_run_segment(
        run=RUN, segment=SEGMENT, position=0, mixture_instance_id=world["mix-1"],
        parent_checkpoint=None, start_kind="fresh", run_record=run_record(launches=launches),
        steps=[(1, 1, None), (2, 1, None)], trace_route=route, trace_cause=cause,
        trace_cause_detail=detail, traces=traces() if route != "none" else [],
        checkpoints=[(CHECKPOINT, 2)])


def refusal(action) -> errors.WriteRefused | None:
    """The write refusal the call raises, or None when it is accepted."""
    try:
        action()
    except errors.WriteRefused as raised:
        return raised
    return None


def register_run(world, tmp_path: Path, *, launches: list[dict], digests: bool = True,
                 **changes) -> dict:
    """The same segment through ``waterwhite register-run``: the run record and the step
    record are files, the store is a file. Returns what the command printed."""
    run_file, steps_file = tmp_path / "n.run.json", tmp_path / "n.steps.json"
    run_file.write_text(json.dumps(run_record(launches=launches, digests=digests, **changes)),
                        encoding="utf-8")
    steps_file.write_text(json.dumps({
        "segment": SEGMENT,
        "steps": [{"step": 1, "batch_size": 1}, {"step": 2, "batch_size": 1}],
        "checkpoints": [{"name": CHECKPOINT, "step": 2}],
    }), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["register-run", "--store", str(world.path), "--run-record", str(run_file),
                     "--steps", str(steps_file), "--mixture", "mix-1"], stdout=out, stderr=err)
    assert code == 0, f"exit {code}, standard error: {err.getvalue()!r}"
    return json.loads(out.getvalue())


def list_cyclic_join_as_verified(world) -> None:
    """The store lists the cyclic sampler of the release as verified, so that a cyclic run
    is held up by nothing but its own record."""
    world.sql(("INSERT INTO verified_join (release_tag, sampler_type, sampler_class) VALUES (?, ?, ?)",
               (RELEASE, "cyclic", CYCLIC_CLASS)))


def stored_cause(printed: dict) -> tuple:
    return printed["route"], printed["cause"], printed["cause_detail"]


def named(answer: dict) -> tuple:
    return answer["answer"], answer["names"], answer["names_detail"]


# --- A direct store call -------------------------------------------------------------------


@pytest.mark.parametrize("source", ["Z", "C"])
def test_control_with_its_launch_the_same_segment_does_not_contain_what_its_traces_do_not_read(
        world, ask, source):
    register_directly(world, launches=[LAUNCH])
    answer = ask(source, CHECKPOINT)
    assert (answer["answer"], answer["names"], answer["untraced_steps"]) == ("does not contain", None, [])


def test_a_segment_with_no_launch_is_refused_a_trace_route_on_a_direct_call(world):
    # The route "cache files", with the traces of both steps: nothing of the segment is
    # written, and so no checkpoint of it exists.
    before = world.store.segments()
    raised = refusal(lambda: register_directly(world, launches=[]))
    assert raised is not None, "the store accepted a trace route for a run record with no launch"
    assert isinstance(raised, errors.InvalidRecord) and raised.name == "invalid record"
    assert world.store.segment(SEGMENT) is None and world.store.checkpoint(CHECKPOINT) is None
    assert world.store.segments() == before


@pytest.mark.parametrize("source", ["Z", "C"])
@pytest.mark.parametrize("cause, detail", [
    (MISSING, NO_DIGEST), (MISSING, None), (UNVERIFIED_VERSION, None), (CYCLIC_NOT_RECORDED, None)])
def test_a_segment_with_no_launch_and_route_none_answers_with_its_stored_cause_and_reason(
        world, ask, source, cause, detail):
    # The segment's own cause and reason are what the answer names; "step untraced" is not.
    register_directly(world, launches=[], route="none", cause=cause, detail=detail)
    assert named(ask(source, CHECKPOINT)) == ("may contain", cause, detail)


# --- Through register-run ------------------------------------------------------------------


@pytest.mark.parametrize("source", ["A", "B", "C", "Z"])
def test_a_plain_run_record_with_no_launch_is_stored_as_missing_run_record_incomplete(
        world, ask, tmp_path, source):
    printed = register_run(world, tmp_path, launches=[])
    assert stored_cause(printed) == ("none", MISSING, "run record incomplete")
    assert named(ask(source, CHECKPOINT)) == ("may contain", MISSING, "run record incomplete")


@pytest.mark.parametrize("source", ["Z", "C"])
def test_a_cyclic_run_record_with_no_launch_and_no_shuffle_digest_is_cyclic_not_recorded(
        world, ask, tmp_path, monkeypatch, source):
    monkeypatch.setitem(sys.modules, "torch", None)  # torch cannot be imported
    list_cyclic_join_as_verified(world)
    printed = register_run(world, tmp_path, launches=[], sampler="cyclic", sampler_class=CYCLIC_CLASS)
    assert stored_cause(printed) == ("none", CYCLIC_NOT_RECORDED, None)
    assert named(ask(source, CHECKPOINT)) == ("may contain", CYCLIC_NOT_RECORDED, None)


@pytest.mark.parametrize("source", ["Z", "C"])
def test_a_run_record_with_no_launch_whose_release_the_store_does_not_list_is_unverified_for_the_version(
        world, ask, tmp_path, source):
    printed = register_run(world, tmp_path, launches=[], release="core_v9.9.9")
    assert stored_cause(printed) == ("none", UNVERIFIED_VERSION, None)
    assert named(ask(source, CHECKPOINT)) == ("may contain", UNVERIFIED_VERSION, None)


# The rest of the order of the join's questions (docs/join.md section 3): each of these records
# holds no launch, and the first question that applies decides.


def test_an_unlisted_release_is_asked_before_the_cyclic_order(world, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    printed = register_run(world, tmp_path, launches=[], sampler="cyclic", sampler_class=CYCLIC_CLASS,
                           release="core_v9.9.9")
    assert stored_cause(printed) == ("none", UNVERIFIED_VERSION, None)


def test_a_digest_not_recorded_is_stored_with_its_reason_though_no_launch_is_held(world, ask, tmp_path):
    printed = register_run(world, tmp_path, launches=[], digests=False)
    assert stored_cause(printed) == ("none", MISSING, NO_DIGEST)
    assert named(ask("Z", CHECKPOINT)) == ("may contain", MISSING, NO_DIGEST)


def test_a_sampler_not_verified_is_stored_as_unverified_for_the_sampler_type(world, ask, tmp_path):
    # The release is listed; a sequential sampler of another class than the verified one is not.
    printed = register_run(world, tmp_path, launches=[], sampler_class="some.other.Sampler")
    assert stored_cause(printed) == ("none", UNVERIFIED_SAMPLER, None)
    answer = ask("Z", CHECKPOINT)
    assert (answer["answer"], answer["names"]) == ("may contain", UNVERIFIED_SAMPLER)
