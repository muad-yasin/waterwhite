"""A checkpoint's four answers (plan 2.4, 2.2(e), 3.5, Decisions 12, 18, 20, 22).

The runs of ``ww_world``: E reads T at step 4 of its second segment and has an
operator note on step 5; D descends from E@6; Y branches from E@3, trains on the
mixture that holds T and never reads it.
"""

from __future__ import annotations

import pytest
import ww_world

from waterwhite import vocabulary as v
from waterwhite.records import StepTrace, TraceRow, TraceSample

T_CHAIN = ["shard-3:D0", "R1:L3", "gen/T:L1", "gen/P:L1", "src/A:L1"]


def hops(answer: dict) -> list[str]:
    return [hop["hop"] for hop in answer["chain"]]


def link(answer: dict, *keys: str) -> dict:
    return {key: answer["failing"].get(key) for key in keys}


def two_row_step(world, name: str, covered: int, note: int | None = None) -> None:
    """A fresh run of one step on mix-2 whose one sample reads three tokens of a
    record of shard-1 and then two tokens of T. ``covered`` is its coverage mark."""
    world.store.register_run_segment(
        run=name, segment=f"{name}/seg-1", position=0, mixture_instance_id=world["mix-2"],
        parent_checkpoint=None, start_kind="fresh",
        run_record=ww_world.run_record(name, f"{name}/seg-1", "fresh", 1, 1),
        steps=[(1, 1, note)], trace_route="cache files",
    )
    world.store.write_step_traces(f"{name}/seg-1", [StepTrace(1, 1, (
        TraceSample(0, 0, 0, 5, covered, (TraceRow(0, 0, 0, 3), TraceRow(1, 0, 0, 2))),
    ))])
    world.store.register_checkpoint(name=f"{name}@1", segment=f"{name}/seg-1", step=1)


# --- The four answers ------------------------------------------------------------------


def test_does_not_contain_is_scoped_to_the_steps_of_the_history(ask):
    answer = ask("A", "E@3")
    assert (answer["answer"], answer["notes"], answer["untraced_steps"]) == ("does not contain", [], [])
    assert answer["scope"]["steps"] == [["E/seg-1", 1, 3]]
    assert answer["scope"]["runs"] == ["E"]
    assert (answer["scope"]["source"], answer["scope"]["members"]) == ("A", ["A/r1"])


def test_contains_states_its_basis_its_first_exposure_and_the_untraced_steps(ask):
    at_4 = ask("A", "E@4")
    assert (at_4["answer"], at_4["basis"], at_4["level"]) == ("contains", v.BASIS_WHOLE_SAMPLE, "consumed")
    assert at_4["first_exposure"] == {"segment": "E/seg-2", "step": 4}
    assert at_4["untraced_steps"] == []
    assert hops(at_4) == ["E/seg-2:step 4", *T_CHAIN]
    assert at_4["notes"] == [v.NOTE_PROCESS_EVIDENCE]
    # Step 5 is lowered by the operator's step record. It changes nothing about the
    # read at step 4 and is listed beside the chain (Decision 20).
    at_6 = ask("A", "E@6")
    assert (at_6["answer"], at_6["first_exposure"]) == ("contains", {"segment": "E/seg-2", "step": 4})
    assert at_6["untraced_steps"] == [["E/seg-2", 5]]


def test_an_ancestors_contains_is_inherited_with_its_basis(ask):
    answer = ask("A", "D@2")
    assert (answer["answer"], answer["basis"], answer["level"]) == (
        "contains", v.BASIS_WHOLE_SAMPLE, "inherited")
    assert answer["first_exposure"] == {"segment": "E/seg-2", "step": 4}
    assert answer["untraced_steps"] == [["E/seg-2", 5]]
    assert hops(answer)[0] == "E/seg-2:step 4"


def test_eligible_is_not_consumed(ask):
    # Y's mixture holds T, and none of its steps read it.
    assert ask("A", "mix-2")["answer"] == "contains"
    answer = ask("A", "Y@2")
    assert answer["answer"] == "does not contain"
    assert answer["scope"]["steps"] == [["E/seg-1", 1, 3], ["Y/seg-1", 1, 2]]
    assert answer["scope"]["runs"] == ["E", "Y"]


def test_not_covered_when_never_registered_or_when_its_segment_is_missing(world, ask):
    unknown = ask("A", "E@12")
    assert (unknown["answer"], unknown["reason"]) == ("not covered", "not registered")
    world.store.register_checkpoint(name="G@1", segment="G/seg-1", step=1)
    orphan = ask("A", "G@1")
    assert (orphan["answer"], orphan["reason"], orphan["kind"]) == ("not covered", "record missing", "checkpoint")


def test_a_descendant_of_a_checkpoint_with_no_record_may_contain(world, ask):
    lost = world.store.register_checkpoint(name="G@1", segment="G/seg-1", step=1)
    ww_world.segment(world.store, "H", "H/seg-1", 0, world["mix-1"], lost, [(1, 1, None)], {1: (0, 0)},
                     start_kind="warm-start")
    world.store.register_checkpoint(name="H@1", segment="H/seg-1", step=1)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED)
    assert link(answer, "artifact", "segment", "reason") == {
        "artifact": "G@1", "segment": "G/seg-1", "reason": "record missing"}


def test_a_segment_started_from_an_unregistered_checkpoint(world, ask):
    ww_world.segment(world.store, "H", "H/seg-1", 0, world["mix-1"], "0" * 64, [(1, 1, None)],
                     {1: (0, 0)}, start_kind="warm-start")
    world.store.register_checkpoint(name="H@1", segment="H/seg-1", step=1)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED)
    assert link(answer, "instance", "reason") == {"instance": "0" * 64, "reason": "record missing"}


# --- (e) every step of the history completely traced ---------------------------------


def test_a_deleted_trace_names_the_step_for_the_checkpoint_and_its_descendants(world, ask):
    world.delete_trace("E/seg-1", 2)
    for name in ("E@3", "Y@2"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED), name
        assert link(answer, "condition", "segment", "step") == {
            "condition": "e", "segment": "E/seg-1", "step": 2}, name
    assert ask("A", "Y@2")["untraced_steps"] == [["E/seg-1", 2]]
    # A shown read elsewhere in the history is not undone by it (Decision 20).
    for name in ("E@6", "D@2"):
        answer = ask("A", name)
        assert answer["answer"] == "contains", name
        assert answer["untraced_steps"] == [["E/seg-1", 2], ["E/seg-2", 5]], name
    # No other kind of artifact has a step trace in its closure.
    assert ask("A", "mix-1")["answer"] == "does not contain"


def test_a_half_covered_sample_makes_its_step_untraced(world, ask):
    world.sql(("UPDATE trace_sample SET covered_tokens = 2 WHERE trace_id = ?",
               (world.trace_id("E/seg-1", 3),)))
    answer = ask("A", "E@3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "e", "segment": "E/seg-1", "step": 3}


def test_an_operator_note_lowers_a_step_even_when_the_stored_mark_is_whole(world, ask):
    # The note says 3 of 5 tokens; the trace was stored with its sample whole.
    world.sql(("UPDATE trace_sample SET covered_tokens = sample_tokens WHERE trace_id = ?",
               (world.trace_id("E/seg-2", 5),)))
    answer = ask("C", "E@6")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "e", "segment": "E/seg-2", "step": 5}
    assert ask("C", "E@4")["answer"] == "does not contain"


def test_a_record_read_in_the_uncovered_part_of_a_sample_is_not_shown_consumed(world, ask):
    # Three covered tokens: all of them belong to the first row, none to T.
    two_row_step(world, "H", covered=3)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "e", "segment": "H/seg-1", "step": 1}
    assert answer["first_exposure"] is None


def test_one_covered_token_of_a_record_is_consumption(world, ask):
    # Four covered tokens: the fourth is T's first (Decision 18). The step is still
    # untraced, and the read is still shown.
    two_row_step(world, "H", covered=4)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["basis"]) == ("contains", v.BASIS_WHOLE_SAMPLE)
    assert answer["first_exposure"] == {"segment": "H/seg-1", "step": 1}
    assert answer["untraced_steps"] == [["H/seg-1", 1]]


def test_where_an_operator_note_covers_the_sample_the_note_is_the_basis(world, ask):
    two_row_step(world, "H", covered=4, note=4)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["basis"]) == ("contains", v.BASIS_OPERATOR_NOTE)
    # A note that leaves T's tokens unread shows no consumption.
    two_row_step(world, "J", covered=5, note=3)
    assert ask("A", "J@1")["answer"] == "may contain"


def test_a_step_missing_from_the_step_record_is_untraced(world, ask):
    ww_world.segment(world.store, "H", "H/seg-1", 0, world["mix-1"], None,
                     [(1, 1, None), (3, 1, None)], {1: (0, 0), 3: (0, 1)})
    world.store.register_checkpoint(name="H@3", segment="H/seg-1", step=3)
    answer = ask("A", "H@3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "e", "segment": "H/seg-1", "step": 2}


def test_steps_before_the_first_registered_one_are_untraced_when_the_launch_shows_them(world, ask):
    # The run record's launch starts at step 1; the step record begins at step 2.
    record = ww_world.run_record("H", "H/seg-1", "fresh", 1, 3)
    world.store.register_run_segment(
        run="H", segment="H/seg-1", position=0, mixture_instance_id=world["mix-1"],
        parent_checkpoint=None, start_kind="fresh", run_record=record,
        steps=[(2, 1, None), (3, 1, None)], trace_route="cache files",
    )
    world.store.write_step_traces("H/seg-1", [
        StepTrace(2, 1, (ww_world.sample(0, 1, 0, 0),)), StepTrace(3, 1, (ww_world.sample(0, 2, 0, 1),))])
    world.store.register_checkpoint(name="H@3", segment="H/seg-1", step=3)
    answer = ask("A", "H@3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_STEP_UNTRACED)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "e", "segment": "H/seg-1", "step": 1}


# --- A record a step consumed fails a condition (plan 2.4) ---------------------------


def test_a_fault_on_a_consumed_record_reaches_the_checkpoints_that_read_it(world, ask):
    world.delete_edge("gen/P", 1)
    at_4 = ask("A", "E@4")
    assert (at_4["answer"], at_4["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert link(at_4, "condition", "artifact", "record") == {
        "condition": "a", "artifact": "gen/P", "record": "L1"}
    # E@6 and D@2 name the same link: step 4 read T before step 5 went untraced, and
    # the untraced step is still listed beside it.
    for name in ("E@6", "D@2"):
        later = ask("A", name)
        assert (later["answer"], later["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN), name
        assert link(later, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "gen/P", "record": "L1"}, name
        assert later["untraced_steps"] == [["E/seg-2", 5]], name
    # A checkpoint whose steps never read T keeps its answer, though Y's mixture holds T.
    assert ask("A", "E@3")["answer"] == "does not contain"
    assert ask("A", "Y@2")["answer"] == "does not contain"


def test_a_deleted_map_reaches_only_the_checkpoints_whose_steps_read_the_shard(world, ask):
    world.delete_map("shard-3")
    assert ask("A", "shard-3")["answer"] == "not covered"
    # E@3 read shard-1 only.
    assert ask("A", "E@3")["answer"] == "does not contain"
    for name in ("E@4", "Y@2"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_UPSTREAM_NOT_COVERED), name
        assert link(answer, "artifact", "reason") == {"artifact": "shard-3", "reason": "record missing"}, name


def test_a_stronger_map_label_reaches_the_checkpoints_that_read_the_shard(world, ask):
    world.sql(("UPDATE binding SET label = 'observed' WHERE binding_id = ?",
               (world.binding_id("shard-1"),)))
    answer = ask("A", "E@3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LABEL_STRONGER)
    assert link(answer, "condition", "artifact") == {"condition": "f", "artifact": "shard-1"}
    assert ask("A", "E@4")["answer"] == "contains"  # T's own chain does not pass that map


def test_a_segment_with_no_mixture_snapshot(world, ask):
    ww_world.segment(world.store, "H", "H/seg-1", 0, None, None, [(1, 1, None)], {1: (0, 0)})
    world.store.register_checkpoint(name="H@1", segment="H/seg-1", step=1)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_SNAPSHOT_MISSING)
    assert link(answer, "segment", "reason") == {"segment": "H/seg-1", "reason": "not registered"}


def test_the_bound_of_a_generation_edge_reaches_a_checkpoint_that_read_its_output(world, ask):
    assert ask("C", "E@4")["scope"]["generation_edges"] == ["translation", "paraphrase"]
    assert ask("C", "E@4")["notes"] == [v.NOTE_GENERATING_MODEL_NOT_ASSESSED]
    world.sql(("UPDATE edge SET bound_established = 0 WHERE edge_id = ?", (world.edge_id("gen/P", 1),)))
    answer = ask("C", "E@4")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    assert link(answer, "condition", "job") == {"condition": "g", "job": "paraphrase"}
    assert ask("C", "E@3")["answer"] == "does not contain"


# --- Versions, samplers, and segments with no trace route (F6, plan 3.5) --------------


def reading_t(world, name: str, **record_args: str) -> None:
    """A fresh one-step run on mix-2 whose step reads T."""
    ww_world.segment(world.store, name, f"{name}/seg-1", 0, world["mix-2"], None, [(1, 1, None)],
                     {1: (1, 0)}, **record_args)
    world.store.register_checkpoint(name=f"{name}@1", segment=f"{name}/seg-1", step=1)


def test_an_unverified_version_gives_may_contain_and_never_contains(world, ask):
    reading_t(world, "V", release="core_v0.99.0")
    answer = ask("A", "V@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_TRACE_UNVERIFIED_VERSION)
    assert answer["failing"]["segment"] == "V/seg-1"
    assert ask("C", "V@1")["answer"] == "may contain"


def test_an_unverified_sampler_on_a_verified_version(world, ask):
    reading_t(world, "V", sampler="cyclic", shuffle=True)
    assert ask("A", "V@1")["names"] == v.NAMES_TRACE_UNVERIFIED_SAMPLER
    reading_t(world, "W", sampler_class="some.other.Sampler")
    assert ask("A", "W@1")["names"] == v.NAMES_TRACE_UNVERIFIED_SAMPLER


def test_a_descendant_of_an_unverified_segment_names_its_ancestor(world, ask):
    reading_t(world, "V", release="core_v0.99.0")
    ww_world.segment(world.store, "H", "H/seg-1", 0, world["mix-1"], world.store.checkpoint("V@1").instance_id,
                     [(1, 1, None)], {1: (0, 0)}, start_kind="warm-start")
    world.store.register_checkpoint(name="H@1", segment="H/seg-1", step=1)
    answer = ask("A", "H@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_ANCESTOR_MAY_CONTAIN)
    assert link(answer, "artifact", "segment") == {"artifact": "V@1", "segment": "V/seg-1"}


@pytest.mark.parametrize("cause, detail", [
    (v.NAMES_CYCLIC_NOT_RECORDED, None),
    (v.NAMES_CYCLIC_DIGEST_MISMATCH, None),
    (v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH),
])
def test_a_segment_with_trace_route_none_answers_with_its_recorded_cause(world, ask, cause, detail):
    # A cyclic run warm-started from E@3, registered with no usable order.
    ww_world.segment(world.store, "K", "K/seg-1", 0, world["mix-2"], world["E@3"],
                     [(1, 1, None), (2, 1, None)], None, start_kind="warm-start", route="none",
                     cause=cause, detail=detail, sampler="cyclic")
    world.store.register_checkpoint(name="K@2", segment="K/seg-1", step=2)
    for source in ("A", "C"):
        answer = ask(source, "K@2")
        assert (answer["answer"], answer["names"], answer["names_detail"]) == ("may contain", cause, detail)
        assert link(answer, "condition", "segment", "step") == {
            "condition": "e", "segment": "K/seg-1", "step": 1}
        assert answer["untraced_steps"] == [["K/seg-1", 1], ["K/seg-1", 2]]
    # Its parent is untouched.
    assert ask("A", "E@3")["answer"] == "does not contain"


def test_a_stored_cause_that_is_not_a_named_cause_is_said_in_words(world, ask):
    # The store writes no such segment. One edited in is still a segment with no
    # trace: the answer names that, says what it found, and passes no reason on.
    ww_world.segment(world.store, "K", "K/seg-1", 0, world["mix-2"], world["E@3"],
                     [(1, 1, None)], None, start_kind="warm-start", route="none",
                     cause=v.NAMES_TRACE_MISSING, detail=v.TRACE_MISSING_NO_DIGEST, sampler="cyclic")
    world.store.register_checkpoint(name="K@1", segment="K/seg-1", step=1)
    assert ask("A", "K@1")["names_detail"] == v.TRACE_MISSING_NO_DIGEST
    world.sql("UPDATE run_segment SET trace_cause = 'because' WHERE name = 'K/seg-1'")
    answer = ask("A", "K@1")
    assert (answer["answer"], answer["names"], answer["names_detail"]) == (
        "may contain", v.NAMES_TRACE_MISSING, None)
    assert "'because'" in answer["failing"]["detail"]
    assert "not a named cause" in answer["failing"]["detail"]


def test_traces_under_a_route_of_none_show_nothing(world, ask):
    reading_t(world, "K")
    assert ask("A", "K@1")["answer"] == "contains"
    world.sql(("UPDATE run_segment SET trace_route = 'none', trace_cause = ? WHERE name = 'K/seg-1'",
               (v.NAMES_CYCLIC_DIGEST_MISMATCH,)))
    answer = ask("A", "K@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_CYCLIC_DIGEST_MISMATCH)


def test_a_cyclic_segment_with_no_shuffle_digest_supports_no_answer_whatever_route_is_stored(world, ask):
    # Plan 2.4, 3.5; DECISIONS.md D-26. The store refuses to write such a route, so
    # the case is a store edited by hand: a segment read with the cyclic sampler, a
    # sampler type the store lists as verified, a route of "cache files", and no
    # shuffle digest in its run record.
    world.sql(("INSERT INTO verified_join (release_tag, sampler_type, sampler_class) VALUES (?, ?, ?)",
               (ww_world.JOIN[0], "cyclic", ww_world.SAMPLER_CLASS)))
    reading_t(world, "K")
    assert ask("A", "K@1")["answer"] == "contains"
    world.sql("UPDATE run_segment SET sampler_type = 'cyclic' WHERE name = 'K/seg-1'")
    answer = ask("A", "K@1")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_CYCLIC_NOT_RECORDED)
    assert answer["untraced_steps"] == [["K/seg-1", 1]]
    # With a digest in the run record the segment reads its traces again.
    world.sql(("INSERT INTO segment_shuffle_digest (segment_id, launch_position, epoch, rank, digest) "
               "SELECT segment_id, 0, 0, 0, ? FROM run_segment WHERE name = 'K/seg-1'", ("0" * 64,)))
    assert ask("A", "K@1")["answer"] == "contains"


def test_the_first_failing_link_is_the_one_at_the_earliest_step(world, ask):
    # P's edge is gone, so the read of T at step 4 fails condition (a); step 5 is
    # untraced by the operator's note. The earlier step is named.
    world.delete_edge("gen/P", 1)
    assert link(ask("A", "E@6"), "condition", "segment", "step") == {
        "condition": "a", "segment": None, "step": None}
    # An untraced step that lies before the read is named in its place.
    world.delete_trace("E/seg-1", 2)
    answer = ask("A", "E@6")
    assert (answer["names"], link(answer, "condition", "segment", "step")) == (
        v.NAMES_STEP_UNTRACED, {"condition": "e", "segment": "E/seg-1", "step": 2})
    assert answer["untraced_steps"] == [["E/seg-1", 2], ["E/seg-2", 5]]


def test_at_one_step_the_untraced_step_comes_before_the_record_it_read(world, ask):
    world.delete_edge("gen/P", 1)
    world.sql(("UPDATE trace_sample SET covered_tokens = 2 WHERE trace_id = ?",
               (world.trace_id("E/seg-2", 4),)))
    answer = ask("A", "E@4")
    assert (answer["names"], link(answer, "condition", "segment", "step")) == (
        v.NAMES_STEP_UNTRACED, {"condition": "e", "segment": "E/seg-2", "step": 4})


def test_a_segments_version_stands_after_its_own_steps(world, ask):
    # An unverified version alone is named. An untraced step of the same segment
    # comes before it.
    reading_t(world, "V", release="core_v0.99.0")
    assert ask("A", "V@1")["names"] == v.NAMES_TRACE_UNVERIFIED_VERSION
    world.delete_trace("V/seg-1", 1)
    assert ask("A", "V@1")["names"] == v.NAMES_STEP_UNTRACED


def test_a_missing_ancestor_record_is_named_before_anything_later(world, ask):
    ww_world.segment(world.store, "H", "H/seg-1", 0, world["mix-1"], "0" * 64,
                     [(1, 1, None), (2, 1, None)], {1: (0, 0)}, start_kind="warm-start")
    world.store.register_checkpoint(name="H@2", segment="H/seg-1", step=2)
    answer = ask("A", "H@2")
    assert answer["names"] == v.NAMES_UPSTREAM_NOT_COVERED
    assert answer["untraced_steps"] == [["H/seg-1", 2]]
