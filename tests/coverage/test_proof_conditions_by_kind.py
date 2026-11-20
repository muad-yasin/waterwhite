"""Conditions of the coverage proof (plan 2.2) that no test failed alone before: each
is applied to one fault, from outside the write API, on the small store of
``ww_world``. The expected words are the plan's: condition (a) names the record that
has no recorded origin (2.2(a), 2.3), (b) the link too weak for the request (2.2(b),
3.2), (c), (d) and (e) stop the query loudly (3.4), (g) the job whose bound is not
established (2.2(g)).

These tests guard rules the build already holds, as far as the build is known to
hold them; a failure here is a defect of the tool and not of the expected value.
"""

from __future__ import annotations

import pytest
import ww_world

from waterwhite import capture, coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Finding, Proof
from waterwhite.records import EdgeInput, EdgeRecord, HarnessLog, HarnessRead


def link(answer: dict, *keys: str) -> dict:
    return {key: answer["failing"].get(key) for key in keys}


# --- (a) --------------------------------------------------------------------------------


def test_a_map_row_that_names_no_record_fails_condition_a(world, ask):
    assert ask("A", "shard-1")["answer"] == "does not contain"
    world.sql(("UPDATE map_row SET parent_occurrence_id = NULL WHERE ordinal = 0 AND binding_id = ?",
               (world.binding_id("shard-1"),)))
    for name in ("shard-1", "mix-1"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN), name
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "shard-1", "record": "D0"}, name


def test_an_edge_that_names_no_parent_fails_condition_a(world, ask):
    assert ask("A", "R2")["answer"] == "does not contain"
    world.sql(("DELETE FROM edge_input WHERE edge_id = ?", (world.edge_id("R2", 1),)))
    for name in ("R2", "shard-1", "mix-1"):
        answer = ask("A", name)
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN), name
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "R2", "record": "L1"}, name


def test_a_candidate_set_with_no_candidate_fails_condition_a(world, ask):
    r2 = world["R2"]
    world.ids["twins"], _ = ww_world.shard(
        world.store, world.folder / "shards/twins", "twins", [r2],
        [ww_world.twins(0, ww_world.ref(r2, 2), ww_world.ref(r2, 3))])
    assert ask("A", "twins")["answer"] == "does not contain"
    world.sql(("DELETE FROM map_candidate WHERE ordinal = 0 AND binding_id = ?",
               (world.binding_id("twins"),)))
    answer = ask("A", "twins")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert link(answer, "condition", "artifact", "record") == {
        "condition": "a", "artifact": "twins", "record": "D0"}


def test_a_trace_row_that_names_a_shard_position_the_snapshot_does_not_hold(world, ask):
    assert ask("A", "Y@2")["answer"] == "does not contain"
    world.sql(("UPDATE trace_row SET shard_position = 7 WHERE trace_id = ?",
               (world.trace_id("Y/seg-1", 1),)))
    answer = ask("A", "Y@2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert link(answer, "condition", "segment", "step") == {
        "condition": "a", "segment": "Y/seg-1", "step": 1}


def test_a_trace_row_that_names_an_ordinal_the_map_does_not_hold(world, ask):
    assert ask("A", "Y@2")["answer"] == "does not contain"
    world.sql(("UPDATE trace_row SET ordinal = 99 WHERE trace_id = ?",
               (world.trace_id("Y/seg-1", 2),)))
    answer = ask("A", "Y@2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    assert link(answer, "condition", "artifact", "record", "segment", "step") == {
        "condition": "a", "artifact": "shard-3", "record": "D99", "segment": "Y/seg-1", "step": 2}


def test_a_trace_row_whose_segment_is_not_its_traces_segment_is_a_read_the_segment_does_not_hold(world, ask):
    """The audit of a078390: ``trace_hits`` finds a segment's rows through the segment each row
    stores, and the walk lists a segment's documents through the trace of each step. A row
    whose stored segment is empty (or another's) left the document listed and with no row to
    read: the read was taken for one in an uncovered range, and the answer stayed "does not
    contain". It is the same class as an ordinal the map does not hold, and has the same
    answer: the step read a document the segment's rows do not hold."""
    assert ask("A", "Y@2")["answer"] == "does not contain"
    world.sql(("UPDATE trace_row SET segment_id = NULL WHERE trace_id = ?",
               (world.trace_id("Y/seg-1", 2),)))
    answer = ask("A", "Y@2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN)
    shown = link(answer, "condition", "artifact", "segment", "step")
    assert shown == {"condition": "a", "artifact": "shard-3", "segment": "Y/seg-1", "step": 2}


# --- (b) --------------------------------------------------------------------------------


def test_a_file_level_edge_cannot_decide_a_record_level_request_for_a_checkpoint(world, ask):
    assert ask("A", "E@4", records=["src/A:L1"])["answer"] == "contains"
    world.sql(("UPDATE edge SET label = 'file-level', derivation = 'file' WHERE edge_id = ?",
               (world.edge_id("gen/T", 1),)))
    for name in ("E@4", "E@6", "D@2"):
        answer = ask("A", name, records=["src/A:L1"])
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_LEVEL_TOO_WEAK), name
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "b", "artifact": "gen/T", "record": "L1"}, name
    # A request about the whole source is one the link can decide (plan 3.2).
    assert ask("A", "E@4")["answer"] == "contains"


def test_a_file_level_link_beside_a_record_level_link_still_decides_the_whole_source(world, ask):
    """One link to source B on a record never masks the file-level link of the same
    file to source A (plan 2.2(b))."""
    ww_world.write_jsonl(world.folder / "legacy/T.jsonl", ["Uebersetzung von P", "another text"])
    imported = capture.import_jsonl(
        world.store, world.folder / "legacy/T.jsonl", name="import/T", member_name="legacy/T",
        file_family=("A/r1", "file family: translations made from source A"))
    assert ask("A", "import/T")["answer"] == "contains"
    world.store.add_source_link(
        imported, "L1", "B/r1",
        v.MAPPED, "a publisher's pointer to B", derivation=v.DERIVED_POINTER,
        verification=v.VERIFIED_PASSED)
    whole = ask("A", "import/T")
    assert whole["answer"] == "contains"
    assert whole["chain"][0]["link"] == "file-level link"
    assert [hop["record"] for hop in whole["chain"]] == ["L1"]


# --- (c), (d), (e) reached through an ancestor or through a map ------------------------


def test_a_map_row_whose_parent_is_gone_stops_the_proof_loudly(world):
    request = requests.withdraw(world.store, "A")
    world.sql(("UPDATE map_row SET parent_occurrence_id = 987654 WHERE ordinal = 0 AND binding_id = ?",
               (world.binding_id("shard-1"),)))
    for name in ("shard-1", "mix-1", "mix-2", "E@3", "D@2"):
        with pytest.raises(errors.EdgeParentMissing):
            coverage.prove(world.store, name, request)
    # shard-3 and what reads only it are outside the fault's closure.
    assert isinstance(coverage.prove(world.store, "shard-3", request), Finding)


def test_a_candidate_whose_record_is_gone_stops_the_proof_loudly(world):
    r2 = world["R2"]
    world.ids["twins"], _ = ww_world.shard(
        world.store, world.folder / "shards/twins", "twins", [r2],
        [ww_world.twins(0, ww_world.ref(r2, 2), ww_world.ref(r2, 3))])
    request = requests.withdraw(world.store, "A")
    assert isinstance(coverage.prove(world.store, "twins", request), Proof)
    world.sql(("UPDATE map_candidate SET occurrence_id = 987654 WHERE ordinal = 0 AND binding_id = ? "
               "AND occurrence_id = ?", (world.binding_id("twins"), world.occurrence_id("R2", 2))))
    with pytest.raises(errors.EdgeParentMissing):
        coverage.prove(world.store, "twins", request)


def test_a_loud_fault_in_an_ancestor_segment_stops_the_checkpoints_that_descend_from_it(world):
    request = requests.withdraw(world.store, "A")
    # A trace of E's second segment whose own count is wrong. D descends from E@6 and Y
    # from E@3, whose history stops before the segment.
    world.sql(("UPDATE step_trace SET consumed_samples = 2 WHERE trace_id = ?",
               (world.trace_id("E/seg-2", 4),)))
    for name in ("E@4", "E@6", "D@2"):
        with pytest.raises(errors.TraceSampleCountMismatch):
            coverage.prove(world.store, name, request)
    for name in ("E@3", "Y@2"):
        assert isinstance(coverage.prove(world.store, name, request), Proof), name


def test_a_map_digest_fault_in_an_ancestors_mixture_stops_the_descendant(world):
    """D reads mix-1 alone, which holds shard-1. Its history starts from E@6, whose
    second segment read mix-2 and so shard-3: the fault of that map lies in D's closure."""
    request = requests.withdraw(world.store, "A")
    world.sql(("UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
               "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
               (world.binding_id("shard-3"),)))
    for name in ("E@6", "D@2", "Y@2"):
        with pytest.raises(errors.MapDigestMismatch):
            coverage.prove(world.store, name, request)
    assert isinstance(coverage.prove(world.store, "E@3", request), Proof)


# --- (g) ---------------------------------------------------------------------------------


def test_an_input_outside_the_hand_off_denies_the_bound(world, ask):
    """A job handed two records of Z whose log reads one of them. The edge's second input
    then leaves the hand-off (a store edited from outside): the log's reads all lie inside
    the hand-off, but one input of the edge does not, and plan 2.2(g) establishes the bound
    from a log that lists no record outside the edge's input scope."""
    z = world["src/Z"]
    first, second = ww_world.ref(z, 1), ww_world.ref(z, 2)
    log = HarnessLog(
        "job-two", "log-digest-job-two", (first, second), (HarnessRead(occurrence=first),), "none",
        harness=v.HARNESS_CONFINED)
    edge = EdgeRecord(
        "L1", (EdgeInput(first, used=True), EdgeInput(second, used=False)), ww_world.step("generate"),
        is_generation=True, harness_log=log, job_name="job-two",
        handed_texts=tuple((r, ww_world.TEXTS[(r.instance_id, r.locator)]) for r in (first, second)))
    world.ids["gen/J2"] = ww_world.record(
        world.store, world.folder / "gen/J2.jsonl", ["made from two z lines"], "gen/J2",
        v.ROLE_INTERMEDIATE, ww_world.step("job-two"), [z], [edge])
    assert ask("A", "gen/J2")["answer"] == "does not contain"
    world.sql(("DELETE FROM harness_handed WHERE log_id = ? AND occurrence_id = ?",
               (world.log_id("job-two"), world.occurrence_id("src/Z", 2))))
    answer = ask("A", "gen/J2")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
    assert link(answer, "condition", "job", "artifact", "record") == {
        "condition": "g", "job": "job-two", "artifact": "gen/J2", "record": "L1"}


def test_a_handed_record_the_edge_leaves_out_denies_the_bound(world, ask):
    """The outside review of a011cd5, finding 1: a job was handed two records of Z and its log
    reads one; the edge lists both. Then the row of the second input leaves the edge (a
    store edited from outside, which the write would have refused). The stored log still
    holds the record, so the bound is not established and no "does not contain" may rest
    on this edge."""
    z = world["src/Z"]
    first, second = ww_world.ref(z, 1), ww_world.ref(z, 2)
    log = HarnessLog(
        "job-left-out", "log-digest-job-left-out", (first, second), (HarnessRead(occurrence=first),),
        "none", harness=v.HARNESS_CONFINED)
    edge = EdgeRecord(
        "L1", (EdgeInput(first, used=True), EdgeInput(second, used=False)), ww_world.step("generate"),
        is_generation=True, harness_log=log, job_name="job-left-out",
        handed_texts=tuple((r, ww_world.TEXTS[(r.instance_id, r.locator)]) for r in (first, second)))
    world.ids["gen/J3"] = ww_world.record(
        world.store, world.folder / "gen/J3.jsonl", ["made from two z lines"], "gen/J3",
        v.ROLE_INTERMEDIATE, ww_world.step("job-left-out"), [z], [edge])
    assert ask("A", "gen/J3")["answer"] == "does not contain"
    world.sql(("DELETE FROM edge_input WHERE edge_id = ? AND occurrence_id = ?",
               (world.edge_id("gen/J3", 1), world.occurrence_id("src/Z", 2))))
    answer = ask("A", "gen/J3")
    assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_BOUND_NOT_ESTABLISHED)
