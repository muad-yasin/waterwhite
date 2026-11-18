"""What the stored evidence promises, tried through the real write API, the real
coverage engine and the real store checks.

Each store is built by the write methods (the hand-built world of
``tests/coverage/ww_world.py`` and small stores of this file). A mutation that models
altered history is a direct SQL edit of the store file the write API wrote; creating,
registering, asking and checking go through the real code.

Every test compares an observed value with one named expected value (the constants and
the ``MUTATIONS`` table below). Whether it is red or green is decided by that value
alone.

The sentences each expectation rests on:

- docs/store.md "The loud checks": a fault of the integrity family stops a query and no
  answer is printed; "States that are not faults" lists a step with no trace as an
  untraced step that the answer reads conservatively.
- docs/store.md "Mixtures, runs, traces, checkpoints": a history that is not complete
  "must never be read as a shorter history".
- docs/store.md "Identity" and "The write methods": an artifact keeps its edge digest,
  a mixture's id is made from its shards and settings, and a segment keeps its run
  record whole with its digest. The loud checks compare a run record, a mixture and
  the numbers of a binding's records with what the store recorded of them (docs/store.md
  checks 6 to 8); they compare no edge, trace row or step inventory with anything:
  the proof reads an edited or removed edge and a missing step as states, and the
  store records no commitment for a trace row. The five alterations below ask
  which of the two kinds each one is, not whether the store resists someone who
  rewrites every commitment.
- docs/store.md "The write methods" and docs/capture.md "Builders for what a step
  states": a harness log is stored with its hand-off and its reads in order, and every
  handed record is an input of the generation edge "in hand-off order".
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from waterwhite import coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.records import HarnessRead, OccurrenceRef
from waterwhite.store.store import Store

REPO = Path(__file__).resolve().parents[2]


def _load_world():
    """The hand-built world, loaded from its file whether or not another test folder
    has put its own folder on the path."""
    held = sys.modules.get("ww_world")
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location("ww_world", REPO / "tests" / "coverage" / "ww_world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ww_world"] = module
    spec.loader.exec_module(module)
    return module


ww = _load_world()

FAULT = "integrity fault"
NO_FAULT = "no fault"
MAY_CONTAIN = "may contain"
CONTAINS = "contains"
DOES_NOT_CONTAIN = "does not contain"
REFUSED = "refused"
ACCEPTED = "accepted"

# --- The expected values ------------------------------------------------------------------

# A completed step whose whole inventory is gone from the store is a state the proof reads
# (docs/store.md "States that are not faults", docs/coverage.md condition (e)): the check
# raises nothing, whole store or scoped to the checkpoint, and the answer keeps the step
# as unknown. It is never read as a shorter history. The store holds nothing that tells
# a removed step from a step the operator's record never listed.
EXPECT_REMOVED_STEP_WHOLE_CHECK = NO_FAULT
EXPECT_REMOVED_STEP_SCOPED_CHECK = NO_FAULT
EXPECT_REMOVED_STEP_QUERY = {MAY_CONTAIN}

# Per altered commitment: does the check raise, and does the query stop with the fault.
# artifact, source and records are the question put to the real engine; before is the
# answer on the unaltered world (read off the world's construction, ww_world's docstring).
# FAULT rows: the store holds a digest or a count of the record, and the check compares it
# (docs/store.md checks 6 to 8). NO_FAULT rows pin a state the store holds no commitment
# for: an edge input moved to another record that exists, and a trace row moved to another
# shard position over the same token span (docs/store.md "States that are not faults");
# the query answers from the altered record.
MUTATIONS = {
    "edge-redirected-to-another-parent": dict(
        check=NO_FAULT, query=DOES_NOT_CONTAIN, artifact="R2", source="B", before=CONTAINS,
        records=lambda world: [world.ref("src/B", 1)]),
    "trace-redirected-to-another-shard-position": dict(
        check=NO_FAULT, query=DOES_NOT_CONTAIN, artifact="E@4", source="A", before=CONTAINS,
        records=lambda world: []),
    "occurrence-locator-changed-with-the-count-kept": dict(
        check=FAULT, query=FAULT, artifact="gen/P", source="A", before=CONTAINS,
        records=lambda world: []),
    "run-record-changed-without-its-digest": dict(
        check=FAULT, query=FAULT, artifact="E@3", source="A", before=DOES_NOT_CONTAIN,
        records=lambda world: []),
    "mixture-snapshot-entry-removed": dict(
        check=FAULT, query=FAULT, artifact="mix-2", source="A", before=CONTAINS,
        records=lambda world: []),
}

# The hand-off of a harness log is read back in the order it was handed.
EXPECT_HANDOFF_READ_BACK_IN_ORDER = True
# The edge inputs and the reads keep their order today; they stay as they are.
EXPECT_EDGE_INPUTS_KEEP_HANDOFF_ORDER = True
EXPECT_READS_KEEP_THEIR_ORDER = True
# A log of a job name already stored, with the same hand-off in another order, differs
# from the stored one.
EXPECT_REORDERED_LOG_OF_A_STORED_JOB = REFUSED

# Two digests for one blend array of one segment.
EXPECT_TWO_BLEND_DIGESTS_IN_ONE_RECORD = REFUSED
EXPECT_ONE_BLEND_DIGEST_IN_ONE_RECORD = ACCEPTED
EXPECT_OTHER_BLEND_DIGEST_FOR_A_STORED_SEGMENT = REFUSED
EXPECT_SECOND_NULL_POSITION_ROW_BY_DIRECT_INSERT = REFUSED
EXPECT_CONTRADICTORY_BLEND_ROWS_DETECTED = True


# --- Helpers ---------------------------------------------------------------------------------


def check_outcome(store: Store, instance_ids=None) -> str:
    try:
        store.check_integrity(instance_ids)
    except errors.IntegrityFault:
        return FAULT
    return NO_FAULT


def query_outcome(store: Store, artifact: str, source: str, records=()) -> tuple[str, dict | None]:
    """Put a withdrawal to the real engine. The outcome is the integrity fault, another
    named refusal (spelt with its class), or the answer's word, with the answer."""
    try:
        request = requests.withdraw(store, source, records=records)
        answer = coverage.answer(store, artifact, request).to_json()
    except errors.IntegrityFault:
        return FAULT, None
    except errors.WaterwhiteError as other:
        return f"{REFUSED}: {type(other).__name__}", None
    return answer["answer"], answer


def shown(answer: dict | None):
    """The parts of an answer worth reading in a failed assertion."""
    if answer is None:
        return None
    keys = ("answer", "names", "chain", "first_exposure", "untraced_steps", "failing")
    return {k: answer[k] for k in keys if k in answer}


def keep_answer(world, name: str, observed: str, answer: dict | None) -> None:
    """Write the whole answer beside the store, in the test's own folder."""
    (world.folder / f"answer-{name}.json").write_text(
        json.dumps({"outcome": observed, "answer": answer}, indent=2, sort_keys=True, default=str), encoding="utf-8")


@pytest.fixture
def world(tmp_path):
    built = ww.build(tmp_path)
    yield built
    built.store.close()


# --- A completed step whose whole inventory is gone ---------------------------------------


def add_run_with_a_middle_step_that_reads_a(world) -> None:
    """Run F of three steps on mix-2 with a checkpoint at step 3. Step 2 reads shard
    position 1, ordinal 0: the record T that descends from source A. Step 1 and step 3
    read shard 1 (records of B and Z). So step 2 is the only exposure to A."""
    ww.segment(world.store, "F", "F/seg-1", 0, world["mix-2"], None,
               [(1, 1, None), (2, 1, None), (3, 1, None)], {1: (0, 0), 2: (1, 0), 3: (0, 2)})
    world.ids["F@3"] = world.store.register_checkpoint(name="F@3", segment="F/seg-1", step=3)


def remove_the_whole_step(world, segment: str, step: int) -> None:
    """Delete the step's rows from the inventory and the trace, from the leaf up: the
    rows of its trace, its samples, its trace, and the step itself."""
    trace = world.trace_id(segment, step)
    segment_id = world.scalar("SELECT segment_id FROM run_segment WHERE name = ?", (segment,))
    world.sql(
        ("DELETE FROM trace_row WHERE trace_id = ?", (trace,)),
        ("DELETE FROM trace_sample WHERE trace_id = ?", (trace,)),
        ("DELETE FROM step_trace WHERE trace_id = ?", (trace,)),
        ("DELETE FROM segment_step WHERE segment_id = ? AND step = ?", (segment_id, step)),
    )


def test_control_the_unaltered_run_contains_a_and_names_step_2_as_the_first_exposure(world):
    add_run_with_a_middle_step_that_reads_a(world)
    assert check_outcome(world.store) == NO_FAULT
    outcome, answer = query_outcome(world.store, "F@3", "A")
    assert outcome == CONTAINS, answer
    assert answer["first_exposure"] == {"segment": "F/seg-1", "step": 2}


def test_control_a_step_whose_trace_alone_is_gone_is_untraced_and_the_answer_may_contain(world):
    # "States that are not faults": a step with no trace is an untraced step.
    add_run_with_a_middle_step_that_reads_a(world)
    world.delete_trace("F/seg-1", 2)
    assert check_outcome(world.store) == NO_FAULT
    outcome, answer = query_outcome(world.store, "F@3", "A")
    assert outcome == MAY_CONTAIN, answer


def test_control_after_the_removal_the_launch_still_says_steps_1_to_3_completed(world):
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    launch = world.store._conn.execute(
        "SELECT first_step, last_step FROM segment_launch WHERE segment_id = "
        "(SELECT segment_id FROM run_segment WHERE name = 'F/seg-1')").fetchone()
    assert tuple(launch) == (1, 3)
    assert [s.step for s in world.store.segment("F/seg-1").steps] == [1, 3]
    assert [c.names for c in world.store.checkpoints_of("F/seg-1")] == [("F@3",)]


def test_a_removed_middle_step_is_a_fault_of_the_whole_store_check(world):
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    observed = check_outcome(world.store)
    assert observed == EXPECT_REMOVED_STEP_WHOLE_CHECK, observed


def test_a_removed_middle_step_is_a_fault_of_the_check_scoped_to_the_checkpoint(world):
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    observed = check_outcome(world.store, [world["F@3"]])
    assert observed == EXPECT_REMOVED_STEP_SCOPED_CHECK, observed


def test_a_removed_middle_step_never_turns_contains_into_a_shorter_history_negative(world):
    # Step 2 was the only exposure to A. The full query must refuse or keep the step
    # as unknown; it must not answer "does not contain".
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    observed, answer = query_outcome(world.store, "F@3", "A")
    keep_answer(world, "removed-middle-step", observed, answer)
    assert observed in EXPECT_REMOVED_STEP_QUERY, (observed, shown(answer))


def passed_query_outcome(store: Store, artifact: str, source: str, scope) -> tuple[str, dict | None]:
    """The same withdrawal put after a loud-check pass: over the whole store (scope None)
    or over the closure of the listed artifacts."""
    try:
        integrity = coverage.check_store(store, scope)
        answer = coverage.answer(store, artifact, requests.withdraw(store, source), integrity=integrity).to_json()
    except errors.IntegrityFault:
        return FAULT, None
    except errors.WaterwhiteError as other:
        return f"{REFUSED}: {type(other).__name__}", None
    return answer["answer"], answer


def test_a_removed_middle_step_reads_may_contain_after_a_whole_store_pass(world):
    # The four kinds of row of step 2 are gone, step 2 was the only exposure to A, and the
    # loud checks passed over the whole store before the query.
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    observed, answer = passed_query_outcome(world.store, "F@3", "A", None)
    keep_answer(world, "removed-middle-step-whole-store-pass", observed, answer)
    assert observed == MAY_CONTAIN, (observed, shown(answer))


def test_a_removed_middle_step_reads_may_contain_after_a_pass_over_the_checkpoint(world):
    add_run_with_a_middle_step_that_reads_a(world)
    remove_the_whole_step(world, "F/seg-1", 2)
    observed, answer = passed_query_outcome(world.store, "F@3", "A", [world["F@3"]])
    keep_answer(world, "removed-middle-step-checkpoint-pass", observed, answer)
    assert observed == MAY_CONTAIN, (observed, shown(answer))


def test_control_the_unaltered_ancestor_segment_is_the_only_exposure_to_record_b_l2(world):
    # E/seg-1 step 2 reads shard 0, ordinal 1: the record B:L2. No other step of the
    # history of E@6 reads it.
    outcome, answer = query_outcome(world.store, "E@6", "B", [world.ref("src/B", 2)])
    assert outcome == CONTAINS, shown(answer)
    assert answer["first_exposure"] == {"segment": "E/seg-1", "step": 2}


def test_a_step_removed_from_an_ancestor_segment_never_turns_contains_into_a_negative(world):
    remove_the_whole_step(world, "E/seg-1", 2)
    observed, answer = query_outcome(world.store, "E@6", "B", [world.ref("src/B", 2)])
    keep_answer(world, "removed-ancestor-step", observed, answer)
    assert observed in EXPECT_REMOVED_STEP_QUERY, (observed, shown(answer))


# --- Five changes to the meaning of recorded evidence ---------------------------------------


def mutate(world, name: str) -> None:
    if name == "edge-redirected-to-another-parent":
        # R2's line 1 is a copy of B:L1; its edge now names Z:L3, a record of the
        # other input of R2, which exists.
        world.sql(("UPDATE edge_input SET occurrence_id = ? WHERE edge_id = ? AND position = 0",
                   (world.occurrence_id("src/Z", 3), world.edge_id("R2", 1))))
    elif name == "trace-redirected-to-another-shard-position":
        # Step 4 of E/seg-2 read shard position 1, ordinal 0 over tokens 0 to 5; the row now
        # names shard position 0, ordinal 0 over the same token span.
        world.sql(("UPDATE trace_row SET shard_position = 0 WHERE trace_id = ?",
                   (world.trace_id("E/seg-2", 4),)))
    elif name == "occurrence-locator-changed-with-the-count-kept":
        # A:L1 of a two-line file is now called L999; the occurrence count is still 2.
        world.sql(("UPDATE occurrence SET locator = 'L999' WHERE occurrence_id = ?",
                   (world.occurrence_id("src/A", 1),)))
    elif name == "run-record-changed-without-its-digest":
        text = world.scalar("SELECT run_record FROM run_segment WHERE name = ?", ("E/seg-1",))
        record = json.loads(text)
        record["settings"]["random_seed"] = 4321
        world.sql(("UPDATE run_segment SET run_record = ? WHERE name = ?",
                   (json.dumps(record, sort_keys=True, separators=(",", ":")), "E/seg-1")))
    elif name == "mixture-snapshot-entry-removed":
        # The mixture keeps its instance id; its snapshot loses the entry at position 1.
        world.sql(("DELETE FROM mixture_shard WHERE position = 1 AND artifact_id = "
                   "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-2"],)))
    else:  # pragma: no cover
        raise AssertionError(name)


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_control_the_unaltered_world_passes_the_check_and_answers_the_question(world, name):
    case = MUTATIONS[name]
    assert check_outcome(world.store) == NO_FAULT
    outcome, answer = query_outcome(world.store, case["artifact"], case["source"], case["records"](world))
    assert outcome == case["before"], answer


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_the_whole_store_check_raises_for(world, name):
    mutate(world, name)
    observed = check_outcome(world.store)
    assert observed == MUTATIONS[name]["check"], observed


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_a_query_over_the_altered_history_stops_with_the_fault_for(world, name):
    case = MUTATIONS[name]
    records = case["records"](world)
    mutate(world, name)
    observed, answer = query_outcome(world.store, case["artifact"], case["source"], records)
    keep_answer(world, name, observed, answer)
    assert observed == case["query"], (observed, shown(answer))


# --- The hand-off of a harness log ----------------------------------------------------------


def build_order_store(tmp_path, folder: str = "order"):
    """Store with a source A of two lines and an output of the job "order-job" that was
    handed A:L2 and then A:L1 and states both as used. Returns the store path, the
    artifact's instance id, the output's instance id and the hand-off in the order given."""
    base = tmp_path / folder
    base.mkdir()
    store = Store.create(base / "s.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN])
    a = ww.ingest(store, base / "A.jsonl", ["alpha", "beta"], "A", "src/A")
    handed = [ww.ref(a, 2), ww.ref(a, 1)]
    reads = [HarnessRead(occurrence=ref) for ref in handed]
    q = ww.record(store, base / "Q.jsonl", ["summary"], "gen/Q", v.ROLE_INTERMEDIATE,
                  ww.step("generate", "Q"), [a],
                  [ww.generation_edge(1, "order-job", handed, handed, reads)])
    store.close()
    return base / "s.db", a, q, handed


def test_a_hand_off_handed_as_l2_then_l1_is_read_back_in_that_order_after_reopening(tmp_path):
    path, _, _, handed = build_order_store(tmp_path)
    with Store.open(path) as store:
        log = store.harness_log("order-job")
        observed = log.handed == tuple(handed)
    assert observed is EXPECT_HANDOFF_READ_BACK_IN_ORDER, [r.key for r in log.handed]


def test_the_edge_inputs_of_that_job_keep_the_hand_off_order_after_reopening(tmp_path):
    path, _, q, handed = build_order_store(tmp_path)
    with Store.open(path) as store:
        edge = store.edge_of(OccurrenceRef(q, "L1"))
        observed = tuple(item.parent for item in edge.inputs) == tuple(handed)
    assert observed is EXPECT_EDGE_INPUTS_KEEP_HANDOFF_ORDER


def test_the_reads_of_that_job_keep_their_order_after_reopening(tmp_path):
    path, _, _, handed = build_order_store(tmp_path)
    with Store.open(path) as store:
        observed = tuple(read.occurrence for read in store.harness_log("order-job").reads) == tuple(handed)
    assert observed is EXPECT_READS_KEEP_THEIR_ORDER


def test_a_log_of_a_stored_job_name_with_the_hand_off_in_another_order_is_refused(tmp_path):
    # The first output was handed L2 then L1. A second output of another file names the
    # same job and the same log (digest, model, reads) with the hand-off L1 then L2.
    path, a, _, _ = build_order_store(tmp_path)
    other = [ww.ref(a, 1), ww.ref(a, 2)]
    reads = [HarnessRead(occurrence=ref) for ref in [ww.ref(a, 2), ww.ref(a, 1)]]
    with Store.open(path) as store:
        try:
            ww.record(store, path.parent / "R.jsonl", ["second summary"], "gen/R", v.ROLE_INTERMEDIATE,
                      ww.step("generate", "R"), [a],
                      [ww.generation_edge(1, "order-job", other, other, reads)])
            observed = ACCEPTED
        except errors.InvalidRecord:
            observed = REFUSED
    assert observed == EXPECT_REORDERED_LOG_OF_A_STORED_JOB


# --- One digest for one blend array --------------------------------------------------------


def blend_record(*digests: str) -> dict:
    record = ww.run_record("G", "G/seg-1", "fresh", 1, 1)
    record["cache_digests"] = [{"kind": v.CACHE_DATASET_INDEX, "shard": None, "digest": d} for d in digests]
    return record


def register_blend_segment(store: Store, *digests: str, run_record=None) -> None:
    store.register_run_segment(
        run="G", segment="G/seg-1", position=0, mixture_instance_id=None, parent_checkpoint=None,
        start_kind="fresh", run_record=run_record or blend_record(*digests), steps=[(1, 1, None)],
        trace_route=v.ROUTE_NONE, trace_cause=v.NAMES_TRACE_MISSING,
        trace_cause_detail=v.TRACE_MISSING_NO_DIGEST,
    )


def registration_outcome(store: Store, *digests: str) -> str:
    try:
        register_blend_segment(store, *digests)
    except errors.WriteRefused:
        return REFUSED
    return ACCEPTED


@pytest.fixture
def blend_path(tmp_path):
    path = tmp_path / "blend.db"
    with Store.create(path, normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        register_blend_segment(store, "a" * 64)
    return path


def insert_second_blend_row(path: Path, digest: str) -> str:
    """A second row for the same segment and dataset_index with a NULL shard position, by
    a direct insert into the file, with foreign keys on."""
    raw = sqlite3.connect(path)
    try:
        raw.execute("PRAGMA foreign_keys = ON")
        raw.execute(
            "INSERT INTO segment_cache_digest (segment_id, kind, shard_position, digest) VALUES "
            "((SELECT segment_id FROM run_segment WHERE name = 'G/seg-1'), 'dataset_index', NULL, ?)",
            (digest,),
        )
        raw.commit()
    except sqlite3.IntegrityError:
        return REFUSED
    finally:
        raw.close()
    return ACCEPTED


def test_control_a_record_with_one_blend_digest_is_registered(tmp_path):
    with Store.create(tmp_path / "one.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        observed = registration_outcome(store, "a" * 64)
        assert [(c.kind, c.shard_position, c.digest) for c in store.segment("G/seg-1").cache_digests] == [
            (v.CACHE_DATASET_INDEX, None, "a" * 64)]
    assert observed == EXPECT_ONE_BLEND_DIGEST_IN_ONE_RECORD


def test_a_run_record_with_two_digests_for_the_blend_array_is_refused_by_the_registration(tmp_path):
    with Store.create(tmp_path / "two.db", normaliser=ww.NORMALISER, verified_joins=[ww.JOIN]) as store:
        observed = registration_outcome(store, "a" * 64, "b" * 64)
        assert store.segment("G/seg-1") is None or observed == ACCEPTED
    assert observed == EXPECT_TWO_BLEND_DIGESTS_IN_ONE_RECORD


def test_a_stored_segment_registered_again_with_another_blend_digest_is_refused(blend_path):
    with Store.open(blend_path) as store:
        observed = registration_outcome(store, "b" * 64)
    assert observed == EXPECT_OTHER_BLEND_DIGEST_FOR_A_STORED_SEGMENT


def test_a_second_row_for_one_blend_array_with_a_null_position_is_refused_by_the_file(blend_path):
    observed = insert_second_blend_row(blend_path, "b" * 64)
    assert observed == EXPECT_SECOND_NULL_POSITION_ROW_BY_DIRECT_INSERT


def test_contradictory_digests_of_one_blend_array_are_caught_by_the_file_or_by_the_check(blend_path):
    # Either the insert is refused, or the store, once it holds both rows, refuses to pass.
    inserted = insert_second_blend_row(blend_path, "b" * 64)
    if inserted == REFUSED:
        detected = True
    else:
        with Store.open(blend_path) as store:
            detected = check_outcome(store) == FAULT
    assert detected is EXPECT_CONTRADICTORY_BLEND_ROWS_DETECTED, inserted
