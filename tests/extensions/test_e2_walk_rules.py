"""The generating-model rule over the library world (File 1 of the spec of E2).

The world is ``tests/coverage/ww_world.py`` (its docstring is the source of the facts below):
sources A, B, C, Z; the generated P (job "paraphrase", from A line 1), T (job "translation",
from P) and S (job "summary", from A line 2 and C line 1); run E, whose checkpoint E@3 read
shard-1 only (it answers "does not contain" for A and for C) and whose checkpoint E@4 read T at
step 4 ("contains" for A, "does not contain" for C); run Y, which branched from E@3 and never
read T ("does not contain" for A). Every log of the world carries the model "none" and is
registered with its texts, so its bound is established.

The standard question is ``kit.summary``: withdraw A narrowed to the record src/A:L1, asked of
gen/S. A test sets the model of a job with ``kit.set_model`` (an edit that leaves the mark
``hand_off_checked`` as built) and turns the switch on with ``kit.turn_on``.

Each expected value comes from the plan line, owner answer or ruling the test's docstring names;
none comes from running the code under test. Every on-state question goes through
``kit.ask_or_error``, so that a question the tool refuses fails at the assertion on a value.
"""

from __future__ import annotations

import json

import pytest
from e2_kit import ask, world  # noqa: F401  (fixtures)

import e2_kit as kit
from waterwhite import capture, coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.records import EdgeInput, EdgeRecord

ww = kit.ww

EXPOSED = v.NAMES_MODEL_EXPOSED  # "generating model exposed"
NOT_REGISTERED = v.NAMES_MODEL_NOT_REGISTERED  # "generating model not registered"
ENTRY_MISSING = v.NAMES_MODEL_ENTRY_MISSING  # "generating-model entry missing"


def verdict(answer: dict) -> tuple:
    """(answer, names, failing.condition, failing.artifact, failing.job) of an answer."""
    return (answer["answer"], answer.get("names"), *kit.link(answer, "condition", "artifact", "job"))


# --- The standing negative with the switch on -----------------------------------------------------


def test_a_model_of_none_lets_the_negative_stand_with_the_switch_on(world, ask):
    """Plan 2.2(h) (signed, plan line 181 (a)): a generating model of "none" lets the negative
    stand. The same question with the switch off is the table row
    c05.record-X@A.unmutated.gen/S."""
    kit.turn_on(world)
    assert kit.summary(ask)["answer"] == "does not contain"


def test_a_standing_negative_with_the_switch_on_carries_no_exposure_note(world, ask):
    """Plan lines 168 and 181 (c): the words "generating-model exposure not assessed" belong to
    the state in which the extension is off."""
    kit.turn_on(world)
    assert kit.summary(ask).get("notes") == []


# --- A registered checkpoint as the model ---------------------------------------------------------


def test_a_clean_checkpoint_as_model_lets_the_negative_stand(world, ask):
    """Plan line 181 (a): a registered checkpoint that answers "does not contain" for the
    request (table row c07.R3.G6 is the same case in the fixture)."""
    kit.set_model(world, "summary", "E@3")
    kit.turn_on(world)
    # A precondition, asked of the tool: E@3 is clean for the request.
    assert ask("A", "E@3", records=kit.SUMMARY_RECORDS)["answer"] == "does not contain"
    assert kit.summary(ask)["answer"] == "does not contain"


def test_an_exposed_checkpoint_as_model_gives_may_contain_naming_it(world, ask):
    """Plan line 181 (b) and line 228 (a); the form of table row c07.R3.G9."""
    kit.set_model(world, "summary", "E@4")
    kit.turn_on(world)
    # A precondition: E@4 answers "contains" for the request.
    assert ask("A", "E@4", records=kit.SUMMARY_RECORDS)["answer"] == "contains"
    assert verdict(kit.summary(ask)) == ("may contain", EXPOSED, "h", "E@4", "summary")


def test_a_model_the_store_was_never_told_of_gives_not_registered(world, ask):
    """Plan line 228 (a), line 345 and line 181 (b); table row c07.R3.GU, whose failing.artifact
    is the spelling of the unregistered model."""
    kit.set_model(world, "summary", "hosted-model-x")
    kit.turn_on(world)
    assert verdict(kit.summary(ask)) == ("may contain", NOT_REGISTERED, "h", "hosted-model-x", "summary")


def test_a_model_text_that_is_a_checkpoints_instance_id_reads_as_not_registered(world, ask):
    """An instance id finds the checkpoint but is no name the store registered it under: the answer
    ends in the sentence of a model that is not registered, and never says that a text "names a
    checkpoint, not a checkpoint"."""
    text = world["E@4"]
    kit.set_model(world, "summary", text)
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert verdict(shown)[:2] == ("may contain", NOT_REGISTERED)
    assert kit.link(shown, "detail") == (f"the generating model {text} is not a registered checkpoint.",)


@pytest.mark.parametrize("text", ["shard-1", "R1"])
def test_a_text_naming_a_non_checkpoint_artifact_is_not_a_model(world, ask, text):
    """The kind rule: only a registered checkpoint counts (plan line 181 (a), "a checkpoint
    registered in the store"). A shard name and a release name are not models."""
    kit.set_model(world, "summary", text)
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == (
        "may contain", NOT_REGISTERED, text)


# --- A log with no model entry ----------------------------------------------------------------------


def test_a_log_with_no_model_entry_gives_may_contain_naming_the_job(world, ask):
    """Plan line 181 (b) and line 228 (b): a log with no model entry is "not established" and the
    answer names the job; table rows c05.g.model-entry-removed.on.E/shard-3 and c07.R3.GM. The log
    stays marked compared, so its bound stands and (h) is reached."""
    kit.set_model(world, "summary", None)
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), *kit.link(shown, "condition", "job")) == (
        "may contain", ENTRY_MISSING, "h", "summary")


def test_the_same_missing_entry_with_the_switch_off_gives_the_negative_with_the_note(world, ask):
    """Table row c05.g.model-entry-removed.off.E/shard-3 and plan line 181 (c). A guard of the
    off state, written because it pairs the test above; it is green on today's tree."""
    kit.set_model(world, "summary", None)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("notes")) == ("does not contain", [kit.NOTE_NOT_ASSESSED])


# --- A checkpoint with no record of its own ---------------------------------------------------------


def test_a_registered_checkpoint_whose_own_record_is_missing_gives_may_contain_naming_the_record(world, ask):
    """Decision 21, plan line 169: "not covered" only for an artifact whose own record is
    missing, and "may contain" for anything downstream of it. The checkpoint's segment is not
    registered (the way tests/coverage/test_coverage_checkpoints.py makes an orphan)."""
    world.store.register_checkpoint(name="G@1", segment="G/seg-1", step=1)
    kit.set_model(world, "summary", "G@1")
    kit.turn_on(world)
    # A precondition: the checkpoint's own answer is "not covered", reason "record missing".
    own = ask("A", "G@1", records=kit.SUMMARY_RECORDS)
    assert (own["answer"], own.get("reason")) == ("not covered", "record missing")
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), *kit.link(shown, "artifact", "reason")) == (
        "may contain", v.NAMES_UPSTREAM_NOT_COVERED, "G@1", "record missing")


# --- The list form of the model entry ---------------------------------------------------------------


def test_a_list_of_clean_models_lets_the_negative_stand(world, ask):
    """The list form is what the harness writes for two or more different models. Every listed
    model must pass: all clean, so the negative stands (plan line 228 (a), read for each model)."""
    kit.set_model(world, "summary", json.dumps(["E@3", "Y@2"]))
    kit.turn_on(world)
    # Preconditions: both checkpoints are clean for the request.
    assert ask("A", "E@3", records=kit.SUMMARY_RECORDS)["answer"] == "does not contain"
    assert ask("A", "Y@2", records=kit.SUMMARY_RECORDS)["answer"] == "does not contain"
    assert kit.summary(ask)["answer"] == "does not contain"


def test_a_list_with_one_exposed_model_names_it(world, ask):
    kit.set_model(world, "summary", json.dumps(["E@3", "E@4"]))
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == ("may contain", EXPOSED, "E@4")


def test_a_list_names_the_first_failing_model_in_listed_order(world, ask):
    """Plan line 183 (c): the first failing link. The unregistered model comes first."""
    kit.set_model(world, "summary", json.dumps(["hosted-model-x", "E@4"]))
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == (
        "may contain", NOT_REGISTERED, "hosted-model-x")


@pytest.mark.parametrize("text", ["[]", '["none"]', '["E@3","E@3"]'])
def test_a_list_that_assesses_no_model_does_not_let_the_negative_stand(world, ask, text):
    """Any other text that starts with "[" is judged as one text, which is not a registered
    checkpoint: nothing may pass with no model assessed (plan line 181 (a))."""
    kit.set_model(world, "summary", text)
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == (
        "may contain", NOT_REGISTERED, text)


# --- The temporal cycle (R-11) ----------------------------------------------------------------------


def test_a_cycle_through_the_models_own_history_is_a_named_error_and_prints_no_answer(world):
    """The translation job's model is E@4, and E@4's history reads T (step 4 reads R1:L3, a copy
    of T:L1). A request refused with a named error, exit code 2, no answer printed (the family
    of the other refusals), and the error names the checkpoint. The printed name is not asserted
    (it is the lead's), only that it is not the old refusal."""
    kit.set_model(world, "translation", "E@4")
    kit.turn_on(world)
    shown = kit.ask_or_error(world.store, "C", "gen/T")
    assert (
        shown["answer"] != "error: request refused",
        shown["answer"].startswith("error: "),
        "E@4" in str(shown.get("detail")),
        shown.get("exit_code") == 2,
    ) == (True, True, True, True)


def test_a_cycle_the_walk_does_not_meet_is_no_error(world, ask):
    """A recorded chain that shows "contains" never reaches condition (h) (plan line 183 (a),
    line 228 (c)), so the cycle is not met. A guard: green on today's tree."""
    kit.set_model(world, "translation", "E@4")
    kit.turn_on(world)
    assert (ask("A", "gen/T")["answer"], ask("A", "E@4")["answer"]) == ("contains", "contains")


def test_a_recorded_contains_in_the_artifact_of_the_cycle_keeps_the_answer(world, ask):
    """Plan line 183 (a): "A recorded chain showing contains wins over a failed condition
    elsewhere". File gen/Q holds a record whose model is the checkpoint M@1, whose history reads
    that very record (a cycle), and a second record made from src/C line 1 (a recorded chain to
    C). Withdraw C: the artifact answers "contains". Only an error raised at the outermost exit
    lets the second record win; a raise inside the model's nested answer would not. A guard on
    today's tree (no cycle is met there)."""
    kit.add_generated(world, "gen/Q", [("q-cycle", "none", "src/Z", 1), ("q-recorded", "none", "src/C", 1)])
    kit.add_history(world, "M", ww.ref(world["gen/Q"], 1))
    kit.set_model(world, "q-cycle", "M@1")
    kit.turn_on(world)
    assert ask("C", "gen/Q")["answer"] == "contains"


def test_a_cycle_met_under_a_models_recorded_contains_does_not_leak_to_its_outputs(world):
    """Plan line 183 (a) and line 228 (a), (c): a model checkpoint whose history reads both
    the cycle record of gen/Q and its recorded chain to C answers "contains" for C; the cycle
    does not rest on that answer. An output made by that checkpoint therefore answers "may
    contain" naming it, whichever question comes first and however often it is asked in one
    session, and is not refused as a cycle. Written after the code (a result of the reading
    of R-11, not a line of the spec's own table of tests); red on today's tree because the output
    is refused as a clear negative past a generation edge."""
    kit.add_generated(world, "gen/Q", [("q-cycle", "none", "src/Z", 1), ("q-recorded", "none", "src/C", 1)])
    kit.add_history(world, "M", [ww.ref(world["gen/Q"], 1), ww.ref(world["gen/Q"], 2)])
    kit.add_generated(world, "gen/N", [("n-job", "M@1", "src/Z", 2)])
    kit.set_model(world, "q-cycle", "M@1")
    kit.turn_on(world)
    request = requests.withdraw(world.store, "C")
    exposed, held = ("may contain", EXPOSED, "M@1"), ("contains", None, None)
    for order, expected in ((("gen/N", "M@1", "gen/N"), [exposed, held, exposed]),
                            (("M@1", "gen/N", "gen/N"), [held, exposed, exposed])):
        session = coverage.Session(world.store, request)
        asked = [kit.ask_or_error(world.store, "C", name, request=request, session=session) for name in order]
        assert [(a["answer"], a.get("names"), kit.link(a, "artifact")[0]) for a in asked] == expected, order


def test_a_cycle_met_under_a_records_own_contains_does_not_leak_to_an_output_that_may_contain(world):
    """Plan line 183 (a) and (c): a record whose own verdict is "contains" does not rest on a
    cycle that was met among its parents. gen/Y holds a record whose model M@1 has a history that
    reads that very record (a cycle). The record of gen/X is made from it and from src/C line 1
    (a recorded chain to C), so it answers "contains" for C although one parent met the cycle. The
    record of gen/O is handed the gen/X record and the one record of gen/U, whose model is not
    registered, and states it used only the second: the gen/X record is in scope only (plan
    Decision 25), and the stated parent fails condition (h), so gen/O answers "may contain"
    naming the unregistered model, and the answer does not rest on the cycle: no error. Only the
    drop of the cycle at the end of the gen/X record's own walk keeps it from reaching gen/O (the
    artifact's own verdict is not "contains", so the drop at the artifact's end does not apply).
    On today's tree the stated parent is clear and the answer names the in-scope-only record."""
    kit.add_generated(world, "gen/Y", [("y-cycle", "none", "src/Z", 1)])
    kit.add_history(world, "M", ww.ref(world["gen/Y"], 1))
    kit.set_model(world, "y-cycle", "M@1")
    kit.add_generated_from(world, "gen/X", "x-job", [("gen/Y", 1), ("src/C", 1)], [0, 1])
    kit.add_generated(world, "gen/U", [("u-job", "hosted-model-x", "src/Z", 2)])
    kit.add_generated_from(world, "gen/O", "o-job", [("gen/X", 1), ("gen/U", 1)], [1])
    kit.turn_on(world)
    shown = kit.ask_or_error(world.store, "C", "gen/O")
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == (
        "may contain", NOT_REGISTERED, "hosted-model-x")


def test_a_model_text_that_overflows_the_parser_is_one_text_that_is_not_registered(world, ask):
    """A model text of 100000 opening brackets is not a list the harness wrote (it cannot be
    parsed), so it reads as one text, as any text that starts with "[" and is not that list does
    (TK-Q7b): not a registered checkpoint, "generating model not registered" naming the text.
    The parser's recursion limit is no new refusal."""
    text = "[" * 100_000
    # A precondition, not the assertion of this test: the parser overflows on this text here, so
    # the test pins the guard and not just any text the parser rejects.
    with pytest.raises(RecursionError):
        json.loads(text)
    kit.set_model(world, "summary", text)
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == (
        "may contain", NOT_REGISTERED, text)


def test_the_cycle_error_does_not_depend_on_the_order_of_the_questions(world):
    """W6: no "does not contain" without the proof. In either order of the two questions both
    give a named error (not the old refusal, exit code 2, no answer), and a question that never
    reaches the cycle is still answered in the same session afterwards (the set of checkpoints
    in progress was released)."""
    kit.set_model(world, "translation", "E@4")
    kit.turn_on(world)
    request = requests.withdraw(world.store, "C")
    for order in (("E@4", "gen/T"), ("gen/T", "E@4")):
        session = coverage.Session(world.store, request)
        shown = [kit.ask_or_error(world.store, "C", name, request=request, session=session) for name in order]
        assert [(s["answer"] != "error: request refused", s["answer"].startswith("error: "), s.get("exit_code"))
                for s in shown] == [(True, True, 2), (True, True, 2)], order
        after = kit.ask_or_error(world.store, "C", "shard-1", request=request, session=session)
        assert after["answer"] == "does not contain", order


# --- The depth of (h) (S-Q10b) -------------------------------------------------------------------------


def test_a_models_own_negative_resting_on_an_exposed_model_two_levels_down_is_may_contain(world, ask):
    """Plan line 193 and line 228 (a): the checkpoint is answered as any checkpoint, by the same
    proof. gen/N is made by a job whose model is E@4; E@4 read T, whose job names an unregistered
    model, so E@4's own answer for C is "may contain" and the output past E@4 names E@4. A reading
    that judged one level only would answer "does not contain" here."""
    kit.add_generated(world, "gen/N", [("nested", "E@4", "src/Z", 1)])
    kit.set_model(world, "translation", "hosted-model-x")
    # A precondition: with the switch off E@4 answers "does not contain" for C.
    assert ask("C", "E@4")["answer"] == "does not contain"
    kit.turn_on(world)
    shown = ask("C", "gen/N")
    assert (shown["answer"], shown.get("names"), kit.link(shown, "artifact")[0]) == ("may contain", EXPOSED, "E@4")


def test_a_models_own_negative_clean_all_the_way_down_lets_the_output_stand(world, ask):
    """The same store with the translation job's model "none": E@4 is clean at both levels, so
    gen/N stands. The case a one-level reading would wrongly turn into "may contain"."""
    kit.add_generated(world, "gen/N", [("nested", "E@4", "src/Z", 1)])
    kit.turn_on(world)
    assert ask("C", "gen/N")["answer"] == "does not contain"


def test_a_chain_that_shows_contains_keeps_contains_with_the_switch_on(world, ask):
    """Plan line 228 (c) and Decision 28: an output with a confirmed affected record among its
    parents keeps "contains". A guard: green on today's tree."""
    kit.turn_on(world)
    assert ask("A", "gen/T")["answer"] == "contains"


def test_a_recorded_chain_wins_over_an_exposed_model_in_the_same_artifact(world, ask):
    """Plan line 183 (a) and Decision 28. File gen/N holds one record made from src/A line 1
    (a recorded chain to A) and one made from Z whose model E@4 is exposed. Withdraw A: the
    artifact answers "contains". A guard: green on today's tree."""
    kit.add_generated(world, "gen/N", [("n-chain", "none", "src/A", 1), ("n-model", "E@4", "src/Z", 1)])
    kit.turn_on(world)
    assert ask("A", "gen/N")["answer"] == "contains"


# --- The loud checks reach the models' histories (R-10) -----------------------------------------------


def test_a_loud_fault_in_the_models_history_stops_the_outputs_answer(world, ask):
    """Plan line 173 (a), line 183 (b) and line 228 (a): the proof walks the ancestor closure and
    four faults stop the query loudly. E@4's history reads shard-3 (through mix-2); the closure of
    gen/S does not hold it. One byte appended to the file of shard-3."""
    kit.set_model(world, "summary", "E@4")
    kit.append_byte(world.folder / "shards" / "shard-3.bin")
    # A precondition: with the switch off the fault lies outside gen/S's closure.
    assert kit.summary(ask)["answer"] == "does not contain"
    kit.turn_on(world)
    assert kit.summary(ask)["answer"] == f"error: {errors.ArtifactDigestMismatch.name}"


def test_a_pass_made_before_the_switch_was_turned_on_does_not_cover_the_models_history(world):
    """An integrity pass holds only while the store is unwritten, and writing the switch is a
    write; the loud checks then run again over the closure with the model's history."""
    kit.set_model(world, "summary", "E@4")
    kit.append_byte(world.folder / "shards" / "shard-3.bin")
    request = requests.withdraw(world.store, "A", records=kit.SUMMARY_RECORDS)
    passed = coverage.check_store(world.store, [world["gen/S"]])
    kit.turn_on(world)
    shown = kit.ask_or_error(world.store, "A", "gen/S", request=request, integrity=passed)
    assert shown["answer"] == f"error: {errors.ArtifactDigestMismatch.name}"


def test_the_closure_reaches_the_model_of_a_model(world, ask):
    """The output O is made by a job whose model is M1@1; M1@1's history reads a record of gen/Q1,
    whose job's model is M2@1; the fault is in the one shard that only M2@1's history holds. The
    closure takes the model's ancestry again for the logs found in it, to a fixpoint."""
    kit.add_history(world, "M2", world.ref("src/Z", 2))
    kit.add_generated(world, "gen/Q1", [("q1job", "M2@1", "src/Z", 1)])
    kit.add_history(world, "M1", world.ref("gen/Q1", 1))
    kit.add_generated(world, "gen/O", [("ojob", "M1@1", "src/Z", 3)])
    kit.append_byte(world.folder / "shards" / "shard-M2.bin")
    # A precondition: with the switch off O answers by the core.
    assert ask("A", "gen/O")["answer"] == "does not contain"
    kit.turn_on(world)
    assert ask("A", "gen/O")["answer"] == f"error: {errors.ArtifactDigestMismatch.name}"


def test_a_file_missing_from_the_models_history_is_listed_in_the_scope_with_the_switch_on(world, ask):
    """Plan line 173 ("the artifact's ancestor closure"): the scope lists the files of the
    closure that could not be compared. The summary job's model is E@3, whose history reads
    shard-1, whose map names records of R2 and so src/B; src/B is not in gen/S's own closure (the
    spec built this with Y@2, whose own closure already holds every shard, so it could not tell
    the two states apart; see the report)."""
    kit.set_model(world, "summary", "E@3")
    (world.folder / "src/B.jsonl").rename(world.folder / "src/B.archived")
    # A precondition: with the switch off the closure is today's and names nothing.
    assert kit.summary(ask)["scope"]["files_not_checked"] == []
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], (shown.get("scope") or {}).get("files_not_checked")) == ("does not contain", ["src/B"])


# --- A publisher's pointer (R-13) -----------------------------------------------------------------------


def with_a_pointer(world) -> None:
    """The mapped-edge store of tests/coverage/test_coverage_proof.py: a text-only file whose
    row L1 a verified publisher's pointer ties to src/A line 1."""
    ww.write_jsonl(world.folder / "pointed.jsonl", ["made from X"])
    instance = capture.import_jsonl(world.store, world.folder / "pointed.jsonl", name="import/M",
                                    member_name="legacy/M")
    world.store.add_recovered_edges(instance, [EdgeRecord(
        "L1", (EdgeInput(world.ref("src/A", 1)),), ww.step("publisher-pointer"),
        derivation=v.DERIVED_POINTER, verification=v.VERIFIED_PASSED, label=v.MAPPED)])


def test_a_pointer_only_edge_names_the_unregistered_model_with_the_switch_on(world, ask):
    """Plan line 228 (d): where a publisher's pointer is the only evidence the generating model
    is not registered, so (h) cannot be met and the outputs of a mapped edge answer "may contain"
    naming the model as unregistered, for a source the edge does not name."""
    with_a_pointer(world)
    kit.turn_on(world)
    shown = ask("C", "import/M")
    assert (shown["answer"], shown.get("names"), kit.link(shown, "condition")[0]) == (
        "may contain", NOT_REGISTERED, "h")


def test_a_pointer_only_edge_with_the_switch_off_keeps_condition_g(world, ask):
    """Plan line 181 (c): with the switch off (h) is not applied; the answer is the core's, the
    bound is not established. A guard: green on today's tree."""
    with_a_pointer(world)
    shown = ask("C", "import/M")
    assert (shown["answer"], shown.get("names"), kit.link(shown, "condition")[0]) == (
        "may contain", v.NAMES_BOUND_NOT_ESTABLISHED, "g")


# --- The order of conditions: (g) before (h) (the owner's answer on compared texts) ---------------------------


def test_a_log_with_no_model_entry_and_no_compared_texts_names_the_bound_first(world, ask):
    """A log whose texts were not shown and compared establishes no bound ("may contain" until
    they are shown, then the clean answer returns), so condition (g) fails before (h) is
    reached, whatever the model entry holds. A guard that fixes the order: green on today's tree."""
    kit.set_model(world, "summary", None)
    kit.unmark(world, "summary")
    kit.turn_on(world)
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), *kit.link(shown, "condition", "job")) == (
        "may contain", v.NAMES_BOUND_NOT_ESTABLISHED, "g", "summary")


def test_showing_the_texts_brings_the_missing_entry_answer_back(world, ask):
    """Once the texts are shown and compared the bound stands, and the answer that (g) had
    blocked is the next one: the model entry is missing."""
    kit.set_model(world, "summary", None)
    kit.unmark(world, "summary")
    kit.turn_on(world)
    handed = [world.ref("src/A", 2), world.ref("src/C", 1)]
    world.store.verify_hand_off("summary", [(r, ww.TEXTS[(r.instance_id, r.locator)]) for r in handed])
    shown = kit.summary(ask)
    assert (shown["answer"], shown.get("names"), kit.link(shown, "condition")[0]) == (
        "may contain", ENTRY_MISSING, "h")
