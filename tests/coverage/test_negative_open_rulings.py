"""Three cases of "does not contain" whose rule the plan words and the build has not
settled. They live in their own file so that a ruling that reads the plan another way
drops the test whole.

- A record that a generation edge made from no input (plan 2.3: an occurrence with
  no recorded origin is never "does not contain"; 2.2(a)).
- A checkpoint's step whose sample is marked whole while the trace rows that should
  cover it are gone (plan 2.2(e), 3.5: a step with an uncovered range is untraced).
- A proof made before a write, used after it (plan 2.2: the walk is made on demand
  from the store as it is).
"""

from __future__ import annotations

import ww_world

from waterwhite import coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Answer, Proof

REFUSALS = (errors.WaterwhiteError, ValueError, TypeError, RuntimeError)


def attempt(call):
    """(result, None) when the call returned, (None, error) when the tool refused."""
    try:
        return call(), None
    except REFUSALS as error:
        return None, error


def link(answer: dict, *keys: str) -> dict:
    return {key: answer["failing"].get(key) for key in keys}


def test_a_record_a_generation_edge_made_from_no_input_has_no_recorded_origin(world, ask):
    """Plan 2.3 says an occurrence whose chain does not end at an ingested revision
    fails condition (a) at that occurrence and is never "does not contain" while it has
    no recorded origin. The job here was handed nothing and the edge names no parent."""
    world.ids["gen/N"] = ww_world.record(
        world.store, world.folder / "gen/N.jsonl", ["made from nothing"], "gen/N",
        v.ROLE_INTERMEDIATE, ww_world.step("job-nothing"), [world["src/Z"]],
        [ww_world.generation_edge(1, "job-nothing", [], [])],
    )
    for source in ("A", "Z"):
        answer = ask(source, "gen/N")
        assert (answer["answer"], answer["names"]) == ("may contain", v.NAMES_MISSING_ORIGIN), source
        assert link(answer, "condition", "artifact", "record") == {
            "condition": "a", "artifact": "gen/N", "record": "L1"}, source


def test_a_step_whose_sample_is_marked_whole_but_has_no_rows_is_not_a_negative(world, ask):
    """E reads T at step 4 (plan 2.4: "contains" while the rows are there). The trace
    rows of that step are removed and its sample keeps the mark "whole". The rows no
    longer show what the step read, so no checkpoint whose history holds the step
    answers "does not contain": it is a loud fault or "may contain"."""
    assert ask("A", "E@4")["answer"] == "contains"
    world.sql(("DELETE FROM trace_row WHERE trace_id = ?", (world.trace_id("E/seg-2", 4),)))
    assert world.scalar("SELECT covered_tokens FROM trace_sample WHERE trace_id = ?",
                        (world.trace_id("E/seg-2", 4),)) == ww_world.SAMPLE_TOKENS
    for name in ("E@4", "E@6", "D@2"):
        try:
            answer = ask("A", name)
        except errors.WaterwhiteError:
            continue  # a loud stop is a refusal to answer
        assert answer["answer"] == "may contain", (name, answer["answer"])


def test_a_proof_made_before_a_write_does_not_build_an_answer_after_it(world):
    store = world.store
    request = requests.withdraw(store, "A")
    proof = coverage.prove(store, "R2", request)
    assert isinstance(proof, Proof)
    # R2's first record now carries a file-level link to A: for a request about the
    # whole of A the store says R2 holds something of A (plan 3.2).
    store.add_source_link(
        world["R2"], "L1", "A/r1", v.FILE_LEVEL, "file family: records drawn from source A",
        derivation=v.DERIVED_FILE)
    assert coverage.answer(store, "R2", request).answer == v.CONTAINS
    shown, _ = attempt(lambda: Answer.from_proof(proof, artifact="R2"))
    assert shown is None, "the proof of the store before the write was shown as its answer after it"
