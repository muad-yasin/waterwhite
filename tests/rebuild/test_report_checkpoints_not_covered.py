"""A checkpoint that answers "not covered" is named in the report, with its reason,
whether or not a run reaches it (the M4 verdict: "Every registered checkpoint that
answers 'not covered' is named in the report with its reason ... Not in the exposed
list ... No table value moves").

The form (the M4 ruling; the docs do not state it yet): the report gains the
last top-level key ``checkpoints_not_covered``, a list in the order of registration of
``{"checkpoint", "segment", "step", "run", "answer", "reason"}``, always present, ``[]``
when none; the text form prints ``not covered: NAME (REASON)`` for each. A checkpoint
whose segment the store has no row for is the one a library call can make
(docs/store.md: "the segment is named by value and need not be registered"); the
coverage rule (docs/coverage.md section 7, "checkpoint: its run segment; not covered
when the segment has no row") answers "not covered", reason "record missing".

Expected values come from the call that registers the checkpoint, from the vocabulary's
definitions of the answer and the reason words, and from ``coverage.answer`` (outside
the rebuild package and the report part of the command line) for the partition.
"""

from __future__ import annotations

from waterwhite import coverage, requests
from waterwhite import vocabulary as v

from rb_kit import need_attr, report_through_cli, run

KEY = "checkpoints_not_covered"

# The five checkpoints of the world, from its construction (tests/coverage/ww_world.py
# lines 385 to 397).
WORLD_CHECKPOINTS = ["E@3", "E@4", "E@6", "D@2", "Y@2"]
ORPHANS = ["Z@1", "B@1"]


def entry_of(name: str, segment: str, step: int) -> dict:
    return {"checkpoint": name, "segment": segment, "step": step, "run": None,
            "answer": v.NOT_COVERED, "reason": v.REASON_RECORD_MISSING}


def two_orphans(world) -> list[dict]:
    """Two checkpoints over segments the store has no row for, registered in an order
    that is not the alphabetical one of their names, after the world's own."""
    world.store.register_checkpoint(name="Z@1", segment="ghost/seg-1", step=1)
    world.store.register_checkpoint(name="B@1", segment="phantom/seg-9", step=4)
    return [entry_of("Z@1", "ghost/seg-1", 1), entry_of("B@1", "phantom/seg-9", 4)]


def through_the_library(world) -> dict:
    build_report = need_attr("waterwhite.rebuild", "build_report")
    return build_report(world.store, requests.withdraw(world.store, "A"))


def through_the_command(world) -> dict:
    return report_through_cli(world)


def the_list(document: dict) -> list[dict]:
    assert KEY in document, f"the report has no {KEY}: {list(document)}"
    return document[KEY]


def names_of(entries: list[dict]) -> list[str]:
    return [entry["checkpoint"] for entry in entries]


def answers_of(world, names: list[str]) -> dict[str, str]:
    """The answer of each checkpoint, from coverage.answer."""
    request = requests.withdraw(world.store, "A")
    return {name: coverage.answer(world.store, name, request).answer for name in names}


def test_a_checkpoint_over_an_unregistered_segment_is_named_with_its_reason_by_the_library(world):
    expected = two_orphans(world)
    document = through_the_library(world)
    assert v.NOT_COVERED == "not covered" and v.REASON_RECORD_MISSING == "record missing"
    assert the_list(document) == expected
    assert list(document)[-2:] == ["exposed_checkpoints", KEY]
    for name in ORPHANS:
        assert name not in names_of(document["exposed_checkpoints"])
        assert name not in [entry["checkpoint"] for entry in document["restarts"]]


def test_a_checkpoint_over_an_unregistered_segment_is_named_with_its_reason_by_the_command(world):
    expected = two_orphans(world)
    document = through_the_command(world)
    assert the_list(document) == expected
    assert list(document)[-2:] == ["exposed_checkpoints", KEY]
    for name in ORPHANS:
        assert name not in names_of(document["exposed_checkpoints"])
        assert name not in [entry["checkpoint"] for entry in document["restarts"]]


def test_the_command_and_the_library_name_the_same_entries(world):
    two_orphans(world)
    assert the_list(through_the_command(world)) == the_list(through_the_library(world))


def test_every_registered_checkpoint_is_negative_or_exposed_or_not_covered(world):
    two_orphans(world)
    answers = answers_of(world, WORLD_CHECKPOINTS + ORPHANS)
    # The partition is not empty in any of its three parts.
    assert v.DOES_NOT_CONTAIN in answers.values(), answers
    assert v.NOT_COVERED in answers.values(), answers
    assert {v.CONTAINS, v.MAY_CONTAIN} & set(answers.values()), answers
    for document in (through_the_library(world), through_the_command(world)):
        exposed = names_of(document["exposed_checkpoints"])
        uncovered = names_of(the_list(document))
        assert len(uncovered) == len(set(uncovered))
        for name, answer in answers.items():
            placed = (name in exposed, name in uncovered)
            if answer == v.DOES_NOT_CONTAIN:
                assert placed == (False, False), name
            elif answer == v.NOT_COVERED:
                assert placed == (False, True), name
            else:
                assert answer in (v.CONTAINS, v.MAY_CONTAIN), name
                assert placed == (True, False), name
        assert set(uncovered) == {n for n, a in answers.items() if a == v.NOT_COVERED}


def test_the_checkpoints_of_the_world_are_not_in_the_list_and_without_an_orphan_it_is_empty(world):
    answers = answers_of(world, WORLD_CHECKPOINTS)
    assert v.DOES_NOT_CONTAIN in answers.values()
    assert {v.CONTAINS, v.MAY_CONTAIN} & set(answers.values())
    assert v.NOT_COVERED not in answers.values()
    for document in (through_the_library(world), through_the_command(world)):
        assert the_list(document) == []


def test_a_checkpoint_of_the_world_stays_out_of_the_list_beside_an_orphan(world):
    expected = two_orphans(world)
    document = through_the_command(world)
    assert the_list(document) == expected
    assert not set(names_of(the_list(document))) & set(WORLD_CHECKPOINTS)


def test_the_text_form_prints_one_line_for_each_checkpoint_that_is_not_covered(world):
    two_orphans(world)
    code, out, err = run(world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    lines = out.splitlines()
    wanted = [f"not covered: Z@1 ({v.REASON_RECORD_MISSING})",
              f"not covered: B@1 ({v.REASON_RECORD_MISSING})"]
    assert [line for line in lines if line.startswith("not covered:")] == wanted, out
    assert not [line for line in lines if line.startswith(("exposed: Z@1", "exposed: B@1"))]


def test_the_text_form_without_such_a_checkpoint_prints_no_such_line_and_the_json_names_none(world):
    assert the_list(through_the_command(world)) == []
    code, out, err = run(world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    assert not [line for line in out.splitlines() if line.startswith("not covered:")]
