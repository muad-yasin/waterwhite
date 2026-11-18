"""With the switch off (and every other extension off) the generating-model rule changes nothing
(File 4 of the spec of E2; the "Off state" section).

The claim is the plan's: while the extension is off, condition (h) is not applied, the model is a
logged attribute of the edge and not an ancestor, and a negative that passes a generation edge
rests on conditions (a) to (g) and carries the words "generating-model exposure not assessed"
(plan line 168 and line 181 (c)). Every test here is green on the tree before the code, as a
guard: they pin today's behaviour, they are not evidence that the extension works.

This round builds the core and this extension only. The write-time refusal of a record-level
cycle is not built (the cut list), so the third assertion of the test of the cycle and the
closure keeps the form "the write is accepted with the switch off" and nothing about the switch on.
"""

from __future__ import annotations

from e2_kit import acceptance, ask, world  # noqa: F401  (fixtures)

import e2_kit as kit

ww = kit.ww

QUESTIONS = [
    ("C", "shard-3", ()),
    ("A", "gen/S", ("src/A:L1",)),
    ("A", "gen/P", ()),
    ("C", "gen/P", ()),
]


def test_no_setting_row_and_an_explicit_off_answer_alike(world):
    """Plan 1.4 (the off column) and plan line 181 (c): the answer with no
    ``extension.generating-model`` row equals, in every field, the answer with the row set to
    "off". The equality is the claim."""
    without = [kit.ask_or_error(world.store, s, a, records=r) for s, a, r in QUESTIONS]
    world.store.set_setting(kit.SWITCH, "off")
    with_off = [kit.ask_or_error(world.store, s, a, records=r) for s, a, r in QUESTIONS]
    assert without == with_off


def test_the_off_state_gives_the_note_on_a_negative_past_a_generation_edge(ask):
    """Table row c05.withdraw-C.unmutated.E/shard-3 with plan line 168."""
    shown = ask("C", "shard-3")
    assert (shown["answer"], shown["notes"]) == ("does not contain", [kit.NOTE_NOT_ASSESSED])


def test_with_the_switch_off_neither_the_cycle_rule_nor_the_model_closure_fires(world, ask):
    """Plan line 181 (c): the model is a logged attribute and not an ancestor, so no cycle is
    possible through it and the model's history is not in the closure. Three assertions, each
    its own statement, no switch row."""
    # (a) The store of the cycle: the translation job's model is E@4, whose history reads T.
    kit.set_model(world, "translation", "E@4")
    shown = ask("C", "gen/T")
    assert (shown["answer"], shown["notes"]) == ("does not contain", [kit.NOTE_NOT_ASSESSED])
    assert {entry["job"]: entry["model"] for entry in shown["scope"]["harness_logs"]}["translation"] == "E@4"
    # (b) The store of the fault in a model's history: a byte appended to shards/shard-3.bin,
    # the summary job's model E@4.
    kit.set_model(world, "summary", "E@4")
    kit.append_byte(world.folder / "shards" / "shard-3.bin")
    shown = kit.summary(ask)
    assert (shown["answer"], shown["notes"]) == ("does not contain", [kit.NOTE_NOT_ASSESSED])
    # (c) The writes that close a record-level cycle are accepted with the switch off: a file
    # whose log names the checkpoint W@1 as its model, then the registration of W@1, whose history
    # reads a record of that file.
    kit.add_generated(world, "gen/W", [("w-edge", "W@1", "src/Z", 1)])
    kit.add_history(world, "W", world.ref("gen/W", 1))
    assert (world.store.harness_log("w-edge").model, world.store.artifact("W@1") is not None) == ("W@1", True)


ROWS = [
    "c05.record-X@A.unmutated.gen/S",
    "c05.record-X@B.unmutated.gen/P",
    "c05.withdraw-C.unmutated.E/shard-3",
    "c05.g.model-entry-removed.off.E/shard-3",
]


def test_the_four_off_state_rows_give_the_tables_answers_with_and_without_an_off_row(acceptance):
    """The four rows, each asked of a private copy with no setting row and again with the row
    "off". Per row: the mismatches against the row's own ``expect`` are empty on both, and the
    two printed answers are equal. Skips with the acceptance helpers' own words."""
    table = acceptance.mods.table
    world = acceptance.world
    rows = {row.id: row for row in table.load_check(5)}
    for row_id in ROWS:
        question = table.interpret(rows[row_id])
        arguments = world.query_arguments(question)
        resolve = lambda ref, q=question: world.locator(q.store, ref)  # noqa: E731
        built = world.store_of(question)
        plain = acceptance.private_copy(built)
        explicit = acceptance.private_copy(built)
        acceptance.switch(explicit, "off")
        outcomes = [world.cli(path, *arguments) for path in (plain, explicit)]
        mismatches = [table.compare_outcome(question.expect, outcome, resolve) for outcome in outcomes]
        assert (mismatches, outcomes[0].printed() == outcomes[1].printed()) == ([[], []], True), row_id
