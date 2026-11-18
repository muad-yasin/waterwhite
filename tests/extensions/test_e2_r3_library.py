"""Release R3 in the library world (File 2 of the spec of E2).

The base is the world of tests/rebuild/rb_kit.py (the library world with the dedup drop of R1),
with the hand-built R3 of ``e2_kit.add_r3_records``: file gen/R3 holds four generation outputs
(jobs r3-g6, r3-g9, r3-gu, r3-gm), each handed one record of Z, every log registered with its
texts and marked compared; release R3 copies the four lines (R3:L1 to R3:L4). The models are
E@3 for the first (a checkpoint that answers "does not contain" for A), E@4 for the second (a
checkpoint that answers "contains" for A), a text that names no registered artifact for the
third, and no model entry for the fourth. Plan 5.1 has the checkpoints 6 and 9 for the first two
(a known difference of this test device: the library world has no such checkpoints).

The plan fixes the four answers (plan line 302): the first record answers "does not contain", and
the other three each "may contain", naming the exposed model, the unregistered model and the
job with its missing model entry in turn, with the three listed as unresolved in the report and
the release answering "may contain" as a whole.
"""

from __future__ import annotations

import json

from e2_kit import ask, r3_world, world  # noqa: F401  (fixtures)

import e2_kit as kit
from waterwhite import coverage, requests
from waterwhite import vocabulary as v

ww = kit.ww


def test_with_the_switch_off_r3_is_built_with_four_generation_edges_and_answers_with_the_note(r3_world):
    """A proof of the kit, green on today's tree. Plan 5.1: four jobs run in the closed-input
    harness, each with an established bound; the note is the off state's. It shows that a red
    in the tests below is the missing rule and not a mis-built R3, and that every log is marked
    (an unmarked log would drop its job from the edges)."""
    shown = kit.ask_or_error(r3_world.store, "A", "R3")
    assert (shown["answer"], shown.get("notes"), sorted((shown.get("scope") or {}).get("generation_edges", []))) == (
        "does not contain", [kit.NOTE_NOT_ASSESSED], ["r3-g6", "r3-g9", "r3-gm", "r3-gu"])


def test_the_four_records_of_r3_give_the_tables_four_verdicts(r3_world):
    """Plan line 302: G6 does not contain, and G9, GU and GM each may contain, naming (in turn)
    the exposed checkpoint, the unregistered model and the missing model entry."""
    kit.turn_on(r3_world)
    session = coverage.Session(r3_world.store, requests.withdraw(r3_world.store, "A"))
    found = []
    for line in range(1, 5):
        status = session.occurrence(ww.ref(r3_world["R3"], line))
        found.append((coverage.is_clear(status), status.failing.names if status.failing is not None else None))
    assert found == [
        (True, None),
        (False, v.NAMES_MODEL_EXPOSED),
        (False, v.NAMES_MODEL_NOT_REGISTERED),
        (False, v.NAMES_MODEL_ENTRY_MISSING),
    ]


def test_r3_as_a_whole_may_contain(r3_world):
    """Plan line 302: the release R3 answers "may contain" as a whole."""
    kit.turn_on(r3_world)
    assert kit.ask_or_error(r3_world.store, "A", "R3")["answer"] == "may contain"


def test_the_report_lists_three_records_of_r3_as_unresolved_and_not_the_clean_one(r3_world):
    """Plan line 302 ("the three listed as unresolved in the report") and line 261 (4.1): the
    report lists a record whose generating model fails condition (h) as unresolved. The report's
    own action words, not an answer."""
    kit.turn_on(r3_world)
    code, out, err = kit.rb.run(r3_world, "report", "withdraw", "A")
    assert code == 0, err
    actions = kit.rb.actions_of(json.loads(out), "R3")
    assert actions == {"R3:L1": "keep", "R3:L2": "unresolved", "R3:L3": "unresolved", "R3:L4": "unresolved"}


def test_the_report_names_the_cause_of_each_unresolved_record(r3_world):
    """Plan line 302 and line 261: the entry of an unresolved record names its cause and its
    failing link (docs/interfaces.md, the record entry: ``names`` and ``failing``). For the
    unregistered model the link carries the model's text."""
    kit.turn_on(r3_world)
    code, out, err = kit.rb.run(r3_world, "report", "withdraw", "A")
    assert code == 0, err
    entries = [e for e in kit.rb.record_entries(json.loads(out), "R3") if e["action"] == "unresolved"]
    entries.sort(key=lambda e: int(e["record"].rsplit(":L", 1)[1]))
    causes = [(e.get("names"), (e.get("failing") or {}).get("condition")) for e in entries]
    assert causes == [
        (v.NAMES_MODEL_EXPOSED, "h"),
        (v.NAMES_MODEL_NOT_REGISTERED, "h"),
        (v.NAMES_MODEL_ENTRY_MISSING, "h"),
    ]
    assert len(entries) == 3 and (entries[1].get("failing") or {}).get("artifact") == kit.R3_UNREGISTERED
