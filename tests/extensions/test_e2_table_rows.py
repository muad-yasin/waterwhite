"""The rows of the frozen table that belong to the generating-model rule (File 3 of the spec of E2).

These are acceptance-level tests: the store is the replica's build under the key
``main+generating-model`` (the main build, then release R3), tokenized by Megatron, with the runs
of main registered, so they need the Megatron environment and the sandbox. Where either is
missing they skip with the acceptance helpers' own words (``acceptance_sets.MEGATRON_VARIABLE``
is not set; ``acceptance_probe.sandbox_missing()``), and a skipped test is not a pass.

Every expected value is the row's own ``expect`` in fixture/table/check05.toml and check07.toml,
read with ``tomllib`` by the frozen reader (``table.load_check``); the extra values of two rows
(the job of the record GM and the spelling of the unregistered model) are read from
fixture/sources/generation_r3.jsonl and the fixture spec with ``json`` and ``tomllib``. Nothing
is read from the tool to make an expectation. A row that is not in the table yet reports "row
missing" and fails: it is never a pass.

The acceptance modules are imported only inside the ``acceptance`` fixture of ``e2_kit``,
because ``acceptance_world`` runs command-line probes when it is imported.
"""

from __future__ import annotations

import pytest
from e2_kit import acceptance  # noqa: F401  (fixture)

import e2_kit as kit
from waterwhite import vocabulary as v
from waterwhite.store.store import Store


def test_with_the_switch_off_the_built_r3_has_four_generation_edges_and_every_log_marked(acceptance):
    """A proof of the build, red only where the build is missing. Plan 5.1 (four jobs run in the
    closed-input harness, each edge with an established bound) and the fixture file's job
    field. A log registered without its texts would drop its job from the list and turn the
    answer into "may contain" naming "upper bound not established"."""
    path = acceptance.store_file(kit.KEY_STORE)
    shown = acceptance.query(path, "query", "withdraw", "A", "--artifact", "R3")
    jobs = sorted(row["job"] for row in kit.r3_fixture_rows().values())
    assert (shown["answer"], shown.get("notes"), sorted(shown["scope"]["generation_edges"])) == (
        "does not contain", [v.NOTE_GENERATING_MODEL_NOT_ASSESSED], jobs)


@pytest.mark.parametrize("row_id", [
    "c05.g.model-entry-removed.on.E/shard-3",
    "c05.g.model-entry-removed.on.gen/P",
    "c05.g.model-entry-removed.on.gen/T",
])
def test_c05_g_model_entry_removed_on_answers_as_the_table_says(acceptance, row_id):
    """Plan line 300, line 181 (b), line 228 (b): the model entry removed from the stored log of
    P's job; with the rule on, the answer names the job and the missing entry. The three rows are
    the one of the plan and the two the table gains for gen/P and gen/T."""
    assert acceptance.put_row(row_id) == []


def test_c07_R3_G6(acceptance):
    assert acceptance.put_row("c07.R3.G6") == []


def test_c07_R3_G9(acceptance):
    """Checkpoint 9's own answer for A is "contains" (row c08.withdraw-A.E@9, expected value
    unchanged): the output row is not the oracle, the checkpoint's row is."""
    # A precondition, asked of the same store, and not the assertion of this test.
    assert acceptance.put_row("c08.withdraw-A.E@9", store=kit.KEY_STORE) == []
    assert acceptance.put_row("c07.R3.G9") == []


def test_c07_R3_GU(acceptance):
    """The printed failing.artifact is the fixture's unregistered model (a list of one text)."""
    assert acceptance.put_row("c07.R3.GU") == []
    entry = acceptance.r3_entry("GU")
    assert (entry.get("failing") or {}).get("artifact") == kit.r3_spec()["never_registered"][0]


def test_c07_R3_GM(acceptance):
    """The printed failing.job is the job of the GM row of the fixture file."""
    assert acceptance.put_row("c07.R3.GM") == []
    entry = acceptance.r3_entry("GM")
    assert (entry.get("failing") or {}).get("job") == kit.r3_fixture_rows()["GM"]["job"]


def test_c07_R3_release(acceptance):
    assert acceptance.put_row("c07.R3.release") == []


def test_the_built_gm_log_has_no_model_entry_and_is_marked_compared(acceptance):
    """Plan line 281 (a) and (b), and the owner's answer on compared texts: the stored log of the
    job of GM has model None and is marked; the logs of the other three jobs hold the model of
    their rows and are marked. It exists so that a build that registers GM through ``read_log``
    alone (a log with no texts) fails at its own assertion."""
    rows = kit.r3_fixture_rows()
    store = Store.open(acceptance.store_file(kit.KEY_STORE))
    try:
        held = {handle: store.harness_log(row["job"]) for handle, row in rows.items()}
    finally:
        store.close()
    assert {handle: (log.model if log is not None else "no log", log is not None and log.hand_off_checked)
            for handle, log in held.items()} == {
        "G6": (rows["G6"]["model"], True),
        "G9": (rows["G9"]["model"], True),
        "GU": (rows["GU"]["model"], True),
        "GM": (None, True),
    }


# The model-history stop (a loud fault on a file only a model's history reads) has no table row: it is
# pinned by E2-W18 in test_e2_walk_rules.py (test_a_loud_fault_in_the_models_history_stops_the_outputs_answer).

def test_the_standing_negative_with_the_switch_on_answers_as_the_table_says(acceptance):
    """The row of the standing negative with the rule on (plan line 168, line 181 (a) and (c)):
    "does not contain", with no exposure note."""
    assert acceptance.put_row(kit.ROW_STANDING_NEGATIVE) == []
