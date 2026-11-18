"""The action of each record in the rebuild report (plan 4.1, 4.2; docs/interfaces.md
section 7.1, docs/rebuild.md section 2.1), on the hand-built world of ``rb_kit.py``
and through ``report_without_restarts``.

The world: R1 holds X (dedup's winner A's copy, B's copy the dropped loser), P, T and
S (generated from A) and two rows of Z; under "withdraw A" the expected actions follow
from plan 4.1 and Decisions 10 and 15: X is re-sourced to B's copy and B's copy is
promoted, P, T and S are retired, the two Z rows are kept.
"""

from __future__ import annotations

from waterwhite import ids
from waterwhite import vocabulary as v

from rb_kit import actions_of, record_entries, record_entry, report_parts

R1_ROWS = [f"R1:L{n}" for n in range(1, 7)]


def test_r1_rows_get_the_actions_of_plan_4_1(world):
    document = report_parts(world)
    assert [e["record"] for e in record_entries(document, "R1")] == R1_ROWS + ["src/B:L1"]
    assert actions_of(document, "R1") == {
        "R1:L1": "re-source", "R1:L2": "retire", "R1:L3": "retire", "R1:L4": "retire",
        "R1:L5": "keep", "R1:L6": "keep", "src/B:L1": "promote",
    }


def test_the_re_source_names_both_parents_and_the_chain_that_made_it_necessary(world):
    entry = record_entry(report_parts(world), "R1", "R1:L1")
    assert (entry["new_parent"], entry["withdrawn_parent"]) == ("src/B:L1", "src/A:L1")
    assert entry["chain"][0]["hop"] == "R1:L1" and entry["chain"][-1]["hop"] == "src/A:L1"
    assert entry["regenerate_with"] is None and entry["step"] is None


def test_the_promote_is_the_dedup_loser_and_belongs_to_the_release_it_returns_to(world):
    entry = record_entry(report_parts(world), "R1", "src/B:L1")
    assert entry["action"] == "promote" and entry["artifact"] == "R1"
    # The chain is the one of the row whose selected parent was withdrawn.
    assert entry["chain"][0]["hop"] == "R1:L1" and entry["chain"][-1]["hop"] == "src/A:L1"


def test_a_retired_row_names_the_job_that_made_it_and_the_step_to_run_again(world):
    document = report_parts(world)
    jobs = {"R1:L2": "paraphrase", "R1:L3": "translation", "R1:L4": "summary"}
    ends = {"R1:L2": "src/A:L1", "R1:L3": "src/A:L1", "R1:L4": "src/A:L2"}
    for record, job in jobs.items():
        entry = record_entry(document, "R1", record)
        assert entry["action"] == "retire" and entry["regenerate_with"] == job, record
        # The step is the generation edge's own: name, version and the digest of its
        # parameters (the edge of the hand-built world states none).
        assert entry["step"] == {"name": "generate", "version": "1",
                                 "parameter_digest": ids.param_digest(None)}, record
        assert entry["chain"][0]["hop"] == record and entry["chain"][-1]["hop"] == ends[record], record


def test_the_chain_of_a_translation_runs_through_the_paraphrase_to_the_source(world):
    entry = record_entry(report_parts(world), "R1", "R1:L3")
    assert [hop["hop"] for hop in entry["chain"]] == ["R1:L3", "gen/T:L1", "gen/P:L1", "src/A:L1"]


def test_a_kept_row_has_an_empty_chain_and_no_parent_to_name(world):
    for record in ("R1:L5", "R1:L6"):
        entry = record_entry(report_parts(world), "R1", record)
        assert entry["chain"] == [] and entry["new_parent"] is None
        assert entry["withdrawn_parent"] is None and entry["regenerate_with"] is None


def test_an_artifact_that_is_kept_lists_no_record(world):
    assert record_entries(report_parts(world), "R2") == []


def test_the_counts_of_a_release_follow_from_its_actions(world):
    entry = next(e for e in report_parts(world)["artifacts"] if e["artifact"] == "R1")
    assert entry["records"] == 6
    assert entry["counts"] == {"keep": 2, "re-source": 1, "promote": 1, "retire": 3, "unresolved": 0}
    assert entry["records_removed"] == 3
    assert entry["removed"] == ["R1:L2", "R1:L3", "R1:L4"]
    # The rows of the successor: kept, re-sourced and unresolved. A promote adds none.
    assert entry["rows_in_successor"] == 3


# --- A release that holds no source id: the edge tells the two builds apart (D-06) --------


def test_two_releases_of_equal_bytes_get_the_actions_their_edges_give(swap_world):
    store = swap_world.store
    assert (store.artifact(swap_world["R1"]).content_digest
            == store.artifact(swap_world["R1-swap"]).content_digest)
    document = report_parts(swap_world)
    assert actions_of(document, "R1")["R1:L1"] == "re-source"
    assert actions_of(document, "R1-swap") == {
        "R1-swap:L1": "keep", "R1-swap:L2": "retire", "R1-swap:L3": "retire",
        "R1-swap:L4": "retire", "R1-swap:L5": "keep", "R1-swap:L6": "keep",
    }


def test_a_swapped_winner_is_neither_re_sourced_nor_promoted(swap_world):
    document = report_parts(swap_world)
    swapped = next(e for e in document["artifacts"] if e["artifact"] == "R1-swap")
    assert swapped["counts"] == {"keep": 3, "re-source": 0, "promote": 0, "retire": 3, "unresolved": 0}
    assert swapped["records_removed"] == 3 and swapped["rows_in_successor"] == 3
    assert not [e for e in record_entries(document, "R1-swap") if e["action"] in ("re-source", "promote")]


# --- A record that cannot be told apart is unresolved and has no action -------------------


def test_a_record_whose_chain_is_broken_is_unresolved_with_the_link_and_the_scope(world):
    # The edge of P in gen/P is gone: P has no recorded origin, and T, made from P,
    # has none either; "no chain was found" is not "shown clean".
    world.delete_edge("gen/P", 1)
    document = report_parts(world)
    assert actions_of(document, "R1") == {
        "R1:L1": "re-source", "R1:L2": "unresolved", "R1:L3": "unresolved", "R1:L4": "retire",
        "R1:L5": "keep", "R1:L6": "keep", "src/B:L1": "promote",
    }
    for record in ("R1:L2", "R1:L3"):
        entry = record_entry(document, "R1", record)
        assert entry["action"] == v.UNRESOLVED
        assert entry["names"] == v.NAMES_MISSING_ORIGIN
        assert entry["failing"]["condition"] == "a" and entry["failing"]["artifact"] == "gen/P"
        assert isinstance(entry["missing_link"], str) and entry["missing_link"]
        assert entry["review_scope"] == {"artifact": "R1", "records": v.SCOPE_THE_RECORD, "count": 1}
        assert entry["new_parent"] is None and entry["regenerate_with"] is None
        assert entry["chain"] == []


def test_an_unresolved_record_is_counted_and_adds_a_row_to_the_successor(world):
    world.delete_edge("gen/P", 1)
    entry = next(e for e in report_parts(world)["artifacts"] if e["artifact"] == "R1")
    assert entry["counts"] == {"keep": 2, "re-source": 1, "promote": 1, "retire": 1, "unresolved": 2}
    assert entry["records_removed"] == 1
    # kept (2) + re-sourced (1) + unresolved (2)
    assert entry["rows_in_successor"] == 5
