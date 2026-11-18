"""The action of each artifact, the counts, the mixtures and the groups of the rebuild
report (plan 4.1; docs/interfaces.md section 7.1, docs/rebuild.md sections 2.2 and
2.3), on the hand-built world of ``rb_kit.py`` and through ``report_without_restarts``.

Under "withdraw A" the world's R1 is rewritten and R2 kept; shard-1 (R2's records)
is kept and shard-3 (T and two rows of Z) rewritten; mix-1 (shard-1) is kept and
mix-2 (shard-1 and shard-3) needs a successor.
"""

from __future__ import annotations

import pytest
from waterwhite import errors

from rb_kit import (
    DOCUMENT_NAME, actions_of, artifact_entry, need_attr, report_parts, sizes_of,
)

ORDER = ["R1", "R2", "shard-1", "shard-3", "mix-1", "mix-2"]


def test_the_artifacts_with_an_entry_are_the_releases_the_shards_and_the_mixtures(world):
    document = report_parts(world)
    # Source files and generation outputs have none; the order is the registration's.
    assert [e["artifact"] for e in document["artifacts"]] == ORDER
    assert {e["artifact"]: e["action"] for e in document["artifacts"]} == {
        "R1": "rewrite", "R2": "keep", "shard-1": "keep", "shard-3": "rewrite",
        "mix-1": "keep", "mix-2": "rewrite",
    }
    assert {e["artifact"]: e["successor"] for e in document["artifacts"]} == {
        "R1": "successor of R1", "R2": None, "shard-1": None,
        "shard-3": "successor of shard-3", "mix-1": None, "mix-2": "successor of mix-2",
    }


def test_an_entry_gives_the_artifacts_own_answer_and_its_revision(world):
    document = report_parts(world)
    answers = {e["artifact"]: e["answer"] for e in document["artifacts"]}
    assert answers == {"R1": "contains", "R2": "does not contain", "shard-1": "does not contain",
                       "shard-3": "contains", "mix-1": "does not contain", "mix-2": "contains"}
    for entry in document["artifacts"]:
        assert entry["instance"] == world[entry["artifact"]]
    assert (artifact_entry(document, "R1")["kind"], artifact_entry(document, "R1")["role"]) == (
        "jsonl", "release")
    assert artifact_entry(document, "shard-3")["kind"] == "shard"
    assert artifact_entry(document, "mix-2")["kind"] == "mixture"


def test_a_kept_shard_that_holds_the_equivalent_copy_receives_nothing(world):
    # shard-1 holds B's copy of X through R2, so the promotion in R1 changes nothing in it.
    entry = artifact_entry(report_parts(world), "shard-1")
    assert entry["action"] == "keep" and not entry["receives"]


def test_a_rewritten_shard_counts_its_documents_and_the_ones_that_leave(world):
    entry = artifact_entry(report_parts(world), "shard-3")
    assert entry["documents"] == 3
    assert entry["records_removed"] == 1 and len(entry["removed"]) == 1
    assert DOCUMENT_NAME.match(entry["removed"][0])
    assert entry["documents_in_successor"] == 2


def test_a_mixture_needs_a_successor_exactly_when_its_snapshot_names_a_rewritten_shard(world):
    document = report_parts(world)
    assert artifact_entry(document, "mix-2")["rewritten_shards"] == ["shard-3"]
    assert artifact_entry(document, "mix-2")["needs_successor"] is True
    assert not artifact_entry(document, "mix-1")["needs_successor"]
    assert document["mixtures"] == {"needs_successor": ["mix-2"], "kept": ["mix-1"], "unresolved": [],
                                    "place_not_stated": []}


def test_the_counts_of_the_report(world):
    counts = report_parts(world)["counts"]
    assert counts["records"]["re-source"] == 1 and counts["records"]["promote"] == 1
    assert counts["records"]["retire"] == 3 and counts["records"]["unresolved"] == 0
    assert counts["records_removed"] == 3
    assert counts["shards"] == {"keep": 1, "rewrite": 1}
    assert counts["mixtures"] == {"keep": 1, "rewrite": 1}
    assert counts["bytes_unknown"] == [] and counts["unresolved_artifacts"] == []


def test_the_bytes_to_rewrite_are_read_from_the_files_of_the_artifacts_that_are_rewritten(world):
    document = report_parts(world)
    release = sizes_of(world.folder, "R1.jsonl")
    shard = sizes_of(world.folder, "shards/shard-3")
    assert artifact_entry(document, "R1")["bytes_to_rewrite"] == release
    assert artifact_entry(document, "shard-3")["bytes_to_rewrite"] == shard
    assert document["counts"]["bytes_to_rewrite"] == release + shard


def test_the_bytes_of_a_shard_whose_file_is_gone_are_unknown_and_left_out_of_the_sum(world):
    release = sizes_of(world.folder, "R1.jsonl")
    for suffix in (".bin", ".idx"):
        path = world.folder / "shards" / f"shard-3{suffix}"
        path.rename(path.with_name(f"moved-shard-3{suffix}"))
    document = report_parts(world)
    assert artifact_entry(document, "shard-3")["bytes_to_rewrite"] is None
    assert document["counts"]["bytes_unknown"] == ["shard-3"]
    assert document["counts"]["bytes_to_rewrite"] == release


# --- An artifact that holds an unresolved record is not kept ------------------------------


def test_an_artifact_that_holds_an_unresolved_record_and_none_in_scope_is_unresolved(world):
    # The edge of the last row of R2 is gone: the row has no recorded origin.
    world.delete_edge("R2", 3)
    document = report_parts(world)
    entry = artifact_entry(document, "R2")
    assert entry["action"] == "unresolved" and entry["successor"] is None
    assert actions_of(document, "R2") == {"R2:L1": "keep", "R2:L2": "keep", "R2:L3": "unresolved"}
    assert "R2" in document["counts"]["unresolved_artifacts"]


def test_an_artifact_whose_only_records_in_scope_are_unresolved_is_unresolved(world):
    world.delete_edge("gen/P", 1)
    document = report_parts(world)
    # T is unresolved, so shard-3 holds nothing shown in scope; mix-2 holds shard-3.
    assert artifact_entry(document, "shard-3")["action"] == "unresolved"
    assert artifact_entry(document, "mix-2")["action"] == "unresolved"
    # R1 holds records that are shown in scope (X, S), so it is still rewritten.
    assert artifact_entry(document, "R1")["action"] == "rewrite"
    assert set(document["counts"]["unresolved_artifacts"]) == {"shard-3", "mix-2"}
    assert document["counts"]["shards"] == {"keep": 1, "rewrite": 0}
    assert document["mixtures"] == {"needs_successor": [], "kept": ["mix-1"], "unresolved": ["mix-2"],
                                    "place_not_stated": []}


# --- The groups ----------------------------------------------------------------------------


def test_the_shards_are_grouped_by_the_runs_whose_mixtures_hold_them(world):
    groups = report_parts(world)["groups"]
    assert groups["by_layout"] == {}
    by_run = groups["by_run"]
    assert set(by_run) == {"E", "D", "Y"}
    assert by_run["E"] == {"counts": {"keep": 1, "rewrite": 1}, "keep": ["shard-1"],
                           "rewrite": ["shard-3"], "unresolved": []}
    assert by_run["Y"] == by_run["E"]
    assert by_run["D"] == {"counts": {"keep": 1, "rewrite": 0}, "keep": ["shard-1"],
                           "rewrite": [], "unresolved": []}


def test_the_shards_are_grouped_by_the_layouts_the_caller_names(world):
    groups = report_parts(world, layouts={"low": "shard-"})["groups"]
    assert groups["by_layout"] == {"low": {"counts": {"keep": 1, "rewrite": 1}, "keep": ["shard-1"],
                                           "rewrite": ["shard-3"], "unresolved": []}}


def test_a_shard_that_two_layouts_match_is_a_usage_error(world):
    function = need_attr("waterwhite.rebuild", "report_without_restarts")
    from waterwhite import coverage, requests

    request = requests.withdraw(world.store, "A")
    session = coverage.Session(world.store, request)
    with pytest.raises(errors.UsageError):
        function(world.store, request, session, layouts={"a": "shard-", "b": "shard-1"})
