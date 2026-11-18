"""What the report says about the per-layout count (plan 4.1, ledger row BF-7;
docs/interfaces.md section 7.1).

No layout is recorded in the store, so ``--layout NAME=PREFIX`` is the only source of
one. A report without it must not just leave the layouts out: ``groups.layouts`` says
whether the count was made, the reason when it was not, and the shards no named prefix
matched. ``groups.by_layout`` and ``groups.by_run`` stay as they were.
"""

from __future__ import annotations

import json

from rb_kit import report_parts, run

BY_RUN_E = {"counts": {"keep": 1, "rewrite": 1}, "keep": ["shard-1"], "rewrite": ["shard-3"],
            "unresolved": []}
BY_RUN_D = {"counts": {"keep": 1, "rewrite": 0}, "keep": ["shard-1"], "rewrite": [],
            "unresolved": []}


def statement(document: dict) -> dict:
    held = document["groups"].get("layouts")
    assert isinstance(held, dict), "groups holds no layouts statement"
    assert set(held) == {"counted", "reason", "unmatched_shards"}, sorted(held)
    return held


def test_a_report_made_without_a_layout_states_that_no_count_was_made_and_names_the_flag(world):
    groups = report_parts(world)["groups"]
    held = statement({"groups": groups})
    assert held["counted"] is False
    assert isinstance(held["reason"], str) and held["reason"]
    assert "--layout" in held["reason"]
    assert held["unmatched_shards"] == []
    # The per-layout groups and the per-run groups are as they were.
    assert groups["by_layout"] == {}
    assert groups["by_run"]["E"] == BY_RUN_E and groups["by_run"]["Y"] == BY_RUN_E
    assert groups["by_run"]["D"] == BY_RUN_D


def test_an_empty_set_of_layouts_is_no_layout_named(world):
    held = statement(report_parts(world, layouts={}))
    assert held["counted"] is False and held["reason"] and "--layout" in held["reason"]


def test_with_layouts_named_the_count_was_made_and_every_shard_matched(world):
    groups = report_parts(world, layouts={"low": "shard-"})["groups"]
    held = statement({"groups": groups})
    assert (held["counted"], held["reason"], held["unmatched_shards"]) == (True, None, [])
    assert list(groups["by_layout"]) == ["low"]
    assert groups["by_layout"]["low"]["keep"] == ["shard-1"]
    assert groups["by_layout"]["low"]["rewrite"] == ["shard-3"]


def test_a_shard_matching_none_of_the_layouts_named_is_reported_unmatched(world):
    groups = report_parts(world, layouts={"first": "shard-1"})["groups"]
    held = statement({"groups": groups})
    assert (held["counted"], held["reason"], held["unmatched_shards"]) == (True, None, ["shard-3"])
    # What by_layout holds is what it held: the one shard the prefix matches.
    assert groups["by_layout"] == {"first": {"counts": {"keep": 1, "rewrite": 0}, "keep": ["shard-1"],
                                             "rewrite": [], "unresolved": []}}
    assert groups["by_run"]["E"] == BY_RUN_E


def test_when_no_named_layout_matches_every_shard_is_listed(world):
    groups = report_parts(world, layouts={"other": "nothing-"})["groups"]
    held = statement({"groups": groups})
    assert held["counted"] is True and held["reason"] is None
    assert held["unmatched_shards"] == ["shard-1", "shard-3"]
    assert groups["by_layout"] == {"other": {"counts": {"keep": 0, "rewrite": 0}, "keep": [],
                                             "rewrite": [], "unresolved": []}}


def test_the_command_says_it_too_with_and_without_the_flag(world):
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    without = statement(json.loads(out))
    assert without["counted"] is False and "--layout" in without["reason"]
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "first=shard-1")
    assert code == 0, err
    document = json.loads(out)
    with_flag = statement(document)
    assert (with_flag["counted"], with_flag["reason"], with_flag["unmatched_shards"]) == (
        True, None, ["shard-3"])
    assert list(document["groups"]["by_layout"]) == ["first"]
