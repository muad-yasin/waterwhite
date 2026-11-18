"""Stating where the layout places the records, and what the report then says
(DECISIONS.md D-141; docs/cli.md "report"; docs/interfaces.md 7.1; docs/rebuild.md 2.2).

``report withdraw A --layout NAME=PREFIX --placement NAME=FILE`` gives a layout's rule
as a JSON file: ``form`` "waterwhite.placement/1", ``by`` "source", ``shards`` (the source
artifact's name to the store name of the shard its records land in) and ``otherwise`` (a
store name or null). A returning copy (a record the report promotes, or re-sources onto)
lands in the shard named for its source: its own artifact where that is a registered
source file (``src/B:L1`` lands under ``src/B``), else the root of its single-parent
record-level chain, else ``otherwise``.

Under "withdraw A" the world's returning copy is ``src/B:L1``. The tests add a second one
where a test needs it: R3, a release whose first row is an "any" edge of A's copy and of
``mid/M:L1`` (a copy of ``src/B:L1``), is re-sourced onto ``mid/M:L1``, which roots in
``src/B``; R4 does the same onto ``mid/N:L1``, made from two parents, which roots nowhere.

The shard that receives a copy reads "rewrite", its ``receives`` names the copy, it gets
the successor name, and ``documents_in_successor`` is its documents, minus the ones that
leave, plus the ones it receives. Shards the store does not hold are named in
``groups.placements[NAME].landing`` and in no artifact entry.
"""

from __future__ import annotations

import json

import pytest
from waterwhite import vocabulary as v

from keep_kit import (
    B_COPY, WORD, add_file, add_mixture, add_run, add_shard, any_edge, entry, many_edge,
    names, parent_of, place_not_stated, placements_of, report_with, reported, sha256_of, single_edge,
    write_placement,
)
from rb_kit import artifact_entry, need_attr, run, sizes_of

X = {"x": "xs/"}
LOW = {"low": "shard-"}


def error_name(err: str) -> str:
    return json.loads(err)["error"]


# --- The flag ----------------------------------------------------------------------------------


def accepted(world, tmp_path) -> None:
    """The flag is one the command takes: a good placement for a named layout is read."""
    good = write_placement(tmp_path, "good", {"src/B": "shard-1"})
    code, document, err = report_with(world, LOW, {"low": good})
    assert code == 0, f"--placement is not accepted: {err}"


def test_a_placement_for_a_layout_that_is_not_named_is_a_usage_error(world, tmp_path):
    accepted(world, tmp_path)
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    code, document, err = report_with(world, {}, {"x": placement})
    assert (code, document, error_name(err)) == (1, None, "usage")
    code, document, err = report_with(world, LOW, {"x": placement})
    assert (code, document, error_name(err)) == (1, None, "usage")


@pytest.mark.parametrize("written", [
    "not json at all",
    json.dumps([1, 2]),
    json.dumps({"form": "waterwhite.placement/2", "by": "source", "shards": {}, "otherwise": None}),
    json.dumps({"form": "waterwhite.placement/1", "by": "line", "shards": {}, "otherwise": None}),
    json.dumps({"form": "waterwhite.placement/1", "by": "source", "shards": [], "otherwise": None}),
    json.dumps({"form": "waterwhite.placement/1", "by": "source", "shards": {"src/B": 7},
                "otherwise": None}),
    json.dumps({"form": "waterwhite.placement/1", "by": "source", "shards": {}, "otherwise": 7}),
])
def test_a_placement_file_that_is_malformed_is_a_usage_error(world, tmp_path, written):
    accepted(world, tmp_path)
    path = tmp_path / "bad.placement.json"
    path.write_text(written, encoding="utf-8")
    code, document, err = report_with(world, LOW, {"low": path})
    assert (code, document, error_name(err)) == (1, None, "usage"), written


# --- groups.placements ----------------------------------------------------------------------------


def test_groups_placements_has_a_key_for_every_named_layout_and_none_without_a_layout(world):
    assert placements_of(reported(world, {})) == {}
    assert placements_of(reported(world, {"low": "shard-", "high": "nothing-"})) == {
        "low": {"placement": None, "landing": [], "place_not_stated": []},
        "high": {"placement": None, "landing": [], "place_not_stated": []},
    }


def test_the_report_states_the_file_it_read_by_path_and_digest(world, tmp_path):
    placement = write_placement(tmp_path, "low", {"src/B": "shard-1"})
    document = reported(world, LOW, {"low": placement})
    stated = placements_of(document)["low"]["placement"]
    assert set(stated) == {"path", "sha256"}
    assert stated["sha256"] == sha256_of(placement)
    assert stated["path"] in (str(placement), str(placement.resolve()))


def test_the_other_groups_are_what_they_were(world, tmp_path):
    placement = write_placement(tmp_path, "low", {"src/B": "shard-1"})
    plain = reported(world, LOW)["groups"]
    placed = reported(world, LOW, {"low": placement})["groups"]
    assert "placements" in placed
    for key in ("by_run", "by_layout", "layouts"):
        assert placed[key] == plain[key], key


# --- A stated landing in a shard of the store that is kept ----------------------------------------


def world_with_a_clean_shard(world):
    """xs/other: one document, R2:L3 (a copy of src/Z:L3); a mixture over it and a run on
    the mixture, one segment of two steps and its checkpoint."""
    add_shard(world, "xs/other", [("R2", 3)])
    add_mixture(world, "mix-x", ["xs/other"])
    add_run(world, "W", [("W/seg-1", "mix-x")])


def test_the_shard_the_placement_names_for_the_copy_reads_rewrite_and_receives_it(world, tmp_path):
    world_with_a_clean_shard(world)
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    shard = entry(document, "xs/other")
    assert shard["action"] == "rewrite"
    assert shard["receives"] == [B_COPY]
    assert shard["successor"] == "successor of xs/other"
    assert shard["answer"] == "does not contain"
    # one document, none removed, one received (docs/interfaces.md 7.1)
    assert (shard["documents"], shard["records_removed"], shard["removed"]) == (1, 0, [])
    assert shard["documents_in_successor"] == 2
    assert shard["bytes_to_rewrite"] == sizes_of(world.folder, "xs/other")


def test_every_reader_of_rewrite_treats_the_receiving_shard_as_rewritten(world, tmp_path):
    world_with_a_clean_shard(world)
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    # the counts: shard-1 kept; shard-3 and xs/other rewritten
    assert document["counts"]["shards"] == {"keep": 1, "rewrite": 2}
    unplaced = place_not_stated(document)
    assert unplaced == {"shards": [], "mixtures": []}
    assert document["counts"]["bytes_to_rewrite"] == (
        sizes_of(world.folder, "R1.jsonl") + sizes_of(world.folder, "shards/shard-3")
        + sizes_of(world.folder, "xs/other"))
    # the group of the layout
    group = document["groups"]["by_layout"]["x"]
    assert (group["rewrite"], group["keep"]) == (["xs/other"], [])
    # the mixture over the shard, and the run on the mixture
    mixture = artifact_entry(document, "mix-x")
    assert (mixture["action"], mixture["rewritten_shards"], mixture["needs_successor"]) == (
        "rewrite", ["xs/other"], True)
    assert mixture["successor"] == "successor of mix-x"
    assert "mix-x" in document["mixtures"]["needs_successor"]
    restart = next(r for r in document["restarts"] if r["run"] == "W")
    assert restart["exact_rerun"] is not None
    assert restart["exact_rerun"]["segment"] == "W/seg-1"
    # the placement is stated and nothing is left landing outside the store
    placed = placements_of(document)["x"]
    assert placed["landing"] == [] and placed["place_not_stated"] == []
    assert placed["placement"]["sha256"] == sha256_of(tmp_path / "x.placement.json")


def test_verify_needs_the_successor_of_the_shard_that_receives_and_of_its_mixture(world, tmp_path):
    world_with_a_clean_shard(world)
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    verify = need_attr("waterwhite.rebuild.verify_sides", "verify")
    held = {e["artifact"]: e for e in verify(world.store, document)["successors"]}
    # no successor is registered in this store: each is asked for, by the name the report gives
    assert (held["xs/other"]["successor"], held["xs/other"]["registered"]) == (
        "successor of xs/other", False)
    assert (held["mix-x"]["successor"], held["mix-x"]["registered"]) == ("successor of mix-x", False)


def test_the_shards_of_the_layout_the_placement_does_not_name_are_unchanged(world, tmp_path):
    world_with_a_clean_shard(world)
    add_shard(world, "xs/held", [("R2", 1)])
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    # xs/held holds the copy itself, but with a placement only the named shard counts
    assert entry(document, "xs/held")["action"] == "keep"
    assert entry(document, "xs/held")["receives"] == []


def test_the_text_form_names_the_shard_that_receives_and_its_successor(world, tmp_path):
    world_with_a_clean_shard(world)
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "x=xs/",
                         "--placement", f"x={placement}", "--format", "text")
    assert code == 0, err
    assert "xs/other (shard): rewrite -> successor of xs/other" in out


# --- A stated landing in a shard that is already rewritten ----------------------------------------------


def test_the_copy_joins_the_receives_of_a_shard_that_is_rewritten_already(world, tmp_path):
    # shard-3: three documents, D0 (T) removed; B's copy lands in it: 3 - 1 + 1.
    placement = write_placement(tmp_path, "low", {"src/B": "shard-3"})
    document = reported(world, LOW, {"low": placement})
    shard = entry(document, "shard-3")
    assert shard["action"] == "rewrite" and shard["successor"] == "successor of shard-3"
    assert shard["receives"] == [B_COPY]
    assert (shard["documents"], shard["removed"], shard["records_removed"]) == (3, ["D0"], 1)
    assert shard["documents_in_successor"] == 3
    assert entry(document, "shard-1")["action"] == "keep"
    assert document["counts"]["shards"] == {"keep": 1, "rewrite": 1}


# --- A stated landing in a shard the store does not hold ---------------------------------------------------


def test_a_landing_in_a_shard_the_store_does_not_hold_is_named_in_the_placements_only(world, tmp_path):
    plain = reported(world, LOW)
    placement = write_placement(tmp_path, "low", {"src/B": "shard-2"})
    document = reported(world, LOW, {"low": placement})
    assert placements_of(document)["low"]["landing"] == [
        {"shard": "shard-2", "in_store": False, "receives": [B_COPY]}]
    assert "shard-2" not in names(document)
    # no artifact entry is added or changed, and no count or list moves
    assert document["artifacts"] == plain["artifacts"]
    assert document["counts"] == plain["counts"]
    assert document["mixtures"] == plain["mixtures"]
    assert document["restarts"] == plain["restarts"]


def test_a_copy_roots_in_the_source_of_its_chain_and_lands_where_that_source_goes(world, tmp_path):
    # R3's first row is re-sourced onto mid/M:L1, a copy of src/B:L1, which roots in src/B.
    b1, a1 = parent_of(world, "src/B", 1), parent_of(world, "src/A", 1)
    add_file(world, "mid/M", ["passage X"], [single_edge(1, b1)], ["src/B"])
    add_file(world, "R3", ["passage X"], [any_edge(1, a1, parent_of(world, "mid/M", 1))],
             ["src/A", "mid/M"], role=v.ROLE_RELEASE)
    placement = write_placement(tmp_path, "low", {"src/B": "shard-2"})
    document = reported(world, LOW, {"low": placement})
    [landing] = placements_of(document)["low"]["landing"]
    assert landing["shard"] == "shard-2" and landing["in_store"] is False
    assert sorted(landing["receives"]) == sorted([B_COPY, "mid/M:L1"])


def test_a_copy_that_roots_nowhere_lands_where_otherwise_says(world, tmp_path):
    # R4's first row is re-sourced onto mid/N:L1, made from two parents: no single-parent chain.
    b1, z3, a1 = (parent_of(world, "src/B", 1), parent_of(world, "src/Z", 3),
                  parent_of(world, "src/A", 1))
    add_file(world, "mid/N", ["passage X z three"], [many_edge(1, b1, z3)], ["src/B", "src/Z"])
    add_file(world, "R4", ["passage X"], [any_edge(1, a1, parent_of(world, "mid/N", 1))],
             ["src/A", "mid/N"], role=v.ROLE_RELEASE)
    placement = write_placement(tmp_path, "low", {"src/B": "shard-2"}, otherwise="shard-9")
    document = reported(world, LOW, {"low": placement})
    landing = sorted(placements_of(document)["low"]["landing"], key=lambda e: e["shard"])
    assert landing == [
        {"shard": "shard-2", "in_store": False, "receives": [B_COPY]},
        {"shard": "shard-9", "in_store": False, "receives": ["mid/N:L1"]},
    ]


def test_a_copy_the_placement_does_not_place_is_unplaced_and_the_negative_shards_say_so(world, tmp_path):
    # With otherwise null, mid/N:L1 has no place. shard-1 holds src/B:L1 but not mid/N:L1,
    # and the placement names no shard it could hold: shard-1 reads the fourth word and
    # names the one copy that is unplaced (not B's, which the placement places).
    b1, z3, a1 = (parent_of(world, "src/B", 1), parent_of(world, "src/Z", 3),
                  parent_of(world, "src/A", 1))
    add_file(world, "mid/N", ["passage X z three"], [many_edge(1, b1, z3)], ["src/B", "src/Z"])
    add_file(world, "R4", ["passage X"], [any_edge(1, a1, parent_of(world, "mid/N", 1))],
             ["src/A", "mid/N"], role=v.ROLE_RELEASE)
    placement = write_placement(tmp_path, "low", {"src/B": "shard-2"}, otherwise=None)
    document = reported(world, LOW, {"low": placement})
    shard = entry(document, "shard-1")
    assert shard["action"] == WORD
    assert shard["returning"] == ["mid/N:L1"] and shard["place"] == "not stated"
    assert placements_of(document)["low"]["landing"] == [
        {"shard": "shard-2", "in_store": False, "receives": [B_COPY]}]
    assert placements_of(document)["low"]["place_not_stated"] == ["shard-1"]


# --- A shard's own re-sourced document, with a placement stated ---------------------------------------------


def test_with_a_placement_the_placement_decides_where_a_shards_own_re_sourced_copy_lands(world, tmp_path):
    # xs/own: D0 is R1:L1, the row re-sourced onto src/B:L1. Without a placement the
    # copy is in the receives of the shard whose document it is; with one, it lands where
    # the placement says (xs/other), and not in xs/own.
    add_shard(world, "xs/own", [("R1", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    plain = reported(world, X)
    assert entry(plain, "xs/own")["receives"] == [B_COPY]
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    assert entry(document, "xs/own")["action"] == "rewrite"
    assert B_COPY not in entry(document, "xs/own")["receives"]
    assert entry(document, "xs/other")["receives"] == [B_COPY]
