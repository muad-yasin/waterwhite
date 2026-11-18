"""Which shard already holds a returning copy (DECISIONS.md D-141; docs/rebuild.md 2.2;
docs/interfaces.md 7.1), and what the report says of a shard that holds none.

Under "withdraw A" the world's one returning copy is ``src/B:L1``: R1's first row is
re-sourced onto it, and the row of B's own file is the promoted record. A shard that
nothing of A stands behind is kept, unless a returning copy may land in it and the shard
does not already hold the copy; then its own contents are clean and the place of the
copy is not stated (the fourth word, ``WORD``).

A shard HOLDS a copy when a document of the shard that the report keeps as it is has a
map parent from which the copy is reached by following single-parent, record-level,
non-generation edges only: the copy itself, its parent, its parent's parent. No "any"
edge is followed, no equivalent copy is counted, and a candidates row, a shard with no
usable map and a document the report changes hold nothing. The count is made over the
shards of one named layout (all the shards that match no named layout are one group, and
with no layout named all shards are one group), and with a placement stated for a layout
only the shard the placement names for the copy can hold it.

The shards these tests add are named ``xs/<case>``; the layout ``x`` is ``xs/``. Their
documents are copies of rows of the world: R2's first row (L1) is a copy of ``src/B:L1``,
R2's third row (L3) a copy of ``src/Z:L3``, which is not B's. The report is asked through
the command, with ``--layout`` and, where a test says so, ``--placement``.
"""

from __future__ import annotations

import pytest
from waterwhite import vocabulary as v

from keep_kit import (
    B_COPY, WORD, add_file, add_shard, any_edge, entry, generation_edge, many_edge,
    parent_of, place_not_stated, reported, single_edge, word_shards, write_placement,
)
from rb_kit import ww

X = {"x": "xs/"}


def actions(document: dict, *shards: str) -> dict[str, str]:
    return {name: entry(document, name)["action"] for name in shards}


def test_the_fourth_word_is_a_word_of_the_vocabulary_and_never_begins_with_keep():
    assert WORD in v.ARTIFACT_ACTIONS, v.ARTIFACT_ACTIONS
    assert not WORD.startswith(v.ACTION_KEEP)


# --- A copy a kept document reaches ----------------------------------------------------------


def test_a_shard_whose_document_is_a_copy_of_the_returning_copy_holds_it(world):
    # xs/held: D0 is R2:L1, the copy of src/B:L1 (two hops: the row, then its parent).
    # xs/other: D0 is R2:L3, a copy of src/Z:L3. The layout x holds the copy in xs/held,
    # so no shard of it is left to receive an unplaced copy: both are kept.
    add_shard(world, "xs/held", [("R2", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/held", "xs/other") == {"xs/held": "keep", "xs/other": "keep"}
    assert place_not_stated(document) == {"shards": [], "mixtures": []}


def test_a_shard_whose_document_is_the_returning_copy_itself_holds_it(world):
    add_shard(world, "xs/direct", [("src/B", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/direct", "xs/other") == {"xs/direct": "keep", "xs/other": "keep"}
    assert place_not_stated(document)["shards"] == []


def test_a_chain_of_single_parent_copy_edges_of_any_length_is_followed(world):
    # xs/deep: D0 is mid/C1:L1, a copy of mid/C2:L1, a copy of mid/C3:L1, a copy of src/B:L1.
    add_file(world, "mid/C3", ["passage X"], [single_edge(1, parent_of(world, "src/B", 1))], ["src/B"])
    add_file(world, "mid/C2", ["passage X"], [single_edge(1, parent_of(world, "mid/C3", 1))], ["mid/C3"])
    add_file(world, "mid/C1", ["passage X"], [single_edge(1, parent_of(world, "mid/C2", 1))], ["mid/C2"])
    add_shard(world, "xs/deep", [("mid/C1", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/deep", "xs/other") == {"xs/deep": "keep", "xs/other": "keep"}
    assert place_not_stated(document)["shards"] == []


def test_a_shard_that_is_rewritten_holds_the_copy_through_the_documents_the_report_keeps(world):
    # xs/rw: D0 is R1:L3 (T, which the report retires), D1 is R2:L1 (a copy of src/B:L1,
    # kept as it is). The shard is rewritten, and it holds the copy through D1.
    add_shard(world, "xs/rw", [("R1", 3), ("R2", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/rw", "xs/other") == {"xs/rw": "rewrite", "xs/other": "keep"}
    assert entry(document, "xs/rw")["removed"] == ["D0"]
    assert place_not_stated(document)["shards"] == []


# --- What the walk does not follow, one clause each ---------------------------------------------


def not_held_cases(world) -> dict[str, tuple]:
    """For each clause of the limited walk, the shard that reaches ``src/B:L1`` only by
    what the walk does not follow, and the action its answer gives it."""
    b1, z3 = parent_of(world, "src/B", 1), parent_of(world, "src/Z", 3)
    add_file(world, "gen/Q", ["q one"], [generation_edge(1, b1)], ["src/B"])
    add_file(world, "mid/Any", ["passage X"], [any_edge(1, b1, z3)], ["src/B", "src/Z"])
    add_file(world, "mid/Many", ["passage X and z three"], [many_edge(1, b1, z3)], ["src/B", "src/Z"])
    add_shard(world, "xs/generated", [("gen/Q", 1)])
    add_shard(world, "xs/anyedge", [("mid/Any", 1)])
    add_shard(world, "xs/several", [("mid/Many", 1)])
    # A candidates row: the document is one of two twins, B's copy and Z's line.
    add_shard(world, "xs/twins", [ww.twins(0, b1, z3)])
    return {
        "xs/generated": "a generation edge", "xs/anyedge": "an any edge",
        "xs/several": "an edge with two parents", "xs/twins": "a candidates row",
    }


@pytest.mark.parametrize("shard", ["xs/generated", "xs/anyedge", "xs/several", "xs/twins"])
def test_a_shard_that_reaches_the_copy_only_by_what_the_walk_does_not_follow_holds_nothing(
        world, shard):
    clause = not_held_cases(world)[shard]
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    # Each of these shards answers "does not contain" (nothing of A stands behind it) and
    # holds no copy, so, with no shard of the layout holding src/B:L1, every one of them
    # reads the fourth word, and so does xs/other beside it.
    assert entry(document, shard)["answer"] == "does not contain", clause
    assert actions(document, shard, "xs/other") == {shard: WORD, "xs/other": WORD}, clause
    assert entry(document, shard)["returning"] == [B_COPY], clause


def test_a_shard_with_no_map_holds_nothing_and_keeps_its_own_action(world):
    add_shard(world, "xs/nomap", [("R2", 1)], rows=False)
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/nomap", "xs/other") == {"xs/nomap": "unresolved", "xs/other": WORD}
    assert word_shards(document) == ["xs/other"]


# --- One group at a time ---------------------------------------------------------------------------


def test_the_copy_is_held_per_layout_a_holder_of_one_layout_does_not_serve_another(world):
    # Layout x has a holder (xs/held); layout y has none (ys/other: R2:L3, a copy of Z's line).
    add_shard(world, "xs/held", [("R2", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    add_shard(world, "ys/other", [("R2", 3)])
    document = reported(world, {"x": "xs/", "y": "ys/"})
    assert actions(document, "xs/held", "xs/other", "ys/other") == {
        "xs/held": "keep", "xs/other": "keep", "ys/other": WORD}
    assert place_not_stated(document)["shards"] == ["ys/other"]


def test_with_no_layout_named_every_shard_is_one_group_and_the_holder_serves_all(world):
    # shard-1 holds src/B:L1 (R2:L1), and it is in the one group with xs/other.
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, {})
    assert actions(document, "shard-1", "xs/other") == {"shard-1": "keep", "xs/other": "keep"}
    assert place_not_stated(document) == {"shards": [], "mixtures": []}


@pytest.mark.parametrize("which", ["world", "swap_world", "repair_world"])
def test_shard_1_holds_the_returning_copy_in_the_base_the_swap_and_the_successors_worlds(
        request, which):
    # In each of the three worlds the one returning copy is src/B:L1 (R1's first row is
    # re-sourced onto it) and shard-1 holds it, so with no layout named nothing is left
    # unplaced and unheld, and no shard reads the word.
    built = request.getfixturevalue(which)
    document = reported(built, {})
    assert entry(document, "shard-1")["action"] == "keep"
    assert place_not_stated(document) == {"shards": [], "mixtures": []}
    assert word_shards(document) == []


def test_the_shards_that_match_no_named_layout_are_one_group(world):
    # xs/other is in layout x, with no holder: the word. ys/other matches no named layout,
    # so it is in one group with shard-1 and shard-3, and shard-1 holds the copy there.
    add_shard(world, "xs/other", [("R2", 3)])
    add_shard(world, "ys/other", [("R2", 3)])
    document = reported(world, X)
    assert actions(document, "xs/other", "ys/other", "shard-1") == {
        "xs/other": WORD, "ys/other": "keep", "shard-1": "keep"}


def test_a_layout_that_matches_no_shard_changes_nothing(world):
    add_shard(world, "xs/other", [("R2", 3)])
    document = reported(world, {"none": "nothing-here/"})
    assert actions(document, "shard-1", "xs/other") == {"shard-1": "keep", "xs/other": "keep"}
    assert place_not_stated(document) == {"shards": [], "mixtures": []}


# --- With a placement stated only the shard it names can hold the copy ---------------------------------


def test_with_a_placement_the_copy_is_held_only_by_the_shard_the_placement_names(world, tmp_path):
    # xs/held holds src/B:L1; the placement says B's copies land in xs/other, which does
    # not hold it: xs/other is rewritten and receives it, and xs/held, not named, stays.
    add_shard(world, "xs/held", [("R2", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    placement = write_placement(tmp_path, "x", {"src/B": "xs/other"})
    document = reported(world, X, {"x": placement})
    assert actions(document, "xs/held", "xs/other") == {"xs/held": "keep", "xs/other": "rewrite"}
    assert entry(document, "xs/other")["receives"] == [B_COPY]
    assert entry(document, "xs/held")["receives"] == []


def test_with_a_placement_that_names_the_holder_the_copy_is_held_and_nothing_is_received(world, tmp_path):
    add_shard(world, "xs/held", [("R2", 1)])
    add_shard(world, "xs/other", [("R2", 3)])
    placement = write_placement(tmp_path, "x", {"src/B": "xs/held"})
    document = reported(world, X, {"x": placement})
    assert actions(document, "xs/held", "xs/other") == {"xs/held": "keep", "xs/other": "keep"}
    assert entry(document, "xs/held")["receives"] == [] and entry(document, "xs/other")["receives"] == []
    assert place_not_stated(document) == {"shards": [], "mixtures": []}
