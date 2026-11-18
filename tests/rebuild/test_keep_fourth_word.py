"""The fourth word of an artifact, "clean, place not stated" (DECISIONS.md D-141;
docs/rebuild.md 2.2; docs/interfaces.md 7.1; docs/cli.md "report").

Where the user states no place for the copies that return (a promoted or re-sourced
record), a shard that nothing of A stands behind does not read "keep" outright: its own
contents are clean, the copies it may receive are named, and their place is not stated.
Every shard of a group whose answer is the negative reads it while one copy of the report
is unplaced and not held in that group (``test_keep_held.py`` says what holds). The
fields of the entry are the ones it had, with ``returning`` (the unplaced copies, as
``artifact:locator``) and ``place`` ("not stated") added.

The world: layout ``x`` is ``xs/``. ``xs/a`` and ``xs/b`` are clean shards of one document
each (R2:L3 and R2:L2, copies of ``src/Z:L3`` and ``src/B:L2``); none holds the one
returning copy ``src/B:L1``, so both read the word. ``xs/nm`` has no map (unresolved).
The mixtures, in the order they are made:

- ``mix-a`` [xs/a]: a shard of it reads the word and none is rewritten: the word;
- ``mix-ab`` [xs/a, shard-1]: shard-1 is kept: the word;
- ``mix-au`` [xs/a, xs/nm]: an unresolved shard and none rewritten: today's rule, unresolved;
- ``mix-ar`` [xs/a, shard-3]: shard-3 is rewritten: a rewrite;
- the world's ``mix-1`` [shard-1] is kept and ``mix-2`` [shard-1, shard-3] a rewrite.

Every mixture is in exactly one of the four lists of ``mixtures``.
"""

from __future__ import annotations

import re

from waterwhite import vocabulary as v

from keep_kit import (
    B_COPY, WORD, add_mixture, add_run, add_shard, entry, place_not_stated, placements_of,
    reported, word_shards,
)
from rb_kit import artifact_entry, run

X = {"x": "xs/"}
ENTRY_KEYS = {"artifact", "instance", "kind", "role", "action", "answer", "chain", "successor",
              "records", "counts", "records_removed", "removed", "rows_in_successor",
              "documents", "documents_in_successor", "receives", "rewritten_shards",
              "needs_successor", "bytes_to_rewrite"}


def world_with_words(world):
    add_shard(world, "xs/a", [("R2", 3)])
    add_shard(world, "xs/b", [("R2", 2)])
    add_shard(world, "xs/nm", [("R2", 3)], rows=False)
    add_mixture(world, "mix-a", ["xs/a"])
    add_mixture(world, "mix-ab", ["xs/a", "shard-1"])
    add_mixture(world, "mix-au", ["xs/a", "xs/nm"])
    add_mixture(world, "mix-ar", ["xs/a", "shard-3"])
    add_run(world, "W", [("W/seg-1", "mix-a")])


def test_every_clean_shard_of_a_group_that_holds_no_copy_reads_the_word(world):
    world_with_words(world)
    document = reported(world, X)
    assert word_shards(document) == ["xs/a", "xs/b"]
    # the shards of other groups keep what they had: the group of the shards no layout names
    # holds the copy in shard-1
    assert entry(document, "shard-1")["action"] == "keep"
    assert entry(document, "shard-3")["action"] == "rewrite"
    assert entry(document, "xs/nm")["action"] == "unresolved"


def test_the_entry_keeps_its_fields_and_names_the_copies_that_may_land_and_that_the_place_is_not_stated(world):
    world_with_words(world)
    shard = entry(reported(world, X), "xs/a")
    assert shard["action"] == WORD and not shard["action"].startswith("keep")
    assert ENTRY_KEYS <= set(shard)
    assert shard["answer"] == "does not contain"
    assert (shard["documents"], shard["documents_in_successor"]) == (1, 1)
    assert (shard["removed"], shard["records_removed"], shard["receives"]) == ([], 0, [])
    assert shard["successor"] is None and shard["bytes_to_rewrite"] is None
    assert shard["returning"] == [B_COPY]
    assert shard["place"] == "not stated"


def test_the_counts_say_the_shards_and_mixtures_that_read_it_and_count_none_of_them_as_keep(world):
    world_with_words(world)
    counts = reported(world, X)["counts"]
    assert place_not_stated({"counts": counts}) == {
        "shards": ["xs/a", "xs/b"], "mixtures": ["mix-a", "mix-ab"]}
    # the two counts keep their two words: shard-1 is the only shard kept, shard-3 the only
    # one rewritten; mix-1 the only mixture kept, mix-2 and mix-ar rewritten
    assert counts["shards"] == {"keep": 1, "rewrite": 1}
    assert counts["mixtures"] == {"keep": 1, "rewrite": 2}
    assert counts["unresolved_artifacts"] == ["xs/nm", "mix-au"]


def test_a_mixture_reads_the_word_only_where_a_shard_does_and_none_is_rewritten(world):
    world_with_words(world)
    document = reported(world, X)
    assert artifact_entry(document, "mix-a")["action"] == WORD
    assert artifact_entry(document, "mix-ab")["action"] == WORD
    assert artifact_entry(document, "mix-au")["action"] == "unresolved"
    assert artifact_entry(document, "mix-ar")["action"] == "rewrite"
    assert artifact_entry(document, "mix-1")["action"] == "keep"
    for name in ("mix-a", "mix-ab"):
        word = artifact_entry(document, name)
        assert (word["successor"], word["needs_successor"], word["rewritten_shards"]) == (
            None, False, [])


def test_every_mixture_is_in_exactly_one_of_the_four_lists(world):
    world_with_words(world)
    document = reported(world, X)
    lists = document["mixtures"]
    assert set(lists) == {"needs_successor", "kept", "unresolved", "place_not_stated"}
    assert sorted(lists["needs_successor"]) == ["mix-2", "mix-ar"]
    assert lists["kept"] == ["mix-1"]
    assert lists["unresolved"] == ["mix-au"]
    assert sorted(lists["place_not_stated"]) == ["mix-a", "mix-ab"]
    held = [name for names in lists.values() for name in names]
    every = [e["artifact"] for e in document["artifacts"] if e["kind"] == v.KIND_MIXTURE]
    assert sorted(held) == sorted(every) and len(held) == len(set(held))


def test_the_group_lists_the_shards_of_its_layout_that_read_the_word(world):
    world_with_words(world)
    placed = placements_of(reported(world, X))["x"]
    assert placed["placement"] is None and placed["landing"] == []
    assert sorted(placed["place_not_stated"]) == ["xs/a", "xs/b"]


def test_with_no_layout_named_all_shards_are_one_group_and_the_word_follows_from_no_holder(world):
    # shard-1, the one holder of src/B:L1, loses its map: it holds nothing, and there is
    # no layout, so the one group of all shards holds the copy nowhere.
    world.delete_map("shard-1")
    add_shard(world, "xs/a", [("R2", 3)])
    document = reported(world, {})
    assert entry(document, "shard-1")["action"] == "unresolved"
    assert word_shards(document) == ["xs/a"]
    assert placements_of(document) == {}
    assert place_not_stated(document)["shards"] == ["xs/a"]


# --- The text form -------------------------------------------------------------------------------------------


def test_the_text_form_prints_a_line_for_each_shard_that_reads_the_word(world):
    world_with_words(world)
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "x=xs/", "--format", "text")
    assert code == 0, err
    lines = out.splitlines()
    for shard in ("xs/a", "xs/b"):
        about = [line for line in lines if shard in line and B_COPY in line]
        assert about, f"no line says {shard} may receive {B_COPY}:\n{out}"
        assert all("not stated" in line for line in about), about
        assert not [line for line in lines if f"{shard} (shard): keep" in line]
    assert not [line for line in lines if "xs/nm" in line and B_COPY in line]


# --- The restart points of a run over such a mixture ---------------------------------------------------------------


def test_no_run_over_a_mixture_that_reads_the_word_reads_no_exact_rerun_needed(world):
    # Run W trains on mix-a only. Where the place of a returning copy is not stated, the
    # shard may change, so the run is not one whose restart can be left unsaid: its entry
    # says something about the exact rerun (a restart point, or that none is stated).
    world_with_words(world)
    document = reported(world, X)
    restart = next(r for r in document["restarts"] if r["run"] == "W")
    assert restart["exact_rerun"] is not None


def test_a_run_over_a_kept_mixture_has_no_exact_rerun(world):
    # The control: with a holder in the layout nothing reads the word, and the same run
    # reads "no exact rerun" (null) as before.
    add_shard(world, "xs/held", [("R2", 1)])
    add_mixture(world, "mix-h", ["xs/held"])
    add_run(world, "W", [("W/seg-1", "mix-h")])
    document = reported(world, X)
    assert entry(document, "xs/held")["action"] == "keep"
    restart = next(r for r in document["restarts"] if r["run"] == "W")
    assert restart["exact_rerun"] is None


# --- A replayable run whose only stale segment reads the word -------------------------------------------------


def test_a_replayable_run_whose_only_stale_segment_reads_the_word_is_given_the_phase_start(world):
    # Run W is replayable: its records hold every setting a replay needs, as the records of the
    # world's runs E and Y do. Its first segment trains on mix-a, which reads the word; its
    # second, on mix-1, is kept, and starts from the checkpoint W@2 registered after the first.
    # The shard of mix-2 loses its map, so mix-2 is unresolved and no mixture of the report is
    # rewritten: the mixture that reads the word is the only one that may change.
    world.delete_map("shard-3")
    add_shard(world, "xs/a", [("R2", 3)])
    add_mixture(world, "mix-a", ["xs/a"])
    add_run(world, "W", [("W/seg-1", "mix-a"), ("W/seg-2", "mix-1")])
    assert world.store.resolve("W@2") is not None, "no checkpoint is registered after W/seg-1"
    document = reported(world, X)

    # the only mixture that may change is the one that reads the word: none is rewritten
    mixtures = [e for e in document["artifacts"] if e["kind"] == v.KIND_MIXTURE]
    assert [e["artifact"] for e in mixtures if e["action"] == WORD] == ["mix-a"]
    assert [e["artifact"] for e in mixtures if e["action"] == v.ACTION_REWRITE] == []
    assert document["mixtures"]["needs_successor"] == []
    assert document["mixtures"]["place_not_stated"] == ["mix-a"]

    restarts = [r for r in document["restarts"] if r["run"] == "W"]
    assert len(restarts) == 1, [r["run"] for r in document["restarts"]]
    exact = restarts[0]["exact_rerun"]
    assert exact is not None, "the run over a mixture that reads the word has no exact rerun"
    assert exact["run_record_replayable"] is True
    assert exact["basis"] == v.RESTART_BASIS_PHASE_START, exact["basis"]
    assert exact["step_comparison"] == v.STEP_COMPARISON_NOT_POSSIBLE, exact["step_comparison"]
    assert (exact["segment"], exact["step"]) == ("W/seg-1", 1)

    code, out, err = run(world, "report", "withdraw", "A", "--layout", "x=xs/", "--format", "text")
    assert code == 0, err
    lines = [line for line in out.splitlines() if v.RESTART_BASIS_PHASE_START in line]
    assert len(lines) == 1, f"one phase-start line, for run W, got {lines}"
    assert re.search(r"\bW\b", lines[0]) and "W/seg-1" in lines[0]
    assert v.STEP_COMPARISON_NOT_POSSIBLE in lines[0]
    assert v.RESTART_EXACT_BY_VERIFY not in out
