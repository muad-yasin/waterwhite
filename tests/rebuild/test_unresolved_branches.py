"""Two branches of ``rebuild/records.py`` that decide the word of a record, each in a
world where nothing else does (plan 4.1, 4.4; docs/interfaces.md sections 7.1 and 7.3;
DECISIONS.md D-72, amended (c)).

- A row of an "any" edge that kept a copy in scope, whose one other copy has a failing
  link: nothing shows that an equivalent copy remains, so the row is unresolved. Without
  the rule the row is retired, and a release with that row is verified as clean.
- A document of a shard whose own verdict is unresolved (its row's label is stronger
  than its map allows) while the one record its row names is judged by its own chain:
  the report's entry for the shard retires the document, and the verification must still
  say that the shard holds an unresolved record.

Both worlds are built with the store's own verbs, as ``test_verify_one_rule.py`` does,
except the edited label of the second, which no write verb gives (a row labelled
"unknown" names no record, and a label stronger than the map allows is refused).
"""

from __future__ import annotations

from waterwhite import requests
from waterwhite import vocabulary as v
from waterwhite.records import EdgeInput, EdgeRecord

from rb_kit import (
    artifact_entry, build, hand_report, need_attr, record_entries, report_parts, ww,
)

SUCCESSOR = "successor of R1"
SHARD_SUCCESSOR = "successor of shard-3"


def verify_entry(world, artifact: str, successor: str) -> dict:
    function = need_attr("waterwhite.rebuild.verify_successors", "check_successors")
    report = hand_report(world, [(artifact, "rewrite", successor)])
    successors, _ = function(world.store, report, requests.withdraw(world.store, "A"), None, None)
    assert [e["artifact"] for e in successors] == [artifact]
    return successors[0]


# --- The "any" edge whose other parent shows nothing ------------------------------------------


def with_undecided_row(folder):
    """The base world and ``successor of R1``: two clean rows and a third whose edge is
    an "any" edge that kept A's copy of SA, the one other parent being a record of an
    intermediate file whose edge is labelled unknown (it shows nothing)."""
    world = build(folder)
    store = world.store
    a, b, z = world["src/A"], world["src/B"], world["src/Z"]
    unknown = EdgeRecord("L1", (EdgeInput(ww.ref(a, 2)),), ww.step("merge"),
                         derivation=v.DERIVED_NONE, label=v.UNKNOWN)
    mid = ww.record(store, folder / "mid.jsonl", ["passage SA"], "mid", v.ROLE_INTERMEDIATE,
                    ww.step("merge", "mid"), [a], [unknown])
    undecided = EdgeRecord("L3", (EdgeInput(ww.ref(a, 2), selected=True), EdgeInput(ww.ref(mid, 1))),
                           ww.step("dedup"), parent_mode=v.MODE_ANY)
    world.ids[SUCCESSOR] = ww.record(
        store, folder / "successors/R1.jsonl", ["passage X", "z one", "passage SA"], SUCCESSOR,
        v.ROLE_RELEASE, ww.step("release", "R1"), [b, z, a, mid],
        [ww.copy_edge(1, ww.ref(b, 1)), ww.copy_edge(2, ww.ref(z, 1)), undecided])
    return world


def test_an_any_row_whose_other_parent_shows_nothing_is_unresolved_in_the_report_and_in_verify(tmp_path):
    world = with_undecided_row(tmp_path / "w")
    try:
        document = report_parts(world)
        words = {e["record"]: e["action"] for e in record_entries(document, SUCCESSOR)}
        assert words == {f"{SUCCESSOR}:L1": v.ACTION_KEEP, f"{SUCCESSOR}:L2": v.ACTION_KEEP,
                         f"{SUCCESSOR}:L3": v.UNRESOLVED}
        row = next(e for e in record_entries(document, SUCCESSOR) if e["record"].endswith(":L3"))
        assert row["failing"]["artifact"] == "mid" and row["new_parent"] is None
        assert artifact_entry(document, SUCCESSOR)["counts"][v.UNRESOLVED] == 1
        # The verification gives the record the same word.
        assert verify_entry(world, "R1", SUCCESSOR)["holds_unresolved"] is True
    finally:
        world.store.close()


# --- A document whose own verdict is unresolved, behind a parent that is not -------------------


def with_edited_document(folder):
    """The base world and ``successor of shard-3``: two documents whose map rows name R1:L2
    (retired under "withdraw A": a paraphrase of A's record) and R1:L5 (kept), and the label
    of the first row edited from outside the write API to one the map does not allow."""
    world = build(folder)
    rows = [ww.recovered(0, ww.ref(world["R1"], 2)), ww.recovered(1, ww.ref(world["R1"], 5))]
    instance, _ = ww.shard(world.store, folder / "successors/shard-3", SHARD_SUCCESSOR,
                           [world["R1"]], rows)
    world.ids[SHARD_SUCCESSOR] = instance
    world.sql(("UPDATE map_row SET label = 'observed' WHERE binding_id = ? AND ordinal = 0",
               (world.binding_id(SHARD_SUCCESSOR),)))
    return world


def test_a_document_unresolved_by_itself_is_retired_by_the_report_and_counted_by_verify(tmp_path):
    world = with_edited_document(tmp_path / "w")
    try:
        entry = artifact_entry(report_parts(world), SHARD_SUCCESSOR)
        # The report judges the document through the record its row names: retired.
        assert entry["removed"] == ["D0"] and entry["records_removed"] == 1
        # The verification reads the document's own verdict too: a label stronger than
        # the map allows shows nothing, so the shard holds an unresolved record.
        assert verify_entry(world, "shard-3", SHARD_SUCCESSOR)["holds_unresolved"] is True
    finally:
        world.store.close()


def with_edited_kept_document(folder):
    """The same shard, with the label of the second row edited instead: the row names
    R1:L5, which "withdraw A" keeps, and the first row (R1:L2) keeps its label."""
    world = build(folder)
    rows = [ww.recovered(0, ww.ref(world["R1"], 2)), ww.recovered(1, ww.ref(world["R1"], 5))]
    instance, _ = ww.shard(world.store, folder / "successors/shard-3", SHARD_SUCCESSOR,
                           [world["R1"]], rows)
    world.ids[SHARD_SUCCESSOR] = instance
    world.sql(("UPDATE map_row SET label = 'observed' WHERE binding_id = ? AND ordinal = 1",
               (world.binding_id(SHARD_SUCCESSOR),)))
    return world


def test_a_document_unresolved_by_itself_is_kept_by_the_report_and_counted_by_verify(tmp_path):
    world = with_edited_kept_document(tmp_path / "w")
    try:
        entry = artifact_entry(report_parts(world), SHARD_SUCCESSOR)
        # The report judges D1 through the record its row names, which stays: D1 is kept,
        # and only D0 leaves.
        assert entry["removed"] == ["D0"] and entry["documents_in_successor"] == 1
        # Verify is never less strict than the report: D1's own verdict is unresolved, and
        # it is the only document that can make the shard hold an unresolved record.
        assert verify_entry(world, "shard-3", SHARD_SUCCESSOR)["holds_unresolved"] is True
    finally:
        world.store.close()
