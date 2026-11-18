"""One rule for what makes a record unresolved, for the report and for ``verify``
(plan 4.1, 4.4; docs/rebuild.md sections 2.1 and 5).

The report counts a record unresolved when any hop of its chain says the evidence is
a whole file or a publisher's pointer: an edge labelled file-level, a file-level link
to a member, a mapped link to a member. ``verify`` asks whether a successor holds a
record that is unresolved. The two must give the same word for the same record, so a
successor is never shown as holding no unresolved record when the report would count
one of its records unresolved.

Every successor here is a release (or a shard or a mixture over one) registered in
the hand-built world through the store's own write methods: ``register_jsonl`` with
its edges, ``add_source_link``, ``write_map``, ``register_mixture``. No row is edited
from outside the write API. The report's side of each case is read from the report
the tool itself makes for the same store (``report_without_restarts``), whose record
entries for the successor give the action of each of its records.

Chains by case (hops listed from the record down):

- a file-level edge directly above A's record: the file-level hop, then A's record;
  the weak hop is the first hop and is not the last;
- a file-level edge in the middle: a copy, the file-level hop, A's record; the weak
  hop is in the middle of the chain;
- a mapped link on the record, or behind a copy: the mapped link is the last hop (a
  hop that names a source link always ends its chain);
- a file-level link on the record, or behind a copy: the file-level link is the last
  hop, the one case ``verify`` already caught (green by design);
- a mapped edge to A's record: a definite action in the report (a pointer-only edge
  is not a source link), so nothing unresolved in either (green by design);
- an edge labelled unknown: a failing link, unresolved in both (green by design).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from waterwhite import requests
from waterwhite import vocabulary as v
from waterwhite.records import EdgeInput, EdgeRecord, SnapshotShard

from rb_kit import (
    artifact_entry, build, hand_report, need_attr, record_entries, report_parts, ww,
)

SUCCESSOR = "successor of R1"
LAST_ROW = "L3"
CLEAN_TEXTS = ["passage X", "z one"]
WEAK_TEXT = "a row of the successor with a weak chain"


@dataclass(frozen=True)
class Case:
    name: str
    shape: str        # how the third row gets its evidence
    weak: bool        # whether the report's rule counts the third row unresolved
    action: str       # the report's action for the third row


CASES = [
    # Red against the old verification: the weak hop is not the last hop, or the last
    # hop is a mapped link the old rule did not look at.
    Case("file-level-edge-on-the-record", "file-edge", True, v.UNRESOLVED),
    Case("file-level-edge-in-the-middle", "file-edge-behind-copy", True, v.UNRESOLVED),
    Case("mapped-link-on-the-record", "mapped-link", True, v.UNRESOLVED),
    Case("mapped-link-behind-a-copy", "mapped-link-behind-copy", True, v.UNRESOLVED),
    # Green by design: the old verification already caught the file-level link as the
    # last hop; and the strong or failing cases the two rules already agree on.
    Case("file-level-link-on-the-record", "file-link", True, v.UNRESOLVED),
    Case("file-level-link-behind-a-copy", "file-link-behind-copy", True, v.UNRESOLVED),
    Case("an-unknown-edge-is-a-failing-link", "unknown-edge", True, v.UNRESOLVED),
    Case("a-mapped-edge-has-a-definite-action", "mapped-edge", False, v.ACTION_RETIRE),
    Case("an-observed-edge-has-a-definite-action", "observed-edge", False, v.ACTION_RETIRE),
]


def _edge_to_a(label: str, derivation: str, verification: str | None, world) -> EdgeRecord:
    return EdgeRecord(LAST_ROW, (EdgeInput(ww.ref(world["src/A"], 1)),), ww.step("merge"),
                      derivation=derivation, verification=verification, label=label)


def _mid(world, folder, how: str) -> str:
    """An intermediate file of one record whose only evidence is ``how``."""
    store = world.store
    a = world["src/A"]
    if how == "file-edge":
        edges = [EdgeRecord("L1", (EdgeInput(ww.ref(a, 1)),), ww.step("merge"),
                            derivation=v.DERIVED_FILE, label=v.FILE_LEVEL)]
        inputs = [a]
    else:
        edges, inputs = [], []
    mid = ww.record(store, folder / "mid.jsonl", ["a middle record"], "mid", v.ROLE_INTERMEDIATE,
                    ww.step("merge", "mid"), inputs, edges)
    if how == "mapped-link":
        store.add_source_link(mid, "L1", "A/r1", v.MAPPED, "a publisher's pointer to A/r1",
                              derivation=v.DERIVED_POINTER, verification=v.VERIFIED_PASSED)
    elif how == "file-link":
        store.add_source_link(mid, "L1", "A/r1", v.FILE_LEVEL, "the file is one of A's",
                              derivation=v.DERIVED_FILE)
    return mid


def with_successor(folder, shape: str):
    """The world with ``successor of R1`` registered: two clean rows (copies of B's
    and Z's records) and a third row whose evidence is ``shape``."""
    world = build(folder)
    store = world.store
    b, z = world["src/B"], world["src/Z"]
    edges = [ww.copy_edge(1, ww.ref(b, 1)), ww.copy_edge(2, ww.ref(z, 1))]
    inputs = [b, z]
    links = None
    if shape == "file-edge":
        edges.append(_edge_to_a(v.FILE_LEVEL, v.DERIVED_FILE, None, world))
        inputs.append(world["src/A"])
    elif shape == "file-edge-behind-copy":
        mid = _mid(world, folder, "file-edge")
        edges.append(ww.copy_edge(3, ww.ref(mid, 1)))
        inputs.append(mid)
    elif shape == "mapped-link":
        links = (v.MAPPED, "a publisher's pointer to A/r1", v.DERIVED_POINTER, v.VERIFIED_PASSED)
    elif shape == "mapped-link-behind-copy":
        mid = _mid(world, folder, "mapped-link")
        edges.append(ww.copy_edge(3, ww.ref(mid, 1)))
        inputs.append(mid)
    elif shape == "file-link":
        links = (v.FILE_LEVEL, "the file is one of A's", v.DERIVED_FILE, None)
    elif shape == "file-link-behind-copy":
        mid = _mid(world, folder, "file-link")
        edges.append(ww.copy_edge(3, ww.ref(mid, 1)))
        inputs.append(mid)
    elif shape == "unknown-edge":
        edges.append(_edge_to_a(v.UNKNOWN, v.DERIVED_NONE, None, world))
        inputs.append(world["src/A"])
    elif shape == "mapped-edge":
        edges.append(_edge_to_a(v.MAPPED, v.DERIVED_POINTER, v.VERIFIED_PASSED, world))
        inputs.append(world["src/A"])
    elif shape == "observed-edge":
        edges.append(_edge_to_a(v.OBSERVED, v.DERIVED_STATED, None, world))
        inputs.append(world["src/A"])
    else:
        raise AssertionError(f"no such shape: {shape}")
    successor = world.ids[SUCCESSOR] = ww.record(
        store, folder / "successors/R1.jsonl", [*CLEAN_TEXTS, WEAK_TEXT], SUCCESSOR, v.ROLE_RELEASE,
        ww.step("release", "R1"), inputs, edges)
    if links is not None:
        label, evidence, derivation, verification = links
        store.add_source_link(successor, LAST_ROW, "A/r1", label, evidence,
                              derivation=derivation, verification=verification)
    return world


def reports_row(document: dict, artifact: str, row: str) -> str:
    found = [e for e in record_entries(document, artifact) if e["record"].rsplit(":", 1)[1] == row]
    assert len(found) == 1, f"{artifact} {row}: {len(found)} record entries"
    return found[0]["action"]


def verify_entry(world) -> dict:
    function = need_attr("waterwhite.rebuild.verify_successors", "check_successors")
    report = hand_report(world, [("R1", "rewrite", SUCCESSOR)])
    successors, _ = function(world.store, report, requests.withdraw(world.store, "A"), None, None)
    assert [e["artifact"] for e in successors] == ["R1"]
    return successors[0]


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_verify_and_the_report_give_one_word_to_the_record(tmp_path, case):
    world = with_successor(tmp_path / "w", case.shape)
    try:
        document = report_parts(world)
        # The report's rule: the two clean rows are kept, the third gets its action.
        assert [reports_row(document, SUCCESSOR, row) for row in ("L1", "L2")] == [v.ACTION_KEEP] * 2
        counts = artifact_entry(document, SUCCESSOR)["counts"]
        assert reports_row(document, SUCCESSOR, LAST_ROW) == case.action, case.name
        assert counts[v.UNRESOLVED] == (1 if case.weak else 0), case.name
        # The verification must agree with the report about the same record.
        entry = verify_entry(world)
        assert entry["registered"] is True
        assert entry["holds_unresolved"] is case.weak, (
            f"{case.name}: the report says {case.action} for the row, verify says "
            f"holds_unresolved {entry['holds_unresolved']}")
    finally:
        world.store.close()


def test_a_successor_with_only_clean_rows_holds_nothing_unresolved(tmp_path):
    world = build(tmp_path / "w", successors=True)
    try:
        document = report_parts(world)
        assert artifact_entry(document, SUCCESSOR)["counts"][v.UNRESOLVED] == 0
        assert verify_entry(world)["holds_unresolved"] is False
    finally:
        world.store.close()


def over_weak_successor(world, folder):
    """A successor shard whose documents are the rows of the weak successor, and a
    successor mixture over it, registered with the store's own write methods."""
    store = world.store
    successor = world[SUCCESSOR]
    rows = [ww.recovered(0, ww.ref(successor, 1)), ww.recovered(1, ww.ref(successor, 3))]
    shard, digest = ww.shard(store, folder / "successors/shard-3", "successor of shard-3",
                             [successor], rows)
    kept = store.artifact(world["shard-1"])
    mixture = store.register_mixture(
        name="successor of mix-2",
        shards=[SnapshotShard(0, kept.instance_id, "shards/shard-1", kept.content_digest),
                SnapshotShard(1, shard, "successors/shard-3", digest)],
        settings={"mix": 2})
    return shard, mixture


@pytest.mark.parametrize("shape", ["file-edge", "mapped-link"])
def test_a_successor_shard_and_mixture_over_such_a_record_hold_it_too(tmp_path, shape):
    world = with_successor(tmp_path / "w", shape)
    try:
        over_weak_successor(world, tmp_path / "w")
        function = need_attr("waterwhite.rebuild.verify_successors", "check_successors")
        report = hand_report(world, [("R1", "rewrite", SUCCESSOR),
                                     ("shard-3", "rewrite", "successor of shard-3"),
                                     ("mix-2", "rewrite", "successor of mix-2")])
        successors, _ = function(world.store, report, requests.withdraw(world.store, "A"), None, None)
        held = {e["artifact"]: e["holds_unresolved"] for e in successors}
        assert held == {"R1": True, "shard-3": True, "mix-2": True}, held
    finally:
        world.store.close()
