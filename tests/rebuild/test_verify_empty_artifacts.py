"""An artifact the repair leaves with no record, named on neither side of the sides
file (plan 4.2, 4.4; docs/rebuild.md section 5), through ``verify`` on the hand-built
world and the tool's own report for it.

The repaired side does not write a file that would hold no record, and a reference
built without the withdrawn source has no such file, so the sides file names it on
neither side. That is not a file the comparison lacks. The tool shows it from its own
record: the report's entry gives ``rows_in_successor`` 0 for a release and
``documents_in_successor`` 0 for a shard. A file that should hold records and is on
neither side is still a file the comparison lacks, and a wrong report still reads
"differ".

The world adds to the hand-built one a release ``R3`` of one row that copies the
generated paraphrase of A's record (the report retires it, so ``R3`` is rewritten to
no row) and a shard ``shard-9`` of one document made from that row (retired the same
way). Both are registered through the store's write methods, and the report is the one
``report withdraw A`` prints. The sides are files written here.
"""

from __future__ import annotations

import json
from pathlib import Path

from waterwhite import vocabulary as v

from rb_kit import (
    build, copy_shard, report_through_cli, run, sides_document, ww, write_json, write_lines,
)

R1_LINES = ["passage X", "z one", "z two"]
R2_LINES = ["passage X", "b two", "z three"]
DESCENDANT = "paraphrase of X"
EMPTY_RELEASE, EMPTY_SHARD = "R3", "shard-9"
EMPTY = (EMPTY_RELEASE, EMPTY_SHARD)
ALL_FILES = ("R1", "R2", "shard-1", "shard-3")


def emptying_world(folder: Path):
    """The hand-built world with the release and the shard the repair leaves empty."""
    world = build(folder)
    store = world.store
    paraphrase = world["gen/P"]
    r3 = world.ids[EMPTY_RELEASE] = ww.record(
        store, folder / "R3.jsonl", ["the paraphrase of X"], EMPTY_RELEASE, v.ROLE_RELEASE,
        ww.step("release", "R3"), [paraphrase], [ww.copy_edge(1, ww.ref(paraphrase, 1))])
    shard, _ = ww.shard(store, folder / "shards/shard-9", EMPTY_SHARD, [r3],
                        [ww.recovered(0, ww.ref(r3, 1))])
    world.ids[EMPTY_SHARD] = shard
    return world


def write_sides(world, *, leave_out=(), repaired_r1=R1_LINES, flip_shard_3=False) -> Path:
    """The sides file of the four files the repair writes, every one on both sides
    but the ones named in ``leave_out``. ``repaired_r1`` and ``flip_shard_3`` make the
    repaired side differ from the reference."""
    folder = world.folder
    repaired, reference = {}, {}
    texts = {"R1": R1_LINES, "R2": R2_LINES}
    for name in ("R1", "R2"):
        if name in leave_out:
            continue
        write_lines(folder / f"sides/repaired/{name}.jsonl", repaired_r1 if name == "R1" else texts[name])
        write_lines(folder / f"sides/reference/{name}.jsonl", texts[name])
        repaired[name], reference[name] = f"sides/repaired/{name}.jsonl", f"sides/reference/{name}.jsonl"
    for name in ("shard-1", "shard-3"):
        if name in leave_out:
            continue
        copy_shard(folder, f"shards/{name}", f"sides/repaired/{name}", flip=flip_shard_3 and name == "shard-3")
        copy_shard(folder, f"shards/{name}", f"sides/reference/{name}")
        repaired[name], reference[name] = f"sides/repaired/{name}", f"sides/reference/{name}"
    return write_json(folder / "sides.json", sides_document(repaired, reference))


def verified(world, sides: Path, report: Path | None = None, *extra: str) -> dict:
    report = report or write_json(world.folder / "report.json", report_through_cli(world))
    code, out, err = run(world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded", *extra)
    assert code == 0, err
    return json.loads(out)


def report_entry(world, name: str) -> dict:
    found = [e for e in report_through_cli(world)["artifacts"] if e["artifact"] == name]
    assert len(found) == 1, f"{name}: {len(found)} entries"
    return found[0]


def said_about(result: dict, name: str) -> list[str]:
    """The sentences the verification gives about ``name`` in its comparison entries."""
    said = []
    for key in ("differences", "releases", "shards"):
        for entry in result[key]:
            if entry.get("artifact") == name:
                said.extend(sorted(text for field, text in entry.items()
                                   if field != "artifact" and isinstance(text, str) and " " in text))
    return said


# --- The report's own record of what is left -------------------------------------------------------


def test_the_report_shows_the_release_and_the_shard_left_with_no_record(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        release, shard = report_entry(world, EMPTY_RELEASE), report_entry(world, EMPTY_SHARD)
        assert (release["action"], release["rows_in_successor"]) == (v.ACTION_REWRITE, 0)
        assert (shard["action"], shard["documents_in_successor"]) == (v.ACTION_REWRITE, 0)
        for name in ("R1", "shard-3"):
            held = report_entry(world, name)
            assert held["action"] == v.ACTION_REWRITE
            assert (held["rows_in_successor"] or held["documents_in_successor"]) > 0
    finally:
        world.store.close()


# --- What an emptied artifact does not block --------------------------------------------------------


def test_a_release_left_with_no_row_and_on_neither_side_does_not_block_equal_multisets(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        result = verified(world, write_sides(world, leave_out=(EMPTY_RELEASE,)))
        assert result["unresolved_records"] == 0
        assert [(e["artifact"], e["equal"]) for e in result["releases"]] == [("R1", True), ("R2", True)]
        assert result["multisets"] == v.SIDES_EQUAL, result["multisets"]
        assert result["differing_records"] == 0
    finally:
        world.store.close()


def test_a_shard_left_with_no_document_and_on_neither_side_does_not_block_equal_digests(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        result = verified(world, write_sides(world, leave_out=(EMPTY_SHARD,)))
        assert [(e["artifact"], e["equal"]) for e in result["shards"]] == [
            ("shard-1", True), ("shard-3", True)]
        assert result["per_shard_digests"] == v.SIDES_EQUAL, result["per_shard_digests"]
    finally:
        world.store.close()


def test_each_emptied_artifact_is_named_with_a_reason_that_is_not_the_one_a_lost_file_gets(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        # Two files that hold records are left out beside the two that hold none.
        result = verified(world, write_sides(world, leave_out=(*EMPTY, "R2", "shard-1")))
        for emptied, lost in ((EMPTY_RELEASE, "R2"), (EMPTY_SHARD, "shard-1")):
            reason, lost_reason = said_about(result, emptied), said_about(result, lost)
            assert reason, f"{emptied} is not named with a reason in {sorted(result)}"
            assert lost_reason, f"{lost} is not named with a reason"
            assert reason != lost_reason, f"{emptied} gets the reason of {lost}: {reason}"
    finally:
        world.store.close()


# --- What stays as it was --------------------------------------------------------------------------


def test_a_file_that_should_hold_records_and_is_on_neither_side_still_blocks_equal(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        for lost in ("R1", "R2"):
            result = verified(world, write_sides(world, leave_out=(*EMPTY, lost)))
            assert result["multisets"] != v.SIDES_EQUAL, lost
            assert said_about(result, lost), lost
        for lost in ("shard-1", "shard-3"):
            result = verified(world, write_sides(world, leave_out=(*EMPTY, lost)))
            assert result["per_shard_digests"] != v.SIDES_EQUAL, lost
            assert said_about(result, lost), lost
    finally:
        world.store.close()


def test_a_wrong_report_still_reads_differ_beside_an_emptied_artifact(tmp_path):
    world = emptying_world(tmp_path / "w")
    try:
        # A descendant of A's record that the report retires is still in the repaired
        # release, and a byte of a repaired shard is not the reference's.
        result = verified(world, write_sides(world, leave_out=EMPTY, repaired_r1=[*R1_LINES, DESCENDANT]))
        assert (result["multisets"], result["differing_records"]) == (v.SIDES_DIFFER, 1)
        result = verified(world, write_sides(world, leave_out=EMPTY, flip_shard_3=True))
        assert result["per_shard_digests"] == v.SIDES_DIFFER
        assert [e["equal"] for e in result["shards"]] == [True, False]
    finally:
        world.store.close()
