"""What ``verify`` does with the fourth word and with a shard both builds wrote that the
report does not name (DECISIONS.md D-141; docs/interfaces.md 7.2 to 7.4; docs/rebuild.md 3
and 5; docs/cli.md "verify").

The word, "clean, place not stated", is an action ``verify`` reads in a report: it is
accepted; the check of kept artifacts (byte identity of the file the repaired side names)
never applies to it, since the shard may have received a copy; no successor is required for
it; and a mixture that reads it is read as a mixture the report leaves unresolved (the
caches of its segments say no corrected mixture is named).

The comparison of the two sides does not read "equal" while a shard both builds wrote goes
uncompared. The shards each build wrote are read from the stores and the disk, not from the
sides file: the repaired side's are the shards the store registers as ``successor of`` and
the layout's prefix (here ``set/``, the folder the report's shards share), the reference
side's are the shard files in the folder of the reference shards the sides file names.
Where such a shard is neither in the report nor in the sides file, the multisets and the
digests do not read "equal", and the shard is named in ``differences`` with a note. A shard
only one build wrote is a difference.

The hand-built world with the successors is used (``rb_kit.py``). The sides are files
written here by hand, and the shards of the sides are copies of the files of a shard of
the world.
"""

from __future__ import annotations

import pytest
from waterwhite import vocabulary as v
from waterwhite.errors import Refused

from keep_kit import WORD, add_mixture, add_run, add_shard
from rb_kit import build, copy_shard, hand_report, need_attr, sides_document, write_lines

CLEAN = ["passage X", "z one", "z two"]


def verify(world, report, sides=None, base=None, **options):
    function = need_attr("waterwhite.rebuild.verify_sides", "verify")
    try:
        return function(world.store, report, sides=sides, base=base, **options)
    except Refused as refused:
        raise AssertionError(f"verify refused the report: {refused}") from refused


def test_the_word_is_an_artifact_action_verify_accepts_with_no_check_of_kept_bytes(repair_world, tmp_path):
    add_shard(repair_world, "xs/a", [("R2", 3)])
    # the repaired side names a file that is not the one the shard was registered as
    copy_shard(repair_world.folder, "xs/a", "repaired/xs-a", flip=True)
    sides = sides_document({"xs/a": "repaired/xs-a"}, {})
    for action, expected in ((v.ACTION_KEEP, [False]), (WORD, [])):
        report = hand_report(repair_world, [("xs/a", action, None)])
        result = verify(repair_world, report, sides, repair_world.folder)
        assert [entry["identical"] for entry in result["kept"] if entry["artifact"] == "xs/a"] == expected


def test_no_successor_is_required_for_a_shard_or_a_mixture_that_reads_the_word(repair_world):
    add_shard(repair_world, "xs/a", [("R2", 3)])
    add_mixture(repair_world, "mix-a", ["xs/a"])
    report = hand_report(repair_world, [
        ("xs/a", WORD, None), ("mix-a", WORD, None), ("R1", "rewrite", "successor of R1")])
    result = verify(repair_world, report)
    assert [entry["artifact"] for entry in result["successors"]] == ["R1"]
    assert [entry["artifact"] for entry in result["history"]] == ["R1"]


def caches_of(world, mixture_action: str) -> list[dict]:
    """The caches entries of run V: its first segment on mix-a, its second on mix-2."""
    add_shard(world, "xs/a", [("R2", 3)])
    add_mixture(world, "mix-a", ["xs/a"])
    add_run(world, "V", [("V/seg-1", "mix-a"), ("V/seg-2", "mix-2")])
    report = hand_report(world, [
        ("mix-a", mixture_action, None), ("mix-2", "rewrite", "successor of mix-2")])
    return verify(world, report)["caches"]


def test_a_mixture_that_reads_the_word_is_read_as_one_the_report_leaves_unresolved(repair_world):
    entries = {e["segment"]: e for e in caches_of(repair_world, WORD)}
    first = entries["V/seg-1"]
    assert first["stale"] is None and first["corrected_mixture"] is None
    assert first["corrected_digest"] is None and first["rebuilt"] is False
    assert isinstance(first["reason"], str) and first["reason"]


def test_the_entry_of_the_word_is_the_entry_of_an_unresolved_mixture_but_for_the_reason(
        repair_world, tmp_path):
    other = build(tmp_path / "other", successors=True)
    try:
        unresolved = {e["segment"]: e for e in caches_of(other, "unresolved")}["V/seg-1"]
    finally:
        other.store.close()
    word = {e["segment"]: e for e in caches_of(repair_world, WORD)}["V/seg-1"]
    assert {k: x for k, x in word.items() if k not in ("reason", "mixture")} == {
        k: x for k, x in unresolved.items() if k not in ("reason", "mixture")}


# --- A shard both builds wrote that nothing names --------------------------------------------------------------


def sides_world(world, *, uncompared: bool, only_repaired=()):
    """The world with a release R1 and a shard set/shard-A, rewritten, with its two sides
    equal; and, as the arguments say, the shards of folder ``set/`` the builds wrote that
    neither the report nor the sides file names. Returns the report, the sides and the base."""
    folder = world.folder
    add_shard(world, "set/shard-A", [("R2", 3)])
    add_shard(world, "successor of set/shard-A", [("R2", 3)])
    copy_shard(folder, "successor of set/shard-A", "reference/shard-A")
    if uncompared:
        add_shard(world, "successor of set/shard-B", [("R2", 2)])
        copy_shard(folder, "successor of set/shard-B", "reference/shard-B")
    for name in only_repaired:
        add_shard(world, f"successor of set/{name}", [("R2", 2)])
    write_lines(folder / "repaired/R1.jsonl", CLEAN)
    write_lines(folder / "reference/R1.jsonl", CLEAN)
    report = hand_report(world, [("R1", "rewrite", "successor of R1"),
                                 ("set/shard-A", "rewrite", "successor of set/shard-A")])
    sides = sides_document(
        {"R1": "repaired/R1.jsonl", "set/shard-A": "successor of set/shard-A"},
        {"R1": "reference/R1.jsonl", "set/shard-A": "reference/shard-A"})
    return report, sides, folder


def test_with_nothing_written_beyond_the_report_the_two_sides_read_equal(repair_world):
    report, sides, base = sides_world(repair_world, uncompared=False)
    result = verify(repair_world, report, sides, base, mode="seeded")
    assert (result["multisets"], result["per_shard_digests"]) == ("equal", "equal")
    assert result["differences"] == []


def test_a_shard_both_builds_wrote_that_nothing_names_stops_both_readings_from_equal(repair_world):
    report, sides, base = sides_world(repair_world, uncompared=True)
    result = verify(repair_world, report, sides, base, mode="seeded")
    assert result["multisets"] != "equal" and result["per_shard_digests"] != "equal"
    [named] = [d for d in result["differences"] if d["artifact"] == "set/shard-B"]
    assert isinstance(named["note"], str) and named["note"]
    assert named["repaired"] is None and named["reference"] is None


def test_a_shard_only_the_reference_build_wrote_is_a_difference(repair_world):
    report, sides, base = sides_world(repair_world, uncompared=False)
    copy_shard(base, "successor of set/shard-A", "reference/shard-C")
    result = verify(repair_world, report, sides, base, mode="seeded")
    assert "set/shard-C" in [d["artifact"] for d in result["differences"]]
    assert result["per_shard_digests"] != "equal"


def test_a_shard_only_the_repaired_build_wrote_is_a_difference(repair_world):
    report, sides, base = sides_world(repair_world, uncompared=False, only_repaired=("shard-D",))
    result = verify(repair_world, report, sides, base, mode="seeded")
    assert "set/shard-D" in [d["artifact"] for d in result["differences"]]
    assert result["per_shard_digests"] != "equal"


# --- A shard both builds wrote that the report names under a layout's landing ----------------------------------
#
# Where the report states a placement, the shard a copy lands in may be one the historical
# store does not hold: the report has no artifact entry for it and names it under
# ``groups.placements[NAME].landing`` (docs/interfaces.md 7.1). Both builds wrote it. The
# repaired and the reference files of such a shard are named in the sides file, as they are
# for every compared shard (docs/interfaces.md 7.2), and the shard is compared like a
# rewritten one: the report names it, so it does not go uncompared (docs/rebuild.md 2.2).

LANDING = "set/shard-B"


def with_landing(report: dict, *shards: str) -> dict:
    """The report with the landing entry of a layout ``partitioned`` that names the shards,
    in the form of docs/interfaces.md 7.1: a placement read, and the copies each receives."""
    report["groups"] = {"placements": {"partitioned": {
        "placement": {"path": "placement.json", "sha256": "0" * 64},
        "landing": [{"shard": name, "in_store": False, "receives": ["src/B:L1"]} for name in shards],
        "place_not_stated": []}}}
    return report


def landing_world(world, *, repaired: bool = True, reference: bool = True, flip: bool = False):
    """The world of ``sides_world`` with set/shard-B written by both builds (a successor
    registered in the store, a shard in the folder of the reference build) and named in the
    report's landing. The sides file names it on the side or sides the arguments say, and
    the reference copy has one byte of its token file changed when ``flip``. Everything
    else compares equal: R1 and set/shard-A, which the report names."""
    report, sides, base = sides_world(world, uncompared=True)
    if flip:
        copy_shard(base, "successor of set/shard-B", "reference/shard-B", flip=True)
    if repaired:
        sides["repaired"][LANDING] = "successor of set/shard-B"
    if reference:
        sides["reference"][LANDING] = "reference/shard-B"
    return with_landing(report, LANDING), sides, base


def named_in_differences(result: dict, name: str) -> bool:
    return name in [entry["artifact"] for entry in result["differences"]]


def only_the_landing_shard_is_open(result: dict):
    """The preconditions of a test about the landing shard alone: nothing else in the
    world is a reason for a reading other than "equal"."""
    assert result["unresolved_records"] == 0
    assert [(e["artifact"], e["equal"]) for e in result["releases"]] == [("R1", True)]
    assert ("set/shard-A", True) in [(e["artifact"], e["equal"]) for e in result["shards"]]


def test_a_landing_shard_both_sides_name_with_equal_bytes_is_compared_and_reads_equal(repair_world):
    report, sides, base = landing_world(repair_world)
    result = verify(repair_world, report, sides, base, mode="seeded")
    only_the_landing_shard_is_open(result)
    assert [(e["artifact"], e["equal"]) for e in result["shards"] if e["artifact"] == LANDING] == [
        (LANDING, True)]
    assert not named_in_differences(result, LANDING)
    assert result["per_shard_digests"] == "equal"
    # the release files of the world compare equal and the landing shard is compared: the
    # multisets read "equal" (the measure of the table row c09.10k.multiset)
    assert result["multisets"] == "equal"
    assert result["differences"] == []


def test_a_landing_shard_both_sides_name_with_differing_bytes_reads_differ(repair_world):
    report, sides, base = landing_world(repair_world, flip=True)
    result = verify(repair_world, report, sides, base, mode="seeded")
    only_the_landing_shard_is_open(result)
    assert [(e["artifact"], e["equal"]) for e in result["shards"] if e["artifact"] == LANDING] == [
        (LANDING, False)]
    assert result["per_shard_digests"] == "differ"


# The guards against a landing name that is only added to the names the report holds, so that
# the shard drops out of the uncompared ones: both builds wrote the shard, and the sides file
# does not name its files on both sides, so nothing compares it.


@pytest.mark.parametrize("on_sides", [
    pytest.param(dict(repaired=False, reference=False), id="neither-side"),
    pytest.param(dict(repaired=True, reference=False), id="the-repaired-side-only"),
    pytest.param(dict(repaired=False, reference=True), id="the-reference-side-only"),
])
def test_a_landing_shard_both_builds_wrote_that_the_sides_do_not_both_name_is_not_equal(
        repair_world, on_sides):
    report, sides, base = landing_world(repair_world, **on_sides)
    result = verify(repair_world, report, sides, base, mode="seeded")
    only_the_landing_shard_is_open(result)
    assert result["multisets"] != "equal", result["multisets"]
    assert result["per_shard_digests"] != "equal", result["per_shard_digests"]
    assert named_in_differences(result, LANDING), [d["artifact"] for d in result["differences"]]
