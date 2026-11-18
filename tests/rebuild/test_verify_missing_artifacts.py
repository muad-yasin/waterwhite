"""A verification never says "equal" over a set that lacks an artifact of the report, and
a kept artifact is compared only with the instance the report names (plan 4.2, 4.4;
docs/interfaces.md section 7.3).

- every artifact of the report that is kept or rewritten and is a JSONL file or a shard
  belongs to a comparison (the multisets of lines, the digests of shards); one that the
  sides do not name on both sides is named in the verification and its comparison does
  not read "equal". The form of the naming is the code's: a new key, a note in the
  differences or a field of an entry all count (``names_it`` below);
- ``identical`` of a kept artifact compares the repaired file with the content digest of
  the report's own instance and with nothing else: where the store does not hold that
  instance it is null, and the entry says why.
"""

from __future__ import annotations

import json

from rb_kit import copy_shard, hand_report, need_attr, sides_document, write_lines
from waterwhite import vocabulary as v

CLEAN = ["passage X", "z one", "z two"]
R2_LINES = ["passage X", "b two", "z three"]
# The keys of the entries of ``kept``, ``successors`` and ``history`` before this ruling.
KNOWN_ENTRY_KEYS = {
    "kept": {"artifact", "identical", "recorded", "found"},
    "successors": {"artifact", "successor", "kind", "registered", "instance", "answer", "names",
                   "holds_unresolved", "matches_sides", "answered"},
    "history": {"artifact", "instance", "instance_unchanged", "answer"},
}


def verify(world, report, sides=None, base=None, **options):
    function = need_attr("waterwhite.rebuild.verify_sides", "verify")
    return function(world.store, report, sides=sides, base=base, **options)


def names_it(result: dict, name: str) -> bool:
    """Whether the verification names ``name`` as a file the comparison lacks. Where
    the name stands in the entries it always had (``kept``, ``successors``, ``history``)
    it must stand there with a key that did not exist before; anywhere else it counts."""
    elsewhere = {key: held for key, held in result.items()
                 if key not in ("request", *KNOWN_ENTRY_KEYS)}
    if f'"{name}"' in json.dumps(elsewhere):
        return True
    for key, known in KNOWN_ENTRY_KEYS.items():
        for entry in result.get(key, []):
            if entry.get("artifact") == name and set(entry) - known:
                return True
    return False


def two_releases(tmp_path, *, with_r2_repaired=True, with_r2_reference=True):
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN)
    write_lines(tmp_path / "reference/R1.jsonl", CLEAN)
    repaired, reference = {"R1": "repaired/R1.jsonl"}, {"R1": "reference/R1.jsonl"}
    if with_r2_repaired:
        write_lines(tmp_path / "repaired/R2.jsonl", R2_LINES)
        repaired["R2"] = "repaired/R2.jsonl"
    if with_r2_reference:
        write_lines(tmp_path / "reference/R2.jsonl", R2_LINES)
        reference["R2"] = "reference/R2.jsonl"
    return sides_document(repaired, reference)


def releases_report(world):
    return hand_report(world, [("R1", "rewrite", "successor of R1"), ("R2", "keep", None)])


def shard_report(world):
    return hand_report(world, [("shard-3", "rewrite", "successor of shard-3"),
                               ("shard-1", "keep", None)])


# --- The multisets of the lines of the files ----------------------------------------------------


def test_a_kept_file_the_sides_do_not_name_at_all_keeps_the_multisets_from_reading_equal(
        repair_world, tmp_path):
    # With every artifact named on both sides the multisets read equal; that is what a
    # missing one must take away.
    complete = verify(repair_world, releases_report(repair_world), two_releases(tmp_path), tmp_path)
    assert complete["multisets"] == v.SIDES_EQUAL
    assert [entry["artifact"] for entry in complete["releases"]] == ["R1", "R2"]
    sides = sides_document({"R1": "repaired/R1.jsonl"}, {"R1": "reference/R1.jsonl"})
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN)
    write_lines(tmp_path / "reference/R1.jsonl", CLEAN)
    result = verify(repair_world, releases_report(repair_world), sides, tmp_path)
    assert result["multisets"] in v.SIDES_RESULTS
    assert result["multisets"] != v.SIDES_EQUAL
    assert names_it(result, "R2")


def test_a_file_named_on_the_repaired_side_only_keeps_the_multisets_from_reading_equal(
        repair_world, tmp_path):
    sides = two_releases(tmp_path, with_r2_reference=False)
    result = verify(repair_world, releases_report(repair_world), sides, tmp_path)
    assert result["multisets"] != v.SIDES_EQUAL and result["multisets"] in v.SIDES_RESULTS
    assert names_it(result, "R2")


def test_a_file_named_on_the_reference_side_only_keeps_the_multisets_from_reading_equal(
        repair_world, tmp_path):
    sides = two_releases(tmp_path, with_r2_repaired=False)
    result = verify(repair_world, releases_report(repair_world), sides, tmp_path)
    assert result["multisets"] != v.SIDES_EQUAL and result["multisets"] in v.SIDES_RESULTS
    assert names_it(result, "R2")


def test_a_rewritten_file_the_sides_do_not_name_keeps_the_multisets_from_reading_equal(
        repair_world, tmp_path):
    write_lines(tmp_path / "repaired/R2.jsonl", R2_LINES)
    write_lines(tmp_path / "reference/R2.jsonl", R2_LINES)
    sides = sides_document({"R2": "repaired/R2.jsonl"}, {"R2": "reference/R2.jsonl"})
    result = verify(repair_world, releases_report(repair_world), sides, tmp_path)
    assert result["multisets"] != v.SIDES_EQUAL and result["multisets"] in v.SIDES_RESULTS
    assert names_it(result, "R1")


# --- The digests of the shards ---------------------------------------------------------------------


def shard_sides(world, *names: str):
    folder = world.folder
    repaired, reference = {}, {}
    for name in names:
        copy_shard(folder, f"shards/{name}", f"sides/repaired/{name}")
        copy_shard(folder, f"shards/{name}", f"sides/reference/{name}")
        repaired[name], reference[name] = f"sides/repaired/{name}", f"sides/reference/{name}"
    return sides_document(repaired, reference)


def test_a_kept_shard_the_sides_do_not_name_keeps_the_digests_from_reading_equal(repair_world):
    complete = verify(repair_world, shard_report(repair_world), shard_sides(repair_world, "shard-3", "shard-1"),
                      repair_world.folder, mode=v.BUILD_SEEDED)
    assert complete["per_shard_digests"] == v.SIDES_EQUAL
    sides = shard_sides(repair_world, "shard-3")
    result = verify(repair_world, shard_report(repair_world), sides, repair_world.folder,
                    mode=v.BUILD_SEEDED)
    assert result["per_shard_digests"] != v.SIDES_EQUAL
    assert result["per_shard_digests"] in v.SIDES_RESULTS
    assert names_it(result, "shard-1")


def test_a_rewritten_shard_the_sides_do_not_name_keeps_the_digests_from_reading_equal(repair_world):
    sides = shard_sides(repair_world, "shard-1")
    result = verify(repair_world, shard_report(repair_world), sides, repair_world.folder,
                    mode=v.BUILD_SEEDED)
    assert result["per_shard_digests"] != v.SIDES_EQUAL
    assert names_it(result, "shard-3")


# --- identical compares with the report's own instance --------------------------------------------------


def test_a_kept_entry_whose_instance_the_store_does_not_hold_is_not_compared_with_the_name(
        repair_world, tmp_path):
    kept_entry = need_attr("waterwhite.rebuild.verify_sides", "_kept_entry")
    store = repair_world.store
    resolved = store.resolve("R2")
    # The file the repaired side names is the file the name R2 resolves to now.
    copy = tmp_path / "repaired/R2.jsonl"
    copy.parent.mkdir(parents=True)
    copy.write_bytes((repair_world.folder / "R2.jsonl").read_bytes())
    entry = {"artifact": "R2", "instance": repair_world["R2"], "kind": v.KIND_JSONL,
             "action": v.ACTION_KEEP, "successor": None}
    # The instance the store holds is the one the file is compared with.
    held = kept_entry(store, entry, {"R2": "repaired/R2.jsonl"}, tmp_path)
    assert held["identical"] is True and held["recorded"] == resolved.content_digest
    # The instance it does not hold: nothing is compared, whatever the name resolves to.
    entry = dict(entry, instance="0" * 64)
    assert store.artifact("0" * 64) is None
    held = kept_entry(store, entry, {"R2": "repaired/R2.jsonl"}, tmp_path)
    assert held["artifact"] == "R2"
    assert held["identical"] is None
    assert held["recorded"] is None and held["recorded"] != resolved.content_digest
    said = {key: text for key, text in held.items()
            if key not in KNOWN_ENTRY_KEYS["kept"] and isinstance(text, str) and text}
    assert said, f"the entry gives no reason: {held}"
