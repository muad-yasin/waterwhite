"""The comparison of the repaired side with the reference side (plan 4.2, 4.4,
Decision 6; docs/interfaces.md sections 7.2 and 7.3, docs/rebuild.md sections 3 and 5),
through ``verify`` on the hand-built world with the successors registered.

The sides are files written here by hand; the report is a document written here by
hand with the keys `verify` reads. The expected values are counts of lines and the
digests of ``waterwhite.ids``.
"""

from __future__ import annotations

import pytest
from waterwhite import errors, ids
from waterwhite import vocabulary as v

from rb_kit import (
    copy_shard, hand_report, need_attr, sides_document, write_json, write_lines,
)

CLEAN = ["passage X", "z one", "z two"]
DESCENDANT = "paraphrase of X"


def verify(world, report, sides=None, base=None, **options):
    function = need_attr("waterwhite.rebuild.verify_sides", "verify")
    return function(world.store, report, sides=sides, base=base, **options)


def release_report(world, *, unresolved: int = 0):
    return hand_report(world, [("R1", "rewrite", "successor of R1")], unresolved=unresolved)


def release_sides(tmp_path, repaired: list[str], reference: list[str]):
    write_lines(tmp_path / "repaired/R1.jsonl", repaired)
    write_lines(tmp_path / "reference/R1.jsonl", reference)
    return sides_document({"R1": "repaired/R1.jsonl"}, {"R1": "reference/R1.jsonl"})


# --- The multiset of (fingerprint, multiplicity) ---------------------------------------------


def test_sides_that_hold_the_same_lines_in_any_order_are_equal(repair_world, tmp_path):
    sides = release_sides(tmp_path, CLEAN, ["z two", "passage X", "z one"])
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert result["form"] == v.FORM_VERIFICATION
    assert (result["multisets"], result["differing_records"], result["differences"]) == ("equal", 0, [])
    assert result["unresolved_records"] == 0
    assert result["releases"] == [{"artifact": "R1", "repaired": 3, "reference": 3, "equal": True}]


def test_a_descendant_the_report_kept_makes_the_sides_differ_by_one_record(repair_world, tmp_path):
    sides = release_sides(tmp_path, CLEAN + [DESCENDANT], CLEAN)
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert (result["multisets"], result["differing_records"]) == ("differ", 1)
    [difference] = result["differences"]
    assert (difference["artifact"], difference["repaired"], difference["reference"]) == ("R1", 1, 0)
    assert difference["locators"] == ["L4"]
    assert isinstance(difference["fingerprint"], str) and difference["fingerprint"]
    assert result["releases"] == [{"artifact": "R1", "repaired": 4, "reference": 3, "equal": False}]


def test_a_record_the_report_dropped_that_the_clean_sources_keep_makes_the_sides_differ(repair_world, tmp_path):
    sides = release_sides(tmp_path, ["passage X", "z one"], CLEAN)
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert (result["multisets"], result["differing_records"]) == ("differ", 1)
    [difference] = result["differences"]
    assert (difference["repaired"], difference["reference"], difference["locators"]) == (0, 1, [])


def test_the_multiplicity_counts_not_only_the_presence(repair_world, tmp_path):
    sides = release_sides(tmp_path, CLEAN + ["z one"], CLEAN)
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert (result["multisets"], result["differing_records"]) == ("differ", 1)
    [difference] = result["differences"]
    assert (difference["repaired"], difference["reference"], len(difference["locators"])) == (2, 1, 2)


def test_the_difference_is_summed_over_every_fingerprint(repair_world, tmp_path):
    sides = release_sides(tmp_path, ["passage X", "z one", DESCENDANT], CLEAN)
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert (result["differing_records"], len(result["differences"])) == (2, 2)


def test_the_text_field_names_the_field_that_holds_the_text(repair_world, tmp_path):
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN, field="body")
    write_lines(tmp_path / "reference/R1.jsonl", CLEAN, field="body")
    sides = sides_document({"R1": "repaired/R1.jsonl"}, {"R1": "reference/R1.jsonl"})
    result = verify(repair_world, release_report(repair_world), sides, tmp_path, text_field="body")
    assert (result["multisets"], result["differing_records"]) == ("equal", 0)


def test_without_a_sides_file_nothing_is_compared_and_the_rest_is_made(repair_world):
    result = verify(repair_world, release_report(repair_world))
    assert result["multisets"] == "not compared" and result["per_shard_digests"] == "not compared"
    assert result["differences"] == []
    assert [entry["artifact"] for entry in result["successors"]] == ["R1"]


def test_a_name_on_one_side_only_is_not_compared_and_is_listed(repair_world, tmp_path):
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN)
    sides = sides_document({"R1": "repaired/R1.jsonl"}, {})
    result = verify(repair_world, release_report(repair_world), sides, tmp_path)
    assert result["multisets"] == "not compared"
    assert "R1" in str(result["differences"])


# --- Equality is claimed only for a report with no unresolved record ----------------------------


def test_with_an_unresolved_record_equality_is_not_claimed(repair_world, tmp_path):
    # No row of the table asks this: plan 4.2 says the claim is withheld.
    sides = release_sides(tmp_path, CLEAN, CLEAN)
    result = verify(repair_world, release_report(repair_world, unresolved=1), sides, tmp_path)
    assert result["unresolved_records"] == 1
    assert result["multisets"] == "not claimed"
    assert result["differing_records"] == 0 and result["differences"] == []


def test_with_an_unresolved_record_the_differences_are_still_listed(repair_world, tmp_path):
    sides = release_sides(tmp_path, CLEAN + [DESCENDANT], CLEAN)
    result = verify(repair_world, release_report(repair_world, unresolved=2), sides, tmp_path)
    assert result["unresolved_records"] == 2 and result["multisets"] == "not claimed"
    assert result["differing_records"] == 1 and len(result["differences"]) == 1


# --- The digests of the shards (seeded mode only) --------------------------------------------------


def shard_digest(prefix) -> str:
    return ids.shard_digest(ids.file_digest(f"{prefix}.bin"), ids.file_digest(f"{prefix}.idx"))


def shard_sides(world, flip: bool):
    folder = world.folder
    repaired = copy_shard(folder, "shards/shard-3", "sides/repaired/shard-3")
    reference = copy_shard(folder, "shards/shard-3", "sides/reference/shard-3", flip=flip)
    sides = sides_document({"shard-3": "sides/repaired/shard-3"}, {"shard-3": "sides/reference/shard-3"})
    report = hand_report(world, [("shard-3", "rewrite", "successor of shard-3")])
    return report, sides, repaired, reference


def test_in_seeded_mode_equal_shards_have_equal_digests(repair_world):
    report, sides, repaired, reference = shard_sides(repair_world, flip=False)
    result = verify(repair_world, report, sides, repair_world.folder, mode="seeded")
    assert result["mode"] == "seeded" and result["per_shard_digests"] == "equal"
    assert result["shards"] == [{"artifact": "shard-3", "repaired": shard_digest(repaired),
                                 "reference": shard_digest(reference), "equal": True}]


def test_in_seeded_mode_one_changed_byte_makes_the_digests_differ(repair_world):
    report, sides, repaired, reference = shard_sides(repair_world, flip=True)
    result = verify(repair_world, report, sides, repair_world.folder, mode="seeded")
    assert result["per_shard_digests"] == "differ"
    [shard] = result["shards"]
    assert shard["equal"] is False and shard["repaired"] != shard["reference"]
    assert shard["reference"] == shard_digest(reference)


def test_in_unseeded_mode_the_digests_are_listed_and_not_compared(repair_world):
    report, sides, repaired, reference = shard_sides(repair_world, flip=True)
    result = verify(repair_world, report, sides, repair_world.folder)
    assert result["mode"] == "unseeded" and result["per_shard_digests"] == "not compared"
    assert result["shards"][0]["repaired"] == shard_digest(repaired)


# --- What the report keeps is byte-identical afterwards ----------------------------------------------


def test_a_kept_release_that_was_not_touched_is_identical(repair_world, tmp_path):
    registered = repair_world.folder / "R2.jsonl"
    copy = tmp_path / "repaired/R2.jsonl"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(registered.read_bytes())
    sides = sides_document({"R2": "repaired/R2.jsonl"}, {})
    report = hand_report(repair_world, [("R2", "keep", None)])
    result = verify(repair_world, report, sides, tmp_path)
    digest = repair_world.store.artifact(repair_world["R2"]).content_digest
    assert result["kept"] == [{"artifact": "R2", "identical": True, "recorded": digest, "found": digest}]
    # A kept artifact has no successor and no history entry.
    assert result["successors"] == [] and result["history"] == []


def test_a_kept_release_that_was_rewritten_is_not_identical(repair_world, tmp_path):
    write_lines(tmp_path / "repaired/R2.jsonl", ["passage X", "b two"])
    sides = sides_document({"R2": "repaired/R2.jsonl"}, {})
    report = hand_report(repair_world, [("R2", "keep", None)])
    [kept] = verify(repair_world, report, sides, tmp_path)["kept"]
    assert kept["identical"] is False
    assert kept["found"] == ids.file_digest(tmp_path / "repaired/R2.jsonl")
    assert kept["recorded"] == repair_world.store.artifact(repair_world["R2"]).content_digest


def test_a_kept_shard_is_identical_when_its_two_files_are(repair_world):
    copy_shard(repair_world.folder, "shards/shard-1", "sides/repaired/shard-1")
    sides = sides_document({"shard-1": "sides/repaired/shard-1"}, {})
    report = hand_report(repair_world, [("shard-1", "keep", None)])
    [kept] = verify(repair_world, report, sides, repair_world.folder)["kept"]
    assert kept["identical"] is True and kept["found"] == kept["recorded"]


def test_a_kept_shard_with_one_changed_byte_is_not_identical(repair_world):
    copy_shard(repair_world.folder, "shards/shard-1", "sides/repaired/shard-1", flip=True)
    sides = sides_document({"shard-1": "sides/repaired/shard-1"}, {})
    report = hand_report(repair_world, [("shard-1", "keep", None)])
    [kept] = verify(repair_world, report, sides, repair_world.folder)["kept"]
    assert kept["identical"] is False and kept["found"] != kept["recorded"]


def test_without_a_file_for_a_kept_artifact_identity_is_not_known(repair_world):
    report = hand_report(repair_world, [("R2", "keep", None)])
    [kept] = verify(repair_world, report)["kept"]
    assert kept["artifact"] == "R2" and kept["identical"] is None


# --- The documents ----------------------------------------------------------------------------------------


def test_the_report_and_the_sides_are_read_as_files_of_their_form(repair_world, tmp_path):
    read_document = need_attr("waterwhite.rebuild.verify_sides", "read_document")
    good = release_report(repair_world)
    path = write_json(tmp_path / "report.json", good)
    assert read_document(path, v.FORM_REPORT, "report") == good
    with pytest.raises(errors.UsageError):
        read_document(tmp_path / "missing.json", v.FORM_REPORT, "report")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(errors.UsageError):
        read_document(tmp_path / "broken.json", v.FORM_REPORT, "report")
    with pytest.raises(errors.Refused) as refused:
        read_document(path, v.FORM_SIDES, "sides file")
    assert refused.value.name == "request refused" and refused.value.exit_code == 2


def test_a_file_the_sides_file_names_that_is_not_there_is_refused(repair_world, tmp_path):
    function = need_attr("waterwhite.rebuild.verify_sides", "verify")
    sides = sides_document({"R1": "repaired/absent.jsonl"}, {"R1": "reference/absent.jsonl"})
    with pytest.raises(errors.Refused):
        function(repair_world.store, release_report(repair_world), sides=sides, base=tmp_path)
