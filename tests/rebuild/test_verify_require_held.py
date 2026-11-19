"""``verify --require-held`` (docs/cli.md "verify", "Gating on the verification"): the
verification is printed as it always is, and the exit code says whether its conditions
hold. Without the flag every verification that is made exits with 0 (docs/cli.md,
"verify": a difference is an answer).

The expected exit codes and the unmet-condition names come from the list in
docs/cli.md, never from running the command.
"""

from __future__ import annotations

import json

from rb_kit import copy_shard, hand_report, run, sides_document, write_json, write_lines

CLEAN = ["passage X", "z one", "z two"]
NOT_HELD = 5
# A repaired file with a line the registered successor lacks is not the file the successor
# was registered as (docs/interfaces.md 7.3, `matches_sides`), so it is unmet beside the
# difference of the multisets.
DIFFER_UNMET = ["multisets: differ", "successor successor of R1: not the file compared"]


def error_of(err: str) -> dict:
    assert err.strip(), "nothing was written to standard error"
    return json.loads(err)


def unmet_of(err: str) -> list[str]:
    """The conditions the command names on standard error: one JSON object."""
    named = json.loads(err)
    assert list(named) == ["verification", "unmet"], named
    assert named["verification"] == "not held"
    return named["unmet"]


def rewritten(world, tmp_path, repaired=CLEAN, reference=CLEAN, unresolved=0):
    report = hand_report(world, [("R1", "rewrite", "successor of R1")], unresolved=unresolved)
    write_lines(tmp_path / "repaired/R1.jsonl", repaired)
    write_lines(tmp_path / "reference/R1.jsonl", reference)
    sides = sides_document({"R1": str(tmp_path / "repaired/R1.jsonl")},
                           {"R1": str(tmp_path / "reference/R1.jsonl")})
    return write_json(tmp_path / "report.json", report), write_json(tmp_path / "sides.json", sides)


def test_a_verification_that_holds_exits_with_0_and_prints_the_same_document(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path)
    code, plain, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides))
    assert (code, err) == (0, "")
    code, gated, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                           "--require-held")
    assert (code, err) == (0, "")
    assert gated == plain
    assert json.loads(gated)["multisets"] == "equal"


def test_a_difference_between_the_sides_exits_with_5_and_prints_the_document(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path, repaired=CLEAN + ["paraphrase of X"])
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["multisets"] == "differ"
    assert unmet_of(err) == DIFFER_UNMET


def test_without_the_flag_a_difference_still_exits_with_0(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path, repaired=CLEAN + ["paraphrase of X"])
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides))
    assert (code, err) == (0, "")
    assert json.loads(out)["multisets"] == "differ"


def test_an_unresolved_record_is_not_claimed_and_exits_with_5(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path, unresolved=1)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["multisets"] == "not claimed"
    assert "multisets: not claimed" in unmet_of(err)


def test_no_sides_file_compares_nothing_and_exits_with_5(repair_world, tmp_path):
    report, _ = rewritten(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["multisets"] == "not compared"
    assert unmet_of(err) == ["sides: not given", "multisets: not compared"]


def kept_shard(world, *, reference_flipped=False, with_sides=True):
    """A report that keeps shard-1, and a sides file in the world's folder that names a
    copy of it on the repaired side and a copy (one byte changed when asked) on the
    reference side."""
    copy_shard(world.folder, "shards/shard-1", "sides/repaired/shard-1")
    copy_shard(world.folder, "shards/shard-1", "sides/reference/shard-1", flip=reference_flipped)
    report = write_json(world.folder / "report.json", hand_report(world, [("shard-1", "keep", None)]))
    sides = write_json(world.folder / "sides.json", sides_document(
        {"shard-1": "sides/repaired/shard-1"}, {"shard-1": "sides/reference/shard-1"}))
    return report, sides


def test_a_seeded_verification_of_a_report_with_no_shard_does_not_ask_for_shard_digests(
        repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded", "--require-held")
    assert (code, err) == (0, "")
    assert json.loads(out)["per_shard_digests"] == "not compared"


def test_a_report_of_one_kept_shard_holds_when_both_sides_agree_in_the_seeded_mode(repair_world):
    report, sides = kept_shard(repair_world)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded", "--require-held")
    assert (code, err) == (0, "")
    document = json.loads(out)
    assert (document["per_shard_digests"], document["multisets"]) == ("equal", "not compared")


def test_a_shard_whose_sides_differ_in_the_seeded_mode_exits_with_5(repair_world):
    report, sides = kept_shard(repair_world, reference_flipped=True)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded", "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["per_shard_digests"] == "differ"
    assert unmet_of(err) == ["per_shard_digests: differ"]


def test_the_seeded_mode_with_a_shard_and_no_sides_file_exits_with_5(repair_world):
    report, _ = kept_shard(repair_world)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--mode", "seeded",
                         "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["per_shard_digests"] == "not compared"
    assert unmet_of(err) == ["sides: not given", "per_shard_digests: not compared",
                             "kept shard-1: not compared"]


def test_the_unseeded_mode_does_not_ask_for_shard_digests(repair_world):
    report, sides = kept_shard(repair_world)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--require-held")
    assert (code, err) == (0, "")
    assert json.loads(out)["per_shard_digests"] == "not compared"


def test_a_file_named_on_one_side_only_is_unmet_in_differences(repair_world, tmp_path):
    report, _ = rewritten(repair_world, tmp_path)
    one_side = write_json(tmp_path / "one-side.json", sides_document(
        {"R1": str(tmp_path / "repaired/R1.jsonl")}, {}))
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(one_side),
                         "--require-held")
    assert code == NOT_HELD
    unmet = unmet_of(err)
    assert unmet[0] == "multisets: not compared"
    assert [line for line in unmet if line.startswith("differences R1: ")], unmet


def test_a_kept_artifact_that_is_not_identical_exits_with_5(repair_world, tmp_path):
    write_lines(tmp_path / "repaired/R2.jsonl", ["passage X", "b two"])
    sides = write_json(tmp_path / "sides.json",
                       sides_document({"R2": str(tmp_path / "repaired/R2.jsonl")}, {}))
    report = write_json(tmp_path / "report.json",
                        hand_report(repair_world, [("R2", "keep", None)]))
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["kept"][0]["identical"] is False
    assert "kept R2: not identical" in unmet_of(err)


def test_a_kept_file_that_the_repaired_side_does_not_name_exits_with_5(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path)
    kept = write_json(tmp_path / "kept-report.json", hand_report(
        repair_world, [("R1", "rewrite", "successor of R1"), ("R2", "keep", None)]))
    code, out, err = run(repair_world, "verify", "--report", str(kept), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["kept"][0]["identical"] is None
    assert "kept R2: not compared" in unmet_of(err)


def test_a_refusal_under_the_flag_keeps_its_code_and_prints_nothing(repair_world, tmp_path):
    report = write_json(tmp_path / "report.json",
                        hand_report(repair_world, [("R1", "rewrite", "successor of R1")], source="Q"))
    code, out, err = run(repair_world, "verify", "--report", str(report), "--require-held")
    assert (code, out, error_of(err)["error"]) == (2, "", "source not registered")


def test_the_unmet_conditions_are_one_line_in_the_text_form(repair_world, tmp_path):
    report, sides = rewritten(repair_world, tmp_path, repaired=CLEAN + ["paraphrase of X"])
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--require-held", "--format", "text")
    assert code == NOT_HELD
    assert out.startswith("verification of ")
    assert err == "not held: " + "; ".join(DIFFER_UNMET) + "\n"


def test_a_report_that_names_a_shard_and_no_sides_file_never_holds(repair_world, tmp_path):
    # Nothing is compared without a sides file, whatever the kinds the report names:
    # the successor of the shard answers "does not contain" (docs/cli.md, the conditions),
    # and the verification still does not hold.
    report = write_json(tmp_path / "report.json", hand_report(
        repair_world, [("shard-3", "rewrite", "successor of shard-3")]))
    code, out, err = run(repair_world, "verify", "--report", str(report), "--require-held")
    assert code == NOT_HELD
    assert json.loads(out)["successors"][0]["answer"] == "does not contain"
    assert unmet_of(err) == ["sides: not given"]


def test_a_successor_of_another_kind_than_the_artifact_it_replaces_never_holds(repair_world):
    """Sol's review of a011cd5, finding 2: the report rewrites a shard, a clean mixture is
    registered under the expected successor name, and both sides name the original shard.
    A mixture has no file to compare, so ``matches_sides`` is null and nothing said the
    successor is the wrong kind of artifact: the gate held and the exit code was 0."""
    folder = repair_world.folder
    copy_shard(folder, "shards/shard-3", "sides/both/shard-3")
    report = write_json(folder / "report.json", hand_report(
        repair_world, [("shard-3", "rewrite", "successor of mix-2")]))
    both = {"shard-3": str(folder / "sides/both/shard-3")}
    sides = write_json(folder / "sides.json", sides_document(both, both))
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded", "--require-held")
    made = json.loads(out)
    assert [(e["kind"], e["matches_sides"]) for e in made["successors"]] == [("mixture", None)]
    assert made["per_shard_digests"] == "equal"
    assert (code, unmet_of(err)) == (
        NOT_HELD, ["successor successor of mix-2: a mixture, not the shard it replaces"])

