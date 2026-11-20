"""The outside review of 504bf64 (Sol's second review), findings 1 and 3 and the two
corrections that touch ``verify``: what a report must be before ``verify`` reads it.

A report's entries are keyed by artifact name in the gate, so an entry has to be one
entry for one artifact: a name named twice, or a name that is not the one the entry's
instance is held under, let a later entry stand in for an earlier one. A report field
that is not of the type ``verify`` reads is a refusal of the document (exit 2), never an
uncaught TypeError. The expected exit codes come from docs/cli.md "verify".
"""

from __future__ import annotations

import json

import pytest

from waterwhite import vocabulary as v
from waterwhite.errors import Refused
from waterwhite.rebuild import verify_sides

from rb_kit import copy_shard, hand_report, run, sides_document, write_json, write_lines

CLEAN = ["passage X", "z one", "z two"]
NOT_HELD = 5
REFUSED = 2


def error_of(err: str) -> dict:
    assert err.strip(), "nothing was written to standard error"
    return json.loads(err)


def r1_files(world, tmp_path, report):
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN)
    write_lines(tmp_path / "reference/R1.jsonl", CLEAN)
    sides = sides_document({"R1": str(tmp_path / "repaired/R1.jsonl")},
                           {"R1": str(tmp_path / "reference/R1.jsonl")})
    return write_json(tmp_path / "report.json", report), write_json(tmp_path / "sides.json", sides)


# --- Finding 1: one entry for one artifact -------------------------------------------------


def test_a_report_that_names_one_artifact_twice_is_refused(repair_world):
    """Sol's reproduction: the report rewrites shard-3 with a mixture as its successor, then
    names shard-3 again in a kept entry that points at a real mixture. Each entry passes the
    per-instance kind check; the later one overwrote the earlier in the tables the gate is
    built from, so the shard comparison was skipped and ``--require-held`` exited 0."""
    folder = repair_world.folder
    copy_shard(folder, "shards/shard-3", "sides/both/shard-3")
    report = hand_report(repair_world, [("shard-3", "rewrite", "successor of mix-2"),
                                        ("mix-2", "keep", None)])
    report["artifacts"][1]["artifact"] = "shard-3"
    both = {"shard-3": str(folder / "sides/both/shard-3")}
    sides = write_json(folder / "sides.json", sides_document(both, both))
    code, out, err = run(repair_world, "verify", "--report", str(write_json(folder / "report.json", report)),
                         "--sides", str(sides), "--mode", "seeded", "--require-held")
    error = error_of(err)
    assert (code, out, error["error"]) == (REFUSED, "", "request refused"), err
    assert "shard-3" in error["detail"] and "twice" in error["detail"], error


def test_a_name_that_is_not_the_name_of_the_instance_the_entry_states_is_refused(repair_world, tmp_path):
    """The simpler variant: a kept entry whose name is R2 and whose instance is R1's (two
    different JSONL artifacts) was accepted; the file the repaired side names for R2 was held
    against R1's digest."""
    report = hand_report(repair_world, [("R1", "keep", None)])
    report["artifacts"][0]["artifact"] = "R2"
    report_path, sides = r1_files(repair_world, tmp_path, report)
    code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(sides),
                         "--require-held")
    error = error_of(err)
    assert (code, out, error["error"]) == (REFUSED, "", "request refused"), err
    assert "R2" in error["detail"] and error["context"]["artifact"] == "R2", error


def test_a_rewritten_entry_whose_name_now_resolves_to_another_instance_still_reads_history_changed(
        repair_world, tmp_path):
    """What the refusal above leaves alone: for a rewritten artifact the report's own check
    is its ``history`` entry (docs/cli.md, the conditions), which says the instance changed
    and exits 5. It is a pin of the behaviour that stays, not a red."""
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    report["artifacts"][0]["instance"] = repair_world["R2"]
    report_path, sides = r1_files(repair_world, tmp_path, report)
    code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD, err
    assert "history R1: instance changed" in json.loads(err)["unmet"]


# --- Finding 3: the types of the fields verify reads ----------------------------------------

BAD_REPORTS = {
    "members is a number": lambda r: r["request"].__setitem__("members", 7),
    "members is text": lambda r: r["request"].__setitem__("members", "A/r1"),
    "a member is a number": lambda r: r["request"].__setitem__("members", [7]),
    "records is a number": lambda r: r["request"].__setitem__("records", 7),
    "a record is a number": lambda r: r["request"].__setitem__("records", [7]),
    "artifact is a list": lambda r: r["artifacts"][0].__setitem__("artifact", ["R1"]),
    "artifact is a number": lambda r: r["artifacts"][0].__setitem__("artifact", 7),
    "kind is a list": lambda r: r["artifacts"][0].__setitem__("kind", ["jsonl"]),
    "instance is a list": lambda r: r["artifacts"][0].__setitem__("instance", ["x"]),
    "successor is a number": lambda r: r["artifacts"][0].__setitem__("successor", 7),
    "successor is a list": lambda r: r["artifacts"][0].__setitem__("successor", ["successor of R1"]),
    "groups is a list": lambda r: r.__setitem__("groups", []),
    "placements is a number": lambda r: r.__setitem__("groups", {"placements": 3}),
    "landing is a number": lambda r: r.__setitem__("groups", {"placements": {"p": {"landing": 7}}}),
}


@pytest.mark.parametrize("label", list(BAD_REPORTS))
def test_a_report_field_of_the_wrong_type_is_a_refusal_not_a_crash(repair_world, tmp_path, label):
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    BAD_REPORTS[label](report)
    report_path, sides = r1_files(repair_world, tmp_path, report)
    code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(sides),
                         "--require-held")
    assert (code, out) == (REFUSED, ""), (label, err[-300:])
    assert error_of(err)["error"] == "request refused", (label, err)


# --- The sides file's paths, where verify reads them ----------------------------------------


def test_verify_refuses_a_sides_document_whose_path_is_not_text_when_it_is_handed_parsed(repair_world, tmp_path):
    """The outside review of 504bf64: the check of the sides file's paths lived in the file
    loader, so the Python ``verify`` handed a parsed document reached ``Path()`` with a
    number. The check is where ``verify`` reads the sides."""
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    for which in ("repaired", "reference"):
        for bad in (7, None, ["a"], {"path": "a"}, True):
            sides = sides_document({"R1": "a.jsonl"}, {"R1": "b.jsonl"})
            sides[which]["R1"] = bad
            with pytest.raises(Refused, match=f"names R1 on its {which} side") as caught:
                verify_sides.verify(repair_world.store, report, sides=sides, base=tmp_path)
            assert (caught.value.context["side"], caught.value.context["artifact"]) == (which, "R1")


# --- The correction of docs/cli.md: a wrong-kind successor has matches_sides null -----------


def test_a_successor_of_another_kind_is_not_read_as_the_file_the_repaired_side_names(repair_world, tmp_path):
    """docs/cli.md: a successor of another kind has no file of the kind to compare, so its
    ``matches_sides`` is null. A JSONL file rewritten with a shard as its successor, and a
    sides file that names a JSONL path: the path was read as a shard prefix and the command
    exited 2 with no verification; it exits 5 and prints the document."""
    report = hand_report(repair_world, [("R1", "rewrite", "successor of shard-3")])
    report_path, sides = r1_files(repair_world, tmp_path, report)
    code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(sides),
                         "--require-held")
    assert code == NOT_HELD, err
    made = json.loads(out)
    assert [(e["kind"], e["matches_sides"]) for e in made["successors"]] == [(v.KIND_SHARD, None)]
    assert json.loads(err)["unmet"] == [
        "successor successor of shard-3: kind shard, but the artifact it replaces is of kind jsonl"]
