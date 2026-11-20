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
    different JSONL artifacts) was accepted: both sides name R2 by a copy of R1's file, which
    is the digest the entry's instance is registered with, so the kept entry read "identical"
    and ``--require-held`` exited 0."""
    report = hand_report(repair_world, [("R1", "keep", None)])
    report["artifacts"][0]["artifact"] = "R2"
    for side in ("repaired", "reference"):
        (tmp_path / side).mkdir()
        (tmp_path / side / "R2.jsonl").write_bytes((repair_world.folder / "R1.jsonl").read_bytes())
    sides = sides_document({"R2": str(tmp_path / "repaired/R2.jsonl")},
                           {"R2": str(tmp_path / "reference/R2.jsonl")})
    code, out, err = run(repair_world, "verify", "--report", str(write_json(tmp_path / "report.json", report)),
                         "--sides", str(write_json(tmp_path / "sides.json", sides)), "--require-held")
    error = error_of(err)
    assert (code, out, error["error"]) == (REFUSED, "", "request refused"), err
    assert "R2" in error["detail"] and error["context"]["artifact"] == "R2", error


def test_an_instance_id_is_not_a_second_name_for_an_artifact_the_report_already_names(repair_world, tmp_path):
    """The audit of a078390: the store resolves an instance id as it resolves a name, so a
    second entry that names the mixture by its instance id passed the repeated-name check and
    the name check. The caches condition is built from the entries by instance, the later one
    winning, so the keep entry hid the stale caches of the rewritten mixture: an exit 5 became 0.
    An artifact that has a name is named by it, and an instance has one entry."""
    mixture = repair_world["mix-2"]
    report = hand_report(repair_world, [("mix-2", "rewrite", "successor of mix-2"), ("mix-2", "keep", None)])
    report["artifacts"][1]["artifact"] = mixture
    code, out, err = run(repair_world, "verify", "--report", str(write_json(tmp_path / "report.json", report)),
                         "--require-held")
    error = error_of(err)
    assert (code, out, error["error"]) == (REFUSED, "", "request refused"), err
    assert mixture in error["detail"] or "mix-2" in error["detail"], error


def test_two_entries_for_one_instance_are_refused_whatever_they_are_named(repair_world, tmp_path):
    report = hand_report(repair_world, [("R1", "keep", None), ("R2", "keep", None)])
    report["artifacts"][1]["instance"] = report["artifacts"][0]["instance"]
    code, out, err = run(repair_world, "verify", "--report", str(write_json(tmp_path / "report.json", report)))
    assert (code, out, error_of(err)["error"]) == (REFUSED, "", "request refused"), err


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
    "action is a list": lambda r: r["artifacts"][0].__setitem__("action", ["rewrite"]),
    "action is a number": lambda r: r["artifacts"][0].__setitem__("action", 7),
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
    error = error_of(err)
    assert error["error"] == "request refused", (label, err)
    # the refusal says what is wrong with the value, where the check is of its type (the entry
    # fields the review's reports gave: the same refusal came before for some of them from
    # other checks, and a test that only asks for exit 2 would not notice the type check go)
    if label.split(" is ")[0] in ("artifact", "instance", "kind", "action", "successor"):
        assert "text" in error["detail"], (label, error)


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


# --- The rewrite exemption: a mixture row named for a shard ------------------------------------


def test_a_rewrite_row_named_for_a_shard_whose_instance_and_kind_are_a_mixture_never_holds(repair_world):
    """A rewrite row named shard-3 whose instance and stated kind are a mixture the store really
    holds, with a mixture successor: the exemption lets the name resolve elsewhere, so only the
    history entry stands in the way. It must."""
    folder = repair_world.folder
    copy_shard(folder, "shards/shard-3", "sides/both/shard-3")
    report = hand_report(repair_world, [("mix-2", "rewrite", "successor of mix-2")])
    report["artifacts"][0]["artifact"] = "shard-3"
    both = {"shard-3": str(folder / "sides/both/shard-3")}
    sides = write_json(folder / "sides.json", sides_document(both, both))
    code, out, err = run(repair_world, "verify", "--report", str(write_json(folder / "report.json", report)),
                         "--sides", str(sides), "--mode", "seeded", "--require-held")
    assert code == NOT_HELD, (code, err)
    assert "history shard-3: instance changed" in json.loads(err)["unmet"]


# --- Text the report cannot carry into a path or a query ---------------------------------------

BAD_TEXT = {
    "a NUL in a successor": lambda r: r["artifacts"][0].__setitem__("successor", "x\u0000y"),
    "a NUL in a rewritten name": lambda r: r["artifacts"][0].__setitem__("artifact", "R1\u0000"),
    "a lone surrogate in a successor": lambda r: r["artifacts"][0].__setitem__("successor", "\ud800"),
    "a lone surrogate in an instance": lambda r: r["artifacts"][0].__setitem__("instance", "\ud800"),
    "a lone surrogate in the source": lambda r: r["request"].__setitem__("source", "\ud800"),
    "a NUL in a member": lambda r: r["request"].__setitem__("members", ["A/r1\u0000"]),
}


@pytest.mark.parametrize("label", list(BAD_TEXT))
def test_text_that_cannot_be_a_name_or_a_path_is_a_refusal_not_a_crash(repair_world, tmp_path, label):
    """The audit of a078390: a NUL ends a path lookup in ValueError and a lone surrogate cannot be
    bound to a query; the command ended in a traceback where the contract says exit 2."""
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    BAD_TEXT[label](report)
    report_path, sides = r1_files(repair_world, tmp_path, report)
    path = report_path.with_name("report-bad.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    code, out, err = run(repair_world, "verify", "--report", str(path), "--sides", str(sides), "--require-held")
    assert (code, out) == (REFUSED, ""), (label, err[-300:])
    assert error_of(err)["error"] == "request refused", (label, err)


def test_a_sides_path_or_name_with_a_nul_is_a_refusal_not_a_crash(repair_world, tmp_path):
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    report_path, sides = r1_files(repair_world, tmp_path, report)
    for where in ("repaired", "reference"):
        held = json.loads(sides.read_text())
        held[where]["R1"] = held[where]["R1"] + "\u0000"
        bad = write_json(tmp_path / f"sides-nul-{where}.json", held)
        code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(bad))
        assert (code, out, error_of(err)["error"]) == (REFUSED, "", "request refused"), (where, err)
    held = json.loads(sides.read_text())
    held["reference"]["R1\ud800"] = "x"
    bad = write_json(tmp_path / "sides-surrogate.json", held)
    code, out, err = run(repair_world, "verify", "--report", str(report_path), "--sides", str(bad))
    assert (code, out, error_of(err)["error"]) == (REFUSED, "", "request refused"), err
