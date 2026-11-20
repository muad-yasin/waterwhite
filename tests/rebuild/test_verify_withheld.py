"""Ruled on the audit of a078390: ``verify`` withholds, as the listing and the report
do, only the answers that rest on a shard whose files are away and which no check has passed.
Each is named with its reason in ``withheld``; the rest of the verification is printed; the
command does not exit 0 (3, the fault's own code)."""

from __future__ import annotations

import json
import sqlite3

from waterwhite.store import Store

from rb_kit import copy_shard, hand_report, run, sides_document, write_json, write_lines


def archive_the_successor_shard(world) -> None:
    folder = world.folder
    (folder / "away").mkdir(exist_ok=True)
    for suffix in (".bin", ".idx"):
        (folder / f"successors/shard-3{suffix}").rename(folder / f"away/shard-3{suffix}")
    world.store.close()
    raw = sqlite3.connect(world.path)
    raw.execute("DELETE FROM shard_files_check")  # a store made before schema 5: no shard was checked
    raw.commit()
    raw.close()
    world.store = Store.open(world.path)


def test_verify_withholds_the_successor_that_rests_on_an_archived_unchecked_shard(repair_world, tmp_path):
    archive_the_successor_shard(repair_world)
    report = hand_report(repair_world, [("shard-3", "rewrite", "successor of shard-3")])
    code, out, err = run(repair_world, "verify", "--report", str(write_json(tmp_path / "report.json", report)))
    assert code == 3, err
    made = json.loads(out)
    assert [(e["artifact"], e["error"]) for e in made["withheld"]] == [
        ("successor of shard-3", "shard files not checked")]
    assert repair_world["successor of shard-3"] in made["withheld"][0]["reason"]
    successor = made["successors"][0]
    assert (successor["registered"], successor["answer"], successor["answered"]) == (True, None, None)
    assert made["history"][0]["answer"] == "contains"  # what does not rest on the shard is printed
    assert json.loads(err)["error"] == "shard files not checked"


def test_a_report_that_withholds_an_artifact_is_not_held_by_require_held(repair_world, tmp_path):
    """The path from a withheld answer in the report to the verification: a report whose shard
    entry is "unresolved" because its answer was withheld (it rests on an archived shard no check
    has passed) names no unresolved record, so the verification's own gate could read it as
    equal on both sides and hold. A report that carries withheld answers never holds."""
    folder = repair_world.folder
    report = hand_report(repair_world, [("R1", "keep", None), ("shard-3", "keep", None)])
    report["withheld"] = [{"artifact": "shard-3", "instance": repair_world["shard-3"], "kind": "shard",
                           "error": "shard files not checked", "shard": repair_world["shard-3"],
                           "state": None, "reason": "the files of the shard are away"}]
    for entry, name in zip(report["artifacts"], ("R1", "shard-3")):
        entry["action"] = "keep" if name == "R1" else "unresolved"
    write_lines(tmp_path / "repaired/R1.jsonl", [])
    both = {"R1": str(folder / "R1.jsonl")}
    code, out, err = run(repair_world, "verify", "--report", str(write_json(tmp_path / "report.json", report)),
                         "--sides", str(write_json(tmp_path / "sides.json", sides_document(both, both))),
                         "--require-held")
    assert code == 5, (code, err)
    assert any(line.startswith("report withheld shard-3") for line in json.loads(err)["unmet"]), err
