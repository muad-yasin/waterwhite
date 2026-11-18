"""The answers for the successors and for the history (plan 4.4; docs/interfaces.md
section 7.3 "successors" and "history", docs/rebuild.md section 5), through
``check_successors`` on the hand-built world with the successors registered.

Under "withdraw A" a successor built with capture on from records of B and Z passes
the proof and "does not contain"; the historical release still "contains". The tool
builds none of these answers: each is the answer of the registered artifact.
"""

from __future__ import annotations

from waterwhite import requests
from waterwhite import vocabulary as v

from rb_kit import build, copy_shard, hand_report, need_attr, sides_document, write_lines

REWRITES = [("R1", "rewrite", "successor of R1"), ("shard-3", "rewrite", "successor of shard-3"),
            ("mix-2", "rewrite", "successor of mix-2")]


def check(world, report=None, sides=None, base=None):
    function = need_attr("waterwhite.rebuild.verify_successors", "check_successors")
    report = report if report is not None else hand_report(world, REWRITES)
    request = requests.withdraw(world.store, "A")
    successors, history = function(world.store, report, request, sides, base)
    return {e["artifact"]: e for e in successors}, {e["artifact"]: e for e in history}


def test_a_successor_built_from_clean_records_does_not_contain_the_source(repair_world):
    successors, _ = check(repair_world)
    assert set(successors) == {"R1", "shard-3", "mix-2"}
    for name, kind in (("R1", "jsonl"), ("shard-3", "shard"), ("mix-2", "mixture")):
        entry = successors[name]
        assert entry["successor"] == f"successor of {name}" and entry["kind"] == kind
        assert entry["registered"] is True
        assert entry["instance"] == repair_world[f"successor of {name}"]
        assert entry["answer"] == "does not contain" and entry["names"] is None, name
        assert entry["holds_unresolved"] is False and entry["matches_sides"] is None
        assert entry["answered"]["answer"] == "does not contain"
        assert entry["answered"]["artifact"] == f"successor of {name}"


def test_the_historical_artifact_still_contains_the_source_under_its_own_instance(repair_world):
    _, history = check(repair_world)
    assert set(history) == {"R1", "shard-3", "mix-2"}
    for name, entry in history.items():
        assert entry["answer"] == "contains", name
        assert entry["instance"] == repair_world[name]
        assert entry["instance_unchanged"] is True, name


def test_a_historical_instance_that_is_not_the_one_the_report_holds_is_flagged(repair_world):
    report = hand_report(repair_world, REWRITES)
    report["artifacts"][0]["instance"] = "0" * 64
    _, history = check(repair_world, report)
    assert history["R1"]["instance_unchanged"] is False
    assert history["shard-3"]["instance_unchanged"] is True


def test_a_kept_artifact_has_no_successor_and_no_history_entry(repair_world):
    report = hand_report(repair_world, REWRITES + [("R2", "keep", None), ("mix-1", "keep", None)])
    successors, history = check(repair_world, report)
    assert set(successors) == set(history) == {"R1", "shard-3", "mix-2"}


def test_a_successor_that_is_not_registered_is_said_so_and_the_history_still_answers(tmp_path):
    world = build(tmp_path / "w")
    try:
        successors, history = check(world)
        for name in ("R1", "shard-3", "mix-2"):
            assert successors[name]["registered"] is False and successors[name]["answer"] is None
            assert history[name]["answer"] == "contains"
    finally:
        world.store.close()


# --- A successor that holds a record that cannot be resolved ----------------------------------------


def test_a_successor_that_holds_an_unresolved_record_may_contain_the_source_naming_the_failing_link(repair_world):
    # The row copied from Z's first record has lost its edge: its origin is unknown.
    repair_world.delete_edge("successor of R1", 2)
    successors, _ = check(repair_world)
    entry = successors["R1"]
    assert entry["answer"] == "may contain" and entry["holds_unresolved"] is True
    # The answer names the link of the proof that failed, as every "may contain" does,
    # and no cause of the rebuild: the named cause "unresolved record" stays unused.
    assert entry["names"] == v.NAMES_MISSING_ORIGIN
    assert entry["names"] != v.NAMES_UNRESOLVED_RECORD
    assert entry["answered"]["failing"]["condition"] == "a"
    assert entry["answered"]["failing"]["artifact"] == "successor of R1"


# --- A missing map, and a successor rebuilt without capture ------------------------------------------


def test_a_successor_shard_with_no_map_is_not_covered_and_what_is_downstream_may_contain(tmp_path):
    world = build(tmp_path / "w", successors=True, successor_map="none")
    try:
        successors, _ = check(world)
        assert successors["shard-3"]["answer"] == "not covered"
        assert successors["mix-2"]["answer"] == "may contain"
    finally:
        world.store.close()


def test_a_successor_rebuilt_without_capture_with_an_identical_candidate_may_contain_the_source(tmp_path):
    # No row of the table asks this (plan 4.4): a document of the successor shard
    # cannot be told from a candidate of A and a candidate of B.
    world = build(tmp_path / "w", successors=True, successor_map="twins")
    try:
        successors, _ = check(world)
        assert successors["shard-3"]["answer"] == "may contain"
        assert successors["R1"]["answer"] == "does not contain"
    finally:
        world.store.close()


# --- The file that was compared is the file that was registered ---------------------------------------


def test_the_answer_is_about_the_file_the_repaired_side_names_only_when_the_digests_agree(repair_world):
    folder = repair_world.folder
    (folder / "sides/repaired").mkdir(parents=True)
    (folder / "sides/repaired/R1.jsonl").write_bytes((folder / "successors/R1.jsonl").read_bytes())
    copy_shard(folder, "successors/shard-3", "sides/repaired/shard-3", flip=True)
    sides = sides_document({"R1": "sides/repaired/R1.jsonl", "shard-3": "sides/repaired/shard-3"}, {})
    successors, _ = check(repair_world, hand_report(repair_world, REWRITES[:2]), sides, folder)
    assert successors["R1"]["matches_sides"] is True
    assert successors["shard-3"]["matches_sides"] is False


def test_a_file_that_differs_in_one_line_does_not_match_the_registered_successor(repair_world):
    write_lines(repair_world.folder / "sides/repaired/R1.jsonl", ["passage X", "z one"])
    sides = sides_document({"R1": "sides/repaired/R1.jsonl"}, {})
    successors, _ = check(repair_world, hand_report(repair_world, REWRITES[:1]), sides, repair_world.folder)
    assert successors["R1"]["matches_sides"] is False
