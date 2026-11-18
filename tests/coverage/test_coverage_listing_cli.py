"""The listing (plan 2.1 "Listings", Decision 26, L10) and the command line
(docs/interfaces.md section 7): JSON on standard output, errors on standard error
with the exit code of their family, and nothing on standard output after an error.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import ww_world

from waterwhite import cli, errors, listing, requests
from waterwhite import vocabulary as v
from waterwhite.records import SnapshotShard
from waterwhite.store.store import Store

SRC = str(Path(__file__).resolve().parents[2] / "src")
GHOST = "f" * 64


def add_ghost_mixture(world) -> None:
    """A mixture whose snapshot names one shard the store was never told of."""
    world.store.register_mixture(
        name="mix-ghost",
        shards=[SnapshotShard(0, world["shard-1"], "shards/shard-1",
                              world.store.artifact("shard-1").content_digest),
                SnapshotShard(1, GHOST, "shards/ghost", "e" * 64)],
        settings={"mix": "ghost"},
    )


def run(world, *argv: str) -> tuple[int, str, str]:
    """Run the command line in this process against the world's store file. The
    world's own connection is closed first, as a real caller's would be."""
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


# --- The listing ------------------------------------------------------------------------


def test_the_listing_states_what_it_covers_and_answers_for_it(world):
    made = listing.listing(world.store, requests.withdraw(world.store, "A")).to_json()
    assert list(made) == ["request", "covers", "answers", "not_covered"]
    assert made["request"] == {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}
    assert made["covers"] == {
        "runs": ["E", "D", "Y"],
        "mixtures": ["mix-1", "mix-2"],
        "shards": ["shard-1", "shard-3"],
        "checkpoints": ["E@3", "E@4", "E@6", "D@2", "Y@2"],
        "releases": ["R1", "R2"],
    }
    assert made["not_covered"] == []
    # One answer for every registered artifact, the source files and the files
    # between source and release among them (DECISIONS.md D-24): by kind, and
    # inside a kind in the order of registration.
    assert [a["artifact"] for a in made["answers"]] == [
        "src/A", "src/B", "src/C", "src/Z", "gen/P", "gen/T", "gen/S", "R1", "R2",
        "shard-1", "shard-3", "mix-1", "mix-2", "E@3", "E@4", "E@6", "D@2", "Y@2",
    ]
    assert len(made["answers"]) == sum(1 for _ in world.store.artifacts())
    answers = {a["artifact"]: a["answer"] for a in made["answers"]}
    assert answers == {
        "src/A": "contains", "src/B": "does not contain", "src/C": "does not contain",
        "src/Z": "does not contain", "gen/P": "contains", "gen/T": "contains", "gen/S": "contains",
        "R1": "contains", "R2": "does not contain", "shard-1": "does not contain",
        "shard-3": "contains", "mix-1": "does not contain", "mix-2": "contains",
        "E@3": "does not contain", "E@4": "contains", "E@6": "contains", "D@2": "contains",
        "Y@2": "does not contain",
    }


def test_the_listing_lists_what_is_named_and_has_no_record_of_its_own(world):
    add_ghost_mixture(world)
    world.store.register_checkpoint(name="G@1", segment="G/seg-1", step=1)
    ww_world.segment(world.store, "H", "H/seg-1", 0, None, "0" * 64, [(1, 1, None)], {1: (0, 0)},
                     start_kind="warm-start")
    world.delete_map("shard-1")
    made = listing.listing(world.store, requests.withdraw(world.store, "A")).to_json()

    # The shard whose map is gone is not among what the listing covers ...
    assert made["covers"]["shards"] == ["shard-3"]
    assert made["covers"]["mixtures"] == ["mix-1", "mix-2", "mix-ghost"]
    assert "G@1" not in made["covers"]["checkpoints"]
    assert made["covers"]["runs"] == ["E", "D", "Y", "H"]
    # ... and is listed, with every other artifact that is only named.
    gaps = {(g["artifact"], g["kind"], g["reason"]) for g in made["not_covered"]}
    assert gaps == {
        ("shard-1", "shard", "record missing"),  # a row and no map
        ("shards/ghost", "shard", "record missing"),  # named by a snapshot, no row
        ("G@1", "checkpoint", "record missing"),  # its run segment has no row
        ("0" * 64, "checkpoint", "record missing"),  # a segment started from it
        (None, "mixture", "not registered"),  # a segment with no snapshot
    }
    by_name = {g["artifact"]: g for g in made["not_covered"]}
    assert by_name["shards/ghost"]["instance"] == GHOST
    assert by_name["shards/ghost"]["named_by"] == [
        {"relation": "mixture shard", "referrer": "mix-ghost", "position": 1}]
    assert {n["referrer"] for n in by_name["shard-1"]["named_by"]} == {"mix-1", "mix-2", "mix-ghost"}
    assert by_name[None]["named_by"] == [{"relation": "segment mixture", "referrer": "H/seg-1"}]

    # Every registered artifact has an answer, and one with no record of its own
    # answers "not covered" (DECISIONS.md D-24). An artifact that is only named
    # has no row, so it has no answer: it stands in the not-covered list alone.
    answers = {a["artifact"]: a for a in made["answers"]}
    assert len(made["answers"]) == sum(1 for _ in world.store.artifacts())
    assert (answers["shard-1"]["answer"], answers["shard-1"]["reason"]) == ("not covered", "record missing")
    assert (answers["G@1"]["answer"], answers["G@1"]["reason"]) == ("not covered", "record missing")
    assert {name for name, a in answers.items() if a["answer"] == "not covered"} == {"shard-1", "G@1"}
    assert "shards/ghost" not in answers and "0" * 64 not in answers
    # Downstream of the missing map is "may contain" (plan 2.1), and a recorded
    # "contains" still wins.
    assert answers["mix-1"]["answer"] == "may contain"
    assert answers["mix-1"]["failing"]["artifact"] == "shard-1"
    assert answers["mix-ghost"]["answer"] == "may contain"
    assert answers["mix-2"]["answer"] == "contains"
    assert v.DOES_NOT_CONTAIN not in {answers[n]["answer"] for n in ("mix-1", "mix-ghost", "E@3")}


def test_the_listing_names_a_registered_mixture_whose_snapshot_is_gone(world):
    world.sql(("DELETE FROM mixture_shard WHERE artifact_id = "
               "(SELECT artifact_id FROM artifact WHERE instance_id = ?)", (world["mix-1"],)))
    made = listing.listing(world.store, requests.withdraw(world.store, "C")).to_json()
    assert made["covers"]["mixtures"] == ["mix-2"]
    (gap,) = made["not_covered"]
    assert (gap["artifact"], gap["kind"], gap["reason"]) == ("mix-1", "mixture", "record missing")
    assert {n["referrer"] for n in gap["named_by"]} == {"E/seg-1", "D/seg-1"}
    answers = {a["artifact"]: a for a in made["answers"]}
    assert (answers["mix-1"]["answer"], answers["mix-1"]["reason"]) == ("not covered", "record missing")
    # The checkpoints that trained on it read shards it no longer names.
    assert answers["E@3"]["answer"] == "may contain"


def test_the_listing_runs_the_loud_checks_over_the_whole_store_first(world):
    # A fault on a file no shard, mixture or checkpoint descends from stops it too.
    (world.folder / "src/Z.jsonl").write_text('{"text": "changed"}\n{"text": "z two"}\n{"text": "z three"}\n')
    request = requests.withdraw(world.store, "A")
    with pytest.raises(errors.IntegrityFault):
        listing.listing(world.store, request)


# --- The command line: answers ----------------------------------------------------------


def test_query_withdraw_prints_one_answer_as_json(world):
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "E@6")
    assert (code, err) == (0, "")
    answer = json.loads(out)
    assert (answer["artifact"], answer["answer"], answer["basis"]) == ("E@6", "contains", v.BASIS_WHOLE_SAMPLE)
    assert answer["request"]["members"] == ["A/r1"]

    code, out, err = run(world, "query", "withdraw", "A", "--record", "src/A:L2", "--member", "A/r1",
                         "--artifact", "gen/S")
    assert (code, err) == (0, "")
    assert json.loads(out)["request"] == {
        "kind": "withdraw", "source": "A", "members": ["A/r1"], "records": ["src/A:L2"]}


def test_query_withdraw_without_an_artifact_prints_the_listing(world):
    code, out, err = run(world, "query", "withdraw", "C")
    assert (code, err) == (0, "")
    made = json.loads(out)
    assert set(made) == {"request", "covers", "answers", "not_covered"}
    assert made["covers"]["releases"] == ["R1", "R2"]


def test_not_covered_and_may_contain_are_answers_with_exit_code_zero(world):
    code, out, _ = run(world, "query", "withdraw", "A", "--artifact", "E@12")
    assert (code, json.loads(out)["answer"], json.loads(out)["reason"]) == (0, "not covered", "not registered")
    world.delete_edge("gen/P", 1)
    code, out, _ = run(world, "query", "withdraw", "A", "--artifact", "gen/P")
    assert (code, json.loads(out)["answer"]) == (0, "may contain")


def test_format_text_is_for_a_person(world):
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "E@6", "--format", "text")
    assert (code, err) == (0, "")
    assert "contains" in out and "first exposure: E/seg-2 step 4" in out
    assert "untraced steps: E/seg-2 step 5" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)
    code, out, _ = run(world, "query", "withdraw", "A", "--format=text")
    assert code == 0 and out.startswith("listing under withdraw A")


def test_the_store_flag_may_stand_before_the_request(world):
    world.store.close()
    out = io.StringIO()
    code = cli.main(["query", "--store", str(world.path), "withdraw", "A", "--artifact", "R2"],
                    stdout=out, stderr=io.StringIO())
    world.store = Store.open(world.path)
    assert (code, json.loads(out.getvalue())["answer"]) == (0, "does not contain")


# --- The command line: errors -----------------------------------------------------------


def expect_error(result: tuple[int, str, str], code: int, name: str) -> dict:
    """An error leaves nothing on standard output, one JSON object on standard
    error and the exit code of its family."""
    got, out, err = result
    assert out == ""
    error = json.loads(err)
    assert (got, error["error"]) == (code, name)
    assert set(error) >= {"error", "detail"}
    return error


def test_each_refusal_exits_with_2_and_prints_no_answer(world):
    expect_error(run(world, "query", "withdraw", "Q"), 2, "source not registered")
    expect_error(run(world, "query", "withdraw", "A", "--member", "B/r1"), 2, "member not registered")
    expect_error(run(world, "query", "withdraw", "A", "--record", "src/B:L1"), 2, "record not in source")
    expect_error(run(world, "query", "withdraw", "A", "--record", "src/A"), 2, "unsupported selector")
    expect_error(run(world, "query", "withdraw", "A", "--selector", "url-prefix=https://x"), 2,
                 "unsupported selector")
    expect_error(run(world, "query", "ban", "passage X"), 2, "extension off")
    expect_error(run(world, "query", "ban", "passage X", "--artifact", "R1"), 2, "extension off")
    expect_error(run(world, "query", "flagged", "flagged.jsonl"), 2, "extension off")


def test_a_store_that_is_not_there_is_refused(tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["query", "withdraw", "A", "--store", str(tmp_path / "none.db")], stdout=out, stderr=err)
    expect_error((code, out.getvalue(), err.getvalue()), 2, "not a store")
    assert not (tmp_path / "none.db").exists()  # a query makes no file
    (tmp_path / "notes.txt").write_text("not a store at all\n")
    for command in (["query", "withdraw", "A"], ["show", "R1"], ["check"]):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main([*command, "--store", str(tmp_path / "notes.txt")], stdout=out, stderr=err)
        expect_error((code, out.getvalue(), err.getvalue()), 2, "not a store")


def test_an_ambiguous_artifact_is_refused(world):
    # R1 registered again with B's copy kept: the same bytes at the same path, two instances.
    a, b = world["src/A"], world["src/B"]
    edges = list(world.store.edges(world["R1"]))
    assert len(edges) == 6
    from waterwhite.records import EdgeInput, EdgeRecord

    swapped = [EdgeRecord("L1", (EdgeInput(ww_world.ref(a, 1)), EdgeInput(ww_world.ref(b, 1), selected=True)),
                          ww_world.step("dedup"), parent_mode=v.MODE_ANY)]
    swapped += [EdgeRecord(e.child.locator, e.inputs, e.transform) for e in edges[1:]]
    count, digest = 6, world.store.artifact("R1").content_digest
    world.store.register_jsonl(
        world.folder / "R1.jsonl", name="R1-swap", role=v.ROLE_RELEASE,
        step=ww_world.step("release", "R1"), member_name=None,
        inputs=list(world.store.artifact("R1").inputs),
        occurrences=ww_world.occurrences(["passage X", "paraphrase of X", "Uebersetzung von P",
                                          "summary of SA and SC", "z one", "z two"]),
        edges=swapped, declared_count=count, declared_digest=digest,
    )
    expect_error(run(world, "query", "withdraw", "A", "--artifact", digest), 2, "ambiguous artifact")
    code, out, _ = run(world, "query", "withdraw", "A", "--artifact", "R1-swap")
    assert (code, json.loads(out)["answer"]) == (0, "contains")  # P, T and S are in it


@pytest.mark.parametrize("argv", [
    ("query", "withdraw", "A", "--bogus"),
    ("query", "withdraw", "A", "--artifact"),
    ("query", "withdraw", "A", "--art", "R1"),  # no abbreviations
    ("query", "withdraw"),
    ("query",),
    ("frobnicate",),
    ("show",),
    ("check", "extra"),
    ("query", "withdraw", "A", "--format", "yaml"),
])
def test_a_command_line_the_tool_cannot_read_exits_with_1(world, argv):
    expect_error(run(world, *argv), 1, "usage")


def test_a_missing_store_flag_and_no_command_are_usage_errors():
    for argv in (["check"], [], ["query", "withdraw", "A"]):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(argv, stdout=out, stderr=err)
        expect_error((code, out.getvalue(), err.getvalue()), 1, "usage")


def test_a_loud_fault_prints_nothing_on_standard_output(world):
    world.sql(("UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
               "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
               (world.binding_id("shard-3"),)))
    for argv in (("query", "withdraw", "A", "--artifact", "shard-3"),
                 ("query", "withdraw", "A", "--artifact", "mix-2"),
                 ("query", "withdraw", "A", "--artifact", "E@6"),
                 ("query", "withdraw", "A"),  # the listing
                 ("check",)):
        error = expect_error(run(world, *argv), 3, "map digest mismatch")
        assert error["detail"]
    # An artifact whose ancestor closure does not hold the fault still answers.
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "shard-1")
    assert (code, err, json.loads(out)["answer"]) == (0, "", "does not contain")
    # In text form the error is a line on standard error, and still no answer.
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "shard-3", "--format", "text")
    assert (code, out) == (3, "") and err.startswith("error: map digest mismatch")


def test_the_three_other_loud_faults_stop_a_query(world):
    world.sql(("UPDATE binding SET declared_count = declared_count - 1 WHERE binding_id = ?",
               (world.binding_id("shard-3"),)))
    expect_error(run(world, "query", "withdraw", "A", "--artifact", "mix-2"), 3, "map count mismatch")
    world.sql(("UPDATE binding SET declared_count = declared_count + 1 WHERE binding_id = ?",
               (world.binding_id("shard-3"),)),
              ("UPDATE step_trace SET consumed_samples = 2 WHERE trace_id = ?",
               (world.trace_id("E/seg-1", 1),)))
    expect_error(run(world, "query", "withdraw", "A", "--artifact", "E@3"), 3, "trace sample count mismatch")
    code, _, _ = run(world, "query", "withdraw", "A", "--artifact", "mix-1")
    assert code == 0  # a trace lies in a checkpoint's closure only
    # A file changed on disk. Which of the two digest faults names it is the store's
    # rule (DECISIONS.md D-18); either way the query stops and prints no answer.
    (world.folder / "R2.jsonl").write_text('{"text": "passage Y"}\n{"text": "b two"}\n{"text": "z three"}\n')
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "shard-1")
    assert (code, out) == (3, "")
    assert json.loads(err)["error"] in ("artifact digest mismatch", "map digest mismatch")


def test_records_removed_with_a_lowered_count_stop_a_query(world):
    """Two edits that hide each other from the store's own rows: three occurrence
    rows of R1 are removed and its binding declares three records fewer. R1's file
    still holds six lines under its registered digest, so every query whose
    closure holds R1 stops with "map count mismatch", and so does ``check``."""
    r1 = world.scalar("SELECT artifact_id FROM artifact WHERE instance_id = ?", (world["R1"],))
    # Lines 3, 5 and 6 stay: shard-3's map names them, and a removed parent of a
    # map row would be the fault "edge parent missing".
    world.sql(("DELETE FROM occurrence WHERE artifact_id = ? AND locator IN ('L1', 'L2', 'L4')", (r1,)),
              ("UPDATE binding SET declared_count = declared_count - 3 WHERE artifact_id = ?", (r1,)))
    assert world.store.binding(world["R1"]).declared_count == world.store.stored_row_count(world["R1"]) == 3
    for argv in (("check",),
                 ("query", "withdraw", "A", "--artifact", "R1"),
                 ("query", "withdraw", "A", "--artifact", "shard-3"),
                 ("query", "withdraw", "A", "--artifact", "mix-2"),
                 ("query", "withdraw", "A", "--artifact", "E@6"),
                 ("query", "withdraw", "A")):  # the listing
        error = expect_error(run(world, *argv), 3, "map count mismatch")
        assert (error["context"]["declared"], error["context"]["on_disk"]) == (3, 6)
        assert error["context"]["instance"] == world["R1"]
        assert error["context"]["path"] == str(world.folder / "R1.jsonl")
        assert "declares 3 records" in error["detail"] and "holds 6" in error["detail"]
    # An artifact whose ancestor closure does not hold R1 still answers.
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", "shard-1")
    assert (code, err, json.loads(out)["answer"]) == (0, "", "does not contain")

    # The file gone from its recorded place is a state and not a fault (DECISIONS.md
    # D-18): the count is then held against the store's rows only, and ``check``
    # names the file it could not read.
    os.rename(world.folder / "R1.jsonl", world.folder / "R1.archived")
    code, out, err = run(world, "check")
    passed = json.loads(out)
    assert (code, err, passed["check"]) == (0, "", "passed")
    assert [m["instance"] for m in passed["missing_files"]] == [world["R1"]]
    os.rename(world.folder / "R1.archived", world.folder / "R1.jsonl")
    expect_error(run(world, "check"), 3, "map count mismatch")


def test_a_line_appended_to_a_file_is_the_artifacts_fault_and_not_a_count_fault(world):
    """The store is untouched and R2's file holds one line more than its binding
    declares. The file no longer has its registered digest, so the fault named is
    "artifact digest mismatch": a count is held against the registered file only."""
    with open(world.folder / "R2.jsonl", "a", encoding="utf-8") as f:
        f.write('{"text": "added later"}\n')
    for argv in (("query", "withdraw", "A", "--artifact", "R2"),
                 ("query", "withdraw", "A", "--artifact", "shard-1"),
                 ("check",)):
        error = expect_error(run(world, *argv), 3, "artifact digest mismatch")
        assert error["context"]["instance"] == world["R2"]


# --- init, show, check ------------------------------------------------------------------


def test_init_creates_a_store_once(tmp_path):
    def init(*more: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["init", "--store", str(tmp_path / "new.db"), *more], stdout=out, stderr=err)
        return code, out.getvalue(), err.getvalue()

    expect_error(init(), 1, "usage")  # the normaliser is named, never assumed
    expect_error(init("--normaliser", "no-such/1"), 4, "unknown normaliser")
    assert not (tmp_path / "new.db").exists()
    expect_error(init("--normaliser", "control-chars/1", "--verified-join", "only-a-tag"), 1, "usage")
    # A join the tool does not ship is refused, as a usage error, and nothing is written.
    expect_error(init("--normaliser", "control-chars/1", "--verified-join",
                      "core_v9.9.9,single,elsewhere.Sampler"), 1, "usage")
    assert list(tmp_path.iterdir()) == []
    code, out, err = init("--normaliser", "control-chars/1", "--verified-join", ",".join(ww_world.JOIN))
    assert (code, err) == (0, "")
    made = json.loads(out)
    assert (made["normaliser"], made["verified_joins"]) == ("control-chars/1", [list(ww_world.JOIN)])
    with Store.open(tmp_path / "new.db", readonly=True) as store:
        assert store.verified_joins() == (ww_world.JOIN,)
    before = (tmp_path / "new.db").read_bytes()
    expect_error(init("--normaliser", "control-chars/1"), 4, "a file already exists at the store path")
    assert (tmp_path / "new.db").read_bytes() == before


def test_check_passes_on_a_whole_store(world):
    code, out, err = run(world, "check")
    assert (code, err) == (0, "")
    assert json.loads(out) == {"check": "passed", "artifacts": len(world.ids), "missing_files": []}


def test_show_says_what_the_store_holds(world):
    code, out, _ = run(world, "show", "mix-2")
    shown = json.loads(out)
    assert (code, shown["kind"], shown["registered"]) == (0, "mixture", True)
    assert [(s["name"], s["has_map"]) for s in shown["shards"]] == [("shard-1", True), ("shard-3", True)]
    assert shown["segments"] == ["E/seg-2", "Y/seg-1"]

    code, out, _ = run(world, "show", "E@6")
    shown = json.loads(out)
    assert (shown["segment"], shown["at_step"], shown["own_record"]) == ("E/seg-2", 6, True)
    assert [p["segment"] for p in shown["history"]["segments"]] == ["E/seg-1", "E/seg-2"]

    code, out, _ = run(world, "show", "R1", "--record", "L1")
    edge = json.loads(out)["edge"]
    assert (edge["parent_mode"], edge["label"], edge["candidate_sources"]) == ("any", "observed", ["A", "B"])
    assert (edge["selected_parent"]["artifact"], edge["selected_parent"]["record"]) == ("src/A", "L1")

    code, out, _ = run(world, "show", "gen/P", "--record", "L1")
    edge = json.loads(out)["edge"]
    assert (edge["job"], edge["model"], edge["bound_field"], edge["bound_from_log"]) == (
        "paraphrase", "none", True, True)
    # A copy edge has no job. A generation edge keeps its job's name when the log
    # is gone (DECISIONS.md D-24).
    assert json.loads(run(world, "show", "R1", "--record", "L2")[1])["edge"]["job"] is None
    world.remove_log("gen/P", 1, "paraphrase")
    edge = json.loads(run(world, "show", "gen/P", "--record", "L1")[1])["edge"]
    assert (edge["job"], edge["harness_log"], edge["model"], edge["bound_field"], edge["bound_from_log"]) == (
        "paraphrase", False, None, True, None)

    code, out, _ = run(world, "show", "shard-3", "--record", "D0")
    document = json.loads(out)
    assert (document["label"], document["parent"]["artifact"], document["parent"]["record"]) == (
        "recovered", "R1", "L3")

    code, out, _ = run(world, "show", "nothing-here")
    assert (code, json.loads(out)) == (0, {
        "artifact": "nothing-here", "registered": False, "reason": "not registered", "named_by": []})
    expect_error(run(world, "show", "R1", "--record", "L4000"), 2, "request refused")


def test_the_module_runs_as_a_program(world):
    world.store.close()
    env = dict(os.environ, PYTHONPATH=SRC)
    done = subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", "query", "withdraw", "A", "--artifact", "R2",
         "--store", str(world.path)],
        capture_output=True, text=True, env=env, timeout=30,
    )
    world.store = Store.open(world.path)
    assert (done.returncode, done.stderr) == (0, "")
    assert json.loads(done.stdout)["answer"] == "does not contain"
    bad = subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", "query", "withdraw", "Q", "--store", str(world.path)],
        capture_output=True, text=True, env=env, timeout=30,
    )
    assert (bad.returncode, bad.stdout) == (2, "")
    assert json.loads(bad.stderr)["error"] == "source not registered"
