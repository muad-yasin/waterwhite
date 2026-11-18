"""The files upstream of the artifacts of the rebuild report (plan 4.1 with its ledger
row BF-6; docs/interfaces.md section 7.1), on the hand-built world of ``rb_kit.py``.

Releases and imports, shards and mixtures are in ``artifacts``. The other registered
files of kind jsonl (source files, generation outputs, the intermediate files that sit between a
source and a release or a shard) are in ``upstream_files``, which follows ``artifacts``, when their
answer is not "does not contain". A generation output that is reused and the revision of
the source that the request names have no action (null) and a reason; a rebuilt intermediate
file is "rewrite"; one that answers "may contain" is "unresolved" with
the answer's own named cause. They are left out of ``counts`` and ``bytes_to_rewrite``.

The kit world has no such intermediate file, so this module registers
three (all of role intermediate, none made by a generation step):

- ``sh3.input``: copies of the first and fifth line of R1; answers "contains";
- ``sh2.san``: one line of Z whose edge is then deleted; answers "may contain" with the
  named cause "missing origin" (condition a);
- ``sh1.input``: one line of Z; answers "does not contain".
"""

from __future__ import annotations

import json

import rb_kit
from rb_kit import ww, hand_report, report_parts, run, write_lines
from waterwhite import coverage, requests
from waterwhite import vocabulary as v

WHOLE_REVISION = "the revision the request withdraws"
FIELDS = {"artifact", "instance", "kind", "role", "answer", "action", "reason"}


def narrowed_reason(count: int) -> str:
    return f"a revision the request names: {count} of its records are withdrawn, the rest stay"


def add_upstream_files(world) -> None:
    store, folder = world.store, world.folder
    r1, z = world["R1"], world["src/Z"]
    world.ids["sh3.input"] = ww.record(
        store, folder / "up/shard-3.input.jsonl", ["passage X", "z one"], "sh3.input",
        v.ROLE_INTERMEDIATE, ww.step("tokenizer-input", "3"), [r1],
        [ww.copy_edge(1, ww.ref(r1, 1)), ww.copy_edge(2, ww.ref(r1, 5))])
    world.ids["sh2.san"] = ww.record(
        store, folder / "up/shard-2.san.jsonl", ["z one"], "sh2.san", v.ROLE_INTERMEDIATE,
        ww.step("sanitizer-input", "2"), [z], [ww.copy_edge(1, ww.ref(z, 1))])
    world.delete_edge("sh2.san", 1)
    world.ids["sh1.input"] = ww.record(
        world.store, folder / "up/shard-1.input.jsonl", ["z two"], "sh1.input",
        v.ROLE_INTERMEDIATE, ww.step("tokenizer-input", "1"), [z], [ww.copy_edge(1, ww.ref(z, 2))])


def listed(document: dict) -> list[dict]:
    held = document.get("upstream_files")
    assert isinstance(held, list), "the report has no list under upstream_files"
    return held


def by_name(document: dict) -> dict[str, dict]:
    entries = listed(document)
    names = [entry["artifact"] for entry in entries]
    assert len(names) == len(set(names)), f"a file is listed twice: {names}"
    return {entry["artifact"]: entry for entry in entries}


def expected_names(world, request) -> list[str]:
    """The ruled list, worked out from the store and the answers: every registered
    artifact of kind jsonl that ``artifacts`` does not hold, whose answer is not "does
    not contain", in registration order."""
    session = coverage.Session(world.store, request)
    names = []
    for artifact in world.store.artifacts():
        if artifact.kind != v.KIND_JSONL or artifact.role in (v.ROLE_RELEASE, v.ROLE_IMPORT):
            continue
        said = coverage.answer(world.store, artifact.instance_id, request, session=session)
        if said.answer != v.DOES_NOT_CONTAIN:
            names.append(artifact.names[0])
    return names


# --- The key and its place ---------------------------------------------------------------


def test_the_report_holds_upstream_files_right_after_artifacts(repair_world, tmp_path):
    keys = list(report_parts(repair_world))
    assert "upstream_files" in keys, keys
    assert keys.index("upstream_files") == keys.index("artifacts") + 1
    assert keys == ["form", "request", "records", "artifacts", "upstream_files", "counts",
                    "mixtures", "groups"]


def test_verify_reads_nothing_of_the_list(repair_world, tmp_path):
    from waterwhite.rebuild import verify_sides

    # The report as the tool prints it holds the key; verify reads the keys it needs and
    # no other, so the list changes nothing it says.
    printed = report_parts(repair_world)
    assert "upstream_files" in printed, sorted(printed)
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    write_lines(tmp_path / "repaired/R1.jsonl", ["passage X", "z one", "z two"])
    write_lines(tmp_path / "reference/R1.jsonl", ["passage X", "z one", "z two"])
    sides = rb_kit.sides_document(dict(R1="repaired/R1.jsonl"), dict(R1="reference/R1.jsonl"))
    plain = verify_sides.verify(repair_world.store, report, sides=sides, base=tmp_path)
    with_list = dict(report, upstream_files=[dict(bogus=1), "not even an object"])
    assert verify_sides.verify(repair_world.store, with_list, sides=sides, base=tmp_path) == plain


# --- The list under a withdrawal of the whole revision -------------------------------------


def test_the_list_holds_every_jsonl_file_that_artifacts_lacks_and_does_not_clear(world):
    add_upstream_files(world)
    document = report_parts(world)
    request = requests.withdraw(world.store, "A")
    assert expected_names(world, request) == [
        "src/A", "gen/P", "gen/T", "gen/S", "sh3.input", "sh2.san"]
    assert [entry["artifact"] for entry in listed(document)] == expected_names(world, request)
    held = {entry["artifact"] for entry in document["artifacts"]}
    for entry in listed(document):
        assert entry["artifact"] not in held
        assert entry["answer"] != v.DOES_NOT_CONTAIN
        assert FIELDS <= set(entry), sorted(FIELDS - set(entry))
        assert entry["instance"] == world[entry["artifact"]]
        assert entry["kind"] == v.KIND_JSONL
    # Cleared files and the files of ``artifacts`` are not in it.
    for cleared in ("src/B", "src/C", "src/Z", "sh1.input", "R1", "R2", "shard-3", "mix-2"):
        assert cleared not in by_name(document)


def test_a_generation_output_the_repair_reuses_takes_no_action_and_says_why(world):
    add_upstream_files(world)
    entries = by_name(report_parts(world))
    for name in ("gen/P", "gen/T", "gen/S"):
        entry = entries[name]
        assert (entry["role"], entry["answer"], entry["action"]) == (
            v.ROLE_INTERMEDIATE, v.CONTAINS, None), name
        reason = entry["reason"].lower()
        assert "retired" in reason and "not rebuilt" in reason, entry["reason"]


def test_the_revision_a_whole_revision_request_names_takes_no_action_and_says_why(world):
    add_upstream_files(world)
    entry = by_name(report_parts(world))["src/A"]
    assert (entry["role"], entry["answer"], entry["action"], entry["reason"]) == (
        v.ROLE_SOURCE, v.CONTAINS, None, WHOLE_REVISION)


def test_an_intermediate_file_ahead_of_a_rebuilt_shard_is_rewrite(world):
    add_upstream_files(world)
    entry = by_name(report_parts(world))["sh3.input"]
    assert (entry["role"], entry["answer"], entry["action"]) == (
        v.ROLE_INTERMEDIATE, v.CONTAINS, v.ACTION_REWRITE)
    assert "reason" in entry


def test_a_file_that_may_contain_is_unresolved_with_the_answers_own_named_cause(world):
    add_upstream_files(world)
    request = requests.withdraw(world.store, "A")
    said = coverage.answer(world.store, world["sh2.san"], request,
                           session=coverage.Session(world.store, request)).to_json()
    entry = by_name(report_parts(world))["sh2.san"]
    assert (entry["answer"], entry["action"]) == (v.MAY_CONTAIN, v.UNRESOLVED)
    assert entry["names"] == said["names"] == "missing origin"
    assert entry["failing"] == said["failing"]
    assert entry["failing"]["condition"] == "a"
    assert "reason" in entry


# --- The list under a request narrowed to records ------------------------------------------


def test_the_revision_a_narrowed_request_names_says_how_many_of_its_records_are_withdrawn(world):
    add_upstream_files(world)
    document = report_parts(world, records=["src/A:L1"])
    request = requests.withdraw(world.store, "A", records=["src/A:L1"])
    entries = by_name(document)
    # The summary S rests on A's second line, so it leaves the list.
    assert list(entries) == expected_names(world, request) == [
        "src/A", "gen/P", "gen/T", "sh3.input", "sh2.san"]
    assert (entries["src/A"]["action"], entries["src/A"]["reason"]) == (None, narrowed_reason(1))
    assert entries["gen/P"]["action"] is None and entries["gen/T"]["action"] is None
    assert entries["sh3.input"]["action"] == v.ACTION_REWRITE
    assert entries["sh2.san"]["action"] == v.UNRESOLVED


def test_the_count_of_the_narrowed_reason_is_the_count_of_records_named(world):
    add_upstream_files(world)
    entries = by_name(report_parts(world, records=["src/A:L1", "src/A:L2"]))
    assert entries["src/A"]["reason"] == narrowed_reason(2)
    assert entries["src/A"]["action"] is None
    assert "gen/S" in entries and entries["gen/S"]["action"] is None


# --- What the list leaves as it was -----------------------------------------------------------


def test_counts_bytes_artifacts_and_every_other_key_are_the_same_with_the_list(world, tmp_path):
    plain = rb_kit.build(tmp_path / "plain")
    try:
        before = report_parts(plain)
    finally:
        plain.store.close()
    add_upstream_files(world)
    after = report_parts(world)
    assert "upstream_files" in after
    for key in ("form", "request", "records", "artifacts", "counts", "mixtures", "groups"):
        assert after[key] == before[key], key
    # The files upstream are outside the totals: R1 and shard-3 are the only bytes.
    sizes = rb_kit.sizes_of(world.folder, "R1.jsonl") + rb_kit.sizes_of(world.folder, "shards/shard-3")
    assert after["counts"]["bytes_to_rewrite"] == sizes
    assert after["counts"]["unresolved_artifacts"] == []
    assert after["counts"]["bytes_unknown"] == []
    for entry in after["artifacts"]:
        assert entry["artifact"] not in ("src/A", "gen/P", "sh3.input", "sh2.san")


def test_the_text_form_names_the_files_upstream(world):
    add_upstream_files(world)
    code, out, err = run(world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    for name in ("src/A", "gen/P", "gen/T", "gen/S", "sh3.input", "sh2.san"):
        assert name in out, f"{name} is not named in the text form"
    assert "sh1.input" not in out


def test_the_json_form_of_the_command_carries_the_list_in_the_same_place(world):
    add_upstream_files(world)
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    document = json.loads(out)
    keys = list(document)
    assert "upstream_files" in keys, keys
    assert keys.index("upstream_files") == keys.index("artifacts") + 1
    assert [entry["artifact"] for entry in document["upstream_files"]] == [
        "src/A", "gen/P", "gen/T", "gen/S", "sh3.input", "sh2.san"]
