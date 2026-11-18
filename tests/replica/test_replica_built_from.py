"""The file-level manifest the build records while it runs (plan 5.5 Check 10; table
README, section "Check 10": the subject ``manifest`` and its condition "written while
the build runs").

What a build with capture on writes, and what a build with capture off does not:

- A build with capture on writes ``built-from.jsonl`` at the root of its build folder
  (not the per-line ``<file>.manifest.jsonl`` that sits beside each record file). It
  holds one JSON row per artifact, ``{"name", "files", "built_from"}``: the artifact's
  name as the build registers it (``src/A``, ``gen/P``, ``R1``, ``import/T``,
  ``E/shard-1.input``, ``E/shard-1``, ``E/mix-1``, ...), its file paths relative to the
  build folder, and the names of the artifacts whose files went into it. Nothing below
  a file: no record, line, ordinal, step, label, count or digest, so no other key.
- A build with capture off writes no such file, at the build and after the recovery of
  its store.

Where each expected value comes from. Names, source files and members: the fixture
spec (``[sources]``, ``[generation]`` and its handed lists, ``[releases]``,
``[imports]``, ``[layouts]``, ``[mixtures]``) and ``docs/replica.md`` sections 2 and 5
(the folder a build writes, the registered names, the stand-in for Megatron is the one
of ``conftest.py``). Nothing is taken from running the recorder.

What the table README says ``built_from`` is: "the names of the artifacts whose files
went into it". Two readings follow from it and are pinned here. A release names the files
of the copies it kept, and no file of a copy its dedup dropped (the dropped copy's file did
not go into the release). A round-robin chunk names the files of the records dealt to that
chunk, so a chunk that holds no record of the file of T does not reach T, P or A (table
rows ``c05.unmutated.E-rr/*``); a record dealt to a chunk and dropped by the sanitizer is
not one the chunk holds, its file did not go into the chunk (as with the copy a dedup
dropped), so the chunk's row does not list that file (docs/replica.md section 16). A set's
chunk rows name the set's own files, and no set shard reaches a file through the release its
lines were cut from.

What the text leaves open, and these tests do not pin: the order of the rows; whether
the sanitizer's input file of a shard has a row of its own (a row named
``<shard>.sanitizer-input`` is allowed, no other row beyond the list below is); the
direct ``built_from`` of a chunk file (only the source files and generation outputs
reached through it are pinned); the order of a release's or a generation output's
``built_from`` list. Rows for
the checkpoints of a run are made after the build, by the run wrapper (docs/replica.md
sections 13 and 16): they are appended to the build folder's file, and are asked at the
end of this file. The repaired side of a repair (section 15) is a
build with capture on and writes the file; its reference side is a build with capture
off and does not: both are asked at the end of this file, with the repaired side's
manifest joined to the historical one as the table's history rows read it.

Nothing of the recorder's module is imported by name: a build is asked for through the
build's own entry points and the file is read as JSON, so a missing recorder is a
failed assertion on the file, not an import error.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib
import json
import os
import shutil
import tomllib
from pathlib import Path

import pytest

from replica.fixture import build_captured_fixture, build_table_store

REPO = Path(__file__).resolve().parents[2]
MANIFEST_FILE = "built-from.jsonl"
ROW_KEYS = {"name", "files", "built_from"}

# The in-process harness: these tests ask what a build records, not what a bound gives
# (a table store runs its generation jobs confined by default).
IN_PROCESS = "in-process"

LAYOUT_OF = {"E": "partitioned", "E-rr": "round-robin"}
SHARDS = ("shard-1", "shard-2", "shard-3")

SOURCES = ("A", "B", "C", "Z")


# --- What the spec says ------------------------------------------------------------------


# [generation] and its handed lists (spec: P is handed A-001, T is handed P, S is
# handed A-002 and C-001); the file of a record id is the file of its source (the
# ids A-.., B-.., C-.., Z-.. belong to [sources.A], ...).
GENERATED_FROM = {
    "gen/P": {"src/A"},
    "gen/T": {"gen/P"},
    "gen/S": {"src/A", "src/C"},
}

# [releases.*] inputs: R1 and R1-swap take A-001, B-001, P, T, S and Z-001..Z-020; R2
# takes B-001..B-013 and Z-021..Z-040. The exact dedup keeps the copy of X from the source
# named by dedup_winner ("A" for R1 and R2, "B" for R1-swap) and drops the other copy. A
# release's row names the files of the copies kept (RELEASE_KEPT) and not the file of the
# copy dedup dropped; RELEASED_FROM is every file its inputs are read from.
RELEASED_FROM = {
    "R1": {"src/A", "src/B", "gen/P", "gen/T", "gen/S", "src/Z"},
    "R1-swap": {"src/A", "src/B", "gen/P", "gen/T", "gen/S", "src/Z"},
    "R2": {"src/B", "src/Z"},
}
RELEASE_KEPT = {
    "R1": {"src/A", "gen/P", "gen/T", "gen/S", "src/Z"},
    "R1-swap": {"src/B", "gen/P", "gen/T", "gen/S", "src/Z"},
    "R2": {"src/B", "src/Z"},
}

# [imports."import/T"]: file_family = "A/r1"; [sources.A] gives member "A/r1" for
# the file sources/A.jsonl, whose artifact is src/A.
IMPORT_NAME = "import/T"
IMPORT_FROM = ["src/A"]

# [layouts.partitioned.shards]: shard-1 and shard-2 hold B and Z records, shard-3
# holds T (a generation output) and Z records. The round-robin layout deals the same
# records over three chunks after one shuffle.
PARTITIONED_SOURCE_FILES = {
    "shard-1": {"src/B", "src/Z"},
    "shard-2": {"src/B", "src/Z"},
    "shard-3": {"gen/T", "src/Z"},
}
EVERY_LISTED = {"src/B", "src/Z", "gen/T"}

# [mixtures]: the list of a mixture is the trainer's order; a bare shard name lies in
# the partitioned layout (registered as E/shard-N), a prefixed name is its own.
MIXTURES = {
    "E/mix-1": ["shard-1", "shard-2"],
    "E/mix-2": ["shard-1", "shard-2", "shard-3"],
    "D/mix-1": ["shard-1", "shard-2"],
    "Y/mix-1": ["shard-1", "shard-2", "shard-3"],
    "K/mix-1": ["shard-3", "shard-2"],
    "E-rr/mix-1": ["E-rr/shard-1", "E-rr/shard-2"],
    "E-rr/mix-2": ["E-rr/shard-1", "E-rr/shard-2", "E-rr/shard-3"],
}


def shard_name(listed: str) -> str:
    """The name a shard is registered under: the spec's bare name belongs to run E."""
    return listed if "/" in listed else f"E/{listed}"


def mixture_folder(mixture: str) -> str:
    return f"mixtures/{mixture}"


def test_the_expected_mixtures_are_the_spec_ones(spec_doc):
    """The table of this file is the spec's [mixtures], not a second opinion."""
    assert {name: list(shards) for name, shards in spec_doc["mixtures"].items()} == MIXTURES


def test_the_expected_sources_and_the_import_are_the_spec_ones(spec_doc):
    assert tuple(spec_doc["sources"]) == SOURCES
    assert [spec_doc["sources"][k]["file"].rsplit("/", 1)[1] for k in SOURCES] == [
        f"{k}.jsonl" for k in SOURCES]
    assert spec_doc["imports"][IMPORT_NAME]["file_family"] == spec_doc["sources"]["A"]["member"]
    assert set(spec_doc["releases"]) == set(RELEASED_FROM)
    assert {name: spec_doc["releases"][name]["dedup_winner"] for name in RELEASED_FROM} == {
        "R1": "A", "R1-swap": "B", "R2": "A"}


# --- Reading the file --------------------------------------------------------------------


def read_manifest(out: Path) -> list[dict]:
    path = out / MANIFEST_FILE
    assert path.is_file(), f"the build with capture on wrote no {MANIFEST_FILE} in its folder"
    text = path.read_text(encoding="utf-8")
    assert text == "" or text.endswith("\n"), "every row ends its line"
    return [json.loads(line) for line in text.split("\n") if line != ""]


def by_name(rows: list[dict]) -> dict[str, dict]:
    named = {}
    for row in rows:
        assert row["name"] not in named, f"two rows for {row['name']}"
        named[row["name"]] = row
    return named


def reach(rows: dict[str, dict], name: str) -> set[str]:
    """Every name reached from ``name`` through built_from lists, the name itself
    left out."""
    seen: set[str] = set()
    todo = list(rows[name]["built_from"])
    while todo:
        current = todo.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in rows:
            todo.extend(rows[current]["built_from"])
    return seen


def kinds_reached(rows: dict[str, dict], name: str) -> set[str]:
    """The source files and generation outputs reached from ``name``."""
    return {n for n in reach(rows, name) if n.startswith(("src/", "gen/"))}


def rows_of(built) -> dict[str, dict]:
    """The manifest of a build, by name (read in the body of a test, so that a missing file
    is a failed assertion of that test and not an error at setup)."""
    return by_name(read_manifest(built.out))


@pytest.fixture(scope="module")
def tokenized(tmp_path_factory, megatron_stand_in):
    """The hand-checked fixture built with capture on, both layouts, every release and
    the import, tokenized by the stand-in (one toy token per byte)."""
    base = tmp_path_factory.mktemp("built-from-tokenized")
    with megatron_stand_in(base / "env") as env:
        built = build_captured_fixture(base / "main", tokenize=True, megatron_env=env,
                                       generation_harness=IN_PROCESS)
    return built


@pytest.fixture(scope="module")
def untokenized(tmp_path_factory):
    return build_captured_fixture(tmp_path_factory.mktemp("built-from-plain") / "main")


# --- Capture on: the rows ----------------------------------------------------------------


def test_the_build_with_capture_on_writes_one_row_per_artifact_with_the_three_keys(tokenized):
    tokenized_rows = rows_of(tokenized)
    rows = read_manifest(tokenized.out)
    assert rows, "the manifest holds no row"
    assert all(set(row) == ROW_KEYS for row in rows), [sorted(r) for r in rows]
    assert len(rows) == len(tokenized_rows), "a name has two rows"
    for row in rows:
        assert isinstance(row["name"], str) and row["name"]
        assert isinstance(row["files"], list) and row["files"]
        assert all(isinstance(f, str) for f in row["files"])
        assert isinstance(row["built_from"], list)
        assert all(isinstance(n, str) for n in row["built_from"])


def test_the_rows_are_exactly_the_artifacts_the_build_makes(tokenized):
    tokenized_rows = rows_of(tokenized)
    expected = {f"src/{k}" for k in SOURCES} | set(GENERATED_FROM) | set(RELEASED_FROM)
    expected |= {IMPORT_NAME}
    for prefix in LAYOUT_OF:
        for shard in SHARDS:
            expected |= {f"{prefix}/{shard}.input", f"{prefix}/{shard}"}
    expected |= set(MIXTURES)
    names = set(tokenized_rows)
    assert expected <= names, sorted(expected - names)
    # Beyond the list, only the sanitizer's input file of a shard may have a row.
    extra = names - expected
    assert all(n.endswith(".sanitizer-input") for n in extra), sorted(extra)


def test_every_file_is_relative_to_the_build_folder_and_is_there(tokenized):
    tokenized_rows = rows_of(tokenized)
    for row in tokenized_rows.values():
        for relative in row["files"]:
            assert not Path(relative).is_absolute(), row["name"]
            assert ".." not in Path(relative).parts, row["name"]
            assert "\\" not in relative, row["name"]
            assert (tokenized.out / relative).is_file(), (row["name"], relative)


def test_a_manifest_file_is_never_a_file_of_an_artifact(tokenized):
    """The per-line manifests beside the record files are test evidence of the build, and
    the file this build writes lists no one of them."""
    tokenized_rows = rows_of(tokenized)
    for row in tokenized_rows.values():
        assert not any(f.endswith(".manifest.jsonl") or f == MANIFEST_FILE
                       for f in row["files"]), row["name"]


def test_a_source_file_has_its_file_and_no_artifact_behind_it(tokenized):
    tokenized_rows = rows_of(tokenized)
    for key in SOURCES:
        row = tokenized_rows[f"src/{key}"]
        assert row["files"] == [f"sources/{key}.jsonl"]
        assert row["built_from"] == []


def test_a_generation_output_names_the_files_its_job_was_handed(tokenized):
    tokenized_rows = rows_of(tokenized)
    for name, handed in GENERATED_FROM.items():
        row = tokenized_rows[name]
        assert f"gen/{name.split('/')[1]}.jsonl" in row["files"], row["files"]
        assert set(row["built_from"]) == handed, name
        assert len(row["built_from"]) == len(handed), "no name twice"


def test_a_release_names_exactly_the_files_of_the_copies_it_kept(tokenized):
    """R1 keeps A's copy of X and so does not name B; R1-swap keeps B's and does not name A;
    the file of a copy that dedup dropped did not go into the release."""
    tokenized_rows = rows_of(tokenized)
    for name, inputs in RELEASED_FROM.items():
        row = tokenized_rows[name]
        assert f"releases/{name}.jsonl" in row["files"], row["files"]
        named = set(row["built_from"])
        assert named == RELEASE_KEPT[name], (name, sorted(named))
        assert named <= inputs
        assert len(row["built_from"]) == len(named), "no name twice"


def test_the_import_names_the_file_of_the_member_its_file_family_gives(tokenized):
    tokenized_rows = rows_of(tokenized)
    row = tokenized_rows[IMPORT_NAME]
    assert row["files"] == ["imports/T.jsonl"]
    # Not the generation output of T that the import's text was made from: the
    # spec's file family is the source A, and the manifest has the family only.
    assert row["built_from"] == IMPORT_FROM


@pytest.mark.parametrize("prefix", sorted(LAYOUT_OF))
def test_a_chunk_file_and_a_shard_have_their_files(tokenized, prefix):
    tokenized_rows = rows_of(tokenized)
    layout = LAYOUT_OF[prefix]
    for shard in SHARDS:
        chunk = tokenized_rows[f"{prefix}/{shard}.input"]
        assert chunk["files"] == [f"pipeline/{layout}/tokenizer-input/{shard}.jsonl"]
        made = tokenized_rows[f"{prefix}/{shard}"]
        assert sorted(made["files"]) == [f"pipeline/{layout}/shards/{shard}.bin",
                                         f"pipeline/{layout}/shards/{shard}.idx"]
        # The tokenizer is handed the chunk file: the shard names it.
        assert made["built_from"] == [f"{prefix}/{shard}.input"]


def test_a_partitioned_shard_reaches_the_files_of_its_own_records(tokenized):
    tokenized_rows = rows_of(tokenized)
    for shard, files in PARTITIONED_SOURCE_FILES.items():
        direct = files
        closure = set(direct)
        if "gen/T" in direct:
            closure |= {"gen/P", "src/A"}  # T was made from P, which was made from A
        for name in (f"E/{shard}.input", f"E/{shard}"):
            assert kinds_reached(tokenized_rows, name) == closure, name


def table_answers(number: int, prefix: str) -> dict[str, str]:
    """The answers of the table rows ``c<number>.unmutated.<prefix>*``, by artifact."""
    with open(REPO / "fixture" / "table" / f"check{number:02d}.toml", "rb") as f:
        rows = tomllib.load(f)["row"]
    return {r["given"]["artifact"]: r["expect"]["answer"] for r in rows
            if r["id"].startswith(f"c{number:02d}.unmutated.{prefix}")}


ROUND_ROBIN_ANSWERS_EXPECTED = {
    "E-rr/shard-1": "does not contain", "E-rr/shard-2": "does not contain",
    "E-rr/shard-3": "contains", "E-rr/mix-1": "does not contain", "E-rr/mix-2": "contains"}


def round_robin_mismatches(rows: dict[str, dict], answers: dict[str, str]) -> list[str]:
    """The names, of the round-robin chunks, their shards and their mixtures, whose walk is
    not what the table says of them (``answers``: the rows ``c05.unmutated.E-rr/*``).

    A chunk that holds a record of T reaches T and through it P and A; a chunk that holds
    none reaches none of the three; every chunk reaches only files its own records could
    come from (the files the spec lists for the layout), and some file."""
    closure = EVERY_LISTED | {"gen/P", "src/A"}
    bad = []
    for shard in SHARDS:
        for name in (f"E-rr/{shard}.input", f"E-rr/{shard}"):
            reached = kinds_reached(rows, name)
            holds_a = answers[f"E-rr/{shard}"] == "contains"
            if not reached or not reached <= closure:
                bad.append(name)
            elif ("src/A" in reached) != holds_a:
                bad.append(name)
            elif holds_a and not {"gen/T", "gen/P"} <= reached:
                bad.append(name)
            elif not holds_a and reached & {"gen/T", "gen/P"}:
                bad.append(name)
    for name in ("E-rr/mix-1", "E-rr/mix-2"):
        if ("src/A" in kinds_reached(rows, name)) != (answers[name] == "contains"):
            bad.append(name)
    return bad


def test_a_round_robin_chunk_reaches_the_files_of_the_records_dealt_to_it(tokenized):
    """The same records are dealt over three chunks after one shuffle. Chunk 3 holds T, so it
    reaches T, P and A; chunks 1 and 2 hold only records of B and Z and reach none of the
    three (spec condition 16, ``c05.unmutated.E-rr/*``). A chunk that listed every file of
    the layout would give chunks 1 and 2 the file of T."""
    answers = table_answers(5, "E-rr/")
    assert answers == ROUND_ROBIN_ANSWERS_EXPECTED
    rows = rows_of(tokenized)
    assert round_robin_mismatches(rows, answers) == []
    # The three chunks together reach every file of the layout.
    union = set().union(*(kinds_reached(rows, f"E-rr/{shard}") for shard in SHARDS))
    assert union == EVERY_LISTED | {"gen/P", "src/A"}


def test_a_round_robin_chunk_that_listed_every_file_is_flagged_by_the_check_above(tokenized):
    """A known-bad document: every chunk file lists every file of the layout."""
    answers = table_answers(5, "E-rr/")
    rows = copy.deepcopy(rows_of(tokenized))
    for shard in SHARDS:
        rows[f"E-rr/{shard}.input"]["built_from"] = sorted(EVERY_LISTED)
    flagged = round_robin_mismatches(rows, answers)
    assert {"E-rr/shard-1", "E-rr/shard-2", "E-rr/mix-1"} <= set(flagged)


FILE_OF_ID_LETTER = {"A": "src/A", "B": "src/B", "C": "src/C", "Z": "src/Z",
                     "P": "gen/P", "T": "gen/T", "S": "gen/S"}


def file_of_id(record_id: str) -> str:
    """The artifact of the file a record id belongs to: the ids A-.., B-.., C-.., Z-.. are the
    records of the sources of that letter and P, T and S are the generation outputs (spec
    [sources] and [generation])."""
    return FILE_OF_ID_LETTER[record_id.split("-")[0]]


@pytest.fixture(scope="module")
def translation_dropped(tmp_path_factory, megatron_stand_in):
    """The fixture built with capture on, round-robin layout only, from a copy of the spec and
    of its source files in a folder of this test, where the text of the generation output T is
    only blank. The sanitizer drops a line whose text is empty after sanitizing, so the one
    record of the file of T that the shuffle deals to a chunk is dropped there. Nothing under
    fixture/ is written."""
    base = tmp_path_factory.mktemp("built-from-dropped")
    root = base / "copy"
    shutil.copytree(REPO / "fixture" / "spec", root / "fixture" / "spec")
    shutil.copytree(REPO / "fixture" / "sources", root / "fixture" / "sources")
    path = root / "fixture" / "sources" / "generation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    (row,) = [r for r in rows if r["id"] == "T"]
    row["text"] = "   "
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    with megatron_stand_in(base / "env") as env:
        return build_captured_fixture(
            base / "main", tokenize=True, megatron_env=env, generation_harness=IN_PROCESS,
            spec=root / "fixture" / "spec" / "fixture_spec.toml", layouts=("round-robin",),
            imports=())


def dropped_chunk(built) -> str:
    """The round-robin chunk the sanitizer's log says it dropped the record T from."""
    log = built.out / "pipeline" / "round-robin" / "sanitizer.log.jsonl"
    skips = [json.loads(line) for line in log.read_text(encoding="utf-8").split("\n") if line]
    (dropped,) = [s for s in skips if s["id"] == "T"]
    assert dropped["origin_file"] == "gen/T.jsonl", dropped
    return dropped["shard"]


def reached_for_dropped_record(rows: dict[str, dict], chunk: str) -> set[str]:
    """What the walk of a chunk's file reaches of the file of the record T, which the sanitizer
    dropped from that chunk: the file of T itself and the files T was made from. Empty when
    the chunk's row does not count the dropped record's file as one that went into it."""
    return kinds_reached(rows, f"E-rr/{chunk}.input") & {"gen/T", "gen/P", "src/A"}


def test_a_record_dealt_to_a_round_robin_chunk_and_dropped_by_the_sanitizer_is_not_listed(
        translation_dropped):
    """The record T is dealt to one chunk and the sanitizer drops it there, so its file did not
    go into that chunk: the chunk's row names the files of the records it holds (B and Z),
    and neither it, its shard nor the mixture that holds it reaches the file of T, nor P or A,
    which only T leads to. The generation output T keeps its own row."""
    chunk = dropped_chunk(translation_dropped)
    built = translation_dropped
    kept = [json.loads(line)["id"] for line in (
        built.out / "pipeline" / "round-robin" / "tokenizer-input" / f"{chunk}.jsonl"
    ).read_text(encoding="utf-8").split("\n") if line]
    assert "T" not in kept
    held_files = {file_of_id(i) for i in kept}
    assert held_files == {"src/B", "src/Z"}, "the chunk holds no record but those of B and Z"
    rows = rows_of(built)
    assert "gen/T" in rows
    assert "gen/T" not in rows[f"E-rr/{chunk}.input"]["built_from"]
    assert kinds_reached(rows, f"E-rr/{chunk}.input") == held_files
    assert kinds_reached(rows, f"E-rr/{chunk}") == held_files
    assert reached_for_dropped_record(rows, chunk) == set()
    # A mixture that holds the chunk reaches no more than the chunks it holds do.
    mixture = [m for m, listed in MIXTURES.items()
               if m.startswith("E-rr/") and f"E-rr/{chunk}" in listed]
    for name in mixture:
        assert not kinds_reached(rows, name) & {"gen/T", "gen/P", "src/A"}, name


def test_a_chunk_row_that_listed_the_dropped_records_file_is_flagged_by_the_check_above(
        translation_dropped):
    """A known-bad document: the chunk's row lists the file of the record the sanitizer dropped
    from it."""
    chunk = dropped_chunk(translation_dropped)
    rows = copy.deepcopy(rows_of(translation_dropped))
    assert reached_for_dropped_record(rows, chunk) == set()
    rows[f"E-rr/{chunk}.input"]["built_from"].append("gen/T")
    assert reached_for_dropped_record(rows, chunk) == {"gen/T", "gen/P", "src/A"}


def test_a_mixture_names_its_shards_in_the_trainers_order_and_has_their_copies(tokenized):
    tokenized_rows = rows_of(tokenized)
    for name, listed in MIXTURES.items():
        row = tokenized_rows[name]
        shards = [shard_name(s) for s in listed]
        assert row["built_from"] == shards, name
        expected_files = sorted(
            f"{mixture_folder(name)}/{s.rsplit('/', 1)[1]}{suffix}"
            for s in shards for suffix in (".bin", ".idx"))
        assert sorted(row["files"]) == expected_files, name


def test_a_mixture_reaches_the_files_of_its_shards_down_to_the_sources(tokenized):
    """The walk the table's rows read: E/mix-1 holds shards 1 and 2 only (B and Z, no A);
    the mixtures that hold shard 3 reach T and through it P and A."""
    tokenized_rows = rows_of(tokenized)
    reached = {m: kinds_reached(tokenized_rows, m) for m in ("E/mix-1", "D/mix-1", "E/mix-2",
                                                              "Y/mix-1", "K/mix-1")}
    assert reached["E/mix-1"] == {"src/B", "src/Z"}
    assert reached["D/mix-1"] == {"src/B", "src/Z"}
    for name in ("E/mix-2", "Y/mix-1", "K/mix-1"):
        assert reached[name] == {"src/B", "src/Z", "gen/T", "gen/P", "src/A"}, name


def test_every_name_in_a_built_from_list_has_a_row_of_its_own(tokenized):
    tokenized_rows = rows_of(tokenized)
    named = {n for row in tokenized_rows.values() for n in row["built_from"]}
    assert named <= set(tokenized_rows), sorted(named - set(tokenized_rows))


def test_no_artifact_is_built_from_itself_and_the_lists_hold_no_cycle(tokenized):
    tokenized_rows = rows_of(tokenized)
    for name in tokenized_rows:
        assert name not in reach(tokenized_rows, name), name


# --- Capture on, no tokenization ---------------------------------------------------------


def test_a_build_without_tokenization_writes_rows_for_what_it_made_and_none_for_shards(
        untokenized):
    rows = by_name(read_manifest(untokenized.out))
    expected = {f"src/{k}" for k in SOURCES} | set(GENERATED_FROM) | set(RELEASED_FROM)
    expected |= {IMPORT_NAME}
    assert expected <= set(rows), sorted(expected - set(rows))
    # No shard and no mixture without tokenization (docs/replica.md section 5).
    assert not [n for n in rows if "mix-" in n]
    assert not [n for n in rows if n.rsplit("/", 1)[-1].startswith("shard-")
                and "." not in n.rsplit("/", 1)[-1]]
    assert set(rows["gen/S"]["built_from"]) == {"src/A", "src/C"}


def test_two_seeded_builds_write_the_same_manifest_bytes(untokenized, tmp_path):
    """The repaired side is compared byte for byte between runs, so the rows depend on
    nothing of the folder or the clock."""
    again = build_captured_fixture(tmp_path / "again")
    assert (untokenized.out / MANIFEST_FILE).is_file(), "the build with capture on wrote no manifest"
    assert (again.out / MANIFEST_FILE).is_file(), "the second build wrote no manifest"
    first = (untokenized.out / MANIFEST_FILE).read_bytes()
    assert first
    assert (again.out / MANIFEST_FILE).read_bytes() == first
    text = first.decode("utf-8")
    assert str(untokenized.out.parent) not in text and str(tmp_path) not in text


def test_the_import_store_of_the_table_has_the_import_row_and_the_row_of_its_family_file(
        tmp_path):
    """The store the rows c06.import.* are asked of (README section 4: the sources plus the
    text-only import of T; its generation step is not captured). The import's row names the
    file of the member its file family gives, and that file has a row for a walk to reach."""
    built = build_table_store(tmp_path / "import", "import")
    rows = rows_of(built)
    assert rows[IMPORT_NAME]["built_from"] == IMPORT_FROM
    assert rows[IMPORT_NAME]["files"] == ["imports/T.jsonl"]
    assert rows["src/A"]["files"] == ["sources/A.jsonl"]
    assert {f"src/{k}" for k in SOURCES} <= set(rows)


# --- Capture off: nothing -----------------------------------------------------------------


def files_named(folder: Path, name: str) -> list[Path]:
    return sorted(p for p in folder.rglob(name))


def test_a_build_with_capture_off_writes_no_manifest(fixture_build):
    out, _ = fixture_build
    assert out.is_dir() and (out / "sources" / "A.jsonl").is_file()
    assert files_named(out, MANIFEST_FILE) == []


def test_the_run_made_without_capture_and_its_recovery_write_no_manifest(tmp_path,
                                                                         no_megatron):
    built = build_table_store(tmp_path / "no-capture", "no-capture", tokenize=True,
                              megatron_env=no_megatron)
    assert (built.out / "pipeline" / "partitioned" / "shards" / "shard-1.bin").is_file()
    assert files_named(built.out, MANIFEST_FILE) == []
    # Nor beside the store the recovery made.
    assert files_named(built.out.parent, MANIFEST_FILE) == []


def test_the_build_with_capture_on_has_the_manifest_the_one_with_capture_off_lacks(
        tokenized, fixture_build):
    """The same spec, the same folders: only capture decides."""
    assert (tokenized.out / MANIFEST_FILE).is_file()
    assert not (fixture_build[0] / MANIFEST_FILE).exists()


# --- The recorder takes nothing from the tool or its store --------------------------------


def test_the_recorder_module_imports_nothing_of_the_tool_nor_its_store(repo):
    """A build with capture off loads nothing of the tool but its normaliser
    (test_replica_build_fixture.py), and the manifest is not taken from the store: the
    recorder module is a plain one."""
    path = repo / "replica" / "built_from.py"
    assert path.is_file(), "the recorder module replica/built_from.py does not exist"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            imported.add(base)
            imported |= {f"{base}.{a.name}" for a in node.names}
    roots = {name.lstrip(".").split(".")[0] for name in imported}
    assert not roots & {"waterwhite", "sqlite3"}, sorted(roots & {"waterwhite", "sqlite3"})
    assert not [n for n in imported if n.endswith("capture_hooks")], sorted(imported)


# --- The repair: the repaired side is a build with capture on, the reference side is not ---


@pytest.fixture(scope="module")
def repair_sides(tmp_path_factory):
    """One repair of the fixture's R1 and R2 (the historical build is captured, without
    tokenization and with no layout), with the reference side of withdraw A: the folders
    of the repaired side and of the reference side."""
    from waterwhite import vocabulary as v

    from replica import cli
    from replica.fixture import default_spec_path

    base = tmp_path_factory.mktemp("built-from-repair")
    historical = build_captured_fixture(base / "history", releases=("R1", "R2"), imports=(),
                                        layouts=())
    report = base / "report.json"
    report.write_text(json.dumps({
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
        "records": [],
        "artifacts": [
            {"artifact": "R1", "action": "rewrite", "successor": "successor of R1"},
            {"artifact": "R2", "action": "keep", "successor": None}],
    }), encoding="utf-8")
    repaired, reference = base / "repaired", base / "reference"
    try:
        code = cli.main(["repair", "--report", str(report), "--historical", str(historical.out),
                         "--out", str(repaired), "--store", str(historical.store),
                         "--spec", str(default_spec_path()),
                         "--reference-withdrawn", "A", "--reference-out", str(reference)])
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    assert code in (0, None), "the repair did not run"
    return historical, repaired, reference


def test_the_repaired_side_writes_the_manifest_and_names_its_release_as_the_table_does(
        repair_sides):
    """Table c09.fixture.successor.R1 asks for the artifact ``successor of R1``: the name the
    repaired side registers it under. What the row reaches under withdraw A is asked by the
    history test below, from both folders."""
    _, repaired, _ = repair_sides
    rows = rows_of_folder(repaired)
    assert "successor of R1" in rows, sorted(rows)
    assert all(set(row) == ROW_KEYS for row in rows.values())
    assert "releases/R1.jsonl" in rows["successor of R1"]["files"]
    for row in rows.values():
        for relative in row["files"]:
            assert (repaired / relative).is_file(), (row["name"], relative)


def test_the_reference_side_is_built_without_capture_and_has_no_manifest(repair_sides):
    historical, repaired, reference = repair_sides
    assert (reference / "releases" / "R1.jsonl").is_file(), "the reference side was not built"
    assert files_named(reference, MANIFEST_FILE) == []
    # The historical build had capture on and holds its own.
    assert (historical.out / MANIFEST_FILE).is_file()


def rows_of_folder(folder: Path) -> dict[str, dict]:
    return by_name(read_manifest(folder))


def table_answer(number: int, row_id: str) -> str:
    with open(REPO / "fixture" / "table" / f"check{number:02d}.toml", "rb") as f:
        (found,) = [r for r in tomllib.load(f)["row"] if r["id"] == row_id]
    return found["expect"]["answer"]


def history_readings(historical: Path, repaired: Path) -> tuple[str, str]:
    """What ``c09.history`` (R1) and ``c09.successors`` (the successor of R1) read, under
    withdraw A, from the manifest of the historical build joined with the manifest the
    repaired build wrote, both read from their folders by the reader of the baselines."""
    manifest = importlib.import_module("baselines.manifest")
    joined = manifest.join_manifests(manifest.load_manifest(historical),
                                     manifest.load_manifest(repaired))
    assert joined is not None, "a folder of the two holds no manifest"
    targets = {"src/A"}
    return (manifest.read_group("c09.history", joined, artifact="R1", targets=targets),
            manifest.read_group("c09.successors", joined, artifact="successor of R1",
                                targets=targets))


def test_the_history_of_r1_is_read_from_both_folders_joined_by_artifact_name(repair_sides):
    """The repaired side wrote a manifest of its own, under the successor names; the
    historical build kept its own. Joined, the historical R1 still contains A (the history
    keeps its attribution) and the successor of R1 does not."""
    historical, repaired, _ = repair_sides
    history, successor = history_readings(historical.out, repaired)
    assert history == table_answer(9, "c09.fixture.R1-history") == "contains"
    assert successor == table_answer(9, "c09.fixture.successor.R1") == "does not contain"


def test_the_history_of_r1_is_not_read_from_the_repaired_folder_alone(repair_sides):
    """A known-bad input: the repaired side's manifest on its own has no row for the
    historical R1, so it cannot give the answer the joined manifests give."""
    _, repaired, _ = repair_sides
    manifest = importlib.import_module("baselines.manifest")
    alone = manifest.read_group("c09.history", manifest.load_manifest(repaired), artifact="R1",
                                targets={"src/A"})
    assert alone == "no record" != table_answer(9, "c09.fixture.R1-history")


# --- The checkpoints of a run: rows added to the build folder's file ---------------------

# [runs.*.segments] and [runs.*.step_record]: a checkpoint's row names the shards its
# segment was handed, in the trainer's order (a bare shard name lies in run E), then the
# segment's parent checkpoint when it has one; its file is its segment's run record,
# <run>/<segment with the slash made a dash>.run.json, in the runs folder.
CHECKPOINT_ROWS = {
    "E@6": ("E/E-seg-1.run.json", ["E/shard-1", "E/shard-2"]),
    "E@9": ("E/E-seg-2.run.json", ["E/shard-1", "E/shard-2", "E/shard-3", "E@6"]),
    "E-rr@6": ("E-rr/E-rr-seg-1.run.json", ["E-rr/shard-1", "E-rr/shard-2"]),
    "D@3": ("D/D-seg-1.run.json", ["E/shard-1", "E/shard-2", "E@9"]),
    "Y@6": ("Y/Y-seg-1.run.json", ["E/shard-1", "E/shard-2", "E/shard-3", "E@6"]),
    "K@3": ("K/K-seg-1.run.json", ["E/shard-3", "E/shard-2", "E@6"]),
    "K@8": ("K/K-seg-1.run.json", ["E/shard-3", "E/shard-2", "E@6"]),
    "K@40": ("K/K-seg-1.run.json", ["E/shard-3", "E/shard-2", "E@6"]),
}


def recorder(name: str):
    """A name of the recorder module, looked up when a test needs it, so that a missing
    name is a failed assertion of that test."""
    module = importlib.import_module("replica.built_from")
    found = getattr(module, name, None)
    assert found is not None, f"replica/built_from.py has no {name}"
    return found


def spec_plans(spec_path):
    from replica.plan import load_fixture_spec
    from replica.run_plan import load_runs_plan

    return load_runs_plan(spec_path).segments, load_fixture_spec(spec_path).layouts


@pytest.fixture
def folders(tmp_path, tokenized):
    """A copy of the manifest of the tokenized build in a build folder of its own, and a
    new runs folder beside it (the shared build is never written)."""
    build = tmp_path / "main" / "build"
    build.mkdir(parents=True)
    shutil.copyfile(tokenized.out / MANIFEST_FILE, build / MANIFEST_FILE)
    runs = tmp_path / "main" / "runs"
    runs.mkdir()
    return build, runs


def test_the_checkpoint_rows_join_the_build_folders_file_with_paths_relative_to_it(
        folders, spec_path, spec_doc):
    derived = {}
    for run_doc in spec_doc["runs"].values():
        for segment, segment_doc in run_doc.get("segments", {}).items():
            handed = [shard_name(s) for s in segment_doc["shards"]]
            if "parent_checkpoint" in segment_doc:
                handed.append(segment_doc["parent_checkpoint"])
            for checkpoint in run_doc["step_record"][segment]["checkpoints"]:
                derived[checkpoint["name"]] = handed
    assert derived == {name: built_from for name, (_, built_from) in CHECKPOINT_ROWS.items()}, (
        "the table of this file is the spec's, not a second opinion")

    build, runs = folders
    before = read_manifest(build)
    segments, layouts = spec_plans(spec_path)
    recorder("record_checkpoints")(build, runs, segments, layouts)
    rows = read_manifest(build)
    assert rows[:len(before)] == before, "the rows the build wrote stay as they were"
    added = by_name(rows[len(before):])
    assert set(added) == set(CHECKPOINT_ROWS)
    for name, (record, built_from) in CHECKPOINT_ROWS.items():
        row = added[name]
        assert set(row) == ROW_KEYS
        assert row["built_from"] == built_from, name
        assert len(row["files"]) == 1, name
        assert not os.path.isabs(row["files"][0]), name
        assert (build / row["files"][0]).resolve() == (runs / record).resolve(), name
    assert not (runs / MANIFEST_FILE).exists(), "no file of this name is written in the runs folder"


def test_the_same_runs_recorded_twice_add_no_row_and_a_row_that_differs_is_refused(
        folders, spec_path):
    from replica.errors import ReplicaError

    build, runs = folders
    segments, layouts = spec_plans(spec_path)
    record = recorder("record_checkpoints")
    record(build, runs, segments, layouts)
    once = (build / MANIFEST_FILE).read_bytes()
    record(build, runs, segments, layouts)
    assert (build / MANIFEST_FILE).read_bytes() == once, "a second call added or changed a row"
    changed = [dataclasses.replace(s, shard_files=s.shard_files[:1]) if s.name == "E/seg-1" else s
               for s in segments]
    with pytest.raises(ReplicaError):
        record(build, runs, changed, layouts)
    assert (build / MANIFEST_FILE).read_bytes() == once, "a refused call left a row behind"


def test_a_build_folder_with_no_manifest_gets_no_checkpoint_row_and_no_file(tmp_path, spec_path):
    build, runs = tmp_path / "build", tmp_path / "runs"
    build.mkdir()
    runs.mkdir()
    segments, layouts = spec_plans(spec_path)
    recorder("record_checkpoints")(build, runs, segments, layouts)
    assert not (build / MANIFEST_FILE).exists()
    assert not (runs / MANIFEST_FILE).exists()


@pytest.mark.megatron
@pytest.mark.slow
def test_build_runs_on_a_capture_on_build_adds_its_checkpoint_rows_to_the_build_folder(
        tmp_path, spec_path):
    """Through the run wrapper itself, on a build made with capture on and Megatron's own
    tokenization: run D is one segment, whose checkpoint D@3 is named by its record."""
    from replica.megatron_tokenizer import ENV_VARIABLE
    from replica.runs import build_runs

    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    built = build_table_store(tmp_path / "main", "main", tokenize=True, parallel=3,
                              generation_harness=IN_PROCESS)
    runs = tmp_path / "runs-main"
    build_runs(runs, built.out, spec_path=spec_path, runs=["D"])
    rows = rows_of(built)
    record, built_from = CHECKPOINT_ROWS["D@3"]
    assert "D@3" in rows, "the build folder's manifest holds no row of the checkpoint D@3"
    assert set(rows["D@3"]) == ROW_KEYS
    assert rows["D@3"]["built_from"] == built_from
    assert [(built.out / f).resolve() for f in rows["D@3"]["files"]] == [(runs / record).resolve()]
    assert (runs / record).is_file()
    assert not (runs / MANIFEST_FILE).exists()


# --- The checkpoint rows on the path the table's stores are made by ----------------------

# The main store holds runs E, D, Y and K, the rr store the run E-rr (table README, section 4).
STORE_RUNS = {
    "main": (["E", "D", "Y", "K"], ["E@6", "E@9", "D@3", "Y@6", "K@3", "K@8", "K@40"]),
    "rr": (["E-rr"], ["E-rr@6"]),
}


def checkpoint_rows_wrong(rows: dict[str, dict], wanted: list[str]) -> list[str]:
    """The checkpoints of ``wanted`` that have no row, or a row that is not the one the spec
    gives (CHECKPOINT_ROWS: the shards the segment was handed, then its parent checkpoint)."""
    wrong = []
    for name in wanted:
        row = rows.get(name)
        if row is None or set(row) != ROW_KEYS or len(row["files"]) != 1:
            wrong.append(name)
        elif row["built_from"] != CHECKPOINT_ROWS[name][1]:
            wrong.append(name)
    return wrong


def test_a_manifest_with_no_checkpoint_row_is_flagged_for_every_checkpoint_asked(tokenized):
    """A known-bad document: the manifest of a capture-on build as it is before any run is
    made from it holds no checkpoint, so every one asked is wrong."""
    rows = rows_of(tokenized)
    wanted = [n for _, names in STORE_RUNS.values() for n in names]
    assert checkpoint_rows_wrong(rows, wanted) == wanted
    rows = copy.deepcopy(rows)
    rows["E@6"] = {"name": "E@6", "files": ["x"], "built_from": ["E/shard-2"]}
    assert checkpoint_rows_wrong(rows, ["E@6"]) == ["E@6"]


@pytest.mark.megatron
@pytest.mark.slow
@pytest.mark.parametrize("store", sorted(STORE_RUNS))
def test_the_runs_of_a_table_store_leave_a_row_for_every_checkpoint_of_the_store(
        tmp_path, spec_path, store):
    """The path the table's stores are made by: the store built with capture on, then the
    replica's ``build-runs`` for the runs of the store handed the build folder. E@6, E@9,
    D@3, Y@6, K@3, K@8 and K@40 (main) and E-rr@6 (rr) each read a row of the build folder's
    manifest, with the shards the segment was handed and its parent checkpoint."""
    from replica.megatron_tokenizer import ENV_VARIABLE
    from replica.runs import build_runs

    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    runs_made, wanted = STORE_RUNS[store]
    built = build_table_store(tmp_path / store, store, tokenize=True, parallel=3,
                              generation_harness=IN_PROCESS)
    before = rows_of(built)
    assert not [n for n in wanted if n in before], "the build holds a checkpoint row before any run"
    runs = tmp_path / f"runs-{store}"
    build_runs(runs, built.out, spec_path=spec_path, runs=runs_made)
    rows = rows_of(built)
    assert checkpoint_rows_wrong(rows, wanted) == []
    for name in wanted:
        record = CHECKPOINT_ROWS[name][0]
        assert (built.out / rows[name]["files"][0]).resolve() == (runs / record).resolve(), name
    assert not (runs / MANIFEST_FILE).exists()


# --- A generated set: the chunk rows name the set's own files ----------------------------

SET_LAYOUTS = ("partitioned", "round-robin")


@pytest.fixture(scope="module")
def captured_set(small_set, tmp_path_factory, megatron_stand_in):
    """The small generated set (dedup on) built with capture on, both layouts, tokenized by
    the stand-in."""
    from replica.sets import build_captured_set

    base = tmp_path_factory.mktemp("built-from-set")
    with megatron_stand_in(base / "env") as env:
        return build_captured_set(base / "captured", small_set, layouts=SET_LAYOUTS,
                                  megatron_env=env)


def set_walk_mismatches(rows: dict[str, dict], keep: list[str], rewrite: list[str],
                        prefix: str) -> list[str]:
    """The shards of a set layout (store prefix ``prefix``), and their chunk files, whose walk
    to the withdrawn source file ``src/A`` is not what the counting script says of the shard:
    a shard in ``rewrite`` reaches ``src/A``, a shard in ``keep`` does not, and each has a
    row."""
    bad = []
    for shard in keep + rewrite:
        for name in (f"{prefix}/{shard}.input", f"{prefix}/{shard}"):
            if name not in rows:
                bad.append(name)
            elif ("src/A" in reach(rows, name)) != (shard in rewrite):
                bad.append(name)
    return bad


def counted_shards(small_set, plant_count, built, layout: str) -> dict:
    return plant_count.count(small_set, chunks=built.out / "pipeline" / layout / "tokenizer-input",
                             layout=layout)["shards"]


def test_no_shard_of_a_set_walks_to_a_through_the_release_its_lines_were_cut_from(
        small_set, plant_count, captured_set):
    """Which shards a withdrawal of A reaches is the counting script's (it reads the ids of
    the chunk files and the plant list): the shards of A and of the generation outputs made
    from A, and in the round-robin layout every chunk. The shards of C and of Z do not."""
    rows = rows_of(captured_set)
    partitioned = counted_shards(small_set, plant_count, captured_set, "partitioned")
    assert sorted(partitioned["keep"]) == ["shard-C", "shard-Z"]
    assert sorted(partitioned["rewrite"]) == ["shard-A", "shard-gen"]
    assert set_walk_mismatches(rows, partitioned["keep"], partitioned["rewrite"], "set") == []
    dealt = counted_shards(small_set, plant_count, captured_set, "round-robin")
    assert dealt["keep"] == [] and len(dealt["rewrite"]) == 20
    assert set_walk_mismatches(rows, dealt["keep"], dealt["rewrite"], "set-rr") == []


def test_a_chunk_row_of_a_set_names_files_of_the_set_and_never_the_release(captured_set):
    rows = rows_of(captured_set)
    own_files = {n for n in rows if n.startswith(("src/", "gen/"))}
    assert {"src/A", "src/B", "src/C", "src/Z"} <= own_files
    chunks = [n for n in rows if n.endswith(".input") and n.startswith(("set/", "set-rr/"))]
    assert len(chunks) == 4 + 20  # shard-B holds no line and has no row
    for name in chunks:
        named = set(rows[name]["built_from"])
        assert named and named <= own_files, (name, sorted(named))


def test_the_release_of_a_set_names_the_files_of_the_rows_it_kept(small_set, read_rows,
                                                                 captured_set):
    """The plant list says which rows of R1 are present (the rest lost the dedup); the file
    of a generated row is its job's output (``gen/<job>``). All of B's rows lost to A's copy,
    so R1 names no file of B."""
    plant = read_rows(small_set / "plant_list.jsonl")
    kept = set()
    for row in plant:
        if row["releases"].get("R1", {}).get("row") != "present":
            continue
        if row["file"].startswith("sources/"):
            kept.add("src/" + row["file"].split("/")[1].removesuffix(".jsonl"))
        else:
            kept.add(f"gen/{row['job']}")
    assert "src/B" not in kept
    row = rows_of(captured_set)["R1"]
    assert set(row["built_from"]) == kept, sorted(set(row["built_from"]) ^ kept)


def test_a_set_chunk_that_names_the_release_is_flagged_by_the_check_above(
        small_set, plant_count, captured_set):
    """A known-bad document: every chunk file names R1, the file its lines were cut from."""
    rows = copy.deepcopy(rows_of(captured_set))
    for name, row in rows.items():
        if name.endswith(".input") and name.startswith(("set/", "set-rr/")):
            row["built_from"] = ["R1"]
    partitioned = counted_shards(small_set, plant_count, captured_set, "partitioned")
    flagged = set_walk_mismatches(rows, partitioned["keep"], partitioned["rewrite"], "set")
    assert {"set/shard-C", "set/shard-Z"} <= set(flagged)
