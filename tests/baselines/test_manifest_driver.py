"""The manifest driver reads three more groups (17 Nov 2026):
``c09.artifact-actions`` (at fixture size), ``c09.successors`` and ``c09.history``.

The table already holds the manifest subject's own walk rule for them (check10.toml, rows
``c10.c09.artifact-actions.manifest``, ``c10.c09.successors.manifest``, ``c10.c09.history.manifest``): "contains: the
walk reaches the file of a member the request names. does not contain: the artifact has a row and the walk reaches no
such file. no record: the artifact has no row." The action of an artifact is its reading turned by the plan's own
definition (contains is rewrite, does not contain is keep). The successors are read in the manifest of the repaired
side (the subject's own run on that build); the historical R1 is read in the rows of both builds joined.

The listing (``c05.listing.map-deleted``) and the file family of an import (``c06.import.file-family``) are read from the
rule text of their rows; the 10k lists (kept-shards, all-rewrite) are read from the manifest's own shard and chunk rows, successor-shards is the no-fold cannot answer, and the R1 history waits for the repaired side.

The tests drive the real ``ManifestDriver`` over a hand-written manifest (a stand-in context), and grade each
reading with the real grader against the listed row's own expectation.
"""

from __future__ import annotations

import importlib
import tomllib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"


def _mod(name: str):
    return importlib.import_module(name)


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows


@lru_cache(maxsize=1)
def _questions() -> dict:
    table10 = _mod("baselines.table10")
    return {q.id: q for q in table10.expand(table10.load_table10())}


def question(listed: str, size: str | None = None):
    table10 = _mod("baselines.table10")
    return _questions()[table10.question_id(listed, "manifest", size)]


def _row(name, files=(), built_from=()):
    return {"name": name, "files": list(files), "built_from": list(built_from)}


# The manifest of the fixture's store main, as the replica writes it in this respect: sources, generation outputs,
# the chunk files of the shards (each built from its sources) and the shards, the release R1 built from the three
# shards and R2 from the first two.
MAIN = [
    _row("src/A", ["sources/A.jsonl"]),
    _row("src/B", ["sources/B.jsonl"]),
    _row("src/Z", ["sources/Z.jsonl"]),
    _row("gen/P", ["gen/P.jsonl"], ["src/A"]),
    _row("gen/T", ["gen/T.jsonl"], ["gen/P"]),
    _row("E/shard-1.input", [], ["src/B", "src/Z"]),
    _row("E/shard-2.input", [], ["src/B", "src/Z"]),
    _row("E/shard-3.input", [], ["src/Z", "gen/T"]),
    _row("E/shard-1", [], ["E/shard-1.input"]),
    _row("E/shard-2", [], ["E/shard-2.input"]),
    _row("E/shard-3", [], ["E/shard-3.input"]),
    _row("R1", [], ["E/shard-1", "E/shard-2", "E/shard-3"]),
    _row("R2", [], ["E/shard-1", "E/shard-2"]),
]
# The repaired side: the successors Check 9 builds from the tool's rebuild report. Withdrawing A makes the
# successor of src/A; no successor holds a file of A.
REPAIRED = [
    _row("successor of src/A", ["sources/A.jsonl"]),
    _row("successor of E/shard-3", [], ["E/shard-3.successor.input"]),
    _row("E/shard-3.successor.input", [], ["src/Z"]),
    _row("successor of R1", [], ["successor of E/shard-3", "E/shard-1", "E/shard-2"]),
]
REQUESTS = {"withdraw A": {"files": ["sources/A.jsonl"]}}


def driver_of(*, repaired=True, manifests=None, repaired_manifests=None):
    subject = _mod("baselines.subjects")
    facts = []
    context = SimpleNamespace(ran_on=set(), keep_fact=lambda *args, **kwargs: facts.append(args))
    inputs = {"store-main": list(MAIN)} if manifests is None else manifests
    given_repaired = (dict(repaired_manifests) if repaired_manifests is not None
                      else {"store-main": list(REPAIRED)} if repaired else {})
    driver = subject.manifest_factory(inputs, requests=REQUESTS, traces={}, repaired=given_repaired)(context)
    return driver, facts


def graded(q, reading, member_files=None):
    grade = _mod("baselines.grade")
    expect = dict(_listed_rows()[q.listed_row]["expect"])
    return grade.grade(q, expect, reading, member_files=member_files)


@pytest.mark.parametrize("listed, action", [
    ("c09.fixture.R1.action", "rewrite"), ("c09.fixture.R2.action", "keep"),
    ("c09.fixture.E/shard-1.action", "keep"), ("c09.fixture.E/shard-2.action", "keep"),
    ("c09.fixture.E/shard-3.action", "rewrite")])
def test_the_action_of_an_artifact_is_its_walk_read_as_rewrite_where_it_contains_and_keep_where_it_does_not(listed, action):
    q = question(listed)
    reading = driver_of()[0].read(q)
    assert dict(reading.raw) == {"action": action}, listed
    assert graded(q, reading).status == "right", listed


def test_the_fixture_counts_name_the_three_shards_of_the_run_and_list_them_by_action():
    q = question("c09.fixture.shards.counts")
    reading = driver_of()[0].read(q)
    raw = dict(reading.raw)
    assert raw["keep"] == ["E/shard-1", "E/shard-2"] and raw["rewrite"] == ["E/shard-3"]
    assert raw["counts"] == {"keep": 2, "rewrite": 1}
    assert graded(q, reading).status == "right"


@pytest.mark.parametrize("listed", ["c09.fixture.successor.E/shard-3", "c09.fixture.successor.R1"])
def test_a_successor_is_read_in_the_manifest_of_the_repaired_side(listed):
    q = question(listed)
    driver, _ = driver_of()
    reading = driver.read(q)
    assert dict(reading.raw) == {"answer": "does not contain"}, "the successors hold no file of A"
    assert graded(q, reading).status == "right"
    # The historical manifest alone holds no row of a successor: read there, the artifact has no record, which is
    # another word. The driver must read the repaired side's own rows.
    only_main, _ = driver_of(repaired=False)
    with pytest.raises(_mod("baselines.combo_dolma").NotAttempted):
        only_main.read(q)


def test_the_historical_r1_is_read_after_the_rows_of_the_successor_build_are_added_to_the_manifest():
    q = question("c09.fixture.R1-history")
    driver, _ = driver_of()
    reading = driver.read(q)
    assert dict(reading.raw) == {"answer": "contains"}, "R1 still walks to the source A, whose file the request names"
    assert graded(q, reading).status == "right"
    # The targets come from the JOINED rows: from the repaired rows alone the historical walk would not reach them.
    from baselines import manifest as manifest_module
    repaired_targets = {r["name"] for r in REPAIRED if set(r["files"]) & {"sources/A.jsonl"}}
    assert manifest_module.read_artifact(list(MAIN), "R1", repaired_targets) == "does not contain"


def test_the_listing_is_the_manifest_after_the_fault_written_from_the_rule_text_and_graded_by_membership():
    """The rule: "list every name the manifest holds, as a row of its own or inside a built_from list, by kind of
    artifact; a name with no row of its own is listed as no record. The reading is that list after the fault." The
    manifest holds no run, so no run is listed; the reading is not fitted to the expectation."""
    q = question("c05.listing.map-deleted")
    driver, _ = driver_of()
    reading = driver.read(q)
    raw = dict(reading.raw)
    assert [item["artifact"] for item in raw["not_covered"]] == ["E/shard-1"], "the fault deleted the row of E/shard-1"
    assert "E/shard-1" not in {name for names in raw["covers"].values() for name in names}
    assert raw["covers"]["shards"] == ["E/shard-2", "E/shard-3"] and raw["covers"]["releases"] == ["R1", "R2"]
    assert raw["covers"]["runs"] == [], "the manifest holds no run: none is made up"
    assert all("reason" not in item for item in raw["not_covered"]), "no reason is read"
    # Graded as the listed row's right text reads: this small manifest lists no run and not every mixture and
    # checkpoint of the row's covers, so the covers are wrong; not_covered (E/shard-1 as no record) is right. A reading
    # that lists every name of the covers is right whatever else it lists, so the grade follows the names and not a fit.
    got = graded(q, reading)
    assert got.status == "wrong" and list(got.evidence) == ["covers"]
    expected = _listed_rows()[q.listed_row]["expect"]
    full = {"covers": expected["covers"], "not_covered": raw["not_covered"]}
    assert graded(q, _mod("baselines.interfaces").Reading(q.id, full, q.route)).status == "right"


def test_the_file_family_of_an_import_is_the_files_its_built_from_list_names():
    """The rule: "read the built_from list of the row of import/T ... the reading is the files that list names"."""
    q = question("c06.import.file-family")
    import_rows = [_row("src/A", ["sources/A.jsonl"]), _row("import/T", ["imports/T.jsonl"], ["src/A"])]
    subject = _mod("baselines.subjects")
    context = SimpleNamespace(ran_on=set(), keep_fact=lambda *args, **kwargs: None)
    driver = subject.manifest_factory({"store-import": import_rows}, requests=REQUESTS, traces={}, repaired={})(context)
    reading = driver.read(q)
    assert dict(reading.raw) == {"link": {"files": ["sources/A.jsonl"]}}
    member_files = {"A/r1": ["sources/A.jsonl"]}
    assert graded(q, reading, member_files).status == "right"
    other = subject.manifest_factory({"store-import": [_row("import/T", [], ["src/B"]), _row("src/B", ["sources/B.jsonl"])]},
                                     requests=REQUESTS, traces={}, repaired={})(context)
    assert graded(q, other.read(q), member_files).status == "wrong"


# The manifest of the 10k generated set in each layout (docs/replica.md section 16, "Rows"): a chunk file is the row <shard>.input,
# a shard is the row whose files are its .bin and .idx and whose built_from is its chunk file. The partitioned build has one
# shard for each source file and one for the generation outputs; the round-robin build has numbered chunks.
def _set_rows(prefix, layout, shards):
    rows = [_row("src/A", ["sources/A.jsonl"]), _row("src/B", ["sources/B.jsonl"]),
            _row("gen/P", ["gen/P.jsonl"], ["src/A"])]
    for name, sources in shards:
        full = f"{prefix}/{name}"
        rows.append(_row(f"{full}.input", [f"pipeline/{layout}/tokenizer-input/{name}.jsonl"], sources))
        rows.append(_row(full, [f"pipeline/{layout}/shards/{name}.bin", f"pipeline/{layout}/shards/{name}.idx"], [f"{full}.input"]))
    return rows


PARTITIONED_10K = _set_rows("set", "partitioned", [("shard-A", ["src/A"]), ("shard-B", ["src/B"]), ("shard-gen", ["gen/P"])])
ROUND_ROBIN_10K = _set_rows("set-rr", "round-robin", [(f"shard-0{n}", ["src/A", "src/B"]) for n in (1, 2, 3)])


def test_the_10k_lists_are_read_from_the_shard_rows_and_the_chunk_rows_of_the_manifest_of_the_layout_the_row_names():
    """The table's rule text: the reading of c09.10k.partitioned.kept-shards is the list of shards read keep, of
    c09.10k.round-robin.all-rewrite the number of chunks read rewrite. The artifacts are the rows the manifest names (a shard: its
    .bin and .idx; a chunk: <shard>.input), never a name chosen to match an expected count."""
    manifests = {"set-10k": ROUND_ROBIN_10K, "set-10k-partitioned": PARTITIONED_10K}
    driver, _ = driver_of(manifests=manifests)
    kept = driver.read(question("c09.10k.partitioned.kept-shards"))
    assert dict(kept.raw) == {"keep": ["shard-B"], "registered_names": ["set/shard-B"]}, \
        "shard-A is A's, shard-gen is derived from A (gen/P is built from src/A); named by the stem of its chunk file, as the plant list's counting names it"
    every = driver.read(question("c09.10k.round-robin.all-rewrite"))
    assert dict(every.raw) == {"keep": 0, "rewrite": 3}, "the chunks of the round-robin build are counted, not its shard rows twice"
    # Neither list counts the other kind of row: 3 chunk rows and 3 shard rows are in the round-robin build.
    assert len([r for r in ROUND_ROBIN_10K if r["name"].endswith(".input")]) == 3
    assert len([r for r in ROUND_ROBIN_10K if r["name"].startswith("set-rr/") and not r["name"].endswith(".input")]) == 3
    # The partitioned shards are read in the manifest of the partitioned build: with only the target's own manifest the row is
    # not read (it is not read from the other layout's manifest).
    only_own, facts = driver_of(manifests={"set-10k": ROUND_ROBIN_10K})
    with pytest.raises(_mod("baselines.combo_dolma").NotAttempted):
        only_own.read(question("c09.10k.partitioned.kept-shards"))
    assert any("no manifest of the partitioned build" in str(fact) for fact in facts)


def test_a_mixture_is_no_shard_row_and_two_keys_of_one_target_are_refused():
    """Section 16: a mixture's files are shard files too, but it is built from shards, not from one chunk file; and only the manifest
    may hold a key of a build added beside the target's own."""
    subjects = _mod("baselines.subjects")
    mixture = _row("set/mix-1", ["pipeline/partitioned/shards/shard-A.bin", "pipeline/partitioned/shards/shard-A.idx"], ["set/shard-A", "set/shard-B"])
    assert subjects.shard_rows(PARTITIONED_10K + [mixture]) == ["set/shard-A", "set/shard-B", "set/shard-gen"]
    with pytest.raises(ValueError, match="is given twice"):
        subjects.keyed_by_target({"set-10k": {}, "set-10k-partitioned": {}}, "trace")


# The repaired side of the 10k set in each layout: the successors the repair builds, under the successor names (docs/replica.md
# section 16: "the repaired side is a build with capture on and writes the file, under the successor names"). The historical R1
# still walks to the source A through the joined rows of the two builds.
REPAIRED_10K = [_row("successor of src/A", ["sources/A.jsonl"]), _row("successor of R1", [], ["successor of src/A"])]


def test_the_historical_r1_of_the_10k_set_is_read_in_each_layout_from_the_joined_rows_of_that_layout_and_its_repaired_side():
    """W2 (18 Nov 2026): c09.10k.R1-history names two layouts; the manifest reads each from the manifest of that layout joined with
    the manifest of that layout's repaired side. A layout with no manifest, or no repaired side, is read as None (not attempted)."""
    release = _row("R1", ["releases/R1.jsonl"], ["src/A", "src/B"])
    own, added = [*ROUND_ROBIN_10K, release], [*PARTITIONED_10K, release]
    q = question("c09.10k.R1-history")
    assert q.route is not None and _mod("baselines.record").layouts_of(q) == ("partitioned", "round-robin")
    manifests = {"set-10k": own, "set-10k-partitioned": added}
    repaired = {"set-10k": REPAIRED_10K, "set-10k-partitioned": REPAIRED_10K}
    driver, _ = driver_of(manifests=manifests, repaired_manifests=repaired)
    got = driver.read_by_layout(q)
    assert set(got) == {"partitioned", "round-robin"}
    assert all(dict(reading.raw) == {"answer": "contains"} for reading in got.values()), got
    # Without the repaired side of the partitioned build that layout is not read; the round-robin one is.
    partial, _ = driver_of(manifests=manifests, repaired_manifests={"set-10k": REPAIRED_10K})
    held = partial.read_by_layout(q)
    assert held["partitioned"] is None and dict(held["round-robin"].raw) == {"answer": "contains"}
    # Without the partitioned manifest at all, likewise.
    only_own, _ = driver_of(manifests={"set-10k": own}, repaired_manifests={"set-10k": REPAIRED_10K})
    assert only_own.read_by_layout(q)["partitioned"] is None
    # The successor shards: cannot answer in each layout the subject read, else not read there.
    nofold = _mod("baselines.dolma").NO_JOIN_TEXT
    sq = question("c09.10k.successor-shards")
    both = driver.read_by_layout(sq)
    assert all(dict(reading.raw) == {"cannot_answer": nofold} for reading in both.values())
    assert only_own.read_by_layout(sq)["partitioned"] is None and only_own.read_by_layout(sq)["round-robin"] is not None
    key = _mod("baselines.subjects").repaired_key
    assert key("main") == "store-main" and key("set-10k") == "set-10k" and key("set-10k-partitioned") == "set-10k-partitioned"


def test_the_successor_shards_of_the_10k_set_are_never_folded_into_one_answer_and_a_store_the_world_lacks_is_not_read():
    """The table names no fold of many shards into one answer (D-SUBJ-11): the manifest gives the subject's cannot answer with the
    table's words for c09.10k.successor-shards, as every subject does. The R1 history waits for the repaired side (W2)."""
    not_attempted = _mod("baselines.combo_dolma").NotAttempted
    dolma = _mod("baselines.dolma")
    driver, _ = driver_of(manifests={"set-10k": ROUND_ROBIN_10K, "set-10k-partitioned": PARTITIONED_10K})
    with pytest.raises(ValueError, match=dolma.NO_JOIN_TEXT[:20]):
        driver.read(question("c09.10k.successor-shards"))
    # The cannot answer holds on a layout the subject read: with only the round-robin manifest the row is not read at all.
    only_own, facts = driver_of(manifests={"set-10k": ROUND_ROBIN_10K})
    with pytest.raises(not_attempted):
        only_own.read(question("c09.10k.successor-shards"))
    assert any("no manifest of the partitioned build" in str(fact) for fact in facts)
    with pytest.raises(not_attempted):
        driver.read(question("c09.10k.R1-history"))
    with pytest.raises(not_attempted):
        driver_of()[0].read(question("c06.import.file-family"))   # the world has no import store in this test
