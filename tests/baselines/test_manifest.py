"""The reader of the file-level manifest, on hand-written manifest documents (plan 5.5
Check 10; table README, section "Check 10": the subject ``manifest``; the reading rules
of the manifest rows of ``check10.toml``).

The manifest is the file ``built-from.jsonl`` that a build with capture on writes into
its build folder: one JSON row per artifact, ``{"name", "files", "built_from"}``. A
build made with capture off has no such file. The documents here are written by hand
from the fixture spec (the same names and shapes as ``tests/replica/
test_replica_built_from.py`` expects of a build); no build runs.

What these tests hold the reader to, from the rule text of the manifest rows:

- ``contains``: the walk from the artifact's row, row by row through ``built_from``,
  reaches the file of a member the request names; ``does not contain``: the artifact
  has a row and the walk reaches no such file; ``no record``: the artifact has no row.
  A name with no row of its own ends its branch. The walk ends on a cycle.
- The readings of the unmutated artifacts equal the answers of the listed rows
  (``check05.toml``, ``c05.unmutated.*``) except where the manifest cannot tell what
  the trace tells: a checkpoint's row names the shards its segment was handed and its
  parent checkpoint, so Y@6 and K@3 (whose segments were handed shard 3, which holds T)
  read ``contains`` where the listed rows say ``does not contain``. The reader reads;
  the grade is another module's.
- The reading under a fault is the reading of the changed document (``baselines.
  faults``), and a name the fault left without a row is ``no record`` when asked and ends
  the branch when walked.
- The two lists of Check 9 (``c09.fixture.mixtures``), the import's file family
  (``c06.import.file-family``) and the listing under the fault ``f.map-deleted``
  (``c05.listing.map-deleted``) are read from the rows as the table's own rows say.
- A build made with capture off has no manifest: the reader reports it absent, a
  reading of its own that is neither ``no record`` nor an answer; a manifest file with no
  row is not absent.

The names asked of ``baselines.manifest``:

- ``load_manifest(folder)``: the rows of ``<folder>/built-from.jsonl`` as a list of
  dicts, in file order; ``None`` where the file does not exist (capture off); a file
  with no row is ``[]``. A line that is not JSON, a row with no ``name`` or ``built_from``
  of the right type, or two rows of one name raise ``ManifestError`` (a ``ValueError``).
- ``CONTAINS``, ``DOES_NOT_CONTAIN``, ``NO_RECORD`` (the three words of the rule) and
  ``ABSENT`` (a fourth word, equal to none of them).
- ``read_artifact(rows, artifact, targets)``: the word for ``artifact``, where
  ``targets`` is the collection of artifact names that are the files of the members the
  request names (withdraw A: ``{"src/A"}``). ``rows`` None gives ``ABSENT``.
- ``mixture_lists(rows, mixtures, targets)``: ``{"needs_successor": [...], "kept": [...]}``,
  sorted lists; a mixture read ``contains`` is in the first, ``does not contain`` in the
  second, ``no record`` in neither. ``rows`` None gives ``ABSENT``.
- ``artifact_actions(rows, artifacts, targets)``: the actions of Check 9 read from the
  walk by the plan's own definition (contains is read as rewrite, does not contain as
  keep): ``{"keep": [...], "rewrite": [...], "counts": {"keep": n, "rewrite": m}}``,
  the lists sorted, an artifact read ``no record`` in neither list. ``rows`` None gives
  ``ABSENT``.
- ``exposure_free_restart(rows, checkpoints, targets)``: ``checkpoints`` are the
  checkpoints of one run in step order. The name of the latest checkpoint read does not
  contain such that no checkpoint up to it (itself included) is read otherwise; None where
  the first is not read does not contain (a checkpoint read ``no record`` is read
  otherwise). ``rows`` None gives ``ABSENT``.
- ``import_family(rows, name)``: the ``built_from`` list of the row of ``name`` as the
  row has it; ``NO_RECORD`` where there is no row; ``ABSENT`` for None.
- ``listing(rows)``: a dict that lists every name the manifest holds, as a row of its
  own or inside a ``built_from`` list, by kind: ``sources`` (``src/``), ``generated``
  (``gen/``), ``releases`` (``R1``, ``R2``), ``imports`` (``import/``), ``chunk_files``
  (names ending ``.input``), ``shards``, ``mixtures`` (``/mix-``), ``checkpoints``
  (names with ``@``) and ``runs`` (the manifest holds none), each a sorted list; a name
  with no row of its own is in ``no_record`` and in no kind. ``rows`` None gives ``ABSENT``.
- ``GROUP_READINGS``: for each group of ``check10.toml`` whose manifest row has a record
  (the groups the table routes to the manifest), the name of the function of this module
  that reads it, as the row's own route and rule name it.
- ``read_group(group, rows, *, targets=(), artifact=None, artifacts=(), checkpoints=())``:
  the reading of a routed group through its function (``artifact`` for a walk and for the
  import's file family, ``artifacts`` for the two lists and the actions, ``checkpoints``
  for the restart; none for the listing). A group the manifest cannot answer (its row
  records none) raises ``ManifestError``.
- ``join_manifests(*manifests)``: the rows of the manifests of two folders (the historical
  build and the repaired one) in one list, in the order given, joined by artifact name; a
  name in two of them is refused (``ManifestError``) unless the two rows are equal, and
  then it is one row, at its first place; a manifest that is None (absent) makes the join
  None. The repaired side's own names carry the prefix ``successor of ``, so a real repair
  shares no name; the tests that read both folders of a repair are in
  ``tests/replica/test_replica_built_from.py``.

The reader changes no document it is given. A module that does not exist yet fails a test
on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import copy
import json
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = "baselines.manifest"
FAULTS = "baselines.faults"
MANIFEST_FILE = "built-from.jsonl"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "replica",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")

# The words of the rule, as the table writes them.
CONTAINS = "contains"
DOES_NOT_CONTAIN = "does not contain"
NO_RECORD = "no record"

WITHDRAW_A = {"src/A"}


def table(number: int) -> dict:
    with open(REPO / "fixture" / "table" / f"check{number:02d}.toml", "rb") as f:
        return tomllib.load(f)


def table_row(number: int, row_id: str) -> dict:
    (found,) = [r for r in table(number)["row"] if r["id"] == row_id]
    return found


# --- The hand-written manifest -----------------------------------------------------------


def row(name: str, files: list[str], built_from: list[str]) -> dict:
    return {"name": name, "files": files, "built_from": built_from}


# A store folder holds the build folder and the runs folder side by side (build/ and runs/,
# as the acceptance world lays a store out): a checkpoint's file, the run record of its
# segment, is written as the path from the build folder to it, which begins with ``..``
# (the table README: file paths relative to the build folder; docs/replica.md section 16).
RUNS_FROM_BUILD = "../runs"


# From the fixture spec: [sources] (one file each), [generation] (P from A-001, T from P,
# S from A-002 and C-001), [releases.R1] and [releases.R2] (the copies kept: R1 keeps A's
# copy of X and so names no file of B), [imports."import/T"] (file family A/r1, the member
# of source A), [layouts.partitioned.shards] (shards 1 and 2 hold B and Z records, shard 3
# holds T and Z records), [mixtures] (the trainer's order), [runs.*.segments] (the shards a
# segment is handed and its parent checkpoint). This is the main store; the rr store holds
# E-rr@6 in a manifest of its own.
def make_manifest() -> list[dict]:
    rows = [row(f"src/{k}", [f"sources/{k}.jsonl"], []) for k in "ABCZ"]
    rows += [
        row("gen/P", ["gen/P.jsonl"], ["src/A"]),
        row("gen/T", ["gen/T.jsonl"], ["gen/P"]),
        row("gen/S", ["gen/S.jsonl"], ["src/A", "src/C"]),
        row("R1", ["releases/R1.jsonl"], ["src/A", "gen/P", "gen/T", "gen/S", "src/Z"]),
        row("R2", ["releases/R2.jsonl"], ["src/B", "src/Z"]),
        row("import/T", ["imports/T.jsonl"], ["src/A"]),
    ]
    chunks = {"shard-1": ["src/B", "src/Z"], "shard-2": ["src/B", "src/Z"],
              "shard-3": ["gen/T", "src/Z"]}
    for shard, files in chunks.items():
        rows.append(row(f"E/{shard}.input", [f"pipeline/partitioned/tokenizer-input/{shard}.jsonl"],
                        list(files)))
        rows.append(row(f"E/{shard}", [f"pipeline/partitioned/shards/{shard}.bin",
                                       f"pipeline/partitioned/shards/{shard}.idx"],
                        [f"E/{shard}.input"]))
    mixtures = {
        "E/mix-1": ["E/shard-1", "E/shard-2"],
        "E/mix-2": ["E/shard-1", "E/shard-2", "E/shard-3"],
        "D/mix-1": ["E/shard-1", "E/shard-2"],
        "Y/mix-1": ["E/shard-1", "E/shard-2", "E/shard-3"],
        "K/mix-1": ["E/shard-3", "E/shard-2"],
    }
    for name, shards in mixtures.items():
        files = [f"mixtures/{name}/{s.rsplit('/', 1)[1]}{suffix}"
                 for s in shards for suffix in (".bin", ".idx")]
        rows.append(row(name, files, list(shards)))
    # A checkpoint's row names the shards its segment was handed and its parent checkpoint.
    rows += [
        row("E@6", [f"{RUNS_FROM_BUILD}/E/E-seg-1.run.json"], ["E/shard-1", "E/shard-2"]),
        row("E@9", [f"{RUNS_FROM_BUILD}/E/E-seg-2.run.json"],
            ["E/shard-1", "E/shard-2", "E/shard-3", "E@6"]),
        row("D@3", [f"{RUNS_FROM_BUILD}/D/D-seg-1.run.json"], ["E/shard-1", "E/shard-2", "E@9"]),
        row("Y@6", [f"{RUNS_FROM_BUILD}/Y/Y-seg-1.run.json"],
            ["E/shard-1", "E/shard-2", "E/shard-3", "E@6"]),
        row("K@3", [f"{RUNS_FROM_BUILD}/K/K-seg-1.run.json"], ["E/shard-3", "E/shard-2", "E@6"]),
        row("K@8", [f"{RUNS_FROM_BUILD}/K/K-seg-1.run.json"], ["E/shard-3", "E/shard-2", "E@6"]),
        row("K@40", [f"{RUNS_FROM_BUILD}/K/K-seg-1.run.json"], ["E/shard-3", "E/shard-2", "E@6"]),
    ]
    return rows


def write_manifest(folder: Path, rows: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / MANIFEST_FILE).write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# Looked up when a test calls it, never at setup: a module that does not exist yet is a
# failed assertion in the body of a test, not an error.
@pytest.fixture
def load_manifest(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "load_manifest")(*a, **k)


@pytest.fixture
def read_artifact(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "read_artifact")(*a, **k)


@pytest.fixture
def mixture_lists(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "mixture_lists")(*a, **k)


@pytest.fixture
def import_family(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "import_family")(*a, **k)


@pytest.fixture
def artifact_actions(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "artifact_actions")(*a, **k)


@pytest.fixture
def exposure_free_restart(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "exposure_free_restart")(*a, **k)


@pytest.fixture
def listing(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "listing")(*a, **k)


@pytest.fixture
def apply_fault(need_attribute):
    return lambda *a, **k: need_attribute(FAULTS, "apply_fault")(*a, **k)


@pytest.fixture
def words(need_attribute):
    """A function: (CONTAINS, DOES_NOT_CONTAIN, NO_RECORD, ABSENT) as the module has them,
    looked up when it is called."""
    return lambda: tuple(need_attribute(MODULE, n) for n in
                         ("CONTAINS", "DOES_NOT_CONTAIN", "NO_RECORD", "ABSENT"))


# --- The hand-written document follows the spec ------------------------------------------


def fixture_spec() -> dict:
    return tomllib.loads((REPO / "fixture" / "spec" / "fixture_spec.toml").read_text("utf-8"))


def test_the_hand_written_release_rows_name_the_copies_kept_and_not_the_copy_dedup_dropped():
    """From the spec: R1 takes A-001 and B-001, the same passage, and keeps the copy of the
    source named by ``dedup_winner`` (A); B-001 is the only record of B it takes, so the file
    of B did not go into R1 and R1 does not list it. R2 takes records of B and Z and none of
    A, so the copy of B is the one kept there."""
    releases = fixture_spec()["releases"]
    assert releases["R1"]["dedup_winner"] == "A"
    inputs = releases["R1"]["inputs"]
    assert "A-001" in inputs and "B-001" in inputs
    assert [i for i in inputs if i.startswith("B-")] == ["B-001"]
    assert all(not i.startswith("A-") for i in releases["R2"]["inputs"])
    rows = {r["name"]: r for r in make_manifest()}
    assert "src/A" in rows["R1"]["built_from"]
    assert "src/B" not in rows["R1"]["built_from"]
    assert {"src/B", "src/Z"} == set(rows["R2"]["built_from"])


def test_the_hand_written_checkpoint_rows_name_the_run_record_of_their_segment_from_the_build():
    """A checkpoint's file is the path from the build folder to the run record of the
    segment that saved it, in the runs folder beside the build folder (docs/replica.md
    section 16); the segment is the one the spec's step record gives the checkpoint. The
    main store holds every checkpoint of the spec but E-rr@6, which the rr store holds."""
    expected = {}
    for run, doc in fixture_spec()["runs"].items():
        for segment, record in doc.get("step_record", {}).items():
            for checkpoint in record["checkpoints"]:
                expected[checkpoint["name"]] = (
                    *Path(RUNS_FROM_BUILD).parts, run, segment.replace("/", "-") + ".run.json")
    rows = {r["name"]: r for r in make_manifest() if "@" in r["name"]}
    assert set(rows) == set(expected) - {"E-rr@6"}
    for name, row_ in rows.items():
        assert [Path(f).parts for f in row_["files"]] == [expected[name]], name
        assert expected[name][0] == "..", name


# --- The words ---------------------------------------------------------------------------


def test_the_three_words_are_the_rule_s_and_the_absent_word_is_a_fourth(words):
    contains, does_not, no_record, absent = words()
    assert (contains, does_not, no_record) == (CONTAINS, DOES_NOT_CONTAIN, NO_RECORD)
    assert absent not in (CONTAINS, DOES_NOT_CONTAIN, NO_RECORD)
    assert isinstance(absent, str) and absent


# --- Loading the file --------------------------------------------------------------------


def test_a_folder_with_no_manifest_file_reads_as_absent_not_as_empty(load_manifest, tmp_path):
    """A build made with capture off wrote no file: the reader says so with None."""
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "A.jsonl").write_text('{"id": "A-001", "text": "x"}\n')
    assert load_manifest(tmp_path) is None


def test_a_manifest_file_with_no_row_is_an_empty_list_and_not_absent(load_manifest, tmp_path):
    (tmp_path / MANIFEST_FILE).write_text("", encoding="utf-8")
    got = load_manifest(tmp_path)
    assert got is not None and list(got) == []


def test_the_rows_are_read_in_file_order_as_they_were_written(load_manifest, tmp_path):
    rows = make_manifest()
    write_manifest(tmp_path, rows)
    assert [dict(r) for r in load_manifest(tmp_path)] == rows


def test_the_per_line_manifests_beside_record_files_are_not_the_manifest(load_manifest,
                                                                         tmp_path):
    """``<file>.manifest.jsonl`` holds the line, handle and origin of each record: it is
    test evidence of the replica and not this manifest."""
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "A.jsonl.manifest.jsonl").write_text(
        '{"line": 1, "handle": "X@A", "origin": ["A-001"]}\n', encoding="utf-8")
    assert load_manifest(tmp_path) is None


@pytest.mark.parametrize("text", [
    "not json\n",
    '{"name": "src/A", "files": ["sources/A.jsonl"]}\n',                     # no built_from
    '{"files": ["sources/A.jsonl"], "built_from": []}\n',                    # no name
    '{"name": "src/A", "files": [], "built_from": "src/B"}\n',               # not a list
    '{"name": "src/A", "files": [], "built_from": []}\n'
    '{"name": "src/A", "files": [], "built_from": []}\n',                    # a name twice
    '["src/A"]\n',                                                           # not an object
])
def test_a_file_that_is_not_a_manifest_is_refused_and_not_read_as_far_as_it_goes(
        need_attribute, tmp_path, text):
    error = need_attribute(MODULE, "ManifestError")
    load = need_attribute(MODULE, "load_manifest")
    assert issubclass(error, ValueError)
    (tmp_path / MANIFEST_FILE).write_text(text, encoding="utf-8")
    with pytest.raises(error):
        load(tmp_path)


# --- The walk ----------------------------------------------------------------------------


def unmutated_answers() -> dict[str, str]:
    """``c05.unmutated.<artifact>`` of the main store (no ``store`` of its own) to the
    answer of the listed row, as the table writes it."""
    return {r["given"]["artifact"]: r["expect"]["answer"] for r in table(5)["row"]
            if r["id"].startswith("c05.unmutated.") and "store" not in r["given"]
            and r["given"].get("state") == "unmutated"}


# The artifacts of the main store the manifest holds a row for and the manifest cannot
# read as the trace does: Y@6 and K@3 are handed shard 3 (which holds T) by their
# segments, though no step of theirs reads T.
MANIFEST_CANNOT_TELL = {"Y@6", "K@3"}


def test_the_readings_of_the_unmutated_artifacts_equal_the_listed_answers_but_two(
        read_artifact):
    rows = make_manifest()
    answers = unmutated_answers()
    asked = [r["name"] for r in rows if r["name"] in answers]
    assert set(asked) == set(answers), "the document holds a row for every artifact of the table"
    differ = {a for a in asked if read_artifact(rows, a, WITHDRAW_A) != answers[a]}
    assert differ == MANIFEST_CANNOT_TELL
    for artifact in MANIFEST_CANNOT_TELL:
        assert answers[artifact] == DOES_NOT_CONTAIN
        assert read_artifact(rows, artifact, WITHDRAW_A) == CONTAINS


def test_a_source_file_that_is_a_target_contains_itself_and_another_does_not(read_artifact):
    rows = make_manifest()
    assert read_artifact(rows, "src/A", WITHDRAW_A) == CONTAINS
    for other in ("src/B", "src/C", "src/Z"):
        assert read_artifact(rows, other, WITHDRAW_A) == DOES_NOT_CONTAIN


def test_the_walk_goes_down_through_every_row_to_the_source_file(read_artifact):
    rows = make_manifest()
    # T was made from P, which was made from A: three rows down.
    assert read_artifact(rows, "gen/T", WITHDRAW_A) == CONTAINS
    # K@8: checkpoint, then shard 3, then its chunk file, then T, then P, then A.
    assert read_artifact(rows, "K@8", WITHDRAW_A) == CONTAINS
    # Another request names another file: withdraw C is reached by S only.
    assert read_artifact(rows, "gen/S", {"src/C"}) == CONTAINS
    assert read_artifact(rows, "gen/T", {"src/C"}) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "E/mix-2", {"src/C"}) == DOES_NOT_CONTAIN


def test_a_request_that_names_two_files_is_met_by_either(read_artifact):
    rows = make_manifest()
    assert read_artifact(rows, "R2", {"src/A", "src/B"}) == CONTAINS
    assert read_artifact(rows, "R2", {"src/A", "src/C"}) == DOES_NOT_CONTAIN


def test_a_name_with_no_row_is_no_record_and_a_checkpoint_never_registered_too(read_artifact):
    rows = make_manifest()
    # c05.unregistered-checkpoint: E@12 was never registered; the reading is no record.
    assert read_artifact(rows, "E@12", WITHDRAW_A) == NO_RECORD
    assert read_artifact(rows, "no-such-name", WITHDRAW_A) == NO_RECORD


def test_an_artifact_with_a_row_and_no_file_reached_is_not_no_record(read_artifact):
    rows = make_manifest()
    rows.append(row("R3", ["releases/R3.jsonl"], []))
    assert read_artifact(rows, "R3", WITHDRAW_A) == DOES_NOT_CONTAIN


def test_a_name_with_no_row_of_its_own_ends_its_branch(read_artifact):
    """gen/P is built from a name no row has (src/Q): the branch ends there, the row of
    gen/P holds no file of a withdrawn member, and the name asked on its own is no record."""
    rows = make_manifest()
    for r in rows:
        if r["name"] == "gen/P":
            r["built_from"] = ["src/Q"]
    assert read_artifact(rows, "gen/P", WITHDRAW_A) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "gen/T", WITHDRAW_A) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "src/Q", WITHDRAW_A) == NO_RECORD


def test_a_cycle_in_built_from_ends_the_walk(read_artifact):
    rows = [row("a", ["a.jsonl"], ["b"]), row("b", ["b.jsonl"], ["a"]),
            row("src/A", ["sources/A.jsonl"], [])]
    assert read_artifact(rows, "a", WITHDRAW_A) == DOES_NOT_CONTAIN
    rows[1]["built_from"] = ["a", "src/A"]
    assert read_artifact(rows, "a", WITHDRAW_A) == CONTAINS


def test_the_reading_does_not_depend_on_the_order_of_the_rows(read_artifact):
    rows = make_manifest()
    shuffled = list(reversed(rows))
    for r in rows:
        assert read_artifact(shuffled, r["name"], WITHDRAW_A) == read_artifact(
            rows, r["name"], WITHDRAW_A), r["name"]


# --- Under the faults of Checks 4 and 5 --------------------------------------------------


def test_the_edge_replaced_by_a_name_no_row_has_cuts_gen_p_and_what_is_built_from_it(
        read_artifact, apply_fault):
    """l.edge-parent: the entry src/A of the built_from list of gen/P is replaced by a name
    that no row has. gen/P and gen/T read ``does not contain``; what lists src/A
    itself (gen/S, R1) still does."""
    rows = apply_fault("l.edge-parent", "manifest", make_manifest())
    assert read_artifact(rows, "gen/P", WITHDRAW_A) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "gen/T", WITHDRAW_A) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "gen/S", WITHDRAW_A) == CONTAINS
    assert read_artifact(rows, "R1", WITHDRAW_A) == CONTAINS
    # E/mix-2 holds shard 3, whose chunk file lists T: the only way to A there is P.
    assert read_artifact(rows, "E/shard-3", WITHDRAW_A) == DOES_NOT_CONTAIN


def test_the_edge_entry_deleted_reads_as_the_edge_replaced(read_artifact, apply_fault):
    """f.edge-deleted: the entry src/A is removed and the row of gen/P stays."""
    deleted = apply_fault("f.edge-deleted", "manifest", make_manifest())
    replaced = apply_fault("l.edge-parent", "manifest", make_manifest())
    assert any(r["name"] == "gen/P" for r in deleted)
    for r in make_manifest():
        assert read_artifact(deleted, r["name"], WITHDRAW_A) == read_artifact(
            replaced, r["name"], WITHDRAW_A), r["name"]


def test_the_row_of_a_shard_deleted_is_no_record_and_ends_the_branch_of_what_names_it(
        read_artifact, apply_fault):
    """f.map-deleted: the row of E/shard-1 is deleted; the rows that name it stay."""
    rows = apply_fault("f.map-deleted", "manifest", make_manifest())
    assert read_artifact(rows, "E/shard-1", WITHDRAW_A) == NO_RECORD
    # E/mix-1 names E/shard-1 (no row now) and E/shard-2 (B and Z): no A on any branch.
    assert read_artifact(rows, "E/mix-1", WITHDRAW_A) == DOES_NOT_CONTAIN
    assert read_artifact(rows, "E/mix-2", WITHDRAW_A) == CONTAINS


# --- The two lists of Check 9 ------------------------------------------------------------


MAIN_MIXTURES = ["E/mix-1", "E/mix-2", "D/mix-1", "Y/mix-1", "K/mix-1"]


def test_the_two_lists_equal_the_listed_row_as_sets(mixture_lists):
    expect = table_row(9, "c09.fixture.mixtures")["expect"]
    got = mixture_lists(make_manifest(), MAIN_MIXTURES, WITHDRAW_A)
    assert set(got) == {"needs_successor", "kept"}
    assert sorted(got["needs_successor"]) == sorted(expect["needs_successor"])
    assert sorted(got["kept"]) == sorted(expect["kept"])
    # The five names are asked once: no mixture is in both lists.
    assert not set(got["needs_successor"]) & set(got["kept"])


def test_a_mixture_with_no_row_is_in_neither_list(mixture_lists):
    rows = [r for r in make_manifest() if r["name"] != "D/mix-1"]
    got = mixture_lists(rows, MAIN_MIXTURES, WITHDRAW_A)
    assert "D/mix-1" not in got["needs_successor"] and "D/mix-1" not in got["kept"]
    assert sorted(got["kept"]) == ["E/mix-1"]
    assert sorted(got["needs_successor"]) == ["E/mix-2", "K/mix-1", "Y/mix-1"]


def test_the_lists_follow_a_fault_on_the_edge(mixture_lists, apply_fault):
    """Under l.edge-parent no mixture reaches A: the manifest has lost the edge, and
    the list that needs a successor is empty."""
    rows = apply_fault("l.edge-parent", "manifest", make_manifest())
    got = mixture_lists(rows, MAIN_MIXTURES, WITHDRAW_A)
    assert got["needs_successor"] == []
    assert sorted(got["kept"]) == sorted(MAIN_MIXTURES)


# --- The other requests and the other routed rows ----------------------------------------


def test_a_request_that_names_source_c_reads_the_listed_answers_of_withdraw_c(read_artifact):
    """c05.withdraw-C.unmutated.*: E/shard-3 (T, P and A, no C) does not contain, R1
    (through S) contains."""
    rows = make_manifest()
    answers = {r["given"]["artifact"]: r["expect"]["answer"] for r in table(5)["row"]
               if r["id"].startswith("c05.withdraw-C.unmutated.")}
    assert answers == {"E/shard-3": DOES_NOT_CONTAIN, "R1": CONTAINS}
    for artifact, answer in answers.items():
        assert read_artifact(rows, artifact, {"src/C"}) == answer, artifact


def test_the_import_under_withdraw_a_reads_the_listed_answer_through_its_file_family(
        read_artifact):
    """c06.import.withdraw-A: the import's row names the file of the family's member."""
    answer = table_row(6, "c06.import.withdraw-A")["expect"]["answer"]
    assert answer == CONTAINS
    assert read_artifact(make_manifest(), "import/T", WITHDRAW_A) == answer


def test_the_actions_of_check_9_are_the_walk_read_as_rewrite_and_keep(artifact_actions):
    """c09.fixture.shards.counts and the actions of R1, R2 and the shards: contains is
    rewrite, does not contain is keep."""
    expect = table_row(9, "c09.fixture.shards.counts")["expect"]
    got = artifact_actions(make_manifest(), ["E/shard-1", "E/shard-2", "E/shard-3"], WITHDRAW_A)
    assert sorted(got["keep"]) == sorted(expect["keep"])
    assert sorted(got["rewrite"]) == sorted(expect["rewrite"])
    assert dict(got["counts"]) == dict(expect["counts"])
    for artifact in ("R1", "R2", "E/shard-1", "E/shard-2", "E/shard-3"):
        action = table_row(9, f"c09.fixture.{artifact}.action")["expect"]["action"]
        read = artifact_actions(make_manifest(), [artifact], WITHDRAW_A)
        assert (artifact in read["rewrite"]) == (action == "rewrite"), artifact
        assert (artifact in read["keep"]) == (action == "keep"), artifact


def test_an_artifact_with_no_row_is_neither_kept_nor_rewritten(artifact_actions):
    rows = [r for r in make_manifest() if r["name"] != "E/shard-2"]
    got = artifact_actions(rows, ["E/shard-1", "E/shard-2", "E/shard-3"], WITHDRAW_A)
    assert got["keep"] == ["E/shard-1"] and got["rewrite"] == ["E/shard-3"]
    assert dict(got["counts"]) == {"keep": 1, "rewrite": 1}


def test_the_exposure_free_restart_is_the_latest_checkpoint_read_does_not_contain_with_none_before_it_otherwise(
        exposure_free_restart):
    expect = table_row(9, "c09.E.exposure-free-restart")["expect"]["exposure_free_restart"]
    assert exposure_free_restart(make_manifest(), ["E@6", "E@9"], WITHDRAW_A) == expect == "E@6"


def test_a_later_checkpoint_read_does_not_contain_after_one_read_contains_is_not_a_restart(
        exposure_free_restart):
    rows = make_manifest()
    rows.append(row("E@12", ["E/E-seg-3.run.json"], ["E/shard-1", "E/shard-2"]))
    # E@9 contains, so E@12 (which does not) is no restart: an earlier checkpoint is read otherwise.
    assert exposure_free_restart(rows, ["E@6", "E@9", "E@12"], WITHDRAW_A) == "E@6"


def test_a_checkpoint_with_no_row_ends_the_prefix_and_a_first_one_read_contains_gives_none(
        exposure_free_restart):
    rows = make_manifest()
    # E@12 has no row: read otherwise than does not contain, it ends the prefix at E@6.
    assert exposure_free_restart(rows, ["E@6", "E@12"], WITHDRAW_A) == "E@6"
    # The first checkpoint is read contains (Y@6 reads contains here): there is none.
    assert exposure_free_restart(rows, ["Y@6", "E@6"], WITHDRAW_A) is None
    assert exposure_free_restart(rows, [], WITHDRAW_A) is None


def test_the_two_new_readings_on_an_absent_manifest_are_absent(words, artifact_actions,
                                                               exposure_free_restart):
    absent = words()[3]
    assert artifact_actions(None, ["R1"], WITHDRAW_A) == absent
    assert exposure_free_restart(None, ["E@6"], WITHDRAW_A) == absent


# --- The import's file family ------------------------------------------------------------


def test_the_import_names_the_file_of_the_member_its_file_family_gives(import_family):
    link = table_row(6, "c06.import.file-family")["expect"]["link"]
    spec = tomllib.loads((REPO / "fixture" / "spec" / "fixture_spec.toml").read_text("utf-8"))
    (source,) = [k for k, v in spec["sources"].items() if v["member"] == link["member"]]
    assert import_family(make_manifest(), "import/T") == [f"src/{source}"]


def test_an_import_with_no_row_is_no_record_and_the_family_list_is_given_as_the_row_has_it(
        import_family):
    rows = [r for r in make_manifest() if r["name"] != "import/T"]
    assert import_family(rows, "import/T") == NO_RECORD
    rows = make_manifest()
    for r in rows:
        if r["name"] == "import/T":
            r["built_from"] = ["src/A", "src/B"]
    assert import_family(rows, "import/T") == ["src/A", "src/B"]


# --- The listing -------------------------------------------------------------------------


def test_the_listing_is_by_kind_and_every_name_is_there_once(listing):
    got = listing(make_manifest())
    assert got["sources"] == [f"src/{k}" for k in "ABCZ"]
    assert got["generated"] == ["gen/P", "gen/S", "gen/T"]
    assert got["releases"] == ["R1", "R2"]
    assert got["imports"] == ["import/T"]
    assert got["chunk_files"] == [f"E/shard-{n}.input" for n in (1, 2, 3)]
    assert got["shards"] == [f"E/shard-{n}" for n in (1, 2, 3)]
    assert got["mixtures"] == sorted(MAIN_MIXTURES)
    assert got["checkpoints"] == sorted(["E@6", "E@9", "D@3", "Y@6", "K@3", "K@8", "K@40"])
    assert got["runs"] == []
    assert got["no_record"] == []


def test_the_listing_after_the_shard_row_is_deleted_lists_it_as_no_record_and_what_is_covered(
        listing, apply_fault):
    """c05.listing.map-deleted: the row of E/shard-1 is deleted, the rows that name it stay.
    The listed row's covers (mixtures, shards, checkpoints, releases) are listed, shard 1
    is listed as no record instead of left out. The manifest holds no run, so ``runs`` is
    empty: a listing of the table's runs is left out, and that is the manifest's reading."""
    expect = table_row(5, "c05.listing.map-deleted")["expect"]
    rows = apply_fault("f.map-deleted", "manifest", make_manifest())
    got = listing(rows)
    for kind in ("mixtures", "shards", "checkpoints", "releases"):
        assert got[kind] == sorted(expect["covers"][kind]), kind
    assert got["no_record"] == sorted(x["artifact"] for x in expect["not_covered"])
    assert "E/shard-1" not in got["shards"]
    assert got["runs"] == []
    assert expect["covers"]["runs"], "the table lists runs the manifest cannot"


def test_a_name_only_inside_a_built_from_list_is_listed_as_no_record_once(listing):
    rows = make_manifest()
    for r in rows:
        if r["name"] == "gen/P":
            r["built_from"] = ["src/Q"]
        if r["name"] == "gen/S":
            r["built_from"] = ["src/Q", "src/C"]
    got = listing(rows)
    assert got["no_record"] == ["src/Q"]
    assert "src/Q" not in got["sources"]


# --- A build with nothing recorded -------------------------------------------------------


def test_a_manifest_that_is_absent_reads_absent_for_every_question_and_never_as_no_record(
        words, read_artifact, mixture_lists, import_family, listing):
    absent = words()[3]
    assert read_artifact(None, "gen/P", WITHDRAW_A) == absent
    assert read_artifact(None, "E@12", WITHDRAW_A) == absent
    assert mixture_lists(None, MAIN_MIXTURES, WITHDRAW_A) == absent
    assert import_family(None, "import/T") == absent
    assert listing(None) == absent
    assert absent != words()[2]


def test_a_manifest_with_no_row_reads_no_record_and_not_absent(words, read_artifact,
                                                               import_family, listing):
    """An empty file is a manifest that holds nothing: every artifact has no row."""
    assert read_artifact([], "gen/P", WITHDRAW_A) == NO_RECORD
    assert import_family([], "import/T") == NO_RECORD
    got = listing([])
    assert got != words()[3] and got["no_record"] == [] and got["sources"] == []


def test_a_folder_without_the_file_is_read_through_to_absent(load_manifest, read_artifact,
                                                           words, tmp_path):
    """The whole path of a capture-off build: no file, so None, so absent."""
    assert read_artifact(load_manifest(tmp_path), "gen/P", WITHDRAW_A) == words()[3]
    write_manifest(tmp_path, make_manifest())
    assert read_artifact(load_manifest(tmp_path), "gen/P", WITHDRAW_A) == CONTAINS


# --- The readings per group --------------------------------------------------------------


def routed_groups() -> dict[str, dict]:
    """The groups of ``check10.toml`` whose manifest row has a record (the others are
    ``none``: the manifest cannot answer them), by group id, each as its manifest row."""
    return {r["given"]["group"]: r for r in table(10)["row"]
            if r["given"]["subject"] == "manifest" and r["record"] != "none"}


def reading_the_row_names(row: dict) -> str:
    """The function a manifest row's own words name: its route and its rule."""
    route, rule = row["route"], row["rule"]
    if route.startswith("list every name"):
        return "listing"
    if route.startswith("read the built_from list of the row of import/T"):
        return "import_family"
    if "the two lists over the five mixtures" in rule:
        return "mixture_lists"
    if "is turned into its action" in rule:
        return "artifact_actions"
    if "the exposure-free restart by the plan's own definition" in rule:
        return "exposure_free_restart"
    return "read_artifact"


@pytest.fixture
def group_readings(need_attribute):
    return lambda: need_attribute(MODULE, "GROUP_READINGS")


@pytest.fixture
def read_group(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "read_group")(*a, **k)


@pytest.fixture
def join_manifests(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "join_manifests")(*a, **k)


def test_the_readings_name_every_group_the_table_routes_to_the_manifest(group_readings):
    routed = routed_groups()
    assert len(routed) > 0
    assert set(group_readings()) == set(routed)


def test_each_group_has_the_reading_its_manifest_row_names(group_readings):
    routed = routed_groups()
    named = {group: reading_the_row_names(row) for group, row in routed.items()}
    # The five readings that are not the walk of one artifact are each the reading of one
    # group, so that a change of the rows' wording is seen here.
    assert sorted(n for n in named.values() if n != "read_artifact") == [
        "artifact_actions", "exposure_free_restart", "import_family", "listing",
        "mixture_lists"]
    assert dict(group_readings()) == named


def test_a_group_the_manifest_cannot_answer_is_refused_and_not_read(read_group, need_attribute):
    error = need_attribute(MODULE, "ManifestError")
    cannot = [r["given"]["group"] for r in table(10)["row"]
              if r["given"]["subject"] == "manifest" and r["record"] == "none"]
    assert "c01.positions" in cannot
    with pytest.raises(error):
        read_group("c01.positions", make_manifest(), artifact="E@9", targets=WITHDRAW_A)


def test_the_walk_groups_are_read_by_the_walk_of_one_artifact(read_group):
    """Every walk group gives, for each artifact of the table's unmutated rows, the answer
    the listed row has (the two the manifest cannot tell as the trace does left out), and
    for E@12 the word of ``c05.unregistered-checkpoint``: its "not covered" is read as no
    record (the rule text of ``c09.successors`` pairs them)."""
    rows = make_manifest()
    answers = {a: w for a, w in unmutated_answers().items() if a not in MANIFEST_CANNOT_TELL}
    unregistered = table_row(5, "c05.unregistered-checkpoint")
    assert unregistered["expect"]["answer"] == "not covered"
    answers[unregistered["given"]["artifact"]] = NO_RECORD
    walk_groups = [g for g, r in routed_groups().items()
                   if reading_the_row_names(r) == "read_artifact"]
    assert len(walk_groups) > 1
    for group in walk_groups:
        for artifact, answer in answers.items():
            assert read_group(group, rows, artifact=artifact, targets=WITHDRAW_A) == answer, (
                group, artifact)


def test_the_other_groups_are_read_as_the_table_rows_of_check_9_and_6_say(read_group):
    rows = make_manifest()
    lists = table_row(9, "c09.fixture.mixtures")["expect"]
    got = read_group("c09.mixture-successors", rows, artifacts=MAIN_MIXTURES, targets=WITHDRAW_A)
    assert sorted(got["needs_successor"]) == sorted(lists["needs_successor"])
    assert sorted(got["kept"]) == sorted(lists["kept"])

    counts = table_row(9, "c09.fixture.shards.counts")["expect"]
    got = read_group("c09.artifact-actions", rows,
                     artifacts=["E/shard-1", "E/shard-2", "E/shard-3"], targets=WITHDRAW_A)
    assert sorted(got["keep"]) == sorted(counts["keep"])
    assert sorted(got["rewrite"]) == sorted(counts["rewrite"])
    assert dict(got["counts"]) == dict(counts["counts"])

    wanted = table_row(9, "c09.E.exposure-free-restart")["expect"]["exposure_free_restart"]
    got = read_group("c09.exposure-free-restart", rows, checkpoints=["E@6", "E@9"],
                     targets=WITHDRAW_A)
    assert got == wanted

    assert read_group("c06.import.file-family", rows, artifact="import/T") == ["src/A"]
    assert read_group("c05.listing.map-deleted", rows)["no_record"] == []


# --- The set-size rows (c09.10k.*), on a hand-written manifest of the 10k set -------------


def make_set_manifest(from_release: bool = False) -> list[dict]:
    """The 10k generated set built with capture on: sources A, B, C and Z, the generation
    outputs P (from A), T (from P) and S (from A and C), the release R1 over them all, the
    partitioned layout (one shard per source, ``shard-gen`` for the generation outputs,
    registered as ``set/<shard>``) and the round-robin layout (20 chunks, registered as
    ``set-rr/<shard>``, each dealt rows of some of the files).

    Changed 17 Nov 2026 (old: the generation outputs of the fixture stores, ``gen/P``, ``gen/T`` and ``gen/S``): a SET build
    names its generation outputs by job, ``gen/paraphrase``, ``gen/translation`` and ``gen/summary`` (docs/replica.md, the
    build of a set: "one output file per job"); the earlier box run of the 10k set showed it (its R1 row names them).

    The chunk row of a shard names the set's own files its lines were taken from, as the
    README row says ("the files of the records its lines hold"): the source files and
    generation outputs the rows of a shard came from. This is the shape a set build writes;
    no set shard walks to ``src/A`` through R1. With ``from_release`` the chunk names the
    release file its lines were cut from: that is the shape a recorder must not write, kept
    here only to show what a reader makes of it. The files of a record dealt to a chunk and
    dropped by the sanitizer are left open."""
    rows = [row(f"src/{k}", [f"sources/{k}.jsonl"], []) for k in "ABCZ"]
    rows += [
        row("gen/paraphrase", ["gen/paraphrase.jsonl"], ["src/A"]),
        row("gen/translation", ["gen/translation.jsonl"], ["gen/paraphrase"]),
        row("gen/summary", ["gen/summary.jsonl"], ["src/A", "src/C"]),
        row("R1", ["releases/R1.jsonl"],
            ["src/A", "src/B", "src/C", "src/Z", "gen/paraphrase", "gen/translation", "gen/summary"]),
    ]
    partitioned = {"shard-A": ["src/A"], "shard-B": ["src/B"], "shard-C": ["src/C"],
                   "shard-Z": ["src/Z"], "shard-gen": ["gen/paraphrase", "gen/translation", "gen/summary"]}
    for shard, files in partitioned.items():
        base = f"pipeline/partitioned/{{}}/{shard}"
        rows.append(row(f"set/{shard}.input", [base.format("tokenizer-input") + ".jsonl"],
                        ["R1"] if from_release else files))
        rows.append(row(f"set/{shard}", [base.format("shards") + ".bin",
                                         base.format("shards") + ".idx"], [f"set/{shard}.input"]))
    for n in range(1, 21):
        shard = f"shard-{n}"
        base = f"pipeline/round-robin/{{}}/{shard}"
        rows.append(row(f"set-rr/{shard}.input", [base.format("tokenizer-input") + ".jsonl"],
                        ["R1"] if from_release else ["src/A", "src/B", "src/Z", "gen/paraphrase"]))
        rows.append(row(f"set-rr/{shard}", [base.format("shards") + ".bin",
                                            base.format("shards") + ".idx"],
                        [f"set-rr/{shard}.input"]))
    return rows


def make_successor_rows() -> list[dict]:
    """The manifest of the repaired side of a repair under withdraw A: a build with
    capture on under the name prefix, from the records of B and Z only."""
    return [
        row("successor of src/B", ["sources/B.jsonl"], []),
        row("successor of src/Z", ["sources/Z.jsonl"], []),
        row("successor of R1", ["releases/R1.jsonl"],
            ["successor of src/B", "successor of src/Z"]),
        row("successor of E/shard-3.input", ["pipeline/partitioned/tokenizer-input/shard-3.jsonl"],
            ["successor of src/Z"]),
        row("successor of E/shard-3", ["pipeline/partitioned/shards/shard-3.bin",
                                       "pipeline/partitioned/shards/shard-3.idx"],
            ["successor of E/shard-3.input"]),
        row("successor of set/shard-A", ["pipeline/partitioned/shards/shard-A.bin"],
            ["successor of src/B"]),
    ]


def test_the_set_size_rows_are_in_routed_groups_and_read_as_their_rows_say(read_group):
    routed = routed_groups()
    in_groups = {g["id"] for g in table(10)["group"]
                 if any(listed.startswith("c09.10k.") for listed in g["rows"])}
    assert {"c09.artifact-actions", "c09.successors", "c09.history"} <= in_groups
    assert {"c09.artifact-actions", "c09.successors", "c09.history"} <= set(routed)

    rows = make_set_manifest()
    partitioned = [r["name"] for r in rows if r["name"].startswith("set/")
                   and not r["name"].endswith(".input")]
    chunks = [r["name"] for r in rows if r["name"].startswith("set-rr/")
              and not r["name"].endswith(".input")]
    assert len(partitioned) == 5 and len(chunks) == 20

    # c09.10k.partitioned.kept-shards: exactly the shards no line of A reaches are keep
    # (by construction of the document: B, C and Z), every other shard is rewrite.
    got = read_group("c09.artifact-actions", rows, artifacts=partitioned, targets=WITHDRAW_A)
    assert got["keep"] == ["set/shard-B", "set/shard-C", "set/shard-Z"]
    assert got["rewrite"] == ["set/shard-A", "set/shard-gen"]

    # c09.10k.round-robin.all-rewrite: the table row's counts, over the 20 chunks.
    expect = table_row(9, "c09.10k.round-robin.all-rewrite")["expect"]
    got = read_group("c09.artifact-actions", rows, artifacts=chunks, targets=WITHDRAW_A)
    assert dict(got["counts"]) == {"keep": expect["keep"], "rewrite": expect["rewrite"]}
    assert got["keep"] == [] and len(got["rewrite"]) == 20


def test_chunks_that_name_the_release_they_were_cut_from_make_every_shard_contain_a(read_group):
    """A document whose chunk rows name the release file R1: R1 holds A's rows, so the walk
    reaches A from every shard and no shard is read keep. The reader reads the document it
    is given. This is not the shape a set build writes (a set's chunk rows name the set's
    own files, and the kept shards of ``c09.10k.partitioned.kept-shards`` read keep); the
    recorder on a set is tested in ``tests/replica/test_replica_built_from.py``."""
    rows = make_set_manifest(from_release=True)
    shards = [r["name"] for r in rows if r["name"].startswith(("set/", "set-rr/"))
              and not r["name"].endswith(".input")]
    assert len(shards) == 25
    got = read_group("c09.artifact-actions", rows, artifacts=shards, targets=WITHDRAW_A)
    assert got["keep"] == []
    assert sorted(got["rewrite"]) == sorted(shards)
    assert dict(got["counts"]) == {"keep": 0, "rewrite": 25}


def test_the_successor_shards_and_the_history_of_the_10k_set_read_as_their_rows_say(
        read_group, join_manifests):
    answer = table_row(9, "c09.10k.successor-shards")["expect"]["answer"]
    history = table_row(9, "c09.10k.R1-history")["expect"]["answer"]
    assert (answer, history) == (DOES_NOT_CONTAIN, CONTAINS)
    joined = join_manifests(make_set_manifest(), make_successor_rows())
    for shard in ("successor of E/shard-3", "successor of set/shard-A"):
        assert read_group("c09.successors", joined, artifact=shard, targets=WITHDRAW_A) == answer
    assert read_group("c09.history", joined, artifact="R1", targets=WITHDRAW_A) == history


# --- Two manifests joined: the historical build and the repaired one ---------------------


def test_the_historical_release_is_read_after_the_successor_rows_are_added(
        join_manifests, read_artifact):
    """c09.history reads R1 after the rows of the successor build have been added to the
    manifest; c09.fixture.successor.R1 reads the successor of R1 in the same list."""
    historical, repaired = make_manifest(), make_successor_rows()
    joined = join_manifests(historical, repaired)
    assert [dict(r) for r in joined] == historical + repaired
    history = table_row(9, "c09.fixture.R1-history")["expect"]["answer"]
    successor = table_row(9, "c09.fixture.successor.R1")["expect"]["answer"]
    assert (history, successor) == (CONTAINS, DOES_NOT_CONTAIN)
    assert read_artifact(joined, "R1", WITHDRAW_A) == history
    assert read_artifact(joined, "successor of R1", WITHDRAW_A) == successor


def test_a_join_refuses_a_name_in_two_manifests_and_changes_none(join_manifests, need_attribute):
    """The second R1 row differs from the first manifest's (it names no file of an input)."""
    error = need_attribute(MODULE, "ManifestError")
    first = make_manifest()
    second = [row("R1", ["releases/R1.jsonl"], [])]
    before = copy.deepcopy(first), copy.deepcopy(second)
    with pytest.raises(error):
        join_manifests(first, second)
    assert (first, second) == before


def test_a_join_keeps_one_row_for_a_name_in_two_manifests_when_the_rows_are_equal(
        join_manifests):
    """Joined by artifact name: a name in both manifests with the same row is one row, at
    its first place; the repaired side's own names are its own."""
    historical, repaired = make_manifest(), make_successor_rows()
    shared = copy.deepcopy([r for r in historical if r["name"] in ("src/B", "src/Z")])
    joined = join_manifests(historical, shared + repaired)
    assert [dict(r) for r in joined] == historical + repaired
    assert sorted(r["name"] for r in joined) == sorted(set(r["name"] for r in joined))


def test_a_join_with_a_manifest_that_is_absent_is_absent(join_manifests):
    """A build made with capture off holds none: the join of it with rows is no whole
    manifest, and says so with the one absent value, not with the rows of the other."""
    assert join_manifests(make_manifest(), None) is None
    assert join_manifests(None, make_manifest()) is None
    assert join_manifests([], []) == []


# --- The document is only read -----------------------------------------------------------


def test_no_document_is_changed_by_a_reading(read_artifact, mixture_lists, import_family,
                                             listing):
    rows = make_manifest()
    before = copy.deepcopy(rows)
    for r in rows:
        read_artifact(rows, r["name"], WITHDRAW_A)
    mixture_lists(rows, MAIN_MIXTURES, WITHDRAW_A)
    import_family(rows, "import/T")
    listing(rows)
    assert rows == before


# --- What the module may hold ------------------------------------------------------------


def _path(need_module) -> Path:
    module = need_module(MODULE)
    path = getattr(module, "__file__", None)
    assert path, f"{MODULE} has no file"
    return Path(path)


def test_the_module_imports_nothing_of_the_tool_the_replica_nor_the_world_helpers(need_module):
    """The reader takes nothing from the tool's store and nothing of the replica: it reads
    the file the build wrote."""
    path = _path(need_module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, sorted(roots & FORBIDDEN_IMPORT_ROOTS)


def test_the_module_holds_no_local_path_and_no_em_dash(need_module):
    text = _path(need_module).read_text(encoding="utf-8")
    for bad in FORBIDDEN_TEXT:
        assert bad not in text, f"{MODULE} holds {bad!r}"
