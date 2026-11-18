"""The Dolma companion rows, read by hand-written files (plan 5.5 Check 10; table README,
section "Check 10": the Dolma toolkit's records, "The Dolma tie to an ordinal").

What the table says, and what these tests hold the reader to:

- The tokenizer writes one row per document, in a file with no header row, five
  columns in this order: ``start, end, id, src, loc``. ``src`` is the input path as
  it was given, ``loc`` the line number counted from 1. The first line of the file is
  a row, not a header. The file may be gzip (``.csv.gz``) or plain.
- The row order of a run is not the input order. A row is tied to a document of a
  shard by its ``src`` and ``loc``, never by its place in the file: the document with
  ordinal n of a shard is tied to the row whose ``src`` is the shard's chunk file and
  whose ``loc`` is n + 1. Chunk files of two layouts share base names, so ``src`` is
  compared whole.
- An id is resolved by finding the file under ``sources/`` or ``gen/`` of the same
  build that holds a row with that id. A file under ``sources/`` is the file of a
  member. The record files only are read: not the ``<file>.manifest.jsonl`` beside
  each of them, and not the harness logs under ``gen/harness/``.
- The reader adds no check that the table does not name, but it never reads a gap
  as an absence: a row of the wrong shape is an error, and a document with no row
  is no tie (None), never its neighbour.

The names asked of ``baselines.companion``:

- ``COLUMNS``: ``("start", "end", "id", "src", "loc")``;
- ``read_rows(path)``: the rows in file order, each with ``start``, ``end`` and
  ``loc`` as integers and ``id`` and ``src`` as text (an attribute or a key);
  ValueError for a row that does not hold five fields, a ``start``, ``end`` or
  ``loc`` that is no integer, or a ``loc`` below 1;
- ``rows_for_src(rows, src)``: the rows whose ``src`` equals ``src`` whole, in the
  order given;
- ``tie_row(rows, src, ordinal)``: the row with that ``src`` and ``loc`` equal to
  ``ordinal + 1``, or None; ValueError where two rows hold that ``src`` and ``loc``;
- ``resolve_id(folder, document_id)``: the path, relative to the build folder and
  written with ``/``, of the file under ``sources/`` or ``gen/`` that holds a row
  with that id, or None;
- ``kind_of(relative_path)``: "source" for a file under ``sources/``, "generation"
  for a file under ``gen/``.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

MODULE = "baselines.companion"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")

# Two layouts share the base name of their chunk files.
LAYOUT_A = "pipeline/layout-a/tokenizer-input/shard-1.jsonl"
LAYOUT_B = "pipeline/layout-b/tokenizer-input/shard-1.jsonl"
OTHER_SHARD = "pipeline/layout-a/tokenizer-input/shard-2.jsonl"

# A hand-written companion file for the five-line chunk file LAYOUT_A, its rows in a
# shuffled order (the tokenizer's row order changes from run to run). Each row is
# (start, end, id, src, loc). The document with ordinal 0 is line 1, ordinal 2 is
# line 3, ordinal 4 is line 5.
SHUFFLED = [
    (0, 40, "B-004", LAYOUT_A, 4),
    (40, 70, "A-001", LAYOUT_A, 1),
    (70, 95, "B-005", LAYOUT_A, 5),
    (95, 120, "A-002", LAYOUT_A, 2),
    (120, 150, "B-003", LAYOUT_A, 3),
]
# Rows of other chunk files in the same companion file, with the same loc numbers.
FOREIGN = [
    (150, 170, "C-001", OTHER_SHARD, 1),
    (170, 190, "C-002", OTHER_SHARD, 2),
    (190, 210, "Z-001", LAYOUT_B, 1),
    (210, 230, "Z-002", LAYOUT_B, 3),
]


def field(obj, name):
    """A field of a row, whether the reader gives rows as objects or as mappings."""
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def write_plain(path: Path, rows) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def write_gzip(path: Path, rows) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def as_tuples(rows):
    return [
        (field(r, "start"), field(r, "end"), field(r, "id"), field(r, "src"), field(r, "loc"))
        for r in rows
    ]


# The functions are looked up when a test calls them, never at setup: a module that
# does not exist yet is a failed assertion in the body of a test, not an error.
@pytest.fixture
def read_rows(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "read_rows")(*a, **k)


@pytest.fixture
def tie_row(need_attribute):
    return lambda *a, **k: need_attribute(MODULE, "tie_row")(*a, **k)


@pytest.fixture
def companion(read_rows, tmp_path):
    """Rows as the reader gives them: a hand-written file is written and read."""
    counter = [0]

    def make(source):
        counter[0] += 1
        return read_rows(write_plain(tmp_path / f"part-{counter[0]}.csv", source))

    return make


# --- The file ----------------------------------------------------------------------------


def test_the_columns_are_the_five_the_readme_names_in_order(need_attribute):
    assert tuple(need_attribute(MODULE, "COLUMNS")) == ("start", "end", "id", "src", "loc")


def test_a_headerless_file_gives_one_row_per_line_the_first_line_included(read_rows, tmp_path):
    path = write_plain(tmp_path / "part-0.csv", SHUFFLED)
    rows = read_rows(path)
    assert len(rows) == 5
    assert as_tuples(rows)[0] == (0, 40, "B-004", LAYOUT_A, 4)


def test_the_rows_are_given_in_file_order_not_sorted(read_rows, tmp_path):
    path = write_plain(tmp_path / "part-0.csv", SHUFFLED + FOREIGN)
    assert as_tuples(read_rows(path)) == SHUFFLED + FOREIGN


def test_start_end_and_loc_are_integers_and_id_and_src_are_text(read_rows, tmp_path):
    path = write_plain(tmp_path / "part-0.csv", SHUFFLED)
    row = read_rows(path)[1]
    assert (field(row, "start"), field(row, "end"), field(row, "loc")) == (40, 70, 1)
    for number in ("start", "end", "loc"):
        assert type(field(row, number)) is int
    assert field(row, "id") == "A-001"
    assert field(row, "src") == LAYOUT_A


def test_a_gzip_file_reads_the_same_as_a_plain_one(read_rows, tmp_path):
    plain = write_plain(tmp_path / "part-0.csv", SHUFFLED + FOREIGN)
    packed = write_gzip(tmp_path / "part-0.csv.gz", SHUFFLED + FOREIGN)
    assert as_tuples(read_rows(packed)) == as_tuples(read_rows(plain)) == SHUFFLED + FOREIGN


def test_a_quoted_field_with_a_comma_is_one_field(read_rows, tmp_path):
    rows = [(0, 9, "odd,id", "pipeline/layout-a/tokenizer-input/shard-1.jsonl", 1)]
    path = write_plain(tmp_path / "part-0.csv", rows)
    assert as_tuples(read_rows(path)) == rows


def test_a_file_with_no_row_gives_no_row(read_rows, tmp_path):
    path = tmp_path / "part-0.csv"
    path.write_text("", encoding="utf-8")
    assert list(read_rows(path)) == []


def test_a_row_with_four_fields_is_an_error_not_a_skipped_row(read_rows, tmp_path):
    path = tmp_path / "part-0.csv"
    path.write_text("0,40,A-001,pipeline/x.jsonl,1\n40,70,A-002,pipeline/x.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_rows(path)


def test_a_row_with_six_fields_is_an_error(read_rows, tmp_path):
    path = tmp_path / "part-0.csv"
    path.write_text("0,40,A-001,pipeline/x.jsonl,1,extra\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_rows(path)


@pytest.mark.parametrize("bad", ["0,40,A-001,pipeline/x.jsonl,one\n",
                                 "zero,40,A-001,pipeline/x.jsonl,1\n",
                                 "0,forty,A-001,pipeline/x.jsonl,1\n"])
def test_a_number_that_is_no_integer_is_an_error(read_rows, tmp_path, bad):
    path = tmp_path / "part-0.csv"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        read_rows(path)


def test_loc_counts_from_one_so_a_loc_of_zero_is_an_error(read_rows, tmp_path):
    path = tmp_path / "part-0.csv"
    path.write_text("0,40,A-001,pipeline/x.jsonl,0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_rows(path)


# --- The rows of one chunk file ----------------------------------------------------------


def test_rows_for_a_src_keep_the_rows_of_that_whole_path_only(need_attribute, companion):
    rows_for_src = need_attribute(MODULE, "rows_for_src")
    rows = companion(SHUFFLED + FOREIGN)
    mine = rows_for_src(rows, LAYOUT_A)
    assert [field(r, "id") for r in mine] == ["B-004", "A-001", "B-005", "A-002", "B-003"]
    # The same base name in the other layout is another file.
    other = rows_for_src(rows, LAYOUT_B)
    assert [field(r, "id") for r in other] == ["Z-001", "Z-002"]
    assert list(rows_for_src(rows, "pipeline/layout-a/tokenizer-input/shard-9.jsonl")) == []


# --- The tie by src and loc --------------------------------------------------------------


def test_the_document_with_ordinal_n_is_the_row_with_loc_n_plus_one(tie_row, companion):
    rows = companion(SHUFFLED + FOREIGN)
    assert field(tie_row(rows, LAYOUT_A, 0), "id") == "A-001"
    assert field(tie_row(rows, LAYOUT_A, 1), "id") == "A-002"
    assert field(tie_row(rows, LAYOUT_A, 2), "id") == "B-003"
    assert field(tie_row(rows, LAYOUT_A, 3), "id") == "B-004"
    assert field(tie_row(rows, LAYOUT_A, 4), "id") == "B-005"


def test_the_tie_is_not_the_place_of_the_row_in_the_file(tie_row, companion):
    """Red example: a tie by row position. Ordinal 0 sits at place 1 of the file and
    place 0 holds the row of line 4; a tie by place would answer B-004 for ordinal 0
    and A-001 for ordinal 1."""
    rows = companion(SHUFFLED)
    assert field(tie_row(rows, LAYOUT_A, 0), "id") != "B-004"
    assert field(tie_row(rows, LAYOUT_A, 1), "id") != "A-001"
    # Whatever the order of the same rows, the tie gives the same row.
    reordered = companion(list(reversed(SHUFFLED)))
    for ordinal in range(5):
        assert field(tie_row(reordered, LAYOUT_A, ordinal), "id") == field(
            tie_row(rows, LAYOUT_A, ordinal), "id")


def test_the_tie_compares_src_whole_so_two_layouts_do_not_mix(tie_row, companion):
    rows = companion(SHUFFLED + FOREIGN)
    assert field(tie_row(rows, LAYOUT_B, 0), "id") == "Z-001"
    assert field(tie_row(rows, OTHER_SHARD, 0), "id") == "C-001"
    assert field(tie_row(rows, OTHER_SHARD, 1), "id") == "C-002"


def test_a_document_with_no_row_is_no_tie_not_its_neighbour(tie_row, companion):
    """Line 2 of LAYOUT_B has no row (a deleted row): ordinal 1 ties to nothing, and
    ordinal 2 (loc 3) still ties to its own row."""
    rows = companion(SHUFFLED + FOREIGN)
    assert tie_row(rows, LAYOUT_B, 1) is None
    assert field(tie_row(rows, LAYOUT_B, 2), "id") == "Z-002"
    # Past the end of the file.
    assert tie_row(rows, LAYOUT_A, 5) is None
    # A chunk file the companion rows do not name.
    assert tie_row(rows, "pipeline/layout-a/tokenizer-input/shard-9.jsonl", 0) is None


def test_the_tie_gives_the_whole_row_with_its_token_range(tie_row, companion):
    rows = companion(SHUFFLED)
    row = tie_row(rows, LAYOUT_A, 3)
    assert (field(row, "start"), field(row, "end")) == (0, 40)


def test_two_rows_with_one_src_and_loc_are_an_error_not_a_pick(tie_row, companion):
    rows = companion(SHUFFLED + [(300, 320, "A-001-again", LAYOUT_A, 1)])
    with pytest.raises(ValueError):
        tie_row(rows, LAYOUT_A, 0)
    # The other documents still tie.
    assert field(tie_row(rows, LAYOUT_A, 1), "id") == "A-002"


# --- The id resolver ---------------------------------------------------------------------


def _write_jsonl(path: Path, ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "text": f"text of {i}"}) + "\n")


def _build(tmp_path: Path) -> Path:
    build = tmp_path / "build"
    _write_jsonl(build / "sources" / "A.jsonl", ["A-001", "A-002"])
    _write_jsonl(build / "sources" / "B.jsonl", ["B-003", "B-004", "B-005"])
    _write_jsonl(build / "gen" / "P.jsonl", ["P-001", "P-002"])
    # The files beside the record files: not records of the build's files.
    _write_jsonl(build / "sources" / "A.jsonl.manifest.jsonl", ["M-001"])
    _write_jsonl(build / "gen" / "harness" / "P" / "job-1.log.jsonl", ["H-001", "A-001"])
    return build


def test_an_id_resolves_to_the_file_of_a_member_under_sources(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = _build(tmp_path)
    assert resolve_id(build, "A-002") == "sources/A.jsonl"
    assert resolve_id(build, "B-005") == "sources/B.jsonl"


def test_an_id_resolves_to_a_generation_output_under_gen(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = _build(tmp_path)
    assert resolve_id(build, "P-002") == "gen/P.jsonl"


def test_an_id_no_file_holds_resolves_to_none(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = _build(tmp_path)
    assert resolve_id(build, "Q-999") is None
    # Whole ids only: a prefix of an id is no id.
    assert resolve_id(build, "A-00") is None
    assert resolve_id(build, "") is None


def test_the_manifest_files_and_the_harness_logs_are_not_read(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = _build(tmp_path)
    assert resolve_id(build, "M-001") is None
    assert resolve_id(build, "H-001") is None
    # A-001 is also named in a harness log; the file that holds it is the source file.
    assert resolve_id(build, "A-001") == "sources/A.jsonl"


def test_a_build_with_no_gen_folder_still_resolves_its_sources(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = tmp_path / "build"
    _write_jsonl(build / "sources" / "A.jsonl", ["A-001"])
    assert resolve_id(build, "A-001") == "sources/A.jsonl"
    assert resolve_id(build, "P-001") is None


def test_the_resolver_accepts_a_text_path_for_the_folder(need_attribute, tmp_path):
    resolve_id = need_attribute(MODULE, "resolve_id")
    build = _build(tmp_path)
    assert resolve_id(str(build), "P-001") == "gen/P.jsonl"


def test_the_kind_of_a_file_is_source_under_sources_and_generation_under_gen(need_attribute):
    kind_of = need_attribute(MODULE, "kind_of")
    assert kind_of("sources/A.jsonl") == "source"
    assert kind_of("sources/Z.jsonl") == "source"
    assert kind_of("gen/P.jsonl") == "generation"


# --- What the module may hold ------------------------------------------------------------


def _path(need_module) -> Path:
    module = need_module(MODULE)
    path = getattr(module, "__file__", None)
    assert path, f"{MODULE} has no file"
    return Path(path)


def test_the_module_imports_nothing_of_the_tool_nor_of_the_world_helpers(need_module):
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
