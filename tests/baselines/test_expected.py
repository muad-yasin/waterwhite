"""The expected answers of the listed rows (plan 5.5 Check 10, table README section
"Grades, and what is compared"): ``baselines.expected``.

Every expected value in this file is read from the table files with ``tomllib``
(``check01.toml`` to ``check09.toml``, the fixture spec) or is written by hand below,
in the shape the counting script (``fixture/plant/count.py``) prints. Nothing is
taken from running the code under test, and no set is generated and no counting
script is run.

The names this file asks of ``baselines.expected``:

- ``expected_for(listed_row, size=None, *, fallbacks=(), spec=None, counted=None)``
  returns an ``Expected`` for one listed row of Checks 1 to 9 asked at ``size`` (None
  for a row asked once; the sizes are the tool's own, as in ``conftest.py``). It holds
  ``listed_row``, ``size``, ``applies`` (False when the fallback in force says the row
  does not apply: that row is not put), ``expect`` (the row's ``expect``, or the entry
  of ``expect_under`` for the fallback in force; every ``pending:<key>`` replaced by
  that item of the spec; an empty mapping when the row does not apply), ``unfilled``
  (the pending keys whose spec item is missing or empty, as a tuple; such a value is
  left as the table wrote it) and ``counted`` (the document a counting script printed
  for the set of that size, or None).
  ``spec`` is a mapping (default: the fixture spec). ``counted`` is None or a mapping
  from a size to a counting document; a sized row reads the document of its own size.
  A listed row the table does not hold, and a row of an extension that is off, raise
  ``KeyError``; a size the row is not asked at raises ``ValueError``. A size the mapping
  holds no document for is none held: nothing is raised, and the value reads "expected value
  not derived".
- ``run_rows(oracle, cuts=None)``: the rows the join must report for an oracle log,
  as a mapping from (segment, step, sample_pos) to a list of (shard position,
  ordinal, token start, token end); ``cuts`` maps a sample to the tokens read of it.
- ``same_shard_pairs(counted)``, ``split_pairs(counted)``, ``records_by_ordinal(counted)``
  and ``record_action_counts(counted)``: parts of a counting document (the shape of
  ``count.py``'s output).
- There is no ``planted_values``: the counted values of every row reach a grade through
  ``concrete_values`` and ``concrete_values_by_layout`` only (see the section on the rows
  the retired reader mapped). A part the counting document does not hold (no chunk files,
  no sanitizer input) is ``PartMissing``, a ``ValueError`` that names the part; it is never
  read as "none". No counting document at all is not that error: where a row's source names
  a counting document and none is given, ``concrete_values`` returns the row's ``Concrete``
  with the keys the document would give in ``open`` (read "expected value not derived"),
  the table's own words in ``expect`` (never compared) and nothing beside them.
- ``c06.generated.documents-mapped`` also holds, in ``counted_values`` (beside ``expect``,
  with the other counted values of a row), three per-shard counts that the row's rule names
  and the grader compares, derived from the ``documents`` field of each shard of the
  document's ``documents`` part: ``documents_per_shard``, ``documents_with_one_tied_row_per_shard``
  (the same number: every document of a shard has exactly one tied row, a member of a group of
  identical documents having its own companion row tied by its own src and loc) and
  ``documents_with_no_tied_row_per_shard`` (0 per shard, as the row's ``unmapped_documents``).
  A shard whose document count is not given leaves the three out: they read "expected value
  not derived". The row stays a one-id reading.
- ``index_of_shards(files)`` reads the shard index files the replica writes (``files`` maps
  a shard name to the path of its ``.idx`` file) with the replica's own reader and returns
  ``{shard: [size of each document, in ordinal order]}``, the form ``expected_for`` takes as
  ``index``. A file that is not a shard index is an error naming the file.
- ``SENTENCE_FIELDS`` is the one public source of the fields whose table value is a
  sentence (listed row to its keys; 19 rows, 23 row and key pairs) and ``ONE_ID_ROWS`` the
  one public name of the rows whose answer is one id per ordinal. A module that grades
  imports these two and no name that starts with an underscore.
- ``Concrete`` holds, beside ``listed_row``, ``size``, ``layout``, ``expect`` and ``open``,
  ``counted_values`` (counted values kept apart from ``expect``) and ``not_compared`` (the
  keys of ``expect`` that are one of the table's not-compared parts, README section "Grades,
  and what is compared"). ``open`` holds only a field read "expected value not derived".
- ``one_id_reading(listed_row, per_ordinal, given)`` compares one id per ordinal with the
  carried expectation (an id, or the list of ids of a group of identical documents): it
  gives ``right``, ``differing`` (the (shard, ordinal) pairs that differ) and ``compared``.
  At an ordinal of a group, one id is right when it is a member of the group, and a list
  is compared as a set: right when it equals the group's ids, any other list wrong.
- ``combine_layouts(listed_row, results)`` gives the one line of a row asked in both layouts,
  from a result per layout: a ``Grade``, the ``ValueError`` a layout's expected value could
  not be derived with, the ``Concrete`` of a layout whose counting document is missing (its
  ``open`` is not empty: read "expected value not derived"), None or an absent layout
  (that layout is not attempted), or one of the two line statuses of an attempt
  (``"not attempted"`` and ``"no answer: failed attempt <id>"``).
  The line is, in this order: wrong if any layout is wrong; the status of a failed attempt
  if any layout's attempt failed (the first one in the row's order); not attempted if any
  layout was not attempted; right if all are right; otherwise cannot answer, naming the
  layout. A ``NotGradable`` is not a layout result.

None of these three modules imports the tool or the acceptance modules
that build a world.
"""

from __future__ import annotations

import ast
import importlib
import json
import struct
import tomllib
from functools import lru_cache
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    found = getattr(need(module_name), attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _read_toml(path: Path) -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def _spec() -> dict:
    """The fixture spec as the file writes it."""
    return _read_toml(SPEC_FILE)


@lru_cache(maxsize=1)
def _listed_rows() -> dict[str, dict]:
    """Every core row of Checks 1 to 9 as the file writes it, by id."""
    rows: dict[str, dict] = {}
    for number in range(1, 10):
        for row in _read_toml(TABLE_DIR / f"check{number:02d}.toml")["row"]:
            if row["core"]:
                rows[row["id"]] = row
    return rows

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")


# --- The reading of the table, written out here ------------------------------------------


def _fill(value, spec):
    """A value with each ``pending:<key>`` replaced by the spec item at that dotted
    key, the reading of the table README (section 5); written here, not imported."""
    if isinstance(value, str) and value.startswith("pending:"):
        found = spec
        for part in value[len("pending:"):].split("."):
            found = found[part]
        return found
    if isinstance(value, dict):
        return {k: _fill(v, spec) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, spec) for v in value]
    return value


def _written_expect(listed_row: str, fallback: str | None = None):
    row = _listed_rows()[listed_row]
    if fallback is not None and fallback in row.get("expect_under", {}):
        return row["expect_under"][fallback]
    return row["expect"]


# --- The pending values and the fallbacks -------------------------------------------------


def test_the_eod_convention_is_read_from_the_spec_item_the_row_names():
    expected_for = need_attr("baselines.expected", "expected_for")
    got = expected_for("c03.precondition.eod-convention")
    assert got.listed_row == "c03.precondition.eod-convention"
    assert got.size is None
    assert got.applies is True
    assert got.unfilled == ()
    assert dict(got.expect) == {
        "spec_key": "tokenizer.eod_in_span",
        "value": _spec()["tokenizer"]["eod_in_span"],
    }


def test_the_positions_of_a_row_are_the_spec_table_it_names_and_the_rest_is_kept():
    expected_for = need_attr("baselines.expected", "expected_for")
    got = expected_for("c01.E.T")
    written = _written_expect("c01.E.T")
    assert got.expect["rows"] == _spec()["positions"]["T"]
    assert got.expect["diff"] == "empty"
    assert got.expect["row"] == written["row"]
    assert got.expect["rows_in_sample"] == 3
    assert set(got.expect) == set(written)


def test_a_spec_without_the_item_leaves_the_value_as_written_and_names_the_key():
    expected_for = need_attr("baselines.expected", "expected_for")
    got = expected_for("c03.precondition.eod-convention", spec={})
    assert got.unfilled == ("tokenizer.eod_in_span",)
    assert got.expect["value"] == "pending:tokenizer.eod_in_span"
    # An item that is empty counts as missing, as the table reader treats it.
    got = expected_for("c03.precondition.eod-convention", spec={"tokenizer": {"eod_in_span": ""}})
    assert got.unfilled == ("tokenizer.eod_in_span",)


def test_a_fallback_in_force_gives_the_expect_under_entry_of_that_fallback():
    expected_for = need_attr("baselines.expected", "expected_for")
    row = "c03.drops.sanitizer"
    plain = expected_for(row, "small")
    under = expected_for(row, "small", fallbacks=("F1",))
    assert dict(plain.expect) == _listed_rows()[row]["expect"]
    assert dict(under.expect) == _listed_rows()[row]["expect_under"]["F1"]
    assert under.expect["map_built_by"] == "token-matching"
    assert "map_built_by" not in plain.expect
    # A fallback the row has no entry for changes nothing.
    assert dict(expected_for(row, "small", fallbacks=("F3",)).expect) == dict(plain.expect)
    # The positions rows read the oracle's golden log under F5 and keep their pending
    # value filled from the spec.
    f5 = expected_for("c01.E.T", fallbacks=("F5",))
    assert f5.expect["judged_against"] == "the golden log from another machine"
    assert f5.expect["rows"] == _spec()["positions"]["T"]


def test_a_row_that_does_not_apply_under_the_fallback_in_force_is_not_put():
    expected_for = need_attr("baselines.expected", "expected_for")
    row = "c03.drops.no-tokens.planted-line"
    assert _listed_rows()[row]["expect_under"]["F1"] == {"applies": False}
    got = expected_for(row, fallbacks=("F1",))
    assert got.applies is False
    assert dict(got.expect) == {}
    assert expected_for(row).applies is True


def test_every_listed_row_at_every_size_gives_the_table_expect_with_pending_filled(facts):
    expected_for = need_attr("baselines.expected", "expected_for")
    spec = _spec()
    rows = _listed_rows()
    assert len(rows) == facts.core_rows_count == 370
    seen = 0
    for listed, sizes in facts.sizes_by_listed.items():
        for size in sizes:
            got = expected_for(listed, size)
            assert got.listed_row == listed and got.size == size, (listed, size)
            assert got.applies is True and got.unfilled == (), (listed, size)
            assert dict(got.expect) == _fill(rows[listed]["expect"], spec), (listed, size)
            seen += 1
    assert seen == facts.questions_per_subject == 391
    # Under every fallback a row names, the entry of that fallback is the expect.
    for listed, row in rows.items():
        for fallback, entry in row.get("expect_under", {}).items():
            size = facts.sizes_by_listed[listed][0]
            got = expected_for(listed, size, fallbacks=(fallback,))
            if entry == {"applies": False}:
                assert got.applies is False, (listed, fallback)
            else:
                assert dict(got.expect) == _fill(entry, spec), (listed, fallback)


# --- The sizes ---------------------------------------------------------------------------


def test_a_size_the_row_is_not_asked_at_is_refused(facts):
    expected_for = need_attr("baselines.expected", "expected_for")
    # The row that names the 100k set and is asked once takes no size.
    with pytest.raises(ValueError):
        expected_for(facts.eod_convention, "100k")
    # A row asked at sizes needs one of them.
    sized = next(l for l, s in facts.sizes_by_listed.items() if s != (None,))
    with pytest.raises(ValueError):
        expected_for(sized)
    with pytest.raises(ValueError):
        expected_for(sized, "10k" if "10k" not in facts.sizes_by_listed[sized] else "nonsense")
    # A c09.10k row is asked once, at 10k by its own id, not at a size.
    ten_k = next(l for l in facts.sizes_by_listed if l.startswith("c09.10k."))
    assert facts.sizes_by_listed[ten_k] == (None,)
    with pytest.raises(ValueError):
        expected_for(ten_k, "10k")
    assert expected_for(ten_k).size is None


def test_a_row_the_table_does_not_hold_is_refused_and_a_row_of_an_extension_a_group_lists_is_read(facts):
    """Changed 18 Nov 2026 (old: "an extension row is refused"): the pass asks a row of an extension only while it is on
    (``table10.expand``), so the expected value of such a row is read like any other row's; a row no group lists is never asked."""
    expected_for = need_attr("baselines.expected", "expected_for")
    with pytest.raises(KeyError):
        expected_for("c99.no-such-row")
    assert facts.extension_ids, "the table has extension rows"
    listed = sorted({row for group in facts.all_groups for row in group["rows"]} & set(facts.extension_ids))
    assert len(listed) == 9, listed
    for row in listed:
        assert expected_for(row).listed_row == row


# --- Per size: each size reads the counting document of its own set -----------------------


def _pair(pair_id, placement, members):
    return {"pair": pair_id, "kind": "pair-member", "placement": placement, "members": members}


def _member(record, shard, line):
    return {
        "id": record, "source": "Z", "shard": shard, "line": line, "ordinal": line - 1,
        "expected_label": "candidates",
    }


SMALL_COUNTED = {
    "check3": {"pairs": {"list": [
        _pair("pair-1", "same-shard", [_member("Z-101", "shard-1", 3), _member("Z-102", "shard-1", 9)]),
        _pair("pair-2", "split", [_member("Z-201", "shard-1", 5), _member("Z-202", "shard-2", 2)]),
    ]}},
    "documents": {"total": 5, "shards": [
        {"shard": "shard-1", "documents": 3, "ids": ["Z-001", "Z-002", "Z-003"],
         "identical": [], "labels": {"recovered": 3, "candidates": 0}},
        {"shard": "shard-2", "documents": 2, "ids": ["Z-004", "Z-005"],
         "identical": [], "labels": {"recovered": 2, "candidates": 0}},
    ]},
    "actions": {"records": {"keep": 9, "re-source": 1, "promote": 1, "retire": 4}},
}

BIG_COUNTED = {
    "check3": {"pairs": {"list": [
        _pair("pair-7", "same-shard", [_member("Z-701", "shard-4", 12), _member("Z-702", "shard-4", 40)]),
        _pair("pair-8", "same-shard", [_member("Z-801", "shard-6", 2), _member("Z-802", "shard-6", 3)]),
        _pair("pair-9", "split", [_member("Z-901", "shard-4", 7), _member("Z-902", "shard-5", 7)]),
        _pair("pair-10", "single", [_member("Z-1001", "shard-5", 1)]),
    ]}},
    "documents": {"total": 4, "shards": [
        {"shard": "shard-4", "documents": 4, "ids": ["Z-010", "Z-011", "Z-012", "Z-013"],
         "identical": [], "labels": {"recovered": 4, "candidates": 0}},
    ]},
    "actions": {"records": {"keep": 90, "re-source": 10, "promote": 10, "retire": 40}},
}


def test_each_size_reads_the_counting_document_of_its_own_set(facts):
    expected_for = need_attr("baselines.expected", "expected_for")
    listed = "c03.pairs.same-shard"
    assert facts.sizes_by_listed[listed] == ("small", "100k")
    counted = {"small": SMALL_COUNTED, "100k": BIG_COUNTED}
    small = expected_for(listed, "small", counted=counted)
    big = expected_for(listed, "100k", counted=counted)
    assert small.counted is SMALL_COUNTED or small.counted == SMALL_COUNTED
    assert big.counted is BIG_COUNTED or big.counted == BIG_COUNTED
    assert small.counted != big.counted
    # The same row, size and spec otherwise: only the document differs.
    assert dict(small.expect) == dict(big.expect) == _listed_rows()[listed]["expect"]
    # No document given: none held.
    assert expected_for(listed, "small").counted is None
    # A size the mapping holds no document for is none held (the value reads "expected value
    # not derived"): nothing is raised, and no other size's document is read in its place.
    missing = _without_key_error(expected_for, listed, "100k", counted={"small": SMALL_COUNTED})
    assert missing.counted is None


def test_the_pairs_are_read_from_the_counting_document_by_placement():
    same_shard_pairs = need_attr("baselines.expected", "same_shard_pairs")
    split_pairs = need_attr("baselines.expected", "split_pairs")
    got = same_shard_pairs(SMALL_COUNTED)
    assert set(got) == {"shard-1"}
    assert [p["pair"] for p in got["shard-1"]] == ["pair-1"]
    assert [m["line"] for m in got["shard-1"][0]["members"]] == [3, 9]
    big = same_shard_pairs(BIG_COUNTED)
    assert {s: [p["pair"] for p in ps] for s, ps in big.items()} == {
        "shard-4": ["pair-7"], "shard-6": ["pair-8"],
    }
    assert [p["pair"] for p in split_pairs(SMALL_COUNTED)] == ["pair-2"]
    # A pair with one member placed, or a split pair, is not a same-shard pair.
    assert [p["pair"] for p in split_pairs(BIG_COUNTED)] == ["pair-9"]
    assert "shard-5" not in big


def test_the_record_of_each_ordinal_is_the_id_of_the_chunk_line_by_shard():
    records_by_ordinal = need_attr("baselines.expected", "records_by_ordinal")
    assert records_by_ordinal(SMALL_COUNTED) == {
        "shard-1": ["Z-001", "Z-002", "Z-003"],
        "shard-2": ["Z-004", "Z-005"],
    }
    assert records_by_ordinal(BIG_COUNTED) == {"shard-4": ["Z-010", "Z-011", "Z-012", "Z-013"]}


def test_the_record_action_counts_are_the_counting_documents_counts():
    record_action_counts = need_attr("baselines.expected", "record_action_counts")
    assert dict(record_action_counts(SMALL_COUNTED)) == {
        "keep": 9, "re-source": 1, "promote": 1, "retire": 4,
    }
    assert dict(record_action_counts(BIG_COUNTED))["retire"] == 40


# --- The plant list through the counting script: what each row reads of the document -------

PLANTED_PAIRS = [
    _pair("pair-1", "same-shard", [_member("Z-101", "shard-1", 3), _member("Z-102", "shard-1", 9)]),
    _pair("pair-2", "split", [_member("Z-201", "shard-1", 5), _member("Z-202", "shard-2", 2)]),
    _pair("pair-3", "same-shard", [_member("Z-301", "shard-2", 1), _member("Z-302", "shard-2", 4)]),
]


def _drop(record, reason, shard, input_line, next_ordinal):
    return {
        "id": record, "source": "Z", "line": 7, "case": "planted", "reason": reason,
        "raw_sha256": "00" * 32, "shard": shard, "input_line": input_line,
        "ordinal_of_next_line": next_ordinal,
    }


PLANTED = {
    "format": "waterwhite.plant-count/4",
    "documents": {"total": 5, "shards": [
        {"shard": "shard-1", "documents": 3, "ids": ["Z-001", "Z-002", "Z-003"],
         "identical": [], "labels": {"recovered": 3, "candidates": 0}},
        {"shard": "shard-2", "documents": 2, "ids": ["Z-004", "Z-005"],
         "identical": [], "labels": {"recovered": 2, "candidates": 0}},
    ]},
    "check3": {
        "pairs": {"required": True, "counts": {}, "list": PLANTED_PAIRS},
        "drops": {"count": 3, "by_reason": {"sanitizer skip": 2, "empty line": 1}, "shards": [
            {"shard": "shard-1", "input_lines": 5, "surviving_lines": 3, "drops": [
                _drop("Z-900", "sanitizer skip", "shard-1", 2, 1),
                _drop("Z-901", "empty line", "shard-1", 5, 3),
            ]},
            {"shard": "shard-2", "input_lines": 3, "surviving_lines": 2, "drops": [
                _drop("Z-902", "sanitizer skip", "shard-2", 1, 0),
            ]},
        ]},
    },
    "check6": {
        "candidates": [
            {"release": "R1", "pair": "pair-x", "text_sha256": "11" * 32, "rows": ["A-1", "B-1"],
             "label": "candidates",
             "candidates": [{"id": "A-1", "source": "A"}, {"id": "B-1", "source": "B"}],
             "sources": ["A", "B"], "may_contain_for": ["A", "B"],
             "withdrawn_source_among_candidates": True},
        ],
        "translations": {
            "count": 2, "ids": "T-000001..T-000002", "file": "gen/translation", "job": "translation",
            "releases": {"R1": 2, "R2": 0},
            "list": [], "shards": [{"shard": "shard-1", "translations": 2}],
            "shards_without": ["shard-2"],
        },
    },
}

# A document made without chunk files and without sanitizer input: the parts are null.
NO_CHUNKS_NO_DROPS = {
    "documents": None,
    "check3": {"pairs": None, "drops": None},
    "check6": {"candidates": [], "translations": {"count": 0, "releases": {}, "shards": None}},
}


# --- The rows the first reading left out: pair kinds, no-token drops, Check 9 on 10k ----

# Hand-written in the shape of the plant script's document (plant document, section 7.2).
# The pair counts are given for two kinds so that a reader of the wrong kind is caught.
PAIR_KINDS_DOC = {
    "check3": {"pairs": {"required": True, "list": [], "counts": {
        "pair-member": {"same-shard": 4, "split": 3, "single": 1, "absent": 0, "multiple": 0},
        "x-copy": {"same-shard": 9, "split": 8, "single": 0, "absent": 0, "multiple": 0},
    }}},
}

# The plant script plants no surviving line that gives no tokens (plant document,
# section 8); this document does, once, so that the count is not vacuously zero.
NO_TOKENS_DOC = json.loads(json.dumps(PLANTED))
NO_TOKENS_DOC["check3"]["drops"]["shards"][1]["drops"].append(
    _drop("Z-903", "no tokens", "shard-2", 3, 2)
)

TEN_K_DOC = {
    "actions": {"records": {"keep": 90, "re-source": 10, "promote": 10, "retire": 40, "none": 7},
                "shards": {"keep": 3, "rewrite": 2}},
    "releases": {
        "R1": {"actions": {"keep": 50, "re-source": 6, "promote": 4, "retire": 25, "none": 4},
               "removed": 25},
        "R2": {"actions": {"keep": 40, "re-source": 4, "promote": 6, "retire": 15, "none": 3},
               "removed": 15},
    },
    "shards": {"layout": "partitioned", "releases": ["R1", "R2"], "lines": 100,
               "keep": ["shard-02", "shard-07", "shard-11"], "rewrite": ["shard-01", "shard-05"],
               "all_rewrite": False, "actions": {"keep": 3, "rewrite": 2}},
}

_RR_SHARDS = [f"shard-{n:02d}" for n in range(1, 21)]
TEN_K_RR_DOC = dict(TEN_K_DOC, shards={
    "layout": "round-robin", "releases": ["R1", "R2"], "lines": 100, "keep": [],
    "rewrite": _RR_SHARDS, "all_rewrite": True, "actions": {"keep": 0, "rewrite": 20},
})

NO_SHARDS_DOC = dict(TEN_K_DOC, shards=None)


# --- The rows a retired reader mapped, now on the concrete path ----------------------------
#
# The reader of the counted values by row (``planted_values`` and its ``Planted``) is retired:
# the counted values of a row reach a grade through ``concrete_values`` and
# ``concrete_values_by_layout`` only. What that reader mapped and the concrete path did not
# is carried by ``Concrete.counted_values``, a mapping kept apart from ``expect``:
#
# - a key the row's expectation holds as a sentence is replaced in ``expect`` as before and
#   is not repeated in ``counted_values``;
# - a counted value under a key the expectation does not hold, or holds as a number or a
#   word the table states itself, sits only in ``counted_values``, so no field the row's own
#   expectation does not carry is ever compared, and a counted number never replaces the
#   table's own;
# - every other row holds an empty mapping.
#
# The mappings that moved, as (row, key, old route, new route):
#
# - c03.pairs.same-shard.swap, candidate_set: the reader's same-shard list; now
#   ``counted_values["candidate_set"]`` (``expect`` keeps the table's word "unchanged");
# - c06.generated.downstream-of-translation, translations: the reader's own key; now
#   ``counted_values["translations"]``;
# - c03.precondition.pair-kinds, same_shard_pairs and split_pairs: the counts of the kind
#   pair-member; now ``counted_values`` (``expect`` keeps the table's "at least" numbers);
# - c03.drops.no-tokens, drops_with_reason_no_tokens, documents_per_shard and
#   surviving_lines_per_shard: now ``counted_values`` (``expect`` keeps the table's 0);
# - c06.generated.drops, documents_per_shard: now ``counted_values``;
# - c09.10k.partitioned.kept-shards, rewrite, and c09.10k.round-robin.all-rewrite, keep and
#   rewrite: now ``counted_values``;
# - c09.10k.action-counts, shard_actions: now ``expect["counts"]["shards"]``;
# - c06.generated.documents-mapped, documents_per_shard, documents_with_one_tied_row_per_shard
#   and documents_with_no_tied_row_per_shard: ``counted_values``. The row's rule names them, so
#   the grader compares them; the table's own ``unmapped_documents`` stays in ``expect``.


def _beside(concrete):
    """The counted values a row reads beside its expectation."""
    assert hasattr(concrete, "counted_values"), "Concrete has no counted_values"
    return dict(concrete.counted_values)


def _not_compared(concrete):
    """The keys of the expectation that are one of the table's not-compared parts."""
    assert hasattr(concrete, "not_compared"), "Concrete has no not_compared"
    return tuple(concrete.not_compared)


def test_the_retired_reader_of_the_counted_values_and_its_result_type_are_gone():
    module = need("baselines.expected")
    assert not hasattr(module, "planted_values")
    assert not hasattr(module, "Planted")
    # The concrete path is the one route: it answers for the rows the old reader answered.
    need_attr("baselines.expected", "concrete_values")
    need_attr("baselines.expected", "concrete_values_by_layout")


def test_a_row_whose_values_all_sit_in_its_expectation_holds_no_counted_value_beside_it():
    counted = {SIZED_SMALL: PLANTED}
    for listed in (
        "c03.records.capture", "c03.records.position-recovery", "c03.records.token-matching",
        "c03.drops.sanitizer", "c03.pairs.same-shard", "c03.pairs.split",
        "c06.generated.candidates",
    ):
        got = _concrete(listed, SIZED_SMALL, counted=counted)
        assert _beside(got) == {}, listed
        assert _not_compared(got) == (), listed
    # The documents-mapped row is not in this list: the three counts its rule names sit beside
    # its expectation (the test of its counts below).
    mapped = _concrete("c06.generated.documents-mapped", SIZED_SMALL, counted=counted)
    assert _not_compared(mapped) == ()
    # The documents of each shard of the sanitizer row are in its expectation, not beside it.
    sanitizer = _concrete("c03.drops.sanitizer", SIZED_SMALL, counted=counted)
    assert sanitizer.expect["documents_per_shard"] == {"shard-1": 3, "shard-2": 2}
    for layout, concrete in _by_layout(
        "c09.10k.action-counts", {"partitioned": PART10, "round-robin": ROUND_ROBIN10}
    ).items():
        assert _beside(concrete) == {}, layout


def test_the_generated_drops_row_holds_the_documents_of_each_shard_beside_its_expectation():
    # The table's expectation of this row has no key for them (its source asks for "per
    # shard the number of surviving lines"); the counted value is kept, and not compared.
    assert "documents_per_shard" not in _written_expect("c06.generated.drops")
    got = _concrete("c06.generated.drops", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert "documents_per_shard" not in got.expect
    assert _beside(got) == {"documents_per_shard": {"shard-1": 3, "shard-2": 2}}


def test_the_swap_row_reads_the_same_shard_pairs_beside_the_word_it_expects():
    listed = "c03.pairs.same-shard.swap"
    written = _fill(_written_expect(listed), _spec())
    assert written["candidate_set"] == "unchanged"
    got = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    # The word is a word of the table and stays; it is not a 20th sentence-valued row.
    assert dict(got.expect) == written
    assert got.open == ()
    assert _beside(got) == {"candidate_set": [
        {"pair": "pair-1", "shard": "shard-1", "lines": [3, 9]},
        {"pair": "pair-3", "shard": "shard-2", "lines": [1, 4]},
    ]}
    # The row does not apply under F1: it is not put.
    assert _listed_rows()[listed]["expect_under"]["F1"] == {"applies": False}
    with pytest.raises(ValueError):
        _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: PLANTED}, fallbacks=("F1",))
    # The row that does apply under F1 keeps its concrete values with it.
    assert _concrete("c03.records.capture", SIZED_SMALL, counted={SIZED_SMALL: PLANTED},
                     fallbacks=("F1",)).expect["map_built_by"] == "token-matching"


def test_the_translation_row_holds_the_releases_and_shards_that_receive_a_translation_beside_its_answer():
    listed = "c06.generated.downstream-of-translation"
    got = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert dict(got.expect) == _fill(_written_expect(listed), _spec())
    assert _beside(got) == {"translations": {
        "file": "gen/translation", "releases": ["R1"], "shards": ["shard-1"],
    }}


def test_the_pair_kinds_row_counts_the_pairs_of_the_kind_pair_member_by_placement():
    listed = "c03.precondition.pair-kinds"
    written = _fill(_written_expect(listed), _spec())
    got = _concrete(listed, "100k", counted={"100k": PAIR_KINDS_DOC})
    # The table's own "at least" numbers stay the expectation; the counts of the kind
    # pair-member are beside them (the counts of the kind x-copy, 9 and 8, are never read).
    assert dict(got.expect) == written
    assert _beside(got) == {"same_shard_pairs": 4, "split_pairs": 3}
    assert got.open == ()


def test_the_no_tokens_row_counts_the_drops_with_that_reason_and_the_documents_by_shard():
    listed = "c03.drops.no-tokens"
    written = _fill(_written_expect(listed), _spec())
    assert written["drops_with_reason_no_tokens"] == 0
    clean = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert dict(clean.expect) == written
    assert _beside(clean) == {
        "drops_with_reason_no_tokens": 0,
        "documents_per_shard": {"shard-1": 3, "shard-2": 2},
        "surviving_lines_per_shard": {"shard-1": 3, "shard-2": 2},
    }
    # A document that holds one such drop: the count is beside the expectation, and the
    # table's own number (a number, not a sentence) stays the expected one.
    dirty = _concrete(listed, "100k", counted={"100k": NO_TOKENS_DOC})
    assert dirty.expect["drops_with_reason_no_tokens"] == 0
    assert _beside(dirty)["drops_with_reason_no_tokens"] == 1
    # Each size reads the document of its own set.
    both = {SIZED_SMALL: PLANTED, "100k": NO_TOKENS_DOC}
    assert _beside(_concrete(listed, SIZED_SMALL, counted=both))["drops_with_reason_no_tokens"] == 0
    assert _beside(_concrete(listed, "100k", counted=both))["drops_with_reason_no_tokens"] == 1


def test_the_two_rows_of_one_layout_of_the_10k_set_hold_the_other_shard_list_beside_their_expectation():
    concrete = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    kept = concrete(expected_for("c09.10k.partitioned.kept-shards", None, counted={None: TEN_K_DOC}))
    assert kept.expect["keep"] == ["shard-02", "shard-07", "shard-11"]
    assert "rewrite" not in kept.expect
    assert _beside(kept) == {"rewrite": ["shard-01", "shard-05"]}
    rewrite = concrete(expected_for("c09.10k.round-robin.all-rewrite", None, counted={None: TEN_K_RR_DOC}))
    assert dict(rewrite.expect) == {"rewrite": 20, "keep": 0}
    assert _beside(rewrite) == {"keep": 0, "rewrite": 20}


def test_each_size_reads_the_counted_values_of_its_own_set_on_the_concrete_path():
    listed = "c03.records.capture"
    counted = {SIZED_SMALL: PLANTED, "100k": BIG_COUNTED}
    small = _concrete(listed, SIZED_SMALL, counted=counted)
    big = _concrete(listed, "100k", counted=counted)
    assert small.expect["per_ordinal"]["shard-2"] == ["Z-004", "Z-005"]
    assert big.expect["per_ordinal"] == {"shard-4": ["Z-010", "Z-011", "Z-012", "Z-013"]}
    assert _beside(small) == {} and _beside(big) == {}
    # A size the mapping holds no document for reads "expected value not derived": its keys
    # are open, nothing is raised, and no other size's document is read in its place.
    missing = _without_key_error(_concrete, listed, "100k", counted={SIZED_SMALL: PLANTED})
    assert set(missing.open) == {"per_ordinal"} and _beside(missing) == {}
    assert dict(missing.expect) == _fill(_written_expect(listed), _spec())


def _refuses_a_missing_part(read) -> bool:
    """True when ``read`` names a part that is missing instead of answering as if the
    set had none of it (no drops, no pairs): the property under test."""
    try:
        read()
    except ValueError:
        return True
    return False


def test_a_part_the_document_does_not_hold_is_a_named_error_on_the_concrete_path_never_an_empty_answer():
    part_missing = need_attr("baselines.expected", "PartMissing")
    assert issubclass(part_missing, ValueError)
    for listed, part, size in (
        ("c03.drops.sanitizer", "check3.drops", SIZED_SMALL),
        ("c06.generated.drops", "check3.drops", SIZED_SMALL),
        ("c03.pairs.same-shard", "check3.pairs", SIZED_SMALL),
        ("c03.pairs.split", "check3.pairs", SIZED_SMALL),
        ("c03.records.capture", "documents", SIZED_SMALL),
        ("c06.generated.documents-mapped", "documents", SIZED_SMALL),
        # The rows the retired reader alone answered name their part too.
        ("c03.pairs.same-shard.swap", "check3.pairs", SIZED_SMALL),
        ("c06.generated.downstream-of-translation", "check6.translations", SIZED_SMALL),
        ("c03.precondition.pair-kinds", "check3.pairs.counts", "100k"),
        ("c03.drops.no-tokens", "check3.drops", "100k"),
    ):
        with pytest.raises(part_missing) as caught:
            _concrete(listed, size, counted={size: NO_CHUNKS_NO_DROPS})
        assert part in str(caught.value), listed
        # A document that is present and lacks the part is this error. No document at all is
        # another reading (see the tests of a missing counting document below).
    # The rows of the 10k set that count a document name the part they need.
    for listed in ("c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"):
        with pytest.raises(part_missing) as caught:
            _concrete(listed, None, counted={None: NO_SHARDS_DOC})
        assert "shards" in str(caught.value), listed
    # The known-bad reader: a null drops part answered as "no drops". The same checker
    # that says the real function refuses says this one does not.
    real = lambda: _concrete("c03.drops.sanitizer", SIZED_SMALL,  # noqa: E731
                             counted={SIZED_SMALL: NO_CHUNKS_NO_DROPS})
    bad = lambda: {"drops": [], "documents_per_shard": {}}  # noqa: E731
    assert _refuses_a_missing_part(real) is True
    assert _refuses_a_missing_part(bad) is False


# --- No counting document at all: the keys the document would give read "expected value not derived"
#
# A row whose source names a counting document (every row below: "the plant list, counted by
# the counting script", or for the records rows the id each chunk-file line carries, which the
# counting script accepts) and that is asked with no document given has no expected value for
# what that document would give. It is not an error and not an empty answer: ``concrete_values``
# returns the row's ``Concrete`` whose ``open`` names the keys the document would give, whose
# ``expect`` is the table's own (the table's words in an open key never stand in and are never
# compared) and which holds nothing beside it. No row of the ten asked about has an expectation
# that is concrete while its source names no counting document: each source names the script.
# The keys are those the rows read when a document is given (the tests above and below).

COUNT_KEYS = (
    "documents_per_shard", "documents_with_one_tied_row_per_shard",
    "documents_with_no_tied_row_per_shard",
)

# The row, and the keys the document would give.
NO_DOCUMENT_OPEN_TEN = {
    "c03.drops.sanitizer": {"drops", "documents_per_shard"},
    "c06.generated.drops": {"drops", "documents_per_shard"},
    "c03.pairs.same-shard": {"candidate_set"},
    "c03.pairs.split": {"record"},
    "c03.records.capture": {"per_ordinal"},
    "c06.generated.documents-mapped": {"per_ordinal", *COUNT_KEYS},
    "c03.pairs.same-shard.swap": {"candidate_set"},
    "c06.generated.downstream-of-translation": {"translations"},
    "c03.precondition.pair-kinds": {"same_shard_pairs", "split_pairs"},
    "c03.drops.no-tokens": {
        "drops_with_reason_no_tokens", "documents_per_shard", "surviving_lines_per_shard",
    },
}
# The same reading for the other rows that read a counting document: (size, keys).
NO_DOCUMENT_OPEN_OTHERS = {
    "c03.records.position-recovery": ("small", {"per_ordinal"}),
    "c03.records.token-matching": ("small", {"per_ordinal"}),
    "c06.generated.candidates": ("small", {"candidate_sources"}),
    "c09.10k.partitioned.kept-shards": (None, {"keep", "rewrite"}),
    "c09.10k.round-robin.all-rewrite": (None, {"keep", "rewrite"}),
}


def _concrete_or_fail(listed, size=None, **kwargs):
    """The concrete values of a row asked with no counting document; an error raised for the
    missing document is a failed assertion that says so."""
    try:
        return _concrete(listed, size, **kwargs)
    except ValueError as problem:
        pytest.fail(
            f"{listed}: a missing counting document raised {problem!r}; "
            "it reads expected value not derived"
        )


def _reads_not_derived(listed, size, keys, **kwargs):
    got = _concrete_or_fail(listed, size, **kwargs)
    assert got.listed_row == listed and got.size == size, listed
    assert set(got.open) == set(keys), listed
    # The table's own expectation stays; its words in an open key are not a value.
    assert dict(got.expect) == _fill(_written_expect(listed), _spec()), listed
    assert _beside(got) == {}, listed
    assert _not_compared(got) == (), listed
    return got


def test_with_no_counting_document_each_of_ten_rows_reads_expected_value_not_derived_and_raises_nothing(facts):
    assert len(NO_DOCUMENT_OPEN_TEN) == 10
    for listed, keys in NO_DOCUMENT_OPEN_TEN.items():
        for size in facts.sizes_by_listed[listed]:
            _reads_not_derived(listed, size, keys)
    # The two with the words of the table itself in their expectation: the word stays as the
    # table wrote it, and the key stays open.
    swap = _reads_not_derived("c03.pairs.same-shard.swap", "small", {"candidate_set"})
    assert swap.expect["candidate_set"] == "unchanged"
    kinds = _reads_not_derived("c03.precondition.pair-kinds", "small",
                               {"same_shard_pairs", "split_pairs"})
    assert kinds.expect["same_shard_pairs_at_least"] == 1


def test_with_no_counting_document_the_other_rows_that_read_one_read_expected_value_not_derived_too():
    for listed, (size, keys) in NO_DOCUMENT_OPEN_OTHERS.items():
        _reads_not_derived(listed, size, keys)
    # The answer word of a candidate set is the table's own and stays a word.
    candidates = _reads_not_derived("c06.generated.candidates", "small", {"candidate_sources"})
    assert candidates.expect["answer_for_each_candidate_source"] == "may contain"


def test_a_size_whose_counting_document_is_missing_reads_not_derived_for_its_counts_and_the_others_keep_theirs(facts):
    listed = "c06.generated.documents-mapped"
    sizes = tuple(facts.sizes_by_listed[listed])
    assert len(sizes) > 1
    keys = NO_DOCUMENT_OPEN_TEN[listed]
    # A document at one size only (the first, then the last): every other size has none.
    for have in (sizes[0], sizes[-1]):
        held = {size: (GROUPS_DOC if size == have else None) for size in sizes}
        assert _concrete_or_fail(listed, have, counted=held).open == (), have
        for size in sizes:
            if size != have:
                _reads_not_derived(listed, size, keys, counted=held)


def test_a_row_that_reads_nothing_of_the_document_gives_the_table_own_expectation_and_nothing_beside_it():
    for listed, size in (
        ("c03.verification-failed", SIZED_SMALL), ("c03.map-fields", SIZED_SMALL),
        ("c05.unmutated.R1", None),
    ):
        got = _concrete(listed, size, counted={size: PLANTED})
        assert dict(got.expect) == _fill(_written_expect(listed), _spec()), listed
        assert got.open == (), listed
        assert _beside(got) == {}, listed
        assert _not_compared(got) == (), listed
    # A row outside the table is refused.
    with pytest.raises(KeyError):
        _concrete("c99.no-such-row")


def test_the_seven_rows_that_state_words_and_numbers_are_read_per_layout_with_no_counted_number():
    # The seven rows of Check 9 on the 10k set whose expectation the old reader refused as a
    # row it did not know (a raised key error): counted from the table by this test.
    concrete = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    rows, both, one = _ten_k_rows()
    assert len(rows) == 10 and len(one) == 2
    seven = [listed for listed in both if listed != "c09.10k.action-counts"]
    assert len(seven) == 7
    for listed in seven:
        for given in (None, {"partitioned": PART10, "round-robin": ROUND_ROBIN10}):
            got = _by_layout(listed, given)
            assert list(got) == LAYOUT_NAMES, listed
            for layout, slot in got.items():
                assert not isinstance(slot, Exception), (listed, layout)
                assert dict(slot.expect) == _fill(_written_expect(listed), _spec()), listed
                assert slot.open == (), listed
                assert _beside(slot) == {}, listed
                assert _not_compared(slot) == (), listed
        # Asked as one value, a row of two layouts is refused.
        with pytest.raises(ValueError, match="layouts"):
            concrete(expected_for(listed))


# --- The oracle's log -------------------------------------------------------------------


def _log(segment, step, sample_pos, shard_position, ordinal, start, end):
    return {
        "segment": segment, "step": step, "sample_pos": sample_pos, "global_index": 0,
        "rank": 0, "shard_position": shard_position, "ordinal": ordinal,
        "token_start": start, "token_end": end,
    }


def test_the_rows_the_join_must_report_are_the_oracles_rows_by_sample():
    run_rows = need_attr("baselines.expected", "run_rows")
    oracle = [
        _log("E/seg-2", 7, 0, 2, 24, 32, 39),
        _log("E/seg-2", 7, 0, 2, 7, 0, 22),
        _log("E/seg-2", 7, 0, 2, 26, 0, 4),
        _log("E/seg-2", 8, 3, 2, 2, 19, 44),
        _log("E/seg-2", 8, 3, 2, 23, 0, 8),
    ]
    got = run_rows(oracle)
    assert {k: list(v) for k, v in got.items()} == {
        ("E/seg-2", 7, 0): [(2, 24, 32, 39), (2, 7, 0, 22), (2, 26, 0, 4)],
        ("E/seg-2", 8, 3): [(2, 2, 19, 44), (2, 23, 0, 8)],
    }
    # The half-covered sample of a lowered step: the first 16 tokens are read, the row
    # that holds the 17th token ends at the cut, no later row is reported.
    cut = run_rows(oracle, cuts={("E/seg-2", 8, 3): 16})
    assert list(cut[("E/seg-2", 8, 3)]) == [(2, 2, 19, 35)]
    assert list(cut[("E/seg-2", 7, 0)]) == [(2, 24, 32, 39), (2, 7, 0, 22), (2, 26, 0, 4)]
    # A cut on a sample the log does not hold is refused, not ignored.
    with pytest.raises(ValueError):
        run_rows(oracle, cuts={("E/seg-2", 9, 0): 4})


def test_the_cut_of_step_8_equals_the_rows_the_spec_states_for_the_covered_part():
    run_rows = need_attr("baselines.expected", "run_rows")
    positions = _spec()["positions"]["step8_cut"]
    oracle = [json.loads(line) for line in positions["log_lines"]]
    fixed = positions["fixed"]
    sample = (fixed["segment"], fixed["step"], fixed["sample_pos"])
    got = run_rows(oracle, cuts={sample: fixed["covered_tokens"]})
    stated = [
        (r["shard_position"], r["ordinal"], r["token_start"], r["token_end"])
        for r in positions["rows_of_the_covered_part"]
    ]
    assert list(got[sample]) == stated


# --- The concrete value of each sentence-valued field ---------------------------------------
#
# Some expected fields of the table are sentences (the table README, "Grades, and what is
# compared", asks for the fact, not the sentence). The names this part asks of
# ``baselines.expected``, all new:
#
# - ``expected_for(..., index=None)``: a new keyword; ``index`` maps a size to
#   ``{shard: [size of each document in the shard's own index]}``, the way ``counted`` maps
#   a size to a counting document. The ``Expected`` holds it as ``index``.
# - ``concrete_values(expected)`` returns a ``Concrete`` with ``listed_row``, ``size``,
#   ``layout`` (the row's ``given.layout`` or None), ``expect`` (the expectation in force
#   with each sentence-valued field replaced by its concrete value; every other field as
#   the table writes it) and ``open`` (a tuple of the keys whose source gives no concrete
#   value; each stays as the table wrote it). A part that a counting document, given, does not
#   hold, or an index that is missing, is a ``PartMissing`` naming it; a counting document not
#   given at all is not: the keys it would give are in ``open``. A row that does not apply under
#   the fallback in force is a ``ValueError``. A row that names two layouts is a
#   ``ValueError`` naming ``layouts``: it is asked per layout.
# - ``concrete_values_by_layout(expected)`` for a row whose ``given`` names ``layouts``:
#   a dict from each layout, in the row's order, to a ``Concrete`` for that layout (with its
#   ``open`` naming the keys a missing counting document would give), or, where the value of
#   that layout cannot be derived, the exception itself (a ``PartMissing`` for a part a given
#   document does not hold, or a ``ValueError`` for a document counted on the other layout),
#   returned and never raised: one layout that cannot be derived never takes the other
#   layout's values away. For such a row ``counted`` maps a layout to the counting
#   document counted on that layout.
# - ``fields_without_concrete_value()``: ONE tuple of ``(row, key, source text)`` for the
#   fields read "expected value not derived" (class iii below); a field derived from its
#   source (class i) or one of the table's not-compared parts (class ii) is not in it.
# - ``whole_run_rows(expected, oracle)``: for the three whole-run diff rows of Check 1,
#   the rows of the oracle's log of the run, as ``run_rows`` gives them, with the cut of
#   step 8 on run E.

SIZED_SMALL = "small"

# Hand-written in the shape of the counting script's documents part (docs/plant.md,
# section 7): the groups of identical documents of a shard list the ordinals, the lines
# and the ids of the group, in order. Shard-1 holds one pair, shard-2 a group of three.
GROUPS_DOC = {
    "documents": {"total": 9, "shards": [
        {"shard": "shard-1", "documents": 5,
         "ids": ["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"],
         "identical": [{"pair": "pair-1", "text_sha256": "aa" * 32, "ordinals": [1, 3],
                        "lines": [2, 4], "ids": ["Z-102", "Z-101"]}],
         "labels": {"recovered": 3, "candidates": 2}},
        {"shard": "shard-2", "documents": 4,
         "ids": ["Z-301", "Z-302", "Z-006", "Z-303"],
         "identical": [{"pair": "pair-9", "text_sha256": "bb" * 32, "ordinals": [0, 1, 3],
                        "lines": [1, 2, 4], "ids": ["Z-301", "Z-302", "Z-303"]}],
         "labels": {"recovered": 1, "candidates": 3}},
    ]},
}

INDEX_SMALL = {"shard-1": [7, 22, 9], "shard-2": [30, 4]}
INDEX_BIG = {"shard-1": [5, 5], "shard-3": [11]}


def _concrete(listed, size=None, *, counted=None, index=None, fallbacks=()):
    """The concrete values of one listed row. The new name is asked first, so a missing
    name is a failed assertion and not an error of an unknown keyword."""
    concrete_values = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    return concrete_values(
        expected_for(listed, size, fallbacks=fallbacks, counted=counted, index=index)
    )


def _without_key_error(call, *args, **kwargs):
    """``call(*args, **kwargs)``; a ``KeyError`` raised for a size the mapping holds nothing
    for is a failed assertion that says so, not an error."""
    try:
        return call(*args, **kwargs)
    except KeyError as problem:
        pytest.fail(
            f"a size the mapping holds nothing for raised {problem!r}; "
            "it reads expected value not derived"
        )


def test_each_ordinal_of_a_row_that_reads_the_documents_is_one_id_or_the_candidate_set_of_its_group():
    # Identical-document groups are carried into per_ordinal: an ordinal of a group reads
    # as the candidate set of the group, every other ordinal as its one planted id.
    stated = {
        "shard-1": ["Z-001", ["Z-102", "Z-101"], "Z-003", ["Z-102", "Z-101"], "Z-005"],
        "shard-2": [["Z-301", "Z-302", "Z-303"], ["Z-301", "Z-302", "Z-303"], "Z-006",
                    ["Z-301", "Z-302", "Z-303"]],
    }
    for listed in (
        "c03.records.capture", "c03.records.position-recovery", "c03.records.token-matching",
        "c06.generated.documents-mapped",
    ):
        got = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC})
        assert got.listed_row == listed and got.size == SIZED_SMALL
        assert got.expect["per_ordinal"] == stated, listed
        assert got.open == (), listed
        # The other fields of the row stay as the table writes them.
        written = _fill(_written_expect(listed), _spec())
        assert got.expect["mismatches" if "mismatches" in written else "unmapped_documents"] == 0


def _count(concrete, key):
    """One of the per-shard counts of the documents-mapped row; a missing one is a failed
    assertion that says so. They sit beside the expectation, with the row's other counted
    values."""
    beside = _beside(concrete)
    assert key in beside, f"the counted values hold no {key}"
    return beside[key]


def test_the_documents_mapped_row_holds_per_shard_the_documents_the_one_tied_row_and_the_none():
    # Hand-counted from the documents part of GROUPS_DOC: every document of a shard has
    # exactly one tied row (a member of a group of identical documents has its own companion
    # row, tied by its own src and loc), so with 5 documents in shard-1 and 4 in shard-2, 5 and
    # 4 have exactly one tied row and no document has none.
    listed = "c06.generated.documents-mapped"
    got = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC})
    assert _count(got, "documents_per_shard") == {"shard-1": 5, "shard-2": 4}
    assert _count(got, "documents_with_one_tied_row_per_shard") == {"shard-1": 5, "shard-2": 4}
    assert _count(got, "documents_with_no_tied_row_per_shard") == {"shard-1": 0, "shard-2": 0}
    # A document with no group: every document of a shard has exactly one tied row.
    plain = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert _count(plain, "documents_per_shard") == {"shard-1": 3, "shard-2": 2}
    assert _count(plain, "documents_with_one_tied_row_per_shard") == {"shard-1": 3, "shard-2": 2}
    assert _count(plain, "documents_with_no_tied_row_per_shard") == {"shard-1": 0, "shard-2": 0}
    # Those three are all the row holds beside its expectation.
    assert set(_beside(got)) == set(COUNT_KEYS)
    # The row's own numbers and its one-id reading stay.
    assert got.expect["unmapped_documents"] == 0
    assert got.open == ()
    assert listed in need_attr("baselines.expected", "ONE_ID_ROWS")
    # Each size reads the document of its own set.
    sizes = {SIZED_SMALL: GROUPS_DOC, "10k": PLANTED, "100k": GROUPS_DOC}
    assert _count(_concrete(listed, "10k", counted=sizes), "documents_per_shard") == {
        "shard-1": 3, "shard-2": 2,
    }
    assert _count(_concrete(listed, "100k", counted=sizes), "documents_per_shard") == {
        "shard-1": 5, "shard-2": 4,
    }


def test_the_documents_mapped_counts_do_not_depend_on_the_pairs_held_beside_the_documents():
    # The counts are read from the documents part alone: with the pairs of Check 3 in the
    # document as well (shard-2 holds a pair of two there, not the group of three) a document
    # of a shard still has exactly one tied row.
    listed = "c06.generated.documents-mapped"
    both = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: ONE_DOCUMENT})
    assert _count(both, "documents_with_one_tied_row_per_shard") == {"shard-1": 5, "shard-2": 4}
    assert _count(both, "documents_with_no_tied_row_per_shard") == {"shard-1": 0, "shard-2": 0}


# The same two shards as GROUPS_DOC, written by hand, without the map labels of the counting
# script: the labels are the tool's map labels and no tie, so the counts do not need them.
NO_LABELS_DOC = {
    "documents": {"total": 9, "shards": [
        {"shard": "shard-1", "documents": 5,
         "ids": ["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"],
         "identical": [{"pair": "pair-1", "text_sha256": "aa" * 32, "ordinals": [1, 3],
                        "lines": [2, 4], "ids": ["Z-102", "Z-101"]}]},
        {"shard": "shard-2", "documents": 4,
         "ids": ["Z-301", "Z-302", "Z-006", "Z-303"],
         "identical": [{"pair": "pair-9", "text_sha256": "bb" * 32, "ordinals": [0, 1, 3],
                        "lines": [1, 2, 4], "ids": ["Z-301", "Z-302", "Z-303"]}]},
    ]},
}


# The same two shards, written by hand, where shard-2 gives no document count.
NO_COUNT_DOC = {
    "documents": {"total": 5, "shards": [
        {"shard": "shard-1", "documents": 5,
         "ids": ["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"],
         "identical": [{"pair": "pair-1", "text_sha256": "aa" * 32, "ordinals": [1, 3],
                        "lines": [2, 4], "ids": ["Z-102", "Z-101"]}]},
        {"shard": "shard-2",
         "ids": ["Z-301", "Z-302", "Z-006", "Z-303"],
         "identical": [{"pair": "pair-9", "text_sha256": "bb" * 32, "ordinals": [0, 1, 3],
                        "lines": [1, 2, 4], "ids": ["Z-301", "Z-302", "Z-303"]}]},
    ]},
}


def test_the_counts_do_not_need_the_map_labels_of_the_documents_part():
    # The labels (recovered, candidates) are the tool's map labels, not ties: a documents
    # part without them gives the same counts, and nothing is open.
    listed = "c06.generated.documents-mapped"
    got = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: NO_LABELS_DOC})
    assert got.open == ()
    assert _count(got, "documents_per_shard") == {"shard-1": 5, "shard-2": 4}
    assert _count(got, "documents_with_one_tied_row_per_shard") == {"shard-1": 5, "shard-2": 4}
    assert _count(got, "documents_with_no_tied_row_per_shard") == {"shard-1": 0, "shard-2": 0}


def test_a_shard_whose_document_count_is_not_given_leaves_the_counts_not_derived():
    # shard-1 gives 5 documents, shard-2 gives none: the three counts are open, never guessed
    # from the ids or read as 0; the row's own number and its one-id reading stay.
    listed = "c06.generated.documents-mapped"
    got = _without_key_error(_concrete, listed, SIZED_SMALL, counted={SIZED_SMALL: NO_COUNT_DOC})
    assert set(got.open) == set(COUNT_KEYS)
    assert _beside(got) == {}
    assert got.expect["unmapped_documents"] == 0
    assert got.expect["per_ordinal"]["shard-1"][0] == "Z-001"
    assert got.expect["per_ordinal"]["shard-2"][2] == "Z-006"
    # The same row asked of a document that does give them has nothing open.
    given = _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC})
    assert given.open == ()


def test_a_group_of_identical_documents_is_read_per_shard_and_a_missing_part_is_named():
    part_missing = need_attr("baselines.expected", "PartMissing")
    # The same ordinals in two shards: each shard's groups are its own.
    got = _concrete("c03.records.capture", SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC})
    assert got.expect["per_ordinal"]["shard-1"][0] == "Z-001"
    assert got.expect["per_ordinal"]["shard-2"][2] == "Z-006"
    # No document part: named, never read as "no documents".
    with pytest.raises(part_missing) as caught:
        _concrete("c03.records.capture", SIZED_SMALL, counted={SIZED_SMALL: NO_CHUNKS_NO_DROPS})
    assert "documents" in str(caught.value)


def test_the_fallback_in_force_changes_no_concrete_value_of_a_row_that_still_applies():
    under = _concrete("c03.records.capture", SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC},
                      fallbacks=("F1",))
    assert under.expect["map_built_by"] == "token-matching"
    assert under.expect["per_ordinal"]["shard-1"][1] == ["Z-102", "Z-101"]
    with pytest.raises(ValueError):
        _concrete("c03.records.position-recovery", SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC},
                  fallbacks=("F1",))


def test_the_spans_of_a_shard_are_the_document_sizes_of_its_own_index():
    # c03.spans.*: the compared value is the size of each document in the shard's own
    # index (the replica's records), and its total; the caller reads the index.
    per_ordinal = _concrete("c03.spans.per-ordinal", SIZED_SMALL, index={SIZED_SMALL: INDEX_SMALL})
    assert per_ordinal.expect["per_ordinal"] == {"shard-1": [7, 22, 9], "shard-2": [30, 4]}
    assert per_ordinal.expect["mismatches"] == 0
    assert per_ordinal.open == ()
    total = _concrete("c03.spans.total", SIZED_SMALL, index={SIZED_SMALL: INDEX_SMALL})
    assert total.expect["per_shard"] == {"shard-1": 38, "shard-2": 34}
    assert total.expect["mismatches"] == 0
    # Each size reads the index of its own set.
    big = _concrete("c03.spans.total", "100k", index={SIZED_SMALL: INDEX_SMALL, "100k": INDEX_BIG})
    assert big.expect["per_shard"] == {"shard-1": 10, "shard-3": 11}


def test_the_spans_without_the_index_of_their_size_name_the_missing_part():
    part_missing = need_attr("baselines.expected", "PartMissing")
    for listed in ("c03.spans.per-ordinal", "c03.spans.total"):
        with pytest.raises(part_missing) as caught:
            _concrete(listed, SIZED_SMALL)
        assert "index" in str(caught.value), listed
        # A size the mapping holds no index for is none held, the same as no index at all:
        # the part is named, and no other size's index is read in its place.
        with pytest.raises(part_missing) as missing:
            _without_key_error(_concrete, listed, "100k", index={SIZED_SMALL: INDEX_SMALL})
        assert "index" in str(missing.value), listed


def _write_index(path, sequence_lengths, document_marks):
    """Write a shard index file by hand, version 1 of the format the replica reads: the
    9-byte header, the version, the token type code (4: four bytes a token), the sequence
    count, the entry count, then the sequence lengths (int32), the sequence pointers (int64,
    in bytes) and the document indices (int64). A document is the run of sequences between
    two document indices."""
    sequences, entries = len(sequence_lengths), len(document_marks)
    pointers, at = [], 0
    for length in sequence_lengths:
        pointers.append(at)
        at += length * 4
    body = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 4, sequences, entries)
    body += b"".join(struct.pack("<i", n) for n in sequence_lengths)
    body += b"".join(struct.pack("<q", n) for n in pointers)
    body += b"".join(struct.pack("<q", n) for n in document_marks)
    path.write_bytes(body)
    return path


def _shard_files(folder):
    """The shard files of INDEX_SMALL, written by hand: shard-1 holds three documents, the
    second of them made of two sequences (10 and 12 tokens); shard-2 holds two."""
    return {
        "shard-1": _write_index(folder / "shard-1.idx", [7, 10, 12, 9], [0, 1, 3, 4]),
        "shard-2": _write_index(folder / "shard-2.idx", [30, 4], [0, 1, 2]),
    }


def test_the_sizes_of_the_documents_are_read_from_the_shards_own_index_files(tmp_path):
    index_of_shards = need_attr("baselines.expected", "index_of_shards")
    got = index_of_shards(_shard_files(tmp_path))
    # The hand values the tests above give for the same two shards.
    assert {shard: list(sizes) for shard, sizes in got.items()} == INDEX_SMALL
    assert list(got) == ["shard-1", "shard-2"]
    # They are the index the spans rows take: the same values as when the caller reads it.
    per_ordinal = _concrete("c03.spans.per-ordinal", SIZED_SMALL, index={SIZED_SMALL: got})
    assert per_ordinal.expect["per_ordinal"] == INDEX_SMALL
    total = _concrete("c03.spans.total", SIZED_SMALL, index={SIZED_SMALL: got})
    assert total.expect["per_shard"] == {"shard-1": 38, "shard-2": 34}


def test_a_file_that_is_not_a_shard_index_is_an_error_that_names_it_and_never_no_documents(tmp_path):
    index_of_shards = need_attr("baselines.expected", "index_of_shards")
    files = _shard_files(tmp_path)
    broken = tmp_path / "shard-3.idx"
    broken.write_bytes(b"this is not an index file at all")
    files["shard-3"] = broken
    with pytest.raises(Exception) as caught:
        index_of_shards(files)
    assert "shard-3.idx" in str(caught.value)


def test_the_plant_fields_of_the_drops_pairs_and_candidates_are_the_counted_values():
    # The counted values of the drops, pairs and candidates are put in place of the sentences.
    drops = _concrete("c03.drops.sanitizer", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert drops.expect["drops"] == [
        {"shard": "shard-1", "id": "Z-900", "reason": "sanitizer skip", "input_line": 2,
         "ordinal_of_next_line": 1},
        {"shard": "shard-1", "id": "Z-901", "reason": "empty line", "input_line": 5,
         "ordinal_of_next_line": 3},
        {"shard": "shard-2", "id": "Z-902", "reason": "sanitizer skip", "input_line": 1,
         "ordinal_of_next_line": 0},
    ]
    assert drops.expect["documents_per_shard"] == {"shard-1": 3, "shard-2": 2}
    assert drops.expect["reasons"] == ["sanitizer skip", "empty line"]
    recovered = _concrete("c06.generated.drops", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert recovered.expect["drops"] == drops.expect["drops"]
    assert recovered.expect["drops_with_reason_no_tokens"] == 0
    same = _concrete("c03.pairs.same-shard", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert same.expect["candidate_set"] == [
        {"pair": "pair-1", "shard": "shard-1", "lines": [3, 9]},
        {"pair": "pair-3", "shard": "shard-2", "lines": [1, 4]},
    ]
    assert same.expect["label"] == "candidates"
    split = _concrete("c03.pairs.split", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert split.expect["record"] == [
        {"pair": "pair-2", "members": [
            {"id": "Z-201", "shard": "shard-1", "line": 5},
            {"id": "Z-202", "shard": "shard-2", "line": 2},
        ]},
    ]
    candidates = _concrete("c06.generated.candidates", SIZED_SMALL, counted={SIZED_SMALL: PLANTED})
    assert candidates.expect["candidate_sources"] == [
        {"release": "R1", "pair": "pair-x", "sources": ["A", "B"]},
    ]
    # The answer word of a candidate set is the plan's own word and stays as written.
    assert candidates.expect["answer_for_each_candidate_source"] == "may contain"
    assert candidates.open == ()


# The fields of the 19 rows whose expected value the table writes as a sentence, by row.
# Each is told by what its row's own source gives: a counted value, the shard's own index,
# an answer or status word that stays as written, or nothing concrete (listed).
SENTENCE_FIELDS = {
    "c02.ii.replay-rejected": ("digests",),
    "c02.iii.no-replay": ("replay",),
    "c03.spans.per-ordinal": ("per_ordinal",),
    "c03.spans.total": ("per_shard",),
    "c03.drops.sanitizer": ("drops", "documents_per_shard"),
    "c03.records.capture": ("per_ordinal",),
    "c03.records.position-recovery": ("per_ordinal",),
    "c03.records.token-matching": ("per_ordinal",),
    "c03.pairs.same-shard": ("candidate_set",),
    "c03.pairs.split": ("record",),
    "c04.declare-499.write": ("map_after",),
    "c06.generated.documents-mapped": ("per_ordinal",),
    "c06.generated.drops": ("drops",),
    "c06.generated.candidates": ("candidate_sources", "answer_for_each_candidate_source"),
    "c06.import.review-scope": ("missing_link",),
    "c08.K.digests-match": (
        "shuffle_digests", "digest_covers", "epoch_0_digests_of_launch_0_and_launch_1",
    ),
    "c09.10k.action-counts": ("counts",),
    "c09.10k.partitioned.kept-shards": ("keep",),
    "c09.E.stale-cache.E/seg-2": ("rebuilt_from",),
}

# The status and answer words that stay as the table writes them: a word, not a sentence
# that stands for a value.
WORDS_THAT_STAY = {
    ("c02.iii.no-replay", "replay"),
    ("c06.generated.candidates", "answer_for_each_candidate_source"),
}

# The seven fields whose table value is a sentence that no counted document or index
# turns into a value. None of them is ever compared as a string; each is in exactly one of
# three classes, told by what the row's own `expect` and `source` say (hand-read from
# check02.toml, check04.toml, check06.toml, check08.toml and check09.toml).
#
# (i) derived as a concrete value from the record its source names: none of the seven. A
#     value would need a record whose shape a document or the spec states, and none of the
#     seven rows names one (each source is a hand-written sentence, or names a run record
#     that has not been made).
#
# (ii) one of the table README's not-compared parts (section "Grades, and what is compared":
#     "the failing link, the chain, the scope"). It stays as the table wrote it, is not in
#     `open` and is named in `not_compared`.
#     c06.import.review-scope `missing_link` = "the record's own parent; only the file-level
#     link to A/r1 is recorded": the failing link; its source adds that the table's reader
#     "skips this row until the report exists", so it is not compared and not listed.
#     c08.K.digests-match `digest_covers` = "the whole order of one rank for one epoch,
#     whatever part of the epoch the launch read": a statement of scope.
NOT_COMPARED_FIELDS = (
    ("c06.import.review-scope", "missing_link"),
    ("c08.K.digests-match", "digest_covers"),
)
#
# (iii) "expected value not derived" until a concrete value exists. It stays as the table
#     wrote it and is listed.
#     c02.ii.replay-rejected `digests` = "differ from the recorded ones": needs the replayed
#     digests and the run record's.
#     c04.declare-499.write `map_after` = "no map for the shard": the table's number
#     `rows_after` = 0 is compared; the state of a store after the refused write is read from
#     the tool's own output, which gives no value independent of it.
#     c08.K.digests-match `shuffle_digests` = "equal, for every (launch, epoch, rank) of the
#     run record" and `epoch_0_digests_of_launch_0_and_launch_1` = "equal, rank by rank":
#     both need run K's run record.
#     c09.E.stale-cache.E/seg-2 `rebuilt_from` = "an empty cache folder"; source: "the
#     corrected order needs an empty cache folder (4.4)": a hand-written sentence that names
#     no record, so no value of it is read from one.
NO_CONCRETE_VALUE = (
    ("c02.ii.replay-rejected", "digests"),
    ("c04.declare-499.write", "map_after"),
    ("c08.K.digests-match", "shuffle_digests"),
    ("c08.K.digests-match", "epoch_0_digests_of_launch_0_and_launch_1"),
    ("c09.E.stale-cache.E/seg-2", "rebuilt_from"),
)

PART10 = {
    "actions": {"records": {"keep": 7800, "re-source": 500, "promote": 500, "retire": 1100,
                            "none": 300},
                "shards": {"keep": 2, "rewrite": 2}},
    "releases": {
        "R1": {"actions": {"keep": 3405, "re-source": 500, "promote": 500, "retire": 1100,
                           "none": 300}, "removed": 1100},
        "R2": {"actions": {"keep": 4395, "re-source": 0, "promote": 0, "retire": 0, "none": 0},
               "removed": 0},
    },
    "shards": {
        "layout": "partitioned", "releases": ["R1", "R2"], "lines": 8300,
        "shards": [
            {"shard": "shard-A", "lines": 1500, "in_scope_lines": 600, "own": 600, "derived": 0,
             "receives": 1, "generated_kept": 0, "action": "rewrite"},
            {"shard": "shard-C", "lines": 300, "in_scope_lines": 0, "own": 0, "derived": 0,
             "receives": 0, "generated_kept": 0, "action": "keep"},
            {"shard": "shard-Z", "lines": 5300, "in_scope_lines": 0, "own": 0, "derived": 0,
             "receives": 0, "generated_kept": 0, "action": "keep"},
            {"shard": "shard-gen", "lines": 1200, "in_scope_lines": 1100, "own": 0,
             "derived": 1100, "receives": 0, "generated_kept": 10, "action": "rewrite"},
        ],
        "keep": ["shard-C", "shard-Z"], "rewrite": ["shard-A", "shard-gen"],
        "all_rewrite": False, "actions": {"keep": 2, "rewrite": 2},
    },
}

_RR_NAMES = [f"shard-{n:02d}" for n in range(1, 21)]
ROUND_ROBIN10 = {
    "actions": dict(PART10["actions"], shards={"keep": 0, "rewrite": 20}),
    "releases": PART10["releases"],
    "shards": {
        "layout": "round-robin", "releases": ["R1", "R2"], "lines": 8300,
        "shards": [
            {"shard": name, "lines": 415, "in_scope_lines": 85, "own": 30, "derived": 55,
             "receives": 0, "generated_kept": 0, "action": "rewrite"}
            for name in _RR_NAMES
        ],
        "keep": [], "rewrite": _RR_NAMES, "all_rewrite": True,
        "actions": {"keep": 0, "rewrite": 20},
    },
}

PARTITIONED_COUNTS = {
    "records": {"keep": 7800, "re-source": 500, "promote": 500, "retire": 1100},
    "records_removed": 1100,
    "shards": {"keep": 2, "rewrite": 2},
}
ROUND_ROBIN_COUNTS = dict(PARTITIONED_COUNTS, shards={"keep": 0, "rewrite": 20})


# One counting document that holds every part a row of Check 3 or 6 reads: the documents
# with their groups, and the pairs, drops and candidates of PLANTED.
ONE_DOCUMENT = dict(GROUPS_DOC, check3=PLANTED["check3"], check6=PLANTED["check6"])


def _every_field_kind_row(listed):
    """Ask one of the 19 rows with every document and index it could read."""
    given = _listed_rows()[listed]["given"]
    if "layouts" in given:
        by_layout = _by_layout(listed, {"partitioned": PART10, "round-robin": ROUND_ROBIN10})
        return by_layout["partitioned"]
    if listed.startswith("c09.10k."):
        document = ROUND_ROBIN10 if given["layout"] == "round-robin" else PART10
        return _concrete(listed, None, counted={None: document})
    if listed.startswith(("c03.", "c06.generated")):
        return _concrete(listed, SIZED_SMALL, counted={SIZED_SMALL: ONE_DOCUMENT},
                         index={SIZED_SMALL: INDEX_SMALL})
    return _concrete(listed)


def test_the_sentence_fields_have_one_public_source_of_19_rows_and_23_pairs(facts):
    # The counts, from the hand-read table of this file and the sizes the table gives.
    pairs = [(listed, key) for listed, keys in SENTENCE_FIELDS.items() for key in keys]
    assert len(SENTENCE_FIELDS) == 19 and len(pairs) == 23
    # The words that stay and the seven fields with no counted value are among the 23.
    assert WORDS_THAT_STAY <= set(pairs)
    assert set(NOT_COMPARED_FIELDS) | set(NO_CONCRETE_VALUE) <= set(pairs)
    assert len(set(NOT_COMPARED_FIELDS) | set(NO_CONCRETE_VALUE)) == 7
    # Counted once per size the row is asked at, the pairs are 40 questions' worth of fields.
    assert sum(len(facts.sizes_by_listed[listed]) for listed, _key in pairs) == 40
    # The public source holds exactly these rows and keys.
    public = need_attr("baselines.expected", "SENTENCE_FIELDS")
    assert {listed: set(keys) for listed, keys in public.items()} == {
        listed: set(keys) for listed, keys in SENTENCE_FIELDS.items()
    }


def test_the_rows_whose_answer_is_one_id_per_ordinal_have_one_public_name():
    public = need_attr("baselines.expected", "ONE_ID_ROWS")
    assert set(public) == set(ONE_ID_ROWS)
    assert len(set(public)) == 4


def test_every_sentence_field_of_the_19_rows_is_in_exactly_one_class():
    # Counted from the table by this test: the 19 rows exist, hold the named fields, and a
    # field is replaced by a concrete value, kept as a word, or is one of the seven in
    # exactly one of the classes above (none of the seven is derived).
    assert len(SENTENCE_FIELDS) == 19
    classes = [set(NOT_COMPARED_FIELDS), set(NO_CONCRETE_VALUE)]
    assert sum(len(one) for one in classes) == len(set().union(*classes)) == 7
    for listed, keys in SENTENCE_FIELDS.items():
        written = _fill(_written_expect(listed), _spec())
        got = _every_field_kind_row(listed)
        for key in keys:
            assert key in written, (listed, key)
            here = (listed, key)
            if here in NO_CONCRETE_VALUE:
                assert key in got.open, here
                assert key not in _not_compared(got), here
                assert got.expect[key] == written[key], here
            elif here in NOT_COMPARED_FIELDS:
                assert key not in got.open, here
                assert key in _not_compared(got), here
                assert got.expect[key] == written[key], here
            elif here in WORDS_THAT_STAY:
                assert key not in got.open, here
                assert key not in _not_compared(got), here
                assert got.expect[key] == written[key], here
            else:
                assert key not in got.open, here
                assert key not in _not_compared(got), here
                assert got.expect[key] != written[key], here
                assert not isinstance(got.expect[key], str), here
        assert set(got.open) == {k for r, k in NO_CONCRETE_VALUE if r == listed}, listed
        assert set(_not_compared(got)) == {k for r, k in NOT_COMPARED_FIELDS if r == listed}, listed
        # Every field the sentence rule does not touch is the table's own.
        for key, value in written.items():
            if key not in keys:
                assert got.expect[key] == value, (listed, key)


def test_the_link_of_the_import_row_and_the_scope_of_the_digest_are_not_compared_and_not_listed():
    review = _concrete("c06.import.review-scope")
    assert review.open == ()
    assert _not_compared(review) == ("missing_link",)
    assert review.expect["missing_link"] == _written_expect("c06.import.review-scope")["missing_link"]
    digests = _concrete("c08.K.digests-match")
    assert _not_compared(digests) == ("digest_covers",)
    assert set(digests.open) == {"shuffle_digests", "epoch_0_digests_of_launch_0_and_launch_1"}


def test_the_fields_read_expected_value_not_derived_come_back_as_one_list_with_the_source_text():
    fields = need_attr("baselines.expected", "fields_without_concrete_value")
    got = fields()
    assert isinstance(got, tuple)
    assert sorted(got) == sorted(
        (listed, key, _listed_rows()[listed]["source"]) for listed, key in NO_CONCRETE_VALUE
    )
    # No row is in it twice; a not-compared part is not in it.
    assert len(set(got)) == len(got) == 5
    listed_pairs = {(listed, key) for listed, key, _source in got}
    assert not listed_pairs & set(NOT_COMPARED_FIELDS)


def test_none_recorded_under_the_cyclic_wrapper_route_is_an_expected_value_and_not_a_field_to_list():
    row = "c08.K.digests-match"
    assert _listed_rows()[row]["expect_under"]["cyclic-wrapper-route"] == {
        "shuffle_digests": "none recorded",
    }
    under = _concrete(row, fallbacks=("cyclic-wrapper-route",))
    assert dict(under.expect) == {"shuffle_digests": "none recorded"}
    assert under.open == ()
    assert _not_compared(under) == ()
    # Without the fallback the same key is a sentence that no run record has turned into a value.
    assert "shuffle_digests" in _concrete(row).open


# --- Check 9 on the 10k set, per layout ----------------------------------------------------


def _ten_k_rows():
    """The c09.10k rows, split by what their ``given`` names: two layouts or one."""
    rows = {listed: row for listed, row in _listed_rows().items() if listed.startswith("c09.10k.")}
    both = sorted(listed for listed, row in rows.items() if "layouts" in row["given"])
    one = sorted(listed for listed, row in rows.items() if "layout" in row["given"])
    return rows, both, one


def _by_layout(listed, counted=None):
    by_layout = need_attr("baselines.expected", "concrete_values_by_layout")
    expected_for = need_attr("baselines.expected", "expected_for")
    return by_layout(expected_for(listed, None, counted=counted))


def test_a_row_asked_in_both_layouts_gives_one_value_per_layout_in_the_rows_order():
    need_attr("baselines.expected", "concrete_values_by_layout")
    # The c09.10k rows, counted from the table: eight name both layouts, two name one.
    rows, both, one = _ten_k_rows()
    assert len(rows) == 10
    assert len(both) == 8
    assert one == ["c09.10k.partitioned.kept-shards", "c09.10k.round-robin.all-rewrite"]
    assert all(rows[listed]["given"]["layouts"] == ["partitioned", "round-robin"] for listed in both)
    counted = {"partitioned": PART10, "round-robin": ROUND_ROBIN10}
    for listed in both:
        got = _by_layout(listed, counted)
        assert list(got) == ["partitioned", "round-robin"], listed
        for layout, concrete in got.items():
            assert concrete.listed_row == listed and concrete.layout == layout, (listed, layout)
            assert concrete.size is None
        if listed != "c09.10k.action-counts":
            # The other seven rows state their values in words and numbers: the table's own.
            for concrete in got.values():
                assert dict(concrete.expect) == _fill(_written_expect(listed), _spec()), listed
                assert concrete.open == (), listed


def test_the_action_counts_differ_by_layout_in_the_shards_and_not_in_the_records():
    got = _by_layout("c09.10k.action-counts", {"partitioned": PART10, "round-robin": ROUND_ROBIN10})
    assert got["partitioned"].expect["counts"] == PARTITIONED_COUNTS
    assert got["round-robin"].expect["counts"] == ROUND_ROBIN_COUNTS
    assert got["partitioned"].expect["record_actions_allowed"] == ["keep", "re-source", "promote", "retire"]
    assert got["partitioned"].open == () and got["round-robin"].open == ()


def _assert_not_derived(slot, layout):
    """The slot of a layout with no counting document: the row's ``Concrete`` for that layout,
    its ``open`` naming the keys the document would give, the table's own words in ``expect``."""
    assert not isinstance(slot, Exception), f"{layout}: a missing document is returned as {slot!r}"
    assert slot.listed_row == "c09.10k.action-counts" and slot.layout == layout
    assert set(slot.open) == {"counts"}, layout
    assert dict(slot.expect) == _fill(_written_expect("c09.10k.action-counts"), _spec()), layout
    assert _beside(slot) == {}, layout


def test_a_layout_that_cannot_be_derived_is_named_and_the_other_layout_keeps_its_values():
    # No document for the round-robin layout: that layout holds the Concrete that reads
    # "expected value not derived", returned.
    got = _by_layout("c09.10k.action-counts", {"partitioned": PART10})
    assert got["partitioned"].expect["counts"] == PARTITIONED_COUNTS
    assert got["partitioned"].open == ()
    _assert_not_derived(got["round-robin"], "round-robin")
    # No document at all: both layouts are named, nothing is raised.
    none = _by_layout("c09.10k.action-counts")
    assert list(none) == ["partitioned", "round-robin"]
    for layout, slot in none.items():
        _assert_not_derived(slot, layout)
    # A document counted on the other layout is no answer for this one.
    wrong = _by_layout("c09.10k.action-counts", {"partitioned": ROUND_ROBIN10,
                                                 "round-robin": ROUND_ROBIN10})
    assert isinstance(wrong["partitioned"], ValueError)
    assert "layout" in str(wrong["partitioned"])
    assert wrong["round-robin"].expect["counts"] == ROUND_ROBIN_COUNTS


def test_a_row_that_names_no_counted_part_gives_its_values_with_no_document_at_all():
    _rows, both, _one = _ten_k_rows()
    for listed in both:
        if listed == "c09.10k.action-counts":
            continue
        got = _by_layout(listed)
        assert all(not isinstance(slot, Exception) for slot in got.values()), listed
        assert got["partitioned"].expect == got["round-robin"].expect, listed


def test_the_two_rows_of_one_layout_give_the_kept_shards_and_the_chunk_counts():
    concrete = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    kept = concrete(expected_for("c09.10k.partitioned.kept-shards", None, counted={None: PART10}))
    assert kept.layout == "partitioned"
    assert kept.expect["keep"] == ["shard-C", "shard-Z"]
    assert kept.expect["every_other_shard"] == "rewrite"
    assert kept.expect["kept_shards_byte_identical"] is True
    assert kept.open == ()
    rewrite = concrete(expected_for("c09.10k.round-robin.all-rewrite", None,
                                    counted={None: ROUND_ROBIN10}))
    assert rewrite.layout == "round-robin"
    assert dict(rewrite.expect) == {"rewrite": 20, "keep": 0}
    # A document counted on the other layout is refused, naming the layout.
    with pytest.raises(ValueError, match="layout"):
        concrete(expected_for("c09.10k.partitioned.kept-shards", None, counted={None: ROUND_ROBIN10}))
    with pytest.raises(ValueError, match="layout"):
        concrete(expected_for("c09.10k.round-robin.all-rewrite", None, counted={None: PART10}))


def test_a_row_that_names_two_layouts_is_not_read_as_one_value():
    concrete = need_attr("baselines.expected", "concrete_values")
    expected_for = need_attr("baselines.expected", "expected_for")
    with pytest.raises(ValueError, match="layouts"):
        concrete(expected_for("c09.10k.action-counts", None, counted={None: PART10}))
    by_layout = need_attr("baselines.expected", "concrete_values_by_layout")
    with pytest.raises(ValueError):
        by_layout(expected_for("c09.10k.partitioned.kept-shards", None, counted={None: PART10}))


# --- The whole-run diff rows take the oracle's log ------------------------------------------


def _spec_lines(name):
    return [json.loads(line) for line in _spec()["positions"][name]["log_lines"]]


def _whole_run(listed, oracle):
    whole_run_rows = need_attr("baselines.expected", "whole_run_rows")
    expected_for = need_attr("baselines.expected", "expected_for")
    return whole_run_rows(expected_for(listed), oracle)


def _plain_rows(got):
    return {sample: list(rows) for sample, rows in got.items()}


def test_the_whole_run_rows_of_run_e_are_the_oracles_rows_with_step_8_cut_at_its_covered_tokens():
    oracle = _spec_lines("skipped_line") + _spec_lines("T") + _spec_lines("step8_cut")
    cut = _spec()["positions"]["step8_cut"]
    got = _plain_rows(_whole_run("c01.E.diff", oracle))
    assert got == {
        ("E/seg-1", 1, 0): [(1, 17, 21, 50)],
        ("E/seg-2", 7, 0): [(2, 24, 32, 39), (2, 7, 0, 22), (2, 26, 0, 4)],
        (cut["fixed"]["segment"], cut["fixed"]["step"], cut["fixed"]["sample_pos"]): [
            (r["shard_position"], r["ordinal"], r["token_start"], r["token_end"])
            for r in cut["rows_of_the_covered_part"]
        ],
    }


def test_the_whole_run_rows_of_the_other_runs_have_no_cut_and_a_log_of_two_runs_is_split_by_run():
    k_lines = _spec_lines("K")
    got = _plain_rows(_whole_run("c01.K.diff", k_lines + _spec_lines("T")))
    # Only run K's own segment is read; both ranks are rows of the run.
    assert set(got) == {("K/seg-1", 15, 0), ("K/seg-1", 1, 0), ("K/seg-1", 8, 3)}
    assert got[("K/seg-1", 8, 3)] == [(0, 25, 45, 51), (0, 13, 0, 18), (0, 7, 0, 9)]
    rr = [
        {"segment": "E-rr/seg-1", "step": 2, "sample_pos": 1, "global_index": 5, "rank": 0,
         "shard_position": 1, "ordinal": 4, "token_start": 0, "token_end": 33},
        {"segment": "E-rr/seg-2", "step": 7, "sample_pos": 0, "global_index": 24, "rank": 0,
         "shard_position": 2, "ordinal": 9, "token_start": 3, "token_end": 12},
    ]
    assert _plain_rows(_whole_run("c01.E-rr.diff", rr + k_lines)) == {
        ("E-rr/seg-1", 2, 1): [(1, 4, 0, 33)],
        ("E-rr/seg-2", 7, 0): [(2, 9, 3, 12)],
    }


def test_the_whole_run_rows_refuse_a_row_that_is_not_a_whole_run_diff_and_a_log_without_the_run():
    whole_run_rows = need_attr("baselines.expected", "whole_run_rows")
    expected_for = need_attr("baselines.expected", "expected_for")
    with pytest.raises(ValueError):
        whole_run_rows(expected_for("c01.E.T"), _spec_lines("T"))
    with pytest.raises(ValueError):
        whole_run_rows(expected_for("c01.K.diff"), _spec_lines("T"))
    with pytest.raises(ValueError):
        whole_run_rows(expected_for("c01.K.diff"), [])


# --- The one id read at an ordinal, and the layouts combined into one line ---------------------

# Hand-written: the carried expectation of three shards. Shard-1 holds a group of two
# identical documents, shard-2 none, shard-3 a group of three.
ONE_ID_EXPECTED = {
    "shard-1": ["Z-001", ["Z-102", "Z-101"], "Z-003", ["Z-102", "Z-101"], "Z-005"],
    "shard-2": ["Z-006", "Z-007"],
    "shard-3": [["Z-301", "Z-302", "Z-303"], "Z-304", ["Z-301", "Z-302", "Z-303"]],
}
ONE_ID_ROWS = (
    "c03.records.capture", "c03.records.position-recovery", "c03.records.token-matching",
    "c06.generated.documents-mapped",
)
# The rows that compare the ordinals of a group; the token-matching row leaves them out.
GROUP_COMPARED_ROWS = tuple(listed for listed in ONE_ID_ROWS if listed != "c03.records.token-matching")
ONE_ID_ALL_RIGHT = {
    "shard-1": ["Z-001", "Z-101", "Z-003", "Z-102", "Z-005"],
    "shard-2": ["Z-006", "Z-007"],
    "shard-3": ["Z-303", "Z-304", "Z-301"],
}


def _one_id(listed, given, per_ordinal=None):
    one_id_reading = need_attr("baselines.expected", "one_id_reading")
    return one_id_reading(listed, ONE_ID_EXPECTED if per_ordinal is None else per_ordinal, given)


def _given(**changes):
    given = {shard: list(ids) for shard, ids in ONE_ID_ALL_RIGHT.items()}
    given.update({name.replace("_", "-"): ids for name, ids in changes.items()})
    return given


def test_one_id_at_an_ordinal_of_a_group_is_right_for_any_member_and_wrong_for_another_id():
    for listed in ONE_ID_ROWS:
        got = _one_id(listed, ONE_ID_ALL_RIGHT)
        assert got.right and got.differing == (), listed
    for listed in GROUP_COMPARED_ROWS:
        # Either member of a pair, and each of the three of a group, is right.
        for first, second in (("Z-101", "Z-102"), ("Z-102", "Z-101")):
            got = _one_id(listed, _given(shard_1=["Z-001", first, "Z-003", second, "Z-005"]))
            assert got.right, (listed, first, second)
        for member in ("Z-301", "Z-302", "Z-303"):
            assert _one_id(listed, _given(shard_3=[member, "Z-304", member])).right, member
        # An id that is not a member of the group is wrong, at that ordinal only.
        other = _one_id(listed, _given(shard_1=["Z-001", "Z-999", "Z-003", "Z-102", "Z-005"]))
        assert not other.right and other.differing == (("shard-1", 1),), listed
        # The same id as a member of ANOTHER group is no member of this one.
        elsewhere = _one_id(listed, _given(shard_3=["Z-101", "Z-304", "Z-301"]))
        assert elsewhere.differing == (("shard-3", 0),), listed


def test_one_id_at_an_ordinal_outside_every_group_is_right_only_when_it_is_that_id():
    for listed in ONE_ID_ROWS:
        got = _one_id(listed, _given(shard_2=["Z-007", "Z-006"]))
        assert not got.right, listed
        assert got.differing == (("shard-2", 0), ("shard-2", 1)), listed
        # The id of a group's member at an ordinal that is outside the group is wrong.
        swapped = _one_id(listed, _given(shard_1=["Z-102", "Z-102", "Z-003", "Z-102", "Z-005"]))
        assert swapped.differing == (("shard-1", 0),), listed


def test_the_token_matching_row_leaves_the_ordinals_of_a_group_out_and_compares_the_others():
    # A pair ordinal is not compared for c03.records.token-matching (the row's own expect:
    # "every document outside a same-shard pair"); every other row compares it.
    wrong_in_groups = _given(
        shard_1=["Z-001", "Z-999", "Z-003", None, "Z-005"], shard_3=["Z-998", "Z-304", "Z-997"],
    )
    left_out = _one_id("c03.records.token-matching", wrong_in_groups)
    assert left_out.right and left_out.differing == ()
    for listed in GROUP_COMPARED_ROWS:
        got = _one_id(listed, wrong_in_groups)
        assert got.differing == (
            ("shard-1", 1), ("shard-1", 3), ("shard-3", 0), ("shard-3", 2),
        ), listed
    # The ordinals outside a group are compared for the token-matching row too.
    other = _one_id("c03.records.token-matching", _given(shard_2=["Z-006", "Z-999"]))
    assert other.differing == (("shard-2", 1),)
    # The count of the ordinals compared: 10 in all, 4 of them in a group.
    assert _one_id("c03.records.capture", ONE_ID_ALL_RIGHT).compared == 10
    assert _one_id("c03.records.token-matching", ONE_ID_ALL_RIGHT).compared == 6


def test_a_gap_in_the_one_id_reading_is_never_read_as_right():
    for listed in ONE_ID_ROWS:
        # A shard the reading lacks: every ordinal of it differs.
        no_shard = {key: value for key, value in ONE_ID_ALL_RIGHT.items() if key != "shard-2"}
        assert _one_id(listed, no_shard).differing == (("shard-2", 0), ("shard-2", 1)), listed
        # A short list: the ordinals it lacks differ.
        short = _one_id(listed, _given(shard_2=["Z-006"]))
        assert short.differing == (("shard-2", 1),), listed
        # An id that is not one id (none, or a list) differs.
        odd = _one_id(listed, _given(shard_2=[None, ["Z-007"]]))
        assert odd.differing == (("shard-2", 0), ("shard-2", 1)), listed
        # An ordinal or a shard the expectation does not hold is a difference, not ignored.
        extra = _one_id(listed, _given(shard_2=["Z-006", "Z-007", "Z-008"], shard_9=["Z-900"]))
        assert not extra.right, listed
        assert set(extra.differing) == {("shard-2", 2), ("shard-9", 0)}, listed


def test_a_list_at_the_ordinal_of_a_group_is_compared_as_a_set():
    for listed in GROUP_COMPARED_ROWS:
        # A list equal to the candidate set, in any order, at every ordinal of a group.
        got = _one_id(listed, _given(
            shard_1=["Z-001", ["Z-101", "Z-102"], "Z-003", ["Z-102", "Z-101"], "Z-005"],
            shard_3=[["Z-303", "Z-301", "Z-302"], "Z-304", ["Z-302", "Z-303", "Z-301"]],
        ))
        assert got.right and got.differing == (), listed
        # Any other list is wrong, at that ordinal only: part of the set, more than the set,
        # a stranger among its members, an empty list, and the set of another group.
        for other in (["Z-102"], ["Z-102", "Z-101", "Z-999"], ["Z-102", "Z-999"], [],
                      ["Z-301", "Z-302", "Z-303"]):
            got = _one_id(listed, _given(shard_1=["Z-001", other, "Z-003", "Z-102", "Z-005"]))
            assert not got.right and got.differing == (("shard-1", 1),), (listed, other)
        # One id stays right for any member of the group and wrong for another id.
        one = _one_id(listed, _given(shard_1=["Z-001", "Z-102", "Z-003", "Z-101", "Z-005"]))
        assert one.right, listed
        # A list at an ordinal outside every group is not a set to compare: it differs.
        outside = _one_id(listed, _given(shard_2=[["Z-006"], "Z-007"]))
        assert outside.differing == (("shard-2", 0),), listed
    # The ordinals of a group are not compared for the token-matching row, lists included.
    left_out = _one_id("c03.records.token-matching", _given(shard_1=["Z-001", ["Z-999"], "Z-003", [], "Z-005"]))
    assert left_out.right and left_out.differing == ()


def test_the_one_id_reading_refuses_a_row_that_has_no_record_per_ordinal():
    need_attr("baselines.expected", "one_id_reading")
    with pytest.raises(ValueError, match="c03.pairs.split"):
        _one_id("c03.pairs.split", ONE_ID_ALL_RIGHT)
    with pytest.raises(ValueError, match="c99.nothing"):
        _one_id("c99.nothing", ONE_ID_ALL_RIGHT)


def test_the_one_id_reading_takes_the_expectation_the_concrete_values_carry():
    # The expectation of a row as concrete_values gives it (the groups carried) is the one
    # the reading is made against: the hand-written document of the groups above.
    one_id_reading = need_attr("baselines.expected", "one_id_reading")
    got = _concrete("c03.records.capture", SIZED_SMALL, counted={SIZED_SMALL: GROUPS_DOC})
    given = {
        "shard-1": ["Z-001", "Z-101", "Z-003", "Z-102", "Z-005"],
        "shard-2": ["Z-303", "Z-301", "Z-006", "Z-302"],
    }
    assert one_id_reading("c03.records.capture", got.expect["per_ordinal"], given).right
    given["shard-2"][2] = "Z-302"
    assert one_id_reading("c03.records.capture", got.expect["per_ordinal"], given).differing == (
        ("shard-2", 2),
    )


LAYOUT_ROW = "c09.10k.multiset"
LAYOUT_NAMES = ["partitioned", "round-robin"]


def _grade(status, evidence=None):
    return need_attr("baselines.interfaces", "Grade")(status=status, evidence=evidence)


def _combined(results, listed=LAYOUT_ROW):
    combine_layouts = need_attr("baselines.expected", "combine_layouts")
    return combine_layouts(listed, results)


# The row whose layout has a counting document to miss (its expectation reads the document).
DERIVED_ROW = "c09.10k.action-counts"


def _underived(layout):
    """The result of a layout whose counting document is missing: the ``Concrete`` of that
    layout whose ``open`` names the key the document would give (read from no document at all)."""
    slot = _by_layout(DERIVED_ROW)[layout]
    _assert_not_derived(slot, layout)
    return slot


def _line_not_derived(got, layout):
    """The slot of a layout in a combined line: read "expected value not derived". The
    text of its evidence is not pinned here."""
    assert got.evidence[layout]["status"] == "expected value not derived"


def test_a_line_asked_in_both_layouts_is_right_only_when_every_layout_is_right():
    # The layouts the line is asked in are the row's own, from the table.
    assert _listed_rows()[LAYOUT_ROW]["given"]["layouts"] == LAYOUT_NAMES
    got = _combined({"partitioned": _grade("right"), "round-robin": _grade("right")})
    assert got.status == "right"
    assert list(got.evidence) == LAYOUT_NAMES
    assert all(slot["status"] == "right" for slot in got.evidence.values())


def test_a_line_is_wrong_when_any_layout_reads_wrong_even_if_the_other_cannot_be_graded():
    right, wrong = _grade("right"), _grade("wrong", ["multisets"])
    cannot = _grade("cannot answer", "the record lacks the field")
    # A layout whose counting document is missing: the Concrete that reads "not derived".
    underived = {layout: _underived(layout) for layout in LAYOUT_NAMES}
    for results in (
        {"partitioned": wrong, "round-robin": right},
        {"partitioned": right, "round-robin": wrong},
        {"partitioned": wrong, "round-robin": wrong},
        {"partitioned": wrong, "round-robin": cannot},
        {"partitioned": cannot, "round-robin": wrong},
        {"partitioned": wrong, "round-robin": underived["round-robin"]},
        {"partitioned": underived["partitioned"], "round-robin": wrong},
        {"partitioned": wrong, "round-robin": None},
        {"round-robin": wrong},
    ):
        got = _combined(results, listed=DERIVED_ROW)
        assert got.status == "wrong", results
    # The evidence is per layout, in the row's order, and keeps what each layout showed.
    got = _combined({"round-robin": wrong, "partitioned": right})
    assert list(got.evidence) == LAYOUT_NAMES
    assert got.evidence["round-robin"] == {"status": "wrong", "evidence": ["multisets"]}
    assert got.evidence["partitioned"] == {"status": "right", "evidence": None}


def test_a_line_that_is_neither_right_nor_wrong_cannot_answer_and_names_each_layout_left_open():
    right = _grade("right")
    cannot = _grade("cannot answer", "the record lacks the field")
    # A layout whose subject could not answer.
    got = _combined({"partitioned": right, "round-robin": cannot})
    assert got.status == "cannot answer"
    assert got.evidence["round-robin"] == {"status": "cannot answer",
                                           "evidence": "the record lacks the field"}
    assert got.evidence["partitioned"]["status"] == "right"
    # Both layouts could not answer.
    both = _combined({"partitioned": cannot, "round-robin": cannot})
    assert both.status == "cannot answer"
    assert [slot["status"] for slot in both.evidence.values()] == ["cannot answer"] * 2


def test_a_layout_whose_expected_value_is_not_derived_leaves_the_line_open_and_is_named():
    right = _grade("right")
    # The expected value of a layout could not be derived (the Concrete that
    # concrete_values_by_layout returns when its counting document is missing): the line
    # cannot answer, and the evidence says the value is missing, which is told apart from a
    # subject that could not answer.
    got = _combined({"partitioned": right, "round-robin": _underived("round-robin")},
                    listed=DERIVED_ROW)
    assert got.status == "cannot answer"
    _line_not_derived(got, "round-robin")
    assert got.evidence["partitioned"]["status"] == "right"


def test_a_layout_with_no_result_is_not_attempted_and_so_is_the_line():
    right = _grade("right")
    cannot = _grade("cannot answer", "the record lacks the field")
    # A result absent for a layout the row names, or None, is that layout NOT ATTEMPTED: the
    # line and the slot both read "not attempted", and each layout is named.
    for results in ({"partitioned": right}, {"partitioned": right, "round-robin": None}):
        missing = _combined(results)
        assert missing.status == "not attempted", results
        assert list(missing.evidence) == LAYOUT_NAMES
        assert missing.evidence["round-robin"]["status"] == "not attempted"
        assert missing.evidence["partitioned"]["status"] == "right"
    first = _combined({"round-robin": right})
    assert first.status == "not attempted"
    assert first.evidence["partitioned"]["status"] == "not attempted"
    assert first.evidence["round-robin"]["status"] == "right"
    # Nothing attempted: every layout is named "not attempted" and the line is too.
    empty = _combined({})
    assert empty.status == "not attempted"
    assert [slot["status"] for slot in empty.evidence.values()] == ["not attempted"] * 2
    # Not attempted comes before cannot answer, and after a wrong line and a failed attempt.
    assert _combined({"partitioned": cannot, "round-robin": None}).status == "not attempted"
    assert _combined({"partitioned": _grade("wrong", ["multisets"])}).status == "wrong"
    failed = _combined_with_states({"partitioned": _failed("a1"), "round-robin": None})
    assert failed.status == _failed("a1")
    assert failed.evidence["round-robin"]["status"] == "not attempted"
    # Not attempted comes before a layout whose expected value is not derived.
    got = _combined({"partitioned": _underived("partitioned"), "round-robin": None},
                    listed=DERIVED_ROW)
    assert got.status == "not attempted"
    _line_not_derived(got, "partitioned")
    assert got.evidence["round-robin"]["status"] == "not attempted"


NOT_ATTEMPTED = "not attempted"


def _failed(attempt_id):
    """The status of a question whose attempt failed, as the table's words give it."""
    return "no answer: failed attempt " + attempt_id


def _combined_with_states(results, listed=LAYOUT_ROW):
    """``combine_layouts`` for results that hold an attempt state or the Concrete of a layout
    whose counting document is missing; a refusal of one is a failed assertion that says so."""
    try:
        return _combined(results, listed=listed)
    except (AttributeError, TypeError) as problem:
        pytest.fail(f"combine_layouts does not take this as a layout's result: {problem!r}")


def test_a_line_takes_the_order_wrong_then_failed_attempt_then_not_attempted_then_right_then_cannot_answer():
    right = _grade("right")
    wrong = _grade("wrong", ["multisets"])
    cannot = _grade("cannot answer", "the record lacks the field")
    # A layout whose counting document is missing; the Concrete of the layout it stands in.
    underived = object()
    one, two = _failed("a1"), _failed("a2")
    # (result of the partitioned layout, result of the round-robin layout, the line's status)
    cases = (
        (wrong, one, "wrong"), (one, wrong, "wrong"),
        (wrong, NOT_ATTEMPTED, "wrong"), (NOT_ATTEMPTED, wrong, "wrong"),
        (right, two, two), (one, right, one),
        (cannot, two, two), (underived, one, one),
        (one, NOT_ATTEMPTED, one), (NOT_ATTEMPTED, two, two),
        (right, NOT_ATTEMPTED, NOT_ATTEMPTED), (NOT_ATTEMPTED, right, NOT_ATTEMPTED),
        (cannot, NOT_ATTEMPTED, NOT_ATTEMPTED), (NOT_ATTEMPTED, underived, NOT_ATTEMPTED),
        (NOT_ATTEMPTED, NOT_ATTEMPTED, NOT_ATTEMPTED),
        (right, right, "right"), (right, cannot, "cannot answer"), (underived, right, "cannot answer"),
    )
    for first, second, status in cases:
        results = {
            layout: _underived(layout) if result is underived else result
            for layout, result in zip(LAYOUT_NAMES, (first, second))
        }
        got = _combined_with_states(results, listed=DERIVED_ROW)
        assert got.status == status, (first, second)
        # The evidence names each layout, in the row's order.
        assert list(got.evidence) == LAYOUT_NAMES, (first, second)
        for layout, result in zip(LAYOUT_NAMES, (first, second)):
            if isinstance(result, str):
                assert got.evidence[layout]["status"] == result, (first, second, layout)
            elif result is underived:
                _line_not_derived(got, layout)


def test_two_failed_attempts_give_the_line_the_status_of_one_of_them_and_the_evidence_names_both():
    one, two = _failed("a1"), _failed("a2")
    got = _combined_with_states({"partitioned": one, "round-robin": two})
    assert got.status in (one, two)
    assert [slot["status"] for slot in got.evidence.values()] == [one, two]


def test_two_failed_attempts_give_the_line_the_status_of_the_first_layout_in_the_rows_order():
    one, two = _failed("a1"), _failed("a2")
    # The id of the first layout of the row (partitioned), whichever way the mapping is given;
    # both ids are in the evidence, in the row's order.
    for results in (
        {"partitioned": one, "round-robin": two},
        {"round-robin": two, "partitioned": one},
    ):
        got = _combined_with_states(results)
        assert got.status == one, results
        assert [slot["status"] for slot in got.evidence.values()] == [one, two]
    # The other way round: the first layout's id is still the line's.
    got = _combined_with_states({"partitioned": two, "round-robin": one})
    assert got.status == two
    assert [slot["status"] for slot in got.evidence.values()] == [two, one]


def test_a_result_that_is_not_a_grade_an_error_none_or_an_attempt_state_is_refused():
    right = _grade("right")
    for bad in ("maybe", "not attempted yet", 7):
        with pytest.raises((TypeError, ValueError)):
            _combined({"partitioned": right, "round-robin": bad})


def test_a_not_gradable_is_not_a_layout_result_and_is_never_labelled_expected_value_not_derived():
    not_gradable = need_attr("baselines.grade", "NotGradable")
    right = _grade("right")
    assert issubclass(not_gradable, ValueError)
    for results in (
        {"partitioned": right, "round-robin": not_gradable("attempt a1 failed")},
        {"partitioned": not_gradable("not a reading"), "round-robin": right},
    ):
        with pytest.raises((TypeError, ValueError)):
            _combined(results)
    # The control: a plain ValueError (a document counted on the other layout, the slot
    # concrete_values_by_layout returns) is still a layout whose expected value could not be
    # derived, and the line cannot answer.
    other_layout = ValueError("the counting document was counted on the other layout")
    got = _combined({"partitioned": right, "round-robin": other_layout})
    assert got.status == "cannot answer"
    assert got.evidence["round-robin"]["status"] == "expected value not derived"
    assert "other layout" in got.evidence["round-robin"]["evidence"]


def test_the_layouts_are_combined_only_for_a_row_that_names_them_and_only_those_it_names():
    right = _grade("right")
    with pytest.raises(ValueError, match="layouts"):
        _combined({"partitioned": right}, listed="c09.10k.partitioned.kept-shards")
    with pytest.raises(ValueError, match="layouts"):
        _combined({"partitioned": right}, listed="c03.records.capture")
    with pytest.raises(ValueError, match="mirrored"):
        _combined({"partitioned": right, "mirrored": right})


# --- What these modules import and hold ---------------------------------------

MODULES = ("baselines.expected", "baselines.grade", "baselines.record")


def test_the_three_modules_import_nothing_of_the_tool_nor_the_world_helpers():
    for name in MODULES:
        path = getattr(need(name), "__file__", None)
        assert path, f"{name} has no file"
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".")[0])
        assert not roots & FORBIDDEN_IMPORT_ROOTS, f"{name} imports {sorted(roots & FORBIDDEN_IMPORT_ROOTS)}"


def test_the_three_modules_hold_no_local_path_and_no_em_dash():
    for name in MODULES:
        path = Path(need(name).__file__)
        text = path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_TEXT:
            assert bad not in text, f"{name} holds {bad!r}"
