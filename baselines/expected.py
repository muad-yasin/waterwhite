"""The expected answers of the listed rows of Checks 1 to 9 (plan 5.5 Check 10; table
README, section "Grades, and what is compared").

An expected answer is the ``expect`` of the row asked, or the entry of its
``expect_under`` for the fallback in force, with every ``pending:<key>`` replaced by
that item of the fixture spec. Nothing here is computed from the tool or from a
replica build: the values are read from the hand-written table, the fixture spec, a
counting document the plant script printed, the oracle's log, or the sizes in a shard's own
index file (``index_of_shards``, with the replica's reader of that file).

The pure readers of the acceptance tests are reused (they import nothing of the
tool at module level): the table reader and the pending filler of ``table.py``, the
join's expected rows of ``acceptance_join.py`` and the pair lists of
``acceptance_sets.py``. This module imports no other acceptance module.

Sizes are the tool's own: ``baselines.table10`` works them out per listed row, with
no fallback in force, and a row is expected only at a size it is asked at.

Some expected fields of the table are sentences that name a fact, not the fact.
``concrete_values`` puts the concrete value in their place, read from the source the
row names (the counting document, the shard's own index, the oracle's log), the way the
tool's own row derives it. A counted value the row's expectation holds no sentence for
(a number or a word the table states itself, or a key it does not hold) is kept beside
the expectation in ``counted_values`` and never replaces the table's own. A sentence is
never compared as a string: it is derived as a concrete value, or it is one of the
table's not-compared parts (``Concrete.not_compared``), or it stays as the table wrote
it and is listed in ``fields_without_concrete_value`` (read "expected value not derived").
A row whose source names a counting document and that is asked with none given reads
"expected value not derived" for every key the document would give (``Concrete.open``);
the table's own words in those keys never stand in for a value.
``index_of_shards`` reads the shard index files the replica writes into the sizes of the
documents the index rows take.

Two rules of the expectation are kept here, as pure functions of values the caller passes.
``one_id_reading`` compares one id per ordinal with the carried expectation, where an
ordinal of a group of identical documents accepts any member of the group, or the whole
group as a set. ``combine_layouts`` puts the results of a line asked in both layouts
into the one line of the row.
"""

from __future__ import annotations

import importlib
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from baselines import table10
from baselines.interfaces import CANNOT_ANSWER, GRADES, NO_ANSWER_PREFIX, RIGHT, WRONG, Grade
from baselines.record import NOT_ATTEMPTED
from replica.shard_index import read_shard_index

# The key of an expectation that says the row does not apply under the fallback in
# force: such a row is not put.
APPLIES_KEY = "applies"

# The key of ``counted`` under which a row asked once reads its counting document.
ONCE = None


@dataclass(frozen=True)
class Expected:
    """The expected answer of one listed row at one size.

    listed_row: the id of the row of Checks 1 to 9.
    size: None for a row asked once, else ``small``, ``10k`` or ``100k``.
    applies: False when the fallback in force says the row does not apply.
    expect: the expectation in force with its pending values filled; an empty
        mapping when the row does not apply. A value the spec could not fill is left
        as the table wrote it, and its key is in ``unfilled``.
    unfilled: the pending keys whose spec item is missing or empty.
    counted: the counting document of this size's set, or None when none was given.
    index: the sizes of the documents in each shard's own index for this size's set
        (shard name to a list, in ordinal order), or None when none was given.
    by_layout: for a row whose ``given`` names two layouts, the counting document of
        each layout as the caller passed it (layout name to document), else None.
    """

    listed_row: str
    size: str | None
    applies: bool
    expect: Mapping[str, Any]
    unfilled: tuple[str, ...]
    counted: Any
    index: Any = None
    by_layout: Mapping[str, Any] | None = None


# --- What is read from the files ------------------------------------------------


# The modules of ``tests/acceptance`` the graders load: the table reader (``table10._acceptance_table``) and the two helpers
# this module asks for by name (``_acceptance_module``). The seal of a record (``runner.frozen_files_hashes``) hashes the file of
# each, so a record made with other helpers cannot be re-graded with these (18 Nov 2026: the seal names every acceptance module
# the graders load; a test reads the source of this module and of table10 for a module not named here).
ACCEPTANCE_MODULES = ("table", "acceptance_join", "acceptance_sets")


def _acceptance_module(name: str):
    """A module of ``tests/acceptance`` by its bare name; ``table10`` puts that
    folder on the path when it loads the table reader."""
    table10._acceptance_table()
    return importlib.import_module(name)


@lru_cache(maxsize=1)
def _core_rows() -> Mapping[str, Any]:
    """Every row of Checks 1 to 9, by id: the core rows and the rows of an extension (the name stays for the
    callers). A pass asks a row of an extension only while that extension is on (``table10.expand``), so the
    expected value of such a row is read like any other; a row no group lists is still no question (``_sizes``)."""
    table = table10._acceptance_table()
    found = {}
    for number in table10.LISTED_CHECKS:
        for row in table.load_check(number, table10.DEFAULT_PATH.parent):
            found[row.id] = row
    return found


@lru_cache(maxsize=1)
def _sizes() -> Mapping[str, tuple[str | None, ...]]:
    return table10.sizes_of_listed_rows(table10.DEFAULT_PATH.parent)


@lru_cache(maxsize=1)
def _fixture_spec() -> Mapping[str, Any]:
    with open(table10.REPO / "fixture" / "spec" / "fixture_spec.toml", "rb") as handle:
        return tomllib.load(handle)


# --- The expected answer of a listed row ----------------------------------------


def expected_for(
    listed_row: str,
    size: str | None = None,
    *,
    fallbacks: Sequence[str] = (),
    spec: Mapping[str, Any] | None = None,
    counted: Mapping[str | None, Any] | None = None,
    index: Mapping[str | None, Any] | None = None,
) -> Expected:
    """The expected answer of ``listed_row`` asked at ``size``.

    ``size`` is None for a row asked once. ``fallbacks`` are the fallbacks in force
    (they change the expectation, never the sizes). ``spec`` is the fixture spec to
    fill pending values from; None reads the committed one. ``counted`` maps a size
    to the counting document of that set; a sized row whose size it holds no document
    for holds none (its value reads "expected value not derived"). A row asked once reads the document under the key None, else
    the one of the set its ``given`` names, else holds none. ``index`` maps a size to
    the sizes of the documents in each shard's own index, and is read the same way.

    For a row whose ``given`` names two layouts, ``counted`` may map each layout to the
    document counted on that layout; the mapping is kept as ``by_layout``.

    Raises ``KeyError`` for a row the table does not hold among the core rows, and
    ``ValueError`` for a size the row is not asked at.
    """
    rows = _core_rows()
    if listed_row not in rows:
        raise KeyError(listed_row)
    asked = _sizes()[listed_row]
    if size not in asked:
        raise ValueError(f"{listed_row} is asked at {list(asked)}, not at {size!r}")
    table = table10._acceptance_table()
    row = rows[listed_row]
    entry = table.expect_in_force(row, tuple(fallbacks))
    document = _fixture_spec() if spec is None else spec
    held = _of_size(row, size, counted)
    sizes_held = _of_size(row, size, index)
    layouts = counted if "layouts" in row.given else None
    if entry.get(APPLIES_KEY) is False:
        return Expected(listed_row, size, False, {}, (), held, sizes_held, layouts)
    filled, unfilled = table.fill_pending(entry, document)
    return Expected(listed_row, size, True, filled, tuple(unfilled), held, sizes_held, layouts)


def _of_size(row: Any, size: str | None, held: Mapping[str | None, Any] | None) -> Any:
    """The item of a mapping from a size that belongs to ``row`` asked at ``size``."""
    if held is None:
        return None
    if size is not ONCE:
        # A size the mapping holds nothing for is none held: the value reads "expected value
        # not derived" (a counting document), or is named as a missing index.
        return held.get(size)
    # A row asked once reads the item under None, else the one of the set its own
    # ``given`` names (the rows of Check 9 asked at 10k by their ids).
    found = held.get(ONCE)
    if found is None:
        found = held.get(row.given.get("set"))
    return found


# --- The oracle's log -----------------------------------------------------------


def run_rows(oracle: Sequence[Mapping[str, Any]], cuts: Mapping[Any, int] | None = None):
    """The rows the join must report for an oracle log: a mapping from (segment,
    step, sample position) to the rows of that sample, each (shard position, ordinal,
    token start, token end), in the order the log gives them. ``cuts`` maps a sample
    to the tokens read of it: the rows that end within them stay, the row that holds
    the next token ends at the cut, no later row is reported. A cut on a sample the
    log does not hold is a ``ValueError``."""
    join = _acceptance_module("acceptance_join")
    return join.expected_rows(oracle, cuts)


# --- A counting document of the plant script ------------------------------------


def same_shard_pairs(counted: Mapping[str, Any]) -> dict[str, list[dict]]:
    """The identical pairs whose two members lie in one shard, by shard."""
    return _acceptance_module("acceptance_sets").same_shard_pairs(counted)


def split_pairs(counted: Mapping[str, Any]) -> list[dict]:
    """The identical pairs whose members lie in different shards."""
    return _acceptance_module("acceptance_sets").split_pairs(counted)


def records_by_ordinal(counted: Mapping[str, Any]) -> dict[str, list[str]]:
    """For each shard, the expected record of each chunk-file line, in ordinal order.

    An ordinal outside every group of identical documents reads as the one planted id
    of its line. An ordinal inside a group (the shard's ``identical`` list) reads as
    the candidate set of the whole group: the list of its ids, in the group's order,
    the same list at each ordinal of the group."""
    documents = counted.get("documents")
    if not documents:
        raise ValueError("the counting document holds no documents part")
    found: dict[str, list] = {}
    for entry in documents["shards"]:
        expected = list(entry["ids"])
        for group in entry.get("identical") or ():
            for ordinal in group["ordinals"]:
                expected[ordinal] = list(group["ids"])
        found[entry["shard"]] = expected
    return found


def record_action_counts(counted: Mapping[str, Any]) -> dict[str, int]:
    """The number of records per rebuild action, as the plant script counts them."""
    return dict(counted["actions"]["records"])


# --- The shard's own index ----------------------------------------------------------


def index_of_shards(files: Mapping[str, Path | str]) -> dict[str, list[int]]:
    """The sizes of the documents in each shard's own index, read with the replica's reader
    of a shard index: ``files`` maps a shard name to the path of its ``.idx`` file, the
    result maps it to the token count of each document in ordinal order (the form
    ``expected_for`` takes as ``index``). The shards keep the order of ``files``. A file that
    is not a shard index is the reader's error, which names the file; it is never read as a
    shard with no documents."""
    return {
        shard: list(read_shard_index(Path(path)).document_tokens) for shard, path in files.items()
    }


# --- What a row reads of the counting document -------------------------------------


class PartMissing(ValueError):
    """The counting document, given, does not hold a part the row reads (no chunk files,
    no sanitizer input), or no index was given for a row that reads one. It is never read
    as "none". A row asked with no counting document at all is not this error: it reads
    "expected value not derived"."""


def _part(counted: Mapping[str, Any] | None, row: str, *path: str) -> Any:
    """The part of the document at ``path``, or ``PartMissing`` naming it."""
    name = ".".join(path)
    found: Any = counted
    for key in path:
        if not isinstance(found, Mapping) or found.get(key) is None:
            raise PartMissing(f"{row} reads {name}, which the counting document does not hold")
        found = found[key]
    return found


def _documents_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    _part(counted, row, "documents")
    return {"per_ordinal": records_by_ordinal(counted)}


# The three counts per shard that the documents-mapped row names beside the line each tied
# row names.
COUNT_DOCUMENTS = "documents_per_shard"
COUNT_ONE_TIED_ROW = "documents_with_one_tied_row_per_shard"
COUNT_NO_TIED_ROW = "documents_with_no_tied_row_per_shard"


def _documents_mapped_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    """The per-ordinal records and, per shard, the three counts, from the ``documents`` field
    each shard of the counting document's ``documents`` part gives (the surviving chunk-file
    lines). Every document of a shard has exactly one tied row, a member of a group of
    identical documents having its own companion row tied by its own src and loc (the
    table's README), so the number with exactly one tied row is the number of documents and
    the number with none is 0. The map labels of the part (``labels``) are the tool's own
    and no tie, so they are not read. A shard that gives no document count leaves the three
    counts out of the answer, so they read "expected value not derived"; a shard's counts
    are never guessed."""
    found = _documents_values(counted, row)
    shards = counted["documents"]["shards"]
    if any(entry.get("documents") is None for entry in shards):
        return found
    documents = {entry["shard"]: entry["documents"] for entry in shards}
    found[COUNT_DOCUMENTS] = documents
    found[COUNT_ONE_TIED_ROW] = dict(documents)
    found[COUNT_NO_TIED_ROW] = {shard: 0 for shard in documents}
    return found


def _drops_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    shards = _part(counted, row, "check3", "drops")["shards"]
    drops = [
        {
            "shard": drop["shard"], "id": drop["id"], "reason": drop["reason"],
            "input_line": drop["input_line"],
            "ordinal_of_next_line": drop["ordinal_of_next_line"],
        }
        for entry in shards for drop in entry["drops"]
    ]
    return {
        "drops": drops,
        "documents_per_shard": {entry["shard"]: entry["surviving_lines"] for entry in shards},
    }


def _same_shard_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    _part(counted, row, "check3", "pairs")
    return {"candidate_set": [
        {"pair": pair["pair"], "shard": shard, "lines": [m["line"] for m in pair["members"]]}
        for shard, pairs in same_shard_pairs(counted).items() for pair in pairs
    ]}


def _split_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    _part(counted, row, "check3", "pairs")
    return {"record": [
        {"pair": pair["pair"], "members": [
            {"id": m["id"], "shard": m["shard"], "line": m["line"]} for m in pair["members"]
        ]}
        for pair in split_pairs(counted)
    ]}


def _candidates_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    return {"candidate_sources": [
        {"release": entry["release"], "pair": entry["pair"], "sources": list(entry["sources"])}
        for entry in _part(counted, row, "check6", "candidates")
    ]}


def _translations_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    held = _part(counted, row, "check6", "translations")
    shards = _part(counted, row, "check6", "translations", "shards")
    return {"translations": {
        "file": held["file"],
        "releases": [name for name, count in held["releases"].items() if count > 0],
        "shards": [entry["shard"] for entry in shards],
    }}


# The placement words and the kind of pair the counting script counts (plant document,
# section 7).
_SAME_SHARD, _SPLIT, _PAIR_MEMBER = "same-shard", "split", "pair-member"
_NO_TOKENS = "no tokens"
_RECORD_ACTIONS = ("keep", "re-source", "promote", "retire")


def _pair_kinds_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    counts = _part(counted, row, "check3", "pairs", "counts")
    placed = counts.get(_PAIR_MEMBER)
    if placed is None:
        raise PartMissing(f"{row} reads check3.pairs.counts.{_PAIR_MEMBER}, which the counting document does not hold")
    return {"same_shard_pairs": placed[_SAME_SHARD], "split_pairs": placed[_SPLIT]}


def _no_tokens_values(counted: Mapping[str, Any], row: str) -> dict[str, Any]:
    shards = _part(counted, row, "check3", "drops")["shards"]
    documents = _part(counted, row, "documents")["shards"]
    return {
        "drops_with_reason_no_tokens": sum(
            1 for entry in shards for drop in entry["drops"] if drop["reason"] == _NO_TOKENS
        ),
        "documents_per_shard": {entry["shard"]: entry["documents"] for entry in documents},
        "surviving_lines_per_shard": {entry["shard"]: entry["surviving_lines"] for entry in shards},
    }


def _shard_lists(
    counted: Mapping[str, Any], row: str, layout: str | None = None
) -> tuple[list[str], list[str]]:
    shards = _part(counted, row, "shards")
    # Each part is counted on the build the row is asked of: a row that names one
    # layout does not read a document counted on the other. A row that names two is
    # read per layout, which the caller names.
    wanted = layout if layout is not None else _core_rows()[row].given.get("layout")
    if wanted is not None and shards.get("layout") != wanted:
        raise ValueError(
            f"{row} is asked on the {wanted} layout; the counting document was counted on "
            f"the {shards.get('layout')} layout"
        )
    return sorted(shards["keep"]), sorted(shards["rewrite"])


def _action_counts_values(
    counted: Mapping[str, Any], row: str, layout: str | None = None
) -> dict[str, Any]:
    records = _part(counted, row, "actions", "records")
    releases = _part(counted, row, "releases")
    keep, rewrite = _shard_lists(counted, row, layout)
    return {"counts": {
        "records": {word: records[word] for word in _RECORD_ACTIONS},
        "records_removed": sum(release["removed"] for release in releases.values()),
        "shards": {"keep": len(keep), "rewrite": len(rewrite)},
    }}


def _kept_shards_values(
    counted: Mapping[str, Any], row: str, layout: str | None = None
) -> dict[str, Any]:
    keep, rewrite = _shard_lists(counted, row, layout)
    return {"keep": keep, "rewrite": rewrite}


def _shard_counts_values(
    counted: Mapping[str, Any], row: str, layout: str | None = None
) -> dict[str, Any]:
    keep, rewrite = _shard_lists(counted, row, layout)
    return {"keep": len(keep), "rewrite": len(rewrite)}


# --- The concrete value of a sentence-valued field ------------------------------

# The fields of the table whose expected value is a sentence that names a fact, by listed
# row (19 rows, 23 row and key pairs). The one public source of them: a field that still
# holds the sentence is never compared as a string, whichever way the expectation reaches a
# grade. ``concrete_values`` puts a value in the place of each, leaves it as one of the
# not-compared parts of the table, or lists it as not derived.
SENTENCE_FIELDS: Mapping[str, frozenset[str]] = MappingProxyType({
    listed: frozenset(keys) for listed, keys in {
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
    }.items()
})

# Two of those fields hold a status or answer word of the table, not a sentence that
# stands for a value: the word stays as written and is compared as the word it is.
WORD_FIELDS = frozenset({
    ("c02.iii.no-replay", "replay"),
    ("c06.generated.candidates", "answer_for_each_candidate_source"),
})

# The seven fields whose sentence no counted document, index or oracle's log turns into a
# value. None is ever compared as a string; each is in exactly one of two classes.
#
# Not compared (the table's README, section "Grades, and what is compared": "the failing
# link, the chain, the scope"): the field stays as written, is not listed as open, and is
# named in ``Concrete.not_compared``. ``missing_link`` is the failing link of the import
# row, whose reader skips the row until the rebuild report exists; ``digest_covers``
# states the scope of a shuffle digest.
_NOT_COMPARED: Mapping[str, tuple[str, ...]] = {
    "c06.import.review-scope": ("missing_link",),
    "c08.K.digests-match": ("digest_covers",),
}

# Read "expected value not derived" until a concrete value exists: the field stays as the
# table wrote it and is listed by ``fields_without_concrete_value``. Each needs a record
# that has not been made (the run record of the replay and of run K, the state of a store
# after a refused write, the cache folder a corrected run is built from).
_NOT_DERIVED: Mapping[str, tuple[str, ...]] = {
    "c02.ii.replay-rejected": ("digests",),
    "c04.declare-499.write": ("map_after",),
    "c08.K.digests-match": ("shuffle_digests", "epoch_0_digests_of_launch_0_and_launch_1"),
    "c09.E.stale-cache.E/seg-2": ("rebuilt_from",),
}


@dataclass(frozen=True)
class Concrete:
    """The expectation of a listed row with each sentence-valued field replaced by its
    concrete value.

    layout: the layout the value was derived for (the row's own, or the one asked for a
        row that names two), or None.
    expect: the expectation in force; a sentence field a source gave a value for holds
        it, every other key is the table's own.
    open: the keys read "expected value not derived": each stays as the table wrote it, or
        is not in ``expect`` at all when the table holds no such key. A row whose source names
        a counting document and that was given none has every key the document would give
        here, and no counted value beside its expectation.
    counted_values: the counted values the row's expectation holds no sentence for (a key
        it does not hold, or a number or a word the table states itself), by key; they
        never replace the table's own.
    not_compared: the keys of ``expect`` that are one of the table's not-compared parts
        and still hold the table's words.
    """

    listed_row: str
    size: str | None
    layout: str | None
    expect: Mapping[str, Any]
    open: tuple[str, ...]
    counted_values: Mapping[str, Any] = field(default_factory=dict)
    not_compared: tuple[str, ...] = ()


def _sizes_per_shard(index: Mapping[str, Sequence[int]], row: str, layout: str | None):
    return {"per_ordinal": {shard: list(sizes) for shard, sizes in index.items()}}


def _total_per_shard(index: Mapping[str, Sequence[int]], row: str, layout: str | None):
    return {"per_shard": {shard: sum(sizes) for shard, sizes in index.items()}}


def _no_layout(read: Callable[[Mapping[str, Any], str], dict[str, Any]]):
    """A reader of a counting document that takes no layout."""
    return lambda counted, row, layout: read(counted, row)


# The rows that read a source, with where each reads its value: ``(source, part, reader,
# keys)``. ``source`` is the counting document or the shard's own index; ``part`` names what
# is missing when no index was given; ``keys`` are the keys the row's reader gives. A reader
# returns values by key; a key whose (row, key) is a sentence field replaces the sentence,
# every other key is kept beside the expectation. A key the source does not give is read
# "expected value not derived": with no counting document, that is every key of the row.
_CONCRETE: Mapping[str, tuple[str, str, Callable[..., dict[str, Any]], tuple[str, ...]]] = {
    "c03.spans.per-ordinal": ("index", "index", _sizes_per_shard, ("per_ordinal",)),
    "c03.spans.total": ("index", "index", _total_per_shard, ("per_shard",)),
    "c03.drops.sanitizer": (
        "counted", "check3.drops", _no_layout(_drops_values), ("drops", "documents_per_shard"),
    ),
    "c06.generated.drops": (
        "counted", "check3.drops", _no_layout(_drops_values), ("drops", "documents_per_shard"),
    ),
    "c03.records.capture": ("counted", "documents", _no_layout(_documents_values), ("per_ordinal",)),
    "c03.records.position-recovery": (
        "counted", "documents", _no_layout(_documents_values), ("per_ordinal",),
    ),
    "c03.records.token-matching": (
        "counted", "documents", _no_layout(_documents_values), ("per_ordinal",),
    ),
    "c06.generated.documents-mapped": (
        "counted", "documents", _no_layout(_documents_mapped_values),
        ("per_ordinal", COUNT_DOCUMENTS, COUNT_ONE_TIED_ROW, COUNT_NO_TIED_ROW),
    ),
    "c03.pairs.same-shard": (
        "counted", "check3.pairs", _no_layout(_same_shard_values), ("candidate_set",),
    ),
    "c03.pairs.same-shard.swap": (
        "counted", "check3.pairs", _no_layout(_same_shard_values), ("candidate_set",),
    ),
    "c03.pairs.split": ("counted", "check3.pairs", _no_layout(_split_values), ("record",)),
    "c06.generated.candidates": (
        "counted", "check6.candidates", _no_layout(_candidates_values), ("candidate_sources",),
    ),
    "c06.generated.downstream-of-translation": (
        "counted", "check6.translations", _no_layout(_translations_values), ("translations",),
    ),
    "c03.precondition.pair-kinds": (
        "counted", "check3.pairs.counts", _no_layout(_pair_kinds_values),
        ("same_shard_pairs", "split_pairs"),
    ),
    "c03.drops.no-tokens": (
        "counted", "check3.drops", _no_layout(_no_tokens_values),
        ("drops_with_reason_no_tokens", "documents_per_shard", "surviving_lines_per_shard"),
    ),
    "c09.10k.action-counts": ("counted", "actions", _action_counts_values, ("counts",)),
    "c09.10k.partitioned.kept-shards": (
        "counted", "shards", _kept_shards_values, ("keep", "rewrite"),
    ),
    "c09.10k.round-robin.all-rewrite": (
        "counted", "shards", _shard_counts_values, ("keep", "rewrite"),
    ),
}


def _concrete_of(expected: Expected, row: Any, layout: str | None, held: Any) -> Concrete:
    """The concrete values of ``row`` for ``layout``, from the counting document ``held``
    and the index of ``expected``. A part a given document lacks, or an index that was not
    given, is a ``PartMissing``; no counting document at all leaves every key the document
    would give in ``open``."""
    expect = dict(expected.expect)
    beside: dict[str, Any] = {}
    not_given: tuple[str, ...] = ()
    where = f" on the {layout} layout" if layout is not None else ""
    if row.id in _CONCRETE:
        source, part, read, keys = _CONCRETE[row.id]
        found = expected.index if source == "index" else held
        if found is not None:
            sentences = SENTENCE_FIELDS.get(row.id, frozenset())
            given = read(found, row.id, layout)
            for key, value in given.items():
                if key in sentences and key in expect:
                    expect[key] = value
                else:
                    beside[key] = value
            not_given = tuple(key for key in keys if key not in given)
        elif source == "index":
            raise PartMissing(f"{row.id}{where} reads {part}: no index was given")
        else:
            # The row's source names a counting document and none was given: the table's
            # own words in these keys never stand in for a value.
            not_given = keys

    # A field is read as the table wrote it only while the expectation in force still holds
    # the table's own words: a fallback that states the value itself ("none recorded") is
    # an expected value, not a field to list.
    def as_written(key: str) -> bool:
        return key in expect and row.expect.get(key) == expect[key]

    unanswered = not_given + tuple(key for key in _NOT_DERIVED.get(row.id, ()) if as_written(key))
    left_out = tuple(key for key in _NOT_COMPARED.get(row.id, ()) if as_written(key))
    return Concrete(row.id, expected.size, layout, expect, unanswered, beside, left_out)


def _applying_row(expected: Expected) -> Any:
    if not expected.applies:
        raise ValueError(
            f"{expected.listed_row} does not apply under the fallback in force: it is not put"
        )
    return _core_rows()[expected.listed_row]


def concrete_values(expected: Expected) -> Concrete:
    """The expectation of a listed row with each sentence-valued field replaced by its
    concrete value, read from the counting document (``counted``) or the shard's own
    index (``index``) the row names; every other field is the table's own. A counted value
    the expectation holds no sentence for is kept in ``Concrete.counted_values``. A field
    read "expected value not derived" stays as written and is named in ``open``; a
    not-compared part stays as written and is named in ``not_compared``.

    A part a given counting document lacks, and an index that was not given, is a
    ``PartMissing`` naming it. No counting document at all is not an error: the keys it
    would give are in ``open`` and the table's own words stay in ``expect``, never compared.
    A row that does not apply
    under the fallback in force is a ``ValueError``, and a row whose ``given`` names two
    layouts is a ``ValueError`` naming ``layouts``: it is asked per layout (see
    ``concrete_values_by_layout``). A row of one layout reads only a document counted on
    that layout.
    """
    row = _applying_row(expected)
    if "layouts" in row.given:
        raise ValueError(f"{row.id} names layouts {row.given['layouts']}: it is asked per layout")
    return _concrete_of(expected, row, row.given.get("layout"), expected.counted)


def concrete_values_by_layout(expected: Expected) -> dict[str, Concrete | ValueError]:
    """For a row whose ``given`` names two layouts, one entry per layout in the row's
    order. The entry is a ``Concrete`` for that layout (with its ``open`` naming the keys a
    missing counting document would give), or, where its value cannot be derived (a part a
    given document lacks, or one counted on the other layout), the ``ValueError`` itself,
    returned and never raised: one layout that cannot be derived takes nothing from the
    other. ``counted`` of such a row maps each layout to the document counted on it."""
    row = _applying_row(expected)
    layouts = row.given.get("layouts")
    if not layouts:
        raise ValueError(f"{row.id} names no layouts: it is read by concrete_values")
    documents = expected.by_layout or {}
    found: dict[str, Concrete | ValueError] = {}
    for layout in layouts:
        try:
            found[layout] = _concrete_of(expected, row, layout, documents.get(layout))
        except ValueError as problem:
            found[layout] = problem
    return found


def fields_without_concrete_value() -> tuple[tuple[str, str, str], ...]:
    """ONE list of the fields read "expected value not derived": for each, the row, the
    key and the source text of the row as the table writes it. A field derived as a
    value and a not-compared part are not in it."""
    rows = _core_rows()
    return tuple((listed, key, rows[listed].source) for listed, keys in _NOT_DERIVED.items() for key in keys)


# --- The whole-run diff rows ------------------------------------------------------

# The rows that ask whether the join's rows for a whole run equal the oracle's log.
_WHOLE_RUN = frozenset({"c01.E.diff", "c01.E-rr.diff", "c01.K.diff"})
# The public name of the three whole-run diff rows (their source is the oracle's log of the run).
WHOLE_RUN_ROWS = _WHOLE_RUN


def whole_run_rows(expected: Expected, oracle: Sequence[Mapping[str, Any]]):
    """The rows the join must report for the whole run a diff row names, computed from the
    oracle's log (``run_rows`` gives the form): the log's rows of that run's segments, with
    the cut of step 8 applied where the spec puts it on a segment of the run. A subject's
    own diff is never taken.

    A row that is not a whole-run diff row, a row that does not apply, and a log with no
    row of the run are a ``ValueError``; so is a cut on a sample the log does not hold.
    """
    row = _applying_row(expected)
    if row.id not in _WHOLE_RUN:
        raise ValueError(f"{row.id} is not a whole-run diff row")
    run = row.given["run"]
    lines = [line for line in oracle if str(line["segment"]).split("/", 1)[0] == run]
    if not lines:
        raise ValueError(f"the oracle's log holds no row of run {run}")
    fixed = _fixture_spec()["positions"]["step8_cut"]["fixed"]
    cuts = {}
    if str(fixed["segment"]).split("/", 1)[0] == run:
        cuts[(fixed["segment"], fixed["step"], fixed["sample_pos"])] = fixed["covered_tokens"]
    return run_rows(lines, cuts)


# --- One id per ordinal against the carried expectation ---------------------------------

# The rows whose answer is a record per ordinal (the one public name of them). A reading
# gives one id at each ordinal; at an ordinal of a group of identical documents the id is
# right when it is any member of the group, and a list is right when it is the group as a
# set. The token-matching row expects the capture map's record only for a document outside
# a group, so the ordinals of a group are not compared for it.
ONE_ID_ROWS = frozenset({
    "c03.records.capture", "c03.records.position-recovery", "c03.records.token-matching",
    "c06.generated.documents-mapped",
})
# The name this set had before it was public; a module that still reads it keeps working.
_ONE_ID_ROWS = ONE_ID_ROWS
_GROUP_ORDINALS_LEFT_OUT = frozenset({"c03.records.token-matching"})


@dataclass(frozen=True)
class OneIdReading:
    """One id per ordinal read against the carried expectation of a row.

    listed_row: the row read against.
    compared: the number of ordinals compared.
    differing: the (shard, ordinal) pairs whose id is not right, and the ordinals of the
        reading that the expectation does not hold; in the expectation's order.
    """

    listed_row: str
    compared: int
    differing: tuple[tuple[str, int], ...]

    @property
    def right(self) -> bool:
        return not self.differing


def one_id_reading(
    listed_row: str,
    per_ordinal: Mapping[str, Sequence[Any]],
    given: Mapping[str, Sequence[Any]],
) -> OneIdReading:
    """Compare one id per ordinal (``given``: shard to its ids in ordinal order) with
    ``per_ordinal`` as ``records_by_ordinal`` carries it: an id at an ordinal outside every
    group, the list of the group's ids at an ordinal inside one.

    An ordinal outside a group is right only when the id is that id. An ordinal of a group
    is right when the id is a member of its list, or when the reading gives a list at it
    that equals the group's ids as a set; any other list differs. A reading that lacks a
    shard or an ordinal, gives something that is neither one id nor (at a group) a list of
    ids, or holds an ordinal or a shard the expectation does not, differs; a gap is never
    read as right. For the token-matching row the ordinals of a group are not compared. A
    row that has no record per ordinal is a ``ValueError``.
    """
    if listed_row not in ONE_ID_ROWS:
        raise ValueError(f"{listed_row} has no record per ordinal to read one id against")
    skip_groups = listed_row in _GROUP_ORDINALS_LEFT_OUT
    compared = 0
    differing: list[tuple[str, int]] = []
    for shard, wanted in per_ordinal.items():
        found = given.get(shard) or ()
        for ordinal, want in enumerate(wanted):
            in_group = isinstance(want, (list, tuple))
            if in_group and skip_groups:
                continue
            compared += 1
            one = found[ordinal] if ordinal < len(found) else None
            if isinstance(one, str):
                right = one in want if in_group else one == want
            elif in_group and isinstance(one, (list, tuple)):
                right = all(isinstance(each, str) for each in one) and set(one) == set(want)
            else:
                right = False
            if not right:
                differing.append((shard, ordinal))
    for shard, found in given.items():
        held = per_ordinal.get(shard)
        differing.extend((shard, ordinal) for ordinal in range(len(held or ()), len(found or ())))
    return OneIdReading(listed_row, compared, tuple(differing))


# --- A line asked in both layouts -------------------------------------------------------

# What the evidence of a layout says when it has no grade of the subject's.
NOT_DERIVED = "expected value not derived"


def _is_failed_attempt(result: Any) -> bool:
    """True for the status of a question whose attempt failed: the prefix and an id."""
    return (
        isinstance(result, str) and result.startswith(NO_ANSWER_PREFIX)
        and len(result) > len(NO_ANSWER_PREFIX)
    )


def combine_layouts(
    listed_row: str, results: Mapping[str, Grade | Concrete | ValueError | str | None]
) -> Grade:
    """The one line of a listed row that names two layouts, from one result per layout:
    a ``Grade`` of that layout, the ``ValueError`` its expected value could not be derived
    with (a slot ``concrete_values_by_layout`` returns), the ``Concrete`` of a layout whose
    counting document is missing (its ``open`` is not empty), None, or one of the two
    statuses of an attempt, ``NOT_ATTEMPTED`` and ``no answer: failed attempt <id>``. A
    result that is None, and a layout the mapping lacks, is that layout NOT ATTEMPTED.

    The line is, in this order: wrong if any layout is wrong; the status of a failed
    attempt if any layout's attempt failed (the first one, in the row's order); not
    attempted if any layout was not attempted; right if all are right; otherwise cannot
    answer, the evidence naming each layout. The evidence has one entry per layout, in the
    row's order, each ``{"status": ..., "evidence": ...}``: the layout's own grade or
    attempt status (``NOT_ATTEMPTED`` for a layout with no result) or ``NOT_DERIVED`` (the
    expected value is missing: the error's text, or the open keys, as evidence). A line
    always ends in a status.

    A row that names no layouts, or a layout the row does not name, is a ``ValueError``. A
    result of any other kind is a ``TypeError`` or a ``ValueError``; so is a ``NotGradable``,
    which is no layout's result (it says the question cannot be put on a grade at all), and
    a ``Concrete`` with nothing open (it holds a value, not a result).
    """
    # Imported here: the grader imports this module.
    from baselines.grade import NotGradable

    layouts = _core_rows()[listed_row].given.get("layouts")
    if not layouts:
        raise ValueError(f"{listed_row} names no layouts: its line is not combined")
    unnamed = [name for name in results if name not in layouts]
    if unnamed:
        raise ValueError(f"{listed_row} names layouts {list(layouts)}, not {unnamed}")
    slots: dict[str, dict[str, Any]] = {}
    for layout in layouts:
        result = results.get(layout)
        if result is None:
            slots[layout] = {"status": NOT_ATTEMPTED, "evidence": None}
        elif isinstance(result, NotGradable):
            raise TypeError(f"{listed_row}, {layout}: a NotGradable is not a layout's result")
        elif isinstance(result, Concrete):
            if not result.open:
                raise TypeError(f"{listed_row}, {layout}: a Concrete with a value is not a layout's result")
            slots[layout] = {
                "status": NOT_DERIVED,
                "evidence": f"{NOT_DERIVED} for {', '.join(result.open)}",
            }
        elif isinstance(result, ValueError):
            slots[layout] = {"status": NOT_DERIVED, "evidence": str(result)}
        elif isinstance(result, str):
            if result != NOT_ATTEMPTED and not _is_failed_attempt(result):
                raise ValueError(f"{listed_row}, {layout}: {result!r} is not the status of an attempt")
            slots[layout] = {"status": result, "evidence": None}
        elif isinstance(result, Grade) and result.status in GRADES:
            slots[layout] = {"status": result.status, "evidence": result.evidence}
        else:
            raise TypeError(f"{listed_row}, {layout}: {result!r} is not a layout's result")
    statuses = [slot["status"] for slot in slots.values()]
    if WRONG in statuses:
        return Grade(status=WRONG, evidence=slots)
    failed = [status for status in statuses if _is_failed_attempt(status)]
    if failed:
        return Grade(status=failed[0], evidence=slots)
    if NOT_ATTEMPTED in statuses:
        return Grade(status=NOT_ATTEMPTED, evidence=slots)
    if all(status == RIGHT for status in statuses):
        return Grade(status=RIGHT, evidence=slots)
    return Grade(status=CANNOT_ANSWER, evidence=slots)
