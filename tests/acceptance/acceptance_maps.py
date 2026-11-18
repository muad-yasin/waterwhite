"""The comparisons of Checks 3 and 6 (plan 5.5): what the command line printed about
tokenization maps, drops and recovered links, held against values that come from
outside the tool.

Every function here takes plain documents: what ``show`` printed, and expected
values from the plant list as the counting script counts it, from a shard's own
index, from the sanitizer's log or from a row of the hand-written table. Nothing
here runs the tool, reads a store or imports anything of the tool, so each
comparison is tested on its own with documents written by hand (``test_maps.py``).

The forms of what ``show`` prints are the ones README.md beside this file fixes
under "What show must print for Checks 3 and 6". A document in another form is an
``OutputForm`` error, never a silent pass.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from table import Mismatch

RECOVERED, CANDIDATES, UNKNOWN = "recovered", "candidates", "unknown"
NO_TOKENS = "no tokens"
SANITIZER_REASONS = ("sanitizer skip", "empty line")

_LINE = re.compile(r"L([1-9]\d*)")


class OutputForm(Exception):
    """A printed document is not in the form the comparisons were written for."""


# A place is (artifact name, line): one line of a registered JSONL file.
Place = tuple[str, int]
# ``identify(place)`` gives the id the replica's file carries on that line, or
# None for a line with no id. The caller reads the file; the tool is not asked.
Identify = Callable[[Place], "str | None"]


def place(entry: object) -> Place:
    """The (artifact, line) a printed parent, candidate or dropped record names."""
    if not isinstance(entry, Mapping) or not isinstance(entry.get("artifact"), str):
        raise OutputForm(f"a record is printed as an object with artifact and record: {entry!r}")
    found = _LINE.fullmatch(str(entry.get("record")))
    if not found:
        raise OutputForm(f"a line of a JSONL file is printed as L<n>: {entry!r}")
    return entry["artifact"], int(found.group(1))


def binding(shown: Mapping, namespace: str, derivation: str) -> Mapping | None:
    """The binding of one map among those ``show SHARD`` prints, or None."""
    held = [b for b in shown.get("bindings", ())
            if b.get("namespace") == namespace and b.get("derivation") == derivation]
    if len(held) > 1:
        raise OutputForm(f"two bindings of the map {namespace}/{derivation}: {held!r}")
    return held[0] if held else None


def map_rows(document: Mapping) -> list[Mapping]:
    """The rows of a map document, checked: row n is the document ``D<n>``."""
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise OutputForm(f"a map is printed with its rows as a list: {str(document)[:200]}")
    for ordinal, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("record") != f"D{ordinal}":
            raise OutputForm(f"row {ordinal} of a map is the document D{ordinal}: {row!r}")
        for key in ("label", "token_start", "token_end", "parent", "candidates"):
            if key not in row:
                raise OutputForm(f"row D{ordinal} of a map has no {key!r}: {row!r}")
    return rows


def count_field(name: str, expected: int, found: Sequence[str]) -> list[Mismatch]:
    """A field that counts findings: a mismatch when the number differs, with the
    first findings in words."""
    if len(found) == expected:
        return []
    return [Mismatch(name, expected, f"{len(found)}; first: {list(found[:3])}")]


# --- Spans (Check 3) ---------------------------------------------------------------------------


def span_findings(rows: Mapping[str, Sequence[Mapping]],
                  sizes: Mapping[str, Sequence[int]]) -> list[str]:
    """Per ordinal: token end minus token start against the document's size in the
    shard's own index. A shard whose map has another number of rows than the index
    has documents is one finding per document that is missing or too many."""
    found = []
    for shard, wanted in sizes.items():
        got = rows[shard]
        for ordinal in range(max(len(got), len(wanted))):
            if ordinal >= len(got):
                found.append(f"{shard} D{ordinal}: no row")
            elif ordinal >= len(wanted):
                found.append(f"{shard} D{ordinal}: a row for a document the index does not hold")
            else:
                span = got[ordinal]["token_end"] - got[ordinal]["token_start"]
                if span != wanted[ordinal]:
                    found.append(f"{shard} D{ordinal}: span {span}, index {wanted[ordinal]}")
    return found


def total_findings(rows: Mapping[str, Sequence[Mapping]],
                   sizes: Mapping[str, Sequence[int]]) -> list[str]:
    """Per shard: the sum of the spans against the total of the shard's index."""
    found = []
    for shard, wanted in sizes.items():
        total = sum(row["token_end"] - row["token_start"] for row in rows[shard])
        if total != sum(wanted):
            found.append(f"{shard}: spans sum to {total}, the index holds {sum(wanted)}")
    return found


# --- Which record a row names (Checks 3 and 6) -------------------------------------------------


def assigned(row: Mapping, identify: Identify) -> tuple[str, frozenset]:
    """What one map row says: its label, and the ids of the lines it names (one
    for a parent, the whole set for candidates, none for an unknown row)."""
    label = row["label"]
    if label == RECOVERED:
        return label, frozenset({identify(place(row["parent"]))})
    if label == CANDIDATES:
        return label, frozenset(identify(place(entry)) for entry in row["candidates"])
    return label, frozenset()


def record_findings(rows: Sequence[Mapping], ids: Sequence[str],
                    identical: Sequence[Sequence[int]], identify: Identify, shard: str,
                    *, twins_as_set: bool) -> list[str]:
    """Per ordinal of one shard: does the row name the planted record?

    ``ids[n]`` is the planted record of document n. ``identical`` lists the groups
    of ordinals that hold the same text. A document outside a group must be
    "recovered" with that one record. For a document inside a group the row must
    name its record among its candidates; with ``twins_as_set`` it must be
    "candidates" with exactly the records of its group (Check 6)."""
    group_of = {ordinal: group for group in identical for ordinal in group}
    found = []
    for ordinal in range(max(len(rows), len(ids))):
        if ordinal >= len(rows) or ordinal >= len(ids):
            found.append(f"{shard} D{ordinal}: the map has {len(rows)} rows, the shard "
                         f"{len(ids)} planted records")
            continue
        label, named = assigned(rows[ordinal], identify)
        want = ids[ordinal]
        if ordinal not in group_of:
            if (label, named) != (RECOVERED, frozenset({want})):
                found.append(f"{shard} D{ordinal}: expected recovered {want}, got {label} "
                             f"{sorted(map(str, named))}")
        elif twins_as_set:
            group = frozenset(ids[o] for o in group_of[ordinal])
            if (label, named) != (CANDIDATES, group):
                found.append(f"{shard} D{ordinal}: expected candidates {sorted(group)}, got "
                             f"{label} {sorted(map(str, named))}")
        elif want not in named:
            found.append(f"{shard} D{ordinal}: {want} is not among {label} "
                         f"{sorted(map(str, named))}")
    return found


def same_as_findings(rows: Sequence[Mapping], reference: Sequence[Mapping],
                     identical: Sequence[Sequence[int]], shard: str) -> list[str]:
    """Per ordinal outside a group of identical documents: does the row name the
    same line as the reference map's row?"""
    twins = {ordinal for group in identical for ordinal in group}
    found = []
    for ordinal in range(max(len(rows), len(reference))):
        if ordinal in twins:
            continue
        if ordinal >= len(rows) or ordinal >= len(reference):
            found.append(f"{shard} D{ordinal}: one of the two maps has no such row")
            continue
        here, there = rows[ordinal], reference[ordinal]
        if here["label"] != RECOVERED or there["label"] != RECOVERED or (
                place(here["parent"]) != place(there["parent"])):
            found.append(f"{shard} D{ordinal}: {here['label']} {here['parent']!r} against "
                         f"{there['label']} {there['parent']!r}")
    return found


# --- Identical pairs (Check 3) -----------------------------------------------------------------


def twin_findings(rows: Sequence[Mapping], pair: Mapping, input_name: str, shard: str,
                  identify: Identify | None = None, *, label: str = CANDIDATES
                  ) -> tuple[list[str], list[str]]:
    """One pair placed in one shard, on one map. ``pair`` is an entry of the
    counting script's pair list. Returns (label findings, candidate-set findings):
    each of its two ordinals must carry ``label``, the row's expected label, and
    name exactly the pair's two chunk-file lines as its candidates. With
    ``identify``, the ids on those two lines must also be the pair's two members
    (the lines of a changed input are read for it)."""
    lines = {member["line"] for member in pair["members"]}
    members = {member["id"] for member in pair["members"]}
    wrong_label, wrong_set = [], []
    for member in pair["members"]:
        ordinal = member["ordinal"]
        where = f"{shard} D{ordinal} (pair {pair['pair']})"
        if ordinal >= len(rows):
            wrong_label.append(f"{where}: no row")
            continue
        row = rows[ordinal]
        if row["label"] != label:
            wrong_label.append(f"{where}: label {row['label']}")
            continue
        places = [place(entry) for entry in row["candidates"]]
        if sorted(places) != sorted((input_name, line) for line in lines):
            wrong_set.append(f"{where}: candidates {sorted(places)}")
        elif identify is not None and {identify(p) for p in places} != members:
            wrong_set.append(f"{where}: the candidate lines carry "
                             f"{sorted(str(identify(p)) for p in places)}, the pair is "
                             f"{sorted(members)}")
    return wrong_label, wrong_set


def split_findings(rows: Sequence[Mapping], member: Mapping, input_name: str,
                   pair: str, *, label: str = RECOVERED) -> list[str]:
    """One member of a pair split across two shards, on one map of its own shard:
    ``label``, the row's expected label, with the member's own line of this
    shard's input as its one parent."""
    ordinal = member["ordinal"]
    where = f"{member['shard']} D{ordinal} (pair {pair})"
    if ordinal >= len(rows):
        return [f"{where}: no row"]
    row = rows[ordinal]
    if row["label"] != label:
        return [f"{where}: label {row['label']}"]
    if row["parent"] is None or place(row["parent"]) != (input_name, member["line"]):
        return [f"{where}: parent {row['parent']!r}, expected line {member['line']} of "
                f"{input_name}"]
    return []


# --- The fields of a map (Checks 3 and 6) ------------------------------------------------------


def compare_binding(name: str, expect: Mapping, shown: Mapping, namespace: str,
                    derivation: str) -> list[Mismatch]:
    """The stored fields of one map against the row's: ``namespace``,
    ``derivation``, ``verification`` and ``label``, each where the row holds it."""
    held = binding(shown, namespace, derivation)
    if held is None:
        return [Mismatch(name, dict(expect), f"the shard holds no map {namespace}/{derivation}")]
    return [Mismatch(f"{name}.{key}", value, held.get(key))
            for key, value in expect.items() if held.get(key) != value]


def labels_of(rows: Sequence[Mapping]) -> Counter:
    return Counter(row["label"] for row in rows)


# --- Drops (Checks 3 and 6) --------------------------------------------------------------------


def drop_findings(printed: Sequence[tuple[str, str]],
                  expected: Sequence[tuple[str, str]]) -> list[str]:
    """Printed drops against expected ones, each a (dropped line, reason). Every
    expected drop must be printed exactly once, and no other drop with a reason of
    the sanitizer: a line recorded twice, under two artifacts, is a finding."""
    got, want = Counter(printed), Counter(expected)
    found = [f"missing: {line} ({reason})" for line, reason in sorted((want - got).elements())]
    found += [f"not expected, or recorded twice: {line} ({reason})"
              for line, reason in sorted((got - want).elements())]
    return found


def no_token_findings(given: Sequence[Sequence[int]], index: Sequence[int],
                      planted: Sequence[int], append_eod: bool) -> list[Mismatch]:
    """Whether the lines of a tokenizer input that give no tokens are the planted
    ones, by two outside witnesses: ``given`` is what the team's tokenizer answered
    for each line (its token ids, in line order), and ``index`` is the size of each
    line's entry in the index of the shard Megatron's preprocessing wrote from the
    same input. A line gives no tokens when the tokenizer's answer is empty and its
    entry in the index is empty; every other line has an entry as large as the
    answer, with the end-of-document token when it is appended."""
    out = []
    none = [number for number, tokens in enumerate(given, 1) if not tokens]
    if none != list(planted):
        out.append(Mismatch("lines the tokenizer gives no tokens for", list(planted), none))
    empty = [number for number, size in enumerate(index, 1) if size == 0]
    if empty != list(planted):
        out.append(Mismatch("lines the shard's index holds no tokens for", list(planted), empty))
    elif len(index) != len(given):
        out.append(Mismatch("lines in the shard's index", len(given), len(index)))
    else:
        wanted = [len(tokens) + (1 if append_eod and tokens else 0) for tokens in given]
        if list(index) != wanted:
            out.append(Mismatch("tokens of each line in the shard's index", wanted, list(index)))
    return out
