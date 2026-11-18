"""Check 9, repair (plan 5.5 Check 9): the rows of fixture/table/check09.toml, read,
given their expected values, and compared, as pure functions.

``table.interpret`` is the reader of Checks 1 to 8 and refuses most rows of Check 9
(the keys ``layouts``, ``mode`` and ``corrected_run``, the two wrong-report states,
the expected fields of a repair). This module reads the rows of Check 9 on its own,
with the same strictness: a key, a state, a request, an artifact or an expected
field it does not know is a ``TableError``, an error of the test and never a skip.

What a row waits for
--------------------
A repair is three commands: ``waterwhite report`` (docs/interfaces.md 7.1), the
replica's ``repair`` (9.5) and ``waterwhite verify`` (7.3). Each row names the exact
outputs it waits for (``WAITS``). Whether an output exists is found out by asking the
command line, once per run (``acceptance_probe.repair_gaps``), so a row starts to run on
the commit that builds the output, with no edit here. Until then it skips with "not built
yet: <the exact output>".

A row whose outputs all exist is put (``put``): the tool's own documents are read in the
form docs/interfaces.md 7.1 and 7.3 fix (``acceptance_report``) and compared with the
row's expected values. A document that is missing a key the row needs is a difference,
never a pass.

Where expected values come from (plan W5, plan 5.1)
---------------------------------------------------
- the fixture rows and the rows of run E: the row itself; the exact-rerun step is
  the spec's ``positions.exact_rerun_restart.step`` (``table.fill_pending``);
- the rows of the 10k set: the plant list as ``fixture/plant/count.py`` counts it
  (``expected_record_counts``, ``expected_actions_by_kind``,
  ``expected_shard_actions``, ``expected_unrelated_paraphrase``). Nothing here reads
  a ``waterwhite`` module, and no count is typed in from docs/plant.md.
- the fingerprint of the multiset comparison is this module's own: the SHA-256 of
  the record's text as UTF-8. It is not the tool's normaliser. Both sides are read
  from the files the replica wrote, which share the replica's pipeline steps by
  design (plan 4.4).

Plan sections served: 5.5 Check 9, 5.1, 4.1 to 4.4, W5. The multiset comparison of
plan 4.2 and 4.4 is made twice and each is held to the table: the tool's own word
(``multisets``), and this module's multiset read from the files the sides file names.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import table
from table import Mismatch, Row, TableError

REPO = table.REPO
SRC = REPO / "src"
SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"
CHECK = 9
SET_10K = "10k"
LAYOUTS = ("partitioned", "round-robin")

# --- Reading a row -------------------------------------------------------------------

REQUESTS = frozenset({"withdraw A"})
# The states of README section 7 that Check 9 uses. The two ``r.*`` states that
# change a rebuild report are applied to the report of the 10k set before the
# replica builds the repaired side.
STATES = frozenset({"unmutated", "r.successor-registered", "r.descendant-kept",
                    "r.unrelated-paraphrase-retired"})
GIVEN_KEYS = frozenset({"set", "layout", "layouts", "request", "state", "mode", "artifact",
                        "run", "segment", "corrected_run"})
ARTIFACTS = frozenset({"each successor shard", "successor of R1", "successor of E/shard-3"})
SEEDED = "seeded"

(K_MULTISET, K_DIFFERENCE, K_DIGESTS, K_COUNTS, K_BY_KIND, K_KEPT_SHARDS, K_ALL_REWRITE,
 K_SUCCESSOR, K_HISTORY, K_FIXTURE_COUNTS, K_ACTION, K_SHARDS_COUNTS, K_MIXTURES,
 K_EXPOSURE_FREE, K_EXACT_RERUN, K_STALE, K_REPORT_ACTION) = (
    "repaired side equals the reference", "repaired side differs from the reference",
    "per-shard digests equal", "action counts of the 10k set", "actions by kind of record",
    "kept shards of the partitioned layout", "all chunks rewritten",
    "answer of a successor", "answer of the historical release",
    "action counts of the fixture's R1", "action of one artifact",
    "shards of run E kept and rewritten", "mixtures that need a successor",
    "exposure-free restart", "exact-rerun restart", "stale cache files",
    "action of one artifact, from the report alone")

# The expected fields of a row decide its kind: the table's own words, nothing else.
SIGNATURES: dict[frozenset[str], str] = {
    frozenset({"multisets", "unresolved_records"}): K_MULTISET,
    frozenset({"multisets", "differing_records"}): K_DIFFERENCE,
    frozenset({"per_shard_digests"}): K_DIGESTS,
    frozenset({"counts", "record_actions_allowed"}): K_COUNTS,
    frozenset({"X_release_rows", "B_copies_of_X", "P_T_S"}): K_BY_KIND,
    frozenset({"keep", "kept_shards_byte_identical", "every_other_shard"}): K_KEPT_SHARDS,
    frozenset({"rewrite", "keep"}): K_ALL_REWRITE,
    frozenset({"answer"}): K_SUCCESSOR,  # K_HISTORY where the artifact is a release
    frozenset({"counts", "records_removed", "rows_in_successor"}): K_FIXTURE_COUNTS,
    frozenset({"action"}): K_REPORT_ACTION,
    frozenset({"action", "byte_identical_afterwards"}): K_ACTION,
    frozenset({"action", "records_removed", "documents_in_successor"}): K_ACTION,
    frozenset({"counts", "keep", "rewrite"}): K_SHARDS_COUNTS,
    frozenset({"needs_successor", "kept"}): K_MIXTURES,
    frozenset({"exposure_free_restart"}): K_EXPOSURE_FREE,
    frozenset({"exact_rerun_restart_step", "not_before_step"}): K_EXACT_RERUN,
    frozenset({"exact_rerun_restart", "step_comparison"}): K_EXACT_RERUN,  # under fallback F7
    frozenset({"stale", "rebuilt", "rebuilt_from"}): K_STALE,
    frozenset({"stale"}): K_STALE,
}


@dataclass(frozen=True)
class Repair:
    """One row of Check 9, read."""

    row: Row
    kind: str
    size: str | None            # "10k", or None for a row of the fixture and of run E
    layouts: tuple[str, ...]    # the layouts a row of the 10k set is asked in; () otherwise
    state: str
    seeded: bool
    artifact: str | None
    run: str | None             # the run of a row of run E
    segment: str | None
    expect: Mapping             # the row's expect, pending values not yet filled

    @property
    def id(self) -> str:
        return self.row.id


def read(row: Row, fallbacks: Sequence[str] = ()) -> Repair:
    """Read a row of Check 9. Raises ``TableError`` for anything it does not know."""
    where = f"row {row.id}"
    given = row.given
    unknown = set(given) - GIVEN_KEYS
    if unknown:
        raise TableError(f"{where}: given holds keys this reader does not know: {sorted(unknown)}")
    if given.get("request") not in REQUESTS:
        raise TableError(f"{where}: Check 9 asks only {sorted(REQUESTS)}, "
                         f"the row asks {given.get('request')!r}")
    state = given.get("state", "unmutated")
    if state not in STATES:
        raise TableError(f"{where}: the state {state!r} is not one Check 9 uses")
    if "layout" in given and "layouts" in given:
        raise TableError(f"{where}: given holds both layout and layouts")
    layouts = tuple(given["layouts"]) if "layouts" in given else (
        (given["layout"],) if "layout" in given else ())
    for layout in layouts:
        if layout not in LAYOUTS:
            raise TableError(f"{where}: unknown layout {layout!r}")
    size = given.get("set")
    if size not in (None, SET_10K):
        raise TableError(f"{where}: Check 9 asks the set {SET_10K!r}, the row asks {size!r}")
    if size is not None and not layouts:
        raise TableError(f"{where}: a row of the {size} set names its layouts")
    mode = given.get("mode")
    if mode not in (None, SEEDED):
        raise TableError(f"{where}: unknown mode {mode!r}")
    artifact = given.get("artifact")
    if artifact is not None and artifact != table.LISTING and artifact not in ARTIFACTS:
        table.artifact_kind(artifact)  # raises TableError for a name docs/interfaces.md 9.2 lacks
    expect = table.expect_in_force(row, fallbacks)
    kind = SIGNATURES.get(frozenset(expect))
    if kind is None:
        raise TableError(f"{where}: no question of Check 9 has the expected fields "
                         f"{sorted(expect)}")
    if kind == K_SUCCESSOR and artifact in ("R1", "R2"):
        kind = K_HISTORY
    return Repair(row, kind, size, layouts, state, mode == SEEDED, artifact, given.get("run"),
                  given.get("segment"), dict(expect))


def read_all(rows: Iterable[Row] | None = None) -> list[Repair]:
    return [read(row) for row in (table.load_check(CHECK) if rows is None else rows)]


# --- What a row waits for ------------------------------------------------------------

GATE_REPORT = "rebuild report"
GATE_SUCCESSOR = "successor"
GATE_VERIFY = "verification of a repair"

_REPORT = (GATE_REPORT, "the tool's rebuild report of `withdraw A` (the command `report`, "
           "docs/interfaces.md 7.1, plan 4.1)")
_SIDE = (GATE_SUCCESSOR, "the repaired side, successors built from that report (the replica's "
         "command `repair`, docs/interfaces.md 9.5, plan 4.4)")
_VERIFY = (GATE_VERIFY, "the tool's verification of a repair (the command `verify`, "
           "docs/interfaces.md 7.3, plan 4.4)")


def _words(gate_and_words: tuple[str, str], extra: str = "") -> tuple[str, str]:
    gate, words = gate_and_words
    return gate, words + (f", with {extra}" if extra else "")


WAITS: dict[str, tuple[tuple[str, str], ...]] = {
    K_MULTISET: (_REPORT, _SIDE, _VERIFY),
    K_DIFFERENCE: (_words(_REPORT, "the change the state makes to it, the first planted translation kept "
                                    "or the first unrelated paraphrase retired (README section 7)"),
                   _SIDE, _VERIFY),
    K_DIGESTS: (_REPORT, _SIDE, _VERIFY),
    K_COUNTS: (_words(_REPORT, "its action counts and the number of records removed (plan 4.1)"),),
    K_BY_KIND: (_words(_REPORT, "its action for each release row (plan 4.1, Decisions 10 and 15)"),),
    K_KEPT_SHARDS: (_words(_REPORT, "its action for each shard (plan 4.1)"), _SIDE, _VERIFY),
    K_ALL_REWRITE: (_words(_REPORT, "its action for each of the 20 chunks (plan 4.1, 4.5)"),),
    K_SUCCESSOR: (_REPORT, _words(_SIDE, "capture on, so that a question can be put to it")),
    K_HISTORY: (_REPORT, _words(_SIDE, "the successors registered beside the historical artifacts")),
    K_FIXTURE_COUNTS: (_words(_REPORT, "its counts, its records removed and the rows of the "
                                       "successor of R1 (plan 4.1)"),),
    K_REPORT_ACTION: (_words(_REPORT, "its action for the artifact (plan 4.1)"),),
    K_ACTION: (_words(_REPORT, "its action for the artifact (plan 4.1)"),
               _words(_SIDE, "the artifact's bytes and records afterwards (plan 4.4)"),
               _words(_VERIFY, "its `kept` entry for the artifact (plan 4.4)")),
    K_SHARDS_COUNTS: (_words(_REPORT, "its count of shards rewritten per layout (plan 4.1)"),),
    K_MIXTURES: (_words(_REPORT, "the mixtures that need a successor (plan 1.1, 4.1)"),),
    K_EXPOSURE_FREE: (_words(_REPORT, "its exposure-free restart point (plan 4.3, Decision 23)"),),
    K_EXACT_RERUN: (_REPORT, _SIDE,
                    _words(_VERIFY, "its exact-rerun restart point, which the report does not "
                                    "state (plan 4.3, Decision 23, F7)")),
    K_STALE: (_REPORT, _SIDE,
              _words(_VERIFY, "its `caches` entry for each segment of the run (plan 4.4)")),
}
# The kinds whose shards are tokenized by Megatron's preprocessing (the replica needs
# the test-time environment for them), or that read a run: the others read records and
# release rows only.
TOKENIZED = frozenset({K_MULTISET, K_DIFFERENCE, K_DIGESTS, K_KEPT_SHARDS, K_ALL_REWRITE,
                       K_SUCCESSOR, K_ACTION, K_SHARDS_COUNTS, K_EXPOSURE_FREE, K_EXACT_RERUN,
                       K_STALE})

def current_gaps() -> dict[str, bool]:
    """gate -> True where what the gate stands for does not exist yet. One source for both
    checks: ``acceptance_probe.repair_gaps``, which asks the command line (`report` and
    `verify` of the tool, `repair` of the replica) and whose entries are the needs of
    ``table`` (``NEED_REPORT``, ``NEED_VERIFY``, ``NEED_SUCCESSOR``)."""
    from acceptance_world import NOT_BUILT

    return {
        GATE_REPORT: table.NEED_REPORT in NOT_BUILT,
        GATE_VERIFY: table.NEED_VERIFY in NOT_BUILT,
        GATE_SUCCESSOR: table.NEED_SUCCESSOR in NOT_BUILT,
    }


# --- Whether a row runs --------------------------------------------------------------

SKIP, PUT = "skip", "put"
LONG_RUNS_VARIABLE = "WATERWHITE_CLEARED_LONG_RUNS"  # the name acceptance_rows reads


@dataclass(frozen=True)
class Decision:
    action: str            # SKIP or PUT
    reason: str = ""


def long_run_cleared(size: str, variable: str | None = None) -> bool:
    cleared = {item.strip() for item in
               (os.environ.get(LONG_RUNS_VARIABLE, "") if variable is None else variable).split(",")}
    return f"{CHECK}:{size}" in cleared


def spec() -> dict:
    with open(SPEC, "rb") as f:
        return tomllib.load(f)


def decide(repair: Repair, gaps: Mapping[str, bool], *, megatron: object,
           long_run_cleared_for_size: bool, spec_: Mapping | None = None) -> Decision:
    """Skip, with the exact thing the row waits for, or go on to put the question.

    The order is fixed: not built yet first (it costs nothing, and is the reason the
    plan asks for), then the pending values of the spec, then the machine (the
    Megatron environment), then the long-run gate of the 10k set. A skipped row is
    not a pass."""
    if not repair.row.core:
        return Decision(SKIP, f"extension off: {repair.row.extension}")
    waiting = [words for gate, words in WAITS[repair.kind] if gaps[gate]]
    if waiting:
        return Decision(SKIP, "not built yet: " + "; ".join(waiting))
    _, empty = table.fill_pending(dict(repair.expect), spec() if spec_ is None else spec_)
    if empty:
        return Decision(SKIP, "not checked: the fixture spec holds no value yet for the pending "
                              "item " + ", ".join(empty))
    if repair.kind in TOKENIZED and not megatron:
        return Decision(SKIP, "not run: its shards are tokenized only where the Megatron "
                              f"environment is set ({megatron_variable()})")
    if repair.size is not None and not long_run_cleared_for_size:
        return Decision(SKIP, f"a long run: Check {CHECK} on the {repair.size} set runs only "
                              f"when {LONG_RUNS_VARIABLE} names {CHECK}:{repair.size}")
    return Decision(PUT)


def megatron_variable() -> str:
    import acceptance_sets

    return acceptance_sets.MEGATRON_VARIABLE


# --- Expected values from the plant list (plan 5.1, W5) ------------------------------

RECORD_ACTIONS = ("keep", "re-source", "promote", "retire")
ACTIONS_OF_AN_ARTIFACT = ("keep", "rewrite")


class NeedsChunkFiles(Exception):
    """The value is counted from the chunk files of the build the row is asked of."""


def expected_record_counts(counted: Mapping, release: str | None = None) -> dict:
    """The number of release rows per action and the records removed, counted by the
    script: over every release (the report covers every affected artifact), or for one.
    ``none`` is the dedup loser that stays dropped; it is no action and is left out."""
    source = counted["actions"]["records"] if release is None else (
        counted["releases"][release]["actions"])
    removed = (sum(counted["releases"][name]["removed"] for name in counted["releases"])
               if release is None else counted["releases"][release]["removed"])
    return {**{word: source[word] for word in RECORD_ACTIONS}, "records_removed": removed}


def expected_shard_actions(counted: Mapping) -> dict[str, list[str]]:
    """The shards that are kept and rewritten, from the chunk files (the script's
    ``shards``, null where it was given none)."""
    shards = counted.get("shards")
    if not shards:
        raise NeedsChunkFiles("the kept-shard list is derived from the plant list and the "
                              "chunk-file ids of the build the row is asked of")
    return {"keep": sorted(shards["keep"]), "rewrite": sorted(shards["rewrite"])}


NOT_COVERED, NOT_REGISTERED = "not covered", "not registered"


def rows_left(entry: Mapping) -> int:
    """How many rows of a shard stay in it after the repair, from the count's table: the
    lines outside the withdrawal's scope. A line in scope is retired, or it is a row of the
    withdrawn source whose text another source also holds and which is handed to that copy,
    and such a row belongs to the file of the copy dedup then keeps, so it leaves the shard
    of the withdrawn source (docs/replica.md, a partitioned shard). ``receives`` is therefore
    not counted: it marks a shard that is rewritten, not a row that stays in it."""
    return entry["lines"] - entry["in_scope_lines"]


def successor_split(counted: Mapping, shards: Sequence[str], prefix: str) -> tuple[list[str], list[str]]:
    """(with a row, without one): the store names of the shards that are not on the kept-shard
    list of the count, by whether the count leaves a row in them. A shard the count has no line
    for holds none. Read from the plant list through the count, never from a report."""
    listed = expected_shard_actions(counted)
    kept = {f"{prefix}{stem}" for stem in listed["keep"]}
    held = {f"{prefix}{entry['shard']}": entry for entry in counted["shards"]["shards"]}
    complement = sorted((set(shards) | set(held)) - kept)
    return ([n for n in complement if n in held and rows_left(held[n]) > 0],
            [n for n in complement if n not in held or rows_left(held[n]) <= 0])


def check_successor(name: str, expect: Mapping, has_rows: bool, answer: str | None,
                    reason: str | None, files: Sequence[str]) -> list[Mismatch]:
    """The successor of one shard that is not on the kept-shard list. A shard that keeps
    a row has a successor that answers as the row says. A shard that keeps none has no
    registered successor (the store answers "not covered", "not registered") and no shard
    file of its name in the repaired build."""
    if has_rows:
        return [Mismatch(f"{m.field} of {name}", m.expected, m.got)
                for m in check_answer(expect, answer)]
    found = [Mismatch(f"{field} of {name}", want, got)
             for field, want, got in (("answer", NOT_COVERED, answer),
                                      ("reason", NOT_REGISTERED, reason)) if got != want]
    if files:
        found.append(Mismatch(f"shard file of {name}", "none", list(files)))
    return found


def new_successor_shards(built, repairs, repair: Repair, layout: str,
                         taken: Iterable[str]) -> tuple[list[str], list[str]]:
    """(written by the reference side, written by the repaired side only): the shards of the
    layout's naming rule that the historical build did not write and that the split above
    does not handle. They are looked for in the shards folders of the two builds. Whether the
    reference side wrote one is read from its own folder, and not asked of the repaired side's
    list, the report, ``verify`` or the repaired summary (the sides file names only the
    artifacts the report lists and the landing shards it names)."""
    import acceptance_sets

    hist = built.historical
    handled = {*hist.shards, *taken}
    names = [name for name in (f"{hist.prefix}{stem}" for stem in
                               repairs.shard_stems(repair.size, layout, repair.state))
             if name not in handled]
    if not names:
        return [], []
    named = acceptance_sets.layout_shards(names, layout, repairs.sets.set_folder(repair.size))
    written = [n for n in named if repairs.reference_shard_files(
        repair.size, layout, n[len(hist.prefix):], repair.state)]
    return written, [n for n in named if n not in written]


def check_successors(built, repairs, repair: Repair, layout: str) -> list[Mismatch]:
    """Every shard that is not on the kept-shard list, asked of the repaired store, and every
    shard the repair writes that the historical build never wrote: where the reference side
    writes it, its successor answers as the row says; where only the repaired side writes it,
    that is a difference (the file is a shard file of a name the reference build does not
    write)."""
    hist = built.historical
    with_rows, without = successor_split(hist.counted, hist.shards, hist.prefix)
    if not with_rows and not without:
        return [Mismatch("successor shards", ">= 1", 0)]
    expect, found = filled_expect(repair), []
    for name in sorted([*with_rows, *without]):
        answer, reason = repairs.answer_and_reason(repair.size, layout, f"successor of {name}",
                                                   repair.state)
        files = [] if name in with_rows else repairs.shard_files(
            repair.size, layout, name[len(hist.prefix):], repair.state)
        found += check_successor(name, expect, name in with_rows, answer, reason, files)
    kept = [f"{hist.prefix}{stem}" for stem in expected_shard_actions(hist.counted)["keep"]]
    written, only_repaired = new_successor_shards(built, repairs, repair, layout,
                                                  [*with_rows, *without, *kept])
    for name in written:
        answer, reason = repairs.answer_and_reason(repair.size, layout, f"successor of {name}",
                                                   repair.state)
        found += check_successor(name, expect, True, answer, reason, [])
    for name in only_repaired:
        files = repairs.shard_files(repair.size, layout, name[len(hist.prefix):], repair.state)
        found.append(Mismatch(f"shard file of {name}", "none", files))
    return found


class PlantDisagreement(Exception):
    """The row's word for a kind of record and the script's counted ids do not agree."""


def expected_actions_by_kind(counted: Mapping, plant_rows: Sequence[Mapping], expect: Mapping,
                             release: str = "R1") -> dict[str, dict[str, str]]:
    """The planted records by kind, each with the action the row names for its kind
    (``expect``: X_release_rows, B_copies_of_X, P_T_S). The kind and the source of a
    record come from the plant list: X's release rows are A's copies of a passage that
    has a copy in B, B's copies are the other copies, and the paraphrases, translations
    and summaries made from A's records are P, T and S. The script's counted ids, which
    it derived again, are the cross-check: a planted id the script counts under another
    action, or a counted id no kind accounts for, raises ``PlantDisagreement``."""
    groups: dict[str, dict[str, str]] = {"X_release_rows": {}, "B_copies_of_X": {}, "P_T_S": {}}
    for row in plant_rows:
        if row["kind"] == "x-copy" and row["source"] == "A":
            groups["X_release_rows"][row["id"]] = expect["X_release_rows"]
        elif row["kind"] == "x-copy" and row["source"] == "B":
            groups["B_copies_of_X"][row["id"]] = expect["B_copies_of_X"]
        elif row["kind"] in ("paraphrase", "translation", "summary"):
            groups["P_T_S"][row["id"]] = expect["P_T_S"]
    counted_ids = {action: {i for item in counted["releases"][release]["ids"][action]
                            for i in _ids(item)} for action in ("re-source", "promote", "retire")}
    for action, ids in counted_ids.items():
        planted = {i for group in groups.values() for i, word in group.items() if word == action}
        if planted != ids:
            raise PlantDisagreement(
                f"the script counts {len(ids)} ids as {action} in {release} and the row's words "
                f"put {len(planted)} planted records there; they differ by "
                f"{sorted(planted ^ ids)[:3]}")
    return groups


def _ids(item: str) -> list[str]:
    """An id written as a range ("A-000001..A-000500") or alone."""
    if ".." not in item:
        return [item]
    first, last = item.split("..")
    prefix, _, low = first.rpartition("-")
    high = last.rpartition("-")[2]
    return [f"{prefix}-{n:0{len(low)}d}" for n in range(int(low), int(high) + 1)]


def expected_unrelated_paraphrase(counted: Mapping) -> str:
    """The record a wrong report retires: the first id of the counting script's list
    of the generated records that stay (README section 7, ``r.unrelated-paraphrase-retired``)."""
    return _ids(counted["generated"]["stay"]["ids"][0])[0]


# --- Comparisons (pure; the tool's own output is put to them once it exists) ---------


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MultisetResult:
    equal: bool
    differing_records: int   # the records by which the two multiplicity tables differ


def compare_multisets(repaired: Iterable[str], reference: Iterable[str]) -> MultisetResult:
    """The (fingerprint, multiplicity) multisets of the texts of the two sides (plan 5.5
    Check 9; plan 4.2 says the retained records and their multiplicities equal a from-scratch run).
    ``differing_records`` adds up, over every fingerprint, how many records the sides
    differ by: one record kept on one side only is 1."""
    left = Counter(fingerprint(text) for text in repaired)
    right = Counter(fingerprint(text) for text in reference)
    gap = sum(abs(left[key] - right[key]) for key in left.keys() | right.keys())
    return MultisetResult(gap == 0, gap)


def check_multisets(expect: Mapping, result: MultisetResult,
                    unresolved: int | None = None) -> list[Mismatch]:
    found: list[Mismatch] = []
    want = expect["multisets"]
    if want not in ("equal", "differ"):
        raise TableError(f"the expected multisets are {want!r}, not 'equal' or 'differ'")
    if (want == "equal") != result.equal:
        found.append(Mismatch("multisets", want, "equal" if result.equal else "differ"))
    if "differing_records" in expect and expect["differing_records"] != result.differing_records:
        found.append(Mismatch("differing_records", expect["differing_records"],
                              result.differing_records))
    if "unresolved_records" in expect and expect["unresolved_records"] != unresolved:
        found.append(Mismatch("unresolved_records", expect["unresolved_records"], unresolved))
    return found


def check_counts(expected: Mapping, got: Mapping, name: str = "counts") -> list[Mismatch]:
    """Every count the expected side holds, against the report's. A count the report
    does not give is a difference (``None``), never a pass."""
    return [Mismatch(f"{name}.{word}", want, got.get(word))
            for word, want in expected.items() if got.get(word) != want]


def check_record_action_words(allowed: Sequence[str], got: Mapping[str, str]) -> list[Mismatch]:
    """A record's action is one of the plan's four words (4.1); there is no 'remove' (D-19)."""
    strange = sorted({action for action in got.values() if action not in allowed})
    return [Mismatch("record_actions_allowed", list(allowed), strange)] if strange else []


def check_actions(expected: Mapping[str, str], got: Mapping[str, str],
                  name: str = "actions") -> list[Mismatch]:
    return [Mismatch(f"{name}[{key}]", want, got.get(key))
            for key, want in sorted(expected.items()) if got.get(key) != want]


def check_shard_actions(expected: Mapping[str, Sequence[str]], got: Mapping[str, str],
                        digests_before: Mapping[str, str] | None = None,
                        digests_after: Mapping[str, str] | None = None) -> list[Mismatch]:
    """Exactly the shards of ``expected['keep']`` are keep, every other one rewrite
    (a shard the report names no action for is a difference), and a kept shard is
    byte-identical afterwards where its digests before and after are given (plan 4.4)."""
    found = check_actions({**dict.fromkeys(expected["keep"], "keep"),
                           **dict.fromkeys(expected["rewrite"], "rewrite")}, got,
                          "shard actions")
    for name in sorted(expected["keep"]):
        if digests_before is None or digests_after is None:
            continue
        if digests_before.get(name) is None or digests_before.get(name) != digests_after.get(name):
            found.append(Mismatch(f"byte identical afterwards[{name}]", digests_before.get(name),
                                  digests_after.get(name)))
    return found


def check_set(name: str, expected: Iterable[str], got: Iterable[str]) -> list[Mismatch]:
    """A list the row compares as a set."""
    want, have = set(expected), set(got)
    return [] if want == have else [Mismatch(name, sorted(want), sorted(have))]


def check_answer(expect: Mapping, got: str | None) -> list[Mismatch]:
    return [] if got == expect["answer"] else [Mismatch("answer", expect["answer"], got)]


def check_restart(expect: Mapping, got: Mapping) -> list[Mismatch]:
    """The exposure-free restart checkpoint, or the exact-rerun restart step (which the
    row also bounds: no step before ``not_before_step`` can differ). Under the fallback F7
    the row expects the phase start stated as an assumption and no step comparison."""
    found: list[Mismatch] = [Mismatch(field, expect[field], got.get(field))
                             for field in ("exact_rerun_restart", "step_comparison")
                             if field in expect and got.get(field) != expect[field]]
    if "exposure_free_restart" in expect and got.get("exposure_free_restart") != expect[
            "exposure_free_restart"]:
        found.append(Mismatch("exposure_free_restart", expect["exposure_free_restart"],
                              got.get("exposure_free_restart")))
    if "exact_rerun_restart_step" in expect:
        step = got.get("exact_rerun_restart_step")
        if step != expect["exact_rerun_restart_step"]:
            found.append(Mismatch("exact_rerun_restart_step", expect["exact_rerun_restart_step"],
                                  step))
        if not isinstance(step, int) or step < expect["not_before_step"]:
            found.append(Mismatch("not_before_step", f">= {expect['not_before_step']}", step))
    return found


def check_stale(expect: Mapping, got: Mapping) -> list[Mismatch]:
    return [Mismatch(field, expect[field], got.get(field))
            for field in ("stale", "rebuilt", "rebuilt_from")
            if field in expect and got.get(field) != expect[field]]


# --- Putting a row to the tool's documents ----------------------------------------------------

FIXTURE_SHARD_PREFIX = "E/"   # the store names of the fixture's partitioned shards (9.2)


def filled_expect(repair: Repair) -> dict:
    """The row's expected values, with the spec's pending items filled in."""
    filled, empty = table.fill_pending(dict(repair.expect), spec())
    if empty:
        raise TableError(f"row {repair.id}: the spec holds no value for {list(empty)}")
    return filled


def fixture_question(repair: Repair, expect: Mapping) -> table.Question:
    """The row as the question ``World``'s report readers take: the fixture's store `main`,
    the request ``withdraw A``, the artifact the row names."""
    return table.Question(
        row=repair.row, kind=repair.kind, store="main", state=repair.state,
        request=table.REQUESTS["withdraw A"],
        artifacts=(repair.artifact,) if repair.artifact else (), record=None, expect=dict(expect))


def _no(what: str, failure: str | None) -> list[Mismatch]:
    return [Mismatch(what, f"a {what}", failure)]


def put_fixture(repair: Repair, world) -> list[Mismatch]:
    """A row of the fixture or of run E: the tool's report of ``withdraw A`` on the store
    `main`, the replica's repair of a copy of it, and the tool's verification of that repair."""
    import acceptance_report as rep

    kind, expect = repair.kind, filled_expect(repair)
    if kind in (K_SUCCESSOR, K_HISTORY, K_REPORT_ACTION):
        return world.put(table.interpret(repair.row, ()))
    question = fixture_question(repair, expect)
    layouts = (f"partitioned={FIXTURE_SHARD_PREFIX}",) if kind == K_SHARDS_COUNTS else ()
    if kind in (K_EXACT_RERUN, K_STALE):
        verification, failure = world.verification_of("main")
        if verification is None:
            return _no("verification", failure)
        if kind == K_STALE:
            return check_stale(expect, rep.cache_fields(verification, repair.segment))
        return check_restart(expect, rep.restart_fields(verification, repair.run))
    document, failure = world.report_of(question, layouts)
    if document is None:
        return _no("rebuild report", failure)
    if kind == K_FIXTURE_COUNTS:
        return rep.compare_artifact_fields(expect, document, repair.artifact)
    if kind == K_SHARDS_COUNTS:
        return rep.compare_group(expect, document, "by_layout", "partitioned")
    if kind == K_MIXTURES:
        return rep.compare_mixtures(expect, document)
    if kind == K_EXPOSURE_FREE:
        return check_restart(expect, {"exposure_free_restart": rep.restart_checkpoint(
            document, repair.run)})
    if kind == K_ACTION:
        reported = {k: v for k, v in expect.items() if k != "byte_identical_afterwards"}
        found = rep.compare_artifact_fields(
            reported, document, repair.artifact,
            lambda handle: world.report_name("main", repair.artifact, handle))
        if "byte_identical_afterwards" in expect:
            if expect["byte_identical_afterwards"] is not True:
                raise TableError(f"row {repair.id}: byte_identical_afterwards is not true")
            verification, failure = world.verification_of("main")
            found += (_no("verification", failure) if verification is None else
                      rep.compare_kept([repair.artifact], verification,
                                       "byte_identical_afterwards"))
        return found
    raise TableError(f"row {repair.id}: no way to put a row of the kind {kind!r}")


def _tagged(layout: str, found: Iterable[Mismatch]) -> list[Mismatch]:
    return [Mismatch(f"{m.field} [{layout}]", m.expected, m.got) for m in found]


def shard_actions(document: Mapping) -> dict[str, str]:
    """Shard name -> the action of its artifact entry."""
    return {e["artifact"]: e.get("action") for e in document.get("artifacts", [])
            if isinstance(e, Mapping) and e.get("kind") == "shard"}


def put_set(repair: Repair, repairs, plant_rows: Sequence[Mapping]) -> list[Mismatch]:
    """A row of the 10k set, in each layout it names: the repair is built and verified
    through the commands, and the row is read from the documents and from the files of the
    two sides. ``repairs`` is ``acceptance_repair.SetRepairs``."""
    import acceptance_repair
    import acceptance_report as rep

    kind, expect, found = repair.kind, filled_expect(repair), []
    for layout in repair.layouts:
        hist = repairs.historical(repair.size, layout)   # the report alone needs no repair
        if kind == K_COUNTS:
            found += _tagged(layout, check_set_counts(expect, hist.report, hist.counted))
            continue
        if kind == K_BY_KIND:
            found += _tagged(layout, check_by_kind(expect, hist, plant_rows, rep))
            continue
        if kind == K_ALL_REWRITE:
            held = Counter(shard_actions(hist.report).values())
            found += _tagged(layout, check_counts(
                {"rewrite": expect["rewrite"], "keep": expect["keep"]},
                {"rewrite": held["rewrite"], "keep": held["keep"]}, "shards"))
            continue
        built = repairs.repaired(repair.size, layout, repair.state)
        if kind in (K_MULTISET, K_DIFFERENCE, K_DIGESTS):
            mode = "seeded" if repair.seeded else None
            verification, failure = repairs.verification(repair.size, layout, repair.state, mode)
            if verification is None:
                found += _tagged(layout, _no("verification", failure))
                continue
            if kind == K_DIGESTS:
                compared, differ = own_digest_gap(built, repairs, repair, layout)
                found += _tagged(layout, check_digests(expect, verification, compared, differ))
                continue
            compared, gap = acceptance_repair.own_multiset_gap(
                built.repaired_files, built.reference_files)
            found += _tagged(layout, check_sides(expect, verification, compared, gap))
        elif kind == K_KEPT_SHARDS:
            found += _tagged(layout, check_kept_shards(built, repairs, repair, rep))
        elif kind == K_SUCCESSOR:
            found += _tagged(layout, check_successors(built, repairs, repair, layout))
        elif kind == K_HISTORY:
            found += _tagged(layout, check_answer(
                expect, repairs.answer(repair.size, layout, repair.artifact, repair.state)))
        else:
            raise TableError(f"row {repair.id}: no way to put a row of the kind {kind!r}")
    return found


def check_sides(expect: Mapping, verification: Mapping, compared: int, gap: int) -> list[Mismatch]:
    """Check 9's multiset rows, held to the table twice: this module's own multiset of the
    files the sides file names, and the tool's own word. Neither stands for the other."""
    found: list[Mismatch] = []
    if compared == 0:
        found.append(Mismatch("files compared", ">= 1", 0))
    found += check_multisets(expect, MultisetResult(gap == 0, gap),
                             verification.get("unresolved_records"))
    if verification.get("multisets") != expect["multisets"]:
        found.append(Mismatch("multisets (the tool's word)", expect["multisets"],
                              verification.get("multisets")))
    if "differing_records" in expect and verification.get("differing_records") != expect[
            "differing_records"]:
        found.append(Mismatch("differing_records (the tool's number)", expect["differing_records"],
                              verification.get("differing_records")))
    return found


def own_digest_gap(built, repairs, repair: Repair, layout: str) -> tuple[int, list[str]]:
    """(shards compared, the shards that differ) for the digest row: the shards the sides file
    names on both sides, as ``own_shard_gap`` reads them, and every other shard of the layout's
    naming rule that either build wrote, found in the two shards folders (the sides file names
    only the artifacts the report lists and the landing shards it names). A shard only one
    build wrote is a difference."""
    import acceptance_repair
    import acceptance_sets

    compared, differ = acceptance_repair.own_shard_gap(built.repaired_files, built.reference_files)
    hist = built.historical
    held = {name for name in set(built.repaired_files) & set(built.reference_files)
            if not acceptance_repair.is_jsonl(built.repaired_files[name])}
    stems = repairs.shard_stems(repair.size, layout, repair.state)
    found = {f"{hist.prefix}{stem}": stem for stem in stems if f"{hist.prefix}{stem}" not in held}
    if not found:
        return compared, differ
    named = acceptance_sets.layout_shards(found, layout, repairs.sets.set_folder(repair.size))
    repaired_folder, reference_folder = repairs.shards_folders(repair.size, layout, repair.state)
    more, unlike = acceptance_repair.own_folder_shard_gap(
        repaired_folder, reference_folder, [found[name] for name in named])
    return compared + more, [*differ, *(f"{hist.prefix}{stem}" for stem in unlike)]


def check_digests(expect: Mapping, verification: Mapping, compared: int,
                  differ: Sequence[str]) -> list[Mismatch]:
    """The per-shard byte digests: this module's own over the two sides' shard files, and the
    tool's word, in seeded mode."""
    if expect["per_shard_digests"] != "equal":
        raise TableError("the expected per_shard_digests is not 'equal'")
    found: list[Mismatch] = []
    if compared == 0:
        found.append(Mismatch("shards compared", ">= 1", 0))
    if differ:
        found.append(Mismatch("per_shard_digests (own digests)", "equal",
                              f"differ: {list(differ)[:3]}"))
    if verification.get("per_shard_digests") != expect["per_shard_digests"]:
        found.append(Mismatch("per_shard_digests (the tool's word)", expect["per_shard_digests"],
                              verification.get("per_shard_digests")))
    return found


def check_set_counts(expect: Mapping, report: Mapping, counted: Mapping) -> list[Mismatch]:
    """The report's action counts against the counting script's, and the words every record
    entry uses against the four of the plan."""
    counts = report.get("counts") if isinstance(report.get("counts"), Mapping) else {}
    records = counts.get("records") if isinstance(counts.get("records"), Mapping) else {}
    got = {**records, "records_removed": counts.get("records_removed")}
    found = check_counts(expected_record_counts(counted), got)
    shards = expected_shard_actions(counted)
    held = counts.get("shards") if isinstance(counts.get("shards"), Mapping) else {}
    found += check_counts({"keep": len(shards["keep"]), "rewrite": len(shards["rewrite"])}, held,
                          "counts.shards")
    words = {f"{e.get('artifact')}:{e.get('record')}": e.get("action")
             for e in report.get("records", []) if isinstance(e, Mapping)}
    return found + check_record_action_words(expect["record_actions_allowed"], words)


def check_by_kind(expect: Mapping, hist, plant_rows: Sequence[Mapping], rep) -> list[Mismatch]:
    """Every planted X row, B copy of X, and P, T and S, against the action the row names
    for its kind. Where each record is comes from the plant list and the replica's manifest
    of the release, never from the tool."""
    import acceptance_repair as repair_files

    groups = expected_actions_by_kind(hist.counted, plant_rows, expect)
    by_id = {row["id"]: row for row in plant_rows}
    release = repair_files.RELEASE
    got: dict[str, str] = {}
    wanted: dict[str, str] = {}
    for group, members in groups.items():
        for record, word in members.items():
            wanted[record] = word
            if group == "B_copies_of_X":
                name = f"src/{by_id[record]['source']}:L{by_id[record]['line']}"
            else:
                line = repair_files.manifest_line(hist.out, repair_files.RELEASE_FILE, record)
                name = f"{release}:L{line}"
            got[record] = rep.effective_action(hist.report, release, name)
    return check_actions(wanted, got, "action of the planted record")


def check_kept_shards(built, repairs, repair: Repair, rep) -> list[Mismatch]:
    """The kept-shard row: exactly the shards of the kept-shard list (from the chunk files) are
    kept and every other one is rewritten, and a kept shard is byte-identical afterwards by this
    module's own digest and by the tool's ``kept`` entries."""
    import acceptance_repair as repair_files

    hist = built.historical
    actions = shard_actions(hist.report)
    listed = expected_shard_actions(hist.counted)
    wanted = {"keep": [f"{hist.prefix}{s}" for s in listed["keep"]],
              "rewrite": [f"{hist.prefix}{s}" for s in listed["rewrite"]]}
    # A shard the count names that the helper's list lacks is a difference, never an error.
    found = [Mismatch(f"listed shard[{name}]", "listed", "not listed")
             for name in sorted([*wanted["keep"], *wanted["rewrite"]]) if name not in hist.shards]

    def digest(prefix: Path | None) -> str | None:
        """None where the shard has no file: a difference for a kept shard, not an error."""
        if prefix is None or not all(Path(f"{prefix}{s}").is_file() for s in (".bin", ".idx")):
            return None
        return repair_files.shard_digest(prefix)

    shards = hist.out / "pipeline" / hist.layout / "shards"
    before = {name: digest(shards / name[len(hist.prefix):]) for name in wanted["keep"]}
    after = {name: digest(built.repaired_files.get(name)) for name in wanted["keep"]}
    found += check_shard_actions(wanted, actions, before, after)
    verification, failure = repairs.verification(repair.size, hist.layout, repair.state, None)
    if verification is None:
        return found + _no("verification", failure)
    return found + rep.compare_kept(wanted["keep"], verification, "kept shard byte identical")
