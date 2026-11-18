#!/usr/bin/env python3
"""The counting script: turns a plant list into the expected answers of a generated set.

It reads what the plant generator wrote (``set.toml`` and ``plant_list.jsonl``)
and, where a question needs them, the ids the replica's chunk files carry after
chunking. It never reads anything the tool wrote, and it shares no code with the
tool or with the generator: the standard library only (plan 5.1, W5).

It does not take the plant list's word for its own outcomes. Scope, dedup winners
and the outcome of every release input under "withdraw A" are worked out again
here from the structure the list states (sources, parents, parent modes, text
digests, release inputs, dedup priority) and compared with the outcome each row
states. One row that disagrees stops the script with a named error and no output.

A generated record is retired because its parents lead back to the withdrawn
source, never because it is generated. The sets hold a few paraphrases of
unrelated records to show the difference (plan 5.1 "Sizes", DECISIONS.md D-23):
their parents lead to no record of A, so they are counted as kept, and they are
printed as a list of their own, the generated records that stay.

For the rows of Checks 3 and 6 that are asked of a generated set it also prints
what each row is asked of: the planted record of every chunk-file line (line i of
a chunk file is document i - 1 of its shard), the documents of one shard that hold
identical text, the planted translations with the releases and shards that receive
each, and, when the sanitizer-input files are given, the shard and the place of
every line the sanitizer must skip.

It prints one JSON document; ``docs/plant.md`` describes every part of it.

Plan sections: 5.1 (where each set's expected answers come from), 5.5 Checks 3, 6
and 9, 4.1 (actions), 4.4 (clean sources), 3.2 and L3 (candidates). DECISIONS.md
D-19: a release row that leaves the successor is retired, whether it is A's own
record or derived from one, and the count of records removed is the number of rows
that leave. D-23: the generated records that stay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

# /2: no outcome "remove" (DECISIONS.md D-19). /3: the generated records that stay
# (DECISIONS.md D-23). /4: the parts the rows of Checks 3 and 6 on a generated set
# are asked of: ``documents``, ``check3.drops`` and ``check6.translations``.
COUNT_FORMAT = "waterwhite.plant-count/4"
SET_FORMAT = "waterwhite.plant-set/1"

KINDS = (
    "x-copy", "passage", "paraphrase", "paraphrase-of-unrelated", "translation", "summary",
    "unrelated", "pair-member", "sanitizer-case",
)
# The kinds of a generated record, each with the job that makes it. The first three
# say the record was made from the withdrawn source's records; the last says it was
# made from an unrelated record (DECISIONS.md D-23).
JOB_OF_KIND = {
    "paraphrase": "paraphrase",
    "translation": "translation",
    "summary": "summary",
    "paraphrase-of-unrelated": "paraphrase",
}
KEPT_KIND = "paraphrase-of-unrelated"
PAIR_KINDS = ("x-copy", "pair-member")
PARENT_MODES = ("all", "any")

ROW_PRESENT = "present"
ROW_LOSER = "dedup-loser"

# The outcomes of a release input: the plan's four action words for a record (4.1)
# and "none" for a dedup loser that stays dropped.
KEEP, RE_SOURCE, PROMOTE, RETIRE, NONE = "keep", "re-source", "promote", "retire", "none"
OUTCOMES = (KEEP, RE_SOURCE, PROMOTE, RETIRE, NONE)
CHANGING = (RE_SOURCE, PROMOTE, RETIRE)  # the outcomes that make a successor necessary
REWRITE = "rewrite"

SKIP_REASONS = ("sanitizer skip", "empty line")

SAME_SHARD, SPLIT, SINGLE, ABSENT, MULTIPLE = "same-shard", "split", "single", "absent", "multiple"

LAYOUTS = ("round-robin", "partitioned")

ROW_KEYS = (
    "id", "file", "line", "source", "kind", "job", "parents", "parent_mode", "pair",
    "twin", "withdraw_A", "releases", "sanitizer", "text_sha256", "chars",
)


class CountError(Exception):
    """Base class. Every failure of this script is one of its subclasses."""

    exit_code = 3


class UsageError(CountError):
    """The arguments do not make a question this script answers."""

    exit_code = 2


class PlantListInconsistent(CountError):
    """A row of the plant list disagrees with the set's structure or with itself."""


class ChunkFilesMismatch(CountError):
    """The chunk files do not hold exactly the lines the plant list says survive."""


class SanitizerInputMismatch(CountError):
    """The sanitizer-input files do not hold exactly the chunk-file lines, in
    order, and the lines the plant list says the sanitizer must skip."""


class PairKindsMissing(CountError):
    """Check 3 needs a same-shard pair and a split pair; one kind is missing."""


class LayoutExpectationFailed(CountError):
    """The chunk files cannot give the answer the plan fixes for their layout."""


class NoGeneratedRecordStays(CountError):
    """A release retires generated records and keeps none. On such a set a report
    that retired every generated record would match every count here, so the set
    cannot stand behind the rebuild check (plan 5.1 "Sizes", DECISIONS.md D-23)."""


# --- Reading -----------------------------------------------------------------------

_RANGE = re.compile(r"^(?P<a>.+)-(?P<x>\d+)\.\.(?P<b>.+)-(?P<y>\d+)$")


def expand(items: list[str]) -> list[str]:
    """Ids from a list of ids and ranges written "A-000001..A-000500"."""
    out: list[str] = []
    for item in items:
        m = _RANGE.match(item)
        if not m:
            out.append(item)
            continue
        if m["a"] != m["b"] or len(m["x"]) != len(m["y"]) or int(m["x"]) > int(m["y"]):
            raise PlantListInconsistent(f"the id range {item!r} is not a range of one file")
        width = len(m["x"])
        out += [f"{m['a']}-{n:0{width}d}" for n in range(int(m["x"]), int(m["y"]) + 1)]
    return out


def compress(ids: list[str]) -> list[str]:
    """The reverse of ``expand``: runs of ids written as ranges."""
    out: list[str] = []
    run: list[tuple[str, int, int]] = []  # prefix, number, width
    for item in ids:
        prefix, _, digits = item.rpartition("-")
        here = (prefix, int(digits), len(digits)) if digits.isdigit() and prefix else None
        if run and here and run[-1][0] == here[0] and run[-1][1] + 1 == here[1] and run[-1][2] == here[2]:
            run.append(here)
            continue
        _flush(run, out)
        if here:
            run = [here]
        else:
            run = []
            out.append(item)
    _flush(run, out)
    return out


def _flush(run: list[tuple[str, int, int]], out: list[str]) -> None:
    if not run:
        return
    first = f"{run[0][0]}-{run[0][1]:0{run[0][2]}d}"
    last = f"{run[-1][0]}-{run[-1][1]:0{run[-1][2]}d}"
    out.append(first if len(run) == 1 else f"{first}..{last}")
    run.clear()


def load_set(folder: Path) -> tuple[dict, list[dict]]:
    """The set description and the plant rows of a set folder."""
    try:
        with open(folder / "set.toml", "rb") as f:
            settings = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise UsageError(f"cannot read set.toml in {folder.name}: {exc}") from exc
    if settings.get("format") != SET_FORMAT:
        raise UsageError(f"set.toml has format {settings.get('format')!r}, expected {SET_FORMAT!r}")
    if settings.get("clean"):
        raise UsageError("this is a clean set: it has no plant list and nothing to count")
    rows: list[dict] = []
    try:
        with open(folder / "plant_list.jsonl", encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise PlantListInconsistent(f"plant list line {number} is not JSON") from exc
                if not isinstance(row, dict) or set(row) != set(ROW_KEYS):
                    raise PlantListInconsistent(
                        f"plant list line {number} does not have exactly the keys of a plant row"
                    )
                rows.append(row)
    except OSError as exc:
        raise UsageError(f"cannot read plant_list.jsonl in {folder.name}: {exc}") from exc
    return settings, rows


# --- Checking the plant list against its own structure -----------------------------


def _fail(row: dict, what: str) -> PlantListInconsistent:
    return PlantListInconsistent(f"row {row.get('id')!r}: {what}")


class Plant:
    """A plant list that has been checked: every stated outcome was derived again."""

    def __init__(self, settings: dict, rows: list[dict]) -> None:
        self.settings = settings
        self.rows = rows
        self.withdrawn = settings["withdrawn"]
        self.by_id: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row["id"], str) or row["id"] in self.by_id:
                raise _fail(row, "the id is missing or used twice")
            self.by_id[row["id"]] = row
        self._check_shapes()
        self.scope: dict[str, tuple[str | None, tuple[str, ...]]] = {}
        self._check_scope()
        self._check_generated_kinds()
        self.pairs: dict[str, list[dict]] = {}
        self._check_pairs()
        self.release_inputs: dict[str, list[str]] = {}
        self.expected: dict[str, dict[str, dict]] = {}
        self._check_releases()
        self._check_kept_placement()

    # --- rows --------------------------------------------------------------------

    def _check_shapes(self) -> None:
        params = self.settings["parameters"]
        if len(self.rows) != params["records"]:
            raise PlantListInconsistent(
                f"the plant list has {len(self.rows)} rows; the set says {params['records']} records"
            )
        kinds = Counter(row["kind"] for row in self.rows)
        if dict(kinds) != dict(self.settings["counts"]):
            raise PlantListInconsistent(
                f"rows per kind {dict(kinds)} differ from the set's counts {dict(self.settings['counts'])}"
            )
        seen_lines: dict[str, set[int]] = {}
        gen_file = self.settings["generation"]["file"]
        for row in self.rows:
            kind = row["kind"]
            if kind not in KINDS:
                raise _fail(row, f"unknown kind {kind!r}")
            if not isinstance(row["line"], int) or row["line"] < 1:
                raise _fail(row, "the line number is not a positive integer")
            if row["line"] in seen_lines.setdefault(row["file"], set()):
                raise _fail(row, "two rows name the same line of one file")
            seen_lines[row["file"]].add(row["line"])
            if kind in JOB_OF_KIND:
                if (row["source"] is not None or row["file"] != gen_file
                        or row["job"] != JOB_OF_KIND[kind]):
                    raise _fail(row, "a generated record has a source, a wrong file or a wrong job")
                if not row["parents"]:
                    raise _fail(row, "a generated record names no parent")
            else:
                source = row["source"]
                declared = self.settings["sources"].get(source) if isinstance(source, str) else None
                if declared is None or row["file"] != declared["file"]:
                    raise _fail(row, "the source is not one the set declares, or the file is not its file")
                if row["id"] != f"{source}-{row['line']:06d}":
                    raise _fail(row, "the id does not name the row's line in its source file")
                if row["parents"] or row["job"] is not None:
                    raise _fail(row, "a source row names parents or a job")
            if (row["parent_mode"] in PARENT_MODES) != bool(row["parents"]):
                raise _fail(row, "the parent mode and the parent list do not go together")
            for parent in row["parents"]:
                if parent not in self.by_id or parent == row["id"]:
                    raise _fail(row, f"the parent {parent!r} is not a row of the plant list")
            if (kind in PAIR_KINDS) != (row["pair"] is not None):
                raise _fail(row, "only an X copy or a pair member carries a pair id, and each must")
            case = row["sanitizer"]
            if (kind == "sanitizer-case") != (case is not None):
                raise _fail(row, "only a sanitizer case carries a sanitizer entry, and each must")
            if case is not None:
                skipped = case.get("skip")
                if skipped not in (True, False) or (case.get("reason") in SKIP_REASONS) != skipped:
                    raise _fail(row, "a skipped line needs one of the two drop reasons, a kept line none")
                if skipped != (row["text_sha256"] is None):
                    raise _fail(row, "a skipped line has no surviving text, a kept line has one")
            elif row["text_sha256"] is None:
                raise _fail(row, "the row has no text digest")
        for name, declared in self.settings["sources"].items():
            lines = seen_lines.get(declared["file"], set())
            if lines != set(range(1, declared["lines"] + 1)):
                raise PlantListInconsistent(
                    f"source {name}: the plant rows do not cover lines 1 to {declared['lines']}"
                )
        gen_rows = len(seen_lines.get(gen_file, set()))
        if gen_rows != self.settings["generation"]["rows"]:
            raise PlantListInconsistent("the generated rows differ in number from the set's count")

    # --- scope -------------------------------------------------------------------

    def _scope_of(self, record: str, path: tuple[str, ...] = ()) -> tuple[str | None, tuple[str, ...]]:
        """Why "withdraw <source>" puts a record in scope, and through which of the
        source's own records: ("own", ()), ("derived", roots) or (None, ())."""
        if record in self.scope:
            return self.scope[record]
        if record in path:
            raise PlantListInconsistent(f"the parents of {record!r} form a loop")
        row = self.by_id[record]
        if row["source"] == self.withdrawn:
            result: tuple[str | None, tuple[str, ...]] = ("own", ())
        elif not row["parents"]:
            result = (None, ())
        else:
            roots: set[str] = set()
            hit = []
            for parent in row["parents"]:
                why, parent_roots = self._scope_of(parent, path + (record,))
                hit.append(why is not None)
                if why == "own":
                    roots.add(parent)
                roots.update(parent_roots)
            inside = any(hit) if row["parent_mode"] == "all" else all(hit)
            result = ("derived", tuple(sorted(roots))) if inside else (None, ())
        self.scope[record] = result
        return result

    def _check_scope(self) -> None:
        for row in self.rows:
            why, roots = self._scope_of(row["id"])
            stated = row["withdraw_A"]
            want = {"in_scope": why is not None, "why": why, "roots": list(roots)}
            if not isinstance(stated, dict) or {**stated, "roots": sorted(stated.get("roots", []))} != want:
                raise _fail(
                    row,
                    f"the row states scope {stated}; its source and parents give {want}",
                )

    def in_scope(self, record: str) -> bool:
        return self.scope[record][0] is not None

    def _check_generated_kinds(self) -> None:
        """What a generated record is called against what its parents give.

        A paraphrase, translation or summary is a record made from the withdrawn
        source's records, so its parents must lead there. A paraphrase of an
        unrelated record names one unrelated source line, and its parents must lead
        to no record of the withdrawn source (DECISIONS.md D-23). The scope used here
        is the one worked out from the parents, not the one the row states."""
        for row in self.rows:
            kind = row["kind"]
            if kind not in JOB_OF_KIND:
                continue
            why = self.scope[row["id"]][0]
            if kind == KEPT_KIND:
                if why is not None:
                    raise _fail(
                        row,
                        f"the row is called a paraphrase of an unrelated record, and its "
                        f"parents lead to source {self.withdrawn}",
                    )
                parents = [self.by_id[parent] for parent in row["parents"]]
                if len(parents) != 1 or parents[0]["kind"] != "unrelated":
                    raise _fail(
                        row,
                        "a paraphrase of an unrelated record names exactly one parent, "
                        "an unrelated line of a source",
                    )
            elif why != "derived":
                raise _fail(
                    row,
                    f"the row is called a {kind} made from records of source "
                    f"{self.withdrawn}, and its parents do not lead there",
                )

    def generated(self) -> list[dict]:
        """The generated records: the rows that came from no source file."""
        return [row for row in self.rows if row["source"] is None]

    def generated_that_stay(self) -> list[dict]:
        """The generated records "withdraw <source>" does not reach, by the scope
        worked out from their parents. Neither the kind nor the id decides."""
        return [row for row in self.generated() if not self.in_scope(row["id"])]

    # --- identical records -------------------------------------------------------

    def _check_pairs(self) -> None:
        by_digest: dict[str, list[dict]] = {}
        for row in self.rows:
            if row["pair"] is not None:
                self.pairs.setdefault(row["pair"], []).append(row)
            if row["text_sha256"] is not None:
                by_digest.setdefault(row["text_sha256"], []).append(row)
        for pair, members in self.pairs.items():
            if len(members) != 2:
                raise PlantListInconsistent(f"pair {pair!r} has {len(members)} members, not two")
            a, b = members
            if a["source"] == b["source"] or a["kind"] != b["kind"]:
                raise PlantListInconsistent(f"pair {pair!r}: its members share a source or differ in kind")
            if a["twin"] != b["id"] or b["twin"] != a["id"]:
                raise PlantListInconsistent(f"pair {pair!r}: its members do not name each other")
            if a["text_sha256"] is None or a["text_sha256"] != b["text_sha256"]:
                raise PlantListInconsistent(f"pair {pair!r}: its members do not hold the same text")
        for digest, same in by_digest.items():
            named = {row["pair"] for row in same}
            if len(same) > 1 and (len(same) != 2 or None in named or len(named) != 1):
                ids = [row["id"] for row in same]
                raise PlantListInconsistent(
                    f"the records {ids} hold the same text and are not one planted pair"
                )

    # --- releases ----------------------------------------------------------------

    def _check_releases(self) -> None:
        stated_inputs: dict[str, set[str]] = {}
        for row in self.rows:
            if not isinstance(row["releases"], dict):
                raise _fail(row, "the releases entry is not a table")
            for release, entry in row["releases"].items():
                stated_inputs.setdefault(release, set()).add(row["id"])
                word = entry.get("withdraw_A") if isinstance(entry, dict) else None
                if word not in OUTCOMES:
                    raise _fail(
                        row,
                        f"in release {release} {word!r} is not an outcome; the outcomes are "
                        f"{', '.join(OUTCOMES)}",
                    )
        releases = self.settings["releases"]
        if set(stated_inputs) - set(releases):
            raise PlantListInconsistent(
                f"rows name releases {sorted(set(stated_inputs) - set(releases))} the set does not declare"
            )
        for name, declared in releases.items():
            inputs = expand(declared["inputs"])
            unknown = [i for i in inputs if i not in self.by_id]
            if unknown or len(set(inputs)) != len(inputs):
                raise PlantListInconsistent(
                    f"release {name}: an input is listed twice or is no plant row ({unknown[:3]})"
                )
            if set(inputs) != stated_inputs.get(name, set()):
                odd = sorted(set(inputs) ^ stated_inputs.get(name, set()))[:5]
                raise PlantListInconsistent(
                    f"release {name}: the set's inputs and the rows that name the release differ ({odd})"
                )
            self.release_inputs[name] = inputs
            self.expected[name] = self._outcomes(name, inputs, declared)
            for record in inputs:
                stated = self.by_id[record]["releases"][name]
                if stated != self.expected[name][record]:
                    raise _fail(
                        self.by_id[record],
                        f"in release {name} the row states {stated}; the set's structure gives "
                        f"{self.expected[name][record]}",
                    )

    def _check_kept_placement(self) -> None:
        """A paraphrase of an unrelated record is a row of a release, and the record
        it was made from is a row of the same release: both sides of a rebuild
        check then hold the pair (plan 5.1 "Sizes")."""
        for row in self.rows:
            if row["kind"] != KEPT_KIND:
                continue
            (parent,) = row["parents"]
            held = [
                name for name, expected in self.expected.items()
                if expected.get(row["id"], {}).get("row") == ROW_PRESENT
            ]
            if not held:
                raise _fail(row, "a paraphrase of an unrelated record is a row of no release")
            for name in held:
                if self.expected[name].get(parent, {}).get("row") != ROW_PRESENT:
                    raise _fail(
                        row,
                        f"its parent {parent} is not a row of release {name}, which holds the row",
                    )

    def _outcomes(self, name: str, inputs: list[str], declared: dict) -> dict[str, dict]:
        """What stands in the release, and what the rebuild does with each input
        under the withdrawal (plan 4.1): worked out from text digests, the dedup
        setting and priority, and scope."""
        dedup = declared["dedup"]
        priority = {source: i for i, source in enumerate(declared["dedup_priority"])}
        groups: dict[str, list[str]] = {}
        for record in inputs:
            digest = self.by_id[record]["text_sha256"]
            # A line with no text is never grouped with another (docs/plant.md).
            groups.setdefault(digest if digest is not None else f"line:{record}", []).append(record)
        out: dict[str, dict] = {}
        for members in groups.values():
            if not dedup or len(members) == 1:
                for record in members:
                    out[record] = {"row": ROW_PRESENT, "withdraw_A": self._plain(record)}
                continue
            sources = [self.by_id[m]["source"] for m in members]
            if None in sources or len(set(sources)) != len(sources) or set(sources) - set(priority):
                raise PlantListInconsistent(
                    f"release {name}: the identical inputs {members} cannot be ranked by dedup priority"
                )
            ranked = sorted(members, key=lambda m: priority[self.by_id[m]["source"]])
            winner, losers = ranked[0], ranked[1:]
            remaining = [m for m in losers if not self.in_scope(m)]
            for record in losers:
                out[record] = {"row": ROW_LOSER, "withdraw_A": NONE}
            if not self.in_scope(winner):
                out[winner] = {"row": ROW_PRESENT, "withdraw_A": KEEP}
            elif self.scope[winner][0] == "own" and remaining:
                out[winner] = {"row": ROW_PRESENT, "withdraw_A": RE_SOURCE, "resource_to": remaining[0]}
                out[remaining[0]] = {"row": ROW_LOSER, "withdraw_A": PROMOTE, "promoted_for": winner}
            else:
                out[winner] = {"row": ROW_PRESENT, "withdraw_A": self._plain(winner)}
        return out

    def _plain(self, record: str) -> str:
        """The outcome of a row no equivalent copy can stand in for: it stays when
        it is out of scope and is retired when it is in scope, as the withdrawn
        source's own record or as a record derived from one (DECISIONS.md D-19)."""
        return KEEP if self.scope[record][0] is None else RETIRE


# --- Counting ----------------------------------------------------------------------


def _ids(ids: list[str], expand_ids: bool) -> list[str]:
    return list(ids) if expand_ids else compress(ids)


def count_records(plant: Plant, expand_ids: bool) -> dict:
    own = [r["id"] for r in plant.rows if plant.scope[r["id"]][0] == "own"]
    derived = [r["id"] for r in plant.rows if plant.scope[r["id"]][0] == "derived"]
    omitted = [r["id"] for r in plant.rows if plant.in_scope(r["id"])]
    total = len(plant.rows)
    return {
        "records": {
            "total": total,
            "by_kind": dict(sorted(Counter(r["kind"] for r in plant.rows).items())),
            "by_source": dict(sorted(Counter(r["source"] or "generated" for r in plant.rows).items())),
            "in_scope": {"total": len(omitted), "own": len(own), "derived": len(derived)},
            "removed": len(omitted),
            "retained": total - len(omitted),
        },
        "clean_omits": {"count": len(omitted), "ids": _ids(omitted, expand_ids)},
    }


def count_generated(plant: Plant, expand_ids: bool) -> dict:
    """The generated records, split by whether the withdrawal reaches them, and the
    ones that stay as a list of their own (DECISIONS.md D-23). ``lines`` is filled
    from the chunk files when a chunk folder is given."""
    generated = plant.generated()
    stay = plant.generated_that_stay()
    return {
        "total": len(generated),
        "derived": len(generated) - len(stay),
        "stay": {
            "count": len(stay),
            "ids": _ids([row["id"] for row in stay], expand_ids),
            "list": [
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "job": row["job"],
                    "parents": list(row["parents"]),
                    "parent_mode": row["parent_mode"],
                    "in_scope": False,
                    "releases": {
                        name: expected[row["id"]]["withdraw_A"]
                        for name, expected in plant.expected.items() if row["id"] in expected
                    },
                    "lines": None,
                }
                for row in stay
            ],
        },
    }


def count_release(plant: Plant, name: str, expand_ids: bool) -> dict:
    inputs = plant.release_inputs[name]
    expected = plant.expected[name]
    declared = plant.settings["releases"][name]
    actions = {outcome: 0 for outcome in OUTCOMES}
    listed: dict[str, list[str]] = {outcome: [] for outcome in OUTCOMES}
    present = losers = in_scope_present = 0
    retire_by_why = {"own": 0, "derived": 0}
    for record in inputs:
        outcome = expected[record]["withdraw_A"]
        actions[outcome] += 1
        listed[outcome].append(record)
        if outcome == RETIRE:
            retire_by_why[plant.scope[record][0]] += 1
        if expected[record]["row"] == ROW_PRESENT:
            present += 1
            in_scope_present += plant.in_scope(record)
        else:
            losers += 1
    # Records removed: the rows that leave the successor. A re-sourced row keeps its
    # place, so it is not among them.
    removed = actions[RETIRE]
    retained = actions[KEEP] + actions[RE_SOURCE]
    # The same number by another road: build the release from scratch without the
    # withdrawn content and count its rows (plan 4.2).
    survivors = [r for r in inputs if not plant.in_scope(r)]
    if declared["dedup"]:
        texts = {plant.by_id[r]["text_sha256"] or f"line:{r}" for r in survivors}
        from_scratch = len(texts)
    else:
        from_scratch = len(survivors)
    # The generated inputs on their own. Their outcomes are counted from the
    # outcome of each input; how many of them the withdrawal does not reach is
    # counted from scope. The two must agree (``identities``).
    generated_inputs = [r for r in inputs if plant.by_id[r]["source"] is None]
    generated_actions = {outcome: 0 for outcome in OUTCOMES}
    for record in generated_inputs:
        generated_actions[expected[record]["withdraw_A"]] += 1
    generated_kept = [r for r in generated_inputs if expected[r]["withdraw_A"] == KEEP]
    out = {
        "inputs": len(inputs),
        "dedup": declared["dedup"],
        "rows_before": present,
        "in_scope_rows_before": in_scope_present,
        "dedup_losers": losers,
        "actions": actions,
        "retire_by_why": retire_by_why,
        "removed": removed,
        "retained": retained,
        "rows_after": retained,
        "rows_from_scratch": from_scratch,
        "successor_needed": any(actions[o] for o in CHANGING),
        "ids": {o: _ids(listed[o], expand_ids) for o in CHANGING},
        "generated": {
            "inputs": len(generated_inputs),
            "out_of_scope": sum(1 for r in generated_inputs if not plant.in_scope(r)),
            "actions": generated_actions,
            "kept_ids": _ids(generated_kept, expand_ids),
        },
    }
    if expand_ids:
        out["re_sourced"] = [
            {"row": r, "to": expected[r]["resource_to"]} for r in listed[RE_SOURCE]
        ]
    return out


def identities(records: dict, releases: dict) -> list[dict]:
    """The sums that must hold. A sum that does not hold stops the script."""
    out = [
        _identity("set: removed + retained = total",
                  records["removed"] + records["retained"], records["total"]),
        _identity("set: own + derived = in scope",
                  records["in_scope"]["own"] + records["in_scope"]["derived"],
                  records["in_scope"]["total"]),
    ]
    for name, r in releases.items():
        a = r["actions"]
        out += [
            _identity(f"{name}: every input has exactly one outcome", sum(a.values()), r["inputs"]),
            _identity(f"{name}: keep + re-source + retire = rows before",
                      a[KEEP] + a[RE_SOURCE] + a[RETIRE], r["rows_before"]),
            _identity(f"{name}: promote + none = dedup losers", a[PROMOTE] + a[NONE], r["dedup_losers"]),
            _identity(f"{name}: re-source = promote", a[RE_SOURCE], a[PROMOTE]),
            _identity(f"{name}: removed = rows in scope before - re-source",
                      r["removed"], r["in_scope_rows_before"] - a[RE_SOURCE]),
            _identity(f"{name}: rows after = rows of a from-scratch build without the source",
                      r["rows_after"], r["rows_from_scratch"]),
            _identity(f"{name}: generated rows kept = generated inputs the source does not reach",
                      r["generated"]["actions"][KEEP], r["generated"]["out_of_scope"]),
        ]
    broken = [i["name"] for i in out if not i["holds"]]
    if broken:
        raise PlantListInconsistent(f"the counts do not add up: {broken}")
    return out


def _identity(name: str, left: int, right: int) -> dict:
    return {"name": name, "left": left, "right": right, "holds": left == right}


def candidates(plant: Plant) -> list[dict]:
    """Check 6: for each release, the rows whose identical text reached it from
    more than one origin. At an unobserved copy step such a row has a set of
    candidate parents, and the answer is "may contain" for each candidate's source
    (plan 3.2, 3.3 "Copy steps", L3)."""
    out = []
    for name, inputs in plant.release_inputs.items():
        groups: dict[str, list[str]] = {}
        for record in inputs:
            digest = plant.by_id[record]["text_sha256"]
            if digest is not None:
                groups.setdefault(digest, []).append(record)
        for digest, members in groups.items():
            sources = sorted({plant.by_id[m]["source"] for m in members})
            if len(sources) < 2:
                continue
            out.append({
                "release": name,
                "pair": plant.by_id[members[0]]["pair"],
                "text_sha256": digest,
                "rows": [m for m in members if plant.expected[name][m]["row"] == ROW_PRESENT],
                "label": "candidates",
                "candidates": [{"id": m, "source": plant.by_id[m]["source"]} for m in members],
                "sources": sources,
                "may_contain_for": sources,
                "withdrawn_source_among_candidates": plant.withdrawn in sources,
            })
    return out


def sanitizer_drops(plant: Plant) -> dict[str, list[dict]]:
    """Check 3: per release, the lines the sanitizer must log as drops."""
    out: dict[str, list[dict]] = {}
    for name, inputs in plant.release_inputs.items():
        drops = []
        for record in inputs:
            row = plant.by_id[record]
            case = row["sanitizer"]
            if case and case["skip"] and plant.expected[name][record]["row"] == ROW_PRESENT:
                drops.append({
                    "id": record, "source": row["source"], "line": row["line"],
                    "case": case["case"], "reason": case["reason"],
                    "raw_sha256": case["raw_sha256"],
                })
        out[name] = drops
    return out


# --- Chunk files -------------------------------------------------------------------


def _natural(name: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def read_chunks(folder: Path) -> dict[str, list[tuple[str, str]]]:
    """Per shard, the (id, text digest) of each line of its tokenizer-input file.

    A shard's file is ``<shard>.jsonl``. A manifest (``*.manifest.jsonl``) and any
    file of another type are left alone."""
    if not folder.is_dir():
        raise UsageError(f"{folder.name} is not a folder of chunk files")
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.name.endswith(".jsonl") and not p.name.endswith(".manifest.jsonl")
    ]
    if not paths:
        raise ChunkFilesMismatch(f"{folder.name} holds no <shard>.jsonl file")
    shards: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(paths, key=lambda p: _natural(p.name)):
        shard = path.name[: -len(".jsonl")]
        lines = []
        with open(path, encoding="utf-8", newline="\n") as f:
            for number, line in enumerate(f, start=1):
                try:
                    row = json.loads(line)
                    record, text = row["id"], row["text"]
                    if not isinstance(record, str) or not isinstance(text, str):
                        raise TypeError
                except (ValueError, TypeError, KeyError) as exc:
                    raise ChunkFilesMismatch(
                        f"{path.name} line {number} is not a row with an id and a text; a chunk "
                        f"file holds surviving lines only"
                    ) from exc
                lines.append((record, hashlib.sha256(text.encode("utf-8")).hexdigest()))
        shards[shard] = lines
    return shards


def count_chunks(plant: Plant, shards: dict[str, list[tuple[str, str]]], layout: str,
                 releases: list[str], require_pairs: bool) -> tuple[dict, dict, dict, dict]:
    """The kept-shard list and the pair classification, from the chunk-file ids,
    where each generated record that stays has its lines, and for every record
    the (shard, line) of each chunk-file line that carries it."""
    expected: Counter = Counter()
    receiving: dict[str, list[str]] = {}  # re-sourced row id -> the promoted ids it brings
    for name in releases:
        for record in plant.release_inputs[name]:
            outcome = plant.expected[name][record]
            case = plant.by_id[record]["sanitizer"]
            if outcome["row"] == ROW_PRESENT and not (case and case["skip"]):
                expected[record] += 1
            if outcome["withdraw_A"] == RE_SOURCE:
                receiving.setdefault(record, []).append(outcome["resource_to"])

    seen: Counter = Counter()
    where: dict[str, list[tuple[str, int]]] = {}
    for shard, lines in shards.items():
        for number, (record, digest) in enumerate(lines, start=1):
            row = plant.by_id.get(record)
            if row is None:
                raise ChunkFilesMismatch(f"{shard} line {number}: the id {record!r} is no plant row")
            if row["text_sha256"] is None:
                raise ChunkFilesMismatch(
                    f"{shard} line {number}: {record} is a line the sanitizer must skip"
                )
            if digest != row["text_sha256"]:
                raise ChunkFilesMismatch(
                    f"{shard} line {number}: the text under id {record} is not that record's "
                    f"sanitized text"
                )
            seen[record] += 1
            where.setdefault(record, []).append((shard, number))
    if seen != expected:
        missing = sorted((expected - seen).elements())
        extra = sorted((seen - expected).elements())
        raise ChunkFilesMismatch(
            f"the chunk files do not hold exactly the surviving rows of {releases}: "
            f"{len(missing)} missing (first: {missing[:5]}), {len(extra)} not expected "
            f"(first: {extra[:5]})"
        )

    table = []
    keep, rewrite = [], []
    staying = {row["id"] for row in plant.generated_that_stay()}
    action_of: dict[str, str] = {}
    for shard, lines in shards.items():
        own = sum(1 for record, _ in lines if plant.scope[record][0] == "own")
        derived = sum(1 for record, _ in lines if plant.scope[record][0] == "derived")
        receives = sorted({to for record, _ in lines for to in receiving.get(record, [])})
        # A generated record that stays is no reason to rewrite its shard: it is
        # counted here and decides nothing.
        action = REWRITE if own or derived or receives else KEEP
        (rewrite if action == REWRITE else keep).append(shard)
        action_of[shard] = action
        table.append({
            "shard": shard, "lines": len(lines), "in_scope_lines": own + derived,
            "own": own, "derived": derived, "receives": len(receives),
            "generated_kept": sum(1 for record, _ in lines if record in staying),
            "action": action,
        })
    if layout == "round-robin":
        chunks = plant.settings["layouts"]["round-robin"]["chunks"]
        if len(shards) != chunks:
            raise LayoutExpectationFailed(
                f"the round-robin layout has {chunks} chunks; the folder holds {len(shards)} shards"
            )
        if keep:
            raise LayoutExpectationFailed(
                f"round-robin layout: the shards {keep} hold no record in scope, so not every "
                f"chunk is a rewrite (plan 4.5, 5.2, Check 9)"
            )
    elif not keep:
        raise LayoutExpectationFailed(
            "partitioned layout: every shard holds a record in scope, so the kept-shard list "
            "is empty and Check 9's question about kept shards would test nothing"
        )
    shard_part = {
        "layout": layout,
        "releases": releases,
        "lines": sum(seen.values()),
        "shards": table,
        "keep": keep,
        "rewrite": rewrite,
        "all_rewrite": not keep,
        "actions": {KEEP: len(keep), REWRITE: len(rewrite)},
    }

    listing = []
    tally: dict[str, Counter] = {"pair-member": Counter(), "x-copy": Counter()}
    for pair, members in plant.pairs.items():
        placed = [
            {"id": m["id"], "source": m["source"], "shard": shard, "line": number, "ordinal": number - 1}
            for m in members for shard, number in where.get(m["id"], [])
        ]
        per_shard = Counter(entry["shard"] for entry in placed)
        for entry in placed:
            entry["expected_label"] = "candidates" if per_shard[entry["shard"]] > 1 else "recovered"
        if not placed:
            placement = ABSENT
        elif len(placed) == 1:
            placement = SINGLE
        elif len(placed) == 2:
            placement = SAME_SHARD if len(per_shard) == 1 else SPLIT
        else:
            placement = MULTIPLE
        kind = members[0]["kind"]
        tally[kind][placement] += 1
        listing.append({"pair": pair, "kind": kind, "placement": placement, "members": placed})
    planted = tally["pair-member"]
    if require_pairs and not (planted[SAME_SHARD] and planted[SPLIT]):
        raise PairKindsMissing(
            f"Check 3 needs a planted pair in one shard and one split across two; the chunk files "
            f"give {planted[SAME_SHARD]} same-shard and {planted[SPLIT]} split "
            f"(other placements: {dict(planted)})"
        )
    pair_part = {
        "required": require_pairs,
        "counts": {
            kind: {p: counter[p] for p in (SAME_SHARD, SPLIT, SINGLE, ABSENT, MULTIPLE)}
            for kind, counter in tally.items()
        },
        "list": listing,
    }
    kept_lines = {
        record: [
            {"shard": shard, "line": number, "ordinal": number - 1,
             "shard_action": action_of[shard]}
            for shard, number in where.get(record, [])
        ]
        for record in sorted(staying)
    }
    return shard_part, pair_part, kept_lines, where


def documents(plant: Plant, shards: dict[str, list[tuple[str, str]]]) -> dict:
    """Checks 3 and 6: per shard, the planted record of each surviving chunk-file
    line. Line i of a chunk file (from 1) is document i - 1 of its shard, so the
    list of ids is the list of expected records by ordinal. ``count_chunks`` has
    already refused a line whose id is no plant row or whose text is not that
    record's sanitized text, so an id here stands for its planted record.

    ``identical`` lists the documents of one shard that hold the same text: no
    derivation can tell them apart, so each is expected with the label
    "candidates" and the whole group as its candidate set, and every other
    document with the label "recovered" (plan 3.3). The groups are found from the
    text digests of the chunk-file lines, not from the pair ids."""
    table = []
    for shard, lines in shards.items():
        by_text: dict[str, list[int]] = {}
        for ordinal, (_, digest) in enumerate(lines):
            by_text.setdefault(digest, []).append(ordinal)
        identical = [
            {
                "pair": plant.by_id[lines[ordinals[0]][0]]["pair"],
                "text_sha256": digest,
                "ordinals": ordinals,
                "lines": [ordinal + 1 for ordinal in ordinals],
                "ids": [lines[ordinal][0] for ordinal in ordinals],
            }
            for digest, ordinals in by_text.items() if len(ordinals) > 1
        ]
        twins = sum(len(group["ordinals"]) for group in identical)
        table.append({
            "shard": shard,
            "documents": len(lines),
            "ids": [record for record, _ in lines],
            "identical": identical,
            "labels": {"recovered": len(lines) - twins, "candidates": twins},
        })
    return {"total": sum(entry["documents"] for entry in table), "shards": table}


def translations(plant: Plant, expand_ids: bool,
                 where: dict[str, list[tuple[str, int]]] | None = None,
                 shards: dict[str, list[tuple[str, str]]] | None = None) -> dict:
    """Check 6: the planted translations, each with the releases that receive it
    as a row and, from the chunk-file ids, the shards that receive it. A
    translation whose making was not recorded has no recorded origin, and
    everything that holds one is asked what it answers (plan 5.5 Check 6, L5):
    the file the translation job wrote, these releases and these shards."""
    rows = [row for row in plant.rows if row["kind"] == "translation"]
    listing = []
    per_release = {name: 0 for name in plant.release_inputs}
    per_shard: Counter = Counter()
    for row in rows:
        receiving = [
            name for name, expected in plant.expected.items()
            if expected.get(row["id"], {}).get("row") == ROW_PRESENT
        ]
        for name in receiving:
            per_release[name] += 1
        lines = None
        if where is not None:
            lines = [
                {"shard": shard, "line": number, "ordinal": number - 1}
                for shard, number in where.get(row["id"], [])
            ]
            per_shard.update(entry["shard"] for entry in lines)
        listing.append({
            "id": row["id"],
            "parents": list(row["parents"]),
            "roots": list(plant.scope[row["id"]][1]),
            "releases": receiving,
            "lines": lines,
        })
    return {
        "count": len(rows),
        "ids": _ids([row["id"] for row in rows], expand_ids),
        "file": plant.settings["generation"]["file"],
        "job": JOB_OF_KIND["translation"],
        "releases": per_release,
        "list": listing,
        "shards": None if shards is None else [
            {"shard": shard, "translations": per_shard[shard]}
            for shard in shards if per_shard[shard]
        ],
        "shards_without": None if shards is None else [
            shard for shard in shards if not per_shard[shard]
        ],
    }


# --- Sanitizer-input files ---------------------------------------------------------


def read_sanitizer_input(folder: Path) -> dict[str, list[bytes]]:
    """Per shard, the lines of its sanitizer-input file as bytes, without the line
    break. A shard's file is ``<shard>.jsonl``; a manifest and any file of another
    type are left alone. A line ends at a line feed and at nothing else."""
    if not folder.is_dir():
        raise UsageError(f"{folder.name} is not a folder of sanitizer-input files")
    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.name.endswith(".jsonl") and not p.name.endswith(".manifest.jsonl")
    ]
    out: dict[str, list[bytes]] = {}
    for path in sorted(paths, key=lambda p: _natural(p.name)):
        lines = path.read_bytes().split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        out[path.name[: -len(".jsonl")]] = lines
    return out


def place_drops(plant: Plant, shards: dict[str, list[tuple[str, str]]],
                raw: dict[str, list[bytes]], releases: list[str]) -> dict:
    """Check 3: the shard and the place of every line the sanitizer must skip.

    A chunk file holds surviving lines only, so where a skipped line was read is
    taken from the sanitizer-input files: a line there whose bytes have the digest
    the plant list states for a line the sanitizer must skip (``raw_sha256``) is
    that line. Every other line must be the next line of the shard's chunk file,
    by its id: the surviving lines keep their order, so a skipped line shifts no
    ordinal. Anything else stops the script.

    ``ordinal_of_next_line`` is the number of surviving lines before the skipped
    one, which is the ordinal the next surviving line of the shard gets."""
    waiting: dict[str, list[dict]] = {}
    for name in releases:
        for drop in sanitizer_drops(plant)[name]:
            waiting.setdefault(drop["raw_sha256"], []).append(drop)
    table = []
    tally = {reason: 0 for reason in SKIP_REASONS}
    without = [shard for shard in shards if shard not in raw]
    if without:
        raise SanitizerInputMismatch(
            f"the chunk files {without[:5]} have no sanitizer input beside them"
        )
    for shard in sorted(set(raw) | set(shards), key=_natural):
        chunk = shards.get(shard, [])
        survivors = 0
        drops = []
        for number, line in enumerate(raw[shard], start=1):
            digest = hashlib.sha256(line).hexdigest()
            if waiting.get(digest):
                planted = waiting[digest].pop(0)
                tally[planted["reason"]] += 1
                drops.append({
                    "id": planted["id"], "source": planted["source"], "line": planted["line"],
                    "case": planted["case"], "reason": planted["reason"],
                    "raw_sha256": digest, "shard": shard, "input_line": number,
                    "ordinal_of_next_line": survivors,
                })
                continue
            try:
                record = json.loads(line.decode("utf-8"))["id"]
                if not isinstance(record, str):
                    raise TypeError
            except (ValueError, TypeError, KeyError) as exc:
                raise SanitizerInputMismatch(
                    f"{shard} sanitizer-input line {number} is neither a line the sanitizer "
                    f"must skip nor a row with an id"
                ) from exc
            if survivors >= len(chunk) or chunk[survivors][0] != record:
                raise SanitizerInputMismatch(
                    f"{shard} sanitizer-input line {number} ({record}) is not the next line of "
                    f"the chunk file (line {survivors + 1}): a surviving line is missing there "
                    f"or the order differs, so an ordinal would shift"
                )
            survivors += 1
        if survivors != len(chunk):
            raise SanitizerInputMismatch(
                f"{shard}: the chunk file holds {len(chunk)} lines and the sanitizer input "
                f"gives {survivors} surviving lines"
            )
        table.append({
            "shard": shard, "input_lines": len(raw[shard]), "surviving_lines": survivors,
            "drops": drops,
        })
    left = sorted(drop["id"] for group in waiting.values() for drop in group)
    if left:
        raise SanitizerInputMismatch(
            f"{len(left)} of the lines the sanitizer must skip are in no sanitizer-input file "
            f"(first: {left[:5]})"
        )
    return {"count": sum(tally.values()), "by_reason": tally, "shards": table}


# --- Entry points ------------------------------------------------------------------


def count(set_folder, *, chunks=None, layout: str | None = None,
          releases: list[str] | None = None, require_pair_kinds: str = "auto",
          expand_ids: bool = False, sanitizer_input=None) -> dict:
    """Everything this script counts, as one document."""
    if (chunks is None) != (layout is None):
        raise UsageError("give the chunk folder and its layout together, or neither")
    if sanitizer_input is not None and chunks is None:
        raise UsageError("the sanitizer-input folder is read beside a chunk folder; give both")
    if layout is not None and layout not in LAYOUTS:
        raise UsageError(f"layout must be one of {LAYOUTS}")
    if require_pair_kinds not in ("auto", "yes", "no"):
        raise UsageError("require_pair_kinds must be auto, yes or no")
    settings, rows = load_set(Path(set_folder))
    try:
        plant = Plant(settings, rows)
    except (KeyError, TypeError, AttributeError) as exc:
        raise PlantListInconsistent(f"the set or a plant row lacks a field: {exc!r}") from exc

    result: dict = {
        "format": COUNT_FORMAT,
        "request": {"kind": "withdraw", "source": plant.withdrawn},
        "set": dict(settings["parameters"]),
    }
    result.update(count_records(plant, expand_ids))
    result["generated"] = count_generated(plant, expand_ids)
    result["releases"] = {
        name: count_release(plant, name, expand_ids) for name in plant.release_inputs
    }
    result["identities"] = identities(result["records"], result["releases"])
    for name, release in result["releases"].items():
        part = release["generated"]["actions"]
        if part[RETIRE] and not part[KEEP]:
            raise NoGeneratedRecordStays(
                f"release {name} retires {part[RETIRE]} generated records and keeps none: a "
                f"report that retired every generated record would match its counts. A set "
                f"written by the current generator holds paraphrases of unrelated records "
                f"that stay (plan 5.1, DECISIONS.md D-23); generate the set again"
            )
    record_actions = {outcome: 0 for outcome in OUTCOMES}
    for release in result["releases"].values():
        for outcome, n in release["actions"].items():
            record_actions[outcome] += n
    result["actions"] = {"records": record_actions, "shards": None}
    result["check3"] = {"sanitizer_drops": sanitizer_drops(plant), "pairs": None, "drops": None}
    result["check6"] = {"candidates": candidates(plant),
                        "translations": translations(plant, expand_ids)}
    result["shards"] = None
    result["documents"] = None

    if chunks is not None:
        named = list(releases or ["R1"])
        for name in named:
            if name not in plant.release_inputs:
                raise UsageError(f"the set has no release {name!r}")
        if len(set(named)) != len(named):
            raise UsageError("a release is named twice")
        required = (
            require_pair_kinds == "yes"
            or (require_pair_kinds == "auto" and not settings["parameters"]["dedup"])
        )
        chunk_lines = read_chunks(Path(chunks))
        shard_part, pair_part, kept_lines, where = count_chunks(
            plant, chunk_lines, layout, named, required
        )
        for entry in result["generated"]["stay"]["list"]:
            entry["lines"] = kept_lines[entry["id"]]
        result["shards"] = shard_part
        result["actions"]["shards"] = shard_part["actions"]
        result["check3"]["pairs"] = pair_part
        result["documents"] = documents(plant, chunk_lines)
        result["check6"]["translations"] = translations(plant, expand_ids, where, chunk_lines)
        if sanitizer_input is not None:
            result["check3"]["drops"] = place_drops(
                plant, chunk_lines, read_sanitizer_input(Path(sanitizer_input)), named
            )
    elif releases:
        raise UsageError("a release is named but no chunk folder is given")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count a plant list into the expected answers of its set (see docs/plant.md)."
    )
    parser.add_argument("--set", required=True, dest="set_folder",
                        help="the folder generate.py wrote (set.toml and plant_list.jsonl)")
    parser.add_argument("--chunks", help="a folder of tokenizer-input files, one <shard>.jsonl per shard")
    parser.add_argument("--layout", choices=LAYOUTS, help="the layout of the chunk folder")
    parser.add_argument("--sanitizer-input", dest="sanitizer_input",
                        help="a folder of sanitizer-input files, one <shard>.jsonl per shard, read "
                             "beside --chunks to place each line the sanitizer must skip")
    parser.add_argument("--release", action="append", dest="releases",
                        help="a release whose rows the chunk folder holds (default R1; repeatable)")
    parser.add_argument("--require-pair-kinds", choices=("auto", "yes", "no"), default="auto",
                        help="fail unless a same-shard and a split pair exist (auto: when dedup is off)")
    parser.add_argument("--expand-ids", action="store_true", help="print id lists in full, not as ranges")
    args = parser.parse_args(argv)
    try:
        result = count(
            args.set_folder, chunks=args.chunks, layout=args.layout, releases=args.releases,
            require_pair_kinds=args.require_pair_kinds, expand_ids=args.expand_ids,
            sanitizer_input=args.sanitizer_input,
        )
    except CountError as exc:
        json.dump({"error": type(exc).__name__, "detail": str(exc)}, sys.stderr)
        sys.stderr.write("\n")
        return exc.exit_code
    json.dump(result, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
