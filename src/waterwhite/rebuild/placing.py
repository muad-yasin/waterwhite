"""Where the records that return to a release land, and which shards already hold them
(plan 4.1, 5.5 Check 9; DECISIONS.md D-141 and D-143 to D-147; docs/rebuild.md 2.2;
docs/interfaces.md 7.1).

A RETURNING COPY is a record the report promotes, or re-sources a row onto (its
``new_parent``). Whether a shard that nothing of the source stands behind can still be
changed by such a copy depends on where the layout puts it, and the store records no
layout. So the report asks two things of the shards of one group (the shards of a named
layout, or all the shards no layout names):

- does a shard ALREADY HOLD the copy: a document of the shard that the report keeps as it
  is has a map parent from which the copy is reached by following only single-parent,
  record-level, non-generation edges (the copy itself, its parent, its parent's
  parent, ...). No "any" edge is followed and no equivalent copy is counted;
- where the caller states the placement of the layout, in which shard the copy lands.

A shard that receives a copy it does not hold is rewritten. Where no place is stated for a
copy that no shard of the group holds, every shard of the group whose own contents are
clean reads ``vocabulary.ACTION_CLEAN_PLACE_NOT_STATED`` and names the copies.

This module reads the entries ``artifacts.py`` built and the store; it judges no record
and builds no verdict (``records.judge_document`` does).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .. import vocabulary as v
from ..errors import UsageError
from ..records import OccurrenceRef
from . import counts as counting
from . import records as judging

# --- The placement the caller states ---------------------------------------------------


@dataclass(frozen=True)
class Placement:
    """A layout's rule, read from its file: the shard each source's copies land in
    (``shards``, by the source artifact's name) and the shard for every other copy
    (``otherwise``, None where none is stated). ``path`` and ``sha256`` say which file
    was read."""

    layout: str
    path: str
    sha256: str
    shards: Mapping[str, str]
    otherwise: str | None

    def stated(self) -> dict:
        return {"path": self.path, "sha256": self.sha256}


def _bad(path, why: str) -> UsageError:
    return UsageError(f"the placement file {path} {why}", path=str(path))


def read_placement(layout: str, path) -> Placement:
    """The placement file of a layout. A file that cannot be read, is not JSON or is not
    a placement of the form is a usage error."""
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        raise UsageError(f"the placement file cannot be read: {error.strerror}",
                         path=str(path)) from None
    try:
        document = json.loads(data.decode("utf-8"))
    except ValueError as error:
        raise _bad(path, f"is not JSON: {error}") from None
    if not isinstance(document, dict):
        raise _bad(path, "is not an object")
    if document.get("form") != v.FORM_PLACEMENT:
        raise _bad(path, f"is not of the form {v.FORM_PLACEMENT}")
    if document.get("by") != v.PLACEMENT_BY_SOURCE:
        raise _bad(path, f"places by {document.get('by')!r}; the only rule is {v.PLACEMENT_BY_SOURCE!r}")
    shards = document.get("shards")
    if not isinstance(shards, dict) or not all(
            isinstance(k, str) and k and isinstance(x, str) and x for k, x in shards.items()):
        raise _bad(path, "does not give `shards` as an object of source name to shard name")
    otherwise = document.get("otherwise")
    if otherwise is not None and (not isinstance(otherwise, str) or not otherwise):
        raise _bad(path, "does not give `otherwise` as a shard name or null")
    return Placement(layout, str(Path(path).resolve()), hashlib.sha256(data).hexdigest(),
                     dict(shards), otherwise)


def check_placements(placements: Mapping[str, Placement] | None,
                     layouts: Mapping[str, str]) -> dict[str, Placement]:
    """The placements the caller gave: each is for a layout that is named, and each shard
    it names (a value of ``shards``, then ``otherwise`` where that is not null) begins with
    that layout's prefix. A target outside the prefix is a usage error, raised here, from
    the file alone, before the store is read for any copy: whether a copy resolves to the
    target does not matter."""
    held: dict[str, Placement] = {}
    for name, placement in (placements or {}).items():
        if name not in layouts:
            raise UsageError(f"--placement names the layout {name!r}, which --layout does not name")
        prefix = layouts[name]
        targets = [*placement.shards.values(), *([] if placement.otherwise is None
                                                 else [placement.otherwise])]
        for target in targets:
            if not target.startswith(prefix):
                raise UsageError(
                    f"the placement file {placement.path} places records of the layout {name!r} "
                    f"in the shard {target!r}, which does not begin with the layout's prefix "
                    f"{prefix!r}", path=placement.path)
        held[name] = placement
    return held


# --- The copies that return -------------------------------------------------------------


class Returning:
    """The returning copies of a report, in the order they are first met."""

    def __init__(self, session) -> None:
        self._session = session
        self.copies: dict[str, OccurrenceRef] = {}

    def add(self, ref: OccurrenceRef) -> str:
        key = judging.key(self._session, ref)
        self.copies.setdefault(key, ref)
        return key


# --- The walk from a copy upward ----------------------------------------------------------


class Chains:
    """The chain of single-parent, record-level, non-generation edges above a record
    (the record first, its root last)."""

    def __init__(self, store) -> None:
        self._store = store
        self._memo: dict[OccurrenceRef, tuple[OccurrenceRef, ...]] = {}

    def _above(self, ref: OccurrenceRef) -> OccurrenceRef | None:
        edge = self._store.edge_of(ref)
        if (edge is None or len(edge.inputs) != 1 or edge.parent_mode == v.MODE_ANY
                or edge.is_generation or edge.label not in v.RECORD_LEVEL_LABELS):
            return None
        return edge.inputs[0].parent

    def of(self, ref: OccurrenceRef) -> tuple[OccurrenceRef, ...]:
        walked: list[OccurrenceRef] = []
        seen: set[OccurrenceRef] = set()
        tail: tuple[OccurrenceRef, ...] = ()
        current: OccurrenceRef | None = ref
        while current is not None:
            if current in self._memo:
                tail = self._memo[current]
                break
            if current in seen:
                break  # a chain that runs into itself ends here
            seen.add(current)
            walked.append(current)
            current = self._above(current)
        for item in reversed(walked):
            tail = (item, *tail)
            self._memo[item] = tail
        return self._memo[ref]


def _kept_parents(store, session, instance: str) -> list[OccurrenceRef]:
    """The map parents of the documents of a shard that the report keeps as they are. A
    shard with no usable map, a candidates row and a document the report changes give
    none."""
    gate = session.shard_gate(instance)
    if gate.missing is not None:
        return []
    found = []
    for row in store.map_rows(instance):
        if row.parent is None or row.label == v.CANDIDATES:
            continue
        if judging.judge_document(store, session, gate, row).action == v.ACTION_KEEP:
            found.append(row.parent)
    return found


def _source_name(store, chains: Chains, ref: OccurrenceRef) -> str | None:
    """The registered source file a copy comes from: its own artifact where that is one,
    else the root of its chain."""
    for candidate in (ref, chains.of(ref)[-1]):
        artifact = store.artifact(candidate.instance_id)
        if artifact is not None and artifact.kind == v.KIND_JSONL and artifact.role == v.ROLE_SOURCE:
            return artifact.names[0] if artifact.names else artifact.instance_id
    return None


def _target(store, chains: Chains, placement: Placement, ref: OccurrenceRef) -> str | None:
    """The shard the placement names for a copy, None where it names none."""
    source = _source_name(store, chains, ref)
    named = None if source is None else placement.shards.get(source)
    return named if named is not None else placement.otherwise


# --- The pass ------------------------------------------------------------------------------


def apply(store, session, shards: dict[str, dict], names_of_shards: Mapping[str, list[str]],
          layouts: Mapping[str, str], placements: Mapping[str, Placement],
          returning: Returning) -> dict[str, dict]:
    """Change the shard entries by where the copies land and which shard holds them, and
    return ``groups.placements``: for every named layout its stated placement, the shards
    outside the store that receive copies, and the shards that read the fourth word.

    ``shards`` maps instance id to entry, in the order of registration; entries are
    changed in place."""
    sections = {name: {"placement": placements[name].stated() if name in placements else None,
                       "landing": [], "place_not_stated": []} for name in layouts}
    copies = returning.copies
    if not copies:
        return sections
    named, rest = counting.membership(names_of_shards, layouts)
    groups: list[tuple[str | None, list[str]]] = [*named.items(), (None, rest)]
    by_name = {name: instance for instance, names in names_of_shards.items() for name in names}
    chains = Chains(store)
    reached: dict[str, set[OccurrenceRef]] = {}

    def held(instance: str) -> set[OccurrenceRef]:
        if instance not in reached:
            reached[instance] = {link for parent in _kept_parents(store, session, instance)
                                 for link in chains.of(parent)}
        return reached[instance]

    unplaced: dict[str | None, list[str]] = {}
    received: dict[str, list[str]] = {}
    for layout, members in groups:
        placement = placements.get(layout) if layout is not None else None
        if placement is None:
            every = set().union(*(held(i) for i in members)) if members else set()
            unplaced[layout] = [key for key, ref in copies.items() if ref not in every]
            continue
        unplaced[layout] = []
        landing: dict[str, list[str]] = {}
        for key, ref in copies.items():
            shard = _target(store, chains, placement, ref)
            if shard is None:
                unplaced[layout].append(key)
            elif shard not in by_name:
                landing.setdefault(shard, []).append(key)
            elif ref not in held(by_name[shard]):
                received.setdefault(by_name[shard], []).append(key)
        sections[layout]["landing"] = [
            {"shard": shard, "in_store": False, "receives": keys} for shard, keys in landing.items()]
        # The placement decides where a shard's own re-sourced documents land too.
        for instance in members:
            shards[instance]["receives"] = []

    for instance, entry in shards.items():
        got = list(dict.fromkeys(received.get(instance, ())))
        entry["receives"].extend(key for key in got if key not in entry["receives"])
        if got and entry["action"] == v.ACTION_KEEP:
            entry["action"] = v.ACTION_REWRITE
            entry["successor"] = v.SUCCESSOR_PREFIX + entry["artifact"]
        if got and entry["action"] == v.ACTION_REWRITE and entry["documents"] is not None:
            entry["documents_in_successor"] = (
                entry["documents"] - entry["records_removed"] + len(got))

    for layout, members in groups:
        left = unplaced[layout]
        if not left:
            continue
        for instance in members:
            entry = shards[instance]
            if entry["action"] != v.ACTION_KEEP:
                continue
            entry["action"] = v.ACTION_CLEAN_PLACE_NOT_STATED
            entry["returning"] = list(left)
            entry["place"] = v.PLACE_NOT_STATED
            if layout is not None:
                sections[layout]["place_not_stated"].append(entry["artifact"])
    return sections


def words(entries: Iterable[dict], kind: str) -> list[str]:
    """The names of the entries of a kind that read the fourth word."""
    return [e["artifact"] for e in entries
            if e["kind"] == kind and e["action"] == v.ACTION_CLEAN_PLACE_NOT_STATED]
