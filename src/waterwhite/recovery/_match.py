"""Matching the records of a later file to the records of its candidate inputs.

The rule (plan 3.1 "Fingerprint", 3.2, 3.3 "Copy steps"):

1. a record of the later file is a candidate child of every record of the named
   earlier files that has the same fingerprint. The fingerprints are the store's:
   no text is loaded to build the lookup;
2. for each such pair the tool reads both records from their files and compares
   their normalised texts directly. Equal fingerprints over different texts are a
   hash collision, and the candidate is dropped;
3. one candidate left: a "recovered" edge. Two or more left: a "candidates" edge
   that names every one of them. None left: no edge;
4. the count. A copy step and a sanitizer write a record at most once for each
   record they read. So when the later file holds a text more often than the
   named inputs hold it, at least one of the copies came through an input nobody
   named, and no rule can tell which. Then no copy of that text gets an edge:
   each keeps no recorded origin, and the run counts them as surplus.

Only the files the caller names are looked at. Nothing is searched by similarity
and the rest of the store is not searched at all (plan L5).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from .. import ids
from .. import vocabulary as v
from ..errors import MapCountMismatch
from ..records import EdgeInput, EdgeRecord, OccurrenceRef, StepRef
from ._files import RegisteredFile


def same_text(later: str, candidate: str) -> bool:
    """The direct comparison of plan 3.1: the two normalised texts themselves. This
    is what tells a real match from two texts that share a fingerprint."""
    return later == candidate


@dataclass
class Matched:
    """What one matching pass found. ``named`` holds, per earlier file (by its
    place in the caller's list), the lines an edge of this pass names."""

    edges: list[EdgeRecord] = field(default_factory=list)
    records: int = 0
    recovered: int = 0
    candidates: int = 0
    no_edge: int = 0
    no_text: int = 0
    surplus: int = 0
    collisions_rejected: int = 0
    named: set[tuple[int, int]] = field(default_factory=set)


def stored_fingerprints(store, file: RegisteredFile) -> list[bytes | None]:
    """The fingerprint the store holds for each line of a file, in line order."""
    found: list[bytes | None] = []
    for occurrence in store.occurrences(file.instance_id):
        if occurrence.ref.locator != ids.line_locator(len(found) + 1):
            raise MapCountMismatch(
                f"{file.name}: the store's records are not its lines 1, 2, 3 and so on "
                f"(found {occurrence.ref.locator} at place {len(found) + 1})",
                artifact=file.name,
            )
        found.append(occurrence.fingerprint)
    return found


def _lookup(
    store,
    earlier: Sequence[RegisteredFile],
    wanted: Collection[bytes],
    excluded: Collection[tuple[int, int]],
) -> dict[bytes, list[tuple[int, int]]]:
    """Fingerprint to (file, line) for the records of the earlier files, from the
    store's occurrences. Only fingerprints the later file holds are kept. The
    order inside one fingerprint is the caller's file order, then line order, so
    the same call builds the same edges."""
    table: dict[bytes, list[tuple[int, int]]] = {}
    for index, file in enumerate(earlier):
        for occurrence in store.occurrences(file.instance_id):
            fingerprint = occurrence.fingerprint
            if fingerprint is None or fingerprint not in wanted:
                continue
            line = ids.parse_locator(occurrence.ref.locator)[1]
            if (index, line) in excluded:
                continue
            table.setdefault(fingerprint, []).append((index, line))
    return table


def _edge(
    child_line: int, parents: Sequence[OccurrenceRef], step: StepRef, derivation: str
) -> EdgeRecord:
    if len(parents) == 1:
        return EdgeRecord(
            child_locator=ids.line_locator(child_line),
            inputs=(EdgeInput(parents[0]),),
            transform=step,
            parent_mode=v.MODE_ALL,
            derivation=derivation,
            verification=v.VERIFIED_PASSED,
            label=v.RECOVERED,
        )
    # Equivalent copies the derivation cannot tell apart: every one is named, and
    # none is marked as the one the step took (plan 3.2, L3).
    return EdgeRecord(
        child_locator=ids.line_locator(child_line),
        inputs=tuple(EdgeInput(parent) for parent in parents),
        transform=step,
        parent_mode=v.MODE_ANY,
        derivation=derivation,
        verification=v.VERIFIED_PASSED,
        label=v.CANDIDATES,
    )


def match_records(
    store,
    later: RegisteredFile,
    earlier: Sequence[RegisteredFile],
    *,
    step: StepRef,
    derivation: str,
    excluded: Collection[tuple[int, int]] = (),
) -> Matched:
    """Match every record of ``later`` against the records of ``earlier``.

    ``excluded`` names (place in ``earlier``, line) pairs that cannot be a parent:
    the lines a step's own log says it wrote nothing for. Nothing is written
    here; the caller hands the edges to the store."""
    fingerprints = stored_fingerprints(store, later)
    wanted = {fingerprint for fingerprint in fingerprints if fingerprint is not None}
    table = _lookup(store, earlier, wanted, frozenset(excluded))

    found = Matched()
    copies: dict[tuple[int, int], tuple[int, int]] = {}
    matched: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for raw in later.stream():
        found.records += 1
        number = found.records
        if number > len(fingerprints):
            later.verify_digest()  # a changed file is named as that first
            raise MapCountMismatch(
                f"{later.name}: the file has more lines than the {len(fingerprints)} "
                "records the store holds for it",
                artifact=later.name,
            )
        text, fingerprint = later.record(raw, number)
        if fingerprint != fingerprints[number - 1]:
            raise later.stored_fingerprint_differs(number)
        if fingerprint is None:
            found.no_text += 1
            found.no_edge += 1
            continue
        places: list[tuple[int, int]] = []
        for index, line in table.get(fingerprint, ()):
            file = earlier[index]
            candidate_text, candidate_fingerprint = file.record(file.raw_line(line), line)
            if candidate_fingerprint != fingerprint:
                raise file.stored_fingerprint_differs(line)
            if same_text(text, candidate_text):
                places.append((index, line))
            else:
                found.collisions_rejected += 1
        if not places:
            found.no_edge += 1
            continue
        # Records with one text have the same confirmed candidates, and records
        # with different texts share none, so the first candidate stands for the
        # text: how often the later file holds it, against how many inputs do.
        seen = copies.get(places[0])
        copies[places[0]] = (1 if seen is None else seen[0] + 1, len(places))
        matched.append((number, tuple(places)))
    for number, places in matched:
        held, parents = copies[places[0]]
        if held > parents:
            # More copies than the named inputs hold: an input nobody named.
            found.surplus += 1
            found.no_edge += 1
            continue
        if parents == 1:
            found.recovered += 1
        else:
            found.candidates += 1
        found.named.update(places)
        refs = [OccurrenceRef(earlier[index].instance_id, ids.line_locator(line))
                for index, line in places]
        found.edges.append(_edge(number, refs, step, derivation))
    if found.records != len(fingerprints):
        raise MapCountMismatch(
            f"{later.name}: the file has {found.records} lines and the store holds "
            f"{len(fingerprints)} records for it",
            artifact=later.name,
        )
    return found
