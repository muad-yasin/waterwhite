"""The action of one record under a withdrawal (plan 4.1, Decisions 10 and 15;
DECISIONS.md D-19; docs/interfaces.md section 7.1, docs/rebuild.md section 2.1).

A record is judged by the verdict the proof gives it for the request, which
``coverage.Session.occurrence`` states, and by the edge that made it. This module
reads verdicts; it builds none (the scanner of the proof's construction holds it to
that): "clear" is asked of ``coverage.is_clear``, and a verdict that is not clear
says what it is by the chain or the failing link it carries.

- clear: **keep**, only where the proof shows the withdrawn source is not behind
  the record;
- a failing link ("may contain"): **unresolved**, no action;
- a chain with a hop of a file-level edge, a file-level link or a publisher's pointer
  (at any place in the chain), and so with no record of the source behind it:
  **unresolved** too, with the whole file as the scope to review;
- a chain that ends at a record of the source: **retire**, **re-source**, or, for
  the dedup loser of a re-sourced row, **promote**.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import coverage
from .. import vocabulary as v
from ..records import OccurrenceRef

# The link words of the hops that end a chain at no record: a link of a whole file,
# a link of a record to a member by a publisher's pointer, and an edge labelled
# file-level (plan 3.2, L1).
_WEAK_LINKS = frozenset({
    v.FILE_LEVEL, v.hop_source_link(v.FILE_LEVEL), v.hop_source_link(v.MAPPED),
})

# The missing link of a record whose only evidence is a publisher's pointer. The
# file-level sentence is the vocabulary's (the table's own words).
MISSING_LINK_POINTER = (
    "the record's own parent; only a publisher's pointer to {member} is recorded"
)
MISSING_LINK_NO_DETAIL = "a link of the chain that no record states"


@dataclass(frozen=True)
class Judgement:
    """What the report says about one record. ``action`` is a word of
    ``vocabulary.ACTIONS`` or ``vocabulary.UNRESOLVED``; the other fields are the
    keys of a record entry (docs/interfaces.md section 7.1), ``new_ref`` being the
    record a re-source takes its parent from, for the promote entry."""

    action: str
    chain: tuple[dict, ...] = ()
    new_parent: str | None = None
    withdrawn_parent: str | None = None
    regenerate_with: str | None = None
    step: dict | None = None
    missing_link: str | None = None
    failing: dict | None = None
    names: str | None = None
    review_scope: dict | None = None
    new_ref: OccurrenceRef | None = None


def key(session, ref: OccurrenceRef) -> str:
    """A record as ``artifact:locator``, the form a request's records use."""
    return f"{session.name(ref.instance_id)}:{ref.locator}"


KEEP = Judgement(v.ACTION_KEEP)


def judge(store, session, status, ref: OccurrenceRef | None) -> Judgement:
    """The judgement of a record whose verdict is ``status``. ``ref`` is the record
    whose own edge decides between retire and re-source: None for a document whose
    map row names no single parent (a set of candidates), which is retired."""
    if coverage.is_clear(status):
        return KEEP
    if status.failing is not None:
        return _unresolved_by_link(session, status.failing, ref)
    weak = _weak_hop(status.chain)
    if weak is not None:
        return _unresolved_by_file(store, session, status.chain, weak)
    if ref is not None:
        taken = _equivalent_parent(store, session, ref)
        if taken is not None:
            return taken(status)
    return _retire(store, status)


def is_unresolved(store, session, status, ref: OccurrenceRef | None) -> bool:
    """Whether the report counts the record unresolved: the judgement of ``judge``,
    which the report's entries read directly, read for its word. The verification of a
    successor calls it (and ``is_unresolved_document`` for a document of a shard, which
    also reads the document's own verdict), so that the verification is never less
    strict than the report. It is the judgement of ``judge`` and nothing beside it."""
    return judge(store, session, status, ref).action == v.UNRESOLVED


def judge_document(store, session, gate, row) -> Judgement:
    """The judgement of one document of a shard, through the map a reader gets. A
    document that is clear is kept. A row that names a single parent is judged by the
    verdict of that parent's record; any other row by the verdict of the document."""
    status = session.document(gate, row)
    if coverage.is_clear(status):
        return KEEP
    if row.parent is not None and row.label != v.CANDIDATES:
        return judge(store, session, session.occurrence(row.parent), row.parent)
    return judge(store, session, status, None)


def is_unresolved_document(store, session, gate, row) -> bool:
    """``is_unresolved`` for one document of a shard: true where the report's entry for
    the shard counts the document unresolved (``judge_document``), and also where the
    document's own verdict is unresolved by itself (a failing link or a weak hop in
    its own chain, which the verdict of a single parent's record may not carry)."""
    return (judge_document(store, session, gate, row).action == v.UNRESOLVED
            or is_unresolved(store, session, session.document(gate, row), None))


def _hops(chain) -> tuple[dict, ...]:
    return tuple(hop.to_json() for hop in chain)


def _weak_hop(chain):
    """The first hop that says the evidence is a whole file or a pointer."""
    for hop in chain:
        if hop.link in _WEAK_LINKS:
            return hop
    return None


def _unresolved_by_link(session, failing, ref: OccurrenceRef | None) -> Judgement:
    return Judgement(
        v.UNRESOLVED,
        missing_link=failing.text or MISSING_LINK_NO_DETAIL,
        failing=failing.to_json(),
        names=failing.names,
        review_scope={"artifact": None if ref is None else session.name(ref.instance_id),
                      "records": v.SCOPE_THE_RECORD, "count": 1},
    )


def _unresolved_by_file(store, session, chain, weak) -> Judgement:
    member = next((hop.member for hop in reversed(chain) if hop.member), None)
    named = member or session.request.source
    pointer = weak.link == v.hop_source_link(v.MAPPED)
    sentence = MISSING_LINK_POINTER if pointer else v.MISSING_LINK_FILE_LEVEL
    count = store.occurrence_count(weak.instance) if weak.instance else 0
    return Judgement(
        v.UNRESOLVED,
        missing_link=sentence.format(member=named),
        review_scope={"artifact": weak.artifact, "records": v.SCOPE_EVERY_RECORD_OF_FILE,
                      "count": count},
    )


def _equivalent_parent(store, session, ref: OccurrenceRef):
    """For a row made by an "any" edge whose selected parent is in scope: a function
    that takes the verdict and gives the re-source, or the retire or unresolved
    judgement the other parents decide. None for every other edge."""
    edge = store.edge_of(ref)
    if edge is None or edge.parent_mode != v.MODE_ANY or edge.label == v.CANDIDATES:
        return None
    selected = edge.selected
    if selected is None:
        return None
    others = [item.parent for item in edge.inputs if item.parent != selected]
    if not others:
        return None
    statuses = [(other, session.occurrence(other)) for other in others]
    for other, status in statuses:
        if coverage.is_clear(status):
            def re_source(own, other=other):
                return Judgement(
                    v.ACTION_RE_SOURCE, chain=_hops(own.chain),
                    new_parent=key(session, other), withdrawn_parent=key(session, selected),
                    new_ref=other,
                )
            return re_source
    # No equivalent parent is shown to remain.
    for other, status in statuses:
        if status.failing is not None:
            def undecided(own, status=status):
                return _unresolved_by_link(session, status.failing, ref)
            return undecided
    return None  # every other parent is in scope as well: the row is retired


def _retire(store, status) -> Judgement:
    """Retire, naming the generation job that made the record and its step where the
    chain crosses a generation edge: the nearest one (DECISIONS.md D-114, D-24)."""
    job, step = None, None
    for hop in status.chain:
        if hop.generation:
            edge = store.edge_of(OccurrenceRef(hop.instance, hop.record))
            if edge is not None:
                job = hop.job or edge.job_name
                step = {"name": edge.transform.name, "version": edge.transform.version,
                        "parameter_digest": edge.transform.param_digest}
            break
    return Judgement(v.ACTION_RETIRE, chain=_hops(status.chain), regenerate_with=job, step=step)


def promoted(store, session, release: str, kept: list[tuple[OccurrenceRef, Judgement]]) -> list[dict]:
    """The promote entries of a release: for each re-sourced row, the dedup loser the
    row's step dropped, when it is the parent the row takes. The drop's output is the
    release and its input one of the equivalent copies the edge names; the group key
    is not read, since a step may give its own."""
    losers = {drop.input for drop in store.drops_by_output(release)
              if drop.reason == v.DROP_DEDUP_LOSER}
    out: list[dict] = []
    seen: set[OccurrenceRef] = set()
    for _ref, judgement in kept:
        new = judgement.new_ref
        if judgement.action != v.ACTION_RE_SOURCE or new is None or new in seen:
            continue
        if new in losers:
            seen.add(new)
            out.append({"ref": new, "chain": judgement.chain})
    return out
