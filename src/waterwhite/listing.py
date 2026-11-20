"""Listings: one request put to every artifact the store holds (plan 2.1
"Listings", Decision 26, L10; DECISIONS.md D-24).

A listing states what it covers: the runs, mixtures, shards, checkpoints and
releases that have a record of their own. It prints an answer for every registered
artifact, whatever its kind and role, and an artifact with no record of its own
gets its answer too: "not covered". And it names in ``not_covered`` every artifact
that has no record of its own, registered or only named by another record, so a
gap is shown and not passed over:

- a shard with a row and no tokenization map, a JSONL file with a row and no
  occurrence record, a mixture with a row and no snapshot entry, a checkpoint
  whose run segment has no row (each of these is registered, so it also has an
  answer);
- a shard a mixture snapshot names, a mixture a run segment names, a checkpoint a
  run segment started from, an input of a step, when the store has no row for it
  (none of these is registered, so none has an answer);
- the mixture of a run segment that was registered with no snapshot (L6);
- a published model the store holds while the published-models extension is off
  (reason "not registered", with the note that the extension is off; it is registered,
  so it also has an answer).

An artifact that was never registered and that no record names cannot appear
(L10): the gap then shows as the difference between ``covers`` and the team's own
inventory.

The store's loud checks run over the whole store before anything is answered
(plan 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import coverage, extensions
from . import vocabulary as v
from .answers import Answer
from .requests import Withdrawal
from .store.views import (
    REL_ARTIFACT_INPUT,
    REL_CHECKPOINT_SEGMENT,
    REL_MIXTURE_SHARD,
    REL_SEGMENT_MIXTURE,
    REL_SEGMENT_PARENT,
    Artifact,
    Reference,
)

COVERS_KEYS = ("runs", "mixtures", "shards", "checkpoints", "releases")

# The order of a listing's answers: by kind, and inside a kind in the order of
# registration. A kind not named here follows these.
_ANSWER_ORDER = (v.KIND_JSONL, v.KIND_SHARD, v.KIND_MIXTURE, v.KIND_CHECKPOINT)

# The kind of thing each relation names.
_KIND_OF = {
    REL_MIXTURE_SHARD: v.KIND_SHARD,
    REL_SEGMENT_MIXTURE: v.KIND_MIXTURE,
    REL_SEGMENT_PARENT: v.KIND_CHECKPOINT,
    REL_ARTIFACT_INPUT: "input",
}


@dataclass(frozen=True)
class NamedBy:
    """One record that names an artifact: the relation, who names it (an artifact's
    name or a run segment's), and the position where the relation has one."""

    relation: str
    referrer: str
    position: int | None = None

    def to_json(self) -> dict:
        out: dict = {"relation": self.relation, "referrer": self.referrer}
        if self.position is not None:
            out["position"] = self.position
        return out


@dataclass(frozen=True)
class NotCovered:
    """An artifact with no record of its own. ``artifact`` is its name where it has
    one, else the path a snapshot gives for it, else its instance id; None for a
    mixture nothing identifies."""

    artifact: str | None
    instance: str | None
    kind: str
    reason: str
    named_by: tuple[NamedBy, ...] = ()
    detail: str | None = None

    def to_json(self) -> dict:
        return {
            "artifact": self.artifact,
            "instance": self.instance,
            "kind": self.kind,
            "reason": self.reason,
            "named_by": [n.to_json() for n in self.named_by],
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Listing:
    request: Withdrawal
    covers: dict[str, tuple[str, ...]]
    answers: tuple[Answer, ...]
    not_covered: tuple[NotCovered, ...]
    withheld: tuple = ()  # the answers that rest on an archived shard no check has passed

    def to_json(self) -> dict:
        made = {
            "request": self.request.to_json(),
            "covers": {key: list(self.covers[key]) for key in COVERS_KEYS},
            "answers": [a.to_json() for a in self.answers],
            "not_covered": [n.to_json() for n in self.not_covered],
        }
        if self.withheld:
            made["withheld"] = [w.to_json() for w in self.withheld]
        return made


def _shown(artifact: Artifact) -> str:
    return artifact.names[0] if artifact.names else artifact.instance_id


def _referrer(store, reference: Reference) -> str:
    """Who names: a run segment by its name, an artifact by a name of its own."""
    if reference.relation in (REL_SEGMENT_MIXTURE, REL_SEGMENT_PARENT):
        return reference.referrer
    held = store.artifact(reference.referrer)
    return reference.referrer if held is None else _shown(held)


def _named_by(store, referred: str) -> tuple[NamedBy, ...]:
    return tuple(
        NamedBy(r.relation, _referrer(store, r), r.position) for r in store.references_to(referred)
    )


def listing(store, request: Withdrawal) -> Listing:
    """Put one request to every artifact the store holds."""
    integrity = coverage.check_store(store)  # the whole store, before any answer
    session = coverage.Session(store, request)
    models_off = not extensions.is_on(store.settings(), extensions.PUBLISHED_MODELS)

    covers: dict[str, list[str]] = {key: [] for key in COVERS_KEYS}
    covers["runs"] = list(store.runs())
    gaps: list[NotCovered] = []

    # Every registered artifact is asked, by kind and inside a kind in the order
    # of registration. ``releases`` and the three lists beside it hold what has a
    # record of its own: that is what ``covers`` names.
    asked: dict[str, list[Artifact]] = {kind: [] for kind in _ANSWER_ORDER}
    releases, shards, mixtures, checkpoints = [], [], [], []
    for artifact in store.artifacts():
        asked.setdefault(artifact.kind, []).append(artifact)
        if artifact.kind == v.KIND_JSONL:
            if store.binding(artifact.instance_id) is None:
                gaps.append(NotCovered(
                    _shown(artifact), artifact.instance_id, v.KIND_JSONL, v.REASON_RECORD_MISSING,
                    _named_by(store, artifact.instance_id), "it has no occurrence record",
                ))
            elif artifact.role == v.ROLE_RELEASE:
                releases.append(artifact)
        elif artifact.kind == v.KIND_SHARD:
            if store.binding(artifact.instance_id) is None:
                gaps.append(NotCovered(
                    _shown(artifact), artifact.instance_id, v.KIND_SHARD, v.REASON_RECORD_MISSING,
                    _named_by(store, artifact.instance_id), "it has no tokenization map",
                ))
            else:
                shards.append(artifact)
        elif artifact.kind == v.KIND_MIXTURE:
            if not store.mixture_shards(artifact.instance_id):
                gaps.append(NotCovered(
                    _shown(artifact), artifact.instance_id, v.KIND_MIXTURE, v.REASON_RECORD_MISSING,
                    _named_by(store, artifact.instance_id), "its snapshot holds no shard",
                ))
            else:
                mixtures.append(artifact)
        elif artifact.kind == v.KIND_CHECKPOINT:
            mark = store.checkpoint(artifact.instance_id)
            if mark is None or store.segment(mark.segment) is None:
                gaps.append(NotCovered(
                    _shown(artifact), artifact.instance_id, v.KIND_CHECKPOINT, v.REASON_RECORD_MISSING,
                    _named_by(store, artifact.instance_id),
                    "its run segment has no record" if mark is None
                    else f"its run segment {mark.segment} has no record",
                ))
            else:
                checkpoints.append(artifact)
        elif artifact.kind == v.KIND_PUBLISHED_MODEL and models_off:
            gaps.append(NotCovered(
                _shown(artifact), artifact.instance_id, v.KIND_PUBLISHED_MODEL, v.REASON_NOT_REGISTERED,
                _named_by(store, artifact.instance_id), v.NOTE_PUBLISHED_MODELS_OFF,
            ))
    for key, held in (("releases", releases), ("shards", shards), ("mixtures", mixtures),
                      ("checkpoints", checkpoints)):
        covers[key] = [_shown(a) for a in held]

    # Named by another record, with no row of its own.
    by_target: dict[tuple[str, str], list[Reference]] = {}
    for reference in store.named_without_record():
        if reference.relation == REL_CHECKPOINT_SEGMENT:
            continue  # the checkpoint is listed above: its own record is that segment
        by_target.setdefault((_KIND_OF[reference.relation], reference.referred), []).append(reference)
    for (kind, referred), references in by_target.items():
        label = referred
        if kind == v.KIND_SHARD:
            first = references[0]
            for entry in store.mixture_shards(first.referrer):
                if entry.position == first.position:
                    label = entry.path
        gaps.append(NotCovered(
            label, referred, kind, v.REASON_RECORD_MISSING,
            tuple(NamedBy(r.relation, _referrer(store, r), r.position) for r in references),
            "the store has no row for it",
        ))

    # A segment registered with no mixture snapshot (L6): what its trainer was
    # handed is not recorded, and nothing identifies the mixture.
    for name in store.segments():
        segment = store.segment(name)
        if segment is not None and segment.mixture_instance_id is None:
            gaps.append(NotCovered(
                None, None, v.KIND_MIXTURE, v.REASON_NOT_REGISTERED,
                (NamedBy(REL_SEGMENT_MIXTURE, name),),
                f"run segment {name} was registered with no mixture snapshot",
            ))

    # One answer per registered artifact. An artifact with no record of its own
    # answers "not covered" here and stands in ``not_covered`` as well. A published
    # model, while its extension is off, answers "not covered" with the reason
    # "not registered" and the note that the extension is off, and stands in
    # ``not_covered`` too (a query that names it gives the same answer). With the switch on this build holds the off
    # state only, so the listing is refused with the error of its query: a listing
    # passes no registered artifact over.
    withheld: list = []

    def answer_for(artifact: Artifact) -> Answer | None:
        if artifact.kind == v.KIND_PUBLISHED_MODEL and models_off:
            return coverage.model_not_covered(artifact, _shown(artifact), request)
        return coverage.answer_or_withheld(store, _shown(artifact), request, sink=withheld,
                                           integrity=integrity, session=session)

    answers = tuple(a for a in (answer_for(artifact) for kind in asked for artifact in asked[kind])
                    if a is not None)
    return Listing(
        request=request,
        covers={key: tuple(value) for key, value in covers.items()},
        answers=answers,
        not_covered=tuple(gaps),
        withheld=tuple(withheld),
    )
