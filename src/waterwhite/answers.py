"""The answer type and its JSON form (docs/interfaces.md section 6).

One of four answers per artifact (plan 2.1): "contains", "may contain", "does not
contain", "not covered". This module holds the types the proof hands back (a
``Proof`` or a ``Finding``) and the ``Answer`` built from one of them.

The rule this module enforces (plan W6, Decision 1): an ``Answer`` that says "does
not contain" exists only around a ``Proof``, and a ``Proof`` is made in exactly one
place, ``waterwhite.coverage.prove``. How:

- ``Proof`` has no usable constructor. Calling it raises.
- The one function that can make a proof (the mint) is handed out once, by
  ``claim_proof_mint``, and only to the module ``waterwhite.coverage``. A second
  claim, or a claim from another module, raises.
- Every proof the mint makes is remembered, with the store it was made from and the
  state of that store when the walk was made. ``Answer`` checks, when it is built and
  again when it is written as JSON, that its proof is one of those, that the proof
  is about the same artifact and the same request, that the name the answer shows
  is a name of that artifact (or its instance id), and that the notes are the
  proof's own. When it is built it checks also that the store has not been written
  since the walk: a proof speaks for the store as it was, and a proof the store has
  outlived ends in ``StoreWrittenDuringQuery`` (docs/interfaces.md section 2). An
  object shaped like a proof that the mint did not make is refused, and so is a
  "does not contain" with no proof at all, with ``ProofRequired``, as is a proof of
  a store that is closed.

So a query that forgets a condition cannot produce "does not contain": without a
proof from the walk there is nothing to build it from.

Plan sections: 2.1 (the four answers), 2.2 (proof or first failing link), 2.4
(basis, first exposure, untraced steps), Decisions 1, 8, 20, 21, limits L8, L12.
"""

from __future__ import annotations

import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field

from . import vocabulary as v
from .errors import StoreWrittenDuringQuery
from .requests import Withdrawal

_MINT_OWNER = "waterwhite.coverage"

_NAMED_CAUSES = frozenset(v.NAMED_CAUSES)


class ProofRequired(TypeError):
    """An answer "does not contain" was asked for without a proof made by
    ``coverage.prove`` for the same artifact and request."""


# --- Links of a chain, and the link a failed proof names -----------------------------


@dataclass(frozen=True)
class Hop:
    """One hop of an evidence chain, from the artifact toward the source record.

    A record hop names an artifact and a locator; a trace hop names a run segment
    and a step. ``link`` says how the hop is tied to the next one: the evidence
    label of the edge or map row that leaves it, "member" on the last hop of a
    chain that ends at a record of a requested member, "file-level link" where a
    whole-file link ends it, "consumed" on a trace hop."""

    artifact: str | None = None
    instance: str | None = None
    record: str | None = None
    segment: str | None = None
    step: int | None = None
    link: str | None = None
    generation: bool = False
    job: str | None = None
    member: str | None = None

    @property
    def text(self) -> str:
        """The hop as the hand-written table spells it: ``artifact:locator`` or
        ``segment:step n``."""
        if self.segment is not None:
            return f"{self.segment}:step {self.step}"
        return f"{self.artifact}:{self.record}"

    def to_json(self) -> dict:
        out: dict = {"hop": self.text}
        for key in ("artifact", "instance", "record", "segment", "step", "link", "job", "member"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.generation:
            out["generation"] = True
        return out


@dataclass(frozen=True)
class Failing:
    """The first link a proof failed at (plan 2.2), and why.

    ``names`` is the named cause, a ``NAMES_*`` word. ``condition`` is the letter of
    the condition of plan 2.2, None where the link is a missing record upstream
    (plan 2.1). ``artifact`` and ``record`` say where the named record lies;
    ``segment`` and ``step`` name a step; ``job`` a generation job. ``detail`` is the
    reason beside the cause where the vocabulary fixes one (``TRACE_MISSING_*``);
    ``reason`` is "record missing" or "not registered" for a missing record
    upstream; ``text`` is a sentence for a person."""

    names: str
    condition: str | None = None
    artifact: str | None = None
    instance: str | None = None
    record: str | None = None
    segment: str | None = None
    step: int | None = None
    job: str | None = None
    detail: str | None = None
    reason: str | None = None
    candidates: tuple[str, ...] = ()
    text: str | None = None

    def to_json(self) -> dict:
        out: dict = {}
        for key in ("condition", "artifact", "instance", "record", "segment", "step", "job", "reason"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.text:
            out["detail"] = self.text
        return out


# --- What a negative answer is scoped to ---------------------------------------------


@dataclass(frozen=True)
class LogEntry:
    """The harness log a negative rests on for one generation job (plan L8):
    the log's digest, how many records the job was handed and how many reads the
    log lists, the attempts the harness refused (they do not break the bound),
    the model entry, which is a logged attribute while the generating-model
    extension is off (and is still printed here when it is on, where the proof
    also reads it), and which harness wrote the log ("in-process" or
    "confined"; None when the log does not say). The bound is read the same way
    for both; the word tells a reader what it rests on (plan F10)."""

    job: str
    log_digest: str
    handed: int
    reads: int
    refused_attempts: tuple[str, ...]
    model: str | None
    harness: str | None = None

    def to_json(self) -> dict:
        return {
            "job": self.job,
            "log_digest": self.log_digest,
            "handed": self.handed,
            "reads": self.reads,
            "refused_attempts": list(self.refused_attempts),
            "model": self.model,
            "harness": self.harness,
        }


@dataclass(frozen=True)
class Scope:
    """What a "does not contain" covers and nothing beyond (plan 2.1, 2.4, L8, L12):
    the source and the members registered under it when the query ran, the records
    the request was narrowed to, for a checkpoint the runs and the steps of its
    history, and the generation jobs whose established bound the proof rests on,
    with their logs. ``walked`` counts the records of the artifact the proof
    visited (documents a checkpoint's steps consumed, for a checkpoint).
    ``files_not_checked`` names the artifacts of the ancestor closure whose file
    was not at its recorded place when the proof ran: for those the digest checks
    compared stored digests only (docs/store.md, "States that are not faults")."""

    source: str
    members: tuple[str, ...]
    records: tuple[str, ...] = ()
    runs: tuple[str, ...] = ()
    steps: tuple[tuple[str, int, int], ...] = ()
    generation_edges: tuple[str, ...] = ()
    harness_logs: tuple[LogEntry, ...] = ()
    walked: int = 0
    files_not_checked: tuple[str, ...] = ()

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "members": list(self.members),
            "records": list(self.records),
            "runs": list(self.runs),
            "steps": [list(s) for s in self.steps],
            "generation_edges": list(self.generation_edges),
            "harness_logs": [entry.to_json() for entry in self.harness_logs],
            "walked": self.walked,
            "files_not_checked": list(self.files_not_checked),
        }


# --- The proof and its one maker ------------------------------------------------------


class Proof:
    """A coverage proof: every condition of plan 2.2 held for one artifact and one
    request, and no record in scope was found. Made by ``coverage.prove`` only."""

    __slots__ = (
        "instance_id", "kind", "names", "request", "scope", "notes", "passed_generation", "__weakref__",
    )

    instance_id: str
    kind: str
    names: tuple[str, ...]
    request: Withdrawal
    scope: Scope
    notes: tuple[str, ...]
    passed_generation: bool

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ProofRequired("a coverage proof is made by waterwhite.coverage.prove and nowhere else")

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a coverage proof cannot be changed")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("a coverage proof cannot be changed")

    def __reduce__(self):
        raise TypeError("a coverage proof cannot be copied or pickled; prove again")

    def __repr__(self) -> str:
        return f"<Proof {self.instance_id[:12]} {self.request.describe()}>"


_CURRENT, _WRITTEN, _UNKNOWN = "current", "written", "unknown"


def _proof_gate():
    """The mint and the register of what it made, closed over so that no module
    attribute leads to either."""
    minted: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # proof -> "store unchanged?"
    claimed: list[str] = []

    def claim():
        """Hand out the one function that makes proofs. Once, to the proof module."""
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if caller != _MINT_OWNER and not caller.startswith(_MINT_OWNER + "."):
            raise ProofRequired(
                f"only {_MINT_OWNER} may make coverage proofs; {caller or 'this caller'} may not"
            )
        if claimed:
            raise ProofRequired(f"the proof mint was already claimed by {claimed[0]}")
        claimed.append(caller)

        def mint(
            *,
            instance_id: str,
            kind: str,
            names: tuple[str, ...],
            request: Withdrawal,
            scope: Scope,
            notes: tuple[str, ...],
            passed_generation: bool,
            current: Callable[[], bool],
        ) -> Proof:
            """``names`` are the artifact's names when the walk was made; ``current``
            says whether the store is still as the walk read it."""
            proof = object.__new__(Proof)
            for key, value in (
                ("instance_id", instance_id),
                ("kind", kind),
                ("names", tuple(names)),
                ("request", request),
                ("scope", scope),
                ("notes", tuple(notes)),
                ("passed_generation", bool(passed_generation)),
            ):
                object.__setattr__(proof, key, value)
            minted[proof] = current
            return proof

        return mint

    def is_minted(candidate: object) -> bool:
        return isinstance(candidate, Proof) and candidate in minted

    def state_of(candidate: object) -> str:
        """``_CURRENT`` for a minted proof whose store is as the walk read it,
        ``_WRITTEN`` for one whose store was written since, and ``_UNKNOWN`` for a
        proof the mint did not make or whose store cannot say what it holds (closed
        or gone: a mistake of the caller, not of the moment)."""
        if not is_minted(candidate):
            return _UNKNOWN
        try:
            return _CURRENT if minted[candidate]() else _WRITTEN
        except Exception:
            return _UNKNOWN

    return claim, is_minted, state_of


claim_proof_mint, _is_minted, _state_of = _proof_gate()


# --- What the proof hands back when it does not hold ---------------------------------


@dataclass(frozen=True)
class Finding:
    """What ``coverage.prove`` returns in place of a proof: a recorded chain that
    shows "contains", the first failing link ("may contain"), or a missing record
    of the artifact itself ("not covered"). ``answer`` is never "does not
    contain"."""

    answer: str
    instance_id: str | None
    kind: str | None
    request: Withdrawal
    reason: str | None = None
    failing: Failing | None = None
    candidates: tuple[str, ...] = ()
    chain: tuple[Hop, ...] = ()
    basis: str | None = None
    notes: tuple[str, ...] = ()
    untraced_steps: tuple[tuple[str, int], ...] = ()
    first_exposure: tuple[str, int] | None = None
    level: str | None = None

    def __post_init__(self) -> None:
        if self.answer not in (v.CONTAINS, v.MAY_CONTAIN, v.NOT_COVERED):
            raise ProofRequired(
                f"a finding says contains, may contain or not covered; {self.answer!r} needs a proof"
            )


# --- The answer ------------------------------------------------------------------------


def _check(answer: "Answer", *, built: bool = False) -> None:
    """The rules every answer keeps. Run when an answer is built (``built``) and when
    it is written. Only a build asks the store whether it is as the proof saw it: an
    answer written later, once its store is closed, is the answer that was built."""
    if answer.answer not in v.ANSWERS:
        raise ValueError(f"not one of the four answers: {answer.answer!r}")
    if answer.answer == v.DOES_NOT_CONTAIN:
        proof = answer.proof
        if not _is_minted(proof):
            raise ProofRequired(
                '"does not contain" is built from a proof made by coverage.prove; '
                f"got {type(proof).__name__}"
            )
        if proof.instance_id != answer.instance or proof.request != answer.request:
            raise ProofRequired("the proof is about another artifact or another request")
        if answer.artifact != proof.instance_id and answer.artifact not in proof.names:
            raise ProofRequired(
                f"the answer shows the name {answer.artifact!r}, which is no name of the "
                "artifact its proof is about"
            )
        if tuple(answer.notes) != proof.notes:
            raise ProofRequired("a negative answer carries the notes of its proof and no others")
        if (
            answer.reason is not None
            or answer.names is not None
            or answer.failing is not None
            or answer.chain
            or answer.candidates
            or answer.basis is not None
            or answer.first_exposure is not None
        ):
            raise ValueError('"does not contain" names no failing link, chain or exposure')
        if built:
            # Last, so that a forged proof or a proof of another artifact is refused as
            # such whatever the store holds: only a proof the mint made can be stale.
            state = _state_of(proof)
            if state == _WRITTEN:
                raise StoreWrittenDuringQuery()
            if state != _CURRENT:
                raise ProofRequired(
                    "the store the proof was made from is closed or gone; prove again"
                )
        return
    if answer.proof is not None:
        raise ValueError(f'"{answer.answer}" carries no proof')
    if answer.answer == v.CONTAINS:
        if not answer.chain:
            raise ValueError('"contains" comes with its chain of evidence')
        if answer.reason is not None or answer.names is not None or answer.failing is not None:
            raise ValueError('"contains" names no failing link')
    elif answer.answer == v.MAY_CONTAIN:
        if answer.failing is None or answer.names not in _NAMED_CAUSES:
            raise ValueError('"may contain" names the first failing link and its named cause')
        if answer.reason is not None or answer.chain:
            raise ValueError('"may contain" carries no chain and no reason')
    else:  # not covered
        if answer.reason not in v.NOT_COVERED_REASONS:
            raise ValueError('"not covered" says "record missing" or "not registered"')
        if answer.names is not None or answer.failing is not None or answer.chain:
            raise ValueError('"not covered" names no link and no chain')


@dataclass(frozen=True)
class Answer:
    """One answer about one artifact under one request (plan 2.1).

    ``artifact`` is the name the query used. Build an answer with ``from_proof`` or
    ``from_finding``; a hand-built "does not contain" is refused."""

    artifact: str
    instance: str | None
    kind: str | None
    request: Withdrawal
    answer: str
    reason: str | None = None
    names: str | None = None
    names_detail: str | None = None
    failing: Failing | None = None
    candidates: tuple[str, ...] = ()
    chain: tuple[Hop, ...] = ()
    basis: str | None = None
    notes: tuple[str, ...] = ()
    untraced_steps: tuple[tuple[str, int], ...] = ()
    first_exposure: tuple[str, int] | None = None
    level: str | None = None
    proof: Proof | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _check(self, built=True)

    @property
    def scope(self) -> Scope | None:
        """What a "does not contain" is scoped to: its proof's scope. None for the
        other three answers."""
        return None if self.proof is None else self.proof.scope

    @property
    def is_negative(self) -> bool:
        """Whether this is the negative answer. A read only: the answer was built
        from a proof when it was made, and asking changes nothing."""
        return self.answer == v.DOES_NOT_CONTAIN

    @classmethod
    def from_proof(cls, proof: Proof, *, artifact: str) -> "Answer":
        """The one way to "does not contain"."""
        if not _is_minted(proof):
            raise ProofRequired(
                '"does not contain" is built from a proof made by coverage.prove; '
                f"got {type(proof).__name__}"
            )
        return cls(
            artifact=artifact,
            instance=proof.instance_id,
            kind=proof.kind,
            request=proof.request,
            answer=v.DOES_NOT_CONTAIN,
            notes=proof.notes,
            proof=proof,
        )

    @classmethod
    def from_finding(cls, finding: Finding, *, artifact: str) -> "Answer":
        failing = finding.failing
        return cls(
            artifact=artifact,
            instance=finding.instance_id,
            kind=finding.kind,
            request=finding.request,
            answer=finding.answer,
            reason=finding.reason,
            names=None if failing is None else failing.names,
            names_detail=None if failing is None else failing.detail,
            failing=failing,
            candidates=finding.candidates,
            chain=finding.chain,
            basis=finding.basis,
            notes=finding.notes,
            untraced_steps=finding.untraced_steps,
            first_exposure=finding.first_exposure,
            level=finding.level,
        )

    @classmethod
    def from_result(cls, result: "Proof | Finding", *, artifact: str) -> "Answer":
        if isinstance(result, Finding):
            return cls.from_finding(result, artifact=artifact)
        return cls.from_proof(result, artifact=artifact)

    def to_json(self) -> dict:
        """The JSON form of docs/interfaces.md section 6, with ``names_detail`` (the
        reason beside a named cause) and ``level`` (consumed or inherited, for a
        checkpoint that contains)."""
        _check(self)
        scope = self.scope
        exposure = self.first_exposure
        return {
            "artifact": self.artifact,
            "instance": self.instance,
            "kind": self.kind,
            "request": self.request.to_json(),
            "answer": self.answer,
            "reason": self.reason,
            "names": self.names,
            "names_detail": self.names_detail,
            "failing": None if self.failing is None else self.failing.to_json(),
            "candidates": list(self.candidates),
            "chain": [hop.to_json() for hop in self.chain],
            "basis": self.basis,
            "notes": list(self.notes),
            "untraced_steps": [[segment, step] for segment, step in self.untraced_steps],
            "first_exposure": None
            if exposure is None
            else {"segment": exposure[0], "step": exposure[1]},
            "level": self.level,
            "scope": None if scope is None else scope.to_json(),
        }
