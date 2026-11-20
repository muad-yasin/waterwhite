"""The coverage proof (plan 2.2): where "does not contain" is decided.

``prove(store, artifact, request)`` walks the artifact's ancestor closure on demand
from the evidence store and returns a ``Proof`` when every condition of plan 2.2
holds and no record in scope was found, or a ``Finding``: the recorded chain that
shows "contains", the first failing link ("may contain"), or the missing record of
the artifact itself ("not covered").

This module is the only maker of proofs. It claims the proof mint of
``waterwhite.answers`` when it is imported; the mint is kept inside ``prove`` and
is reachable from no other place, so the answer "does not contain" can only come
out of the walk below (plan W6, Decision 1).

The conditions and where each is applied:

- (a), (b), (f) for edges and map rows, (g): ``_walk.Session``, per record;
- (c), (d), the count half of (e), the artifact digests: the store's loud checks,
  run here before the walk; a fault raises and no answer exists (plan 3.4). A pass
  that says the checks ran is made by ``check_store`` only and holds until the store
  is written;
- (e) and plan 2.4: ``_checkpoint.checkpoint_outcome``;
- (h), the generating-model rule: ``_walk.Session._model_link``, per record, where the walk
  meets a generation edge whose bound is established, and only while the generating-model
  extension is on (the switch is read once per query, ``Session.generating_model_on``). While
  it is off the model is a logged attribute and the negative says what was not assessed
  (``_conclude``). A temporal ancestry cycle the walk met is raised here, in ``prove``, where
  a verdict leaves the session, unless a recorded "contains" wins.

docs/coverage.md states the rules as built.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from collections.abc import Callable, Iterable

from .. import extensions
from .. import vocabulary as v
from ..answers import Answer, Finding, Proof, Scope, claim_proof_mint
from ..errors import Refused, ShardFilesNotChecked, TemporalAncestryCycle, UnsupportedSelector
from ..requests import Withdrawal
from ..store.views import FILE_ABSENT, Artifact
from ._walk import CLEAR, HOLDS, MAY, NOT_COVERED, Outcome, Session

__all__ = ["IntegrityPass", "Session", "Withheld", "answer", "answer_or_withheld", "check_store", "is_clear",
           "model_not_covered", "prove", "shard_withheld", "untraced_steps", "withhold_named"]


def is_clear(verdict) -> bool:
    """Whether a verdict the walk gave (a record's, from ``Session.occurrence``, or
    an artifact's) is the one that every chain ended outside the request. A read
    only: it makes no verdict and no answer. A module that reports on a verdict
    asks here, and names neither the verdict nor the negative answer itself."""
    return verdict.verdict == CLEAR


def untraced_steps(store, segment: str) -> tuple[tuple[int, str], ...]:
    """The untraced steps of a registered segment as the proof reads them (plan
    2.2(e)), each with its reason; ``show --trace`` lists them."""
    from ._checkpoint import untraced_steps as read

    return read(store, segment)


class IntegrityPass:
    """Says that the store's loud checks ran, and over what: the whole store, or
    the ancestor closures of some artifacts. ``prove`` runs the checks itself for
    any artifact a pass does not cover.

    Only ``check_store`` makes a pass that covers anything: a pass made by calling
    this class says nothing about the store, and a pass stops covering once the
    store is written after it (the mark ``Store.state_mark`` gives)."""

    def __init__(self, store, instance_ids: tuple[str, ...] | None) -> None:
        self._store = store
        self._whole = instance_ids is None
        self._ids = frozenset(instance_ids or ())

    def covers(self, store, instance_id: str) -> bool:
        return (
            store is self._store
            and (self._whole or instance_id in self._ids)
            and _pass_holds(self, store)
        )


def _pass_register() -> tuple[Callable, Callable]:
    """The register of the passes ``check_store`` made, each with the store's mark
    from before its checks ran. Closed over, as the proof mint is, so that no module
    attribute leads to it."""
    made: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def issue(held: IntegrityPass, mark: tuple[int, int]) -> IntegrityPass:
        made[held] = mark
        return held

    def holds(held: IntegrityPass, store) -> bool:
        mark = made.get(held)
        return mark is not None and store.state_mark() == mark

    return issue, holds


_issue_pass, _pass_holds = _pass_register()


def check_store(store, instance_ids: Iterable[str] | None = None) -> IntegrityPass:
    """Run the store's loud checks (plan 3.4) over the ancestor closure of the given
    artifacts, or over the whole store when None. A fault raises its named error. The
    pass holds until the store is next written."""
    settings = store.settings()
    extensions.check_switches(settings)  # a switch that cannot be read stops every answer
    listed = None if instance_ids is None else tuple(instance_ids)
    mark = store.state_mark()  # before the checks: a write during them leaves the pass stale
    if _models_in_closure(settings):
        store.check_integrity(listed, models=True)
    else:
        store.check_integrity(listed)
    return _issue_pass(IntegrityPass(store, listed), mark)


def _models_in_closure(settings: dict[str, str]) -> bool:
    """Whether the loud checks of a query hold the ancestry of the generating models in the
    closure of the artifact asked about (R-10): only with the generating-model switch on."""
    return settings.get(extensions.setting_key(extensions.GENERATING_MODEL)) == extensions.ON


def _model_off_finding(artifact: Artifact, request: Withdrawal) -> Finding:
    """The one finding for a published model the store holds while the published-models
    extension is off (plan 1.4, 3.1, Section 7): "not covered", reason "not registered",
    and the note that the extension is off. A query and a listing both take it from here."""
    return Finding(answer=v.NOT_COVERED, instance_id=artifact.instance_id, kind=v.KIND_PUBLISHED_MODEL,
                   request=request, reason=v.REASON_NOT_REGISTERED, notes=(v.NOTE_PUBLISHED_MODELS_OFF,))


def model_not_covered(artifact: Artifact, name: str, request: Withdrawal) -> Answer:
    """The listing's answer for a published model the store holds while the
    published-models extension is off: the finding of ``_model_off_finding``. Only
    this package makes an answer."""
    return Answer.from_finding(_model_off_finding(artifact, request), artifact=name)


def _unregistered(store, ref: str, request: Withdrawal) -> Finding:
    """An artifact the store has no row for (plan 2.1): "record missing" when
    another record names it, "not registered" when none does."""
    named = [r for r in store.references_to(ref) if not r.registered]
    return Finding(
        answer=v.NOT_COVERED,
        instance_id=ref if named else None,
        kind=None,
        request=request,
        reason=v.REASON_RECORD_MISSING if named else v.REASON_NOT_REGISTERED,
    )


def _conclude(mint, walk: Session, artifact: Artifact, outcome: Outcome) -> Proof | Finding:
    request = walk.request
    common = dict(instance_id=artifact.instance_id, kind=artifact.kind, request=request)
    if outcome.verdict == NOT_COVERED:
        return Finding(answer=v.NOT_COVERED, reason=outcome.reason, **common)
    if outcome.verdict == HOLDS:
        # A chain that passes a generation edge is process evidence (Decision 25).
        notes = (v.NOTE_PROCESS_EVIDENCE,) if outcome.generation else ()
        return Finding(
            answer=v.CONTAINS, chain=outcome.chain, basis=outcome.basis, notes=notes,
            untraced_steps=outcome.untraced_steps, first_exposure=outcome.first_exposure,
            level=outcome.level, **common,
        )
    if outcome.verdict == MAY:
        failing = outcome.failing
        assert failing is not None
        return Finding(
            answer=v.MAY_CONTAIN, failing=failing,
            candidates=failing.candidates or walk.candidate_sources(artifact),
            untraced_steps=outcome.untraced_steps, **common,
        )
    assert outcome.verdict == CLEAR

    # A negative that rests on the map of a shard whose files are not here needs the check of
    # those files: with them gone, the offsets in their index cannot be looked at now.
    missing = _missing_files(walk, artifact)
    _require_shard_files_checked(walk.store, missing)

    # Conditions (a) to (g) hold and no record in scope was found.
    notes: tuple[str, ...] = ()
    if outcome.generation and not walk.generating_model_on:
        # Condition (h) is applied in ``_walk.Session`` while the generating-model extension is
        # on, per record. While it is off the model is a logged attribute and the negative says
        # what was not assessed (plan 2.1, 2.2(h)).
        notes = (v.NOTE_GENERATING_MODEL_NOT_ASSESSED,)
    scope = Scope(
        source=request.source,
        members=request.members,
        records=request.record_names,
        runs=outcome.runs,
        steps=outcome.steps,
        generation_edges=outcome.jobs,
        harness_logs=tuple(walk.logs[job] for job in outcome.jobs),
        walked=outcome.walked,
        files_not_checked=tuple(walk.name(lacking.instance_id) for lacking in missing),
    )
    return mint(
        instance_id=artifact.instance_id, kind=artifact.kind, names=artifact.names, request=request,
        scope=scope, notes=notes, passed_generation=outcome.generation, current=walk.holds_for_proof(),
    )


def _fault(instance: str, path: str, state: str | None) -> ShardFilesNotChecked:
    where = f"the files of shard {instance} are not at {path}"
    if state == "failed":
        detail = (f"{where}, and when they were last checked they were not a shard of this "
                  "layout (the offsets or the size of its index): the negative is not given")
    else:
        detail = (f"{where}, and no check of them has passed (the store registered the shard "
                  "before its offsets and size were held to the token reader): the negative is "
                  "not given until the files are put back and `waterwhite check` has read them")
    return ShardFilesNotChecked(detail, instance=instance, path=path, state=state)


def _require_shard_files_checked(store, missing) -> None:
    """The negative is not given while a shard of the closure is without its files and no
    check of them has passed (``ShardFilesNotChecked``): none was made, or it failed."""
    for lacking in missing:
        if lacking.kind != v.KIND_SHARD:
            continue
        state = store.shard_files_state(lacking.instance_id)
        if state == "passed":
            continue
        raise _fault(lacking.instance_id, lacking.path, state)


def _missing_files(walk: Session, artifact: Artifact):
    """The files of the artifact's closure that are not at their recorded place. The closure holds
    the ancestry of the generating models only with the switch on (R-10)."""
    if walk.generating_model_on:
        return walk.store.missing_files([artifact.instance_id], models=True)
    return walk.store.missing_files([artifact.instance_id])


def _make_prove():
    mint = claim_proof_mint()

    def prove(
        store,
        artifact: Artifact | str,
        request: Withdrawal,
        *,
        integrity: IntegrityPass | None = None,
        session: Session | None = None,
    ) -> Proof | Finding:
        """Prove that an artifact holds nothing the request puts in scope, or say
        why not (plan 2.2).

        ``artifact`` is a name, an instance id, a content digest or a path, as a
        query names it, or an ``Artifact``. ``request`` is a withdrawal expanded by
        ``waterwhite.requests``. The store's loud checks run first over the
        artifact's ancestor closure unless ``integrity`` says they already did.
        ``session`` lets several calls for one request share what they walked."""
        if not isinstance(request, Withdrawal):
            raise UnsupportedSelector(
                "the proof takes a withdrawal expanded by waterwhite.requests; every other "
                "request is refused there"
            )
        extensions.check_switches(store.settings())
        if isinstance(artifact, Artifact):
            # What the store holds for the instance id decides, not the object handed in.
            found = store.artifact(artifact.instance_id)
            if found is None:
                return _unregistered(store, artifact.instance_id, request)
        else:
            found = store.resolve(str(artifact))
            if found is None:
                return _unregistered(store, str(artifact), request)
        if found.kind == v.KIND_PUBLISHED_MODEL:
            if not extensions.is_on(store.settings(), extensions.PUBLISHED_MODELS):
                # Off: the model is not registered (plan 1.4, 3.1, Section 7), as the listing says.
                if integrity is None or not integrity.covers(store, found.instance_id):
                    store.check_integrity([found.instance_id])
                return _model_off_finding(found, request)
            raise Refused(
                "the published-models extension is switched on in this store, and this build "
                "of the tool holds only its off state",
                extension=extensions.PUBLISHED_MODELS,
            )
        # The session (and its mark of the store) comes first, so that a write after
        # the mark is found whether it falls before the checks, during the walk or after.
        walk = session if session is not None else Session(store, request)
        if walk.store is not store or walk.request != request:
            raise ValueError("the session belongs to another store or another request")
        walk.ensure_current()
        if integrity is None or not integrity.covers(store, found.instance_id):
            if walk.generating_model_on:
                store.check_integrity([found.instance_id], models=True)
            else:
                store.check_integrity([found.instance_id])
        outcome = walk.outcome(found)
        walk.ensure_current()
        if outcome.cycle is not None:
            # A temporal ancestry cycle (R-11): the answer would rest on it and no recorded
            # "contains" wins, so the request is refused and no answer is printed. Raised here, at
            # the outermost exit of the artifact's answer, and never inside a nested ``outcome``.
            raise TemporalAncestryCycle(
                f"the history of checkpoint {outcome.cycle.model} reads "
                f"{outcome.cycle.record}, which that checkpoint's own output feeds.",
                checkpoint=outcome.cycle.model, record=outcome.cycle.record,
            )
        return _conclude(mint, walk, found, outcome)

    return prove


prove = _make_prove()


def answer(
    store,
    artifact: str,
    request: Withdrawal,
    *,
    integrity: IntegrityPass | None = None,
    session: Session | None = None,
) -> Answer:
    """The answer about one artifact, named as a query names it (plan 2.1)."""
    extensions.check_switches(store.settings())  # before an unregistered name is answered "not covered"
    found = store.resolve(artifact)
    if found is None:
        return Answer.from_finding(_unregistered(store, artifact, request), artifact=artifact)
    shown = artifact if artifact in found.names else (found.names[0] if found.names else found.instance_id)
    result = prove(store, found, request, integrity=integrity, session=session)
    return Answer.from_result(result, artifact=shown)


@dataclass(frozen=True)
class Withheld:
    """An answer the tool does not give because it rests on a shard whose files are away and
    which no check has passed (``ShardFilesNotChecked``): the artifact it was asked of, the
    shard, and why. A listing, a report and a verification print these beside the answers they
    do give (docs/cli.md, "check")."""

    artifact: str
    instance: str
    kind: str
    shard: str
    state: str | None
    reason: str
    error: str = ShardFilesNotChecked.name

    def to_json(self) -> dict:
        return {"artifact": self.artifact, "instance": self.instance, "kind": self.kind,
                "error": self.error, "shard": self.shard, "state": self.state, "reason": self.reason}


def _put(sink: list, item: Withheld) -> None:
    """One entry for an artifact, however many answers of the command ask for it."""
    if not any(held.instance == item.instance and held.artifact == item.artifact for held in sink):
        sink.append(item)


def _withheld(store, ref, error: ShardFilesNotChecked) -> Withheld:
    found = store.resolve(ref) if isinstance(ref, str) else ref
    shown = ref if found is None else (found.names[0] if found.names else found.instance_id)
    return Withheld(
        artifact=shown if isinstance(shown, str) else str(shown),
        instance="" if found is None else found.instance_id,
        kind="" if found is None else found.kind,
        shard=error.context.get("instance", ""), state=error.context.get("state"), reason=error.detail)


def answer_or_withheld(store, artifact: str, request: Withdrawal, *, sink: list | None,
                       integrity: IntegrityPass | None = None, session: Session | None = None):
    """``answer``, except that an answer that rests on an archived shard no check has passed
    is put in ``sink`` as a ``Withheld`` and ``None`` is returned: the caller goes on with the
    answers it can give and says which it cannot. With no ``sink`` the fault is raised, as
    ``answer`` raises it (a query that names the artifact)."""
    if sink is None:
        return answer(store, artifact, request, integrity=integrity, session=session)
    try:
        return answer(store, artifact, request, integrity=integrity, session=session)
    except ShardFilesNotChecked as error:
        _put(sink, _withheld(store, artifact, error))
        return None


def withhold_named(store, ref, shard: Artifact | None, sink: list) -> None:
    """Put the artifact named ``ref`` in ``sink``, withheld for ``shard`` (the shard whose
    files are away) when the caller found it before asking the question."""
    if shard is None:
        return  # the question itself raised, and answer_or_withheld has put it in the sink
    path = store.disk_path(shard.instance_id)
    found = store.resolve(ref)
    _put(sink, Withheld(
        artifact=ref, instance="" if found is None else found.instance_id,
        kind="" if found is None else found.kind, shard=shard.instance_id,
        state=store.shard_files_state(shard.instance_id),
        reason=_fault(shard.instance_id, str(path), store.shard_files_state(shard.instance_id)).detail))


def shard_withheld(store, shard: Artifact, sink: list | None) -> bool:
    """Whether the shard's own files are away and no check has passed. Its documents are then
    not judged through its map (a "keep" or a "remove" of a document rests on offsets nothing can
    vouch for while the files are away); with a ``sink`` the shard is put in it."""
    if shard.kind != v.KIND_SHARD or store.file_state(shard.instance_id) != FILE_ABSENT:
        return False
    state = store.shard_files_state(shard.instance_id)
    if state == "passed":
        return False
    if sink is not None:
        path = store.disk_path(shard.instance_id)
        _put(sink, _withheld(store, shard, _fault(shard.instance_id, str(path), state)))
    return True
