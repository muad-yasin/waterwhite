"""The exposure-free restart of every run, the checkpoints that carry exposure and
the checkpoints that answer "not covered" (plan 4.3, Decision 23; docs/interfaces.md
section 7.1 "Restart points", docs/rebuild.md section 4).

The three functions read the coverage answer of each registered checkpoint through the
session they are handed and build none. A checkpoint qualifies as a restart when
its own answer is negative: its consumed history is completely traced, the proof of
plan 2.2 passes for it and it holds no exposure. The word is "proven" because a
restart is shown clean only from the traces the store holds.
"""

from __future__ import annotations

from .. import coverage
from .. import vocabulary as v
from . import replay_check


def _runs_with_checkpoints(store) -> list[tuple[str, list]]:
    """Each run that has registered checkpoints, with them in the order of its
    segments and then of their steps."""
    found = []
    for run in store.runs():
        checkpoints = []
        for segment in store.segments(run):
            checkpoints.extend(store.checkpoints_of(segment))
        if checkpoints:
            found.append((run, checkpoints))
    return found


def _name(checkpoint) -> str:
    return checkpoint.names[0] if checkpoint.names else checkpoint.instance_id


def _first_segment_of(store, run: str):
    names = store.segments(run)
    return store.segment(names[0]) if names else None


def _lineage_start(store, run: str) -> dict | None:
    """The first step of the first segment of the run the given run descends from
    (the run itself when it descends from none): the restart that is free of
    exposure by construction (plan 4.3)."""
    segment = _first_segment_of(store, run)
    if segment is None:
        return None
    seen = {segment.name}
    while segment.parent_checkpoint:
        parent = store.checkpoint(segment.parent_checkpoint)
        above = None if parent is None else store.segment(parent.segment)
        top = None if above is None else _first_segment_of(store, above.run)
        if top is None or top.name in seen:
            break  # a history that stops at a missing record, or loops, ends the lineage here
        segment = top
        seen.add(segment.name)
    return {
        "run": segment.run,
        "segment": segment.name,
        "step": replay_check.first_step(segment),
        "basis": v.RESTART_BASIS_LINEAGE_START,
    }


def unproven_entry(store, run: str) -> dict:
    """The entry of a run for which no restart is proven: no checkpoint is named, and
    ``from_scratch`` is the start of its lineage (D-68). The entry of a run with no
    registered checkpoint is this one."""
    return {
        "run": run, "proven": False, "checkpoint": None, "segment": None,
        "step": None, "basis": v.RESTART_NOT_PROVEN,
        "from_scratch": _lineage_start(store, run),
    }


def exposure_free_restarts(store, request, session) -> list[dict]:
    """The ``restarts`` entries of the report, one per run that has registered
    checkpoints, in the order the runs were registered."""
    entries = []
    for run, checkpoints in _runs_with_checkpoints(store):
        qualifying = None
        for checkpoint in checkpoints:
            answer = coverage.answer(store, _name(checkpoint), request, session=session)
            if answer.is_negative:
                qualifying = checkpoint
        if qualifying is not None:
            entries.append({
                "run": run, "proven": True, "checkpoint": _name(qualifying),
                "segment": qualifying.segment, "step": qualifying.step,
                "basis": v.RESTART_BASIS_LAST_CLEAR, "from_scratch": None,
            })
        else:
            entries.append(unproven_entry(store, run))
    return entries


def exposed_checkpoints(store, request, session) -> list[dict]:
    """The ``exposed_checkpoints`` entries of the report: every registered
    checkpoint whose answer is that the source is behind it or may be. No count of
    reads is given, since exposure counts belong to an extension that is off."""
    entries = []
    for run, checkpoints in _runs_with_checkpoints(store):
        for checkpoint in checkpoints:
            answer = coverage.answer(store, _name(checkpoint), request, session=session)
            if answer.answer not in (v.CONTAINS, v.MAY_CONTAIN):
                continue
            exposure = answer.first_exposure
            entries.append({
                "checkpoint": _name(checkpoint),
                "run": run,
                "segment": checkpoint.segment,
                "step": checkpoint.step,
                "answer": answer.answer,
                "first_exposure": None if exposure is None
                else {"segment": exposure[0], "step": exposure[1]},
                "level": answer.level,
                "basis": answer.basis,
                "untraced_steps": [[segment, step] for segment, step in answer.untraced_steps],
            })
    return entries


def checkpoints_not_covered(store, request, session) -> list[dict]:
    """The ``checkpoints_not_covered`` entries of the report: every registered
    checkpoint whose answer is "not covered", in the order of registration, whether or
    not a run reaches it. A checkpoint can be registered over a segment the store holds
    no row for (``Store.register_checkpoint`` names the segment by value), and such a
    checkpoint is in no run, so neither ``restarts`` nor ``exposed_checkpoints`` can
    name it. Each entry says what the answer says: ``answer`` and ``reason`` are the
    answer's words, never a check of the segment made here; ``run`` is the run the
    store gives the segment, null when the store has no such segment. With
    ``exposed_checkpoints`` every registered checkpoint is placed: a negative one in
    neither list, an exposed one in ``exposed_checkpoints``, a "not covered" one here."""
    entries = []
    for artifact in store.artifacts(kind=v.KIND_CHECKPOINT):
        answer = coverage.answer(store, artifact.instance_id, request, session=session)
        if answer.answer != v.NOT_COVERED:
            continue
        checkpoint = store.checkpoint(artifact.instance_id)
        if checkpoint is None:
            # An artifact of the kind with no checkpoint row (not made through the
            # store's calls): it is named, with no segment or step to give.
            entries.append({
                "checkpoint": artifact.names[0] if artifact.names else artifact.instance_id,
                "segment": None, "step": None, "run": None,
                "answer": answer.answer, "reason": answer.reason,
            })
            continue
        segment = store.segment(checkpoint.segment)
        entries.append({
            "checkpoint": _name(checkpoint),
            "segment": checkpoint.segment,
            "step": checkpoint.step,
            "run": None if segment is None else segment.run,
            "answer": answer.answer,
            "reason": answer.reason,
        })
    return entries
