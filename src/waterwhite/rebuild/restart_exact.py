"""The exact-rerun restart (plan 4.3, F7; docs/interfaces.md section 7.3 "restarts",
docs/rebuild.md section 4).

For a run that has a stale segment, the first step at which the corrected data order
differs from the historical one, found by building the corrected order and comparing
the two. Where the corrected order cannot be built (the run record lacks a setting, a
shard of the corrected mixture is not registered or not readable, the join is not
verified for the run's release), the phase start is stated as an assumption and the
entry says that no step comparison is possible (F7).

Whether a run record holds every setting a replay needs is the check of
``replay_check``, which the report asks as well: a run that fails it is given the
phase start without the replay being tried, and the report states the same restart
for it (docs/interfaces.md 7.1).
"""

from __future__ import annotations

from collections.abc import Mapping

from .. import vocabulary as v
from ..errors import IntegrityFault, WaterwhiteError
from . import replay_check, verify_caches


def _reads(trace) -> tuple:
    """What a step read: for each sample, its place in the step and its rows."""
    return tuple(
        (sample.sample_pos, sample.rank,
         tuple((row.shard_position, row.ordinal, row.token_start, row.token_end) for row in sample.rows))
        for sample in trace.samples)


def _first_difference(store, segment, corrected_instance: str) -> int | None:
    """The first step of the segment whose reads differ between the registered
    traces and the traces of the corrected order, or None when none does. The steps
    are walked in order, and a step that either side holds no trace for ends the
    walk: nothing after it can be compared. Raises ``LookupError`` with the reason
    where the corrected order cannot be built or a step cannot be compared."""
    corrected = {trace.step: trace
                 for trace in verify_caches.corrected_traces(store, segment, corrected_instance).traces}
    for item in sorted(segment.steps, key=lambda registered: registered.step):
        held = store.step_trace(segment.name, item.step)
        found = corrected.get(item.step)
        if held is None or found is None:
            raise LookupError(f"step {item.step} has no trace on one of the two sides")
        if _reads(found) != _reads(held):
            return item.step
    return None


def _compared(store, stale: list[dict]) -> tuple[str, int | None]:
    """(the segment, the step) of the first difference over the stale segments of a
    run in their order; the first stale segment and no step where none differs."""
    for entry in stale:
        segment = store.segment(entry["segment"])
        corrected = store.resolve(entry["corrected_mixture"])
        if segment is None or corrected is None:
            raise LookupError("the segment or its corrected mixture is not registered")
        step = _first_difference(store, segment, corrected.instance_id)
        if step is not None:
            return segment.name, step
    return stale[0]["segment"], None


def exact_rerun_restarts(store, report: Mapping, caches: list[dict]) -> list[dict]:
    """The ``restarts`` entries of the verification, one per run that has a stale
    segment, in the order the runs first appear in ``caches``."""
    entries = []
    for run in dict.fromkeys(item["run"] for item in caches if item.get("stale")):
        order = {name: place for place, name in enumerate(store.segments(run))}
        stale = sorted((item for item in caches if item["run"] == run and item.get("stale")),
                       key=lambda item: order.get(item["segment"], len(order)))
        made = None
        held = [store.segment(item["segment"]) for item in stale]
        if replay_check.replayable(held) and all(item.get("rebuilt") for item in stale):
            try:
                made = _compared(store, stale)
            except IntegrityFault:
                raise
            except (LookupError, WaterwhiteError, ValueError, OSError):
                made = None  # the corrected order cannot be built: F7's fallback below
        if made is not None:
            segment, step = made
            entries.append({"run": run, "segment": segment, "step": step,
                            "basis": v.RESTART_BASIS_ORDER_DIFFERS,
                            "step_comparison": v.STEP_COMPARISON_MADE})
            continue
        segment, step = replay_check.phase_start(store, stale[0]["segment"])
        entries.append({"run": run, "segment": segment, "step": step,
                        "basis": v.RESTART_BASIS_PHASE_START,
                        "step_comparison": v.STEP_COMPARISON_NOT_POSSIBLE})
    return entries


def exact_rerun_of_report(store, artifacts: list[dict]) -> dict[str, dict]:
    """The ``exact_rerun`` object of the report's ``restarts`` entry, for every run that
    has a stale segment: a segment whose mixture the report says needs a successor.
    A run that has none is not in the result (its entry holds null).

    A replayable run (``replay_check.replayable`` over its stale segments, the check
    ``verify`` makes) gets no step: its restart is stated by ``verify`` once the
    successor shards exist, because the corrected data order is built from them. Any
    other run gets the phase start as an assumption, the same segment, step and basis
    ``exact_rerun_restarts`` states for it. So does a run with a segment on a mixture
    that reads the fourth word: ``verify`` names no corrected mixture for it, so the
    restart it could state later is not stated, and the phase start of the earliest
    segment over a mixture that may change is given instead."""
    rewritten = {entry["instance"] for entry in artifacts
                 if entry["kind"] == v.KIND_MIXTURE and entry["action"] == v.ACTION_REWRITE}
    # A mixture that reads the fourth word may or may not change: where the copies that
    # return land is not stated. A run over one is never told it needs no exact rerun.
    unstated = {entry["instance"] for entry in artifacts
                if entry["kind"] == v.KIND_MIXTURE
                and entry["action"] == v.ACTION_CLEAN_PLACE_NOT_STATED}
    changed = rewritten | unstated
    found: dict[str, dict] = {}
    for run in store.runs():
        stale = []
        for name in store.segments(run):
            held = store.segment(name)
            if held is not None and held.mixture_instance_id in changed:
                stale.append(held)
        if not stale:
            continue
        replayable = replay_check.replayable(stale)
        if replayable and not any(held.mixture_instance_id in unstated for held in stale):
            found[run] = {"run_record_replayable": True, "segment": stale[0].name, "step": None,
                          "basis": v.RESTART_EXACT_BY_VERIFY, "step_comparison": None}
        else:
            segment, step = replay_check.phase_start(store, stale[0].name)
            found[run] = {"run_record_replayable": replayable, "segment": segment, "step": step,
                          "basis": v.RESTART_BASIS_PHASE_START,
                          "step_comparison": v.STEP_COMPARISON_NOT_POSSIBLE}
    return found


__all__ = ["exact_rerun_of_report", "exact_rerun_restarts"]
