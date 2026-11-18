"""The one place where the join meets the store's write method.

The join decides a segment's trace route before the segment exists in the store,
because ``Store.register_run_segment`` takes the route, the cause, the completed
steps, the step traces and the checkpoints in one call and one transaction (plan
3.5, Decision 19: the trace is written once, when the run is registered). A stop
leaves nothing of the segment written.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._join import JoinResult


@dataclass(frozen=True)
class SegmentRegistration:
    """What a segment's registration needs beside the join's result: the names,
    the place inside the run, and the two artifacts the segment names by value
    (its mixture snapshot and the checkpoint it started from, each by instance
    id, or None)."""

    run: str
    segment: str
    position: int
    start_kind: str
    mixture_instance_id: str | None = None
    parent_checkpoint: str | None = None


def register_segment(store, registration: SegmentRegistration, run_record: Mapping | None,
                     result: JoinResult) -> None:
    """Register the segment with the join's route, cause, steps, traces and the
    checkpoints of the step record, in one store call."""
    store.register_run_segment(
        run=registration.run,
        segment=registration.segment,
        position=registration.position,
        mixture_instance_id=registration.mixture_instance_id,
        parent_checkpoint=registration.parent_checkpoint,
        start_kind=registration.start_kind,
        run_record=None if run_record is None else dict(run_record),
        steps=result.steps,
        trace_route=result.route,
        trace_cause=result.cause,
        trace_cause_detail=result.cause_detail,
        traces=result.traces,
        checkpoints=result.checkpoints,
    )
