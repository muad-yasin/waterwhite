"""What the report and the verification both ask of a run's records for its exact-rerun
restart (plan 4.3; docs/interfaces.md 7.1 and 7.3, docs/rebuild.md section 4).

- ``replayable``: whether the run records of some segments hold every setting a replay
  needs. It is the one check, made by ``join.missing_replay_settings``, which
  ``join.replay_arrays`` refuses by as well; the report and ``verify`` both call it, so
  they cannot disagree about which run is replayable. It looks at the records and at
  nothing else: it does not try the replay.
- ``phase_start``: the restart the tool gives for a run that is not replayable: the first
  step of the earliest stale segment, as an assumption.
"""

from __future__ import annotations

from collections.abc import Iterable

from .. import join
from ..errors import WaterwhiteError


def first_step(segment) -> int | None:
    """The first step a segment took, from its launches, else from its steps."""
    firsts = [launch.first_step for launch in segment.launches]
    firsts += [item.step for item in segment.steps]
    return min(firsts) if firsts else None


def record_replayable(run_record) -> bool:
    """Whether one run record holds every setting a replay needs. A segment with no
    record, and a record the join cannot read, are not replayable."""
    if run_record is None:
        return False
    try:
        parsed = join.parse_run_record(run_record)
    except WaterwhiteError:
        return False
    return not join.missing_replay_settings(parsed)


def replayable(segments: Iterable) -> bool:
    """Whether a run is replayable, given its stale segments (``store.segment`` values):
    every one of them has a run record that holds every setting a replay needs."""
    held = list(segments)
    return bool(held) and all(
        segment is not None and record_replayable(segment.run_record) for segment in held)


def phase_start(store, first_stale_segment: str) -> tuple[str, int | None]:
    """(the segment, its first step): the restart of a run that cannot be replayed."""
    segment = store.segment(first_stale_segment)
    return first_stale_segment, None if segment is None else first_step(segment)


__all__ = ["first_step", "phase_start", "record_replayable", "replayable"]
