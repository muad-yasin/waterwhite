"""A checkpoint's answer (plan 2.4, Decisions 12, 18, 20, 22).

A checkpoint is answered from what its history consumed, not from what its mixture
held. Its history is the completed steps of its segment up to its step and, through
the checkpoint its segment started from, the same of every ancestor.

- "contains": a trace row of a completed step, inside the covered part of a sample,
  reads a record in scope, on a segment whose Megatron release, sampler type and
  sampler class the store lists as verified. An exposure in an ancestor's steps is
  found the same way and is inherited. The answer states its basis, names the first
  exposure and lists the untraced steps of the history beside the chain. It holds
  whatever else in the history is untraced.
- "does not contain": every step of the history is registered and completely
  traced, every segment is verified and has a trace route, every mixture snapshot
  is there, the proof holds for every document the steps read, and none is in
  scope.
- "may contain" otherwise, naming the first failing link (plan 2.2).

Which link is first. The history is read oldest step first, and the first failing
link is the one at the earliest step: a step that is untraced, or a document a step
read whose chain fails a condition. At one step the causes come in the order plan
2.4 lists them: the untraced step, then the consumed record. A cause that belongs
to a whole segment stands where the segment stands: a segment with no trace route
at its first step; an unverified version or sampler after the segment's own steps;
a missing record of an ancestor before everything, since what came before it is
unknown. ``_Order`` holds this rule and nothing else does.

A segment whose trace route is "none" has no usable trace: each of its steps is
untraced, and the cause is the one the run's registration stored for the segment
("data-order trace missing" with its reason, "cyclic order not recorded", "cyclic
order digest mismatch"), plan 3.5. A segment with no trace route answers so whether
or not its run record holds a launch: the store refuses a route for a record with no
launch and a trace under a route of "none", so no store the tool writes holds a trace
the proof could take for evidence. Two cases remain for a store edited by hand, a
stored trace route that its run record cannot support: a cyclic segment whose run
record holds no shuffle digest names "cyclic order not recorded", and a segment whose
run record holds no launch names "data-order trace missing" with the reason "run
record incomplete", as the join would have stored it (``_no_usable_trace``).
"""

from __future__ import annotations

from .. import vocabulary as v
from ..answers import Failing, Hop
from ..store.views import REL_CHECKPOINT_SEGMENT, ROUTE_NONE, Artifact, HistorySegment
from ._walk import CLEAR, HOLDS, MAY, NOT_COVERED, Outcome, Session

_NAMED_CAUSES = frozenset(v.NAMED_CAUSES)

# The order of plan 2.4's causes, used between causes that stand at the same step.
_UNTRACED = 1  # a step of the history is untraced
_RECORD = 2  # a record its steps consumed fails a condition of the proof
_VERSION = 3  # the Megatron version is unverified
_SAMPLER = 4  # the sampler type is unverified
_ANCESTOR = 6  # an ancestor answers "may contain"
_AFTER_STEPS = float("inf")


class _Order:
    """Keeps the first failing link of a history: the smallest (segment position,
    step, cause) seen. A later link with the same place does not replace it."""

    def __init__(self) -> None:
        self.key: tuple[int, float, int] | None = None
        self.failing: Failing | None = None

    def see(self, segment_position: int, step: float, cause: int, failing: Failing | None) -> None:
        """Take in a link. None is "no link here" and is passed by a caller that
        holds an optional link (a shard's label gate); a verdict that is "may
        contain" always carries its link (``_walk.Status`` refuses one without)."""
        if failing is None:
            return
        key = (segment_position, step, cause)
        if self.key is None or key < self.key:
            self.key, self.failing = key, failing


def _steps_not_registered(part: HistorySegment, first_launched: int | None) -> list[tuple[int, str]]:
    """Step numbers that lie in the history and are not among the segment's
    registered steps: steps between the first step its launches record and the
    first registered one, a gap between two registered steps, or the checkpoint's
    own step. Nothing is known about what such a step read."""
    missing: list[tuple[int, str]] = []
    if first_launched is not None and part.steps:
        for step in range(first_launched, min(part.steps[0], part.upto_step + 1)):
            missing.append((step, "the segment's launches start at step "
                                  f"{first_launched} and the step is not registered"))
    for before, after in zip(part.steps, part.steps[1:]):
        for step in range(before + 1, after):
            missing.append((step, "the step is not among the segment's registered steps"))
    if not part.upto_registered:
        missing.append((
            part.upto_step,
            "the checkpoint's step is not among the segment's registered steps"
            + (f" (they end at {part.steps[-1]})" if part.steps else ""),
        ))
    return missing


class _Traces:
    """Reads one step trace at a time, to place a row inside its sample."""

    def __init__(self, store) -> None:
        self._store = store
        self._key: tuple[str, int] | None = None
        self._trace = None

    def get(self, segment: str, step: int):
        if self._key != (segment, step):
            self._trace = self._store.step_trace(segment, step)
            self._key = (segment, step)
        return self._trace

    def row_offset(self, segment: str, hit) -> int | None:
        """How many of the sample's tokens lie before this row."""
        trace = self.get(segment, hit.step)
        if trace is None:
            return None
        for sample in trace.samples:
            if sample.sample_pos == hit.sample_pos:
                return sum(r.token_end - r.token_start for r in sample.rows[: hit.row_pos])
        return None


def _no_usable_trace(segment) -> str | None:
    """Why a segment's traces decide nothing, or None.

    "route": its route is "none", whatever else its run record holds, and it names the
    cause and reason stored for it: a segment registered with no trace route answers
    so, with or without a launch in its run record (plan 3.5; DECISIONS.md D-25). The
    store refuses a trace under such a segment, so none is read.
    "order": it has a stored trace route and was read with the cyclic sampler, and its
    run record holds no shuffle digest (the order of its samples is unknown; plan 2.4,
    3.5; D-26). The join asks this before the launch, so this case does too.
    "launch": it has a stored trace route and its run record holds no launch, so
    nothing says which step comes first or where its counter starts and no step of it
    is known to be whole (plan 2.2(e), 3.1, 3.5; D-25). The store refuses to write
    either of the last two cases; a store edited by hand can hold them, and they are
    read as the join would have stored them."""
    if segment.trace_route == ROUTE_NONE:
        return "route"
    if segment.sampler_type == v.SAMPLER_CYCLIC and not segment.shuffle_digests:
        return "order"
    if not segment.launches:
        return "launch"
    return None


def _open_steps(store, traces: "_Traces", segment, part: HistorySegment, upto: int
                ) -> tuple[list[tuple[int, str]], dict[int, tuple[int, int]]]:
    """Condition 2.2(e) for the part of a segment up to ``upto``: every step that is
    not registered, has no trace, has a sample with an uncovered range, or is lowered
    by the operator's note, each with its reason in words, in step order; and the
    operator's notes, by step, as (tokens read of the last sample, its position)."""
    notes: dict[int, tuple[int, int]] = {}
    first_launched = min((launch.first_step for launch in segment.launches), default=None)
    open_steps = _steps_not_registered(part, first_launched)
    for state in store.trace_states(part.segment, upto_step=upto):
        why = None
        if not state.traced:
            why = "the step has no trace"
        elif state.uncovered_samples:
            why = f"{state.uncovered_samples} of its samples have an uncovered token range"
        elif state.unbacked_samples:
            why = (f"{state.unbacked_samples} of its samples are marked as covered beyond the rows "
                   "the trace holds for them")
        if state.traced and state.last_sample_tokens_read is not None:
            notes[state.step] = (state.last_sample_tokens_read, state.sample_rows - 1)
            trace = traces.get(part.segment, state.step)
            samples = () if trace is None else trace.samples
            if why is None and samples and state.last_sample_tokens_read < samples[-1].sample_tokens:
                why = ("the operator's step record gives "
                       f"{state.last_sample_tokens_read} of {samples[-1].sample_tokens} tokens "
                       "of its last sample as read")
        if why is not None:
            open_steps.append((state.step, why))
    open_steps.sort()
    return open_steps, notes


def untraced_steps(store, name: str) -> tuple[tuple[int, str], ...]:
    """The untraced steps of a whole registered segment, as condition 2.2(e) reads
    them for a checkpoint at its last step, each with its reason: every registered
    step when the segment's traces decide nothing (``_no_usable_trace``), else the
    steps with no trace or a trace with an uncovered range, and the steps the
    segment's launches and registered steps leave unaccounted for. Empty for a
    segment that is not registered."""
    segment = store.segment(name)
    if segment is None:
        return ()
    registered = tuple(step.step for step in segment.steps)
    reason = _no_usable_trace(segment)
    if reason is not None:
        why = {"route": "the segment was registered with no data-order trace",
               "order": "the segment was read with the cyclic sampler and its run record holds "
                        "no shuffle digest",
               "launch": "the segment's run record holds no launch"}[reason]
        return tuple((step, why) for step in registered)
    part = HistorySegment(name, "", registered[-1] if registered else 0, registered, True)
    return tuple(_open_steps(store, _Traces(store), segment, part, part.upto_step)[0])


NO_ROWS = "no rows"  # the segment's rows hold no read of a document its traces list


def _first_covered_read(store, traces: _Traces, segment: str, document, upto: int,
                        notes: dict[int, tuple[int, int]]) -> tuple[int, str] | str | None:
    """The first step whose trace reads the document inside the covered part of a
    sample, with the basis of that read (plan 2.4). A read of one token counts
    (Decision 18). A row that lies wholly in an uncovered range is no evidence.
    ``NO_ROWS`` when the document is listed as read (``trace_documents`` goes through the
    trace of each step) and ``trace_hits`` finds no row of it in the segment: a row that
    names another segment than its trace's, or none, which is no read in an uncovered range."""
    seen = False
    for hit in store.trace_hits(segment, document.shard_position, document.ordinal, upto_step=upto):
        seen = True
        if hit.token_end <= hit.token_start:
            continue  # a row that reads no token
        covered = hit.covered_tokens
        basis = v.BASIS_WHOLE_SAMPLE
        note = notes.get(hit.step)
        if note is not None and hit.sample_pos == note[1]:
            # The operator's step record covers this sample and can only lower it.
            covered = min(covered, note[0])
            basis = v.BASIS_OPERATOR_NOTE
        if covered >= hit.sample_tokens:
            return hit.step, basis
        offset = traces.row_offset(segment, hit)
        if offset is not None and offset < covered:
            return hit.step, basis
    return None if seen else NO_ROWS


def checkpoint_outcome(walk: Session, artifact: Artifact) -> Outcome:
    store = walk.store
    mark = store.checkpoint(artifact.instance_id)
    if mark is None or store.segment(mark.segment) is None:
        # Its own record, the run segment, is missing (plan 2.4).
        return Outcome(NOT_COVERED, reason=v.REASON_RECORD_MISSING)
    history = store.checkpoint_history(artifact.instance_id)
    assert history is not None
    verified = set(store.verified_joins())
    verified_tags = {join[0] for join in verified}
    traces = _Traces(store)

    first = _Order()
    untraced: list[tuple[str, int]] = []
    exposure: tuple[int, int, str, str, tuple[Hop, ...]] | None = None
    holds_generation = False
    clear_generation = False
    jobs: dict[str, None] = {}
    walked = 0
    runs: list[str] = []
    steps_scope: list[tuple[str, int, int]] = []

    if not history.complete:
        gone = history.missing
        assert gone is not None
        text = (f"the history stops at a missing record: {gone.relation} {gone.referred}, named "
                f"by {walk.name(gone.referrer)}; what was read before it is unknown")
        if gone.relation == REL_CHECKPOINT_SEGMENT:
            # An ancestor checkpoint whose own record, its run segment, is missing.
            missing = Failing(
                names=v.NAMES_UPSTREAM_NOT_COVERED, artifact=walk.name(gone.referrer),
                instance=gone.referrer, segment=gone.referred, reason=v.REASON_RECORD_MISSING,
                text=text,
            )
        else:
            # The checkpoint a segment started from has no row.
            missing = Failing(
                names=v.NAMES_UPSTREAM_NOT_COVERED, artifact=gone.referred, instance=gone.referred,
                segment=gone.referrer, reason=v.REASON_RECORD_MISSING, text=text,
            )
        first.see(-1, 0, _ANCESTOR, missing)

    last = len(history.segments) - 1
    for index, part in enumerate(history.segments):
        segment = store.segment(part.segment)
        assert segment is not None
        if segment.run not in runs:
            runs.append(segment.run)
        if part.steps:
            steps_scope.append((part.segment, part.steps[0], part.steps[-1]))
        upto = part.upto_step
        opens_at = part.steps[0] if part.steps else upto

        # A segment whose traces decide nothing: each of its steps is untraced and the
        # cause is its own (plan 3.5).
        no_trace = _no_usable_trace(segment)
        if no_trace is not None:
            listed = list(part.steps) or [upto]
            untraced.extend((part.segment, step) for step in listed)
            if no_trace in ("order", "launch"):
                if no_trace == "order":
                    cause, detail = v.NAMES_CYCLIC_NOT_RECORDED, None
                    text = (f"segment {part.segment} was read with the cyclic sampler and its run "
                            "record holds no shuffle digest")
                else:
                    cause, detail = v.NAMES_TRACE_MISSING, v.TRACE_MISSING_RECORD_INCOMPLETE
                    text = (f"step {listed[0]} of {part.segment}: the segment's run record holds no "
                            "launch, so no step of it is known to be whole")
                first.see(index, opens_at, _UNTRACED, Failing(
                    names=cause, condition=v.CONDITION_E, segment=part.segment, step=listed[0],
                    detail=detail, text=text,
                ))
                continue
            if segment.trace_cause in _NAMED_CAUSES:
                cause, detail = segment.trace_cause, segment.trace_cause_detail
                text = (f"segment {part.segment} was registered with no data-order trace"
                        + (f": {detail}" if detail else ""))
            else:
                # The store writes no such segment: the stored cause was edited. The
                # segment has no trace route, which is all that is known, and the
                # answer says so in words; the stored reason is not passed on.
                cause, detail = v.NAMES_TRACE_MISSING, None
                text = (f"segment {part.segment} has no data-order trace, and the cause stored "
                        f"for it, {segment.trace_cause!r}, is not a named cause")
            first.see(index, opens_at, _UNTRACED, Failing(
                names=cause, condition=v.CONDITION_E, segment=part.segment, step=listed[0],
                detail=detail, text=text,
            ))
            continue

        # Condition 2.2(e): every step registered, traced, and whole.
        open_steps, notes = _open_steps(store, traces, segment, part, upto)
        for step, why in open_steps:
            untraced.append((part.segment, step))
            first.see(index, step, _UNTRACED, Failing(
                names=v.NAMES_STEP_UNTRACED, condition=v.CONDITION_E, segment=part.segment,
                step=step, text=f"step {step} of {part.segment}: {why}",
            ))

        # F6: the join is verified per release, sampler type and sampler class.
        join = (segment.release_tag, segment.sampler_type, segment.sampler_class)
        if join not in verified:
            tag_known = segment.release_tag in verified_tags
            names = v.NAMES_TRACE_UNVERIFIED_SAMPLER if tag_known else v.NAMES_TRACE_UNVERIFIED_VERSION
            what = (f"segment {part.segment} ran on release {segment.release_tag!r} with sampler "
                    f"{segment.sampler_type!r} ({segment.sampler_class!r}), which the store does "
                    "not list as verified")
            if index == last:
                first.see(index, _AFTER_STEPS, _SAMPLER if tag_known else _VERSION,
                          Failing(names=names, segment=part.segment, text=what))
            else:
                first.see(index, _AFTER_STEPS, _ANCESTOR, Failing(
                    names=v.NAMES_ANCESTOR_MAY_CONTAIN, artifact=walk.name(part.checkpoint),
                    instance=part.checkpoint, segment=part.segment,
                    text=f"the ancestor checkpoint {walk.name(part.checkpoint)} answers "
                         f"\"may contain\" ({names}): {what}",
                ))
            continue  # its trace rows decide nothing

        # The mixture snapshot ties a trace row's shard position to a shard (L6).
        mixture = segment.mixture_instance_id
        if mixture is None or store.artifact(mixture) is None:
            first.see(index, opens_at, _RECORD, Failing(
                names=v.NAMES_SNAPSHOT_MISSING, instance=mixture, segment=part.segment,
                reason=v.REASON_NOT_REGISTERED if mixture is None else v.REASON_RECORD_MISSING,
                text=f"segment {part.segment} has no mixture snapshot in the store, so the "
                     "shards its steps read cannot be named",
            ))
            continue
        snapshot = {entry.position: entry for entry in store.mixture_shards(mixture)}

        # Every document a traced step of this part of the history read.
        for document in store.trace_documents(part.segment, upto_step=upto):
            walked += 1
            at = document.first_step
            entry = snapshot.get(document.shard_position)
            if entry is None:
                first.see(index, at, _RECORD, Failing(
                    names=v.NAMES_MISSING_ORIGIN, condition=v.CONDITION_A, segment=part.segment,
                    step=at,
                    text=f"step {at} of {part.segment} read shard position "
                         f"{document.shard_position}, which the snapshot does not hold",
                ))
                continue
            gate = walk.shard_gate(entry.shard_instance_id, entry.path)
            if gate.missing is not None:
                first.see(index, at, _RECORD, gate.missing)
                continue
            first.see(index, at, _RECORD, gate.label)
            row = store.map_row(entry.shard_instance_id, document.ordinal)
            if row is None:
                first.see(index, at, _RECORD, Failing(
                    names=v.NAMES_MISSING_ORIGIN, condition=v.CONDITION_A,
                    artifact=walk.name(entry.shard_instance_id), instance=entry.shard_instance_id,
                    record=f"D{document.ordinal}", segment=part.segment, step=at,
                    text=f"step {at} of {part.segment} read document {document.ordinal} of "
                         f"{walk.name(entry.shard_instance_id)}, which its map does not hold",
                ))
                continue
            status = walk.document(gate, row)
            if status.verdict == HOLDS:
                read = _first_covered_read(store, traces, part.segment, document, upto, notes)
                if read == NO_ROWS:
                    # The same answer as an ordinal the map does not hold: the step read a
                    # document and the segment's rows do not hold the read.
                    first.see(index, at, _RECORD, Failing(
                        names=v.NAMES_MISSING_ORIGIN, condition=v.CONDITION_A,
                        artifact=walk.name(entry.shard_instance_id), instance=entry.shard_instance_id,
                        record=f"D{document.ordinal}", segment=part.segment, step=at,
                        text=f"step {at} of {part.segment} read document {document.ordinal} of "
                             f"{walk.name(entry.shard_instance_id)}, and the segment's trace rows "
                             "hold no read of it",
                    ))
                    continue
                if read is None:
                    continue  # read in an uncovered range only: the step is untraced
                holds_generation = holds_generation or status.generation
                if exposure is None or (index, read[0]) < exposure[:2]:
                    exposure = (index, read[0], part.segment, read[1], status.chain)
            elif status.verdict == MAY:
                first.see(index, at, _RECORD, status.failing)
            else:
                clear_generation = clear_generation or status.generation
                for job in status.jobs:
                    jobs.setdefault(job)

    if exposure is not None:
        index, step, segment_name, basis, chain = exposure
        hop = Hop(segment=segment_name, step=step, link=v.HOP_CONSUMED)
        return Outcome(
            HOLDS, chain=(hop,) + chain, generation=holds_generation, walked=walked, basis=basis,
            untraced_steps=tuple(untraced), first_exposure=(segment_name, step),
            level=v.LEVEL_CONSUMED if index == last else v.LEVEL_INHERITED,
        )
    if first.failing is not None:
        return Outcome(MAY, failing=first.failing, walked=walked, untraced_steps=tuple(untraced))
    return Outcome(
        CLEAR, generation=clear_generation, jobs=tuple(jobs), walked=walked,
        runs=tuple(runs), steps=tuple(steps_scope),
    )
