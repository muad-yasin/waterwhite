"""The exposure join of one run segment (plan 3.5; docs/join.md).

``compute_join`` decides, in a fixed order, whether the segment's data order can
be trusted and by which route, and then writes down what every completed step
read. It touches no store. ``join_segment`` in the package's ``__init__`` reads
the files, calls it and hands the result to the store.

The order of the questions, each with the cause it gives when the answer is no.

1. Does the run state a release the join is not verified for (plan F6, the
   store's list of verified joins)? Then: "data-order trace unverified for this
   version", whatever else the run recorded.
2. For a cyclic run: does the run record hold shuffle digests at all? No:
   "cyclic order not recorded", whatever cache files were kept (plan 3.5).
3. Does the run record hold a digest for every index array of the segment? No,
   or there is no run record: "data-order trace missing", reason "no recorded
   digest". No replay is tried (plan 3.5, Check 2 (iii)).
4. The rest of the gate. A run record that states no release: "data-order trace
   unverified for this version". A release that is verified, and not with this
   sampler type and sampler class: "data-order trace unverified for this sampler
   type". The cyclic sampler without data sharding follows a rule the join was
   not verified for and gets the same cause.
5. Does the run record hold what the walk needs (the shard list, the batch
   sizes, the sequence length, the extra-token flag, a launch)? No:
   "data-order trace missing", reason "run record incomplete": the join cannot
   read the run and guesses nothing. A run read with the extra token off: the
   same cause, reason "extra token off".
6. Is each shard's index file there, and is it the file the trainer was handed
   (its SHA-256 equal to the one in the run record's shard list, Decision 27)?
   A digest that was not recorded: reason "no recorded digest". A digest that
   differs: reason "digest mismatch". A file that is gone or cannot be read:
   reason "shard index unreadable"; a shard with no prefix in the list is a
   record that lacks what the walk needs: reason "run record incomplete".
7. The index arrays. First route: the cache files, when a file with the recorded
   digest is found for every array. Second route: the replay, when its arrays
   have the recorded digests. Neither: "data-order trace missing", reason
   "digest mismatch" (plan 3.5, Check 2 (ii)); with no cache file kept and a
   setting of the replay not recorded, reason "no recorded digest". Arrays with
   the recorded digests whose shape no array of their kind has: reason "index
   array malformed".
8. For a cyclic run: for every epoch a launch reads in and every rank, does the
   run record hold the epoch's shuffle digest? No: "cyclic order not recorded".
   Then the join computes each such order itself, by calling torch as the
   sampler does (``_shuffle``), and compares its digest and count with the
   recorded ones. Different: "cyclic order digest mismatch". Where torch cannot
   be imported the registration is refused ("torch not available"). Each order
   computed is kept in ``JoinResult.shuffle`` with its digest and count beside the
   recorded ones, those that differ included.

When every answer is yes the route is "cache files" or "replay" and each step the
map covers gets a trace. A step the map does not cover (no launch holds it, its
batch size is unknown or no whole number of rounds, a sequential pass ended
before it) gets no trace and is listed with the reason: it is an untraced step
(plan 2.2(e)), not a cause of the segment.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import ids
from .. import vocabulary as v
from ..errors import ReplayNotPossible, UnsupportedFormat
from ..records import StepTrace, TraceSample
from ..store import artifact_files
from ._arrays import ArrayKey, IndexArrays, digest_of, needed_keys, read_cache_files
from ._record import Launch, RunRecord, StepRecord, parse_run_record, parse_step_record
from ._replay import replay_arrays
from ._rows import SampleResolver, cut
from ._sampler import (
    CannotPlace,
    CyclicShape,
    check_sequential,
    cyclic_epochs,
    cyclic_step,
    sequential_step,
)
from ._shuffle import epoch_permutation, rank_order

OrderKey = tuple[int, int, int]  # (launch position, epoch, rank)


@dataclass(frozen=True)
class ReplayReport:
    """What became of the replay route (plan 3.5; Check 2 of plan 5.5)."""

    attempted: bool
    accepted: bool
    differing: tuple[ArrayKey, ...] = ()  # arrays whose replayed digest is not the recorded one
    why_not: str | None = None  # one sentence, when it was not attempted or not accepted
    reason: str | None = None  # a word of TRACE_MISSING_REASONS, when the replay was refused or
    #                            not tried for want of a digest to hold it against (Check 2)


_NO_REPLAY = ReplayReport(False, False)


@dataclass(frozen=True)
class ShuffleCheck:
    """One order the join computed for a cyclic run and held against the run record:
    the (launch, epoch, rank) it is the order of, the digest and count of the order
    the join computed, the digest and count the run record holds (the count is None
    where the record gives none) and whether the two agree. ``equal`` is the very
    comparison that accepts or refuses the order."""

    launch: int
    epoch: int
    rank: int
    digest: str
    count: int
    recorded_digest: str
    recorded_count: int | None
    equal: bool


@dataclass(frozen=True)
class JoinResult:
    """The join of one segment.

    ``route``, ``cause`` and ``cause_detail`` are what ``Store.register_run_segment``
    takes as the segment's trace route, cause and cause detail: words of
    ``waterwhite.vocabulary``. ``steps`` are the segment's completed steps as that
    method takes them, ``traces`` what ``Store.write_step_traces`` takes.
    ``explanation`` is one sentence for a person and is stored nowhere. ``shuffle``
    holds the orders the join computed for a cyclic run and held against the run
    record, one for each (launch, epoch, rank) its launches read, in that order; it
    is empty for any other run, and for a run that stopped at an earlier question
    (nothing was computed)."""

    segment: str | None
    route: str
    cause: str | None
    cause_detail: str | None
    explanation: str | None
    steps: tuple[tuple[int, int, int | None], ...]
    traces: tuple[StepTrace, ...]
    untraced: tuple[tuple[int, str], ...]  # with a route: registered steps that got no trace
    unsized: tuple[int, ...]  # steps left out because nothing gives their batch size
    checkpoints: tuple[tuple[str, int], ...]  # (name, step), as the step record names them
    dataset_length: int | None
    replay: ReplayReport
    shuffle: tuple[ShuffleCheck, ...] = ()

    @property
    def steps_written(self) -> int:
        return len(self.traces)

    @property
    def untraced_steps(self) -> tuple[int, ...]:
        """The steps of the segment that are untraced in the sense of plan 2.2(e): every
        step when the route is "none"; else the registered steps with no trace, the
        steps whose trace has an uncovered token range and the steps left out of the
        registration for want of a batch size."""
        steps = {step for step, _, _ in self.steps} | set(self.unsized)
        if self.route != v.ROUTE_NONE:
            steps -= {trace.step for trace in self.traces} - set(self.lowered_steps)
        return tuple(sorted(steps))

    @property
    def lowered_steps(self) -> tuple[int, ...]:
        """Steps whose trace holds a sample with an uncovered token range."""
        return tuple(trace.step for trace in self.traces
                     if any(sample.uncovered is not None for sample in trace.samples))


def _batch_sizes(record: RunRecord | None, step_record: StepRecord) -> dict[int, int | None]:
    """The batch size of every completed step: the step record's where it gives
    one (the operator's batch schedule, plan F2), else the global batch size.
    The completed steps are the steps of the launches and the steps the step
    record lists. A run record that names a batch schedule gives no default: the
    form of that field is not fixed, and the join does not read it."""
    numbers = set(step_record.steps)
    default = None
    if record is not None:
        for launch in record.launches:
            numbers.update(range(launch.first_step, launch.last_step + 1))
        if record.batch_schedule is None:
            default = record.global_batch_size
    sizes: dict[int, int | None] = {}
    for step in sorted(numbers):
        stated = step_record.steps.get(step, (None, None))[0]
        sizes[step] = default if stated is None else stated
    return sizes


def _shard_lengths(record: RunRecord, record_dir: str | os.PathLike | None):
    """The token count of each document of each shard, read from the shard's
    index file, after the file was held against the run record's digest.
    Returns (arrays, None, None), or (None, reason, sentence); the reason is always
    a word of ``TRACE_MISSING_STORED_REASONS``."""
    lengths = []
    for entry in record.shards:
        what = f"shard {entry.position} of the run record's list"
        if entry.prefix is None:
            return None, v.TRACE_MISSING_RECORD_INCOMPLETE, (
                f"{what} has no prefix, so its index file cannot be found")
        prefix = Path(entry.prefix)
        if not prefix.is_absolute() and record_dir is not None:
            prefix = Path(record_dir) / prefix
        _, idx_path = artifact_files.shard_paths(prefix)
        if not idx_path.is_file():
            return None, v.TRACE_MISSING_SHARD_INDEX_UNREADABLE, (
                f"the index file of {what} is not at the place the run record names")
        if entry.idx_sha256 is None:
            return None, v.TRACE_MISSING_NO_DIGEST, (
                f"the run record holds no digest for the index file of {what}")
        if ids.file_digest(idx_path) != entry.idx_sha256:
            return None, v.TRACE_MISSING_DIGEST_MISMATCH, (
                f"the index file of {what} is not the file the trainer was handed: its digest "
                "differs from the run record's")
        try:
            lengths.append(artifact_files.read_shard_index(idx_path).sequence_lengths)
        except UnsupportedFormat as error:
            return None, v.TRACE_MISSING_SHARD_INDEX_UNREADABLE, (
                f"the index file of {what} cannot be read: {error.detail}")
    return lengths, None, None


@dataclass(frozen=True)
class _Placed:
    launch: Launch
    step: int
    consumed: int
    batch: int


def _place_steps(record: RunRecord, sizes: Mapping[int, int | None], length: int, cyclic: bool
                 ) -> tuple[list[_Placed], dict[int, str], dict[int, CyclicShape]]:
    """Give every step of every launch its counter, as far as the map covers it.
    The counter of a step is the launch's counter plus the batch sizes of the
    launch's steps before it (plan F4). Once a step of a launch cannot be placed,
    the counter of every later step of that launch is unknown too."""
    placed: list[_Placed] = []
    untraced: dict[int, str] = {}
    shapes: dict[int, CyclicShape] = {}
    micro = record.micro_batch_size
    for launch in record.launches:
        ranks = launch.data_parallel_size
        consumed = launch.consumed_at_start
        why = None
        if cyclic:
            try:
                shapes[launch.position] = CyclicShape.of(length, micro, ranks)
            except CannotPlace as error:
                why = str(error)
        for step in range(launch.first_step, launch.last_step + 1):
            batch = sizes.get(step)
            if why is None:
                try:
                    if batch is None:
                        raise CannotPlace(f"nothing gives the batch size of step {step}")
                    if cyclic:
                        if batch != record.global_batch_size:
                            raise CannotPlace(
                                f"the step record gives step {step} a batch of {batch} samples "
                                f"and the global batch size is {record.global_batch_size}; plan "
                                "3.5 names no batch schedule for the cyclic sampler")
                        cyclic_epochs(consumed, batch, micro, ranks, shapes[launch.position])
                    else:
                        check_sequential(consumed, batch, micro, ranks, length)
                except CannotPlace as error:
                    why = str(error)
            if why is not None:
                untraced[step] = f"launch {launch.position}: {why}"
                continue
            placed.append(_Placed(launch, step, consumed, batch))
            consumed += batch
    return placed, untraced, shapes


def _cyclic_orders(record: RunRecord, placed: Sequence[_Placed], shapes: Mapping[int, CyclicShape]):
    """The order of every epoch a launch reads in, for every rank, computed by the
    join and held against the run record's digest and count (plan 3.5, the owner's
    ruling on the order of a cyclic run).

    Returns (orders, checks, None, None), or (None, checks, cause, sentence). Whether
    the record holds every digest is decided before anything is drawn, so a run that
    records too little never loads torch and has no checks. Every needed order is
    drawn and held, so ``checks`` lists them all, the ones that differ too. A computed
    order that differs from the recorded one gives "cyclic order digest mismatch"
    (naming the first that differs) and is never used."""
    needed: set[OrderKey] = set()
    for item in placed:
        launch = item.launch
        for epoch in cyclic_epochs(item.consumed, item.batch, record.micro_batch_size,
                                   launch.data_parallel_size, shapes[launch.position]):
            needed.update((launch.position, epoch, rank)
                          for rank in range(launch.data_parallel_size))
    for key in sorted(needed):
        if key not in record.shuffle_digests:
            return None, (), v.NAMES_CYCLIC_NOT_RECORDED, (
                f"the run record holds no shuffle digest for launch {key[0]}, epoch {key[1]}, "
                f"rank {key[2]}")
    held: dict[OrderKey, list[int]] = {}
    checks: list[ShuffleCheck] = []
    permutations: dict[tuple[int, int], list[int]] = {}
    for key in sorted(needed):
        launch, epoch, rank = key
        bucket = shapes[launch].bucket
        if (epoch, bucket) not in permutations:
            permutations[(epoch, bucket)] = epoch_permutation(epoch, bucket)
        order = rank_order(permutations[(epoch, bucket)], rank, bucket)
        digest, count = record.shuffle_digests[key]
        computed = ids.order_digest(order)
        equal = computed == digest and (count is None or count == len(order))
        checks.append(ShuffleCheck(launch, epoch, rank, computed, len(order), digest, count, equal))
        held[key] = order
    for check in checks:
        if not check.equal:
            return None, tuple(checks), v.NAMES_CYCLIC_DIGEST_MISMATCH, (
                f"the order the join computes for launch {check.launch}, epoch {check.epoch}, "
                f"rank {check.rank} from the run record's sizes is not the order whose digest "
                "the run record holds")
    return held, tuple(checks), None, None


def compute_join(
    run_record: Mapping | None,
    *,
    verified_joins: Iterable[tuple[str, str, str]],
    record_dir: str | os.PathLike | None = None,
    cache_dir: str | os.PathLike | None = None,
    step_record: Mapping | None = None,
) -> JoinResult:
    """Join one run segment. See the module text for the order of the questions.

    ``run_record`` is the run record as a JSON object, or None for a run that has
    none. ``verified_joins`` is the store's list (``Store.verified_joins()``).
    ``record_dir`` is the folder the run record lies in: a shard prefix of the
    record is relative to it. ``cache_dir`` is the kept data-order cache folder.
    ``step_record`` is the operator's step record. The orders of a cyclic run are
    computed here, by calling torch as the sampler does, and held against the run
    record's digests.

    Raises ``InvalidRecord`` for a run record or step record that is not one, and
    ``TorchUnavailable`` for a cyclic run whose shuffle must be computed and torch
    cannot be imported."""
    record = None if run_record is None else parse_run_record(run_record)
    steps_given = parse_step_record(step_record, None if record is None else record.segment)
    sizes = _batch_sizes(record, steps_given)
    steps = tuple((step, size, steps_given.steps.get(step, (None, None))[1])
                  for step, size in sizes.items() if size is not None)
    common = dict(
        segment=None if record is None else record.segment,
        steps=steps,
        unsized=tuple(step for step, size in sizes.items() if size is None),
        checkpoints=steps_given.checkpoints,
    )

    def no_trace(cause: str, detail: str | None, why: str, *, replay: ReplayReport = _NO_REPLAY,
                 length: int | None = None, shuffle: tuple[ShuffleCheck, ...] = ()) -> JoinResult:
        return JoinResult(route=v.ROUTE_NONE, cause=cause, cause_detail=detail, explanation=why,
                          traces=(), untraced=(), dataset_length=length, replay=replay,
                          shuffle=shuffle, **common)

    # 1. The release the run states (plan F6).
    verified = {tuple(join) for join in verified_joins}
    tags = {join[0] for join in verified}
    if record is not None and record.release_tag is not None and record.release_tag not in tags:
        return no_trace(v.NAMES_TRACE_UNVERIFIED_VERSION, None, (
            f"the join is not verified for the release the run states, {record.release_tag!r}"))

    # 2. A cyclic run with no recorded shuffle (plan 3.5).
    cyclic = record is not None and record.sampler_type == v.SAMPLER_CYCLIC
    if cyclic and not record.shuffle_digests:
        return no_trace(v.NAMES_CYCLIC_NOT_RECORDED, None,
                        "the run record of this cyclic run holds no shuffle digest")

    # 3. The recorded digests of the index arrays (plan 3.5, replay rule).
    never = ReplayReport(False, False, reason=v.TRACE_MISSING_NO_DIGEST,
                         why_not="no replay is tried without a recorded digest to hold it against")
    if record is None:
        return no_trace(v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST,
                        "the run has no run record, so no digest was recorded", replay=never)
    shard_count = v.shard_count_of(len(record.shards), (shard for _, shard in record.cache_digests))
    keys = needed_keys(shard_count)
    absent = [f"{kind} of shard {shard}" if shard is not None else f"{kind} of the blend"
              for kind, shard in keys if (kind, shard) not in record.cache_digests]
    if shard_count == 0 or absent:
        return no_trace(v.NAMES_TRACE_MISSING, v.TRACE_MISSING_NO_DIGEST, (
            "the run record holds no digest of an index array" if shard_count == 0 else
            "the run record holds no digest for " + ", ".join(absent)), replay=never)

    # 4. The rest of the gate: a release that is stated, and the sampler (plan F6, 3.5).
    if record.release_tag is None:
        return no_trace(v.NAMES_TRACE_UNVERIFIED_VERSION, None,
                        "the run record states no release, so the join is verified for none")
    key = (record.release_tag, record.sampler_type, record.sampler_class)
    if key not in verified:
        return no_trace(v.NAMES_TRACE_UNVERIFIED_SAMPLER, None, (
            f"the join is verified for release {key[0]!r} and not for the sampler "
            f"{key[1]!r} ({key[2]!r})"))
    if not cyclic and record.sampler_type != v.SAMPLER_SEQUENTIAL:
        return no_trace(v.NAMES_TRACE_UNVERIFIED_SAMPLER, None, (
            f"the join walks the samplers 'single' and 'cyclic', not {record.sampler_type!r}"))
    if cyclic and record.data_sharding is not True:
        return no_trace(v.NAMES_TRACE_UNVERIFIED_SAMPLER, None, (
            "the join is verified for the cyclic sampler with data sharding on; the run record "
            f"states data_sharding {record.data_sharding!r}, which follows another rule"))

    # 5. What the walk needs of the run record.
    needs = {
        "the shard list": record.shards or None,
        "the micro batch size": record.micro_batch_size,
        "the global batch size": record.global_batch_size,
        "the sequence length": record.sequence_length,
        "the extra-token flag": record.add_extra_token,
        "a launch": record.launches or None,
    }
    lacking = [name for name, value in needs.items() if value is None]
    if lacking:
        return no_trace(v.NAMES_TRACE_MISSING, v.TRACE_MISSING_RECORD_INCOMPLETE,
                        "the run record does not hold " + ", ".join(lacking))
    if record.add_extra_token is not True:
        return no_trace(v.NAMES_TRACE_MISSING, v.TRACE_MISSING_EXTRA_TOKEN_OFF, (
            "the run was read with the extra token off; the join is verified for samples of "
            "sequence_length + 1 tokens only (plan 3.5, F9)"))

    # 6. The shards' own index files.
    lengths, reason, why = _shard_lengths(record, record_dir)
    if lengths is None:
        return no_trace(v.NAMES_TRACE_MISSING, reason, why)

    # 7. The index arrays: the cache files, else the replay.
    found, missing, any_file = read_cache_files(cache_dir, record.cache_digests, keys)
    replay = _NO_REPLAY
    if not missing:
        route = v.ROUTE_CACHE_FILES
    else:
        route = v.ROUTE_REPLAY
        try:
            found = replay_arrays(record, lengths)
        except ReplayNotPossible as error:
            # Cache files that are there and differ: a mismatch. Cache files and a
            # setting of the replay both gone: nothing a replay could be held
            # against, which docs/interfaces.md section 2 gives the reason "no
            # recorded digest".
            return no_trace(
                v.NAMES_TRACE_MISSING,
                v.TRACE_MISSING_DIGEST_MISMATCH if any_file else v.TRACE_MISSING_NO_DIGEST,
                ("the cache folder holds no file with the recorded digests" if any_file else
                 "no cache file was kept") + f", and no replay is possible: {error.detail}",
                replay=ReplayReport(
                    False, False, why_not=error.detail,
                    reason=v.TRACE_MISSING_DIGEST_MISMATCH if any_file else v.TRACE_MISSING_NO_DIGEST))
        differing = tuple(k for k in keys
                          if k not in found or digest_of(found[k]) != record.cache_digests[k])
        if differing:
            return no_trace(
                v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH,
                f"the replay gives {len(differing)} of {len(keys)} index arrays a digest that "
                "is not the recorded one, so the replay supports no answer",
                replay=ReplayReport(True, False, differing,
                                    "the replayed digests differ from the recorded ones",
                                    reason=v.TRACE_MISSING_DIGEST_MISMATCH))
        replay = ReplayReport(True, True)
    arrays = IndexArrays.build(found, shard_count)
    if arrays is None:
        # Reached by the cache-files route only (the replay builds each array in its
        # kind's shape): the kept files hold the recorded digests, so the record and the
        # files agree, and the arrays are no index arrays of their kinds. Differing
        # evidence, read only from the files, is a cause and not an error.
        return no_trace(v.NAMES_TRACE_MISSING, v.TRACE_MISSING_ARRAY_MALFORMED,
                        "an index array has a shape no array of its kind has", replay=replay)

    # 8. The steps, and for a cyclic run the order of every epoch they read in.
    placed, untraced, shapes = _place_steps(record, sizes, arrays.length, cyclic)
    held: Mapping[OrderKey, list[int]] = {}
    checks: tuple[ShuffleCheck, ...] = ()
    if cyclic:
        held, checks, cause, why = _cyclic_orders(record, placed, shapes)
        if held is None:
            return no_trace(cause, None, why, replay=replay, length=arrays.length, shuffle=checks)

    resolver = SampleResolver(arrays, lengths, record.sequence_length, 1)
    lowered = {step: last for step, _, last in steps if last is not None}
    micro = record.micro_batch_size
    traces = []
    for item in placed:
        position, ranks = item.launch.position, item.launch.data_parallel_size
        if cyclic:
            draws = cyclic_step(item.consumed, item.batch, micro, ranks, shapes[position],
                                lambda epoch, rank: held[(position, epoch, rank)])
        else:
            draws = sequential_step(item.consumed, item.batch, micro, ranks, arrays.length)
        samples = []
        for draw in draws:
            rows, covered = resolver.resolve(draw.global_index)
            if draw.sample_pos == item.batch - 1 and item.step in lowered:
                # The operator's note can only lower the mark (plan 3.5).
                covered = min(covered, lowered[item.step])
                rows = cut(rows, covered)
            samples.append(TraceSample(draw.sample_pos, draw.global_index, draw.rank,
                                       resolver.sample_tokens, covered, rows))
        traces.append(StepTrace(item.step, item.batch, tuple(samples)))

    registered = {step for step, _, _ in steps}
    for step in registered - {item.step for item in placed}:
        untraced.setdefault(step, "no launch of the run record holds the step")
    return JoinResult(
        route=route, cause=None, cause_detail=None, explanation=None, traces=tuple(traces),
        untraced=tuple(sorted((step, why) for step, why in untraced.items() if step in registered)),
        dataset_length=arrays.length, replay=replay, shuffle=checks, **common,
    )
