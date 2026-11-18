"""Stepping through a segment: which samples each completed step reads.

Two walks over the same launches, both made of Megatron's own sampler objects:

- ``drive_orders`` asks the sampler classes alone for their index lists, over the
  steps each launch completes.
- ``consume`` is the independent consumer of plan 5.3. It hands each sampler to a
  torch DataLoader over Megatron's dataset and logs what the item getter reads.

A third drive, ``epoch_orders``, gives the order digests of a cyclic run (plan
3.5; DECISIONS.md D-16): for every epoch a launch touches it builds the sampler
class at the start of that epoch and takes the whole pass, whatever part of the
epoch the launch read.

The step rule all of them implement: a step reads ``global_batch_size`` samples.
Each of the n data-parallel ranks has its own sampler, built with that rank's
number and the size n, and no process group. In one step each rank draws k micro
batches of m samples from its sampler, where k = global_batch_size / (m * n) (see
``Segment.micro_batches``). Ranks run one after another in this one process; a
rank's reads depend on its own sampler only, so the order of ranks changes
nothing but the order of the log lines.

Where a sample sits inside its step (``sample_pos`` of docs/interfaces.md 8.3):
micro-batch round first, then rank, then place in the micro batch, so
``sample_pos = round * m * n + rank * m + place``. For the sequential sampler this
is the order of the global index: Megatron-LM's ``MegatronPretrainingSampler``
gives rank r the slice ``[r * m, (r + 1) * m)`` of each run of m * n consecutive
indices.
"""

from __future__ import annotations

import json
from pathlib import Path

from oracle.config import DATALOADER_CYCLIC, Launch, Segment, new_output_path
from oracle.digests import order_digest
from oracle.errors import LaunchOutrunsDataset, OracleError, SamplerRefused
from oracle.log_rows import LOG_FIELDS
from oracle.megatron_data import (
    Built,
    RankStream,
    ReadTap,
    Samplers,
    TappedDataset,
    make_loader,
    make_sampler,
)

SUMMARY_FORMAT = "waterwhite.oracle-summary/1"



def _where(segment: Segment, launch: Launch, rank: int) -> str:
    return f"segment {segment.segment!r}, launch {launch.position}, rank {rank}"


# --- A launch that would outrun its dataset ---------------------------------------------------


def restart_allowed(segment: Segment) -> bool:
    """Whether a rank's sampler may be started again when its pass ends inside a
    launch. The cyclic sampler moves to its next epoch that way. The sequential
    sampler would start again at the consumed count it was built with; that is
    allowed only when the config says so."""
    return segment.dataloader_type == DATALOADER_CYCLIC or segment.allow_sequential_restart


def check_launches_fit(segment: Segment, built: Built) -> None:
    """Refuse a sequential launch that asks for more steps than its dataset holds.

    Megatron's sequential sampler walks the indices from the launch's consumed
    count to the dataset's length and gives one micro batch per rank for every
    full run of ``micro_batch_size * data_parallel_size`` indices
    (``MegatronPretrainingSampler.__iter__``). When the walk ends, a trainer's
    loop starts the pass again and the sampler begins again at the consumed count
    it was built with, so the launch reads samples a second time. The step-to-
    sample map of plan 3.5 does not cover that, and the fixture avoids it. So the
    launch is refused here unless the config sets ``allow_sequential_restart``.

    A launch whose consumed count is at or past the dataset's length is refused
    by Megatron's class itself; that refusal keeps its own name."""
    if restart_allowed(segment):
        return
    for launch in segment.launches:
        if launch.consumed_at_start >= built.samples:
            continue
        per_round = segment.micro_batch_size * launch.data_parallel_size
        needed = launch.steps * segment.micro_batches(launch)
        available = (built.samples - launch.consumed_at_start) // per_round
        if needed > available:
            raise LaunchOutrunsDataset(
                f"segment {segment.segment!r}, launch {launch.position}: steps "
                f"{launch.first_step} to {launch.last_step} need {needed} micro batches per "
                f"rank, and the dataset of {built.samples} samples gives {available} from "
                f"consumed count {launch.consumed_at_start}; the sequential sampler would "
                "start its pass again. Lower the steps, raise train_iters or the shards, or "
                "set allow_sequential_restart to true"
            )


# --- The orders the sampler classes give ------------------------------------------------------


class _Orders:
    """The indices each rank drew, kept apart by launch, rank and pass."""

    def __init__(self) -> None:
        self._lists: dict[tuple[int, int, int], dict] = {}

    def add(self, launch: Launch, rank: int, stream: RankStream, indices, step: int) -> None:
        key = (launch.position, rank, stream.passes)
        entry = self._lists.setdefault(
            key, {"epoch": stream.epoch, "indices": [], "first_step": step, "last_step": step}
        )
        if entry["epoch"] != stream.epoch:
            raise OracleError(f"the epoch of launch {key[0]}, rank {rank} changed inside a pass")
        entry["indices"].extend(int(i) for i in indices)
        entry["last_step"] = step

    def rows(self) -> list[dict]:
        return [
            {
                "launch": launch,
                "rank": rank,
                "pass": pass_number,
                "epoch": entry["epoch"],
                "first_step": entry["first_step"],
                "last_step": entry["last_step"],
                "count": len(entry["indices"]),
                "digest": order_digest(entry["indices"]),
                "indices": list(entry["indices"]),
            }
            for (launch, rank, pass_number), entry in sorted(self._lists.items())
        ]


def drive_orders(segment: Segment, built: Built, samplers: Samplers) -> list[dict]:
    """Drive the sampler classes over the segment's steps and return what each
    rank drew: one row per launch, rank and pass of the sampler, with the epoch
    the cyclic sampler named, the first and last step of the pass, the indices in
    order and their digest. The dataset is not read."""
    check_launches_fit(segment, built)
    orders = _Orders()
    for launch in segment.launches:
        rounds = segment.micro_batches(launch)
        streams = []
        for rank in range(launch.data_parallel_size):
            sampler = make_sampler(samplers, segment, launch, rank, built)
            streams.append(
                RankStream(
                    sampler,
                    sampler,
                    _where(segment, launch, rank),
                    restart_allowed=restart_allowed(segment),
                )
            )
        for step in range(launch.first_step, launch.last_step + 1):
            for _round in range(rounds):
                for rank, stream in enumerate(streams):
                    batch = stream.next()
                    orders.add(launch, rank, stream, batch, step)
    return orders.rows()


# --- The order digests of a cyclic run --------------------------------------------------------


def _whole_pass(sampler, where: str) -> list[int]:
    """Every index one pass of a sampler yields, in order."""
    try:
        return [int(i) for batch in sampler for i in batch]
    except AssertionError as e:
        raise SamplerRefused(f"{where}: Megatron's sampler stopped: {e!r}") from e


def _is_part_of(part: list[int], whole: list[int]) -> bool:
    """Whether ``part`` is one unbroken run of ``whole``. A rank's order inside an
    epoch names no index twice, so the place of the first index decides it."""
    if not part or part[0] not in whole:
        return False
    at = whole.index(part[0])
    return whole[at : at + len(part)] == part


def epoch_orders(
    segment: Segment, built: Built, samplers: Samplers, orders: list[dict]
) -> list[dict]:
    """The whole order of every epoch a launch of a cyclic run touches, per rank
    (DECISIONS.md D-16). ``orders`` are the rows of ``drive_orders``: they say
    which epochs each launch read in.

    How the whole order is got, with nothing but the sampler class. A sampler
    built at consumed count 0 is run for one pass: that is epoch 0, and the class
    then holds, in ``consumed_samples``, the count one epoch adds. A sampler built
    at ``epoch`` times that count must name the epoch itself (``sampler.epoch``);
    its one pass is that epoch's whole order for the rank. The part the launch
    drew must be one unbroken run of it, or the run stops.

    A sequential run has no shuffle: an empty list."""
    if segment.dataloader_type != DATALOADER_CYCLIC:
        return []
    launches = {launch.position: launch for launch in segment.launches}
    drawn: dict[tuple[int, int, int], list[int]] = {}
    for order in orders:
        if order["epoch"] is None:
            raise OracleError("the cyclic sampler named no epoch for a pass")
        key = (order["launch"], order["epoch"], order["rank"])
        if key in drawn:
            raise OracleError(f"two passes of one launch, epoch and rank: {key}")
        drawn[key] = order["indices"]

    whole_orders: dict[tuple[int, int, int], list[int]] = {}  # (size, rank, epoch) -> order
    per_epoch: dict[tuple[int, int], int] = {}  # (size, rank) -> consumed count per epoch

    def whole(launch: Launch, rank: int, epoch: int) -> list[int]:
        size = launch.data_parallel_size
        where = f"{_where(segment, launch, rank)}, epoch {epoch}"
        if (size, rank) not in per_epoch:
            probe = make_sampler(samplers, segment, launch, rank, built, consumed_samples=0)
            first = _whole_pass(probe, where)
            added = int(probe.consumed_samples)
            if int(probe.epoch) != 0 or added <= 0 or not first:
                raise OracleError(
                    f"{where}: a sampler built at consumed count 0 did not give epoch 0"
                )
            per_epoch[(size, rank)] = added
            whole_orders[(size, rank, 0)] = first
        if (size, rank, epoch) not in whole_orders:
            start = epoch * per_epoch[(size, rank)]
            sampler = make_sampler(samplers, segment, launch, rank, built, consumed_samples=start)
            indices = _whole_pass(sampler, where)
            if int(sampler.epoch) != epoch:
                raise OracleError(
                    f"{where}: a sampler built at consumed count {start} named epoch "
                    f"{int(sampler.epoch)}"
                )
            whole_orders[(size, rank, epoch)] = indices
        return whole_orders[(size, rank, epoch)]

    rows = []
    for (position, epoch, rank), part in sorted(drawn.items()):
        indices = whole(launches[position], rank, epoch)
        if not _is_part_of(part, indices):
            raise OracleError(
                f"segment {segment.segment!r}, launch {position}, rank {rank}: what the launch "
                f"drew in epoch {epoch} is not one unbroken run of the epoch's whole order"
            )
        rows.append(
            {
                "launch": position,
                "epoch": epoch,
                "rank": rank,
                "digest": order_digest(indices),
                "count": len(indices),
                "indices": list(indices),
            }
        )
    return rows


def shuffle_digests(
    segment: Segment, built: Built, samplers: Samplers, orders: list[dict]
) -> list[dict]:
    """The ``shuffle_digests`` of a run record (docs/interfaces.md 8.1).

    One row per launch, epoch and rank of a cyclic run: the digest of the whole
    order of that epoch for that rank, and its length. A launch that starts or
    ends inside an epoch records the whole epoch all the same, so a resume inside
    an epoch records the digest the launch before it recorded (DECISIONS.md
    D-16). A sequential run gets an empty list."""
    return [
        {key: row[key] for key in ("launch", "epoch", "rank", "digest", "count")}
        for row in epoch_orders(segment, built, samplers, orders)
    ]


# --- The consumer -----------------------------------------------------------------------------


def _walk(segment: Segment, built: Built, samplers: Samplers, emit) -> dict:
    """Run the independent consumer over a segment, hand each log row to ``emit``
    and return the summary. See ``consume``."""
    check_launches_fit(segment, built)
    tap = ReadTap.attach(built)
    orders = _Orders()
    launches = []
    rows_written = 0
    samples_read = 0
    m = segment.micro_batch_size
    previous: Launch | None = None
    for launch in segment.launches:
        n = launch.data_parallel_size
        rounds = segment.micro_batches(launch)
        datasets, streams = [], []
        for rank in range(n):
            sampler = make_sampler(samplers, segment, launch, rank, built)
            dataset = TappedDataset(built, tap)
            datasets.append(dataset)
            streams.append(
                RankStream(
                    sampler,
                    make_loader(dataset, sampler),
                    _where(segment, launch, rank),
                    restart_allowed=restart_allowed(segment),
                )
            )
        for step in range(launch.first_step, launch.last_step + 1):
            for round_number in range(rounds):
                for rank, stream in enumerate(streams):
                    stream.next()
                    fetched = datasets[rank].take()
                    if len(fetched) != m:
                        raise OracleError(
                            f"{_where(segment, launch, rank)}, step {step}: a micro batch of "
                            f"{len(fetched)} samples, expected {m}"
                        )
                    orders.add(launch, rank, stream, [index for index, _ in fetched], step)
                    for place, (index, reads) in enumerate(fetched):
                        sample_pos = round_number * m * n + rank * m + place
                        samples_read += 1
                        for read in reads:
                            row = (
                                segment.segment,
                                step,
                                sample_pos,
                                index,
                                rank,
                                read.shard_position,
                                read.ordinal,
                                read.token_start,
                                read.token_end,
                            )
                            emit(dict(zip(LOG_FIELDS, row)))
                            rows_written += 1
        continues = None
        if previous is not None:
            continues = (
                launch.step_at_start == previous.last_step
                and launch.consumed_at_start
                == previous.consumed_at_start + previous.steps * segment.data.global_batch_size
            )
        launches.append(
            {
                "position": launch.position,
                "data_parallel_size": n,
                "micro_batches_per_step": rounds,
                "consumed_at_start": launch.consumed_at_start,
                "first_step": launch.first_step,
                "last_step": launch.last_step,
                "passes_per_rank": [stream.passes for stream in streams],
                "continues_the_launch_before": continues,
            }
        )
        previous = launch
    order_rows = orders.rows()
    return {
        "format": SUMMARY_FORMAT,
        "run": segment.run,
        "segment": segment.segment,
        "dataloader_type": segment.dataloader_type,
        "sampler_class": samplers.class_name(segment.dataloader_type),
        "dataset_samples": built.samples,
        "requested_samples": segment.data.requested_samples,
        "sample_tokens": segment.data.sample_tokens,
        "launches": launches,
        "samples_read": samples_read,
        "samples_checked_against_their_reads": samples_read,
        "rows_written": rows_written,
        # True when a rank's sampler ended and was started again inside a launch.
        "a_pass_ended_inside_a_launch": any(
            passes > 1 for launch in launches for passes in launch["passes_per_rank"]
        ),
        "read_orders": [{k: v for k, v in row.items() if k != "indices"} for row in order_rows],
    }


def consume(segment: Segment, built: Built, samplers: Samplers, log_path: Path) -> dict:
    """Run the independent consumer over a segment and write its log.

    For every sample of every completed step, Megatron's item getter is asked for
    the sample through a DataLoader, and one log line is written per call it makes
    to the indexed dataset: the shard, the document number, and the token range
    inside the document (the offset, and the offset plus the number of tokens that
    came back). ``ReadTap`` checks that the tokens the item getter returns are the
    concatenation of those reads. Returns a summary of the run."""
    check_launches_fit(segment, built)  # before the log file is made
    new_output_path(log_path)
    with open(log_path, "w", encoding="utf-8") as log:
        return _walk(segment, built, samplers, lambda row: log.write(json.dumps(row) + "\n"))


def consume_rows(segment: Segment, built: Built, samplers: Samplers) -> tuple[list[dict], dict]:
    """The consumer with its log kept in memory: the rows a log file would hold,
    in order, and the summary. The seed search reads its candidates this way and
    writes no log for them."""
    rows: list[dict] = []
    summary = _walk(segment, built, samplers, rows.append)
    return rows, summary
