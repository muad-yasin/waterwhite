"""The training stand-in: the run record and the step record of a segment.

A training wrapper writes a run record while the run's index caches exist (plan
3.3 "Training"). No model is trained here (plan 5.3, limit L4), so this module
stands in for the wrapper: it writes the record of docs/interfaces.md 8.1 from the
datasets Megatron built, the sampler classes that would run, and the launches the
config states. The step record of 8.2 is what an operator supplies at
registration; it is written from the config, which may lower one step's last
sample.

Standard library only; the objects it is handed come from ``megatron_data``.
"""

from __future__ import annotations

import os
from pathlib import Path

from oracle.config import Segment
from oracle.digests import file_sha256

RUN_RECORD_FORMAT = "waterwhite.run-record/1"


def _relative(path: Path, folder: Path) -> str:
    """A path relative to a folder, with forward slashes."""
    return Path(os.path.relpath(path, folder)).as_posix()


def run_record(
    segment: Segment,
    *,
    record_folder: Path,
    sampler_class: str,
    versions: dict,
    cache_rows: list[dict],
    shuffle_rows: list[dict],
) -> dict:
    """The run record of one segment.

    ``record_folder`` is the folder the record will be written to: a shard prefix
    is stored relative to it. ``cache_rows`` and ``shuffle_rows`` hold the digests
    (kind, shard, digest; launch, epoch, rank, digest, count). ``batch_schedule``
    is null: the global batch size is fixed inside a segment (plan F2)."""
    data = segment.data
    return {
        "format": RUN_RECORD_FORMAT,
        "run": segment.run,
        "segment": segment.segment,
        "start_kind": segment.start_kind,
        "parent_checkpoint": segment.parent_checkpoint,
        "settings": {
            "dataloader_type": segment.dataloader_type,
            "sampler_class": sampler_class,
            "data_sharding": segment.data_sharding,
            "micro_batch_size": segment.micro_batch_size,
            "global_batch_size": data.global_batch_size,
            "sequence_length": data.sequence_length,
            "add_extra_token_to_sequence": data.add_extra_token_to_sequence,
            "random_seed": data.random_seed,
            "split": data.split,
            "train_iters": data.train_iters,
            "train_samples": data.train_samples,
            "batch_schedule": None,
        },
        "versions": dict(versions),
        "shards": [
            {
                "position": position,
                "prefix": _relative(prefix, record_folder),
                "bin_sha256": file_sha256(str(prefix) + ".bin"),
                "idx_sha256": file_sha256(str(prefix) + ".idx"),
            }
            for position, prefix in enumerate(data.shards)
        ],
        "launches": [
            {
                "position": launch.position,
                "world_size": launch.world_size,
                "data_parallel_size": launch.data_parallel_size,
                "consumed_at_start": launch.consumed_at_start,
                "step_at_start": launch.step_at_start,
                "first_step": launch.first_step,
                "last_step": launch.last_step,
            }
            for launch in segment.launches
        ],
        "cache_digests": [
            {"kind": row["kind"], "shard": row["shard"], "digest": row["digest"]}
            for row in cache_rows
        ],
        "shuffle_digests": [
            {
                "launch": row["launch"],
                "epoch": row["epoch"],
                "rank": row["rank"],
                "digest": row["digest"],
                "count": row["count"],
            }
            for row in shuffle_rows
        ],
    }


def step_record(segment: Segment) -> dict:
    """The step record of one segment (docs/interfaces.md 8.2).

    Every completed step with its batch size, which is the global batch size. A
    step the config lowers also carries ``last_sample_tokens_read``; a step
    without it is not lowered. Steps are numbered as the trainer numbers them:
    each launch counts on from its ``step_at_start``."""
    lowered = dict(segment.lowered)
    steps = []
    for step in segment.steps():
        row = {"step": step, "batch_size": segment.data.global_batch_size}
        if step in lowered:
            row["last_sample_tokens_read"] = lowered[step]
        steps.append(row)
    return {
        "segment": segment.segment,
        "steps": steps,
        "checkpoints": [{"name": name, "step": step} for name, step in segment.checkpoints],
    }
