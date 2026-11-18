"""What the tools do, as functions: build and digest an index, write a segment's
records, run a segment whole, run several segments.

The entry scripts are thin wrappers around these. ``read_segment`` is the same walk
with nothing written but the cache folder: the seed search calls it for each
candidate, in one process.
"""

from __future__ import annotations

import os

from oracle import environment, records, stepping
from oracle.config import (
    DataOrder,
    Segment,
    check_starts,
    new_output_path,
    write_json,
)
from oracle.digests import file_sha256
from oracle.errors import ConfigError, OracleError
from oracle.megatron_data import (
    Built,
    build_train_dataset,
    cache_digests,
    load_samplers,
    prepare_cache_dir,
)

INDEX_FORMAT = "waterwhite.oracle-index-digests/1"

SEGMENT_OUTPUTS = ("digests", "run_record", "step_record", "log", "summary")


# --- Index build and replay --------------------------------------------------------------


def build_index(data: DataOrder) -> tuple[Built, dict]:
    """Build the datasets of a segment into its new, empty cache folder and digest
    the index arrays. Returns the built datasets and the digests document."""
    built = build_train_dataset(data)
    rows = cache_digests(built)
    document = {
        "format": INDEX_FORMAT,
        "settings": {
            "sequence_length": data.sequence_length,
            "add_extra_token_to_sequence": data.add_extra_token_to_sequence,
            "random_seed": data.random_seed,
            "split": data.split,
            "requested_samples": data.requested_samples,
        },
        "blend": built.blend is not None,
        "dataset_samples": built.samples,
        "shards": [
            {
                "position": position,
                "bin_sha256": file_sha256(str(prefix) + ".bin"),
                "idx_sha256": file_sha256(str(prefix) + ".idx"),
                "documents": len(dataset.dataset),
                "samples": len(dataset),
            }
            for position, (prefix, dataset) in enumerate(zip(data.shards, built.shards))
        ],
        "cache_digests": rows,
    }
    return built, document


# --- One segment ---------------------------------------------------------------------------


def _check_outputs_free(segment: Segment, keys: tuple[str, ...]) -> None:
    for key in keys:
        new_output_path(segment.output(key))
    paths = [str(segment.output(key)) for key in keys]
    if len(set(paths)) != len(paths):
        raise ConfigError(f"segment {segment.segment!r}: two outputs share one path")


def segment_records(segment: Segment, built: Built, cache_rows: list[dict]) -> tuple[dict, list[dict]]:
    """The run record of a segment, and the orders the sampler classes gave.

    The record is made for the folder ``outputs.run_record`` lies in."""
    samplers = load_samplers(segment.sampler_file)
    orders = stepping.drive_orders(segment, built, samplers)
    record = records.run_record(
        segment,
        record_folder=segment.output("run_record").parent,
        sampler_class=samplers.class_name(segment.dataloader_type),
        versions=environment.versions(samplers.bridge_file),
        cache_rows=cache_rows,
        shuffle_rows=stepping.shuffle_digests(segment, built, samplers, orders),
    )
    return record, orders


def consume_segment(segment: Segment, built: Built) -> dict:
    """Run the consumer and write the log at ``outputs.log``.

    The log is written under the name ``<log>.partial`` and takes its own name
    only when the whole segment was read, so a run that stops half way leaves no
    file that looks like a log."""
    log_path = segment.output("log")
    new_output_path(log_path)
    partial = log_path.with_name(log_path.name + ".partial")
    summary = stepping.consume(segment, built, load_samplers(segment.sampler_file), partial)
    os.replace(partial, log_path)
    return summary


def run_segment(segment: Segment) -> dict:
    """Everything for one segment from one build: the index into the segment's own
    new cache folder, then the digests document, the run record, the step record,
    the oracle log and the summary.

    The consumer's reads are compared with the orders the sampler classes gave
    for the run record; a difference stops the run before a record is written."""
    _check_outputs_free(segment, SEGMENT_OUTPUTS)
    prepare_cache_dir(segment.data.cache_dir)
    built, document = build_index(segment.data)
    record, orders = segment_records(segment, built, document["cache_digests"])
    summary = consume_segment(segment, built)
    driven = [{k: v for k, v in row.items() if k != "indices"} for row in orders]
    if driven != summary["read_orders"]:
        raise OracleError(
            f"segment {segment.segment!r}: the consumer read other indices than the sampler "
            "classes gave when driven alone"
        )
    summary["cache_digests"] = record["cache_digests"]
    summary["shuffle_digests"] = record["shuffle_digests"]
    write_json(segment.output("digests"), document)
    write_json(segment.output("run_record"), record)
    write_json(segment.output("step_record"), records.step_record(segment))
    write_json(segment.output("summary"), summary)
    return summary


def read_segment(segment: Segment) -> dict:
    """One segment read whole, with its log kept in memory.

    The index is built into the segment's new, empty cache folder, as always; the
    sampler classes are driven alone and through the consumer, and the two are
    compared, as ``run_segment`` does. No output file is written and the config
    needs no ``outputs``. Returns the log rows, the summary, the order digests a
    run record would hold, and the number of documents and samples Megatron
    reports for each shard."""
    built = build_train_dataset(segment.data)
    samplers = load_samplers(segment.sampler_file)
    orders = stepping.drive_orders(segment, built, samplers)
    rows, summary = stepping.consume_rows(segment, built, samplers)
    driven = [{k: v for k, v in row.items() if k != "indices"} for row in orders]
    if driven != summary["read_orders"]:
        raise OracleError(
            f"segment {segment.segment!r}: the consumer read other indices than the sampler "
            "classes gave when driven alone"
        )
    return {
        "rows": rows,
        "summary": summary,
        "shuffle_digests": stepping.shuffle_digests(segment, built, samplers, orders),
        "shards": [
            {"position": position, "documents": len(dataset.dataset), "samples": len(dataset)}
            for position, dataset in enumerate(built.shards)
        ],
    }


# --- Several segments ------------------------------------------------------------------------


def run_segments(segments: list[Segment]) -> dict:
    """Run several segments, each with its own new cache folder.

    Every check that needs no build is made for all segments first, so a wrong
    path in the last segment stops the run before the first one is built."""
    starts = check_starts(segments)
    caches = [str(segment.data.cache_dir.resolve()) for segment in segments]
    if len(set(caches)) != len(caches):
        raise ConfigError("two segments share one cache_dir; each needs its own new folder")
    outputs = [str(segment.output(key)) for segment in segments for key in SEGMENT_OUTPUTS]
    if len(set(outputs)) != len(outputs):
        raise ConfigError("two segments share an output path")
    for segment in segments:
        _check_outputs_free(segment, SEGMENT_OUTPUTS)
        prepare_cache_dir(segment.data.cache_dir)
    results = []
    for segment in segments:
        summary = run_segment(segment)
        results.append(
            {
                "segment": segment.segment,
                "start": starts[segment.segment],
                "dataset_samples": summary["dataset_samples"],
                "first_step": segment.launches[0].first_step,
                "last_step": segment.launches[-1].last_step,
                "samples_read": summary["samples_read"],
                "rows_written": summary["rows_written"],
                "a_pass_ended_inside_a_launch": summary["a_pass_ended_inside_a_launch"],
            }
        )
    return {"segments": results}
