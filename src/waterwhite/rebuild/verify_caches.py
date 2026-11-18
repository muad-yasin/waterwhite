"""Stale index caches (plan 4.4, 3.5; docs/interfaces.md section 7.3 "caches",
docs/rebuild.md section 5 "Stale caches").

Megatron's cache key holds the paths of the shards and not their content, so a
trainer that finds index caches under the old paths reads them even when the shards
behind them changed. The tool compares the mixture digest of the snapshot a segment
was registered with against the digest of the mixture the corrected run reads. The
mixture digest is ``ids.mixture_content_digest`` of the content digests of the
shards of the snapshot, in the trainer's order.

For a stale segment the corrected data order is derived again from the run record
and the document lengths of the successor shards. No cache file is read.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from .. import ids, join
from .. import vocabulary as v
from ..errors import IntegrityFault, WaterwhiteError
from ..store import artifact_files
from . import verify_sides


def mixture_digest(store, instance_id: str) -> str:
    """The mixture digest of the snapshot a mixture was registered with."""
    return ids.mixture_content_digest(
        [entry.content_digest for entry in store.mixture_shards(instance_id)])


def shard_lengths(store, mixture_instance_id: str) -> list:
    """The document lengths of each shard of a mixture's snapshot, in the trainer's
    order, read from the shards' index files where the store has them. Raises
    ``LookupError`` with the reason where a shard cannot be read."""
    lengths = []
    for entry in store.mixture_shards(mixture_instance_id):
        prefix = store.disk_path(entry.shard_instance_id)
        if prefix is None:
            raise LookupError(f"the shard {entry.path} is not registered")
        idx_path = artifact_files.shard_paths(prefix)[1]
        if not idx_path.is_file():
            raise LookupError(f"the index file of the shard {entry.path} is not at its registered place")
        lengths.append(artifact_files.read_shard_index(idx_path).sequence_lengths)
    return lengths


def corrected_run_record(store, segment, mixture_instance_id: str):
    """The segment's run record with the corrected mixture's shards in its shard
    list, and the index arrays derived from the document lengths of those shards,
    with the digests of what was derived in place of the recorded ones. Raises
    ``LookupError`` where the corrected order cannot be derived, with the reason."""
    if segment.run_record is None:
        raise LookupError("the segment has no run record")
    record = copy.deepcopy(segment.run_record)
    parsed = join.parse_run_record(record)
    lengths = shard_lengths(store, mixture_instance_id)
    try:
        arrays = join.replay_arrays(parsed, lengths)
    except WaterwhiteError as error:
        raise LookupError(error.detail or error.name) from None
    shards = []
    for position, entry in enumerate(store.mixture_shards(mixture_instance_id)):
        prefix = store.disk_path(entry.shard_instance_id)
        idx_path = artifact_files.shard_paths(prefix)[1]
        shards.append({"position": position, "prefix": str(prefix), "idx_sha256": ids.file_digest(idx_path)})
    record["shards"] = shards
    record["cache_digests"] = [
        {"kind": kind, "shard": shard, "digest": join.digest_of(array)}
        for (kind, shard), array in arrays.items()]
    return record


def corrected_traces(store, segment, corrected_instance: str):
    """The join of the segment's corrected order: the traces a run on the corrected
    mixture reads, built from an empty cache folder. Raises ``LookupError`` with the
    reason where the order cannot be built, among them a release or a sampler the join
    is not verified for."""
    record = corrected_run_record(store, segment, corrected_instance)
    step_record = {"steps": [
        {"step": item.step, "batch_size": item.batch_size,
         "last_sample_tokens_read": item.last_sample_tokens_read} for item in segment.steps]}
    result = join.compute_join(
        record, verified_joins=store.verified_joins(), step_record=step_record)
    if result.route == v.ROUTE_NONE:
        raise LookupError(result.explanation or "the join gives the corrected run no trace")
    return result


def _try_rebuild(store, segment, corrected_instance: str) -> tuple[bool, str | None]:
    """Whether the corrected order of a segment can be built again, and if not, why.
    A failure of the build is a reason, never an error; a loud fault of the store is
    not a reason and stops the command."""
    try:
        corrected_traces(store, segment, corrected_instance)
    except IntegrityFault:
        raise
    except LookupError as error:
        return False, str(error)
    except Exception as error:  # a record or file the build cannot read is a reason
        return False, f"{type(error).__name__}: {error}"
    return True, None


def _mixture_entries(store, report: Mapping) -> dict[str, dict]:
    """The report's mixture entries by the instance id of the mixture they name, the
    report's own instance id and the one the store holds for the name alike."""
    found = {}
    for entry in verify_sides.report_artifacts(report):
        if entry["kind"] != v.KIND_MIXTURE:
            continue
        found[entry["instance"]] = entry
        held = store.resolve(entry["artifact"])
        if held is not None:
            found.setdefault(held.instance_id, entry)
    return found


def check_caches(store, report: Mapping) -> list[dict]:
    """The ``caches`` entries: one per run segment of every run that has a segment
    on a mixture the report rewrites, in the order of the runs and their segments."""
    mixtures = _mixture_entries(store, report)
    entries = []
    for run in store.runs():
        segments = [store.segment(name) for name in store.segments(run)]
        segments = [s for s in segments if s is not None and s.mixture_instance_id is not None]
        if not any(mixtures.get(s.mixture_instance_id, {}).get("action") == v.ACTION_REWRITE
                   for s in segments):
            continue
        for segment in segments:
            entries.append(_entry(store, run, segment, mixtures.get(segment.mixture_instance_id)))
    return entries


def _entry(store, run: str, segment, mixture: dict | None) -> dict:
    held = store.artifact(segment.mixture_instance_id)
    name = mixture["artifact"] if mixture else (held.names[0] if held and held.names else segment.mixture_instance_id)
    digest = mixture_digest(store, segment.mixture_instance_id)
    entry = {
        "run": run, "segment": segment.name, "mixture": name,
        "corrected_mixture": None, "digest": digest, "corrected_digest": None,
        "stale": None, "rebuilt": False, "rebuilt_from": None, "reason": None,
    }
    action = None if mixture is None else mixture["action"]
    if action in (None, v.ACTION_KEEP):
        # The mixture the corrected run reads is this one: nothing changed under it.
        entry.update(corrected_mixture=name, corrected_digest=digest, stale=False)
        return entry
    if action == v.ACTION_CLEAN_PLACE_NOT_STATED:
        # Read as a mixture the report leaves unresolved: whether a shard of it receives a
        # copy is not stated, so no corrected mixture is named (D-146).
        entry["reason"] = ("the report does not state where the records that return land, "
                           "so no corrected mixture is named")
        return entry
    if action != v.ACTION_REWRITE:
        entry["reason"] = "the report leaves the mixture unresolved, so no corrected mixture is named"
        return entry
    successor_name = mixture.get("successor") or f"{v.SUCCESSOR_PREFIX}{name}"
    successor = store.resolve(successor_name)
    if successor is None:
        entry["reason"] = f"the report rewrites the mixture and no mixture is registered as {successor_name}"
        return entry
    corrected = mixture_digest(store, successor.instance_id)
    entry.update(corrected_mixture=successor_name, corrected_digest=corrected, stale=digest != corrected)
    if entry["stale"]:
        rebuilt, reason = _try_rebuild(store, segment, successor.instance_id)
        entry.update(rebuilt=rebuilt, reason=reason,
                     rebuilt_from=v.REBUILT_FROM_EMPTY_CACHE if rebuilt else None)
    return entry


__all__ = ["check_caches", "corrected_run_record", "corrected_traces", "mixture_digest"]
