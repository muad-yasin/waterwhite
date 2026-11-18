"""The checkpoint exposure join (plan 3.5; Checks 1, 2 and 8 of plan 5.5).

For a run segment the join says which (shard, document ordinal, token range)
rows each completed training step read. It works from the run record the
training wrapper wrote, the operator's step record and the shards' index files.
It runs no trainer and imports neither the trainer's framework nor its tensor
library, except that a cyclic run's shuffle is computed by calling torch, which
is imported then and only then (docs/join.md section 6). Otherwise numpy is all
it needs.

    from waterwhite.join import SegmentRegistration, join_segment

    result = join_segment(
        store,
        SegmentRegistration(run="E", segment="E/seg-1", position=0, start_kind="fresh",
                            mixture_instance_id=mixture),
        "runs/E/seg-1/run_record.json",
        cache_dir="runs/E/seg-1/cache",
        step_record="runs/E/seg-1/step_record.json",
    )
    result.route          # "cache files", "replay" or "none"
    result.cause          # with "none": the named cause the segment is registered with
    result.steps_written  # how many steps got a trace

Where the data order cannot be trusted the join writes no trace: the segment is
registered with the route "none" and the cause plan 3.5 fixes. docs/join.md
describes the routes, the causes and their order, and what is not built. The
command `waterwhite register-run` calls it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from ..errors import InvalidRecord, UsageError
from ._arrays import digest_of, needed_keys, read_cache_files
from ._epochs import step_epochs
from ._join import JoinResult, ReplayReport, ShuffleCheck, compute_join
from ._record import RunRecord, parse_run_record
from ._replay import missing_replay_settings, replay_arrays
from ._shuffle import epoch_permutation, rank_order
from ._store_adapter import SegmentRegistration, register_segment
from ._write import write_replay

# The package's surface. The modules behind it (``_sampler``, ``_rows``,
# ``_arrays``, ``_replay``, ``_record``) are its own parts. ``_sampler.CannotPlace``
# is a signal between them and never leaves ``compute_join``.
__all__ = [
    "JoinResult",
    "ReplayReport",
    "RunRecord",
    "SegmentRegistration",
    "ShuffleCheck",
    "compute_join",
    "digest_of",
    "epoch_permutation",
    "join_segment",
    "missing_replay_settings",
    "needed_keys",
    "parse_run_record",
    "rank_order",
    "read_cache_files",
    "read_json",
    "register_segment",
    "step_epochs",
    "replay_arrays",
    "write_replay",
]


def read_json(path: str | os.PathLike, what: str) -> dict:
    """Read a run record or a step record file. A path that cannot be read is a
    usage error (exit 1) and one that holds no JSON object is an invalid record:
    a file that was named is a mistake of the registration, not a run with
    nothing recorded."""
    try:
        with open(path, encoding="utf-8") as f:
            document = json.load(f)
    except OSError as error:
        raise UsageError(f"the {what} cannot be read: {error.strerror}", path=str(path)) from None
    except ValueError as error:
        raise InvalidRecord(f"the {what} is not JSON: {error}", path=str(path)) from None
    if not isinstance(document, dict):
        raise InvalidRecord(f"the {what} is not a JSON object", path=str(path))
    return document


def join_segment(
    store,
    segment: SegmentRegistration,
    run_record_path: str | os.PathLike | None,
    cache_dir: str | os.PathLike | None = None,
    step_record: Mapping | str | os.PathLike | None = None,
) -> JoinResult:
    """Join one run segment and register it in the store with what the join found.

    ``run_record_path`` is the run record file (docs/interfaces.md 8.1), or None
    for a run that has none. ``cache_dir`` is the kept data-order cache folder
    (8.4). ``step_record`` is the operator's step record (8.2), as an object or as
    the path of its file.

    The segment is registered, in one store call and one transaction, with the
    join's trace route, cause, completed steps and traces, and the checkpoints of
    the step record (plan 3.5, Decision 19). The result says which route was
    taken, the cause when there is none, and how many steps got a trace.

    Raises ``InvalidRecord`` for a file that is not a run record or a step record,
    ``UsageError`` for a path that cannot be read, ``TorchUnavailable`` for a
    cyclic run whose shuffle must be computed here and torch cannot be imported
    (nothing is written then), and whatever the store raises for a registration it
    refuses."""
    document = None if run_record_path is None else read_json(run_record_path, "run record")
    if step_record is not None and not isinstance(step_record, Mapping):
        step_record = read_json(step_record, "step record")
    result = compute_join(
        document,
        verified_joins=store.verified_joins(),
        record_dir=None if run_record_path is None else Path(run_record_path).parent,
        cache_dir=cache_dir,
        step_record=step_record,
    )
    register_segment(store, segment, document, result)
    return result
