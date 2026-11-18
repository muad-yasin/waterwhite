"""Reading the run record and the step record (docs/interfaces.md 8.1, 8.2).

The run record is what the training wrapper wrote while the run's index caches
existed (plan 3.3 "Training", 3.1 "run segment"). The step record is what the
operator supplies when the run is registered (plan 3.1, F2, F4).

Two rules hold here.

- A field the wrapper could not record is absent or null. It is read as None and
  never replaced by a default: the join then names the cause the plan fixes for
  the missing field (plan 3.5).
- A field that is present with the wrong type is not a missing field. The record
  is then refused with "invalid record", as the store refuses it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .. import vocabulary as v
from ..errors import InvalidRecord

_CACHE_KINDS = v.CACHE_KINDS_PER_SHARD + v.CACHE_KINDS_PER_BLEND


@dataclass(frozen=True)
class Launch:
    """One start of the trainer inside a segment (plan 3.1, F4). The place of a
    launch in the data is its consumed-sample counter, never its step number."""

    position: int
    data_parallel_size: int
    consumed_at_start: int
    first_step: int
    last_step: int


@dataclass(frozen=True)
class ShardEntry:
    """One shard of the list the trainer was handed, in its order (Decision 27)."""

    position: int
    prefix: str | None
    idx_sha256: str | None


@dataclass(frozen=True)
class RunRecord:
    """The fields of a run record the join reads. None: not recorded."""

    segment: str | None
    release_tag: str | None
    sampler_type: str | None
    sampler_class: str | None
    data_sharding: bool | None
    micro_batch_size: int | None
    global_batch_size: int | None
    sequence_length: int | None
    add_extra_token: bool | None
    random_seed: int | None
    split: str | None
    train_iters: int | None
    train_samples: int | None
    batch_schedule: object
    shards: tuple[ShardEntry, ...]
    launches: tuple[Launch, ...]
    cache_digests: Mapping[tuple[str, int | None], str]
    shuffle_digests: Mapping[tuple[int, int, int], tuple[str, int | None]]

    @property
    def requested_samples(self) -> int | None:
        """The number of samples the trainer asked the dataset for: the recorded
        train_samples, else train_iters times the global batch size."""
        if self.train_samples is not None:
            return self.train_samples
        if self.train_iters is None or self.global_batch_size is None:
            return None
        return self.train_iters * self.global_batch_size


@dataclass(frozen=True)
class StepRecord:
    """The operator's step record. ``steps`` maps a step to its batch size and to
    the tokens read of its last sample, each None where the record is silent."""

    steps: Mapping[int, tuple[int | None, int | None]]
    checkpoints: tuple[tuple[str, int], ...]


def _need(condition: object, detail: str) -> None:
    if not condition:
        raise InvalidRecord(detail)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _opt(holder: Mapping, key: str, kind: type, what: str):
    value = holder.get(key)
    if value is None:
        return None
    ok = _is_int(value) if kind is int else isinstance(value, kind)
    _need(ok, f"run record: {what}.{key} should be {kind.__name__}, got {value!r}")
    return value


def _objects(document: Mapping, key: str) -> list:
    value = document.get(key)
    if value is None:
        return []
    _need(isinstance(value, list) and all(isinstance(item, Mapping) for item in value),
          f"run record: {key} is a list of objects")
    return value


def parse_run_record(document: Mapping) -> RunRecord:
    """Read a run record. Raises ``InvalidRecord`` for a document that is not a run
    record of the known format or that holds a field of the wrong type."""
    _need(isinstance(document, Mapping), "a run record is a JSON object")
    _need(document.get("format") == v.RUN_RECORD_FORMAT,
          f"run record: format {document.get('format')!r}, expected {v.RUN_RECORD_FORMAT!r}")
    settings = document.get("settings") or {}
    versions = document.get("versions") or {}
    _need(isinstance(settings, Mapping) and isinstance(versions, Mapping),
          "run record: settings and versions are objects")

    shards = []
    for item in _objects(document, "shards"):
        position = item.get("position")
        _need(_is_int(position) and position >= 0, "run record: a shard has a position")
        shards.append(ShardEntry(position, _opt(item, "prefix", str, "shards"),
                                 _opt(item, "idx_sha256", str, "shards")))
    shards.sort(key=lambda entry: entry.position)
    _need([entry.position for entry in shards] == list(range(len(shards))),
          "run record: the shard positions are 0, 1, 2 and so on, each once")

    launches = []
    for item in _objects(document, "launches"):
        keys = ("position", "data_parallel_size", "consumed_at_start", "first_step", "last_step")
        _need(all(_is_int(item.get(key)) for key in keys),
              f"run record: a launch needs the numbers {', '.join(keys)}; got {dict(item)!r}")
        launch = Launch(*(item[key] for key in keys))
        _need(launch.consumed_at_start >= 0 and launch.first_step <= launch.last_step,
              f"run record: launch {launch.position} has a negative counter or no step")
        launches.append(launch)
    launches.sort(key=lambda launch: launch.position)
    _need(len({launch.position for launch in launches}) == len(launches),
          "run record: two launches share a position")
    taken: set[int] = set()
    for launch in launches:
        own = set(range(launch.first_step, launch.last_step + 1))
        _need(not (own & taken), f"run record: launch {launch.position} repeats a step of "
                                 "another launch")
        taken |= own

    cache: dict[tuple[str, int | None], str] = {}
    for item in _objects(document, "cache_digests"):
        kind, shard, digest = item.get("kind"), item.get("shard"), item.get("digest")
        _need(kind in _CACHE_KINDS, f"run record: unknown cache kind {kind!r}")
        _need(isinstance(digest, str) and digest, f"run record: the digest of {kind} is text")
        if kind in v.CACHE_KINDS_PER_SHARD:
            _need(_is_int(shard), f"run record: a {kind} digest names its shard position")
        else:
            _need(shard is None, f"run record: a {kind} digest belongs to the blend")
        _need((kind, shard) not in cache, f"run record: two digests for {kind} of shard {shard!r}")
        cache[(kind, shard)] = digest

    shuffle: dict[tuple[int, int, int], tuple[str, int | None]] = {}
    for item in _objects(document, "shuffle_digests"):
        key = (item.get("launch"), item.get("epoch"), item.get("rank"))
        _need(all(_is_int(part) for part in key) and isinstance(item.get("digest"), str),
              f"run record: a shuffle digest needs launch, epoch, rank and digest; got {dict(item)!r}")
        _need(key not in shuffle, f"run record: two shuffle digests for launch, epoch, rank {key}")
        shuffle[key] = (item["digest"], _opt(item, "count", int, "shuffle_digests"))

    return RunRecord(
        segment=_opt(document, "segment", str, "the record"),
        release_tag=_opt(versions, "release_tag", str, "versions"),
        sampler_type=_opt(settings, "dataloader_type", str, "settings"),
        sampler_class=_opt(settings, "sampler_class", str, "settings"),
        data_sharding=_opt(settings, "data_sharding", bool, "settings"),
        micro_batch_size=_opt(settings, "micro_batch_size", int, "settings"),
        global_batch_size=_opt(settings, "global_batch_size", int, "settings"),
        sequence_length=_opt(settings, "sequence_length", int, "settings"),
        add_extra_token=_opt(settings, "add_extra_token_to_sequence", bool, "settings"),
        random_seed=_opt(settings, "random_seed", int, "settings"),
        split=_opt(settings, "split", str, "settings"),
        train_iters=_opt(settings, "train_iters", int, "settings"),
        train_samples=_opt(settings, "train_samples", int, "settings"),
        batch_schedule=settings.get("batch_schedule"),
        shards=tuple(shards),
        launches=tuple(launches),
        cache_digests=cache,
        shuffle_digests=shuffle,
    )


def parse_step_record(document: Mapping | None, segment: str | None) -> StepRecord:
    """Read a step record. None is a run registered with no step record. A record
    written for another segment is refused: its steps would lower the wrong run."""
    if document is None:
        return StepRecord({}, ())
    _need(isinstance(document, Mapping), "a step record is a JSON object")
    named = document.get("segment")
    _need(named is None or segment is None or named == segment,
          f"step record: it is written for segment {named!r}, the run record is of {segment!r}")
    steps: dict[int, tuple[int | None, int | None]] = {}
    for item in document.get("steps") or []:
        _need(isinstance(item, Mapping) and _is_int(item.get("step")) and item["step"] >= 0,
              f"step record: a step names its number; got {item!r}")
        batch, last = item.get("batch_size"), item.get("last_sample_tokens_read")
        _need(batch is None or (_is_int(batch) and batch >= 0),
              f"step record: the batch size of step {item['step']} is a number")
        _need(last is None or (_is_int(last) and last >= 0),
              f"step record: the tokens read of the last sample of step {item['step']} is a number")
        _need(item["step"] not in steps, f"step record: step {item['step']} is given twice")
        steps[item["step"]] = (batch, last)
    checkpoints = []
    for item in document.get("checkpoints") or []:
        _need(isinstance(item, Mapping) and isinstance(item.get("name"), str)
              and _is_int(item.get("step")),
              f"step record: a checkpoint has a name and a step; got {item!r}")
        checkpoints.append((item["name"], item["step"]))
    return StepRecord(steps, tuple(checkpoints))
