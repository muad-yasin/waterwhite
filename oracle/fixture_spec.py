"""Reading the fixture spec (plan 5.1) for the oracle's fixture runs and seed search.

The spec is one TOML file written by hand. This module reads the parts the oracle
needs: the runs with their segments, launches and step records, the settings all
runs share, the mixtures, the named records, and the numbered conditions of the
seed search. It checks that these parts agree with each other and refuses the
file otherwise; it fills in nothing the spec does not state, except the two
rules written at ``_launches``.

A shard is named in the spec as a segment lists it: ``shard-1`` is a shard of the
partitioned layout, ``E-rr/shard-1`` is shard-1 of the round-robin layout.

Standard library only.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from oracle.errors import ConfigError

SPEC_FORMAT = "waterwhite.fixture-spec/1"

LAYOUT_PARTITIONED = "partitioned"
LAYOUT_ROUND_ROBIN = "round-robin"
# How a segment names a shard of the round-robin layout: "<run>-rr/<shard>".
_ROUND_ROBIN_MARK = "-rr/"

_RUN_SETTINGS = (
    "sequence_length",
    "global_batch_size",
    "micro_batch_size",
    "data_parallel_size",
    "world_size",
    "dataloader_type",
    "data_sharding",
    "random_seed",
    "steps",
)
_RUN_KEYS = _RUN_SETTINGS + ("checkpoints", "segments", "step_record", "same_settings_as", "layout")
_SEGMENT_KEYS = (
    "steps",
    "shards",
    "start_kind",
    "parent_checkpoint",
    "mixture",
    "train_iters",
    "consumed_at_start",
    "launches",
    "resume_from",
)
_LAUNCH_KEYS = (
    "position",
    "world_size",
    "data_parallel_size",
    "consumed_at_start",
    "step_at_start",
    "first_step",
    "last_step",
)
_COMMON_KEYS = (
    "add_extra_token_to_sequence",
    "split",
    "megatron_bridge",
    "sampler_class_single",
    "sampler_class_cyclic",
    "caches_kept",
)


@dataclass(frozen=True)
class ShardRef:
    """One shard as a segment names it."""

    name: str  # the name in the spec: "shard-1", "E-rr/shard-1"
    layout: str  # "partitioned" or "round-robin"
    file: str  # the shard's file name inside its layout: "shard-1"


@dataclass(frozen=True)
class LaunchPlan:
    world_size: int
    data_parallel_size: int
    consumed_at_start: int
    step_at_start: int
    steps: int


@dataclass(frozen=True)
class SegmentPlan:
    run: str
    name: str  # "E/seg-2"
    folder: str  # the name without its run: "seg-2"
    first_step: int
    last_step: int
    shards: tuple[ShardRef, ...]
    start_kind: str
    parent_checkpoint: str | None
    mixture: str
    train_iters: int
    launches: tuple[LaunchPlan, ...]
    lowered: tuple[tuple[int, int], ...]  # (step, tokens read of its last sample)
    checkpoints: tuple[tuple[str, int], ...]  # (name, step)


@dataclass(frozen=True)
class RunPlan:
    name: str
    layout: str
    seed_of: str  # the run whose random_seed this run uses: itself, or the run it copies
    random_seed: int  # as the spec states it
    sequence_length: int
    global_batch_size: int
    micro_batch_size: int
    data_parallel_size: int
    world_size: int
    dataloader_type: str
    data_sharding: bool
    steps: int
    segments: tuple[SegmentPlan, ...]


@dataclass(frozen=True)
class Condition:
    n: int
    id: str
    sentence: str


@dataclass(frozen=True)
class FixtureSpec:
    path: Path
    raw: dict
    runs: dict[str, RunPlan]  # in the order of the file
    add_extra_token_to_sequence: bool
    split: str
    sampler_classes: dict[str, str]  # dataloader type -> the class the spec names
    handles: dict[str, str]  # handle -> record id
    sources: dict[str, tuple[str, ...]]  # source -> its record ids
    conditions: tuple[Condition, ...]

    def source_of(self, record: str) -> str | None:
        """The source a record id belongs to, or None for a generation output."""
        for source, records in self.sources.items():
            if record in records:
                return source
        return None

    def derived_from(self, source: str, base: Path) -> frozenset[str]:
        """The record ids a withdrawal of ``source`` puts in scope: the source's own
        records and every generation output made from one of them, through any
        number of jobs. The generation file is the one the spec names, read under
        ``base`` (the repository root); a job's parents are the records its row
        says it used."""
        if source not in self.sources:
            raise ConfigError(f"the spec has no source {source!r}")
        generation = _table(self.raw, "generation", "the spec")
        file = base / _string(generation, "file", "generation")
        rows = []
        try:
            with open(file, encoding="utf-8") as f:
                for number, line in enumerate(f, start=1):
                    row = json.loads(line)
                    if not isinstance(row.get("id"), str) or not isinstance(row.get("used"), list):
                        raise ConfigError(f"{file.name} line {number}: no 'id' and 'used'")
                    rows.append(row)
        except FileNotFoundError as e:
            raise ConfigError(f"the spec's generation file is not under the base folder: {file}") from e
        except json.JSONDecodeError as e:
            raise ConfigError(f"{file.name}: not JSON lines: {e}") from e
        scope = set(self.sources[source])
        grew = True
        while grew:
            grew = False
            for row in rows:
                if row["id"] not in scope and scope.intersection(row["used"]):
                    scope.add(row["id"])
                    grew = True
        return frozenset(scope)


# --- Small readers ------------------------------------------------------------------------------


def _table(d: dict, key: str, where: str) -> dict:
    value = d.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: missing table {key!r}")
    return value


def _string(d: dict, key: str, where: str) -> str:
    value = d.get(key)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}: {key!r} must be a non-empty string, got {value!r}")
    return value


def _integer(d: dict, key: str, where: str, *, minimum: int = 0) -> int:
    value = d.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{where}: {key!r} must be an integer of at least {minimum}, got {value!r}")
    return value


def _only(d: dict, allowed: tuple[str, ...], where: str) -> None:
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; known: {list(allowed)}")


def expand_ids(items: list, where: str) -> tuple[str, ...]:
    """Expand the spec's id ranges: ``"Z-001..Z-020"`` is every id from the first
    to the last, the two written with one prefix and one width."""
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise ConfigError(f"{where}: a record id must be a string, got {item!r}")
        if ".." not in item:
            out.append(item)
            continue
        first, _, last = item.partition("..")
        head, _, low = first.rpartition("-")
        head_last, _, high = last.rpartition("-")
        if head != head_last or not low.isdigit() or not high.isdigit() or len(low) != len(high):
            raise ConfigError(f"{where}: cannot read the id range {item!r}")
        if int(low) > int(high):
            raise ConfigError(f"{where}: the id range {item!r} runs backwards")
        out.extend(f"{head}-{n:0{len(low)}d}" for n in range(int(low), int(high) + 1))
    return tuple(out)


def shard_ref(name: str, where: str) -> ShardRef:
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}: a shard name must be a string, got {name!r}")
    if _ROUND_ROBIN_MARK in name:
        file = name.split(_ROUND_ROBIN_MARK, 1)[1]
        layout = LAYOUT_ROUND_ROBIN
    else:
        file, layout = name, LAYOUT_PARTITIONED
    if not file or "/" in file or file in (".", ".."):
        raise ConfigError(f"{where}: cannot read the shard name {name!r}")
    return ShardRef(name=name, layout=layout, file=file)


# --- Runs ---------------------------------------------------------------------------------------


def _launches(raw: dict, settings: dict, first: int, last: int, where: str) -> tuple[LaunchPlan, ...]:
    """The launches of a segment.

    A segment that lists none has one launch over all its steps, at the
    consumed count the segment states, with the run's data-parallel size; its
    world size is the run's ``world_size`` when stated, else the data-parallel
    size. A listed launch must state its counters, and its steps follow from
    ``step_at_start`` and ``last_step``."""
    consumed = _integer(raw, "consumed_at_start", where)
    if "launches" not in raw:
        return (
            LaunchPlan(
                world_size=settings["world_size"],
                data_parallel_size=settings["data_parallel_size"],
                consumed_at_start=consumed,
                step_at_start=first - 1,
                steps=last - first + 1,
            ),
        )
    items = raw["launches"]
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{where}: 'launches' must be a non-empty list")
    launches = []
    expected_first = first
    for i, item in enumerate(items):
        at = f"{where}, launches[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{at}: must be a table")
        _only(item, _LAUNCH_KEYS, at)
        if _integer(item, "position", at) != i:
            raise ConfigError(f"{at}: its position must be {i}")
        step_at_start = _integer(item, "step_at_start", at)
        if _integer(item, "first_step", at, minimum=1) != step_at_start + 1:
            raise ConfigError(f"{at}: first_step must be step_at_start + 1")
        last_step = _integer(item, "last_step", at, minimum=1)
        if step_at_start + 1 != expected_first or last_step < expected_first:
            raise ConfigError(
                f"{at}: its steps must start at step {expected_first}, where the launch before "
                "it (or the segment) left off"
            )
        launches.append(
            LaunchPlan(
                world_size=_integer(item, "world_size", at, minimum=1),
                data_parallel_size=_integer(item, "data_parallel_size", at, minimum=1),
                consumed_at_start=_integer(item, "consumed_at_start", at),
                step_at_start=step_at_start,
                steps=last_step - step_at_start,
            )
        )
        expected_first = last_step + 1
    if expected_first != last + 1:
        raise ConfigError(f"{where}: its launches end at step {expected_first - 1}, not at {last}")
    if launches[0].consumed_at_start != consumed:
        raise ConfigError(
            f"{where}: its first launch starts at consumed count "
            f"{launches[0].consumed_at_start}, the segment states {consumed}"
        )
    return tuple(launches)


def _step_record(raw: dict, settings: dict, first: int, last: int, where: str):
    """A segment's step record table: the steps it lowers, and its checkpoints."""
    _only(raw, ("steps", "checkpoints"), where)
    lowered, checkpoints = [], []
    for i, item in enumerate(raw.get("steps", [])):
        at = f"{where}, steps[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{at}: must be a table")
        _only(item, ("step", "batch_size", "last_sample_tokens_read"), at)
        step = _integer(item, "step", at, minimum=1)
        if not first <= step <= last:
            raise ConfigError(f"{at}: step {step} is not a step of the segment")
        if _integer(item, "batch_size", at, minimum=1) != settings["global_batch_size"]:
            raise ConfigError(
                f"{at}: batch_size is not the run's global batch size "
                f"{settings['global_batch_size']}; the oracle reads a fixed batch size only"
            )
        if "last_sample_tokens_read" in item:
            lowered.append((step, _integer(item, "last_sample_tokens_read", at)))
    for i, item in enumerate(raw.get("checkpoints", [])):
        at = f"{where}, checkpoints[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{at}: must be a table")
        _only(item, ("name", "step"), at)
        step = _integer(item, "step", at, minimum=1)
        if not first <= step <= last:
            raise ConfigError(f"{at}: step {step} is not a step of the segment")
        checkpoints.append((_string(item, "name", at), step))
    return tuple(lowered), tuple(checkpoints)


def _run(name: str, raw: dict, all_runs: dict, mixtures: dict) -> RunPlan:
    where = f"runs.{name}"
    _only(raw, _RUN_KEYS, where)
    seed_of = name
    source = raw
    if "same_settings_as" in raw:
        seed_of = _string(raw, "same_settings_as", where)
        if seed_of not in all_runs or "same_settings_as" in all_runs[seed_of]:
            raise ConfigError(f"{where}: same_settings_as names no run with settings of its own")
        stated = sorted(set(raw).intersection(_RUN_SETTINGS))
        if stated:
            raise ConfigError(f"{where}: it copies the settings of {seed_of!r} and also states {stated}")
        source = all_runs[seed_of]
    at = f"runs.{seed_of}"
    settings = {
        key: _integer(source, key, at, minimum=1)
        for key in ("sequence_length", "global_batch_size", "micro_batch_size", "data_parallel_size", "steps")
    }
    settings["random_seed"] = _integer(source, "random_seed", at)
    settings["world_size"] = (
        _integer(source, "world_size", at, minimum=1) if "world_size" in source else settings["data_parallel_size"]
    )
    settings["dataloader_type"] = _string(source, "dataloader_type", at)
    sharding = source.get("data_sharding", True)  # Megatron-LM's default
    if not isinstance(sharding, bool):
        raise ConfigError(f"{at}: data_sharding must be true or false")
    layout = raw.get("layout", LAYOUT_PARTITIONED)
    if layout not in (LAYOUT_PARTITIONED, LAYOUT_ROUND_ROBIN):
        raise ConfigError(f"{where}: unknown layout {layout!r}")

    step_records = raw.get("step_record", {})
    segments_raw = _table(raw, "segments", where)
    unknown = sorted(set(step_records) - set(segments_raw))
    if unknown:
        raise ConfigError(f"{where}: step_record names segment(s) the run does not have: {unknown}")
    segments = []
    expected_first = 1
    for segment_name, segment_raw in segments_raw.items():
        seg_where = f'{where}.segments."{segment_name}"'
        if not isinstance(segment_raw, dict):
            raise ConfigError(f"{seg_where}: must be a table")
        _only(segment_raw, _SEGMENT_KEYS, seg_where)
        if not segment_name.startswith(name + "/"):
            raise ConfigError(f"{seg_where}: a segment's name starts with its run and a slash")
        folder = segment_name[len(name) + 1 :]
        if not folder or "/" in folder or folder in (".", ".."):
            raise ConfigError(f"{seg_where}: cannot make a folder name of {segment_name!r}")
        steps = segment_raw.get("steps")
        if (
            not isinstance(steps, list)
            or len(steps) != 2
            or not all(isinstance(s, int) and not isinstance(s, bool) for s in steps)
        ):
            raise ConfigError(f"{seg_where}: 'steps' must be [first, last]")
        first, last = steps
        if first != expected_first or last < first:
            raise ConfigError(f"{seg_where}: its steps must start at step {expected_first}")
        expected_first = last + 1
        shards_raw = segment_raw.get("shards")
        if not isinstance(shards_raw, list) or not shards_raw:
            raise ConfigError(f"{seg_where}: 'shards' must be a non-empty list")
        shards = tuple(shard_ref(item, seg_where) for item in shards_raw)
        wrong = [s.name for s in shards if s.layout != layout]
        if wrong:
            raise ConfigError(f"{seg_where}: the run's layout is {layout!r}, but it lists {wrong}")
        mixture = _string(segment_raw, "mixture", seg_where)
        if mixtures.get(mixture) != shards_raw:
            raise ConfigError(
                f"{seg_where}: the spec's mixture {mixture!r} is {mixtures.get(mixture)!r}, the "
                f"segment lists {shards_raw!r}"
            )
        start_kind = _string(segment_raw, "start_kind", seg_where)
        parent = segment_raw.get("parent_checkpoint")
        if parent is not None and (not isinstance(parent, str) or not parent):
            raise ConfigError(f"{seg_where}: parent_checkpoint must be a name")
        lowered, checkpoints = _step_record(
            step_records.get(segment_name, {}), settings, first, last, f'{where}.step_record."{segment_name}"'
        )
        segments.append(
            SegmentPlan(
                run=name,
                name=segment_name,
                folder=folder,
                first_step=first,
                last_step=last,
                shards=shards,
                start_kind=start_kind,
                parent_checkpoint=parent,
                mixture=mixture,
                train_iters=_integer(segment_raw, "train_iters", seg_where, minimum=1),
                launches=_launches(segment_raw, settings, first, last, seg_where),
                lowered=lowered,
                checkpoints=checkpoints,
            )
        )
    if expected_first != settings["steps"] + 1:
        raise ConfigError(
            f"{where}: its segments end at step {expected_first - 1}, the run has {settings['steps']} steps"
        )
    # A resume inside the run continues the counters of the segment before it: a
    # check the oracle's own config check cannot make when the resume names no
    # checkpoint.
    for before, after in zip(segments, segments[1:]):
        if after.start_kind != "resume":
            continue
        reached = before.launches[-1].consumed_at_start + before.launches[-1].steps * settings["global_batch_size"]
        if after.launches[0].consumed_at_start != reached:
            raise ConfigError(
                f"{where}: segment {after.name!r} resumes at consumed count "
                f"{after.launches[0].consumed_at_start}, segment {before.name!r} reaches {reached}"
            )
        # The checkpoint a resume names is one the segment before it registers at
        # its last step. A misspelt name would otherwise pass as a checkpoint from
        # outside the spec, and the start would go unchecked.
        named = after.parent_checkpoint
        if named is not None and (named, before.last_step) not in before.checkpoints:
            raise ConfigError(
                f"{where}: segment {after.name!r} resumes from {named!r}, which the step record of "
                f"{before.name!r} does not register at step {before.last_step}"
            )
    stated = raw.get("checkpoints")
    recorded = sorted(step for segment in segments for _, step in segment.checkpoints)
    if not isinstance(stated, list) or sorted(stated) != recorded:
        raise ConfigError(
            f"{where}: 'checkpoints' is {stated!r}, its step records name checkpoints at {recorded}"
        )
    return RunPlan(
        name=name,
        layout=layout,
        seed_of=seed_of,
        random_seed=settings["random_seed"],
        sequence_length=settings["sequence_length"],
        global_batch_size=settings["global_batch_size"],
        micro_batch_size=settings["micro_batch_size"],
        data_parallel_size=settings["data_parallel_size"],
        world_size=settings["world_size"],
        dataloader_type=settings["dataloader_type"],
        data_sharding=sharding,
        steps=settings["steps"],
        segments=tuple(segments),
    )


def load_spec(path: str | Path) -> FixtureSpec:
    """Read and check the fixture spec."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"fixture spec not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"the fixture spec is not TOML: {path}: {e}") from e
    if raw.get("format") != SPEC_FORMAT:
        raise ConfigError(f"the fixture spec's format is {raw.get('format')!r}, expected {SPEC_FORMAT!r}")

    common = _table(raw, "runs_common", "the spec")
    _only(common, _COMMON_KEYS, "runs_common")
    if common.get("megatron_bridge") != "none":
        raise ConfigError(
            "runs_common.megatron_bridge is not \"none\": the fixture runs drive Megatron-LM's "
            "sampler classes only"
        )
    if common.get("add_extra_token_to_sequence") is not True:
        raise ConfigError("runs_common.add_extra_token_to_sequence must be true")
    tokenizer = _table(raw, "tokenizer", "the spec")
    if tokenizer.get("type") != "GPT2BPETokenizer":
        raise ConfigError("tokenizer.type must be \"GPT2BPETokenizer\"")

    mixtures = _table(raw, "mixtures", "the spec")
    runs_raw = _table(raw, "runs", "the spec")
    runs = {}
    for name, run_raw in runs_raw.items():
        if not isinstance(run_raw, dict):
            raise ConfigError(f"runs.{name}: must be a table")
        runs[name] = _run(name, run_raw, runs_raw, mixtures)
    names = [segment.name for run in runs.values() for segment in run.segments]
    if len(set(names)) != len(names):
        raise ConfigError("the spec names a segment twice")

    sources = {}
    for source, source_raw in _table(raw, "sources", "the spec").items():
        records = source_raw.get("records") if isinstance(source_raw, dict) else None
        if not isinstance(records, list):
            raise ConfigError(f"sources.{source}: 'records' must be a list")
        sources[source] = expand_ids(records, f"sources.{source}")
    handles = _table(raw, "handles", "the spec")
    if not all(isinstance(v, str) and v for v in handles.values()):
        raise ConfigError("handles: every handle names a record id")

    conditions = []
    search = _table(raw, "search", "the spec")
    for i, item in enumerate(search.get("condition", [])):
        at = f"search.condition[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{at}: must be a table")
        conditions.append(
            Condition(n=_integer(item, "n", at, minimum=1), id=_string(item, "id", at), sentence=_string(item, "sentence", at))
        )
    if [c.n for c in conditions] != list(range(1, len(conditions) + 1)):
        raise ConfigError("search.condition: the conditions must be numbered 1, 2, 3, ... in order")

    return FixtureSpec(
        path=path,
        raw=raw,
        runs=runs,
        add_extra_token_to_sequence=True,
        split=_string(common, "split", "runs_common"),
        sampler_classes={
            "single": _string(common, "sampler_class_single", "runs_common"),
            "cyclic": _string(common, "sampler_class_cyclic", "runs_common"),
        },
        handles=dict(handles),
        sources=sources,
        conditions=tuple(conditions),
    )
