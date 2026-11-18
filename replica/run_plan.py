"""The runs of the fixture spec, as the replica's run wrapper needs them.

The spec's ``[runs]`` tables fix, for every run and segment, the settings of the
trainer's launch, the shards handed to it, the launches with their train state, the
operator's step record and the checkpoints (docs/interfaces.md 9.3; plan 5.1). The
replica reads them here with its own reader; it shares nothing with the oracle's.

Three values the spec does not state for every run are left as None here and are
filled from the environment the run is made in: ``data_sharding`` (the spec states
it for the cyclic run only; Megatron's argument default applies to the others),
the sampler class and the release tag (``replica.runs``).

Pure Python: no Megatron, no torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecError
from .plan import BuildPlan, MixturePlan, _load_toml, load_fixture_spec

DATALOADER_SINGLE = "single"
DATALOADER_CYCLIC = "cyclic"
DATALOADERS = (DATALOADER_SINGLE, DATALOADER_CYCLIC)


@dataclass(frozen=True)
class Launch:
    """One start of the trainer inside a segment, with its real sizes and the train
    state it starts from (plan 3.1, F4)."""

    position: int
    world_size: int
    data_parallel_size: int
    consumed_at_start: int
    step_at_start: int
    first_step: int
    last_step: int

    @property
    def steps(self) -> int:
        return self.last_step - self.step_at_start


@dataclass(frozen=True)
class RunSettings:
    """What the spec states for one run (the settings every segment of it shares)."""

    sequence_length: int
    global_batch_size: int
    micro_batch_size: int
    data_parallel_size: int
    world_size: int
    random_seed: int
    dataloader_type: str
    data_sharding: bool | None  # None: the spec does not state it


@dataclass(frozen=True)
class SegmentRun:
    """One run segment of the spec."""

    run: str
    name: str  # "E/seg-1"
    start_kind: str
    parent_checkpoint: str | None
    mixture: str
    layout: str
    shard_files: tuple[str, ...]  # the files of the layout, in the trainer's order
    train_iters: int
    first_step: int
    last_step: int
    launches: tuple[Launch, ...]
    lowered: tuple[tuple[int, int], ...]  # (step, tokens read of its last sample)
    checkpoints: tuple[tuple[str, int], ...]  # (name, step)
    settings: RunSettings

    @property
    def stem(self) -> str:
        """The segment's name with the slash replaced by a dash: the start of the
        names of its files."""
        return self.name.replace("/", "-")

    def steps(self) -> list[int]:
        """Every completed step of the segment, in order."""
        return [s for launch in self.launches for s in range(launch.first_step, launch.last_step + 1)]


@dataclass(frozen=True)
class RunsPlan:
    """The runs of a spec, in the spec's order (the order the runs are made and
    registered in)."""

    segments: tuple[SegmentRun, ...]
    release_tag: str
    sampler_classes: dict[str, str]  # dataloader type -> dotted class name
    add_extra_token_to_sequence: bool
    split: str
    megatron_bridge: str

    def runs(self) -> list[str]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.run not in seen:
                seen.append(segment.run)
        return seen

    def verified_joins(self, runs: Sequence[str] | None = None) -> tuple[tuple[str, str, str], ...]:
        """The (release tag, sampler type, sampler class) of every run named (all
        when None), sorted: the rows a store lists as verified joins (plan F6) so that
        the runs registered in it are not cut off by an unverified version. The tag
        is the one the spec pins, which the run wrapper holds against the checkout
        it runs in; the class is the spec's for the run's sampler type."""
        kinds = {s.settings.dataloader_type for s in self.segments if runs is None or s.run in runs}
        return tuple(sorted((self.release_tag, kind, self.sampler_classes[kind]) for kind in kinds))

    def of_run(self, run: str) -> list[SegmentRun]:
        return [s for s in self.segments if s.run == run]


def _int(table: dict, key: str, where: str, *, minimum: int = 0) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SpecError(f"{where}: {key} must be a whole number of at least {minimum}, got {value!r}")
    return value


def _str(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or value == "":
        raise SpecError(f"{where}: {key} must be a non-empty string, got {value!r}")
    return value


def _settings(run: str, table: dict, all_runs: dict) -> RunSettings:
    source = table
    if "same_settings_as" in table:
        other = table["same_settings_as"]
        if other not in all_runs or "same_settings_as" in all_runs[other]:
            raise SpecError(f"runs.{run}: same_settings_as names no run with settings of its own")
        source = all_runs[other]
    where = f"runs.{run}"
    sharding = source.get("data_sharding")
    if sharding is not None and not isinstance(sharding, bool):
        raise SpecError(f"{where}: data_sharding must be true or false")
    dp = _int(source, "data_parallel_size", where, minimum=1)
    kind = _str(source, "dataloader_type", where)
    if kind not in DATALOADERS:
        raise SpecError(f"{where}: dataloader_type must be one of {list(DATALOADERS)}, got {kind!r}")
    return RunSettings(
        sequence_length=_int(source, "sequence_length", where, minimum=1),
        global_batch_size=_int(source, "global_batch_size", where, minimum=1),
        micro_batch_size=_int(source, "micro_batch_size", where, minimum=1),
        data_parallel_size=dp,
        world_size=_int(source, "world_size", where, minimum=1) if "world_size" in source else dp,
        random_seed=_int(source, "random_seed", where),
        dataloader_type=kind,
        data_sharding=sharding,
    )


def _launches(table: dict, settings: RunSettings, first: int, last: int, where: str) -> tuple[Launch, ...]:
    consumed = _int(table, "consumed_at_start", where)
    if "launches" not in table:
        return (Launch(0, settings.world_size, settings.data_parallel_size, consumed,
                       first - 1, first, last),)
    items = table["launches"]
    if not isinstance(items, list) or not items:
        raise SpecError(f"{where}: launches must be a non-empty list")
    launches = []
    expected = first
    for i, item in enumerate(items):
        at = f"{where} launches[{i}]"
        if not isinstance(item, dict) or item.get("position") != i:
            raise SpecError(f"{at}: must be a table whose position is {i}")
        step_at_start = _int(item, "step_at_start", at)
        launch = Launch(
            position=i,
            world_size=_int(item, "world_size", at, minimum=1),
            data_parallel_size=_int(item, "data_parallel_size", at, minimum=1),
            consumed_at_start=_int(item, "consumed_at_start", at),
            step_at_start=step_at_start,
            first_step=_int(item, "first_step", at, minimum=1),
            last_step=_int(item, "last_step", at, minimum=1),
        )
        if launch.first_step != step_at_start + 1 or launch.first_step != expected or launch.last_step < expected:
            raise SpecError(f"{at}: its steps must start at step {expected}, where the launch before "
                            "it (or the segment) left off")
        launches.append(launch)
        expected = launch.last_step + 1
    if expected != last + 1 or launches[0].consumed_at_start != consumed:
        raise SpecError(f"{where}: its launches do not run from step {first} at consumed count "
                        f"{consumed} to step {last}")
    return tuple(launches)


def _step_record(table: dict, first: int, last: int, where: str):
    lowered, checkpoints = [], []
    for i, item in enumerate(table.get("steps", [])):
        at = f"{where} steps[{i}]"
        step = _int(item, "step", at, minimum=1)
        if not first <= step <= last:
            raise SpecError(f"{at}: step {step} is not a step of the segment")
        if "last_sample_tokens_read" in item:
            lowered.append((step, _int(item, "last_sample_tokens_read", at, minimum=1)))
    for i, item in enumerate(table.get("checkpoints", [])):
        at = f"{where} checkpoints[{i}]"
        step = _int(item, "step", at, minimum=1)
        if not first <= step <= last:
            raise SpecError(f"{at}: step {step} is not a step of the segment")
        checkpoints.append((_str(item, "name", at), step))
    return tuple(lowered), tuple(checkpoints)


def load_runs_plan(spec_path: Path, base: Path | None = None) -> RunsPlan:
    """Every run segment of the fixture spec at ``spec_path``, in the spec's order."""
    spec_path = Path(spec_path)
    doc = _load_toml(spec_path)
    plan: BuildPlan = load_fixture_spec(spec_path, base)  # the layouts and the mixtures
    mixtures: dict[str, MixturePlan] = {m.name: m for m in plan.mixtures}
    common = doc.get("runs_common", {})
    megatron = doc.get("megatron", {})
    where = str(spec_path)
    runs = doc.get("runs")
    if not isinstance(runs, dict) or not runs:
        raise SpecError(f"{where}: no [runs]")
    segments: list[SegmentRun] = []
    for run, table in runs.items():
        w = f"{where} [runs.{run}]"
        settings = _settings(run, table, runs)
        step_records = table.get("step_record", {})
        seg_tables = table.get("segments")
        if not isinstance(seg_tables, dict) or not seg_tables:
            raise SpecError(f"{w}: no segments")
        for name, seg in seg_tables.items():
            sw = f'{w} segments."{name}"'
            if not name.startswith(run + "/"):
                raise SpecError(f"{sw}: a segment's name starts with its run and a slash")
            steps = seg.get("steps")
            if not (isinstance(steps, list) and len(steps) == 2
                    and all(isinstance(s, int) and not isinstance(s, bool) for s in steps)):
                raise SpecError(f"{sw}: steps must be [first, last]")
            first, last = steps
            mixture = mixtures.get(_str(seg, "mixture", sw))
            if mixture is None:
                raise SpecError(f"{sw}: mixture {seg.get('mixture')!r} is not in [mixtures]")
            lowered, checkpoints = _step_record(step_records.get(name, {}), first, last,
                                                f'{w} step_record."{name}"')
            parent = seg.get("parent_checkpoint")
            if parent is not None and (not isinstance(parent, str) or parent == ""):
                raise SpecError(f"{sw}: parent_checkpoint must be a name")
            segments.append(SegmentRun(
                run=run, name=name, start_kind=_str(seg, "start_kind", sw),
                parent_checkpoint=parent, mixture=mixture.name, layout=mixture.layout,
                shard_files=tuple(mixture.shards),
                train_iters=_int(seg, "train_iters", sw, minimum=1),
                first_step=first, last_step=last,
                launches=_launches(seg, settings, first, last, sw),
                lowered=lowered, checkpoints=checkpoints, settings=settings))
    classes = {
        DATALOADER_SINGLE: _str(common, "sampler_class_single", f"{where} [runs_common]"),
        DATALOADER_CYCLIC: _str(common, "sampler_class_cyclic", f"{where} [runs_common]"),
    }
    if common.get("add_extra_token_to_sequence") is not True:
        raise SpecError(f"{where} [runs_common]: the replica's runs need add_extra_token_to_sequence = true")
    return RunsPlan(
        segments=tuple(segments),
        release_tag=_str(megatron, "release_tag", f"{where} [megatron]"),
        sampler_classes=classes,
        add_extra_token_to_sequence=True,
        split=_str(common, "split", f"{where} [runs_common]"),
        megatron_bridge=_str(common, "megatron_bridge", f"{where} [runs_common]"),
    )
