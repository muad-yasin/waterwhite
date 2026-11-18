"""A run over a generated set: the replica's run for the measurement of plan 5.6
(plan 3.3 "Training", 5.4; docs/replica.md section 14).

    from replica.probe_run import probe_run

    printed = probe_run(set_build, store, out)       # the same as ``python -m replica probe-run``

It makes one run read with the sequential sampler over the shards a generated-set
build wrote (``<set build>/pipeline/<layout>/shards/*``), with the settings of run E
of the fixture spec, and puts it into the store the set was built into:

1. the mixture: a snapshot of those shards, in the order the run reads them, registered
   through ``Store.register_mixture`` as the mixture capture point does;
2. the run: Megatron's own dataset code over the shards, through the one launch path of
   the run wrapper (``replica.runs.make_segment_runs``), which writes the cache folder,
   the run record and the step record;
3. the registration: ``waterwhite register-run``, started as the registration of the
   fixture's runs is (``replica.register_runs``).

The store must already list the sequential join of the spec's release as verified
(plan F6; D-37, D-38): the list is written when a store is made and the replica never
writes it here. The check is the first thing done, before the output folder, the
mixture or the worker.

This module imports nothing of ``oracle/`` and nothing of ``waterwhite.join``; it
writes the store only through ``Store.register_mixture`` and the tool's own command.
The replica process imports neither Megatron nor torch.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from pathlib import Path

from waterwhite import vocabulary as v
from waterwhite.errors import WaterwhiteError
from waterwhite.store.store import Store

from . import megatron_tokenizer, pipeline_links
from .build import require_new_folder
from .capture_hooks import MIXTURE_SELECTION
from .errors import InputError, RunError, SpecError
from .fixture import default_spec_path
from .plan import LAYOUTS, LAYOUT_ROUND_ROBIN, load_fixture_spec
from .register_runs import register_runs
from .run_plan import DATALOADER_SINGLE, Launch, RunsPlan, SegmentRun, load_runs_plan
from .runs import make_segment_runs
from .shard_index import read_shard_index

TEMPLATE_RUN = "E"  # the run whose settings the probe takes
PROBE_RUN = "probe"
PROBE_SEGMENT = "probe/seg-1"
MIXTURE_NAME = "probe/{layout}"
CHECKPOINT_NAME = "probe@{step}"


def shard_order_key(name: str) -> list:
    """The order the run reads the shards of a layout in: the names sorted, each run
    of digits as a number, so ``shard-2`` comes before ``shard-10``. A partitioned set
    names its shards by source (``shard-A``, ``shard-Z2``, ``shard-gen``); the same rule
    orders them."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def shards_of(set_build: Path, layout: str) -> list[str]:
    """The shard names (file names without ``.bin`` or ``.idx``) a set build holds for
    ``layout``, in the order the run reads them. A shard with one of its two files
    missing is refused."""
    folder = Path(set_build) / "pipeline" / layout / "shards"
    if not folder.is_dir():
        raise InputError(f"not a set build with the layout {layout!r} (no {folder})")
    found = {path.name[: -len(suffix)] for suffix in (".bin", ".idx")
             for path in folder.glob(f"*{suffix}")}
    names = sorted(found, key=shard_order_key)
    if not names:
        raise InputError(f"the set build holds no shard under {folder}")
    for name in names:
        for suffix in (".bin", ".idx"):
            if not (folder / f"{name}{suffix}").is_file():
                raise InputError(f"the shard {name!r} has no {name}{suffix} under {folder}")
    return names


def _joins_row(plan: RunsPlan) -> tuple[str, str, str]:
    """The (release tag, sampler type, sampler class) the probe's run needs listed: the
    spec's pinned release, the sequential sampler, and the spec's class for it."""
    return (plan.release_tag, DATALOADER_SINGLE, plan.sampler_classes[DATALOADER_SINGLE])


def _require_listed(store: Path, row: tuple[str, str, str]) -> None:
    try:
        with Store.open(store, readonly=True) as held:
            listed = held.verified_joins()
    except WaterwhiteError as error:
        raise RunError(f"the store cannot be read: {error}") from error
    if row not in listed:
        raise RunError(
            f"the store lists no verified join for the release {row[0]!r}, sampler type "
            f"{row[1]!r}, sampler class {row[2]!r} (it lists {list(listed)}): a store that is to "
            "hold a run lists the joins of its runs when it is made, and the probe run never "
            "writes that list (D-37, D-38)")


def _whole_steps(set_build: Path, layout: str, names: list[str], sequence_length: int,
                 global_batch_size: int) -> int:
    """The steps the set gives: one pass over each shard (Megatron's dataset takes
    ``(tokens - 1) // sequence_length`` samples of a shard when no sample count is
    asked of the shard; the sequence takes one token more than its length), in whole
    steps of the global batch size."""
    folder = Path(set_build) / "pipeline" / layout / "shards"
    samples = sum((read_shard_index(folder / f"{name}.idx").tokens - 1) // sequence_length
                  for name in names)
    return samples // global_batch_size


def probe_segment(plan: RunsPlan, layout: str, names: list[str], steps: int) -> SegmentRun:
    """The one segment of the probe: run E's settings, a fresh start, one launch over
    ``steps`` steps from sample 0, no lowered step, and one checkpoint at the last
    step (a store answers a question about a checkpoint, so the segment has one)."""
    template = plan.of_run(TEMPLATE_RUN)[0]
    settings = template.settings
    if settings.dataloader_type != DATALOADER_SINGLE:
        raise SpecError(f"run {TEMPLATE_RUN} of the spec is not a sequential run")
    launch = Launch(position=0, world_size=settings.world_size,
                    data_parallel_size=settings.data_parallel_size, consumed_at_start=0,
                    step_at_start=0, first_step=1, last_step=steps)
    return SegmentRun(
        run=PROBE_RUN, name=PROBE_SEGMENT, start_kind="fresh", parent_checkpoint=None,
        mixture=MIXTURE_NAME.format(layout=layout), layout=layout, shard_files=tuple(names),
        train_iters=steps, first_step=1, last_step=steps, launches=(launch,), lowered=(),
        checkpoints=((CHECKPOINT_NAME.format(step=steps), steps),), settings=settings)


def _register_mixture(store: Path, set_build: Path, layout: str, names: list[str],
                      mixture: str) -> str:
    """The mixture capture point of the replica, for a run over a set build: the
    shards named, each by its registered instance (found by the path its files lie at),
    in the order the run reads them, with the digest of the two files."""
    folder = Path(set_build) / "pipeline" / layout / "shards"
    try:
        with Store.open(store) as held:
            entries = []
            for position, name in enumerate(names):
                prefix = folder / name
                shards = [a for a in held.artifacts_by_path(prefix) if a.kind == v.KIND_SHARD]
                if len(shards) != 1:
                    raise RunError(
                        f"the store holds {len(shards)} registered shards at {prefix}; the store "
                        "of the probe run is the one the set was built into with capture on")
                entries.append(pipeline_links.snapshot_shard(held, position,
                                                             shards[0].instance_id, prefix))
            return held.register_mixture(
                name=mixture, shards=entries,
                settings={"selection": MIXTURE_SELECTION,
                          "directory": pipeline_links.stored_path(held, folder)})
    except WaterwhiteError as error:
        raise RunError(f"the store refused the mixture of the probe run: {error}") from error


def probe_run(set_build: Path, store: Path, out: Path, *, layout: str = LAYOUT_ROUND_ROBIN,
              steps: int | None = None, spec_path: Path | None = None,
              megatron_env: megatron_tokenizer.MegatronEnv | None = None,
              timeout: float = 600.0) -> dict:
    """Make the run and register it. ``set_build`` is a folder ``replica build-set`` (or
    ``replica.sets.build_captured_set``) wrote with capture on and tokenization on,
    ``store`` the store it was built into, ``out`` a new or empty folder for the run's
    records. ``steps`` is the number of steps; with none, the set's one pass in whole
    steps. ``spec_path`` is the fixture spec that gives the settings of run E, the pinned
    Megatron release and the tokenizer (the set must have been tokenized with that
    tokenizer); ``timeout`` is what the Megatron worker may take.

    Returns ``{"segment", "mixture", "shards", "steps", "seconds", "register_run"}``:
    the segment's name, the mixture's name, the number of shards, the steps, the seconds
    of the three parts (``mixture``: the digests of the shard files and the registration
    of the snapshot; ``run``: the worker, which is Megatron's dataset build and the
    digests of what it kept, and the writing of the run record and the step record;
    ``register``: ``waterwhite register-run``), and what register-run printed."""
    if layout not in LAYOUTS:
        raise SpecError(f"the layout is one of {list(LAYOUTS)}, got {layout!r}")
    set_build, store, out = Path(set_build).resolve(), Path(store).resolve(), Path(out).resolve()
    spec = Path(spec_path) if spec_path is not None else default_spec_path()
    plan = load_runs_plan(spec)
    _require_listed(store, _joins_row(plan))  # first: nothing else is done for a store that refuses

    names = shards_of(set_build, layout)
    settings = plan.of_run(TEMPLATE_RUN)[0].settings
    available = _whole_steps(set_build, layout, names, settings.sequence_length,
                             settings.global_batch_size)
    if steps is None:
        steps = available
    if steps < 1 or steps > available:
        raise SpecError(f"the set gives {available} whole steps of {settings.global_batch_size} "
                        f"samples; steps asked for: {steps}")
    segment = probe_segment(plan, layout, names, steps)
    probe_plan = replace(plan, segments=(segment,))
    require_new_folder(out, "the probe run")
    env = megatron_env or megatron_tokenizer.MegatronEnv.from_environment()
    tokenizer = load_fixture_spec(spec).tokenizer
    env.tokenizer_files(tokenizer)

    started = time.perf_counter()
    _register_mixture(store, set_build, layout, names, segment.mixture)
    seconds = {"mixture": time.perf_counter() - started}

    out.mkdir(parents=True, exist_ok=True)
    rows, _versions, parts = make_segment_runs(out, set_build, probe_plan, [segment], env,
                                               tokenizer, timeout)
    # The run is the worker and the records; copying the shard files is not Megatron's work.
    seconds["run"] = parts["worker"] + parts["records"]
    if rows[0]["dataset_samples"] < steps * settings.global_batch_size:
        raise RunError(f"Megatron's dataset holds {rows[0]['dataset_samples']} samples, fewer "
                       f"than the {steps * settings.global_batch_size} the run reads")

    started = time.perf_counter()
    (printed,) = register_runs(store, out, probe_plan, runs=[PROBE_RUN])
    seconds["register"] = time.perf_counter() - started
    return {"segment": segment.name, "mixture": segment.mixture, "shards": len(names),
            "steps": steps, "seconds": seconds, "register_run": printed}
