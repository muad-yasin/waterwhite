"""A generated set with a store, in one call: what Checks 3 and 6 need of the
generated sets (plan 5.5; docs/plant.md).

    from replica.sets import build_captured_set, build_uncaptured_set
    from replica.fixture import add_recovery_maps, recover_uncaptured
    from replica import setups

    # The capture maps, and the two recovery maps beside them in one store.
    built = build_captured_set(folder_a, set_dir)
    add_recovery_maps(built, also_token_matching=True)

    # The run made without capture, recovered in both modes.
    run = build_uncaptured_set(folder_b, set_dir)
    recovered = recover_uncaptured(folder_b, run, also_token_matching=True)

    # A set-up: copy the folder of the run, change an input, then recover.
    shutil.copytree(run.out, folder_c / "build")
    setups.change_first_line(folder_c / "build", "round-robin")
    failed = recover_uncaptured(folder_c, run.moved_to(folder_c / "build"),
                                also_token_matching=True)

Check 3 runs on the set of 100,000 records in the round-robin layout, with the
set's dedup switched off (``fixture/plant/generate.py --dedup off``), so that both
members of each identical pair reach a shard. The same calls run on a small set.

The ids of the chunk files are the replica's own. They are written into the
tokenizer input of every shard, with capture on and off, and the tool is never
handed them: it reads the ``text`` field of a registered file and stores a
locator and a fingerprint per line (plan Check 3).
"""

from __future__ import annotations

from pathlib import Path

from . import megatron_tokenizer
from .build import HARNESS_IN_PROCESS, BuildOptions, require_new_folder
from .fixture import (
    BUILD_DIR,
    STORE_FILE,
    CapturedFixture,
    _captured_build,
    build_uncaptured,
    default_spec_path,
)
from .plan import LAYOUT_ROUND_ROBIN, BuildPlan, load_set, load_tokenizer_settings


def _set_plan(set_dir: Path, tokenizer_spec: Path | None, tokenize: bool) -> BuildPlan:
    """The plan of a generated set. A set names no tokenizer: it is given the one
    of ``tokenizer_spec``, by default the fixture spec of this repository."""
    tokenizer = None
    if tokenize:
        tokenizer = load_tokenizer_settings(
            Path(tokenizer_spec) if tokenizer_spec is not None else default_spec_path())
    return load_set(Path(set_dir), tokenizer)


def build_captured_set(
    folder: Path,
    set_dir: Path,
    *,
    tokenizer_spec: Path | None = None,
    layouts: tuple[str, ...] = (LAYOUT_ROUND_ROBIN,),
    tokenize: bool = True,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
    generation_harness: str = HARNESS_IN_PROCESS,
    verified_joins: tuple[tuple[str, str, str], ...] = (),
) -> CapturedFixture:
    """Build a generated set into ``folder`` with capture on: ``folder/store.sqlite``
    and ``folder/build``. With ``tokenize`` True every shard is registered with
    its capture map over its tokenizer input. ``folder`` must be new or empty.

    ``generation_harness`` is "in-process" by default: one sandbox per generated
    row is about 84 ms on a laptop, and a build may run its jobs in process only
    where no graded row compares an answer on it (Check 3 compares maps). A build
    whose answers are graded passes "confined".

    ``verified_joins`` are the (release tag, sampler type, sampler class) rows the new
    store lists as joins verified (plan F6); a store a run is registered in lists the
    join of that run (D-37), as ``replica.probe_run`` needs for the sequential one."""
    folder = Path(folder)
    require_new_folder(folder, "a captured set")
    plan = _set_plan(set_dir, tokenizer_spec, tokenize)
    if tokenize and megatron_env is None:
        megatron_env = megatron_tokenizer.MegatronEnv.from_environment()
    options = BuildOptions(layouts=layouts, tokenize=tokenize, parallel=parallel,
                           tokenizer_timeout=tokenizer_timeout,
                           generation_harness=generation_harness)
    return _captured_build(folder / STORE_FILE, folder / BUILD_DIR, None, options,
                           megatron_env, plan=plan, verified_joins=verified_joins)


def build_uncaptured_set(
    folder: Path,
    set_dir: Path,
    *,
    tokenizer_spec: Path | None = None,
    layouts: tuple[str, ...] = (LAYOUT_ROUND_ROBIN,),
    tokenize: bool = True,
    parallel: int = 1,
    tokenizer_timeout: float = 900.0,
    megatron_env: megatron_tokenizer.MegatronEnv | None = None,
):
    """Build a generated set into ``folder/build`` with capture off and unseeded
    shuffles: the run made without capture (plan Check 6). No store is made.
    ``replica.fixture.recover_uncaptured`` recovers it afterwards."""
    plan = _set_plan(set_dir, tokenizer_spec, tokenize)
    options = BuildOptions(layouts=layouts, seeded=False, tokenize=tokenize, parallel=parallel,
                           tokenizer_timeout=tokenizer_timeout)
    return build_uncaptured(folder, plan, options, megatron_env=megatron_env)
