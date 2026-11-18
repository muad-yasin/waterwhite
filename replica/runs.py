"""The replica's run wrapper: it stands in for the thin wrapper around the team's
training launch (plan 3.3 "Training", 5.4, 3.5; docs/interfaces.md 8.1, 8.2, 8.4).

For every run segment of the fixture spec it launches Megatron's own dataset code
on the fixture's shards, so that Megatron writes the data-order cache folder with
``path_to_cache`` set, and then writes, beside it:

    <runs>/<run>/<segment, slash replaced by dash>.run.json     the run record
    <runs>/<run>/<segment...>.steps.json                         the step record
    <runs>/<run>/<segment...>.cache/                             Megatron's cache folder, as written
    <runs>/<run>/shards/<shard>.bin, .idx                        the shards the record names

The rows of the checkpoints of the runs are appended to the manifest (``built-from.jsonl``)
of the build folder the runs were made from, when that build has one (replica/built_from.py);
no manifest file lies in the runs folder.

The run record's cache digests are taken from the kept files once Megatron has
written them, and a cyclic run's shuffle digests are computed in the run's own
environment from the launch's real sizes, with the replica's own call to torch
(``megatron_runs_worker.py``). This module imports no Megatron and no torch: the
work that needs them is one subprocess of the Megatron environment.

Nothing here reads the oracle, starts an oracle script, or calls the join. The
record is written here, on the replica's side, and registered through the store
(``replica.register_runs``); the oracle's own record of the same segment is the
side a test compares it with.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from . import megatron_tokenizer
from .build import WORK_DIR, BuildOptions, build, file_sha256, require_new_folder
from .built_from import record_checkpoints
from .errors import InputError, RunError, SpecError
from .fixture import default_spec_path
from .plan import load_fixture_spec
from .run_plan import RunsPlan, SegmentRun, load_runs_plan

RUN_RECORD_FORMAT = "waterwhite.run-record/1"
RUN_RECORD_SUFFIX = ".run.json"
STEP_RECORD_SUFFIX = ".steps.json"
CACHE_SUFFIX = ".cache"
SHARDS_FOLDER = "shards"
FIXTURE_FOLDER = "fixture-build"  # under the scratch folder, when the caller names none
WORKER_FOLDER = "runs-worker"  # under the scratch folder

_WORKER = Path(__file__).with_name("megatron_runs_worker.py")


def tool_source() -> str:
    """The folder that holds the package ``waterwhite``, for the worker's import path.
    The worker imports ``waterwhite.ids`` only (the digests: one definition)."""
    import waterwhite

    return str(Path(waterwhite.__file__).resolve().parent.parent)


# --- The Megatron release the run is made on -----------------------------------------------


def _git_dir(checkout: Path) -> Path:
    dot_git = checkout / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            target = Path(text[len("gitdir:"):].strip())
            return target if target.is_absolute() else (checkout / target).resolve()
    raise RunError(f"not a git checkout: {checkout}")


def checkout_commit_and_tags(checkout: Path) -> tuple[str, list[str]]:
    """The commit a Megatron-LM checkout is at and the tags that name it, read from
    its ``.git`` folder; no ``git`` command runs."""
    git = _git_dir(checkout)
    packed = []
    packed_file = git / "packed-refs"
    if packed_file.is_file():
        for line in packed_file.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            if line.startswith("^"):
                if packed:
                    packed[-1][2] = line[1:].strip()
                continue
            sha, _, name = line.partition(" ")
            packed.append([sha.strip(), name.strip(), None])
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        ref = head[len("ref:"):].strip()
        loose = git / ref
        if loose.is_file():
            commit = loose.read_text(encoding="utf-8").strip()
        else:
            found = [sha for sha, name, _ in packed if name == ref]
            if not found:
                raise RunError(f"cannot resolve HEAD of the checkout {checkout}")
            commit = found[0]
    else:
        commit = head
    tags = {name[len("refs/tags/"):] for sha, name, peeled in packed
            if name.startswith("refs/tags/") and commit in (sha, peeled)}
    tags_dir = git / "refs" / "tags"
    if tags_dir.is_dir():
        for root, _dirs, files in os.walk(tags_dir):
            for file in files:
                path = Path(root) / file
                if path.read_text(encoding="utf-8").strip() == commit:
                    tags.add(path.relative_to(tags_dir).as_posix())
    return commit, sorted(tags)


def observed_release_tag(env: megatron_tokenizer.MegatronEnv, expected: str,
                         installed_version: str) -> str:
    """The release the code of this environment is, observed and not copied from the
    spec: the one tag that names the commit of the Megatron-LM checkout. It must be
    the tag the spec pins (plan F6), and the installed package must be built from
    that commit (its local version suffix is the commit's start) and carry the
    tag's version. Anything else is refused: the record would name code that did
    not run."""
    commit, tags = checkout_commit_and_tags(env.root / "src" / "Megatron-LM")
    if len(tags) != 1:
        raise RunError(f"the Megatron-LM checkout at {commit[:12]} is named by {len(tags)} tags "
                       f"({tags}); the replica needs exactly one")
    tag = tags[0]
    if tag != expected:
        raise RunError(f"the spec pins the Megatron release {expected!r}, the environment's "
                       f"checkout is {tag!r}")
    base, _, local = installed_version.partition("+")
    if tag != f"core_v{base}":
        raise RunError(f"the installed megatron-core is {installed_version}, the checkout's tag "
                       f"is {tag}")
    if local and not commit.startswith(local):
        raise RunError(f"the installed megatron-core {installed_version} is not built from the "
                       f"checkout's commit {commit[:12]}")
    return tag


# --- The records ----------------------------------------------------------------------------


def _relative_posix(path: Path, folder: Path) -> str:
    return Path(os.path.relpath(path, folder)).as_posix()


def run_record(segment: SegmentRun, *, record_folder: Path, shard_prefixes: Sequence[Path],
               fixed: dict, versions: dict, split: str) -> dict:
    """The run record of one segment (docs/interfaces.md 8.1). ``fixed`` is what the
    worker reported for it: the sampler class, the data-sharding flag, the cache
    digests and the shuffle digests. A prefix is stored relative to the record's
    folder."""
    s = segment.settings
    return {
        "format": RUN_RECORD_FORMAT,
        "run": segment.run,
        "segment": segment.name,
        "start_kind": segment.start_kind,
        "parent_checkpoint": segment.parent_checkpoint,
        "settings": {
            "dataloader_type": s.dataloader_type,
            "sampler_class": fixed["sampler_class"],
            "data_sharding": fixed["data_sharding"],
            "micro_batch_size": s.micro_batch_size,
            "global_batch_size": s.global_batch_size,
            "sequence_length": s.sequence_length,
            "add_extra_token_to_sequence": True,
            "random_seed": s.random_seed,
            "split": split,
            "train_iters": segment.train_iters,
            "train_samples": None,
            "batch_schedule": None,
        },
        "versions": {key: versions[key] for key in
                     ("release_tag", "megatron_core", "megatron_bridge", "torch")},
        "shards": [
            {"position": position,
             "prefix": _relative_posix(prefix, record_folder),
             "bin_sha256": file_sha256(Path(str(prefix) + ".bin")),
             "idx_sha256": file_sha256(Path(str(prefix) + ".idx"))}
            for position, prefix in enumerate(shard_prefixes)
        ],
        "launches": [
            {"position": launch.position, "world_size": launch.world_size,
             "data_parallel_size": launch.data_parallel_size,
             "consumed_at_start": launch.consumed_at_start,
             "step_at_start": launch.step_at_start,
             "first_step": launch.first_step, "last_step": launch.last_step}
            for launch in segment.launches
        ],
        "cache_digests": [{"kind": r["kind"], "shard": r["shard"], "digest": r["digest"]}
                          for r in fixed["cache_digests"]],
        "shuffle_digests": [{"launch": r["launch"], "epoch": r["epoch"], "rank": r["rank"],
                             "digest": r["digest"], "count": r["count"]}
                            for r in fixed["shuffle_digests"]],
    }


def step_record(segment: SegmentRun) -> dict:
    """The step record of one segment (docs/interfaces.md 8.2): every completed step
    with its batch size, the global batch size; a step the operator lowers carries
    ``last_sample_tokens_read``; the checkpoints of the segment."""
    lowered = dict(segment.lowered)
    rows = []
    for step in segment.steps():
        row = {"step": step, "batch_size": segment.settings.global_batch_size}
        if step in lowered:
            row["last_sample_tokens_read"] = lowered[step]
        rows.append(row)
    return {"segment": segment.name, "steps": rows,
            "checkpoints": [{"name": name, "step": step} for name, step in segment.checkpoints]}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- Where things are -----------------------------------------------------------------------


def segment_paths(out: Path, segment: SegmentRun) -> dict[str, Path]:
    """The three paths one segment keeps under the runs folder ``out``."""
    folder = Path(out) / segment.run
    return {"run_record": folder / (segment.stem + RUN_RECORD_SUFFIX),
            "step_record": folder / (segment.stem + STEP_RECORD_SUFFIX),
            "cache": folder / (segment.stem + CACHE_SUFFIX)}


def _fixture_shard(fixture: Path, segment: SegmentRun, shard: str) -> Path:
    return fixture / "pipeline" / segment.layout / "shards" / shard


# --- The worker -----------------------------------------------------------------------------


def _run_worker(env: megatron_tokenizer.MegatronEnv, plan_path: Path, result_path: Path,
                log_path: Path, timeout: float) -> dict:
    """Run the worker to its end with the environment's interpreter, ``-I``, and the
    offline settings every Megatron subprocess of the replica gets. Its log is kept."""
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            [str(env.python), "-I", str(_WORKER), str(plan_path), str(result_path)],
            stdout=log, stderr=subprocess.STDOUT, env=env.child_environment(),
            start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RunError(f"the run wrapper's worker did not finish within {timeout:.0f} seconds "
                           f"and was stopped; see {log_path}") from None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)  # its own process group, by its id
                process.wait()
    if code != 0:
        tail = log_path.read_bytes()[-2000:].decode("utf-8", errors="replace")
        raise RunError(f"the run wrapper's worker exited with code {code}; see {log_path}\n{tail}")
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunError(f"the run wrapper's worker left no readable result ({exc}); "
                       f"see {log_path}") from exc


# --- The command ----------------------------------------------------------------------------


def make_segment_runs(out: Path, fixture: Path, plan: RunsPlan, segments: Sequence[SegmentRun],
                      env: megatron_tokenizer.MegatronEnv, tokenizer, timeout: float
                      ) -> tuple[list[dict], dict, dict]:
    """The one launch path of the replica's runs: for each of ``segments`` copy its
    shards from ``fixture`` (a folder holding ``pipeline/<layout>/shards/``) into the
    run's folder under ``out``, launch Megatron's dataset code once, in one subprocess
    of ``env``, into a new empty cache folder per segment, and write the run record and
    the step record beside it. ``tokenizer`` is the spec's tokenizer table. ``out``
    exists. Returns one summary row per segment, the versions the records name, and
    the seconds of the three parts: ``copy`` (the shard files), ``worker`` (Megatron's
    dataset build and the digests of what it kept) and ``records`` (the two records of
    every segment)."""
    vocab, merges = env.tokenizer_files(tokenizer)
    work = out / WORK_DIR
    work.mkdir(exist_ok=True)
    seconds: dict[str, float] = {}

    started = time.perf_counter()
    # The shards first: the record names them relative to its folder, and Megatron's
    # cache files and descriptions carry the paths of the files it was handed.
    prefixes: dict[str, list[Path]] = {}
    for segment in segments:
        folder = out / segment.run
        for shard in segment.shard_files:
            source = _fixture_shard(fixture, segment, shard)
            target = folder / SHARDS_FOLDER / shard
            for suffix in (".bin", ".idx"):
                if not Path(str(source) + suffix).is_file():
                    raise InputError(f"the fixture build holds no shard file {source}{suffix}")
                if not Path(str(target) + suffix).exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(source) + suffix, str(target) + suffix)
        prefixes[segment.name] = [out / segment.run / SHARDS_FOLDER / s for s in segment.shard_files]
        segment_paths(out, segment)["cache"].mkdir(parents=True)

    seconds["copy"] = time.perf_counter() - started
    worker_plan = {
        "src": tool_source(),
        "tokenizer": {"type": tokenizer.type, "vocab_file": str(vocab), "merges_file": str(merges)},
        "segments": [
            {"name": s.name,
             "cache_dir": str(segment_paths(out, s)["cache"]),
             "shard_prefixes": [str(p) for p in prefixes[s.name]],
             "dataloader_type": s.settings.dataloader_type,
             "data_sharding": s.settings.data_sharding,
             "micro_batch_size": s.settings.micro_batch_size,
             "global_batch_size": s.settings.global_batch_size,
             "sequence_length": s.settings.sequence_length,
             "random_seed": s.settings.random_seed,
             "split": plan.split,
             "train_iters": s.train_iters,
             "add_extra_token_to_sequence": plan.add_extra_token_to_sequence,
             "launches": [{"position": l.position, "data_parallel_size": l.data_parallel_size,
                           "consumed_at_start": l.consumed_at_start, "step_at_start": l.step_at_start,
                           "first_step": l.first_step, "last_step": l.last_step}
                          for l in s.launches]}
            for s in segments],
    }
    scratch = work / WORKER_FOLDER
    scratch.mkdir(exist_ok=True)
    plan_path, result_path = scratch / "plan.json", scratch / "result.json"
    write_json(plan_path, worker_plan)
    started = time.perf_counter()
    result = _run_worker(env, plan_path, result_path, scratch / "worker.log", timeout)
    seconds["worker"] = time.perf_counter() - started
    started = time.perf_counter()

    by_name = {row["name"]: row for row in result["segments"]}
    if sorted(by_name) != sorted(s.name for s in segments):
        raise RunError("the worker reported other segments than the plan named")
    release = observed_release_tag(env, plan.release_tag, result["versions"]["megatron_core"])
    versions = {
        "release_tag": release,
        "megatron_core": result["versions"]["megatron_core"],
        "megatron_bridge": None if plan.megatron_bridge == "none" else plan.megatron_bridge,
        "torch": result["versions"]["torch"],
    }
    summary_rows = []
    for segment in segments:
        fixed = by_name[segment.name]
        expected_class = plan.sampler_classes[segment.settings.dataloader_type]
        if fixed["sampler_class"] != expected_class:
            raise RunError(f"segment {segment.name!r}: the environment's sampler class is "
                           f"{fixed['sampler_class']}, the spec names {expected_class}")
        paths = segment_paths(out, segment)
        record = run_record(segment, record_folder=paths["run_record"].parent,
                            shard_prefixes=prefixes[segment.name], fixed=fixed, versions=versions,
                            split=plan.split)
        write_json(paths["run_record"], record)
        write_json(paths["step_record"], step_record(segment))
        summary_rows.append({
            "run": segment.run, "segment": segment.name,
            "run_record": _relative_posix(paths["run_record"], out),
            "step_record": _relative_posix(paths["step_record"], out),
            "cache": _relative_posix(paths["cache"], out),
            "dataset_samples": fixed["dataset_samples"],
            "cache_digests": len(record["cache_digests"]),
            "shuffle_digests": len(record["shuffle_digests"]),
        })
    seconds["records"] = time.perf_counter() - started
    return summary_rows, versions, seconds


def build_runs(out: Path, fixture_build: Path | None = None, *, spec_path: Path | None = None,
               runs: Sequence[str] | None = None,
               megatron_env: megatron_tokenizer.MegatronEnv | None = None,
               timeout: float = 600.0) -> dict:
    """Make the fixture's runs: for each segment of the spec (or of the runs named),
    in the spec's order, launch Megatron's dataset code into a new cache folder and
    write the run record and the step record beside it. ``out`` must be new or
    empty. ``fixture_build`` is a folder ``replica build-fixture`` wrote, with its
    shards; with none, the fixture is built first into the scratch folder of
    ``out`` (``.work/fixture-build``, tokenized by Megatron's own preprocessing).

    Returns a summary that can be written as JSON."""
    out = Path(out).resolve()
    spec = Path(spec_path) if spec_path is not None else default_spec_path()
    plan: RunsPlan = load_runs_plan(spec)
    chosen = list(plan.runs()) if runs is None else list(runs)
    unknown = [r for r in chosen if r not in plan.runs()]
    if unknown or len(set(chosen)) != len(chosen) or not chosen:
        raise SpecError(f"the runs asked for are each named once from the spec's {plan.runs()}, "
                        f"got {list(runs or [])}")
    chosen = [r for r in plan.runs() if r in chosen]  # the spec's order is the run order
    require_new_folder(out, "the runs")
    env = megatron_env or megatron_tokenizer.MegatronEnv.from_environment()
    segments = [s for s in plan.segments if s.run in chosen]
    build_plan = load_fixture_spec(spec)
    tokenizer = build_plan.tokenizer
    env.tokenizer_files(tokenizer)  # refuse a missing tokenizer file before anything is built

    out.mkdir(parents=True, exist_ok=True)
    work = out / WORK_DIR
    work.mkdir(exist_ok=True)
    if fixture_build is None:
        fixture = work / FIXTURE_FOLDER
        build(load_fixture_spec(spec), fixture,
              BuildOptions(tokenize=True, parallel=3), megatron_env=env)
    else:
        fixture = Path(fixture_build).resolve()
        if not (fixture / "pipeline").is_dir():
            raise InputError(f"not a fixture build (no pipeline/ folder): {fixture}")

    summary_rows, versions, _seconds = make_segment_runs(
        out, fixture, plan, segments, env, tokenizer, timeout)
    # The fixture build made with capture on recorded its manifest: the runs made from it
    # add the rows of their checkpoints to that file. A fixture build with no manifest
    # (made with capture off, or built here into the scratch folder) gets none.
    record_checkpoints(fixture, out, segments, build_plan.layouts)
    return {"out": str(out), "runs": chosen, "segments": summary_rows,
            "versions": versions}
