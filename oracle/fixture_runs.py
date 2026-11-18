"""The fixture's runs from the spec (plan 5.1, 5.3): one command that reads the
fixture spec and a built fixture folder and writes, for every run segment the
spec names, the segment config, a new cache folder, the run record, the step
record and the oracle's log. Layout and use: docs/oracle.md, "Fixture runs".

    python -I oracle/fixture_runs.py --spec fixture/spec/fixture_spec.toml \\
        --fixture <built fixture> --out <new folder> [--seed RUN=N ...]

The spec says which shards, settings, launches, lowered steps and checkpoints
each segment has. The seeds come from the spec; ``--seed`` replaces one run's
seed for this call. Everything else is the oracle's own work, segment by segment
(``pipeline.run_segment``): nothing here computes a read position.

This module imports Megatron only when it runs a segment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle import config as oracle_config  # noqa: E402
from oracle import log_rows, positions  # noqa: E402
from oracle.digests import file_sha256  # noqa: E402
from oracle.errors import ConfigError, OracleError  # noqa: E402
from oracle.fixture_spec import FixtureSpec, RunPlan, SegmentPlan, ShardRef, load_spec  # noqa: E402

RUNS_FORMAT = "waterwhite.oracle-fixture-runs/1"

# The layout of the output folder.
RUNS_FOLDER = "runs"
OVERVIEW_FILE = "fixture_runs.json"
CONFIG_FILE = "config.json"
CACHE_FOLDER = "cache"
RUN_RECORD_FILE = "run_record.json"
STEP_RECORD_FILE = "step_record.json"
LOG_FILE = "oracle_log.jsonl"
DIGESTS_FILE = "index_digests.json"
SUMMARY_FILE = "summary.json"

SHARDS_FROM_PIPELINE = "pipeline"
SHARDS_FROM_MIXTURES = "mixtures"
SHARDS_FROM = (SHARDS_FROM_PIPELINE, SHARDS_FROM_MIXTURES)

_ENV_TEXT = "$" + oracle_config.ENV_VARIABLE


# --- Where things are in a built fixture ----------------------------------------------------


def shard_prefix(fixture: Path, segment: SegmentPlan, ref: ShardRef, shards_from: str) -> Path:
    """The prefix of a shard's ``.bin`` and ``.idx`` in a built fixture
    (docs/replica.md, "The folder one build writes"): the layout's own shard
    file, or the copy inside the segment's mixture folder."""
    if shards_from == SHARDS_FROM_PIPELINE:
        return fixture / "pipeline" / ref.layout / "shards" / ref.file
    if shards_from == SHARDS_FROM_MIXTURES:
        return fixture / "mixtures" / segment.mixture / ref.file
    raise ConfigError(f"shards are taken from one of {list(SHARDS_FROM)}, got {shards_from!r}")


def tokenizer_input(fixture: Path, ref: ShardRef) -> Path:
    """The tokenizer-input file of a shard: line i is document i of the shard."""
    return fixture / "pipeline" / ref.layout / "tokenizer-input" / f"{ref.file}.jsonl"


def sanitizer_log(fixture: Path, layout: str) -> Path:
    return fixture / "pipeline" / layout / "sanitizer.log.jsonl"


def find_tokenizer(snapshot: str | Path | None) -> dict:
    """The GPT-2 tokenizer files: ``vocab.json`` and ``merges.txt`` of the given
    snapshot folder, or of the one snapshot in the Megatron environment's hub
    cache. More than one snapshot there is refused: name the one to use."""
    if snapshot is None:
        root = oracle_config.megatron_env() / "hf_home" / "hub" / "models--gpt2" / "snapshots"
        found = sorted(path.parent for path in root.glob("*/vocab.json"))
        if len(found) != 1:
            raise ConfigError(
                f"expected one gpt2 snapshot with vocab.json in the environment's hub cache, "
                f"found {len(found)}; name one with --tokenizer-snapshot"
            )
        folder = found[0]
    else:
        folder = Path(os.path.abspath(snapshot))
    vocab, merges = folder / "vocab.json", folder / "merges.txt"
    for path in (vocab, merges):
        if not path.is_file():
            raise ConfigError(f"the tokenizer snapshot holds no {path.name}: {folder.name}")
    return {"type": "GPT2BPETokenizer", "vocab_file": vocab, "merges_file": merges}


def path_text(path: Path, folder: Path) -> str:
    """A path as a config file in ``folder`` holds it: under the Megatron
    environment it starts with the variable's name, else it is relative to the
    folder. No absolute path is written."""
    path = Path(os.path.abspath(path))
    env = os.environ.get(oracle_config.ENV_VARIABLE)
    if env:
        env_path = Path(os.path.abspath(env))
        if path == env_path or env_path in path.parents:
            return _ENV_TEXT + "/" + path.relative_to(env_path).as_posix()
    return Path(os.path.relpath(path, folder)).as_posix()


# --- Seeds and segment configs --------------------------------------------------------------


def parse_seed_overrides(items: list[str]) -> dict[str, int]:
    """``RUN=N`` pairs of the command line."""
    out: dict[str, int] = {}
    for item in items:
        run, sep, value = item.partition("=")
        if not sep or not run or not value.isdigit():
            raise ConfigError(f"a seed override is RUN=N with N a whole number, got {item!r}")
        if run in out:
            raise ConfigError(f"two seed overrides for run {run!r}")
        out[run] = int(value)
    return out


def resolve_seeds(spec: FixtureSpec, overrides: dict[str, int]) -> dict[str, int]:
    """The ``random_seed`` of every run: the spec's, or the override. A run that
    copies another run's settings has that run's seed, override included, and
    takes no override of its own."""
    for run in overrides:
        if run not in spec.runs:
            raise ConfigError(f"seed override for {run!r}: the spec has no such run")
        if spec.runs[run].seed_of != run:
            raise ConfigError(
                f"seed override for {run!r}: the run has the seed of run "
                f"{spec.runs[run].seed_of!r}; override that one"
            )
    return {
        name: overrides.get(run.seed_of, spec.runs[run.seed_of].random_seed)
        for name, run in spec.runs.items()
    }


def segment_raw(
    spec: FixtureSpec,
    run: RunPlan,
    segment: SegmentPlan,
    *,
    seed: int,
    fixture: Path,
    folder: Path,
    tokenizer: dict,
    shards_from: str = SHARDS_FROM_PIPELINE,
    outputs: bool = True,
) -> dict:
    """The oracle's segment config (docs/oracle.md, "The segment config") for one
    segment of the spec, as the file ``folder/config.json`` holds it."""
    raw = {
        "run": run.name,
        "segment": segment.name,
        "start_kind": segment.start_kind,
        "shards": [
            path_text(shard_prefix(fixture, segment, ref, shards_from), folder) for ref in segment.shards
        ],
        "tokenizer": {
            "type": tokenizer["type"],
            "vocab_file": path_text(tokenizer["vocab_file"], folder),
            "merges_file": path_text(tokenizer["merges_file"], folder),
        },
        "settings": {
            "dataloader_type": run.dataloader_type,
            "data_sharding": run.data_sharding,
            "micro_batch_size": run.micro_batch_size,
            "global_batch_size": run.global_batch_size,
            "sequence_length": run.sequence_length,
            "add_extra_token_to_sequence": spec.add_extra_token_to_sequence,
            "random_seed": seed,
            "split": spec.split,
            "train_iters": segment.train_iters,
        },
        "sampler_file": None,
        "launches": [
            {
                "world_size": launch.world_size,
                "data_parallel_size": launch.data_parallel_size,
                "consumed_at_start": launch.consumed_at_start,
                "step_at_start": launch.step_at_start,
                "steps": launch.steps,
            }
            for launch in segment.launches
        ],
        "cache_dir": CACHE_FOLDER,
        "step_record": {
            "lowered": [
                {"step": step, "last_sample_tokens_read": tokens} for step, tokens in segment.lowered
            ],
            "checkpoints": [{"name": name, "step": step} for name, step in segment.checkpoints],
        },
    }
    if segment.parent_checkpoint is not None:
        raw["parent_checkpoint"] = segment.parent_checkpoint
    if outputs:
        raw["outputs"] = {
            "digests": DIGESTS_FILE,
            "run_record": RUN_RECORD_FILE,
            "step_record": STEP_RECORD_FILE,
            "log": LOG_FILE,
            "summary": SUMMARY_FILE,
        }
    return raw


@dataclass(frozen=True)
class Planned:
    """One segment ready to run: where it goes and what the oracle is asked."""

    run: RunPlan
    plan: SegmentPlan
    folder: Path
    raw: dict
    segment: oracle_config.Segment


def select_runs(spec: FixtureSpec, names: list[str] | None) -> list[RunPlan]:
    if names is None:
        return list(spec.runs.values())
    unknown = [name for name in names if name not in spec.runs]
    if unknown or len(set(names)) != len(names) or not names:
        raise ConfigError(f"--runs names runs of the spec, each once; the spec has {list(spec.runs)}")
    return [run for name, run in spec.runs.items() if name in names]


def plan_segments(
    spec: FixtureSpec,
    runs: list[RunPlan],
    seeds: dict[str, int],
    *,
    fixture: Path,
    root: Path,
    tokenizer: dict,
    shards_from: str = SHARDS_FROM_PIPELINE,
    outputs: bool = True,
) -> list[Planned]:
    """The segment config of every segment of the given runs, parsed and checked.
    Segment ``E/seg-2`` goes to ``root/E/seg-2``. Nothing is written."""
    planned = []
    for run in runs:
        for segment in run.segments:
            folder = root / run.name / segment.folder
            raw = segment_raw(
                spec,
                run,
                segment,
                seed=seeds[run.name],
                fixture=fixture,
                folder=folder,
                tokenizer=tokenizer,
                shards_from=shards_from,
                outputs=outputs,
            )
            planned.append(
                Planned(run=run, plan=segment, folder=folder, raw=raw, segment=oracle_config.parse_segment(raw, folder))
            )
    return planned


# --- The cut of a lowered step ----------------------------------------------------------------


def step_cut(rows: list[dict], *, segment: str, step: int, sample_pos: int, covered: int) -> dict:
    """Where a lowered step's coverage ends, read off the oracle's own log (the
    spec's rule at ``[runs.E.step_record]``): the step's last sample is the one at
    ``sample_pos``; its first ``covered`` tokens stay; the row that holds the next
    token is cut there, and no row after it is kept."""
    found = [
        sample
        for sample in log_rows.samples_of(rows)
        if (sample["segment"], sample["step"], sample["sample_pos"]) == (segment, step, sample_pos)
    ]
    if len(found) != 1:
        raise OracleError(f"the log of {segment!r} holds no sample {sample_pos} of step {step}")
    cut = log_rows.cut_sample(found[0], covered)
    return {
        "segment": segment,
        "step": step,
        "sample_pos": sample_pos,
        "covered_tokens": covered,
        "sample_tokens": cut["sample_tokens"],
        "uncovered_from": cut["cut"],
        "cut_falls_inside_a_row": not cut["starts_a_row"],
        "covered_rows": cut["covered_rows"],
        "log_lines": found[0]["rows"],
    }


# --- The command ------------------------------------------------------------------------------


def _new_or_empty(path: Path, what: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ConfigError(f"{what} must be a new or empty folder: {path}")


def run_fixture(
    spec_path: str | Path,
    fixture: str | Path,
    out: str | Path,
    *,
    seed_overrides: dict[str, int] | None = None,
    runs: list[str] | None = None,
    shards_from: str = SHARDS_FROM_PIPELINE,
    tokenizer_snapshot: str | Path | None = None,
) -> dict:
    """Run every segment of the spec's runs (or of the runs named) over a built
    fixture and write the output folder. Returns the overview that is also
    written to ``fixture_runs.json``."""
    spec = load_spec(spec_path)
    fixture = Path(os.path.abspath(fixture))
    out = Path(os.path.abspath(out))
    if not (fixture / "pipeline").is_dir():
        raise ConfigError(f"not a built fixture (no pipeline/ folder): {fixture}")
    _new_or_empty(out, "the output folder")
    overrides = dict(seed_overrides or {})
    seeds = resolve_seeds(spec, overrides)
    selected = select_runs(spec, runs)
    planned = plan_segments(
        spec,
        selected,
        seeds,
        fixture=fixture,
        root=out / RUNS_FOLDER,
        tokenizer=find_tokenizer(tokenizer_snapshot),
        shards_from=shards_from,
    )
    starts = oracle_config.check_starts([item.segment for item in planned])

    from oracle import pipeline  # imports Megatron

    for item in planned:
        oracle_config.write_json(item.folder / CONFIG_FILE, item.raw)

    segments, cuts = [], []
    for item in planned:
        summary = pipeline.run_segment(item.segment)
        expected_class = spec.sampler_classes[item.run.dataloader_type]
        if summary["sampler_class"] != expected_class:
            raise OracleError(
                f"segment {item.plan.name!r}: the oracle drove {summary['sampler_class']}, the "
                f"spec names {expected_class}"
            )
        with open(item.folder / DIGESTS_FILE, encoding="utf-8") as f:
            index = json.load(f)
        relative = Path(RUNS_FOLDER) / item.run.name / item.plan.folder
        segments.append(
            {
                "run": item.run.name,
                "segment": item.plan.name,
                "folder": relative.as_posix(),
                "random_seed": seeds[item.run.name],
                "start": starts[item.plan.name],
                "mixture": item.plan.mixture,
                "shards": [
                    {
                        "position": position,
                        "name": ref.name,
                        "layout": ref.layout,
                        "tokenizer_input": Path(os.path.relpath(tokenizer_input(fixture, ref), out)).as_posix(),
                        "documents": index["shards"][position]["documents"],
                        "samples": index["shards"][position]["samples"],
                    }
                    for position, ref in enumerate(item.plan.shards)
                ],
                "dataset_samples": summary["dataset_samples"],
                "first_step": item.plan.first_step,
                "last_step": item.plan.last_step,
                "samples_read": summary["samples_read"],
                "rows_written": summary["rows_written"],
            }
        )
        if item.plan.lowered:
            rows = positions.read_log([item.folder / LOG_FILE])
            for step, tokens in item.plan.lowered:
                cuts.append(
                    step_cut(
                        rows,
                        segment=item.plan.name,
                        step=step,
                        sample_pos=item.run.global_batch_size - 1,
                        covered=tokens,
                    )
                )

    overview = {
        "format": RUNS_FORMAT,
        "spec": Path(os.path.relpath(os.path.abspath(spec.path), out)).as_posix(),
        "spec_sha256": file_sha256(spec.path),
        "fixture": Path(os.path.relpath(fixture, out)).as_posix(),
        "shards_from": shards_from,
        "seeds": {run.name: seeds[run.name] for run in selected},
        "seed_overrides": overrides,
        "segments": segments,
        "step_cuts": cuts,
    }
    oracle_config.write_json(out / OVERVIEW_FILE, overview)
    return overview


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run every run segment of the fixture spec over a built fixture and write "
        "its config, cache folder, run record, step record and oracle log."
    )
    parser.add_argument("--spec", required=True, help="the fixture spec (TOML)")
    parser.add_argument("--fixture", required=True, help="the folder a fixture build wrote")
    parser.add_argument("--out", required=True, help="the output folder; new or empty")
    parser.add_argument(
        "--seed", action="append", default=[], metavar="RUN=N",
        help="use random_seed N for a run instead of the spec's; repeatable",
    )
    parser.add_argument("--runs", help="comma-separated run names; default: every run of the spec")
    parser.add_argument(
        "--shards-from", choices=SHARDS_FROM, default=SHARDS_FROM_PIPELINE,
        help="take shard files from pipeline/<layout>/shards (default) or from mixtures/<mixture>",
    )
    parser.add_argument("--tokenizer-snapshot", help="folder with vocab.json and merges.txt")
    args = parser.parse_args(argv)
    # Megatron reports through the logging module; its lines go to standard error.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        overview = run_fixture(
            args.spec,
            args.fixture,
            args.out,
            seed_overrides=parse_seed_overrides(args.seed),
            runs=args.runs.split(",") if args.runs else None,
            shards_from=args.shards_from,
            tokenizer_snapshot=args.tokenizer_snapshot,
        )
    except OracleError as e:
        print(json.dumps({"error": type(e).__name__, "detail": str(e)}), file=sys.stderr)
        return 2
    print(json.dumps(overview, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
