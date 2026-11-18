"""The runs of the table's stores: built by the replica's run wrapper, registered
through the tool, and read back from what the tool prints (plan 3.3 "Training",
3.5, 5.4; docs/interfaces.md sections 7, 8.1 and 8.2).

Three parts, kept apart so that what can be checked in seconds is:

- the plan of a registration (``segments_of``, ``record_variant``): which segments a
  store registers, in which order, with which mixture and parent checkpoint, and how
  the starting states of Check 2 change a run record. Pure, written from the fixture
  spec and fixture/table/README.md section 7, and tested on its own (``test_runs.py``);
- the commands (``capture_missing``, ``build_runs``, ``register_run``): the replica's
  run command writes one folder per run, the tool's ``register-run`` registers one
  segment from it. Both are found by asking: where either command does not exist the
  rows that need a registered run are skipped with "not built yet: run capture",
  and start to run on the commit that adds the second one, with no edit here;
- the oracle's own runs (``oracle_runs``), the grading side of Check 1: the oracle's
  log is read from a run of ``oracle/fixture_runs.py`` over the same build. The tool
  and the replica never read it.

The one thing this module has to assume is the flags of the replica's run command
(``build_runs_arguments``), as docs/replica.md section 13 gives them. A command that exists and refuses these flags is an error of the run, never
"not built yet": a seam between two parts must be seen.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"
ORACLE_RUNS = REPO / "oracle" / "fixture_runs.py"

PROBE_TIMEOUT = 60.0     # seconds, for a command that only prints its help
BUILD_TIMEOUT = 900.0    # seconds, for the replica's run command
REGISTER_TIMEOUT = 300.0  # seconds, for one segment
ORACLE_TIMEOUT = 600.0   # seconds, for the oracle's seven segments

# The runs each store of the table registers (fixture/table/README.md section 4), in
# the order a registration needs them: a segment is registered after the checkpoint
# it starts from.
RUNS_OF_STORE: dict[str, tuple[str, ...]] = {"main": ("E", "D", "Y", "K"), "rr": ("E-rr",)}

# The runs of Check 2 whose files a starting state withholds or changes.
WITHHELD_RUNS = ("E", "D", "Y")
CYCLIC_RUN = "K"
STATES = ("c2.i", "c2.ii", "c2.iii", "c2.iv")

# The settings a replay rebuilds the index from, withheld in the state c2.iii.
REPLAY_SETTINGS = ("random_seed", "split", "train_iters", "train_samples")


class RunWorldError(Exception):
    """The run world could not be built. An error of the run, never a skip."""


# --- The plan of a registration (pure) ---------------------------------------------------


@dataclass(frozen=True)
class SegmentPlan:
    run: str
    name: str
    mixture: str
    parent_checkpoint: str | None

    @property
    def stem(self) -> str:
        """The segment's name with the slash replaced by a dash: the stem of its files."""
        return self.name.replace("/", "-")


@functools.cache
def spec() -> dict:
    with open(SPEC, "rb") as f:
        return tomllib.load(f)


def segments_of(fixture_spec: Mapping, runs: Sequence[str]) -> list[SegmentPlan]:
    """The segments of ``runs`` in registration order, each with the mixture it trains
    on and the checkpoint it starts from, as the spec states them. A segment whose
    parent checkpoint no earlier segment's step record registers is an error: the
    store would be asked to start a run from a checkpoint it does not hold."""
    registered: set[str] = set()
    out: list[SegmentPlan] = []
    for run in runs:
        held = fixture_spec["runs"][run]
        for name, segment in held["segments"].items():
            parent = segment.get("parent_checkpoint")
            if parent is not None and parent not in registered:
                raise RunWorldError(
                    f"the segment {name} starts from {parent}, which no segment registered "
                    f"before it registers (registered: {sorted(registered)})")
            out.append(SegmentPlan(run, name, segment["mixture"], parent))
            registered |= {c["name"] for c in held["step_record"][name]["checkpoints"]}
    return out


def segment_names(fixture_spec: Mapping, run: str) -> list[str]:
    """The names of the segments of one run, in the spec's order. Unlike ``segments_of`` it
    asks nothing of the checkpoint a segment starts from: run K starts from a checkpoint of
    run E, and its rows are read from its own segment alone."""
    return list(fixture_spec["runs"][run]["segments"])


def lowered_samples(fixture_spec: Mapping, run: str) -> dict[tuple[str, int, int], int]:
    """The samples the operator's step record lowers, as (segment, step, sample_pos) ->
    tokens read: the last sample of each step the record names (docs/interfaces.md 8.2;
    the spec's cut rule: the sample at ``sample_pos`` global batch size - 1)."""
    held = fixture_spec["runs"][run]
    batch = held.get("global_batch_size", fixture_spec["runs"][held.get("same_settings_as", run)]
                     ["global_batch_size"])
    return {(name, row["step"], batch - 1): row["last_sample_tokens_read"]
            for name, record in held["step_record"].items() for row in record["steps"]
            if "last_sample_tokens_read" in row}


@dataclass(frozen=True)
class Files:
    """What the replica writes for one segment (a folder per run, a stem per segment)."""

    run_record: Path
    steps: Path
    cache: Path


def files_of(runs_dir: Path, segment: SegmentPlan) -> Files:
    folder = Path(runs_dir) / segment.run
    return Files(folder / f"{segment.stem}.run.json", folder / f"{segment.stem}.steps.json",
                 folder / f"{segment.stem}.cache")


def record_variant(state: str, run: str, record: Mapping) -> tuple[dict, bool]:
    """A run record and whether its cache folder is handed over, as the starting
    state ``state`` of Check 2 has them (fixture/table/README.md section 7). The
    record given is not changed. ``unmutated`` and any state that leaves a run alone
    return it as it is, with its cache."""
    changed = copy.deepcopy(dict(record))
    settings = changed.setdefault("settings", {})
    if state == "c2.i" and run in WITHHELD_RUNS:
        return changed, False  # the cache files only; settings, seeds and digests kept
    if state == "c2.ii" and run in WITHHELD_RUNS:
        if not isinstance(settings.get("random_seed"), int):
            raise RunWorldError(f"the run record of {run} states no seed to change")
        settings["random_seed"] += 1
        return changed, False
    if state == "c2.iii" and run in WITHHELD_RUNS:
        for key in REPLAY_SETTINGS:
            settings[key] = None
        changed["cache_digests"] = []
        return changed, False
    if state == "c2.iv" and run == CYCLIC_RUN:
        if not changed.get("shuffle_digests"):
            raise RunWorldError("the run record of run K holds no shuffle digests to remove")
        changed["shuffle_digests"] = []
        return changed, True  # its cache files are kept
    return changed, True


# --- The commands -----------------------------------------------------------------------------


def environment(root: Path | None = None) -> dict[str, str]:
    """The environment of a command: the repository and the tool on the path, and the
    test-time Megatron environment where one is given."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO), str(SRC)])}
    if root is not None:
        env.update({"WATERWHITE_MEGATRON_ENV": str(root),
                    "HF_HOME": str(root / "hf_home"), "HF_HUB_OFFLINE": "1"})
    return env


def megatron_python(root: Path) -> Path:
    return root / "venv" / "bin" / "python"


def _ask(command: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                          env=dict(env), cwd=REPO)


def _is_absent(done, command: str) -> bool:
    """Whether a help request ended because the command does not exist: the usage error
    of a parser that does not know the word (replica: exit code 2, tool: the error
    "usage", both naming the word as an invalid choice). Any other failure, an import
    error among them, is not "absent": it is an error of the run, and the rows would
    otherwise be skipped for a command that is there and broken."""
    if done.returncode == 0:
        return False
    if "invalid choice" in done.stderr and f"'{command}'" in done.stderr:
        return True
    raise RunWorldError(f"asking for the help of {command} ended with exit code "
                        f"{done.returncode}: {done.stderr[-400:]}")


@functools.cache
def capture_missing() -> str | None:
    """Why the runs cannot be built and registered on this commit, or None when both
    commands exist: the replica's run command and the tool's ``register-run``. Found by
    asking each for its help."""
    env = environment()
    missing = []
    if _is_absent(_ask([sys.executable, "-m", "replica", "build-runs", "--help"], env),
                  "build-runs"):
        missing.append("the replica has no command build-runs")
    if _is_absent(_ask([sys.executable, "-m", "waterwhite.cli", "register-run", "--help"], env),
                  "register-run"):
        missing.append("the tool has no command register-run")
    return "; ".join(missing) or None


def build_runs_arguments(spec_file: Path, build: Path, runs_dir: Path,
                         runs: Sequence[str]) -> list[str]:
    """The flags this folder uses for the replica's run command: the new folder the runs
    go into, the folder a tokenized fixture build wrote, the spec and the runs to make
    (a store's build holds the layout of its own runs only). They are those of
    docs/replica.md section 13; the command's own help is the contract, and this is the one place
    to change when it says otherwise."""
    return ["build-runs", "--out", str(runs_dir), "--fixture-build", str(build),
            "--spec", str(spec_file), "--runs", ",".join(runs)]


def _run(command: Sequence[str], env: Mapping[str, str], timeout: float, what: str):
    done = subprocess.run(list(command), capture_output=True, text=True, timeout=timeout,
                          env=dict(env), cwd=REPO)
    if done.returncode != 0:
        raise RunWorldError(f"{what} ended with exit code {done.returncode}: "
                            f"{done.stderr[-600:]}")
    return done


def build_runs(spec_file: Path, build: Path, runs_dir: Path, runs: Sequence[str],
               root: Path) -> None:
    """Run the replica's run wrapper over a fixture build, in the Megatron environment
    (it imports torch). ``runs_dir`` must be new."""
    command = [str(megatron_python(root)), "-m", "replica",
               *build_runs_arguments(spec_file, build, runs_dir, runs)]
    _run(command, environment(root), BUILD_TIMEOUT, "the replica's command build-runs")


def register_arguments(store: Path, segment: SegmentPlan, files: Files, *, cache: bool,
                       run_record: Path | None = None) -> list[str]:
    """``waterwhite register-run`` for one segment (docs/interfaces.md section 7)."""
    out = ["register-run", "--store", str(store),
           "--run-record", str(run_record or files.run_record),
           "--steps", str(files.steps), "--mixture", segment.mixture]
    if cache:
        out += ["--cache", str(files.cache)]
    if segment.parent_checkpoint is not None:
        out += ["--parent-checkpoint", segment.parent_checkpoint]
    return out


def register_run(store: Path, segment: SegmentPlan, files: Files, root: Path, *, cache: bool = True,
                 run_record: Path | None = None) -> dict:
    """Register one segment. The interpreter is the Megatron environment's: a cyclic
    segment needs torch to compute its shuffle, and where torch cannot be imported the
    tool refuses the registration by name. Returns what the command printed."""
    command = [str(megatron_python(root)), "-m", "waterwhite.cli",
               *register_arguments(store, segment, files, cache=cache, run_record=run_record)]
    done = _run(command, environment(root), REGISTER_TIMEOUT,
                f"register-run of {segment.name}")
    try:
        printed = json.loads(done.stdout)
    except ValueError:
        raise RunWorldError(f"register-run of {segment.name} printed no JSON document: "
                            f"{done.stdout[:200]!r}") from None
    if not isinstance(printed, dict) or printed.get("segment") != segment.name:
        raise RunWorldError(f"register-run of {segment.name} printed {printed!r}")
    return printed


def replay(run_record: Path, out: Path, root: Path) -> dict:
    """``waterwhite replay --run-record PATH --out DIR``: the index arrays the replay of the
    join derives from a run record and its shards, written to the folder ``out``. The command
    writes no store. Returns what it printed:
    ``{"segment", "arrays": [{"kind", "shard", "file", "digest", "recorded_digest",
    "equal"}]}``. A record the replay cannot be made from is refused by the command, and
    that is an error of the run here."""
    command = [str(megatron_python(root)), "-m", "waterwhite.cli", "replay",
               "--run-record", str(run_record), "--out", str(out)]
    done = _run(command, environment(root), REGISTER_TIMEOUT, f"replay of {run_record.name}")
    try:
        printed = json.loads(done.stdout)
    except ValueError:
        raise RunWorldError(f"replay of {run_record.name} printed no JSON document: "
                            f"{done.stdout[:200]!r}") from None
    if not isinstance(printed, dict) or not isinstance(printed.get("arrays"), list):
        raise RunWorldError(f"replay of {run_record.name} printed {printed!r}")
    return printed


def register_runs(store: Path, runs_dir: Path, runs: Sequence[str], root: Path, *,
                  state: str = "unmutated", scratch: Path | None = None) -> dict[str, dict]:
    """Register every segment of ``runs`` in order, the run records and caches as the
    starting state ``state`` has them. A changed run record is written under
    ``scratch`` and the replica's own file is left as it is. Returns, per segment, what
    ``register-run`` printed."""
    results = {}
    for segment in segments_of(spec(), runs):
        files = files_of(runs_dir, segment)
        record_path, cache = None, True
        if state != "unmutated":
            with open(files.run_record, encoding="utf-8") as f:
                record = json.load(f)
            changed, cache = record_variant(state, segment.run, record)
            if changed != record:
                if scratch is None:
                    raise RunWorldError("a starting state changes a run record: give a folder for it")
                record_path = Path(scratch) / segment.run / files.run_record.name
                record_path.parent.mkdir(parents=True, exist_ok=True)
                # The shards of a record are relative to its folder (docs/interfaces.md 8.1):
                # the changed record sits where the original does, in a folder of its own,
                # with the shard folder beside it.
                shards = files.run_record.parent / "shards"
                link = record_path.parent / "shards"
                if shards.exists() and not link.exists():
                    link.symlink_to(shards, target_is_directory=True)
                record_path.write_text(json.dumps(changed, indent=2), encoding="utf-8")
        results[segment.name] = register_run(store, segment, files, root, cache=cache,
                                             run_record=record_path)
    return results


def verified_joins(store: Path) -> list[tuple[str, str, str]]:
    """The joins a store lists as verified (``init --verified-join`` writes them when the
    store is made; nothing writes them later, docs/cli.md). Read here and never written:
    a store whose build listed none is the build's fault, and this folder does not supply
    what the replica, standing in for the team's set-up, did not."""
    with closing(sqlite3.connect(store)) as db:
        return [tuple(row) for row in db.execute(
            "SELECT release_tag, sampler_type, sampler_class FROM verified_join "
            "ORDER BY sampler_type")]


def snapshot_store(store: Path, target: Path) -> Path:
    """A copy of a store file, taken with SQLite's own backup, beside the original: a
    store's paths are relative to its folder, so the copy must stay in it."""
    with closing(sqlite3.connect(store)) as source, closing(sqlite3.connect(target)) as copy_:
        source.backup(copy_)
    return target


# --- The oracle's own runs: the grading side of Check 1 ------------------------------------


def oracle_arguments(build: Path, out: Path, runs: Sequence[str]) -> list[str]:
    """The arguments of ``oracle/fixture_runs.py`` (docs/oracle.md, "Fixture runs"): the spec,
    the fixture build, the new folder for the runs, and the runs of the store the build was
    made for. Without ``--runs`` the oracle runs every run of the spec, and it refuses a
    build that holds the shards of one layout only."""
    return ["--spec", str(SPEC), "--fixture", str(build), "--out", str(out),
            "--runs", ",".join(runs)]


def oracle_runs(build: Path, out: Path, root: Path, runs: Sequence[str]) -> Path:
    """Run ``oracle/fixture_runs.py`` over a fixture build into the new folder ``out``
    and return the folder of its runs (docs/oracle.md, "Fixture runs"). The oracle is
    started as a process of its own and imports nothing of the tool."""
    command = [str(megatron_python(root)), "-I", str(ORACLE_RUNS),
               *oracle_arguments(build, out, runs)]
    _run(command, environment(root), ORACLE_TIMEOUT, "oracle/fixture_runs.py")
    return Path(out) / "runs"


def oracle_log(runs: Path, segment: str) -> list[dict]:
    """The oracle's log of one segment: one row per line (docs/interfaces.md 8.3)."""
    path = Path(runs) / segment / "oracle_log.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line]
