"""Register the runs ``replica build-runs`` wrote, through the tool's own command line
(docs/interfaces.md 7; plan 3.3 "Training").

One call of ``waterwhite register-run`` per segment, in the spec's run order, each as
a subprocess: a segment's parent checkpoint is registered by an earlier segment, so
the order is part of the call. The store is made and the mixtures are registered by
the caller (the build of the fixture's store registers them under the spec's names);
this module only starts the command.

The command is built by the tool; where it is absent ``register_run_available``
says so and nothing is started.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .errors import RunError
from .run_plan import RunsPlan, SegmentRun
from .runs import segment_paths, tool_source

COMMAND = "register-run"


def default_command() -> list[str]:
    """The tool's command line as a program: this interpreter, with the tool's own
    ``cli`` module. A cyclic segment needs torch there (the shuffle is computed by
    calling it), so the interpreter that runs this must be one that has it."""
    return [sys.executable, "-m", "waterwhite.cli"]


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (tool_source(), env.get("PYTHONPATH", "")) if p)
    return env


def register_argv(store: Path, runs_folder: Path, segment: SegmentRun, *,
                  command: Sequence[str] | None = None) -> list[str]:
    """The command line that registers one segment: the run record, its step record,
    its kept cache folder, the mixture it trains on and, for a segment that starts
    from a checkpoint, that checkpoint."""
    paths = segment_paths(runs_folder, segment)
    argv = [*(command or default_command()), COMMAND,
            "--store", str(store),
            "--run-record", str(paths["run_record"]),
            "--steps", str(paths["step_record"]),
            "--cache", str(paths["cache"]),
            "--mixture", segment.mixture]
    if segment.parent_checkpoint is not None:
        argv += ["--parent-checkpoint", segment.parent_checkpoint]
    return argv


def register_run_available(command: Sequence[str] | None = None) -> bool:
    """Whether the tool's command line has ``register-run``: its ``--help`` exits 0.
    A command line that does not start at all is an error, not an absent command."""
    base = [*(command or default_command())]
    tool = subprocess.run([*base, "--help"], capture_output=True, text=True,
                          env=_environment(), timeout=120)
    if tool.returncode != 0:
        raise RunError(f"the tool's command line does not start: {tool.stderr.strip()[-300:]}")
    done = subprocess.run([*base, COMMAND, "--help"], capture_output=True, text=True,
                          env=_environment(), timeout=120)
    return done.returncode == 0


def register_runs(store: Path, runs_folder: Path, plan: RunsPlan, *,
                  runs: Sequence[str] | None = None, command: Sequence[str] | None = None,
                  runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[dict]:
    """Register every segment of ``plan`` (or of the runs named) that ``runs_folder``
    holds, in the plan's order. Returns what each command printed, parsed. A command
    that exits other than 0 stops the registration with its own words."""
    printed = []
    for segment in plan.segments:
        if runs is not None and segment.run not in runs:
            continue
        argv = register_argv(Path(store), Path(runs_folder), segment, command=command)
        done = runner(argv, capture_output=True, text=True, env=_environment())
        if done.returncode != 0:
            raise RunError(f"register-run of {segment.name!r} exited with code {done.returncode}: "
                           f"{(done.stderr or done.stdout).strip()}")
        try:
            printed.append(json.loads(done.stdout))
        except ValueError as exc:
            raise RunError(f"register-run of {segment.name!r} printed no JSON: "
                           f"{done.stdout[:200]!r}") from exc
    return printed
