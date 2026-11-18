"""What the tool and the machine can do today, found out by asking them.

A list of what is "not built yet" goes stale: the output it names appears and the
rows stay skipped. Here the three outputs of ``show`` that Checks 3 and 6 need are
looked for by a small probe through the command line, once per run, so that the
rows start to run on the commit that prints them, with no edit in this folder:

1. the rows of a named map (``show SHARD --map NAMESPACE/DERIVATION``);
2. ``drops`` on every ``show ARTIFACT``;
3. ``append_eod`` and ``eod_in_span`` on every binding.

Two forms of the runs are asked the same way (the forms of Checks 1 and 2): ``show SEGMENT --trace`` and ``replay``. A flag or a command the
parser does not know ends with the usage error, naming the word; a form that exists
and is asked about a name it does not hold is refused by name (``request refused``);
any other ending is an error of the probe, never "not built yet".

The probe registers a two-line JSONL file in a scratch store with ``init`` and
``recover file``, which exist, and reads ``show`` of it. ``classify_show`` is the
pure half and is tested on documents written by hand (``test_probe.py``).

The second thing asked is whether the machine can build the sandbox of the
confined harness (docs/interfaces.md section 5.3, ``find_sandbox``). Only a log
written by the confined harness establishes a bound (Decision 13), so a row that
rests on a bound is not run where there is no sandbox: it is skipped with the
tool's own reason and counted as not run, never as a pass.
"""

from __future__ import annotations

import functools
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

import table

SRC = Path(__file__).resolve().parents[2] / "src"
PROBE_TIMEOUT = 30.0  # seconds, for one command

# The words of the rows that wait (the entries of ``acceptance_world.NOT_BUILT``).
REASONS = {
    table.NEED_SHOW_MAP_ROWS: "show prints the rows of no map but the one a reader gets "
                              "first, and those one document at a time",
    table.NEED_SHOW_DROPS: "show prints a record's drops without the artifact each was "
                           "recorded under, and no drops of an artifact",
    table.NEED_SHOW_MAP_SETTINGS: "show prints no append-eod setting of a map",
    table.NEED_SHOW_TRACE: "show has no --trace form, so the rows the join wrote for a "
                           "segment cannot be read through the command line",
    table.NEED_REPLAYED_ARRAYS: "the command line has no replay command, so the arrays a "
                                "replay derives cannot be written and compared",
    table.NEED_REPORT: "the command line has no `report` command, so no rebuild report is "
                       "printed (docs/interfaces.md 7.1)",
    table.NEED_VERIFY: "the command line has no `verify` command, so no repair is verified "
                       "(docs/interfaces.md 7.3)",
    table.NEED_SUCCESSOR: "no successor can be registered: the replica has no `repair` command "
                          "(docs/interfaces.md 9.5)",
}


class ProbeError(Exception):
    """The probe could not be run. A probe that cannot run says nothing about what
    ``show`` prints, so it is an error of the run and never "not built"."""


def _cli(store: Path, *arguments: str) -> table.Outcome:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    command = [sys.executable, "-m", "waterwhite.cli", *arguments, "--store", str(store)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=PROBE_TIMEOUT,
                          env=env, cwd=store.parent)
    return table.Outcome(done.returncode, done.stdout, done.stderr)


def classify_show(shown: Mapping, selected: table.Outcome) -> dict[str, str]:
    """The needs of Checks 3 and 6 that ``show`` does not meet, each with its
    reason, from what ``show ARTIFACT`` printed for a registered JSONL file and
    from what ``show ARTIFACT --map NAMESPACE/DERIVATION`` ended with.

    A flag the command does not know is a usage error (exit 1, "usage"). A flag it
    knows, asked about an artifact that is no shard, is refused with a named error
    of its own, so any other ending is taken to be the flag read."""
    gaps = {}
    if selected.error_name() == "usage":
        gaps[table.NEED_SHOW_MAP_ROWS] = REASONS[table.NEED_SHOW_MAP_ROWS]
    if not isinstance(shown.get("drops"), list):
        gaps[table.NEED_SHOW_DROPS] = REASONS[table.NEED_SHOW_DROPS]
    bindings = shown.get("bindings")
    if not isinstance(bindings, list) or not bindings or not all(
            isinstance(held, Mapping) and "append_eod" in held and "eod_in_span" in held
            for held in bindings):
        gaps[table.NEED_SHOW_MAP_SETTINGS] = REASONS[table.NEED_SHOW_MAP_SETTINGS]
    return gaps


def classify_trace(traced: table.Outcome) -> dict[str, str]:
    """``show NAME --trace`` asked of a name that is no registered segment. A parser
    that does not know the flag ends with the usage error (exit 1). A form that exists
    refuses the name: "request refused", exit 2. Anything else is an error of the probe."""
    if traced.exit_code == 1 and traced.error_name() == "usage":
        return {table.NEED_SHOW_TRACE: REASONS[table.NEED_SHOW_TRACE]}
    if traced.exit_code == 2 and traced.error_name() == "request refused":
        return {}
    raise ProbeError("show of a name that is no segment, with --trace, ended with exit code "
                     f"{traced.exit_code} and error {traced.error_name()!r}: "
                     f"{traced.stderr[:300]}")


def classify_command(helped: table.Outcome, word: str, need: str) -> dict[str, str]:
    """The help of one command word. A parser that does not know the word ends with the
    usage error that names it as an invalid choice (exit 1); a command that exists
    prints its help (exit 0). Anything else is an error of the probe."""
    if helped.exit_code == 0:
        return {}
    if (helped.exit_code == 1 and helped.error_name() == "usage"
            and "invalid choice" in helped.stderr and f"'{word}'" in helped.stderr):
        return {need: REASONS[need]}
    raise ProbeError(f"the help of {word} ended with exit code {helped.exit_code}: "
                     f"{helped.stderr[:300]}")


def classify_replica_command(helped: table.Outcome, word: str) -> bool:
    """True where the replica's parser does not know the command word: its usage error
    (exit 2, argparse) names the word as an invalid choice. A command that exists
    prints its help (exit 0). Any other ending is an error of the probe, never "not
    built yet"."""
    if helped.exit_code == 0:
        return False
    if (helped.exit_code == 2 and "invalid choice" in helped.stderr
            and f"'{word}'" in helped.stderr):
        return True
    raise ProbeError(f"the help of the replica's {word} ended with exit code "
                     f"{helped.exit_code}: {helped.stderr[:300]}")


def _command(*arguments: str) -> table.Outcome:
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-m", "waterwhite.cli", *arguments],
                          capture_output=True, text=True, timeout=PROBE_TIMEOUT, env=env)
    return table.Outcome(done.returncode, done.stdout, done.stderr)


def _replica_command(*arguments: str) -> table.Outcome:
    root = SRC.parent
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(root)])}
    done = subprocess.run([sys.executable, "-m", "replica", *arguments], capture_output=True,
                          text=True, timeout=PROBE_TIMEOUT, env=env, cwd=root)
    return table.Outcome(done.returncode, done.stdout, done.stderr)


def repair_gaps_of(report: dict[str, str], verify: dict[str, str],
                   replica_missing: bool) -> dict[str, str]:
    """The needs of Checks 7 and 9 that are not met, from what the three probes found.
    ``report`` and ``verify`` are ``classify_command`` of the two command words, and
    ``replica_missing`` is True where the replica has no ``repair`` word. A successor
    is registered by the replica from a report, so it waits for both."""
    gaps = {**report, **verify}
    if replica_missing or table.NEED_REPORT in report:
        why = []
        if table.NEED_REPORT in report:
            why.append("the tool prints no report to build it from (`report`)")
        if replica_missing:
            why.append("the replica has no `repair` command (docs/interfaces.md 9.5)")
        gaps[table.NEED_SUCCESSOR] = "no successor can be registered: " + "; ".join(why)
    return gaps


@functools.cache
def repair_gaps() -> dict[str, str]:
    """What the rebuild's three commands (`waterwhite report`, `waterwhite verify`,
    `python -m replica repair`) lack today, found by asking each for its help."""
    return repair_gaps_of(
        classify_command(_command("report", "--help"), "report", table.NEED_REPORT),
        classify_command(_command("verify", "--help"), "verify", table.NEED_VERIFY),
        classify_replica_command(_replica_command("repair", "--help"), "repair"))


@functools.cache
def show_gaps() -> dict[str, str]:
    """``classify_show`` of what the tool prints today."""
    with tempfile.TemporaryDirectory(prefix="waterwhite-probe-") as folder:
        folder = Path(folder).resolve()
        found = folder / "probe.jsonl"
        found.write_text('{"text": "alpha beta"}\n{"text": "gamma delta"}\n', encoding="utf-8")
        store = folder / "probe.sqlite"
        steps = [("init", "--normaliser", "control-chars/1"),
                 ("recover", "file", str(found), "--name", "probe/a")]
        for step in steps:
            done = _cli(store, *step)
            if done.exit_code != 0:
                raise ProbeError(f"the probe's {step[0]} ended with exit code {done.exit_code}: "
                                 f"{done.stderr[:300]}")
        shown = _cli(store, "show", "probe/a")
        if shown.exit_code != 0:
            raise ProbeError(f"the probe's show ended with exit code {shown.exit_code}: "
                             f"{shown.stderr[:300]}")
        selected = _cli(store, "show", "probe/a", "--map", "capture/position")
        gaps = classify_show(shown.printed(), selected)
        gaps.update(classify_trace(_cli(store, "show", "probe/no-segment", "--trace")))
        gaps.update(classify_command(_command("replay", "--help"), "replay",
                                     table.NEED_REPLAYED_ARRAYS))
        return gaps


# --- The sandbox of the confined harness ---------------------------------------------------


@functools.cache
def sandbox_missing() -> str | None:
    """Why this machine cannot run the confined harness, in the tool's own words,
    or None when it can (``waterwhite.confine.find_sandbox``)."""
    from waterwhite import confine
    from waterwhite.errors import SandboxUnavailable

    try:
        confine.find_sandbox()
    except SandboxUnavailable as unavailable:
        return str(unavailable)
    return None


def generation_arguments(build) -> dict:
    """The argument that makes a store's generation jobs run in the confined harness,
    for a replica function that takes one, where the sandbox can be made. Only a log
    of the confined harness establishes a bound (Decision 13), so a store built with
    in-process logs would hold edges with no bound, and the rows that rest on one
    would be graded on a store that cannot give it. Where the sandbox cannot be
    made, the jobs run in process, and the rows that rest on a bound are not run
    (``rests_on_bound``); the store's other rows still are. A replica function with
    no such argument is called without one: its logs are the replica's own choice."""
    import inspect

    if "generation_harness" not in inspect.signature(build).parameters:
        return {}
    return {"generation_harness": "confined" if sandbox_missing() is None else "in-process"}


def rests_on_bound(question: table.Question) -> bool:
    """Whether the row's expected answer rests on an established bound (plan 2.2(g),
    3.3): a negative past a generation edge for a source the edge does not name, or
    an answer that is "may contain" because a bound is not established.

    Only a log written by the confined harness establishes a bound (Decision 13),
    and the confined harness needs a sandbox. Where there is none the row is not
    run: a "may contain" that is expected because no bound is established would
    pass for the wrong reason, and a negative cannot be had at all.

    A row is told to rest on a bound when its state is a fault on a generation edge
    or on its log (``g.*``), when it expects the condition "g" or lists generation
    edges in its scope, or when it expects "does not contain" or "may contain" of a
    file a generation job wrote or of a file that holds one. ``World`` stops a row it did not count
    here when the tool's own answer lists generation edges."""
    expect = question.expect
    failing = expect.get("failing")
    scope = expect.get("scope")
    if question.state.startswith("g."):
        return True
    if isinstance(failing, dict) and failing.get("condition") == "g":
        return True
    if isinstance(scope, dict) and scope.get("generation_edges"):
        return True
    # Of a file a generation job wrote, or one that holds one, a negative needs the
    # bound, and so does a "may contain" whose first failing link lies past a
    # generation edge: with no bound that edge's link (condition g) comes first. A
    # missing origin (condition a) fails before any edge; "contains" needs no bound.
    turns_on_bound = expect.get("answer") == "does not contain" or (
        expect.get("answer") == "may contain"
        and not (isinstance(failing, dict) and failing.get("condition") == "a"))
    return (question.kind == table.KIND_ANSWER and turns_on_bound
            and any(name in GENERATED_OR_HOLDING_ONE for name in question.artifacts))


# The files that hold what a generation job wrote: the job outputs, the releases that
# carry them, and the shard that holds T (README section 4).
GENERATED_OR_HOLDING_ONE = frozenset({
    "gen/P", "gen/T", "gen/S", "R1", "R1-swap", "E/shard-3", "E-rr/shard-3"})
