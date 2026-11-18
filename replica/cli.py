"""The replica's command line.

    python -m replica build-fixture --spec FILE --out DIR [options]
    python -m replica build-set     --set DIR  --out DIR --layout LAYOUT [options]
    python -m replica digests       --out DIR
    python -m replica build-runs    --out DIR [--fixture-build DIR] [options]
    python -m replica register-runs --store FILE --runs-folder DIR [options]
    python -m replica probe-run     --set-build DIR --store FILE --out DIR [options]
    python -m replica repair        --report FILE --historical DIR --out DIR --store FILE
                                    (--spec FILE | --set DIR) [options]

A build prints its summary as JSON on standard output. An error prints one line on
standard error and exits with 1; a wrong command line exits with 2.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from . import megatron_tokenizer
from .build import (
    HARNESS_CONFINED,
    HARNESS_IN_PROCESS,
    MIXTURE_COPY,
    MIXTURE_SYMLINK,
    BuildOptions,
    RecordChanges,
    build,
    check_build,
    tree_digests,
)
from .errors import OutputError, ReplicaError, SpecError
from .plan import (
    LAYOUT_PARTITIONED,
    LAYOUT_ROUND_ROBIN,
    LAYOUTS,
    load_fixture_spec,
    load_set,
    load_tokenizer_settings,
)

BOTH = "both"


def _pairs(values: list[str], what: str) -> dict[str, str]:
    out = {}
    for value in values:
        left, sep, right = value.partition("=")
        if not sep or left == "" or right == "":
            raise SpecError(f"{what} is written LEFT=RIGHT, got {value!r}")
        out[left] = right
    return out


def _add_build_options(parser: argparse.ArgumentParser, *, layout_required: bool) -> None:
    parser.add_argument("--out", required=True, type=Path,
                        help="the build folder; it must be new or empty")
    parser.add_argument("--layout", choices=(LAYOUT_PARTITIONED, LAYOUT_ROUND_ROBIN, BOTH),
                        required=layout_required, default=None if layout_required else BOTH)
    parser.add_argument("--no-tokenize", action="store_true",
                        help="stop before Megatron's preprocessing; no shards, no mixtures")
    parser.add_argument("--unseeded", action="store_true",
                        help="draw every shuffle from the operating system's randomness")
    parser.add_argument("--shuffle-seed", action="append", default=[], metavar="LAYOUT=N",
                        help="replace the shuffle seed of a layout")
    parser.add_argument("--jobs", type=int, default=1,
                        help="tokenizer runs at a time (default 1)")
    parser.add_argument("--tokenizer-timeout", type=float, default=900.0,
                        help="seconds one tokenizer run may take (default 900)")
    parser.add_argument("--leave-out", action="append", default=[], metavar="ID",
                        help="a record id to leave out of releases, imports and shards")
    parser.add_argument("--swap", action="append", default=[], metavar="OLD=NEW",
                        help="take record NEW wherever record OLD is selected")
    parser.add_argument("--mixture-files", choices=(MIXTURE_COPY, MIXTURE_SYMLINK),
                        default=MIXTURE_COPY)
    parser.add_argument("--release", action="append", default=None, metavar="NAME",
                        help="build this release only (repeatable; default: every release)")
    parser.add_argument("--import", action="append", default=None, metavar="NAME",
                        dest="imports",
                        help="build this import only (repeatable; default: every import)")
    parser.add_argument("--capture", action="store_true",
                        help="record the build in a store through the tool's wrappers")
    parser.add_argument("--store", type=Path, help="the store file (with --capture)")
    parser.add_argument("--list-join", action="append", default=[], choices=("single", "cyclic"),
                        metavar="SAMPLER", dest="list_joins",
                        help="a new store lists the join of the spec's release for this sampler "
                             "type as verified (repeatable; with --capture; the spec is --spec "
                             "or --tokenizer-spec): the store a run is registered in lists it")
    parser.add_argument("--name-prefix", default="",
                        help="put before every name registered in the store")
    parser.add_argument("--generation-harness", choices=(HARNESS_IN_PROCESS, HARNESS_CONFINED),
                        default=HARNESS_IN_PROCESS,
                        help="how a captured generation job runs: in this process (the "
                             "default), or confined, as a program in the tool's sandbox "
                             "(with --capture; refused where the sandbox cannot be built)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m replica", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("build-fixture", help="build the hand-checked fixture")
    fixture.add_argument("--spec", required=True, type=Path)
    fixture.add_argument("--base", type=Path,
                         help="the folder the spec's file paths are relative to "
                              "(default: three levels above the spec file)")
    _add_build_options(fixture, layout_required=False)
    generated = commands.add_parser("build-set", help="build a generated set")
    generated.add_argument("--set", required=True, type=Path, dest="set_dir",
                           help="the folder fixture/plant/generate.py wrote (it holds set.toml)")
    generated.add_argument("--tokenizer-spec", type=Path,
                           help="a TOML file with the [tokenizer] table of the build, such "
                                "as the fixture spec (needed unless --no-tokenize)")
    _add_build_options(generated, layout_required=True)
    digests = commands.add_parser("digests", help="print the SHA-256 of every file of a build")
    digests.add_argument("--out", required=True, type=Path)
    runs = commands.add_parser(
        "build-runs", help="make the fixture's runs: Megatron's cache folders, run records, step records")
    runs.add_argument("--out", required=True, type=Path, help="the runs folder; it must be new or empty")
    runs.add_argument("--fixture-build", type=Path,
                      help="a folder build-fixture wrote, tokenized (default: build one into the "
                           "runs folder's scratch folder first)")
    runs.add_argument("--spec", type=Path, help="the fixture spec (default: the repository's)")
    runs.add_argument("--runs", help="comma-separated run names (default: every run of the spec)")
    runs.add_argument("--timeout", type=float, default=600.0,
                      help="seconds the Megatron worker may take (default 600)")
    probe = commands.add_parser(
        "probe-run", help="make one run over a generated set's shards and register it")
    probe.add_argument("--set-build", required=True, type=Path, dest="set_build",
                       help="the folder build-set wrote, tokenized, with capture on")
    probe.add_argument("--store", required=True, type=Path,
                       help="the store the set was built into; it must list the sequential join")
    probe.add_argument("--out", required=True, type=Path,
                       help="the folder for the run's records; it must be new or empty")
    probe.add_argument("--layout", choices=LAYOUTS, default=LAYOUT_ROUND_ROBIN,
                       help="the layout whose shards the run reads (default round-robin)")
    probe.add_argument("--steps", type=int,
                       help="the steps of the run (default: the set's one pass, in whole steps)")
    probe.add_argument("--spec", type=Path, help="the fixture spec (default: the repository's)")
    probe.add_argument("--timeout", type=float, default=600.0,
                       help="seconds the Megatron worker may take (default 600)")
    repaired = commands.add_parser(
        "repair", help="build the repaired side (and the reference side) of a rebuild report")
    repaired.add_argument("--report", required=True, type=Path,
                          help="the rebuild report `waterwhite report` printed, as a JSON file")
    repaired.add_argument("--historical", required=True, type=Path,
                          help="the build folder the store's historical artifacts were made from")
    repaired.add_argument("--out", required=True, type=Path,
                          help="the folder of the repaired side; it must be new or empty")
    repaired.add_argument("--store", required=True, type=Path,
                          help="the historical store; the successors are registered in it")
    what = repaired.add_mutually_exclusive_group(required=True)
    what.add_argument("--spec", type=Path, help="the fixture spec the historical build used")
    what.add_argument("--set", type=Path, dest="set_dir",
                      help="the generated set the historical build used")
    repaired.add_argument("--base", type=Path,
                          help="the folder the spec's file paths are relative to")
    repaired.add_argument("--tokenizer-spec", type=Path, dest="tokenizer_spec",
                          help="a TOML file with the [tokenizer] table (a set, unless --no-tokenize)")
    repaired.add_argument("--no-tokenize", action="store_true",
                          help="stop before Megatron's preprocessing, on both sides (default: "
                               "tokenize where the historical build did)")
    repaired.add_argument("--layout", choices=(LAYOUT_PARTITIONED, LAYOUT_ROUND_ROBIN, BOTH),
                          default=None,
                          help="the layouts to build (default: the ones the historical build made)")
    repaired.add_argument("--unseeded", action="store_true")
    repaired.add_argument("--shuffle-seed", action="append", default=[], metavar="LAYOUT=N")
    repaired.add_argument("--jobs", type=int, default=1)
    repaired.add_argument("--tokenizer-timeout", type=float, default=900.0)
    repaired.add_argument("--generation-harness", choices=(HARNESS_IN_PROCESS, HARNESS_CONFINED),
                          default=None,
                          help="the harness of the historical build's jobs (default: the one "
                               "its harness logs name)")
    repaired.add_argument("--historical-name-prefix", default="", dest="historical_name_prefix",
                          help="the name prefix the historical build registered its names under")
    repaired.add_argument("--reference-set", type=Path, dest="reference_set",
                          help="the folder the set's generator wrote without the withdrawn content")
    repaired.add_argument("--reference-withdrawn", dest="reference_withdrawn",
                          help="the fixture's reference side: the withdrawn source, whose records "
                               "and everything derived from them the spec's build leaves out")
    repaired.add_argument("--reference-out", type=Path, dest="reference_out",
                          help="the folder of the reference side; it must be new or empty")
    repaired.add_argument("--sides", type=Path,
                          help="write the sides file `waterwhite verify` reads (a new file)")
    registered = commands.add_parser(
        "register-runs", help="register the runs build-runs wrote, one `waterwhite register-run` each")
    registered.add_argument("--store", required=True, type=Path)
    registered.add_argument("--runs-folder", required=True, type=Path, dest="runs_folder",
                            help="the folder build-runs wrote")
    registered.add_argument("--spec", type=Path, help="the fixture spec (default: the repository's)")
    registered.add_argument("--runs", help="comma-separated run names (default: every run of the spec)")
    return parser


def _seeds(args: argparse.Namespace) -> dict[str, int]:
    seeds = {}
    for layout, value in _pairs(args.shuffle_seed, "--shuffle-seed").items():
        if layout not in LAYOUTS:
            raise SpecError(f"--shuffle-seed names layout {layout!r}; layouts are {LAYOUTS}")
        try:
            seeds[layout] = int(value)
        except ValueError:
            raise SpecError(f"--shuffle-seed {layout}={value}: not an integer") from None
    if args.unseeded and seeds:
        raise SpecError("--unseeded and --shuffle-seed exclude each other")
    return seeds


def _options(args: argparse.Namespace) -> BuildOptions:
    layouts = LAYOUTS if args.layout == BOTH else (args.layout,)
    seeds = _seeds(args)
    return BuildOptions(
        layouts=layouts,
        seeded=not args.unseeded,
        shuffle_seeds=seeds,
        tokenize=not args.no_tokenize,
        parallel=args.jobs,
        tokenizer_timeout=args.tokenizer_timeout,
        changes=RecordChanges(leave_out=frozenset(args.leave_out),
                              swap=_pairs(args.swap, "--swap")),
        mixture_files=args.mixture_files,
        releases=None if args.release is None else tuple(args.release),
        imports=None if args.imports is None else tuple(args.imports),
        store_name_prefix=args.name_prefix,
        generation_harness=args.generation_harness,
    )


def _refuse_store_in_build_folder(store: Path, out: Path) -> None:
    """A build folder holds the build's files and nothing else: a store file
    inside it would be listed by ``digests`` and is the same bytes in no second
    build. So a store path inside the build folder is refused, before anything is
    written."""
    store_path, folder = Path(os.path.abspath(store)), Path(os.path.abspath(out))
    if store_path == folder or folder in store_path.parents:
        raise OutputError(f"the store file {store} lies inside the build folder {out}; a "
                          "build folder holds the build's files only: name a store path "
                          "outside it, for example beside it")


def _listed_joins(args: argparse.Namespace) -> tuple[tuple[str, str, str], ...]:
    """The joins ``--list-join`` names, as the spec's runs state them (plan F6)."""
    if not args.list_joins:
        return ()
    if not args.capture:
        raise SpecError("--list-join goes with --capture: a store lists its joins when it is made")
    spec = args.spec if args.command == "build-fixture" else args.tokenizer_spec
    if spec is None:
        raise SpecError("--list-join reads the spec's runs: give --spec (build-fixture) or "
                        "--tokenizer-spec (build-set)")
    from .run_plan import load_runs_plan

    return tuple(row for row in load_runs_plan(spec).verified_joins()
                 if row[1] in args.list_joins)


def _run_build(args: argparse.Namespace) -> dict:
    if args.command == "build-fixture":
        plan = load_fixture_spec(args.spec, args.base)
    else:
        # A generated set names no tokenizer; the build is given one.
        tokenizer = None
        if args.tokenizer_spec is not None:
            tokenizer = load_tokenizer_settings(args.tokenizer_spec)
        elif not args.no_tokenize:
            raise SpecError("a generated set names no tokenizer: give --tokenizer-spec FILE "
                            "(a file with a [tokenizer] table, such as the fixture spec) "
                            "or --no-tokenize")
        plan = load_set(args.set_dir, tokenizer)
    options = _options(args)
    if args.capture != (args.store is not None):
        raise SpecError("--capture and --store go together")
    joins = _listed_joins(args)
    capture = None
    if args.capture:
        # Everything the build would refuse is refused before the store file is
        # made, so that a refused build leaves no store behind.
        _refuse_store_in_build_folder(args.store, args.out)
        check_build(plan, args.out, options, capturing=True)
        # The store, the capture wrappers and the harness are imported here, and
        # only when capture is on.
        from .capture_hooks import StoreCapture

        capture = StoreCapture.open(args.store, normaliser=plan.normaliser,
                                    verified_joins=joins)
    try:
        summary = build(plan, args.out, options, capture=capture)
    finally:
        if capture is not None:
            capture.close()
    summary["out"] = str(args.out)
    return summary


def _terminate(signum, frame) -> None:
    # Leave through the normal exits, so that running tokenizer subprocesses are
    # stopped and not left behind.
    megatron_tokenizer.stop_all()
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    try:
        if args.command == "digests":
            if not args.out.is_dir():
                raise SpecError(f"{args.out} is not a folder")
            for path, digest in tree_digests(args.out).items():
                print(f"{digest}  {path}")
            return 0
        if args.command == "build-runs":
            from .fixture import default_spec_path
            from .runs import build_runs

            summary = build_runs(args.out, args.fixture_build,
                                 spec_path=args.spec or default_spec_path(),
                                 runs=args.runs.split(",") if args.runs else None,
                                 timeout=args.timeout)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        if args.command == "probe-run":
            from .probe_run import probe_run

            printed = probe_run(args.set_build, args.store, args.out, layout=args.layout,
                                steps=args.steps, spec_path=args.spec, timeout=args.timeout)
            print(json.dumps(printed, indent=2, sort_keys=True))
            return 0
        if args.command == "repair":
            from .repair import repair

            printed = repair(
                args.report, args.historical, args.out, args.store,
                spec=args.spec, set_dir=args.set_dir, tokenizer_spec=args.tokenizer_spec,
                base=args.base, layout=args.layout, tokenize=False if args.no_tokenize else None,
                seeded=not args.unseeded, shuffle_seeds=_seeds(args),
                parallel=args.jobs, tokenizer_timeout=args.tokenizer_timeout,
                generation_harness=args.generation_harness,
                historical_name_prefix=args.historical_name_prefix,
                reference_set=args.reference_set, reference_withdrawn=args.reference_withdrawn,
                reference_out=args.reference_out, sides=args.sides)
            print(json.dumps(printed, indent=2, sort_keys=True))
            return 0
        if args.command == "register-runs":
            from .fixture import default_spec_path
            from .register_runs import register_runs
            from .run_plan import load_runs_plan

            plan = load_runs_plan(args.spec or default_spec_path())
            printed = register_runs(args.store, args.runs_folder, plan,
                                    runs=args.runs.split(",") if args.runs else None)
            print(json.dumps(printed, indent=2, sort_keys=True))
            return 0
        summary = _run_build(args)
    except ReplicaError as exc:
        print(f"replica: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0
