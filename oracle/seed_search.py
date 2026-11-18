"""The seed search of the fixture spec (plan 5.1): try candidate values of each
run's ``random_seed`` on one built fixture, in one process, and report the seeds
under which every numbered condition of the spec's ``[search]`` table holds. Use
and output: docs/oracle.md, "Seed search".

    python -I oracle/seed_search.py --spec fixture/spec/fixture_spec.toml \\
        --fixture <built fixture> --out <new folder> --seeds 0:4999 --time-limit 900

For each candidate the oracle builds the run's index and reads its steps with
Megatron's own code, the log kept in memory (``pipeline.read_segment``); the
predicates of ``oracle/search_conditions.py`` are then checked on that log and
on the replica's manifests. Run E is searched first; a seed that satisfies run
E's conditions is tried on the round-robin variant, which has the same seed, and
is rejected when that variant's conditions fail. Runs D, Y and K have seeds of
their own and are searched one after another. When every run has a seed, the
fixture runs are written once with those seeds and all conditions are checked
again on the written files.

``--check-runs DIR`` checks the conditions on an output folder of
``oracle/fixture_runs.py`` instead of searching; that mode needs no Megatron.

The search only reports. A person copies the seeds and positions into the spec.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle import config as oracle_config  # noqa: E402
from oracle import fixture_runs, records, search_evidence  # noqa: E402
from oracle.digests import file_sha256  # noqa: E402
from oracle.errors import ConfigError, OracleError  # noqa: E402
from oracle.fixture_spec import FixtureSpec, RunPlan, load_spec  # noqa: E402
from oracle.search_conditions import (  # noqa: E402
    CHECKS,
    BuildEvidence,
    Evidence,
    Outcome,
    SegmentEvidence,
    check_spec_conditions,
    checks_of,
    evaluate,
)

SEARCH_FORMAT = "waterwhite.oracle-seed-search/1"
RESULT_FILE = "seed_search.json"
FRAGMENT_FILE = "positions_fragment.toml"
SCRATCH_FOLDER = "scratch"
CONFIRM_FOLDER = "confirm"

ENDED_ACCEPTED = "accepted"
ENDED_TIME = "time limit"
ENDED_RANGE = "range exhausted"

_CHECK_BY_NUMBER = {check.n: check for check in CHECKS}


# --- Small parsers ------------------------------------------------------------------------------


def parse_seed_range(text: str) -> range:
    """``FIRST:LAST``, both included."""
    first, sep, last = text.partition(":")
    if not sep or not first.isdigit() or not last.isdigit() or int(first) > int(last):
        raise ConfigError(f"--seeds is FIRST:LAST with whole numbers, FIRST not above LAST; got {text!r}")
    return range(int(first), int(last) + 1)


def parse_numbers(text: str | None, what: str) -> tuple[int, ...]:
    if not text:
        return ()
    parts = text.split(",")
    if not all(part.isdigit() for part in parts):
        raise ConfigError(f"{what} is a comma-separated list of numbers, got {text!r}")
    return tuple(int(part) for part in parts)


def repository_root(spec: FixtureSpec, base: str | Path | None) -> Path:
    """Where the spec's file paths start: the folder three levels above the spec
    file (it is kept at fixture/spec/), or the folder named."""
    if base is not None:
        return Path(os.path.abspath(base))
    return Path(os.path.abspath(spec.path)).parents[2]


# --- The TOML fragment ---------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key)


def toml_value(value) -> str:
    """One value in TOML: whole numbers, true and false, strings, lists and
    inline tables of these."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a TOML basic string
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{toml_key(str(k))} = {toml_value(v)}" for k, v in value.items()) + " }"
    raise TypeError(f"no TOML form for {value!r}")


def _log_line_text(line: dict) -> str:
    text = json.dumps(line)
    return f"'{text}'" if "'" not in text else json.dumps(text)


def positions_tables(spec: FixtureSpec, outcomes: dict[int, Outcome]) -> dict[str, dict]:
    """The outcomes arranged as the spec's ``[positions]`` tables: each table
    lists its conditions; it gets their facts and the log lines that show them.
    A fact name two conditions of one table both use is kept apart by the
    condition's number. A table with no condition (the restart of Check 9) or
    with none that was checked is left out."""
    tables: dict[str, dict] = {}
    for name, table in spec.raw.get("positions", {}).items():
        numbers = [n for n in table.get("conditions", []) if n in _CHECK_BY_NUMBER]
        if not any(n in outcomes for n in numbers):
            continue  # a table none of whose conditions was checked in this call
        facts: dict = {}
        lines: list[dict] = []
        for n in numbers:
            if n not in outcomes:
                continue
            for key, value in outcomes[n].facts.items():
                if key in facts and facts[key] != value:
                    key = f"c{n:02d}_{key}"
                facts[key] = value
            for line in outcomes[n].lines:
                if line not in lines:
                    lines.append(line)
        tables[name] = {
            "conditions": numbers,
            "holds": {str(n): outcomes[n].holds for n in numbers if n in outcomes},
            "details": {str(n): outcomes[n].detail for n in numbers if n in outcomes},
            "facts": facts,
            "log_lines": lines,
        }
    return tables


def toml_fragment(
    spec: FixtureSpec, seeds: dict[str, dict], tables: dict[str, dict], status: list[str]
) -> str:
    """The result as a TOML fragment in the shape of the spec's ``[runs]`` seeds
    and ``[positions]`` tables, for a person to copy by hand. ``seeds`` maps a
    run to ``{"seed", "accepted"}``."""
    out = [
        "# Seed search over one built fixture (oracle/seed_search.py), from the oracle's logs and",
        "# the replica's manifests only. Copy by hand into fixture/spec/fixture_spec.toml.",
        "# The layout shuffle seeds are those of the build that was searched.",
    ]
    out += [f"# {line}" for line in status]
    for run in spec.runs.values():
        if run.name not in seeds:
            continue
        entry = seeds[run.name]
        out += ["", f"[runs.{toml_key(run.name)}]"]
        note = "every condition of the run holds" if entry["accepted"] else "NOT ACCEPTED: " + entry["note"]
        out.append(f"random_seed = {entry['seed']}  # {note}")
        for other in spec.runs.values():
            if other.seed_of == run.name and other.name != run.name:
                out.append(f"# run {other.name} has the seed of run {run.name}")
    for name, table in tables.items():
        out += ["", f"[positions.{toml_key(name)}]", f"conditions = {toml_value(table['conditions'])}"]
        for n in table["conditions"]:
            if str(n) not in table["holds"]:
                out.append(f"# condition {n}: not checked in this call")
                continue
            word = "holds" if table["holds"][str(n)] else "FAILS"
            out.append(f"# condition {n} {word}: {table['details'][str(n)]}")
        for key, value in table["facts"].items():
            out.append(f"{toml_key(key)} = {toml_value(value)}")
        if table["log_lines"]:
            out.append("log_lines = [")
            out += [f"    {_log_line_text(line)}," for line in table["log_lines"]]
            out.append("]")
        else:
            out.append("log_lines = []")
    return "\n".join(out) + "\n"


def _condition_rows(outcomes: dict[int, Outcome]) -> list[dict]:
    return [
        {"n": n, "id": _CHECK_BY_NUMBER[n].id, "holds": outcome.holds, "detail": outcome.detail}
        for n, outcome in sorted(outcomes.items())
    ]


# --- The search ----------------------------------------------------------------------------------


class Search:
    """One search over one built fixture. Holds what does not change between
    candidates: the spec, the replica's evidence, the tokenizer files."""

    def __init__(
        self,
        spec: FixtureSpec,
        build: BuildEvidence,
        *,
        fixture: Path,
        out: Path,
        tokenizer: dict,
        shards_from: str,
        clock: Callable[[], float],
    ) -> None:
        self.spec = spec
        self.build = build
        self.fixture = fixture
        self.out = out
        self.tokenizer = tokenizer
        self.shards_from = shards_from
        self.clock = clock
        from oracle import pipeline  # imports Megatron

        self._pipeline = pipeline

    def own_seed_runs(self) -> list[str]:
        """The runs with a seed of their own, in the order of the spec."""
        return [run.name for run in self.spec.runs.values() if run.seed_of == run.name]

    def companions(self, run: str) -> list[str]:
        """The runs that have the seed of ``run``."""
        return [other.name for other in self.spec.runs.values() if other.seed_of == run and other.name != run]

    def read_run(self, run: RunPlan, seed: int) -> list[SegmentEvidence]:
        """Read every segment of a run under a candidate seed, logs in memory.
        Each segment gets a cache folder of its own under ``scratch/``; Megatron
        cannot build an index without writing it."""
        root = self.out / SCRATCH_FOLDER / f"seed-{seed}"
        planned = fixture_runs.plan_segments(
            self.spec,
            [run],
            {run.name: seed},
            fixture=self.fixture,
            root=root,
            tokenizer=self.tokenizer,
            shards_from=self.shards_from,
            outputs=False,
        )
        segments = []
        for item in planned:
            read = self._pipeline.read_segment(item.segment)
            segments.append(
                SegmentEvidence(
                    name=item.plan.name,
                    shards=tuple(ref.name for ref in item.plan.shards),
                    rows=read["rows"],
                    summary=read["summary"],
                    shuffle_digests=read["shuffle_digests"],
                    step_record=records.step_record(item.segment),
                    shard_documents=tuple(entry["documents"] for entry in read["shards"]),
                )
            )
        return segments

    def phase(self, run_name: str, seeds: range, deadline: float, accept: int, best: int) -> dict:
        """Search one run's seed. A candidate is accepted when the run's own
        conditions hold and then those of every run that has its seed."""
        run = self.spec.runs[run_name]
        companions = self.companions(run_name)
        numbers = [check.n for name in [run_name, *companions] for check in checks_of(name)]
        evaluated: Counter = Counter()
        held: Counter = Counter()
        failing_sets: Counter = Counter()
        kept: list[dict] = []  # the best candidates so far, with their logs and outcomes
        accepted: list[int] = []
        tried = 0
        ended_by = ENDED_RANGE
        started = self.clock()

        def with_companions(candidate: dict) -> None:
            evidence = candidate["evidence"]
            for other in companions:
                evidence.runs[other] = self.read_run(self.spec.runs[other], candidate["seed"])
                candidate["outcomes"].update(evaluate(evidence, tuple(c.n for c in checks_of(other))))
            candidate["companions"] = True

        def count(outcomes: dict[int, Outcome], only: tuple[int, ...] | None = None) -> None:
            for n, outcome in outcomes.items():
                if only is None or n in only:
                    evaluated[n] += 1
                    held[n] += outcome.holds

        for seed in seeds:
            if self.clock() >= deadline:
                ended_by = ENDED_TIME
                break
            evidence = Evidence(self.spec, self.build, {run_name: self.read_run(run, seed)})
            candidate = {"seed": seed, "evidence": evidence, "outcomes": evaluate(evidence), "companions": False}
            count(candidate["outcomes"])
            if all(outcome.holds for outcome in candidate["outcomes"].values()):
                own = tuple(candidate["outcomes"])
                with_companions(candidate)
                count(candidate["outcomes"], tuple(n for n in candidate["outcomes"] if n not in own))
            tried += 1
            fails = sorted(n for n, outcome in candidate["outcomes"].items() if not outcome.holds)
            candidate["fails"] = fails
            failing_sets[tuple(fails)] += 1
            if not fails:
                accepted.append(seed)
            kept.append(candidate)
            kept.sort(key=lambda c: (len(c["fails"]), c["seed"]))
            del kept[max(best, accept) :]  # every accepted seed stays in the report
            if len(accepted) >= accept:
                ended_by = ENDED_ACCEPTED
                break

        # When no candidate reached the runs that share the seed, read them once
        # for the best candidate, so the caller learns whether their conditions
        # hold on this build.
        if kept and companions and not any(candidate["companions"] for candidate in kept):
            first = kept[0]
            before = tuple(first["outcomes"])
            with_companions(first)
            count(first["outcomes"], tuple(n for n in first["outcomes"] if n not in before))
            failing_sets[tuple(first["fails"])] -= 1
            first["fails"] = sorted(n for n, outcome in first["outcomes"].items() if not outcome.holds)
            failing_sets[tuple(first["fails"])] += 1
            failing_sets = +failing_sets  # drop a set no candidate has any more
            kept.sort(key=lambda c: (len(c["fails"]), c["seed"]))

        return {
            "run": run_name,
            "with": companions,
            "tried": tried,
            "ended_by": ended_by,
            "seconds": round(self.clock() - started, 1),
            "accepted": accepted,
            "conditions": [
                {"n": n, "id": _CHECK_BY_NUMBER[n].id, "evaluated": evaluated[n], "held": held[n]} for n in numbers
            ],
            "failing_sets": [
                {"fails": list(fails), "candidates": candidates}
                for fails, candidates in sorted(failing_sets.items(), key=lambda item: (len(item[0]), -item[1], item[0]))[:20]
            ],
            "best": [
                {
                    "seed": candidate["seed"],
                    # no failure means the runs that share the seed were read too
                    "accepted": not candidate["fails"],
                    "fails": candidate["fails"],
                    "conditions": _condition_rows(candidate["outcomes"]),
                    # for an accepted seed: the positions and the log lines that show them
                    "positions": None if candidate["fails"] else positions_tables(self.spec, candidate["outcomes"]),
                }
                for candidate in kept
            ],
            "_kept": kept,
        }


def search(
    spec_path: str | Path,
    fixture: str | Path,
    out: str | Path,
    *,
    seeds: range,
    time_limit: float,
    runs: list[str] | None = None,
    accept: int = 1,
    best: int = 5,
    base: str | Path | None = None,
    shards_from: str = fixture_runs.SHARDS_FROM_PIPELINE,
    tokenizer_snapshot: str | Path | None = None,
    allow_reworded: tuple[int, ...] = (),
    confirm: bool = True,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[dict, str]:
    """Run the search. Returns the result document and the TOML fragment; both
    are also written into ``out``.

    ``time_limit`` is in seconds of ``clock`` and covers the candidates only:
    the import of Megatron before them and the confirming run after them are
    outside it. Each run still to be searched gets an equal share of what is
    left. Which candidates fit into the time depends on the machine; what is
    found for one candidate does not."""
    spec = load_spec(spec_path)
    reworded = check_spec_conditions(spec, allow_reworded)
    fixture = Path(os.path.abspath(fixture))
    out = Path(os.path.abspath(out))
    if not (fixture / "pipeline").is_dir():
        raise ConfigError(f"not a built fixture (no pipeline/ folder): {fixture}")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ConfigError(f"the output folder must be new or empty: {out}")
    if accept < 1 or best < 1 or time_limit <= 0:
        raise ConfigError("--accept and --best are at least 1 and --time-limit is above 0")
    build = search_evidence.load_build(spec, fixture, repository_root(spec, base))
    engine = Search(
        spec,
        build,
        fixture=fixture,
        out=out,
        tokenizer=fixture_runs.find_tokenizer(tokenizer_snapshot),
        shards_from=shards_from,
        clock=clock,
    )
    searchable = engine.own_seed_runs()
    if runs is None:
        selected = searchable
    else:
        if not runs or any(name not in searchable for name in runs) or len(set(runs)) != len(runs):
            raise ConfigError(f"--runs names runs with a seed of their own, each once: {searchable}")
        selected = [name for name in searchable if name in runs]

    deadline = clock() + time_limit
    phases = []
    for position, name in enumerate(selected):
        now = clock()
        share = max(deadline - now, 0.0) / (len(selected) - position)
        phases.append(engine.phase(name, seeds, now + share, accept, best))

    shown: dict[str, dict] = {}
    outcomes: dict[int, Outcome] = {}
    for phase in phases:
        kept = phase.pop("_kept")
        if not kept:
            continue
        first = kept[0]
        is_accepted = bool(phase["accepted"]) and first["seed"] == phase["accepted"][0]
        shown[phase["run"]] = {
            "seed": first["seed"],
            "accepted": is_accepted,
            "note": "condition(s) " + ", ".join(str(n) for n in first["fails"]) + " fail" if first["fails"] else "",
        }
        outcomes.update(first["outcomes"])
    chosen = {phase["run"]: (phase["accepted"][0] if phase["accepted"] else None) for phase in phases}
    everything = selected == searchable and all(seed is not None for seed in chosen.values())

    confirmed = None
    if everything and confirm:
        folder = out / CONFIRM_FOLDER
        fixture_runs.run_fixture(
            spec_path,
            fixture,
            folder,
            seed_overrides=chosen,
            shards_from=shards_from,
            tokenizer_snapshot=tokenizer_snapshot,
        )
        written = evaluate(search_evidence.evidence_from_runs(spec, build, fixture, folder))
        fails = sorted(n for n, outcome in written.items() if not outcome.holds)
        missing = sorted(set(_CHECK_BY_NUMBER) - set(written))
        confirmed = {
            "folder": CONFIRM_FOLDER,
            "all_conditions_hold": not fails and not missing,
            "fails": fails,
            "not_evaluated": missing,
            "conditions": _condition_rows(written),
        }
        outcomes = written

    evaluated_numbers = {row["n"] for phase in phases for row in phase["conditions"] if row["evaluated"]}
    never_held = sorted(
        row["n"] for phase in phases for row in phase["conditions"] if row["evaluated"] and not row["held"]
    )
    all_hold = everything and (confirmed is None or confirmed["all_conditions_hold"])
    without_seed = [run for run, seed in chosen.items() if seed is None]
    if all_hold and confirmed is not None:
        status = ["Status: every search condition holds for these seeds, checked again on the written runs."]
    elif all_hold:
        status = ["Status: every search condition holds for these seeds in the search; no confirming run was made."]
    elif everything:
        status = [
            "Status: NOT ACCEPTED. The seeds satisfied every condition in the search, but the written runs fail "
            "condition(s) " + ", ".join(str(n) for n in confirmed["fails"] + confirmed["not_evaluated"]) + "."
        ]
    elif not without_seed:
        status = [
            "Status: PARTIAL. Every condition of the runs searched holds for the seeds below.",
            "Runs with a seed of their own that were not searched: "
            + ", ".join(run for run in searchable if run not in selected),
        ]
    else:
        status = [
            "Status: NOT ACCEPTED. No seed tried satisfies run(s) " + ", ".join(without_seed) + "; a seed marked "
            "NOT ACCEPTED below is the best candidate tried, not a result.",
            "Conditions no candidate satisfied: " + (", ".join(str(n) for n in never_held) or "none"),
        ]
    tables = positions_tables(spec, outcomes)
    fragment = toml_fragment(spec, shown, tables, status)
    result = {
        "format": SEARCH_FORMAT,
        "mode": "search",
        "spec": Path(os.path.relpath(os.path.abspath(spec.path), out)).as_posix(),
        "spec_sha256": file_sha256(spec.path),
        "fixture": Path(os.path.relpath(fixture, out)).as_posix(),
        "candidate_seeds": {"first": seeds.start, "last": seeds.stop - 1},
        "time_limit_seconds": time_limit,
        "runs_searched": selected,
        "reworded_conditions_let_through": reworded,
        "phases": phases,
        "chosen_seeds": chosen,
        "seeds_shown": {run: entry["seed"] for run, entry in shown.items()},
        "all_conditions_hold": all_hold,
        "never_held": never_held,
        "not_evaluated": sorted(set(_CHECK_BY_NUMBER) - evaluated_numbers),
        "confirmed": confirmed,
        "positions": tables,
    }
    oracle_config.write_json(out / RESULT_FILE, result)
    (out / FRAGMENT_FILE).write_text(fragment, encoding="utf-8")
    return result, fragment


def check_runs(
    spec_path: str | Path,
    fixture: str | Path,
    runs_folder: str | Path,
    *,
    base: str | Path | None = None,
    allow_reworded: tuple[int, ...] = (),
) -> tuple[dict, str]:
    """Check the conditions on an output folder of the fixture runs. No search
    and no Megatron: only files are read. Returns the result document and the
    TOML fragment."""
    spec = load_spec(spec_path)
    reworded = check_spec_conditions(spec, allow_reworded)
    fixture = Path(os.path.abspath(fixture))
    runs_folder = Path(os.path.abspath(runs_folder))
    build = search_evidence.load_build(spec, fixture, repository_root(spec, base))
    evidence = search_evidence.evidence_from_runs(spec, build, fixture, runs_folder)
    outcomes = evaluate(evidence)
    fails = sorted(n for n, outcome in outcomes.items() if not outcome.holds)
    missing = sorted(set(_CHECK_BY_NUMBER) - set(outcomes))
    overview_file = runs_folder / fixture_runs.OVERVIEW_FILE
    seeds = {}
    if overview_file.is_file():
        with open(overview_file, encoding="utf-8") as f:
            seeds = json.load(f).get("seeds", {})
    shown = {}
    for run in spec.runs.values():
        if run.seed_of != run.name or run.name not in seeds:
            continue
        own = [check.n for other in spec.runs.values() if other.seed_of == run.name for check in checks_of(other.name)]
        failing = [n for n in own if n in fails]
        unchecked = [n for n in own if n in missing]
        note = "; ".join(
            part
            for part in (
                "condition(s) " + ", ".join(map(str, failing)) + " fail" if failing else "",
                "condition(s) " + ", ".join(map(str, unchecked)) + " not checked" if unchecked else "",
            )
            if part
        )
        shown[run.name] = {"seed": seeds[run.name], "accepted": not failing and not unchecked, "note": note}
    all_hold = not fails and not missing
    status = (
        ["Status: every search condition holds on the written runs."]
        if all_hold
        else [
            "Status: NOT ACCEPTED on the written runs.",
            "Conditions that fail: " + (", ".join(map(str, fails)) or "none"),
            "Conditions not checked (their run has no folder): " + (", ".join(map(str, missing)) or "none"),
        ]
    )
    tables = positions_tables(spec, outcomes)
    result = {
        "format": SEARCH_FORMAT,
        "mode": "check-runs",
        "runs_checked": list(evidence.runs),
        "seeds": seeds,
        "reworded_conditions_let_through": reworded,
        "all_conditions_hold": all_hold,
        "fails": fails,
        "not_evaluated": missing,
        "conditions": _condition_rows(outcomes),
        "positions": tables,
    }
    return result, toml_fragment(spec, shown, tables, status)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Search candidate random seeds of the fixture's runs on one built fixture, or "
        "check the search conditions on written fixture runs."
    )
    parser.add_argument("--spec", required=True, help="the fixture spec (TOML)")
    parser.add_argument("--fixture", required=True, help="the folder a fixture build wrote")
    parser.add_argument("--out", help="the output folder of a search; new or empty")
    parser.add_argument("--seeds", default="0:999", metavar="FIRST:LAST", help="candidate seeds, both ends included (default 0:999)")
    parser.add_argument("--time-limit", type=float, default=600.0, metavar="SECONDS", help="time for the candidates of all runs together (default 600)")
    parser.add_argument("--runs", help="comma-separated runs to search, of those with a seed of their own; default: all")
    parser.add_argument("--accept", type=int, default=1, help="stop a run's search after this many accepted seeds (default 1)")
    parser.add_argument("--best", type=int, default=5, help="how many of the best candidates to report per run (default 5)")
    parser.add_argument("--base", help="the repository root, for the spec's file paths; default: three levels above the spec")
    parser.add_argument("--shards-from", choices=fixture_runs.SHARDS_FROM, default=fixture_runs.SHARDS_FROM_PIPELINE)
    parser.add_argument("--tokenizer-snapshot", help="folder with vocab.json and merges.txt")
    parser.add_argument("--allow-reworded", metavar="N,N", help="condition numbers whose sentence in the spec may differ from the one the predicate was written against")
    parser.add_argument("--no-confirm", action="store_true", help="do not write the fixture runs for the seeds found")
    parser.add_argument("--check-runs", metavar="DIR", help="check the conditions on an output folder of fixture_runs.py instead of searching")
    parser.add_argument("--print", dest="show", choices=("both", "json", "toml"), default="both", help="what to print on standard output (default both: the JSON, then the TOML fragment)")
    args = parser.parse_args(argv)
    # Megatron reports each index build through the logging module. A search makes
    # thousands of builds, so only warnings are shown.
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    try:
        allow = parse_numbers(args.allow_reworded, "--allow-reworded")
        if args.check_runs:
            result, fragment = check_runs(args.spec, args.fixture, args.check_runs, base=args.base, allow_reworded=allow)
        else:
            if not args.out:
                raise ConfigError("a search needs --out, a new or empty folder")
            result, fragment = search(
                args.spec,
                args.fixture,
                args.out,
                seeds=parse_seed_range(args.seeds),
                time_limit=args.time_limit,
                runs=args.runs.split(",") if args.runs else None,
                accept=args.accept,
                best=args.best,
                base=args.base,
                shards_from=args.shards_from,
                tokenizer_snapshot=args.tokenizer_snapshot,
                allow_reworded=allow,
                confirm=not args.no_confirm,
            )
    except OracleError as e:
        print(json.dumps({"error": type(e).__name__, "detail": str(e)}), file=sys.stderr)
        return 2
    if args.show in ("both", "json"):
        print(json.dumps(result, indent=2, sort_keys=True))
    if args.show == "both":
        print()
    if args.show in ("both", "toml"):
        print(fragment, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
