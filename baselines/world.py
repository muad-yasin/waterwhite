"""The world builder of Check 10 (plan 5.5 Check 10; docs/baselines.md, "The world builder").

``python -m baselines.world --out DIR --megatron-env DIR --tokenizer FILE --jobs N [--sizes ...]``
builds, with the repository's own builders called unchanged, every store and set the runner is
put to, names every file it made in ``DIR/world.json`` (the file ``python -m baselines --world``
reads), and hashes every file that file names in ``DIR/WORLD.sha256``. Nothing in it is typed by
hand: each input comes from a builder, from the frozen table, from the fixture spec or from the
plant list, or is a named input of the command line (the tokenizer file, the Megatron
environment folder).

What lies in ``DIR``:

- ``work/`` holds every build (a Dolma build lies inside the work folder of a run), the tie
  record, the counting documents, the trace documents and the copy of the tokenizer file;
- ``world.json``, written after everything in ``work/``;
- ``WORLD.sha256``, one line for each file the world names, ``<sha256>`` two spaces ``<path
  relative to DIR>``, ``world.json`` included, in path order;
- ``world-report.txt``, the counts and the seconds of each step (it is not named by the world
  and not hashed, since it holds seconds).

Independent builds run at the same time, each in a process of its own, as many as ``--jobs``
says. Inside a build, the steps that read and write files of their own go on together too (the runs,
the repair and the tie show of a store; the counting and the tie show of a set; the trace of each
starting state), and the tie show of one build uses one tokenizer for each worker, at most one for
each chunk file. With one worker every step runs in order. The same number is handed to every
builder that takes a job count. ``world-report.txt`` says how many cores the command kept busy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib
import importlib.util
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from baselines import builds, dolma, record, table10, tie, trace
from replica.plan import LAYOUTS, load_tokenizer_settings
from replica.shard_files import SUFFIX_TOKENIZER_INPUT

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"
MEGATRON_VARIABLE = "WATERWHITE_MEGATRON_ENV"

WORLD_FILE = "world.json"
# The extensions the builder has a build for: the main store is then the replica's build of the extension (``builds.store_call``;
# the R3 release of the generating-model extension exists only in the build ``main+generating-model``), so the rows of the
# extension are served by it. ``--extensions`` names no extension the builder cannot build.
BUILT_EXTENSIONS: tuple[str, ...] = tuple(builds.MAIN_WITH_EXTENSION)
HASH_FILE = "WORLD.sha256"
REPORT_FILE = "world-report.txt"
WORK = "work"

CHUNK_FOLDER = "tokenizer-input"
SHARD_FOLDER = "shards"
SHARD_SUFFIXES = (".idx", ".bin")
SUCCESSOR_PREFIX = "successor of "
# The folder of the main build a successor's chunk file is copied into: a layout name of its own,
# so that the runner's mixer, which takes the chunk files of one layout, does not mix it with the
# shards of the historical layout.
SUCCESSOR_FOLDER = "successor"
# The source of the scratch store whose 500-line file the store ingests under a source of its own.
SCRATCH_SOURCE_FILE = "c04.jsonl"
# The record the table names W is the spec's shared-token position.
SPEC_POSITION_OF_HANDLE = {"W": "shared_token"}
# The four starting states of Check 2 the main store's runs are registered again from.
RUN_WORLD_STATES = ("c2.i", "c2.ii", "c2.iii", "c2.iv")

TOOL_TIMEOUT = 900.0       # seconds, for one command of the tool
REPLICA_TIMEOUT = 3600.0   # seconds, for one command of the replica
TOKENIZER_TIMEOUT = 600.0  # seconds, for one answer of the line tokenizer

# The stores that are OriginBlame shards: every store of the list has shards of its own (since 17 Nov 2026 the table holds OriginBlame rows for no-capture
# and scratch-notokens too).
SHARD_STORES = ("main", "rr", "no-capture", "scratch-500", "scratch-notokens")
# The order the stores are listed in the world file.
STORE_ORDER = ("main", "rr", "no-capture", "scratch-500", "scratch-notokens")
# The table's stores that hold no chunk file and no shard (a Dolma entry or an OriginBlame shard has nothing to run on): the
# world builds them for the manifest and the trace, and the none rows of the tools on them are put with the attempts of the pass.
SHARDLESS_STORES = ("swap", "import")
# A generated set made without capture is a build of its own: the world key of its sizes is ``no-capture-<size>`` (no folder
# of a size, ``set-<size>``).
NO_CAPTURE = "no-capture"
# The set sizes that are also built in a second layout (W1, 18 Nov 2026: Check 9's rows of the 10k set name both layouts), and the
# layout: the build is the replica's build-set of the same generated set, capture on, in that layout.
LAYOUT_BUILDS: Mapping[str, tuple[str, ...]] = {"10k": ("partitioned",)}
KILL_ROWS = ("c04.kill-after-100", "c04.retry")


class WorldError(Exception):
    """The world cannot be built as asked."""


class TieMergeError(ValueError):
    """Two builds give different results for one chunk file."""


# --- The tie record: one result for each chunk file -------------------------------------------------

TIE_FIELDS = ("shown", "documents", "lines", "first_mismatch")


def merge_tie_results(results_by_build: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict]:
    """One result for each chunk file out of the results of several builds. Two builds that
    name one chunk file are merged when their results are equal; a difference raises
    ``TieMergeError`` naming the chunk file and both builds. The result is in chunk-file order."""
    merged: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for build in sorted(results_by_build):
        for result in results_by_build[build]:
            name = result["chunk_file"]
            if name not in merged:
                merged[name] = (build, result)
                continue
            first_build, first = merged[name]
            differing = [key for key in TIE_FIELDS if first[key] != result[key]]
            if differing:
                words = "; ".join(key + " " + repr(first[key]) + " against " + repr(result[key]) for key in differing)
                raise TieMergeError(
                    f"the chunk file {name} has a different tie result in {first_build} and in {build}: {words}")
    return [dict(merged[name][1]) for name in sorted(merged)]


# --- Small helpers ------------------------------------------------------------------------------------


def _acceptance(name: str):
    """A module of ``tests/acceptance`` by its bare name; the table reader puts that folder on
    the path."""
    table10._acceptance_table()
    return importlib.import_module(name)


def _resolve(dotted: str):
    module, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(REPO)])
    return env


def _run(argv: Sequence[str], *, timeout: float, what: str) -> subprocess.CompletedProcess:
    done = subprocess.run([str(word) for word in argv], cwd=REPO, env=_environment(), capture_output=True,
                          text=True, timeout=timeout, check=False)
    if done.returncode != 0:
        raise WorldError(f"{what} ended with exit code {done.returncode}: {done.stderr[-800:]}")
    return done


def _sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, document) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _plant_script(name: str):
    """A script of ``fixture/plant``, loaded by file: the folder holds scripts, not a package."""
    path = REPO / "fixture" / "plant" / name
    spec = importlib.util.spec_from_file_location(f"world_plant_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Log:
    """The seconds of the steps of one task, printed as they end."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.steps: list[tuple[str, float]] = []

    @contextlib.contextmanager
    def step(self, what: str):
        started = time.monotonic()
        print(f"[{self.label}] {what}: started", flush=True)
        try:
            yield
        finally:
            seconds = time.monotonic() - started
            self.steps.append((f"{self.label}: {what}", round(seconds, 1)))
            print(f"[{self.label}] {what}: {seconds:.1f} seconds", flush=True)


def _shards_of(files: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    """The shards a build registered, by store name: the tokenizer input and the prefix of the
    two shard files, both relative to the build folder. A shard is a name whose ``.input`` is
    registered beside it."""
    found: dict[str, tuple[str, str]] = {}
    for key in sorted(files):
        if key.endswith(SUFFIX_TOKENIZER_INPUT):
            name = key[: -len(SUFFIX_TOKENIZER_INPUT)]
            if name in files:
                found[name] = (files[key], files[name])
    return found


def _layout_of(chunk: str) -> str | None:
    """The layout a chunk file lies in, ``<layout>/tokenizer-input/<shard>.jsonl``; None for a
    folder that is no layout of the build (the scratch stores)."""
    parts = PurePosixPath(chunk).parts
    if CHUNK_FOLDER not in parts or parts.index(CHUNK_FOLDER) == 0:
        return None
    name = parts[parts.index(CHUNK_FOLDER) - 1]
    return name if name in LAYOUTS else None


def spread(weights: Sequence[int], jobs: int) -> list[list[int]]:
    """The indexes of ``weights`` in as many groups as there are workers, and no more groups than
    weights: the heaviest weight first, each into the group whose total is smallest so far. One
    worker gives one group that holds every index."""
    if not weights:
        return []
    count = max(1, min(jobs, len(weights)))
    groups: list[list[int]] = [[] for _ in range(count)]
    loads = [0] * count
    for index in sorted(range(len(weights)), key=lambda number: (-weights[number], number)):
        target = min(range(count), key=lambda number: (loads[number], number))
        groups[target].append(index)
        loads[target] += max(1, weights[index])
    return [sorted(group) for group in groups]


def set_chunk_file(size: str, shard: str) -> str:
    """The name of the chunk file of a shard of the set of a size, relative to the build. Every
    size of a generated set deals its records over the same chunks, so the shard names of two
    sizes are the same names, and the tie record is keyed by the name of the chunk file. The
    smallest size keeps the path the build wrote; every other size names a copy under a folder of
    its own."""
    if size == builds.SET_SIZES[0]:
        return f"pipeline/{builds.LAYOUT}/{CHUNK_FOLDER}/{shard}.jsonl"
    return f"set-{size}/{CHUNK_FOLDER}/{shard}.jsonl"


def name_set_chunk_files(size: str, build) -> dict[str, str]:
    """The chunk file of each shard of the set of a size, by shard name, as ``set_chunk_file``
    names it. A size that is not the smallest gets a copy of each chunk file of its build at that
    name (the copy is the same bytes; nothing the build wrote is changed)."""
    return name_set_chunk_files_from(size, build)[0]


def name_set_chunk_files_from(size: str, build) -> tuple[dict[str, str], dict[str, str]]:
    """``name_set_chunk_files`` and, written at the moment each file is named or copied, where it came from: the second mapping
    gives for each shard the build's own path of the chunk file (relative to the build). It is the named path itself where
    no copy was made (the smallest size), so a reader never derives the copy's source from equal bytes (W9, 18 Nov 2026)."""
    build = Path(build)
    named: dict[str, str] = {}
    copy_of: dict[str, str] = {}
    for path in sorted((build / "pipeline" / builds.LAYOUT / CHUNK_FOLDER).glob("*.jsonl")):
        if "." in path.stem:
            # A file beside the chunk files that is no chunk file (a manifest is ``<file>.manifest.jsonl``).
            continue
        relative = set_chunk_file(size, path.stem)
        original = path.relative_to(build).as_posix()
        if relative != original:
            target = build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        named[path.stem] = relative
        copy_of[path.stem] = original
    return named, copy_of


def _tie_results(items: Sequence[tuple[str, Path, Path]], work: Path, jobs: int) -> list[dict]:
    """The tie show of each chunk file: ``(name as recorded, path of the chunk file, prefix of
    its shard)``, in the order of ``items``. The tokenizer is the line tokenizer of the Megatron
    environment; with several workers the chunk files are spread over one tokenizer for each
    worker (at most one for each chunk file), each with a work folder of its own."""
    from replica import megatron_tokenizer

    environment = megatron_tokenizer.MegatronEnv.from_environment()
    settings = load_tokenizer_settings(SPEC)
    work.mkdir(parents=True, exist_ok=True)
    groups = spread([chunk.stat().st_size for _, chunk, _ in items], jobs)

    def show(number: int, group: Sequence[int]) -> dict[int, dict]:
        with megatron_tokenizer.line_tokenizer(environment, settings, work / f"tokenizer-{number}",
                                               timeout=TOKENIZER_TIMEOUT) as tokenize:
            return {index: tie.show_chunk_file(items[index][0], items[index][1], items[index][2], tokenize)
                    for index in group}

    found: dict[int, dict] = {}
    if len(groups) <= 1:
        for group in groups:
            found.update(show(1, group))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
            futures = [pool.submit(show, number, group) for number, group in enumerate(groups, 1)]
            for future in futures:
                found.update(future.result())
    return [found[index] for index in range(len(items))]


# --- The trace documents of a store with runs ---------------------------------------------------------


def _reason_of(found) -> str:
    """What a start of the export command came to, in words, when it gave no document."""
    exit_code = getattr(found, "exit_code", None)
    output = " ".join(str(getattr(found, "output", "")).split())
    return f"exit code {exit_code}: {output[:300]}"


def _read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _export_segments(store: Path, runs_dir: Path, runs: Sequence[str], run_record_of, *,
                     refusals_allowed: bool) -> tuple[list[dict], dict]:
    """The segment records of the trace of every segment the runs register, made by the tool's
    own show command and ``baselines.trace.export_segment``. A segment the tool refuses by name
    (the documented refusal) is kept with the tool's words where ``refusals_allowed`` (a starting
    state of Check 2 changes what the tool can say); any other end, and a refusal in the store as
    built, ends the command: it is an error of the build, never an answer."""
    acceptance_runs = _acceptance("acceptance_runs")
    acceptance_repair = _acceptance("acceptance_repair")
    documents: list[dict] = []
    failed: dict[str, str] = {}
    for plan in acceptance_runs.segments_of(acceptance_runs.spec(), runs):
        files = acceptance_runs.files_of(runs_dir, plan)
        words = ["show", plan.name, "--trace"]
        try:
            code, out, err = acceptance_repair.tool_command(store, words, TOOL_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise WorldError(f"show {plan.name} --trace did not end within {TOOL_TIMEOUT:.0f} seconds") from None
        found = trace.classify_show(plan.name, [*words, "--store", str(store)], code,
                                    out if code == 0 else err + out, f"trace-{plan.stem}")
        if isinstance(found, trace.Refusal) and refusals_allowed:
            failed[plan.name] = _reason_of(found)
            continue
        if not isinstance(found, dict):
            raise WorldError(f"show {plan.name} --trace gave no trace: {_reason_of(found)}")
        parent_shown = None
        if plan.parent_checkpoint is not None:
            code, out, err = acceptance_repair.tool_command(store, ["show", plan.parent_checkpoint], TOOL_TIMEOUT)
            try:
                parent_shown = json.loads(out) if code == 0 else None
            except ValueError:
                parent_shown = None
        try:
            documents.append(trace.export_segment(found, _read_json(run_record_of(plan, files)),
                                                  _read_json(files.steps), parent_shown))
        except (trace.ExportError, OSError, ValueError) as error:
            raise WorldError(f"the trace of {plan.name} cannot be exported: {str(error)[:300]}") from error
    return documents, failed


def _trace_record(directory: Path, name: str, documents: list[dict], failed: dict, none: str) -> dict:
    """Write the document of one store or state and say what it came to."""
    if not documents:
        return {"document": None, "failed": dict(failed), "none": none}
    path = _write_json(directory / name, {"segments": documents})
    return {"document": str(path), "failed": dict(failed), "none": None}


def _state_trace(state: str, store_before: Path, runs_dir: Path, runs: Sequence[str], root: Path,
                 directory: Path, scratch: Path, log: _Log) -> dict:
    """The trace written from one starting state of Check 2: the runs are registered again in a
    fresh copy of the store as it was before any run, the run records and cache folders as the
    state has them."""
    acceptance_runs = _acceptance("acceptance_runs")
    fresh = store_before.with_name(f"store-{state}.sqlite")
    acceptance_runs.snapshot_store(store_before, fresh)
    records = scratch / f"records-{state}"
    records.mkdir(parents=True)
    try:
        with log.step(f"register the runs in the state {state}"):
            acceptance_runs.register_runs(fresh, runs_dir, runs, root, state=state, scratch=records)
    except (acceptance_runs.RunWorldError, subprocess.TimeoutExpired) as error:
        raise WorldError(f"the runs were not registered in the state {state}: {str(error)[-400:]}") from error

    def run_record_of(plan, files):
        changed = records / plan.run / files.run_record.name
        return changed if changed.is_file() else files.run_record

    with log.step(f"export the trace of the state {state}"):
        documents, failed = _export_segments(fresh, runs_dir, runs, run_record_of, refusals_allowed=True)
    return _trace_record(directory, f"state-{state}.json", documents, failed,
                         f"the tool printed no trace in the state {state}")


def _runs_part(name: str, built, work: Path, jobs: int, log: _Log) -> dict:
    """The runs of a store built, registered through the tool, and the trace documents the tool
    prints for them (and, for the store that has them, the four starting states of Check 2)."""
    acceptance_runs = _acceptance("acceptance_runs")
    acceptance_sets = _acceptance("acceptance_sets")
    missing = acceptance_runs.capture_missing()
    if missing is not None:
        raise WorldError(f"the runs of the store {name} cannot be made: {missing}")
    root = acceptance_sets.need_megatron()
    runs = acceptance_runs.RUNS_OF_STORE[name]
    store = Path(built.store)
    if not acceptance_runs.verified_joins(store):
        raise WorldError(f"the store {name} that is to hold runs {list(runs)} was built with no verified join")
    before = store.with_name("before-runs.sqlite")
    acceptance_runs.snapshot_store(store, before)
    runs_dir = store.parent / "runs"
    with log.step("build the runs"):
        acceptance_runs.build_runs(built.spec, built.out, runs_dir, runs, root)
    directory = work / "traces" / f"store-{name}"
    # The oracle's own log of each run (the source the three whole-run diff rows name): the oracle runs over the same build
    # (``oracle/fixture_runs.py`` through the acceptance helper, nothing of the oracle or the replica changed); one file of
    # lines for each run is kept, hashed with the world.
    with log.step("run the oracle"):
        oracle_runs = acceptance_runs.oracle_runs(built.out, work / "oracle-runs" / name, root, runs)
    oracle_logs: dict[str, str] = {}
    for run in runs:
        rows = []
        for path in sorted(Path(oracle_runs).rglob("oracle_log.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and str(json.loads(line)["segment"]).split("/", 1)[0] == run:
                    rows.append(line)
        if not rows:
            raise WorldError(f"the oracle's log of the run {run} of the store {name} holds no row")
        target = work / "oracle" / f"{run}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(line + "\n" for line in rows), encoding="utf-8")
        oracle_logs[run] = str(target)

    def own() -> dict:
        # The segments are registered one after another: a segment's position in its run is the
        # number of segments registered before it.
        with log.step("register the runs"):
            acceptance_runs.register_runs(store, runs_dir, runs, root)

        def own_record(plan, files):
            return files.run_record

        with log.step("export the trace"):
            documents, failed = _export_segments(store, runs_dir, runs, own_record, refusals_allowed=False)
        return _trace_record(directory, "trace.json", documents, failed, "the tool printed no trace of any segment")

    # The registration in the store and the registration in each starting state of the store as it
    # was before any run touch files of their own: they go on together.
    parts: dict[str, Any] = {"own": own}
    if name == "main":
        for state in RUN_WORLD_STATES:
            parts[state] = (lambda state=state: _state_trace(
                state, before, runs_dir, runs, root, directory, work / "records" / name, log))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(parts)))) as pool:
        futures = {key: pool.submit(part) for key, part in parts.items()}
        done = {key: future.result() for key, future in futures.items()}
    found = done.pop("own")
    found["states"] = done if name == "main" else None
    found["oracle_logs"] = oracle_logs
    return found


# --- The repaired side, for the successor of a shard ---------------------------------------------------


def _repair_part(built, copy: Path, source: str, prefix: str, jobs: int, log: _Log) -> dict:
    """The report of the withdrawal of ``source`` from the tool, then the replica's repair, both on
    the copy of the store that was made before any run was registered in it."""
    acceptance_repair = _acceptance("acceptance_repair")
    store = copy / Path(built.store).name
    historical = copy / Path(built.out).name
    with log.step("report"):
        code, out, err = acceptance_repair.tool_command(
            store, ["report", "withdraw", source, "--layout", f"partitioned={prefix}"], TOOL_TIMEOUT)
    if code != 0:
        raise WorldError(f"the report of the withdrawal of {source} ended with exit code {code}: {err[:300]}")
    report = copy / "report.json"
    report.write_text(out, encoding="utf-8")
    repaired = copy / "repaired"
    with log.step("repair"):
        printed = acceptance_repair.replica_command([
            "repair", "--report", report, "--historical", historical, "--out", repaired, "--store", store,
            "--spec", built.spec, "--layout", "partitioned", "--jobs", jobs])
    return {"out": str(repaired), "successors": dict(printed["successors"]),
            "files": dict(printed["repaired"]["files"])}


def _set_repair(size: str, layout: str, set_folder: Path, out: Path, store: Path, summary: Mapping[str, Any], work_path: Path,
                jobs: int, timeout: float, log: "_Log") -> dict:
    """The repaired side of a generated set built in ``layout`` (W2): Check 9's report and repair commands on a COPY of the set's
    store, taken beside it (the paths of a store are relative to its folder): the report of the withdrawal of A for the shards the
    documents say the layout registers (a partitioned layout is stated its placement from the documented rule), then the replica's
    repair of the set into a folder of its own, under the harness the build's own logs name. No reference side is built (D-WORLD-13).
    Returns the repaired folder and the copy's path."""
    acceptance_repair, acceptance_sets, acceptance_runs = (_acceptance(name) for name in
                                                           ("acceptance_repair", "acceptance_sets", "acceptance_runs"))
    if "instances" not in summary:
        raise WorldError(f"the build of the {size} set in the {layout} layout printed no instances: no report can name its shards")
    shards = acceptance_sets.layout_shards(summary["instances"], layout, set_folder)
    if not shards:
        raise WorldError(f"the build of the {size} set in the {layout} layout registered no shard: no report can name its layout")
    prefix = acceptance_repair.shard_prefix(shards)
    folder = work_path / "repair" / f"set-{size}-{layout}"
    folder.mkdir(parents=True, exist_ok=True)
    copy = acceptance_runs.snapshot_store(store, Path(store).with_name(f"{Path(store).stem}-repair{Path(store).suffix}"))
    placement = ([] if layout != "partitioned" else
                 ["--placement", f"{layout}={acceptance_repair.write_placement(folder, set_folder, prefix)}"])
    with log.step(f"report in the {layout} layout"):
        code, stdout, stderr = acceptance_repair.tool_command(
            copy, ["report", "withdraw", "A", "--layout", f"{layout}={prefix}", *placement], TOOL_TIMEOUT)
    if code != 0:
        raise WorldError(f"the report of the withdrawal of A on the {size} set in the {layout} layout ended with exit code {code}: "
                         f"{stderr[:300]}")
    report = folder / "report.json"
    report.write_text(stdout, encoding="utf-8")
    repaired = folder / "repaired"
    with log.step(f"repair in the {layout} layout"):
        printed = acceptance_repair.replica_command(["repair", "--report", report, "--historical", out, "--out", repaired,
                                                     "--store", copy, "--set", set_folder, "--layout", layout, "--jobs", jobs,
                                                     "--tokenizer-spec", SPEC], timeout)
    if not (repaired / "built-from.jsonl").is_file():
        raise WorldError(f"the repair of the {size} set in the {layout} layout wrote no file-level manifest")
    # ``files`` (the files of the repaired build by registered name) and ``kept`` (the artifacts the repair kept) are what the
    # successor shards are read from (``_successor_shards_of``); the equality gate reads the folder only.
    return {"layout": layout, "out": str(repaired), "store": str(copy), "prefix": prefix,
            "files": dict((printed.get("repaired") or {}).get("files") or {}), "kept": dict(printed.get("kept") or {})}


def _successor_shards_of(repaired: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """The successor shards of a repair: the shards the repaired build registered under ``successor of <name>`` (a shard is a name
    whose ``.input`` is registered beside it), by that name, with their tokenizer input and the prefix of their two shard files, both
    relative to the repaired folder. A shard the repair kept is registered under that name too (the same instance under its second
    name, ``kept``) and is no successor; a shard that no record reaches is registered under none. A landing shard (the report names it
    under the layout's landing, and no artifact of the historical build is its) is a successor shard like a rewritten one."""
    kept = {entry["registered_as"] for entry in (repaired.get("kept") or {}).values() if entry.get("registered_as")}
    return {name: files for name, files in _shards_of(repaired.get("files") or {}).items()
            if name.startswith(SUCCESSOR_PREFIX) and name not in kept}


def _place_successors(repaired: Mapping[str, Any], build: Path, layout: str, label: str) -> dict:
    """The successor shards of a set's repair put where OriginBlame reads a shard: the tokenizer input and the two shard files of
    each are COPIED into the folder ``successor`` of the historical build ``build`` (as the successor of the third shard of main is,
    D-WORLD-9), under a name that holds the layout so that no name is shared by two builds of the world; the repaired folder is not
    touched (the equality gate compares it). No tie show is made: only OriginBlame reads these chunk files, and the tie record is read
    for the chunk files of a Dolma run (the successor of main is one, these are not). Returns the shards by registered name, each with
    its ``chunk_file`` and shard ``prefix`` relative to ``build``."""
    found = _successor_shards_of(repaired)
    if not found:
        raise WorldError(f"the repair of {label} registered no successor shard")
    out, shards = Path(repaired["out"]), {}
    for name in sorted(found):
        chunk, prefix = found[name]
        relative = f"{SUCCESSOR_FOLDER}/{CHUNK_FOLDER}/{layout}-{PurePosixPath(chunk).name}"
        target = build / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / chunk, target)
        shard_prefix = f"{SUCCESSOR_FOLDER}/{SHARD_FOLDER}/{layout}-{PurePosixPath(prefix).name}"
        (build / shard_prefix).parent.mkdir(parents=True, exist_ok=True)
        for suffix in SHARD_SUFFIXES:
            shutil.copyfile(out / f"{prefix}{suffix}", build / f"{shard_prefix}{suffix}")
        shards[name] = {"chunk_file": relative, "prefix": shard_prefix}
    return {"shards": shards}


# --- One task for each independent build ---------------------------------------------------------------


def _repair_copy_call(name: str, copy: Path, jobs: int, extensions: Sequence[str]) -> "builds.Call | None":
    """The call that makes the copy the repair of ``name`` works on, or None where the copy is a copy of the store's folder.
    ``replica repair`` knows only the releases of the fixture spec's ``[releases.*]`` tables, and the report of a store that holds
    R3 (an ``[extension_releases.R3]``) names R3: the repair of the keyed store ``main+generating-model`` ends with "no release
    named ['R3']" (box row, 18 Nov 2026). With the extension on, the repair of main is therefore made on a plain build of main in
    the folder of the copy, which is the build of the world without the extension (a seeded store: the same bytes), so the
    repaired side is the same in both worlds."""
    if name == "main" and extensions:
        return builds.store_call("main", copy, parallel=jobs)
    return None


def _store_task(name: str, folder: str, jobs: int, work: str, successors: Sequence[str], source: str | None,
                extensions: Sequence[str] = ()) -> dict:
    """One store of the table, built tokenized, with the runs and trace documents it holds, the
    repair its successor shards come from, and the tie show of each of its chunk files."""
    folder, work_path = Path(folder), Path(work)
    folder.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(f"store-{name}")
    acceptance_probe = _acceptance("acceptance_probe")
    _acceptance("acceptance_runs")
    _acceptance("acceptance_repair")
    _acceptance("acceptance_sets")
    call = builds.store_call(name, folder, parallel=jobs, extensions=extensions if name == "main" else ())
    function = _resolve(call.function)
    keywords = dict(call.kwargs)
    keywords.update(acceptance_probe.generation_arguments(function))
    with log.step("build the store"):
        built = function(*call.args, **keywords)
    shards = _shards_of(built.files)
    if not shards:
        raise WorldError(f"the store {name} registered no shard")
    build = Path(built.out)
    if name == "scratch-500":
        # The store ingests its 500-line file under a source of its own: the file is a record file
        # of that source, put where the readers of record files look.
        (build / "sources").mkdir(exist_ok=True)
        for chunk, _ in shards.values():
            shutil.copyfile(build / chunk, build / "sources" / SCRATCH_SOURCE_FILE)
    copy = None
    repair_built = built
    if successors:
        copy = work_path / "repair" / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        plain = _repair_copy_call(name, copy, jobs, extensions)
        if plain is None:
            with log.step("copy the store for the repair"):
                shutil.copytree(folder, copy)
        else:
            with log.step("build the store again without the extension, for the repair"):
                repair_built = function(*plain.args, **{**plain.kwargs, **acceptance_probe.generation_arguments(function)})
    chunk_files = {shard: chunk for shard, (chunk, _) in shards.items()}
    items = [(chunk, build / chunk, build / prefix) for chunk, prefix in shards.values()]

    def own_tie() -> list[dict]:
        with log.step("tie show"):
            return _tie_results(items, work_path / "tie-work" / name, jobs)

    # The tie show of the build's own chunk files, the runs and the repair read files the others do
    # not write, so they go on together (one after another with one worker).
    parts: dict[str, Any] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, 3))) as pool:
        futures = {"tie": pool.submit(own_tie)}
        if name in builds.RUNS_OF_STORE:
            futures["runs"] = pool.submit(_runs_part, name, built, work_path, jobs, log)
        if copy is not None:
            prefixes = sorted({shard.rsplit("/", 1)[0] + "/" for shard in shards})
            if len(prefixes) != 1:
                raise WorldError(f"the shards of {name} lie under {prefixes}: the report names one prefix")
            futures["repair"] = pool.submit(_repair_part, repair_built, copy, source, prefixes[0], jobs, log)
        for key, future in futures.items():
            parts[key] = future.result()
    results = list(parts["tie"])
    successor_files: dict[str, str] = {}
    successor_prefixes: dict[str, str] = {}
    successor_build = None
    if "repair" in parts:
        repair = parts["repair"]
        repaired = Path(repair["out"])
        successor_items = []
        for artifact in successors:
            historical = artifact[len(SUCCESSOR_PREFIX):]
            if repair["successors"].get(historical) != artifact:
                raise WorldError(f"the repair registered no {artifact}: it registered {sorted(repair['successors'])}")
            chunk, prefix = repair["files"][artifact + SUFFIX_TOKENIZER_INPUT], repair["files"][artifact]
            relative = f"{SUCCESSOR_FOLDER}/{CHUNK_FOLDER}/{PurePosixPath(chunk).name}"
            target = build / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repaired / chunk, target)
            successor_files[artifact] = relative
            # The index and the packed file of the repaired side's tokenized shard go into the main
            # build too, by the route of the chunk file, so that the successor is a shard of OriginBlame
            # inside the build of main.
            shard_prefix = f"{SUCCESSOR_FOLDER}/{SHARD_FOLDER}/{PurePosixPath(prefix).name}"
            (build / shard_prefix).parent.mkdir(parents=True, exist_ok=True)
            for suffix in SHARD_SUFFIXES:
                shutil.copyfile(repaired / f"{prefix}{suffix}", build / f"{shard_prefix}{suffix}")
            successor_prefixes[artifact] = shard_prefix
            successor_items.append((relative, target, repaired / prefix))
        successor_build = str(repaired)
        with log.step("tie show of the successor"):
            results += _tie_results(successor_items, work_path / "tie-work" / f"{name}-successor", jobs)
    return {"name": name, "folder": str(folder), "build": str(build), "store": str(built.store),
            "files": dict(built.files), "shards": {shard: list(pair) for shard, pair in shards.items()},
            "chunk_files": {**chunk_files, **successor_files}, "successor_prefixes": successor_prefixes,
            "tie": results, "runs": parts.get("runs"), "repaired": successor_build, "steps": log.steps}


def _shardless_task(name: str, folder: str, jobs: int, work: str) -> dict:
    """One store of the table that holds no chunk file and no shard (swap, import), built by the replica's own route for the
    table's stores (``build_table_store``, builders unchanged): its manifest and nothing a tool runs on."""
    folder = Path(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(f"store-{name}")
    _acceptance("acceptance_sets")
    call = builds.store_call(name, folder, parallel=jobs)
    function = _resolve(call.function)
    keywords = dict(call.kwargs)
    keywords.update(_acceptance("acceptance_probe").generation_arguments(function))
    with log.step("build the store"):
        built = function(*call.args, **keywords)
    if _shards_of(built.files):
        raise WorldError(f"the store {name} registered a shard: it is no store without chunk file and shard")
    return {"name": name, "folder": str(folder), "build": str(built.out), "store": str(built.store),
            "files": dict(built.files), "steps": log.steps}


def own_chunk_files(build, layout: str) -> dict[str, str]:
    """The chunk file of each shard of a build in ``layout``, by shard name, as the build wrote it (no copy: the path
    ``pipeline/<layout>/tokenizer-input/<shard>.jsonl`` is the build's own and unique among the builds of the world)."""
    build = Path(build)
    return {path.stem: path.relative_to(build).as_posix()
            for path in sorted((build / "pipeline" / layout / CHUNK_FOLDER).glob("*.jsonl")) if "." not in path.stem}


def _summary_of(done: subprocess.CompletedProcess, what: str) -> dict:
    """The JSON summary a replica command printed on standard output."""
    try:
        summary = json.loads(done.stdout)
    except ValueError:
        raise WorldError(f"{what} printed no JSON summary: {done.stdout[:200]!r}") from None
    if not isinstance(summary, dict):
        raise WorldError(f"{what} printed a summary that is no object")
    return summary


def _layout_build(size: str, folder: Path, set_folder: Path, layout: str, jobs: int, work_path: Path, log: "_Log",
                  timeout: float) -> dict:
    """The generated set of ``size`` built again in ``layout`` (the replica's build-set, capture on: the route of the round-robin
    build, which J23 showed equal to Check 9's in every file a question reads), its counting document for that layout and the tie
    show of its chunk files. Its chunk files keep the build's own paths. Returns the same fields as the set's own build."""
    out, store = folder / f"build-{layout}", folder / f"store-{layout}.sqlite"
    with log.step(f"build the set in the {layout} layout"):
        done = _run(builds.set_build_argv(set_folder, out, store=store, python=sys.executable, tokenizer_spec=SPEC, jobs=jobs, layout=layout),
                    timeout=timeout, what=f"the replica's build of the {size} set in the {layout} layout")
    summary = _summary_of(done, f"the replica's build of the {size} set in the {layout} layout")
    pipeline = out / "pipeline" / layout
    named = own_chunk_files(out, layout)
    if not named:
        raise WorldError(f"the build of the {size} set in the {layout} layout wrote no chunk file")
    counting = _plant_script("count.py")
    sanitizer = pipeline / "sanitizer-input"

    def count() -> dict:
        with log.step(f"count the set in the {layout} layout"):
            try:
                counted = counting.count(set_folder, chunks=pipeline / CHUNK_FOLDER, layout=layout,
                                         sanitizer_input=sanitizer if sanitizer.is_dir() else None, require_pair_kinds="auto")
            except counting.CountError as refusal:
                raise WorldError(f"the counting script refuses the {size} set in the {layout} layout: "
                                 f"{type(refusal).__name__}: {refusal}") from refusal
        return json.loads(json.dumps(counted))

    def show() -> list[dict]:
        items = [(named[shard], out / named[shard], out / f"pipeline/{layout}/{SHARD_FOLDER}/{shard}") for shard in sorted(named)]
        with log.step(f"tie show in the {layout} layout"):
            return _tie_results(items, work_path / "tie-work" / f"set-{size}-{layout}", jobs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, 2))) as pool:
        counting_future, show_future = pool.submit(count), pool.submit(show)
        counted, results = counting_future.result(), show_future.result()
    counted_file = _write_json(work_path / "counted" / f"{size}-{layout}.json", counted)
    shards = [entry["shard"] for entry in counted["documents"]["shards"]]
    if sorted(shards) != sorted(named):
        raise WorldError(f"the counting document of the {size} set in the {layout} layout names the shards {sorted(shards)} and "
                         f"its build wrote the chunk files {sorted(named)}")
    if counted["shards"].get("layout") != layout:
        raise WorldError(f"the counting document of the {size} set in the {layout} layout was counted on the layout "
                         f"{counted['shards'].get('layout')}")
    chunk_files = {shard: named[shard] for shard in shards}
    repaired = _set_repair(size, layout, set_folder, out, store, summary, work_path, jobs, timeout, log)
    placed = _place_successors(repaired, out, layout, f"set-{size}-{layout}")
    return {"layout": layout, "build": str(out), "store": str(store), "counted": str(counted_file), "chunk_files": chunk_files,
            "copy_of": dict(chunk_files), "prefixes": {shard: f"pipeline/{layout}/{SHARD_FOLDER}/{shard}" for shard in shards},
            "tie": results, "repaired": {**repaired, "successor_shards": placed["shards"]}}


def _set_task(size: str, folder: str, jobs: int, work: str, variant: str = "capture") -> dict:
    """One generated set: the plant generator, the replica's capture build in the round-robin
    layout (variant ``capture``) or the build made without capture and recovered, by Check 6's own route
    (``replica.fixture.build_set_store``, variant ``no-capture``), the counting script on the chunk files of that
    build, and the tie show of each of its chunk files. The build made without capture is a set of its own: its label is
    ``no-capture-<size>``, its chunk files are named apart (copies under ``set-no-capture-<size>``) and its counting
    document does not require a same-shard pair, as Check 6 counts it (the unseeded shuffles do not fix where a pair lies)."""
    folder, work_path = Path(folder), Path(work)
    folder.mkdir(parents=True, exist_ok=True)
    label = size if variant == "capture" else f"{NO_CAPTURE}-{size}"
    log = _Log(f"set-{label}")
    acceptance_sets = _acceptance("acceptance_sets")
    set_folder, out, store = folder / "set", folder / "build", folder / "store.sqlite"
    with log.step("generate the set"):
        _run(builds.generate_argv(size, set_folder, python=sys.executable),
             timeout=acceptance_sets.BUILD_TIMEOUT[size], what=f"the plant generator for the {size} set")
    if variant == "capture":
        with log.step("build the set"):
            done = _run(builds.set_build_argv(set_folder, out, store=store, python=sys.executable, tokenizer_spec=SPEC, jobs=jobs),
                        timeout=acceptance_sets.BUILD_TIMEOUT[size], what=f"the replica's build of the {size} set")
        build_summary = _summary_of(done, f"the replica's build of the {size} set") if size in LAYOUT_BUILDS else None
    else:
        from replica import fixture as replica_fixture

        with log.step("build the set without capture"):
            built = replica_fixture.build_set_store(folder / "no-capture", set_folder, NO_CAPTURE, layout=builds.LAYOUT,
                                                    tokenizer_spec=SPEC, parallel=jobs)
        out, store = Path(built.out), Path(built.store)
    pipeline = out / "pipeline" / builds.LAYOUT
    named, copy_of = name_set_chunk_files_from(label, out)
    if not named:
        raise WorldError(f"the build of the {label} set wrote no chunk file")
    counting = _plant_script("count.py")
    sanitizer = pipeline / "sanitizer-input"

    def count() -> dict:
        with log.step("count the set"):
            try:
                counted = counting.count(set_folder, chunks=pipeline / CHUNK_FOLDER, layout=builds.LAYOUT,
                                         sanitizer_input=sanitizer if sanitizer.is_dir() else None,
                                         require_pair_kinds="auto" if variant == "capture" else "no")
            except counting.CountError as refusal:
                raise WorldError(f"the counting script refuses the {label} set: {type(refusal).__name__}: {refusal}") from refusal
        return json.loads(json.dumps(counted))

    def pipeline_prefix(shard: str) -> str:
        return f"pipeline/{builds.LAYOUT}/{SHARD_FOLDER}/{shard}"

    def show() -> list[dict]:
        items = [(named[shard], out / named[shard], out / pipeline_prefix(shard)) for shard in sorted(named)]
        with log.step("tie show"):
            return _tie_results(items, work_path / "tie-work" / f"set-{label}", jobs)

    # The counting script and the tie show both read the chunk files and write nothing there: they go
    # on together (one after another with one worker).
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, 2))) as pool:
        counting_future, show_future = pool.submit(count), pool.submit(show)
        counted, results = counting_future.result(), show_future.result()
    counted_file = _write_json(work_path / "counted" / f"{label}.json", counted)
    shards = [entry["shard"] for entry in counted["documents"]["shards"]]
    if sorted(shards) != sorted(named):
        raise WorldError(f"the counting document of the {label} set names the shards {sorted(shards)} and its build "
                         f"wrote the chunk files {sorted(named)}")
    chunk_files = {shard: named[shard] for shard in shards}
    copies = {shard: copy_of[shard] for shard in shards}
    prefixes = {shard: pipeline_prefix(shard) for shard in shards}
    splits = [pair for pair in counted["check3"]["pairs"]["list"] if pair["placement"] == "split"]
    extra, own_repaired = {}, None
    if variant == "capture":
        for layout in LAYOUT_BUILDS.get(size, ()):
            extra[layout] = _layout_build(size, folder, set_folder, layout, jobs, work_path, log,
                                          acceptance_sets.BUILD_TIMEOUT[size])
        if size in LAYOUT_BUILDS:
            own_repaired = _set_repair(size, builds.LAYOUT, set_folder, out, store, build_summary, work_path, jobs,
                                       acceptance_sets.BUILD_TIMEOUT[size], log)
            placed = _place_successors(own_repaired, out, builds.LAYOUT, f"set-{size}")
            own_repaired = {**own_repaired, "successor_shards": placed["shards"]}
    return {"size": size, "label": label, "build": str(out), "store": str(store), "counted": str(counted_file),
            "chunk_files": chunk_files, "copy_of": copies, "prefixes": prefixes, "split_pairs": splits, "tie": results,
            "layouts": extra, "repaired": own_repaired, "steps": log.steps}


def _run_tasks(tasks: Sequence[tuple[str, Any, tuple]], jobs: int) -> dict[str, dict]:
    """Run the tasks: in order with one worker, else up to ``jobs`` of them at a time, each in a
    process of its own."""
    if jobs == 1:
        return {label: function(*arguments) for label, function, arguments in tasks}
    context = multiprocessing.get_context("spawn")
    found: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(jobs, len(tasks)), mp_context=context) as pool:
        futures = {label: pool.submit(function, *arguments) for label, function, arguments in tasks}
        for label, future in futures.items():
            try:
                found[label] = future.result()
            except BaseException as error:
                for other in futures.values():
                    other.cancel()
                raise WorldError(f"the build {label} failed: {type(error).__name__}: {error}") from error
    return found


# --- What the table asks of the world ------------------------------------------------------------------


def _routed_givens(extensions: Collection[str] = ()) -> list[tuple[Any, Mapping[str, Any]]]:
    """Each question the table routes to a subject, with the ``given`` of its listed row. The rows of
    an extension are asked only while it is among ``extensions`` (the table's own rule)."""
    givens = dolma._listed_givens()
    return [(question, givens[question.listed_row]) for question in table10.expand(table10.load_table10(), extensions)
            if question.route is not None]


def _successor_artifacts(extensions: Collection[str] = ()) -> tuple[tuple[str, ...], str | None]:
    """The shards of a repaired side the routed rows name (``successor of E/shard-3``) and the
    source whose withdrawal makes them, read from the ``request`` of the same rows."""
    table = table10._acceptance_table()
    names: set[str] = set()
    sources: set[str] = set()
    for _, given in _routed_givens(extensions):
        artifact = given.get("artifact")
        if isinstance(artifact, str) and artifact.startswith(SUCCESSOR_PREFIX) and dolma._kind_of(artifact) == "shard":
            names.add(artifact)
            request = table.REQUESTS[given["request"]]
            if request.kind != "withdraw" or request.source is None:
                raise WorldError(f"the row of {artifact} names the request {given['request']!r}, which is no withdrawal of a source")
            sources.add(request.source)
    if len(sources) > 1:
        raise WorldError(f"the rows of the successors name more than one source: {sorted(sources)}")
    return tuple(sorted(names)), (next(iter(sources)) if sources else None)


def _stopped_targets(extensions: Collection[str] = ()) -> set[tuple[str, str]]:
    """The targets of the questions that stop the writer of the toolkit and run it again."""
    return {tuple(record.attempt_target(question)) for question, _ in _routed_givens(extensions)
            if question.subject == "dolma" and question.listed_row in KILL_ROWS}


def _record_id(build: Path, source: str, handle: str) -> str:
    """The id of the record of ``source`` the table names by ``handle``: the line the replica's
    manifest of the source file gives it, read from the file itself."""
    table = table10._acceptance_table()
    path = build / "sources" / f"{source}.jsonl"
    manifest = Path(f"{path}.manifest.jsonl")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").split("\n") if line]
    wanted = table.manifest_handles(handle)
    lines = [row["line"] for row in rows if row["handle"] in wanted]
    if len(lines) != 1:
        raise WorldError(f"the manifest of {path.name} holds {handle!r} on {len(lines)} lines; one was expected")
    text = path.read_bytes().decode("utf-8").split("\n")[lines[0] - 1]
    return json.loads(text)["id"]


def _ids_of(path: Path) -> set[str]:
    found = set()
    for line in path.read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict) and "id" in row:
            found.add(row["id"])
    return found


def _requests(main_build: Path, extensions: Collection[str] = ()) -> dict[str, dict]:
    """Each request word a routed row carries, with the record files of its source (relative to a
    build) and the id of the one record it narrows to, read from the main build and the spec."""
    table = table10._acceptance_table()
    with open(SPEC, "rb") as handle:
        spec = tomllib.load(handle)
    words = sorted({given["request"] for _, given in _routed_givens(extensions) if "request" in given})
    found: dict[str, dict] = {}
    for word in words:
        if word not in table.REQUESTS:
            raise WorldError(f"the table carries a request word {word!r} that its reader does not know")
        request = table.REQUESTS[word]
        if request.kind != "withdraw":
            continue
        if len(request.records) > 1:
            raise WorldError(f"{word!r} narrows to more than one record")
        position = None
        if request.records and request.records[0] in SPEC_POSITION_OF_HANDLE:
            position = spec["positions"][SPEC_POSITION_OF_HANDLE[request.records[0]]]
        source = request.source if request.source is not None else position["source"]
        relative = f"sources/{source}.jsonl"
        if not (main_build / relative).is_file():
            raise WorldError(f"the main build holds no {relative}, which {word!r} names")
        entry: dict[str, Any] = {"files": [relative]}
        if request.records:
            handle = request.records[0]
            record_id = position["record"] if position is not None else _record_id(main_build, source, handle)
            if record_id not in _ids_of(main_build / relative):
                raise WorldError(f"{relative} holds no record {record_id}, which {word!r} names")
            entry["record"] = record_id
        found[word] = entry
    return found


# --- The world file --------------------------------------------------------------------------------------


def _relative(path, directory: Path) -> str:
    relative = os.path.relpath(os.path.normpath(str(path)), os.path.normpath(str(directory)))
    if relative.startswith("..") or os.path.isabs(relative):
        raise WorldError(f"{path} lies outside the world folder {directory}")
    return Path(relative).as_posix()


def _entry(path, directory: Path) -> dict[str, str]:
    return {"path": _relative(path, directory), "sha256": _sha256_of(path)}


def _repository_entry(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(REPO).as_posix(), "sha256": _sha256_of(path), "repository": True}


def _state_value(item: Mapping[str, Any], directory: Path) -> dict[str, Any]:
    """The trace of one starting state: its document, or why the tool gave none, and the segments
    the tool refused by name with its words."""
    value: dict[str, Any] = ({"document": _entry(item["document"], directory)} if item["document"] is not None
                             else {"none": item["none"]})
    if item["failed"]:
        value["failed"] = dict(item["failed"])
    return value


def _trace_value(found: Mapping[str, Any] | None, directory: Path, none: str) -> dict[str, Any]:
    """The trace inputs of one store: the exported document, the documents of the starting
    states (the store that has them), the segments that gave none; or why there is none."""
    if found is None:
        return {"none": none}
    if found["document"] is None:
        return {"none": found["none"], "failed": dict(found["failed"])}
    value: dict[str, Any] = {"document": _entry(found["document"], directory)}
    if found["failed"]:
        value["failed"] = dict(found["failed"])
    if found.get("states") is not None:
        value["state_traces"] = {
            state: (_state_value(item, directory)) for state, item in found["states"].items()}
    return value


def _listed_paths(world: Mapping[str, Any], builds_listed: Mapping[str, Path], directory: Path) -> list[str]:
    """Every file the world names, by its path relative to the folder, in path order: the files of
    each key of the world file, the record files and chunk files of every build, the request
    files each build holds, and the files ``subject_inputs`` names."""
    found: set[Path] = {Path(world["tokenizer"]), Path(world["tie_record"])}
    for key in ("counted", "counted_by_layout", "counted_no_capture", "oracle_logs"):
        found.update(Path(path) for path in world.get(key, {}).values())
    for key in ("index", "index_no_capture"):
        for files in world.get(key, {}).values():
            found.update(Path(path) for path in files.values())
    for entry in world["stores"]:
        found.update(Path(entry["build"]) / relative for relative in entry["chunk_files"].values())
        # The file a copy was made from (W9) is a file the world names: it is hashed whether or not another key lists it.
        found.update(Path(entry["build"]) / relative for relative in (entry.get("copy_of") or {}).values())
    for entry in world["shards"]:
        found.add(Path(entry["build"]) / entry["chunk_file"])
        found.update(Path(entry["idx_prefix"] + suffix) for suffix in (".idx", ".bin"))
    for build in builds_listed.values():
        found.update(dolma._record_files(build))
        if (build / "built-from.jsonl").is_file():
            found.add(build / "built-from.jsonl")
        for wanted in world["requests"].values():
            found.update(build / relative for relative in wanted["files"] if (build / relative).is_file())
    stack: list[Any] = [world["subject_inputs"]]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str) and node.get("repository") is not True:
                found.add(directory / node["path"])
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return sorted({_relative(path, directory) for path in found})


def _store_entry(result: Mapping[str, Any], stopped: set[tuple[str, str]]) -> dict[str, Any]:
    entry: dict[str, Any] = {"target": ["store", result["name"]], "build": result["build"],
                             "chunk_files": dict(result["chunk_files"])}
    if ("store", result["name"]) in stopped:
        entry["stop_and_run_again"] = True
    return entry


def _run_of(shard: str) -> str | None:
    """The run a shard of a store belongs to: the name before its slash, where the fixture spec
    holds a run of that name."""
    prefix = shard.split("/", 1)[0]
    return prefix if prefix in _acceptance("acceptance_runs").spec()["runs"] else None


def _layout_parts(sizes: Sequence[str], sets: Mapping[str, dict]) -> dict[str, Any]:
    """What the builds ADDED beside a set's own (W1: the 10k set in the partitioned layout) put into the world: a Dolma entry for
    each, naming its layout (the key of the world's entries is the target and the layout), the shard entries of OriginBlame,
    the counting document of each layout (``counted_by_layout``: the layout to the document, the set's own round-robin document
    included, which holds only while ONE size has two builds) and the tie results and builds to list. Pure over the results of the
    set tasks."""
    entries: list[dict[str, Any]] = []
    shard_entries: list[dict[str, Any]] = []
    successor_entries: list[dict[str, Any]] = []
    counted: dict[str, str] = {}
    ties: dict[str, Sequence[Mapping[str, Any]]] = {}
    listed: dict[str, str] = {}
    repaired: dict[str, str] = {}
    sized = [size for size in sizes if sets[size].get("layouts")]
    if len(sized) > 1:
        raise WorldError(f"counted_by_layout is keyed by the layout alone: the sets {sized} each hold builds in two layouts")
    for size in sized:
        counted[builds.LAYOUT] = sets[size]["counted"]
        for layout, result in sets[size]["layouts"].items():
            entries.append({"target": ["set", size], "layout": layout, "build": result["build"],
                            "chunk_files": dict(result["chunk_files"]), "copy_of": dict(result["copy_of"])})
            for shard in result["chunk_files"]:
                shard_entries.append({"name": shard, "target": ["set", size], "layout": layout, "run": None, "successor": False,
                                      "build": result["build"], "chunk_file": result["chunk_files"][shard],
                                      "idx_prefix": str(Path(result["build"]) / result["prefixes"][shard])})
            counted[layout] = result["counted"]
            ties[f"set-{size}-{layout}"] = result["tie"]
            listed[f"set-{size}-{layout}"] = result["build"]
            repaired[f"set-{size}-{layout}"] = result["repaired"]["out"]
        # The repaired side of the set's own build (the round-robin layout) and of each added build, by the key of the build.
        repaired[f"set-{size}"] = sets[size]["repaired"]["out"]
        # The successor shards of the repaired sides, the set's own build first, then each added build: OriginBlame shards flagged
        # successor, in the layout of their build, each inside its build (``_place_successors``). The assembly puts them LAST.
        for layout, build, repair in ([(builds.LAYOUT, sets[size].get("build"), sets[size]["repaired"])]
                                      + [(name, result["build"], result["repaired"]) for name, result in sets[size]["layouts"].items()]):
            for name, placed in (repair.get("successor_shards") or {}).items():
                successor_entries.append({"name": name, "target": ["set", size], "layout": layout, "run": None, "successor": True,
                                          "build": build, "chunk_file": placed["chunk_file"],
                                          "idx_prefix": str(Path(build) / placed["prefix"])})
    return {"entries": entries, "shards": shard_entries, "successor_shards": successor_entries, "counted_by_layout": counted,
            "ties": ties, "builds": listed, "repaired": repaired}


def _assemble(directory: Path, tokenizer: Path, sizes: Sequence[str], stores: Mapping[str, dict],
              sets: Mapping[str, dict], nc_sets: Mapping[str, dict] | None = None,
              shardless: Mapping[str, dict] | None = None,
              extensions: Collection[str] = ()) -> tuple[dict[str, Any], dict[str, Path]]:
    work = directory / WORK
    nc_sets, shardless = nc_sets or {}, shardless or {}
    added = _layout_parts(sizes, sets)
    stopped = _stopped_targets(extensions)
    world: dict[str, Any] = {"tokenizer": str(tokenizer), "extensions": sorted(extensions)}
    merged = merge_tie_results({**{f"store-{name}": stores[name]["tie"] for name in STORE_ORDER},
                                **{f"set-{size}": sets[size]["tie"] for size in sizes},
                                **{f"set-{NO_CAPTURE}-{size}": nc_sets[size]["tie"] for size in nc_sets},
                                **added["ties"]})
    tie_path = work / "tie.json"
    tie.write_result(tie_path, merged, {"tokenizer": "the line tokenizer of the Megatron environment, "
                                                     "with the settings of the fixture spec"})
    world["tie_record"] = str(tie_path)
    world["requests"] = _requests(Path(stores["main"]["build"]), extensions)
    entries = [_store_entry(stores[name], stopped) for name in STORE_ORDER]
    for size in sizes:
        # The plant list's split pairs of THIS set (W8, 18 Nov 2026: a question reads exactly one build; one list from the smallest
        # set was read for every size and the 100k questions were put against the small set's pairs).
        entries.append({"target": ["set", size], "build": sets[size]["build"],
                        "chunk_files": dict(sets[size]["chunk_files"]), "copy_of": dict(sets[size]["copy_of"]),
                        "split_pairs": sets[size]["split_pairs"]})
    for size in sorted(nc_sets, key=sizes.index):
        entries.append({"target": ["set", f"{NO_CAPTURE}-{size}"], "build": nc_sets[size]["build"],
                        "chunk_files": dict(nc_sets[size]["chunk_files"]), "copy_of": dict(nc_sets[size]["copy_of"])})
    entries.extend(added["entries"])
    world["stores"] = entries
    shard_entries: list[dict[str, Any]] = []
    for name in SHARD_STORES:
        result = stores[name]
        for shard in sorted(result["shards"]):
            chunk, prefix = result["shards"][shard]
            shard_entries.append({"name": shard, "target": ["store", name], "layout": _layout_of(chunk),
                                  "run": _run_of(shard), "successor": False, "build": result["build"],
                                  "chunk_file": chunk, "idx_prefix": str(Path(result["build"]) / prefix)})
    for size in sizes:
        result = sets[size]
        for shard in result["chunk_files"]:
            shard_entries.append({"name": shard, "target": ["set", size], "layout": builds.LAYOUT, "run": None,
                                  "successor": False, "build": result["build"],
                                  "chunk_file": result["chunk_files"][shard],
                                  "idx_prefix": str(Path(result["build"]) / result["prefixes"][shard])})
    for size in sorted(nc_sets, key=sizes.index):
        result = nc_sets[size]
        for shard in result["chunk_files"]:
            shard_entries.append({"name": shard, "target": ["set", f"{NO_CAPTURE}-{size}"], "layout": builds.LAYOUT,
                                  "run": None, "successor": False, "build": result["build"],
                                  "chunk_file": result["chunk_files"][shard],
                                  "idx_prefix": str(Path(result["build"]) / result["prefixes"][shard])})
    shard_entries.extend(added["shards"])
    # The successors of a shard of a store come last, so that every shard number and every step id
    # that follows from the order of the shards stays; each lies inside the build of its store.
    for name in SHARD_STORES:
        result = stores[name]
        for artifact in sorted(result["successor_prefixes"]):
            chunk = result["chunk_files"][artifact]
            shard_entries.append({"name": artifact, "target": ["store", name], "layout": _layout_of(chunk),
                                  "run": _run_of(artifact), "successor": True, "build": result["build"],
                                  "chunk_file": chunk,
                                  "idx_prefix": str(Path(result["build"]) / result["successor_prefixes"][artifact])})
    # The successors of the shards of the sets (the 10k set's repaired sides) come after those of the stores, last of all.
    shard_entries.extend(added["successor_shards"])
    world["shards"] = shard_entries
    world["oracle_logs"] = {run: path for name in STORE_ORDER if stores[name].get("runs")
                            for run, path in stores[name]["runs"]["oracle_logs"].items()}
    world["counted"] = {size: sets[size]["counted"] for size in sizes}
    if added["counted_by_layout"]:
        world["counted_by_layout"] = dict(added["counted_by_layout"])
    world["index"] = {size: {shard: str(Path(sets[size]["build"]) / f"{prefix}.idx")
                             for shard, prefix in sets[size]["prefixes"].items()} for size in sizes}
    if nc_sets:
        # The build made without capture has its own counting document and index: its chunk files differ from the capture
        # set's of the same size (the runner reads these keys for the rows that name that build).
        world["counted_no_capture"] = {size: nc_sets[size]["counted"] for size in nc_sets}
        world["index_no_capture"] = {size: {shard: str(Path(nc_sets[size]["build"]) / f"{prefix}.idx")
                                            for shard, prefix in nc_sets[size]["prefixes"].items()} for size in nc_sets}
    builds_listed: dict[str, Path] = {f"store-{name}": Path(stores[name]["build"]) for name in STORE_ORDER}
    builds_listed.update({f"store-{name}": Path(shardless[name]["build"]) for name in SHARDLESS_STORES if name in shardless})
    builds_listed.update({f"set-{size}": Path(sets[size]["build"]) for size in sizes})
    builds_listed.update({f"set-{NO_CAPTURE}-{size}": Path(nc_sets[size]["build"]) for size in nc_sets})
    builds_listed.update({label: Path(build) for label, build in added["builds"].items()})
    world["subject_inputs"] = _subject_inputs(directory, world, stores, sets, sizes, builds_listed,
                                              nc_sets=nc_sets, shardless=shardless, repaired_sets=added["repaired"])
    return world, builds_listed


def _megatron_files() -> dict[str, dict[str, str]]:
    """The two files of the Megatron environment the line tokenizer reads (the vocabulary and
    the merges the fixture spec names), each by its name inside the environment folder and its
    sha256: the one reading of "the environment folder is hashed" the world makes."""
    with open(SPEC, "rb") as handle:
        settings = tomllib.load(handle)["tokenizer"]
    root = Path(os.environ[MEGATRON_VARIABLE])
    return {key: {"file": settings[key], "sha256": _sha256_of(root / settings[key])}
            for key in ("vocab_file", "merges_file")}


def _subject_inputs(directory: Path, world: Mapping[str, Any], stores: Mapping[str, dict], sets: Mapping[str, dict],
                    sizes: Sequence[str], builds_listed: Mapping[str, Path], nc_sets: Mapping[str, dict] | None = None,
                    shardless: Mapping[str, dict] | None = None, repaired_sets: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """What the four subjects the runner does not drive read: the file-level manifest of every
    build, the trace documents of every store that has runs, the frozen files of the repository
    the trace reads its places from, and the keys of the world whose files they read."""
    manifest: dict[str, Any] = {}
    for label, build in builds_listed.items():
        file = build / "built-from.jsonl"
        manifest[label] = ({"built_from": _entry(file, directory)} if file.is_file()
                           else {"none": "a build made without capture writes no file-level manifest"})
    repaired = stores["main"]["repaired"]
    if repaired is not None and (Path(repaired) / "built-from.jsonl").is_file():
        manifest["repaired-main"] = {"built_from": _entry(Path(repaired) / "built-from.jsonl", directory)}
    for label, folder in (repaired_sets or {}).items():
        # The repaired side of a generated set, one for each build the set has (W2): ``repaired-set-10k`` for its own build and
        # ``repaired-set-10k-partitioned`` for the build added beside it (the names ``subjects.repaired_key`` reads).
        manifest[f"repaired-{label}"] = {"built_from": _entry(Path(folder) / "built-from.jsonl", directory)}
    traces: dict[str, Any] = {}
    for name in STORE_ORDER:
        reason = (f"the table registers no run in the store {name}; the stores that hold runs are "
                  f"{', '.join(sorted(builds.RUNS_OF_STORE))}")
        traces[f"store-{name}"] = _trace_value(stores[name]["runs"], directory, reason)
    for name in SHARDLESS_STORES:
        if name in (shardless or {}):
            traces[f"store-{name}"] = {"none": f"the table registers no run in the store {name}; the stores that hold "
                                               f"runs are {', '.join(sorted(builds.RUNS_OF_STORE))}"}
    for size in sizes:
        traces[f"set-{size}"] = {"none": "a generated set registers no run"}
    for size in (nc_sets or {}):
        traces[f"set-{NO_CAPTURE}-{size}"] = {"none": "a generated set registers no run"}
    table = REPO / "fixture" / "table"
    return {
        "manifest": manifest,
        "trace": traces,
        "repository": [_repository_entry(table / "check01.toml"), _repository_entry(SPEC)],
        "megatron_tokenizer_files": _megatron_files(),
        "records": {label: sorted(path.relative_to(build).as_posix() for path in dolma._record_files(build))
                    for label, build in builds_listed.items()},
        "world_keys": ["tie_record", "requests", "stores", "shards", "index", "counted", "oracle_logs",
                       *(["counted_by_layout"] if world.get("counted_by_layout") else []),
                       *(["index_no_capture", "counted_no_capture"] if nc_sets else [])],
        "tracked_order": [entry["name"] for entry in world["shards"]],
    }


def _cpu_seconds() -> float:
    """The processor time of this process and of every process it started and waited for."""
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def _busy(cpu: float, total: float) -> str:
    """How many cores the command kept busy on average, in words."""
    return f"cores busy on average: {cpu / max(total, 0.001):.1f} of {os.cpu_count()} cores"


def _report(directory: Path, jobs: int, sizes: Sequence[str], world: Mapping[str, Any], listed: int,
            steps: Sequence[tuple[str, float]], total: float, builds_listed: Mapping[str, Path],
            cpu: float) -> str:
    lines = [
        f"world folder: {directory.name}",
        f"workers used: {jobs}",
        f"sizes built: {', '.join(sizes)}",
        f"builds: {len(builds_listed)} ({', '.join(sorted(builds_listed))})",
        f"stores of Dolma: {len(world['stores'])}; shards of OriginBlame: {len(world['shards'])}",
        f"files named by world.json and hashed: {listed} hashes in {HASH_FILE}",
        f"total seconds: {total:.1f}",
        f"cpu seconds (this command and every process it started): {cpu:.1f}",
        _busy(cpu, total),
        "seconds of each step:",
    ]
    lines += [f"  {label}: {seconds} seconds" for label, seconds in steps]
    exported = world["subject_inputs"]["trace"]
    lines.append("trace documents:")
    for key in sorted(exported):
        value = exported[key]
        if "document" in value:
            lines.append(f"  {key}: exported")
            for state, item in sorted(value.get("state_traces", {}).items()):
                lines.append(f"  {key} {state}: " + ("exported" if "document" in item else f"none ({item['none']})"))
                for segment, why in sorted(item.get("failed", {}).items()):
                    lines.append(f"  {key} {state} segment {segment}: none ({why})")
            for segment, why in sorted(value.get("failed", {}).items()):
                lines.append(f"  {key} segment {segment}: none ({why})")
        else:
            lines.append(f"  {key}: none ({value['none']})")
    return "\n".join(lines) + "\n"


# --- The command --------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m baselines.world",
        description="Build the stores and sets the baselines are run on, name every file in world.json "
                    "and hash every file it names in WORLD.sha256.")
    parser.add_argument("--out", required=True, type=Path, help="the new folder the world is built in (refused if it exists)")
    parser.add_argument("--megatron-env", required=True, type=Path, dest="megatron_env",
                        help="the Megatron environment folder (WATERWHITE_MEGATRON_ENV)")
    parser.add_argument("--tokenizer", required=True, type=Path, help="the tokenizer file; it is copied into the world and hashed")
    parser.add_argument("--jobs", type=int, default=1, help="workers: builds that run at the same time, and the job count of every builder")
    parser.add_argument("--sizes", nargs="+", choices=builds.SET_SIZES, default=[builds.SET_SIZES[0]],
                        help="the generated sets to build (default: small only)")
    parser.add_argument("--extensions", nargs="*", default=[], metavar="NAME",
                        help="the extensions whose rows the table asks (default: none). The world records them in "
                             "world.json, and a pass with an extension on is refused on a world built without it")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    directory = Path(os.path.abspath(arguments.out))
    if directory.exists():
        print(f"{directory} exists: a world is built in a new folder", file=sys.stderr)
        return 2
    if arguments.jobs < 1:
        print("--jobs is a number from 1 up", file=sys.stderr)
        return 2
    extensions = tuple(sorted(set(arguments.extensions)))
    unbuilt = sorted(set(extensions) - set(BUILT_EXTENSIONS))
    if unbuilt:
        print(f"--extensions names {unbuilt}: the builder has no build for it (it builds {list(BUILT_EXTENSIONS) or 'none'})",
              file=sys.stderr)
        return 2
    known = {row.extension for row in table10.load_table10().rows if row.extension is not None}
    if not set(extensions) <= known:
        print(f"--extensions names {sorted(set(extensions) - known)}; the table holds the extensions {sorted(known)}",
              file=sys.stderr)
        return 2
    tokenizer, megatron = Path(os.path.abspath(arguments.tokenizer)), Path(os.path.abspath(arguments.megatron_env))
    if not tokenizer.is_file():
        print(f"the tokenizer file {tokenizer} is not a file", file=sys.stderr)
        return 2
    if not megatron.is_dir():
        print(f"the Megatron environment folder {megatron} is not a folder", file=sys.stderr)
        return 2
    sizes = [size for size in builds.SET_SIZES if size in arguments.sizes]
    jobs = arguments.jobs
    os.environ[MEGATRON_VARIABLE] = str(megatron)
    os.environ.setdefault("HF_HOME", str(megatron / "hf_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.monotonic()
    cpu_started = _cpu_seconds()
    work = directory / WORK
    work.mkdir(parents=True)
    steps: list[tuple[str, float]] = []
    began = time.monotonic()
    copied = work / "inputs" / tokenizer.name
    copied.parent.mkdir()
    shutil.copyfile(tokenizer, copied)
    steps.append(("copy the tokenizer file", round(time.monotonic() - began, 1)))
    successors, source = _successor_artifacts(extensions)
    tasks: list[tuple[str, Any, tuple]] = [
        ("store-main", _store_task, ("main", str(work / "stores" / "main"), jobs, str(work), successors, source, extensions))]
    for size in reversed(sizes):
        tasks.append((f"set-{size}", _set_task, (size, str(work / "sets" / size), jobs, str(work))))
    for size in reversed(sizes):
        tasks.append((f"set-{NO_CAPTURE}-{size}", _set_task,
                      (size, str(work / "sets" / f"{NO_CAPTURE}-{size}"), jobs, str(work), NO_CAPTURE)))
    for name in STORE_ORDER[1:]:
        tasks.append((f"store-{name}", _store_task, (name, str(work / "stores" / name), jobs, str(work), (), None)))
    for name in SHARDLESS_STORES:
        tasks.append((f"store-{name}", _shardless_task, (name, str(work / "stores" / name), jobs, str(work))))
    try:
        done = _run_tasks(tasks, jobs)
    except WorldError as error:
        print(f"the world was not built: {error}", file=sys.stderr)
        return 1
    except TieMergeError as error:
        print(f"the world was not built: {error}", file=sys.stderr)
        return 1
    stores = {name: done[f"store-{name}"] for name in STORE_ORDER}
    shardless = {name: done[f"store-{name}"] for name in SHARDLESS_STORES}
    sets = {size: done[f"set-{size}"] for size in sizes}
    nc_sets = {size: done[f"set-{NO_CAPTURE}-{size}"] for size in sizes}
    for label in done:
        steps.extend(tuple(step) for step in done[label]["steps"])
    began = time.monotonic()
    try:
        world, builds_listed = _assemble(directory, copied, sizes, stores, sets, nc_sets, shardless, extensions)
    except (WorldError, TieMergeError) as error:
        print(f"the world was not built: {error}", file=sys.stderr)
        return 1
    steps.append(("assemble the world", round(time.monotonic() - began, 1)))
    began = time.monotonic()
    listed = _listed_paths(world, builds_listed, directory)
    world_file = _write_json(directory / WORLD_FILE, world)
    lines = [(path, _sha256_of(directory / path)) for path in sorted(set(listed + [WORLD_FILE]))]
    (directory / HASH_FILE).write_text("".join(f"{digest}  {path}\n" for path, digest in lines), encoding="utf-8")
    steps.append(("hash the files", round(time.monotonic() - began, 1)))
    total, cpu = time.monotonic() - started, _cpu_seconds() - cpu_started
    report = _report(directory, jobs, sizes, world, len(lines), steps, total, builds_listed, cpu)
    (directory / REPORT_FILE).write_text(report, encoding="utf-8")
    print(f"world built in {directory.name}: {len(builds_listed)} builds, {len(lines)} files hashed in {HASH_FILE}, "
          f"{jobs} workers, {total:.1f} seconds, {cpu:.1f} cpu seconds, {_busy(cpu, total)}; "
          f"{world_file.name}, {HASH_FILE} and {REPORT_FILE} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
