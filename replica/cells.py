"""The cell runner of plan 5.6 (docs/replica.md section 17).

A cell is one workload of the measurement of plan 5.6 (W7): a number of records, a record
length, derivations per source record, a number of sources and a layout, made by
``fixture/plant/generate.py``. For each cell the runner builds the set with
``python -m replica build-set`` at two worker settings and with capture off and on, in three
paired runs, and records the figures of every run. The command line:

    python -m replica.cells grid
    python -m replica.cells run   --out DIR --host-type TYPE --tree ID [options]
    python -m replica.cells rerun --out DIR --run ID --host-type TYPE --tree ID [options]
    python -m replica.cells level --records DIR...

Run on its own it reads nothing of the tool except through the tool's commands: it calls
``replica`` and ``waterwhite.cli`` as programs, unchanged, and keeps what they print. The
only code of the replica it imports is the reader of the tokenizer settings, to name the
workers value of Megatron's preprocessing.

Files of a batch folder: ``records/<run id>.json`` (one per finished or failed run),
``pairs/<pair id>.json`` (one per recorded pair), ``held/<run id>.json`` (a run the
before-hook did not let start), ``hooks.jsonl``, ``refused_pairs.json``,
``capture_wiring.json`` and ``work/<run id>/`` (the run's own folder).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]

SEED = 5606  # the one seed of every generated cell (plan 5.6)
RECORDS_10K = 10000
RECORDS_100K = 100000
LENGTHS = (500, 2000, 8000)
DERIVATIONS = (0, 1, 2)
SOURCES = (4, 10)
WORKERS = (1, 4)
PAIRS = (1, 2, 3)
LEVELS = (2, 3, 4, 6, 8, 12, 16)
SAMPLE_INTERVAL_SECONDS = 0.1
MIDDLE_LENGTH, MIDDLE_DERIVATIONS, MIDDLE_SOURCES = 2000, 1, 4

NOT_APPLICABLE = "not applicable"
NOT_MEASURED = "not measured"

LAYOUT_PARTITIONED = "partitioned"
LAYOUT_ROUND_ROBIN = "round-robin"
HARNESS_IN_PROCESS = "in-process"
HARNESS_CONFINED = "confined"

LEVEL_LABEL = "a measurement for the owner's rule, never a 5.6 figure"
HARNESS_STATEMENT = ("in-process: the generation jobs ran in the build's own process, and an in-process "
                     "log establishes no bound on what a job could have reached")
CONFINED_LABEL = ("the confined harness, measured once for the middle cell at both worker settings: a "
                  "separate figure, outside the 240 runs of plan 5.6")
MEGATRON_WORKERS_FROM = ("the [megatron] workers value of the tokenizer spec the build is handed with "
                         "--tokenizer-spec, read as tokenizer.workers by replica.plan.load_tokenizer_settings")
WORKERS_MEANING = ("python -m replica build-set --jobs {jobs}: the number of tokenizer runs at a time "
                   "(one run per shard file of the build); not Megatron's own workers value, which is "
                   "recorded beside it")

# The figures that exist only where a store exists.
ON_ONLY_FIGURES = ("store_file_bytes", "query_seconds", "proof_walk_seconds", "metadata_bytes",
                   "repair_bytes")
LEVEL_METRICS = ("throughput_records_per_second", "query_seconds", "proof_walk_seconds")
MATCHED_KEYS = ("input_sha256", "shard_order", "tokenizer_sha256")
HOOK_OUTPUT_CHARACTERS = 200

WINDOW_HOLDER = "TIMED WINDOW held by replica.cells\n"


class CellError(Exception):
    """A refusal of the runner itself (a window file that exists, a bad plan)."""


class PairMismatch(CellError):
    """The two halves of a pair differ in input, shard order or tokenizer."""

    def __init__(self, differs: Sequence[str]):
        self.differs = list(differs)
        super().__init__("the pair differs in " + ", ".join(self.differs))


class LevelRefused(CellError):
    """The level report does not answer: too few alone runs, or mixed cells."""


class RunFailed(Exception):
    """Raised by a step that failed. ``commands`` are the commands it had run."""

    def __init__(self, message: str, exit_code: int = 1, commands: Sequence[Mapping] = ()):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code
        self.commands = [dict(c) for c in commands]


# --- The grid and the plan ----------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    records: int
    length: int
    derivations: int
    sources: int
    layout: str
    generator_extra: tuple = ()

    @property
    def name(self) -> str:
        base = (f"n{self.records}-len{self.length}-d{self.derivations}-s{self.sources}-{self.layout}")
        if self.generator_extra:
            base += "-" + "-".join(str(part).lstrip("-") for part in self.generator_extra)
        return base

    def as_dict(self) -> dict:
        return {"records": self.records, "length": self.length, "derivations": self.derivations,
                "sources": self.sources, "layout": self.layout,
                "generator_extra": list(self.generator_extra)}


@dataclass(frozen=True)
class RunSpec:
    cell: Cell
    jobs: int
    capture: bool
    pair: int
    harness: str = HARNESS_IN_PROCESS

    @property
    def pair_id(self) -> str:
        suffix = "" if self.harness == HARNESS_IN_PROCESS else f"-{self.harness}"
        return f"{self.cell.name}-j{self.jobs}-p{self.pair}{suffix}"

    @property
    def run_id(self) -> str:
        return f"{self.pair_id}-{'on' if self.capture else 'off'}"


def cell_grid() -> list[Cell]:
    """The 20 cells of plan 5.6: 18 at 10k records in the partitioned layout, the middle cell
    also in the round-robin layout, and the middle cell at 100k records."""
    cells = [Cell(RECORDS_10K, length, derivations, sources, LAYOUT_PARTITIONED)
             for length in LENGTHS for derivations in DERIVATIONS for sources in SOURCES]
    cells.append(Cell(RECORDS_10K, MIDDLE_LENGTH, MIDDLE_DERIVATIONS, MIDDLE_SOURCES, LAYOUT_ROUND_ROBIN))
    cells.append(Cell(RECORDS_100K, MIDDLE_LENGTH, MIDDLE_DERIVATIONS, MIDDLE_SOURCES, LAYOUT_PARTITIONED))
    return cells


def middle_cell() -> Cell:
    return Cell(RECORDS_10K, MIDDLE_LENGTH, MIDDLE_DERIVATIONS, MIDDLE_SOURCES, LAYOUT_PARTITIONED)


def run_plan(cells: Iterable[Cell] | None = None) -> list[RunSpec]:
    """The runs: per cell, two worker settings, three pairs, an off run and an on run each
    (12 a cell, 240 for the grid). All of them run the in-process harness."""
    chosen = cell_grid() if cells is None else list(cells)
    return [RunSpec(cell, jobs, capture, pair)
            for cell in chosen for jobs in WORKERS for pair in PAIRS for capture in (False, True)]


def confined_plan(cell: Cell | None = None) -> list[RunSpec]:
    """The confined harness, measured once for the middle cell: both worker settings, three
    runs each, capture on. It is outside the 240."""
    chosen = middle_cell() if cell is None else cell
    return [RunSpec(chosen, jobs, True, pair, HARNESS_CONFINED) for jobs in WORKERS for pair in PAIRS]


def run_folders(work: Path, spec: RunSpec) -> dict:
    """The folders of one run, inside its work folder."""
    work = Path(os.path.abspath(work))
    return {"work": str(work), "set": str(work / "set"), "build": str(work / "build"),
            "store": str(work / "store.sqlite") if spec.capture else NOT_APPLICABLE,
            "probe": str(work / "probe"), "repaired": str(work / "repaired")}


# --- The commands -------------------------------------------------------------------------------


def generate_argv(cell: Cell, out: Path, *, python: str, repo: Path | None = None) -> list[str]:
    root = Path(repo) if repo is not None else REPO
    argv = [python, str(root / "fixture" / "plant" / "generate.py"),
            "--out", str(out), "--records", str(cell.records), "--seed", str(SEED),
            "--length", str(cell.length), "--derivations", str(cell.derivations),
            "--sources", str(cell.sources), "--dedup", "on"]
    return argv + [str(part) for part in cell.generator_extra]


def build_argv(spec: RunSpec, *, set_dir: Path, build_dir: Path, store: Path, python: str,
               tokenizer_spec: Path, tokenizer_timeout: float | None = None) -> list[str]:
    argv = [python, "-m", "replica", "build-set", "--set", str(set_dir), "--out", str(build_dir),
            "--layout", spec.cell.layout, "--tokenizer-spec", str(tokenizer_spec),
            "--jobs", str(spec.jobs)]
    if tokenizer_timeout is not None:
        argv += ["--tokenizer-timeout", str(tokenizer_timeout)]
    if spec.capture:
        argv += ["--capture", "--store", str(store), "--list-join", "single",
                 "--generation-harness", spec.harness]
    return argv


def probe_argv(build_dir: Path, store: Path, out: Path, layout: str, *, steps: int | None = None,
               timeout: float | None = None, python: str) -> list[str]:
    argv = [python, "-m", "replica", "probe-run", "--set-build", str(build_dir), "--store", str(store),
            "--out", str(out), "--layout", layout]
    if steps is not None:
        argv += ["--steps", str(steps)]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    return argv


def checkpoint_artifact(steps: int) -> str:
    """The checkpoint probe-run registers at its last step (docs/replica.md section 14)."""
    return f"probe@{steps}"


def query_argv(store: Path, artifact: str | None, *, python: str) -> list[str]:
    """``waterwhite query withdraw A``: with ``--artifact`` the one answer of that artifact, without
    it the listing of the whole store, which makes the proof of every artifact it answers."""
    argv = [python, "-m", "waterwhite.cli", "query", "withdraw", "A", "--store", str(store)]
    if artifact is not None:
        argv += ["--artifact", artifact]
    return argv


# --- What is read from the machine and from files --------------------------------------------------


def read_machine(*, role: str, host_type: str, tree: str, proc_root: Path | str = "/proc",
                 etc_root: Path | str = "/etc") -> dict:
    """The machine, read when the run starts: cores and memory from ``proc_root``, the operating
    system from ``etc_root``. The role, the type and the tree id are the caller's."""
    proc, etc = Path(proc_root), Path(etc_root)
    cores = sum(1 for line in (proc / "cpuinfo").read_text(encoding="utf-8").splitlines()
                if re.match(r"processor\s*:", line))
    ram = None
    for line in (proc / "meminfo").read_text(encoding="utf-8").splitlines():
        found = re.match(r"MemTotal:\s+(\d+)\s*kB", line)
        if found:
            ram = int(found.group(1)) * 1024
            break
    values = {}
    for line in (etc / "os-release").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip().strip('"').strip("'")
    name = values.get("PRETTY_NAME") or " ".join(
        part for part in (values.get("NAME"), values.get("VERSION")) if part)
    if cores < 1 or ram is None or not name:
        raise CellError(f"the machine cannot be read from {proc} and {etc}")
    return {"host_role": role, "type": host_type, "cores": cores, "ram_bytes": ram, "os": name,
            "tree": tree}


def file_digests(folder: Path) -> dict:
    """The sha256 of every file under ``folder``, by its path relative to it."""
    root = Path(folder)
    found = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        found[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return found


def tokenizer_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def shard_order(build: Path, layout: str) -> list[str]:
    """The shard names the build wrote for ``layout``, in the order the run reads them (digit
    runs read as numbers, docs/replica.md section 14)."""
    folder = Path(build) / "pipeline" / layout / "shards"
    names = {path.name[: -len(suffix)] for suffix in (".bin", ".idx") for path in folder.glob(f"*{suffix}")}
    return sorted(names, key=_natural_key)


def folder_bytes(folder: Path) -> int:
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


def check_matched(off: Mapping, on: Mapping) -> dict:
    """The three values that make a pair one pair (plan 5.6: input, order and tokenizer
    matched): the sha256 of every file of the generated input, the shard order the build wrote
    and the sha256 of the tokenizer spec. Equal: they are returned. Else ``PairMismatch``."""
    differs = [key for key in MATCHED_KEYS if off.get(key) != on.get(key)]
    missing = [key for key in MATCHED_KEYS if key not in off or key not in on]
    if differs or missing:
        raise PairMismatch(sorted(set(differs + missing), key=MATCHED_KEYS.index))
    return {key: off[key] for key in MATCHED_KEYS}


# --- Figures ------------------------------------------------------------------------------------------


def median_throughput(records: Iterable[Mapping]) -> float:
    """The median of the throughput of the runs given: runs that finished, with no sampler."""
    values = [r["figures"]["throughput_records_per_second"] for r in records
              if r.get("exit") == 0 and not r.get("sampler")
              and isinstance(r.get("figures", {}).get("throughput_records_per_second"), (int, float))]
    if not values:
        raise CellError("no finished run without the sampler to take a median of")
    return statistics.median(values)


def peak_from_samples(samples: Sequence[Sequence[int]]) -> tuple[int, int]:
    """The greatest sum of the resident sizes of one sample, and the greatest single size."""
    sums = [sum(sample) for sample in samples if sample]
    largest = [max(sample) for sample in samples if sample]
    if not sums:
        return 0, 0
    return max(sums), max(largest)


def _process_table() -> dict:
    """pid -> (parent pid, resident bytes) of every process, read from /proc."""
    page = os.sysconf("SC_PAGE_SIZE")
    table = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as handle:
                data = handle.read()
            rest = data[data.rindex(b")") + 2:].split()
            parent = int(rest[1])
            with open(f"/proc/{entry}/statm", "rb") as handle:
                resident = int(handle.read().split()[1])
        except (OSError, ValueError, IndexError):
            continue
        table[int(entry)] = (parent, resident * page)
    return table


def _tree_sizes(root: int) -> list[int]:
    """The resident sizes of ``root`` and of every process below it, by parent links."""
    table = _process_table()
    children: dict[int, list[int]] = {}
    for pid, (parent, _size) in table.items():
        children.setdefault(parent, []).append(pid)
    sizes, queue, seen = [], [root], set()
    while queue:
        pid = queue.pop()
        if pid in seen or pid not in table:
            continue
        seen.add(pid)
        sizes.append(table[pid][1])
        queue.extend(children.get(pid, []))
    return sizes


_STOP = threading.Event()  # set when a batch is interrupted: every running step ends its process


def _end_process(process: subprocess.Popen) -> None:
    """End a step's process by its pid: SIGTERM, then SIGKILL after 30 seconds."""
    if process.poll() is not None:
        return
    try:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def _drive(process: subprocess.Popen, *, sampler: bool, timeout: float | None) -> dict:
    """Wait for ``process``; with ``sampler`` sample the resident sizes of its process tree every
    SAMPLE_INTERVAL_SECONDS. The wall time is measured around the wait itself."""
    started = time.perf_counter()
    samples: list[list[int]] = []
    try:
        while True:
            if sampler:
                sample = _tree_sizes(process.pid)
                if sample:
                    samples.append(sample)
            try:
                process.wait(timeout=SAMPLE_INTERVAL_SECONDS)
                break
            except subprocess.TimeoutExpired:
                pass
            if _STOP.is_set():
                _end_process(process)
                raise KeyboardInterrupt
            if timeout is not None and time.perf_counter() - started > timeout:
                _end_process(process)
                wall = time.perf_counter() - started
                return {"exit": 124, "wall_seconds": wall, "samples": samples, "timed_out": True}
    except BaseException:
        _end_process(process)
        raise
    wall = time.perf_counter() - started
    return {"exit": process.returncode, "wall_seconds": wall, "samples": samples, "timed_out": False}


def sample_command(argv: Sequence[str], *, cwd: Path | None = None, env: Mapping | None = None,
                   stdout=None, stderr=None, timeout: float | None = None) -> dict:
    """Run ``argv`` and sample the sum of the resident sizes of its process tree every 0.1 s.
    Returns the exit, the wall seconds, the samples (one list of sizes per sample) and the greatest
    sum and the greatest single size."""
    process = subprocess.Popen(list(argv), cwd=cwd, env=None if env is None else dict(env),
                               stdout=stdout, stderr=stderr, start_new_session=True)
    result = _drive(process, sampler=True, timeout=timeout)
    peak_sum, peak_largest = peak_from_samples(result["samples"])
    return {"exit": result["exit"], "wall_seconds": result["wall_seconds"], "samples": result["samples"],
            "peak_sum_bytes": peak_sum, "peak_largest_bytes": peak_largest}


def _copy_store(store: Path, copy_dir: Path) -> Path:
    """A copy of the store file (and its write-ahead file, if one stands beside it) in ``copy_dir``."""
    copy_dir = Path(copy_dir)
    copy_dir.mkdir(parents=True, exist_ok=True)
    target = copy_dir / Path(store).name
    n = 0
    while target.exists():
        n += 1
        target = copy_dir / f"{Path(store).name}.{n}"
    shutil.copy2(store, target)
    wal = Path(str(store) + "-wal")
    if wal.is_file():
        shutil.copy2(wal, Path(str(target) + "-wal"))
    return target


METADATA_TABLES = {"records": ("occurrence",), "parent_links": ("edge_input",),
                   "step_traces": ("step_trace", "trace_sample", "trace_row")}


def metadata_bytes(store_file: Path, copy_dir: Path) -> dict:
    """Bytes of metadata per record, per parent link and per stored step trace.

    The bytes of a table are the pages of its b-tree and of every index of it, summed from
    ``dbstat`` on a COPY of the store file that is kept in ``copy_dir`` (the store itself is
    never opened). Records are the rows of ``occurrence``, parent links the rows of
    ``edge_input``, stored step traces the rows of ``step_trace``, whose bytes include
    ``trace_sample`` and ``trace_row``. A kind with no row has no figure per row: "not measured"."""
    copy = _copy_store(Path(store_file), Path(copy_dir))
    try:
        connection = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
        connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.OperationalError:
        connection = sqlite3.connect(f"file:{copy}?mode=ro&immutable=1", uri=True)
    try:
        sizes, rows = {}, {}
        for kind, tables in METADATA_TABLES.items():
            total = 0
            for table in tables:
                names = [name for (name,) in connection.execute(
                    "SELECT name FROM sqlite_master WHERE tbl_name = ? AND type IN ('table', 'index')",
                    (table,))]
                for name in names:
                    (pages,) = connection.execute(
                        "SELECT coalesce(sum(pgsize), 0) FROM dbstat WHERE name = ?", (name,)).fetchone()
                    total += pages
            sizes[kind] = total
            rows[kind] = connection.execute(f"SELECT count(*) FROM {tables[0]}").fetchone()[0]
    finally:
        connection.close()

    def per(kind: str):
        return sizes[kind] / rows[kind] if rows[kind] else NOT_MEASURED

    return {"per_record": per("records"), "per_parent_link": per("parent_links"),
            "per_step_trace": per("step_traces"),
            "rows": {"records": rows["records"], "parent_links": rows["parent_links"],
                     "step_traces": rows["step_traces"]}}


def repair_bytes(*, historical: Path, store_before: int, store_after: int, repaired: Path) -> dict:
    """What a repair reads and rewrites. It reads the historical build (every file) and the store
    as it stood; it rewrites the repaired build (every file) and the store's growth."""
    return {"read_bytes": folder_bytes(historical) + store_before,
            "rewritten_bytes": folder_bytes(repaired) + max(store_after - store_before, 0)}


def capture_wiring(repo: Path | None = None) -> dict:
    """The lines that wire capture into one generation job of the replica: a static count of the
    code lines (no blank line, comment or docstring) of the two methods of ``StoreCapture`` a
    generation job passes through, with the lines themselves and the two places the build calls them."""
    root = Path(repo) if repo is not None else REPO
    path = "replica/capture_hooks.py"
    source = (root / path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StoreCapture")
    lines = source.splitlines()
    methods = ["generate", "record_generation"]
    counted: list[int] = []
    for name in methods:
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == name)
        skip: set[int] = set()
        first = method.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            skip = set(range(first.lineno, first.end_lineno + 1))
        for number in range(method.lineno, method.end_lineno + 1):
            text = lines[number - 1].strip()
            if number in skip or not text or text.startswith("#"):
                continue
            counted.append(number)
    build = (root / "replica" / "build.py").read_text(encoding="utf-8").splitlines()
    sites = [{"file": "replica/build.py", "line": number} for number, text in enumerate(build, 1)
             if "capture.generate(" in text or "capture.record_generation(" in text]
    return {"file": path, "class": "StoreCapture", "methods": methods, "lines": counted,
            "count": len(counted), "call_sites": sites,
            "what": "code lines of the two methods one captured generation job passes through"}


# --- The steps of a real run ----------------------------------------------------------------------------


def _relative(path: Path, folder: Path) -> str:
    return os.path.relpath(os.path.abspath(path), os.path.abspath(folder))


class CommandSteps:
    """The real steps of a run: generate the cell, build it, and with capture on probe-run, query,
    walk, count the store's bytes and measure a repair. ``repair`` is a callable that stands in
    for the repair step; it is handed one mapping and returns ``{"read_bytes", "rewritten_bytes"}``."""

    def __init__(self, *, repo: Path | None = None, python: str | None = None,
                 tokenizer_spec: Path | None = None, probe_steps: int | None = None,
                 repair: Callable[[Mapping], Mapping] | None = None,
                 tokenizer_timeout: float = 3600.0, probe_timeout: float = 3600.0) -> None:
        self.repo = Path(repo) if repo is not None else REPO
        self.python = python or sys.executable
        self.tokenizer_spec = (Path(tokenizer_spec) if tokenizer_spec is not None
                               else self.repo / "fixture" / "spec" / "fixture_spec.toml")
        self.probe_steps = probe_steps
        self.repair = repair
        self.tokenizer_timeout = tokenizer_timeout
        self.probe_timeout = probe_timeout

    def _env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(self.repo / "src"), str(self.repo)])
        env.setdefault("HF_HUB_OFFLINE", "1")
        root = env.get("WATERWHITE_MEGATRON_ENV", "")
        if root:
            env.setdefault("HF_HOME", str(Path(root) / "hf_home"))
        return env

    def measure(self, spec: RunSpec, work: Path, *, sampler: bool) -> dict:
        work = Path(os.path.abspath(work))
        folders = run_folders(work, spec)
        batch = work.parent.parent
        logs = work / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = self._env()
        commands: list[dict] = []

        def step(name: str, argv: Sequence[str], *, sampled: bool = False,
                 stdout_to: Path | None = None) -> dict:
            out_path = Path(stdout_to) if stdout_to is not None else logs / f"{name}.log"
            err_path = logs / f"{name}.err"
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                if sampled:
                    done = sample_command(argv, cwd=self.repo, env=env, stdout=out, stderr=err)
                else:
                    process = subprocess.Popen(list(argv), cwd=self.repo, env=env, stdout=out,
                                               stderr=err, start_new_session=True)
                    done = _drive(process, sampler=False, timeout=None)
            command = {"name": name, "argv": [str(a) for a in argv], "exit": done["exit"],
                       "seconds": done["wall_seconds"], "output_file": _relative(out_path, batch),
                       "error_file": _relative(err_path, batch)}
            commands.append(command)
            if done["exit"] != 0:
                tail = err_path.read_text(encoding="utf-8", errors="replace")[-300:].strip()
                raise RunFailed(f"{name} exited {done['exit']}: {tail}", exit_code=done["exit"],
                                commands=commands)
            result = dict(done)
            result["command"] = command
            return result

        cell = spec.cell
        step("generate", generate_argv(cell, Path(folders["set"]), python=self.python, repo=self.repo))
        input_sha = file_digests(Path(folders["set"]))
        store = Path(folders["store"]) if spec.capture else Path(work / "no-store")
        built = step("build-set",
                     build_argv(spec, set_dir=Path(folders["set"]), build_dir=Path(folders["build"]),
                                store=store, python=self.python, tokenizer_spec=self.tokenizer_spec,
                                tokenizer_timeout=self.tokenizer_timeout),
                     sampled=sampler)
        raw: dict[str, Any] = {
            "wall_seconds": built["wall_seconds"],
            "build_folder_bytes": folder_bytes(Path(folders["build"])),
            "megatron_workers": _megatron_workers(self.tokenizer_spec),
            "input_sha256": input_sha,
            "shard_order": shard_order(Path(folders["build"]), cell.layout),
            "tokenizer_sha256": tokenizer_digest(self.tokenizer_spec),
            "peak_memory": ({"sum_bytes": built["peak_sum_bytes"],
                             "largest_process_bytes": built["peak_largest_bytes"]} if sampler else None),
            "commands": commands,
        }
        if spec.capture:
            self._with_store(spec, work, folders, store, step, raw)
        return raw

    def _with_store(self, spec: RunSpec, work: Path, folders: dict, store: Path, step, raw: dict) -> None:
        cell = spec.cell
        probe_out = Path(folders["probe"])
        step("probe-run", probe_argv(Path(folders["build"]), store, probe_out, cell.layout,
                                     steps=self.probe_steps, timeout=self.probe_timeout,
                                     python=self.python))
        printed = json.loads(Path(work / "logs" / "probe-run.log").read_text(encoding="utf-8"))
        steps_run = int(printed["steps"])
        query = step("query", query_argv(store, checkpoint_artifact(steps_run), python=self.python))
        walk = step("proof-walk", query_argv(store, None, python=self.python))
        raw["query_seconds"] = query["wall_seconds"]
        raw["proof_walk_seconds"] = walk["wall_seconds"]
        raw["store_file_bytes"] = os.path.getsize(store)
        raw["metadata_bytes"] = metadata_bytes(store, work / "dbstat-copy")
        context = {"python": self.python, "repo": str(self.repo), "work": str(work), "store": str(store),
                   "build": folders["build"], "set": folders["set"], "layout": cell.layout,
                   "jobs": spec.jobs, "tokenizer_spec": str(self.tokenizer_spec),
                   "repaired": folders["repaired"], "store_copy": str(work / "store-repair.sqlite"),
                   "report": str(work / "report.json"), "step": step,
                   "tokenizer_timeout": self.tokenizer_timeout}
        raw["repair_bytes"] = dict((self.repair or _repair_step)(context))


def _megatron_workers(tokenizer_spec: Path) -> int:
    from .plan import load_tokenizer_settings

    return load_tokenizer_settings(Path(tokenizer_spec)).workers


def _repair_step(context: Mapping) -> dict:
    """The repair of the withdrawal of source A, on a copy of the store: the report the tool prints,
    then ``python -m replica repair``; the bytes are the file sizes before and after."""
    step = context["step"]
    store_copy = _copy_store(Path(context["store"]), Path(context["work"]) / "repair-store")
    before = os.path.getsize(store_copy)
    step("report", [context["python"], "-m", "waterwhite.cli", "report", "withdraw", "A", "--store",
                    str(store_copy)], stdout_to=Path(context["report"]))
    step("repair", [context["python"], "-m", "replica", "repair", "--report", context["report"],
                    "--historical", context["build"], "--out", context["repaired"], "--store",
                    str(store_copy), "--set", context["set"], "--tokenizer-spec",
                    context["tokenizer_spec"], "--layout", context["layout"], "--jobs",
                    str(context["jobs"]), "--tokenizer-timeout", str(context["tokenizer_timeout"])])
    return repair_bytes(historical=Path(context["build"]), store_before=before,
                        store_after=os.path.getsize(store_copy), repaired=Path(context["repaired"]))


# --- The batch -----------------------------------------------------------------------------------------


@dataclass
class BatchResult:
    finished: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    refused_pairs: list = field(default_factory=list)
    stopped: dict | None = None


def _write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_records(out: Path) -> dict:
    found = {}
    folder = Path(out) / "records"
    if folder.is_dir():
        for path in sorted(folder.glob("*.json")):
            found[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _call_hook(command: str | Sequence[str] | None) -> dict | None:
    """One hook: the command, its exit status and the first 200 characters of its output."""
    if command is None:
        return None
    argv = shlex.split(command) if isinstance(command, str) else [str(part) for part in command]
    try:
        done = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              errors="replace")
        status, text = done.returncode, done.stdout
    except OSError as error:
        status, text = 127, f"{type(error).__name__}: {error}"
    return {"command": " ".join(argv), "exit": status, "output": text[:HOOK_OUTPUT_CHARACTERS]}


class _Running:
    """Which runs are going at the same time: the count beside each run at its start and its maximum."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running: dict[str, dict] = {}

    def enter(self, run_id: str) -> None:
        with self._lock:
            at_start = len(self._running)
            self._running[run_id] = {"at_start": at_start, "maximum": at_start}
            for entry in self._running.values():
                entry["maximum"] = max(entry["maximum"], len(self._running) - 1)

    def leave(self, run_id: str) -> dict:
        with self._lock:
            return self._running.pop(run_id)


def _record_base(spec: RunSpec, *, at_once: int, sampler: bool, machine: Mapping, hold: dict | None,
                 folders: dict, beside: dict) -> dict:
    record = {
        "run": spec.run_id, "pair": spec.pair_id, "pair_number": spec.pair, "cell": spec.cell.as_dict(),
        "jobs": spec.jobs, "capture": spec.capture, "sampler": sampler, "at_once": at_once,
        "beside": beside, "machine": dict(machine), "hold": hold, "folders": folders,
        "workers_meaning": WORKERS_MEANING.format(jobs=spec.jobs),
        "megatron_workers_from": MEGATRON_WORKERS_FROM,
    }
    if not spec.capture:
        record["harness"] = NOT_APPLICABLE
    else:
        record["harness"] = spec.harness
        if spec.harness == HARNESS_IN_PROCESS:
            record["harness_statement"] = HARNESS_STATEMENT
        else:
            record["label"] = CONFINED_LABEL
    return record


def _figures(spec: RunSpec, raw: Mapping, sampler: bool) -> dict:
    wall = raw["wall_seconds"]
    figures = {
        "wall_seconds": wall,
        "throughput_records_per_second": spec.cell.records / wall if wall > 0 else NOT_MEASURED,
        "peak_memory": (raw["peak_memory"] if sampler and raw.get("peak_memory") else NOT_MEASURED),
        "build_folder_bytes": raw["build_folder_bytes"],
    }
    for key in ON_ONLY_FIGURES:
        if not spec.capture:
            figures[key] = NOT_APPLICABLE
        elif key not in raw:
            raise RunFailed(f"the steps gave no figure {key} for a capture-on run")
        else:
            figures[key] = raw[key]
    return figures


def _execute_one(spec: RunSpec, out: Path, steps, *, at_once: int, sampler: bool, machine: Mapping,
                 hold: dict | None, running: _Running) -> dict:
    """One run: its own work folder, the steps, the record on disk. KeyboardInterrupt and SystemExit
    pass through and leave no record."""
    work = out / "work" / spec.run_id
    if work.exists() and any(work.iterdir()):
        n = 1
        while (out / "work" / f"{spec.run_id}.earlier-{n}").exists():
            n += 1
        work.rename(out / "work" / f"{spec.run_id}.earlier-{n}")  # an earlier stop: set aside, never deleted
    work.mkdir(parents=True, exist_ok=True)
    folders = run_folders(work, spec)
    running.enter(spec.run_id)
    failure = None
    raw: Mapping = {}
    commands: list = []
    try:
        raw = steps.measure(spec, work, sampler=sampler)
        commands = [dict(c) for c in raw.get("commands", [])]
        figures = _figures(spec, raw, sampler)
        matched = {key: raw[key] for key in MATCHED_KEYS}
    except RunFailed as stopped:
        failure = {"message": stopped.message, "exit": stopped.exit_code}
        commands = stopped.commands
    except Exception as error:  # a step that breaks its own contract is a failed run, kept in the record
        failure = {"message": f"{type(error).__name__}: {error}", "exit": 1}
    finally:
        beside = running.leave(spec.run_id)
    record = _record_base(spec, at_once=at_once, sampler=sampler, machine=machine, hold=hold,
                          folders=folders, beside=beside)
    record["commands"] = commands
    if failure is None:
        record.update(exit=0, failure=None, figures=figures, matched=matched,
                      megatron_workers=raw.get("megatron_workers"))
    else:
        record.update(exit=failure["exit"], failure=failure["message"], figures={}, matched=None,
                      megatron_workers=None)
    _write_json(out / "records" / f"{spec.run_id}.json", record)
    return record


def _run_group(group: Sequence[RunSpec], out: Path, steps, *, at_once: int, sampler: bool,
               machine: Mapping, holds: Mapping, running: _Running) -> list[dict]:
    if len(group) == 1:
        spec = group[0]
        return [_execute_one(spec, out, steps, at_once=at_once, sampler=sampler, machine=machine,
                             hold=holds.get(spec.run_id), running=running)]
    results: dict[str, Any] = {}

    def work(spec: RunSpec) -> None:
        try:
            results[spec.run_id] = _execute_one(spec, out, steps, at_once=at_once, sampler=sampler,
                                                machine=machine, hold=holds.get(spec.run_id),
                                                running=running)
        except BaseException as error:  # KeyboardInterrupt in a step stops the whole group
            results[spec.run_id] = error
            if isinstance(error, KeyboardInterrupt):
                _STOP.set()

    threads = [threading.Thread(target=work, args=(spec,), name=spec.run_id) for spec in group]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.1)
    except KeyboardInterrupt:
        _STOP.set()
        for thread in threads:
            thread.join()
        raise
    stopping = [r for r in results.values() if isinstance(r, BaseException)]
    for error in stopping:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise error
    for error in stopping:
        raise error
    return [results[spec.run_id] for spec in group]


def _settle_pairs(all_specs: Sequence[RunSpec], out: Path) -> list[dict]:
    """Write the pair file of every pair whose two halves finished and match; refuse, by name, a pair
    that differs. Read from the run records, so a resumed batch settles the pairs of the runs it skipped."""
    records = _read_records(out)
    refused = []
    halves: dict[str, list[RunSpec]] = {}
    for spec in all_specs:
        halves.setdefault(spec.pair_id, []).append(spec)
    for pair_id, members in halves.items():
        by_capture = {spec.capture: spec for spec in members}
        if set(by_capture) != {False, True}:
            continue
        off, on = (records.get(by_capture[flag].run_id) for flag in (False, True))
        if not off or not on or off.get("exit") != 0 or on.get("exit") != 0:
            continue
        path = Path(out) / "pairs" / f"{pair_id}.json"
        try:
            matched = check_matched(off["matched"], on["matched"])
        except PairMismatch as mismatch:
            refused.append({"pair": pair_id, "differs": mismatch.differs})
            if path.exists():
                os.replace(path, path.with_name(path.name + ".stale"))
            continue
        _write_json(path, {"pair": pair_id, "matched": matched,
                           "runs": [off["run"], on["run"]]})
    _write_json(Path(out) / "refused_pairs.json", refused)
    return refused


def _execute_batch(todo: Sequence[RunSpec], all_specs: Sequence[RunSpec], skipped: Sequence[str],
                   out: Path, steps, *, at_once: int, window_file, machine: Mapping, sampler: bool,
                   repo, before_run, after_run) -> BatchResult:
    if at_once < 1:
        raise CellError("--at-once is at least 1")
    ids = [spec.run_id for spec in all_specs]
    if len(set(ids)) != len(ids):
        raise CellError("two runs of the plan have one run id")
    out = Path(out)
    for folder in ("records", "pairs", "work"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    _write_json(out / "capture_wiring.json", capture_wiring(repo))
    result = BatchResult(skipped=list(skipped))
    running = _Running()
    _STOP.clear()
    window = None
    try:
        if window_file is not None and todo:
            window = Path(window_file)
            try:
                with open(window, "x", encoding="utf-8") as handle:
                    handle.write(WINDOW_HOLDER)
            except FileExistsError:
                window = None
                raise CellError(f"the window file {window_file} exists: another timed batch holds it") from None
        pending = list(todo)
        while pending:
            group, pending = pending[:at_once], pending[at_once:]
            hold = _call_hook(before_run)
            if hold is not None and hold["exit"] != 0:
                for spec in group:
                    _write_json(out / "held" / f"{spec.run_id}.json",
                                {"run": spec.run_id, "hold": hold,
                                 "status": "not started: the before-hook did not hold the box"})
                result.stopped = {"hook": hold, "runs": [spec.run_id for spec in group],
                                  "status": "stopped before a run: the batch resumes from here"}
                _log_hook(out, "before", group, hold)
                break
            if hold is not None:
                _log_hook(out, "before", group, hold)
            for spec in group:
                held = out / "held" / f"{spec.run_id}.json"
                if held.exists():
                    os.replace(held, held.with_name(held.name + ".earlier"))
            try:
                records = _run_group(group, out, steps, at_once=at_once, sampler=sampler, machine=machine,
                                     holds={spec.run_id: hold for spec in group}, running=running)
            finally:
                released = _call_hook(after_run)
                if released is not None:
                    _log_hook(out, "after", group, released)
            for record in records:
                if record["exit"] == 0:
                    result.finished.append(record["run"])
        result.refused_pairs = _settle_pairs(all_specs, out)
    finally:
        if window is not None:
            try:
                window.unlink()
            except FileNotFoundError:
                pass
    final = _read_records(out)
    result.failed = [run_id for run_id in ids if final.get(run_id, {}).get("exit", 0) != 0]
    return result


def _log_hook(out: Path, when: str, group: Sequence[RunSpec], hook: Mapping) -> None:
    line = {"when": when, "runs": [spec.run_id for spec in group], **hook}
    with open(Path(out) / "hooks.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, sort_keys=True) + "\n")


def run_batch(specs: Iterable[RunSpec], out: Path, steps, *, at_once: int = 1,
              window_file: Path | None = None, machine: Mapping, sampler: bool = False,
              repo: Path | None = None, before_run: str | Sequence[str] | None = None,
              after_run: str | Sequence[str] | None = None) -> BatchResult:
    """Run ``specs`` in plan order, ``at_once`` at a time. A run with a record of exit 0 is skipped
    (a stopped batch resumes); a run with a failed record stays failed and is not run again (see
    ``rerun``). While the batch runs it holds ``window_file``. ``before_run`` is called before each
    run (or each group of ``at_once`` runs) and the run starts only on status 0; ``after_run`` is
    called after each run or group, also after a failure or an interrupt."""
    all_specs = list(specs)
    existing = _read_records(Path(out))
    skipped = [spec.run_id for spec in all_specs if existing.get(spec.run_id, {}).get("exit") == 0]
    todo = [spec for spec in all_specs if spec.run_id not in existing]
    return _execute_batch(todo, all_specs, skipped, Path(out), steps, at_once=at_once,
                          window_file=window_file, machine=machine, sampler=sampler, repo=repo,
                          before_run=before_run, after_run=after_run)


def rerun(run_id: str, specs: Iterable[RunSpec], out: Path, steps, *, machine: Mapping,
          sampler: bool = False, repo: Path | None = None, window_file: Path | None = None,
          before_run=None, after_run=None, at_once: int = 1) -> BatchResult:
    """Run one named run again, whatever its record says. The earlier record is kept beside the new
    one under ``records/earlier/``, and the run's earlier work folder is set aside, never deleted."""
    all_specs = list(specs)
    chosen = [spec for spec in all_specs if spec.run_id == run_id]
    if not chosen:
        raise CellError(f"the plan has no run {run_id}")
    path = Path(out) / "records" / f"{run_id}.json"
    if path.exists():
        earlier = Path(out) / "records" / "earlier"
        earlier.mkdir(parents=True, exist_ok=True)
        n = 1
        while (earlier / f"{run_id}.{n}.json.kept").exists():
            n += 1
        os.replace(path, earlier / f"{run_id}.{n}.json.kept")
    return _execute_batch(chosen, all_specs, [], Path(out), steps, at_once=at_once,
                          window_file=window_file, machine=machine, sampler=sampler, repo=repo,
                          before_run=before_run, after_run=after_run)


# --- The level test -------------------------------------------------------------------------------------


def _metric(record: Mapping, key: str) -> float:
    value = record.get("figures", {}).get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LevelRefused(f"a run has no number for {key}: {value!r}")
    return float(value)


def level_report(records: Iterable[Mapping]) -> dict:
    """For each worker setting separately, the highest level (runs at once) at which throughput,
    query time and proof-walk time all stay inside the band the alone runs span (their minimum to
    maximum, both included). Bytes, counts and line counts are not in the band. It answers only with
    at least three alone runs for each worker setting."""
    usable = [r for r in records if r.get("exit") == 0 and r.get("capture") is True
              and not r.get("sampler") and r.get("harness", HARNESS_IN_PROCESS) == HARNESS_IN_PROCESS]
    cells = {json.dumps(r.get("cell"), sort_keys=True) for r in usable}
    if len(cells) > 1:
        raise LevelRefused("the records are of more than one cell: give the records of one cell")
    report: dict[str, Any] = {
        "label": LEVEL_LABEL,
        "cell": json.loads(next(iter(cells))) if cells else None,
        "jobs": {},
        "not_covered": ["cells of 8000 characters a record", "cells of 10 sources",
                        "the 100k cell, whose runs are ten times as large"],
        "covered_by": ("only a level run on that cell itself (its alone runs and its rising-level "
                       "runs); the level found for the middle cell does not carry to it, and until it "
                       "is run the heavier cells run alone, one run at a time"),
    }
    for jobs in WORKERS:
        alone = [r for r in usable if r.get("jobs") == jobs and r.get("at_once", 1) == 1]
        if len(alone) < 3:
            raise LevelRefused(f"three alone runs are needed for --jobs {jobs}, found {len(alone)}")
        band = {}
        for key in LEVEL_METRICS:
            values = [_metric(r, key) for r in alone]
            band[key] = [min(values), max(values)]
        levels = {}
        left_out = []
        for r in usable:
            level = r.get("at_once", 1)
            if r.get("jobs") == jobs and level > 1:
                # A short group (the last runs of a batch) never reached the level it was asked for:
                # the record's own count of runs beside it says how many really ran together.
                reached = r.get("beside", {}).get("maximum", level - 1) + 1
                if reached < level:
                    left_out.append({"run": r.get("run"), "at_once": level, "reached": reached})
                    continue
                levels.setdefault(level, []).append(r)
        detail = []
        for level in sorted(levels):
            outside = sorted({key for r in levels[level] for key in LEVEL_METRICS
                              if not band[key][0] <= _metric(r, key) <= band[key][1]})
            detail.append({"level": level, "runs": len(levels[level]), "inside": not outside,
                           "outside": outside})
        inside = [d["level"] for d in detail if d["inside"]]
        highest = max(inside) if inside else 1
        report["jobs"][str(jobs)] = {
            "alone_runs": len(alone), "band": band, "levels": detail, "highest_level": highest,
            "left_out_short_groups": left_out,
            "every_lower_level_inside": all(d["inside"] for d in detail if d["level"] < highest),
        }
    return report


# --- The command line -----------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m replica.cells", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("grid", help="print the cell and run counts and the cells")

    def batch_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--out", required=True, type=Path, help="the batch folder")
        sub.add_argument("--host-type", required=True, help="the machine type, for example CPX62")
        sub.add_argument("--host-role", default="server")
        sub.add_argument("--tree", required=True, help="the tree id the code of the run was taken from")
        sub.add_argument("--python", default=sys.executable)
        sub.add_argument("--repo", type=Path, default=REPO)
        sub.add_argument("--tokenizer-spec", type=Path, default=None)
        sub.add_argument("--tokenizer-timeout", type=float, default=3600.0)
        sub.add_argument("--at-once", type=int, default=1)
        sub.add_argument("--window-file", type=Path, default=None)
        sub.add_argument("--sampler", action="store_true",
                         help="sample peak memory every 0.1 s (wall times of such runs are flagged)")
        sub.add_argument("--before-run", default=None, help="a command called before each run or group")
        sub.add_argument("--after-run", default=None, help="a command called after each run or group")
        sub.add_argument("--cell", action="append", default=None, metavar="NAME",
                         help="only this cell of the grid (repeatable)")
        sub.add_argument("--confined", action="store_true",
                         help="the plan of the confined figure instead of the 240 runs")
        sub.add_argument("--only-capture", action="store_true", help="the capture-on runs only")
        sub.add_argument("--only-jobs", type=int, choices=WORKERS, default=None)

    batch_options(commands.add_parser("run", help="run a batch of the plan"))
    again = commands.add_parser("rerun", help="run one named run again")
    batch_options(again)
    again.add_argument("--run", required=True, help="the run id")
    level = commands.add_parser("level", help="the level report of the records of alone and rising runs")
    level.add_argument("--records", required=True, nargs="+", type=Path,
                       help="folders of run records (a batch folder's records folder)")
    return parser


def _selected(args: argparse.Namespace) -> list[RunSpec]:
    if args.confined:
        specs = confined_plan()
    else:
        grid = cell_grid()
        if args.cell:
            names = {cell.name: cell for cell in grid}
            unknown = [n for n in args.cell if n not in names]
            if unknown:
                raise CellError(f"no such cell: {unknown}; the names are those of `grid`")
            grid = [names[n] for n in args.cell]
        specs = run_plan(grid)
    if args.only_capture:
        specs = [s for s in specs if s.capture]
    if args.only_jobs is not None:
        specs = [s for s in specs if s.jobs == args.only_jobs]
    return specs


def _raise_interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "grid":
        grid = cell_grid()
        print(json.dumps({"cell_count": len(grid), "run_count": len(run_plan()),
                          "confined_run_count": len(confined_plan()),
                          "cells": [cell.name for cell in grid]}, indent=2))
        return 0
    if args.command == "level":
        records = []
        for folder in args.records:
            records += [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]
        try:
            report = level_report(records)
        except LevelRefused as refused:
            print(f"LevelRefused: {refused}", file=sys.stderr)
            return 2
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    try:
        specs = _selected(args)
        steps = CommandSteps(repo=args.repo, python=args.python, tokenizer_spec=args.tokenizer_spec,
                             tokenizer_timeout=args.tokenizer_timeout)
        machine = read_machine(role=args.host_role, host_type=args.host_type, tree=args.tree)
        signal.signal(signal.SIGTERM, _raise_interrupt)
        common = dict(machine=machine, sampler=args.sampler, repo=args.repo, window_file=args.window_file,
                      before_run=args.before_run, after_run=args.after_run, at_once=args.at_once)
        if args.command == "rerun":
            result = rerun(args.run, specs, args.out, steps, **common)
        else:
            result = run_batch(specs, args.out, steps, **common)
    except CellError as refused:
        print(f"cells: {type(refused).__name__}: {refused}", file=sys.stderr)
        return 2
    print(json.dumps({"finished": result.finished, "skipped": result.skipped, "failed": result.failed,
                      "refused_pairs": result.refused_pairs, "stopped": result.stopped}, indent=2))
    return 0 if not (result.failed or result.refused_pairs or result.stopped) else 1


if __name__ == "__main__":
    sys.exit(main())
