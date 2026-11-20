"""The world builder of Check 10 (plan 5.5 Check 10; docs/baselines.md, "The world file"):
``baselines.world``, run as ``python -m baselines.world``.

The builder is a command of the repository. It builds, with the repository's own builders called
unchanged, every store and set a run of the baselines is put to, names every file it made in one
file (``world.json``) that the runner reads, and hashes every file that file names. Nothing in it
is typed by hand: each input comes from a builder, from the frozen table or from the plant list,
or is a named input of the command line.

The module does not exist when these tests are written, so each test asks for it with ``need`` or
``need_attr`` in its own body, before anything is built: a missing module is a failed assertion.
The worlds are built once for the whole file by the real command with the real builders at the
smallest size (the small set): one with ``--jobs 1`` and one with ``--jobs 4``, and on a machine
with 24 GB of memory and eight cores or more a third with one worker for each core (at most
sixteen), all at the same time so that the machine is used; a smaller machine builds ``--jobs 1``
and then ``--jobs 2``, one after the other, because its memory does not hold the tokenizers of
several worlds. Every test that reads a world reads one of those folders.

The names this file asks of ``baselines.world``:

- ``python -m baselines.world --out DIR --megatron-env DIR --tokenizer FILE --jobs N
  [--sizes small [10k [100k]]]`` (default: the small set only). It refuses an ``--out`` that
  exists, writes nothing outside it, prints a summary, and exits 0 when it built the world.
- ``main(argv=None) -> int`` (the command's entry).
- ``merge_tie_results(results_by_build)``: ``results_by_build`` maps the name of a build to the
  list of results ``baselines.tie.show_chunk_file`` gave for its chunk files. It gives one result
  for each chunk file (the one the builds agree on) and raises ``TieMergeError`` (a
  ``ValueError``) naming the chunk file and both builds when two builds differ.
- ``TieMergeError``.

What a world folder holds (the shapes below are what these tests read):

- ``world.json`` (the format of "The world file"), ``WORLD.sha256`` (one line for each file,
  ``<64 hex digits>`` two spaces ``<path relative to the folder>``, world.json itself included),
  ``world-report.txt`` and ``work/`` (every build lies inside it). ``world.json`` is written last
  of what lies in ``work/``.
- The tokenizer given on the command line is copied into the folder, and ``tokenizer`` names the
  copy.
- ``stores`` has one entry for each of the five stores that hold shards (main, rr, no-capture,
  scratch-500, scratch-notokens) and one for the small set; ``shards`` has one entry for each
  shard of main (partitioned layout, run E), rr (round-robin layout, run E-rr), scratch-500 and
  the small set, and last of all the successor of the third shard of main: a shard named
  ``successor of E/shard-3``, flagged successor, of the target main, whose chunk file, index and
  packed file lie inside the build of main. ``index`` and ``counted`` hold the small set under the
  key ``small``.
- ``subject_inputs`` is an object with the keys ``manifest`` and ``trace``. A file it names is
  an object ``{"path": ..., "sha256": ...}`` (the path relative to the folder; for a file of the
  repository, relative to the repository and with ``"repository": true``). ``trace`` has one
  key for each Dolma entry, written ``<kind>-<name>`` of the store or set (``store-main``,
  ``set-small``); its value names the exported trace documents, or is ``{"none": "<why>"}``
  for a store with no run registered. The value of ``store-main``, where it names files, has
  ``state_traces`` with the four starting states of Check 2 as keys.
- A build made without capture draws its shuffles from the operating system's randomness
  (``replica/fixture.py``, ``replica/build.py``): the path of every file of such a build
  carries the words ``no-capture``. The seeded files of two worlds (one worker and several) have
  the same sha256; the three chunk files of that build hold the same lines over the shards
  together in any order (the read is set down before the tests of those files); every other
  file of that build has the same sha256. The sha256 of every file of one world is in
  ``WORLD.sha256``, and a changed byte in a copy of a world is refused by name of the file.

The values these tests expect come from the table, the README, the plant script and the
documents, and from files this file reads itself; never from the code under test.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tokenize
import tomllib
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"
MEGATRON_VARIABLE = "WATERWHITE_MEGATRON_ENV"

# The command's own limit for all the worlds together, below the limit of one test file.
BUILD_SECONDS = 3000.0
# A machine with this much memory and at least eight cores builds all the worlds at the same time,
# the last of them with one worker for each core (at most sixteen), so that it is used and not left
# idle; a smaller machine (a laptop under a memory cap) builds the one-worker world and then a
# world with four workers (the count of the worlds of record), one after the other, because several worlds with their
# tokenizers do not fit in its memory.
BIG_MEMORY_BYTES = 24 * 2**30
BIG_CORES = 8
MOST_WORKERS = 16
SEVERAL_WORKERS_SMALL = 4   # the worker count of the worlds of record (the passes are built with --jobs 4): compared, not only exercised


def _memory_limit() -> int:
    """The memory this process may use: the smallest limit set on its control group or on a group
    above it, or the memory of the machine."""
    limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        relative = next(line.split(":", 2)[2] for line in lines if line.startswith("0::"))
    except (OSError, StopIteration):
        return limit
    folder = Path("/sys/fs/cgroup") / relative.strip().lstrip("/")
    for candidate in [folder, *folder.parents]:
        if not str(candidate).startswith("/sys/fs/cgroup"):
            break
        try:
            held = (candidate / "memory.max").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if held.isdigit():
            limit = min(limit, int(held))
    return limit


def _plan() -> tuple[tuple[int, ...], bool]:
    """The worker counts of the worlds this machine builds, and whether they are built together."""
    cores = len(os.sched_getaffinity(0))
    if _memory_limit() >= BIG_MEMORY_BYTES and cores >= BIG_CORES:
        return (1, 4, min(cores, MOST_WORKERS)), True
    return (1, SEVERAL_WORKERS_SMALL), False


WORLDS, TOGETHER = _plan()
SEVERAL = WORLDS[1]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
ANY_SHA256 = "0" * 64

# The stores and the set the world builds as Dolma entries, as targets.
STORES_WITH_SHARDS = ("main", "rr", "no-capture", "scratch-500", "scratch-notokens")
# Changed 17 Nov 2026 (old: the five stores and the small set): the small set made without capture is a Dolma entry too.
DOLMA_TARGETS = tuple(("store", name) for name in STORES_WITH_SHARDS) + (("set", "small"), ("set", "no-capture-small"))
# The stores with no chunk file and no shard (swap and import): the tool holds no entry for them, and the questions of their rows
# are still asked of it, as the "none" rows the runner puts with the tool's attempts of the pass (D-SUBJ-9). Added 17 Nov 2026.
SHARDLESS_TARGETS = (("store", "swap"), ("store", "import"))
# The OriginBlame shards. Changed 17 Nov 2026 (old: main, rr, scratch-500 and the small set, "no entry for scratch-notokens,
# the table has no row for it"; the table holds OriginBlame rows for the stores no-capture and scratch-notokens and for the
# set made without capture, so each has its shards).
SHARD_TARGETS = (("store", "main"), ("store", "rr"), ("store", "no-capture"), ("store", "scratch-500"),
                 ("store", "scratch-notokens"), ("set", "small"), ("set", "no-capture-small"))
SUCCESSOR_KEY = "successor of E/shard-3"
# The shards of the stores of the fixture, by the names the table gives them. The successor of the
# third shard of main is the fourth shard of main and the last shard of the whole world.
SHARDS_OF = {
    "main": ("E/shard-1", "E/shard-2", "E/shard-3", SUCCESSOR_KEY),
    "rr": ("E-rr/shard-1", "E-rr/shard-2", "E-rr/shard-3"),
    "no-capture": ("E/shard-1", "E/shard-2", "E/shard-3"),
    "scratch-500": ("c04/shard-500",),
    "scratch-notokens": ("c03/shard-notokens",),
}
LAYOUT_OF = {"main": "partitioned", "rr": "round-robin", "no-capture": "partitioned"}
RUN_OF = {"main": "E", "rr": "E-rr", "no-capture": "E"}
CHECK_2_STATES = ("c2.i", "c2.ii", "c2.iii", "c2.iv")
REPOSITORY_INPUTS = ("fixture/table/check01.toml", "fixture/spec/fixture_spec.toml")

# The plant generator's own options for the small set are those of ``baselines.builds``; the
# lines that are not JSON it plants sit in Z (docs/plant.md, section 4.5).
PLANT_FILE = "plant_list.jsonl"

TOKENIZER_NAME = "tokenizer.json"
TOKENIZER_BYTES = b'{"stand_in": "a tokenizer file the world hashes and copies"}\n'

# What stays out of a tracked file, written in pieces so that this file holds none of it.
BANNED_WORDS = (
    "R2" + "A", "ROUND" + "2", "rul" + "ing", "amend" + "ed", "uns" + "ure", "BA" + "RS", "spec " + "1",
    "spec " + "section", "C" + "&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net", "advi" + "sor",
    "/ho" + "me/", "Proj" + "ects/",
)
BANNED_PATTERNS = (
    re.compile(r"\b[GQ]\d\b"),
    re.compile(r"\bitem \d"),
    re.compile(r"\bround \d", re.IGNORECASE),
    re.compile(r"\b(work" + r"flow|ag" + r"ent|br" + r"ief)\b", re.IGNORECASE),
    re.compile(chr(0x2014)),
)
# Words of how the module was built, banned in the comments and docstrings of the module only: the
# runner's own term for a step of its pre-run order stands in the document of the baselines, and
# one of them is the name of a builtin.
PROCESS_WORDS = re.compile(r"\b(sta" + r"ge|sli" + r"ce|wa" + r"ve)\b", re.IGNORECASE)


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    found = getattr(need(module_name), attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def _banned_in(text: str) -> list[str]:
    found = [word for word in BANNED_WORDS if word in text]
    found += [pattern.pattern for pattern in BANNED_PATTERNS if pattern.search(text)]
    return found


# --- The table, as these tests read it ------------------------------------------------------------


@lru_cache(maxsize=1)
def _core_rows() -> dict:
    """Every core row of Checks 1 to 9, by id, read from the table files."""
    found = {}
    for number in range(1, 10):
        with open(TABLE_DIR / f"check{number:02d}.toml", "rb") as handle:
            for row in tomllib.load(handle)["row"]:
                if row["core"]:
                    found[row["id"]] = row
    return found


@lru_cache(maxsize=1)
def _questions():
    table10 = need("baselines.table10")
    return table10.expand(table10.load_table10())


def _routed(subject: str):
    """The questions of one subject that have a route, with the target each is asked on."""
    record = need("baselines.record")
    return [(question, tuple(record.attempt_target(question)))
            for question in _questions() if question.subject == subject and question.route is not None]


def _given(question) -> dict:
    return _core_rows()[question.listed_row]["given"]


def _artifacts_named(targets_questions) -> set[str]:
    """The shard artifacts the routed questions of a store name: the artifacts of the kind
    ``shard`` (the runner's own ``dolma._kind_of``), the shards a fault is put to
    (``dolma.FAULT_SHARD``) and the shards the table's row of the counts of run E lists."""
    dolma = need("baselines.dolma")
    names: set[str] = set()
    for question in targets_questions:
        given = _given(question)
        artifact = given.get("artifact")
        if artifact is not None and dolma._kind_of(artifact) == "shard":
            names.add(artifact)
        if given.get("state") in dolma.FAULT_SHARD:
            names.add(dolma.FAULT_SHARD[given["state"]])
        if question.listed_row == "c09.fixture.shards.counts":
            expect = _core_rows()[question.listed_row]["expect"]
            names.update(expect["keep"])
            names.update(expect["rewrite"])
    return names


# --- The builds: the real command, once for the file ----------------------------------------------

_STATE: dict = {}


def _environment() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)])
    root = os.environ.get(MEGATRON_VARIABLE, "")
    if root:
        env.setdefault("HF_HOME", str(Path(root) / "hf_home"))
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _world_argv(directory: Path, megatron: str, tokenizer: Path, jobs: int) -> list[str]:
    return [sys.executable, "-m", "baselines.world", "--out", str(directory), "--megatron-env", str(megatron),
            "--tokenizer", str(tokenizer), "--jobs", str(jobs)]


def _build_worlds(factory) -> dict:
    """Build the worlds of ``WORLDS`` (one worker, several, and on a big machine one for each
    core): all at the same time where the machine holds them, else one after the other. Keep what
    each left. A build that runs over the limit is stopped by its own process group id."""
    root = os.environ.get(MEGATRON_VARIABLE, "")
    if not root:
        pytest.skip(f"needs the Megatron environment: {MEGATRON_VARIABLE} is not set")
    base = factory.mktemp("worlds")
    inputs = factory.mktemp("world-inputs")
    tokenizer = inputs / TOKENIZER_NAME
    tokenizer.write_bytes(TOKENIZER_BYTES)
    deadline = time.monotonic() + BUILD_SECONDS

    def start(jobs: int):
        directory = base / f"jobs-{jobs}"
        log = inputs / f"jobs-{jobs}.log"
        handle = open(log, "wb")
        process = subprocess.Popen(_world_argv(directory, root, tokenizer, jobs), cwd=REPO, env=_environment(),
                                   stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        return process, handle, directory, log

    def finish(jobs: int, started) -> SimpleNamespace:
        process, handle, directory, log = started
        try:
            code = process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code = "over the limit of the test"
        handle.close()
        text = log.read_text(encoding="utf-8", errors="replace")
        return SimpleNamespace(jobs=jobs, code=code, directory=directory, text=text, tokenizer=tokenizer)

    runs = {}
    if TOGETHER:
        started = {jobs: start(jobs) for jobs in WORLDS}
        for jobs in WORLDS:
            runs[jobs] = finish(jobs, started[jobs])
    else:
        for jobs in WORLDS:
            runs[jobs] = finish(jobs, start(jobs))
    return runs


def _built(factory, jobs: int) -> Path:
    """The folder of the world built with ``jobs`` workers. The module is asked for first, with
    the names its builders of a set and of the tie show must hold, so that a missing name fails the
    test at its assertion and starts no build."""
    need("baselines.world")
    for name in ("set_chunk_file", "name_set_chunk_files", "spread"):
        need_attr("baselines.world", name)
    if "worlds" not in _STATE:
        _STATE["worlds"] = _build_worlds(factory)
    run = _STATE["worlds"][jobs]
    assert run.code == 0, f"python -m baselines.world --jobs {jobs} ended with {run.code}: {run.text[-3000:]}"
    return run.directory


def _run_of(jobs: int):
    return _STATE["worlds"][jobs]


def _world_json(directory: Path) -> dict:
    return json.loads((directory / "world.json").read_text(encoding="utf-8"))


def _sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _listed(directory: Path) -> list[tuple[str, str]]:
    """The lines of ``WORLD.sha256``: (relative path, sha256), in file order."""
    found = []
    for line in (directory / "WORLD.sha256").read_text(encoding="utf-8").splitlines():
        parsed = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", line)
        assert parsed is not None, f"a line of WORLD.sha256 is not a hash, two spaces and a path: {line!r}"
        found.append((parsed.group(2), parsed.group(1)))
    return found


def _relative(directory: Path, value) -> str:
    """The path of a file the world names, relative to the world folder; a path outside the
    folder is a failed assertion."""
    path = Path(os.path.normpath(str(value)))
    base = Path(os.path.normpath(str(directory)))
    assert path.is_absolute(), f"{value} is not an absolute path"
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        raise AssertionError(f"{value} lies outside the world folder {directory}") from None


def _label(entry: dict) -> str:
    return "-".join(entry["target"])


def _named_files(world: dict) -> dict[str, Path]:
    """Every file ``world.json`` names, by a label, as an absolute path: the files of the keys of
    "The world file". The record files of a request are those that exist in each build."""
    found: dict[str, Path] = {}
    for key in ("tokenizer", "tie_record"):
        found[key] = Path(world[key])
    for key in ("counted", "counted_by_layout"):
        for size, path in world.get(key, {}).items():
            found[f"{key}[{size}]"] = Path(path)
    for size, files in world.get("index", {}).items():
        for shard, path in files.items():
            found[f"index[{size}][{shard}]"] = Path(path)
    builds: dict[str, Path] = {}
    for entry in world["stores"]:
        build = Path(entry["build"])
        builds[_label(entry)] = build
        for artifact, relative in entry["chunk_files"].items():
            found[f"stores[{_label(entry)}][{artifact}]"] = build / relative
        for artifact, relative in (entry.get("copy_of") or {}).items():
            found[f"stores[{_label(entry)}].copy_of[{artifact}]"] = build / relative
    for entry in world["shards"]:
        build = Path(entry["build"])
        builds[f"shard {entry['name']}"] = build
        found[f"shards[{entry['name']}].chunk_file"] = build / entry["chunk_file"]
        for suffix in (".idx", ".bin"):
            found[f"shards[{entry['name']}]{suffix}"] = Path(entry["idx_prefix"] + suffix)
    for word, wanted in world["requests"].items():
        for label, build in builds.items():
            for relative in wanted["files"]:
                if (build / relative).is_file():
                    found[f"requests[{word}] in {label}: {relative}"] = build / relative
    return found


def _file_entries(node):
    """Every object of a document that names a file and its hash."""
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(node.get("sha256"), str):
            yield node
        for value in node.values():
            yield from _file_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from _file_entries(value)


def _lines_of(path) -> int:
    """The lines of a chunk file as the tie counts them: split at the line feed, a last empty
    piece dropped."""
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return len(lines)


def _ids_of(path) -> set:
    """The ids of the lines of a JSONL file that are JSON objects."""
    found = set()
    for line in Path(path).read_bytes().split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except ValueError:
            continue
        if isinstance(row, dict) and "id" in row:
            found.add(row["id"])
    return found


def _dolma_entries(world: dict) -> dict[tuple[str, str], dict]:
    # The Dolma entry of each target: a build ADDED beside the target's own names its layout (the 10k set in the partitioned layout)
    # and is not the entry of the target (the world's key is the target and the layout; ``_layout_entries`` reads those).
    return {tuple(entry["target"]): entry for entry in world["stores"] if not entry.get("layout")}


# --- The plant: the generated set, counted by the plant script ------------------------------------


def _load_plant_script(name: str):
    path = REPO / "fixture" / "plant" / name
    spec = importlib.util.spec_from_file_location(f"world_test_plant_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _planted(factory) -> Path:
    """The small set written by the plant generator with the options of ``baselines.builds``,
    in a folder of its own (the same seed and options give the same bytes, docs/plant.md)."""
    if "plant" not in _STATE:
        builds = need("baselines.builds")
        folder = factory.mktemp("planted") / "set"
        done = subprocess.run(builds.generate_argv("small", folder, python=sys.executable), cwd=REPO,
                              capture_output=True, text=True, timeout=900, check=False)
        assert done.returncode == 0, done.stderr
        _STATE["plant"] = folder
    return _STATE["plant"]


def _planted_unparseable(folder: Path) -> dict[str, int]:
    """The lines that are not JSON, per file, counted from the plant list."""
    found: dict[str, int] = {}
    for line in (folder / PLANT_FILE).read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        row = json.loads(line)
        if row["sanitizer"] and row["sanitizer"]["case"] == "unparseable":
            found[row["file"]] = found.get(row["file"], 0) + 1
    return found


# --- The command ------------------------------------------------------------------------------------


def test_the_world_builder_is_a_module_that_runs_as_a_command_with_its_options():
    need_attr("baselines.world", "main")
    done = subprocess.run([sys.executable, "-m", "baselines.world", "--help"], cwd=REPO, env=_environment(),
                          capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-1000:]
    for option in ("--out", "--megatron-env", "--tokenizer", "--jobs", "--sizes"):
        assert option in done.stdout, option


def test_an_output_folder_that_exists_is_refused_before_anything_is_built(tmp_path):
    need("baselines.world")
    existing = tmp_path / "already-there"
    existing.mkdir()
    tokenizer = tmp_path / TOKENIZER_NAME
    tokenizer.write_bytes(TOKENIZER_BYTES)
    root = os.environ.get(MEGATRON_VARIABLE) or str(tmp_path)
    done = subprocess.run(_world_argv(existing, root, tokenizer, 1), cwd=REPO, env=_environment(),
                          capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode != 0, "a folder that exists was accepted"
    assert existing.name in done.stdout + done.stderr, "the refusal does not name the folder"
    assert list(existing.iterdir()) == [], "the refused folder was written into"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["already-there", TOKENIZER_NAME], \
        "a refused run wrote beside the folder"


# --- What the command leaves ----------------------------------------------------------------------


def test_the_command_leaves_its_files_prints_a_summary_and_writes_nothing_outside_its_folder(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    for name in ("world.json", "WORLD.sha256", "world-report.txt"):
        assert (directory / name).is_file(), name
    assert (directory / "work").is_dir()
    assert _run_of(SEVERAL).text.strip(), "the command printed no summary"
    assert sorted(path.name for path in directory.parent.iterdir()) == sorted(f"jobs-{jobs}" for jobs in WORLDS), \
        "something lies beside the output folders of the worlds"
    # world.json is written last of what lies in the work folder.
    newest = max(path.lstat().st_mtime_ns for path in (directory / "work").rglob("*") if not path.is_dir())
    assert (directory / "world.json").stat().st_mtime_ns >= newest, "a file of work/ is newer than world.json"


def test_the_report_counts_the_builds_the_files_the_hashes_the_seconds_and_the_workers(tmp_path_factory, capsys):
    for jobs in WORLDS:
        directory = _built(tmp_path_factory, jobs)
        report = (directory / "world-report.txt").read_text(encoding="utf-8")
        # The report of each world goes into the log of the run: the seconds of each step and the
        # cores kept busy are read from it.
        with capsys.disabled():
            print(f"\n--- world-report.txt of the world built with {jobs} workers ---\n{report}", flush=True)
        lines = report.splitlines()
        cpu = [line for line in lines if "cpu second" in line.lower()]
        assert cpu and re.search(r"\d+\.\d", cpu[0]), "the report holds no line of the cpu seconds"
        busy = [line for line in lines if "cores busy" in line.lower()]
        assert busy and re.search(r"\d+\.\d", busy[0]), "the report does not say how many cores were kept busy"
        assert "cores busy" in _run_of(jobs).text.lower(), "the summary the command prints does not say how many cores were busy"
        workers = [line for line in lines if "worker" in line.lower()]
        assert workers and any(re.search(rf"\b{jobs}\b", line) for line in workers), \
            f"the report does not say that {jobs} workers were used"
        hashes = [line for line in lines if "hash" in line.lower()]
        assert any(re.search(rf"\b{len(_listed(directory))}\b", line) for line in hashes), \
            "the report does not count the hashes of WORLD.sha256"
        assert any("second" in line.lower() for line in lines), "the report holds no seconds"
        assert any("build" in line.lower() for line in lines), "the report counts no builds"


# --- world.json ---------------------------------------------------------------------------------------


def test_world_json_names_the_stores_the_set_and_the_shards_the_runner_is_put_to(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    assert world["extensions"] == [], "world.json records the extensions the world was built for (none by default)"
    # Changed 18 Nov 2026 (old: "split_pairs" was a key of the world, the smallest set's list read for every size): the plant
    # list's split pairs are in the entry of each set (W8: a question reads exactly one build).
    for key in ("tokenizer", "tie_record", "requests", "stores", "shards", "counted", "index",
                "subject_inputs"):
        assert key in world, key
    assert "split_pairs" not in world, "one list of split pairs for every size is what box pass 9 read wrong at 100k"
    work = Path(os.path.normpath(str(directory / "work")))
    entries = _dolma_entries(world)
    assert sorted(entries) == sorted(DOLMA_TARGETS), "the stores of the Dolma entries are not the five and the small set"
    for target, entry in entries.items():
        build = Path(os.path.normpath(entry["build"]))
        assert build != work and work in build.parents, f"the build of {target} lies outside the work folder"
        assert entry["chunk_files"], target
        for artifact, relative in entry["chunk_files"].items():
            assert not os.path.isabs(relative) and ".." not in Path(relative).parts, (target, artifact)
            assert (build / relative).is_file(), f"{target}: the chunk file of {artifact} is not there"
        if target[0] == "set":
            # W9 (18 Nov 2026): a set's entry says where each chunk file was copied from, as a field (never by equal bytes).
            assert sorted(entry["copy_of"]) == sorted(entry["chunk_files"]), f"{target}: copy_of names other shards"
            for artifact, original in entry["copy_of"].items():
                assert (build / original).is_file(), f"{target}: the original of {artifact} is not in the build"
                assert (build / original).read_bytes() == (build / entry["chunk_files"][artifact]).read_bytes(), \
                    f"{target}: copy_of of {artifact} does not name the file the copy was made from"
    stopped = [target for target, entry in entries.items() if entry.get("stop_and_run_again")]
    assert stopped == [("store", "scratch-500")], "only the store whose writer is stopped runs again"
    # The shards of OriginBlame.
    by_target: dict[tuple[str, str], list[dict]] = {}
    for shard in world["shards"]:
        by_target.setdefault(tuple(shard["target"]), []).append(shard)
        build = Path(os.path.normpath(shard["build"]))
        assert work in build.parents, f"the build of the shard {shard['name']} lies outside the work folder"
        assert (build / shard["chunk_file"]).is_file(), shard["name"]
        for suffix in (".idx", ".bin"):
            assert Path(shard["idx_prefix"] + suffix).is_file(), (shard["name"], suffix)
    assert sorted(by_target) == sorted(SHARD_TARGETS), "one entry for each target a tool is put to"
    for name, wanted in SHARDS_OF.items():
        found = by_target[("store", name)]
        assert tuple(shard["name"] for shard in found) == wanted, name
        for shard in found:
            # Main holds four shards: the three of run E and the successor of the third, which is
            # flagged. Its layout and run are not set down here (no row of the table reads them).
            successor = shard["name"] == SUCCESSOR_KEY
            if name in LAYOUT_OF and not successor:
                assert shard.get("layout") == LAYOUT_OF[name], (name, shard["name"])
                assert shard.get("run") == RUN_OF[name], (name, shard["name"])
            assert bool(shard.get("successor", False)) is successor, (name, shard["name"])
    # The shards of the small set are those of its Dolma entry, in the one layout it is built in.
    set_shards = [shard["name"] for shard in by_target[("set", "small")]]
    assert sorted(set_shards) == sorted(entries[("set", "small")]["chunk_files"]), \
        "the set's shards are not the names of its chunk files"
    assert all(shard.get("layout") == "round-robin" for shard in by_target[("set", "small")])


def test_every_artifact_the_routed_questions_of_a_store_name_is_a_key_of_its_chunk_files(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    entries = _dolma_entries(_world_json(directory))
    asked: dict[tuple[str, str], list] = {}
    for question, target in _routed("dolma"):
        asked.setdefault(target, []).append(question)
    assert ("store", "main") in asked
    for target, entry in entries.items():
        if target[0] != "store":
            continue
        wanted = _artifacts_named(asked.get(target, []))
        missing = sorted(wanted - set(entry["chunk_files"]))
        assert not missing, f"{target}: no chunk file for {missing}; it holds {sorted(entry['chunk_files'])}"
    assert {"E/shard-1", "E/shard-2", "E/shard-3"} <= _artifacts_named(asked[("store", "main")])


def test_the_successor_of_shard_3_is_a_chunk_file_inside_the_build_of_main(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    entry = _dolma_entries(world)[("store", "main")]
    assert SUCCESSOR_KEY in entry["chunk_files"], sorted(entry["chunk_files"])
    build = Path(os.path.normpath(entry["build"]))
    relative = entry["chunk_files"][SUCCESSOR_KEY]
    assert not os.path.isabs(relative) and ".." not in Path(relative).parts
    successor = build / relative
    assert successor.is_file()
    historical = build / entry["chunk_files"]["E/shard-3"]
    assert relative != entry["chunk_files"]["E/shard-3"], "the successor is named as the historical file is"
    assert successor.read_bytes() != historical.read_bytes(), "the successor holds the bytes of the historical shard"
    # Withdrawing A leaves no record of A in the repaired shard.
    withdrawn = _ids_of(build / "sources" / "A.jsonl")
    assert withdrawn, "source A holds no record"
    assert not (_ids_of(successor) & withdrawn), "the successor still holds a record of A"
    # No other store names a successor.
    for target, other in _dolma_entries(world).items():
        if target != ("store", "main"):
            assert SUCCESSOR_KEY not in other["chunk_files"], target


def _successor_shard(world: dict) -> dict:
    """The one shard of the world flagged successor."""
    flagged = [shard["name"] for shard in world["shards"] if shard.get("successor")]
    assert flagged == [SUCCESSOR_KEY], f"the shards flagged successor are {flagged}, not the successor of shard 3 alone"
    return next(shard for shard in world["shards"] if shard.get("successor"))


def _documents_in_successor() -> int:
    """The documents the successor of shard 3 holds, from the row of the table that says it."""
    return _core_rows()["c09.fixture.E/shard-3.action"]["expect"]["documents_in_successor"]


def test_the_successor_of_shard_3_is_the_last_shard_of_the_world_with_its_files_inside_the_build_of_main(
        tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    shards = world["shards"]
    successor = _successor_shard(world)
    assert shards[-1] is successor, f"the last shard of the world is {shards[-1]['name']}, not the successor"
    entry = _dolma_entries(world)[("store", "main")]
    assert tuple(successor["target"]) == ("store", "main")
    assert successor["build"] == entry["build"], "the successor is not a shard of the build of main"
    assert successor["chunk_file"] == entry["chunk_files"][SUCCESSOR_KEY], \
        "the chunk file of the successor shard is not the one main names for it"
    # Every other shard keeps its place, so that every number and every step id that follows from the
    # order of the shards stays: main, rr, scratch-500, then the shards of the set; none is flagged.
    # (Changed 17 Nov 2026: the shards of the stores no-capture and scratch-notokens come after rr and before scratch-500's
    # successors' place, and the small set made without capture follows the small set.)
    earlier = (SHARDS_OF["main"][:3] + SHARDS_OF["rr"] + SHARDS_OF["no-capture"] + SHARDS_OF["scratch-500"]
               + SHARDS_OF["scratch-notokens"])
    assert [shard["name"] for shard in shards[:len(earlier)]] == list(earlier), "a shard moved"
    assert all(tuple(shard["target"]) in (("set", "small"), ("set", "no-capture-small"))
               for shard in shards[len(earlier):-1]), \
        "a shard other than those of the sets lies between the shards of the stores and the successor"
    assert not any(shard.get("successor") for shard in shards[:-1])
    # tracked_order names the shards in that order, the successor last.
    order = world["subject_inputs"]["tracked_order"]
    assert order == [shard["name"] for shard in shards], "tracked_order is not the order of shards"
    assert order[-1] == SUCCESSOR_KEY
    # The index and the packed file of the shard are inside the build of main, and they are the
    # repaired shard's, not the historical third shard's.
    build = Path(os.path.normpath(entry["build"]))
    prefix = Path(os.path.normpath(successor["idx_prefix"]))
    assert build in prefix.parents, "the index and the packed file of the successor lie outside the build of main"
    third = next(shard for shard in shards if shard["name"] == "E/shard-3")
    for suffix in (".idx", ".bin"):
        mine, theirs = Path(str(prefix) + suffix), Path(third["idx_prefix"] + suffix)
        assert mine.is_file(), f"the successor has no {suffix} file"
        assert mine != theirs and mine.read_bytes() != theirs.read_bytes(), \
            f"the {suffix} file of the successor is the historical third shard's"
    index = need_attr("baselines.originblame", "read_shard_index")(Path(str(prefix) + ".idx"))
    held = _documents_in_successor()
    assert index.documents == held == _lines_of(build / successor["chunk_file"]), \
        f"the successor holds {index.documents} documents in its index, {_lines_of(build / successor['chunk_file'])} lines " \
        f"in its chunk file, and the table says {held}"


def test_the_runner_plans_the_successor_as_the_fourth_shard_of_main_with_the_documents_the_table_says(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    specs = need_attr("baselines.runner", "originblame_shards")(world)
    assert [spec.name for spec in specs if spec.target == ("store", "main")] == list(SHARDS_OF["main"])
    assert [spec.name for spec in specs if spec.successor] == [SUCCESSOR_KEY]
    successor = specs[-1]
    assert successor.name == SUCCESSOR_KEY and successor.successor is True and successor.target == ("store", "main")
    assert Path(successor.build) == Path(_dolma_entries(world)[("store", "main")]["build"])
    assert len(successor.documents) == _documents_in_successor()
    assert all(document.section and document.file for document in successor.documents), \
        "a document of the successor has no section or no packed file"


def test_the_rows_that_read_across_the_shards_of_main_leave_the_successor_out_and_the_successor_row_finds_it(
        tmp_path_factory, tmp_path):
    """The reads of the shards a row picks (the tool's own selection) over the four shards of main:
    the row of the counts of run E picks the three shards of run E, as the table's keep and rewrite
    lists say, and no row but those that name a successor picks the successor; the routed row of the
    successor of shard 3 picks it, so that it gets a line."""
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    originblame, record = need("baselines.originblame"), need("baselines.record")
    specs = need_attr("baselines.runner", "originblame_shards")(world)
    tool = originblame.OriginBlame(tmp_path / "work", tmp_path / "env", ["a-prefix"], None, specs, {})
    on_main = [question for question, target in _routed("originblame") if target == ("store", "main")]
    assert on_main, "the table routes no row of OriginBlame to the store main"
    picked: dict[str, list[str]] = {}
    for question in on_main:
        given = originblame._given(question)
        if question.listed_row in originblame.SPANS_ROWS:
            continue
        for layout in (record.layouts_of(question) or [None]):
            kind = "set" if layout is not None else originblame._kind_of(given.get("artifact"))
            if kind not in ("shard", "set"):
                continue
            found = tool._select(question, given, layout, kind)
            picked.setdefault(question.listed_row, []).extend(spec.name for _, spec in found)
    for listed, names in picked.items():
        if "successor" not in listed:
            assert SUCCESSOR_KEY not in names, f"{listed} picks the successor among {names}"
    counts = _core_rows()["c09.fixture.shards.counts"]["expect"]
    assert sorted(picked["c09.fixture.shards.counts"]) == sorted(counts["keep"] + counts["rewrite"]), \
        "the row of the counts of run E does not pick exactly the shards of run E the table lists"
    assert picked["c09.fixture.successor.E/shard-3"] == [SUCCESSOR_KEY], \
        "the routed row of the successor of shard 3 finds no shard of that name on main"


def test_the_requests_are_the_words_of_the_routed_rows_with_the_record_files_of_each_build(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    dolma = need("baselines.dolma")
    givens = dolma._listed_givens()
    every = {givens[question.listed_row]["request"] for question in _questions()
             if question.route is not None and "request" in givens[question.listed_row]}
    plain = {givens[question.listed_row]["request"] for question in _questions()
             if question.route is not None and question.subject in ("dolma", "originblame")
             and "request" in givens[question.listed_row]}
    held = set(world["requests"])
    assert plain <= held, f"the words of the routed rows of dolma and originblame that requests lacks: {sorted(plain - held)}"
    assert held <= every, f"requests holds words no routed row carries: {sorted(held - every)}"
    entries = _dolma_entries(world)
    main = Path(entries[("store", "main")]["build"])
    for word, wanted in world["requests"].items():
        found = re.fullmatch(r"withdraw ([A-Z]+)(?:, record (\S+))?", word)
        if found is None or word == "withdraw W":
            assert isinstance(wanted["files"], list), word
            continue
        source = found.group(1)
        assert wanted["files"] == [f"sources/{source}.jsonl"], word
        if found.group(2) is not None:
            assert isinstance(wanted.get("record"), str) and wanted["record"], f"{word} names no record"
            assert wanted["record"] in _ids_of(main / wanted["files"][0]), f"{word}: the record is not in its file"
    if "withdraw W" in world["requests"]:
        assert world["requests"]["withdraw W"]["files"] == ["sources/Z.jsonl"]
        assert world["requests"]["withdraw W"].get("record"), "withdraw W names the one record W"
    # A file a word names is in the build of every question that carries the word.
    for question, target in _routed("dolma"):
        word = givens[question.listed_row].get("request")
        if word is None or target not in entries:
            continue
        for relative in world["requests"][word]["files"]:
            assert (Path(entries[target]["build"]) / relative).is_file(), \
                f"{question.listed_row}: {word} names {relative}, which {target} does not hold"


def test_the_counting_document_and_the_split_pairs_are_what_the_plant_script_counts(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    entry = _dolma_entries(world)[("set", "small")]
    build = Path(entry["build"])
    chunks = build / "pipeline" / "round-robin" / "tokenizer-input"
    sanitizer = build / "pipeline" / "round-robin" / "sanitizer-input"
    counted = _load_plant_script("count.py").count(
        _planted(tmp_path_factory), chunks=chunks, layout="round-robin",
        sanitizer_input=sanitizer if sanitizer.is_dir() else None, require_pair_kinds="auto")
    wanted = json.loads(json.dumps(counted))
    held = json.loads(Path(world["counted"]["small"]).read_text(encoding="utf-8"))
    for part in ("documents", "check3"):
        assert held[part] == wanted[part], f"the counting document differs from the plant script's in {part}"
    splits = [pair for pair in wanted["check3"]["pairs"]["list"] if pair["placement"] == "split"]
    def shape(pairs):
        # What the Dolma subject reads of a pair: its name and the shard and ordinal of each member.
        return [(pair["pair"], [(member["shard"], member["ordinal"]) for member in pair["members"]])
                for pair in pairs]

    assert splits, "the counting document places no pair split"
    # Changed 18 Nov 2026 (old: world["split_pairs"], one list for every size, equal to the small set's): each set entry holds
    # the pairs its own counting document places split.
    assert shape(entry["split_pairs"]) == shape(splits), \
        "split_pairs of the small set are not the pairs the counting document places split"
    for key, other in _dolma_entries(world).items():
        if key[0] != "set" or key[1] not in world["counted"]:
            continue
        own = json.loads(Path(world["counted"][key[1]]).read_text(encoding="utf-8"))
        own_splits = [pair for pair in own["check3"]["pairs"]["list"] if pair["placement"] == "split"]
        assert shape(other["split_pairs"]) == shape(own_splits), f"the split pairs of {key} are not its own counting document's"
    shards = [item["shard"] for item in wanted["documents"]["shards"]]
    assert sorted(entry["chunk_files"]) == sorted(shards), "the chunk files of the set are not its counted shards"
    for shard in shards:
        assert entry["chunk_files"][shard] == f"pipeline/round-robin/tokenizer-input/{shard}.jsonl"
    index = world["index"]["small"]
    assert sorted(index) == sorted(shards)
    for shard, path in index.items():
        assert path.endswith(".idx") and Path(path).is_file(), shard


def test_the_tie_record_holds_every_chunk_file_of_every_dolma_build_with_its_own_line_count(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    tie = need("baselines.tie")
    shown = tie.read_result(Path(world["tie_record"]))
    held = shown["chunk_files"]
    assert held, "the tie record holds no chunk file"
    for target, entry in _dolma_entries(world).items():
        for artifact, relative in entry["chunk_files"].items():
            if artifact == SUCCESSOR_KEY:
                continue
            assert relative in held, f"{target}: the tie record has no result for {artifact} ({relative})"
            result = held[relative]
            for key in ("shown", "documents", "lines", "first_mismatch"):
                assert key in result, (relative, key)
            assert result["lines"] == _lines_of(Path(entry["build"]) / relative), (target, relative)


# --- The runner reads the world --------------------------------------------------------------------


def _read_by_the_runner(directory: Path, tmp_path: Path, capsys):
    """Run the runner's own ``main`` on the world with stand-ins for the tools: its pass stops at
    the first moment a tool would start, after the checks a real pass makes before any tool. Gives
    what main did (None when the pass was reached) and the configuration the pass was given."""
    runner = need("baselines.runner")
    seen = []

    class Reached(Exception):
        """The stand-in pass was reached: the world was read and every check before a tool passed."""

    def stand_in(config, starter):
        seen.append(config)
        questions = [q for q in config.questions if q.subject in tuple(config.subjects)]
        runner.check_putters(questions, config.fault_putter)
        if config.requests is not None and config.builds is not None:
            runner.check_request_files(config.requests, config.builds, questions)
        raise Reached()

    env = tmp_path / "env"
    record = tmp_path / "record"
    env.mkdir()
    record.mkdir()
    program = tmp_path / "program"
    program.write_bytes(b"a stand-in program")
    arguments = [
        "--subjects", "dolma", "originblame", "--step-cap", "600", "--commit", COMMIT,
        "--work", directory / "work", "--run-record", tmp_path / "run-record", "--lock-prefix", "a-prefix",
        "--world", directory / "world.json", "--program-sha256", ANY_SHA256, "--lock-wait-bound", "120",
        "--tool", "dolma", env, record, ANY_SHA256,
        "--tool", "originblame", env, record, ANY_SHA256, program,
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "run_pass", stand_in)
        try:
            code = runner.main([str(word) for word in arguments])
        except Reached:
            code = None
        except SystemExit as leaving:
            code = ("exit", leaving.code)
    return code, seen, capsys.readouterr()


def test_the_runner_reads_the_world_and_passes_every_check_before_a_tool_with_no_stop(tmp_path_factory, tmp_path, capsys):
    directory = _built(tmp_path_factory, SEVERAL)
    code, seen, output = _read_by_the_runner(directory, tmp_path, capsys)
    assert code is None, f"the runner stopped before its pass: {code!r}; {output.err[-1500:]}"
    assert len(seen) == 1
    config = seen[0]
    targets = {("dolma", tuple(target)) for target in DOLMA_TARGETS}
    shardless = {("dolma", tuple(target)) for target in SHARDLESS_TARGETS}
    asked = {(question.subject, tuple(need("baselines.record").attempt_target(question)))
             for question in config.questions}
    # Changed 17 Nov 2026 (old: the pairs asked of Dolma and of OriginBlame are inside the targets of their entries): the rows of the
    # shardless stores swap and import are asked of both too (D-SUBJ-9); they are the only pairs beyond the entries, and both are asked.
    assert {pair for pair in asked if pair[0] == "dolma"} <= targets | shardless
    assert shardless <= asked, sorted(shardless - asked)
    shardless_ob = {("originblame", tuple(target)) for target in SHARDLESS_TARGETS}
    assert {pair for pair in asked if pair[0] == "originblame"} <= {("originblame", t) for t in SHARD_TARGETS} | shardless_ob
    assert shardless_ob <= asked, sorted(shardless_ob - asked)
    # Every question of the five stores and the small set that the table routes is asked.
    routed = {(subject, target) for subject in ("dolma", "originblame") for _, target in _routed(subject)
              if (target in DOLMA_TARGETS if subject == "dolma" else target in SHARD_TARGETS)}
    assert routed <= asked, sorted(routed - asked)
    assert config.requests is not None and config.builds is not None
    # Changed 17 Nov 2026 (old: set-small, store-main, store-rr, store-scratch-500): the builds with shards of OriginBlame
    # now include the stores no-capture and scratch-notokens and the small set made without capture.
    assert sorted(config.builds) == ["set-no-capture-small", "set-small", "store-main", "store-no-capture", "store-rr",
                                     "store-scratch-500", "store-scratch-notokens"]


def test_the_shards_of_originblame_are_planned_by_the_runner_and_count_the_lines_that_are_not_json(
        tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    runner = need("baselines.runner")
    counts: dict = {}
    specs = runner.originblame_shards(world, counts)
    assert [spec.name for spec in specs] == [shard["name"] for shard in world["shards"]]
    assert all(spec.documents for spec in specs), "a shard has no document"
    planted = _planted_unparseable(_planted(tmp_path_factory))
    assert planted, "the plant list holds no line that is not JSON"
    assert counts["set-small"] == planted
    # The same count by the reader of the Dolma subject.
    dolma = need("baselines.dolma")
    entry = _dolma_entries(world)[("set", "small")]
    seen: dict = {}
    dolma._holders_of_ids(Path(entry["build"]), seen)
    assert seen == planted


# --- WORLD.sha256 -----------------------------------------------------------------------------------


def _gate(directory: Path) -> dict[str, str]:
    """The refusal of a world folder whose files are not the ones ``WORLD.sha256`` lists: a path
    listed twice, ``world.json`` not listed, a path that is absolute or leaves the folder, a
    listed file that is not there, or one whose sha256 is not the listed one. It fails with an
    assertion that names the path; it gives the listed hashes, by path, when it accepts."""
    listed = _listed(directory)
    paths = [path for path, _ in listed]
    assert len(paths) == len(set(paths)), "a path is listed twice"
    held = dict(listed)
    assert "world.json" in held, "world.json is not listed"
    for path, digest in listed:
        assert not os.path.isabs(path) and ".." not in Path(path).parts, path
        assert (directory / path).is_file(), f"{path} is listed and is not in the folder"
        assert _sha256_of(directory / path) == digest, f"the hash of {path} does not match"
    return held


def test_the_hash_list_names_every_file_the_world_names_and_every_hash_matches(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    held = _gate(directory)
    for label, path in _named_files(world).items():
        assert path.is_file(), f"{label}: {path} is not a file"
        assert _relative(directory, path) in held, f"{label} is named by world.json and has no line in WORLD.sha256"
    # The tokenizer given on the command line is copied in, byte for byte.
    assert _sha256_of(Path(world["tokenizer"])) == hashlib.sha256(TOKENIZER_BYTES).hexdigest()


def _normalized(path: Path, directory: Path) -> str:
    """The hash of a file in which the path of its own world folder is replaced by a fixed word:
    two worlds in two folders hold the same files, and a file that names its folder is the same
    file."""
    data = path.read_bytes()
    for text in {str(directory), os.path.normpath(str(directory)), str(directory.resolve())}:
        data = data.replace(text.encode("utf-8"), b"<world folder>")
    return hashlib.sha256(data).hexdigest()


def _made_without_capture(path: str) -> bool:
    return "no-capture" in path


def _tie_seeded(directory: Path):
    """The tie record of a world split in two: the entries of the seeded builds (and the run record) as they are, and the names of
    the entries of the builds made without capture (their results differ between worlds)."""
    held = json.loads((directory / "work" / "tie.json").read_text(encoding="utf-8"))
    entries = held["chunk_files"]
    return ({name: value for name, value in entries.items() if "no-capture" not in name}, held.get("run"),
            sorted(name for name in entries if "no-capture" in name))


# The worker counts compared. The world of one worker is the first; every other world of ``WORLDS``
# is held to it. On a machine under 24 GB of memory or under eight cores that is one other world,
# built with two workers after the first; on a big machine it is four workers and one for each core.
# Each count is in the id of the test (for example ``[2-workers]``) and in its failure text.
MORE_WORKERS = [pytest.param(jobs, id=f"{jobs}-workers") for jobs in WORLDS[1:]]


@pytest.mark.parametrize("jobs", MORE_WORKERS)
def test_the_seeded_files_have_the_same_sha_with_one_worker_and_with_several(tmp_path_factory, jobs):
    """Every file the world names, but those of the build made without capture, has the same sha256
    in the world of one worker and in the world of ``jobs`` workers (two on a small machine). The
    files of the build made without capture are held to their own bar in the next tests: their
    shuffles come from the operating system's randomness."""
    one = _built(tmp_path_factory, 1)
    first = _listed(one)
    more = _built(tmp_path_factory, jobs)
    second = _listed(more)
    counts = f"1 and {jobs} workers (the worlds of this machine: {list(WORLDS)})"
    assert [path for path, _ in first] == [path for path, _ in second], \
        f"the worlds of {counts} do not list the same paths"
    differ = []
    for (path, left), (_, right) in zip(first, second):
        if left == right or _made_without_capture(path):
            continue
        if path == "work/tie.json":
            # The tie record mixes the seeded builds' results with those of the unseeded set's chunk files (which differ
            # between worlds): the seeded entries must be equal across worlds, the others are named alike and guarded by
            # the hash list. (Dated extension, 17 Nov 2026; old: the whole file by its sha256.)
            if _tie_seeded(one) != _tie_seeded(more):
                differ.append(path)
            continue
        if _normalized(one / path, one) != _normalized(more / path, more):
            differ.append(path)
    assert not differ, f"with {counts} these seeded files differ: {differ}"
    # The filter above is not empty: the world does name files of a build made without capture.
    paths = {path for path, _ in first if _made_without_capture(path)}
    assert paths, "no path of the world names a build made without capture"


# --- The files of the build made without capture ----------------------------------------------------
#
# The shuffles of that build come from the operating system's randomness, on purpose, so two worlds
# hold its chunk files in another order. What a read of the builder shows (the bar is the lines, not
# the order):
#
# - ``UnseededShuffler.shuffle`` (replica/steps.py:128-129) is ``_fisher_yates`` (replica/steps.py:70-75)
#   over a copy of the items: swaps of two positions only, so it adds, drops and changes no item.
# - The store made without capture is built in the partitioned layout alone (replica/fixture.py:95-96),
#   and the one place a shuffler is made is replica/build.py:566-571. In that layout each shard is
#   shuffled once, on the records it is given and nothing else (replica/steps.py:167-172,
#   ``partitioned_shards``), so the records of a shard do not move to another shard.
# - After the shuffle ``with_inserted_lines`` (replica/build.py:548-562) puts the line the spec writes
#   at a fixed place of the chunk the sanitizer reads, and ``sanitize`` (replica/steps.py:212-230) keeps
#   or skips each record by its own text, keeping the order. The tokenizer input is one row
#   ``{"id", "text"}`` for each surviving record (replica/build.py:625-627), and the id of a record is
#   its own id or ``<file>#L<line>`` of the file it was first read from (replica/records.py:33-35),
#   never its place in the chunk.
#
# So the unseeded shuffle only reorders: the lines over the shards together are the same in every
# world, and the order of the lines in a shard is the one thing that differs.


def _raw_lines(path: Path) -> list[bytes]:
    """The lines of a file as bytes, the line feed that ends the last one not counted."""
    parts = Path(path).read_bytes().split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def _no_capture_shards(directory: Path) -> list[str]:
    """The chunk files of the builds made without capture (the store and the small set), as paths relative to the world
    folder, from their entries in ``world.json``. (Changed 17 Nov 2026: old, the store's alone.)"""
    entries = _dolma_entries(_world_json(directory))
    found = []
    for target in (("store", "no-capture"), ("set", "no-capture-small")):
        entry = entries[target]
        found += [_relative(directory, Path(entry["build"]) / relative) for relative in entry["chunk_files"].values()]
    return sorted(found)


def _build_key(path: str) -> str:
    """The build made without capture a path of the world belongs to: the store or the small set."""
    return "set" if path.startswith("work/sets/no-capture-") or path.startswith("work/counted/no-capture-") else "store"


def _derived_of_an_unseeded_build(path: str) -> bool:
    """A file made FROM the chunk files of a build made without capture, whose bytes follow the unseeded line order: the
    tokenized shard's .idx and .bin and the counting document of the set. Such a file is guarded by its own world's
    WORLD.sha256 and is not compared across worlds. (Dated extension of the bar of D-WORLD-8, 17 Nov 2026. Old: every other
    file of a build made without capture was compared by its sha256 across worlds. Read from the two worlds of the first
    hold, the files that differed beside the chunk files were exactly these: the .idx and .bin of the 3 shards of the store and
    the 20 shards of the set, and counted/no-capture-small.json; every other file of both builds was equal.)"""
    return path.endswith((".idx", ".bin")) or (path.startswith("work/counted/no-capture-") and path.endswith(".json"))


def _held_originals(directory: Path, entries: Mapping, named: Sequence[str]) -> set[str]:
    """The files the chunk files of the builds made without capture were copied from (``copy_of``): each is listed in
    WORLD.sha256 (it is in ``named``) and holds the bytes of its copy within its own world. The paths of those originals."""
    originals = set()
    for target in (("store", "no-capture"), ("set", "no-capture-small")):
        entry = entries[target]
        for shard, original in (entry.get("copy_of") or {}).items():
            path = _relative(directory, Path(entry["build"]) / original)
            assert path in named, f"the file {shard} of {target} was copied from has no line in WORLD.sha256"
            copy = _relative(directory, Path(entry["build"]) / entry["chunk_files"][shard])
            assert (directory / path).read_bytes() == (directory / copy).read_bytes(), \
                f"the file {shard} of {target} was copied from is not the bytes of the copy"
            originals.add(path)
    return originals


def _no_capture_paths(directory: Path) -> tuple[list[str], list[str]]:
    """The paths the world lists that belong to the builds made without capture: the chunk files of their shards, and every
    other path of those builds that is no derived file (compared by sha256 across worlds, as before). The derived files are
    guarded by the hash list only (``_derived_of_an_unseeded_build``)."""
    shards = _no_capture_shards(directory)
    named = [path for path, _ in _listed(directory) if _made_without_capture(path)]
    assert set(shards) <= set(named), "a chunk file of the build made without capture has no line in WORLD.sha256"
    # W9 (18 Nov 2026): the file a chunk file of the set was copied from (copy_of) is listed too, and its bytes are the copy's,
    # so it follows the unseeded line order like the copy: it is held to the copy (the same bytes in its own world) and not
    # compared by sha256 across worlds. (Found by box set J21: the first run of this file with copy_of listed read the original
    # shard-01.jsonl to shard-06.jsonl as "other files" and refused them.)
    originals = _held_originals(directory, _dolma_entries(_world_json(directory)), named)
    named = [path for path in named if path not in originals or path in shards]
    derived = [path for path in named if path not in shards and _derived_of_an_unseeded_build(path)]
    assert any(path.endswith(".idx") for path in derived) and any(path.startswith("work/counted/") for path in derived), \
        "the derived files of the unseeded builds are not listed in WORLD.sha256"
    return shards, [path for path in named if path not in shards and not _derived_of_an_unseeded_build(path)]


def _no_capture_differences(left: Path, right: Path, shards: Sequence[str], others: Sequence[str]) -> list[str]:
    """How the files of the builds made without capture in the folder ``right`` differ from those in
    ``left``: the lines over all the shards of EACH BUILD together are compared as a multiset (the order of the
    lines and the shard a line is in are free; never a union of the two builds, so a line that moved between the builds is a
    difference), and every other file of the store built without capture by its sha256. Gives the differences in words,
    empty where there is none. (Changed 17 Nov 2026, old: one multiset over the store's three chunk files.)"""
    found = []
    for build in ("store", "set"):
        held, other = Counter(), Counter()
        for path in (p for p in shards if _build_key(p) == build):
            held.update(_raw_lines(left / path))
            other.update(_raw_lines(right / path))
        if held != other:
            only_left, only_right = held - other, other - held
            found.append(f"the lines over the shards of the build made without capture ({build}) together differ: "
                         f"{sum(only_left.values())} line(s) only in the "
                         f"first world, {sum(only_right.values())} only in the second; first world only: "
                         f"{[line[:60] for line in list(only_left)[:2]]}, second world only: "
                         f"{[line[:60] for line in list(only_right)[:2]]}")
    for path in others:
        if _normalized(left / path, left) != _normalized(right / path, right):
            found.append(f"{path} differs by its sha256")
    return found


@pytest.mark.parametrize("jobs", MORE_WORKERS)
def test_the_files_of_the_build_made_without_capture_hold_the_same_lines_with_one_worker_and_with_several(
        tmp_path_factory, jobs):
    """The three chunk files of the store made without capture hold, over the shards together, the
    same multiset of lines in the world of one worker and in the world of ``jobs`` workers (two on a
    small machine), whatever the order. The read that makes this the bar: the unseeded shuffle only
    reorders (replica/steps.py:70-75 and :128-129; the one shuffler is made at replica/build.py:566-571
    and the store is built in the partitioned layout alone, replica/fixture.py:95-96; the sanitizer
    and the rows of the tokenizer input depend on a record alone, replica/steps.py:212-230 and
    replica/build.py:625-627). Every other file of that build keeps the bar of the seeded files: the
    same sha256, except the file a chunk file of the set was copied from (``copy_of``, W9, 18 Nov 2026; D-WORLD-8): its bytes are the
    copy's, so it is held to its own copy within its world and is not compared across worlds."""
    one = _built(tmp_path_factory, 1)
    more = _built(tmp_path_factory, jobs)
    counts = f"1 and {jobs} workers (the worlds of this machine: {list(WORLDS)})"
    assert [path for path, _ in _listed(one)] == [path for path, _ in _listed(more)], \
        f"the worlds of {counts} do not list the same paths"
    shards, others = _no_capture_paths(one)
    assert len(shards) == 3 + 20, f"the store made without capture has three shards and the small set made without capture twenty, the world names {shards}"
    assert _no_capture_shards(more) == shards, f"the worlds of {counts} name other chunk files for that store"
    differ = _no_capture_differences(one, more, shards, others)
    assert not differ, f"with {counts} the files of the build made without capture differ: {differ}"


def _flipped(line: bytes) -> bytes:
    """The line with one bit of the byte in its middle changed."""
    middle = len(line) // 2
    return line[:middle] + bytes([line[middle] ^ 1]) + line[middle + 1:]


def _written(folder: Path, path: str, lines: Sequence[bytes]) -> None:
    (folder / path).write_bytes(b"".join(line + b"\n" for line in lines))


def test_the_order_free_compare_accepts_other_orders_and_refuses_one_changed_line(
        tmp_path_factory, tmp_path, capsys):
    """The compare of the test above, put to copies of the files of one world. A copy with every
    line in another order, and one line moved to another shard, is accepted. A copy with ONE changed
    line, one dropped line, or one line replaced by a copy of another is refused, and the words of
    each refusal go to the log."""
    directory = _built(tmp_path_factory, SEVERAL)
    shards, others = _no_capture_paths(directory)
    assert len(shards) == 3 + 20, shards
    named = shards + others

    def copy_of(name: str) -> Path:
        folder = tmp_path / name
        for path in named:
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / path, folder / path)
        return folder

    unchanged = copy_of("unchanged")
    assert _no_capture_differences(directory, unchanged, shards, others) == []

    # The same lines in another order: every shard reversed, and the first line of the first shard
    # moved to the end of the second shard.
    reordered = copy_of("reordered")
    lines = {path: _raw_lines(directory / path) for path in shards}
    assert all(len(found) >= 2 for found in lines.values()), "a shard holds fewer than two lines"
    moved = lines[shards[0]][0]
    new = {path: list(reversed(found)) for path, found in lines.items()}
    new[shards[0]] = list(reversed(lines[shards[0]][1:]))
    new[shards[1]] = new[shards[1]] + [moved]
    for path in shards:
        _written(reordered, path, new[path])
    assert [_raw_lines(reordered / path) for path in shards] != [lines[path] for path in shards], "nothing was reordered"
    assert _no_capture_differences(directory, reordered, shards, others) == []

    refused = {}
    # ONE changed line.
    changed = copy_of("changed")
    found = list(lines[shards[1]])
    found[len(found) // 2] = _flipped(found[len(found) // 2])
    _written(changed, shards[1], found)
    refused["one changed line"] = _no_capture_differences(directory, changed, shards, others)
    # One dropped line.
    dropped = copy_of("dropped")
    _written(dropped, shards[2], lines[shards[2]][1:])
    refused["one dropped line"] = _no_capture_differences(directory, dropped, shards, others)
    # One line replaced by a copy of another one of the same shard: the same count of lines.
    twice = copy_of("twice")
    found = list(lines[shards[0]])
    assert found[0] != found[1], "the first two lines of a shard are equal"
    found[0] = found[1]
    _written(twice, shards[0], found)
    refused["one line replaced by a copy of another"] = _no_capture_differences(directory, twice, shards, others)
    # One line moved from a chunk file of the set to one of the store: each build keeps its own multiset, so a union would
    # accept it and the per-build compare refuses it.
    across = copy_of("across")
    in_set = [path for path in shards if _build_key(path) == "set"][0]
    in_store = [path for path in shards if _build_key(path) == "store"][0]
    set_lines, store_lines = _raw_lines(directory / in_set), _raw_lines(directory / in_store)
    _written(across, in_set, set_lines[1:])
    _written(across, in_store, store_lines + [set_lines[0]])
    refused["one line moved between the two builds"] = _no_capture_differences(directory, across, shards, others)
    # One changed line in a chunk file of the STORE built without capture: its own per-build compare stays shown.
    store_changed = copy_of("store-changed")
    found = list(store_lines)
    found[len(found) // 2] = _flipped(found[len(found) // 2])
    _written(store_changed, in_store, found)
    refused["one changed line in a chunk file of the store"] = _no_capture_differences(
        directory, store_changed, shards, others)
    # A changed byte in any other file of the build made without capture.
    if others:
        altered = copy_of("altered")
        data = (altered / others[0]).read_bytes()
        assert data, f"{others[0]} is empty"
        (altered / others[0]).write_bytes(_flipped(data))
        refused["a changed byte of another file of the build"] = _no_capture_differences(
            directory, altered, shards, others)
    with capsys.disabled():
        for name, differ in refused.items():
            print(f"\n--- the order-free compare on a copy with {name} ---\n{differ}", flush=True)
    for name, differ in refused.items():
        assert differ, f"the compare accepted a copy with {name}"
    assert any("together differ" in word for word in refused["one changed line"]), refused
    assert any("together differ" in word for word in refused["one line replaced by a copy of another"])
    assert any("together differ" in word for word in refused["one line moved between the two builds"])
    assert any("(store)" in word and "together differ" in word for word in refused["one changed line in a chunk file of the store"])


def _copy_of_listed_files(directory: Path, destination: Path) -> Path:
    """A copy of the world folder that holds the files ``WORLD.sha256`` lists and the list."""
    for path in ["WORLD.sha256", *[path for path, _ in _listed(directory)]]:
        (destination / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(directory / path, destination / path)
    return destination


def test_the_hash_list_refuses_a_changed_byte_a_missing_file_and_a_path_listed_twice(
        tmp_path_factory, tmp_path, capsys):
    """The refusal of ``_gate`` (the same one the hash-list test above makes), put to copies of a
    world: the unchanged copy is accepted; one changed byte in a file of the build made without
    capture, in a seeded file, and in ``world.json`` is each refused by name of its path and accepted
    again once the byte is put back; a listed file that is not there, and a path listed twice, are
    refused. The words of each refusal go to the log."""
    directory = _built(tmp_path_factory, SEVERAL)
    copy = _copy_of_listed_files(directory, tmp_path / "copy")
    assert _gate(copy) == _gate(directory), "the copy of the world is not accepted as the world is"
    listed = [path for path, _ in _listed(directory)]
    shards, _ = _no_capture_paths(directory)
    seeded = [path for path in listed if path.endswith(".jsonl") and not _made_without_capture(path)]
    assert seeded, "no seeded chunk or record file is listed"
    planted = {"a file of the build made without capture": shards[0],
               "a seeded file": seeded[0],
               "world.json": "world.json"}
    refusals = {}
    for name, path in planted.items():
        original = (copy / path).read_bytes()
        assert original, f"{path} is empty"
        (copy / path).write_bytes(_flipped(original))
        try:
            with pytest.raises(AssertionError, match=re.escape(path)) as refused:
                _gate(copy)
        finally:
            (copy / path).write_bytes(original)
        refusals[name] = str(refused.value)
        assert _gate(copy) == _gate(directory), f"the copy is not accepted once the byte of {path} is put back"
    # A listed file that is not there.
    gone = seeded[-1]
    moved_away = tmp_path / "moved-away"
    shutil.move(str(copy / gone), moved_away)
    try:
        with pytest.raises(AssertionError, match=re.escape(gone)) as refused:
            _gate(copy)
    finally:
        shutil.move(str(moved_away), copy / gone)
    refusals["a listed file that is not there"] = str(refused.value)
    # A path listed twice.
    listing = (copy / "WORLD.sha256").read_text(encoding="utf-8")
    first_line = listing.splitlines()[0]
    (copy / "WORLD.sha256").write_text(listing + first_line + "\n", encoding="utf-8")
    try:
        with pytest.raises(AssertionError, match="listed twice") as refused:
            _gate(copy)
    finally:
        (copy / "WORLD.sha256").write_text(listing, encoding="utf-8")
    refusals["a path listed twice"] = str(refused.value)
    assert _gate(copy) == _gate(directory)
    with capsys.disabled():
        for name, words in refusals.items():
            print(f"\n--- the gate on a copy with {name} ---\n{words}", flush=True)


# --- subject_inputs ---------------------------------------------------------------------------------


def test_the_subject_inputs_name_and_hash_what_the_four_subjects_the_runner_does_not_drive_read(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    inputs = world["subject_inputs"]
    assert isinstance(inputs, dict)
    for key in ("manifest", "trace"):
        assert key in inputs, key
    held = dict(_listed(directory))
    entries = list(_file_entries(inputs))
    assert entries, "subject_inputs names no file"
    for entry in entries:
        if entry.get("repository") is True:
            assert (REPO / entry["path"]).is_file(), entry["path"]
            assert _sha256_of(REPO / entry["path"]) == entry["sha256"], entry["path"]
            continue
        assert not os.path.isabs(entry["path"]) and ".." not in Path(entry["path"]).parts, entry["path"]
        assert (directory / entry["path"]).is_file(), f"{entry['path']} is named and not in the folder"
        assert entry["path"] in held, f"{entry['path']} has no line in WORLD.sha256"
        assert held[entry["path"]] == entry["sha256"] == _sha256_of(directory / entry["path"]), entry["path"]
    # The frozen files of the repository the subjects read are named by path and hash.
    repository = {entry["path"] for entry in entries if entry.get("repository") is True}
    for path in REPOSITORY_INPUTS:
        assert path in repository, path


def test_the_manifest_inputs_hold_the_built_from_file_of_every_build_that_has_one(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    named = {entry["path"]: entry["sha256"] for entry in _file_entries(world["subject_inputs"]["manifest"])
             if entry.get("repository") is not True}
    seen = 0
    for target, entry in _dolma_entries(world).items():
        file = Path(entry["build"]) / "built-from.jsonl"
        if not file.is_file():
            continue
        seen += 1
        relative = _relative(directory, file)
        assert relative in named, f"the manifest inputs do not name the built-from file of {target}"
        assert named[relative] == _sha256_of(file), target
    assert seen, "no build of the world holds a built-from file"


def test_the_trace_inputs_name_the_exported_documents_or_say_none_with_the_reason(tmp_path_factory):
    directory = _built(tmp_path_factory, SEVERAL)
    world = _world_json(directory)
    trace = world["subject_inputs"]["trace"]
    report = (directory / "world-report.txt").read_text(encoding="utf-8")
    with_runs = set(need_attr("baselines.builds", "RUNS_OF_STORE"))
    for target in DOLMA_TARGETS:
        key = "-".join(target)
        assert key in trace, f"no trace input for {key}"
        value = trace[key]
        has_runs = target[0] == "store" and target[1] in with_runs
        files = list(_file_entries(value))
        if files:
            assert has_runs, f"{key} has no run and names trace documents"
            continue
        reason = value.get("none") if isinstance(value, dict) else None
        assert isinstance(reason, str) and reason.strip(), f"{key} names no trace document and gives no reason"
        if has_runs:
            assert any(key in line and "none" in line.lower() for line in report.splitlines()), \
                f"world-report.txt does not say by name that {key} has no trace"
    main = trace["store-main"]
    if isinstance(main, dict) and list(_file_entries(main)):
        states = main.get("state_traces")
        assert isinstance(states, dict) and set(states) == set(CHECK_2_STATES), states
        for state, value in states.items():
            assert list(_file_entries(value)) or (isinstance(value, dict) and value.get("none")), state


# --- The tie merge rule -----------------------------------------------------------------------------

CHUNK_1 = "pipeline/partitioned/tokenizer-input/shard-1.jsonl"
CHUNK_2 = "pipeline/partitioned/tokenizer-input/shard-2.jsonl"


def _result(chunk, *, shown=True, documents=3, lines=3, first_mismatch=None):
    return {"chunk_file": chunk, "shown": shown, "documents": documents, "lines": lines,
            "first_mismatch": first_mismatch}


def test_results_two_builds_agree_on_merge_into_one_result_for_each_chunk_file(tmp_path):
    merge = need_attr("baselines.world", "merge_tie_results")
    first = [_result(CHUNK_1), _result(CHUNK_2, documents=5, lines=5)]
    second = [_result(CHUNK_2, documents=5, lines=5), _result(CHUNK_1)]
    only_here = _result("pipeline/round-robin/tokenizer-input/shard-01.jsonl", documents=7, lines=7)
    merged = merge({"store-main": first, "store-no-capture": second + [only_here]})
    by_chunk = {result["chunk_file"]: result for result in merged}
    assert len(merged) == len(by_chunk) == 3, "a chunk file two builds name appears twice"
    assert by_chunk[CHUNK_1] == _result(CHUNK_1)
    assert by_chunk[CHUNK_2] == _result(CHUNK_2, documents=5, lines=5)
    assert by_chunk[only_here["chunk_file"]] == only_here
    # Three builds that agree, and the order of the builds, change nothing.
    again = merge({"b": second, "a": first, "c": list(reversed(first))})
    assert sorted(result["chunk_file"] for result in again) == [CHUNK_1, CHUNK_2]
    # What it gives is what the tie record writes and reads back.
    tie = need("baselines.tie")
    path = tmp_path / "tie.json"
    tie.write_result(path, merged, {"tokenizer": "a stand-in line tokenizer"})
    assert sorted(tie.read_result(path)["chunk_files"]) == sorted(by_chunk)


@pytest.mark.parametrize("change", [
    {"shown": False, "first_mismatch": 1},
    {"documents": 4},
    {"lines": 4},
    {"first_mismatch": 0},
])
def test_two_builds_that_differ_on_a_chunk_file_stop_the_merge_naming_both(change):
    merge = need_attr("baselines.world", "merge_tie_results")
    error_type = need_attr("baselines.world", "TieMergeError")
    assert issubclass(error_type, ValueError)
    left = [_result(CHUNK_1), _result(CHUNK_2)]
    right = [dict(_result(CHUNK_1), **change), _result(CHUNK_2)]
    with pytest.raises(error_type) as caught:
        merge({"store-main": left, "store-no-capture": right})
    text = str(caught.value)
    assert "store-main" in text and "store-no-capture" in text, text
    assert CHUNK_1 in text, text


SET_SHARDS = ("shard-01", "shard-02", "shard-03")


def test_each_size_of_a_set_names_its_chunk_files_apart_and_the_smallest_keeps_the_path_of_the_build():
    name = need_attr("baselines.world", "set_chunk_file")
    builds = need("baselines.builds")
    # Every size of a generated set deals its records over the same twenty chunks, so the shard
    # names of two sizes are the same names; the tie record is keyed by the name of the chunk file.
    names = {size: [name(size, shard) for shard in SET_SHARDS] for size in builds.SET_SIZES}
    smallest = builds.SET_SIZES[0]
    assert names[smallest] == [f"pipeline/round-robin/tokenizer-input/{shard}.jsonl" for shard in SET_SHARDS]
    held = [chunk for chunks in names.values() for chunk in chunks]
    assert len(set(held)) == len(held) == len(builds.SET_SIZES) * len(SET_SHARDS), \
        "two sizes of a set name one chunk file"
    for size, chunks in names.items():
        for shard, chunk in zip(SET_SHARDS, chunks):
            assert not os.path.isabs(chunk) and ".." not in Path(chunk).parts, chunk
            assert chunk.endswith(f"/tokenizer-input/{shard}.jsonl"), chunk
            if size != smallest:
                assert size in chunk, f"the chunk file {chunk} of the size {size} does not say its size"


def test_the_tie_results_of_the_sizes_of_a_set_merge_because_their_chunk_files_are_named_apart():
    merge = need_attr("baselines.world", "merge_tie_results")
    name = need_attr("baselines.world", "set_chunk_file")
    builds = need("baselines.builds")
    # The same shard name in three sizes holds a different number of documents in each.
    results = {f"set-{size}": [_result(name(size, "shard-01"), documents=documents, lines=documents)]
               for size, documents in zip(builds.SET_SIZES, (40, 500, 5000))}
    merged = merge(results)
    assert sorted(result["documents"] for result in merged) == [40, 500, 5000], \
        "the results of the sizes of a set were not kept apart"


def test_a_size_that_is_not_the_smallest_gets_copies_of_the_chunk_files_of_its_build(tmp_path):
    name_files = need_attr("baselines.world", "name_set_chunk_files")
    builds = need("baselines.builds")
    folder = tmp_path / "build" / "pipeline" / "round-robin" / "tokenizer-input"
    folder.mkdir(parents=True)
    for shard in SET_SHARDS:
        (folder / f"{shard}.jsonl").write_bytes(f'{{"id": "{shard}", "text": "a line of {shard}"}}\n'.encode("utf-8"))
    build = tmp_path / "build"
    before = sorted(path.relative_to(build).as_posix() for path in build.rglob("*") if path.is_file())
    own = name_files(builds.SET_SIZES[0], build)
    assert sorted(own) == sorted(SET_SHARDS)
    assert sorted(path.relative_to(build).as_posix() for path in build.rglob("*") if path.is_file()) == before, \
        "the smallest size copied a chunk file"
    for size in builds.SET_SIZES[1:]:
        named = name_files(size, build)
        assert sorted(named) == sorted(SET_SHARDS)
        for shard, relative in named.items():
            assert relative not in own.values()
            assert (build / relative).read_bytes() == (folder / f"{shard}.jsonl").read_bytes(), \
                f"the copy of {shard} for the size {size} is not the chunk file"
    assert sorted(path.name for path in folder.iterdir()) == [f"{shard}.jsonl" for shard in SET_SHARDS], \
        "a copy was written into the folder of the build's own chunk files"


def test_an_original_of_an_unseeded_copy_must_be_listed_and_hold_the_bytes_of_its_copy(tmp_path):
    """The bar extension of D-WORLD-8 (W9, granted 18 Nov 2026): the exemption covers only the originals ``copy_of`` names, and
    each must be listed and equal its copy's bytes in its own world. A fabricated world of one copy: it holds, and an original
    with one changed byte, and an original with no line in the hash list, are each refused."""
    build = tmp_path / "work" / "sets" / "no-capture-small" / "no-capture" / "build"
    original, copy = "pipeline/round-robin/tokenizer-input/shard-01.jsonl", "set-no-capture-small/tokenizer-input/shard-01.jsonl"
    for relative in (original, copy):
        (build / relative).parent.mkdir(parents=True, exist_ok=True)
        (build / relative).write_bytes(b'{"id": "a", "text": "one line"}\n')
    entry = {"build": str(build), "chunk_files": {"shard-01": copy}, "copy_of": {"shard-01": original}}
    entries = {("store", "no-capture"): {"build": str(build), "chunk_files": {}}, ("set", "no-capture-small"): entry}
    named = [str((build / original).relative_to(tmp_path)), str((build / copy).relative_to(tmp_path))]
    assert _held_originals(tmp_path, entries, named) == {named[0]}
    (build / original).write_bytes(b'{"id": "a", "text": "one lime"}\n')
    with pytest.raises(AssertionError, match="is not the bytes of the copy"):
        _held_originals(tmp_path, entries, named)
    (build / original).write_bytes(b'{"id": "a", "text": "one line"}\n')
    with pytest.raises(AssertionError, match="has no line in WORLD.sha256"):
        _held_originals(tmp_path, entries, named[1:])


def _layout_result(layout, shards, build):
    return {"layout": layout, "build": build, "store": build + ".sqlite", "counted": f"/w/counted/10k-{layout}.json",
            "chunk_files": {name: f"pipeline/{layout}/tokenizer-input/{name}.jsonl" for name in shards},
            "copy_of": {name: f"pipeline/{layout}/tokenizer-input/{name}.jsonl" for name in shards},
            "prefixes": {name: f"pipeline/{layout}/shards/{name}" for name in shards}, "tie": [{"file": layout}],
            "repaired": {"layout": layout, "out": f"{build}-repaired", "store": f"{build}-repair.sqlite", "prefix": "set/"}}


def test_the_builds_added_beside_a_sets_own_are_entries_shards_documents_and_ties_of_their_own_layout():
    """W1 (18 Nov 2026): the pure part of the world assembly for the 10k set in the partitioned layout: a Dolma entry that names
    its layout (the set's own entry is untouched: ruling A), the OriginBlame shards, ``counted_by_layout`` with both layouts'
    documents, the tie and build labels; and a refusal where two sizes would each hold two builds."""
    parts = need_attr("baselines.world", "_layout_parts")
    world = need("baselines.world")
    own = {"counted": "/w/counted/10k.json", "repaired": {"layout": "round-robin", "out": "/w/b-own-repaired", "store": "/w/s", "prefix": "set-rr/"},
           "layouts": {"partitioned": _layout_result("partitioned", ["shard-A", "shard-B"], "/w/b-part")}}
    small = {"counted": "/w/counted/small.json", "layouts": {}}
    got = parts(["small", "10k"], {"small": small, "10k": own})
    assert got["entries"] == [{"target": ["set", "10k"], "layout": "partitioned", "build": "/w/b-part",
                               "chunk_files": {"shard-A": "pipeline/partitioned/tokenizer-input/shard-A.jsonl",
                                               "shard-B": "pipeline/partitioned/tokenizer-input/shard-B.jsonl"},
                               "copy_of": {"shard-A": "pipeline/partitioned/tokenizer-input/shard-A.jsonl",
                                           "shard-B": "pipeline/partitioned/tokenizer-input/shard-B.jsonl"}}]
    assert [(s["name"], s["target"], s["layout"]) for s in got["shards"]] == [("shard-A", ["set", "10k"], "partitioned"), ("shard-B", ["set", "10k"], "partitioned")]
    assert got["shards"][0]["idx_prefix"] == "/w/b-part/pipeline/partitioned/shards/shard-A"
    assert got["counted_by_layout"] == {"round-robin": "/w/counted/10k.json", "partitioned": "/w/counted/10k-partitioned.json"}
    assert got["ties"] == {"set-10k-partitioned": [{"file": "partitioned"}]} and got["builds"] == {"set-10k-partitioned": "/w/b-part"}
    assert got["repaired"] == {"set-10k": "/w/b-own-repaired", "set-10k-partitioned": "/w/b-part-repaired"}, "W2: one repaired side per build"
    nothing = parts(["small"], {"small": small})
    assert nothing["entries"] == [] and nothing["shards"] == [] and nothing["counted_by_layout"] == {}
    assert nothing["repaired"] == {}
    twice = {"counted": "/w/counted/small.json", "layouts": {"partitioned": _layout_result("partitioned", ["shard-A"], "/w/b2")}}
    with pytest.raises(world.WorldError, match="keyed by the layout alone"):
        parts(["small", "10k"], {"small": twice, "10k": own})


def _repaired_with_successors(folder, names, layout="round-robin"):
    """A repair summary of the form ``replica repair`` prints: the successor of each rewritten shard under its registered name, and the
    files of the repaired build by registered name (a shard is a name whose ``.input`` is registered beside it), written to ``folder``."""
    suffix = need_attr("replica.shard_files", "SUFFIX_TOKENIZER_INPUT")
    files = {}
    for name in names:
        registered = f"successor of set/{name}"
        chunk, prefix = f"pipeline/{layout}/tokenizer-input/{name}.jsonl", f"pipeline/{layout}/shards/{name}"
        files[registered + suffix], files[registered] = chunk, prefix
        files[registered + ".sanitizer-input"] = f"pipeline/{layout}/sanitizer-input/{name}.jsonl"   # as the replica prints it: no shard
        (folder / chunk).parent.mkdir(parents=True, exist_ok=True)
        (folder / chunk).write_text(f"{name} chunk\n", encoding="utf-8")
        (folder / prefix).parent.mkdir(parents=True, exist_ok=True)
        (folder / f"{prefix}.idx").write_bytes(f"{name} idx".encode())
        (folder / f"{prefix}.bin").write_bytes(f"{name} bin".encode())
    files["successor of R1"] = "releases/R1.jsonl"          # a release is a successor too, and no shard
    # A shard the repair kept is registered under the successor name too (the same instance): it is no successor shard.
    files["successor of set/shard-kept" + suffix], files["successor of set/shard-kept"] = "pipeline/x/tokenizer-input/kept.jsonl", "pipeline/x/shards/kept"
    return {"out": str(folder), "files": files,
            "successors": {**{f"set/{name}": f"successor of set/{name}" for name in names}, "R1": "successor of R1"},
            "kept": {"set/shard-kept": {"registered_as": "successor of set/shard-kept", "same_instance": True}}}


def test_the_successor_shards_of_a_repair_are_the_rewritten_shards_with_their_chunk_file_and_shard_prefix(tmp_path):
    """The repair of a set prints the successor of every artifact it rewrote (``successors``) and the files of the repaired build; the
    successor SHARDS are the successors that are shards (a ``.input`` registered beside the name): not a release, not a shard the
    repair kept (a kept shard has no successor)."""
    of = need_attr("baselines.world", "_successor_shards_of")
    suffix = need_attr("replica.shard_files", "SUFFIX_TOKENIZER_INPUT")
    repaired = _repaired_with_successors(tmp_path, ["shard-01", "shard-02"])
    assert of(repaired) == {
        "successor of set/shard-01": ("pipeline/round-robin/tokenizer-input/shard-01.jsonl", "pipeline/round-robin/shards/shard-01"),
        "successor of set/shard-02": ("pipeline/round-robin/tokenizer-input/shard-02.jsonl", "pipeline/round-robin/shards/shard-02")}
    assert of({"successors": {}, "files": {}}) == {}
    # A landing shard (no artifact of the historical build is its) is registered under the successor name and is no kept shard.
    landing = _repaired_with_successors(tmp_path, ["shard-B"])
    assert list(of(landing)) == ["successor of set/shard-B"]
    assert suffix  # the tokenizer-input suffix is the replica's own


def test_the_successor_shards_of_a_repaired_set_are_copied_into_the_build_beside_the_historical_files(tmp_path, monkeypatch):
    """The route of D-WORLD-9 for the main store, for a set: the chunk file and the two shard files of each successor shard are COPIED
    into the folder ``successor`` of the historical build (the repaired folder is not touched, so the files the repair-equality gate
    compares stay as they are), under a name that holds the layout (no name is shared by two builds of the world). No tie show is made
    (the tie record is read for the chunk files of a Dolma run, and these are read by OriginBlame alone)."""
    world_mod = need("baselines.world")
    folder, build = tmp_path / "repaired", tmp_path / "build"
    folder.mkdir(), build.mkdir()
    repaired = _repaired_with_successors(folder, ["shard-01", "shard-02"])
    before = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
    monkeypatch.setattr(world_mod, "_tie_results", lambda *a, **k: pytest.fail("a tie show of the successor shards of a set is made"))
    got = world_mod._place_successors(repaired, build, "round-robin", "set-10k")
    assert got["shards"] == {
        "successor of set/shard-01": {"chunk_file": "successor/tokenizer-input/round-robin-shard-01.jsonl",
                                      "prefix": "successor/shards/round-robin-shard-01"},
        "successor of set/shard-02": {"chunk_file": "successor/tokenizer-input/round-robin-shard-02.jsonl",
                                      "prefix": "successor/shards/round-robin-shard-02"}}
    assert (build / "successor/tokenizer-input/round-robin-shard-01.jsonl").read_text(encoding="utf-8") == "shard-01 chunk\n"
    assert (build / "successor/shards/round-robin-shard-02.idx").read_bytes() == b"shard-02 idx"
    assert (build / "successor/shards/round-robin-shard-02.bin").read_bytes() == b"shard-02 bin"
    assert sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()) == before, "the repaired folder is untouched"
    with pytest.raises(world_mod.WorldError, match="registered no successor shard"):
        world_mod._place_successors({"out": str(folder), "files": {}, "successors": {}}, build, "round-robin", "set-10k")


def test_the_successor_shards_of_the_10k_set_are_shard_entries_of_both_layouts_that_come_last(tmp_path):
    """The shard entries of OriginBlame for the successors of the 10k set, one for each successor shard of each build (the set's own
    round-robin build and the one added beside it), ``successor`` true, each in the layout of its build: the pure part of the world
    assembly puts them under their own key, which the assembly appends after every other shard so that no shard number or step id moves."""
    parts = need_attr("baselines.world", "_layout_parts")
    added = _layout_result("partitioned", ["shard-A", "shard-gen"], "/w/b-part")
    added["repaired"]["successor_shards"] = {"successor of set/shard-gen": {"chunk_file": "successor/tokenizer-input/partitioned-shard-gen.jsonl",
                                                                          "prefix": "successor/shards/partitioned-shard-gen"}}
    own = {"counted": "/w/counted/10k.json", "build": "/w/b-own",
           "repaired": {"layout": "round-robin", "out": "/w/b-own-repaired", "store": "/w/s", "prefix": "set-rr/",
                        "successor_shards": {"successor of set/shard-01": {"chunk_file": "successor/tokenizer-input/round-robin-shard-01.jsonl",
                                                                         "prefix": "successor/shards/round-robin-shard-01"}}},
           "layouts": {"partitioned": added}}
    got = parts(["small", "10k"], {"small": {"counted": "/w/counted/small.json", "layouts": {}}, "10k": own})
    assert got["successor_shards"] == [
        {"name": "successor of set/shard-01", "target": ["set", "10k"], "layout": "round-robin", "run": None, "successor": True,
         "build": "/w/b-own", "chunk_file": "successor/tokenizer-input/round-robin-shard-01.jsonl",
         "idx_prefix": "/w/b-own/successor/shards/round-robin-shard-01"},
        {"name": "successor of set/shard-gen", "target": ["set", "10k"], "layout": "partitioned", "run": None, "successor": True,
         "build": "/w/b-part", "chunk_file": "successor/tokenizer-input/partitioned-shard-gen.jsonl",
         "idx_prefix": "/w/b-part/successor/shards/partitioned-shard-gen"}]
    assert [s["successor"] for s in got["shards"]] == [False, False], "the successors are not among the shards of the builds themselves"
    assert parts(["small"], {"small": {"counted": "/w/counted/small.json", "layouts": {}}})["successor_shards"] == []


def test_the_repair_of_a_set_keeps_the_files_and_the_kept_artifacts_the_replica_printed(tmp_path, monkeypatch):
    """``_set_repair`` returns, besides the repaired folder, the files of the repaired build and the kept artifacts the replica printed
    (the keys the successor shards are read from; the equality gate reads the folder only)."""
    world_mod = need("baselines.world")
    printed = {"successors": {"set/shard-01": "successor of set/shard-01"}, "repaired": {"files": {"successor of set/shard-01": "p"}},
               "kept": {"set/shard-C": {"registered_as": "successor of set/shard-C", "same_instance": True}}}
    repair = SimpleNamespace(
        shard_prefix=lambda shards: "set/", write_placement=lambda *a: tmp_path / "placement.json",
        tool_command=lambda *a, **k: (0, "{}", ""), replica_command=lambda *a, **k: printed)
    sets = SimpleNamespace(layout_shards=lambda instances, layout, folder: ["set/shard-01"])
    runs = SimpleNamespace(snapshot_store=lambda store, copy: Path(copy))
    monkeypatch.setattr(world_mod, "_acceptance", lambda name: {"acceptance_repair": repair, "acceptance_sets": sets, "acceptance_runs": runs}[name])
    out = tmp_path / "out"
    (out).mkdir()
    (tmp_path / "work").mkdir()

    calls = []

    def replica(arguments, timeout=None):
        calls.append([str(a) for a in arguments])
        (tmp_path / "work" / "repair" / "set-10k-round-robin" / "repaired").mkdir(parents=True, exist_ok=True)
        (tmp_path / "work" / "repair" / "set-10k-round-robin" / "repaired" / "built-from.jsonl").write_text("", encoding="utf-8")
        return printed

    repair.replica_command = replica
    got = world_mod._set_repair("10k", "round-robin", tmp_path / "set", out, tmp_path / "store.sqlite", {"instances": {}}, tmp_path / "work", 1, 10.0,
                                world_mod._Log("t"))
    assert got["files"] == {"successor of set/shard-01": "p"} and got["kept"] == printed["kept"]
    # The command the repair-equality gate also runs is the same: the flags and their order are the ones of D-WORLD-13 (no reference side).
    assert [a for a in calls[0] if a.startswith("--")] == ["--report", "--historical", "--out", "--store", "--set", "--layout", "--jobs",
                                                          "--tokenizer-spec"], calls[0]
    assert got["out"].endswith("repaired") and got["layout"] == "round-robin"


def test_the_repaired_side_of_each_10k_build_is_named_in_the_world_as_the_driver_reads_it(tmp_path, monkeypatch):
    """W2 (18 Nov 2026): the names the world writes into ``subject_inputs.manifest`` for the repaired side of the set's own build and of
    the build added beside it, loaded by the driver's own loader (``subjects.load_subject_inputs``), come out as the keys the manifest
    driver reads (``set-10k``, ``set-10k-partitioned``), each with the rows of ITS repaired folder and not the other's; the repaired
    side of the store main keeps its key. Through the world's own pure assembly of the layouts and of the subject inputs."""
    world_mod, subjects = need("baselines.world"), need("baselines.subjects")
    monkeypatch.setattr(world_mod, "_megatron_files", lambda: {})
    directory = tmp_path / "world"

    def folder(name, row):
        path = directory / name
        path.mkdir(parents=True)
        (path / "built-from.jsonl").write_text(json.dumps({"name": row, "files": [], "built_from": []}) + "\n", encoding="utf-8")
        return path

    own_build, added_build, main_build = folder("b-own", "own"), folder("b-part", "added"), folder("b-main", "main")
    result = _layout_result("partitioned", ["shard-A"], str(added_build))
    result["repaired"]["out"] = str(folder("r-part", "successor of src/A (partitioned)"))
    own = {"counted": "/w/counted/10k.json", "repaired": {"out": str(folder("r-own", "successor of src/A (round-robin)"))},
           "layouts": {"partitioned": result}}
    added = world_mod._layout_parts(["10k"], {"10k": own})
    stores = {name: {"runs": None, "repaired": None} for name in world_mod.STORE_ORDER}
    stores["main"]["repaired"] = str(folder("r-main", "successor of main"))
    inputs = world_mod._subject_inputs(directory, {"shards": [], "counted_by_layout": added["counted_by_layout"]}, stores, {"10k": own}, ["10k"],
                                       {"set-10k": own_build, "set-10k-partitioned": added_build, "store-main": main_build},
                                       repaired_sets=added["repaired"])
    assert {key for key in inputs["manifest"] if key.startswith("repaired-")} == {"repaired-main", "repaired-set-10k", "repaired-set-10k-partitioned"}
    loaded = subjects.load_subject_inputs({"subject_inputs": inputs}, directory)
    assert {key: [row["name"] for row in rows] for key, rows in loaded["repaired"].items()} == {
        "store-main": ["successor of main"], "set-10k": ["successor of src/A (round-robin)"],
        "set-10k-partitioned": ["successor of src/A (partitioned)"]}


def test_a_chunk_file_names_the_file_it_was_copied_from_and_the_smallest_size_names_itself(tmp_path):
    """W9 (18 Nov 2026): the source of a copy is written when the copy is made, never found afterwards by equal bytes. Two shards
    hold the SAME bytes, so a reader that derived the source from the bytes could not tell which one a copy came from."""
    name_from = need_attr("baselines.world", "name_set_chunk_files_from")
    builds = need("baselines.builds")
    folder = tmp_path / "build" / "pipeline" / "round-robin" / "tokenizer-input"
    folder.mkdir(parents=True)
    for shard in SET_SHARDS:
        (folder / f"{shard}.jsonl").write_bytes(b'{"id": "same", "text": "the same line in every shard"}\n')
    build = tmp_path / "build"
    named, copy_of = name_from(builds.SET_SIZES[0], build)
    assert copy_of == named, "the smallest size is not a copy: its field names its own path"
    for size in builds.SET_SIZES[1:]:
        named, copy_of = name_from(size, build)
        assert sorted(copy_of) == sorted(SET_SHARDS)
        for shard in SET_SHARDS:
            assert copy_of[shard] == f"pipeline/round-robin/tokenizer-input/{shard}.jsonl", (size, shard)
            assert named[shard] != copy_of[shard]


def test_the_chunk_files_of_a_tie_show_are_spread_over_as_many_tokenizers_as_there_are_workers():
    spread = need_attr("baselines.world", "spread")
    weights = [50, 40, 40, 30, 20, 10, 10, 5]
    everything = list(range(len(weights)))
    one = spread(weights, 1)
    assert len(one) == 1 and sorted(one[0]) == everything, "one worker does not take every chunk file"
    four = spread(weights, 4)
    assert len(four) == 4 and sorted(index for group in four for index in group) == everything
    loads = [sum(weights[index] for index in group) for group in four]
    assert max(loads) - min(loads) <= max(weights), f"the groups are not balanced: {loads}"
    many = spread(weights, 16)
    assert len(many) == len(weights) and all(len(group) == 1 for group in many), \
        "more workers than chunk files do not give one group for each chunk file"
    assert spread([], 4) == []


# --- world.py and the documents --------------------------------------------------------------------


def test_the_module_holds_no_absolute_path_and_no_sha256_of_a_file():
    need("baselines.world")
    path = REPO / "baselines" / "world.py"
    assert path.is_file(), "baselines/world.py does not exist"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    inside_format = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
    strings = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inside_format]
    absolute = [value for value in strings if re.fullmatch(r"/[^\s]+", value)]
    assert absolute == [], f"the module holds an absolute path: {absolute}"
    assert re.findall(r"\b[0-9a-f]{64}\b", text) == [], "the module holds a sha256 literal"
    assert _banned_in(text) == [], _banned_in(text)
    # The words of how the module was built stand in no comment and no docstring.
    prose = [token.string for token in tokenize.generate_tokens(io.StringIO(text).readline)
             if token.type == tokenize.COMMENT]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            prose.append(ast.get_docstring(node) or "")
    found = [piece for piece in prose if PROCESS_WORDS.search(piece)]
    assert found == [], f"a comment or docstring of the module names how it was built: {found[:3]}"


def test_the_document_of_the_baselines_has_a_section_on_the_world_builder():
    text = (REPO / "docs" / "baselines.md").read_text(encoding="utf-8")
    heading = "The world builder (temporary section WORLD)"
    assert re.search(rf"(?m)^#+ {re.escape(heading)}$", text), "docs/baselines.md has no section of that name"
    section = re.split(rf"(?m)^#+ {re.escape(heading)}$", text)[1]
    section = re.split(r"(?m)^## ", section)[0]
    for word in ("python -m baselines.world", "--out", "--megatron-env", "--tokenizer", "--jobs", "--sizes",
                 "world.json", "WORLD.sha256", "world-report.txt", "subject_inputs", "work"):
        assert word in section, word
    assert _banned_in(section) == [], _banned_in(section)


def test_the_world_builder_takes_the_extensions_and_hands_them_to_every_scan_of_the_table_rows():
    """``--extensions`` reaches the routed-row scan (the successors, the stopped targets, the request
    words) and is recorded in world.json; with none, no row of an extension is scanned."""
    world = need("baselines.world")
    assert world._parser().parse_args(["--out", "o", "--megatron-env", "m", "--tokenizer", "t"]).extensions == []
    named = world._parser().parse_args(["--out", "o", "--megatron-env", "m", "--tokenizer", "t",
                                        "--extensions", "generating-model"])
    assert named.extensions == ["generating-model"]
    off = world._routed_givens()
    on = world._routed_givens(("generating-model",))
    assert {q.id for q, _ in off} < {q.id for q, _ in on}, "the rows of the extension that is on are not scanned"
    assert all(q.id in {x.id for x, _ in on} for q, _ in off)
    left = {q.id for q, _ in off}
    extra = [q for q, _ in on if q.id not in left]
    assert len(extra) == 6 and {q.group for q in extra} == {"c05.withdraw-C-on", "c07.R3-release"}, [q.id for q in extra]


def test_the_world_builder_refuses_an_extension_it_has_no_build_for(tmp_path, capsys):
    """world.json records the extensions the world was built for, so a name there must be one the builder builds: the rows of an
    extension that needs a build of its own are not served by the core builds, and a flag is no build. The builder names the
    extensions it has a build for (``BUILT_EXTENSIONS``); any other name is refused before anything is built."""
    world = need("baselines.world")
    assert tuple(world.BUILT_EXTENSIONS) == ("generating-model",)
    code = world.main(["--out", str(tmp_path / "w"), "--megatron-env", str(tmp_path), "--tokenizer", str(tmp_path / "t.json"),
                       "--extensions", "no-such-extension"])
    assert code == 2
    assert "no-such-extension" in capsys.readouterr().err
    assert not (tmp_path / "w").exists(), "nothing is built"


def test_the_repair_of_the_main_store_is_made_on_a_plain_build_of_main_when_the_extension_is_on(tmp_path):
    """Found by the box row of this unit (18 Nov 2026): ``replica repair`` knows only the releases of the spec's ``[releases.*]``
    tables, and the report of a store that holds R3 (an ``[extension_releases.R3]``) names R3, so the repair of the keyed store
    died with "no release named ['R3']". The repaired side of main is the one of the world without the extension: with the
    extension on, the copy the repair works on is a second, plain build of main made in the folder of the copy (the build of a
    seeded store is the same bytes), and with no extension, or for any other store, the copy is a copy of the store's folder
    (None)."""
    world, builds = need("baselines.world"), need("baselines.builds")
    call = world._repair_copy_call("main", tmp_path / "copy", 2, ("generating-model",))
    assert call == builds.store_call("main", tmp_path / "copy", parallel=2)
    assert call.kwargs["store"] == "main"
    assert world._repair_copy_call("main", tmp_path / "copy", 2, ()) is None
    assert world._repair_copy_call("rr", tmp_path / "copy", 2, ("generating-model",)) is None


def _world_with_the_extension(factory) -> SimpleNamespace:
    """One world built with the generating-model extension on, with ``SEVERAL`` workers (four, the record's count; built after
    the others, so it adds one build and no overlap), the small set (the world of the default test, plus ``--extensions``),
    kept for the tests that read it. It is compared with the world of the same worker count. A build over the limit is stopped by its process group id, as the others."""
    if "extension-world" not in _STATE:
        root = os.environ.get(MEGATRON_VARIABLE, "")
        if not root:
            pytest.skip(f"needs the Megatron environment: {MEGATRON_VARIABLE} is not set")
        base = factory.mktemp("extension-world")
        inputs = factory.mktemp("extension-world-inputs")
        tokenizer = inputs / TOKENIZER_NAME
        tokenizer.write_bytes(TOKENIZER_BYTES)
        directory = base / f"jobs-{SEVERAL}"
        log = inputs / "extension-world.log"
        with open(log, "wb") as handle:
            process = subprocess.Popen([*_world_argv(directory, root, tokenizer, SEVERAL), "--extensions", "generating-model"], cwd=REPO,
                                       env=_environment(), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=BUILD_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = "over the limit of the test"
        _STATE["extension-world"] = SimpleNamespace(code=code, directory=directory,
                                                    text=log.read_text(encoding="utf-8", errors="replace")[-3000:])
    return _STATE["extension-world"]


MAIN_BUILT_FROM = "work/stores/main/build/built-from.jsonl"
R3_ROW_NAMES = ["gen/R3-models", "R3"]
CHECKPOINTS = ("E@6", "K@8")   # names of checkpoint rows in the tests below; a row is a checkpoint by its file, not its name


def _rows_named(names: list[str]) -> list[str]:
    """Rows of a manifest: a checkpoint (docs/replica.md section 16, "The checkpoints of a run") names one path, to the run
    record of its segment; every other row here names a file of the build."""
    return [json.dumps({"name": name, "built_from": [],
                        "files": [f"../runs/{name[0]}/seg-1/{name[0]}-seg-1.run.json"] if name in CHECKPOINTS
                        else [f"sources/{name.replace('/', '-')}.jsonl"]}) for name in names]


def test_the_rows_of_r3_sit_before_the_rows_of_the_checkpoints_and_the_other_rows_are_the_plain_rows():
    """docs/replica.md section 16: a build writes the rows of its artifacts as it makes them (the two rows of R3 are written
    inside the build, in its R3 step, ``replica/capture_hooks.build_extension_release``), and the runs, made after the build,
    append the rows of their checkpoints (each ``files`` is one path to a run record). So the manifest of main in a world with the
    extension on is the main build's rows, the two rows of R3, then the rows of the checkpoints. Read against the plain manifest,
    the rows other than R3's are the plain rows in the same order, and R3's two rows are there once each, ``gen/R3-models``
    first, before the first checkpoint row, whatever the checkpoint is called."""
    plain = ["A", "B", "E@6", "K@8"]
    assert _built_from_problem(_rows_named(plain), _rows_named(["A", "B", *R3_ROW_NAMES, "E@6", "K@8"])) is None
    # R3's rows after any checkpoint (the first one is not called K@ here, as on the box), or after all of them
    assert _built_from_problem(_rows_named(plain), _rows_named(["A", "B", "E@6", *R3_ROW_NAMES, "K@8"])) is not None
    assert _built_from_problem(_rows_named(plain), _rows_named([*plain, *R3_ROW_NAMES])) is not None
    # a plain row moved, a plain row missing, an R3 row missing, an R3 row twice
    assert _built_from_problem(_rows_named(plain), _rows_named(["B", "A", *R3_ROW_NAMES, "E@6", "K@8"])) is not None
    assert _built_from_problem(_rows_named(plain), _rows_named(["A", *R3_ROW_NAMES, "E@6", "K@8"])) is not None
    assert _built_from_problem(_rows_named(plain), _rows_named([*plain, "R3"])) is not None
    assert _built_from_problem(_rows_named(plain), _rows_named(["A", "B", *R3_ROW_NAMES, "R3", "E@6", "K@8"])) is not None


def _built_from_problem(was: list[str], now: list[str]) -> str | None:
    """None when ``now`` is ``was`` with the rows ``gen/R3-models`` and ``R3`` added once each in that order, before the first
    row of a checkpoint (a row whose one file is a run record), else what differs."""
    rows = [json.loads(line) for line in now]
    names = [row["name"] for row in rows]
    added = [name for name in names if name in R3_ROW_NAMES]
    if added != R3_ROW_NAMES:
        return f"rows of R3: {added}"
    if [line for line, row in zip(now, rows) if row["name"] not in R3_ROW_NAMES] != was:
        return "the plain rows moved, or one is missing"
    first = next((i for i, row in enumerate(rows) if len(row["files"]) == 1 and row["files"][0].endswith(".run.json")), len(rows))
    if names.index("R3") > first:
        return "the rows of R3 come after the first row of a checkpoint"
    return None


def test_a_world_built_with_the_generating_model_extension_holds_the_core_files_and_the_files_of_r3(tmp_path_factory):
    """The main store of that world is the replica's build ``main+generating-model``: the main build, then R3. Against the world
    without the extension (the same worker count and set: the world the other tests already built), by the bar of the test of the seeded files (the folder of the world read as
    ``<world folder>``, the files of the builds made without capture left to their own tests, the tie record by its seeded
    entries): the paths of the plain world are all listed; ``world.json`` is left out of the byte compare (it holds the new key
    ``extensions`` and R3's files in its lists) and its ``stores``, ``shards`` and ``requests`` are equal; every other file the
    plain world lists has the same bytes, except ``built-from.jsonl`` of the build of main, whose rows are the plain rows in the
    same order with the two rows ``gen/R3-models`` and ``R3`` added among them, after the build's rows and before the rows of the
    checkpoints, which the runs append later (the repaired side's ``built-from.jsonl`` is expected
    byte-equal: a difference there is a finding to read, not a failure to fix blind); and the paths the plain world does not
    list are R3's only (a path that holds ``R3``). A byte that moved in a core file would be a core question read from other
    input. NOT RUN without the Megatron environment."""
    plain = _built(tmp_path_factory, SEVERAL)
    run = _world_with_the_extension(tmp_path_factory)
    assert run.code == 0, f"the build with --extensions generating-model ended with {run.code}: {run.text}"
    other = run.directory
    assert _world_json(other)["extensions"] == ["generating-model"]
    assert _world_json(plain)["extensions"] == []
    base, new = _listed(plain), _listed(other)
    assert set(path for path, _ in base) <= set(path for path, _ in new), \
        f"paths the plain world lists and the world with the extension does not: {sorted(set(p for p, _ in base) - set(p for p, _ in new))[:5]}"
    differ = []
    for path, left in base:
        right = dict(new)[path]
        if left == right or _made_without_capture(path) or path == "world.json":
            continue
        if path == "work/tie.json":
            if _tie_seeded(plain) != _tie_seeded(other):
                differ.append(path)
            continue
        if path == MAIN_BUILT_FROM:
            problem = _built_from_problem((plain / path).read_text(encoding="utf-8").splitlines(),
                                          (other / path).read_text(encoding="utf-8").splitlines())
            if problem is not None:
                differ.append(f"{path} ({problem})")
            continue
        if _normalized(plain / path, plain) != _normalized(other / path, other):
            differ.append(path)
    assert not differ, f"core files whose bytes moved with the extension on: {differ[:8]}"
    first, second = _world_json(plain), _world_json(other)
    for key in ("stores", "shards", "requests"):
        assert _normalized_json(first[key], plain) == _normalized_json(second[key], other), f"world.json {key} differs"
    extra = [path for path, _ in new if path not in set(p for p, _ in base)]
    assert extra and all("R3" in path for path in extra), [p for p in extra if "R3" not in p][:5]


def _normalized_json(value, directory: Path) -> str:
    """A part of world.json as text with the folder of its world read as ``<world folder>``."""
    text = json.dumps(value, sort_keys=True)
    for folder in {str(directory), os.path.normpath(str(directory)), str(directory.resolve())}:
        text = text.replace(folder, "<world folder>")
    return text


def test_decisions_hold_the_world_builder_entries_for_the_tie_merge_the_successor_and_the_requests():
    text = (REPO / "DECISIONS.md").read_text(encoding="utf-8")
    blocks = {}
    for block in re.split(r"(?m)^## (?=D-WORLD-\d+ )", text)[1:]:
        blocks[int(re.match(r"D-WORLD-(\d+) ", block).group(1))] = re.split(r"(?m)^## ", block)[0]
    assert blocks, "DECISIONS.md holds no entry D-WORLD-n"
    assert sorted(blocks) == list(range(1, len(blocks) + 1)), "the entries are not numbered from 1 without a gap"
    for number, body in blocks.items():
        for label in ("Built", "Other option", "Result of the other option", "What would change it"):
            assert label in body, f"D-WORLD-{number} has no {label!r}"
        assert "traced" in body.lower() or "inferred" in body.lower(), number
        assert _banned_in(body) == [], (number, _banned_in(body))
    titles = " ".join(body.splitlines()[0].lower() for body in blocks.values())
    for topic in ("tie", "successor", "request"):
        assert topic in titles, f"no entry on {topic}"


def test_the_document_and_the_decisions_say_the_unseeded_files_are_compared_order_free():
    text = (REPO / "docs" / "baselines.md").read_text(encoding="utf-8")
    heading = "The world builder (temporary section WORLD)"
    section = re.split(rf"(?m)^#+ {re.escape(heading)}$", text)[1]
    section = " ".join(re.split(r"(?m)^## ", section)[0].split())
    order_free = re.compile(r"order[- ]free", re.IGNORECASE)
    assert order_free.search(section), "the section of the world builder does not say the unseeded files are compared order-free"
    assert "no-capture" in section and "WORLD.sha256" in section
    assert _banned_in(section) == [], _banned_in(section)
    blocks = {}
    for block in re.split(r"(?m)^## (?=D-WORLD-\d+ )", (REPO / "DECISIONS.md").read_text(encoding="utf-8"))[1:]:
        blocks[int(re.match(r"D-WORLD-(\d+) ", block).group(1))] = " ".join(re.split(r"(?m)^## ", block)[0].split())
    assert 5 in blocks and order_free.search(blocks[5]), "D-WORLD-5 does not say the unseeded files are compared order-free"
    assert 8 in blocks, "DECISIONS.md holds no entry D-WORLD-8 for the same-files bar"
    body = blocks[8]
    assert order_free.search(body.split(" - Built:")[0]), "the title of D-WORLD-8 does not say order-free"
    assert order_free.search(body), "D-WORLD-8 does not say order-free"
    for word in ("WORLD.sha256", "refus"):
        assert word in body, f"D-WORLD-8 does not mention {word!r}"
    # The worker counts compared are stated: the world of one worker and a world of four (the count of the worlds of record,
    # CNC ruling of 18 Nov 2026; old: two on a small machine).
    assert re.search(r"one worker", body) and re.search(r"\bfour\b", body), "D-WORLD-8 does not state the worker counts compared"
    assert "its memory was not measured" in section, "the section of the world builder names no limit of the large-machine branch"
    for number in (5, 8):
        assert _banned_in(blocks[number]) == [], (number, _banned_in(blocks[number]))


def test_the_readme_gives_the_cap_on_the_workers_of_the_large_branch():
    """The outside review of 504bf64: the README said the larger branch builds a world "with
    one worker per core"; the plan above caps the count (``MOST_WORKERS``), so on a machine with
    more cores the statement was not what the test does."""
    text = " ".join((REPO / "README.md").read_text(encoding="utf-8").split())
    assert f"a worker for each core, at most {MOST_WORKERS}" in text
    assert "one worker per core" not in text


def _flag_values(argv):
    """The flags of one command and the word after each, in order (the values that name a path are kept as they are)."""
    words = [str(word) for word in argv]
    return [(words[i], words[i + 1]) for i in range(1, len(words) - 1, 2)]


@pytest.mark.parametrize("layout", ["round-robin", "partitioned"])
def test_the_worlds_repair_of_a_set_is_the_repair_check_9_makes_without_the_reference_side(tmp_path, monkeypatch, layout):
    """W2 (18 Nov 2026): what shows the repaired side the world builds for the 10k set equals Check 9's repair. Both go through the
    same two commands of ``acceptance_repair`` (``tool_command`` for the report, ``replica_command`` for the repair), recorded here with
    the same historical build in front of each. The REPORT is the same command word for word (the placement path apart, which names
    a file of the folder each writes in); the REPAIR is the same command with the three flags of the reference side (``--reference-set``,
    ``--reference-out``, ``--sides``) left out and no other flag added, dropped, reordered or given another value
    (the repair writes the repaired side into ``--out`` and the successors into ``--store`` before the reference side is built, and
    the reference side and the sides file touch neither: replica/repair.py, ``repair``)."""
    import sqlite3  # noqa: PLC0415
    from contextlib import closing  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    world_mod = need("baselines.world")
    repair_files = world_mod._acceptance("acceptance_repair")
    sets_mod = world_mod._acceptance("acceptance_sets")
    prefix = "set" if layout == "partitioned" else "set-rr"
    names = ["shard-A", "shard-B", "shard-C", "shard-Z", "shard-gen"] if layout == "partitioned" else ["shard-01", "shard-02"]
    instances = {f"{prefix}/{name}": f"i{n}" for n, name in enumerate(names)}
    set_folder = tmp_path / "set"
    set_folder.mkdir()
    (set_folder / "set.toml").write_text("".join(f'[sources.{n}]\nfile = "sources/{n}.jsonl"\n\n' for n in ("A", "B", "C", "Z")), encoding="utf-8")
    store = tmp_path / "hist" / "store.sqlite"
    store.parent.mkdir()
    with closing(sqlite3.connect(store)) as db:
        db.execute("create table t (x)")
        db.commit()
    historical_out = tmp_path / "hist" / "build"
    historical_out.mkdir()
    recorded = {"world": [], "check9": []}
    which = ["world"]

    def tool(store_path, arguments, timeout=0):
        recorded[which[0]].append(("tool", [str(a) for a in arguments]))
        return 0, json.dumps({"artifacts": []}), ""

    def replica(arguments, timeout=0):
        words = [str(a) for a in arguments]
        if words[0] == "build-set":
            # Check 9's own historical build, which the world's repair takes as given: the store it registers in, and what it printed.
            Path(words[words.index("--out") + 1]).mkdir(parents=True, exist_ok=True)
            made = Path(words[words.index("--store") + 1])
            made.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(made)) as db:
                db.execute("create table t (x)")
                db.commit()
            return {"instances": instances}
        recorded[which[0]].append(("replica", words))
        (Path(words[words.index("--out") + 1])).mkdir(parents=True, exist_ok=True)
        (Path(words[words.index("--out") + 1]) / "built-from.jsonl").write_text("{}\n", encoding="utf-8")
        if "--sides" in words:
            Path(words[words.index("--sides") + 1]).write_text(json.dumps({"repaired": {}, "reference": {}}), encoding="utf-8")
        return {}

    monkeypatch.setattr(repair_files, "tool_command", tool)
    monkeypatch.setattr(repair_files, "replica_command", replica)
    summary = {"instances": instances}
    work = tmp_path / "work"
    work.mkdir()
    world_mod._set_repair("10k", layout, set_folder, historical_out, store, summary, work, sets_mod.JOBS, 60, world_mod._Log("t"))
    which[0] = "check9"
    monkeypatch.setattr(sets_mod, "need_megatron", lambda: None)
    monkeypatch.setattr(sets_mod, "load_script", lambda file: SimpleNamespace(count=lambda *args, **kwargs: {}))
    repairs = repair_files.SetRepairs(SimpleNamespace(base=tmp_path / "c9", set_folder=lambda size: set_folder))
    repairs._clean["10k"] = tmp_path / "clean"
    repairs.repaired("10k", layout)
    (w_report,), (c_report,) = [[c for c in recorded[k] if c[0] == "tool"] for k in ("world", "check9")]
    (w_repair,), (c_repair,) = [[c for c in recorded[k] if c[0] == "replica"] for k in ("world", "check9")]

    def without_placement_path(argv):
        return [word.split("=")[0] + "=" if word.startswith(f"{layout}=") and "/" in word and word != f"{layout}={prefix}/" else word
                for word in argv]

    assert without_placement_path(w_report[1]) == without_placement_path(c_report[1])
    reference_flags = {"--reference-set", "--reference-out", "--sides"}
    world_flags = _flag_values(w_repair[1])
    check9_flags = [(flag, value) for flag, value in _flag_values(c_repair[1]) if flag not in reference_flags]
    assert [flag for flag, _ in world_flags] == [flag for flag, _ in check9_flags]
    path_flags = {"--report", "--historical", "--out", "--store", "--set", "--reference-set", "--reference-out", "--sides"}
    assert [(flag, value) for flag, value in world_flags if flag not in path_flags] == [
        (flag, value) for flag, value in check9_flags if flag not in path_flags]
    # The three flags of the reference side are the ones Check 9's repair has and the world's has not.
    assert {flag for flag, _ in _flag_values(c_repair[1])} - {flag for flag, _ in world_flags} == reference_flags
    # Both repair the same set folder.
    assert dict(world_flags)["--set"] == dict(check9_flags)["--set"] == str(set_folder)
