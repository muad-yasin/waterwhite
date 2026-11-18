"""What the tests of the generating-model rule (E2) share. There is no conftest.py here:
other extension lanes use this folder too, so the fixtures live in this module and a test
file imports the ones it needs.

Three things are in it.

1. The hand-built worlds. ``ww_world`` (tests/coverage) is the library world: sources A, B, C,
   Z, the generated P, T and S, releases R1 and R2, two shards, two mixtures and runs E, D and
   Y. It is loaded by its file, the way tests/rebuild/rb_kit.py loads it, so that one copy of
   its module (and of its table of texts) serves every test folder. The kit adds to it:

   - ``add_generated``: a file of generation outputs, each handed one record of a source, every
     log registered WITH its texts (a log without them is not marked compared and establishes
     no bound, so the walk would stop at condition (g) before it reached condition (h));
   - ``add_history``: a checkpoint whose one step reads one document of a one-document shard
     whose map names a given record (the history that a cycle or a model's ancestry needs);
   - ``add_r3_records``: the hand-built form of release R3 (F7, F8, F9 of the spec).

2. ``ask_or_error``: the answer in its JSON form, or, where the tool raises a named error, a
   dictionary with ``"answer": "error: <name>"``, ``"exit_code"`` and ``"detail"``. A test of an
   on-state question uses it and never the bare ``coverage.answer``, so that a question the tool
   refuses fails at the test's assertion on a value and not with an exception.

3. The acceptance level (File 3 of the spec): ``AcceptanceStores`` builds the store of the key
   ``main+generating-model`` and puts a row of the table to a private copy of it. The frozen
   acceptance modules are imported ONLY inside ``acceptance_modules()``, never at the top of this
   module, because ``acceptance_world`` runs command-line probes when it is imported.

Every expected value in the test files comes from the plan, an owner answer, a ruling or the
frozen table, as the spec names; none comes from the code under test or from running it.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from waterwhite import coverage, errors, extensions, requests
from waterwhite import vocabulary as v
from waterwhite.records import SnapshotShard
from waterwhite.store.store import Store

REPO = Path(__file__).resolve().parents[2]
SWITCH = extensions.setting_key(extensions.GENERATING_MODEL)


def _load(name: str, relative: str):
    """A module of the tests, loaded from its file whether or not another test folder has put
    its own folder on the path (one copy, so one table of texts)."""
    held = sys.modules.get(name)
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ww = _load("ww_world", "tests/coverage/ww_world.py")
rb = _load("rb_kit", "tests/rebuild/rb_kit.py")

NOTE_NOT_ASSESSED = v.NOTE_GENERATING_MODEL_NOT_ASSESSED
SUMMARY_RECORDS = ["src/A:L1"]


# --- Fixtures -----------------------------------------------------------------------------


@pytest.fixture
def world(tmp_path):
    built = ww.build(tmp_path)
    yield built
    built.store.close()


@pytest.fixture
def ask(world):
    """Put a withdrawal to one artifact; the answer's JSON form, or the error as
    ``ask_or_error`` gives it. The store is read as it is when the question is put."""

    def put(source: str, artifact: str, *, records=()) -> dict:
        return ask_or_error(world.store, source, artifact, records=records)

    return put


@pytest.fixture
def r3_world(tmp_path):
    """The world of tests/rebuild/rb_kit.py with the hand-built release R3 added."""
    built = rb.build(tmp_path / "w")
    add_r3_records(built)
    yield built
    built.store.close()


# --- Asking -------------------------------------------------------------------------------


def ask_or_error(store, source: str, artifact: str, *, records=(), request=None, session=None,
                 integrity=None) -> dict:
    """The answer of ``coverage.answer`` as JSON, or the named error as a dictionary
    (``answer`` "error: <name>", ``exit_code``, ``detail``). Only the tool's named errors are
    caught: any other exception is a fault of the test and propagates."""
    if request is None:
        request = requests.withdraw(store, source, records=records)
    try:
        return coverage.answer(store, artifact, request, integrity=integrity, session=session).to_json()
    except errors.WaterwhiteError as error:
        return {"answer": f"error: {error.name}", "exit_code": error.exit_code, "detail": error.detail}


def summary(ask) -> dict:
    """The standard question of the spec: withdraw A narrowed to the record src/A:L1, asked of
    gen/S. S is made from A line 2 and C line 1, so the request names neither parent and the
    negative reaches the summary job's generation edge."""
    return ask("A", "gen/S", records=SUMMARY_RECORDS)


def link(answer: dict, *keys: str) -> tuple:
    """The named keys of an answer's failing link; every key is None on an answer with no
    failing link, an error object included."""
    failing = answer.get("failing") or {}
    return tuple(failing.get(key) for key in keys)


# --- Editing the library world ---------------------------------------------------------------


def turn_on(world) -> None:
    """The one way the switch is turned on (F12): the setting of the store file."""
    world.store.set_setting(SWITCH, "on")


def set_model(world, job: str, text: str | None) -> None:
    """The model entry of the stored log of a job, edited from outside the write API. The
    mark ``hand_off_checked`` stays as it was built, so the bound stands."""
    world.sql(("UPDATE harness_log SET model = ? WHERE log_id = ?", (text, world.log_id(job))))


def unmark(world, job: str) -> None:
    """Clear the mark: the log reads as registered without its texts."""
    world.sql(("UPDATE harness_log SET hand_off_checked = 0 WHERE log_id = ?", (world.log_id(job),)))


def append_byte(path: Path) -> None:
    """One byte appended to a file the store registered: its digest no longer matches."""
    with open(path, "ab") as f:
        f.write(b"\x00")


def add_generated(world, name: str, rows: list[tuple[str, str | None, str, int]]) -> str:
    """A file of generation outputs. Each row is (job, model, source name, line): the job is
    handed that one record of the source, states it used it, and its log carries ``model``
    ("none", a text, or None for a log with no model entry). Every log is registered with its
    texts, so it is marked compared. Returns the instance id; the file is ``world[name]``."""
    edges, texts, inputs = [], [], []
    for number, (job, model, source, line) in enumerate(rows, start=1):
        parent = ww.ref(world[source], line)
        edge = ww.generation_edge(number, job, [parent], [parent])
        if model != "none":
            edge = dataclasses.replace(edge, harness_log=dataclasses.replace(edge.harness_log, model=model))
        edges.append(edge)
        texts.append(f"made by {job}")
        inputs.append(world[source])
    world.ids[name] = ww.record(
        world.store, world.folder / f"{name}.jsonl", texts, name, v.ROLE_INTERMEDIATE,
        ww.step("generate", name), list(dict.fromkeys(inputs)), edges)
    return world[name]


def add_generated_from(world, name: str, job: str, handed: list[tuple[str, int]], used: list[int]) -> str:
    """A one-record file made by one job that is handed several records, each (artifact name,
    line), and states it used those at the places ``used`` (counted from 0 in ``handed``); a
    handed record that is not stated is in scope only, and is not a parent. The log carries the
    model "none" and its texts, so it is marked compared. Returns the instance id."""
    refs = [world.ref(artifact, line) for artifact, line in handed]
    edge = ww.generation_edge(1, job, refs, [refs[place] for place in used])
    world.ids[name] = ww.record(
        world.store, world.folder / f"{name}.jsonl", [f"made by {job}"], name, v.ROLE_INTERMEDIATE,
        ww.step("generate", name), list(dict.fromkeys(r.instance_id for r in refs)), [edge])
    return world[name]


def add_history(world, tag: str, record) -> str:
    """A checkpoint ``<tag>@1`` whose history reads the documents of one shard, each of which
    a map row ties to one record: ``record`` is an ``OccurrenceRef`` or a list of them, and step
    n reads document n-1. The checkpoint (named ``<tag>@1`` whatever the number of steps) marks
    the last step, so its history reads exactly those records and nothing else. Returns the
    checkpoint's name."""
    store = world.store
    refs = list(record) if isinstance(record, (list, tuple)) else [record]
    shard_name = f"shard-{tag}"
    instance, digest = ww.shard(
        store, world.folder / "shards" / shard_name, shard_name,
        list(dict.fromkeys(r.instance_id for r in refs)),
        [ww.recovered(n, r) for n, r in enumerate(refs)])
    mixture = store.register_mixture(
        name=f"mix-{tag}", shards=[SnapshotShard(0, instance, f"shards/{shard_name}", digest)],
        settings={"mix": tag})
    steps = [(n, 1, None) for n in range(1, len(refs) + 1)]
    ww.segment(store, tag, f"{tag}/seg-1", 0, mixture, None, steps,
               {n: (0, n - 1) for n in range(1, len(refs) + 1)})
    name = f"{tag}@1"
    world.ids[shard_name] = instance
    world.ids[name] = store.register_checkpoint(name=name, segment=f"{tag}/seg-1", step=len(refs))
    return name


# The hand-built R3 (F7, F8, F9): jobs and models of the four generated records, in the order of
# the file. Checkpoints 3 and 4 of the library world stand for the plan's checkpoints 6 and 9
# (F9: a known difference of the test device): E@3 answers "does not contain" for A and E@4 answers
# "contains", the same cause words as the plan's pair.
R3_ROWS = [
    ("r3-g6", "E@3", "src/Z", 1),
    ("r3-g9", "E@4", "src/Z", 2),
    ("r3-gu", "hosted-model-x", "src/Z", 3),
    ("r3-gm", None, "src/Z", 1),
]
R3_UNREGISTERED = "hosted-model-x"


def add_r3_records(world) -> None:
    """File gen/R3 (four generation edges, registered with their texts) and release R3, which
    copies its four lines (R3:L1 to R3:L4)."""
    add_generated(world, "gen/R3", R3_ROWS)
    texts = [f"made by {job}" for job, *_ in R3_ROWS]
    world.ids["R3"] = ww.record(
        world.store, world.folder / "R3.jsonl", texts, "R3", v.ROLE_RELEASE, ww.step("release", "R3"),
        [world["gen/R3"]], [ww.copy_edge(n, ww.ref(world["gen/R3"], n)) for n in range(1, len(R3_ROWS) + 1)])


# --- The acceptance level (File 3) --------------------------------------------------------------

KEY_STORE = "main+generating-model"
# A row that is not in the table reports "row missing" and never a pass.
ROW_STANDING_NEGATIVE = "c05.withdraw-C.on.E/shard-3"
R3_FIXTURE_ROWS = REPO / "fixture" / "sources" / "generation_r3.jsonl"
FIXTURE_SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"


def r3_fixture_rows() -> dict[str, dict]:
    """The four rows of fixture/sources/generation_r3.jsonl, by id, read with ``json``."""
    lines = R3_FIXTURE_ROWS.read_text(encoding="utf-8").splitlines()
    return {row["id"]: row for row in (json.loads(line) for line in lines if line.strip())}


def r3_spec() -> dict:
    """``[extension_releases.R3]`` of the fixture spec."""
    with open(FIXTURE_SPEC, "rb") as f:
        return tomllib.load(f)["extension_releases"]["R3"]


def acceptance_modules() -> SimpleNamespace:
    """The frozen acceptance helpers, imported now and not at the top of any module."""
    folder = str(REPO / "tests" / "acceptance")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import acceptance_faults
    import acceptance_probe
    import acceptance_runs
    import acceptance_sets
    import acceptance_world
    import table

    from replica import fixture as replica_fixture

    return SimpleNamespace(faults=acceptance_faults, probe=acceptance_probe, runs=acceptance_runs,
                           sets=acceptance_sets, world=acceptance_world, table=table,
                           replica_fixture=replica_fixture)


class AcceptanceStores:
    """The stores a File 3 test asks, under one folder.

    The frozen ``acceptance_world.World`` does not know the key ``main+generating-model``: it
    builds a store by the names of ``table.TOKENIZED_STORES`` and registers runs by
    ``acceptance_runs.RUNS_OF_STORE``. This class builds the key's store from the same public
    pieces (the replica's ``build_table_store``, the replica's run command and the tool's
    ``register-run`` with the runs of ``main``) and hands it to the World's cache, so that the
    World can copy it with a fault. The world builder's own wiring of this key
    replaces this."""

    def __init__(self, mods: SimpleNamespace, base: Path) -> None:
        self.mods = mods
        self.base = Path(base)
        self.world = mods.world.World(self.base)
        self._private = 0

    def key_store(self):
        """The store of the build key, with the runs of main registered (checkpoints 6 and 9
        of run E exist, so R3's logs name registered checkpoints)."""
        mods = self.mods
        if KEY_STORE not in self.world._built:
            root = mods.sets.need_megatron()
            missing = mods.runs.capture_missing()
            if missing is not None:
                pytest.skip(f"needs the run capture: {missing}")
            arguments = {"tokenize": True, "parallel": 3}
            arguments.update(mods.probe.generation_arguments(mods.replica_fixture.build_table_store))
            built = mods.replica_fixture.build_table_store(
                self.base / "stores" / KEY_STORE, KEY_STORE, **arguments)
            runs = mods.runs.RUNS_OF_STORE["main"]
            runs_dir = built.store.parent / "runs"
            mods.runs.build_runs(built.spec, built.out, runs_dir, runs, root)
            mods.runs.register_runs(built.store, runs_dir, runs, root)
            self.world._built[KEY_STORE] = built
        return self.world._built[KEY_STORE]

    def private_copy(self, built) -> Path:
        """A folder of its own with the store as ``built`` holds it, so that a row's switch is
        written on a copy and no other row reads it."""
        self._private += 1
        target = self.base / "private" / f"{self._private:03d}"
        shutil.copytree(built.store.parent, target)
        return target / built.store.name

    def store_file(self, store: str, state: str = "unmutated") -> Path:
        """A private copy of a store (the build key or a store of the table) in a state."""
        if store == KEY_STORE:
            self.key_store()
        return self.private_copy(self.world.copy(store, state))

    def put_row(self, row_id: str, *, store: str | None = None) -> list[str]:
        """Put one row of the table, with its extension on, to a private copy of the store it
        names (the build key where it asks about R3), and return what differs from the row's
        ``expect``. An empty list means the row answers yes. The expected values are the row's:
        nothing is read from the tool to make them. A row that is not in the table, or a fault
        state no helper knows, is reported, never passed."""
        mods = self.mods
        table = mods.table
        row = next((r for r in table.load_check(int(row_id[1:3])) if r.id == row_id), None)
        if row is None:
            return [f"row missing: {row_id}"]
        question = table.interpret(row)
        if store is None:
            store = KEY_STORE if question.artifacts == ("R3",) else question.store
        try:
            path = self.store_file(store, question.state)
        except Exception as failure:  # a state that no helper of tests/acceptance knows yet
            return [f"the state {question.state!r} of {row_id} cannot be put: {failure}"]
        if row.extension is not None:
            self.switch(path, "on")
        resolve = lambda ref: self.world.locator(store, ref)  # noqa: E731
        # The fields the rebuild report gives (``question.deferred``) are not the answer's: they
        # are taken out as ``World._answer`` takes them out.
        expect = {k: v_ for k, v_ in question.expect.items() if k not in question.deferred}
        if question.record is not None:
            return self._record(question, store, path, expect, resolve)
        asked = {k: v_ for k, v_ in expect.items() if k not in ("unresolved", "not_unresolved")}
        outcome = self.world.cli(path, *self.world.query_arguments(question))
        found = [str(m) for m in table.compare_outcome(asked, outcome, resolve)]
        if "unresolved" in expect or "not_unresolved" in expect:
            found += self._unresolved(question, store, path, expect)
        return found

    @staticmethod
    def switch(path: Path, value: str) -> None:
        """Write the switch of the generating-model extension on a store file (F12)."""
        held = Store.open(path)
        try:
            held.set_setting(SWITCH, value)
        finally:
            held.close()

    def r3_entry(self, handle: str) -> dict:
        """The rebuild report's entry of one record of R3 (by its handle in the fixture file),
        from a private copy of the key's store with the switch on."""
        path = self.store_file(KEY_STORE)
        self.switch(path, "on")
        by_handle = {h: key for key, h in self._handles(KEY_STORE, "R3").items()}
        wanted = by_handle[handle]
        return next(e for e in self._report(path)["records"] if e["artifact"] == "R3" and e["record"] == wanted)

    def _report(self, path: Path) -> dict:
        outcome = self.world.cli(path, "report", "withdraw", "A")
        if outcome.exit_code != 0:
            raise AssertionError(f"report withdraw A exited with {outcome.exit_code}: {outcome.stderr[:300]}")
        return outcome.printed()

    def _handles(self, store: str, artifact: str) -> dict[str, str]:
        """The record of each handle of a file the build registered, as ``artifact:L<n>``,
        from the manifest the replica wrote beside the file."""
        built = self.world.store(store)
        return {f"{artifact}:L{row['line']}": row["handle"] for row in built.manifest_rows(artifact)}

    def _record(self, question, store: str, path: Path, expect: dict, resolve) -> list[str]:
        """A row about one record of R3 (F22, A): the rebuild report's record entry, action
        "keep" read as "does not contain" and "unresolved" as "may contain", with the entry's
        ``names`` and ``failing``."""
        document = self._report(path)
        by_record = {handle: key for key, handle in self._handles(store, question.artifact).items()}
        wanted = by_record[question.record]
        entries = [e for e in document["records"]
                   if e["artifact"] == question.artifact and e["record"] == wanted]
        if len(entries) != 1:
            return [f"the report holds {len(entries)} entries for {wanted}"]
        entry = entries[0]
        words = {"keep": "does not contain", "unresolved": "may contain"}
        shown = {"answer": words.get(entry["action"], entry["action"]), "names": entry.get("names"),
                 "failing": entry.get("failing")}
        return [str(m) for m in self.mods.table.compare_answer(expect, shown, resolve)]

    def _unresolved(self, question, store: str, path: Path, expect: dict) -> list[str]:
        document = self._report(path)
        handles = self._handles(store, question.artifact)
        listed = sorted(handles[e["record"]] for e in document["records"]
                        if e["artifact"] == question.artifact and e["action"] == "unresolved")
        out = []
        if "unresolved" in expect and listed != sorted(expect["unresolved"]):
            out.append(f"unresolved: expected {sorted(expect['unresolved'])}, the report lists {listed}")
        for handle in expect.get("not_unresolved", ()):
            if handle in listed:
                out.append(f"not_unresolved: {handle} is listed as unresolved")
        return out

    def query(self, path: Path, *arguments: str) -> dict:
        """The printed answer of one command of the tool on a store file."""
        outcome = self.world.cli(path, *arguments)
        if outcome.exit_code != 0:
            raise AssertionError(f"{' '.join(arguments)} exited with {outcome.exit_code}: {outcome.stderr[:300]}")
        return outcome.printed()


@pytest.fixture(scope="session")
def acceptance(tmp_path_factory):
    """The acceptance stores of this session. Skipped, with the acceptance helpers' own words,
    where the Megatron environment is not set or the sandbox cannot be made: a skipped test is
    not a pass."""
    mods = acceptance_modules()
    if mods.sets.megatron_root() is None:
        pytest.skip(f"needs the Megatron environment: {mods.sets.MEGATRON_VARIABLE} is not set")
    reason = mods.probe.sandbox_missing()
    if reason is not None:
        pytest.skip(reason)
    # One set of stores for the whole session, though the fixture is imported into several test
    # modules (pytest instantiates an imported fixture once per module): the builds are long.
    if not _STORES:
        _STORES.append(AcceptanceStores(mods, tmp_path_factory.mktemp("e2-acceptance")))
    return _STORES[0]


_STORES: list[AcceptanceStores] = []


def run_cli(path: Path, *arguments: str) -> subprocess.CompletedProcess:
    """One command of the tool in a subprocess (the way the frozen World puts a question)."""
    return subprocess.run(
        [sys.executable, "-m", "waterwhite.cli", *arguments, "--store", str(path)],
        capture_output=True, text=True, timeout=60.0,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
