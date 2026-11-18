"""The stores the hand-written table asks, and the way each kind of question is put.

``World`` builds each store of fixture/table/README.md section 4 once, with the
replica, and hands out fresh copies with one fault applied (section 7). It
resolves record handles to locators through the replica's manifests, never through
the tool. ``put`` asks one interpreted row and returns the fields that differ.

Every question but one goes through the command line: a subprocess of
``python -m waterwhite.cli``, with the answer read from standard output and a
refusal or loud stop from the exit code and the error object on standard error.
The exception is the harness row of Check 5: the command line has no command that
runs a job, so that row runs a job script of ``jobs/`` in the confined harness
(``waterwhite.confine.run_confined_job``), and is not run where the sandbox cannot
be made.

The rows of Checks 3 and 6 are put here too. What they compare is in
``acceptance_maps.py``; the generated sets they are asked of are in
``acceptance_sets.py``. Their expected values come from the plant list as the
counting script counts it, from the shard's own index, from the replica's
sanitizer log and from the hand-written table, never from the tool.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import acceptance_faults
import acceptance_join
import acceptance_maps as maps
import acceptance_probe
import acceptance_repair
import acceptance_report as rep
import acceptance_runs
import acceptance_scratch
import acceptance_sets
import acceptance_trace
import table
from acceptance_sets import SetBuild, Unavailable
from table import Mismatch, Outcome, Question, RecordRef

from replica import fixture as replica_fixture
from replica.fixture import CapturedFixture

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
CLI_TIMEOUT = 30.0  # seconds, for one command
CONFINED_TIMEOUT = 60.0  # seconds, for one confined job
JOBS = Path(__file__).resolve().parent / "jobs"

# What the replica and the tool do not build yet is not a list kept here: it is found out
# by asking. ``acceptance_probe`` asks the command line, once per run, for the help of
# `show` forms, `replay`, `report`, `verify` and the replica's `repair`, and an output the
# tool prints leaves ``NOT_BUILT``, so the rows that wait for it start to run on the
# commit that builds it, with no edit in this folder. A row that needs a thing in
# ``NOT_BUILT`` is skipped with the need as its reason; a skipped row is not a pass.
# ``World`` refuses to go on when the replica registers an artifact of a kind that
# ``NOT_BUILT`` still calls missing, so the list cannot go stale without a test saying so.
#
# Two things are not in this list, because the replica says them itself and a list
# here would go stale without a test noticing. Whether the replica builds a store
# of the table: it refuses the ones it does not build, by name
# (``replica.fixture.TABLE_STORES_NOT_BUILT``). Whether a store holds shards or
# mixtures: the names its build registered say so (``World.unmet``).

# The needs a registered run meets (README.md, "The run world"): a step trace, a
# checkpoint and the record W that a run reads exist once the runs of a store are
# registered. They are asked of the run world as one thing, ``NEED_RUN_CAPTURE``.
RUN_NEEDS = frozenset({table.NEED_TRACE, table.NEED_CHECKPOINT, table.NEED_SHARED_TOKEN,
                       table.NEED_RUN_CAPTURE})

# What ``show`` does not print (README.md, "What show must print for Checks 3 and
# 6") and what the rebuild's commands lack (the report, the verification, the
# replica's repair word) are found by the probe, once per run.
NOT_BUILT: dict[str, str] = {**acceptance_probe.show_gaps(), **acceptance_probe.repair_gaps()}

# The state of README section 7 that registers the successors of a repair beside the
# historical artifacts of a store (Checks 7 and 9).
REPAIRED = "r.successor-registered"

# What this folder did to a store before putting a row to it, in words: the summary of a run
# prints each, so that a pass is never read as the tool's own when a set-up made it possible.
SET_UP_NOTES: list[str] = []

# A row of a store that is tokenized only where Megatron is set, run without it.
NEED_MEGATRON_ENV = "the Megatron environment"

# The needs the store's own build answers: a store holds a shard or a mixture when its
# build registered one, so no list here can go stale.
# A tokenization map is written with its shard, so a store that registered a shard holds one.
REGISTERED_KIND = {table.NEED_SHARD: table.ARTIFACT_SHARD,
                   table.NEED_MIXTURE: table.ARTIFACT_MIXTURE,
                   table.NEED_MAP: table.ARTIFACT_SHARD}

# Where ``show`` prints no rows of a named map, the rows of one shard are read
# document by document, when the map asked for is the one a reader gets first and
# the shard holds no more documents than this.
PER_DOCUMENT_LIMIT = 64
SHOW_WORKERS = 8  # command lines at a time, where a row asks many small questions
ROW_KEYS = ("record", "label", "token_start", "token_end", "parent", "candidates")
# The store names of the fixture's shards in the partitioned layout
# (docs/interfaces.md section 9.2).
FIXTURE_SHARDS = "E/"


class WorldError(Exception):
    """The acceptance infrastructure cannot do what a row asks of it."""


class World:
    """The stores of one test session, under one folder.

    ``store(name)`` is the store as the replica built it, plus what this folder does to a
    store before a row is put to it (the runs registered, ``SET_UP_NOTES``). Rows read it
    and write nothing; a row with a fault, or one that writes, works on
    ``copy(name, state)``."""

    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        self._built: dict[str, CapturedFixture] = {}
        self._failed: dict[str, Exception] = {}
        self._copies: dict[tuple[str, str], CapturedFixture] = {}
        self._shown: dict[tuple[str, ...], Mapping] = {}
        self._lines: dict[Path, list[str]] = {}
        self._scratch = 0
        # The repairs of the stores of the table, and the documents the report and verify
        # printed for them (name, command line) -> the document or the failure.
        self._repairs: dict[str, acceptance_repair.Repaired] = {}
        self._documents: dict[tuple[str, ...], tuple[Mapping | None, str | None]] = {}
        # What register-run printed: name -> per segment; (name, state) for a starting state.
        self._registered: dict = {}
        # The folder of the changed run records of a starting state: (name, state) -> folder.
        self._records: dict[tuple[str, str], Path] = {}
        self.harness_evidence: dict = {}
        self.sets = acceptance_sets.Sets(self.base / "sets")

    # --- Stores ------------------------------------------------------------------------

    @staticmethod
    def refused(name: str) -> str | None:
        """Why the replica does not build a store of the table, in its own words,
        or None when it builds it."""
        return getattr(replica_fixture, "TABLE_STORES_NOT_BUILT", {}).get(name)

    def store(self, name: str) -> CapturedFixture:
        if name in self._failed:
            # A build that failed leaves its folder behind, and a second build into it
            # is refused for that reason. Every row of the store reports the first cause.
            raise self._failed[name]
        if name not in self._built:
            try:
                self._built[name] = self._build(name)
            except Unavailable:
                raise
            except Exception as failure:
                self._failed[name] = failure
                raise
        return self._built[name]

    def _build(self, name: str) -> CapturedFixture:
        if self.refused(name) is not None:
            raise Unavailable(f"not built yet: the store {name!r} of the table (the "
                              f"replica: {self.refused(name)})")
        arguments = {}
        if name in table.TOKENIZED_STORES:
            acceptance_sets.need_megatron()
            arguments = {"tokenize": True, "parallel": 3}
        elif name in table.TOKENIZED_WHERE_POSSIBLE and acceptance_sets.megatron_root():
            arguments = {"tokenize": True, "parallel": 3}
        arguments.update(acceptance_probe.generation_arguments(
            replica_fixture.build_table_store))
        built = replica_fixture.build_table_store(self.base / "stores" / name, name,
                                                  **arguments)
        self._register_runs(name, built)
        for registered in built.instances:
            try:
                needs = table.artifact_needs(registered)
            except table.TableError:
                continue  # a name the table asks no row about says nothing of NOT_BUILT
            stale = [need for need in needs if need in NOT_BUILT]
            if stale:
                raise WorldError(
                    f"the replica registered {registered!r} in the store {name!r}, and "
                    f"tests/acceptance/acceptance_world.py NOT_BUILT still lists {stale} as "
                    "not built: remove the entry and add what its rows need")
        return built

    def _register_runs(self, name: str, built: CapturedFixture) -> None:
        """The runs of a store of the table, built by the replica's run wrapper and
        registered through the tool, where both exist and the store holds shards.
        A store with no runs after this is one the rows that need a run skip, by
        ``unmet``. A failure of either command is an error of the run."""
        runs = acceptance_runs.RUNS_OF_STORE.get(name)
        root = acceptance_sets.megatron_root()
        if (runs is None or root is None or acceptance_runs.capture_missing() is not None
                or not any(_kind_or_none(n) == table.ARTIFACT_SHARD for n in built.instances)):
            return
        # The store before any run, kept beside it: Check 2 registers the runs again
        # in a fresh store from a starting state, never in one that holds the traces.
        if not acceptance_runs.verified_joins(built.store):
            # Every segment would register with "data-order trace unverified for this
            # version" and no row could be answered from a trace: a fault of the build
            # that made the store, reported as an error of the run, never mended here.
            raise WorldError(f"the store {name!r} that is to hold runs {list(runs)} was built "
                             "with no verified join")
        acceptance_runs.snapshot_store(built.store, self._before_runs(built))
        runs_dir = built.store.parent / "runs"
        acceptance_runs.build_runs(built.spec, built.out, runs_dir, runs, root)
        self._registered[name] = acceptance_runs.register_runs(built.store, runs_dir, runs, root)

    @staticmethod
    def _before_runs(built: CapturedFixture) -> Path:
        return built.store.with_name("before-runs.sqlite")

    def registered(self, name: str, state: str = "unmutated") -> dict[str, dict]:
        """What ``register-run`` printed for each segment of a store, in the starting
        state ``state`` of Check 2 (a fresh store from the one before any run)."""
        self.store(name)
        if state == "unmutated":
            if name not in self._registered:
                raise Unavailable("not built yet: run capture (the store has no registered run)")
            return self._registered[name]
        self.copy(name, state)
        return self._registered[(name, state)]

    def _cache_state(self, name: str, state: str) -> CapturedFixture:
        """The store ``name`` with its runs registered in a fresh store from a starting
        state of Check 2: the store as it was before any run, the run records and
        cache folders as the state has them."""
        built = self.store(name)
        root = acceptance_sets.need_megatron()
        if name not in self._registered:
            raise Unavailable("not built yet: run capture (the store has no registered run)")
        fresh = self._before_runs(built).with_name(f"store-{state}.sqlite")
        acceptance_runs.snapshot_store(self._before_runs(built), fresh)
        self._records[(name, state)] = self.scratch(f"records-{state}")
        self._registered[(name, state)] = acceptance_runs.register_runs(
            fresh, built.store.parent / "runs", acceptance_runs.RUNS_OF_STORE[name], root,
            state=state, scratch=self._records[(name, state)])
        return dataclasses.replace(built, store=fresh)

    def _fresh_copy(self, name: str, label: str) -> CapturedFixture:
        built = self.store(name)
        folder = self.base / "copies" / f"{name}--{label}"
        shutil.copytree(built.store.parent, folder)
        return dataclasses.replace(built, store=folder / built.store.name,
                                   out=folder / built.out.name)

    def copy(self, name: str, state: str) -> CapturedFixture:
        """The store ``name`` in ``state``. An unmutated store, and a name that is
        on purpose not registered, are the store as built. A fault is put on a
        fresh copy of the store's folder, one fault per copy; the copy is made once
        and every row of that state reads it."""
        if state == REPAIRED:
            repaired = self._repaired(name)
            return dataclasses.replace(self.store(name), store=repaired.store, out=repaired.out)
        if state in ("unmutated", "q.unregistered", "t.no-tokens-line"):
            # The set-up t.no-tokens-line is the store scratch-notokens itself.
            return self.store(name)
        key = (name, state)
        if state in acceptance_runs.STATES:
            if key not in self._copies:
                self._copies[key] = self._cache_state(name, state)
            return self._copies[key]
        if key not in self._copies:
            copied = self._fresh_copy(name, state)
            acceptance_faults.apply(state, copied.store, lambda ref: self.locate(name, ref))
            self._copies[key] = copied
        return self._copies[key]

    def unmet(self, question: Question) -> list[str]:
        """The needs of a question that are not met, in the question's order: the
        ones ``NOT_BUILT`` lists, and a shard or a mixture where the store's build
        registered none. Whether a store that Megatron tokenizes holds shards cannot be told
        on a machine without the Megatron environment; the row is then skipped for
        the environment when it is put."""
        out = []
        for need in question.needs:
            if need in NOT_BUILT:
                out.append(need)
            elif need in RUN_NEEDS:
                if acceptance_runs.capture_missing() is not None:
                    # The reason is one thing whichever run need asked: say it once.
                    if table.NEED_RUN_CAPTURE not in out:
                        out.append(table.NEED_RUN_CAPTURE)
                elif acceptance_sets.megatron_root() is None and NEED_MEGATRON_ENV not in out:
                    out.append(NEED_MEGATRON_ENV)
            elif need in REGISTERED_KIND and not question.sized and need not in out:
                name = question.store
                if self.refused(name) is not None:
                    out.append(need)
                elif (name in table.TOKENIZED_WHERE_POSSIBLE
                      and acceptance_sets.megatron_root() is None):
                    if NEED_MEGATRON_ENV not in out:
                        out.append(NEED_MEGATRON_ENV)
                elif name not in table.TOKENIZED_STORES or acceptance_sets.megatron_root():
                    kinds = [k for k in map(_kind_or_none, self.store(name).instances) if k]
                    if REGISTERED_KIND[need] not in kinds:
                        out.append(need)
        return out

    def scratch(self, label: str) -> Path:
        """A new folder for one row's own files."""
        self._scratch += 1
        folder = self.base / "scratch" / f"{self._scratch:03d}-{label}"
        folder.mkdir(parents=True)
        return folder

    # --- Handles, through the replica's manifests ------------------------------------------

    def line(self, store: str, ref: RecordRef) -> int:
        """The line of a record in its file, read from the manifest the replica
        wrote beside the file."""
        built = self.store(store)
        wanted = table.manifest_handles(ref.handle)
        lines = [row["line"] for row in built.manifest_rows(ref.artifact)
                 if row["handle"] in wanted]
        if len(lines) != 1:
            raise WorldError(f"the manifest of {ref.artifact} in the store {store!r} holds "
                             f"{ref.handle!r} on {len(lines)} lines; one was expected")
        return lines[0]

    def locator(self, store: str, ref: RecordRef) -> str:
        """``L<n>``: line n of a JSONL file, counted from 1; ``D<n>``: document n of a
        shard, counted from 0 (docs/interfaces.md section 3). A shard has no manifest
        of its own: its document n is line n + 1 of its tokenizer input, whose
        manifest the replica wrote (plan F1: one document per line, in order; the spec's
        ordinals are tied to these lines by tests/oracle/test_oracle_spec_positions.py). That
        holds where no line of the input gives no tokens, as in every shard of run E
        (the spec plants none there); Check 3's scratch shard, which holds such a line,
        is graded by its own comparison and never asked through this lookup."""
        if _kind_or_none(ref.artifact) == table.ARTIFACT_SHARD:
            line = self.line(store, RecordRef(f"{ref.artifact}.input", ref.handle))
            return f"D{line - 1}"
        return f"L{self.line(store, ref)}"

    def locate(self, store: str, ref: RecordRef) -> tuple[str, str]:
        """(instance id, locator) of a record, from the replica's records."""
        return self.store(store).instances[ref.artifact], self.locator(store, ref)

    def text(self, store: str, ref: RecordRef) -> str:
        """The text of a record, read from the line its manifest names."""
        path = self.store(store).file(ref.artifact)
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        return json.loads(lines[self.line(store, ref) - 1])["text"]

    # --- The command line ---------------------------------------------------------------------

    def cli(self, store_file: Path, *arguments: str) -> Outcome:
        """Run one command of the tool in a subprocess."""
        done = subprocess.run(
            [sys.executable, "-m", "waterwhite.cli", *arguments, "--store", str(store_file)],
            capture_output=True, text=True, timeout=CLI_TIMEOUT, cwd=self.base,
            env={**os.environ, "PYTHONPATH": str(SRC)})
        return Outcome(done.returncode, done.stdout, done.stderr)

    def shown(self, store_file: Path, *arguments: str, again: bool = False) -> Mapping:
        """What ``show`` prints, which must be an answer (exit code 0). A store is
        not written after its fault is put on it, so one command is run once per
        store file; ``again`` runs it anew, for the one row that writes between
        two readings."""
        key = (str(store_file), *arguments)
        if again or key not in self._shown:
            outcome = self.cli(store_file, "show", *arguments)
            if outcome.exit_code != 0:
                raise WorldError(f"show {' '.join(arguments)} ended with exit code "
                                 f"{outcome.exit_code}: {outcome.stderr[:300]}")
            self._shown[key] = outcome.printed()
        return self._shown[key]

    def query_arguments(self, question: Question) -> list[str]:
        request = question.request
        if request.kind == "ban":
            banned = table.record_ref(request.banned)
            arguments = ["query", "ban", self.text(question.store, banned)]
        elif request.kind == "withdraw":
            source, handles = request.source, request.records
            if source is None:
                # The request withdraw W: the source and the record are the spec's items
                # (positions.shared_token, README section 3); the row names the record.
                fixed = acceptance_runs.spec()["positions"]["shared_token"]
                source, handles = fixed["source"], (fixed["record"],)
                if table.fill_pending(question.record, acceptance_runs.spec())[0] != fixed["record"]:
                    raise WorldError(f"row {question.row.id}: the record W of the row is not the "
                                     "spec's shared-token record")
            arguments = ["query", "withdraw", source]
            for handle in handles:
                ref = table.record_ref(handle)
                arguments += ["--record", f"{ref.artifact}:{self.locator(question.store, ref)}"]
        else:
            raise WorldError(f"no way to put a request of kind {request.kind!r}")
        if question.artifacts != (table.LISTING,):
            arguments += ["--artifact", question.artifact]
        return arguments

    # --- Putting a question --------------------------------------------------------------------

    def put(self, question: Question, size: str | None = None) -> list[Mismatch]:
        """Ask one row and return the expected fields the outcome does not match.
        A row that is asked of a generated set is asked of the set of ``size``."""
        if question.sized != (size is not None):
            raise WorldError(f"row {question.row.id}: a row is asked of a generated set of a "
                             "named size, or of a store of the table")
        try:
            ask = _WAYS[question.kind]
        except KeyError:
            raise WorldError(f"row {question.row.id}: nothing is built that puts a question of "
                             f"the kind {question.kind!r}, and the row names no need that is "
                             "missing") from None
        return ask(self, question, size) if question.sized else ask(self, question)

    def store_of(self, question: Question) -> CapturedFixture:
        """The store a question is put to: the store in the question's state, and, for a
        question about a successor, the store the repair registered it in."""
        state = question.state
        if state == "unmutated" and any(
                name != table.LISTING and _kind_or_none(name) == table.ARTIFACT_SUCCESSOR
                for name in question.artifacts):
            state = REPAIRED
        return self.copy(question.store, state)

    def _answer(self, question: Question) -> list[Mismatch]:
        store = self.store_of(question)
        # The fields the rebuild report gives are read from it, the others from the answer.
        in_report = {k: v for k, v in question.expect.items() if k in question.deferred}
        asked = dataclasses.replace(
            question, expect={k: v for k, v in question.expect.items() if k not in in_report})
        out: list[Mismatch] = []
        if asked.expect:
            outcome = self.cli(store.store, *self.query_arguments(question))
            self._bound_is_counted(question, outcome)
            out += table.compare_outcome(asked.expect, outcome,
                                         lambda ref: self.locator(question.store, ref))
        if in_report:
            out += self._report_fields(dataclasses.replace(question, expect=in_report))
        return out

    def _bound_is_counted(self, question: Question, outcome: Outcome) -> None:
        """A row whose answer lists generation edges rests on a bound. When the
        row is not among those ``acceptance_probe.rests_on_bound`` counts, a machine
        with no sandbox would run it as if no bound mattered."""
        if outcome.exit_code != 0:
            return
        scope = outcome.printed().get("scope")
        if (isinstance(scope, Mapping) and scope.get("generation_edges")
                and not acceptance_probe.rests_on_bound(question)):
            raise WorldError(
                f"row {question.row.id}: the answer lists the generation edges "
                f"{scope['generation_edges']}, so it rests on a bound, and "
                "acceptance_probe.rests_on_bound does not count the row: a machine with no "
                "sandbox would run it")

    # --- The rebuild report, the repair and the verification (Checks 6, 7 and 9) -------------

    def _document(self, store: Path, arguments: Sequence[str], form: str,
                  what: str) -> tuple[Mapping | None, str | None]:
        """(document, None), or (None, why not) where the command ended with an error. Each
        command line is run once per store file."""
        key = (str(store), *arguments)
        if key not in self._documents:
            outcome = self.cli(store, *arguments)
            if outcome.exit_code != 0:
                self._documents[key] = (None, f"no {what}: exit code {outcome.exit_code}, "
                                              f"error {outcome.error_name()!r}, standard "
                                              f"error {outcome.stderr[:300]!r}")
            else:
                self._documents[key] = (rep.document(outcome.printed(), form, what), None)
        return self._documents[key]

    def report_of(self, question: Question,
                  layouts: Sequence[str] = ()) -> tuple[Mapping | None, str | None]:
        """The rebuild report of the question's request, put to the store as the replica built
        it (a report is made before any repair). ``layouts`` are ``NAME=PREFIX`` words."""
        store = self.copy(question.store, "unmutated")
        asked = dataclasses.replace(question, artifacts=(table.LISTING,))
        arguments = ["report", *self.query_arguments(asked)[1:]]
        for layout in layouts:
            arguments += ["--layout", layout]
        return self._document(store.store, arguments, rep.REPORT_FORM, "rebuild report")

    def report_name(self, store: str, artifact: str, handle: str) -> str:
        """The name a report gives a record: ``artifact:locator`` of the file that holds the
        handle, which is the artifact itself, or, for a record the artifact does not hold (the
        dedup loser that a promote returns), the record's home file (README section 5)."""
        try:
            return f"{artifact}:{self.locator(store, RecordRef(artifact, handle))}"
        except WorldError:
            home = table.record_ref(handle)
            return f"{home.artifact}:{self.locator(store, home)}"

    def _parent_name(self, store: str, text: str) -> str:
        ref = table.record_ref(text)
        return f"{ref.artifact}:{self.locator(store, ref)}"

    def _report_fields(self, question: Question) -> list[Mismatch]:
        """The expected fields of one artifact's entry of the report (Check 9's action rows,
        Check 7's counts and the action of R2)."""
        document, failure = self.report_of(question)
        if document is None:
            return [Mismatch("rebuild report", "a rebuild report", failure)]
        return rep.compare_artifact_fields(
            question.expect, document, question.artifact,
            lambda handle: self.report_name(question.store, question.artifact, handle))

    def _report_record_rows(self, question: Question) -> list[Mismatch]:
        document, failure = self.report_of(question)
        if document is None:
            return [Mismatch("rebuild report", "a rebuild report", failure)]
        names = [self.report_name(question.store, question.artifact, handle)
                 for handle in acceptance_repair.handles_of(question.row.given)]
        return rep.compare_record_rows(
            question.expect, document, question.artifact, names,
            lambda text: self._parent_name(question.store, text))

    def _record_rows(self, question: Question) -> list[Mismatch]:
        """A row about one record or several, under a request: what the report says of them,
        what a successor holds, what the history of a release still says."""
        expect = question.expect
        report_keys = {k: v for k, v in expect.items() if k in table.REPORT_KEYS}
        successor_keys = {"parent", "label", "rows_in_successor", "absent"}
        known = set(report_keys) | successor_keys | {"selected_parent", "answer", "instance_id"}
        if set(expect) - known:
            raise WorldError(f"row {question.row.id}: nothing reads the expected fields "
                             f"{sorted(set(expect) - known)} of a record under a request")
        out: list[Mismatch] = []
        if report_keys:
            out += self._report_record_rows(dataclasses.replace(question, expect=report_keys))
        if successor_keys & set(expect):
            out += self._successor_rows(dataclasses.replace(
                question, expect={k: v for k, v in expect.items() if k in successor_keys}))
        if "selected_parent" in expect:
            out += self._attribution(dataclasses.replace(
                question, expect={"selected_parent": expect["selected_parent"]}))
        if "answer" in expect:
            out += self._answer(dataclasses.replace(
                question, expect={"answer": expect["answer"]}, deferred={}))
        if "instance_id" in expect:
            out += self._instance_unchanged(question)
        return out

    def _instance_unchanged(self, question: Question) -> list[Mismatch]:
        """The historical artifact is the same instance after the repair (plan 4.4)."""
        if question.expect["instance_id"] != "unchanged":
            raise WorldError(f"row {question.row.id}: no way to read instance_id = "
                             f"{question.expect['instance_id']!r}")
        after = self.shown(self.store_of(question).store, question.artifact).get("instance")
        before = self.store(question.store).instances[question.artifact]
        return [] if after == before else [Mismatch("instance_id", before, after)]

    def _registered_file(self, store: CapturedFixture, shown: Mapping) -> Path:
        """The file ``show`` says an artifact was registered from."""
        path = shown.get("path")
        if not isinstance(path, str):
            raise WorldError(f"show prints no path for {shown.get('artifact')!r}: {path!r}")
        for candidate in (Path(path), store.store.parent / path, store.out / path):
            if candidate.is_file():
                return candidate
        raise WorldError(f"the file {path!r} that {shown.get('artifact')!r} was registered "
                         "from is not there")

    def _successor_rows(self, question: Question) -> list[Mismatch]:
        """Where a row of the successor comes from, and what the successor holds. The row X is
        found in the successor's file by its text (the docs fix no manifest of a successor);
        its edge is read with ``show --record``."""
        expect = question.expect
        store = self.copy(question.store, REPAIRED)
        shown = self.shown(store.store, question.artifact)
        texts = acceptance_repair.jsonl_texts(self._registered_file(store, shown))
        out: list[Mismatch] = []
        if "rows_in_successor" in expect and len(texts) != expect["rows_in_successor"]:
            out.append(Mismatch("rows_in_successor", expect["rows_in_successor"], len(texts)))
        if "absent" in expect:
            held = [h for h in expect["absent"]
                    if self.text(question.store, table.record_ref(h)) in texts]
            if held:
                out.append(Mismatch("absent", expect["absent"], f"still present: {held}"))
        if {"parent", "label"} & set(expect):
            wanted = self.text(question.store, RecordRef("R1", question.record))
            rows = [n for n, text in enumerate(texts, 1) if text == wanted]
            edge = (self.shown(store.store, question.artifact, "--record", f"L{rows[0]}").get(
                "edge") if len(rows) == 1 else None)
            if not isinstance(edge, Mapping):
                got = f"{len(rows)} rows with the text of {question.record}, no edge"
                out += [Mismatch(name, expect[name], got) for name in ("parent", "label")
                        if name in expect]
            else:
                parent = edge.get("selected_parent")
                if parent is None and len(edge.get("parents", [])) == 1:
                    parent = edge["parents"][0]
                named = (f"{parent['artifact']}:{parent['record']}"
                         if isinstance(parent, Mapping) else None)
                wanted_parent = (self._parent_name(question.store, expect["parent"])
                                 if "parent" in expect else None)
                if "parent" in expect and named != wanted_parent:
                    out.append(Mismatch("parent", wanted_parent, named))
                if "label" in expect and edge.get("label") != expect["label"]:
                    out.append(Mismatch("label", expect["label"], edge.get("label")))
        return out

    def _repaired(self, name: str) -> acceptance_repair.Repaired:
        """The store ``name`` repaired as the report of ``withdraw A`` says, in a COPY of its
        folder: the report is printed by the tool and kept as a file, the replica builds the
        repaired side from that file with capture on into the copy, and writes the sides file.
        The store the other rows read never holds a successor."""
        if name not in self._repairs:
            built = self.store(name)
            folder = self.scratch(f"repair-{name}")
            code, stdout, stderr = acceptance_repair.tool_command(
                built.store, ["report", "withdraw", "A", "--layout",
                              f"partitioned={FIXTURE_SHARDS}"], CLI_TIMEOUT)
            if code != 0:
                raise WorldError(f"the report that drives the repair of {name!r} ended with exit "
                                 f"code {code}: {stderr[:300]}")
            report = folder / "report.json"
            report.write_text(stdout, encoding="utf-8")
            # The whole folder of the store is copied, so that its relative paths hold, and the
            # repaired side is built inside the copy.
            copied = self._fresh_copy(name, REPAIRED)
            sides = folder / "sides.json"
            repaired = copied.store.parent / "repaired"
            try:
                summary = acceptance_repair.replica_command([
                    "repair", "--report", report, "--historical", built.out, "--out", repaired,
                    "--store", copied.store, "--spec", built.spec, "--layout", "partitioned",
                    "--sides", sides])
            except acceptance_repair.RepairError as failure:
                raise WorldError(str(failure)) from failure
            self._repairs[name] = acceptance_repair.Repaired(
                copied.store, report, sides, repaired, summary)
        return self._repairs[name]

    def verification_of(self, name: str,
                        mode: str | None = None) -> tuple[Mapping | None, str | None]:
        """What ``verify`` prints for the repair of the store ``name`` (see ``_repaired``)."""
        repaired = self._repaired(name)
        arguments = ["verify", "--report", str(repaired.report), "--sides", str(repaired.sides)]
        if mode is not None:
            arguments += ["--mode", mode]
        return self._document(repaired.store, arguments, rep.VERIFICATION_FORM, "verification")

    def _listing(self, question: Question) -> list[Mismatch]:
        """The listing of a request: a refusal or loud stop, or what it covers and
        what it lists as not covered (docs/cli.md, the listing)."""
        store = self.copy(question.store, question.state)
        outcome = self.cli(store.store, *self.query_arguments(question))
        expect = question.expect
        if not set(expect) - table.STOP_KEYS:
            return table.compare_stop(expect, outcome)
        if set(expect) - table.LISTING_KEYS:
            raise WorldError(f"row {question.row.id}: a listing is compared by its refusal or "
                             f"by what it covers and does not cover; the row expects "
                             f"{sorted(expect)}")
        if outcome.exit_code != 0:
            return [Mismatch("listing", "a listing", f"no listing: exit code {outcome.exit_code}, "
                             f"error {outcome.error_name()!r}")]
        return table.compare_listing(expect, outcome.printed())

    def _attribution(self, question: Question) -> list[Mismatch]:
        """Which copy a release row is attributed to: the edge of the row, as
        ``show --record`` prints it."""
        store = self.store_of(question)
        row = RecordRef(question.artifact, question.record)
        edge = self.shown(store.store, question.artifact, "--record",
                          self.locator(question.store, row)).get("edge")
        if not isinstance(edge, Mapping):
            return [Mismatch("edge", "an edge of the row", edge)]

        def named(parent: object) -> str | None:
            if not isinstance(parent, Mapping):
                return None
            return f"{parent['artifact']}:{parent['record']}"

        def wanted(text: str) -> str:
            ref = table.record_ref(text)
            return f"{ref.artifact}:{self.locator(question.store, ref)}"

        out = []
        expect = question.expect
        selected = named(edge.get("selected_parent"))
        if "selected_parent" in expect and selected != wanted(expect["selected_parent"]):
            out.append(Mismatch("selected_parent", wanted(expect["selected_parent"]), selected))
        if "parent_mode" in expect and edge.get("parent_mode") != expect["parent_mode"]:
            out.append(Mismatch("parent_mode", expect["parent_mode"], edge.get("parent_mode")))
        if "loser" in expect:
            # The loser of the dedup group: the equivalent parent the step did not keep.
            others = sorted(p for p in map(named, edge.get("parents", [])) if p != selected)
            if others != [wanted(expect["loser"])]:
                out.append(Mismatch("loser", [wanted(expect["loser"])], others))
        return out

    def _instances(self, question: Question) -> list[Mismatch]:
        """Two artifacts compared by what ``show`` prints about each."""
        store = self.copy(question.store, question.state)
        first, second = (self.shown(store.store, name) for name in question.artifacts)

        def same(key: str) -> bool:
            if key not in first or key not in second:
                raise WorldError(f"show prints no {key!r} for {question.artifacts}")
            return first[key] == second[key]

        def key_of(name: str) -> str:
            # An occurrence key is the instance id plus a locator (plan 3.1).
            locator = self.locator(question.store, RecordRef(name, question.record))
            record = self.shown(store.store, name, "--record", locator)
            return f"{record['instance']}:{record['record']}"

        def inputs(shown: Mapping) -> list[str]:
            return [entry["instance"] for entry in shown["inputs"]]

        readings = {
            "content_digests": lambda: same("content_digest"),
            "instance_ids": lambda: same("instance"),
            "instance_id": lambda: same("instance"),
            "producing_step": lambda: same("step"),
            "input_instance_ids": lambda: inputs(first) == inputs(second),
            "edge_digest": lambda: same("edge_digest"),
            "occurrence_keys_of_X": lambda: (
                key_of(question.artifacts[0]) == key_of(question.artifacts[1])),
        }
        words = {"equal": True, "different": False, "changes": False}
        out = []
        for name, word in question.expect.items():
            if word not in words or name not in readings:
                raise WorldError(f"row {question.row.id}: no way to read {name} = {word!r}")
            equal = readings[name]()
            if equal != words[word]:
                out.append(Mismatch(name, word, "equal" if equal else "different"))
        if table.NEED_SHOW_EDGE_DIGEST in NOT_BUILT and "edge_digest" in first:
            raise WorldError("show prints an edge digest now: remove its entry from NOT_BUILT, "
                             "and the field is compared")
        return out

    def _census(self, store_file: Path, names) -> tuple[int, int]:
        """(registered artifacts, stored record rows) of a store, from ``check``
        and ``show``."""
        check = self.cli(store_file, "check")
        if check.exit_code != 0:
            raise WorldError(f"check ended with exit code {check.exit_code}: {check.stderr[:300]}")
        rows = 0
        for name in names:
            shown = self.shown(store_file, name, again=True)
            if shown["kind"] == "mixture" and "stored_rows" not in shown:
                continue  # a mixture snapshot holds no record rows, and show prints none
            rows += shown["stored_rows"]
        return check.printed()["artifacts"], rows

    def _reregister(self, question: Question) -> list[Mismatch]:
        """The same history registered again: the replica runs the store's build a
        second time into a fresh copy of the store."""
        self._scratch += 1
        copied = self._fresh_copy(question.store, f"registered-again-{self._scratch:03d}")
        name = question.artifact
        before_id = self.shown(copied.store, name, again=True)["instance"]
        before = self._census(copied.store, copied.instances)
        again = replica_fixture.register_again(copied)
        after_id = self.shown(copied.store, name, again=True)["instance"]
        after = self._census(copied.store, again.instances)
        out = []
        expect = question.expect
        if "instance_id" in expect:
            if expect["instance_id"] != "unchanged":
                raise WorldError(f"row {question.row.id}: instance_id = {expect['instance_id']!r}")
            if not (before_id == after_id == again.instances[name]):
                out.append(Mismatch("instance_id", "unchanged",
                                    f"{before_id} before, {after_id} after, "
                                    f"{again.instances[name]} returned to the second build"))
        if "new_artifacts" in expect and after[0] - before[0] != expect["new_artifacts"]:
            out.append(Mismatch("new_artifacts", expect["new_artifacts"], after[0] - before[0]))
        if "new_occurrences" in expect and after[1] - before[1] != expect["new_occurrences"]:
            out.append(Mismatch("new_occurrences", expect["new_occurrences"], after[1] - before[1]))
        return out

    def _harness(self, question: Question) -> list[Mismatch]:
        """A job run in the confined harness that asks for a record outside its
        hand-off. The job is a script of this folder (``jobs/asks_outside_record.py``)
        run in the bubblewrap sandbox; where the sandbox cannot be made the row is
        not run, by name, and never run in the process (docs/interfaces.md section
        5.3: there is no fallback)."""
        from waterwhite import confine
        from waterwhite.harness import Handed
        from waterwhite.records import OccurrenceRef

        try:
            confine.find_sandbox()
        except confine.SandboxUnavailable as unavailable:
            raise Unavailable(f"not run: the confined harness needs bwrap: {unavailable}") from None
        given, store = question.row.given, question.store
        handed = [table.record_ref(handle) for handle in given["handed"]]
        outside = table.record_ref(given["asks_for"])
        outside_id, outside_locator = self.locate(store, outside)

        log_path = self.scratch("harness") / f"{given['job']}.log.jsonl"
        result = confine.run_confined_job(
            confine.python_program(JOBS / "asks_outside_record.py", [outside_id, outside_locator]),
            job_name=given["job"], log_path=log_path, timeout_seconds=CONFINED_TIMEOUT,
            handed=[Handed(ref=OccurrenceRef(*self.locate(store, ref)),
                           text=self.text(store, ref)) for ref in handed])
        with open(log_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f.read().split("\n") if line]
        self.harness_evidence = {
            "harness": [e.get("harness") for e in entries if "harness" in e],
            "outputs": [o.text for o in result.outputs],
            "reads": [e for e in entries if e.get("kind") == "read"]}
        # The log says which harness wrote it; a log of any other harness is not
        # what this row asks about.
        if self.harness_evidence["harness"] != ["confined"]:
            raise WorldError("the log of the harness row names the harness "
                             f"{self.harness_evidence['harness']}, not 'confined'")
        # What the job was given back is what it emitted: "refused", or the text.
        (answer,) = self.harness_evidence["outputs"]
        # The log names what was asked for as text; the record is the one whose
        # instance id and locator that text holds.
        attempts = [e for e in entries if e.get("kind") == "read" and "outside_ref" in e
                    and outside_id in e["outside_ref"] and outside_locator in e["outside_ref"]]
        expect = question.expect
        got = {
            "read_refused": answer == "refused",
            "text_returned": answer.startswith("delivered:"),
            "log_shows_attempt": bool(attempts),
        }
        out = [Mismatch(name, expect[name], value) for name, value in got.items()
               if name in expect and expect[name] != value]
        if "log_entry" in expect:
            wanted = dict(expect["log_entry"])
            if (set(wanted) != {"outside_ref", "refused"}
                    or wanted["outside_ref"] != given["asks_for"]):
                raise WorldError(f"row {question.row.id}: log_entry {wanted!r}")
            if not any(e.get("refused") is wanted["refused"] for e in attempts):
                out.append(Mismatch("log_entry", wanted, attempts))
        return out


    # --- Checks 3 and 6: what show prints about maps, drops and links ----------------------

    def file_of(self, store_file: Path, artifact: str) -> Path:
        """The file of a registered artifact, at the path the store keeps for it:
        relative to the folder that holds the store file."""
        shown = self.shown(store_file, artifact)
        if not shown.get("registered") or not shown.get("path"):
            raise WorldError(f"show prints no path for {artifact!r}: {str(shown)[:200]}")
        return (store_file.parent / shown["path"]).resolve()

    def lines_of(self, path: Path) -> list[str]:
        """The lines of a file the replica wrote, read here and never by the tool."""
        if path not in self._lines:
            with open(path, encoding="utf-8", newline="\n") as f:
                lines = f.read().split("\n")
            self._lines[path] = lines[:-1] if lines and lines[-1] == "" else lines
        return self._lines[path]

    def identifier(self, store_file: Path) -> maps.Identify:
        """``identify(place)``: the id the replica's file carries on a line that
        ``show`` named. The tool is asked only where the file lies."""
        def identify(named: maps.Place) -> str | None:
            artifact, line = named
            lines = self.lines_of(self.file_of(store_file, artifact))
            if not 1 <= line <= len(lines):
                return None
            try:
                found = json.loads(lines[line - 1]).get("id")
            except (ValueError, AttributeError):
                return None
            return found if isinstance(found, str) else None
        return identify

    def map_rows(self, store_file: Path, shard: str, namespace: str,
                 derivation: str) -> list[Mapping]:
        """Every row of one named map of a shard, by ordinal."""
        if table.NEED_SHOW_MAP_ROWS not in NOT_BUILT:
            return maps.map_rows(self.shown(store_file, shard, "--map",
                                            f"{namespace}/{derivation}"))
        held = maps.binding(self.shown(store_file, shard), namespace, derivation)
        if held is None:
            raise WorldError(f"{shard} holds no map {namespace}/{derivation}")
        if held["declared_count"] > PER_DOCUMENT_LIMIT:
            raise Unavailable(f"not built yet: {table.NEED_SHOW_MAP_ROWS}")
        rows = []
        for ordinal in range(held["declared_count"]):
            printed = self.shown(store_file, shard, "--record", f"D{ordinal}")
            read = printed.get("map") or {}
            if (read.get("namespace"), read.get("derivation")) != (namespace, derivation):
                raise Unavailable(
                    f"not built yet: {table.NEED_SHOW_MAP_ROWS} (show --record reads the map "
                    f"{read.get('namespace')}/{read.get('derivation')} of {shard} first)")
            rows.append({key: printed.get(key) for key in ROW_KEYS})
        return maps.map_rows({"rows": rows})

    def drops_of(self, store_file: Path, artifact: str) -> list[Mapping]:
        """The drops recorded with an artifact as their output."""
        shown = self.shown(store_file, artifact)
        if shown.get("registered") is False:
            return []  # a name the store has no row for has no drops recorded under it
        drops = shown.get("drops")
        if not isinstance(drops, list):
            raise maps.OutputForm(f"show {artifact} prints no list of drops")
        return drops

    def _log_places(self, log: Sequence[Mapping]) -> dict[tuple[str, int], Mapping]:
        """The two places the replica's sanitizer log gives a skipped line: the
        line of the sanitizer's input and the line the pipeline listed it from,
        each as (path relative to the build folder, line)."""
        places = {}
        for row in log:
            places[(row["input"], row["line"])] = row
            if row.get("origin_file") is not None:
                places[(row["origin_file"], row["origin_line"])] = row
        return places

    def _printed_drops(self, store_file: Path, out: Path, artifacts: Sequence[str],
                       log: Sequence[Mapping]) -> list[tuple[Mapping | None, str, str]]:
        """The drops recorded under ``artifacts``: (the row of the sanitizer's log
        for the dropped line or None, the dropped line as printed, the reason)."""
        places = self._log_places(log)
        printed = []
        for artifact in artifacts:
            for drop in self.drops_of(store_file, artifact):
                name, line = maps.place(drop)
                path = self.file_of(store_file, name)
                try:
                    relative = path.relative_to(Path(out).resolve()).as_posix()
                except ValueError:
                    relative = None
                printed.append((places.get((relative, line)), f"{name}:L{line}",
                                drop.get("reason")))
        return printed

    def _spec_item(self, question: Question) -> list[Mismatch]:
        """An item the fixture spec must hold before a check runs. The row's
        ``value`` is the spec's own item; what is checked is that the spec records
        one: a yes or a no, and with a yes the token."""
        expect = question.expect
        if expect["spec_key"] != "tokenizer.eod_in_span":
            raise WorldError(f"row {question.row.id}: no reading of the spec item "
                             f"{expect['spec_key']!r} is written")
        tokenizer = acceptance_sets.tokenizer_table()
        recorded = tokenizer.get("eod_in_span")
        if not isinstance(recorded, bool):
            return [Mismatch("value", "true or false, recorded in the spec", recorded)]
        if recorded != expect["value"]:
            return [Mismatch("value", expect["value"], recorded)]
        if recorded and not isinstance(tokenizer.get("eod_token_id"), int):
            return [Mismatch("value", "the end-of-document token beside the convention",
                             tokenizer.get("eod_token_id"))]
        return []

    def _pair_kinds(self, question: Question, size: str) -> list[Mismatch]:
        """Counted from the plant list and the chunk-file ids alone: the tool is
        not asked."""
        _, counted = self.sets.chunked(size)
        planted = counted["check3"]["pairs"]["counts"]["pair-member"]
        out = []
        for name, placement in (("same_shard_pairs_at_least", "same-shard"),
                                ("split_pairs_at_least", "split")):
            if planted[placement] < question.expect[name]:
                out.append(Mismatch(name, question.expect[name], planted[placement]))
        return out

    def _set(self, question: Question, size: str) -> tuple[SetBuild, Path]:
        """The build of the generated set a row is asked of, and the store file in
        the state the row names."""
        captured = question.store != acceptance_sets.NO_CAPTURE
        build = self.sets.build(size, acceptance_sets.CAPTURE if captured
                                else acceptance_sets.NO_CAPTURE)
        build.need_shards()
        if question.state != "unmutated":
            return build, self.sets.state(build, question.state)
        if captured and any(namespace == "recovery" for namespace, _ in question.maps):
            return build, self.sets.state(build, acceptance_sets.STATE_RECOVERED)
        return build, build.store

    def _all_rows(self, store_file: Path, build: SetBuild, held: tuple[str, str]) -> dict:
        return {shard: self.map_rows(store_file, name, *held)
                for name, shard in build.shards.items()}

    def _spans(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        rows = self._all_rows(store_file, build, question.maps[0])
        sizes = {shard: self.sets.sizes(build, shard) for shard in rows}
        found = (maps.span_findings if question.kind == table.KIND_SPANS
                 else maps.total_findings)(rows, sizes)
        return maps.count_field("mismatches", question.expect["mismatches"], found)

    def _sanitizer_drops(self, build: SetBuild, store_file: Path
                         ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        """(printed, expected, reasons printed) for the lines the sanitizer skipped:
        each drop as (the id of the dropped line, its reason). The expected ones
        are the counting script's, placed in their shards from the sanitizer-input
        files; the replica's sanitizer log must say the same."""
        log = build.sanitizer_log()
        counted = [drop for shard in build.counted["check3"]["drops"]["shards"]
                   for drop in shard["drops"]]
        logged = sorted((row["shard"], row["line"], row["id"], row["reason"]) for row in log)
        if logged != sorted((d["shard"], d["input_line"], d["id"], d["reason"]) for d in counted):
            raise WorldError("the replica's sanitizer log and the counting script place the "
                             "skipped lines differently; no drop can be graded on this build")
        artifacts = [name for shard in build.shards for name in (shard, f"{shard}.input")]
        printed = [(row["id"] if row is not None else where, reason)
                   for row, where, reason in self._printed_drops(store_file, build.out,
                                                                 artifacts, log)]
        of_sanitizer = [drop for drop in printed if drop[1] != maps.NO_TOKENS]
        return of_sanitizer, [(d["id"], d["reason"]) for d in counted], [r for _, r in printed]

    def _set_drops(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        expect = question.expect
        printed, wanted, reasons = self._sanitizer_drops(build, store_file)
        out = []
        found = maps.drop_findings(printed, wanted)
        if found:
            out.append(Mismatch("drops", f"{len(wanted)} drops, one per line the sanitizer "
                                         "skipped", f"{len(found)} findings; first: {found[:3]}"))
        if {reason for _, reason in printed} != set(expect["reasons"]):
            out.append(Mismatch("reasons", sorted(expect["reasons"]),
                                sorted({reason for _, reason in printed})))
        if "documents_per_shard" in expect:
            held = question.maps[0]
            wrong = []
            for name, shard in build.shards.items():
                bound = maps.binding(self.shown(store_file, name), *held)
                lines = build.counted_shard("documents", shard)["documents"]
                if bound is None or bound.get("declared_count") != lines:
                    wrong.append(f"{shard}: {bound and bound.get('declared_count')} documents, "
                                 f"{lines} surviving lines")
            if wrong:
                out.append(Mismatch("documents_per_shard", expect["documents_per_shard"],
                                    wrong[:3]))
        if "drops_with_reason_no_tokens" in expect:
            out += maps.count_field("drops_with_reason_no_tokens",
                                    expect["drops_with_reason_no_tokens"],
                                    [r for r in reasons if r == maps.NO_TOKENS])
        if "ordinals_shifted" in expect:
            shifted = []
            identify = self.identifier(store_file)
            for name, shard in build.shards.items():
                if build.counted_shard("drops", shard)["drops"]:
                    entry = build.counted_shard("documents", shard)
                    shifted += maps.record_findings(
                        self.map_rows(store_file, name, *question.maps[0]), entry["ids"],
                        [group["ordinals"] for group in entry["identical"]], identify, shard,
                        twins_as_set=False)
            out += maps.count_field("ordinals_shifted", expect["ordinals_shifted"], shifted)
        return out

    def _set_no_tokens(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        expect = question.expect
        _, _, reasons = self._sanitizer_drops(build, store_file)
        out = maps.count_field("drops_with_reason_no_tokens",
                               expect["drops_with_reason_no_tokens"],
                               [r for r in reasons if r == maps.NO_TOKENS])
        settings = [(maps.binding(self.shown(store_file, name), *question.maps[0]) or {})
                    .get("append_eod") for name in build.shards]
        recorded = all(isinstance(value, bool) for value in settings)
        if recorded != expect["append_eod_recorded"]:
            out.append(Mismatch("append_eod_recorded", expect["append_eod_recorded"], settings[:3]))
        elif recorded and any(value != expect["append_eod"] for value in settings):
            out.append(Mismatch("append_eod", expect["append_eod"], sorted(set(settings))))
        return out

    def _planted_line(self, question: Question) -> list[Mismatch]:
        """The scratch shard whose third input line gives no tokens: each map the
        row names must give the next ordinal to the next line that has tokens."""
        store_file = self.copy(question.store, question.state).store
        shard, expect = question.artifact, question.expect
        input_name = f"{shard}.input"
        shown, identify = self.shown(store_file, shard), self.identifier(store_file)
        out = []

        def differ(name: str, wanted: object, got: object) -> None:
            if wanted != got:
                out.append(Mismatch(name, wanted, got))

        differ("input_lines", expect["input_lines"],
               self.shown(store_file, input_name).get("stored_rows"))
        drops = sorted((maps.place(drop), drop.get("reason"))
                       for name in (shard, input_name) for drop in self.drops_of(store_file, name))
        differ("drops", sorted(((input_name, d["line"]), d["reason"]) for d in expect["drops"]),
               drops)
        differ("drops with the reason no tokens", len(expect["drops"]),
               sum(1 for _, reason in drops if reason == maps.NO_TOKENS))
        out += self._planted_line_gives_no_tokens(store_file, shard, input_name, expect)
        for held in question.maps:
            where = "/".join(held)
            bound = maps.binding(shown, *held) or {}
            rows = self.map_rows(store_file, shard, *held)
            differ(f"documents ({where})", expect["documents"], bound.get("declared_count"))
            differ(f"verification ({where})", expect["verification"], bound.get("verification"))
            differ(f"label ({where})", [expect["label"]],
                   sorted({bound.get("label"), *(row["label"] for row in rows)}))
            differ(f"append_eod_recorded ({where})", expect["append_eod_recorded"],
                   isinstance(bound.get("append_eod"), bool))
            parents = [maps.place(row["parent"]) if row["parent"] else None for row in rows]
            differ(f"ordinal_of_line ({where})", dict(expect["ordinal_of_line"]),
                   {str(named[1]): ordinal for ordinal, named in enumerate(parents)
                    if named is not None and named[0] == input_name})
            differ(f"records ({where})", list(expect["records"]),
                   [_table_handle(identify(named)) if named else None for named in parents])
        return out

    def _planted_line_gives_no_tokens(self, store_file: Path, shard: str, input_name: str,
                                      expect: Mapping) -> list[Mismatch]:
        """The planted line is shown to give no tokens by a real run of the team's
        tokenizer, not taken from the row: Megatron's own tokenizer is run on every
        line of the input the replica registered, and the shard that Megatron's
        preprocessing wrote from that input is read with the replica's reader of
        its index. The line gives no tokens when the tokenizer's answer is empty and
        the index holds no tokens at its place; every other line gives tokens, and
        the index holds as many as the tokenizer gave, with the end-of-document
        token. A planted line that gives tokens would make every comparison of the
        row a comparison about a line that is not the case it names."""
        from replica.shard_index import read_shard_index

        tokenize, _ = self.sets.tokenizer()
        append_eod = acceptance_sets.tokenizer_table()["append_eod"]
        lines = self.lines_of(self.file_of(store_file, input_name))
        given = [tokenize(json.loads(line)["text"]) for line in lines]
        index = read_shard_index(Path(f"{self.file_of(store_file, shard)}.idx")).document_tokens
        planted = sorted(drop["line"] for drop in expect["drops"])
        return maps.no_token_findings(given, index, planted, append_eod)

    def _records(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        identify = self.identifier(store_file)
        found = []
        for name, shard in build.shards.items():
            self.sets.check_planted_tokens(build, shard)
            entry = build.counted_shard("documents", shard)
            identical = [group["ordinals"] for group in entry["identical"]]
            rows = self.map_rows(store_file, name, *question.maps[0])
            if question.kind == table.KIND_RECORDS_AS_CAPTURE:
                capture = self.map_rows(store_file, name, *table.MAP_NAMES["capture"])
                found += maps.same_as_findings(rows, capture, identical, shard)
            else:
                found += maps.record_findings(rows, entry["ids"], identical, identify, shard,
                                              twins_as_set=False)
        return maps.count_field("mismatches", question.expect["mismatches"], found)

    def _set_documents(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        identify = self.identifier(store_file)
        found = []
        for name, shard in build.shards.items():
            entry = build.counted_shard("documents", shard)
            found += maps.record_findings(
                self.map_rows(store_file, name, *question.maps[0]), entry["ids"],
                [group["ordinals"] for group in entry["identical"]], identify, shard,
                twins_as_set=True)
        return maps.count_field("unmapped_documents", question.expect["unmapped_documents"], found)

    def _twins(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        swapped = question.kind == table.KIND_TWINS_SWAPPED
        by_shard = acceptance_sets.same_shard_pairs(build.counted)
        if not by_shard:
            raise WorldError("no pair of this set has both members in one shard; the row has "
                             "nothing to be asked of")
        identify = self.identifier(store_file) if swapped else None
        labels, sets = [], []
        for name, shard in build.shards.items():
            input_name = f"{name}.input.swapped" if swapped else f"{name}.input"
            for held in question.maps if shard in by_shard else ():
                rows = self.map_rows(store_file, name, *held)
                for pair in by_shard[shard]:
                    wrong_label, wrong_set = maps.twin_findings(
                        rows, pair, input_name, shard, identify, label=question.expect["label"])
                    labels += [f"{'/'.join(held)} {text}" for text in wrong_label]
                    sets += [f"{'/'.join(held)} {text}" for text in wrong_set]
        expect = question.expect
        out = []
        if labels:
            out.append(Mismatch("label", expect["label"], f"{len(labels)}; first: {labels[:3]}"))
        if sets:
            out.append(Mismatch("candidate_set", expect["candidate_set"],
                                f"{len(sets)}; first: {sets[:3]}"))
        if "other_labels_for_these_ordinals" in expect:
            out += maps.count_field("other_labels_for_these_ordinals",
                                    expect["other_labels_for_these_ordinals"], labels)
        return out

    def _split_pairs(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        names = {shard: name for name, shard in build.shards.items()}
        pairs = acceptance_sets.split_pairs(build.counted)
        if not pairs:
            raise WorldError("no pair of this set is split across two shards")
        found = []
        for held in question.maps:
            for pair in pairs:
                for member in pair["members"]:
                    name = names[member["shard"]]
                    found += [f"{'/'.join(held)} {text}" for text in maps.split_findings(
                        self.map_rows(store_file, name, *held), member, f"{name}.input",
                        pair["pair"], label=question.expect["label"])]
        if found:
            return [Mismatch("record", f"{question.expect['label']}: "
                             f"{question.expect['record']}", f"{len(found)}; first: {found[:3]}")]
        return []

    def _verification_failed(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        name = next(iter(build.shards))
        shown, expect = self.shown(store_file, name), question.expect
        out = []
        for held in question.maps:
            where = "/".join(held)
            bound = maps.binding(shown, *held)
            if bound is None:
                out.append(Mismatch(f"verification ({where})", expect["verification"],
                                    f"{name} holds no such map"))
                continue
            for key in ("verification", "label"):
                if bound.get(key) != expect[key]:
                    out.append(Mismatch(f"{key} ({where})", expect[key], bound.get(key)))
            if "labels_of_documents" in expect:
                labels = sorted(maps.labels_of(self.map_rows(store_file, name, *held)))
                if labels != sorted(expect["labels_of_documents"]):
                    out.append(Mismatch(f"labels_of_documents ({where})",
                                        sorted(expect["labels_of_documents"]), labels))
        return out

    def _map_fields(self, question: Question, size: str) -> list[Mismatch]:
        build, store_file = self._set(question, size)
        expect = question.expect
        fields = {"capture": "capture", "position_recovery": "position recovery",
                  "token_matching_recovery": "token-matching recovery"}
        out, labels = [], set()
        for name, shard in build.shards.items():
            shown = self.shown(store_file, name)
            for key, named in fields.items():
                out += maps.compare_binding(f"{key} ({shard})", expect[key], shown,
                                            *table.MAP_NAMES[named])
            if len(shown.get("bindings", ())) != expect["maps_per_shard"]:
                out.append(Mismatch(f"maps_per_shard ({shard})", expect["maps_per_shard"],
                                    len(shown.get("bindings", ()))))
            if "labels_of_documents" in expect:
                for held in question.maps:
                    labels |= set(maps.labels_of(self.map_rows(store_file, name, *held)))
        if "labels_of_documents" in expect and labels != set(expect["labels_of_documents"]):
            out.append(Mismatch("labels_of_documents", sorted(expect["labels_of_documents"]),
                                sorted(labels)))
        return out[:12]

    def _many(self, store_file: Path, commands: Sequence[Sequence[str]]) -> list[Outcome]:
        """Many small command lines on one store, a few at a time."""
        with ThreadPoolExecutor(SHOW_WORKERS) as pool:
            return list(pool.map(lambda arguments: self.cli(store_file, *arguments), commands))

    def _set_candidates(self, question: Question, size: str) -> list[Mismatch]:
        """Every row whose identical text reached a release from more than one
        source: the label and the sources of its link, and the answer the release
        gives for the candidate records of each of those sources."""
        build, store_file = self._set(question, size)
        expect = question.expect
        entries = build.counted["check6"]["candidates"]
        if not entries:
            raise WorldError("the counting script lists no row with candidates in this set")
        asked, of_source = [], {}
        for entry in entries:
            manifest = {row["handle"]: row["line"] for row in acceptance_sets.read_rows(
                Path(f"{self.file_of(store_file, entry['release'])}.manifest.jsonl"))}
            for record in entry["rows"]:
                asked.append((entry, record, ("show", entry["release"], "--record",
                                              f"L{manifest[record]}")))
            for candidate in entry["candidates"]:
                line = int(candidate["id"].rsplit("-", 1)[1])  # an id names a line
                of_source.setdefault((entry["release"], candidate["source"]), []).append(
                    f"src/{candidate['source']}:L{line}")
        labels, sources = [], []
        for (entry, record, _), outcome in zip(asked, self._many(store_file,
                                                                 [a[2] for a in asked])):
            edge = outcome.printed().get("edge") if outcome.exit_code == 0 else None
            where = f"{entry['release']} row {record}"
            if not isinstance(edge, Mapping) or edge.get("label") != expect["label"]:
                labels.append(f"{where}: {edge and edge.get('label')}")
            elif sorted(edge.get("candidate_sources", ())) != sorted(entry["sources"]):
                sources.append(f"{where}: {edge.get('candidate_sources')}, the plant list "
                               f"names {entry['sources']}")
        out = []
        if labels:
            out.append(Mismatch("label", expect["label"], f"{len(labels)}; first: {labels[:3]}"))
        if sources:
            out.append(Mismatch("candidate_sources", expect["candidate_sources"],
                                f"{len(sources)}; first: {sources[:3]}"))
        for (release, source), records in sorted(of_source.items()):
            arguments = ["query", "withdraw", source]
            for record in records:
                arguments += ["--record", record]
            outcome = self.cli(store_file, *arguments, "--artifact", release)
            answer = outcome.printed().get("answer") if outcome.exit_code == 0 else (
                f"no answer: exit code {outcome.exit_code}, {outcome.error_name()!r}")
            if answer != expect["answer_for_each_candidate_source"]:
                out.append(Mismatch(f"answer_for_each_candidate_source ({release}, {source})",
                                    expect["answer_for_each_candidate_source"], answer))
        return out

    def _set_downstream(self, question: Question, size: str) -> list[Mismatch]:
        """Everything that holds a planted translation: the file its job wrote, the
        releases and the shards that receive one, each asked for its answer."""
        build, store_file = self._set(question, size)
        part = build.counted["check6"]["translations"]
        if not part["count"]:
            raise WorldError("the set plants no translation")
        names = {shard: name for name, shard in build.shards.items()}
        asked = [f"gen/{part['job']}"]
        asked += [release for release, rows in part["releases"].items() if rows]
        asked += [names[entry["shard"]] for entry in part["shards"]]
        out = []
        arguments = ["query", "withdraw", question.request.source]
        for artifact in asked:
            outcome = self.cli(store_file, *arguments, "--artifact", artifact)
            out += [Mismatch(f"{m.field} ({artifact})", m.expected, m.got)
                    for m in table.compare_outcome(question.expect, outcome, _no_handles)]
        return out

    def _documents(self, question: Question) -> list[Mismatch]:
        """One shard of a store of the table: how many documents its map holds, how
        many of them with each label, and the map's stored fields."""
        store_file = self.copy(question.store, question.state).store
        (held,) = question.maps
        expect = question.expect
        bound = maps.binding(self.shown(store_file, question.artifact), *held)
        if bound is None:
            return [Mismatch("derivation", expect["derivation"],
                             f"{question.artifact} holds no map {'/'.join(held)}")]
        labels = maps.labels_of(self.map_rows(store_file, question.artifact, *held))
        got = {"documents": bound.get("declared_count"), "recovered": labels[maps.RECOVERED],
               "candidates": labels[maps.CANDIDATES], "unknown": labels[maps.UNKNOWN],
               "derivation": bound.get("derivation"), "verification": bound.get("verification")}
        return [Mismatch(name, expect[name], got[name]) for name in expect
                if expect[name] != got[name]]

    def _drops(self, question: Question) -> list[Mismatch]:
        """Every drop of a store of the table with a reason of the sanitizer or of
        the tokenizer, each recorded once."""
        built = self.copy(question.store, question.state)
        expect = question.expect
        log = acceptance_sets.read_rows(
            built.out / "pipeline" / "partitioned" / "sanitizer.log.jsonl")
        printed = self._printed_drops(built.store, built.out, sorted(built.instances), log)
        # ``input_of`` names a shard; the dropped line is a line of that shard's
        # sanitizer input, which the replica's log names for the line.
        wanted = [(f"{drop['input_of']}:L{drop['line']}", drop["reason"])
                  for drop in expect["drops"]]
        got = [(f"{FIXTURE_SHARDS}{row['shard']}:L{row['line']}" if row is not None else where,
                reason)
               for row, where, reason in printed
               if reason in (*maps.SANITIZER_REASONS, maps.NO_TOKENS)]
        out = []
        found = maps.drop_findings(got, wanted)
        if found:
            out.append(Mismatch("drops", wanted, found))
        out += maps.count_field("empty_line_drops", expect["empty_line_drops"],
                                [g[0] for g in got if g[1] == "empty line"])
        out += maps.count_field("no_tokens_drops", expect["no_tokens_drops"],
                                [g[0] for g in got if g[1] == maps.NO_TOKENS])
        return out

    def _record_shown(self, question: Question) -> Mapping:
        store_file = self.copy(question.store, question.state).store
        row = RecordRef(question.artifact, question.record)
        return self.shown(store_file, question.artifact, "--record",
                          self.locator(question.store, row))

    def _edge_label(self, question: Question) -> list[Mismatch]:
        """How one row of a file is linked: the label of its edge and the sources
        of its candidates."""
        edge, expect = self._record_shown(question).get("edge"), question.expect
        if not isinstance(edge, Mapping):
            return [Mismatch("label", expect["label"], "the row has no edge")]
        out = []
        if edge.get("label") != expect["label"]:
            out.append(Mismatch("label", expect["label"], edge.get("label")))
        if sorted(edge.get("candidate_sources", ())) != sorted(expect["candidates"]):
            out.append(Mismatch("candidates", sorted(expect["candidates"]),
                                edge.get("candidate_sources")))
        return out

    def _file_family(self, question: Question) -> list[Mismatch]:
        """The file family of an import: a link of the file as a whole."""
        store_file = self.copy(question.store, question.state).store
        wanted = dict(question.expect["link"])
        links = self.shown(store_file, question.artifact).get("source_links", [])
        whole = [link for link in links if link.get("record") is None]
        if any(all(link.get(key) == value for key, value in wanted.items()) for link in whole):
            return []
        return [Mismatch("link", wanted, links)]


    # --- Check 4: writes to the scratch store -----------------------------------------------

    def _scratch_write(self, question: Question) -> list[Mismatch]:
        """A write to the map of the scratch store's shard, on a fresh copy of the store:
        the writer killed after 100 rows, the write retried after that, or a wrong count
        declared. What the store holds afterwards is read through ``show`` and put in
        the table's own sentences (``acceptance_scratch.held``)."""
        state, expect = question.state, question.expect
        wanted = {sentence for key in ("one_of", "never") for sentence in expect.get(key, ())}
        wanted |= {expect["map_after"]} if "map_after" in expect else set()
        unknown = wanted - acceptance_scratch.SENTENCES
        if unknown:
            raise WorldError(f"row {question.row.id}: the sentences {sorted(unknown)} are none "
                             "the scratch comparison was written for")
        self._scratch += 1
        copied = self._fresh_copy(question.store, f"{state}-{self._scratch:03d}")
        refused, error = None, None
        if state == "w.kill-after-100":
            acceptance_scratch.kill_writer(copied.store, 100)
        elif state == "w.retry":
            acceptance_scratch.kill_writer(copied.store, 100)
            done = acceptance_scratch.write_complete_map(copied.store)
            refused = acceptance_scratch.refusal(done)
            if done.returncode != 0 and refused not in acceptance_scratch.NAMED_MISMATCHES:
                raise WorldError(f"the retried write ended with exit code {done.returncode}: "
                                 f"{done.stderr[-400:]}")
        elif state == "w.declare-499":
            error = acceptance_scratch.declare_wrong_count(copied.store, 499)
        else:
            raise WorldError(f"row {question.row.id}: no write is written for the state {state}")
        shown = self.shown(copied.store, acceptance_scratch.SHARD, again=True)
        sentences = acceptance_scratch.held(shown, refused)
        out = []
        if "one_of" in expect and not sentences & set(expect["one_of"]):
            out.append(Mismatch("one_of", expect["one_of"], sorted(sentences)))
        if "never" in expect and sentences & set(expect["never"]):
            out.append(Mismatch("never", f"none of {expect['never']}",
                                sorted(sentences & set(expect["never"]))))
        if "error" in expect and error != expect["error"]:
            out.append(Mismatch("error", expect["error"], error))
        if "map_after" in expect and expect["map_after"] not in sentences:
            out.append(Mismatch("map_after", expect["map_after"], sorted(sentences)))
        if "rows_after" in expect and (shown.get("stored_rows") or 0) != expect["rows_after"]:
            out.append(Mismatch("rows_after", expect["rows_after"], shown.get("stored_rows")))
        return out

    # --- Checks 1, 2 and 8: the runs ------------------------------------------------------

    def _registration(self, question: Question) -> list[Mismatch]:
        """What ``register-run`` printed for the segments of a starting state of Check
        2: the route it took, the named cause it gave and, where it prints one, the
        replay it tried (``replay``: attempted, accepted, reason); and the arrays
        ``replay`` derives for a segment (``replayed_arrays``, ``digests``)."""
        given, expect = question.row.given, question.expect
        names = list(given.get("segments") or [given["segment"]])
        printed = self.registered(question.store, question.state)
        out, no_replay_field = [], []
        for name in names:
            got = printed[name]
            where = f" ({name})"
            if "replay" in expect:
                # A replay that was tried and rejected, or none tried, is a segment with
                # no trace route and a named cause (docs/interfaces.md: route "none").
                if expect["replay"] not in ("rejected", "not attempted"):
                    raise WorldError(f"row {question.row.id}: replay = {expect['replay']!r}")
                if (got.get("route"), got.get("cause")) != ("none", "data-order trace missing"):
                    out.append(Mismatch("replay" + where, expect["replay"],
                                        f"route {got.get('route')!r}, cause {got.get('cause')!r}"))
                if "replay" not in got:
                    no_replay_field.append(name)
                else:
                    wanted = ({"attempted": True, "accepted": False} if expect["replay"] ==
                              "rejected" else {"attempted": False, "accepted": None})
                    wanted["reason"] = expect.get("reason")
                    if got["replay"] != wanted:
                        out.append(Mismatch("replay object" + where, wanted, got["replay"]))
            if "reason" in expect and got.get("cause_detail") != expect["reason"]:
                out.append(Mismatch("reason" + where, expect["reason"], got.get("cause_detail")))
        if "replayed_arrays" in expect:
            out += self._replayed_arrays(question, names)
        if "digests" in expect:
            out += self._replayed_digests(question, names)
        if no_replay_field:
            # register-run prints the replay object (docs/cli.md): its absence is a failure
            # of the row, never a skip.
            out.append(Mismatch("register-run's replay object (attempted, accepted, reason)",
                                "printed", "absent"))
        return out

    def _segment_plan(self, store: str, name: str):
        return next(s for s in acceptance_runs.segments_of(
            acceptance_runs.spec(), acceptance_runs.RUNS_OF_STORE[store]) if s.name == name)

    def _record_path(self, store: str, state: str, name: str) -> Path:
        """The run record a starting state of Check 2 handed to register-run for a segment:
        the replica's own file, or the changed one of the state."""
        plan = self._segment_plan(store, name)
        files = acceptance_runs.files_of(self.store(store).store.parent / "runs", plan)
        changed = self._records.get((store, state))
        if changed is not None:
            path = Path(changed) / plan.run / files.run_record.name
            if path.is_file():
                return path
        return files.run_record

    def _replay(self, store: str, state: str, name: str) -> tuple[dict, Path]:
        """``waterwhite replay`` over the run record a state handed over, into a folder of
        its own. Returns what it printed and the folder."""
        record = self._record_path(store, state, name)
        out = self.scratch(f"replay-{state}-{name.replace('/', '-')}")
        return acceptance_runs.replay(record, out, acceptance_sets.need_megatron()), out

    def _replayed_arrays(self, question: Question, names: Sequence[str]) -> list[Mismatch]:
        """Check 2 (i): each array ``replay`` wrote, compared with the kept cache file of
        the same kind and shard (the replica's cache folder of the segment, which the
        state withheld from register-run)."""
        expect, store, state = question.expect, question.store, question.state
        kinds = expect.get("kinds") or (acceptance_trace.PER_SHARD_KINDS
                                        + acceptance_trace.BLEND_KINDS)
        out = []
        for name in names:
            printed, folder = self._replay(store, state, name)
            files = acceptance_runs.files_of(self.store(store).store.parent / "runs",
                                             self._segment_plan(store, name))
            with open(files.run_record, encoding="utf-8") as f:
                kept = acceptance_trace.cache_files(files.cache, json.load(f))
            found = acceptance_trace.compare_replay(printed, folder, kept, kinds)
            if found:
                out.append(Mismatch(f"replayed_arrays ({name})", "byte-identical",
                                    f"{len(found)} findings; first: {found[:3]}"))
        return out

    def _replayed_digests(self, question: Question, names: Sequence[str]) -> list[Mismatch]:
        """Check 2 (ii): the digests of the arrays ``replay`` derives from the changed
        record are not the digests recorded at training time. At least one array must
        differ from its recorded digest, and ``equal`` must say what the two digests say."""
        if question.expect["digests"] != "differ from the recorded ones":
            raise WorldError(f"row {question.row.id}: digests = {question.expect['digests']!r}")
        out = []
        for name in names:
            printed, _ = self._replay(question.store, question.state, name)
            arrays = printed["arrays"]
            differing = [a for a in arrays if a["digest"] != a["recorded_digest"]]
            wrong = [a for a in arrays if a["equal"] is not None
                     and (a["digest"] == a["recorded_digest"]) != a["equal"]]
            if not differing:
                out.append(Mismatch(f"digests ({name})", "differ from the recorded ones",
                                    "every replayed array has its recorded digest"))
            if wrong:
                out.append(Mismatch(f"digests ({name})", "equal agrees with the two digests",
                                    f"{wrong[0]['kind']} says equal {wrong[0]['equal']}"))
        return out

    def oracle_runs(self, store: str) -> Path:
        """The oracle's own runs over the build of a store (the oracle's fixture-runs
        script, run once), the grading side of Check 1."""
        built = self.store(store)
        key = ("oracle", store)
        if key not in self._registered:
            root = acceptance_sets.need_megatron()
            self._registered[key] = acceptance_runs.oracle_runs(
                built.out, self.base / "oracle-runs" / store, root,
                acceptance_runs.RUNS_OF_STORE[store])
        return self._registered[key]

    def trace(self, store: str, segment: str) -> Mapping:
        """``show SEGMENT --trace`` of a registered segment, as the command line prints it."""
        return self.shown(self.store(store).store, segment, "--trace")

    def trace_rows(self, store: str, segment: str) -> list[Mapping]:
        """The rows the join reported for a segment, in the shape of the oracle's log."""
        return acceptance_trace.flatten(self.trace(store, segment))

    # Fields of a row of Check 1 that describe what is compared in words.
    PROSE_KEYS = frozenset({"compared", "judged_against", "rule"})
    # The fields of a row of Checks 1 and 8 that ``_join_rows`` compares.
    JOIN_KEYS = frozenset({
        "diff", "rows", "row", "rows_in_sample", "rows_of_T_in_run_E", "rows_of_W_in_steps_1_to_9",
        "sample_tokens", "covered_tokens", "uncovered_in_sample", "starts_at", "consumed_samples",
        "loud_stop", "steps_with_uncovered_range", "steps_without_trace", "samples_per_step",
        "samples_per_rank", "ranks", "sample_pos_of_rank_0", "sample_pos_of_rank_1", "first_step",
        "last_step", "rows_that_name_T", "first_exposure", "documents_in_shard",
        "consumed_at_start", "skipped_line", "boundary_after_step"})
    # The forms a row of Check 1 reads that a command may not print yet: a row that needs one
    # is skipped with its name, after every other field of the row was compared.
    NO_LAUNCHES = "show --trace launches"
    NO_EPOCHS = "show --trace epochs"
    NO_SHUFFLE = "register-run's shuffle"
    # The epoch a launch reads first (the row c01.K.resume: "inside epoch 0").
    FIRST_EPOCH = 0

    def _join_rows(self, question: Question) -> list[Mismatch]:
        """The join's rows against the oracle's log (plan 5.5 Check 1): the diff over the
        run, the segment, the step or the sample the row names, with the lowered samples
        cut by the spec's rule; the rows of a position the spec records, each among the
        join's rows of its own sample; and the fields of the table that follow from the
        rows themselves (the coverage mark, the samples of a step, the rank of a sample).
        Every row reads the printed ``show SEGMENT --trace``; the oracle's log is the
        grading side and is never read by the tool."""
        given, expect = question.row.given, question.expect
        unread = set(expect) - self.JOIN_KEYS - self.PROSE_KEYS
        if unread:
            raise WorldError(f"row {question.row.id}: no comparison of {sorted(unread)} is "
                             "written until the tool prints what they read")
        fixture = acceptance_runs.spec()
        store = question.store
        run = given.get("run") or given["segment"].split("/")[0]
        run_names = acceptance_runs.segment_names(fixture, run)
        names = [given["segment"]] if "segment" in given else run_names
        first, last = expect.get("first_step"), expect.get("last_step")

        def in_launch(row: Mapping) -> bool:
            return ((first is None or int(row["step"]) >= first)
                    and (last is None or int(row["step"]) <= last))

        traces = {name: self.trace(store, name) for name in names}
        # The rows of a document through the whole run, whatever segment the row names.
        run_reported = [row for segment in run_names
                        for row in self.trace_rows(store, segment)]
        reported = [row for name in names for row in acceptance_trace.flatten(traces[name])
                    if in_launch(row)]
        out: list[Mismatch] = []
        if "diff" in expect:
            if expect["diff"] != "empty":
                raise WorldError(f"row {question.row.id}: diff = {expect['diff']!r}")
            oracle_root = self.oracle_runs(store)
            cuts = {sample: tokens for sample, tokens in
                    acceptance_runs.lowered_samples(fixture, run).items() if sample[0] in names}
            oracle = [row for name in names
                      for row in acceptance_runs.oracle_log(oracle_root, name) if in_launch(row)]
            within = (None, given.get("step"), given.get("sample_pos"))
            found = acceptance_join.diff(reported, oracle, cuts, within=within)
            if found:
                out.append(Mismatch("diff", "empty", f"{len(found)} findings; first: {found[:3]}"))
        if first is not None or last is not None:
            held = {int(step) for trace in traces.values() for step in
                    acceptance_trace.steps(trace)}
            missing = [step for step in range(first or 1, (last or max(held, default=0)) + 1)
                       if step not in held]
            if missing:
                out.append(Mismatch("first_step / last_step", f"steps {first} to {last} traced",
                                    f"no trace for steps {missing[:5]}"))
        position = expect.get("rows")
        if "rows" in expect:
            lines = position.get("log_lines") if isinstance(position, Mapping) else None
            if not lines:
                raise WorldError(f"row {question.row.id}: rows names no log lines of the spec")
            if "rows_of_the_covered_part" in position:
                out += self._covered_rows(position, reported)
            else:
                # A row about one launch asks for the lines of its own steps only.
                lines = [line for line in lines
                         if in_launch(json.loads(line) if isinstance(line, str) else line)]
                missing = acceptance_join.rows_named(lines, reported)
                if missing:
                    out.append(Mismatch("rows", f"{len(lines)} rows of the oracle's log",
                                        f"{len(missing)} missing; first: {missing[:3]}"))
        out += self._join_shape(question, traces, reported, run_reported, position)
        absent = self._join_forms(question, traces, out)
        # The tool prints these forms (docs/cli.md): one it leaves out is a failure of the
        # row, never a skip.
        out += [Mismatch(form, "printed", "absent") for form in absent]
        return out

    def _join_forms(self, question: Question, traces: Mapping[str, Mapping],
                    out: list[Mismatch]) -> list[str]:
        """The fields of a Check 1 row that are read from a form of their own: the launch
        (``launches`` of ``show --trace``), the step after which the epoch changes (the
        ``epochs`` of its steps) and the skipped line of a shard (the ``drops`` of ``show``).
        Mismatches go to ``out``; the names of the forms the command does not print are
        returned."""
        given, expect = question.row.given, question.expect
        absent: list[str] = []
        if {"consumed_at_start", "first_step", "last_step"} & set(expect) and "launch" in given:
            held = acceptance_trace.launches(traces[given["segment"]])
            if held is None:
                absent.append(self.NO_LAUNCHES)
            else:
                launch = acceptance_trace.launch_at(held, given["launch"])
                if launch is None:
                    out.append(Mismatch("launch", f"one launch at position {given['launch']}",
                                        f"{len(held)} launches: "
                                        f"{[each.get('position') for each in held]}"))
                else:
                    for name in ("consumed_at_start", "first_step", "last_step"):
                        if name in expect and launch.get(name) != expect[name]:
                            out.append(Mismatch(f"{name} (launch {given['launch']})",
                                                expect[name], launch.get(name)))
        if "boundary_after_step" in expect:
            trace = traces[given["segment"]]
            if not acceptance_trace.epochs_printed(trace):
                absent.append(self.NO_EPOCHS)
            else:
                after = acceptance_runs.spec()["positions"]["K"]["epoch_of_that_step"]
                boundary, found = acceptance_trace.epoch_boundary(trace, self.FIRST_EPOCH, after)
                if boundary != expect["boundary_after_step"]:
                    out.append(Mismatch("boundary_after_step", expect["boundary_after_step"],
                                        boundary))
                if found:
                    out.append(Mismatch("epochs", "one epoch to a step, 0 first, one more after "
                                        "the boundary", f"{len(found)} findings; first: "
                                        f"{found[:3]}"))
        if "skipped_line" in expect:
            out += self._skipped_line(question)
        return absent

    def _skipped_line(self, question: Question) -> list[Mismatch]:
        """The skipped sanitizer line of a shard, from ``show``: the drop that names the line
        as ``L<n>`` with its reason. The sanitizer's drop is recorded with the file it wrote
        as its output, which is a registered input of the shard (docs/cli.md, ``show``): the
        drops of the shard and of its inputs are read, and the line is the one drop of the
        row's reason."""
        wanted = question.expect["skipped_line"]
        store_file = self.store(question.store).store
        shard = self.shown(store_file, question.row.given["artifact"])
        names = [question.row.given["artifact"]] + [
            held["name"] for held in shard.get("inputs", ()) if held.get("registered") is True]
        drops = [drop for name in names for drop in self.drops_of(store_file, name)
                 if drop.get("reason") == wanted["reason"]]
        got = [drop.get("record") for drop in drops]
        if got == [f"L{wanted['input_line']}"]:
            return []
        return [Mismatch("skipped_line", {"record": f"L{wanted['input_line']}",
                                          "reason": wanted["reason"]},
                         {"records": got, "reason": wanted["reason"], "read": names})]

    def _k_digests(self, question: Question) -> list[Mismatch]:
        """Check 8's row of the shuffle: what ``register-run`` printed for run K's segment.
        One entry for each (launch, epoch, rank) the launches read, each equal to the record,
        and the epoch 0 digests of the launches equal rank by rank. The cells are the spec's:
        the launches of the segment, the steps an epoch holds and the data-parallel size."""
        given = question.row.given
        printed = self.registered(question.store, question.state)[given["segment"]]
        if "shuffle" not in printed:
            # register-run prints the computed shuffle (docs/cli.md): its absence is a
            # failure of the row, never a skip.
            return [Mismatch(self.NO_SHUFFLE, "printed", "absent")]
        fixture = acceptance_runs.spec()
        segment = fixture["runs"][given["run"]]["segments"][given["segment"]]
        in_an_epoch = fixture["positions"]["K"]["epoch_steps"]
        ranks = fixture["runs"][given["run"]]["data_parallel_size"]
        cells = sorted((launch["position"], epoch, rank)
                       for launch in segment["launches"]
                       for epoch in {(step - 1) // in_an_epoch
                                     for step in range(launch["first_step"],
                                                       launch["last_step"] + 1)}
                       for rank in range(ranks))
        wanted = fixture["positions"]["K"]["shuffle_digests"]
        if len(cells) != wanted:
            raise WorldError(f"row {question.row.id}: the spec's launches read {len(cells)} "
                             f"cells, its positions.K.shuffle_digests says {wanted}")
        # What was recorded is the run record the replica wrote (docs/interfaces.md 8.1), not
        # what the tool says it recorded.
        with open(self._record_path(question.store, question.state, given["segment"]),
                  encoding="utf-8") as f:
            held = {(e["launch"], e["epoch"], e["rank"]): (e["digest"], e["count"])
                    for e in json.load(f)["shuffle_digests"]}
        if sorted(held) != cells:
            raise WorldError(f"row {question.row.id}: the run record holds the shuffle digests "
                             f"{sorted(held)}, the spec's launches read {cells}")
        found = acceptance_trace.shuffle_findings(printed["shuffle"], cells, held)
        if not found:
            return []
        return [Mismatch("shuffle_digests", "equal, for every (launch, epoch, rank) of the run "
                         "record", f"{len(found)} findings; first: {found[:3]}")]

    @staticmethod
    def _covered_rows(position: Mapping, reported: Sequence[Mapping]) -> list[Mismatch]:
        """The rows of the half-covered sample: exactly the rows the spec lists for the
        covered part (the row that holds the first uncovered token ends at the cut)."""
        fixed = position["fixed"]
        sample = (fixed["segment"], fixed["step"], fixed["sample_pos"])
        got = acceptance_join.group(reported).get(sample, [])
        wanted = [(r["shard_position"], r["ordinal"], r["token_start"], r["token_end"])
                  for r in position["rows_of_the_covered_part"]]
        return [] if got == wanted else [Mismatch("rows", wanted, got)]

    def _join_shape(self, question: Question, traces: Mapping[str, Mapping],
                    reported: Sequence[Mapping], run_reported: Sequence[Mapping],
                    position: object) -> list[Mismatch]:
        """The fields of a Check 1 or 8 row that the printed trace answers by itself."""
        given, expect = question.row.given, question.expect
        spec_positions = acceptance_runs.spec()["positions"]
        out: list[Mismatch] = []

        def differ(field: str, wanted: object, got: object) -> None:
            if wanted != got:
                out.append(Mismatch(field, wanted, got))

        run_segments = list(traces)
        place, sample = None, None
        if "step" in given and "sample_pos" in given and "segment" in given:
            place = (given["segment"], given["step"], given["sample_pos"])
            sample = acceptance_trace.samples(traces[given["segment"]]).get(place)
            if sample is None and any(key in expect for key in (
                    "covered_tokens", "sample_tokens", "uncovered_in_sample", "starts_at",
                    "rows_that_name_T")):
                # A trace that holds no such sample says nothing about what it names or
                # covers: it is a finding, never a pass.
                out.append(Mismatch("sample", f"a sample at {place}", "the trace holds none"))
        if sample is not None:
            if "covered_tokens" in expect:
                differ("covered_tokens", expect["covered_tokens"], sample["covered_tokens"])
            if "sample_tokens" in expect:
                differ("sample_tokens", expect["sample_tokens"], sample["sample_tokens"])
            if "uncovered_in_sample" in expect:
                differ("uncovered_in_sample", expect["uncovered_in_sample"],
                       acceptance_trace.uncovered(sample))
            if "starts_at" in expect:
                rows = acceptance_join.group(reported).get(place, [])
                differ("starts_at (the token the first uncovered range starts at)",
                       expect["starts_at"]["token"], rows[-1][3] if rows else None)
        if "rows_that_name_T" in expect and place is not None:
            t = spec_positions["T"]
            named = [row for row in acceptance_join.rows_naming(
                reported, t["shard_position"], t["ordinal"])
                if (row["segment"], int(row["step"]), int(row["sample_pos"])) == place]
            differ("rows_that_name_T", expect["rows_that_name_T"], len(named))
        if "rows_of_T_in_run_E" in expect or "rows_in_sample" in expect:
            out += self._the_row_of_t(question, run_reported, position)
        elif "row" in expect:
            out += self._the_row_of_w(question, run_reported)
        if "documents_in_shard" in expect:
            shard = self.shown(self.store(question.store).store, given["artifact"])
            differ("documents_in_shard", expect["documents_in_shard"], shard.get("stored_rows"))
        if "consumed_samples" in expect:
            for trace in traces.values():
                step = acceptance_trace.steps(trace).get(given["step"])
                got = None if step is None else (step["consumed_samples"], len(step["samples"]))
                differ("consumed_samples", (expect["consumed_samples"],) * 2, got)
        if "loud_stop" in expect:
            checked = self.cli(self.store(question.store).store, "check")
            differ("loud_stop", expect["loud_stop"], checked.exit_code == 3)
        if "steps_with_uncovered_range" in expect:
            got = sorted({s for t in traces.values() for s in
                          acceptance_trace.steps_with_uncovered_range(t)})
            differ("steps_with_uncovered_range", expect["steps_with_uncovered_range"], got)
        if "steps_without_trace" in expect:
            # The steps that must be there are the oracle's (the grading side), not the
            # ones the trace happens to print: a trace with no steps has no step with a gap.
            gone = set()
            for name, trace in traces.items():
                gone |= {step for segment, step in trace["untraced_steps"]
                         if segment in run_segments}
                gone |= self.oracle_steps(question.store, name) - set(
                    acceptance_trace.steps(trace))
            differ("steps_without_trace", expect["steps_without_trace"], sorted(gone))
        if "samples_per_step" in expect:
            out += self._rank_layout(expect, traces, {
                name: self.oracle_steps(question.store, name) for name in traces})
        if "first_exposure" in expect:
            answered = self.cli(self.store(question.store).store, *self.query_arguments(question))
            out += table.compare_outcome(
                {"first_exposure": expect["first_exposure"]}, answered,
                lambda ref: self.locator(question.store, ref))
        return out

    @staticmethod
    def _the_row_of_t(question: Question, reported: Sequence[Mapping],
                      position: Mapping) -> list[Mismatch]:
        """T whole as the second row of one sample of step 7, and in no other row of the
        run: the sample's rows, the row at ``row_in_sample`` (counted from 1), and every
        row of the run that names T."""
        expect, out = question.expect, []
        fixed = position["fixed"]
        sample = (fixed["segment"], fixed["step"], position["sample_pos"])
        rows = acceptance_join.group(reported).get(sample, [])
        if "rows_in_sample" in expect and len(rows) != expect["rows_in_sample"]:
            out.append(Mismatch("rows_in_sample", expect["rows_in_sample"], len(rows)))
        if "row" in expect:
            wanted = (position["shard_position"], position["ordinal"],
                      expect["row"]["token_start"], expect["row"]["token_end"])
            at = expect["row"]["row_in_sample"] - 1
            got = rows[at] if 0 <= at < len(rows) else None
            if got != wanted:
                out.append(Mismatch("row", {"row_in_sample": at + 1, "span": wanted},
                                    {"row_in_sample": at + 1, "span": got}))
        if "rows_of_T_in_run_E" in expect:
            named = acceptance_join.rows_naming(reported, position["shard_position"],
                                                position["ordinal"])
            where = [(r["segment"], int(r["step"]), int(r["sample_pos"]), r["token_start"],
                      r["token_end"]) for r in named]
            wanted_rows = [(w["segment"], w["step"], w["sample_pos"], w["token_start"],
                            w["token_end"]) for w in position["rows_of_T_in_run_E"]]
            if len(named) != expect["rows_of_T_in_run_E"] or where != wanted_rows:
                out.append(Mismatch("rows_of_T_in_run_E", wanted_rows, where))
        return out

    def _the_row_of_w(self, question: Question, reported: Sequence[Mapping]) -> list[Mismatch]:
        """The one token of W: one row in the steps 1 to 9 names it, it spans the one
        token, and the sample that holds it has the 33 tokens of the plan."""
        expect, out = question.expect, []
        w = acceptance_runs.spec()["positions"]["shared_token"]
        named = acceptance_join.rows_naming(reported, w["shard_position"], w["ordinal"],
                                            steps=range(1, 10))
        if len(named) != expect["rows_of_W_in_steps_1_to_9"]:
            out.append(Mismatch("rows_of_W_in_steps_1_to_9", expect["rows_of_W_in_steps_1_to_9"],
                                len(named)))
        if len(named) == 1:
            row = named[0]
            span = {"token_start": row["token_start"], "token_end": row["token_end"]}
            if span != expect["row"]:
                out.append(Mismatch("row", expect["row"], span))
            held = acceptance_trace.samples(self.trace(question.store, row["segment"])).get(
                (row["segment"], int(row["step"]), int(row["sample_pos"])))
            if "sample_tokens" in expect and (held or {}).get("sample_tokens") != expect[
                    "sample_tokens"]:
                out.append(Mismatch("sample_tokens", expect["sample_tokens"],
                                    (held or {}).get("sample_tokens")))
        return out

    def oracle_steps(self, store: str, segment: str) -> set[int]:
        """The steps the oracle's log of a segment holds: the steps a trace of it must have."""
        root = self.oracle_runs(store)
        return {int(row["step"]) for row in acceptance_runs.oracle_log(root, segment)}

    @staticmethod
    def _rank_layout(expect: Mapping, traces: Mapping[str, Mapping],
                     wanted_steps: Mapping[str, set[int]]) -> list[Mismatch]:
        """Run K's samples of every step: how many, which ranks, and the place of each rank's
        samples (rank 0 holds sample_pos 0 and 1, rank 1 holds 2 and 3). The ranks are the
        trace's own field and the expected values are the table's, written before the tool;
        the steps that must be there are the oracle's, so that a trace which prints too few
        steps does not pass for the steps it leaves out."""
        out = []
        for name, trace in traces.items():
            laid = acceptance_trace.rank_layout(trace)
            absent = sorted(wanted_steps[name] - set(laid))
            if absent:
                out.append(Mismatch(f"{name} steps", "every step the oracle's log holds",
                                    f"no trace for steps {absent[:5]}"))
            for step, held in laid.items():
                by_rank = held["by_rank"]
                got = {"samples_per_step": held["samples"], "ranks": sorted(by_rank),
                       "samples_per_rank": sorted({len(v) for v in by_rank.values()}),
                       "sample_pos_of_rank_0": by_rank.get(0),
                       "sample_pos_of_rank_1": by_rank.get(1)}
                wanted = {key: ([expect[key]] if key == "samples_per_rank" else expect[key])
                          for key in got if key in expect}
                if {key: got[key] for key in wanted} != wanted:
                    out.append(Mismatch(f"{name} step {step}", wanted,
                                        {key: got[key] for key in wanted}))
                    break
        return out


def _kind_or_none(name: str) -> str | None:
    try:
        return table.artifact_kind(name)
    except table.TableError:
        return None


def _table_handle(record: str | None) -> str | None:
    """The handle the table writes for an unrelated record: ``U-<nnn>`` for
    ``Z-<nnn>`` (README section 5)."""
    if isinstance(record, str) and record.startswith("Z-"):
        return f"U-{record[2:]}"
    return record


def _no_handles(ref: RecordRef) -> str:
    raise WorldError(f"a row asked of a generated set names no record by a handle: {ref}")


_WAYS = {
    table.KIND_ANSWER: World._answer,
    table.KIND_LISTING: World._listing,
    table.KIND_ATTRIBUTION: World._attribution,
    table.KIND_INSTANCES: World._instances,
    table.KIND_REREGISTER: World._reregister,
    table.KIND_HARNESS: World._harness,
    # Checks 3 and 6
    table.KIND_SPEC_ITEM: World._spec_item,
    table.KIND_PAIR_KINDS: World._pair_kinds,
    table.KIND_SPANS: World._spans,
    table.KIND_SPAN_TOTALS: World._spans,
    table.KIND_SET_DROPS: World._set_drops,
    table.KIND_SET_NO_TOKENS: World._set_no_tokens,
    table.KIND_PLANTED_LINE: World._planted_line,
    table.KIND_RECORDS: World._records,
    table.KIND_RECORDS_AS_CAPTURE: World._records,
    table.KIND_TWINS: World._twins,
    table.KIND_TWINS_SWAPPED: World._twins,
    table.KIND_SPLIT_PAIRS: World._split_pairs,
    table.KIND_VERIFICATION_FAILED: World._verification_failed,
    table.KIND_MAP_FIELDS: World._map_fields,
    table.KIND_SET_DOCUMENTS: World._set_documents,
    table.KIND_SET_DROPS_RECOVERED: World._set_drops,
    table.KIND_SET_CANDIDATES: World._set_candidates,
    table.KIND_SET_DOWNSTREAM: World._set_downstream,
    table.KIND_DOCUMENTS: World._documents,
    table.KIND_DROPS: World._drops,
    table.KIND_EDGE_LABEL: World._edge_label,
    table.KIND_FILE_FAMILY: World._file_family,
    # Checks 1, 2 and 8
    table.KIND_RUN_REGISTRATION: World._registration,
    table.KIND_JOIN_ROWS: World._join_rows,
    table.KIND_K_DIGESTS: World._k_digests,
    table.KIND_SCRATCH_WRITE: World._scratch_write,
    # Checks 6, 7 and 9: the rebuild report and what a repair registers
    table.KIND_RECORD: World._record_rows,
    table.KIND_REPORT: World._report_fields,
}
