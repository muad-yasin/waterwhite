"""The cell runner of plan 5.6 (``replica/cells.py``, run as ``python -m replica.cells``).

A cell is one workload of the measurement of plan 5.6 (W7): a number of records, a
record length, derivations per source record, a number of sources and a layout, made
by ``fixture/plant/generate.py``. The runner builds each cell with ``build-set`` at
two worker settings and with capture off and on, in three paired runs, and records
the figures of every run. These tests are written before the module: every test
asks for it with ``need`` or ``need_attr`` as its first line, so a missing module is
an assertion that fails, not an import error.

Expected values come from plan 5.6, the fixture spec, docs/plant.md, docs/replica.md,
docs/cli.md and the schema of the store, from hand arithmetic, or from the test's
own reading of a file. Nothing is read out of a run of the code under test.

The interface these tests fix (every name is in ``replica.cells``):

    SEED, LENGTHS, DERIVATIONS, SOURCES, WORKERS, PAIRS, LEVELS, SAMPLE_INTERVAL_SECONDS
    NOT_APPLICABLE ("not applicable"), NOT_MEASURED ("not measured")
    LAYOUT_PARTITIONED, LAYOUT_ROUND_ROBIN, HARNESS_IN_PROCESS, HARNESS_CONFINED
    LEVEL_LABEL
    Cell(records, length, derivations, sources, layout, generator_extra=())
        .name, .as_dict()      generator_extra: more flags handed to the generator
    RunSpec(cell, jobs, capture, pair, harness="in-process")
        .pair_id, .run_id      the two halves of a pair share pair_id
    cell_grid() -> the cells;  run_plan(cells=None) -> the 240 runs, or those of the
        named cells;  confined_plan(cell=None) -> the runs of the confined figure
    read_machine(role=, host_type=, tree=, proc_root=, etc_root=) -> dict
    check_matched(off, on) -> the three matched values; raises PairMismatch
    run_batch(specs, out, steps, *, at_once=1, window_file=None, machine,
              sampler=False, repo=None) -> BatchResult
        .finished, .skipped, .failed (run ids), .refused_pairs (dicts)
    rerun(run_id, specs, out, steps, *, machine, sampler=False, repo=None)
    RunFailed(message, exit_code=1, commands=())   raised by a step that failed
    CommandSteps(repo=, python=, tokenizer_spec=, probe_steps=None, repair=None)
        the real steps; ``repair`` is a stand-in callable for the repair step
    A steps object has ``measure(spec, work, *, sampler) -> dict`` (the figures a
        run measured; see ``raw`` below for the keys). Every test but one hands the
        runner a stand-in of it through the ``steps`` argument.
    median_throughput(records), peak_from_samples(samples), sample_command(argv),
    metadata_bytes(store_file, copy_dir) -> per_record, per_parent_link, per_step_trace, rows
        (the bytes of a table are the pages of its b-trees and its indexes, from dbstat on a copy;
        records = rows of occurrence, parent links = rows of edge_input, stored step traces =
        rows of step_trace, whose bytes include trace_sample and trace_row)
    repair_bytes(historical=, store_before=, store_after=, repaired=) -> read_bytes, rewritten_bytes
        (folders by their file sizes, the store by its two sizes in bytes), capture_wiring(repo),
    file_digests(folder), tokenizer_digest(path), shard_order(build, layout),
    generate_argv, build_argv, probe_argv, query_argv, checkpoint_artifact,
    level_report(records), LevelRefused, main(argv)

Files of a batch folder: ``records/<run id>.json`` (one per finished or failed run),
``pairs/<pair id>.json`` (one per recorded pair), ``capture_wiring.json``.

A run record holds: run, pair, cell (``Cell.as_dict()``), jobs, capture, harness,
harness_statement (capture-on, in-process: the words "an in-process log establishes no
bound"), label (the confined figure only), sampler, at_once, beside (at_start, maximum),
machine, megatron_workers with megatron_workers_from, workers_meaning, matched, folders
(work, build, store), commands (name, argv, exit, seconds, output_file relative to the
batch folder), exit, failure, and figures: wall_seconds, throughput_records_per_second,
peak_memory, build_folder_bytes, store_file_bytes, query_seconds, proof_walk_seconds,
metadata_bytes, repair_bytes. A figure that exists only with capture on reads "not
applicable" in an off run; a peak memory taken with no sampler reads "not measured".
A pair file holds pair and matched (input_sha256 as one sha256 per file of the generated
folder, shard_order, tokenizer_sha256). The matched values are `check_matched`'s three keys.
A refused pair is {pair, differs: [names]} in ``BatchResult.refused_pairs``.

The commands of a real run are named "generate", "build-set", "probe-run", "query" (the
withdraw query on the checkpoint probe@N, N the steps probe-run printed) and "proof-walk"
(the same query over the whole store, no --artifact); a capture-off run has the first two
only. With ``at_once`` 1 the runs go in plan order. An interrupt (KeyboardInterrupt) in a
step stops the batch and is raised by ``run_batch`` after the window file is removed; a
run in progress then has no record. ``RunFailed`` in a step makes a failure record
(exit = its exit_code) and the batch goes on.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import re
import shlex
import signal
import sqlite3
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

MODULE = "replica.cells"


def need(name: str):
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    module = need(module_name)
    found = getattr(module, attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


def cells_module():
    return need(MODULE)


MIDDLE = (10000, 2000, 1, 4)  # plan 5.6: the middle cell


def middle_cell(cells, *, records=10000, layout="partitioned"):
    found = [c for c in cells.cell_grid()
             if (c.records, c.length, c.derivations, c.sources) == (records, 2000, 1, 4)
             and c.layout == layout]
    assert len(found) == 1, f"expected one middle cell of {records} records in {layout}, got {len(found)}"
    return found[0]


# --- Stand-ins -------------------------------------------------------------------------


def make_machine(cells, folder: Path, *, processors=16, mem_kb=31457280, pretty="Ubuntu 24.04.4 LTS",
                 tree="aaaa1111"):
    """A machine read from files the test writes, so that nothing is typed in the record."""
    proc = folder / "proc"
    etc = folder / "etc"
    proc.mkdir(parents=True, exist_ok=True)
    etc.mkdir(parents=True, exist_ok=True)
    (proc / "cpuinfo").write_text(
        "".join(f"processor\t: {n}\nmodel name\t: stand-in\n\n" for n in range(processors)),
        encoding="utf-8")
    (proc / "meminfo").write_text(f"MemTotal:       {mem_kb} kB\nMemFree:        1 kB\n", encoding="utf-8")
    (etc / "os-release").write_text(f'NAME="Ubuntu"\nPRETTY_NAME="{pretty}"\n', encoding="utf-8")
    return cells.read_machine(role="server", host_type="CPX62", tree=tree, proc_root=proc, etc_root=etc)


def raw(spec, *, wall=20.0, sampler=False, input_sha=None, order=None, tokenizer="bb"):
    """What a stand-in step returns for one run: the keys the real steps return."""
    figures = {
        "wall_seconds": wall,
        "build_folder_bytes": 1000,
        "megatron_workers": 3,
        "input_sha256": input_sha or {"set.toml": "aa", "A.jsonl": "cc"},
        "shard_order": order or ["shard-A", "shard-Z", "shard-gen"],
        "tokenizer_sha256": tokenizer,
        "peak_memory": {"sum_bytes": 5000, "largest_process_bytes": 3000} if sampler else None,
        "commands": [{"name": "build-set", "argv": ["x", "build-set"], "exit": 0, "seconds": wall,
                      "output_file": "logs/build-set.log"}],
    }
    if spec.capture:
        figures.update(
            store_file_bytes=2000, query_seconds=0.5, proof_walk_seconds=1.5,
            metadata_bytes={"per_record": 10.0, "per_parent_link": 11.0, "per_step_trace": 12.0},
            repair_bytes={"read_bytes": 100, "rewritten_bytes": 50})
    return figures


class FakeSteps:
    """A stand-in for the real steps. It records what the runner called it with."""

    def __init__(self, cells, *, fail=None, interrupt=(), window=None, barrier=None, hold=None,
                 wall=20.0, differ=None):
        self.cells = cells
        self.fail = dict(fail or {})
        self.interrupt = set(interrupt)
        self.window = window
        self.barrier = barrier
        self.hold = dict(hold or {})
        self.wall = wall
        self.differ = differ or {}
        self.calls = []
        self.works = []
        self.samplers = []
        self.window_seen = []
        self._lock = threading.Lock()

    def measure(self, spec, work, *, sampler):
        with self._lock:
            self.calls.append(spec.run_id)
            self.works.append(Path(work))
            self.samplers.append(sampler)
            if self.window is not None:
                self.window_seen.append(Path(self.window).exists())
        if self.barrier is not None:
            self.barrier.wait(timeout=20)
        if spec.run_id in self.interrupt:
            raise KeyboardInterrupt
        if spec.run_id in self.hold:
            deadline = time.monotonic() + self.hold[spec.run_id]
            while time.monotonic() < deadline:
                time.sleep(0.05)
        if spec.run_id in self.fail:
            raise self.cells.RunFailed(
                self.fail[spec.run_id], exit_code=7,
                commands=[{"name": "build-set", "argv": ["x"], "exit": 7, "seconds": 1.0,
                           "output_file": "logs/failed.log"}])
        figures = raw(spec, wall=self.wall, sampler=sampler)
        figures.update(self.differ.get(spec.run_id, {}))
        return figures


def read_record(out: Path, run_id: str) -> dict:
    return json.loads((out / "records" / f"{run_id}.json").read_text(encoding="utf-8"))


def run(cells, specs, out, steps, tmp_path, **kwargs):
    kwargs.setdefault("machine", make_machine(cells, tmp_path / "machine"))
    return cells.run_batch(specs, out, steps, **kwargs)


def flag_value(argv, flag):
    assert flag in argv, f"{flag} is not in {argv}"
    return argv[argv.index(flag) + 1]


# --- (a) the module --------------------------------------------------------------------


def test_the_cell_runner_module_exists():
    need(MODULE)


# --- (b) the grid and the run count ------------------------------------------------------


def test_the_grid_is_the_eighteen_cells_plus_the_round_robin_and_the_100k_cell():
    cells = cells_module()
    grid = cells.cell_grid()
    assert len(grid) == 20
    assert len({c.name for c in grid}) == 20
    ten_k = [c for c in grid if c.records == 10000 and c.layout == "partitioned"]
    assert len(ten_k) == 18
    assert {(c.length, c.derivations, c.sources) for c in ten_k} == {
        (length, derivations, sources)
        for length in (500, 2000, 8000) for derivations in (0, 1, 2) for sources in (4, 10)}
    round_robin = [c for c in grid if c.layout == "round-robin" and c.records == 10000]
    assert [(c.records, c.length, c.derivations, c.sources) for c in round_robin] == [MIDDLE]
    hundred_k = [c for c in grid if c.records == 100000]
    assert [(c.length, c.derivations, c.sources) for c in hundred_k] == [MIDDLE[1:]]
    assert {c.records for c in grid} == {10000, 100000}


def test_the_run_count_is_240_and_derived_from_the_grid():
    cells = cells_module()
    plan = cells.run_plan()
    assert len(plan) == 240
    assert len({r.run_id for r in plan}) == 240
    by_cell = {}
    for r in plan:
        by_cell.setdefault(r.cell.name, []).append(r)
    assert len(by_cell) == 20
    for runs in by_cell.values():
        assert len(runs) == 12
        assert {(r.jobs, r.capture, r.pair) for r in runs} == {
            (jobs, capture, pair) for jobs in (1, 4) for capture in (False, True) for pair in (1, 2, 3)}
    assert sum(1 for r in plan if r.cell.records == 10000 and r.cell.layout == "partitioned") == 216
    assert sum(1 for r in plan if r.cell.layout == "round-robin" and r.cell.records == 10000) == 12
    assert sum(1 for r in plan if r.cell.records == 100000) == 12
    # The plan runs the in-process harness only; the run count is a function of the grid.
    assert {r.harness for r in plan} == {"in-process"}
    assert len(cells.run_plan([cells.cell_grid()[0]])) == 12


def test_a_pair_is_one_off_run_and_one_on_run_at_the_same_workers():
    cells = cells_module()
    plan = cells.run_plan([middle_cell(cells)])
    pairs = {}
    for r in plan:
        pairs.setdefault(r.pair_id, []).append(r)
    assert len(pairs) == 6  # two worker settings, three pairs
    for halves in pairs.values():
        assert sorted(r.capture for r in halves) == [False, True]
        assert len({(r.jobs, r.pair, r.cell.name) for r in halves}) == 1
        assert len({r.run_id for r in halves}) == 2


def test_the_confined_figure_is_outside_the_240_for_the_middle_cell_at_both_worker_settings():
    cells = cells_module()
    confined = cells.confined_plan()
    in_240 = {r.run_id for r in cells.run_plan()}
    assert len(confined) == 6  # both worker settings, three runs each
    assert all(r.harness == "confined" and r.capture for r in confined)
    assert {r.jobs for r in confined} == {1, 4}
    assert all(r.run_id not in in_240 for r in confined)
    assert {(r.cell.records, r.cell.length, r.cell.derivations, r.cell.sources) for r in confined} == {MIDDLE}


def test_the_generator_is_called_with_the_cells_parameters_a_fixed_seed_and_dedup_on(repo):
    cells = cells_module()
    out = Path("out-folder")
    first, second = cells.cell_grid()[0], cells.cell_grid()[-1]
    argv = cells.generate_argv(first, out, python="PY", repo=repo)
    assert argv[0] == "PY" and argv[1].replace("\\", "/").endswith("fixture/plant/generate.py")
    assert flag_value(argv, "--records") == str(first.records)
    assert flag_value(argv, "--length") == str(first.length)
    assert flag_value(argv, "--derivations") == str(first.derivations)
    assert flag_value(argv, "--sources") == str(first.sources)
    assert flag_value(argv, "--out") == str(out)
    assert flag_value(argv, "--dedup") == "on"
    other = cells.generate_argv(second, out, python="PY", repo=repo)
    assert flag_value(argv, "--seed") == flag_value(other, "--seed") == str(cells.SEED)


def test_the_build_command_is_build_set_with_jobs_layout_and_capture_flags(repo):
    cells = cells_module()
    cell = middle_cell(cells)
    off, on = [r for r in cells.run_plan([cell]) if r.jobs == 4 and r.pair == 1]
    if off.capture:
        off, on = on, off
    paths = dict(set_dir=Path("s"), build_dir=Path("b"), store=Path("st.sqlite"),
                 python="PY", tokenizer_spec=repo / "fixture" / "spec" / "fixture_spec.toml")
    argv_off = cells.build_argv(off, **paths)
    argv_on = cells.build_argv(on, **paths)
    for argv in (argv_off, argv_on):
        assert argv[:4] == ["PY", "-m", "replica", "build-set"]
        assert flag_value(argv, "--jobs") == "4"
        assert flag_value(argv, "--layout") == cell.layout
        assert flag_value(argv, "--set") == "s" and flag_value(argv, "--out") == "b"
        assert flag_value(argv, "--tokenizer-spec").endswith("fixture_spec.toml")
    assert "--capture" not in argv_off and "--store" not in argv_off
    assert "--capture" in argv_on
    assert flag_value(argv_on, "--store") == "st.sqlite"
    assert flag_value(argv_on, "--list-join") == "single"
    assert flag_value(argv_on, "--generation-harness") == "in-process"
    confined = cells.confined_plan(cell)[0]
    assert flag_value(cells.build_argv(confined, **paths), "--generation-harness") == "confined"


def test_the_probe_and_the_query_commands_follow_the_documented_forms():
    cells = cells_module()
    assert cells.checkpoint_artifact(17) == "probe@17"
    probe = cells.probe_argv(Path("b"), Path("st"), Path("p"), "partitioned", python="PY")
    assert probe[:4] == ["PY", "-m", "replica", "probe-run"]
    assert flag_value(probe, "--set-build") == "b" and flag_value(probe, "--store") == "st"
    assert flag_value(probe, "--out") == "p" and flag_value(probe, "--layout") == "partitioned"
    assert "--steps" not in probe
    assert flag_value(cells.probe_argv(Path("b"), Path("st"), Path("p"), "round-robin", steps=30, python="PY"),
                      "--steps") == "30"
    query = cells.query_argv(Path("st"), "probe@17", python="PY")
    where = query.index("query")
    assert query[where:where + 3] == ["query", "withdraw", "A"]
    assert flag_value(query, "--store") == "st" and flag_value(query, "--artifact") == "probe@17"
    # The coverage-proof walk is the same command over the whole store: no --artifact.
    walk = cells.query_argv(Path("st"), None, python="PY")
    where = walk.index("query")
    assert walk[where:where + 3] == ["query", "withdraw", "A"]
    assert "--artifact" not in walk


# --- (c) matched input, order and tokenizer ----------------------------------------------


MATCHED = {"input_sha256": {"set.toml": "aa", "A.jsonl": "cc"},
           "shard_order": ["shard-A", "shard-Z", "shard-gen"], "tokenizer_sha256": "bb"}


def test_an_equal_pair_is_accepted_and_returns_the_three_values():
    cells = cells_module()
    got = cells.check_matched(dict(MATCHED), dict(MATCHED))
    assert got == MATCHED


@pytest.mark.parametrize("field,other", [
    ("input_sha256", {"set.toml": "aa", "A.jsonl": "dd"}),
    ("shard_order", ["shard-Z", "shard-A", "shard-gen"]),
    ("tokenizer_sha256", "ee"),
])
def test_a_pair_that_differs_in_any_one_of_the_three_values_is_refused_by_name(field, other):
    cells = cells_module()
    on = dict(MATCHED)
    on[field] = other
    with pytest.raises(cells.PairMismatch) as caught:
        cells.check_matched(dict(MATCHED), on)
    assert field in str(caught.value)


def test_the_batch_records_an_equal_pair_and_refuses_to_record_a_differing_one(tmp_path):
    cells = cells_module()
    cell = middle_cell(cells)
    specs = [r for r in cells.run_plan([cell]) if r.jobs == 1]  # three pairs
    assert len(specs) == 6
    bad_on = next(r for r in specs if r.capture and r.pair == 2)
    good = next(r for r in specs if r.pair == 1)
    steps = FakeSteps(cells, differ={bad_on.run_id: {"tokenizer_sha256": "different"}})
    out = tmp_path / "batch"
    result = run(cells, specs, out, steps, tmp_path, at_once=1)
    recorded = sorted(p.stem for p in (out / "pairs").glob("*.json"))
    assert recorded == sorted({r.pair_id for r in specs} - {bad_on.pair_id})
    pair = json.loads((out / "pairs" / f"{good.pair_id}.json").read_text(encoding="utf-8"))
    assert pair["matched"] == {"input_sha256": {"set.toml": "aa", "A.jsonl": "cc"},
                               "shard_order": ["shard-A", "shard-Z", "shard-gen"],
                               "tokenizer_sha256": "bb"}
    assert [(p["pair"], p["differs"]) for p in result.refused_pairs] == [
        (bad_on.pair_id, ["tokenizer_sha256"])]


# --- Figures of one run ------------------------------------------------------------------


def test_throughput_is_records_per_second_of_the_build_wall_time(tmp_path):
    cells = cells_module()
    cell = middle_cell(cells)
    spec = cells.RunSpec(cell=cell, jobs=1, capture=False, pair=1)
    out = tmp_path / "batch"
    run(cells, [spec], out, FakeSteps(cells, wall=20.0), tmp_path)
    record = read_record(out, spec.run_id)
    assert record["figures"]["wall_seconds"] == 20.0
    assert record["figures"]["throughput_records_per_second"] == pytest.approx(10000 / 20.0)  # 500.0
    assert record["cell"] == cell.as_dict()
    assert record["jobs"] == 1 and record["capture"] is False and record["exit"] == 0


def test_the_median_throughput_is_over_the_pairs_and_ignores_sampler_runs():
    cells = cells_module()

    def record(throughput, sampler=False, exit_code=0):
        return {"cell": {}, "jobs": 1, "capture": True, "exit": exit_code, "sampler": sampler,
                "figures": {"throughput_records_per_second": throughput}}

    three = [record(100.0), record(300.0), record(200.0)]
    assert cells.median_throughput(three) == pytest.approx(200.0)
    assert cells.median_throughput(three + [record(999.0, sampler=True)]) == pytest.approx(200.0)
    assert cells.median_throughput(three + [record(1.0, exit_code=7)]) == pytest.approx(200.0)


def test_an_off_run_has_no_store_so_its_on_only_figures_are_not_applicable_never_zero(tmp_path):
    cells = cells_module()
    cell = middle_cell(cells)
    off = cells.RunSpec(cell=cell, jobs=1, capture=False, pair=1)
    on = cells.RunSpec(cell=cell, jobs=1, capture=True, pair=1)
    out = tmp_path / "batch"
    run(cells, [off, on], out, FakeSteps(cells), tmp_path)
    figures_off = read_record(out, off.run_id)["figures"]
    for key in ("store_file_bytes", "query_seconds", "proof_walk_seconds", "metadata_bytes", "repair_bytes"):
        assert figures_off[key] == cells.NOT_APPLICABLE == "not applicable", key
    assert figures_off["wall_seconds"] == 20.0 and figures_off["build_folder_bytes"] == 1000
    figures_on = read_record(out, on.run_id)["figures"]
    assert figures_on["store_file_bytes"] == 2000
    assert figures_on["query_seconds"] == 0.5 and figures_on["proof_walk_seconds"] == 1.5
    assert figures_on["metadata_bytes"] == {"per_record": 10.0, "per_parent_link": 11.0,
                                            "per_step_trace": 12.0}
    assert figures_on["repair_bytes"] == {"read_bytes": 100, "rewritten_bytes": 50}


def test_peak_memory_is_a_figure_only_of_a_run_with_the_sampler(tmp_path):
    cells = cells_module()
    cell = middle_cell(cells)
    spec = cells.RunSpec(cell=cell, jobs=1, capture=False, pair=1)
    without = tmp_path / "without"
    run(cells, [spec], without, FakeSteps(cells), tmp_path)
    record = read_record(without, spec.run_id)
    assert record["sampler"] is False
    assert record["figures"]["peak_memory"] == cells.NOT_MEASURED == "not measured"
    with_sampler = tmp_path / "with"
    steps = FakeSteps(cells)
    run(cells, [spec], with_sampler, steps, tmp_path, sampler=True)
    record = read_record(with_sampler, spec.run_id)
    assert steps.samplers == [True]
    assert record["sampler"] is True
    assert record["figures"]["peak_memory"] == {"sum_bytes": 5000, "largest_process_bytes": 3000}


def test_a_run_record_keeps_every_command_with_its_exit_and_output_file(tmp_path):
    cells = cells_module()
    spec = cells.RunSpec(cell=middle_cell(cells), jobs=1, capture=False, pair=1)
    out = tmp_path / "batch"
    run(cells, [spec], out, FakeSteps(cells), tmp_path)
    record = read_record(out, spec.run_id)
    assert record["commands"] == [{"name": "build-set", "argv": ["x", "build-set"], "exit": 0,
                                   "seconds": 20.0, "output_file": "logs/build-set.log"}]


# --- (e) machine, sampler flag, harness word, runs beside ----------------------------------


def test_the_machine_is_read_at_run_time_from_the_machine_not_typed(tmp_path):
    cells = cells_module()
    big = make_machine(cells, tmp_path / "m1", processors=16, mem_kb=31457280, pretty="Ubuntu 24.04.4 LTS",
                       tree="t1")
    assert big == {"host_role": "server", "type": "CPX62", "cores": 16,
                   "ram_bytes": 31457280 * 1024, "os": "Ubuntu 24.04.4 LTS", "tree": "t1"}
    small = make_machine(cells, tmp_path / "m2", processors=4, mem_kb=8388608, pretty="Other OS 1",
                         tree="t2")
    assert (small["cores"], small["ram_bytes"], small["os"], small["tree"]) == (4, 8388608 * 1024, "Other OS 1", "t2")


def test_the_machine_read_from_this_host_has_the_cores_its_proc_lists():
    cells = cells_module()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        pytest.skip("no /proc/cpuinfo on this host")
    expected = sum(1 for line in cpuinfo.read_text(encoding="utf-8").splitlines()
                   if re.match(r"processor\s*:", line))
    machine = cells.read_machine(role="server", host_type="CPX62", tree="t")
    assert machine["cores"] == expected and machine["ram_bytes"] > 0 and machine["os"]


def test_every_record_carries_machine_sampler_harness_workers_and_beside_count(tmp_path, repo):
    cells = cells_module()
    cell = middle_cell(cells)
    specs = [r for r in cells.run_plan([cell]) if r.jobs == 4 and r.pair == 1]
    machine = make_machine(cells, tmp_path / "machine")
    out = tmp_path / "batch"
    cells.run_batch(specs, out, FakeSteps(cells), at_once=1, machine=machine, repo=repo)
    for spec in specs:
        record = read_record(out, spec.run_id)
        assert record["machine"] == machine
        assert record["sampler"] is False
        assert record["jobs"] == 4 and "--jobs 4" in record["workers_meaning"]
        assert record["megatron_workers"] == 3
        assert "tokenizer.workers" in record["megatron_workers_from"]
        assert record["at_once"] == 1
        assert record["beside"] == {"at_start": 0, "maximum": 0}
        assert "harness" in record
        if spec.capture:
            assert record["harness"] == "in-process"
            assert "an in-process log establishes no bound" in record["harness_statement"]
    # The one static figure, written once beside the records (the same as capture_wiring(repo)).
    assert json.loads((out / "capture_wiring.json").read_text(encoding="utf-8")) == cells.capture_wiring(repo)


def test_the_confined_records_name_their_harness_and_make_no_in_process_statement(tmp_path):
    cells = cells_module()
    spec = cells.confined_plan()[0]
    out = tmp_path / "batch"
    run(cells, [spec], out, FakeSteps(cells), tmp_path)
    record = read_record(out, spec.run_id)
    assert record["harness"] == "confined"
    assert "in-process log establishes no bound" not in record.get("harness_statement", "")
    assert "confined" in record["label"] and "240" in record["label"]


def test_two_runs_at_once_run_together_in_their_own_folders_and_state_the_level(tmp_path):
    cells = cells_module()
    cell = middle_cell(cells)
    specs = [r for r in cells.run_plan([cell]) if r.jobs == 1 and r.pair == 1]
    assert len(specs) == 2
    steps = FakeSteps(cells, barrier=threading.Barrier(2))
    out = tmp_path / "batch"
    result = run(cells, specs, out, steps, tmp_path, at_once=2)
    assert sorted(result.finished) == sorted(r.run_id for r in specs)
    assert len(set(steps.works)) == 2
    for work in steps.works:
        assert work.is_dir() and out in work.parents
    for spec in specs:
        record = read_record(out, spec.run_id)
        assert record["at_once"] == 2  # both halves of the pair at the same level
        assert record["beside"]["maximum"] == 1
        assert record["beside"]["at_start"] in (0, 1)


# --- (d) resume, failures, explicit rerun -------------------------------------------------


def test_a_stopped_batch_resumes_without_redoing_finished_runs_and_never_reruns_a_failure(tmp_path):
    cells = cells_module()
    specs = list(cells.run_plan([middle_cell(cells)]))
    ids = [r.run_id for r in specs]
    out = tmp_path / "batch"
    first = FakeSteps(cells, fail={ids[2]: "build-set exited 7"}, interrupt={ids[4]})
    with pytest.raises(KeyboardInterrupt):
        run(cells, specs, out, first, tmp_path, at_once=1)
    assert first.calls == ids[:5]
    present = sorted(p.stem for p in (out / "records").glob("*.json"))
    assert present == sorted([ids[0], ids[1], ids[2], ids[3]])
    failed = read_record(out, ids[2])
    assert failed["exit"] == 7 and "build-set exited 7" in failed["failure"]
    assert "throughput_records_per_second" not in failed.get("figures", {})
    before = hashlib.sha256((out / "records" / f"{ids[2]}.json").read_bytes()).hexdigest()

    second = FakeSteps(cells)
    result = run(cells, specs, out, second, tmp_path, at_once=1)
    assert second.calls == ids[4:]  # the finished four are skipped, the failure too
    assert ids[2] not in second.calls
    assert result.failed == [ids[2]]
    assert sorted(result.skipped) == sorted([ids[0], ids[1], ids[3]])
    assert hashlib.sha256((out / "records" / f"{ids[2]}.json").read_bytes()).hexdigest() == before

    third = FakeSteps(cells)
    cells.rerun(ids[2], specs, out, third, machine=make_machine(cells, tmp_path / "machine"))
    assert third.calls == [ids[2]]
    assert read_record(out, ids[2])["exit"] == 0


# --- (g) the window file --------------------------------------------------------------------


def test_the_window_file_exists_during_a_batch_and_is_gone_after(tmp_path):
    cells = cells_module()
    specs = list(cells.run_plan([middle_cell(cells)]))[:3]
    window = tmp_path / "TIMED_WINDOW"
    steps = FakeSteps(cells, window=window)
    assert not window.exists()
    run(cells, specs, tmp_path / "batch", steps, tmp_path, window_file=window)
    assert steps.window_seen == [True, True, True]
    assert not window.exists()


def test_the_window_file_is_gone_after_a_run_failed_and_after_a_stop(tmp_path):
    cells = cells_module()
    specs = list(cells.run_plan([middle_cell(cells)]))[:3]
    window = tmp_path / "TIMED_WINDOW"
    failing = FakeSteps(cells, window=window, fail={specs[1].run_id: "failed"})
    run(cells, specs, tmp_path / "failed", failing, tmp_path, window_file=window)
    assert not window.exists()
    stopping = FakeSteps(cells, window=window, interrupt={specs[1].run_id})
    with pytest.raises(KeyboardInterrupt):
        run(cells, specs, tmp_path / "stopped", stopping, tmp_path, window_file=window)
    assert stopping.window_seen == [True, True]
    assert not window.exists()


def test_the_window_file_is_gone_after_a_real_interrupt_signal(tmp_path):
    cells = cells_module()
    specs = list(cells.run_plan([middle_cell(cells)]))[:3]
    window = tmp_path / "TIMED_WINDOW"
    steps = FakeSteps(cells, window=window, hold={specs[1].run_id: 4.0})
    timer = threading.Timer(1.0, os.kill, args=(os.getpid(), signal.SIGINT))
    timer.start()
    try:
        try:
            run(cells, specs, tmp_path / "batch", steps, tmp_path, window_file=window)
        except KeyboardInterrupt:
            pass
    finally:
        timer.cancel()
    assert steps.window_seen[:2] == [True, True]
    assert not window.exists()


# --- The hooks between runs and the run command (the before-run and after-run commands) ---------

HOOK_SCRIPT = '''\
import sys
log, name, exits = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
with open(log, "a", encoding="utf-8") as handle:
    handle.write(name + "\\n")
with open(log, encoding="utf-8") as handle:
    index = sum(1 for line in handle if line.strip() == name) - 1
print("h" * 300)
sys.exit(int(exits[index]) if index < len(exits) else 0)
'''


class LoggingSteps(FakeSteps):
    """The stand-in steps, writing a line to the hooks' log when a run starts, so that the order of
    the hooks and the runs can be read from one file."""

    def __init__(self, cells, log, **kwargs):
        super().__init__(cells, **kwargs)
        self.log = Path(log)

    def measure(self, spec, work, *, sampler):
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write(f"run:{spec.run_id}\n")
        return super().measure(spec, work, sampler=sampler)


def hook_command(tmp_path, log, name, exits="0"):
    """A stand-in hook: a command that appends its name to ``log`` and exits with the status the
    list ``exits`` holds for its n-th call (0 beyond the list), after printing 300 characters."""
    script = tmp_path / "hook.py"
    if not script.exists():
        script.write_text(HOOK_SCRIPT, encoding="utf-8")
    return shlex.join([sys.executable, str(script), str(log), name, exits])


def logged(log):
    path = Path(log)
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []


def hooked_specs(cells, count=3):
    return list(cells.run_plan([middle_cell(cells)]))[:count]


def test_a_before_hook_that_fails_stops_the_batch_and_the_refused_run_has_its_record(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells)
    ids = [s.run_id for s in specs]
    log, out = tmp_path / "log", tmp_path / "batch"
    steps = LoggingSteps(cells, log)
    result = run(cells, specs, out, steps, tmp_path, before_run=hook_command(tmp_path, log, "before", "0,7"),
                 after_run=hook_command(tmp_path, log, "after"))
    assert logged(log)[:3] == ["before", f"run:{ids[0]}", "after"]
    assert f"run:{ids[1]}" not in logged(log) and f"run:{ids[2]}" not in logged(log)
    assert steps.calls == [ids[0]] and result.finished == [ids[0]] and result.stopped
    assert sorted(p.stem for p in (out / "records").glob("*.json")) == [ids[0]]  # the finished run stays
    held = json.loads((out / "held" / f"{ids[1]}.json").read_text(encoding="utf-8"))
    assert held["run"] == ids[1] and held["hold"]["exit"] == 7
    assert held["hold"]["output"] == "h" * 200  # the first 200 characters of what the hook printed
    assert not (out / "held" / f"{ids[2]}.json").exists()


def test_a_before_hook_that_cannot_be_started_refuses_every_run(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells)
    out = tmp_path / "batch"
    steps = FakeSteps(cells)
    result = run(cells, specs, out, steps, tmp_path, before_run=str(tmp_path / "no-such-hook"))
    assert steps.calls == [] and result.finished == [] and result.stopped
    held = json.loads((out / "held" / f"{specs[0].run_id}.json").read_text(encoding="utf-8"))
    assert held["hold"]["exit"] != 0 and held["hold"]["output"]
    assert list((out / "records").glob("*.json")) == []


def test_a_before_hook_that_passes_is_recorded_for_every_run_of_a_group_and_none_without_a_hook(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells, 4)
    log, out = tmp_path / "log", tmp_path / "batch"
    before = hook_command(tmp_path, log, "before")
    run(cells, specs, out, LoggingSteps(cells, log), tmp_path, before_run=before)
    assert logged(log).count("before") == 4  # one call before each run
    for spec in specs:
        hold = read_record(out, spec.run_id)["hold"]
        assert hold["exit"] == 0 and hold["output"] == "h" * 200 and str(tmp_path / "hook.py") in hold["command"]
    assert list((out / "held").glob("*.json")) == []
    grouped_log, grouped = tmp_path / "grouped-log", tmp_path / "grouped"
    run(cells, specs, grouped, LoggingSteps(cells, grouped_log), tmp_path, at_once=2,
        before_run=hook_command(tmp_path, grouped_log, "before"))
    assert logged(grouped_log).count("before") == 2  # one call before each group of two
    assert all(read_record(grouped, s.run_id)["hold"]["exit"] == 0 for s in specs)
    bare = tmp_path / "bare"
    run(cells, specs[:1], bare, FakeSteps(cells), tmp_path)
    assert read_record(bare, specs[0].run_id).get("hold") is None


def test_the_hooks_run_before_and_after_each_run_in_that_order_and_the_after_hook_follows_a_failed_run(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells)
    ids = [s.run_id for s in specs]
    log, out = tmp_path / "log", tmp_path / "batch"
    steps = LoggingSteps(cells, log, fail={ids[1]: "build-set exited 7"})
    run(cells, specs, out, steps, tmp_path, before_run=hook_command(tmp_path, log, "before"),
        after_run=hook_command(tmp_path, log, "after"))
    assert logged(log) == ["before", f"run:{ids[0]}", "after", "before", f"run:{ids[1]}", "after",
                           "before", f"run:{ids[2]}", "after"]
    failed = read_record(out, ids[1])
    assert failed["exit"] == 7 and failed["hold"]["exit"] == 0


def test_the_after_hook_is_called_after_an_interrupted_run_too(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells)
    ids = [s.run_id for s in specs]
    log, out = tmp_path / "log", tmp_path / "batch"
    steps = LoggingSteps(cells, log, interrupt={ids[1]})
    with pytest.raises(KeyboardInterrupt):
        run(cells, specs, out, steps, tmp_path, before_run=hook_command(tmp_path, log, "before"),
            after_run=hook_command(tmp_path, log, "after"))
    assert logged(log) == ["before", f"run:{ids[0]}", "after", "before", f"run:{ids[1]}", "after"]


def test_a_resumed_batch_calls_the_before_hook_again_for_its_first_run(tmp_path):
    cells = cells_module()
    specs = hooked_specs(cells)
    ids = [s.run_id for s in specs]
    out = tmp_path / "batch"
    log = tmp_path / "log-1"
    run(cells, specs, out, LoggingSteps(cells, log), tmp_path,
        before_run=hook_command(tmp_path, log, "before", "0,7"))
    assert [p.stem for p in (out / "records").glob("*.json")] == [ids[0]]
    again = tmp_path / "log-2"
    steps = LoggingSteps(cells, again)
    result = run(cells, specs, out, steps, tmp_path, before_run=hook_command(tmp_path, again, "before"),
                 after_run=hook_command(tmp_path, again, "after"))
    assert logged(again)[:3] == ["before", f"run:{ids[1]}", "after"]  # the hook is asked again first
    assert steps.calls == ids[1:] and result.skipped == [ids[0]] and not result.stopped
    assert read_record(out, ids[1])["hold"]["exit"] == 0


def refusing_hook():
    return shlex.join([sys.executable, "-c", "print('no box'); raise SystemExit(9)"])


def main_with_sigterm_kept(cells, argv):
    """main installs a handler for SIGTERM; the test puts the old one back."""
    old = signal.getsignal(signal.SIGTERM)
    try:
        return cells.main(argv)
    finally:
        signal.signal(signal.SIGTERM, old)


def batch_argv(command, cell, out, *more):
    return [command, "--out", str(out), "--host-type", "CPX62", "--tree", "abc1234", "--cell", cell.name,
            *more]


def test_the_run_command_stops_at_a_refusing_before_hook_before_any_step_starts(tmp_path, capsys):
    cells = cells_module()
    cell = middle_cell(cells)
    out, window = tmp_path / "batch", tmp_path / "TIMED_WINDOW"
    code = main_with_sigterm_kept(cells, batch_argv("run", cell, out, "--at-once", "2", "--window-file",
                                                    str(window), "--before-run", refusing_hook()))
    printed = json.loads(capsys.readouterr().out)
    assert code == 1
    assert printed["finished"] == [] and printed["failed"] == [] and printed["stopped"]
    plan = cells.run_plan([cell])
    assert sorted(p.stem for p in (out / "held").glob("*.json")) == sorted(s.run_id for s in plan[:2])
    assert list((out / "records").glob("*.json")) == []
    assert not any((out / "work").iterdir())  # no run had a work folder: no step started
    assert not window.exists()


def test_the_rerun_command_names_one_run_and_the_run_command_refuses_names_the_plan_lacks(tmp_path, capsys):
    cells = cells_module()
    cell = middle_cell(cells)
    out = tmp_path / "batch"
    first = cells.run_plan([cell])[0].run_id
    code = main_with_sigterm_kept(cells, batch_argv("rerun", cell, out, "--run", first,
                                                    "--before-run", refusing_hook()))
    assert code == 1 and json.loads(capsys.readouterr().out)["stopped"]
    assert (out / "held" / f"{first}.json").is_file() and list((out / "records").glob("*.json")) == []
    code = main_with_sigterm_kept(cells, batch_argv("rerun", cell, out, "--run", "no-such-run"))
    assert code == 2 and "no run" in capsys.readouterr().err
    unknown = batch_argv("run", cell, tmp_path / "other")
    unknown[unknown.index("--cell") + 1] = "no-such-cell"
    assert main_with_sigterm_kept(cells, unknown) == 2
    assert "no such cell" in capsys.readouterr().err


# --- (f) the level report ------------------------------------------------------------------


def level_record(cells, jobs, at_once, throughput, query, walk):
    return {"cell": middle_cell(cells).as_dict(), "jobs": jobs, "capture": True, "exit": 0,
            "at_once": at_once, "sampler": False,
            "figures": {"throughput_records_per_second": throughput, "query_seconds": query,
                        "proof_walk_seconds": walk, "build_folder_bytes": 5}}


def level_records(cells):
    records = []
    for thr, q, w in ((100.0, 1.0, 3.0), (104.0, 1.2, 3.4), (102.0, 1.1, 3.2)):  # jobs 1 alone: the band
        records.append(level_record(cells, 1, 1, thr, q, w))
    # jobs 1: levels 2, 3 and 4 stay inside the band (3 sits on its edge), 6 and 8 fall out of it.
    for level, (thr, q, w) in {2: (101.0, 1.1, 3.1), 3: (104.0, 1.2, 3.4), 4: (100.0, 1.0, 3.0),
                               6: (90.0, 1.1, 3.1), 8: (80.0, 2.0, 5.0)}.items():
        records.append(level_record(cells, 1, level, thr, q, w))
    for thr, q, w in ((200.0, 2.0, 6.0), (210.0, 2.4, 6.8), (205.0, 2.2, 6.4)):  # jobs 4 alone
        records.append(level_record(cells, 4, 1, thr, q, w))
    # jobs 4: 2 is inside; 3 loses on query time only, 4 on walk time only, 6 on throughput only.
    for level, (thr, q, w) in {2: (205.0, 2.2, 6.4), 3: (205.0, 5.0, 6.4), 4: (205.0, 2.2, 9.0),
                               6: (150.0, 2.2, 6.4)}.items():
        records.append(level_record(cells, 4, level, thr, q, w))
    return records


def test_the_level_report_names_the_highest_level_inside_the_alone_band_for_each_workers_value():
    cells = cells_module()
    report = cells.level_report(level_records(cells))
    assert report["label"] == "a measurement for the owner's rule, never a 5.6 figure"
    assert report["jobs"]["1"]["highest_level"] == 4
    assert report["jobs"]["4"]["highest_level"] == 2
    assert report["jobs"]["1"]["band"]["throughput_records_per_second"] == [100.0, 104.0]
    assert report["jobs"]["4"]["band"]["query_seconds"] == [2.0, 2.4]
    assert report["jobs"]["4"]["band"]["proof_walk_seconds"] == [6.0, 6.8]
    assert "build_folder_bytes" not in report["jobs"]["1"]["band"]  # bytes are not in the band


def test_the_level_report_names_the_heavier_cells_as_not_covered_and_says_what_covers_them():
    cells = cells_module()
    report = cells.level_report(level_records(cells))
    text = " ".join(report["not_covered"])
    assert "8000" in text and "10 sources" in text and "100k" in text
    assert isinstance(report["covered_by"], str) and report["covered_by"]


def test_the_level_report_refuses_with_fewer_than_three_alone_runs():
    cells = cells_module()
    records = [r for r in level_records(cells) if not (r["jobs"] == 4 and r["at_once"] == 1)]
    records.append(level_record(cells, 4, 1, 200.0, 2.0, 6.0))
    records.append(level_record(cells, 4, 1, 210.0, 2.4, 6.8))
    with pytest.raises(cells.LevelRefused) as caught:
        cells.level_report(records)
    assert "three" in str(caught.value) and "alone" in str(caught.value)
    with pytest.raises(cells.LevelRefused):
        cells.level_report([])


def test_the_level_command_answers_nothing_when_the_alone_runs_are_too_few(tmp_path, capsys):
    cells = cells_module()
    folder = tmp_path / "records"
    folder.mkdir()
    for n, record in enumerate(level_records(cells)[:2] + level_records(cells)[3:8]):
        (folder / f"r{n}.json").write_text(json.dumps(record), encoding="utf-8")
    code = cells.main(["level", "--records", str(folder)])
    captured = capsys.readouterr()
    assert code != 0 and captured.out == ""
    assert "LevelRefused" in captured.err


def test_the_level_command_prints_the_report_for_enough_alone_runs(tmp_path, capsys):
    cells = cells_module()
    folder = tmp_path / "records"
    folder.mkdir()
    for n, record in enumerate(level_records(cells)):
        (folder / f"r{n}.json").write_text(json.dumps(record), encoding="utf-8")
    assert cells.main(["level", "--records", str(folder)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["jobs"]["1"]["highest_level"] == 4 and printed["label"].startswith("a measurement for")
    assert cells.LEVELS == (2, 3, 4, 6, 8, 12, 16)


def test_the_grid_command_prints_the_counts(capsys):
    cells = cells_module()
    assert cells.main(["grid"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["cell_count"] == 20 and printed["run_count"] == 240


# --- Measuring helpers ------------------------------------------------------------------------


def test_peak_memory_is_the_largest_sum_over_the_samples_with_the_largest_single_process():
    cells = cells_module()
    samples = [[100, 50], [200, 90], [250]]  # sums 150, 290, 250
    assert cells.peak_from_samples(samples) == (290, 250)
    assert cells.SAMPLE_INTERVAL_SECONDS == 0.1


def test_the_sampler_sums_the_resident_sizes_of_the_whole_process_tree():
    cells = cells_module()
    child = "x = bytearray(60 * 1024 * 1024); x[::4096] = b'a' * len(x[::4096]); import time; time.sleep(1.2)"
    parent = ("import subprocess, sys; "
              f"subprocess.run([sys.executable, '-c', {child!r}], check=True)")
    result = cells.sample_command([sys.executable, "-c", parent])
    assert result["exit"] == 0 and result["wall_seconds"] >= 1.2
    assert len(result["samples"]) >= 2
    assert result["peak_sum_bytes"] == max(sum(s) for s in result["samples"])
    assert result["peak_largest_bytes"] == max(max(s) for s in result["samples"])
    assert result["peak_largest_bytes"] >= 60 * 1024 * 1024
    assert result["peak_sum_bytes"] > result["peak_largest_bytes"]  # parent and child both counted


def make_hand_store(path: Path) -> None:
    """Tables with the names and shapes of the store's (schema.sql) holding few rows, so that each
    b-tree fits on one 4096-byte page: the bytes of each table are counted by hand below."""
    db = sqlite3.connect(path)
    db.execute("PRAGMA page_size = 4096")
    db.executescript("""
        CREATE TABLE occurrence (occurrence_id INTEGER PRIMARY KEY, artifact_id INTEGER NOT NULL,
            binding_id INTEGER NOT NULL, locator TEXT NOT NULL, fingerprint BLOB,
            UNIQUE (artifact_id, locator));
        CREATE INDEX occurrence_fingerprint ON occurrence(fingerprint);
        CREATE TABLE edge_input (edge_id INTEGER NOT NULL, position INTEGER NOT NULL,
            occurrence_id INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 1,
            selected INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (edge_id, position)) WITHOUT ROWID;
        CREATE INDEX edge_input_occurrence ON edge_input(occurrence_id);
        CREATE TABLE step_trace (trace_id INTEGER PRIMARY KEY, segment_id INTEGER NOT NULL,
            step INTEGER NOT NULL, consumed_samples INTEGER NOT NULL, UNIQUE (segment_id, step));
        CREATE TABLE trace_sample (trace_id INTEGER NOT NULL, sample_pos INTEGER NOT NULL,
            global_index INTEGER NOT NULL, PRIMARY KEY (trace_id, sample_pos)) WITHOUT ROWID;
        CREATE TABLE trace_row (trace_id INTEGER NOT NULL, sample_pos INTEGER NOT NULL,
            row_pos INTEGER NOT NULL, ordinal INTEGER NOT NULL,
            PRIMARY KEY (trace_id, sample_pos, row_pos)) WITHOUT ROWID;
        CREATE INDEX trace_row_document ON trace_row(ordinal);
        INSERT INTO occurrence VALUES (1, 1, 1, 'L1', x'01'), (2, 1, 1, 'L2', x'02'), (3, 1, 1, 'L3', x'03');
        INSERT INTO edge_input VALUES (1, 0, 1, 1, 0), (1, 1, 2, 1, 0);
        INSERT INTO step_trace VALUES (1, 1, 1, 2);
        INSERT INTO trace_sample VALUES (1, 0, 5), (1, 1, 6);
        INSERT INTO trace_row VALUES (1, 0, 0, 3), (1, 1, 0, 4);
    """)
    db.commit()
    db.close()


def test_metadata_bytes_are_counted_by_dbstat_on_a_copy_and_the_store_is_not_touched(tmp_path):
    cells = cells_module()
    store = tmp_path / "store.sqlite"
    make_hand_store(store)
    before = hashlib.sha256(store.read_bytes()).hexdigest()
    copies = tmp_path / "copies"
    copies.mkdir()
    folder_before = sorted(p.name for p in tmp_path.iterdir())
    figures = cells.metadata_bytes(store, copies)
    # Pages of 4096 bytes, one per b-tree. occurrence: the table, its UNIQUE index and one index
    # = 3 pages for 3 records. edge_input: the table (without rowid) and one index = 2 pages for
    # 2 parent links. step_trace 2 pages, trace_sample 1, trace_row 2 = 5 pages for 1 stored trace.
    assert figures["per_record"] == pytest.approx(3 * 4096 / 3)
    assert figures["per_parent_link"] == pytest.approx(2 * 4096 / 2)
    assert figures["per_step_trace"] == pytest.approx(5 * 4096 / 1)
    assert figures["rows"] == {"records": 3, "parent_links": 2, "step_traces": 1}
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == folder_before  # no -journal, no -wal beside it
    assert any(p.is_file() for p in copies.iterdir())  # the count ran on a copy


def test_a_dbstat_that_the_sqlite_build_lacks_is_no_reason_to_skip_the_figure():
    cells_module()
    db = sqlite3.connect(":memory:")
    try:
        db.execute("SELECT count(*) FROM dbstat")
    except sqlite3.OperationalError as error:  # pragma: no cover
        pytest.fail(f"this sqlite3 has no dbstat: {error}")
    finally:
        db.close()


def test_repair_bytes_are_the_sizes_of_what_a_repair_reads_and_rewrites(tmp_path):
    cells = cells_module()
    historical = tmp_path / "historical"
    repaired = tmp_path / "repaired"
    historical.mkdir()
    repaired.mkdir()
    (historical / "a.bin").write_bytes(b"x" * 1000)
    (historical / "b.bin").write_bytes(b"x" * 500)
    (repaired / "c.bin").write_bytes(b"x" * 700)
    got = cells.repair_bytes(historical=historical, store_before=4096, store_after=4096 + 8192,
                             repaired=repaired)
    # reads: the historical build and the store as it stood; rewrites: the repaired build and the
    # store's growth.
    assert got == {"read_bytes": 1500 + 4096, "rewritten_bytes": 700 + 8192}


def test_the_capture_wiring_of_one_generation_job_is_a_static_line_count_with_places(repo):
    cells = cells_module()
    hooks = repo / "replica" / "capture_hooks.py"
    source = hooks.read_text(encoding="utf-8")
    tree = ast.parse(source)
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StoreCapture")
    lines = source.splitlines()
    expected = []
    for name in ("generate", "record_generation"):
        method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)
        doc = method.body[0]
        skip = set()
        if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) and isinstance(doc.value.value, str):
            skip = set(range(doc.lineno, doc.end_lineno + 1))
        for number in range(method.lineno, method.end_lineno + 1):
            text = lines[number - 1].strip()
            if number in skip or not text or text.startswith("#"):
                continue
            expected.append(number)
    wiring = cells.capture_wiring(repo)
    assert wiring["file"] == "replica/capture_hooks.py"
    assert wiring["methods"] == ["generate", "record_generation"]
    assert wiring["lines"] == expected
    assert wiring["count"] == len(expected) > 20
    build = (repo / "replica" / "build.py").read_text(encoding="utf-8").splitlines()
    sites = [n for n, text in enumerate(build, 1)
             if "capture.generate(" in text or "capture.record_generation(" in text]
    assert [(s["file"], s["line"]) for s in wiring["call_sites"]] == [("replica/build.py", n) for n in sites]
    assert len(sites) == 2


def test_the_digests_of_an_input_folder_are_per_file_and_the_tokenizer_digest_is_of_the_spec(tmp_path):
    cells = cells_module()
    folder = tmp_path / "set"
    (folder / "sub").mkdir(parents=True)
    (folder / "set.toml").write_bytes(b"alpha")
    (folder / "sub" / "A.jsonl").write_bytes(b"beta\n")
    expected = {"set.toml": hashlib.sha256(b"alpha").hexdigest(),
                "sub/A.jsonl": hashlib.sha256(b"beta\n").hexdigest()}
    assert cells.file_digests(folder) == expected
    spec = tmp_path / "spec.toml"
    spec.write_bytes(b"[tokenizer]\n")
    assert cells.tokenizer_digest(spec) == hashlib.sha256(b"[tokenizer]\n").hexdigest()


def test_the_shard_order_is_the_build_listing_with_digit_runs_read_as_numbers(tmp_path):
    cells = cells_module()
    for layout, names, expected in (
            ("round-robin", ["shard-10", "shard-2", "shard-1"], ["shard-1", "shard-2", "shard-10"]),
            ("partitioned", ["shard-gen", "shard-Z2", "shard-A"], ["shard-A", "shard-Z2", "shard-gen"])):
        shards = tmp_path / layout / "pipeline" / layout / "shards"
        shards.mkdir(parents=True)
        for name in names:
            (shards / f"{name}.bin").write_bytes(b"")
            (shards / f"{name}.idx").write_bytes(b"")
        assert cells.shard_order(tmp_path / layout, layout) == expected


# --- (h) one real miniature cell, end to end ---------------------------------------------------


def natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def folder_bytes(folder: Path) -> int:
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())


def digests_of(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


@pytest.mark.megatron
@pytest.mark.slow
def test_one_real_miniature_cell_runs_end_to_end_through_the_real_commands(
        tmp_path, repo, spec_path, spec_doc, plant_generate):
    cells = cells_module()
    from replica.megatron_tokenizer import ENV_VARIABLE

    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    # The smallest cell the generator allows: 100 records (one summary passage), two chunks, two
    # pairs and three sanitizer cases (docs/plant.md section 3).
    extra = ("--chunks", "2", "--pairs", "2", "--sanitizer-cases", "3")
    cell = cells.Cell(records=100, length=200, derivations=1, sources=4, layout="partitioned",
                      generator_extra=extra)
    off = cells.RunSpec(cell=cell, jobs=1, capture=False, pair=1)
    on = cells.RunSpec(cell=cell, jobs=1, capture=True, pair=1)
    # Nothing is stood in: the capture-on run measures its repair through the real commands too
    # (`waterwhite report withdraw A`, then `python -m replica repair`, docs/replica.md section 15).
    steps = cells.CommandSteps(repo=repo, python=sys.executable, tokenizer_spec=spec_path)
    machine = cells.read_machine(role="server", host_type="CPX62", tree="miniature")
    out = tmp_path / "batch"
    window = tmp_path / "TIMED_WINDOW"
    result = cells.run_batch([off, on], out, steps, at_once=1, window_file=window, machine=machine,
                             repo=repo)
    assert sorted(result.finished) == sorted([off.run_id, on.run_id]) and result.failed == []
    assert not window.exists()
    rec_off, rec_on = read_record(out, off.run_id), read_record(out, on.run_id)
    for record in (rec_off, rec_on):
        assert record["exit"] == 0 and record["machine"] == machine and record["sampler"] is False
        assert record["beside"] == {"at_start": 0, "maximum": 0}
        figures = record["figures"]
        assert figures["wall_seconds"] > 0
        assert figures["throughput_records_per_second"] == pytest.approx(100 / figures["wall_seconds"])
        folders = record["folders"]
        assert figures["build_folder_bytes"] == folder_bytes(Path(folders["build"]))
        for command in record["commands"]:
            assert command["exit"] == 0 and command["argv"]
            assert (out / command["output_file"]).is_file()
    # What the matched values are, from outside the run: the generator's own bytes for the same
    # parameters, the spec file, the docs' shard naming, and the spec's workers value.
    expected_set = tmp_path / "expected-set"
    plant_generate.generate(expected_set, records=100, seed=cells.SEED, length=200, derivations=1,
                            sources=4, dedup=True, chunks=2, pairs=2, sanitizer_cases=3)
    expected_matched_input = digests_of(expected_set)
    spec_digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    workers = spec_doc.get("megatron", {}).get("workers", 1)
    pair = json.loads((out / "pairs" / f"{off.pair_id}.json").read_text(encoding="utf-8"))
    assert pair["matched"]["input_sha256"] == expected_matched_input
    assert pair["matched"]["tokenizer_sha256"] == spec_digest
    order = pair["matched"]["shard_order"]
    assert order == sorted(order, key=natural_key)
    assert {"shard-A", "shard-Z", "shard-gen"} <= set(order)  # docs/replica.md: one shard per source, and shard-gen
    for record in (rec_off, rec_on):
        assert record["megatron_workers"] == workers
    assert rec_off["harness"] != "confined" and rec_on["harness"] == "in-process"
    # Off: no store, so the on-only figures are not applicable. On: all of them are numbers.
    for key in ("store_file_bytes", "query_seconds", "proof_walk_seconds", "metadata_bytes", "repair_bytes"):
        assert rec_off["figures"][key] == "not applicable"
    on_figures = rec_on["figures"]
    assert on_figures["store_file_bytes"] == os.path.getsize(rec_on["folders"]["store"]) > 0
    assert on_figures["query_seconds"] > 0 and on_figures["proof_walk_seconds"] > 0
    assert all(on_figures["metadata_bytes"][k] > 0 for k in ("per_record", "per_parent_link", "per_step_trace"))
    # The repair ran for real, on a copy of the store: what it read is the historical build and the store
    # as it stood, what it rewrote is the repaired build and the growth of the store's copy.
    work = Path(rec_on["folders"]["work"])
    store_before = os.path.getsize(rec_on["folders"]["store"])
    store_copy = work / "repair-store" / Path(rec_on["folders"]["store"]).name
    growth = max(os.path.getsize(store_copy) - store_before, 0)
    repaired = Path(rec_on["folders"]["repaired"])
    assert (repaired / "pipeline").is_dir(), "the repair wrote no build"
    assert on_figures["repair_bytes"] == {
        "read_bytes": folder_bytes(Path(rec_on["folders"]["build"])) + store_before,
        "rewritten_bytes": folder_bytes(repaired) + growth}
    assert on_figures["repair_bytes"]["read_bytes"] > 0 and on_figures["repair_bytes"]["rewritten_bytes"] > 0
    report = json.loads((work / "report.json").read_text(encoding="utf-8"))
    assert report["form"] == "waterwhite.report/1"  # docs/replica.md section 15
    kept = {c["name"]: c for c in rec_on["commands"]}
    assert str(store_copy) in kept["report"]["argv"]
    assert "withdraw" in kept["report"]["argv"] and "A" in kept["report"]["argv"]
    assert kept["report"]["exit"] == kept["repair"]["exit"] == 0
    assert (out / kept["report"]["output_file"]).is_file() and (out / kept["repair"]["output_file"]).is_file()
    assert "--historical" in kept["repair"]["argv"] and str(store_copy) in kept["repair"]["argv"]
    # The commands of the on run: the real forms of docs/replica.md section 9 and docs/cli.md.
    by_name = {c["name"]: c["argv"] for c in rec_on["commands"]}
    assert "--capture" in by_name["build-set"] and "--capture" not in {
        a for c in rec_off["commands"] for a in c["argv"]}
    assert "probe-run" in by_name
    query = by_name["query"]
    steps_run = int(flag_value(query, "--artifact").split("@")[1])
    assert steps_run > 0 and flag_value(query, "--artifact") == f"probe@{steps_run}"
    walk = by_name["proof-walk"]
    assert "--artifact" not in walk and walk[walk.index("query"):walk.index("query") + 3] == ["query", "withdraw", "A"]
    assert "probe-run" not in {c["name"] for c in rec_off["commands"]}


# --- The documents ---------------------------------------------------------------------------


def section_of(text: str, heading_part: str):
    parts = re.split(r"(?m)^(?=## )", text)
    for part in parts:
        first = part.split("\n", 1)[0]
        if first.startswith("## ") and heading_part.lower() in first.lower():
            return part
    return None


def new_entries(text: str):
    """The entries of the cell runner. Until they are numbered they carry the labels D-CELL-1,
    D-CELL-2, ... and never a D-number."""
    return [part for part in re.split(r"(?m)^(?=## D-)", text) if re.match(r"## D-CELL-\d+ ", part)]


def test_the_docs_have_a_section_the_cell_runner_with_the_commands_and_the_rules(repo):
    cells_module()
    text = (repo / "docs" / "replica.md").read_text(encoding="utf-8")
    section = section_of(text, "The cell runner")
    assert section is not None, "docs/replica.md has no section 'The cell runner'"
    lowered = section.lower()
    for needle in ("20 cells", "240", "--at-once", "--window-file", "in-process", "no bound", "not applicable",
                   "sampler", "--jobs", "query withdraw", "level", "not a 5.6 figure", "confined", "rk3",
                   "17 nov 2026", "--host-type", "megatron", "tokenizer.workers", "dbstat", "resume"):
        assert needle in lowered, f"the section does not say {needle!r}"
    assert "cpx" not in lowered, "the section names a fixed machine type: the record carries what --host-type gives"


def test_decisions_has_entries_for_the_harness_the_worker_meaning_the_240_and_the_server(repo):
    cells_module()
    entries = new_entries((repo / "DECISIONS.md").read_text(encoding="utf-8"))
    assert len(entries) >= 4, f"expected at least four new entries, found {len(entries)}"
    labels = [re.match(r"## D-CELL-(\d+) ", entry).group(1) for entry in entries]
    assert labels == [str(n) for n in range(1, len(labels) + 1)], labels
    for entry in entries:
        assert "Other option" in entry and "What would change it" in entry, entry.split("\n", 1)[0]
    joined = [e.lower() for e in entries]

    def one(*needles):
        return any(all(n in e for n in needles) for e in joined)

    assert one("harness", "in-process", "confined")
    assert one("--jobs", "workers", "megatron")
    assert one("240", "cells")
    assert one("rk3", "17 nov 2026", "--host-type", "server")
    assert not any("cpx" in e for e in joined), "an entry names a fixed machine type: the record carries what --host-type gives"


# The words no added line may carry (the words are cut in two here so that this file holds none).
BANNED = [
    chr(0x2014), "/ho" + "me/", "Pro" + "jects/", "C&" + "C", "Super" + "Builder", "Op" + "us", "Son" + "net",
    "advi" + "sor", "R2" + "A", "ROUND" + "2", "rul" + "ing", "amen" + "ded", "uns" + "ure", "BA" + "RS",
    "spec " + "section",
]
BANNED_PATTERNS = [r"\bG\d", r"\bQ\d", r"\bround \d", r"\bitem \d", r"\bspec" + r" 1\b"]
# Words of how the work was organised: banned in prose (docs, entries, comments), not in code names.
BANNED_PROSE = ["work" + "flow", "slic" + "e", "wav" + "e", "ag" + "ent", "bri" + "ef", "sta" + "ge"]


def banned_in(text: str, prose: bool = True):
    found = [word for word in BANNED + (BANNED_PROSE if prose else []) if word in text]
    found += [p for p in BANNED_PATTERNS if re.search(p, text)]
    return found


def test_the_new_docs_and_entries_carry_none_of_the_banned_words(repo):
    cells_module()
    docs = (repo / "docs" / "replica.md").read_text(encoding="utf-8")
    section = section_of(docs, "The cell runner")
    assert section is not None
    assert banned_in(section) == []
    entries = new_entries((repo / "DECISIONS.md").read_text(encoding="utf-8"))
    assert entries, "DECISIONS.md has no entry of the cell runner to read"
    for entry in entries:
        assert banned_in(entry) == [], entry.split("\n", 1)[0]
    source = repo / "replica" / "cells.py"
    assert source.is_file()
    assert banned_in(source.read_text(encoding="utf-8"), prose=False) == []
