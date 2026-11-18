"""The build option that runs the generation jobs in the confined harness, and the
command line's store path (plan 3.3 "Generation", F10; docs/confine.md).

The confined runs need Linux and a bubblewrap that can create user namespaces.
Every test that starts one carries the marker "sandbox" and asks for the fixture
``sandbox`` (conftest.py); where the sandbox cannot be built it is skipped with
the harness's own reason and is not worked around. The refusal itself is tested
on every machine.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from waterwhite.store.store import Store

from replica import cli
from replica.build import BuildOptions, build
from replica.errors import CaptureError, SpecError
from replica.fixture import build_captured_fixture

JOBS = ["paraphrase", "translation", "summary"]
SELECTION = dict(layouts=(), releases=("R1",), imports=())


def without_harness_word(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "harness"}


@pytest.mark.sandbox
def test_the_stores_records_are_the_same_with_the_confined_harness(sandbox, tmp_path):
    plain = build_captured_fixture(tmp_path / "in-process", **SELECTION)
    confined = build_captured_fixture(tmp_path / "confined", generation_harness="confined",
                                      **SELECTION)
    assert confined.options.generation_harness == "confined"
    # An instance id covers the content, the step, the inputs and the stated
    # edges: the same ids mean the same files and the same parent lists.
    assert confined.instances == plain.instances
    with Store.open(plain.store, readonly=True) as a, Store.open(confined.store,
                                                                readonly=True) as b:
        for job in JOBS:
            first, second = a.harness_log(job), b.harness_log(job)
            # A log names the harness that wrote it (DECISIONS.md D-30), so the
            # log's digest differs; every other field of the stored log is equal.
            assert (first.harness, second.harness) == ("in-process", "confined")
            assert first.log_digest != second.log_digest
            assert bare_log(second) == bare_log(first)
            assert second.model == "none"
            assert second.bound_established() and not first.bound_established()
        for name in ("gen/P", "gen/T", "gen/S", "R1"):
            instance = plain.instances[name]
            assert [bare_edge(e) for e in b.edges(instance)] == [
                bare_edge(e) for e in a.edges(instance)]
            for in_process, confined_edge in zip(a.edges(instance), b.edges(instance)):
                if confined_edge.is_generation:
                    assert (in_process.bound_field, confined_edge.bound_field) == (False, True)
            assert list(b.occurrences(instance)) == list(a.occurrences(instance))
        b.check_integrity()
    # The log files differ in the harness word of the job entry and in nothing
    # else: the confined job made the same requests in the same order.
    for output in ("P", "T", "S"):
        relative = f"gen/harness/{output}/{output}.log.jsonl"
        in_process = read_log(plain.out / relative)
        sandboxed = read_log(confined.out / relative)
        assert [e.get("harness") for e in in_process if e["kind"] == "job"] == ["in-process"]
        assert [e.get("harness") for e in sandboxed if e["kind"] == "job"] == ["confined"]
        assert [without_harness_word(e) for e in sandboxed] == [
            without_harness_word(e) for e in in_process]
        assert (confined.out / f"gen/{output}.jsonl").read_bytes() == (
            plain.out / f"gen/{output}.jsonl").read_bytes()


def bare_log(log):
    """A stored log without the two fields that name the harness."""
    return None if log is None else replace(log, log_digest=None, harness=None)


def bare_edge(edge):
    """An edge as the store holds it, its log read without its digest and harness
    word, and without its stored bound field: an edge cites the log of its job, the
    log names its harness, and only a confined log establishes a bound
    (Decision 13), so the field is True for the confined build and False for the
    in-process one."""
    return replace(edge, harness_log=bare_log(edge.harness_log), bound_field=None)


def read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_confined_harness_is_refused_by_name_where_the_sandbox_is_unavailable(
        tmp_path, monkeypatch):
    # No bubblewrap on the path: the build is refused with the tool's own error
    # name before anything is written, and no job runs unconfined in its place.
    monkeypatch.setenv("PATH", str(tmp_path / "no-such-folder"))
    with pytest.raises(CaptureError, match="^sandbox unavailable: "):
        build_captured_fixture(tmp_path / "f", generation_harness="confined", **SELECTION)
    assert not (tmp_path / "f").exists()


def test_the_confined_harness_needs_a_captured_generation_step(fixture_plan, tmp_path):
    with pytest.raises(SpecError, match="does not capture its generation step"):
        build(fixture_plan, tmp_path / "out",
              BuildOptions(tokenize=False, generation_harness="confined"))
    with pytest.raises(SpecError, match="does not capture its generation step"):
        build_captured_fixture(tmp_path / "f", generation_harness="confined",
                               capture_generation=False, layouts=(), releases=(), imports=())
    with pytest.raises(SpecError, match="in-process or confined"):
        build(fixture_plan, tmp_path / "out",
              BuildOptions(tokenize=False, generation_harness="sandboxed"))
    assert not (tmp_path / "out").exists() and not (tmp_path / "f").exists()


# --- The command line's store path ---------------------------------------------------------


def run_cli(spec_path, out, *more) -> int:
    return cli.main(["build-fixture", "--spec", str(spec_path), "--out", str(out),
                     "--no-tokenize", "--layout", "partitioned", *more])


def test_a_store_path_inside_the_build_folder_is_refused_and_leaves_no_file(spec_path, tmp_path,
                                                                           capsys):
    out = tmp_path / "out"
    # The store file used to be created before the build's test for a new or
    # empty folder, so the build refused its own folder and left the store behind.
    assert run_cli(spec_path, out, "--capture", "--store", str(out / "store.sqlite")) == 1
    said = capsys.readouterr()
    assert said.out == "" and "OutputError" in said.err
    assert "lies inside the build folder" in said.err
    assert not out.exists()
    # Beside the build folder the same build runs.
    assert run_cli(spec_path, out, "--capture", "--store", str(tmp_path / "store.sqlite")) == 0
    assert (tmp_path / "store.sqlite").is_file() and (out / "releases" / "R1.jsonl").is_file()


def test_a_refused_build_leaves_no_store_file_behind(spec_path, tmp_path, capsys):
    used = tmp_path / "used"
    used.mkdir()
    (used / "x").write_text("x", encoding="utf-8")
    store = tmp_path / "stores" / "store.sqlite"
    assert run_cli(spec_path, used, "--capture", "--store", str(store)) == 1
    assert "OutputError" in capsys.readouterr().err
    assert not store.exists() and not store.parent.exists()
    # A release the spec does not hold is refused before the store is made too.
    assert run_cli(spec_path, tmp_path / "new", "--capture", "--store", str(store),
                   "--release", "R9") == 1
    assert "SpecError" in capsys.readouterr().err
    assert not store.exists() and not (tmp_path / "new").exists()


def test_the_command_line_takes_the_harness_of_the_generation_jobs(spec_path, tmp_path, capsys):
    # Confined needs capture: without it the build is refused by name.
    assert run_cli(spec_path, tmp_path / "out", "--generation-harness", "confined") == 1
    assert "does not capture its generation step" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_store_of_the_table_runs_its_generation_jobs_confined_by_default(sandbox, tmp_path):
    """The table's rows compare answers on its stores, and only a log the confined
    harness wrote establishes a bound (Decision 13), so build_table_store runs a
    store's generation jobs confined unless told otherwise."""
    from replica.fixture import build_table_store

    built = build_table_store(tmp_path / "main", "main")
    with Store.open(built.store, readonly=True) as store:
        for job in JOBS:
            log = store.harness_log(job)
            assert log.harness == "confined" and log.bound_established()
        (edge,) = store.edges(built.instances["gen/P"])
        assert edge.bound_field is True
