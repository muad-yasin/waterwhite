"""The replica's run wrapper (plan 3.3 "Training", 5.4, 3.5; docs/replica.md, "The runs").

Two kinds of test:

- plain ones (no marker, a few seconds): the spec's runs as the replica reads them,
  the step record, the release tag read from a checkout, the order and arguments of
  the registration commands, and that the command line imports with no torch;
- the Megatron ones (markers ``megatron`` and ``slow``, skipped unless
  ``WATERWHITE_MEGATRON_ENV`` is set): the runs are made once with Megatron's own
  dataset code, and each segment's record is compared with the oracle's record of the
  same segment, made by the oracle's own script over the same fixture build. Run them
  one hold at a time through the CPU lock.

The oracle is the comparison side. Nothing of it is imported by the replica; this file
starts its script as a process and reads the files it wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from replica import megatron_tokenizer
from replica.errors import OutputError, RunError, SpecError
from replica.megatron_tokenizer import ENV_VARIABLE
from replica.register_runs import register_run_available, register_runs
from replica.run_plan import load_runs_plan
from replica.runs import (
    CACHE_SUFFIX,
    RUN_RECORD_SUFFIX,
    STEP_RECORD_SUFFIX,
    build_runs,
    observed_release_tag,
    segment_paths,
    step_record,
)

SEGMENTS = ("E/seg-1", "E/seg-2", "E-rr/seg-1", "E-rr/seg-2", "D/seg-1", "Y/seg-1", "K/seg-1")
RUN_ORDER = ["E", "E-rr", "D", "Y", "K"]
ORACLE_FOLDER = "runs"  # under the oracle's output folder: runs/<run>/<segment folder>/


@pytest.fixture(scope="module")
def plan(spec_path):
    return load_runs_plan(spec_path)


# --- Plain tests ---------------------------------------------------------------------------


def test_the_spec_gives_seven_segments_in_run_order(plan):
    assert [s.name for s in plan.segments] == list(SEGMENTS)
    assert plan.runs() == RUN_ORDER
    by_name = {s.name: s for s in plan.segments}
    # E-rr copies the settings of E, seed included; K is the cyclic run with two launches.
    assert by_name["E-rr/seg-1"].settings == by_name["E/seg-1"].settings
    assert by_name["E/seg-1"].settings.data_sharding is None  # the spec does not state it
    k = by_name["K/seg-1"]
    assert (k.settings.dataloader_type, k.settings.data_sharding) == ("cyclic", True)
    assert [(l.position, l.consumed_at_start, l.step_at_start, l.first_step, l.last_step)
            for l in k.launches] == [(0, 0, 0, 1, 3), (1, 12, 3, 4, 40)]
    assert by_name["E/seg-2"].launches[0].consumed_at_start == 24
    assert by_name["E/seg-2"].launches[0].step_at_start == 6
    # The segment's files take its name with the slash replaced by a dash.
    assert by_name["E-rr/seg-2"].stem == "E-rr-seg-2"


def test_the_step_record_lists_every_completed_step_and_lowers_one(plan):
    by_name = {s.name: s for s in plan.segments}
    record = step_record(by_name["E/seg-2"])
    assert [row["step"] for row in record["steps"]] == [7, 8, 9, 10, 11, 12]
    assert {row["batch_size"] for row in record["steps"]} == {4}
    lowered = [row for row in record["steps"] if "last_sample_tokens_read" in row]
    assert lowered == [{"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}]
    assert record["checkpoints"] == [{"name": "E@9", "step": 9}]
    # A resume inside the cyclic run's second launch counts on from the first.
    assert [row["step"] for row in step_record(by_name["K/seg-1"])["steps"]] == list(range(1, 41))


def _fake_checkout(root: Path, *, commit: str, tags: list[str]) -> Path:
    git = root / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(commit + "\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled\n"
        + "".join(f"{'0' * 40} refs/tags/{t}\n^{commit}\n" for t in tags),
        encoding="utf-8")
    return root


def _env_with_checkout(tmp_path, **kwargs):
    root = tmp_path / "env"
    _fake_checkout(root / "src" / "Megatron-LM", **kwargs)
    return megatron_tokenizer.MegatronEnv(root)


def test_the_release_tag_is_read_from_the_checkout_and_must_agree_with_the_install(tmp_path):
    commit = "685fb8b" + "1" * 33
    env = _env_with_checkout(tmp_path, commit=commit, tags=["core_v0.20.0"])
    assert observed_release_tag(env, "core_v0.20.0", "0.20.0+685fb8b") == "core_v0.20.0"
    with pytest.raises(RunError, match="pins the Megatron release"):
        observed_release_tag(env, "core_v0.19.0", "0.20.0+685fb8b")
    with pytest.raises(RunError, match="is not built from the checkout's commit"):
        observed_release_tag(env, "core_v0.20.0", "0.20.0+abcdef0")
    with pytest.raises(RunError, match="installed megatron-core is 0.21.0"):
        observed_release_tag(env, "core_v0.20.0", "0.21.0+685fb8b")


def test_a_checkout_named_by_two_tags_or_none_is_refused(tmp_path):
    commit = "685fb8b" + "1" * 33
    two = _env_with_checkout(tmp_path / "a", commit=commit, tags=["core_v0.20.0", "other"])
    none = _env_with_checkout(tmp_path / "b", commit=commit, tags=[])
    for env in (two, none):
        with pytest.raises(RunError, match="exactly one"):
            observed_release_tag(env, "core_v0.20.0", "0.20.0+685fb8b")


def test_a_runs_folder_that_holds_files_is_refused_and_so_is_an_unknown_run(tmp_path, spec_path):
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file").write_text("x", encoding="utf-8")
    with pytest.raises(OutputError, match="not empty"):
        build_runs(tmp_path / "taken", spec_path=spec_path)
    with pytest.raises(SpecError, match="each named once"):
        build_runs(tmp_path / "new", spec_path=spec_path, runs=["Q"])


def test_the_command_line_and_the_wrapper_import_with_no_torch(repo):
    code = ("import sys\nsys.modules['torch'] = None\n"
            "import replica.cli, replica.runs, replica.register_runs, replica.run_plan\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo)]))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=repo)
    assert done.returncode == 0, done.stderr


class _Recorder:
    """A stand-in for ``subprocess.run``: it keeps each command line and prints what
    the tool's command prints, so the order and the arguments can be checked without
    the command."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        name = argv[argv.index("--run-record") + 1]
        segment = Path(name).name.removesuffix(RUN_RECORD_SUFFIX)
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps({"segment": segment}),
                                           stderr="")


def test_registration_starts_one_command_per_segment_in_run_order(plan, tmp_path):
    recorder = _Recorder()
    printed = register_runs(tmp_path / "store.sqlite", tmp_path / "runs", plan,
                            command=["tool"], runner=recorder)
    order = [call[call.index("--run-record") + 1] for call in recorder.calls]
    assert [Path(p).name for p in order] == [
        "E-seg-1.run.json", "E-seg-2.run.json", "E-rr-seg-1.run.json", "E-rr-seg-2.run.json",
        "D-seg-1.run.json", "Y-seg-1.run.json", "K-seg-1.run.json"]
    assert len(printed) == 7
    # Every parent checkpoint is registered by a segment that came before it.
    registered, parents = set(), {}
    for segment in plan.segments:
        registered.update(name for name, _ in segment.checkpoints)
        parents[segment.name] = (segment.parent_checkpoint, set(registered))
    for segment in plan.segments:
        parent, before = parents[segment.name]
        assert parent is None or parent in before
    first, second = recorder.calls[0], recorder.calls[1]
    assert first[:2] == ["tool", "register-run"]
    assert "--parent-checkpoint" not in first          # E/seg-1 starts fresh
    assert second[second.index("--parent-checkpoint") + 1] == "E@6"
    assert second[second.index("--mixture") + 1] == "E/mix-2"
    assert second[second.index("--steps") + 1].endswith("E-seg-2" + STEP_RECORD_SUFFIX)
    assert second[second.index("--cache") + 1].endswith("E-seg-2" + CACHE_SUFFIX)


def test_a_failing_command_stops_the_registration_with_its_words(plan, tmp_path):
    def refuses(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 4, stdout="", stderr="write refused")

    with pytest.raises(RunError, match="code 4: write refused"):
        register_runs(tmp_path / "s", tmp_path / "r", plan, command=["tool"], runner=refuses)


# --- The Megatron tests ----------------------------------------------------------------------

def _need_megatron():
    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def made(tmp_path_factory, spec_path, repo):
    """The runs made once by the replica's wrapper (the fixture build tokenized and
    all), and the oracle's own records of the same segments over the same build."""
    _need_megatron()
    root = tmp_path_factory.mktemp("runs")
    out = root / "replica-runs"
    summary = build_runs(out, spec_path=spec_path)
    fixture = out / ".work" / "fixture-build"
    oracle_out = root / "oracle-runs"
    env = megatron_tokenizer.MegatronEnv.from_environment()
    done = subprocess.run(
        [str(env.python), "-I", str(repo / "oracle" / "fixture_runs.py"), "--spec", str(spec_path),
         "--fixture", str(fixture), "--out", str(oracle_out)],
        capture_output=True, text=True, env=env.child_environment(), timeout=600)
    assert done.returncode == 0, done.stderr[-3000:]
    return out, oracle_out, summary


def _oracle_folder(oracle_out: Path, segment: str) -> Path:
    run, _, folder = segment.partition("/")
    return oracle_out / ORACLE_FOLDER / run / folder


def _paths(out: Path, plan, name: str):
    segment = next(s for s in plan.segments if s.name == name)
    return segment, segment_paths(out, segment)


@pytest.mark.megatron
@pytest.mark.slow
@pytest.mark.parametrize("name", SEGMENTS)
def test_the_replicas_run_record_equals_the_oracles_field_by_field(made, plan, name):
    out, oracle_out, _ = made
    _, paths = _paths(out, plan, name)
    ours = _read(paths["run_record"])
    theirs = _read(_oracle_folder(oracle_out, name) / "run_record.json")
    assert sorted(ours) == sorted(theirs)
    # The one field the oracle cannot write the same: a shard prefix is relative to
    # the record's own folder, and the two records lie in different folders. Compared
    # instead: the shard positions and the digests of both files.
    for record in (ours, theirs):
        assert [sorted(row) for row in record["shards"]] == [
            ["bin_sha256", "idx_sha256", "position", "prefix"]] * len(record["shards"])
    strip = lambda rows: [{k: v for k, v in row.items() if k != "prefix"} for row in rows]  # noqa: E731
    assert strip(ours["shards"]) == strip(theirs["shards"])
    for field in sorted(set(ours) - {"shards"}):
        assert ours[field] == theirs[field], f"{name}: field {field!r} differs from the oracle's"
    assert ours["format"] == "waterwhite.run-record/1"


@pytest.mark.megatron
@pytest.mark.slow
@pytest.mark.parametrize("name", SEGMENTS)
def test_the_replicas_step_record_equals_the_oracles(made, plan, name):
    out, oracle_out, _ = made
    _, paths = _paths(out, plan, name)
    assert _read(paths["step_record"]) == _read(_oracle_folder(oracle_out, name) / "step_record.json")


def _array_digest(array) -> str:
    """The digest ``docs/interfaces.md`` 8.1 defines for an index array, written
    again here from its definition (canonical JSON of the domain tag, dtype and
    shape, a zero byte, the bytes in C order), not taken from the replica."""
    import numpy

    head = json.dumps(["ww.array.v1", array.dtype.str, list(array.shape)], sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(head + b"\x00" + numpy.ascontiguousarray(array).tobytes()).hexdigest()


@pytest.mark.megatron
@pytest.mark.slow
@pytest.mark.parametrize("name", SEGMENTS)
def test_the_cache_digests_equal_digests_of_the_files_megatron_kept(made, plan, name):
    out, _, _ = made
    _, paths = _paths(out, plan, name)
    record = _read(paths["run_record"])
    import numpy

    kept = sorted(paths["cache"].glob("*.npy"))
    shards = len(record["shards"])
    blend = 2 if shards > 1 else 0
    assert len(kept) == 3 * shards + blend
    assert len(list(paths["cache"].glob("*description.txt"))) == shards + (1 if blend else 0)
    from_files: dict[str, list[str]] = {}
    for path in kept:
        kind = path.name.rsplit("-", 1)[1].removesuffix(".npy")
        from_files.setdefault(kind, []).append(_array_digest(numpy.load(path, allow_pickle=False)))
    from_record: dict[str, list[str]] = {}
    for row in record["cache_digests"]:
        from_record.setdefault(row["kind"], []).append(row["digest"])
    assert {k: sorted(v) for k, v in from_files.items()} == {k: sorted(v) for k, v in from_record.items()}
    # Three kinds per shard position, then the two of the blend, whose shard is null.
    assert [(r["kind"], r["shard"]) for r in record["cache_digests"]] == (
        [(kind, position) for position in range(shards)
         for kind in ("document_index", "sample_index", "shuffle_index")]
        + ([("dataset_index", None), ("dataset_sample_index", None)] if blend else []))


@pytest.mark.megatron
@pytest.mark.slow
def test_the_shard_files_a_record_names_lie_in_its_run_folder(made, plan):
    out, _, _ = made
    for segment in plan.segments:
        paths = segment_paths(out, segment)
        record = _read(paths["run_record"])
        assert [row["position"] for row in record["shards"]] == list(range(len(segment.shard_files)))
        for row, shard in zip(record["shards"], segment.shard_files):
            assert row["prefix"] == f"shards/{shard}"
            base = paths["run_record"].parent / row["prefix"]
            for suffix, key in ((".bin", "bin_sha256"), (".idx", "idx_sha256")):
                data = Path(str(base) + suffix).read_bytes()
                assert hashlib.sha256(data).hexdigest() == row[key]


@pytest.mark.megatron
@pytest.mark.slow
def test_the_cyclic_run_records_the_whole_epoch_of_every_launch_and_rank(made, plan):
    out, _, _ = made
    _, paths = _paths(out, plan, "K/seg-1")
    record = _read(paths["run_record"])
    rows = record["shuffle_digests"]
    # 57 samples in the blend, a bucket of 28 for each of two ranks, 14 steps an epoch:
    # launch 0 (steps 1 to 3) touches epoch 0, launch 1 (steps 4 to 40) epochs 0, 1, 2.
    assert [(r["launch"], r["epoch"], r["rank"]) for r in rows] == [
        (0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1), (1, 2, 0), (1, 2, 1)]
    assert {r["count"] for r in rows} == {28}
    by_key = {(r["launch"], r["epoch"], r["rank"]): r["digest"] for r in rows}
    # A resume inside an epoch records that epoch's whole order: the same digest as the launch before.
    assert by_key[(0, 0, 0)] == by_key[(1, 0, 0)] and by_key[(0, 0, 1)] == by_key[(1, 0, 1)]
    assert len(set(by_key.values())) == 6  # another epoch or rank is another order
    sequential = _read(_paths(out, plan, "E/seg-2")[1]["run_record"])
    assert sequential["shuffle_digests"] == []


@pytest.mark.megatron
@pytest.mark.slow
def test_each_run_has_the_files_laid_out_as_the_interface_says(made, plan):
    out, _, summary = made
    assert summary["runs"] == RUN_ORDER
    # The names are written out here, as docs/interfaces.md names them, and not taken
    # from the replica.
    expected = {
        "E": ["E-seg-1", "E-seg-2"], "E-rr": ["E-rr-seg-1", "E-rr-seg-2"],
        "D": ["D-seg-1"], "Y": ["Y-seg-1"], "K": ["K-seg-1"]}
    for run, stems in expected.items():
        names = sorted(p.name for p in (out / run).iterdir())
        assert names == sorted(
            ["shards"] + [stem + suffix for stem in stems
                          for suffix in (".run.json", ".steps.json", ".cache")])
    assert sorted(p.name for p in (out / "E" / "shards").iterdir()) == [
        f"shard-{n}.{ext}" for n in (1, 2, 3) for ext in ("bin", "idx")]
    assert sorted(p.name for p in (out / "K" / "shards").iterdir()) == [
        f"shard-{n}.{ext}" for n in (2, 3) for ext in ("bin", "idx")]


@pytest.mark.megatron
@pytest.mark.slow
def test_register_run_takes_every_segment_of_the_runs(tmp_path, spec_path, plan):
    """Through the tool's own command, against the stores of the table: the `main`
    store holds the partitioned layout (runs E, D, Y and K), the `rr` store the
    round-robin one (run E-rr). Skipped where the command is absent; the order and the
    arguments are tested above without it. Each segment is registered with the route
    of its kept cache files, and the steps it leaves untraced are the ones the spec's
    step record lowers (the replica's stores list the joins their runs use as verified,
    so no segment is cut off by an unverified version)."""
    _need_megatron()
    if not register_run_available():
        pytest.skip("not built yet: register-run")
    from replica.fixture import build_table_store

    for store_name, runs in (("main", ["E", "D", "Y", "K"]), ("rr", ["E-rr"])):
        built = build_table_store(tmp_path / store_name, store_name, tokenize=True, parallel=3,
                                  generation_harness="in-process")
        out = tmp_path / f"runs-{store_name}"
        build_runs(out, built.out, spec_path=spec_path, runs=runs)
        printed = register_runs(built.store, out, plan, runs=runs)
        wanted = [s for s in plan.segments if s.run in runs]
        assert [row["segment"] for row in printed] == [s.name for s in wanted]
        for segment, row in zip(wanted, printed):
            assert (row["route"], row["cause"]) == ("cache files", None), segment.name
            assert row["steps"] == len(segment.steps()), segment.name
            assert row["untraced_steps"] == [[segment.name, step] for step, _ in segment.lowered], \
                segment.name
            assert row["traced_steps"] >= row["steps"] - len(segment.lowered), segment.name
