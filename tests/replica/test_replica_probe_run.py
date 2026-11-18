"""``python -m replica probe-run``: one run read with the sequential sampler over the
shards a generated-set build wrote, registered in the store the set was built into
(docs/replica.md section 14; plan 3.3 "Training", 5.4, 5.6).

The refusal tests run in a second and need no Megatron. The run itself starts
Megatron's own dataset code, carries the markers ``megatron`` and ``slow``, and is
skipped unless ``WATERWHITE_MEGATRON_ENV`` is set; run it through the CPU lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica.megatron_tokenizer import ENV_VARIABLE
from replica.run_plan import load_runs_plan
from replica.sets import build_captured_set
from replica.shard_index import read_shard_index

SEQUENCE_LENGTH = 32  # run E of the fixture spec
GLOBAL_BATCH = 4
CHUNKS = 12  # twelve shards, named shard-01 to shard-12


def _probe(repo, *args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo)]))
    return subprocess.run([sys.executable, "-m", "replica", "probe-run", *args],
                          capture_output=True, text=True, env=env, cwd=repo, timeout=1500)


def _sequential_join(spec_path):
    plan = load_runs_plan(spec_path)
    return (plan.release_tag, "single", plan.sampler_classes["single"])


def _cyclic_join(spec_path):
    plan = load_runs_plan(spec_path)
    return (plan.release_tag, "cyclic", plan.sampler_classes["cyclic"])


# --- The refusal: the store's list is the build's (D-37, D-38) ---------------------------------


def _store_without_the_sequential_join(tmp_path, spec_path):
    """A store that lists the cyclic join only, and a set build whose shard files are
    placeholders: the refusal comes before anything is read or made."""
    store = tmp_path / "store.sqlite"
    Store.create(store, normaliser="control-chars/1",
                 verified_joins=[_cyclic_join(spec_path)]).close()
    shards = tmp_path / "build" / "pipeline" / "round-robin" / "shards"
    shards.mkdir(parents=True)
    for suffix in (".bin", ".idx"):
        (shards / f"shard-1{suffix}").write_bytes(b"")
    return store, tmp_path / "build"


def test_a_store_that_does_not_list_the_sequential_join_is_refused_and_nothing_is_written(
        tmp_path, repo, spec_path):
    store, build = _store_without_the_sequential_join(tmp_path, spec_path)
    out = tmp_path / "probe"
    done = _probe(repo, "--set-build", str(build), "--store", str(store), "--out", str(out))
    assert done.returncode == 1, done.stdout + done.stderr
    tag, kind, cls = _sequential_join(spec_path)
    assert done.stderr.startswith("replica: RunError: ")
    assert "lists no verified join" in done.stderr and tag in done.stderr
    assert f"'{kind}'" in done.stderr and cls in done.stderr
    assert done.stdout == ""
    assert not out.exists()
    with Store.open(store, readonly=True) as held:
        assert held.verified_joins() == (_cyclic_join(spec_path),)
        assert list(held.artifacts()) == [] and held.runs() == ()


def test_the_python_call_refuses_the_same_way(tmp_path, spec_path):
    from replica.errors import RunError
    from replica.probe_run import probe_run

    store, build = _store_without_the_sequential_join(tmp_path, spec_path)
    with pytest.raises(RunError, match="lists no verified join"):
        probe_run(build, store, tmp_path / "probe")
    assert not (tmp_path / "probe").exists()


def test_the_probe_imports_no_oracle_no_join_and_no_torch(repo):
    code = ("import sys\nsys.modules['torch'] = None\nimport replica.cli, replica.probe_run\n"
            "bad = [m for m in sys.modules if m == 'oracle' or m.startswith('oracle.')"
            " or m == 'waterwhite.join' or m.startswith('waterwhite.join.')]\n"
            "assert not bad, bad\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(repo / "src"), str(repo)]))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=repo)
    assert done.returncode == 0, done.stderr


# --- The run --------------------------------------------------------------------------------


@pytest.mark.megatron
@pytest.mark.slow
def test_probe_run_registers_one_traced_run_over_the_small_set(tmp_path, repo, spec_path,
                                                               plant_generate, set_parameters):
    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    set_dir = tmp_path / "set"
    plant_generate.generate(set_dir, **{**set_parameters, "chunks": CHUNKS})
    built = build_captured_set(tmp_path / "captured", set_dir, parallel=4,
                               verified_joins=(_sequential_join(spec_path),))
    layout = built.summary["layouts"]["round-robin"]
    expected_names = [row["store_name"] for row in layout["shards"] if not row["empty"]]
    assert expected_names == [f"set-rr/shard-{n:02d}" for n in range(1, CHUNKS + 1)]

    done = _probe(repo, "--set-build", str(built.out), "--store", str(built.store),
                  "--out", str(tmp_path / "probe"))
    assert done.returncode == 0, done.stderr[-3000:]
    printed = json.loads(done.stdout)
    assert sorted(printed) == ["mixture", "register_run", "seconds", "segment", "shards", "steps"]
    assert sorted(printed["seconds"]) == ["mixture", "register", "run"]
    assert all(isinstance(value, float) and value > 0 for value in printed["seconds"].values())

    # register-run's own word: the route is the kept cache files, and no cause.
    registered = printed["register_run"]
    assert (registered["route"], registered["cause"]) == ("cache files", None)
    assert registered["segment"] == printed["segment"]

    # The steps are the set's one pass, in whole steps of the run's global batch size.
    samples = sum((read_shard_index(f"{built.file(name)}.idx").tokens - 1) // SEQUENCE_LENGTH
                  for name in expected_names)
    assert printed["steps"] == samples // GLOBAL_BATCH and printed["steps"] > 0
    assert printed["shards"] == CHUNKS

    with Store.open(built.store, readonly=True) as store:
        segment = store.segment(printed["segment"])
        assert segment is not None and segment.trace_route == v.ROUTE_CACHE_FILES
        # The segment holds as many traced steps as steps, and none is untraced.
        traced = list(store.step_traces(printed["segment"]))
        assert len(traced) == printed["steps"] == registered["steps"]
        assert registered["traced_steps"] == printed["steps"]
        assert registered["untraced_steps"] == []
        # The mixture holds the set's shards, in the order the run reads them.
        mixture = store.resolve(printed["mixture"])
        assert mixture is not None and mixture.kind == v.KIND_MIXTURE
        assert segment.mixture_instance_id == mixture.instance_id
        assert [entry.shard_instance_id for entry in store.mixture_shards(mixture.instance_id)] == [
            built.instances[name] for name in expected_names]
        assert store.verified_joins() == (_sequential_join(spec_path),)
        store.check_integrity()
