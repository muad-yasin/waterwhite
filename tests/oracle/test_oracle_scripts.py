"""The oracle's entry scripts, each run as its own process with the environment's
interpreter and ``-I``, from a folder that is not the repository.

These are the checks that need a fresh process: that a second process gives the
same order digests for a cyclic run, and that every script takes a JSON config,
writes its files and prints its result alone on standard output. The fixture runs
and the seed search take options instead of a config file; they are run the same
way.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.megatron, pytest.mark.slow]

ORACLE = Path(__file__).resolve().parents[2] / "oracle"
SECONDS = 50


def write_config(folder: Path, name: str, raw: dict) -> Path:
    path = folder / name
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return path


def start(corpus, script: str, config: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [str(corpus.python), "-I", str(ORACLE / script), str(config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=config.parent,
        env=dict(os.environ),
    )


def finish(process: subprocess.Popen) -> tuple[int, str, str]:
    try:
        out, err = process.communicate(timeout=SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()  # this test's own child, by its handle
        pytest.fail(f"{process.args[2]} did not end in {SECONDS} s")
    return process.returncode, out, err


def result_of(process: subprocess.Popen) -> dict:
    code, out, err = finish(process)
    assert code == 0, err[-2000:]
    return json.loads(out)  # standard output holds the result and nothing else


def test_cyclic_order_digests_repeat_in_a_fresh_process(corpus, segment_config):
    """Four processes, each driving MegatronPretrainingRandomSampler at
    data-parallel size 2 with sharding on over two epochs, each into its own new
    cache folder. The training stand-in twice: equal order digests, equal cache
    digests. The consumer twice: equal logs, and the orders it read are the ones
    the stand-in recorded."""
    launches = [{"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 30}]
    settings = dict(dataloader_type="cyclic", micro_batch_size=2, data_sharding=True, train_iters=30, launches=launches)
    base = segment_config.base
    first = start(corpus, "run_record.py", write_config(base, "one.json", segment_config(**settings)))
    second = start(corpus, "run_record.py", write_config(base, "two.json", segment_config(**settings)))
    third = start(corpus, "consume.py", write_config(base, "three.json", segment_config(**settings)))
    fourth = start(corpus, "consume.py", write_config(base, "four.json", segment_config(**settings)))
    one, two, read_a, read_b = result_of(first), result_of(second), result_of(third), result_of(fourth)
    assert len(one["shuffle_digests"]) == 4
    assert one["shuffle_digests"] == two["shuffle_digests"]
    assert one["cache_digests"] == two["cache_digests"] == read_a["cache_digests"]
    assert (base / "out-2" / "log.jsonl").read_bytes() == (base / "out-3" / "log.jsonl").read_bytes()
    assert read_a["read_orders"] == read_b["read_orders"]
    assert sorted(
        (o["launch"], o["epoch"], o["rank"], o["digest"], o["count"]) for o in read_a["read_orders"]
    ) == [(d["launch"], d["epoch"], d["rank"], d["digest"], d["count"]) for d in one["shuffle_digests"]]
    by_key = {(d["epoch"], d["rank"]): d["digest"] for d in one["shuffle_digests"]}
    assert by_key[(0, 0)] != by_key[(1, 0)] and by_key[(0, 1)] != by_key[(1, 1)]
    assert by_key[(0, 0)] != by_key[(0, 1)]
    # the record on disk is what was printed
    assert json.loads((base / "out-0" / "run_record.json").read_text()) == one
    assert json.loads((base / "out-0" / "step_record.json").read_text())["steps"][0] == {"step": 1, "batch_size": 4}


def test_index_build_and_replay_scripts(corpus, segment_config):
    base = segment_config.base
    build = segment_config()
    built = result_of(start(corpus, "build_index.py", write_config(base, "build.json", build)))
    assert built["format"] == "waterwhite.oracle-index-digests/1"
    assert len(built["cache_digests"]) == 8
    assert json.loads((base / build["outputs"]["digests"]).read_text()) == built

    def replay(name: str, seed: int) -> dict:
        raw = segment_config(random_seed=seed)
        return {
            "shards": raw["shards"],
            "tokenizer": raw["tokenizer"],
            "settings": {k: raw["settings"][k] for k in
                         ("sequence_length", "random_seed", "split", "global_batch_size", "train_iters")},
            "cache_dir": f"replay-{name}",
            "compare_with": build["outputs"]["digests"],
        }

    same = start(corpus, "replay_index.py", write_config(base, "same.json", replay("same", 1234)))
    other = start(corpus, "replay_index.py", write_config(base, "other.json", replay("other", 4321)))
    # a replay into the folder the first build filled is refused
    refused = start(corpus, "replay_index.py", write_config(base, "refused.json", {**replay("x", 1234), "cache_dir": build["cache_dir"]}))
    same, other = result_of(same), result_of(other)
    assert same["comparison"]["all_equal"] is True and same["comparison"]["with"] == "digests.json"
    assert other["comparison"]["all_equal"] is False
    assert sorted({r["kind"] for r in other["comparison"]["rows"] if not r["equal"]}) == [
        "document_index", "sample_index", "shuffle_index"]
    assert all(r["equal"] for r in other["comparison"]["rows"] if r["shard"] is None)
    code, out, err = finish(refused)
    assert code == 2 and out == ""
    assert json.loads(err.strip().splitlines()[-1])["error"] == "CacheFolderNotEmpty"


def test_consumer_probe_driver_and_positions_scripts(corpus, segment_config):
    base = segment_config.base
    consume_raw = segment_config(segment="T/seg-1")
    probe_raw = {
        "input": str(corpus.inputs["gamma"]),
        "shard": str(corpus.prefixes["gamma"]),
        "tokenizer": corpus.tokenizer,
        "outputs": {"report": "probe/gamma.json"},
    }
    resume = [{"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24, "step_at_start": 6, "steps": 6}]
    driver_raw = {
        "defaults": {"tokenizer": corpus.tokenizer, "settings": {"sequence_length": 32, "split": "1,0,0"}},
        "summary": "driver.json",
        "segments": [],
    }
    for raw in (
        segment_config(segment="E/seg-1", train_iters=6, step_record={"checkpoints": [{"name": "E@6", "step": 6}]}),
        segment_config(segment="E/seg-2", shards=("alpha", "beta", "gamma"), train_iters=12, start_kind="resume",
                       parent_checkpoint="E@6", launches=resume),
    ):
        del raw["tokenizer"], raw["settings"]["sequence_length"], raw["settings"]["split"]
        driver_raw["segments"].append(raw)

    consumer = start(corpus, "consume.py", write_config(base, "consume.json", consume_raw))
    probe = start(corpus, "f1_probe.py", write_config(base, "probe.json", probe_raw))
    driver = start(corpus, "run_segments.py", write_config(base, "driver-config.json", driver_raw))

    summary = result_of(consumer)
    assert summary["samples_read"] == 60 and summary["samples_checked_against_their_reads"] == 60
    log = base / consume_raw["outputs"]["log"]
    assert len(log.read_text().splitlines()) == summary["rows_written"]
    assert json.loads((base / consume_raw["outputs"]["summary"]).read_text()) == summary

    verdict = result_of(probe)
    assert verdict["f1_holds"] is True and verdict["eod"] == {"token_id": 50256, "in_span": True}
    assert (verdict["lines"], verdict["documents"]) == (52, 52)
    assert json.loads((base / "probe" / "gamma.json").read_text()) == verdict

    driven = result_of(driver)
    assert [s["segment"] for s in driven["segments"]] == ["E/seg-1", "E/seg-2"]
    assert json.loads((base / "driver.json").read_text()) == driven
    for raw in driver_raw["segments"]:
        for key, path in raw["outputs"].items():
            assert (base / path).is_file(), key
        assert len(list((base / raw["cache_dir"]).glob("*.npy"))) == 3 * len(raw["shards"]) + 2

    # The positions report needs no Megatron: run it on the consumer's log.
    positions_raw = {
        "logs": [consume_raw["outputs"]["log"]],
        "segments": {"T/seg-1": [str(corpus.inputs["alpha"]), str(corpus.inputs["beta"])]},
        "outputs": {"report": "positions.json"},
    }
    code, out, err = finish(start(corpus, "positions.py", write_config(base, "positions-config.json", positions_raw)))
    assert code == 0, err[-2000:]
    assert out.startswith("Positions report.") and "T/seg-1 step 15" in out
    report = json.loads((base / "positions.json").read_text())
    assert len(report["steps"]) == 15


# --- The fixture runs and the seed search ------------------------------------------------------------


def start_with(corpus, script: str, args: list, cwd: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [str(corpus.python), "-I", str(ORACLE / script), *[str(a) for a in args]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=dict(os.environ),
    )


def test_fixture_runs_and_seed_search_scripts(corpus, built_fixture, tmp_path):
    """Two processes: the fixture runs for runs D and K with a seed override, and a
    search of run D's seed. Then, with no Megatron, the conditions checked on the
    written runs, and a second call into the same folder refused."""
    spec = ORACLE.parent / "fixture" / "spec" / "fixture_spec.toml"
    common = ["--spec", spec, "--fixture", built_fixture]
    runs = start_with(corpus, "fixture_runs.py", [*common, "--out", "runs", "--runs", "D,K", "--seed", "K=2"], tmp_path)
    search = start_with(
        corpus, "seed_search.py",
        [*common, "--out", "search", "--seeds", "0:2", "--time-limit", "20", "--runs", "D"], tmp_path,
    )
    overview = result_of(runs)
    assert [s["segment"] for s in overview["segments"]] == ["D/seg-1", "K/seg-1"]
    assert overview["seeds"]["K"] == 2 and overview["seed_overrides"] == {"K": 2}
    assert json.loads((tmp_path / "runs" / "fixture_runs.json").read_text()) == overview
    assert json.loads((tmp_path / "runs" / "runs" / "K" / "seg-1" / "run_record.json").read_text())["settings"]["random_seed"] == 2

    code, out, err = finish(search)
    assert code == 0, err[-2000:]
    document, _, fragment = out.partition("\n\n# Seed search")
    result = json.loads(document)  # the JSON first, then the TOML fragment
    assert result["format"] == "waterwhite.oracle-seed-search/1" and result["runs_searched"] == ["D"]
    assert result["phases"][0]["accepted"] == [0] and result["chosen_seeds"] == {"D": 0}
    assert result["confirmed"] is None  # not every run was searched
    assert "[runs.D]" in fragment and "[positions.D]" in fragment
    assert json.loads((tmp_path / "search" / "seed_search.json").read_text()) == result

    check = start_with(corpus, "seed_search.py", [*common, "--check-runs", "runs", "--print", "json"], tmp_path)
    checked = result_of(check)
    assert checked["mode"] == "check-runs" and checked["runs_checked"] == ["D", "K"]
    assert {c["n"]: c["holds"] for c in checked["conditions"]}[13] is True

    code, out, err = finish(start_with(corpus, "fixture_runs.py", [*common, "--out", "runs"], tmp_path))
    assert code == 2 and out == "" and json.loads(err.strip().splitlines()[-1])["error"] == "ConfigError"
    code, out, err = finish(start_with(corpus, "seed_search.py", [*common, "--seeds", "9:1", "--out", "s2"], tmp_path))
    assert code == 2 and out == "" and "FIRST:LAST" in err
