"""The fixture runs and the seed search, run against Megatron in this process.

They run the real fixture spec's runs over a stand-in built fixture made of the
throwaway shards (conftest.py, ``built_fixture``). The stand-in has the layout of
a fixture build and not its texts, so these tests are about what is written
where, that the search reads what the fixture runs write, and how a search
counts, stops and confirms. Whether each condition's predicate checks its
sentence is tested on hand-made logs in test_oracle_search.py.
"""

from __future__ import annotations

import dataclasses
import json
import time
import tomllib
from pathlib import Path

import pytest

from oracle import fixture_runs, search_conditions, search_evidence, seed_search
from oracle.errors import ConfigError
from oracle.fixture_spec import load_spec
from oracle.search_conditions import Outcome

pytestmark = pytest.mark.megatron

REPO = Path(__file__).resolve().parents[2]
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
SEGMENT_FOLDERS = ("E/seg-1", "E/seg-2", "E-rr/seg-1", "E-rr/seg-2", "D/seg-1", "Y/seg-1", "K/seg-1")
# The conditions that hold on the stand-in whatever the seed: they rest on the
# spec's steps, launches and sizes alone.
HOLD_ON_THE_STAND_IN = (1, 13, 17, 18, 19, 20)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def megatron_epochs(total: int, rank: int, epochs: int) -> list[list[int]]:
    """Megatron-LM's cyclic sampler class driven directly from consumed count 0,
    with no oracle code between: the whole order of each of the first epochs."""
    from megatron.training.datasets import data_samplers

    sampler = data_samplers.MegatronPretrainingRandomSampler(
        None, total_samples=total, consumed_samples=0, micro_batch_size=2,
        data_parallel_rank=rank, data_parallel_size=2, data_sharding=True,
    )
    return [[int(i) for batch in sampler for i in batch] for _ in range(epochs)]


def test_fixture_runs_write_every_segment_of_the_spec(built_fixture, tmp_path):
    from waterwhite import ids

    spec = load_spec(SPEC_FILE)
    out = tmp_path / "runs"
    overview = fixture_runs.run_fixture(SPEC_FILE, built_fixture, out, seed_overrides={"E": 5})

    # The layout: runs/<run>/<segment>/ with the five named things and two more files.
    assert sorted(p.name for p in out.iterdir()) == ["fixture_runs.json", "runs"]
    assert overview == read_json(out / "fixture_runs.json")
    assert [s["folder"] for s in overview["segments"]] == [f"runs/{name}" for name in SEGMENT_FOLDERS]
    for name in SEGMENT_FOLDERS:
        assert sorted(p.name for p in (out / "runs" / name).iterdir()) == [
            "cache", "config.json", "index_digests.json", "oracle_log.jsonl", "run_record.json",
            "step_record.json", "summary.json",
        ]
    written = "".join(
        (out / "runs" / name / file).read_text(encoding="utf-8")
        for name in SEGMENT_FOLDERS
        for file in ("config.json", "run_record.json", "step_record.json", "oracle_log.jsonl", "summary.json", "index_digests.json")
    ) + (out / "fixture_runs.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in written and str(built_fixture) not in written

    # Seeds: the spec's, and the override for run E, which the round-robin variant shares.
    assert overview["seeds"] == {"E": 5, "E-rr": 5, "D": spec.runs["D"].random_seed,
                                 "Y": spec.runs["Y"].random_seed, "K": spec.runs["K"].random_seed}
    assert overview["seed_overrides"] == {"E": 5}
    records = {name: read_json(out / "runs" / name / "run_record.json") for name in SEGMENT_FOLDERS}
    assert [records[name]["settings"]["random_seed"] for name in SEGMENT_FOLDERS[:4]] == [5, 5, 5, 5]

    # Run records: what the spec states, and paths relative to the record's folder.
    e2 = records["E/seg-2"]
    assert (e2["run"], e2["segment"], e2["start_kind"], e2["parent_checkpoint"]) == ("E", "E/seg-2", "resume", "E@6")
    assert e2["settings"]["sampler_class"] == spec.sampler_classes["single"]
    assert (e2["settings"]["train_iters"], e2["settings"]["sequence_length"], e2["settings"]["split"]) == (12, 32, "1,0,0")
    assert e2["launches"] == [{"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24,
                               "step_at_start": 6, "first_step": 7, "last_step": 12}]
    assert e2["shuffle_digests"] == []
    for entry, shard in zip(e2["shards"], ("shard-1", "shard-2", "shard-3")):
        assert not entry["prefix"].startswith("/")
        prefix = (out / "runs" / "E" / "seg-2" / entry["prefix"]).resolve()
        assert prefix == (built_fixture / "pipeline" / "partitioned" / "shards" / shard).resolve()
        assert entry["bin_sha256"] == ids.file_digest(str(prefix) + ".bin")
    rr2 = records["E-rr/seg-2"]
    # the round-robin variant resumes from its own registered checkpoint (DECISIONS.md D-24)
    assert (rr2["start_kind"], rr2["parent_checkpoint"]) == ("resume", "E-rr@6")
    assert read_json(out / "runs" / "E-rr" / "seg-1" / "step_record.json")["checkpoints"] == [
        {"name": "E-rr@6", "step": 6}]
    assert read_json(out / "runs" / "E-rr" / "seg-2" / "step_record.json")["checkpoints"] == []
    starts = {s["segment"]: s["start"] for s in overview["segments"]}
    assert "the state of 'E-rr@6'" in starts["E-rr/seg-2"] and "the state of 'E@6'" in starts["E/seg-2"]
    assert rr2["shards"][0]["prefix"].endswith("pipeline/round-robin/shards/shard-1")
    assert (records["D/seg-1"]["start_kind"], records["D/seg-1"]["parent_checkpoint"]) == ("warm-start", "E@9")
    assert records["D/seg-1"]["launches"][0]["consumed_at_start"] == 0
    assert len(list((out / "runs" / "E" / "seg-2" / "cache").iterdir())) == 3 * 4 + 3  # kept as Megatron wrote it

    # Step records come from the spec's tables; the cut of step 8 from the oracle's own log.
    steps = read_json(out / "runs" / "E" / "seg-2" / "step_record.json")
    assert steps["steps"][1] == {"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}
    assert [s for s in steps["steps"] if "last_sample_tokens_read" in s] == [steps["steps"][1]]
    assert steps["checkpoints"] == [{"name": "E@9", "step": 9}]
    assert read_json(out / "runs" / "K" / "seg-1" / "step_record.json")["checkpoints"] == [
        {"name": "K@3", "step": 3}, {"name": "K@8", "step": 8}, {"name": "K@40", "step": 40}]
    assert not any("last_sample_tokens_read" in s for name in ("E-rr/seg-1", "E-rr/seg-2")
                   for s in read_json(out / "runs" / name / "step_record.json")["steps"])
    (cut,) = overview["step_cuts"]
    last = [r for r in read_log(out / "runs" / "E" / "seg-2" / "oracle_log.jsonl") if (r["step"], r["sample_pos"]) == (8, 3)]
    assert cut["log_lines"] == last and (cut["covered_tokens"], cut["sample_tokens"]) == (16, 33)
    # walk the sample's rows token by token: token 16 of the sample is where the uncovered range starts
    tokens = [(r["shard_position"], r["ordinal"], t) for r in last for t in range(r["token_start"], r["token_end"])]
    assert len(tokens) == 33
    assert cut["uncovered_from"] == dict(zip(("shard_position", "ordinal", "token"), tokens[16]))
    covered = [(r["shard_position"], r["ordinal"], t) for r in cut["covered_rows"] for t in range(r["token_start"], r["token_end"])]
    assert covered == tokens[:16]

    # Run K: both launches, and the order digests of whole epochs (DECISIONS.md D-16).
    k = records["K/seg-1"]
    assert k["settings"]["sampler_class"] == spec.sampler_classes["cyclic"]
    assert [(l["consumed_at_start"], l["first_step"], l["last_step"], l["data_parallel_size"]) for l in k["launches"]] == [
        (0, 1, 3, 2), (12, 4, 40, 2)]
    length = read_json(out / "runs" / "K" / "seg-1" / "summary.json")["dataset_samples"]
    assert length == 59  # shard-3 and shard-2 of the stand-in: 35 and 24 samples
    digests = {(d["launch"], d["epoch"], d["rank"]): (d["count"], d["digest"]) for d in k["shuffle_digests"]}
    assert sorted(digests) == sorted(
        [(0, 0, rank) for rank in (0, 1)] + [(1, epoch, rank) for epoch in (0, 1, 2) for rank in (0, 1)]
    )
    for rank in (0, 1):
        epochs = megatron_epochs(length, rank, 3)
        assert len(epochs[0]) == 28  # 59 samples: 14 steps of 2 per rank; launch 0 read 6 of them
        assert digests[(0, 0, rank)] == digests[(1, 0, rank)] == (28, ids.order_digest(epochs[0]))
        assert digests[(1, 1, rank)] == (28, ids.order_digest(epochs[1]))
        assert digests[(1, 2, rank)] == (28, ids.order_digest(epochs[2]))
    k_rows = read_log(out / "runs" / "K" / "seg-1" / "oracle_log.jsonl")
    assert len({(r["step"], r["sample_pos"]) for r in k_rows}) == 160

    # An output folder is written once.
    with pytest.raises(ConfigError, match="new or empty"):
        fixture_runs.run_fixture(SPEC_FILE, built_fixture, out)
    with pytest.raises(ConfigError, match="no pipeline/ folder"):
        fixture_runs.run_fixture(SPEC_FILE, tmp_path / "nowhere", tmp_path / "other")


def test_a_search_reads_in_memory_what_the_fixture_runs_write(built_fixture, tmp_path):
    """One seed, two routes: the search's read of a run with its log in memory,
    and the files the fixture runs write. Rows, order digests, step records and
    document counts are the same, so a condition gives one answer on both."""
    spec = load_spec(SPEC_FILE)
    build = search_evidence.load_build(spec, built_fixture, REPO)
    fixture_runs.run_fixture(SPEC_FILE, built_fixture, tmp_path / "runs", seed_overrides={"E": 3, "K": 3})
    written = search_evidence.evidence_from_runs(spec, build, built_fixture, tmp_path / "runs")
    assert list(written.runs) == ["E", "E-rr", "D", "Y", "K"]
    engine = seed_search.Search(
        spec, build, fixture=built_fixture, out=tmp_path / "search",
        tokenizer=fixture_runs.find_tokenizer(None), shards_from="pipeline", clock=time.monotonic,
    )
    for run in ("E", "E-rr", "K"):
        for memory, file in zip(engine.read_run(spec.runs[run], 3), written.runs[run], strict=True):
            assert memory.rows == file.rows and memory.rows
            assert memory.shuffle_digests == file.shuffle_digests
            assert memory.step_record == file.step_record
            assert memory.shard_documents == file.shard_documents
            assert memory.summary["read_orders"] == file.summary["read_orders"]
            assert memory.summary["launches"] == file.summary["launches"]
    assert len(written.runs["K"][0].shuffle_digests) == 8

    # Logs of one build are not read against another build's manifests.
    other = tmp_path / "other-build"
    import shutil

    shutil.copytree(built_fixture, other)
    shard = other / "pipeline" / "partitioned" / "shards" / "shard-2.bin"
    shard.write_bytes(shard.read_bytes()[:-2])
    with pytest.raises(ConfigError, match="not the shard of this built fixture"):
        search_evidence.evidence_from_runs(spec, search_evidence.load_build(spec, other, REPO), other, tmp_path / "runs")


def test_the_conditions_are_checked_on_written_runs(built_fixture, tmp_path):
    fixture_runs.run_fixture(SPEC_FILE, built_fixture, tmp_path / "runs")
    result, fragment = seed_search.check_runs(SPEC_FILE, built_fixture, tmp_path / "runs")
    assert result["mode"] == "check-runs" and result["not_evaluated"] == []
    assert [c["n"] for c in result["conditions"]] == list(range(1, 24))
    holds = {c["n"]: c["holds"] for c in result["conditions"]}
    assert all(holds[n] for n in HOLD_ON_THE_STAND_IN), result["conditions"]
    assert holds[2] is False  # the stand-in's sanitizer log is empty
    assert result["all_conditions_hold"] is False and 2 in result["fails"]
    parsed = tomllib.loads(fragment)
    assert parsed["positions"]["K"]["blend_length"] == 59 and parsed["positions"]["K"]["epoch_steps"] == 14
    assert "NOT ACCEPTED" in fragment
    # only the runs with a folder are checked
    fixture_runs.run_fixture(SPEC_FILE, built_fixture, tmp_path / "only-d", runs=["D"])
    partly, _ = seed_search.check_runs(SPEC_FILE, built_fixture, tmp_path / "only-d")
    assert [c["n"] for c in partly["conditions"]] == [13] and len(partly["not_evaluated"]) == 22


def test_a_search_counts_each_condition_and_stops_at_its_time_limit(built_fixture, tmp_path):
    """A clock that moves one second per reading makes the limit exact: 40 seconds
    for four runs. Run E can never be accepted on the stand-in (condition 2), so
    its share of the time ends its search; run D is accepted at its first seed."""
    ticks = iter(range(100_000))
    result, fragment = seed_search.search(
        SPEC_FILE, built_fixture, tmp_path / "out", seeds=range(0, 50), time_limit=40.0,
        clock=lambda: float(next(ticks)),
    )
    assert [p["run"] for p in result["phases"]] == ["E", "D", "Y", "K"]
    e, d = result["phases"][0], result["phases"][1]
    assert (e["with"], e["ended_by"], e["accepted"]) == (["E-rr"], "time limit", [])
    assert 1 <= e["tried"] < 50
    counts = {c["n"]: (c["evaluated"], c["held"]) for c in e["conditions"]}
    assert sorted(counts) == list(range(1, 13)) + [16, 17]
    assert counts[1] == (e["tried"], e["tried"]) and counts[2] == (e["tried"], 0)
    # the round-robin variant was read once, for the best candidate, since no seed reached it
    assert counts[17] == (1, 1) and counts[16][0] == 1
    assert sum(s["candidates"] for s in e["failing_sets"]) == e["tried"]
    assert all(2 in s["fails"] for s in e["failing_sets"])
    assert len(e["best"]) == min(5, e["tried"]) and e["best"][0]["accepted"] is False
    assert [c["n"] for c in e["best"][0]["conditions"]] == list(range(1, 13)) + [16, 17]
    assert (d["tried"], d["ended_by"], d["accepted"]) == (1, "accepted", [0])
    # an accepted seed comes with its positions and log lines; a failing candidate with its verdicts only
    assert d["best"][0]["accepted"] is True and len(d["best"][0]["positions"]["D"]["log_lines"]) >= 12
    assert e["best"][0]["positions"] is None
    assert result["chosen_seeds"]["E"] is None and result["chosen_seeds"]["D"] == 0
    assert result["all_conditions_hold"] is False and result["confirmed"] is None
    assert 2 in result["never_held"] and result["not_evaluated"] == []
    assert result["candidate_seeds"] == {"first": 0, "last": 49}
    assert read_json(tmp_path / "out" / "seed_search.json") == json.loads(json.dumps(result))
    assert (tmp_path / "out" / "positions_fragment.toml").read_text(encoding="utf-8") == fragment
    parsed = tomllib.loads(fragment)
    assert parsed["runs"]["D"] == {"random_seed": 0} and "NOT ACCEPTED" in fragment
    assert not (tmp_path / "out" / "confirm").exists()
    # every candidate got cache folders of its own
    assert len(list((tmp_path / "out" / "scratch").iterdir())) >= e["tried"]

    with pytest.raises(ConfigError, match="new or empty"):
        seed_search.search(SPEC_FILE, built_fixture, tmp_path / "out", seeds=range(0, 2), time_limit=5.0)
    with pytest.raises(ConfigError, match="seed of their own"):
        seed_search.search(SPEC_FILE, built_fixture, tmp_path / "out2", seeds=range(0, 2), time_limit=5.0, runs=["E-rr"])


def test_seeds_that_satisfy_everything_are_confirmed_on_written_runs(built_fixture, tmp_path, monkeypatch):
    """How a search ends when it finds seeds. The stand-in cannot satisfy the
    conditions that rest on the fixture's texts, so those predicates are replaced
    by ones that always hold; the six that rest on steps and sizes stay real.
    The search then accepts the first seed of each run, writes the fixture runs
    with those seeds, checks every condition again on the written files and says
    so. With the real predicates back, the same written runs fail."""
    forced = tuple(
        check if check.n in HOLD_ON_THE_STAND_IN
        else dataclasses.replace(check, check=lambda ev: Outcome(True, "forced to hold in this test"))
        for check in search_conditions.CHECKS
    )
    with monkeypatch.context() as patch:
        patch.setattr(search_conditions, "CHECKS", forced)
        result, fragment = seed_search.search(
            SPEC_FILE, built_fixture, tmp_path / "out", seeds=range(4, 9), time_limit=1000.0
        )
        assert [(p["run"], p["tried"], p["ended_by"], p["accepted"]) for p in result["phases"]] == [
            (run, 1, "accepted", [4]) for run in ("E", "D", "Y", "K")
        ]
        assert result["chosen_seeds"] == {"E": 4, "D": 4, "Y": 4, "K": 4}
        assert result["all_conditions_hold"] is True
        confirmed = result["confirmed"]
        assert (confirmed["folder"], confirmed["all_conditions_hold"], confirmed["fails"], confirmed["not_evaluated"]) == (
            "confirm", True, [], [])
        assert [c["n"] for c in confirmed["conditions"]] == list(range(1, 24))
        records = [read_json(tmp_path / "out" / "confirm" / "runs" / name / "run_record.json") for name in SEGMENT_FOLDERS]
        assert {record["settings"]["random_seed"] for record in records} == {4}
        parsed = tomllib.loads(fragment)
        assert {run: table["random_seed"] for run, table in parsed["runs"].items()} == {"E": 4, "D": 4, "Y": 4, "K": 4}
        assert "Status: every search condition holds for these seeds, checked again on the written runs." in fragment
        # the positions come from the written runs: K's facts are those of its files
        assert parsed["positions"]["K"]["blend_length"] == 59
        again, _ = seed_search.check_runs(SPEC_FILE, built_fixture, tmp_path / "out" / "confirm")
        assert again["all_conditions_hold"] is True and again["seeds"]["K"] == 4
    real, real_fragment = seed_search.check_runs(SPEC_FILE, built_fixture, tmp_path / "out" / "confirm")
    assert real["all_conditions_hold"] is False and 2 in real["fails"]
    assert "NOT ACCEPTED" in real_fragment
