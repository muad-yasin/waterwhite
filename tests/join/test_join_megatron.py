"""The join against Megatron's own behaviour (plan 5.5 Check 1 and Check 2; F3, F9).

The expected side is made by Megatron's own code and never by the tool: the
replica builds the hand-checked fixture, the oracle runs Megatron's dataset and
sampler classes over its shards and writes, per run segment, the run record, the
cache folder and the log of what every step read (docs/oracle.md, "Fixture
runs"). The tool's side is ``waterwhite.join`` run on each segment's run record.
The comparison lives here: the oracle imports nothing of the tool and the tool
nothing of the oracle.

Runs compared: E and its round-robin variant E-rr (both layouts), D, Y and the
cyclic run K, seven segments, each joined by the cache-files route and, with the
cache folder withheld, by the replay. Run K's shuffle is computed by the join
itself, by calling torch as the sampler does, and held against the digests the
oracle recorded (the table row c08.K.digests-match).

Building the fixture and running the oracle takes about two minutes. To reuse a
finished pair, set ``WATERWHITE_JOIN_RUNS`` to the folder ``oracle/fixture_runs.py``
wrote (its ``--out``); the run records in it name the fixture's shards by
relative path, so the fixture folder must still lie where it was built.

The tests fail, and are not skipped, when the Megatron environment is missing and the
run selects them: a run that does not name the marker has them deselected by
tests/conftest.py (and says so); ``-m "not megatron"`` leaves them out on purpose.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import join_kit as kit
import numpy as np
import pytest

from waterwhite import ids
from waterwhite import vocabulary as v
from waterwhite.join import (
    compute_join,
    digest_of,
    needed_keys,
    parse_run_record,
    read_cache_files,
    epoch_permutation,
    rank_order,
    replay_arrays,
)
from waterwhite.store import artifact_files

pytestmark = pytest.mark.megatron

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "fixture" / "spec" / "fixture_spec.toml"
SEGMENTS = ("E/seg-1", "E/seg-2", "E-rr/seg-1", "E-rr/seg-2", "D/seg-1", "Y/seg-1", "K/seg-1")
SEQUENTIAL = SEGMENTS[:6]
BUILD_SECONDS = 150
SAMPLE_TOKENS = 33  # sequence length 32 and the extra token (fixture spec, [sizes])


@pytest.fixture(scope="session")
def megatron_env():
    value = os.environ.get("WATERWHITE_MEGATRON_ENV")
    if not value or not Path(value).is_dir():
        pytest.fail("WATERWHITE_MEGATRON_ENV does not name the Megatron environment folder; "
                    "leave these tests out with -m 'not megatron'")
    env = Path(value)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HOME", str(env / "hf_home"))
        patch.setenv("HF_HUB_OFFLINE", "1")
        yield env


def run(command: list[str], log: Path, **env: str) -> None:
    with open(log, "wb") as out:
        done = subprocess.run(command, cwd=REPO_ROOT, stdout=out, stderr=subprocess.STDOUT,
                              env={**os.environ, **env}, timeout=BUILD_SECONDS, check=False)
    if done.returncode != 0:
        pytest.fail(f"{command[2:4]} ended with exit {done.returncode}; its output is in {log}")


@pytest.fixture(scope="session")
def runs(megatron_env, tmp_path_factory) -> Path:
    """The folder of the oracle's fixture runs: ``runs/<run>/<segment>/...``."""
    given = os.environ.get("WATERWHITE_JOIN_RUNS")
    if given:
        return Path(given)
    base = tmp_path_factory.mktemp("join-megatron")
    python = str(megatron_env / "venv" / "bin" / "python")
    run([python, "-m", "replica", "build-fixture", "--spec", str(SPEC), "--out",
         str(base / "fixture"), "--jobs", "3"], base / "build-fixture.log", PYTHONPATH="src")
    run([python, "-I", "oracle/fixture_runs.py", "--spec", str(SPEC), "--fixture",
         str(base / "fixture"), "--out", str(base / "runs")], base / "fixture-runs.log")
    return base / "runs"


class Run:
    """One segment as the oracle left it."""

    def __init__(self, runs: Path, segment: str) -> None:
        self.folder = runs / "runs" / segment
        self.record = json.loads((self.folder / "run_record.json").read_text(encoding="utf-8"))
        self.step_record = json.loads((self.folder / "step_record.json").read_text(encoding="utf-8"))
        self.log = [json.loads(line) for line in
                    (self.folder / "oracle_log.jsonl").read_text(encoding="utf-8").splitlines()]
        self.cache = self.folder / "cache"


def dataset_length(run_: Run) -> int:
    """The length of the segment's dataset, as the oracle's index build reports it."""
    document = json.loads((run_.folder / "index_digests.json").read_text(encoding="utf-8"))
    return int(document["dataset_samples"])


def joined(run_: Run, *, cache: bool, record: dict | None = None, step_record: dict | None = None):
    return compute_join(run_.record if record is None else record, verified_joins=kit.JOINS,
                        record_dir=run_.folder, cache_dir=run_.cache if cache else None,
                        step_record=step_record)


def log_rows(result, segment: str) -> list[dict]:
    """The join's rows in the form of the oracle's log (docs/interfaces.md 8.3)."""
    return [{"segment": segment, "step": trace.step, "sample_pos": sample.sample_pos,
             "global_index": sample.global_index, "rank": sample.rank,
             "shard_position": row.shard_position, "ordinal": row.ordinal,
             "token_start": row.token_start, "token_end": row.token_end}
            for trace in result.traces for sample in trace.samples for row in sample.rows]


@pytest.mark.parametrize("cache", [True, False], ids=["cache-files", "replay"])
@pytest.mark.parametrize("segment", SEGMENTS)
def test_the_joins_rows_equal_the_oracles_log_line_by_line(runs, segment, cache):
    """Check 1: every (step, sample, shard, ordinal, token start, token end) the
    join reports is the oracle's, in the oracle's order, on both layouts and on
    runs E, E-rr, D, Y and K. No step record is given here, so every sample is
    whole, as the oracle's log always is."""
    run_ = Run(runs, segment)
    result = joined(run_, cache=cache)
    assert result.cause is None, result.explanation
    assert result.route == (v.ROUTE_CACHE_FILES if cache else v.ROUTE_REPLAY)
    assert result.dataset_length == dataset_length(run_)
    mine = log_rows(result, segment)
    assert len(mine) == len(run_.log)
    for line, (ours, theirs) in enumerate(zip(mine, run_.log), start=1):
        assert ours == theirs, f"line {line} of the log of {segment}"
    assert result.untraced == ()
    assert all(sample.covered_tokens == sample.sample_tokens == SAMPLE_TOKENS
               for trace in result.traces for sample in trace.samples)


@pytest.mark.parametrize("segment", SEGMENTS)
def test_the_replayed_arrays_are_byte_identical_to_the_withheld_ones(runs, segment):
    """Check 2 (i): the arrays the join derives from the settings and the seed are
    the arrays in the cache files Megatron wrote, byte for byte, kind for kind."""
    run_ = Run(runs, segment)
    record = parse_run_record(run_.record)
    keys = needed_keys(len(record.shards))
    kept, missing, _ = read_cache_files(run_.cache, record.cache_digests, keys)
    assert missing == []
    lengths = [artifact_files.read_shard_index(
        artifact_files.shard_paths(run_.folder / entry.prefix)[1]).sequence_lengths
        for entry in record.shards]
    replayed = replay_arrays(record, lengths)
    assert sorted(replayed, key=str) == sorted(keys, key=str)
    for key in keys:
        assert replayed[key].dtype == kept[key].dtype, key
        assert replayed[key].shape == kept[key].shape, key
        assert replayed[key].tobytes() == np.ascontiguousarray(kept[key]).tobytes(), key
        assert digest_of(replayed[key]) == record.cache_digests[key], key


@pytest.mark.parametrize("segment", SEQUENTIAL)
def test_a_replay_from_a_changed_seed_is_rejected(runs, segment):
    """Check 2 (ii): cache files withheld and the seed in the settings changed."""
    run_ = Run(runs, segment)
    record = json.loads(json.dumps(run_.record))
    record["settings"]["random_seed"] += 1
    result = joined(run_, cache=False, record=record)
    assert (result.route, result.cause, result.cause_detail) == (
        v.ROUTE_NONE, v.NAMES_TRACE_MISSING, v.TRACE_MISSING_DIGEST_MISMATCH)
    assert result.replay.attempted and not result.replay.accepted and result.traces == ()


def test_the_lowered_step_of_run_e_is_cut_where_the_oracle_says(runs):
    """Check 1, the half-covered sample of step 8: with the operator's step record
    the rows up to the cut equal the oracle's, the row that holds the next token
    ends at the cut, the rest of the sample is an uncovered range, and the sample
    is still counted. The cut the join makes is held against the cut the oracle
    worked out from its own log (``step_cuts`` of its overview)."""
    run_ = Run(runs, "E/seg-2")
    overview = json.loads((runs / "fixture_runs.json").read_text(encoding="utf-8"))
    (expected,) = [c for c in overview["step_cuts"] if c["segment"] == "E/seg-2"]
    result = joined(run_, cache=True, step_record=run_.step_record)
    mine = log_rows(result, "E/seg-2")
    at = (expected["step"], expected["sample_pos"])
    assert [row for row in mine if (row["step"], row["sample_pos"]) != at] == [
        row for row in run_.log if (row["step"], row["sample_pos"]) != at]
    assert [row for row in mine if (row["step"], row["sample_pos"]) == at] == expected["covered_rows"]
    (trace,) = [t for t in result.traces if t.step == expected["step"]]
    assert trace.consumed_samples == len(trace.samples) == 4
    sample = trace.samples[expected["sample_pos"]]
    assert (sample.sample_tokens, sample.covered_tokens) == (
        expected["sample_tokens"], expected["covered_tokens"])
    assert sample.uncovered == (expected["covered_tokens"], expected["sample_tokens"])
    last = sample.rows[-1]
    assert (last.shard_position, last.ordinal, last.token_end) == (
        expected["uncovered_from"]["shard_position"], expected["uncovered_from"]["ordinal"],
        expected["uncovered_from"]["token"])
    assert result.lowered_steps == (expected["step"],)


def test_the_shuffle_the_join_computes_for_run_k_matches_the_recorded_digests(runs):
    """Table row c08.K.digests-match (plan 5.5 Check 8; 3.5, the order of a cyclic
    run): for every (launch, epoch, rank) of run K's run record, the order the join
    computes from the record's sizes, by its call to torch, has the digest and the
    count the oracle recorded, which the oracle took from Megatron's own sampler
    class. A digest covers a whole epoch's order for one rank, whatever part of it
    the launch read, so launch 0 and launch 1 record epoch 0 alike, rank by rank."""
    run_ = Run(runs, "K/seg-1")
    record = run_.record
    settings = record["settings"]
    assert settings["dataloader_type"] == "cyclic" and settings["data_sharding"] is True
    ranks = {launch["position"]: launch["data_parallel_size"] for launch in record["launches"]}
    # The sizes, worked out here: the dataset loses its last part round and each rank
    # owns an equal bucket of what is left.
    micro = settings["micro_batch_size"]
    bucket = (dataset_length(run_) // (micro * ranks[0])) * micro
    assert record["shuffle_digests"]
    for row in record["shuffle_digests"]:
        order = rank_order(epoch_permutation(row["epoch"], bucket), row["rank"], bucket)
        where = (row["launch"], row["epoch"], row["rank"])
        assert ids.order_digest(order) == row["digest"], where
        assert len(order) == row["count"] == bucket, where
    held = {(r["launch"], r["epoch"], r["rank"]): r["digest"] for r in record["shuffle_digests"]}
    for rank in range(ranks[0]):
        assert held[(0, 0, rank)] == held[(1, 0, rank)]
    # The whole join agrees: it accepts every digest and traces all 40 steps.
    result = joined(run_, cache=True)
    assert (result.route, result.cause, result.steps_written) == (v.ROUTE_CACHE_FILES, None, 40)


def test_run_k_with_its_shuffle_digests_removed_or_changed(runs):
    """Plan 3.5, the ruling on the cyclic order; Check 2 (iv)."""
    run_ = Run(runs, "K/seg-1")
    # Check 2 (iv): cache files kept, the shuffle digests removed from the record.
    record = {**run_.record, "shuffle_digests": []}
    result = joined(run_, cache=True, record=record)
    assert (result.route, result.cause, result.traces) == (v.ROUTE_NONE, v.NAMES_CYCLIC_NOT_RECORDED, ())
    # One recorded digest changed: the computed order no longer matches it.
    record = json.loads(json.dumps(run_.record))
    record["shuffle_digests"][-1]["digest"] = ids.order_digest([0])
    result = joined(run_, cache=True, record=record)
    assert (result.route, result.cause) == (v.ROUTE_NONE, v.NAMES_CYCLIC_DIGEST_MISMATCH)


@pytest.mark.parametrize("train_iters", [5, 8, 12])
def test_the_replay_of_a_segment_of_one_shard_equals_megatrons_build(runs, megatron_env, tmp_path,
                                                                    train_iters):
    """A segment of one shard has no blend, and its one dataset is built for as
    many epochs as the requested samples need. The fixture has no such segment, so
    Megatron's builder is run here on shard 1 of run E through the oracle's index
    build, for a request inside one epoch (5 steps of 4), one that needs a second
    epoch for a few samples (8) and one that needs most of a second epoch (12)."""
    from oracle import config as oracle_config
    from oracle import pipeline

    base = json.loads((runs / "runs" / "E/seg-1" / "config.json").read_text(encoding="utf-8"))
    prefix = (runs / "runs" / "E/seg-1" / base["shards"][0]).resolve()
    settings = {**base["settings"], "train_iters": train_iters}
    config = {"shards": [str(prefix)], "tokenizer": base["tokenizer"], "settings": settings,
              "cache_dir": "cache", "outputs": {"digests": "digests.json"}}
    path = kit.write_json(tmp_path / "config.json", config)
    data, _, _ = oracle_config.load_data_order(path)
    _, document = pipeline.build_index(data)
    recorded = {(row["kind"], row["shard"]): row["digest"] for row in document["cache_digests"]}
    assert sorted(recorded) == sorted(needed_keys(1)) and document["blend"] is False

    record = parse_run_record({"format": "waterwhite.run-record/1", "settings": settings})
    lengths = [artifact_files.read_shard_index(artifact_files.shard_paths(prefix)[1]).sequence_lengths]
    replayed = replay_arrays(record, lengths)
    assert {key: digest_of(array) for key, array in replayed.items()} == recorded
    assert int(replayed[("sample_index", 0)].shape[0]) - 1 == document["dataset_samples"]
