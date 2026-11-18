"""What ``show SEGMENT --trace`` and ``replay`` print, read by ``acceptance_trace`` and
compared by ``acceptance_join``, shown on documents and files written by hand.

The two forms are not in the tool this file was written against, so nothing in the
table's rows shows these comparisons to work or to fail. The documents here have the
shape the forms print (a trace: steps, each with samples, each with rows; a replay: arrays,
each with its kind, shard and digests); the values are the fixture spec's
(``positions``), never an output of the tool.
"""

from __future__ import annotations

import json
from pathlib import Path

import acceptance_join as join
import acceptance_runs as runs
import acceptance_trace as trace
import numpy as np
import pytest

POSITIONS = runs.spec()["positions"]


def row(shard: int, ordinal: int, start: int, end: int) -> dict:
    return {"shard": f"E/shard-{shard + 1}", "shard_position": shard, "ordinal": ordinal,
            "token_start": start, "token_end": end}


def sample(pos: int, index: int, rank: int, rows: list[dict], covered: int = 33) -> dict:
    return {"sample_pos": pos, "global_index": index, "rank": rank, "sample_tokens": 33,
            "covered_tokens": covered, "rows": rows}


# Step 7 of E/seg-2 as the spec records the sample of T (positions.T), then step 8 whose
# last sample is cut at 16 tokens (positions.step8_cut).
T = POSITIONS["T"]
CUT = POSITIONS["step8_cut"]
DOC = {
    "segment": "E/seg-2", "route": "cache files", "cause": None, "cause_detail": None,
    "steps": [
        {"step": 7, "consumed_samples": 4, "samples": [
            sample(0, 24, 0, [row(2, 24, 32, 39), row(T["shard_position"], T["ordinal"], 0, 22),
                              row(2, 26, 0, 4)]),
            sample(1, 25, 0, [row(0, 3, 0, 33)])]},
        {"step": 8, "consumed_samples": 4, "samples": [
            sample(3, CUT["global_index"], 0, [row(**{
                "shard": CUT["shard_position"], "ordinal": r["ordinal"],
                "start": r["token_start"], "end": r["token_end"]})
                for r in CUT["rows_of_the_covered_part"]], covered=16)]}],
    "untraced_steps": [["E/seg-2", 8]],
}


def test_the_rows_of_a_printed_trace_are_the_oracle_s_rows_in_reading_order():
    rows = trace.flatten(DOC)
    first = rows[0]
    assert (first["segment"], first["step"], first["sample_pos"], first["global_index"],
            first["rank"]) == ("E/seg-2", 7, 0, 24, 0)
    assert [(r["ordinal"], r["token_start"], r["token_end"]) for r in rows[:3]] == [
        (24, 32, 39), (T["ordinal"], 0, 22), (26, 0, 4)]
    assert len(rows) == 3 + 1 + len(CUT["rows_of_the_covered_part"])


def test_a_trace_document_that_lacks_a_field_is_an_error_naming_it():
    broken = {"segment": "E/seg-2", "steps": [{"step": 7, "consumed_samples": 4}]}
    with pytest.raises(trace.TraceError, match="samples"):
        trace.flatten(broken)


def test_a_whole_sample_has_no_uncovered_range_and_a_cut_one_starts_at_the_mark():
    held = trace.samples(DOC)
    assert trace.uncovered(held[("E/seg-2", 7, 0)]) is None
    assert trace.uncovered(held[("E/seg-2", 8, 3)]) == CUT["uncovered_in_sample"]
    assert trace.steps_with_uncovered_range(DOC) == [8]


def test_the_rank_layout_counts_the_samples_of_a_step_and_the_places_of_each_rank():
    k = {"segment": "K/seg-1", "steps": [{"step": 1, "consumed_samples": 4, "samples": [
        sample(0, 0, 0, []), sample(1, 1, 0, []), sample(2, 2, 1, []), sample(3, 3, 1, [])]}]}
    assert trace.rank_layout(k) == {1: {"samples": 4, "by_rank": {0: [0, 1], 1: [2, 3]}}}
    k["steps"][0]["samples"][1]["rank"] = 1
    assert trace.rank_layout(k)[1]["by_rank"] == {0: [0], 1: [1, 2, 3]}


# --- The diff, with the place and the rank of a sample --------------------------------------


def oracle_rows() -> list[dict]:
    return [{k: v for k, v in r.items()} for r in trace.flatten(DOC)]


def test_the_trace_of_the_spec_s_rows_has_an_empty_diff_with_the_oracle_s_log_cut_at_step_8():
    cut_log = oracle_rows()
    # The oracle's log is not lowered: its row at the cut runs on past it.
    cut_log[-1]["token_end"] = 44
    cut_log.append({**cut_log[-1], "ordinal": 23, "token_start": 0, "token_end": 8})
    assert join.diff(trace.flatten(DOC), cut_log, {("E/seg-2", 8, 3): 16}) == []


def test_the_diff_compares_the_planned_fields_and_the_trace_rows_carry_more():
    reported = trace.flatten(DOC)
    assert {"global_index", "rank"} <= set(reported[0])
    reported[0]["global_index"] = 99
    assert join.diff(reported, oracle_rows(), {}) == []   # D-35: the rows are the question
    reported[0]["token_end"] += 1
    assert len(join.diff(reported, oracle_rows(), {})) == 1


def test_rows_naming_a_document_are_counted_by_segment_and_step():
    rows = trace.flatten(DOC)
    assert len(join.rows_naming(rows, T["shard_position"], T["ordinal"])) == 1
    assert join.rows_naming(rows, T["shard_position"], T["ordinal"], steps=range(8, 10)) == []
    assert join.rows_naming(rows, T["shard_position"], T["ordinal"], segments=["E/seg-1"]) == []


# --- The replayed arrays against the kept cache ---------------------------------------------

KINDS = trace.PER_SHARD_KINDS + trace.BLEND_KINDS
RECORD = {"shards": [{"position": 0, "prefix": "shards/shard-1"},
                     {"position": 1, "prefix": "shards/shard-2"}]}


def write_cache(folder: Path) -> dict[tuple[str, int | None], np.ndarray]:
    """A cache folder as Megatron writes one: per dataset a description naming its path
    and its arrays under the same stem, and the blend's arrays under the blend's stem."""
    folder.mkdir(parents=True)
    made = {}
    for position, (stem, name) in enumerate((("aa", "shard-1"), ("bb", "shard-2"))):
        prefix = f"{stem}-GPTDataset-train-"
        (folder / f"{prefix}description.txt").write_text(
            json.dumps({"class": "GPTDataset", "dataset_path": f"/elsewhere/{name}"}))
        for number, kind in enumerate(trace.PER_SHARD_KINDS):
            array = np.arange(6, dtype=np.int64) * (position + 2) + number
            np.save(folder / f"{prefix}{kind}.npy", array)
            made[kind, position] = array
    for number, kind in enumerate(trace.BLEND_KINDS):
        array = np.arange(4, dtype=np.int16) + number
        np.save(folder / f"cc-BlendedDataset-train-{kind}.npy", array)
        made[kind, None] = array
    return made


def write_replay(out: Path, made: dict) -> dict:
    out.mkdir(parents=True)
    entries = []
    for (kind, shard), array in made.items():
        name = f"{kind}.npy" if shard is None else f"{kind}-{shard}.npy"
        np.save(out / name, array)
        entries.append({"kind": kind, "shard": shard, "file": name, "digest": "d",
                        "recorded_digest": "d", "equal": True})
    return {"segment": "E/seg-1", "arrays": entries}


def test_the_kept_cache_files_are_found_by_kind_and_shard_from_the_descriptions(tmp_path):
    made = write_cache(tmp_path / "cache")
    kept = trace.cache_files(tmp_path / "cache", RECORD)
    assert sorted(kept, key=str) == sorted(made, key=str)
    assert kept["sample_index", 1].name == "bb-GPTDataset-train-sample_index.npy"
    assert kept["dataset_index", None].name == "cc-BlendedDataset-train-dataset_index.npy"
    assert len(kept) == 8


def test_a_description_that_names_no_shard_of_the_record_is_an_error(tmp_path):
    write_cache(tmp_path / "cache")
    with pytest.raises(trace.TraceError, match="no shard of the run record"):
        trace.cache_files(tmp_path / "cache", {"shards": [{"position": 0, "prefix": "x/other"}]})


def test_replayed_arrays_equal_to_the_kept_ones_give_no_finding(tmp_path):
    made = write_cache(tmp_path / "cache")
    printed = write_replay(tmp_path / "out", made)
    kept = trace.cache_files(tmp_path / "cache", RECORD)
    assert trace.compare_replay(printed, tmp_path / "out", kept, KINDS) == []


def test_a_changed_value_a_changed_dtype_and_a_changed_shape_are_each_found(tmp_path):
    made = write_cache(tmp_path / "cache")
    kept = trace.cache_files(tmp_path / "cache", RECORD)
    changed = dict(made)
    changed["document_index", 0] = made["document_index", 0] + 1
    changed["sample_index", 1] = made["sample_index", 1].astype(np.int32)
    changed["shuffle_index", 0] = made["shuffle_index", 0][:-1]
    printed = write_replay(tmp_path / "out", changed)
    found = trace.compare_replay(printed, tmp_path / "out", kept, KINDS)
    assert len(found) == 3
    assert any("bytes differ" in f and "document_index" in f for f in found)
    assert any("dtype" in f and "sample_index" in f for f in found)
    assert any("shape" in f and "shuffle_index" in f for f in found)


def test_an_array_that_was_not_written_and_one_the_cache_does_not_hold_are_found(tmp_path):
    made = write_cache(tmp_path / "cache")
    kept = trace.cache_files(tmp_path / "cache", RECORD)
    fewer = {k: v for k, v in made.items() if k[0] != "dataset_index"}
    printed = write_replay(tmp_path / "out", fewer)
    found = trace.compare_replay(printed, tmp_path / "out", kept, KINDS)
    assert any("no array of the kind dataset_index" in f for f in found)
    assert any("kept in the cache and not written" in f for f in found)
    extra = write_replay(tmp_path / "out2", {**made, ("shuffle_index", 5): made["shuffle_index", 0]})
    assert any("the kept cache holds no file" in f
               for f in trace.compare_replay(extra, tmp_path / "out2", kept, KINDS))
