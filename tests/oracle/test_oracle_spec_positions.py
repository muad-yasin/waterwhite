"""The fixture spec as it is committed (plan 5.1): its seeds are written, nothing
is pending, and every ``[positions]`` table agrees with the oracle log lines it
quotes.

No Megatron is needed: the log lines are the ones copied into the spec. What
these tests can show is that the spec does not contradict itself, and that a
record it names sits at the ordinal it names in a build of the spec (the replica
without tokenization). That the lines are the oracle's lines on the final build
is shown by the fixture runs and ``seed_search.py --check-runs`` (docs/oracle.md,
"The search that fixed the spec's seeds").
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from oracle import log_rows
from oracle.fixture_spec import load_spec

REPO = Path(__file__).resolve().parents[2]
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
SAMPLE_TOKENS = 33  # sequence_length + 1


@pytest.fixture(scope="module")
def raw() -> dict:
    with open(SPEC_FILE, "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def positions(raw) -> dict:
    return raw["positions"]


def lines_of(table: dict) -> list[dict]:
    """A table's log lines as rows. Each is one line of the oracle's log: a JSON
    object with the log's fields in the log's order."""
    rows = []
    for line in table["log_lines"]:
        assert isinstance(line, str)
        row = json.loads(line)
        assert list(row) == list(log_rows.LOG_FIELDS), line
        assert json.dumps(row) == line  # as the log writes it, to the character
        rows.append(row)
    return rows


def span(row: dict) -> tuple[int, int, int, int]:
    return (row["shard_position"], row["ordinal"], row["token_start"], row["token_end"])


@pytest.fixture(scope="module")
def tokenizer_inputs(tmp_path_factory) -> dict[str, list[str]]:
    """The record id of every tokenizer-input line of the partitioned layout, from
    one build of the spec by the replica, without tokenization. Line i (from 0)
    is document i of the shard."""
    from replica.build import BuildOptions, build
    from replica.plan import load_fixture_spec

    out = tmp_path_factory.mktemp("spec-positions") / "out"
    build(load_fixture_spec(SPEC_FILE), out, BuildOptions(tokenize=False, layouts=("partitioned",)))
    ids = {}
    for shard in ("shard-1", "shard-2", "shard-3"):
        path = out / "pipeline" / "partitioned" / "tokenizer-input" / f"{shard}.jsonl"
        ids[shard] = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
    return ids


def test_nothing_is_pending_and_every_seed_is_written(raw):
    assert raw["pending"] == {"items": []}
    assert "PENDING" not in SPEC_FILE.read_text(encoding="utf-8")
    spec = load_spec(SPEC_FILE)
    assert {name: run.random_seed for name, run in spec.runs.items()} == {
        "E": 578, "E-rr": 578, "D": 0, "Y": 0, "K": 4,
    }
    assert (raw["layouts"]["partitioned"]["shuffle_seed"], raw["layouts"]["round-robin"]["shuffle_seed"]) == (0, 0)
    assert raw["tokenizer"]["eod_token_id"] == 50256 and raw["tokenizer"]["eod_in_span"] is True
    # every table of [positions] names its conditions and quotes log lines (or none)
    numbers = {condition.n for condition in spec.conditions}
    for name, table in raw["positions"].items():
        assert set(table["conditions"]) <= numbers, name
        assert isinstance(table["log_lines"], list), name
        assert "fill" not in table, name
    covered = sorted(n for table in raw["positions"].values() for n in table["conditions"])
    assert sorted(set(covered)) == sorted(numbers - {1})  # condition 1 fixes sizes, not a position


def test_the_round_robin_variant_resumes_from_its_registered_checkpoint(raw):
    rr = raw["runs"]["E-rr"]
    assert rr["checkpoints"] == [6]
    assert rr["step_record"]["E-rr/seg-1"]["checkpoints"] == [{"name": "E-rr@6", "step": 6}]
    assert rr["segments"]["E-rr/seg-2"]["parent_checkpoint"] == "E-rr@6"
    assert rr["step_record"]["E-rr/seg-1"]["steps"] == [] and rr["step_record"]["E-rr/seg-2"]["steps"] == []


def test_t_is_whole_in_one_sample_of_step_7_as_its_second_document(raw, positions, tokenizer_inputs):
    table = positions["T"]
    rows = lines_of(table)
    (sample,) = log_rows.samples_of(rows)
    assert (sample["segment"], sample["step"], sample["sample_pos"], sample["global_index"]) == (
        "E/seg-2", 7, table["sample_pos"], table["global_index"])
    assert log_rows.sample_tokens(sample) == SAMPLE_TOKENS
    first, second, third = rows
    tokens_of_t = raw["sizes"]["tokens_of_T"]
    assert span(second) == (table["shard_position"], table["ordinal"], 0, tokens_of_t)
    assert table["token_end"] == table["fixed"]["token_end"] == tokens_of_t == 22
    assert table["rows_in_sample"] == 3 and table["fixed"]["row_in_sample"] == 2
    as_neighbour = lambda row, record: {"record": record, "shard_position": row["shard_position"],  # noqa: E731
                                        "ordinal": row["ordinal"], "token_start": row["token_start"],
                                        "token_end": row["token_end"]}
    names = tokenizer_inputs["shard-3"]
    assert names[second["ordinal"]] == "T" == table["record"]
    assert table["first_row"] == as_neighbour(first, names[first["ordinal"]])
    assert table["neighbours"] == [table["first_row"], as_neighbour(third, names[third["ordinal"]])]
    assert all(n["record"].startswith("Z-") and n["shard_position"] == 2 for n in table["neighbours"])
    assert table["rows_of_T_in_run_E"] == [
        {"segment": "E/seg-2", "step": 7, "sample_pos": table["sample_pos"], "token_start": 0, "token_end": 22}]
    assert table["rows_in_steps_1_to_6_on_shard_position_2"] == 0


def test_w_is_one_token_of_an_unrelated_record_of_shard_3(positions, tokenizer_inputs):
    table = positions["shared_token"]
    rows = lines_of(table)
    (sample,) = log_rows.samples_of(rows)
    assert log_rows.sample_tokens(sample) == SAMPLE_TOKENS
    last = rows[-1]
    assert span(last) == (2, table["ordinal"], 0, 1) and table["shard_position"] == 2
    assert (sample["segment"], sample["step"], sample["sample_pos"], sample["global_index"]) == (
        "E/seg-2", table["step"], table["sample_pos"], table["global_index"])
    assert table["step"] in (7, 8, 9) and (table["step"], table["sample_pos"]) != (8, 3)
    names = tokenizer_inputs["shard-3"]
    assert names[table["ordinal"]] == table["record"] and table["record"].startswith("Z-") and table["source"] == "Z"
    # not the sample that holds T, and not a row of T
    t = positions["T"]
    assert (sample["step"], sample["sample_pos"]) != (7, t["sample_pos"])
    assert all(row["ordinal"] != t["ordinal"] for row in rows)
    assert (table["fixed"]["shard"], table["fixed"]["before_checkpoint"], table["fixed"]["after_checkpoint"]) == (
        "shard-3", "E@6", "E@9")


def test_the_cut_of_step_8_is_the_rule_applied_to_the_quoted_lines(raw, positions, tokenizer_inputs):
    table = positions["step8_cut"]
    rows = lines_of(table)
    (sample,) = log_rows.samples_of(rows)
    assert (sample["segment"], sample["step"], sample["sample_pos"]) == ("E/seg-2", 8, 3)
    (lowered,) = raw["runs"]["E"]["step_record"]["E/seg-2"]["steps"]
    assert lowered == {"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}
    cut = log_rows.cut_sample(sample, lowered["last_sample_tokens_read"])
    assert cut["sample_tokens"] == SAMPLE_TOKENS == table["fixed"]["sample_tokens"]
    assert cut["starts_a_row"] is False
    assert cut["cut"] == {"shard_position": table["shard_position"], "ordinal": table["ordinal"], "token": table["token"]}
    assert [dict(zip(("shard_position", "ordinal", "token_start", "token_end"), span(row))) for row in cut["covered_rows"]] == (
        table["rows_of_the_covered_part"])
    assert table["uncovered_in_sample"] == [16, 33]
    names = tokenizer_inputs["shard-3"]
    assert [names[row["ordinal"]] for row in rows] == table["records_of_the_last_sample_of_step_8"]
    assert table["record"] == names[table["ordinal"]] and "T" not in table["records_of_the_last_sample_of_step_8"]


def test_the_other_positions_of_run_e_agree_with_their_lines(positions, tokenizer_inputs):
    shards = ("shard-1", "shard-2", "shard-3")
    record = lambda row: tokenizer_inputs[shards[row["shard_position"]]][row["ordinal"]]  # noqa: E731

    skipped = positions["skipped_line"]
    (row,) = lines_of(skipped)
    assert (row["shard_position"], row["ordinal"], record(row)) == (1, skipped["ordinal"], skipped["record"])
    # document i is the (i + 1)-th line the sanitizer kept; line 6 was skipped
    assert skipped["sanitizer_input_line"] == skipped["ordinal"] + 2 > skipped["fixed"]["line"] == 6
    assert len(tokenizer_inputs["shard-2"]) == skipped["fixed"]["documents"] == 19

    repeated = positions["repeated_draw"]
    first, second = lines_of(repeated)
    assert (first["shard_position"], first["ordinal"]) == (second["shard_position"], second["ordinal"]) == (
        repeated["shard_position"], repeated["ordinal"])
    assert first["step"] <= 6 < second["step"] and first["shard_position"] in (0, 1)
    assert record(first) == repeated["record"]
    assert repeated["same_range_read_again"] is (span(first) == span(second))

    ordinal = positions["shared_ordinal"]
    first, second = lines_of(ordinal)
    assert first["ordinal"] == second["ordinal"] == ordinal["ordinal"]
    assert [first["shard_position"], second["shard_position"]] == ordinal["shard_positions"] == [0, 1]
    assert [record(first), record(second)] == [read["record"] for read in ordinal["reads"]]

    split = positions["split_document"]
    first, second = lines_of(split)
    assert (first["shard_position"], first["ordinal"]) == (second["shard_position"], second["ordinal"])
    assert (first["segment"], first["step"], first["sample_pos"]) != (second["segment"], second["step"], second["sample_pos"])
    assert second["token_start"] == first["token_end"] - 1  # the token the two samples share
    assert min(log_rows.row_tokens(first), log_rows.row_tokens(second)) >= 2
    assert record(first) == split["record"] not in ("T", positions["shared_token"]["record"])

    both = lines_of(positions["seg1_shards"])
    assert sorted(row["shard_position"] for row in both) == [0, 1] and all(row["step"] <= 6 for row in both)


def test_the_positions_of_runs_d_y_and_k_agree_with_their_lines(raw, positions, tokenizer_inputs):
    d = positions["D"]
    samples = log_rows.samples_of(lines_of(d))
    assert [[s["step"], s["sample_pos"], s["global_index"]] for s in samples] == d["samples"]
    assert [s["global_index"] for s in samples] == list(range(12))
    assert all(log_rows.sample_tokens(s) == SAMPLE_TOKENS for s in samples)
    assert all(row["shard_position"] in (0, 1) for s in samples for row in s["rows"])

    y = positions["Y"]
    (row,) = lines_of(y)
    assert row["segment"] == "Y/seg-1" and row["shard_position"] == 2 == y["shard_position_of_T"]
    assert tokenizer_inputs["shard-3"][y["ordinal_of_T"]] == "T" and row["ordinal"] != y["ordinal_of_T"]
    assert y["rows_that_name_T"] == 0

    k = positions["K"]
    assert k["blend_length"] == raw["sizes"]["dataset_length"]["K/seg-1"]
    assert k["epoch_steps"] == k["blend_length"] // 4 and 9 <= k["epoch_steps"] <= 36
    assert k["step_after_the_boundary"] == k["epoch_steps"] + 1
    rows = lines_of(k)
    assert rows[0]["step"] == k["step_after_the_boundary"]
    assert rows[1]["step"] <= 3 and rows[1]["shard_position"] == 0  # shard-3 is position 0 in run K
    first_read = k["first_read_of_T"]
    assert first_read["step"] in (7, 8) and first_read["rank"] == 1 == k["fixed"]["first_read_of_T_rank"]
    (sample,) = log_rows.samples_of(rows[2:])
    assert (sample["step"], sample["sample_pos"], sample["global_index"], sample["rank"]) == (
        first_read["step"], first_read["sample_pos"], first_read["global_index"], first_read["rank"])
    assert log_rows.sample_tokens(sample) == SAMPLE_TOKENS
    assert any(row["shard_position"] == 0 and row["ordinal"] == y["ordinal_of_T"] for row in sample["rows"])
    assert k["rows_that_name_T_in_steps_1_to_8"] == [{"step": first_read["step"], "rank": 1}]


def test_the_exact_rerun_restart_is_not_before_step_7(positions):
    table = positions["exact_rerun_restart"]
    assert table["conditions"] == [] and table["step"] >= table["fixed"]["not_before_step"] == 7
    assert table["steps_that_differ"][0] == table["step"]
    historical, corrected = lines_of(table)
    assert historical["step"] == corrected["step"] == table["step"]
    assert (historical["sample_pos"], historical["global_index"]) == (corrected["sample_pos"], corrected["global_index"])
    assert span(historical) != span(corrected)


def test_the_sizes_agree_with_megatrons_rules_for_them(raw):
    """A shard of n tokens holds (n - 1) div 32 samples of 33 tokens that share
    their last token with the next one, and a blend without weights is as long
    as the smaller of the samples asked for and the samples its shards hold
    (docs/oracle.md, "Index build"). The numbers themselves are the oracle's."""
    sizes = raw["sizes"]
    for name, shard in sizes["shards"].items():
        assert shard["samples"] == (shard["tokens"] - 1) // 32, name
    spec = load_spec(SPEC_FILE)
    for run in spec.runs.values():
        for segment in run.segments:
            held = sum(sizes["shards"][ref.name]["samples"] for ref in segment.shards)
            asked = segment.train_iters * run.global_batch_size
            assert sizes["dataset_length"][segment.name] == min(asked, held), segment.name
    assert (sizes["tokens_of_X"], sizes["tokens_of_P"], sizes["tokens_of_T"]) == (16, 15, 22)
    assert sizes["shards"]["shard-2"]["documents"] == 19
