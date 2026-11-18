"""Checks of the oracle that need no Megatron environment: the wall, the digests,
the config rules, the records and the positions report.

The oracle is where expected answers come from, so it must share no code with the
tool (plan W5). These tests may import both sides and compare them.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from oracle import config as oracle_config
from oracle import digests, environment, positions, records
from oracle.errors import ConfigError, OracleError

ORACLE_DIR = Path(oracle_config.__file__).resolve().parent

# Modules that must run where neither Megatron nor torch nor numpy is installed.
# fixture_runs and seed_search reach Megatron only through oracle.pipeline, which they
# import when they run a segment.
STDLIB_ONLY = (
    "__init__", "cli", "config", "digests", "environment", "errors", "positions", "records",
    "fixture_spec", "log_rows", "search_conditions", "search_evidence", "fixture_runs", "seed_search",
)


def _imports(path: Path) -> set[str]:
    """Top-level names of every module a file imports, at any depth of the file."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


# --- The wall (plan W5) ---------------------------------------------------------------


def test_oracle_imports_nothing_from_the_tool():
    files = sorted(ORACLE_DIR.glob("*.py"))
    assert files
    for path in files:
        assert "waterwhite" not in _imports(path), path.name
        assert "replica" not in _imports(path), path.name


def test_only_named_modules_import_megatron_or_torch():
    for name in STDLIB_ONLY:
        imported = _imports(ORACLE_DIR / f"{name}.py")
        # environment.py imports megatron and torch inside two functions to read
        # their versions; nothing else in this list may name them at all.
        allowed = {"megatron", "torch"} if name == "environment" else set()
        assert not (imported & {"megatron", "torch", "numpy", "transformers"}) - allowed, name


# --- Digests: the oracle's own code against the tool's definition ---------------------------


def test_array_digest_equals_the_tools_definition():
    from waterwhite import ids

    cases = [
        ("<i4", (3,), bytes(range(12))),
        ("<u4", (0,), b""),
        ("<i4", (2, 2), bytes(range(16))),
        ("<i8", (1,), b"\x01" + b"\x00" * 7),
    ]
    for dtype, shape, data in cases:
        assert digests.array_digest(dtype, shape, data) == ids.array_digest(dtype, shape, data)
    # the shape is part of the digest: the same bytes as 4 values or as 2 x 2
    assert digests.array_digest("<i4", (4,), bytes(16)) != digests.array_digest("<i4", (2, 2), bytes(16))


def test_order_digest_equals_the_tools_definition():
    from waterwhite import ids

    for indices in ([], [0], [5, 3, 9, 0], list(range(100))):
        assert digests.order_digest(indices) == ids.order_digest(indices)
    assert digests.order_digest([1, 2]) != digests.order_digest([2, 1])


def test_file_digest_equals_the_tools_definition(tmp_path):
    from waterwhite import ids

    path = tmp_path / "f.bin"
    path.write_bytes(b"waterwhite" * 1000)
    assert digests.file_sha256(path) == ids.file_digest(path)


def test_compare_digests_names_each_difference():
    recorded = [
        {"kind": "document_index", "shard": 0, "digest": "a"},
        {"kind": "dataset_index", "shard": None, "digest": "b"},
        {"kind": "shuffle_index", "shard": 1, "digest": "c"},
    ]
    replayed = [
        {"kind": "document_index", "shard": 0, "digest": "a"},
        {"kind": "dataset_index", "shard": None, "digest": "x"},
        {"kind": "sample_index", "shard": 0, "digest": "d"},
    ]
    result = digests.compare_digests(replayed, recorded)
    assert result["all_equal"] is False
    assert [(r["kind"], r["shard"], r["equal"]) for r in result["rows"]] == [
        ("document_index", 0, True),
        ("dataset_index", None, False),
    ]
    assert result["only_recorded"] == [["shuffle_index", 1]]
    assert result["only_replayed"] == [["sample_index", 0]]
    assert digests.compare_digests(recorded, recorded)["all_equal"] is True
    with pytest.raises(OracleError):
        digests.compare_digests(recorded + recorded[:1], recorded)


# --- Config rules ---------------------------------------------------------------------------


@pytest.fixture
def files(tmp_path):
    """Stand-in files, so the path checks of the config pass. Nothing reads them."""
    for name in ("s1", "s2", "s3"):
        (tmp_path / f"{name}.bin").write_bytes(name.encode() * 4)
        (tmp_path / f"{name}.idx").write_bytes(name.encode() * 2)
    (tmp_path / "vocab.json").write_text("{}")
    (tmp_path / "merges.txt").write_text("")
    return tmp_path


def _segment(**changes) -> dict:
    raw = {
        "run": "E",
        "segment": "E/seg-1",
        "start_kind": "fresh",
        "shards": ["s1", "s2"],
        "tokenizer": {"type": "GPT2BPETokenizer", "vocab_file": "vocab.json", "merges_file": "merges.txt"},
        "settings": {
            "dataloader_type": "single",
            "micro_batch_size": 4,
            "global_batch_size": 4,
            "sequence_length": 32,
            "random_seed": 7,
            "split": "1,0,0",
            "train_iters": 6,
        },
        "launches": [
            {"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0, "step_at_start": 0, "steps": 6}
        ],
        "cache_dir": "cache/E-seg-1",
        "step_record": {"checkpoints": [{"name": "E@6", "step": 6}]},
        "outputs": {
            "digests": "out/d.json",
            "run_record": "out/r.json",
            "step_record": "out/s.json",
            "log": "out/l.jsonl",
            "summary": "out/m.json",
        },
    }
    for key, value in changes.items():
        if key == "settings":
            raw["settings"] = {**raw["settings"], **value}
        else:
            raw[key] = value
    return raw


def test_segment_config_is_read_with_megatrons_defaults(files):
    segment = oracle_config.parse_segment(_segment(), files)
    assert segment.data.requested_samples == 24  # train_iters times the global batch size
    assert segment.data.add_extra_token_to_sequence is True
    assert segment.data.sample_tokens == 33
    assert segment.data_sharding is True
    assert segment.steps() == [1, 2, 3, 4, 5, 6]
    assert segment.data.shards == (files / "s1", files / "s2")
    assert segment.data.cache_dir == files / "cache" / "E-seg-1"
    assert segment.micro_batches(segment.launches[0]) == 1
    with_samples = oracle_config.parse_segment(_segment(settings={"train_samples": 40}), files)
    assert with_samples.data.requested_samples == 40


@pytest.mark.parametrize(
    "changes, word",
    [
        ({"settings": {"dataloader_typ": "single"}}, "unknown key"),
        ({"extra": 1}, "unknown key"),
        ({"shards": []}, "at least one"),
        ({"shards": ["s1", "s1"]}, "twice"),
        ({"shards": ["s1", "nope"]}, "no file"),
        ({"start_kind": "restart"}, "start_kind"),
        ({"start_kind": "warm-start"}, "parent_checkpoint"),
        ({"allow_sequential_restart": "yes"}, "true or false"),
        ({"parent_checkpoint": "X@1"}, "fresh"),
        ({"settings": {"dataloader_type": "batch"}}, "dataloader_type"),
        ({"settings": {"micro_batch_size": 3}}, "not a multiple"),
        ({"settings": {"micro_batches_per_step": 2}}, "micro_batches_per_step"),
        ({"settings": {"sequence_length": True}}, "integer"),
        ({"settings": {"add_extra_token_to_sequence": False}}, "extra token on only"),
        ({"tokenizer": {"type": "NullTokenizer", "vocab_file": "vocab.json", "merges_file": "merges.txt"}}, "not supported"),
        ({"step_record": {"lowered": [{"step": 9, "last_sample_tokens_read": 16}]}}, "not a step"),
        ({"step_record": {"lowered": [{"step": 2, "last_sample_tokens_read": 33}]}}, "does not lower"),
        ({"step_record": {"checkpoints": [{"name": "E@7", "step": 7}]}}, "not a step"),
        ({"sampler_file": "missing/samplers.py"}, "sampler_file"),
    ],
)
def test_segment_config_refuses(files, changes, word):
    with pytest.raises(ConfigError) as error:
        oracle_config.parse_segment(_segment(**changes), files)
    assert word in str(error.value)


def test_launches_must_not_share_a_step(files):
    launches = [
        {"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 4},
        {"world_size": 2, "data_parallel_size": 2, "consumed_at_start": 8, "step_at_start": 2, "steps": 4},
    ]
    with pytest.raises(ConfigError, match="logged once"):
        oracle_config.parse_segment(_segment(launches=launches, settings={"micro_batch_size": 2}), files)


def test_micro_batches_per_step_follow_the_sizes(files):
    launches = [
        {"world_size": 4, "data_parallel_size": 2, "consumed_at_start": 0, "step_at_start": 0, "steps": 2},
    ]
    segment = oracle_config.parse_segment(
        _segment(launches=launches, step_record={}, settings={"micro_batch_size": 2, "global_batch_size": 8}),
        files,
    )
    # global batch 8 = 2 ranks x micro batch 2 x 2 micro batches each
    assert segment.micro_batches(segment.launches[0]) == 2
    with pytest.raises(ConfigError, match="does not divide"):
        launches[0]["world_size"] = 3
        oracle_config.parse_segment(
            _segment(launches=launches, step_record={}, settings={"micro_batch_size": 2}), files
        )


def test_paths_resolve_by_the_stated_rules(files, monkeypatch):
    monkeypatch.setenv(oracle_config.ENV_VARIABLE, str(files))
    assert oracle_config.resolve_path("a/b", files) == files / "a" / "b"
    assert oracle_config.resolve_path("x/../s1", files) == files / "s1"
    assert oracle_config.resolve_path("$WATERWHITE_MEGATRON_ENV/s1", Path("/nowhere")) == files / "s1"
    with pytest.raises(ConfigError):
        oracle_config.resolve_path("$HOME/x", files)
    monkeypatch.delenv(oracle_config.ENV_VARIABLE)
    with pytest.raises(OracleError):
        oracle_config.resolve_path("$WATERWHITE_MEGATRON_ENV/s1", files)


def test_an_existing_output_is_never_overwritten(files):
    target = files / "out" / "x.json"
    oracle_config.write_json(target, {"b": 1, "a": 2})
    assert target.read_text() == '{\n  "a": 2,\n  "b": 1\n}\n'
    with pytest.raises(ConfigError, match="exists already"):
        oracle_config.write_json(target, {})


def _three_segments() -> dict:
    seg2 = _segment(
        segment="E/seg-2",
        start_kind="resume",
        parent_checkpoint="E@6",
        shards=["s1", "s2", "s3"],
        settings={"train_iters": 12},
        launches=[
            {"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24, "step_at_start": 6, "steps": 6}
        ],
        cache_dir="cache/E-seg-2",
        step_record={"lowered": [{"step": 8, "last_sample_tokens_read": 16}], "checkpoints": [{"name": "E@9", "step": 9}]},
        outputs={k: f"out2/{k}" for k in oracle_config.OUTPUT_KEYS},
    )
    warm = _segment(
        run="K",
        segment="K/seg-1",
        start_kind="warm-start",
        parent_checkpoint="E@6",
        cache_dir="cache/K-seg-1",
        step_record={},
        outputs={k: f"out3/{k}" for k in oracle_config.OUTPUT_KEYS},
    )
    return {"segments": [_segment(), seg2, warm]}


def test_several_segments_and_how_each_starts(files):
    segments = oracle_config.parse_segments(_three_segments(), files)
    starts = oracle_config.check_starts(segments)
    assert starts["E/seg-1"].startswith("fresh")
    assert "step 6 and consumed count 24" in starts["E/seg-2"]
    assert starts["K/seg-1"].startswith("warm-start")


def test_a_resume_must_start_where_its_checkpoint_stood(files):
    raw = _three_segments()
    raw["segments"][1]["launches"][0]["consumed_at_start"] = 20
    with pytest.raises(ConfigError, match="consumed count 24"):
        oracle_config.check_starts(oracle_config.parse_segments(raw, files))
    raw = _three_segments()
    raw["segments"][0]["launches"][0]["consumed_at_start"] = 4
    with pytest.raises(ConfigError, match="fresh"):
        oracle_config.check_starts(oracle_config.parse_segments(raw, files))


def test_a_resume_may_name_no_checkpoint(files):
    """A resume of a run that registered no checkpoint at that step (the fixture's
    round-robin variant of run E): the segment is read, its record holds null,
    and the check of the starts says that the counters were not checked."""
    raw = _three_segments()
    del raw["segments"][1]["parent_checkpoint"]
    segments = oracle_config.parse_segments(raw, files)
    assert segments[1].start_kind == "resume" and segments[1].parent_checkpoint is None
    assert "no parent checkpoint is named" in oracle_config.check_starts(segments)["E/seg-2"]
    record = records.run_record(
        segments[1], record_folder=files, sampler_class="x.Y", versions={}, cache_rows=[], shuffle_rows=[]
    )
    assert (record["start_kind"], record["parent_checkpoint"]) == ("resume", None)


def test_a_sequential_restart_is_off_unless_the_config_allows_it(files):
    assert oracle_config.parse_segment(_segment(), files).allow_sequential_restart is False
    assert oracle_config.parse_segment(_segment(allow_sequential_restart=True), files).allow_sequential_restart is True


def test_defaults_are_merged_key_by_key(files):
    raw = _segment()
    defaults = {"tokenizer": raw.pop("tokenizer"), "settings": {"sequence_length": 32, "split": "1,0,0", "random_seed": 1}}
    raw["settings"] = {k: v for k, v in raw["settings"].items() if k not in ("sequence_length", "split")}
    segments = oracle_config.parse_segments({"defaults": defaults, "segments": [raw]}, files)
    assert segments[0].data.sequence_length == 32
    assert segments[0].data.random_seed == 7  # the segment's value replaces the default


# --- Records ----------------------------------------------------------------------------------


def test_step_record_lowers_only_the_named_step(files):
    segments = oracle_config.parse_segments(_three_segments(), files)
    record = records.step_record(segments[1])
    assert record == {
        "segment": "E/seg-2",
        "steps": [
            {"step": 7, "batch_size": 4},
            {"step": 8, "batch_size": 4, "last_sample_tokens_read": 16},
            {"step": 9, "batch_size": 4},
            {"step": 10, "batch_size": 4},
            {"step": 11, "batch_size": 4},
            {"step": 12, "batch_size": 4},
        ],
        "checkpoints": [{"name": "E@9", "step": 9}],
    }


def test_run_record_has_the_fields_of_the_interface(files):
    segment = oracle_config.parse_segments(_three_segments(), files)[1]
    record = records.run_record(
        segment,
        record_folder=files / "out2",
        sampler_class="some.module.Sampler",
        versions={"release_tag": "t", "megatron_core": "c", "megatron_bridge": None, "torch": "x"},
        cache_rows=[{"kind": "document_index", "shard": 0, "digest": "d", "dtype": "<i4", "shape": [3]}],
        shuffle_rows=[],
    )
    # the field names of docs/interfaces.md 8.1
    assert set(record) == {
        "format", "run", "segment", "start_kind", "parent_checkpoint", "settings", "versions",
        "shards", "launches", "cache_digests", "shuffle_digests",
    }
    assert record["format"] == "waterwhite.run-record/1"
    assert set(record["settings"]) == {
        "dataloader_type", "sampler_class", "data_sharding", "micro_batch_size", "global_batch_size",
        "sequence_length", "add_extra_token_to_sequence", "random_seed", "split", "train_iters",
        "train_samples", "batch_schedule",
    }
    assert record["start_kind"] == "resume" and record["parent_checkpoint"] == "E@6"
    assert [s["prefix"] for s in record["shards"]] == ["../s1", "../s2", "../s3"]
    assert record["shards"][0]["bin_sha256"] == digests.file_sha256(files / "s1.bin")
    assert record["shards"][2]["idx_sha256"] == digests.file_sha256(files / "s3.idx")
    assert record["launches"] == [
        {"position": 0, "world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24,
         "step_at_start": 6, "first_step": 7, "last_step": 12}
    ]
    assert record["cache_digests"] == [{"kind": "document_index", "shard": 0, "digest": "d"}]
    assert record["shuffle_digests"] == []
    json.dumps(record)


# --- The release tag is read from the checkout's files -----------------------------------------


def test_checkout_ref_reads_tag_and_commit(tmp_path):
    commit = "685fb8bd970185e9ac3283fe32a5d6667bf4b268"
    other = "04ce2b3efd6f2e387fc244e9f37afd0a1f7952fb"
    git = tmp_path / "repo" / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(commit + "\n")
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted \n"
        f"{commit} refs/tags/core_v0.20.0\n"
        f"{other} refs/tags/elsewhere\n"
    )
    assert environment.checkout_ref(tmp_path / "repo") == {"commit": commit, "tags": ["core_v0.20.0"]}
    assert environment.ref_name(tmp_path / "repo") == "core_v0.20.0"
    # a branch checkout with no tag on its commit is named by the commit
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "main").write_text("1" * 40 + "\n")
    assert environment.ref_name(tmp_path / "repo") == "1" * 40
    # an annotated tag is matched through its peeled line
    (git / "packed-refs").write_text(f"{other} refs/tags/v9\n^{'1' * 40}\n")
    assert environment.ref_name(tmp_path / "repo") == "v9"
    inside = tmp_path / "repo" / "a" / "b.py"
    inside.parent.mkdir()
    inside.write_text("")
    assert environment.checkout_of(inside) == tmp_path / "repo"


# --- The positions report ------------------------------------------------------------------------


def _row(segment, step, sample_pos, global_index, shard, ordinal, start, end):
    return {
        "segment": segment, "step": step, "sample_pos": sample_pos, "global_index": global_index,
        "rank": 0, "shard_position": shard, "ordinal": ordinal, "token_start": start, "token_end": end,
    }


def _hand_log() -> list[dict]:
    """A log written by hand, sequence length 8 (9 tokens per sample).

    Shard a: documents a0 (6 tokens), a1 (10), a2 (9). Shard b: b0 (12), b1 (20).
    Sample 0 reads a0 whole and the first three tokens of a1, the third as its
    extra token. Sample 1 goes on in a1 from token 2 to its end and takes its extra
    token from a2. Sample 2 is in shard b. In segment two, sample 0 of step 1 reads
    b0's start again."""
    return [
        _row("S/1", 1, 0, 0, 0, 0, 0, 6),
        _row("S/1", 1, 0, 0, 0, 1, 0, 3),
        _row("S/1", 1, 1, 1, 0, 1, 2, 10),
        _row("S/1", 1, 1, 1, 0, 2, 0, 1),
        _row("S/1", 2, 0, 2, 1, 0, 0, 9),
        _row("S/2", 1, 0, 0, 0, 0, 0, 9),
    ]


def _hand_report(log=None):
    return positions.build_report(
        _hand_log() if log is None else log,
        {"S/1": ["a.jsonl", "b.jsonl"], "S/2": ["b.jsonl"]},
        {"a.jsonl": ["a0", "a1", "a2"], "b.jsonl": ["b0", "b1"]},
    )


def test_positions_report_names_records_per_step():
    report = _hand_report()
    assert [(s["segment"], s["step"]) for s in report["steps"]] == [("S/1", 1), ("S/1", 2), ("S/2", 1)]
    first = report["steps"][0]["samples"]
    assert [[r["record"] for r in s["rows"]] for s in first] == [["a0", "a1"], ["a1", "a2"]]
    assert [s["tokens"] for s in first] == [9, 9]
    # in segment two, shard position 0 is b.jsonl
    assert report["steps"][2]["samples"][0]["rows"][0]["record"] == "b0"


def test_positions_report_flags_a_split_document():
    split = _hand_report()["flags"]["split_across_samples"]
    assert [(s["record"], s["several_parts_read"]) for s in split] == [("a1", True)]
    assert [(r["token_start"], r["token_end"]) for r in split[0]["reads"]] == [(0, 3), (2, 10)]
    # with the first sample left out, the log still shows a1 starts inside
    later = _hand_report(_hand_log()[2:])["flags"]["split_across_samples"]
    assert [(s["record"], s["several_parts_read"]) for s in later] == [("a1", False)]


def test_positions_report_flags_a_read_through_the_extra_token_only():
    only = _hand_report()["flags"]["read_through_extra_token_only"]
    assert [(o["record"], o["step"], o["token_start"], o["document_read_otherwise"]) for o in only] == [
        ("a2", 1, 0, False)
    ]
    # once another sample reads a2's body, the flag says so
    log = _hand_log()
    log.insert(5, _row("S/1", 3, 0, 3, 0, 2, 0, 9))
    only = _hand_report(log)["flags"]["read_through_extra_token_only"]
    assert only[0]["document_read_otherwise"] is True
    assert only[0]["first_other_read"] == {"segment": "S/1", "step": 3}


def test_positions_report_flags_a_repeated_draw():
    repeated = _hand_report()["flags"]["read_in_more_than_one_step"]
    assert [(r["record"], r["same_tokens_read_again"]) for r in repeated] == [("b0", True)]
    assert repeated[0]["steps"] == [["S/1", 2], ["S/2", 1]]
    # a1 is read by two samples of one step: split, not read in two steps
    assert "a1" not in [r["record"] for r in repeated]


def test_positions_report_flags_the_same_ordinal_in_two_shards():
    same = _hand_report()["flags"]["same_ordinal_in_two_shards"]
    assert [s["ordinal"] for s in same] == [0]
    assert [(x["shard"], x["record"]) for x in same[0]["shards"]] == [("a.jsonl", "a0"), ("b.jsonl", "b0")]


def test_positions_report_as_text_and_its_refusals():
    text = positions.render(_hand_report())
    assert "S/1 step 1" in text
    assert "a0[0:6] a1[0:3]~" in text
    assert "a2[0:1]!extra-only" in text
    with pytest.raises(ConfigError, match="not in the config"):
        positions.build_report(
            _hand_log(), {"S/1": ["a.jsonl", "b.jsonl"]}, {"a.jsonl": ["a0", "a1", "a2"], "b.jsonl": ["b0", "b1"]}
        )
    with pytest.raises(ConfigError, match="input lines"):
        positions.build_report(
            _hand_log(), {"S/1": ["a.jsonl", "b.jsonl"], "S/2": ["b.jsonl"]}, {"a.jsonl": ["a0"], "b.jsonl": ["b0"]}
        )
    with pytest.raises(ConfigError, match="not consecutive"):
        _hand_report(_hand_log() + [_row("S/1", 1, 0, 0, 0, 0, 0, 6)])


def test_positions_script_reads_its_config(tmp_path, capsys):
    (tmp_path / "a.jsonl").write_text("".join(json.dumps({"id": i, "text": "t"}) + "\n" for i in ("a0", "a1", "a2")))
    (tmp_path / "b.jsonl").write_text("".join(json.dumps({"id": i, "text": "t"}) + "\n" for i in ("b0", "b1")))
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _hand_log()))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "logs": ["log.jsonl"],
                "segments": {"S/1": ["a.jsonl", "b.jsonl"], "S/2": ["b.jsonl"]},
                "outputs": {"report": "out/report.json"},
            }
        )
    )
    assert positions.main([str(tmp_path / "config.json")]) == 0
    assert "Documents split across samples: 1" in capsys.readouterr().out
    written = json.loads((tmp_path / "out" / "report.json").read_text())
    assert written == json.loads(json.dumps(_hand_report()))
    # a second run would overwrite the report: refused, exit 2, nothing on standard output
    assert positions.main([str(tmp_path / "config.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "exists already" in captured.err
