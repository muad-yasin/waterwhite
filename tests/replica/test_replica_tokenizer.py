"""The tokenizer step without Megatron: the index reader, the environment check,
what every subprocess is handed, the guard against a preprocessing run that hangs,
and the mixtures and capture points that follow tokenization (with the stand-in of
conftest.py for the calls that run Megatron). One test at the end runs the real preprocessing and
needs the Megatron environment."""

from __future__ import annotations

import dataclasses
import os
import struct
import sys
import time

import numpy
import pytest

from replica import megatron_tokenizer
from replica.build import BuildOptions, build, file_sha256
from replica.errors import SpecError, TokenizerError
from replica.megatron_tokenizer import ENV_VARIABLE, MegatronEnv
from replica.plan import TokenizerSettings
from replica.shard_index import read_shard_index


def index_bytes(document_lengths, code=8):
    """An index file as the preprocessing writes it: one sequence per document."""
    lengths = numpy.array(document_lengths, dtype="<i4")
    width = {8: 2, 4: 4}[code]
    pointers = numpy.concatenate(([0], numpy.cumsum(lengths[:-1], dtype="<i8") * width)).astype("<i8")
    marks = numpy.arange(len(lengths) + 1, dtype="<i8")
    return (b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, code, len(lengths), len(marks))
            + lengths.tobytes() + pointers.tobytes() + marks.tobytes())


def test_the_index_reader_counts_documents_and_tokens(tmp_path):
    path = tmp_path / "shard.idx"
    path.write_bytes(index_bytes([5, 1, 12]))
    index = read_shard_index(path)
    assert (index.documents, index.tokens, index.document_tokens, index.token_bytes) == (
        3, 18, (5, 1, 12), 2)


@pytest.mark.parametrize("damage", [
    lambda data: b"NOTANIDX\x00" + data[9:],                    # another header
    lambda data: data[:9] + struct.pack("<Q", 2) + data[17:],   # another version
    lambda data: data[:-8],                                     # cut short
    lambda data: data[:17] + b"\x63" + data[18:],               # unknown token type
])
def test_the_index_reader_refuses_a_file_it_does_not_recognise(tmp_path, damage):
    path = tmp_path / "shard.idx"
    path.write_bytes(damage(index_bytes([5, 1, 12])))
    with pytest.raises(TokenizerError):
        read_shard_index(path)


SETTINGS = TokenizerSettings(revision="0123abcd", vocab_file="hf_home/files/vocab.json",
                             merges_file="hf_home/files/merges.txt")


def fake_environment(root, settings=SETTINGS) -> MegatronEnv:
    """An environment folder that holds the two tokenizer files and nothing that runs."""
    for relative in (settings.vocab_file, settings.merges_file):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("{}", encoding="utf-8")
    return MegatronEnv(root)


def test_the_environment_must_be_named_and_whole(tmp_path):
    with pytest.raises(TokenizerError, match=ENV_VARIABLE):
        MegatronEnv.from_environment({})
    with pytest.raises(TokenizerError, match="is missing"):
        MegatronEnv.from_environment({ENV_VARIABLE: str(tmp_path)})


def test_the_tokenizer_files_are_the_ones_the_spec_names_inside_the_environment(tmp_path):
    env = fake_environment(tmp_path)
    assert env.tokenizer_files(SETTINGS) == (
        tmp_path / "hf_home" / "files" / "vocab.json", tmp_path / "hf_home" / "files" / "merges.txt")
    with pytest.raises(TokenizerError, match="vocab_file is hf_home/files/vocab.json"):
        MegatronEnv(tmp_path / "empty").tokenizer_files(SETTINGS)
    elsewhere = dataclasses.replace(SETTINGS, merges_file="hf_home/other/merges.txt")
    with pytest.raises(TokenizerError, match="merges_file is hf_home/other/merges.txt"):
        env.tokenizer_files(elsewhere)


def test_only_the_settings_the_layout_rests_on_are_accepted():
    megatron_tokenizer.check_settings(SETTINGS)
    for change in (dict(type="NullTokenizer"), dict(name="gpt2-large"), dict(partitions=2),
                   dict(revision="")):
        with pytest.raises(SpecError):
            megatron_tokenizer.check_settings(dataclasses.replace(SETTINGS, **change))


def test_every_subprocess_is_handed_both_tokenizer_files_and_an_offline_hub(tmp_path,
                                                                           monkeypatch):
    # Without the two files Megatron's tokenizer class tries a download into the
    # home folder's cache; HF_HOME and HF_HUB_OFFLINE=1 keep every run inside the
    # environment folder and off the network.
    env = fake_environment(tmp_path / "env")
    runs = []

    def record_run(cmd, *, cwd, env, log_path, timeout, what):
        runs.append((list(cmd), dict(env)))
        if "--output-prefix" in cmd:
            prefix = cmd[cmd.index("--output-prefix") + 1]
            for suffix in (".bin", ".idx"):
                open(f"{prefix}_text_document{suffix}", "wb").close()
        else:
            log_path.write_text(f"{megatron_tokenizer.EOD_TOKEN_ID}\n", encoding="utf-8")

    monkeypatch.setattr(megatron_tokenizer, "_run", record_run)
    monkeypatch.setenv("PYTHONPATH", "somewhere")
    megatron_tokenizer.preflight(env, SETTINGS, tmp_path / "work" / "preflight")
    source = tmp_path / "chunk.jsonl"
    source.write_text('{"id": "a", "text": "one"}\n', encoding="utf-8")
    job = megatron_tokenizer.ShardJob(name="shard-1", input_path=source,
                                      prefix=tmp_path / "shards" / "shard-1",
                                      work_dir=tmp_path / "work" / "shard-1")
    megatron_tokenizer.tokenize_shards(env, SETTINGS, [job])
    (preflight, preflight_env), (run, run_env) = runs
    assert preflight[-2:] == ["gpt2", "0123abcd"]            # the hub name and the revision
    assert run[run.index("--vocab-file") + 1] == str(env.root / SETTINGS.vocab_file)
    assert run[run.index("--merge-file") + 1] == str(env.root / SETTINGS.merges_file)
    assert run[run.index("--partitions") + 1] == "1" and "--append-eod" in run
    assert run[run.index("--tokenizer-type") + 1] == "GPT2BPETokenizer"
    for child in (preflight_env, run_env):
        assert child["HF_HOME"] == str(env.root / "hf_home")
        assert child["HF_HUB_OFFLINE"] == "1" and child["TRANSFORMERS_OFFLINE"] == "1"
        assert "PYTHONPATH" not in child
    # The script reads a copy of the chunk in a new folder, not the chunk itself.
    assert run[run.index("--input") + 1] == str(tmp_path / "work" / "shard-1" / "shard-1.jsonl")


def test_the_mixtures_of_the_spec_are_built_in_the_trainers_order(fixture_plan, tmp_path,
                                                                 no_megatron):
    out = tmp_path / "out"
    env = fake_environment(tmp_path / "env", fixture_plan.tokenizer)
    summary = build(fixture_plan, out, BuildOptions(), megatron_env=env)
    assert list(summary["mixtures"]) == ["E/mix-1", "E/mix-2", "D/mix-1", "Y/mix-1", "K/mix-1",
                                         "E-rr/mix-1", "E-rr/mix-2"]
    # Run K is handed shard-3 first: the summary keeps the spec's order, which a
    # directory listing of the folder would not give.
    assert summary["mixtures"]["K/mix-1"] == {
        "layout": "partitioned", "shards": ["shard-3", "shard-2"],
        "store_names": ["E/shard-3", "E/shard-2"], "folder": "mixtures/K/mix-1"}
    assert sorted(p.name for p in (out / "mixtures" / "K" / "mix-1").iterdir()) == [
        "shard-2.bin", "shard-2.idx", "shard-3.bin", "shard-3.idx"]
    # A round-robin mixture holds the round-robin layout's shards.
    assert summary["mixtures"]["E-rr/mix-2"]["store_names"] == [
        "E-rr/shard-1", "E-rr/shard-2", "E-rr/shard-3"]
    for suffix in (".bin", ".idx"):
        assert file_sha256(out / "mixtures" / "E-rr" / "mix-1" / f"shard-2{suffix}") == file_sha256(
            out / "pipeline" / "round-robin" / "shards" / f"shard-2{suffix}")
    assert summary["segments"]["K/seg-1"] == {"run": "K", "position": 1, "mixture": "K/mix-1"}
    assert list(summary["segments"]) == ["E/seg-1", "E/seg-2", "E-rr/seg-1", "E-rr/seg-2",
                                         "D/seg-1", "Y/seg-1", "K/seg-1"]
    shards = {s["shard"]: s for s in summary["layouts"]["partitioned"]["shards"]}
    assert [shards[s]["documents"] for s in ("shard-1", "shard-2", "shard-3")] == [18, 19, 27]
    assert summary["tokenizer"]["revision"] == fixture_plan.tokenizer.revision
    assert summary["tokenizer"]["vocab_file"] == fixture_plan.tokenizer.vocab_file


def test_a_build_of_one_layout_makes_only_that_layouts_mixtures(fixture_plan, tmp_path,
                                                              no_megatron):
    env = fake_environment(tmp_path / "env", fixture_plan.tokenizer)
    summary = build(fixture_plan, tmp_path / "out", BuildOptions(layouts=("round-robin",)),
                    megatron_env=env)
    assert list(summary["mixtures"]) == ["E-rr/mix-1", "E-rr/mix-2"]
    assert list(summary["segments"]) == ["E-rr/seg-1", "E-rr/seg-2"]


def test_the_build_calls_the_capture_points_of_tokenization_mixtures_and_runs(
        fixture_plan, tmp_path, no_megatron):
    # Tokenization maps and the mixture snapshot are capture points (plan 5.4): the
    # build calls them at their places, with the shard's files, the order the
    # trainer is handed the shards in, and the mixture's folder. Run registration
    # is no part of the build: its hook is called and writes nothing.
    from waterwhite.store.store import Store

    from replica.capture_hooks import StoreCapture

    calls = []

    class Recording(StoreCapture):
        def shard_tokenized(self, files, tokenizer, tokenize, normaliser):
            calls.append(("shard", files.store_name, files.tokenizer_input.name,
                          files.prefix.name, files.listed, tokenize("ab"), normaliser))
            return super().shard_tokenized(files, tokenizer, tokenize, normaliser)

        def mixture_selected(self, folder, relative, name, shards):
            calls.append(("mixture", name, relative, [store_name for store_name, _ in shards]))
            return super().mixture_selected(folder, relative, name, shards)

        def run_segment_ready(self, segment, folder, mixture_name):
            super().run_segment_ready(segment, folder, mixture_name)
            calls.append(("segment", segment.name, mixture_name))

    capture = Recording.open(tmp_path / "store.sqlite", normaliser=fixture_plan.normaliser)
    try:
        build(fixture_plan, tmp_path / "out", BuildOptions(layouts=("partitioned",)),
              capture=capture, megatron_env=no_megatron)
    finally:
        capture.close()
    # A partitioned shard lists the files of its own records and no other.
    assert calls[:3] == [
        ("shard", "E/shard-1", "shard-1.jsonl", "shard-1",
         ("sources/B.jsonl", "sources/Z.jsonl"), [97, 98], "control-chars/1"),
        ("shard", "E/shard-2", "shard-2.jsonl", "shard-2",
         ("sources/B.jsonl", "sources/Z.jsonl"), [97, 98], "control-chars/1"),
        ("shard", "E/shard-3", "shard-3.jsonl", "shard-3",
         ("gen/T.jsonl", "sources/Z.jsonl"), [97, 98], "control-chars/1")]
    assert ("mixture", "K/mix-1", "mixtures/K/mix-1", ["E/shard-3", "E/shard-2"]) in calls
    assert [c for c in calls if c[0] == "segment"] == [
        ("segment", "E/seg-1", "E/mix-1"), ("segment", "E/seg-2", "E/mix-2"),
        ("segment", "D/seg-1", "D/mix-1"), ("segment", "Y/seg-1", "Y/mix-1"),
        ("segment", "K/seg-1", "K/mix-1")]
    with Store.open(tmp_path / "store.sqlite", readonly=True) as store:
        assert {a.kind for a in store.artifacts()} == {"jsonl", "shard", "mixture"}
        # Runs, segments and checkpoints are not registered yet.
        assert store.runs() == () and store.artifact("E@6") is None


def _run(tmp_path, code, timeout):
    megatron_tokenizer._stop.clear()
    megatron_tokenizer._run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ),
                            log_path=tmp_path / "run.log", timeout=timeout, what="the test run")


def test_a_run_that_prints_an_error_and_does_not_exit_is_stopped(tmp_path):
    # Megatron's script hangs when a worker dies: the error is printed and the
    # parent waits for ever. The replica stops such a run and says why.
    code = ("import os, time\n"
            "print(os.getpid(), flush=True)\n"
            "print('Traceback (most recent call last):', flush=True)\n"
            "time.sleep(120)\n")
    started = time.monotonic()
    with pytest.raises(TokenizerError, match="printed an error and did not exit"):
        _run(tmp_path, code, timeout=60)
    assert time.monotonic() - started < 20
    pid = int((tmp_path / "run.log").read_text().split()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the subprocess is gone


def test_a_run_past_its_time_limit_is_stopped(tmp_path):
    started = time.monotonic()
    with pytest.raises(TokenizerError, match="did not finish within 2 seconds"):
        _run(tmp_path, "import time; time.sleep(120)", timeout=2)
    assert time.monotonic() - started < 20


def test_a_run_that_fails_reports_its_exit_code_and_log(tmp_path):
    with pytest.raises(TokenizerError, match="exited with code 3"):
        _run(tmp_path, "import sys; print('no such tokenizer'); sys.exit(3)", timeout=30)
    _run(tmp_path, "print('fine')", timeout=30)
    assert (tmp_path / "run.log").read_text().strip() == "fine"


@pytest.mark.megatron
@pytest.mark.slow
def test_the_fixture_is_tokenized_one_document_per_surviving_line(fixture_plan, tmp_path,
                                                                  read_rows):
    """Runs Megatron's preprocessing (about 15 seconds). On a shared machine run it
    through the CPU lock."""
    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    out = tmp_path / "out"
    summary = build(fixture_plan, out, BuildOptions(layouts=("partitioned",), parallel=3))
    shards = {s["shard"]: s for s in summary["layouts"]["partitioned"]["shards"]}
    assert list(shards) == ["shard-1", "shard-2", "shard-3"]
    # Shard-2 reads 20 lines, one of them the spec's skipped line, and holds 19
    # documents: the skip is a drop and shifts no ordinal (the spec's condition 2).
    assert [(s["documents"], s["skipped"]) for s in shards.values()] == [(18, 0), (19, 1), (27, 0)]
    for name, entry in shards.items():
        prefix = out / "pipeline" / "partitioned" / "shards" / name
        lines = read_rows(out / "pipeline" / "partitioned" / "tokenizer-input" / f"{name}.jsonl")
        index = read_shard_index(f"{prefix}.idx")
        assert index.documents == len(lines) == entry["documents"] == entry["input_lines"]
        assert index.tokens == entry["tokens"]
        assert os.path.getsize(f"{prefix}.bin") == index.tokens * index.token_bytes
        # --append-eod: every document ends with the end-of-document token.
        tokens = numpy.fromfile(f"{prefix}.bin", dtype="<u2")
        ends = numpy.cumsum(index.document_tokens) - 1
        assert (tokens[ends] == megatron_tokenizer.EOD_TOKEN_ID).all()
    assert summary["tokenizer"]["append_eod"] is True and summary["tokenizer"]["partitions"] == 1
    # Directory-selected mixtures hold the shard files they select, the same bytes.
    assert sorted(summary["mixtures"]) == ["D/mix-1", "E/mix-1", "E/mix-2", "K/mix-1", "Y/mix-1"]
    assert summary["tokenizer"]["revision"] == fixture_plan.tokenizer.revision
    assert sorted(p.name for p in (out / "mixtures" / "E" / "mix-1").iterdir()) == [
        "shard-1.bin", "shard-1.idx", "shard-2.bin", "shard-2.idx"]
    assert len(list((out / "mixtures" / "E" / "mix-2").iterdir())) == 6
    for name in ("shard-1", "shard-2"):
        for suffix in (".bin", ".idx"):
            assert file_sha256(out / "mixtures" / "E" / "mix-1" / f"{name}{suffix}") == file_sha256(
                out / "pipeline" / "partitioned" / "shards" / f"{name}{suffix}")
