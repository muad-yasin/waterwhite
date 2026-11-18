"""Fixtures for the oracle's tests that need the Megatron environment.

They make a small throwaway corpus (three JSONL files of short English lines) and
tokenize it with Megatron's own preprocessing script, set up as the fixture spec
says: GPT-2 BPE, ``--append-eod``, one partition, a fresh input folder per file,
the tokenizer's files named so nothing is fetched, and the hub switched off.

A test marked ``megatron`` fails, and is not skipped, when the environment is
missing: a skipped oracle would let the checks that rest on it pass unseen. Leave
them out with ``-m "not megatron"``.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS_SECONDS = 50

_SUBJECTS = ["The harbour pilot", "A quiet librarian", "My neighbour's dog", "The night baker",
             "An old surveyor", "The ferry captain", "A young glazier", "The town clerk",
             "Our postman", "The beekeeper"]
_VERBS = ["counted", "painted", "repaired", "measured", "carried", "watched", "sorted", "folded",
          "polished", "weighed"]
_OBJECTS = ["the green lanterns", "seven wooden crates", "a broken compass", "the winter ledgers",
            "two copper kettles", "the garden fence", "a stack of maps", "the last ferry tickets",
            "an empty birdcage", "the church clock"]
_TAILS = ["before the rain came", "while the kettle boiled", "under a pale sky",
          "without saying a word", "long after midnight", "as the tide went out",
          "beside the old mill", "with great care", "for the third time that week",
          "until the bell rang"]

# name, number of lines, seed of the generator
CORPUS_FILES = (("alpha", 44, 11), ("beta", 36, 22), ("gamma", 52, 33))


def corpus_rows(name: str, count: int, seed: int) -> list[dict]:
    """The rows of one throwaway file. Seeded, so every run makes the same text."""
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        sentences = []
        for _ in range(rng.choice([1, 1, 1, 2, 2, 3, 5])):
            sentence = f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)}"
            if rng.random() < 0.6:
                sentence += " " + rng.choice(_TAILS)
            sentences.append(sentence + ".")
        rows.append({"id": f"{name}-{i:03d}", "text": " ".join(sentences)})
    return rows


@dataclass(frozen=True)
class Corpus:
    inputs: dict[str, Path]  # name -> tokenizer-input JSONL
    prefixes: dict[str, Path]  # name -> shard prefix
    tokenizer: dict  # the "tokenizer" object of an oracle config
    python: Path  # the environment's interpreter
    env: Path


@pytest.fixture(scope="session")
def megatron_env():
    """The environment folder, with the Hugging Face cache pointed into it and the
    hub switched off for the whole session."""
    value = os.environ.get("WATERWHITE_MEGATRON_ENV")
    if not value or not Path(value).is_dir():
        pytest.fail(
            "WATERWHITE_MEGATRON_ENV does not name the Megatron environment folder "
            "(tools/setup_megatron_env.sh builds it); leave these tests out with -m 'not megatron'"
        )
    env = Path(value)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_HOME", str(env / "hf_home"))
        patch.setenv("HF_HUB_OFFLINE", "1")
        yield env


@pytest.fixture(scope="session")
def corpus(megatron_env, tmp_path_factory) -> Corpus:
    snapshots = sorted((megatron_env / "hf_home" / "hub" / "models--gpt2" / "snapshots").glob("*/vocab.json"))
    if len(snapshots) != 1:
        pytest.fail(f"expected one gpt2 snapshot with vocab.json in the environment, found {len(snapshots)}")
    tokenizer = {
        "type": "GPT2BPETokenizer",
        "vocab_file": str(snapshots[0]),
        "merges_file": str(snapshots[0].with_name("merges.txt")),
    }
    python = megatron_env / "venv" / "bin" / "python"
    script = megatron_env / "src" / "Megatron-LM" / "tools" / "preprocess_data.py"

    # Build the tokenizer here first: when it cannot be built, the preprocessing
    # script's worker fails and the script hangs instead of exiting.
    from oracle.config import parse_tokenizer
    from oracle.megatron_data import build_megatron_tokenizer

    assert int(build_megatron_tokenizer(parse_tokenizer(tokenizer, REPO_ROOT)).eod) == 50256

    root = tmp_path_factory.mktemp("oracle-corpus")
    (root / "shards").mkdir()
    inputs, prefixes, running = {}, {}, []
    for name, count, seed in CORPUS_FILES:
        folder = root / "input" / name  # a fresh input folder: partition files land beside the input
        folder.mkdir(parents=True)
        inputs[name] = folder / f"{name}.jsonl"
        inputs[name].write_text(
            "".join(json.dumps(row) + "\n" for row in corpus_rows(name, count, seed)), encoding="utf-8"
        )
        prefixes[name] = root / "shards" / f"{name}_text_document"
        command = [
            str(python), "-I", str(script),
            "--input", str(inputs[name]),
            "--output-prefix", str(root / "shards" / name),
            "--tokenizer-type", "GPT2BPETokenizer",
            "--vocab-file", tokenizer["vocab_file"],
            "--merge-file", tokenizer["merges_file"],
            "--append-eod", "--workers", "1", "--partitions", "1",
        ]
        log = open(root / f"{name}.preprocess.log", "wb")
        running.append(
            (name, log, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        )
    for name, log, process in running:
        try:
            code = process.wait(timeout=PREPROCESS_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)  # this test's own process group, by its id
            pytest.fail(f"preprocessing of {name} did not end in {PREPROCESS_SECONDS} s; see its log in {root}")
        finally:
            log.close()
        if code != 0 or not Path(str(prefixes[name]) + ".bin").is_file():
            pytest.fail(f"preprocessing of {name} failed with exit {code}; see its log in {root}")
    return Corpus(inputs=inputs, prefixes=prefixes, tokenizer=tokenizer, python=python, env=megatron_env)


# Which throwaway file stands for which shard of a built fixture, per layout.
STAND_IN_SHARDS = {
    "partitioned": {"shard-1": "alpha", "shard-2": "beta", "shard-3": "gamma"},
    "round-robin": {"shard-1": "beta", "shard-2": "gamma", "shard-3": "alpha"},
}
STAND_IN_T_ORDINAL = 10  # the document of each layout's shard-3 that is given the id "T"


@pytest.fixture(scope="session")
def built_fixture(corpus, tmp_path_factory) -> Path:
    """A folder laid out as a fixture build of the replica lays it out
    (docs/replica.md), made of the throwaway shards: ``pipeline/<layout>/`` with
    the shard files, the tokenizer inputs, their manifests and an empty sanitizer
    log. The fixture spec's runs can be run over it. Its texts are not the
    fixture's, so most read positions the spec asks for are not there; one
    document of each layout's shard-3 carries the id ``T``."""
    root = tmp_path_factory.mktemp("oracle-built")
    for layout, shards in STAND_IN_SHARDS.items():
        folder = root / "pipeline" / layout
        (folder / "shards").mkdir(parents=True)
        (folder / "tokenizer-input").mkdir()
        (folder / "sanitizer.log.jsonl").write_text("", encoding="utf-8")
        for shard, name in shards.items():
            for suffix in (".bin", ".idx"):
                shutil.copy(str(corpus.prefixes[name]) + suffix, folder / "shards" / (shard + suffix))
            rows = [json.loads(line) for line in corpus.inputs[name].read_text(encoding="utf-8").splitlines()]
            if shard == "shard-3":
                rows[STAND_IN_T_ORDINAL]["id"] = "T"
            target = folder / "tokenizer-input" / f"{shard}.jsonl"
            target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            target.with_name(target.name + ".manifest.jsonl").write_text(
                "".join(
                    json.dumps({"line": n, "handle": row["id"], "origin": [row["id"]]}) + "\n"
                    for n, row in enumerate(rows, start=1)
                ),
                encoding="utf-8",
            )
    return root


@pytest.fixture
def segment_config(corpus, tmp_path):
    """A function that makes a segment config over the throwaway shards. Every call
    gets its own new cache folder and output paths under the test's folder."""
    counter = iter(range(1000))

    def make(
        *,
        shards=("alpha", "beta"),
        segment="T/seg-1",
        launches=None,
        step_record=None,
        sampler_file=None,
        start_kind="fresh",
        parent_checkpoint=None,
        allow_sequential_restart=None,
        **settings,
    ) -> dict:
        n = next(counter)
        merged = {
            "dataloader_type": "single",
            "micro_batch_size": 4,
            "global_batch_size": 4,
            "sequence_length": 32,
            "random_seed": 1234,
            "split": "1,0,0",
            "train_iters": 15,
        }
        merged.update(settings)
        raw = {
            "run": segment.split("/")[0],
            "segment": segment,
            "start_kind": start_kind,
            "shards": [str(corpus.prefixes[name]) for name in shards],
            "tokenizer": corpus.tokenizer,
            "settings": merged,
            "launches": launches
            or [
                {"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 0, "step_at_start": 0,
                 "steps": merged["train_iters"]}
            ],
            "cache_dir": f"cache-{n}",
            "step_record": step_record or {},
            "outputs": {
                key: f"out-{n}/{key}.json" if key != "log" else f"out-{n}/log.jsonl"
                for key in ("digests", "run_record", "step_record", "log", "summary")
            },
        }
        if parent_checkpoint is not None:
            raw["parent_checkpoint"] = parent_checkpoint
        if sampler_file is not None:
            raw["sampler_file"] = sampler_file
        if allow_sequential_restart is not None:
            raw["allow_sequential_restart"] = allow_sequential_restart
        return raw

    make.base = tmp_path
    return make
