"""Shared pieces of the replica's tests."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import struct
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "sandbox: runs a job in the bubblewrap sandbox (Linux, unprivileged user namespaces)",
    )


@pytest.fixture(scope="session")
def sandbox():
    """A test that starts the confined harness asks for this fixture. Where
    bubblewrap is missing or cannot create its namespaces it is skipped, by this
    name and with the harness's own reason, and is not worked around."""
    from waterwhite.confine import SandboxUnavailable, find_sandbox

    try:
        return find_sandbox()
    except SandboxUnavailable as unavailable:
        pytest.skip(str(unavailable))

# One small generated set (docs/plant.md): 800 lines in all. Worked out from the
# counts of docs/plant.md section 2:
#   X pairs 800 // 20 = 40, summary passages 800 // 100 = 8
#   A = 40 + 8 = 48 lines, B = 2 * 40 = 80, C = 2 * 8 + 24 = 40
#   generated = 40 P + 2 UP + 40 T + 8 S = 90, Z = 800 - 48 - 80 - 40 - 90 = 542
#   Z lines 1 to 271 are inputs of R1 (24 pair twins and 9 sanitizer cases among them)
SMALL_SET = dict(records=800, seed=7, length=160, pairs=24, sanitizer_cases=3, chunks=20)


def _read_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n") if line != ""]


def _load_script(name: str, file: str):
    """``fixture/plant`` is a folder of scripts, not a package: load one by file."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "fixture" / "plant" / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def repo() -> Path:
    return REPO


@pytest.fixture(scope="session")
def spec_path() -> Path:
    return REPO / "fixture" / "spec" / "fixture_spec.toml"


@pytest.fixture(scope="session")
def spec_doc(spec_path) -> dict:
    """The fixture spec as TOML, for the tables the tests read themselves."""
    with open(spec_path, "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="session")
def read_rows():
    """Read a JSONL file into a list of rows."""
    return _read_rows


@pytest.fixture(scope="session")
def fixture_plan(spec_path):
    from replica.plan import load_fixture_spec

    return load_fixture_spec(spec_path)


@pytest.fixture(scope="session")
def fixture_build(fixture_plan, tmp_path_factory):
    """One seeded build of the hand-checked fixture, both layouts, no tokenization."""
    from replica.build import BuildOptions, build

    out = tmp_path_factory.mktemp("fixture-build") / "out"
    summary = build(fixture_plan, out, BuildOptions(tokenize=False))
    return out, summary


@pytest.fixture(scope="session")
def plant_generate():
    """The plant generator, ``fixture/plant/generate.py``."""
    return _load_script("plant_generate", "generate.py")


@pytest.fixture(scope="session")
def plant_count():
    """The counting script, ``fixture/plant/count.py``."""
    return _load_script("plant_count", "count.py")


@pytest.fixture(scope="session")
def set_parameters() -> dict:
    """The generator's parameters of the small set."""
    return dict(SMALL_SET)


@pytest.fixture(scope="session")
def small_set(plant_generate, tmp_path_factory) -> Path:
    """A generated set of 800 lines with dedup on, as the generator writes it."""
    folder = tmp_path_factory.mktemp("small-set") / "set"
    plant_generate.generate(folder, **SMALL_SET)
    return folder


@pytest.fixture(scope="session")
def small_build(small_set, tmp_path_factory):
    """One build of the small set, both layouts, no tokenization."""
    from replica.build import BuildOptions, build
    from replica.plan import load_set

    out = tmp_path_factory.mktemp("small-build") / "out"
    summary = build(load_set(small_set), out, BuildOptions(tokenize=False))
    return out, summary


# --- A stand-in for the calls that run Megatron -------------------------------------------
#
# The toy tokenizer gives one token per byte of a text. Its shards are written in
# the layout Megatron's preprocessing writes (version 1 of the index format, token
# type code 8, two bytes a token): one sequence per line that gives tokens, each
# ended by the end-of-document token, and for a line that gives none a document
# boundary with no sequence. So the tool reads a toy shard as it reads a real one.


def toy_tokenize(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def write_toy_shard(prefix: Path, texts: list[str], eod: int) -> None:
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for text in texts:
        document = toy_tokenize(text)
        if document:
            document.append(eod)
            pointers.append(2 * len(tokens))
            lengths.append(len(document))
            tokens.extend(document)
        boundaries.append(len(lengths))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}H", *tokens))
    Path(f"{prefix}.idx").write_bytes(
        b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, 8, len(lengths), len(boundaries))
        + struct.pack(f"<{len(lengths)}i", *lengths)
        + struct.pack(f"<{len(pointers)}q", *pointers)
        + struct.pack(f"<{len(boundaries)}q", *boundaries))


class ToyLineTokenizer:
    """The toy tokenizer as the build's function of one text."""

    def __call__(self, text: str) -> list[int]:
        return toy_tokenize(text)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        pass


@contextlib.contextmanager
def _megatron_stand_in(root: Path):
    """Stand-ins for the calls that run Megatron, for as long as the block lasts:
    the tokenizer is not built, each chunk becomes a toy shard, and the tokenizer
    as a function is the toy one. Gives an environment folder that holds the two
    tokenizer files the fixture spec names and nothing that runs."""
    from replica import megatron_tokenizer
    from replica.plan import load_tokenizer_settings

    def tokenize_shards(env, settings, jobs, *, parallel=1, timeout=900.0):
        for job in jobs:
            texts = [row["text"] for row in _read_rows(job.input_path)]
            write_toy_shard(job.prefix, texts, megatron_tokenizer.EOD_TOKEN_ID)

    tokenizer = load_tokenizer_settings(REPO / "fixture" / "spec" / "fixture_spec.toml")
    for relative in (tokenizer.vocab_file, tokenizer.merges_file):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text("{}", encoding="utf-8")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(megatron_tokenizer, "preflight", lambda env, settings, work_dir: None)
        patch.setattr(megatron_tokenizer, "tokenize_shards", tokenize_shards)
        patch.setattr(megatron_tokenizer, "line_tokenizer",
                      lambda env, settings, work_dir, timeout=120.0: ToyLineTokenizer())
        yield megatron_tokenizer.MegatronEnv(root)


@pytest.fixture(scope="session")
def megatron_stand_in():
    """The stand-in as a context manager, for a fixture of a wider scope:
    ``with megatron_stand_in(folder) as env``."""
    return _megatron_stand_in


@pytest.fixture()
def no_megatron(tmp_path):
    """The stand-in for one test. Its value is the stand-in environment."""
    with _megatron_stand_in(tmp_path / "megatron-stand-in") as env:
        yield env


@pytest.fixture(scope="session")
def toy_tokens():
    """The toy tokenizer of the stand-in: one token per byte of a text."""
    return toy_tokenize
