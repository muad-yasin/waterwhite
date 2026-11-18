"""The builds the baselines are run on (plan 5.5 Check 10; table README, section 4):
plans only.

Each function returns plain data, either a ``Call`` (a function of the replica, by
its dotted name, with its arguments) or an argv for the replica's command line or
for the plant's generator script. Nothing is started and nothing is imported from
the replica here: the runner that executes a plan resolves the name itself, so that
this module stays data.

The stores are the seven names of the table's README, section 4, the two scratch
stores included (rows of Checks 3 and 4 ask them). The stores that hold shards are
five: main, rr, no-capture, scratch-500 (its row names a tokenized shard) and
scratch-notokens (its row names a shard with its capture map); the replica refuses a
scratch store built without tokenization. These five are built tokenized, and swap
and import, whose rows name no shard, are not.
The runs of a store are those the README names for it, in the order of the fixture
spec's run tables. A generated set is built at the sizes at which the table's rows
are asked, with the generator's parameters of each size.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUILD_STORE = "replica.fixture.build_table_store"
BUILD_SET_STORE = "replica.fixture.build_set_store"

LAYOUT = "round-robin"
GENERATOR = "fixture/plant/generate.py"

TABLE_STORES = ("main", "rr", "swap", "no-capture", "import", "scratch-500", "scratch-notokens")
# The stores that hold shards: they are built tokenized.
TOKENIZED_STORES = ("main", "rr", "no-capture", "scratch-500", "scratch-notokens")

# The extensions whose rows are served by the main build plus a release of that extension, a build of its own: the key of the
# replica's build (``replica.fixture.EXTENSION_STORES``). Data only; a test holds it to the replica's own table.
MAIN_WITH_EXTENSION = {"generating-model": "main+generating-model"}

# The runs registered in a store, in the order of the spec's run tables.
RUNS_OF_STORE = {"main": ("E", "D", "Y", "K"), "rr": ("E-rr",)}

SET_SIZES = ("small", "10k", "100k")
# The generator's parameters of each size, and whether dedup is on.
SET_PARAMETERS = {
    "small": {"records": 800, "seed": 7, "length": 160, "pairs": 24, "sanitizer_cases": 3},
    "10k": {"records": 10_000, "seed": 1},
    "100k": {"records": 100_000, "seed": 1},
}
SET_DEDUP = {"small": False, "10k": True, "100k": False}
_FLAG = {"records": "--records", "seed": "--seed", "length": "--length", "pairs": "--pairs",
         "sanitizer_cases": "--sanitizer-cases"}


@dataclass(frozen=True)
class Call:
    """A call of a function of the replica: the dotted name, its positional
    arguments and its keyword arguments."""

    function: str
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


def store_call(store: str, folder, *, spec=None, parallel: int = 1, extensions=()) -> Call:
    """The call that builds one store of the table into ``folder``. With extensions on, the store ``main`` is the build of the
    extension (``MAIN_WITH_EXTENSION``): the main build and then the release of the extension, in the same store; no other
    store has such a build."""
    if store not in TABLE_STORES:
        raise ValueError(f"{store!r} is no store the baselines are asked on: {TABLE_STORES}")
    named = store
    if extensions:
        unknown = sorted(set(extensions) - set(MAIN_WITH_EXTENSION))
        if unknown or store != "main":
            raise ValueError(f"no build of the store {store!r} with the extensions {sorted(extensions)}: only the store main has "
                             f"one, for {sorted(MAIN_WITH_EXTENSION)}")
        if len(set(extensions)) != 1:
            raise ValueError(f"one extension at a time has a build of its own: {sorted(set(extensions))}")
        named = MAIN_WITH_EXTENSION[next(iter(extensions))]
    kwargs: dict[str, Any] = {"store": named, "tokenize": store in TOKENIZED_STORES,
                              "parallel": parallel}
    if spec is not None:
        kwargs["spec"] = spec
    return Call(BUILD_STORE, (folder,), kwargs)


def _runs(store: str) -> str:
    if store not in RUNS_OF_STORE:
        raise ValueError(f"the store {store!r} holds no runs; the stores that do: {sorted(RUNS_OF_STORE)}")
    return ",".join(RUNS_OF_STORE[store])


def runs_argv(store: str, out, *, fixture_build, python, spec=None, timeout=None) -> list[str]:
    """The replica's ``build-runs`` for the runs of a store."""
    argv = [str(python), "-m", "replica", "build-runs", "--out", str(out),
            "--fixture-build", str(fixture_build), "--runs", _runs(store)]
    if spec is not None:
        argv += ["--spec", str(spec)]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    return argv


def register_runs_argv(store: str, store_file, runs_folder, *, python, spec=None) -> list[str]:
    """The replica's ``register-runs`` for the runs of a store."""
    argv = [str(python), "-m", "replica", "register-runs", "--store", str(store_file),
            "--runs-folder", str(runs_folder), "--runs", _runs(store)]
    if spec is not None:
        argv += ["--spec", str(spec)]
    return argv


def generate_argv(size, out, *, python) -> list[str]:
    """The plant's generator for the set of a size."""
    if size not in SET_SIZES:
        raise ValueError(f"{size!r} is no set size the table asks: {SET_SIZES}")
    argv = [str(python), GENERATOR, "--out", str(out)]
    for key, value in SET_PARAMETERS[size].items():
        argv += [_FLAG[key], str(value)]
    argv += ["--dedup", "on" if SET_DEDUP[size] else "off"]
    return argv


def _inside(path, folder) -> bool:
    path, folder = os.path.normpath(Path(path)), os.path.normpath(Path(folder))
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def set_build_argv(set_folder, out, *, store, python, tokenizer_spec, jobs: int = 1, layout: str = LAYOUT) -> list[str]:
    """The replica's ``build-set`` of a generated set with capture on, in the
    round-robin layout (``layout`` names another: the partitioned build of the 10k set,
    Check 9's route). The store file may not lie inside the build folder."""
    if _inside(store, out):
        raise ValueError("the store file must lie outside the build folder")
    return [str(python), "-m", "replica", "build-set", "--set", str(set_folder), "--out", str(out),
            "--layout", layout, "--tokenizer-spec", str(tokenizer_spec), "--capture",
            "--store", str(store), "--jobs", str(jobs)]


def set_store_call(set_folder, out, *, tokenizer_spec, jobs: int = 1) -> Call:
    """The call that builds a generated set with no capture and recovers it."""
    return Call(BUILD_SET_STORE, (out, set_folder),
                {"store": "no-capture", "layout": LAYOUT, "tokenizer_spec": tokenizer_spec,
                 "parallel": jobs})
