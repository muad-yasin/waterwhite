"""The builds the baselines are run on (plan 5.5 Check 10; table README, sections 4
and "Check 10"): plans only. Each function of ``baselines.builds`` returns plain data
(a call to a function of the replica, or an argv) and starts nothing: an autouse
fixture makes every way of starting a process, and every entry point of the replica
that builds, raise, so a plan that runs anything fails its test.

Where the expected values come from:

- the stores are the names of the table in README section 4; the stores that hold
  shards (and so are built tokenized) are the ones whose README row mentions shards;
- the runs of a store are the README's ("runs E, D, Y and K" for ``main``, run E
  rebuilt for ``rr``), in the order of the ``[runs.*]`` tables of the fixture spec;
- the flags of the replica's command line are checked against ``replica/cli.py``;
- the generator's parameters per size are those of ``tests/acceptance/
  acceptance_sets.py`` (``SIZES`` and ``DEDUP``), which the table's rows read, and
  the sizes are the sizes at which the table's rows are asked (``facts``).

The names this file asks of ``baselines.builds``:

- ``Call``: a frozen dataclass with ``function`` (the dotted name of a function of the
  replica, for example ``replica.fixture.build_table_store``), ``args`` (a tuple) and
  ``kwargs`` (a dict);
- ``TABLE_STORES``: the seven stores the baselines are asked on, as names: every store
  of README section 4, the two scratch stores included (rows of Checks 3 and 4 ask them);
- ``store_call(store, folder, *, spec=None, parallel=1)``: the ``Call`` of
  ``replica.fixture.build_table_store`` for a store, with ``args`` holding the folder,
  ``kwargs["store"]`` the name, and ``kwargs.get("tokenize", False)`` true for the
  stores that hold shards only; a name that is no store of the list raises
  ``ValueError``; ``spec`` and ``parallel`` are passed on as ``kwargs["spec"]`` and
  ``kwargs["parallel"]`` when they are given;
- ``RUNS_OF_STORE``: a dict, ``main`` to ``("E", "D", "Y", "K")`` and ``rr`` to
  ``("E-rr",)``;
- ``runs_argv(store, out, *, fixture_build, python, spec=None, timeout=None)``: the
  argv of ``python -m replica build-runs`` with ``--out``, ``--fixture-build`` and
  ``--runs`` (the runs of the store joined by commas, in the spec's order), and
  ``--spec`` and ``--timeout`` when given; a store with no runs raises
  ``ValueError``;
- ``register_runs_argv(store, store_file, runs_folder, *, python, spec=None)``: the
  argv of ``python -m replica register-runs`` with ``--store``, ``--runs-folder``,
  ``--runs`` and ``--spec`` when given;
- ``SET_SIZES``: ``("small", "10k", "100k")``, the sizes of a generated set at which
  the table's rows are asked;
- ``generate_argv(size, out, *, python)``: the argv of the generator script of the
  plant, ``fixture/plant/generate.py`` (as that relative path), with ``--out`` and
  the size's parameters: ``--records``, ``--seed`` and, only where the size sets
  them, ``--length``, ``--pairs`` and ``--sanitizer-cases``, and always ``--dedup``
  as ``on`` or ``off``; a size that is not in ``SET_SIZES`` raises ``ValueError``;
- ``set_build_argv(set_folder, out, *, store, python, tokenizer_spec, jobs=1)``: the
  argv of ``python -m replica build-set`` with capture on, in the round-robin layout,
  with ``--set``, ``--out``, ``--layout``, ``--tokenizer-spec``, ``--capture``,
  ``--store`` and ``--jobs``; a ``store`` file inside ``out`` raises ``ValueError``
  (the replica refuses it);
- ``set_store_call(set_folder, out, *, tokenizer_spec, jobs=1)``: the ``Call`` of
  ``replica.fixture.build_set_store`` (a generated set built with no capture and
  recovered), ``args`` holding ``out`` then ``set_folder``, ``kwargs["layout"]`` the
  round-robin layout, ``kwargs["tokenizer_spec"]`` and ``kwargs["parallel"]``.

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import hashlib
from types import SimpleNamespace
import dataclasses
import importlib
import os
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

# Helpers that read a module's source and a folder's bytes. Nothing here imports the
# ``baselines`` package. The word lists are written here, never taken from the
# code under test.


def _make_sbx_kit() -> SimpleNamespace:
    def tree_digest(folder: Path) -> dict[str, str]:
        """Every entry of a folder: relative path -> a digest of its bytes, or the link
        text of a link, or "dir". Equal digests mean nothing was added, removed or
        changed."""
        folder = Path(folder)
        out: dict[str, str] = {}
        for current, dirs, files in os.walk(folder, followlinks=False):
            for name in dirs + files:
                path = Path(current) / name
                rel = str(path.relative_to(folder))
                if path.is_symlink():
                    out[rel] = "link:" + os.readlink(path)
                elif path.is_dir():
                    out[rel] = "dir"
                else:
                    out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return out


    # --- Reading a module's source ---------------------------------------------------------


    def source_text(module) -> str:
        return Path(module.__file__).read_text(encoding="utf-8")


    def string_constants(module) -> list[str]:
        """Every string constant of a module's source, except the pieces of an f-string
        (``f"{folder}/RECORD"`` has the piece ``"/RECORD"``, which is no location)."""
        tree = ast.parse(source_text(module))
        pieces = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)
                  for part in ast.walk(node) if isinstance(part, ast.Constant)}
        return [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in pieces]


    SYSTEM_FOLDERS = ("usr", "etc", "dev", "proc", "tmp", "lib", "lib64", "bin", "sbin")
    # Words of a location of this machine, written in two pieces so that this file holds
    # none of them whole.
    LOCAL_WORDS = ("/ho" + "me/", "/ro" + "ot/", "Proj" + "ects/", ".wo" + "rk/")


    def local_path_constants(module) -> list[str]:
        """The string constants of a module that look like a location of this machine:
        an absolute path (one word with no space, starting with a slash, more than a bare
        separator) whose first folder is no system folder, or any text naming a home
        folder or a working folder of this repository."""
        bad = []
        for text in string_constants(module):
            is_path = text.startswith("/") and text.strip("/") != "" and not any(c.isspace() for c in text)
            if is_path and text.lstrip("/").split("/", 1)[0] not in SYSTEM_FOLDERS:
                bad.append(text)
            elif any(word in text for word in LOCAL_WORDS):
                bad.append(text)
        return bad

    def flags(argv: list[str]) -> dict[str, str]:
        """The ``--flag value`` pairs of an argv (a flag with no value, followed by
        another flag or the end, maps to the empty string)."""
        out: dict[str, str] = {}
        items = list(argv)
        for i, item in enumerate(items):
            if item.startswith("--"):
                nxt = items[i + 1] if i + 1 < len(items) else ""
                out[item] = "" if nxt.startswith("--") else nxt
        return out

    return SimpleNamespace(tree_digest=tree_digest, source_text=source_text, string_constants=string_constants, SYSTEM_FOLDERS=SYSTEM_FOLDERS, LOCAL_WORDS=LOCAL_WORDS, local_path_constants=local_path_constants, flags=flags)


sbx_kit = _make_sbx_kit()

REPO = Path(__file__).resolve().parents[2]
README = REPO / "fixture" / "table" / "README.md"
SPEC = REPO / "fixture" / "spec" / "fixture_spec.toml"
CLI = REPO / "replica" / "cli.py"

# README section 4 names these runs for these stores.
README_RUNS = {"main": ("E", "D", "Y", "K"), "rr": ("E-rr",)}
LAYOUT = "round-robin"


@pytest.fixture(autouse=True)
def nothing_starts(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a plan of a build must not start anything")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(os, "system", boom)
    for target in ("replica.fixture.build_table_store", "replica.fixture.build_set_store",
                   "replica.runs.build_runs", "replica.register_runs.register_runs"):
        monkeypatch.setattr(target, boom)


def _stores_section() -> dict[str, str]:
    """README section 4: store name -> the text of its row."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## 4. Stores", 1)[1].split("\n## 5.", 1)[0]
    rows = {}
    for line in section.splitlines():
        found = re.match(r"\| `([a-z0-9-]+)` \| (.*) \|$", line)
        if found and found.group(1) != "store":
            rows[found.group(1)] = found.group(2)
    return rows


def _builds(need_module):
    return need_module("baselines.builds")


def _spec_runs() -> tuple[str, ...]:
    with open(SPEC, "rb") as f:
        return tuple(tomllib.load(f)["runs"].keys())


def _cli_text() -> str:
    return CLI.read_text(encoding="utf-8")


def test_names_of_the_builds_module(need_attribute):
    for name in ("store_call", "runs_argv", "register_runs_argv", "generate_argv",
                 "set_build_argv", "set_store_call"):
        assert callable(need_attribute("baselines.builds", name))
    for name in ("TABLE_STORES", "RUNS_OF_STORE", "SET_SIZES"):
        assert need_attribute("baselines.builds", name) is not None
    assert dataclasses.is_dataclass(need_attribute("baselines.builds", "Call"))


def test_the_stores_are_all_seven_of_the_table(need_module):
    builds = _builds(need_module)
    readme = _stores_section()
    assert set(readme) == {"main", "rr", "swap", "no-capture", "import", "scratch-500",
                           "scratch-notokens"}
    assert set(builds.TABLE_STORES) == set(readme)
    assert len(builds.TABLE_STORES) == 7


def test_the_two_scratch_stores_are_planned_tokenized(need_module, tmp_path):
    # Each scratch store of the README holds one shard, so it is built tokenized (the
    # replica refuses a scratch store built without tokenization).
    builds = _builds(need_module)
    readme = _stores_section()
    scratch_with_shard = {name for name, row in readme.items()
                          if name.startswith("scratch") and "shard" in row}
    assert scratch_with_shard == {"scratch-500", "scratch-notokens"}
    for store in sorted(scratch_with_shard):
        assert store in builds.TABLE_STORES, store
        call = builds.store_call(store, tmp_path / store)
        assert call.function == "replica.fixture.build_table_store"
        assert call.kwargs["store"] == store
        assert call.kwargs["tokenize"] is True, store


def test_a_store_is_one_call_of_the_replicas_builder_with_the_stores_own_name(need_module, tmp_path):
    builds = _builds(need_module)
    module, _, attribute = "replica.fixture.build_table_store".rpartition(".")
    assert callable(getattr(importlib.import_module(module), attribute))
    for store in builds.TABLE_STORES:
        call = builds.store_call(store, tmp_path / store)
        assert call.function == "replica.fixture.build_table_store"
        assert [Path(a) for a in call.args] == [tmp_path / store]
        assert call.kwargs["store"] == store


def test_the_stores_that_hold_shards_are_built_tokenized_and_no_other(need_module, tmp_path):
    builds = _builds(need_module)
    readme = _stores_section()
    # The five stores whose README row names a shard, the two scratch stores included
    # (README 105 and 106); swap and import hold none.
    holds_shards = {name for name, row in readme.items() if "shard" in row}
    assert holds_shards == {"main", "rr", "no-capture", "scratch-500", "scratch-notokens"}
    for store in builds.TABLE_STORES:
        call = builds.store_call(store, tmp_path / store)
        assert bool(call.kwargs.get("tokenize", False)) == (store in holds_shards), store


def test_the_main_store_of_a_world_with_an_extension_is_the_replicas_build_of_that_extension(need_module, tmp_path):
    """The rows of an extension that is on are served by the main build plus the release of that extension, a build of its own
    (``replica.fixture.EXTENSION_STORES``: ``main+generating-model`` is the main build, then R3). ``store_call("main", ...,
    extensions=...)`` is that build, tokenized like main; with no extension it is the call of main, unchanged."""
    builds = _builds(need_module)
    fixture = importlib.import_module("replica.fixture")
    assert builds.MAIN_WITH_EXTENSION == {"generating-model": "main+generating-model"}
    for name, key in builds.MAIN_WITH_EXTENSION.items():
        assert key in fixture.EXTENSION_STORES and fixture.EXTENSION_STORES[key][0] == "main", name
    plain = builds.store_call("main", tmp_path / "main")
    assert builds.store_call("main", tmp_path / "main", extensions=()) == plain
    on = builds.store_call("main", tmp_path / "main", extensions=("generating-model",))
    assert on.function == plain.function and on.args == plain.args
    assert on.kwargs == {**plain.kwargs, "store": "main+generating-model"}
    assert on.kwargs["tokenize"] is True
    with pytest.raises(ValueError):
        builds.store_call("rr", tmp_path / "rr", extensions=("generating-model",))
    with pytest.raises(ValueError):
        builds.store_call("main", tmp_path / "main", extensions=("no-such-extension",))


def test_the_spec_and_the_parallel_count_are_passed_on(need_module, tmp_path):
    builds = _builds(need_module)
    call = builds.store_call("main", tmp_path / "main", spec=tmp_path / "spec.toml", parallel=3)
    assert Path(call.kwargs["spec"]) == tmp_path / "spec.toml"
    assert call.kwargs["parallel"] == 3
    assert builds.store_call("main", tmp_path / "main").kwargs.get("spec") is None


def test_a_name_that_is_no_store_is_refused(need_module, tmp_path):
    builds = _builds(need_module)
    with pytest.raises(ValueError):
        builds.store_call("no-such-store", tmp_path / "x")


def test_the_runs_of_a_store_are_the_readmes_in_the_order_of_the_spec(need_module):
    builds = _builds(need_module)
    spec_order = _spec_runs()
    assert spec_order == ("E", "E-rr", "D", "Y", "K")
    expected = {store: tuple(r for r in spec_order if r in runs) for store, runs in README_RUNS.items()}
    assert expected == {"main": ("E", "D", "Y", "K"), "rr": ("E-rr",)}
    assert {k: tuple(v) for k, v in builds.RUNS_OF_STORE.items()} == expected


def test_build_runs_is_the_replicas_command_with_the_runs_of_the_store(need_module, tmp_path):
    builds = _builds(need_module)
    for store, runs in README_RUNS.items():
        argv = builds.runs_argv(store, tmp_path / "runs", fixture_build=tmp_path / "fixture",
                                python="py")
        assert argv[:4] == ["py", "-m", "replica", "build-runs"]
        assert sbx_kit.flags(argv) == {
            "--out": str(tmp_path / "runs"), "--fixture-build": str(tmp_path / "fixture"),
            "--runs": ",".join(runs)}
        for flag in sbx_kit.flags(argv):
            assert f'"{flag}"' in _cli_text(), f"{flag} is a flag of the replica's command line"
        assert "build-runs" in _cli_text()


def test_build_runs_passes_the_spec_and_the_timeout_when_given(need_module, tmp_path):
    builds = _builds(need_module)
    argv = builds.runs_argv("main", tmp_path / "runs", fixture_build=tmp_path / "fixture",
                            python="py", spec=tmp_path / "spec.toml", timeout=900)
    given = sbx_kit.flags(argv)
    assert given["--spec"] == str(tmp_path / "spec.toml")
    assert float(given["--timeout"]) == 900.0


def test_a_store_with_no_runs_has_no_run_commands(need_module, tmp_path):
    builds = _builds(need_module)
    for store in ("swap", "import", "no-capture"):
        with pytest.raises(ValueError):
            builds.runs_argv(store, tmp_path / "runs", fixture_build=tmp_path / "f", python="py")
        with pytest.raises(ValueError):
            builds.register_runs_argv(store, tmp_path / "s.sqlite", tmp_path / "runs", python="py")


def test_register_runs_is_the_replicas_command_with_the_runs_of_the_store(need_module, tmp_path):
    builds = _builds(need_module)
    for store, runs in README_RUNS.items():
        argv = builds.register_runs_argv(store, tmp_path / "store.sqlite", tmp_path / "runs",
                                         python="py", spec=tmp_path / "spec.toml")
        assert argv[:4] == ["py", "-m", "replica", "register-runs"]
        assert sbx_kit.flags(argv) == {
            "--store": str(tmp_path / "store.sqlite"), "--runs-folder": str(tmp_path / "runs"),
            "--runs": ",".join(runs), "--spec": str(tmp_path / "spec.toml")}
        for flag in sbx_kit.flags(argv):
            assert f'"{flag}"' in _cli_text()
        assert "register-runs" in _cli_text()


def test_the_set_sizes_are_those_at_which_the_tables_rows_are_asked(need_module, facts):
    builds = _builds(need_module)
    asked = {size for sizes in facts.sizes_by_listed.values() for size in sizes if size is not None}
    assert asked == {"small", "10k", "100k"}
    assert set(builds.SET_SIZES) == asked


def test_the_generator_gets_the_parameters_of_each_size(need_module, tmp_path, acceptance_modules):
    builds = _builds(need_module)
    sets = acceptance_modules("acceptance_sets")
    flag_of = {"records": "--records", "seed": "--seed", "length": "--length", "pairs": "--pairs",
               "sanitizer_cases": "--sanitizer-cases"}
    for size in builds.SET_SIZES:
        argv = builds.generate_argv(size, tmp_path / size, python="py")
        assert argv[0] == "py" and argv[1] == "fixture/plant/generate.py"
        assert (REPO / argv[1]).is_file()
        given = sbx_kit.flags(argv)
        want = {flag_of[key]: str(value) for key, value in sets.SIZES[size].items()}
        want["--out"] = str(tmp_path / size)
        want["--dedup"] = "on" if sets.DEDUP[size] else "off"
        assert given == want, size


def test_the_three_sizes_differ_where_the_acceptance_tests_say_they_do(need_module, tmp_path):
    builds = _builds(need_module)
    records = {size: sbx_kit.flags(builds.generate_argv(size, tmp_path / size, python="py"))["--records"]
               for size in builds.SET_SIZES}
    assert records == {"small": "800", "10k": "10000", "100k": "100000"}
    dedup = {size: sbx_kit.flags(builds.generate_argv(size, tmp_path / size, python="py"))["--dedup"]
             for size in builds.SET_SIZES}
    assert dedup == {"small": "off", "10k": "on", "100k": "off"}


def test_a_size_that_is_not_asked_is_refused(need_module, tmp_path):
    builds = _builds(need_module)
    for size in ("1M", "500", None, ""):
        with pytest.raises(ValueError):
            builds.generate_argv(size, tmp_path / "x", python="py")


def test_a_generated_set_is_built_with_capture_in_the_round_robin_layout(need_module, tmp_path):
    builds = _builds(need_module)
    out = tmp_path / "build"
    argv = builds.set_build_argv(tmp_path / "set", out, store=tmp_path / "store.sqlite", python="py",
                                 tokenizer_spec=tmp_path / "spec.toml", jobs=4)
    assert argv[:4] == ["py", "-m", "replica", "build-set"]
    assert sbx_kit.flags(argv) == {
        "--set": str(tmp_path / "set"), "--out": str(out), "--layout": LAYOUT,
        "--tokenizer-spec": str(tmp_path / "spec.toml"), "--capture": "",
        "--store": str(tmp_path / "store.sqlite"), "--jobs": "4"}
    for flag in sbx_kit.flags(argv):
        assert f'"{flag}"' in _cli_text()


def test_a_store_file_inside_the_build_folder_is_refused(need_module, tmp_path):
    builds = _builds(need_module)
    out = tmp_path / "build"
    with pytest.raises(ValueError):
        builds.set_build_argv(tmp_path / "set", out, store=out / "store.sqlite", python="py",
                              tokenizer_spec=tmp_path / "spec.toml")


def test_a_generated_set_with_no_capture_is_one_call_of_the_replicas_recovering_builder(
        need_module, tmp_path):
    builds = _builds(need_module)
    module, _, attribute = "replica.fixture.build_set_store".rpartition(".")
    assert callable(getattr(importlib.import_module(module), attribute))
    call = builds.set_store_call(tmp_path / "set", tmp_path / "build", tokenizer_spec=tmp_path / "spec.toml",
                                 jobs=4)
    assert call.function == "replica.fixture.build_set_store"
    assert [Path(a) for a in call.args] == [tmp_path / "build", tmp_path / "set"]
    assert call.kwargs["layout"] == LAYOUT
    assert Path(call.kwargs["tokenizer_spec"]) == tmp_path / "spec.toml"
    assert call.kwargs["parallel"] == 4
    assert call.kwargs.get("store", "no-capture") == "no-capture"


def test_the_plans_are_plain_data_and_make_nothing_on_disk(need_module, tmp_path):
    builds = _builds(need_module)
    before = sbx_kit.tree_digest(tmp_path)
    builds.store_call("main", tmp_path / "main")
    builds.runs_argv("main", tmp_path / "runs", fixture_build=tmp_path / "f", python="py")
    builds.generate_argv("small", tmp_path / "set", python="py")
    builds.set_store_call(tmp_path / "set", tmp_path / "build", tokenizer_spec=tmp_path / "spec.toml")
    assert sbx_kit.tree_digest(tmp_path) == before


def test_the_module_holds_no_location_of_this_machine(need_module):
    assert sbx_kit.local_path_constants(_builds(need_module)) == []
