"""``python -m replica repair`` with the fourth word of an artifact, and the refusal of a
shard that lies in no layout (DECISIONS.md D-141, D-120; docs/rebuild.md 2.2;
docs/interfaces.md 7.1 and 9.5; docs/replica.md, the repair section).

A report may give a shard the action "clean, place not stated": its own contents are clean
and the place of the copies that return is not stated. The replica reads the word like the
two it knew, and builds such a shard by its own layout, registering it as
``successor of <name>`` like every shard of the repaired build.

A shard entry of the report that lies in no layout of the build is refused. The refusal reads
the report and the layouts only, so it comes before the repaired side and the reference side
are built: a refused repair registers no successor and writes no folder.

The historical builds are the hand-checked fixture, captured, as ``test_replica_repair.py``
makes it; the partitioned one is tokenized with the stand-in of the conftest.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pytest
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica import cli, repair
from replica.errors import InputError
from replica.fixture import build_captured_fixture, default_spec_path

REPO = Path(__file__).resolve().parents[2]
WORD = "clean, place not stated"


def module_values(path: Path) -> dict:
    """The values the module assigns at its top level to a name, from its syntax tree."""
    found: dict = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = eval(  # noqa: S307 - constants, tuples and names only
                    compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}}, dict(found))
            except Exception:
                continue
    return found


def a_report(*artifacts: dict) -> dict:
    return {
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
        "records": [],
        "artifacts": list(artifacts),
    }


def run(argv, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --- The word is read ------------------------------------------------------------------------------------


def test_the_replica_reads_the_word_beside_the_two_it_knew():
    held = module_values(REPO / "replica" / "repair.py")
    assert WORD in held["ARTIFACT_ACTIONS"], held["ARTIFACT_ACTIONS"]
    assert WORD in v.ARTIFACT_ACTIONS
    assert set(held["ARTIFACT_ACTIONS"]) == set(v.ARTIFACT_ACTIONS) | {v.UNRESOLVED}


def test_a_report_that_gives_a_shard_the_word_is_read(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(a_report(
        {"artifact": "E/shard-1", "kind": "shard", "action": WORD, "successor": None})),
        encoding="utf-8")
    try:
        read = repair.read_report(path)
    except InputError as refused:
        raise AssertionError(f"the replica refused the report: {refused}") from refused
    assert read["artifacts"][0]["action"] == WORD


# --- A shard that reads the word is built by its layout ---------------------------------------------------------


@pytest.fixture(scope="module")
def partitioned(tmp_path_factory, megatron_stand_in):
    """The historical build of the fixture in its partitioned layout, tokenized by the
    stand-in, captured: the shards ``E/shard-1`` to ``E/shard-3``."""
    base = tmp_path_factory.mktemp("repair-keep-partitioned")
    with megatron_stand_in(base / "env") as env:
        built = build_captured_fixture(
            base / "history", tokenize=True, layouts=("partitioned",), releases=("R1", "R2"),
            imports=(), megatron_env=env)
    return built, env


def test_a_shard_that_reads_the_word_is_built_and_registered_as_a_successor(
        partitioned, tmp_path, megatron_stand_in, capsys):
    built, env = partitioned
    history = tmp_path / "history"
    shutil.copytree(built.out.parent, history)
    store = history / built.store.name
    report = tmp_path / "report.json"
    report.write_text(json.dumps(a_report(
        {"artifact": "E/shard-1", "kind": "shard", "action": WORD, "successor": None},
        {"artifact": "E/shard-2", "kind": "shard", "action": "keep", "successor": None})),
        encoding="utf-8")
    out = tmp_path / "repaired"
    with megatron_stand_in(tmp_path / "stand-in"):  # the stand-in is also the environment the command reads
        code, printed, err = run(
            ["repair", "--report", str(report), "--historical", str(history / built.out.name),
             "--out", str(out), "--store", str(store), "--spec", str(default_spec_path()),
             "--layout", "partitioned"], capsys)
    assert code == 0, err
    shards = out / "pipeline" / "partitioned" / "shards"
    assert (shards / "shard-1.bin").is_file() and (shards / "shard-1.idx").is_file()
    with Store.open(store, readonly=True) as opened:
        assert opened.resolve("successor of E/shard-1") is not None
        assert opened.resolve("successor of E/shard-2") is not None


# --- The refusal of a shard that lies in no layout comes before the builds ------------------------------------------


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    folder = tmp_path_factory.mktemp("repair-keep-historical") / "history"
    return build_captured_fixture(folder, releases=("R1", "R2"), imports=(), layouts=())


def test_a_shard_that_lies_in_no_layout_is_refused_before_either_build_is_made(
        historical, tmp_path, capsys):
    history = tmp_path / "history"
    shutil.copytree(historical.out.parent, history)
    store = history / historical.store.name
    report = tmp_path / "report.json"
    report.write_text(json.dumps(a_report(
        {"artifact": "R1", "kind": "jsonl", "action": "rewrite", "successor": "successor of R1"},
        {"artifact": "R2", "kind": "jsonl", "action": "keep", "successor": None},
        {"artifact": "nowhere/shard-9", "kind": "shard", "action": "rewrite",
         "successor": "successor of nowhere/shard-9"})), encoding="utf-8")
    out, reference, sides = tmp_path / "repaired", tmp_path / "reference", tmp_path / "sides.json"
    code, printed, err = run(
        ["repair", "--report", str(report), "--historical", str(history / historical.out.name),
         "--out", str(out), "--store", str(store), "--spec", str(default_spec_path()),
         "--reference-withdrawn", "A", "--reference-out", str(reference),
         "--sides", str(sides)], capsys)
    assert code == 1, (code, err)
    assert printed.strip() == ""
    lines = err.strip().splitlines()
    assert len(lines) == 1 and "nowhere/shard-9" in lines[0], err
    assert not out.exists(), "the repaired side was built"
    assert not reference.exists(), "the reference side was built"
    assert not sides.exists(), "a sides file was written"
    with Store.open(store, readonly=True) as opened:
        assert opened.resolve("successor of R1") is None, "a successor was registered"
