"""A repair of a partitioned build whose report states no placement, verified with a
reference side (docs/rebuild.md 2.2; docs/interfaces.md sections 7.1 and 7.3; docs/cli.md,
"report" and "verify").

The build is the one of ``test_replica_repair_empty_shard.py``: the small generated set
with no generation job, built with capture on in the partitioned layout, one shard per
source file. Under "withdraw A" the report retires every document of A's shard and
re-sources A's copies of the planted pairs in the release to B's copies. Here the report is
asked with no placement, so it does not say where those copies land. The shards of C and of
Z hold nothing of A, and whether a returning copy lands in them is not stated: their own
contents are clean, and the report says so with the fourth word of an artifact, never with
"keep". Both builds tokenize the copies into a shard of B's file, which the report does not
name, so the verification compares nothing of it: it does not read "equal", and it names the
shard.

The whole flow runs on the documented command lines in this process, as the other test does.
The tokenizer is the stand-in of the replica tests' conftest.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import pytest
from waterwhite import cli as tool
from waterwhite import vocabulary as v

from replica import cli as replica_cli
from replica.fixture import default_spec_path
from replica.sets import build_captured_set

WORD = "clean, place not stated"
EMPTIED = "set/shard-A"
STAYING = ("set/shard-C", "set/shard-Z")
UNNAMED = "set/shard-B"
PREFIX = "set/"


def run_tool(store: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    try:
        code = tool.main([*argv, "--store", str(store)], stdout=out, stderr=err)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def repaired(plant_generate, set_parameters, tmp_path_factory, megatron_stand_in):
    """The historical build, the tool's report with no placement, the replica's repair with
    its reference side and its sides file. Gives the folder, the store, the report file and
    its document, and the sides file."""
    base = tmp_path_factory.mktemp("no-placement")
    parameters = {**set_parameters, "derivations": 0}
    plant_generate.generate(base / "set", **parameters)
    plant_generate.generate(base / "clean", clean=True, **parameters)
    with megatron_stand_in(base / "env") as env:
        built = build_captured_set(base / "history", base / "set", layouts=("partitioned",),
                                   megatron_env=env)
        code, out, err = run_tool(built.store, "report", "withdraw", "A", "--layout",
                                  f"partitioned={PREFIX}")
        assert code == 0, err
        report_file = base / "report.json"
        report_file.write_text(out, encoding="utf-8")
        sides = base / "sides.json"
        argv = ["repair", "--report", str(report_file), "--historical", str(built.out),
                "--out", str(base / "repaired"), "--store", str(built.store),
                "--set", str(base / "set"), "--layout", "partitioned",
                "--tokenizer-spec", str(default_spec_path()),
                "--reference-set", str(base / "clean"), "--reference-out", str(base / "reference"),
                "--sides", str(sides)]
        with contextlib.redirect_stdout(io.StringIO()):
            assert replica_cli.main(argv) == 0
    return {"base": base, "store": built.store, "report": json.loads(out),
            "report_file": report_file, "sides": sides}


def verification(repaired, *extra: str) -> dict:
    code, out, err = run_tool(repaired["store"], "verify", "--report", str(repaired["report_file"]),
                              "--sides", str(repaired["sides"]), *extra)
    assert code == 0, err
    return json.loads(out)


def test_the_shards_of_c_and_z_read_that_their_own_contents_are_clean_and_the_place_is_not_stated(repaired):
    entries = {e["artifact"]: e for e in repaired["report"]["artifacts"]}
    emptied = entries[EMPTIED]
    assert (emptied["kind"], emptied["action"]) == (v.KIND_SHARD, v.ACTION_REWRITE)
    for name in STAYING:
        assert entries[name]["action"] == WORD, (name, entries[name]["action"])


def test_the_multisets_read_not_compared_while_a_shard_both_builds_wrote_is_named_nowhere(repaired):
    result = verification(repaired, "--mode", "seeded")
    assert result["unresolved_records"] == 0
    assert [(e["artifact"], e["equal"]) for e in result["releases"]] == [("R1", True), ("R2", True)]
    assert (result["multisets"], result["differing_records"]) == ("not compared", 0)
    named = [d for d in result["differences"] if d["artifact"] == UNNAMED]
    assert len(named) == 1, [d["artifact"] for d in result["differences"]]
    assert isinstance(named[0]["note"], str) and "not compared" in named[0]["note"], named[0]


def test_the_digests_read_not_compared_in_seeded_mode_while_that_shard_goes_uncompared(repaired):
    result = verification(repaired, "--mode", "seeded")
    assert result["per_shard_digests"] == "not compared", result["per_shard_digests"]
