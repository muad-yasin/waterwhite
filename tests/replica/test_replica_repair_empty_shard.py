"""A repair of a partitioned build under which one source's shard is left with no
document, verified with a reference side (plan 4.2, 4.4, 5.5 Check 9; docs/interfaces.md
sections 7.2, 7.3 and 9.5).

The set is the small generated one with no generation job, built with capture on in
the partitioned layout, one shard per source file. Under "withdraw A" the report
retires every document of A's shard (they are A's own records) and prints no document
left for it, so the repaired side does not write the shard, and the reference side
(built from the sources that never held A) has no shard for A either. Beside that,
the report re-sources A's copies of the planted pairs in the release to B's copies;
the repaired and the reference builds both tokenize those into a shard of B's file.
The report is asked with the placement of the partitioned layout (docs/cli.md,
"report"), which the test writes from the layout rule of docs/replica.md (one shard
per source file, and a shard for the generation outputs) and the set's own set.toml. The
shard of B is no shard the historical store holds, so the report names it as a landing
shard of the layout (docs/interfaces.md 7.1) and the comparison reads it like a
rewritten one: the shard of B is compared, on both sides, in seeded mode.

The whole flow runs on the documented command lines in this process: the tool's
``report`` on the historical store, ``python -m replica repair`` with a reference set
and a sides file, and the tool's ``verify``. The tokenizer is the stand-in of the
replica tests' conftest. The tool's own report is where the shard is shown to be left
with no document (``documents_in_successor``); what the replica prints about its own
registrations is read by no assertion here.
"""

from __future__ import annotations

import contextlib
import io
import json
import tomllib
from pathlib import Path

import pytest
from waterwhite import cli as tool
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica import cli as replica_cli
from replica.fixture import default_spec_path
from replica.sets import build_captured_set

EMPTIED = "set/shard-A"
STAYING = ("set/shard-C", "set/shard-Z")
LANDING = "set/shard-B"
PREFIX = "set/"


def run_tool(store: Path, *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    try:
        code = tool.main([*argv, "--store", str(store)], stdout=out, stderr=err)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    return code, out.getvalue(), err.getvalue()


def placement_of(set_folder: Path, prefix: str) -> dict:
    """The placement file of the partitioned layout of a generated set, written from the
    layout rule of docs/replica.md (the table "What set.toml does not say": one shard
    ``shard-<NAME>`` per source file and one shard ``shard-gen`` for the generation
    outputs) and the source names of the set's own set.toml: the copies of ``src/<NAME>``
    land in ``<prefix>shard-<NAME>``, everything else in ``<prefix>shard-gen``."""
    with open(set_folder / "set.toml", "rb") as f:
        declared = tomllib.load(f)["sources"]
    names = [Path(entry["file"]).stem if isinstance(entry, dict) and "file" in entry else key
             for key, entry in declared.items()]
    return {"form": "waterwhite.placement/1", "by": "source",
            "shards": {f"src/{name}": f"{prefix}shard-{name}" for name in names},
            "otherwise": f"{prefix}shard-gen"}


def returning_copies_of(plant_count, set_folder: Path, source: str) -> list[str]:
    """The records of ``src/<source>`` that the report promotes or re-sources onto under
    "withdraw A", as ``src/<source>:L<line>``: for each release, the copy the plant list
    says a re-sourced row takes (``resource_to``), which is also the loser of dedup the
    report promotes. Read from the plant list (fixture/plant/count.py), never from a report."""
    settings, rows = plant_count.load_set(set_folder)
    plant = plant_count.Plant(settings, rows)
    found = set()
    for release, inputs in plant.release_inputs.items():
        for record in inputs:
            outcome = plant.expected[release][record]
            if outcome["withdraw_A"] != plant_count.RE_SOURCE:
                continue
            copy = plant.by_id[outcome["resource_to"]]
            if copy["source"] == source:
                found.add(f"src/{source}:L{copy['line']}")
    return sorted(found)


@pytest.fixture(scope="module")
def repaired(plant_generate, set_parameters, tmp_path_factory, megatron_stand_in):
    """The historical build, the tool's report, the replica's repair with its reference
    side and its sides file. Gives the folder, the store, the report file and its
    document, and the sides file."""
    base = tmp_path_factory.mktemp("empty-shard")
    parameters = {**set_parameters, "derivations": 0}
    plant_generate.generate(base / "set", **parameters)
    plant_generate.generate(base / "clean", clean=True, **parameters)
    with megatron_stand_in(base / "env") as env:
        built = build_captured_set(base / "history", base / "set", layouts=("partitioned",),
                                   megatron_env=env)
        placement = base / "placement.json"
        document = placement_of(base / "set", PREFIX)
        placement.write_text(json.dumps(document), encoding="utf-8")
        # the source artifacts the placement names are the ones the build registered
        with Store.open(built.store, readonly=True) as opened:
            for source in document["shards"]:
                assert opened.resolve(source) is not None, f"the store has no artifact {source}"
        code, out, err = run_tool(built.store, "report", "withdraw", "A", "--layout",
                                  f"partitioned={PREFIX}", "--placement", f"partitioned={placement}")
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


def verification(repaired, sides: Path, *extra: str) -> dict:
    code, out, err = run_tool(repaired["store"], "verify", "--report", str(repaired["report_file"]),
                              "--sides", str(sides), *extra)
    assert code == 0, err
    return json.loads(out)


def said_about(result: dict, name: str) -> list[str]:
    """The sentences the verification gives about ``name`` in its comparison entries."""
    said = []
    for key in ("differences", "releases", "shards"):
        for entry in result[key]:
            if entry.get("artifact") == name:
                said.extend(sorted(text for field, text in entry.items()
                                   if field != "artifact" and isinstance(text, str) and " " in text))
    return said


def without(repaired, name: str) -> Path:
    """The sides file of the repair with one more artifact named on neither side."""
    document = json.loads(repaired["sides"].read_text(encoding="utf-8"))
    for side in ("repaired", "reference"):
        document[side].pop(name, None)
    path = repaired["sides"].with_name(f"sides-without-{name.replace('/', '-')}.json")
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# --- The case, as the tool's own record and the sides file show it -------------------------------


def test_the_report_shows_the_shard_of_a_left_with_no_document_and_the_release_re_sourced(repaired):
    entries = {e["artifact"]: e for e in repaired["report"]["artifacts"]}
    emptied = entries[EMPTIED]
    assert (emptied["kind"], emptied["action"]) == (v.KIND_SHARD, v.ACTION_REWRITE)
    assert emptied["documents"] > 0
    assert emptied["documents_in_successor"] == 0
    assert emptied["records_removed"] == emptied["documents"]
    for name in STAYING:
        assert entries[name]["action"] == v.ACTION_KEEP, name
    counts = entries["R1"]["counts"]
    assert counts[v.ACTION_RE_SOURCE] > 0 and counts[v.ACTION_PROMOTE] > 0
    assert counts[v.UNRESOLVED] == 0 and repaired["report"]["counts"]["records"][v.UNRESOLVED] == 0


def test_the_sides_file_names_the_shard_on_neither_side_and_the_others_on_both(repaired):
    document = json.loads(repaired["sides"].read_text(encoding="utf-8"))
    for side in ("repaired", "reference"):
        assert EMPTIED not in document[side], side
        assert set(STAYING) <= set(document[side]), side
    assert {"R1", "R2"} <= set(document["repaired"]) & set(document["reference"])


# --- What verify says ---------------------------------------------------------------------------


def test_the_multisets_read_equal_beside_the_emptied_shard(repaired):
    result = verification(repaired, repaired["sides"], "--mode", "seeded")
    assert result["unresolved_records"] == 0
    assert [(e["artifact"], e["equal"]) for e in result["releases"]] == [("R1", True), ("R2", True)]
    assert (result["multisets"], result["differing_records"]) == (v.SIDES_EQUAL, 0)


def test_the_digests_read_equal_in_seeded_mode_beside_the_emptied_shard(repaired):
    result = verification(repaired, repaired["sides"], "--mode", "seeded")
    # The shards in the order of the sides file the repair writes: the report's shards,
    # then the landing shard the placement names (the store does not hold it).
    assert [(e["artifact"], e["equal"]) for e in result["shards"]] == [
        ("set/shard-C", True), ("set/shard-Z", True), ("set/shard-B", True)]
    assert result["per_shard_digests"] == v.SIDES_EQUAL, result["per_shard_digests"]


def test_the_emptied_shard_is_named_with_a_reason_other_than_a_lost_shards(repaired):
    # The same repair, with shard C named on neither side as well: C holds documents, so
    # its absence is a file the comparison lacks.
    result = verification(repaired, without(repaired, STAYING[0]), "--mode", "seeded")
    reason, lost_reason = said_about(result, EMPTIED), said_about(result, STAYING[0])
    assert reason, f"{EMPTIED} is not named with a reason"
    assert lost_reason, f"{STAYING[0]} is not named with a reason"
    assert reason != lost_reason, f"{EMPTIED} gets the reason of {STAYING[0]}: {reason}"


def test_a_shard_that_holds_documents_and_is_on_neither_side_still_blocks_equal(repaired):
    result = verification(repaired, without(repaired, STAYING[1]), "--mode", "seeded")
    assert result["per_shard_digests"] != v.SIDES_EQUAL
    assert said_about(result, STAYING[1])


def test_the_report_names_the_shard_of_b_as_a_landing_shard_the_store_does_not_hold(
        repaired, plant_count):
    # With the placement stated, the copies of B's file land in the shard of B; the
    # historical store holds no such shard (the set has no document of B's file left for
    # it to tokenize into one), so the report names it under the layout's landing.
    expected = returning_copies_of(plant_count, repaired["base"] / "set", "B")
    assert expected, "the plant list has no returning copy of src/B"
    placements = repaired["report"]["groups"]["placements"]
    landing = [entry for entry in placements["partitioned"]["landing"] if entry["shard"] == LANDING]
    assert len(landing) == 1, placements["partitioned"]["landing"]
    assert landing[0]["in_store"] is False
    assert sorted(landing[0]["receives"]) == expected


def test_the_sides_file_names_the_landing_shard_on_both_sides(repaired):
    # The shard of B is named under landing, so the repair names its files in the sides
    # file on both sides, as it does for every shard the comparison reads (docs/interfaces.md 7.2).
    landing = [entry["shard"] for entry in
               repaired["report"]["groups"]["placements"]["partitioned"]["landing"]]
    assert LANDING in landing, landing
    document = json.loads(repaired["sides"].read_text(encoding="utf-8"))
    for side in ("repaired", "reference"):
        assert LANDING in document[side], (side, sorted(document[side]))
