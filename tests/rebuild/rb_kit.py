"""What the tests of the rebuild report and its verification share.

The store they use is the hand-built one of ``tests/coverage/ww_world.py`` (sources
A, B, C, Z; the generated P, T and S; releases R1 and R2; two shards, two mixtures,
runs E, D and Y), with the additions this file makes: the drop of the dedup loser
in R1, a release R1-swap whose dedup kept B's copy, and the successors a repair registers (``successor of R1``, ``successor of
shard-3``, ``successor of mix-2``).

Expected values in the tests come from the documents of the form (docs/interfaces.md
sections 7.1 to 7.4, docs/cli.md, docs/rebuild.md), from the construction of the
world below, and from ``waterwhite.ids`` where a digest is named by its definition.
Nothing here or in a test reads a value out of the code under test.

A module of the rebuild package does not exist when these tests are written, so a
test asks for it with ``need``: a missing module is an assertion that fails, not an
import error.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

from waterwhite import cli, coverage, requests
from waterwhite import vocabulary as v
from waterwhite.records import DropRecord, EdgeInput, EdgeRecord, SnapshotShard
from waterwhite.store.store import Store

REPO = Path(__file__).resolve().parents[2]


def _load_world():
    """The hand-built world, loaded from its file whether or not another test folder
    has put its own folder on the path."""
    held = sys.modules.get("ww_world")
    if held is not None:
        return held
    spec = importlib.util.spec_from_file_location("ww_world", REPO / "tests" / "coverage" / "ww_world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["ww_world"] = module
    spec.loader.exec_module(module)
    return module


ww = _load_world()

REQUEST_A = {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}


# --- A module that may not exist yet -------------------------------------------------


def load(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def need(name: str):
    """The module, or a failed assertion that says it does not exist."""
    module = load(name)
    assert module is not None, f"the module {name} does not exist"
    return module


def need_attr(module_name: str, attribute: str):
    module = need(module_name)
    found = getattr(module, attribute, None)
    assert found is not None, f"{module_name} has no {attribute}"
    return found


# --- The world -----------------------------------------------------------------------

RELEASE_TEXTS = ["passage X", "paraphrase of X", "Uebersetzung von P", "summary of SA and SC",
                 "z one", "z two"]


def build(folder: Path, *, swap: bool = False, successors: bool = False,
          successor_map: str = "whole", unseeded: bool = False):
    """The base world and what a test asks to be added to it.

    ``swap`` adds R1-swap: the bytes of R1, with dedup's winner B's copy of X and A's
    copy the loser. ``successors`` registers what a repair
    makes: ``successor of R1`` (X from B's copy, and the two Z rows), ``successor of
    shard-3`` and ``successor of mix-2``; ``successor_map`` "none" leaves the
    successor shard with no map, "twins" gives its first document a map row of two
    candidates (a shard rebuilt without capture). ``unseeded`` records no random seed
    in the run records of the world."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = ww.run_record

    def without_seed(*args, **kwargs):
        record = saved(*args, **kwargs)
        record["settings"]["random_seed"] = None
        return record

    if unseeded:
        ww.run_record = without_seed
    try:
        world = ww.build(folder)
    finally:
        ww.run_record = saved
    store = world.store
    a, b, c, z = (world[n] for n in ("src/A", "src/B", "src/C", "src/Z"))
    p, t, s = world["gen/P"], world["gen/T"], world["gen/S"]
    # The record that dedup dropped in favour of A's copy: B's copy of X.
    store.add_drops(world["R1"], [DropRecord(ww.ref(b, 1), v.DROP_DEDUP_LOSER, "dedup", "L1")])
    if swap:
        winner_b = EdgeRecord(
            "L1", (EdgeInput(ww.ref(a, 1)), EdgeInput(ww.ref(b, 1), selected=True)),
            ww.step("dedup"), parent_mode=v.MODE_ANY,
        )
        world.ids["R1-swap"] = ww.record(
            store, folder / "R1-swap.jsonl", RELEASE_TEXTS, "R1-swap", v.ROLE_RELEASE,
            ww.step("release", "R1"), [a, b, p, t, s, z],
            [winner_b, ww.copy_edge(2, ww.ref(p, 1)), ww.copy_edge(3, ww.ref(t, 1)),
             ww.copy_edge(4, ww.ref(s, 1)), ww.copy_edge(5, ww.ref(z, 1)),
             ww.copy_edge(6, ww.ref(z, 2))],
        )
        store.add_drops(world["R1-swap"], [DropRecord(ww.ref(a, 1), v.DROP_DEDUP_LOSER, "dedup", "L1")])
    if successors:
        add_successors(world, folder, successor_map)
    return world


def add_successors(world, folder: Path, successor_map: str = "whole") -> None:
    store = world.store
    b, z = world["src/B"], world["src/Z"]
    texts = ["passage X", "z one", "z two"]
    successor = world.ids["successor of R1"] = ww.record(
        store, folder / "successors/R1.jsonl", texts, "successor of R1", v.ROLE_RELEASE,
        ww.step("release", "R1"), [b, z],
        [ww.copy_edge(1, ww.ref(b, 1)), ww.copy_edge(2, ww.ref(z, 1)), ww.copy_edge(3, ww.ref(z, 2))],
    )
    if successor_map == "none":
        rows = None
    elif successor_map == "twins":
        rows = [ww.twins(0, ww.ref(world["src/A"], 1), ww.ref(b, 1)), ww.recovered(1, ww.ref(successor, 3))]
    else:
        rows = [ww.recovered(0, ww.ref(successor, 2)), ww.recovered(1, ww.ref(successor, 3))]
    shard, digest = ww.shard(
        store, folder / "successors/shard-3", "successor of shard-3", [successor], rows)
    world.ids["successor of shard-3"] = shard
    kept = store.artifact(world["shard-1"])
    world.ids["successor of mix-2"] = store.register_mixture(
        name="successor of mix-2",
        shards=[SnapshotShard(0, kept.instance_id, "shards/shard-1", kept.content_digest),
                SnapshotShard(1, shard, "successors/shard-3", digest)],
        settings={"mix": 2},
    )


def sizes_of(folder: Path, name: str) -> int:
    """The size of a file of the world, or of the two files of a shard."""
    path = folder / name
    if path.suffix == "":
        return sum((path.parent / f"{path.name}{suffix}").stat().st_size for suffix in (".bin", ".idx"))
    return path.stat().st_size


# --- The command line ----------------------------------------------------------------


def run(world, *argv: str):
    """Run the command line in this process against the world's store file, as
    ``tests/coverage/test_coverage_listing_cli.py`` does. Returns the exit code, what
    went to standard output and what went to standard error."""
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    try:
        try:
            code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
        except SystemExit as leaving:
            code = int(leaving.code or 0)
    finally:
        world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


def report_through_cli(world, *argv: str) -> dict:
    code, out, err = run(world, "report", "withdraw", "A", *argv)
    assert code == 0, f"report withdraw A exited with {code}: {err}"
    return json.loads(out)


# --- The report through the seam of docs/interfaces.md section 7.4 --------------------


def report_parts(world, source: str = "A", *, layouts=None, records=()) -> dict:
    """Every key of the report but the restarts, from ``report_without_restarts``."""
    rebuild = need("waterwhite.rebuild")
    function = getattr(rebuild, "report_without_restarts", None)
    assert function is not None, "waterwhite.rebuild has no report_without_restarts"
    request = requests.withdraw(world.store, source, records=records)
    session = coverage.Session(world.store, request)
    return function(world.store, request, session, layouts=layouts)


def artifact_entry(document: dict, name: str) -> dict:
    found = [entry for entry in document["artifacts"] if entry["artifact"] == name]
    assert len(found) == 1, f"{name}: {len(found)} entries among {[e['artifact'] for e in document['artifacts']]}"
    return found[0]


def record_entries(document: dict, artifact: str) -> list[dict]:
    return [entry for entry in document["records"] if entry["artifact"] == artifact]


def record_entry(document: dict, artifact: str, record: str) -> dict:
    found = [entry for entry in record_entries(document, artifact) if entry["record"] == record]
    assert len(found) == 1, f"{artifact} {record}: {len(found)} entries"
    return found[0]


def actions_of(document: dict, artifact: str) -> dict[str, str]:
    return {entry["record"]: entry["action"] for entry in record_entries(document, artifact)}


# --- The documents a repair is verified with, written by hand ---------------------------


def hand_report(world, artifacts, *, unresolved: int = 0, source: str = "A") -> dict:
    """A report as the form of section 7.1 fixes it, with the keys `verify` reads and
    no others. ``artifacts`` is a list of (name, action, successor name or None)."""
    entries = []
    for name, action, successor in artifacts:
        held = world.store.resolve(name)
        entries.append({"artifact": name, "instance": held.instance_id, "kind": held.kind,
                        "action": action, "successor": successor})
    return {
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": source, "members": [f"{source}/r1"], "records": []},
        "artifacts": entries,
        "counts": {"records": {"keep": 0, "re-source": 0, "promote": 0, "retire": 0,
                               "unresolved": unresolved}},
    }


def sides_document(repaired: dict, reference: dict) -> dict:
    return {"form": v.FORM_SIDES, "repaired": repaired, "reference": reference}


def write_lines(path: Path, texts: list[str], field: str = "text") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({field: text}) + "\n" for text in texts), encoding="utf-8")
    return path


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def copy_shard(folder: Path, source_name: str, target_name: str, *, flip: bool = False) -> Path:
    """A copy of the two files of a shard of the world, with one byte of the token file
    changed when ``flip``. Returns the prefix of the copy."""
    source = folder / source_name
    target = folder / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".bin", ".idx"):
        data = bytearray((source.parent / f"{source.name}{suffix}").read_bytes())
        if flip and suffix == ".bin":
            data[0] ^= 0xFF
        (target.parent / f"{target.name}{suffix}").write_bytes(bytes(data))
    return target


DOCUMENT_NAME = re.compile(r"^D\d+$")
