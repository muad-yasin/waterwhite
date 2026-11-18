"""What the tests of the keep rule share: worlds that add shards, mixtures, runs and
release files to the hand-built world of ``rb_kit.py``, a writer of placement files, and
the report asked through the command line with layouts and placements.

Under "withdraw A" the world's one returning copy is ``src/B:L1``: R1's first row is
re-sourced onto it and its row in the source file is the promoted record. Shard-1 holds
it (its first document is R2's first row, a copy of ``src/B:L1``), so under no layout
nothing is left unplaced.

The shards the tests add are named under a folder of their own (``xs/``, ``ys/``) so
that a layout names them by prefix and the world's own two shards (``shard-1``,
``shard-3``) fall in no named layout. Every shard that is added holds one document or two,
each tokenized from one row of a file of the world; what each document is a copy of is
the construction below, written in the tests that use it.

Expected values in the tests come from docs/rebuild.md section 2.2, docs/interfaces.md
section 7.1, DECISIONS.md D-141 and the construction of the worlds. Nothing here reads
a value out of the code under test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from waterwhite import vocabulary as v
from waterwhite.records import EdgeInput, EdgeRecord, SnapshotShard

from rb_kit import run, ww, write_json

# The fourth word of an artifact: its own contents are clean, and where the copies that
# return land is not stated (DECISIONS.md D-141; docs/rebuild.md 2.2).
WORD = "clean, place not stated"
PLACEMENT_FORM = "waterwhite.placement/1"
B_COPY = "src/B:L1"


# --- Adding to the world ------------------------------------------------------------------


def parent_of(world, artifact: str, line: int):
    """The record ``artifact:L<line>`` of the world."""
    return ww.ref(world[artifact], line)


def add_shard(world, name: str, parents, *, rows: bool = True, **options) -> str:
    """A shard registered as ``name`` with one document per parent, the document n
    mapped to the record ``parents[n]`` (a ``(artifact, line)`` pair, or a ready map
    row). ``rows=False`` registers it with no map at all. Returns its instance id."""
    held = []
    for ordinal, parent in enumerate(parents):
        if isinstance(parent, tuple) and isinstance(parent[0], str):
            held.append(ww.recovered(ordinal, parent_of(world, *parent)))
        else:
            held.append(parent)
    instance, _digest = ww.shard(
        world.store, world.folder / name, name, [world["R2"]],
        held if rows else None, **options)
    world.ids[name] = instance
    return instance


def add_mixture(world, name: str, shards: list[str]) -> str:
    """A mixture over the named shards in this order."""
    snapshot = []
    for position, shard in enumerate(shards):
        held = world.store.artifact(world[shard])
        snapshot.append(SnapshotShard(position, held.instance_id, shard, held.content_digest))
    world.ids[name] = world.store.register_mixture(
        name=name, shards=snapshot, settings={"mix": name})
    return world.ids[name]


def add_run(world, run_name: str, segments: list[tuple[str, str]]) -> None:
    """A run of the given (segment name, mixture name) pairs, each of two steps that
    read the first document of the first shard of its mixture, and a checkpoint at the
    last step of each segment named ``<run>@<segment's last step>``. A segment after the
    first starts from the checkpoint of the one before it."""
    parent = None
    first = 1
    for position, (segment, mixture) in enumerate(segments):
        steps = [(first, 1, None), (first + 1, 1, None)]
        ww.segment(world.store, run_name, segment, position, world[mixture], parent, steps,
                   {first: (0, 0), first + 1: (0, 0)})
        mark = f"{run_name}@{first + 1}"
        parent = world.ids[mark] = world.store.register_checkpoint(
            name=mark, segment=segment, step=first + 1)
        first += 2


def add_file(world, name: str, texts: list[str], edges: list[EdgeRecord], inputs: list[str], *,
             role: str = v.ROLE_INTERMEDIATE) -> str:
    """A JSONL file of the world, with the edges given for its rows."""
    world.ids[name] = ww.record(
        world.store, world.folder / f"{name}.jsonl", texts, name, role,
        ww.step("derive", name), [world[i] for i in inputs], edges)
    return world.ids[name]


def single_edge(line: int, parent) -> EdgeRecord:
    """A row copied from one parent: a single-parent, record-level, non-generation edge."""
    return ww.copy_edge(line, parent)


def many_edge(line: int, *parents) -> EdgeRecord:
    """A row made from several parents, all of them used (not an "any" edge)."""
    return EdgeRecord(f"L{line}", tuple(EdgeInput(p) for p in parents), ww.step("combine"),
                      parent_mode=v.MODE_ALL)


def any_edge(line: int, selected, *others) -> EdgeRecord:
    """A row whose parents are equivalent copies, ``selected`` the one taken."""
    return EdgeRecord(
        f"L{line}", (EdgeInput(selected, selected=True), *(EdgeInput(o) for o in others)),
        ww.step("dedup"), parent_mode=v.MODE_ANY)


def generation_edge(line: int, parent) -> EdgeRecord:
    """A row a generation job made from one record it was handed and used."""
    return ww.generation_edge(line, "rewrite-job", [parent], [parent])


# --- Placement files and the command ------------------------------------------------------


def placement_document(shards: dict[str, str], otherwise: str | None = None) -> dict:
    """A placement file as docs/cli.md writes it: the shard each source's copies land in."""
    return {"form": PLACEMENT_FORM, "by": "source", "shards": dict(shards), "otherwise": otherwise}


def write_placement(folder: Path, name: str, shards: dict[str, str],
                    otherwise: str | None = None) -> Path:
    return write_json(folder / f"{name}.placement.json", placement_document(shards, otherwise))


def report_with(world, layouts: dict[str, str], placements: dict[str, Path] | None = None,
                *extra: str) -> tuple[int, dict | None, str]:
    """``report withdraw A`` with the layouts and placements given. Returns the exit code,
    the document (None when the command did not print one) and standard error."""
    argv = ["report", "withdraw", "A"]
    for name, prefix in layouts.items():
        argv += ["--layout", f"{name}={prefix}"]
    for name, path in (placements or {}).items():
        argv += ["--placement", f"{name}={path}"]
    code, out, err = run(world, *argv, *extra)
    return code, (json.loads(out) if code == 0 and out.strip() else None), err


def reported(world, layouts: dict[str, str], placements: dict[str, Path] | None = None) -> dict:
    code, document, err = report_with(world, layouts, placements)
    assert code == 0, f"the report exited with {code}: {err}"
    return document


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word_shards(document: dict) -> list[str]:
    """The shards whose action is the fourth word, in the report's order."""
    return [e["artifact"] for e in document["artifacts"]
            if e["kind"] == v.KIND_SHARD and e["action"] == WORD]


def place_not_stated(document: dict) -> dict:
    """``counts.place_not_stated`` of the report: a failed assertion when it has none."""
    counts = document["counts"]
    assert "place_not_stated" in counts, f"counts has no place_not_stated: {sorted(counts)}"
    return counts["place_not_stated"]


def placements_of(document: dict) -> dict:
    """``groups.placements`` of the report: a failed assertion when it has none."""
    groups = document["groups"]
    assert "placements" in groups, f"groups has no placements: {sorted(groups)}"
    return groups["placements"]


def entry(document: dict, name: str) -> dict:
    found = [e for e in document["artifacts"] if e["artifact"] == name]
    assert len(found) == 1, f"{name}: {len(found)} entries among {[e['artifact'] for e in document['artifacts']]}"
    return found[0]


def names(document: dict) -> list[str]:
    return [e["artifact"] for e in document["artifacts"]]
