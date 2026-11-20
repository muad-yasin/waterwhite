"""The answers for the successors and for the history (plan 4.4; docs/interfaces.md
section 7.3 "successors" and "history", docs/rebuild.md section 5).

A successor is a registered artifact like any other, so the answer about it is
``coverage.answer`` of the artifact under the report's request, in one coverage
session. This module builds no answer and no verdict: it puts the question and
prints what came back, with two facts of its own: whether the successor holds a
record the report would count unresolved (by the report's own check), and whether the
file the repaired side names is the file the successor was registered as.
"""

from __future__ import annotations

from collections.abc import Mapping

from .. import coverage
from .. import vocabulary as v
from . import records as judging
from . import verify_sides


def _holds_unresolved(store, session, instance_id: str) -> bool:
    """Whether an artifact holds a record that is unresolved, by the judgement the report
    reads for its own records (``records.judge``, through ``records.is_unresolved``, and
    for a document also the document's own verdict): the verdicts of the
    artifact's own records, a document through the map a reader gets, and a mixture
    through any shard of its snapshot."""
    artifact = store.artifact(instance_id)
    if artifact is None:
        return False
    if artifact.kind == v.KIND_JSONL:
        return any(judging.is_unresolved(store, session, session.occurrence(held.ref), held.ref)
                   for held in store.occurrences(instance_id))
    if artifact.kind == v.KIND_SHARD:
        gate = session.shard_gate(instance_id)
        if gate.missing is not None:
            return False  # no map: nothing is a record of it, and the answer says so
        return any(judging.is_unresolved_document(store, session, gate, row)
                   for row in store.map_rows(instance_id))
    if artifact.kind == v.KIND_MIXTURE:
        return any(_holds_unresolved(store, session, entry.shard_instance_id)
                   for entry in store.mixture_shards(instance_id))
    return False


def _matches_sides(held, name: str, kind: str, sides: Mapping | None, base) -> bool | None:
    """Whether the file the repaired side names for the artifact has the digest the
    successor is registered with; null without a sides file, where the repaired side
    names no file for it, for an artifact with no file, and for a successor that is not the
    kind of artifact it replaces (``kind``, the report's): the file the side names is of the
    replaced artifact's kind, and it is not read as the files of another (docs/cli.md)."""
    named = None if sides is None else (sides.get("repaired") or {}).get(name)
    if named is None or held.kind != kind or held.kind not in (v.KIND_JSONL, v.KIND_SHARD):
        return None
    return verify_sides.file_digest_of(held.kind, verify_sides.side_path(base, named)) == held.content_digest


def check_successors(store, report: Mapping, request, sides: Mapping | None, base) -> tuple[list, list]:
    """(the ``successors`` entries, the ``history`` entries), one of each for every
    artifact the report rewrites, in the order of the report."""
    session = coverage.Session(store, request)
    successors, history = [], []
    for entry in verify_sides.report_artifacts(report):
        if entry["action"] != v.ACTION_REWRITE:
            continue
        name = entry["artifact"]
        successor_name = entry.get("successor") or f"{v.SUCCESSOR_PREFIX}{name}"
        held = store.resolve(successor_name)
        if held is None:
            successors.append({
                "artifact": name, "successor": successor_name, "kind": entry["kind"],
                "registered": False, "instance": None, "answer": None, "names": None,
                "holds_unresolved": None, "matches_sides": None, "answered": None,
            })
        else:
            answered = coverage.answer(store, successor_name, request, session=session)
            successors.append({
                "artifact": name, "successor": successor_name, "kind": held.kind,
                "registered": True, "instance": held.instance_id,
                "answer": answered.answer, "names": answered.names,
                "holds_unresolved": _holds_unresolved(store, session, held.instance_id),
                "matches_sides": _matches_sides(held, name, entry["kind"], sides, base),
                "answered": answered.to_json(),
            })
        historical = store.resolve(name)
        said = coverage.answer(store, name, request, session=session)
        history.append({
            "artifact": name,
            "instance": None if historical is None else historical.instance_id,
            "instance_unchanged": historical is not None and historical.instance_id == entry["instance"],
            "answer": said.answer,
        })
    return successors, history


__all__ = ["check_successors"]
