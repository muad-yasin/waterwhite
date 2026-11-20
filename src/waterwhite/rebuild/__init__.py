"""The rebuild report of a withdrawal (plan 4.1, 4.3; docs/interfaces.md section 7).

``build_report`` is the document of section 7.1: per affected artifact and per record
its action and its reason, the counts, the restart points, the checkpoints that
carry exposure and the checkpoints that answer "not covered". It is computed on demand from the store as it is, in one coverage
session, and nothing is written. The tool executes none of it: the team's own builder
does, and ``verify`` checks what the builder made (docs/rebuild.md).

The package reads answers and verdicts and builds none: every "keep" comes from the
proof, asked through ``coverage.answer``, ``coverage.is_clear`` and
``Answer.is_negative``.

- ``records.py``: the action of one record;
- ``artifacts.py``: the action of one artifact;
- ``upstream.py``: the files upstream of the artifacts, which ``artifacts`` does not
  hold, with their answer, action and reason;
- ``counts.py``: the counts, the mixtures that need a successor and the groups;
- ``placing.py``: which shard already holds a record that returns to a release, where a
  stated placement lands it, and the fourth word of an artifact where no place is stated.

The restart points are the part of ``restart_free.py`` (the exposure-free restart) and
``restart_exact.py`` (the exact-rerun restart each entry ends with); this module reaches
them at the one call of ``build_report``, which builds ``checkpoints_not_covered`` (the
checkpoints that answer "not covered") beside ``exposed_checkpoints``.
"""

from __future__ import annotations

from collections.abc import Mapping

from .. import coverage
from .. import vocabulary as v
from ..requests import Withdrawal
from . import artifacts as acting
from . import counts as counting
from . import placing, upstream

__all__ = ["build_report", "report_without_restarts"]


def report_without_restarts(store, request: Withdrawal, session, *,
                            layouts: Mapping[str, str] | None = None,
                            placements: Mapping[str, placing.Placement] | None = None,
                            integrity=None, withheld: list | None = None) -> dict:
    """Every key of the report but ``restarts``, ``exposed_checkpoints`` and
    ``checkpoints_not_covered``. It reads
    through the ``coverage.Session`` it is given and ends with "store written during
    the query" if the store was written after the session began. ``integrity`` is the
    pass of ``coverage.check_store`` when the caller has run the loud checks.
    ``placements`` is a layout's stated placement of records, by the layout's name
    (``placing.read_placement``); each must be for a layout that ``layouts`` names."""
    held_layouts = counting.check_layouts(layouts)
    held_placements = placing.check_placements(placements, held_layouts)
    ordered = [a for a in store.artifacts() if acting.wanted(a)]
    by_instance = {a.instance_id: a for a in ordered}

    held_back: list = [] if withheld is None else withheld

    def answered(artifact):
        """The answer, or None when it rests on an archived shard no check has passed (it is
        then in ``withheld`` and the artifact's entry is unresolved with nothing judged)."""
        if coverage.shard_withheld(store, artifact, held_back):
            return None
        return coverage.answer_or_withheld(store, artifact.instance_id, request, sink=held_back,
                                           integrity=integrity, session=session)

    # The shards first: a mixture takes its action from the shards of its snapshot. The
    # records that return to a release are collected as the entries are made.
    returning = placing.Returning(session)
    shards: dict[str, dict] = {}
    for artifact in ordered:
        if artifact.kind == v.KIND_SHARD:
            said = answered(artifact)
            shards[artifact.instance_id] = (
                acting.withheld_entry(store, artifact) if said is None
                else acting.shard_entry(store, session, artifact, said, returning))

    entries: list[dict] = []
    listing: list[dict] = []
    for artifact in ordered:
        if artifact.kind == v.KIND_JSONL:
            said = answered(artifact)
            if said is None:
                entry, records = acting.withheld_entry(store, artifact), []
            else:
                entry, records = acting.jsonl_entry(store, session, artifact, said, returning)
            listing.extend(records)
        elif artifact.kind == v.KIND_SHARD:
            entry = shards[artifact.instance_id]
        else:
            entry = None  # made once the shards are placed
        entries.append(entry)

    # Where the returning records land, and which shard holds them already, decides what a
    # shard that nothing of the source stands behind reads; then the mixtures follow.
    names_of_shards = {i: list(by_instance[i].names or (i,)) for i in shards}
    placed = placing.apply(store, session, shards, names_of_shards, held_layouts,
                           held_placements, returning)
    for place, artifact in enumerate(ordered):
        if artifact.kind == v.KIND_MIXTURE:
            said = answered(artifact)
            entries[place] = (acting.withheld_entry(store, artifact) if said is None
                              else acting.mixture_entry(store, artifact, said, shards))

    counting.add_bytes(store, by_instance, entries)
    upstream_files = upstream.upstream_entries(store, request, session, integrity=integrity)
    document = {
        "form": v.FORM_REPORT,
        "request": request.to_json(),
        "records": listing,
        "artifacts": entries,
        "upstream_files": upstream_files,
        "counts": counting.totals(entries),
        "mixtures": counting.mixtures(entries),
        "groups": {
            "by_run": counting.by_run(store, shards),
            "by_layout": counting.by_layout(names_of_shards, shards, held_layouts),
            "layouts": counting.layout_statement(names_of_shards, shards, held_layouts),
            "placements": placed,
        },
    }
    if withheld is None and held_back:
        document["withheld"] = [w.to_json() for w in held_back]
    session.ensure_current()
    return document


def build_report(store, request: Withdrawal, *,
                 layouts: Mapping[str, str] | None = None,
                 placements: Mapping[str, placing.Placement] | None = None) -> dict:
    """The document of section 7.1 in full, ending with ``exposed_checkpoints`` and
    ``checkpoints_not_covered``. The loud checks of the store run first
    over the whole store; one fault ends the report."""
    integrity = coverage.check_store(store)
    session = coverage.Session(store, request)
    withheld: list = []
    document = report_without_restarts(
        store, request, session, layouts=layouts, placements=placements, integrity=integrity,
        withheld=withheld)
    session.ensure_current()
    from . import restart_exact, restart_free  # the parts that hold the restart points

    entries = {entry["run"]: entry
               for entry in restart_free.exposure_free_restarts(store, request, session, withheld)}
    exact = restart_exact.exact_rerun_of_report(store, document["artifacts"])
    restarts = []
    for run in store.runs():
        # A run with a stale segment and no registered checkpoint has no exposure-free
        # restart to prove; it still has the entry that says so and its exact rerun.
        entry = entries.get(run)
        if entry is None and run in exact:
            entry = restart_free.unproven_entry(store, run)
        if entry is not None:
            restarts.append(entry | {"exact_rerun": exact.get(run)})
    document["restarts"] = restarts
    document["exposed_checkpoints"] = restart_free.exposed_checkpoints(
        store, request, session, withheld)
    document["checkpoints_not_covered"] = restart_free.checkpoints_not_covered(
        store, request, session, withheld)
    if withheld:
        document["withheld"] = [w.to_json() for w in withheld]
    session.ensure_current()
    return document
