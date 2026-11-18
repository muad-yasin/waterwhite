"""The action of each artifact of the rebuild report (plan 4.1, 4.5; docs/interfaces.md
section 7.1, docs/rebuild.md section 2.2).

An artifact is judged by its own answer to the request, which ``coverage.answer``
gives, and by what its records are judged to be:

- the negative answer: **keep**. This is the only road to keep; "no chain was found"
  is not enough, and the answer is the proof's;
- "contains": **rewrite** when some record or document has a definite action (retire,
  re-source), **unresolved** when everything in scope is unresolved;
- "may contain" or "not covered": **unresolved**.

A mixture needs a successor exactly when its snapshot names a shard that is
rewritten. This module reads answers and verdicts and builds none.

The negative answer is the proof's, and a shard's "keep" is then only its own answer: where
a record that returns to a release may land in the shard, ``placing.py`` changes the entry
after it is built (a shard that receives a record is rewritten; a shard in a group whose
copies have no stated place reads the fourth word).
"""

from __future__ import annotations

from .. import vocabulary as v
from . import records as judging

RECORD_ACTIONS = (v.ACTION_KEEP, v.ACTION_RE_SOURCE, v.ACTION_PROMOTE, v.ACTION_RETIRE, v.UNRESOLVED)


def shown(artifact) -> str:
    """The name an artifact goes by in the report."""
    return artifact.names[0] if artifact.names else artifact.instance_id


def successor_name(name: str) -> str:
    return v.SUCCESSOR_PREFIX + name


def wanted(artifact) -> bool:
    """Which artifacts have an entry: a release or an import, a shard, a mixture.
    Source files, generation outputs and the files between them have none."""
    if artifact.kind == v.KIND_JSONL:
        return artifact.role in (v.ROLE_RELEASE, v.ROLE_IMPORT)
    return artifact.kind in (v.KIND_SHARD, v.KIND_MIXTURE)


def _entry(artifact, answer) -> dict:
    """An entry with every key the form has; the kinds fill their own."""
    return {
        "artifact": shown(artifact),
        "instance": artifact.instance_id,
        "kind": artifact.kind,
        "role": artifact.role,
        "action": None,
        "answer": answer.answer,
        "chain": [hop.to_json() for hop in answer.chain],
        "successor": None,
        "records": None,
        "counts": None,
        "records_removed": None,
        "removed": None,
        "rows_in_successor": None,
        "documents": None,
        "documents_in_successor": None,
        "receives": None,
        "rewritten_shards": None,
        "needs_successor": None,
        "bytes_to_rewrite": None,
    }


def _record_entry(artifact: str, record: str, judgement) -> dict:
    return {
        "artifact": artifact,
        "record": record,
        "action": judgement.action,
        "chain": list(judgement.chain),
        "new_parent": judgement.new_parent,
        "withdrawn_parent": judgement.withdrawn_parent,
        "regenerate_with": judgement.regenerate_with,
        "step": judgement.step,
        "missing_link": judgement.missing_link,
        "failing": judgement.failing,
        "names": judgement.names,
        "review_scope": judgement.review_scope,
    }


def _tally(actions: list[str]) -> dict[str, int]:
    return {word: actions.count(word) for word in RECORD_ACTIONS}


def _artifact_action(answer, definite: bool) -> str:
    """The action of an artifact from its answer and whether any of what it holds
    has a definite action."""
    if answer.is_negative:
        return v.ACTION_KEEP
    if answer.answer == v.CONTAINS and definite:
        return v.ACTION_REWRITE
    return v.UNRESOLVED


# --- A JSONL file ---------------------------------------------------------------------


def jsonl_entry(store, session, artifact, answer, returning=None) -> tuple[dict, list[dict]]:
    """(the artifact's entry, its record entries). A kept file lists no record.
    ``returning`` (``placing.Returning``) is told the records the file promotes and the
    ones it is re-sourced onto."""
    entry = _entry(artifact, answer)
    name = entry["artifact"]
    total = store.occurrence_count(artifact.instance_id)
    entry["records"] = total
    if answer.is_negative:
        entry["action"] = v.ACTION_KEEP
        entry["counts"] = _tally([v.ACTION_KEEP] * total)
        entry["records_removed"] = 0
        entry["removed"] = []
        return entry, []

    judged = []
    for occurrence in store.occurrences(artifact.instance_id):
        status = session.occurrence(occurrence.ref)
        judged.append((occurrence.ref, judging.judge(store, session, status, occurrence.ref)))
    promoted = judging.promoted(store, session, artifact.instance_id, judged)
    if returning is not None:
        for _ref, judgement in judged:
            if judgement.action == v.ACTION_RE_SOURCE and judgement.new_ref is not None:
                returning.add(judgement.new_ref)
        for item in promoted:
            returning.add(item["ref"])

    listing = [_record_entry(name, judging.key(session, ref), j) for ref, j in judged]
    for item in promoted:
        promote = judging.Judgement(v.ACTION_PROMOTE, chain=item["chain"])
        listing.append(_record_entry(name, judging.key(session, item["ref"]), promote))

    actions = [j.action for _ref, j in judged] + [v.ACTION_PROMOTE] * len(promoted)
    counts = _tally(actions)
    entry["counts"] = counts
    definite = counts[v.ACTION_RETIRE] + counts[v.ACTION_RE_SOURCE] > 0
    entry["action"] = _artifact_action(answer, definite)
    entry["removed"] = [judging.key(session, ref) for ref, j in judged if j.action == v.ACTION_RETIRE]
    entry["records_removed"] = len(entry["removed"])
    entry["rows_in_successor"] = (
        counts[v.ACTION_KEEP] + counts[v.ACTION_RE_SOURCE] + counts[v.UNRESOLVED])
    if entry["action"] == v.ACTION_REWRITE:
        entry["successor"] = successor_name(name)
    return entry, listing


# --- A shard --------------------------------------------------------------------------


def shard_entry(store, session, artifact, answer, returning=None) -> dict:
    """The entry of a shard. Its documents are judged through the map a reader gets,
    and not listed as records: ``removed`` names the documents that leave, and
    ``receives`` the records the shard takes in a re-source. ``returning``
    (``placing.Returning``) is told the records its own documents are re-sourced onto."""
    entry = _entry(artifact, answer)
    name = entry["artifact"]
    binding = store.binding(artifact.instance_id)
    entry["documents"] = None if binding is None else binding.declared_count
    entry["removed"] = []
    entry["receives"] = []
    if answer.is_negative:
        entry["action"] = v.ACTION_KEEP
        entry["records_removed"] = 0
        entry["documents_in_successor"] = entry["documents"]
        return entry

    gate = session.shard_gate(artifact.instance_id)
    definite = False
    if gate.missing is None:
        for row in store.map_rows(artifact.instance_id):
            judgement = judging.judge_document(store, session, gate, row)
            if judgement.action == v.ACTION_KEEP:
                continue
            if judgement.action == v.ACTION_RETIRE:
                entry["removed"].append(f"D{row.ordinal}")
                definite = True
            elif judgement.action == v.ACTION_RE_SOURCE:
                definite = True
                if judgement.new_parent not in entry["receives"]:
                    entry["receives"].append(judgement.new_parent)
                if returning is not None and judgement.new_ref is not None:
                    returning.add(judgement.new_ref)
    entry["records_removed"] = len(entry["removed"])
    if entry["documents"] is not None:
        entry["documents_in_successor"] = entry["documents"] - entry["records_removed"]
    entry["action"] = _artifact_action(answer, definite)
    if entry["action"] == v.ACTION_REWRITE:
        entry["successor"] = successor_name(name)
    return entry


# --- A mixture ------------------------------------------------------------------------


def mixture_entry(store, artifact, answer, shard_entries: dict[str, dict]) -> dict:
    """The entry of a mixture, from the actions of the shards of its snapshot."""
    entry = _entry(artifact, answer)
    rewritten = []
    members = [shard_entries.get(m.shard_instance_id)
               for m in store.mixture_shards(artifact.instance_id)]
    for held in members:
        if held is not None and held["action"] == v.ACTION_REWRITE:
            rewritten.append(held["artifact"])
    entry["rewritten_shards"] = rewritten
    entry["needs_successor"] = bool(rewritten)
    if rewritten:
        entry["action"] = v.ACTION_REWRITE
        entry["successor"] = successor_name(entry["artifact"])
    elif answer.is_negative:
        # Kept, unless a shard of the snapshot may receive a copy whose place is not stated.
        unstated = [held for held in members if held is not None
                    and held["action"] == v.ACTION_CLEAN_PLACE_NOT_STATED]
        if unstated:
            entry["action"] = v.ACTION_CLEAN_PLACE_NOT_STATED
            entry["returning"] = list(dict.fromkeys(
                key for held in unstated for key in held["returning"]))
            entry["place"] = v.PLACE_NOT_STATED
        else:
            entry["action"] = v.ACTION_KEEP
    else:
        entry["action"] = v.UNRESOLVED
    return entry
