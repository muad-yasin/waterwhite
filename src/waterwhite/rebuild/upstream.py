"""The files upstream of the artifacts of the rebuild report (plan 4.1, BF-6;
docs/interfaces.md section 7.1, docs/rebuild.md section 2.2a).

``artifacts`` holds the releases and imports, the shards and the mixtures. Every other
registered file of kind jsonl is a source file, a generation output or a file between
a source and a release or a shard. Those whose answer to the request is not the
negative answer are named here, in the order of the store's registration, with their
answer, an action where the repair has one, and the reason. The list is outside
``counts`` and ``bytes_to_rewrite``, and nothing reads it but a person: the replica and
``verify`` take no key of it.

The action of an entry:

- a source revision named by the request, and a generation output the repaired side
  reuses as the same instance: no action (null). The repair never edits a source
  revision, and the records of a generation output that the request reaches leave the
  release that copies them, so the output itself is left as it is;
- an intermediate file ahead of a release or a shard whose answer is "contains":
  **rewrite**, the repair writes it again;
- a file whose answer is "may contain" or "not covered", whatever its role:
  **unresolved**; a "may contain" carries the answer's own ``names`` and ``failing``.

A generation output is a file whose records carry a generation edge in the store: the
store records which step made a record, and a record of a generation step is the one
whose edge says so. This module reads answers and builds none.
"""

from __future__ import annotations

from .. import coverage
from .. import vocabulary as v
from . import artifacts as acting

REASON_REVISION_WITHDRAWN = "the revision the request withdraws"
REASON_OUTPUT_REUSED = (
    "the repaired build reuses this file as the same instance; the records of it that "
    "the request reaches leave the release that copies them (retired there), and the "
    "file itself is not rebuilt")
REASON_REBUILT = (
    "an intermediate file the repair rebuilds: the records that reach the withdrawn "
    "content leave it")
REASON_SOURCE_NOT_NAMED = (
    "a source file the request does not name holds records the request reaches; "
    "the repair rebuilds no source file")


def narrowed_reason(count: int) -> str:
    """The reason of a revision a request narrowed to records names."""
    return f"a revision the request names: {count} of its records are withdrawn, the rest stay"


def _named_files(store, request) -> set[str]:
    """The instance ids of the files that hold a part of a member the request names."""
    held: set[str] = set()
    for name in request.members:
        member = store.member(name)
        if member is not None:
            held.update(instance for instance, _locator in member.parts)
    return held


def _made_by_generation(store, instance_id: str) -> bool:
    return any(edge.is_generation for edge in store.edges(instance_id))


def _entry(artifact, answer) -> dict:
    return {
        "artifact": acting.shown(artifact),
        "instance": artifact.instance_id,
        "kind": artifact.kind,
        "role": artifact.role,
        "answer": answer.answer,
        "action": None,
        "reason": None,
    }


def _unresolved(entry: dict, answer) -> dict:
    entry["action"] = v.UNRESOLVED
    if answer.answer == v.MAY_CONTAIN:
        entry["reason"] = (f"nothing shows whether the request reaches this file: "
                           f"the proof stops at {answer.names}")
        entry["names"] = answer.names
        entry["failing"] = None if answer.failing is None else answer.failing.to_json()
    else:
        entry["reason"] = (f"the file is not covered ({answer.reason}); "
                           "nothing shows whether the request reaches it")
    return entry


def upstream_entries(store, request, session, *, integrity=None) -> list[dict]:
    """The ``upstream_files`` of the report: one entry for every registered file of
    kind jsonl that ``artifacts`` does not hold and whose answer is not the negative
    one, in registration order."""
    named = _named_files(store, request)
    listed: list[dict] = []
    for artifact in store.artifacts():
        if artifact.kind != v.KIND_JSONL or acting.wanted(artifact):
            continue
        answer = coverage.answer(store, artifact.instance_id, request,
                                 integrity=integrity, session=session)
        if answer.is_negative:
            continue
        entry = _entry(artifact, answer)
        if answer.answer != v.CONTAINS:
            listed.append(_unresolved(entry, answer))
        elif artifact.role == v.ROLE_SOURCE:
            if artifact.instance_id in named:
                withdrawn = sum(1 for ref in request.records
                                if ref.instance_id == artifact.instance_id)
                entry["reason"] = (narrowed_reason(withdrawn) if request.record_level
                                   else REASON_REVISION_WITHDRAWN)
            else:
                entry["action"] = v.UNRESOLVED
                entry["reason"] = REASON_SOURCE_NOT_NAMED
            listed.append(entry)
        elif _made_by_generation(store, artifact.instance_id):
            entry["reason"] = REASON_OUTPUT_REUSED
            listed.append(entry)
        else:
            entry["action"] = v.ACTION_REWRITE
            entry["reason"] = REASON_REBUILT
            listed.append(entry)
    return listed
