"""Capture: the thin wrappers a team's pipeline calls at its capture points (plan
3.3, 5.4, Decision 7).

Three functions read a JSONL file, compute its digest, its line count and one
fingerprint per line, and hand the records to the store:

- ``ingest_jsonl``: a file of a source revision (plan 3.3 "Ingest");
- ``record_jsonl``: a file a step wrote, with the edges and drops the step stated;
- ``import_jsonl``: a text-only import with no recorded producing step (plan L1).

The small builders below them make the records a pipeline states (a step, a copy
edge, a selection edge with its drops, a generation edge from a harness result).

A pipeline imports four things of the tool and no more: this module,
``waterwhite.harness`` (to run a generation job), ``waterwhite.normalise`` (when
its sanitizer is a named normaliser) and ``waterwhite.store.Store`` (to open the
store it writes to). Everything else it needs is named here: the record types it
builds, the id functions it calls and the words of the vocabulary it writes. The
names are the same objects as in ``records``, ``ids`` and ``vocabulary``.

Capture decides nothing about answers and keeps no state. The store is the one
writer and the one judge of a record (plan 3.4); capture checks only what makes its
own output well formed. No record text reaches the store: capture passes locators,
fingerprints and links (plan 3.1).

While the generating-model extension is off (plan 1.4): the model a job called is a field
of its harness log and nothing here treats it as an input or an ancestor. With the extension
on, the proof reads that field (plan 2.2(h)); capture writes it the same way in both states.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import ids, normalise
from . import vocabulary as v
from .errors import InvalidRecord
from .ids import file_digest, line_locator, ordinal_locator, param_digest, shard_digest
from .records import (
    OWN_ARTIFACT,
    DropRecord,
    EdgeInput,
    EdgeRecord,
    HarnessLog,
    MapRecord,
    MapRow,
    OccurrenceRecord,
    OccurrenceRef,
    SnapshotShard,
    StepRef,
)
from .vocabulary import (
    CANDIDATES,
    DERIVED_FILE,
    DERIVED_POSITION,
    DERIVED_STATED,
    DERIVED_TOKEN_MATCHING,
    DROP_DEDUP_LOSER,
    DROP_EMPTY_LINE,
    DROP_FILTER,
    DROP_NO_TOKENS,
    DROP_REASONS,
    DROP_SANITIZER_SKIP,
    FILE_LEVEL,
    MODE_ALL,
    MODE_ANY,
    NAMESPACE_CAPTURE,
    NAMESPACE_RECOVERY,
    OBSERVED,
    PART_FILE,
    PART_FOLDER,
    PART_RECORDS,
    PART_REVISION,
    RECOVERED,
    ROLE_IMPORT,
    ROLE_INTERMEDIATE,
    ROLE_RELEASE,
    ROLE_SOURCE,
    UNKNOWN,
    VERIFIED_FAILED,
    VERIFIED_NOT_RUN,
    VERIFIED_PASSED,
)

if TYPE_CHECKING:  # the harness is not needed to import this module
    from .harness import JobOutput, JobResult

__all__ = [
    # the capture points and what they read
    "FileScan",
    "import_jsonl",
    "ingest_jsonl",
    "line_text",
    "record_jsonl",
    "scan_jsonl",
    "store_normaliser",
    "normalise_text",
    # builders of what a step states
    "copy_edge",
    "drop",
    "generation_edge",
    "generation_edges",
    "occurrence",
    "selection_edge",
    "step_ref",
    # record types a pipeline builds or hands on (waterwhite.records)
    "OWN_ARTIFACT",
    "DropRecord",
    "EdgeInput",
    "EdgeRecord",
    "HarnessLog",
    "MapRecord",
    "MapRow",
    "OccurrenceRecord",
    "OccurrenceRef",
    "SnapshotShard",
    "StepRef",
    # id functions (waterwhite.ids)
    "file_digest",
    "line_locator",
    "ordinal_locator",
    "param_digest",
    "shard_digest",
    # words (waterwhite.vocabulary)
    "CANDIDATES",
    "DERIVED_FILE",
    "DERIVED_POSITION",
    "DERIVED_STATED",
    "DERIVED_TOKEN_MATCHING",
    "DROP_DEDUP_LOSER",
    "DROP_EMPTY_LINE",
    "DROP_FILTER",
    "DROP_NO_TOKENS",
    "DROP_REASONS",
    "DROP_SANITIZER_SKIP",
    "FILE_LEVEL",
    "MODE_ALL",
    "MODE_ANY",
    "NAMESPACE_CAPTURE",
    "NAMESPACE_RECOVERY",
    "OBSERVED",
    "PART_FILE",
    "PART_FOLDER",
    "PART_KIND_FILE",
    "PART_RECORDS",
    "PART_REVISION",
    "RECOVERED",
    "ROLE_IMPORT",
    "ROLE_INTERMEDIATE",
    "ROLE_RELEASE",
    "ROLE_SOURCE",
    "UNKNOWN",
    "VERIFIED_FAILED",
    "VERIFIED_NOT_RUN",
    "VERIFIED_PASSED",
]

# The part kind of a member that is one file (plan 2.3). The older name of
# ``PART_FILE``, kept for code written against it.
PART_KIND_FILE = PART_FILE

PathLike = str | os.PathLike


# --- Reading a JSONL file ------------------------------------------------------------


@dataclass(frozen=True)
class FileScan:
    """What one read of a JSONL file gives: its line count, its SHA-256 and one
    occurrence per line, all taken from the same bytes."""

    count: int
    digest: str
    occurrences: tuple[OccurrenceRecord, ...]


def line_text(raw: bytes, text_field: str = "text") -> str | None:
    """The text a line holds, or None when it holds none: the line is not UTF-8, is
    not parseable JSON, is not a JSON object, has no ``text_field`` or holds
    something other than text there. ``raw`` is the line without its line feed.

    This is the one definition of "the text a line holds" in the tool. The
    capture wrappers fingerprint it, the recovery of a copy step and of a
    sanitizer compares it, and a tokenization map tokenizes it again
    (``waterwhite.recovery``, ``waterwhite.tokenmap``)."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    if not isinstance(value, dict):
        return None
    text = value.get(text_field)
    return text if isinstance(text, str) else None


def scan_jsonl(path: PathLike, *, normaliser: str, text_field: str = "text") -> FileScan:
    """Read a JSONL file once and return its count, digest and occurrences.

    What a line is, is defined once, in ``ids.split_lines``, and the store counts
    lines with the same function: a line ends at a line feed byte, and a last
    piece with no line feed after it is a line when it is not empty. Line n,
    counted from 1, has the locator ``L<n>``. Every line is an occurrence (plan
    2.3). A line that holds a text which is not empty after the normaliser carries
    that text's fingerprint; every other line (blank, unparseable, no text field,
    empty after normalisation) carries none. The digest is the file's content
    digest (``ids.file_digest``), taken over the very bytes the lines are split
    from, so the count, the digest and the occurrences cannot disagree."""
    function = normalise.get(normaliser)
    hasher = ids.ContentHasher()
    occurrences: list[OccurrenceRecord] = []
    number = 0
    for raw in ids.read_lines(path, hasher):
        number += 1
        text = line_text(raw, text_field)
        fingerprint = None
        if text is not None:
            normalised = function(text)
            if normalised:
                try:
                    fingerprint = ids.fingerprint(normalised)
                except UnicodeEncodeError:
                    raise InvalidRecord(
                        f"{os.fspath(path)}, line {number}: the text holds a surrogate "
                        f"code point with no partner and has no UTF-8 form, so it has "
                        f"no fingerprint; the file is not registered",
                        line=number,
                    ) from None
        occurrences.append(OccurrenceRecord(ids.line_locator(number), fingerprint))
    return FileScan(count=number, digest=hasher.hexdigest(), occurrences=tuple(occurrences))


def store_normaliser(store: object) -> str:
    """The name of the normaliser a store's fingerprints are taken after."""
    name = getattr(store, "normaliser", None)
    if callable(name):
        name = name()
    if not isinstance(name, str):
        raise InvalidRecord("the store does not name its normaliser")
    return name


def normalise_text(normaliser: str, text: str) -> str:
    """A text after the named normaliser. A pipeline's sanitizer step calls this
    with the store's normaliser, so the record it writes and the fingerprint of the
    record it read agree (plan 3.1, F8)."""
    return normalise.apply(normaliser, text)


def _need_text(what: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise InvalidRecord(f"{what} is missing")


# --- The three capture points ----------------------------------------------------------


def ingest_jsonl(
    store,
    path: PathLike,
    *,
    source: str,
    member: str,
    revision: str,
    name: str | None = None,
    text_field: str = "text",
) -> str:
    """Observed ingest of one file of a source revision (plan 3.3 "Ingest").

    Pins the file by its digest, registers it with one occurrence per line under the
    artifact name ``name`` (the member's name when none is given), registers the
    source and the member (one part: the whole file) and returns the instance id,
    which is ``ids.ingest_instance_id`` of the content digest and the member name.
    A member name that is taken by another source, revision or part kind is
    refused before anything is written."""
    _need_text("the source name", source)
    _need_text("the member name", member)
    _need_text("the revision", revision)
    artifact_name = member if name is None else name
    _need_text("the artifact name", artifact_name)
    normaliser = store_normaliser(store)
    # An ingest is three writes: the file, the source, the member. The store
    # refuses the third when the member name is taken by another source, revision
    # or part kind. That is read here, before the first write, so that a refused
    # ingest leaves no file behind that no member covers.
    held = store.member(member)
    if held is not None and (held.source, held.revision, held.part_kind) != (source, revision, PART_FILE):
        raise InvalidRecord(
            f"the member {member!r} is registered already, under the source {held.source!r} "
            f"at revision {held.revision!r} as a {held.part_kind}; an ingest under "
            f"{source!r} at {revision!r} needs another member name",
            member=member,
        )
    scan = scan_jsonl(path, normaliser=normaliser, text_field=text_field)
    instance = store.register_jsonl(
        path,
        name=artifact_name,
        role=v.ROLE_SOURCE,
        step=None,
        member_name=member,
        inputs=(),
        occurrences=scan.occurrences,
        edges=(),
        drops=(),
        declared_count=scan.count,
        declared_digest=scan.digest,
    )
    _same_ingest_id(instance, scan.digest, member)
    store.add_source(source)
    store.add_member(source, member, revision, PART_FILE, ((instance, ""),))
    return instance


def record_jsonl(
    store,
    path: PathLike,
    *,
    name: str,
    role: str,
    step: StepRef,
    inputs: Sequence[str],
    edges: Sequence[EdgeRecord],
    drops: Sequence[DropRecord] = (),
    text_field: str = "text",
) -> str:
    """Register a file a step wrote, with the edges and drops the step stated.

    The count and the digest are taken first, in the one read that also gives the
    occurrences, and are declared to the store, which reads the file itself and
    refuses the write when the file no longer matches. ``inputs`` are the instance
    ids of the step's input artifacts, in the step's order. Returns the instance id
    the store computed (plan 3.1, Decision 29)."""
    if role not in (v.ROLE_INTERMEDIATE, v.ROLE_RELEASE):
        raise InvalidRecord(
            f"a file a step wrote has the role {v.ROLE_INTERMEDIATE!r} or "
            f"{v.ROLE_RELEASE!r}, not {role!r}"
        )
    if not isinstance(step, StepRef):
        raise InvalidRecord("a recorded file names the step that wrote it")
    _need_text("the artifact name", name)
    if isinstance(inputs, str):
        raise InvalidRecord("the inputs are a sequence of instance ids, not one text")
    inputs = tuple(inputs)
    for item in inputs:
        _need_text("an input's instance id", item)
    scan = scan_jsonl(path, normaliser=store_normaliser(store), text_field=text_field)
    edges, drops = tuple(edges), tuple(drops)
    _check_edges(edges, scan.count)
    _check_drops(drops)
    return store.register_jsonl(
        path,
        name=name,
        role=role,
        step=step,
        member_name=None,
        inputs=inputs,
        occurrences=scan.occurrences,
        edges=edges,
        drops=drops,
        declared_count=scan.count,
        declared_digest=scan.digest,
    )


def import_jsonl(
    store,
    path: PathLike,
    *,
    name: str,
    member_name: str,
    file_family: tuple[str, str] | None = None,
    text_field: str = "text",
) -> str:
    """Register a text-only import with no recorded producing step (plan L1).

    The instance comes from the content digest and ``member_name``, so the same
    bytes imported under two member names are two instances (plan Decision 29).
    ``file_family`` is ``(member, evidence)``: the registered member the whole file
    is known to derive from, and what that knowledge rests on. With it the file gets
    one file-level link to that member, stored with the derivation "file"; a family
    whose member is not registered is refused before anything is written. Without
    it the import's records have no recorded origin. No record-level link is written
    either way: which record came from which item stays unknown (plan L1, L5)."""
    _need_text("the artifact name", name)
    _need_text("the member name", member_name)
    if file_family is not None:
        if not isinstance(file_family, tuple) or len(file_family) != 2:
            raise InvalidRecord("a file family is (member, evidence)")
        _need_text("the file family's member", file_family[0])
        _need_text("the file family's evidence", file_family[1])
    normaliser = store_normaliser(store)
    if file_family is not None:
        # Read before the file is registered: the link is a second write, and a
        # family that names no registered member would leave the import behind
        # without it.
        if store.member(file_family[0]) is None:
            raise InvalidRecord(
                f"the file family names the member {file_family[0]!r}, which is not "
                "registered; nothing is imported",
                member=file_family[0],
            )
    scan = scan_jsonl(path, normaliser=normaliser, text_field=text_field)
    instance = store.register_jsonl(
        path,
        name=name,
        role=v.ROLE_IMPORT,
        step=None,
        member_name=member_name,
        inputs=(),
        occurrences=scan.occurrences,
        edges=(),
        drops=(),
        declared_count=scan.count,
        declared_digest=scan.digest,
    )
    _same_ingest_id(instance, scan.digest, member_name)
    if file_family is not None:
        store.add_source_link(
            instance, None, file_family[0], v.FILE_LEVEL, file_family[1],
            derivation=v.DERIVED_FILE, verification=None,
        )
    return instance


def _same_ingest_id(instance: object, digest: str, member_name: str) -> None:
    expected = ids.ingest_instance_id(digest, member_name)
    if instance != expected:
        raise InvalidRecord(
            f"the store returned the instance id {instance!r}; a file with no producing "
            f"step has the id of its content digest and member name, {expected!r}",
            member=member_name,
        )


def _check_edges(edges: tuple[EdgeRecord, ...], count: int) -> None:
    children: set[str] = set()
    for edge in edges:
        if not isinstance(edge, EdgeRecord):
            raise InvalidRecord(f"not an edge record: {edge!r}")
        try:
            kind, number = ids.parse_locator(edge.child_locator)
        except (ValueError, TypeError):
            kind, number = "", 0
        if kind != "L" or not 1 <= number <= count or edge.child_locator != f"L{number}":
            raise InvalidRecord(
                f"an edge names the child {edge.child_locator!r}; the file has the "
                f"lines L1 to L{count}",
                child=str(edge.child_locator),
            )
        if edge.child_locator in children:
            raise InvalidRecord(
                f"two edges name the child {edge.child_locator}", child=edge.child_locator
            )
        children.add(edge.child_locator)
        if edge.derivation != v.DERIVED_STATED:
            raise InvalidRecord(
                f"the edge of {edge.child_locator} is derived by {edge.derivation!r}; a "
                f"step's file takes the edges the step stated, and a link derived "
                f"afterwards is added to the store as a recovered edge",
                child=edge.child_locator,
            )


def _check_drops(drops: tuple[DropRecord, ...]) -> None:
    for item in drops:
        if not isinstance(item, DropRecord) or item.reason not in v.DROP_REASONS:
            raise InvalidRecord(f"not a drop with a known reason: {item!r}")


# --- Builders for the records a pipeline states ---------------------------------------


def step_ref(name: str, version: str, params: object = None) -> StepRef:
    """A step: its name, its version and the digest of its parameters (any JSON
    value: lists, objects, text, whole numbers, true, false, null)."""
    _need_text("the step name", name)
    _need_text("the step version", version)
    try:
        digest = ids.param_digest(params)
    except (TypeError, ValueError) as error:
        raise InvalidRecord(
            f"the parameters of step {name!r} are not a JSON value ({error})", step=name
        ) from None
    return StepRef(name=name, version=version, param_digest=digest)


def occurrence(instance_id: str, line: int) -> OccurrenceRef:
    """The record at line ``line`` (counted from 1) of a registered JSONL file."""
    _need_text("the instance id", instance_id)
    return OccurrenceRef(instance_id, _line(line))


def _line(line: int) -> str:
    if not isinstance(line, int) or isinstance(line, bool) or line < 1:
        raise InvalidRecord(f"a line number is a whole number from 1: {line!r}")
    return ids.line_locator(line)


def _ref(value: object) -> OccurrenceRef:
    if not isinstance(value, OccurrenceRef):
        raise InvalidRecord(f"not an occurrence reference: {value!r}")
    return value


def copy_edge(child_line: int, parent: OccurrenceRef, step: StepRef) -> EdgeRecord:
    """The stated edge of a copy step (shuffle, chunk, merge, sanitize): the record
    at ``child_line`` of the step's output was made from ``parent`` alone."""
    return EdgeRecord(
        child_locator=_line(child_line),
        inputs=(EdgeInput(_ref(parent)),),
        transform=step,
        parent_mode=v.MODE_ALL,
        derivation=v.DERIVED_STATED,
        label=v.OBSERVED,
    )


def selection_edge(
    child_line: int,
    group: Sequence[OccurrenceRef],
    kept: OccurrenceRef,
    step: StepRef,
    *,
    group_key: str | None = None,
) -> tuple[EdgeRecord, tuple[DropRecord, ...]]:
    """The stated edge and the drops of a selection (dedup, plan 3.3 "Selection").

    ``group`` is every equivalent copy the step saw, in the step's order, ``kept``
    the one it selected. The edge has parent mode "any", names every copy as a
    parent and marks the kept one as selected. Each other copy is a drop with the
    reason "dedup loser". ``group_key`` ties the drops of one group together; when
    the step gives none, it is the locator of the output line the group became."""
    copies = tuple(_ref(item) for item in group)
    if len(set(copies)) != len(copies):
        raise InvalidRecord("a selection group names a copy twice")
    if _ref(kept) not in copies:
        raise InvalidRecord(f"the kept copy {kept.key} is not in its group", kept=kept.key)
    if group_key is not None:
        _need_text("the group key", group_key)
    child = _line(child_line)
    key = child if group_key is None else group_key
    edge = EdgeRecord(
        child_locator=child,
        inputs=tuple(EdgeInput(item, used=True, selected=item == kept) for item in copies),
        transform=step,
        parent_mode=v.MODE_ANY,
        derivation=v.DERIVED_STATED,
        label=v.OBSERVED,
    )
    drops = tuple(
        DropRecord(input=item, reason=v.DROP_DEDUP_LOSER, step_name=step.name, group_key=key)
        for item in copies
        if item != kept
    )
    return edge, drops


def drop(input: OccurrenceRef, reason: str, step: StepRef) -> DropRecord:
    """A record a step removed, with one of the reasons of
    ``vocabulary.DROP_REASONS`` (plan 3.1 "Drop")."""
    if reason not in v.DROP_REASONS:
        raise InvalidRecord(f"unknown drop reason: {reason!r}")
    return DropRecord(input=_ref(input), reason=reason, step_name=step.name)


def generation_edge(
    child_line: int, output: "JobOutput", log: HarnessLog, step: StepRef
) -> EdgeRecord:
    """The stated edge of one output of a job run in the harness (plan 3.3
    "Generation", Decision 25).

    Every handed record is an input of the edge, in hand-off order. Which of them
    are parents:

    - a job handed exactly one record: that record is a parent of each output, even
      when the output states nothing (the sole input; DECISIONS.md D-21);
    - a job handed two or more: one the output states it used is a parent, and one
      it does not state is in scope only and is not a parent.

    The parent mode is "all", the edge is a generation edge and carries the job's
    harness log and, in a field of its own, the job's name as the log gives it
    (DECISIONS.md D-24), so that the edge still names its job when the log is
    gone. While the generating-model extension is off, the model the job called stays
    a field of that log and is not an input; capture writes it the same way with the
    extension on, and the proof then reads it (plan 2.2(h)).

    The edge also carries the texts the job was handed (``handed_texts``), copied
    from the log. The store compares each with the record its reference names
    when the edge is registered. A log that holds none (one read from its file,
    fetched from the store or built by hand) gives an edge with none, and that edge is registered without the
    comparison."""
    if not isinstance(log, HarnessLog):
        raise InvalidRecord("a generation edge carries its job's harness log")
    used = tuple(output.used)
    handed = set(log.handed)
    for ref in used:
        if ref not in handed:
            raise InvalidRecord(
                f"the output states it used {_ref(ref).key}, which job {log.job_name!r} "
                f"was not handed",
                job=log.job_name,
            )
    stated = set(used)
    sole = len(set(log.handed)) == 1
    return EdgeRecord(
        child_locator=_line(child_line),
        inputs=tuple(EdgeInput(ref, used=sole or ref in stated) for ref in log.handed),
        transform=step,
        parent_mode=v.MODE_ALL,
        derivation=v.DERIVED_STATED,
        label=v.OBSERVED,
        is_generation=True,
        harness_log=log,
        job_name=log.job_name,
        handed_texts=log.handed_texts,
    )


def generation_edges(
    result: "JobResult", child_lines: Sequence[int], step: StepRef
) -> tuple[EdgeRecord, ...]:
    """One generation edge per output of a harness result. ``child_lines`` gives,
    for each output in order, the line it was written to."""
    lines = tuple(child_lines)
    if len(lines) != len(result.outputs):
        raise InvalidRecord(
            f"job {result.log.job_name!r} gave {len(result.outputs)} outputs and "
            f"{len(lines)} lines are named",
            job=result.log.job_name,
        )
    return tuple(
        generation_edge(line, output, result.log, step)
        for line, output in zip(lines, result.outputs)
    )
