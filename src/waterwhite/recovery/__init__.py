"""Recovery: links for a pipeline whose steps were not observed (plan 3.3 "Copy
steps" and "Deterministic rewrite", 5.4, Decision 7).

A team that ran its pipeline without the capture wrappers still has the files the
steps left behind. This package registers such a file and derives, step by step,
which record of an earlier file each record of a later file was made from:

- ``register_found_file``: a file with no recorded producing step, registered
  with the role it has in the pipeline;
- ``recover_copy_step``: a step that passes records on unchanged (shuffle, chunk,
  merge), matched by fingerprint and confirmed by reading both texts;
- ``recover_sanitizer``: a deterministic rewrite, matched after the store's
  normaliser, with the skipped lines read from the sanitizer's own log as drops.

One call is one step and one later file, and returns a ``RecoveryResult``. The
caller names the files that can be the step's inputs. Nothing else of the store is
searched, and nothing is ever matched by similarity (plan L5).

A recovery of links (the edge recovery and the degraded one) writes through three
methods of the store and no other:
``add_recovered_edges``, ``add_drops`` and ``add_source_link``, each call one
transaction. None of them changes an instance id (DECISIONS.md D-05). Running the
same recovery again writes nothing and the result says so. With ``write=False``
a recovery writes nothing at all and returns what it found.

A line a sanitizer skipped is recorded once, as a drop whose output is the first
registered artifact that lacks the line: here, the file the sanitizer wrote.
``recover_sanitizer`` refuses a log whose lines the store already holds as drops
for a shard (``waterwhite.tokenmap`` recorded them from the same log).

docs/recovery.md describes the rules in full.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from .. import capture, normalise
from .. import vocabulary as v
from ..errors import DropAlreadyRecorded, InvalidRecord, UnknownNormaliser
from ..records import DropRecord, EdgeRecord, OccurrenceRef, StepRef
from ..store.views import Artifact
from ._files import RegisteredFile, resolve_jsonl, shown_name
from ._match import match_records
from ._result import (
    KIND_COPY_STEP,
    KIND_SANITIZER,
    OUTCOME_FILE_LEVEL,
    OUTCOME_RECORD_LEVEL,
    OUTCOME_UNKNOWN,
    OUTCOMES,
    RecoveryResult,
)
from ._sanitizer_log import SANITIZER_REASONS, LoggedSkip, read_sanitizer_log

__all__ = [
    "KIND_COPY_STEP",
    "KIND_SANITIZER",
    "OUTCOMES",
    "OUTCOME_FILE_LEVEL",
    "OUTCOME_RECORD_LEVEL",
    "OUTCOME_UNKNOWN",
    "SANITIZER_REASONS",
    "LoggedSkip",
    "RecoveryResult",
    "read_sanitizer_log",
    "recover_copy_step",
    "recover_sanitizer",
    "register_found_file",
    "sanitize_step",
]

# The roles a file found afterwards can have: its place in the pipeline, or a
# text-only import from outside it. A source file is ingested, not found.
FOUND_ROLES = (v.ROLE_INTERMEDIATE, v.ROLE_RELEASE, v.ROLE_IMPORT)

PathLike = str | os.PathLike


# --- Registering a file the pipeline left ---------------------------------------------


def register_found_file(
    store,
    path: PathLike,
    *,
    name: str,
    role: str = v.ROLE_INTERMEDIATE,
    text_field: str = "text",
) -> str:
    """Register a file found afterwards, with no recorded producing step, and
    return its instance id.

    ``role`` is the place the file has in the pipeline: "intermediate" (a chunk
    file, a sanitizer's input, a tokenizer's input) or "release". A listing
    names a found release among the releases it covers. "import" registers a
    text-only import from outside the pipeline, as ``capture.import_jsonl`` does
    with no file family. The store has no role word for "found afterwards": a
    found file is told from a captured one by having no producing step.

    The file is read once, as every captured file is: one occurrence per line
    with its fingerprint, the declared count and digest checked by the store
    against the file. Its instance id is taken from its content digest and
    ``name``, which is also the name it is registered under. No member and no
    source is created: until a recovery links them, the records of the file have
    no recorded origin, and an answer about the file says so (plan 2.2(a), L12).

    The same file registered again under the same name and role is the same
    instance and writes nothing."""
    if role not in FOUND_ROLES:
        raise InvalidRecord(
            f"a file found afterwards has the role {', '.join(repr(r) for r in FOUND_ROLES)}, "
            f"not {role!r}; a source file is ingested (capture.ingest_jsonl)"
        )
    if not isinstance(name, str) or not name:
        raise InvalidRecord("the artifact name is missing")
    scan = capture.scan_jsonl(
        path, normaliser=capture.store_normaliser(store), text_field=text_field
    )
    return store.register_jsonl(
        path,
        name=name,
        role=role,
        step=None,
        member_name=name,
        inputs=(),
        occurrences=scan.occurrences,
        edges=(),
        drops=(),
        declared_count=scan.count,
        declared_digest=scan.digest,
    )


def sanitize_step(version: str, params: object = None) -> StepRef:
    """The step a sanitizer is recovered under when the pipeline names no other:
    the name "sanitize" (``vocabulary.STEP_SANITIZE``), the version of the
    sanitizer and the digest of its parameters. It is the step name a map
    recovered through a sanitizer's log gives the same drops
    (``tokenmap.SanitizerRun``)."""
    return capture.step_ref(v.STEP_SANITIZE, version, params)


# --- Shared pieces ---------------------------------------------------------------------


def _need_step(step: object) -> StepRef:
    if not isinstance(step, StepRef) or not step.name:
        raise InvalidRecord(
            "a recovery names the step it recovers (capture.step_ref(name, version, params))"
        )
    return step


def _write_edges(store, instance_id: str, edges: Sequence[EdgeRecord]) -> int:
    """Hand the edges to the store in one call and return how many were new. The
    store passes over an edge it holds already and refuses, whole, a call with an
    edge for a record that has another edge."""
    if not edges:
        return 0
    held = {edge.child.locator for edge in store.edges(instance_id)}
    store.add_recovered_edges(instance_id, edges)
    return sum(1 for edge in edges if edge.child_locator not in held)


def _plan_drops(store, output_id: str, drops: Sequence[DropRecord]) -> int:
    """How many of the drops the store does not hold yet. A drop the store holds
    with another reason is refused here, before anything of the run is written."""
    if not drops:
        return 0
    held = {
        (drop.input, drop.step_name): (drop.reason, drop.group_key)
        for drop in store.drops_by_output(output_id)
    }
    new = 0
    for drop in drops:
        stored = held.get((drop.input, drop.step_name))
        if stored is None:
            new += 1
        elif stored != (drop.reason, drop.group_key):
            raise InvalidRecord(
                f"the log gives {drop.input.key} the reason {drop.reason!r}, and the store "
                f"holds a drop of this record by step {drop.step_name!r} for this output "
                f"with the reason {stored[0]!r}; a stored drop is not replaced",
                record=drop.input.key,
                step=drop.step_name,
            )
    return new


def _files_read(files: Sequence[RegisteredFile], log_path: PathLike | None = None) -> tuple[str, ...]:
    names = [file.name for file in files if file.was_read]
    if log_path is not None:
        names.append(os.fspath(log_path))
    return tuple(names)


# --- A copy step -----------------------------------------------------------------------


def recover_copy_step(
    store,
    later: str,
    earlier: Sequence[str],
    *,
    step: StepRef,
    text_field: str = "text",
    write: bool = True,
) -> RecoveryResult:
    """Recover the links of one copy step (shuffle, chunk, merge) for one file it
    wrote (plan 3.3 "Copy steps").

    ``later`` is the file the step wrote; ``earlier`` lists every file the step
    can have read, in the caller's order. Each is a registered JSONL artifact,
    named by a registered name or its instance id. ``step`` names the step; it
    becomes the transform of every edge written.

    For each record of the later file the candidates are the records of the
    earlier files with the same fingerprint, each confirmed by comparing the two
    normalised texts. One candidate: a "recovered" edge (derivation
    "fingerprint", verification passed). Two or more: a "candidates" edge naming
    all of them. None: no edge.

    "One candidate" means one among the files named. Name every file the step can
    have read: a twin in a file left out is not seen. A text the later file holds
    more often than the named files do shows such a file: no copy of it gets an
    edge, and the result counts the copies in ``surplus``.

    With ``write`` false nothing is written and the result alone comes back."""
    step = _need_step(step)
    if isinstance(earlier, str):
        raise InvalidRecord("the possible inputs are a sequence of files, not one text")
    later_artifact = resolve_jsonl(store, later, "the later file")
    earlier_artifacts = [resolve_jsonl(store, ref, "a possible input") for ref in earlier]
    if not earlier_artifacts:
        raise InvalidRecord("a copy step is recovered against at least one possible input")
    seen: set[str] = set()
    for artifact in earlier_artifacts:
        if artifact.instance_id == later_artifact.instance_id:
            raise InvalidRecord(
                f"{shown_name(artifact)} is named as the later file and as its own input"
            )
        if artifact.instance_id in seen:
            raise InvalidRecord(f"{shown_name(artifact)} is named twice among the inputs")
        seen.add(artifact.instance_id)

    # A copy step is confirmed on normalised texts, so the function is needed. Its
    # absence is the named error "unknown normaliser": nothing degrades silently.
    function = normalise.get(store.normaliser)
    later_file = RegisteredFile(store, later_artifact, function=function, text_field=text_field)
    earlier_files = [
        RegisteredFile(store, artifact, function=function, text_field=text_field)
        for artifact in earlier_artifacts
    ]
    try:
        found = match_records(
            store, later_file, earlier_files, step=step, derivation=v.DERIVED_FINGERPRINT
        )
    finally:
        for file in earlier_files:
            file.close()
    written = _write_edges(store, later_file.instance_id, found.edges) if write else 0
    return RecoveryResult(
        kind=KIND_COPY_STEP,
        step=step,
        artifact=later_file.name,
        instance_id=later_file.instance_id,
        outcome=OUTCOME_RECORD_LEVEL,
        reason=None,
        records=found.records,
        recovered=found.recovered,
        candidates=found.candidates,
        no_edge=found.no_edge,
        no_text=found.no_text,
        collisions_rejected=found.collisions_rejected,
        drops=0,
        input_lines_unaccounted=None,
        edges_written=written,
        drops_written=0,
        links_written=0,
        members_linked=(),
        files_read=_files_read([later_file, *earlier_files]),
        surplus=found.surplus,
        dry_run=not write,
    )


# --- The sanitizer ---------------------------------------------------------------------


def _file_members(store, instance_id: str) -> tuple[str, ...]:
    """The members a file as a whole is tied to in the store's own records: the
    members whose part is the whole file, and the members a file-level link of the
    file names."""
    names = {member.name for member in store.members() if (instance_id, "") in member.parts}
    names.update(
        link.member
        for link in store.file_source_links(instance_id)
        if link.label == v.FILE_LEVEL
    )
    return tuple(sorted(names))


def _refuse_recorded_skips(
    store, later: Artifact, earlier: Artifact, drops: Sequence[DropRecord]
) -> None:
    """A skipped or empty line is recorded once, as a drop whose output is the
    first registered artifact that lacks the line. A logged line that already
    has a sanitizer's drop for a shard was recorded by a map recovery that read
    this log with only the sanitizer's input registered: recording it again for
    the later file would count the line twice. So would a second step name for
    the same line and the same later file. Both are refused before anything is
    written. A drop of the line for another JSONL file is no conflict: the
    sanitizer ran twice and wrote two files."""
    for drop in drops:
        for held in store.drops_of(drop.input):
            if held.reason not in v.SANITIZER_DROP_REASONS:
                continue
            if held.output_instance_id == later.instance_id:
                if held.step_name != drop.step_name:
                    raise DropAlreadyRecorded(
                        f"{drop.input.key} already has a sanitizer's drop for "
                        f"{shown_name(later)} under the step name {held.step_name!r}; this "
                        f"recovery names the step {drop.step_name!r}",
                        record=drop.input.key, step=held.step_name,
                    )
                continue
            output = store.artifact(held.output_instance_id)
            if output is not None and output.kind == v.KIND_SHARD:
                raise DropAlreadyRecorded(
                    f"{drop.input.key} already has its drop under the shard "
                    f"{shown_name(output)}, whose map was recovered through this log; the "
                    "line is not recorded twice. Run this recovery with no log to record "
                    "the edges alone",
                    record=drop.input.key, output=output.instance_id, step=held.step_name,
                )


def _degraded(
    store,
    later: Artifact,
    earlier: Artifact,
    *,
    step: StepRef,
    drops: Sequence[DropRecord],
    logged: bool,
    log_path: PathLike | None,
    error: UnknownNormaliser,
    write: bool,
) -> RecoveryResult:
    """The sanitizer cannot be re-run: no record is matched (plan 3.3, F8).

    The link degrades to file-level only where something other than the
    caller's word supports it, in one of two ways:

    - a file family: the later file itself holds a file-level link to a member.
      The link is there already; nothing is written;
    - the step's own statement: the sanitizer's log was given, the lines it
      names and the lines of the later file add up to the lines of the earlier
      file, and the store ties the earlier file as a whole to members. The later
      file then gets a file-level link to each of them, and the evidence text
      names the log's count.

    Otherwise the outcome is "unknown": that the later file was written from the
    earlier one is then the caller's statement alone, and nothing is written
    about the records. The log's drops are recorded either way; they are the
    step's own statement and need no function."""
    fingerprints = [occurrence.fingerprint for occurrence in store.occurrences(later.instance_id)]
    no_text = sum(1 for fingerprint in fingerprints if fingerprint is None)
    family = tuple(sorted({
        link.member for link in store.file_source_links(later.instance_id)
        if link.label == v.FILE_LEVEL
    }))
    earlier_lines = store.occurrence_count(earlier.instance_id)
    supported = logged and earlier_lines - len(drops) == len(fingerprints)
    from_earlier = _file_members(store, earlier.instance_id) if supported else ()
    new_drops = _plan_drops(store, later.instance_id, drops) if write else 0
    if drops and write:
        store.add_drops(later.instance_id, drops)
    evidence = (
        f"written by the step {step.name!r} (version {step.version!r}) from the file "
        f"{earlier.instance_id}, by the step's own log: {len(drops)} of its "
        f"{earlier_lines} lines are logged as skipped and this file holds "
        f"{len(fingerprints)}; the normaliser {store.normaliser!r} was not available "
        "when this was recovered, so no record was matched"
    )
    held = {
        (link.member, link.label, link.evidence, link.derivation, link.verification)
        for link in store.file_source_links(later.instance_id)
    }
    links_written = 0
    for member in from_earlier:
        if not write:
            break
        if (member, v.FILE_LEVEL, evidence, v.DERIVED_FILE, None) not in held:
            links_written += 1
        store.add_source_link(
            later.instance_id, None, member, v.FILE_LEVEL, evidence,
            derivation=v.DERIVED_FILE, verification=None,
        )
    members = tuple(sorted(set(family) | set(from_earlier)))
    why = f"{error.name}: {error.detail}"
    if not members:
        if not logged:
            why += ("; no sanitizer's log was given, so that this file was written from "
                    f"{shown_name(earlier)} is the caller's statement alone")
        elif not supported:
            why += (f"; the log names {len(drops)} of the {earlier_lines} lines of "
                    f"{shown_name(earlier)} and this file holds {len(fingerprints)}, so the "
                    "log does not show that this file was written from it")
        else:
            why += f"; the store ties {shown_name(earlier)} as a whole to no member"
    return RecoveryResult(
        kind=KIND_SANITIZER,
        step=step,
        artifact=shown_name(later),
        instance_id=later.instance_id,
        outcome=OUTCOME_FILE_LEVEL if members else OUTCOME_UNKNOWN,
        reason=why,
        records=len(fingerprints),
        recovered=0,
        candidates=0,
        no_edge=len(fingerprints),
        no_text=no_text,
        collisions_rejected=0,
        drops=len(drops),
        input_lines_unaccounted=None,
        edges_written=0,
        drops_written=new_drops,
        links_written=links_written,
        members_linked=members,
        files_read=() if log_path is None else (os.fspath(log_path),),
        dry_run=not write,
    )


def recover_sanitizer(
    store,
    later: str,
    earlier: str,
    *,
    step: StepRef,
    log_path: PathLike | None = None,
    log_input: str | None = None,
    text_field: str = "text",
    write: bool = True,
) -> RecoveryResult:
    """Recover the links of a sanitizer, a deterministic rewrite, for one file it
    wrote (plan 3.3 "Deterministic rewrite", F8).

    ``earlier`` is the file the sanitizer read and ``later`` the file it wrote,
    both registered. ``log_path`` is the sanitizer's own log of skipped lines
    (``read_sanitizer_log`` gives its form) and ``log_input`` the value its rows
    carry for ``earlier`` when the log covers several files.

    The store's fingerprints are taken after the store's normaliser, which is the
    pipeline's sanitizer (DECISIONS.md D-17). So the function has been re-run on
    every line of the earlier file before any fingerprint is matched, and it is
    run again on both lines of each pair for the direct comparison. One candidate
    gives a "recovered" edge with the derivation "function", two or more a
    "candidates" edge, none no edge.

    Every line the log names is a drop of ``step`` for the later file, and is no
    candidate parent: the step's own log says it wrote nothing for that line.
    ``sanitize_step`` builds the step under the name a map recovery gives the
    same drops. A log whose lines the store already holds as drops for a shard
    is refused with "drop already recorded": a skipped line is recorded once.

    When the store's normaliser is not registered in this process the function
    cannot be re-run. No record is then matched and the result's ``outcome`` is
    "file-level" or "unknown" (see ``RecoveryResult``); it is never a guess.

    With ``write`` false nothing is written and the result alone comes back."""
    step = _need_step(step)
    if not isinstance(earlier, str):
        raise InvalidRecord("a sanitizer reads one file; name it by a name or an instance id")
    later_artifact = resolve_jsonl(store, later, "the later file")
    earlier_artifact = resolve_jsonl(store, earlier, "the sanitizer's input")
    if earlier_artifact.instance_id == later_artifact.instance_id:
        raise InvalidRecord(
            f"{shown_name(later_artifact)} is named as the later file and as its own input"
        )

    skips: tuple[LoggedSkip, ...] = ()
    if log_path is not None:
        skips = read_sanitizer_log(log_path, input=log_input)
    elif log_input is not None:
        raise InvalidRecord("an input file of a log is named and no log is given")
    lines = store.occurrence_count(earlier_artifact.instance_id)
    for skip in skips:
        if skip.line > lines:
            raise InvalidRecord(
                f"the log names line {skip.line} of {shown_name(earlier_artifact)}, which "
                f"has {lines} lines; the log is not the log of this file",
                line=skip.line,
            )
    drops = tuple(
        DropRecord(
            input=OccurrenceRef(earlier_artifact.instance_id, capture.line_locator(skip.line)),
            reason=skip.reason,
            step_name=step.name,
        )
        for skip in skips
    )
    _refuse_recorded_skips(store, later_artifact, earlier_artifact, drops)

    try:
        function = normalise.get(store.normaliser)
    except UnknownNormaliser as error:
        return _degraded(
            store, later_artifact, earlier_artifact,
            step=step, drops=drops, logged=log_path is not None, log_path=log_path,
            error=error, write=write,
        )

    later_file = RegisteredFile(store, later_artifact, function=function, text_field=text_field)
    earlier_file = RegisteredFile(store, earlier_artifact, function=function, text_field=text_field)
    logged = {skip.line for skip in skips}
    try:
        found = match_records(
            store, later_file, [earlier_file],
            step=step, derivation=v.DERIVED_FUNCTION,
            excluded={(0, line) for line in logged},
        )
    finally:
        earlier_file.close()
    accounted = logged | {line for _index, line in found.named}
    new_drops = written = 0
    if write:
        new_drops = _plan_drops(store, later_file.instance_id, drops)
        written = _write_edges(store, later_file.instance_id, found.edges)
        if drops:
            store.add_drops(later_file.instance_id, drops)
    return RecoveryResult(
        kind=KIND_SANITIZER,
        step=step,
        artifact=later_file.name,
        instance_id=later_file.instance_id,
        outcome=OUTCOME_RECORD_LEVEL,
        reason=None,
        records=found.records,
        recovered=found.recovered,
        candidates=found.candidates,
        no_edge=found.no_edge,
        no_text=found.no_text,
        collisions_rejected=found.collisions_rejected,
        drops=len(drops),
        input_lines_unaccounted=lines - len(accounted),
        edges_written=written,
        drops_written=new_drops,
        links_written=0,
        members_linked=(),
        files_read=_files_read([later_file, earlier_file], log_path),
        surplus=found.surplus,
        dry_run=not write,
    )
