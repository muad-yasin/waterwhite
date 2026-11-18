"""The capture wrapper around a tokenizer run, and the recovery of a map for a run
made without it (plan 3.3 "Tokenization", 5.4, Decision 4).

Both end in the same place: the maps of one shard are derived and verified
(``_derive``), turned into map records and handed to the store with the document
count and the digest that were read from the shard's files before the first
token was read. The store reads the files again and refuses a count or a digest
that does not match them, so a shard that changed in between gets no map.

What is written is one call of ``store.write_maps``, one transaction (plan 3.4):
every map that was derived and, when one of them verified, the drops that belong
to the shard. The store holds each map against what the shard already has in the
same namespace and derivation: an equal one is left alone, another one beside a
verified map refuses the whole call, and then nothing is written. A process
killed at any point leaves the maps with their drops, or none of it.

A capture map goes to the namespace "capture" and a recovery map to the namespace
"recovery" (DECISIONS.md D-20), so a recovery never touches a map the wrapper
wrote.

A line a sanitizer skipped is recorded once, as a drop whose output is the first
registered artifact that lacks the line. ``recover_map`` records the skips of a
sanitizer's log for the shard only when no registered file stands between the
sanitizer's input and the shard; otherwise it refuses the log by name.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .. import ids
from .. import vocabulary as v
from ..errors import (
    ArtifactDigestMismatch,
    DropAlreadyRecorded,
    InvalidRecord,
    MapCountMismatch,
)
from ..records import DropRecord, MapRecord, MapRow, OccurrenceRef, StepRef
from ..store import artifact_files
from ._derive import (
    Assignment,
    DocumentClasses,
    Retokenizer,
    Tokenize,
    Verification,
    derive_position,
    derive_token_matching,
)
from ._inputs import (
    SanitizerRun,
    TokenizerSettings,
    numbered_lines,
    require_registered_bytes,
    sanitizer_function,
    text_reader,
)
from ._shard import ShardTokens

RunTokenizer = Callable[[Path, Path], object]


@dataclass(frozen=True)
class MapOutcome:
    """One map of a shard as it was derived: the record handed to the store, what
    its verification found, and whether the store holds it now. ``recovered`` and
    ``candidates`` count the documents with each label."""

    record: MapRecord
    verification: Verification
    recovered: int
    candidates: int
    stored: bool

    @property
    def namespace(self) -> str:
        return self.record.namespace

    @property
    def derivation(self) -> str:
        return self.record.derivation

    @property
    def label(self) -> str:
        return self.record.label

    def to_json(self) -> dict:
        """The outcome without its rows: what the map is, what its verification
        found, how many documents carry each label, and whether it is stored."""
        check = self.verification
        return {
            "namespace": self.namespace,
            "derivation": self.derivation,
            "verification": check.result,
            "label": self.label,
            "append_eod": self.record.append_eod,
            "eod_in_span": self.record.eod_in_span,
            "documents": len(self.record.rows),
            "recovered": self.recovered,
            "candidates": self.candidates,
            "unknown": len(self.record.rows) - self.recovered - self.candidates,
            "spans_match": check.spans_match,
            "count_matches": check.count_matches,
            "documents_compared": check.documents_compared,
            "failures": check.failures,
            "reasons": list(check.reasons),
            "stored": self.stored,
        }


@dataclass(frozen=True)
class ShardMaps:
    """What one call derived for one shard.

    ``maps`` holds one outcome per mode that ran, in the order they ran.
    ``drops`` are the drops that belong to the shard: those of the sanitizer's log
    and the lines that gave no tokens. They are empty when no map verified.
    ``notes`` says in words what was not done and why."""

    shard_instance_id: str
    input_instance_id: str
    document_count: int
    surviving_lines: int
    maps: tuple[MapOutcome, ...]
    drops: tuple[DropRecord, ...]
    notes: tuple[str, ...]

    def map(self, derivation: str) -> MapOutcome | None:
        """The outcome of one mode, or None when that mode did not run."""
        for outcome in self.maps:
            if outcome.derivation == derivation:
                return outcome
        return None

    @property
    def verified(self) -> bool:
        """True when at least one of the maps passed its verification."""
        return any(m.verification.result == v.VERIFIED_PASSED for m in self.maps)

    def to_json(self) -> dict:
        """The result as a JSON object: the maps without their rows, the drops as
        (record, reason, step) and the notes."""
        return {
            "shard": self.shard_instance_id,
            "input": self.input_instance_id,
            "documents": self.document_count,
            "surviving_lines": self.surviving_lines,
            "verified": self.verified,
            "maps": [outcome.to_json() for outcome in self.maps],
            "drops": [
                {"record": drop.input.locator, "reason": drop.reason, "step": drop.step_name}
                for drop in self.drops
            ],
            "notes": list(self.notes),
        }


# --- Naming things in the store ------------------------------------------------------------


def _artifact(store, ref: str, kind: str, what: str):
    found = store.artifact(ref) if isinstance(ref, str) and ref else None
    if found is None:
        raise InvalidRecord(f"{what} {ref!r} is not registered")
    if found.kind != kind:
        raise InvalidRecord(f"{what} {ref!r} is a {found.kind}, not a {kind}")
    return found


def _disk_path(store, artifact, what: str) -> Path:
    """Where an artifact's file lies, as the store gives it (``Store.disk_path``)."""
    path = store.disk_path(artifact.instance_id)
    if path is None:
        raise InvalidRecord(f"{what} has no recorded path")
    return path


# --- From an assignment to a map record -----------------------------------------------------


def _map_record(
    assignment: Assignment,
    classes: DocumentClasses | None,
    namespace: str,
    settings: TokenizerSettings,
    input_id: str,
    fingerprints: Sequence[bytes | None],
) -> tuple[MapRecord, int, int]:
    """The map record of one assignment, with the label the mode and the result
    give (plan 3.3): a passed verification gives "recovered" for a document whose
    tokens equal no other document's in the shard and "candidates", with the
    lines of all its twins, for the rest; anything else gives "unknown" for the
    map and for every row, and an "unknown" row names no record."""
    passed = assignment.verification.result == v.VERIFIED_PASSED
    starts, ends, line_of = assignment.starts, assignment.ends, assignment.line_of
    rows: list[MapRow] = []
    recovered = candidates = 0

    def ref(line: int) -> OccurrenceRef:
        return OccurrenceRef(input_id, ids.line_locator(line))

    for ordinal in range(line_of.shape[0]):
        start, end = int(starts[ordinal]), int(ends[ordinal])
        if not passed:
            rows.append(MapRow(ordinal, start, end, v.UNKNOWN))
            continue
        members = classes.members(ordinal) if classes is not None else (ordinal,)
        if len(members) == 1:
            line = int(line_of[ordinal])
            rows.append(MapRow(ordinal, start, end, v.RECOVERED, parent=ref(line),
                               fingerprint=fingerprints[line - 1]))
            recovered += 1
            continue
        lines = sorted(int(line_of[m]) for m in members)
        prints = {fingerprints[line - 1] for line in lines}
        rows.append(MapRow(ordinal, start, end, v.CANDIDATES,
                           candidates=tuple(ref(line) for line in lines),
                           fingerprint=prints.pop() if len(prints) == 1 else None))
        candidates += 1
    record = MapRecord(
        namespace=namespace,
        derivation=assignment.derivation,
        verification=assignment.verification.result,
        label=v.RECOVERED if passed else v.UNKNOWN,
        append_eod=settings.append_eod,
        eod_in_span=settings.eod_in_span,
        rows=tuple(rows),
    )
    return record, recovered, candidates


def _drops(
    assignment: Assignment, input_id: str, sanitizer: SanitizerRun | None, tokenize_step: str
) -> tuple[DropRecord, ...]:
    drops = []
    if sanitizer is not None:
        for skip in sanitizer.skips:
            drops.append(DropRecord(OccurrenceRef(input_id, ids.line_locator(skip.line)),
                                    skip.reason, sanitizer.step_name))
    for line in assignment.no_token_lines:
        drops.append(DropRecord(OccurrenceRef(input_id, ids.line_locator(line)),
                                v.DROP_NO_TOKENS, tokenize_step))
    return tuple(sorted(drops, key=lambda d: ids.parse_locator(d.input.locator)[1]))


def _refuse_recorded_skips(store, shard, input, sanitizer: SanitizerRun) -> None:
    """A skipped or empty line is recorded once, as a drop whose output is the
    first registered artifact that lacks the line. A sanitizer's log is refused
    here, before anything is derived, when the store shows that this is not the
    shard:

    - the input is itself the output of recorded sanitizer drops. It is then the
      file the sanitizer wrote; it lacks the skipped lines already, and the log
      numbers the lines of another file;
    - a line the log names already has its drop under a registered JSONL file.
      That file is the first registered artifact that lacks the line, and the
      shard's map belongs over it, with no log;
    - a line the log names already has a sanitizer's drop for this shard under
      another step name.

    A drop of the same line for another shard is no conflict: two tokenizer runs
    over one sanitizer input are two outputs."""
    input_name = input.names[0] if input.names else input.instance_id
    for held in store.drops_by_output(input.instance_id):
        if held.reason in v.SANITIZER_DROP_REASONS:
            raise DropAlreadyRecorded(
                f"{input_name} is the output of recorded sanitizer drops (step "
                f"{held.step_name!r}): it is the file the sanitizer wrote and lacks the "
                "skipped lines already. Recover the map over it with no sanitizer's log",
                input=input.instance_id, step=held.step_name,
            )
    for skip in sanitizer.skips:
        ref = OccurrenceRef(input.instance_id, ids.line_locator(skip.line))
        for held in store.drops_of(ref):
            if held.reason not in v.SANITIZER_DROP_REASONS:
                continue
            if held.output_instance_id == shard.instance_id:
                if held.step_name != sanitizer.step_name:
                    raise DropAlreadyRecorded(
                        f"line {skip.line} of {input_name} already has a sanitizer's drop "
                        f"for this shard under the step name {held.step_name!r}; this run "
                        f"names the step {sanitizer.step_name!r}",
                        record=ref.key, step=held.step_name,
                    )
                continue
            output = store.artifact(held.output_instance_id)
            if output is not None and output.kind == v.KIND_JSONL:
                shown = output.names[0] if output.names else output.instance_id
                raise DropAlreadyRecorded(
                    f"line {skip.line} of {input_name} already has its drop under {shown}, "
                    "a registered file that lacks the line. Recover the map over that "
                    "file with no sanitizer's log; the line is not recorded twice",
                    record=ref.key, output=output.instance_id, step=held.step_name,
                )


# --- The work both entry points share --------------------------------------------------------


def _check_arguments(
    settings: object,
    tokenize: object,
    sanitizer: object,
    f1_holds: bool,
    also_token_matching: bool,
) -> None:
    """Refuse a call that cannot be carried out, before a tokenizer is run or
    anything is registered."""
    if not isinstance(settings, TokenizerSettings):
        raise InvalidRecord("the settings of the tokenizer run are a TokenizerSettings")
    if sanitizer is not None and not isinstance(sanitizer, SanitizerRun):
        raise InvalidRecord("the sanitizer's run is a SanitizerRun")
    if tokenize is not None and not callable(tokenize):
        raise InvalidRecord("tokenize is a function from a text to its token ids")
    if tokenize is None and not f1_holds:
        raise InvalidRecord(
            "F1 is said not to hold, so the map is built by token matching, and token "
            "matching needs the tokenize function"
        )
    if tokenize is None and also_token_matching:
        raise InvalidRecord("token matching needs the tokenize function")


def _derive_and_write(
    store,
    shard,
    input,
    *,
    namespace: str,
    settings: TokenizerSettings,
    tokenize: Tokenize | None,
    sanitizer: SanitizerRun | None,
    f1_holds: bool,
    also_token_matching: bool,
    text_field: str,
    write: bool,
) -> ShardMaps:
    shard_id, input_id = shard.instance_id, input.instance_id
    prefix = _disk_path(store, shard, "the shard")
    for file in artifact_files.shard_paths(prefix):
        if not file.is_file():
            raise InvalidRecord(f"the shard's two files are not at {prefix}: no file at {file}")
    # The count and the digest every map of this call is declared with, taken
    # before the first token is read.
    on_disk = artifact_files.read_shard(prefix)
    if on_disk.digest != shard.content_digest:
        raise ArtifactDigestMismatch(
            f"shard {shard_id} is registered with digest {shard.content_digest} and its "
            f"files have {on_disk.digest}",
            instance=shard_id, registered=shard.content_digest, on_disk=on_disk.digest,
        )
    input_path = _disk_path(store, input, "the input")
    if not input_path.is_file():
        raise InvalidRecord(
            f"the input's file is not at {input_path}; a map is derived from its lines"
        )
    input_name = input.names[0] if input.names else input_id
    # Checked once before any line is tokenized, and again by every pass over the
    # lines, on the very bytes that pass read.
    require_registered_bytes(ids.file_digest(input_path), input.content_digest, input_name)

    def lines():
        return numbered_lines(input_path, input.content_digest, input_name)

    fingerprints = [occurrence.fingerprint for occurrence in store.occurrences(input_id)]
    skipped: frozenset[int] = frozenset()
    retokenize = None
    if sanitizer is not None:
        skipped = frozenset(skip.line for skip in sanitizer.skips)
        beyond = sorted(line for line in skipped if line > len(fingerprints))
        if beyond:
            raise InvalidRecord(
                f"the sanitizer's log names line {beyond[0]} and the input holds "
                f"{len(fingerprints)} lines; it is the log of another file"
            )
    if tokenize is not None:
        again = None if sanitizer is None else sanitizer_function(sanitizer, store.normaliser)
        retokenize = Retokenizer(tokenize, settings, text_reader(text_field, again))

    notes: list[str] = []
    assignments: list[Assignment] = []
    with ShardTokens(prefix) as tokens:
        classes = DocumentClasses(tokens) if tokenize is not None else None
        position = None
        if f1_holds:
            position = derive_position(lines, tokens, classes, retokenize, skipped)
            assignments.append(position)
        else:
            notes.append("F1 is said not to hold: position mode was not run")
        failed = position is not None and position.verification.result == v.VERIFIED_FAILED
        if not f1_holds or failed or also_token_matching:
            if retokenize is None:
                notes.append(
                    "position mode failed its verification and token matching was not "
                    "run: no tokenize function was given"
                )
            else:
                assignments.append(
                    derive_token_matching(lines, tokens, classes, retokenize, skipped))
    if retokenize is None and position is not None and not failed:
        notes.append("no tokenize function was given: the verification was not run")

    line_count = assignments[0].surviving_lines + len(skipped)
    if line_count != len(fingerprints):
        raise MapCountMismatch(
            f"the store holds {len(fingerprints)} records of {input_name} and its file "
            f"has {line_count} lines",
            instance=input_id, stored=len(fingerprints), on_disk=line_count,
        )

    document_count = on_disk.document_count
    planned = []
    for assignment in assignments:
        if assignment.line_of.shape[0] != document_count:
            raise ArtifactDigestMismatch(
                f"shard {shard_id} changed while its map was derived: its index held "
                f"{document_count} documents and then {assignment.line_of.shape[0]}",
                instance=shard_id,
            )
        planned.append((assignment, *_map_record(
            assignment, classes, namespace, settings, input_id, fingerprints)))

    verified = next(
        (a for a in assignments if a.verification.result == v.VERIFIED_PASSED), None)
    drops: tuple[DropRecord, ...] = ()
    if verified is not None:
        drops = _drops(verified, input_id, sanitizer, shard.step.name)
    else:
        notes.append("no map verified: no drop is recorded")

    stored = [False] * len(planned)
    if write:
        # One transaction: every map of this call and the drops, or nothing. The
        # store leaves a map it holds already alone and refuses the whole call
        # for a map or a drop that cannot stand beside what it holds.
        store.write_maps(
            shard_id, [record for _, record, _, _ in planned],
            declared_count=document_count, declared_digest=on_disk.digest, drops=drops,
        )
        stored = [True] * len(planned)
    else:
        notes.append("nothing was written: the call asked for the result only")

    return ShardMaps(
        shard_instance_id=shard_id,
        input_instance_id=input_id,
        document_count=document_count,
        surviving_lines=assignments[0].surviving_lines,
        maps=tuple(
            MapOutcome(record, assignment.verification, recovered, candidates, was_stored)
            for (assignment, record, recovered, candidates), was_stored in zip(planned, stored)
        ),
        drops=drops,
        notes=tuple(notes),
    )


# --- Capture ---------------------------------------------------------------------------------


def map_tokenized_shard(
    store,
    *,
    input: str,
    prefix: str | os.PathLike,
    name: str,
    step: StepRef,
    settings: TokenizerSettings,
    tokenize: Tokenize | None = None,
    f1_holds: bool = True,
    text_field: str = "text",
) -> ShardMaps:
    """Register the shard a tokenizer run just left at ``prefix`` and write its
    capture map.

    ``input`` names the registered JSONL artifact the tokenizer read (a name or an
    instance id). The shard is registered under ``name`` with ``step`` and that
    artifact as its one input. The map is built in position mode; where its
    verification fails, or with ``f1_holds`` false, it is built by token matching
    (plan 3.3). ``tokenize`` is the team's tokenizer as a function from a text to
    its token ids, without the end-of-document token; the tool ships none.
    Without it nothing is compared and the map is "not run" and "unknown"."""
    _check_arguments(settings, tokenize, None, f1_holds, False)
    artifact = _artifact(store, input, v.KIND_JSONL, "the input")
    shard_id = store.register_shard(prefix, name=name, step=step, inputs=[artifact.instance_id])
    shard = _artifact(store, shard_id, v.KIND_SHARD, "the shard")
    return _derive_and_write(
        store, shard, artifact, namespace=v.NAMESPACE_CAPTURE, settings=settings,
        tokenize=tokenize, sanitizer=None, f1_holds=f1_holds, also_token_matching=False,
        text_field=text_field, write=True,
    )


def tokenize_shard(
    store,
    *,
    input: str,
    prefix: str | os.PathLike,
    name: str,
    step: StepRef,
    run_tokenizer: RunTokenizer,
    settings: TokenizerSettings,
    tokenize: Tokenize | None = None,
    f1_holds: bool = True,
    text_field: str = "text",
) -> ShardMaps:
    """The capture wrapper: run the team's tokenizer, then register the shard and
    write its capture map (``map_tokenized_shard``).

    ``run_tokenizer(input_path, prefix)`` runs the team's tokenizer, unchanged, as
    a process of its own, on the file at ``input_path``, and leaves
    ``<prefix>.bin`` and ``<prefix>.idx``. The wrapper checks first that the file
    still holds the registered bytes, and that no file lies at the prefix yet: it
    registers what this run wrote, and a pair left by an earlier run is not taken
    for it."""
    if not callable(run_tokenizer):
        raise InvalidRecord("run_tokenizer is a function of the input path and the shard prefix")
    _check_arguments(settings, tokenize, None, f1_holds, False)
    artifact = _artifact(store, input, v.KIND_JSONL, "the input")
    input_path = _disk_path(store, artifact, "the input")
    if not input_path.is_file():
        raise InvalidRecord(f"the input's file is not at {input_path}")
    on_disk = ids.file_digest(input_path)
    if on_disk != artifact.content_digest:
        raise ArtifactDigestMismatch(
            f"{input!r} is registered with digest {artifact.content_digest} and its file "
            f"has {on_disk}; the tokenizer is not run on other bytes",
            registered=artifact.content_digest, on_disk=on_disk,
        )
    for file in artifact_files.shard_paths(prefix):
        if file.exists():
            raise InvalidRecord(
                f"a file already lies at {file}; the wrapper registers the pair this run "
                "of the tokenizer writes"
            )
    run_tokenizer(input_path, Path(os.fspath(prefix)))
    return map_tokenized_shard(
        store, input=artifact.instance_id, prefix=prefix, name=name, step=step,
        settings=settings, tokenize=tokenize, f1_holds=f1_holds, text_field=text_field,
    )


# --- Recovery --------------------------------------------------------------------------------


def recover_map(
    store,
    *,
    shard: str,
    input: str,
    settings: TokenizerSettings,
    tokenize: Tokenize | None = None,
    sanitizer: SanitizerRun | None = None,
    f1_holds: bool = True,
    also_token_matching: bool = False,
    text_field: str = "text",
    write: bool = True,
) -> ShardMaps:
    """Recover the map of a registered shard whose tokenizer ran without the
    wrapper, into the namespace "recovery" (plan 3.3, 5.4).

    ``input`` names the registered JSONL artifact whose lines became the
    documents. With ``sanitizer`` None it is the file the tokenizer read: every
    line survived. With a ``SanitizerRun`` it is the file the sanitizer read: the
    lines its log names are drops and take no ordinal, and the text of every
    other line is passed through the sanitizer again before it is tokenized again.

    Position mode runs first. Token matching runs when position mode's
    verification fails, when ``f1_holds`` is false (position mode is then not
    run), and when ``also_token_matching`` asks for both maps so that they can be
    compared. Every mode that ran leaves its map. With ``write`` false nothing is
    written and the result alone is returned, so a result can be looked at
    before it is stored.

    A sanitizer's log is refused with "drop already recorded" when the store
    already holds the skips under a registered file between the sanitizer's
    input and the shard, or when ``input`` is itself that file: a skipped line
    is recorded once (``_refuse_recorded_skips``)."""
    _check_arguments(settings, tokenize, sanitizer, f1_holds, also_token_matching)
    shard_artifact = _artifact(store, shard, v.KIND_SHARD, "the shard")
    input_artifact = _artifact(store, input, v.KIND_JSONL, "the input")
    if sanitizer is not None:
        _refuse_recorded_skips(store, shard_artifact, input_artifact, sanitizer)
    return _derive_and_write(
        store, shard_artifact, input_artifact, namespace=v.NAMESPACE_RECOVERY,
        settings=settings, tokenize=tokenize, sanitizer=sanitizer, f1_holds=f1_holds,
        also_token_matching=also_token_matching, text_field=text_field, write=write,
    )
