"""What the replica tells the tool about the files of the pipeline steps.

One shard of one layout is made of four things: the chunk as the sanitizer read
it, the sanitizer's log, the tokenizer input and the shard's two files
(``replica.shard_files``). This module holds the calls that put the first three
into a store, for the capture hooks (``replica.capture_hooks``) and for the
recovery of a run made without capture (``replica.recover``) alike. It is imported
by those two and by the probe run (``replica.probe_run``), which registers a mixture.

The copy steps and the sanitizer are recovery points in both cases (plan 5.4):
their files are registered as files found afterwards and their links recovered
through ``waterwhite.recovery``. Tokenization differs. With capture the shard's
map is the capture map, written by ``waterwhite.tokenmap.map_tokenized_shard``.
Without it the shard is registered afterwards and its map recovered by
``waterwhite.tokenmap.recover_map``.

A dropped line is recorded exactly once, as a drop whose output is the first
registered artifact that lacks the line:

- a line the sanitizer skipped: the tokenizer input, when that file is registered
  (``waterwhite.recovery.recover_sanitizer`` reads the sanitizer's log); the shard,
  when only the sanitizer's input was kept (``recover_map`` is then given the log);
- a line of the tokenizer input that gives no tokens: the shard.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from waterwhite import capture as ww_capture
from waterwhite import recovery as ww_recovery
from waterwhite import tokenmap as ww_tokenmap
from waterwhite.store import Store

from .errors import CaptureError, SpecError
from .plan import TokenizerSettings
from .shard_files import ShardFiles

STEP_VERSION = "1"
STEP_COPY = "replica.shuffle-chunk-merge"
STEP_SANITIZE = ww_tokenmap.STEP_SANITIZE


def open_store(store_path: Path, normaliser: str, *,
               verified_joins: Sequence[tuple[str, str, str]] = ()) -> Store:
    """Create the store at ``store_path`` with ``normaliser`` and ``verified_joins``,
    or open the one that is there (its list was written when it was made and is not
    changed). A store whose fingerprints were taken after another normaliser is
    refused: the sanitizer of the build and the store's fingerprints would not
    agree (plan 3.1, F8)."""
    store_path = Path(store_path)
    if not store_path.exists():
        store_path.parent.mkdir(parents=True, exist_ok=True)
        return Store.create(store_path, normaliser=normaliser, verified_joins=verified_joins)
    store = Store.open(store_path)
    held = ww_capture.store_normaliser(store)
    if held != normaliser:
        store.close()
        raise CaptureError(f"the store at {store_path} was created with the normaliser "
                           f"{held!r} and this build sanitizes with {normaliser!r}")
    return store


# --- A mixture's snapshot ------------------------------------------------------------------------


def stored_path(store: Store, path: Path) -> str:
    """A path as the store keeps one (docs/interfaces.md section 4): relative to the
    folder of the store file when it lies under it, absolute otherwise."""
    folder = Path(store.path).resolve().parent
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(folder).as_posix()
    except ValueError:
        return str(absolute)


def snapshot_shard(store: Store, position: int, instance_id: str,
                   prefix: Path) -> ww_capture.SnapshotShard:
    """One entry of a mixture's snapshot (plan Decision 27): the shard by its
    registered instance, where its files lie as the store keeps a path, and the digest
    of the two files that lie there. The store refuses an entry whose digest differs
    from the digest its shard is registered with."""
    digest = ww_capture.shard_digest(ww_capture.file_digest(f"{prefix}.bin"),
                                     ww_capture.file_digest(f"{prefix}.idx"))
    return ww_capture.SnapshotShard(position, instance_id, stored_path(store, prefix), digest)


# --- The steps as the store names them -------------------------------------------------------


def map_settings(tokenizer: TokenizerSettings) -> ww_tokenmap.TokenizerSettings:
    """The settings every map records (plan 3.3, F1): ``--append-eod`` as the run
    was given it, and the end-of-document convention as the spec records it from
    the oracle's output. A build whose tokenizer table does not record the
    convention cannot write a map."""
    if tokenizer.eod_in_span is None:
        raise SpecError("the tokenizer table does not record the end-of-document "
                        "convention (eod_in_span); a tokenization map is written only "
                        "once the fixture spec records it (plan 3.3, F1)")
    if tokenizer.eod_in_span and tokenizer.eod_token_id is None:
        raise SpecError("the tokenizer table records eod_in_span = true and no eod_token_id")
    return ww_tokenmap.TokenizerSettings(append_eod=tokenizer.append_eod,
                                         eod_in_span=tokenizer.eod_in_span,
                                         eod_token_id=tokenizer.eod_token_id)


def tokenize_step(tokenizer: TokenizerSettings):
    """The step a shard is registered under: the hub revision of the tokenizer's
    files as the version, and the switches of the run as its parameters."""
    return ww_tokenmap.tokenize_step(tokenizer.revision, {
        "type": tokenizer.type,
        "name": tokenizer.name,
        "vocab_file": tokenizer.vocab_file,
        "merges_file": tokenizer.merges_file,
        "append_eod": tokenizer.append_eod,
        "partitions": tokenizer.partitions,
    })


def copy_step(layout: str):
    """The copy steps of one layout (list, shuffle, chunk, merge, shuffle), as one
    step: the files between them are not kept."""
    return ww_capture.step_ref(STEP_COPY, STEP_VERSION, {"layout": layout})


def sanitize_step(normaliser: str):
    """The sanitizer: its version is the name of the normaliser it applies. The step
    is the tool's own (``recovery.sanitize_step``), so that a map recovered through
    the sanitizer's log writes its drops under the same step name."""
    return ww_recovery.sanitize_step(normaliser)


# --- The files before the shard ----------------------------------------------------------------


def register_sanitizer_input(store: Store, files: ShardFiles, earlier: Sequence[str]) -> dict:
    """Register the chunk as the sanitizer read it, as a file found afterwards, and
    recover the copy steps between ``earlier`` and it by fingerprint.

    ``earlier`` names the registered files the copy steps can have read for this
    shard. With none of them registered nothing is matched: the records of the
    chunk keep no recorded origin, and the result says so."""
    name = files.sanitizer_input_name
    instance = ww_recovery.register_found_file(store, files.sanitizer_input, name=name)
    report = {"name": name, "instance": instance, "earlier": list(earlier)}
    if earlier:
        result = ww_recovery.recover_copy_step(store, name, list(earlier),
                                               step=copy_step(files.layout))
        report.update(recovered=result.recovered, candidates=result.candidates,
                      no_edge=result.no_edge)
    else:
        report.update(recovered=0, candidates=0, no_edge=None)
    return report


def register_tokenizer_input(store: Store, files: ShardFiles, normaliser: str) -> dict:
    """Register the tokenizer input, as a file found afterwards, and recover the
    sanitizer between the chunk and it. Every line the sanitizer's log names for
    this chunk becomes a drop whose output is the tokenizer input."""
    name = files.tokenizer_input_name
    instance = ww_recovery.register_found_file(store, files.tokenizer_input, name=name)
    result = ww_recovery.recover_sanitizer(
        store, name, files.sanitizer_input_name, step=sanitize_step(normaliser),
        log_path=files.sanitizer_log, log_input=files.log_input)
    if result.outcome != ww_recovery.OUTCOME_RECORD_LEVEL:
        raise CaptureError(f"{name}: the sanitizer could not be re-run ({result.reason})")
    return {"name": name, "instance": instance, "recovered": result.recovered,
            "candidates": result.candidates, "no_edge": result.no_edge,
            "drops": result.drops, "input_lines_unaccounted": result.input_lines_unaccounted}


def sanitizer_run(files: ShardFiles) -> ww_tokenmap.SanitizerRun:
    """The sanitizer's run over this shard's chunk, read from its log, for a map
    that is recovered over the chunk itself."""
    skips = ww_tokenmap.read_sanitizer_log(files.sanitizer_log, input=files.log_input)
    return ww_tokenmap.SanitizerRun(skips, step_name=STEP_SANITIZE)


# --- What a map call gave, as plain values ------------------------------------------------------


def maps_report(result) -> dict:
    """A ``tokenmap.ShardMaps`` as a build's summary keeps it."""
    return {
        "shard_instance": result.shard_instance_id,
        "documents": result.document_count,
        "surviving_lines": result.surviving_lines,
        "verified": result.verified,
        "maps": [{
            "namespace": outcome.namespace,
            "derivation": outcome.derivation,
            "verification": outcome.verification.result,
            "label": outcome.label,
            "recovered": outcome.recovered,
            "candidates": outcome.candidates,
            "stored": outcome.stored,
            "failures": outcome.verification.failures,
            "reasons": list(outcome.verification.reasons),
        } for outcome in result.maps],
        "drops": [{"record": f"{drop.input.instance_id}:{drop.input.locator}",
                   "line": int(drop.input.locator[1:]), "reason": drop.reason,
                   "step": drop.step_name} for drop in result.drops],
        "notes": list(result.notes),
    }
