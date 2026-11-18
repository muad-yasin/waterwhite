"""Digests, fingerprints and identifiers: one definition each.

Every hash the tool stores or compares is computed here, so two machines that hold
the same records compute the same ids (plan 3.1, Decision 29). The one definition
of what a line of a JSONL file is lives here too, because a locator is a line
number and so belongs to a record's identity.

Choices recorded in DECISIONS.md (plan 5.8, B-2):
- a content digest is SHA-256, the digest a team can check with standard tools;
- a fingerprint is BLAKE2b with a 16-byte digest (plan 3.1, Decision 11);
- an instance id and an edge digest are BLAKE2b with a 32-byte digest.

Hash inputs are encoded as canonical JSON (sorted keys, no spaces, ASCII only) with
a domain tag as the first list item, so an id of one kind can never equal an id of
another kind computed from the same fields.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Sequence

FINGERPRINT_BYTES = 16
INSTANCE_ID_BYTES = 32

_READ_CHUNK = 1 << 20


def canonical(value: object) -> bytes:
    """Canonical JSON bytes of a value built from lists, dicts, str, int, bool, None."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _blake(data: bytes, size: int) -> bytes:
    return hashlib.blake2b(data, digest_size=size).digest()


# --- Content digests ---------------------------------------------------------------


def bytes_digest(data: bytes) -> str:
    """SHA-256 of bytes, lower-case hex."""
    return _sha256(data)


class ContentHasher:
    """A content digest taken as the bytes go by: SHA-256, lower-case hex.

    ``update`` takes the next bytes, ``hexdigest`` gives the digest of everything
    taken so far, ``size`` the number of bytes. ``file_digest`` is this hasher fed
    with a whole file, so a digest taken while a file is read for something else
    (``read_lines``) is the file's content digest."""

    def __init__(self) -> None:
        self._hash = hashlib.sha256()
        self._size = 0

    def update(self, data: bytes) -> None:
        self._hash.update(data)
        self._size += len(data)

    def hexdigest(self) -> str:
        return self._hash.hexdigest()

    @property
    def size(self) -> int:
        return self._size


def file_digest(path: str | os.PathLike) -> str:
    """SHA-256 of a file's bytes, lower-case hex. This is a JSONL artifact's content
    digest and the digest the loud checks compare (plan 2.2(d), 3.4)."""
    hasher = ContentHasher()
    with open(path, "rb") as f:
        while chunk := f.read(_READ_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def shard_digest(bin_digest: str, idx_digest: str) -> str:
    """Content digest of a tokenized shard: its .bin and .idx file digests together.

    A shard is the pair, so a change to either file changes the digest."""
    return _sha256(canonical(["ww.shard.v1", bin_digest, idx_digest]))


def array_digest(dtype: str, shape: Sequence[int], data: bytes) -> str:
    """Digest of one data-order index array (plan 3.5, replay rule).

    Taken over the array's dtype string (numpy's ``dtype.str``, for example ``<i4``),
    its shape and its bytes in C order, and not over the ``.npy`` file: the file name
    and the ``description.txt`` beside it carry paths, and the ``.npy`` header can
    change with the numpy version, while the array does not."""
    h = hashlib.sha256()
    h.update(canonical(["ww.array.v1", dtype, list(shape)]))
    h.update(b"\x00")
    h.update(data)
    return h.hexdigest()


def order_digest(indices: Iterable[int]) -> str:
    """Digest of the ordered list of sample indices one rank reads in one epoch of a
    cyclic run (plan 3.5, the owner's ruling on the cyclic order).

    SHA-256 of the JSON list of the indices, in order, with no spaces."""
    return _sha256(json.dumps([int(i) for i in indices], separators=(",", ":")).encode("ascii"))


def param_digest(params: object) -> str:
    """Digest of a step's parameters (any JSON value), for edges and instance ids."""
    return _sha256(canonical(["ww.params.v1", params]))


# --- Fingerprints (plan 3.1, Decision 11) -------------------------------------------


def fingerprint(normalised_text: str) -> bytes:
    """128-bit fingerprint of a record's text after the fixed normalisation.

    The caller normalises first. Equal fingerprints mean equal after normalisation,
    not equal raw bytes and not the same origin; recovery compares the two texts
    directly before it attaches a record to a candidate."""
    return _blake(normalised_text.encode("utf-8"), FINGERPRINT_BYTES)


# --- Lines of a JSONL file (plan 2.3, 3.1 "Occurrence") ------------------------------

LINE_FEED = b"\n"  # the byte 0x0A


def split_lines(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """The lines of a JSONL file, from its bytes in order, in pieces of any size.

    This is the one definition of a line, used by everything that counts lines or
    gives a line its locator:

    - a line ends at a line feed byte (0x0A), and at nothing else: a carriage
      return ends no line and stays part of its line;
    - the bytes after the last line feed are a line when there is at least one of
      them, so a file whose last line has no line feed keeps that line;
    - a file of no bytes has no lines; an empty line (two line feeds in a row, or
      a line feed at the very start) is a line.

    Each line is given without its line feed. Line n, counted from 1, is the one
    ``line_locator(n)`` names."""
    held = b""
    for chunk in chunks:
        if not chunk:
            continue
        pieces = (held + chunk).split(LINE_FEED) if held else chunk.split(LINE_FEED)
        held = pieces.pop()
        yield from pieces
    if held:
        yield held


def read_lines(path: str | os.PathLike, hasher: ContentHasher | None = None) -> Iterator[bytes]:
    """Read a file once and give its lines (``split_lines``). When a hasher is
    given, every byte read goes through it, so once the last line has been given
    its digest is the file's content digest: the file is digested and split in one
    read, and the two cannot come from different bytes."""

    def chunks() -> Iterator[bytes]:
        with open(path, "rb") as f:
            while chunk := f.read(_READ_CHUNK):
                if hasher is not None:
                    hasher.update(chunk)
                yield chunk

    return split_lines(chunks())


# --- Locators and occurrence keys (plan 3.1, "Occurrence") --------------------------


def line_locator(line_number: int) -> str:
    """Locator of a record in a JSONL artifact: its 1-based line number."""
    if line_number < 1:
        raise ValueError("line numbers start at 1")
    return f"L{line_number}"


def ordinal_locator(ordinal: int) -> str:
    """Locator of a document in a tokenized shard: its 0-based ordinal."""
    if ordinal < 0:
        raise ValueError("ordinals start at 0")
    return f"D{ordinal}"


def parse_locator(locator: str) -> tuple[str, int]:
    """Split a locator into its kind ("L" or "D") and its number."""
    kind, number = locator[:1], locator[1:]
    if kind not in ("L", "D") or not number.isdigit():
        raise ValueError(f"not a locator: {locator!r}")
    return kind, int(number)


def occurrence_key(instance_id: str, locator: str) -> str:
    """The key of an occurrence: the artifact instance plus the locator."""
    return f"{instance_id}:{locator}"


# --- Edge digest and instance id (plan 3.1, Decision 29) ----------------------------

# A parent reference inside an edge record: (instance id, locator). A parent in the
# same artifact is named by its locator alone, written here with instance id None.
ParentRef = tuple[str | None, str]


def edge_record(
    child_locator: str,
    parents: Sequence[ParentRef],
    selected: ParentRef | None = None,
) -> list:
    """One of an artifact's own edge records in the form the edge digest takes.

    The child is named by its locator alone and never by an occurrence key, so the
    digest does not contain the artifact's own instance id (no cycle). For an "any"
    edge the parent the step selected is a field of its own, so two histories that
    differ only in which equivalent parent dedup kept give different digests."""
    return [
        child_locator,
        [[p[0] or "", p[1]] for p in parents],
        None if selected is None else [selected[0] or "", selected[1]],
    ]


def edge_digest(records: Iterable[list]) -> str:
    """Digest of an artifact's own edge records, independent of the order given.

    Only the edges the producing step stated as it ran belong here. A link derived
    afterwards (a recovered edge, a tokenization map) is evidence about the instance
    and never part of its identity (DECISIONS.md, B-2c)."""
    encoded = sorted(canonical(r) for r in records)
    h = hashlib.blake2b(digest_size=INSTANCE_ID_BYTES)
    h.update(b"ww.edges.v1")
    for item in encoded:
        h.update(len(item).to_bytes(8, "big"))
        h.update(item)
    return h.hexdigest()


def instance_id(
    content_digest: str,
    step_name: str,
    step_version: str,
    step_param_digest: str,
    input_instance_ids: Sequence[str],
    edges: str,
) -> str:
    """Instance id of an artifact a step produced.

    The order of computation is fixed: content digest, then edge digest, then this
    id; only then are occurrences keyed and edges stored."""
    payload = [
        "ww.instance.v1",
        content_digest,
        step_name,
        step_version,
        step_param_digest,
        list(input_instance_ids),
        edges,
    ]
    return _blake(canonical(payload), INSTANCE_ID_BYTES).hex()


def ingest_instance_id(content_digest: str, member_name: str) -> str:
    """Instance id of an ingested file, or of an import with no recorded producing
    step: its content digest and the member name it is registered under. The same
    bytes registered under two members are two instances (plan L12)."""
    payload = ["ww.instance.ingest.v1", content_digest, member_name]
    return _blake(canonical(payload), INSTANCE_ID_BYTES).hex()


# --- Artifacts that have no file: mixtures and checkpoints --------------------------
# The store is handed no file for either, so each gets a content digest over what
# it holds, and an instance id through ``instance_id`` like every artifact a step
# made. The two step names below are the tool's own.

MIXTURE_STEP = ("waterwhite.mixture", "1")
CHECKPOINT_STEP = ("waterwhite.checkpoint", "1")


def no_edges_digest() -> str:
    """The edge digest of an artifact whose step stated no edge (a shard, a
    mixture, a checkpoint)."""
    return edge_digest(())


def mixture_content_digest(shard_digests: Iterable[str]) -> str:
    """Content digest of a mixture: the content digests of its shards, in the
    trainer's order (plan Decision 27)."""
    return _sha256(canonical(["ww.mixture.v1", list(shard_digests)]))


def mixture_instance_id(
    shard_instance_ids: Iterable[str], shard_digests: Iterable[str], settings: object
) -> str:
    """Instance id of a mixture snapshot: which shard instances, in which order,
    with which content digests and which settings (any JSON value). The path the
    trainer was handed for a shard is not part of it."""
    return instance_id(
        mixture_content_digest(shard_digests),
        MIXTURE_STEP[0],
        MIXTURE_STEP[1],
        param_digest(settings),
        list(shard_instance_ids),
        no_edges_digest(),
    )


def checkpoint_params(segment: str, step: int) -> dict:
    """The parameters of the step that marks a checkpoint: what it marks."""
    return {"segment": segment, "step": int(step)}


def checkpoint_content_digest(segment: str, step: int) -> str:
    """A checkpoint is a marker at a completed step of a run segment, so its
    content digest is taken over what it marks."""
    return _sha256(canonical(["ww.checkpoint.v1", segment, int(step)]))


def checkpoint_instance_id(segment: str, step: int) -> str:
    """Instance id of the checkpoint at ``step`` of ``segment``.

    It depends on nothing else, so a caller can name a checkpoint as a segment's
    parent before the checkpoint is registered."""
    return instance_id(
        checkpoint_content_digest(segment, step),
        CHECKPOINT_STEP[0],
        CHECKPOINT_STEP[1],
        param_digest(checkpoint_params(segment, step)),
        [],
        no_edges_digest(),
    )
