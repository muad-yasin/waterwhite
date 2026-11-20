"""Reading an artifact on disk: what the store compares a declared record with.

Two formats (plan 2.3, 5.7): a JSONL file, and a tokenized shard, which is a pair
of files ``<prefix>.bin`` and ``<prefix>.idx`` in Megatron's indexed-dataset layout.

The index file is read here from its documented layout. No Megatron code is
imported or copied (plan 3.5). The layout, little-endian throughout:

    9 bytes   the magic ``MMIDIDX`` followed by two zero bytes
    8 bytes   unsigned version, 1
    1 byte    the code of the token dtype of the .bin file
    8 bytes   unsigned S, the number of sequences
    8 bytes   unsigned D, the number of entries of the document index
    4 * S     int32, the token count of each sequence
    8 * S     int64, the byte offset of each sequence in the .bin file
    8 * D     int64, the document index: entry 0 is 0, entry k is the number of
              sequences written when document k ended
    1 * S     int8 sequence modes, present for a multimodal dataset only

What counts as a document here: the preprocessing writes one sequence per input
line that gives tokens, and for a line that gives no tokens it writes a document
boundary with no sequence. The trainer reads sequences. So the document count of a
shard is the number of documents that hold a sequence, and a boundary with no
sequence adds no ordinal (plan 3.3 position mode, F1). A document with more than
one sequence would make the trainer's numbering part from the documents', and such
a shard is refused as an unsupported format.

The tokens of the ``.bin`` file are read here too, one document at a time
(``ShardTokens``), so the shard format has one reader in the tool. Where a
document lies in the ``.bin`` file is computed from the token counts of the
index: the sequences are written one after the other in index order, so sequence
k starts where the sequences before it end. The ``.bin`` file must have exactly
the size the index gives; a shard that does not is refused as an unsupported
format. The byte offsets the index stores are not read.

Plan sections: 3.4 (the declared count and digest are checked against the artifact
on disk), 3.3 (tokenization), 2.3 and 5.7 (formats).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import ids
from ..errors import UnsupportedFormat

INDEX_MAGIC = b"MMIDIDX\x00\x00"
INDEX_VERSION = 1
_HEADER = struct.Struct("<9sQBQQ")  # magic, version, dtype code, S, D: 34 bytes
_DTYPE_CODES = frozenset(range(1, 9))  # the codes the layout defines
# The size in bytes of one item, by the dtype code (as Megatron's DType.size gives it,
# checked against core_v0.20.0): uint8, int8, int16, int32, int64, float64, float32, uint16.
_ITEM_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}
_READ_CHUNK = 1 << 20  # bytes passed over at a time when they are only digested

# The token type of a .bin file, by the code its index names, as the layout
# documents it. Codes 6 and 7 are floating-point types and hold no token ids.
_TOKEN_TYPES = {1: "u1", 2: "i1", 3: "<i2", 4: "<i4", 5: "<i8", 8: "<u2"}


def token_dtype(dtype_code: int) -> np.dtype:
    """The numpy type of the tokens of a ``.bin`` file whose index names this
    code. A code that is not a whole-number type (the two floating-point codes,
    or a code the layout does not define) is refused as an unsupported format."""
    if dtype_code not in _TOKEN_TYPES:
        raise UnsupportedFormat(
            f"the token type code {dtype_code} holds no token ids"
        )
    return np.dtype(_TOKEN_TYPES[dtype_code])


@dataclass(frozen=True)
class JsonlFile:
    """A JSONL file as found on disk."""

    line_count: int
    digest: str


def read_jsonl(path: str | os.PathLike) -> JsonlFile:
    """Line count and content digest of a JSONL file, taken in one read.

    The lines are the lines of ``ids.split_lines``, the one definition the capture
    wrappers use too, so the count a wrapper declares and the count the store
    finds are counts of the same thing. The digest is taken over the bytes the
    lines were split from and is the value of ``ids.file_digest``."""
    hasher = ids.ContentHasher()
    count = 0
    for _ in ids.read_lines(path, hasher):
        count += 1
    return JsonlFile(line_count=count, digest=hasher.hexdigest())


@dataclass(frozen=True)
class ShardIndex:
    """The parts of a shard's index file the tool reads."""

    dtype_code: int
    sequence_lengths: np.ndarray  # int32, tokens per sequence
    document_index: np.ndarray  # int64, D entries, entry 0 is 0

    @property
    def sequence_count(self) -> int:
        return int(self.sequence_lengths.shape[0])

    @property
    def document_count(self) -> int:
        """The number of documents that hold tokens: the number of ordinals."""
        return self.sequence_count

    @property
    def empty_documents(self) -> int:
        """Document boundaries with no sequence (lines that gave no tokens)."""
        return int(self.document_index.shape[0]) - 1 - self.sequence_count

    @property
    def token_count(self) -> int:
        return int(self.sequence_lengths.sum(dtype=np.int64))

    def token_starts(self) -> np.ndarray:
        """Where each sequence starts in the shard's token stream, with one more
        entry for the end of the last one: sequence k spans ``starts[k]`` to
        ``starts[k + 1]`` (int64, S + 1 entries)."""
        starts = np.zeros(self.sequence_count + 1, dtype=np.int64)
        np.cumsum(self.sequence_lengths.astype(np.int64), out=starts[1:])
        return starts


def shard_paths(prefix: str | os.PathLike) -> tuple[Path, Path]:
    """The .bin and .idx paths of a shard prefix."""
    text = os.fspath(prefix)
    return Path(text + ".bin"), Path(text + ".idx")


def read_shard_index(
    idx_path: str | os.PathLike, hasher: ids.ContentHasher | None = None
) -> ShardIndex:
    """Read an index file. Raises ``UnsupportedFormat`` for anything that is not a
    version-1 index with one sequence per non-empty document.

    When a hasher is given, every byte of the file goes through it in this one
    read, whether the file can be read as an index or not. So once the call has
    returned, or has raised ``UnsupportedFormat``, the hasher's digest is the file's
    content digest (the value of ``ids.file_digest``), and the index and the digest
    cannot come from different bytes."""
    where = os.fspath(idx_path)
    with open(idx_path, "rb") as f:
        size = os.fstat(f.fileno()).st_size

        def take(count: int) -> bytes:
            data = f.read(count)
            if hasher is not None:
                hasher.update(data)
            return data

        def pass_over(count: int | None) -> None:
            """Bytes the index reader does not need: ``count`` of them, or with
            None the rest of the file. With no hasher they are not read."""
            if hasher is None:
                if count is not None:
                    f.seek(count, os.SEEK_CUR)
                return
            while count is None or count > 0:
                data = take(_READ_CHUNK if count is None else min(count, _READ_CHUNK))
                if not data:
                    return
                if count is not None:
                    count -= len(data)

        try:
            index = _index_from(take, size, where)
        except UnsupportedFormat:
            pass_over(None)
            raise
        pass_over(None)
    return index


def _array(data: bytes, dtype: str, count: int, where: str) -> np.ndarray:
    if len(data) != count * np.dtype(dtype).itemsize:
        raise UnsupportedFormat(f"index file ended inside an array: {where}")
    return np.frombuffer(data, dtype=dtype)


def _check_offsets(lengths: np.ndarray, offsets: np.ndarray, code: int, where: str) -> None:
    """Megatron reads a sequence at the byte offset the index stores (core_v0.20.0,
    ``indexed_dataset.py``: ``__getitem__`` and ``get`` read at ``sequence_pointers``),
    while Waterwhite takes a sequence's place from the lengths. For the contiguous
    layout it supports, the offset of sequence i is the item size times the lengths
    before it; an index whose stored offsets differ would show Waterwhite one document
    and Megatron another, so it is refused."""
    expected = np.zeros(lengths.size, dtype=np.int64)
    if lengths.size > 1:
        np.cumsum(lengths[:-1].astype(np.int64), out=expected[1:])
    expected *= _ITEM_SIZES[code]
    wrong = np.flatnonzero(offsets != expected)
    if wrong.size:
        first = int(wrong[0])
        raise UnsupportedFormat(
            f"the byte offset of sequence {first} in the index is {int(offsets[first])}, "
            f"the lengths give {int(expected[first])}: Megatron reads the stored offsets: {where}"
        )


def _index_from(take, size: int, where: str) -> ShardIndex:
    """The index, from the bytes ``take`` gives in file order."""
    head = take(_HEADER.size)
    if len(head) != _HEADER.size:
        raise UnsupportedFormat(f"index file too short for its header: {where}")
    magic, version, code, sequences, entries = _HEADER.unpack(head)
    if magic != INDEX_MAGIC:
        raise UnsupportedFormat(f"not a Megatron index file (bad magic): {where}")
    if version != INDEX_VERSION:
        raise UnsupportedFormat(f"index file version {version}, expected 1: {where}")
    if code not in _DTYPE_CODES:
        raise UnsupportedFormat(f"index file names an unknown dtype code {code}: {where}")
    needed = _HEADER.size + 12 * sequences + 8 * entries
    if size < needed:
        raise UnsupportedFormat(
            f"index file holds {size} bytes, its header needs {needed}: {where}"
        )
    lengths = _array(take(4 * sequences), "<i4", sequences, where)
    if sequences and int(lengths.min()) < 0:
        raise UnsupportedFormat(f"the index holds a sequence of negative length: {where}")
    offsets = _array(take(8 * sequences), "<i8", sequences, where)
    _check_offsets(lengths, offsets, code, where)
    documents = _array(take(8 * entries), "<i8", entries, where)
    if entries < 1 or int(documents[0]) != 0:
        raise UnsupportedFormat(f"document index does not start at 0: {where}")
    if int(documents[-1]) != sequences:
        raise UnsupportedFormat(
            f"document index ends at sequence {int(documents[-1])} of {sequences}: {where}"
        )
    steps = np.diff(documents)
    if steps.size and (int(steps.min()) < 0 or int(steps.max()) > 1):
        raise UnsupportedFormat(
            "a document of the shard holds more than one sequence, or the document "
            f"index runs backwards: {where}"
        )
    return ShardIndex(dtype_code=code, sequence_lengths=lengths, document_index=documents)


@dataclass(frozen=True)
class ShardFiles:
    """A tokenized shard as found on disk."""

    document_count: int
    digest: str  # ids.shard_digest of the two file digests
    bin_digest: str
    idx_digest: str


def check_bin_size(index: ShardIndex, bin_path: Path) -> None:
    """The ``.bin`` file is as long as its index says: the tokens of every sequence, each
    the size of the index's token type. The token reader holds a shard to this, so the
    registration and the integrity check hold it to the same size: a pair that the reader
    refuses is no shard to register, to give a map or to answer a negative from."""
    needed = int(index.token_starts()[-1]) * _ITEM_SIZES[index.dtype_code]
    size = os.path.getsize(bin_path)
    if size != needed:
        raise UnsupportedFormat(
            f"the .bin file holds {size} bytes and its index gives {needed}: {bin_path}"
        )


def read_shard(prefix: str | os.PathLike) -> ShardFiles:
    """Document count and content digest of the shard at ``prefix``."""
    bin_path, idx_path = shard_paths(prefix)
    index = read_shard_index(idx_path)
    check_bin_size(index, bin_path)
    bin_digest = ids.file_digest(bin_path)
    idx_digest = ids.file_digest(idx_path)
    return ShardFiles(
        document_count=index.document_count,
        digest=ids.shard_digest(bin_digest, idx_digest),
        bin_digest=bin_digest,
        idx_digest=idx_digest,
    )


def shard_digest_only(prefix: str | os.PathLike) -> str:
    """The content digest of a shard without reading its index as an index."""
    bin_path, idx_path = shard_paths(prefix)
    return ids.shard_digest(ids.file_digest(bin_path), ids.file_digest(idx_path))


@dataclass(frozen=True)
class ShardOnDisk:
    """A shard whose two files were each read once: what a loud check compares.

    ``digest`` is the content digest of the pair, whatever the index holds.
    ``document_count`` is the number of documents the index holds; it is None,
    with ``unreadable`` saying why, when the index cannot be read as an index."""

    digest: str
    document_count: int | None
    unreadable: str | None = None


def read_shard_once(prefix: str | os.PathLike) -> ShardOnDisk:
    """Content digest and document count of the shard at ``prefix``, each file read
    once: the index file is digested and read as an index in the same read. An
    index that cannot be read as an index, or whose ``.bin`` is not the size it gives, is
    not an error here; the digest still says whether the pair is the registered one."""
    bin_path, idx_path = shard_paths(prefix)
    hasher = ids.ContentHasher()
    count, unreadable = None, None
    try:
        index = read_shard_index(idx_path, hasher)
        check_bin_size(index, bin_path)
        count = index.document_count
    except UnsupportedFormat as exc:
        unreadable = exc.detail
    digest = ids.shard_digest(ids.file_digest(bin_path), hasher.hexdigest())
    return ShardOnDisk(digest=digest, document_count=count, unreadable=unreadable)


class ShardTokens:
    """The documents of the shard at ``prefix``, read one span at a time.

    ``index`` is the shard's index. ``starts`` holds, for each ordinal, where its
    tokens start in the shard's token stream, with one more entry for the end of
    the last document, so ordinal k spans ``starts[k]`` to ``starts[k + 1]``.
    Nothing here holds more than one document's tokens at a time: a document is
    read from its own byte range and handed back. Use it as a context manager,
    or call ``close``."""

    def __init__(self, prefix: str | os.PathLike) -> None:
        self.bin_path, self.idx_path = shard_paths(prefix)
        self.index = read_shard_index(self.idx_path)
        try:
            self._dtype = token_dtype(self.index.dtype_code)
        except UnsupportedFormat as error:
            raise UnsupportedFormat(f"{error.detail}: {self.idx_path}") from None
        if self.index.sequence_count and int(self.index.sequence_lengths.min()) < 1:
            raise UnsupportedFormat(
                "the index holds a sequence of no tokens; a line that gives no tokens "
                f"leaves a document boundary with no sequence: {self.idx_path}"
            )
        self.starts = self.index.token_starts()
        check_bin_size(self.index, self.bin_path)
        self._file = None

    @property
    def count(self) -> int:
        """The number of documents that hold tokens: the number of ordinals."""
        return self.index.document_count

    @property
    def token_count(self) -> int:
        return int(self.starts[-1])

    def span(self, ordinal: int) -> tuple[int, int]:
        """The half-open token span of one document in the shard's token stream."""
        return int(self.starts[ordinal]), int(self.starts[ordinal + 1])

    def read(self, ordinal: int) -> np.ndarray:
        """The tokens of one document, as an int64 array."""
        if not 0 <= ordinal < self.count:
            raise IndexError(f"no document {ordinal} in a shard of {self.count}")
        start, end = self.span(ordinal)
        if self._file is None:
            self._file = open(self.bin_path, "rb")
        size = self._dtype.itemsize
        self._file.seek(start * size)
        data = self._file.read((end - start) * size)
        if len(data) != (end - start) * size:
            raise UnsupportedFormat(
                f"the .bin file ended inside document {ordinal}: {self.bin_path}"
            )
        return np.frombuffer(data, dtype=self._dtype).astype(np.int64)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ShardTokens":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
