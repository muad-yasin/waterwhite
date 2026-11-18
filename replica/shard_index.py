"""Read the index file of a tokenized shard, to count its documents and tokens.

This is the replica's own reader of the ``.idx`` file that Megatron's
preprocessing writes (version 1 of the format): a 9-byte header, the version, the
token type code, the sequence count, the document count, then the sequence
lengths, the sequence pointers and the document indices. It imports nothing from
Megatron and refuses a file it does not recognise.

The build uses it for one gate: a shard must hold one document per line of its
tokenizer input (plan F1).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy

from .errors import TokenizerError

_HEADER = b"MMIDIDX\x00\x00"
# Token type codes of the index format, with the bytes one token takes in the .bin.
_TOKEN_BYTES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 8, 7: 4, 8: 2}


@dataclass(frozen=True)
class ShardIndex:
    """The counts of one shard. ``document_tokens[n]`` is the token count of
    document n, the end-of-document token included when the run appended one."""

    documents: int
    tokens: int
    document_tokens: tuple[int, ...]
    token_bytes: int


def read_shard_index(idx_path: Path) -> ShardIndex:
    data = Path(idx_path).read_bytes()
    fixed = len(_HEADER) + 8 + 1 + 8 + 8
    if len(data) < fixed or data[:len(_HEADER)] != _HEADER:
        raise TokenizerError(f"{idx_path}: not a shard index file")
    version, code, sequences, entries = struct.unpack_from("<QBQQ", data, len(_HEADER))
    if version != 1:
        raise TokenizerError(f"{idx_path}: index format version {version}, expected 1")
    if code not in _TOKEN_BYTES:
        raise TokenizerError(f"{idx_path}: unknown token type code {code}")
    need = fixed + sequences * 4 + sequences * 8 + entries * 8
    if len(data) < need:
        raise TokenizerError(f"{idx_path}: file is shorter than its header says")
    lengths = numpy.frombuffer(data, dtype="<i4", count=sequences, offset=fixed)
    marks = numpy.frombuffer(data, dtype="<i8", count=entries, offset=fixed + sequences * 12)
    if entries < 1 or int(marks[0]) != 0 or int(marks[-1]) != sequences:
        raise TokenizerError(f"{idx_path}: document indices do not span the sequences")
    cumulative = numpy.concatenate(([0], numpy.cumsum(lengths, dtype=numpy.int64)))
    per_document = cumulative[marks[1:]] - cumulative[marks[:-1]]
    return ShardIndex(
        documents=entries - 1,
        tokens=int(lengths.sum(dtype=numpy.int64)),
        document_tokens=tuple(int(n) for n in per_document),
        token_bytes=_TOKEN_BYTES[code],
    )
