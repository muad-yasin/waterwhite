"""The documents of a tokenized shard, as a map derivation reads them.

The shard format has one reader in the tool, ``waterwhite.store.artifact_files``:
it reads the index, names the token type of a dtype code and fetches the tokens
of one document from the ``.bin`` file (``ShardTokens``). This module names that
reader for the package and adds the one thing a map derivation needs beside it:
a digest of a token sequence, to find documents that hold the same tokens
without keeping them.

Plan sections: 3.3 "Tokenization" (the map stores a token start and a token end
per ordinal; identical twins), 2.3 and 5.7 (formats).
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..store.artifact_files import ShardTokens

__all__ = ["DIGEST_BYTES", "ShardTokens", "token_digest"]

DIGEST_BYTES = 16


def token_digest(tokens: np.ndarray) -> bytes:
    """A digest of one token sequence (an int64 array), used to group equal
    sequences without keeping them. Two sequences are called equal only after
    their tokens were compared directly; the digest only finds the pairs."""
    data = np.ascontiguousarray(tokens, dtype="<i8").tobytes()
    return hashlib.blake2b(data, digest_size=DIGEST_BYTES).digest()
