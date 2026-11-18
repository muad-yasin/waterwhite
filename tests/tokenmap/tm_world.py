"""Tiny shards and a toy tokenizer for the tests of ``waterwhite.tokenmap``.

Nothing here runs Megatron. A shard is written by hand in the real ``.bin`` and
``.idx`` layout (the layout ``waterwhite.store.artifact_files`` documents), and
the "team's tokenizer" is a toy: a text gives the bytes of its UTF-8 form, and a
document ends with the end-of-document id 256 when the run appends one. A text of
no characters gives no tokens, and then no end-of-document token either, as in
the preprocessing this stands in for.

The file is also a script: ``python tm_world.py INPUT PREFIX`` tokenizes a JSONL
file into a shard, so a test can run the toy tokenizer as a process of its own.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

EOD = 256
NORMALISER = "control-chars/1"

# The code and the struct letter of the token types the tests write.
INT32 = (4, "i")
UINT16 = (8, "H")


def toy_tokenize(text: str) -> list[int]:
    """The toy tokenizer: one token per byte of the text."""
    return list(text.encode("utf-8"))


def write_shard(prefix: Path, documents: list[list[int]], token_type=INT32) -> None:
    """Write ``<prefix>.bin`` and ``<prefix>.idx``. Each entry of ``documents`` is
    one document as the preprocessing ends it; an empty entry is a line that gave
    no tokens: a document boundary with no sequence."""
    code, letter = token_type
    size = struct.calcsize(letter)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    lengths, pointers, boundaries, tokens = [], [], [0], []
    for document in documents:
        if document:
            pointers.append(size * len(tokens))
            lengths.append(len(document))
            tokens.extend(document)
        boundaries.append(len(lengths))
    Path(f"{prefix}.bin").write_bytes(struct.pack(f"<{len(tokens)}{letter}", *tokens))
    index = b"MMIDIDX\x00\x00" + struct.pack("<QBQQ", 1, code, len(lengths), len(boundaries))
    index += struct.pack(f"<{len(lengths)}i", *lengths)
    index += struct.pack(f"<{len(pointers)}q", *pointers)
    index += struct.pack(f"<{len(boundaries)}q", *boundaries)
    Path(f"{prefix}.idx").write_bytes(index)


def documents_of(texts: list[str], *, append_eod: bool = True) -> list[list[int]]:
    """What the toy preprocessing writes for these texts, in order."""
    out = []
    for text in texts:
        tokens = toy_tokenize(text)
        if tokens and append_eod:
            tokens.append(EOD)
        out.append(tokens)
    return out


def read_texts(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def toy_preprocess(input_path: Path, prefix: Path, *, append_eod: bool = True,
                   order: list[int] | None = None, token_type=INT32) -> None:
    """Stand in for a team's tokenizer run: one document per line of the input,
    in input order. ``order`` writes the documents in another order (a list of
    line positions from 0), which is what a run that breaks plan F1 does."""
    documents = documents_of(read_texts(Path(input_path)), append_eod=append_eod)
    if order is not None:
        documents = [documents[i] for i in order]
    write_shard(Path(prefix), documents, token_type)


def write_rows(path: Path, texts: list[str], *, id_prefix: str = "U") -> Path:
    """A tokenizer-input file: rows {"id", "text"}, one per text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"id": f"{id_prefix}-{n:03d}", "text": text})
            for n, text in enumerate(texts, start=1)]
    path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")
    return path


def write_raw_lines(path: Path, lines: list[str]) -> Path:
    """A file of raw lines, readable or not, as a sanitizer reads them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(line + "\n" for line in lines).encode("utf-8"))
    return path


if __name__ == "__main__":
    toy_preprocess(Path(sys.argv[1]), Path(sys.argv[2]))
