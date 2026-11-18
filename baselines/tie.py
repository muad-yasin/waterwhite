"""The tie show for the Dolma toolkit's rows (plan 5.5 Check 10; table README, section
"The Dolma tie to an ordinal").

The tie is shown for one chunk file when, for every document of its shard, the shard's
tokens at ordinal n equal the tokenization of line n + 1 of the chunk file (the ordinal
counts from 0, the line from 1). This module makes that comparison and keeps its result;
nothing else reads the shard against the chunk file.

The chunk file of a shard is its tokenizer input, one ``{"id", "text"}`` row a line.
The shard is the pair ``<prefix>.bin`` and ``<prefix>.idx`` that Megatron's
preprocessing wrote. A line is tokenized by its row's ``text``, not by the raw line.
The shard holds the end-of-document token after the tokens of each document that has
any; a document that gives no tokens holds nothing.

The module starts nothing and constructs no tokenizer: the tokenizer is handed in as a
function from one text to its token ids, the way the replica's line tokenizer is. It
imports nothing of the Dolma toolkit and nothing of the tool.

When the shard holds another number of documents than the file has lines, the tie is
not shown and no document is compared, so ``first_mismatch`` is None. Otherwise the
documents are compared in ordinal order and the comparison stops at the first one that
differs, which is the one ``first_mismatch`` names.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy

from replica.errors import TokenizerError
from replica.shard_index import read_shard_index

# The end-of-document token of the build's tokenizer.
EOD_TOKEN_ID = 50256

# Token type codes of the index format, mapped to what the .bin holds. A code that is
# not here (the floating point ones) is not a token shard the show can read.
_DTYPES = {1: "<u1", 2: "<i1", 3: "<i2", 4: "<i4", 5: "<i8", 8: "<u2"}
# The code byte sits after the 9-byte header and the 8-byte version.
_CODE_OFFSET = 9 + 8


class TieError(Exception):
    """A file the show needs cannot be read: there is no result, and so no "shown"."""


def _why(error: Exception) -> str:
    """What went wrong, without the path of a file: the system's words for an OSError,
    the name of the error's class for the rest."""
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    if isinstance(error, ValueError) and not isinstance(error, UnicodeDecodeError):
        return str(error)
    return type(error).__name__


def _shard_documents(shard_prefix: Path) -> list[list[int]]:
    """The tokens of each document of a shard, in ordinal order."""
    prefix = str(shard_prefix)
    idx_path = Path(prefix + ".idx")
    bin_path = Path(prefix + ".bin")
    try:
        index = read_shard_index(idx_path)
        code = idx_path.read_bytes()[_CODE_OFFSET]
        data = bin_path.read_bytes()
    except (OSError, TokenizerError) as error:
        raise TieError(f"the shard {Path(shard_prefix).name} cannot be read: "
                       f"{_why(error)}") from error
    if code not in _DTYPES:
        raise TieError(f"the shard {Path(shard_prefix).name} has token type code {code}, "
                       f"which the show does not read")
    if len(data) != index.tokens * index.token_bytes:
        raise TieError(f"the shard {Path(shard_prefix).name} holds {len(data)} bytes in its "
                       f"token file, its index counts {index.tokens} tokens of "
                       f"{index.token_bytes} bytes")
    flat = numpy.frombuffer(data, dtype=_DTYPES[code]).tolist()
    documents, start = [], 0
    for count in index.document_tokens:
        documents.append(flat[start:start + count])
        start += count
    return documents


def _chunk_texts(tokenizer_input: Path) -> list[str]:
    """The text of each line of a chunk file, in line order."""
    try:
        content = Path(tokenizer_input).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise TieError(f"the chunk file {Path(tokenizer_input).name} cannot be read: "
                       f"{_why(error)}") from error
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    texts = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            text = row["text"]
        except (ValueError, KeyError, TypeError) as error:
            raise TieError(f"line {number} of the chunk file {Path(tokenizer_input).name} "
                           f"is not a row with a text: {error!r}") from error
        if not isinstance(text, str):
            raise TieError(f"line {number} of the chunk file {Path(tokenizer_input).name} "
                           f"has a text that is not a string")
        texts.append(text)
    return texts


def show_chunk_file(chunk_file: str, tokenizer_input: Path, shard_prefix: Path,
                    tokenizer) -> dict:
    """The result for one chunk file. ``chunk_file`` is its name as recorded, a relative
    path; ``tokenizer_input`` and ``shard_prefix`` say where its lines and its shard are
    read. The result has ``chunk_file``, ``shown``, ``documents``, ``lines`` and
    ``first_mismatch``."""
    name = PurePosixPath(chunk_file)
    if name.is_absolute() or ".." in name.parts:
        raise TieError(f"the chunk file name {chunk_file!r} is not a relative path")
    texts = _chunk_texts(tokenizer_input)
    documents = _shard_documents(shard_prefix)
    first_mismatch = None
    shown = len(documents) == len(texts)
    if shown:
        for ordinal, (text, held) in enumerate(zip(texts, documents)):
            tokens = [int(token) for token in tokenizer(text)]
            wanted = tokens + [EOD_TOKEN_ID] if tokens else []
            if held != wanted:
                shown = False
                first_mismatch = ordinal
                break
    return {
        "chunk_file": chunk_file,
        "shown": shown,
        "documents": len(documents),
        "lines": len(texts),
        "first_mismatch": first_mismatch,
    }


def show_shard(files, tokenizer) -> dict:
    """The result for the shard of a ``replica.shard_files.ShardFiles``; the chunk file
    is named by its path relative to the build folder."""
    return show_chunk_file(files.tokenizer_input_relative, files.tokenizer_input,
                           files.prefix, tokenizer)


def write_result(path: Path, results: list[dict], run: dict) -> None:
    """Write the record of a run to ``path``: the run's own words and one entry for each
    result, by chunk file. Two results for one chunk file are an error. A run or a chunk
    file name that holds a local path, a session name or the host name of this machine is
    refused with a ``ValueError`` (the refusal of ``baselines.record``), and no file is
    made."""
    # Imported here: baselines.record imports this module.
    from baselines import record

    chunk_files: dict[str, dict] = {}
    for result in results:
        name = result["chunk_file"]
        if name in chunk_files:
            raise TieError(f"two results for the chunk file {name}")
        chunk_files[name] = {
            "shown": bool(result["shown"]),
            "documents": result["documents"],
            "lines": result["lines"],
            "first_mismatch": result["first_mismatch"],
        }
    document = {"run": run, "chunk_files": chunk_files}
    record.refuse_private(document, "tie result")
    Path(path).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def read_result(path: Path) -> dict:
    """The record read back. A ``path`` that does not exist gives a record that holds no
    result; a file that is not a record is an error."""
    path = Path(path)
    if not path.exists():
        return {"run": None, "chunk_files": {}}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TieError(f"the tie record {path.name} cannot be read: {_why(error)}") from error
    if not isinstance(document, dict) or not isinstance(document.get("chunk_files"), dict):
        raise TieError(f"the tie record {path.name} holds no chunk_files table")
    return document


def is_shown(record: dict, chunk_file: str) -> bool:
    """True only for a chunk file the record holds as shown; a chunk file it does not
    hold is not shown."""
    entry = record.get("chunk_files", {}).get(chunk_file)
    return isinstance(entry, dict) and entry.get("shown") is True
