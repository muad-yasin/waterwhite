"""F1 probe: compare a tokenizer-input JSONL file with the shard Megatron's
preprocessing made from it, and report whether there is one document per input
line, in input order, and what the end-of-document convention is. Config and
output: docs/oracle.md, "F1 probe".

The shard is read with Megatron's ``IndexedDataset`` and each line is tokenized
with the tokenizer Megatron's ``build_tokenizer`` builds, by the call the
preprocessing script makes (``tokenizer.tokenize(text)``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import (  # noqa: E402
    TokenizerSpec,
    load_json,
    new_output_path,
    parse_tokenizer,
    resolve_path,
    write_json,
)
from oracle.errors import ConfigError  # noqa: E402

PROBE_FORMAT = "waterwhite.oracle-f1-probe/1"


def read_texts(path: Path, text_field: str) -> list[str]:
    """The text of every line of a JSONL file, in order. Every line must be a JSON
    object with the text field; the preprocessing script reads the file the same
    way and stops on a line that is not."""
    texts = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            try:
                row = json.loads(line)
                text = row[text_field]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ConfigError(f"{path.name} line {number}: no JSON object with {text_field!r}: {e}") from e
            if not isinstance(text, str):
                raise ConfigError(f"{path.name} line {number}: {text_field!r} is not a string")
            texts.append(text)
    return texts


def probe(input_path: Path, shard_prefix: Path, tokenizer: TokenizerSpec, text_field: str = "text") -> dict:
    """Compare the lines of ``input_path`` with the documents of the shard.

    Document i of the shard is read as Megatron reads it, through the pointers and
    lengths of the ``.idx`` file, so its tokens are its span in the index. It is
    compared with the tokens of line i twice: followed by the tokenizer's
    end-of-document token, and without it."""
    from oracle.megatron_data import build_megatron_tokenizer, open_shard  # imports Megatron

    megatron_tokenizer = build_megatron_tokenizer(tokenizer)
    eod = int(megatron_tokenizer.eod)
    texts = read_texts(input_path, text_field)
    expected = [[int(t) for t in megatron_tokenizer.tokenize(text)] for text in texts]
    shard = open_shard(shard_prefix)
    sequences = len(shard)
    boundaries = [int(x) for x in shard.document_indices]
    documents = len(boundaries) - 1
    stored = [[int(t) for t in shard[i]] for i in range(sequences)]

    zero_token_lines = [i + 1 for i, ids in enumerate(expected) if not ids]
    compared = min(len(expected), sequences)
    with_eod = sum(stored[i] == expected[i] + [eod] for i in range(compared))
    without_eod = sum(stored[i] == expected[i] for i in range(compared))
    first_mismatch = next(
        (
            i
            for i in range(compared)
            if stored[i] != expected[i] + [eod] and stored[i] != expected[i]
        ),
        None,
    )

    counts_equal = len(expected) == sequences == documents
    one_sequence_per_document = boundaries == list(range(documents + 1))
    one_document_per_line = counts_equal and one_sequence_per_document and not zero_token_lines
    if counts_equal and with_eod == sequences and sequences > 0:
        eod_in_span = True
    elif counts_equal and without_eod == sequences and sequences > 0:
        eod_in_span = False
    else:
        eod_in_span = None
    input_order = eod_in_span is not None

    another_order = None
    if not input_order:
        have = sorted(map(tuple, stored))
        another_order = have == sorted(tuple(e + [eod]) for e in expected if e) or have == sorted(
            tuple(e) for e in expected if e
        )

    return {
        "format": PROBE_FORMAT,
        "input": input_path.name,
        "shard": shard_prefix.name,
        "lines": len(expected),
        "sequences": sequences,
        "documents": documents,
        "tokens": sum(len(s) for s in stored),
        "zero_token_lines": zero_token_lines,
        "documents_equal_to_line_plus_eod": with_eod,
        "documents_equal_to_line_alone": without_eod,
        "first_line_that_matches_neither": None if first_mismatch is None else first_mismatch + 1,
        "one_document_per_line": one_document_per_line,
        "input_order": input_order,
        "same_documents_in_another_order": another_order,
        "eod": {
            "token_id": eod,
            # true: every document's span in the .idx ends with the EOD token;
            # false: no document's span holds it; null: no single convention.
            "in_span": eod_in_span,
        },
        "f1_holds": one_document_per_line and input_order,
    }


def work(config_path: str) -> dict:
    raw, base = load_json(config_path)
    unknown = sorted(set(raw) - {"input", "shard", "tokenizer", "text_field", "outputs"})
    if unknown:
        raise ConfigError(f"unknown key(s) {unknown} in the config")
    for key in ("input", "shard", "tokenizer"):
        if key not in raw:
            raise ConfigError(f"missing key {key!r}")
    input_path = resolve_path(raw["input"], base)
    shard_prefix = resolve_path(raw["shard"], base)
    if not input_path.is_file():
        raise ConfigError(f"input is not a file: {input_path}")
    for suffix in (".bin", ".idx"):
        if not Path(str(shard_prefix) + suffix).is_file():
            raise ConfigError(f"no file {shard_prefix}{suffix}")
    tokenizer = parse_tokenizer(raw["tokenizer"], base)
    text_field = raw.get("text_field", "text")
    out = None
    outputs = raw.get("outputs", {})
    if not isinstance(outputs, dict) or set(outputs) - {"report"}:
        raise ConfigError("'outputs' may hold 'report' only")
    if outputs.get("report") is not None:
        out = resolve_path(outputs["report"], base)
        new_output_path(out)
    result = probe(input_path, shard_prefix, tokenizer, text_field)
    if out is not None:
        write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
