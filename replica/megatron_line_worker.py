"""The build's tokenizer, one text at a time. A script for the Megatron environment.

``replica.megatron_tokenizer.LineTokenizer`` runs this file with the interpreter
of the test-time environment. The replica itself never imports it, because it
imports Megatron.

    python -I megatron_line_worker.py CHECKOUT TYPE VOCAB_FILE MERGES_FILE

It builds the tokenizer as ``tools/preprocess_data.py`` of the checkout does: the
same argument parser, and the same four values set after parsing, which size a
padded vocabulary and change no token. Then it answers one line per line: a JSON
text in, the JSON list of its token ids out, with no end-of-document token. A
text that gives no tokens gives the empty list.

The replies leave on a descriptor of their own, and standard output is pointed at
standard error, so nothing a library prints can be taken for a reply.
"""

import argparse
import json
import os
import sys


def main() -> None:
    replies = os.fdopen(os.dup(1), "w", encoding="ascii")
    os.dup2(2, 1)
    checkout, kind, vocab, merges = sys.argv[1:5]
    sys.path.append(checkout)
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
    from megatron.training.arguments import _add_tokenizer_args

    parser = argparse.ArgumentParser(add_help=False)
    _add_tokenizer_args(parser)
    args = parser.parse_args(
        ["--tokenizer-type", kind, "--vocab-file", vocab, "--merge-file", merges])
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    tokenizer = build_tokenizer(args)
    replies.write(json.dumps({"ready": True, "eod": int(tokenizer.eod)}) + "\n")
    replies.flush()
    for line in sys.stdin:
        ids = [int(token) for token in tokenizer.tokenize(json.loads(line))]
        replies.write(json.dumps(ids) + "\n")
        replies.flush()


if __name__ == "__main__":
    main()
