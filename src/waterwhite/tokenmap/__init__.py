"""Tokenization maps: which input line each document of a tokenized shard came
from (plan 3.3 "Tokenization", 3.4, Decision 4, F1; Checks 3 and 6 of plan 5.5).

A team's tokenizer keeps the text of each line and drops everything else, and
the shard's index stores document boundaries only. So the link from a document
to its record is never observed: it is derived, by position or by token
matching, verified, and labelled by what the verification showed.

Three entry points:

- ``tokenize_shard`` is the capture wrapper. It runs the team's tokenizer,
  unchanged, through a function the pipeline hands it, registers the shard and
  writes its capture map.
- ``map_tokenized_shard`` does the same for a shard whose tokenizer run has just
  ended, for a pipeline that tokenizes many shards in one go.
- ``recover_map`` writes the map of a run made without the wrapper, into the
  recovery namespace, with the skips of a sanitizer's log as drops.

The tool ships no tokenizer and imports none. The verification tokenizes each
line again through a ``tokenize(text)`` function the caller supplies:
``command_tokenizer`` gives one that asks an external command, and
``entry_point_tokenizer`` one that the team names as ``module:function``.

docs/tokenmap.md describes the package in full.
"""

from ._derive import (
    Assignment,
    DocumentClasses,
    Retokenizer,
    Tokenize,
    Verification,
    check_spans,
    derive_position,
    derive_token_matching,
)
from ._inputs import (
    SANITIZER_REASONS,
    STEP_SANITIZE,
    STEP_TOKENIZE,
    SanitizerRun,
    SanitizerSkip,
    TokenizerSettings,
    read_sanitizer_log,
    tokenize_step,
)
from ._shard import ShardTokens
from ._tokenizers import (
    DEFAULT_TOKENIZER_TIMEOUT,
    TOKENIZER_PROTOCOL,
    CommandTokenizer,
    command_tokenizer,
    entry_point_tokenizer,
)
from ._wrapper import (
    MapOutcome,
    ShardMaps,
    map_tokenized_shard,
    recover_map,
    tokenize_shard,
)

__all__ = [
    "DEFAULT_TOKENIZER_TIMEOUT",
    "SANITIZER_REASONS",
    "STEP_SANITIZE",
    "STEP_TOKENIZE",
    "TOKENIZER_PROTOCOL",
    "Assignment",
    "CommandTokenizer",
    "DocumentClasses",
    "MapOutcome",
    "Retokenizer",
    "SanitizerRun",
    "SanitizerSkip",
    "ShardMaps",
    "ShardTokens",
    "Tokenize",
    "TokenizerSettings",
    "Verification",
    "check_spans",
    "command_tokenizer",
    "derive_position",
    "derive_token_matching",
    "entry_point_tokenizer",
    "map_tokenized_shard",
    "read_sanitizer_log",
    "recover_map",
    "tokenize_shard",
    "tokenize_step",
]
