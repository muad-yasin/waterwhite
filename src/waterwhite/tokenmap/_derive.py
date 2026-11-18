"""The two ways a shard's documents are tied to the lines they were tokenized
from, and the verification of each (plan 3.3 "Tokenization", Decision 4, F1).

Nothing here writes to the store. Each mode reads the input lines once, in
order, and the shard's documents one span at a time, and returns an
``Assignment``: which line each ordinal was given, which lines gave no tokens,
and what the verification found.

How a line that gives no tokens shows in a shard (plan F1). The preprocessing
ends a document for every input line. A line with tokens adds one sequence, and
the index's document list moves on by one. A line with no tokens adds no
sequence: the document list gets one more entry with the same value as the one
before it. Such an entry is a document nobody can fetch and it has no ordinal.
So the document list has one step per input line, a step of one for a line that
became a document and a step of nothing for a line that gave no tokens, and
position mode reads off which line was which.

Position mode: walk the input lines and the steps of the document list together.
A step of one gives the line the next ordinal. A step of nothing makes the line a
"no tokens" drop, and the next line still gets the next ordinal.

Token-matching mode: tokenize every input line again and look for the documents
that hold exactly those tokens. No assumption about order is needed.

Verification, for both modes: every ordinal's span has the size the shard's own
index records; in position mode the number of steps equals the number of
surviving input lines; and every document's tokens equal the tokens its line
gives when it is tokenized again. Token matching also requires that the lines
with tokens and the documents are the same collection of token sequences: no
document without a line, no line without a document, and as many lines as
documents for each sequence that occurs more than once.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from .. import vocabulary as v
from ..errors import InvalidRecord
from ..store.artifact_files import ShardIndex
from ._inputs import TokenizerSettings
from ._shard import ShardTokens, token_digest

Tokenize = Callable[[str], Sequence[int]]
LineSource = Callable[[], Iterator[tuple[int, bytes]]]

REASON_LIMIT = 5  # how many findings of a verification are kept in words


@dataclass(frozen=True)
class Verification:
    """What the verification of one map found (plan 3.3).

    ``result`` is "passed", "failed" or "not run". It is "not run" when nothing
    failed and no ``tokenize`` function was given, so no document was compared.
    ``count_matches`` is None in token-matching mode, which counts no lines.
    ``failures`` counts every finding; ``reasons`` keeps the first few in words."""

    result: str
    spans_match: bool
    count_matches: bool | None
    documents_compared: int
    failures: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, eq=False)
class Assignment:
    """One mode's result for one shard.

    ``line_of[k]`` is the number of the input line ordinal k was given, or 0 when
    it was given none. ``starts`` and ``ends`` are the documents' spans in the
    shard's token stream. ``no_token_lines`` are the surviving lines that gave no
    tokens. ``surviving_lines`` counts the input lines the sanitizer's log does
    not name."""

    derivation: str
    starts: np.ndarray
    ends: np.ndarray
    line_of: np.ndarray
    no_token_lines: tuple[int, ...]
    surviving_lines: int
    verification: Verification


class _Findings:
    def __init__(self) -> None:
        self.count = 0
        self.reasons: list[str] = []

    def add(self, reason: str) -> None:
        self.count += 1
        if len(self.reasons) < REASON_LIMIT:
            self.reasons.append(reason)


# --- Tokens ---------------------------------------------------------------------------


def token_array(values: object) -> np.ndarray:
    """What a ``tokenize`` function returned, as an int64 array. Anything that is
    not a flat sequence of whole numbers is refused."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise InvalidRecord(f"the tokenize function returned no token sequence: {error}") from None
    if array.ndim != 1:
        raise InvalidRecord(
            f"the tokenize function returned something of {array.ndim} dimensions; "
            "a text gives one flat sequence of token ids"
        )
    if array.size == 0:
        return np.zeros(0, dtype=np.int64)
    if array.dtype.kind not in "iu":
        raise InvalidRecord(
            f"the tokenize function returned values of type {array.dtype}; token ids are "
            "whole numbers"
        )
    return array.astype(np.int64)


def same_tokens(document: np.ndarray, expected: np.ndarray) -> bool:
    """The token comparison of the verification: the same ids in the same order."""
    return document.shape == expected.shape and bool(np.array_equal(document, expected))


class Retokenizer:
    """Gives, for a raw input line, the tokens its document must hold: the line's
    text through the caller's ``tokenize`` function, followed by the
    end-of-document token when the convention says a document's span holds it. A
    line that gives no tokens gets no end-of-document token either, as in the
    preprocessing, so it stays a line with no tokens. None for a line that holds
    no text."""

    def __init__(
        self,
        tokenize: Tokenize,
        settings: TokenizerSettings,
        text_of: Callable[[bytes], str | None],
    ) -> None:
        self._tokenize = tokenize
        self._text_of = text_of
        self._eod = (
            np.array([settings.eod_token_id], dtype=np.int64) if settings.eod_in_span else None
        )

    def __call__(self, raw: bytes) -> np.ndarray | None:
        text = self._text_of(raw)
        if text is None:
            return None
        tokens = token_array(self._tokenize(text))
        if tokens.size and self._eod is not None:
            tokens = np.concatenate((tokens, self._eod))
        return tokens


# --- Documents with equal tokens --------------------------------------------------------


class DocumentClasses:
    """Which documents of a shard hold the same tokens (identical twins).

    Built in one pass over the shard. A digest of each document's tokens finds the
    pairs, and two documents are put in one class only after their tokens were
    compared directly, so a digest collision cannot make twins. Two documents with
    one digest and different tokens are kept in ``collisions`` and fail the
    verification of every map of the shard."""

    def __init__(self, shard: ShardTokens) -> None:
        self.first = np.arange(shard.count, dtype=np.int64)
        self.by_digest: dict[bytes, int] = {}
        self.collisions: list[tuple[int, int]] = []
        self._twins: dict[int, list[int]] = {}
        for ordinal in range(shard.count):
            tokens = shard.read(ordinal)
            digest = token_digest(tokens)
            held = self.by_digest.get(digest)
            if held is None:
                self.by_digest[digest] = ordinal
                continue
            if not same_tokens(shard.read(held), tokens):
                self.collisions.append((held, ordinal))
                continue
            self.first[ordinal] = held
            self._twins.setdefault(held, [held]).append(ordinal)

    def members(self, ordinal: int) -> tuple[int, ...]:
        """Every ordinal whose tokens equal this document's, itself included, in
        ordinal order. One entry for a document with no twin."""
        first = int(self.first[ordinal])
        return tuple(self._twins.get(first, (first,)))


# --- The checks every mode shares --------------------------------------------------------


def check_spans(starts: np.ndarray, ends: np.ndarray, index: ShardIndex) -> str | None:
    """Hold a map's spans against the shard's own index: every ordinal's span has
    the size the index records for it, and the spans follow one another from
    token 0 with no gap, so that they end at the index's total. Returns the first
    difference in words, or None."""
    lengths = index.sequence_lengths.astype(np.int64)
    if starts.shape != lengths.shape or ends.shape != lengths.shape:
        return f"{starts.shape[0]} spans for the {lengths.shape[0]} documents of the index"
    sizes = ends - starts
    wrong = np.flatnonzero(sizes != lengths)
    if wrong.size:
        k = int(wrong[0])
        return (
            f"document D{k} spans {int(sizes[k])} tokens and the index records "
            f"{int(lengths[k])} ({wrong.size} documents differ)"
        )
    if lengths.size:
        if int(starts[0]) != 0:
            return f"the first span starts at token {int(starts[0])}, not 0"
        gaps = np.flatnonzero(starts[1:] != ends[:-1])
        if gaps.size:
            k = int(gaps[0]) + 1
            return f"document D{k} starts at token {int(starts[k])}, the one before it ends at {int(ends[k - 1])}"
    return None


def _finish(
    derivation: str,
    shard: ShardTokens,
    classes: DocumentClasses | None,
    line_of: np.ndarray,
    no_token_lines: list[int],
    surviving: int,
    found: _Findings,
    compared: int,
    count_matches: bool | None,
    ran: bool,
) -> Assignment:
    starts, ends = shard.starts[:-1], shard.starts[1:]
    span_fault = check_spans(starts, ends, shard.index)
    if span_fault is not None:
        found.add(span_fault)
    if classes is not None:
        for held, ordinal in classes.collisions:
            found.add(f"documents D{held} and D{ordinal} share a digest and hold different tokens")
    if found.count:
        result = v.VERIFIED_FAILED
    elif not ran:
        result = v.VERIFIED_NOT_RUN
    else:
        result = v.VERIFIED_PASSED
    return Assignment(
        derivation=derivation,
        starts=starts,
        ends=ends,
        line_of=line_of,
        no_token_lines=tuple(no_token_lines),
        surviving_lines=surviving,
        verification=Verification(
            result=result,
            spans_match=span_fault is None,
            count_matches=count_matches,
            documents_compared=compared,
            failures=found.count,
            reasons=tuple(found.reasons),
        ),
    )


# --- Position mode ---------------------------------------------------------------------


def derive_position(
    lines: LineSource,
    shard: ShardTokens,
    classes: DocumentClasses | None,
    retokenize: Retokenizer | None,
    skipped: Container[int],
) -> Assignment:
    """Position mode (it needs plan F1): the surviving input lines, in order, are
    the documents of the shard's index, in order.

    ``lines`` gives a fresh iterator of (line number, raw line). ``skipped`` holds
    the line numbers a sanitizer's log names; they take no ordinal. With
    ``retokenize`` None no document is compared and the verification is "not
    run", unless the count already failed."""
    steps = np.diff(shard.index.document_index)
    boundaries = int(steps.shape[0])
    line_of = np.zeros(shard.count, dtype=np.int64)
    found = _Findings()
    no_token_lines: list[int] = []
    surviving = ordinal = compared = 0
    for number, raw in lines():
        if number in skipped:
            continue
        place = surviving
        surviving += 1
        if place >= boundaries:
            continue  # more lines than documents: counted, and reported below
        if int(steps[place]) == 0:
            # A document boundary with no sequence: the line gave no tokens.
            no_token_lines.append(number)
            if retokenize is not None:
                expected = retokenize(raw)
                if expected is None:
                    found.add(f"line {number} holds no text to tokenize")
                elif expected.size:
                    found.add(
                        f"line {number} gave no tokens by the shard's index and gives "
                        f"{expected.size} when it is tokenized again"
                    )
            continue
        document = ordinal
        ordinal += 1
        line_of[document] = number
        if retokenize is None:
            continue
        compared += 1
        expected = retokenize(raw)
        if expected is None:
            found.add(f"line {number} holds no text to tokenize")
        elif not same_tokens(shard.read(document), expected):
            found.add(f"document D{document} does not hold the tokens of line {number}")
    count_matches = surviving == boundaries
    if not count_matches:
        found.add(
            f"{surviving} surviving input lines and {boundaries} documents in the shard's "
            f"index ({shard.count} with tokens, {boundaries - shard.count} without)"
        )
    return _finish(v.DERIVED_POSITION, shard, classes, line_of, no_token_lines, surviving,
                   found, compared, count_matches, retokenize is not None)


# --- Token-matching mode ---------------------------------------------------------------


def derive_token_matching(
    lines: LineSource,
    shard: ShardTokens,
    classes: DocumentClasses,
    retokenize: Retokenizer,
    skipped: Container[int],
) -> Assignment:
    """Token-matching mode: every input line is tokenized again and given to a
    document that holds exactly its tokens.

    Lines with equal tokens are given to the twin documents in order; which twin
    gets which line carries no meaning, and the map names the whole set for each
    of them. A line that gives no tokens is a "no tokens" drop. Each document is
    compared once, token by token, with the line it is given."""
    line_of = np.zeros(shard.count, dtype=np.int64)
    found = _Findings()
    no_token_lines: list[int] = []
    taken: dict[int, int] = {}
    surviving = compared = 0
    for number, raw in lines():
        if number in skipped:
            continue
        surviving += 1
        expected = retokenize(raw)
        if expected is None:
            found.add(f"line {number} holds no text to tokenize")
            continue
        if expected.size == 0:
            no_token_lines.append(number)
            continue
        first = classes.by_digest.get(token_digest(expected))
        if first is None:
            found.add(f"no document holds the tokens of line {number}")
            continue
        members = classes.members(first)
        place = taken.get(first, 0) if len(members) > 1 else int(line_of[first] != 0)
        if place >= len(members):
            found.add(
                f"line {number} is one line more than the {len(members)} documents that "
                "hold its tokens"
            )
            continue
        document = members[place]
        compared += 1
        if not same_tokens(shard.read(document), expected):
            found.add(f"document D{document} does not hold the tokens of line {number}")
            continue
        line_of[document] = number
        if len(members) > 1:
            taken[first] = place + 1
    unmatched = np.flatnonzero(line_of == 0)
    if unmatched.size:
        found.add(
            f"{unmatched.size} documents hold the tokens of no input line, "
            f"the first is D{int(unmatched[0])}"
        )
    return _finish(v.DERIVED_TOKEN_MATCHING, shard, classes, line_of, no_token_lines,
                   surviving, found, compared, None, True)
