"""What a map is derived from, beside the shard: the settings of the tokenizer
run, the lines of the input file and the skips of a sanitizer's log.

Plan sections: 3.3 "Tokenization" (the ``--append-eod`` setting and the
end-of-document convention are recorded on every map; a recovery reads skipped and
empty lines from the sanitizer's log), 3.1 "Drop", F1 and F8.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .. import capture, ids, normalise
from .. import vocabulary as v
from ..capture import line_text
from ..errors import ArtifactDigestMismatch, InvalidRecord
from ..records import StepRef

# The step names and the sanitizer's reasons are the vocabulary's, so that the
# recovery of a sanitizer (``waterwhite.recovery``) and a map recovered through a
# sanitizer's log write the same words for the same drop.
STEP_TOKENIZE = v.STEP_TOKENIZE
STEP_SANITIZE = v.STEP_SANITIZE

# The reasons a sanitizer's log gives for a skipped line. They are spelled as the
# tool's drop reasons; a log that uses another word is refused, not guessed at.
SANITIZER_REASONS = v.SANITIZER_DROP_REASONS


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def tokenize_step(version: str, params: object = None) -> StepRef:
    """The step a shard is registered under: the name "tokenize", the version of
    the team's tokenizer and the digest of its parameters (any JSON value; put the
    tokenizer's name, its files' revision and its switches there)."""
    return capture.step_ref(STEP_TOKENIZE, version, params)


@dataclass(frozen=True)
class TokenizerSettings:
    """The two settings of a tokenizer run that every map records (plan 3.3, F1).

    ``append_eod``: the run was told to end every document with the
    end-of-document token (Megatron's ``--append-eod``).

    ``eod_in_span``: the end-of-document convention. True when that token is the
    last token of every document's own span in the shard's index; False when no
    document's span holds it. It is a fact about the shard, taken from a reading
    of the shard outside the tool, and the verification holds every document
    against it.

    ``eod_token_id`` is the token itself. It is needed when ``eod_in_span`` is
    true, because the tokens of a document are then the tokens of its line
    followed by this one."""

    append_eod: bool
    eod_in_span: bool
    eod_token_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.append_eod, bool) or not isinstance(self.eod_in_span, bool):
            raise InvalidRecord("append_eod and eod_in_span are true or false")
        if self.eod_in_span and not self.append_eod:
            raise InvalidRecord(
                "eod_in_span is true and append_eod is false: a run that appends no "
                "end-of-document token leaves none in a document's span"
            )
        if self.eod_token_id is not None and (
            not _is_int(self.eod_token_id) or self.eod_token_id < 0
        ):
            raise InvalidRecord(f"not a token id: {self.eod_token_id!r}")
        if self.eod_in_span and self.eod_token_id is None:
            raise InvalidRecord(
                "eod_in_span is true and no eod_token_id is given; the verification "
                "needs the token to know what a document ends with"
            )


@dataclass(frozen=True)
class SanitizerSkip:
    """One line a sanitizer skipped, as its log gives it: the line of the
    sanitizer's input file, counted from 1, and the reason."""

    line: int
    reason: str

    def __post_init__(self) -> None:
        if not _is_int(self.line) or self.line < 1:
            raise InvalidRecord(f"a line number is a whole number from 1: {self.line!r}")
        if self.reason not in SANITIZER_REASONS:
            raise InvalidRecord(
                f"line {self.line}: a sanitizer's log gives the reason "
                f"{' or '.join(repr(r) for r in SANITIZER_REASONS)}, not {self.reason!r}"
            )


@dataclass(frozen=True)
class SanitizerRun:
    """Says that the input artifact of a recovery is the file the sanitizer read,
    not the file it wrote (plan 3.3, "Deterministic rewrite" and "Tokenization").

    ``skips`` are the lines the sanitizer's log names; each becomes a drop and
    takes no ordinal. Every other line survived. Its text is passed through the
    sanitizer again before it is tokenized again (plan F8): ``normaliser`` names
    that function in ``waterwhite.normalise``, and None means the store's own
    normaliser. ``step_name`` is the step the drops are recorded for."""

    skips: tuple[SanitizerSkip, ...] = ()
    normaliser: str | None = None
    step_name: str = STEP_SANITIZE

    def __post_init__(self) -> None:
        skips = tuple(self.skips)
        object.__setattr__(self, "skips", skips)
        seen: set[int] = set()
        for skip in skips:
            if not isinstance(skip, SanitizerSkip):
                raise InvalidRecord(f"expected a sanitizer skip, got {skip!r}")
            if skip.line in seen:
                raise InvalidRecord(f"line {skip.line} is skipped twice")
            seen.add(skip.line)
        if not isinstance(self.step_name, str) or not self.step_name:
            raise InvalidRecord("the sanitizer's step has a name")
        if self.normaliser is not None and not isinstance(self.normaliser, str):
            raise InvalidRecord("a normaliser is named by text")


def read_sanitizer_log(
    path: str | os.PathLike, *, shard: str | None = None, input: str | None = None
) -> tuple[SanitizerSkip, ...]:
    """The skips of one input file, read from a sanitizer's log.

    The log is JSON lines, one object per skipped line, with at least ``line``
    (the line of the sanitizer's input file, from 1) and ``reason`` ("sanitizer
    skip" or "empty line"). A log that covers several files tells them apart by
    its fields ``shard`` and ``input``: name one of them, or both, and only the
    rows that carry that value are read. With neither named, every row is read,
    and a log whose rows name more than one shard or input is refused, since its
    line numbers would be those of several files. The skips come back in line
    order."""
    where = os.path.basename(os.fspath(path))
    # The log is read as a stream: what is kept is one skip per row that is read,
    # never the rows themselves.
    skips: dict[int, SanitizerSkip] = {}
    named: dict[str, set[str]] = {"shard": set(), "input": set()}
    for number, raw in enumerate(ids.read_lines(path), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            row = None
        if not isinstance(row, dict):
            raise InvalidRecord(f"{where}, row {number}: not a JSON object")
        if shard is not None and row.get("shard") != shard:
            continue
        if input is not None and row.get("input") != input:
            continue
        if shard is None and input is None:
            for key, values in named.items():
                values.add(json.dumps(row.get(key), sort_keys=True))
                if len(values) > 1:
                    raise InvalidRecord(
                        f"{where}: its rows name more than one value of {key!r}, so its "
                        "line numbers are those of several files; name the one to read"
                    )
        at = f"{where}, row {number}"
        try:
            skip = SanitizerSkip(row.get("line"), row.get("reason"))
        except InvalidRecord as error:
            raise InvalidRecord(f"{at}: {error.detail}") from None
        if skip.line in skips:
            raise InvalidRecord(f"{at}: line {skip.line} is logged twice")
        skips[skip.line] = skip
    return tuple(skips[line] for line in sorted(skips))


def text_reader(
    text_field: str, sanitizer: Callable[[str], str] | None
) -> Callable[[bytes], str | None]:
    """The function that gives the text the tokenizer saw for one raw line: the
    text field, passed through the sanitizer when the lines are those the
    sanitizer read."""

    def read(raw: bytes) -> str | None:
        text = line_text(raw, text_field)
        if text is None or sanitizer is None:
            return text
        return sanitizer(text)

    return read


def sanitizer_function(run: SanitizerRun, store_normaliser: str) -> Callable[[str], str]:
    """The sanitizer to run again: the named normaliser, or the store's own. A name
    this build does not hold is the error "unknown normaliser" (plan F8)."""
    return normalise.get(run.normaliser or store_normaliser)


def numbered_lines(
    path: str | os.PathLike, registered_digest: str, name: str
) -> Iterator[tuple[int, bytes]]:
    """The lines of a registered JSONL file with their numbers from 1.

    Lines are the lines of ``ids.split_lines``, so line n is the record the store
    holds under ``L<n>``. The bytes are digested as they are read, and when the
    last line has been given the digest must be the one the artifact is
    registered with; otherwise the lines just read were not the registered
    records and "artifact digest mismatch" is raised."""
    hasher = ids.ContentHasher()
    number = 0
    for raw in ids.read_lines(path, hasher):
        number += 1
        yield number, raw
    require_registered_bytes(hasher.hexdigest(), registered_digest, name)


def require_registered_bytes(on_disk: str, registered_digest: str, name: str) -> None:
    """Raise "artifact digest mismatch" unless a file's digest is the one its
    artifact is registered with."""
    if on_disk != registered_digest:
        raise ArtifactDigestMismatch(
            f"{name} is registered with digest {registered_digest} and its file has "
            f"{on_disk}; no map is derived from lines that are not the registered records",
            registered=registered_digest, on_disk=on_disk,
        )
