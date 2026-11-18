"""The team's tokenizer, as a recovery reaches it when no Python function is at
hand (plan 3.3 "Tokenization": the tool ships no tokenizer and imports none).

A map is verified by tokenizing each input line again, and token matching builds
the map that way. ``tokenize_shard``, ``map_tokenized_shard`` and ``recover_map``
take the tokenizer as a function ``tokenize(text) -> token ids``. This module
gives two ways to get such a function without the tool importing a tokenizer
library:

- ``command_tokenizer(argv)``: the tokenizer runs as an external command, a
  process of its own that the team wrote around its tokenizer. The tool writes
  one text per line and reads one list of token ids per line;
- ``entry_point_tokenizer("module:function")``: the tool imports the module the
  team names and calls its function. The import is the team's module, by name,
  at the moment it is asked for; nothing of it is imported otherwise.

Both give the token ids of a text without the end-of-document token; the map
derivation adds that token itself where the convention says a document's span
holds it (``TokenizerSettings``).

The protocol of a tokenizer command, ``waterwhite.tokenizer/1``. The command
reads its standard input and writes its standard output, UTF-8, one JSON object
per line, and answers each request before it reads the next:

    request   {"text": "<the text of one line>"}
    reply     {"tokens": [<token id>, ...]}

A reply is flushed as soon as it is written. The command ends when its standard
input is closed. Anything else on its standard output, an exit before the input
is closed, or no reply within the time limit is the named error "tokenizer
failed", and no map is derived from the run. What the command writes to its
standard error is kept and the end of it is quoted in that error.
"""

from __future__ import annotations

import importlib
import json
import os
import select
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence

from ..errors import InvalidRecord, TokenizerFailed

__all__ = [
    "DEFAULT_TOKENIZER_TIMEOUT",
    "TOKENIZER_PROTOCOL",
    "CommandTokenizer",
    "command_tokenizer",
    "entry_point_tokenizer",
]

TOKENIZER_PROTOCOL = "waterwhite.tokenizer/1"

DEFAULT_TOKENIZER_TIMEOUT = 120.0  # seconds one text may take, the start of the command included

_KEPT_ERROR_BYTES = 2000
_READ_CHUNK = 1 << 16


def _is_token(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class CommandTokenizer:
    """A tokenizer command, started once and asked text by text.

    Call it with a text to get the text's token ids. Use it as a context
    manager, or call ``close``: closing ends the command by closing its input and
    waits for it. ``exit_status`` then holds the status it ended with, or None
    when it had to be stopped. A status other than 0 after the last reply is not
    an error here: every reply was given, and a map rests on the comparison of
    those replies with the shard, not on how the command ended. The caller
    reports it."""

    def __init__(self, argv: Sequence[str], *, timeout_seconds: float = DEFAULT_TOKENIZER_TIMEOUT) -> None:
        if isinstance(argv, str):
            argv = shlex.split(argv)
        self.argv = tuple(argv)
        if not self.argv or not all(isinstance(part, str) and part for part in self.argv):
            raise InvalidRecord("a tokenizer command is a program and its arguments")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not timeout_seconds > 0):
            raise InvalidRecord(
                f"the time limit of a tokenizer command is above 0 seconds: {timeout_seconds!r}"
            )
        self._timeout = float(timeout_seconds)
        self._buffer = bytearray()
        self._asked = 0
        self.exit_status: int | None = None
        self._errors = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                self.argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._errors, bufsize=0,
            )
        except OSError as error:
            self._errors.close()
            raise TokenizerFailed(
                f"the tokenizer command {self.argv[0]!r} could not be started: {error}",
                command=self.argv[0],
            ) from None

    # -- the pipe --------------------------------------------------------------------

    def _said(self) -> str:
        """The end of what the command wrote to its standard error."""
        try:
            size = self._errors.seek(0, os.SEEK_END)
            self._errors.seek(max(0, size - _KEPT_ERROR_BYTES))
            return self._errors.read().decode("utf-8", "replace").strip()
        except (OSError, ValueError):
            return ""

    def _failed(self, what: str) -> TokenizerFailed:
        said = self._said()
        self._stop()
        return TokenizerFailed(
            f"the tokenizer command {self.argv[0]!r} {what}"
            + (f"; it said last: {said}" if said else ""),
            command=self.argv[0], text_number=self._asked,
        )

    def _read_line(self) -> bytes:
        deadline = time.monotonic() + self._timeout
        stream = self._process.stdout
        while True:
            cut = self._buffer.find(b"\n")
            if cut >= 0:
                line = bytes(self._buffer[:cut])
                del self._buffer[: cut + 1]
                return line
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([stream], [], [], left)[0]:
                raise self._failed(
                    f"gave no reply for text {self._asked} within {self._timeout:g} seconds"
                )
            chunk = os.read(stream.fileno(), _READ_CHUNK)
            if not chunk:
                raise self._failed(
                    f"ended before it answered text {self._asked} "
                    f"(exit status {self._process.wait()})"
                )
            self._buffer += chunk

    def __call__(self, text: str) -> list[int]:
        if self._process is None:
            raise TokenizerFailed("the tokenizer command was closed", command=self.argv[0])
        self._asked += 1
        request = json.dumps({"text": text}, ensure_ascii=True).encode("ascii") + b"\n"
        try:
            self._process.stdin.write(request)
        except (BrokenPipeError, OSError):
            raise self._failed(f"closed its input before text {self._asked}") from None
        raw = self._read_line()
        try:
            reply = json.loads(raw.decode("utf-8"))
        except (ValueError, RecursionError):
            reply = None
        tokens = reply.get("tokens") if isinstance(reply, dict) else None
        if not isinstance(tokens, list) or not all(_is_token(token) for token in tokens):
            raise self._failed(
                f"answered text {self._asked} with something that is not "
                '{"tokens": [token ids]}'
            )
        return tokens

    # -- the end ---------------------------------------------------------------------

    def _stop(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        for stream in (process.stdin, process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if process.poll() is None:
            process.kill()  # this very process, by its own handle
        process.wait()
        self._errors.close()

    def close(self) -> None:
        """Close the command's input and wait for it to end. A command that does
        not end within the time limit is stopped."""
        process = self._process
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        try:
            self.exit_status = process.wait(timeout=self._timeout)
        except subprocess.TimeoutExpired:
            self.exit_status = None
        self._stop()

    def __enter__(self) -> "CommandTokenizer":
        return self

    def __exit__(self, kind: object, *rest: object) -> None:
        if kind is None:
            self.close()
        else:
            self._stop()


def command_tokenizer(
    argv: Sequence[str] | str, *, timeout_seconds: float = DEFAULT_TOKENIZER_TIMEOUT
) -> CommandTokenizer:
    """Start a tokenizer command and return the function that asks it.

    ``argv`` is the program and its arguments, as a list or as one text that is
    split the way a shell splits words (no shell runs it). ``timeout_seconds``
    is how long one text may take. Use the result as a context manager."""
    return CommandTokenizer(argv, timeout_seconds=timeout_seconds)


def entry_point_tokenizer(spec: str) -> Callable[[str], Sequence[int]]:
    """The function a team names as ``module:function`` (or
    ``package.module:object.method``), imported here and returned.

    The tool imports the named module and nothing else. A module that cannot be
    imported, a name the module does not hold, or a name that is not callable is
    the error "tokenizer failed"."""
    if not isinstance(spec, str) or spec.count(":") != 1:
        raise InvalidRecord(f"a tokenizer entry point is written module:function, not {spec!r}")
    module_name, _, path = spec.partition(":")
    if not module_name or not path:
        raise InvalidRecord(f"a tokenizer entry point is written module:function, not {spec!r}")
    try:
        found: object = importlib.import_module(module_name)
    except Exception as error:  # the team's module, whatever it raises on import
        raise TokenizerFailed(
            f"the module {module_name!r} of the tokenizer entry point could not be "
            f"imported: {type(error).__name__}: {error}",
            entry=spec,
        ) from None
    for part in path.split("."):
        if not hasattr(found, part):
            raise TokenizerFailed(
                f"the tokenizer entry point {spec!r} names {part!r}, which is not there",
                entry=spec,
            )
        found = getattr(found, part)
    if not callable(found):
        raise TokenizerFailed(f"the tokenizer entry point {spec!r} is not callable", entry=spec)
    function = found

    def tokenize(text: str) -> Sequence[int]:
        try:
            return function(text)
        except Exception as error:  # the team's function, whatever it raises
            raise TokenizerFailed(
                f"the tokenizer entry point {spec!r} raised {type(error).__name__}: {error}",
                entry=spec,
            ) from None

    return tokenize
