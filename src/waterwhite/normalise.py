"""Named normalisers: the fixed function a record's text passes before it is
fingerprinted (plan 3.1, Decision 11, F8).

A normaliser is a function from text to text with a name of the form
``"<name>/<version>"``. A store records the one name its fingerprints were taken
after, and every fingerprint of that store is taken after that function. A name
means one function for good: a changed rule is a new version, never an edit, since
an edit would change what equal fingerprints mean in every store written before it.

An unknown name is the named error "unknown normaliser" (``errors.UnknownNormaliser``,
also importable from here). There is no default.

The first build ships one normaliser, ``control-chars/1``. It is defined here and
nowhere else (DECISIONS.md D-17): a pipeline whose sanitizer is this function
imports it from this module in every mode, with capture on or off, so a sanitized
record and the fingerprint of the record it was made from agree (plan 3.1, F8) and
no second copy can drift. docs/capture.md states its rule in full.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from . import ids
from .errors import InvalidRecord, UnknownNormaliser

__all__ = [
    "CONTROL_CHARS_1",
    "Normaliser",
    "UnknownNormaliser",
    "apply",
    "control_chars_1",
    "get",
    "names",
    "register",
    "text_fingerprint",
]

Normaliser = Callable[[str], str]


_NAME_FORM = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$")

_REGISTRY: dict[str, Normaliser] = {}


def register(name: str, function: Normaliser) -> None:
    """Register a normaliser under ``"<name>/<version>"``.

    A team whose pipeline has its own sanitizer registers that function here, under
    its own name and version, before it creates its store (plan 3.1, F8).
    Registering the same function again under its name does nothing. A second,
    different function under a taken name is refused."""
    if not isinstance(name, str) or not _NAME_FORM.match(name):
        raise InvalidRecord(
            f"a normaliser is named '<name>/<version>' in lower-case letters, digits, "
            f"'.', '_' and '-': {name!r}"
        )
    if not callable(function):
        raise InvalidRecord(f"the normaliser {name!r} is not a function")
    held = _REGISTRY.get(name)
    if held is not None and held is not function:
        raise InvalidRecord(
            f"the normaliser name {name!r} is taken; a changed rule needs a new version",
            normaliser=name,
        )
    _REGISTRY[name] = function


def names() -> tuple[str, ...]:
    """The registered names, sorted."""
    return tuple(sorted(_REGISTRY))


def get(name: str) -> Normaliser:
    """The function registered under ``name``, or the error "unknown normaliser"."""
    function = _REGISTRY.get(name) if isinstance(name, str) else None
    if function is None:
        raise UnknownNormaliser(
            f"{name!r} is not a registered normaliser (registered: {', '.join(names())})",
            normaliser=name,
        )
    return function


def apply(name: str, text: str) -> str:
    """The text after the named normaliser."""
    return get(name)(text)


def text_fingerprint(name: str, text: str) -> bytes:
    """The fingerprint of a text: ``ids.fingerprint`` of the text after the named
    normaliser. Equal results mean equal after normalisation (plan 3.1)."""
    return ids.fingerprint(apply(name, text))


# --- control-chars/1 -----------------------------------------------------------------
#
# The rule is written as two fixed lists of code points and never as a lookup in the
# interpreter's Unicode tables, so two machines with different Python versions give
# one result for one text.

# Removed wherever they stand: every code point of the Unicode general category Cc
# (control), which is U+0000 to U+001F and U+007F to U+009F, except the tab (U+0009)
# and the line feed (U+000A). The carriage return (U+000D) is removed, so a
# "\r\n" pair becomes "\n".
CONTROL_CHARS_1_KEPT: frozenset[int] = frozenset({0x09, 0x0A})
CONTROL_CHARS_1_REMOVED: frozenset[int] = frozenset(
    cp
    for cp in (*range(0x00, 0x20), *range(0x7F, 0xA0))
    if cp not in CONTROL_CHARS_1_KEPT
)

# Stripped from both ends after the removal: the tab, the line feed and the
# Unicode white-space characters that are not controls.
CONTROL_CHARS_1_STRIPPED: str = (
    "\t\n   "
    + "".join(chr(cp) for cp in range(0x2000, 0x200B))
    + "    　"
)

_CONTROL_CHARS_1_DELETE = {cp: None for cp in CONTROL_CHARS_1_REMOVED}


def control_chars_1(text: str) -> str:
    """``control-chars/1``: remove the control characters, then strip the ends.

    Characters of the category Cf (format: zero-width joiners, the soft hyphen,
    direction marks, the byte order mark) are kept. Applying the function twice
    gives the same text as applying it once."""
    if not isinstance(text, str):
        raise TypeError(f"a normaliser takes text, not {type(text).__name__}")
    return text.translate(_CONTROL_CHARS_1_DELETE).strip(CONTROL_CHARS_1_STRIPPED)


CONTROL_CHARS_1 = "control-chars/1"

register(CONTROL_CHARS_1, control_chars_1)
