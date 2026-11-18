"""The normaliser registry and control-chars/1 (plan 3.1, Decision 11, F8).

The expected results here are written by hand from the rule in docs/capture.md.
"""

from __future__ import annotations

import unicodedata

import pytest

from waterwhite import ids, normalise
from waterwhite.errors import EXIT_WRITE_REFUSED, InvalidRecord, WaterwhiteError

N1 = "control-chars/1"


def test_the_first_build_ships_control_chars_1():
    assert N1 in normalise.names()
    assert normalise.get(N1) is normalise.control_chars_1
    assert normalise.CONTROL_CHARS_1 == N1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain text", "plain text"),
        ("nul\x00inside", "nulinside"),
        ("bell\x07 and escape\x1b[0m", "bell and escape[0m"),
        ("tab\tstays", "tab\tstays"),
        ("line one\nline two", "line one\nline two"),
        ("windows\r\nline end", "windows\nline end"),
        ("lone\rcarriage return", "lonecarriage return"),
        ("vertical\x0btab and form\x0cfeed", "verticaltab and formfeed"),
        ("delete\x7fgone", "deletegone"),
        ("c1 next line\x85gone", "c1 next linegone"),
        ("c1 last\x9fgone", "c1 lastgone"),
        ("unit\x1fseparator", "unitseparator"),
    ],
)
def test_control_characters_are_removed_except_tab_and_line_feed(raw, expected):
    assert normalise.apply(N1, raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "zero‍width joiner",
        "zero‌width non-joiner",
        "soft­hyphen",
        "﻿byte order mark at the start",
        "left‎to right mark",
        "word⁠joiner",
    ],
)
def test_format_characters_are_kept(text):
    assert normalise.apply(N1, text) == text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  two spaces each side  ", "two spaces each side"),
        ("\t\ntab and line feed at the ends\n\t", "tab and line feed at the ends"),
        (" no-break space and ideographic space　", "no-break space and ideographic space"),
        (" line and paragraph separators ", "line and paragraph separators"),
        ("inner  spaces stay", "inner  spaces stay"),
        # The removal comes first, so a control at an end does not shield the
        # white space beside it.
        ("\x00  control then spaces", "control then spaces"),
        ("spaces then control  \x1f", "spaces then control"),
        ("trailing carriage return\r\n", "trailing carriage return"),
        ("   ", ""),
        ("\x00\x01", ""),
        ("", ""),
    ],
)
def test_the_ends_are_stripped_after_the_removal(raw, expected):
    assert normalise.apply(N1, raw) == expected


def test_the_removed_set_is_the_category_cc_without_tab_and_line_feed():
    # The rule is the fixed list. This shows the list is what its words say, on
    # the Unicode tables of the interpreter that runs the test.
    cc = {cp for cp in range(0x110000) if unicodedata.category(chr(cp)) == "Cc"}
    assert len(cc) == 65
    assert normalise.CONTROL_CHARS_1_REMOVED == cc - {0x09, 0x0A}
    assert len(normalise.CONTROL_CHARS_1_REMOVED) == 63


def test_the_stripped_set_is_unicode_white_space_that_is_not_removed():
    # Same kind of cross-check as above, against str.isspace of this interpreter.
    expected = {
        chr(cp)
        for cp in range(0x110000)
        if chr(cp).isspace() and cp not in normalise.CONTROL_CHARS_1_REMOVED
    }
    assert set(normalise.CONTROL_CHARS_1_STRIPPED) == expected
    assert len(normalise.CONTROL_CHARS_1_STRIPPED) == 21


def test_every_single_code_point_is_removed_stripped_or_kept():
    stripped = set(normalise.CONTROL_CHARS_1_STRIPPED)
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        inside = normalise.control_chars_1(f"a{ch}b")
        alone = normalise.control_chars_1(ch)
        if cp in normalise.CONTROL_CHARS_1_REMOVED:
            assert (inside, alone) == ("ab", ""), hex(cp)
        elif ch in stripped:
            assert (inside, alone) == (f"a{ch}b", ""), hex(cp)
        else:
            assert (inside, alone) == (f"a{ch}b", ch), hex(cp)


@pytest.mark.parametrize(
    "text",
    ["  a\x00 \r\n", "\x00 \x00 a", "a \x00", "\t\x0b\ta\x0c\n", "x\r\ny\r", "　\x85 z"],
)
def test_normalising_twice_changes_nothing(text):
    once = normalise.apply(N1, text)
    assert normalise.apply(N1, once) == once


def test_a_normaliser_takes_text_only():
    with pytest.raises(TypeError):
        normalise.control_chars_1(b"bytes")  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["control-chars/2", "control-chars", "", "CONTROL-CHARS/1", None])
def test_an_unknown_name_is_a_named_error_and_never_a_default(name):
    with pytest.raises(normalise.UnknownNormaliser) as caught:
        normalise.get(name)
    assert caught.value.name == "unknown normaliser"
    assert isinstance(caught.value, WaterwhiteError)
    assert caught.value.exit_code == EXIT_WRITE_REFUSED
    with pytest.raises(normalise.UnknownNormaliser):
        normalise.apply(name, "text")
    with pytest.raises(normalise.UnknownNormaliser):
        normalise.text_fingerprint(name, "text")


def test_a_name_means_one_function():
    def lower(text: str) -> str:
        return text.lower()

    normalise.register("test-lower/1", lower)
    normalise.register("test-lower/1", lower)  # the same function again: nothing happens
    assert normalise.apply("test-lower/1", "ABC") == "abc"
    with pytest.raises(InvalidRecord):
        normalise.register("test-lower/1", str.upper)
    with pytest.raises(InvalidRecord):
        normalise.register(N1, lower)
    assert normalise.get(N1) is normalise.control_chars_1


@pytest.mark.parametrize("name", ["no-version", "Upper/1", "two/parts/3", "/1", "name/", " x/1"])
def test_a_name_has_the_form_name_slash_version(name):
    with pytest.raises(InvalidRecord):
        normalise.register(name, str.lower)


def test_fingerprints_are_equal_for_texts_equal_after_normalisation():
    base = normalise.text_fingerprint(N1, "The same passage.")
    assert len(base) == ids.FINGERPRINT_BYTES == 16
    for other in (
        "  The same passage.  ",
        "The same\x00 passage.",
        "The same passage.\r\n",
        "\x1bThe same passage.\x7f",
    ):
        assert normalise.text_fingerprint(N1, other) == base, repr(other)
    assert base == ids.fingerprint("The same passage.")


def test_fingerprints_differ_for_texts_that_differ_after_normalisation():
    base = normalise.text_fingerprint(N1, "The same passage.")
    for other in (
        "the same passage.",
        "The same  passage.",
        "The same\tpassage.",
        "The same\npassage.",
        "The same passage",
        "The same pas‍sage.",
        "﻿The same passage.",
    ):
        assert normalise.text_fingerprint(N1, other) != base, repr(other)
