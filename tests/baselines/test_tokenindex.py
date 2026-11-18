"""OriginBlame's token-index entries and forget set, read from hand-written files (plan 5.5
Check 10; table README, section "Check 10": OriginBlame's records and "The OriginBlame
tie by token range").

What the table says, and what these tests hold the reader to:

- A token-index entry has ``sources`` (section hashes), ``token_count``, ``tokenizer``
  and ``revoked``. It has no id, no offset and no line hash. Entries are kept in the
  order the documents were tracked.
- The forget set is ``ceil(N / 8)`` bytes, one bit per entry, in entry order, LEAST
  significant bit first, 1 for revoked: entry 0 is bit 0 of byte 0, entry 8 is bit 0
  of byte 1.
- The token range of an entry is the running sum of the counts of the entries before
  it up to that sum plus its own count. A document is tied to the entry whose range
  equals its own. The tie is valid only where one process tracked the documents of the
  shard in packed order and the run says so; otherwise the question is cannot answer
  and the missing field is an id or an offset on the entry.
- The reader never reads a gap as an absence: a forget set whose length does not fit
  the entries is an error, and an entry with no token count is an error.

The names asked of ``baselines.tokenindex``:

- ``read_entries(path)``: the entries of one JSON-lines file in file order, each with
  ``sources`` (a sequence of text), ``token_count`` (an integer), ``tokenizer`` (text)
  and ``revoked`` (a boolean), as an attribute or a key; ValueError for an entry that
  lacks one of the four or whose ``token_count`` is no integer;
- ``forget_bits(data, count)``: a list of ``count`` booleans from the bytes of a
  forget set, least significant bit first; ValueError where ``len(data)`` is not
  ``ceil(count / 8)``;
- ``read_forget_set(path, count)``: the same, from a file;
- ``token_ranges(entries)``: the list of ``(start, end)`` by running sum, end
  excluded, starting at 0;
- ``tie_by_range(entries, start, end, *, one_process_packed)``: the place of the entry
  whose range is ``(start, end)``, or None where none is; ``TieNotValid`` (a
  ValueError) is raised when ``one_process_packed`` is false, with a message that names
  "an id or an offset".

A module that does not exist yet fails a test on an assertion, not on an import.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

MODULE = "baselines.tokenindex"

FORBIDDEN_IMPORT_ROOTS = {
    "waterwhite",
    "acceptance_rows",
    "acceptance_world",
    "acceptance_repair",
    "acceptance_check09",
}
FORBIDDEN_TEXT = (chr(0x2014), "/ho" + "me/", "Proj" + "ects/")

# Ten hand-written entries: a section hash, a count, the tokenizer and the flag.
COUNTS = [10, 20, 30, 40, 5, 6, 7, 8, 9, 10]
SECTIONS = ["h-a", "h-a", "h-b", "h-b", "h-a", "h-c", "h-c", "h-a", "h-b", "h-c"]


def field(obj, name):
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def entry_line(section: str, count: int, revoked: bool = False) -> str:
    return json.dumps({"revoked": revoked, "sources": [section], "token_count": count,
                       "tokenizer": "gpt2"}, sort_keys=True)


def write_entries(path: Path, counts=COUNTS, sections=SECTIONS) -> Path:
    path.write_text("".join(entry_line(s, c) + "\n" for s, c in zip(sections, counts)),
                    encoding="utf-8")
    return path


# --- The entries -------------------------------------------------------------------------


def test_the_entries_come_in_file_order_with_the_four_fields(need_attribute, tmp_path):
    read_entries = need_attribute(MODULE, "read_entries")
    entries = read_entries(write_entries(tmp_path / "001"))
    assert len(entries) == 10
    assert [field(e, "token_count") for e in entries] == COUNTS
    assert [list(field(e, "sources")) for e in entries] == [[s] for s in SECTIONS]
    assert {field(e, "tokenizer") for e in entries} == {"gpt2"}
    assert [field(e, "revoked") for e in entries] == [False] * 10
    assert all(type(field(e, "token_count")) is int for e in entries)


def test_an_entry_marked_revoked_is_read_as_revoked(need_attribute, tmp_path):
    read_entries = need_attribute(MODULE, "read_entries")
    path = tmp_path / "001"
    path.write_text(entry_line("h-a", 10) + "\n" + entry_line("h-b", 20, True) + "\n",
                    encoding="utf-8")
    assert [field(e, "revoked") for e in read_entries(path)] == [False, True]


def test_an_entry_with_no_token_count_is_an_error_not_a_skipped_entry(need_attribute, tmp_path):
    read_entries = need_attribute(MODULE, "read_entries")
    path = tmp_path / "001"
    path.write_text(
        entry_line("h-a", 10) + "\n"
        + json.dumps({"revoked": False, "sources": ["h-b"], "tokenizer": "gpt2"}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        read_entries(path)


def test_an_entry_whose_count_is_no_integer_is_an_error(need_attribute, tmp_path):
    read_entries = need_attribute(MODULE, "read_entries")
    path = tmp_path / "001"
    path.write_text(
        json.dumps({"revoked": False, "sources": ["h-b"], "token_count": "ten",
                    "tokenizer": "gpt2"}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError):
        read_entries(path)


def test_an_empty_file_has_no_entries(need_attribute, tmp_path):
    read_entries = need_attribute(MODULE, "read_entries")
    path = tmp_path / "001"
    path.write_text("", encoding="utf-8")
    assert list(read_entries(path)) == []


# --- The forget set: least significant bit first -----------------------------------------


def test_bit_zero_of_the_first_byte_is_entry_zero(need_attribute):
    """Red example: most significant bit first. The byte 0b00000001 revokes entry 0,
    and the byte 0b10000000 revokes entry 7."""
    forget_bits = need_attribute(MODULE, "forget_bits")
    assert list(forget_bits(bytes([0b00000001]), 8)) == [True] + [False] * 7
    assert list(forget_bits(bytes([0b10000000]), 8)) == [False] * 7 + [True]


def test_the_bits_of_a_byte_run_from_bit_zero_upward(need_attribute):
    forget_bits = need_attribute(MODULE, "forget_bits")
    # 0b00000101: entries 0 and 2.
    assert list(forget_bits(bytes([0b00000101]), 8)) == [True, False, True] + [False] * 5
    # A byte read most significant bit first would give entries 5 and 7.
    assert list(forget_bits(bytes([0b10100000]), 8)) == [False] * 5 + [True, False, True]


def test_entry_eight_is_bit_zero_of_the_second_byte(need_attribute):
    """Entries 0, 2 and 9 revoked out of ten: byte 0 is 5, byte 1 is 2."""
    forget_bits = need_attribute(MODULE, "forget_bits")
    bits = list(forget_bits(bytes([5, 2]), 10))
    assert len(bits) == 10
    assert [i for i, bit in enumerate(bits) if bit] == [0, 2, 9]
    assert all(type(bit) is bool for bit in bits)


def test_only_count_bits_are_given_the_padding_of_the_last_byte_is_not_an_entry(need_attribute):
    forget_bits = need_attribute(MODULE, "forget_bits")
    assert len(list(forget_bits(bytes([0, 0]), 9))) == 9
    assert len(list(forget_bits(bytes([0, 0]), 16))) == 16


def test_three_hundred_sixty_entries_are_forty_five_bytes_and_the_rows_of_one_source(need_attribute):
    """360 entries need 45 bytes, and 120 revoked entries are the size of one source of
    the three. Here the revoked ones are entries 120 to 239, a choice of this test:
    fifteen zero bytes, fifteen full bytes, fifteen zero bytes."""
    forget_bits = need_attribute(MODULE, "forget_bits")
    data = b"\x00" * 15 + b"\xff" * 15 + b"\x00" * 15
    assert len(data) == 45
    bits = list(forget_bits(data, 360))
    assert len(bits) == 360
    assert [i for i, bit in enumerate(bits) if bit] == list(range(120, 240))


def test_no_entry_has_an_empty_forget_set(need_attribute):
    forget_bits = need_attribute(MODULE, "forget_bits")
    assert list(forget_bits(b"", 0)) == []


@pytest.mark.parametrize("data", [b"", bytes([5]), bytes([5, 2, 0])])
def test_a_forget_set_of_the_wrong_length_is_an_error_not_a_shorter_or_longer_read(need_attribute, data):
    """Ten entries need two bytes. A set that is short or long is not read as bits of
    the entries it happens to cover."""
    forget_bits = need_attribute(MODULE, "forget_bits")
    with pytest.raises(ValueError):
        forget_bits(data, 10)


def test_a_forget_set_is_read_from_a_file(need_attribute, tmp_path):
    read_forget_set = need_attribute(MODULE, "read_forget_set")
    path = tmp_path / "forget.bin"
    path.write_bytes(bytes([5, 2]))
    bits = list(read_forget_set(path, 10))
    assert [i for i, bit in enumerate(bits) if bit] == [0, 2, 9]
    short = tmp_path / "short.bin"
    short.write_bytes(bytes([5]))
    with pytest.raises(ValueError):
        read_forget_set(short, 10)


# --- The token ranges --------------------------------------------------------------------


def test_the_ranges_are_the_running_sum_of_the_counts(need_attribute, tmp_path):
    token_ranges = need_attribute(MODULE, "token_ranges")
    read_entries = need_attribute(MODULE, "read_entries")
    entries = read_entries(write_entries(tmp_path / "001", counts=[10, 20, 30],
                                         sections=["h-a", "h-b", "h-c"]))
    assert [tuple(r) for r in token_ranges(entries)] == [(0, 10), (10, 30), (30, 60)]


def test_the_ranges_of_ten_entries_end_at_the_sum_of_all_counts(need_attribute, tmp_path):
    token_ranges = need_attribute(MODULE, "token_ranges")
    read_entries = need_attribute(MODULE, "read_entries")
    ranges = [tuple(r) for r in token_ranges(read_entries(write_entries(tmp_path / "001")))]
    assert ranges[0] == (0, 10)
    assert ranges[4] == (100, 105)
    assert ranges[-1] == (135, 145)
    assert sum(COUNTS) == 145
    # Each range starts where the one before it ends.
    assert all(ranges[i][1] == ranges[i + 1][0] for i in range(len(ranges) - 1))


def test_no_entry_has_no_range(need_attribute):
    token_ranges = need_attribute(MODULE, "token_ranges")
    assert list(token_ranges([])) == []


# --- The tie by token range --------------------------------------------------------------


# Looked up when a test calls it, never at setup: a module that does not exist yet is
# a failed assertion in the body of a test, not an error.
@pytest.fixture
def three_entries(need_attribute, tmp_path):
    def make():
        read_entries = need_attribute(MODULE, "read_entries")
        return read_entries(write_entries(tmp_path / "001", counts=[10, 20, 30],
                                          sections=["h-a", "h-b", "h-c"]))

    return make


def test_a_document_is_tied_to_the_entry_whose_range_equals_its_own(need_attribute, three_entries):
    entries = three_entries()
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    assert tie_by_range(entries, 0, 10, one_process_packed=True) == 0
    assert tie_by_range(entries, 10, 30, one_process_packed=True) == 1
    assert tie_by_range(entries, 30, 60, one_process_packed=True) == 2


def test_a_range_that_is_not_an_entrys_whole_range_is_no_tie(need_attribute, three_entries):
    entries = three_entries()
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    # Right start, wrong end; wrong start, right end; inside an entry; past the sum.
    assert tie_by_range(entries, 10, 29, one_process_packed=True) is None
    assert tie_by_range(entries, 11, 30, one_process_packed=True) is None
    assert tie_by_range(entries, 12, 18, one_process_packed=True) is None
    assert tie_by_range(entries, 60, 70, one_process_packed=True) is None
    # A range that spans two entries is no entry's range.
    assert tie_by_range(entries, 0, 30, one_process_packed=True) is None


def test_the_tie_is_not_valid_unless_the_run_says_one_process_in_packed_order(need_attribute, three_entries):
    entries = three_entries()
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    not_valid = need_attribute(MODULE, "TieNotValid")
    assert issubclass(not_valid, ValueError)
    # The range would match entry 1, and still the tie is not given.
    with pytest.raises(not_valid) as raised:
        tie_by_range(entries, 10, 30, one_process_packed=False)
    assert "an id or an offset" in str(raised.value)


def test_the_tie_that_is_not_valid_is_raised_even_where_no_entry_matches(need_attribute, three_entries):
    entries = three_entries()
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    not_valid = need_attribute(MODULE, "TieNotValid")
    with pytest.raises(not_valid):
        tie_by_range(entries, 11, 12, one_process_packed=False)


def test_the_tie_needs_the_run_to_say_so_it_has_no_default(need_attribute, three_entries):
    entries = three_entries()
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    with pytest.raises(TypeError):
        tie_by_range(entries, 10, 30)


def test_a_tied_entry_gives_its_bit_in_the_forget_set(need_attribute, three_entries):
    entries = three_entries()
    """The three entries as a shard's store: the forget set 0b010 revokes entry 1 only,
    so the document of range (10, 30) is revoked and the other two are not."""
    tie_by_range = need_attribute(MODULE, "tie_by_range")
    forget_bits = need_attribute(MODULE, "forget_bits")
    bits = list(forget_bits(bytes([0b010]), 3))
    assert bits[tie_by_range(entries, 10, 30, one_process_packed=True)] is True
    assert bits[tie_by_range(entries, 0, 10, one_process_packed=True)] is False
    assert bits[tie_by_range(entries, 30, 60, one_process_packed=True)] is False


# --- What the module may hold ------------------------------------------------------------


def _path(need_module) -> Path:
    module = need_module(MODULE)
    path = getattr(module, "__file__", None)
    assert path, f"{MODULE} has no file"
    return Path(path)


def test_the_module_imports_nothing_of_the_tool_nor_of_the_world_helpers(need_module):
    path = _path(need_module)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & FORBIDDEN_IMPORT_ROOTS, sorted(roots & FORBIDDEN_IMPORT_ROOTS)


def test_the_module_holds_no_local_path_and_no_em_dash(need_module):
    text = _path(need_module).read_text(encoding="utf-8")
    for bad in FORBIDDEN_TEXT:
        assert bad not in text, f"{MODULE} holds {bad!r}"
