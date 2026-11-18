"""The pure steps: id ranges, shuffles, round-robin dealing, the sanitizer, dedup."""

from __future__ import annotations

import pytest
from waterwhite import normalise as tool_normalise

from replica.errors import InputError, SpecError
from replica.records import parse_line
from replica.steps import (
    DETAIL_EMPTY_AFTER,
    REASON_EMPTY_LINE,
    REASON_SANITIZER_SKIP,
    RULE_FIRST,
    RULE_OFF,
    RULE_SINGLE,
    RULE_WINNER,
    SeededShuffler,
    UnseededShuffler,
    chunk_names,
    deal_round_robin,
    dedup,
    expand_ids,
    normaliser,
    partitioned_shards,
    round_robin_chunks,
    sanitize,
)

# The one normaliser of the first build, taken from the tool as the replica takes it.
CONTROL_CHARS = tool_normalise.get("control-chars/1")


class InOrder:
    """A shuffler that leaves every list as it is, to show the dealing alone."""

    def shuffle(self, items, label):
        return list(items)


def record(ref, text, source="Z", line=1, artifact="sources/Z.jsonl"):
    import json

    return parse_line(json.dumps({"id": ref, "text": text}), artifact=artifact, line=line,
                      source=source)


# --- Id ranges ---------------------------------------------------------------------


def test_expand_ids_expands_a_range_and_keeps_single_ids():
    assert expand_ids(["A-001", "Z-009..Z-012", "P"]) == [
        "A-001", "Z-009", "Z-010", "Z-011", "Z-012", "P"]


def test_expand_ids_range_of_one():
    assert expand_ids(["B-007..B-007"]) == ["B-007"]


@pytest.mark.parametrize("item", [
    "Z-001..Y-004",   # two prefixes
    "Z-01..Z-004",    # two widths
    "Z-009..Z-003",   # downwards
    "Z-001..",        # no end
    "..Z-004",        # no start
    "Z-001..Zend",    # no number
    "",
])
def test_expand_ids_refuses_a_bad_range(item):
    with pytest.raises(SpecError):
        expand_ids([item])


# --- Shuffles ----------------------------------------------------------------------


def test_seeded_shuffle_is_pinned():
    # The order is part of the fixture's ground: a change of the algorithm must fail here.
    assert SeededShuffler(0).shuffle(list(range(10)), "x") == [4, 0, 6, 3, 5, 1, 8, 7, 2, 9]
    assert SeededShuffler(1).shuffle(list(range(10)), "x") == [0, 9, 8, 4, 2, 1, 7, 3, 5, 6]
    assert SeededShuffler(0).shuffle(list(range(10)), "y") == [3, 2, 9, 4, 6, 1, 8, 7, 5, 0]


def test_seeded_shuffle_repeats_and_keeps_every_item():
    items = [f"r{n}" for n in range(200)]
    first = SeededShuffler(7).shuffle(items, "file/A")
    assert first == SeededShuffler(7).shuffle(items, "file/A")
    assert sorted(first) == sorted(items)
    assert first != items
    assert items == [f"r{n}" for n in range(200)]  # the input is not changed


def test_seeded_shuffle_refuses_a_seed_that_is_no_integer():
    with pytest.raises(SpecError):
        SeededShuffler("0")
    with pytest.raises(SpecError):
        SeededShuffler(True)


def test_unseeded_shuffle_keeps_every_item_and_differs_between_calls():
    items = list(range(200))
    shuffler = UnseededShuffler()
    first, second = shuffler.shuffle(items, "a"), shuffler.shuffle(items, "a")
    assert sorted(first) == items and sorted(second) == items
    assert first != second  # 200 items: equal orders are out of reach


# --- Round-robin dealing -----------------------------------------------------------


def test_deal_round_robin():
    assert deal_round_robin(list(range(7)), 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert deal_round_robin([], 2) == [[], []]
    assert deal_round_robin([1, 2], 1) == [[1, 2]]


@pytest.mark.parametrize("chunks", [0, -1, 1.5, True])
def test_deal_round_robin_refuses_a_bad_chunk_count(chunks):
    with pytest.raises(SpecError):
        deal_round_robin([1], chunks)


def test_chunk_names_are_padded_to_the_width_of_the_count():
    assert chunk_names(3) == ["shard-1", "shard-2", "shard-3"]
    assert chunk_names(20)[0] == "shard-01" and chunk_names(20)[-1] == "shard-20"
    assert chunk_names(20) == sorted(chunk_names(20))


def test_round_robin_deals_each_file_from_the_first_chunk_and_merges_in_listing_order():
    files = [("a", ["a1", "a2", "a3", "a4", "a5"]), ("b", ["b1", "b2"])]
    assert round_robin_chunks(files, 3, InOrder()) == [
        ("shard-1", ["a1", "a4", "b1"]),
        ("shard-2", ["a2", "a5", "b2"]),
        ("shard-3", ["a3"]),
    ]


def test_round_robin_a_file_as_long_as_the_chunk_count_reaches_every_chunk():
    # Plan 4.5: a file made wholly of one source with at least N records reaches all N.
    files = [("a", [f"a{n}" for n in range(20)]), ("z", [f"z{n}" for n in range(300)])]
    for seed in range(5):
        chunks = round_robin_chunks(files, 20, SeededShuffler(seed))
        assert all(any(item.startswith("a") for item in items) for _, items in chunks)
        assert sorted(i for _, items in chunks for i in items) == sorted(files[0][1] + files[1][1])


def test_round_robin_is_reproducible_under_a_seed():
    files = [("a", list(range(50))), ("b", list(range(50, 90)))]
    assert round_robin_chunks(files, 4, SeededShuffler(3)) == round_robin_chunks(
        files, 4, SeededShuffler(3))
    assert round_robin_chunks(files, 4, SeededShuffler(3)) != round_robin_chunks(
        files, 4, SeededShuffler(4))


def test_partitioned_shards_keep_their_records():
    shards = [("shard-1", list(range(30))), ("shard-2", list(range(30, 45)))]
    built = partitioned_shards(shards, SeededShuffler(0))
    assert [name for name, _ in built] == ["shard-1", "shard-2"]
    assert sorted(built[0][1]) == list(range(30)) and sorted(built[1][1]) == list(range(30, 45))
    assert built[0][1] != list(range(30))


# --- The normaliser and the sanitizer ----------------------------------------------


def test_the_sanitizer_uses_the_tools_normaliser_and_holds_no_copy_of_its_own():
    # DECISIONS.md D-17: one definition, in waterwhite.normalise, in every mode.
    import replica.steps

    assert normaliser("control-chars/1") is tool_normalise.get("control-chars/1")
    assert normaliser("control-chars/1") is tool_normalise.control_chars_1
    assert not hasattr(replica.steps, "control_chars_1")
    assert not hasattr(replica.steps, "NORMALISERS")
    with pytest.raises(SpecError, match="unknown normaliser"):
        normaliser("lowercase/1")


def test_the_sanitizer_follows_the_tools_rule_where_readings_of_it_differ():
    # control-chars/1 keeps a tab and a line feed inside a text and removes every
    # other control character. A copy written from the words "remove control
    # characters" would drop the tab too; the sanitizer must not.
    kept, skips = sanitize([record("Z-1", "  a\u0000b\tc\u0007d\u0085e \n")], CONTROL_CHARS)
    assert [r.text for r in kept] == ["ab\tcde"] and skips == []
    umlauts = "K" + chr(0xE4) + "se und " + chr(0xC4) + "pfel"
    assert sanitize([record("Z-2", umlauts)], CONTROL_CHARS)[0][0].text == umlauts


def test_sanitize_skips_and_logs_and_keeps_the_order_of_the_survivors():
    lines = [
        record("Z-1", "one"),
        record("Z-2", "\u0000\u0007"),                                  # empty after sanitizing
        parse_line("", artifact="sources/Z.jsonl", line=3, source="Z"),  # blank line
        parse_line("{not json", artifact="sources/Z.jsonl", line=4, source="Z"),
        parse_line('{"id": "Z-5"}', artifact="sources/Z.jsonl", line=5, source="Z"),
        parse_line('{"text": "no id"}', artifact="sources/Z.jsonl", line=6, source="Z"),
        parse_line("[1, 2]", artifact="sources/Z.jsonl", line=7, source="Z"),
        record("Z-8", " two\u0001 "),
    ]
    kept, skips = sanitize(lines, CONTROL_CHARS)
    assert [(r.ref, r.text) for r in kept] == [("Z-1", "one"), ("Z-8", "two")]
    assert [(s.position, s.reason, s.detail) for s in skips] == [
        (2, REASON_EMPTY_LINE, DETAIL_EMPTY_AFTER),
        (3, REASON_EMPTY_LINE, "blank line"),
        (4, REASON_SANITIZER_SKIP, "not JSON"),
        (5, REASON_SANITIZER_SKIP, "no text field"),
        (6, REASON_SANITIZER_SKIP, "no id field"),
        (7, REASON_SANITIZER_SKIP, "not a JSON object"),
    ]
    assert [s.record.has_id for s in skips] == [True, False, False, True, False, False]
    assert skips[3].record.ref == "Z-5" and skips[1].record.ref == "sources/Z.jsonl#L3"


def test_the_sanitizer_spells_its_reasons_as_the_tool_spells_its_drop_reasons():
    from waterwhite import vocabulary

    from replica.build import REASON_DEDUP_LOSER

    assert REASON_SANITIZER_SKIP == vocabulary.DROP_SANITIZER_SKIP
    assert REASON_EMPTY_LINE == vocabulary.DROP_EMPTY_LINE
    assert REASON_DEDUP_LOSER == vocabulary.DROP_DEDUP_LOSER


# --- A line whose id is its place (a generated set) ----------------------------------


def test_a_line_named_by_its_place_keeps_that_id_whatever_it_holds():
    at = dict(artifact="sources/Z.jsonl", source="Z")
    broken = parse_line('{"id": "Z-000007", "text": "cut off', line=7, ref="Z-000007", **at)
    assert (broken.ref, broken.has_id, broken.readable, broken.fault) == (
        "Z-000007", True, False, "not JSON")
    prose = parse_line("line Z-000008 was never written as a record", line=8, ref="Z-000008", **at)
    assert (prose.ref, prose.has_id, prose.readable) == ("Z-000008", True, False)
    empty = parse_line('{"id": "Z-000009", "text": ""}', line=9, ref="Z-000009", **at)
    assert (empty.ref, empty.readable, empty.text) == ("Z-000009", True, "")
    with pytest.raises(InputError, match="the id of this line is 'Z-000010'"):
        parse_line('{"id": "Z-000011", "text": "x"}', line=10, ref="Z-000010", **at)


# --- Dedup ---------------------------------------------------------------------------


def test_dedup_keeps_the_copy_of_the_winner_source_at_the_place_of_the_first_copy():
    a = record("A-1", "same text", source="A", artifact="sources/A.jsonl")
    b = record("B-1", "same text", source="B", artifact="sources/B.jsonl")
    u = record("Z-1", "other")
    for winner, kept_ref in (("A", "A-1"), ("B", "B-1")):
        for order in ([a, b, u], [b, u, a]):
            kept = dedup(order, CONTROL_CHARS, enabled=True, priority=(winner,))
            assert len(kept) == 2
            group = kept[0]
            assert group.record.ref == kept_ref and group.rule == RULE_WINNER
            assert [c.ref for c in group.copies] == [r.ref for r in order if r is not u]
            assert [c.ref for c in group.losers] == [r.ref for r in (a, b) if r.ref != kept_ref]
            assert kept[1].record is u and kept[1].rule == RULE_SINGLE and kept[1].losers == ()


def test_dedup_a_group_without_a_listed_source_keeps_its_first_copy():
    first = record("Z-1", "twin", source="Z")
    second = record("C-1", "twin", source="C")
    kept = dedup([first, second], CONTROL_CHARS, enabled=True, priority=("A",))
    assert [(k.record.ref, k.rule) for k in kept] == [("Z-1", RULE_FIRST)]
    kept = dedup([first, second], CONTROL_CHARS, enabled=True)
    assert [(k.record.ref, k.rule) for k in kept] == [("Z-1", RULE_FIRST)]


def test_dedup_follows_the_whole_priority_list():
    # docs/plant.md: among equal texts the copy from the source that comes first in
    # dedup_priority is kept, wherever it stands in the input.
    z = record("Z-1", "twin", source="Z")
    c = record("C-1", "twin", source="C")
    z_again = record("Z-2", "twin", source="Z")
    kept = dedup([z, c, z_again], CONTROL_CHARS, enabled=True, priority=("A", "B", "C", "Z"))
    assert [(k.record.ref, k.rule) for k in kept] == [("C-1", RULE_WINNER)]
    assert [c.ref for c in kept[0].losers] == ["Z-1", "Z-2"]
    kept = dedup([z, c, z_again], CONTROL_CHARS, enabled=True, priority=("B", "Z", "C"))
    assert kept[0].record.ref == "Z-1"  # two copies from one source: the earlier one


def test_dedup_compares_the_normalised_text():
    plain = record("A-1", "text", source="A")
    dirty = record("B-1", " te\u0000xt ", source="B")
    kept = dedup([plain, dirty], CONTROL_CHARS, enabled=True, priority=("B",))
    assert [k.record.ref for k in kept] == ["B-1"]
    assert kept[0].record.text == " te\u0000xt "  # the kept copy's own text, untouched


def test_dedup_never_groups_a_line_with_no_text_and_it_stays_a_row():
    # DECISIONS.md D-17; docs/plant.md 5.1 point 3: a text that is empty after
    # normalisation and a line that does not parse are in no group, however alike
    # they are, and each stays a row until the sanitizer drops it.
    lines = [
        record("Z-1", "\u0000"),
        record("Z-2", "\u0007"),
        record("Z-3", ""),
        record("Z-4", ""),
        parse_line("{", artifact="sources/Z.jsonl", line=5, source="Z"),
        parse_line("{", artifact="sources/Z.jsonl", line=6, source="Z"),
        record("Z-7", "kept"),
    ]
    kept = dedup(lines, CONTROL_CHARS, enabled=True, priority=("A",))
    assert [k.record for k in kept] == lines
    assert all(k.rule == RULE_SINGLE and k.losers == () for k in kept)
    survivors, skips = sanitize([k.record for k in kept], CONTROL_CHARS)
    assert [r.ref for r in survivors] == ["Z-7"]
    assert [s.reason for s in skips] == [REASON_EMPTY_LINE] * 4 + [REASON_SANITIZER_SKIP] * 2


def test_dedup_switched_off_keeps_every_copy():
    a = record("A-1", "same", source="A")
    b = record("B-1", "same", source="B")
    kept = dedup([a, b], CONTROL_CHARS, enabled=False, priority=("A",))
    assert [(k.record.ref, k.rule) for k in kept] == [("A-1", RULE_OFF), ("B-1", RULE_OFF)]
