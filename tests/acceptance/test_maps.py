"""The comparisons of Checks 3 and 6, tested on their own with documents written by
hand in the form ``show`` prints them.

Most rows of Checks 3 and 6 cannot be put today: the replica registers no shard of
a generated set and ``show`` prints no rows of a named map. Until they can, this
file is the only place their comparisons run. Each test gives a comparison a
document that is right and documents that are wrong in one way each, and the
comparison must tell them apart. Nothing here builds a shard.
"""

from __future__ import annotations

import acceptance_maps as maps
import acceptance_sets
import pytest

INPUT = "set-rr/shard-01.input"


def line(n: int, artifact: str = INPUT) -> dict:
    return {"artifact": artifact, "instance": "72b0", "record": f"L{n}", "members": []}


def recovered(ordinal: int, start: int, end: int, parent: int) -> dict:
    return {"record": f"D{ordinal}", "label": "recovered", "token_start": start, "token_end": end,
            "parent": line(parent), "candidates": []}


def twin(ordinal: int, start: int, end: int, lines: tuple[int, ...]) -> dict:
    return {"record": f"D{ordinal}", "label": "candidates", "token_start": start,
            "token_end": end, "parent": None, "candidates": [line(n) for n in lines]}


def unknown(ordinal: int, start: int, end: int) -> dict:
    return {"record": f"D{ordinal}", "label": "unknown", "token_start": start, "token_end": end,
            "parent": None, "candidates": []}


# A shard of four documents: lines 2 and 4 of its input hold the same text.
ROWS = [recovered(0, 0, 10, 1), twin(1, 10, 20, (2, 4)), recovered(2, 20, 26, 3),
        twin(3, 26, 36, (2, 4))]
SIZES = [10, 10, 6, 10]
IDS = ["Z-000007", "C-000301", "T-000004", "Z-000090"]  # the planted record of each line
PAIR = {"pair": "PAIR-000001", "kind": "pair-member", "placement": "same-shard", "members": [
    {"id": "C-000301", "source": "C", "shard": "shard-01", "line": 2, "ordinal": 1},
    {"id": "Z-000090", "source": "Z", "shard": "shard-01", "line": 4, "ordinal": 3}]}

SHOWN = {"artifact": "set-rr/shard-01", "kind": "shard", "bindings": [
    {"namespace": "capture", "declared_count": 4, "derivation": "position",
     "verification": "passed", "label": "recovered"},
    {"namespace": "recovery", "declared_count": 4, "derivation": "position",
     "verification": "failed", "label": "unknown"}]}


def identify(named: maps.Place) -> str | None:
    artifact, number = named
    return IDS[number - 1] if artifact == INPUT and 1 <= number <= len(IDS) else None


def changed(rows: list[dict], ordinal: int, **fields) -> list[dict]:
    return [dict(row, **fields) if n == ordinal else row for n, row in enumerate(rows)]


# --- The form of what is printed ---------------------------------------------------------------


def test_a_map_is_read_row_by_row_and_another_form_is_an_error():
    assert maps.map_rows({"map": {}, "rows": ROWS}) == ROWS
    assert maps.place(line(3)) == (INPUT, 3)
    with pytest.raises(maps.OutputForm, match="is the document D1"):
        maps.map_rows({"rows": [ROWS[0], ROWS[2]]})          # a row is missing
    with pytest.raises(maps.OutputForm, match="no 'token_end'"):
        maps.map_rows({"rows": [{k: v for k, v in ROWS[0].items() if k != "token_end"}]})
    with pytest.raises(maps.OutputForm, match="rows as a list"):
        maps.map_rows({"artifact": "set-rr/shard-01"})        # what show prints today
    with pytest.raises(maps.OutputForm, match="L<n>"):
        maps.place({"artifact": INPUT, "record": "D3"})
    assert maps.binding(SHOWN, "recovery", "position")["verification"] == "failed"
    assert maps.binding(SHOWN, "recovery", "token-matching") is None


# --- Spans ---------------------------------------------------------------------------------------


def test_a_span_is_held_against_the_size_in_the_index_and_the_sum_against_the_total():
    rows, sizes = {"shard-01": ROWS}, {"shard-01": SIZES}
    assert maps.span_findings(rows, sizes) == []
    assert maps.total_findings(rows, sizes) == []
    assert maps.count_field("mismatches", 0, []) == []
    # One span one token short: one ordinal differs, and so does the sum.
    short = {"shard-01": changed(ROWS, 2, token_end=25)}
    assert maps.span_findings(short, sizes) == ["shard-01 D2: span 5, index 6"]
    assert maps.total_findings(short, sizes) == ["shard-01: spans sum to 35, the index holds 36"]
    (mismatch,) = maps.count_field("mismatches", 0, maps.span_findings(short, sizes))
    assert mismatch.field == "mismatches" and mismatch.expected == 0
    # A map with a row too few, and one with a row too many.
    assert maps.span_findings({"shard-01": ROWS[:3]}, sizes) == ["shard-01 D3: no row"]
    assert len(maps.span_findings(rows, {"shard-01": SIZES[:3]})) == 1
    # Two wrong spans can cancel in the sum: the per-ordinal question still sees both.
    moved = {"shard-01": changed(changed(ROWS, 0, token_end=11), 1, token_start=11)}
    assert maps.total_findings(moved, sizes) == []
    assert len(maps.span_findings(moved, sizes)) == 2


# --- Which record a row names --------------------------------------------------------------------


def test_each_ordinal_names_its_planted_record():
    identical = [[1, 3]]
    assert maps.record_findings(ROWS, IDS, identical, identify, "shard-01",
                                twins_as_set=False) == []
    assert maps.record_findings(ROWS, IDS, identical, identify, "shard-01",
                                twins_as_set=True) == []
    # A displaced document: ordinal 2 is given line 1.
    displaced = changed(ROWS, 2, parent=line(1))
    (found,) = maps.record_findings(displaced, IDS, identical, identify, "shard-01",
                                    twins_as_set=False)
    assert "D2" in found and "T-000004" in found
    # A parent in a file whose line carries another id, or no id.
    elsewhere = changed(ROWS, 0, parent=line(1, "set-rr/shard-02.input"))
    assert len(maps.record_findings(elsewhere, IDS, identical, identify, "s",
                                    twins_as_set=False)) == 1
    # An unknown row, and a candidate set where one record was expected.
    assert len(maps.record_findings(changed(ROWS, 0, **{k: v for k, v in unknown(0, 0, 10).items()
                                                     if k != "record"}),
                                    IDS, identical, identify, "s", twins_as_set=False)) == 1
    assert len(maps.record_findings(ROWS, IDS, [], identify, "s", twins_as_set=False)) == 2
    # Check 6 asks for the whole candidate set of a twin, Check 3's record question
    # only that the planted record is among the candidates.
    half = changed(ROWS, 1, candidates=[line(2), line(3)])
    assert maps.record_findings(half, IDS, identical, identify, "s", twins_as_set=False) == []
    assert len(maps.record_findings(half, IDS, identical, identify, "s", twins_as_set=True)) == 1
    # A map with fewer rows than the shard has planted records.
    assert len(maps.record_findings(ROWS[:3], IDS, identical, identify, "s",
                                    twins_as_set=False)) == 1


def test_the_token_matching_map_names_the_same_lines_outside_the_twins():
    assert maps.same_as_findings(ROWS, ROWS, [[1, 3]], "shard-01") == []
    other = changed(ROWS, 2, parent=line(1))
    assert len(maps.same_as_findings(other, ROWS, [[1, 3]], "shard-01")) == 1
    # Which twin gets which line means nothing: a difference there is no finding.
    twins = changed(ROWS, 1, candidates=[line(4), line(2)])
    assert maps.same_as_findings(twins, ROWS, [[1, 3]], "shard-01") == []
    assert len(maps.same_as_findings(ROWS[:3], ROWS, [[1]], "shard-01")) == 1


# --- Identical pairs -----------------------------------------------------------------------------


def test_a_pair_in_one_shard_is_candidates_with_exactly_its_two_lines():
    assert maps.twin_findings(ROWS, PAIR, INPUT, "shard-01") == ([], [])
    assert maps.twin_findings(ROWS, PAIR, INPUT, "shard-01", identify) == ([], [])
    # One twin labelled as if the link were known.
    labels, sets = maps.twin_findings(changed(ROWS, 1, **{
        k: v for k, v in recovered(1, 10, 20, 2).items() if k != "record"}), PAIR, INPUT, "s")
    assert len(labels) == 1 and sets == []
    # A candidate set with a third line, and one with a line of another file.
    labels, sets = maps.twin_findings(changed(ROWS, 3, candidates=[line(2), line(3), line(4)]),
                                      PAIR, INPUT, "s")
    assert labels == [] and len(sets) == 1
    other_file = changed(ROWS, 3, candidates=[line(2), line(4, "set-rr/shard-02.input")])
    assert len(maps.twin_findings(other_file, PAIR, INPUT, "s")[1]) == 1
    # After the two lines changed places the set is the same two lines, and the
    # ids on them are still the pair's two members.
    swapped_ids = [IDS[0], IDS[3], IDS[2], IDS[1]]
    assert maps.twin_findings(ROWS, PAIR, INPUT, "s",
                              lambda named: swapped_ids[named[1] - 1]) == ([], [])
    assert len(maps.twin_findings(ROWS, PAIR, INPUT, "s",
                                  lambda named: "Z-000001")[1]) == 2
    # A row that expects another label finds every twin wrong.
    assert len(maps.twin_findings(ROWS, PAIR, INPUT, "s", label="recovered")[0]) == 2


def test_a_member_of_a_split_pair_is_recovered_with_the_line_of_its_own_shard():
    member = {"id": "T-000004", "source": None, "shard": "shard-01", "line": 3, "ordinal": 2}
    assert maps.split_findings(ROWS, member, INPUT, "PAIR-000002") == []
    assert len(maps.split_findings(changed(ROWS, 2, parent=line(1)), member, INPUT, "P")) == 1
    as_twin = changed(ROWS, 2, label="candidates", parent=None, candidates=[line(3), line(9)])
    assert len(maps.split_findings(as_twin, member, INPUT, "P")) == 1
    assert len(maps.split_findings(ROWS[:2], member, INPUT, "P")) == 1
    assert len(maps.split_findings(ROWS, member, INPUT, "P", label="candidates")) == 1


# --- The stored fields of a map ------------------------------------------------------------------


def test_the_fields_of_a_map_are_read_from_its_binding():
    want = {"namespace": "capture", "derivation": "position", "verification": "passed"}
    assert maps.compare_binding("capture", want, SHOWN, "capture", "position") == []
    (wrong,) = maps.compare_binding("position_recovery", dict(want, namespace="recovery"), SHOWN,
                                    "recovery", "position")
    assert (wrong.field, wrong.expected, wrong.got) == (
        "position_recovery.verification", "passed", "failed")
    (missing,) = maps.compare_binding("token_matching_recovery", want, SHOWN, "recovery",
                                      "token-matching")
    assert "holds no map" in missing.got
    assert maps.labels_of(ROWS) == {"recovered": 2, "candidates": 2}
    assert maps.labels_of([unknown(0, 0, 4)]) == {"unknown": 1}


# --- Drops ---------------------------------------------------------------------------------------


def test_every_skipped_line_is_one_drop_and_no_line_is_recorded_twice():
    wanted = [("Z-000249", "sanitizer skip"), ("Z-000149", "empty line")]
    assert maps.drop_findings(list(reversed(wanted)), wanted) == []
    # A drop that is missing, one with another reason, one nobody planted.
    assert maps.drop_findings(wanted[:1], wanted) == ["missing: Z-000149 (empty line)"]
    assert len(maps.drop_findings([wanted[0], ("Z-000149", "sanitizer skip")], wanted)) == 2
    assert len(maps.drop_findings([*wanted, ("set-rr/shard-01.input:L3", "empty line")],
                                  wanted)) == 1
    # The same line recorded under two artifacts is one drop too many.
    (twice,) = maps.drop_findings([*wanted, wanted[0]], wanted)
    assert "recorded twice" in twice and "Z-000249" in twice


# --- The set-ups of Check 3 ----------------------------------------------------------------------


def test_the_set_ups_change_the_lines_the_table_names_and_no_other():
    lines = ['{"id": "a"}', '{"id": "b"}', '{"id": "c"}', '{"id": "d"}']
    assert acceptance_sets.swapped_lines(lines, [PAIR]) == [lines[0], lines[3], lines[2], lines[1]]
    changed_lines = acceptance_sets.changed_first_line(lines)
    assert len(changed_lines) == 4 and changed_lines[1:] == lines[1:]
    assert acceptance_sets.CHANGED_TEXT in changed_lines[0]
    counted = {"check3": {"pairs": {"list": [PAIR, {"pair": "X-000001", "placement": "split",
                                                    "members": []}]}}}
    assert acceptance_sets.same_shard_pairs(counted) == {"shard-01": [PAIR]}
    assert [p["pair"] for p in acceptance_sets.split_pairs(counted)] == ["X-000001"]


# --- A line that gives no tokens -----------------------------------------------------------------


def test_the_planted_line_is_shown_to_give_no_tokens_by_the_tokenizer_and_the_index():
    # Five lines; line 3 is the planted one. The tokenizer's answers, and the entry of
    # each line in the index of the shard that was written from them (each with the
    # end-of-document token).
    given = [[5, 6], [7], [], [8, 9, 10], [11]]
    index = [3, 2, 0, 4, 2]
    assert maps.no_token_findings(given, index, [3], True) == []
    # A planted line that gives tokens: the row would be about a line that is not the case.
    found = maps.no_token_findings([[5, 6], [7], [1], [8, 9, 10], [11]], [3, 2, 2, 4, 2], [3],
                                   True)
    assert [m.field for m in found] == ["lines the tokenizer gives no tokens for",
                                        "lines the shard's index holds no tokens for"]
    # A line that gives no tokens that was not planted.
    found = maps.no_token_findings([[5, 6], [], [], [8, 9, 10], [11]], index, [3], True)
    assert found[0].got == [2, 3]
    # An index that holds tokens for the planted line, or none for another.
    (found,) = maps.no_token_findings(given, [3, 2, 1, 4, 2], [3], True)
    assert found.field == "lines the shard's index holds no tokens for" and found.got == []
    # An index of another length, or of other sizes, than the tokenizer's answers give.
    (found,) = maps.no_token_findings(given, [3, 2, 0, 4], [3], True)
    assert found.field == "lines in the shard's index"
    (found,) = maps.no_token_findings(given, [3, 2, 0, 5, 2], [3], True)
    assert found.field == "tokens of each line in the shard's index"
    assert maps.no_token_findings(given, [2, 1, 0, 3, 1], [3], False) == []
