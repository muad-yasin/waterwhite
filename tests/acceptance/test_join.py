"""Check 1's comparison (``acceptance_join``) shown to fail on rows written by hand.

A row the join has not been asked is skipped by the world, so nothing in the table's
rows shows that the diff can fail. Here it is put to rows of the shapes the plan names
(a missing sample, a changed token end, a row the oracle's log has none of, the cut
of the half-covered sample), with the expected rows taken from the fixture spec.
"""

from __future__ import annotations

import json

import acceptance_join as join
import acceptance_runs as runs

SPEC = runs.spec()


def row(step: int, sample: int, shard: int, ordinal: int, start: int, end: int,
        segment: str = "E/seg-2") -> dict:
    return {"segment": segment, "step": step, "sample_pos": sample, "global_index": 0, "rank": 0,
            "shard_position": shard, "ordinal": ordinal, "token_start": start, "token_end": end}


ORACLE = [row(7, 0, 2, 24, 32, 39), row(7, 0, 2, 7, 0, 22), row(7, 0, 2, 26, 0, 4),
          row(7, 1, 0, 3, 0, 33)]


def test_the_same_rows_give_an_empty_diff():
    assert join.diff(list(ORACLE), ORACLE) == []


def test_a_changed_token_end_is_found_with_its_place():
    reported = [dict(r) for r in ORACLE]
    reported[1]["token_end"] = 21
    (found,) = join.diff(reported, ORACLE)
    assert "('E/seg-2', 7, 0)" in found and "row 1 differs" in found


def test_a_sample_the_join_does_not_report_is_found():
    (found,) = join.diff(ORACLE[:3], ORACLE)
    assert "('E/seg-2', 7, 1)" in found and "reports no row" in found


def test_a_sample_only_the_join_reports_is_found():
    (found,) = join.diff(ORACLE + [row(8, 0, 0, 1, 0, 33)], ORACLE)
    assert "('E/seg-2', 8, 0)" in found and "none of" in found


def test_the_order_of_the_rows_of_a_sample_counts():
    reported = [ORACLE[0], ORACLE[2], ORACLE[1], ORACLE[3]]
    (found,) = join.diff(reported, ORACLE)
    assert "('E/seg-2', 7, 0)" in found and "row 1 differs" in found


def test_a_comparison_can_be_restricted_to_a_step_or_a_sample():
    reported = [dict(r) for r in ORACLE]
    reported[3]["ordinal"] = 4
    assert join.diff(reported, ORACLE, within=(None, 7, 0)) == []
    assert len(join.diff(reported, ORACLE, within=(None, 7, 1))) == 1
    assert len(join.diff(reported, ORACLE)) == 1


def test_the_oracle_and_the_join_are_compared_by_the_planned_fields_only():
    reported = [dict(r, global_index=99, rank=1) for r in ORACLE]
    assert join.diff(reported, ORACLE) == []


def test_the_cut_sample_of_step_8_ends_at_the_cut_the_spec_states():
    fixed = SPEC["positions"]["step8_cut"]
    lines = [json.loads(line) for line in fixed["log_lines"]
             if json.loads(line)["step"] == 8 and json.loads(line)["sample_pos"] == 3]
    cut = join.cut_sample(join.group(lines)[("E/seg-2", 8, 3)], fixed["fixed"]["covered_tokens"])
    wanted = [(c["shard_position"], c["ordinal"], c["token_start"], c["token_end"])
              for c in fixed["rows_of_the_covered_part"]]
    assert cut == wanted


def test_a_sample_cut_at_a_row_boundary_keeps_whole_rows_and_a_cut_inside_a_row_shortens_it():
    spans = [(0, 1, 0, 10), (0, 2, 0, 10), (0, 3, 0, 13)]
    assert join.cut_sample(spans, 10) == [(0, 1, 0, 10)]
    assert join.cut_sample(spans, 15) == [(0, 1, 0, 10), (0, 2, 0, 5)]
    assert join.cut_sample(spans, 33) == spans
    assert join.cut_sample(spans, 0) == []


def test_the_diff_cuts_the_lowered_sample_before_comparing():
    oracle = [row(8, 3, 2, 2, 19, 44), row(8, 3, 2, 23, 0, 8)]
    cut = {("E/seg-2", 8, 3): 16}
    assert join.diff([row(8, 3, 2, 2, 19, 35)], oracle, cut) == []
    assert len(join.diff(oracle, oracle, cut)) == 1  # a join that reports past the cut
    assert len(join.diff([row(8, 3, 2, 2, 19, 36)], oracle, cut)) == 1


def test_a_cut_of_a_sample_the_oracle_does_not_hold_is_an_error_of_the_test():
    import pytest

    with pytest.raises(ValueError, match="no sample"):
        join.diff([], ORACLE, {("E/seg-2", 8, 3): 16})


def test_the_rows_of_a_position_the_spec_states_must_be_among_the_joins_rows():
    fixed = SPEC["positions"]["T"]
    reported = [json.loads(line) for line in fixed["log_lines"]]
    assert join.rows_named(fixed["log_lines"], reported) == []
    del reported[1]
    (missing,) = join.rows_named(fixed["log_lines"], reported)
    assert "(2, 7, 0, 22)" in missing
