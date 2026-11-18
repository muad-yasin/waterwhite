"""Check 9, repair (plan 5.5 Check 9): the rows of fixture/table/check09.toml, one
test per row, and the tests of the reading, the expected values and the comparisons
the rows use (acceptance_check09.py).

A repair is three commands: `waterwhite report` (docs/interfaces.md 7.1), the replica's
`repair` (9.5) and `waterwhite verify` (7.3). A row skips with "not built yet: <the exact
output>" while its outputs are missing; once they exist it is put to the tool's documents,
read in the form the docs fix (`acceptance_report.py`), and graded. A skipped row is not a
pass.

How a row becomes a test, and how to read the counts: README.md in this folder.
The rows of the 10k set are asked in both layouts inside one test, since the
summary of conftest.py counts one test per row; they carry no marker of their own
and are held back by the long-run gate (WATERWHITE_CLEARED_LONG_RUNS naming 9:10k),
checked after "not built yet".

The tests below the rows put every reading and every comparison to documents written by
hand, one that agrees and one that differs, so that each can be seen to fail: the tool's
commands are not in the tree they were written in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

import acceptance_check09 as c9
import acceptance_sets
import pytest
import table
from acceptance_rows import FALLBACKS_IN_FORCE, PROPERTY
from table import Mismatch, Outcome, TableError

ROWS = table.load_check(c9.CHECK)


def _execute(repair: c9.Repair, decision: c9.Decision, note: dict, world, repairs) -> None:
    __tracebackhide__ = True
    if decision.action == c9.SKIP:
        note["skip"] = decision.reason
        pytest.skip(decision.reason)
    try:
        mismatches = (c9.put_fixture(repair, world) if repair.size is None else
                      c9.put_set(repair, repairs, repairs.plant_rows(repair.size)))
    except acceptance_sets.Unavailable as missing:
        note["skip"] = str(missing)
        pytest.skip(str(missing))
    if mismatches:
        raise table.RowMismatch(repair.id, mismatches)


@pytest.mark.parametrize("row", ROWS, ids=[row.id for row in ROWS])
def test_check09(row, request, world, repairs):
    note = {"check": c9.CHECK, "row": row.id, "part": "row", "skip": None, "size": None}
    request.node.user_properties.append((PROPERTY, note))
    repair = c9.read(row, FALLBACKS_IN_FORCE)  # a row that cannot be read is an error of the test, never a skip
    decision = c9.decide(repair, c9.current_gaps(), megatron=acceptance_sets.megatron_root(),
                         long_run_cleared_for_size=bool(
                             repair.size and c9.long_run_cleared(repair.size)))
    _execute(repair, decision, note, world, repairs)


# --- The reading of the rows ---------------------------------------------------------


def _row(given: dict, expect: dict, **more) -> table.Row:
    base = dict(id="c09.test", check=9, question="q", plan="5.5 Check 9", core=True,
                extension=None, given=given, expect=expect, expect_under={}, source="hand-written table",
                rule=None)
    base.update(more)
    return table.Row(**base)


def test_every_row_of_the_file_is_read_and_has_a_kind():
    repairs = c9.read_all()
    assert len(repairs) == len(ROWS) == 25
    assert {r.kind for r in repairs} <= set(c9.WAITS)
    assert set(c9.WAITS) == set(c9.SIGNATURES.values()) | {c9.K_HISTORY}


def test_the_rows_of_the_10k_set_name_their_layouts_and_the_others_name_none():
    for repair in c9.read_all():
        assert (repair.size == "10k") == repair.id.startswith("c09.10k."), repair.id
        assert bool(repair.layouts) == (repair.size is not None) or repair.id in (
            "c09.fixture.shards.counts",), repair.id


def test_the_two_wrong_reports_and_the_seeded_row_are_read_with_their_states_and_mode():
    by_id = {r.id: r for r in c9.read_all()}
    assert by_id["c09.10k.wrong-report"].state == "r.descendant-kept"
    assert by_id["c09.10k.wrong-report.unrelated-paraphrase-retired"].state == (
        "r.unrelated-paraphrase-retired")
    assert by_id["c09.10k.shard-digests"].seeded and not by_id["c09.10k.multiset"].seeded
    assert by_id["c09.10k.wrong-report"].kind == c9.K_DIFFERENCE
    assert by_id["c09.10k.R1-history"].kind == c9.K_HISTORY
    assert by_id["c09.fixture.R1-history"].kind == c9.K_HISTORY
    assert by_id["c09.10k.successor-shards"].kind == c9.K_SUCCESSOR
    assert by_id["c09.fixture.R1.action"].kind == c9.K_REPORT_ACTION


@pytest.mark.parametrize(("given", "expect"), [
    ({"request": "withdraw A", "colour": "red"}, {"answer": "contains"}),          # unknown key
    ({"request": "withdraw A", "state": "f.trace-deleted"}, {"answer": "contains"}),  # a state of another check
    ({"request": "withdraw C"}, {"answer": "contains"}),                            # another request
    ({"request": "withdraw A", "set": "100k", "layouts": ["partitioned"]}, {"answer": "contains"}),
    ({"request": "withdraw A", "set": "10k"}, {"answer": "contains"}),              # no layouts
    ({"request": "withdraw A", "artifact": "R9"}, {"answer": "contains"}),          # no such artifact
    ({"request": "withdraw A", "mode": "unseeded"}, {"answer": "contains"}),
    ({"request": "withdraw A"}, {"colour": "red"}),                                 # unknown expected field
])
def test_a_row_this_reader_does_not_know_is_an_error_and_never_a_skip(given, expect):
    with pytest.raises(TableError):
        c9.read(_row(given, expect))


# --- What a row waits for ------------------------------------------------------------

ALL_MISSING = {c9.GATE_REPORT: True, c9.GATE_VERIFY: True, c9.GATE_SUCCESSOR: True}
NONE_MISSING = {c9.GATE_REPORT: False, c9.GATE_VERIFY: False, c9.GATE_SUCCESSOR: False}


def _decide(repair, gaps, *, megatron=object(), cleared=True, spec_=None):
    return c9.decide(repair, gaps, megatron=megatron, long_run_cleared_for_size=cleared,
                     spec_=spec_)


def test_with_the_outputs_missing_every_row_skips_and_names_the_exact_thing():
    for repair in c9.read_all():
        decision = _decide(repair, ALL_MISSING)
        assert decision.action == c9.SKIP and decision.reason.startswith("not built yet: "), repair.id
        assert "`report`" in decision.reason and "plan 4." in decision.reason, repair.id


def test_the_reason_lists_only_the_outputs_that_are_still_missing():
    repair = next(r for r in c9.read_all() if r.id == "c09.10k.multiset")
    only_verify = {**NONE_MISSING, c9.GATE_VERIFY: True}
    reason = _decide(repair, only_verify).reason
    assert "`verify`" in reason and "`report`" not in reason and "repaired side" not in reason


def test_the_machine_and_the_long_run_gate_come_after_not_built_yet():
    by_id = {r.id: r for r in c9.read_all()}
    shards = by_id["c09.10k.partitioned.kept-shards"]
    assert _decide(shards, ALL_MISSING, megatron=None, cleared=False).reason.startswith(
        "not built yet: ")
    assert _decide(shards, NONE_MISSING, megatron=None, cleared=False).reason.startswith(
        "not run: its shards are tokenized only where the Megatron environment is set")
    assert _decide(shards, NONE_MISSING, cleared=False).reason.startswith("a long run: Check 9 on the 10k")
    assert _decide(shards, NONE_MISSING).action == c9.PUT
    counts = by_id["c09.fixture.R1.counts"]  # reads release rows only: no Megatron, no long run
    assert _decide(counts, NONE_MISSING, megatron=None, cleared=False).action == c9.PUT


def test_a_pending_value_the_spec_does_not_hold_skips_as_not_checked():
    repair = next(r for r in c9.read_all() if r.id == "c09.E.exact-rerun-restart")
    assert _decide(repair, NONE_MISSING).action == c9.PUT
    reason = _decide(repair, NONE_MISSING, spec_={"positions": {}}).reason
    assert reason.startswith("not checked: ") and "positions.exact_rerun_restart.step" in reason


def test_the_long_run_gate_reads_its_variable():
    assert c9.long_run_cleared("10k", "3:100k, 9:10k") and not c9.long_run_cleared("10k", "3:10k")
    assert not c9.long_run_cleared("10k", "")


def test_a_row_whose_outputs_all_exist_is_put_and_its_differences_fail_it(monkeypatch):
    repair = c9.read_all()[0]
    note = {"skip": None}
    monkeypatch.setattr(c9, "put_set", lambda *a, **k: [Mismatch("multisets", "equal", "differ")])
    with pytest.raises(table.RowMismatch, match="multisets"):
        _execute(repair, _decide(repair, NONE_MISSING), note, None, _Stub())
    monkeypatch.setattr(c9, "put_set", lambda *a, **k: [])
    _execute(repair, _decide(repair, NONE_MISSING), note, None, _Stub())  # no difference: passes
    with pytest.raises(pytest.skip.Exception):
        _execute(repair, _decide(repair, ALL_MISSING), note, None, _Stub())
    assert note["skip"].startswith("not built yet: ")


class _Stub:
    def plant_rows(self, size):
        return []


def test_the_gaps_are_the_probes_and_each_is_a_boolean():
    # Not an assertion about the tool: the three commands are asked, the answer is a boolean.
    assert set(c9.current_gaps()) == {c9.GATE_REPORT, c9.GATE_VERIFY, c9.GATE_SUCCESSOR}
    assert all(isinstance(v, bool) for v in c9.current_gaps().values())


# --- Expected values from the plant list ---------------------------------------------


@pytest.fixture(scope="module")
def counting():
    return acceptance_sets.load_script("count.py")


@pytest.fixture(scope="module")
def set10k(tmp_path_factory):
    folder = tmp_path_factory.mktemp("check09-10k") / "set"
    acceptance_sets.load_script("generate.py").generate(
        folder, dedup=acceptance_sets.DEDUP["10k"], **acceptance_sets.SIZES["10k"])
    return folder


@pytest.fixture(scope="module")
def counted10k(set10k, counting):
    return counting.count(set10k)


@pytest.fixture(scope="module")
def plant_rows(set10k, counting):
    return counting.load_set(set10k)[1]


def test_the_counts_of_the_10k_rows_are_the_scripts_and_hold_together(counted10k):
    total = c9.expected_record_counts(counted10k)
    r1 = c9.expected_record_counts(counted10k, "R1")
    r2 = c9.expected_record_counts(counted10k, "R2")
    # D-19: records removed is the number of rows that leave, which is the retire count.
    assert total["records_removed"] == total["retire"] == r1["retire"] + r2["retire"]
    # Plan 4.1: re-source and promote are the two rows of one event.
    assert total["re-source"] == total["promote"] > 0
    assert r2["keep"] > 0 and r2["retire"] == 0
    assert total["keep"] == r1["keep"] + r2["keep"]
    assert set(total) == {*c9.RECORD_ACTIONS, "records_removed"}


def test_the_actions_by_kind_name_every_planted_x_and_p_t_s(counted10k, plant_rows):
    expect = next(r for r in ROWS if r.id == "c09.10k.actions-by-kind").expect
    groups = c9.expected_actions_by_kind(counted10k, plant_rows, expect)
    assert set(groups) == {"X_release_rows", "B_copies_of_X", "P_T_S"}
    assert set(groups["X_release_rows"].values()) == {"re-source"}
    assert set(groups["B_copies_of_X"].values()) == {"promote"}
    assert set(groups["P_T_S"].values()) == {"retire"}
    sizes = Counter(row["kind"] for row in plant_rows)
    assert len(groups["P_T_S"]) == sizes["paraphrase"] + sizes["translation"] + sizes["summary"]
    # The two ways of counting agree: the groups hold exactly the ids the script counts.
    counts = c9.expected_record_counts(counted10k, "R1")
    assert len(groups["X_release_rows"]) == counts["re-source"]
    assert len(groups["B_copies_of_X"]) == counts["promote"]
    assert len(groups["P_T_S"]) == counts["retire"]


def test_a_row_word_the_script_does_not_count_is_a_disagreement_and_never_a_keep(
        counted10k, plant_rows):
    expect = {"X_release_rows": "re-source", "B_copies_of_X": "keep", "P_T_S": "retire"}
    with pytest.raises(c9.PlantDisagreement, match="promote"):
        c9.expected_actions_by_kind(counted10k, plant_rows, expect)


def test_the_record_a_wrong_report_retires_is_the_first_generated_record_that_stays(
        counted10k, plant_rows):
    record = c9.expected_unrelated_paraphrase(counted10k)
    row = next(r for r in plant_rows if r["id"] == record)
    assert record == "UP-000001" and row["withdraw_A"]["in_scope"] is False
    assert row["releases"]["R1"]["withdraw_A"] == "keep"


def test_the_kept_shard_list_needs_the_chunk_files_of_the_build_asked(counted10k):
    with pytest.raises(c9.NeedsChunkFiles):
        c9.expected_shard_actions(counted10k)


def _chunked(folder, out, layout):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(c9.SRC), str(c9.REPO)])}
    done = subprocess.run([sys.executable, "-m", "replica", "build-set", "--set", str(folder),
                           "--out", str(out), "--layout", layout, "--no-tokenize"],
                          cwd=c9.REPO, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-500:]
    return out / "pipeline" / layout / "tokenizer-input"


@pytest.fixture(scope="module")
def small_chunked(tmp_path_factory, counting):
    base = tmp_path_factory.mktemp("check09-small")
    folder = base / "set"
    acceptance_sets.load_script("generate.py").generate(
        folder, dedup=acceptance_sets.DEDUP[acceptance_sets.SMALL],
        **acceptance_sets.SIZES[acceptance_sets.SMALL])
    # The small set is built with dedup off, which asks for pair kinds the chunk files of a
    # partitioned layout cannot show; the pairs are Check 3's, not this check's.
    return {layout: counting.count(folder, chunks=_chunked(folder, base / layout, layout),
                                   layout=layout, require_pair_kinds="no")
            for layout in c9.LAYOUTS}


def test_the_kept_shards_and_the_all_rewrite_row_read_from_the_chunk_files(small_chunked):
    partitioned = c9.expected_shard_actions(small_chunked["partitioned"])
    assert partitioned["keep"] and partitioned["rewrite"]
    assert not set(partitioned["keep"]) & set(partitioned["rewrite"])
    round_robin = c9.expected_shard_actions(small_chunked["round-robin"])
    assert round_robin["keep"] == [] and len(round_robin["rewrite"]) == 20


# --- The comparisons, put to documents written by hand so that they can be seen to fail


def test_the_multisets_of_two_equal_sides_are_equal_whatever_the_order():
    result = c9.compare_multisets(["a", "b", "b"], ["b", "a", "b"])
    assert result.equal and result.differing_records == 0
    assert c9.check_multisets({"multisets": "equal", "unresolved_records": 0}, result, 0) == []


def test_a_kept_descendant_makes_the_sides_differ_by_one_record():
    result = c9.compare_multisets(["a", "b", "T"], ["a", "b"])
    assert not result.equal and result.differing_records == 1
    assert c9.check_multisets({"multisets": "differ", "differing_records": 1}, result) == []
    wrong = c9.check_multisets({"multisets": "equal", "unresolved_records": 0}, result, 0)
    assert [m.field for m in wrong] == ["multisets"]


def test_a_retired_record_the_reference_holds_makes_the_sides_differ_by_one_record():
    result = c9.compare_multisets(["a", "b"], ["a", "b", "UP-000001 text"])
    assert not result.equal and result.differing_records == 1


def test_a_record_held_twice_on_one_side_and_once_on_the_other_differs():
    result = c9.compare_multisets(["a", "a"], ["a"])
    assert not result.equal and result.differing_records == 1


def test_two_sides_that_differ_by_two_records_are_not_taken_for_one():
    found = c9.check_multisets({"multisets": "differ", "differing_records": 1},
                               c9.compare_multisets(["a", "x", "y"], ["a"]))
    assert [m.field for m in found] == ["differing_records"]


def test_an_unresolved_record_the_report_holds_is_a_difference():
    found = c9.check_multisets({"multisets": "equal", "unresolved_records": 0},
                               c9.compare_multisets(["a"], ["a"]), 2)
    assert [(m.field, m.got) for m in found] == [("unresolved_records", 2)]


def test_an_expected_multiset_word_that_is_neither_equal_nor_differ_is_an_error():
    with pytest.raises(TableError):
        c9.check_multisets({"multisets": "same"}, c9.compare_multisets([], []))


def test_the_counts_are_compared_one_by_one_and_a_missing_count_is_a_difference():
    expected = {"keep": 20, "re-source": 1, "promote": 1, "retire": 3, "unresolved": 0}
    assert c9.check_counts(expected, dict(expected)) == []
    got = {**expected, "retire": 2}
    del got["unresolved"]
    assert c9.check_counts(expected, got) == [Mismatch("counts.retire", 3, 2),
                                              Mismatch("counts.unresolved", 0, None)]


def test_the_fixture_rows_expected_counts_are_the_tables_and_add_up():
    row = next(r for r in ROWS if r.id == "c09.fixture.R1.counts")
    counts = row.expect["counts"]
    assert counts["keep"] + counts["re-source"] + counts["retire"] == 24
    assert row.expect["rows_in_successor"] == counts["keep"] + counts["re-source"]
    assert row.expect["records_removed"] == counts["retire"]


def test_a_record_action_outside_the_four_words_is_found_and_remove_is_not_a_word():
    allowed = ["keep", "re-source", "promote", "retire"]
    assert c9.check_record_action_words(allowed, {"A-1": "retire", "B-1": "promote"}) == []
    (found,) = c9.check_record_action_words(allowed, {"A-1": "remove", "B-1": "keep"})
    assert found.got == ["remove"]


def test_an_action_by_kind_that_is_wrong_or_missing_is_found():
    expected = {"A-1": "re-source", "A-2": "re-source"}
    assert c9.check_actions(expected, {"A-1": "re-source", "A-2": "re-source"}) == []
    assert c9.check_actions(expected, {"A-1": "re-source", "A-2": "keep"}) == [
        Mismatch("actions[A-2]", "re-source", "keep")]
    assert c9.check_actions(expected, {"A-1": "re-source"}) == [
        Mismatch("actions[A-2]", "re-source", None)]


SHARDS = {"keep": ["shard-C", "shard-Z"], "rewrite": ["shard-A", "shard-gen"]}


def test_the_shard_actions_must_be_exactly_the_kept_shard_list():
    got = {"shard-C": "keep", "shard-Z": "keep", "shard-A": "rewrite", "shard-gen": "rewrite"}
    same = {"shard-C": "x", "shard-Z": "y"}
    assert c9.check_shard_actions(SHARDS, got, same, dict(same)) == []
    assert [m.field for m in c9.check_shard_actions(SHARDS, {**got, "shard-Z": "rewrite"})] == [
        "shard actions[shard-Z]"]
    assert [m.field for m in c9.check_shard_actions(SHARDS, {**got, "shard-A": "keep"})] == [
        "shard actions[shard-A]"]
    assert [m.field for m in c9.check_shard_actions(
        SHARDS, {k: v for k, v in got.items() if k != "shard-gen"})] == [
        "shard actions[shard-gen]"]


def test_a_kept_shard_that_changed_its_bytes_or_has_no_digest_is_found():
    got = {"shard-C": "keep", "shard-Z": "keep", "shard-A": "rewrite", "shard-gen": "rewrite"}
    before = {"shard-C": "x", "shard-Z": "y"}
    (changed,) = c9.check_shard_actions(SHARDS, got, before, {"shard-C": "x", "shard-Z": "z"})
    assert changed.field == "byte identical afterwards[shard-Z]"
    (missing,) = c9.check_shard_actions(SHARDS, got, before, {"shard-C": "x"})
    assert missing.got is None


def test_a_set_of_mixtures_is_compared_as_a_set():
    assert c9.check_set("kept", ["E/mix-1", "D/mix-1"], ["D/mix-1", "E/mix-1"]) == []
    assert c9.check_set("kept", ["E/mix-1", "D/mix-1"], ["E/mix-1"])[0].field == "kept"


def test_an_answer_that_is_not_the_expected_one_is_found():
    assert c9.check_answer({"answer": "does not contain"}, "does not contain") == []
    assert c9.check_answer({"answer": "does not contain"}, "may contain")[0].got == "may contain"
    assert c9.check_answer({"answer": "contains"}, None)[0].got is None


def test_the_exposure_free_restart_is_compared_and_a_missing_one_is_a_difference():
    expect = {"exposure_free_restart": "E@6"}
    assert c9.check_restart(expect, {"exposure_free_restart": "E@6"}) == []
    assert c9.check_restart(expect, {"exposure_free_restart": "E@9"})[0].got == "E@9"
    assert c9.check_restart(expect, {})[0].got is None


def test_the_exact_rerun_step_is_the_specs_and_no_step_before_the_bound_is_allowed():
    row = next(r for r in ROWS if r.id == "c09.E.exact-rerun-restart")
    expect, empty = table.fill_pending(dict(row.expect), c9.spec())
    assert empty == () and expect["exact_rerun_restart_step"] == 7 and expect["not_before_step"] == 7
    assert c9.check_restart(expect, {"exact_rerun_restart_step": 7}) == []
    early = c9.check_restart(expect, {"exact_rerun_restart_step": 6})
    assert [m.field for m in early] == ["exact_rerun_restart_step", "not_before_step"]
    assert [m.field for m in c9.check_restart(expect, {"exact_rerun_restart_step": 9})] == [
        "exact_rerun_restart_step"]
    assert [m.field for m in c9.check_restart(expect, {})] == [
        "exact_rerun_restart_step", "not_before_step"]


def test_the_stale_caches_are_compared_field_by_field():
    expect = {"stale": True, "rebuilt": True, "rebuilt_from": "an empty cache folder"}
    assert c9.check_stale(expect, dict(expect)) == []
    found = c9.check_stale(expect, {"stale": False, "rebuilt": True})
    assert [m.field for m in found] == ["stale", "rebuilt_from"]
    assert c9.check_stale({"stale": False}, {"stale": False}) == []
    assert c9.check_stale({"stale": False}, {})[0].got is None


def test_the_fallback_f7_changes_what_the_exact_rerun_row_expects_and_is_read_and_compared():
    row = next(r for r in ROWS if r.id == "c09.E.exact-rerun-restart")
    assert set(c9.read(row).expect) == {"exact_rerun_restart_step", "not_before_step"}
    under = c9.read(row, ("F7",))
    assert under.kind == c9.K_EXACT_RERUN and set(under.expect) == {"exact_rerun_restart",
                                                                    "step_comparison"}
    assert c9.check_restart(under.expect, {"exact_rerun_restart": under.expect[
        "exact_rerun_restart"], "step_comparison": "not possible"}) == []
    found = c9.check_restart(under.expect, {"exact_rerun_restart_step": 7})
    assert [m.field for m in found] == ["exact_rerun_restart", "step_comparison"]


# --- The commands that make a repair: the probe, and the words of the form --------------------

import copy  # noqa: E402

import acceptance_probe as probe  # noqa: E402
import acceptance_repair as repair_files  # noqa: E402
import acceptance_report as rep  # noqa: E402
from test_check07 import StubWorld, _report  # noqa: E402

INVALID_REPORT = Outcome(1, "", json.dumps({"error": "usage", "detail":
                                            "argument <command>: invalid choice: 'report' (choose "
                                            "from 'init', 'query')"}))
INVALID_REPAIR = Outcome(2, "", "python -m replica: error: argument command: invalid choice: "
                                "'repair' (choose from 'build-fixture', 'build-set')")


def test_a_replica_that_does_not_know_repair_is_missing_it_and_one_that_prints_help_has_it():
    assert probe.classify_replica_command(INVALID_REPAIR, "repair") is True
    assert probe.classify_replica_command(Outcome(0, "usage: python -m replica repair", ""),
                                          "repair") is False


def test_a_tool_that_does_not_know_report_leaves_the_report_waiting_and_one_that_does_not():
    assert set(probe.classify_command(INVALID_REPORT, "report", table.NEED_REPORT)) == {
        table.NEED_REPORT}
    assert probe.classify_command(Outcome(0, "usage: waterwhite report", ""), "report",
                                  table.NEED_REPORT) == {}
    with pytest.raises(probe.ProbeError):   # the parser refused another word
        probe.classify_command(INVALID_REPORT, "verify", table.NEED_VERIFY)


def test_any_other_ending_of_the_replica_probe_is_an_error_of_the_probe_and_not_a_skip():
    with pytest.raises(probe.ProbeError):
        probe.classify_replica_command(Outcome(1, "", "ImportError: no module named numpy"),
                                       "repair")
    with pytest.raises(probe.ProbeError):   # the parser refused another word
        probe.classify_replica_command(INVALID_REPAIR, "digests")


def test_the_gaps_of_the_three_commands_are_told_apart_and_a_successor_waits_for_two():
    reasons = probe.REASONS
    missing_report = {table.NEED_REPORT: reasons[table.NEED_REPORT]}
    missing_verify = {table.NEED_VERIFY: reasons[table.NEED_VERIFY]}
    assert probe.repair_gaps_of({}, {}, False) == {}
    only_verify = probe.repair_gaps_of({}, missing_verify, False)
    assert set(only_verify) == {table.NEED_VERIFY}
    only_repair = probe.repair_gaps_of({}, {}, True)
    assert set(only_repair) == {table.NEED_SUCCESSOR} and "`repair`" in only_repair[
        table.NEED_SUCCESSOR] and "`report`" not in only_repair[table.NEED_SUCCESSOR]
    no_report = probe.repair_gaps_of(missing_report, {}, False)
    assert set(no_report) == {table.NEED_REPORT, table.NEED_SUCCESSOR}
    assert "`report`" in no_report[table.NEED_SUCCESSOR]
    assert set(probe.repair_gaps_of(missing_report, missing_verify, True)) == {
        table.NEED_REPORT, table.NEED_VERIFY, table.NEED_SUCCESSOR}


def test_the_needs_of_the_two_checks_carry_the_words_the_skipped_rows_name():
    assert (table.NEED_REPORT, table.NEED_SUCCESSOR) == ("rebuild report", "successor")
    assert {table.NEED_REPORT, table.NEED_VERIFY, table.NEED_SUCCESSOR} <= set(probe.REASONS)
    assert set(probe.repair_gaps()) <= {table.NEED_REPORT, table.NEED_VERIFY, table.NEED_SUCCESSOR}


def test_the_records_a_row_names_are_one_handle_a_list_or_a_range():
    assert repair_files.handles_of({"record": "X"}) == ["X"]
    assert repair_files.handles_of({"records": ["P", "T", "S"]}) == ["P", "T", "S"]
    named = repair_files.handles_of({"records": "U-001..U-020"})
    assert len(named) == 20 and named[0] == "U-001" and named[-1] == "U-020"
    with pytest.raises(TableError):
        repair_files.handles_of({"artifact": "R1"})


def test_a_wrong_report_changes_one_entry_and_a_report_that_lacks_it_is_an_error_of_the_run():
    report = _report()
    edited = repair_files.edited_report(report, "R1", "R1:L3", "retire", "keep")
    assert [e["action"] for e in edited["records"] if e["record"] == "R1:L3"] == ["keep"]
    assert [e["action"] for e in report["records"] if e["record"] == "R1:L3"] == ["retire"]
    assert sum(a["action"] != b["action"] for a, b in zip(
        report["records"], edited["records"], strict=True)) == 1
    with pytest.raises(repair_files.RepairError, match="whose action is 'keep'"):
        repair_files.edited_report(report, "R1", "R1:L3", "keep", "retire")   # it is a retire
    with pytest.raises(repair_files.RepairError):
        repair_files.edited_report(report, "R1", "R1:L99", "keep", "retire")  # no such entry


def test_the_manifest_line_of_a_handle_is_read_from_the_replicas_file(tmp_path):
    (tmp_path / "releases").mkdir()
    (tmp_path / "releases" / "R1.jsonl.manifest.jsonl").write_text(
        "".join(json.dumps({"line": n, "handle": h}) + "\n"
                for n, h in enumerate(["A-1", "T-000001", "UP-000001"], 1)), encoding="utf-8")
    assert repair_files.manifest_line(tmp_path, "releases/R1.jsonl", "T-000001") == 2
    with pytest.raises(repair_files.RepairError):
        repair_files.manifest_line(tmp_path, "releases/R1.jsonl", "T-000002")


def test_the_own_multiset_is_read_from_the_files_the_sides_name_and_sees_a_kept_record(tmp_path):
    def side(name, texts):
        path = tmp_path / name
        path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
        return path

    repaired = {"R1": side("r.jsonl", ["a", "b", "b"]), "E/shard-1": tmp_path / "shard-1"}
    equal = {"R1": side("e.jsonl", ["b", "a", "b"]), "E/shard-1": tmp_path / "shard-1"}
    kept = {"R1": side("k.jsonl", ["a", "b"]), "E/shard-1": tmp_path / "shard-1"}
    assert repair_files.own_multiset_gap(repaired, equal) == (1, 0)
    assert repair_files.own_multiset_gap(repaired, kept) == (1, 1)
    assert repair_files.own_multiset_gap(repaired, {}) == (0, 0)  # nothing named on both sides


UNWRITTEN = "line Z-000057 was never written as a record: some words\n"


def _jsonl(tmp_path, name, *lines):
    path = tmp_path / name
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _text_line(text):
    return json.dumps({"id": "x", "text": text}) + "\n"


def test_a_line_that_holds_no_text_is_counted_and_is_not_an_error_of_the_own_multiset(tmp_path):
    left = _jsonl(tmp_path, "l.jsonl", _text_line("a"), UNWRITTEN, _text_line("b"), "\n")
    right = _jsonl(tmp_path, "r.jsonl", UNWRITTEN, _text_line("b"), _text_line("a"))
    assert repair_files.own_multiset_gap({"R1": left}, {"R1": right}) == (1, 0)


@pytest.mark.parametrize("extra", [
    UNWRITTEN,                     # not JSON
    "[1, 2]\n",                    # JSON, but not an object
    json.dumps({"id": "x"}) + "\n",   # an object without a text
    json.dumps({"id": "x", "text": 7}) + "\n",   # an object whose text is no string
])
def test_one_more_line_with_no_text_on_one_side_is_a_difference_of_one(tmp_path, extra):
    held = _jsonl(tmp_path, "h.jsonl", _text_line("a"), UNWRITTEN)
    more = _jsonl(tmp_path, "m.jsonl", _text_line("a"), UNWRITTEN, extra)
    assert repair_files.own_multiset_gap({"R1": held}, {"R1": more}) == (1, 1)
    assert repair_files.own_multiset_gap({"R1": more}, {"R1": held}) == (1, 1)


def test_a_line_with_no_text_is_not_read_as_an_empty_text(tmp_path):
    none = _jsonl(tmp_path, "n.jsonl", _text_line("a"), UNWRITTEN)
    empty = _jsonl(tmp_path, "e.jsonl", _text_line("a"), _text_line(""))
    assert repair_files.own_multiset_gap({"R1": none}, {"R1": empty}) == (1, 2)


def test_the_own_shard_digest_sees_one_changed_byte(tmp_path):
    for name, data in (("a.bin", b"1234"), ("a.idx", b"idx"), ("b.bin", b"1234"),
                       ("b.idx", b"idx"), ("c.bin", b"1235"), ("c.idx", b"idx")):
        (tmp_path / name).write_bytes(data)
    pair = {"a": tmp_path / "a", "b": tmp_path / "b"}
    assert repair_files.own_shard_gap(pair, {"a": tmp_path / "b", "b": tmp_path / "a"}) == (2, [])
    assert repair_files.own_shard_gap({"a": tmp_path / "a"}, {"a": tmp_path / "c"}) == (1, ["a"])


def test_the_shards_of_one_layout_share_a_name_prefix():
    assert repair_files.shard_prefix(["set-rr/shard-01", "set-rr/shard-02"]) == "set-rr/"
    with pytest.raises(repair_files.RepairError):
        repair_files.shard_prefix(["a/shard-1", "b/shard-1"])


# --- The verification and the report of the fixture, put to documents written by hand ----------

SHARD3_LINE = "D12"


def _fixture_report() -> dict:
    report = _report()
    shard = [
        {"artifact": "E/shard-1", "kind": "shard", "action": "keep"},
        {"artifact": "E/shard-2", "kind": "shard", "action": "keep"},
        {"artifact": "E/shard-3", "kind": "shard", "action": "rewrite", "records_removed": 1,
         "removed": [SHARD3_LINE], "documents": 27, "documents_in_successor": 26},
    ]
    report["artifacts"] += shard
    report["mixtures"] = {"needs_successor": ["Y/mix-1", "K/mix-1", "E/mix-2"],
                          "kept": ["D/mix-1", "E/mix-1"], "unresolved": []}
    report["groups"] = {"by_run": {}, "by_layout": {"partitioned": {
        "counts": {"keep": 2, "rewrite": 1}, "keep": ["E/shard-1", "E/shard-2"],
        "rewrite": ["E/shard-3"], "unresolved": []}}}
    report["restarts"] = [{"run": "E", "proven": True, "checkpoint": "E@6", "step": 6}]
    return report


def _verification() -> dict:
    return {
        "form": "waterwhite.verification/1", "mode": "unseeded", "unresolved_records": 0,
        "multisets": "equal", "differing_records": 0, "differences": [],
        "per_shard_digests": "not compared",
        "kept": [{"artifact": a, "identical": True} for a in ("R2", "E/shard-1", "E/shard-2")],
        "caches": [
            {"run": "E", "segment": "E/seg-1", "stale": False, "rebuilt": False,
             "rebuilt_from": None},
            {"run": "E", "segment": "E/seg-2", "stale": True, "rebuilt": True,
             "rebuilt_from": "an empty cache folder"}],
        "restarts": [{"run": "E", "segment": "E/seg-2", "step": 7, "step_comparison": "made"}],
    }


class FixtureWorld(StubWorld):
    def __init__(self, base, report, verification):
        super().__init__(base, report)
        self.verification = verification

    def locator(self, store, ref):
        if (ref.artifact, ref.handle) == ("E/shard-3", "T"):
            return SHARD3_LINE
        return super().locator(store, ref)

    def verification_of(self, name, mode=None):
        return self.verification, None


def _fixture_put(world, row_id):
    repair = next(r for r in c9.read_all() if r.id == row_id)
    return c9.put_fixture(repair, world)


@pytest.fixture
def fixture_world(tmp_path):
    return FixtureWorld(tmp_path, _fixture_report(), _verification())


@pytest.mark.parametrize("row_id", [
    "c09.fixture.R1.counts", "c09.fixture.R1.action", "c09.fixture.R2.action",
    "c09.fixture.E/shard-1.action", "c09.fixture.E/shard-2.action", "c09.fixture.E/shard-3.action",
    "c09.fixture.shards.counts", "c09.fixture.mixtures", "c09.E.exposure-free-restart",
    "c09.E.exact-rerun-restart", "c09.E.stale-cache.E/seg-2", "c09.E.stale-cache.E/seg-1"])
def test_a_fixture_row_agrees_with_a_report_and_a_verification_written_to_the_table(
        fixture_world, row_id):
    assert _fixture_put(fixture_world, row_id) == []


def _with(report=None, verification=None, tmp_path=None, **edits):
    report, verification = copy.deepcopy(report or _fixture_report()), copy.deepcopy(
        verification or _verification())
    return FixtureWorld(tmp_path, report, verification)


def test_the_fixture_rows_fail_where_the_report_differs(tmp_path):
    report = _fixture_report()
    report["artifacts"][0]["rows_in_successor"] = 22
    assert [m.field for m in _fixture_put(_with(report, tmp_path=tmp_path),
                                           "c09.fixture.R1.counts")] == ["rows_in_successor"]
    report = _fixture_report()
    report["artifacts"][5]["removed"] = ["D13"]
    assert "records_removed" in [m.field for m in _fixture_put(
        _with(report, tmp_path=tmp_path), "c09.fixture.E/shard-3.action")]
    report = _fixture_report()
    report["artifacts"][5]["documents_in_successor"] = 27
    assert [m.field for m in _fixture_put(_with(report, tmp_path=tmp_path),
                                           "c09.fixture.E/shard-3.action")] == [
        "documents_in_successor"]
    report = _fixture_report()
    report["mixtures"]["needs_successor"] = ["E/mix-2"]
    assert [m.field for m in _fixture_put(_with(report, tmp_path=tmp_path),
                                           "c09.fixture.mixtures")] == ["needs_successor"]
    report = _fixture_report()
    report["groups"]["by_layout"]["partitioned"]["keep"] = ["E/shard-1"]
    assert [m.field for m in _fixture_put(_with(report, tmp_path=tmp_path),
                                           "c09.fixture.shards.counts")] == ["keep"]
    report = _fixture_report()
    report["restarts"][0]["proven"] = False
    assert [m.field for m in _fixture_put(_with(report, tmp_path=tmp_path),
                                           "c09.E.exposure-free-restart")] == [
        "exposure_free_restart"]


def test_the_fixture_rows_fail_where_the_verification_differs(tmp_path):
    verification = _verification()
    verification["kept"][1]["identical"] = False
    assert [m.field for m in _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                                           "c09.fixture.E/shard-1.action")] == [
        "byte_identical_afterwards[E/shard-1]"]
    verification = _verification()
    verification["kept"][1]["identical"] = None      # a null is no pass
    assert _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                        "c09.fixture.E/shard-1.action")
    verification = _verification()
    verification["kept"] = []                         # no entry is no pass
    assert _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                        "c09.fixture.R2.action")
    verification = _verification()
    verification["caches"][0]["stale"] = True
    assert [m.field for m in _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                                           "c09.E.stale-cache.E/seg-1")] == ["stale"]
    verification = _verification()
    verification["caches"][1]["rebuilt_from"] = "the kept cache files"
    assert [m.field for m in _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                                           "c09.E.stale-cache.E/seg-2")] == ["rebuilt_from"]
    verification = _verification()
    verification["restarts"][0]["step"] = 6
    assert [m.field for m in _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                                           "c09.E.exact-rerun-restart")] == [
        "exact_rerun_restart_step", "not_before_step"]
    verification = _verification()
    verification["caches"] = []                       # no entry for the segment
    assert [m.field for m in _fixture_put(_with(verification=verification, tmp_path=tmp_path),
                                           "c09.E.stale-cache.E/seg-1")] == ["stale"]


def test_a_failed_verification_is_a_difference_and_never_a_pass(tmp_path):
    class Refusing(FixtureWorld):
        def verification_of(self, name, mode=None):
            return None, "no verification: exit code 2"

    world = Refusing(tmp_path, _fixture_report(), _verification())
    for row_id in ("c09.fixture.R2.action", "c09.E.stale-cache.E/seg-1",
                   "c09.E.exact-rerun-restart"):
        assert [m.field for m in _fixture_put(world, row_id)] == ["verification"], row_id


def test_the_exact_rerun_restart_under_f7_is_read_in_the_rows_words():
    verification = _verification()
    verification["restarts"][0].update(step=7, step_comparison="not possible")
    assert rep.restart_fields(verification, "E") == {
        "step_comparison": "not possible",
        "exact_rerun_restart": "the phase start, step 7, stated as an assumption"}
    assert rep.restart_fields(verification, "Z") == {}


def test_the_stale_and_restart_readers_give_nothing_for_what_the_document_does_not_hold():
    assert rep.cache_fields(_verification(), "E/seg-9") == {}
    assert rep.restart_checkpoint(_fixture_report(), "Y") is None
    assert rep.kept_identical(_verification(), "E/shard-3") is None
    with pytest.raises(TableError, match="form"):
        rep.document({"form": "waterwhite.report/1"}, rep.VERIFICATION_FORM, "verification")
    with pytest.raises(TableError):
        rep.document([], rep.REPORT_FORM, "report")


# --- The rows of the 10k set: the two multisets, the digests and the counts --------------------


def test_the_multiset_rows_are_held_to_the_table_by_both_the_own_multiset_and_the_tools_word():
    equal = {"multisets": "equal", "unresolved_records": 0}
    verification = {"multisets": "equal", "unresolved_records": 0, "differing_records": 0}
    assert c9.check_sides(equal, verification, 4, 0) == []
    # The tool says equal and the files differ: the own multiset catches it.
    assert [m.field for m in c9.check_sides(equal, verification, 4, 1)] == ["multisets"]
    # The files agree and the tool says differ: the tool's word catches it.
    assert [m.field for m in c9.check_sides(equal, {**verification, "multisets": "differ"}, 4,
                                            0)] == ["multisets (the tool's word)"]
    assert [m.field for m in c9.check_sides(equal, verification, 0, 0)] == ["files compared"]
    assert [m.field for m in c9.check_sides(equal, {**verification, "unresolved_records": 2}, 4,
                                            0)] == ["unresolved_records"]
    wrong = {"multisets": "differ", "differing_records": 1}
    assert c9.check_sides(wrong, {"multisets": "differ", "differing_records": 1}, 4, 1) == []
    assert [m.field for m in c9.check_sides(wrong, {"multisets": "differ", "differing_records": 2},
                                            4, 1)] == ["differing_records (the tool's number)"]
    assert [m.field for m in c9.check_sides(wrong, {"multisets": "differ", "differing_records": 1},
                                            4, 2)] == ["differing_records"]


def test_the_digest_row_is_held_to_the_table_by_the_own_digests_and_the_tools_word():
    expect = {"per_shard_digests": "equal"}
    assert c9.check_digests(expect, {"per_shard_digests": "equal"}, 3, []) == []
    assert [m.field for m in c9.check_digests(expect, {"per_shard_digests": "equal"}, 3,
                                              ["a"])] == ["per_shard_digests (own digests)"]
    assert [m.field for m in c9.check_digests(expect, {"per_shard_digests": "not compared"}, 3,
                                              [])] == ["per_shard_digests (the tool's word)"]
    assert [m.field for m in c9.check_digests(expect, {"per_shard_digests": "equal"}, 0,
                                              [])] == ["shards compared"]


COUNTED = {"actions": {"records": {"keep": 5, "re-source": 2, "promote": 2, "retire": 3}},
           "releases": {"R1": {"removed": 3, "actions": {"keep": 5, "re-source": 2, "promote": 2,
                                                         "retire": 3}}},
           "shards": {"keep": ["shard-A"], "rewrite": ["shard-B", "shard-C"]}}


def _counts_report():
    return {"counts": {"records": {"keep": 5, "re-source": 2, "promote": 2, "retire": 3,
                                   "unresolved": 0}, "records_removed": 3,
                       "shards": {"keep": 1, "rewrite": 2}},
            "records": [{"artifact": "R1", "record": f"R1:L{n}", "action": a}
                        for n, a in enumerate(["keep", "re-source", "promote", "retire"], 1)]}


def test_the_action_counts_are_the_scripts_and_a_word_outside_the_four_is_found():
    expect = {"record_actions_allowed": ["keep", "re-source", "promote", "retire"]}
    assert c9.check_set_counts(expect, _counts_report(), COUNTED) == []
    report = _counts_report()
    report["counts"]["records"]["retire"] = 2
    report["counts"]["shards"]["keep"] = 2
    assert [m.field for m in c9.check_set_counts(expect, report, COUNTED)] == [
        "counts.retire", "counts.shards.keep"]
    report = _counts_report()
    report["records"][0]["action"] = "remove"
    assert [m.field for m in c9.check_set_counts(expect, report, COUNTED)] == [
        "record_actions_allowed"]
    report = _counts_report()
    del report["counts"]["records_removed"]
    assert [(m.field, m.got) for m in c9.check_set_counts(expect, report, COUNTED)] == [
        ("counts.records_removed", None)]


def test_a_row_of_a_layout_names_the_layout_in_each_difference():
    found = c9._tagged("round-robin", [Mismatch("rewrite", 20, 19)])
    assert [(m.field, m.expected, m.got) for m in found] == [("rewrite [round-robin]", 20, 19)]


def test_the_shard_actions_are_the_artifact_entries_of_kind_shard():
    assert c9.shard_actions(_fixture_report()) == {
        "E/shard-1": "keep", "E/shard-2": "keep", "E/shard-3": "rewrite"}


# --- The command lines of a repair, with the commands replaced by recorders --------------------

import sqlite3  # noqa: E402
from contextlib import closing  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from acceptance_world import REPAIRED, World, WorldError  # noqa: E402

from replica.fixture import CapturedFixture  # noqa: E402

SIDES = {"form": "waterwhite.sides/1", "repaired": {}, "reference": {}}


def _options(argv):
    """The options of a replica command as a dict: every one takes a value."""
    assert argv[0] == "repair" and len(argv) % 2 == 1
    return dict(zip(argv[1::2], argv[2::2], strict=True))


def _recorders(monkeypatch, report):
    calls = []

    def tool(store, arguments, timeout=0):
        calls.append(("tool", Path(store), [str(a) for a in arguments]))
        return 0, json.dumps(report), ""

    def replica(arguments, timeout=0):
        argv = [str(a) for a in arguments]
        calls.append(("replica", argv))
        Path(_options(argv)["--sides"]).write_text(json.dumps(SIDES), encoding="utf-8")
        return {"successors": {}}

    monkeypatch.setattr(repair_files, "tool_command", tool)
    monkeypatch.setattr(repair_files, "replica_command", replica)
    return calls


class RecordedWorld(World):
    def __init__(self, base, built):
        super().__init__(base)
        self.built, self.commands = built, []

    def store(self, name):
        return self.built

    def cli(self, store_file, *arguments):
        self.commands.append((Path(store_file), list(arguments)))
        return Outcome(0, json.dumps(_verification()), "")


def test_the_repair_runs_the_documented_command_lines_on_a_copy_of_the_stores_folder(
        tmp_path, monkeypatch):
    folder = tmp_path / "stores" / "main"
    (folder / "out").mkdir(parents=True)
    (folder / "store.sqlite").write_bytes(b"store")
    spec = tmp_path / "fixture_spec.toml"
    spec.write_text("", encoding="utf-8")
    built = CapturedFixture(store=folder / "store.sqlite", out=folder / "out",
                            instances={"R1": "inst-R1"}, summary={}, spec=spec)
    calls = _recorders(monkeypatch, _fixture_report())
    world = RecordedWorld(tmp_path / "world", built)
    fixture = world.copy("main", REPAIRED)
    # 1. the report, asked of the historical store, with the layout named for the groups
    kind, store, argv = calls[0]
    assert (kind, store) == ("tool", built.store)
    assert argv == ["report", "withdraw", "A", "--layout", "partitioned=E/"]
    # 2. the replica's repair, as docs/interfaces.md 9.5 gives it, on a COPY of the folder
    kind, argv = calls[1]
    options = _options(argv)
    assert set(options) == {"--report", "--historical", "--out", "--store", "--spec", "--layout",
                            "--sides"}
    assert options["--historical"] == str(built.out) and options["--spec"] == str(spec)
    assert options["--layout"] == "partitioned"
    copied = Path(options["--store"])
    assert copied != built.store and copied.parent != built.store.parent
    assert copied.read_bytes() == b"store" and built.store.read_bytes() == b"store"
    assert Path(options["--out"]).parent == copied.parent   # inside the copy's folder
    assert Path(options["--report"]).read_text(encoding="utf-8") == json.dumps(_fixture_report())
    # the store the question is put to is the copy, in the shape of every other store
    assert isinstance(fixture, CapturedFixture) and fixture.store == copied
    assert fixture.instances == built.instances
    assert world.copy("main", REPAIRED) is not None and len(calls) == 2   # run once
    # 3. the verification, handed the report as a file, the sides file, and the mode
    verification, failure = world.verification_of("main", "seeded")
    assert failure is None and verification["form"] == "waterwhite.verification/1"
    world.verification_of("main")
    assert world.commands[0] == (copied, ["verify", "--report", options["--report"], "--sides",
                                          options["--sides"], "--mode", "seeded"])
    assert world.commands[1][1] == ["verify", "--report", options["--report"], "--sides",
                                    options["--sides"]]


def test_a_repair_that_cannot_run_is_an_error_of_the_run_and_never_a_skip_or_a_pass(
        tmp_path, monkeypatch):
    folder = tmp_path / "stores" / "main"
    (folder / "out").mkdir(parents=True)
    (folder / "store.sqlite").write_bytes(b"store")
    built = CapturedFixture(store=folder / "store.sqlite", out=folder / "out", instances={},
                            summary={}, spec=tmp_path / "spec.toml")

    def broken(arguments, timeout=0):
        raise repair_files.RepairError("replica repair ended with exit code 1")

    _recorders(monkeypatch, _fixture_report())
    monkeypatch.setattr(repair_files, "replica_command", broken)
    with pytest.raises(WorldError, match="exit code 1"):
        RecordedWorld(tmp_path / "world", built).copy("main", REPAIRED)


def _set_repairs(tmp_path, monkeypatch):
    folder = tmp_path / "10k" / "repair-partitioned"
    (folder / "build" / "releases").mkdir(parents=True)
    (folder / "build" / "releases" / "R1.jsonl.manifest.jsonl").write_text(
        "".join(json.dumps({"line": n, "handle": h}) + "\n"
                for n, h in enumerate(["A-000001", "T-000001", "UP-000001"], 1)),
        encoding="utf-8")
    with closing(sqlite3.connect(folder / "store.sqlite")) as db:
        db.execute("create table t (x)")
        db.commit()
    report = _report()
    report["records"] = [_entry_of("R1:L2", "retire"), _entry_of("R1:L3", "keep")]
    hist = repair_files.Historical("partitioned", folder / "store.sqlite", folder / "build",
                                   "set/", ("set/shard-A",), {}, report)
    sets = SimpleNamespace(base=tmp_path, set_folder=lambda size: tmp_path / "set")
    repairs = repair_files.SetRepairs(sets)
    repairs._historical[("10k", "partitioned")] = hist
    repairs._clean["10k"] = tmp_path / "clean"
    return repairs, hist, _recorders(monkeypatch, _verification())


def _entry_of(record, action):
    return {"artifact": "R1", "record": record, "action": action}


def test_the_set_repair_runs_beside_the_historical_store_with_the_edited_report(
        tmp_path, monkeypatch):
    repairs, hist, calls = _set_repairs(tmp_path, monkeypatch)
    built = repairs.repaired("10k", "partitioned", "r.descendant-kept")
    options = _options(calls[0][1])
    assert set(options) == {"--report", "--historical", "--out", "--store", "--set",
                            "--layout", "--reference-set", "--reference-out", "--sides", "--jobs",
                            "--tokenizer-spec"}
    copied = Path(options["--store"])
    assert copied != hist.store and copied.parent == hist.store.parent   # beside, not the same
    assert options["--historical"] == str(hist.out) and options["--layout"] == "partitioned"
    assert options["--set"] == str(tmp_path / "set")
    assert options["--reference-set"] == str(tmp_path / "clean")
    assert Path(options["--out"]).parent == copied.parent
    sent = json.loads(Path(options["--report"]).read_text(encoding="utf-8"))
    assert [e["action"] for e in sent["records"]] == ["keep", "keep"]     # T-000001 was line 2
    assert [e["action"] for e in hist.report["records"]] == ["retire", "keep"]   # not changed
    assert built.repair.store == copied
    verification, failure = repairs.verification("10k", "partitioned", "r.descendant-kept",
                                                 "seeded")
    assert failure is None and verification["form"] == "waterwhite.verification/1"
    assert calls[1] == ("tool", copied, ["verify", "--report", options["--report"], "--sides",
                                         options["--sides"], "--mode", "seeded"])
    repairs.answer("10k", "partitioned", "successor of set/shard-A", "r.descendant-kept")
    assert calls[2] == ("tool", copied, ["query", "withdraw", "A", "--artifact",
                                         "successor of set/shard-A"])
    assert repairs.repaired("10k", "partitioned", "r.descendant-kept") is built   # built once


def test_the_unrelated_paraphrase_state_retires_the_kept_entry_and_nothing_else(
        tmp_path, monkeypatch):
    repairs, hist, calls = _set_repairs(tmp_path, monkeypatch)
    repairs.repaired("10k", "partitioned", "r.unrelated-paraphrase-retired")
    sent = json.loads(Path(_options(calls[0][1])["--report"]).read_text(encoding="utf-8"))
    assert [e["action"] for e in sent["records"]] == ["retire", "retire"]   # UP-000001 was line 3
    with pytest.raises(table.TableError):
        repairs.repaired("10k", "partitioned", "f.trace-deleted")


def test_the_report_alone_needs_no_repair_for_the_rows_that_read_only_it(tmp_path, monkeypatch):
    repairs, hist, calls = _set_repairs(tmp_path, monkeypatch)
    hist_report = {"counts": {"records": {}, "shards": {"keep": 0, "rewrite": 20}},
                   "artifacts": [{"artifact": f"set/shard-{n}", "kind": "shard",
                                  "action": "rewrite"} for n in range(20)]}
    repairs._historical[("10k", "round-robin")] = repair_files.Historical(
        "round-robin", hist.store, hist.out, "set/", (), {}, hist_report)
    repair = next(r for r in c9.read_all() if r.id == "c09.10k.round-robin.all-rewrite")
    assert c9.put_set(repair, repairs, []) == []
    assert calls == []    # no replica, no verify: the historical build and its report only
    hist_report["artifacts"][3]["action"] = "keep"
    assert [m.field for m in c9.put_set(repair, repairs, [])] == ["shards.rewrite [round-robin]",
                                                                  "shards.keep [round-robin]"]


# --- The shards of each layout, and the successor of each shard that is left --------------------

import dataclasses  # noqa: E402

SOURCES = ("A", "B", "C", "Z")


def _write_set(base: Path, sources=SOURCES) -> Path:
    """A set folder that holds what the helpers read of a set: its sources table."""
    folder = base / "set"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "set.toml").write_text(
        "".join(f'[sources.{name}]\nfile = "sources/{name}.jsonl"\n\n' for name in sources),
        encoding="utf-8")
    return folder


def _table_entry(shard, lines, in_scope, *, receives=0, generated_kept=0):
    return {"shard": shard, "lines": lines, "in_scope_lines": in_scope, "own": in_scope,
            "derived": 0, "receives": receives, "generated_kept": generated_kept,
            "action": "rewrite" if in_scope or receives else "keep"}


def _counted_of(entries):
    return {"shards": {"shards": entries,
                       "keep": [e["shard"] for e in entries if e["action"] == "keep"],
                       "rewrite": [e["shard"] for e in entries if e["action"] == "rewrite"]}}


def _store_file(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(folder / "store.sqlite")) as db:
        db.execute("create table t (x)")
        db.commit()
    return folder / "store.sqlite"


def _stub_historical(base, monkeypatch, layout, instances, *, sources=SOURCES):
    """``SetRepairs.historical`` with the two commands recorded: what the build registered is
    ``instances``, and the count is not asked of a real chunk folder."""
    _store_file(base / "10k" / f"repair-{layout}")
    calls = []
    monkeypatch.setattr(acceptance_sets, "need_megatron", lambda: None)
    monkeypatch.setattr(acceptance_sets, "load_script", lambda file: SimpleNamespace(
        count=lambda *args, **kwargs: {}))
    monkeypatch.setattr(repair_files, "replica_command", lambda arguments, timeout=0: {
        "instances": dict(instances)})

    def tool(store, arguments, timeout=0):
        calls.append([str(a) for a in arguments])
        return 0, json.dumps({"artifacts": []}), ""

    monkeypatch.setattr(repair_files, "tool_command", tool)
    sets = SimpleNamespace(base=base, set_folder=lambda size: _write_set(base, sources))
    return repair_files.SetRepairs(sets), calls


def _shards_held(repairs, layout):
    """The shard list the helper holds for a layout, or the refusal in its place, so that a
    list that is wrong shows as a difference."""
    try:
        return repairs.historical("10k", layout).shards
    except repair_files.RepairError as refusal:
        return f"refused: {refusal}"


PARTITIONED_INSTANCES = {
    "set/shard-A": "i1", "set/shard-B": "i2", "set/shard-C": "i3", "set/shard-Z": "i4",
    "set/shard-gen": "i5",
    # companions of a shard and every other instance of the build: none is a shard
    "set/shard-A.input": "c1", "set/shard-A.sanitizer-input": "c2", "set/shard-gen.input": "c3",
    "set/shard-gen.sanitizer-input": "c4", "set/shard-Z.input": "c5",
    "set/shard-Q": "o1", "set/shard-Z2": "o2", "set/shard-a": "o3", "set/shard-1": "o4",
    "set/shard-genx": "o5", "set/mix-1": "o6", "src/A": "o7", "gen/paraphrase": "o8", "R1": "o9",
    "set-rr/shard-01": "o10", "other/shard-A": "o11",
}


def test_a_partitioned_layout_lists_one_shard_per_source_and_shard_gen(tmp_path, monkeypatch):
    repairs, calls = _stub_historical(tmp_path, monkeypatch, "partitioned", PARTITIONED_INSTANCES)
    assert _shards_held(repairs, "partitioned") == (
        "set/shard-A", "set/shard-B", "set/shard-C", "set/shard-Z", "set/shard-gen")
    # one shared folder name, and the placement file the helper wrote beside the build
    assert calls and calls[0][-4:-1] == ["--layout", "partitioned=set/", "--placement"]
    placement = repair_files.placement_path(tmp_path / "10k" / "repair-partitioned")
    assert calls[0][-1] == f"partitioned={placement}"


def test_a_partitioned_shard_that_is_not_written_is_absent_and_the_sets_sources_decide(
        tmp_path, monkeypatch):
    absent = {k: v for k, v in PARTITIONED_INSTANCES.items() if k != "set/shard-B"}
    repairs, _ = _stub_historical(tmp_path, monkeypatch, "partitioned", absent)
    assert _shards_held(repairs, "partitioned") == (
        "set/shard-A", "set/shard-C", "set/shard-Z", "set/shard-gen")
    # a set of ten sources has a shard for Z2 as well; a set of four has none
    other = tmp_path / "ten"
    other.mkdir()
    repairs, _ = _stub_historical(other, monkeypatch, "partitioned", PARTITIONED_INSTANCES,
                                  sources=(*SOURCES, "Z2"))
    assert _shards_held(repairs, "partitioned") == (
        "set/shard-A", "set/shard-B", "set/shard-C", "set/shard-Z", "set/shard-Z2",
        "set/shard-gen")


ROUND_ROBIN_INSTANCES = {
    "set-rr/shard-01": "i1", "set-rr/shard-02": "i2", "set-rr/shard-03": "i3",
    "set-rr/shard-01.input": "c1", "set-rr/shard-01.sanitizer-input": "c2",
    "set-rr/mix-1": "o1", "R1": "o2", "set/shard-A": "o3", "set/shard-gen": "o4",
}


def test_a_round_robin_layout_lists_its_numbered_chunks_as_it_did(tmp_path, monkeypatch):
    repairs, calls = _stub_historical(tmp_path, monkeypatch, "round-robin", ROUND_ROBIN_INSTANCES)
    assert _shards_held(repairs, "round-robin") == (
        "set-rr/shard-01", "set-rr/shard-02", "set-rr/shard-03")
    assert calls[0][-2:] == ["--layout", "round-robin=set-rr/"]
    build = acceptance_sets.SetBuild("small", "capture", tmp_path, tmp_path / "s.sqlite", tmp_path,
                                     dict(ROUND_ROBIN_INSTANCES), {})
    assert build.shards == {"set-rr/shard-01": "shard-01", "set-rr/shard-02": "shard-02",
                            "set-rr/shard-03": "shard-03"}


class _LayoutWorld:
    """One layout of the 10k set as the recorded commands leave it: the historical build with
    its count (``entries``: the count's table, one entry per shard), a report that rewrites
    every shard the count rewrites (the successor row is held to the count, never to this
    report), and the tool's answer to each question put to the repaired store (``answers``:
    artifact -> (answer, reason))."""

    def __init__(self, base, layout, entries, answers, *, prefix, files=(), report_actions=None,
                 default=None):
        folder = base / "10k" / f"repair-{layout}"
        (folder / "build").mkdir(parents=True)
        # the shard actions the report states: the count's, unless a test says otherwise
        actions = report_actions if report_actions is not None else [
            (e["shard"], e["action"]) for e in entries]
        report = {"artifacts": [{"artifact": f"{prefix}{stem}", "kind": "shard", "action": action}
                                for stem, action in actions]}
        self.hist = repair_files.Historical(
            layout, _store_file(folder), folder / "build", prefix,
            tuple(f"{prefix}{e['shard']}" for e in entries), _counted_of(entries), report)
        shards = folder / "repaired-unmutated" / "pipeline" / layout / "shards"
        shards.mkdir(parents=True)
        for name in files:
            (shards / name).write_bytes(b"x")
        self.answers, self.asked, self.default = answers, [], default


def _put_successors(tmp_path, monkeypatch, worlds):
    """Row c09.10k.successor-shards put to the recorded layouts: (the fields of the
    differences, the questions put to the repaired stores)."""
    by_folder = {f"repair-{layout}": world for layout, world in worlds.items()}

    def tool(store, arguments, timeout=0):
        world = by_folder[Path(store).parent.name]
        if arguments[0] == "query":
            artifact = arguments[arguments.index("--artifact") + 1]
            world.asked.append(artifact)
            answer, reason = (world.answers[artifact] if world.default is None
                              else world.answers.get(artifact, world.default))
            return 0, json.dumps({"answer": answer, "reason": reason}), ""
        return 0, json.dumps(world.hist.report), ""

    def replica(arguments, timeout=0):
        Path(_options([str(a) for a in arguments])["--sides"]).write_text(
            json.dumps(SIDES), encoding="utf-8")
        return {"successors": {}}

    monkeypatch.setattr(repair_files, "tool_command", tool)
    monkeypatch.setattr(repair_files, "replica_command", replica)
    repairs = repair_files.SetRepairs(SimpleNamespace(
        base=tmp_path, set_folder=lambda size: tmp_path / "set"))
    repairs._clean["10k"] = tmp_path / "clean"
    for layout, world in worlds.items():
        repairs._historical[("10k", layout)] = world.hist
    row = next(r for r in c9.read_all() if r.id == "c09.10k.successor-shards")
    found = c9.put_set(dataclasses.replace(row, layouts=tuple(worlds)), repairs, [])
    return [m.field for m in found], [q for world in worlds.values() for q in world.asked]


# The partitioned layout of the 10k set as its count shows it: every row of A's file is in
# scope (A is the withdrawn source), so its shard keeps no row and the repair writes none; the
# shard of generated records keeps the records that stay; B, C and Z are untouched.
PARTITIONED_ENTRIES = [
    _table_entry("shard-A", 40, 40, receives=6),
    _table_entry("shard-B", 30, 0),
    _table_entry("shard-C", 20, 0),
    _table_entry("shard-Z", 50, 0),
    _table_entry("shard-gen", 12, 9, generated_kept=3),
]
NOT_REGISTERED = ("not covered", "not registered")
UNDER = ("does not contain", None)


def _partitioned(tmp_path, answers, files=()):
    return _LayoutWorld(tmp_path, "partitioned", PARTITIONED_ENTRIES, answers, prefix="set/",
                        files=files)


def test_a_shard_with_no_row_left_by_the_count_needs_no_successor_and_one_with_rows_needs_one(
        tmp_path, monkeypatch):
    world = _partitioned(tmp_path, {
        "successor of set/shard-A": NOT_REGISTERED, "successor of set/shard-gen": UNDER})
    fields, asked = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == []
    assert sorted(asked) == ["successor of set/shard-A", "successor of set/shard-gen"]


@pytest.mark.parametrize(("answers", "files", "wrong"), [
    # rows are left in the shard of generated records, and no successor is registered for it
    ({"successor of set/shard-A": NOT_REGISTERED, "successor of set/shard-gen": NOT_REGISTERED},
     (), ["answer of set/shard-gen"]),
    # no row is left in A's shard, and a successor is registered for it
    ({"successor of set/shard-A": UNDER, "successor of set/shard-gen": UNDER},
     (), ["answer of set/shard-A", "reason of set/shard-A"]),
    # the name is unregistered and a shard file of that name was written all the same
    ({"successor of set/shard-A": NOT_REGISTERED, "successor of set/shard-gen": UNDER},
     ("shard-A.bin", "shard-A.idx"), ["shard file of set/shard-A"]),
    # another way not to be registered is not the same answer
    ({"successor of set/shard-A": ("not covered", "record missing"),
      "successor of set/shard-gen": UNDER}, (), ["reason of set/shard-A"]),
])
def test_a_successor_that_is_missing_or_present_against_the_count_fails_the_row(
        tmp_path, monkeypatch, answers, files, wrong):
    world = _partitioned(tmp_path, answers, files)
    fields, _ = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == [f"{field} [partitioned]" for field in wrong]


def test_the_rule_holds_in_both_layouts_and_a_file_of_another_name_is_not_a_shard_file(
        tmp_path, monkeypatch):
    chunks = [_table_entry(f"shard-{n:02d}", 30, 4) for n in range(1, 4)]
    rr = _LayoutWorld(tmp_path, "round-robin", chunks, {
        f"successor of set-rr/shard-{n:02d}": UNDER for n in range(1, 4)}, prefix="set-rr/",
        files=("shard-A.bin",))
    pt = _partitioned(tmp_path, {"successor of set/shard-A": NOT_REGISTERED,
                                 "successor of set/shard-gen": UNDER}, files=("shard-B.bin",))
    fields, asked = _put_successors(tmp_path, monkeypatch, {"round-robin": rr, "partitioned": pt})
    assert fields == []
    assert sorted(asked) == ["successor of set-rr/shard-01", "successor of set-rr/shard-02",
                             "successor of set-rr/shard-03", "successor of set/shard-A",
                             "successor of set/shard-gen"]


def test_the_row_still_needs_one_shard_that_is_not_kept(tmp_path, monkeypatch):
    entries = [_table_entry(stem, 10, 0) for stem in ("shard-A", "shard-B")]
    world = _LayoutWorld(tmp_path, "partitioned", entries, {}, prefix="set/")
    fields, asked = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == ["successor shards [partitioned]"] and asked == []


# --- The repair and the historical build are run as the replica needs them -----------------------


def _after(argv, flag):
    """The word that follows ``flag`` in a command line, or None."""
    argv = [str(a) for a in argv]
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def test_the_historical_build_of_a_set_runs_its_generation_jobs_confined(tmp_path, monkeypatch):
    repairs, _ = _stub_historical(tmp_path, monkeypatch, "partitioned", PARTITIONED_INSTANCES)
    built = []
    monkeypatch.setattr(repair_files, "replica_command", lambda arguments, timeout=0: (
        built.append([str(a) for a in arguments]) or {"instances": dict(PARTITIONED_INSTANCES)}))
    repairs.historical("10k", "partitioned")
    assert built[0][0] == "build-set"
    assert (_after(built[0], "--generation-harness"), _after(built[0], "--tokenizer-spec")) == (
        "confined", str(acceptance_sets.SPEC))


def test_the_repair_of_a_set_is_given_the_tokenizer_spec_of_the_historical_build(
        tmp_path, monkeypatch):
    repairs, hist, calls = _set_repairs(tmp_path, monkeypatch)
    repairs.repaired("10k", "partitioned")
    assert _options(calls[0][1]).get("--tokenizer-spec") == str(acceptance_sets.SPEC)


def _kept_world(tmp_path, listed):
    """The partitioned layout with the shards ``listed`` in the helper's list; the count also
    names shard-Z, kept. Only shard-B has files, the same before and after the repair."""
    entries = [_table_entry("shard-A", 40, 40), _table_entry("shard-B", 30, 0),
               _table_entry("shard-Z", 50, 0)]
    out = tmp_path / "build"
    before, after = out / "pipeline" / "partitioned" / "shards", tmp_path / "repaired"
    before.mkdir(parents=True)
    after.mkdir()
    for folder in (before, after):
        for suffix in (".bin", ".idx"):
            (folder / f"shard-B{suffix}").write_bytes(b"same")
    report = {"artifacts": [{"artifact": f"set/{e['shard']}", "kind": "shard",
                             "action": e["action"]} for e in entries]}
    hist = repair_files.Historical("partitioned", tmp_path / "s.sqlite", out, "set/",
                                   tuple(f"set/{stem}" for stem in listed), _counted_of(entries),
                                   report)
    verification = {"kept": [{"artifact": f"set/shard-{x}", "identical": True} for x in "BZ"]}
    built = SimpleNamespace(historical=hist, repaired_files={"set/shard-B": after / "shard-B"})
    repairs = SimpleNamespace(verification=lambda size, layout, state, mode: (verification, None))
    return built, repairs, SimpleNamespace(size="10k", state="unmutated")


def _kept_fields(built, repairs, repair):
    try:
        return [m.field for m in c9.check_kept_shards(built, repairs, repair, rep)]
    except Exception as failure:   # a failure of the helper shows as a difference, not an error
        return f"raised {type(failure).__name__}: {failure}"


def test_a_shard_of_the_count_that_the_helpers_list_lacks_is_a_difference_and_not_an_error(
        tmp_path):
    built, repairs, repair = _kept_world(tmp_path, ["shard-A", "shard-B"])
    assert _kept_fields(built, repairs, repair) == [
        "listed shard[set/shard-Z]", "byte identical afterwards[set/shard-Z]"]


def test_a_listed_kept_shard_with_no_files_is_a_difference_and_not_an_error(tmp_path):
    built, repairs, repair = _kept_world(tmp_path, ["shard-A", "shard-B", "shard-Z"])
    assert _kept_fields(built, repairs, repair) == ["byte identical afterwards[set/shard-Z]"]


def test_the_shards_asked_for_a_successor_are_the_counts_whatever_the_report_says(
        tmp_path, monkeypatch):
    # the report omits A, which the count rewrites and leaves no row in, calls the shard of
    # generated records kept, and rewrites B, which the count keeps
    world = _LayoutWorld(
        tmp_path, "partitioned", PARTITIONED_ENTRIES, {
            "successor of set/shard-A": NOT_REGISTERED, "successor of set/shard-gen": UNDER},
        prefix="set/", default=UNDER, report_actions=[
            ("shard-B", "rewrite"), ("shard-C", "keep"), ("shard-Z", "keep"),
            ("shard-gen", "keep")])
    fields, asked = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == []
    assert sorted(asked) == ["successor of set/shard-A", "successor of set/shard-gen"]


# --- A shard the repair writes that the historical build never wrote ------------------------------

# The historical build wrote the shards of A, C, Z and the generated records. No row of B
# survived its dedup there, so it has no shard-B, and the count has no line for one.
NO_B_ENTRIES = [e for e in PARTITIONED_ENTRIES if e["shard"] != "shard-B"]
WRITTEN_BY = ("shard-B.bin", "shard-B.idx")


def _new_shard_world(tmp_path, answers, *, reference=(), repaired=()):
    """The partitioned layout with no shard-B in the historical build; ``reference`` and
    ``repaired`` are the files the two sides' builds hold in their shards folders."""
    _write_set(tmp_path)
    world = _LayoutWorld(tmp_path, "partitioned", NO_B_ENTRIES, answers, prefix="set/",
                         files=repaired)
    folder = world.hist.store.parent / "reference-unmutated" / "pipeline" / "partitioned" / "shards"
    folder.mkdir(parents=True)
    for name in reference:
        (folder / name).write_bytes(b"x")
    return world


BASE_ANSWERS = {"successor of set/shard-A": NOT_REGISTERED, "successor of set/shard-gen": UNDER}


def test_a_new_shard_the_reference_writes_is_asked_for_its_successor_and_it_must_answer_as_the_row_says(
        tmp_path, monkeypatch):
    world = _new_shard_world(tmp_path, {**BASE_ANSWERS, "successor of set/shard-B": UNDER},
                             reference=(*WRITTEN_BY, "shard-Q.bin"), repaired=(*WRITTEN_BY,
                                                                             "shard-Q.bin"))
    fields, asked = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == []
    assert sorted(asked) == ["successor of set/shard-A", "successor of set/shard-B",
                             "successor of set/shard-gen"]   # shard-Q is no shard of this set


def test_a_new_shard_the_reference_writes_whose_successor_answers_wrongly_fails_the_row(
        tmp_path, monkeypatch):
    world = _new_shard_world(tmp_path, {**BASE_ANSWERS, "successor of set/shard-B": NOT_REGISTERED},
                             reference=WRITTEN_BY, repaired=WRITTEN_BY)
    fields, _ = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == ["answer of set/shard-B [partitioned]"]


def test_a_shard_only_the_repaired_side_writes_fails_the_row_and_is_asked_nothing(
        tmp_path, monkeypatch):
    world = _new_shard_world(tmp_path, BASE_ANSWERS, repaired=WRITTEN_BY)
    fields, asked = _put_successors(tmp_path, monkeypatch, {"partitioned": world})
    assert fields == ["shard file of set/shard-B [partitioned]"]
    assert sorted(asked) == ["successor of set/shard-A", "successor of set/shard-gen"]


# --- The digest row also covers a shard the report does not name ---------------------------------

LAYOUT_SHARDS = ("pipeline", "partitioned", "shards")


def _pair(folder, shard, data=b"same", suffixes=(".bin", ".idx")):
    folder.mkdir(parents=True, exist_ok=True)
    for suffix in suffixes:
        (folder / f"{shard}{suffix}").write_bytes(data)


def _put_digest_row(tmp_path, monkeypatch, *, named=("gen",), repaired=None, reference=None):
    """Row c09.10k.shard-digests put to the partitioned layout with recorded commands. The
    sides file names the shards ``named`` (stems, equal bytes on both sides, kept in the two
    shards folders as well); ``repaired`` and ``reference`` are further files of the two
    shards folders: stem -> (bytes, suffixes). The answer of the tool is "equal"."""
    _write_set(tmp_path)
    world = _LayoutWorld(tmp_path, "partitioned", NO_B_ENTRIES, {}, prefix="set/")
    folder = world.hist.store.parent
    side = {"repaired": folder / "repaired-unmutated", "reference": folder / "reference-unmutated"}
    shards = {key: path.joinpath(*LAYOUT_SHARDS) for key, path in side.items()}
    for stem in named:
        for key in side:
            _pair(shards[key], f"shard-{stem}")
    for key, held in (("repaired", repaired or {}), ("reference", reference or {})):
        for stem, (data, suffixes) in held.items():
            _pair(shards[key], stem, data, suffixes)
    shards["reference"].mkdir(parents=True, exist_ok=True)

    def tool(store, arguments, timeout=0):
        return 0, json.dumps({"per_shard_digests": "equal"}), ""

    def replica(arguments, timeout=0):
        document = {"form": "waterwhite.sides/1",
                    **{key: {f"set/shard-{stem}": str(shards[key] / f"shard-{stem}")
                             for stem in named} for key in side}}
        Path(_options([str(a) for a in arguments])["--sides"]).write_text(
            json.dumps(document), encoding="utf-8")
        return {"successors": {}}

    monkeypatch.setattr(repair_files, "tool_command", tool)
    monkeypatch.setattr(repair_files, "replica_command", replica)
    repairs = repair_files.SetRepairs(SimpleNamespace(
        base=tmp_path, set_folder=lambda size: tmp_path / "set"))
    repairs._clean["10k"] = tmp_path / "clean"
    repairs._historical[("10k", "partitioned")] = world.hist
    row = next(r for r in c9.read_all() if r.id == "c09.10k.shard-digests")
    return [m.field for m in c9.put_set(
        dataclasses.replace(row, layouts=("partitioned",)), repairs, [])]


OWN_DIGESTS = "per_shard_digests (own digests) [partitioned]"


def test_a_shard_both_builds_wrote_that_the_sides_file_does_not_name_is_compared_by_bytes(
        tmp_path, monkeypatch):
    fields = _put_digest_row(tmp_path, monkeypatch,
                             repaired={"shard-B": (b"one", (".bin", ".idx"))},
                             reference={"shard-B": (b"two", (".bin", ".idx"))})
    assert fields == [OWN_DIGESTS]


@pytest.mark.parametrize("named", [(), ("gen",)])
def test_the_same_shard_with_equal_bytes_on_both_sides_is_counted_and_agrees(
        tmp_path, monkeypatch, named):
    # with no shard in the sides file, "shards compared" is the one difference left unless the
    # shard of the folders is counted; a stray file of another name is no shard
    same = {"shard-B": (b"same", (".bin", ".idx"))}
    fields = _put_digest_row(tmp_path, monkeypatch, named=named,
                             repaired={**same, "shard-Q": (b"one", (".bin",))},
                             reference={**same, "shard-Q": (b"two", (".bin",))})
    assert fields == []


@pytest.mark.parametrize("side", ["repaired", "reference"])
def test_a_shard_only_one_build_wrote_is_a_difference(tmp_path, monkeypatch, side):
    held = {"shard-B": (b"same", (".bin", ".idx"))}
    fields = _put_digest_row(tmp_path, monkeypatch, **{side: held})
    assert fields == [OWN_DIGESTS]


def test_a_shard_one_build_wrote_in_part_is_a_difference_and_not_an_error(tmp_path, monkeypatch):
    fields = _put_digest_row(tmp_path, monkeypatch,
                             repaired={"shard-B": (b"same", (".bin",))},
                             reference={"shard-B": (b"same", (".bin", ".idx"))})
    assert fields == [OWN_DIGESTS]


# --- Where the returning copies land: the placement the report is asked with, and its landing ----------------
#
# The partitioned report of every generated set is asked with a placement written from the documented
# layout rule (docs/replica.md: one shard per source file, named for the file, and shard-gen for the
# generation outputs); the fixture (a record-list layout) and the round-robin layout state none. The
# landing the report gives for a shard the store does not hold (set/shard-B in the 10k set) is held to
# the same rule and the plant list, never to the report.


def _helper(module, name):
    """A function of a helper module, or a failed assertion that names it: the tests below
    state what the helper must give, and a helper that lacks a function fails them by name."""
    found = getattr(module, name, None)
    assert callable(found), f"{module.__name__} has no {name}"
    return found


def test_the_placement_file_is_the_documented_rule_for_the_sources_of_the_set(tmp_path):
    folder = _write_set(tmp_path, ("A", "B", "C", "Z"))
    written = _helper(repair_files, "write_placement")(tmp_path, folder, "set/")
    assert written == _helper(repair_files, "placement_path")(tmp_path)
    assert json.loads(written.read_text(encoding="utf-8")) == {
        "form": "waterwhite.placement/1",
        "by": "source",
        "shards": {"src/A": "set/shard-A", "src/B": "set/shard-B", "src/C": "set/shard-C",
                   "src/Z": "set/shard-Z"},
        "otherwise": "set/shard-gen",
    }


def test_a_source_the_set_adds_has_its_shard_in_the_placement(tmp_path):
    folder = _write_set(tmp_path, ("A", "B", "Z2"))
    shards = _helper(repair_files, "placement_for")(folder, "set/")["shards"]
    assert shards == {"src/A": "set/shard-A", "src/B": "set/shard-B", "src/Z2": "set/shard-Z2"}


def test_the_round_robin_report_is_asked_with_no_placement(tmp_path, monkeypatch):
    repairs, calls = _stub_historical(tmp_path, monkeypatch, "round-robin", ROUND_ROBIN_INSTANCES)
    repairs.historical("10k", "round-robin")
    assert "--placement" not in calls[0]


def _fake_plant(*outcomes):
    """A plant list of one release whose rows are re-sourced as the outcomes say: each outcome
    is (the id of the copy it takes, the source and line of the copy)."""
    inputs = [f"A-{n:06d}" for n, _ in enumerate(outcomes, 1)]
    expected = {"R1": {record: {"row": "present", "withdraw_A": "re-source", "resource_to": copy_id}
                       for record, (copy_id, _) in zip(inputs, outcomes)}}
    by_id = {copy_id: {"id": copy_id, "source": source, "line": line}
             for copy_id, (source, line) in outcomes}
    return SimpleNamespace(release_inputs={"R1": inputs}, expected=expected, by_id=by_id)


def test_the_expected_landing_names_each_shard_the_store_does_not_hold_with_the_copies_of_its_source():
    plant = _fake_plant(("B-000007", ("B", 7)), ("B-000002", ("B", 2)), ("C-000004", ("C", 4)))
    assert _helper(repair_files, "returning_copies")(plant) == {
        "B": ["src/B:L2", "src/B:L7"], "C": ["src/C:L4"]}
    assert _helper(repair_files, "expected_landing")(plant, "set/", held=["set/shard-C", "set/shard-A"]) == [
        {"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L2", "src/B:L7"]}]


def test_the_landing_of_the_report_agrees_with_the_rule_or_fails_by_name():
    wanted = [{"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L2", "src/B:L7"]}]
    check_landing = _helper(repair_files, "check_landing")
    assert check_landing(wanted, [
        {"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L7", "src/B:L2"]}]) == []
    for got, field in [
        ([], "landing of set/shard-B"),                                              # not named
        ([{"shard": "set/shard-B", "in_store": True, "receives": ["src/B:L2", "src/B:L7"]}],
         "landing of set/shard-B"),                                                  # held
        ([{"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L2"]}],
         "landing of set/shard-B"),                                                  # one copy short
        ([{"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L2", "src/B:L7"]},
          {"shard": "set/shard-Q", "in_store": False, "receives": []}],
         "landing of set/shard-Q"),                                                  # another shard
    ]:
        assert [m.field for m in check_landing(wanted, got)] == [field], got
    assert [m.field for m in check_landing(wanted, None)] == ["landing"]
    assert [m.field for m in check_landing([], [])] == ["landing"]    # nothing expected tests nothing


def test_the_landing_of_a_set_is_read_from_the_plant_list_and_the_report_of_the_historical_build(
        tmp_path, monkeypatch):
    repairs, hist, _ = _set_repairs(tmp_path, monkeypatch)
    plant = _fake_plant(("B-000007", ("B", 7)), ("B-000002", ("B", 2)))
    monkeypatch.setattr(acceptance_sets, "load_script", lambda file: SimpleNamespace(
        load_set=lambda folder: ({}, []), Plant=lambda settings, rows: plant, RE_SOURCE="re-source"))
    hist_report = dict(hist.report, groups={"placements": {"partitioned": {"landing": [
        {"shard": "set/shard-B", "in_store": False, "receives": ["src/B:L2", "src/B:L7"]}]}}})
    repairs._historical[("10k", "partitioned")] = repair_files.Historical(
        "partitioned", hist.store, hist.out, "set/", ("set/shard-A", "set/shard-Z"), {}, hist_report)
    put_landing = _helper(repair_files, "put_landing")
    assert put_landing(repairs, "10k") == []
    hist_report["groups"]["placements"]["partitioned"]["landing"][0]["receives"] = ["src/B:L2"]
    assert [m.field for m in put_landing(repairs, "10k")] == ["landing of set/shard-B"]


@pytest.mark.parametrize("size", ["10k"])
def test_the_copies_of_b_land_in_the_shard_of_b_the_store_does_not_hold(size, repairs):
    # A long run on the built set: the one test that asks the report for its landing. It is
    # no row of the table and is not counted among the rows of the check.
    if not acceptance_sets.megatron_root():
        pytest.skip("not run: its shards are tokenized only where the Megatron environment is set")
    if not c9.long_run_cleared(size):
        pytest.skip(f"a long run: only when {c9.LONG_RUNS_VARIABLE} names {c9.CHECK}:{size}")
    mismatches = _helper(repair_files, "put_landing")(repairs, size)
    assert not mismatches, [str(m) for m in mismatches]
