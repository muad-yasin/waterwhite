"""What the reader and the world say of the rows of Checks 1, 2, 4 and 8: how each row
is read, what it waits for, and the comparisons added for them (``loud_stop``, a
listing's ``covers``). Nothing here starts the tool.
"""

from __future__ import annotations

import acceptance_runs
import acceptance_sets
import acceptance_world
import pytest
import table
from acceptance_world import World
from table import Mismatch, Outcome, Row, TableError


def row(given: dict, expect: dict, check: int = 4) -> Row:
    return Row(id=f"c0{check}.test", check=check, question="?", plan="5.5", core=True,
               extension=None, given=given, expect=expect, expect_under={}, source="test",
               rule=None)


def kinds(check: int) -> dict[str, tuple[str, tuple[str, ...]]]:
    return {r.id: (q.kind, q.needs) for r in table.load_check(check)
            for q in [table.interpret(r)]}


# --- Every row of the five checks is read --------------------------------------------------------


@pytest.mark.parametrize("check", [1, 2, 4, 5, 8])
def test_every_row_of_the_check_is_read(check):
    assert len(kinds(check)) == len(table.load_check(check))


def test_check_4_rows_ask_a_stop_a_write_to_the_scratch_store_or_an_answer():
    by = kinds(4)
    assert by["c04.edge-parent.R1"] == (table.KIND_ANSWER, ())
    assert by["c04.artifact-digest.R2"] == (table.KIND_ANSWER, ())
    assert by["c04.trace-count.E@6"] == (table.KIND_ANSWER, (table.NEED_TRACE, table.NEED_CHECKPOINT))
    assert by["c04.count-499"][1] == (table.NEED_SCRATCH, table.NEED_SHARD)
    assert by["c04.kill-after-100"] == (table.KIND_SCRATCH_WRITE,
                                        (table.NEED_SCRATCH, table.NEED_SHARD))
    assert by["c04.declare-499.query"][0] == table.KIND_ANSWER


def test_check_2_rows_ask_a_registration_or_the_answers_after_it():
    by = kinds(2)
    assert by["c02.ii.replay-rejected"] == (table.KIND_RUN_REGISTRATION, (table.NEED_RUN_CAPTURE,))
    assert table.NEED_REPLAYED_ARRAYS in by["c02.i.arrays.E/seg-1"][1]
    assert by["c02.ii.E@6"][0] == table.KIND_ANSWER
    assert by["c02.iv.K/mix-1"][1] == (table.NEED_RUN_CAPTURE, table.NEED_MIXTURE)


def test_check_1_rows_that_compare_the_join_wait_for_the_runs_and_for_its_rows():
    by = kinds(1)
    for name in ("c01.E.diff", "c01.E.T", "c01.K.warm-start", "c01.E.step8.counted"):
        assert by[name][0] == table.KIND_JOIN_ROWS
        assert {table.NEED_RUN_CAPTURE, table.NEED_SHOW_TRACE} <= set(by[name][1])
    assert by["c01.E.E@9"][0] == table.KIND_ANSWER


def test_check_8_reads_the_shared_token_rows_as_answers_to_the_request_withdraw_w():
    by = kinds(8)
    assert by["c08.shared-token.E@9"] == (table.KIND_ANSWER,
                                          (table.NEED_CHECKPOINT, table.NEED_SHARED_TOKEN))
    assert by["c08.K.digests-match"][0] == table.KIND_K_DIGESTS
    assert by["c08.withdraw-A.E@9.step-8-holds-no-T"][0] == table.KIND_JOIN_ROWS


def test_a_given_of_the_run_kind_that_matches_no_question_is_an_error():
    with pytest.raises(TableError, match="no question of Checks 1, 2, 4 or 8"):
        table.interpret(row({"run": "E", "request": "withdraw A", "artifact": "E@9"},
                            {"answer": "contains"}))
    with pytest.raises(TableError, match="a write to the scratch store"):
        table.interpret(row({"store": "scratch-500", "artifact": "c04/shard-500",
                             "state": "w.retry", "run": "E"}, {"one_of": []}))


# --- loud_stop, listings, handles ------------------------------------------------------------


ANSWER = {"answer": "may contain", "reason": None, "names": None, "names_detail": None,
          "failing": None, "candidates": [], "chain": [], "basis": None, "notes": [],
          "untraced_steps": [], "first_exposure": None, "scope": None}


def test_loud_stop_false_is_an_answer_and_not_a_stop():
    expect = {"answer": "may contain", "loud_stop": False}
    answered = Outcome(0, __import__("json").dumps(ANSWER), "")
    assert table.compare_outcome(expect, answered, lambda ref: "L1") == []
    stopped = Outcome(3, "", '{"error": "trace sample count mismatch"}')
    (found,) = table.compare_outcome(expect, stopped, lambda ref: "L1")
    assert found.field == "loud_stop" and "trace sample count mismatch" in str(found.got)
    # An answer that differs is still found, loud_stop or not.
    (other,) = table.compare_outcome({"answer": "contains", "loud_stop": False}, answered,
                                     lambda ref: "L1")
    assert other.field == "answer"


def test_loud_stop_beside_a_stop_is_an_error():
    with pytest.raises(TableError, match="loud_stop"):
        table.compare_outcome({"error": "x", "loud_stop": False}, Outcome(3, "", ""),
                              lambda ref: "L1")


LISTING = {"covers": {"runs": ["E", "D"], "shards": ["E/shard-2", "E/shard-3"]},
           "not_covered": [{"artifact": "E/shard-1", "reason": "record missing", "kind": "shard"}]}
WANTED = {"covers": {"runs": ["D", "E"], "shards": ["E/shard-2", "E/shard-3"]},
          "not_covered": [{"artifact": "E/shard-1", "reason": "record missing"}]}


def test_a_listing_is_compared_by_what_it_covers_and_lists_as_not_covered_as_sets():
    assert table.compare_listing(WANTED, LISTING) == []
    omitted = {**LISTING, "not_covered": []}
    (found,) = table.compare_listing(WANTED, omitted)
    assert found.field == "not_covered"
    left_in = {**LISTING, "covers": {**LISTING["covers"], "shards": ["E/shard-1", "E/shard-2",
                                                                      "E/shard-3"]}}
    (found,) = table.compare_listing(WANTED, left_in)
    assert found.field == "covers.shards"
    wrong_reason = {**LISTING, "not_covered": [{"artifact": "E/shard-1", "reason": "not registered"}]}
    assert [m.field for m in table.compare_listing(WANTED, wrong_reason)] == ["not_covered"]


def test_the_listing_row_of_check_5_needs_the_runs_because_it_names_runs_and_checkpoints():
    (listing,) = [r for r in table.load_check(5) if r.id == "c05.listing.map-deleted"]
    assert table.interpret(listing).needs == (table.NEED_MAP, table.NEED_RUN_CAPTURE)


def test_w_is_a_record_of_z_and_the_table_names_it_by_its_record_id():
    assert table.record_ref("Z-089") == table.RecordRef("src/Z", "Z-089")
    assert table.manifest_handles("Z-089") == ("Z-089",)


# --- What the world says of the run needs ----------------------------------------------------


class Built:
    def __init__(self, names):
        self.instances = {name: "id" for name in names}


def need(*needs, store="main"):
    r = row({}, {}, 5)
    return table.Question(row=r, kind=table.KIND_ANSWER, store=store, state="unmutated",
                          request=None, artifacts=("src/A",), record=None, expect={}, needs=needs)


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_the_run_needs_are_one_need_whichever_asked(world, monkeypatch):
    monkeypatch.setattr(acceptance_runs, "capture_missing", lambda: "no command")
    asked = need(table.NEED_TRACE, table.NEED_CHECKPOINT, table.NEED_SHARED_TOKEN,
                 table.NEED_RUN_CAPTURE)
    assert world.unmet(asked) == [table.NEED_RUN_CAPTURE]


def test_the_run_needs_are_met_where_the_commands_exist_and_the_environment_is_set(
        world, monkeypatch, tmp_path):
    monkeypatch.setattr(acceptance_runs, "capture_missing", lambda: None)
    asked = need(table.NEED_TRACE, table.NEED_CHECKPOINT)
    monkeypatch.delenv(acceptance_sets.MEGATRON_VARIABLE, raising=False)
    assert world.unmet(asked) == [acceptance_world.NEED_MEGATRON_ENV]
    monkeypatch.setenv(acceptance_sets.MEGATRON_VARIABLE, str(tmp_path))
    assert world.unmet(asked) == []


def test_a_map_is_met_where_the_stores_build_registered_a_shard(world, monkeypatch, tmp_path):
    held = ["src/A"]
    monkeypatch.setattr(world, "store", lambda name: Built(held))
    monkeypatch.setenv(acceptance_sets.MEGATRON_VARIABLE, str(tmp_path))
    assert world.unmet(need(table.NEED_MAP)) == [table.NEED_MAP]
    held.append("E/shard-1")
    assert world.unmet(need(table.NEED_MAP)) == []
    monkeypatch.delenv(acceptance_sets.MEGATRON_VARIABLE)
    assert world.unmet(need(table.NEED_MAP)) == [acceptance_world.NEED_MEGATRON_ENV]


def test_what_no_command_prints_is_not_built_whatever_else_is(world, monkeypatch, tmp_path):
    monkeypatch.setattr(acceptance_runs, "capture_missing", lambda: None)
    monkeypatch.setenv(acceptance_sets.MEGATRON_VARIABLE, str(tmp_path))
    monkeypatch.setitem(acceptance_world.NOT_BUILT, table.NEED_REPORT, "no report")
    assert world.unmet(need(table.NEED_RUN_CAPTURE, table.NEED_REPORT)) == [table.NEED_REPORT]


def test_the_shuffle_row_needs_the_runs_only_and_waits_by_name_where_the_output_is_absent():
    """The shuffle is read from what register-run prints: the row asks no need of the probe,
    and ``World._k_digests`` skips it by name (test_launches_epochs_shuffle.py)."""
    assert kinds(8)["c08.K.digests-match"] == (table.KIND_K_DIGESTS, (table.NEED_RUN_CAPTURE,))
    assert not hasattr(table, "NEED_SHOW_SHUFFLE")


def test_the_request_withdraw_w_asks_the_spec_s_source_and_record(world, monkeypatch):
    shared = acceptance_runs.spec()["positions"]["shared_token"]
    seen = []
    monkeypatch.setattr(world, "locator", lambda store, ref: seen.append(ref) or "L89")
    r = row({"request": "withdraw W", "artifact": "E@9", "state": "unmutated",
             "record": "pending:positions.shared_token.record"},
            {"answer": "contains"}, check=8)
    question = table.interpret(r)
    assert world.query_arguments(question) == [
        "query", "withdraw", shared["source"], "--record", "src/Z:L89", "--artifact", "E@9"]
    assert seen == [table.RecordRef("src/Z", shared["record"])]
    wrong = row({"request": "withdraw W", "artifact": "E@9", "record": "pending:positions.T"},
                {"answer": "contains"}, check=8)
    with pytest.raises(acceptance_world.WorldError, match="not the spec's shared-token record"):
        world.query_arguments(table.interpret(wrong))
