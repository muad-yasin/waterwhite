"""The reader of the hand-written table, tested on its own: a row it cannot
interpret is an error, and each field is compared the way fixture/table/README.md
section 6 says. Nothing here runs the tool.
"""

from __future__ import annotations

import pytest
import table
from table import Mismatch, Outcome, RecordRef, Row, TableError

NOTE = "the parent list is process evidence"
LINES = {"gen/T:T": "L1", "gen/P:P": "L1", "src/A:X@A": "L1", "src/C:SC": "L1", "gen/S:S": "L1",
         "R1:S": "L4"}


def resolve(ref: RecordRef) -> str:
    return LINES[str(ref)]


def row(given: dict, expect: dict, *, core: bool = True, extension=None, expect_under=None) -> Row:
    return Row(id="c05.test", check=5, question="?", plan="5.5", core=core, extension=extension,
               given=given, expect=expect, expect_under=expect_under or {}, source="test",
               rule=None)


def answer(**fields) -> dict:
    base = {"answer": "contains", "reason": None, "names": None, "names_detail": None,
            "failing": None, "candidates": [], "chain": [], "basis": None, "notes": [],
            "untraced_steps": [], "first_exposure": None, "scope": None}
    return base | fields


# --- Loading ---------------------------------------------------------------------------------


def test_the_committed_files_of_checks_5_and_7_load():
    assert len(table.load_check(5, table.TABLE_DIR)) == 224  # 221 and the three rows of E2 the amendment adds
    assert len(table.load_check(7, table.TABLE_DIR)) == 27


def test_a_row_with_an_unknown_key_or_a_repeated_id_is_refused(tmp_path):
    good = ('[[row]]\nid = "c07.a"\nquestion = "?"\nplan = "5.5"\ncore = true\n'
            'given = {}\nexpect = {}\nsource = "s"\n')
    (tmp_path / "check07.toml").write_text(good + good, encoding="utf-8")
    with pytest.raises(TableError, match="used twice"):
        table.load_check(7, tmp_path)
    (tmp_path / "check07.toml").write_text(good + 'expects = {}\n', encoding="utf-8")
    with pytest.raises(TableError, match="unknown keys"):
        table.load_check(7, tmp_path)
    with pytest.raises(TableError, match="does not exist"):
        table.load_check(5, tmp_path)


# --- Interpreting a row's given ----------------------------------------------------------------


def test_an_answer_row_names_its_store_state_request_and_artifact():
    q = table.interpret(row({"request": "withdraw A, record X@A", "artifact": "gen/T",
                             "state": "f.edge-file-level"}, {"answer": "may contain"}))
    assert (q.kind, q.store, q.state, q.artifact) == (
        table.KIND_ANSWER, "main", "f.edge-file-level", "gen/T")
    assert q.request.source == "A" and q.request.records == ("X@A",)
    assert q.needs == () and q.deferred == {}


@pytest.mark.parametrize("given", [
    {"request": "withdraw A", "artifact": "R1", "colour": "blue"},   # a key nobody defined
    {"request": "withdraw A", "artifact": "R1", "store": "attic"},
    {"request": "withdraw A", "artifact": "R1", "state": "f.unheard-of"},
    {"request": "withdraw Q", "artifact": "R1"},
    {"request": "withdraw A", "artifact": "the big file"},
    # README section 7 asks this fault with the record-level request.
    {"request": "withdraw A", "artifact": "gen/T", "state": "f.edge-file-level"},
    {"request": "withdraw A", "artifact": "R1", "artifacts": ["R1", "R2"]},
    {"artifact": "R1"},                                              # nothing is asked
    {"request": "withdraw A", "artifact": "R1", "extensions_on": ["content-ban"]},
    {"artifact": "R1", "action": "register R1 again, differently"},
])
def test_a_given_that_cannot_be_read_is_an_error(given):
    with pytest.raises(TableError):
        table.interpret(row(given, {"answer": "contains"}))


def test_an_expected_field_the_question_does_not_give_is_an_error():
    with pytest.raises(TableError, match="gives no field"):
        table.interpret(row({"request": "withdraw A", "artifact": "R1"},
                            {"answer": "contains", "colour": "blue"}))


def test_needs_follow_from_the_state_the_store_and_the_artifact():
    def needs(given):
        return table.interpret(row(given, {"answer": "contains"})).needs

    assert needs({"request": "withdraw A", "artifact": "E@6", "state": "f.trace-deleted"}) == (
        table.NEED_TRACE, table.NEED_CHECKPOINT)
    assert needs({"request": "withdraw A", "artifact": "E-rr/mix-1", "store": "rr"}) == (
        table.NEED_SHARD, table.NEED_MIXTURE)
    assert needs({"request": "withdraw A", "artifact": "src/A", "state": "f.map-deleted"}) == (
        table.NEED_MAP,)
    # The state q.unregistered asks about a name nothing holds: it needs no checkpoint.
    assert needs({"request": "withdraw A", "artifact": "E@12", "state": "q.unregistered"}) == ()
    off = table.interpret(row({"request": "ban X", "artifact": table.LISTING},
                              {"error": "extension off"}))
    assert (off.kind, off.needs) == (table.KIND_LISTING, ())


def test_a_field_of_the_rebuild_report_beside_answer_fields_is_deferred():
    q = table.interpret(row({"request": "withdraw A", "artifact": "R2"},
                            {"answer": "does not contain", "action": "keep", "notes": []}))
    assert q.kind == table.KIND_ANSWER and q.needs == ()
    assert q.deferred == {"action": table.NEED_REPORT}
    # While the report is not built the other fields are compared without it.
    waiting = q.waiting({table.NEED_REPORT: "not built"})
    assert dict(q.without(waiting).expect) == {"answer": "does not contain", "notes": []}
    assert q.waiting({}) == {} and "action" in q.without(q.waiting({})).expect
    # A field with no comparison written is an error, never ignored.
    with pytest.raises(TableError, match="no comparison"):
        table.compare_answer(q.expect, answer(), resolve)
    # A row that expects only report fields needs the report as a whole.
    q = table.interpret(row({"request": "withdraw A", "artifact": "R1", "record": "P"},
                            {"action": "retire"}))
    assert q.kind == table.KIND_RECORD and q.needs == (table.NEED_REPORT,)


def test_an_extension_row_is_read_as_one_and_the_fallback_in_force_picks_the_expectation():
    given = {"request": "ban X", "artifact": "R1", "extensions_on": ["content-ban"]}
    q = table.interpret(row(given, {"answer": "contains"}, core=False, extension="content-ban"))
    assert q.kind == table.KIND_EXTENSION
    k = row({"request": "withdraw A", "artifact": "K@3"}, {"answer": "does not contain"},
            expect_under={"cyclic-wrapper-route": {"answer": "may contain"}})
    assert table.expect_in_force(k) == {"answer": "does not contain"}
    assert table.expect_in_force(k, ("cyclic-wrapper-route",)) == {"answer": "may contain"}


# --- Handles ---------------------------------------------------------------------------------


def test_handles_have_a_home_file_and_a_spelling_in_the_manifests():
    assert table.record_ref("X@A") == RecordRef("src/A", "X@A")
    assert table.record_ref("src/B:X@B") == RecordRef("src/B", "X@B")
    assert table.record_ref("P", "R1") == RecordRef("R1", "P")
    assert table.record_ref("U-007") == RecordRef("src/Z", "U-007")
    assert table.manifest_handles("X") == ("X@A", "X@B")
    assert table.manifest_handles("U-007") == ("Z-007",)
    assert table.manifest_handles("T") == ("T",)
    with pytest.raises(TableError):
        table.record_ref("G6")


# --- Comparing an answer -------------------------------------------------------------------------


def fields(mismatches: list[Mismatch]) -> list[str]:
    return [m.field for m in mismatches]


def test_only_the_fields_a_row_holds_are_compared():
    got = answer(answer="contains", notes=[NOTE], basis="something")
    assert table.compare_answer({"answer": "contains"}, got, resolve) == []
    assert fields(table.compare_answer({"answer": "may contain"}, got, resolve)) == ["answer"]


def test_notes_are_the_full_set_in_any_order():
    got = answer(notes=["b", "a"])
    assert table.compare_answer({"notes": ["a", "b"]}, got, resolve) == []
    assert fields(table.compare_answer({"notes": ["a"]}, got, resolve)) == ["notes"]
    assert fields(table.compare_answer({"notes": []}, got, resolve)) == ["notes"]
    assert table.compare_answer({"notes": []}, answer(notes=[]), resolve) == []


def test_a_chain_passes_the_named_hops_in_order_with_other_hops_between():
    chain = [{"hop": "E/shard-3:D4"}, {"hop": "gen/T:L1"}, {"hop": "work/copy:L9"},
             {"hop": "gen/P:L1"}, {"hop": "src/A:L1"}]
    got = answer(chain=chain)
    assert table.compare_answer({"chain": ["gen/T:T", "gen/P:P", "src/A:X@A"]}, got, resolve) == []
    backwards = {"chain": ["gen/P:P", "gen/T:T"]}
    assert fields(table.compare_answer(backwards, got, resolve)) == ["chain"]
    assert fields(table.compare_answer({"chain": ["gen/S:S"]}, got, resolve)) == ["chain"]
    assert table.compare_answer({"chain": ["E/seg-2:step 7"]},
                                answer(chain=["E/seg-2:step 7", "E/shard-3:D4"]), resolve) == []


def test_the_failing_link_is_compared_key_by_key_and_its_record_is_one_record():
    got = answer(answer="may contain", failing={
        "condition": "g", "artifact": "src/C", "record": "L1", "job": "summary", "detail": "..."})
    expect = {"failing": {"condition": "g", "job": "summary", "record": "SC"}}
    assert table.compare_answer(expect, got, resolve) == []
    # The same locator in another file is another record.
    elsewhere = answer(failing=dict(got["failing"], artifact="gen/S"))
    assert fields(table.compare_answer(expect, elsewhere, resolve)) == ["failing.record"]
    wrong = {"failing": {"condition": "a", "job": "paraphrase", "record": "SC"}}
    assert fields(table.compare_answer(wrong, got, resolve)) == ["failing.condition", "failing.job"]
    assert fields(table.compare_answer(expect, answer(failing=None), resolve)) == ["failing"]
    with pytest.raises(TableError):
        table.compare_answer({"failing": {"colour": "blue"}}, got, resolve)


def test_scope_and_checkpoint_fields():
    got = answer(answer="does not contain", untraced_steps=[["E/seg-2", 8]],
                 first_exposure={"segment": "E/seg-2", "step": 7},
                 scope={"source": "A", "members": ["A/r1"], "records": ["src/A:L1"],
                        "steps": [["E/seg-1", 1, 6]],
                        "generation_edges": ["paraphrase", "translation"],
                        "walked": 3})
    expect = {"scope": {"source": "A", "members": ["A/r1"], "records": ["X@A"],
                        "steps": [["E/seg-1", 1, 6]],
                        "generation_edges": ["translation", "paraphrase"]},
              "untraced_steps": [["E/seg-2", 8]],
              "first_exposure": {"segment": "E/seg-2", "step": 7}}
    assert table.compare_answer(expect, got, resolve) == []
    off = {"scope": {"source": "C", "generation_edges": ["summary"], "steps": []},
           "untraced_steps": [], "first_exposure": {"segment": "E/seg-2", "step": 8}}
    assert fields(table.compare_answer(off, got, resolve)) == [
        "untraced_steps", "first_exposure", "scope.source", "scope.generation_edges", "scope.steps"]


# --- Comparing what the command line printed -------------------------------------------------


def test_a_stop_is_the_error_name_the_exit_code_and_an_empty_standard_output():
    expect = {"error": "extension off", "exit_code": 2, "stdout_empty": True}
    stopped = Outcome(2, "", '{"error": "extension off", "detail": "..."}\n')
    assert table.compare_outcome(expect, stopped, resolve) == []
    answered = Outcome(0, '{"answer": "contains"}\n', "")
    assert fields(table.compare_outcome(expect, answered, resolve)) == [
        "error", "exit_code", "stdout_empty"]


def test_a_stop_where_an_answer_was_expected_is_a_mismatch_of_the_answer():
    stopped = Outcome(3, "", '{"error": "map digest mismatch"}\n')
    got = table.compare_outcome({"answer": "contains"}, stopped, resolve)
    assert fields(got) == ["answer"] and "map digest mismatch" in got[0].got
    with pytest.raises(TableError):
        table.compare_outcome({"answer": "contains", "exit_code": 0}, stopped, resolve)


# --- The row forms of Checks 3 and 6 ----------------------------------------------------------

SET = {"set": "100k", "layout": "round-robin"}


def test_the_committed_files_of_checks_3_and_6_load_and_every_row_has_one_form():
    rows3, rows6 = table.load_check(3, table.TABLE_DIR), table.load_check(6, table.TABLE_DIR)
    assert (len(rows3), len(rows6)) == (15, 17)
    kinds = {r.id: table.interpret(r).kind for r in rows3 + rows6}
    # The same keys with another sentence are another question.
    assert kinds["c03.spans.per-ordinal"] == table.KIND_SPANS
    assert kinds["c03.records.capture"] == table.KIND_RECORDS
    assert kinds["c03.records.position-recovery"] == table.KIND_RECORDS
    assert kinds["c03.records.token-matching"] == table.KIND_RECORDS_AS_CAPTURE
    assert kinds["c03.drops.no-tokens.planted-line"] == table.KIND_PLANTED_LINE
    assert kinds["c06.fixture.candidates.X"] == table.KIND_EDGE_LABEL
    assert kinds["c06.import.file-family"] == table.KIND_FILE_FAMILY
    # Rows of Check 6 are answers and are put as the rows of Checks 5 and 7 are. The
    # import's parent and missing link are fields of the answer (plan 2.1, L1): the
    # row that asks them is an answer row once the table says so. Its action and
    # the scope needing review are fields of the rebuild report (4.1): that row is
    # one of the report, which waits for it.
    answers = sorted(name for name, kind in kinds.items() if kind == table.KIND_ANSWER)
    always = {"c06.fixture.downstream.E/shard-3", "c06.fixture.downstream.R1",
              "c06.fixture.downstream.gen/T", "c06.import.record-request",
              "c06.import.withdraw-A"}
    assert always <= set(answers) <= always | {"c06.import.parent-unresolved"}
    reports = {name: kind for name, kind in kinds.items()
               if kind in (table.KIND_RECORD, table.KIND_REPORT)}
    assert "c06.import.review-scope" in reports
    for name in reports:
        assert table.NEED_REPORT in table.interpret(
            next(r for r in rows3 + rows6 if r.id == name)).needs
    assert set(kinds.values()) <= table.FORM_KINDS | {table.KIND_ANSWER, table.KIND_RECORD,
                                                      table.KIND_REPORT}
    # Every row but the precondition on the spec and the planted line is asked of a set
    # or of a store of the table, never of both.
    for r in rows3 + rows6:
        q = table.interpret(r)
        assert q.sized == ("set" in r.given and q.kind != table.KIND_SPEC_ITEM)


def test_a_row_of_check_3_is_told_by_its_sentence_and_names_its_maps():
    q = table.interpret(row({**SET, "map": "capture"},
                            {"per_ordinal": table.S_SPAN, "mismatches": 0}))
    assert (q.kind, q.sized, q.maps, q.needs) == (
        table.KIND_SPANS, True, (("capture", "position"),), (table.NEED_SHOW_MAP_ROWS,))
    q = table.interpret(row({**SET, "maps": ["capture", "position recovery",
                                             "token-matching recovery"]},
                            {"label": "recovered", "record": table.S_OWN_SHARD_LINE}))
    assert q.kind == table.KIND_SPLIT_PAIRS and q.maps == (
        ("capture", "position"), ("recovery", "position"), ("recovery", "token-matching"))
    assert table.sizes_of(q.row) == ("100k",)
    assert table.sizes_of(row({"set": "generated", "store": "no-capture"}, {})) == ("10k", "100k")
    # A sentence that was changed is a question no comparison was written for.
    with pytest.raises(TableError, match="in words"):
        table.interpret(row({**SET, "map": "capture"},
                            {"per_ordinal": "token_end equals the document's size",
                             "mismatches": 0}))
    with pytest.raises(TableError, match="no map"):
        table.interpret(row({**SET, "map": "the best map"},
                            {"per_ordinal": table.S_SPAN, "mismatches": 0}))
    with pytest.raises(TableError, match="no row form"):
        table.interpret(row({**SET, "map": "capture"}, {"colour": "blue"}))
    with pytest.raises(TableError, match="no set"):
        table.sizes_of(row({"set": "1M"}, {}))


def test_fields_that_wait_are_deferred_and_a_fallback_takes_a_row_out():
    maps = ["capture", "position recovery", "token-matching recovery"]
    q = table.interpret(row({**SET, "maps": maps}, {
        "capture": {}, "position_recovery": {}, "token_matching_recovery": {},
        "maps_per_shard": 3, "labels_of_documents": ["recovered", "candidates"]}))
    assert q.kind == table.KIND_MAP_FIELDS and q.needs == ()
    assert q.deferred == {"labels_of_documents": table.NEED_SHOW_MAP_ROWS}
    # The import's parent and missing link are fields of the answer (plan 2.1, L1);
    # its action and the scope needing review are fields of the rebuild report (4.1).
    q = table.interpret(row({"store": "import", "request": "withdraw A, record X@A",
                             "artifact": "import/T", "state": "unmutated"},
                            {"answer": "may contain", "candidates": ["A"],
                             "failing": {"condition": "b", "artifact": "import/T",
                                         "record": "T"}}))
    assert q.kind == table.KIND_ANSWER and q.needs == () and q.deferred == {}
    q = table.interpret(row({"store": "import", "request": "withdraw A", "artifact": "import/T",
                             "record": "T", "state": "unmutated"},
                            {"action": "unresolved", "missing_link": "...",
                             "review_scope": {"count": 1}}))
    assert q.kind == table.KIND_RECORD and q.needs == (table.NEED_REPORT,)
    # The stores of Checks 3 and 6 carry no need of their own: the replica says
    # whether it builds them.
    q = table.interpret(row({"store": "no-capture", "artifact": "E/shard-1",
                             "map": "position recovery"},
                            {"documents": 18, "recovered": 18, "candidates": 0, "unknown": 0,
                             "derivation": "position", "verification": "passed"}))
    assert (q.kind, q.needs, q.sized) == (table.KIND_DOCUMENTS, (table.NEED_SHARD,), False)
    swapped = row({**SET, "map": "position recovery", "state": "t.twin-swap"},
                  {"label": "candidates", "candidate_set": table.S_UNCHANGED},
                  expect_under={"F1": {"applies": False}})
    assert table.interpret(swapped).kind == table.KIND_TWINS_SWAPPED
    assert table.interpret(swapped, ("F1",)).kind == table.KIND_DOES_NOT_APPLY


def test_never_lists_the_answers_a_row_must_not_get():
    got = answer(answer="may contain")
    expect = {"answer": "may contain", "never": ["does not contain", "contains"]}
    assert table.compare_answer(expect, got, resolve) == []
    assert fields(table.compare_answer({"never": ["may contain"]}, got, resolve)) == ["never"]
    q = table.interpret(row({"store": "no-capture", "request": "withdraw A", "artifact": "R1"},
                            expect))
    assert q.kind == table.KIND_ANSWER and q.needs == ()


# --- Pending values -------------------------------------------------------------------------------


def test_a_pending_value_is_filled_from_the_spec_or_reported_as_empty():
    spec = {"positions": {"K": {"first_read_of_T": {"step": 5}}, "T": {}}}
    expect = {"first_exposure": {"segment": "K/seg-1",
                                 "step": "pending:positions.K.first_read_of_T.step"}}
    assert table.fill_pending(expect, spec) == (
        {"first_exposure": {"segment": "K/seg-1", "step": 5}}, ())
    filled, empty = table.fill_pending({"a": ["pending:positions.T", "pending:positions.W.step"]},
                                       spec)
    assert empty == ("positions.T", "positions.W.step")
