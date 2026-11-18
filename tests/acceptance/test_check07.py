"""Check 7, requests (plan 5.5 Check 7): the rows of fixture/table/check07.toml,
one test per row, graded against the tool through its command line.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(7)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check07(row, part, world, request):
    grade(row, part, world, request)


# --- The reading of the rebuild report, put to documents written by hand -----------------------
#
# The tool's `report` and the replica's `repair` are not in the tree these tests were written
# in, so each reading is put to a report written by hand in the form of docs/interfaces.md 7.1,
# through the same World methods the rows use, with the command line and the replica's
# manifests replaced. A report that agrees must give no difference; the same report changed in
# one place must give the difference. These tests are the evidence that a row can fail.

import copy  # noqa: E402
import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from acceptance_world import REPAIRED, World, WorldError  # noqa: E402
from table import Outcome  # noqa: E402

LINES = {("R1", "X"): "L1", ("R1", "P"): "L2", ("R1", "T"): "L3", ("R1", "S"): "L4",
         ("src/B", "X@B"): "L1", ("src/A", "X@A"): "L1", ("R1-swap", "X"): "L1",
         ("R1-swap", "P"): "L2", ("R1-swap", "T"): "L3", ("R1-swap", "S"): "L4",
         ("import/T", "T"): "L1", **{("R1", f"U-{n:03d}"): f"L{4 + n}" for n in range(1, 21)}}
TEXTS = {("R1", "X"): "text of X", ("gen/P", "P"): "text of P", ("gen/T", "T"): "text of T",
         ("gen/S", "S"): "text of S"}


def _entry(record, action, **more):
    base = {"artifact": "R1", "record": record, "action": action, "chain": [],
            "new_parent": None, "withdrawn_parent": None, "regenerate_with": None, "step": None,
            "missing_link": None, "failing": None, "names": None, "review_scope": None}
    return {**base, **more}


def _report() -> dict:
    return {
        "form": "waterwhite.report/1",
        "request": {"kind": "withdraw", "source": "A", "members": [], "records": []},
        "records": [
            _entry("R1:L1", "re-source", new_parent="src/B:L1", withdrawn_parent="src/A:L1"),
            _entry("src/B:L1", "promote"),
            _entry("R1:L2", "retire", regenerate_with="paraphrase"),
            _entry("R1:L3", "retire", regenerate_with="translation"),
            _entry("R1:L4", "retire", regenerate_with="summary"),
            *[_entry(f"R1:L{n}", "keep") for n in range(5, 25)],
            _entry("import/T:L1", "unresolved", artifact="import/T",
                   missing_link="the record's own parent; only the file-level link to A/r1 is "
                                "recorded",
                   review_scope={"artifact": "import/T", "records": "every record of the file",
                                 "count": 1}),
            _entry("R1-swap:L1", "keep", artifact="R1-swap"),
            *[_entry(f"R1-swap:L{n}", "retire", artifact="R1-swap") for n in (2, 3, 4)],
        ],
        "artifacts": [
            {"artifact": "R1", "kind": "jsonl", "action": "rewrite", "answer": "contains",
             "counts": {"keep": 20, "re-source": 1, "promote": 1, "retire": 3, "unresolved": 0},
             "records_removed": 3, "removed": ["R1:L2", "R1:L3", "R1:L4"],
             "rows_in_successor": 21},
            {"artifact": "R2", "kind": "jsonl", "action": "keep", "answer": "does not contain"},
            {"artifact": "R1-swap", "kind": "jsonl", "action": "rewrite", "answer": "contains",
             "counts": {"keep": 21, "re-source": 0, "promote": 0, "retire": 3, "unresolved": 0},
             "records_removed": 3},
        ],
    }


class StubWorld(World):
    """A World whose store, command line, manifests and registered files are written by hand."""

    def __init__(self, base, report, *, successor_texts=(), edges=None, answers=None):
        super().__init__(base)
        self.report, self.edges, self.answers = report, edges or {}, answers or {}
        self.successor_file = base / "successor-R1.jsonl"
        self.successor_file.write_text(
            "".join(json.dumps({"text": t}) + "\n" for t in successor_texts), encoding="utf-8")
        self.commands: list[tuple[str, ...]] = []

    def store(self, name):
        return SimpleNamespace(instances={"R1": "inst-R1", "R2": "inst-R2"}, out=self.base)

    def copy(self, name, state):
        return SimpleNamespace(store=self.base / f"{name}-{state}.sqlite", out=self.base,
                               instances={})

    def locator(self, store, ref):
        try:
            return LINES[(ref.artifact, ref.handle)]
        except KeyError:
            raise WorldError(f"no line of {ref}") from None

    def text(self, store, ref):
        return TEXTS[(ref.artifact, ref.handle)]

    def cli(self, store_file, *arguments):
        self.commands.append(arguments)
        if arguments[0] == "report":
            return Outcome(0, json.dumps(self.report), "")
        artifact = arguments[arguments.index("--artifact") + 1]
        return Outcome(0, json.dumps({"answer": self.answers[artifact], "notes": []}), "")

    def shown(self, store_file, *arguments, again=False):
        if arguments == ("successor of R1",):
            return {"artifact": arguments[0], "path": str(self.successor_file)}
        if arguments == ("R1",):
            return {"artifact": "R1", "instance": "inst-R1"}
        return {"edge": self.edges[arguments]}


def _row(row_id):
    return next(r for r in (*ROWS, *table.load_check(6)) if r.id == row_id)


def _put(world, row_id):
    return world.put(table.interpret(_row(row_id), ()))


@pytest.fixture
def stub(tmp_path):
    return StubWorld(tmp_path, _report(), answers={"R2": "does not contain", "R1": "contains"})


@pytest.mark.parametrize("row_id", [
    "c07.report.R1.X", "c07.report.R1.X@B", "c07.report.R1.P", "c07.report.R1.T",
    "c07.report.R1.S", "c07.report.R1.U", "c07.swap.report.X", "c07.swap.report.P-T-S",
    "c07.swap.report.counts", "c07.withdraw-A.R2", "c06.import.review-scope"])
def test_a_report_that_agrees_with_the_row_gives_no_difference(stub, row_id):
    assert _put(stub, row_id) == []
    assert any(c[0] == "report" for c in stub.commands)


def test_the_report_is_asked_with_the_request_of_the_row_and_never_with_an_artifact(stub):
    _put(stub, "c07.report.R1.X")
    assert stub.commands == [("report", "withdraw", "A")]


def _changed(report, record, **changes):
    edited = copy.deepcopy(report)
    entry = next(e for e in edited["records"] if e["record"] == record)
    entry.update(changes)
    return edited


@pytest.mark.parametrize(("row_id", "record", "changes", "field"), [
    ("c07.report.R1.X", "R1:L1", {"action": "keep"}, "action"),
    ("c07.report.R1.X", "R1:L1", {"new_parent": "src/A:L1"}, "new_parent"),
    ("c07.report.R1.X", "R1:L1", {"withdrawn_parent": None}, "withdrawn_parent"),
    ("c07.report.R1.X@B", "src/B:L1", {"action": "keep"}, "action"),
    ("c07.report.R1.P", "R1:L2", {"regenerate_with": "summary"}, "regenerate_with"),
    ("c07.report.R1.S", "R1:L4", {"action": "keep"}, "action"),
    ("c07.report.R1.U", "R1:L9", {"action": "retire"}, "action"),
])
def test_a_report_that_differs_in_one_place_gives_that_difference(tmp_path, row_id, record,
                                                                  changes, field):
    world = StubWorld(tmp_path, _changed(_report(), record, **changes))
    assert [m.field for m in _put(world, row_id)] == [field]


def test_a_missing_entry_is_a_difference_and_never_a_keep(tmp_path):
    report = _report()
    report["records"] = [e for e in report["records"] if e["record"] != "R1:L7"]
    found = _put(StubWorld(tmp_path, report), "c07.report.R1.U")
    assert sorted(m.field for m in found) == ["action", "count"]
    assert next(m for m in found if m.field == "count").got == 19


def test_a_record_of_a_kept_artifact_is_kept_through_its_artifact_entry(tmp_path):
    report = _report()
    report["records"] = [e for e in report["records"] if e["artifact"] != "R1-swap"]
    report["artifacts"][2].update(action="keep")
    # No record of R1-swap is listed and the artifact is kept: each reads as keep, so the row
    # that expects retire for P, T and S differs and the row that expects keep for X agrees.
    assert [m.field for m in _put(StubWorld(tmp_path, report), "c07.swap.report.P-T-S")] == [
        "action"]
    assert _put(StubWorld(tmp_path, report), "c07.swap.report.X") == []


def test_the_counts_of_an_artifact_are_compared_one_by_one(tmp_path):
    report = _report()
    report["artifacts"][2]["counts"]["keep"] = 20
    del report["artifacts"][2]["counts"]["unresolved"]
    found = _put(StubWorld(tmp_path, report), "c07.swap.report.counts")
    assert [(m.field, m.got) for m in found] == [("counts.keep", 20), ("counts.unresolved", None)]


def test_the_action_of_r2_is_read_from_the_report_beside_the_answer(tmp_path):
    report = _report()
    report["artifacts"][1]["action"] = "rewrite"
    world = StubWorld(tmp_path, report, answers={"R2": "does not contain"})
    assert [m.field for m in _put(world, "c07.withdraw-A.R2")] == ["action"]
    world = StubWorld(tmp_path, _report(), answers={"R2": "contains"})
    assert [m.field for m in _put(world, "c07.withdraw-A.R2")] == ["answer"]


def test_an_artifact_with_no_entry_is_a_difference(tmp_path):
    report = _report()
    report["artifacts"] = report["artifacts"][:1]
    world = StubWorld(tmp_path, report, answers={"R2": "does not contain"})
    assert [m.field for m in _put(world, "c07.withdraw-A.R2")] == ["artifact entry of R2"]


def test_a_document_of_another_form_is_an_error_of_the_test_and_a_failed_command_a_difference(
        tmp_path):
    with pytest.raises(table.TableError, match="form"):
        _put(StubWorld(tmp_path, {**_report(), "form": "waterwhite.report/2"}),
             "c07.report.R1.X")

    class Refusing(StubWorld):
        def cli(self, store_file, *arguments):
            return Outcome(2, "", json.dumps({"error": "source not registered"}))

    found = _put(Refusing(tmp_path, _report()), "c07.report.R1.X")
    assert [m.field for m in found] == ["rebuild report"] and "source not registered" in found[0].got


def test_the_unresolved_import_row_reads_the_missing_link_and_the_scope(tmp_path):
    report = _changed(_report(), "import/T:L1", missing_link="another sentence",
                      review_scope={"artifact": "import/T", "records": "the record", "count": 1})
    found = _put(StubWorld(tmp_path, report), "c06.import.review-scope")
    assert sorted(m.field for m in found) == ["missing_link", "review_scope"]


# --- The successor and the history: the store the repair registered them in ---------------------

EDGE_B = {"selected_parent": {"artifact": "src/B", "record": "L1"}, "label": "observed",
          "parents": []}
SUCCESSOR_TEXTS = ["text of X"] + [f"unrelated {n}" for n in range(20)]


def test_the_successor_row_agrees_when_x_comes_from_b_and_the_retired_rows_are_gone(tmp_path):
    world = StubWorld(tmp_path, _report(), successor_texts=SUCCESSOR_TEXTS,
                      edges={("successor of R1", "--record", "L1"): EDGE_B},
                      answers={"successor of R1": "does not contain"})
    assert _put(world, "c07.successor.R1.X") == []


@pytest.mark.parametrize(("texts", "edge", "field"), [
    (SUCCESSOR_TEXTS + ["extra"], EDGE_B, "rows_in_successor"),
    ([*SUCCESSOR_TEXTS[:20], "text of P"], EDGE_B, "absent"),
    (SUCCESSOR_TEXTS, {**EDGE_B, "selected_parent": {"artifact": "src/A", "record": "L1"}},
     "parent"),
    (SUCCESSOR_TEXTS, {**EDGE_B, "label": "inferred"}, "label"),
])
def test_the_successor_row_fails_where_the_successor_differs(tmp_path, texts, edge, field):
    world = StubWorld(tmp_path, _report(), successor_texts=texts,
                      edges={("successor of R1", "--record", "L1"): edge})
    assert field in {m.field for m in _put(world, "c07.successor.R1.X")}


def test_the_successor_is_put_to_the_repaired_store_and_never_to_the_one_the_other_rows_read(
        tmp_path):
    asked = []

    class Spy(StubWorld):
        def copy(self, name, state):
            asked.append((name, state))
            return super().copy(name, state)

    world = Spy(tmp_path, _report(), successor_texts=SUCCESSOR_TEXTS,
                edges={("successor of R1", "--record", "L1"): EDGE_B})
    _put(world, "c07.successor.R1.X")
    assert asked and {state for _, state in asked} == {REPAIRED}


def test_the_history_row_agrees_and_fails_when_the_instance_changed(tmp_path):
    edge = {"selected_parent": {"artifact": "src/A", "record": "L1"}}
    world = StubWorld(tmp_path, _report(), edges={("R1", "--record", "L1"): edge},
                      answers={"R1": "contains"})
    assert _put(world, "c07.history.R1.X") == []
    world.store = lambda name: SimpleNamespace(instances={"R1": "another"})
    assert [m.field for m in _put(world, "c07.history.R1.X")] == ["instance_id"]
