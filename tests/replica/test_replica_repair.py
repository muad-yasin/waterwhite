"""``python -m replica repair``: the replica stands in for the team's builder and turns a
rebuild report into a repaired side (plan 4.4; docs/interfaces.md section 9.5).

The report is a file the replica reads with the ``json`` module: `form`, `artifacts`
and `records`. The reports below are written by hand from the hand-written table
(``fixture/table/check07.toml``, row ids ``c07.report.R1.*``): on the fixture, under
"withdraw A", X of R1 is re-sourced from A's copy to B's copy, B's copy is promoted, P,
T and S are retired, and the 20 rows of Z are kept; R2 is kept. A test changes one entry
of the report, as the states ``r.descendant-kept`` and ``r.unrelated-paraphrase-retired``
do, and reads what the repaired side holds from the files and the manifests the build
wrote and from the store it registered into. The ids of the records are the spec's
handles (``fixture/spec/fixture_spec.toml``, ``[handles]``): A's copy of X is ``A-001``,
B's is ``B-001``, P, T and S are their own ids.
"""

from __future__ import annotations

import json
import shutil

import pytest
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica import cli
from replica.fixture import build_captured_fixture, default_spec_path


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """The historical build of the fixture, captured into a store, without tokenization
    and with no layout: the sources, the generation outputs and the releases R1 and R2."""
    folder = tmp_path_factory.mktemp("repair-historical") / "history"
    return build_captured_fixture(folder, releases=("R1", "R2"), imports=(), layouts=())


@pytest.fixture
def copy_of_history(historical, tmp_path):
    """A copy of the historical folder that a repair may register into."""
    target = tmp_path / "history"
    shutil.copytree(historical.out.parent, target)
    return historical.__class__(
        store=target / historical.store.name, out=target / historical.out.name,
        instances=dict(historical.instances), summary=historical.summary,
        files=dict(historical.files), spec=historical.spec or default_spec_path(), options=historical.options,
        plan=historical.plan,
    )


def table_report(built, **changes: str) -> dict:
    """The report of "withdraw A" on the fixture as the table has it. ``changes`` maps
    a record of R1 (``R1:L7``) to another action."""
    rows = {row["handle"]: row["line"] for row in built.manifest_rows("R1")}
    entries = [
        {"artifact": "R1", "record": f"R1:L{rows['X@A']}", "action": "re-source",
         "new_parent": "src/B:L1", "withdrawn_parent": "src/A:L1"},
        {"artifact": "R1", "record": "src/B:L1", "action": "promote"},
    ]
    for handle, line in rows.items():
        if handle in ("P", "T", "S"):
            entries.append({"artifact": "R1", "record": f"R1:L{line}", "action": "retire",
                            "regenerate_with": {"P": "paraphrase", "T": "translation",
                                                "S": "summary"}[handle]})
        elif handle != "X@A":
            entries.append({"artifact": "R1", "record": f"R1:L{line}", "action": "keep"})
    for entry in entries:
        entry["action"] = changes.get(entry["record"], entry["action"])
    return {
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
        "records": entries,
        "artifacts": [
            {"artifact": "R1", "action": "rewrite", "successor": "successor of R1"},
            {"artifact": "R2", "action": "keep", "successor": None},
        ],
    }


def repair(built, tmp_path, report: dict, capsys, *extra: str):
    """Run the command; returns (exit code, the summary it printed or None, standard error,
    the output folder)."""
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "repaired"
    spec = built.spec or default_spec_path()
    argv = ["repair", "--report", str(report_file), "--historical", str(built.out),
            "--out", str(out), "--store", str(built.store), "--spec", str(spec), *extra]
    try:
        code = cli.main(argv)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    captured = capsys.readouterr()
    summary = json.loads(captured.out) if captured.out.strip() else None
    return code, summary, captured.err, out


def lines_of(path):
    return [json.loads(line)["text"] for line in path.read_text(encoding="utf-8").split("\n") if line]


def handles_of(out):
    manifest = out / "releases" / "R1.jsonl.manifest.jsonl"
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").split("\n") if line]


def test_the_report_drives_the_leave_out_and_the_swap(copy_of_history, tmp_path, capsys):
    built = copy_of_history
    code, summary, err, out = repair(built, tmp_path, table_report(built), capsys)
    assert code == 0, err
    assert summary["left_out"] == ["P", "S", "T"]
    assert "A-001" in json.dumps(summary["swapped"]) and "B-001" in json.dumps(summary["swapped"])
    rows = handles_of(out)
    assert len(rows) == 21
    assert [row["handle"] for row in rows if row["handle"] in ("P", "T", "S")] == []
    # X keeps its place and takes B's copy: the row says so in its origin.
    assert rows[0]["origin"] == ["B-001"]
    assert len(lines_of(out / "releases" / "R1.jsonl")) == 21


def test_a_retired_record_that_the_report_keeps_stays_in_the_repaired_release(
        copy_of_history, tmp_path, capsys):
    built = copy_of_history
    line = built.line_of("R1", "P")
    code, summary, err, out = repair(built, tmp_path, table_report(built, **{f"R1:L{line}": "keep"}),
                                     capsys)
    assert code == 0, err
    assert summary["left_out"] == ["S", "T"]
    assert "P" in [row["handle"] for row in handles_of(out)]
    assert len(handles_of(out)) == 22


def test_a_kept_record_that_the_report_retires_leaves_the_repaired_release(
        copy_of_history, tmp_path, capsys):
    built = copy_of_history
    line = built.line_of("R1", "Z-001")
    code, summary, err, out = repair(built, tmp_path, table_report(built, **{f"R1:L{line}": "retire"}),
                                     capsys)
    assert code == 0, err
    assert summary["left_out"] == ["P", "S", "T", "Z-001"]
    assert "Z-001" not in [row["handle"] for row in handles_of(out)]
    assert len(handles_of(out)) == 20


def test_an_unresolved_record_changes_nothing(copy_of_history, tmp_path, capsys):
    built = copy_of_history
    line = built.line_of("R1", "Z-002")
    report = table_report(built, **{f"R1:L{line}": "unresolved"})
    code, summary, err, out = repair(built, tmp_path, report, capsys)
    assert code == 0, err
    assert summary["left_out"] == ["P", "S", "T"]
    assert len(handles_of(out)) == 21


def test_the_successors_are_registered_beside_the_historical_artifacts(
        copy_of_history, tmp_path, capsys):
    built = copy_of_history
    code, summary, err, out = repair(built, tmp_path, table_report(built), capsys)
    assert code == 0, err
    with Store.open(built.store, readonly=True) as store:
        historical_r1, historical_r2 = store.resolve("R1"), store.resolve("R2")
        successor_r1 = store.resolve("successor of R1")
        successor_r2 = store.resolve("successor of R2")
        assert historical_r1.instance_id == built.instances["R1"]
        assert historical_r2.instance_id == built.instances["R2"]
        # A rewritten file is a new instance; a file the repair left as it was is the same
        # instance, under a second name (DECISIONS.md D-14).
        assert successor_r1 is not None and successor_r1.instance_id != historical_r1.instance_id
        assert successor_r2 is not None and successor_r2.instance_id == historical_r2.instance_id
        assert {"R2", "successor of R2"} <= set(store.artifact(historical_r2.instance_id).names)
        # The sources were not changed: the second build gives them a second name only.
        assert store.resolve("successor of src/B").instance_id == built.instances["src/B"]


def test_the_sides_file_names_the_repaired_file_of_every_artifact_of_the_report(
        copy_of_history, tmp_path, capsys):
    built = copy_of_history
    sides_file = tmp_path / "sides.json"
    code, summary, err, out = repair(built, tmp_path, table_report(built), capsys,
                                     "--sides", str(sides_file))
    assert code == 0, err
    sides = json.loads(sides_file.read_text(encoding="utf-8"))
    assert sides["form"] == v.FORM_SIDES
    assert set(sides["repaired"]) == {"R1", "R2"} and sides["reference"] == {}
    repaired_r1 = sides_file.parent / sides["repaired"]["R1"]
    assert repaired_r1.is_file() and len(lines_of(repaired_r1)) == 21


def test_a_record_that_no_manifest_names_is_refused_with_one_line(copy_of_history, tmp_path, capsys):
    built = copy_of_history
    report = table_report(built)
    report["records"].append({"artifact": "R1", "record": "R1:L999", "action": "retire"})
    code, summary, err, out = repair(built, tmp_path, report, capsys)
    assert (code, summary) == (1, None)
    assert len(err.strip().splitlines()) == 1


def test_a_report_of_another_form_is_refused(copy_of_history, tmp_path, capsys):
    built = copy_of_history
    report = table_report(built)
    report["form"] = v.FORM_SIDES
    code, summary, err, out = repair(built, tmp_path, report, capsys)
    assert (code, summary) == (1, None)
    assert len(err.strip().splitlines()) == 1
