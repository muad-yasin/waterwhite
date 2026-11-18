"""``python -m replica repair`` reads nothing of the report's ``upstream_files`` (plan 4.1
with its ledger row BF-6; docs/interfaces.md sections 7.1 and 9.5).

The list names files upstream of the report's artifacts for the person reading it. The
repair reads the records and the artifacts and no other key, so the same report with the
list as the tool prints it, with junk under the key and with the key left out gives the
same summary, the same sides file and the same repaired files.
"""

from __future__ import annotations

import io
import json
import shutil

import pytest
from waterwhite import cli as tool_cli

from replica import cli
from replica.fixture import build_captured_fixture, default_spec_path


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """The historical build of the fixture, captured into a store, without tokenization."""
    folder = tmp_path_factory.mktemp("repair-upstream-historical") / "history"
    return build_captured_fixture(folder, releases=("R1", "R2"), imports=(), layouts=())


def copy_of(historical, where):
    shutil.copytree(historical.out.parent, where)
    return where / historical.store.name, where / historical.out.name


def printed_report(historical, tmp_path) -> dict:
    """The report of "withdraw A" as the tool prints it, asked of a copy of the store."""
    store, _ = copy_of(historical, tmp_path / "asked")
    out, err = io.StringIO(), io.StringIO()
    try:
        code = tool_cli.main(["report", "withdraw", "A", "--store", str(store)],
                             stdout=out, stderr=err)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    assert code == 0, err.getvalue()
    return json.loads(out.getvalue())


def tree_of(folder) -> dict:
    return dict((path.relative_to(folder).as_posix(), path.read_bytes())
                for path in sorted(folder.rglob("*")) if path.is_file())


def repair_with(historical, tmp_path, name, report, capsys) -> dict:
    """Run the repair on a fresh copy of the history with ``report``; what it printed,
    its exit code, the sides file and every file it wrote, with the folder's own name
    taken out of the text."""
    store, out_of_history = copy_of(historical, tmp_path / name / "history")
    report_file = tmp_path / name / "report.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")
    out, sides = tmp_path / name / "repaired", tmp_path / name / "sides.json"
    argv = ["repair", "--report", str(report_file), "--historical", str(out_of_history),
            "--out", str(out), "--store", str(store), "--spec", str(default_spec_path()),
            "--sides", str(sides)]
    try:
        code = cli.main(argv)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    captured = capsys.readouterr()
    base = str(tmp_path / name)
    return dict(
        code=code,
        stdout=captured.out.replace(base, "<folder>"),
        stderr=captured.err.replace(base, "<folder>"),
        sides=sides.read_text(encoding="utf-8").replace(base, "<folder>") if sides.exists() else None,
        files=tree_of(out) if out.exists() else None,
    )


def test_the_replica_reads_nothing_of_upstream_files(historical, tmp_path, capsys):
    printed = printed_report(historical, tmp_path)
    # The report as the tool prints it holds the key; the repair reads the keys it needs
    # and no other.
    assert "upstream_files" in printed, sorted(printed)
    junk = dict(printed, upstream_files=[dict(bogus=1), "not even an object"])
    without = dict((key, value) for key, value in printed.items() if key != "upstream_files")
    results = [repair_with(historical, tmp_path, name, report, capsys)
               for name, report in (("printed", printed), ("junk", junk), ("without", without))]
    assert results[0]["code"] == 0, results[0]["stderr"]
    assert results[0]["files"], "the repair wrote no file"
    assert results[1] == results[0]
    assert results[2] == results[0]
