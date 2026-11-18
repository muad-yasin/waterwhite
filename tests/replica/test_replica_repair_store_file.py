"""``python -m replica repair`` with a ``--store`` that is a file and is not a store
(docs/interfaces.md section 9.5).

A path that holds no file is the replica's own input error (``test_replica_repair_store.py``).
A path that holds a file which is not a store is the same error, in the same one line,
and the repair changes nothing: the file keeps its bytes, and no output folder, sides
file or sibling file appears. What the tool says when it opens such a file is its own
error, which the replica's command line does not print as its own.

The report is read by the replica as a file and nothing else of it matters here.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from waterwhite import vocabulary as v

from replica import cli
from replica.fixture import build_captured_fixture, default_spec_path


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """The historical build of the fixture, captured into a store, without tokenization."""
    folder = tmp_path_factory.mktemp("repair-store-file-historical") / "history"
    return build_captured_fixture(folder, releases=("R1", "R2"), imports=(), layouts=())


def a_report() -> dict:
    return {
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
        "records": [],
        "artifacts": [
            {"artifact": "R1", "action": "rewrite", "successor": "successor of R1"},
            {"artifact": "R2", "action": "keep", "successor": None},
        ],
    }


def arbitrary_bytes(path):
    path.write_bytes(b"this is a note, not a store\n" * 40)


def foreign_database(path):
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, body TEXT)")
        connection.execute("INSERT INTO notes (body) VALUES ('not a store')")
        connection.commit()
    finally:
        connection.close()


def repair_into(built, folder, store, capsys):
    report = folder / "report.json"
    report.write_text(json.dumps(a_report()), encoding="utf-8")
    out, sides = folder / "repaired", folder / "sides.json"
    argv = ["repair", "--report", str(report), "--historical", str(built.out), "--out", str(out),
            "--store", str(store), "--spec", str(default_spec_path()), "--sides", str(sides)]
    try:
        code = cli.main(argv)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    except Exception as error:  # an error that is not the replica's own is a failure to assert on
        code = f"{type(error).__name__}: {error}"
    captured = capsys.readouterr()
    return code, captured.out, captured.err, out, sides


@pytest.mark.parametrize("make", [arbitrary_bytes, foreign_database],
                         ids=["arbitrary-bytes", "foreign-database"])
def test_a_file_that_is_not_a_store_is_refused_in_one_line_and_nothing_changes(
        historical, tmp_path, capsys, make):
    store = tmp_path / "notes.db"
    make(store)
    before = store.read_bytes()
    code, out, err, repaired, sides = repair_into(historical, tmp_path, store, capsys)
    assert code == 1, (code, err)
    assert out.strip() == ""
    lines = err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("replica: InputError:"), err
    assert store.read_bytes() == before, "the file was changed"
    assert not repaired.exists(), "the output folder was written"
    assert not sides.exists(), "the sides file was written"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["notes.db", "report.json"], (
        "a file or folder was created beside the store")
