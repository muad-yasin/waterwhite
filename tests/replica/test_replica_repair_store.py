"""``python -m replica repair`` with a ``--store`` that holds no store (docs/interfaces.md
section 9.5): the historical store is the one the successors are registered in, so a path
that holds none is the replica's own input error, found before anything is built or
created. Opening a store at a path creates one, which is how a mistyped path used to end
in a repair into an empty store.

The report is read by the replica as a file and nothing else of it matters here: the
refusal must come whatever the report holds, and nothing may be written.
"""

from __future__ import annotations

import json

import pytest
from waterwhite import vocabulary as v

from replica import cli
from replica.fixture import build_captured_fixture, default_spec_path


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    """The historical build of the fixture, captured into a store, without tokenization."""
    folder = tmp_path_factory.mktemp("repair-store-historical") / "history"
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


def repair_into(built, tmp_path, store, capsys):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(a_report()), encoding="utf-8")
    out, sides = tmp_path / "repaired", tmp_path / "sides.json"
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


@pytest.mark.parametrize("where", ["none.db", "nowhere/none.db"])
def test_a_store_path_that_holds_no_store_is_refused_and_nothing_is_created(
        historical, tmp_path, capsys, where):
    store = tmp_path / where
    code, out, err, repaired, sides = repair_into(historical, tmp_path, store, capsys)
    assert code == 1, (code, err)
    assert out.strip() == ""
    lines = err.strip().splitlines()
    assert len(lines) == 1 and lines[0].startswith("replica: InputError:"), err
    assert str(store) in lines[0]
    assert not store.exists(), "a store was created at the path"
    assert store.parent == tmp_path or not store.parent.exists(), "a folder was created for the store"
    assert not repaired.exists(), "the output folder was written"
    assert not sides.exists(), "the sides file was written"
