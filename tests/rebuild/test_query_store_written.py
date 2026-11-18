"""A command whose store is written by another connection during the walk ends with
the named error of the form (docs/interfaces.md section 2 "The store was written
during the query", docs/cli.md "The error 'store written during the query'").

The error is a refusal of the family "request refused": its name is "store written
during the query", the exit code is 2, the JSON error goes to standard error, and
nothing goes to standard output. It replaces the ``ValueError`` a session raises
today. Every command that reads through a coverage session ends this way: `query`
(an answer and a listing), `report` and `verify`.

The write is made through a second connection to the store file, from a wrapper of
the session's walk that fires once, at the first call of the walk, after the
session has taken its mark of the store.
"""

from __future__ import annotations

import io
import json
import sqlite3

import pytest
from waterwhite import cli, coverage, errors, requests
from waterwhite import vocabulary as v

import rb_kit
from rb_kit import hand_report, write_json

NAME = "store written during the query"


def arm(monkeypatch, path) -> list[int]:
    """Make the first call of the walk write to the store through another connection."""
    fired: list[int] = []

    def write_once() -> None:
        if fired:
            return
        fired.append(1)
        side = sqlite3.connect(path)
        try:
            side.execute("CREATE TABLE IF NOT EXISTS written_during_the_walk (x INTEGER)")
            side.commit()
        finally:
            side.close()

    for method in ("occurrence", "outcome"):
        original = getattr(coverage.Session, method)

        def wrapped(self, *args, _original=original, **kwargs):
            write_once()
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(coverage.Session, method, wrapped)
    return fired


def invoke(world, *argv):
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    raised = None
    code = None
    try:
        code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    except BaseException as exc:  # what the command let out is part of what is asserted
        raised = exc
    return code, out.getvalue(), err.getvalue(), raised


def commands(world, tmp_path):
    report = write_json(tmp_path / "report.json", hand_report(
        world, [("R1", "rewrite", "successor of R1")]))
    return {
        "query an artifact": ("query", "withdraw", "A", "--artifact", "R1"),
        "query listing": ("query", "withdraw", "A"),
        "report": ("report", "withdraw", "A"),
        "verify": ("verify", "--report", str(report)),
    }


@pytest.mark.parametrize("which", ["query an artifact", "query listing", "report", "verify"])
def test_a_store_written_during_the_walk_ends_the_command_with_the_named_error(
        which, tmp_path, monkeypatch):
    world = rb_kit.build(tmp_path / "w", successors=True)
    argv = commands(world, tmp_path)[which]
    fired = arm(monkeypatch, world.path)
    code, out, err, raised = invoke(world, *argv)
    assert fired, f"{which}: the walk made no call, so no write was made during it"
    assert raised is None, f"{which}: the command let {raised!r} out"
    assert out == "", f"{which}: standard output holds {out!r}"
    assert err.strip(), f"{which}: nothing was written to standard error"
    error = json.loads(err)
    assert error["error"] == NAME == v.ERROR_STORE_WRITTEN_DURING_QUERY
    assert isinstance(error["detail"], str) and error["detail"]
    assert code == 2


def test_the_error_is_a_class_of_the_errors_module_in_the_family_request_refused():
    found = getattr(errors, "StoreWrittenDuringQuery", None)
    assert found is not None, "waterwhite.errors has no StoreWrittenDuringQuery"
    assert issubclass(found, errors.Refused)
    assert (found.name, found.exit_code) == (NAME, 2)


def test_a_session_whose_store_is_written_refuses_with_the_named_error_and_not_a_value_error(tmp_path):
    found = getattr(errors, "StoreWrittenDuringQuery", None)
    assert found is not None, "waterwhite.errors has no StoreWrittenDuringQuery"
    world = rb_kit.build(tmp_path / "w")
    try:
        request = requests.withdraw(world.store, "A")
        session = coverage.Session(world.store, request)
        side = sqlite3.connect(world.path)
        try:
            side.execute("CREATE TABLE IF NOT EXISTS written_after_the_session (x INTEGER)")
            side.commit()
        finally:
            side.close()
        with pytest.raises(found):
            coverage.prove(world.store, "R1", request, session=session)
    finally:
        world.store.close()
