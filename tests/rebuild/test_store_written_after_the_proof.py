"""A write by another connection in the windows that ``test_query_store_written.py``
does not reach (docs/interfaces.md section 2; docs/cli.md "The error 'store written
during the query'"):

- after ``prove`` returned a proof and before the answer is built from it: the
  command ends with the named error, exit code 2, the JSON error on standard error and
  nothing on standard output;
- after the last answer of the command is built and before anything is printed: the
  same.

The first window is made by wrapping ``coverage.prove`` so that the write follows the
first call that returns a proof; the walk before it has finished its checks, so the
session's own next check is not what finds the write. Every command reaches
``coverage.prove`` through ``coverage.answer``. The second window is made by wrapping
``coverage.answer`` so that the write follows the last call of it that the command
makes, whose number a first run of the same command counts: the answer has been built
and returned, and nothing has been printed.

``invoke`` catches everything the command lets out, so another refusal, or a crash,
is an assertion that fails with what came out.
"""

from __future__ import annotations

import io
import json
import sqlite3

import pytest
from waterwhite import cli, coverage
from waterwhite.answers import Proof

import rb_kit
from rb_kit import hand_report, write_json

NAME = "store written during the query"


def write_by_another_connection(path) -> None:
    side = sqlite3.connect(path)
    try:
        side.execute("CREATE TABLE IF NOT EXISTS written_in_the_window (x INTEGER)")
        side.commit()
    finally:
        side.close()


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
        "query an artifact": ("query", "withdraw", "A", "--artifact", "R2"),
        "query listing": ("query", "withdraw", "A"),
        "report": ("report", "withdraw", "A"),
        "verify": ("verify", "--report", str(report)),
    }


WHICH = ["query an artifact", "query listing", "report", "verify"]


def assert_refused_by_name(which, fired, code, out, err, raised) -> None:
    assert fired, f"{which}: the command made no call in the window, so no write was made in it"
    assert raised is None, f"{which}: the command let {raised!r} out"
    assert out == "", f"{which}: standard output holds {out!r}"
    assert err.strip(), f"{which}: nothing was written to standard error; the command answered"
    error = json.loads(err)
    assert error["error"] == NAME, f"{which}: the error is {error['error']!r}"
    assert isinstance(error["detail"], str) and error["detail"]
    assert code == 2, f"{which}: the exit code is {code}"


@pytest.mark.parametrize("which", WHICH)
def test_a_write_after_the_proof_and_before_the_answer_ends_the_command_with_the_named_error(
        which, tmp_path, monkeypatch):
    world = rb_kit.build(tmp_path / "w", successors=True)
    try:
        argv = commands(world, tmp_path)[which]
        fired: list[int] = []
        original = coverage.prove

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, Proof) and not fired:
                fired.append(1)
                write_by_another_connection(world.path)
            return result

        monkeypatch.setattr(coverage, "prove", wrapped)
        code, out, err, raised = invoke(world, *argv)
        assert_refused_by_name(which, fired, code, out, err, raised)
    finally:
        world.store.close()


# `report` is not in this list: it ends with a check of its own session after its last
# answer, so a write in this window is already refused by name.
AFTER_THE_LAST_ANSWER = ["query an artifact", "query listing", "verify"]


@pytest.mark.parametrize("which", AFTER_THE_LAST_ANSWER)
def test_a_write_after_the_last_answer_and_before_the_output_ends_the_command_with_the_named_error(
        which, tmp_path, monkeypatch):
    world = rb_kit.build(tmp_path / "w", successors=True)
    try:
        argv = commands(world, tmp_path)[which]
        built: list[int] = []
        target: list[int] = []
        fired: list[int] = []
        original = coverage.answer

        def counted(*args, **kwargs):
            made = original(*args, **kwargs)
            built.append(1)
            if target and len(built) == target[0]:
                fired.append(1)
                write_by_another_connection(world.path)
            return made

        monkeypatch.setattr(coverage, "answer", counted)
        first = invoke(world, *argv)  # the same command with no write: how many answers it builds
        assert first[3] is None and first[0] == 0, f"{which}: the command failed with no write: {first}"
        assert built, f"{which}: the command built no answer"
        target.append(len(built))
        built.clear()
        code, out, err, raised = invoke(world, *argv)
        assert_refused_by_name(which, fired, code, out, err, raised)
    finally:
        world.store.close()
