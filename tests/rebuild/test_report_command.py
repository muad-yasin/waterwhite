"""The ``report`` command (docs/cli.md "report"; docs/interfaces.md sections 2 and 7):
the JSON document on standard output, the refusals with their exit codes and nothing
on standard output after an error, the loud checks, and that nothing is written.
"""

from __future__ import annotations

import json
import sqlite3

import pytest
from waterwhite import cli
from waterwhite import vocabulary as v

from rb_kit import REQUEST_A, report_parts, run

KEYS = ["form", "request", "records", "artifacts", "upstream_files", "counts", "mixtures", "groups",
        "restarts", "exposed_checkpoints", "checkpoints_not_covered"]


def error_of(err: str) -> dict:
    assert err.strip(), "nothing was written to standard error"
    return json.loads(err)


def test_the_report_has_every_key_of_the_form_in_its_order(world):
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    document = json.loads(out)
    assert list(document) == KEYS
    assert document["form"] == v.FORM_REPORT
    assert document["request"] == REQUEST_A


def test_the_report_of_the_command_is_the_report_of_the_parts(world):
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    printed = json.loads(out)
    parts = report_parts(world)
    assert {k: printed[k] for k in printed if k not in ("restarts", "exposed_checkpoints", "checkpoints_not_covered")} == parts


def test_a_layout_named_on_the_command_line_makes_a_group(world):
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "low=shard-")
    assert code == 0, err
    assert list(json.loads(out)["groups"]["by_layout"]) == ["low"]


def test_a_layout_that_is_not_written_name_equals_prefix_and_a_shard_two_prefixes_match_are_usage_errors(world):
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "low=shard-")
    assert code == 0, err
    for bad in (["--layout", "nonsense"], ["--layout", "a=shard-", "--layout", "b=shard-1"]):
        code, out, err = run(world, "report", "withdraw", "A", *bad)
        assert (code, out, error_of(err)["error"]) == (1, "", "usage"), bad


def test_the_text_form_names_every_artifact_with_its_action_and_successor(world):
    code, out, err = run(world, "report", "withdraw", "A", "--format", "text")
    assert code == 0, err
    for line_part in ("R1", "successor of R1", "successor of shard-3", "successor of mix-2"):
        assert line_part in out
    with pytest.raises(ValueError):
        json.loads(out)


def test_a_request_of_an_extension_that_is_off_is_refused_before_any_report(world):
    for argv in (("report", "ban", "passage X"), ("report", "flagged", "words.txt")):
        code, out, err = run(world, *argv)
        assert (code, out, error_of(err)["error"]) == (2, "", "extension off"), argv


def test_the_refusals_of_query_withdraw_are_the_refusals_of_report_withdraw(world):
    cases = [
        (("Q",), "source not registered"),
        (("A", "--member", "A/r9"), "member not registered"),
        (("A", "--record", "src/A:L9"), "record not in source"),
        (("A", "--selector", "kind=doc"), "unsupported selector"),
    ]
    for argv, name in cases:
        code, out, err = run(world, "report", "withdraw", *argv)
        assert (code, out, error_of(err)["error"]) == (2, "", name), argv


def test_a_loud_fault_stops_the_report_with_exit_3_and_no_output(world):
    world.sql(("UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
               "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
               (world.binding_id("shard-3"),)))
    code, out, err = run(world, "report", "withdraw", "A")
    assert (code, out, error_of(err)["error"]) == (3, "", "map digest mismatch")


def test_a_path_that_holds_no_store_is_refused(tmp_path):
    from io import StringIO

    out, err = StringIO(), StringIO()
    code = cli.main(["report", "withdraw", "A", "--store", str(tmp_path / "none.db")],
                    stdout=out, stderr=err)
    assert (code, out.getvalue(), error_of(err.getvalue())["error"]) == (2, "", "not a store")


def dump(path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def test_the_report_writes_nothing(world):
    world.store.close()
    before = dump(world.path)
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    world.store.close()
    assert dump(world.path) == before
