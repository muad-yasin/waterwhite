"""A switch that cannot be read is named before an argument value is interpreted or an input
file is read (the owner's Q8 of 18 Nov 2026, "checked first, everything stops"; D-CORE-6).

``report`` read its ``--layout`` and ``--placement`` values and ``verify`` read its two documents
before the store was opened, so a store with a switch that holds "ON" gave a usage error or a
report error and not the named switch refusal. The store is now opened first (it must be opened
to read the switch), the switch is checked, and only then is an argument interpreted or a file
read. What the parser refuses (a missing value, an unknown option) is still refused first.
Nothing changes for a switch that can be read, except which refusal comes first when the store
cannot be opened at all (the store is named, not the argument).
"""

from __future__ import annotations

import io
import json

import pytest
import ww_world

from waterwhite import cli, errors, extensions
from waterwhite import vocabulary as v
from waterwhite.rebuild import verify_sides
from waterwhite.store.store import Store


def break_switch(world, extension: str = extensions.CONTENT_BAN, value: str = "ON") -> None:
    key = extensions.setting_key(extension)
    world.sql(("INSERT INTO setting (key, value) VALUES (?, ?) "
               "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value)))


def run(world, *argv: str) -> tuple[int, str, str]:
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


def named_switch(code: int, out: str, err: str) -> bool:
    return (code, out) == (errors.EXIT_REFUSED, "") and "holds 'ON'" in err


@pytest.fixture
def world(tmp_path):
    built = ww_world.build(tmp_path)
    yield built
    built.store.close()


def test_a_broken_switch_is_named_before_a_layout_is_read(world):
    break_switch(world)
    assert named_switch(*run(world, "report", "withdraw", "A", "--layout", "BAD"))


def test_a_broken_switch_is_named_before_a_placement_file_is_opened(world):
    break_switch(world)
    missing = world.path.parent / "no-such-placement.json"
    assert named_switch(*run(world, "report", "withdraw", "A", "--layout", "x=y", "--placement", f"x={missing}"))


def test_a_broken_switch_is_named_before_the_report_or_sides_file_is_read(world):
    break_switch(world)
    garbage = world.path.parent / "garbage.json"
    garbage.write_text("not json")
    assert named_switch(*run(world, "verify", "--report", str(garbage)))
    assert named_switch(*run(world, "verify", "--report", str(world.path.parent / "no-such-report.json")))
    assert named_switch(*run(world, "verify", "--report", str(garbage), "--sides", str(garbage)))


def test_a_broken_switch_is_named_before_the_shape_or_the_request_kind_of_a_report(world):
    document = world.path.parent / "ban-report.json"
    document.write_text(json.dumps({"form": v.FORM_REPORT, "artifacts": [], "counts": {"records": {"unresolved": 0}},
                                    "request": {"kind": "ban", "text": "x", "source": "x"}}))
    code, out, err = run(world, "verify", "--report", str(document))  # a switch that can be read: refused as before
    assert (code, out) == (errors.EXIT_REFUSED, "") and "the report is for a request this build does not answer" in err
    break_switch(world)
    assert named_switch(*run(world, "verify", "--report", str(document)))


def test_the_library_verify_names_a_broken_switch_before_anything_it_is_given_is_looked_at(world):
    break_switch(world, extensions.ADAPTERS)
    with pytest.raises(errors.Refused, match="'adapters' holds 'ON'"):
        verify_sides.verify(world.store, {}, mode="no-such-mode")


def test_a_switch_that_can_be_read_changes_no_refusal_of_these_commands(world):
    code, out, err = run(world, "report", "withdraw", "A", "--layout", "BAD")
    assert (code, out) == (errors.EXIT_USAGE, "") and "--layout takes NAME=PREFIX" in err
    garbage = world.path.parent / "garbage.json"
    garbage.write_text("not json")
    code, out, err = run(world, "verify", "--report", str(garbage))
    assert (code, out) == (errors.EXIT_USAGE, "") and "is not JSON" in err


def test_the_parser_still_refuses_a_missing_value_before_the_switch_is_read(world):
    break_switch(world)
    code, out, err = run(world, "report", "withdraw", "A", "--layout")
    assert (code, out) == (errors.EXIT_USAGE, "")


def test_a_store_that_cannot_be_opened_is_named_before_an_argument(world, tmp_path):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["report", "withdraw", "A", "--layout", "BAD", "--store", str(tmp_path / "none.sqlite")],
                    stdout=out, stderr=err)
    assert (code, out.getvalue()) == (errors.EXIT_REFUSED, "") and "no file at" in err.getvalue()
    code = cli.main(["verify", "--report", str(tmp_path / "none.json"), "--store", str(tmp_path / "none.sqlite")],
                    stdout=io.StringIO(), stderr=err)
    assert code == errors.EXIT_REFUSED


def test_a_command_line_with_no_store_names_the_missing_store_before_an_argument(world):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["report", "withdraw", "A", "--layout", "BAD"], stdout=out, stderr=err)
    assert (code, out.getvalue()) == (errors.EXIT_USAGE, "") and "the command needs --store FILE" in err.getvalue()
