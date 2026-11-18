"""What ``verify`` will certify a repair for (plan 4.2, 4.4; docs/interfaces.md section
7.3): never for a scope narrower than the report's, and never for artifacts the store it
is handed does not hold. It is not keyed on which store file it is handed: a report
printed from one store and checked on a faithful copy where the successors have been
registered is certified, as in the acceptance world.

A refusal is a named error of the refusal family: exit 2, nothing on standard output.

Where the code already gives the result, the test is green by design; those are the
two that hand the store a request wider than it expands (a member or a record the store
does not hold, refused when the request of the report is put to the store again), and
the copy that must stay certified.
"""

from __future__ import annotations

import json
import shutil

import rb_kit
from rb_kit import ww, hand_report, run, write_json
from waterwhite import vocabulary as v
from waterwhite.store.store import Store


def error_of(err: str) -> dict:
    assert err.strip(), "nothing was written to standard error"
    return json.loads(err)


def refused(world, report: dict, tmp_path, label: str) -> None:
    path = write_json(tmp_path / f"report-{label}.json", report)
    code, out, err = run(world, "verify", "--report", str(path))
    assert code == 2, f"{label}: exit {code}, standard output {out[:200]!r}"
    assert out == "", label
    assert isinstance(error_of(err)["error"], str) and error_of(err)["error"], label


# --- A scope narrower than the report's ------------------------------------------------------


def test_a_report_for_more_members_than_the_store_expands_to_is_refused(repair_world, tmp_path):
    # Green by design: the request of the report is put to the store again and a member
    # the store does not hold is refused ("member not registered").
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    report["request"]["members"] = ["A/r1", "A/r2"]
    refused(repair_world, report, tmp_path, "members")


def test_a_report_for_a_record_the_store_does_not_hold_is_refused(repair_world, tmp_path):
    # Green by design: "record not in source".
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")])
    report["request"]["records"] = ["src/A:L1", "src/A:L9"]
    refused(repair_world, report, tmp_path, "records")


# --- An artifact the store does not hold ---------------------------------------------------------


def test_a_report_that_names_an_instance_the_store_does_not_hold_is_not_certified(repair_world, tmp_path):
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1"), ("R2", "keep", None)])
    report["artifacts"][1]["instance"] = "0" * 64
    refused(repair_world, report, tmp_path, "instance")


def test_a_report_printed_from_another_store_that_holds_more_is_not_certified(world, tmp_path):
    other = rb_kit.build(tmp_path / "other", swap=True)
    try:
        code, out, err = run(other, "report", "withdraw", "A")
        assert code == 0, err
        printed = json.loads(out)
    finally:
        other.store.close()
    assert "R1-swap" in [entry["artifact"] for entry in printed["artifacts"]]
    refused(world, printed, tmp_path, "other store")


# --- A faithful copy with the successors registered is certified (green by design) ------------------


def test_a_report_printed_from_a_store_is_certified_on_its_copy_with_the_successors_registered(
        world, tmp_path):
    code, out, err = run(world, "report", "withdraw", "A")
    assert code == 0, err
    path = write_json(tmp_path / "report.json", json.loads(out))
    world.store.close()
    shutil.copytree(world.folder, tmp_path / "copy")
    copy = ww.World(tmp_path / "copy", Store.open(tmp_path / "copy" / "store.db"), dict(world.ids))
    try:
        rb_kit.add_successors(copy, tmp_path / "copy")
        code, out, err = run(copy, "verify", "--report", str(path))
        assert code == 0, err
        document = json.loads(out)
        assert document["form"] == v.FORM_VERIFICATION
        assert [entry["registered"] for entry in document["successors"]] == [True, True, True]
    finally:
        copy.store.close()
        world.store = Store.open(world.path)
