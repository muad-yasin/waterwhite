"""``python -m replica repair`` and a landing that lies in no layout of the build
(DECISIONS.md D-141, D-144; docs/replica.md, the repair section; docs/interfaces.md 7.1
and 9.5).

A report may name, under ``groups.placements[NAME].landing``, a shard the store does not
hold, which the copies that return are placed in. Under the repair a landing shard is a
shard the report names, so it is held to what every shard the report names is held to: it
must lie in a layout of the build. A landing under a prefix that the build has no layout
for is refused by the refusal of a shard that lies in no layout, in one line that names
the shard, and the refusal comes before either side is built: nothing is registered in
the store and no folder is written.

The report is edited by hand in the manner of ``test_replica_repair_keep.py``: its
artifacts are a rewritten release and a kept one, which no layout is asked about, so the
only name that lies outside every layout is the landing's. The historical build is the
hand-checked fixture, captured, with no layout.
"""

from __future__ import annotations

import json
import shutil

import pytest
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica import cli
from replica.fixture import build_captured_fixture, default_spec_path

LANDING = "nowhere/shard-9"


def a_report_with_a_landing(shard: str) -> dict:
    """A report whose artifacts name no shard, and whose one layout's placement lands the
    copy of ``src/B:L1`` in ``shard``, a shard the store does not hold."""
    return {
        "form": v.FORM_REPORT,
        "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
        "records": [],
        "artifacts": [
            {"artifact": "R1", "kind": "jsonl", "action": "rewrite", "successor": "successor of R1"},
            {"artifact": "R2", "kind": "jsonl", "action": "keep", "successor": None},
        ],
        "groups": {"placements": {"x": {
            "placement": {"path": "placement.json", "sha256": "0" * 64},
            "landing": [{"shard": shard, "in_store": False, "receives": ["src/B:L1"]}],
            "place_not_stated": [],
        }}},
    }


def run(argv, capsys):
    try:
        code = cli.main(argv)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    folder = tmp_path_factory.mktemp("repair-landing-historical") / "history"
    return build_captured_fixture(folder, releases=("R1", "R2"), imports=(), layouts=())


def test_a_landing_that_lies_in_no_layout_is_refused_before_either_build_is_made(
        historical, tmp_path, capsys):
    history = tmp_path / "history"
    shutil.copytree(historical.out.parent, history)
    store = history / historical.store.name
    report = tmp_path / "report.json"
    report.write_text(json.dumps(a_report_with_a_landing(LANDING)), encoding="utf-8")
    out, reference, sides = tmp_path / "repaired", tmp_path / "reference", tmp_path / "sides.json"
    code, printed, err = run(
        ["repair", "--report", str(report), "--historical", str(history / historical.out.name),
         "--out", str(out), "--store", str(store), "--spec", str(default_spec_path()),
         "--reference-withdrawn", "A", "--reference-out", str(reference),
         "--sides", str(sides)], capsys)
    assert code != 0, f"the repair of a landing in no layout exited {code}; it printed {printed[:160]!r}"
    assert printed.strip() == "", f"the refused repair printed {printed[:160]!r}"
    assert LANDING in err, f"the refusal does not name the landing shard {LANDING!r}: {err!r}"
    assert not out.exists(), "the repaired side was built"
    assert not reference.exists(), "the reference side was built"
    assert not sides.exists(), "a sides file was written"
    with Store.open(store, readonly=True) as opened:
        assert opened.resolve("successor of R1") is None, "a successor was registered"
