"""The one file of the core result (core_result_file.py): the records of the first run with each gated
long-run record replaced by the record that ran it, written gzip, read back, compared.

Synthetic records only: no store, no tool, no check is run here."""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import core_record as cr
import core_result_file as crf
import pytest

HERE = Path(__file__).resolve().parent
GATE_REASON = "a long run: Check 3 on the 100k set runs only when WATERWHITE_CLEARED_LONG_RUNS names 3:100k"


def rec(row, *, size=None, ended="passed", reason=None, calls=None, core=True, unseeded=False):
    return {"calls": [{"argv": ["tool", row], "exit": 0, "out": row}] if calls is None else calls, "check": 3, "core": core,
            "ended": ended, "evidence": None, "extension": None, "format": cr.FORMAT, "kind": "q", "mismatch": None,
            "part": "p", "reason": reason, "row": row, "size": size, "store": "main", "unseeded": unseeded}


def gated(row, size="100k"):
    return rec(row, size=size, ended="skipped", reason=GATE_REASON, calls=[])


def ran(row, size="100k", out="the long run's answer"):
    return rec(row, size=size, calls=[{"argv": ["tool", row, size], "exit": 0, "out": out}])


def base_records():
    return [rec("c1"), rec("c2", unseeded=True), gated("c3"), gated("c4", "10k"),
            rec("x1", core=False, ended="skipped", reason="extension off: content-ban", calls=[])]


def addenda():
    return {"l10k.jsonl": [rec("c1"), rec("c2", unseeded=True, calls=[{"argv": ["tool", "other"], "exit": 0, "out": "o"}]), ran("c4", "10k")],
            "l100k.jsonl": [rec("c1"), ran("c3"), gated("c4", "10k")]}


def test_the_gated_records_are_replaced_by_the_records_that_ran_them_and_nothing_else_moves():
    records, replaced, left_out = crf.combine(base_records(), addenda())
    by_key = {cr.key_of(r): r for r in records}
    assert [cr.key_of(r) for r in records] == sorted(by_key) and len(records) == 5
    assert replaced == ["c3|p|100k", "c4|p|10k"]
    assert by_key["c3|p|100k"]["ended"] == "passed" and by_key["c3|p|100k"]["calls"][0]["out"] == "the long run's answer"
    assert by_key["c4|p|10k"]["calls"][0]["argv"] == ["tool", "c4", "10k"]
    assert by_key["c1|p|-"] == rec("c1") and by_key["x1|p|-"]["ended"] == "skipped"
    # c2's unseeded record of the base is kept (its calls differ in the addendum: the guard view does not read them)
    assert by_key["c2|p|-"]["calls"][0]["argv"] == ["tool", "c2"]
    assert left_out == 4  # c1 twice, c2 once, and the skipped gate of c4 in the 100k run
    assert left_out + len(replaced) == sum(len(v) for v in addenda().values())


@pytest.mark.parametrize("what", ["gate_not_run", "gate_run_twice", "key_not_in_base", "duplicate_differs"])
def test_it_refuses_where_it_would_have_to_choose(what):
    base, more = base_records(), addenda()
    if what == "gate_not_run":
        more["l10k.jsonl"] = [r for r in more["l10k.jsonl"] if r["row"] != "c4"]
    elif what == "gate_run_twice":
        more["l100k.jsonl"].append(ran("c4", "10k", out="another"))
    elif what == "key_not_in_base":
        more["l100k.jsonl"].append(rec("zz"))
    elif what == "duplicate_differs":
        more["l100k.jsonl"][0] = rec("c1", ended="failed")
    with pytest.raises(ValueError):
        crf.combine(base, more)


def test_the_gzip_file_is_reproducible_and_carries_no_name_and_no_time(tmp_path):
    records, replaced, left_out = crf.combine(base_records(), addenda())
    header = {"tree": "1c132e32", "replaced_keys": replaced}
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    sha_a, sha_b = crf.write_gzip(records, header, a), crf.write_gzip(records, header, b)
    assert a.read_bytes() == b.read_bytes() and sha_a == sha_b
    raw = a.read_bytes()
    assert raw[:2] == b"\x1f\x8b" and raw[3] == 0 and raw[4:8] == b"\x00\x00\x00\x00"  # no flags (no name), no time
    assert cr.digest(gzip.decompress(raw).decode("ascii")) == sha_a


def test_a_gzip_file_and_a_plain_file_read_alike_and_the_header_must_match(tmp_path):
    records, replaced, _ = crf.combine(base_records(), addenda())
    gz, plain = tmp_path / "x.gz", tmp_path / "x.jsonl"
    crf.write_gzip(records, {"tree": "t"}, gz)
    plain.write_text(crf.text_of(records, {"tree": "t"}), encoding="ascii")
    assert crf.read_any(gz) == crf.read_any(plain) == records
    plain.write_text(plain.read_text(encoding="ascii").replace('"records":5', '"records":6'), encoding="ascii")
    with pytest.raises(ValueError):
        crf.read_any(plain)


def run_diff(a: Path, b: Path):
    return subprocess.run([sys.executable, "-I", str(HERE / "core_result_file.py"), "diff", str(a), str(b)], capture_output=True, text=True)


def committed(tmp_path, records=None, replaced=None):
    records, replaced_keys, _ = crf.combine(base_records(), addenda()) if records is None else (records, replaced, 0)
    path = tmp_path / "committed.gz"
    crf.write_gzip(records, {"tree": "t", "replaced_keys": replaced_keys}, path)
    return path, records


def test_the_real_diff_command_says_full_only_for_a_full_run_and_lists_the_extension_rows(tmp_path):
    one, records = committed(tmp_path)
    two = tmp_path / "two.gz"
    crf.write_gzip(records, {"tree": "other tree"}, two)  # a header differing is not a difference of rows
    done = run_diff(one, two)
    assert done.returncode == 0
    assert done.stdout.splitlines()[0] == "FULL: no difference in the core rows (4 compared, 0 not run here)"
    assert done.stdout.splitlines()[1] == "extension rows not compared (core = false): x1|p|-"


def test_the_real_diff_command_names_a_record_changed_inside_the_gzip_file(tmp_path):
    one, records = committed(tmp_path)
    two = tmp_path / "two.gz"
    changed = json.loads(json.dumps(records))
    for r in changed:
        if r["row"] == "c3":
            r["calls"][0]["out"] = "a different answer"
    crf.write_gzip(changed, {"tree": "t"}, two)
    done = run_diff(one, two)
    assert done.returncode == 1 and "c3|p|100k: differs in calls" in done.stdout.splitlines()
    assert "DIFFERENCES: 1 of the core keys" in done.stdout.splitlines()


def test_a_partial_run_never_prints_the_full_verdict_and_names_the_rows_not_run(tmp_path):
    one, records = committed(tmp_path)
    two = tmp_path / "two.gz"
    short = [gated("c3") if r["row"] == "c3" else gated("c4", "10k") if r["row"] == "c4" else r for r in records]
    crf.write_gzip(short, {"tree": "t"}, two)
    done = run_diff(one, two)
    assert done.returncode == 3 and "FULL" not in done.stdout
    assert "PARTIAL, not the full verdict: 2 core rows compared, 2 not run here:" in done.stdout
    assert "  not run: c3|p|100k" in done.stdout and "  not run: c4|p|10k" in done.stdout


@pytest.mark.parametrize("what", ["other_reason", "key_not_in_the_list", "core_key_missing_second", "core_key_missing_first"])
def test_only_the_named_gates_with_their_own_reason_are_not_run_here(tmp_path, what):
    one, records = committed(tmp_path)
    two = tmp_path / "two.gz"
    later = json.loads(json.dumps(records))
    if what == "other_reason":
        later = [rec("c3", size="100k", ended="skipped", reason="no reason of a gate", calls=[]) if r["row"] == "c3" else r for r in later]
    elif what == "key_not_in_the_list":
        later = [gated("c1", None) if r["row"] == "c1" else r for r in later]
    elif what == "core_key_missing_second":
        later = [r for r in later if r["row"] != "c1"]
    else:
        records_wo = [r for r in records if r["row"] != "c1"]
        one, _ = committed(tmp_path, records_wo, ["c3|p|100k", "c4|p|10k"])
    crf.write_gzip(later, {"tree": "t"}, two)
    done = run_diff(one, two)
    assert done.returncode == 1 and "FULL" not in done.stdout and "PARTIAL" not in done.stdout, done.stdout


def test_the_combine_command_writes_the_file_and_names_what_it_did(tmp_path):
    for name, records in {"base.jsonl": base_records(), **addenda()}.items():
        (tmp_path / name).write_text(crf.text_of(records, {}), encoding="ascii")
    out = tmp_path / "core_result.jsonl.gz"
    done = subprocess.run([sys.executable, "-I", str(HERE / "core_result_file.py"), "combine", str(out), '{"tree":"1c132e32"}',
                           str(tmp_path / "base.jsonl"), str(tmp_path / "l10k.jsonl"), str(tmp_path / "l100k.jsonl")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    head = crf.header_of(out)
    assert head["tree"] == "1c132e32" and head["replaced_keys"] == ["c3|p|100k", "c4|p|10k"]
    assert set(head["sources_sha256"]) == {"base.jsonl", "l10k.jsonl", "l100k.jsonl"}
    assert all(r["ended"] != "skipped" or not r["core"] for r in crf.read_any(out))
    assert cr.lacking(crf.read_any(out)) == {}
