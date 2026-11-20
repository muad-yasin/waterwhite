"""The ``verify`` command (docs/cli.md "verify"; docs/interfaces.md sections 7.2 and
7.3): the JSON document on standard output, a difference as an answer with exit 0,
the refusals with their exit codes, and that nothing is written.
"""

from __future__ import annotations

import json
import sqlite3

from waterwhite import vocabulary as v

from rb_kit import hand_report, run, sides_document, write_json, write_lines

KEYS = ["form", "request", "mode", "unresolved_records", "multisets", "differing_records",
        "differences", "releases", "per_shard_digests", "shards", "kept", "successors", "history",
        "caches", "restarts"]
CLEAN = ["passage X", "z one", "z two"]


def error_of(err: str) -> dict:
    assert err.strip(), "nothing was written to standard error"
    return json.loads(err)


def files(world, tmp_path, repaired=CLEAN, reference=CLEAN):
    report = hand_report(world, [("R1", "rewrite", "successor of R1")])
    write_lines(tmp_path / "repaired/R1.jsonl", repaired)
    write_lines(tmp_path / "reference/R1.jsonl", reference)
    sides = sides_document({"R1": str(tmp_path / "repaired/R1.jsonl")},
                           {"R1": str(tmp_path / "reference/R1.jsonl")})
    return write_json(tmp_path / "report.json", report), write_json(tmp_path / "sides.json", sides)


def test_verify_prints_the_document_of_the_form(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides))
    assert code == 0, err
    document = json.loads(out)
    assert list(document) == KEYS
    assert document["form"] == v.FORM_VERIFICATION and document["mode"] == "unseeded"
    assert document["multisets"] == "equal" and document["differing_records"] == 0
    assert [e["answer"] for e in document["successors"]] == ["does not contain"]
    assert [e["answer"] for e in document["history"]] == ["contains"]


def test_a_difference_between_the_sides_is_an_answer_and_exits_with_0(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path, repaired=CLEAN + ["paraphrase of X"])
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--mode", "seeded")
    assert code == 0, err
    document = json.loads(out)
    assert (document["mode"], document["multisets"], document["differing_records"]) == (
        "seeded", "differ", 1)


def test_the_text_field_is_passed_on(repair_world, tmp_path):
    report, _ = files(repair_world, tmp_path)
    write_lines(tmp_path / "repaired/R1.jsonl", CLEAN, field="body")
    write_lines(tmp_path / "reference/R1.jsonl", CLEAN, field="body")
    sides = tmp_path / "sides.json"
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides),
                         "--text-field", "body")
    assert code == 0, err
    assert json.loads(out)["multisets"] == "equal"


def test_a_report_that_is_not_a_report_is_refused_with_exit_2(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides))
    assert code == 0, err
    # A sides file handed as the report is JSON of another form.
    code, out, err = run(repair_world, "verify", "--report", str(sides))
    assert (code, out, error_of(err)["error"]) == (2, "", "request refused")


def test_a_report_that_lacks_a_key_verify_reads_is_refused_with_exit_2(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report))
    assert code == 0, err
    held = json.loads(report.read_text(encoding="utf-8"))
    cases = []
    for key in ("request", "artifacts", "counts"):
        broken = json.loads(json.dumps(held))
        del broken[key]
        cases.append(broken)
    broken = json.loads(json.dumps(held))
    del broken["artifacts"][0]["action"]
    cases.append(broken)
    for number, broken in enumerate(cases):
        path = write_json(tmp_path / f"broken-{number}.json", broken)
        code, out, err = run(repair_world, "verify", "--report", str(path))
        assert (code, out, error_of(err)["error"]) == (2, "", "request refused"), number


def test_a_sides_path_that_is_not_text_is_refused_naming_the_side_and_the_artifact(repair_world, tmp_path):
    """Sol's review of a011cd5: a number as a path in an otherwise valid sides file
    raised an uncaught TypeError. A path is text; anything else is a refusal of the
    document, with exit 2, naming the side and the artifact."""
    report, sides = files(repair_world, tmp_path)
    held = json.loads(sides.read_text(encoding="utf-8"))
    for which in ("repaired", "reference"):
        for bad in (7, None, ["a"], {"path": "a"}, True):
            broken = json.loads(json.dumps(held))
            broken[which]["R1"] = bad
            path = write_json(tmp_path / f"neutral-{abs(hash(json.dumps(bad)))}.json", broken)
            code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(path))
            error = error_of(err)
            assert (code, out, error["error"]) == (2, "", "request refused"), (which, bad)
            # The side and the artifact are in the detail and in the context, and the file
            # is named neutrally so that its path cannot stand for either.
            assert f"names R1 on its {which} side" in error["detail"], (which, bad, error)
            assert (error["context"]["side"], error["context"]["artifact"]) == (which, "R1")


def test_a_report_or_sides_file_that_cannot_be_read_is_a_usage_error(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path)
    code, out, err = run(repair_world, "verify", "--report", str(report))
    assert code == 0, err
    code, out, err = run(repair_world, "verify", "--report", str(tmp_path / "absent.json"))
    assert (code, out, error_of(err)["error"]) == (1, "", "usage")
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides",
                         str(tmp_path / "absent.json"))
    assert (code, out, error_of(err)["error"]) == (1, "", "usage")
    code, out, err = run(repair_world, "verify", "--report", str(report), "--mode", "sometimes")
    assert (code, out, error_of(err)["error"]) == (1, "", "usage")


def test_the_request_of_the_report_is_put_to_the_store_and_refused_like_a_query(repair_world, tmp_path):
    report = hand_report(repair_world, [("R1", "rewrite", "successor of R1")], source="Q")
    path = write_json(tmp_path / "report.json", report)
    code, out, err = run(repair_world, "verify", "--report", str(path))
    assert (code, out, error_of(err)["error"]) == (2, "", "source not registered")


def dump(path) -> list[str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def test_verify_writes_nothing(repair_world, tmp_path):
    report, sides = files(repair_world, tmp_path)
    repair_world.store.close()
    before = dump(repair_world.path)
    code, out, err = run(repair_world, "verify", "--report", str(report), "--sides", str(sides))
    assert code == 0, err
    repair_world.store.close()
    assert dump(repair_world.path) == before

