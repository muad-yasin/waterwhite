"""The re-grade command of Check 10 (``baselines.regrade``, 18 Nov 2026): the readings a pass kept are graded again from the
expected-value sources its record kept, without running any tool; a record that stopped, or whose seal does not match, is refused;
a line that rests on an attempt or holds no reading is kept as it is; the record itself is never changed.

The record is written here by hand (one question of the manifest subject, the file family of an import, whose expected value needs
only the world's request entries), sealed as the runner seals it.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


def _mod(name):
    return importlib.import_module(name)


def _question(listed, subject, size=None):
    table10 = _mod("baselines.table10")
    return {q.id: q for q in table10.expand(table10.load_table10())}[table10.question_id(listed, subject, size)]


def _record_folder(tmp_path, *, status="wrong", extra_lines=(), frozen=True):
    runner, record, interfaces = _mod("baselines.runner"), _mod("baselines.record"), _mod("baselines.interfaces")
    folder = tmp_path / "record"
    folder.mkdir()
    runner.keep_expected_sources({"requests": {"withdraw A": {"files": ["sources/A.jsonl"]}}}, folder)
    if not frozen:          # a record of a pass made before the frozen files were kept
        (folder / "frozen-files.json").rename(tmp_path / "moved-frozen-files.json")
    family = _question("c06.import.file-family", "manifest")
    rec = record.Record(question_id=family.id, status=status, raw={"link": {"files": ["sources/A.jsonl"]}}, evidence=(["link"] if status == "wrong" else None),
                        route=family.route, marks=(), excluded_from_tally=family.excluded_from_tally, attempt_id=None)
    lines = [record.answer_line(family, rec, commit="a" * 40, step_ids=[])]
    lines += list(extra_lines)
    record.write_answers(folder / "answers.jsonl", lines)
    (folder / "start.txt").write_text("start: commit " + "a" * 40 + "; subjects manifest; step cap 900.0 seconds;\n", encoding="utf-8")
    (folder / "summary.txt").write_text("Check 10: open\nonce: open\nsmall: open\n10k: open\n100k: open\nplan sizes: open\n", encoding="utf-8")
    runner.seal_run_record(folder)
    return folder


def test_a_kept_reading_is_graded_again_and_the_changed_line_is_named_and_the_record_stays_as_sealed(tmp_path):
    regrade = _mod("baselines.regrade")
    folder = _record_folder(tmp_path, status="wrong")          # the pass graded it wrong: a grader defect a re-grade repairs
    before = (folder / "sha256sums.txt").read_text(encoding="utf-8")
    result = regrade.regrade(folder, tmp_path / "out", commit="b" * 40)
    assert result["changed"] == [("c06.import.file-family|manifest", "wrong", "right")]
    # The changed line says which commit graded it again (its own "commit" stays the pass's); regrade.json names both commits,
    # the seal of the record it read and that the frozen files were checked.
    changed_line = json.loads((tmp_path / "out" / "answers.regraded.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert changed_line["regraded_by_commit"] == "b" * 40 and changed_line["commit"] == "a" * 40
    meta = json.loads((tmp_path / "out" / "regrade.json").read_text(encoding="utf-8"))
    assert meta["grading_commit"] == "b" * 40 and meta["record_commit"] == "a" * 40 and meta["frozen_files"] == "checked against the record"
    assert meta["code_tree_clean"] in (True, False, None), "the output says whether the code was the commit's own (None: no git)"
    assert meta["record_seal_sha256"] == _mod("baselines.runner")._sha256_of(folder / "sha256sums.txt")
    lines = [json.loads(t) for t in (tmp_path / "out" / "answers.regraded.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["status"] for line in lines] == ["right"]
    assert (tmp_path / "out" / "regrade-changes.tsv").read_text(encoding="utf-8").strip().endswith("wrong\tright")
    assert "manifest once right: 1" in (tmp_path / "out" / "regrade-summary.txt").read_text(encoding="utf-8")
    assert regrade.seal_problems(tmp_path / "out") == []
    # The record the lines came from is not changed.
    assert (folder / "sha256sums.txt").read_text(encoding="utf-8") == before and regrade.seal_problems(folder) == []
    # A line the grader still grades as the pass did is not named as changed.
    (tmp_path / "second").mkdir()
    again = _record_folder(tmp_path / "second", status="right")
    assert regrade.regrade(again, tmp_path / "out2", commit="b" * 40)["changed"] == []


def test_a_record_that_stopped_or_whose_seal_does_not_match_is_refused_and_nothing_is_written(tmp_path):
    regrade, runner = _mod("baselines.regrade"), _mod("baselines.runner")
    (tmp_path / "a").mkdir()
    stopped = _record_folder(tmp_path / "a")
    (stopped / "stopped.json").write_text(json.dumps({"error": "ValueError: x"}), encoding="utf-8")
    runner.seal_run_record(stopped)
    with pytest.raises(regrade.RegradeRefused, match="never graded"):
        regrade.regrade(stopped, tmp_path / "out-a")
    assert not (tmp_path / "out-a").exists()
    (tmp_path / "b").mkdir()
    torn = _record_folder(tmp_path / "b")
    (torn / "answers.jsonl").write_text((torn / "answers.jsonl").read_text(encoding="utf-8").replace("wrong", "right"), encoding="utf-8")
    with pytest.raises(regrade.RegradeRefused, match="not what its seal says"):
        regrade.regrade(torn, tmp_path / "out-b")
    assert not (tmp_path / "out-b").exists()
    # The command refuses with its own exit status.
    assert regrade.main([str(stopped), "--out", str(tmp_path / "out-c")]) == 3


def test_a_line_on_an_attempt_or_with_no_reading_is_kept_as_it_is(tmp_path):
    regrade, record = _mod("baselines.regrade"), _mod("baselines.record")
    none_q = _question("c06.generated.candidates", "manifest", "small")
    kept = record.Record(question_id=none_q.id, status="cannot answer", raw=None, evidence="a manifest for this build",
                         route=none_q.route, marks=(), excluded_from_tally=none_q.excluded_from_tally, attempt_id=None)
    folder = _record_folder(tmp_path, status="right", extra_lines=[record.answer_line(none_q, kept, commit="a" * 40, step_ids=[])])
    result = regrade.regrade(folder, tmp_path / "out", commit="b" * 40)
    assert result["changed"] == [] and result["kept as they were"] == 1


def test_the_manifest_reading_no_record_is_the_answer_not_covered_and_a_regrade_repairs_the_pass_3_lines(tmp_path):
    """Table README, "Readings": the reading "no record" equals the answer "not covered". The manifest writes "no record" and every
    other subject "not covered"; the grader compared the two words as plain text (pass 3: two lines graded wrong)."""
    grade = _mod("baselines.grade")
    question = _question("c05.unregistered-checkpoint", "manifest")
    expected = {"answer": "not covered"}
    reading = _mod("baselines.interfaces").Reading(question.id, {"answer": "no record"}, question.route)
    assert grade.grade(question, expected, reading).status == "right"
    other = _mod("baselines.interfaces").Reading(question.id, {"answer": "does not contain"}, question.route)
    assert grade.grade(question, expected, other).status == "wrong"


def test_the_frozen_files_the_graders_read_are_checked_against_the_hashes_the_record_kept(tmp_path):
    regrade, runner = _mod("baselines.regrade"), _mod("baselines.runner")
    (tmp_path / "a").mkdir()
    moved = _record_folder(tmp_path / "a")
    kept = json.loads((moved / "frozen-files.json").read_text(encoding="utf-8"))
    assert "fixture/spec/fixture_spec.toml" in kept and "fixture/table/check10.toml" in kept
    kept["fixture/spec/fixture_spec.toml"] = "0" * 64          # the spec of the pass was another file than the one found now
    (moved / "frozen-files.json").write_text(json.dumps(kept), encoding="utf-8")
    runner.seal_run_record(moved)
    with pytest.raises(regrade.RegradeRefused, match="frozen files the graders read differ.*fixture_spec.toml"):
        regrade.regrade(moved, tmp_path / "out-a")
    assert not (tmp_path / "out-a").exists()
    # A record with no frozen hashes (made before they were kept) is re-graded and the output says so.
    (tmp_path / "b").mkdir()
    old = _record_folder(tmp_path / "b", frozen=False)
    regrade.regrade(old, tmp_path / "out-b", commit="b" * 40)
    assert json.loads((tmp_path / "out-b" / "regrade.json").read_text(encoding="utf-8"))["frozen_files"] == "frozen files not recorded"


def test_a_record_whose_acceptance_helper_differs_or_was_not_hashed_is_refused_naming_the_helper(tmp_path):
    """18 Nov 2026: the seal names every acceptance module the graders load (``expected.ACCEPTANCE_MODULES``); a helper
    that is another file than the one the pass had is a refusal, like the spec."""
    regrade, runner, expected = _mod("baselines.regrade"), _mod("baselines.runner"), _mod("baselines.expected")
    (tmp_path / "a").mkdir()
    moved = _record_folder(tmp_path / "a")
    kept = json.loads((moved / "frozen-files.json").read_text(encoding="utf-8"))
    for module in expected.ACCEPTANCE_MODULES:
        assert f"tests/acceptance/{module}.py" in kept, module
    kept["tests/acceptance/acceptance_join.py"] = "0" * 64     # the helper of the pass was another file than the one found now
    (moved / "frozen-files.json").write_text(json.dumps(kept), encoding="utf-8")
    runner.seal_run_record(moved)
    with pytest.raises(regrade.RegradeRefused, match="frozen files the graders read differ.*acceptance_join.py"):
        regrade.regrade(moved, tmp_path / "out-a")
    assert not (tmp_path / "out-a").exists()
    # A record made before the helpers were hashed holds the spec and the table hashes only: it is refused too, naming the
    # helpers (the seal changed; the old records are re-graded from a checkout of the code and files they were made with).
    (tmp_path / "b").mkdir()
    old = _record_folder(tmp_path / "b")
    kept = {name: digest for name, digest in json.loads((old / "frozen-files.json").read_text(encoding="utf-8")).items()
            if not name.startswith("tests/acceptance/")}
    (old / "frozen-files.json").write_text(json.dumps(kept), encoding="utf-8")
    runner.seal_run_record(old)
    with pytest.raises(regrade.RegradeRefused, match="frozen files the graders read differ.*acceptance_sets.py"):
        regrade.regrade(old, tmp_path / "out-b", commit="b" * 40)


def test_the_modules_the_graders_load_from_the_acceptance_folder_are_all_named_for_the_seal():
    """The constant is not a list kept by hand beside the code: a module asked for by name in expected.py, or imported by the
    table loader, that the constant does not name would be a helper no record hashes."""
    import re

    expected, table10 = _mod("baselines.expected"), _mod("baselines.table10")
    asked = set(re.findall(r'_acceptance_module\("(\w+)"\)', Path(expected.__file__).read_text(encoding="utf-8")))
    asked |= set(re.findall(r"^\s+import (\w+)\s+# noqa: PLC0415", Path(table10.__file__).read_text(encoding="utf-8"), re.M))
    assert asked and asked <= set(expected.ACCEPTANCE_MODULES), sorted(asked - set(expected.ACCEPTANCE_MODULES))
    for module in expected.ACCEPTANCE_MODULES:
        assert (Path(table10.REPO) / "tests" / "acceptance" / f"{module}.py").is_file(), module


def test_a_line_that_was_an_expected_value_not_derived_and_is_graded_now_leaves_the_open_list_and_the_flags_follow(tmp_path):
    regrade, runner, record = _mod("baselines.regrade"), _mod("baselines.runner"), _mod("baselines.record")
    family = _question("c06.import.file-family", "manifest")
    folder = tmp_path / "record"
    folder.mkdir()
    runner.keep_expected_sources({"requests": {"withdraw A": {"files": ["sources/A.jsonl"]}}}, folder)
    held = record.Record(question_id=family.id, status="cannot answer", raw={"link": {"files": ["sources/A.jsonl"]}},
                         evidence="expected value not derived: the request had no files then", route=family.route, marks=(),
                         excluded_from_tally=family.excluded_from_tally, attempt_id=None)
    record.write_answers(folder / "answers.jsonl", [record.answer_line(family, held, commit="a" * 40, step_ids=[])])
    (folder / "start.txt").write_text("start: commit " + "a" * 40 + "; subjects manifest; step cap 900.0 seconds;\n", encoding="utf-8")
    # The pass's own summary: the question open (it holds no grade), so the flags read open.
    table10 = _mod("baselines.table10")
    everyone = [q.id for q in table10.expand(table10.load_table10()) if q.subject == "manifest"]
    (folder / "summary.txt").write_text("Check 10: open\nonce: open\nsmall: open\n10k: open\n100k: open\nplan sizes: open\n"
                                        + "\n".join(everyone) + "\n", encoding="utf-8")
    runner.seal_run_record(folder)
    result = regrade.regrade(folder, tmp_path / "out", commit="b" * 40)
    assert result["changed"] == [(family.id, "cannot answer", "right")]
    text = (tmp_path / "out" / "regrade-summary.txt").read_text(encoding="utf-8")
    assert "manifest once right: 1" in text and "manifest once expected value not derived" not in text
    assert text.splitlines()[0] == "Check 10: open", "the other questions of the subject hold no line: still open"
    assert family.id not in text.splitlines() and everyone[0] in text.splitlines(), "the question left the open list, the others stay"
    # A grade that is now not derived joins the open list (the other direction), counted under its own word.
    assert regrade._derived({"status": "cannot answer", "evidence": "not gradable: x"}) is True
    assert regrade._derived({"status": "right", "evidence": None}) is False


def test_a_regrade_opens_only_files_of_the_record_its_output_and_the_frozen_files_of_the_repository(tmp_path):
    """The graders open no world folder and no other build: the paths opened, outside the python files, are the record's, the
    output folder's and the fixture spec and table files (their sha256 the record keeps)."""
    import os
    import sys
    regrade = _mod("baselines.regrade")
    folder = _record_folder(tmp_path, status="wrong")
    opened: list[str] = []
    active = [True]

    def hook(event, args):
        if active[0] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(args[0]))

    sys.addaudithook(hook)
    try:
        regrade.regrade(folder, tmp_path / "out", commit="b" * 40)
    finally:
        active[0] = False
    repo = Path(_mod("baselines.table10").REPO).resolve()
    allowed = (str(folder), str(tmp_path / "out"), str(repo / "fixture"))
    stray = sorted({p for p in opened if not p.endswith((".py", ".pyc", ".so")) and p.startswith("/")
                    and not p.startswith(("/proc", "/sys", "/dev", "/usr", "/etc", "/lib"))
                    and "/site-packages/" not in p and not p.startswith(allowed) and str(tmp_path) in p})
    assert stray == [], stray
    assert any(p.startswith(str(repo / "fixture")) for p in opened), "the frozen files are read from the repository"


def test_a_tree_with_no_git_needs_the_commit_named_and_the_refusals_come_before_that(tmp_path, monkeypatch):
    """Found on the box (J12, 18 Nov 2026): the trees there are extracted from a tarball and hold no git, so the default commit
    (the head of the repository) does not exist. A re-grade then asks for --commit instead of raising an error of git, and a record
    that is refused is refused for its own reason first."""
    regrade = _mod("baselines.regrade")
    monkeypatch.setattr(regrade, "_head_commit", lambda: None)
    folder = _record_folder(tmp_path, status="wrong")
    with pytest.raises(regrade.RegradeRefused, match="give --commit"):
        regrade.regrade(folder, tmp_path / "out-none")
    assert not (tmp_path / "out-none").exists()
    assert regrade.regrade(folder, tmp_path / "out-given", commit="b" * 40)["changed"]
    (tmp_path / "s").mkdir()
    stopped = _record_folder(tmp_path / "s")
    (stopped / "stopped.json").write_text("{}", encoding="utf-8")
    _mod("baselines.runner").seal_run_record(stopped)
    with pytest.raises(regrade.RegradeRefused, match="never graded"):
        regrade.regrade(stopped, tmp_path / "out-s")
