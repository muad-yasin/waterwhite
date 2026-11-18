"""Re-grade the kept readings of a run record without running any tool (Check 10).

``python -m baselines.regrade RUN_RECORD --out OUT`` reads the readings a pass kept (``answers.jsonl``: the raw reading of each
line), builds the graders from the expected-value sources the record kept (``runner.world_from_record``: the counting documents,
the shard indexes, the oracle's logs and the world's request entries), grades each reading again and writes the lines with the new
grade, the lines whose status or evidence changed, and the summary. The record itself is never changed.

What is not re-graded, and why:

- a record that holds ``stopped.json`` is REFUSED: a pass that stopped is never graded, and a re-grade must not give it grades;
- a record whose ``sha256sums.txt`` does not match its files is REFUSED (the readings are not the ones the pass sealed);
- a record whose ``frozen-files.json`` (the sha256 of the fixture spec, of the table files and of the acceptance modules the graders
  read from the repository) differs from the files found, or lacks a file the seal now names, is REFUSED, naming them: the expected values would not be the ones the pass had; a record with no
  such file (a pass made before it was kept) is re-graded and the output is labelled "frozen files not recorded";
- a line that rests on a failed attempt (it has an ``attempt_id``) or that holds no reading (``raw`` is None: a none row, a
  no-join row, a question the table routes to nobody) is kept as it is: its status is not a grade of a reading;
- a line of a row that names two layouts is re-graded only when every layout has its raw reading; a layout with none (not
  attempted, or an attempt) keeps the line as it is.

The summary is made again from the lines; the questions open are those of the pass's own summary, moved by the lines the re-grade
changed (an expected value that was not derived and is now graded leaves the list; a grade that is now not derived joins it), and the
flags follow from that list. A none line stands only with the attempt of its subject, which the record does not keep as such; it is
never re-graded, so it never moves.

Written 18 Nov 2026 (the re-grade reader owed to the pass of record).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import record, runner, summary, table10
from .interfaces import Reading

STOPPED = runner.STOPPED_FILE
SEAL = runner.SEAL_FILE
ANSWERS = runner.ANSWERS_FILE
REGRADED_LINES = "answers.regraded.jsonl"
CHANGES = "regrade-changes.tsv"
SUMMARY = "regrade-summary.txt"
REGRADE_FILE = "regrade.json"


class RegradeRefused(ValueError):
    """The record is not one a re-grade may read: it stopped, or it is not what its seal says."""


def seal_problems(folder: Path) -> list[str]:
    """The files of ``folder`` that its ``sha256sums.txt`` does not match (missing, changed, or not listed)."""
    folder = Path(folder)
    seal = folder / SEAL
    if not seal.is_file():
        return [f"{SEAL} is missing"]
    listed: dict[str, str] = {}
    for text in seal.read_text(encoding="utf-8").splitlines():
        digest, _, name = text.partition("  ")
        listed[name] = digest
    problems = []
    for name, digest in listed.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"{name} is listed and missing")
        elif runner._sha256_of(path) != digest:
            problems.append(f"{name} differs from its sha256")
    present = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p != seal}
    problems += [f"{name} is not listed" for name in sorted(present - set(listed))]
    return problems


def _subjects_of(run_record: Path) -> list[str]:
    found = re.search(r"subjects ([^;]+);", (run_record / "start.txt").read_text(encoding="utf-8"))
    if found is None:
        raise RegradeRefused("start.txt names no subjects")
    return [name.strip() for name in found.group(1).split(",")]


def regrade_line(line: Mapping[str, Any], question, grade_of, grade_layouts_of) -> dict | None:
    """The line with its grade made again from its kept reading, or None where the line is not re-gradable."""
    if line.get("attempt_id") is not None or line.get("raw") is None:
        return None
    layouts = record.layouts_of(question)
    if layouts is not None:
        raw = line["raw"]
        if not isinstance(raw, Mapping) or any(raw.get(name) is None for name in layouts):
            return None
        grade = runner._grade_or_not_derived(lambda: grade_layouts_of(
            question, {name: Reading(question.id, raw[name], question.route) for name in layouts}))
    else:
        grade = runner._grade_or_not_derived(lambda: grade_of(question, Reading(question.id, line["raw"], question.route)))
    changed = dict(line)
    changed["status"] = grade.status
    changed["evidence"] = record._plain(grade.evidence)
    return changed


def _derived(line) -> bool:
    """A line that counts under its own word (expected value not derived): no grade, the question stays open."""
    return bool(line is not None and line.get("status") == record.CANNOT_ANSWER and isinstance(line.get("evidence"), str)
                and line["evidence"].startswith(summary.NOT_DERIVED_EVIDENCE))


def _summary_text(run_record: Path, every, old_lines, lines) -> str:
    """The summary after the re-grade, in the form of the pass's own. The counts are made from the lines. The questions open are
    those of the pass's own summary, moved by the lines the re-grade changed: a line that was an expected value not derived and
    is now a grade leaves the open list, a grade that is now not derived joins it (the only changes that move a question in or
    out; right, wrong and cannot answer among themselves do not). The flags follow from the open list. This needs no store attempt:
    a none line, which stands only with the attempt of its subject, is never re-graded."""
    original = (run_record / "summary.txt").read_text(encoding="utf-8").splitlines()
    open_before = [item for item in original if re.fullmatch(r"[A-Za-z0-9_.@/|+-]+", item) and ":" not in item and " " not in item]
    old_by_id = {line["id"]: line for line in old_lines}
    by_id = {line["id"]: line for line in lines}
    left = {i for i, line in by_id.items() if _derived(old_by_id.get(i)) and not _derived(line)}
    joined = {i for i, line in by_id.items() if not _derived(old_by_id.get(i)) and _derived(line)}
    still = set(open_before) - left
    ordered = [q.id for q in every if q.id in still or q.id in joined]
    counts: dict = {}
    tallied: dict = {}
    labels_open = set()
    for question in every:
        line = by_id.get(question.id)
        word = (record.NOT_ATTEMPTED if line is None else summary.NOT_DERIVED if _derived(line)
                else summary._word_of(line["status"]))
        key = (question.subject, summary._label(question), word)
        counts[key] = counts.get(key, 0) + 1
        if not question.excluded_from_tally:
            tallied[key] = tallied.get(key, 0) + 1
        if question.id in still or question.id in joined:
            labels_open.add(summary._label(question))
    done = summary.Summary(
        counts=counts, tallied=tallied, complete=not ordered, open=tuple(ordered),
        plan_sizes_complete=not ({"10k", "100k"} & labels_open),
        size_complete={label: label not in labels_open for label in record.LABELS})
    return summary.render(done)


def _extensions_of(run_record: Path) -> list[str]:
    """The extensions the pass asked the rows of: the file beside start.txt; a record without one asked none."""
    found = Path(run_record) / runner.EXTENSIONS_FILE
    return found.read_text(encoding="utf-8").split() if found.is_file() else []


def _commit_of_start(run_record: Path) -> str | None:
    found = re.search(r"commit ([0-9a-f]{7,40})", (run_record / "start.txt").read_text(encoding="utf-8"))
    return found.group(1) if found else None


def _head_commit() -> str | None:
    """The head of the repository the code is read from, or None where it has no git (a tree extracted from a tarball)."""
    done = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(table10.REPO))
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else None


def _code_tree_clean() -> bool | None:
    """Whether the repository the code is read from has no change beside its commit (``git status --porcelain`` of the tracked
    files is empty): a re-grade made in a changed tree grades with code its commit does not hold, and says so. None where there is no
    git."""
    done = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                          cwd=Path(table10.REPO))
    return not done.stdout.strip() if done.returncode == 0 else None


def frozen_problems(run_record: Path) -> list[str] | None:
    """The frozen files that differ from the hashes the record kept; None where the record kept none."""
    kept = run_record / runner.FROZEN_FILES
    if not kept.is_file():
        return None
    recorded = json.loads(kept.read_text(encoding="utf-8"))
    now = runner.frozen_files_hashes()
    return [name for name in sorted(set(recorded) | set(now)) if recorded.get(name) != now.get(name)]


def regrade(run_record, out, commit: str | None = None) -> dict[str, Any]:
    """Re-grade the lines of ``run_record`` into the new folder ``out`` with the code of ``commit`` (default: the head of the
    repository). Returns the counts and the changes."""
    run_record, out = Path(run_record), Path(out)
    if (run_record / STOPPED).exists():
        raise RegradeRefused(f"{run_record} holds {STOPPED}: a pass that stopped is never graded")
    problems = seal_problems(run_record)
    if problems:
        raise RegradeRefused(f"{run_record} is not what its seal says: " + "; ".join(problems[:5]))
    moved = frozen_problems(run_record)
    if moved:
        raise RegradeRefused("the frozen files the graders read differ from the ones the pass had: " + ", ".join(moved))
    commit = commit or _head_commit()
    if commit is None:
        raise RegradeRefused("the code tree has no git commit to name (an extracted tree): give --commit")
    if out.exists():
        raise RegradeRefused(f"{out} exists: nothing is overwritten")
    world = runner.world_from_record(run_record)
    grade_of, grade_layouts_of = runner._graders(world)
    selected = _subjects_of(run_record)
    every = [q for q in table10.expand(table10.load_table10(), _extensions_of(run_record)) if q.subject in selected]
    by_id = {q.id: q for q in every}
    lines = [json.loads(text) for text in (run_record / ANSWERS).read_text(encoding="utf-8").splitlines() if text.strip()]
    new_lines, changes, kept = [], [], 0
    for line in lines:
        question = by_id.get(line["id"])
        again = regrade_line(line, question, grade_of, grade_layouts_of) if question is not None else None
        if again is None:
            new_lines.append(line)
            kept += 1
            continue
        if again["status"] != line["status"] or again["evidence"] != line["evidence"]:
            again["regraded_by_commit"] = commit
            changes.append((line["id"], line["status"], again["status"]))
        new_lines.append(again)
    out.mkdir(parents=True)
    (out / REGRADED_LINES).write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in new_lines), encoding="utf-8")
    (out / CHANGES).write_text("".join(f"{i}\t{old}\t{new}\n" for i, old, new in changes), encoding="utf-8")
    (out / SUMMARY).write_text(_summary_text(run_record, every, lines, new_lines), encoding="utf-8")
    (out / REGRADE_FILE).write_text(json.dumps({
        "grading_commit": commit, "record_commit": _commit_of_start(run_record), "code_tree_clean": _code_tree_clean(),
        "record_seal_sha256": runner._sha256_of(run_record / SEAL),
        "frozen_files": "checked against the record" if moved is not None else "frozen files not recorded"},
        indent=1, sort_keys=True) + "\n", encoding="utf-8")
    runner.seal_run_record(out)
    return {"lines": len(lines), "kept as they were": kept, "changed": changes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m baselines.regrade", description=__doc__.split("\n")[0])
    parser.add_argument("run_record", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="the new folder the re-graded lines are written to")
    parser.add_argument("--commit", help="the commit whose code grades (default: the head of the repository)")
    arguments = parser.parse_args(argv)
    try:
        result = regrade(arguments.run_record, arguments.out, arguments.commit)
    except RegradeRefused as refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 3
    print(f"re-graded {result['lines']} lines: {len(result['changed'])} changed, {result['kept as they were']} kept as they were")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
