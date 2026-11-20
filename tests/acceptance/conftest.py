"""Fixtures of the acceptance tests, and the count of rows a run prints.

The stores are built once per session, by the replica, under pytest's base temp
folder. The summary at the end of a run counts rows by check: how many ran, how
they ended, and how many were skipped for each reason.

A row of Check 3 or 6 that is asked of a generated set has one test per size. The
summary counts such a row once in the check's number of rows and gives the outcomes
size by size, so that a row asked of three sets is never read as three rows.
"""

from __future__ import annotations

from collections import Counter

import pytest
import table
from acceptance_rows import PART_ROW, PROPERTY
from acceptance_repair import SetRepairs
from acceptance_sets import SIZES
from acceptance_world import SET_UP_NOTES, World


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> World:
    return World(tmp_path_factory.mktemp("acceptance"))


@pytest.fixture(autouse=True)
def _stores_are_not_written(world):
    """With WATERWHITE_ACCEPTANCE_GUARD set, the test that writes to a built store fails (see
    acceptance_world.GUARD_VARIABLE)."""
    yield
    world.assert_stores_unchanged()


@pytest.fixture(scope="session")
def repairs(world) -> SetRepairs:
    """The generated sets built, repaired and verified through the commands, once each (Check 9)."""
    return SetRepairs(world.sets)


OUTCOMES = ("passed", "failed", "xfailed", "xpassed", "skipped", "error")


def _outcome_lines(write, count: Counter, skipped: Counter, deferred: int, indent: str) -> None:
    ran = sum(count[o] for o in ("passed", "failed", "xfailed", "xpassed"))
    write(f"{indent}ran {ran}: passed {count['passed']}, failed {count['failed']}, "
          f"expected failures {count['xfailed']}, "
          f"expected failures that passed {count['xpassed']}")
    for reason, number in sorted(skipped.items(), key=lambda item: (-item[1], item[0])):
        write(f"{indent}skipped {number}: {reason}")
    if deferred:
        write(f"{indent}{deferred} of the rows that ran hold an expected field that was not "
              "compared; each has a skipped test of its own that names the field")


def pytest_terminal_summary(terminalreporter) -> None:
    # (check, size) -> outcomes; size is None for a row that is asked once.
    rows: dict[tuple[int, str | None], Counter] = {}
    skipped: dict[tuple[int, str | None], Counter] = {}
    waiting: dict[tuple[int, str | None], set[str]] = {}
    ran_rows: dict[tuple[int, str | None], set[str]] = {}
    ids: dict[int, set[str]] = {}
    for outcome in OUTCOMES:
        for report in terminalreporter.stats.get(outcome, []):
            note = dict(getattr(report, "user_properties", [])).get(PROPERTY)
            if note is None or (report.when != "call" and outcome != "error"):
                continue
            key = (note["check"], note.get("size"))
            if note["part"] != PART_ROW:
                # The second test of a row with a field that cannot be compared yet.
                # It counts only where the row itself ran.
                if note.get("deferred"):
                    waiting.setdefault(key, set()).add(note["row"])
                continue
            ids.setdefault(note["check"], set()).add(note["row"])
            rows.setdefault(key, Counter())[outcome] += 1
            if outcome not in ("skipped", "error"):
                ran_rows.setdefault(key, set()).add(note["row"])
            if outcome == "skipped":
                skipped.setdefault(key, Counter())[note["skip"] or "no reason given"] += 1
    if not rows:
        return
    parts = Counter({key: len(names & ran_rows.get(key, set()))
                     for key, names in waiting.items()})
    write = terminalreporter.write_line
    terminalreporter.section("acceptance rows by check (plan 5.5)")
    write("A skipped row is not a pass. An expected failure is not a pass.")
    for note in dict.fromkeys(SET_UP_NOTES):
        write(f"Set-up of this folder: {note}")
    for check in sorted(ids):
        scopes = [key for key in rows if key[0] == check]
        total = len(table.load_check(check))
        if scopes == [(check, None)] and total == len(ids[check]):
            count = rows[check, None]
            ran = sum(count[o] for o in ("passed", "failed", "xfailed", "xpassed"))
            errors = (f", errors before the question was put {count['error']}"
                      if count["error"] else "")
            write(f"Check {check}: {sum(count.values())} rows; ran {ran}, "
                  f"skipped {count['skipped']}{errors}")
            _outcome_lines(write, count, skipped.get((check, None), Counter()),
                           parts[check, None], "  ")
            continue
        left_out = total - len(ids[check])
        write(f"Check {check}: {total} rows" + (
            f"; {left_out} of them not selected in this run (their tests carry a marker "
            "the run left out)" if left_out else ""))
        order = [None, *SIZES]
        for key in sorted(scopes, key=lambda k: order.index(k[1])):
            count = rows[key]
            ran = sum(count[o] for o in ("passed", "failed", "xfailed", "xpassed"))
            errors = (f", errors before the question was put {count['error']}"
                      if count["error"] else "")
            where = ("asked once, of a store of the table or of the spec" if key[1] is None
                     else f"asked of the generated set {key[1]} "
                          f"({SIZES[key[1]]['records']} records)")
            write(f"  {where}: {sum(count.values())} rows; ran {ran}, "
                  f"skipped {count['skipped']}{errors}")
            _outcome_lines(write, count, skipped.get(key, Counter()), parts[key], "    ")
