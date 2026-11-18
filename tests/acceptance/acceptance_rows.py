"""How a row of the hand-written table becomes a test.

``params(rows)`` gives one test per row, named by the row's id. A row that holds an
expected field which cannot be compared yet, beside fields that can, gets a second
test named ``<row id> [not compared: <fields>]``. That second test is always
skipped, with what the fields wait for as its reason, so the first one passing is
never read as the whole row passing.

``sized_params(rows, small)`` does the same for a check with rows that are asked
of a generated set (Checks 3 and 6). Such a row gets one test per size, named
``<row id> [<size>]``: the small set of this folder, and each size the plan names
for the row, which carries the markers ``slow`` and ``megatron``. A row that is
asked of a store of the table, or of the spec, gets one test as before.

``grade`` runs one of these tests:

1. the row is interpreted (``table.interpret``); a row that cannot be read is an
   error of the test, never a skip;
2. an extension row is skipped with "extension off"; a row a fallback in force
   takes out of its check is skipped with the fallback; a row that needs something
   not built yet is skipped with the need; a row with a pending value whose spec
   item is empty is skipped as not checked;
3. a row listed in ``expected_failures.py`` is marked as an expected failure with
   its diagnosis;
4. the question is put (``acceptance_world.World.put``) and every expected field
   that differs is reported. A question that finds what it needs missing while it
   is put (a store the replica does not build, a set with no shard registered, a
   machine without the Megatron environment) is skipped with that reason.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import tomllib
from collections.abc import Sequence

import acceptance_probe
import acceptance_sets
import pytest
import table
from acceptance_sets import SMALL, Unavailable
from acceptance_world import NEED_MEGATRON_ENV, NOT_BUILT, World
from expected_failures import EXPECTED_FAILURES

PART_ROW = "row"
PART_DEFERRED = "fields not compared"

# The key of the note a test leaves for the summary (conftest.py).
PROPERTY = "acceptance"

# The fallbacks of README section 8 that are in force. None is: no check has been
# answered under one.
FALLBACKS_IN_FORCE: tuple[str, ...] = ()

# The kinds of row that are asked of a generated set and need no shard: they are
# answered from the plant list and the chunk files alone.
WITHOUT_SHARDS = frozenset({table.KIND_PAIR_KINDS})


def _parts(row: table.Row, name: str) -> list[tuple[str, str]]:
    """(part, test id) of the tests one row gets under ``name``."""
    out = [(PART_ROW, name)]
    try:
        waiting = table.interpret(row, FALLBACKS_IN_FORCE).waiting(NOT_BUILT)
    except table.TableError:
        return out  # the row's own test raises it again, where it is reported
    if waiting:
        out.append((PART_DEFERRED, f"{name} [not compared: {', '.join(sorted(waiting))}]"))
    return out


def params(rows: Sequence[table.Row]) -> list:
    return [pytest.param(row, part, id=name)
            for row in rows for part, name in _parts(row, row.id)]


# A set of the plan's sizes is a long build. It runs only when the environment
# variable WATERWHITE_CLEARED_LONG_RUNS names it as "<check>:<size>" (a comma list,
# for example "3:100k"), set for a run that has been cleared; a marker alone would
# let a run that selects "slow" start it.
CLEARED_LONG_RUNS = "WATERWHITE_CLEARED_LONG_RUNS"


def long_run_gate(check: int, size: str):
    cleared = {item.strip() for item in os.environ.get(CLEARED_LONG_RUNS, "").split(",")}
    return pytest.mark.skipif(
        f"{check}:{size}" not in cleared,
        reason=f"a long run: Check {check} on the {size} set runs only when "
               f"{CLEARED_LONG_RUNS} names {check}:{size}",
    )


def sized_params(rows: Sequence[table.Row], small: str = SMALL) -> list:
    """One test per row and size. ``size`` is None for a row that is asked once."""
    out = []
    for row in rows:
        try:
            question = table.interpret(row, FALLBACKS_IN_FORCE)
        except table.TableError:
            out.append(pytest.param(row, PART_ROW, None, id=row.id))
            continue
        if not question.sized:
            marks = ([pytest.mark.megatron] if question.kind in table.FORM_KINDS | {
                table.KIND_ANSWER} and question.store in table.TOKENIZED_STORES else [])
            out += [pytest.param(row, part, None, id=name, marks=marks)
                    for part, name in _parts(row, row.id)]
            continue
        for size in (small, *table.sizes_of(row)):
            marks = [] if question.kind in WITHOUT_SHARDS else [pytest.mark.megatron]
            if size != small:
                marks.append(pytest.mark.slow)
                marks.append(long_run_gate(row.check, size))
            out += [pytest.param(row, part, size, id=name, marks=marks)
                    for part, name in _parts(row, f"{row.id} [{size}]")]
    return out


@functools.cache
def _spec() -> dict:
    with open(table.REPO / "fixture" / "spec" / "fixture_spec.toml", "rb") as f:
        return tomllib.load(f)


def grade(row: table.Row, part: str, world: World, request: pytest.FixtureRequest,
          size: str | None = None) -> None:
    __tracebackhide__ = True  # a failure shows the row's fields, not this function
    note = {"check": row.check, "row": row.id, "part": part, "skip": None, "size": size}
    request.node.user_properties.append((PROPERTY, note))

    def skip(reason: str) -> None:
        note["skip"] = reason
        pytest.skip(reason)

    question = table.interpret(row, FALLBACKS_IN_FORCE)

    if question.kind == table.KIND_EXTENSION:
        skip(f"extension off: {row.extension}")
    if question.kind == table.KIND_DOES_NOT_APPLY:
        skip("does not apply under the fallback in force: " + ", ".join(FALLBACKS_IN_FORCE))
    absent = world.unmet(question)
    if NEED_MEGATRON_ENV in absent:
        skip("not run: its store is tokenized only where the Megatron environment is set "
             f"({acceptance_sets.MEGATRON_VARIABLE})")
    if absent:
        skip("not built yet: " + ", ".join(absent))
    if acceptance_probe.rests_on_bound(question) and acceptance_probe.sandbox_missing() is not None:
        skip("not run: it rests on a bound, which only the confined harness establishes, and "
             "the sandbox cannot be made: " + acceptance_probe.sandbox_missing())
    waiting = question.waiting(NOT_BUILT)
    if part == PART_DEFERRED:
        note["deferred"] = sorted(waiting)
        skip("not built yet: " + ", ".join(sorted(set(waiting.values())))
             + " (one expected field of a row whose other fields are compared)")
    question = question.without(waiting)

    expect, empty = table.fill_pending(dict(question.expect), _spec())
    if empty:
        skip("not checked: the fixture spec holds no value yet for the pending item "
             + ", ".join(empty))
    question = dataclasses.replace(question, expect=expect)

    diagnosis = EXPECTED_FAILURES.get(row.id)
    if diagnosis is not None:
        request.node.add_marker(pytest.mark.xfail(
            reason=diagnosis.reason(), raises=table.RowMismatch, strict=False))

    try:
        mismatches = world.put(question, size)
    except Unavailable as missing:
        skip(str(missing))
    if not mismatches:
        return
    failure = table.RowMismatch(row.id, mismatches)
    if diagnosis is not None and not set(failure.fields) <= set(diagnosis.fields):
        # The row fails in a field its diagnosis does not cover: that is a new
        # failure, and the expected-failure marker must not hide it.
        raise AssertionError(
            f"{failure}\n  the diagnosis in expected_failures.py covers only "
            f"{list(diagnosis.fields)}")
    raise failure
