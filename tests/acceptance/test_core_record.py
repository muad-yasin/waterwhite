"""The recorder of the core result (fixture/table/README.md "Core result"): one test per
row and size of Checks 1 to 9, each putting the row's question the way its own check
does (the check's test function is called, so the skips, the expected failures and the
long-run gates are the check's own) and keeping what the tool gave (core_record.py).

It runs only when ``WATERWHITE_CORE_RECORD`` is 1: it is a second run of every check,
made to be recorded, and a full suite does not need it. The records are test properties
of the junit file; ``python3 -I tests/acceptance/core_record.py merge OUT JUNIT...``
makes the one file from them.

The checks that need no run are not skipped: the table of rows named here is compared
with the table of the checks, and the comparison of two files is shown to name a record
that was altered.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import acceptance_probe
import acceptance_rows
import acceptance_sets
import core_record as cr
import pytest
import table
import test_check01
import test_check02
import test_check03
import test_check04
import test_check05
import test_check06
import test_check07
import test_check08
import test_check09

from replica import fixture as replica_fixture

# The stores of the replica's table that are built unseeded (capture False: every shuffle drawn from the
# operating system's randomness): replica/fixture.py TABLE_STORES. A row is marked by the store it is asked of.
UNSEEDED_STORES = frozenset(name for name, variant in replica_fixture.TABLE_STORES.items() if not variant.capture)

SWITCH = "WATERWHITE_CORE_RECORD"
SELECT_CHECKS = "WATERWHITE_CORE_RECORD_CHECKS"  # optional: a comma list of check numbers, for a probe
SELECT_ROWS = "WATERWHITE_CORE_RECORD_ROWS"      # optional: a comma list of row ids, for a probe

# check number -> (its test function, its rows, whether a row is asked of sets of several sizes)
CHECKS = {
    1: (test_check01.test_check01, test_check01.ROWS, False),
    2: (test_check02.test_check02, test_check02.ROWS, False),
    3: (test_check03.test_check03, test_check03.ROWS, True),
    4: (test_check04.test_check04, test_check04.ROWS, False),
    5: (test_check05.test_check05, test_check05.ROWS, False),
    6: (test_check06.test_check06, test_check06.ROWS, True),
    7: (test_check07.test_check07, test_check07.ROWS, False),
    8: (test_check08.test_check08, test_check08.ROWS, False),
    9: (test_check09.test_check09, test_check09.ROWS, None),  # its own way of putting a row
}


def _params() -> list:
    out = []
    for number, (_, rows, sized) in CHECKS.items():
        if sized is None:
            out += [pytest.param(number, row, "row", None, None, id=f"c{number}:{row.id}") for row in rows]
            continue
        for p in (acceptance_rows.sized_params(rows) if sized else acceptance_rows.params(rows)):
            row, part, *more = p.values
            size = more[0] if more else None
            # A long-run gate is a skipif mark: it would skip the test before it can leave a record.
            # The gate is read here and applied inside the test, which records the skip.
            gate = next((m for m in p.marks if m.name == "skipif"), None)
            gated = gate.kwargs.get("reason") if gate is not None and gate.args and gate.args[0] else None
            marks = [m for m in p.marks if m.name != "skipif"]
            out.append(pytest.param(number, row, part, size, gated, id=f"c{number}:{p.id}", marks=marks))
    return out


PARAMS = _params()


@pytest.fixture(scope="session", autouse=True)
def stores_built_before_the_rows(world):
    """The stores of the table are built before the first row, outside any record, so that
    no row's record holds the work of building a store another row also reads. Only when the
    recorder is switched on, and only the stores of the checks selected."""
    if os.environ.get(SWITCH) != "1":
        return
    # The probes of the helpers are asked once per process (functools.cache) by whichever row comes first:
    # asked here, outside any record.
    for probe in (acceptance_probe.repair_gaps, acceptance_probe.show_gaps, acceptance_probe.sandbox_missing):
        probe()
    selected = os.environ.get(SELECT_CHECKS)
    # The generated sets are built, and recovered, once, for whichever row asks first: asked here too.
    if not selected or {"3", "6"} & set(selected.split(",")):
        for mode in (acceptance_sets.CAPTURE, acceptance_sets.NO_CAPTURE):
            try:
                built = world.sets.build(acceptance_rows.SMALL, mode)
                built.need_shards()
                world.sets.state(built, acceptance_sets.STATE_RECOVERED)
            except BaseException:  # the rows say why they are skipped
                pass
    names = set()
    for number, (_, rows, sized) in CHECKS.items():
        if sized is None or (selected and str(number) not in selected.split(",")):
            continue
        for row in rows:
            try:
                names.add(table.interpret(row, acceptance_rows.FALLBACKS_IN_FORCE).store)
            except table.TableError:
                continue
    for name in sorted(n for n in names if n):
        try:
            world.store(name)
        except BaseException:  # the row asks again and is skipped or fails there, with its own reason
            pass


def _put(number, row, part, size, world, request, repairs):
    function, _, sized = CHECKS[number]
    if sized is None:
        return function(row, request, world, repairs)
    if sized:
        return function(row, part, size, world, request)
    return function(row, part, world, request)


def record_row(put, *, number, row, part, size, gated, world, request, record_property):
    """Puts one row with ``put`` and keeps what the tool gave, whatever the row ends in; the
    exception of the row is raised again, so the test ends as the check's own test ends."""
    scrub = cr.Scrub({str(world.base): "<BASE>", str(table.REPO): "<REPO>", sys.executable: "<PYTHON>",
                      str(Path(sys.executable).parent): "<PYBIN>"})
    calls = cr.Calls(scrub)
    world.harness_evidence = {}
    # The world keeps what ``show`` printed, once per command and store file: a row that comes after another
    # that asked the same would start no program. Cleared, every row starts its own, whatever ran before it.
    getattr(world, "_shown", {}).clear()
    try:
        question = table.interpret(row, acceptance_rows.FALLBACKS_IN_FORCE)
        kind, store = question.kind, question.store
    except table.TableError:
        kind = store = None
    unseeded = store in UNSEEDED_STORES
    ended, reason, mismatch = "passed", None, None
    try:
        if gated:
            pytest.skip(gated)
        with calls.watch():
            put()
    except BaseException as error:
        if isinstance(error, pytest.skip.Exception):
            ended, reason = "skipped", getattr(error, "msg", None) or str(error)
        elif isinstance(error, (table.RowMismatch, AssertionError)):
            ended = "failed"
            mismatch = scrub(str(error))[:4000]
            if request.node.get_closest_marker("xfail") is not None:
                reason = "an expected failure of expected_failures.py"
        else:
            ended, mismatch = "error", scrub(f"{type(error).__name__}: {error}")[:4000]
        raise
    finally:
        # The check's own note for the summary is taken off: a recorder run must not count a row twice.
        request.node.user_properties[:] = [p for p in request.node.user_properties
                                           if p[0] != acceptance_rows.PROPERTY]
        evidence = copy.deepcopy(world.harness_evidence) or None
        record = cr.row_record(
            check=number, row_id=row.id, size=size, part=part, core=row.core, extension=row.extension,
            ended=ended, reason=reason, calls=calls.calls, kind=kind, store=store, unseeded=unseeded,
            evidence=None if evidence is None else cr.canonical_scrub(scrub, evidence),
            mismatch=mismatch)
        line = cr.canonical(record)
        record_property(cr.PROPERTY_RECORD, line)
        record_property(cr.PROPERTY_SHA, cr.digest(line))


@pytest.mark.parametrize(("number", "row", "part", "size", "gated"), PARAMS)
def test_core_record(number, row, part, size, gated, world, repairs, request, record_property):
    if os.environ.get(SWITCH) != "1":
        pytest.skip(f"the recorder runs when {SWITCH} is 1")
    checks, rows = os.environ.get(SELECT_CHECKS), os.environ.get(SELECT_ROWS)
    if (checks and str(number) not in checks.split(",")) or (rows and row.id not in rows.split(",")):
        pytest.skip("not selected for this probe")
    record_row(lambda: _put(number, row, part, size, world, request, repairs), number=number, row=row,
               part=part, size=size, gated=gated, world=world, request=request, record_property=record_property)


# --- what needs no run -------------------------------------------------------------------------------


def test_every_core_row_of_the_checks_has_a_record_test():
    named = {p.values[1].id for p in PARAMS}
    table_rows = {row.id for n in CHECKS for row in table.load_check(n) if row.core}
    assert table_rows <= named, f"core rows with no record test: {sorted(table_rows - named)}"
    assert len({(p.values[1].id, p.values[2], p.values[3]) for p in PARAMS}) == len(PARAMS)


def _record(row, **more):
    base = dict(check=1, row_id=row, size=None, part="row", core=True, extension=None, ended="passed",
                reason=None, calls=[{"args": ["x"], "exit": 0, "stdout": "{}"}], evidence=None, mismatch=None)
    base.update(more)
    return cr.row_record(**base)


def test_the_comparison_names_a_record_that_was_altered_and_one_that_is_missing():
    before = [_record("c01.a"), _record("c01.b"), _record("c01.e", core=False, extension="x")]
    altered = copy.deepcopy(before)
    altered[1]["calls"][0]["stdout"] = '{"answer":"may contain"}'
    altered[2]["calls"][0]["stdout"] = "{}  "  # an extension row: not part of the core result
    assert cr.differences(before, before) == []
    assert cr.differences(before, altered) == ["c01.b|row|-: differs in calls"]
    assert cr.differences(before, altered[:1] + altered[2:]) == ["c01.b|row|-: only in the first"]


def test_the_merge_refuses_a_duplicate_and_a_record_that_does_not_match_its_digest(tmp_path):
    line = cr.canonical(_record("c01.a"))

    def junit(name, sha):
        path = tmp_path / name
        path.write_text(
            '<testsuites><testsuite><testcase name="t"><properties>'
            f'<property name="core_record" value=\'{line}\'/><property name="core_record_sha256" value="{sha}"/>'
            "</properties></testcase></testsuite></testsuites>", encoding="ascii")
        return path

    good = junit("a.xml", cr.digest(line))
    assert [r["row"] for r in cr.merge([good])] == ["c01.a"]
    with pytest.raises(ValueError, match="twice"):
        cr.merge([good, junit("b.xml", cr.digest(line))])
    with pytest.raises(ValueError, match="sha256"):
        cr.merge([junit("c.xml", "0" * 64)])


class _Node:
    user_properties: list

    def __init__(self):
        self.user_properties = [(acceptance_rows.PROPERTY, {"check": 1})]

    def get_closest_marker(self, name):
        return None


def _fake(tmp_path):
    from types import SimpleNamespace
    props = []
    world = SimpleNamespace(base=tmp_path, harness_evidence={})
    request = SimpleNamespace(node=_Node())
    return world, request, props


@pytest.mark.parametrize(("raises", "ended"), [(AssertionError("the answer was wrong"), "failed"),
                                                 (RuntimeError("the store broke"), "error")])
def test_a_row_that_fails_is_recorded_and_the_failure_is_raised_again(tmp_path, raises, ended):
    row = next(r for r in table.load_check(1) if r.core)
    world, request, props = _fake(tmp_path)

    def put():
        raise raises

    with pytest.raises(type(raises)):
        record_row(put, number=1, row=row, part="row", size=None, gated=None, world=world, request=request,
                   record_property=lambda k, v: props.append((k, v)))
    record = cr.json.loads(dict(props)[cr.PROPERTY_RECORD])
    assert record["ended"] == ended and str(raises) in record["mismatch"]
    assert dict(props)[cr.PROPERTY_SHA] == cr.digest(dict(props)[cr.PROPERTY_RECORD])
    assert request.node.user_properties == []  # the check's own note for the summary is not left behind


def test_a_gated_row_is_recorded_as_skipped(tmp_path):
    row = next(r for r in table.load_check(1) if r.core)
    world, request, props = _fake(tmp_path)
    with pytest.raises(pytest.skip.Exception):
        record_row(lambda: None, number=1, row=row, part="row", size="10k", gated="a long run", world=world,
                   request=request, record_property=lambda k, v: props.append((k, v)))
    record = cr.json.loads(dict(props)[cr.PROPERTY_RECORD])
    assert (record["ended"], record["reason"], record["size"]) == ("skipped", "a long run", "10k")


def test_a_wrong_value_in_a_row_of_an_unseeded_store_turns_the_comparison_red_and_a_new_build_does_not():
    def call():  # one call object per record: two records never share one
        return {"args": ["x"], "exit": 0, "stdout": '{"instance":"aaaa"}'}
    before = [_record("c06.u", unseeded=True, calls=[call()]), _record("c01.s", calls=[call()])]
    other_build = copy.deepcopy(before)
    other_build[0]["calls"][0]["stdout"] = '{"instance":"bbbb"}'  # another build: another id, still right
    assert cr.differences(before, other_build) == []
    wrong = copy.deepcopy(before)
    wrong[0]["ended"] = "failed"  # the grader found the value of this build wrong
    assert cr.differences(before, wrong) == ["c06.u|row|-: differs in ended"]
    refused = copy.deepcopy(before)
    refused[0]["calls"][0]["exit"] = 2  # the tool refused where it answered
    assert cr.differences(before, refused) == ["c06.u|row|-: differs in exits"]
    more_shards = copy.deepcopy(before)
    more_shards[0]["calls"].append(call())  # another build drew another number of shards: another number of calls
    assert cr.differences(before, more_shards) == []
    none_started = copy.deepcopy(before)
    none_started[0]["calls"] = []  # but a row that started no program where it did before is a difference
    assert cr.differences(before, none_started) == ["c06.u|row|-: differs in exits, program_started"]
    seeded = copy.deepcopy(before)
    seeded[1]["calls"][0]["stdout"] = '{"instance":"bbbb"}'  # a seeded row is never narrowed
    assert cr.differences(before, seeded) == ["c01.s|row|-: differs in calls"]


def test_the_rows_of_the_unseeded_store_are_selected_by_the_replicas_store_table():
    assert UNSEEDED_STORES == {"no-capture"}
    assert {name for name, v in replica_fixture.TABLE_STORES.items() if v.capture} >= {"main"}
