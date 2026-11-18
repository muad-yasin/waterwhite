"""The loader of ``fixture/table/check10.toml`` and the expander from its groups to
the questions of Check 10 (plan 5.5 Check 10; table README, section "Check 10").

Every expected value is read from the table files by ``tests/baselines/conftest.py``
(``tomllib`` on ``check01.toml`` to ``check10.toml``); the loader is compared with
the file, never with a value it computed. The numbers (66 groups, 370 core rows,
2346 questions, 147 left out of the tally, 737 with a route) are the counts taken
from the table at 5b76911 and frozen before the run; each is asserted as well as the
count taken from the file, so a table that moved is noticed.

The unit of a question is a listed row, a subject and the size at which the tool's
own row is asked. The sizes come from the tool's own reading of its rows
(``table.interpret(row, fallbacks).sized`` and ``table.sizes_of``), worked out by
``conftest.py``: a listed row read as sized is asked at "small" and at each size
``sizes_of`` gives (17 rows: the 13 generated rows of Check 3 at small and 100k,
the 4 generated rows of Check 6 at small, 10k and 100k); every other listed row is
asked once (size None), among them ``c03.precondition.eod-convention``, which names
the 100k set, and the ``c09.10k`` rows.

The names this file asks of ``baselines.table10``:

- ``load_table10(path=None)``: the table as it stands in ``fixture/table/check10.toml``
  (or in the file ``path`` names), with ``groups`` and ``rows`` in file order;
- a group has ``id``, ``check``, ``question``, ``rows`` (the listed row ids, a
  tuple) and ``source``;
- a row has ``id``, ``group``, ``subject``, ``question``, ``plan``, ``core``,
  ``record``, ``route``, ``missing``, ``rule``, ``marks``, ``tie_mark``,
  ``ordinal``, ``tie``, ``needs_ordinal``, ``excluded_from_tally``, ``fault``,
  ``conditions``, ``expect`` and ``source``; a field the row does not hold is
  ``None``, ``marks`` and ``conditions`` are tuples;
- ``expand(table)``: one ``baselines.interfaces.Question`` for every listed row of
  every group, every subject and every size at which the listed row is asked; a
  question holds ``size`` (None for a listed row asked once, else "small", "10k" or
  "100k"), and its ``id`` is unique across sizes.

A module that does not exist yet fails a test on an assertion (``need``), not on
an import.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

# --- Helpers (called in the body of a test, so a missing module is a failed test) ----

_NO_SIZE_FIELD = object()


def _size(question):
    """The size a question carries; a question without the field fails on an
    assertion."""
    size = getattr(question, "size", _NO_SIZE_FIELD)
    assert size is not _NO_SIZE_FIELD, f"question {question.id} has no size"
    return size


def _sizes_per_listed_row(questions) -> dict[str, list]:
    """The size list of each listed row as the expander gives it: the distinct sizes
    its questions carry, in order of first appearance."""
    sizes: dict[str, list] = {}
    for question in questions:
        found = sizes.setdefault(question.listed_row, [])
        if _size(question) not in found:
            found.append(_size(question))
    return sizes


def _differences_from_the_tools_rows(sizes_by_listed: dict, reference: dict) -> list[str]:
    """The listed rows whose size set differs from the tool's own, one text each; an
    empty list is agreement. The tie test and its known-bad reading both use it."""
    out = []
    for listed in sorted(set(sizes_by_listed) | set(reference)):
        got = set(sizes_by_listed.get(listed, ()))
        want = set(reference.get(listed, ()))
        if got != want:
            out.append(f"{listed}: expander {sorted(map(str, got))}, tool {sorted(map(str, want))}")
    return out


def _sizes_of_the_tools_own_rows(facts, acceptance_module) -> dict[str, set]:
    """What ``tests/acceptance/acceptance_rows.sized_params`` gives for the 370
    listed rows: one entry per size at which the tool's own row is asked (None when
    asked once)."""
    table = acceptance_module("table")
    acceptance_rows = acceptance_module("acceptance_rows")
    found: dict[str, set] = {}
    for number in range(1, 10):
        for param in acceptance_rows.sized_params(table.load_check(number)):
            row, _part, size = param.values
            if row.id in facts.core_check:
                found.setdefault(row.id, set()).add(size)
    return found


def _loaded(need_module):
    table10 = need_module("baselines.table10")
    table = table10.load_table10()
    return table10, table


def _expanded(need_module):
    table10, table = _loaded(need_module)
    return table10, table, table10.expand(table)


def _as_plain(value):
    """A loaded value as plain Python data, so that it is compared with the file's."""
    if isinstance(value, dict) or hasattr(value, "items"):
        return {k: _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


# --- The file as the table's own test code reads it ----------------------------------


def test_the_loader_is_held_to_the_counts_the_file_gives_and_the_frozen_counts(need_module, facts):
    """The counts the other tests rest on, taken from the file (and the same counts as
    taken from the table at 5b76911 and frozen before the run), and the loader's own
    counts against them."""
    table10, table = _loaded(need_module)
    assert (len(table.groups), len(table.rows)) == (70, 420)   # the file; a pass with every extension off reads 66 and 396
    assert len(table10.expand(table)) == 2346
    assert len(facts.groups) == facts.groups_count == 66
    assert len(facts.core_check) == facts.core_rows_count == 370
    assert len(facts.rows) == facts.groups_count * facts.subject_count
    assert facts.subjects == facts.subjects_in_readme_order
    assert sum(len(g["rows"]) for g in facts.groups) == 370
    # The questions of the per-size unit: 370 listed rows, 6 subjects, and the extra
    # sizes of the 17 sized listed rows (13 at one more size, 4 at two more).
    questions = len(facts.question_triples())
    assert questions == facts.questions_count == 2346
    assert questions == 370 * 6 + 6 * (13 * 1 + 4 * 2)
    sized = [listed for listed, sizes in facts.sizes_by_listed.items() if sizes != (None,)]
    assert len(sized) == facts.sized_listed_rows_count == 17


# --- Groups ----------------------------------------------------------------------------


def test_the_loader_reads_the_70_groups_in_file_order(need_module, facts):
    _, table = _loaded(need_module)
    assert len(table.groups) == 70
    assert [g.id for g in table.groups] == [g["id"] for g in facts.all_groups]
    for loaded, written in zip(table.groups, facts.all_groups):
        assert loaded.check == written["check"], loaded.id
        assert loaded.question == written["question"], loaded.id
        assert loaded.source == written["source"], loaded.id
        assert tuple(loaded.rows) == tuple(written["rows"]), loaded.id


def test_the_groups_rows_are_the_370_core_rows_and_the_nine_rows_of_the_generating_model_extension_each_once(need_module, facts):
    _, table = _loaded(need_module)
    listed = [row_id for group in table.groups for row_id in group.rows]
    assert len(listed) == 379
    assert not [i for i, n in Counter(listed).items() if n > 1], "a row is listed twice"
    extension = {i for i in facts.extension_ids if facts.extension_of[i] == "generating-model"}
    assert len(extension) == 9
    assert set(listed) == set(facts.core_check) | extension
    # The three rows of the content-ban extension (an extension that is not built) are in no group.
    assert not set(listed) & {i for i in facts.extension_ids if facts.extension_of[i] == "content-ban"}
    # The check a group names is the check its listed rows come from.
    checks = {**facts.core_check, **facts.extension_check}
    for group in table.groups:
        for row_id in group.rows:
            assert checks[row_id] == group.check, (group.id, row_id)


# --- Rows ------------------------------------------------------------------------------


def test_the_loader_reads_the_420_rows_with_every_field_the_file_holds(need_module, facts):
    _, table = _loaded(need_module)
    assert len(table.rows) == 420
    assert [r.id for r in table.rows] == [r["id"] for r in facts.all_rows]
    for loaded, written in zip(table.rows, facts.all_rows):
        where = loaded.id
        assert loaded.group == written["given"]["group"], where
        assert loaded.subject == written["given"]["subject"], where
        assert loaded.question == written["question"], where
        assert loaded.plan == written["plan"], where
        assert loaded.core is written["core"], where
        assert loaded.record == written["record"], where
        assert loaded.route == written.get("route"), where
        assert loaded.rule == written["rule"], where
        assert loaded.source == written["source"], where
        assert loaded.fault == written.get("fault"), where
        assert loaded.needs_ordinal is written["needs_ordinal"], where
        assert loaded.excluded_from_tally is written["excluded_from_tally"], where
        assert tuple(loaded.conditions) == tuple(written["given"]["conditions"]), where
        assert _as_plain(loaded.expect) == written["expect"], where


def test_every_group_has_one_row_for_each_of_the_six_subjects(need_module, facts):
    _, table = _loaded(need_module)
    for group in table.groups:
        subjects = [r.subject for r in table.rows if r.group == group.id]
        assert sorted(subjects) == sorted(facts.subjects_in_readme_order), group.id


def test_a_row_without_a_record_carries_its_missing_text_verbatim(need_module, facts):
    """Q4 (a): the record takes each row's `missing` text word for word; a row holds
    either a route or a missing text, never both and never neither."""
    _, table = _loaded(need_module)
    with_missing = 0
    for loaded, written in zip(table.rows, facts.all_rows):
        assert ("route" in written) != ("missing" in written), written["id"]
        assert loaded.missing == written.get("missing"), loaded.id
        if "missing" in written:
            with_missing += 1
            assert written["record"] == "none", written["id"]
            assert loaded.missing, loaded.id
    assert with_missing == 420 - sum(1 for r in facts.all_rows if "route" in r)
    assert with_missing > 0


def test_the_marks_come_from_both_the_marks_field_and_the_ordinal_block(need_module, facts):
    """`marks` holds two of the three marks; the third, the Dolma tie's, lives only in
    `ordinal.shown.mark`. A loader that read `marks` alone would lose it, and one
    that merged the two would put the tie's mark on an answer that did not use the
    tie, so the two are kept apart."""
    _, table = _loaded(need_module)
    seen_tie_marks = 0
    dolma_rows_with_no_marks_and_a_tie = 0
    for loaded, written in zip(table.rows, facts.all_rows):
        assert tuple(loaded.marks) == tuple(written["marks"]), loaded.id
        if "ordinal" in written:
            seen_tie_marks += 1
            assert loaded.tie_mark == written["ordinal"]["shown"]["mark"], loaded.id
            assert loaded.tie_mark == facts.mark_tie, loaded.id
            assert facts.mark_tie not in loaded.marks, loaded.id
            if not written["marks"]:
                dolma_rows_with_no_marks_and_a_tie += 1
                assert loaded.subject == "dolma", loaded.id
        else:
            assert loaded.tie_mark is None, loaded.id
    assert seen_tie_marks == sum(1 for r in facts.all_rows if "ordinal" in r) == 22
    assert dolma_rows_with_no_marks_and_a_tie == sum(
        1 for r in facts.all_rows if "ordinal" in r and not r["marks"]
    )
    assert dolma_rows_with_no_marks_and_a_tie == 4


def test_the_ordinal_and_tie_blocks_are_read_whole(need_module, facts):
    _, table = _loaded(need_module)
    ordinal_rows = [r for r in table.rows if r.ordinal is not None]
    tie_rows = [r for r in table.rows if r.tie is not None]
    assert len(ordinal_rows) == sum(1 for r in facts.all_rows if "ordinal" in r) == 22
    assert len(tie_rows) == sum(1 for r in facts.all_rows if "tie" in r) == 14
    for loaded, written in zip(table.rows, facts.all_rows):
        if "ordinal" in written:
            assert _as_plain(loaded.ordinal) == written["ordinal"], loaded.id
        else:
            assert loaded.ordinal is None, loaded.id
        if "tie" in written:
            assert _as_plain(loaded.tie) == written["tie"], loaded.id
        else:
            assert loaded.tie is None, loaded.id
    # A row needs an ordinal exactly when it carries the two readings.
    assert {r.id for r in table.rows if r.needs_ordinal} == {r.id for r in ordinal_rows}


def test_the_rows_left_out_of_the_tally_are_the_ones_the_readme_names(need_module, facts):
    """README: the trace and both combinations, for a question of Check 1 or 2, and
    no other row. The rule is applied here, from the group's check and the subject;
    the loader's flag is the file's own."""
    _, table = _loaded(need_module)
    flagged = {r.id for r in table.rows if r.excluded_from_tally}
    by_rule = {
        r.id for r in table.rows if facts.tally_excluded(r.group, r.subject)
    }
    assert flagged == by_rule
    assert len(flagged) == 7 * 3 == 21


# --- The questions ---------------------------------------------------------------------


def test_the_expander_gives_2346_questions_one_per_listed_row_subject_and_size(need_module, facts):
    table10, table, questions = _expanded(need_module)
    assert len(questions) == facts.questions_count == 2346
    triples = [(q.listed_row, q.subject, _size(q)) for q in questions]
    assert not [t for t, n in Counter(triples).items() if n > 1], "a question is put twice"
    assert sorted(triples, key=str) == sorted(facts.question_triples(), key=str)
    # Every listed row appears once for each of the six subjects at each size it is
    # asked at (the size list worked out by the test from the tool's own rows).
    per_row = Counter(q.listed_row for q in questions)
    assert set(per_row) == set(facts.core_check)
    for listed, count in per_row.items():
        assert count == 6 * len(facts.sizes_by_listed[listed]), listed
    # A question has an id of its own, a string, and no two share one, across sizes.
    ids = [q.id for q in questions]
    assert all(isinstance(i, str) and i for i in ids)
    assert len(set(ids)) == 2346


def test_the_questions_split_by_size_and_by_subject_as_the_unit_gives(need_module, facts):
    """2118 asked once, 102 at small, 24 at 10k, 102 at 100k; 391 for each subject."""
    _, _, questions = _expanded(need_module)
    by_size = Counter(_size(q) for q in questions)
    assert dict(by_size) == facts.questions_by_size
    assert by_size[None] == 2118
    assert (by_size["small"], by_size["10k"], by_size["100k"]) == (102, 24, 102)
    assert sum(by_size.values()) == 2346
    # The same split from the test's own size lists: 6 subjects for each listed row.
    counted = Counter()
    for sizes in facts.sizes_by_listed.values():
        for size in sizes:
            counted[size] += facts.subject_count
    assert counted == by_size
    # Every subject is asked the same 391 questions.
    by_subject = Counter(q.subject for q in questions)
    assert set(by_subject) == set(facts.subjects_in_readme_order)
    assert set(by_subject.values()) == {391}
    assert facts.questions_per_subject == 391
    # No other size word appears, and the small set is a size apart from the plan's.
    assert set(by_size) == {None, "small", "10k", "100k"}
    plan_sizes = {size for size in by_size if size not in (None, "small")}
    assert plan_sizes == {"10k", "100k"}


def test_the_sized_listed_rows_are_the_17_the_tools_own_rows_size(need_module, facts):
    """The 13 generated rows of Check 3 are asked at small and 100k, the 4 generated
    rows of Check 6 at small, 10k and 100k; every other listed row once. The edge:
    c03.precondition.eod-convention names the 100k set and is asked once; the c09.10k
    rows are asked once, at no size of their own."""
    _, _, questions = _expanded(need_module)
    expander = _sizes_per_listed_row(questions)
    sized = {listed: sizes for listed, sizes in expander.items() if sizes != [None]}
    assert len(sized) == 17
    check3 = {l: s for l, s in sized.items() if facts.core_check[l] == 3}
    check6 = {l: s for l, s in sized.items() if facts.core_check[l] == 6}
    assert len(check3) == 13 and len(check6) == 4 and len(sized) == 13 + 4
    assert all(sizes == ["small", "100k"] for sizes in check3.values())
    assert all(sizes == ["small", "10k", "100k"] for sizes in check6.values())
    assert all(facts.core_check[l] in (3, 6) for l in sized)
    # The edge, by name: it names a 100k set in its given and is still asked once.
    assert facts.row_by_listed_given(facts.eod_convention)["set"] == "100k"
    assert expander[facts.eod_convention] == [None]
    # The c09.10k rows are asked once.
    ten_k_rows = [l for l in expander if l.startswith("c09.10k.")]
    assert len(ten_k_rows) == 10
    assert all(expander[l] == [None] for l in ten_k_rows)
    # Each subject of a listed row is asked at the same sizes.
    sets_by_row_subject: dict = {}
    for q in questions:
        sets_by_row_subject.setdefault((q.listed_row, q.subject), []).append(_size(q))
    for (listed, subject), sizes in sets_by_row_subject.items():
        assert sizes == expander[listed], (listed, subject)


def test_the_expanders_size_lists_equal_the_tools_own_for_every_listed_row(
    need_module, facts, acceptance_modules
):
    """THE TIE TEST: for each of the 370 listed rows of Checks 1 to 9, the sizes the
    expander asks it at are the sizes `acceptance_rows.sized_params` gives (the same
    set; None when asked once)."""
    _, _, questions = _expanded(need_module)
    expander = _sizes_per_listed_row(questions)
    reference = _sizes_of_the_tools_own_rows(facts, acceptance_modules)
    assert set(reference) == set(facts.core_check) == set(expander)
    assert len(reference) == 370
    assert _differences_from_the_tools_rows(expander, reference) == []
    # The edge, by name.
    assert set(expander[facts.eod_convention]) == reference[facts.eod_convention] == {None}


def _every_row_naming_a_set_is_sized(facts, sets) -> dict[str, list]:
    """A KNOWN-BAD expander, for the test below only: it sizes every listed row whose
    own row names a set (small, then the sizes the set stands for)."""
    bad: dict[str, list] = {}
    for listed, sizes in facts.sizes_by_listed.items():
        named = facts.row_by_listed_given(listed).get("set")
        bad[listed] = ["small", *sets[named]] if named in sets else list(sizes)
    return bad


def test_the_tie_comparison_rejects_an_expander_that_sizes_every_row_naming_a_set(
    need_module, facts, acceptance_modules
):
    """The red of the tie test: an expander that sizes each row naming a set gives
    c03.precondition.eod-convention a small and a 100k question and the c09.10k rows a
    small and a 10k one; the comparison names them."""
    sets = {"100k": ("100k",), "generated": ("10k", "100k"), "10k": ("10k",)}
    reference = _sizes_of_the_tools_own_rows(facts, acceptance_modules)
    bad = _every_row_naming_a_set_is_sized(facts, sets)
    differences = _differences_from_the_tools_rows(bad, reference)
    assert any(d.startswith(facts.eod_convention + ":") for d in differences), differences
    assert any(d.startswith("c09.10k.") for d in differences), differences
    # What the tool's own rows give is not rejected.
    assert _differences_from_the_tools_rows(
        {l: list(s) for l, s in facts.sizes_by_listed.items()}, reference
    ) == []
    # A row dropped from the expander is a difference too.
    short = dict(bad)
    del short[facts.eod_convention]
    assert any(
        d.startswith(facts.eod_convention + ":")
        for d in _differences_from_the_tools_rows(short, reference)
    )
    # The same known-bad sizes put through the loader's table and the real expander:
    # what the expander gives from them is rejected by the same comparison.
    table10, table = _loaded(need_module)
    assert hasattr(table, "sizes"), "the table has no sizes per listed row"
    bad_table = dataclasses.replace(table, sizes={l: tuple(s) for l, s in bad.items()})
    through_the_expander = _sizes_per_listed_row(table10.expand(bad_table))
    assert _differences_from_the_tools_rows(through_the_expander, reference) == differences


def test_a_question_points_at_the_table_row_of_its_group_and_subject(need_module, facts):
    interfaces = need_module("baselines.interfaces")
    _, table, questions = _expanded(need_module)
    row_by_id = {r.id: r for r in table.rows}
    group_of_listed = {
        listed: group.id for group in table.groups for listed in group.rows
    }
    for q in questions:
        assert isinstance(q, interfaces.Question), q.id
        assert q.group == group_of_listed[q.listed_row], q.id
        row = row_by_id[q.row_id]
        assert (row.group, row.subject) == (q.group, q.subject), q.id
        assert q.check == facts.core_check[q.listed_row], q.id
        written = facts.row_by_id[q.row_id]
        assert q.group == written["given"]["group"], q.id
        assert q.subject == written["given"]["subject"], q.id


def test_147_questions_are_recorded_but_left_out_of_the_tally(need_module, facts):
    _, _, questions = _expanded(need_module)
    flagged = [q for q in questions if q.excluded_from_tally]
    assert len(flagged) == facts.excluded_count == 147
    by_rule = {
        (listed, subject, size)
        for listed, subject, size in facts.question_triples()
        if facts.tally_excluded(facts.group_of(listed), subject)
    }
    assert {(q.listed_row, q.subject, _size(q)) for q in flagged} == by_rule
    # Checks 1 and 2 are asked once, so the per-size unit leaves the 147 as they were.
    assert {_size(q) for q in flagged} == {None}
    # The flag a question carries is its row's.
    for q in questions:
        assert q.excluded_from_tally is facts.row_by_id[q.row_id]["excluded_from_tally"], q.id


def test_737_questions_have_a_route_and_1609_a_missing_text(need_module, facts):
    _, _, questions = _expanded(need_module)
    routed = [q for q in questions if q.route is not None]
    # The count and the triples from the file alone: every listed row of a group
    # whose row for a subject holds a route, once for that subject and each size.
    routed_in_file = {
        (listed, row["given"]["subject"], size)
        for row in facts.rows
        if "route" in row
        for listed in facts.group_by_id[row["given"]["group"]]["rows"]
        for size in facts.sizes_by_listed[listed]
    }
    assert len(routed_in_file) == facts.route_count == 737
    assert {(q.listed_row, q.subject, _size(q)) for q in routed} == routed_in_file
    assert len(routed) == 737
    # A row holds a route exactly when its record is not "none"; both readings of the
    # file give the same 737, and the rest record "none".
    assert all(("route" in row) == (row["record"] != "none") for row in facts.rows)
    recorded_none = [q for q in questions if facts.row_by_id[q.row_id]["record"] == "none"]
    assert len(recorded_none) == facts.no_record_count == 1609
    assert len(routed) + len(recorded_none) == 2346
    # The rest carry the row's `missing` text word for word.
    for q in questions:
        written = facts.row_by_id[q.row_id]
        assert q.route == written.get("route"), q.id
        assert q.missing == written.get("missing"), q.id
        if q.route is None:
            assert q.missing, q.id
    by_subject = Counter(q.subject for q in routed)
    expected = Counter()
    for q in questions:
        if "route" in facts.row_by_id[q.row_id]:
            expected[q.subject] += 1
    assert by_subject == expected
    assert sum(by_subject.values()) == 737


def test_a_question_carries_its_rows_marks_and_the_tie_mark_apart(need_module, facts):
    _, _, questions = _expanded(need_module)
    with_tie = 0
    for q in questions:
        written = facts.row_by_id[q.row_id]
        assert tuple(q.marks) == tuple(written["marks"]), q.id
        if "ordinal" in written:
            with_tie += 1
            assert q.needs_ordinal is True, q.id
            assert q.tie_mark == written["ordinal"]["shown"]["mark"] == facts.mark_tie, q.id
        else:
            assert q.needs_ordinal is False, q.id
            assert q.tie_mark is None, q.id
    expected = sum(
        len(facts.sizes_by_listed[listed])
        for written in facts.rows
        if "ordinal" in written
        for listed in facts.group_by_id[written["given"]["group"]]["rows"]
    )
    assert with_tie == expected
    assert with_tie > 0


def test_the_expansion_is_the_same_each_time(need_module):
    table10, table = _loaded(need_module)
    first = table10.expand(table)
    second = table10.expand(table10.load_table10())
    assert [q.id for q in first] == [q.id for q in second]
    assert first == second


# --- Step 1, commit 4a (18 Nov 2026): the readers take the extension rows a group lists ---------------------------------
# The owner's answer Q1 puts the cases of an extension into Check 10: a group of check10.toml lists rows of Checks 5, 6, 7 and 8 that are
# not core. Each reader of "the rows a group lists" must hold such a row. The row is made here from a real row of Check 5 with `core`
# false, so no table file changes.

def _with_an_extension_row(monkeypatch):
    """The acceptance table reader with one more row in Check 5: a copy of the first row, not core, under a new id."""
    import importlib  # noqa: PLC0415

    table10 = importlib.import_module("baselines.table10")
    acceptance = table10._acceptance_table()
    real = acceptance.load_check

    def load(number, folder):
        rows = list(real(number, folder))
        if number == 5:
            rows.append(dataclasses.replace(rows[0], id="c05.extension.fake-row", core=False))
        return rows

    monkeypatch.setattr(acceptance, "load_check", load)
    return table10


def test_the_given_of_an_extension_row_a_group_lists_is_read(monkeypatch):
    import importlib  # noqa: PLC0415

    table10 = _with_an_extension_row(monkeypatch)
    record = importlib.import_module("baselines.record")
    record._listed_givens.cache_clear()
    try:
        assert "c05.extension.fake-row" in record._listed_givens()
    finally:
        record._listed_givens.cache_clear()


def test_the_sizes_of_an_extension_row_a_group_lists_are_read(monkeypatch):
    table10 = _with_an_extension_row(monkeypatch)
    assert "c05.extension.fake-row" in table10.sizes_of_listed_rows(table10.DEFAULT_PATH.parent)


# --- An extension that is off is not asked (18 Nov 2026; plan :305, table README "An extension row is put only while its extension is on") ----


def test_a_row_of_an_extension_that_is_off_is_not_asked_and_an_extension_that_is_on_is_asked_with_its_own_rows_only(need_module):
    """The table of Check 10 holds the groups of the generating-model extension (core = false, extension named on every row of
    them). With no extension on, no question of those groups is made: the 2346 questions of the core rows; with the extension on, the
    questions of its nine listed rows are added, six subjects each, and nothing else."""
    table10, table = _loaded(need_module)
    off = table10.expand(table)
    assert len(off) == 2346
    on = table10.expand(table, extensions=("generating-model",))
    extra = [q for q in on if q.id not in {o.id for o in off}]
    assert len(on) == len(off) + len(extra) == 2400 and len(extra) == 54
    rows = {row.id: row for row in table.rows}
    assert {rows[q.row_id].extension for q in extra} == {"generating-model"}
    assert {rows[q.row_id].extension for q in off} == {None}
    # An extension that is not named stays off even when another is on.
    assert len(table10.expand(table, extensions=("content-ban",))) == 2346
