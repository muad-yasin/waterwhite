"""The structure of the hand-written table (plan 5.1, 5.5; DECISIONS.md D-10).

These tests read ``fixture/table/check01.toml`` to ``check10.toml`` as files and
ask nothing of the tool: every file parses, every row has the keys
``fixture/table/README.md`` gives it, row ids are unique, every answer word,
reason and named cause in an ``expect`` block is a string the tool's vocabulary
defines, every ``pending:<key>`` names a value the fixture spec holds, and
``check10.toml`` groups the core rows of Checks 1 to 9 as its section of the
README says.

They do not check that an expected answer is right. That is the table's own
claim, and the checks of plan 5.5 put it to the tool.

The one file of the tool read here is ``waterwhite/vocabulary.py``: the table is
compared with the tool's spelling of its fixed words, never with an answer the
tool computed. Error names (``expect.error``) are not compared here.

Fast: standard library and that one module; no Megatron.
"""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from waterwhite import vocabulary

REPO = Path(__file__).resolve().parents[2]
TABLE = REPO / "fixture" / "table"
SPEC_FILE = REPO / "fixture" / "spec" / "fixture_spec.toml"
README = TABLE / "README.md"

CHECK_FILES = tuple(f"check{n:02d}.toml" for n in range(1, 11))
ANSWER_FILES = CHECK_FILES[:9]  # Checks 1 to 9: rows that ask the tool
BASELINE_FILE = CHECK_FILES[9]  # Check 10: the reading rules for the baselines

# README section 1: the keys of a row, and what `source` starts with.
ROW_KEYS = ("id", "question", "plan", "core", "given", "expect", "source")
ROW_OPTIONAL_KEYS = ("expect_under", "rule", "extension")
SOURCE_CATEGORIES = (
    "hand-written table",
    "the oracle's log",
    "the plant list",
    "the withheld files",
    "a from-scratch reference",
    "the replica's records",
)
# README sections 3, 4 and 8, and "Core result".
REQUESTS = ("withdraw A", "withdraw C", "withdraw A, record X@A", "withdraw B, record X@B", "withdraw W", "ban X")
STORES = ("main", "rr", "swap", "no-capture", "import", "scratch-500", "scratch-notokens")
FALLBACKS = ("F1", "F3", "F5", "F7", "cyclic-wrapper-route")
# The twelve extension rows: these seven by name, and the five rows `c07.R3.*`.
EXTENSION_ROWS = {
    "c05.g.model-entry-removed.on.E/shard-3": "generating-model",
    "c05.g.model-entry-removed.on.gen/P": "generating-model",
    "c05.g.model-entry-removed.on.gen/T": "generating-model",
    "c05.withdraw-C.on.E/shard-3": "generating-model",
    "c07.ban.R1": "content-ban",
    "c07.ban.R2": "content-ban",
    "c07.ban.scope": "content-ban",
}

# README, "Check 10".
SUBJECTS = ("manifest", "trace", "dolma", "originblame", "dolma+trace", "originblame+trace")
CONDITIONS = (
    "sandbox",
    "relative paths",
    "row count checked",
    "one process in packed order",
    "one revoke per fresh store",
    "written while the build runs",
    "the join's own position output",
)
GRADES = ("right", "wrong", "cannot answer")
GRADE_KEYS = {"right": "right", "wrong": "wrong", "cannot answer": "cannot_answer"}
GROUP_KEYS = ("id", "check", "question", "rows", "source")
BASELINE_ROW_KEYS = ROW_KEYS + ("record", "rule", "marks", "excluded_from_tally", "needs_ordinal")
BASELINE_OPTIONAL_KEYS = ("missing", "route", "fault", "ordinal", "tie", "extension", "repeats")


def _vocabulary(prefix: str) -> frozenset[str]:
    return frozenset(v for k, v in vars(vocabulary).items() if k.startswith(prefix) and isinstance(v, str))


NAMED_CAUSES = _vocabulary("NAMES_")
BASES = _vocabulary("BASIS_")
NOTES = _vocabulary("NOTE_")
# What each answer key of an expect block may hold (README section 6).
VOCABULARY = {
    "answer": frozenset(vocabulary.ANSWERS),
    "names": NAMED_CAUSES,
    "names_detail": frozenset(vocabulary.TRACE_MISSING_REASONS),
    # "reason" is the reason beside "not covered"; in a row about a replay or a
    # drop it is that reason, also one of the tool's fixed words
    "reason": frozenset(vocabulary.NOT_COVERED_REASONS + vocabulary.TRACE_MISSING_REASONS + vocabulary.DROP_REASONS),
    "basis": BASES,
    "notes": NOTES,
    "action": frozenset(vocabulary.ACTIONS + (vocabulary.UNRESOLVED,)),
    "label": frozenset(vocabulary.LABELS),
    "derivation": frozenset(vocabulary.DERIVATIONS),
    "verification": frozenset(vocabulary.VERIFICATIONS),
}


# --- Reading the files ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables() -> dict[str, dict]:
    """Every table file, parsed. A file that does not parse fails here, for every test."""
    out = {}
    for name in CHECK_FILES:
        with open(TABLE / name, "rb") as f:
            out[name] = tomllib.load(f)
    return out


@pytest.fixture(scope="module")
def answer_rows(tables) -> list[tuple[str, dict]]:
    """(file, row) for every row of Checks 1 to 9."""
    return [(name, row) for name in ANSWER_FILES for row in tables[name]["row"]]


@pytest.fixture(scope="module")
def baseline(tables) -> dict:
    return tables[BASELINE_FILE]


@pytest.fixture(scope="module")
def spec() -> dict:
    with open(SPEC_FILE, "rb") as f:
        return tomllib.load(f)


def dicts_in(node):
    """Every table (dict) inside a value, the value itself included, at any depth."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from dicts_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from dicts_in(value)


def strings_in(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from strings_in(value)
    elif isinstance(node, list):
        for value in node:
            yield from strings_in(value)


def starts_with_a_category(source: str) -> bool:
    """A row's `source` starts with one of the categories of README section 1."""
    return any(source == c or source.startswith((c + ":", c + " ", c + ",")) for c in SOURCE_CATEGORIES)


def names_categories_only(source: str) -> bool:
    """In check10.toml a `source` starts with the categories the rows of its group
    name: one, or several joined by "; ", then a colon."""
    head, colon, _ = source.partition(":")
    return bool(colon) and all(part in SOURCE_CATEGORIES for part in head.split("; "))


# --- Every file ---------------------------------------------------------------------------------------


def test_the_table_is_ten_files_and_each_parses(tables):
    assert sorted(p.name for p in TABLE.glob("*.toml")) == sorted(CHECK_FILES)
    for name in ANSWER_FILES:
        assert set(tables[name]) == {"row"} and tables[name]["row"], name
    assert set(tables[BASELINE_FILE]) == {"group", "row"}


def test_row_ids_are_unique_over_all_files(tables):
    ids = Counter(row["id"] for name in CHECK_FILES for row in tables[name]["row"])
    assert [i for i, n in ids.items() if n > 1] == []
    # a group has an id of its own kind; it may read like a row's (group `c01.E@9`
    # lists the one row `c01.E@9`)
    groups = Counter(group["id"] for group in tables[BASELINE_FILE]["group"])
    assert [i for i, n in groups.items() if n > 1] == []


def test_no_file_holds_a_long_dash_or_a_local_path():
    local_path = "/" + "home" + "/"  # spelled in parts, so this file holds no such path either
    for path in sorted(TABLE.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2014) not in text and chr(0x2013) not in text, path.name
        assert local_path not in text and "\t" not in text, path.name


# --- Rows of Checks 1 to 9 ------------------------------------------------------------------------------


def test_every_row_has_the_keys_of_a_row(answer_rows):
    for name, row in answer_rows:
        where = f"{name}: {row.get('id')!r}"
        assert set(ROW_KEYS) <= set(row), where
        assert set(row) <= set(ROW_KEYS + ROW_OPTIONAL_KEYS), where
        for key in ("id", "question", "plan", "source"):
            assert isinstance(row[key], str) and row[key].strip() == row[key] != "", f"{where}: {key}"
        assert isinstance(row["core"], bool), where
        assert isinstance(row["given"], dict) and row["given"], where
        assert isinstance(row["expect"], dict) and row["expect"], where
        # a row's id starts with its check: c08. in check08.toml
        assert row["id"].startswith(f"c{name[5:7]}."), where
        assert starts_with_a_category(row["source"]), f"{where}: source starts with no category of README section 1"
        if "rule" in row:
            assert isinstance(row["rule"], str) and row["rule"], where


def test_an_extension_row_is_one_of_the_twelve_and_names_its_extension(answer_rows):
    extension = {row["id"]: row.get("extension") for _, row in answer_rows if not row["core"]}
    r3 = sorted(i for i in extension if i.startswith("c07.R3."))
    assert len(extension) == 12 and len(r3) == 5
    assert all(extension[i] == "generating-model" for i in r3)
    assert {i: e for i, e in extension.items() if not i.startswith("c07.R3.")} == EXTENSION_ROWS
    # a core row names no extension
    assert [row["id"] for _, row in answer_rows if row["core"] and "extension" in row] == []


def test_given_names_requests_stores_and_fallbacks_the_readme_defines(answer_rows):
    for name, row in answer_rows:
        where = f"{name}: {row['id']!r}"
        given = row["given"]
        if "request" in given:
            assert given["request"] in REQUESTS, where
        if "store" in given:
            assert given["store"] in STORES, where
        under = row.get("expect_under", {})
        assert isinstance(under, dict) and set(under) <= set(FALLBACKS), where
        assert all(isinstance(value, dict) and value for value in under.values()), where


def test_every_answer_word_reason_and_named_cause_is_the_vocabularys(answer_rows):
    checked = Counter()
    for name, row in answer_rows:
        where = f"{name}: {row['id']!r}"
        for block in dicts_in([row["expect"], row.get("expect_under", {})]):
            for key, allowed in VOCABULARY.items():
                if key not in block:
                    continue
                values = block[key] if isinstance(block[key], list) else [block[key]]
                for value in values:
                    assert isinstance(value, str) and value in allowed, f"{where}: {key} = {value!r}"
                    checked[key] += 1
            # the reason beside "not covered" is one of its two; a named cause and
            # its detail belong to "may contain"
            if block.get("answer") == vocabulary.NOT_COVERED and "reason" in block:
                assert block["reason"] in vocabulary.NOT_COVERED_REASONS, where
            if "names" in block and "answer" in block:
                assert block["answer"] == vocabulary.MAY_CONTAIN, where
            if "names_detail" in block:
                assert block.get("names") == vocabulary.NAMES_TRACE_MISSING, where
    # the walk found the words it is there for: every answer and every kind of cause
    assert checked["answer"] > 250 and checked["names"] > 50 and checked["reason"] >= 8
    assert set(checked) == set(VOCABULARY)


def test_every_pending_value_names_a_value_the_spec_holds(tables, spec):
    """`"pending:<key>"` stands for the spec item `<key>` (README section 5). Since
    the seed search every such key holds a value, and the spec lists none as
    pending."""
    assert spec["pending"] == {"items": []}
    seen = set()
    for name in CHECK_FILES:
        for row in tables[name]["row"]:
            for text in strings_in(row):
                if "pending:" not in text:
                    continue
                assert text.startswith("pending:"), f"{name}: {row['id']!r}: {text!r}"
                key = text[len("pending:"):]
                node = spec
                for part in key.split("."):
                    assert isinstance(node, dict) and part in node, f"{name}: {row['id']!r}: the spec has no {key!r}"
                    node = node[part]
                assert node not in ("", [], {}, None), f"{name}: {row['id']!r}: the spec's {key!r} is empty"
                seen.add(key)
    assert {"positions.T", "positions.shared_token.record", "positions.K.first_read_of_T.step", "tokenizer.eod_in_span"} <= seen


# --- Check 10 --------------------------------------------------------------------------------------------


def test_a_group_has_its_keys_and_lists_rows_of_its_check(baseline, answer_rows):
    core_of = {row["id"]: name for name, row in answer_rows}  # core rows and extension rows
    for group in baseline["group"]:
        where = f"group {group.get('id')!r}"
        assert set(group) == set(GROUP_KEYS), where
        assert isinstance(group["check"], int) and 1 <= group["check"] <= 9, where
        assert group["id"].startswith(f"c{group['check']:02d}."), where
        assert isinstance(group["question"], str) and group["question"], where
        assert names_categories_only(group["source"]), where
        rows = group["rows"]
        assert isinstance(rows, list) and rows and len(set(rows)) == len(rows), where
        for row_id in rows:
            assert core_of.get(row_id) == f"check{group['check']:02d}.toml", f"{where}: {row_id!r}"


def test_every_row_is_in_exactly_one_group_except_the_content_ban(baseline, answer_rows):
    """README, "Groups": every core row of Checks 1 to 9 is in exactly one group, with no
    exception, and so is every row of the generating-model extension (the subjects have no
    switch, so its cases are in the comparison, as groups of their own); the three rows of the
    content-ban extension are in none."""
    listed = Counter(row_id for group in baseline["group"] for row_id in group["rows"])
    assert [i for i, n in listed.items() if n > 1] == []
    unlisted = {row["id"] for _, row in answer_rows if not row["core"] and row.get("extension") == "content-ban"}
    assert len(unlisted) == 3
    every = {row["id"] for _, row in answer_rows}
    assert set(listed).isdisjoint(unlisted)  # the rows of an extension that is not built are in no group
    assert sorted(every - unlisted - set(listed)) == []  # no other row is left out
    assert set(listed) == every - unlisted


def test_every_group_is_asked_of_the_six_subjects(baseline):
    groups = {group["id"]: group for group in baseline["group"]}
    by_group: dict[str, list[str]] = {}
    for row in baseline["row"]:
        where = f"{row.get('id')!r}"
        assert set(BASELINE_ROW_KEYS) <= set(row), where
        assert set(row) <= set(BASELINE_ROW_KEYS + BASELINE_OPTIONAL_KEYS), where
        given = row["given"]
        assert set(given) == {"subject", "group", "conditions"}, where
        assert given["group"] in groups and given["subject"] in SUBJECTS, where
        assert row["id"] == f"c10.{given['group']}.{given['subject']}", where
        assert isinstance(given["conditions"], list) and given["conditions"], where
        assert set(given["conditions"]) <= set(CONDITIONS), where
        assert (row["core"] is True and "extension" not in row) or (row["core"] is False and row.get("extension") == "generating-model"), where
        assert row["plan"].startswith("5.5 Check 10"), where
        assert row["question"] == groups[given["group"]]["question"], where
        assert names_categories_only(row["source"]), where
        assert isinstance(row["rule"], str) and row["rule"], where
        assert isinstance(row["marks"], list) and all(isinstance(m, str) and m for m in row["marks"]), where
        assert isinstance(row["excluded_from_tally"], bool) and isinstance(row["needs_ordinal"], bool), where
        by_group.setdefault(given["group"], []).append(given["subject"])
    assert sorted(by_group) == sorted(groups)
    assert all(subjects == list(SUBJECTS) for subjects in by_group.values()), "six rows per group, in the README's order"
    assert len(baseline["row"]) == 6 * len(groups)


def test_a_baseline_row_names_its_record_or_what_is_missing_and_no_expected_grade(baseline):
    groups = {group["id"]: group for group in baseline["group"]}
    for row in baseline["row"]:
        where = f"{row['id']!r}"
        expect = row["expect"]
        one_of = expect["one_of"]
        assert isinstance(one_of, list) and one_of and one_of == [g for g in GRADES if g in one_of], where
        assert set(expect) <= {"one_of", *GRADE_KEYS.values()}, where  # no expected grade
        assert isinstance(row["record"], str) and row["record"], where
        if row["record"] == "none":
            # the subject lacks the record: the row can only be graded "cannot answer"
            assert one_of == ["cannot answer"] and set(expect) == {"one_of"}, where
            assert isinstance(row.get("missing"), str) and row["missing"], where
            assert "route" not in row, where
        else:
            assert "missing" not in row and isinstance(row.get("route"), str) and row["route"], where
            assert "right" in one_of and "wrong" in one_of, where
            assert set(expect) == {"one_of", *(GRADE_KEYS[g] for g in one_of)}, where
            assert all(isinstance(expect[GRADE_KEYS[g]], str) and expect[GRADE_KEYS[g]] for g in one_of), where
        subject = row["given"]["subject"]
        assert ("ordinal" in row) == row["needs_ordinal"], where
        if row["needs_ordinal"]:
            assert subject in ("dolma", "dolma+trace"), where
        if "tie" in row:
            assert subject in ("originblame", "originblame+trace"), where
        if "fault" in row:
            assert groups[row["given"]["group"]]["check"] in (4, 5), where


def test_the_readme_lists_the_groups_with_their_row_counts(baseline):
    """README, "Groups": one table line per group with its number of rows and
    its question, and one sentence with the totals."""
    text = README.read_text(encoding="utf-8")
    lines = re.findall(r"^\| `(c\d\d\.[^`]+)` \| (\d+) \| (.+) \|$", text, flags=re.MULTILINE)
    in_readme = {group: (int(count), question) for group, count, question in lines}
    in_file = {group["id"]: (len(group["rows"]), group["question"]) for group in baseline["group"]}
    assert in_readme == in_file
    totals = re.search(r"(\d+) groups cover (\d+) listed rows \([^)]*\); with six subjects each, `check10.toml` has (\d+) rows", text)
    assert totals is not None
    listed = sum(len(group["rows"]) for group in baseline["group"])
    assert tuple(map(int, totals.groups())) == (len(baseline["group"]), listed, len(baseline["row"]))


def repeat_problems(baseline_rows, groups, answer_ids):
    """The rows that break the README's rule for a repeat ("Marks and the tally"): a row left out of the tally outside Checks 1 and 2
    names its counterpart in `repeats`, the counterpart is a row of Checks 1 to 9, and `repeats` on a row the tally keeps is no key."""
    found = []
    for row in baseline_rows:
        group = groups[row["given"]["group"]]
        left_out, names = row["excluded_from_tally"], "repeats" in row
        if names and not left_out:
            found.append(f"{row['id']}: repeats on a row the tally keeps")
        if left_out and group["check"] not in (1, 2) and not names:
            found.append(f"{row['id']}: left out of the tally outside Checks 1 and 2 and names no counterpart")
        if names and row["repeats"] not in answer_ids:
            found.append(f"{row['id']}: repeats {row['repeats']!r}, which is no row of Checks 1 to 9")
    return found


def test_a_row_left_out_as_a_repeat_names_its_counterpart(baseline, answer_rows):
    groups = {group["id"]: group for group in baseline["group"]}
    answer_ids = {row["id"] for _, row in answer_rows}
    assert repeat_problems(baseline["row"], groups, answer_ids) == []


def test_the_repeat_rule_refuses_a_row_that_breaks_it():
    """Against a known-bad input: the three ways to break the rule are each found, and the two rows that keep it are not."""
    groups = {"g1": {"check": 5}, "g2": {"check": 1}}
    rows = [
        {"id": "a", "given": {"group": "g1"}, "excluded_from_tally": True},
        {"id": "b", "given": {"group": "g1"}, "excluded_from_tally": False, "repeats": "c05.unmutated.R1"},
        {"id": "c", "given": {"group": "g1"}, "excluded_from_tally": True, "repeats": "c05.nothing"},
        {"id": "d", "given": {"group": "g2"}, "excluded_from_tally": True},
        {"id": "e", "given": {"group": "g1"}, "excluded_from_tally": True, "repeats": "c05.unmutated.R1"},
    ]
    assert [p.split(":")[0] for p in repeat_problems(rows, groups, {"c05.unmutated.R1"})] == ["a", "b", "c"]
