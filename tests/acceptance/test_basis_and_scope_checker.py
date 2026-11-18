"""The check that every answer carries its basis and scope, put to documents written by
hand: a good document passes, and each document that lacks one thing is named. Nothing
here starts the tool.

The documents follow docs/interfaces.md section 6 and docs/cli.md ("query withdraw"):
the keys, and the lines of the text form.
"""

from __future__ import annotations

import copy

import acceptance_basis as basis
import pytest

BASIS = "whole-sample read assumed (F9, L9)"
REQUEST = {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}
CHAIN = [{"hop": "E/seg-2:step 7", "segment": "E/seg-2", "step": 7, "link": "consumed"},
         {"hop": "E/shard-3:D4", "artifact": "E/shard-3", "instance": "i", "record": "D4",
          "link": "recovered"}]
NOTE = "the parent list is process evidence"


def document(**fields) -> dict:
    base = {"artifact": "E@9", "instance": "i", "kind": "checkpoint", "request": REQUEST,
            "answer": "contains", "reason": None, "names": None, "names_detail": None,
            "failing": None, "candidates": [], "chain": CHAIN, "basis": BASIS,
            "notes": [NOTE], "untraced_steps": [["E/seg-2", 8]],
            "first_exposure": {"segment": "E/seg-2", "step": 7}, "level": "consumed",
            "scope": None}
    return base | fields


CONTAINS_TEXT = "\n".join([
    "E@9 (checkpoint) under withdraw A (members: A/r1)",
    "  contains",
    "  chain: E/seg-2:step 7 <- E/shard-3:D4",
    f"  basis: {BASIS}",
    "  first exposure: E/seg-2 step 7 (consumed)",
    "  untraced steps: E/seg-2 step 8",
    f"  note: {NOTE}"])
CONTAINS_EXPECT = {"answer": "contains", "basis": BASIS,
                   "first_exposure": {"segment": "E/seg-2", "step": 7},
                   "untraced_steps": [["E/seg-2", 8]], "notes": [NOTE]}

SCOPE = {"source": "A", "members": ["A/r1"], "records": [], "runs": ["E"],
         "steps": [["E/seg-1", 1, 6]], "generation_edges": [], "harness_logs": [],
         "walked": 3, "files_not_checked": []}
NEGATIVE_TEXT = "\n".join([
    "E@6 (checkpoint) under withdraw A (members: A/r1)",
    "  does not contain",
    "  scope: source A, members A/r1",
    "    steps: E/seg-1 1 to 6",
    "    records walked: 3"])
NEGATIVE_EXPECT = {"answer": "does not contain",
                   "scope": {"source": "A", "members": ["A/r1"],
                             "steps": [["E/seg-1", 1, 6]]}}

MAY_TEXT = "\n".join([
    "E@6 (checkpoint) under withdraw A (members: A/r1)",
    "  may contain: step untraced",
    "  failing link: condition e, segment E/seg-1, step 3"])
MAY_EXPECT = {"answer": "may contain", "names": "step untraced"}


def negative(**fields) -> dict:
    return document(artifact="E@6", answer="does not contain", chain=[], basis=None,
                    notes=[], untraced_steps=[], first_exposure=None, level=None,
                    scope=SCOPE) | fields


def may(**fields) -> dict:
    return document(artifact="E@6", answer="may contain", names="step untraced",
                    failing={"condition": "e", "segment": "E/seg-1", "step": 3}, chain=[],
                    basis=None, notes=[], untraced_steps=[["E/seg-1", 3]],
                    first_exposure=None, level=None) | fields


def test_documents_that_carry_what_their_kind_of_answer_carries_pass():
    assert basis.problems(document(), CONTAINS_TEXT, CONTAINS_EXPECT, "E/seg-2") == []
    assert basis.problems(negative(), NEGATIVE_TEXT, NEGATIVE_EXPECT, "E/seg-1") == []
    assert basis.problems(may(), MAY_TEXT, MAY_EXPECT, "E/seg-1") == []


def drop(doc: dict, key: str) -> dict:
    return {k: v for k, v in doc.items() if k != key}


def swapped(doc: dict) -> dict:
    items = list(doc.items())
    items[0], items[1] = items[1], items[0]
    return dict(items)


def without_line(text: str, start: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(start))


BAD = {
    "contains with no basis": (document(basis=None), CONTAINS_TEXT, CONTAINS_EXPECT, "basis"),
    "contains with another basis": (document(basis="operator note"), CONTAINS_TEXT,
                                    CONTAINS_EXPECT, "basis"),
    "contains with no level": (document(level=None), CONTAINS_TEXT, CONTAINS_EXPECT, "level"),
    "contains with no first exposure": (document(first_exposure=None), CONTAINS_TEXT,
                                        CONTAINS_EXPECT, "first exposure"),
    "contains with another first exposure": (
        document(first_exposure={"segment": "E/seg-2", "step": 9}), CONTAINS_TEXT,
        CONTAINS_EXPECT, "first exposure"),
    "contains whose untraced steps are not listed": (
        document(untraced_steps=[]), CONTAINS_TEXT, CONTAINS_EXPECT, "untraced"),
    "contains with no chain": (document(chain=[]), CONTAINS_TEXT, CONTAINS_EXPECT, "chain"),
    "contains whose level says inherited for its own segment": (
        document(level="inherited"), CONTAINS_TEXT, CONTAINS_EXPECT, "level"),
    "contains that also holds a scope": (document(scope=SCOPE), CONTAINS_TEXT,
                                         CONTAINS_EXPECT, "scope"),
    "contains whose text omits the basis": (document(), without_line(CONTAINS_TEXT, "  basis"),
                                            CONTAINS_EXPECT, "text"),
    "contains whose text omits the first exposure": (
        document(), without_line(CONTAINS_TEXT, "  first exposure"), CONTAINS_EXPECT, "text"),
    "contains whose text omits the untraced steps": (
        document(), without_line(CONTAINS_TEXT, "  untraced"), CONTAINS_EXPECT, "text"),
    "contains whose text omits a note": (document(), without_line(CONTAINS_TEXT, "  note"),
                                         CONTAINS_EXPECT, "text"),
    "a document that lacks a key": (drop(document(), "level"), CONTAINS_TEXT, CONTAINS_EXPECT,
                                    "keys"),
    "a document whose keys are in another order": (swapped(document()), CONTAINS_TEXT,
                                                   CONTAINS_EXPECT, "keys"),
    "another answer than the row expects": (document(), CONTAINS_TEXT,
                                            CONTAINS_EXPECT | {"answer": "may contain"},
                                            "answer"),
    "negative with no scope": (negative(scope=None), NEGATIVE_TEXT, NEGATIVE_EXPECT, "scope"),
    "negative whose scope has no steps key": (
        negative(scope={k: v for k, v in SCOPE.items() if k != "steps"}), NEGATIVE_TEXT,
        NEGATIVE_EXPECT, "scope"),
    "negative whose scope names another source": (
        negative(scope=SCOPE | {"source": "B"}), NEGATIVE_TEXT, NEGATIVE_EXPECT, "scope"),
    "negative whose scope steps differ": (
        negative(scope=SCOPE | {"steps": [["E/seg-1", 1, 5]]}), NEGATIVE_TEXT, NEGATIVE_EXPECT,
        "scope"),
    "negative whose text omits the scope": (negative(), without_line(NEGATIVE_TEXT, "  scope"),
                                            NEGATIVE_EXPECT, "text"),
    "negative that carries a basis": (negative(basis=BASIS), NEGATIVE_TEXT, NEGATIVE_EXPECT,
                                      "basis"),
    "may contain with a basis": (may(basis=BASIS), MAY_TEXT, MAY_EXPECT, "basis"),
    "may contain with no named cause": (may(names=None), MAY_TEXT, MAY_EXPECT, "names"),
    "may contain with no failing link": (may(failing=None), MAY_TEXT, MAY_EXPECT, "failing"),
}


@pytest.mark.parametrize(("doc", "text", "expect", "word"), BAD.values(), ids=list(BAD))
def test_a_document_that_lacks_one_thing_is_named(doc, text, expect, word):
    own = "E/seg-2" if doc["answer"] == "contains" else "E/seg-1"
    found = basis.problems(copy.deepcopy(doc), text, expect, own)
    assert found, "the check let the document pass"
    assert any(word in line for line in found), found
