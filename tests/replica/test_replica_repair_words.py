"""The words of the replica's repair command are the tool's (docs/interfaces.md
section 1, 9.5).

The replica does not import the tool's vocabulary outside its capture hooks, so
``replica/repair.py`` spells the thirteen words it reads (the two form words, the
successor prefix, the five record action words, the three artifact action words and the
two kinds) and this test reads them with ``ast``, never by importing the replica, and
compares them with ``src/waterwhite/vocabulary.py``.

The replica's two action tuples are the vocabulary's ``RECORD_ACTIONS`` and
``ARTIFACT_ACTIONS`` each plus ``UNRESOLVED``: a record or an artifact the report lists
as unresolved is a word the replica reads and the vocabulary keeps apart.

This test is green on the tree it was written for, by design: the words are equal.
Its red is shown by pointing it at a copy of ``repair.py`` with one word changed:
``WATERWHITE_REPAIR_SOURCE`` names the file it reads instead of ``replica/repair.py``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

from waterwhite import vocabulary as v

REPO = Path(__file__).resolve().parents[2]
REPAIR = Path(os.environ.get("WATERWHITE_REPAIR_SOURCE") or REPO / "replica" / "repair.py")

# The replica's name for each word and the vocabulary's name for the same word.
WORDS = {
    "REPORT_FORM": "FORM_REPORT",
    "SIDES_FORM": "FORM_SIDES",
    "SUCCESSOR_PREFIX": "SUCCESSOR_PREFIX",
    "ACTION_KEEP": "ACTION_KEEP",
    "ACTION_RE_SOURCE": "ACTION_RE_SOURCE",
    "ACTION_PROMOTE": "ACTION_PROMOTE",
    "ACTION_RETIRE": "ACTION_RETIRE",
    "ACTION_UNRESOLVED": "UNRESOLVED",
    "ARTIFACT_KEEP": "ACTION_KEEP",
    "ARTIFACT_REWRITE": "ACTION_REWRITE",
    "ARTIFACT_UNRESOLVED": "UNRESOLVED",
    "KIND_JSONL": "KIND_JSONL",
    "KIND_SHARD": "KIND_SHARD",
}
TUPLES = {"RECORD_ACTIONS": v.RECORD_ACTIONS + (v.UNRESOLVED,),
          "ARTIFACT_ACTIONS": v.ARTIFACT_ACTIONS + (v.UNRESOLVED,)}


def module_values(path: Path) -> dict:
    """The values the module assigns at its top level to a name, worked out from the
    syntax tree alone: a constant, a tuple of them, or a name assigned before."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = eval(  # noqa: S307 - constants, tuples and names only
                    compile(ast.Expression(node.value), str(path), "eval"), {"__builtins__": {}}, dict(found))
            except Exception:
                continue
    return found


def test_the_thirteen_words_of_the_replica_are_the_vocabularys():
    held = module_values(REPAIR)
    missing = [name for name in WORDS if name not in held]
    assert not missing, f"{REPAIR} does not assign {missing} to a constant"
    assert len(WORDS) == 13
    for name, word in WORDS.items():
        assert held[name] == getattr(v, word), f"{name} is {held[name]!r}, the vocabulary's {word} is {getattr(v, word)!r}"


def test_the_action_tuples_of_the_replica_are_the_vocabularys_plus_unresolved():
    held = module_values(REPAIR)
    for name, expected in TUPLES.items():
        assert held.get(name) == expected, f"{name} is {held.get(name)!r}, expected {expected!r}"
