"""Every way an answer "does not contain" can be made goes through ``coverage.prove``
and a proof that checked every condition of plan 2.2 (plan 2.2, Decision 1, W6).

The older tests of ``test_coverage_answers.py`` close the doors of the answer type
(no constructor, a mint claimed once, a forged proof refused). The tests here try the
doors that stay open when the caller is a program with the public names of the
coverage module in hand: an ``Artifact`` made around the store, a ``Session`` or an
``IntegrityPass`` kept across a write or made by hand, a proof shown under another
artifact's name, and a scan of the source for the one place that names the negative.

A test that tries a manipulation accepts any refusal as the safe outcome and does
not name the way the tool refuses; it fails when the manipulation gives a proof, or
a "does not contain", that a fresh walk of the store does not give.
"""

from __future__ import annotations

import ast
import dataclasses
import sqlite3
from pathlib import Path

import pytest

import waterwhite
from waterwhite import coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Answer, Finding, Proof, ProofRequired

# What a refusal can be: a named error of the tool, or the type errors of a closed door.
REFUSALS = (errors.WaterwhiteError, ValueError, TypeError, RuntimeError)


def attempt(call):
    """(result, None) when the call returned, (None, error) when the tool refused."""
    try:
        return call(), None
    except REFUSALS as error:
        return None, error


@pytest.fixture
def request_a(world):
    return requests.withdraw(world.store, "A")


# --- The mint --------------------------------------------------------------------------


def test_a_second_claim_of_the_mint_from_the_coverage_module_is_refused():
    """Guards what already holds. The coverage module has claimed the mint by being
    imported; code that names itself as that module, as a reload would, still cannot
    take a second one (docs/coverage.md, "How does not contain is closed off", 2)."""
    assert coverage.prove is not None
    namespace = {"__name__": "waterwhite.coverage._second"}
    with pytest.raises(ProofRequired, match="already"):
        exec("from waterwhite import answers\nanswers.claim_proof_mint()", namespace)


# --- The name an answer shows ----------------------------------------------------------


def test_a_proof_cannot_be_shown_under_the_name_of_another_artifact(world, request_a):
    proof = coverage.prove(world.store, "R2", request_a)
    assert isinstance(proof, Proof)
    # The honest use keeps working: the name of the artifact the proof is about.
    honest = Answer.from_proof(proof, artifact="R2")
    assert (honest.answer, honest.to_json()["artifact"]) == (v.DOES_NOT_CONTAIN, "R2")
    # R1 contains A. An answer that says R1 does not contain A, built from the proof
    # about R2, is a wrong answer about R1.
    shown, _ = attempt(lambda: Answer.from_proof(proof, artifact="R1"))
    assert shown is None, (
        f"a proof about R2 was shown as the answer {shown.answer!r} about {shown.artifact!r}"
    )


# --- An artifact handed in -------------------------------------------------------------


def test_an_artifact_object_is_not_trusted_over_the_store(world, request_a):
    """The coverage module takes a name or an ``Artifact``. An ``Artifact`` whose kind
    is not the kind the store registered must not decide which rule walks the
    artifact: a shard walked as a file holds no file records and so no record in scope."""
    store = world.store
    wrong = []
    for name in ("shard-3", "E@6"):
        genuine = coverage.prove(store, name, request_a)
        assert isinstance(genuine, Finding) and genuine.answer == v.CONTAINS, name
        registered = store.resolve(name)
        for kind in (v.KIND_JSONL, v.KIND_SHARD, v.KIND_MIXTURE, v.KIND_CHECKPOINT):
            if kind == registered.kind:
                continue
            forged = dataclasses.replace(registered, kind=kind)
            result, _ = attempt(lambda: coverage.prove(store, forged, request_a))
            if result is None:
                continue
            said = v.DOES_NOT_CONTAIN if isinstance(result, Proof) else result.answer
            if said != genuine.answer:
                wrong.append((name, kind, said))
    assert wrong == []


# --- A session kept across a write -----------------------------------------------------


def test_a_session_does_not_outlive_a_write_to_the_store(world):
    """A session holds what a walk learned. After the store is written the walk is
    out of date: the session must not hand back a negative the store no longer gives."""
    store = world.store
    request = requests.withdraw(store, "B", records=["src/B:L1"])
    session = coverage.Session(store, request)
    before = coverage.prove(store, "gen/P", request, session=session)
    assert isinstance(before, Proof)
    # A file-level link from P's record to B: the record-level request about B can no
    # longer be decided for it (plan 3.2, 2.2(b)).
    store.add_source_link(
        world["gen/P"], "L1", "B/r1", v.FILE_LEVEL,
        "file family: paraphrases of passages that source B supplies", derivation=v.DERIVED_FILE,
    )
    fresh = coverage.prove(store, "gen/P", request)
    assert isinstance(fresh, Finding) and fresh.answer == v.MAY_CONTAIN
    stale, _ = attempt(lambda: coverage.prove(store, "gen/P", request, session=session))
    if stale is not None:
        said = v.DOES_NOT_CONTAIN if isinstance(stale, Proof) else stale.answer
        assert said == fresh.answer, "the old session gave the answer of the store before the write"


# --- An integrity pass made by hand, or kept across a fault ----------------------------


def break_map_digest(world, name: str) -> None:
    world.sql(("UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
               "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
               (world.binding_id(name),)))


def test_a_pass_made_by_hand_does_not_spare_the_loud_checks(world, request_a):
    """Four faults stop a query and print no answer, whatever the answer would have
    been (plan 2.2, 3.4). A pass the store's own check did not make says nothing about
    the store, whatever it claims to cover."""
    break_map_digest(world, "shard-3")
    with pytest.raises(errors.MapDigestMismatch):
        coverage.prove(world.store, "shard-3", request_a)
    whole = coverage.IntegrityPass(world.store, None)
    partial = coverage.IntegrityPass(world.store, (world["shard-3"], world["mix-2"]))
    for made in (whole, partial):
        for name in ("shard-3", "mix-2"):
            result, _ = attempt(lambda: coverage.prove(world.store, name, request_a, integrity=made))
            assert result is None, f"{name}: a hand-made pass let a query past a map digest fault"


def test_a_pass_does_not_outlive_a_fault_written_after_it(world, request_a):
    store = world.store
    whole = coverage.check_store(store)
    assert isinstance(coverage.prove(store, "shard-3", request_a, integrity=whole), Finding)
    # A fault written after the pass, through a second connection to the same store.
    side = sqlite3.connect(world.path)
    try:
        side.execute(
            "UPDATE binding SET declared_digest = 'f' || substr(declared_digest, 2) "
            "WHERE binding_id = ? AND substr(declared_digest, 1, 1) <> 'f'",
            (world.binding_id("shard-3"),))
        side.commit()
    finally:
        side.close()
    with pytest.raises(errors.MapDigestMismatch):
        coverage.prove(store, "shard-3", request_a)
    result, _ = attempt(lambda: coverage.prove(store, "shard-3", request_a, integrity=whole))
    assert result is None, "a pass made before the fault let the query past it"


# --- The scan of the source ------------------------------------------------------------

# The one module that spells the negative and builds it, and the files that hold the
# words themselves. The coverage package is the maker of the verdict "clear".
ALLOWED = ("answers.py", "vocabulary.py")
NAMES_ONLY_THE_MAKERS_USE = {
    "DOES_NOT_CONTAIN", "claim_proof_mint", "from_proof", "_is_minted", "CLEAR", "Outcome", "Tally",
}
PRIVATE_MODULES = ("_walk", "_checkpoint")
NEGATIVE_WORD = v.DOES_NOT_CONTAIN


def offences(source: str) -> list[tuple[int, str]]:
    """What a module outside the coverage package and ``answers.py`` must not hold:
    a name that builds the negative or the verdict "clear", the word of the negative as
    a string of its own, a call of the answer's constructor, or an import of the
    walk's private modules."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in NAMES_ONLY_THE_MAKERS_USE:
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in NAMES_ONLY_THE_MAKERS_USE:
            found.append((node.lineno, node.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.strip().lower() == NEGATIVE_WORD:
            found.append((node.lineno, repr(node.value)))
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name in ("Answer", "Proof", "Finding"):
                found.append((node.lineno, f"{name}(...)"))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in NAMES_ONLY_THE_MAKERS_USE:
                    found.append((node.lineno, "import of " + alias.name))
            modules = (node.module or "").split(".")
            if any(part in PRIVATE_MODULES for part in modules) or (
                    "coverage" in modules and any(a.name in PRIVATE_MODULES for a in node.names)):
                found.append((node.lineno, "import of " + (node.module or ".")))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(part in PRIVATE_MODULES for part in alias.name.split(".")):
                    found.append((node.lineno, "import of " + alias.name))
    return found


def test_the_scan_finds_each_way_to_build_the_negative_outside_the_makers():
    """The scan can fail: each planted line is one it must report."""
    planted = {
        "a name": "from waterwhite.vocabulary import DOES_NOT_CONTAIN\n",
        "the word": "answer = 'does not contain'\n",
        "the mint": "from waterwhite.answers import claim_proof_mint\n",
        "the shortcut": "result = Answer.from_proof(p, artifact='x')\n",
        "the verdict": "from waterwhite.coverage._walk import CLEAR\n",
        "the constructor": "x = Answer(artifact='a', answer='b')\n",
        "a private module": "from waterwhite.coverage import _walk\n",
    }
    for what, source in planted.items():
        assert offences(source) != [], what
    assert offences("from waterwhite import coverage\nx = coverage.answer(store, name, request)\n") == []


def test_no_module_outside_the_makers_builds_a_negative_or_the_verdict_clear():
    package = Path(waterwhite.__file__).parent
    found = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts[0] == "coverage" or path.name in ALLOWED and len(relative.parts) == 1:
            continue
        for line, what in offences(path.read_text(encoding="utf-8")):
            found.append((str(relative), line, what))
    assert found == []
