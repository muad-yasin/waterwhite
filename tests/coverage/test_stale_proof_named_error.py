"""A store written by another connection after a proof was made, and before the
answer is built from it, ends in the named error "store written during the query"
(docs/interfaces.md section 2), not in the refusal of a proof that was never made
(``ProofRequired``). A proof that was not made by ``coverage.prove`` stays
``ProofRequired`` whether or not the store was written.

The write is made through a second connection to the store file, as
``test_proof_only_construction.py`` makes its fault. A test that expects the named
error catches whatever the call raises and asserts its type, so another refusal is
a failed assertion.
"""

from __future__ import annotations

import sqlite3

import pytest

from waterwhite import coverage, errors, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Answer, Proof, ProofRequired

NAME = "store written during the query"


@pytest.fixture
def request_a(world):
    return requests.withdraw(world.store, "A")


def write_by_another_connection(path) -> None:
    side = sqlite3.connect(path)
    try:
        side.execute("CREATE TABLE IF NOT EXISTS written_after_the_proof (x INTEGER)")
        side.commit()
    finally:
        side.close()


def raised_by(call):
    """What the call raised, or None when it returned."""
    try:
        call()
    except BaseException as error:  # the type of what came out is the assertion
        return error
    return None


def assert_named(error, what: str) -> None:
    assert error is not None, f"{what}: nothing was raised; an answer was built from a stale proof"
    assert isinstance(error, errors.StoreWrittenDuringQuery), (
        f"{what}: raised {type(error).__name__}: {error}")
    assert (error.name, error.exit_code) == (NAME, 2)


def test_an_answer_built_from_a_proof_the_store_has_outlived_ends_in_the_named_error(world, request_a):
    proof = coverage.prove(world.store, "R2", request_a)
    assert isinstance(proof, Proof)
    # A proof that holds while the store is as it was.
    assert Answer.from_proof(proof, artifact="R2").is_negative
    write_by_another_connection(world.path)
    forged = object.__new__(Proof)
    for name in ("instance_id", "kind", "request", "scope", "notes", "passed_generation"):
        object.__setattr__(forged, name, getattr(proof, name))
    # A proof the mint did not make stays a refusal of its own kind after the write.
    assert isinstance(raised_by(lambda: Answer.from_proof(forged, artifact="R2")), ProofRequired)
    assert isinstance(raised_by(lambda: Answer(
        artifact="R2", instance=world["R2"], kind=v.KIND_JSONL, request=request_a,
        answer=v.DOES_NOT_CONTAIN)), ProofRequired)
    assert_named(raised_by(lambda: Answer.from_proof(proof, artifact="R2")), "from_proof")
    assert_named(raised_by(lambda: Answer.from_result(proof, artifact="R2")), "from_result")


def test_the_window_between_the_last_check_of_prove_and_the_answer_ends_in_the_named_error(
        world, request_a, monkeypatch):
    """The write falls after ``prove`` has returned its proof and before the answer is
    built, which is the window the check inside ``prove`` cannot see."""
    fired: list[int] = []
    original = coverage.prove

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if isinstance(result, Proof) and not fired:
            fired.append(1)
            write_by_another_connection(world.path)
        return result

    monkeypatch.setattr(coverage, "prove", wrapped)
    error = raised_by(lambda: coverage.answer(world.store, "R2", request_a))
    assert fired, "prove did not return a proof for R2, so no write was made in the window"
    assert_named(error, "coverage.answer")


def test_the_command_can_be_run_again_once_the_store_is_quiet(world, request_a):
    proof = coverage.prove(world.store, "R2", request_a)
    assert isinstance(proof, Proof)
    write_by_another_connection(world.path)
    assert_named(raised_by(lambda: Answer.from_result(proof, artifact="R2")), "the stale proof")
    again = coverage.answer(world.store, "R2", requests.withdraw(world.store, "A"))
    assert again.is_negative, "a new proof made after the write was refused too"
