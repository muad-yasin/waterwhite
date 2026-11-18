"""The answer type: "does not contain" exists only around a proof that
``coverage.prove`` made (plan W6, Decision 1)."""

from __future__ import annotations

import copy
import dataclasses
import pickle

import pytest

from waterwhite import answers, coverage, requests
from waterwhite import vocabulary as v
from waterwhite.answers import Answer, Failing, Finding, Hop, Proof, ProofRequired, Scope


@pytest.fixture
def request_a(world):
    return requests.withdraw(world.store, "A")


@pytest.fixture
def proof(world, request_a):
    made = coverage.prove(world.store, "R2", request_a)
    assert isinstance(made, Proof)
    return made


def negative(world, request, **changes):
    """A hand-built "does not contain" about R2, with no proof unless one is given."""
    fields = dict(artifact="R2", instance=world["R2"], kind=v.KIND_JSONL, request=request,
                  answer=v.DOES_NOT_CONTAIN)
    fields.update(changes)
    return Answer(**fields)


def test_does_not_contain_cannot_be_built_without_a_proof(world, request_a):
    with pytest.raises(ProofRequired):
        negative(world, request_a)


def test_a_proof_has_no_constructor_and_a_forged_one_is_refused(world, request_a, proof):
    with pytest.raises(ProofRequired):
        Proof()
    with pytest.raises(ProofRequired):
        Proof(instance_id=world["R2"], kind=v.KIND_JSONL, request=request_a)
    # An object with the shape and the fields of a proof, made around the constructor.
    forged = object.__new__(Proof)
    for name in ("instance_id", "kind", "request", "scope", "notes", "passed_generation"):
        object.__setattr__(forged, name, getattr(proof, name))
    with pytest.raises(ProofRequired):
        negative(world, request_a, proof=forged)
    with pytest.raises(ProofRequired):
        Answer.from_proof(forged, artifact="R2")
    # A scope alone, or any other object, is no proof either.
    with pytest.raises(ProofRequired):
        negative(world, request_a, proof=Scope("A", ("A/r1",)))


def test_only_the_coverage_module_can_take_the_mint():
    with pytest.raises(ProofRequired, match="waterwhite.coverage"):
        answers.claim_proof_mint()


def test_a_proof_cannot_be_changed_copied_or_pickled(proof):
    with pytest.raises(AttributeError):
        proof.instance_id = "another"
    with pytest.raises(AttributeError):
        del proof.scope
    for duplicate in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            duplicate(proof)


def test_a_proof_answers_only_for_its_own_artifact_and_request(world, request_a, proof):
    good = Answer.from_proof(proof, artifact="R2")
    assert good.answer == v.DOES_NOT_CONTAIN and good.scope is proof.scope
    assert good.to_json()["scope"]["members"] == ["A/r1"]
    with pytest.raises(ProofRequired):  # another artifact
        negative(world, request_a, instance=world["R1"], proof=proof)
    with pytest.raises(ProofRequired):  # another request
        negative(world, requests.withdraw(world.store, "C"), proof=proof)
    with pytest.raises(ProofRequired):  # notes the proof does not carry
        negative(world, request_a, proof=proof, notes=("anything",))


def test_no_other_answer_can_be_turned_into_does_not_contain(world, request_a, proof):
    contains = coverage.answer(world.store, "R1", request_a)
    assert contains.answer == v.CONTAINS
    with pytest.raises((ProofRequired, ValueError)):
        dataclasses.replace(contains, answer=v.DOES_NOT_CONTAIN)
    with pytest.raises((ProofRequired, ValueError)):
        dataclasses.replace(contains, answer=v.DOES_NOT_CONTAIN, proof=proof, chain=(), notes=())
    # An answer changed behind the type's back is caught when it is written.
    object.__setattr__(contains, "answer", v.DOES_NOT_CONTAIN)
    with pytest.raises((ProofRequired, ValueError)):
        contains.to_json()


def test_a_finding_never_says_does_not_contain(request_a):
    with pytest.raises(ProofRequired):
        Finding(answer=v.DOES_NOT_CONTAIN, instance_id="x", kind=v.KIND_JSONL, request=request_a)


def test_the_other_three_answers_keep_their_own_rules(world, request_a, proof):
    base = dict(artifact="R1", instance=world["R1"], kind=v.KIND_JSONL, request=request_a)
    hop = Hop(artifact="R1", record="L1", link="member")
    failing = Failing(names=v.NAMES_MISSING_ORIGIN, condition=v.CONDITION_A)
    Answer(answer=v.CONTAINS, chain=(hop,), **base)
    Answer(answer=v.MAY_CONTAIN, names=v.NAMES_MISSING_ORIGIN, failing=failing, **base)
    Answer(answer=v.NOT_COVERED, reason=v.REASON_RECORD_MISSING, **base)
    for wrong in (
        dict(answer="probably not"),
        dict(answer=v.CONTAINS),  # no chain
        dict(answer=v.CONTAINS, chain=(hop,), proof=proof),  # a proof beside a positive
        dict(answer=v.MAY_CONTAIN, names="a cause of my own", failing=failing),
        dict(answer=v.MAY_CONTAIN, names=v.NAMES_MISSING_ORIGIN),  # no failing link
        dict(answer=v.NOT_COVERED),  # no reason
        dict(answer=v.NOT_COVERED, reason="somewhere else"),
    ):
        with pytest.raises(ValueError):
            Answer(**base, **wrong)


def test_the_json_form_holds_the_fields_of_the_interface(world, request_a):
    answer = coverage.answer(world.store, "gen/T", request_a).to_json()
    assert list(answer) == [
        "artifact", "instance", "kind", "request", "answer", "reason", "names", "names_detail",
        "failing", "candidates", "chain", "basis", "notes", "untraced_steps", "first_exposure",
        "level", "scope",
    ]
    assert (answer["artifact"], answer["instance"], answer["kind"]) == ("gen/T", world["gen/T"], "jsonl")
    assert answer["request"] == {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}
    assert [hop["hop"] for hop in answer["chain"]] == ["gen/T:L1", "gen/P:L1", "src/A:L1"]
    assert answer["chain"][0] == {
        "hop": "gen/T:L1", "artifact": "gen/T", "instance": world["gen/T"], "record": "L1",
        "link": "observed", "job": "translation", "generation": True,
    }
    assert answer["chain"][-1]["member"] == "A/r1"
    assert Hop(segment="E/seg-2", step=7).text == "E/seg-2:step 7"
