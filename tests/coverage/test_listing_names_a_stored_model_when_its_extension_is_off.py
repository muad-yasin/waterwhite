"""A store that holds a published model, with the published-models extension off, is
listed with the model named as "not covered" (the owner's answer of 18 Nov 2026: "list
it as not covered"). Before this the whole listing was refused "extension off", so one
stored model made every listing of the store impossible.

The model gets one answer in the listing (answer "not covered", reason "not registered", the
note that the extension is off) and one entry in ``not_covered`` that names the extension.
Every other answer is the answer it was. A query that names the model gives the same answer
(plan 1.4, 3.1, Section 7; D-CORE-5, which changed the reason word of D-CORE-2 and the
refusal of its query), and with the switch on the build answers nothing about a model (it
holds only the off state).
"""

from __future__ import annotations

import io
import json

import pytest
import ww_world

from waterwhite import cli, coverage, errors, extensions, listing, requests
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

MODEL = "model-1"
MODEL_ID = "d" * 64  # this build has no write path for a published model (the extension is off): the row is written as the store file holds it


def with_a_model(world) -> None:
    world.sql(("INSERT INTO artifact (instance_id, kind, role, content_digest, edge_digest) "
               "VALUES (?, 'published-model', NULL, ?, ?)", (MODEL_ID, "e" * 64, "f" * 64)),
              ("INSERT INTO artifact_name (name, artifact_id) "
               "SELECT ?, artifact_id FROM artifact WHERE instance_id = ?", (MODEL, MODEL_ID)))
    world.ids[MODEL] = MODEL_ID


def listed(world) -> dict:
    return listing.listing(world.store, requests.withdraw(world.store, "A")).to_json()


def run(world, *argv: str) -> tuple[int, str, str]:
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


def test_the_listing_names_the_stored_model_as_not_covered_and_says_the_extension_is_off(world):
    with_a_model(world)
    made = listed(world)
    answer = next(a for a in made["answers"] if a["artifact"] == MODEL)
    assert (answer["answer"], answer["reason"], answer["kind"]) == (
        "not covered", "not registered", v.KIND_PUBLISHED_MODEL)
    assert answer["notes"] == [v.NOTE_PUBLISHED_MODELS_OFF]
    assert answer["instance"] == world[MODEL]
    gap = next(g for g in made["not_covered"] if g["artifact"] == MODEL)
    assert (gap["kind"], gap["reason"], gap["instance"]) == (
        v.KIND_PUBLISHED_MODEL, "not registered", world[MODEL])
    assert gap["detail"] == v.NOTE_PUBLISHED_MODELS_OFF


def test_every_other_answer_of_the_listing_is_what_it_was(world):
    before = listed(world)
    with_a_model(world)
    after = listed(world)
    keep = lambda made: {a["artifact"]: a for a in made["answers"] if a["artifact"] != MODEL}  # noqa: E731
    assert keep(after) == keep(before)


def test_a_query_that_names_the_model_answers_what_the_listing_says_about_it(world):
    with_a_model(world)
    request = requests.withdraw(world.store, "A")
    made = coverage.answer(world.store, MODEL, request).to_json()
    listed_answer = next(a for a in listing.listing(world.store, request).to_json()["answers"]
                         if a["artifact"] == MODEL)
    assert (made["answer"], made["reason"], made["notes"]) == (
        "not covered", "not registered", [v.NOTE_PUBLISHED_MODELS_OFF])
    assert made == listed_answer


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_the_command_line_query_that_names_the_model_answers_not_covered_not_registered(world, fmt):
    with_a_model(world)
    code, out, err = run(world, "query", "withdraw", "A", "--artifact", MODEL, "--format", fmt)
    assert (code, err) == (0, "")
    assert "not covered" in out and "not registered" in out and v.NOTE_PUBLISHED_MODELS_OFF in out
    assert "extension off" not in out
    if fmt == "json":
        answer = json.loads(out)
        assert (answer["artifact"], answer["answer"], answer["reason"]) == (MODEL, "not covered", "not registered")


def test_a_query_that_names_the_model_still_stops_for_a_broken_switch(world):
    with_a_model(world)
    world.sql(("INSERT INTO setting (key, value) VALUES ('extension.generating-model', 'ON')", ()))
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        coverage.answer(world.store, MODEL, requests.withdraw(world.store, "A"))


def test_with_the_switch_on_the_listing_is_still_refused_because_the_build_holds_the_off_state_only(world):
    with_a_model(world)
    world.store.set_setting(extensions.setting_key(extensions.PUBLISHED_MODELS), "on")
    with pytest.raises(errors.Refused, match="holds only its off state"):
        listed(world)


@pytest.mark.parametrize("fmt", ["json", "text"])
def test_the_command_line_lists_a_store_that_holds_a_model(world, fmt):
    with_a_model(world)
    code, out, err = run(world, "query", "withdraw", "A", "--format", fmt)
    assert (code, err) == (0, "")
    assert MODEL in out and v.NOTE_PUBLISHED_MODELS_OFF in out
    assert "extension off" not in out  # the reason word of D-CORE-2 is gone from the listing
    if fmt == "json":
        assert any(a["artifact"] == MODEL and a["reason"] == "not registered" for a in json.loads(out)["answers"])


def test_extension_off_is_no_longer_a_reason_of_not_covered():
    assert not hasattr(v, "REASON_EXTENSION_OFF")
    assert v.NOT_COVERED_REASONS == (v.REASON_RECORD_MISSING, v.REASON_NOT_REGISTERED)


def test_the_report_of_a_store_that_holds_a_model_is_what_it_was(world):
    # Not part of the listing: probe of what the report does with a stored model (see D-CORE-2).
    before = run(world, "report", "withdraw", "A")
    assert before[0] == 0  # the report is made, not refused: the comparison below is of two reports
    with_a_model(world)
    after = run(world, "report", "withdraw", "A")
    assert after == before
