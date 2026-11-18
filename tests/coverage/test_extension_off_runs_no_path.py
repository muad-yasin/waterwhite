"""A request of an extension that is off is refused with a named error, and no
path of the extension runs (plan 1.4, 2.3, L7).

Two of the six extensions have a request of their own: a content ban and a request
that starts from flagged records. This file puts both to the command line while
their switches are off. "No path runs" is shown by what the refusal leaves
untouched: no answer and no listing is computed, and a flagged file that does not
exist gives "extension off", not an error about the file, so the refusal comes
before the file is looked at.

The other extensions have no request of their own in this build. A published model
in a store answers "not covered", reason "not registered", when it is asked about (D-CORE-5):
test_coverage_proof.py::test_a_published_model_is_an_extension_that_is_off. The
generating-model switch, off by default, changes no request; switched on, a
negative past a generation edge is judged by its model (condition (h)):
test_coverage_proof.py::test_with_the_extension_on_a_negative_past_a_generation_edge_is_judged_by_its_model.
Exposure counts and adapters have no request form and no code in this build.
"""

from __future__ import annotations

import io
import json

import pytest

from waterwhite import cli, coverage, errors, extensions, listing, requests


@pytest.fixture
def untouched(monkeypatch):
    """Make the way past the refusal fail loudly: an answer or a listing. A test
    that reaches one of them fails with its name."""
    reached: list[str] = []

    def no_answer(*args, **kwargs):
        reached.append("answer")
        raise AssertionError("an answer was computed")

    def no_listing(*args, **kwargs):
        reached.append("listing")
        raise AssertionError("a listing was computed")

    monkeypatch.setattr(coverage, "answer", no_answer)
    monkeypatch.setattr(listing, "listing", no_listing)
    return reached


@pytest.mark.parametrize("argv", [
    ["query", "ban", "passage X"],
    ["query", "ban", "passage X", "--artifact", "R1"],
    ["query", "flagged", "MISSING"],
    ["query", "flagged", "MISSING", "--artifact", "R1"],
])
def test_a_request_of_an_extension_that_is_off_is_refused_before_any_path_of_it(world, untouched, argv):
    missing = world.folder / "no-such-folder" / "flagged.jsonl"
    argv = [str(missing) if a == "MISSING" else a for a in argv]
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    error = json.loads(err.getvalue())
    assert (code, error["error"], out.getvalue()) == (errors.EXIT_REFUSED, "extension off", "")
    assert untouched == []
    assert not missing.exists()


def test_every_extension_is_off_in_a_new_store(world):
    settings = world.store.settings()
    assert all(not extensions.is_on(settings, name) for name in extensions.EXTENSIONS)
    for call in (lambda: requests.ban(world.store, "passage X"),
                 lambda: requests.flagged(world.store, "flagged.jsonl")):
        with pytest.raises(errors.ExtensionOff):
            call()
