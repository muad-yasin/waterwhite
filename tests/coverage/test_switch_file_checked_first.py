"""A switch that cannot be read stops every answer (the owner's answer of 18 Nov 2026:
"checked first, everything stops").

An extension's switch is "on" or "off". A store whose ``setting`` table holds anything
else under one of the six switch keys (a value edited by hand, for example "ON") is
refused when an answer is asked for, before the request, the proof, the listing or the
report looks at anything else. Before this, only an answer that read the broken switch
stopped; an answer that never read it (a "contains" at a shard that passes no
generation edge) was given. The fix is writing the switch (``Store.set_setting``),
which does not pass through the check, and the commands that are not answers
(``show``, ``check``) keep working on such a store.
"""

from __future__ import annotations

import io
import json

import pytest
import ww_world

from waterwhite import cli, coverage, errors, extensions, listing, requests
from waterwhite.store.store import Store

ALL_SIX = list(extensions.EXTENSIONS)


def break_switch(world, extension: str, value: str = "ON") -> None:
    key = extensions.setting_key(extension)
    world.sql(("INSERT INTO setting (key, value) VALUES (?, ?) "
               "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value)))


def run(world, *argv: str) -> tuple[int, str, str]:
    world.store.close()
    out, err = io.StringIO(), io.StringIO()
    code = cli.main([*argv, "--store", str(world.path)], stdout=out, stderr=err)
    world.store = Store.open(world.path)
    return code, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("extension", ALL_SIX)
def test_a_broken_switch_of_any_extension_stops_an_answer_that_never_reads_it(world, ask, extension):
    assert ask("A", "shard-3")["answer"] == "contains"  # given before the switch breaks
    break_switch(world, extension)
    with pytest.raises(errors.Refused, match=f"the switch of the extension '{extension}' holds 'ON'"):
        ask("A", "shard-3")


def test_a_broken_switch_stops_the_answer_for_a_name_the_store_does_not_hold(world, ask):
    break_switch(world, extensions.EXPOSURE_COUNTS)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        ask("A", "no-such-artifact")


def test_a_broken_switch_stops_the_proof_and_the_answer_built_on_a_request_made_before_it_broke(world):
    request = requests.withdraw(world.store, "A")
    break_switch(world, extensions.ADAPTERS)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        coverage.prove(world.store, "shard-1", request)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        coverage.answer(world.store, "no-such-artifact", request)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        coverage.check_store(world.store)


def test_a_broken_switch_stops_the_listing(world):
    request = requests.withdraw(world.store, "A")
    break_switch(world, extensions.PUBLISHED_MODELS)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        listing.listing(world.store, request)


def test_the_switch_is_refused_before_a_source_or_a_selector_is_looked_at(world):
    break_switch(world, extensions.CONTENT_BAN)
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        requests.withdraw(world.store, "no-such-source")
    with pytest.raises(errors.Refused, match="holds 'ON'"):
        requests.withdraw(world.store, "A", selectors=["kind=web"])


def test_a_ban_with_another_switch_broken_names_the_broken_switch_not_extension_off(world):
    break_switch(world, extensions.FLAGGED_RECORDS)
    with pytest.raises(errors.Refused, match="'flagged-records' holds 'ON'") as caught:
        requests.ban(world.store, "a text")
    assert not isinstance(caught.value, errors.ExtensionOff)
    with pytest.raises(errors.Refused, match="'flagged-records' holds 'ON'") as caught:
        requests.flagged(world.store, "flags.jsonl")
    assert not isinstance(caught.value, errors.ExtensionOff)


@pytest.mark.parametrize("argv", [
    ["query", "withdraw", "A", "--artifact", "shard-3"],
    ["query", "withdraw", "A"],
    ["query", "ban", "a text"],
    ["report", "withdraw", "A"],
])
def test_the_command_line_stops_with_the_refusal_exit_code_and_prints_nothing_on_standard_output(world, argv):
    break_switch(world, extensions.GENERATING_MODEL)
    code, out, err = run(world, *argv)
    assert (code, out) == (errors.EXIT_REFUSED, "")
    assert "holds 'ON'" in err and "generating-model" in err


def test_commands_that_are_not_answers_keep_working_on_a_store_with_a_broken_switch(world):
    break_switch(world, extensions.GENERATING_MODEL)
    code, out, _ = run(world, "show", "shard-1")
    assert code == 0 and json.loads(out)["artifact"] == "shard-1"
    code, out, _ = run(world, "check")
    assert code == 0 and json.loads(out)["check"] == "passed"


def test_writing_the_switch_fixes_the_store_and_the_answers_come_back(world, ask):
    break_switch(world, extensions.GENERATING_MODEL)
    with pytest.raises(errors.Refused):
        ask("A", "shard-3")
    world.store.set_setting(extensions.setting_key(extensions.GENERATING_MODEL), "off")
    assert ask("A", "shard-3")["answer"] == "contains"


def test_a_setting_under_the_switch_prefix_that_names_no_extension_stops_every_answer(world, ask):
    world.sql(("INSERT INTO setting (key, value) VALUES ('extension.no-such-extension', 'on')", ()))
    with pytest.raises(errors.Refused, match="'extension.no-such-extension' names no extension"):
        ask("A", "shard-3")
    code, out, err = run(world, "query", "withdraw", "A")
    assert (code, out) == (errors.EXIT_REFUSED, "") and "names no extension" in err


def test_the_tools_own_method_clears_a_switch_it_cannot_read_and_still_cannot_write_one(world, ask):
    world.sql(("INSERT INTO setting (key, value) VALUES ('extension.no-such-extension', 'on')", ()))
    break_switch(world, extensions.ADAPTERS)
    with pytest.raises(errors.InvalidRecord):
        world.store.set_setting("extension.no-such-extension", "on")  # not written again
    with pytest.raises(errors.InvalidRecord):
        world.store.set_setting(extensions.setting_key(extensions.ADAPTERS), "ON")
    world.store.remove_setting("extension.no-such-extension")
    world.store.remove_setting(extensions.setting_key(extensions.ADAPTERS))
    world.store.remove_setting("never-held")  # nothing to remove: nothing changes
    assert ask("A", "shard-3")["answer"] == "contains"
    assert world.store.settings() == {}


def test_a_missing_switch_row_is_off_and_not_broken(world, ask):
    assert extensions.setting_key(extensions.GENERATING_MODEL) not in world.store.settings()
    assert ask("A", "shard-3")["answer"] == "contains"
