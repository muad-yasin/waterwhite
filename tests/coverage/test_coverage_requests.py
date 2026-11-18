"""The withdraw request and the refusals (plan 2.3, Decision 24, L7)."""

from __future__ import annotations

import pytest
import ww_world

from waterwhite import errors, extensions, requests
from waterwhite.records import OccurrenceRef, WithdrawRequest


def test_a_source_expands_to_every_member_registered_when_the_query_runs(world, tmp_path):
    first = requests.withdraw(world.store, "A")
    assert (first.source, first.members, first.records, first.narrowed) == ("A", ("A/r1",), (), False)
    assert first.to_json() == {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}

    # A second revision registered under the same source is in the next expansion.
    ww_world.ingest(world.store, tmp_path / "src/A2.jsonl", ["a later text"], "A", "src/A2", "A/r2")
    assert requests.withdraw(world.store, "A").members == ("A/r1", "A/r2")
    narrowed = requests.withdraw(world.store, "A", members=["A/r2", "A/r2"])
    assert (narrowed.members, narrowed.narrowed) == (("A/r2",), True)


def test_records_narrow_a_withdrawal_and_are_named_artifact_colon_locator(world):
    request = requests.withdraw(world.store, "A", records=["src/A:L2"])
    assert request.records == (world.ref("src/A", 2),)
    assert request.record_names == ("src/A:L2",)
    assert request.record_level
    assert request.to_json()["records"] == ["src/A:L2"]
    # A reference, or the instance id in place of the name, names the same record.
    by_ref = requests.withdraw(world.store, "A", records=[world.ref("src/A", 2)])
    by_id = requests.withdraw(world.store, "A", records=[f"{world['src/A']}:L2", "src/A:L2"])
    assert by_ref.records == by_id.records == request.records
    assert requests.expand(world.store, WithdrawRequest("A", (), (world.ref("src/A", 2),))) == by_ref


def test_a_source_with_no_registered_member_is_refused(world):
    with pytest.raises(errors.SourceNotRegistered) as refused:
        requests.withdraw(world.store, "Q")
    assert (refused.value.name, refused.value.exit_code) == ("source not registered", 2)
    world.store.add_source("EMPTY")  # a name with no member is no better
    with pytest.raises(errors.SourceNotRegistered):
        requests.withdraw(world.store, "EMPTY")


def test_a_member_of_another_source_is_refused(world):
    with pytest.raises(requests.MemberNotRegistered) as refused:
        requests.withdraw(world.store, "A", members=["B/r1"])
    assert (refused.value.name, refused.value.exit_code) == ("member not registered", 2)
    assert isinstance(refused.value, errors.Refused)


@pytest.mark.parametrize("record", ["src/B:L1", "gen/P:L1", "src/A:L9", "nowhere:L1"])
def test_a_record_outside_the_source_is_refused(world, record):
    with pytest.raises(requests.RecordNotInSource) as refused:
        requests.withdraw(world.store, "A", records=[record])
    assert (refused.value.name, refused.value.exit_code) == ("record not in source", 2)


def test_a_record_of_a_member_the_request_leaves_out_is_refused(world, tmp_path):
    ww_world.ingest(world.store, tmp_path / "src/A2.jsonl", ["a later text"], "A", "src/A2", "A/r2")
    with pytest.raises(requests.RecordNotInSource):
        requests.withdraw(world.store, "A", members=["A/r2"], records=["src/A:L1"])
    assert requests.withdraw(world.store, "A", members=["A/r1"], records=["src/A:L1"]).members == ("A/r1",)


@pytest.mark.parametrize("record", ["src/A", "src/A:", ":L1", "src/A:7", "shard-1:D0"])
def test_a_record_that_cannot_be_read_is_an_unsupported_selector(world, record):
    with pytest.raises(errors.UnsupportedSelector):
        requests.withdraw(world.store, "A", records=[record])


def test_a_ban_and_a_flagged_request_are_refused_while_their_extensions_are_off(world):
    for call in (lambda: requests.ban(world.store, "passage X"),
                 lambda: requests.flagged(world.store, "flagged.jsonl"),
                 lambda: requests.parse(world.store, {"kind": "ban", "text": "passage X"}),
                 lambda: requests.parse(world.store, {"kind": "flagged", "path": "f.jsonl"})):
        with pytest.raises(errors.ExtensionOff) as refused:
            call()
        assert (refused.value.name, refused.value.exit_code) == ("extension off", 2)


def test_a_switched_on_extension_is_refused_as_not_built_and_never_answered(world):
    world.store.set_setting(extensions.setting_key(extensions.CONTENT_BAN), "on")
    with pytest.raises(errors.Refused) as refused:
        requests.ban(world.store, "passage X")
    assert not isinstance(refused.value, errors.ExtensionOff)
    assert "only its off state" in refused.value.detail


def test_an_unsupported_selector_is_refused_and_never_approximated(world):
    with pytest.raises(errors.UnsupportedSelector) as refused:
        requests.withdraw(world.store, "A", selectors=["url-prefix=https://example.org/"])
    assert (refused.value.name, refused.value.exit_code) == ("unsupported selector", 2)
    with pytest.raises(errors.UnsupportedSelector):
        requests.parse(world.store, {"kind": "withdraw", "source": "A", "url_prefix": "https://x"})
    with pytest.raises(errors.UnsupportedSelector):
        requests.parse(world.store, {"kind": "erase", "source": "A"})
    with pytest.raises(errors.UnsupportedSelector):
        requests.expand(world.store, "A")  # not a withdrawal at all


def test_a_request_given_as_a_mapping_round_trips(world):
    request = requests.withdraw(world.store, "A", records=["src/A:L1"])
    assert requests.parse(world.store, request.to_json()) == request
    assert requests.parse(world.store, {"source": "C"}).members == ("C/r1",)
    assert isinstance(request.records[0], OccurrenceRef)
