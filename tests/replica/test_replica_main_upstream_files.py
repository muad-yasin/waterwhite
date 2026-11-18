"""The files upstream of the artifacts of the rebuild report, on the store `main` of the
hand-written table built with Megatron's preprocessing (plan 4.1 with its ledger row
BF-6; docs/interfaces.md section 7.1).

The report of "withdraw A" on this store holds, under ``upstream_files``, seven files and
no more: the three generation outputs and the revision of source A (no
action), the two files between a source and the rebuilt shard (rewrite) and the sanitizer
input of shard-2 (unresolved, "missing origin", condition a). The order is the store's
registration order. Files that answer "does not contain" are not listed.

The test builds the store, which takes 15 to 40 seconds, so it carries the markers
``megatron`` and ``slow`` and is skipped unless ``WATERWHITE_MEGATRON_ENV`` is set. The
generation jobs run in process: the seven answers are the same with the confined harness.
"""

from __future__ import annotations

import io
import json
import os

import pytest
from waterwhite import cli, coverage, requests
from waterwhite import vocabulary as v
from waterwhite.store.store import Store

from replica.fixture import build_table_store
from replica.megatron_tokenizer import ENV_VARIABLE

pytestmark = [pytest.mark.megatron, pytest.mark.slow]

IN_PROCESS = "in-process"
RULED = frozenset(("gen/P", "gen/S", "gen/T", "src/A", "E/shard-3.input",
                   "E/shard-3.sanitizer-input", "E/shard-2.sanitizer-input"))
FIELDS = frozenset(("artifact", "instance", "kind", "role", "answer", "action", "reason"))


@pytest.fixture(scope="module")
def main_store(tmp_path_factory):
    if not os.environ.get(ENV_VARIABLE):
        pytest.skip(f"{ENV_VARIABLE} is not set")
    return build_table_store(tmp_path_factory.mktemp("main-upstream") / "main", "main",
                             tokenize=True, parallel=3, generation_harness=IN_PROCESS)


@pytest.fixture(scope="module")
def printed(main_store):
    out, err = io.StringIO(), io.StringIO()
    try:
        code = cli.main(["report", "withdraw", "A", "--store", str(main_store.store)],
                        stdout=out, stderr=err)
    except SystemExit as leaving:
        code = int(leaving.code or 0)
    assert code == 0, err.getvalue()
    return json.loads(out.getvalue())


@pytest.fixture(scope="module")
def registration_order(main_store):
    """The ruled files in the order the store registered them."""
    with Store.open(main_store.store, readonly=True) as store:
        return [a.names[0] for a in store.artifacts() if a.names and a.names[0] in RULED]


def listed(printed) -> dict:
    held = printed.get("upstream_files")
    assert isinstance(held, list), f"the report has no list under upstream_files: {list(printed)}"
    names = [entry["artifact"] for entry in held]
    assert len(names) == len(set(names)), f"a file is listed twice: {names}"
    return dict((entry["artifact"], entry) for entry in held)


def test_the_list_holds_the_seven_ruled_files_in_registration_order_after_artifacts(
        printed, registration_order):
    keys = list(printed)
    assert "upstream_files" in keys, keys
    assert keys.index("upstream_files") == keys.index("artifacts") + 1
    assert set(registration_order) == RULED
    assert [entry["artifact"] for entry in printed["upstream_files"]] == registration_order


def test_every_entry_has_its_fields_and_the_instance_the_store_registered(printed, main_store):
    for name, entry in listed(printed).items():
        assert FIELDS <= set(entry), (name, sorted(FIELDS - set(entry)))
        assert entry["instance"] == main_store.instances[name]
        assert entry["kind"] == v.KIND_JSONL
        assert entry["answer"] != v.DOES_NOT_CONTAIN


def test_the_generation_outputs_and_the_revision_the_request_names_take_no_action(printed):
    entries = listed(printed)
    for name in ("gen/P", "gen/S", "gen/T"):
        entry = entries[name]
        assert (entry["role"], entry["answer"], entry["action"]) == (
            v.ROLE_INTERMEDIATE, v.CONTAINS, None), name
        reason = entry["reason"].lower()
        assert "retired" in reason and "not rebuilt" in reason, entry["reason"]
    source = entries["src/A"]
    assert (source["role"], source["answer"], source["action"], source["reason"]) == (
        v.ROLE_SOURCE, v.CONTAINS, None, "the revision the request withdraws")


def test_the_intermediate_files_ahead_of_the_rebuilt_shard_are_rewrite(printed):
    entries = listed(printed)
    for name in ("E/shard-3.input", "E/shard-3.sanitizer-input"):
        entry = entries[name]
        assert (entry["role"], entry["answer"], entry["action"]) == (
            v.ROLE_INTERMEDIATE, v.CONTAINS, v.ACTION_REWRITE), name
        assert "reason" in entry


def test_the_sanitizer_input_of_shard_2_is_unresolved_with_its_named_cause(printed, main_store):
    entry = listed(printed)["E/shard-2.sanitizer-input"]
    with Store.open(main_store.store, readonly=True) as store:
        request = requests.withdraw(store, "A")
        said = coverage.answer(store, main_store.instances["E/shard-2.sanitizer-input"], request,
                               session=coverage.Session(store, request)).to_json()
    assert (entry["answer"], entry["action"]) == (v.MAY_CONTAIN, v.UNRESOLVED)
    assert entry["names"] == said["names"] == "missing origin"
    assert entry["failing"] == said["failing"]
    assert entry["failing"]["condition"] == "a"
    assert "reason" in entry
