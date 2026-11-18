"""The list of joins that plan Check 1 verified ships with the tool, and a store made
by ``init`` takes it from the tool (plan 3.5, F6, Check 1; DECISIONS.md D-26, D-37).

The values are the fixture's own, read from the spec the run plan is written in: the
pinned release and the two sampler classes of the two sampler types its runs use. They
are not read from the tool.

No store made through the command line lists a join the tool does not ship: ``init
--verified-join`` is refused, as a usage error and with nothing written, for any join
outside the shipped list, and a subset of the list is accepted and is the store's whole
list. A test that needs another list makes its store through the library
(``Store.create``), which writes the joins its caller gives and is outside what the tool
verified. Where the default lives for a store made by the library is not fixed here.
"""

from __future__ import annotations

import io
import json
import re
import tomllib
from pathlib import Path

import ww_world

from waterwhite import cli, coverage, requests
from waterwhite import vocabulary as v
from waterwhite.records import SnapshotShard
from waterwhite.store.store import Store

ROOT = Path(__file__).resolve().parents[2]
SPEC = tomllib.loads((ROOT / "fixture/spec/fixture_spec.toml").read_text(encoding="utf-8"))
TAG = SPEC["megatron"]["release_tag"]
SINGLE = (TAG, "single", SPEC["runs_common"]["sampler_class_single"])
CYCLIC = (TAG, "cyclic", SPEC["runs_common"]["sampler_class_cyclic"])
SHIPPED = {SINGLE, CYCLIC}


def init(path: Path, *more: str) -> tuple[int, dict | None, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(["init", "--store", str(path), "--normaliser", ww_world.NORMALISER, *more],
                    stdout=out, stderr=err)
    return code, (json.loads(out.getvalue()) if out.getvalue() else None), err.getvalue()


def test_the_spec_names_the_joins_the_tool_ships():
    assert TAG == "core_v0.20.0"
    assert SINGLE[2].endswith("MegatronPretrainingSampler")
    assert CYCLIC[2].endswith("MegatronPretrainingRandomSampler")


def test_a_store_made_by_init_lists_the_joins_check_1_verified(tmp_path):
    code, made, err = init(tmp_path / "new.db")
    assert (code, err) == (0, "")
    assert {tuple(join) for join in made["verified_joins"]} == SHIPPED
    assert len(made["verified_joins"]) == len(SHIPPED)
    with Store.open(tmp_path / "new.db", readonly=True) as store:
        assert set(store.verified_joins()) == SHIPPED
        assert len(store.verified_joins()) == len(SHIPPED)


OUTSIDE = {
    "an unknown release": ("core_v9.9.9", "single", SINGLE[2]),
    "an unknown sampler class": (TAG, "single", "elsewhere.Sampler"),
    "an unknown sampler type": (TAG, "elsewhere", SINGLE[2]),
    "the right class under the other sampler type": (TAG, "single", CYCLIC[2]),
    "the other class under the other sampler type": (TAG, "cyclic", SINGLE[2]),
    "the shipped triple with its tag changed": ("core_v0.20.1", "cyclic", CYCLIC[2]),
}


def assert_refused_with_nothing_written(tmp_path: Path, *joins: tuple[str, str, str]) -> None:
    flags = [part for join in joins for part in ("--verified-join", ",".join(join))]
    code, made, err = init(tmp_path / "new.db", *flags)
    assert code == 1, (code, made, err)
    assert made is None
    assert json.loads(err)["error"] == "usage", err
    assert list(tmp_path.iterdir()) == []


def test_a_join_outside_the_shipped_list_is_refused_by_init_with_nothing_written(tmp_path):
    """The flag names a join the tool does not ship: a usage error, exit code 1, nothing
    on standard output, no store file and no file beside it. The cases differ from a
    shipped join in one part, so a check of the parts one by one, not of the triple, is
    caught."""
    for why, join in OUTSIDE.items():
        assert join not in SHIPPED, why
        assert_refused_with_nothing_written(tmp_path, join)


def test_one_join_outside_the_shipped_list_refuses_the_whole_command(tmp_path):
    """A shipped join beside one that is not shipped does not make the command pass:
    the store is not made with the shipped one only."""
    for join in (("core_v9.9.9", "single", "elsewhere.Sampler"), OUTSIDE["an unknown sampler class"]):
        assert_refused_with_nothing_written(tmp_path, SINGLE, join)
        assert_refused_with_nothing_written(tmp_path, join, CYCLIC)


def test_a_subset_of_the_shipped_list_is_accepted_and_is_the_whole_list_of_the_store(tmp_path):
    """The flag replaces the shipped list and cannot add to it. The first line is the
    refusal of a join outside the list, so a tree that accepts one cannot pass here."""
    assert_refused_with_nothing_written(tmp_path, ("core_v9.9.9", "single", "elsewhere.Sampler"))
    code, made, err = init(tmp_path / "new.db", "--verified-join", ",".join(CYCLIC))
    assert (code, err) == (0, "")
    assert [tuple(join) for join in made["verified_joins"]] == [CYCLIC]
    with Store.open(tmp_path / "new.db", readonly=True) as store:
        assert store.verified_joins() == (CYCLIC,)
    both, made, err = init(tmp_path / "both.db", "--verified-join", ",".join(CYCLIC),
                           "--verified-join", ",".join(SINGLE))
    assert (both, err) == (0, "")
    assert {tuple(join) for join in made["verified_joins"]} == SHIPPED


def test_a_store_made_by_the_library_lists_the_joins_its_caller_gives(tmp_path):
    """Guards what already holds: ``Store.create`` writes the joins it is given, shipped
    or not, and is outside what the tool verified."""
    extra = ("core_v9.9.9", "single", "elsewhere.Sampler")
    with Store.create(tmp_path / "lib.db", normaliser=ww_world.NORMALISER,
                      verified_joins=[extra]) as store:
        assert store.verified_joins() == (extra,)


def make_run(store: Store, folder: Path, name: str, sampler: str, sampler_class: str) -> None:
    """One run of one step that reads the one document of a shard made from A's line."""
    a = store.resolve("src/A").instance_id
    shard, digest = ww_world.shard(
        store, folder / f"shards/{name}", f"shard-{name}", [a], [ww_world.recovered(0, ww_world.ref(a, 1))])
    mixture = store.register_mixture(
        name=f"mix-{name}", shards=[SnapshotShard(0, shard, f"shards/{name}", digest)],
        settings={"mix": name})
    ww_world.segment(
        store, name, f"{name}/seg-1", 0, mixture, None, [(1, 1, None)], {1: (0, 0)},
        sampler=sampler, sampler_class=sampler_class, shuffle=sampler == "cyclic")
    store.register_checkpoint(name=f"{name}@1", segment=f"{name}/seg-1", step=1)


def test_a_store_made_by_init_gives_a_verified_run_its_basis(tmp_path):
    """The gate of plan 2.4 and F6: a checkpoint of a run on the pinned release and
    one of the two sampler types answers "contains" when its step read the record, and
    not "may contain" naming an unverified version or sampler type."""
    code, _, err = init(tmp_path / "store.db")
    assert (code, err) == (0, "")
    with Store.open(tmp_path / "store.db") as store:
        ww_world.ingest(store, tmp_path / "src/A.jsonl", ["passage X"], "A", "src/A")
        for name, (_tag, sampler, sampler_class) in (("S", SINGLE), ("C", CYCLIC)):
            make_run(store, tmp_path, name, sampler, sampler_class)
        request = requests.withdraw(store, "A")
        answers = {name: coverage.answer(store, f"{name}@1", request).to_json() for name in ("S", "C")}
    assert {name: a["answer"] for name, a in answers.items()} == {"S": "contains", "C": "contains"}, {
        name: (a["answer"], a["names"]) for name, a in answers.items()}
    assert all(a["names"] is None for a in answers.values())


def test_the_init_output_in_the_docs_is_the_output_init_prints(tmp_path):
    """docs/cli.md prints the shape of what init writes; its schema_version is the
    store's."""
    text = (ROOT / "docs/cli.md").read_text(encoding="utf-8")
    section = text.split("\n## init", 1)[1].split("\n## ", 1)[0]
    shown = re.search(r'"schema_version": (\d+)', section)
    assert shown is not None
    _, made, _ = init(tmp_path / "new.db")
    assert int(shown.group(1)) == made["schema_version"]
