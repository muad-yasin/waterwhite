"""What ``World`` says of a store, tested with a replica function written by hand: the
needs a store's own build answers (a shard, a mixture) and the report of a build that
failed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import acceptance_world
import pytest
import table
from acceptance_world import World


class Built:
    def __init__(self, names):
        self.instances = {name: "id" for name in names}


def question(*needs):
    row = table.Row(id="c05.x", check=5, question="", plan="", core=True, extension=None,
                    given={}, expect={}, expect_under={}, source="", rule=None)
    return table.Question(row=row, kind=table.KIND_ANSWER, store="main", state="unmutated",
                          request=None, artifacts=("E/mix-1",), record=None, expect={},
                          needs=needs)


def test_a_shard_and_a_mixture_are_met_when_the_stores_build_registered_one(tmp_path,
                                                                            monkeypatch):
    import acceptance_sets

    world = World(tmp_path)
    monkeypatch.setattr(world, "store", lambda name: Built(held))
    need = question(table.NEED_MIXTURE, table.NEED_SHARD)
    held = ["src/A", "E/shard-1"]
    # "main" is tokenized only where the Megatron environment is set; without it a
    # row that needs its shards is not run, and says why.
    monkeypatch.delenv(acceptance_sets.MEGATRON_VARIABLE, raising=False)
    assert world.unmet(need) == [acceptance_world.NEED_MEGATRON_ENV]
    monkeypatch.setenv(acceptance_sets.MEGATRON_VARIABLE, str(tmp_path))
    assert world.unmet(need) == [table.NEED_MIXTURE]
    held = ["src/A", "E/mix-1"]
    assert world.unmet(need) == [table.NEED_SHARD]
    held = ["src/A", "E/shard-1", "E/mix-1"]
    assert world.unmet(need) == []


def test_a_build_that_failed_is_reported_with_its_first_cause_to_every_row(tmp_path, monkeypatch):
    calls = []

    def broken(folder, name, **arguments):
        calls.append(name)
        raise RuntimeError(f"first cause {len(calls)}")

    monkeypatch.setattr(acceptance_world.replica_fixture, "build_table_store", broken)
    world = World(tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="first cause 1"):
            world.store("main")
    assert calls == ["main"]


def test_a_set_build_that_failed_is_reported_with_its_first_cause_to_every_row(tmp_path,
                                                                               monkeypatch):
    import acceptance_sets

    calls = []

    def broken(self, size):
        calls.append(size)
        raise RuntimeError(f"first cause {len(calls)}")

    monkeypatch.setattr(acceptance_sets.Sets, "_captured", broken)
    sets = acceptance_sets.Sets(tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="first cause 1"):
            sets.build("small", acceptance_sets.CAPTURE)
    assert calls == ["small"]


def test_the_manifest_of_a_release_is_read_once_however_many_candidate_entries_name_it(tmp_path, monkeypatch):
    import acceptance_sets

    # two releases with manifests that give other lines to the same handles: an entry must read its own release's manifest
    shards = {"R": tmp_path / "release-r", "S": tmp_path / "release-s"}
    for release, offset in (("R", 0), ("S", 100)):
        Path(f"{shards[release]}.manifest.jsonl").write_text(
            "".join(json.dumps({"handle": f"h{i}", "line": i + 1 + offset}) + "\n" for i in range(40)), encoding="utf-8")
    entries = [{"release": "RS"[i % 2], "rows": [f"h{i}"], "candidates": [{"id": f"x-{i + 1}", "source": "A"}],
                "sources": ["A"]} for i in range(40)]
    done = SimpleNamespace(exit_code=0, printed=lambda: {"edge": {"label": "L", "candidate_sources": ["A"]},
                                                         "answer": "yes"})
    world = World(tmp_path)
    asked, reads = [], []
    monkeypatch.setattr(world, "_set", lambda question, size: (
        SimpleNamespace(counted={"check6": {"candidates": entries}}), tmp_path / "store.sqlite"))
    monkeypatch.setattr(world, "file_of", lambda store_file, artifact: shards[artifact])
    monkeypatch.setattr(world, "_many", lambda store_file, commands: asked.extend(commands) or [done] * len(commands))
    monkeypatch.setattr(world, "cli", lambda store_file, *arguments: done)
    real = acceptance_sets.read_rows
    monkeypatch.setattr(acceptance_sets, "read_rows", lambda path: reads.append(path) or real(path))
    expect = {"label": "L", "candidate_sources": ["A"], "answer_for_each_candidate_source": "yes"}
    found = world._set_candidates(SimpleNamespace(expect=expect), "100k")
    assert found == []
    assert asked == [("show", "RS"[i % 2], "--record", f"L{i + 1 + (100 if i % 2 else 0)}") for i in range(40)]
    assert len(reads) == 2, f"the manifests were read {len(reads)} times for 40 entries of two releases"
