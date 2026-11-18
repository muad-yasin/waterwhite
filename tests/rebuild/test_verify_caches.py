"""Stale index caches (plan 4.4, 3.5; docs/interfaces.md section 7.3 "caches",
docs/rebuild.md section 5 "Stale caches"), through ``check_caches`` on the hand-built
world with the successors registered.

Megatron's cache key holds the paths of the shards and not their content. The tool
compares the mixture digest of the snapshot a segment was registered with against
the digest of the corrected mixture. The mixture digest is
``waterwhite.ids.mixture_content_digest`` of the content digests of the shards of the
snapshot in the trainer's order (docs/interfaces.md 7.3), which the tests compute
here from the registered shards.

Not tested here: that a stale segment's corrected order is built from an empty cache
folder (``rebuilt`` and ``rebuilt_from``). It needs a replay of real index arrays;
the hand-built world's shards are too short for one. The table row
``c09.E.stale-cache.E/seg-2`` holds it.
"""

from __future__ import annotations

from waterwhite import ids

from rb_kit import build, hand_report, need_attr

REPORT = [("mix-1", "keep", None), ("mix-2", "rewrite", "successor of mix-2")]


def digest_of(world, *shards: str) -> str:
    return ids.mixture_content_digest(
        [world.store.artifact(world[name]).content_digest for name in shards])


def check(world, artifacts=REPORT):
    function = need_attr("waterwhite.rebuild.verify_caches", "check_caches")
    return {e["segment"]: e for e in function(world.store, hand_report(world, artifacts))}


def test_a_segment_whose_mixture_the_report_keeps_is_not_stale(repair_world):
    entry = check(repair_world)["E/seg-1"]
    assert (entry["run"], entry["mixture"], entry["corrected_mixture"]) == ("E", "mix-1", "mix-1")
    same = digest_of(repair_world, "shard-1")
    assert (entry["digest"], entry["corrected_digest"]) == (same, same)
    assert (entry["stale"], entry["rebuilt"], entry["rebuilt_from"]) == (False, False, None)


def test_a_segment_whose_mixture_is_rewritten_is_stale_when_the_digests_differ(repair_world):
    entry = check(repair_world)["E/seg-2"]
    assert (entry["run"], entry["mixture"]) == ("E", "mix-2")
    assert entry["corrected_mixture"] == "successor of mix-2"
    assert entry["digest"] == digest_of(repair_world, "shard-1", "shard-3")
    assert entry["corrected_digest"] == digest_of(repair_world, "shard-1", "successor of shard-3")
    assert entry["digest"] != entry["corrected_digest"] and entry["stale"] is True


def test_every_segment_of_a_run_whose_mixture_is_rebuilt_has_an_entry_and_other_runs_none(repair_world):
    found = check(repair_world)
    # E and Y train on mix-2, which is rewritten; D trains on mix-1 only.
    assert set(found) == {"E/seg-1", "E/seg-2", "Y/seg-1"}
    assert {name: e["stale"] for name, e in found.items()} == {
        "E/seg-1": False, "E/seg-2": True, "Y/seg-1": True}
    assert found["Y/seg-1"]["digest"] == found["E/seg-2"]["digest"]


def test_a_report_that_keeps_every_mixture_leaves_every_cache_alone(repair_world):
    function = need_attr("waterwhite.rebuild.verify_caches", "check_caches")
    report = hand_report(repair_world, [("mix-1", "keep", None), ("mix-2", "keep", None)])
    entries = function(repair_world.store, report)
    assert not [e for e in entries if e["stale"]]


def test_a_rewritten_mixture_with_no_registered_successor_has_no_verdict(tmp_path):
    world = build(tmp_path / "w")
    try:
        entry = check(world)["E/seg-2"]
        assert entry["corrected_mixture"] is None and entry["corrected_digest"] is None
        assert entry["stale"] is None
        assert entry["digest"] == digest_of(world, "shard-1", "shard-3")
    finally:
        world.store.close()
