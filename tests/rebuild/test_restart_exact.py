"""The exact-rerun restart (plan 4.3, F7; docs/interfaces.md section 7.3 "restarts",
docs/rebuild.md section 4), on the hand-built world of ``rb_kit.py``.

The restart is stated for the runs that have a stale segment; its input here is the
list of ``caches`` entries of the form (section 7.3), written by hand, so that these
tests stand on the restart alone.

Not tested here: the step of a seeded run, found by building the corrected order from
the successor shards and comparing it with the historical one. Its expected value is
the oracle's log of the corrected run, which the table row
``c09.E.exact-rerun-restart`` holds; the hand-built world has no shard long enough for
a replay, and a step computed here would come from the tool.
"""

from __future__ import annotations

from waterwhite import vocabulary as v

from rb_kit import hand_report, need_attr


def cache(run: str, segment: str, mixture: str, *, stale: bool) -> dict:
    return {
        "run": run, "segment": segment, "mixture": mixture,
        "corrected_mixture": f"successor of {mixture}" if stale else mixture,
        "digest": "0" * 64, "corrected_digest": ("1" if stale else "0") * 64,
        "stale": stale, "rebuilt": False, "rebuilt_from": None, "reason": "not tested here",
    }


def restarts(world, caches):
    function = need_attr("waterwhite.rebuild.restart_exact", "exact_rerun_restarts")
    report = hand_report(world, [("mix-1", "keep", None), ("mix-2", "rewrite", "successor of mix-2")])
    return function(world.store, report, caches)


def test_an_unseeded_run_gets_the_phase_start_as_an_assumption(unseeded_repair_world):
    caches = [cache("E", "E/seg-1", "mix-1", stale=False), cache("E", "E/seg-2", "mix-2", stale=True)]
    assert restarts(unseeded_repair_world, caches) == [{
        "run": "E", "segment": "E/seg-2", "step": 4,
        "basis": v.RESTART_BASIS_PHASE_START, "step_comparison": v.STEP_COMPARISON_NOT_POSSIBLE,
    }]


def test_the_phase_start_is_the_first_step_of_the_earliest_segment_that_is_rebuilt(unseeded_repair_world):
    # One entry per run that has a stale segment. Run Y has one segment, which starts at step 1.
    caches = [
        cache("E", "E/seg-1", "mix-1", stale=False), cache("E", "E/seg-2", "mix-2", stale=True),
        cache("Y", "Y/seg-1", "mix-2", stale=True),
    ]
    found = {e["run"]: e for e in restarts(unseeded_repair_world, caches)}
    assert {run: (e["segment"], e["step"]) for run, e in found.items()} == {
        "E": ("E/seg-2", 4), "Y": ("Y/seg-1", 1)}
    assert {e["step_comparison"] for e in found.values()} == {v.STEP_COMPARISON_NOT_POSSIBLE}


def test_a_run_with_no_stale_segment_has_no_restart(unseeded_repair_world):
    assert restarts(unseeded_repair_world, [cache("E", "E/seg-1", "mix-1", stale=False),
                                           cache("E", "E/seg-2", "mix-2", stale=False)]) == []
    assert restarts(unseeded_repair_world, []) == []
