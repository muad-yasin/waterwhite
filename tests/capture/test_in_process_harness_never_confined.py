"""A log the in-process harness writes never establishes a bound (plan 3.3 "Generation",
F10, Decision 13; docs/confine.md).

Only ``waterwhite.confine`` writes a log that says "confined". ``harness.run_job``, the
public in-process entry point, is called here the way a caller of it would call it,
naming the harness "confined" for a job of its own that reads only what it was handed. A
caller that states a confinement this module does not give must not be able to get a log
that says so, or one whose ``bound_established()`` is true: the call is refused with a
``TypeError`` (the function has no such argument) or it writes "in-process". Either is
checked on the log, which is read from disk.
"""

from __future__ import annotations

import json

from waterwhite import harness
from waterwhite import vocabulary as v
from waterwhite.harness import Handed, JobOutput, run_job
from waterwhite.records import OccurrenceRef

A1 = OccurrenceRef("a" * 64, "L1")


def job(ctx):
    return JobOutput(text="made from " + ctx.read(A1), used=(A1,))


def test_run_job_called_with_the_word_confined_never_leaves_a_confined_log(tmp_path):
    log_path = tmp_path / "job.log.jsonl"
    try:
        result = run_job(job, job_name="claims-a-bound", handed=[Handed(A1, "first text")],
                         log_path=log_path, harness=v.HARNESS_CONFINED)
    except TypeError:
        # The public function takes no such argument: nothing was run, and nothing on
        # disk says "confined".
        assert not log_path.exists() or v.HARNESS_CONFINED not in log_path.read_text(encoding="ascii")
        return
    # Otherwise the call went through, and the log it wrote is the in-process one.
    assert result.log.harness == v.HARNESS_IN_PROCESS
    assert not result.log.bound_established()
    entries = [json.loads(line) for line in log_path.read_text(encoding="ascii").splitlines()]
    assert [e["harness"] for e in entries if e["kind"] == "job"] == [v.HARNESS_IN_PROCESS]
    assert v.HARNESS_CONFINED not in log_path.read_text(encoding="ascii")
    assert not harness.read_log(log_path).bound_established()
