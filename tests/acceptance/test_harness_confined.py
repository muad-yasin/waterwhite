"""The harness row of Check 5 (c05.harness.outside-read) runs on the confined
harness, in a real bubblewrap sandbox, and what shows it is read back here.

Where bwrap is missing or cannot make its namespaces, the test is skipped with
the sandbox's own reason: the row is not run, which is not a pass."""

from __future__ import annotations

import pytest
import table
from acceptance_sets import Unavailable

ROW = "c05.harness.outside-read"


def test_the_outside_read_row_runs_confined_and_the_log_shows_it(world):
    row = next(r for r in table.load_check(5) if r.id == ROW)
    try:
        mismatches = world.put(table.interpret(row))
    except Unavailable as missing:
        pytest.skip(str(missing))
    assert mismatches == []
    seen = world.harness_evidence
    # The log's job entry names the confined harness.
    assert seen["harness"] == ["confined"]
    # The job was handed one record and read it; the one request outside the
    # hand-off was refused and logged.
    refused = [r for r in seen["reads"] if r["refused"]]
    assert len(seen["reads"]) == 2 and len(refused) == 1
    assert "outside_ref" in refused[0] and "instance_id" not in refused[0]
    # What the job was given back for that request is nothing: it says "refused",
    # not "delivered:" and a text.
    assert seen["outputs"] == ["refused"]
