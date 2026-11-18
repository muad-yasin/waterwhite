"""Check 1, the exposure join (plan 5.5 Check 1): the rows of fixture/table/check01.toml.
The join's rows are compared with the oracle's log (plan 5.3), which is the grading
side: it is read from a run of the oracle over the same build, and the tool never sees
it. A row about the rows of a step trace waits until a command prints them.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(1)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check01(row, part, world, request):
    grade(row, part, world, request)
