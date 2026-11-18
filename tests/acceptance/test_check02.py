"""Check 2, the cache states of run E (plan 5.5 Check 2): the rows of
fixture/table/check02.toml. Each case registers the runs in a fresh store from its
starting state (README section 7: c2.i to c2.iv), never in a store that holds the
traces, and asks what register-run said and what the checkpoints and mixtures answer
after it.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(2)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check02(row, part, world, request):
    grade(row, part, world, request)
