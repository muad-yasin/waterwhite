"""Check 4, store integrity (plan 5.5 Check 4): the rows of fixture/table/check04.toml.
Each injected fault is put on a copy of a store (README section 7) and the query is
asked through the command line: the loud faults stop it, a deleted or half-covered trace
does not.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(4)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check04(row, part, world, request):
    grade(row, part, world, request)
