"""Check 5, the coverage gate (plan 5.5 Check 5): the rows of
fixture/table/check05.toml, one test per row, graded against the tool through its
command line.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(5)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check05(row, part, world, request):
    grade(row, part, world, request)
