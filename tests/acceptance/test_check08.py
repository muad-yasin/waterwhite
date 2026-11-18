"""Check 8, checkpoints with the evidence of run E (plan 5.5 Check 8): the rows of
fixture/table/check08.toml, asked of the store main with its runs registered.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, params

ROWS = table.load_check(8)


@pytest.mark.parametrize(("row", "part"), params(ROWS))
def test_check08(row, part, world, request):
    grade(row, part, world, request)
