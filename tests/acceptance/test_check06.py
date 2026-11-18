"""Check 6, missing evidence (plan 5.5 Check 6): the rows of
fixture/table/check06.toml, graded against the tool through its command line.

Four of the seventeen rows are asked of a generated set made without capture and
get one test per size: the small set of this folder, and the 10k and 100k sets
the plan names, which carry the markers ``slow`` and ``megatron``. Their expected
values come from the plant list as the counting script counts it. The other rows
are asked of the table's stores ``no-capture`` and ``import``.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, sized_params

ROWS = table.load_check(6)


@pytest.mark.parametrize(("row", "part", "size"), sized_params(ROWS))
def test_check06(row, part, size, world, request):
    grade(row, part, world, request, size)
