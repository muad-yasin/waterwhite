"""Check 3, tokenization links (plan 5.5 Check 3): the rows of
fixture/table/check03.toml, graded against the tool through its command line.

Thirteen of the fifteen rows are asked of a generated set and get one test per
size: the small set of this folder, and the 100k set the plan names, which carries
the markers ``slow`` and ``megatron``. The expected values come from the plant
list as the counting script counts it, from each shard's own index and from the
team's tokenizer, never from the tool.

How a row becomes a test, and how to read the counts: README.md in this folder.
"""

from __future__ import annotations

import pytest
import table
from acceptance_rows import grade, sized_params

ROWS = table.load_check(3)


@pytest.mark.parametrize(("row", "part", "size"), sized_params(ROWS))
def test_check03(row, part, size, world, request):
    grade(row, part, world, request, size)
