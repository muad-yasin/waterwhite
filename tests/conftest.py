"""Collection rules for the whole tree.

A test marked ``megatron`` needs the test-time Megatron environment (tools/setup_megatron_env.sh,
named by WATERWHITE_MEGATRON_ENV). Where the variable is not set the project's rule is "leave them
out with ``-m 'not megatron'``": the oracle's fixtures FAIL such a test rather than skip it, because a
skipped oracle would let the checks that rest on it pass unseen (tests/oracle/conftest.py). On a clean
checkout nobody has typed that option, so the tests marked ``megatron`` are deselected here, and the
run says how many and why. A run that selects them on purpose (``-m megatron``, or any expression that
names the marker) is left as it is: without the environment they fail, as before.
"""

from __future__ import annotations

import os

VARIABLE = "WATERWHITE_MEGATRON_ENV"
_left_out: list[int] = [0]


def pytest_collection_modifyitems(config, items):
    if os.environ.get(VARIABLE) or "megatron" in (config.getoption("-m") or ""):
        return
    kept, left = [], []
    for item in items:
        (left if item.get_closest_marker("megatron") else kept).append(item)
    if left:
        config.hook.pytest_deselected(items=left)
        items[:] = kept
        _left_out[0] = len(left)


def pytest_terminal_summary(terminalreporter):
    if _left_out[0]:
        terminalreporter.write_line(
            f"{_left_out[0]} tests marked megatron were deselected: {VARIABLE} is not set "
            "(set it to the folder tools/setup_megatron_env.sh builds to run them)")
