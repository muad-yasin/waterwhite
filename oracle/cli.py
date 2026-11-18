"""The shared shape of the oracle's entry scripts.

Each script takes one argument, the path of a JSON config file, does its work,
prints its result as JSON on standard output and exits 0. A failure the oracle
names itself prints ``{"error": ..., "detail": ...}`` on standard error and exits
2, with nothing on standard output. Any other exception is a fault and is left to
show its traceback.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable

from oracle.errors import OracleError

EXIT_OK = 0
EXIT_FAILED = 2


def run_tool(
    description: str,
    work: Callable[[str], object],
    argv: list[str] | None = None,
    *,
    render: Callable[[object], str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="path of the JSON config file")
    args = parser.parse_args(argv)
    # Megatron reports through the logging module ("Build and save the GPTDataset
    # train indices"). The root logger is set up here, before Megatron is imported,
    # so those lines go to standard error and standard output holds the result only.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    try:
        result = work(args.config)
    except OracleError as e:
        print(json.dumps({"error": type(e).__name__, "detail": str(e)}), file=sys.stderr)
        return EXIT_FAILED
    if render is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render(result))
    return EXIT_OK
