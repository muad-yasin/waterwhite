"""The driver: run several run segments from one JSON file, each with its own new
cache folder, and write for each its digests, run record, step record, oracle log
and summary. One command can so produce a fresh segment, a resume on a new
mixture and a warm start. Config and output: docs/oracle.md, "Driver".
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import (  # noqa: E402
    load_json,
    new_output_path,
    parse_segments,
    resolve_path,
    write_json,
)
from oracle.errors import ConfigError  # noqa: E402


def work(config_path: str) -> dict:
    raw, base = load_json(config_path)
    out = None
    if raw.get("summary") is not None:
        if not isinstance(raw["summary"], str):
            raise ConfigError("'summary' must be a path")
        out = resolve_path(raw["summary"], base)
        new_output_path(out)

    segments = parse_segments(raw, base)

    from oracle import pipeline  # imports Megatron

    result = pipeline.run_segments(segments)
    if out is not None:
        write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
