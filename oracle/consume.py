"""The consumer (plan 5.3): build the datasets of one run segment, drive Megatron's
samplers over its launches and log, for each completed step, what the dataset's
item getter reads. Config and output: docs/oracle.md, "Consumer".

The index is built into the config's new, empty cache folder. The log goes to
``outputs.log``; a summary goes to ``outputs.summary`` when that is named.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # run as a script; -I keeps the script's folder off sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oracle.cli import run_tool  # noqa: E402
from oracle.config import load_segment, new_output_path, write_json  # noqa: E402


def work(config_path: str) -> dict:
    segment = load_segment(config_path)
    new_output_path(segment.output("log"))
    if "summary" in segment.outputs:
        new_output_path(segment.output("summary"))

    from oracle import pipeline  # imports Megatron

    built, document = pipeline.build_index(segment.data)
    summary = pipeline.consume_segment(segment, built)
    summary["cache_digests"] = [
        {"kind": r["kind"], "shard": r["shard"], "digest": r["digest"]}
        for r in document["cache_digests"]
    ]
    if "summary" in segment.outputs:
        write_json(segment.output("summary"), summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
