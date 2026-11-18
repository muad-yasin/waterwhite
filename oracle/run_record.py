"""The training stand-in: write the run record of one run segment, and its step
record when the config names one. Config and output: docs/oracle.md, "Run record
and step record".

The index is built into the config's new, empty cache folder, as it would exist
while the run trains; the cache digests are taken from it and, for a cyclic run,
the order digests come from driving the sampler class.
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
    new_output_path(segment.output("run_record"))
    if "step_record" in segment.outputs:
        new_output_path(segment.output("step_record"))

    from oracle import pipeline, records  # imports Megatron

    built, document = pipeline.build_index(segment.data)
    record, _orders = pipeline.segment_records(segment, built, document["cache_digests"])
    write_json(segment.output("run_record"), record)
    if "step_record" in segment.outputs:
        write_json(segment.output("step_record"), records.step_record(segment))
    return record


def main(argv: list[str] | None = None) -> int:
    return run_tool(__doc__, work, argv)


if __name__ == "__main__":
    sys.exit(main())
