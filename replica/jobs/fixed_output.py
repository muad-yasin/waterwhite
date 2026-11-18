"""The replica's generation stand-in as a program for the confined harness.

    fixed_output.py OUTPUT_JSON USED_JSON

``OUTPUT_JSON`` is the fixed output as a JSON text: what a model would have
written, had one been called. ``USED_JSON`` is a JSON list of places in the
hand-off, counted from 0: the handed records the output states it used.

The job reads every handed record through the harness, in hand-off order, calls
no model and emits the one output. It makes the requests the in-process stand-in
makes, in the same order, so the harness writes the same log for both.

It runs inside the sandbox with the standard library and ``waterwhite_job`` only.
"""

import json
import sys

import waterwhite_job


def main() -> None:
    text = json.loads(sys.argv[1])
    used = set(json.loads(sys.argv[2]))
    with waterwhite_job.connect() as job:
        for ref in job.handed:
            job.read(ref)
        job.emit(text, used=[ref for place, ref in enumerate(job.handed) if place in used])


if __name__ == "__main__":
    main()
