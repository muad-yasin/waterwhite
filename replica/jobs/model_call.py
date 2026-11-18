"""The replica's model-calling generation job as a program for the confined harness (the R3
step of the build, the generating-model extension).

    model_call.py MODEL_JSON USED_JSON

``MODEL_JSON`` is the identifier of the model the job calls, as a JSON text. ``USED_JSON`` is a
JSON list of places in the hand-off, counted from 0: the handed records the output states it
used.

The job reads every handed record through the harness, in hand-off order, calls the model once
through the harness (the sandbox has no network, so this is the only route to a model) with
the texts it read as the prompt, and emits what the model answered. The harness logs the model's
identifier and forwards the call to the client its caller configured.

It runs inside the sandbox with the standard library and ``waterwhite_job`` only.
"""

import json
import sys

import waterwhite_job


def main() -> None:
    model = json.loads(sys.argv[1])
    used = set(json.loads(sys.argv[2]))
    with waterwhite_job.connect() as job:
        texts = [job.read(ref) for ref in job.handed]
        answer = job.call_model(model, "\n".join(texts))
        job.emit(answer, used=[ref for place, ref in enumerate(job.handed) if place in used])


if __name__ == "__main__":
    main()
