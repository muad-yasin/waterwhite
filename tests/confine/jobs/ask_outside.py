"""Asks for a record that was not handed, and for a path, and goes on.

The first argument is the instance id of the record outside the hand-off."""

import sys

import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed
    answers = []
    for asked in (waterwhite_job.Ref(sys.argv[1], "L1"), "/data/corpus/B.jsonl"):
        try:
            job.read(asked)
            answers.append("delivered")
        except waterwhite_job.Refused:
            answers.append("refused")
    text = job.read(only)
    job.emit(",".join(answers) + "|" + text, used=[only])
