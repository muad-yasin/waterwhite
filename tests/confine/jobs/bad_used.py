"""States that an output used a record the job was not handed.

The first argument is the instance id of that record."""

import sys

import waterwhite_job

with waterwhite_job.connect() as job:
    job.emit("text", used=[waterwhite_job.Ref(sys.argv[1], "L1")])
