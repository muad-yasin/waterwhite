"""Reads a record, emits an output, then fails."""

import sys

import waterwhite_job

job = waterwhite_job.connect()
job.emit(job.read(job.handed[0]), used=[job.handed[0]])
print("the job gives up", file=sys.stderr)
sys.exit(3)
