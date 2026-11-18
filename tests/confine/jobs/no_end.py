"""Emits an output and exits with status 0 without the end message."""

import waterwhite_job

job = waterwhite_job.connect()
job.emit(job.read(job.handed[0]), used=[job.handed[0]])
