"""Reads its one handed record and emits its text unchanged."""

import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed
    job.emit(job.read(only), used=[only])
