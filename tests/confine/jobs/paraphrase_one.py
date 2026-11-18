"""Reads its one handed record and emits it in capitals, stating that record as used."""

import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed
    job.emit(job.read(only).upper(), used=[only])
