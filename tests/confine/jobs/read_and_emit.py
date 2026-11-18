"""Reads its handed records in a fixed order and emits two outputs."""

import waterwhite_job

with waterwhite_job.connect() as job:
    first, second, third = job.handed
    texts = [job.read(third), job.read(first), job.read(third)]
    job.emit(" + ".join(texts), used=[third, first])
    job.emit("made by " + job.job_name, used=[])
