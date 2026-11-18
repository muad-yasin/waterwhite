"""Asks for something long that was not handed, and for a model with a long
identifier: two texts of the job's choosing that would reach the log."""

import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed
    text = job.read(only)
    try:
        job.read(text * 500)
        asked = "delivered"
    except waterwhite_job.Refused:
        asked = "refused"
    try:
        job.call_model("m" * 300, "a prompt")
        called = "called"
    except waterwhite_job.RequestFailed as error:
        called = error.name
    job.emit(asked + "|" + called, used=[only])
