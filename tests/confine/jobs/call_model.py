"""Calls two models through the client the harness hands it."""

import waterwhite_job

with waterwhite_job.connect() as job:
    (only,) = job.handed
    text = job.read(only)
    first = job.call_model("stand-in-1", "rewrite: " + text)
    second = job.call_model("stand-in-2", "shorten: " + first)
    again = job.call_model("stand-in-1", "once more")
    try:
        job.call_model("none", "a reserved identifier")
        reserved = "called"
    except waterwhite_job.RequestFailed as error:
        reserved = error.name
    job.emit("|".join([first, second, again, reserved]), used=[only])
