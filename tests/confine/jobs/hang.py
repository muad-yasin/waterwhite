"""Emits an output and then never ends."""

import time

import waterwhite_job

job = waterwhite_job.connect()
job.emit("text", used=[])
time.sleep(3600)
