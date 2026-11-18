"""Emits an output through the client, then writes something on the request pipe
that is no request, and exits with status 0."""

import os

import waterwhite_job

job = waterwhite_job.connect()
job.emit(job.read(job.handed[0]), used=[job.handed[0]])
request_fd = int(os.environ[waterwhite_job.ENV_FDS].split(",")[0])
os.write(request_fd, b"\x00\xff this is not a request\n")
