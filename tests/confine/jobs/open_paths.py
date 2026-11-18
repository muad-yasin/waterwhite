"""Tries to open, as files, each path it is given, then reads its one handed record
through the harness, and reports as its one output, in JSON, what happened.

Arguments: the paths to try."""

import json
import sys

import waterwhite_job


def attempt(path):
    try:
        with open(path, "rb") as handle:
            return {"ok": True, "bytes": len(handle.read())}
    except OSError as error:
        return {"ok": False, "error": type(error).__name__, "errno": error.errno}


with waterwhite_job.connect() as job:
    report = {path: attempt(path) for path in sys.argv[1:]}
    (only,) = job.handed
    report["handed"] = job.read(only)
    job.emit(json.dumps(report, sort_keys=True), used=[only])
