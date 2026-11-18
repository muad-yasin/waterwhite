"""The closed-input harness for a job that runs out of process, confined (plan 3.3
"Generation", Decision 13, F10, 5.8 B-1).

    from waterwhite.confine import python_program, run_confined_job
    from waterwhite.harness import Handed

    result = run_confined_job(
        python_program("jobs/paraphrase.py"),
        job_name="paraphrase",
        handed=[Handed(ref, text)],
        log_path="logs/paraphrase.jsonl",
        timeout_seconds=600,
    )
    result.outputs        # harness.JobOutput, in the job's order
    result.log            # records.HarnessLog, read back from the log file

The job runs in a bubblewrap sandbox with no network and no file of the machine's
data. It asks the harness for each record over a pipe, and the harness logs every
request before it answers. The log file, ``HarnessLog`` and the edges built from
the result are the ones of ``waterwhite.harness``. docs/confine.md describes the
sandbox, the protocol, and what this does and does not establish.

``waterwhite_job`` in this package is the job-side client. It is a file for the
sandbox, written against the standard library alone, and is not imported here.
"""

from __future__ import annotations

from ..errors import ConfinedJobFailed, JobProtocolViolation, SandboxUnavailable
from .runner import DEFAULT_MAX_REQUEST_BYTES, MAX_LOGGED_TEXT, PROTOCOL, run_confined_job
from .sandbox import (
    CLIENT_FOLDER,
    DEFAULT_SCRATCH_BYTES,
    JOB_FOLDER,
    SANDBOX_COMMAND,
    SCRATCH_FOLDER,
    CodeMount,
    Program,
    Sandbox,
    find_sandbox,
    python_program,
)

__all__ = [
    "CLIENT_FOLDER",
    "CodeMount",
    "ConfinedJobFailed",
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_SCRATCH_BYTES",
    "JOB_FOLDER",
    "JobProtocolViolation",
    "MAX_LOGGED_TEXT",
    "PROTOCOL",
    "Program",
    "SANDBOX_COMMAND",
    "SCRATCH_FOLDER",
    "Sandbox",
    "SandboxUnavailable",
    "find_sandbox",
    "python_program",
    "run_confined_job",
]
