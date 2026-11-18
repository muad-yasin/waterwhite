"""The job side of the confined harness (plan 3.3 "Generation", F10).

A job that runs inside the sandbox imports this one file and nothing else of the
tool. It needs the standard library only. The harness mounts it read-only at
``/waterwhite/client`` and puts that folder on the module path, so a job writes:

    import waterwhite_job

    with waterwhite_job.connect() as job:
        (only,) = job.handed
        text = job.read(only)
        job.emit(text.upper(), used=[only])

Everything the job gets, it gets by asking the harness over two pipes it
inherited: one request per line, one reply per line, each a JSON object (the
protocol ``waterwhite.confine/1``, described in docs/confine.md). The harness
writes a read or a model call to its log before it answers.

Leaving the ``with`` block without an error sends the end message. A job that
exits without it, or exits with a status other than zero, has failed, and the
harness returns none of its outputs.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

__all__ = [
    "ENV_FDS",
    "HarnessGone",
    "Job",
    "JobError",
    "PROTOCOL",
    "Ref",
    "Refused",
    "RequestFailed",
    "connect",
]

PROTOCOL = "waterwhite.confine/1"

# The harness sets this variable to "<request fd>,<reply fd>": the job writes its
# requests to the first and reads the replies from the second.
ENV_FDS = "WATERWHITE_JOB_FDS"

# The error name the harness gives a read outside the hand-off.
_REFUSED_READ = "read outside the hand-off"


class Ref(NamedTuple):
    """A record at a place: an artifact instance and a locator."""

    instance_id: str
    locator: str


class JobError(Exception):
    """Base class of what this module raises."""


class Refused(JobError):
    """The harness refused a read: the record was not handed to this job. Nothing
    was delivered, and the attempt is in the harness log. A job may catch this and
    go on."""


class RequestFailed(JobError):
    """The harness answered a request with an error other than a refused read.
    ``name`` is the error's fixed name, ``detail`` its text."""

    def __init__(self, name: str, detail: str) -> None:
        self.name = name
        self.detail = detail
        super().__init__(f"{name}: {detail}" if detail else name)


class HarnessGone(JobError):
    """The pipes to the harness are closed, were never there, or carried something
    that is not a reply of the protocol."""


def _ref_json(ref: object) -> object:
    if isinstance(ref, Ref):
        return {"instance_id": ref.instance_id, "locator": ref.locator}
    return ref


class Job:
    """The connection of one job to its harness. ``connect`` builds it.

    ``job_name`` is the job's name and ``handed`` the references of the records it
    was handed, in hand-off order. Knowing them is not a read."""

    def __init__(self, request_fd: int, reply_fd: int) -> None:
        self._requests = os.fdopen(request_fd, "wb", buffering=0)
        self._replies = os.fdopen(reply_fd, "rb")
        self._ended = False
        greeting = self._reply()
        handed = greeting.get("handed")
        if greeting.get("protocol") != PROTOCOL or not isinstance(handed, list):
            raise HarnessGone(f"the harness does not speak {PROTOCOL}")
        self.job_name: str = greeting.get("job_name", "")
        self.handed: tuple[Ref, ...] = tuple(
            Ref(item["instance_id"], item["locator"]) for item in handed
        )

    def _reply(self) -> dict:
        line = self._replies.readline()
        if not line:
            raise HarnessGone("the harness closed the pipe")
        try:
            reply = json.loads(line.decode("utf-8"))
        except ValueError as error:
            raise HarnessGone(f"not a reply of the protocol ({error})") from None
        if not isinstance(reply, dict):
            raise HarnessGone("not a reply of the protocol")
        return reply

    def _ask(self, request: dict) -> dict:
        if self._ended:
            raise HarnessGone("the job has sent its end message")
        data = json.dumps(request, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        try:
            self._requests.write(data + b"\n")
        except OSError as error:
            raise HarnessGone(f"the harness closed the pipe ({error})") from None
        reply = self._reply()
        if reply.get("ok") is True:
            return reply
        name, detail = str(reply.get("error", "")), str(reply.get("detail", ""))
        if name == _REFUSED_READ:
            raise Refused(detail or name)
        raise RequestFailed(name, detail)

    def read(self, ref: object) -> str:
        """The text of a handed record. Anything else is refused with ``Refused``;
        the harness logs the attempt either way. ``ref`` is a ``Ref``; any other
        JSON value is sent as it is and refused."""
        return self._ask({"op": "read", "ref": _ref_json(ref)})["text"]

    def call_model(self, model: str, prompt: str) -> str:
        """Call a model through the harness, which logs the model's identifier and
        then forwards the call to the client its caller configured. The sandbox has
        no network, so this is the only route to a model."""
        return self._ask({"op": "model", "model": model, "prompt": prompt})["text"]

    def emit(self, text: str, used: object = ()) -> int:
        """Hand one output to the harness with the handed records the output states
        it used. Returns the output's index, counted from 0."""
        refs = [_ref_json(ref) for ref in used]  # type: ignore[attr-defined]
        return self._ask({"op": "output", "text": text, "used": refs})["index"]

    def end(self) -> None:
        """Tell the harness the job is done. Nothing can be asked afterwards."""
        if not self._ended:
            self._ask({"op": "end"})
            self._ended = True
            self._requests.close()
            self._replies.close()

    def __enter__(self) -> "Job":
        return self

    def __exit__(self, kind: object, error: object, trace: object) -> None:
        if kind is None:
            self.end()


def connect() -> Job:
    """Connect to the harness over the two inherited pipes."""
    value = os.environ.get(ENV_FDS)
    if value is None:
        raise HarnessGone(f"{ENV_FDS} is not set: this process was not started by the harness")
    try:
        request_fd, reply_fd = (int(part) for part in value.split(","))
    except ValueError:
        raise HarnessGone(f"{ENV_FDS} does not name two file descriptors: {value!r}") from None
    return Job(request_fd, reply_fd)
