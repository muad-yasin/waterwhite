"""Runs a job in the sandbox and serves it its records (plan 3.3 "Generation",
Decision 13, F10, 5.8 B-1).

``run_confined_job`` runs the harness's code (``harness._run_job``, not the public
``run_job``, which can only write "in-process") with one fixed in-process job: a
broker, part of the tool, that starts the real job in the sandbox and answers
its requests from the ``JobContext`` the harness hands the broker. So every read
and every model call of the confined job goes through the very code that serves
an in-process job, and the log file is written by the very writer: the format,
the order of the entries, ``HarnessLog``, the bound and the edges are the same.

The difference is what the log is evidence of. The in-process harness logs the
reads that went through its context and cannot say what else the job opened. The
confined job has no file of the machine's data and no network to open, so the
pipe is its one way to a record or a model, and what passes the pipe is logged
before it is answered. docs/confine.md says what this establishes and what not.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import threading
import time
from collections.abc import Sequence

from .. import harness
from .. import vocabulary as v
from ..errors import ConfinedJobFailed, HarnessRefused, InvalidRecord, JobProtocolViolation
from ..harness import Handed, JobContext, JobOutput, JobResult, ModelClient
from ..records import OccurrenceRef
from . import sandbox as _sandbox
from .sandbox import DEFAULT_SCRATCH_BYTES, Program, Sandbox

__all__ = ["DEFAULT_MAX_REQUEST_BYTES", "MAX_LOGGED_TEXT", "PROTOCOL", "run_confined_job"]

# The same text as waterwhite_job.PROTOCOL. A test holds the two together.
PROTOCOL = "waterwhite.confine/1"

DEFAULT_MAX_REQUEST_BYTES = 64 * 1024 * 1024

_OP_READ = "read"
_OP_MODEL = "model"
_OP_OUTPUT = "output"
_OP_END = "end"

_KEPT_OUTPUT_BYTES = 4096  # how much of the job's last output an error quotes
_READ_CHUNK = 1 << 16

# The job chooses two texts that reach the log: what it asked for when that was
# not handed to it, and the identifier of a model. Both are bounded, so that a job
# cannot grow the log, or write a record's text into it at length.
MAX_LOGGED_TEXT = 200


class _TimeLimit(Exception):
    """The run's time limit passed while the broker waited on a pipe."""


def _ref_json(ref: OccurrenceRef) -> dict:
    return {"instance_id": ref.instance_id, "locator": ref.locator}


def _as_ref(value: object) -> OccurrenceRef | None:
    """A reference as it travels on the pipe, or None for anything else."""
    if (
        isinstance(value, dict)
        and set(value) == {"instance_id", "locator"}
        and isinstance(value["instance_id"], str)
        and isinstance(value["locator"], str)
    ):
        return OccurrenceRef(value["instance_id"], value["locator"])
    return None


def _asked_text(value: object) -> str:
    """How a request for something that was not handed is named in the log: the
    text itself, or the JSON form of anything else; when that is longer than
    ``MAX_LOGGED_TEXT`` characters, its start followed by the SHA-256 of all of
    it, so that two long requests can still be told apart."""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except RecursionError:
            raise JobProtocolViolation("a request is nested too deeply") from None
    if len(value) <= MAX_LOGGED_TEXT:
        return value
    digest = hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{value[:MAX_LOGGED_TEXT]}...(sha256:{digest})"


class _Pipes:
    """The broker's ends of the two pipes, read and written under the time limit."""

    def __init__(self, read_fd: int, write_fd: int, deadline: float, max_line: int) -> None:
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._deadline = deadline
        self._max_line = max_line
        self._buffer = bytearray()
        self._at_end = False
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)

    def _wait(self, readable: bool) -> None:
        left = self._deadline - time.monotonic()
        if left <= 0:
            raise _TimeLimit()
        poller = select.poll()
        if readable:
            poller.register(self._read_fd, select.POLLIN)
        else:
            poller.register(self._write_fd, select.POLLOUT)
        # A closed far end shows as an event too; the read or the write that
        # follows then reports it.
        if not poller.poll(left * 1000):
            raise _TimeLimit()

    def read_line(self) -> bytes | None:
        """The next line without its line feed, or None when the job has closed
        its end. Bytes after the last line feed count as a line."""
        while True:
            cut = self._buffer.find(b"\n")
            if cut >= 0:
                line = bytes(self._buffer[:cut])
                del self._buffer[: cut + 1]
                return line
            if len(self._buffer) > self._max_line:
                raise JobProtocolViolation(
                    f"a request line is longer than the limit of {self._max_line} bytes"
                )
            if self._at_end:
                if not self._buffer:
                    return None
                line = bytes(self._buffer)
                self._buffer.clear()
                return line
            self._wait(readable=True)
            try:
                chunk = os.read(self._read_fd, _READ_CHUNK)
            except BlockingIOError:
                continue
            if not chunk:
                self._at_end = True
            self._buffer += chunk

    def write_line(self, message: dict) -> None:
        """Write one reply. A job that has closed its end shows as a broken pipe,
        which is raised as it is."""
        data = json.dumps(message, ensure_ascii=True, separators=(",", ":")).encode("ascii")
        view = memoryview(data + b"\n")
        while view:
            self._wait(readable=False)
            try:
                written = os.write(self._write_fd, view)
            except BlockingIOError:
                continue
            view = view[written:]


class _Tail:
    """Drains the job's standard output and standard error and keeps the end of it,
    so that a full pipe never stops the job and an error can quote what it said."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._kept = bytearray()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            chunk = self._stream.read1(_READ_CHUNK)
            if not chunk:
                return
            self._kept += chunk
            del self._kept[:-_KEPT_OUTPUT_BYTES]

    def text(self) -> str:
        self._thread.join(timeout=5)
        return bytes(self._kept).decode("utf-8", "replace").strip()


def _field(request: dict, name: str, kind: type, op: str) -> object:
    value = request.get(name)
    if not isinstance(value, kind):
        raise JobProtocolViolation(
            f"a {op!r} request needs the field {name!r} as {kind.__name__}", op=op
        )
    return value


class _Broker:
    """Answers one confined job's requests from the harness's context."""

    def __init__(self, job_name: str, context: JobContext, pipes: _Pipes) -> None:
        self._job_name = job_name
        self._context = context
        self._handed = set(context.handed())
        self._pipes = pipes
        self.outputs: list[JobOutput] = []

    def greet(self) -> None:
        self._pipes.write_line(
            {
                "protocol": PROTOCOL,
                "job_name": self._job_name,
                "handed": [_ref_json(ref) for ref in self._context.handed()],
            }
        )

    def serve(self) -> bool:
        """Answer requests until the job ends or closes its pipe. True when the
        job sent its end message and nothing after it."""
        while True:
            line = self._pipes.read_line()
            if line is None:
                return False
            request = self._parse(line)
            op = request.get("op")
            if op == _OP_READ:
                reply = self._read(request)
            elif op == _OP_MODEL:
                reply = self._model(request)
            elif op == _OP_OUTPUT:
                reply = self._output(request)
            elif op == _OP_END:
                self._pipes.write_line({"ok": True})
                if self._pipes.read_line() is not None:
                    raise JobProtocolViolation("the job wrote on its pipe after its end message")
                return True
            else:
                raise JobProtocolViolation(f"unknown request {_asked_text(op)[:80]!r}")
            self._pipes.write_line(reply)

    @staticmethod
    def _parse(line: bytes) -> dict:
        try:
            request = json.loads(line.decode("utf-8"))
        except ValueError as error:
            raise JobProtocolViolation(f"a request line is not JSON ({error})") from None
        except RecursionError:
            raise JobProtocolViolation("a request line is nested too deeply") from None
        if not isinstance(request, dict):
            raise JobProtocolViolation("a request line is not a JSON object")
        return request

    def _read(self, request: dict) -> dict:
        if "ref" not in request:
            raise JobProtocolViolation("a 'read' request needs the field 'ref'", op=_OP_READ)
        asked = request["ref"]
        ref: object = _as_ref(asked)
        if ref is None:
            ref = _asked_text(asked)
        elif ref not in self._handed:
            ref = _asked_text(ref.key)
        try:
            # The context writes the read to the log, on disk, before it returns
            # the text. For anything that was not handed it writes the refused
            # attempt and raises; the text given to it here is what the log names.
            text = self._context.read(ref)  # type: ignore[arg-type]
        except HarnessRefused as refused:
            return {"ok": False, "error": refused.name, "detail": refused.detail}
        return {"ok": True, "text": text}

    def _model(self, request: dict) -> dict:
        prompt = _field(request, "prompt", str, _OP_MODEL)
        if "model" not in request:
            raise JobProtocolViolation("a 'model' request needs the field 'model'", op=_OP_MODEL)
        model = request["model"]
        if isinstance(model, str) and len(model) > MAX_LOGGED_TEXT:
            return {
                "ok": False,
                "error": InvalidRecord.name,
                "detail": f"a model identifier has at most {MAX_LOGGED_TEXT} characters",
            }
        try:
            # The context writes the identifier to the log before it calls the
            # caller's client. A missing client, and an error of the client
            # itself, end the run: they are raised through to the caller.
            text = self._context.call_model(model, prompt)
        except InvalidRecord as refused:
            if refused.context.get("client"):
                # The caller's client returned something that cannot be logged.
                # That ends the run; it is no fault of the job's request.
                raise
            # An identifier of a reserved form. Nothing was logged or called.
            return {"ok": False, "error": refused.name, "detail": refused.detail}
        if not isinstance(text, str):
            raise InvalidRecord(
                f"the model client of job {self._job_name!r} returned "
                f"{type(text).__name__}; a model client returns text",
                job=self._job_name,
            )
        return {"ok": True, "text": text}

    def _output(self, request: dict) -> dict:
        text = _field(request, "text", str, _OP_OUTPUT)
        stated = _field(request, "used", list, _OP_OUTPUT)
        index = len(self.outputs)
        used: list[OccurrenceRef] = []
        problem: str | None = None
        for item in stated:
            ref = _as_ref(item)
            if ref is None or ref not in self._handed:
                named = _asked_text(ref.key if ref is not None else item)
                problem = (
                    f"output {index} of job {self._job_name!r} states it used {named}, "
                    f"which was not handed to the job"
                )
                break
            if ref in used:
                problem = f"output {index} of job {self._job_name!r} states a used record twice"
                break
            used.append(ref)
        if problem is not None:
            # The same refusal the in-process harness gives such an output. The
            # job is told, and the run ends.
            try:
                self._pipes.write_line(
                    {"ok": False, "error": InvalidRecord.name, "detail": problem}
                )
            except BrokenPipeError:
                pass
            raise InvalidRecord(problem, job=self._job_name)
        self.outputs.append(JobOutput(text=text, used=tuple(used)))
        return {"ok": True, "index": index}


def _close(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _run_in_sandbox(
    box: Sandbox,
    program: Program,
    context: JobContext,
    *,
    job_name: str,
    timeout_seconds: float,
    scratch_bytes: int,
    max_request_bytes: int,
) -> tuple[JobOutput, ...]:
    """The broker: start ``program`` in the sandbox, serve it, and return its
    outputs only when it ended as a job must."""
    request_read, request_write = os.pipe()  # the job writes, the broker reads
    reply_read, reply_write = os.pipe()  # the broker writes, the job reads
    process: subprocess.Popen | None = None
    tail: _Tail | None = None
    limit = f"its time limit of {timeout_seconds:g} seconds"
    try:
        argv = box.argv(
            program,
            request_fd=request_write,
            reply_fd=reply_read,
            scratch_bytes=scratch_bytes,
        )
        deadline = time.monotonic() + timeout_seconds
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            pass_fds=(request_write, reply_read),
            close_fds=True,
            env={},
        )
        tail = _Tail(process.stdout)
        # The broker keeps only its own ends, so that the job's exit closes the
        # request pipe and shows here as its end.
        _close(request_write)
        _close(reply_read)
        request_write = reply_read = -1
        pipes = _Pipes(request_read, reply_write, deadline, max_request_bytes)
        broker = _Broker(job_name, context, pipes)
        try:
            broker.greet()
            ended = broker.serve()
        except _TimeLimit:
            raise ConfinedJobFailed(
                f"job {job_name!r} ran past {limit} and was stopped", job=job_name
            ) from None
        except BrokenPipeError:
            ended = False
        try:
            status = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            raise ConfinedJobFailed(
                f"job {job_name!r} ran past {limit} and was stopped", job=job_name
            ) from None
        if status != 0:
            said = tail.text()
            raise ConfinedJobFailed(
                f"job {job_name!r} ended with exit status {status}"
                + (f"; it said last: {said}" if said else ""),
                job=job_name,
                status=status,
            )
        if not ended:
            raise ConfinedJobFailed(
                f"job {job_name!r} closed its pipe without the end message", job=job_name
            )
        return tuple(broker.outputs)
    finally:
        if process is not None and process.poll() is None:
            process.kill()  # the sandbox takes every process in it down with it
            process.wait()
        for fd in (request_read, request_write, reply_read, reply_write):
            if fd >= 0:
                _close(fd)
        if process is not None and process.stdout is not None:
            if tail is not None:
                tail.text()
            process.stdout.close()


def run_confined_job(
    program: Program,
    *,
    job_name: str,
    handed: Sequence[Handed],
    log_path: str | os.PathLike,
    timeout_seconds: float,
    model_client: ModelClient | None = None,
    scratch_bytes: int = DEFAULT_SCRATCH_BYTES,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> JobResult:
    """Run a job out of process, confined, and return its outputs and its log.

    ``program`` is what runs in the sandbox (``python_program`` builds one for a
    Python job). ``handed`` is the hand-off: the job gets the text of one of these
    records, and of no other, by asking for it. ``log_path`` names the log file to
    write; the file is the same kind ``harness.run_job`` writes. ``model_client``
    is the function ``(model, prompt) -> text`` the job's model calls are
    forwarded to; it runs here, outside the sandbox, and may return a
    ``harness.ModelReply`` to report which model answered. The log's job entry
    names the harness "confined". ``timeout_seconds`` bounds
    the whole run, model calls included; a job still running then is stopped.

    Raises ``SandboxUnavailable`` before anything is written when the sandbox
    cannot be built here. Raises ``ConfinedJobFailed`` or ``JobProtocolViolation``
    when the job did not end as a job must; the log is then closed with status
    "failed" and no output is returned. Everything ``harness.run_job`` refuses is
    refused here too."""
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not timeout_seconds > 0
    ):
        raise InvalidRecord(f"a confined job has a time limit above 0 seconds: {timeout_seconds!r}")
    for name, value in (("scratch_bytes", scratch_bytes), ("max_request_bytes", max_request_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidRecord(f"{name} is a whole number above 0: {value!r}")
    _sandbox._check_program(program)
    box = _sandbox.find_sandbox()

    def broker(context: JobContext) -> tuple[JobOutput, ...]:
        return _run_in_sandbox(
            box,
            program,
            context,
            job_name=job_name,
            timeout_seconds=float(timeout_seconds),
            scratch_bytes=scratch_bytes,
            max_request_bytes=max_request_bytes,
        )

    return harness._run_job(
        broker,
        job_name=job_name,
        handed=handed,
        log_path=log_path,
        model_client=model_client,
        harness=v.HARNESS_CONFINED,
    )
