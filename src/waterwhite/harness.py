"""The closed-input harness for an in-process generation job (plan 3.3 "Generation",
Decision 13, F10).

``run_job`` hands a job a ``JobContext``. Through it the job can read only the
records it was handed. Every read is written to a log file before the text is
returned; a read of anything else is refused with ``HarnessRefused`` and the attempt
is written to the log too. A model call goes through ``JobContext.call_model``,
which writes the model's identifier to the log before the call is made.

What this harness does NOT establish: the job is a Python callable in the caller's
own process, so it can open files and sockets by routes that never pass the
context. The log shows the reads that went through the harness and nothing else.
It is the closed-input harness of the plan only for a job that is confined by some
other means. ``waterwhite.confine`` runs a job out of process, confined, through
this same code (plan 5.8 B-1, F10). Every log says which of the two harnesses
wrote it. docs/capture.md and docs/confine.md say this at length.

The log file is JSON lines, one entry per line, in the canonical JSON form of
``ids.canonical``. It holds references and never a record's text, and it holds no
clock time, so the same job with the same hand-off and the same reads gives the
same bytes and the same digest.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import ids
from . import vocabulary as v
from .errors import (
    HarnessRefused,
    InvalidRecord,
    ModelClientMissing,
    WaterwhiteError,
    WriteRefused,
)
from .records import HarnessLog, HarnessRead, OccurrenceRef

__all__ = [
    "Handed",
    "HarnessLog",
    "HarnessRead",
    "HarnessRefused",
    "JobContext",
    "JobOutput",
    "JobResult",
    "LOG_FORMAT",
    "MODEL_NONE",
    "ModelClient",
    "ModelClientMissing",
    "ModelReply",
    "OccurrenceRef",
    "model_field",
    "read_log",
    "run_job",
]

LOG_FORMAT = "waterwhite.harness-log/1"

# The model field of a log whose job called no model (plan 3.3, F10).
MODEL_NONE = "none"

ENTRY_JOB = "job"  # the first entry: the job's name and its hand-off
ENTRY_READ = "read"  # one read, allowed or refused
ENTRY_MODEL_CALL = "model_call"  # one call through the model client
ENTRY_MODEL_REPLY = "model_reply"  # the model the client reports answered a call
ENTRY_OUTPUT = "output"  # one output and the handed records it states it used
ENTRY_MODEL = "model"  # the model entry: what the job called, or "none"
ENTRY_END = "end"  # the last entry: "completed" or "failed"

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"



@dataclass(frozen=True)
class ModelReply:
    """What a model client may return in place of plain text: the text, and the
    identifier of the model that answered, as the client reports it. The harness
    then logs that identifier beside the one the job asked for, and the log's
    model entry names the reported one."""

    text: str
    model: str


ModelClient = Callable[[str, str], "str | ModelReply"]


@dataclass(frozen=True)
class Handed:
    """One record handed to a job: where it sits and its text."""

    ref: OccurrenceRef
    text: str


@dataclass(frozen=True)
class JobOutput:
    """One output of a job and the handed records it states it used.

    ``used`` is what the output states, and the log keeps it as stated. Which
    handed records become the output's parents is decided where the edge is built
    (``capture.generation_edge``): the stated ones, and for a job handed exactly
    one record that record even when the output states nothing (DECISIONS.md D-21)."""

    text: str
    used: tuple[OccurrenceRef, ...]


@dataclass(frozen=True)
class JobResult:
    """What ``run_job`` returns: the outputs in the job's order and the log, read
    back from the file the harness wrote. The log carries the texts the job was
    handed in ``handed_texts``, which the file does not hold; a log read from the
    file later (``read_log``) carries none."""

    outputs: tuple[JobOutput, ...]
    log: HarnessLog


def _ref_json(ref: OccurrenceRef) -> dict:
    return {"instance_id": ref.instance_id, "locator": ref.locator}


def _asked_for(ref: object) -> str:
    """How a refused request is named in the log."""
    if isinstance(ref, OccurrenceRef):
        return ref.key
    if isinstance(ref, str):
        return ref
    return repr(ref)


def model_field(called: Sequence[str]) -> str:
    """The model field of a log from the models a job called, in call order.

    No call: ``"none"``. One model, however often: its identifier. Two or more
    different models: a JSON list of the identifiers in the order of their first
    call, for example ``["m-1","m-2"]``. An identifier may not start with ``[``, so
    the list form cannot be mistaken for one model."""
    distinct = list(dict.fromkeys(called))
    if not distinct:
        return MODEL_NONE
    if len(distinct) == 1:
        return distinct[0]
    return ids.canonical(distinct).decode("ascii")


def _check_model_id(model: object) -> str:
    if (
        not isinstance(model, str)
        or not model
        or model == MODEL_NONE
        or model.startswith("[")
        or model != model.strip()
    ):
        raise InvalidRecord(
            f"a model identifier is non-empty text with no space at its ends, is not "
            f"{MODEL_NONE!r} and does not start with '[': {model!r}"
        )
    return model


class _LogWriter:
    """Appends entries to the log file. An entry is on disk when ``write`` returns."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = open(path, "xb")
        except FileExistsError:
            raise WriteRefused(
                f"a harness log exists at {path}; a log is evidence and is not overwritten",
                path=str(path),
            ) from None
        if os.name == "posix":
            # Make the new file's directory entry durable as well as its bytes.
            folder = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(folder)
            finally:
                os.close(folder)
        self._seq = 0

    def write(self, kind: str, **fields: object) -> int:
        """Append one entry and return its sequence number."""
        seq = self._seq
        entry = {"seq": seq, "kind": kind, **fields}
        self._file.write(ids.canonical(entry) + b"\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._seq += 1
        return seq

    def close(self) -> None:
        self._file.close()


class JobContext:
    """What a job is given: its hand-off, and nothing else.

    A job does not build one; ``run_job`` does. After the job has ended the context
    is closed and delivers nothing."""

    def __init__(
        self,
        job_name: str,
        handed: Sequence[Handed],
        writer: _LogWriter,
        model_client: ModelClient | None,
    ) -> None:
        self._job_name = job_name
        self._order = tuple(h.ref for h in handed)
        self._texts = {h.ref: h.text for h in handed}
        self._writer = writer
        self._model_client = model_client
        self._called: list[str] = []
        self._closed = False
        self._lock = threading.Lock()

    def handed(self) -> tuple[OccurrenceRef, ...]:
        """The references of the handed records, in hand-off order. Listing them is
        not a read and is not logged."""
        return self._order

    def read(self, ref: OccurrenceRef) -> str:
        """The text of a handed record. The read is in the log, on disk, before the
        text is returned. A reference outside the hand-off is refused with
        ``HarnessRefused``, and the attempt is logged with ``refused`` true."""
        with self._lock:
            if self._closed:
                raise HarnessRefused(
                    f"job {self._job_name!r} has ended; its context delivers nothing",
                    job=self._job_name,
                )
            if isinstance(ref, OccurrenceRef) and ref in self._texts:
                self._writer.write(ENTRY_READ, refused=False, **_ref_json(ref))
                return self._texts[ref]
            asked = _asked_for(ref)
            self._writer.write(ENTRY_READ, refused=True, outside_ref=asked)
        raise HarnessRefused(
            f"{asked} was not handed to job {self._job_name!r}",
            job=self._job_name,
            ref=asked,
        )

    def call_model(self, model: str, prompt: str) -> str:
        """Call a model through the client the harness was given. The identifier is
        in the log, on disk, before the client is called, so the log names every
        model the job asked for, whether or not the call returned.

        A client that returns a ``ModelReply`` reports which model answered. That
        identifier is written to the log after the call, tied to the call's
        entry, and the log's model entry then names it in place of the one the
        job asked for: the job chooses the text of its request, the client knows
        what it called."""
        with self._lock:
            if self._closed:
                raise HarnessRefused(
                    f"job {self._job_name!r} has ended; its context calls no model",
                    job=self._job_name,
                )
            model = _check_model_id(model)
            call = self._writer.write(ENTRY_MODEL_CALL, model=model)
            place = len(self._called)
            self._called.append(model)
            client = self._model_client
        if client is None:
            raise ModelClientMissing(
                f"job {self._job_name!r} called the model {model!r} and run_job was given "
                f"no model_client",
                job=self._job_name,
                model=model,
            )
        reply = client(model, prompt)
        if not isinstance(reply, ModelReply):
            return reply
        if not isinstance(reply.text, str):
            raise InvalidRecord(
                f"the model client of job {self._job_name!r} returned a reply whose text "
                f"is {type(reply.text).__name__}; a model client returns text",
                job=self._job_name, client=True,
            )
        try:
            reported = _check_model_id(reply.model)
        except InvalidRecord as error:
            raise InvalidRecord(
                f"the model client of job {self._job_name!r} reported a model that "
                f"cannot be logged: {error.detail}",
                job=self._job_name, client=True,
            ) from None
        with self._lock:
            if self._closed:
                raise HarnessRefused(
                    f"job {self._job_name!r} has ended; its context logs no model reply",
                    job=self._job_name,
                )
            self._writer.write(ENTRY_MODEL_REPLY, call=call, model=reported)
            self._called[place] = reported
        return reply.text

    def _finish(self) -> tuple[str, list[str]]:
        """Close the context; return the model field and the distinct models."""
        self._closed = True
        return model_field(self._called), list(dict.fromkeys(self._called))


def _check_handed(handed: Sequence[Handed]) -> tuple[Handed, ...]:
    checked = tuple(handed)
    seen: set[OccurrenceRef] = set()
    for item in checked:
        if (
            not isinstance(item, Handed)
            or not isinstance(item.ref, OccurrenceRef)
            or not isinstance(item.text, str)
        ):
            raise InvalidRecord(f"a hand-off holds Handed(ref, text) items: {item!r}")
        if item.ref in seen:
            raise InvalidRecord(f"{item.ref.key} is handed twice", ref=item.ref.key)
        seen.add(item.ref)
    return checked


def _check_outputs(job_name: str, returned: object, handed: set[OccurrenceRef]) -> tuple:
    if isinstance(returned, JobOutput):
        returned = (returned,)
    try:
        outputs = tuple(returned)  # type: ignore[arg-type]
    except TypeError:
        raise InvalidRecord(
            f"job {job_name!r} returned {type(returned).__name__}; a job returns a "
            f"JobOutput or a sequence of them",
            job=job_name,
        ) from None
    for index, output in enumerate(outputs):
        if not isinstance(output, JobOutput) or not isinstance(output.text, str):
            raise InvalidRecord(
                f"output {index} of job {job_name!r} is not a JobOutput with a text",
                job=job_name,
            )
        if not isinstance(output.used, tuple):
            raise InvalidRecord(
                f"output {index} of job {job_name!r}: used is a tuple of references",
                job=job_name,
            )
        for ref in output.used:
            if not isinstance(ref, OccurrenceRef) or ref not in handed:
                raise InvalidRecord(
                    f"output {index} of job {job_name!r} states it used {_asked_for(ref)}, "
                    f"which was not handed to the job",
                    job=job_name,
                    ref=_asked_for(ref),
                )
        if len(set(output.used)) != len(output.used):
            raise InvalidRecord(
                f"output {index} of job {job_name!r} states a used record twice",
                job=job_name,
            )
    return outputs


def run_job(
    job: Callable[[JobContext], object],
    *,
    job_name: str,
    handed: Sequence[Handed],
    log_path: str | os.PathLike,
    model_client: ModelClient | None = None,
) -> JobResult:
    """Run an in-process job in the harness and return its outputs and its log.

    ``job`` is called once with a ``JobContext`` and returns a ``JobOutput`` or a
    sequence of them. ``handed`` is the hand-off. ``log_path`` names the log file to
    write; a file that already exists there is refused. ``model_client`` is the
    function ``(model, prompt) -> text`` behind ``JobContext.call_model``; a job
    that calls no model needs none. It may return a ``ModelReply`` in place of
    the text, to report which model answered.

    The log's job entry always says "in-process": this function runs the job in
    the caller's own process, and a log it writes never says more (Decision 13: a
    bound is established only by a log the confined harness wrote). It takes no
    argument that could change the word. ``waterwhite.confine`` reaches the code
    below through ``_run_job``, which is not part of the surface, and is the only
    writer of "confined".

    An output that states a used record that was not handed is refused with
    ``InvalidRecord``. When the job raises, or its outputs are refused, the log is
    closed with status "failed" and the error is raised again; the log on disk still
    shows every read up to then (``read_log`` reads it)."""
    return _run_job(job, job_name=job_name, handed=handed, log_path=log_path,
                    model_client=model_client, harness=v.HARNESS_IN_PROCESS)


def _run_job(
    job: Callable[[JobContext], object],
    *,
    job_name: str,
    handed: Sequence[Handed],
    log_path: str | os.PathLike,
    model_client: ModelClient | None,
    harness: str,
) -> JobResult:
    """The code of ``run_job``, with the word the log's job entry carries for the
    harness. Not public: only ``waterwhite.confine`` calls it with "confined", for
    the broker that runs the job out of process, and ``run_job`` with "in-process"."""
    if not isinstance(job_name, str) or not job_name:
        raise InvalidRecord("a job has a name")
    if harness not in v.HARNESS_KINDS:
        raise InvalidRecord(
            f"a harness is one of {', '.join(repr(k) for k in v.HARNESS_KINDS)}, "
            f"not {harness!r}"
        )
    items = _check_handed(handed)
    path = Path(log_path)
    writer = _LogWriter(path)
    try:
        writer.write(
            ENTRY_JOB,
            format=LOG_FORMAT,
            job_name=job_name,
            handed=[_ref_json(h.ref) for h in items],
            harness=harness,
        )
        context = JobContext(job_name, items, writer, model_client)
        try:
            outputs = _check_outputs(job_name, job(context), set(context.handed()))
        except BaseException as error:
            model, models = context._finish()
            writer.write(ENTRY_MODEL, model=model, models=models)
            writer.write(
                ENTRY_END,
                status=STATUS_FAILED,
                error=error.name if isinstance(error, WaterwhiteError) else type(error).__name__,
            )
            raise
        model, models = context._finish()
        for index, output in enumerate(outputs):
            writer.write(ENTRY_OUTPUT, index=index, used=[_ref_json(r) for r in output.used])
        writer.write(ENTRY_MODEL, model=model, models=models)
        writer.write(ENTRY_END, status=STATUS_COMPLETED, outputs=len(outputs))
    finally:
        writer.close()
    # The file holds references and never a text. The log this call returns carries
    # the texts it was handed, in hand-off order, outside the log's equality, its
    # representation and its digest, so that the registration of the output can
    # compare them with the records the references name.
    log = dataclasses.replace(
        read_log(path), handed_texts=tuple((item.ref, item.text) for item in items)
    )
    return JobResult(outputs=outputs, log=log)


def _parse_ref(entry: dict, where: str) -> OccurrenceRef:
    instance, locator = entry.get("instance_id"), entry.get("locator")
    if not isinstance(instance, str) or not isinstance(locator, str):
        raise InvalidRecord(f"{where}: a reference needs instance_id and locator")
    return OccurrenceRef(instance, locator)


def read_log(log_path: str | os.PathLike) -> HarnessLog:
    """Read a harness log file into the record the store keeps (plan 3.3, F10).

    ``log_digest`` is the SHA-256 of the file as it is on disk. ``model`` comes from
    the file's model entry: ``"none"``, an identifier, or the list form of
    ``model_field``. A file with no model entry gives ``None``, which is read as
    "not established" and never as "no model". A model entry that disagrees with the
    file's own call and reply entries is refused, and so is a file that is not a
    harness log. ``harness`` comes from the job entry and is None for a log that
    does not say which harness wrote it."""
    path = Path(log_path)
    data = path.read_bytes()
    job_name: str | None = None
    handed: tuple[OccurrenceRef, ...] = ()
    reads: list[HarnessRead] = []
    called: list[str] = []
    call_place: dict[int, int] = {}
    replied: set[int] = set()
    kind_of_harness: str | None = None
    model: str | None = None
    has_model_entry = False
    last_seq = -1
    for number, raw in enumerate(data.split(b"\n"), start=1):
        if not raw:
            continue
        where = f"{path}, line {number}"
        try:
            entry = json.loads(raw.decode("utf-8"))
        except ValueError as error:
            raise InvalidRecord(f"{where}: not a harness log entry ({error})") from None
        if not isinstance(entry, dict):
            raise InvalidRecord(f"{where}: not a harness log entry")
        seq, kind = entry.get("seq"), entry.get("kind")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= last_seq:
            raise InvalidRecord(f"{where}: entries are out of order")
        last_seq = seq
        if job_name is None:
            if kind != ENTRY_JOB or entry.get("format") != LOG_FORMAT:
                raise InvalidRecord(f"{where}: a harness log starts with its job entry")
            name, listed = entry.get("job_name"), entry.get("handed")
            if not isinstance(name, str) or not name or not isinstance(listed, list):
                raise InvalidRecord(f"{where}: the job entry needs job_name and handed")
            job_name = name
            handed = tuple(_parse_ref(item, where) for item in listed)
            kind_of_harness = entry.get("harness")
            if kind_of_harness is not None and kind_of_harness not in v.HARNESS_KINDS:
                raise InvalidRecord(
                    f"{where}: the job entry names the harness {kind_of_harness!r}; a "
                    f"harness is one of {', '.join(repr(k) for k in v.HARNESS_KINDS)}"
                )
        elif kind == ENTRY_READ:
            refused = entry.get("refused")
            if not isinstance(refused, bool):
                raise InvalidRecord(f"{where}: a read entry says whether it was refused")
            if "outside_ref" in entry:
                outside = entry["outside_ref"]
                if not isinstance(outside, str):
                    raise InvalidRecord(f"{where}: outside_ref is text")
                reads.append(HarnessRead(occurrence=None, outside_ref=outside, refused=refused))
            else:
                reads.append(HarnessRead(occurrence=_parse_ref(entry, where), refused=refused))
        elif kind == ENTRY_MODEL_CALL:
            call_place[seq] = len(called)
            called.append(_check_model_id(entry.get("model")))
        elif kind == ENTRY_MODEL_REPLY:
            call = entry.get("call")
            if isinstance(call, bool) or call not in call_place or call in replied:
                raise InvalidRecord(
                    f"{where}: a model reply names one call entry before it, once"
                )
            replied.add(call)
            called[call_place[call]] = _check_model_id(entry.get("model"))
        elif kind == ENTRY_MODEL:
            if has_model_entry or not isinstance(entry.get("model"), str):
                raise InvalidRecord(f"{where}: a log has one model entry, and it is text")
            has_model_entry = True
            model = entry["model"]
        elif kind in (ENTRY_OUTPUT, ENTRY_END):
            continue
        else:
            raise InvalidRecord(f"{where}: unknown entry kind {kind!r}")
    if job_name is None:
        raise InvalidRecord(f"{path}: not a harness log (no job entry)")
    if has_model_entry and model != model_field(called):
        raise InvalidRecord(
            f"{path}: the model entry says {model!r} and the call and reply entries say "
            f"{model_field(called)!r}",
            job=job_name,
        )
    return HarnessLog(
        job_name=job_name,
        log_digest=ids.bytes_digest(data),
        handed=handed,
        reads=tuple(reads),
        model=model,
        harness=kind_of_harness,
    )
