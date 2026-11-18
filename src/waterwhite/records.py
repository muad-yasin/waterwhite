"""Record types: what a capture point hands the store, and what a query hands back.

These are plain data. The store writes them (store/), the capture wrappers and the
recovery build them (capture, recovery), the proof reads them (coverage). The
shapes follow plan 3.1; the JSON form of an answer follows plan 2.1 and 2.4 and is
described in docs/interfaces.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import vocabulary as v

# --- References ---------------------------------------------------------------------

# The instance id "" names the artifact itself. It is written in a reference that
# is handed to the store with an artifact's own records (an edge's parent, a handed
# record or a read of a harness log) when the record meant lies in the very
# artifact being written: that artifact's instance id is computed from its edge
# records and so does not exist while they are written (plan 3.1). The store puts
# the real id in its place once it is known, and every reference read back from
# the store carries a full instance id. In the edge digest such a parent is named
# by its locator alone. Nothing else may carry an empty instance id.
OWN_ARTIFACT = ""


@dataclass(frozen=True)
class OccurrenceRef:
    """A record at a place: an artifact instance plus a locator (plan 3.1).

    ``instance_id`` is an artifact's instance id, or ``OWN_ARTIFACT`` (the empty
    string) for a record of the artifact that is being written."""

    instance_id: str
    locator: str

    @property
    def key(self) -> str:
        return f"{self.instance_id}:{self.locator}"


@dataclass(frozen=True)
class StepRef:
    """The step that made an artifact or a link: name, version, parameter digest."""

    name: str
    version: str
    param_digest: str


# --- What capture writes -------------------------------------------------------------


@dataclass(frozen=True)
class OccurrenceRecord:
    """One record of a JSONL artifact as written to the store."""

    locator: str
    fingerprint: bytes | None  # None for a line that holds no text


@dataclass(frozen=True)
class EdgeInput:
    """A record an edge names.

    ``used`` True: a parent, the sole input or one the output states it used.
    ``used`` False: handed to the job and not stated as used; in scope only and not a
    parent (plan 3.1, Decision 25). A job handed exactly one record has that record
    as a parent of each of its outputs, whatever the output states (DECISIONS.md
    D-21). ``selected`` marks the one parent the step of an "any" edge kept (plan
    3.1, Decision 29).

    The parent of a stated edge lies in an artifact the step declared as an input,
    or in the artifact being written (``OWN_ARTIFACT``); the store refuses any
    other (DECISIONS.md D-15). An edge derived afterwards may name any registered
    record."""

    parent: OccurrenceRef
    used: bool = True
    selected: bool = False


@dataclass(frozen=True)
class HarnessRead:
    """One read the closed-input harness logged. ``occurrence`` is set for a handed
    record; ``outside_ref`` names what was asked for when it lay outside the
    hand-off, and ``refused`` says whether the harness refused it."""

    occurrence: OccurrenceRef | None = None
    outside_ref: str | None = None
    refused: bool = False


@dataclass(frozen=True)
class HarnessLog:
    """The stored log of one generation job (plan 3.3, F10).

    ``model`` is the identifier of the model the job called through the client the
    harness handed it, ``"none"`` when the log shows it called no model, and None
    when the log has no model entry (read as not established, never as no model).
    Where the client reported which model answered a call, the identifier is the
    reported one.

    ``harness`` says which harness wrote the log: ``vocabulary.HARNESS_IN_PROCESS``
    or ``vocabulary.HARNESS_CONFINED``, and None when the log does not say. Only a
    confined log establishes a bound (``bound_established``; Decision 13, plan F10;
    docs/confine.md section 5).

    ``handed_texts`` is what ``run_job`` and ``run_confined_job`` were handed: each
    handed reference with its text, in hand-off order. It is not part of the log. The
    log file holds references and never a text, so a log read from its file carries
    none, and the store never keeps them: they are held on the log a job returns so
    that the registration of its output can compare them with the records the
    references name (docs/store.md, "The write methods"). They take no part in the
    log's equality, its representation or its digest.

    ``hand_off_checked`` is the store's own statement, never the caller's: True when
    the texts handed to the job were compared with the records their references name
    (a text for every reference of the hand-off, each matching its record; or
    ``Store.verify_hand_off`` did the same later), False for a log read back from its
    file, a log registered without every text, and every log stored under schema
    version 3 or earlier. A value given when a log is handed to the store is not read.
    A stored log establishes no bound while it is False (``bound_established``). It is
    not part of the log's equality, its representation or its digest."""

    job_name: str
    log_digest: str
    handed: tuple[OccurrenceRef, ...]
    reads: tuple[HarnessRead, ...]
    model: str | None
    harness: str | None = None
    handed_texts: tuple[tuple[OccurrenceRef, str], ...] = field(
        default=(), compare=False, repr=False
    )
    hand_off_checked: bool = field(default=False, compare=False, repr=False)

    def bound_stated(self) -> bool:
        """What the log itself says: the confined harness wrote it and it lists no read
        of a record outside the hand-off (``reads_within_hand_off``). The edge's own
        bound field is written from this at registration. A log the in-process
        harness wrote, or one that names no harness, states no bound: that harness
        cannot say what else the job opened (Decision 13, plan 3.3, DECISIONS.md
        D-27)."""
        return self.harness == v.HARNESS_CONFINED and self.reads_within_hand_off()

    def bound_established(self) -> bool:
        """True when the log states a bound (``bound_stated``) and the store compared
        the texts handed to the job with the records their references name
        (``hand_off_checked``). A confined log whose texts were never shown and
        compared establishes none: what the job was given is then only the caller's
        statement, and a negative past its edge is "may contain"."""
        return self.hand_off_checked and self.bound_stated()

    def reads_within_hand_off(self) -> bool:
        """True when the log lists no read of a record outside the hand-off that the
        harness let through, whichever harness wrote it. A refused attempt is logged
        and does not count: the job did not receive the record (DECISIONS.md D-07).
        This is what the log says; whether it is a bound is ``bound_established``."""
        handed = set(self.handed)
        for read in self.reads:
            if read.occurrence is not None and read.occurrence in handed:
                continue
            if read.refused:
                continue
            return False
        return True


@dataclass(frozen=True)
class EdgeRecord:
    """A child occurrence and the records its step names (plan 3.1, "Edge").

    ``job_name`` is the name of the generation job that made the child. A
    generation edge always carries it and no other edge does (DECISIONS.md D-24):
    the edge keeps the name itself, so an answer can still name the job when the
    job's harness log is gone (plan Check 5). For a job run in the harness it is
    the job name of ``harness_log``, and ``capture.generation_edge`` sets it from
    there. A pipeline whose job wrote its own parent list (a generation edge with
    no harness log) passes the name of that job.

    ``handed_texts`` holds the texts the job was handed, each with the reference it
    was handed under, as the job's log carries them; ``capture.generation_edge``
    copies them from the log. The store compares each with the record its reference
    names when it registers the edge, and refuses the registration when one is not
    that record; the texts the edge's log carries are compared the same way, each
    once. They are not stored and take no part in the edge's equality or
    representation. An edge and a log with none are registered without that
    comparison."""

    child_locator: str
    inputs: tuple[EdgeInput, ...]
    transform: StepRef
    parent_mode: str = v.MODE_ALL
    derivation: str = v.DERIVED_STATED
    verification: str | None = None
    label: str = v.OBSERVED
    is_generation: bool = False
    harness_log: HarnessLog | None = None
    job_name: str | None = None
    handed_texts: tuple[tuple[OccurrenceRef, str], ...] = field(
        default=(), compare=False, repr=False
    )


@dataclass(frozen=True)
class DropRecord:
    """An occurrence a step removed, with the reason (plan 3.1, "Drop")."""

    input: OccurrenceRef
    reason: str
    step_name: str
    group_key: str | None = None


@dataclass(frozen=True)
class MapRow:
    """One document of a tokenized shard (plan 3.3, "Tokenization").

    ``token_start`` and ``token_end`` are the document's span in the shard's token
    stream, half-open. ``parent`` is the JSONL occurrence the document was tokenized
    from when the label is "recovered"; ``candidates`` holds the twins' occurrences
    when it is "candidates"."""

    ordinal: int
    token_start: int
    token_end: int
    label: str
    parent: OccurrenceRef | None = None
    candidates: tuple[OccurrenceRef, ...] = ()
    fingerprint: bytes | None = None


@dataclass(frozen=True)
class MapRecord:
    """A tokenization map with the fields every map stores (plan 3.3, 3.4)."""

    namespace: str  # vocabulary.NAMESPACE_CAPTURE or NAMESPACE_RECOVERY
    derivation: str  # vocabulary.DERIVED_POSITION or DERIVED_TOKEN_MATCHING
    verification: str  # passed, failed or not run
    label: str  # the map's label, which follows from the two fields above
    append_eod: bool
    eod_in_span: bool
    rows: tuple[MapRow, ...]


@dataclass(frozen=True)
class SnapshotShard:
    """One shard of a mixture snapshot, in the trainer's order (Decision 27)."""

    position: int
    shard_instance_id: str
    path: str
    content_digest: str


# --- Step traces (plan 3.5) ------------------------------------------------------------


@dataclass(frozen=True)
class TraceRow:
    """A (shard, ordinal, token start, token end) row: the part of one document a
    sample read. The token range is inside the document, half-open."""

    shard_position: int
    ordinal: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class TraceSample:
    """One sample a step read: its rows and its coverage mark."""

    sample_pos: int
    global_index: int
    rank: int
    sample_tokens: int
    covered_tokens: int
    rows: tuple[TraceRow, ...]

    @property
    def uncovered(self) -> tuple[int, int] | None:
        """The uncovered token range of the sample, or None when it is whole."""
        if self.covered_tokens >= self.sample_tokens:
            return None
        return (self.covered_tokens, self.sample_tokens)


@dataclass(frozen=True)
class StepTrace:
    """What one completed step read, written once at registration (Decision 19)."""

    step: int
    consumed_samples: int
    samples: tuple[TraceSample, ...]


# --- Requests ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WithdrawRequest:
    """Withdraw a source (plan 2.3, Decision 24).

    ``source`` names the source; ``members`` narrows to named members (empty: every
    member registered under the source when the query runs); ``records`` narrows to
    a list of occurrences. The answer lists the members the request expanded to."""

    source: str
    members: tuple[str, ...] = ()
    records: tuple[OccurrenceRef, ...] = ()

    kind: str = field(default="withdraw", init=False)
