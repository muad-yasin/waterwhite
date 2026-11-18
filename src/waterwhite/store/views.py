"""Typed results of the store's read methods, and the fixed words the store keeps.

A read method never hands back a raw row. It hands back one of the plain data
types below, or a record type of ``waterwhite.records`` where one fits (``MapRow``,
``SnapshotShard``, ``StepTrace``, ``HarnessLog``).

Plan sections: 3.1 (records), 3.4 (the store), 2.1 (which artifact has a record of
its own and which is only named), 2.4 (a checkpoint's history).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..records import EdgeInput, HarnessLog, OccurrenceRef, StepRef

# The start kinds, the trace routes and the member part kinds are words of
# ``waterwhite.vocabulary``. They are named here as well, so that code written
# against this module before the vocabulary carried them keeps working; they are
# the same objects, not copies.
from ..vocabulary import (  # noqa: F401
    PART_FILE,
    PART_FOLDER,
    PART_KINDS,
    PART_RECORDS,
    PART_REVISION,
    ROUTE_CACHE_FILES,
    ROUTE_NONE,
    ROUTE_REPLAY,
    START_FRESH,
    START_KINDS,
    START_RESUME,
    START_WARM,
    TRACE_ROUTES,
)

# How one record names another by value (plan 2.1, the listing of Decision 26).
REL_MIXTURE_SHARD = "mixture shard"  # a snapshot names a shard instance
REL_SEGMENT_MIXTURE = "segment mixture"  # a run segment names its mixture instance
REL_CHECKPOINT_SEGMENT = "checkpoint segment"  # a checkpoint names its segment
REL_SEGMENT_PARENT = "segment parent checkpoint"  # a segment names where it started
REL_ARTIFACT_INPUT = "artifact input"  # an artifact names an input of its step

RELATIONS = (
    REL_MIXTURE_SHARD,
    REL_SEGMENT_MIXTURE,
    REL_CHECKPOINT_SEGMENT,
    REL_SEGMENT_PARENT,
    REL_ARTIFACT_INPUT,
)

# Where an artifact's file is when the store looks (see ``Store.file_state``).
FILE_PRESENT = "present"
FILE_ABSENT = "absent"  # archived or moved: a state the proof reads, not a fault
FILE_PARTIAL = "partial"  # a shard with one of its two files gone: a loud fault
FILE_NONE = "no file"  # a mixture or a checkpoint: the store holds no path for it


# --- Artifacts ----------------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """One registered artifact instance (plan 3.1, Decision 29)."""

    instance_id: str
    kind: str
    role: str | None
    content_digest: str
    path: str | None  # as stored: relative to the store's folder, or absolute
    step: StepRef | None  # None for an ingested file or an import
    member_name: str | None
    edge_digest: str
    names: tuple[str, ...]
    inputs: tuple[str, ...]  # the producing step's inputs, in order, by instance id


@dataclass(frozen=True)
class Reference:
    """One record naming another by value.

    ``referrer`` is the instance id of the naming artifact, or the name of the
    naming run segment. ``referred`` is the instance id that is named; for
    ``REL_CHECKPOINT_SEGMENT`` it is the segment's name. ``registered`` says whether
    the named thing has a row of its own."""

    relation: str
    referrer: str
    referred: str
    position: int | None
    registered: bool


@dataclass(frozen=True)
class Binding:
    """An artifact's own record of the records it holds (plan 3.4).

    For a JSONL artifact the map fields are None. ``label`` is returned as stored:
    a label edited after writing is not corrected here, proof condition 2.2(f)
    reads it."""

    instance_id: str
    namespace: str
    declared_count: int
    declared_digest: str
    derivation: str | None
    verification: str | None
    label: str | None
    append_eod: bool | None
    eod_in_span: bool | None


@dataclass(frozen=True)
class MissingFile:
    """An artifact whose file is not where the store recorded it."""

    instance_id: str
    kind: str
    path: str  # the resolved place the store looked
    state: str  # FILE_ABSENT or FILE_PARTIAL


# --- Occurrences, edges, links ------------------------------------------------------


@dataclass(frozen=True)
class Occurrence:
    """A record of a JSONL artifact: its place and its fingerprint."""

    ref: OccurrenceRef
    fingerprint: bytes | None


@dataclass(frozen=True)
class StoredEdge:
    """An edge as the store holds it (plan 3.1, "Edge").

    ``inputs`` holds every record the edge names, in order, each with its full
    instance id. ``bound_field`` is the stored field and nothing more: the proof
    derives the bound again from ``harness_log`` (plan 2.2(g)). ``job_name`` is
    the name of the generation job as the edge itself keeps it (DECISIONS.md
    D-24): it is there for every generation edge, also when ``harness_log`` is
    None because the log is gone, and None for every other edge."""

    child: OccurrenceRef
    inputs: tuple[EdgeInput, ...]
    transform: StepRef
    parent_mode: str
    derivation: str
    verification: str | None
    label: str
    is_generation: bool
    harness_log: HarnessLog | None
    bound_field: bool
    job_name: str | None = None

    @property
    def parents(self) -> tuple[OccurrenceRef, ...]:
        """The records the child was made from (``used`` inputs)."""
        return tuple(i.parent for i in self.inputs if i.used)

    @property
    def in_scope_only(self) -> tuple[OccurrenceRef, ...]:
        """Records handed to the job and not stated as used (plan Decision 25)."""
        return tuple(i.parent for i in self.inputs if not i.used)

    @property
    def selected(self) -> OccurrenceRef | None:
        """The one parent an "any" edge's step kept, or None."""
        for i in self.inputs:
            if i.selected:
                return i.parent
        return None


@dataclass(frozen=True)
class MapLink:
    """A shard document whose map row names a JSONL occurrence (the walk downward).

    ``via`` is "parent" for a row that names the occurrence as its one parent and
    "candidate" for a row that holds it in its candidate set. ``namespace`` and
    ``derivation`` together name the map the row belongs to: a shard can hold one
    map per namespace and derivation mode (DECISIONS.md D-20)."""

    shard_instance_id: str
    namespace: str
    ordinal: int
    label: str
    via: str
    derivation: str


@dataclass(frozen=True)
class SourceLink:
    """A link from a record, or from every record of a file (``locator`` None), to a
    source member with no parent record behind it (plan 2.2(b), Decision 3).

    ``label``, ``derivation`` and ``verification`` are returned as stored. The
    write API never stores a label the other two do not allow; one edited
    afterwards is not corrected here, proof condition 2.2(f) reads it."""

    instance_id: str
    locator: str | None
    member: str
    source: str
    label: str
    evidence: str
    derivation: str
    verification: str | None


@dataclass(frozen=True)
class Member:
    """A member of a source: a revision or a part of one (plan 2.3)."""

    name: str
    source: str
    revision: str
    part_kind: str
    parts: tuple[tuple[str, str], ...]  # (instance id, locator or "" for the whole file)


@dataclass(frozen=True)
class StoredDrop:
    """An occurrence a step removed (plan 3.1, "Drop")."""

    input: OccurrenceRef
    reason: str
    step_name: str
    output_instance_id: str | None
    group_key: str | None


# --- Runs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Launch:
    position: int
    world_size: int
    data_parallel_size: int
    consumed_at_start: int
    step_at_start: int
    first_step: int
    last_step: int


@dataclass(frozen=True)
class CacheDigest:
    kind: str
    shard_position: int | None  # None for the blend's two arrays
    digest: str


@dataclass(frozen=True)
class ShuffleDigest:
    launch: int
    epoch: int
    rank: int
    digest: str


@dataclass(frozen=True)
class SegmentStep:
    step: int
    batch_size: int
    last_sample_tokens_read: int | None


@dataclass(frozen=True)
class Segment:
    """A run segment with everything registered beside it (plan 3.1, 3.5).

    A field the run record did not hold is None: it is never guessed."""

    run: str
    name: str
    position: int
    mixture_instance_id: str | None
    parent_checkpoint: str | None
    start_kind: str
    run_record: dict | None
    run_record_digest: str | None
    sampler_type: str | None
    sampler_class: str | None
    data_sharding: bool | None
    micro_batch_size: int | None
    global_batch_size: int | None
    sequence_length: int | None
    add_extra_token: bool | None
    random_seed: int | None
    megatron_core_version: str | None
    megatron_bridge_version: str | None
    torch_version: str | None
    release_tag: str | None
    trace_route: str
    trace_cause: str | None
    trace_cause_detail: str | None
    launches: tuple[Launch, ...]
    cache_digests: tuple[CacheDigest, ...]
    shuffle_digests: tuple[ShuffleDigest, ...]
    steps: tuple[SegmentStep, ...]


@dataclass(frozen=True)
class TraceState:
    """What the store holds for one registered step, without its rows.

    ``traced`` False: no trace row, the step is untraced. ``uncovered_samples`` is
    the number of the trace's samples whose coverage mark is below the sample's
    size; a step with any is untraced too (plan 2.2(e)). ``unbacked_samples`` is the
    number of samples whose mark claims more tokens than the sample's stored rows
    add up to (rows that are gone, or a mark written beyond them): the rows no
    longer show what the step read there, so such a step is untraced as well. None
    of the three is a fault."""

    step: int
    batch_size: int
    last_sample_tokens_read: int | None
    traced: bool
    consumed_samples: int | None
    sample_rows: int
    uncovered_samples: int
    unbacked_samples: int = 0


@dataclass(frozen=True)
class ConsumedDocument:
    """A shard document some traced step of a segment read, with the first such step."""

    shard_position: int
    ordinal: int
    first_step: int


@dataclass(frozen=True)
class TraceHit:
    """One trace row that reads a given document."""

    step: int
    sample_pos: int
    row_pos: int
    token_start: int
    token_end: int
    sample_tokens: int
    covered_tokens: int


@dataclass(frozen=True)
class Checkpoint:
    instance_id: str
    names: tuple[str, ...]
    segment: str  # by value: the segment may have no row of its own
    step: int


@dataclass(frozen=True)
class HistorySegment:
    """The part of one segment that lies in a checkpoint's history.

    ``steps`` are the registered steps of the segment up to ``upto_step``, in
    order. ``upto_registered`` is False when ``upto_step`` itself is not among the
    segment's registered steps, so a reader can tell a history that ends on a
    registered step from one that names a step the step record does not hold."""

    segment: str
    checkpoint: str  # instance id of the checkpoint through which the history enters
    upto_step: int
    steps: tuple[int, ...]
    upto_registered: bool


@dataclass(frozen=True)
class History:
    """A checkpoint's history through its ancestors, oldest segment first.

    ``complete`` is True only when the walk reached a segment that started from
    scratch. Otherwise ``missing`` names the record the walk stopped at: a segment
    with no row, or a parent checkpoint with no row. A history that is not complete
    is never a shorter history: the part before ``missing`` is unknown."""

    checkpoint: str
    segments: tuple[HistorySegment, ...]
    complete: bool
    missing: Reference | None

    def steps(self) -> tuple[tuple[str, int], ...]:
        """The ordered list of (segment, step), oldest first."""
        return tuple((s.segment, step) for s in self.segments for step in s.steps)
