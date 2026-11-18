"""The hand-written table as the acceptance tests read it (plan 5.5, W5).

Three things, none of which touches the tool:

- ``load_check`` reads one file of ``fixture/table/`` into rows;
- ``interpret`` turns a row's ``given`` into a ``Question``: which store of the
  table's README, which state or fault, which request, which artifact, how the
  question is put, and what must exist before it can be put (``needs``);
- ``compare_outcome`` and its helpers compare what the command line printed with
  exactly the fields the row's ``expect`` holds, each field the way the README's
  section 6 says.

A row this module cannot interpret raises ``TableError``. Nothing is skipped here:
whether a need is met is decided by the caller, from what the replica builds.

Everything is written from fixture/table/README.md and docs/interfaces.md sections
6, 7 and 9. The words of the answers are the table's own; nothing is imported from
the tool.
"""

from __future__ import annotations

import json
import os
import re
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO / "fixture" / "table"

# A scratch copy of the table can be graded in place of the committed one: the
# variable names the folder that holds the ``check<NN>.toml`` files. It exists for
# the red run (README.md beside this file) and for nothing else.
TABLE_DIR_VARIABLE = "WATERWHITE_ACCEPTANCE_TABLE"


class TableError(Exception):
    """A row, or a value in one, that this module cannot interpret."""


class RowMismatch(AssertionError):
    """The tool's outcome differs from the row in the fields listed."""

    def __init__(self, row_id: str, mismatches: Sequence["Mismatch"]) -> None:
        self.row_id = row_id
        self.mismatches = tuple(mismatches)
        lines = [f"row {row_id}: {len(self.mismatches)} field(s) differ"]
        lines += [f"  {m}" for m in self.mismatches]
        super().__init__("\n".join(lines))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(m.field for m in self.mismatches)


# --- Rows --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One ``[[row]]`` of a table file (README section 1)."""

    id: str
    check: int
    question: str
    plan: str
    core: bool
    extension: str | None
    given: Mapping
    expect: Mapping
    expect_under: Mapping
    source: str
    rule: str | None


ROW_KEYS = {"id", "question", "plan", "core", "extension", "given", "expect", "expect_under",
            "source", "rule"}
ROW_REQUIRED = ("id", "question", "plan", "core", "given", "expect", "source")


def table_dir() -> Path:
    """The folder the table is read from: the committed one, or the scratch copy
    the variable ``WATERWHITE_ACCEPTANCE_TABLE`` names."""
    named = os.environ.get(TABLE_DIR_VARIABLE)
    return Path(named) if named else TABLE_DIR


def load_check(number: int, folder: Path | None = None) -> tuple[Row, ...]:
    """The rows of ``check<NN>.toml``, in file order."""
    path = (folder if folder is not None else table_dir()) / f"check{number:02d}.toml"
    try:
        with open(path, "rb") as f:
            document = tomllib.load(f)
    except FileNotFoundError:
        raise TableError(f"the table file {path} does not exist") from None
    if set(document) != {"row"}:
        raise TableError(f"{path.name}: expected only [[row]] entries, found {sorted(document)}")
    rows = []
    seen: set[str] = set()
    for position, raw in enumerate(document["row"], 1):
        where = f"{path.name}, row {position} ({raw.get('id', 'no id')})"
        unknown = set(raw) - ROW_KEYS
        missing = [key for key in ROW_REQUIRED if key not in raw]
        if unknown or missing:
            raise TableError(f"{where}: unknown keys {sorted(unknown)}, missing keys {missing}")
        if raw["id"] in seen:
            raise TableError(f"{where}: the id is used twice")
        seen.add(raw["id"])
        if not raw["id"].startswith(f"c{number:02d}."):
            raise TableError(f"{where}: the id does not start with c{number:02d}.")
        if raw["core"] is False and not raw.get("extension"):
            raise TableError(f"{where}: an extension row names its extension")
        if raw["core"] is True and raw.get("extension"):
            raise TableError(f"{where}: a core row names no extension")
        rows.append(Row(
            id=raw["id"], check=number, question=raw["question"], plan=raw["plan"],
            core=raw["core"], extension=raw.get("extension"), given=raw["given"],
            expect=raw["expect"], expect_under=raw.get("expect_under", {}),
            source=raw["source"], rule=raw.get("rule")))
    return tuple(rows)


# --- Requests (README section 3) ---------------------------------------------------------


@dataclass(frozen=True)
class Request:
    """A request of the table. ``records`` are the handles a withdrawal is narrowed
    to. ``banned`` is the handle whose text a ban names. ``extension`` is the
    extension the request belongs to; a core request has none."""

    name: str
    kind: str
    source: str | None = None
    records: tuple[str, ...] = ()
    banned: str | None = None
    extension: str | None = None


REQUESTS: dict[str, Request] = {r.name: r for r in (
    Request("withdraw A", "withdraw", source="A"),
    Request("withdraw C", "withdraw", source="C"),
    Request("withdraw A, record X@A", "withdraw", source="A", records=("X@A",)),
    Request("withdraw B, record X@B", "withdraw", source="B", records=("X@B",)),
    # The source of W (B or Z) is the spec's pending item positions.shared_token.
    Request("withdraw W", "withdraw", source=None, records=("W",)),
    Request("ban X", "ban", banned="X@A", extension="content-ban"),
)}


# --- Handles (README section 5) ----------------------------------------------------------

# Where the record of a bare handle lies.
HANDLE_HOME = {
    "X@A": "src/A", "X@B": "src/B", "SA": "src/A", "SC": "src/C",
    "P": "gen/P", "T": "gen/T", "S": "gen/S",
}
JOBS = {"paraphrase": "P", "translation": "T", "summary": "S"}  # job -> the record it makes

_UNRELATED = re.compile(r"U-(\d+)")
_RECORD_OF_Z = re.compile(r"Z-\d+")  # the record id of an unrelated record, as the spec writes W


@dataclass(frozen=True)
class RecordRef:
    """A record named by the table: the artifact that holds it and its handle."""

    artifact: str
    handle: str

    def __str__(self) -> str:
        return f"{self.artifact}:{self.handle}"


def record_ref(text: str, artifact: str | None = None) -> RecordRef:
    """Read ``artifact:handle``, or a bare handle. A bare handle lies in
    ``artifact`` when one is given and in its home file otherwise."""
    if ":" in text:
        named, handle = text.split(":", 1)
        return RecordRef(named, handle)
    if artifact is not None:
        return RecordRef(artifact, text)
    if text in HANDLE_HOME:
        return RecordRef(HANDLE_HOME[text], text)
    if _UNRELATED.fullmatch(text) or _RECORD_OF_Z.fullmatch(text):
        return RecordRef("src/Z", text)
    raise TableError(f"the handle {text!r} has no home file in README section 5")


def manifest_handles(handle: str) -> tuple[str, ...]:
    """The handles a replica manifest may carry for a handle of the table.

    ``X`` is the row of a release that holds the passage X: the manifest names the
    copy that was kept, ``X@A`` or ``X@B``. ``U-<nnn>`` is the unrelated record
    ``Z-<nnn>``. Every other handle is spelled alike in both."""
    if handle == "X":
        return ("X@A", "X@B")
    unrelated = _UNRELATED.fullmatch(handle)
    if unrelated:
        return (f"Z-{unrelated.group(1)}",)
    return (handle,)


# ``resolve(RecordRef)`` gives the record's locator in its artifact. The caller
# builds it over the replica's manifests.
Resolver = Callable[[RecordRef], str]


# --- Needs: what must exist before a row can be put -------------------------------------------

NEED_SHARD = "shard"
NEED_MIXTURE = "mixture"
NEED_CHECKPOINT = "checkpoint"
NEED_TRACE = "step trace"
NEED_MAP = "tokenization map"
NEED_REPORT = "rebuild report"
NEED_SUCCESSOR = "successor"
NEED_VERIFY = "verification of a repair"
NEED_SCRATCH = "scratch store of Check 4"
NEED_SHARED_TOKEN = "shared-token document W"
NEED_SHOW_EDGE_DIGEST = "edge digest in the output of show"
# What the run world needs (acceptance_world.World): the replica's run wrapper
# (``python -m replica build-runs``) and ``waterwhite register-run``. The three needs
# above that a registered run gives (a step trace, a checkpoint, the record W that a
# run reads) are all met by the one thing: ``World.unmet`` asks for it by this name.
NEED_RUN_CAPTURE = "run capture"
# The two forms of the runs that Checks 1 and 2 read: the rows of a segment's step traces
# (``show SEGMENT --trace``), which Check 1 compares with the oracle's log, and the arrays
# a replay rebuilt (``replay``), which Check 2 (i) compares with the withheld ones. A row
# that needs one is skipped with the form's name while the probe finds the form absent.
NEED_SHOW_TRACE = "show --trace"
NEED_REPLAYED_ARRAYS = "waterwhite replay"
# Three things a row of Check 3 or 6 needs printed and that ``show`` does not print
# (README.md beside this file, "What show must print for Checks 3 and 6").
NEED_SHOW_MAP_ROWS = "rows of a named map in the output of show"
NEED_SHOW_DROPS = "drops by output artifact in the output of show"
NEED_SHOW_MAP_SETTINGS = "append-eod setting of a map in the output of show"

# Stores (README section 4): what each needs beyond the JSONL artifacts. Whether
# the replica builds a store at all is not written here: the replica says so
# itself when it is asked for the store (``acceptance_world.World.store``).
STORES: dict[str, tuple[str, ...]] = {
    "main": (),
    "swap": (),
    "import": (),
    "rr": (NEED_SHARD,),
    "no-capture": (),
    "scratch-500": (NEED_SCRATCH,),
    "scratch-notokens": (),
}

# The stores whose shards Megatron's preprocessing writes: building one needs the
# test-time Megatron environment.
TOKENIZED_STORES = frozenset({"rr", "no-capture", "scratch-500", "scratch-notokens"})
# The store "main" holds shards too, but most of its rows ask about record files
# only. It is tokenized where the Megatron environment is set, so that its shard
# rows run, and built without tokenization elsewhere, so that its other rows run
# without Megatron (acceptance_world.World).
TOKENIZED_WHERE_POSSIBLE = frozenset({"main"})


@dataclass(frozen=True)
class State:
    """A state of README section 7: the record its fault is put on must exist
    (``needs``), and the catalogue fixes the request it is asked with (``request``;
    None where the catalogue names none)."""

    needs: tuple[str, ...] = ()
    request: str | None = None


# The states that Checks 3, 5, 6 and 7 use. A state of another check is not listed
# and is refused until the tests of that check add it.
STATES: dict[str, State] = {
    "unmutated": State(),
    "f.trace-deleted": State((NEED_TRACE,), "withdraw A"),
    "f.trace-half": State((NEED_TRACE,), "withdraw A"),
    "f.edge-deleted": State((), "withdraw A"),
    "f.map-deleted": State((NEED_MAP,), "withdraw A"),
    "f.edge-file-level": State((), "withdraw A, record X@A"),
    "f.digest": State((NEED_MAP,), "withdraw A"),
    "f.label": State((NEED_MAP,), "withdraw A"),
    "g.bound-field": State((), "withdraw C"),
    "g.log-outside-read": State((), "withdraw C"),
    "g.log-removed": State((), "withdraw C"),
    "g.model-entry-removed": State((), "withdraw C"),
    "s.in-scope-only": State((), "withdraw C"),
    "s.file-level-link-B": State((), "withdraw B, record X@B"),
    "h.outside-read": State(),
    "q.unregistered": State(),
    # Check 4 (README section 7): the faults of the store's own checks. The ones that
    # write a map need the scratch store; a fault on a trace needs the runs.
    "l.edge-parent": State((), "withdraw A"),
    "l.artifact-digest": State((), "withdraw A"),
    "l.trace-count": State((NEED_TRACE,), "withdraw A"),
    "l.count-499": State((NEED_SCRATCH,), "withdraw A"),
    "w.kill-after-100": State((NEED_SCRATCH,)),
    "w.retry": State((NEED_SCRATCH,)),
    "w.declare-499": State((NEED_SCRATCH,)),
    # Check 2 (README section 7): the run is registered in a fresh store from a
    # starting state. A row may ask the registration or the answers after it.
    "c2.i": State((NEED_RUN_CAPTURE,)),
    "c2.ii": State((NEED_RUN_CAPTURE,)),
    "c2.iii": State((NEED_RUN_CAPTURE,)),
    "c2.iv": State((NEED_RUN_CAPTURE,)),
    "r.successor-registered": State((NEED_SUCCESSOR,), "withdraw A"),
    # Check 3: set-ups on a tokenizer input (README section 7).
    "t.no-tokens-line": State(),
    "t.twin-swap": State(),
    "t.verification-fails": State(),
}

LISTING = "(listing)"

ARTIFACT_JSONL = "jsonl"
ARTIFACT_SHARD = "shard"
ARTIFACT_MIXTURE = "mixture"
ARTIFACT_CHECKPOINT = "checkpoint"
ARTIFACT_SUCCESSOR = "successor"
ARTIFACT_R3 = "release R3"

_ARTIFACT_FORMS = (
    (re.compile(r"src/[A-Z]|gen/[A-Z]|R1|R2|R1-swap|import/T"), ARTIFACT_JSONL),
    (re.compile(r"R3"), ARTIFACT_R3),
    (re.compile(r"E(-rr)?/shard-\d+"), ARTIFACT_SHARD),
    # The scratch shards of Checks 3 and 4 (README section 4).
    (re.compile(r"c03/shard-notokens|c04/shard-500"), ARTIFACT_SHARD),
    (re.compile(r"[A-Z](-rr)?/mix-\d+"), ARTIFACT_MIXTURE),
    (re.compile(r"[A-Z](-rr)?@\d+"), ARTIFACT_CHECKPOINT),
    (re.compile(r"successor of .+"), ARTIFACT_SUCCESSOR),
)
_ARTIFACT_NEEDS = {
    ARTIFACT_JSONL: (),
    ARTIFACT_R3: (),  # exists only with the generating-model extension on
    ARTIFACT_SHARD: (NEED_SHARD,),
    ARTIFACT_MIXTURE: (NEED_MIXTURE,),
    ARTIFACT_CHECKPOINT: (NEED_CHECKPOINT,),
    ARTIFACT_SUCCESSOR: (NEED_SUCCESSOR,),
}


def artifact_kind(name: str) -> str:
    """The kind of an artifact, read from the form of its name (docs/interfaces.md
    section 9.2)."""
    for form, kind in _ARTIFACT_FORMS:
        if form.fullmatch(name):
            return kind
    raise TableError(f"{name!r} is not an artifact name of docs/interfaces.md section 9.2")


def artifact_needs(name: str) -> tuple[str, ...]:
    """What must be built before an artifact of this name can be asked about."""
    return _ARTIFACT_NEEDS[artifact_kind(name)]


# --- Questions -----------------------------------------------------------------------------

KIND_ANSWER = "answer"              # query <request> --artifact NAME
KIND_LISTING = "listing"            # query <request>, no artifact
KIND_ATTRIBUTION = "attribution"    # show ARTIFACT --record LOCATOR
KIND_INSTANCES = "instances"        # show of two artifacts, compared
KIND_REREGISTER = "reregister"      # the same build registered again
KIND_HARNESS = "harness"            # a job run in the harness
KIND_RECORD = "record under a request"   # a rebuild report row, a successor, a history
KIND_REPORT = "report"              # the counts of a rebuild report
KIND_EXTENSION = "extension row"    # put only while its extension is on
KIND_SCRATCH_WRITE = "write to the scratch store"   # Check 4: a map written, stopped, retried
KIND_JOIN_ROWS = "rows of the exposure join"        # Check 1, 8: the join against the oracle's log
KIND_RUN_REGISTRATION = "registration of run segments"   # Check 2: what register-run prints
KIND_K_DIGESTS = "shuffle digests of run K"         # Check 8: the join's shuffle and the record

# ``never`` is not a field of an answer: it lists answers the row must not get
# (README section 6).
ANSWER_KEYS = frozenset({
    "answer", "reason", "names", "names_detail", "failing", "candidates", "chain", "basis",
    "notes", "untraced_steps", "first_exposure", "scope", "never"})
STOP_KEYS = frozenset({"error", "exit_code", "stdout_empty"})
# ``loud_stop = false`` (Check 4): the question is answered, it is not a loud stop.
LOUD_STOP = "loud_stop"
SCRATCH_WRITE_KEYS = frozenset({"one_of", "never", "error", "map_after", "rows_after"})
REGISTRATION_KEYS = frozenset({"replay", "reason", "digests", "replayed_arrays", "kinds"})
REPORT_KEYS = frozenset({"action", "new_parent", "withdrawn_parent", "regenerate_with", "count",
                         "counts", "missing_link", "review_scope"})
LISTING_KEYS = frozenset({"covers", "not_covered"})
ATTRIBUTION_KEYS = frozenset({"selected_parent", "parent_mode", "loser"})
INSTANCES_KEYS = frozenset({"content_digests", "instance_ids", "occurrence_keys_of_X",
                            "instance_id", "producing_step", "input_instance_ids",
                            "edge_digest"})
REREGISTER_KEYS = frozenset({"instance_id", "new_artifacts", "new_occurrences"})
HARNESS_KEYS = frozenset({"read_refused", "text_returned", "log_shows_attempt", "log_entry"})

GIVEN_KEYS = frozenset({"store", "state", "request", "artifact", "artifacts", "record",
                        "records", "extensions_on", "action", "job", "handed", "asks_for",
                        "set", "layout", "map", "maps",
                        # Checks 1, 2 and 8: the part of a run a question is about.
                        "run", "segment", "segments", "step", "launch", "sample_pos"})

# The one set-up a row of Check 7 gives in plain words.
REREGISTER_ACTION = "register R1 again with the same file, step, inputs and edges"


@dataclass(frozen=True)
class Question:
    """A row's ``given``, interpreted.

    ``needs`` lists, in order, everything that must exist before the row can be
    put: the record a fault is put on, the store, the artifact, what the expected
    fields are read from. ``expect`` is the expected outcome that holds (the
    fallback's, when one is in force). ``deferred`` maps an expected field that
    is read from something other than the row's own question (the rebuild report,
    a field ``show`` would have to print) to that need: while the need is not
    built the row's other fields are compared without it (``without``)."""

    row: Row
    kind: str
    store: str
    state: str
    request: Request | None
    artifacts: tuple[str, ...]
    record: str | None
    expect: Mapping
    needs: tuple[str, ...] = ()
    deferred: Mapping[str, str] = field(default_factory=dict)
    # For a row of Check 3 or 6: the maps it names, each as (namespace,
    # derivation), and whether it is asked of a generated set (once per size).
    maps: tuple[tuple[str, str], ...] = ()
    sized: bool = False

    @property
    def artifact(self) -> str:
        if len(self.artifacts) != 1:
            raise TableError(f"row {self.row.id}: one artifact was expected, it names "
                             f"{list(self.artifacts)}")
        return self.artifacts[0]

    def waiting(self, not_built: Mapping[str, str]) -> dict[str, str]:
        """The deferred fields whose need is among ``not_built``."""
        return {name: need for name, need in self.deferred.items() if need in not_built}

    def without(self, names: Mapping[str, str]) -> "Question":
        """The same question with the fields ``names`` taken out of ``expect``."""
        kept = {k: v for k, v in self.expect.items() if k not in names}
        return replace(self, expect=kept)


def expect_in_force(row: Row, fallbacks: Sequence[str] = ()) -> Mapping:
    """The row's expected outcome: ``expect``, or the entry of ``expect_under``
    for the fallback in force (README section 8)."""
    named = [name for name in fallbacks if name in row.expect_under]
    if len(named) > 1:
        raise TableError(f"row {row.id}: the fallbacks {named} are in force together and the "
                         "row gives an expectation for each")
    return row.expect_under[named[0]] if named else row.expect


def interpret(row: Row, fallbacks: Sequence[str] = ()) -> Question:
    """Read a row's ``given``. Raises ``TableError`` for anything it cannot read:
    an unknown key, store, state, request or artifact name, a state asked with
    another request than its catalogue entry fixes, an expected field the way of
    putting the question does not produce."""
    given = row.given
    where = f"row {row.id}"
    unknown = set(given) - GIVEN_KEYS
    if unknown:
        raise TableError(f"{where}: given holds keys this reader does not know: {sorted(unknown)}")
    store = given.get("store", "main")
    if store not in STORES:
        raise TableError(f"{where}: unknown store {store!r}")
    state = given.get("state", "unmutated")
    if state not in STATES:
        raise TableError(f"{where}: the state {state!r} is not one Checks 3, 5, 6 and 7 use")
    request = None
    if "request" in given:
        if given["request"] not in REQUESTS:
            raise TableError(f"{where}: unknown request {given['request']!r}")
        request = REQUESTS[given["request"]]
    fixed = STATES[state].request
    if fixed is not None and (request is None or request.name != fixed):
        raise TableError(f"{where}: the state {state} is asked with {fixed!r} "
                         f"(README section 7), the row asks {given.get('request')!r}")
    if "artifact" in given and "artifacts" in given:
        raise TableError(f"{where}: given holds both artifact and artifacts")
    artifacts = tuple(given["artifacts"]) if "artifacts" in given else (
        (given["artifact"],) if "artifact" in given else ())
    kinds = [None if name == LISTING else artifact_kind(name) for name in artifacts]
    record = given.get("record")
    if record is not None and not isinstance(record, str):
        raise TableError(f"{where}: record is one handle")
    expect = expect_in_force(row, fallbacks)

    needs: list[str] = list(STATES[state].needs) + list(STORES[store])
    if state != "q.unregistered":  # that state asks about a name nothing holds, on purpose
        for kind in kinds:
            needs.extend(_ARTIFACT_NEEDS.get(kind, ()))
    if request is not None and request.kind == "withdraw" and request.source is None:
        needs.append(NEED_SHARED_TOKEN)

    def question(kind: str, allowed: frozenset[str] | None, *, extra_needs: Sequence[str] = (),
                 deferred: Mapping[str, str] | None = None,
                 maps: tuple[tuple[str, str], ...] = (), sized: bool = False) -> Question:
        deferred = dict(deferred or {})
        if allowed is not None:
            strange = set(expect) - allowed - set(deferred)
            if strange:
                raise TableError(f"{where}: a question put as {kind!r} gives no field "
                                 f"{sorted(strange)}")
        ordered = list(dict.fromkeys([*needs, *extra_needs]))
        return Question(row=row, kind=kind, store=store, state=state, request=request,
                        artifacts=artifacts, record=record, expect=dict(expect),
                        needs=tuple(ordered), deferred=deferred, maps=maps, sized=sized)

    if not row.core:
        return question(KIND_EXTENSION, None)
    if "extensions_on" in given:
        raise TableError(f"{where}: a core row switches no extension on")

    if dict(expect) == NOT_UNDER_FALLBACK:
        return question(KIND_DOES_NOT_APPLY, None)
    form = form_of(row, expect)
    if form is not None:
        return question(form.kind, None, extra_needs=form.needs, deferred=form.deferred,
                        maps=maps_named(given, where), sized=form.sized)
    if (given.keys() & RUN_PART_KEYS) or (state.startswith(("c2.", "w.")) and request is None):
        return _run_question(row, given, state, request, artifacts, record, expect, question)
    if set(given) & FORM_ONLY_KEYS:
        raise TableError(f"{where}: no row form of Checks 3 and 6 has the given keys "
                         f"{sorted(given)} with the expected fields {sorted(expect)}")

    if state == "h.outside-read":
        if set(given) != {"state", "job", "handed", "asks_for"} or given["job"] not in JOBS:
            raise TableError(f"{where}: a harness row gives state, job, handed and asks_for")
        return question(KIND_HARNESS, HARNESS_KEYS)
    if set(given) & {"job", "handed", "asks_for"}:
        raise TableError(f"{where}: job, handed and asks_for belong to the state h.outside-read")

    if "action" in given:
        if given["action"] != REREGISTER_ACTION or request is not None or artifacts != ("R1",):
            raise TableError(f"{where}: the only set-up in words this reader knows is "
                             f"{REREGISTER_ACTION!r}, for R1 and with no request")
        return question(KIND_REREGISTER, REREGISTER_KEYS)

    if request is None:
        if len(artifacts) == 2 and "records" not in given:
            deferred = ({"edge_digest": NEED_SHOW_EDGE_DIGEST} if "edge_digest" in expect else {})
            return question(KIND_INSTANCES, INSTANCES_KEYS, deferred=deferred)
        if len(artifacts) == 1 and record is not None and "records" not in given:
            return question(KIND_ATTRIBUTION, ATTRIBUTION_KEYS)
        raise TableError(f"{where}: with no request, a row names two artifacts to compare or "
                         "one artifact and one record")

    if len(artifacts) != 1:
        raise TableError(f"{where}: a request is put to one artifact or to {LISTING!r}")
    report_fields = set(expect) & REPORT_KEYS
    if request.name == "withdraw W":
        # README section 3: the request is narrowed to the record W, which the row
        # names again by the spec's pending item.
        if record is None or not report_fields <= set():
            raise TableError(f"{where}: the request withdraw W names the record W")
        return question(KIND_ANSWER, ANSWER_KEYS | STOP_KEYS | {LOUD_STOP})
    if record is not None or "records" in given:
        if kinds[0] == ARTIFACT_SUCCESSOR or NEED_SUCCESSOR in needs:
            return question(KIND_RECORD, None)
        if report_fields:
            return question(KIND_RECORD, None, extra_needs=(NEED_REPORT,))
        raise TableError(f"{where}: a request about one record of a JSONL artifact is asked "
                         "of a rebuild report or of a successor; this row is neither")
    if artifacts[0] == LISTING:
        # What a listing covers names runs and checkpoints: it needs the runs.
        return question(KIND_LISTING, STOP_KEYS | LISTING_KEYS,
                        extra_needs=(NEED_RUN_CAPTURE,) if "covers" in expect else ())
    if report_fields and report_fields == set(expect):
        return question(KIND_REPORT, None, extra_needs=(NEED_REPORT,))
    return question(KIND_ANSWER, ANSWER_KEYS | STOP_KEYS | {LOUD_STOP},
                    deferred={name: NEED_REPORT for name in sorted(report_fields)})


# The fields of a Check 1 row that are read from an output of their own, in the world's
# ``_join_rows``: the launch's counter and steps (``launches`` of ``show --trace``), the epoch
# of each step (``epochs`` of its steps), the skipped line of a shard (the ``drops`` of
# ``show`` of the shard) and the document count of a shard (its stored rows). A form the tool
# does not print yet skips the row by name after everything else the row names was compared;
# no field is left out of a row for good.

# The keys of ``given`` that name a part of a run (Checks 1, 2 and 8).
RUN_PART_KEYS = frozenset({"run", "segment", "segments", "step", "launch", "sample_pos"})


def _run_question(row: Row, given: Mapping, state: str, request: Request | None,
                  artifacts: tuple[str, ...], record: str | None, expect: Mapping,
                  question) -> Question:
    """The rows that ask about a part of a run, a registration or a write to the
    scratch store (Checks 1, 2, 4 and 8)."""
    where = f"row {row.id}"
    if state.startswith("w.") and request is None:
        if set(given) - {"store", "artifact", "state"}:
            raise TableError(f"{where}: a write to the scratch store names a store, an "
                             "artifact and a state")
        return question(KIND_SCRATCH_WRITE, SCRATCH_WRITE_KEYS)
    if state.startswith("c2.") and request is None:
        if set(given) - {"state", "segment", "segments"}:
            raise TableError(f"{where}: a registration names a state and its segments")
        replayed = {name: NEED_REPLAYED_ARRAYS for name in ("digests",) if name in expect}
        needs = (NEED_REPLAYED_ARRAYS,) if "replayed_arrays" in expect else ()
        return question(KIND_RUN_REGISTRATION, REGISTRATION_KEYS, extra_needs=needs,
                        deferred=replayed)
    if "shuffle_digests" in expect:
        if set(given) != {"run", "segment", "state"}:
            raise TableError(f"{where}: the shuffle of a run names a run, a segment and a state")
        # c08.K.digests-match. Question 8 of docs/join.md section 3 is the tool's own
        # comparison of its shuffle with the recorded digest, and this row reads the
        # ``shuffle`` that register-run prints (a row of its own is skipped by name where the
        # command does not print it). What grades the ORDER, not the digest, is Check 1's rows on
        # run K (c01.K.diff, c01.K.warm-start, c01.K.resume, c01.K.epoch-boundary,
        # c01.K.ranks): the join's shuffle and the replica's were written from one text of
        # the interface (docs/interfaces.md 8.5), so they could share a wrong rule, and
        # only the oracle's log, which Megatron's own sampler wrote, tells them apart.
        return question(KIND_K_DIGESTS, None, extra_needs=(NEED_RUN_CAPTURE,))
    if "sample_pos" in given or ("run" in given and request is None):
        # Check 1, and the one row of Check 8 that reads the rows of a step.
        return question(KIND_JOIN_ROWS, None, extra_needs=(NEED_RUN_CAPTURE, NEED_SHOW_TRACE))
    raise TableError(f"{where}: no question of Checks 1, 2, 4 or 8 has the given keys "
                     f"{sorted(given)} with the expected fields {sorted(expect)}")


# --- The row forms of Checks 3 and 6 --------------------------------------------------------
#
# Most rows of these two checks ask a question in plain words: the README's
# section 6 calls their expected fields "plain-word keys, values named in the
# row's question". A sentence cannot be compared with what the tool prints, so the
# reader knows each sentence the two files use and which comparison answers it.
# A form is told by the keys of its ``given``, the keys of its ``expect`` and its
# sentences. A row whose sentence was changed matches no form and is an error of
# the test: the comparison written for the old sentence does not answer the new one.

KIND_SPEC_ITEM = "spec item"                          # read from the fixture spec
KIND_PAIR_KINDS = "pair kinds"                        # counted from the plant list
KIND_SPANS = "spans per ordinal"
KIND_SPAN_TOTALS = "span totals"
KIND_SET_DROPS = "sanitizer drops of a set"
KIND_SET_NO_TOKENS = "no-tokens drops of a set"
KIND_PLANTED_LINE = "planted line without tokens"
KIND_RECORDS = "record per ordinal"
KIND_RECORDS_AS_CAPTURE = "record per ordinal as the capture map"
KIND_TWINS = "same-shard pairs"
KIND_TWINS_SWAPPED = "same-shard pairs, lines swapped"
KIND_SPLIT_PAIRS = "split pairs"
KIND_VERIFICATION_FAILED = "verification failed"
KIND_MAP_FIELDS = "map fields"
KIND_SET_DOCUMENTS = "documents of a set mapped"
KIND_SET_DROPS_RECOVERED = "drops of a set made without capture"
KIND_SET_CANDIDATES = "candidates of a set"
KIND_SET_DOWNSTREAM = "downstream of a translation in a set"
KIND_DOCUMENTS = "documents of a shard mapped"
KIND_DROPS = "drops of a store"
KIND_EDGE_LABEL = "label of a row's link"
KIND_FILE_FAMILY = "file family"
KIND_DOES_NOT_APPLY = "does not apply under the fallback in force"

# ``expect_under`` of a row that a fallback takes out of the check.
NOT_UNDER_FALLBACK = {"applies": False}

# The given keys only a form of Checks 3 and 6 uses.
FORM_ONLY_KEYS = frozenset({"set", "layout", "map", "maps"})

# The maps a row names (README section 2), as (namespace, derivation).
MAP_NAMES: dict[str, tuple[str, str]] = {
    "capture": ("capture", "position"),
    "position recovery": ("recovery", "position"),
    "token-matching recovery": ("recovery", "token-matching"),
}

# The generated sets a row is asked of (README section 2): the sizes the plan
# names. The tests add a small set of their own (README.md beside this file).
SET_SIZES: dict[str, tuple[str, ...]] = {"100k": ("100k",), "generated": ("10k", "100k")}

VALUE = object()  # an expected field that is compared with what was printed or counted

S_SPAN = "token_end - token_start equals the document's size in the shard's index"
S_SPAN_TOTAL = "sum of spans equals the index total"
S_DROP_PER_LOGGED_LINE = ("one drop per skipped or empty line of the sanitizer's log, with its "
                          "reason")
S_SURVIVING_LINES = "the number of surviving lines of the shard's input"
S_PLANTED_RECORD = "the planted record whose tokens equal the shard's tokens at that ordinal"
S_AS_CAPTURE = "the same record as the capture map, for every document outside a same-shard pair"
S_TWO_LINES = "exactly the two chunk-file lines of the pair"
S_UNCHANGED = "unchanged"
S_OWN_SHARD_LINE = "the line of the pair's member in the same shard's chunk file"
S_ONE_OR_SET = ("one recovered occurrence, or the candidate set of an identical pair in the "
                "same shard")
S_DROP_PER_PLANTED_LINE = ("one per planted skipped line and empty line of the sanitizer's log, "
                           "each with its reason")
S_PAIR_SOURCES = "the sources of the pair's members, as the plant list names them"


@dataclass(frozen=True)
class Form:
    """One row form: the keys its ``given`` holds, its expected fields (each a
    sentence the field must be, or ``VALUE``), what must be printed before it can
    be put (``needs``), the fields that wait for something while the others are
    compared (``deferred``), and whether it is asked of a generated set."""

    kind: str
    given: frozenset[str]
    expect: Mapping[str, object]
    needs: tuple[str, ...] = ()
    deferred: Mapping[str, str] = field(default_factory=dict)
    sized: bool = False


def _form(kind: str, given: str, expect: Mapping[str, object], *, needs: tuple[str, ...] = (),
          deferred: Mapping[str, str] | None = None, sized: bool = False) -> Form:
    return Form(kind, frozenset(given.split()), expect, needs, dict(deferred or {}), sized)


_ROWS = (NEED_SHOW_MAP_ROWS,)

FORMS: tuple[Form, ...] = (
    # Check 3
    _form(KIND_SPEC_ITEM, "set layout", {"spec_key": VALUE, "value": VALUE}),
    _form(KIND_PAIR_KINDS, "set layout",
          {"same_shard_pairs_at_least": VALUE, "split_pairs_at_least": VALUE}, sized=True),
    _form(KIND_SPANS, "set layout map", {"per_ordinal": S_SPAN, "mismatches": VALUE},
          needs=_ROWS, sized=True),
    _form(KIND_SPAN_TOTALS, "set layout map", {"per_shard": S_SPAN_TOTAL, "mismatches": VALUE},
          needs=_ROWS, sized=True),
    _form(KIND_SET_DROPS, "set layout map",
          {"drops": S_DROP_PER_LOGGED_LINE, "reasons": VALUE,
           "documents_per_shard": S_SURVIVING_LINES},
          needs=(NEED_SHOW_DROPS,), sized=True),
    _form(KIND_SET_NO_TOKENS, "set layout map",
          {"drops_with_reason_no_tokens": VALUE, "append_eod_recorded": VALUE,
           "append_eod": VALUE},
          needs=(NEED_SHOW_DROPS, NEED_SHOW_MAP_SETTINGS), sized=True),
    _form(KIND_PLANTED_LINE, "store artifact state maps",
          {"input_lines": VALUE, "documents": VALUE, "drops": VALUE, "ordinal_of_line": VALUE,
           "records": VALUE, "label": VALUE, "verification": VALUE,
           "append_eod_recorded": VALUE},
          needs=(NEED_SHOW_MAP_ROWS, NEED_SHOW_DROPS, NEED_SHOW_MAP_SETTINGS)),
    _form(KIND_RECORDS, "set layout map", {"per_ordinal": S_PLANTED_RECORD, "mismatches": VALUE},
          needs=_ROWS, sized=True),
    _form(KIND_RECORDS_AS_CAPTURE, "set layout map",
          {"per_ordinal": S_AS_CAPTURE, "mismatches": VALUE}, needs=_ROWS, sized=True),
    _form(KIND_TWINS, "set layout maps",
          {"label": VALUE, "candidate_set": S_TWO_LINES, "other_labels_for_these_ordinals": VALUE},
          needs=_ROWS, sized=True),
    _form(KIND_TWINS_SWAPPED, "set layout map state",
          {"label": VALUE, "candidate_set": S_UNCHANGED}, needs=_ROWS, sized=True),
    _form(KIND_SPLIT_PAIRS, "set layout maps", {"label": VALUE, "record": S_OWN_SHARD_LINE},
          needs=_ROWS, sized=True),
    _form(KIND_VERIFICATION_FAILED, "set layout state maps",
          {"verification": VALUE, "label": VALUE, "labels_of_documents": VALUE},
          deferred={"labels_of_documents": NEED_SHOW_MAP_ROWS}, sized=True),
    _form(KIND_MAP_FIELDS, "set layout maps",
          {"capture": VALUE, "position_recovery": VALUE, "token_matching_recovery": VALUE,
           "maps_per_shard": VALUE, "labels_of_documents": VALUE},
          deferred={"labels_of_documents": NEED_SHOW_MAP_ROWS}, sized=True),
    # Check 6
    _form(KIND_SET_DOCUMENTS, "set store map",
          {"per_ordinal": S_ONE_OR_SET, "unmapped_documents": VALUE}, needs=_ROWS, sized=True),
    _form(KIND_SET_DROPS_RECOVERED, "set store map",
          {"drops": S_DROP_PER_PLANTED_LINE, "reasons": VALUE,
           "drops_with_reason_no_tokens": VALUE, "ordinals_shifted": VALUE},
          needs=(NEED_SHOW_DROPS, NEED_SHOW_MAP_ROWS), sized=True),
    _form(KIND_SET_CANDIDATES, "set store",
          {"label": VALUE, "candidate_sources": S_PAIR_SOURCES,
           "answer_for_each_candidate_source": VALUE}, sized=True),
    _form(KIND_SET_DOWNSTREAM, "set store request", {"answer": VALUE, "never": VALUE},
          sized=True),
    # The rows of one shard are few: they are read document by document where
    # ``show`` prints no rows of a named map (acceptance_world.World.map_rows).
    _form(KIND_DOCUMENTS, "store artifact map",
          {"documents": VALUE, "recovered": VALUE, "candidates": VALUE, "unknown": VALUE,
           "derivation": VALUE, "verification": VALUE}),
    _form(KIND_DROPS, "store",
          {"drops": VALUE, "empty_line_drops": VALUE, "no_tokens_drops": VALUE},
          needs=(NEED_SHOW_DROPS,)),
    _form(KIND_EDGE_LABEL, "store artifact record", {"label": VALUE, "candidates": VALUE}),
    _form(KIND_FILE_FAMILY, "store artifact", {"link": VALUE}),
)

FORM_KINDS = frozenset(form.kind for form in FORMS) | {KIND_DOES_NOT_APPLY}


def form_of(row: Row, expect: Mapping) -> Form | None:
    """The form of Checks 3 and 6 a row has, or None when it has none. The keys of
    ``given`` and of ``expect`` must be the form's, and each sentence the form's
    sentence."""
    given = frozenset(row.given)
    found = [form for form in FORMS
             if form.given == given and set(form.expect) == set(expect)
             and all(want is VALUE or expect[name] == want
                     for name, want in form.expect.items())]
    if len(found) > 1:
        raise TableError(f"row {row.id}: the forms {[form.kind for form in found]} all fit it")
    if found:
        return found[0]
    near = [form for form in FORMS
            if form.given == given and set(form.expect) == set(expect)]
    if near:
        raise TableError(
            f"row {row.id}: its question in words is none this reader knows; for these keys "
            f"it knows {[{k: v for k, v in f.expect.items() if v is not VALUE} for f in near]}")
    return None


def maps_named(given: Mapping, where: str) -> tuple[tuple[str, str], ...]:
    """The maps a row names with ``map`` or ``maps``, each as (namespace,
    derivation)."""
    if "map" in given and "maps" in given:
        raise TableError(f"{where}: given holds both map and maps")
    names = [given["map"]] if "map" in given else list(given.get("maps", ()))
    unknown = [name for name in names if name not in MAP_NAMES]
    if unknown:
        raise TableError(f"{where}: the README names no map {unknown}; it names "
                         f"{sorted(MAP_NAMES)}")
    return tuple(MAP_NAMES[name] for name in names)


def sizes_of(row: Row) -> tuple[str, ...]:
    """The generated sets the plan asks a row of: the sizes its ``set`` names."""
    name = row.given.get("set")
    if name not in SET_SIZES:
        raise TableError(f"row {row.id}: the README names no set {name!r}; it names "
                         f"{sorted(SET_SIZES)}")
    return SET_SIZES[name]


# --- Pending values (README section 5) -----------------------------------------------------

PENDING = "pending:"


def fill_pending(value: object, spec: Mapping) -> tuple[object, tuple[str, ...]]:
    """Replace every ``"pending:<key>"`` inside ``value`` by the spec item
    ``<key>`` (a dotted path into the fixture spec). Returns the filled value and
    the keys whose spec item is missing or empty; a row with such a key is not
    checked."""
    empty: list[str] = []

    def fill(item: object) -> object:
        if isinstance(item, str) and item.startswith(PENDING):
            key = item[len(PENDING):]
            found: object = spec
            for part in key.split("."):
                found = found.get(part) if isinstance(found, Mapping) else None
            if found is None or found == "" or found == [] or found == {} or not key:
                empty.append(key)
                return item
            return found
        if isinstance(item, Mapping):
            return {k: fill(v) for k, v in item.items()}
        if isinstance(item, list):
            return [fill(v) for v in item]
        return item

    return fill(value), tuple(empty)


# --- Comparison (README section 6) ---------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """One expected field the outcome does not match."""

    field: str
    expected: object
    got: object

    def __str__(self) -> str:
        return f"{self.field}: expected {self.expected!r}, got {self.got!r}"


def hop_name(hop: object) -> str:
    """The name of one hop of an answer's chain. docs/interfaces.md section 6 calls
    a chain a list of named hops; the command line prints each hop as an object
    whose ``hop`` field is that name (docs/cli.md)."""
    if isinstance(hop, str):
        return hop
    if isinstance(hop, Mapping) and isinstance(hop.get("hop"), str):
        return hop["hop"]
    raise TableError(f"a chain hop with no name: {hop!r}")


def expected_hop(text: str, resolve: Resolver) -> str:
    """A hop as the table writes it, in the answer's spelling: ``artifact:handle``
    becomes ``artifact:locator``; ``segment:step n`` stays."""
    if ":" not in text:
        raise TableError(f"a chain hop is artifact:handle or segment:step n, got {text!r}")
    head, tail = text.split(":", 1)
    if tail.startswith("step "):
        return text
    return f"{head}:{resolve(RecordRef(head, tail))}"


def is_subsequence(wanted: Sequence[str], chain: Sequence[str]) -> bool:
    """Whether ``chain`` passes the hops of ``wanted`` in their order. Other hops
    may lie between them."""
    rest = iter(chain)
    return all(any(hop == passed for passed in rest) for hop in wanted)


def _pairs(items: object) -> set[tuple]:
    if not isinstance(items, list):
        return {("not a list", repr(items))}
    return {tuple(item) for item in items}


def _as_set(items: object) -> set:
    return set(items) if isinstance(items, list) else {("not a list", repr(items))}


def _known_keys(field_name: str, expected: Mapping, keys: frozenset[str]) -> None:
    strange = set(expected) - keys
    if strange:
        raise TableError(f"{field_name}: the README names no key {sorted(strange)}")


FAILING_KEYS = frozenset({"condition", "artifact", "record", "segment", "step", "job"})
SCOPE_KEYS = frozenset({"source", "members", "records", "steps", "generation_edges"})
EXPOSURE_KEYS = frozenset({"segment", "step"})


def compare_answer(expect: Mapping, answer: Mapping, resolve: Resolver) -> list[Mismatch]:
    """Compare one printed answer with the answer fields ``expect`` holds, and
    with no other field."""
    out: list[Mismatch] = []
    strange = set(expect) - ANSWER_KEYS
    if strange:
        raise TableError(f"no comparison is written for the expected field(s) {sorted(strange)} "
                         "of an answer")

    def differ(name: str, expected: object, got: object) -> None:
        out.append(Mismatch(name, expected, got))

    for name in ("answer", "reason", "names", "names_detail", "basis"):
        if name in expect and answer.get(name) != expect[name]:
            differ(name, expect[name], answer.get(name))

    if "never" in expect and answer.get("answer") in expect["never"]:
        differ("never", f"none of {sorted(expect['never'])}", answer.get("answer"))

    if "failing" in expect:
        wanted, got = expect["failing"], answer.get("failing")
        _known_keys("failing", wanted, FAILING_KEYS)
        if not isinstance(got, Mapping):
            differ("failing", wanted, got)
        else:
            for key in ("condition", "artifact", "segment", "step", "job"):
                if key in wanted and got.get(key) != wanted[key]:
                    differ(f"failing.{key}", wanted[key], got.get(key))
            if "record" in wanted:
                # A handle names one record: a file and a place in it. Where the
                # row names the link's artifact, the record lies in it; else in the
                # handle's home file. The answer must name that record. When the
                # answer says which artifact its record lies in, that is part of
                # which record it names.
                ref = record_ref(wanted["record"], wanted.get("artifact"))
                named = (got.get("artifact"), got.get("record"))
                locator = resolve(ref)
                if named[1] != locator or named[0] not in (None, ref.artifact):
                    differ("failing.record", f"{ref} = {ref.artifact}:{locator}",
                           f"{named[0]}:{named[1]}")

    if "candidates" in expect and _as_set(answer.get("candidates")) != set(expect["candidates"]):
        differ("candidates", sorted(expect["candidates"]), answer.get("candidates"))

    if "chain" in expect:
        wanted_hops = [expected_hop(hop, resolve) for hop in expect["chain"]]
        chain = answer.get("chain")
        got_hops = [hop_name(hop) for hop in chain] if isinstance(chain, list) else None
        if got_hops is None or not is_subsequence(wanted_hops, got_hops):
            differ("chain", wanted_hops, got_hops)

    if "notes" in expect and _as_set(answer.get("notes")) != set(expect["notes"]):
        differ("notes", sorted(expect["notes"]), answer.get("notes"))

    if "untraced_steps" in expect and (
            _pairs(answer.get("untraced_steps")) != _pairs(expect["untraced_steps"])):
        differ("untraced_steps", expect["untraced_steps"], answer.get("untraced_steps"))

    if "first_exposure" in expect:
        wanted, got = expect["first_exposure"], answer.get("first_exposure")
        _known_keys("first_exposure", wanted, EXPOSURE_KEYS)
        if not isinstance(got, Mapping) or any(got.get(k) != v for k, v in wanted.items()):
            differ("first_exposure", wanted, got)

    if "scope" in expect:
        wanted, got = expect["scope"], answer.get("scope")
        _known_keys("scope", wanted, SCOPE_KEYS)
        if not isinstance(got, Mapping):
            differ("scope", wanted, got)
        else:
            if "source" in wanted and got.get("source") != wanted["source"]:
                differ("scope.source", wanted["source"], got.get("source"))
            # The README gives no order for the lists of a scope: compared as sets.
            for key in ("members", "generation_edges"):
                if key in wanted and _as_set(got.get(key)) != set(wanted[key]):
                    differ(f"scope.{key}", sorted(wanted[key]), got.get(key))
            if "records" in wanted:
                refs = [record_ref(text) for text in wanted["records"]]
                named = {f"{ref.artifact}:{resolve(ref)}" for ref in refs}
                if _as_set(got.get("records")) != named:
                    differ("scope.records", sorted(named), got.get("records"))
            if "steps" in wanted and _pairs(got.get("steps")) != _pairs(wanted["steps"]):
                differ("scope.steps", wanted["steps"], got.get("steps"))
    return out


@dataclass(frozen=True)
class Outcome:
    """What one run of the command line gave."""

    exit_code: int
    stdout: str
    stderr: str

    def error_name(self) -> str | None:
        """The ``error`` field of the JSON object on standard error, if there is one."""
        try:
            printed = json.loads(self.stderr)
        except ValueError:
            return None
        return printed.get("error") if isinstance(printed, Mapping) else None

    def printed(self) -> object:
        """The JSON document on standard output."""
        try:
            return json.loads(self.stdout)
        except ValueError:
            raise TableError(f"standard output is not one JSON document: {self.stdout[:200]!r}; "
                             f"exit code {self.exit_code}; standard error "
                             f"{self.stderr[:300]!r}") from None


def compare_stop(expect: Mapping, outcome: Outcome) -> list[Mismatch]:
    """Compare a refusal or a loud stop: the error's fixed name on standard error,
    the exit code, and nothing on standard output."""
    out = []
    if "error" in expect and outcome.error_name() != expect["error"]:
        out.append(Mismatch("error", expect["error"],
                            outcome.error_name() or f"no error object; exit {outcome.exit_code}"))
    if "exit_code" in expect and outcome.exit_code != expect["exit_code"]:
        out.append(Mismatch("exit_code", expect["exit_code"], outcome.exit_code))
    if "stdout_empty" in expect and (outcome.stdout == "") != expect["stdout_empty"]:
        out.append(Mismatch("stdout_empty", expect["stdout_empty"], outcome.stdout[:200]))
    return out


def compare_listing(expect: Mapping, listing: object) -> list[Mismatch]:
    """Compare one printed listing with ``covers`` and ``not_covered`` of a row: each
    list of ``covers`` as a set of artifact names, and ``not_covered`` as the set of
    (artifact, reason) pairs (README section 6: the key names the value, each list is
    compared as a set)."""
    if not isinstance(listing, Mapping):
        raise TableError(f"a listing is a JSON object, the command line printed {listing!r}")
    out: list[Mismatch] = []
    if "covers" in expect:
        printed = listing.get("covers")
        for kind, wanted in expect["covers"].items():
            got = printed.get(kind) if isinstance(printed, Mapping) else None
            if _as_set(got) != set(wanted):
                out.append(Mismatch(f"covers.{kind}", sorted(wanted), got))
    if "not_covered" in expect:
        wanted = {(item["artifact"], item["reason"]) for item in expect["not_covered"]}
        listed = listing.get("not_covered")
        got = ({(item.get("artifact"), item.get("reason")) for item in listed}
               if isinstance(listed, list) else {("not a list", repr(listed))})
        if got != wanted:
            out.append(Mismatch("not_covered", sorted(wanted), sorted(map(str, got))))
    return out


def compare_outcome(expect: Mapping, outcome: Outcome, resolve: Resolver) -> list[Mismatch]:
    """Compare one run of ``query`` for one artifact with a row's ``expect``: an
    answer, or a refusal or loud stop."""
    expect = dict(expect)
    loud = expect.pop(LOUD_STOP, None)
    if loud is not None and set(expect) & STOP_KEYS:
        raise TableError("an expectation holds a stop and loud_stop together")
    stop = set(expect) & STOP_KEYS
    if loud is False and outcome.exit_code == 3:
        # The answer would not be compared: a loud stop prints none. Say so first.
        return [Mismatch(LOUD_STOP, False, f"a loud stop: {outcome.error_name()!r}")]
    if stop:
        if set(expect) - STOP_KEYS:
            raise TableError(f"an expectation holds a stop {sorted(stop)} and answer fields "
                             f"{sorted(set(expect) - STOP_KEYS)} together")
        return compare_stop(expect, outcome)
    if outcome.exit_code != 0:
        return [Mismatch("answer", expect.get("answer", "an answer"),
                         f"no answer: exit code {outcome.exit_code}, error "
                         f"{outcome.error_name()!r}, standard error {outcome.stderr[:300]!r}")]
    answer = outcome.printed()
    if not isinstance(answer, Mapping):
        raise TableError(f"an answer is a JSON object, the command line printed {answer!r}")
    return compare_answer(expect, answer, resolve)
