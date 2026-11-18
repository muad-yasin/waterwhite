"""The walk of the coverage proof over records, shards and mixtures (plan 2.2).

For one request the walk gives every record it meets one of three verdicts:

- HOLDS: a recorded chain leads from the record to a record the request names;
- MAY: a link on the way is missing, weak or unverified (the first such link is
  kept, with the condition of plan 2.2 it fails);
- CLEAR: every chain of the record ends at ingested revisions outside the request,
  through links that can decide the request.

An artifact's verdict follows from its records': one HOLDS is enough for
"contains", whatever fails elsewhere (plan 2.2 closing paragraph, Decision 20);
else one MAY gives "may contain" naming the first failing link; only when every
record is CLEAR does the proof hold. The walk never stops early: it reads every
record of the artifact, so the answer does not depend on where a record sits.

Condition (h), the generating-model rule (plan 2.2(h), Decision 28), is applied here, per
record, where the walk meets a generation edge whose bound is established and the edge does not
name the requested source, and only while the generating-model switch is on (read once, when the
session begins). The model is read from the edge's stored harness log: "none" lets the record
stand; a registered checkpoint is answered for the same request, in the same session, and lets the
record stand only if its own answer is "does not contain"; any other model, and a log with no model
entry, gives "may contain" naming it. A model whose answer is being made when its own output is met
is a temporal ancestry cycle: the walk carries it as a deferred failure (the record is "may
contain") and notes it in the session; the error is raised where a verdict leaves the session
(``coverage.prove``), and only if no recorded "contains" of the artifact wins.

What each evidence label decides is plan 3.2, applied per (record, requested
source): the walk looks at each link for what it says about the requested source
and never lets a strong link to one source stand in for a weak link to another
(Decision 3).

Everything is read from the store on demand, one record at a time, and through
the store's documented read methods only: this package runs no SQL of its own. See
docs/coverage.md for what the walk reads per artifact.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, replace

from .. import extensions
from .. import vocabulary as v
from ..answers import Failing, Hop, LogEntry
from ..errors import IntegrityFault, StoreWrittenDuringQuery
from ..records import MapRow, OccurrenceRef
from ..requests import Withdrawal
from ..store.views import Artifact, Binding, SourceLink, StoredEdge

CLEAR = 0
MAY = 1
HOLDS = 2
NOT_COVERED = 3  # an artifact only: its own record is missing

# Statuses of records are kept while a walk runs, so that a record many others
# derive from is walked once. The table is a cache and nothing more: when it is
# full it is emptied, and a record met again is walked again.
MEMO_LIMIT = 200_000


@dataclass(frozen=True, slots=True)
class Status:
    """The verdict on one record.

    ``chain`` (HOLDS): the hops from this record to the requested source record.
    ``failing`` (MAY): the first failing link. ``generation``: the chain (HOLDS) or
    the proof (CLEAR) passes a generation edge. ``jobs`` (CLEAR): the generation
    jobs whose established bound the verdict rests on, nearest first. ``sources``:
    the sources at the ends of the record's chains, for naming candidates."""

    verdict: int
    chain: tuple[Hop, ...] = ()
    failing: Failing | None = None
    generation: bool = False
    jobs: tuple[str, ...] = ()
    sources: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        # A verdict that is not CLEAR carries what it rests on. Checked here, once,
        # because the tallies below read ``failing`` and ``chain``: a MAY with no
        # link, passed over, would count as CLEAR.
        if self.verdict == MAY and self.failing is None:
            raise ValueError('a "may contain" verdict names its failing link')
        if self.verdict == HOLDS and not self.chain:
            raise ValueError('a "contains" verdict carries its chain')
        if self.verdict not in (CLEAR, MAY, HOLDS):
            raise ValueError(f"not a verdict on a record: {self.verdict!r}")


@dataclass(frozen=True)
class CycleMark:
    """A temporal ancestry cycle the walk met (R-11): ``model`` is the name of the checkpoint
    whose answer was being made, ``record`` the output record (``artifact:locator``) whose
    generation edge names it as its model, and which the checkpoint's own history reads."""

    model: str
    record: str


@dataclass(frozen=True)
class Outcome:
    """The verdict on one artifact, before it becomes a proof or a finding. ``cycle`` is set
    on an outcome that is not "contains" when a record of the artifact, or of what the
    artifact rests on, met a cycle: ``coverage.prove`` raises the named error for it, where the
    verdict leaves the session."""

    verdict: int
    reason: str | None = None
    chain: tuple[Hop, ...] = ()
    failing: Failing | None = None
    generation: bool = False
    jobs: tuple[str, ...] = ()
    walked: int = 0
    # A checkpoint only (plan 2.4):
    basis: str | None = None
    untraced_steps: tuple[tuple[str, int], ...] = ()
    first_exposure: tuple[str, int] | None = None
    level: str | None = None
    runs: tuple[str, ...] = ()
    steps: tuple[tuple[str, int, int], ...] = ()
    cycle: CycleMark | None = None

    def __post_init__(self) -> None:
        if self.verdict == MAY and self.failing is None:
            raise ValueError('a "may contain" verdict names its failing link')
        if self.verdict == HOLDS and not self.chain:
            raise ValueError('a "contains" verdict carries its chain')
        if self.verdict == NOT_COVERED and self.reason not in v.NOT_COVERED_REASONS:
            raise ValueError('a "not covered" verdict says which record is missing')
        if self.verdict not in (CLEAR, MAY, HOLDS, NOT_COVERED):
            raise ValueError(f"not a verdict on an artifact: {self.verdict!r}")


@dataclass(frozen=True)
class FileFacts:
    """What the walk learns once about a JSONL artifact, so that a record of it
    costs no query the artifact already answers: whether it has a record of its
    own, which members cover the whole file, whether members name single records
    of it, whether any of its records has an edge, and its source links."""

    gate: Failing | None
    file_members: tuple[tuple[str, str], ...]
    record_parts: bool
    has_edges: bool
    file_links: tuple[SourceLink, ...]
    record_links: bool


@dataclass(frozen=True)
class ShardGate:
    """What stands before a shard's documents: its map. ``missing`` is set when the
    shard has no record of its own (no row, or no map); ``label`` when the map's
    stored label is not one its derivation and verification allow (2.2(f))."""

    instance_id: str
    binding: Binding | None
    missing: Failing | None
    label: Failing | None


class Tally:
    """Adds up the verdicts of an artifact's records, in walk order."""

    def __init__(self) -> None:
        self.holds: tuple[Hop, ...] | None = None
        self.holds_generation = False
        self.failing: Failing | None = None
        self.generation = False
        self.jobs: dict[str, None] = {}
        self.walked = 0

    def add(self, status: Status) -> None:
        self.walked += 1
        if status.verdict == HOLDS:
            if self.holds is None:
                self.holds = status.chain
            self.holds_generation = self.holds_generation or status.generation
        elif status.verdict == MAY:
            self.fail(status.failing)
        else:
            self.generation = self.generation or status.generation
            for job in status.jobs:
                self.jobs.setdefault(job)

    def fail(self, failing: Failing | None) -> None:
        if self.failing is None and failing is not None:
            self.failing = failing

    def merge(self, outcome: Outcome) -> None:
        """Take in the verdict of an artifact one level up (a mixture's shard). The
        caller names a shard that is "not covered" before it comes here."""
        self.walked += outcome.walked
        if outcome.verdict == NOT_COVERED:
            raise ValueError("an artifact with no record of its own has no verdict to merge")
        if outcome.verdict == HOLDS:
            if self.holds is None:
                self.holds = outcome.chain
            self.holds_generation = self.holds_generation or outcome.generation
        elif outcome.verdict == MAY:
            self.fail(outcome.failing)
        else:
            self.generation = self.generation or outcome.generation
            for job in outcome.jobs:
                self.jobs.setdefault(job)

    def outcome(self) -> Outcome:
        if self.holds is not None:
            return Outcome(HOLDS, chain=self.holds, generation=self.holds_generation, walked=self.walked)
        if self.failing is not None:
            return Outcome(MAY, failing=self.failing, walked=self.walked)
        return Outcome(CLEAR, generation=self.generation, jobs=tuple(self.jobs), walked=self.walked)


def _merge_clear(statuses: Iterable[Status], *, generation: bool = False, job: str | None = None) -> Status:
    jobs: dict[str, None] = {}
    if job is not None:
        jobs[job] = None
    sources: set[str] = set()
    for status in statuses:
        generation = generation or status.generation
        for name in status.jobs:
            jobs.setdefault(name)
        sources.update(status.sources)
    return Status(CLEAR, generation=generation, jobs=tuple(jobs), sources=frozenset(sources))


class Session:
    """One walk: a store, a request, and what the walk has learned so far.

    A session can answer for many artifacts (a listing uses one): the verdict of a
    shard that several mixtures hold is then computed once."""

    def __init__(self, store, request: Withdrawal) -> None:
        if not isinstance(request, Withdrawal):
            raise TypeError("a session takes a request expanded by waterwhite.requests")
        self.store = store
        self.request = request
        self.logs: dict[str, LogEntry] = {}
        self._members = frozenset(request.members)
        self._records = frozenset(request.records)
        self._memo: dict[OccurrenceRef, Status] = {}
        self._names: dict[str, str] = {}
        self._facts: dict[str, FileFacts] = {}
        self._shard_gates: dict[str, ShardGate] = {}
        self._outcomes: dict[str, Outcome] = {}
        self._mark = store.state_mark()
        # The generating-model switch, read once for this query (R-8): a plain comparison with
        # "on", since every entry point has already refused a switch it cannot read
        # (``extensions.check_switches``). The integrity closure, condition (h) and the note of
        # the answer all use this one value.
        self.generating_model_on = (
            store.settings().get(extensions.setting_key(extensions.GENERATING_MODEL)) == extensions.ON
        )
        # The checkpoints whose answer is being made, and the cycles the walk has met (R-11).
        # A record or an artifact whose computation met a cycle is kept out of the caches.
        self._in_progress: set[str] = set()
        self._cycles: list[CycleMark] = []

    def ensure_current(self) -> None:
        """Refuse when the store was written after this session began: what the walk
        learned is then out of date, and a negative built from it would be the
        answer of a store that no longer exists (plan 2.2: the walk is made on demand
        from the store as it is). A new session reads the store as it is."""
        if self.store.state_mark() != self._mark:
            raise StoreWrittenDuringQuery()

    def holds_for_proof(self) -> Callable[[], bool]:
        """What a proof keeps to say whether the store is still as this walk read it:
        the store held weakly and the mark taken when the session began. It returns
        False for a store that was written, and raises for one that is gone or closed,
        which the answer type tells apart (docs/interfaces.md section 2)."""
        held = weakref.ref(self.store)
        mark = self._mark

        def current() -> bool:
            store = held()
            if store is None:
                raise LookupError("the store the proof was made from is gone")
            return store.state_mark() == mark  # a closed store refuses and raises

        return current

    # ------------------------------------------------------------------------------
    # Names and small builders
    # ------------------------------------------------------------------------------

    def name(self, instance_id: str) -> str:
        """A name of an artifact, or its instance id when it has none."""
        found = self._names.get(instance_id)
        if found is None:
            artifact = self.store.artifact(instance_id)
            found = artifact.names[0] if artifact is not None and artifact.names else instance_id
            self._names[instance_id] = found
        return found

    def _hop(self, ref: OccurrenceRef, link: str, **more: object) -> Hop:
        return Hop(
            artifact=self.name(ref.instance_id), instance=ref.instance_id, record=ref.locator,
            link=link, **more,
        )

    def _fail(self, names: str, condition: str | None, ref: OccurrenceRef | None, text: str,
              **more: object) -> Failing:
        if ref is None:
            return Failing(names=names, condition=condition, text=text, **more)
        return Failing(
            names=names, condition=condition, artifact=self.name(ref.instance_id),
            instance=ref.instance_id, record=ref.locator, text=text, **more,
        )

    def missing_record(self, instance_id: str, label: str | None, reason: str, text: str) -> Failing:
        """The link a downstream artifact names when something upstream has no
        record of its own (plan 2.1, Decision 21)."""
        return Failing(
            names=v.NAMES_UPSTREAM_NOT_COVERED, artifact=label or self.name(instance_id),
            instance=instance_id, reason=reason, text=text,
        )

    def _file_facts(self, instance_id: str) -> FileFacts:
        """Read once per JSONL artifact on the way. Its ``gate`` is set when the
        artifact has no record of its own (its binding is gone): everything derived
        from it then answers "may contain" naming it (Decision 21)."""
        facts = self._facts.get(instance_id)
        if facts is None:
            store = self.store
            gate = None
            if store.binding(instance_id) is None:
                gate = self.missing_record(
                    instance_id, None, v.REASON_RECORD_MISSING,
                    f"{self.name(instance_id)} has no record of its own: its occurrence "
                    "record is missing",
                )
            edges = store.edges(instance_id)
            has_edges = next(edges, None) is not None
            edges.close()
            links = store.record_source_links(instance_id)
            record_links = next(links, None) is not None
            links.close()
            facts = FileFacts(
                gate=gate,
                file_members=store.file_members(instance_id),
                record_parts=store.has_record_parts(instance_id),
                has_edges=has_edges,
                file_links=store.file_source_links(instance_id),
                record_links=record_links,
            )
            self._facts[instance_id] = facts
        return facts

    # ------------------------------------------------------------------------------
    # One record of a JSONL artifact
    # ------------------------------------------------------------------------------

    def occurrence(self, ref: OccurrenceRef) -> Status:
        """The verdict on one JSONL record. Parents are walked first; the walk keeps
        its own stack, so a long chain needs no deep recursion, and a chain that
        runs into itself is a chain with no origin."""
        hit = self._memo.get(ref)
        if hit is not None:
            return hit
        # Each frame keeps the number of cycles met before it began: a verdict computed while a
        # cycle was met depends on which checkpoints are in progress, so it enters no cache
        # (R-9). A "contains" does not depend on it and is kept.
        stack: list[tuple[OccurrenceRef, Generator, int]] = [(ref, self._eval(ref), len(self._cycles))]
        active = {ref}
        value: Status | None = None
        while stack:
            key, running, met = stack[-1]
            try:
                need = running.send(value)
            except StopIteration as done:
                stack.pop()
                active.discard(key)
                value = done.value
                if value.verdict == HOLDS:
                    # A recorded "contains" wins and does not rest on a cycle that was met on the
                    # way: the cycles noted since this frame began are dropped, so that they do
                    # not leak to an answer that never rests on them.
                    del self._cycles[met:]
                if len(self._cycles) == met:
                    if len(self._memo) >= MEMO_LIMIT:
                        self._memo.clear()
                    self._memo[key] = value
                continue
            hit = self._memo.get(need)
            if hit is not None:
                value = hit
            elif need in active:
                value = Status(MAY, failing=self._fail(
                    v.NAMES_MISSING_ORIGIN, v.CONDITION_A, need,
                    f"the chain of {need.key} leads back to itself and reaches no ingested revision",
                ))
            else:
                stack.append((need, self._eval(need), len(self._cycles)))
                active.add(need)
                value = None
        assert value is not None
        return value

    def _requested(self, ref: OccurrenceRef, covering) -> str | None:
        """The requested member that covers this ingested record, or None when the
        request does not name the record. ``covering`` holds (member, source)."""
        if self._records and ref not in self._records:
            return None
        for member, _source in covering:
            if member in self._members:
                return member
        return None

    def _eval(self, ref: OccurrenceRef) -> Generator[OccurrenceRef, Status, Status]:
        facts = self._file_facts(ref.instance_id)
        if facts.gate is not None:
            return Status(MAY, failing=facts.gate)
        store = self.store
        found: list[Status] = []

        # Where a chain ends: a record of an ingested revision (2.2(a), 2.3).
        covering = facts.file_members
        if facts.record_parts:
            covering = covering + store.record_members(ref)
        if covering:
            sources = frozenset(source for _member, source in covering)
            member = self._requested(ref, covering)
            if member is not None:
                found.append(Status(HOLDS, chain=(self._hop(ref, v.HOP_MEMBER, member=member),),
                                    sources=sources))
            else:
                found.append(Status(CLEAR, sources=sources))

        # The record's edge to its parents.
        edge = store.edge_of(ref) if facts.has_edges else None
        if edge is not None:
            found.append((yield from self._edge(ref, edge)))

        # Links to a member with no parent record behind them. Each is read for what
        # it says about the requested source only (Decision 3).
        links = store.source_links(ref) if facts.record_links else facts.file_links
        for link in links:
            status = self._link(ref, link)
            if status is not None:
                found.append(status)

        holds = [s for s in found if s.verdict == HOLDS]
        if holds:
            first = holds[0]
            if not first.generation and any(s.generation for s in holds):
                first = replace(first, generation=True)
            return first
        for status in found:
            if status.verdict == MAY:
                return status
        if not covering and edge is None:
            beside = ""
            if links:
                named = ", ".join(sorted({link.member for link in links}))
                beside = f"; only a link to {named} is recorded"
            return Status(MAY, failing=self._fail(
                v.NAMES_MISSING_ORIGIN, v.CONDITION_A, ref,
                f"{self.name(ref.instance_id)}:{ref.locator} has no recorded origin: no edge and "
                f"no ingested revision{beside}",
            ))
        return _merge_clear(found)

    # ------------------------------------------------------------------------------
    # An edge (plan 3.1, 3.2, 3.3)
    # ------------------------------------------------------------------------------

    def _job(self, edge: StoredEdge) -> str:
        """The job an answer names for an edge. A generation edge keeps the name
        of its job itself (DECISIONS.md D-24), so the job is named whether its
        harness log is stored or gone (plan Check 5). An edge that is not a
        generation edge has no job: for a publisher's pointer that fails (g) the
        answer gives the name of the edge's step."""
        if edge.is_generation:
            if not edge.job_name:
                # The schema holds no such edge (a CHECK on the edge table).
                raise IntegrityFault(
                    f"the generation edge of {edge.child.key} names no job",
                    child=edge.child.key,
                )
            return edge.job_name
        return edge.transform.name

    def _bound_established(self, edge: StoredEdge) -> bool:
        """Condition 2.2(g): the upper bound of a generation edge. It is read from
        the stored harness log (the log lists no read outside the hand-off that the
        harness let through), the log is the log of the job the edge names, the
        stored field must agree, every input must lie in the hand-off, and a
        mapped edge never has one (plan 3.3, Decision 13)."""
        log = edge.harness_log
        if not edge.is_generation or edge.label != v.OBSERVED or log is None:
            return False
        if log.job_name != edge.job_name:
            return False
        if not edge.bound_field or not log.bound_established():
            return False
        handed = set(log.handed)
        return all(item.parent in handed for item in edge.inputs)

    def _edge(self, ref: OccurrenceRef, edge: StoredEdge) -> Generator[OccurrenceRef, Status, Status]:
        where = f"{self.name(ref.instance_id)}:{ref.locator}"
        try:
            allowed = v.labels_allowed(edge.derivation, edge.verification)
        except ValueError:
            allowed = ()
        if edge.label not in allowed:
            return Status(MAY, failing=self._fail(
                v.NAMES_LABEL_STRONGER, v.CONDITION_F, ref,
                f"the edge of {where} is labelled {edge.label!r}; its derivation "
                f"{edge.derivation!r} with verification {edge.verification!r} allows "
                f"{', '.join(allowed) or 'no label'}",
            ))
        if edge.label == v.UNKNOWN:
            return Status(MAY, failing=self._fail(
                v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                f"the edge of {where} is labelled unknown: no link is recorded",
            ))
        generation = edge.is_generation
        job = self._job(edge) if generation else None
        hop = self._hop(ref, edge.label, generation=generation, job=job)

        if edge.label == v.FILE_LEVEL:
            # Known for the file, not for the record. It can show "contains" for a
            # request about the whole source and decides nothing else (3.2, L1).
            if self.request.record_level:
                return Status(MAY, failing=self._fail(
                    v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                    f"the link of {where} is file-level and the request names records",
                ))
            results = []
            for parent in edge.parents:
                results.append((yield parent))
            for status in results:
                if status.verdict == HOLDS:
                    return Status(HOLDS, chain=(hop,) + status.chain,
                                  generation=generation or status.generation, sources=status.sources)
            return Status(MAY, failing=self._fail(
                v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                f"the link of {where} is file-level: it cannot show that the record holds "
                "nothing of the source",
            ))

        parents = edge.parents
        if edge.label == v.CANDIDATES or (edge.parent_mode == v.MODE_ANY and edge.selected is None):
            results = []
            for parent in parents:
                results.append((yield parent))
            return self._twins(ref, hop, results)

        # A record-level link: observed, mapped or recovered.
        if edge.parent_mode == v.MODE_ANY:
            # Equivalent copies: the history keeps the copy the step selected
            # (plan 2.3); the others decide what a rebuild does, not this answer.
            parents = (edge.selected,)
        scope_only = edge.in_scope_only
        if generation and len(edge.inputs) == 1 and scope_only:
            # A job's sole handed record is a parent of its output, stated or not
            # (plan 3.1 "Edge", DECISIONS.md D-21).
            parents, scope_only = scope_only, ()
        results = []
        for parent in parents:
            results.append((yield parent))
        handed = []
        for parent in scope_only:
            handed.append((parent, (yield parent)))

        for status in results:
            if status.verdict == HOLDS:
                # The chain shown is the first parent's. Whether any chain of the
                # record passes a generation edge is kept for the note of 2.1.
                passes = generation or any(s.generation for s in results if s.verdict == HOLDS)
                return Status(HOLDS, chain=(hop,) + status.chain, generation=passes,
                              sources=status.sources)
        for status in results:
            if status.verdict == MAY:
                return status
        for parent, status in handed:
            if status.verdict == HOLDS:
                # Handed to the job and not stated as used: not a parent, and never
                # "contains" on that basis (Decision 25).
                return Status(MAY, failing=Failing(
                    names=v.NAMES_IN_SCOPE_ONLY, condition=v.CONDITION_G,
                    artifact=self.name(parent.instance_id), instance=parent.instance_id,
                    record=parent.locator, job=self._job(edge),
                    text=f"the job that made {where} was handed "
                         f"{self.name(parent.instance_id)}:{parent.locator} and its output does "
                         "not state it used it",
                ))
            if status.verdict == MAY:
                return status
        if not parents and not handed:
            # An edge that names no input gives the record no recorded origin, a job
            # that was handed nothing included (plan 2.3: an occurrence whose chain
            # does not end at an ingested revision fails 2.2(a)).
            return Status(MAY, failing=self._fail(
                v.NAMES_MISSING_ORIGIN, v.CONDITION_A, ref,
                f"the edge of {where} names no parent",
            ))

        # The edge does not name the requested source. Past a generation edge that
        # is a claim about what the job read, and needs the bound (2.2(g), L8).
        if generation or edge.label == v.MAPPED:
            if self.generating_model_on and edge.label == v.MAPPED:
                # A publisher's pointer is the only evidence: the generating model is not
                # registered, so (h) cannot be met (plan 3.3, line "Where a publisher's pointer
                # is the only evidence"). It names no model, so the link is the record's own.
                return Status(MAY, failing=self._fail(
                    v.NAMES_MODEL_NOT_REGISTERED, v.CONDITION_H, ref,
                    f"the generating model of {where} is not registered: a publisher's "
                    "pointer is the only evidence.",
                    job=self._job(edge),
                ))
            if not self._bound_established(edge):
                return Status(MAY, failing=self._fail(
                    v.NAMES_BOUND_NOT_ESTABLISHED, v.CONDITION_G, ref,
                    self._why_no_bound(edge, where), job=self._job(edge),
                ))
            log = edge.harness_log
            assert log is not None
            if log.job_name not in self.logs:
                self.logs[log.job_name] = LogEntry(
                    job=log.job_name, log_digest=log.log_digest, handed=len(log.handed),
                    reads=len(log.reads),
                    refused_attempts=tuple(
                        r.outside_ref or (r.occurrence.key if r.occurrence is not None else "")
                        for r in log.reads if r.refused
                    ),
                    model=log.model,
                    harness=log.harness,
                )
            if self.generating_model_on:
                # Condition (h), after (g): the bound is established and the log is in hand.
                failing = self._model_link(ref, edge, log, where)
                if failing is not None:
                    return Status(MAY, failing=failing)
            return _merge_clear((*results, *(s for _, s in handed)), generation=True, job=log.job_name)
        return _merge_clear((*results, *(s for _, s in handed)))

    def _why_no_bound(self, edge: StoredEdge, where: str) -> str:
        log = edge.harness_log
        if edge.label == v.MAPPED:
            return (f"the edge of {where} is a publisher's pointer: it names what the record "
                    "was made from and shows nothing about what else the job read")
        if not edge.is_generation:
            return f"the edge of {where} has no harness log"
        if log is None:
            return (f"the generation edge of {where}, made by job {edge.job_name!r}, has no "
                    "stored harness log; a bound field alone does not establish what the job read")
        if log.job_name != edge.job_name:
            return (f"the generation edge of {where} names the job {edge.job_name!r} and its "
                    f"stored harness log is the log of the job {log.job_name!r}")
        if log.harness != v.HARNESS_CONFINED:
            written_by = ("the in-process harness" if log.harness == v.HARNESS_IN_PROCESS
                          else "a harness it does not name")
            return (f"the stored harness log of job {log.job_name!r} was written by {written_by}, "
                    "which cannot say what else the job opened; only a log of the confined "
                    "harness establishes a bound")
        if not log.reads_within_hand_off():
            return (f"the stored harness log of job {log.job_name!r} lists a read outside its "
                    "hand-off that was not refused")
        if not log.hand_off_checked:
            return (f"the stored harness log of job {log.job_name!r} was never compared with the "
                    "records it was handed: the texts handed to the job were not shown and "
                    "compared, so what the job was given is only the caller's statement")
        if not edge.bound_field:
            return f"the generation edge of {where} is marked as having no established bound"
        return f"an input of the generation edge of {where} lies outside the hand-off of its job"

    def _model_link(self, ref: OccurrenceRef, edge: StoredEdge, log, where: str) -> Failing | None:
        """Condition 2.2(h) for a generation edge whose bound is established and whose log is in
        hand: None when it holds, else the failing link. The model is read from the edge's
        STORED harness log, never from the job's own statement.

        A log with no model entry is "not established", never "no model" (it names the job). Each
        model the entry names, in listed order (``resolve_model``: the list form, one reading), must
        pass: "none"; or a registered checkpoint whose own answer, for the same request and in this
        session, is "does not contain". The first that fails gives the link."""
        job = self._job(edge)
        if log.model is None:
            return self._fail(
                v.NAMES_MODEL_ENTRY_MISSING, v.CONDITION_H, ref,
                f"the harness log of job {job} has no model entry.", job=job,
            )
        for model in self.store.resolve_model(log.model):
            if model.none:
                continue
            checkpoint = model.checkpoint
            if checkpoint is None:
                if model.kind is None:
                    said = f"the generating model {model.text} is not a registered checkpoint"
                else:
                    said = f"{model.text} names a {model.kind}, not a checkpoint"
                return Failing(
                    names=v.NAMES_MODEL_NOT_REGISTERED, condition=v.CONDITION_H, artifact=model.text,
                    job=job, text=f"{said}.",
                )
            if checkpoint.instance_id in self._in_progress:
                # The model's own answer is being made and this output is among what its history
                # reads: a cycle. It is carried as a failed link and noted; the error is raised
                # where the verdict leaves the session, unless a recorded "contains" wins (R-11).
                self._cycles.append(CycleMark(model.text, where))
                return self._model_exposed(checkpoint, model.text, job,
                                           "the history of the model reads this output")
            outcome = self.outcome(checkpoint)
            if outcome.verdict == NOT_COVERED:
                return self.missing_record(
                    checkpoint.instance_id, model.text, outcome.reason or v.REASON_RECORD_MISSING,
                    f"the generating model {model.text} has no record of its own.",
                )
            if outcome.verdict != CLEAR:
                answers = v.CONTAINS if outcome.verdict == HOLDS else v.MAY_CONTAIN
                return self._model_exposed(
                    checkpoint, model.text, job, f"the generating model {model.text} answers "
                    f'"{answers}" for this request')
        return None

    def _model_exposed(self, checkpoint: Artifact, text: str, job: str, why: str) -> Failing:
        return Failing(
            names=v.NAMES_MODEL_EXPOSED, condition=v.CONDITION_H, artifact=text,
            instance=checkpoint.instance_id, job=job, text=f"{why}.",
        )

    def _twins(self, ref: OccurrenceRef, hop: Hop, results: list[Status]) -> Status:
        """A set of candidate parents the derivation cannot tell apart (plan 3.2,
        L3): "contains" when every candidate belongs to the request, a link to
        other sources when none does, "may contain" for each candidate's source
        otherwise."""
        where = f"{hop.artifact}:{hop.record}"
        if not results:
            return Status(MAY, failing=self._fail(
                v.NAMES_MISSING_ORIGIN, v.CONDITION_A, ref, f"{where} names no candidate parent",
            ))
        holds = [s for s in results if s.verdict == HOLDS]
        sources = frozenset().union(*(s.sources for s in results))
        if len(holds) == len(results):
            return Status(HOLDS, chain=(hop,) + holds[0].chain,
                          generation=any(s.generation for s in holds), sources=sources)
        if holds:
            return Status(MAY, failing=self._fail(
                v.NAMES_CANDIDATES, v.CONDITION_B, ref,
                f"{where} has {len(results)} candidate parents and {len(holds)} of them "
                "belong to the request",
                candidates=tuple(sorted(sources)),
            ))
        for status in results:
            if status.verdict == MAY:
                return status
        return _merge_clear(results)

    # ------------------------------------------------------------------------------
    # A link to a member with no parent record behind it (plan 3.2, L1)
    # ------------------------------------------------------------------------------

    def _link(self, ref: OccurrenceRef, link) -> Status | None:
        """What one source link says about the requested source. None: nothing."""
        where = f"{self.name(ref.instance_id)}:{ref.locator}"
        requested = link.member in self._members
        sources = frozenset({link.source})
        # Condition 2.2(f) holds for a source link as for an edge and a map: a
        # label edited to one its derivation and verification do not allow shows
        # nothing, for the member it names or for any other.
        try:
            allowed = v.labels_allowed(link.derivation, link.verification)
        except ValueError:
            allowed = ()
        if link.label not in allowed:
            return Status(MAY, failing=self._fail(
                v.NAMES_LABEL_STRONGER, v.CONDITION_F, ref,
                f"the link of {where} to {link.member} is labelled {link.label!r}; its "
                f"derivation {link.derivation!r} with verification {link.verification!r} "
                f"allows {', '.join(allowed) or 'no label'}",
            ), sources=sources)
        hop = self._hop(ref, v.hop_source_link(link.label), member=link.member)
        if link.label == v.FILE_LEVEL:
            if not requested:
                return None
            if self.request.record_level:
                return Status(MAY, failing=self._fail(
                    v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                    f"{where} has a file-level link to {link.member} ({link.evidence}) and the "
                    "request names records",
                ), sources=sources)
            return Status(HOLDS, chain=(hop,), sources=sources)
        if link.label == v.MAPPED:
            if not requested:
                return Status(MAY, failing=self._fail(
                    v.NAMES_BOUND_NOT_ESTABLISHED, v.CONDITION_G, ref,
                    f"{where} is tied to {link.member} by a publisher's pointer, which shows "
                    "nothing about what else it was made from",
                ), sources=sources)
            if self.request.record_level:
                return Status(MAY, failing=self._fail(
                    v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                    f"{where} is tied to {link.member} by a pointer that names no record, and "
                    "the request names records",
                ), sources=sources)
            return Status(HOLDS, chain=(hop,), sources=sources)
        if requested:
            return Status(MAY, failing=self._fail(
                v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                f"the link of {where} to {link.member} is labelled unknown",
            ), sources=sources)
        return None

    # ------------------------------------------------------------------------------
    # A shard and its documents (plan 3.3 "Tokenization")
    # ------------------------------------------------------------------------------

    def shard_gate(self, instance_id: str, label: str | None = None) -> ShardGate:
        """The shard's own record: the map a reader gets (the store picks it,
        DECISIONS.md D-20). ``label`` is how to name a shard the store has no row
        for (the path a snapshot gives)."""
        gate = self._shard_gates.get(instance_id)
        if gate is not None:
            return gate
        store = self.store
        artifact = store.artifact(instance_id)
        binding = None if artifact is None or artifact.kind != v.KIND_SHARD else store.binding(instance_id)
        missing = None
        wrong = None
        if binding is None:
            shown = label if artifact is None else self.name(instance_id)
            missing = self.missing_record(
                instance_id, shown, v.REASON_RECORD_MISSING,
                f"the shard {shown} has no record of its own: "
                + ("the store has no row for it" if artifact is None else "it has no tokenization map"),
            )
        else:
            try:
                allowed = v.labels_allowed(binding.derivation, binding.verification)
            except ValueError:
                allowed = ()
            if binding.label not in allowed:
                wrong = Failing(
                    names=v.NAMES_LABEL_STRONGER, condition=v.CONDITION_F,
                    artifact=self.name(instance_id), instance=instance_id,
                    text=f"the {binding.namespace} map of {self.name(instance_id)} is labelled "
                         f"{binding.label!r}; its derivation {binding.derivation!r} with "
                         f"verification {binding.verification!r} allows "
                         f"{', '.join(allowed) or 'no label'}",
                )
        gate = ShardGate(instance_id, binding, missing, wrong)
        self._shard_gates[instance_id] = gate
        return gate

    def document(self, gate: ShardGate, row: MapRow) -> Status:
        """The verdict on one document of a shard, through its map row."""
        binding = gate.binding
        assert binding is not None
        ref = OccurrenceRef(gate.instance_id, f"D{row.ordinal}")
        where = f"{self.name(gate.instance_id)}:{ref.locator}"
        hop = self._hop(ref, row.label)
        try:
            allowed = v.labels_allowed(binding.derivation, binding.verification)
        except ValueError:
            allowed = ()
        if row.label not in allowed:
            return Status(MAY, failing=self._fail(
                v.NAMES_LABEL_STRONGER, v.CONDITION_F, ref,
                f"the map row of {where} is labelled {row.label!r}; the map's derivation "
                f"{binding.derivation!r} with verification {binding.verification!r} allows "
                f"{', '.join(allowed) or 'no label'}",
            ))
        if row.label == v.UNKNOWN:
            return Status(MAY, failing=self._fail(
                v.NAMES_LEVEL_TOO_WEAK, v.CONDITION_B, ref,
                f"the map row of {where} is labelled unknown: the map's verification is "
                f"{binding.verification!r}",
            ))
        if row.label == v.CANDIDATES:
            return self._twins(ref, hop, [self.occurrence(c) for c in row.candidates])
        if row.parent is None:
            return Status(MAY, failing=self._fail(
                v.NAMES_MISSING_ORIGIN, v.CONDITION_A, ref, f"the map row of {where} names no record",
            ))
        status = self.occurrence(row.parent)
        if status.verdict == HOLDS:
            return replace(status, chain=(hop,) + status.chain)
        return status

    # ------------------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------------------

    def outcome(self, artifact: Artifact) -> Outcome:
        """The verdict on one registered artifact. Kept for the session. The kind
        that decides which rule walks the artifact is the kind the store holds for
        its instance id, never the one of the object handed in.

        A checkpoint's instance id is in the session's set of answers being made while its
        history is walked (released in a ``finally``). A cycle met while the answer is made
        (``Session._model_link``) leaves its mark on the outcome, when the outcome is not "contains",
        and keeps the outcome out of the cache: it depends on which checkpoints are in progress.
        Nothing is raised here (R-11): a nested call returns the outcome with its mark, and the
        error is raised by ``coverage.prove``, where the verdict leaves the session."""
        self.ensure_current()
        found = self._outcomes.get(artifact.instance_id)
        if found is None:
            held = self.store.artifact(artifact.instance_id)
            if held is None:
                raise ValueError(f"the store has no row for the artifact {artifact.instance_id}")
            artifact = held
            met = len(self._cycles)
            tracked = artifact.kind == v.KIND_CHECKPOINT
            if tracked:
                self._in_progress.add(artifact.instance_id)
            try:
                if artifact.kind == v.KIND_JSONL:
                    found = self._jsonl(artifact)
                elif artifact.kind == v.KIND_SHARD:
                    found = self._shard(artifact)
                elif artifact.kind == v.KIND_MIXTURE:
                    found = self._mixture(artifact)
                elif tracked:
                    from ._checkpoint import checkpoint_outcome

                    found = checkpoint_outcome(self, artifact)
                else:
                    raise ValueError(f"no answer rule for an artifact of kind {artifact.kind!r}")
            finally:
                if tracked:
                    self._in_progress.discard(artifact.instance_id)
            if found.verdict == HOLDS:
                del self._cycles[met:]  # a recorded "contains" does not rest on a cycle met on the way
            if len(self._cycles) > met:
                if found.verdict == MAY:
                    found = replace(found, cycle=self._cycles[met])
                return found
            self._outcomes[artifact.instance_id] = found
        return found

    def _jsonl(self, artifact: Artifact) -> Outcome:
        """A JSONL file: every line is a record (plan 2.3)."""
        if self.store.binding(artifact.instance_id) is None:
            return Outcome(NOT_COVERED, reason=v.REASON_RECORD_MISSING)
        tally = Tally()
        for occurrence in self.store.occurrences(artifact.instance_id):
            tally.add(self.occurrence(occurrence.ref))
        return tally.outcome()

    def _shard(self, artifact: Artifact) -> Outcome:
        """A shard: every document, through the map a reader gets."""
        gate = self.shard_gate(artifact.instance_id)
        if gate.missing is not None:
            return Outcome(NOT_COVERED, reason=v.REASON_RECORD_MISSING)
        tally = Tally()
        tally.fail(gate.label)
        for row in self.store.map_rows(artifact.instance_id):
            tally.add(self.document(gate, row))
        return tally.outcome()

    def _mixture(self, artifact: Artifact) -> Outcome:
        """A mixture: the shards of its snapshot, in the trainer's order. A shard
        with no record of its own is named and the mixture answers "may contain"
        (plan 2.1, Decision 21), unless another shard shows "contains"."""
        snapshot = self.store.mixture_shards(artifact.instance_id)
        if not snapshot:
            # A mixture's own record is its snapshot (plan 2.1). The store writes no
            # mixture without one, so an empty snapshot is a record that is gone,
            # and no shard walked is no proof of anything.
            return Outcome(NOT_COVERED, reason=v.REASON_RECORD_MISSING)
        tally = Tally()
        for entry in snapshot:
            gate = self.shard_gate(entry.shard_instance_id, entry.path)
            if gate.missing is not None:
                tally.fail(gate.missing)
                continue
            shard = self.store.artifact(entry.shard_instance_id)
            assert shard is not None
            tally.merge(self.outcome(shard))
        return tally.outcome()

    # ------------------------------------------------------------------------------
    # Candidate sources of a "may contain" (plan 2.1)
    # ------------------------------------------------------------------------------

    def candidate_sources(self, artifact: Artifact) -> tuple[str, ...]:
        """The sources among the inputs of the step that made the artifact, followed
        back through the artifacts those inputs were made from. Artifacts are few;
        no record is read."""
        store = self.store
        seen: set[str] = set()
        queue = [artifact.instance_id]
        sources: set[str] = set()
        while queue:
            instance_id = queue.pop()
            if instance_id in seen:
                continue
            seen.add(instance_id)
            held = store.artifact(instance_id)
            if held is None:
                continue
            sources.update(store.sources_of_artifact(instance_id))
            queue.extend(held.inputs)
            if held.kind == v.KIND_MIXTURE:
                queue.extend(s.shard_instance_id for s in store.mixture_shards(instance_id))
            elif held.kind == v.KIND_CHECKPOINT:
                history = store.checkpoint_history(instance_id)
                for part in () if history is None else history.segments:
                    segment = store.segment(part.segment)
                    if segment is not None and segment.mixture_instance_id is not None:
                        queue.append(segment.mixture_instance_id)
        return tuple(sorted(sources))
