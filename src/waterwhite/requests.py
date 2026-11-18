"""Requests: what a query asks, expanded against the store when the query runs.

The first build answers one kind of request, the withdrawal of a source (plan 2.3,
Decision 24). A source is a named set of registered members. A withdrawal names
the source and may narrow to some of its members and to a list of records. It is
expanded here, when the query runs, to the members it covers, and every answer
prints that list so a team can see a revision it forgot to group under the source.

A request this build cannot answer is refused with a named error before any answer
exists. Nothing is approximated (plan 2.3 "Unsupported requests", L7):

- a source with no registered member: "source not registered";
- a named member that is not registered under the source: "member not registered";
- a record that is not a record of the members the request covers: "record not in
  source";
- a content ban, a request that starts from flagged records: "extension off" while
  their extensions are off (plan 1.4);
- a selector on metadata, such as a URL prefix: "unsupported selector".

The plan names "source not registered" for a withdrawal (plan 2.3); "member not
registered" and "record not in source" are the build's additions, which plan 5.8,
B-7 leaves to the builder (DECISIONS.md D-26).

The error classes are those of ``waterwhite.errors``; this module defines none.

Plan sections: 2.3, 1.4, Decision 24, limits L7 and L12.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from . import extensions, ids
from .errors import (
    MemberNotRegistered,
    RecordNotInSource,
    Refused,
    SourceNotRegistered,
    UnsupportedSelector,
)
from .records import OccurrenceRef, WithdrawRequest

KIND_WITHDRAW = "withdraw"
KIND_BAN = "ban"
KIND_FLAGGED = "flagged"

# The keys a withdrawal given as a mapping may hold. Any other key is a selector
# this build does not read.
_WITHDRAW_KEYS = frozenset({"kind", "source", "members", "records"})


@dataclass(frozen=True)
class Withdrawal:
    """A withdrawal as it ran: the source, the members it expanded to, and the
    records it was narrowed to (empty: every record of those members).

    ``record_names`` spells each record as ``ARTIFACT:LOCATOR`` with a name of the
    artifact, in the order of ``records``. ``narrowed`` says whether the request
    named members itself or took every member of the source."""

    source: str
    members: tuple[str, ...]
    records: tuple[OccurrenceRef, ...] = ()
    record_names: tuple[str, ...] = ()
    narrowed: bool = field(default=False, compare=False)

    kind = KIND_WITHDRAW

    @property
    def record_level(self) -> bool:
        """True for a request about named records. Only a record-level link can
        decide it (plan 3.2)."""
        return bool(self.records)

    def to_json(self) -> dict:
        return {
            "kind": self.kind,
            "source": self.source,
            "members": list(self.members),
            "records": list(self.record_names),
        }

    def describe(self) -> str:
        text = f"withdraw {self.source}"
        if self.record_names:
            text += ", records " + ", ".join(self.record_names)
        return text


# --- Building a withdrawal -----------------------------------------------------------


def split_record(text: str) -> tuple[str, str]:
    """Split ``ARTIFACT:LOCATOR`` at its last colon. The locator is the line of a
    JSONL file, ``L<n>`` counted from 1."""
    artifact, colon, locator = str(text).rpartition(":")
    if not colon or not artifact or not locator:
        raise UnsupportedSelector(
            f"a record is named ARTIFACT:LOCATOR, for example src/A:L3; got {text!r}", record=text
        )
    try:
        kind, _ = ids.parse_locator(locator)
    except ValueError:
        kind = ""
    if kind != "L":
        raise UnsupportedSelector(
            f"a withdrawal names records of a source file by line, L<n>; got {locator!r} in {text!r}",
            record=text,
        )
    return artifact, locator


def _resolve_record(store, record: OccurrenceRef | str) -> tuple[OccurrenceRef, str]:
    """One named record as (reference, ``name:locator``). Refuses a record the store
    does not hold."""
    if isinstance(record, OccurrenceRef):
        artifact_ref, locator = record.instance_id, record.locator
        shown = record.key
    else:
        artifact_ref, locator = split_record(record)
        shown = str(record)
    artifact = store.resolve(artifact_ref)
    if artifact is None:
        raise RecordNotInSource(f"{shown}: no artifact {artifact_ref!r} is registered", record=shown)
    ref = OccurrenceRef(artifact.instance_id, locator)
    if store.occurrence(ref) is None:
        raise RecordNotInSource(f"{shown}: the store holds no such record", record=shown)
    name = artifact.names[0] if artifact.names else artifact.instance_id
    return ref, f"{name}:{locator}"


def expand(store, request: WithdrawRequest) -> Withdrawal:
    """Expand a withdrawal against the store as it is now (plan 2.3, Decision 24).

    The members are every member registered under the source, or the named ones.
    A source with no member is refused, and so is a named member the source does
    not hold and a record that no covered member holds."""
    if not isinstance(request, WithdrawRequest):
        raise UnsupportedSelector(f"not a withdrawal: {request!r}")
    source = request.source
    if not isinstance(source, str) or not source:
        raise SourceNotRegistered("a withdrawal names a source")
    registered = store.members(source)
    if not registered:
        raise SourceNotRegistered(
            f"no member is registered under the source {source!r}; register its revisions "
            "first, a withdrawal of an unknown source answers nothing",
            source=source,
        )
    known = {m.name for m in registered}
    asked = tuple(dict.fromkeys(request.members))
    for name in asked:
        if name not in known:
            raise MemberNotRegistered(
                f"{name!r} is not a member of the source {source!r}; its members are "
                + ", ".join(sorted(known)),
                source=source,
                member=name,
            )
    wanted = set(asked) if asked else known
    members = tuple(m.name for m in registered if m.name in wanted)

    refs: list[OccurrenceRef] = []
    names: list[str] = []
    for record in request.records:
        ref, shown = _resolve_record(store, record)
        if ref in refs:
            continue
        holders = {m.name for m in store.members_covering(ref)}
        if not holders & wanted:
            raise RecordNotInSource(
                f"{shown} is not a record of {', '.join(members)}; a withdrawal of "
                f"{source!r} can only be narrowed to records of its own members",
                source=source,
                record=shown,
            )
        refs.append(ref)
        names.append(shown)
    return Withdrawal(source, members, tuple(refs), tuple(names), narrowed=bool(asked))


def withdraw(
    store,
    source: str,
    *,
    members: Iterable[str] = (),
    records: Iterable[OccurrenceRef | str] = (),
    selectors: Sequence[str] = (),
) -> Withdrawal:
    """Build and expand a withdrawal. ``records`` are references or
    ``ARTIFACT:LOCATOR`` texts. ``selectors`` are metadata selectors (``KEY=VALUE``);
    this build reads none of them, so one is enough to refuse the request. The
    switches are checked first: a switch that cannot be read refuses the request
    before its source, members, records or selectors are looked at."""
    extensions.check_switches(store.settings())
    for selector in selectors:
        raise UnsupportedSelector(
            f"the selector {selector!r} is not supported: a withdrawal names a source, "
            "members of it, or records of it",
            selector=selector,
        )
    listed = tuple(records)
    for record in listed:
        if not isinstance(record, OccurrenceRef):
            split_record(record)  # a record that cannot be read is refused before the store is asked
    return expand(store, WithdrawRequest(source, tuple(members), listed))  # type: ignore[arg-type]


# --- Requests of extensions that are off ---------------------------------------------


def ban(store, text: str) -> Withdrawal:
    """A content ban across origins (plan 2.3). An extension: refused with
    "extension off" while its switch is off. This build holds the off state only,
    so with the switch on the request is refused as not built. The switches are
    checked first, so a broken switch of another extension is named, not "extension off"."""
    extensions.check_switches(store.settings())
    extensions.require(store.settings(), extensions.CONTENT_BAN, "a content ban")
    raise Refused(
        "the content-ban extension is switched on in this store, and this build of the tool "
        "holds only its off state",
        extension=extensions.CONTENT_BAN,
    )


def flagged(store, path: str) -> Withdrawal:
    """A request that starts from flagged records (plan 2.3). An extension: refused
    with "extension off" while its switch is off. The switches are checked first."""
    extensions.check_switches(store.settings())
    extensions.require(store.settings(), extensions.FLAGGED_RECORDS, "a request from flagged records")
    raise Refused(
        "the flagged-records extension is switched on in this store, and this build of the "
        "tool holds only its off state",
        extension=extensions.FLAGGED_RECORDS,
    )


def parse(store, spec: Mapping) -> Withdrawal:
    """A request given as a mapping, for example the ``request`` block of an answer:
    ``{"kind": "withdraw", "source": "A", "members": [...], "records": [...]}``."""
    if not isinstance(spec, Mapping):
        raise UnsupportedSelector(f"a request is a mapping with a kind; got {spec!r}")
    kind = spec.get("kind", KIND_WITHDRAW)
    if kind == KIND_BAN:
        return ban(store, str(spec.get("text", "")))
    if kind == KIND_FLAGGED:
        return flagged(store, str(spec.get("path", "")))
    if kind != KIND_WITHDRAW:
        raise UnsupportedSelector(f"the request kind {kind!r} is not supported", kind=str(kind))
    unread = sorted(set(spec) - _WITHDRAW_KEYS)
    if unread:
        raise UnsupportedSelector(
            f"the selector {unread[0]!r} is not supported: a withdrawal names a source, "
            "members of it, or records of it",
            selector=unread[0],
        )
    return withdraw(
        store,
        spec.get("source", ""),
        members=spec.get("members") or (),
        records=spec.get("records") or (),
    )
