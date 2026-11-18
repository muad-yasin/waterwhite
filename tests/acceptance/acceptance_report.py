"""Reading the rebuild report and the verification of a repair, as pure functions
(docs/interfaces.md sections 7.1 and 7.3).

Every function takes a parsed document and the expected fields of a row, and
returns the fields that differ. None runs a command and none imports a module of
the tool (plan W5): the words compared with are the table's own, the form read is
the one the documents fix. The commands are run by ``acceptance_world.World``; the
tests of this folder put these functions to documents written by hand, one that
agrees and one that differs, so that each can be seen to fail
(``test_check07.py``, ``test_check09.py``).

What a document must be: a JSON object whose ``form`` is the format word. A document
of another form is an error of the test (``TableError``), because no row can be
answered from it. A key that is missing or null is never read as true and never as
a pass: it is a difference, with ``None`` as what was got.

Traps of the form that these functions keep to (docs/interfaces.md 7.1):

- an artifact that is kept lists no records, so a record of it is kept through its
  artifact's entry; an artifact that is rewritten lists every record it holds;
- a promote entry has the release as ``artifact`` and a line of another file as
  ``record`` (``src/B:L1``);
- a record is named ``artifact:locator``; the locator of a line is ``L<n>``, of a
  document ``D<n>``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from table import Mismatch, TableError

REPORT_FORM = "waterwhite.report/1"
VERIFICATION_FORM = "waterwhite.verification/1"
SIDES_FORM = "waterwhite.sides/1"

# The fields of a record entry that a row of the table names (table.REPORT_KEYS holds
# the same words, with ``counts``, which is an artifact's).
RECORD_FIELDS = ("action", "new_parent", "withdrawn_parent", "regenerate_with",
                 "missing_link", "review_scope")
# What a row names of an artifact entry of the report.
ARTIFACT_FIELDS = ("action", "counts", "records_removed", "rows_in_successor",
                   "documents_in_successor")

Resolve = Callable[[str], str]  # a handle of the table -> the record's name in a report


def document(value: object, form: str, what: str) -> Mapping:
    """``value`` as a document of ``form``, or ``TableError``."""
    if not isinstance(value, Mapping):
        raise TableError(f"the {what} is a JSON object, the command printed {value!r}")
    if value.get("form") != form:
        raise TableError(f"the {what} has the form {value.get('form')!r}, not {form!r} "
                         "(docs/interfaces.md section 7)")
    return value


def _entries(report: Mapping, key: str) -> list[Mapping]:
    found = report.get(key)
    return [e for e in found if isinstance(e, Mapping)] if isinstance(found, list) else []


def artifact_entry(report: Mapping, name: str) -> Mapping | None:
    """The entry of ``artifacts`` for an artifact name, or None."""
    for entry in _entries(report, "artifacts"):
        if entry.get("artifact") == name:
            return entry
    return None


def record_entries(report: Mapping, artifact: str, record: str) -> list[Mapping]:
    """The entries of ``records`` that say something of one record of one artifact."""
    return [e for e in _entries(report, "records")
            if e.get("artifact") == artifact and e.get("record") == record]


def effective_action(report: Mapping, artifact: str, record: str) -> str | None:
    """The action a report gives one record: its own entry, or, for a record of an
    artifact whose action is keep (which lists none), keep. None where the report
    says nothing of the record. An artifact with no entry gives nothing."""
    found = record_entries(report, artifact, record)
    if found:
        return found[0].get("action")
    entry = artifact_entry(report, artifact)
    if entry is not None and entry.get("action") == "keep":
        return "keep"
    return None


def compare_record_rows(expect: Mapping, report: Mapping, artifact: str,
                        records: Sequence[str], resolve: Resolve) -> list[Mismatch]:
    """The rows that name one or several records of an artifact (Check 7's report rows,
    Check 6's import row). ``records`` are the names the report would give them,
    ``resolve`` turns a handle of the table into such a name (a parent of ``X`` is
    ``src/B:L1``). Every record must have the expected action; ``count`` is the
    number of records the report says something of."""
    found: list[Mismatch] = []
    actions = {name: effective_action(report, artifact, name) for name in records}
    if "action" in expect:
        wrong = {name: got for name, got in actions.items() if got != expect["action"]}
        if wrong:
            found.append(Mismatch("action", expect["action"],
                                  next(iter(wrong.values())) if len(records) == 1 else wrong))
    if "count" in expect:
        listed = sum(1 for got in actions.values() if got is not None)
        if listed != expect["count"]:
            found.append(Mismatch("count", expect["count"], listed))
    for key in ("new_parent", "withdrawn_parent"):
        if key in expect:
            wanted = resolve(expect[key])
            for name in records:
                entries = record_entries(report, artifact, name)
                got = entries[0].get(key) if entries else None
                if got != wanted:
                    found.append(Mismatch(key, wanted, got))
    for key in ("regenerate_with", "missing_link"):
        if key in expect:
            for name in records:
                entries = record_entries(report, artifact, name)
                got = entries[0].get(key) if entries else None
                if got != expect[key]:
                    found.append(Mismatch(key, expect[key], got))
    if "review_scope" in expect:
        for name in records:
            entries = record_entries(report, artifact, name)
            got = entries[0].get("review_scope") if entries else None
            want = expect["review_scope"]
            shown = ({k: got.get(k) for k in want} if isinstance(got, Mapping) else got)
            if shown != want:
                found.append(Mismatch("review_scope", want, shown))
    unknown = set(expect) - set(RECORD_FIELDS) - {"count"}
    if unknown:
        raise TableError(f"a record row of the report names {sorted(unknown)}")
    return found


def _bare(name: str, artifact: str) -> str:
    """A removed record as its locator: ``R1:L2`` and ``L2`` both give ``L2``."""
    return name[len(artifact) + 1:] if name.startswith(f"{artifact}:") else name


def compare_artifact_fields(expect: Mapping, report: Mapping, artifact: str,
                            resolve: Resolve | None = None) -> list[Mismatch]:
    """The fields of one artifact entry a row names. A missing entry is one difference.
    ``records_removed`` is a number, or a list of handles that ``resolve`` turns into
    the locators of the records removed (the entry's ``removed``)."""
    entry = artifact_entry(report, artifact)
    if entry is None:
        return [Mismatch(f"artifact entry of {artifact}", "an entry", None)]
    found: list[Mismatch] = []
    for key in ("action", "rows_in_successor", "documents_in_successor"):
        if key in expect and entry.get(key) != expect[key]:
            found.append(Mismatch(key, expect[key], entry.get(key)))
    if "counts" in expect:
        got = entry.get("counts")
        got = got if isinstance(got, Mapping) else {}
        found += [Mismatch(f"counts.{word}", want, got.get(word))
                  for word, want in expect["counts"].items() if got.get(word) != want]
    if "records_removed" in expect:
        want = expect["records_removed"]
        if isinstance(want, list):
            if resolve is None:
                raise TableError("a list of records removed needs a way to name them")
            names = {_bare(resolve(handle), artifact) for handle in want}
            removed = entry.get("removed")
            listed = ({_bare(str(n), artifact) for n in removed}
                      if isinstance(removed, list) else None)
            if listed != names:
                found.append(Mismatch("records_removed", sorted(names),
                                      None if listed is None else sorted(listed)))
            if entry.get("records_removed") != len(names):
                found.append(Mismatch("records_removed (number)", len(names),
                                      entry.get("records_removed")))
        elif entry.get("records_removed") != want:
            found.append(Mismatch("records_removed", want, entry.get("records_removed")))
    unknown = set(expect) - set(ARTIFACT_FIELDS)
    if unknown:
        raise TableError(f"an artifact row of the report names {sorted(unknown)}")
    return found


def compare_mixtures(expect: Mapping, report: Mapping) -> list[Mismatch]:
    """``needs_successor`` and ``kept`` of the report's ``mixtures``, each as a set."""
    held = report.get("mixtures")
    held = held if isinstance(held, Mapping) else {}
    found = []
    for key in ("needs_successor", "kept"):
        if key in expect:
            got = held.get(key)
            if not isinstance(got, list) or set(got) != set(expect[key]):
                found.append(Mismatch(key, sorted(expect[key]), got))
    return found


def compare_group(expect: Mapping, report: Mapping, kind: str, name: str) -> list[Mismatch]:
    """One group of ``groups`` (``by_run`` or ``by_layout``): its ``counts`` and its
    ``keep`` and ``rewrite`` lists as sets."""
    groups = report.get("groups")
    held = groups.get(kind) if isinstance(groups, Mapping) else None
    group = held.get(name) if isinstance(held, Mapping) else None
    if not isinstance(group, Mapping):
        return [Mismatch(f"groups.{kind}.{name}", "a group", group)]
    found = []
    if "counts" in expect:
        got = group.get("counts") if isinstance(group.get("counts"), Mapping) else {}
        found += [Mismatch(f"counts.{word}", want, got.get(word))
                  for word, want in expect["counts"].items() if got.get(word) != want]
    for key in ("keep", "rewrite"):
        if key in expect:
            got = group.get(key)
            if not isinstance(got, list) or set(got) != set(expect[key]):
                found.append(Mismatch(key, sorted(expect[key]), got))
    return found


def restart_checkpoint(report: Mapping, run: str) -> str | None:
    """``exposure_free_restart`` of a run: the ``checkpoint`` of its ``restarts`` entry,
    where the entry says ``proven``. A run with no entry, or an entry that proves none,
    gives None."""
    for entry in _entries(report, "restarts"):
        if entry.get("run") == run and entry.get("proven") is True:
            return entry.get("checkpoint")
    return None


# --- The verification ---------------------------------------------------------------------


def verification_entry(verification: Mapping, key: str, field: str, value: str) -> Mapping | None:
    for entry in _entries(verification, key):
        if entry.get(field) == value:
            return entry
    return None


def kept_identical(verification: Mapping, artifact: str) -> bool | None:
    """``identical`` of the ``kept`` entry of an artifact: True only where it is True;
    None where there is no entry or the tool says nothing."""
    entry = verification_entry(verification, "kept", "artifact", artifact)
    return entry.get("identical") if entry is not None else None


def compare_kept(artifacts: Iterable[str], verification: Mapping, name: str) -> list[Mismatch]:
    """Every artifact named must have a ``kept`` entry whose ``identical`` is True. A
    null or a missing entry is a difference (the repair was not compared)."""
    return [Mismatch(f"{name}[{artifact}]", True, kept_identical(verification, artifact))
            for artifact in sorted(artifacts) if kept_identical(verification, artifact) is not True]


def cache_fields(verification: Mapping, segment: str) -> dict:
    """``stale``, ``rebuilt`` and ``rebuilt_from`` of a segment's ``caches`` entry; an
    entry that is not there gives an empty dict, so every expected field is a difference."""
    entry = verification_entry(verification, "caches", "segment", segment)
    return {key: entry.get(key) for key in ("stale", "rebuilt", "rebuilt_from")} if entry else {}


def restart_fields(verification: Mapping, run: str) -> dict:
    """The exact-rerun restart of a run, in the words of Check 9's rows. Its ``step`` is
    ``exact_rerun_restart_step`` where the comparison was made. Under the fallback F7
    (no comparison possible) the row's own sentence is built from the entry's step."""
    for entry in _entries(verification, "restarts"):
        if entry.get("run") == run:
            got = {"step_comparison": entry.get("step_comparison")}
            if entry.get("step_comparison") == "made":
                got["exact_rerun_restart_step"] = entry.get("step")
            elif entry.get("step_comparison") == "not possible" and isinstance(
                    entry.get("step"), int):
                got["exact_rerun_restart"] = (
                    f"the phase start, step {entry['step']}, stated as an assumption")
            return got
    return {}
