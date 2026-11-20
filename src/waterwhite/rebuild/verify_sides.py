"""The verification of a repair (plan 4.2, 4.4, Decision 6; docs/interfaces.md
sections 7.2 and 7.3, docs/rebuild.md sections 3 and 5).

``verify`` is handed the report the repair was driven by and, optionally, the sides
file that names the repaired side's and the reference side's file for each artifact.
It compares what the sides hold, checks that what the report keeps is untouched, and
puts the successors, the history and the index caches to the parts that answer for
them. It builds no coverage answer and writes nothing.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .. import capture, extensions, ids, normalise, requests
from .. import vocabulary as v
from ..errors import Refused, UsageError
from ..store import artifact_files

# The artifact actions a report holds: an artifact is kept or rewritten, or it reads the
# fourth word (its own contents are clean and the place of the copies that return is not
# stated), or it is unresolved and has no action to verify.
_ARTIFACT_ACTIONS = (*v.ARTIFACT_ACTIONS, v.UNRESOLVED)
# The actions of an artifact that has a file the two builds can be compared on. The
# fourth word is among them: the shard may have received a copy, so its bytes are the ones
# most likely to differ, and they are compared, never held to the recorded digest.
_COMPARED_ACTIONS = (v.ACTION_KEEP, v.ACTION_REWRITE, v.ACTION_CLEAN_PLACE_NOT_STATED)


def read_document(path, form: str, what: str) -> dict:
    """Read a report or a sides file. A file that cannot be read, or is not JSON, is
    a usage error; JSON that is not a document of ``form`` is refused."""
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as error:
        raise UsageError(f"the {what} cannot be read: {error.strerror}", path=str(path)) from None
    except ValueError as error:
        raise UsageError(f"the {what} is not JSON: {error}", path=str(path)) from None
    if not isinstance(document, dict) or document.get("form") != form:
        raise Refused(f"the {what} is not a document of the form {form}", path=str(path))
    return document


def _check_sides_paths(document: Mapping) -> None:
    """Every value of the two sides is a path, so text (docs/interfaces.md 7.2). A value
    that is not is refused, once, naming the side and the artifact, before any reader takes
    it for a path. It is checked where ``verify`` reads the sides, so a parsed document
    handed to the function is held to it as a file is. A side that is neither null nor an
    object is refused where the side is read ("the sides file's repaired is not an object")."""
    for which in ("repaired", "reference"):
        side = document.get(which)
        if not isinstance(side, Mapping):
            continue
        for artifact, named in side.items():
            _clean(artifact, f"a name on the {which} side of the sides file", side=which)
            if isinstance(named, str):
                _clean(named, f"the path of {artifact} on its {which} side", side=which, artifact=artifact)
            if not isinstance(named, str):
                raise Refused(
                    f"the sides file names {artifact} on its {which} side by something that is "
                    f"not a path (text): {json.dumps(named)}",
                    side=which, artifact=artifact)


# --- The report, as far as verify reads it ---------------------------------------------


def _need(condition: bool, detail: str, **context: object) -> None:
    if not condition:
        raise Refused(detail, **context)


def _carries(text: str) -> bool:
    """Whether a text can be a name or a path and be bound to a query: no NUL (a path lookup
    ends in ValueError on one) and no unpaired surrogate (it cannot be encoded)."""
    if "\x00" in text:
        return False
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _clean(value: object, what: str, **context: object) -> None:
    """Refuse a text that holds a character a name or a path cannot hold."""
    if isinstance(value, str):
        _need(_carries(value), f"{what} holds a NUL or an unpaired surrogate, which no name or path holds",
              **context)


def report_artifacts(report: Mapping) -> list[dict]:
    """The artifact entries of a report, checked for the keys ``verify`` reads:
    ``artifact``, ``instance``, ``kind``, ``action`` and ``successor`` (null allowed)."""
    _need(report.get("form") == v.FORM_REPORT, f"the report is not a document of the form {v.FORM_REPORT}")
    entries = report.get("artifacts")
    _need(isinstance(entries, list), "the report holds no list of artifacts")
    named: set[str] = set()
    instances: set[str] = set()
    for entry in entries:
        _need(isinstance(entry, Mapping), "an entry of the report's artifacts is not an object")
        for key in ("artifact", "instance", "kind", "action", "successor"):
            _need(key in entry, f"an entry of the report's artifacts lacks {key}")
        for key in ("artifact", "instance", "kind", "action"):
            _need(isinstance(entry[key], str), f"an entry of the report's artifacts gives {key} "
                  f"as something that is not text: {json.dumps(entry[key])}")
        _need(entry["successor"] is None or isinstance(entry["successor"], str),
              f"the report gives {entry['artifact']} a successor that is not text: "
              f"{json.dumps(entry['successor'])}", artifact=entry["artifact"])
        for key in ("artifact", "instance", "kind", "action", "successor"):
            _clean(entry[key], f"the {key} of an entry of the report's artifacts")
        _need(entry["action"] in _ARTIFACT_ACTIONS,
              f"the report gives {entry['artifact']} the action {entry['action']!r}")
        # An entry is one entry for one artifact: the gate and the comparison are keyed by the
        # name, so a name named twice lets the later entry stand in for the earlier.
        _need(entry["artifact"] not in named,
              f"the report names {entry['artifact']} twice; an artifact has one entry",
              artifact=entry["artifact"])
        _need(entry["instance"] not in instances,
              f"the report gives the instance of {entry['artifact']} to two entries; an artifact "
              "has one entry, whatever it is called",
              artifact=entry["artifact"])
        named.add(entry["artifact"])
        instances.add(entry["instance"])
    return [dict(entry) for entry in entries]


def instance_held(store, instance: object) -> bool:
    """Whether the store holds an artifact with exactly this instance id."""
    if not isinstance(instance, str):
        return False
    found = store.artifact(instance)
    return found is not None and found.instance_id == instance


def require_instances_held(store, artifacts: list[dict]) -> None:
    """A report is certified only against a store that holds every artifact it names,
    each by its instance id (plan 4.2, 4.4). A store that lacks one is not the store
    the report was made for, or a store that has lost what the report is about; it
    is refused, whichever file it is."""
    for entry in artifacts:
        _need(instance_held(store, entry["instance"]),
              f"the store does not hold the instance the report names for {entry['artifact']}; "
              "a repair is verified against the store the report was made for, or a copy of it",
              artifact=entry["artifact"])
        # The kind the report states is the kind the store holds that instance as: the
        # successor's kind is held against it, and a stated kind the store does not
        # hold would let a successor of another kind stand for the artifact.
        held = store.artifact(entry["instance"]).kind
        _need(entry["kind"] == held,
              f"the report names {entry['artifact']} as a {entry['kind']}, and the store "
              f"holds that instance as a {held}",
              artifact=entry["artifact"])
        # The name is the name of that instance. A rewritten artifact is the exception: its
        # own check is the ``history`` entry, which says "instance changed" for a name the
        # store now holds as another instance, and the gate does not hold on it. Every other
        # entry is compared only through its name, so a name of another artifact would let
        # that artifact's file stand for this one.
        if entry["action"] != v.ACTION_REWRITE:
            # By name: the store also resolves an instance id as a name, and an entry that
            # names its artifact by the instance id is a second name for it. An artifact with
            # no name is shown by its instance id (``rebuild.artifacts.shown``).
            held_artifact = store.artifact(entry["instance"])
            _need(entry["artifact"] in held_artifact.names
                  or (not held_artifact.names and entry["artifact"] == entry["instance"]),
                  f"the report names {entry['artifact']} with the instance of another artifact, or "
                  "by something that is not a name the store holds it under; an entry's name is "
                  "the name the store holds its instance under",
                  artifact=entry["artifact"])


def report_unresolved(report: Mapping) -> int:
    counts = report.get("counts")
    records = counts.get("records") if isinstance(counts, Mapping) else None
    unresolved = records.get("unresolved") if isinstance(records, Mapping) else None
    _need(isinstance(unresolved, int) and not isinstance(unresolved, bool),
          "the report holds no count of unresolved records")
    return unresolved


def report_request(store, report: Mapping):
    """The request of the report, put again to the store; refused as ``query`` refuses it."""
    held = report.get("request")
    _need(isinstance(held, Mapping) and isinstance(held.get("source"), str),
          "the report holds no request that names a source")
    _need(held.get("kind", requests.KIND_WITHDRAW) == requests.KIND_WITHDRAW,
          "the report is for a request this build does not answer")
    _clean(held["source"], "the source of the report's request")
    for key in ("members", "records"):
        listed = held.get(key)
        _need(listed is None or (isinstance(listed, list) and all(isinstance(n, str) for n in listed)),
              f"the report's request gives {key} as something that is not a list of text: "
              f"{json.dumps(listed)}")
        for member in listed or ():
            _clean(member, f"a {key[:-1]} of the report's request")
    return requests.withdraw(
        store, held["source"], members=held.get("members") or (), records=held.get("records") or ())


# --- The files a sides document names ---------------------------------------------------


def side_path(base, named: str) -> Path:
    path = Path(named)
    if path.is_absolute() or base is None:
        return path
    return Path(base) / path


def _present(kind: str, path: Path) -> bool:
    if kind == v.KIND_SHARD:
        bin_path, idx_path = artifact_files.shard_paths(path)
        return bin_path.is_file() and idx_path.is_file()
    return path.is_file()


def require_file(kind: str, path: Path) -> None:
    """A file the sides file names that is not there is a refusal of the document."""
    if not _present(kind, path):
        raise Refused(f"a file the sides file names is not there: {path}", path=str(path))


def file_digest_of(kind: str, path: Path) -> str:
    """The content digest of the file a side names: a shard's pair of files, or a
    JSONL file."""
    require_file(kind, path)
    if kind == v.KIND_SHARD:
        return artifact_files.shard_digest_only(path)
    return ids.file_digest(path)


def _side(sides: Mapping | None, which: str) -> dict:
    held = None if sides is None else sides.get(which)
    _need(held is None or isinstance(held, Mapping), f"the sides file's {which} is not an object")
    return dict(held or {})


# --- The multiset of (fingerprint, multiplicity) ------------------------------------------


def _read_multiset(store, path: Path, text_field: str) -> tuple[Counter, dict, int]:
    """Stream one JSONL file once: the multiset of the lines' fingerprints (a line
    that holds no text has none), where each fingerprint's lines are, and the number
    of lines. The fingerprint is the tool's own, of the text after the store's
    normaliser, as the lines of a registered file have it."""
    function = normalise.get(store.normaliser)
    counts: Counter = Counter()
    where: dict = {}
    number = 0
    for raw in ids.read_lines(path):
        number += 1
        text = capture.line_text(raw, text_field)
        fingerprint = None
        if text is not None:
            normalised = function(text)
            if normalised:
                fingerprint = ids.fingerprint(normalised)
        counts[fingerprint] += 1
        where.setdefault(fingerprint, []).append(number)
    return counts, where, number


def _fingerprint_text(fingerprint: bytes | None) -> str | None:
    return None if fingerprint is None else fingerprint.hex()


def _compare_files(store, name: str, repaired: Path, reference: Path, text_field: str):
    """The release entry and the differences of one JSONL file named on both sides."""
    held, where, lines = _read_multiset(store, repaired, text_field)
    wanted, _, reference_lines = _read_multiset(store, reference, text_field)
    differences = []
    for fingerprint in held.keys() | wanted.keys():
        if held[fingerprint] == wanted[fingerprint]:
            continue
        locators = [ids.line_locator(n) for n in where.get(fingerprint, [])]
        differences.append({
            "artifact": name, "fingerprint": _fingerprint_text(fingerprint),
            "repaired": held[fingerprint], "reference": wanted[fingerprint], "locators": locators,
        })
    differences.sort(key=lambda d: (d["artifact"], min(
        (int(loc[1:]) for loc in d["locators"]), default=1 << 62), d["fingerprint"] or ""))
    entry = {"artifact": name, "repaired": lines, "reference": reference_lines,
             "equal": not differences}
    return entry, differences


NOTE_NEITHER_SIDE = "named on neither side, not compared"
NOTE_INSTANCE_NOT_HELD = "the store does not hold the instance the report names, so nothing is compared"
NOTE_EMPTIED = {
    v.KIND_JSONL: "the report leaves the file no row, so neither side writes it; nothing to compare",
    v.KIND_SHARD: "the report leaves the shard no document, so neither side writes it; nothing to compare",
}


def _one_sided(name: str, which: str, note: str | None = None) -> dict:
    return {"artifact": name, "fingerprint": None, "repaired": None, "reference": None,
            "locators": [], "note": note or f"named on the {which} side only, not compared"}


def left_with_no_record(store, entry: Mapping) -> bool:
    """Whether the tool's own record shows that the repair leaves an artifact of the
    report no record: a rewritten file whose entry says no row is in its successor, or
    a rewritten shard whose entry says no document is. The report's counts are the
    record, never the replica's word; a report that gives no count shows nothing, and
    neither does a store that holds a successor of the artifact with records in it."""
    if entry["action"] != v.ACTION_REWRITE or entry["kind"] not in NOTE_EMPTIED:
        return False
    count = entry.get("rows_in_successor" if entry["kind"] == v.KIND_JSONL else "documents_in_successor")
    if count != 0 or isinstance(count, bool):
        return False
    held = store.resolve(entry.get("successor") or f"{v.SUCCESSOR_PREFIX}{entry['artifact']}")
    if held is None:
        return True
    if held.kind == v.KIND_JSONL:
        return store.occurrence_count(held.instance_id) == 0
    binding = store.binding(held.instance_id)
    return binding is not None and binding.declared_count == 0


# --- The shards each build wrote ----------------------------------------------------------------

NOTE_NOT_IN_REPORT_BOTH = ("both builds wrote the shard and the report does not name it, "
                           "so it is not compared")
NOTE_ONLY_REPAIRED = "only the repaired build wrote the shard, and the report does not name it"
NOTE_ONLY_REFERENCE = "only the reference build wrote the shard, and the report does not name it"
NOTE_LANDING_NOT_BOTH = ("both builds wrote the shard, the report names it under a layout's landing, "
                         "and the sides file does not name it on both sides, so it is not compared")
NOTE_LANDING_ONLY_REPAIRED = "only the repaired build wrote the shard, which the report names under a layout's landing"
NOTE_LANDING_ONLY_REFERENCE = "only the reference build wrote the shard, which the report names under a layout's landing"


def report_landing(report: Mapping, artifacts: list[dict]) -> list[str]:
    """The shards the report names under ``groups.placements[NAME].landing`` that it has no
    artifact entry for (docs/interfaces.md 7.1), in the report's order, each once. The
    report names them, so a comparison of both builds reads them as it reads a shard of
    the report (D-148)."""
    held = {entry["artifact"] for entry in artifacts}
    groups = report.get("groups")
    _need(groups is None or isinstance(groups, Mapping), "the report's groups is not an object")
    placements = (groups or {}).get("placements")
    _need(placements is None or isinstance(placements, Mapping),
          "the report's groups.placements is not an object")
    names: list[str] = []
    for section in (placements or {}).values():
        landing = section.get("landing") if isinstance(section, Mapping) else None
        _need(landing is None or isinstance(landing, list),
              "a placement of the report holds a landing that is not a list")
        for item in landing or []:
            shard = item.get("shard") if isinstance(item, Mapping) else None
            if isinstance(shard, str) and shard and shard not in held and shard not in names:
                names.append(shard)
    return names


def _group_prefix(name: str, named: str | None) -> str:
    """The layout prefix of a shard's store name, ending in "/", or "" when it has none:
    the name without the file name the sides file gives its files, and failing that the
    folder part of the name."""
    if named is not None:
        stem = Path(named).name
        if name.endswith("/" + stem):
            return name[: len(name) - len(stem)]
    return name[: name.rfind("/") + 1]


def _shard_stems(folder: Path) -> list[str]:
    """The shards a folder holds whole: a stem with its ``.bin`` and its ``.idx``."""
    try:
        held = {entry.name for entry in folder.iterdir() if entry.is_file()}
    except OSError:
        return []
    return sorted(name[: -len(".bin")] for name in held
                  if name.endswith(".bin") and name[: -len(".bin")] + ".idx" in held)


def shards_each_build_wrote(store, artifacts: list[dict], repaired_side: Mapping,
                            reference_side: Mapping, base,
                            landing: Sequence[str] = ()) -> tuple[list[str], list[str], list[str]]:
    """(both, repaired only, reference only): the shards of each layout that the report
    has no artifact entry for and that the builds wrote, read from the store and from the
    disk and never from the sides file's list of names (D-148).

    ``landing`` is the shards the report names under a layout's landing. They are shards
    the report names, so they share a layout with the entries like any shard; they stay in
    the three lists when the builds wrote them, and the caller settles which of them the
    sides file already compares or names (``verify``).

    A layout is the names that share a prefix: the prefix of each shard the report names
    (D-148 gives the rule). The repaired build registers every shard it writes as
    ``successor of <name>``, so the repaired side's shards are those of the store; the
    reference build has no store, so its shards are the pairs of files in the folder of
    the reference files the sides file names for the layout. A layout whose reference
    folder the sides file does not give is left out: what the reference build wrote there
    is not known, and nothing is said of it."""
    named = {entry["artifact"] for entry in artifacts}
    groups: dict[str, set[Path]] = {}
    shard_names = [entry["artifact"] for entry in artifacts if entry["kind"] == v.KIND_SHARD]
    for name in [*shard_names, *landing]:
        given = reference_side.get(name)
        prefix = _group_prefix(name, given if given is not None else repaired_side.get(name))
        if not prefix:
            continue
        folders = groups.setdefault(prefix, set())
        if given is not None:
            folders.add(side_path(base, given).parent)
    repaired_names = [held for artifact in store.artifacts(v.KIND_SHARD) for held in artifact.names
                      if held.startswith(v.SUCCESSOR_PREFIX)]
    both, repaired_only, reference_only = [], [], []
    for prefix, folders in sorted(groups.items()):
        if not folders:
            continue
        repaired = {held[len(v.SUCCESSOR_PREFIX):] for held in repaired_names
                    if held.startswith(v.SUCCESSOR_PREFIX + prefix)}
        reference = {prefix + stem for folder in folders for stem in _shard_stems(folder)}
        both += sorted((repaired & reference) - named)
        repaired_only += sorted((repaired - reference) - named)
        reference_only += sorted((reference - repaired) - named)
    return both, repaired_only, reference_only


# --- verify --------------------------------------------------------------------------------


def verify(store, report: Mapping, *, sides: Mapping | None = None, base=None,
           mode: str = v.BUILD_UNSEEDED, text_field: str = "text") -> dict:
    """The document of docs/interfaces.md 7.3. ``report`` and ``sides`` are parsed
    documents; ``base`` is the folder the relative paths of ``sides`` are taken from."""
    from . import restart_exact, verify_caches, verify_successors

    extensions.check_switches(store.settings())  # before the mode, the report's shape or its request kind is looked at (D-CORE-6)
    if mode not in v.BUILD_MODES:
        raise UsageError(f"--mode is one of {', '.join(v.BUILD_MODES)}; got {mode!r}")
    if sides is not None:
        _need(isinstance(sides, Mapping), "the sides file is not a document")
        _check_sides_paths(sides)
    artifacts = report_artifacts(report)
    unresolved = report_unresolved(report)
    request = report_request(store, report)
    require_instances_held(store, artifacts)
    kinds = {entry["artifact"]: entry["kind"] for entry in artifacts}
    landing = report_landing(report, artifacts)
    for name in landing:
        kinds[name] = v.KIND_SHARD  # a shard the report names under a layout's landing

    repaired_side, reference_side = _side(sides, "repaired"), _side(sides, "reference")
    releases: list[dict] = []
    differences: list[dict] = []
    shards: list[dict] = []
    compared_shards = 0
    for name in [*repaired_side, *(n for n in reference_side if n not in repaired_side)]:
        kind = kinds.get(name)
        which = "repaired" if name in repaired_side else "reference"
        if kind is None:
            differences.append(_one_sided(name, which, "not an artifact of the report, not compared"))
            continue
        if kind not in (v.KIND_JSONL, v.KIND_SHARD):
            continue  # a mixture or a checkpoint has no file to compare
        if name not in repaired_side or name not in reference_side:
            differences.append(_one_sided(name, which))
            continue
        repaired = side_path(base, repaired_side[name])
        reference = side_path(base, reference_side[name])
        for path in (repaired, reference):
            require_file(kind, path)  # a file that is not there is refused before anything is read
        if kind == v.KIND_JSONL:
            entry, found = _compare_files(store, name, repaired, reference, text_field)
            releases.append(entry)
            differences.extend(found)
        else:
            compared_shards += 1
            held, wanted = file_digest_of(kind, repaired), file_digest_of(kind, reference)
            shards.append({"artifact": name, "repaired": held, "reference": wanted,
                           "equal": held == wanted})

    # A file the report keeps or rewrites that the sides do not name on both sides is a
    # file the comparison lacks: it is named, and its comparison never reads "equal".
    # The exception is a rewritten file that neither side names and that the tool's own
    # record shows is left no record: there is nothing to compare, and the entry says so.
    lacking = {kind: [] for kind in (v.KIND_JSONL, v.KIND_SHARD)}
    for entry in artifacts:
        name = entry["artifact"]
        if entry["action"] not in _COMPARED_ACTIONS or entry["kind"] not in lacking:
            continue
        if name in repaired_side and name in reference_side:
            continue
        on_neither = name not in repaired_side and name not in reference_side
        emptied = on_neither and left_with_no_record(store, entry)
        if not emptied:
            lacking[entry["kind"]].append(name)
        if sides is not None and on_neither:
            differences.append(_one_sided(
                name, "neither", NOTE_EMPTIED[entry["kind"]] if emptied else NOTE_NEITHER_SIDE))

    # A shard the builds wrote that the report does not name (D-148). With no sides file
    # nothing is said of it, as for every file the sides leave out (D-82).
    # A shard the report names under a layout's landing is a shard the report names: the
    # loop above compared it where the sides name it on both sides and named it where they
    # name it on one (that one-sided line is the only one such a name gets). One that both
    # builds wrote and the sides do not name on both sides is a file the comparison lacks,
    # as a shard of the report is; it is named once, and a landing shard only one build
    # wrote still reads as a shard one build alone wrote.
    uncompared = []
    if sides is not None:
        both, repaired_only, reference_only = shards_each_build_wrote(
            store, artifacts, repaired_side, reference_side, base, landing)
        landed = set(landing)
        both_named = {name for name in landed if name in repaired_side and name in reference_side}
        one_named = {name for name in landed if (name in repaired_side) != (name in reference_side)}
        uncompared = [name for name in both if name not in both_named]
        differences.extend(
            _one_sided(name, "neither", NOTE_LANDING_NOT_BOTH if name in landed else NOTE_NOT_IN_REPORT_BOTH)
            for name in uncompared if name not in one_named)
        differences.extend(
            _one_sided(name, "repaired", NOTE_LANDING_ONLY_REPAIRED if name in landed else NOTE_ONLY_REPAIRED)
            for name in repaired_only if name not in one_named)
        differences.extend(
            _one_sided(name, "reference", NOTE_LANDING_ONLY_REFERENCE if name in landed else NOTE_ONLY_REFERENCE)
            for name in reference_only if name not in one_named)
        uncompared_one_sided = bool(repaired_only or reference_only)
    else:
        uncompared_one_sided = False

    if unresolved > 0:
        multisets = v.SIDES_NOT_CLAIMED
    elif not releases:
        multisets = v.SIDES_NOT_COMPARED
    elif not all(r["equal"] for r in releases):
        multisets = v.SIDES_DIFFER
    else:
        multisets = v.SIDES_NOT_COMPARED if lacking[v.KIND_JSONL] or uncompared else v.SIDES_EQUAL
    if mode != v.BUILD_SEEDED or not compared_shards:
        per_shard = v.SIDES_NOT_COMPARED
    elif uncompared_one_sided or not all(s["equal"] for s in shards):
        per_shard = v.SIDES_DIFFER
    else:
        per_shard = v.SIDES_NOT_COMPARED if lacking[v.KIND_SHARD] or uncompared else v.SIDES_EQUAL

    kept = []
    for entry in artifacts:
        if entry["action"] != v.ACTION_KEEP:
            continue
        kept.append(_kept_entry(store, entry, repaired_side, base))

    held_back: list = []
    successors, history = verify_successors.check_successors(store, report, request, sides, base, held_back)
    caches = verify_caches.check_caches(store, report)
    restarts = restart_exact.exact_rerun_restarts(store, report, caches)
    made = {
        "form": v.FORM_VERIFICATION,
        "request": request.to_json(),
        "mode": mode,
        "unresolved_records": unresolved,
        "multisets": multisets,
        "differing_records": sum(abs(d["repaired"] - d["reference"])
                                 for d in differences if d["repaired"] is not None),
        "differences": differences,
        "releases": releases,
        "per_shard_digests": per_shard,
        "shards": shards,
        "kept": kept,
        "successors": successors,
        "history": history,
        "caches": caches,
        "restarts": restarts,
    }
    if held_back:
        made["withheld"] = [w.to_json() for w in held_back]
    return made


def _kept_entry(store, entry: dict, repaired_side: Mapping, base) -> dict:
    """Whether the file the repaired side names for a kept artifact is the file the
    artifact was registered as (plan 4.4). Null where there is nothing to compare.

    The file is compared with the content digest of the report's own instance and with
    nothing else: a store that does not hold that instance gives null for ``identical``
    and ``recorded``, and the entry says why in ``reason``, whatever the artifact's name
    resolves to in that store now."""
    name = entry["artifact"]
    held = store.artifact(entry["instance"]) if instance_held(store, entry["instance"]) else None
    recorded = None if held is None else held.content_digest
    found = None
    if entry["kind"] in (v.KIND_JSONL, v.KIND_SHARD) and name in repaired_side:
        found = file_digest_of(entry["kind"], side_path(base, repaired_side[name]))
    identical = None if found is None or recorded is None else found == recorded
    made = {"artifact": name, "identical": identical, "recorded": recorded, "found": found}
    if held is None:
        made["reason"] = NOTE_INSTANCE_NOT_HELD
    return made


__all__ = ["file_digest_of", "read_document", "report_artifacts", "report_request",
           "report_unresolved", "side_path", "verify"]
