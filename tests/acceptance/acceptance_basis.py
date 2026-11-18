"""What an answer must carry, read from one answer's two printed forms.

``problems`` is a pure function: given the JSON document of one answer, the text the
same command printed with ``--format text``, the row's expected fields and the segment
that is the checkpoint's own, it lists what the answer lacks. Nothing here starts the
tool. The rules are those of docs/interfaces.md section 6 and docs/coverage.md
section 2.
"""

from __future__ import annotations

from collections.abc import Mapping

# docs/interfaces.md section 6: every answer has every key, in this order.
ANSWER_KEYS = ("artifact", "instance", "kind", "request", "answer", "reason", "names",
               "names_detail", "failing", "candidates", "chain", "basis", "notes",
               "untraced_steps", "first_exposure", "level", "scope")
# docs/interfaces.md section 6: the keys of the scope of a "does not contain".
SCOPE_KEYS = ("source", "members", "records", "runs", "steps", "generation_edges",
              "harness_logs", "walked", "files_not_checked")
LEVELS = ("consumed", "inherited")


def _pairs(value) -> list:
    return [list(item) for item in value]


def _exposure_line(exposure: Mapping, level: str | None) -> str:
    return (f"  first exposure: {exposure['segment']} step {exposure['step']}"
            + (f" ({level})" if level else ""))


def _untraced_line(untraced) -> str:
    return "  untraced steps: " + ", ".join(f"{segment} step {step}" for segment, step in untraced)


def _notes(document: Mapping, lines: list[str], expect: Mapping, out: list[str]) -> None:
    notes = document["notes"]
    if "notes" in expect and sorted(notes) != sorted(expect["notes"]):
        out.append(f"notes: {notes} are not the expected {expect['notes']}")
    for note in notes:
        if f"  note: {note}" not in lines:
            out.append(f"text: the note {note!r} is not printed")


def _contains(document: Mapping, lines: list[str], expect: Mapping,
              own_segment: str | None, out: list[str]) -> None:
    chain = document["chain"]
    if not chain:
        out.append("chain: a contains comes with its chain of evidence")
    elif "  chain: " + " <- ".join(hop["hop"] for hop in chain) not in lines:
        out.append("text: the chain is not printed")
    if document["scope"] is not None:
        out.append("scope: a contains has no scope, the key is null")
    _notes(document, lines, expect, out)
    untraced = document["untraced_steps"]
    if "untraced_steps" in expect and _pairs(untraced) != _pairs(expect["untraced_steps"]):
        out.append(f"untraced steps: {untraced} are not the expected {expect['untraced_steps']}")
    if untraced and _untraced_line(untraced) not in lines:
        out.append("text: the untraced steps are not printed")
    if not untraced and any(line.startswith("  untraced steps") for line in lines):
        out.append("text: untraced steps are printed for an answer that lists none")
    if document["kind"] != "checkpoint":
        for key in ("basis", "level", "first_exposure"):
            if document[key] is not None:
                out.append(f"{key}: only a checkpoint's contains states one")
        return
    stated = document["basis"]
    if not isinstance(stated, str) or not stated:
        out.append("basis: a checkpoint's contains states its basis")
    else:
        if "basis" in expect and stated != expect["basis"]:
            out.append(f"basis: {stated!r} is not the expected {expect['basis']!r}")
        if f"  basis: {stated}" not in lines:
            out.append("text: the basis is not printed")
    level = document["level"]
    if level not in LEVELS:
        out.append(f"level: {level!r} is neither consumed nor inherited")
    exposure = document["first_exposure"]
    if not isinstance(exposure, Mapping) or set(exposure) != {"segment", "step"}:
        out.append("first exposure: a checkpoint's contains states segment and step")
        return
    if "first_exposure" in expect and dict(exposure) != dict(expect["first_exposure"]):
        out.append(f"first exposure: {dict(exposure)} is not the expected "
                   f"{dict(expect['first_exposure'])}")
    if own_segment is not None and level in LEVELS:
        wanted = "consumed" if exposure["segment"] == own_segment else "inherited"
        if level != wanted:
            out.append(f"level: {level} for an exposure in {exposure['segment']}, "
                       f"the checkpoint's own segment being {own_segment}")
    if _exposure_line(exposure, level) not in lines:
        out.append("text: the first exposure is not printed")


def _negative(document: Mapping, lines: list[str], expect: Mapping, out: list[str]) -> None:
    for key in ("chain", "candidates", "failing", "names", "basis", "first_exposure", "level"):
        if document[key] not in (None, [], ()):
            out.append(f"{key}: a does not contain names no {key}")
    _notes(document, lines, expect, out)
    scope = document["scope"]
    if not isinstance(scope, Mapping):
        out.append("scope: a does not contain carries the scope of its proof")
        return
    if tuple(scope) != SCOPE_KEYS:
        out.append(f"scope: keys {list(scope)} are not {list(SCOPE_KEYS)}")
        return
    if scope["source"] != document["request"]["source"]:
        out.append("scope: the source is not the request's")
    for key, wanted in expect.get("scope", {}).items():
        got = scope[key]
        if key == "steps":
            same = _pairs(got) == _pairs(wanted)
        elif isinstance(wanted, list):
            same = sorted(got) == sorted(wanted)  # members, records, jobs: a set
        else:
            same = got == wanted
        if not same:
            out.append(f"scope: {key} {got} is not the expected {wanted}")
    if document["kind"] == "checkpoint" and not (scope["runs"] and scope["steps"]):
        out.append("scope: a checkpoint's scope lists its runs and steps")
    head = f"  scope: source {scope['source']}, members {', '.join(scope['members'])}"
    if head not in lines:
        out.append("text: the scope is not printed")
    if scope["steps"] and ("    steps: " + ", ".join(f"{s} {a} to {b}" for s, a, b in scope["steps"])
                           not in lines):
        out.append("text: the steps of the scope are not printed")
    if f"    records walked: {scope['walked']}" not in lines:
        out.append("text: the count of records walked is not printed")


def _may_contain(document: Mapping, lines: list[str], expect: Mapping, out: list[str]) -> None:
    names = document["names"]
    if not names:
        out.append("names: a may contain names its cause")
    elif "names" in expect and names != expect["names"]:
        out.append(f"names: {names!r} is not the expected {expect['names']!r}")
    if "names_detail" in expect and document["names_detail"] != expect["names_detail"]:
        out.append(f"names: the reason {document['names_detail']!r} is not the expected "
                   f"{expect['names_detail']!r}")
    failing = document["failing"]
    if not isinstance(failing, Mapping):
        out.append("failing: a may contain names its first failing link")
    else:
        for key, wanted in expect.get("failing", {}).items():
            if key != "record" and failing.get(key) != wanted:
                out.append(f"failing: {key} {failing.get(key)!r} is not the expected {wanted!r}")
        if not any(line.startswith("  failing link: ") for line in lines):
            out.append("text: the failing link is not printed")
    if names and not any(names in line for line in lines[1:2]):
        out.append("text: the cause is not on the line of the answer")
    for key in ("basis", "first_exposure", "level", "scope"):
        if document[key] is not None:
            out.append(f"{key}: a may contain states no {key}")


def problems(document: Mapping, text: str, expect: Mapping,
             own_segment: str | None = None) -> list[str]:
    """What the answer lacks. Empty: the answer carries what its kind of answer must."""
    if tuple(document) != ANSWER_KEYS:
        return [f"keys: {list(document)} are not the keys of docs/interfaces.md section 6, "
                "in that order"]
    out: list[str] = []
    lines = text.splitlines()
    if document["answer"] != expect.get("answer"):
        out.append(f"answer: {document['answer']!r} is not the expected {expect.get('answer')!r}")
    if not lines or not lines[0].startswith(f"{document['artifact']} ("):
        out.append("text: the first line does not name the artifact")
    if len(lines) < 2 or lines[1].strip().split(":")[0] != document["answer"]:
        out.append("text: the second line is not the answer")
    if document["answer"] == "contains":
        _contains(document, lines, expect, own_segment, out)
    elif document["answer"] == "does not contain":
        _negative(document, lines, expect, out)
    elif document["answer"] == "may contain":
        _may_contain(document, lines, expect, out)
    return out
