#!/usr/bin/env python3
"""List every plan citation in the repository and check that its target exists.

The build plan is in this repository, as docs/WATERWHITE_BUILD_PLAN.markdown; code and docs
cite it by section number. The tool does not read that file as one of the tracked files:
its suffix is .markdown. This script takes the plan's path, collects the anchors the plan defines
and reports every citation in the tracked files whose target is not among them.
It checks that a target exists; whether the target says what the citing line
claims is read by a person.

Anchors: numbered sections such as 2.2, proof conditions 2.2(a) to (h), Decisions,
Checks, limits L1 to L12, assumptions F1 to F10, working rules W1 to W7, risks RK1
to RK4, builder items B-1 to B-13 with their lettered parts, and the ledger ids,
which are the first column of the ledger table at the top of the plan.

What counts as a citation:

- a number after the word "plan" or "Plan" ("plan 2.2(e)", "Plan 3.4", "plan
  Section 7", "Plan sections served: 5.1") and a section "of the plan" ("section
  3.4 of the plan"). A letter in brackets belongs to the number, so "plan 2.2(i)"
  is looked up as 2.2(i);
- "Decision 12", "Check 5" and lists of them ("Checks 1 to 9", "Checks 1-9",
  "Decisions 12, 20, and 31");
- a limit, assumption, rule or risk by its letter (L7, F6, W5, RK2) and a builder
  item (B-7, B-2a);
- "ledger id M9";
- every section number in the ``plan = "..."`` field of a row of the hand-written
  table;
- a reference to DECISIONS.md (D-7, D-07), which must name an entry of that file.

A bare decimal number is ambiguous: a doc cites its own sections, and a number can
be a version or a measurement. It counts only in a list that holds one of the
citations above ("plan 3.5 and 4.1", "(2.4, 3.4, L7)"), inside brackets that hold
one ("(plan 3.1, run segment; 3.5)"), or in brackets that hold nothing but section
numbers ("(4.5)"). It never counts after the word "section" with no "plan", or
after a file name ("docs/interfaces.md 9.2"). A bare "Section 7" with no "plan"
counts only in a list that holds a citation. A number with a letter in brackets
("condition 2.2(f)") counts wherever it stands. A bare number in any other place
("the proof of 2.2") is not counted.

A citation may wrap: the lines of one paragraph, one comment block or one
docstring are read as one text, so "plan" at the end of a line and the number at
the start of the next are one citation.

What is not read: this file itself, and Python code outside comments and strings.
In a string of Python code that is no docstring, "L1" can be a locator and "B-1" a
record id. There a bare letter or builder item counts only after the word "plan",
in a list or in brackets with a citation, or in brackets that hold two or more
letters and nothing else ("(F9, L9)"), and brackets that hold only numbers do not
count. A locator after a quote, a back-tick, a colon or "#" is skipped everywhere.

Usage: check_plan_citations.py <plan.md> [--list]
Exit 0 when every citation resolves, 1 otherwise.
"""

from __future__ import annotations

import bisect
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

# --- The anchors of the plan ----------------------------------------------------------


def plan_anchors(plan_text: str) -> set[str]:
    anchors: set[str] = set()
    body = plan_text
    # Top-level sections: "## 2. Answer rules"
    for m in re.finditer(r"^## (\d+)\. ", body, re.M):
        anchors.add(m.group(1))
    # Subsections: "**2.2 The coverage proof" at the start of a paragraph
    for m in re.finditer(r"^\*\*(\d+\.\d+) ", body, re.M):
        anchors.add(m.group(1))
    # Proof conditions "- (a) every occurrence"
    for m in re.finditer(r"^- \(([a-z])\)", body, re.M):
        anchors.add(f"2.2({m.group(1)})")
    for m in re.finditer(r"^### Decision (\d+):", body, re.M):
        anchors.add(f"Decision {m.group(1)}")
    # Checks are the numbered items of 5.5: "1. **Exposure join"
    for m in re.finditer(r"^(\d+)\. \*\*", body, re.M):
        if int(m.group(1)) <= 10:
            anchors.add(f"Check {m.group(1)}")
    for m in re.finditer(r"^- (L\d+|F\d+|W\d+|RK\d+|B-\d+)\. (.*)$", body, re.M):
        anchors.add(m.group(1))
        if m.group(1).startswith("B-"):
            # The lettered parts of a builder item: "(a) the hash function ...; (b) ..."
            for letter in re.findall(r"\(([a-z])\) ", m.group(2)):
                anchors.add(f"{m.group(1)}({letter})")
    # Ledger ids: the first column of the table under "## Ledger"
    ledger = re.search(r"^## Ledger\b.*?(?=^---\s*$|^# |\Z)", body, re.M | re.S)
    if ledger:
        for m in re.finditer(r"^\| *([A-Z][A-Z0-9]*(?:-\d+)?) *\|", ledger.group(0), re.M):
            anchors.add(f"ledger id {m.group(1)}")
    return anchors


def decision_entries(root: Path) -> set[int]:
    path = root / "DECISIONS.md"
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8")
    return {int(n) for n in re.findall(r"^## D-(\d+)\b", text, re.M)}


# --- A file as segments: the stretches of text a citation can wrap inside -------------


@dataclass
class Segment:
    """One paragraph, comment block, docstring, string or line, as one text.

    ``text`` is the lines joined by one space. ``prose`` is False for a string of
    Python code that is no docstring."""

    text: str
    starts: list[int]
    lines: list[int]
    prose: bool = True

    def line_at(self, offset: int) -> int:
        return self.lines[bisect.bisect_right(self.starts, offset) - 1]


def make_segment(pieces: list[tuple[int, str]], prose: bool = True) -> Segment | None:
    text, starts, lines = "", [], []
    for number, piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if text:
            text += " "
        starts.append(len(text))
        lines.append(number)
        text += piece
    return Segment(text, starts, lines, prose) if text else None


ITEM_START = re.compile(r"\s*(?:[-*+] |\d+\. )")


def paragraphs(block: list[tuple[int, str]], markdown: bool = False) -> list[list[tuple[int, str]]]:
    """Split numbered lines into paragraphs. A blank line and the start of a list
    item end a paragraph. In markdown a heading, a table row and a line of a
    fenced block stand alone."""
    out: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    fenced = False

    def flush() -> None:
        if current:
            out.append(list(current))
            current.clear()

    for number, raw in block:
        stripped = raw.strip()
        if markdown and stripped.startswith("```"):
            fenced = not fenced
            flush()
            continue
        if not stripped:
            flush()
            continue
        if markdown and (fenced or stripped.startswith(("|", "#"))):
            flush()
            out.append([(number, raw)])
            continue
        if ITEM_START.match(raw):
            flush()
        current.append((number, raw))
    flush()
    return out


def keep(segments: list[Segment], pieces: list[tuple[int, str]], prose: bool = True) -> None:
    segment = make_segment(pieces, prose)
    if segment is not None:
        segments.append(segment)


def line_segments(text: str, comment: str) -> list[Segment]:
    """A file read line by line: a run of comment lines is one block of
    paragraphs, every other line stands alone."""
    segments: list[Segment] = []
    block: list[tuple[int, str]] = []

    def flush() -> None:
        for paragraph in paragraphs(block):
            keep(segments, paragraph)
        block.clear()

    for number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith(comment):
            block.append((number, stripped[len(comment):]))
            continue
        flush()
        keep(segments, [(number, raw)])
    flush()
    return segments


def markdown_segments(text: str) -> list[Segment]:
    segments: list[Segment] = []
    for paragraph in paragraphs(list(enumerate(text.splitlines(), 1)), markdown=True):
        keep(segments, paragraph)
    return segments


OPENING_QUOTES = re.compile(r"^[A-Za-z]*(?:\"\"\"|'''|\"|')")
CLOSING_QUOTES = re.compile(r"(?:\"\"\"|'''|\"|')$")
STATEMENT_START = (None, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
STRING_PART = getattr(tokenize, "FSTRING_MIDDLE", None)


def python_segments(text: str) -> list[Segment]:
    """Comments and strings only. A run of whole-line comments is one block of
    paragraphs; a docstring is split into paragraphs; any other string is one
    segment with ``prose`` False."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return line_segments(text, "#")
    segments: list[Segment] = []
    block: list[tuple[int, str]] = []

    def flush() -> None:
        for paragraph in paragraphs(block):
            keep(segments, paragraph)
        block.clear()

    code = [t for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL)]
    place = {id(t): i for i, t in enumerate(code)}
    for token in tokens:
        row = token.start[0]
        if token.type == tokenize.COMMENT:
            body = token.string.lstrip("#")
            if token.line[: token.start[1]].strip():
                keep(segments, [(row, body)])  # a comment after code stands alone
                continue
            if block and block[-1][0] != row - 1:
                flush()
            block.append((row, body))
        elif token.type == tokenize.STRING or (STRING_PART and token.type == STRING_PART):
            i = place[id(token)]
            before = code[i - 1].type if i else None
            after = code[i + 1].type if i + 1 < len(code) else tokenize.ENDMARKER
            docstring = (token.type == tokenize.STRING and before in STATEMENT_START
                         and after in (tokenize.NEWLINE, tokenize.ENDMARKER))
            pieces = [(row + k, part) for k, part in enumerate(token.string.split("\n"))]
            if not docstring:
                keep(segments, pieces, prose=False)
                continue
            pieces[0] = (pieces[0][0], OPENING_QUOTES.sub("", pieces[0][1]))
            pieces[-1] = (pieces[-1][0], CLOSING_QUOTES.sub("", pieces[-1][1].rstrip()))
            for paragraph in paragraphs(pieces):
                keep(segments, paragraph)
    flush()
    return segments


def file_segments(path: Path, text: str) -> list[Segment]:
    if path.suffix == ".py":
        return python_segments(text)
    if path.suffix == ".md":
        return markdown_segments(text)
    return line_segments(text, "--" if path.suffix == ".sql" else "#")


# --- Citations in one segment ---------------------------------------------------------

NUMBER = r"\d{1,2}(?:\.\d{1,2})?(?:\([a-z]{1,4}\))?(?![\w%]|\.\d)"
WHOLE = r"\d{1,3}(?!\d|\.\d)"
BETWEEN = r"\s*(?:,\s*and\b|,\s*or\b|,|\band\b|\bor\b|\bto\b|-|\u2013)\s*"
WHOLE_LIST = rf"{WHOLE}(?:{BETWEEN}{WHOLE})*"
NUMBER_LIST = rf"{NUMBER}(?:{BETWEEN}{NUMBER})*"
LEDGER_ID = r"[A-Z][A-Z0-9]*(?:-\d+)?(?![\w-])"
NOT_A_LOCATOR = r"(?<![\"'`:#/\w-])"

LEDGER = re.compile(rf"\bledger\s+ids?\s+({LEDGER_ID}(?:{BETWEEN}{LEDGER_ID})*)")
OF_THE_PLAN = re.compile(rf"\b[Ss]ections?\s+({NUMBER_LIST})\s+of\s+the\s+(?:build\s+)?plan\b")
PLAN_SECTIONS = re.compile(rf"\b[Pp]lan(?:'s)?\s+[Ss]ections?\s+({WHOLE_LIST})")
PLAN_NUMBER = re.compile(
    rf"\b[Pp]lan(?:'s)?\s+(?:[Ss]ections?(?:\s+(?:it\s+serves|served))?:?\s+)?({NUMBER})")
PLAN_LETTER = re.compile(
    r"\b[Pp]lan\s+(?:(?:limit|assumption|rule|risk)\s+)?((?:RK|L|F|W)[1-9]\d?)\b")
DECISIONS = re.compile(rf"\bDecisions?\s+({WHOLE_LIST})")
CHECKS = re.compile(rf"\bChecks?\s+({WHOLE_LIST})")
LETTER = re.compile(rf"{NOT_A_LOCATOR}((?:RK|L|F|W)[1-9]\d?)\b")
BUILDER = re.compile(rf"{NOT_A_LOCATOR}B-(\d{{1,2}})(?:\(([a-z])\)|([a-z]))?(?![\w-])")
BARE_DECIMAL = re.compile(r"(?<![\w.#/$-])([1-9]\d?\.\d{1,2}(?:\([a-z]{1,4}\))?)(?![\w%]|\.\d)")
BARE_SECTION = re.compile(rf"\bSections?\s+({WHOLE_LIST})")
# What stands right before a number that is a section of another document.
OTHER_DOCUMENT = re.compile(r"(?:\.[a-z]{2,5}[`'\"]?,?\s+(?:[Ss]ections?\s+)?|\b[Ss]ections?\s+)$")
OTHER_DOCUMENT_NAMED = re.compile(r"\.(?:md|toml|py|sql|sh)\b|\b[Ss]ections?\s+\d")
SENTENCE_END = re.compile(r"[.!?]\s+(?=[A-Z`\"(])")
LINK = re.compile(r"\s*(?:[,;]|,?\s*\b(?:and|or|to)\b|-|\u2013)?\s*")
RANGE_LINK = re.compile(r"\s*(?:\bto\b|-|\u2013)\s*")
ITEM_HEAD = re.compile(r"(?:^|[,;]|\b(?:and|or|to))\s*$")
ONLY_BETWEEN = re.compile(r"(?:\s|[,;]|\b(?:and|or|to)\b)*")
PLAN_FIELD = re.compile(r'^\s*plan\s*=\s*"((?:[^"\\]|\\.)*)"')
DECISION_ENTRY = re.compile(r"\bD-(\d{1,2})\b")


@dataclass
class Candidate:
    start: int
    end: int
    labels: list[tuple[int, str]]
    sure: bool                  # False: counts only where the text around it says so
    decimal: bool = False       # a bare decimal number
    other_document: bool = False
    group: int = 0
    counted: bool = False
    extra: list[tuple[int, str]] = field(default_factory=list)


def whole_numbers(kind: str, text: str, offset: int) -> list[tuple[int, str]]:
    """The labels of a list of whole numbers: "12, 20, and 31", "1 to 9", "1-9"."""
    out: list[tuple[int, str]] = []
    last, last_end = None, 0
    for m in re.finditer(r"\d+", text):
        number = int(m.group())
        ranged = last is not None and RANGE_LINK.fullmatch(text[last_end:m.start()])
        first = last + 1 if ranged and last < number <= last + 200 else number
        for n in range(first, number + 1):
            out.append((offset + m.start(), f"{kind}{n}"))
        last, last_end = number, m.end()
    return out


def decimals(text: str, offset: int) -> list[tuple[int, str]]:
    return [(offset + m.start(), m.group()) for m in re.finditer(NUMBER, text)]


def builder_label(m: re.Match) -> str:
    letter = m.group(2) or m.group(3)
    return f"B-{int(m.group(1))}" + (f"({letter})" if letter else "")


def between_range(first: str, second: str) -> list[str]:
    """The labels between the two ends of "3.2 to 3.5" or "L6 to L8"."""
    a = re.fullmatch(r"(\d+)\.(\d+)", first)
    b = re.fullmatch(r"(\d+)\.(\d+)", second)
    if a and b and a.group(1) == b.group(1):
        return [f"{a.group(1)}.{n}" for n in range(int(a.group(2)) + 1, int(b.group(2)))]
    a = re.fullmatch(r"(RK|L|F|W|B-)(\d+)", first)
    b = re.fullmatch(r"(RK|L|F|W|B-)(\d+)", second)
    if a and b and a.group(1) == b.group(1):
        return [f"{a.group(1)}{n}" for n in range(int(a.group(2)) + 1, int(b.group(2)))]
    return []


def candidates(segment: Segment) -> list[Candidate]:
    s = segment.text
    found: list[Candidate] = []
    taken = [False] * len(s)

    def free(m: re.Match) -> bool:
        return not any(taken[m.start():m.end()])

    def add(m: re.Match, labels: list[tuple[int, str]], sure: bool, **more: bool) -> None:
        for i in range(m.start(), m.end()):
            taken[i] = True
        found.append(Candidate(m.start(), m.end(), labels, sure, **more))

    for m in LEDGER.finditer(s):
        ids = re.finditer(LEDGER_ID, m.group(1))
        add(m, [(m.start(1) + i.start(), f"ledger id {i.group()}") for i in ids], True)
    for m in OF_THE_PLAN.finditer(s):
        if free(m):
            add(m, decimals(m.group(1), m.start(1)), True)
    for m in PLAN_SECTIONS.finditer(s):
        if free(m):
            add(m, whole_numbers("", m.group(1), m.start(1)), True)
    for m in PLAN_NUMBER.finditer(s):
        if free(m):
            add(m, [(m.start(1), m.group(1))], True)
    for m in PLAN_LETTER.finditer(s):
        if free(m):
            add(m, [(m.start(1), m.group(1))], True)
    for pattern, kind in ((DECISIONS, "Decision "), (CHECKS, "Check ")):
        for m in pattern.finditer(s):
            if free(m):
                add(m, whole_numbers(kind, m.group(1), m.start(1)), True)
    for m in LETTER.finditer(s):
        if free(m):
            add(m, [(m.start(1), m.group(1))], segment.prose)
    for m in BUILDER.finditer(s):
        if free(m):
            add(m, [(m.start(), builder_label(m))], segment.prose)
    for m in BARE_SECTION.finditer(s):
        if free(m):
            add(m, whole_numbers("", m.group(1), m.start(1)), False,
                other_document=bool(OTHER_DOCUMENT.search(s[:m.start()])))
    for m in BARE_DECIMAL.finditer(s):
        if free(m):
            other = bool(OTHER_DOCUMENT.search(s[:m.start()]))
            # A number with a letter in brackets, as in 2.2(f), is a proof condition
            # of the plan wherever it stands.
            lettered = m.group(1).endswith(")") and not other
            add(m, [(m.start(1), m.group(1))], lettered, decimal=True, other_document=other)
    found.sort(key=lambda c: c.start)
    return found


def names_another_document(s: str, position: int) -> bool:
    """True when the sentence up to ``position`` names a file or a section and
    does not say "plan" after it."""
    start = 0
    for m in SENTENCE_END.finditer(s, 0, position):
        start = m.end()
    sentence = s[start:position]
    last = None
    for last in OTHER_DOCUMENT_NAMED.finditer(sentence):
        pass
    return last is not None and not re.search(r"\b[Pp]lan\b", sentence[last.end():])


def citations(segment: Segment) -> list[tuple[int, str]]:
    """Every plan citation of a segment as (offset, label)."""
    s = segment.text
    found = candidates(segment)
    if not found:
        return []
    # A letter in brackets after a number, as in 2.2(e), opens no bracket group.
    flat = re.sub(r"(?<=\d)\(([a-z]{1,4})\)", lambda m: "<" + m.group(1) + ">", s)
    group = [0] * len(flat)
    opened_at = {0: -1}
    stack = [0]
    for i, char in enumerate(flat):
        if char == "(":
            stack.append(i + 1)
            opened_at[i + 1] = i
        group[i] = stack[-1]
        if char == ")" and len(stack) > 1:
            stack.pop()

    def own_text(start: int, end: int, g: int) -> str:
        """The text from start to end that lies in group g itself, without what
        deeper brackets and quotation marks hold."""
        text = "".join(flat[i] for i in range(start, end) if group[i] == g)
        return re.sub(r'"[^"]*"', "", text)

    for c in found:
        c.group = group[c.start]
        c.counted = c.sure
    by_group: dict[int, list[Candidate]] = {}
    for c in found:
        by_group.setdefault(c.group, []).append(c)

    for g, members in by_group.items():
        # Lists: neighbours with nothing but a comma, "and", "to" or a gloss between.
        runs: list[list[Candidate]] = [[members[0]]]
        for previous, c in zip(members, members[1:]):
            gap = own_text(previous.end, c.start, g)
            linked = LINK.fullmatch(gap) is not None
            if linked and not gap.strip() and previous.decimal and c.decimal:
                linked = False
            if linked:
                runs[-1].append(c)
                if RANGE_LINK.fullmatch(gap) and len(previous.labels) == 1 == len(c.labels):
                    c.extra = [(c.start, label) for label in
                               between_range(previous.labels[0][1], c.labels[0][1])]
            else:
                runs.append([c])
        for run in runs:
            holds_citation = any(c.sure for c in run)
            other = False
            for c in run:
                if c.sure:
                    other = False
                    continue
                other = other or c.other_document
                c.other_document = other
                if holds_citation and not other:
                    c.counted = True
        if g == 0:
            continue
        # Inside brackets.
        opened = opened_at[g]
        closed = next((i for i in range(opened + 1, len(flat))
                       if flat[i] == ")" and group[i] == g), len(flat))
        if any(c.sure for c in members):
            for c in members:
                # A number anywhere in brackets that hold a citation; a letter in a
                # string of code only at the start of an item.
                if not c.counted and not c.other_document:
                    c.counted = c.decimal or bool(
                        ITEM_HEAD.search(own_text(opened + 1, c.start, g)))
        elif (all(c.decimal and not c.other_document for c in members) if segment.prose
              else len(members) > 1 and not any(c.decimal for c in members)):
            # Brackets that hold nothing but section numbers, as in "(4.5)"; in a
            # string of code, nothing but two or more letters, as in "(F9, L9)".
            rest, at = "", opened + 1
            for c in members:
                rest += own_text(at, c.start, g)
                at = c.end
            rest += own_text(at, closed, g)
            if ONLY_BETWEEN.fullmatch(rest) and not names_another_document(s, opened):
                for c in members:
                    c.counted = True

    # The plan field of a table row: every section number in it.
    field_match = PLAN_FIELD.match(s) if len(segment.lines) == 1 else None
    if field_match:
        for c in found:
            if c.decimal and field_match.start(1) <= c.start < field_match.end(1):
                c.counted = True

    out: list[tuple[int, str]] = []
    for c in found:
        if c.counted:
            out.extend(c.labels)
            out.extend(c.extra)
    return sorted(out)


def scan(path: Path, text: str, anchors: set[str], entries: set[int]) -> list[tuple[int, str, bool]]:
    """The citations of one file as (line, citation, whether its target exists)."""
    out: list[tuple[int, str, bool]] = []
    for segment in file_segments(path, text):
        for offset, label in citations(segment):
            out.append((segment.line_at(offset), label, label in anchors))
        for m in DECISION_ENTRY.finditer(segment.text):
            out.append((segment.line_at(m.start()), f"D-{m.group(1)}", int(m.group(1)) in entries))
    return sorted(out, key=lambda item: item[0])


def tracked_files(root: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    keep_suffix = (".py", ".md", ".sql", ".toml", ".sh")
    return [root / p for p in out if p.endswith(keep_suffix)]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    plan = Path(argv[1]).read_text(encoding="utf-8")
    anchors = plan_anchors(plan)
    here = Path(__file__).resolve()
    root = here.parent.parent
    entries = decision_entries(root)
    missing: list[str] = []
    count = 0
    for path in tracked_files(root):
        if path.resolve() == here:
            continue  # the examples in this file are no citations
        for number, cite, ok in scan(path, path.read_text(encoding="utf-8"), anchors, entries):
            count += 1
            if "--list" in argv:
                print(f"{path.relative_to(root)}:{number}: {cite}{'' if ok else '  <-- no such anchor'}")
            if not ok:
                missing.append(f"{path.relative_to(root)}:{number}: {cite}")
    print(f"{count} citations, {len(missing)} without a target, {len(anchors)} anchors in the plan")
    for item in missing:
        print("missing:", item)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
