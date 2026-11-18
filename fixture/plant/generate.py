#!/usr/bin/env python3
"""The plant generator: writes one generated set and the list of what it planted.

A generated set repeats the structure of the hand-checked fixture at scale (plan
5.1): sources A and B that independently supply the same whole passages (the X
pairs), paraphrases P made from A's copies, translations T made from the
paraphrases, summaries S made from one passage of A and one of C (parent mode
"all"), unrelated records, and the releases R1 and R2. One thing is added to the
fixture's structure (plan 5.1 "Sizes", DECISIONS.md D-23): a few paraphrases made
from unrelated records of Z, which the withdrawal of A leaves alone. Without them
every generated record of a set would derive from A, and a report that retired
every generated record could not be told from one that retires by lineage.

The plant list written beside the set is where the expected answers of the 10k
and 100k sets come from (plan W5), so this script is written before the tool
exists and shares no code with it: it uses the standard library only.

Every byte it writes is a function of the parameters and the seed. There is no
clock and no process-wide random state: each text and each placement is drawn from
a byte stream keyed by the seed and by the thing being made, so the same call
gives the same bytes, and a record's text does not depend on what else is in the
set. That is also why the clean set (``--clean``, plan 4.4) holds the remaining
sources byte for byte, and the paraphrases of unrelated records with the texts
they have in the full set.

The file formats, every key of ``set.toml`` and ``plant_list.jsonl`` and the
guarantees the set gives are described in ``docs/plant.md``.

Plan sections: 5.1 (structure, sizes, plant list), 5.2 (the rebuild check's
condition on A's records), 5.5 Checks 3, 6 and 9, 5.6 (the measured cells), 4.1
(actions), 4.4 (clean sources), W3 (self-authored text), W5. DECISIONS.md D-19
fixes the outcome of a release row that copies A's own record; D-23 adds the
paraphrases of unrelated records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

SET_FORMAT = "waterwhite.plant-set/1"
RUN_FORMAT = "waterwhite.plant-run/1"
STREAM_DOMAIN = "waterwhite.plant/1"

WITHDRAWN = "A"  # the source whose withdrawal the plant list describes
NORMALISER = "control-chars/1"  # the normalisation the set's guarantees are stated for
ID_WIDTH = 6
MAX_LINES_PER_FILE = 10**ID_WIDTH - 1

BASE_SOURCES = ("A", "B", "C", "Z")
FURTHER_SOURCES = ("Z2", "Z3", "Z4", "Z5", "Z6", "Z7")

KIND_X = "x-copy"
KIND_PASSAGE = "passage"
KIND_PARAPHRASE = "paraphrase"
KIND_TRANSLATION = "translation"
KIND_SUMMARY = "summary"
KIND_KEPT_PARAPHRASE = "paraphrase-of-unrelated"
KIND_UNRELATED = "unrelated"
KIND_PAIR = "pair-member"
KIND_SANITIZER = "sanitizer-case"

JOB_PARAPHRASE = "paraphrase"
JOB_TRANSLATION = "translation"
JOB_SUMMARY = "summary"

# Paraphrases made from unrelated records of Z (plan 5.1 "Sizes", DECISIONS.md
# D-23). Their ids carry a prefix of their own, and no other id of a set starts
# with its first letter, so no reading of an id can take one for a paraphrase of A.
KEPT_PREFIX = "UP"
KEPT_PER_RECORDS = 1000  # one for every thousand records of the set
KEPT_MINIMUM = 2  # so that retiring one of them still leaves one that stays

# How a release input stands in the release before the withdrawal.
ROW_PRESENT = "present"
ROW_LOSER = "dedup-loser"

# What the rebuild does with a release input under "withdraw A" (plan 4.1, 4.4):
# the plan's four action words for a record, and "none". A row that copies A's own
# record and has no equivalent copy to take its place is retired like any other
# descendant: it descends from the withdrawn record through a single-parent copy
# edge, and there is no further word for it (DECISIONS.md D-19).
ACT_KEEP = "keep"
ACT_RE_SOURCE = "re-source"
ACT_PROMOTE = "promote"
ACT_RETIRE = "retire"
ACT_NONE = "none"  # never a row of the release, and none of the successor

# Sanitizer cases (plan 5.1, Check 3). The two reasons are the tool's drop reasons,
# spelled here because this script imports nothing from the tool.
CASE_UNPARSEABLE = "unparseable"
CASE_EMPTY = "empty"
CASE_CONTROL_ONLY = "control-only"
CASE_CONTROL_AND_SPACE = "control-and-space"
CASE_CONTROL_INSIDE = "control-inside"
REASON_SKIP = "sanitizer skip"
REASON_EMPTY = "empty line"

# The control characters this script plants: the C0 range without NUL, tab, line
# feed and carriage return. Every reading of "control character" removes these, and
# no text holds a tab, a line break or any other character a reading could dispute.
CONTROL_CHARS = tuple(chr(c) for c in (*range(1, 9), 11, 12, *range(14, 32)))


class PlantError(Exception):
    """Base class of this script's errors. ``exit_code`` is what the command exits with."""

    exit_code = 2


class ParameterError(PlantError):
    """The parameters cannot give a set with the guarantees of docs/plant.md."""


class OutputFolderNotEmpty(PlantError):
    """The output folder already holds something. A set is never written over files."""


class SelfCheckFailed(PlantError):
    """The generator caught itself writing something its guarantees forbid."""

    exit_code = 3


# --- Word lists (self-authored; no corpus text, plan W3) ---------------------------

EN_ADJ = (
    "quiet narrow copper distant patient hollow bright weathered careful sudden plain "
    "heavy early late wooden silver steady crooked gentle broad pale sturdy restless "
    "modest ancient fresh tidy dusty mild steep shallow curious loyal brisk humble "
    "rusty velvet amber northern southern eastern western hidden open frozen warm "
    "salted painted"
).split()

EN_NOUN = (
    "harbor ledger orchard lantern bridge archive meadow kiln compass quarry ferry "
    "granary workshop signal canal cellar tower garden market valley anchor barrel "
    "basket beacon bell boat cabin candle cart chimney clock courtyard crane dock "
    "engine fence field forge fountain gate hammer hedge hill kettle ladder lock loom "
    "mill notebook oven path pier plow pump rail river roof rope shed stable stair "
    "trellis wagon well"
).split()

EN_VERB = (
    "measured repaired counted followed crossed watched carried painted opened closed "
    "marked cleaned lifted sorted guarded described recorded mapped visited weighed "
    "mended polished tested traced copied moved stacked filled emptied lit turned "
    "raised lowered checked named listed drew built found kept left met passed "
    "reached sealed shared sketched stored"
).split()

EN_TAIL = (
    "before noon",
    "after the rain",
    "without haste",
    "in early spring",
    "by the old road",
    "at first light",
    "for the third time",
    "with a borrowed tool",
    "under a grey sky",
    "during the long winter",
    "on the quiet side",
    "near the last bend",
    "as the tide went out",
    "while the wind held",
    "at the turn of the week",
    "in a single afternoon",
    "once the frost had gone",
    "by lamp and by hand",
    "with great care",
    "in plain sight",
    "from the upper floor",
    "across two seasons",
    "at the usual hour",
    "between two storms",
    "until the bell rang",
    "along the low wall",
    "over many evenings",
    "with steady hands",
    "beyond the far gate",
    "since the first thaw",
    "at the end of the lane",
    "through the dry months",
)

EN_PLACE = (
    "north gate",
    "old mill",
    "stone quay",
    "lower field",
    "east wall",
    "salt marsh",
    "long barn",
    "clock tower",
    "upper weir",
    "reed bank",
    "market square",
    "timber yard",
    "hill road",
    "south pier",
    "grain loft",
    "ferry landing",
    "mill pond",
    "boat shed",
    "west orchard",
    "back lane",
    "iron bridge",
    "brick kiln",
    "chalk pit",
    "cattle ford",
    "signal hut",
    "rope walk",
    "well house",
    "sheep fold",
    "tide gate",
    "coal wharf",
    "apple store",
    "toll house",
)

EN_LEAD = (
    "In that season",
    "By all accounts",
    "Some days later",
    "As the notes say",
    "Before the harvest",
    "After the survey",
    "On the second visit",
    "For a full month",
    "At the same time",
    "Without any notice",
    "Early that year",
    "Late in the day",
    "From then on",
    "Step by step",
    "Once again",
    "More than once",
)

# German, for the translations. Each noun is listed once as a subject and once as
# an object, with its article, so a sentence needs no declension rule.
DE_SUBJECT = (
    "der Hafen",
    "die Brücke",
    "das Archiv",
    "der Garten",
    "die Mühle",
    "das Boot",
    "der Turm",
    "die Werkstatt",
    "das Lager",
    "der Kanal",
    "die Laterne",
    "das Tor",
    "der Markt",
    "die Fähre",
    "das Feld",
    "der Brunnen",
    "die Scheune",
    "das Dach",
    "der Wagen",
    "die Uhr",
    "das Fass",
    "der Zaun",
    "die Leiter",
    "das Seil",
    "der Ofen",
    "die Glocke",
    "das Ufer",
    "der Hügel",
    "die Treppe",
    "das Tal",
    "der Keller",
    "die Schmiede",
)

DE_OBJECT = (
    "den Hafen",
    "die Brücke",
    "das Archiv",
    "den Garten",
    "die Mühle",
    "das Boot",
    "den Turm",
    "die Werkstatt",
    "das Lager",
    "den Kanal",
    "die Laterne",
    "das Tor",
    "den Markt",
    "die Fähre",
    "das Feld",
    "den Brunnen",
    "die Scheune",
    "das Dach",
    "den Wagen",
    "die Uhr",
    "das Fass",
    "den Zaun",
    "die Leiter",
    "das Seil",
    "den Ofen",
    "die Glocke",
    "das Ufer",
    "den Hügel",
    "die Treppe",
    "das Tal",
    "den Keller",
    "die Schmiede",
)

DE_VERB = (
    "erreichte prüfte ordnete beschrieb zählte verband öffnete bewachte zeichnete "
    "reparierte sammelte verglich bemalte reinigte füllte leerte schloss trug fand "
    "hielt verließ baute maß wog kopierte verschob stapelte hob senkte nannte teilte "
    "lagerte"
).split()

DE_TAIL = (
    "am Morgen",
    "nach dem Regen",
    "ohne Eile",
    "im frühen Frühling",
    "an der alten Straße",
    "beim ersten Licht",
    "zum dritten Mal",
    "mit geliehenem Werkzeug",
    "unter grauem Himmel",
    "im langen Winter",
    "auf der stillen Seite",
    "vor der letzten Biegung",
    "bei ablaufender Flut",
    "solange der Wind hielt",
    "zur Wochenmitte",
    "an einem einzigen Nachmittag",
    "nach dem letzten Frost",
    "bei Lampenlicht",
    "mit großer Sorgfalt",
    "vor aller Augen",
    "vom oberen Stockwerk",
    "über zwei Jahreszeiten",
    "zur gewohnten Stunde",
    "zwischen zwei Stürmen",
    "bis die Glocke läutete",
    "entlang der niedrigen Mauer",
    "an vielen Abenden",
    "mit ruhiger Hand",
    "hinter dem fernen Tor",
    "seit dem ersten Tauwetter",
    "am Ende des Weges",
    "in den trockenen Monaten",
)

DE_LEAD = (
    "In jener Zeit",
    "Nach allen Berichten",
    "Einige Tage später",
    "Laut den Notizen",
    "Vor der Ernte",
    "Nach der Vermessung",
    "Beim zweiten Besuch",
    "Einen ganzen Monat lang",
    "Zur selben Zeit",
    "Ohne jede Ankündigung",
    "Früh in jenem Jahr",
    "Spät am Tag",
    "Von da an",
    "Schritt für Schritt",
    "Noch einmal",
    "Mehr als einmal",
)

# 64 syllables; four of them spell the code word that makes a text unique.
SYLLABLES = tuple(c + v for c in "bdfgklmnprstvzhj" for v in "aeio")

BYTES_PER_SENTENCE = 8
MIN_SENTENCE_CHARS = 24  # shorter than any sentence either language can give


# --- The seeded generator ----------------------------------------------------------


def keyed_bytes(seed: int, key: str, count: int) -> bytes:
    """``count`` bytes that depend on the seed and the key and on nothing else.

    This is the script's only source of variation: an extendable-output hash of the
    seed and a key that names what is being drawn."""
    return hashlib.shake_256(f"{STREAM_DOMAIN}|{seed}|{key}".encode()).digest(count)


def pick_distinct(seed: int, key: str, population: int, count: int) -> list[int]:
    """``count`` distinct integers from 1..population, in drawn order."""
    if count > population:
        raise ParameterError(f"cannot draw {count} distinct positions from {population}")
    pool = list(range(1, population + 1))
    stream = keyed_bytes(seed, key, 4 * count)
    for i in range(count):
        r = int.from_bytes(stream[4 * i : 4 * i + 4], "big")
        j = i + r % (population - i)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:count]


def code_word(serial: int) -> str:
    """The four-syllable word of a serial number. Different serials, different words."""
    base = len(SYLLABLES)
    if not 0 <= serial < base**4:
        raise SelfCheckFailed(f"serial {serial} has no four-syllable code word")
    parts = []
    for _ in range(4):
        serial, digit = divmod(serial, base)
        parts.append(SYLLABLES[digit])
    return "".join(reversed(parts)).capitalize()


def plant_normalise(text: str) -> str:
    """Remove control characters and strip the ends: what ``control-chars/1`` is
    stated to do. Written here on its own, because nothing is imported from the tool;
    the set is built so that every reading of that statement agrees on its texts."""
    kept = "".join(ch for ch in text if not (ord(ch) < 0x20 or 0x7F <= ord(ch) < 0xA0))
    return kept.strip()


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- Sentences ---------------------------------------------------------------------


def _stream_for(seed: int, key: str, target: int) -> bytes:
    return keyed_bytes(seed, key, BYTES_PER_SENTENCE * (target // MIN_SENTENCE_CHARS + 2))


def english_text(first: str, stream: bytes, target: int) -> str:
    """Sentences after ``first`` until the text is at least ``target`` characters."""
    adj, noun, verb, tail, place, lead = EN_ADJ, EN_NOUN, EN_VERB, EN_TAIL, EN_PLACE, EN_LEAD
    na, nn, nv, nt, np_, nl = len(adj), len(noun), len(verb), len(tail), len(place), len(lead)
    parts = [first]
    size = len(first)
    i = 0
    while size < target:
        b = stream[i : i + BYTES_PER_SENTENCE]
        if len(b) < BYTES_PER_SENTENCE:
            raise SelfCheckFailed("the byte stream of a text ran out before its length was reached")
        i += BYTES_PER_SENTENCE
        pattern = b[0] & 3
        if pattern == 0:
            s = (
                f"The {adj[b[1] % na]} {noun[b[2] % nn]} {verb[b[3] % nv]} "
                f"the {noun[b[4] % nn]} {tail[b[5] % nt]}."
            )
        elif pattern == 1:
            s = (
                f"{lead[b[1] % nl]}, the {noun[b[2] % nn]} {verb[b[3] % nv]} "
                f"the {adj[b[4] % na]} {noun[b[5] % nn]}."
            )
        elif pattern == 2:
            s = (
                f"Near the {place[b[1] % np_]}, the {adj[b[2] % na]} {noun[b[3] % nn]} "
                f"{verb[b[4] % nv]} the {noun[b[5] % nn]}."
            )
        else:
            s = (
                f"The {noun[b[1] % nn]} {verb[b[2] % nv]} the {adj[b[3] % na]} "
                f"{noun[b[4] % nn]} {tail[b[5] % nt]}."
            )
        parts.append(s)
        size += len(s) + 1
    return " ".join(parts)


def german_text(first: str, stream: bytes, target: int) -> str:
    """The German counterpart of ``english_text``, for the translations."""
    subj, obj, verb, tail, lead = DE_SUBJECT, DE_OBJECT, DE_VERB, DE_TAIL, DE_LEAD
    ns, no, nv, nt, nl = len(subj), len(obj), len(verb), len(tail), len(lead)
    parts = [first]
    size = len(first)
    i = 0
    while size < target:
        b = stream[i : i + BYTES_PER_SENTENCE]
        if len(b) < BYTES_PER_SENTENCE:
            raise SelfCheckFailed("the byte stream of a text ran out before its length was reached")
        i += BYTES_PER_SENTENCE
        pattern = b[0] & 3
        if pattern == 0:
            s = f"{_cap(subj[b[1] % ns])} {verb[b[2] % nv]} {obj[b[3] % no]} {tail[b[4] % nt]}."
        elif pattern == 1:
            s = f"{lead[b[1] % nl]} {verb[b[2] % nv]} {subj[b[3] % ns]} {obj[b[4] % no]}."
        elif pattern == 2:
            s = (
                f"{_cap(subj[b[1] % ns])} {verb[b[2] % nv]} {obj[b[3] % no]}, und "
                f"{subj[b[4] % ns]} {verb[b[5] % nv]} {obj[b[6] % no]}."
            )
        else:
            s = f"{_cap(tail[b[1] % nt])} {verb[b[2] % nv]} {subj[b[3] % ns]} {obj[b[4] % no]}."
        parts.append(s)
        size += len(s) + 1
    return " ".join(parts)


def _cap(phrase: str) -> str:
    return phrase[0].upper() + phrase[1:]


# --- Parameters and layout ---------------------------------------------------------


@dataclass(frozen=True)
class Params:
    records: int
    seed: int
    length: int = 2000
    derivations: int = 1
    sources: int = 4
    dedup: bool = True
    chunks: int = 20
    pairs: int = 300
    sanitizer_cases: int = 6
    clean: bool = False


@dataclass
class Layout:
    """Where everything sits. All counts follow from the parameters alone."""

    params: Params
    sources: tuple[str, ...]
    lines: dict[str, int]  # source name -> number of lines in its file
    n_x: int  # X pairs: lines 1..n_x of A and of B
    n_s: int  # summary passages: A lines n_x+1.., C lines 1..n_s
    z_r1: dict[str, int]  # per unrelated source: lines 1..z_r1 are inputs of R1
    n_kept: int = 0  # paraphrases of unrelated records of Z
    z_pair_line: dict[int, int] = field(default_factory=dict)  # Z line -> pair number
    z_case_line: dict[int, int] = field(default_factory=dict)  # Z line -> case number
    z_kept_parent: list[int] = field(default_factory=list)  # entry k-1: the Z line UP-k is made from

    @property
    def n_p(self) -> int:
        return self.n_x * self.params.derivations

    @property
    def n_sum(self) -> int:
        return self.n_s * self.params.derivations

    @property
    def n_derived(self) -> int:
        """Generated records that derive from A: every P, T and S."""
        return 2 * self.n_p + self.n_sum

    @property
    def n_generated(self) -> int:
        return self.n_derived + self.n_kept


def kept_paraphrases(p: Params) -> int:
    """How many paraphrases of unrelated records a set holds.

    One for every thousand records, and never fewer than two: 10 in the 10k set and
    100 in the 100k set, so their share is the same in both. One would be enough to
    tell a report that retires by lineage from one that retires every generated
    record; two are the least with which a wrong report can retire one of them and
    leave another in place. A set with no derivations has no generation job at
    all, so it holds none: there is no generated record to mistake them for."""
    if p.derivations == 0:
        return 0
    return max(KEPT_MINIMUM, p.records // KEPT_PER_RECORDS)


def plan_layout(p: Params) -> Layout:
    """Turn the parameters into counts and placements, or refuse them."""
    _check_types(p)
    names = BASE_SOURCES + (FURTHER_SOURCES if p.sources == 10 else ())
    n_x = p.records // 20
    n_s = p.records // 100
    if n_x < p.chunks:
        raise ParameterError(
            f"{p.records} records give {n_x} X copies in A's file, fewer than the {p.chunks} "
            f"chunks: A's records could miss a chunk (plan 4.5, 5.2); use at least "
            f"{20 * p.chunks} records or fewer chunks"
        )
    if n_s < 1:
        raise ParameterError("too few records for one summary passage (needs 100 or more)")
    lines = {"A": n_x + n_s, "B": 2 * n_x, "C": 2 * n_s + p.pairs}
    # The paraphrases of unrelated records are inside the total: each takes the
    # place of one unrelated line, so the set still holds ``records`` records.
    n_kept = kept_paraphrases(p)
    generated = p.derivations * (2 * n_x + n_s) + n_kept
    rest = p.records - sum(lines.values()) - generated
    if p.sources == 10:
        share = rest // 10
        if share < 2:
            raise ParameterError("too few records to give each of ten sources two lines")
        for name in FURTHER_SOURCES:
            lines[name] = share
        lines["Z"] = rest - share * len(FURTHER_SOURCES)
    else:
        lines["Z"] = rest
    special = p.pairs + 3 * p.sanitizer_cases + n_kept
    z_r1 = {name: lines[name] // 2 for name in names if name not in ("A", "B", "C")}
    if lines["Z"] < 2 or z_r1["Z"] < special + p.chunks:
        raise ParameterError(
            f"{p.records} records leave too few lines in source Z for {p.pairs} pair members, "
            f"{3 * p.sanitizer_cases} sanitizer cases, {n_kept} parents of paraphrases and "
            f"{p.chunks} further unrelated records; "
            f"use more records, fewer pairs or fewer sanitizer cases"
        )
    for name, count in lines.items():
        if count > MAX_LINES_PER_FILE:
            raise ParameterError(f"source {name} would hold {count} lines; ids have {ID_WIDTH} digits")
    if generated // 2 > MAX_LINES_PER_FILE:
        raise ParameterError(f"too many generated records for ids of {ID_WIDTH} digits")
    layout = Layout(params=p, sources=names, lines=lines, n_x=n_x, n_s=n_s, z_r1=z_r1,
                    n_kept=n_kept)
    # One draw places everything planted in Z, so no line is drawn twice: the pair
    # members, the sanitizer cases, and the unrelated records paraphrases are made
    # from. All of them lie in the part of Z that is an input of R1.
    drawn = pick_distinct(p.seed, "placement|Z", z_r1["Z"], special)
    cases_end = p.pairs + 3 * p.sanitizer_cases
    for number, line in enumerate(drawn[: p.pairs], start=1):
        layout.z_pair_line[line] = number
    for number, line in enumerate(drawn[p.pairs : cases_end]):
        layout.z_case_line[line] = number
    layout.z_kept_parent = list(drawn[cases_end:])
    return layout


def _check_types(p: Params) -> None:
    for name in ("records", "seed", "length", "derivations", "sources", "chunks", "pairs", "sanitizer_cases"):
        value = getattr(p, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ParameterError(f"{name} must be an integer, got {value!r}")
    if p.seed < 0:
        raise ParameterError("seed must be zero or more")
    if p.length < 120:
        raise ParameterError("length must be 120 characters or more")
    if p.derivations not in (0, 1, 2):
        raise ParameterError("derivations must be 0, 1 or 2 (plan 5.6)")
    if p.sources not in (4, 10):
        raise ParameterError("sources must be 4 or 10 (plan 5.6)")
    if p.chunks < 2:
        raise ParameterError("chunks must be 2 or more")
    if p.pairs < 2:
        raise ParameterError("pairs must be 2 or more, so that both kinds of placement can occur")
    if p.sanitizer_cases < 3:
        raise ParameterError("sanitizer_cases must be 3 or more, so that every variant is planted")


# --- Ids ---------------------------------------------------------------------------


def rid(prefix: str, number: int) -> str:
    return f"{prefix}-{number:0{ID_WIDTH}d}"


def compress(ids: list[str]) -> list[str]:
    """Write a list of ids as ranges "A-000001..A-000500" where they run on."""
    out: list[str] = []
    start = prev = None
    for item in ids:
        prefix, _, digits = item.rpartition("-")
        number = int(digits)
        if prev is not None and prev[0] == prefix and prev[1] + 1 == number:
            prev = (prefix, number)
            continue
        if start is not None:
            out.append(_range(start, prev))
        start = prev = (prefix, number)
    if start is not None:
        out.append(_range(start, prev))
    return out


def _range(start: tuple[str, int], end: tuple[str, int]) -> str:
    first = rid(*start)
    return first if start == end else f"{first}..{rid(*end)}"


# --- The set -----------------------------------------------------------------------


class _Files:
    """Writes the set's files and keeps each one's digest, size and line count."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stats: dict[str, dict] = {}
        self._open: dict[str, object] = {}
        self._hash: dict[str, object] = {}

    def start(self, rel: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        self._open[rel] = open(path, "wb")
        self._hash[rel] = hashlib.sha256()
        self.stats[rel] = {"bytes": 0, "lines": 0}

    def line(self, rel: str, text: str) -> None:
        data = text.encode("utf-8") + b"\n"
        self._open[rel].write(data)
        self._hash[rel].update(data)
        self.stats[rel]["bytes"] += len(data)
        self.stats[rel]["lines"] += 1

    def finish(self, rel: str) -> None:
        self._open.pop(rel).close()
        self.stats[rel]["sha256"] = self._hash.pop(rel).hexdigest()

    def whole(self, rel: str, content: str) -> None:
        data = content.encode("utf-8")
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.stats[rel] = {
            "bytes": len(data),
            "lines": content.count("\n"),
            "sha256": hashlib.sha256(data).hexdigest(),
        }


class _Builder:
    def __init__(self, layout: Layout) -> None:
        self.layout = layout
        self.p = layout.params
        self.order = {name: i for i, name in enumerate(layout.sources)}
        self.job_index = {
            "P": len(layout.sources),
            "T": len(layout.sources) + 1,
            "S": len(layout.sources) + 2,
            KEPT_PREFIX: len(layout.sources) + 3,
        }
        self.digests: dict[str, str] = {}  # text digest -> pair id, or "" when it must be unique
        self.counts: dict[str, int] = {}
        self.pair_line = {number: line for line, number in layout.z_pair_line.items()}

    # --- texts -------------------------------------------------------------------

    def _code(self, prefix: str, number: int) -> str:
        index = self.order.get(prefix, self.job_index.get(prefix))
        return code_word(index * (MAX_LINES_PER_FILE + 1) + number)

    def source_text(self, prefix: str, number: int) -> str:
        """The text of the record a source line owns. An identical copy reuses its
        owner's text, so the copy carries the owner's code word."""
        owner = rid(prefix, number)
        first = f"Entry {self._code(prefix, number)} opens this note."
        target = self.p.length
        return english_text(first, _stream_for(self.p.seed, f"text|{owner}", target), target)

    def paraphrase_text(self, number: int, parent: tuple[str, int]) -> str:
        first = f"Paraphrase {self._code('P', number)} restates entry {self._code(*parent)}."
        key = f"text|{rid('P', number)}|{rid(*parent)}"
        target = self.p.length
        return english_text(first, _stream_for(self.p.seed, key, target), target)

    def kept_paraphrase_text(self, number: int, z_line: int) -> str:
        first = (
            f"Paraphrase {self._code(KEPT_PREFIX, number)} restates entry "
            f"{self._code('Z', z_line)}."
        )
        key = f"text|{rid(KEPT_PREFIX, number)}|{rid('Z', z_line)}"
        target = self.p.length
        return english_text(first, _stream_for(self.p.seed, key, target), target)

    def translation_text(self, number: int) -> str:
        first = f"Übersetzung {self._code('T', number)} überträgt Paraphrase {self._code('P', number)}."
        key = f"text|{rid('T', number)}|{rid('P', number)}"
        target = self.p.length
        return german_text(first, _stream_for(self.p.seed, key, target), target)

    def summary_text(self, number: int, a: tuple[str, int], c: tuple[str, int]) -> str:
        first = (
            f"Summary {self._code('S', number)} joins entry {self._code(*a)} "
            f"with entry {self._code(*c)}."
        )
        key = f"text|{rid('S', number)}|{rid(*a)}|{rid(*c)}"
        target = max(120, self.p.length // 2)
        return english_text(first, _stream_for(self.p.seed, key, target), target)

    # --- rows --------------------------------------------------------------------

    def _note(self, digest: str | None, pair: str | None, record: str) -> None:
        if digest is None:
            return
        seen = self.digests.get(digest)
        if seen is None:
            self.digests[digest] = pair or ""
        elif not pair or seen != pair:
            raise SelfCheckFailed(f"the text of {record} repeats another record's text")

    def _row(
        self,
        *,
        record: str,
        file: str,
        line: int,
        source: str | None,
        kind: str,
        text: str | None,
        parents: list[str] | None = None,
        roots: list[str] | None = None,
        pair: str | None = None,
        twin: str | None = None,
        releases: dict | None = None,
        sanitizer: dict | None = None,
        job: str | None = None,
    ) -> dict:
        digest = None if text is None else text_digest(text)
        self._note(digest, pair, record)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        own = source == WITHDRAWN
        derived = bool(roots)
        return {
            "id": record,
            "file": file,
            "line": line,
            "source": source,
            "kind": kind,
            "job": job,
            "parents": parents or [],
            "parent_mode": "all" if parents else None,
            "pair": pair,
            "twin": twin,
            "withdraw_A": {
                "in_scope": own or derived,
                "why": "own" if own else ("derived" if derived else None),
                "roots": roots or [],
            },
            "releases": releases or {},
            "sanitizer": sanitizer,
            "text_sha256": digest,
            "chars": None if text is None else len(text),
        }

    def source_lines(self, name: str):
        """Yield (raw line, plant row) for every line of one source file."""
        lay, p = self.layout, self.p
        file = f"sources/{name}.jsonl"
        on = p.dedup
        for n in range(1, lay.lines[name] + 1):
            record = rid(name, n)
            releases: dict = {}
            pair = twin = sanitizer = None
            kind = KIND_UNRELATED
            raw = None
            if name == "A":
                if n <= lay.n_x:
                    kind, pair, twin = KIND_X, rid("X", n), rid("B", n)
                    text = self.source_text("A", n)
                    releases["R1"] = (
                        {"row": ROW_PRESENT, "withdraw_A": ACT_RE_SOURCE, "resource_to": twin}
                        if on
                        else {"row": ROW_PRESENT, "withdraw_A": ACT_RETIRE}
                    )
                else:
                    kind = KIND_PASSAGE
                    text = self.source_text("A", n)
            elif name == "B":
                if n <= lay.n_x:
                    kind, pair, twin = KIND_X, rid("X", n), rid("A", n)
                    text = self.source_text("A", n)
                    releases["R1"] = (
                        {"row": ROW_LOSER, "withdraw_A": ACT_PROMOTE, "promoted_for": twin}
                        if on
                        else {"row": ROW_PRESENT, "withdraw_A": ACT_KEEP}
                    )
                else:
                    text = self.source_text("B", n)
                releases["R2"] = {"row": ROW_PRESENT, "withdraw_A": ACT_KEEP}
            elif name == "C":
                if n <= lay.n_s:
                    kind = KIND_PASSAGE
                    text = self.source_text("C", n)
                elif n <= lay.n_s + p.pairs:
                    number = n - lay.n_s
                    kind, pair = KIND_PAIR, rid("PAIR", number)
                    twin = rid("Z", self.pair_line[number])
                    text = self.source_text("C", n)
                    releases["R1"] = {"row": ROW_PRESENT, "withdraw_A": ACT_KEEP}
                else:
                    text = self.source_text("C", n)
            else:
                in_r1 = n <= lay.z_r1[name]
                release = "R1" if in_r1 else "R2"
                releases[release] = {"row": ROW_PRESENT, "withdraw_A": ACT_KEEP}
                if name == "Z" and n in lay.z_pair_line:
                    number = lay.z_pair_line[n]
                    kind, pair = KIND_PAIR, rid("PAIR", number)
                    twin = rid("C", lay.n_s + number)
                    text = self.source_text("C", lay.n_s + number)
                    if on:
                        releases["R1"] = {"row": ROW_LOSER, "withdraw_A": ACT_NONE}
                elif name == "Z" and n in lay.z_case_line:
                    kind = KIND_SANITIZER
                    raw, text, sanitizer = self._sanitizer_case(record, n, lay.z_case_line[n])
                else:
                    text = self.source_text(name, n)
            if raw is None:
                raw = json.dumps({"id": record, "text": text}, ensure_ascii=False)
            row = self._row(
                record=record,
                file=file,
                line=n,
                source=name,
                kind=kind,
                text=text,
                pair=pair,
                twin=twin,
                releases=releases,
                sanitizer=sanitizer,
            )
            yield raw, row

    def _sanitizer_case(self, record: str, n: int, number: int) -> tuple[str, str | None, dict]:
        """One sanitizer case: its raw line, the text that survives (None when the
        line is skipped) and its plant entry. Cases cycle through the three groups
        and, inside each group, through its variants."""
        seed = self.p.seed
        group, variant = number % 3, (number // 3) % 3
        base = self.source_text("Z", n)
        if group == 0:
            words = base[: 60 + 7 * variant]
            if variant == 0:
                raw = '{"id": "' + record + '", "text": "' + words
            elif variant == 1:
                raw = f"line {record} was never written as a record: {words}"
            else:
                raw = '{"id": "' + record + '" "text": "' + words + '"}'
            try:
                json.loads(raw)
            except ValueError:
                pass
            else:
                raise SelfCheckFailed(f"the unparseable line planted at {record} parses")
            return raw, None, self._case(CASE_UNPARSEABLE, True, REASON_SKIP, raw)
        stream = keyed_bytes(seed, f"sanitizer|{record}", 64)
        if group == 1:
            if variant == 0:
                case, text = CASE_EMPTY, ""
            elif variant == 1:
                case = CASE_CONTROL_ONLY
                text = "".join(CONTROL_CHARS[b % len(CONTROL_CHARS)] for b in stream[:5])
            else:
                case = CASE_CONTROL_AND_SPACE
                text = " ".join(CONTROL_CHARS[b % len(CONTROL_CHARS)] for b in stream[:4]) + " "
            if plant_normalise(text) != "":
                raise SelfCheckFailed(f"the empty case planted at {record} is not empty")
            raw = json.dumps({"id": record, "text": text}, ensure_ascii=False)
            return raw, None, self._case(case, True, REASON_EMPTY, raw)
        # Control characters inside real text: at both ends and at four drawn places.
        inner = sorted(
            1 + int.from_bytes(stream[8 + 4 * k : 12 + 4 * k], "big") % (len(base) - 1)
            for k in range(4)
        )
        text = base
        for k, at in reversed(list(enumerate([0, *inner, len(base)]))):
            run = CONTROL_CHARS[stream[k] % len(CONTROL_CHARS)]
            if stream[32 + k] & 1:
                run += CONTROL_CHARS[stream[40 + k] % len(CONTROL_CHARS)]
            text = text[:at] + run + text[at:]
        if plant_normalise(text) != base or text == base:
            raise SelfCheckFailed(f"the control-character case planted at {record} is wrong")
        raw = json.dumps({"id": record, "text": text}, ensure_ascii=False)
        return raw, base, self._case(CASE_CONTROL_INSIDE, False, None, raw)

    @staticmethod
    def _case(case: str, skip: bool, reason: str | None, raw: str) -> dict:
        return {
            "case": case,
            "skip": skip,
            "reason": reason,
            "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        }

    def generation_lines(self, clean: bool = False):
        """Yield (raw line, plant row) for every row of generation.jsonl: the
        paraphrases of A's records, the paraphrases of unrelated records, then all
        translations, then all summaries.

        With ``clean`` only the paraphrases of unrelated records are yielded: they
        are the generated records that do not derive from A, so the clean sources
        hold them (plan 4.4, 5.1 "Sizes"), each with the text it has in the full set."""
        lay = self.layout
        d = 0 if clean else self.p.derivations
        file = "generation.jsonl"
        retire = {"R1": {"row": ROW_PRESENT, "withdraw_A": ACT_RETIRE}}
        keep = {"R1": {"row": ROW_PRESENT, "withdraw_A": ACT_KEEP}}
        line = 0
        for k in range(1, lay.n_x + 1):
            for v in range(1, d + 1):
                number = (k - 1) * d + v
                parent = rid("A", k)
                text = self.paraphrase_text(number, ("A", k))
                line += 1
                yield self._generated(
                    rid("P", number), JOB_PARAPHRASE, [parent], text,
                    self._row(record=rid("P", number), file=file, line=line, source=None,
                              kind=KIND_PARAPHRASE, text=text, parents=[parent],
                              roots=[parent], releases=dict(retire), job=JOB_PARAPHRASE),
                )
        # A paraphrase of an unrelated record of Z is made like P: the job
        # "paraphrase" is handed one record and uses it. Its parent is not A's and
        # leads to no record of A, so the row carries no root: "withdraw A" does not
        # put it in scope, and it stays a row of R1.
        for number, z_line in enumerate(lay.z_kept_parent, start=1):
            record = rid(KEPT_PREFIX, number)
            parent = rid("Z", z_line)
            text = self.kept_paraphrase_text(number, z_line)
            line += 1
            yield self._generated(
                record, JOB_PARAPHRASE, [parent], text,
                self._row(record=record, file=file, line=line, source=None,
                          kind=KIND_KEPT_PARAPHRASE, text=text, parents=[parent],
                          roots=None, releases=dict(keep), job=JOB_PARAPHRASE),
            )
        for k in range(1, lay.n_x + 1):
            for v in range(1, d + 1):
                number = (k - 1) * d + v
                parent = rid("P", number)
                text = self.translation_text(number)
                line += 1
                yield self._generated(
                    rid("T", number), JOB_TRANSLATION, [parent], text,
                    self._row(record=rid("T", number), file=file, line=line, source=None,
                              kind=KIND_TRANSLATION, text=text, parents=[parent],
                              roots=[rid("A", k)], releases=dict(retire), job=JOB_TRANSLATION),
                )
        for j in range(1, lay.n_s + 1):
            for v in range(1, d + 1):
                number = (j - 1) * d + v
                a = ("A", lay.n_x + j)
                c = ("C", (j - 1 + v - 1) % lay.n_s + 1)
                parents = [rid(*a), rid(*c)]
                text = self.summary_text(number, a, c)
                line += 1
                yield self._generated(
                    rid("S", number), JOB_SUMMARY, parents, text,
                    self._row(record=rid("S", number), file=file, line=line, source=None,
                              kind=KIND_SUMMARY, text=text, parents=parents,
                              roots=[rid(*a)], releases=dict(retire), job=JOB_SUMMARY),
                )

    @staticmethod
    def _generated(record: str, job: str, parents: list[str], text: str, row: dict):
        raw = json.dumps(
            {"id": record, "job": job, "handed": parents, "used": parents, "text": text},
            ensure_ascii=False,
        )
        return raw, row


# --- set.toml ----------------------------------------------------------------------


def _toml_list(items: list[str], indent: str = "    ") -> str:
    if not items:
        return "[]"
    if len(items) <= 3 and sum(len(i) for i in items) < 70:
        return "[" + ", ".join(json.dumps(i) for i in items) + "]"
    return "[\n" + "".join(f"{indent}{json.dumps(i)},\n" for i in items) + "]"


def _release_inputs(layout: Layout, clean: bool) -> dict[str, list[str]]:
    """The inputs of R1 and R2, in the order the plant list's outcomes assume."""
    lay = layout
    r1: list[str] = []
    if not clean:
        r1 += [rid("A", k) for k in range(1, lay.n_x + 1)]
    r1 += [rid("B", k) for k in range(1, lay.n_x + 1)]
    if not clean:
        r1 += [rid("P", k) for k in range(1, lay.n_p + 1)]
    r1 += [rid(KEPT_PREFIX, k) for k in range(1, lay.n_kept + 1)]
    if not clean:
        r1 += [rid("T", k) for k in range(1, lay.n_p + 1)]
        r1 += [rid("S", k) for k in range(1, lay.n_sum + 1)]
    r1 += [rid("C", lay.n_s + k) for k in range(1, lay.params.pairs + 1)]
    r2 = [rid("B", k) for k in range(1, lay.lines["B"] + 1)]
    for name in lay.sources:
        if name in ("A", "B", "C"):
            continue
        r1 += [rid(name, k) for k in range(1, lay.z_r1[name] + 1)]
        r2 += [rid(name, k) for k in range(lay.z_r1[name] + 1, lay.lines[name] + 1)]
    return {"R1": r1, "R2": r2}


def render_set_toml(layout: Layout, counts: dict[str, int]) -> str:
    p = layout.params
    names = [n for n in layout.sources if not (p.clean and n == WITHDRAWN)]
    out = [
        "# One generated set, written by fixture/plant/generate.py. Every key is described",
        "# in docs/plant.md. An id names a line: the number of an id is the line number of",
        '# its row in its file, and "A-000001..A-000500" means every id from the first to',
        "# the last.",
        "",
        f"format = {json.dumps(SET_FORMAT)}",
        f"clean = {str(p.clean).lower()}",
        f"withdrawn = {json.dumps(WITHDRAWN)}",
        "",
        "[parameters]",
        f"records = {p.records}",
        f"length = {p.length}",
        f"derivations = {p.derivations}",
        f"sources = {p.sources}",
        f"seed = {p.seed}",
        f"dedup = {str(p.dedup).lower()}",
        f"chunks = {p.chunks}",
        f"pairs = {p.pairs}",
        f"sanitizer_cases = {p.sanitizer_cases}",
        "",
        "[normaliser]",
        f"name = {json.dumps(NORMALISER)}",
        "",
        "[counts]",
    ]
    out += [f"{json.dumps(kind)} = {counts[kind]}" for kind in sorted(counts)]
    out.append("")
    for name in names:
        out += [
            f"[sources.{name}]",
            f'file = "sources/{name}.jsonl"',
            f'member = "{name}/r1"',
            'revision = "r1"',
            f"lines = {layout.lines[name]}",
            f"records = {_toml_list([_range((name, 1), (name, layout.lines[name]))])}",
            "",
        ]
    n_p = 0 if p.clean else layout.n_p
    n_sum = 0 if p.clean else layout.n_sum
    n_kept = layout.n_kept
    # The job "paraphrase" made A's paraphrases and the paraphrases of unrelated
    # records; its ids are listed in the order of its rows in the file.
    paraphrases = [_range(("P", 1), ("P", n_p))] if n_p else []
    if n_kept:
        paraphrases.append(_range((KEPT_PREFIX, 1), (KEPT_PREFIX, n_kept)))
    out += [
        "[generation]",
        'file = "generation.jsonl"',
        'model = "none"',
        f"rows = {2 * n_p + n_sum + n_kept}",
        "",
        "[generation.jobs]",
        f"paraphrase = {_toml_list(paraphrases)}",
        f"translation = {_toml_list([_range(('T', 1), ('T', n_p))] if n_p else [])}",
        f"summary = {_toml_list([_range(('S', 1), ('S', n_sum))] if n_sum else [])}",
        "",
    ]
    inputs = _release_inputs(layout, p.clean)
    for release in ("R1", "R2"):
        out += [
            f"[releases.{release}]",
            f"inputs = {_toml_list(compress(inputs[release]))}",
            f"dedup = {str(p.dedup).lower()}",
            f"dedup_winner = {json.dumps(names[0])}",
            f"dedup_priority = {_toml_list(names)}",
            "",
        ]
    out += ['[layouts."round-robin"]', f"chunks = {p.chunks}", ""]
    return "\n".join(out)


# --- Entry points ------------------------------------------------------------------


def generate(
    out,
    *,
    records: int,
    seed: int,
    length: int = 2000,
    derivations: int = 1,
    sources: int = 4,
    dedup: bool = True,
    chunks: int = 20,
    pairs: int = 300,
    sanitizer_cases: int = 6,
    clean: bool = False,
) -> dict:
    """Write one set into the folder ``out`` and return the run's summary.

    With ``clean`` the same set is written without the withdrawn content: no source
    A, no record derived from it, no plant list (plan 4.4). The generated records
    that do not derive from A, the paraphrases of unrelated records, are written.
    The folder must be empty or absent."""
    params = Params(
        records=records, seed=seed, length=length, derivations=derivations, sources=sources,
        dedup=dedup, chunks=chunks, pairs=pairs, sanitizer_cases=sanitizer_cases, clean=clean,
    )
    layout = plan_layout(params)
    root = Path(out)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise OutputFolderNotEmpty(f"{root.name} already holds files; give an empty or new folder")
    root.mkdir(parents=True, exist_ok=True)

    files = _Files(root)
    builder = _Builder(layout)
    plant = "plant_list.jsonl"
    if not clean:
        files.start(plant)
    written = 0
    for name in layout.sources:
        if clean and name == WITHDRAWN:
            continue
        rel = f"sources/{name}.jsonl"
        files.start(rel)
        for raw, row in builder.source_lines(name):
            if "\n" in raw or "\r" in raw:
                raise SelfCheckFailed(f"the line of {row['id']} holds a line break")
            files.line(rel, raw)
            written += 1
            if not clean:
                files.line(plant, json.dumps(row))
        files.finish(rel)
    files.start("generation.jsonl")
    for raw, row in builder.generation_lines(clean=clean):
        if "\n" in raw or "\r" in raw:
            raise SelfCheckFailed(f"the line of {row['id']} holds a line break")
        if clean and row["withdraw_A"]["in_scope"]:
            raise SelfCheckFailed(f"the clean set would hold {row['id']}, which derives from A")
        files.line("generation.jsonl", raw)
        written += 1
        if not clean:
            files.line(plant, json.dumps(row))
    files.finish("generation.jsonl")
    if not clean:
        files.finish(plant)

    # The clean set is the full set less A's lines and less every generated record
    # that derives from A. The paraphrases of unrelated records stay in it.
    expected = records
    if clean:
        expected -= layout.lines[WITHDRAWN] + layout.n_derived
    counts = dict(builder.counts)
    if counts.get(KIND_KEPT_PARAPHRASE, 0) != layout.n_kept:
        raise SelfCheckFailed(
            f"wrote {counts.get(KIND_KEPT_PARAPHRASE, 0)} paraphrases of unrelated records "
            f"for a set that must hold {layout.n_kept}"
        )
    if written != expected or sum(counts.values()) != expected:
        raise SelfCheckFailed(f"wrote {written} lines for a set that must hold {expected}")
    files.whole("set.toml", render_set_toml(layout, counts))

    listing = sorted(files.stats)
    whole = hashlib.sha256()
    for rel in listing:
        whole.update(f"{rel}\0{files.stats[rel]['sha256']}\n".encode())
    return {
        "format": RUN_FORMAT,
        "parameters": {
            "records": records, "length": length, "derivations": derivations,
            "sources": sources, "seed": seed, "dedup": dedup, "chunks": chunks,
            "pairs": pairs, "sanitizer_cases": sanitizer_cases, "clean": clean,
        },
        "counts": counts,
        "lines_written": written,
        "files": {rel: files.stats[rel] for rel in listing},
        "set_digest": whole.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write one generated set and its plant list (see docs/plant.md)."
    )
    parser.add_argument("--out", required=True, help="an empty or new folder")
    parser.add_argument("--records", type=int, required=True, help="lines in all files together")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--length", type=int, default=2000, help="characters per record, about")
    parser.add_argument("--derivations", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--sources", type=int, default=4, choices=(4, 10))
    parser.add_argument("--dedup", choices=("on", "off"), default="on")
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--sanitizer-cases", type=int, default=6, dest="sanitizer_cases")
    parser.add_argument("--clean", action="store_true",
                        help="write the set without source A and what derives from it")
    args = parser.parse_args(argv)
    try:
        summary = generate(
            args.out, records=args.records, seed=args.seed, length=args.length,
            derivations=args.derivations, sources=args.sources, dedup=args.dedup == "on",
            chunks=args.chunks, pairs=args.pairs, sanitizer_cases=args.sanitizer_cases,
            clean=args.clean,
        )
    except PlantError as exc:
        json.dump({"error": type(exc).__name__, "detail": str(exc)}, sys.stderr)
        sys.stderr.write("\n")
        return exc.exit_code
    json.dump(summary, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
