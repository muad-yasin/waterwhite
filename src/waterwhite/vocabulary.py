"""The fixed words of the tool: answers, evidence labels, modes, actions, named causes.

Every string a person or a test reads from an answer is defined here once. Modules
import these names and never spell the strings themselves, so a check that compares
an answer with the hand-written table compares against one spelling.

Plan sections: 2.1 (answers), 2.2 (proof conditions), 2.3 (members), 2.4
(checkpoint causes), 3.1 (run segments), 3.2 (labels), 3.3 (derivations), 3.5
(data-order causes and trace routes), 4.1 (actions).
"""

from __future__ import annotations

# --- The four answers (plan 2.1, Decision 8) -------------------------------------

CONTAINS = "contains"
MAY_CONTAIN = "may contain"
DOES_NOT_CONTAIN = "does not contain"
NOT_COVERED = "not covered"

ANSWERS = (CONTAINS, MAY_CONTAIN, DOES_NOT_CONTAIN, NOT_COVERED)


def is_negative_word(word: str) -> bool:
    """True for the word of the negative answer. A module that only reads an answer's
    word asks this and does not spell the word (tests/coverage/test_proof_only_construction.py)."""
    return word == DOES_NOT_CONTAIN

# "not covered" always carries one of these two reasons (plan 2.1, Decision 21). A published model
# the store holds while its extension is off carries "not registered" too (plan 1.4, 3.1, Section 7).
REASON_RECORD_MISSING = "record missing"  # another record names the artifact
REASON_NOT_REGISTERED = "not registered"  # the store has never been told of it

NOT_COVERED_REASONS = (REASON_RECORD_MISSING, REASON_NOT_REGISTERED)

# --- Evidence labels (plan 3.2, Decision 14) ---------------------------------------

OBSERVED = "observed"
MAPPED = "mapped"
RECOVERED = "recovered"
CANDIDATES = "candidates"
FILE_LEVEL = "file-level"
UNKNOWN = "unknown"

LABELS = (OBSERVED, MAPPED, RECOVERED, CANDIDATES, FILE_LEVEL, UNKNOWN)

# --- How a link was derived (plan 3.3, 3.4) ---------------------------------------
# The store derives which labels a link may carry from its derivation; a label
# stronger than the derivation allows is refused at write time and caught by
# proof condition (f) if edited afterwards.

DERIVED_STATED = "stated"  # written by the step itself as it ran (wrapper or job)
DERIVED_POINTER = "pointer"  # a publisher's pointer, verified on the data rows
DERIVED_FUNCTION = "function"  # a deterministic function re-run before matching
DERIVED_FINGERPRINT = "fingerprint"  # a fingerprint match plus direct comparison
DERIVED_POSITION = "position"  # document position (tokenization, position mode)
DERIVED_TOKEN_MATCHING = "token-matching"  # tokens matched to a re-tokenization
DERIVED_FILE = "file"  # the record's file is known to derive from a source
DERIVED_NONE = "none"  # no link recorded

DERIVATIONS = (
    DERIVED_STATED,
    DERIVED_POINTER,
    DERIVED_FUNCTION,
    DERIVED_FINGERPRINT,
    DERIVED_POSITION,
    DERIVED_TOKEN_MATCHING,
    DERIVED_FILE,
    DERIVED_NONE,
)

# Verification result of a derived link or map (plan 3.3).
VERIFIED_PASSED = "passed"
VERIFIED_FAILED = "failed"
VERIFIED_NOT_RUN = "not run"

VERIFICATIONS = (VERIFIED_PASSED, VERIFIED_FAILED, VERIFIED_NOT_RUN)

# Derivations that are inferred by a rule. A link derived this way can carry only
# "recovered", "candidates" or "unknown", and the first two only after its
# verification passed (plan 2.2(f), 3.3).
RULE_DERIVATIONS = (
    DERIVED_FUNCTION,
    DERIVED_FINGERPRINT,
    DERIVED_POSITION,
    DERIVED_TOKEN_MATCHING,
)


def labels_allowed(derivation: str, verification: str | None) -> tuple[str, ...]:
    """The labels a link with this derivation and verification result may carry.

    This is the one statement of the rule of plan 3.4 ("the store refuses a label
    stronger than the derivation allows") and of proof condition 2.2(f).
    """
    if derivation == DERIVED_STATED:
        return (OBSERVED,)
    if derivation == DERIVED_POINTER:
        # A pointer counts as "mapped" only for rows its verification covered.
        return (MAPPED,) if verification == VERIFIED_PASSED else (UNKNOWN,)
    if derivation in RULE_DERIVATIONS:
        if verification == VERIFIED_PASSED:
            return (RECOVERED, CANDIDATES)
        return (UNKNOWN,)
    if derivation == DERIVED_FILE:
        return (FILE_LEVEL,)
    if derivation == DERIVED_NONE:
        return (UNKNOWN,)
    raise ValueError(f"unknown derivation: {derivation!r}")


# Labels that can decide a record-level request (plan 3.2, column "Decides").
# "candidates" decides "contains" only for identical twins that all belong to the
# requested source; the proof handles that case itself.
RECORD_LEVEL_LABELS = (OBSERVED, MAPPED, RECOVERED)

# --- Parent modes (plan 3.1, Decision 15) -----------------------------------------

MODE_ALL = "all"  # every parent was used (a summary of A and C)
MODE_ANY = "any"  # the parents are equivalent copies and one was selected (dedup)

PARENT_MODES = (MODE_ALL, MODE_ANY)

# --- Proof conditions (plan 2.2) ---------------------------------------------------

CONDITION_A = "a"  # every occurrence has a recorded chain back to ingested revisions
CONDITION_B = "b"  # each link's level can decide the request
CONDITION_C = "c"  # every parent named by an edge exists
CONDITION_D = "d"  # every map's digest and record count match its artifact
CONDITION_E = "e"  # every step of a checkpoint's history has a complete trace
CONDITION_F = "f"  # each map's label agrees with its derivation and verification
CONDITION_G = "g"  # the upper bound of each generation edge is established
CONDITION_H = "h"  # generating-model rule (applied only when that extension is on)

CONDITIONS = (
    CONDITION_A,
    CONDITION_B,
    CONDITION_C,
    CONDITION_D,
    CONDITION_E,
    CONDITION_F,
    CONDITION_G,
    CONDITION_H,
)

# --- Named causes a "may contain" can carry (the "names" field of an answer) -------
# Where the plan fixes the words, they are fixed here.

NAMES_TRACE_MISSING = "data-order trace missing"  # plan 3.5, Check 2
NAMES_TRACE_UNVERIFIED_VERSION = "data-order trace unverified for this version"  # F6
NAMES_TRACE_UNVERIFIED_SAMPLER = "data-order trace unverified for this sampler type"  # F6
NAMES_CYCLIC_NOT_RECORDED = "cyclic order not recorded"  # plan 3.5
NAMES_CYCLIC_DIGEST_MISMATCH = "cyclic order digest mismatch"  # plan 3.5
NAMES_SNAPSHOT_MISSING = "mixture snapshot missing"  # L6
NAMES_STEP_UNTRACED = "step untraced"  # 2.2(e): a missing trace or an uncovered range
NAMES_MISSING_ORIGIN = "missing origin"  # 2.2(a), L12: a chain that ends short
NAMES_UPSTREAM_NOT_COVERED = "upstream record missing"  # 2.1: downstream of "not covered"
NAMES_BOUND_NOT_ESTABLISHED = "upper bound not established"  # 2.2(g), L8: names the job
NAMES_IN_SCOPE_ONLY = "in scope only"  # Decision 25: handed, not stated as used
NAMES_LEVEL_TOO_WEAK = "evidence level cannot decide the request"  # 2.2(b)
NAMES_LABEL_STRONGER = "label stronger than derivation"  # 2.2(f)
NAMES_CANDIDATES = "candidate parents"  # 3.2, L3
NAMES_ANCESTOR_MAY_CONTAIN = "ancestor may contain"  # 2.4
NAMES_UNRESOLVED_RECORD = "unresolved record"  # 4.4: a successor that holds one

# The three causes of condition 2.2(h). They are named only while the
# generating-model extension is on (plan 1.4, 3.3, Decision 28); with it off no
# answer carries them and the rows of the table that expect them are not run.
NAMES_MODEL_ENTRY_MISSING = "generating-model entry missing"  # the log has no model entry
NAMES_MODEL_EXPOSED = "generating model exposed"  # the model answers "contains" or "may contain"
NAMES_MODEL_NOT_REGISTERED = "generating model not registered"  # no checkpoint of the store

GENERATING_MODEL_CAUSES = (
    NAMES_MODEL_ENTRY_MISSING,
    NAMES_MODEL_EXPOSED,
    NAMES_MODEL_NOT_REGISTERED,
)

# Every named cause, in one place: what the "names" field of an answer can hold,
# and what a run segment with no trace route can be registered with. A new cause
# is added here as well as above; a test holds the two together.
NAMED_CAUSES = (
    NAMES_TRACE_MISSING,
    NAMES_TRACE_UNVERIFIED_VERSION,
    NAMES_TRACE_UNVERIFIED_SAMPLER,
    NAMES_CYCLIC_NOT_RECORDED,
    NAMES_CYCLIC_DIGEST_MISMATCH,
    NAMES_SNAPSHOT_MISSING,
    NAMES_STEP_UNTRACED,
    NAMES_MISSING_ORIGIN,
    NAMES_UPSTREAM_NOT_COVERED,
    NAMES_BOUND_NOT_ESTABLISHED,
    NAMES_IN_SCOPE_ONLY,
    NAMES_LEVEL_TOO_WEAK,
    NAMES_LABEL_STRONGER,
    NAMES_CANDIDATES,
    NAMES_ANCESTOR_MAY_CONTAIN,
    NAMES_UNRESOLVED_RECORD,
    *GENERATING_MODEL_CAUSES,
)

# The causes a run segment with no trace route can carry, and nothing else (plan
# 2.4, 3.5, F6): the three of plan 3.5 and the two unverified-join causes of F6.
# A cause that names a link of the proof (a step, a record, a bound) belongs to an
# answer and never to a segment.
TRACE_CAUSES = (
    NAMES_TRACE_MISSING,
    NAMES_TRACE_UNVERIFIED_VERSION,
    NAMES_TRACE_UNVERIFIED_SAMPLER,
    NAMES_CYCLIC_NOT_RECORDED,
    NAMES_CYCLIC_DIGEST_MISMATCH,
)

# The reason given beside NAMES_TRACE_MISSING (the "names_detail" field of an
# answer). Plan 3.5 gives two reasons and no third: a run with no recorded digest
# (its caches were never digested, or its cache files and settings are gone, so
# there is nothing a replay could be held against), and a replay whose digest
# differs from the recorded one. Both stay with the replay.
TRACE_MISSING_NO_DIGEST = "no recorded digest"
TRACE_MISSING_DIGEST_MISMATCH = "digest mismatch"

TRACE_MISSING_REASONS = (TRACE_MISSING_NO_DIGEST, TRACE_MISSING_DIGEST_MISMATCH)

# The four reasons the tool adds where it knows what the record, or a file it names,
# lacks and no digest is the question (docs/join.md sections 3 and 11). Each is stored
# beside NAMES_TRACE_MISSING at registration, at the gate that decided it: a setting the
# walk needs is not recorded (the shard list, the micro or global batch size, the
# sequence length, the extra-token flag, a launch); a shard's index file is gone or
# cannot be read; the run was read with the extra token off; a kept cache file holds the
# recorded digest of its kind but an array of a shape no index array of that kind has.
TRACE_MISSING_RECORD_INCOMPLETE = "run record incomplete"
TRACE_MISSING_SHARD_INDEX_UNREADABLE = "shard index unreadable"
TRACE_MISSING_EXTRA_TOKEN_OFF = "extra token off"
TRACE_MISSING_ARRAY_MALFORMED = "index array malformed"

TRACE_MISSING_RECORD_REASONS = (
    TRACE_MISSING_RECORD_INCOMPLETE,
    TRACE_MISSING_SHARD_INDEX_UNREADABLE,
    TRACE_MISSING_EXTRA_TOKEN_OFF,
    TRACE_MISSING_ARRAY_MALFORMED,
)

# Every reason a segment may hold beside NAMES_TRACE_MISSING, and no other.
TRACE_MISSING_STORED_REASONS = TRACE_MISSING_REASONS + TRACE_MISSING_RECORD_REASONS

# --- Notes an answer carries, word for word ----------------------------------------

# Every "contains" whose chain passes a generation edge (plan 2.1, Decision 25).
NOTE_PROCESS_EVIDENCE = (
    "the parent list is process evidence (what the harness handed the job and what "
    "the output states it used), not proof that the output depends on the source"
)
# Every negative whose proof passed a generation edge while the generating-model
# extension is off (plan 2.2(h), 3.3).
NOTE_GENERATING_MODEL_NOT_ASSESSED = "generating-model exposure not assessed"

# Beside "not registered", on the "not covered" answer of a published model the store holds
# while the published-models extension is off (the owner's answer of 18 Nov 2026: say the extension is off).
NOTE_PUBLISHED_MODELS_OFF = "the published-models extension is off"

# The basis every checkpoint "contains" states (plan 2.4, Check 8).
BASIS_WHOLE_SAMPLE = "whole-sample read assumed (F9, L9)"
BASIS_OPERATOR_NOTE = "operator note"

# --- How a hop of an evidence chain is tied to the next (the "link" of a hop) ------
# A hop on an edge or a map row carries that link's evidence label. The other
# three: the last hop of a chain that ends at a record of a requested member; a
# hop that ends a chain at a source link, spelled with the link's label; a step
# of a run that read the next hop's record.

HOP_MEMBER = "member"
HOP_CONSUMED = "consumed"


def hop_source_link(label: str) -> str:
    """The link word of a hop that ends a chain at a source link: "file-level
    link" or "mapped link"."""
    return f"{label} link"


# --- Checkpoint evidence levels (plan 2.4) ----------------------------------------

LEVEL_ELIGIBLE = "eligible"
LEVEL_CONSUMED = "consumed"
LEVEL_INHERITED = "inherited"

# --- Rebuild actions (plan 4.1, Decisions 10 and 15) ------------------------------

ACTION_KEEP = "keep"
ACTION_REWRITE = "rewrite"
ACTION_RE_SOURCE = "re-source"
ACTION_PROMOTE = "promote"
ACTION_RETIRE = "retire"

ACTIONS = (ACTION_KEEP, ACTION_REWRITE, ACTION_RE_SOURCE, ACTION_PROMOTE, ACTION_RETIRE)

# A record whose answer is "may contain" has no action; the report lists it here.
UNRESOLVED = "unresolved"

# --- The rebuild report and the verification of a repair (plan 4.1 to 4.4) ---------
# The words of the report's and the verification's JSON (docs/interfaces.md 7.1 to
# 7.3). The actions above are not changed: a record takes one of the first four
# or the word UNRESOLVED, an artifact one of KEEP and REWRITE or UNRESOLVED.

# The fourth word of an artifact (docs/rebuild.md 2.2): the artifact's own contents are
# clean, and the place of the records that return to the release is not stated, so the
# artifact may receive one. It is not "keep" and does not begin with it: nothing is
# claimed about the bytes of the successor.
ACTION_CLEAN_PLACE_NOT_STATED = "clean, place not stated"
PLACE_NOT_STATED = "not stated"

RECORD_ACTIONS = (ACTION_KEEP, ACTION_RE_SOURCE, ACTION_PROMOTE, ACTION_RETIRE)
ARTIFACT_ACTIONS = (ACTION_KEEP, ACTION_REWRITE, ACTION_CLEAN_PLACE_NOT_STATED)

# The file the caller states a layout's placement of records in (docs/cli.md "report").
FORM_PLACEMENT = "waterwhite.placement/1"
PLACEMENT_BY_SOURCE = "source"

# The format word of each document, written in its key "form".
FORM_REPORT = "waterwhite.report/1"
FORM_VERIFICATION = "waterwhite.verification/1"
FORM_SIDES = "waterwhite.sides/1"

# The name of a successor is this prefix and the name of the artifact it replaces:
# "successor of R1". The replica's name prefix for the repaired side is this text.
SUCCESSOR_PREFIX = "successor of "

# What an unresolved record's review scope names (plan 4.1). The scope is the whole
# file where the evidence is a file-level link, which cannot say which record of
# the file the withdrawn source reaches (plan 3.2, L1), and the record otherwise.
SCOPE_EVERY_RECORD_OF_FILE = "every record of the file"
SCOPE_THE_RECORD = "the record"

# The missing link of an unresolved record whose evidence is a file-level link:
# a sentence with the member the link names, filled with str.format(member=...).
# Any other missing link is given as the "detail" of the answer's failing link.
MISSING_LINK_FILE_LEVEL = (
    "the record's own parent; only the file-level link to {member} is recorded"
)

# Restart points (plan 4.3, Decision 23, F7).
RESTART_BASIS_LAST_CLEAR = (
    "the last checkpoint whose consumed history is completely traced and holds no exposure"
)
RESTART_NOT_PROVEN = "no exposure-free restart is proven"
RESTART_BASIS_LINEAGE_START = (
    "the first step of the lineage, free of exposure by construction"
)
RESTART_BASIS_ORDER_DIFFERS = (
    "the first step at which the corrected order differs from the historical one"
)
RESTART_BASIS_PHASE_START = "the phase start, stated as an assumption"

# What the report says of the exact-rerun restart of a run whose record holds every
# setting a replay needs: the restart is stated by the verification, not by the report
# (plan 4.3; docs/interfaces.md 7.1).
RESTART_EXACT_BY_VERIFY = (
    "stated by the verification of the repair, once the successor shards exist"
)

# Whether the exact-rerun restart rests on a comparison of two data orders (F7).
STEP_COMPARISON_MADE = "made"
STEP_COMPARISON_NOT_POSSIBLE = "not possible"

# How a repaired side is compared with the reference side (plan 4.2, 4.4). The
# multisets and the shard digests are each "equal", "differ" or "not compared";
# the multisets are also "not claimed" for a report that holds an unresolved
# record (plan 4.2), and a repair is then not judged equal or different.
SIDES_EQUAL = "equal"
SIDES_DIFFER = "differ"
SIDES_NOT_COMPARED = "not compared"
SIDES_NOT_CLAIMED = "not claimed"

SIDES_RESULTS = (SIDES_EQUAL, SIDES_DIFFER, SIDES_NOT_COMPARED, SIDES_NOT_CLAIMED)

# Whether every step of the build was seeded (plan 4.2: bytes match only then).
BUILD_SEEDED = "seeded"
BUILD_UNSEEDED = "unseeded"

BUILD_MODES = (BUILD_SEEDED, BUILD_UNSEEDED)

# Where the corrected data order of a stale segment is built from (plan 4.4).
REBUILT_FROM_EMPTY_CACHE = "an empty cache folder"

# The name of the error of a read made against a store that was written after the
# read began (docs/interfaces.md section 2). The class StoreWrittenDuringQuery of
# waterwhite/errors.py states the same words as a literal, as every error name there
# is, so that the replica loads no vocabulary through the errors; a test pins the two
# equal.
ERROR_STORE_WRITTEN_DURING_QUERY = "store written during the query"

# --- Artifact kinds and roles ------------------------------------------------------

KIND_JSONL = "jsonl"
KIND_SHARD = "shard"
KIND_MIXTURE = "mixture"
KIND_CHECKPOINT = "checkpoint"
KIND_PUBLISHED_MODEL = "published-model"  # an extension, off in the first build (1.4)

ARTIFACT_KINDS = (KIND_JSONL, KIND_SHARD, KIND_MIXTURE, KIND_CHECKPOINT, KIND_PUBLISHED_MODEL)

ROLE_SOURCE = "source"  # a file of an ingested source revision
ROLE_INTERMEDIATE = "intermediate"  # a pipeline file between source and release
ROLE_RELEASE = "release"  # a JSONL release
ROLE_IMPORT = "import"  # a text-only import with no recorded producing step

JSONL_ROLES = (ROLE_SOURCE, ROLE_INTERMEDIATE, ROLE_RELEASE, ROLE_IMPORT)

# --- Members of a source (plan 2.3, Decision 24) ----------------------------------
# What a member is a part of. Its parts then say which occurrences it covers.

PART_REVISION = "revision"  # a whole revision of the source
PART_FOLDER = "folder"  # the files of one folder of a revision
PART_FILE = "file"  # one file
PART_RECORDS = "records"  # a list of single records

PART_KINDS = (PART_REVISION, PART_FOLDER, PART_FILE, PART_RECORDS)

# --- Why a step dropped a record (plan 3.1, "Drop") -------------------------------

DROP_SANITIZER_SKIP = "sanitizer skip"  # unparseable line, read from the step's log
DROP_EMPTY_LINE = "empty line"
DROP_NO_TOKENS = "no tokens"  # a line that gives no tokens adds no document (F1)
DROP_FILTER = "filter"
DROP_DEDUP_LOSER = "dedup loser"

DROP_REASONS = (
    DROP_SANITIZER_SKIP,
    DROP_EMPTY_LINE,
    DROP_NO_TOKENS,
    DROP_FILTER,
    DROP_DEDUP_LOSER,
)

# The two reasons a sanitizer's own log gives for a line it wrote nothing for
# (plan 3.3 "Deterministic rewrite"). A drop with one of them is a sanitizer's
# drop, whatever its step is called. Such a line is recorded once, as a drop
# whose output is the first registered artifact that lacks the line.
SANITIZER_DROP_REASONS = (DROP_SANITIZER_SKIP, DROP_EMPTY_LINE)

# The step names the recovery of a sanitizer and the tokenization maps write
# when the caller names no other, so that the drops of one sanitizer carry one
# step name whichever of the two recorded them.
STEP_SANITIZE = "sanitize"
STEP_TOKENIZE = "tokenize"

# --- Which harness wrote a harness log (plan 3.3 "Generation", F10) ----------------
# The in-process harness logs the reads that passed its context and cannot say
# what else the job opened; the confined harness runs the job where the pipe to
# the harness is its one way to a record. A log states which of the two wrote it.

HARNESS_IN_PROCESS = "in-process"
HARNESS_CONFINED = "confined"

HARNESS_KINDS = (HARNESS_IN_PROCESS, HARNESS_CONFINED)

# --- How far the links of a recovery run reach (plan 3.3, F8) ----------------------

OUTCOME_RECORD_LEVEL = "record-level"  # every link was derived per record
OUTCOME_FILE_LEVEL = FILE_LEVEL  # degraded: the file is tied to a member, no record is
OUTCOME_UNKNOWN = UNKNOWN  # degraded: nothing is known about the records

RECOVERY_OUTCOMES = (OUTCOME_RECORD_LEVEL, OUTCOME_FILE_LEVEL, OUTCOME_UNKNOWN)

# --- Samplers and map namespaces ---------------------------------------------------

SAMPLER_SEQUENTIAL = "single"  # Megatron's dataloader_type for the sequential sampler
SAMPLER_CYCLIC = "cyclic"

SAMPLER_TYPES = (SAMPLER_SEQUENTIAL, SAMPLER_CYCLIC)

NAMESPACE_CAPTURE = "capture"  # written by the wrapper as the step ran
NAMESPACE_RECOVERY = "recovery"  # written afterwards; never overwrites a capture map

NAMESPACES = (NAMESPACE_CAPTURE, NAMESPACE_RECOVERY)

# --- Run segments (plan 3.1, 3.5, F4) ---------------------------------------------
# How a segment starts. A fresh start has no parent checkpoint. A resume goes on
# from a checkpoint and keeps its counters. A warm start loads a checkpoint's
# weights and restarts the step count and the consumed-sample counter.

START_FRESH = "fresh"
START_RESUME = "resume"
START_WARM = "warm-start"

START_KINDS = (START_FRESH, START_RESUME, START_WARM)

# The format word of a run record (docs/interfaces.md 8.1).
RUN_RECORD_FORMAT = "waterwhite.run-record/1"

# How the step traces of a segment were obtained, decided once at registration
# (plan 3.5, Decision 19). With ROUTE_NONE the segment names its cause.
ROUTE_CACHE_FILES = "cache files"  # read from the data-order cache files the run kept
ROUTE_REPLAY = "replay"  # the index build replayed, its digests equal to the recorded ones
ROUTE_NONE = "none"  # no trace: the segment carries a named cause

TRACE_ROUTES = (ROUTE_CACHE_FILES, ROUTE_REPLAY, ROUTE_NONE)

# The five kinds of data-order cache file (plan 3.5): three per shard, two per blend.
CACHE_DOCUMENT_INDEX = "document_index"
CACHE_SAMPLE_INDEX = "sample_index"
CACHE_SHUFFLE_INDEX = "shuffle_index"
CACHE_DATASET_INDEX = "dataset_index"
CACHE_DATASET_SAMPLE_INDEX = "dataset_sample_index"

CACHE_KINDS_PER_SHARD = (CACHE_DOCUMENT_INDEX, CACHE_SAMPLE_INDEX, CACHE_SHUFFLE_INDEX)
CACHE_KINDS_PER_BLEND = (CACHE_DATASET_INDEX, CACHE_DATASET_SAMPLE_INDEX)


def cache_keys(shard_count: int) -> list[tuple[str, int | None]]:
    """The index arrays a segment of this many shards has, as (kind, shard position)
    with None for the blend: three per shard and, from two shards on, two for the
    blend (plan 3.5). The one place that says it: the join and the store both ask."""
    keys: list[tuple[str, int | None]] = [
        (kind, position) for position in range(shard_count) for kind in CACHE_KINDS_PER_SHARD
    ]
    if shard_count >= 2:
        keys += [(kind, None) for kind in CACHE_KINDS_PER_BLEND]
    return keys


def shard_count_of(listed: int, digest_positions) -> int:
    """How many shards a run record's segment has: the length of its shard list or,
    when it lists none, one more than the highest shard position a cache digest
    names (0 when neither says). The join and the store read a record the same way."""
    return listed or 1 + max((p for p in digest_positions if p is not None), default=-1)
