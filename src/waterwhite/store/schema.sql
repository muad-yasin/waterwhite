-- Waterwhite evidence store, schema version 1 (plan 3.4, Decision 2).
--
-- One SQLite file holds every record the answers rest on. The store module is the
-- only writer. It opens the file with journal_mode=WAL, synchronous=FULL and
-- foreign_keys=ON, and migrates the schema stepwise from the version in `meta`.
--
-- Rules that hold for every table:
-- - no record text is stored, only locators, fingerprints and links (plan 3.1);
-- - a reference that must be able to dangle is stored by value (an instance id or
--   a name as text), so that "named by another record but has no record of its
--   own" can be told from "never registered" (plan 2.1, Decision 26);
-- - coverage is computed on demand from these tables; there is no second
--   per-artifact ledger (plan 3.4).

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
-- Rows written at creation: schema_version = '1'; normaliser = the name and
-- version of the fixed normalisation the fingerprints were taken after (plan 3.1).

CREATE TABLE setting (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
-- Extension switches, one row per extension, value 'on' or 'off' (plan 1.4).
-- A missing row means off.

-- ---------------------------------------------------------------------------------
-- Sources and their members (plan 2.3, 3.1, Decision 24)
-- ---------------------------------------------------------------------------------

CREATE TABLE source (
    source_id INTEGER PRIMARY KEY,
    name      TEXT NOT NULL UNIQUE
);

-- A member is a revision of a source or a part of one (a file, a folder, a list of
-- records). Its parts say which occurrences it covers.
CREATE TABLE member (
    member_id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source(source_id),
    name      TEXT NOT NULL UNIQUE,
    revision  TEXT NOT NULL,
    part_kind TEXT NOT NULL CHECK (part_kind IN ('revision', 'folder', 'file', 'records'))
);

-- locator '' means every record of the file; otherwise one record of it.
CREATE TABLE member_part (
    member_id   INTEGER NOT NULL REFERENCES member(member_id),
    artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
    locator     TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (member_id, artifact_id, locator)
) WITHOUT ROWID;

-- ---------------------------------------------------------------------------------
-- Artifacts (plan 3.1, "Artifact instance"; Decision 29)
-- ---------------------------------------------------------------------------------

CREATE TABLE artifact (
    artifact_id    INTEGER PRIMARY KEY,
    instance_id    TEXT NOT NULL UNIQUE,   -- ids.instance_id or ids.ingest_instance_id
    kind           TEXT NOT NULL CHECK (kind IN
                       ('jsonl', 'shard', 'mixture', 'checkpoint', 'published-model')),
    role           TEXT CHECK (role IN ('source', 'intermediate', 'release', 'import')),
    content_digest TEXT NOT NULL,          -- the digest conditions stay on this digest
    path           TEXT,                   -- where the file was at registration; for a
                                           -- shard the prefix without .bin / .idx
    step_name      TEXT,                   -- producing step; NULL for an ingest or import
    step_version   TEXT,
    param_digest   TEXT,
    member_name    TEXT,                   -- ingest or import: the member registered under
    edge_digest    TEXT NOT NULL           -- ids.edge_digest over the stated edge records
);

-- A name a team gives an artifact. A query names an artifact by one of these or by
-- its instance id. Several names may point at one instance.
CREATE TABLE artifact_name (
    name        TEXT PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id)
) WITHOUT ROWID;

-- The producing step's inputs, in order, by instance id (by value: an input need
-- not be registered).
CREATE TABLE artifact_input (
    artifact_id       INTEGER NOT NULL REFERENCES artifact(artifact_id),
    position          INTEGER NOT NULL,
    input_instance_id TEXT NOT NULL,
    PRIMARY KEY (artifact_id, position)
) WITHOUT ROWID;

-- ---------------------------------------------------------------------------------
-- Bindings: an artifact's own record of the records it holds (plan 3.4)
-- ---------------------------------------------------------------------------------
-- A JSONL artifact has one binding (namespace 'capture') for its occurrence rows.
-- A shard has one binding per namespace and derivation mode for its tokenization
-- map: 'capture' written by the wrapper, 'recovery' written afterwards, and in
-- either namespace at most one map derived by position and one derived by token
-- matching, so the two recovery maps can be stored side by side and compared
-- (DECISIONS.md D-20). A binding and its rows are written in one transaction,
-- refused unless the declared count and digest match the artifact on disk. A
-- binding that exists is whole.
--
-- Which of a shard's maps a reader gets: the first in this order (DECISIONS.md
-- D-20 as amended by D-24).
--   1. verification 'passed' before any other, across both namespaces;
--   2. namespace 'capture' before 'recovery';
--   3. derivation 'position' before 'token-matching'.
-- So among verified maps: the capture map; else the position-mode recovery map;
-- else the token-matching recovery map. Only when no map of the shard is verified
-- does a reader get one that is not, in the same order. Such a map is still the
-- shard's own record, and its label can only be 'unknown'.
--
-- An artifact of kind jsonl or shard with no binding has no record of its own and
-- answers "not covered", "record missing" (plan 2.1).

CREATE TABLE binding (
    binding_id      INTEGER PRIMARY KEY,
    artifact_id     INTEGER NOT NULL REFERENCES artifact(artifact_id),
    namespace       TEXT NOT NULL CHECK (namespace IN ('capture', 'recovery')),
    declared_count  INTEGER NOT NULL,      -- lines of a JSONL file, documents of a shard
    declared_digest TEXT NOT NULL,         -- the artifact's content digest when written
    -- For a tokenization map only (NULL for a JSONL binding):
    derivation      TEXT CHECK (derivation IN ('position', 'token-matching')),
    verification    TEXT CHECK (verification IN ('passed', 'failed', 'not run')),
    label           TEXT CHECK (label IN ('recovered', 'candidates', 'unknown', 'observed',
                                          'mapped', 'file-level')),
    append_eod      INTEGER,               -- the --append-eod setting (F1)
    eod_in_span     INTEGER                -- end-of-document convention: 1 when the
                                           -- EOD token lies inside a document's span
);
-- One binding per artifact, namespace and derivation. The derivation of a JSONL
-- binding is NULL, and NULLs never collide in a UNIQUE constraint, so the rule is
-- an index over the derivation with NULL read as ''.
CREATE UNIQUE INDEX binding_one_per_mode
    ON binding(artifact_id, namespace, COALESCE(derivation, ''));
-- Step 2 (schema_2.sql) replaces this index: from version 2 on, only a map whose
-- verification passed is one per namespace and derivation.
-- `label` is the map's stored label. The API never writes a label stronger than the
-- derivation allows (vocabulary.labels_allowed); the CHECK is wider on purpose, so a
-- label edited afterwards is representable and proof condition 2.2(f) catches it.

-- Records of a JSONL artifact.
CREATE TABLE occurrence (
    occurrence_id INTEGER PRIMARY KEY,
    artifact_id   INTEGER NOT NULL REFERENCES artifact(artifact_id),
    binding_id    INTEGER NOT NULL REFERENCES binding(binding_id),
    locator       TEXT NOT NULL,           -- ids.line_locator
    fingerprint   BLOB,                    -- 16 bytes; NULL for a line with no text
    UNIQUE (artifact_id, locator)
);
CREATE INDEX occurrence_fingerprint ON occurrence(fingerprint);

-- Documents of a tokenized shard: the rows of its map. A shard's occurrence is
-- (shard instance, ordinal); the row links it to the JSONL occurrence it was
-- tokenized from.
CREATE TABLE map_row (
    binding_id           INTEGER NOT NULL REFERENCES binding(binding_id),
    ordinal              INTEGER NOT NULL, -- 0-based document number in the shard
    parent_occurrence_id INTEGER REFERENCES occurrence(occurrence_id),
                                           -- NULL when label is 'unknown' or 'candidates'
    fingerprint          BLOB,
    token_start          INTEGER NOT NULL, -- the document's span in the shard's token
    token_end            INTEGER NOT NULL, -- stream, half-open [start, end)
    label                TEXT NOT NULL CHECK (label IN ('recovered', 'candidates', 'unknown',
                                                        'observed', 'mapped', 'file-level')),
    PRIMARY KEY (binding_id, ordinal)
) WITHOUT ROWID;
-- The walk downward from a JSONL record into the shards that hold it.
CREATE INDEX map_row_parent ON map_row(parent_occurrence_id);

-- The candidate set of a 'candidates' row: the occurrences of the identical twins.
CREATE TABLE map_candidate (
    binding_id    INTEGER NOT NULL,
    ordinal       INTEGER NOT NULL,
    occurrence_id INTEGER NOT NULL REFERENCES occurrence(occurrence_id),
    PRIMARY KEY (binding_id, ordinal, occurrence_id),
    FOREIGN KEY (binding_id, ordinal) REFERENCES map_row(binding_id, ordinal)
) WITHOUT ROWID;
CREATE INDEX map_candidate_occurrence ON map_candidate(occurrence_id);

-- ---------------------------------------------------------------------------------
-- Edges between JSONL occurrences (plan 3.1 "Edge", 3.2, 3.3)
-- ---------------------------------------------------------------------------------

-- The stored log of one generation job run in the closed-input harness (plan 3.3,
-- F10). `model` is the model the job called, 'none' when the log shows it called
-- none, NULL when the log has no model entry (read as "not established").
CREATE TABLE harness_log (
    log_id     INTEGER PRIMARY KEY,
    job_name   TEXT NOT NULL UNIQUE,
    log_digest TEXT NOT NULL,              -- digest of the log file the harness wrote
    model      TEXT
    -- step 2 (schema_2.sql) adds the column harness: which harness wrote the log;
    -- step 4 (schema_4.sql) adds hand_off_checked: whether the handed texts were compared
);

-- The records handed to the job: its input scope.
CREATE TABLE harness_handed (
    log_id        INTEGER NOT NULL REFERENCES harness_log(log_id),
    occurrence_id INTEGER NOT NULL REFERENCES occurrence(occurrence_id),
    PRIMARY KEY (log_id, occurrence_id)
) WITHOUT ROWID;

-- Every read the harness logged, in order. A read of a handed record names its
-- occurrence. A read outside the hand-off names what was asked for in `outside_ref`;
-- `refused` is 1 when the harness refused it (the attempt is logged either way).
CREATE TABLE harness_read (
    log_id        INTEGER NOT NULL REFERENCES harness_log(log_id),
    seq           INTEGER NOT NULL,
    occurrence_id INTEGER REFERENCES occurrence(occurrence_id),
    outside_ref   TEXT,
    refused       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (log_id, seq)
) WITHOUT ROWID;

-- One edge per child occurrence. A recovered edge is written only for a child that
-- has no stated edge.
CREATE TABLE edge (
    edge_id            INTEGER PRIMARY KEY,
    child_id           INTEGER NOT NULL UNIQUE REFERENCES occurrence(occurrence_id),
    transform_name     TEXT NOT NULL,
    transform_version  TEXT NOT NULL,
    param_digest       TEXT NOT NULL,
    parent_mode        TEXT NOT NULL CHECK (parent_mode IN ('all', 'any')),
    derivation         TEXT NOT NULL CHECK (derivation IN
                           ('stated', 'pointer', 'function', 'fingerprint', 'file', 'none')),
    verification       TEXT CHECK (verification IN ('passed', 'failed', 'not run')),
    label              TEXT NOT NULL CHECK (label IN
                           ('observed', 'mapped', 'recovered', 'candidates',
                            'file-level', 'unknown')),
    is_generation      INTEGER NOT NULL DEFAULT 0,
    job_name           TEXT,                                     -- see below
    log_id             INTEGER REFERENCES harness_log(log_id),   -- generation edge in the harness
    bound_established  INTEGER NOT NULL DEFAULT 0,
    -- `job_name` is the name of the generation job that made the child. The edge
    -- keeps it by value, so that an answer can name the job when the job's harness
    -- log is gone (DECISIONS.md D-24, plan Check 5). Every generation edge has one
    -- and no other edge does. For a job run in the harness it equals the job name
    -- of the log the edge points at.
    CHECK ((is_generation = 1 AND job_name IS NOT NULL AND job_name <> '')
           OR (is_generation = 0 AND job_name IS NULL))
    -- `bound_established` is a field the wrapper writes. The proof never trusts it
    -- alone: condition 2.2(g) derives the bound again from the stored log, so a
    -- field edited to 1 with the log removed is caught.
);

-- The records an edge names. `used` = 1: a parent (the sole input, or one the
-- output states it used). `used` = 0: handed to the job and not stated as used, in
-- scope only, not a parent (Decision 25). `selected` = 1 on the one parent an "any"
-- edge's step kept. The parent is referenced by id; a parent row that is gone is
-- the loud fault "edge parent missing" (plan 3.4).
CREATE TABLE edge_input (
    edge_id       INTEGER NOT NULL REFERENCES edge(edge_id),
    position      INTEGER NOT NULL,
    occurrence_id INTEGER NOT NULL,        -- no REFERENCES on purpose: see above
    used          INTEGER NOT NULL DEFAULT 1,
    selected      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (edge_id, position)
) WITHOUT ROWID;
CREATE INDEX edge_input_occurrence ON edge_input(occurrence_id);

-- A link from a record, or from every record of a file (occurrence_id NULL), to a
-- source member with no parent record behind it: file-level evidence (the record's
-- file is known to derive from the source, L1) or a publisher's pointer that names
-- a source. This is where one occurrence can hold a weaker link to a second source
-- beside a strong link to a first (plan 2.2(b), Decision 3).
--
-- A source link stores how it was derived and the result of its verification, and
-- the label rule of the edges and maps holds for it (vocabulary.labels_allowed): a
-- file-level link has the derivation 'file'; a 'mapped' link needs the derivation
-- 'pointer' with verification 'passed'; anything else can only be 'unknown'. The
-- API refuses a stronger label. The CHECK on `label` is wider than the rule, so a
-- label edited afterwards is representable and proof condition 2.2(f) catches it.
CREATE TABLE source_link (
    link_id       INTEGER PRIMARY KEY,
    artifact_id   INTEGER NOT NULL REFERENCES artifact(artifact_id),
    occurrence_id INTEGER REFERENCES occurrence(occurrence_id),
    member_id     INTEGER NOT NULL REFERENCES member(member_id),
    label         TEXT NOT NULL CHECK (label IN ('file-level', 'mapped', 'unknown')),
    evidence      TEXT NOT NULL,           -- what the link rests on, for example the
                                           -- file family of a text-only import
    derivation    TEXT NOT NULL CHECK (derivation IN ('pointer', 'file', 'none')),
    verification  TEXT CHECK (verification IN ('passed', 'failed', 'not run'))
);
CREATE INDEX source_link_artifact ON source_link(artifact_id);

-- An occurrence a step removed (plan 3.1 "Drop"). `artifact_id` and `locator` name
-- the record in the step's input. `group_key` ties a dedup loser to its group.
-- `output_instance_id` is the artifact the step wrote: one step name can drop the
-- same record once for each artifact it writes (two builds of a release from the
-- same inputs), so the output is part of what makes a drop one drop.
CREATE TABLE drop_record (
    drop_id     INTEGER PRIMARY KEY,
    artifact_id INTEGER NOT NULL REFERENCES artifact(artifact_id),
    locator     TEXT NOT NULL,
    reason      TEXT NOT NULL CHECK (reason IN
                    ('sanitizer skip', 'empty line', 'no tokens', 'filter', 'dedup loser')),
    step_name   TEXT NOT NULL,
    output_instance_id TEXT NOT NULL,      -- the artifact the step wrote, by value
    group_key   TEXT,
    UNIQUE (artifact_id, locator, step_name, output_instance_id)
);
CREATE INDEX drop_record_output ON drop_record(output_instance_id);
CREATE INDEX drop_record_group ON drop_record(group_key);

-- ---------------------------------------------------------------------------------
-- Mixtures, runs, traces, checkpoints (plan 3.1, 3.3, 3.5)
-- ---------------------------------------------------------------------------------

-- A mixture snapshot: the shard files the trainer was handed, in its order, with
-- each file's digest (Decision 27). The mixture is an artifact of kind 'mixture';
-- its own record is this snapshot.
CREATE TABLE mixture_shard (
    artifact_id       INTEGER NOT NULL REFERENCES artifact(artifact_id),
    position          INTEGER NOT NULL,    -- the dataset number of the blend
    shard_instance_id TEXT NOT NULL,       -- by value: the shard may be unregistered
    path              TEXT NOT NULL,       -- the shard prefix the trainer was handed
    content_digest    TEXT NOT NULL,
    PRIMARY KEY (artifact_id, position)
) WITHOUT ROWID;

-- The settings a mixture snapshot was registered with, whole, as canonical JSON
-- (ids.canonical). The artifact row holds their digest (`param_digest`), which is
-- part of the mixture's instance id; this row keeps the settings themselves so
-- that they can be read back.
CREATE TABLE mixture_settings (
    artifact_id INTEGER PRIMARY KEY REFERENCES artifact(artifact_id),
    settings    TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE run (
    run_id INTEGER PRIMARY KEY,
    name   TEXT NOT NULL UNIQUE
);

-- A run segment: one mixture, one set of data-order settings, one or more launches.
-- `run_record` is the run record the training wrapper wrote at training time, kept
-- whole as canonical JSON; the columns beside it are read from it at registration.
CREATE TABLE run_segment (
    segment_id            INTEGER PRIMARY KEY,
    run_id                INTEGER NOT NULL REFERENCES run(run_id),
    name                  TEXT NOT NULL UNIQUE,
    position              INTEGER NOT NULL,          -- order inside the run
    mixture_instance_id   TEXT,                      -- by value; NULL = no snapshot (L6)
    parent_checkpoint     TEXT,                      -- instance id of the checkpoint it
                                                     -- starts from, by value; NULL = scratch
    start_kind            TEXT NOT NULL CHECK (start_kind IN ('fresh', 'resume', 'warm-start')),
    run_record            TEXT,                      -- NULL = registered without one
    run_record_digest     TEXT,
    sampler_type          TEXT,                      -- 'single' | 'cyclic'
    sampler_class         TEXT,
    data_sharding         INTEGER,
    micro_batch_size      INTEGER,
    global_batch_size     INTEGER,
    sequence_length       INTEGER,
    add_extra_token       INTEGER,
    random_seed           INTEGER,
    megatron_core_version TEXT,
    megatron_bridge_version TEXT,
    torch_version         TEXT,
    release_tag           TEXT,
    -- How the step traces of this segment were obtained, decided once at
    -- registration (Decision 19): 'cache files', 'replay' or 'none'; with 'none',
    -- `trace_cause` holds the named cause and `trace_cause_detail` its reason.
    trace_route           TEXT NOT NULL CHECK (trace_route IN ('cache files', 'replay', 'none')),
    trace_cause           TEXT,
    trace_cause_detail    TEXT,
    UNIQUE (run_id, position)
);

-- Each launch of the segment (a start or a resume), with its real sizes (M3).
CREATE TABLE segment_launch (
    segment_id          INTEGER NOT NULL REFERENCES run_segment(segment_id),
    position            INTEGER NOT NULL,
    world_size          INTEGER NOT NULL,
    data_parallel_size  INTEGER NOT NULL,
    consumed_at_start   INTEGER NOT NULL,  -- the consumed-sample counter at this start (F4)
    step_at_start       INTEGER NOT NULL,
    first_step          INTEGER NOT NULL,  -- first and last completed step of the launch
    last_step           INTEGER NOT NULL,
    PRIMARY KEY (segment_id, position)
) WITHOUT ROWID;

-- Digests of the data-order cache arrays, recorded at training time (plan 3.5).
-- `shard_position` is NULL for the blend's two arrays.
CREATE TABLE segment_cache_digest (
    segment_id     INTEGER NOT NULL REFERENCES run_segment(segment_id),
    kind           TEXT NOT NULL CHECK (kind IN ('document_index', 'sample_index',
                       'shuffle_index', 'dataset_index', 'dataset_sample_index')),
    shard_position INTEGER,
    digest         TEXT NOT NULL,          -- ids.array_digest
    UNIQUE (segment_id, kind, shard_position)
);

-- Digest of each epoch's shuffle per rank, recorded at training time for a cyclic
-- run (the owner's ruling on the cyclic order, plan 3.5).
CREATE TABLE segment_shuffle_digest (
    segment_id         INTEGER NOT NULL REFERENCES run_segment(segment_id),
    launch_position    INTEGER NOT NULL,
    epoch              INTEGER NOT NULL,
    rank               INTEGER NOT NULL,
    digest             TEXT NOT NULL,      -- ids.order_digest
    PRIMARY KEY (segment_id, launch_position, epoch, rank)
) WITHOUT ROWID;

-- The completed steps of the segment as registered: each step's batch size (the
-- global batch size, or the value of the registered batch schedule, F2) and, only
-- where the operator supplies it, how many tokens of the step's last sample were
-- read. A note can only lower a step's coverage (plan 3.1, 3.5).
CREATE TABLE segment_step (
    segment_id            INTEGER NOT NULL REFERENCES run_segment(segment_id),
    step                  INTEGER NOT NULL,
    batch_size            INTEGER NOT NULL,
    last_sample_tokens_read INTEGER,
    PRIMARY KEY (segment_id, step)
) WITHOUT ROWID;

-- A step trace: what one completed step read, written by the join once, when the
-- run is registered (Decision 19). `consumed_samples` is the trace's own count; a
-- present trace whose count differs from its step's batch size is the loud fault
-- "trace sample count mismatch". A step with no trace row is untraced.
CREATE TABLE step_trace (
    trace_id         INTEGER PRIMARY KEY,
    segment_id       INTEGER NOT NULL REFERENCES run_segment(segment_id),
    step             INTEGER NOT NULL,
    consumed_samples INTEGER NOT NULL,
    UNIQUE (segment_id, step)
);

-- One row per sample the step read. `covered_tokens` is the coverage mark: how many
-- of the sample's tokens the rows cover; the uncovered range runs from there to
-- `sample_tokens`. A step with an uncovered range is untraced (2.2(e)).
CREATE TABLE trace_sample (
    trace_id       INTEGER NOT NULL REFERENCES step_trace(trace_id),
    sample_pos     INTEGER NOT NULL,       -- position inside the step, from 0
    global_index   INTEGER NOT NULL,       -- the sample's index in the blend
    rank           INTEGER NOT NULL DEFAULT 0,
    sample_tokens  INTEGER NOT NULL,       -- sequence_length + 1 with the extra token
    covered_tokens INTEGER NOT NULL,
    PRIMARY KEY (trace_id, sample_pos)
) WITHOUT ROWID;

-- The (shard, ordinal, token start, token end) rows of a sample (plan 3.5). The
-- token range is inside the document, half-open, counted from the document's first
-- token. `shard_position` is the shard's place in the segment's mixture snapshot.
CREATE TABLE trace_row (
    trace_id       INTEGER NOT NULL,
    sample_pos     INTEGER NOT NULL,
    row_pos        INTEGER NOT NULL,
    shard_position INTEGER NOT NULL,
    ordinal        INTEGER NOT NULL,
    token_start    INTEGER NOT NULL,
    token_end      INTEGER NOT NULL,
    PRIMARY KEY (trace_id, sample_pos, row_pos),
    FOREIGN KEY (trace_id, sample_pos) REFERENCES trace_sample(trace_id, sample_pos)
) WITHOUT ROWID;
CREATE INDEX trace_row_document ON trace_row(shard_position, ordinal);

-- A checkpoint is an artifact of kind 'checkpoint': a marker at a completed step.
-- Its own record is its run segment, named by value so that a missing segment is
-- "not covered", "record missing" (plan 2.4).
CREATE TABLE checkpoint (
    artifact_id  INTEGER PRIMARY KEY REFERENCES artifact(artifact_id),
    segment_name TEXT NOT NULL,
    step         INTEGER NOT NULL
) WITHOUT ROWID;

-- The (Megatron version, sampler type, sampler class) combinations for which
-- Check 1 verified the join. A run outside this list answers "may contain" naming
-- "data-order trace unverified for this version" or "... for this sampler type"
-- (F6). The rows are shipped with the tool and written when the store is created.
CREATE TABLE verified_join (
    release_tag   TEXT NOT NULL,
    sampler_type  TEXT NOT NULL,
    sampler_class TEXT NOT NULL,
    PRIMARY KEY (release_tag, sampler_type, sampler_class)
) WITHOUT ROWID;
