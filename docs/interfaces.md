# Interfaces

This file fixes the contracts between the parts of the repository. A module is
written against this file and the code under `src/waterwhite/` that it names
(`vocabulary.py`, `errors.py`, `ids.py`, `records.py`, `extensions.py`,
`store/schema.sql`). Where this file and that code disagree, the code wins and this
file is corrected. "The plan" is the build plan; it is cited by section number
(for example "plan 2.2(e)") and is not copied here. `DECISIONS.md` holds the
choices made while building, each as an entry `D-<nn>`.

## 1. Layout, and the wall between the tool and its expected answers

| Path | What it is | May import |
|---|---|---|
| `src/waterwhite/` | the tool | the standard library, numpy |
| `src/waterwhite/rebuild/` | the report of a repair and its verification (sections 7.1 to 7.4) | the standard library, numpy and the tool's other packages; it reads answers and verdicts and never builds a "does not contain" answer or the verdict CLEAR, which only the coverage package and `answers.py` do |
| `replica/` | the pipeline replica (plan 5.2): test infrastructure that stands in for a team's pipeline, builder and training wrapper | the standard library, numpy, and the tool's public packages for a pipeline: `waterwhite.capture`, `waterwhite.harness`, `waterwhite.normalise`, `waterwhite.store.Store`, `waterwhite.vocabulary`, `waterwhite.errors`, `waterwhite.tokenmap`, `waterwhite.recovery`, `waterwhite.confine` and `waterwhite.ids` |
| `oracle/` | the test-time oracle (plan 3.5, 5.3): runs Megatron's own dataset and sampler code and logs what each step reads | Megatron (test-time only), numpy; never `waterwhite` |
| `fixture/sources/` | the self-authored texts of the hand-checked fixture (plan 5.1, W3) | |
| `fixture/spec/` | the fixture spec (plan 5.1) | |
| `fixture/table/` | the hand-written table: every expected answer of the hand-checked fixture | |
| `fixture/plant/` | the plant generator and the counting script for the 10k and 100k sets | the standard library only; never `waterwhite`, `replica` or `oracle` |
| `tests/` | the checks | everything above |
| `tools/` | developer scripts (the Megatron environment) | |

What the replica imports of the tool, and when:

- `waterwhite.normalise` in every mode, with capture on or off. Its sanitizer is
  the normaliser `control-chars/1`, and that function is defined once, in
  `waterwhite.normalise` (DECISIONS.md D-17). The replica carries no copy of it.
- `waterwhite.capture`, `waterwhite.harness`, `waterwhite.store.Store` and
  `waterwhite.vocabulary` in a build only with capture on, and only in its capture
  hooks (`replica/capture_hooks.py`). The hooks take two role words from
  `waterwhite.vocabulary`, `ROLE_INTERMEDIATE` and `ROLE_RELEASE`.
- `waterwhite.capture`, `waterwhite.store.Store`, `waterwhite.recovery` and
  `waterwhite.tokenmap` at the top of the modules that register files after a build
  made without capture (`replica/recover.py`, `replica/pipeline_links.py`) and that
  build the table's two scratch stores (`replica/scratch.py`): they ingest and import
  files, name steps and take file digests with the tool's capture functions, as a
  team's own recovery would. A build with capture off does not load them.
- `waterwhite.tokenmap` at its tokenization capture point (the capture map,
  section 5.1), and for a run made without capture, afterwards, together with
  `waterwhite.recovery` (sections 5.1 and 5.2).
- `waterwhite.confine` where a generation job runs out of process (section 5.3).
- `waterwhite.vocabulary`, `waterwhite.errors` and `waterwhite.store.store` in
  `replica/probe_run.py` (the probe-run command), at the top of the module and outside
  the capture hooks. The command is a pipeline step that registers what it runs: it
  checks the store's list of verified joins, registers the mixture through
  `Store.register_mixture` and starts the tool's `register-run`, so it reads the words
  of an artifact kind, catches the tool's named errors and opens the store. It reads no
  answer, no report and no verdict, and imports nothing of `waterwhite.rebuild`,
  `waterwhite.coverage` or `waterwhite.answers`.
- `waterwhite.ids` in `replica/megatron_runs_worker.py`, inside two of its functions
  (`_cache_digests` and `_shuffle_digests`): the worker records the digests of the index
  arrays Megatron wrote and of the shuffle order the way the run record holds them
  (`ids.array_digest`, `ids.order_digest`). It reads no answer, no report and no verdict.
- The package `waterwhite` itself in `replica/runs.py`, only to find the folder of the
  tool's source (`waterwhite.__file__`).
- Nothing of `waterwhite.rebuild`, `waterwhite.coverage` or `waterwhite.answers`. The
  replica reads the report that `waterwhite report` printed, and writes the sides
  file that `waterwhite verify` reads, as plain JSON files with the `json` module
  (sections 7.1, 7.2 and 9.5). A report is therefore a file the replica was
  handed, which a test can change before the repaired side is built.

A pipeline with capture on needs `waterwhite.capture`, `waterwhite.harness`,
`waterwhite.normalise` and `waterwhite.store.Store`, and at its tokenization step
`waterwhite.tokenmap`; a run made without capture is recovered afterwards with
`waterwhite.tokenmap` and `waterwhite.recovery`; `waterwhite.confine` only where a
generation job runs out of process. `waterwhite.capture` names the record types,
id functions and words a pipeline writes (section 5), the role words among them,
so the replica's import of `waterwhite.vocabulary` adds no word that
`waterwhite.capture` does not hand out.

Inside the tool three things are defined once and used by every package:

- the text a line holds: `capture.line_text(raw, text_field)`;
- where a registered artifact's file lies: `Store.disk_path(instance_id)`;
- the shard format: `waterwhite.store.artifact_files` reads the index
  (`read_shard_index`), names the token type of a dtype code (`token_dtype`) and
  reads one document's tokens (`ShardTokens`).

The wall (plan W5): an expected answer never comes from the tool's own code. So
`oracle/` and `fixture/plant/` import nothing from `waterwhite`; the fixture spec
and the table are written by hand from the plan, the replica's manifests and the
oracle's log, and never from a `waterwhite` output; and they are committed before
the tool module they grade.

No Megatron code is copied into this repository (plan 3.5). No text of a corpus
that forbids redistribution, and no real benchmark item, is committed (plan W3).

## 2. Words and errors

Every answer word, label, mode, action and named cause is a constant in
`waterwhite/vocabulary.py`. Code imports them. The hand-written table spells the
same strings. Beside the words of the answers, the vocabulary holds:

- the start kinds of a run segment: `START_FRESH` "fresh", `START_RESUME`
  "resume", `START_WARM` "warm-start";
- the trace routes: `ROUTE_CACHE_FILES` "cache files", `ROUTE_REPLAY` "replay",
  `ROUTE_NONE` "none";
- the part kinds of a member: `PART_REVISION` "revision", `PART_FOLDER` "folder",
  `PART_FILE` "file", `PART_RECORDS` "records";
- the two reasons beside "data-order trace missing" (plan 3.5 gives two and no
  third): `TRACE_MISSING_NO_DIGEST` "no recorded digest" and
  `TRACE_MISSING_DIGEST_MISMATCH` "digest mismatch". A run whose cache files and
  settings are gone has nothing a replay could be held against, and carries the
  reason "no recorded digest". Both stay with the replay;
- the four reasons the tool adds beside "data-order trace missing", where it knows
  what the record, or a file it names, lacks and no digest is the question,
  `TRACE_MISSING_RECORD_REASONS`: `TRACE_MISSING_RECORD_INCOMPLETE` "run record
  incomplete" (a setting the join's walk needs is not recorded: the shard list, the
  micro or global batch size, the sequence length, the extra-token flag, or a
  launch), `TRACE_MISSING_SHARD_INDEX_UNREADABLE` "shard index unreadable" (a shard's
  index file is gone or cannot be read at registration),
  `TRACE_MISSING_EXTRA_TOKEN_OFF` "extra token off" (the run was read with the extra
  token off) and `TRACE_MISSING_ARRAY_MALFORMED` "index array malformed" (a kept cache
  file holds the recorded digest of its kind, but an array of a shape no index array of
  that kind has). The join stores the word at registration, at the gate that decided it.
  `TRACE_MISSING_STORED_REASONS` holds all six words. The store accepts beside
  "data-order trace missing" one of these six or none (a caller that does not know
  the reason stores none), and no reason beside any other cause;
- the three named causes of the generating-model rule, which an answer carries
  only while that extension is on (plan 2.2(h)): `NAMES_MODEL_ENTRY_MISSING`
  "generating-model entry missing", `NAMES_MODEL_EXPOSED` "generating model
  exposed", `NAMES_MODEL_NOT_REGISTERED` "generating model not registered";
- `NAMED_CAUSES`, the tuple of every named cause (every `NAMES_*` constant), and
  `TRACE_CAUSES`, the five of them a run segment with no trace route can carry
  (plan 2.4, 3.5, F6): "data-order trace missing", "data-order trace unverified for
  this version", "data-order trace unverified for this sampler type", "cyclic order
  not recorded", "cyclic order digest mismatch". A cause that names a link of the
  proof (a step, a record, a bound) belongs to an answer and never to a segment;
- `cache_keys(shard_count)`, the index arrays a segment of that many shards has
  (three per shard and, from two shards on, two for the blend), and
  `shard_count_of(listed, positions)`, how many shards a run record's segment has:
  the length of its shard list or, with none listed, one more than the highest shard
  position a cache digest names. The join and the store both ask these two, so the
  store never refuses a route the join produced;
- the words that tie a hop of a chain to the next where the tie is not an evidence
  label: `HOP_MEMBER` "member", `HOP_CONSUMED` "consumed", and
  `hop_source_link(label)`, which gives "file-level link" and "mapped link";
- the reasons of a sanitizer's own drops, `SANITIZER_DROP_REASONS` ("sanitizer
  skip", "empty line"), and the step names `STEP_SANITIZE` "sanitize" and
  `STEP_TOKENIZE` "tokenize" that the recovery of a sanitizer and the
  tokenization maps write when a pipeline names no other (section 5.4);
- which harness wrote a harness log: `HARNESS_IN_PROCESS` "in-process",
  `HARNESS_CONFINED` "confined" (`HARNESS_KINDS`);
- how far the links of a recovery run reach: `OUTCOME_RECORD_LEVEL`
  "record-level", `OUTCOME_FILE_LEVEL` "file-level", `OUTCOME_UNKNOWN` "unknown"
  (`RECOVERY_OUTCOMES`).
- the words of the rebuild report and the verification of a repair (sections 7.1
  to 7.3), none of which is an answer word: the actions of a record
  (`RECORD_ACTIONS`: "keep", "re-source", "promote", "retire") and of an artifact
  (`ARTIFACT_ACTIONS`: "keep", "rewrite"), each beside `UNRESOLVED` "unresolved";
  `ACTIONS` stays the five words of plan 4.1; the format words of the three
  documents, `FORM_REPORT` "waterwhite.report/1", `FORM_VERIFICATION`
  "waterwhite.verification/1" and `FORM_SIDES` "waterwhite.sides/1";
  `SUCCESSOR_PREFIX` "successor of "; the two words of a review scope,
  `SCOPE_EVERY_RECORD_OF_FILE` "every record of the file" and `SCOPE_THE_RECORD` "the
  record"; `MISSING_LINK_FILE_LEVEL`, the sentence of a missing link whose evidence is
  a file-level link, with a place for the member; the restart words
  `RESTART_BASIS_LAST_CLEAR`, `RESTART_NOT_PROVEN` "no exposure-free restart is
  proven", `RESTART_BASIS_LINEAGE_START`, `RESTART_BASIS_ORDER_DIFFERS` and
  `RESTART_BASIS_PHASE_START`, and beside them `STEP_COMPARISON_MADE` "made" and
  `STEP_COMPARISON_NOT_POSSIBLE` "not possible"; the results of a comparison of the
  two sides, `SIDES_EQUAL` "equal", `SIDES_DIFFER` "differ", `SIDES_NOT_COMPARED`
  "not compared" and `SIDES_NOT_CLAIMED` "not claimed" (`SIDES_RESULTS`); the two
  modes of a build, `BUILD_SEEDED` "seeded" and `BUILD_UNSEEDED` "unseeded"
  (`BUILD_MODES`); and `REBUILT_FROM_EMPTY_CACHE` "an empty cache folder". The named
  cause `NAMES_UNRESOLVED_RECORD` stays a constant and no answer carries it in this
  build: the answer of an artifact that holds an unresolved record names the link
  of the proof that failed, as every "may contain" does (section 7.3).

Every error name is a class in `waterwhite/errors.py`, whichever module raises
it. A module imports the class from there and defines none of its own. The name
is fixed; the detail after it is free text.

| Family | Exit code | Names |
|---|---|---|
| refused request (`Refused`) | 2 | "source not registered", "member not registered" (`MemberNotRegistered`), "record not in source" (`RecordNotInSource`), "extension off", "unsupported selector", "unsupported format", "ambiguous artifact", "not a store" (`NotAStore`), "store of a newer schema version" (`NewerSchemaVersion`), "read outside the hand-off" (`HarnessRefused`), "temporal ancestry cycle" (`TemporalAncestryCycle`): the answer would rest on a cycle through a model's own history, "store written during the query" (`StoreWrittenDuringQuery`): the store changed, by this process or by another connection, between the moment a query began and the moment its answer was to be made; nothing is answered (below) |
| loud integrity fault (`IntegrityFault`) | 3 | "map count mismatch", "map digest mismatch", "edge parent missing", "trace sample count mismatch", "artifact digest mismatch", "stored digest mismatch", "locator domain mismatch", "cache digest conflict", "shard files not checked" (a negative answer rests on a shard whose files are away and no check of them has passed, docs/cli.md "check") |
| refused write (`WriteRefused`) | 4 | "declared count mismatch", "declared digest mismatch", "label stronger than derivation", "invalid record", "unknown normaliser" (`UnknownNormaliser`), "a file already exists at the store path" (`StoreFileExists`), "drop already recorded" (`DropAlreadyRecorded`): a sanitizer's log handed for a line whose skip the store holds under another artifact (section 5.4), "torch not available" (`TorchUnavailable`): a run read with the cyclic sampler is registered and its shuffle must be computed, and torch cannot be imported (section 8.5); nothing is written |
| no family | 1 | "usage" (`UsageError`): a command line the tool cannot read or will not carry out, such as `init --verified-join` naming a join outside the shipped list (docs/cli.md, "init"); "no model client" (`ModelClientMissing`): a job called a model and the harness was given no client; the three errors of the confined harness (section 5.3): "sandbox unavailable" (`SandboxUnavailable`), "confined job failed" (`ConfinedJobFailed`), "job protocol violation" (`JobProtocolViolation`); "tokenizer failed" (`TokenizerFailed`): the team's tokenizer, run as a command or loaded by name for a map recovery, could not be started, ended early or answered with something that is not token ids; "replay not possible" (`ReplayNotPossible`): the exposure join was asked to derive the index arrays again and the run record lacks a setting the replay needs (the join turns it into a cause and never lets it out of a registration; `waterwhite replay` lets it out, and it also names a record that lists no shard or a shard whose index file is not there, cannot be read or is not the file the record's digest names) |

Of the refusals of a withdrawal the plan names one, "source not registered" (plan
2.3). "member not registered" and "record not in source" are additions that the
plan leaves to the builder (plan 5.8, B-7); DECISIONS.md D-26 records them.

**The store was written during the query.** A coverage session, an integrity pass
and a proof each hold for one state of the store (DECISIONS.md D-45, D-46, D-48).
When the store is written after a session began, or after the proof an answer is to
be made from, the library raises `StoreWrittenDuringQuery`, a refusal of the family
"request refused" with the exit code 2 and the name "store written during the
query". The detail says that the store changed and that the command can be run
again. The class lives in `waterwhite/errors.py` like every other and states its name
as a literal like every other, importing nothing of the vocabulary (the replica with
capture off loads no vocabulary; the vocabulary constant
`ERROR_STORE_WRITTEN_DURING_QUERY` holds the same words, and a test pins the two
equal). It replaces the `ValueError` that `coverage.Session` and `prove` raised for a
write ("the store was written after this session began"), and it replaces the
`ProofRequired` that the answer type raised for a write after prove's last check and
before the answer was built: the answer type checks a proof it minted against the
store's mark, and a proof the store has outlived ends in `StoreWrittenDuringQuery`,
whichever way the answer is built. Which is which is decided by what the proof is: a
proof the mint did not make (forged, or of another artifact or another request, a name
that is no name of the artifact, other notes) is `ProofRequired` whatever the store
holds, because the checks of the proof's identity come first; a proof the mint made,
whose store is open and was written, is `StoreWrittenDuringQuery`; and a proof whose
store is closed or gone stays `ProofRequired`, since a closed store cannot say whether
it was written and using it is a mistake of the caller (DECISIONS.md D-75). A session
that belongs to another store or another request stays as it is, for the same reason.
Every command that reads through a session ends with the JSON error on standard error
and no output on standard output: `query`, `report` and `verify`. Each of them also takes
the store's mark as it opens the store and compares it once more after its last answer
is built and before anything is printed, so a write by another connection in any window
of the command, between prove's last check and the answer, between two answers, or
after the last answer, ends in the same error (DECISIONS.md D-76); a write after the
output is printed is outside the command. The code is 2 and not
3, because plan 2.2 closes the list of loud faults at four and this is none of
them, and not 4, because the command wrote nothing; it is not 1, because the command
line was right. Nothing is retried by the tool: a wrongly timed read is the caller's
to repeat.

A refusal that has no name of its own carries the name of its family: "request
refused", "integrity fault" or "write refused". `report` and `verify` give no other
new name: a report or a sides file that cannot be read is "usage" (exit 1), as a run
record is, and one that is JSON but not a document of its form, or that names a file
which is not there, is "request refused" (exit 2) with the reason in the detail. So is
a report that names, by its instance id, an artifact the store handed to `verify` does
not hold (section 7.3): that refusal has no name of its own and adds none to the table.
Section 7 and `docs/cli.md` say where the command line gives one.

Exit codes: 0 answered; 1 usage; 2 request refused (named error, no answer);
3 loud integrity fault (named mismatch, no answer); 4 write refused; 5 `verify
--require-held` only: the verification was made and printed, and its conditions are
not all held.

## 3. Identity

All in `waterwhite/ids.py`.

- Content digest: SHA-256 hex. A JSONL artifact: of the file. A shard:
  `shard_digest(sha256(.bin), sha256(.idx))`. `ContentHasher` takes the same digest
  as the bytes go by.
- A line of a JSONL file is defined once, by `split_lines`, and the store and the
  capture wrappers both use it: a line ends at the byte 0x0A and at nothing else
  (a carriage return ends no line); the bytes after the last line feed are a line
  when there is at least one of them; an empty file has 0 lines; an empty line is
  a line. `read_lines(path, hasher)` reads a file once and gives its lines while
  the hasher takes its digest, so a file is digested and split in one read.
- Fingerprint: BLAKE2b, 16 bytes, of the record's text after the store's
  normaliser.
- Locator: `L<n>` is line n of a JSONL file, counted from 1; `D<n>` is document n
  of a shard, counted from 0.
- Instance id (plan 3.1, Decision 29): for an ingested file or an import,
  `ingest_instance_id(content digest, member name)`; for anything a step produced,
  `instance_id(content digest, step name, step version, parameter digest, input
  instance ids in order, edge digest)`.
- Edge digest: over the edge records the producing step stated as it ran, each as
  `edge_record(child locator, parents, selected parent)`. A link derived afterwards
  (a recovered edge, a tokenization map, a source link) is not part of an
  artifact's identity (DECISIONS.md D-05).
- The instance id `""` (the empty string, `records.OWN_ARTIFACT`) names the
  artifact itself. It is written in a reference handed to the store together with
  an artifact's own records (a parent of an edge, a handed record or a read of a
  harness log) when the record meant lies in the artifact being written, whose
  instance id does not exist yet. The edge digest names such a parent by its
  locator alone. The store puts the real id in its place; a reference read back
  from the store always carries a full instance id.
- A mixture and a checkpoint have no file. `mixture_instance_id(shard instance
  ids in order, shard content digests in order, settings)` and
  `checkpoint_instance_id(segment, step)` give their ids. A checkpoint's id
  depends on nothing else, so a segment can name its parent checkpoint before that
  checkpoint is registered.
- A path is where a file lay when it was registered. It is part of no id.
- An artifact has one or more names in the store. A query names an artifact by a
  name or by its instance id. A query by path or content digest that matches more
  than one instance is refused with "ambiguous artifact".

## 4. The store (`waterwhite.store`)

One SQLite file, schema in `store/schema.sql`, the store module its only writer.
The package exports one class: `from waterwhite.store import Store`. Its write
methods are fixed here so that capture code can be written beside it; its read
methods are the store's own design and are documented in `docs/store.md`.

```python
class Store:
    @classmethod
    def create(cls, path, *, normaliser: str,
               verified_joins: Iterable[tuple[str, str, str]] = ()) -> "Store": ...
                                    # (release tag, sampler type, sampler class), plan F6
    @classmethod
    def open(cls, path, *, readonly: bool = False) -> "Store": ...
    def close(self) -> None: ...

    @property
    def normaliser(self) -> str: ...     # the name the store was created with
    def disk_path(self, instance_id: str) -> Path | None: ...
                                         # where the artifact's file lies now; a
                                         # shard's prefix; None with no row or path

    # settings (extension switches)
    def settings(self) -> dict[str, str]: ...
    def set_setting(self, key: str, value: str) -> None: ...

    # sources and members (plan 2.3)
    def add_source(self, name: str) -> None: ...                       # idempotent
    def add_member(self, source: str, name: str, revision: str, part_kind: str,
                   parts: Sequence[tuple[str, str]]) -> None: ...      # (instance id, locator or "")

    # artifacts; each call is one transaction and returns the instance id
    def register_jsonl(self, path, *, name: str, role: str,
                       step: StepRef | None, member_name: str | None,
                       inputs: Sequence[str],
                       occurrences: Sequence[OccurrenceRecord],
                       edges: Sequence[EdgeRecord] = (),
                       drops: Sequence[DropRecord] = (),
                       declared_count: int, declared_digest: str) -> str: ...
    def register_shard(self, prefix, *, name: str, step: StepRef,
                       inputs: Sequence[str]) -> str: ...              # the artifact row only
    def write_map(self, shard_instance_id: str, map: MapRecord, *,
                  declared_count: int, declared_digest: str,
                  drops: Sequence[DropRecord] = ()) -> None: ...
    def write_maps(self, shard_instance_id: str, maps: Sequence[MapRecord], *,
                   declared_count: int, declared_digest: str,
                   drops: Sequence[DropRecord] = ()) -> None: ...   # one transaction
    def register_mixture(self, *, name: str, shards: Sequence[SnapshotShard],
                         settings: dict) -> str: ...                   # the settings are kept whole
    def register_run_segment(self, *, run: str, segment: str, position: int,
                             mixture_instance_id: str | None,
                             parent_checkpoint: str | None, start_kind: str,
                             run_record: dict | None,
                             steps: Sequence[tuple[int, int, int | None]],
                             trace_route: str, trace_cause: str | None = None,
                             trace_cause_detail: str | None = None,
                             traces: Sequence[StepTrace] = (),
                             checkpoints: Sequence[tuple[str, int]] = ()) -> None: ...  # one transaction
    def write_step_traces(self, segment: str, traces: Sequence[StepTrace]) -> None: ...
    def register_checkpoint(self, *, name: str, segment: str, step: int) -> str: ...

    # links and drops added after registration; never change an instance id
    def add_recovered_edges(self, instance_id: str, edges: Sequence[EdgeRecord]) -> None: ...
    def add_drops(self, instance_id: str, drops: Sequence[DropRecord]) -> None: ...
    def add_source_link(self, instance_id: str, locator: str | None, member: str,
                        label: str, evidence: str, derivation: str,
                        verification: str | None = None) -> None: ...

    # the loud checks, implemented once, here (plan 3.4)
    def check_integrity(self, instance_ids: Iterable[str] | None = None, *,
                        models: bool = False) -> None: ...   # models=True adds the generating models' ancestry to the closure
```

Rules the store enforces:

- **Opening.** `create` on a path that holds a file is refused with "a file
  already exists at the store path". `open` on a path that holds no store is "not
  a store"; on a store of a newer schema version it is "store of a newer schema
  version". `verified_joins` are written when the store is created and at no
  other time; the library writes the ones it is given, and `init` gives a store the
  joins that Check 1 verified, shipped with the tool, or, with `--verified-join`, a
  subset of them: `init` refuses a join outside the shipped list (docs/cli.md,
  "init"). A store made through the library with its caller's own list is outside what
  the tool verified.
- **Declared against disk.** `register_jsonl` and `write_map` read the artifact on
  disk and refuse with `DeclaredCountMismatch` or `DeclaredDigestMismatch` unless
  the declared count and digest match it. A refused or interrupted write leaves no
  binding and no rows.
- **A map and its drops are one transaction (plan 3.4).** `write_map` takes the
  drops of the same step (`drops=`: the lines a tokenizer gave no tokens for, the
  skips a recovery read from a sanitizer's log) and writes them with the map, so
  a process killed at any point leaves the map with its drops or neither.
  `write_maps` does the same for every map one derivation run gave for one shard
  (a position map and a token-matching map): all of them and the drops, or
  nothing. A map or a drop the store refuses refuses the whole call.
- **Roles.** A file a step produced is an "intermediate" or a "release". A file
  with no producing step is registered under a member name and carries the role
  it has: "source" for an ingest, "import" for a text-only import, and
  "intermediate" or "release" for a pipeline file found afterwards. The store has
  no role word of its own for "found afterwards": such a file is told from a
  captured one by having no producing step, and a listing names a found release
  among the releases it covers.
- **Labels.** A label stronger than `vocabulary.labels_allowed(derivation,
  verification)` is refused with `LabelStrongerThanDerivation`. The rule holds for
  an edge, a tokenization map and its rows, and a source link.
- **Identity.** `register_jsonl` computes the edge digest from the edges whose
  derivation is "stated", then the instance id, then keys the occurrences.
- **A second registration (DECISIONS.md D-14).** Registering a history whose
  instance id the store already holds returns the stored instance when everything
  agrees, and writes nothing but a new name. When a field the id does not cover
  differs, the registration is refused with `InvalidRecord` and nothing changes:
  the role, a line's fingerprint, an edge's records in scope only, its label,
  derivation, verification or transform, the name of its job, its harness log (the
  job name, the log's digest, the model entry, the hand-off, the reads), or a drop
  the stored instance does not hold. The path is not compared.
- **Parents (DECISIONS.md D-15).** A parent of a stated edge lies in an artifact
  the step declared in `inputs`, or in the artifact itself (instance id `""`);
  otherwise `InvalidRecord`. An edge added with `add_recovered_edges` is exempt
  and may name any registered record.
- **Generation edges.** A parent or an in-scope record of a generation edge must
  lie inside the hand-off of its harness log; otherwise `InvalidRecord`. When the
  log's hand-off is exactly one record, the edge must name it as a parent
  (DECISIONS.md D-21); an edge that marks it in scope only, or leaves it out, is
  `InvalidRecord`. An edge that leaves out any record of the hand-off, a parent or
  in scope only, is `InvalidRecord` too, and a stored edge that does so gives no
  bound to a negative.
- **Handed texts.** `EdgeRecord.handed_texts` holds the texts a job was handed, each
  with its reference; `capture.generation_edge` copies them from the job's log, and
  the log's own `handed_texts` are compared too. The
  store compares each with the record its reference names (the fingerprint of the
  text after the store's normaliser against the stored fingerprint) before it writes
  anything, in `register_jsonl` and when a registered history is handed again (and in
  `add_recovered_edges`, for an edge built by hand that carries texts), and refuses
  the registration with `InvalidRecord`,
  naming the job and the reference, when one is not that record. The texts are not
  stored and are no part of an edge's equality. An edge and a log with none (the log
  read from a file, fetched from the store or built by hand, or no log) are not
  compared, and the log is stored as not compared (`hand_off_checked` 0: it establishes
  no bound until `Store.verify_hand_off` is given the texts, docs/store.md); a reference to a
  record that holds no text takes only a text that is empty after the normaliser;
  docs/capture.md, section 3, lists the routes.
- **An edge keeps the name of its job (DECISIONS.md D-24).** `EdgeRecord.job_name`
  is the name of the generation job that made the child. Every generation edge
  (`is_generation` true) carries one, and no other edge does; an edge with a
  harness log carries the job name of that log. Anything else is `InvalidRecord`.
  The store keeps the name on the edge itself, so a stored edge reads back with
  its `job_name` when its harness log is gone, and an answer can name the job
  (plan Check 5). `capture.generation_edge` sets it from the log. A pipeline whose
  job wrote its own parent list (a generation edge with no harness log) passes
  the job's name itself.
- **A harness log keeps the harness that wrote it.** `HarnessLog.harness` is
  "in-process", "confined" or None (a log that does not say). The store keeps
  the word with the log and refuses any other. A second registration whose log
  names another harness is refused like any other difference in a log (D-14).
- **Settings.** `set_setting` takes any key and value, except under the prefix
  `extension.`: there the key names one of the six extensions and the value is
  "on" or "off", else `InvalidRecord`. A stored switch that holds anything else is
  not read as off: every request, proof, listing, report and `verify` is refused first, whether or not
  it reads that switch (`extensions.check_switches`). `show`, `check`, `migrate` and the
  `recover` and `register-run` commands are not answers and keep working.
- **`add_member`** is idempotent for an identical call. A member that exists with
  the same source, revision and part kind gains the parts it does not hold yet;
  another source, revision or part kind under a taken member name is refused.
- **`add_drops`** records drops of the step that wrote a registered artifact when
  they are known only after registration (the lines a tokenizer gave no tokens
  for, the skips a recovery reads from a sanitizer's log). A drop is one drop per
  dropped record, step name and output artifact: the same step name can drop one
  record once for each artifact it writes. The same drop again changes nothing; the
  same record, step name and output with another reason or group key is refused.
- **`add_source_link`** stores the link's derivation and verification. A
  "file-level" link has the derivation "file". A "mapped" link needs the
  derivation "pointer" with verification "passed". A pointer that is not verified,
  and the derivation "none", can only be "unknown". No other derivation is
  accepted: a source link has no parent record behind it.
- **Maps (DECISIONS.md D-20, amended by D-24).** A namespace and derivation of a
  shard hold at most one map whose verification passed, so a position-mode and a
  token-matching recovery map stand side by side. A map that is held is never
  overwritten and never deleted, and a verified map is final for its namespace
  and derivation: another map there is refused. A map that did not verify
  ("failed" or "not run") does not take its namespace and derivation for good: a
  later map of the same namespace and derivation is written beside it, so a
  corrected run counts. A reader gets the first map in this order: a verified
  map before one that is not, across both namespaces; among verified maps the
  capture map, then the position-mode recovery map, then the token-matching
  recovery map. Only a shard with no verified map gives a reader a map that is
  not verified. Such a map is still the shard's own record, and its links can
  only be "unknown". Every map, read or not, is bound to its shard by count and
  digest and is held to the loud checks (plan 2.2(d)); condition 2.2(f) is read
  on the map a reader gets.
- **Loud checks (DECISIONS.md D-18).** `check_integrity` raises the first fault it
  finds in the given artifacts' ancestor closure (all artifacts when None), in
  this order:
  1. `MapCountMismatch`: a binding's declared count against the rows the store
     holds under it;
  2. `MapDigestMismatch`: a binding's declared digest against the digest its
     artifact is registered with;
  3. `EdgeParentMissing`: an edge input, a map row or a candidate entry that names
     an occurrence row which is gone;
  4. `TraceSampleCountMismatch`: a present trace's consumed-sample count against
     its step's batch size and against its own sample rows;
  5. `ArtifactDigestMismatch`: an artifact's registered digest against its file on
     disk; a shard with one of its two files gone; a mixture snapshot entry's
     digest against the digest its shard is registered with. Then, only for a
     file that is present and has its registered digest, `MapCountMismatch`
     again: a binding's declared count against the number of records the file
     holds, counted in the read that takes the digest.

  Items 1 to 4 compare the store's own records and never open a file. So a
  changed file, with the store untouched, raises "artifact digest mismatch" and
  no map fault, and rows removed from the store together with a lowered count
  raise "map count mismatch" while the file is present. Every query calls
  `check_integrity` before it prints anything. A missing or partly covered trace
  is not a fault; it is an untraced step (plan 2.2(e)). An artifact whose file is
  gone from its recorded place is not a fault either: the records still answer,
  and its declared count is compared with the store's rows only.
- **Paths.** A path stored in the store is relative to the folder that holds the
  store file when the artifact lies under that folder, and absolute otherwise.

## 5. Capture and the harness (`waterwhite.capture`, `waterwhite.harness`)

The thin wrappers a team's pipeline calls at its capture points (plan 3.3, 5.4).
They compute fingerprints, digests and ids and call the store. `docs/capture.md`
describes them in full.

```python
# waterwhite.capture
def ingest_jsonl(store, path, *, source: str, member: str, revision: str,
                 name: str | None = None, text_field: str = "text") -> str: ...
def record_jsonl(store, path, *, name: str, role: str, step: StepRef,
                 inputs: Sequence[str], edges: Sequence[EdgeRecord],
                 drops: Sequence[DropRecord] = (), text_field: str = "text") -> str: ...
def import_jsonl(store, path, *, name: str, member_name: str,
                 file_family: tuple[str, str] | None = None,
                 text_field: str = "text") -> str: ...   # (member, evidence): file-level link

# builders of what a step states
def step_ref(name: str, version: str, params: object = None) -> StepRef: ...
def occurrence(instance_id: str, line: int) -> OccurrenceRef: ...
def copy_edge(child_line: int, parent: OccurrenceRef, step: StepRef) -> EdgeRecord: ...
def selection_edge(child_line: int, group: Sequence[OccurrenceRef], kept: OccurrenceRef,
                   step: StepRef, *, group_key: str | None = None
                   ) -> tuple[EdgeRecord, tuple[DropRecord, ...]]: ...
def drop(input: OccurrenceRef, reason: str, step: StepRef) -> DropRecord: ...
def generation_edge(child_line: int, output: JobOutput, log: HarnessLog,
                    step: StepRef) -> EdgeRecord: ...
def generation_edges(result: JobResult, child_lines: Sequence[int],
                     step: StepRef) -> tuple[EdgeRecord, ...]: ...

def line_text(raw: bytes, text_field: str = "text") -> str | None: ...
                                             # the text a line holds, or None; the one
                                             # definition in the tool

# waterwhite.harness (plan 3.3 "Generation", F10)
@dataclass(frozen=True)
class ModelReply:
    text: str
    model: str                           # the model that answered, as the client reports it

ModelClient = Callable[[str, str], str | ModelReply]   # (model, prompt) -> text

@dataclass(frozen=True)
class Handed:
    ref: OccurrenceRef
    text: str

@dataclass(frozen=True)
class JobOutput:
    text: str
    used: tuple[OccurrenceRef, ...]      # the handed records the output states it used

@dataclass(frozen=True)
class JobResult:
    outputs: tuple[JobOutput, ...]       # in the job's order
    log: HarnessLog                      # read back from the file the harness wrote, plus
                                         # handed_texts (the file holds none)

class JobContext:
    def handed(self) -> tuple[OccurrenceRef, ...]: ...
    def read(self, ref: OccurrenceRef) -> str: ...   # logs the read; a ref outside the
                                                     # hand-off is refused (HarnessRefused)
                                                     # and the attempt is logged
    def call_model(self, model: str, prompt: str) -> str: ...   # logs the model id, then
                                                     # calls the client run_job was given

def run_job(job, *, job_name: str, handed: Sequence[Handed], log_path,
            model_client: ModelClient | None = None) -> JobResult: ...
def read_log(log_path) -> HarnessLog: ...
```

**What a pipeline imports.** `waterwhite.capture`, `waterwhite.harness`,
`waterwhite.normalise` and `waterwhite.store.Store`, and nothing else of the tool.
`capture` names, beside its own functions: the record types `StepRef`,
`OccurrenceRef`, `OccurrenceRecord`, `EdgeRecord`, `EdgeInput`, `DropRecord`,
`HarnessLog`, `MapRecord`, `MapRow`, `SnapshotShard` and `OWN_ARTIFACT`; the id
functions `line_locator`, `ordinal_locator`, `param_digest`, `file_digest`,
`shard_digest`; and the words a pipeline writes: the roles, the parent modes, the
drop reasons, the labels, the derivations and verification results of a map, the
namespaces and the part kinds (`capture.__all__` lists them). They are the same
objects as in `records`, `ids` and `vocabulary`.

**Parents of a generated output** (plan Decision 25, DECISIONS.md D-21). The
wrapper writes each output's edge from the hand-off and `JobOutput.used`. Every
handed record is an input of the edge. A job handed exactly one record: that
record is a parent of each output, even when the output states nothing. A job
handed two or more: a record the output states it used is a parent, and a handed
record it does not state is in scope only. `generation_edge` applies this, and
the store refuses an edge that contradicts it (section 4). The edge it returns
carries the harness log, in `job_name` the log's job name (DECISIONS.md D-24), and
in `handed_texts` the texts the log carries.

**Before an ingest or an import writes.** `ingest_jsonl` and `import_jsonl` make
more than one write. Each reads `store.member(name)` first and refuses with
`InvalidRecord`, before the first write, what the store would refuse at a later
one: a member name taken by another source, revision or part kind, and a file
family whose member is not registered.

**`run_job`.** `job` is called once with a `JobContext` and returns a `JobOutput`
or a sequence of them. `model_client` is the function behind
`JobContext.call_model`; a job that calls no model needs none, and a model call
with no client is logged and then fails with "no model client". A log file that
already exists at `log_path` is refused: a log is evidence and is not overwritten.
When the job raises, the log is closed with status "failed" and the error is
raised again.

**The log file.** JSON lines, one entry per line, in canonical JSON (sorted keys,
no spaces, ASCII). It holds references and identifiers: no record text, no prompt
and no clock time, so the same job with the same hand-off and the same reads gives
the same bytes. Every entry has `seq` (0, 1, 2 and so on) and `kind`:

| `kind` | When | Other fields |
|---|---|---|
| `job` | first | `format` ("waterwhite.harness-log/1"), `job_name`, `handed` (a list of `{"instance_id", "locator"}` in hand-off order), `harness` ("in-process" or "confined": which harness wrote the log) |
| `read` | each read, before the text is returned | `refused`, and `instance_id` with `locator` for a handed record, or `outside_ref` (text) for anything else |
| `model_call` | each call, before the client is called | `model`: the identifier the job asked for |
| `model_reply` | after a call whose client returned a `ModelReply` | `call` (the `seq` of the call's entry), `model`: the identifier the client reports answered |
| `output` | per output, after the job returned | `index`, `used` (as stated) |
| `model` | once, at the end | `model`, `models` |
| `end` | last | `status` ("completed" or "failed"), and `outputs` or `error` |

`read_log` reads a log file into a `records.HarnessLog`: `job_name`, `log_digest`
(the SHA-256 of the file), `handed`, `reads`, `model` and `harness`. `run_job`
returns that reading of the file it wrote, with `handed_texts` set: the handed
references with their texts, in hand-off order. The texts are in no file, take no
part in the log's equality, representation or digest, and are never stored; a log
from `read_log` holds none. The store compares them where the output is registered
(section 4, "Handed texts"). The model field is `"none"` when the
model entry says the job called no model, an identifier when it called one model,
a JSON list of identifiers when it called several, and `None` when the file has
no model entry, which is read as "not established" and never as "no model".

**Which model a log names.** The identifier the job asked for is written before
the client is called, so the log names every model a job asked for, whether or
not the call returned. A client that returns a `ModelReply` reports which model
answered; that identifier is written after the call, and the model entry names it
in place of the one the job asked for. The job chooses the text of its request;
the client, which runs on the caller's side, knows what it called.

With the generating-model switch on, the ancestor closure of a query also holds the ancestry of the generating models (docs/coverage.md, section 3).

**Which harness a log names.** `harness` is "in-process" for a job `run_job` ran
as a function of the caller's process, and "confined" for a job
`waterwhite.confine` ran in its sandbox. `run_job` takes no argument for the word
and always writes "in-process"; only `waterwhite.confine` writes "confined", by
way of a function of the harness that is not part of its surface (Decision 13).
A log file written by hand outside the tool is outside what the store can check
(docs/confine.md section 5). A log that does not say reads as `None`.
Condition 2.2(g) reads the word: only a log the confined harness wrote
establishes a bound (Decision 13: a bound is established only by a closed-input
harness whose access log lists every record the job read). A log the in-process
harness wrote, or one that names no harness, establishes none, and past its edge
the answer is "may contain" naming the job (plan F10's fallback). An answer names
the harness beside every log its negative rests on (section 6).

**The bound.** `HarnessLog.reads_within_hand_off()` is true when the log lists no
read outside the hand-off that the harness let through; a refused attempt is
logged and does not count (DECISIONS.md D-07). `HarnessLog.bound_stated()` is
true when, besides, the confined harness wrote the log (Decision 13; DECISIONS.md
D-27); the edge's own bound field is written from it. `HarnessLog.bound_established()` is
`bound_stated()` and the store's mark `hand_off_checked`, which is set when the texts handed
to the job were compared with the records their references name, at registration or by
`Store.verify_hand_off`; a stored log without the mark establishes no bound, and the answer
past its edge is "may contain" naming "upper bound not established".

`run_job` is built for an in-process job. A job that runs out of process is
confined by `waterwhite.confine` (section 5.3; plan 5.8, B-1), which runs it in a
sandbox and serves it its handed records through this same harness.
`docs/capture.md` says what the in-process harness does not establish and
`docs/confine.md` what the confined one does.

**The normaliser.** A normaliser is named `"<name>/<version>"` and registered in
`waterwhite.normalise`; the store records the one its fingerprints were taken
after and gives its name as `store.normaliser`. The first build ships
`control-chars/1`: it removes the characters of the Unicode category Cc except the
tab and the line feed, keeps the category Cf, and strips a fixed list of
white-space code points from both ends. It is defined once, in
`waterwhite.normalise`, and a pipeline whose sanitizer is this function imports it
from there in every mode (DECISIONS.md D-17), so a sanitized record and its
fingerprint agree (plan 3.1, F8).

### 5.1 Tokenization maps (`waterwhite.tokenmap`)

Which input line each document of a shard came from is derived, never observed
(plan 3.3 "Tokenization"). `docs/tokenmap.md` describes the package in full.

```python
# capture: the wrapper around the team's tokenizer run
def tokenize_shard(store, *, input: str, prefix, name: str, step: StepRef,
                   run_tokenizer: Callable[[Path, Path], object],
                   settings: TokenizerSettings, tokenize=None, f1_holds: bool = True,
                   text_field: str = "text") -> ShardMaps: ...
def map_tokenized_shard(store, *, input: str, prefix, name: str, step: StepRef,
                        settings: TokenizerSettings, tokenize=None, f1_holds: bool = True,
                        text_field: str = "text") -> ShardMaps: ...
# recovery: the map of a registered shard whose tokenizer ran without the wrapper
def recover_map(store, *, shard: str, input: str, settings: TokenizerSettings,
                tokenize=None, sanitizer: SanitizerRun | None = None,
                f1_holds: bool = True, also_token_matching: bool = False,
                text_field: str = "text", write: bool = True) -> ShardMaps: ...

def tokenize_step(version: str, params: object = None) -> StepRef: ...   # the step "tokenize"
def read_sanitizer_log(path, *, shard=None, input=None) -> tuple[SanitizerSkip, ...]: ...
TokenizerSettings(append_eod: bool, eod_in_span: bool, eod_token_id: int | None = None)
SanitizerRun(skips=(), normaliser: str | None = None, step_name: str = "sanitize")

# the team's tokenizer, without the tool importing a tokenizer library
def command_tokenizer(argv, *, timeout_seconds: float = 120.0) -> CommandTokenizer: ...
def entry_point_tokenizer(spec: str) -> Callable[[str], Sequence[int]]: ...   # "module:function"
```

The package also exports the result types `ShardMaps` and `MapOutcome` (each with
`to_json()`), `Verification`, `Assignment`, `SanitizerSkip`, the two derivation
functions `derive_position` and `derive_token_matching` with `DocumentClasses`,
`Retokenizer`, `check_spans` and `Tokenize`, the reader `ShardTokens` (the store's,
`artifact_files.ShardTokens`), `CommandTokenizer`, and the constants
`STEP_TOKENIZE`, `STEP_SANITIZE`, `SANITIZER_REASONS`, `TOKENIZER_PROTOCOL` and
`DEFAULT_TOKENIZER_TIMEOUT`.

- `tokenize(text)` is the team's tokenizer as a function from a text to its token
  ids, without the end-of-document token. The tool ships none and imports none.
  `command_tokenizer` asks an external command, one text per line (the protocol
  `waterwhite.tokenizer/1`, `docs/cli.md`); `entry_point_tokenizer` imports the
  module the team names. Without a function nothing is compared and the map is
  "not run" and "unknown".
- A capture map goes to the namespace "capture", a recovery map to "recovery".
  Position mode runs first; token matching runs when position mode's verification
  fails, when `f1_holds` is false, and when `also_token_matching` asks for both.
- Everything one call derived is written in one `store.write_maps` call, one
  transaction: the maps and, when one of them verified, the drops (lines with no
  tokens; the skips of a sanitizer's log). With `write=False` nothing is written.

### 5.2 Recovery (`waterwhite.recovery`)

Links for steps nobody observed (plan 3.3 "Copy steps" and "Deterministic
rewrite", 5.4). `docs/recovery.md` describes the package in full.

```python
def register_found_file(store, path, *, name: str, role: str = "intermediate",
                        text_field: str = "text") -> str: ...
def recover_copy_step(store, later: str, earlier: Sequence[str], *, step: StepRef,
                      text_field: str = "text", write: bool = True) -> RecoveryResult: ...
def recover_sanitizer(store, later: str, earlier: str, *, step: StepRef,
                      log_path=None, log_input: str | None = None,
                      text_field: str = "text", write: bool = True) -> RecoveryResult: ...
def sanitize_step(version: str, params: object = None) -> StepRef: ...   # the step "sanitize"
def read_sanitizer_log(path, *, input: str | None = None) -> tuple[LoggedSkip, ...]: ...
```

It also exports `RecoveryResult` (with `to_json()`), `LoggedSkip`, the kinds
`KIND_COPY_STEP` and `KIND_SANITIZER`, the outcomes `OUTCOME_RECORD_LEVEL`,
`OUTCOME_FILE_LEVEL`, `OUTCOME_UNKNOWN` (`OUTCOMES`), and `SANITIZER_REASONS`.

- A found file is registered with the role it has in the pipeline
  ("intermediate", "release", or "import") and no producing step (section 4,
  "Roles").
- One candidate parent gives a "recovered" edge, two or more a "candidates" edge,
  none no edge. A text the later file holds more often than the named inputs do
  is surplus: no copy of it gets an edge, each keeps no recorded origin ("may
  contain", "missing origin"), and the result counts the copies in `surplus`.
- When the store's normaliser is not available, `recover_sanitizer` matches no
  record. Its outcome is "file-level" only where a file family of the later file,
  or the step's own log together with the earlier file's membership, supports
  the link; on the caller's statement alone it is "unknown".
- With `write=False` nothing is written and `RecoveryResult.dry_run` is true.

### 5.3 The confined harness (`waterwhite.confine`)

A generation job run out of process, in a bubblewrap sandbox with no network and
no file of the machine's data, served its handed records over a pipe by the
harness of this section (plan 3.3 "Generation", F10, 5.8 B-1). `docs/confine.md`
describes the sandbox, the protocol and what this establishes.

```python
def run_confined_job(program: Program, *, job_name: str, handed: Sequence[Handed],
                     log_path, timeout_seconds: float,
                     model_client: ModelClient | None = None,
                     scratch_bytes: int = DEFAULT_SCRATCH_BYTES,
                     max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES) -> JobResult: ...
def python_program(script, args=(), code=()) -> Program: ...
def find_sandbox() -> Sandbox: ...
```

It also exports `Program`, `CodeMount`, `Sandbox`, the three error classes (they
are classes of `waterwhite.errors`), and the constants `PROTOCOL`,
`SANDBOX_COMMAND`, `JOB_FOLDER`, `CLIENT_FOLDER`, `SCRATCH_FOLDER`,
`DEFAULT_SCRATCH_BYTES`, `DEFAULT_MAX_REQUEST_BYTES` and `MAX_LOGGED_TEXT`.

`run_confined_job` returns the `JobResult` of `run_job`, and the log is the same
format, written by the same code, with `harness` "confined" in its job entry.
`capture.generation_edges` and the store take a confined result as any other.
There is no fallback: where the sandbox cannot be built the call ends with
"sandbox unavailable" and no log is written.

### 5.4 What a pipeline calls, and when

| Step | With capture, as the step runs | Without capture, afterwards |
|---|---|---|
| ingest | `capture.ingest_jsonl` | the same call: a source file is ingested either way |
| copy steps (shuffle, chunk, merge) | `capture.record_jsonl` with `capture.copy_edge` per line | `recovery.register_found_file` for each file kept, then `recovery.recover_copy_step` per later file |
| sanitizer | `capture.record_jsonl` with a copy edge per surviving line and a `capture.drop` per skipped line | `recovery.recover_sanitizer`, with the sanitizer's log |
| generation | `harness.run_job` or `confine.run_confined_job`, then `capture.generation_edges` and `capture.record_jsonl` | not recovered: a parent of a generated record is never inferred (plan L5) |
| selection (dedup) | `capture.selection_edge`, then `capture.record_jsonl` | `recovery.recover_copy_step`: equal texts are "candidates" |
| tokenization | `tokenmap.tokenize_shard`, or `tokenmap.map_tokenized_shard` after the run | `store.register_shard`, then `tokenmap.recover_map` |
| mixture, run, checkpoint | `store.register_mixture`, then `waterwhite register-run` (one `register_run_segment` call with the step traces and the checkpoints) | the same calls, from what the run kept |

**A skipped line is recorded once.** A line the sanitizer skipped, or an empty
line, is recorded exactly once, as a drop whose output is the first registered
artifact that lacks the line (plan 3.1 "Drop", 3.3):

- With capture, the tokenizer's input is registered (section 9.2: the shard's name
  followed by `.input`), the sanitizer's drops are recorded with that file as
  their output, and the capture map is written over that file. The map then sees
  only lines that give no tokens as drops.
- Without capture, where the tokenizer's input was kept: it is registered as a
  found file, `recovery.recover_sanitizer` records the edges and the drops
  between the sanitizer's input and it, and `tokenmap.recover_map` runs over it
  with no sanitizer's log.
- Without capture, where only the sanitizer's input was kept: `recover_map` is
  given that file with a `SanitizerRun` built from the log, and records the drops
  with the shard as their output.

Never both for one line. The two packages hold each other to it and refuse with
"drop already recorded", before anything is written:

- `recover_map` with a sanitizer's log, when its input is itself the output of
  recorded sanitizer drops (it is the file the sanitizer wrote), or when a line
  the log names already has its drop under a registered JSONL file;
- `recover_sanitizer` with a log, when a line it names already has its drop under
  a shard. Run without the log, it records the edges alone;
- either, when the line already has a sanitizer's drop for the same output under
  another step name.

A sanitizer's drop is a drop with one of `vocabulary.SANITIZER_DROP_REASONS`.
Both packages read a log's reasons by that tuple, and both name the step
"sanitize" unless told otherwise (`recovery.sanitize_step`,
`tokenmap.SanitizerRun.step_name`). Two outputs of one kind are no conflict: one
sanitizer input sanitized into two files, or tokenized into two shards, has one
drop for each.

## 6. Answers

`coverage.prove(store, artifact, request)` returns a proof object or the first
failing link (plan 2.2). The answer type has no way to build "does not contain"
except from a proof (plan W6, Decision 1). A proof belongs to the artifact it was
made for and to the store as it was read: an answer shows a name of that artifact
(or its instance id), and no answer is built from a proof made before the store was
written (docs/coverage.md, "How does not contain is closed off").

The JSON form of one answer (the table's `expect` blocks use the same field names):

```json
{
  "artifact": "R1",
  "instance": "<instance id>",
  "kind": "jsonl",
  "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
  "answer": "contains",
  "reason": null,
  "names": null,
  "names_detail": null,
  "failing": null,
  "candidates": [],
  "chain": [],
  "basis": null,
  "notes": [],
  "untraced_steps": [],
  "first_exposure": null,
  "level": null,
  "scope": null
}
```

Every answer has every one of these keys, in this order.

- `artifact`: the name the query used; `instance` and `kind` are null for an
  artifact the store has no row for.
- `request`: the request as it ran: its kind, the source, the members it expanded
  to, and the records it was narrowed to, each as `name:locator`.
- `answer`: one of the four answers.
- `reason`: for "not covered" only: "record missing" or "not registered" (a published
  model stored while its extension is off is "not registered", with the note "the
  published-models extension is off" in `notes`).
- `names`: for "may contain": the named cause, a `NAMES_*` constant.
- `names_detail`: the reason beside a named cause, when the cause has one. Beside
  "data-order trace missing" it is one of the six words of
  `TRACE_MISSING_STORED_REASONS`, or null where the segment was registered by a
  caller that gave none. It is the reason stored at registration, never derived when
  the answer is made.
- `failing`: for "may contain": the first failing link, an object with the keys
  below and no others. A key that does not apply to the link is left out, and a
  reader treats a key that is left out and a key that is null alike.

  | Key | Value |
  |---|---|
  | `condition` | the letter of the proof condition of plan 2.2 that failed, "a" to "h"; left out where the link is a missing record upstream |
  | `artifact` | the artifact the link belongs to, by a name |
  | `instance` | that artifact's instance id |
  | `record` | the record the link sits on |
  | `segment` | the run segment, for a link of a checkpoint's history |
  | `step` | the step inside that segment, a number |
  | `job` | the name of the generation job, for a link of a generation edge. The edge keeps the name itself, so it is given whether the job's harness log is stored or gone (DECISIONS.md D-24). For an edge that is not a generation edge and fails (g), a publisher's pointer, it is the name of the edge's step |
  | `reason` | for a missing record upstream: "record missing" or "not registered" |
  | `detail` | one sentence for a person; its wording is not fixed |

- `candidates`: for "may contain": the candidate sources. Through a candidate set,
  the sources of the candidates; otherwise the sources among the inputs of the
  step that made the artifact, followed back through what those were made from.
- `chain`: for "contains": the evidence chain as an ordered list of hops, from the
  artifact asked about to the source record. A hop is an object. Its key `hop` is
  the hop's short form: `artifact:locator` for a record (`L<n>` a line, `D<n>` a
  document of a shard), `segment:step n` for a training step. Its other keys,
  each present where it applies: `artifact`, `instance` and `record`, or `segment`
  and `step`; `link`, how the hop is tied to the next one (the evidence label of
  its edge or map row, "member" on a record of a requested member, "file-level
  link" or "mapped link" where a source link ends the chain, "consumed" on a
  step); `member`; `job`, with `generation: true`, on a hop whose edge is a
  generation edge.
- `basis`: for a checkpoint's "contains": the basis it states (plan 2.4).
- `notes`: the full sentences of `vocabulary` (`NOTE_*`), word for word, not
  short keys.
- `untraced_steps`: for a checkpoint: `[segment, step]` pairs.
- `first_exposure`: for a checkpoint that "contains": `{"segment": ..., "step": ...}`.
- `level`: for a checkpoint that "contains": "consumed" when a step of its own
  segment read the record, "inherited" when a step of an ancestor's did.
- `scope`: for "does not contain": what the proof covers (plan 2.4, L8), an object
  with these keys:

  | Key | Value |
  |---|---|
  | `source` | the source of the request |
  | `members` | the members the request expanded to |
  | `records` | the records the request was narrowed to; empty when it was not |
  | `runs` | for a checkpoint: the runs of its history |
  | `steps` | for a checkpoint: a list of `[segment, first step, last step]` |
  | `generation_edges` | the names of the jobs whose established bound the proof rests on |
  | `harness_logs` | one object per such job: `job`, `log_digest`, `handed` and `reads` (two counts), `refused_attempts`, `model`, and `harness` ("in-process", "confined", or null when the log does not say) |
  | `walked` | how many records the proof visited |
  | `files_not_checked` | the artifacts of the ancestor closure whose file was not at its recorded place |

A listing (plan 2.1, Decision 26; DECISIONS.md D-24) is
`{"request": ..., "covers": {"runs": [], "mixtures": [], "shards": [], "checkpoints": [], "releases": []}, "answers": [...], "not_covered": [...]}`.

- `covers` names what has a record of its own: the runs, the mixtures, the shards
  with a map, the checkpoints whose run segment is registered, the JSONL files
  with the role "release".
- `answers` holds one answer, in the form above, for every registered artifact,
  whatever its kind and role: source files and the files between source and
  release among them, and an artifact with no record of its own, which answers
  "not covered". The order is by kind (JSONL files, shards, mixtures,
  checkpoints) and inside a kind the order of registration.
- `not_covered` names every artifact that has no record of its own, registered or
  only named by another record. An entry is
  `{"artifact", "instance", "kind", "reason", "named_by", "detail"}`: `artifact`
  is a name, else the path a snapshot gives, else the instance id, and null for
  the mixture of a segment registered with no snapshot; `kind` is "jsonl",
  "shard", "mixture", "checkpoint", or "input" for a step's input the store has
  no row for; `named_by` is a list of `{"relation", "referrer", "position"}`
  (`position` only where the relation has one). An artifact that is only named
  has no row and so no answer.

## 7. Command line

`waterwhite <command> --store <file> ...`, JSON on standard output by default
(`--format text` for a person). An error prints
`{"error": name, "detail": ..., "context": {...}}` on standard error and exits
with the family's code; nothing is printed on standard output then. `docs/cli.md`
describes every command, flag and output in full.

| Command | What it does | Exit codes |
|---|---|---|
| `init --normaliser NAME/VERSION [--verified-join TAG,TYPE,CLASS ...]` | create a store that lists the joins the tool ships as verified, or, with the flag, only the joins of that list that it names; a join outside the list is refused | 0; 1; 4 |
| `query withdraw <source> [--member M ...] [--record ARTIFACT:LOCATOR ...] [--selector KEY=VALUE ...] [--artifact NAME]` | one answer, or with no `--artifact` the listing | 0; 1; 2; 3 |
| `query ban <text> [--artifact NAME]`, `query flagged <file> [--artifact NAME]` | requests of two extensions that are off: refused with "extension off" | 1; 2 |
| `show <artifact> [--record LOCATOR]` | what the store holds about an artifact or one record of it; runs no loud check | 0; 1; 2; 3 |
| `show <shard> --map NAMESPACE/DERIVATION [--verification RESULT]` | the rows of one named map of a shard; runs no loud check | 0; 1; 2; 3 |
| `show <segment> --trace [--step N]` | the rows the join wrote for a registered run segment, in reading order, with its route, cause, the run record's launches (`launches`, as registered) and untraced steps; for a cyclic segment each step also names the epochs it reads in (`epochs`); `--step` keeps one step; runs no loud check | 0; 1; 2; 3 |
| `check` | run the loud checks over the whole store, after reading the two files of every shard at its place and recording what the read found (it opens the store for writing, and refuses a store that needs a migration step) | 0; 1; 2; 3; 4 |
| `report withdraw <source> [--member M ...] [--record ARTIFACT:LOCATOR ...] [--selector KEY=VALUE ...] [--layout NAME=PREFIX ...]` | the rebuild report of a withdrawal (plan 4.1, 4.3): per affected artifact and per record its action and its reason, the counts, the restart points, the checkpoints that carry exposure and the checkpoints that answer "not covered"; computed from the store as it is, writes nothing (section 7.1) | 0; 1; 2; 3 |
| `report ban <text>`, `report flagged <file>` | requests of two extensions that are off: refused with "extension off" | 1; 2 |
| `verify --report FILE [--sides FILE] [--mode seeded\|unseeded] [--text-field FIELD]` | the verification of a repair (plan 4.2, 4.4): the repaired side against the reference side, the artifacts the report keeps, the answers for the successors and for the history, the stale caches and the exact-rerun restart; opens the store read-only (sections 7.2 and 7.3) | 0; 1; 2; 3 |
| `register-run --run-record PATH [--steps PATH] [--cache DIR] [--mixture NAME] [--parent-checkpoint NAME]` | register one run segment with its steps, checkpoints and step traces, in one transaction; it prints the route, the cause, the untraced steps, what became of the replay (`replay`: `attempted`, `accepted`, `reason`) and, for a cyclic segment, the shuffle the join computed beside the recorded one (`shuffle`, one entry per launch, epoch and rank; `[]` otherwise) | 0; 1; 4 |
| `replay --run-record PATH --out DIR` | derive the segment's index arrays again from its run record and its shards, write each to `DIR` as a `.npy` file and print its digest beside the recorded one; opens no store | 0; 1; 4 |
| `recover file <path> --name NAME [--role intermediate\|release\|import]` | register a JSONL file found afterwards | 0; 1; 2; 3; 4 |
| `recover shard <prefix> --name NAME [--input ARTIFACT ...] --tokenizer-version V` | register a tokenized shard found afterwards (the artifact row; its map is `recover map`) | 0; 1; 2; 4 |
| `recover copy-step <later> --input ARTIFACT ... --step-name NAME` | the links of a copy step for one file it wrote | 0; 1; 2; 3; 4 |
| `recover sanitizer <later> --input ARTIFACT [--log FILE [--log-input VALUE]]` | the links and the drops of the sanitizer for one file it wrote | 0; 1; 2; 3; 4 |
| `recover map <shard> --input ARTIFACT --mode position\|token-matching\|both --append-eod yes\|no --eod-in-span yes\|no [--tokenizer-command CMD \| --tokenizer-entry MODULE:FUNCTION] [--sanitizer-log FILE]` | the tokenization map of a registered shard | 0; 1; 2; 3; 4 |

Every command but `replay` takes `--store FILE`, and every command takes `--format json|text`. A store path that
holds no store is "not a store" (exit 2) for `query`, `show`, `report` and `verify`, which
open the store read-only, and for `check` and `recover`, which open it for writing, or
read-only with `--dry-run` (every `recover` command takes it, prints its result
and writes nothing). A recovery whose links did not verify still exits with 0:
the result says what was found. The tokenizer a map recovery needs is the
team's own, named by one of two flags, so the tool imports no tokenizer
library; `docs/cli.md` gives the protocol of a tokenizer command. `init` on a path that holds a file is "a file already
exists at the store path" (exit 4), and with an unknown normaliser "unknown
normaliser" (exit 4); it makes no file then.

`--record ARTIFACT:LOCATOR` narrows the withdrawal to single records of the source
(plan 2.3: a list of records). `ARTIFACT` is a name or an instance id of a
registered file of the source and `LOCATOR` a line locator, for example
`src/B:L1`. The option may be given several times. The request's `records` field
then holds the records named. `--selector` stands for a selector on metadata; none
is supported, and one is enough to refuse the request with "unsupported selector".

A later milestone adds `extension`.

### 7.1 The rebuild report (`report`)

`waterwhite report withdraw <source> --store FILE` puts the same request as
`query withdraw` (same `--member`, `--record` and `--selector`, same refusals) and
prints the rebuild report of plan 4.1 and 4.3: what the team's own builder must do
to make the artifacts downstream of the source clean again. The report is computed
on demand, from the store as it is, and nothing is written: the document the
command prints is the report, and a team that wants to keep one keeps the file. The
store holds no report, because a stored report would outlive the state of the store
it was made for and every write would have to invalidate it; `verify` is handed the
file instead (section 7.3). `--layout NAME=PREFIX`, which may be given several
times, names a layout: the shards whose store name begins with `PREFIX` form the
layout `NAME`. The store holds runs and mixtures and no layout, so a layout is the
team's to name; a shard that two prefixes match is a usage error. `--placement
NAME=FILE`, one per layout, states where that layout puts the records that return to a
release (below, "Where a returning record lands"); a placement for a layout that `--layout`
does not name, a file that is not a placement, and a file that names a shard whose store
name does not begin with the layout's prefix, are usage errors. `report ban` and
`report flagged` are refused with "extension off". Like a listing, the report runs
the loud checks over the whole store first; one fault ends it with exit 3.

The JSON form. Every key is present, in this order; a key that does not apply to an
entry is null, and a reader treats a missing key and a null alike.

```json
{
  "form": "waterwhite.report/1",
  "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
  "records": [
    {"artifact": "R1", "record": "R1:L1", "action": "re-source", "chain": [],
     "new_parent": "src/B:L1", "withdrawn_parent": "src/A:L1",
     "regenerate_with": null, "step": null,
     "missing_link": null, "failing": null, "names": null, "review_scope": null},
    {"artifact": "R1", "record": "src/B:L1", "action": "promote", "chain": [], "...": "..."},
    {"artifact": "R1", "record": "R1:L2", "action": "retire", "chain": [],
     "regenerate_with": "paraphrase",
     "step": {"name": "replica.generation", "version": "1", "parameter_digest": "..."},
     "...": "..."},
    {"artifact": "R1", "record": "R1:L5", "action": "keep", "chain": [], "...": "..."}
  ],
  "artifacts": [
    {"artifact": "R1", "instance": "<instance id>", "kind": "jsonl", "role": "release",
     "action": "rewrite", "answer": "contains", "chain": [],
     "successor": "successor of R1",
     "records": 24,
     "counts": {"keep": 20, "re-source": 1, "promote": 1, "retire": 3, "unresolved": 0},
     "records_removed": 3, "removed": ["R1:L2", "R1:L3", "R1:L4"], "rows_in_successor": 21,
     "documents": null, "documents_in_successor": null, "receives": null,
     "rewritten_shards": null, "needs_successor": null, "bytes_to_rewrite": 1234}
  ],
  "upstream_files": [
    {"artifact": "src/A", "instance": "<instance id>", "kind": "jsonl", "role": "source",
     "answer": "contains", "action": null, "reason": "the revision the request withdraws"},
    {"artifact": "gen/P", "instance": "<instance id>", "kind": "jsonl", "role": "intermediate",
     "answer": "contains", "action": null,
     "reason": "the repaired build reuses this file as the same instance; the records of it that the request reaches leave the release that copies them (retired there), and the file itself is not rebuilt"},
    {"artifact": "E/shard-3.input", "instance": "<instance id>", "kind": "jsonl",
     "role": "intermediate", "answer": "contains", "action": "rewrite",
     "reason": "an intermediate file the repair rebuilds: the records that reach the withdrawn content leave it"},
    {"artifact": "E/shard-2.sanitizer-input", "instance": "<instance id>", "kind": "jsonl",
     "role": "intermediate", "answer": "may contain", "action": "unresolved",
     "reason": "nothing shows whether the request reaches this file: the proof stops at missing origin",
     "names": "missing origin", "failing": {"condition": "a", "...": "..."}}
  ],
  "counts": {
    "records": {"keep": 20, "re-source": 1, "promote": 1, "retire": 3, "unresolved": 0},
    "records_removed": 3,
    "shards": {"keep": 2, "rewrite": 1},
    "mixtures": {"keep": 1, "rewrite": 1},
    "bytes_to_rewrite": 5678,
    "bytes_unknown": [],
    "unresolved_artifacts": [],
    "place_not_stated": {"shards": [], "mixtures": []}
  },
  "mixtures": {"needs_successor": ["E/mix-2"], "kept": ["E/mix-1"], "unresolved": [],
               "place_not_stated": []},
  "groups": {
    "by_run": {"E": {"counts": {"keep": 2, "rewrite": 1}, "keep": ["E/shard-1", "E/shard-2"],
                     "rewrite": ["E/shard-3"], "unresolved": []}},
    "by_layout": {},
    "layouts": {"counted": false,
                "reason": "no layout is recorded in the store, so shards are not counted per layout; give each layout with --layout NAME=PREFIX to count them",
                "unmatched_shards": []},
    "placements": {}
  },
  "restarts": [
    {"run": "E", "proven": true, "checkpoint": "E@6", "segment": "E/seg-1", "step": 6,
     "basis": "the last checkpoint whose consumed history is completely traced and holds no exposure",
     "from_scratch": null,
     "exact_rerun": {"run_record_replayable": true, "segment": "E/seg-2", "step": null,
                     "basis": "stated by the verification of the repair, once the successor shards exist",
                     "step_comparison": null}}
  ],
  "exposed_checkpoints": [
    {"checkpoint": "E@9", "run": "E", "segment": "E/seg-2", "step": 9, "answer": "contains",
     "first_exposure": {"segment": "E/seg-2", "step": 7}, "level": "consumed",
     "basis": "whole-sample read assumed (F9, L9)", "untraced_steps": []}
  ],
  "checkpoints_not_covered": [
    {"checkpoint": "Z@1", "segment": "ghost/seg-1", "step": 1, "run": null,
     "answer": "not covered", "reason": "record missing"}
  ]
}
```

The keys of the table's report rows are keys of this document: `action`,
`new_parent`, `withdrawn_parent`, `regenerate_with`, `counts`, `missing_link` and
`review_scope` are those of a record entry or an artifact entry, and `count`, the
number of records a row names, is what a reader gets by counting the entries. The
keys of Check 9's rows are `records_removed` (a number in a JSONL entry and in a shard
entry, and the list `removed` for the records themselves), `rows_in_successor`,
`documents_in_successor`, `needs_successor` and `kept` (of `mixtures`), `counts`, the
lists `keep` and `rewrite` (of a group) and `exposure_free_restart`, which is the
`checkpoint` of the `restarts` entry of the run.

**Which artifacts have an entry.** Every registered JSONL file whose role is
"release" or "import", every shard that has a row, and every mixture. Source files,
generation outputs and the pipeline files between them have none in `artifacts`: the
records of them that the report names are named through the release rows that copy
them, and the files themselves that the request does not clear are in `upstream_files`
(below). A run and a checkpoint have no action; they have restart points. The entries
come in the order of the artifacts' registration.

**The files upstream** (plan 4.1, BF-6). `upstream_files` comes right after `artifacts`
and lists every registered file of kind jsonl that `artifacts` does not hold (a source
file, a generation output, an intermediate file ahead of a release or a shard) whose
answer to the request is not "does not contain", in the order of registration. An
entry is `{"artifact", "instance", "kind", "role", "answer", "action", "reason"}`, and
where the answer is "may contain" it also carries `names` and `failing`, the answer's
own named cause and failing link (section 6). Its `action`:

- null for a generation output that the repaired side reuses as the same instance
  (the records of it that the request reaches leave the release that copies them, and
  the output itself is not rebuilt), and for a revision named by the request: the
  repair never edits a source revision. The `reason` of a generation output says that, and the `reason` of
  the revision is "the revision the request withdraws" for a request that names the
  whole revision and "a revision the request names: N of its records are withdrawn, the
  rest stay" for a request narrowed to records, N being the number of records the
  request names in that file;
- "rewrite" for an intermediate file ahead of a shard or a release whose answer is
  "contains": the repair writes it again. A file is told from a generation output by
  the generation edges the store holds for its records: a file that holds one is a
  generation output;
- "unresolved" for a file whose answer is "may contain" or "not covered", and for a
  source file the request does not name whose answer is "contains", with the reason.

These files are kept out of `counts` and `bytes_to_rewrite`, and out of `groups`, `mixtures`
and every other key, which are as the form above gives them. The replica
(`replica/repair.py`) and `verify` read nothing of `upstream_files`: a report with the
list changed or left out gives the same repair and the same verification.

**The action of a record** (plan 4.1). A record is judged by the verdict the proof
of plan 2.2 gives it for the request, which `coverage.Session.occurrence` states:

- the verdict "does not contain": **keep**. A record is kept only where the proof
  shows the withdrawn source is not behind it; "no chain was found" is not enough;
- the verdict "may contain": **unresolved**, with `missing_link`, `failing`, `names`
  and `review_scope` (below), and no other action (plan 4.1);
- the verdict "contains" where a hop of the chain, at any place in it, is a file-level
  edge, a file-level link or a link by a publisher's pointer (which names no record
  either), and so there is no record-level evidence behind the record: **unresolved**
  as well, with the same fields. A file-level link says that
  the file derives from the source and cannot say which of its records the source
  reaches (plan 3.2, L1): the file answers "contains", each record is
  "may contain" once the request names records, and the report lists the record as
  unresolved, with the whole file as the scope to review (Check 6, the import);
- the verdict "contains" through record-level evidence: **retire**, **re-source** or
  **promote** by the rules of
  plan 4.1 and Decisions 10 and 15: a descendant is retired when any parent of an
  "all" edge is in scope, or every parent of an "any" edge is; a record that is the
  withdrawn source's own copy, through a single-parent copy edge with no equivalent
  copy, is retired (DECISIONS.md D-19); a release row whose "any" edge selected a
  withdrawn parent while an equivalent parent remains is re-sourced, with
  `withdrawn_parent` the selected parent and `new_parent` the equivalent one that
  remains; and the dedup loser of that row, the record the step dropped with the
  reason "dedup loser", is promoted. Re-source and promote are the two entries of one
  event.

`record` is the record the action is about, as `artifact:locator` in the form a
request's `records` use (`R1:L7`, `src/B:L1`). `artifact` is the affected artifact
the action belongs to: for a promoted record it is the release the record returns
to, and `record` is the dropped line in the file the drop names. `chain` is the
evidence chain of plan 4.1's "reason", in the form of an answer's chain (section 6),
for the actions retire, re-source and promote (for a promote, the chain of the row
whose selected parent was withdrawn), and empty otherwise. For a **retire**,
`regenerate_with` is the name of the generation job that made the record, as the
nearest generation edge of its chain keeps it (DECISIONS.md D-114 and D-24, so the report names
the job even when its harness log is gone), and `step` is that edge's step as
`{"name", "version", "parameter_digest"}`, so that the team can run the job again
(plan 4.1); both are null for a retired copy of the source's own record, where no
generation edge lies on the chain. For an **unresolved** record `review_scope` is
`{"artifact", "records", "count"}`: the artifact, `records` the word
`SCOPE_EVERY_RECORD_OF_FILE` where the evidence is a file-level link and
`SCOPE_THE_RECORD` otherwise, and `count` the number of records a person has to
review (the line count of the file for the whole file). `missing_link` is, where the
evidence is a file-level link, `MISSING_LINK_FILE_LEVEL` with the member put in (the
table's sentence "the record's own parent; only the file-level link to A/r1 is
recorded"); otherwise the `detail` of the failing link, whose wording is not fixed.
`failing` is the failing link in the form of an answer (section 6) and `names` the
named cause of the record's verdict.

**Which records an artifact lists.** An artifact whose action is keep lists none:
every record of it is kept. An artifact whose action is rewrite or unresolved lists
every record it holds, the kept ones too, in the order of the file, and then its
promote entries. A wrong report is made by changing the `action` of one entry of such
a list, to retire a kept record or to keep a retired one (the states
`r.descendant-kept` and `r.unrelated-paraphrase-retired`).

**The action of an artifact** (plan 4.1). `answer` and `chain` are the artifact's own
answer to the request, as `query` gives it, and the action follows from it: "does not
contain" is **keep**, "contains" is **rewrite** (the artifact is rebuilt without the
affected occurrences, a shard re-tokenized), and "may contain" or "not covered" is
**unresolved**. Two cases are decided by the records. A record of an artifact whose
answer is "contains" can itself be unresolved, and the artifact is still rewritten
when some other record has a definite action (retire, re-source or promote); an
artifact whose answer is "contains" only through file-level evidence, so that every
record in scope is unresolved, is unresolved itself. A shard whose answer is "does not
contain" is rewritten and not kept when it receives a record. An artifact whose answer
is "may contain" and none of whose records is shown in scope is not kept: nothing
shows it clean, and a successor that holds the record answers "may contain" again. A shard
or a mixture whose own answer is "does not contain" has a fourth word where a record that
returns to a release may land in it and the report does not say where: **clean, place not
stated** (`ACTION_CLEAN_PLACE_NOT_STATED`, "Where a returning record lands" below). `instance`
is the artifact's revision, the instance id the report was made for. `successor` is
`successor of <artifact>` for a rewrite and null otherwise (section 9.2). For a JSONL
file `records` is its line count, `counts` the five record actions,
`records_removed` the retire count (the number of rows that leave the successor; it
is a count and no action), `removed` those rows as locators, and `rows_in_successor`
the rows the successor holds: keep, re-source and unresolved (a promote adds no row:
the re-sourced row is the one the promoted record serves). For a shard `documents` is
its document count, `removed` the documents (`D<n>`) whose record is retired,
`records_removed` their number, `documents_in_successor` `documents` less
`records_removed`, and `receives` the promoted and re-sourced records, as
`artifact:locator`, whose text the shard does not hold today and would hold in the
successor: a shard that already holds the copy receives nothing (in the fixture, shard 1
holds B's copy of X from the source file, so the promotion in the release changes nothing
in it and its action is keep; "already holds" is defined below), and a shard that does
receive a record is rewritten even when none of its own lines is in scope. A shard that
receives records because a placement names it has `documents_in_successor` equal to
`documents` less the documents that leave, plus the records it receives. For a mixture
`rewritten_shards` names the shards of its snapshot whose action is rewrite and
`needs_successor` is true exactly when there is one. `bytes_to_rewrite` is the size
of the files of an artifact whose action is rewrite (a shard: its two files), read
from the disk when the report is made, and null where the file is not at its recorded
place; the store holds no size.

**The counts.** `counts.records` adds the record counts of every JSONL entry;
`counts.shards` and `counts.mixtures` count the entries whose action is keep or
rewrite; `bytes_to_rewrite` adds the sizes that are known and `bytes_unknown` names
the artifacts whose file was not found; `unresolved_artifacts` names the artifacts
whose action is unresolved; `place_not_stated` is `{"shards": [names], "mixtures":
[names]}`, the artifacts that read the fourth word, a sibling key of `unresolved_artifacts`
(the two counts keep their two words and count none of them). `mixtures` answers plan 1.1's
question, which mixtures need a successor: `needs_successor` and `kept`, each a list of
mixture names, `unresolved` and `place_not_stated`; every mixture is in exactly one of the
four. The report prints them; the listing of section 6 keeps the keys it has.
`groups.by_run` groups the shards by the runs whose segments' mixtures snapshot them,
and `groups.by_layout` by the `--layout` prefixes (an empty object where none is
given); a group is `{"counts": {"keep", "rewrite"}, "keep": [names], "rewrite":
[names], "unresolved": [names]}`. `groups.layouts` states what `by_layout` could not
(plan 4.1, BF-7): `{"counted": bool, "reason": text or null, "unmatched_shards":
[names]}`. Without `--layout` `counted` is false, `reason` says the store records no
layout and names the flag, and `unmatched_shards` is empty; with layouts named `counted`
is true, `reason` is null, and `unmatched_shards` lists, in the order of registration,
every shard that no named prefix matches. The per-run groups are unchanged, and a shard
that reads the fourth word is in none of the lists `keep`, `rewrite` and `unresolved` of a
group (the lists keep their words).

**Where a returning record lands** (D-141). A *returning copy* is a record the report
promotes, or re-sources a row onto (its `new_parent`), from any artifact entry. The
store records no layout, so the report reads two things of the shards of one group (the
shards of a named layout; the shards that match no named layout together; with no layout
named, all shards as one group):

- **A shard already holds a copy** when a document of the shard that the report keeps as it
  is (the verdict of the document is clear) has a map parent from which the copy is reached
  by following only single-parent, record-level, non-generation edges (the copy itself, its
  parent, its parent's parent). No "any" edge is followed and no equivalent copy is counted;
  a row of candidates, a shard with no usable map and a document the report changes hold
  nothing.
- **Where the placement states the copy lands.** `--placement NAME=FILE` (docs/cli.md) gives
  `{"form": "waterwhite.placement/1", "by": "source", "shards": {source artifact: shard
  store name}, "otherwise": shard store name or null}`. A copy lands in the shard named for
  its source: its own artifact where that is a registered source file, else the source at
  the root of its single-parent record-level chain, else `otherwise`; null is "not stated".
  Every shard the file names, a value of `shards` and `otherwise` where it is not null,
  must begin with the prefix of the layout the placement is for. A file that names one
  that does not is refused as a usage error before the report is built: the error names
  the file, the layout, its prefix and the first such shard, and no report is printed.
  A target is a landing only where it lies inside the prefix and the store does not hold it.

With a placement stated for the layout, a copy counts as held only by the shard the
placement names for it. A named shard that does not hold the copy reads rewrite, its
`receives` names the copy and it gets the successor name; a named shard that is rewritten
already takes the copy into `receives`; a shard the store does not hold gets no artifact entry
and no count moves: it is named in `groups.placements[NAME].landing`. With no placement, or
for a copy the placement does not place, and while some copy of the report is neither placed
nor held by a shard of the group, every shard of the group whose own answer is "does not
contain" reads **"clean, place not stated"**: its entry keeps its fields and gains
`returning` (the copies that are placed nowhere and held by no shard of the group, as
`artifact:locator`) and `place` ("not stated"); a mixture over such a shard and no rewritten
one reads it as well (with `returning` the union, and `place`). A shard that is rewritten or
unresolved keeps its action. The word is not "keep" and does not begin with it.

`groups.placements` is a sibling of `groups.by_run`, `groups.by_layout` and
`groups.layouts`: for every named layout `{"placement": {"path", "sha256"} or null,
"landing": [{"shard", "in_store": false, "receives": [copies]}], "place_not_stated": [shard
names]}`, `placement` the file read (null where none was given for the layout) and
`place_not_stated` the shards of the layout that read the word. It is `{}` with no
`--layout`. `groups.by_layout` and `groups.layouts` do not change.

**The restart where the place is not stated.** A run that has a segment on a mixture that
reads the fourth word is never given "no exact rerun": its `exact_rerun` is the phase start
of its earliest segment over a mixture that is rewritten or reads the word, as an assumption
(`RESTART_BASIS_PHASE_START`, `STEP_COMPARISON_NOT_POSSIBLE`), because `verify` names no
corrected mixture for it. `run_record_replayable` still says what the run records say, so a
replayable run in this case has `run_record_replayable` true and the phase start as its step
(the exception to the two cases below).

**Where an entry's keys are null.** A kept JSONL file has `records`, `counts` (all of
its lines under keep), `records_removed` 0 and `removed` empty, and `rows_in_successor`
null: it has no successor. A shard has no `records`, `counts` or `rows_in_successor`,
and its documents are not listed in `records`: they are judged through the map a reader
gets, and the entry carries `documents`, `removed`, `records_removed`,
`documents_in_successor` and `receives`. A document is judged by the record its map row
names, and a document whose row names a set of candidates is retired when the walk
shows every candidate in scope. A mixture has no file, so its `bytes_to_rewrite` is null
and it is not named in `bytes_unknown`.

**Which parent a re-source takes.** For a row of an "any" edge whose selected parent is
in scope, the parents the edge names besides the selected one are asked in the order the
edge gives them. The first one that remains (the walk shows it clear of the request) is
`new_parent`, and the row is re-sourced. If none remains and some of them is "may
contain", nothing shows that an equivalent parent remains and nothing shows that none
does: the row is unresolved, with the failing link of that parent. If every one is in
scope, the row is retired. A re-source gets a promote entry only where the step that made
the row recorded a drop with the reason "dedup loser" whose output is the release and
whose input is `new_parent`; the edge's group key is not read, since a step may give its
own. Without such a drop there is no record to bring back and the report names none.

**The missing link of a pointer.** Where the evidence is a publisher's pointer that names
a member and no record, `missing_link` is "the record's own parent; only a publisher's
pointer to MEMBER is recorded" and `review_scope` is the whole file, as for a file-level
link.

**Reading a verdict.** The report reads verdicts and answers and builds none. It asks
`coverage.is_clear(verdict)` of a record's verdict (or an artifact's) and
`Answer.is_negative` of an answer; a verdict that is not clear says what it is by the
`chain` or the `failing` link it carries. Both are reads only; the test that scans the
source for the construction of the negative (`docs/coverage.md`, "How 'does not
contain' is closed off") keeps every other module from naming the verdict "clear" or
the negative answer.

**Restart points** (plan 4.3, Decision 23). `restarts` has one entry per run that has
registered checkpoints, and one for a run that has a stale segment and registers none
(below). Where `proven` is true, `checkpoint` names the last
checkpoint of the run whose consumed history is completely traced, passes the proof
of plan 2.2 and holds no exposure, and `basis` is `RESTART_BASIS_LAST_CLEAR`. Where
none qualifies (exposure in the first segment, or an earlier step untraced) `proven`
is false, `checkpoint`, `segment` and `step` are null, `basis` is the sentence
`RESTART_NOT_PROVEN`, "no exposure-free restart is proven", and `from_scratch` names
the first step of the lineage (`{"run", "segment", "step", "basis"}`: the first
segment of the run the lineage descends from, and `RESTART_BASIS_LINEAGE_START`) as
the restart that is free of exposure by construction. The wording says "proven"
because the tool can prove a restart clean only from traces it holds; a restart it
cannot prove is not called clean. `exposed_checkpoints` lists the checkpoints whose
answer is "contains" or "may contain", with the answer's own fields; no count of
reads is given, since exposure counts are an extension that is off (plan 2.4).

**The checkpoints that answer "not covered".** `checkpoints_not_covered` is the last
key of the report, always present: `[]` when there is none. It has one entry for every
registered checkpoint whose answer is "not covered", in the order the checkpoints were
registered, whether or not a run reaches it. An entry is `{"checkpoint", "segment",
"step", "run", "answer", "reason"}`: `checkpoint` is the name the entries of
`exposed_checkpoints` use (its first name, else its instance id), `segment` and `step`
are the checkpoint's own, `answer` is "not covered" and `reason` is the answer's reason
(`REASON_RECORD_MISSING`, "record missing", for a checkpoint whose segment has no row).
Both `answer` and `reason` are read from the answer and not from a check of the
segment made for the report. `run` is the run the store gives the segment, and null
when the store has no such segment. An artifact of the checkpoint kind that has no
checkpoint row (the store's calls never make one) is named too, with `segment`, `step`
and `run` null. A checkpoint registered through the library over a
segment that is not registered (`Store.register_checkpoint`, docs/store.md) is in no
run, so no entry of `restarts` and none of `exposed_checkpoints` can name it; this key
does. The invariant: every registered checkpoint is negative (in none of the two lists),
or in `exposed_checkpoints`, or in `checkpoints_not_covered`. The text form prints
`not covered: NAME (REASON)` for each entry, after the `exposed:` lines (docs/cli.md).

**The exact-rerun restart in the report** (plan 4.3, BF-5). The exact-rerun restart
itself is not computed by the report: it needs the corrected data order, which needs
the document lengths of the successor shards, and those exist only after the team's
builder has made them from this report. It is stated by `verify` (section 7.3). The
report states, for every run that has a stale segment, which of the two cases the run
is in. A segment is stale where its mixture needs a successor under the request: the
mixtures the report lists under `needs_successor` say which. A run that has a stale
segment and no registered checkpoint has an entry in `restarts` too: the entry of a
run for which no exposure-free restart is proven, with its `exact_rerun`.

Every entry of `restarts` ends with the key `exact_rerun`, which is null for a run
that has no stale segment, and otherwise the object
`{"run_record_replayable", "segment", "step", "basis", "step_comparison"}`, where
`segment` is the earliest stale segment of the run.

- `run_record_replayable` is true when the run records of the run's stale segments each
  hold every setting a replay needs (section 7.3 gives the definition): the sequence
  length, the random seed, the split, the add-extra-token setting, and the train
  iterations or the train samples with the global batch size. It is one check, made by
  one function (`join.missing_replay_settings`) that `join.replay_arrays` refuses by as
  well, and the report and `verify` both ask it, so that they cannot state different
  restarts for one run. The key is not called "seeded", because the word is
  `verify --mode`'s (plan 4.2's build mode, the byte compare of the shards), which has
  nothing to do with it.
- For a replayable run `step` is null, `basis` is `RESTART_EXACT_BY_VERIFY` and
  `step_comparison` is null: the restart is left to `verify`, which can state it
  after the successor shards exist, and the report names no step.
- For a run that is not replayable `step` is the phase start, the first step of the
  earliest stale segment, as an assumption: `basis` is `RESTART_BASIS_PHASE_START` and
  `step_comparison` is `STEP_COMPARISON_NOT_POSSIBLE`. `verify` states the same
  segment, step and basis for the run, and without trying the replay.

The text form of the report also prints the files upstream of the artifacts (each with
its action or "no action" and its reason) and, from `groups.layouts`, a line when the
report made no count per layout or when a shard matched no named layout. It prints one line
for each shard that reads the fourth word (its own contents are clean, the copies it may
receive, their place not stated), and for each placement stated the file read and the shards
outside the store that copies land in. It prints one
line for each run that has a stale segment, under
the line of the run in the list of restarts, with the same statement. The check reads
the run records as they are registered, and a registered run record cannot be changed
through the tool; a `verify` handed another store than the one the report was made
from is refused where that store lacks an artifact the report names or expands the
report's request to less (section 7.3, DECISIONS.md D-74 and D-80).

**While the store is being written.** `report` reads through one coverage session. If
the store is written during the command, it ends with "store written during the
query" (section 2) and prints no report.

### 7.2 The sides file

The two sides of a repair are produced by the team's builder, and only the repaired
side is registered in the store (its successors, under the names of section 9.2).
`waterwhite verify` is therefore handed both sides as explicit files in a JSON
document, and never reads a builder's folder: the tool cannot know a team's layout,
and the wall of section 1 keeps the expected answers out of its code.

```json
{
  "form": "waterwhite.sides/1",
  "repaired": {"R1": "repaired/releases/R1.jsonl", "E/shard-3": "repaired/shards/shard-3"},
  "reference": {"R1": "reference/releases/R1.jsonl", "E/shard-3": "reference/shards/shard-3"}
}
```

The keys of `repaired` and `reference` are the names of the historical artifacts,
the ones the report names. A value is the path of a JSONL file, or the prefix of a
shard (the path without `.bin` and `.idx`), relative to the folder of the sides file
or absolute: text, and any other value (a number, null, a list) is refused as "request
refused", naming the side and the artifact. The repaired side names every artifact the report lists with the action
keep, rewrite or "clean, place not stated" that the builder wrote. The reference side is
built by the same builder from source revisions that never held the withdrawn content,
and names the same artifacts. A name on one side only is not compared and is listed in
`differences`. The sides also name each shard the report lists under a layout's
`groups.placements[NAME].landing` (section 7.1) that the build holds, after the artifacts
of the report: a landing shard has no artifact entry, and the report names it there. The
document is written by whoever built the sides; the replica writes it with `repair
--sides` (section 9.5), which leaves out a landing shard the build does not hold.

The list of names in a sides file is not what says which shards a build wrote. `verify`
reads that from the two builds themselves (section 7.3): the shards the repaired store
registers as `successor of <name>`, and the shard files on the disk of the reference
build, in the folder of the reference files the sides file names. A shard a build wrote
that neither the report nor the sides file names is therefore found, and is told apart
from one that was never written. A shard the report names only under a landing counts as
named by the report.

### 7.3 The verification (`verify`)

`waterwhite verify --store FILE --report FILE [--sides FILE] [--mode seeded|unseeded]
[--text-field FIELD]` verifies a repair. `--report` is the report the repair was
driven by, read as a file and never computed again: a report computed after the
successors are registered would be another report, and a wrong report is exactly
what the verification is to catch. `--sides` is the file of section 7.2; without it
the comparisons of the sides are "not compared" and the rest is made. `--mode` says
whether every step of the build was seeded and defaults to "unseeded", the weaker
claim (plan 4.2); `--text-field` is the field of a JSONL line that holds its text
(default `text`). The command opens the store read-only and writes nothing.

What `verify` reads of the report: `form`; `request`, which it puts again to the
store (`source` is text, `members` and `records` lists of text); for each of `artifacts`,
`artifact`, `instance`, `kind` and `action` (text) and `successor` (text or null), and,
where a rewritten artifact's own count is the record of what it leaves, `rows_in_successor`
and `documents_in_successor`; `counts.records.unresolved`; and the shards named under
`groups.placements.*.landing[].shard`. A report whose `form` is another, that lacks one of
these, or that gives one as something it is not (a number where text is read, an object
where a list is) is refused ("request refused"). A report or a sides file that cannot be read is
"usage". Of the report, `verify` reads no other key: `upstream_files` (section 7.1) is never read.

`verify` never certifies a repair for less than the report is about. It is not keyed on
which store file it is handed (the report is printed from one store and checked on a
copy where the successors are registered), but on what that store holds. Two refusals
("request refused", exit 2, nothing on standard output) give the result: the request
of the report is put to the store again, with the members and records it lists, so a
member or a record of the report that the store no longer holds ends in "member not
registered" or "record not in source", and the scope the store expands to is never
narrower than the report's; and an artifact entry whose `instance` the store does not
hold, a report that names one artifact (by name, or by an instance) in two entries, and an
entry, other than a rewritten artifact's, that is not named by a name the store holds its
instance under end in "request refused", naming the artifact. A faithful copy of the store, with
the successors registered, holds every instance of the report and is certified.

```json
{
  "form": "waterwhite.verification/1",
  "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
  "mode": "seeded",
  "unresolved_records": 0,
  "multisets": "equal",
  "differing_records": 0,
  "differences": [],
  "releases": [{"artifact": "R1", "repaired": 21, "reference": 21, "equal": true}],
  "per_shard_digests": "equal",
  "shards": [{"artifact": "E/shard-3", "repaired": "<digest>", "reference": "<digest>",
              "equal": true}],
  "kept": [{"artifact": "R2", "identical": true, "recorded": "<digest>", "found": "<digest>"}],
  "successors": [
    {"artifact": "R1", "successor": "successor of R1", "kind": "jsonl", "registered": true,
     "instance": "<instance id>", "answer": "does not contain", "names": null,
     "holds_unresolved": false, "matches_sides": true,
     "answered": {"...": "an answer in the form of section 6"}}
  ],
  "history": [
    {"artifact": "R1", "instance": "<instance id>", "instance_unchanged": true,
     "answer": "contains"}
  ],
  "caches": [
    {"run": "E", "segment": "E/seg-2", "mixture": "E/mix-2",
     "corrected_mixture": "successor of E/mix-2", "digest": "<digest>",
     "corrected_digest": "<digest>", "stale": true, "rebuilt": true,
     "rebuilt_from": "an empty cache folder", "reason": null}
  ],
  "restarts": [
    {"run": "E", "segment": "E/seg-2", "step": 7,
     "basis": "the first step at which the corrected order differs from the historical one",
     "step_comparison": "made"}
  ]
}
```

- `multisets` is the comparison of plans 4.2 and 4.4: for every JSONL file named on
  both sides, the multiset of (fingerprint, multiplicity) of its lines, the
  fingerprint being the tool's own (section 3) of the text after the store's
  normaliser. It is "equal", "differ", "not compared" (no file is named on both
  sides) or "not claimed": equality is claimed only for a report with no unresolved
  record (plan 4.2), so with `unresolved_records` above 0 the word is "not claimed"
  and the differences are still listed. A file or shard the report keeps or rewrites
  that the sides do not name on both sides keeps the word from reading "equal": it is
  "differ" where a compared file differs, and "not compared" otherwise (one exception,
  below: a file the repair leaves no row), and so does a shard that both builds wrote
  and the report does not name (a bullet below), or that the report names under a landing
  and the sides do not name on both sides. `differing_records` is the sum, over every
  fingerprint of every compared file, of the difference between the two
  multiplicities, so one kept descendant that the reference does not hold makes it 1.
  `differences` lists them as `{"artifact", "fingerprint", "repaired", "reference",
  "locators"}`, the locators being lines of the repaired file; `releases` has one
  entry per compared file with the two line counts.
- `per_shard_digests` compares, for every shard named on both sides, the shard digest
  of its two files (section 3). Bytes match only where every step is seeded (plan
  4.2), so with `--mode unseeded` it is "not compared" and `shards` still gives the
  two digests. A shard the report keeps or rewrites that the sides do not name on both
  sides keeps this word from reading "equal" in the same way: it is "not compared"
  while a shard lacks, and "differ" where a compared shard differs. So does a shard that
  one build alone wrote and the report does not name: the word is then "differ" (in the
  seeded mode, where a shard was compared; otherwise it stays "not compared").
- A file or shard the report keeps or rewrites that the sides do not name on both sides
  is named in `differences` (with a `note`, no counts and no locators, and not in
  `differing_records`) once a sides file is given: a name on one side only as it was,
  a name on neither side with the note "named on neither side, not compared". Without
  `--sides` `differences` is empty and the comparisons read "not compared".
- A shard the report names under a layout's `groups.placements[NAME].landing` is a shard
  the report names, so it is compared like a rewritten shard: where the sides name it on
  both sides it is one of the `shards` (its digest feeds `per_shard_digests`, and its being
  named on both sides is what stops it holding either word back), and where they name it on one side only it is
  listed in `differences` as a name on one side only. A landing shard that both builds
  wrote and that the sides do not name on both sides is a file the comparison lacks: it is
  named in `differences` (a name on neither side with a `note` that says the report names
  it under a landing; a name on one side with the line of a name on one side only), and `multisets` and `per_shard_digests` do not read "equal" while it
  lacks. A landing shard that only one build wrote is named as a shard one build alone
  wrote, and the word of `per_shard_digests` is then "differ" in the seeded mode. A
  landing shard that neither build wrote is not mentioned. Without `--sides` nothing is
  said of a landing shard.
- An artifact of the report that is rewritten, is on neither side, and that the tool's
  own record shows is left no record does not hold a comparison back: the report's
  entry gives `rows_in_successor` 0 for a file, or `documents_in_successor` 0 for a
  shard, and the store holds no successor of it with records. A repair does not write
  such a file and a reference built without the withdrawn source has none, so there is
  nothing to compare. Its `differences` entry has a note of its own ("the report leaves
  the file no row, so neither side writes it; nothing to compare", and the same for a
  shard with "document"), not the note of a file that was lost, and no new key is
  added. The count is the report's, never the replica's account of what it registered;
  a report that gives no count, or an artifact whose count is above 0, is a file the
  comparison lacks as before. A shard whose documents are all re-sourced to other
  records is not shown by the report's `documents_in_successor`, which counts the
  documents that stay, so it is not read as left empty.
- A shard that both builds wrote and that the report does not name is a shard `verify`
  compared nothing of: `multisets` and `per_shard_digests` do not read "equal" while one
  exists, they read "not compared" (a "differ" stays "differ"), and the shard is named in
  `differences` with `repaired` and `reference` null, no locators and a `note`. A shard
  that only one build wrote is a difference of the same kind (`differences`, with a
  `note` that names the build), and the word of `per_shard_digests` is "differ" in the seeded mode where a shard was compared (otherwise "not compared"). Which
  shards each build wrote is read from the builds and not from the sides file: the
  repaired build's are the shards of the store registered as `successor of <name>`
  under the prefix of the layout (the prefix the report's shard names share, in front of
  the file name the sides file gives the shard), the reference build's the pairs of
  `.bin` and `.idx` files in the folder of the reference files the sides file names for
  the layout. Nothing is said of a layout whose reference folder the sides file does not
  give, and nothing of any shard without `--sides`. A shard the report names is judged as
  before, by the bullets above.
- `kept` has one entry for every artifact the report keeps: `identical` is true when
  the file the repaired side names for it has the digest the artifact was registered
  with (plan 4.4: kept artifacts are reused untouched and are byte-identical
  afterwards); it is null without `--sides` or where the repaired side names no file
  for it. `recorded` and `found` are the two digests. The file is compared with the
  content digest of the report's own `instance` and with nothing else: where the store
  does not hold that instance `identical` and `recorded` are null and the entry gains a
  `reason` that says so, whatever artifact the name resolves to now. (A `verify` of
  such a report is refused before this entry is made, above; the entry says so for a
  caller of `verify_sides` that reaches it.)
- `successors` has one entry for every artifact the report rewrites. The tool builds
  no answer here: `answered` is `coverage.answer` of the registered successor under
  the report's request, and `answer` its word. A successor shard that passes the
  proof and holds no unresolved record "does not contain"; one that holds an
  unresolved record answers "may contain", naming the link of the proof that failed
  and no cause of the rebuild, and `holds_unresolved` is true; a successor with a
  missing map answers "not covered". `registered` is false and `answer` null where no
  artifact is registered under the successor's name; the verification is then not
  complete and says so. `matches_sides` says that the file the repaired side names has
  the digest the successor is registered with (null without `--sides`): an answer
  about another file says nothing about the one that was compared. It is also null for
  a successor that is not a file or shard, such as a mixture registered under the name
  of a shard's successor, and for a file or shard of the other kind than the artifact it
  replaces (the file the side names is of the replaced artifact's kind and is not read as
  the other's): `verify --require-held` does not hold for it (docs/cli.md, "Gating on the
  verification").
- `history` has one entry for every artifact the report rewrites: the historical
  artifact still answers "contains" for the withdrawn source, because history keeps
  its attribution, and `instance_unchanged` is true when its instance id is the one
  the report holds. The re-query that must find nothing is the one on `successors`.
- `caches` has one entry per run segment of the runs whose mixtures are rebuilt (plan
  4.4). The mixture digest is `ids.mixture_content_digest` of the content digests of
  the shards of the mixture's snapshot, in the trainer's order: it covers what the
  index caches were built from, and nothing of the folder the files were read from.
  `digest` is that of the snapshot the segment was registered with.
  `corrected_mixture` is the mixture the corrected run reads: the mixture itself where
  the report keeps it (its digest is then `corrected_digest` too, and `stale` is
  false), and the registered mixture called `successor of <mixture>` where the report
  rewrites it. Where the report rewrites the mixture and no successor is registered,
  `corrected_mixture`, `corrected_digest` and `stale` are null. `stale` is true when
  the two digests differ; Megatron does not detect that, because its cache key holds paths
  and not content (plan 4.4). A stale segment is `rebuilt` when its corrected order is
  derived again from the run record and the document lengths of the successor shards,
  the kept cache files never read (`rebuilt_from` is `REBUILT_FROM_EMPTY_CACHE`);
  where that is not possible `rebuilt` is false and `reason` says why (the run record
  lacks a setting, a shard is not registered). A segment that is not stale is left
  alone: `stale` false, `rebuilt` false, `rebuilt_from` null.
- `restarts` has one entry per run that has a stale segment: the exact-rerun restart
  of plan 4.3. A run is replayable when its run records hold every setting a replay
  needs: the sequence length, the random seed, the split, the add-extra-token setting,
  and the train iterations or the train samples with the global batch size. This is
  the check the report makes too (section 7.1), by the same function, and it has
  nothing to do with `--mode`, which is plan 4.2's build mode. For a replayable run
  `step` is the first step at which the corrected data order differs from the
  historical one, found by building the corrected order and comparing; `basis` is
  `RESTART_BASIS_ORDER_DIFFERS` and `step_comparison` is `STEP_COMPARISON_MADE`. For
  a run that is not replayable `step` is the phase start, the first step of the
  earliest stale segment of the run, `basis` is `RESTART_BASIS_PHASE_START` and
  `step_comparison` is `STEP_COMPARISON_NOT_POSSIBLE` (F7), and the replay is not
  tried. The report states the same segment, step and basis for that run.

The keys of Check 9's rows are keys of this document: `multisets`,
`unresolved_records` and `differing_records`; `per_shard_digests`;
`byte_identical_afterwards` and `kept_shards_byte_identical`, which are `identical` of
the `kept` entries (of the artifact named, and of every kept shard of the kept-shard
list); `answer` of a `successors` entry, and of a `history` entry for the historical
release; `stale`, `rebuilt` and `rebuilt_from` of a `caches` entry; and
`exact_rerun_restart_step`, which is `step` of the `restarts` entry of the run, where
`step_comparison` is "made" (under the fallback F7 the row reads `basis` and
`step_comparison`).

What the form above leaves to the reader of the code, as built:

- For an artifact the report rewrites whose `successor` is null, the successor is
  looked up as `successor of <artifact>`.
- A line that holds no text (blank, not JSON, no text field, empty after the
  normaliser) has no fingerprint; such lines are counted together under the
  fingerprint null, so a difference in their number is a difference of the sides.
- A name that only one side holds, or that the report does not list, is added to
  `differences` with `fingerprint`, `repaired`, `reference` null, no locators and a
  `note`; it is not added to `differing_records`. A mixture has no file and is not
  compared.
- `holds_unresolved` is true when the report's own check (section 7.1, the judgement
  the report's entries read, which `verify` reads too, so that `verify` is never less
  strict than the report) would count some record of the successor (of any
  shard of a mixture's snapshot) unresolved: one the proof cannot decide, or one with
  a hop in its chain, at any place, that is a file-level edge, a file-level link to a
  member or a link by a publisher's pointer, or one whose other parents decide
  nothing (for a document of a shard, also where the document's own verdict is any of
  these); false for a shard that has no map, which has no record; null where no
  successor is registered. `kept` has
  `identical` and `found` null for a mixture or a checkpoint.
- An artifact whose action is "clean, place not stated" (section 7.1) is read by `verify`
  as follows. The word is an action `verify` accepts. The check of kept artifacts, the
  byte identity of the file the repaired side names, never applies to it: the shard may
  have received a returning copy, so its bytes are not claimed to be the recorded ones,
  and it has no `kept` entry. No successor is required of it, so it has no `successors`
  or `history` entry. The files the sides name for it are compared like those of a kept
  or a rewritten artifact, and a shard of it that the sides do not name on both sides
  keeps the two comparisons from reading "equal". A mixture that reads the word is read
  as a mixture the report leaves unresolved: its `caches` entries have
  `corrected_mixture`, `corrected_digest` and `stale` null, and a `reason` that says the
  report does not state where the records that return land.
- `caches` has no entry for a segment that names no mixture. A mixture that the
  report leaves unresolved, or rewrites with no successor registered, has
  `corrected_mixture`, `corrected_digest` and `stale` null and a `reason`; a mixture
  that reads "clean, place not stated" has the same entry with a reason of its own. The text
  form (`--format text`) prints a line for each such entry, with the segment, the
  mixture and the reason; the JSON form is as above and the exit code is 0.
- The corrected order of a stale segment is built with the join's own functions from
  the segment's run record, the document lengths of the corrected mixture's shards
  and the store's verified joins; the recorded digests of the index arrays are
  replaced by those of what was derived, since the corrected shards are other files.
  Where that fails for any reason (the run record lacks a setting, a shard is not
  registered or not readable, the join is not verified for the
  run's release, a cyclic run whose shuffle cannot be held) `rebuilt` is false with the
  reason, and the run's `restarts` entry is the phase start with `step_comparison`
  "not possible". The steps of a stale segment are compared in order; a step that
  either side holds no trace for, reached before a difference is found, also gives the
  phase start. Where the corrected order is built and no step reads differently, `step`
  is null.

A verification that finds a difference exits with 0: a difference is an answer; with
`--require-held` the same verification exits with 5 when its conditions are not all
held (docs/cli.md, "Gating on the verification"). The
refusals are those of `query withdraw` for the report's request ("source not
registered", "member not registered", "record not in source"), "request refused" for
a malformed document or for a file the sides file names that is not there, "store
written during the query" (section 2), "not a store", and the loud faults (exit 3)
over the artifacts it answers. Each file is read once and streamed: the verification
holds a multiset of fingerprints and not the lines.

### 7.4 The Python seam

The parts of the rebuild are written by different owners against these names; the
JSON of sections 7.1 and 7.3 is the contract, and these functions are the only places
where the parts reach one another. Each takes an open store and returns plain
JSON-ready values (`dict`, `list`, `str`, `int`, `bool`, `None`), so that a test
compares them with the forms above.

```python
# waterwhite/rebuild/__init__.py
def report_without_restarts(store, request, session, *,
                            layouts: Mapping[str, str] | None = None,
                            placements: Mapping[str, Placement] | None = None,
                            integrity=None) -> dict: ...
    # every key of 7.1 but `restarts`, `exposed_checkpoints` and `checkpoints_not_covered`,
    # which it leaves out. It
    # reads through the `coverage.Session` it is given. This is the part that needs no
    # other file: it is what the tests of the report's records, artifacts and counts call.
def build_report(store, request, *, layouts: Mapping[str, str] | None = None,
                 placements: Mapping[str, Placement] | None = None) -> dict: ...
    # the document of 7.1 in full. `request` is what requests.withdraw returned; `layouts`
    # maps a layout name to a shard-name prefix; `placements` maps a layout name that
    # `layouts` names to the placement read from its file (`placing.read_placement`). It
    # opens one coverage.Session, calls
    # report_without_restarts, and adds `restarts`, `exposed_checkpoints` and
    # `checkpoints_not_covered` from the functions of restart_free, and the `exact_rerun` key of every `restarts` entry from
    # restart_exact.exact_rerun_of_report, imported at the call.

# waterwhite/rebuild/upstream.py
def upstream_entries(store, request, session, *, integrity=None) -> list[dict]: ...
    # the `upstream_files` entries of 7.1, in the order of the artifacts' registration;
    # report_without_restarts calls it after the entries of `artifacts`

# waterwhite/rebuild/restart_free.py
def exposure_free_restarts(store, request, session) -> list[dict]: ...
    # the `restarts` entries of 7.1, one per run, in the order of the runs' registration
    # without `exact_rerun`: build_report adds that key to each entry, and adds the entry of
    # a run that has a stale segment and no registered checkpoint (unproven_entry)
def exposed_checkpoints(store, request, session) -> list[dict]: ...
    # the `exposed_checkpoints` entries of 7.1
def checkpoints_not_covered(store, request, session) -> list[dict]: ...
    # the `checkpoints_not_covered` entries of 7.1, in the order the checkpoints were registered

# waterwhite/rebuild/replay_check.py
def replayable(segments) -> bool: ...
    # the one check of 7.1 and 7.3: every segment given has a run record that holds
    # every setting a replay needs (it calls join.missing_replay_settings)
def phase_start(store, first_stale_segment: str) -> tuple[str, int | None]: ...
    # (the segment, its first step): the phase start of a run that is not replayable

# waterwhite/rebuild/verify_sides.py
def read_document(path, form: str, what: str) -> dict: ...
    # reads a report or a sides file: UsageError when it cannot be read,
    # Refused ("request refused") when it is not a document of `form`
def verify(store, report: Mapping, *, sides: Mapping | None = None, base=None,
           mode: str = "unseeded", text_field: str = "text") -> dict: ...
    # the document of 7.3. `report` and `sides` are parsed documents; `base` is the
    # folder that relative paths of `sides` are taken from. Refused ("request refused")
    # when the store does not hold the instance of an entry of the report's `artifacts`,
    # or holds it as another kind than the entry's `kind`.
    # With a sides file it also reads which shards each build wrote, from the store and from
    # the folder of the reference files (`shards_each_build_wrote`, 7.3), and not from the
    # list of names of the sides file. A shard the report names under a layout's landing is
    # compared like a shard of the report where the sides name it on both sides

# waterwhite/rebuild/verify_successors.py
def check_successors(store, report: Mapping, request, sides: Mapping | None, base) -> tuple[list, list]: ...
    # (the `successors` entries, the `history` entries)

# waterwhite/rebuild/verify_caches.py
def check_caches(store, report: Mapping) -> list[dict]: ...
    # the `caches` entries

# waterwhite/rebuild/restart_exact.py
def exact_rerun_restarts(store, report: Mapping, caches: list[dict]) -> list[dict]: ...
    # the `restarts` entries of 7.3
def exact_rerun_of_report(store, artifacts: list[dict]) -> dict[str, dict]: ...
    # the `exact_rerun` object of 7.1 for every run that has a stale segment, by run name
```

`cli.py` has two entries: `report` (calls `build_report` and prints it) and `verify`
(calls `read_document` for each file, then `verify`). The part that builds the report
and the parts that build its restarts are files that meet only at the functions named
above. The rebuild package reads verdicts through
`coverage.Session` and answers through `coverage.answer`; it builds none, and the
scanner of the proof's construction holds it to that.

## 8. Training records

### 8.1 The run record

Written by the training wrapper at training time (plan 3.3 "Training"; in the
first build the replica stands in for the wrapper). One JSON file per run segment:

```json
{
  "format": "waterwhite.run-record/1",
  "run": "E",
  "segment": "E/seg-1",
  "start_kind": "fresh",
  "parent_checkpoint": null,
  "settings": {
    "dataloader_type": "single",
    "sampler_class": "megatron.training.datasets.data_samplers.MegatronPretrainingSampler",
    "data_sharding": true,
    "micro_batch_size": 4,
    "global_batch_size": 4,
    "sequence_length": 32,
    "add_extra_token_to_sequence": true,
    "random_seed": 1234,
    "split": "1,0,0",
    "train_iters": 6,
    "train_samples": null,
    "batch_schedule": null
  },
  "versions": {
    "release_tag": "core_v0.20.0",
    "megatron_core": "0.20.0",
    "megatron_bridge": null,
    "torch": "2.14.1+cpu"
  },
  "shards": [
    {"position": 0, "prefix": "shards/shard-1", "bin_sha256": "...", "idx_sha256": "..."}
  ],
  "launches": [
    {"position": 0, "world_size": 1, "data_parallel_size": 1,
     "consumed_at_start": 0, "step_at_start": 0, "first_step": 1, "last_step": 6}
  ],
  "cache_digests": [
    {"kind": "document_index", "shard": 0, "digest": "..."},
    {"kind": "dataset_index", "shard": null, "digest": "..."}
  ],
  "shuffle_digests": [
    {"launch": 0, "epoch": 0, "rank": 0, "digest": "...", "count": 24}
  ]
}
```

- `shards` is the ordered list of shard prefixes the trainer was handed, with each
  file's digest (plan Decision 27); a prefix is relative to the run record's folder.
- `versions.megatron_core` is the installed package's version as its metadata
  gives it. It may carry a local version suffix after a plus sign: an install from
  a checkout reads, for example, `0.20.0+685fb8b`. `versions.release_tag` names
  the release and is what the list of verified joins is keyed on.
- `cache_digests` are `ids.array_digest` values of the index arrays, taken while
  the caches existed: three kinds per shard (`document_index`, `sample_index`,
  `shuffle_index`, each with its shard position) and two for the blend
  (`dataset_index`, `dataset_sample_index`, with `shard` null). A segment of one
  shard builds no blend: it has three cache digests and no blend rows.
- `shuffle_digests` belong to cyclic runs. For a sequential run the list is empty.
  One entry holds, for one launch, epoch and rank, the `ids.order_digest` of the
  whole epoch's order for that rank, whatever part of the epoch the launch read,
  and `count`, the number of indices in that order (DECISIONS.md D-16). A resume
  inside an epoch therefore records the same digest for that epoch and rank as the
  launch before it. The digests are computed in the run's own environment (plan
  3.5).
- A field the wrapper could not record is absent or null; the join then gives the
  named cause the plan fixes for it and never guesses.

### 8.2 The step record

Supplied by the operator when a run is registered (plan 3.1, F4):
`{"segment": "E/seg-2", "steps": [{"step": 8, "batch_size": 4, "last_sample_tokens_read": 16}], "checkpoints": [{"name": "E@9", "step": 9}]}`.
A step with no `last_sample_tokens_read` is not lowered. Steps are numbered inside
the run as the trainer numbers them; a warm start restarts the count (F4).

### 8.3 The oracle's log

Written by `oracle/` from Megatron's own code (plan 5.3). JSON lines, one per row a
step read, in reading order:

```json
{"segment": "E/seg-2", "step": 7, "sample_pos": 1, "global_index": 25, "rank": 0,
 "shard_position": 2, "ordinal": 4, "token_start": 0, "token_end": 21}
```

- `sample_pos` is the place of the sample inside its step. It counts micro-batch
  round first, then rank, then the place in the micro batch: with n ranks and a
  micro batch of m samples, `sample_pos = round * m * n + rank * m + place`.
- `global_index` is the index asked of the top dataset: an index into the blend,
  or into the one shard's dataset when the segment has one shard and so no blend.
- `rank` is the data-parallel rank whose sampler drew the sample.
- `shard_position` is the position of the shard in the segment's shard list.
- `ordinal` is the sequence number in the shard's `.idx`, the number the indexed
  dataset is asked for. It is the document number `D<n>` when each line that gives
  tokens is one document in input order (plan F1).
- `token_start` and `token_end` are inside the document, half-open, as Megatron's
  indexed dataset is asked for them, and include the extra token of each sample
  (sequence_length + 1 tokens per sample).

The oracle logs what the dataset's item getter is asked to read; it does not
compute spans by a rule of its own. `docs/oracle.md` describes it.

### 8.4 Cache files

A segment's data-order cache folder is the one Megatron wrote with `path_to_cache`
set, kept as written: the `.npy` index arrays (three kinds per shard and, for a
blend, two more) and each `description.txt`. The blend's `description.txt` gives
the order of the shard list.

### 8.5 The shuffle of a cyclic run

A run read with the cyclic sampler and data sharding on draws, for each epoch, one
permutation of each rank's bucket from a generator seeded with the epoch number
(`torch.Generator`, `manual_seed(epoch)`, `torch.randperm(bucket)`), and keeps it
in no file. The wrapper records, per launch, epoch and rank, the digest and the
count of the rank's whole order for the epoch (8.1, `shuffle_digests`). The join
(`docs/join.md` section 6) computes the same order by calling torch with the run
record's sizes, and the order counts only when its digest and count equal the
recorded ones. The wrapper takes its digests with its own call to torch, never
through a function of `src/waterwhite`. The join imports torch for a cyclic run
only; where it cannot, registration is refused with "torch not available" (exit
code 4) and nothing is written. torch is an optional dependency of the tool.

## 9. The fixture

### 9.1 Files

- `fixture/sources/<SOURCE>.jsonl`: one source revision per file, rows
  `{"id": "<SOURCE>-<nnn>", "text": "..."}`. Sources `A`, `B`, `C`, `Z`, each with
  one member named `<SOURCE>/r1` at revision `r1`.
- `fixture/sources/generation.jsonl`: the fixed outputs of the generation jobs,
  rows `{"id", "job", "handed": [ids], "used": [ids], "text"}`. No model is called.
- A generation output file has rows `{"id": ..., "text": ...}`.
- A release (`R1`, `R2`) is one JSONL file whose rows are `{"text": "..."}` and
  nothing else: a release row carries no id that marks its source (DECISIONS.md
  D-06). Which input each row came from is in the store's edges and in the
  replica's manifest. A text-only import has the same row form.
- A tokenizer-input file (`pipeline/<layout>/tokenizer-input/<shard>.jsonl`, also
  called a chunk file) holds the lines that survived the sanitizer, in order, with
  their sanitized text, rows `{"id": ..., "text": ...}`. Line i of the file is
  document i of the shard. The ids are the replica's own, kept up to the tokenizer
  input and never given to the tool (plan Check 3).
- The sanitizer log (`pipeline/<layout>/sanitizer.log.jsonl`) has one row per
  skipped line: `{"shard", "input", "line", "id", "origin_file", "origin_line",
  "reason", "detail"}`. `input` and `line` name the line of the chunk as the
  sanitizer read it (its sanitizer-input file), `origin_file` and `origin_line`
  the place the pipeline read the line from, `id` is null when the line has
  none, and `reason` is "empty line" or "sanitizer skip", spelled as the tool's
  drop reasons. docs/replica.md describes the row.
- The replica writes, beside every record file it makes, a manifest
  `<file>.manifest.jsonl` with one row per line:
  `{"line": n, "handle": "...", "origin": ["A-001"]}`. Manifests are test
  evidence. The tool never reads them.

`docs/replica.md` describes the folder one build writes. The generated sets of
10,000 and 100,000 records, their files and the plant list are described in
`docs/plant.md`.

### 9.2 Names

The names the completed spec sets (`fixture/spec/fixture_spec.toml`) and the table
uses (`fixture/table/README.md`):

| Thing | Name in the store |
|---|---|
| source files | `src/A`, `src/B`, `src/C`, `src/Z` |
| generation outputs before the release | `gen/P`, `gen/T`, `gen/S` |
| releases | `R1`, `R2`; the swapped-winner build `R1-swap`; the text-only import of T `import/T` |
| shards, partitioned layout | `E/shard-1`, `E/shard-2`, `E/shard-3` |
| shards, round-robin variant | `E-rr/shard-1` to `E-rr/shard-3` |
| the tokenizer input of a shard | the shard's name followed by `.input`, for example `E/shard-1.input` |
| mixtures | `E/mix-1` (shards 1, 2), `E/mix-2` (shards 1, 2, 3), `D/mix-1` (shards 1, 2), `Y/mix-1` (shards 1, 2, 3), `K/mix-1` (shard 3, then shard 2: on purpose not in name order); round-robin `E-rr/mix-1`, `E-rr/mix-2` |
| runs and segments | run `E`: `E/seg-1` (steps 1 to 6), `E/seg-2` (steps 7 to 12, a resume from `E@6`); run `E-rr`: `E-rr/seg-1`, `E-rr/seg-2` (a resume from `E-rr@6`); run `D`: `D/seg-1` (a warm start from `E@9`); run `Y`: `Y/seg-1` (a warm start from `E@6`); run `K`, the cyclic run: `K/seg-1` (a warm start from `E@6`, two launches) |
| checkpoints | `E@6`, `E@9`, `D@3`, `Y@6`, `K@3`, `K@8`, `K@40`; the round-robin variant registers one, `E-rr@6`, the parent of `E-rr/seg-2`, and no answer is tabulated for it (DECISIONS.md D-24) |
| Check 4's scratch shard | `c04/shard-500` |
| Check 3's scratch shard | `c03/shard-notokens`, its tokenizer input `c03/shard-notokens.input` |
| the sanitizer input of a shard (a run made without capture) | the shard's name followed by `.sanitizer-input` |
| the successors of a repair | `successor of <name>` for every artifact the report rewrites: `successor of R1`, `successor of E/shard-3`, `successor of E/mix-2`; the repaired build is registered with the name prefix `successor of `, so a file it leaves as it was is the same instance as the historical one, which gains a second name (DECISIONS.md D-14), and a rewritten one is a new instance |
| a generated set's shards, round-robin layout | `set-rr/<shard>`, with `.input` and `.sanitizer-input` as above; the set's files `R1`, `R2`, `src/<SOURCE>`, `gen/paraphrase`, `gen/translation`, `gen/summary` |

The jobs are named `paraphrase` (makes P), `translation` (makes T) and `summary`
(makes S). In a generated set a job of many rows runs once per row, named
`<job>/<row number>`, so no record id reaches the store through a job name
(plan Check 3: the tool is not given the chunk-file ids). The stores the table asks (`main`, `rr`, `swap`, `no-capture`,
`import`, `scratch-500`, `scratch-notokens`) are listed in `fixture/table/README.md`.

Record handles the table and the spec use:

- `X@A` and `X@B`: the two copies of the shared passage, `A-001` and `B-001`;
- `SA` and `SC`: the passages of A and of C that S is made from, `A-002` and
  `C-001`;
- `P`, `T`, `S`: the paraphrase, the translation and the summary;
- `X`: the row of a release that holds the passage X;
- `W`: the document read only through the shared last token of a sample, an
  unrelated record of Z in shard 3, fixed by the seed search (DECISIONS.md D-13
  as amended by D-24);
- a record with no named handle, such as an unrelated record of Z, has its record
  id as its handle (`Z-041`) in the spec and in the replica's manifests. The
  table's README writes `U-<nnn>` for the unrelated record `Z-<nnn>`.

The spec's handle list maps each named handle to its source row id.

### 9.3 The fixture spec

`fixture/spec/fixture_spec.toml`, one file, written by hand (plan 5.1): the
Megatron release tag and which package holds which code, the tokenizer and the
end-of-document convention, the normaliser, whether release rows carry a source
id, the sources and handles, the generation rows, the releases and the import,
the two layouts with their shuffle seeds, the skipped sanitizer line, the mixtures
in the trainer's order, every run with its segments, launches, step record and
checkpoints, and the seed search: the conditions a seed must meet, each with the
table rows that rest on it, and the read positions the search fixes with the
oracle log lines that show them. A `[sizes]` table holds the token counts of X,
P and T, the documents, tokens and samples of every shard and the dataset
length of every segment. Its `[pending]` list is empty: every `pending:<key>`
the table uses resolves to a value of the spec.

### 9.4 The hand-written table

`fixture/table/check<NN>.toml`, one file per check of plan 5.5. Every row:

```toml
[[row]]
id = "c05.unmutated.R1"             # stable id, quoted by the test that grades it
question = "Under withdraw A, what does release R1 answer?"
plan = "5.5 Check 5; 2.1; 5.1"
core = true                          # false for an extension row; `extension = "..."` then
given = { request = "withdraw A", artifact = "R1", state = "unmutated" }
expect = { answer = "contains", notes = ["the parent list is process evidence (what the harness handed the job and what the output states it used), not proof that the output depends on the source"] }
source = "hand-written table: R1 holds X with A's copy selected, and P, T and S (5.1); the chains of P, T and S pass generation edges (2.1)"
```

`expect` uses the field names of section 6; its `notes` are the full `NOTE_*`
sentences, and a hop of its `chain` names a record by its handle
(`gen/T:T`) where an answer names it by its locator. A test compares exactly the
fields a row's `expect` holds. A row whose
expected answer depends on an assumption names the fallback and what it expects
under it (`expect_under = { F1 = { ... } }`). `fixture/table/README.md` describes
the table in full.

### 9.5 The replica's repair word

`python -m replica repair` stands in for the team's builder (plan 4.4). It is the
one command that turns a report into a repaired side, and it is described for its
owner in `docs/replica.md`; this section fixes what the tool's documents need of it.

```
python -m replica repair --report FILE --historical DIR --out DIR --store FILE
        (--spec FILE | --set DIR) [--layout partitioned|round-robin|both]
        [--reference-set DIR --reference-out DIR] [--sides FILE]
        [--unseeded] [--shuffle-seed LAYOUT=N] [--jobs N] [--tokenizer-timeout S]
        [--generation-harness in-process|confined]
```

- `--report FILE` is a report of section 7.1, read with the `json` module as a file
  (section 1). The replica reads `form`, `artifacts` and `records` and nothing else of
  it (so it reads nothing of `upstream_files`: the same report with that list as
  printed, with other content under it or without the key gives the same repair). A
  file whose `form` is another, or that cannot be read, is refused.
- `--store FILE` is the historical store the successors are registered in. A path that
  holds no store is refused (exit 1, one line on standard error) before anything is
  built or created: opening a path would create an empty store. A path that holds a
  file that is not a store (arbitrary bytes, or a database that has no schema version
  of the tool's store) is refused the same way, in the replica's own input error and
  not in the tool's "not a store"; the file is read without a connection that can
  write, so its bytes are not changed and no file or folder is made beside it.
- `--historical DIR` is the folder of the build the store's historical artifacts were
  made from. The replica turns each `artifact:locator` of the report into the id it
  gave that record, through the manifest beside the file that artifact name stands
  for (the replica's own folder layout and manifests, `docs/replica.md` section 2;
  the tool never reads them). A record that no manifest line names is refused.
- From the report it builds the repaired side into `--out` with capture on into
  `--store`, and `--name-prefix` set to the text `successor of ` (`SUCCESSOR_PREFIX`).
  It leaves out (`--leave-out`) the record of every entry whose action is retire, and
  swaps (`--swap OLD=NEW`) the record of `withdrawn_parent` for that of `new_parent`
  for every re-source; a promote entry needs no parameter of its own (the swap puts
  the equivalent copy in the place of the withdrawn one). An entry whose action is keep
  or unresolved changes nothing, so a report edited by hand to keep a retired record
  builds a side that holds it, and a report edited to retire a kept record builds a
  side that lacks it.
- `--reference-set DIR --reference-out DIR` builds the reference side by the same
  steps from the folder that the set's generator wrote without the withdrawn
  content. Without them no reference side is built and the sides file names the
  repaired side only.
- `--sides FILE` writes the sides file of section 7.2, naming for every artifact of
  the report the file the repaired side and the reference side hold for it, and then each
  shard the report names under a layout's landing that the build holds.
- It prints one JSON summary on standard output: `left_out`, `swapped`, `successors`
  (each historical name beside the name it is registered under), the summaries of the
  builds, and the path of the sides file. Exit 0 built, 1 refused with one line on
  standard error, 2 a wrong command line.
- A second build into the same store gives no new instance for a file that is
  unchanged (DECISIONS.md D-14): a kept artifact is the same instance under a
  second name.
