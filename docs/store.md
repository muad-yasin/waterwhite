# The evidence store

The store is one SQLite file. It holds every record an answer rests on: sources and
their members, artifacts, occurrences and fingerprints, edges with their harness
logs, drops, tokenization maps, mixtures, run segments, step traces and
checkpoints. The package `waterwhite.store` is the only code that writes it. It
stores locators, fingerprints and links, and never the text of a record.

Coverage is not stored. A query computes it from these tables each time, so there
is no second ledger that could disagree with the edges after a crash.

Code: `src/waterwhite/store/` (`store.py` the class and its write methods,
`_reads.py` the read methods, `_integrity.py` the loud checks, `views.py` the
result types, `artifact_files.py` the readers of a JSONL file and of a shard (its
index, the token type of its `.bin` file and one document's tokens),
`_migrate.py` the schema steps, `schema.sql` the schema). Ids and the
definition of a line come from `waterwhite.ids`, words from
`waterwhite.vocabulary`, errors from `waterwhite.errors`. Tests:
`tests/store/` (`test_store_maps_and_paths.py` for an artifact's place on disk, a map with
its drops, a map beside one that did not verify, the role of a found file and
the harness a log names) and, together with the capture wrappers,
`tests/integration/test_store_capture.py`.

Plan sections served: 3.4 (the store), 3.1 and Decision 29 (records and identity),
3.2 (labels), 2.1 and Decisions 21 and 26 (own record, or only named), 2.2 (what
the proof walks, the loud faults), 2.4 (a checkpoint's history), 3.5 and
Decision 19 (step traces), Decisions 1, 2 and 3. `DECISIONS.md` entries D-14,
D-15, D-18, D-20, D-21 and D-24 are choices this module carries out; each is named
where it applies.

## Opening a store

```python
from waterwhite.store import Store

store = Store.create("work/evidence.db", normaliser="control-chars/1",
                     verified_joins=[("core_v0.20.0", "single", "<sampler class>")])
store = Store.open("work/evidence.db")                 # read and write
store = Store.open("work/evidence.db", readonly=True)  # read only
store.close()                                          # or: with Store.open(...) as store:
```

`Store` is the one name the package exports.

- `create` makes the file, writes `meta` (`schema_version`, `normaliser`) and the
  `verified_join` rows it is given, all in one transaction. `verified_joins` are
  the (release tag, sampler type, sampler class) combinations the exposure join
  was verified for (plan F6); they are written here and at no other time. It
  refuses a path that already holds a file, whatever the file is, with
  `StoreFileExists` ("a file already exists at the store path"), and a normaliser
  name that is not `<name>/<version>` with `InvalidRecord`.
- `open` refuses a path that holds no store with `NotAStore` ("not a store"): no
  file, a file that is not an SQLite file, or one with no schema version of this
  tool. A store written by a newer schema version is refused with
  `NewerSchemaVersion` ("store of a newer schema version").
- Every connection runs with `journal_mode=WAL`, `synchronous=FULL` and
  `foreign_keys=ON`.
- The schema is versioned in `meta.schema_version` and migrated one step at a time
  (`_migrate.STEPS`: version n is reached from n - 1 by step n; each step and its
  version row are one transaction). The schema is at version 4. Step 1 is
  `schema.sql`; step 2 is `schema_2.sql`: a harness log gains the column `harness`
  (empty for every log written before, so after migration such a log establishes
  no bound: DECISIONS.md D-27), and the index over a shard's maps is
  narrowed to the verified ones, so a map that did not verify no longer holds its
  namespace and derivation (D-28). Step 3 is `schema_3.sql`: the hand-off of a
  harness log gains its position (below, "The write methods"), and one array of a
  segment's blend gets a unique index over the segment and the kind where the
  shard position is null, which the table's own `UNIQUE` cannot give because
  SQLite reads two nulls as different. Step 4 is `schema_4.sql`: a harness log gains the
  column `hand_off_checked` (0 for every log already stored: nothing says its texts were ever
  compared, so after migration such a log establishes no bound until the texts are shown,
  "The write methods"). `create` runs every step, so a new store
  and a migrated one hold the same schema. `open` runs the missing steps; a
  read-only open of a store that needs a step is refused (`Refused`). A writable open
  migrates, and a command that does so says so in its output (`"migrated"`, with a sentence
  that clean answers past a generation step now read "may contain" until the texts are shown).
- A migration step can refuse a store. Step 3 does when a segment holds more than
  one row for one array of its blend, whether the digests differ or not: no write of
  this tool makes such a row, and the unique index cannot be made over it. The
  refusal is `CacheDigestConflict` ("cache digest conflict", exit 3), naming the
  segment and the array. The step changes nothing: its transaction is rolled back,
  every row is kept, and the tool changes no row to make the index fit. Each step is
  its own transaction, so a store that came from version 1 keeps the version 2 it
  reached, and any other keeps the version it had. A writable `open` and `migrate`
  meet the refusal; a read-only open of the store is refused as for any store that
  needs a step (`Refused`, "migrate first"), and `migrate` then gives the reason. A
  store that holds no such row migrates, and a log written before step 3 is read as
  described under "The write methods".
- A path in the store is relative to the folder that holds the store file when the
  artifact lies under that folder, and absolute otherwise. So a store and the files
  beside it can be moved together. `store.disk_path(instance_id)` gives where an
  artifact's file lies now by this rule (for a shard: the prefix), or `None` when
  the store has no row or no path for it. It is the one place the rule is read
  from outside the store: the tokenization maps and the recovery call it and
  compute no path themselves.
- Copying a store. The store file and its write-ahead log `<store>-wal` are one
  state, and a copy of the two files is a consistent copy only when nothing writes
  to the store or checkpoints it from the first byte copied to the last. Closing one
  `Store` handle does not give that: `close` tries to fold the log into the file,
  and it does so only when no other connection is reading. Another connection can
  still write or checkpoint between the copy of the file and the copy of its log,
  and a pair copied at two moments can lose records or not open. Two ways are safe:
  - **Quiescence.** Stop every process that has the store open (a pipeline that
    records into it, a query, a recovery), check that no other connection holds it,
    and copy the file together with `<store>-wal` if that file is there. A killed
    process leaves such a file and it belongs to the store.
  - **A snapshot taken by SQLite**, which is safe while other connections work. The
    backup interface of the `sqlite3` module:

    ```python
    import sqlite3

    source = sqlite3.connect("work/evidence.db")
    copy = sqlite3.connect("work/evidence-copy.db")
    source.backup(copy)
    copy.close()
    source.close()
    ```

    or the statement `VACUUM INTO 'work/evidence-copy.db'` run on the store file (the
    target must not exist). Either gives one file that holds a single consistent state
    and needs no `-wal` beside it.

  Either way, a copy holds the store's records and not the artifact files, which are
  separate files and are copied separately. A copy that is opened for use reads a
  relative path from the folder of its own file, so put it in the same folder as the
  original (as above) or move the artifacts with it.
- `store.normaliser` (a property) is the name of the normaliser the store's
  fingerprints were taken after. `store.meta()`, `store.schema_version`,
  `store.path` and `store.verified_joins()` read the rest of what `create` wrote.
  `Store.create` writes the joins it is given and no others: the list the tool
  ships is written by the command `init` (docs/cli.md, "init"), not by the library.
  `init` refuses a join outside the shipped list; the library does not, so a store
  made through it with the caller's own list is outside what the tool verified.
- `store.state_mark()` returns a mark of the store as it is: a tuple that is the
  same on two calls with no write between them and differs after any write, made
  through this store or committed by another connection to the file. A reader that
  keeps what it learned (a coverage session, an integrity pass, a proof) keeps the
  mark with it and is out of date when the mark has changed. Reads change nothing.
  A closed store has no mark and the call is refused with `WriteRefused`. A change
  to an artifact's file on disk is not a write to the store and does not move it.
  So a pass or a proof that is kept by its mark is kept for the database, not for the
  artifact files: it is keyed to the mark and says nothing of whether a file still has
  the bytes that were checked. The files must stay unchanged between the check and the
  use of what it found, or the check is run again; `check_integrity` run afresh reads
  each file and finds a changed one. The same holds for the capture reads
  (docs/capture.md, "How a file is read"): two matching reads say the file agreed at
  those two moments.
- What the store protects against. Its write methods never overwrite recorded
  history, and its loud checks (below) compare what it records with other records and
  with the files it names. That guards against accidental edits and inconsistent
  imports. Waterwhite does not defend against direct edits of its database: a store
  whose rows were edited outside the tool is read as stored. The loud checks find
  some inconsistent edits, not all, and a check that finds nothing is not proof that
  the file was not edited.
- Of the six extensions only the generating-model rule is built beside its off state. The generating-model switch, written as "on" in `setting`, turns its rule on (docs/coverage.md, section 6); the switches of the other five turn nothing on: a request that needs one of them is refused (docs/coverage.md, sections 6 and 11; docs/cli.md, "query
  ban, query flagged"). With every switch off the store behaves as the core.
- `store.settings()` returns the `setting` rows as a dict; `store.set_setting(key,
  value)` writes one. A missing key means off (`waterwhite.extensions`). A key
  under the prefix `extension.` is the switch of an extension: it must name one
  of the six extensions and hold "on" or "off", else the write is refused with
  `InvalidRecord`. Any other key is the team's own. A stored switch that holds
  another value (edited from outside) is not read as off: `extensions.check_switches`
  refuses with `Refused` before any request, proof, listing, report or `verify` is made, and
  `extensions.is_on` refuses the same way. `set_setting` does not pass through that check. A key under the prefix that names no
  extension of this build is refused the same way, and `remove_setting(key)` clears any setting.
- `Store.check_integrity` and `Store.missing_files` take `models=True` to add the ancestry of the generating models to the closure; `Store.resolve_model(text)` reads a log's model text (the list form, "none", a registered checkpoint by exact name) and is the one reading the walk and the closure share.
- The schema of version 1 was revised in place once, when the edge gained its
  `job_name` column (D-24), before any store left the test folders. A version-1
  file made by an earlier build lacks that column. `open` refuses it with
  `NotAStore` and says why; no migration step exists for it. Make such a store
  again. Every later change raises the version and adds a step.

## Writing

### The rules every write method keeps

1. One method call is exactly one transaction (`BEGIN IMMEDIATE`). It is committed
   whole or rolled back whole. A process killed in the middle leaves nothing of
   the call behind.
2. Checks come first, rows afterwards. A check that can refuse runs before the
   first row of the call is written.
3. A refusal is a named error of `waterwhite.errors` in the write family (exit
   code 4): `DeclaredCountMismatch`, `DeclaredDigestMismatch`,
   `LabelStrongerThanDerivation`, `InvalidRecord`. A record that the schema itself
   rejects is reported as `InvalidRecord`. A write on a read-only store is
   `WriteRefused`.
4. Writing the same thing again changes nothing. Writing something else under a
   key that is taken is refused. The write methods never overwrite a recorded row:
   recorded history is not changed through them. That is a rule of the write
   methods, not a protection against an edit made to the file from outside; the loud
   checks (below) find some of those.
5. The store reads the artifact on disk itself. It does not trust a declared count
   or digest.

### Identity

`register_jsonl` works in a fixed order: it reads the file (line count and
SHA-256), takes the edge digest over the edges whose derivation is "stated"
(`ids.edge_record`, `ids.edge_digest`), computes the instance id, and only then
keys and stores the occurrences and edges.

- An ingested file or an import (no producing step): `ids.ingest_instance_id(content
  digest, member name)`.
- A file a step produced: `ids.instance_id(content digest, step name, step version,
  parameter digest, input instance ids in order, edge digest)`.
- The same bytes with another history are another instance: other inputs, or
  another parent kept by a dedup step. The two share their content digest and have
  different occurrence keys.
- A name names one instance for good. Registering another history under a taken
  name is refused.
- In a reference handed to the store, the instance id `""` (`records.OWN_ARTIFACT`)
  names the artifact that is being written, because that artifact's id does not
  exist yet. In the edge digest such a parent is named by its locator alone.
- A record that is in scope only (handed to a job, not stated as used) is not a
  parent and is not part of the edge digest.
- A link derived afterwards (a recovered edge, a tokenization map, a source link)
  never changes an instance id.
- A mixture's id is `ids.mixture_instance_id`: its shards' instance ids and
  content digests, in order, and its settings; not the paths. A checkpoint's id is
  `ids.checkpoint_instance_id(segment, step)`, so a segment can name its parent
  checkpoint before that checkpoint is registered.

### A second registration of the same history (D-14)

Registering a history whose instance id the store already holds returns the
stored instance and writes nothing. Under a new name it adds that name and nothing
else.

The id covers the content, the step, the inputs, the parents of the stated edges
and the selected parent. A second registration can agree on all of these and
still differ elsewhere. Such a difference is refused with `InvalidRecord`, the
stored record stays as it was, and nothing is dropped silently. What is compared:

- the kind and the role;
- the fingerprint of every line;
- for every edge handed: its transform, parent mode, derivation, verification,
  label and generation mark; the name of its job (D-24); its inputs in order, with
  the records that are in scope only; and its harness log whole (job name, the
  log's digest, the model entry, the hand-off in order, the reads);
- an edge handed for a child the stored instance holds no edge for (an edge found
  after registration goes through `add_recovered_edges`);
- a drop handed that the stored instance does not hold (a drop found after
  registration goes through `add_drops`).

What is not compared: the path. The same history registered from another place is
the same instance, and the stored path stays the first one. This holds for a
shard and for the paths of a mixture's snapshot too. A drop the stored instance
holds and the second registration leaves out is not a difference either: the
store cannot tell it from a drop added later by `add_drops`.

### The write methods

| Method | What it writes | Refuses |
|---|---|---|
| `add_source(name)` | a source name; idempotent | |
| `add_member(source, name, revision, part_kind, parts)` | a member and its parts `(instance id, locator or "")`. An identical call again changes nothing. An existing member with the same source, revision and part kind gains the parts it lacks | an unregistered source, artifact or record; another source, revision or part kind under the same member name; a part kind outside `vocabulary.PART_KINDS` |
| `register_jsonl(path, name=, role=, step=, member_name=, inputs=, occurrences=, edges=, drops=, declared_count=, declared_digest=)` | the artifact, its name and inputs, its binding, one occurrence per line, its edges with their harness logs, its drops; returns the instance id | see below, including a handed text that is not the record its reference names |
| `register_shard(prefix, name=, step=, inputs=)` | the artifact row of a shard (`<prefix>.bin`, `<prefix>.idx`) and nothing else; returns the instance id | a missing file; an index that is not a version-1 Megatron index (`UnsupportedFormat`) |
| `write_map(shard_instance_id, map, declared_count=, declared_digest=, drops=())` | the binding of one namespace and derivation, every row of the map, and the drops of the same step, in one transaction | see below |
| `write_maps(shard_instance_id, maps, declared_count=, declared_digest=, drops=())` | the same for every map one derivation run gave for the shard, in one transaction: all of them and the drops, or nothing | as `write_map`; two maps of one namespace and derivation in one call; no map |
| `register_mixture(name=, shards=, settings=)` | a mixture, its snapshot rows and its settings whole, as canonical JSON; shards are named by value and need not be registered; returns the instance id | positions that are not 0, 1, 2 and so on; a registered shard whose digest differs from the snapshot's; settings that are not a JSON object of plain JSON values |
| `register_run_segment(run=, segment=, position=, mixture_instance_id=, parent_checkpoint=, start_kind=, run_record=, steps=, trace_route=, trace_cause=, trace_cause_detail=, traces=(), checkpoints=())` | in one transaction: the run (if new), the segment, the run record whole as canonical JSON with its digest, the columns read from it, launches, cache and shuffle digests, the completed steps, the step traces (checked as `write_step_traces` checks them) and the checkpoints `(name, step)`. The same segment again writes only a trace or a checkpoint it lacks | a fresh start with a parent; a resume or warm start without one; a trace route "none" without one of the five trace causes (docs/interfaces.md section 2: `TRACE_CAUSES`), or a cause beside another route; a trace route other than "none" for a run record that holds no digest of an index array the segment has (three per shard and, from two shards on, two for the blend; a replay counts only against digests recorded at training time, plan 3.5, and kept cache files only against the same digests, DECISIONS.md D-41), and none at all for a call with no run record; a trace route other than "none" for a run record that states the cyclic sampler and holds no shuffle digest (D-26); a trace route other than "none" for a run record that holds no launch (empty, null or absent); a trace for a segment whose route is "none" (here and in `write_step_traces`); a trace cause detail beside any cause but "data-order trace missing", or beside it a word other than its six reasons (docs/interfaces.md section 2); a run record of another format, run, segment or start kind; a field of the wrong type; a parent whose history holds the segment; a taken position; a trace that does not fit its step's batch size; a checkpoint at a step the segment does not have; any of these leaves nothing of the call written |
| `write_step_traces(segment, traces)` | one trace per step with its samples and rows | an unregistered segment or step; a trace whose count differs from its step's batch size or from its number of samples; a coverage mark outside the sample; another trace for a step that has one |
| `register_checkpoint(name=, segment=, step=)` | a checkpoint; the segment is named by value; returns the instance id | a step that is not among a registered segment's steps; a checkpoint that would lie in its own history |
| `add_recovered_edges(instance_id, edges)` | edges derived after registration | a "stated" edge; a child that already has another edge (it is not replaced); a handed text that is not the record its reference names |
| `add_source_link(instance_id, locator, member, label, evidence, derivation, verification=None)` | a link from one record, or with `locator=None` from every record of the file, to a member, with how it was derived; the same link again changes nothing | see below |
| `add_drops(instance_id, drops)` | drops recorded for the step that wrote a registered artifact, known after registration (a tokenizer's lines with no tokens, a sanitizer log read by a recovery) | an unregistered artifact; a record the store does not hold; the same record, step name and output with another reason or group key |
| `set_setting(key, value)` | one setting row; a key that is taken gets the new value | under the prefix `extension.`: a name that is no extension, a value other than "on" or "off" |

The start kinds, the trace routes and the part kinds are constants of
`waterwhite.vocabulary` (`START_*`, `ROUTE_*`, `PART_*`).

`register_jsonl` refuses the following. Checks of the handed records come first,
then the file on disk, then what needs the store:

- a role or step that do not fit: a file with no step is registered under a
  member name, with no inputs, stated edges or drops, and carries the role it
  has ("source" for an ingest, "import" for a text-only import, "intermediate"
  or "release" for a pipeline file found afterwards); a file a step produced is
  an "intermediate" or a "release" and has no member name;
- an edge that breaks the record model (`InvalidRecord`): an unknown word, two
  edges for one child, a parent named twice, an "any" edge that is stated and does
  not mark the one parent its step kept, an "all" edge that marks one, an in-scope
  record on an edge that is not a generation edge, a harness log on such an edge,
  a derivation that belongs to a map;
- an edge that does not keep the name of its job as D-24 asks (`InvalidRecord`): a
  generation edge with no `job_name`, an edge that is not a generation edge with
  one, or a `job_name` that differs from the job name of the edge's harness log;
- a label not in `vocabulary.labels_allowed(derivation, verification)`
  (`LabelStrongerThanDerivation`);
- a generation edge with a harness log whose inputs do not all lie in the log's
  hand-off (`InvalidRecord`);
- a generation edge with a harness log that leaves out a record the log says was
  handed: every handed record is an input of the edge, a parent or in scope only
  (`InvalidRecord`; plan 3.3, Decision 25);
- an edge that carries handed texts (`EdgeRecord.handed_texts`, or
  `HarnessLog.handed_texts` on its log) and breaks one of these rules
  (`InvalidRecord`): the texts are a tuple; each is a (reference, text) pair whose
  reference is an occurrence reference with a text instance id and a valid line
  locator, and whose text is a string; the edge is a generation edge; when the edge
  carries a harness log, the reference lies in that log's hand-off;
- a generation edge whose harness log hands exactly one record and which does not
  name that record as a parent: it marks it in scope only, or leaves it out
  (`InvalidRecord`, D-21). A job's sole handed record is a parent of each output,
  whatever the output states;
- a stated edge with a parent that lies in no artifact of `inputs` and not in the
  artifact itself (`InvalidRecord`, D-15). The inputs are part of the instance id,
  so a parent outside them would be history the id does not cover. A record in
  scope only is not a parent and is not held to this. A recovered edge is exempt;
- a missing file (`InvalidRecord`); a declared line count that differs from the
  file's, or a number of occurrences that differs from it
  (`DeclaredCountMismatch`); a declared digest that differs from the file's
  (`DeclaredDigestMismatch`);
- occurrences that are not exactly one per line (`InvalidRecord`);
- a parent, a handed record or a dropped record that the store does not hold; a
  harness log that differs from the stored log of the same job name; a name that
  names another instance (`InvalidRecord`);
- a generation edge that carries the texts its job was handed, on the edge
  (`EdgeRecord.handed_texts`, which `capture.generation_edge` copies from the log
  `run_job` or `run_confined_job` returned) or on its log (`HarnessLog.handed_texts`),
  when one of them is not the record its reference names (`InvalidRecord`, naming the
  job and the reference). A record with no fingerprint holds no text: a text that is
  empty after the normaliser is that record, any other is refused. The
  comparison is the fingerprint of the text after the store's normaliser against the
  fingerprint stored for that record. It is made for each handed text, whichever handed references it covers (a hand-built edge or log may cover only some; the others are not compared), before any row
  of the write goes in, so a refusal leaves the store as it was. The same check is
  made when a history that is registered already is handed again, and by
  `add_recovered_edges` for an edge built by hand that carries texts (a stated edge,
  which is what `generation_edge` builds, is refused there). An edge and a log that
  carry no texts, which is the edge of a log read from its file, of a log built by
  hand without its texts, of a log fetched from the store (`store.harness_log`) and
  of a job that wrote its own parent list, are registered without the comparison, and
  their log is stored as not compared (below). The texts are not stored, and the stored
  log of a job carries none (docs/capture.md, section 3, lists which registrations
  compare);
- a second registration that disagrees with the stored instance (`InvalidRecord`,
  D-14, above).

A harness log is stored once per job name, with its handed list and its reads in
order. The hand-off is read back in the order it was handed: each handed record
has its position in the log (a record handed twice is stored once, at its first
position), and `harness_log(job)` and `edge_of(ref).harness_log` give the
hand-off by position, also after the store was closed and opened again. A second
registration of a stored job compares the hand-off in order, so the same records in
another order are a different log (`InvalidRecord`). A log written before schema
version 3 kept the hand-off as a set: it has no position, is read back in the order of
its occurrence ids, and a second registration of its job compares its hand-off as a
set. `edge.bound_established` is written from `HarnessLog.bound_stated()`: what the log
itself says (the confined harness wrote it and it lists no read outside its hand-off).
A log also carries the store's own mark `hand_off_checked`: 1 when the registration that
stored it carried a text for every reference of the hand-off (on the edges of the job or on
its log) and each matched its record, 0 otherwise (a log read from its file, a log with
only some of its texts, a log with an empty hand-off, every log of a store migrated from
version 3). A value a caller puts in `HarnessLog.hand_off_checked` is not read, and
registering a job again never changes the stored mark. `HarnessLog.bound_established()` is
`bound_stated()` and the mark: a stored log whose texts were never shown and compared
establishes no bound, and a negative past its edge is "may contain", naming "upper bound not
established" and the job. `store.verify_hand_off(job, texts)` (the command `compare-hand-off`,
docs/cli.md) takes the text of every reference of the stored log's hand-off and of no other,
compares each as at registration, and marks the log: a text that is not its record, a missing
or an extra reference, a job with no stored log and an empty hand-off are refused
(`InvalidRecord`, naming the job and the reference, never a text) and change nothing; a log
that is marked stays marked; the in-process harness's log establishes no bound either way.
A read of a record the store does not hold is kept as text. The log's `harness`
("in-process", "confined", or `None` for a log that does not say) is kept with
it; another word is refused (`InvalidRecord`).

An edge keeps the name of its job (D-24). `EdgeRecord.job_name` is written to the
edge's own row, beside the pointer to the harness log. The schema holds the rule
too: a generation edge has a job name and no other edge has one. So the name
outlives the log: when a log is removed, every edge of its job still says which
job made it, and an answer that fails for the missing log names that job (plan
Check 5). `capture.generation_edge` sets the field from the log. A pipeline whose
job wrote its own parent list (a generation edge with no harness log) passes the
job's name.

A drop is one drop per dropped record, step name and output artifact. The same
step name can drop one record once for each artifact it writes, for example two
builds of a release from the same inputs.

`add_source_link` refuses:

- a derivation other than "pointer", "file" or "none" (`InvalidRecord`): a source
  link has no parent record behind it, so it is never stated by a step and never
  derived by a rule that names a parent;
- a label its derivation and verification do not allow
  (`LabelStrongerThanDerivation`). A "file-level" link has the derivation "file".
  A "mapped" link needs the derivation "pointer" with verification "passed". A
  pointer that is not verified, and the derivation "none", can only be "unknown";
- an unknown verification result, an unregistered artifact, record or member, an
  empty evidence text (`InvalidRecord`).

`write_map` refuses:

- an unregistered instance, or one that is not a shard; an unknown namespace; a
  derivation other than "position" or "token-matching" (`InvalidRecord`);
- a map label, or a row label, not in `labels_allowed(derivation, verification)`
  (`LabelStrongerThanDerivation`): a map whose verification did not pass can only
  be "unknown";
- shard files that are not at their recorded place (`InvalidRecord`);
- a declared document count that differs from the shard's index, or a number of
  rows that differs from it (`DeclaredCountMismatch`); a declared digest that
  differs from the shard on disk (`DeclaredDigestMismatch`); files that match the
  declaration and no longer match the registered instance
  (`ArtifactDigestMismatch`);
- rows that are not exactly one per ordinal; a "recovered" row without one parent;
  a "candidates" row without two or more candidates; an "unknown" row that names a
  record; a parent or candidate the store does not hold (`InvalidRecord`);
- another map for a namespace and derivation that already hold a verified one
  (`InvalidRecord`). The same map again changes nothing;
- a drop handed with the map that the store refuses (an unknown reason, a record
  the store does not hold, the same record, step name and output with another
  reason). The map is then not written either.

A namespace and derivation of a shard hold at most one map whose verification
passed (D-20). A recovery map goes to the namespace "recovery" and never touches
a capture map, and the position-mode and the token-matching recovery map are
stored side by side, so both can be read and compared.

**A map that did not verify does not take its place for good.** A map whose
verification is "failed" or "not run" carries the label "unknown" and names no
record. A later map of the same namespace and derivation is written beside it,
whether it verified or not; nothing is overwritten and nothing is deleted. A
verified map is final for its namespace and derivation: any other map there is
refused. So a recovery run with the wrong log or the wrong tokenizer leaves its
failed map as a record of that run, and the corrected run still counts. What
this does to the conditions of plan 2.2:

- (d): every stored map, the one that did not verify included, is a binding with
  its own declared count and digest, and `check_integrity` holds each of them
  against the shard;
- (f): the label rule holds for each map at write time. The proof reads it on
  the map a reader gets, which is the verified one where there is one;
- plan 3.3's "a wrapper-written map is never overwritten" holds as before: a
  recovery writes to its own namespace, and inside a namespace nothing replaces
  a map.

The schema carries the rule as a unique index over the verified maps of a
namespace and derivation, so a second verified map cannot be put there by any
route.

**A map and its drops.** `write_map` writes the drops handed with the map in the
same transaction (plan 3.4). A refused or interrupted `write_map` leaves no
binding, no rows and no drop; a shard with no map has no record of its own. A map
that is held already is left alone, and a drop handed with it that the store
does not hold yet is still written.

### What the store reads from disk

- A JSONL file: its SHA-256 (the value of `ids.file_digest`) and its line count,
  taken in one read. What a line is, is defined once, in `ids.split_lines`, and
  the capture wrappers count with the same function: a line ends at the byte 0x0A
  and at nothing else; the bytes after the last line feed are a line when there is
  at least one of them; an empty file has no lines; an empty line is a line and
  gets an occurrence with no fingerprint.
- A shard: `ids.shard_digest` of the two files' digests, and the document count
  from the `.idx` file. The index is read from its documented layout
  (`artifact_files.read_shard_index`); the tool imports and ships no Megatron code.
  The document count is the number of documents that hold tokens. A line that gave
  no tokens leaves a boundary with no sequence in the index and adds no ordinal.
  A shard in which a document holds more than one sequence is refused as an
  unsupported format.
- A shard's tokens: `artifact_files` is the one reader of the shard format in the
  tool. `token_dtype(code)` names the token type of the `.bin` file for the code
  its index gives (the two floating-point codes hold no token ids and are refused
  as an unsupported format). `ShardIndex.token_starts()` gives where each
  document starts in the token stream. `ShardTokens(prefix)` reads one document
  at a time from its own byte range (`read(ordinal)`, an int64 array) and never
  holds more than one document's tokens. The store itself reads no token; the
  tokenization maps do, through this reader.

The write methods read files. The loud checks read each file once, and that one
read gives its digest and its record count (below). For a shard the index file is
digested and read as an index in the same read
(`artifact_files.read_shard_once`).

### A hook for tests that stop a writer

`store.set_row_hook(fn)` installs `fn(table, written)`. It is called inside a
write before each row of `table` is handed to SQLite, with the number of rows of
that table the call has stored so far, and once after the last. A test that needs
a writer killed after 100 map rows installs a hook that kills its own process at
`("map_row", 100)`. The tables it is called for are `occurrence`, `edge`,
`map_row`, `map_candidate` and `drop_record`. The hook must not write to the
store.

## Reading

Every read method returns typed results: the types of `waterwhite.store.views`, or
a record type of `waterwhite.records` where one fits. A method marked *lazy*
returns an iterator and reads as the caller goes. Do not write to the store while
such an iterator is open.

### Artifacts

| Method | Returns |
|---|---|
| `artifact(ref)` | the `Artifact` with this name, or else with this instance id; `None` when there is no row |
| `artifacts(kind=None)` | every `Artifact` in registration order, one kind when given |
| `artifacts_by_digest(digest)`, `artifacts_by_path(path)` | every instance that matches (two instances can share bytes or a path) |
| `artifact_by_digest(digest)`, `artifact_by_path(path)` | the one instance that matches; `None` for none; `AmbiguousArtifact` for several |
| `resolve(ref)` | what a query names: a name, an instance id, a content digest or a path, tried in that order; `AmbiguousArtifact` when a digest or path matches several |
| `bindings(instance_id)` | every `Binding` of the artifact, in the reader's order (below): the first is the one a reader gets; empty when the artifact has no record of its own |
| `binding(instance_id, namespace=None, *, derivation=None, verification=None)` | one `Binding`; with nothing named, the reader's binding |
| `stored_row_count(instance_id, namespace=None, *, derivation=None, verification=None)` | how many rows the store holds under that binding |
| `disk_path(instance_id)` | where the artifact's file lies now (a shard's prefix), or `None` |
| `file_state(instance_id)` | "present", "absent", "partial" (a shard with one file gone) or "no file" |
| `missing_files(instance_ids=None, *, models=False)` | the `MissingFile` entries of a closure, or of the whole store; with `models=True` the closure also holds the ancestry of the generating models (the generating-model switch on) |

An `Artifact` carries the instance id, kind, role, content digest, stored path,
producing step, member name, edge digest, names and input instance ids.

### Named, with no record of its own

A record names some things by value, so that the thing can be missing without the
naming record being lost. `Reference(relation, referrer, referred, position,
registered)` describes one such naming. The relations: "mixture shard" (a snapshot
names a shard instance), "segment mixture", "checkpoint segment" (the referred
thing is a segment name), "segment parent checkpoint", "artifact input".

| Method | Returns |
|---|---|
| `named_without_record()` | every `Reference` whose target has no row: what a listing shows as "not covered", "record missing" |
| `references_to(referred)` | every `Reference` to this instance id or segment name, registered or not |

For an artifact with no row: at least one reference means "record missing", none
means "not registered". A shard that has a row and no binding also has no record
of its own: `bindings()` is empty.

### Occurrences and edges

| Method | Returns |
|---|---|
| `occurrence(ref)` | the `Occurrence` (reference and fingerprint), or `None` |
| `occurrences(instance_id)` | *lazy*: the records of a JSONL artifact in line order |
| `occurrence_count(instance_id)` | the number of occurrence rows |
| `occurrences_by_fingerprint(fingerprint)` | *lazy*: every record with this fingerprint, in any artifact |
| `edge_of(ref)` | the `StoredEdge` whose child is this occurrence, or `None` (a missing link, not a fault) |
| `edges(instance_id)` | *lazy*: every edge whose child lies in the artifact |
| `edges_naming(ref)` | *lazy*: the walk downward: every edge that names this occurrence as an input |
| `documents_naming(ref)` | *lazy*: every shard document whose map row names this occurrence as parent or candidate (`MapLink`), from the map a reader gets of each shard |
| `harness_log(job_name)` | the stored `HarnessLog` of a job, or `None` |

A `StoredEdge` carries the child, the transform, the parent mode, derivation,
verification and label as stored, `inputs` (each a `records.EdgeInput` with the
full parent reference, `used` and `selected`), `is_generation`, `harness_log`,
`bound_field` and `job_name`. Its properties `parents`, `in_scope_only` and
`selected` split the inputs. `bound_field` is the stored field only. A proof
derives the bound from `harness_log.bound_established()`; `harness_log` is `None`
when the log is gone. `job_name` is read from the edge's own row: it is there for
every generation edge, with or without its log, and `None` for any other edge.
The label is returned as stored, also when it was edited to one its derivation
does not allow.

A `MapLink` carries the shard's instance id, the namespace and the derivation of
the map the row belongs to, the ordinal, the row's label, and `via` ("parent" or
"candidate").

### Tokenization maps and the reader's order

A shard can hold several maps: per namespace ("capture", written by the
wrapper as the tokenizer ran; "recovery", written afterwards) and derivation
("position", "token-matching") one verified map, and beside it the maps of that
namespace and derivation that did not verify. **A reader gets the first map in
this order (D-20 as amended by D-24):**

1. a map whose verification passed before one whose verification did not, across
   both namespaces;
2. the capture namespace before the recovery namespace;
3. the position mode before token matching.

So among verified maps a reader gets the capture map; else the position-mode
recovery map; else the token-matching recovery map. A shard whose capture map
failed its verification and that has a verified recovery map is read through the
recovery map: under the other order it would answer from "unknown" links although
a verified map is stored. Only a shard with no verified map gives a reader a map
that is not verified, by rules 2 and 3. Such a map can only carry the label
"unknown", and it is still the shard's own record: a shard whose only map failed
its verification answers from "unknown" links and is not a shard with no map.

`binding`, `map_rows`, `map_row`, `stored_row_count` and `documents_naming` follow
this order when neither a namespace nor a derivation is named. Name a namespace, a
derivation or both to read another map; the same order picks among what is left,
so a namespace and derivation that hold a verified map beside one that did not
verify give the verified one. Name the verification result too
(`verification="failed"`) to read the other. Among maps that did not verify the
first that was written comes first.

| Method | Returns |
|---|---|
| `map_rows(shard, namespace=None, *, first_ordinal=0, last_ordinal=None, derivation=None, verification=None)` | *lazy*: `records.MapRow` per document in ordinal order, each with its parent or its candidate set |
| `map_row(shard, ordinal, namespace=None, *, derivation=None)` | one `MapRow`, or `None` |

A shard with no map gives no rows. Ask `binding` to tell "no map" from "an empty
map".

### Sources, members, source links

| Method | Returns |
|---|---|
| `sources()` | the source names |
| `members(source=None)` | the `Member` entries of a source (all when `None`), each with its parts |
| `member(name)` | one `Member`, or `None` |
| `member_occurrences(member_name)` | *lazy*: the occurrences the member covers (every record of a whole-file part, the one record of a record part) |
| `members_covering(ref)` | the members whose parts cover this occurrence: where a chain back to an ingested revision ends |
| `file_members(instance_id)` | the `(member name, source name)` pairs of the members that hold the whole file of this artifact, in the order the members were registered |
| `has_record_parts(instance_id)` | whether any member names single records of this artifact |
| `record_members(ref)` | the `(member name, source name)` pairs of the members that name this one record as a part; the members of its whole file are not among them |
| `sources_of_artifact(instance_id)` | the names, sorted, of the sources that have a member with a part in this artifact or that a source link of it names; it follows no edge |
| `source_links(ref)` | the `SourceLink` entries that hold for one occurrence: its own, then those of its whole file |
| `file_source_links(instance_id)` | the links of the whole file |
| `record_source_links(instance_id)` | *lazy*: every link of a single record of the artifact |

`file_members`, `has_record_parts` and `record_members` are what a walk over many
records reads in place of `members_covering`, which returns each member with all
of its parts: the first two are read once per artifact, and the third only for an
artifact where the second is true. None of them loads a member's list of parts.
`record_members` finds its rows through the artifact and the locator, for which
the schema has no index: it scans the member parts, so it is meant for artifacts
with few record members. The three together cover exactly what `members_covering`
covers.

A `SourceLink` carries the instance id, the locator (`None` for the whole file),
the member, its source, the label, the evidence text, and the derivation and
verification as stored. A source link never replaces an edge. A record with an
observed edge to one source and a file-level link to another shows both, so a
request about the second source is not decided by the first (plan 2.2(b),
Decision 3).

### Drops

| Method | Returns |
|---|---|
| `drops_from(input_instance_id)` | *lazy*: the `StoredDrop` entries of an input artifact |
| `drops_of(ref)` | the drops of one record: one per step name and output that removed it |
| `drops_in_group(group_key)` | the drops that carry a group key (the losers of one dedup group; two outputs that use the same key each contribute theirs) |
| `drops_by_output(output_instance_id)` | *lazy*: the drops recorded for the step that wrote an artifact |

### Mixtures, runs, traces, checkpoints

| Method | Returns |
|---|---|
| `mixture_shards(instance_id)` | the snapshot as `records.SnapshotShard` entries in the trainer's order |
| `mixture_settings(instance_id)` | the settings the mixture was registered with, as the JSON object handed in; `None` when the id is no registered mixture |
| `mixtures_holding(shard_instance_id)` | the instance ids of the mixtures that name the shard |
| `runs()`, `segments(run=None)` | names, in registration order and in run order |
| `segments_using(mixture_instance_id)` | the segments that trained on a mixture |
| `segments_started_from(checkpoint_instance_id)` | the segments whose parent is this checkpoint |
| `segment(name)` | a `Segment`: its fields, the run record as a dict, `launches`, `cache_digests`, `shuffle_digests`, `steps`; a field the run record did not hold is `None` |
| `trace_states(segment, upto_step=None)` | *lazy*: one `TraceState` per registered step: traced or not, the trace's count, its sample rows, the number of samples with an uncovered range, the number of samples whose coverage mark claims more tokens than their stored rows add up to |
| `step_trace(segment, step)` | the `records.StepTrace` with samples and rows, or `None` for an untraced step |
| `step_traces(segment, first_step=None, last_step=None)` | *lazy*: the traces in step order, one step loaded at a time |
| `trace_documents(segment, upto_step=None)` | *lazy*: the distinct (shard position, ordinal) documents the traced steps read, each with the first step that read it |
| `trace_hits(segment, shard_position, ordinal, upto_step=None)` | *lazy*: every trace row that reads one document, with its sample's coverage mark |
| `checkpoint(ref)` | a `Checkpoint` by name or instance id, or `None` |
| `checkpoints_of(segment)` | the checkpoints of a segment in step order |
| `checkpoint_history(ref)` | a `History`, or `None` for an unregistered checkpoint |

`History.steps()` is the ordered list of (segment, step) through the ancestors,
oldest first: for the checkpoint's own segment the registered steps up to its
step, then the same for the segment its parent checkpoint lies in, and so on.
`History.segments` gives the same per segment. `complete` is true only when the
walk reached a segment that started from scratch. Otherwise `missing` holds the
`Reference` the walk stopped at (a segment with no row, or a parent checkpoint
with no row), and the part before it is unknown: a history that is not complete
must never be read as a shorter history.

## The loud checks

`store.check_integrity(instance_ids=None, *, models=False)` is the one implementation of the loud
faults (plan 3.4, the last paragraph of 2.2). A fault raises a named error of the
integrity family (exit code 3). A query calls this before it prints anything and
prints no answer when it raises.

What each check compares (D-18), in the order in which they raise:

| Order | Error | What it compares |
|---|---|---|
| 1 | `MapCountMismatch` | a binding's declared count with the number of rows the store holds under it (occurrence rows of a JSONL artifact, map rows of a shard) |
| 2 | `MapDigestMismatch` | a binding's declared digest with the digest its artifact is registered with |
| 3 | `EdgeParentMissing` | an edge input, a map row or a candidate entry with the occurrence row it names: the row is gone |
| 4 | `TraceSampleCountMismatch` | a present trace's consumed-sample count with its step's registered batch size, and with its own number of sample rows; a trace whose step has no registered batch size |
| 5 | `ArtifactDigestMismatch` | an artifact's registered digest with its file on disk; a shard with one of its two files gone; a mixture snapshot entry's digest with the digest its shard is registered with |
| 5, after the digest | `MapCountMismatch` | only for a file that is present and has its registered digest: every binding's declared count with the number of records the file holds (the lines of a JSONL file as `ids.split_lines` defines them, the documents of a shard's index) |
| 6 | `StoredDigestMismatch` | a segment's run record, the canonical text the store holds, with the digest registered with it; a segment registered with no run record has neither. The registration takes the digest over the text the store holds (the canonical text of the record as read back), whatever the record's keys were when it was handed in |
| 7 | `StoredDigestMismatch` | a mixture's snapshot entries and settings with what the mixture was registered with: its content digest (from the entries' digests in position order), its parameter digest (from its settings, when the stored settings text is the canonical text of the document it parses to) and its instance id (from the entries' shard instance ids, those two digests and the tool's own step); a mixture with no snapshot entry at all is not compared (below, "States that are not faults") |
| 8 | `LocatorDomainMismatch` | the numbers of the records under a binding with the count the binding declares: the occurrence rows of a JSONL artifact that declares n lines are L1 to Ln, the map rows of a shard that declares n documents are the ordinals 0 to n-1; an artifact whose file is gone from its recorded place is not looked at (D-18) |
| 9 | `CacheDigestConflict` | a segment's recorded digests for one array of the blend (shard position null): there is one. The schema's unique index (version 3) refuses a second row, so the check meets one only in a store whose index was removed; a store that holds two rows is refused at migration with the same error (above, "Opening a store") |

What the store protects against. Checks 6 to 9 hold a record against a digest or a
count the store recorded of it, so they find a record that was altered or removed
without the digest or the count that went with it: an accidental edit, an
inconsistent import, a half-restored copy. They do not defeat a change that
rewrites every commitment beside the record, such as a run record and its digest
together or a mixture's snapshot, settings and id together, and the store makes no
claim that they do. That the store is written only by its own methods is a rule of
the code, not protection against someone who edits the file.

A binding's declared count is so compared twice: with the store's own rows in
check 1 of this table, and with the file in check 5, in the read that takes the
file's digest. The second comparison is there because the first one can be
satisfied by two edits that hide each other: rows removed from the store and the
declared count lowered by as many. The store then agrees with itself, and the
file still holds every record. With the file present that is "map count
mismatch": the message names the declared count and the count on disk, and the
error's context carries `declared`, `on_disk`, `instance`, `path`, `namespace`
and `derivation`. The message of check 1 names the declared count and the
store's rows, and its context carries `stored` in place of `on_disk` and `path`.

The store's checks 1 to 4 compare records of the store with other records of the
store. They never open an artifact's file. Of this table, check 5 alone reads
files: each file once, for its digest and its record count together. The
consequence, and the reason for the split:

- A file that differs from the registered one, with the store untouched, raises
  "artifact digest mismatch" and no map fault. This holds for one changed byte
  and for an added or removed line alike: the count is held against a file only
  when the file has its registered digest.
- A count or digest edited in the store raises its map fault: in checks 1 and 2
  when the store's own records then disagree, and in check 5 when the store
  agrees with itself and disagrees with the file it is registered with.

So the two kinds of fault can be told apart, and no record of a file that is
present is invisible to the count (plan 2.2(d), 2.3).

A shard whose index cannot be read as an index has no count to compare. When
its digest differs from the registered one, that is "artifact digest mismatch"
as for any changed file. When the digest is the registered one (the store was
edited to agree with such a pair) and the shard has a map, the check stops with
"map count mismatch" and says that the documents cannot be counted; `on_disk`
is then null.

The order in which a fault is found:

1. The kinds are tried in the order of the table. A store with a fault of
   checks 1 to 4 and a fault of check 5 names the fault of checks 1 to 4.
2. Inside checks 1, 2 and 5 the artifacts are taken in registration order, and an
   artifact's bindings in the order they were written.
3. Inside check 3: edge inputs first, then the parents of map rows, then candidate
   entries.
4. Inside check 4: the segments in registration order, each one's steps in step
   order.
5. Inside check 5 each artifact is looked at once and its file is read once: for
   a shard, one of its two files gone; then the digest; then, when the digest is
   the registered one, the declared count of each binding. An earlier artifact's
   count fault is therefore named before a later artifact's digest fault. For a
   mixture: its snapshot entries by position. A snapshot entry that names a
   shard the store was never told of is not compared: there is nothing
   registered to hold it against.
6. Checks 6 to 9 come after check 5, so a store with a fault of checks 1 to 5 and a
   fault of checks 6 to 9 names the earlier one. Inside check 6 the segments are
   taken in registration order. Inside check 7 the mixtures in registration order;
   for one mixture the content digest is held first, then the settings, then the
   instance id. Inside check 8 the bindings in the order of check 1, and for one
   binding the first row (by write order for a JSONL artifact, the highest
   ordinal for a shard) outside its range is the one named. Inside check 9 the
   segments in registration order, each one's kinds in alphabetical order.

The first fault found is raised and the check stops. A read method that meets a
dangling parent raises the same `EdgeParentMissing`.

Scope. With `instance_ids`, the checks run over those artifacts and their ancestor
closure:

- a JSONL artifact: its step's inputs, and the artifacts that hold the records its
  edges and harness logs name;
- a shard: its step's inputs, and the artifacts the rows of any of its maps name;
- a mixture: its shards;
- a checkpoint: its segment's mixture and the checkpoint the segment started from,
  and for the trace check the steps of its history only (a later step of the same
  segment is outside it).

Checks 6 and 9 run over the run segments of the histories of the checkpoints in
the closure (a later step of the same segment does not matter to them: they hold
a segment, not a step). Check 7 runs over the mixtures in the closure and check 8
over the bindings of the artifacts in it. A scope that holds no checkpoint meets
no run segment, and a run record or a blend digest that is altered is not
found by a query about a JSONL artifact, a shard or a mixture.

An id the store has no row for has no closure. With `None` the whole store is
checked.

### States that are not faults

The proof reads these and answers from them. `check_integrity` does not raise:

- a step with no trace, a trace with an uncovered range, and a trace with a sample
  whose coverage mark claims more tokens than the sample's stored rows add up to
  (its rows are gone, or the mark was written beyond them): an untraced step
  (`trace_states`, `step_trace`);
- a step that a segment's launches cover and that is not among its registered
  steps, whatever the reason: the operator's step record never listed it, or its
  step row and its whole trace were removed together. The store holds nothing that
  tells the two apart (a launch records its first and last step, and no count or
  digest of the registered steps is recorded). The step is untraced, and so is
  every step that lies in the history and is not registered
  (`coverage.untraced_steps`, docs/coverage.md condition (e)); a checkpoint's
  answer reads it as "may contain" and never as a shorter history;
- an artifact with no binding: no record of its own (`bindings`);
- a mixture with no snapshot entry at all: no record of its own, "not covered",
  "record missing"; check 7 compares the entries a mixture still holds, and an
  entry removed from a snapshot that keeps others is the fault;
- an occurrence with no edge: a missing link (`edge_of` returns `None`);
- an edge label, a map label, a source link's label or a bound field edited after
  writing: returned as stored, for conditions 2.2(f) and 2.2(g);
- an edge input moved to another record that exists, an input's stated-as-used
  flag edited, or an edge removed: an artifact keeps the digest of the edges its
  step stated, and no check recomputes it from the stored edges. The edge is read
  as stored, and an edge that is gone is an occurrence with no edge (above);
- a trace row moved to another shard position or document over the same token
  span: the store records no digest of a trace's rows (the run record's cache
  digests cover the index arrays, not the stored rows), and check 4 compares
  counts only. The trace is read as stored;
- a harness log that was removed or gained a read: `harness_log` is `None`, or
  `bound_established()` is false;
- a reference handed to a job together with the text of another record, in a log
  registered without its texts, on an edge with none (read back from its file, fetched
  from the store with `harness_log`, or built by hand): the log
  holds no text and the store has nothing to compare, so it registers the log and
  the edge for the reference and stores the log as not compared: the hand-off is only the
  caller's statement and the log establishes no bound until the texts are shown and
  compared (`verify_hand_off`; docs/confine.md section 5). A result as `run_job` or `run_confined_job` returned
  it carries its texts and is compared, and the registration is refused;
- an artifact whose file is no longer at its recorded place (archived, or the
  store was moved without it): the artifact check has nothing to compare, the
  records stay and still answer, and `missing_files` lists the artifact so that an
  answer can say so. A team that needs a gone file to stop a query would get that
  as a setting; it is not the default. For a gone file the declared count is
  compared with the store's rows only, and check 8 does not look at it. Rows
  removed from the store together with a lowered count are then not found, for as
  long as the file is away: nothing is left to count them against. A negative
  answer names such a file in its
  scope (`files_not_checked`), `check` names it in `missing_files`, and the count
  fault is raised as soon as the file is back at its recorded place.

## Limits

- A verified map and a recovered edge cannot be replaced by a later, better one.
  A map that did not verify can be followed by another map of its namespace and
  derivation (above); an edge has no such rule, so a recovery writes no edge it
  could not stand by (docs/recovery.md, "Surplus copies").
- The store does not tell a second registration that leaves a stored drop out
  from one made before the drop was added (above).
- `check_integrity` over a checkpoint reads the file of every ancestor once on
  each call, for its digest and its record count. Its cost at scale is not
  measured here.
- A file that is gone from its recorded place is not counted: rows removed from
  the store together with a lowered declared count stay unseen until the file is
  back (above, "States that are not faults"). The same holds for check 8: a
  locator changed in the store is not found while the file is away.
- The store's size and speed at scale are not measured here (plan risk RK3).
