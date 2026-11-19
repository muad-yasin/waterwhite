# The pipeline replica

The replica is test infrastructure. It stands in for a team's data pipeline, so
that the tool has something real to trace. It is not part of the tool and the tool
never imports it.

It is this project's own implementation of the published steps of a pretraining
data pipeline (plan Section 13, the heading on the published pipeline; plan 5.2).
No code of that pipeline is copied or run.

Plan sections it serves: 5.1 (the fixture is built with it), 5.2 (the steps and the
two layouts), 5.4 (the capture points and the recovery points), 4.4 (it builds both
sides of a repair), 3.3 "Tokenization" and F1 (the conditions of a tokenizer run
and the capture map), 3.3 "Mixture selection" and Decision 27 (the snapshot), 3.3
"Generation" and F10 (the harness a job runs in), F8 (the sanitizer is the tool's
named normaliser), Checks 3 and 6 of 5.5 (the maps of a generated set, the run
made without capture and its recovery), 3.3 "Training" and 5.4 (the run wrapper,
section 13), and 5.6 (the run over a generated set, section 14).

## 1. What it does

One build takes source files and writes releases, the files the pipeline steps
make, tokenized shards and mixtures. The steps, in order:

1. **Sources.** Each source file is copied byte for byte.
2. **Generation stand-in.** Each job emits its fixed output from
   `generation.jsonl`. No model is called. A job is handed records of the source
   files and of earlier outputs, never of a release: generation reads the source
   revision, not the deduplicated release (plan 5.1).
3. **Releases.** A release is a list of records, with an exact dedup on the
   normalised text that can be switched off. Its rows hold the text only
   (DECISIONS.md D-06).
4. **The pipeline**, once per layout: list the files, shuffle, deal round-robin
   into N chunks, merge, shuffle, sanitize, tokenize.
5. **Mixtures.** A mixture is a folder that holds the shard files it selects.

Steps 2 and 3 are the fixture's additions. The published pipeline has no dedup and
no generation step.

### The dedup step

Records whose normalised texts are equal form a group. The group keeps one line,
at the place of its first copy. Which copy is kept follows the release's dedup
priority, a list of source names: the kept copy is the first copy from the source
that stands first in the list. A group with no copy from a listed source keeps its
first copy. So a release has the same lines whichever copy wins; only the
attribution moves.

A line that does not parse, and a record whose text is empty after normalising, is
never put in a group, however many such lines there are. Each stays a row of the
release and is carried on until the sanitizer drops it. The plant list of a
generated set counts on this (docs/plant.md section 5.1).

### The two layouts

- **round-robin**: every listed file is shuffled, then dealt over the N chunks
  starting at the first chunk. The piles of all files are merged chunk by chunk, in
  listing order. Each chunk is then shuffled once more. A file with at least N
  records therefore reaches every chunk (plan 4.5).
- **partitioned**: each shard holds the records it is given and nothing else. The
  records of one shard are merged and shuffled once. A withdrawal's records then
  sit in few shards and the others can be reused whole.

Round-robin chunks are named `shard-1` to `shard-N`, zero-padded to the width of N
(`shard-01` to `shard-20`).

### Seeded and unseeded

Every shuffle has both modes.

- **Seeded** (the default): the same seed gives the same bytes, on any machine and
  under any Python version. The draws come from SHA-256 over the seed, a label of
  the shuffle (`file/<name>`, `chunk/<shard>`, `shard/<shard>`) and a counter. A
  layout has one seed; each shuffle has its own label.
- **Unseeded** (`--unseeded`): the order comes from the operating system's
  randomness, on purpose, as a `shuf` with no seed does. Two builds differ and
  nothing about the order is recorded.

### The sanitizer

The sanitizer applies the build's normaliser to each text. The normaliser is the
function registered under its name in `waterwhite.normalise`: the replica holds no
copy of it and imports it in every mode, with capture on and with capture off
(DECISIONS.md D-17). The first build has one, `control-chars/1`; docs/capture.md
gives its rule in full. So a sanitized record and the fingerprint the tool takes of
the record it was made from agree.

A line is skipped, and the skip logged, when it is

- blank, or empty after sanitizing: reason `empty line`;
- not JSON, not a JSON object, or without a string `text` (in the fixture also
  without a string `id`): reason `sanitizer skip`.

The two reasons are spelled as the tool spells its drop reasons. Surviving lines
keep their order. A skipped line is a drop: it becomes no document, so the line
after it gets the ordinal the skipped line would have had.

### Tokenization

Each chunk is tokenized by `tools/preprocess_data.py` of a Megatron-LM checkout,
run as a subprocess with the interpreter of the test-time environment. The replica
imports nothing from Megatron. The conditions of every run (plan F1, 5.3):

- `--tokenizer-type GPT2BPETokenizer`, with the vocabulary file and the merges
  file handed over by path. The spec's `[tokenizer]` table names both files, as
  paths relative to the environment folder, and the hub revision they belong to.
  Without the files Megatron's tokenizer class tries to download them;
- `HF_HOME` set to the environment's `hf_home/` and `HF_HUB_OFFLINE=1` (also
  `TRANSFORMERS_OFFLINE=1`) for every subprocess, so that nothing is fetched and
  nothing is written to a cache in the home folder;
- `--append-eod`;
- `--partitions 1` and one worker, so that document order is line order;
- a new, empty input folder for each run;
- the tokenizer built once, offline, at the recorded revision, before the first
  run;
- a time limit, and a watch on the script's output: the script hangs when one of
  its worker processes dies, so a run that prints an error and does not exit is
  stopped and reported.

After a run the replica reads the shard's index file and refuses the build unless
the shard holds one document per line of its tokenizer input, and the `.bin` file
has the size the index gives.

**The tokenizer as a function of one text.** A tokenization map is verified by
tokenizing every input line again (plan 3.3), and the tool ships no tokenizer. The
replica hands it Megatron's own: `megatron_tokenizer.LineTokenizer` is one more
subprocess of the environment, `replica/megatron_line_worker.py`, which builds the
tokenizer as `tools/preprocess_data.py` does (the same argument parser, the same
values set after parsing) and answers one text per line with its token ids,
without the end-of-document token. It is started offline with both tokenizer
files, checks the end-of-document id, and is stopped when a reply does not come
within the tokenizer's time limit. It starts in about seven seconds; one call took
about 0.4 milliseconds in one run of 2,000 short texts on a shared laptop, which
is no benchmark.

The environment folder is read from the variable `WATERWHITE_MEGATRON_ENV`. It
holds `venv/`, `src/Megatron-LM` and `hf_home/`; `tools/setup_megatron_env.sh`
builds it.

Megatron-LM (NVIDIA, commit `685fb8bd970185e9ac3283fe32a5d6667bf4b268`, tag `core_v0.20.0`) is installed into that folder by the script, not redistributed with Waterwhite; its licence is BSD-3-Clause plus its third-party notices (`LICENSE` in the Megatron-LM checkout).

## 2. The folder one build writes

```
sources/<KEY>.jsonl                                the source files
gen/<NAME>.jsonl                                   generation outputs, rows {"id", "text"}
gen/harness/<NAME>/<id>.log.jsonl                  the harness log of each job (capture on only;
                                                   with either harness, the file differs only in
                                                   the harness word of its job entry)
releases/<NAME>.jsonl                              rows {"text"} only
releases/<NAME>.dedup.jsonl                        the dedup step's own log (when dedup is on)
imports/<NAME>.jsonl                               a text-only import, rows {"text"}
pipeline/<layout>/sanitizer-input/<shard>.jsonl    a chunk as the sanitizer reads it
pipeline/<layout>/sanitizer.log.jsonl              one row per skipped line
pipeline/<layout>/tokenizer-input/<shard>.jsonl    sanitized, surviving lines, rows {"id", "text"}
pipeline/<layout>/shards/<shard>.bin, .idx         the tokenized shard
mixtures/<name>/<shard>.bin, .idx                  the shard files a mixture selects
<file>.manifest.jsonl                              beside every record file above
built-from.jsonl                                   the file-level manifest (capture on only, section 16)
.work/                                             scratch: tokenizer input copies and logs,
                                                   the log of the tokenizer as a function
```

`<layout>` is `partitioned` or `round-robin`.

- **Sanitizer input**: the chunk after the last shuffle, one line per record, each
  line as its source file or generation output holds it, byte for byte, readable
  or not. A line number in the sanitizer's log is a line of this file.
- **Tokenizer input**: the lines that survived the sanitizer, in order, each
  written as ASCII JSON `{"id", "text"}` with the sanitized text. Line i (from 1)
  is document i - 1 of the shard.
- **Manifest**: one row per line of its file,
  `{"line": n, "handle": "...", "origin": ["A-001"]}`. `handle` is the spec's
  handle of the record when it has one (`X@A`, `P`), else its id. `origin` is the
  id of the record the line is a copy of; for a generation output it is the ids the
  output states it used. The line the spec writes into a chunk (below) has the
  handle `sanitizer_skip` and an empty origin. Manifests are test evidence. The
  tool never reads them.
- **Sanitizer log**:
  `{"shard", "input", "line", "id", "origin_file", "origin_line", "reason", "detail"}`.
  `input` and `line` name the sanitizer-input file and the line there. `id` is the
  line's id, null when it has none. `origin_file` and `origin_line` name the place
  the pipeline listed the line from (a source or generation file in the fixture, a
  release in a generated set); both are null for a line that came from no file.
- **Dedup log**: one row per group with more than one copy,
  `{"line", "kept", "dropped", "rule"}`.
- **Ids** are the replica's own. They are kept up to the tokenizer input (plan
  Check 3) and a release row carries none.
- **Mixtures** hold copies of the shard files (`--mixture-files symlink` makes
  relative links instead).
- A shard that no line reaches is not written at all. A shard all of whose lines
  are skipped has a sanitizer input and no tokenizer input. The build's summary
  marks both `"empty": true`, and a mixture that names such a shard is refused.

Everything outside `.work/` is the same bytes in a second seeded build.
`python -m replica digests --out DIR` prints the SHA-256 of each such file.

## 3. Input mode one: the hand-checked fixture

`build-fixture` reads `fixture/spec/fixture_spec.toml`. Records are selected by id;
`"Z-001..Z-020"` means every id from the first to the last. What the replica reads:

| Spec | Use |
|---|---|
| `megatron.partitions`, `megatron.workers` | the tokenizer run; partitions must be 1 |
| `tokenizer.type`, `.name`, `.append_eod` | the tokenizer run; the type must be `GPT2BPETokenizer`, the name `gpt2` |
| `tokenizer.revision`, `.vocab_file`, `.merges_file` | the hub revision of the tokenizer files, and the two files as paths relative to the environment folder |
| `normaliser.name` | the sanitizer and the dedup comparison |
| `release_rows.carry_source_id` | must be false |
| `sources.<S>` | the file, the member, the revision, and the ids the file must hold in order |
| `handles` | the handle of each named record, for the manifests |
| `generation.file`, `generation.model` | the fixed outputs; the model must be `"none"` |
| `releases.<R>` | `inputs`, `dedup`, `dedup_winner` (a dedup priority of one source) |
| `imports.<name>` | `records`, `file_family`, `evidence` |
| `layouts.partitioned` | `shuffle_seed` and the record list of each shard |
| `layouts.round-robin` | `shuffle_seed` and `chunks` |
| `sanitizer_skip` | the skipped sanitizer line: `shard`, `line`, `text`, `layouts` |
| `mixtures` | each mixture's name and the list of its shards, in the trainer's order |
| `runs.<R>.segments.<seg>` | `mixture` and `shards`: which mixture a segment trains on |

Every other table and key of the spec is left alone: the runs' settings, the step
records, the seed search and the read positions belong to the oracle and to the
hand-written table.

File paths in the spec are relative to the repository root, taken as the folder
three levels above the spec file; `--base` names another.

What this mode does:

- **The round-robin variant** takes the records of all partitioned shards as one
  list, shuffles it once, deals it over the chunks and shuffles each chunk.
- **Shards read the sources and the generation outputs directly**, not a release.
- **The skipped sanitizer line.** `[sanitizer_skip]` holds one line that does not
  parse. After the shuffle the replica writes it into the sanitizer input of the
  named shard at the named line, in each layout the table lists. The sanitizer
  skips it and logs it with the reason `sanitizer skip`. The line belongs to no
  source and is never ingested. In the fixture it is line 6 of `shard-2`, which so
  reads 20 lines and holds 19 documents. It is the fixture's only skipped line.
- **Mixtures.** `[mixtures]` gives each mixture as a list of shard names. A bare
  name (`shard-1`) is a shard of the partitioned layout. A name with a store
  prefix (`E-rr/shard-1`) is the shard of the layout with that prefix: `E` is the
  partitioned layout, `E-rr` the round-robin one. All shards of a mixture lie in
  one layout. The list is the order the trainer is handed the shards in (plan
  Decision 27); run K's is `shard-3`, `shard-2` on purpose. The folder
  `mixtures/<name>/` holds the files, and the build's summary keeps the order,
  which a listing of the folder would not give.
- **Segments.** Each segment names its mixture and lists its shards. The two lists
  must be equal, in order; a spec where they differ is refused. The summary lists
  each segment whose mixture was built with its run, its place in the run and its
  mixture.
- **The import** of `import/T` is written to `imports/T.jsonl`.
- Each generation output gets a file of its own, named by its id.

Store names (docs/interfaces.md section 9.2): `src/<KEY>`, `gen/<ID>`, the release
names, `import/T`, and for a shard `E/<shard>` or `E-rr/<shard>`.

## 4. Input mode two: a generated set

`build-set` reads the folder that `fixture/plant/generate.py` writes
(docs/plant.md section 4): `set.toml`, `sources/<NAME>.jsonl`, `generation.jsonl`,
and `plant_list.jsonl`, which the replica never opens. A clean set (written with
`--clean`, with no source A) is read the same way.

What the replica reads of `set.toml`:

| Key | Use |
|---|---|
| `format` | must be `"waterwhite.plant-set/1"` |
| `normaliser.name` | the sanitizer and the dedup comparison |
| `parameters.seed` | the shuffle seed of both layouts |
| `sources.<NAME>` | `file`, `member`, `revision`, `lines`, and `records`, the id of every line |
| `generation` | `file`, `model` (must be `"none"`), `rows`, and `jobs.<job>`, the ids each job made |
| `releases.<R>` | `inputs` as ids and ranges, `dedup`, `dedup_winner`, `dedup_priority` |
| `layouts."round-robin".chunks` | the chunk count |

**An id names a line.** `Z-000037` is line 37 of `sources/Z.jsonl`, whatever that
line holds. A line that does not parse has no `id` field and still has its id, so
a release can select it, the copy steps carry it as it is, and the sanitizer's log
names it. A row whose `id` field differs from the id of its line is refused, and so
is a file with another number of lines than the set lists.

What `set.toml` does not say, and what the replica does about it:

| Not in `set.toml` | The replica |
|---|---|
| a tokenizer | takes it from `--tokenizer-spec FILE`, a TOML file with a `[tokenizer]` table and optionally `[megatron]` (the fixture spec is one). Without it the build must run with `--no-tokenize`. |
| a shuffle seed for the pipeline | uses `parameters.seed`, the seed the set was generated with; `--shuffle-seed` replaces it. Both sides of a repair then share one seed. |
| which release is chunked | chunks the rows of `R1` and of no other release. `R2` is built as a file only. |
| a partitioned layout | one shard per source file, named `shard-<NAME>` (`shard-A`, `shard-Z2`), and one shard `shard-gen` for all generation outputs. |
| mixtures | builds none. The folder `pipeline/<layout>/shards/` holds every shard of a layout and can be selected as it is. |
| a name for the set | registers its shards as `set/<shard>` (partitioned) and `set-rr/<shard>` (round-robin). `--name-prefix` puts more before it. |
| one output file per job | writes `gen/<job>.jsonl` (`gen/paraphrase.jsonl`, `gen/translation.jsonl`, `gen/summary.jsonl`) with the job's rows in file order, registered as `gen/<job>`. |

What this mode does:

- **The shards are built from the rows of `R1`**, after its dedup. With
  `dedup = false` both copies of a pair reach a shard (plan Check 3).
- **The listed files of the round-robin layout** are the files the inputs of `R1`
  come from, in the order the inputs first name them. The rows of the release that
  came from one file are shuffled and dealt as that file.
- **A partitioned shard** holds the rows of the release that came from its files.
  A shard with no row (source B when every one of its rows of `R1` lost the dedup)
  is not written.
- **A row dedup kept** belongs to the file of the kept copy.
- **A job with one row** in `generation.jsonl` is one harness run named by the job.
  A job with several rows is one run per row, named `<job>/<N>`, where N is the
  row's number in the job's output, counted from 1. A job name is stored, and the
  ids of a set are not the tool's (plan Check 3), so no name holds an id.

`fixture/plant/count.py --chunks` reads the folder
`pipeline/<layout>/tokenizer-input/` of a build.

## 5. Capture

Capture is a switch on every build.

**Off** (the default): the replica loads one module of the tool, its normaliser.
The store, the capture wrappers and the harness are not imported. A run made this
way is the "run made without capture" of plan Check 6. Section 6 says how such a
run is put into a store afterwards.

**On** (`--capture --store FILE`): the build calls the tool at the capture points
of plan 5.4 (docs/interfaces.md sections 4 and 5, docs/capture.md,
docs/tokenmap.md). All of it is in `replica/capture_hooks.py`. The store file is
created with the build's normaliser when it does not exist and opened otherwise; a
store made with another normaliser is refused.

| Capture point | Call | What is handed over |
|---|---|---|
| each source file | `capture.ingest_jsonl` | source, member, revision; name `src/<KEY>` |
| each generation job | `harness.run_job`, or `confine.run_confined_job` (below) | the handed records as their registered files hold them; the job reads each of them through the harness, calls no model and states what it used. The log goes to a new file under `gen/harness/` |
| each generation output file | `capture.record_jsonl` | one generation edge per row, built by `capture.generation_edge` from what the job's output states it used, with the harness log; name `gen/<NAME>` |
| each release (the dedup selection) | `capture.record_jsonl` | for a row with no equivalent copy a single-parent edge (`capture.copy_edge`); for a dedup group an "any" edge that names every copy with the kept one selected, and a drop `dedup loser` for each loser (`capture.selection_edge`) |
| the text-only import | `capture.import_jsonl` | the file family `(member, evidence)`; member name: the import's name |
| each tokenized shard | `tokenmap.map_tokenized_shard` | the registered tokenizer input, the shard's file prefix, the name `E/<shard>`, the tokenize step, the settings of the run and the tokenizer as a function of one text. The tool registers the shard and writes its capture map |
| each mixture | `Store.register_mixture` | the snapshot: the shards in the trainer's order, each by its registered instance, with the digest of the two files that lie in the mixture folder; the settings |

The text-only import is not among the capture points plan 5.4 lists. The fixture
holds it (plan 5.1) as a derivation that was never recorded (plan L1), and the
build registers it at the same place.

A generated set is captured at the same points, under the names of section 4.

The inputs a step declares are the artifacts of every record its edges and drops
name, each once, in the order the step first names them. Every parent lies in a
declared input (DECISIONS.md D-15).

The release step's parameters do not include the winner. Two releases that differ
only in which copy dedup kept then have the same bytes, the same step and the same
inputs, and differ in their edges alone, so the store gives them two instance ids
(plan Check 7, Decision 29).

**The store's drops.** The store holds one drop per dropped record, step name and
output artifact. Two releases that drop the same record (the same dedup loser)
each hold their own drop, so a second release with other content is registered
beside the first one.

### The tokenization capture point

The hook `shard_tokenized(files, tokenizer, tokenize, normaliser)` is called after
each shard is tokenized and held against its input. `files` is a
`replica.shard_files.ShardFiles`: the chunk as the sanitizer read it, the
sanitizer's log, the tokenizer input, the prefix of the shard's two files and the
record files the copy steps list for the shard. In order:

1. The chunk as the sanitizer read it is registered as a file found afterwards
   (`recovery.register_found_file`), under `<shard>.sanitizer-input`, and the copy
   steps before it are recovered by fingerprint (`recovery.recover_copy_step`).
2. The tokenizer input is registered the same way under `<shard>.input`, and the
   sanitizer between the two files is recovered (`recovery.recover_sanitizer`)
   with the sanitizer's log.
3. The shard is registered and its capture map is written over the tokenizer
   input (`tokenmap.map_tokenized_shard`).

The copy steps and the sanitizer are recovery points (plan 5.4): the build states
no edge for the files they write, with capture on as with capture off. Their files
are registered at this point because the capture map names the lines of the
tokenizer input, and a document's chain to its source record passes them.

**The files the copy steps list for a shard** are the possible inputs a recovery
of the copy steps names. They come from the layout, not from the shuffled chunk: a
partitioned shard lists the files of its own records, and every listed file can
reach every round-robin chunk. In the fixture the shards of run E list the files
of B and Z, and shard 3 also `gen/T`; none lists the file of A, so B's copy of the
shared passage has one parent there and not two candidates (plan L12). In a
generated set the listed file is the release `R1`, whose rows the pipeline chunks.
The build's summary keeps the list per shard (`listed`). A listed file that is not
registered (a generation output of a build that leaves the generation step
uncaptured) is passed over, and the records copied from it keep no recorded
origin.

**A dropped line is recorded exactly once**, as a drop whose output is the first
registered artifact that lacks the line:

| The line | Where its drop is recorded |
|---|---|
| a line the sanitizer skipped, or an empty line, with the tokenizer input registered | under the tokenizer input, by `recovery.recover_sanitizer` from the sanitizer's log. The map is written over the tokenizer input and sees no such line |
| the same, for a run that kept only the sanitizer's input (section 6) | under the shard, by `tokenmap.recover_map`, which is given the sanitizer's log |
| a line of the tokenizer input that gives no tokens | under the shard, by the map call |

The step name of a sanitizer's drop is "sanitize" in both cases, and of a line
without tokens "tokenize".

**The map.** The settings every map records come from the build's tokenizer
table: `append_eod`, and the end-of-document convention `eod_in_span` with
`eod_token_id`, which the fixture spec records from the oracle's output (plan 3.3,
F1). A table that does not record the convention is refused before anything is
written. The shard is registered under the step "tokenize", with the hub revision
of the tokenizer files as its version and the switches of the run as its
parameters. The tool builds the map in position mode and verifies every document
against its line. A capture map that does not verify stops the build.

### The mixture capture point

The hook `mixture_selected(folder, relative, name, shards)` is called after each
mixture folder is filled. `shards` is the ordered list the trainer is handed, each
entry the shard's store name and its file prefix inside the folder. The snapshot
keeps that order (plan Decision 27), names each shard by its registered instance
and carries the digest of the two files in the mixture folder; the store refuses
an entry whose digest differs from the digest its shard is registered with. A
path of the snapshot is kept as the store keeps a path: relative to the store's
folder when it lies under it.

The settings of a mixture are `{"selection": "directory", "directory":
"mixtures/<name>"}`: the trainer is handed the files of one folder, and which
folder, relative to the build folder. A mixture's instance id covers its shards
and its settings, so `E/mix-1` and `D/mix-1`, which select the same two shards
from two folders, are two mixtures.

### The hook the build calls, and where runs are registered

One method of `StoreCapture` is called by the build at its place and holds no code:

| Method | Called | What it does |
|---|---|---|
| `run_segment_ready(segment, folder, mixture_name)` | once per segment of the spec whose mixture was built | nothing |

So a store the build makes holds no run, no segment, no step trace and no
checkpoint. The runs are made and registered after the build, in a second step
(section 13): `build-runs` writes the run records, step records and cache folders,
and `register-runs` hands them to the tool's `register-run`.

### Names and a second build

`--name-prefix P` puts `P` before every name registered, so a second build (a
repaired side) can share a store with the first. Member names and harness job
names carry no prefix: a file with the same bytes and the same history is the same
instance in both builds and gains the second name. A file found afterwards is
registered under its prefixed name, which is part of its instance id.

### The confined harness

Only a log the confined harness wrote establishes a bound (Decision 13), and only when the
texts handed to the job were compared with the records at registration (the build hands
each job the text of each record, so they are), so a
build whose answers the table grades runs its generation jobs confined:
`build_table_store` defaults to `generation_harness="confined"` (a store that
captures no generation step, `import`, runs no job in a harness). A build may run
its jobs in process only where no graded row compares an answer on it:
`build_captured_fixture`, `build_captured_set` and the command line default to
in process, and their answers past a generation edge are then "may contain". The
option `generation_harness="confined"` (`--generation-harness confined`) runs each
job through `confine.run_confined_job`: the job is a program of
its own, `replica/jobs/fixed_output.py`, started in the tool's sandbox with no file
of the build and no network (docs/confine.md). It asks the harness for each handed
record, in hand-off order, and emits the one output.

- The program's two arguments are the fixed output and the places in the hand-off
  of the records the output states it used. The fixed output stands in for what a
  model would have written; it is no handed record. The texts of the handed
  records reach the job only through the harness.
- The confined job makes the requests the in-process job makes, in the same
  order. A harness log names the harness that wrote it (DECISIONS.md D-30): the
  job entry holds `"harness": "in-process"` or `"harness": "confined"`, so the two
  log files, and the log digest the store keeps, differ in that word and in nothing
  else. The store's instance ids, edges (the log's digest and harness word aside),
  occurrences and logs are otherwise the same either way.
- One sandbox is started per generated row, not per job: a build of the 800-record
  set (90 generated rows in three jobs) makes 90 starts, each about 84 ms, and a
  build of the 100,000-record set (11,100 generated rows, docs/plant.md section 7.1)
  would make 11,100.
- The option needs capture with the generation step captured; without it the
  build is refused.
- Where the sandbox cannot be built the build is refused before anything is
  written, with the tool's own error name at the start of the message: "sandbox
  unavailable". No job is run unconfined in its place.
- `generation_timeout` (120 seconds by default) is the time limit of one job.

### The captured fixture in one call

```python
from replica.fixture import build_captured_fixture

built = build_captured_fixture(folder)                  # no tokenization
built = build_captured_fixture(folder, tokenize=True)   # needs the Megatron environment

built.store       # folder/store.sqlite, the evidence store
built.out         # folder/build, the build folder; every manifest lies beside its file
built.manifest("releases/R1.jsonl")   # the path of one manifest
built.instances   # {"src/A": "<instance id>", ..., "R1": ..., "import/T": ...}
built.files       # {"src/A": "sources/A.jsonl", ..., "R1": "releases/R1.jsonl", ...}
built.summary     # the build's summary (section 9)
```

`folder` must be new or empty. The store lies beside the build folder, so every
path in it is relative and the folder can be moved or copied as a whole. Optional
arguments: `spec` (another spec file), `layouts`, `releases` and `imports` (the
releases and imports to build and register; by default all of them),
`capture_generation` and `generation_harness` (below and above), `parallel`,
`tokenizer_timeout`, `megatron_env`. Everything the build would refuse is refused
before the store file is made.

After the call without tokenization the store holds eleven artifacts: `src/A`,
`src/B`, `src/C`, `src/Z` with their members; `gen/P`, `gen/T`, `gen/S` with their
generation edges and harness logs; `R1`, `R1-swap` and `R2` with one edge per row
and their drops; `import/T` with its file-level link to `A/r1`.
`build_captured_fixture(folder, releases=("R1", "R2"), imports=())` gives a store
without the swapped release and without the import. With tokenization the store
also holds, for every shard of both layouts, `<shard>.sanitizer-input`,
`<shard>.input` and the shard with its capture map, and the seven mixtures.
`built.files` gives a shard the prefix of its two files and a mixture its folder.

**Handles.** A test names a record by its handle and needs its place in a file.
The place comes from the manifests, never from the store:

```python
built.file("R1")               # the record file registered under a name
built.manifest_rows("R1")      # its manifest rows: {"line", "handle", "origin"}
built.line_of("R1", "S")       # 4: the line, counted from 1, that holds the handle
built.ordinal_of("E/shard-3", "T")   # the document of a shard that holds the handle
```

`line_of` refuses a handle that no line or more than one line of the file carries,
and a name the build did not register. A release's manifest names the copy dedup
kept (`X@A` in `R1`, `X@B` in `R1-swap`); what the hand-written table calls `X`
there is for the table's reader to translate. `ordinal_of` takes the line from the
manifest of the shard's tokenizer input and reads the shard's index file with the
replica's own reader to pass over a line that gave no tokens; such a line is no
document and is refused.

**Leaving the generation step uncaptured.** `capture_generation=False` (also
`BuildOptions.capture_generation`) writes the generation outputs as a build with
capture off does: no job runs in the harness, no log is written and no output file
is registered, so the store holds nothing about them. A release that selects a
generated record cannot be captured in such a build and is refused, because its
edge would name a parent the store was never told of.

### The stores of the hand-written table

`fixture/table/README.md` section 4 names the stores its rows are asked of. One
call builds one of them:

```python
from replica.fixture import build_table_store, register_again

built = build_table_store(folder, "main")
```

| Store | What the call registers |
|---|---|
| `main` | the four sources; `gen/P`, `gen/T`, `gen/S`; `R1`, `R2`; for each of `E/shard-1` to `E/shard-3` its `.sanitizer-input`, its `.input` and the shard with its capture map; the mixtures `E/mix-1`, `E/mix-2`, `D/mix-1`, `Y/mix-1`, `K/mix-1`. No swapped release and no import |
| `rr` | the four sources; `gen/P`, `gen/T`, `gen/S` (the chain of T runs through them); for each of `E-rr/shard-1` to `E-rr/shard-3` its two input files and the shard with its capture map; `E-rr/mix-1`, `E-rr/mix-2`. No release |
| `swap` | the four sources; `gen/P`, `gen/T`, `gen/S`; `R1` and `R1-swap`. No `R2`, no pipeline files |
| `import` | the four sources and `import/T` and nothing else: the generation outputs are written and not registered (`capture_generation=False`), so the store knows T only through the import's file family |
| `no-capture` | the main store's files made with no wrapper and unseeded shuffles, recovered afterwards (section 6): the four sources ingested; `gen/P`, `gen/T`, `gen/S` as plain files with no edge; `R1` and `R2` with recovered links; for each shard its two input files and the shard with its position-mode recovery map. No mixture |
| `scratch-notokens` | Check 3's set-up `t.no-tokens-line` (section 7) |
| `scratch-500` | Check 4's store (section 7) |

Every store of the table has a build; `TABLE_STORES_NOT_BUILT` is empty. A name
the table does not define is refused.

`replica.fixture.build_table_store(folder, "main+generating-model")` builds the main store and then release R3 (`replica.capture_hooks.build_extension_release`): the four jobs of fixture/sources/generation_r3.jsonl run in the confined harness with a model client that reports each row's model; each log is registered with the texts its job was handed; the log of the job `model-entry-removed` has its model entry line removed from the harness's own file and the texts re-attached; release R3 copies the four lines. The key is not in `TABLE_STORES`; the main build is unchanged and the step writes files of its own. The runs of main are registered afterwards as for `main`.

**What the call does not register for `main` and `rr`** of the table's list is in
`replica.fixture.TABLE_STORES_LACK`, by name: `main` lacks runs E, D, Y and K with
their segments, run records and step traces, and the checkpoints `E@6`, `E@9`,
`D@3`, `Y@6`, `K@3`, `K@8` and `K@40`; `rr` lacks its two segments and the
checkpoint `E-rr@6`. They come from a second step after the build:
`replica.runs.build_runs` makes the runs and `replica.register_runs.register_runs`
(or `python -m replica register-runs`) registers them through the tool's
`register-run` (section 13). A row of the table that asks a checkpoint, a trace or
a run has a target once both have run.

**The joins a store lists as verified.** A store lists, when it is made, the
(release tag, sampler type, sampler class) rows its join is verified for (plan F6);
a run registered under a join the list does not hold has no trace route, and the
cause is "data-order trace unverified for this version" where the release is not on
the list and "data-order trace unverified for this sampler type" where it is and the
sampler is not (docs/join.md). `main` and `rr` therefore list the joins of the runs they will hold,
taken from the spec: the release it pins and the sampler class it names for each
sampler type those runs use (`main`: `single` and `cyclic`; `rr`: `single`). The
other stores list none. The list is written at creation and never changed, so a
store made without it cannot be given it afterwards.

**Tokenization.** `build_table_store(folder, name)` does not tokenize unless
`tokenize=True` is passed, as before shards were registered, so a caller that
passes no flag needs no Megatron environment. `main` and `rr` then hold no shard,
no map, no mixture and neither input file of a shard, since those are registered
at the tokenization capture point; `no-capture` holds the two input files of each
shard with their recovered links, and no shard. A scratch store is nothing but a
shard and is refused by name. With `tokenize=True` the store is built as the table
lists it: `main`, `rr`, `no-capture` and the two scratch stores hold shards, so
they are tokenized and need the Megatron environment; a missing environment is
refused before anything is written. `build_captured_fixture` with no selection
builds everything the spec holds into one store, without tokenization unless asked.

`register_again(built)` runs the same build a second time into the same store. It
writes its files into `build-again` beside the first build folder (another name
with `register_again(built, "third")`) and registers every artifact again under
the same names, with the same steps, inputs and edges. The files have the same
bytes at another path, and a path is part of no instance id, so the store must
hand back the instances it holds (plan Check 7, "registering the same history
again"). With tokenization the shards, their maps and the mixtures are handed back
too, and no second map is written.

## 6. The run made without capture, and its recovery

Plan Check 6 asks about a run made with no wrapper: unseeded shuffles, and ids
that the tokenizer drops. The replica makes such a run and recovers it in two
calls (`replica/fixture.py`, `replica/recover.py`):

```python
from replica.fixture import build_uncaptured, recover_uncaptured

run = build_uncaptured(folder, plan, options)       # folder/build; no store
built = recover_uncaptured(folder, run)             # folder/store.sqlite
```

`build_uncaptured` is a build with capture off. `recover_uncaptured` puts its
files into a new store through the tool's public calls and nothing else:

| What | Call | Result |
|---|---|---|
| each source file | `capture.ingest_jsonl` | as with capture on |
| each generation output | `recovery.register_found_file` | a plain file with no edge, no member and no source. No job ran in a harness, and a generation's parent is never recovered by similarity (plan L5), so its records have no recorded origin |
| each release | `recovery.register_found_file`, then `recovery.recover_copy_step` against the files its inputs come from | a recovered link per row. The dedup selection was not observed, so the row whose text A and B both supply gets both copies as candidates (plan L3) |
| a text-only import | `capture.import_jsonl` | as with capture on: the file family is the team's statement |
| per shard, the two input files | as at the tokenization capture point (section 5) | recovered links, and the sanitizer's skips as drops under the tokenizer input |
| per shard, the shard | `Store.register_shard`, then `tokenmap.recover_map` over the tokenizer input | the position-mode map in the recovery namespace |

No mixture is registered: a run made without capture has no mixture snapshot (plan
L6), and whether one may be registered afterwards is open (plan 5.8, B-4). The
summary lists the mixture folders under `recovery.mixtures_not_registered`.

A found file keeps its pipeline role (docs/recovery.md; DECISIONS.md D-29): `R1`
and `R2` are registered with the role "release", so a listing of this store covers
them among its releases, and every other found file (the generation outputs, the
chunk as the sanitizer read it, the tokenizer input) has the role "intermediate".
The role "import" is kept for a text-only import.

Options of the recovery:

- `kept="sanitizer-input"`: the run kept only the chunk as the sanitizer read it.
  The tokenizer input is not registered; the shard's map is recovered over the
  chunk with the sanitizer's log (`tokenmap.SanitizerRun`), and each skipped line
  is a drop whose output is the shard.
- `also_token_matching=True`: the token-matching map is recovered beside the
  position-mode map of every shard, so that the two can be compared (plan Check
  3).

A map whose verification fails is stored as it is, "unknown", and reported in the
summary. It does not stop the recovery.

`recover_uncaptured` writes a new store and refuses a folder that holds one. It
refuses a build that was made with capture on. The result is the same type a
captured build gives; its summary also holds `recovery`, what each call found.

**The ids.** The ids of the chunk files are the replica's own. They are written
into the tokenizer input of every shard, with capture on and off, and the tool is
not handed them: it reads the `text` field of a registered file and stores a
locator and a fingerprint per line. In a run made without capture the store holds
no id at all. With capture on, one kind of id reaches the store: the name of a
harness run of a generated set is `<job>/<N>`, N being the row's number in the
job's output (section 4). No id of the set is in a store, on either side, and a
test reads the whole store file for one.

## 7. The scratch stores, and a generated set with a store

### The scratch stores

Two stores of the table are no build of the fixture's pipeline
(`replica/scratch.py`). Each writes one tokenizer input, tokenizes it with
Megatron's preprocessing under the fixture's settings and registers what its check
needs. Both lie in `pipeline/scratch/` of their build folder.

**`scratch-notokens`** (Check 3, set-up `t.no-tokens-line`): `src/Z` is ingested. A
tokenizer input of five lines is written: lines 1, 2, 4 and 5 hold the sanitized
texts of `Z-001` to `Z-004`, and line 3 is a record whose text is empty, put there
past the sanitizer. It is registered as `c03/shard-notokens.input` and linked to
`src/Z` by re-running the sanitizer. The shard `c03/shard-notokens` is registered
with its capture map; its position-mode recovery map is then recovered from the
same input. Megatron's preprocessing writes four sequences for the five lines and
keeps a fifth, empty entry in the index's document list; the summary's
`line_tokens` holds 0 for that line. Line 3 is one drop, "no tokens", whose output
is the shard.

**`scratch-500`** (Check 4): `src/A` is ingested. A file of 500 lines, each with a
text of its own, is ingested under a source of its own (`c04`, member `c04/r1`) as
`c04/shard-500.input`. Its tokenized shard is registered as `c04/shard-500` under
the tokenize step, with that file as its input, and with no map: the rows of Check
4 write the map themselves (one is stopped part way, one declares a wrong count).
Nothing of this store needs a run registration. The rows of Check 4 that inject a
fault into a trace are asked of a copy of `main` with its runs registered (section 13).

### A generated set with a store

`replica/sets.py` builds a generated set with a store, for Checks 3 and 6:

```python
from replica.sets import build_captured_set, build_uncaptured_set
from replica.fixture import add_recovery_maps, recover_uncaptured
from replica import setups

built = build_captured_set(folder_a, set_dir)        # the capture maps
add_recovery_maps(built, also_token_matching=True)   # the two recovery maps beside them

run = build_uncaptured_set(folder_b, set_dir)        # no wrapper, unseeded
recovered = recover_uncaptured(folder_b, run, also_token_matching=True)
```

- `replica.fixture.build_set_store(folder, set_dir, "no-capture", layout="round-robin",
  tokenizer_spec=None, parallel=1)` is the second call in one: it builds the set
  with no wrapper and unseeded shuffles, tokenizes it and recovers it by the rule
  of the table's store of that name (the tokenizer input kept, the position-mode
  map only), into `folder/store.sqlite` beside `folder/build`. It returns the same
  type as the other builds: `.store`, `.out` and `.instances` (`set-rr/<shard>`,
  `set-rr/<shard>.input`, `R1`, `src/<source>`, `gen/<job>`). A store kind other than
  `"no-capture"` is refused by name.
- Both calls take the round-robin layout by default and the tokenizer of
  `tokenizer_spec`, by default the fixture spec. Check 3 runs on a set generated
  with `--dedup off`, so that both members of each pair reach a shard.
- `add_recovery_maps` recovers the position-mode and the token-matching map of
  every captured shard into the recovery namespace of the same store, over the
  same tokenizer inputs. A shard then holds three maps side by side (DECISIONS.md
  D-20).
- The two set-ups of Check 3 that change an input before a recovery is run are in
  `replica/setups.py`. They act on the files of a run made without capture:

  ```python
  shutil.copytree(run.out, folder_c / "build")
  setups.swap_twin_lines(folder_c / "build", "round-robin", [("shard-07", 12, 40)])
  setups.change_first_line(folder_d / "build", "round-robin")      # the first shard
  failed = recover_uncaptured(folder_d, run.moved_to(folder_d / "build"),
                              also_token_matching=True)
  ```

  `swap_twin_lines` (`t.twin-swap`) swaps the two lines of an identical pair in a
  shard's tokenizer input and refuses two lines with different texts. Which pairs
  lie in one shard is the caller's knowledge, from the plant list and the
  chunk-file ids (`fixture/plant/count.py --chunks`). `change_first_line`
  (`t.verification-fails`) replaces line 1 of the first shard's tokenizer input by
  a record whose text is "This line was changed after the shard was tokenized."
  Both keep the manifest in step with the file, and neither touches the shard.

The functions were run on a set of 800 records. The run on the set of 100,000
records is not part of the replica's tests.

## 8. Leaving records out, swapping records

A build takes two plain parameters (plan 4.4), so the same replica can build a
repaired side and a reference side:

- `--leave-out ID` (repeatable): the record is written to no release, no import and
  no shard input.
- `--swap OLD=NEW` (repeatable): record NEW takes the place of record OLD, and its
  group in a layout, wherever OLD was selected.

Source files and generation outputs are built as their inputs say; the changes act
where records are selected. The rebuild check itself is not part of the replica: the
replica builds the sides of a repair from a rebuild report (section 15) and the tool's
`verify` compares them.

A swap onto a record that the same list selects already is refused ("selected
twice"), with one exception: a dedup group selects every copy of its text, so where
OLD and NEW have the same text after the build's normaliser and NEW is in the list,
OLD just leaves and NEW stays at its own place. That is what a build from sources
that never held OLD selects, and it is how a withdrawn copy is swapped for the
equivalent copy that remains (section 15). A swap onto a record with another text
that the list holds is still refused.

In seeded mode a shard none of whose records changed is the same bytes as before,
and a repaired build equals a build from sources that never held the removed
records: a generated set built with A's records and everything derived from them
left out gives the same releases and pipeline files as the clean set.

## 9. Command line

```
python -m replica build-fixture --spec FILE --out DIR
        [--base DIR] [--layout partitioned|round-robin|both] [--no-tokenize]
        [--unseeded] [--shuffle-seed LAYOUT=N] [--jobs N] [--tokenizer-timeout S]
        [--leave-out ID] [--swap OLD=NEW] [--mixture-files copy|symlink]
        [--release NAME] [--import NAME]
        [--capture --store FILE] [--name-prefix P]
        [--generation-harness in-process|confined]

python -m replica build-set --set DIR --out DIR --layout partitioned|round-robin|both
        [--tokenizer-spec FILE] [the same options]

python -m replica digests --out DIR

python -m replica build-runs --out DIR [--fixture-build DIR] [--spec FILE]
        [--runs E,K] [--timeout S]

python -m replica register-runs --store FILE --runs-folder DIR [--spec FILE] [--runs E,K]

python -m replica probe-run --set-build DIR --store FILE --out DIR
        [--layout partitioned|round-robin] [--steps N] [--spec FILE] [--timeout S]

python -m replica repair --report FILE --historical DIR --out DIR --store FILE
        (--spec FILE | --set DIR) [--layout partitioned|round-robin|both]
        [--reference-set DIR | --reference-withdrawn SOURCE] [--reference-out DIR]
        [--sides FILE] [--base DIR] [--tokenizer-spec FILE] [--no-tokenize]
        [--unseeded] [--shuffle-seed LAYOUT=N] [--jobs N] [--tokenizer-timeout S]
        [--generation-harness in-process|confined] [--historical-name-prefix P]
```

`build-runs` and `register-runs` are described in section 13, `probe-run` in section 14,
`repair` in section 15.
`--list-join single|cyclic` (repeatable, with `--capture`) makes a new store list the
join of the spec's release for that sampler type as verified, so that a store a run is
registered in lists it (section 14).

- `--out` must be new or empty.
- `--layout` defaults to `both` for the fixture and is required for a set.
- `--no-tokenize` stops before Megatron's preprocessing: no shards, no mixtures.
- `--jobs N` runs N tokenizer subprocesses at a time (default 1).
- `--release NAME` and `--import NAME` (repeatable) build only the named releases
  and imports; without them every one is built.
- `--capture --store FILE` records the build in the store at `FILE`. The store
  path must lie outside the build folder: a build folder holds the build's files
  only, and a store file in it would be listed by `digests` and differ in every
  build. A path inside the build folder is refused. Everything the build would
  refuse (a folder that is not empty, an unknown release, a missing Megatron
  environment, an unavailable sandbox) is refused before the store file is made,
  so a refused build leaves no store behind.
- `--generation-harness confined` runs each captured generation job in the tool's
  sandbox (section 5); the default is `in-process`.
- A build prints a JSON summary on standard output: the releases; per layout the
  seed and, per shard, the lines the sanitizer read (`chunk_lines`), the surviving
  lines (`input_lines`), the skipped lines, the record files the copy steps list
  for it (`listed`) and, after tokenization, its documents, tokens and file
  digests; the tokenizer's settings, revision, files and end-of-document
  convention; the mixtures with their shards in the trainer's order; the
  segments. With capture on it also holds `instances`, each registered name with
  its instance id; `files`, each registered name with what it was registered
  from, relative to the build folder (a record file, for a shard the prefix of
  its two files, for a mixture its folder); and per tokenized shard `capture`:
  what the recovery of its two input files found, and its maps with their
  derivation, verification, label, the counts of "recovered" and "candidates"
  documents and the drops the map call recorded.
- Exit code 0: built. 1: refused, with one line on standard error. 2: a wrong
  command line.

The package `waterwhite` must be importable, because the sanitizer takes its
normaliser from it. Run from the repository root with the `src` folder on the
Python path, for example:

```
export PYTHONPATH=src
export WATERWHITE_MEGATRON_ENV=.work/megatron
python -m replica build-fixture --spec fixture/spec/fixture_spec.toml --out .work/fixture --jobs 3

python fixture/plant/generate.py --out .work/s10k --records 10000 --seed 1
python -m replica build-set --set .work/s10k --out .work/s10k-build --layout both --no-tokenize
python fixture/plant/count.py --set .work/s10k \
        --chunks .work/s10k-build/pipeline/partitioned/tokenizer-input --layout partitioned
```

From Python:

```python
from replica.build import BuildOptions, RecordChanges, build
from replica.plan import load_fixture_spec, load_set, load_tokenizer_settings

plan = load_fixture_spec("fixture/spec/fixture_spec.toml")
summary = build(plan, "out", BuildOptions(layouts=("partitioned",),
                                          shuffle_seeds={"partitioned": 7}))

tokenizer = load_tokenizer_settings("fixture/spec/fixture_spec.toml")
summary = build(load_set("set-folder", tokenizer), "out-2")
```

## 10. What it refuses

Each of these stops the build with an error; nothing is patched over.

- an output folder that exists and is not empty;
- a spec or set file with a missing or wrong item, an unknown normaliser, an
  unknown layout, release rows that carry a source id, a generation model other
  than `"none"`, more than one tokenizer partition, a tokenizer type other than
  `GPT2BPETokenizer`, tokenizer files given by an absolute path;
- a mixture that names a shard its layout does not have, shards of two layouts, or
  a shard twice; a segment whose shard list differs from its mixture's; a segment
  whose mixture is not defined;
- a skipped sanitizer line that parses, or one placed past the end of its chunk;
- a source file whose ids differ from the ids the spec lists, an id used twice, a
  fixture line that is not a readable record; in a set, a row with another id than
  its line's, or a file with another number of lines than listed;
- a generation row that uses a record it was not handed, names a record that does
  not exist, or is handed a line with no text; a job the set does not name;
- a record selected twice in one release or shard, a swap onto a record that does
  not exist, a record both swapped and left out; a release or import asked for by
  a name the spec does not hold;
- a mixture that names an empty shard;
- tokenization of a set with no tokenizer given; tokenization without
  `WATERWHITE_MEGATRON_ENV`, with a piece of the environment missing, or with a
  tokenizer file missing at the path the spec names;
- a tokenizer run that fails, hangs or passes its time limit;
- a shard whose document count differs from its input's line count;
- with capture on: a store created with another normaliser, and everything the
  store and the wrappers refuse (for example a harness log file that exists
  already, or a job name whose stored log differs);
- with capture on and tokenization: a tokenizer table that does not record the
  end-of-document convention, and a capture map that does not verify;
- a store path inside the build folder (the command line);
- the confined harness without a captured generation step, or on a machine where
  its sandbox cannot be built ("sandbox unavailable");
- a recovery of a build that was made with capture on, or into a folder that holds
  a store already; a scratch store asked for without tokenization;
- a twin swap of two lines that hold different texts;
- a probe run (section 14) into a store that does not list the sequential join of
  the spec's release, into a store that holds no registered shard at the path of one
  of the set's shards, with a number of steps the set does not give, with an output
  folder that is not new or empty, or over a layout the set build does not hold.

## 11. What it does not do

It does not train, and it does not read what a training step consumed: the oracle
does everything that runs Megatron's sampler over a dataset and logs the reads.
The run wrapper of section 13 builds the index caches and writes the run record
and the step record, but runs no sampler and reads no sample. A build registers no
run, no segment, no step trace and no checkpoint; the hook point for them is named
in section 5, and `register-runs` (section 13) registers a segment from the files
the run wrapper wrote. `probe-run` (section 14) makes one run and registers it, and
trains nothing either. It derives no tokenization map and no link itself: the
tool's wrappers and recoveries do, from the files and the settings the replica
hands them. It registers no mixture for a run made without capture. It does not
judge a repair (section 15 builds the two sides of one, and the tool's `verify`
compares them), and it does not run the set of 100,000 records in its own tests.

## 12. Tests

`tests/replica/` holds the tests. Most run without Megatron. The tests marked
`megatron` and `slow` run the real tokenizer and are skipped unless
`WATERWHITE_MEGATRON_ENV` is set: one tokenizes the fixture, and
`test_replica_megatron_stores.py` builds the stores that hold shards (`main`,
`no-capture`, the two scratch stores) and a set of 800 records in four chunks,
captured and uncaptured, each test in 15 to 40 seconds.

Everything else that needs a shard uses the stand-in of `conftest.py` for the
calls that run Megatron: a toy tokenizer of one token per byte, whose shards are
written in the layout Megatron's preprocessing writes, with a document boundary
and no sequence for a line that gives no tokens. The tool reads a toy shard as it
reads a real one, so these tests run the real store, wrappers, recoveries and
tokenization maps in a few seconds:

- `test_replica_capture.py`: the capture points of the first milestone, the
  stores `main` (without tokenization), `swap` and `import`, the handle lookup,
  the second registration, two releases that drop the same record;
- `test_replica_maps.py`: the capture maps and mixture snapshots of `main` and
  `rr`, a dropped line recorded once, the `no-capture` store and its recovery in
  both variants of what a run kept, the two scratch stores;
- `test_replica_set_maps.py`: a set of 800 records in 20 chunks with its dedup
  off: the three maps of every shard side by side, the planted pairs by the
  counting script's placement, the run made without capture, the two set-ups, and
  that no id of the set reaches a store, job names included. The shuffles of the
  run made without capture are drawn from the operating system and about three
  draws in ten place no planted pair in one shard, so its fixture draws again
  (twelve times at most) until a draw has one;
- `test_replica_harness_cli.py`: the confined harness against the in-process one,
  its refusals, and the command line's store path. A test that starts the
  confined harness carries the marker `sandbox` and asks for the fixture `sandbox`
  of `conftest.py`; where bubblewrap is missing it is skipped, by that name, with
  the harness's own reason (`sandbox unavailable: the command 'bwrap' ... is not on
  PATH`), and nothing is run unconfined in its place;
- `test_replica_tokenizer.py`: the tokenizer step, the mixtures, and the capture
  points the build calls;
- `test_replica_probe_run.py`: the run over a generated set (section 14). Its
  refusals, and that the module imports no oracle, no join and no torch, run in a
  second. The test marked `megatron` and `slow` builds a set of 800 records in twelve
  chunks with capture on and runs `probe-run` over it once (about a minute); run it
  through the CPU lock.
- `test_replica_runs.py`: the run wrapper (section 13). Its plain tests run in a
  second: the spec's runs, the step record, the release tag read from a checkout,
  and the order and arguments of the registration commands. Its tests marked
  `megatron` and `slow` make the fixture's runs once and compare each segment's
  record with the oracle's; run them through the CPU lock.
- `test_replica_repair.py`: the repair driver (section 15), on a report written or
  edited by hand: the parameters a report gives the build, the refusals, the names of
  the successors and the second build into one store.

- `test_replica_built_from.py`: the file-level manifest (section 16): a build with
  capture on writes the rows, a build with capture off writes no file, the repaired
  side writes it and the reference side does not, and the rows of the checkpoints of a run
  are appended to the build folder's file (three test items of the file, marked
  `megatron` and `slow`, make runs with Megatron: one test, and one test run for each of
  the table's two stores; each is skipped when `WATERWHITE_MEGATRON_ENV` is unset). The
  other tests of the file use the stand-in of `conftest.py` where they need a shard, so
  they need no Megatron.

The acceptance tests (`tests/acceptance/`) build their stores with
`build_table_store` and put the table's rows to the tool. The generated-set tests
build a set of 800 lines with `fixture/plant/generate.py` and compare the build
with the plant list and with what `fixture/plant/count.py` counts.

## 13. The runs

The replica's run wrapper stands in for the thin wrapper around a team's training
launch (plan 3.3 "Training", 5.4). No model is trained (plan 5.3, limit L4). For
each run segment of the fixture spec it does what a launch does before its first
step, and writes the record that the wrapper writes (docs/interfaces.md 8.1, 8.2,
8.4).

```
python -m replica build-runs --out DIR [--fixture-build DIR] [--spec FILE]
                             [--runs E,K] [--timeout SECONDS]
python -m replica register-runs --store FILE --runs-folder DIR [--spec FILE] [--runs E,K]
```

| Flag | Of | Meaning |
|---|---|---|
| `--out DIR` | build-runs | the runs folder; new or empty |
| `--fixture-build DIR` | build-runs | a tokenized folder `build-fixture` wrote (default: build one into `DIR/.work/fixture-build` first) |
| `--spec FILE` | both | the fixture spec (default: the repository's) |
| `--runs A,B` | both | comma-separated run names (default: every run of the spec) |
| `--timeout SECONDS` | build-runs | what the Megatron worker may take (default 600) |
| `--store FILE` | register-runs | the store the runs are registered in; required |
| `--runs-folder DIR` | register-runs | the folder `build-runs` wrote; required |

`build-runs` prints its summary as JSON; `register-runs` prints a JSON list, one
object for each segment, which is what `waterwhite register-run` printed for it. An
error prints one line on standard error and exits 1.

```python
from replica.runs import build_runs

summary = build_runs("runs-folder", "fixture-build-folder")   # or no second argument
```

**What it needs.** A built, tokenized fixture (`build-fixture` without
`--no-tokenize`; `--fixture-build` names its folder). With none named, the fixture is
built first into `DIR/.work/fixture-build`. `WATERWHITE_MEGATRON_ENV` names the
Megatron environment, as for tokenization. `--out` must be new or empty. The segments
are those of the spec's `[runs]`, in the spec's order (E with `E/seg-1` and
`E/seg-2`, then E-rr, D, Y and K, the cyclic run); `--runs E,K` limits the work to
the runs named.

**What it writes**, for a run folder `DIR/<run>/`:

```
<segment, slash replaced by dash>.run.json    the run record, format waterwhite.run-record/1
<segment...>.steps.json                       the step record
<segment...>.cache/                           the data-order cache folder Megatron wrote, as written
shards/<shard>.bin, .idx                      the shard files the record's prefixes name
```

The rows of the checkpoints are not written in `DIR`: when the fixture build named by
`--fixture-build` holds a `built-from.jsonl` of its own, they are appended to that file
(section 16).

A shard prefix in a record is relative to the record's folder (`shards/shard-1`). The
shards are copied into the run folder first and Megatron is handed those copies, so
the paths its cache descriptions carry are the paths the record names.
`DIR/.work/` holds scratch (the plan and the log of the worker, and the fixture build
when none was named) and is not part of the output.

**A cache folder holds absolute paths.** Each `description.txt` file Megatron writes
into a kept cache folder names the dataset it describes by absolute path: the shard
prefix as handed to Megatron (`DIR/<run>/shards/<shard>`), and the tokenizer's vocabulary and
merges files under the Megatron environment's own folders. The path is part of the
text the leading hash of each file name is made from (Megatron's own
`unique_description_hash`), so a copy of the folder to another place still names the
old paths. A cache folder is therefore never committed; it lies in a scratch or
output folder of the machine that made it. The run record does not depend on it: it names a shard by a prefix relative
to its own folder.

**How the cache is made.** One subprocess of the Megatron environment
(`megatron_runs_worker.py`, started with the environment's interpreter and `-I`, the
hub off) builds each segment's training datasets with Megatron's own
`BlendedMegatronDatasetBuilder` over `GPTDataset`, with a `GPTDatasetConfig` made the
way `pretrain_gpt.py` makes one and `path_to_cache` set to the segment's new, empty
cache folder, and the sample count a trainer asks for (train iterations times the
global batch size). Megatron writes the five kinds of index array and the
`description.txt` files itself. The worker then reads the kept `.npy` files back and
digests each array (`waterwhite.ids.array_digest`, plan 3.5); a file is tied to its
dataset through the description hash Megatron puts at the start of its name, and a
cache folder that holds anything else is refused. The replica process itself imports
neither Megatron nor torch.

**The shuffle digests of the cyclic run** (plan 3.3 "Training", 3.5; docs/interfaces.md
8.1). The wrapper computes them in the run's own environment, from the launch's real
sizes: the length of the dataset Megatron built (not the spec's table of sizes), the
micro batch size, the data-parallel size, the data-sharding flag and the launch's
consumed count and steps. For every launch, every epoch the launch touches (from
`consumed_at_start // n` to `(consumed_at_start + steps * global_batch - 1) // n`,
`n` being the dataset length less its remainder modulo `micro_batch * data_parallel_size`)
and every rank it draws one whole order, with the replica's own call to torch: a
`torch.Generator` seeded with the epoch number and `torch.randperm` over the rank's
bucket (with sharding on, `start + randperm(bucket)`; with it off, the rank's stride
of one permutation of the whole dataset). That is the draw Megatron's cyclic sampler
makes; the sampler class is not run. The digest is `waterwhite.ids.order_digest` of
the whole epoch's order, whatever part of it the launch read, with its length as
`count`; a resume inside an epoch therefore records the same digest for that epoch
and rank as the launch before it. No function of the join is called: the digests the
join computes are checked against these, and a shared function could not make that
check fail. A sequential run records no shuffle digests.

**Values the spec does not state, and where each comes from.** `data_sharding` for
the sequential runs is read from Megatron's own argument table (its default); the
cyclic run's is the spec's. `settings.sampler_class` is the class of that name in the
installed `megatron.training.datasets.data_samplers`, and must be the one the spec's
`[runs_common]` names. `versions.release_tag` is read from the Megatron-LM checkout of
the environment (the one tag that names its commit; no `git` command runs); it must
be the tag the spec pins (plan F6), the installed `megatron-core` must carry that
tag's version and be built from that commit, or the wrapper stops. `megatron_core` and
`torch` are the installed versions. A segment that lists no launches has one launch
over all its steps, at the consumed count it states, with the run's data-parallel
size.

**The step record** lists every completed step of the segment with the global batch
size, a step the spec's step record lowers with its `last_sample_tokens_read`, and the
segment's checkpoints (docs/interfaces.md 8.2).

**Registration.** `register-runs --store FILE --runs-folder DIR` (or
`replica.register_runs.register_runs`) starts one `waterwhite register-run` per
segment, as a subprocess, in the spec's run order, so that a parent checkpoint is
registered before the segment that starts from it. Each call passes the run record,
the step record, the kept cache folder, the mixture's name and, for a segment that
starts from a checkpoint, that checkpoint:

```
waterwhite register-run --store FILE --run-record DIR/<run>/<segment>.run.json
    --steps DIR/<run>/<segment>.steps.json --cache DIR/<run>/<segment>.cache
    --mixture <mixture name> [--parent-checkpoint <name>]
```

The store, and the mixtures under the spec's names, are the caller's (the table's
stores register them, and list the joins of their runs as verified, section 5). The
interpreter that runs `register-runs` starts the tool's command (`python -m
waterwhite.cli`), so for the cyclic run, whose shuffle the join computes by calling
torch, it must be one that has torch. A command that exits other than 0 stops the
registration with its own words, so a segment after it is not registered. Where the
tool's command does not exist, `register_run_available` says so and nothing is
started. The Megatron test `test_register_run_takes_every_segment_of_the_runs`
registers the runs of a `main` store (E, D, Y and K) and of an `rr` store (E-rr),
each built with `build_table_store(..., tokenize=True)` and `build_runs`, and asserts
for every segment the route "cache files", no cause, and the untraced steps the
spec's step record lowers.

**What it refuses.** A runs folder that is not new or empty; a run the spec does not
name; a fixture build with no `pipeline/` folder or a missing shard file; an
environment whose checkout is not the pinned release or whose install is not built
from it; a sampler class other than the spec's; a launch whose sizes the cyclic
sampler cannot take; a cache folder that was not empty; a worker that fails or passes
its time limit.

**Compared with the oracle.** `oracle/fixture_runs.py` writes the record of the same
segment from the same fixture build (through `oracle/run_record.py`'s code), on the grading side. A test compares the two
field by field; the one field that differs by construction is a shard prefix, which
is relative to the record's own folder.

## 14. The run over a generated set

`probe-run` makes one run over the shards a generated-set build wrote and puts it into
the store the set was built into. It is the run of the measurement of plan 5.6 (plan 3.3
"Training", 5.4). No model is trained (plan 5.3, limit L4).

```
python -m replica probe-run --set-build DIR --store FILE --out DIR
        [--layout partitioned|round-robin] [--steps N] [--spec FILE] [--timeout S]
```

```python
from replica.probe_run import probe_run

printed = probe_run("set-build-folder", "store.sqlite", "probe-folder")
```

| Flag | Meaning |
|---|---|
| `--set-build DIR` | a folder `build-set` wrote, tokenized, with capture on; the run reads `pipeline/<layout>/shards/` of it |
| `--store FILE` | the store the set was built into; it must list the sequential join (below) |
| `--out DIR` | where the run's records go; new or empty |
| `--layout L` | the layout whose shards are read (default `round-robin`) |
| `--steps N` | the steps of the run (default: the set's one pass in whole steps, below) |
| `--spec FILE` | the fixture spec that gives the settings, the pinned release and the tokenizer (default: the repository's); the set must have been tokenized with that tokenizer |
| `--timeout S` | what the Megatron worker may take (default 600; a set of 10,000 records needs the figure raised if the default is not enough) |

**The store must list the sequential join, and is never written to list it.** The first
thing the command does is read the store's list of verified joins (plan F6) and look for
the row (the spec's pinned release tag, sampler type `single`, the spec's class for it).
With no such row it stops with `RunError` and has made nothing: no output folder, no
mixture, no run. A store lists its joins when it is made (D-37, D-38): a build makes
that store with `build_captured_set(..., verified_joins=...)`, or on the command line
with `build-set --capture --store FILE --tokenizer-spec FILE --list-join single`, which
reads the row from the `[runs]` of the spec file. A store made by `waterwhite init`
lists the joins the tool ships (D-51) and passes this check with no flag; the replica's
own stores are made by the library and list what the build gives them.

**The run.** One segment, `probe/seg-1` of the run `probe`, with the settings of run E of
the spec (sequence length, micro and global batch size, seed, split, data-parallel size)
and a fresh start from sample 0 in one launch. The sampler is the sequential one.

- **The shards and their order.** Every shard of the layout in the set build, the names
  sorted with each run of digits taken as a number (`shard-2` before `shard-10`; a
  partitioned set's `shard-A`, `shard-Z2`, `shard-gen` in that order). The mixture, the
  worker and the run record all take this one list. A shard with one of its two files
  missing is refused.
- **The steps.** With no `--steps`, the whole steps of one pass over the set: for each
  shard `(tokens - 1) // sequence_length` samples (Megatron's dataset takes that many
  of a shard when it is asked for no sample count of the shard, as a blend with no
  weights asks), added over the shards and divided by the global batch size, rounded
  down. `--steps` may be fewer and never more. After the worker has run, the dataset
  Megatron built must hold at least the samples of those steps, or the run is refused.
- **One checkpoint**, `probe@<last step>`, at the last step, so that a question about a
  checkpoint can be put to the store afterwards. The step record states it; nothing is
  lowered.

**What it makes.** In order:

1. **The mixture** `probe/<layout>`, registered with `Store.register_mixture`: the shards
   in the run's order, each by its registered instance (found by the path its files lie
   at), each with the digest of its two files, as the mixture capture point of section 5
   does. The settings say the files of the layout's shard folder were selected.
2. **The run**, through the one launch path of section 13 (`replica.runs.make_segment_runs`,
   which `build-runs` uses too): the shard files are copied into `<out>/probe/shards/`,
   Megatron's own dataset code is launched in a subprocess of the Megatron environment
   into a new empty cache folder, and the run record and the step record are written
   (docs/interfaces.md 8.1, 8.2, 8.4).
3. **The registration**, `waterwhite register-run` as a subprocess (section 13), with the
   mixture, the run record, the step record and the cache folder.

The command imports nothing of `oracle/` and nothing of `waterwhite.join`, and writes the
store only through `Store.register_mixture` and `waterwhite register-run`. The replica
process imports neither Megatron nor torch.

**What it prints** (JSON):

```
{"segment": "probe/seg-1", "mixture": "probe/round-robin", "shards": <N>, "steps": <N>,
 "seconds": {"mixture": <S>, "run": <S>, "register": <S>},
 "register_run": { ...what waterwhite register-run printed... }}
```

`shards` is the number of shards. The three keys of `seconds` are each one wall-clock
reading of one run, in seconds:

| Key | From | To |
|---|---|---|
| `mixture` | the first shard is looked up in the store | the snapshot is registered; the digests of the shard files are inside it |
| `run` | the worker is started | the run record and the step record are written; the digests of the two files of each shard in the record are inside it |
| `register` | `waterwhite register-run` is started | it has printed; the join, the stored step traces and the checkpoint are inside it |

**What it measures, and what it does not.** It gives the three times above, and
`register_run` gives the route, the steps and the traced steps the tool stored. It does
not measure the bytes of metadata per stored step trace, the query time or the
coverage-proof walk time of plan 5.6, nor peak memory or disk. The copy of the shard
files into the run folder is not in `run`. Nothing here is a median of paired runs
(plan 5.6 asks for three), and a figure is one run of one workload, never a claim beyond
it (W7). It was run on a store built with capture on; a store made by recovering a run
made without capture (section 6) was not tried. A second `probe-run` into the same store
registers the same segment name again; the replica does not check what the tool does
with it, and its tests do not make one.

**What it refuses.** A store that cannot be read or lists no sequential join of the
spec's release; a layout the set build does not hold, or one with no shard, or a shard
with a file missing; a store that holds no registered shard at the path of one of the
set's shards (or more than one), or whose registered digest differs from the files; a
`--steps` below 1 or above what the set gives; an output folder that is not new or
empty; the environment's refusals of section 13 (a checkout that is not the pinned
release, a sampler class other than the spec's, a worker that fails or passes its time
limit); a dataset smaller than the run's reads.

## 15. Repairing from a report

`python -m replica repair` stands in for the team's own builder when it acts on a
rebuild report (plan 4.1, 4.4). It reads the report that `waterwhite report` printed,
builds the repaired side, builds the reference side, and writes the sides file that
`waterwhite verify` reads (docs/interfaces.md sections 7.1 to 7.3 and 9.5). The same
work is one call, `replica.repair.repair(...)`, with the options of the command as
keyword arguments; it returns the summary the command prints.

```
python -m replica repair --report FILE --historical DIR --out DIR --store FILE
        (--spec FILE | --set DIR) [--layout partitioned|round-robin|both]
        [--reference-set DIR | --reference-withdrawn SOURCE] [--reference-out DIR]
        [--sides FILE] [--base DIR] [--tokenizer-spec FILE] [--no-tokenize]
        [--unseeded] [--shuffle-seed LAYOUT=N] [--jobs N] [--tokenizer-timeout S]
        [--generation-harness in-process|confined] [--historical-name-prefix P]
```

| Option | Meaning |
|---|---|
| `--report FILE` | the report, a JSON file of the form `waterwhite.report/1` |
| `--historical DIR` | the folder of the build the store's historical artifacts were made from |
| `--out DIR` | the folder of the repaired side; new or empty |
| `--store FILE` | the historical store: the successors are registered in it. A path that holds no store, or a file that is not a store, is refused before anything is built or created (below) |
| `--spec FILE` or `--set DIR` | what the historical build was made from: the fixture spec, or a generated set (with `--tokenizer-spec`, as for `build-set`, unless `--no-tokenize`) |
| `--layout` | the layouts to build (default: the ones the historical build made) |
| `--reference-out DIR` | the folder of the reference side; new or empty. With it, `--reference-set DIR` (a set) or `--reference-withdrawn SOURCE` (the fixture) |
| `--sides FILE` | write the sides file; a new file outside both build folders |
| `--unseeded`, `--shuffle-seed`, `--jobs`, `--tokenizer-timeout`, `--base` | as for `build-fixture` and `build-set`; both sides get the same values |
| `--no-tokenize` | stop before Megatron's preprocessing, on both sides (default: tokenize where the historical build did) |
| `--generation-harness` | default: the harness the historical build's own logs name (below) |
| `--historical-name-prefix P` | the name prefix the historical build registered its names under, if it used one |

### How the report is read

The report is read with the `json` module as a file, and the replica imports no
module of the tool's rebuild, coverage or answer code (docs/interfaces.md section 1).
It reads `form`, `artifacts` and `records` and nothing else: not `counts`, not the
lists `removed` of an artifact, not the restarts, and not `upstream_files` (the files
upstream of the artifacts, which the report names for the person who reads it: the
same report with that list as printed, with other content under it or without the key
builds the same repaired side). A file that cannot be read, whose
form is another, or whose entries lack a key or hold an action word the replica does
not know is refused. The words of an artifact the replica knows are keep, rewrite,
unresolved and "clean, place not stated" (a shard that reads the last is built by its
layout, like every shard of the repaired build, and is registered as `successor of <name>`;
nothing in the build depends on the word, since the replica builds a shard from the
records it keeps). The build is driven by the `action` of each entry of `records`
alone, so a report edited by hand drives the build as it was edited (the states
`r.descendant-kept` and `r.unrelated-paraphrase-retired` of fixture/table/README.md
section 7):

| Action of an entry | What the build is given |
|---|---|
| retire | `--leave-out ID` for the record's id |
| re-source | `--swap OLD=NEW`, OLD the id of `withdrawn_parent`, NEW the id of `new_parent` (below) |
| promote | nothing: it is the other half of a re-source |
| keep | nothing |
| unresolved | nothing |

A retire and a keep of one id in two entries is refused, naming both entries: a build
leaves a record out of every file at once, and an id that the report both retires and
keeps has no side to build. A record that the report lists in more than one place (its
row in a release and its document in a shard) is one id, so a wrong report is made by
editing every entry of the record: an edit of one of them is refused as a
contradiction and is no wrong side. An unresolved entry is no contradiction: the record named
by a retire entry elsewhere is left out of the whole build, so an unresolved entry for
the same record in another file (the fixture's text-only import of T, when the release
row of T is retired) changes nothing and the record is not in that file of the repaired
side either.

**A re-source is a swap.** The withdrawn parent and the new parent are two copies of
one text, and a dedup group selects both, so `--swap OLD=NEW` leaves OLD out and keeps
NEW at its own place (section 8): the release has B's copy of X where B's copy stood.
A re-source whose new parent is not selected beside the old one is a plain swap, NEW
at the place of OLD. One id re-sourced to two records is refused.

### How a record is told apart

The report names a record as `artifact:locator`. The replica gives records ids that
the tool never sees (plan Check 3), and finds the id through the files of the
historical build:

| Artifact name | File | Id of line n |
|---|---|---|
| `src/<KEY>` | `sources/<KEY>.jsonl` | `origin` of manifest row n |
| `gen/<NAME>` | `gen/<NAME>.jsonl` | the row's own `id` field (its `origin` is what the job used) |
| `import/<NAME>` | `imports/<NAME>.jsonl` | `origin` of manifest row n |
| any other name | `releases/<name>.jsonl` | `origin` of manifest row n |
| a shard (`E/shard-3`) | the tokenizer input of the shard, `pipeline/<layout>/tokenizer-input/<shard>.jsonl` | for document n: `origin` of the manifest row of the n-th line that gave tokens (the shard's index file says which lines gave none) |

A locator is `L<n>`, a line of a record file counted from 1, or `D<n>`, a document of a
shard counted from 0. A line or a document that the file does not have, a `D<n>` of an
artifact that is no shard of the build, an `L<n>` of a shard, a manifest row with no
single origin, an artifact whose file is not in the historical folder and a name the
historical prefix does not begin are refused: no id is guessed.

### The repaired side

The same build as the historical one, with the records above left out or swapped,
with capture on into `--store` and `--name-prefix` set to `successor of ` followed by
the historical name prefix. "The same build" is read from the historical folder
unless an option says otherwise: the layouts are those that have a folder under
`pipeline/` (`--layout` names others), shards are tokenized where the historical build
tokenized them (`--no-tokenize` stops before), and the harness is the one the harness
logs name (`--generation-harness` names another). Three defaults have no option and
come from the historical folder only: the releases are the files of `releases/`, the
imports are those of the spec whose file is in `imports/`, and the generation step is
captured where the historical folder holds harness logs. The names follow docs/interfaces.md section 9.2:

- a rewritten artifact is registered as `successor of <name>`: `successor of R1`,
  `successor of E/shard-3`, `successor of E/mix-2`, `successor of set/shard-B`;
- an artifact the build leaves as it was and whose history is the same is the same
  instance as before and gains the second name (DECISIONS.md D-14): a source, a
  generation output, a release or an import of unchanged content. A shard is not: the
  shard's two input files are registered as found files under their own names, which
  carry the prefix, and a name is part of a found file's instance id, so a shard the
  repair leaves as it was is a new instance under its second name with the same
  bytes, and so is a mixture that snapshots it. The summary says which kept artifacts
  are the same instance (`kept`); the byte equality of a kept shard is read from the
  files, by the digests the sides file names;
- member names and the names of harness jobs carry no prefix.

The harness must be the one the historical build used: the store refuses a job name
whose stored log differs, and the harness word is in the log (DECISIONS.md D-30). The
driver reads it from the historical build's harness logs unless
`--generation-harness` names it; a build with no log is in process. What the store or
the tool's wrappers refuse (for example a second repair of one store with another
report: the name `successor of R1` is taken by an instance of another history) is a
refused repair, in one line.

### The reference side

Built with capture off and no store, from sources that never held the withdrawn
content, with the layouts, seeds and options of the repaired side:

- a generated set: from the folder the set's generator wrote with `--clean`
  (`--reference-set`);
- the fixture: from the spec, with the records of the withdrawn source and every
  generation output made from them, directly or through other outputs, left out
  (`--reference-withdrawn A`). The withdrawn source is the caller's argument and the
  derived records come from the spec's lists of what each output used, never from the
  report, so a wrong report cannot make the two sides agree.

A seeded repaired side equals the reference side file by file: the files under
`releases/`, `pipeline/` and `mixtures/` of the two sides are the same files with the
same bytes (section 8). The seeds are not read from the historical folder: a caller
that built the historical side unseeded or with `--shuffle-seed` gives the same
options to `repair`.

### The sides file

`--sides FILE` writes `waterwhite.sides/1` (docs/interfaces.md section 7.2): for every
artifact of the report whose action is keep, rewrite or "clean, place not stated" and
whose kind is jsonl or shard, the file (a release or an import) or the shard prefix that the repaired side
holds, and the one the reference side holds, relative to the folder of the sides file.
A shard is found by its layout: the store name begins with the layout's name prefix
(`E/`, `E-rr/`, `set/`, `set-rr/`), and the rest is the shard's file name. A mixture
has no file of its own and is not named, and an artifact a side did not write (a shard
no record reaches is not written) is left out of that side. After the artifacts of the
report, the file also names each shard the report lists under a layout's
`groups.placements[NAME].landing` (a shard the historical store does not hold, which the
copies placed there make) that the side holds, found by its layout in the same way; a
landing shard a side did not write is left out of that side, and one that lies in no layout
is refused (below). Such a shard has no entry in the report's artifacts, and the report names it
under the landing, so `verify` compares it (docs/interfaces.md 7.3). A shard entry of kind
"shard" that lies in no layout of the build (its store name begins with no layout's
prefix) is refused as an input error, in one line, naming the shard and the layout prefixes,
and so is a shard the report names under a layout's landing that lies in no layout (a
landing shard is a shard the report names). The refusal comes before either side is built: it reads the report and the layouts and
nothing a build makes. A refused repair registers no successor and writes no folder. The
refusal is made when `--sides` is given, since the sides file is where a shard is looked
up by its layout. The sides file's list of names is not what `verify` reads to know which
shards a build wrote (docs/interfaces.md 7.3): it reads the shards the repaired store
registers as `successor of <name>` and the shard files of the reference build.

### What it prints

One JSON object: `report`; `left_out` and `swapped` (the build's parameters);
`unresolved` and `promoted` (the counts of those entries); `harness`; `name_prefix`;
`successors` (each historical name beside the name its successor is registered
under); `successors_not_registered` (rewritten artifacts the build registered no
successor for, as a shard that no record reaches); `kept` (for each artifact the report
keeps, the name it is registered under and whether that is the instance the report
holds); `repaired` and `reference` (the summaries of the two builds, section 9);
`sides` (the path of the sides file). Exit codes as for the other commands: 0 built, 1
refused, 2 a wrong command line. Everything the build would refuse is refused before
the store or a folder is touched (a report that cannot be read or has an unmappable
record, a shard that lies in no layout when a sides file is asked for, a folder that is not new, a missing environment, and a `--store` path that
holds no store: the successors are registered in the historical store, and opening a
path would create an empty one, so the command refuses it as an input error and
creates neither the store nor a folder for it; a `--store` path that holds a file that
is not a store, as arbitrary bytes or a database with no schema version of the tool's
store, is the same input error, in one line, found by reading the file through a
connection that cannot write, so the file keeps its bytes and no sibling file or folder
is made), so such a refusal leaves no store behind. What the store refuses while the repaired side is being registered (for
example a successor name that another history holds) leaves the repaired folder
part-written and the names registered so far in the store.

### What it does not do

It does not compute a report, decide what is in scope or judge the result: it builds
what the report says and never reads the plant list. It registers no run, so a
successor mixture is registered and a run segment is not; the restart and the caches
of a run are the tool's `verify`. It takes no
account of an `unresolved` entry. It writes nothing outside its two output folders and
the store, and the sides file.

## 16. The file-level manifest

A build made with capture on records, while it runs, which artifacts' files went into
which. The file is `built-from.jsonl` at the root of the build folder; it is not the
`<file>.manifest.jsonl` that sits beside each record file (section 2), which holds a
line, a handle and an origin per record and is test evidence of the replica. The file
of this section is a reference point for the baselines of plan 5.5 Check 10 (the
subject `manifest` of the table's README), and nothing in the tool reads it.

**Rows.** One JSON line per artifact, with exactly three keys:

```
{"name": "R1", "files": ["releases/R1.jsonl"], "built_from": ["src/A", "gen/P", "gen/T", "gen/S", "src/Z"]}
```

`name` is the name the build registers the artifact under (section 5; a build with a
name prefix, as the repaired side of a repair, carries the prefix, so its release is
`successor of R1`). `files` are paths relative to the build folder. `built_from`
names the artifacts whose files went into the artifact. There is nothing below a file:
no record, line, ordinal, step, label, count or digest.

| Artifact | Files | `built_from` |
|---|---|---|
| a source file, `src/<KEY>` | `sources/<KEY>.jsonl` | none |
| a generation output, `gen/<NAME>` | `gen/<NAME>.jsonl` | the files of the records its jobs were handed (not only those a job states it used) |
| a release | `releases/<NAME>.jsonl` | the files of the copies kept; the file of a copy the dedup dropped did not go into the release and is not listed |
| the text-only import | `imports/<NAME>.jsonl` | the file of the member its file family gives, not the file its text was made from |
| a chunk file, `<shard>.input` | its tokenizer input | the files of the records its lines hold; the line the spec writes into a chunk has no file |
| a shard | its `.bin` and `.idx` | its chunk file |
| a mixture | the shard files in its folder | its shards, in the trainer's order |
| a checkpoint (below) | its segment's run record | the shards the segment was handed, in the trainer's order, then the segment's parent checkpoint |

A chunk's own records decide its list: in the partitioned layout that is the files of
the shard's own groups, in the round-robin layout the files of the records the shuffle
dealt to that chunk (a record the sanitizer dropped is not in the chunk file, so its file
is not listed for it). In a generated set the records of a chunk were cut from a release,
and the chunk names the set's own files they came from (the source files and generation
outputs of the set), never the release. The chunk as the sanitizer reads it has no row. A shard that no line
reaches, and every artifact of a build made without tokenization after the sources, the
generation outputs, the releases and the import, has no row, as it has no registration. A
generation output of a build that captures no generation step has no row, and a file
without a row names nothing.

**Written while the build runs, from the build only.** The recorder
(`replica/built_from.py`) is created with the build folder and appends each row when
the build has made the artifact, at the build's own capture points. It is handed the names
the build chose, the paths it wrote and the files of the records it read, and nothing from
the store or the tool's capture: it imports neither. The rows hold no absolute path and no
clock, so two seeded builds write the same bytes.

**Capture off writes nothing.** A build made with capture off makes no recorder and
writes no file, at the build and after the recovery of its store, and a reader reports the
manifest absent. The reference side of a repair is such a build; the repaired side is a
build with capture on and writes the file, under the successor names.

**The checkpoints of a run.** A run is made after the build, by the run wrapper
(section 13), in a folder of its own. `build-runs` appends the rows of the checkpoints of
the segments it made to the `built-from.jsonl` of the build folder it was handed, in the
spec's order, when that file exists (a build with capture on). A checkpoint's `files` is
one path, relative to the build folder, to the run record of its segment in the runs
folder; it begins with `..` when the runs folder lies beside the build folder. No manifest
file is written in the runs folder. A fixture build with no manifest, and the fixture
`build-runs` makes for itself, get no row. The rows name shards by their registered names,
so one walk reads the whole manifest of the build. Making the same runs again from the same
build leaves the rows as they are; a checkpoint already named with another `built_from`
list is refused, and nothing is appended.

**Not in it.** The two scratch stores (section 7) are not made by the build and write no
manifest. A generated set built with capture on (section 4) writes the file too, under the set's
names: its chunk rows name the set's own files, so a shard of a set reaches a source file
only by its own records and never through the release its lines were cut from. The files of
a run that was not made by `build-runs` have no rows.

## 17. The cell runner

`python -m replica.cells` (module `replica/cells.py`) runs the measurement of plan 5.6 (W7): a
grid of cells, each built at two worker settings with capture off and on, and one record per
run. It calls `python -m replica` and `waterwhite.cli` as programs, unchanged; a figure the
tools do not print is not made by the runner (it is returned as an open point instead).

```
python -m replica.cells grid
python -m replica.cells run   --out DIR --host-type TYPE --tree ID [--at-once N] [--window-file FILE]
        [--sampler] [--before-run COMMAND] [--after-run COMMAND] [--cell NAME ...] [--confined]
        [--only-capture] [--only-jobs 1|4] [--tokenizer-spec FILE] [--python PY] [--host-role ROLE]
python -m replica.cells rerun --out DIR --run ID --host-type TYPE --tree ID [the same options]
python -m replica.cells level --records DIR...
```

### The cells and the runs

A cell is the record count, the record length, the derivations per source record, the number
of sources and a layout. The grid is 20 cells: at 10k records the record length {500, 2000,
8000} x derivations {0, 1, 2} x sources {4, 10} in the partitioned layout (18 cells), the
middle cell (2000 characters, 1 derivation, 4 sources) also in the round-robin layout, and
the middle cell at 100k records. `fixture/plant/generate.py` makes each set, with one fixed
`--seed` and `--dedup on`. Each cell has 12 runs: workers {1, 4} x capture {off, on} x three
paired runs (a pair is one off run and one on run at the same workers). The plan is 240 runs:
216 + 12 round-robin + 12 for 100k. `grid` prints the counts, derived from the grid.

### What a worker is

The worker setting is `python -m replica build-set --jobs 1` or `--jobs 4`: the number of
tokenizer runs at a time. It is not Megatron's own workers value. Every record also states
Megatron's workers value, read from the run's settings: the `[megatron]` table's `workers`
of the tokenizer spec the build is handed (`--tokenizer-spec`), which the build reads as
`tokenizer.workers` (`replica.plan.load_tokenizer_settings`, 1 in the fixture spec). The
record names that place in `megatron_workers_from`.

### Harness

Every capture-on run of the 240 uses the in-process harness (`--generation-harness
in-process`), and its record says so in the word "in-process" and states that an in-process
log establishes no bound. The confined harness (`--confined`) is measured once, for the middle
cell: both worker settings, three runs each, a separate figure labelled as such and outside
the 240.

### Matched pairs

The two halves of a pair must match in input, order and tokenizer: the sha256 of every file
of the generated input folder, the listing of the shard names in the order the build wrote
them (digit runs read as numbers), and the sha256 of the tokenizer spec file. Each run's
record holds the three values. When both halves have finished, equal values are written to
`pairs/<pair id>.json`; values that differ are refused: no pair file, the pair and the names
that differ are listed in `refused_pairs.json` and in the result of the batch.

### Figures of one run

A capture-off run has no store: it has throughput, peak memory, the disk of the build folder
and the wall time. Query time, proof-walk time, metadata bytes, repair bytes and the store's
size exist for capture-on runs only and read "not applicable" in an off run, never zero.
Each figure sits beside its workload (the cell's parameters) and the machine.

| Figure | How it is taken |
|---|---|
| throughput | records per second of the wall time of `build-set`; the median over the three pairs, for off and for on (`median_throughput`) |
| wall time | the `build-set` command, run with no sampler; a record with the sampler says `sampler: true` and has its peak memory, all others read "not measured" |
| peak memory | the sum of the resident sizes of the whole process tree, sampled every 0.1 s from /proc, the greatest sum, with the largest single process beside it |
| disk | the bytes of the build folder, and of the store file |
| metadata bytes | per record (rows of `occurrence`), per parent link (rows of `edge_input`) and per stored step trace (rows of `step_trace`, whose bytes include `trace_sample` and `trace_row`): the pages of each b-tree and of its indexes, summed from `dbstat` on a copy of the store file, kept in the run's folder; the store itself is not opened |
| query time | `waterwhite query withdraw A --artifact probe@N`, N the steps `probe-run` printed (D-40) |
| proof-walk time | `waterwhite query withdraw A` over the whole store with no `--artifact`: the listing, which makes the proof of every artifact it answers (docs/cli.md "query withdraw", docs/coverage.md section 3 names the library call, no command of its own) |
| repair bytes | `waterwhite report withdraw A`, then `python -m replica repair`, on a copy of the store: the bytes of the historical build and of the store as it stood (read), and of the repaired build and the store's growth (rewritten) |
| capture wiring | one static figure, `capture_wiring.json`: the code lines of `StoreCapture.generate` and `StoreCapture.record_generation`, with their line numbers, and the two places in `replica/build.py` that call them |

In the first measurement the bytes of a build folder and of a repair differed by a few bytes between runs of one setting, and the records do not say why. The wall times, and the figures made from them, can depend on what else the machine runs. When the caller supplies a window file, the runner holds it during the batch. The first measurement used no window file. The run records describe overlap among the runner's own builds. `docs/measurements.md` gives the conditions of the first measurement and the spread of its bytes.

### Machine

Every record names the machine: the role ("server"), the type as `--host-type` gives it, the cores, the memory
and the operating system, read from /proc and /etc at the start of the batch, and the tree id
the code was taken from. This departs from risk RK3 of the plan (a laptop workload only): the
project owner decided on 17 Nov 2026 to measure on the server (DECISIONS.md, the entry on
the machine).

The cores are those the operating system shows. On a virtual machine with shared virtual CPUs they are not dedicated cores, and the type string the caller gives is the place to say so.

### Runs at once, resume, failures

`--at-once N` (default 1) runs N runs at once, each in its own process and its own work folder
(`work/<run id>`); both halves of a pair run at the same level. Every record states how many
runs ran beside it (at its start and the maximum during it). A stopped batch resumes: a run
whose record has exit 0 is not run again. A failed run stays in the record with its exit,
its message and the commands it had run, and is never run again silently; `rerun --run ID`
is the explicit command (the earlier record is kept under `records/earlier/`).

`--window-file FILE` is held while a batch runs: the runner creates it when the timed runs
start and removes it at the end, also after a failure or an interrupt. A file that exists
already is refused. `--before-run COMMAND` is called before each run (each group of `--at-once`
runs); its exit status and the first 200 characters of its output are the run's `hold` field;
status 0 means the box is held for cells and the run may start, any other status stops the
batch cleanly (`held/<run id>.json` says which runs did not start; the batch resumes from
there). `--after-run COMMAND` is called after each run or group, also after a failure or an
interrupt. Both calls are logged in `hooks.jsonl`. The commands are the caller's.

### The level test

`level --records DIR...` reads the records of a cell run alone three times and of the same cell
at rising levels (`--at-once` 2, 3, 4, 6, 8, 12, 16), each level in a batch folder of its own.
For `--jobs 1` and `--jobs 4` separately it reports the highest level at which throughput,
query time and walk time all stay inside the band the alone runs span (minimum to maximum,
both included); bytes, counts and line counts are not in the band. A run of a short last group
that never had N runs beside it (its own `beside` maximum says so) is left out and named. With fewer than three alone
runs for either setting it answers nothing. A level report is not a 5.6 figure: its output is
labelled "a measurement for the owner's rule, never a 5.6 figure". The heavier cells (8000 characters, 10 sources, 100k) are
named as not covered unless the level was run at that cell.
