# The oracle and the training stand-in

`oracle/` is test infrastructure. It is not part of the tool and the tool never
imports it. It serves plan 3.5 (the test-time oracle), plan 5.3 (what stands in
for training), plan 5.1 (the fixture's runs and the seed search that fixes its
read positions), the framework assumptions F1 to F6 and F9, and Checks 1, 2 and
8 of plan 5.5.

It does three things.

1. **Oracle.** It runs Megatron's own dataset and sampler code over tokenized
   shards and writes down what each completed step reads. Check 1 compares the
   tool's exposure join with this log.
2. **Training stand-in.** No model is trained in the first build (plan 5.3, limit
   L4). So the oracle also writes the run record a training wrapper would write
   while a run's index caches exist, and the step record an operator supplies when
   a run is registered (docs/interfaces.md 8.1, 8.2).
3. **The fixture's runs.** One command reads the fixture spec and a built
   fixture and does both jobs for every run the spec names ("Fixture runs").
   Another searches the runs' random seeds for the read positions the spec asks
   for ("Seed search").

It runs Megatron's code and copies none of it (plan 3.5). It imports nothing from
`waterwhite` (plan W5): an expected answer must not come from the tool's code.
Where the run record needs a digest the tool also computes, `oracle/digests.py`
computes it again from the written definition, and a test compares the two.

## What you need

- The Megatron environment that `tools/setup_megatron_env.sh` builds. Set
  `WATERWHITE_MEGATRON_ENV` to its folder. It holds `venv/`, a Megatron-LM checkout
  at tag `core_v0.20.0` under `src/Megatron-LM`, Megatron-Bridge's source under
  `src/Megatron-Bridge` (not installed), and `hf_home/` with the GPT-2 tokenizer
  files.
- `HF_HOME` set to `$WATERWHITE_MEGATRON_ENV/hf_home` and `HF_HUB_OFFLINE=1`. The
  oracle refuses to import Megatron without both. It uses no network.

Every tool but the last two of the table below is one script that takes one JSON
config file:

```
$WATERWHITE_MEGATRON_ENV/venv/bin/python -I oracle/<tool>.py <config.json>
```

`fixture_runs.py` and `seed_search.py` take options instead; their sections give
the command lines.

`-I` keeps the interpreter isolated; each script puts the repository root on its
own import path. A script prints its result as JSON on standard output and exits
0 (`positions.py` prints text and writes its JSON to a file). Megatron's own log
lines and warnings go to standard error. A failure the
oracle names prints `{"error": <class name>, "detail": ...}` on standard error
and exits 2, with nothing on standard output.

One run takes about 6 to 10 seconds on a laptop, nearly all of it the import of
torch and Megatron. `oracle/positions.py` and `oracle/seed_search.py
--check-runs` import neither and run with any Python 3.12.

| Script | What it does |
|---|---|
| `build_index.py` | builds a segment's index and writes the digests of its arrays |
| `consume.py` | the consumer: logs what each step reads |
| `run_record.py` | writes the run record, and the step record |
| `replay_index.py` | rebuilds the index and compares digests (Check 2) |
| `f1_probe.py` | checks one document per input line, and the end-of-document convention |
| `positions.py` | says which records each step read, and flags positions |
| `run_segments.py` | runs several segments from one file: everything above per segment |
| `fixture_runs.py` | runs every run segment of the fixture spec over a built fixture |
| `seed_search.py` | searches the runs' random seeds for the spec's read positions; or checks the spec's conditions on written fixture runs |

Modules that are not entry scripts: `megatron_data.py` is the one module that
imports Megatron; `stepping.py` walks the steps; `pipeline.py` holds what the
scripts do, as functions; `fixture_spec.py` reads the fixture spec;
`log_rows.py` groups log rows into samples and cuts a sample;
`search_conditions.py` holds the spec's numbered conditions as predicates;
`search_evidence.py` reads a built fixture and written runs for them.

## Rules every tool keeps

- **A cache folder is new or empty.** Every tool that builds an index builds it
  into the `cache_dir` of its config and refuses a folder that holds anything.
  Megatron loads index files it finds without checking them against the data: its
  cache key holds paths and settings, not content. An old file would be read back
  as if it had just been built.
- **No output is overwritten.** An output path that exists is refused.
- **Paths.** A path in a config that starts with `$WATERWHITE_MEGATRON_ENV` has
  that part replaced by the variable's value. Any other relative path is relative
  to the config file's folder. No output file holds an absolute path.
- **Unknown keys are errors.** A misspelt key cannot pass as a default.
- **Nothing is guessed.** A failure stops the run with a named error.

## The segment config

One run segment is one shard list, one set of data-order settings and one or more
launches. `build_index.py`, `consume.py` and `run_record.py` read this object;
`run_segments.py` reads a list of them.

```json
{
  "run": "E",
  "segment": "E/seg-2",
  "start_kind": "resume",
  "parent_checkpoint": "E@6",
  "shards": ["shards/shard-1_text_document", "shards/shard-2_text_document"],
  "tokenizer": {
    "type": "GPT2BPETokenizer",
    "vocab_file": "$WATERWHITE_MEGATRON_ENV/hf_home/hub/models--gpt2/snapshots/<revision>/vocab.json",
    "merges_file": "$WATERWHITE_MEGATRON_ENV/hf_home/hub/models--gpt2/snapshots/<revision>/merges.txt"
  },
  "settings": {
    "dataloader_type": "single",
    "data_sharding": true,
    "micro_batch_size": 4,
    "global_batch_size": 4,
    "sequence_length": 32,
    "random_seed": 1234,
    "split": "1,0,0",
    "train_iters": 12,
    "train_samples": null
  },
  "sampler_file": null,
  "launches": [
    {"world_size": 1, "data_parallel_size": 1, "consumed_at_start": 24, "step_at_start": 6, "steps": 6}
  ],
  "cache_dir": "cache/E-seg-2",
  "step_record": {
    "lowered": [{"step": 8, "last_sample_tokens_read": 16}],
    "checkpoints": [{"name": "E@9", "step": 9}]
  },
  "outputs": {
    "digests": "out/E-seg-2.digests.json",
    "run_record": "out/E-seg-2.run.json",
    "step_record": "out/E-seg-2.steps.json",
    "log": "out/E-seg-2.log.jsonl",
    "summary": "out/E-seg-2.summary.json"
  }
}
```

| Key | Meaning |
|---|---|
| `run`, `segment` | names, as docs/interfaces.md 9.2 gives them |
| `start_kind` | `fresh`, `resume` or `warm-start`. A fresh segment has no `parent_checkpoint`. A warm start must name one. A resume names the checkpoint it continues from, or none when the run registered no checkpoint at that step; its run record then holds null |
| `shards` | shard prefixes (the path without `.bin` and `.idx`), in the order the trainer is handed them. A prefix may appear once |
| `tokenizer` | `GPT2BPETokenizer` only, with both files named. Without the files Megatron tries to fetch them and writes under the home folder |
| `settings.dataloader_type` | `single` (the sequential sampler) or `cyclic` |
| `settings.data_sharding` | default true, Megatron-LM's default. Read by the cyclic sampler only; recorded for both |
| `settings.micro_batch_size`, `global_batch_size` | see "The step rule" |
| `settings.micro_batches_per_step` | optional. If given it must equal what the sizes give |
| `settings.sequence_length`, `random_seed`, `split` | passed to Megatron's dataset config. `random_seed` is the dataset's seed |
| `settings.train_iters`, `train_samples` | the requested number of samples is `train_samples` when set, else `train_iters` times `global_batch_size` |
| `settings.add_extra_token_to_sequence` | may be given as true. False is refused (see "Limits") |
| `sampler_file` | null: Megatron-LM's sampler classes. Else the path of Megatron-Bridge's `.../megatron/bridge/data/samplers.py`, which is then loaded from that path |
| `launches` | each start of the trainer inside the segment: its world size, its data-parallel size, the consumed-sample counter and the step it starts from, and how many steps it completes. Its steps are `step_at_start + 1` to `step_at_start + steps`. Two launches may not share a step |
| `cache_dir` | the new, empty folder Megatron writes its index files to. It is kept as written (docs/interfaces.md 8.4) |
| `step_record.lowered` | steps whose last sample was read only in part, with the number of tokens read. It must be lower than a sample's token count |
| `step_record.checkpoints` | checkpoint names with their steps, for the step record |
| `outputs` | where each file goes. A tool needs the ones it writes |
| `allow_sequential_restart` | optional, default false. See "When a pass ends inside a launch" |

## Index build: `build_index.py`

Needs `shards`, `tokenizer`, `settings` (`sequence_length`, `random_seed`, `split`,
`global_batch_size`, `train_iters`, optionally `train_samples`), `cache_dir` and
`outputs.digests`. Other keys of a segment config are ignored, so a full segment
config works.

It builds the training dataset with Megatron's builder,
`BlendedMegatronDatasetBuilder(GPTDataset, [requested, 0, 0], ..., config).build()`,
with `path_to_cache` set to `cache_dir` and the blend given without weights. Then
it digests each index array Megatron wrote.

Output (`waterwhite.oracle-index-digests/1`):

```json
{
  "format": "waterwhite.oracle-index-digests/1",
  "settings": {"sequence_length": 32, "add_extra_token_to_sequence": true,
               "random_seed": 1234, "split": "1,0,0", "requested_samples": 48},
  "blend": true,
  "dataset_samples": 48,
  "shards": [{"position": 0, "bin_sha256": "...", "idx_sha256": "...", "documents": 44, "samples": 36}],
  "cache_digests": [
    {"kind": "document_index", "shard": 0, "digest": "...", "dtype": "<i4", "shape": [44]},
    {"kind": "dataset_index", "shard": null, "digest": "...", "dtype": "<i2", "shape": [48]}
  ]
}
```

Rules, and where Megatron fixes them:

- **Five kinds of array.** Per shard: `document_index`, `sample_index`,
  `shuffle_index` (`GPTDataset._build_document_sample_shuffle_indices`,
  `megatron/core/datasets/gpt_dataset.py`). For the blend: `dataset_index`,
  `dataset_sample_index` (`BlendedDataset._build_indices`,
  `megatron/core/datasets/blended_dataset.py`).
- **When there is a blend.** With two or more prefixes the builder makes a
  `BlendedDataset`; with one prefix and no weights it returns the one `GPTDataset`
  and writes no blend files (`_build_blended_dataset_splits`,
  `megatron/core/datasets/blended_megatron_dataset_builder.py`). A one-shard
  segment therefore has three digests and no blend rows.
- **The length of the dataset.** In a blend without weights each shard is built
  for one pass over its documents, and the blend's length is the smaller of the
  requested samples and the sum of the shards' samples (same function).
- **Which file belongs to which shard.** Megatron starts each cache file's name
  with a hash of the dataset's description. The oracle takes the hash from the
  dataset object at each position of the shard list, so a digest is keyed by the
  shard's position and never by a guess from the file order. It also checks that
  the dataset at position p was built from prefix p.
- **The digest** is over the array's dtype string, its shape and its bytes in C
  order, as `ids.array_digest` defines it, not over the `.npy` file. Each array on
  disk must equal the array the dataset object holds, and the folder must hold
  exactly these files and one `description.txt` per dataset.
- **The split.** Only a split that gives every document to training (such as
  `1,0,0`) is accepted. The run record keys a cache digest by kind and shard and
  has no place for a split.

## Consumer: `consume.py`

Needs the whole segment config, with `outputs.log`; `outputs.summary` is optional.
It builds the index into `cache_dir`, then walks the launches.

For every sample of every completed step it asks Megatron's item getter for the
sample and writes one line per read the getter makes. The log has the format of
docs/interfaces.md 8.3:

```json
{"segment": "E/seg-2", "step": 7, "sample_pos": 1, "global_index": 25, "rank": 0, "shard_position": 1, "ordinal": 11, "token_start": 19, "token_end": 29}
```

How a row comes to exist. `GPTDataset.__getitem__` fetches a sample through
`_query_document_sample_shuffle_indices`, which calls
`IndexedDataset.get(document, offset=, length=)` once for each document the sample
touches (`megatron/core/datasets/indexed_dataset.py`). The oracle replaces that
method, on each shard's dataset object, with a function that calls the original
and notes three things: the document number it was called with (`ordinal`), the
offset (`token_start`), and the offset plus the number of tokens that came back
(`token_end`). The oracle computes no span by a rule of its own.

What the fields of a row mean:

- `segment`, `step`: the run segment and the completed step that read the row.
  Steps are numbered as the trainer numbers them inside the run.
- `sample_pos` is the place of the row's sample inside its step, counted from 0.
  A step reads `global_batch_size` samples, so its last sample has `sample_pos`
  `global_batch_size - 1`. The order is the one "The step rule" gives: micro-batch
  round, then rank, then place in the micro batch. It is a place in a step and
  an index into nothing.
- `global_index` is the index the sampler handed to the item getter: an index
  into the dataset the sampler runs over, which is the segment's blend, or the
  one `GPTDataset` when the segment has one shard. It is valid inside its
  segment only: each segment builds its own blend, so index 5 of two segments
  is two different samples. In a cyclic run an index comes back in every epoch.
- `rank` is the data-parallel rank whose sampler drew the sample.
- `shard_position` is the position, in the segment's shard list, of the dataset
  whose `get` was called. For a blend it must equal the `dataset_id` the item
  carries.
- `ordinal` is the number `IndexedDataset.get` is called with: the sequence
  number of a document in the shard's `.idx`, counted from 0. It is the document
  number `D<n>` of docs/interfaces.md 3, and document i is line i of the shard's
  tokenizer input when the F1 probe holds for the shard. It is counted per shard:
  the same ordinal in two shards names two documents. It is not a place in
  Megatron's shuffled document order.
- `token_start`, `token_end`: the range read inside that document, half-open.
- A sample's rows are consecutive in the log, in the order of the reads, and
  their ranges add up to `sequence_length + 1` tokens.

The check on every sample. The item getter returns `tokens` and `labels`; the
sample is `tokens` followed by the last label, `sequence_length + 1` tokens. The
oracle requires that this equals the concatenation of the logged reads, token for
token, and that no read runs past its document's length. If not, the run stops
with `ReadMismatch`. Padding or a read the oracle did not see would show up here.
The log is written as `<log>.partial` and takes its own name only when the whole
segment was read.

### The step rule

A step reads `global_batch_size` samples. Each of the n data-parallel ranks has
its own sampler, built with that rank's number and the size n. No process group
is created: Megatron-LM's loader factory reads global arguments and the process
groups of a distributed launch, so the oracle builds the sampler classes itself
with the arguments that factory passes (`build_pretraining_data_loader`,
`megatron/training/datasets/data_samplers.py`; plan F5).

In one step each rank draws k micro batches of m samples (m is
`micro_batch_size`), where k = `global_batch_size` / (m * n) and must divide.
Megatron fixes k in `ConstantNumMicroBatchesCalculator`
(`megatron/core/num_microbatches_calculator.py`), and its trainer adds
n * m * k to the consumed-sample counter after each step
(`megatron/training/training.py`). With `global_batch_size` = m * n this is one
micro batch per rank per step.

Ranks are run one after another in one process. A rank's reads depend on its own
sampler only.

`sample_pos`, the place of a sample inside its step, counts micro-batch round
first, then rank, then place in the micro batch:
`sample_pos = round * m * n + rank * m + place`. For the sequential sampler this
is the order of the global index.

Each sampler is handed to a torch `DataLoader` as its batch sampler, with no
worker processes, as Megatron's factories build theirs. `pin_memory` is left off;
it moves no data on a machine with no accelerator.

### The samplers

- **Sequential** (`dataloader_type` "single"): `MegatronPretrainingSampler`. It
  walks the indices from the consumed count to the dataset's length and gives
  rank r the slice `[r * m, (r + 1) * m)` of each run of m * n indices. So a step
  reads the next `global_batch_size` indices in a row, counted from the consumed
  count of its launch.
- **Cyclic** (`dataloader_type` "cyclic"): `MegatronPretrainingRandomSampler`. The
  epoch is the consumed count divided by the dataset's length (less a last
  partial batch), and the order is `torch.randperm` from a generator seeded with
  the epoch number. With `data_sharding` on, each rank permutes its own bucket;
  with it off, one permutation is dealt over the ranks.
- **Megatron-Bridge's classes** have the same names and live in
  `src/megatron/bridge/data/samplers.py` of its checkout. When `sampler_file`
  names that file it is executed from its path; Megatron-Bridge need not be
  installed, since the file imports only the standard library, numpy and torch.
  The run record then names the class as
  `megatron.bridge.data.samplers.<class>` and carries the checkout's tag as the
  Megatron-Bridge version.

### Starts, resumes and several launches

A launch starts wherever its `consumed_at_start` says: 0 for a fresh start or a
warm start that restarts the counter, the saved count for a resume. The count is
passed to the sampler as Megatron's trainer passes it. The step number comes
from `step_at_start` and never decides which samples are read (plan F4).

One segment may hold several launches, for example a cyclic run resumed inside
an epoch. They share the datasets built once for the segment; a real relaunch
would load the same cache files.

### When a pass ends inside a launch

A trainer wraps its loader so that a new pass starts when one ends (`cyclic_iter`
in `megatron/training/training.py`, and in Megatron-Bridge's `data/loaders.py`).
The oracle does the same: when a rank's iterator ends, it asks for a new one.
What the new pass yields is the sampler's own doing.

- The cyclic sampler moves to its next epoch. This is how a cyclic run crosses an
  epoch boundary.
- The sequential sampler would start again from the consumed count it was built
  with, not from 0, and read samples a second time. That happens when a launch
  runs more steps than its dataset holds from its consumed count on, which a
  blend without weights allows when the pool is smaller than the requested
  samples. The step-to-sample map of plan 3.5 does not cover it and the fixture
  avoids it, so **such a launch is refused** with `LaunchOutrunsDataset`, by
  every tool that walks a launch (the consumer, the run record, the driver, the
  fixture runs), after the index build and before a log or record is written.
  The message gives the micro batches the launch needs and the number the
  dataset gives. With `"allow_sequential_restart": true` in the segment config
  the launch is read as the sampler behaves, and the summary says
  `a_pass_ended_inside_a_launch: true`.

The count that decides it: from consumed count c, a dataset of N samples gives
each rank `(N - c) div (micro_batch_size * data_parallel_size)` micro batches
(`MegatronPretrainingSampler.__iter__` walks the indices c to N and yields once
per full run of that many); a launch needs its steps times the micro batches per
step. A launch that ends exactly at the dataset's end is read. Should the count
ever disagree with the class, the walk itself stops at the end of the pass with
the same error. A launch whose consumed count is at or past N is refused by
Megatron's class and keeps the name `SamplerRefused`.

### The summary

`waterwhite.oracle-summary/1`: the sampler class, the dataset's length, per launch
the micro batches per step and the passes of each rank, the samples read and rows
written, `read_orders`, and the cache digests. `read_orders` has one row per
launch, rank and pass of the sampler: the epoch the cyclic sampler named for the
pass (null for the sequential sampler), the first and the last step the pass was
read in, and the count and the digest of the indices that were read. It says what
a launch read; the run record's `shuffle_digests` say what the whole epoch's
order was.

## Run record and step record: `run_record.py`

Needs the whole segment config, with `outputs.run_record`; `outputs.step_record`
is optional. It builds the index into `cache_dir`, as it would exist while the
run trains, and writes the record of docs/interfaces.md 8.1.

| Field | Where it comes from |
|---|---|
| `settings` | the config. `sampler_class` is the module and name of the class that was driven. `batch_schedule` is null: the global batch size is fixed inside a segment (plan F2) |
| `versions.release_tag` | the tag that names the commit the Megatron-LM checkout is at, read from its `.git` folder; the commit when no single tag names it |
| `versions.megatron_core` | the installed package's version, as the package metadata gives it. An install from a checkout carries the commit after a plus sign, for example `0.20.0+685fb8b` |
| `versions.megatron_bridge` | null, or the tag (else commit) of the checkout that holds `sampler_file` |
| `versions.torch` | `torch.__version__` |
| `shards` | each prefix relative to the run record's folder, with the SHA-256 of its `.bin` and `.idx` |
| `launches` | the config's launches, with `first_step` and `last_step` |
| `cache_digests` | the index build's digests: kind, shard, digest |
| `shuffle_digests` | cyclic runs only; an empty list for a sequential run |

Before the release tag is written, the installed Megatron files the rules rest on
are compared with the checkout, file by file (`oracle/environment.py`,
`RULE_FILES`). If one differs the run stops: the tag would name code that did not
run.

**The order digests of a cyclic run** (plan 3.5, the ruling on the cyclic order;
DECISIONS.md D-16). `shuffle_digests` holds one row for every launch, every
epoch that launch touches and every rank: the digest of **the whole epoch's
order for that rank**, and `count`, the length of that whole order. It does not
matter which part of the epoch the launch read. So a resume inside an epoch
records, for that epoch, the same digest and count as the launch before it, and
a join can compare an epoch's shuffle and not a slice of it. The digest is
`ids.order_digest`'s definition: SHA-256 of the JSON list of the indices, in
order, with no spaces. A sequential run has no shuffle: `shuffle_digests` is an
empty list.

How the whole order is got, with nothing but the sampler class and without
reading the dataset. The launch's own steps are driven first; the passes of the
sampler say which epochs the launch touches. A sampler built at consumed count 0
(with the launch's micro batch size, data-parallel size, sharding flag and the
rank) is run for one pass: that is epoch 0, and the class then holds, in its
`consumed_samples`, the count one epoch adds. A sampler built at `epoch` times
that count is asked for one pass; the class must name that epoch itself
(`sampler.epoch`), and the pass is the epoch's whole order. The part the launch
drew must be one unbroken run of it. If either check fails the run stops and no
record is written.

Example: 60 samples, data-parallel size 2, micro batch 2, so 30 indices per rank
per epoch. A launch of steps 1 to 8 reads 16 indices per rank of epoch 0; a
resume for steps 9 to 20 reads the other 14 and the first 10 of epoch 1. The
record holds six rows, each with count 30: launch 0 with epoch 0, launch 1 with
epochs 0 and 1, for both ranks, and the two rows of epoch 0 for one rank are
equal. The 16, 14 and 10 are in the summary's `read_orders`.

The step record (docs/interfaces.md 8.2) lists every completed step of the
segment with `batch_size` (the global batch size). A step named under
`step_record.lowered` also carries `last_sample_tokens_read`. The oracle's log is
not changed by a lowering: it always holds every row the step read.

## Replay: `replay_index.py`

For Check 2. Needs `shards`, `tokenizer`, the data-order `settings` and a new
`cache_dir`. `compare_with` may name a run record or a digests file;
`outputs.digests` may name a file for the result.

It rebuilds the index from the settings and the seed and prints the digests. With
`compare_with` it adds, per array, the recorded digest, the replayed digest and
whether they are equal, plus the arrays only one side has, and `all_equal`. A
difference is a result, not a failure: the exit code stays 0.

## F1 probe: `f1_probe.py`

```json
{"input": "chunks/shard-1.jsonl", "shard": "shards/shard-1_text_document",
 "tokenizer": {"type": "GPT2BPETokenizer", "vocab_file": "...", "merges_file": "..."},
 "text_field": "text", "outputs": {"report": "f1/shard-1.json"}}
```

It reads the shard with Megatron's `IndexedDataset` and tokenizes each input line
with the tokenizer Megatron's `build_tokenizer` builds, by the call the
preprocessing script makes (`Encoder.encode` in `tools/preprocess_data.py`). It
compares document i with line i twice: followed by the tokenizer's
end-of-document token, and without it.

Output (`waterwhite.oracle-f1-probe/1`), the fields that carry the verdict:

- `one_document_per_line`: the line count, the sequence count and the document
  count are equal, each document is one sequence, and no line gives zero tokens.
- `input_order`: every document equals its own line under one convention.
- `eod.token_id`: the tokenizer's end-of-document token.
- `eod.in_span`: true when every document's span in the `.idx` ends with that
  token; false when none holds it; null when there is no single convention.
- `zero_token_lines`: lines that give no token. The preprocessing script writes no
  document for such a line, so documents and lines part there (plan F1).
- `same_documents_in_another_order`: set when the order fails; true points at
  preprocessing with more than one partition.
- `f1_holds`: `one_document_per_line` and `input_order`.

## Positions report: `positions.py`

A report for a person who reads a run's log: which records each step read. The
seed search of plan 5.1 itself is `seed_search.py`. It needs no Megatron.

```json
{"logs": ["out/E-seg-1.log.jsonl", "out/E-seg-2.log.jsonl"],
 "segments": {"E/seg-1": ["chunks/shard-1.jsonl", "chunks/shard-2.jsonl"],
              "E/seg-2": ["chunks/shard-1.jsonl", "chunks/shard-2.jsonl", "chunks/shard-3.jsonl"]},
 "id_field": "id", "outputs": {"report": "out/positions.json"}}
```

`segments` gives, per segment, the tokenizer-input file of each shard in the
order of the shard list. A record is named by the `id` of its input line: line i
(from 0) is document i. That holds when the F1 probe says so for the shard; this
report does not check it. The same file listed in two segments is one shard.

It prints, per step and sample, the records read with their token ranges, then
four flags. The JSON form (`waterwhite.oracle-positions/1`) goes to
`outputs.report`.

Words. With the extra token on, the last token of a sample's last row is the
sample's extra token: the first token of the next sample in sample-index order.
The body of a row is its range without that token.

| Flag | Rule |
|---|---|
| `split_across_samples` | a document with a row that starts after token 0, or with two bodies that do not overlap. `several_parts_read` says whether two parts are both in the logs |
| `read_through_extra_token_only` | a row that is the extra token alone. `document_read_otherwise` says whether any other row of these logs reads the document's body, and where first |
| `read_in_more_than_one_step` | a record with rows in two or more steps. `same_tokens_read_again` is true when a body token was read in two steps: a repeated draw |
| `same_ordinal_in_two_shards` | a document number read in two shards |

A document whose first part alone was read is not flagged as split: the log does
not hold document lengths.

## Driver: `run_segments.py`

```json
{"defaults": {"tokenizer": {"...": "..."}, "settings": {"sequence_length": 32, "split": "1,0,0"}},
 "summary": "out/driver.json",
 "segments": [{"...": "a segment config"}, {"...": "another"}]}
```

Each segment is a segment config. A key of a segment replaces the default;
`settings` and `tokenizer` are merged key by key. Every segment needs its own
`cache_dir` and all five outputs. For each segment the driver builds the index
once and writes the digests, the run record, the step record, the log and the
summary from that one build.

Checks made before anything is built:

- A fresh segment starts at consumed count 0 and step 0.
- A resume whose parent checkpoint is named in the file (in some segment's
  `step_record.checkpoints`) must start at that checkpoint's step and at the
  consumed count the parent segment had reached there. A resume on a new mixture
  is such a segment: new shard list, new cache folder, the old counter. A resume
  whose checkpoint is not in the file, or that names none, is taken as stated,
  and the result says that its counters were not checked.
- A warm start states its own counters and they are taken as given. On
  Megatron-Bridge's path they restart at 0 (plan F4).
- No two segments share a cache folder or an output path, and every cache folder
  is new or empty.

Inside a segment, the indices the consumer read are compared with the indices
the sampler classes gave when driven alone for the run record. A difference
stops the run before a record is written.

## Fixture runs: `fixture_runs.py`

One command for all runs of the hand-checked fixture (plan 5.1): run E with its
two segments, its round-robin variant E-rr, descendant D, branch Y and the
cyclic run K with its two launches.

```
$WATERWHITE_MEGATRON_ENV/venv/bin/python -I oracle/fixture_runs.py \
    --spec fixture/spec/fixture_spec.toml --fixture <built fixture> --out <new folder> \
    [--seed RUN=N ...] [--runs E,K] [--shards-from pipeline|mixtures] \
    [--tokenizer-snapshot DIR]
```

`--fixture` is the folder a fixture build of the replica wrote (docs/replica.md,
"The folder one build writes"); it is only read. `--out` must be new or empty.
The command prints the overview described below as JSON and exits 0; a refusal
prints `{"error": ..., "detail": ...}` on standard error and exits 2. All seven
segments take a few seconds after the import.

What it takes from the spec (`oracle/fixture_spec.py` reads and checks it):

| Spec | Use |
|---|---|
| `runs_common` | the split, the extra token (must be on), `megatron_bridge` (must be `"none"`: Megatron-LM's sampler classes are driven), and the sampler class each dataloader type must turn out to be |
| `runs.<R>` | sequence length, batch sizes, data-parallel size, `world_size` (the data-parallel size when absent), dataloader type, `data_sharding` (true when absent), `random_seed`, `steps`, `checkpoints` |
| `runs.<R>.same_settings_as` | the run has the named run's settings and seed; `layout` says which layout its shards are from |
| `runs.<R>.segments.<S>` | `steps` (first and last), `shards`, `start_kind`, `parent_checkpoint`, `mixture`, `train_iters`, `consumed_at_start`, and `launches` when the segment has more than one |
| `runs.<R>.step_record.<S>` | `steps`: the steps whose last sample is lowered, with `last_sample_tokens_read`; `checkpoints` |
| `mixtures` | each segment's `shards` must be the list its `mixture` has here |

The spec is refused when these disagree: segments that do not cover steps 1 to
`steps` in order, launches that do not cover their segment, a resume inside a
run that does not start at the consumed count the segment before it reached, a
resume inside a run that names a checkpoint the step record of the segment
before it does not register at its last step, a `checkpoints` list that is not
the steps of the step records' checkpoints, a step record row whose
`batch_size` is not the global batch size, an unknown key.

Rules:

- **Seeds.** Each run's `random_seed` is the spec's. `--seed RUN=N` replaces it
  for this call. A run with `same_settings_as` has the other run's seed, the
  override included; an override for such a run is refused.
- **Shard names.** `shard-1` is a shard of the partitioned layout:
  `pipeline/partitioned/shards/shard-1`. `E-rr/shard-1` is shard-1 of the
  round-robin layout: `pipeline/round-robin/shards/shard-1`. The position of a
  shard in a segment's list is its `shard_position`. With `--shards-from
  mixtures` the files are taken from `mixtures/<mixture>/<shard>` instead; the
  bytes, and so the index arrays and digests, are the same.
- **Launches.** A segment that lists none has one launch over all its steps at
  the consumed count the segment states. Run K's segment lists two.
- **Step record.** Written from the spec's table: every completed step with its
  batch size, the lowered steps with `last_sample_tokens_read`, and the
  checkpoints. The oracle's log is not changed by a lowering.
- **The cut of a lowered step** is the spec's rule applied to the oracle's own
  log: the step's last sample is the one at `sample_pos` `global_batch_size - 1`;
  its first `last_sample_tokens_read` tokens are covered; the row that holds the
  next token is cut there and no row after it is kept. The overview gives the
  shard position, ordinal and document token where the uncovered range starts,
  the rows of the covered part, and the sample's log lines.
- **Starts** are checked over all segments before anything is built, as the
  driver checks them. E/seg-2 must start where E@6 stood, and E-rr/seg-2 where
  E-rr@6 stood: the round-robin variant registers that one checkpoint, in the
  step record of E-rr/seg-1, so that its second segment resumes from a
  registered checkpoint (DECISIONS.md D-24). The hand-written table asks it
  nothing.
- **Paths.** A shard prefix in a run record is relative to the run record's
  folder. A path in `config.json` is relative to that file's folder, or starts
  with `$WATERWHITE_MEGATRON_ENV`. No file the oracle writes holds an absolute
  path; Megatron's own `description.txt` files in the cache folders do.
- **The tokenizer files** are `vocab.json` and `merges.txt` of the one GPT-2
  snapshot in the environment's hub cache, or of `--tokenizer-snapshot`.

The output folder:

```
<out>/fixture_runs.json                     the overview
<out>/runs/<run>/<segment>/config.json      the segment config that was run
<out>/runs/<run>/<segment>/cache/           the data-order cache folder, as Megatron wrote it
<out>/runs/<run>/<segment>/run_record.json  docs/interfaces.md 8.1
<out>/runs/<run>/<segment>/step_record.json docs/interfaces.md 8.2
<out>/runs/<run>/<segment>/oracle_log.jsonl docs/interfaces.md 8.3
<out>/runs/<run>/<segment>/index_digests.json   "Index build"
<out>/runs/<run>/<segment>/summary.json         "The summary"
```

`<segment>` is the segment's name without its run: segment `E/seg-2` is the
folder `runs/E/seg-2`, and the seven folders are `E/seg-1`, `E/seg-2`,
`E-rr/seg-1`, `E-rr/seg-2`, `D/seg-1`, `Y/seg-1`, `K/seg-1`.

The overview (`waterwhite.oracle-fixture-runs/1`): the spec's path and SHA-256,
the fixture's path, `seeds` (the seed each run was run with), `seed_overrides`,
`segments` (per segment: its folder, how its start was checked, its mixture, per
shard the name, layout, tokenizer-input file and the documents and samples
Megatron reports, the dataset's length, the samples read and rows written), and
`step_cuts`.

## Seed search: `seed_search.py`

Every read position the fixture states is fixed by a search over candidate seeds
against the oracle's log (plan 5.1). The spec's `[search]` table states the
positions as numbered conditions; this tool tries candidate values of each run's
`random_seed` on one built fixture and reports the seeds under which they hold.
It reports only. A person copies the result into the spec.

```
$WATERWHITE_MEGATRON_ENV/venv/bin/python -I oracle/seed_search.py \
    --spec fixture/spec/fixture_spec.toml --fixture <built fixture> --out <new folder> \
    --seeds FIRST:LAST --time-limit SECONDS \
    [--runs E,D,Y,K] [--accept N] [--best N] [--no-confirm] [--print both|json|toml] \
    [--base DIR] [--shards-from pipeline|mixtures] [--tokenizer-snapshot DIR] \
    [--allow-reworded N,N]
```

How it works:

- **One process.** Megatron is imported once. For each candidate the run's
  index is built and its steps are read with the same code the fixture runs
  use, with the log kept in memory. On a laptop that is about 20 to 30
  candidates a second for run E and fewer for run K.
- **The order.** Run E first: conditions 1 to 12. A seed that satisfies them is
  tried on the round-robin variant, which has run E's seed (conditions 16 and
  17), and is rejected when one of those fails. Then runs D (13), Y (14, 15)
  and K (18 to 23), each over the same candidate range; their conditions rest on
  their own logs. If no seed of run E got as far as the round-robin variant, the
  variant is read once for the best candidate, so the report still says whether
  16 and 17 hold on this build.
- **Candidates** are the seeds `FIRST` to `LAST`, both included, in order. A
  run's search stops at its first accepted seed (`--accept N`: after N).
- **The time limit** covers the candidates of all runs together. Each run still
  to be searched gets an equal share of what is left. The import before and the
  confirming run after are outside it. Which candidates fit into the time
  depends on the machine; what is found for one candidate does not. To repeat a
  search exactly, give a range it can finish.
- **Confirming.** When every run has an accepted seed, the fixture runs are
  written once with those seeds into `<out>/confirm` and all 23 conditions are
  checked again on the written files. The positions in the output then come from
  those files.
- **Cache folders.** Megatron cannot build an index without writing it, so each
  candidate leaves small cache folders under `<out>/scratch` (about 26 files per
  candidate of run E). Nothing reads them afterwards; the folder can be removed.
- **The spec's sentences.** Each predicate was written against one sentence.
  The tool compares the spec's sentences with a mark kept beside each predicate
  and refuses to run when a number, an id or a sentence differs: read the
  predicate against the new sentence, then update its mark in
  `oracle/search_conditions.py` or pass `--allow-reworded`.

Standard output holds the result as JSON and, after one empty line, a TOML
fragment (`--print` picks one). Both are also written to
`<out>/seed_search.json` and `<out>/positions_fragment.toml`.

The JSON (`waterwhite.oracle-seed-search/1`):

- `phases`: per run searched: `tried`, `ended_by` (`accepted`, `time limit` or
  `range exhausted`), `seconds`, `accepted` (the seeds), `conditions` (per
  condition: how many candidates it was `evaluated` on and how many it `held`
  for), `failing_sets` (each set of failing conditions with the number of
  candidates that failed exactly that set, fewest failures first: it shows
  which conditions bind together), and `best` (the candidates with the fewest
  failures, accepted seeds first and all of them: each with every condition's
  verdict and one line of detail, and each accepted seed with its `positions`,
  the values and log lines that show every position under that seed).
- `chosen_seeds`: per run its first accepted seed, or null.
- `all_conditions_hold`: true when every run has a seed and the confirming run,
  if made, agrees. `confirmed`: the verdicts on the written runs.
- `never_held`: conditions no candidate satisfied. `not_evaluated`: conditions
  of runs that were not searched.
- `positions`: per `[positions]` table of the spec, its conditions, whether
  each holds, the values found and the log lines that show them.

The TOML fragment has `[runs.<R>]` with `random_seed` and one `[positions.<name>]`
table per table of the spec, with the values found and `log_lines`, each line a
JSON string as the log holds it. A seed or table of a candidate that was not
accepted is marked `NOT ACCEPTED` or `FAILS` in a comment: it is there to show
what fails, not to be copied. The fragment does not give
`positions.exact_rerun_restart` (not a search condition) or the layout shuffle
seeds (they belong to the build that was searched).

When no candidate satisfies everything, read `never_held` and `failing_sets`. A
condition that holds for no candidate, or for none together with the others,
is a property of the build: change a layout shuffle seed, or enlarge shard-3
with unrelated records as plan 5.1 says, build again and search again.

**Checking written runs.** `--check-runs DIR` checks the conditions on an output
folder of `fixture_runs.py` and prints the same kind of result; it searches
nothing and needs no Megatron (any Python 3.12). The run records' shard digests
must be those of `--fixture`, so logs of one build are not read against another
build's manifests.

```
python -I oracle/seed_search.py --spec fixture/spec/fixture_spec.toml \
    --fixture <built fixture> --check-runs <output of fixture_runs.py>
```

### The conditions

Each predicate is a function `c01` to `c23` in `oracle/search_conditions.py`. It
reads the oracle's log rows of one run, with the record each row names (line i of
a shard's tokenizer-input manifest is document i), the oracle's summary and the
run record's order digests, the step records, and the replica's manifests and
sanitizer log. A row "names" a record when its shard position and ordinal are
that record's document. Shard positions in run E: 0 is shard-1, 1 is shard-2,
2 is shard-3. In run K: 0 is shard-3, 1 is shard-2.

The sanitizer log is read as the replica writes it (docs/replica.md, "The
folder one build writes"): a row's `input` names the sanitizer-input file and
its `line` the line there, counted from 1. A row of a shard whose `input` is
not that shard's sanitizer-input file, or that has no `line`, is refused.

| n | Run | What the predicate checks |
|---|---|---|
| 1 | E | Segment 1's log holds 24 samples, four in each of steps 1 to 6, with global indices 0 to 23 each once; segment 2's holds 24, four in each of steps 7 to 12, with 24 to 47 each once |
| 2 | E | Shard-2 holds 19 documents, by its tokenizer input and by Megatron's count in every segment that lists it, and the sanitizer log lists one skipped line for it, at place 6 of its input |
| 3 | E | Some row names a document of shard-2 whose place in the sanitizer's input is after 6. The place of document i is that of the (i + 1)-th line the sanitizer kept, kept and skipped lines both counted |
| 4 | E | Some sample of step 7 has T as its second row, covering T's tokens 0 to 22 (all of T), with a third row after it; its first row is on T's shard and names a record of source Z; T has exactly one row in the log of run E; and the step records of run E lower step 8 of segment 2 and nothing else. The outcome carries the first row and both neighbours |
| 5 | E | No row of steps 1 to 6 is on shard position 2, and none names a record that "withdraw A" puts in scope |
| 6 | E | The sample at step 8, sample position 3, has no row that names T |
| 7 | E | In that sample, the token after the covered ones (the 17th) is not the first token of a row. The outcome carries the cut: shard position, ordinal, document token, and the covered rows |
| 8 | E | Some document of shard-1 or shard-2 has a row in a step from 1 to 6 and a row in a step from 7 to 12. A pair that reads the same token range is shown first, then a pair whose ranges share a token |
| 9 | E | Some ordinal has rows on two different shards; the lowest is shown |
| 10 | E | Some document that is neither T nor a document fitting condition 12 has two rows in two different samples, each two tokens or more, the second starting at the first one's last token (its start is the other's end minus one) |
| 11 | E | Steps 1 to 6 have a row on shard position 0 and one on shard position 1 |
| 12 | E | Some document W of shard-3, a record of source Z, has exactly one row in steps 1 to 9, and that row is the last row of its sample, in step 7, 8 or 9, not in the sample at step 8 position 3, not in a sample that has a row of T, and covers W's token 0 alone. Rows of W after step 9 are not looked at. The first such document in log order is W |
| 13 | D | The log holds 12 samples, four in each of steps 1 to 3, with global indices 0 to 11, and every row is on shard position 0 or 1 |
| 14 | Y | The log holds 24 samples, four in each of steps 1 to 6, with global indices 0 to 23; one of run Y's shards holds T; and no row names T's document |
| 15 | Y | Some row is on shard position 2 |
| 16 | E-rr | In the round-robin layout's manifests, shard-3 has a line for T, and no line of shard-1 or shard-2 is a record that "withdraw A" puts in scope |
| 17 | E-rr | The log holds 24 samples, four in each of steps 1 to 6, and 24, four in each of steps 7 to 12, and no step record of the run lowers a step |
| 18 | K | The log holds 160 samples in steps 1 to 40, two of each rank in every step, and the oracle's summary says the samplers of steps 1 to 3 were built at consumed count 0 and those of steps 4 to 40 at 12 |
| 19 | K | The dataset length N in the oracle's summary gives N div 4 from 9 to 36 |
| 20 | K | For each rank, the pass of the sampler that starts at step N div 4 + 1 is the one the class named epoch 1; the run record holds a shuffle digest for every launch, epoch and rank the summary lists a pass for; and wherever one launch read an epoch whole, the indices that rank read in those steps, in log order, give the recorded digest |
| 21 | K | Steps 1 to 3 have a row on shard position 0, and that position is shard-3 |
| 22 | K | The lowest step with a row that names T's document is 7 or 8 |
| 23 | K | Every row of steps 1 to 8 that names T has rank 1. With no such row this holds emptily, and 22 fails |

The records "withdraw A" puts in scope are A's records and every generation
output made from one of them through any number of jobs, read from the spec's
sources and its generation file.

**T fits inside one sample.** A sample is 33 tokens and T is 22 with its
end-of-document token (DECISIONS.md D-24). Whether T lies whole inside one
sample depends on where a shard's packing puts it, and the packing follows the
run's seed. What the conditions ask:

- Condition 4 asks that run E reads T whole, in one sample of step 7, as that
  sample's second document. Three things in the log show it. T's row starts at
  token 0 and ends at token 22, the token count the tokenizer gives for T. A
  third row follows it in the same sample: Megatron's item getter reads every
  document of a sample but the last one to its end
  (`GPTDataset._query_document_sample_shuffle_indices` passes no length for
  them), so a row that another row follows ends where its document ends. And T
  has no other row in run E. The third row matters: without it T's last token
  could be the sample's 33rd, which the next sample starts with, and T would lie
  in two samples.
- "No earlier exposure" is condition 5: no row of shard-3 at all in steps 1 to
  6. Segment 1 does not list shard-3, so this holds for every seed; the
  predicate checks it on the log all the same.
- Condition 6 keeps T out of the half-covered sample of step 8. With condition
  4 it cannot fail; it stays a condition of its own because a table row rests
  on it.
- Runs Y and K pack shard-3 under seeds of their own, so there T may lie across
  two samples. Conditions 14, 22 and 23 therefore speak of every row that names
  T, whatever its token range: run Y must read no token of T; run K's first
  step with a token of T must be 7 or 8, and up to step 8 only rank 1 may read
  one.

**W is a document of shard-3** (DECISIONS.md D-13 as amended by D-24). In
shard-1 and shard-2 no document can be read for the first time between
checkpoints 6 and 9: segment 2 packs those shards with the seed segment 1 used,
and the samples of them that steps 7 to 9 read are samples steps 1 to 6 read
before. Shard-3 enters with segment 2. Condition 12 asks for a sample of step
7, 8 or 9 whose 33rd token is token 0 of an unrelated record of shard-3 that no
other row of steps 1 to 9 names. That sample is not the one that holds T, so
the two positions are shown by two samples, and it is not the half-covered
sample of step 8.

The outcome of condition 4 lists every row of run E that names T.

### The final search

The seeds found on one build are that build's. A change to a text, to a shard's
record list or to a layout shuffle seed makes another build, and the search is
then run again. To repeat it: build the fixture (docs/replica.md) and run, from
the repository root, under the machine's CPU rules:

```
export WATERWHITE_MEGATRON_ENV=<environment folder>
export HF_HOME=$WATERWHITE_MEGATRON_ENV/hf_home HF_HUB_OFFLINE=1
$WATERWHITE_MEGATRON_ENV/venv/bin/python -I oracle/seed_search.py \
    --spec fixture/spec/fixture_spec.toml --fixture <built fixture> \
    --out <new folder> --seeds 0:9999 --time-limit 1200
```

It ends early when every run has a seed. `--runs E` (or `D,Y,K`) searches part
of the runs, so a search can be made in several short calls; each run's search
starts at `FIRST` and stops at its first accepted seed, so the seed found is
the smallest in the range. Then copy `positions_fragment.toml` into the spec by
hand, build nothing again, and check the spec's seeds with

```
$WATERWHITE_MEGATRON_ENV/venv/bin/python -I oracle/fixture_runs.py \
    --spec fixture/spec/fixture_spec.toml --fixture <built fixture> --out <new folder 2>
python -I oracle/seed_search.py --spec fixture/spec/fixture_spec.toml \
    --fixture <built fixture> --check-runs <new folder 2>
```

which must say `"all_conditions_hold": true`.

### The search that fixed the spec's seeds

The build: the fixture's final texts (X 16 tokens, P 15, T 22, each with its
end-of-document token), both layout shuffle seeds 0. The partitioned shards
hold 18, 19 and 27 documents (798, 827 and 1044 tokens; 24, 25 and 32 samples);
the round-robin chunks 22, 21 and 21 documents (951, 874 and 844 tokens; 29, 27
and 26 samples). Two builds from the spec give the same bytes.

| Run | Candidates, from seed 0 | Result |
|---|---|---|
| E | 579 in 23 seconds | seed 578 accepted. Condition 4 held for 8 candidates, condition 12 for 40, condition 6 for 555, condition 7 for 563; conditions 1, 2, 3, 5, 8, 9, 10 and 11 for all. 499 candidates failed 4 and 12 only |
| E-rr | 1 (seed 578) | conditions 16 and 17 held |
| D | 1 | seed 0 accepted |
| Y | 1 | seed 0 accepted |
| K | 5 | seed 4 accepted. Condition 22 held for 1 candidate, condition 23 for 3; 18 to 21 for all |

No layout shuffle seed was changed and shard-3 was not enlarged. The fixture
runs written from the spec (`fixture_runs.py`, no seed override) pass
`--check-runs` with all 23 conditions; with run E's seed set to 577 the same
check fails conditions 4 and 12. The positions under these seeds, with their
log lines, are the spec's `[positions]` tables; the sizes are its `[sizes]`
table.

Under these seeds: the first sample of step 7 reads the last 7 tokens of Z-088,
T whole and the first 4 tokens of Z-067. W is Z-089, read through the 33rd token
of the third sample of step 9. The cut of step 8 falls at token 35 of Z-080, a
document of shard-3. The repeated draw is one whole sample of Z-051, read at
step 6 and again at step 10. Run Y reads no token of T. Run K's blend holds 57
samples, an epoch of 14 steps, and its first read of T is at step 8 on rank 1.

`positions.exact_rerun_restart` is not a search condition. It was read off two
logs: the log of run E, and the log of run E over a build of the spec with T
left out (`python -m replica build-fixture ... --layout partitioned --leave-out
T`, then `fixture_runs.py --runs E`). Steps 1 to 6 give the same rows in both;
step 7 is the first that differs.

An earlier build, with a T of 195 tokens and W asked for in shard-1 or shard-2,
had no seed for run E: T lay across about seven samples, and no document of
shard-1 or shard-2 had exactly one row in steps 1 to 9, for the reason given
above under "W is a document of shard-3". DECISIONS.md D-24 records both
changes.

## Limits

- **GPT-2 BPE only.** Another tokenizer type is refused (DECISIONS.md D-09).
- **The extra token must be on.** `add_extra_token_to_sequence: false` is refused.
  With it off, Megatron's item getter writes the tokenizer's pad token id into
  the last label, and the GPT-2 tokenizer has none: in a run on the throwaway
  corpus the getter stopped with a `TypeError` in `GPTDataset.__getitem__`.
- **Blends without weights only**, and training-only splits only.
- **Ranks are simulated** one after another in one process. No distributed launch
  is made and no trainer runs. What the log shows is what Megatron's dataset and
  sampler code reads, not what a trainer consumed (plan F9).
- **One machine.** Whether the index arrays and `torch.randperm`'s orders are the
  same on another machine or torch build was not tested.

## What was shown in runs, and what was only read

Shown, on a throwaway corpus of three JSONL files (44, 36 and 52 short English
lines; 36, 24 and 35 samples at sequence length 32), global batch 4, with
megatron-core installed from the checkout at `core_v0.20.0` (the package reports
its version as `0.20.0+685fb8b`), torch `2.14.1+cpu` and Megatron-Bridge's
sampler file at `v0.7.0`. Each is a test under `tests/oracle/`.

| What | Megatron code driven |
|---|---|
| Two index builds into two new folders give equal digests for all five kinds. Another seed changes the three kinds of every shard and neither of the blend's two | `BlendedMegatronDatasetBuilder.build`, `GPTDataset`, `BlendedDataset` |
| The oracle's digests equal `waterwhite.ids.array_digest` applied to the `.npy` files in the cache folder | (the tool's definition against the oracle's) |
| Sequential log on a two-shard blend: the logged ranges, applied to the raw bytes of the `.bin` files with document boundaries from an independent tokenization, give 33 tokens per sample, equal to what an uninstrumented `GPTDataset.__getitem__` returns | `MegatronPretrainingSampler`, `GPTDataset.__getitem__`, `IndexedDataset.get` |
| A resume at consumed count 24 logs the tail of the full stream | `MegatronPretrainingSampler` |
| Sequential, data-parallel size 2, two micro batches per rank per step: a step reads eight consecutive indices; each rank's indices equal the class driven directly | `MegatronPretrainingSampler` |
| A sequential launch that outruns its dataset is refused before a log or record is written; a launch that ends at the dataset's end is read; with `allow_sequential_restart` the launch is read and starts again at its own consumed count | `MegatronPretrainingSampler` |
| Cyclic, data-parallel size 2, sharding on, two epochs: the logged indices equal the class driven directly; each rank reads its bucket once per epoch; the two epochs differ; the digests equal `waterwhite.ids.order_digest` of those lists | `MegatronPretrainingRandomSampler` |
| The same order digests from a second, fresh process | `MegatronPretrainingRandomSampler`, through `run_record.py` |
| Cyclic, two launches in one segment, a resume inside an epoch and an epoch boundary: the log equals that of one launch over all steps. Each launch records the whole order of every epoch it touches (30 indices per rank, for 16, 14 and 10 read), the two launches record the same digest for the epoch they share, and each digest equals the class driven directly from the epoch's start | `MegatronPretrainingRandomSampler` |
| The fixture spec's seven run segments over a stand-in built fixture made of the throwaway shards: every file of the layout, seeds from the spec and an override, run records with relative paths, run K's eight order digests of whole epochs equal to the class driven directly, the cut of step 8 equal to a token-by-token walk of the log | both sampler classes, the index build, through `fixture_runs.py` |
| A candidate read in memory by the search gives the rows, order digests, step records and document counts the fixture runs write for the same seed | the same, through `seed_search.py` |
| A search stops a run at its share of the time limit (shown with a clock that is handed in), counts each condition, and, when every run has a seed, writes the fixture runs and checks all conditions on the files. The confirming path was run with the predicates that need the fixture's texts replaced by ones that always hold; the stand-in cannot satisfy them | the same |
| Cyclic, sharding off: Megatron-LM's and Megatron-Bridge's classes log the same while the micro batch size divides the dataset, and differ at micro batch 7 over 60 samples. The sequential classes of the two log the same | both code bases' sampler classes |
| In a shard's own sample order, the last token a sample reads is the first token the next sample reads (58 of 58 pairs) | `GPTDataset.__getitem__` |
| A read the log does not explain stops the run | (known-bad input: an altered `IndexedDataset.get`) |
| F1 on the throwaway corpus: one document per line in input order; the end-of-document token 50256 is inside every document's span | `IndexedDataset`, `build_tokenizer` |
| The replay gives the recorded digests; with another seed the arrays that differ are the three of each shard | the index build |

Read in Megatron's code and not shown in a run:

- That a trainer starts a new pass through `cyclic_iter`. The oracle's own loop
  stands in for it; no trainer ran.
- That the trainer adds n * m * k to the consumed count per step. The oracle
  takes each launch's count from the config.
- Megatron-LM's and Megatron-Bridge's loader factories. Neither was called; the
  sampler classes were built with the factories' arguments.
- Anything about a real distributed launch, a GPU, or a checkpoint's saved
  counter (plan F4).

Not run at all: the 10k and 100k sets of plan 5.6, a tokenizer other than GPT-2
BPE, a split with validation documents, and a blend with weights.

## Tests

- `tests/oracle/test_oracle_plain.py`: no Megatron needed. The wall, the digests
  against the tool's definitions, the config rules, the records, the positions
  report.
- `tests/oracle/test_oracle_search.py`: no Megatron needed. The spec reader and
  what it refuses, the cut of a sample, the readers of a built fixture, the
  segment configs made from the spec, the TOML fragment, and every condition's
  predicate: on runs written by hand in which all 23 conditions hold, and on 58
  changes of those runs, each of which breaks the sentence of one condition and
  must make its predicate say no.
- `tests/oracle/test_oracle_spec_positions.py`: no Megatron needed. The spec as
  it is committed: nothing pending, every `[positions]` table agrees with the
  log lines it quotes, and the facts the conditions state (T whole as a second
  row at step 7, W's one token, the cut of step 8) can be read off those lines.
- `tests/oracle/test_oracle_megatron.py` (marker `megatron`): the oracle against
  Megatron in one process.
- `tests/oracle/test_oracle_fixture_runs.py` (marker `megatron`): the fixture
  runs and the seed search over a stand-in built fixture, in one process.
- `tests/oracle/test_oracle_scripts.py` (markers `megatron` and `slow`): each
  entry script as its own process.

The `megatron` tests tokenize the throwaway corpus with Megatron's preprocessing
script first, as the fixture spec sets it up: GPT-2 BPE, `--append-eod`, one
worker, one partition, a fresh input folder per file. The stand-in built fixture
of `test_oracle_fixture_runs.py` is made of those shards, laid out as a fixture
build lays them out; it has the fixture's layout and not its texts. The tests
fail, and are not skipped, when the environment is missing; leave them out with
`-m "not megatron"`. The script tests start several processes that each import
Megatron and take about a minute together.

```
WATERWHITE_MEGATRON_ENV=<folder> $WATERWHITE_MEGATRON_ENV/venv/bin/python -I -m pytest tests/oracle
```
