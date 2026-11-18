# The exposure join

`waterwhite.join` says, for one run segment, which (shard, document ordinal,
token range) rows each completed training step read, and writes them to the
store as step traces. It is the join of plan 3.5. It works from the run record
the training wrapper wrote, the operator's step record and the shards' index
files. It runs no trainer and imports neither the trainer's framework nor its
tensor library, with one exception: for a run read with the cyclic sampler it
calls torch to compute the shuffle, and imports it then and only then (section
6). Beside the standard library numpy is the only thing it needs otherwise.

Where the plan says the data order cannot be trusted, the join writes no trace.
The segment is then registered with the trace route "none" and the named cause
the plan fixes, and every checkpoint of it answers "may contain" naming that
cause (plan 2.4).

Plan sections served: 3.5, 2.4, 2.2(e), 3.1 (run segment, step trace), 3.4 (the
trace count), assumptions F2, F3, F4, F6 and F9, limits L9 and L10, Decisions 18,
19 and 27, and Checks 1, 2 and 8 of 5.5. `DECISIONS.md` entries D-08 (what a
cache digest is taken over) and D-16 (what a shuffle digest covers) apply
unchanged.

Code: `src/waterwhite/join/`. Tests: `tests/join/`. The command `waterwhite
register-run` (docs/cli.md) calls it for one segment and registers the segment,
its steps, its traces and its checkpoints in one store call.

## 1. In short

```python
from waterwhite.join import SegmentRegistration, join_segment

result = join_segment(
    store,
    SegmentRegistration(run="E", segment="E/seg-2", position=1, start_kind="resume",
                        mixture_instance_id=mixture, parent_checkpoint=checkpoint),
    "runs/E/seg-2/run_record.json",
    cache_dir="runs/E/seg-2/cache",
    step_record="runs/E/seg-2/step_record.json",
)
result.route          # "cache files", "replay" or "none"
result.cause          # with "none": the named cause, a NAMES_* word of the vocabulary
result.cause_detail   # beside "data-order trace missing": one of the six reasons (section 3)
result.steps_written  # how many steps got a trace
result.untraced       # registered steps with no trace, each with one sentence
result.checkpoints    # (name, step) pairs of the step record; not registered here
```

`join_segment` reads the files, joins, and registers the segment, its completed
steps, its traces and the checkpoints of the step record with one call of
`Store.register_run_segment`, in one transaction. `compute_join` does the join
alone and touches no store: it takes the run record as an object and the store's
list of verified joins, and returns the same result.

A cyclic run also needs torch (section 6). Where torch cannot be imported and
the shuffle has to be computed, `join_segment` raises "torch not available" and
writes nothing.

## 2. What the join reads

| Input | What is read from it |
|---|---|
| the run record (docs/interfaces.md 8.1) | `versions.release_tag`; of `settings`: `dataloader_type`, `sampler_class`, `data_sharding`, `micro_batch_size`, `global_batch_size`, `sequence_length`, `add_extra_token_to_sequence`, `random_seed`, `split`, `train_iters`, `train_samples`, `batch_schedule`; `shards` (position, prefix, `idx_sha256`); `launches` (position, `data_parallel_size`, `consumed_at_start`, `first_step`, `last_step`); `cache_digests`; `shuffle_digests` |
| the step record (8.2) | per step `batch_size` and `last_sample_tokens_read`; `checkpoints` |
| each shard's `.idx` file | the token count of every document. The file must have the SHA-256 the run record's shard list holds for it |
| the cache folder (8.4) | the `.npy` index arrays. `description.txt` is not read |

The `.bin` file of a shard is never opened. A shard prefix of the run record is
relative to the folder the run record lies in.

A field the wrapper could not record is absent or null. The join reads it as
"not recorded" and names a cause; it never puts a default in its place. A field
that is present with the wrong type is a different thing: the document is then
not a run record, and it is refused with "invalid record" before anything is
written.

## 3. The questions, in order

The join asks eight questions in a fixed order. The first one that fails ends
it: the route is "none", no trace is written, and the segment carries the cause.

| # | Question | Cause when it fails | Reason beside it |
|---|---|---|---|
| 1 | Is the release the run states on the store's list of verified joins? (A run that states none passes here.) | "data-order trace unverified for this version" | |
| 2 | For a cyclic run: does the run record hold shuffle digests at all? | "cyclic order not recorded" | |
| 3 | Does the run record hold a digest for every index array? (No run record: no.) | "data-order trace missing" | "no recorded digest" |
| 4 | Does the run record state a release? | "data-order trace unverified for this version" | |
| 4 | Is the release verified with this sampler type and sampler class? Is a cyclic run's data sharding on? | "data-order trace unverified for this sampler type" | |
| 5 | Does the run record hold the shard list, the micro and global batch size, the sequence length, the extra-token flag and a launch? | "data-order trace missing" | "run record incomplete" |
| 5 | Is the extra token on? | "data-order trace missing" | "extra token off" |
| 6 | Is each shard's index file there and readable? (A shard of the list with no prefix is a record that lacks it.) | "data-order trace missing" | "shard index unreadable"; "run record incomplete" for a shard with no prefix |
| 6 | Does the run record hold that file's digest? | "data-order trace missing" | "no recorded digest" |
| 6 | Does the file have that digest? | "data-order trace missing" | "digest mismatch" |
| 7 | Do the cache files, or else the replay, give every index array with its recorded digest? | "data-order trace missing" | "digest mismatch"; "no recorded digest" when no cache file was kept and a setting of the replay was not recorded |
| 7 | Has every array the shape an index array of its kind has? | "data-order trace missing" | "index array malformed" |
| 8 | For a cyclic run: does the run record hold the shuffle digest of every epoch a launch reads in, for every rank? | "cyclic order not recorded" | |
| 8 | Do the order the join computes (section 6) and the recorded one have the same digest and the same count? | "cyclic order digest mismatch" | |

Why this order. A release the join was not verified for decides everything
(plan F6): on any such release the join writes no trace and names the version,
whatever else the run recorded. Then comes what the run did not record
(questions 2 and 3), because nothing done after the run can supply it. Plan 3.5
says a cyclic run with no recorded shuffle answers "cyclic order not recorded"
whatever cache files it kept, so question 2 stands before question 3. Plan
Check 2 (iii) fixes the answer of a run whose settings and digests are withheld
as "data-order trace missing", and the sampler type is one of the settings, so
question 3 stands before the sampler half of the gate. Section 10 gives the
other order and what would change the choice.

The last reason of question 7 follows docs/interfaces.md section 2: a run whose
cache files and settings are gone has nothing a replay could be held against.

The reasons. The plan gives two, "no recorded digest" and "digest mismatch", and they
stay with the replay and the digests (questions 3, 6 and 7). Four more words stand
beside "data-order trace missing" where the tool knows what the record, or a file it
names, lacks and no digest is the question: "run record incomplete", "shard index
unreadable", "extra token off" and "index array malformed"
(`vocabulary.TRACE_MISSING_RECORD_REASONS`, docs/interfaces.md section 2).
The join stores the word as the segment's `trace_cause_detail` at the gate that
decided the state, `register-run` prints it as `cause_detail` and an answer carries it
as `names_detail`. A run record with no launch is the first of these: the join stores
"data-order trace missing" with "run record incomplete", and the store refuses it any
trace route (docs/coverage.md section 8). The store accepts beside "data-order trace
missing" one of the six words or none, so that a direct caller that does not know the
reason may store none, and no reason beside any other cause. The join stores no null
reason: question 7's arrays whose recorded digests hold but whose shape no array of
their kind has get "index array malformed" (section 10, item 4: the record and the
files agree, and the files are not what their kind must be; differing evidence is a
cause).

When every answer is yes, the route is "cache files" or "replay" and each step
the map covers gets a trace. A step the map does not cover gets no trace and is
listed in `result.untraced` with its reason. It is an untraced step of plan
2.2(e), not a cause of the segment:

- no launch of the run record holds the step;
- nothing gives its batch size, or its batch size is not a whole number of
  micro-batch rounds;
- a sequential launch reached the end of its dataset before the step was whole
  (the sampler would start a new pass, and plan 3.5 does not say what that pass
  reads);
- a step of a cyclic run has another batch size than the global one (plan 3.5
  names no batch schedule for the cyclic sampler);
- a cyclic launch starts at a counter that lies inside a round, which the
  sampler itself refuses.

Once one step of a launch cannot be placed, the counter of every later step of
that launch is unknown, so they get no trace either.

## 4. The two routes to the index arrays

A segment has three arrays per shard (document index, sample index, shuffle
index) and, with two or more shards, two for the blend (dataset index, dataset
sample index). A segment of one shard builds no blend.

An array counts only when its digest (`ids.array_digest`, over the dtype, the
shape and the bytes) equals the digest the run record holds for its kind and
shard position.
For the cache files this is stricter than plan 3.5 and 4.6, which ask a recorded
digest of a replay only: a decision of the project's owner (DECISIONS.md D-41).

**Cache files** come first (plan 3.5: the join resolves from the data-order
cache files). The join loads every file of the cache folder whose name ends in
one of the five kinds and takes, for each recorded (kind, shard position), the
file whose array has the recorded digest. That is also how a file is tied to a
shard: by the digest recorded for the shard's position in the run record's
shard list. The file name is a hash of paths and settings and is used only to
tell the kinds apart. The route is "cache files" only when every array is found.

**Replay** is tried when a cache file is missing or does not match. The join
derives all arrays again from the run record's settings and seed and the shards'
document lengths (`_replay.py` states the semantics in its own words), and
accepts the result only when every array has its recorded digest. A wrong seed,
a wrongly stated setting, another release or another machine's arithmetic shows
as a digest that differs, and the replay then supports no answer (plan 3.5).
With no recorded digest no replay is tried at all.

`result.replay` says what became of the replay: `attempted`, `accepted`, the
arrays whose digests differ, one sentence (`why_not`) and `reason`, the word of
docs/interfaces.md section 2 when the replay was refused ("digest mismatch") or not
tried for want of a digest to hold it against ("no recorded digest"), else None.
`register-run` prints it as `{"attempted", "accepted", "reason"}` with `accepted`
null where `attempted` is false (docs/cli.md). The cases: the cache files matched
(not attempted, no reason); the run record holds no digest of an index array, or
no cache file was kept and a setting the replay needs is missing (not attempted,
"no recorded digest"); cache files were kept that do not match and no replay is
possible (not attempted, "digest mismatch");
the replay was made and a digest differs (attempted, not accepted, "digest
mismatch"); the replay was made and every digest is the recorded one (attempted,
accepted). A segment that ended at an earlier question (an unverified release, a
cyclic run with no shuffle digest) never reached the arrays and has the default. `replay_arrays(record, lengths)`
gives the replayed arrays themselves, so a check can hold them against withheld
files byte for byte (Check 2 (i)).

What the replay needs of the run record: `sequence_length`, `random_seed`,
`split`, `add_extra_token_to_sequence`, and `train_samples` or `train_iters`
with `global_batch_size`.

**Writing the replay to files.** `waterwhite replay --run-record PATH --out DIR`
(docs/cli.md) calls `write_replay(document, record_dir, out_dir)`: it reads the
run record and the shards' index files as question 6 does, derives every array with
`replay_arrays`, and writes each to `DIR` as `<kind>.npy` (the blend's two) or
`<kind>-<shard position>.npy` (a shard's three), dtype and shape as the replay made
them. It opens no store and registers nothing. It prints each array's digest beside
the digest the run record holds (null where it holds none) and whether they are
equal. It compares digests only: whether the written bytes equal the kept cache
file of the same kind and shard is the comparison of Check 2 (i), made by whoever
holds both. Where the replay cannot be made it raises "replay not possible" (a
setting missing, no shard listed, a shard's index file absent, unreadable, or not
the file whose digest the record holds), where join and registration would have
stored a cause; a file already in `DIR` that it would write is a usage error, and
nothing is written then.

## 5. From a step to its samples, and from a sample to its rows

**The counter, not the step number (plan F4).** Each launch has its own
consumed-sample counter. A resume carries the counter on, a warm start restarts
it at 0. The counter of a step is its launch's counter plus the batch sizes of
the launch's steps before it. The step number names the trace and decides
nothing.

**Batch sizes (plan F2).** The completed steps of a segment are the steps of
its launches and the steps the step record lists. A step's batch size is the
step record's `batch_size` where it gives one, else the global batch size. For
the sequential sampler the step record is so the operator's batch schedule. A
step whose batch size nothing gives cannot be registered, since the store keeps
a batch size for every step: it is left out and named in `result.unsized`.

**Rounds and places.** With `n` data-parallel ranks and a micro batch of `m`,
one round is `m * n` samples. `sample_pos` counts the round, then the rank, then
the place in the micro batch (docs/interfaces.md 8.3).

- Sequential sampler: a step reads the next batch-size indices in a row from its
  counter. Rank `r` holds places `r * m` to `(r + 1) * m` of each round.
- Cyclic sampler with data sharding: the dataset of `N` samples loses its last
  `N mod (m * n)` samples; the rest is one epoch. Each rank owns
  `(N div (m * n)) * m` consecutive indices, its bucket. The counter gives the
  epoch and the place inside it, and a rank at place `p` of the epoch has read
  `p / n` indices of its order for that epoch. The order comes from the run's
  record (section 6).

**Rows.** For the index a sampler asks of the top dataset: the blend's two
arrays give the shard and the sample number in it; the shard's shuffle index
gives a place in its sample index; two neighbouring entries of the sample index
give the first and the last document of the sample with a token offset each; the
document index gives the ordinals. A sample spans `sequence_length + 1` tokens.
Its last token is the first token of the next sample in sample-index order and
is read as a label, so a document a step read only through that token has a row
of one token and is consumed (plan 3.5, Decision 18).

**Coverage.** A sample whose rows come to its token count is whole (plan F9). A
sample that points outside an array or a document, or whose rows hold more
tokens than a sample has, is written with no row and a coverage mark of 0. A
sample whose rows hold fewer tokens keeps them and has that count as its mark.
Nothing is filled in.

**A lowered step.** When the step record gives `last_sample_tokens_read` for a
step, the last sample of that step (the one at `sample_pos` batch size minus 1)
is cut: rows that end inside the tokens read stay, the row that holds the next
token ends there, no later row is written, and the mark is the number of tokens
read. The sample is still counted, so the trace still holds as many samples as
the step's batch size and the loud check of plan 3.4 does not stop. A number at
or above the sample's token count changes nothing: a note can only lower a mark.

## 6. The order of a cyclic run

The trainer's sampler draws each epoch's order from torch's random generator,
seeded with the epoch number (not with the run's seed), and writes it to no file.
The join computes that order itself, from the run record's sizes, and holds it
against the shuffle digest the run record holds (plan 3.5):

- the dataset has `N` samples (the blend's length, from the index arrays), the
  launch `n` data-parallel ranks and a micro batch of `m`. Data sharding is on.
  Each rank owns a bucket of `(N div (m * n)) * m` consecutive indices, and rank
  `r` starts at `r * bucket`;
- for each epoch a launch reads in, the join makes the sampler's own draw:
  `g = torch.Generator()`, `g.manual_seed(epoch)`, `torch.randperm(bucket,
  generator=g)`, and adds `r * bucket` to each place for rank `r`. It does not
  drive the trainer's sampler class and does not restate torch's generator in
  another library (choice 1 in section 10);
- the order counts only when `ids.order_digest` of the whole order and its length
  equal the digest and the count the run record holds for that (launch, epoch,
  rank) (DECISIONS.md D-16: a digest covers a whole epoch's order for one rank,
  whatever part of it the launch read). Otherwise: "cyclic order digest
  mismatch". A launch, epoch or rank with no digest in the run record, or a run
  record with none: "cyclic order not recorded", whatever cache files were kept.
  The join asks for every digest before it draws anything, so a run that
  recorded too little never loads torch;
- the wrapper computes the digests in the run's own environment with its own call
 
  to torch, and never through a function of this package: if writer and join
  shared one function, the digests could not disagree.
- the join keeps every order it computed with its digest and count beside the
  recorded ones (`JoinResult.shuffle`, one entry per (launch, epoch, rank) its
  launches read, in that order, the ones that differ too). `register-run` prints
  that list as `shuffle` (docs/cli.md): what it shows is what the join held against
  the record, never a second computation. It is empty for a sequential run, for a
  record with no shuffle digest or a missing one (the join asks before it draws),
  and for a run an earlier question stopped.
- `show --trace` places the steps of a stored cyclic segment again to print their
  `epochs` (`join.step_epochs`, which uses `_sampler.cyclic_epochs`): a launch
  starts at its counter, each step adds its batch size. The dataset's length is not
  stored; a launch's sizes come from the count of its recorded shuffle digests (the
  bucket), and an epoch holds `bucket * n` samples.

torch is imported inside `_shuffle.epoch_permutation`, at the moment a cyclic
run's shuffle is drawn, and nowhere else (`tests/join/test_join_imports.py`
reads every import of the package). Importing `waterwhite.join`, joining a
sequential run and joining a cyclic run answered by an earlier question load no
torch. Where torch cannot be imported at that moment the join raises "torch not
available" (`TorchUnavailable`, exit code 4, a refused write): the registration
is refused and nothing is written. The cause words say what a run record holds,
not what a machine has, so no cause word is stored for this.

The cyclic sampler with data sharding off follows another rule (plan 3.5, input
4). The join is not verified for it and answers "data-order trace unverified
for this sampler type".

## 7. What reaches the store

`join_segment` calls `Store.register_run_segment` once, with the run record
whole, the completed steps as (step, batch size, tokens read of the last
sample), the route, with the route "none" the cause and its reason, the step
traces (`traces=`) and the checkpoints of the step record (`checkpoints=`, each
a name and a step). One call is one transaction: a trace that does not fit its
step's batch size, or a checkpoint at a step the segment does not have, leaves
nothing of the segment written. Each trace counts as many samples as its step's
batch size.

The traces are written once (plan Decision 19). Joining the same segment again
from the same files changes nothing. Joining it again from other evidence is
refused by the store: a registered segment is not changed.

The store refuses a route other than "none" for a segment whose run record
holds no digest of an index array the segment has (three per shard and, with two
or more shards, two for the blend; with no run record at all it holds none), for a
cyclic segment whose run record holds no shuffle digest (DECISIONS.md D-26, D-33), and
for a run record that holds no launch (empty, null or absent: nothing says which step
comes first), so the route follows from the run record and not from the caller, whoever
calls the store. It refuses any trace for a segment whose route is "none", through
`write_step_traces` and through `register_run_segment(traces=...)`; both refusals are
"invalid record" and write nothing. So no store the tool writes holds a trace for a
segment whose run record holds no launch. It refuses a segment with no route that
names a cause outside the five of `vocabulary.TRACE_CAUSES`, and a reason beside any
cause but "data-order trace missing" or outside the six words of
`vocabulary.TRACE_MISSING_STORED_REASONS` (a null reason is accepted). The join and the
store count a segment's shards the same way (`vocabulary.shard_count_of`,
`vocabulary.cache_keys`), so the store never refuses a route the join produced.
Coverage reads a segment whose route is "none" by its stored cause and reason, with or
without a launch in its run record (docs/coverage.md section 8). A run record with no
launch that `register-run` registers has the route "none" and, when no earlier
question applies, the cause "data-order trace missing" with the reason "run record
incomplete" (question 5), and its checkpoints answer "may contain" naming them.

`_store_adapter.py` is the one module that calls the store's write method.

`result.untraced_steps` gives the steps that are untraced in the sense of plan
2.2(e): every step when the route is "none"; else the registered steps with no
trace, the steps whose trace has an uncovered token range and the steps left
out of the registration for want of a batch size. The command prints them.

## 8. Errors

- "usage" (`UsageError`, exit code 1): a run record or step record path that
  cannot be read.
- "invalid record" (`InvalidRecord`, of `waterwhite.errors`, exit code 4): a run
  record or step record file that is not JSON or not an object; a run
  record of another format; a field of the wrong type; shard positions that are
  not 0, 1, 2 and so on; two launches that share a step; a step record written
  for another segment. The store raises the same error for a registration it
  refuses, for example a run record that names another segment than the
  registration.
- "replay not possible" (`ReplayNotPossible`, of `waterwhite.errors`, exit code
  1): raised by `replay_arrays` when the run record lacks a setting the replay
  needs, and by `write_replay` also for a record that lists no shard and for a
  shard whose index file is absent, unreadable or not the one the record's digest
  names. `compute_join` and `join_segment` never let it out: inside a join it
  becomes the cause of question 7.
- "torch not available" (`TorchUnavailable`, exit code 4): a run read with the
  cyclic sampler whose shuffle must be computed, on a machine where torch
  cannot be imported (section 6). Nothing is written.
- Nothing is raised for evidence that is missing, differs or is not verified:
  those are causes (section 3).

The package's surface is what `waterwhite.join.__all__` names. The modules
behind it are its own parts.

## 9. Tests

Default tests, no Megatron needed, under a second:

- `tests/join/join_kit.py` builds the hand world: two tiny shards, index arrays
  written by hand, and the rows of every sample derived by hand in its text.
- `test_join_rows.py`: the chain from a sample to its rows, the extra token, a
  document split across samples, the same ordinal in two shards, samples that do
  not resolve, the cut.
- `test_join_sampler.py`: both samplers, two ranks, several rounds, the epoch
  boundary, a resume inside an epoch.
- `test_join_walk.py`: whole joins of the hand world: a resume, a warm start,
  two launches, the batch schedule, the lowered step, the cyclic run with the
  orders the test draws with its own call to torch (these need torch, marker
  `megatron`).
- `test_join_replay.py`: the replay against arrays Megatron's code built, kept
  as numbers in `golden_megatron.py`, and the blend and the sample index derived
  by hand.
- `test_join_causes.py`: every cause of section 3 and every refusal of section 8.
- `test_join_store.py`: registration, traces read back, the lowered step in the
  store, a second join, and the answer a checkpoint of a segment with no trace
  gives through `waterwhite.coverage`.
- `tests/coverage/test_show_trace_and_replay.py`: `show --trace` and `replay`
  through the command line.
- `test_join_imports.py`: in a fresh interpreter the package loads numpy and no
  trainer, oracle, replica or torch module; a sequential join and a cyclic join
  answered by an earlier question load no torch; `_shuffle.py` is the one file
  that imports torch, inside a function.

`test_join_megatron.py` (marker `megatron`) is the comparison with Megatron's
own behaviour. It builds the hand-checked fixture with the replica, runs the
oracle's fixture runs, joins all seven segments of runs E, E-rr, D, Y and K by
both routes and compares the join's rows with the oracle's log line by line. It
also holds the replayed arrays against the kept cache files byte for byte, the
cut of step 8 against the cut the oracle worked out, and the replay of a
one-shard segment against Megatron's builder. The row c08.K.digests-match
(plan 5.5 Check 8) is `test_the_shuffle_the_join_computes_for_run_k_matches_the_recorded_digests`:
for every (launch, epoch, rank) of run K's run record the join's own shuffle has
the digest and the count the oracle recorded from Megatron's sampler class.

```
WATERWHITE_MEGATRON_ENV=<folder> <its python> -m pytest tests/join
```

## 10. Choices the plan does not settle

1. **Where the order of a cyclic run comes from** (the ruling of
   plan 3.5). Built: the
   join calls torch from the tool, for cyclic runs only, as the sampler does
   (`torch.Generator` seeded with the epoch, `torch.randperm` over the bucket),
   and accepts the order only when its digest and count equal the recorded ones.
   Result: exact answers for a cyclic run whose digests were recorded, on any
   machine with torch; a torch build that draws differently shows as "cyclic
   order digest mismatch" and is never used; the tool needs torch for those runs.
   Other option: a numpy re-implementation of torch's generator. Result: no
   dependency beyond numpy, and the tool restates another library's generator,
   exact only while torch's algorithm stays. It is not available today:
   `torch.randperm` equals none of numpy's `RandomState`, `default_rng` or
   `Generator(MT19937)` permutations for n of 24, 40, 160 and 1000 and seeds 0, 1
   and 1234 (a probe on 15 Nov 2026). What would change it: a numpy route shown
   equal to `torch.randperm` over the sizes and seeds of the fixture and of a
   field run; or a runtime that cannot carry torch for cyclic runs, in which case
   every cyclic run answers "may contain" naming "cyclic order not recorded"
   (the plan's fallback). Dependencies: numpy stays the one runtime dependency;
   torch is an optional dependency, needed for cyclic runs only (a CPU build is
   enough).
2. **The order of the questions** (section 3). Built: a stated release that is
   not verified first, then what was not recorded, then the sampler half of the
   gate. The other option: the whole gate first. Result of the other option: a
   run whose settings are withheld names "data-order trace unverified for this
   sampler type", which is not the cause plan Check 2 (iii) expects, and a run
   with no run record names the version where the built order names the missing
   digest. What would change it: a reading of Check 2 (iii) in which the sampler
   type stays in the run record.
3. **A cache folder that does not match.** Built: the join goes on to the
   replay, which counts when its digests match. The other option: files that are
   there and do not match end the join with "digest mismatch". Result of the
   other option: a run whose cache folder was rebuilt later at the same place
   loses exact answers it could have had from its settings. What would change
   it: a ruling that a cache folder with wrong files is itself a finding that
   must stop the run's registration.
4. **Missing or differing evidence is a cause; a malformed record is an error.**
   The other option: raise for a missing shard file or a missing setting too.
   Result of the built option: a run whose shards are gone can still be
   registered, and its checkpoints answer "may contain" with the cause. Result
   of the other: the caller must decide what to register. What would change it:
   a registration command that should never register a segment without a trace
   unless told to.
5. **A cyclic run with one unusable order gets no trace at all.** The other
   option: trace the steps of the epochs whose orders are good and leave the
   rest untraced. Result of the built option: every checkpoint of the segment
   names the cyclic cause, as plan 2.4 words it. Result of the other: a
   checkpoint before the bad epoch keeps an exact answer, and a later one names
   "step untraced" and not the cyclic cause. What would change it: a store that
   can hold a cause for a range of steps.
6. **A step the map does not cover is untraced; the segment keeps its route.**
   The other option: one unplaceable step refuses the whole segment. Result of
   the built option: a sequential launch that ran past its dataset keeps exact
   traces up to that point. What would change it: nothing found.
7. **A run read with the extra token off gets no trace.** The other option:
   resolve samples of `sequence_length` tokens. Result of the other option:
   traces for a span no check verified (plan F9 holds only for the +1 span).
   What would change it: an oracle run with the extra token off.

## 11. Open questions this module does not settle

- **docs/interfaces.md 8.1.** `settings.batch_schedule` has no stated form. The
  join does not read it; a run record that sets it gets its batch sizes from the
  step record only.
- **docs/interfaces.md 8.2.** It does not say whether the step record lists
  every completed step or only the lowered ones. The join takes the union of the
  launches' steps and the listed steps.
- **The third reason (answered).** Beside "data-order trace missing" the plan
  gives two reasons and `TRACE_MISSING_REASONS` holds those two. Three states of the
  join fit neither: a setting of the walk was not recorded, a shard's index file is
  gone or unreadable, the extra token is off, and the arrays of question 7 hold the
  recorded digests but have a shape no array of their kind has. They carry four words
  of their own, `TRACE_MISSING_RECORD_REASONS` (section 3, docs/interfaces.md section
  2), stored at registration, so that an answer never hides a reason the tool knows.
  The words are stored, not derived when an answer is made: a shard file moved or
  changed after registration would otherwise change the reason an answer gives.
- **The wrapper's side of the ruling.** The shuffle digests are computed by the wrapper
  in the run's own environment with its own call to torch, as section 6 says. A
  run record whose digests were taken in another way (for example from a
  different torch generator, or with the run's seed in place of the epoch) is a
  "cyclic order digest mismatch" and no answer.

## 12. Not built, and limits

- The cyclic sampler without data sharding, and Megatron-Bridge's sampler file,
  are walked only if a store lists them as verified; the first is refused by the
  join whatever the store lists.
- A blend with weights and a split that gives documents to validation are
  outside what the fixture verifies. The cache-files route reads their arrays
  like any others; the replay states the rule for a blend without weights only,
  and its digests will differ for anything else.
- The replay's blend and sample index run in Python loops and numpy calls sized
  for a laptop corpus. Nothing is measured at scale (plan RK3).
- All traces of a segment are held in memory and written in one transaction.
- A cyclic run's shuffle is drawn for each epoch a launch reads in, in Python
  lists of the bucket's size (a cost of about the dataset's length per epoch).
- The shuffle is exact only as long as torch's `randperm` draws what it drew when
  the wrapper took the digest: another torch build that draws differently shows
  as "cyclic order digest mismatch" and supports no answer.
- The replay was compared with Megatron's arrays on one machine. Another
  machine's arithmetic in the blend, or another numpy build, shows as a digest
  mismatch and is not otherwise handled.
- The join verifies sample positions against Megatron's dataset and sampler
  code. It observes no trainer (plan F9, as amended).
