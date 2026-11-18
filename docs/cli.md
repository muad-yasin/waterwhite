# The command line

```
waterwhite <command> --store <file> [--format json|text] ...
```

Code: `src/waterwhite/cli.py`. Tests: `tests/coverage/test_coverage_listing_cli.py`, `tests/coverage/test_show_maps_and_drops.py`,
for `register-run` `tests/coverage/test_register_run_command.py`, for `show --trace` and
`replay` `tests/coverage/test_show_trace_and_replay.py`,
for `recover` `tests/tokenmap/test_cli_recover.py` and, for `report` and `verify`, the
files of `tests/rebuild/`. The rules behind the
answers are in `docs/coverage.md`, those behind `recover` in `docs/recovery.md`
and `docs/tokenmap.md`. Plan sections served: 2.1 (answers, listings), 2.3
(requests, refusals), 3.3 and 5.4 (recovery), 3.4 (the loud checks), 4.1 to 4.4 (report and
verification).

The program is installed as `waterwhite`. It also runs as
`python -m waterwhite.cli`.

## What every command does

- It prints one JSON document on standard output. With `--format text` it prints a
  few lines for a person in its place.
- `--store FILE` names the evidence store. It may stand anywhere after the
  command.
- Nothing is printed before the command has finished. The output is built whole
  and written once.
- An error prints one JSON object on standard error and nothing on standard
  output:

  ```json
  {"error": "source not registered", "detail": "no member is registered under ...", "context": {...}}
  ```

  `error` is the fixed name of the error, a class of `waterwhite.errors`.
  `detail` says what and where; its wording is not fixed. `context` holds the
  values the error was raised with (for example the source, the member, the
  path), and is an empty object when there are none. With `--format text` the
  error is one line, `error: <name>: <detail>`.
- An unknown command, an unknown flag, a shortened flag, a missing argument and a
  missing `--store` are usage errors: the error name is "usage" and `context`
  holds the usage line where the parser gives one.
- `--help` prints the help text and exits with 0.
- `query`, `show`, `check`, `report` and `verify` open the store read-only. `replay` opens none and takes no
  `--store`. A path that holds no
  store (no file, a file that is not an SQLite file, a file with no schema of
  this tool) is refused with "not a store"; a store written by a newer build
  with "store of a newer schema version". Both exit with 2. No file is made.
- `recover` writes to the store. With `--dry-run` it opens the store read-only,
  prints its result and writes nothing.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | answered. "may contain" and "not covered" are answers |
| 1 | usage (error name "usage"); for `recover map` also "tokenizer failed" |
| 2 | the request was refused: a named error, no answer |
| 3 | a loud integrity fault: the mismatch is named, no answer |
| 4 | a write was refused |
| 5 | `verify --require-held` only: the verification was made and printed, and its conditions are not all held |

The flags every command takes:

| Flag | Meaning |
|---|---|
| `--store FILE` | the evidence store; required |
| `--format json` or `--format text` | JSON (the default) or lines for a person; it also sets the form of an error |

## init

```
waterwhite init --store work/evidence.db --normaliser control-chars/1
```

Creates a store. It lists the joins the tool ships as verified (below), or, with
`--verified-join`, only the ones of them that the flag names.

- `--normaliser NAME/VERSION` is required. It names the function a record's text
  passes before it is fingerprinted, and the store keeps it for good. There is no
  default. An unknown name is refused (exit 4, "unknown normaliser") and no file is
  made.
- The joins the tool ships. A store made with no `--verified-join` lists all the joins
  that plan Check 1 verified (plan 3.5, F6; DECISIONS.md D-26): Megatron release
  `core_v0.20.0`, with each of the two sampler types a run can be read with and the
  sampler class Megatron builds for it: `single` with
  `megatron.training.datasets.data_samplers.MegatronPretrainingSampler`, `cyclic`
  with `megatron.training.datasets.data_samplers.MegatronPretrainingRandomSampler`.
  The list is the tool's, a literal of `waterwhite.shipped_joins`, and is not read
  from any file beside the store. A checkpoint of a run outside the list answers
  "may contain", naming the version or the sampler type.
- `--verified-join TAG,TYPE,CLASS` names one join of the shipped list, and may be
  given more than once: a Megatron release tag, sampler type and sampler class. Given,
  the joins named are the whole list of the store and the rest of the shipped list
  is not written; the flag can only narrow the list: it cannot add a join and cannot
  leave the list empty. Each is three parts with commas between; anything else is a
  usage error. A join that is not one of the shipped joins, whatever part differs (the
  release, the sampler type, the sampler class, or a right class under the other
  sampler type), is a usage error too (exit 1, error name "usage", nothing on standard
  output, no file made), and one such join refuses the whole command, whatever other
  joins it names. The list is written when the store is made and at no other time.
- A store made through the library (`Store.create`, docs/store.md) lists the joins its
  caller gives and nothing else. When that list holds a join outside the shipped list,
  the store is outside what the tool verified: the tool's answers on it rest on the
  caller's word for the join, not on Check 1.
- A path that already holds a file, whatever the file is, is refused (exit 4,
  "a file already exists at the store path"). The file is left as it was.
- A folder that does not exist is refused (exit 4, "write refused").

Output: `{"created": <path>, "schema_version": 4, "normaliser": ..., "verified_joins": [[tag, type, class], ...]}`.

## compare-hand-off

`waterwhite compare-hand-off --job NAME --texts FILE --store FILE`: shows a stored job the
texts it was handed and marks its harness log compared (docs/store.md, "The write methods").
`FILE` is a JSON object `{"texts": [{"record": "ARTIFACT:L<n>", "text": "..."}]}` with one text
for every record handed to the job, as the job was handed it, and none for any other. Each text
is compared with the record by fingerprint after the store's normaliser; a record that holds no
text takes only a text that is empty after the normaliser. Until it is done, a log registered
without its texts, and every log of a store migrated from schema version 3, establishes no
bound: a negative past its edge reads "may contain", naming "upper bound not established".
Output: `{"job": NAME, "references": <n>, "hand_off_checked": true}`. A text that is not its
record, a missing or an extra record, a job with no stored log, a log with an empty hand-off
and a texts file that is not of this form are refused ("invalid record", exit 4) and change
nothing; a refusal names the job and the record and never a text. The in-process harness's
log establishes no bound whatever is compared. What the comparison shows is that each text
given is the record its reference names; that the job was handed those texts is the team's
statement, as it was at registration.

## query withdraw

```
waterwhite query withdraw <source> --store FILE
        [--member MEMBER ...] [--record ARTIFACT:LOCATOR ...] [--artifact NAME]
```

Withdraws a source and answers.

- `<source>` is the source's name. The request covers every member registered
  under it when the query runs. The answer lists them.
- `--member` narrows the request to the named members. May be given more than
  once.
- `--record` narrows it to single records of the source, each written
  `ARTIFACT:LOCATOR`: a name of a registered JSONL file and a line, for example
  `src/A:L3` (lines count from 1). May be given more than once.
- `--artifact NAME` asks about one artifact and prints one answer. The artifact is
  named by a name, an instance id, a content digest or a path. A digest or path
  that matches more than one instance is refused ("ambiguous artifact").
- Without `--artifact` the command prints the listing: what the store covers, one
  answer for every registered artifact (source files and the files between
  source and release among them; an artifact with no record of its own answers
  "not covered"), and in `not_covered` every artifact that has no record of its
  own, registered or only named by another record.
- `--selector KEY=VALUE` stands for a selector on metadata, such as a URL prefix.
  This build supports none: the request is refused with "unsupported selector".

Before it answers, the query runs the store's loud checks: over the ancestor
closure of the artifact it asks about, or over the whole store for a listing. A
fault exits with code 3 and prints no answer. A query about an artifact whose
closure does not hold the fault still answers.

One answer:

```json
{
  "artifact": "E@9",
  "instance": "...",
  "kind": "checkpoint",
  "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
  "answer": "contains",
  "reason": null,
  "names": null,
  "names_detail": null,
  "failing": null,
  "candidates": [],
  "chain": [{"hop": "E/seg-2:step 7", "segment": "E/seg-2", "step": 7, "link": "consumed"},
            {"hop": "E/shard-3:D4", "artifact": "E/shard-3", "instance": "...", "record": "D4", "link": "recovered"}],
  "basis": "whole-sample read assumed (F9, L9)",
  "notes": ["the parent list is process evidence ..."],
  "untraced_steps": [["E/seg-2", 8]],
  "first_exposure": {"segment": "E/seg-2", "step": 7},
  "level": "consumed",
  "scope": null
}
```

Every answer has every one of these keys, in this order. They are described in
`docs/interfaces.md`, section 6, and `docs/coverage.md`, section 2.

The listing:

```json
{
  "request": {"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []},
  "covers": {"runs": ["E"], "mixtures": ["E/mix-1"], "shards": ["E/shard-2"],
             "checkpoints": ["E@6"], "releases": ["R1"]},
  "answers": [{"artifact": "src/A", "answer": "contains", "...": "..."},
              {"artifact": "E/shard-1", "answer": "not covered", "reason": "record missing", "...": "..."}],
  "not_covered": [{"artifact": "E/shard-1", "instance": "...", "kind": "shard",
                   "reason": "record missing",
                   "named_by": [{"relation": "mixture shard", "referrer": "E/mix-1", "position": 0}],
                   "detail": "it has no tokenization map"}]
}
```

`answers` holds one answer for every registered artifact, each in the form above:
JSONL files first, then shards, mixtures and checkpoints, and inside a kind the
order of registration. `docs/coverage.md`, section 9, describes the listing.

The same answer with `--format text`:

```
E@9 (checkpoint) under withdraw A (members: A/r1)
  contains
  chain: E/seg-2:step 7 <- E/shard-3:D4 <- ...
  basis: whole-sample read assumed (F9, L9)
  first exposure: E/seg-2 step 7 (consumed)
  untraced steps: E/seg-2 step 8
  note: the parent list is process evidence ...
```

Refusals (exit 2, no answer):

| What was asked | `error` |
|---|---|
| a source with no registered member | source not registered |
| `--member` that is not a member of the source | member not registered |
| `--record` that no covered member holds | record not in source |
| `--record` not written `ARTIFACT:L<n>`; `--selector` | unsupported selector |
| `--artifact` given as a digest or path that several instances share | ambiguous artifact |
| a store file that is not there or is not a store | not a store |
| a store written by a newer build | store of a newer schema version |
| With the generating-model switch on, the answer for the artifact would rest on a cycle through a model's own history and no recorded "contains" wins; nothing is printed | temporal ancestry cycle |

The plan names "source not registered" for a withdrawal (plan 2.3); "member not
registered" and "record not in source" are the build's additions, which plan 5.8,
B-7 leaves to the builder (DECISIONS.md D-26).

A loud fault (exit 3) names one of "map count mismatch", "map digest mismatch",
"edge parent missing", "trace sample count mismatch", "artifact digest mismatch",
"stored digest mismatch", "locator domain mismatch", "cache digest conflict".

## query ban, query flagged

```
waterwhite query ban <text> --store FILE [--artifact NAME]
waterwhite query flagged <file> --store FILE [--artifact NAME]
```

A content ban across origins and a request that starts from flagged records are
extensions, and they are off. Both commands are read and refused with "extension
off" (exit 2). No answer is ever printed for them in this build: with the
extension switched on in the store, the refusal is "request refused" (exit 2),
because this build holds only the off state.

## show

```
waterwhite show <artifact> --store FILE [--record LOCATOR]
waterwhite show <shard> --store FILE --map NAMESPACE/DERIVATION [--verification RESULT]
waterwhite show <segment> --store FILE --trace [--step N]
```

Prints what the store holds about one artifact: its instance id, kind, role,
names, content digest, edge digest (the digest of the edges its step stated,
which is part of its instance id, plan 3.1), stored path and whether the file
is there, its producing step and inputs, its own record (the bindings with their
declared count and digest and, for a map, its derivation, verification, label and
end-of-document settings), the drops whose output it is, and the records that name
it. A mixture also shows its snapshot, a shard the mixtures that hold it, a
checkpoint its segment, step and history.

Every entry of `bindings` carries `append_eod` and `eod_in_span` (the recorded
`--append-eod` setting and whether the end-of-document token lies inside a
document's span, F1): true or false for a tokenization map, null for a binding that
is no map.

`drops` is a list for every registered artifact, empty when there are none: one
entry `{"artifact", "instance", "record", "reason", "step"}` per drop whose output
is this artifact, that is, per line the step that wrote it removed. `artifact`,
`instance` and `record` name the dropped line (`record` is `L<n>`). A JSONL file
carries the drops of the sanitizer that wrote it, a shard those of its tokenizer
(the lines that give no tokens).

For a name the store has no row for it prints
`{"artifact": ..., "registered": false, "reason": "record missing" | "not registered", "named_by": [...]}`
and exits with 0.

`--record LOCATOR` shows one record: `L<n>` a line of a JSONL file, with its
fingerprint, the members that cover it, its edge (label, derivation, parent mode,
parents, the selected parent, records in scope only, the sources of its parents,
and for a generation edge the job, the model entry, the stored bound field and the
bound as the stored log gives it), its source links and its drops; `D<n>` a
document of a shard, with its map row (label, parent or candidates, the
candidates' sources). Under `--record` a drop is printed as
`{"reason", "step", "output"}`: `output` is a name of the artifact the step wrote
that lacks the record (the `drops` of an artifact, above, is the same fact seen
from the output).

The edge of a record is printed as
`{"transform", "parent_mode", "derivation", "verification", "label", "parents", "selected_parent", "in_scope_only", "candidate_sources", "is_generation", "job", "model", "harness", "harness_log", "bound_field", "bound_from_log"}`.
`job` is the name the edge itself keeps: it is there for a generation edge also
when its harness log is gone, and null for any other edge. `harness_log` says
whether the log is stored; `model`, `harness` and `bound_from_log` are null when
it is not. `harness` is the harness the log names, "in-process" or "confined".

With `--record`, these are refused with "request refused" (exit 2): an artifact
the store has no row for; a record the store does not hold; a shard with no map,
or a document its map does not hold; a locator that is not `L<n>` for a JSONL
file or `D<n>` for a shard.

`show` reports and decides nothing. It does not run the loud checks, so it can be
used on a store that `check` faults. Reading one record can still meet a parent
that is gone, and then exits with 3 ("edge parent missing"). An artifact given as
a digest or path that several instances share is refused ("ambiguous artifact").

### show --map

`--map NAMESPACE/DERIVATION` prints the rows of that map of a shard, for example
`capture/position`, `recovery/position` or `recovery/token-matching`. A shard can
hold several maps (docs/store.md, "Tokenization maps and the reader's order"), and
`--record` reads only the one a reader gets first; `--map` reads the one it names,
so a recovery map beside a capture map can be read.

```json
{"artifact": "shard-1", "instance": "...", "kind": "shard",
 "map": {"namespace": "recovery", "derivation": "position", "verification": "passed",
         "label": "recovered", "declared_count": 4, "append_eod": true, "eod_in_span": false},
 "rows": [{"record": "D0", "label": "recovered", "token_start": 0, "token_end": 1,
           "parent": {"artifact": "...", "instance": "...", "record": "L1", "members": ["..."]},
           "candidates": []}]}
```

Row `n` is `D<n>`, in ordinal order. `parent` and `candidates` are printed as
`show --record D<n>` prints them (an object with `artifact`, `instance`, `record`
and `members`; `parent` is null for a "candidates" or "unknown" document, and
`candidates` is empty for a "recovered" one).

Where the namespace and derivation hold a map that did not verify beside a
verified one (DECISIONS.md D-28 as amended), the verified one is printed, as a
reader would take it; `map.verification` says which. The output then also holds
`other_maps`, one entry per other map of that namespace and derivation, each
`{"namespace", "derivation", "verification", "label", "declared_count",
"show_with"}`, where `show_with` is the flags that print that map (the result's
word in quotes where it has a space: `--verification "not run"`), or null where the
flags cannot name it. The key is absent when there is no other, so the output has
five keys in the usual case and six in this one. `--verification passed|failed|"not run"` names the
result of the map to print; it is a usage error without `--map`.

Refused with "request refused" (exit 2), the detail saying which: an artifact the
store has no row for; an artifact that is no shard; a shard with no map, or one
that holds no map of the named namespace and derivation (and verification); for
the last, `context.held` lists what the shard holds as
`NAMESPACE/DERIVATION (RESULT)`. A `--map` that is not two non-empty words
joined by one `/`, `--map` together with `--record`, and `--verification` without
`--map` are usage errors (exit 1). Where a namespace and derivation hold several
maps with one result that is not "passed", `--verification` prints the first that
was written; the others are listed in `other_maps` with a null `show_with`, since
the flags cannot name them.

### show --trace

```
waterwhite show <segment> --store FILE --trace [--step N]
```

Prints the rows the exposure join wrote for a registered run segment, in reading
order (plan 3.5; docs/join.md). Read-only. The name is the segment's, as
`register-run` took it from the run record.

A segment read with the sequential sampler:

```json
{"segment": "E/seg-2", "route": "cache files", "cause": null, "cause_detail": null,
 "launches": [{"position": 0, "world_size": 1, "data_parallel_size": 1,
               "consumed_at_start": 0, "step_at_start": 0, "first_step": 1, "last_step": 8}],
 "steps": [{"step": 7, "consumed_samples": 28,
            "samples": [{"sample_pos": 0, "global_index": 24, "rank": 0,
                         "sample_tokens": 33, "covered_tokens": 33,
                         "rows": [{"shard": "E/shard-2", "shard_position": 1, "ordinal": 4,
                                   "token_start": 0, "token_end": 21}]}]}],
 "untraced_steps": [["E/seg-2", 8]]}
```

A segment read with the cyclic sampler, whose step 2 reads on both sides of an
epoch boundary (the samples and rows are left out here):

```json
{"segment": "K/seg-1", "route": "cache files", "cause": null, "cause_detail": null,
 "launches": [{"position": 0, "world_size": 2, "data_parallel_size": 2,
               "consumed_at_start": 0, "step_at_start": 0, "first_step": 1, "last_step": 2}],
 "steps": [{"step": 1, "consumed_samples": 4, "epochs": [0], "samples": ["..."]},
           {"step": 2, "consumed_samples": 4, "epochs": [0, 1], "samples": ["..."]}],
 "untraced_steps": []}
```

- `route` is "cache files", "replay" or "none"; with "none", `cause` and
  `cause_detail` are the segment's stored cause and reason, and `steps` is empty
  (a segment with no trace route has no rows to print). Otherwise both are null.
  Beside "data-order trace missing" the reason is one of the six words of
  `TRACE_MISSING_STORED_REASONS` (docs/interfaces.md section 2), or null.
- `launches` holds the run record's launches as registered, in position order, each
  with `position`, `world_size`, `data_parallel_size`, `consumed_at_start`,
  `step_at_start`, `first_step` and `last_step`. It is `[]` when the record holds
  none, and it is the same whatever the route and whatever `--step` keeps. A
  step's place in the data is its launch's `consumed_at_start` plus the batch sizes
  of the launch's steps before it, never its step number (a resume carries the
  counter on, a warm start restarts it at 0).
- `steps` holds one entry per traced step in step order, `samples` one per sample in
  `sample_pos` order, `rows` one per row in the order the join resolved them. The
  fields of a sample and of a row are those of a stored trace (docs/store.md):
  `covered_tokens` below `sample_tokens` is a sample with an uncovered token range
  (the rows cover the first `covered_tokens` tokens only). `shard` is the name the
  shard at `shard_position` is registered under in the segment's mixture; where the
  shard has no name of its own it is the path the snapshot gives, and it is null
  where the segment has no mixture or the snapshot has no shard at that position.
- For a segment read with the cyclic sampler each step also holds `epochs`: the
  epochs its samples are drawn from, in order. One epoch is the usual case; a step
  whose rounds lie on both sides of an epoch boundary has two. They are placed as
  the join places the step (docs/join.md section 5): the launch's counter plus the
  batch sizes of the launch's earlier steps, on the launch's sizes, which are taken
  from the run record's shuffle digests (the count of a launch's digests is the
  length of one rank's order in an epoch, and an epoch holds that many samples for
  each data-parallel rank). `epochs` is null for a step of a launch whose digests
  give no count. A segment read with the sequential sampler has no `epochs` key.
- `untraced_steps` lists, as `[segment, step]`, the steps the proof reads as
  untraced (plan 2.2(e)): the same rule a checkpoint at the segment's last step
  applies, with the route, the cyclic order and the launches taken into account
  (docs/coverage.md section 8), so it can name steps `steps` shows (a trace with an
  uncovered range).
- `--step N` keeps that step only, in `steps` and in `untraced_steps`; a step with
  no trace gives an empty `steps`.

| Exit | When |
|---|---|
| 1 | `--trace` with `--record`, `--map` or `--verification`; `--step` without `--trace` |
| 2 | "request refused": the name is no registered run segment |

## check

```
waterwhite check --store FILE
```

Runs the loud checks over the whole store: a map or occurrence record whose count
or digest does not match its artifact, an edge whose parent is gone, a trace whose
sample count differs from its step's batch size, an artifact whose file no longer
has the registered digest. The first fault found exits with code 3 and its name.

With no fault it prints
`{"check": "passed", "artifacts": <n>, "missing_files": [...]}`. `missing_files`
lists artifacts whose file is not at its recorded place. That is a state, not a
fault: the records still answer, and a negative answer names such files in its
scope.

## migrate

```
waterwhite migrate --store FILE
```

Brings a store made by an earlier build to this build's schema version, one step
at a time, each step one transaction (plan 3.4; docs/store.md). Every other
command that reads a store opens it read-only and refuses one that needs a step,
naming this command. A store already at this version is left as it is. It prints
`{"schema_version_found": <n>, "schema_version": <m>, "steps_run": <m - n>}`, and, when it
ran steps, `"migrated": {"from", "to", "note"}`: from version 4 on the note says that every
stored harness log now reads as not compared, so that a clean answer past a generation step
reads "may contain" until `compare-hand-off` is given the texts. Any other command that opens
a store for writing and has to migrate it adds the same `"migrated"` entry to its output. The
refusals are those of opening a store: "not a store", "store of a newer schema
version". A step can refuse a store it cannot bring forward: step 3 refuses one that
holds two rows for one array of a segment's blend with "cache digest conflict" (exit 3,
docs/store.md "Opening a store"), and the store is left as it was. Exit 0 when the
store is at this version afterwards.

## register-run

```
waterwhite register-run --store FILE --run-record PATH [--steps PATH] [--cache DIR]
                        [--mixture NAME] [--parent-checkpoint NAME]
```

Registers ONE run segment, in one store call and one transaction (a stop leaves
nothing half written): the run record whole, its launches and digests, its
completed steps, its checkpoints (from the step record) and the segment's step
traces, which the exposure join makes (docs/join.md, plan 3.5).

| Flag | What it is |
|---|---|
| `--run-record PATH` | the run record the training wrapper wrote (docs/interfaces.md 8.1). It names the run, the segment and the start kind; they are taken from it. A shard prefix in it is relative to the record's folder |
| `--steps PATH` | the operator's step record (8.2): per step a batch size and the tokens read of its last sample, and the checkpoints `{name, step}` to register |
| `--cache DIR` | the data-order cache folder the run kept (8.4). Without it, or where its files do not have the recorded digests, the join tries the replay |
| `--mixture NAME` | the mixture the segment trained on, by the name it is registered under. A name that is no registered mixture is refused. Without it the segment has no snapshot, and a checkpoint of it answers "may contain" naming "mixture snapshot missing" |
| `--parent-checkpoint NAME` | the checkpoint a resume or warm start goes on from, by name; it must be registered. Defaults to the run record's own `parent_checkpoint`; the two must agree when both are given |

The position of the segment inside its run is not a flag: a segment already in the
store keeps its position (so the same registration again changes nothing), a new
one takes the next free position of its run.

The trace route follows from the run record and the files it names and from
nothing else: no flag states it. "cache files" when every index array has the
digest the run record holds; "replay" when the arrays derived again from the
record's settings have them; "none" with one of the five named causes of
docs/interfaces.md section 2 otherwise (docs/join.md section 3 gives their order).
A segment read with the cyclic sampler needs torch (docs/join.md section 6): the
join computes each epoch's shuffle by calling it, and the order counts only when
its digest and count equal the recorded ones. A cyclic run whose record holds no
shuffle digest is registered, with the cause "cyclic order not recorded", without
torch.

It prints, and exits 0:

```json
{"segment": "E/seg-2", "route": "cache files", "cause": null, "cause_detail": null,
 "steps": 6, "traced_steps": 6,
 "untraced_steps": [["E/seg-2", 8]], "checkpoints": ["E@9"],
 "replay": {"attempted": false, "accepted": null, "reason": null},
 "shuffle": []}
```

and, for a segment read with the cyclic sampler whose digests the join held against
the orders it computed:

```json
{"segment": "K/seg-1", "route": "cache files", "cause": null, "cause_detail": null,
 "steps": 4, "traced_steps": 4, "untraced_steps": [], "checkpoints": ["K@2"],
 "replay": {"attempted": false, "accepted": null, "reason": null},
 "shuffle": [{"launch": 0, "epoch": 0, "rank": 0,
              "digest": "a50c12ec...", "count": 3,
              "recorded_digest": "a50c12ec...", "recorded_count": 3, "equal": true},
             {"launch": 0, "epoch": 0, "rank": 1,
              "digest": "455e1e52...", "count": 3,
              "recorded_digest": "455e1e52...", "recorded_count": 3, "equal": true}]}
```

`cause` and `cause_detail` are the segment's stored cause and reason when the route
is "none", and null otherwise. Beside "data-order trace missing" the reason is one of
the six words of `TRACE_MISSING_STORED_REASONS` (docs/interfaces.md section 2): the
plan's two, "no recorded digest" and "digest mismatch", and the four the join adds
where it knows what the record, or a file it names, lacks, "run record incomplete" (a
setting the walk needs or a launch is not recorded), "shard index unreadable" (a
shard's index file is gone or cannot be read), "extra token off", and "index array
malformed" (a kept cache file with the recorded digest holds an array of a shape no
array of its kind has). A run record that holds no launch is
registered with the route "none", the cause "data-order trace missing" and the reason
"run record incomplete".

`replay` says what became of the replay route (docs/join.md section 4).
`attempted` is false when the cache files matched, and also when no replay could
be held against a recorded digest; `accepted` is null when it was not attempted,
else true or false; `reason` is null, or the reason of docs/interfaces.md section
2 when the replay was refused ("digest mismatch") or not tried for want of a
digest ("no recorded digest"). A segment that never reached the question of the
arrays (an unverified release, a cyclic run with no shuffle digest) has the
default: not attempted, null, null.

`shuffle` is `[]` for a segment read with the sequential sampler. For a cyclic one
it holds one entry for each (launch, epoch, rank) the launches read, in that order.
`digest` and `count` are those of the order the join computed (`ids.order_digest`
of the whole order and its length), the very order held against the record: the
command does not compute it a second time. `recorded_digest` and `recorded_count`
are the record's (`recorded_count` is null where the record gives none), and
`equal` is true when the digests are the same and the count is too, or is not
recorded. A mismatch is listed with `equal` false, beside the segment's cause
"cyclic order digest mismatch". Where the join computes nothing, `shuffle` is `[]`:
the record holds no shuffle digest, or lacks one the launches need ("cyclic order
not recorded"), or an earlier question of docs/join.md section 3 stopped the join.
Where torch cannot be imported, the registration is refused as above.

`steps` counts the registered completed steps and `traced_steps` those with a
trace. `untraced_steps` lists every step that plan 2.2(e) counts as untraced: all
steps when the route is "none"; else the steps with no trace, the steps whose
trace has an uncovered token range (the operator's note lowered a sample) and the
steps the registration left out because nothing gave their batch size.

| Exit | When |
|---|---|
| 1 | a run record or step record path that cannot be read ("usage"); the arguments of the command |
| 4 | "invalid record": a run record or step record that is malformed (not JSON, not an object, not of the known format, a field of the wrong type, a run record that names no run, segment or start kind, a step record written for another segment), or a registration the store refuses (a mixture or parent checkpoint the store does not hold, a trace or a checkpoint the store cannot tie to the segment, such as a checkpoint at a step the segment does not have, a segment name or position that is taken by other contents); "torch not available": a cyclic run whose shuffle must be computed on a machine where torch cannot be imported. Nothing is written in any of these cases |

"invalid record" exits with 4 wherever it is given, the exit code of its family
(docs/interfaces.md section 2).

## replay

```
waterwhite replay --run-record PATH --out DIR
```

Derives a segment's data-order index arrays again from its run record and its
shards (the replay route of the join, docs/join.md section 4) and writes each
array to `DIR` as `<kind>.npy` for the blend's two and `<kind>-<shard position>.npy`
for a shard's three, as the replay made it (dtype and shape untouched). It opens no
store and writes nothing else. `DIR` is made when it is not there. It prints, and
exits 0:

```json
{"segment": "E/seg-2",
 "arrays": [{"kind": "document_index", "shard": 0, "file": "DIR/document_index-0.npy",
             "digest": "...", "recorded_digest": "...", "equal": true}]}
```

`shard` is null for the blend's arrays. `digest` is `ids.array_digest` of the
written array, `recorded_digest` the digest the run record holds for that kind and
shard position (null where it holds none), and `equal` whether the two are the same,
null where none was recorded. The command compares digests only. Whether the bytes
written equal a kept cache file is for whoever holds both files to find out; a
digest that is equal says so, and one that is not says the replay does not give the
arrays the run was trained on (plan 3.5, Check 2 (i)).

| Exit | When |
|---|---|
| 1 | "replay not possible": the run record lacks a setting the replay needs (docs/interfaces.md section 2), lists no shard, or names a shard whose index file is not at the place the record gives, cannot be read or is not the file the record's digest names. "usage": a file the replay would write is already in `DIR`; nothing is written then, and a written array is never overwritten; a run record path that cannot be read |
| 4 | "invalid record": the run record is not JSON, not an object, not of the known format or holds a field of the wrong type |

## recover

`recover` registers files a run left behind and derives the links of steps
nobody observed (plan 3.3, 5.4). It has five sub-commands. A run made without
capture is recovered in this order: the files, then the copy steps and the
sanitizer, then the shards and their maps. The source files are ingested from
Python (`capture.ingest_jsonl`); the command line has no ingest.

Every sub-command takes `--dry-run`: the store is opened read-only, the result is
printed and nothing is written. An artifact is named by a registered name or by
its instance id.

Exit codes of `recover`:

| Code | When |
|---|---|
| 0 | the command ran and printed its result. A recovery that found no link, and a map whose verification failed or was not run, are results: read `verified`, `outcome` and the counts |
| 1 | a command line that cannot be carried out ("usage"): a missing flag, both tokenizer flags, a mode that needs a tokenizer and has none, flags of a log with no log, a value that is not JSON. Also "tokenizer failed": the tokenizer command could not be started, ended early, gave no reply in time or answered with something that is not token ids; the entry point could not be imported or raised |
| 2 | "not a store"; "store of a newer schema version"; an `--input` of `recover shard` that is not registered ("request refused"); a shard whose index is not the documented layout ("unsupported format") |
| 3 | a registered file that no longer has the digest it was registered with ("artifact digest mismatch"); a file whose line count differs from the store's records ("map count mismatch"); a record that no longer matches the digest the store holds of it ("stored digest mismatch"); records not numbered as the binding declares ("locator domain mismatch"); two digests for one array of a blend ("cache digest conflict") |
| 4 | a write the recovery or the store refuses: "invalid record" (a file or shard that is not registered, a name that is taken, a log that does not fit, an edge or a verified map that would be replaced, settings that contradict each other), "unknown normaliser", "declared count mismatch", "declared digest mismatch", "drop already recorded" |

Nothing is written when a command exits with a code other than 0, with one
exception: `recover sanitizer` makes two writes, the edges and then the drops,
and a process that stops between them leaves the edges. Running it again adds
the drops.

### recover file

```
waterwhite recover file <path> --name NAME --store FILE
        [--role intermediate|release|import] [--text-field FIELD] [--dry-run]
```

Registers a JSONL file found afterwards, with no recorded producing step.

- `--name` is the name it is registered under, and part of its instance id: the
  same bytes under two names are two instances.
- `--role` is the place the file has in the pipeline: `intermediate` (the
  default: a chunk file, a sanitizer's input, a tokenizer's input), `release`, or
  `import` for a text-only import from outside the pipeline. A listing names a
  found release among the releases it covers.
- `--text-field` names the field of a line that holds its text (default `text`).

Output: `{"artifact", "role", "already_registered", "instance", "records", "content_digest", "dry_run"}`.
`already_registered` says whether the name was taken before the call. With
`--dry-run`, `instance` is null and `records` and `content_digest` are those of
the file as read.

Until a recovery links them, the records of a found file have no recorded origin,
and a query about the file answers "may contain" naming "missing origin".

### recover shard

```
waterwhite recover shard <prefix> --name NAME --tokenizer-version VERSION --store FILE
        [--input ARTIFACT ...] [--tokenizer-params JSON] [--dry-run]
```

Registers a tokenized shard found afterwards: the pair `<prefix>.bin` and
`<prefix>.idx`. It writes the artifact row only. The shard has no record of its
own, and answers "not covered", until `recover map` writes its map.

- `--input` names a registered file the tokenizer read; it may be given more than
  once. The inputs are part of the shard's instance id.
- `--tokenizer-version` and `--tokenizer-params` (a JSON value: the tokenizer's
  name, the revision of its files, its switches) make the step "tokenize" the
  shard is registered under. They are part of the instance id too.

Output: `{"artifact", "already_registered", "instance", "documents", "content_digest", "inputs", "has_map", "dry_run"}`.

### recover copy-step

```
waterwhite recover copy-step <later> --input ARTIFACT [--input ARTIFACT ...]
        --step-name NAME --store FILE
        [--step-version VERSION] [--step-params JSON] [--text-field FIELD] [--dry-run]
```

Recovers the links of one copy step (shuffle, chunk, merge) for one file it
wrote. `<later>` is that file and each `--input` a file the step can have read.
Name every one: a file left out is not searched. `--step-version` is "not
recorded" unless given.

Output, the fields of `RecoveryResult` (docs/recovery.md section 7):

```json
{"kind": "copy step", "step": {"name": "...", "version": "...", "param_digest": "..."},
 "artifact": "chunk", "instance": "...", "outcome": "record-level", "reason": null,
 "records": 5, "recovered": 3, "candidates": 1, "no_edge": 1, "no_text": 1, "surplus": 0,
 "collisions_rejected": 0, "drops": 0, "input_lines_unaccounted": null,
 "edges_written": 4, "drops_written": 0, "links_written": 0, "wrote_nothing": false,
 "members_linked": [], "files_read": ["chunk", "src/A", "src/B"], "dry_run": false}
```

`recovered + candidates + no_edge == records`. `surplus` counts, among `no_edge`,
the copies of a text the later file holds more often than the named inputs do:
an input was not named, and no copy of that text is given a parent. A second run
writes nothing (`wrote_nothing` true).

### recover sanitizer

```
waterwhite recover sanitizer <later> --input ARTIFACT --store FILE
        [--log FILE] [--log-input VALUE] [--step-name NAME] [--step-version VERSION]
        [--step-params JSON] [--text-field FIELD] [--dry-run]
```

Recovers the links of the sanitizer for the one file it wrote (`<later>`) from
the one file it read (`--input`), and records the lines its log names as drops
whose output is `<later>`.

- `--log` is the sanitizer's own log of skipped lines; `--log-input` is the value
  its rows carry in `input` for this file when the log covers several.
- `--step-name` is "sanitize" unless given. It is the step name a map recovered
  through a sanitizer's log gives the same drops.

Output: as for `recover copy-step`, with `kind` "sanitizer", `drops` and
`input_lines_unaccounted`. `outcome` is "record-level" in the normal case and
"file-level" or "unknown" when the store's normaliser is not available in the
process (docs/recovery.md section 6); `reason` then says why.

Refused with "drop already recorded" (exit 4): a log whose lines the store
already holds as drops for a shard. A map recovery read this log when only the
sanitizer's input was registered; the line is recorded once. Run the command
with no `--log` to record the edges alone.

### recover map

```
waterwhite recover map <shard> --input ARTIFACT --store FILE
        --append-eod yes|no --eod-in-span yes|no [--eod-token-id ID]
        [--mode position|token-matching|both]
        [--tokenizer-command COMMAND | --tokenizer-entry MODULE:FUNCTION]
        [--tokenizer-timeout SECONDS]
        [--sanitizer-log FILE [--log-shard VALUE] [--log-input VALUE]
         [--sanitizer NAME/VERSION] [--sanitizer-step NAME]]
        [--text-field FIELD] [--dry-run]
```

Recovers the tokenization map of a registered shard into the namespace
"recovery" (docs/tokenmap.md section 7).

- `--input` is the registered file whose lines became the documents: the file
  the tokenizer read, or, with `--sanitizer-log`, the file the sanitizer read.
- `--append-eod` and `--eod-in-span` are the two settings every map records.
  Neither has a default: a wrong one would be stored on the map. `--eod-token-id`
  is needed with `--eod-in-span yes`.
- `--mode position` (the default) builds the map by position. When its
  verification fails and a tokenizer is given, token matching takes over, as the
  plan fixes the order, and both maps are stored. `--mode token-matching` does
  not run position mode (the run is said not to keep plan F1). `--mode both`
  builds both so that they can be compared. The last two need a tokenizer.
- With no tokenizer flag nothing is compared: the map is "not run" and
  "unknown". A later run with the tokenizer writes its map beside that one.

**The tokenizer.** The tool ships no tokenizer and imports no tokenizer library.
The team's own tokenizer is named by one of two flags:

- `--tokenizer-command COMMAND`: an external command, started once, with no
  shell (the text is split into words as a shell would). It speaks the protocol
  `waterwhite.tokenizer/1` on its standard input and output, UTF-8, one JSON
  object per line. For each text the tool writes `{"text": "<the text>"}` and
  reads `{"tokens": [<token id>, ...]}`. The command answers each request before
  it reads the next, flushes each reply, and ends when its input is closed. The
  token ids are those of the text alone, without the end-of-document token. A
  text that gives no tokens is answered with an empty list. What the command
  writes to standard error is kept, and its end is quoted when the command
  fails. `--tokenizer-timeout` is how long one text may take (120 seconds unless
  given). A wrapper is a few lines:

  ```python
  import json, sys
  from team_pipeline import tokenizer          # the team's own
  for line in sys.stdin:
      tokens = tokenizer.encode(json.loads(line)["text"])
      print(json.dumps({"tokens": tokens}), flush=True)
  ```

- `--tokenizer-entry MODULE:FUNCTION`: a Python function of the team, imported by
  name in the tool's process (`package.module:object.method` is accepted). It
  takes a text and returns its token ids. The module must be importable; the
  tool imports that module and nothing else.

A command that exits with a status other than 0 after its last reply does not
fail the recovery: every reply was given, and the map rests on the comparison of
those replies with the shard. The result's `notes` say so.

- `--sanitizer-log FILE` says that `--input` is the file the sanitizer read. The
  lines the log names are drops and take no ordinal; every other line is passed
  through the sanitizer again before it is tokenized again. `--log-shard` and
  `--log-input` pick the rows of a log that covers several files. `--sanitizer`
  names the function to run again (the store's normaliser unless given), and
  `--sanitizer-step` the step the drops are recorded for ("sanitize" unless
  given). The drops are recorded with the shard as their output.

Output:

```json
{"artifact": "shard-1", "input_artifact": "clean", "mode": "position",
 "shard": "<instance id>", "input": "<instance id>", "documents": 4, "surviving_lines": 4,
 "verified": true,
 "maps": [{"namespace": "recovery", "derivation": "position", "verification": "passed",
           "label": "recovered", "append_eod": true, "eod_in_span": true, "documents": 4,
           "recovered": 4, "candidates": 0, "unknown": 0, "spans_match": true,
           "count_matches": true, "documents_compared": 4, "failures": 0, "reasons": [],
           "stored": true}],
 "drops": [{"record": "L3", "reason": "no tokens", "step": "tokenize"}],
 "notes": [], "dry_run": false}
```

`maps` holds one entry per mode that ran, without the rows (`show <shard> --record
D<n>` prints one). `verified` is true when one of the maps passed. `drops` are
recorded only when one did.

Refused with "drop already recorded" (exit 4), before anything is derived: a
`--sanitizer-log` when `--input` is itself the output of recorded sanitizer
drops (it is the file the sanitizer wrote; recover the map over it with no log),
and when a line the log names already has its drop under a registered file
(recover the map over that file with no log). A skipped line is recorded once
(docs/interfaces.md section 5.4).

## report

```
waterwhite report withdraw <source> --store FILE
        [--member MEMBER ...] [--record ARTIFACT:LOCATOR ...] [--selector KEY=VALUE ...]
        [--layout NAME=PREFIX ...] [--placement NAME=FILE ...]
waterwhite report ban <text> --store FILE
waterwhite report flagged <file> --store FILE
```

Prints the rebuild report of a withdrawal (plan 4.1, 4.3): for every release, import,
shard and mixture the action the team's own builder must take, for every record of an
artifact that is rewritten its own action and the reason, the counts, the restart
points of every run (the exposure-free restart of each run, and for each run that has
a stale segment a statement of its exact-rerun restart), the checkpoints that carry
exposure, and the checkpoints that answer "not covered" (`checkpoints_not_covered`).
Beside the artifacts it lists, in `upstream_files`, the source files, the
generation outputs and the intermediate files ahead of a release or a shard that the
request does not clear, with the answer, the action where there is one and the reason
(plan 4.1, BF-6): they are kept out of `counts` and `bytes_to_rewrite`. The tool executes
nothing: the report is what a builder is handed, and `verify` is what checks what the
builder made. The JSON form is in `docs/interfaces.md` section 7.1 and the rules
behind it in `docs/rebuild.md`.

- `<source>`, `--member`, `--record` and `--selector` are those of `query withdraw`
  and are refused in the same way.
- `--layout NAME=PREFIX` names a layout: the shards whose store name begins with
  `PREFIX`. May be given more than once. The store records no layout, so the groups
  `groups.by_layout` of the report are the ones named here, and none when none is
  named. A shard that two prefixes match is a usage error. `groups.layouts` says
  whether the per-layout count was made: without `--layout` it says it was not and
  names this flag; with layouts named it lists the shards that no prefix matches
  (plan 4.1, BF-7).
- `--placement NAME=FILE` states where the layout `NAME` places the records that return to
  a release (the records the report promotes, and the ones it re-sources a row onto). One
  per layout, and `NAME` must be a layout that `--layout` names: a placement for another
  name is a usage error. `FILE` is JSON:
  `{"form": "waterwhite.placement/1", "by": "source", "shards": {"<source artifact>":
  "<shard store name>", ...}, "otherwise": "<shard store name>" or null}`. "By source" is
  the only rule. A returning copy lands in the shard the file names for its source: the
  copy's own artifact where that is a registered source file (`src/B:L1` lands under
  `src/B`), else the source at the root of its chain of single-parent record-level edges,
  and where there is none, or the file names no shard for it, in `otherwise`; null says
  the place is not stated. A file that cannot be read, is not JSON or is not of this form
  is a usage error. Every shard the file names, a value of `shards` and `otherwise` where
  it is not null, must have a store name that begins with the prefix of the layout `NAME`;
  a file that names one that does not is a usage error that names the file, the layout, its
  prefix and the first such shard, and no report is printed. A shard inside the prefix that
  the store does not hold is not refused: it is a landing (below). The report states the
  file it read, path and SHA-256, under
  `groups.placements`; the store does not keep it. Without a placement the report does not
  say where a returning copy lands, and says so (below).
- `report ban` and `report flagged` are requests of extensions that are off, and are
  refused with "extension off" (exit 2), as the same words of `query` are.
- The report is computed from the store as it is when the command runs, and nothing
  is written; a team that wants to keep a report keeps the file the command printed.
- Like a listing, the command runs the loud checks over the whole store before it
  reports. One fault stops it (exit 3, no report).

Actions. A record is kept, re-sourced, promoted or retired, or it is unresolved and
has no action; an artifact is kept or rewritten, or it is unresolved, or, for a shard
or a mixture, it reads "clean, place not stated".

| Action | Of | Means |
|---|---|---|
| keep | a record, an artifact | the proof shows the withdrawn source is not behind it |
| rewrite | an artifact | it is rebuilt without the affected occurrences (a shard is re-tokenized) |
| re-source | a record | an "any" row whose chosen parent is withdrawn takes an equivalent parent that remains (`new_parent`, `withdrawn_parent`) |
| promote | a record | the dedup loser whose winner is withdrawn returns to the release |
| retire | a record | a descendant leaves the successor; `regenerate_with` and `step` name the generation job and step to run again |
| clean, place not stated | an artifact | its own contents are clean, and the records that return to a release may land in it: the place is not stated (`--placement`), so the report does not say "keep"; `returning` names the copies |
| unresolved | a record, an artifact | the answer is "may contain" or "not covered", or the only evidence is a file-level link; `missing_link`, `failing` and `review_scope` say what is missing and what a person must review |

A release row made by an "any" edge is re-sourced to the first equivalent parent that
remains, and the dedup loser the step dropped in its favour is promoted; where no
equivalent parent is shown to remain because one of them cannot be told, the row is
unresolved (`docs/interfaces.md` section 7.1). A record whose only evidence is a
publisher's pointer is unresolved like one whose only evidence is a file-level link.

Where a successor is to be named, the name is `successor of <artifact>`
(`docs/interfaces.md` 9.2).

Refusals (exit 2, no report):

| What was asked | `error` |
|---|---|
| a source with no registered member | source not registered |
| `--member` that is not a member of the source | member not registered |
| `--record` that no covered member holds | record not in source |
| `--selector`, or `--record` not written `ARTIFACT:L<n>` | unsupported selector |
| `report ban`, `report flagged` | extension off |
| `report ban`, `report flagged` with the extension switched on in the store | request refused |
| a store that is not there, not a store, or written by a newer build | not a store; store of a newer schema version |
| the store written by another process while the command ran | store written during the query |

A `--layout` that is not written `NAME=PREFIX`, a shard matched by two prefixes, a
`--placement` that is not written `NAME=FILE` or names a layout `--layout` does not, a
layout given two files, a placement file that cannot be read or is not of the form, and a
placement file that names a shard outside the prefix of its layout, are usage errors (exit 1). A loud fault of the store is exit 3.

The same report with `--format text` is a few lines for a person: the request, the
counts, one line per artifact with its action and successor name, and the restart
points, with a list of the files upstream, a line for a per-layout count that was
not made (no `--layout` given), a line for each shard that reads "clean, place not stated"
(its own contents are clean, the copies it may receive, their place not stated), the placement
file read for each layout and the shards outside the store that copies land in, and one line
under the restart of each run that has a stale segment that says
what the report states of its exact-rerun restart: that it is left to the
verification (`RESTART_EXACT_BY_VERIFY`), or, for a run whose record lacks a setting a
replay needs, or with a segment on a mixture that reads "clean, place not stated", the phase
start as an assumption with no step comparison possible. After the
restarts it has one line `exposed: NAME (ANSWER)` for each checkpoint in `exposed_checkpoints`
and then one line `not covered: NAME (REASON)` for each checkpoint in
`checkpoints_not_covered`. Its wording is not fixed; the JSON is the contract.

## verify

```
waterwhite verify --store FILE --report FILE [--sides FILE]
        [--mode seeded|unseeded] [--text-field FIELD] [--require-held]
```

Verifies a repair (plan 4.2, 4.4): that the repaired side holds what a from-scratch
build without the withdrawn content holds, that the artifacts the report keeps are
byte-identical afterwards, what the successors and the history answer, which index
caches are stale, and the exact-rerun restart. The command opens the store
read-only and writes nothing. The JSON form is in `docs/interfaces.md` section 7.3.

- `--report FILE` is the report the repair was driven by, as `report` printed it (or
  as it was changed, for a test of a wrong report). It is read as a file and not
  computed again.
- `--sides FILE` is the sides file (`docs/interfaces.md` 7.2): the repaired side's
  and the reference side's file for every artifact, by the artifact's historical
  name. Without it the comparisons of the two sides are "not compared" and the
  successors, the history, the caches and the restarts are still made. A file or shard
  the report keeps or rewrites that the sides file does not name on both sides is named
  in `differences`, and the comparison it belongs to is "not compared" and never "equal",
  unless the report's own counts show the repair leaves it no record (no row in the
  successor of a file, no document in that of a shard): it is then named in
  `differences` with that reason and does not hold the comparison back. A shard the
  report names under `groups.placements` as a landing (a shard the store does not hold
  that receives returning copies) is a shard the report names: the repair names it on
  each side of the sides file whose build wrote it, and where both sides name it it is
  compared like a rewritten shard. A shard that both builds wrote and the report does not name keeps both comparisons from reading
  "equal" too: they read "not compared" and the shard is named in `differences` with a
  note. Which shards each build wrote is read from the builds (the repaired store's
  `successor of` shards, and the shard files in the folder of the reference files the
  sides file names), never from the sides file's list of names. A shard only one build
  wrote is a difference and `per_shard_digests` reads "differ" (in the seeded mode, where a
  shard was compared).
- `--mode seeded|unseeded` says whether every step of the build was seeded. In
  "seeded" the per-shard digests are compared as well; the default "unseeded" asks
  for equality of the record multiset only (plan 4.2).
- `--text-field FIELD` is the field of a JSONL line that holds its text (default
  `text`), as in `recover`.
- `--require-held` makes the exit code say whether the verification holds (below,
  "Gating on the verification"). It changes nothing else: the document is the same.

- Which artifacts are compared is decided by the `kind` the report gives each one,
  not by the file name: a JSONL file by its multiset, a shard by its digest. A name
  that only one side names is listed in `differences` and not compared. A file the
  sides file names that is not there is refused before anything is compared.
- An artifact the report gives the action "clean, place not stated" is accepted. It is
  never checked for byte identity (a copy that returns may have changed it), needs no
  successor, and its files are compared like those of a kept or rewritten artifact; a
  mixture that reads it is read as one the report leaves unresolved, with a reason of
  its own.
- The report is read for `form`, `request`, each artifact's `artifact`, `instance`,
  `kind`, `action` and `successor` (null allowed), and `counts.records.unresolved`; a
  report without one of these is refused. Nothing else is read: `upstream_files` is not.
- A report is certified only against a store that holds what it names: an artifact entry
  whose `instance` the store does not hold is refused, and so is a request the store
  no longer expands as the report lists it (a member or a record it no longer holds). A
  faithful copy of the store, with the successors registered, is certified.

The same verification with `--format text` is a few lines for a person: the request,
the two comparisons, one line per kept artifact, successor and history entry, the stale
caches, one line for every cache entry the verification has no answer for (a mixture
the report leaves unresolved or gives the action "clean, place not stated", or rewrites
with no successor registered) with its reason, and the restarts. Its wording is not fixed; the JSON is the contract.

Without `--require-held`, a difference between the sides is an answer and exits with 0:
`multisets` and `per_shard_digests` say "differ" and `differences` names the records. Equality is
claimed only for a report with no unresolved record; otherwise `multisets` says "not
claimed" (plan 4.2). The answers of the successors are those `query` gives; the tool
builds none of them.

Refusals (exit 2): the refusals of `query withdraw` for the request the report holds;
"request refused" for a report or sides file that is JSON but not a document of its
form, for a file the sides file names that is not there, and for a report that names
an instance the store does not hold; "store written during the query"; "not a store". A report or sides file that cannot be read is "usage"
(exit 1). A loud fault in the closure of an artifact it answers is exit 3.

### Gating on the verification

Exit 0 says that `verify` made its verification, not that the repair holds: "differ",
"not compared" and "not claimed" are answers and exit with 0. A script that gates on the
exit status alone therefore continues after a difference. Two ways to gate are
supported.

1. Read the document. `multisets` is "equal", and in `--mode seeded` so is
   `per_shard_digests`, for a repair that holds; the other conditions below are keys of
   the same document, and the report says which artifacts are files and which are shards.
2. Add `--require-held`. The document is printed on standard output exactly as without
   the flag, and the exit code is 5 when any condition below is unmet, 0 when all hold.
   On standard error one more line names what is unmet: with `--format json` the object
   `{"verification": "not held", "unmet": ["multisets: differ", ...]}`, with
   `--format text` the line `not held: multisets: differ; ...`. The refusals keep
   their codes (1, 2 and 3) and print no document, flag or not; the default exit
   status of `verify` is unchanged.

```
waterwhite verify --store evidence.db --report report.json --sides sides.json --require-held \
    > verification.json || { echo "repair not verified" >&2; exit 1; }
```

The conditions, each unmet one named on standard error by the words in quotes. A
condition that needs a comparison is unmet when the comparison was not made, except
where the report names nothing of the kind to compare (stated below):

- a sides file is given when the report names a JSONL file or a shard (`"sides: not
  given"`): without `--sides` nothing is compared, so such a verification never holds.
- `multisets` is "equal" (`"multisets: differ"`, `"multisets: not compared"`,
  `"multisets: not claimed"`); an unresolved record in the report reads "not claimed".
  Where the report names no JSONL file the word is "not compared" whatever happens, and
  that word alone is not unmet; a report that names only shards is gated by the
  conditions that follow. A report whose only JSONL files are ones the repair
  leaves no row has no comparison to make and reads "not compared", which is unmet: the
  gate does not hold for it.
- `per_shard_digests`: "differ" is unmet in either mode. In `--mode seeded`, where the
  report names a shard, it must be "equal" (`"per_shard_digests: not compared"` is unmet):
  a seeded verification that compared no shard does not hold. In the default
  `--mode unseeded` bytes are not claimed and "not compared" is its normal word.
- `differences` has no entry with a `note`, except the notes that say there is nothing
  to compare because the repair leaves the file or shard no record
  (`"differences NAME: NOTE"`). A file or shard that was named on one side only, on
  neither, or that both builds wrote and the report does not name is unmet here even
  where the word of the comparison above does not show it.
- every `kept` entry of a JSONL file or a shard has `identical` true (`"kept NAME: not
  identical"`, or `"kept NAME: not compared"` when the repaired side names no file for
  it). A kept mixture or checkpoint has no file and is not asked.
- every `successors` entry is registered, answers "does not contain", and does not have
  `matches_sides` false (`"successor NAME: not registered"`, `"successor NAME: may
  contain"` or the word of whatever other answer it gave, `"successor NAME: not the file
  compared"`).
- every `history` entry has `instance_unchanged` true (`"history NAME: instance changed"`).
- every `caches` entry has an answer, and a stale one is rebuilt (`"caches SEGMENT: no
  answer"`, `"caches SEGMENT: stale, not rebuilt"`). A stale cache that is rebuilt holds:
  the corrected order is derived, and `restarts` is information the gate does not read.

The list follows what the description of `verify` above says it checks; it adds no
check to the verification. A store, a report or a sides file that is
refused never reaches it.

## The error "store written during the query"

`query`, `report` and `verify` read through a coverage session, which holds for one
state of the store. If the store is written while the command runs, by another process
or by the same one, in any window up to the moment it prints (between the last check of
a proof and the answer built from it, between two answers, or after the last answer),
the command ends with the JSON error on standard error, nothing on standard output,
and exit 2:

```json
{"error": "store written during the query", "detail": "the store changed after the query began; run the command again", "context": {}}
```

A command that finds the store as it was answers; the command does not retry. A write
after the output is printed is outside the command.

## Not built

`extension` belongs to a later part of the tool. There is no command that ingests a
source file or wraps a tokenizer run: those are calls of a pipeline
(`capture.ingest_jsonl`, `tokenmap.tokenize_shard`). No command switches an extension:
the switch is a setting of the store (`Store.set_setting`, `docs/store.md`). No
command builds a repaired artifact: the team's own builder does, from the report.
