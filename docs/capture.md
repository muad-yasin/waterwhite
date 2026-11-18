# Capture, the normaliser and the harness

Three modules sit between a team's pipeline and the evidence store:

- `waterwhite.normalise`: the named function a record's text passes before it is
  fingerprinted.
- `waterwhite.capture`: the wrappers a pipeline calls where it reads a source file,
  writes a file, or takes in a file with no history.
- `waterwhite.harness`: runs a generation job so that its reads of the records it
  was handed are logged.

They compute digests, fingerprints and references and hand records to the store.
They decide nothing about answers and keep no state. The store is the only writer
and the judge of every record. No record text reaches the store or a harness log:
only locators, fingerprints and links.

A pipeline imports four things of the tool for these capture points:
`waterwhite.capture`, `waterwhite.harness`, `waterwhite.normalise` (when its
sanitizer is a named normaliser) and `waterwhite.store.Store` (to open the store
it writes to). Section 2 lists what `capture` names so that this is enough. Three
more packages serve the points this file does not cover: `waterwhite.tokenmap`
wraps the tokenizer run and writes the shard's map (docs/tokenmap.md),
`waterwhite.confine` runs a generation job out of process in a sandbox
(docs/confine.md), and `waterwhite.recovery` derives the links of a run made
without capture (docs/recovery.md). docs/interfaces.md section 5.4 says which a
pipeline calls at which step.

Every named error these modules raise is a class of `waterwhite.errors`.

Plan sections served: 3.1 (records, fingerprint, instance id), 3.3 (ingest, copy
steps, generation, selection), 2.3 (every line of a registered file is an
occurrence), 5.4 (the capture points), 2.2(g) and Decision 13 (the harness log
behind a generation edge's bound), Decisions 11, 24, 25 and 29, limits L1, L3, L5
and L8, assumptions F8 and F10. `DECISIONS.md` entries D-07 (a refused read),
D-17 (one definition of the normaliser) and D-21 (a sole handed record) are named
where they apply.

Tests: `tests/capture/`, and with the real store
`tests/integration/test_store_capture.py`.

## 1. The normaliser

A fingerprint is a 128-bit hash of a record's text after one fixed normalisation.
Equal fingerprints mean "equal after that normalisation". They do not mean equal
bytes, and they do not mean the same origin.

A normaliser has a name of the form `<name>/<version>`. A store records one name
when it is created, and every fingerprint in that store is taken after that
function.

```python
from waterwhite import normalise

normalise.names()                                  # ('control-chars/1',)
normalise.apply("control-chars/1", " a\x00b\r\n")  # 'ab'
normalise.text_fingerprint("control-chars/1", "some text")   # 16 bytes
normalise.get("control-chars/2")                   # raises UnknownNormaliser
```

- An unknown name raises `UnknownNormaliser` (error name "unknown normaliser", a
  refused write; the class lives in `waterwhite.errors` and is also importable
  from `waterwhite.normalise`). There is no default normaliser.
- A store names its normaliser as `store.normaliser`.
- A name means one function for good. `normalise.register(name, function)` adds a
  team's own sanitizer under its own name. Registering a different function under
  a taken name is refused. A changed rule is a new version.

### control-chars/1, in full

The function does two things, in this order.

1. It removes these code points wherever they stand: U+0000 to U+0008, U+000B to
   U+001F, and U+007F to U+009F. That is every character of the Unicode category
   Cc (control) except the tab (U+0009) and the line feed (U+000A). There are 63
   of them.
2. It strips these code points from both ends: the tab, the line feed, the space
   (U+0020), U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F and
   U+3000. That is Unicode white space that is not a control. There are 21 of them.

What follows from the rule:

- A tab or a line feed inside a text stays. A record with paragraphs keeps them.
- The carriage return (U+000D) is removed. A `\r\n` pair becomes `\n`. A carriage
  return that stands alone is removed and its neighbours join.
- U+0085 (next line) is a control and is removed.
- Characters of the category Cf (format) stay: zero-width joiners and non-joiners,
  the soft hyphen, direction marks, the byte order mark U+FEFF. Two texts that
  differ only in such a character have different fingerprints. A byte order mark
  at the start of a text is not stripped.
- White space inside a text is not changed or collapsed.
- The removal comes first, so a control character at an end does not shield the
  white space next to it.
- Applying the function twice gives the same text as applying it once. So a record
  a sanitizer step wrote with this function has the same fingerprint as the record
  it was made from.

The two lists are written out in the code as fixed code points. The function does
not ask the interpreter's Unicode tables, so its result does not change with the
Python version.

The function is defined once, here, and nowhere else (DECISIONS.md D-17). A
pipeline whose sanitizer step is this function imports it from
`waterwhite.normalise` in every mode, also when it records nothing:
`normalise.apply("control-chars/1", text)`, or
`capture.normalise_text(store.normaliser, text)` where it has the store at hand.
A second copy in the pipeline could drift from this one, and a drift would show
as lost links, not as a failed test.

## 2. Capture

### How a file is read

Each capture function reads the file once. From the same bytes it takes the
SHA-256 of the file (the content digest), the number of lines, and one occurrence
per line. It declares the count and the digest to the store. The store reads the
file itself and refuses the write when the file no longer matches, so a file that
changes underneath is not registered.

- What a line is, is defined once, in `ids.split_lines`, and the store counts the
  lines of the file with the same function, so the count a wrapper declares and
  the count the store finds are counts of the same thing. A line ends at a line
  feed byte (0x0A). A last piece with no line feed after it is a line when it is
  not empty. An empty file has no lines. A carriage return ends no line. (A reader
  that opens the file in text mode with universal newlines splits at a lone
  carriage return too; such a file then has different line numbers there.)
- The file is read once (`ids.read_lines` with an `ids.ContentHasher`): the
  digest is taken over the very bytes the lines are split from.
- Line n, counted from 1, is the occurrence with the locator `L<n>`.
- Every line is an occurrence. A line carries a fingerprint when it is valid UTF-8,
  parses as a JSON object, holds a string in the text field (`text` unless
  `text_field` says otherwise), and that string is not empty after the store's
  normaliser. Every other line is an occurrence with no fingerprint: a blank line,
  a line that does not parse, a line with no text field or with something other
  than a string there, a text that is empty after normalisation.
- Lines are parsed with Python's `json.loads`, as Megatron's preprocessing does.
- Refused: a text that holds a surrogate code point with no partner (possible
  through a JSON escape). Such a text has no UTF-8 form and so no fingerprint.
  The whole call raises `InvalidRecord` naming the line, and nothing is written.
  It is not passed over as a line with no text, because it holds text.

Capture does not judge whether a file is a corpus. A file of lines that hold no
text is registered as that many occurrences with no fingerprint.

`capture.scan_jsonl(path, normaliser=..., text_field="text")` does this reading on
its own and returns the count, the digest and the occurrences.

What the two reads establish. The wrapper reads the file once and the store reads
it again before it writes a row; the write goes ahead only when the two agree. That
is agreement at those two reads. It does not make the path immutable afterwards: a
file replaced later is not noticed by anything that is keyed to the store (the
store's mark, `state_mark()`, moves on a write to the store and not on a change to a
file, docs/store.md "Opening a store"). A registered file is meant to stay as it is: the
loud checks compare it with its registered digest whenever they run, and a pass of
them speaks for the bytes that were read in that pass. These checks catch accidental
edits and inconsistent imports. They are not a defence against someone who rewrites
every commitment the store holds.

### ingest_jsonl: a file of a source revision

```python
instance = capture.ingest_jsonl(store, "sources/A.jsonl",
                                source="A", member="A/r1", revision="r1",
                                name="src/A")
```

It first reads `store.member(member)`. A member name that is already registered
under another source, at another revision or as another kind of part is refused
with `InvalidRecord`, and nothing is written. The store would refuse the same at
the third call below, after the file was registered, and would leave a file that
no member covers.

It then makes three writing calls to the store, in this order:

1. `register_jsonl` with role "source", no step, the member name, no inputs, the
   occurrences, the declared count and digest. The artifact's name is `name`, or
   the member's name when none is given.
2. `add_source(source)`.
3. `add_member(source, member, revision, "file", [(instance, "")])`: the member is
   the whole file (`vocabulary.PART_FILE`).

It returns the instance id, which is `ids.ingest_instance_id(content digest,
member name)`. The same bytes ingested under two members are two instances.
Capture computes that id itself and raises `InvalidRecord` when the store returns
another one. The same ingest run again returns the same id and writes nothing.

The registration comes first, so a refused file leaves no source and no member
behind. If the process stops between the calls, the file is registered with no
member. Its records then have no recorded origin until the ingest is run again.

### record_jsonl: a file a step wrote

```python
step = capture.step_ref("shuffle", "1", {"seed": 7})
edges = [capture.copy_edge(1, capture.occurrence(src_a, 3), step),
         capture.copy_edge(2, capture.occurrence(src_a, 1), step)]
instance = capture.record_jsonl(store, "work/shuffled.jsonl", name="shuffled",
                                role="intermediate", step=step,
                                inputs=[src_a], edges=edges)
```

`inputs` are the instance ids of the step's input artifacts, in the step's order.
`edges` and `drops` are what the step stated as it ran. The function makes one
call, `register_jsonl`, and returns the instance id the store computed from the
content digest, the step, the inputs and the stated edges.

Refused with `InvalidRecord`, before anything is written:

- a role other than "intermediate" or "release";
- no step, or no artifact name;
- inputs that are one text and not a sequence of instance ids, or an empty id;
- an edge whose child is not a line of the file (`L1` to `L<count>`), or two edges
  for one child;
- an edge that is not stated (a link derived afterwards is a recovered edge and
  goes to the store by its own method);
- a drop with a reason outside the vocabulary.

Everything else about a record is the store's to check (docs/store.md): labels,
parents that must exist, the hand-off of a generation edge. Two of its rules meet
a pipeline often:

- A parent of a stated edge must lie in an artifact named in `inputs`, or in the
  file itself (DECISIONS.md D-15). So `inputs` lists every artifact the step read
  a parent from. A parent in the file being written is named with the instance id
  `""`: `capture.OccurrenceRef(capture.OWN_ARTIFACT, capture.line_locator(n))`.
- The same file registered again with the same step, inputs and edges is the same
  instance, and the store returns it. If the second registration differs in
  something the instance id does not cover (a record in scope only, a label, the
  harness log of a job that was run again, a fingerprint taken from another text
  field), the store refuses it with `InvalidRecord` (DECISIONS.md D-14). A job
  that is run again gets a new job name and its output a new step version or
  parameters, when the new run is meant to be a new record.

### import_jsonl: a file with no recorded history

```python
instance = capture.import_jsonl(
    store, "legacy/T.jsonl", name="import/T", member_name="legacy/T",
    file_family=("A/r1", "file family: translations made from source A"))
```

It calls `register_jsonl` with role "import", no step and the member name. The
instance id comes from the content digest and the member name, so the same bytes
imported under two member names are two instances.

`file_family` is `(member, evidence)`: a registered member the whole file is known
to derive from, and what that knowledge rests on. With it, the function adds one
file-level link from the whole file to that member, stored with the derivation
"file" (`add_source_link(instance, None, member, "file-level", evidence,
derivation="file", verification=None)`). A file family without a member or without
evidence text is refused here. So is one that names a member the store does not
hold: the function reads `store.member(member)` before it registers the file, so
a refused import leaves nothing behind, where the store's own refusal of the link
would come after the file was registered.

Without a file family the import's records have no recorded origin. Either way no
record-level link is written. Which record came from which item stays unknown, and
a parent is never guessed from similar text.

### Builders for what a step states

| Call | What it returns |
|---|---|
| `step_ref(name, version, params)` | a `StepRef` with the digest of the parameters (any JSON value) |
| `occurrence(instance_id, line)` | the reference of line `line` of a registered file |
| `copy_edge(child_line, parent, step)` | one parent, mode "all", stated, observed |
| `selection_edge(child_line, group, kept, step, group_key=None)` | `(edge, drops)` for a dedup group |
| `drop(input, reason, step)` | a drop with a reason of the vocabulary |
| `generation_edge(child_line, output, log, step)` | the edge of one harness output |
| `generation_edges(result, child_lines, step)` | one edge per output of a harness result |

`selection_edge`: `group` is every equivalent copy the step saw, in the step's
order, and `kept` the one it selected. The edge has mode "any", names every copy as
a parent and marks the kept one as selected. Every other copy is a drop with the
reason "dedup loser" and a group key. The group key is the step's own when it gives
one, and otherwise the locator of the output line the group became. Two builds
that keep different copies give edges that differ in the selected parent, so the
store gives them different instance ids even when the files are byte for byte the
same. A kept copy outside its group and a copy named twice are refused.

`generation_edge`: every record the job was handed is an input of the edge, in
hand-off order. Which of them are parents (plan Decision 25, DECISIONS.md D-21):

- A job handed exactly one record: that record is a parent of each output (`used`
  true), even when the output states nothing. It is the output's sole input.
- A job handed two or more records: a record the output states it used is a
  parent. A handed record the output does not state is in scope only (`used`
  false) and is not a parent.

The mode is "all", the edge is marked as a generation edge and carries the job's
harness log. An output that states a record the job was not handed is refused.

The edge also keeps the name of its job (DECISIONS.md D-24): `generation_edge`
sets `EdgeRecord.job_name` to the job name of the log. The store writes the name
on the edge itself, so an answer can still name the job when the log is gone
(plan Check 5). A pipeline that builds a generation edge by hand passes
`job_name` itself: the log's job name when the edge has a log, and the name of
the job when the job wrote its own parent list and there is no log. The store
refuses a generation edge with no job name, a job name on any other edge, and a
job name that differs from the log's.
The store holds a generation edge to the same rule: an edge written by hand that
marks a job's sole handed record as in scope only is refused there.

### What capture names for a pipeline

So that a pipeline imports nothing of the tool beside `capture`, `harness`,
`normalise` and `Store`, `capture` names (`capture.__all__`):

| What | Names |
|---|---|
| record types (`waterwhite.records`) | `StepRef`, `OccurrenceRef`, `OccurrenceRecord`, `EdgeRecord`, `EdgeInput`, `DropRecord`, `HarnessLog`, `MapRecord`, `MapRow`, `SnapshotShard`, and `OWN_ARTIFACT` |
| id functions (`waterwhite.ids`) | `line_locator`, `ordinal_locator`, `param_digest`, `file_digest`, `shard_digest` |
| roles | `ROLE_SOURCE`, `ROLE_INTERMEDIATE`, `ROLE_RELEASE`, `ROLE_IMPORT` |
| parent modes | `MODE_ALL`, `MODE_ANY` |
| drop reasons | `DROP_SANITIZER_SKIP`, `DROP_EMPTY_LINE`, `DROP_NO_TOKENS`, `DROP_FILTER`, `DROP_DEDUP_LOSER`, and the tuple `DROP_REASONS` |
| labels | `OBSERVED`, `RECOVERED`, `CANDIDATES`, `FILE_LEVEL`, `UNKNOWN` |
| derivations and verification results | `DERIVED_STATED`, `DERIVED_FILE`, `DERIVED_POSITION`, `DERIVED_TOKEN_MATCHING`, `VERIFIED_PASSED`, `VERIFIED_FAILED`, `VERIFIED_NOT_RUN` |
| map namespaces | `NAMESPACE_CAPTURE`, `NAMESPACE_RECOVERY` |
| member part kinds | `PART_REVISION`, `PART_FOLDER`, `PART_FILE`, `PART_RECORDS` (`PART_KIND_FILE` is the older name of `PART_FILE`) |

They are the same objects as in `records`, `ids` and `vocabulary`, not copies.
The map types, the namespaces and the shard digest are there for a pipeline's
tokenization and mixture capture points. Tokenization has its wrapper in
`waterwhite.tokenmap`: `tokenize_shard` runs the team's tokenizer, registers the
shard and writes its capture map, and `map_tokenized_shard` does the same for a
pair a tokenizer run has just left (docs/tokenmap.md). The mixture capture point
calls the store's `register_mixture` itself.

`capture.line_text(raw, text_field="text")` is the reader of one line: the text a
line holds, or `None` when it holds none (the rules of "How a file is read"
above). It is the one definition in the tool. The capture wrappers fingerprint
what it returns, the recovery compares it, and a tokenization map tokenizes it
again.

### What capture expects of the store

The write methods of `docs/interfaces.md` section 4, and the name of the store's
normaliser as `store.normaliser`. The real store gives it as a property;
`capture.store_normaliser` also takes a method with no arguments, for a stand-in.
A store that does not name its normaliser is refused.

## 3. The harness

```python
from waterwhite.harness import Handed, JobOutput, run_job

def paraphrase(ctx):
    (only,) = ctx.handed()
    text = ctx.read(only)
    return JobOutput(text=rewrite(text), used=(only,))

result = run_job(paraphrase, job_name="paraphrase",
                 handed=[Handed(ref, text)], log_path="logs/paraphrase.jsonl")
result.outputs        # the job's outputs, in its order
result.log            # a records.HarnessLog, read back from the log file
```

`run_job(job, *, job_name, handed, log_path, model_client=None)` returns a
`JobResult(outputs, log)`.

`Handed(ref, text)` hands a job the text of the record at `ref`. `run_job` and
`run_confined_job` hold no store, so they run whatever they are handed: a job handed
the reference of one record with the text of another runs. The log they return
carries the handed texts in `log.handed_texts`, each reference with its text in
hand-off order. The log file holds references and no text; the texts are held on the
returned log only, take no part in its equality, its representation or its digest,
and are never stored. A log read from its file (`read_log`) carries none.

The texts are compared where the output is registered. They travel on the log
(`HarnessLog.handed_texts`) and on the edge `generation_edge` builds from it
(`EdgeRecord.handed_texts`), and the store compares every text of both, each once,
with the record its reference names: the fingerprint of the text after the store's
normaliser against the fingerprint stored for that record. A text that is not the
record is refused with `InvalidRecord`, naming the job and the reference, and
nothing of the registration is written (docs/store.md, "The write methods"). A text
that is the record after the normaliser is the record: under `control-chars/1`
control characters are removed and the ends stripped, so a text that differs only in
those matches; a format character, white space inside the text, or the text of
another record does not. A record that has no fingerprint (a line that holds no
text, or one that is empty after the normaliser) holds no text: a handed text that
is empty after the normaliser is that record, and any other text is refused in the
same way. An edge and a log that carry different texts under one reference are
refused unless each of them is the record after the normaliser.

Which registrations compare, and which are stored as not compared (the log establishes no
bound, and a negative past it is "may contain", until the texts are shown):

- Compared: each text that the edge or the log of a registration carries. That is the
  result of `run_job` or `run_confined_job`, held in memory, whose edges come from
  `generation_edge` or `generation_edges` and are registered with `record_jsonl` or
  `Store.register_jsonl`; a pipeline that keeps the result of its job and registers
  the output in the same process takes this route, and so do the replica's capture
  hooks. The comparison is made for each edge on its own, so a fresh result of a job
  whose log is already stored is compared too, also when the same history is
  registered again. It is also an `EdgeRecord` built by hand that carries texts on
  itself or on its log, and a `HarnessLog` built by hand and given its texts.
  (`Store.add_recovered_edges` makes the same comparison for an edge built by hand,
  but it refuses a stated edge, which is what `generation_edge` builds.)
- A log is stored as **compared** (`hand_off_checked`, docs/store.md) only when the texts
  of its registration cover every reference of its hand-off and each matched; every other
  registration stores it as not compared, and a negative past such a log reads "may contain"
  until `Store.verify_hand_off` (the command `compare-hand-off`) is given the original texts.
- Not compared: a handed reference that neither the edge nor the log carries a text
  for. A hand-built edge or log may give texts for only some of the handed
  references: those are compared and the others are not. All the references of a
  registration that carries no texts anywhere are not compared: a result rebuilt from
  the log file (`JobResult(outputs, read_log(path))`), which a pipeline that
  registers an output in a later process is left with; a `HarnessLog` built by hand
  without `handed_texts`, on an edge with none; an edge with no handed texts whose
  job wrote its own parent list and has no log; and a log read back from the store
  (`store.harness_log(job)`) passed to `generation_edge`, because a stored log holds
  no texts and gives an edge with none. For these the hand-off is only the caller's
  statement that each text is the record at its reference, so the log is stored as not
  compared: it establishes no bound and a negative past it is "may contain" until
  `Store.verify_hand_off` (the command `compare-hand-off`) is given the original texts
  (docs/confine.md, "The hand-off is what it says").
- A log registered for the generating-model rule carries the texts its job was handed, or condition (g) fails before (h): a log read back from its file has no texts, establishes no bound, and its answer names the bound and not the model entry.

The command line registers no generation edge and no harness log, so the
comparison is a matter of the Python calls above.

The job is a function that takes a `JobContext` and returns a `JobOutput` or a
sequence of them. Through the context it can:

- `handed()`: list the references it was handed. This is not a read.
- `read(ref)`: get the text of a handed record. The read is written to the log
  file and synced to disk before the text is returned.
- `call_model(model, prompt)`: call a model through the client `run_job` was given
  (`model_client=`, a function from model and prompt to text). The model's
  identifier is written to the log before the client is called. A client may
  return `harness.ModelReply(text, model)` in place of the text, to report which
  model answered: that identifier is written to the log after the call (an entry
  `model_reply` that names the call's entry), and the log's model entry names it
  in place of the one the job asked for. A reported identifier that cannot be
  logged (empty, "none", one that starts with "[") ends the job with
  `InvalidRecord`.

The log's job entry names the harness that ran the job. `run_job` always writes
"in-process": it takes no word for it, so no caller of the in-process harness can
write a log that establishes a bound (Decision 13). Only `waterwhite.confine`
writes "confined", through a function of this module that is not public; a log
file written by hand outside the tool is outside what the store can check
(docs/confine.md). `HarnessLog.harness` holds the word, and `None` for a log file that
does not say.

### What is refused

- A read of anything that was not handed raises `HarnessRefused` (error name "read
  outside the hand-off", of the family of refused requests, exit code 2). The
  attempt is written to the log with `refused` true and nothing is delivered. A
  job may catch the error and go on. A refused attempt does not break the bound of
  the job's edges: the job did not get the record (DECISIONS.md D-07), and the
  stored log still gives an established bound. A job that does not catch it fails;
  the log on disk still shows the attempt.
- An output that states a used record that was not handed raises `InvalidRecord`.
  So does a used record stated twice and a return value that is not `JobOutput`s.
- A hand-off that names one record twice, or a job with no name: `InvalidRecord`,
  before any log is written.
- A log file that already exists at `log_path`: `WriteRefused`. A log is evidence
  and is not overwritten. A pipeline that runs a job again writes to a new path.
- A model identifier that is empty, is the word "none", starts with `[` or has a
  space at an end: `InvalidRecord`. These forms are kept for the log's own use.
- A model call when no client was given: the call is logged, then
  `ModelClientMissing` (error name "no model client", exit code 1: the harness
  was set up without what the job needs).
- Any use of the context after the job has ended: `HarnessRefused`.

The harness does not compare an output's `used` list with the reads in the log. An
output may state a handed record it never read through the context. The used list
is what the output states, and an answer that rests on it says so. The log keeps
the list as stated; the rule for a sole handed record (D-21) is applied where the
edge is built, not in the log.

### The log file

JSON lines, one entry per line, written as canonical JSON (sorted keys, no spaces,
ASCII). It holds references and identifiers. It holds no record text, no prompt
and no clock time, so the same job with the same hand-off and the same reads gives
the same bytes and the same digest.

| `kind` | When | Fields beside `seq` and `kind` |
|---|---|---|
| `job` | first | `format` ("waterwhite.harness-log/1"), `job_name`, `handed` (references in hand-off order) |
| `read` | each read | `refused`, and `instance_id` with `locator` for a handed record or `outside_ref` for anything else |
| `model_call` | each call | `model` |
| `output` | per output, after the job returned | `index`, `used` |
| `model` | once, at the end | `model`, `models` |
| `end` | last | `status` ("completed" or "failed"), and `outputs` or `error` |

`harness.read_log(path)` reads a log file into a `HarnessLog`. `run_job` returns
exactly that reading of the file it wrote, so the returned log and the file cannot
disagree. `log_digest` is the SHA-256 of the file.

The model field of the log (`HarnessLog.model`):

- `"none"`: the log's model entry says the job called no model.
- an identifier: the job called that one model, once or many times.
- a JSON list such as `["m-2","m-1"]`: the job called two or more different
  models; the list holds each once, in the order of its first call.
- `None`: the file has no model entry. This is read as "not established" and
  never as "no model".

The model entry names every model the job asked for, whether or not the call
returned. `read_log` refuses a file whose model entry disagrees with its own call
entries, and a file that is not a harness log.

### What an in-process harness does NOT establish

The plan's closed-input harness means: the job can reach only the records it was
handed, and no other record path and no record-bearing network source. This
module does not give that.

The job is a Python function running in the caller's own process. It can open any
file the process can open. It can open a socket. It can import a client library
and call a model without `call_model`. It can read the context's own memory. None
of that passes the harness, and none of it is in the log.

So:

- The log shows the reads and model calls that went through the context, and
  nothing else.
- "The log lists no read outside the hand-off" is true evidence about a job only
  when the job is confined by some other means, or when the team knows the job's
  code and accepts it. The fixture's jobs are such jobs: fixed outputs, written in
  this repository, with no file or network access of their own.
- For any other job, an answer that rests on this log rests on trust in the job.

Confinement of a job that runs out of process is `waterwhite.confine`
(docs/confine.md; plan 5.8 B-1, assumption F10). It starts the job in a sandbox
with no network and no file of the machine's data, and serves it its handed
records through this harness, so the log file is this format and the store reads
it as it reads any other. The job entry of every log says which harness wrote
it, and an answer names that harness beside each log its negative rests on
(docs/coverage.md section 6). What is said above about an in-process job stays
true of a log whose harness is "in-process".

### Declared ancestry, observed reads and absence of influence

An answer built on a generation edge rests on three different kinds of statement, and
the store keeps them apart.

- **Declared ancestry.** The parents of an output are the records its `used` list
  states, plus a job's sole handed record (D-21). The harness logs the list as
  stated and does not compare it with the reads. A job handed A and B that reads
  both and states only B as used gives an edge with B as a parent and A as an input
  in scope only. A request that reaches A's record then meets a record in scope
  only, and the answer is "may contain" naming "in scope only", with the job and
  the record (docs/coverage.md, section 6). It is never "contains" on that basis,
  and it is never "does not contain". The tool does not turn the stated list into a
  proof that the output depends on its parents, or that it does not depend on the
  records it left out; an answer that passes a generation edge carries a note that
  the parent list is process evidence.
- **Observed reads.** The reads the log shows: the records the job read through the
  context. They are what the harness saw and nothing more (above).
- **A proof of absence.** A "does not contain" past a generation edge rests on the
  bound of plan 2.2(g): the job read nothing beyond what it was handed. Only a log
  the confined harness wrote establishes it. It says that no record outside the
  hand-off was read. It does not say which of the handed records the output
  depends on.

So the word "exact" about an answer means exact about what was recorded and about the
trust assumptions that answer names: the harness behind the log, and the job's own
statement of what it used. It does not mean that the stated parents are the true
dependences of the output.

The hand-off carries a text with each reference: `Handed(ref, text)`. `run_job` checks
that each item is a `Handed` with an occurrence reference and a string, and that no
reference is handed twice. It does not check that the text is the text of the record
the reference names: it holds no store. A job handed `Handed(ref_A, text_B)` reads text B
and the log shows a read of A. The comparison is made where the output is registered
(above, after `run_job`'s signature): the store refuses an output whose
job was handed a text that is not the record its reference names. It cannot make it
for a log registered without its texts, on an edge with none (one read from its file,
fetched from the store or built by hand), and
an answer that rests on such a log rests on the caller handing each record's own text.

## 4. The generating-model extension

The model a job called is a field of its harness log and nothing more. Capture
does not treat it as an input, a parent or an ancestor, does not look it up in the
store and does not read the extension's switch. A generation edge built here has
the handed records as its inputs whatever the model field says.

With the switch off (the default) this is the core's scope, stated as a fact and not as a switch that is merely off; with the switch on, the second and the last bullet below say what changes:

- The default ancestry follows recorded inputs and parents. It does not follow a
  source's influence through the training of the model that generated an output. A
  withdrawal of a source reaches what is derived from its records by the recorded
  edges, and does not reach what a generating model learned from it.
- With the switch off, a negative past a generation edge does not cover that model-mediated route. It
  carries the note "generating-model exposure not assessed", and its scope lists the
  model entry of each log it rests on. With the switch on, the negative rests on each edge's model being "none" or a registered checkpoint that answers "does not contain", and the note is not printed.
- Nor does the scope say that every output of such a model carries the source. It
  says that the route is not assessed, in either direction.
- With the switch on the proof reads the model field of the stored log (docs/coverage.md, section 6); capture writes it the same way in both states and still does not look it up.
