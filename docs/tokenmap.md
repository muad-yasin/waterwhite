# Tokenization maps

A tokenized shard holds token ids and document boundaries and nothing else. The
tokenizer reads one text field of each input line and drops the rest, ids
included. So which line a document came from is never written down by the step
itself. `waterwhite.tokenmap` derives that link, verifies it, and stores it as the
shard's map: for every document its ordinal, its token span, and the input line it
came from, or the set of lines it may have come from.

The package wraps a team's tokenizer run at capture time, and recovers the map of
a run that was made without the wrapper. It ships no tokenizer and imports none.

Code: `src/waterwhite/tokenmap/` (`_wrapper.py` the three entry points and what is
written, `_derive.py` the two derivation modes and their verification, `_shard.py`
the digest that finds twins, `_inputs.py` the settings, the input lines and the
sanitizer's log, `_tokenizers.py` the team's tokenizer as a command or by name).
Tests: `tests/tokenmap/`. The package is part of the tool: its errors are classes
of `waterwhite.errors`, it reads a line through `capture.line_text`, finds a
file through `Store.disk_path`, and reads a shard through
`waterwhite.store.artifact_files`. The command line reaches `recover_map` as
`waterwhite recover map` (docs/cli.md).

"The plan" is the build plan, cited by section; `DECISIONS.md` entries are cited
as `D-<nn>`.

| Part | Plan sections it serves |
|---|---|
| the capture wrapper (`tokenize_shard`, `map_tokenized_shard`) | 3.3 "Tokenization", 5.4 (capture points), Decision 4 |
| position mode | 3.3 (position mode), F1 |
| token-matching mode | 3.3 (token-matching mode), F1's fallback |
| verification and labels | 3.3 (the verification paragraph), 3.2, 2.2(f) |
| drops for lines with no tokens and for sanitizer skips | 3.1 "Drop", F1, F8 |
| recovery (`recover_map`) | 3.3, 5.4 (recovery points), Check 6 |
| writing through `store.write_maps`, one transaction | 3.4, 2.2(d); D-05, D-18, D-20 with its amendment in D-24 |
| a sanitizer's skip recorded once | 3.1 "Drop", 3.3; docs/interfaces.md section 5.4 |
| the questions the tests ask on toy shards | 5.5 Check 3 and Check 6 |

## 1. How a pipeline calls the wrapper

```python
from waterwhite import tokenmap
from waterwhite.store import Store

store = Store.open("work/evidence.db")

def run_tokenizer(input_path, prefix):
    # The team's own tokenizer run, unchanged, as a process of its own.
    # It reads input_path and leaves <prefix>.bin and <prefix>.idx.
    subprocess.run([...], check=True)

result = tokenmap.tokenize_shard(
    store,
    input="E/shard-1.input",            # the registered JSONL file the tokenizer reads
    prefix="pipeline/shards/shard-1",
    name="E/shard-1",
    step=tokenmap.tokenize_step("core_v0.20.0", {"tokenizer": "gpt2", "append_eod": True}),
    run_tokenizer=run_tokenizer,
    settings=tokenmap.TokenizerSettings(append_eod=True, eod_in_span=True, eod_token_id=50256),
    tokenize=team_tokenizer.tokenize,   # text -> token ids, without the end-of-document token
)
```

What the call does, in order:

1. It refuses arguments that cannot be carried out, before anything runs.
2. It checks that the input file still holds the bytes it was registered with, and
   that no file lies at `<prefix>.bin` or `<prefix>.idx` yet. The wrapper registers
   the pair this run writes; a pair left by an earlier run is not taken for it.
3. It calls `run_tokenizer(input_path, prefix)`.
4. It registers the shard with `store.register_shard`, under `name`, with `step`
   and the input artifact as its one input. `tokenize_step(version, params)` makes
   a step named "tokenize".
5. It derives the capture map, verifies it and writes it (sections 3 to 6).

`map_tokenized_shard` is steps 1, 4 and 5 for a pair that already lies at the
prefix. A pipeline that tokenizes many shards in one go and reaches its capture
point afterwards calls it once per shard.

The input must be a registered JSONL artifact, named by a name or an instance id.
It is the file the tokenizer reads: its line n is the record `L<n>`, and a map row
names its parent by that locator.

### The settings

`TokenizerSettings(append_eod, eod_in_span, eod_token_id=None)`. Both flags are
stored on every map.

- `append_eod`: the run was told to end every document with the end-of-document
  token (`--append-eod`). It is recorded as given.
- `eod_in_span`: the end-of-document convention. True when that token is the last
  token of every document's own span in the shard's index; false when no
  document's span holds it. It is a fact about the shard and comes from a reading
  of the shard outside the tool (in this repository, from the oracle's output,
  recorded in the fixture spec). The verification holds every document against
  it.
- `eod_token_id`: the token. Needed when `eod_in_span` is true.

`eod_in_span` true with `append_eod` false is refused: a run that appends no such
token leaves none in a span.

### The `tokenize` function

`tokenize(text)` returns the token ids of one text, as the team's tokenizer gives
them, **without** the end-of-document token. The tool adds that token itself when
`eod_in_span` is true. A text that gives no tokens gets none added, as in the
preprocessing, so it stays a text with no tokens.

Without `tokenize` no document is compared: the verification is "not run" and the
map is "unknown".

The tool ships no tokenizer and imports no tokenizer library. Where the team's
tokenizer is not at hand as a Python function, two helpers give one:

- `tokenmap.command_tokenizer(argv, timeout_seconds=120.0)` starts an external
  command once and asks it text by text. It speaks the protocol
  `waterwhite.tokenizer/1`: on its standard input one JSON object per line,
  `{"text": ...}`; on its standard output one per line, `{"tokens": [...]}`,
  each flushed before the next request is read. Use the result as a context
  manager; it is the `tokenize` function. `exit_status` holds the status the
  command ended with. A command that cannot be started, ends early, gives no
  reply within the time limit or answers with anything else raises "tokenizer
  failed", and no map is derived from the run.
- `tokenmap.entry_point_tokenizer("module:function")` imports the module the
  team names and returns its function. An import that fails, a name that is not
  there and an error the function raises are "tokenizer failed".

The command line takes the same two as `--tokenizer-command` and
`--tokenizer-entry` (docs/cli.md, `recover map`).

### What a tokenizer run must keep for position mode

Position mode rests on plan F1: one document per input line, in input order. For
Megatron's preprocessing that needs one partition (or the option that keeps the
samples in order when there are several), since with more partitions the lines are
dealt out in turn. `docs/replica.md` lists the conditions of a run in full. A
pipeline whose run does not keep them says so with `f1_holds=False`, and the map
is built by token matching.

## 2. How a line that gives no tokens shows

The preprocessing ends a document for every input line. In the shard's index:

- a line with tokens adds one sequence, and the document list moves on by one;
- a line with no tokens adds no sequence, and the document list gets one more
  entry with the same value as the entry before it.

Five input lines of which the third gives no tokens therefore give four
sequences and the document list `0, 1, 2, 2, 3, 4`. The repeated entry is a
document no reader can fetch. It has no ordinal: an ordinal is the number the
dataset's reader is asked for, the sequence number.

So the document list has one step per input line. Position mode reads off which
line was which: a step of one gives the line the next ordinal; a step of nothing
makes the line a drop with the reason "no tokens", and the next line still gets
the next ordinal. Nothing shifts.

This was seen once on a shard written by Megatron's preprocessing at the pinned
tag (six lines, the third empty: document list `0, 1, 2, 2, 3, 4, 5`), in a
scratch run that is not part of the test suite.

## 3. The two derivation modes

**Position mode** runs first. It walks the surviving input lines and the steps of
the document list together, as section 2 describes. It tokenizes nothing to build
the map.

**Token-matching mode** takes over when position mode's verification fails, and is
the only mode when the caller says F1 does not hold. Every input line is tokenized
again and given to a document that holds exactly its tokens. Lines with equal
tokens are given to the twin documents in order; which twin gets which line means
nothing, and the map names the whole set for each of them. A line that gives no
tokens is a "no tokens" drop.

Every mode that ran leaves its map (section 6). After a failed position mode the
shard therefore holds two maps of one namespace: the position map, failed and
"unknown", and the token-matching map. A reader gets the verified one first
(D-24).

## 4. Verification

Verification runs for both modes.

| Check | Position mode | Token matching |
|---|---|---|
| every ordinal's span has the size the shard's own index records, and the spans follow one another from token 0 with no gap, so they end at the index's total | yes | yes |
| the number of documents of the index, with and without tokens, equals the number of surviving input lines | yes | does not apply |
| every document's tokens equal the tokens its line gives when it is tokenized again (followed by the end-of-document token when `eod_in_span` is true) | yes, every document | yes, every document |
| a line the index says gave no tokens gives none when it is tokenized again | yes | by construction |
| the lines with tokens and the documents are the same collection of token sequences: no document without a line, no line without a document, as many lines as documents for a sequence that occurs more than once | follows from the two checks above | yes |
| two documents put in one class of twins hold the same tokens, compared directly | yes | yes |

Token comparison runs on every document, not on a sample. Twins are found by a
digest of each document's tokens and confirmed by comparing the tokens directly,
so a digest collision cannot make twins; two documents with one digest and
different tokens fail the verification.

The result is one of three words:

- **passed**: every check held.
- **failed**: at least one did not. The result names the first few findings in
  words, for example `document D1 does not hold the tokens of line 2`.
- **not run**: nothing failed and no `tokenize` function was given, so no
  document was compared.

### What is not verified

- That `tokenize` is the tokenizer that wrote the shard. The comparison of every
  document with its line is all the evidence there is; a function that gives the
  same tokens for these lines passes.
- Which twin sits at which ordinal. No verification can show it, and the label
  says so (section 5).
- `append_eod`. It is recorded as given. What is held against the shard is
  `eod_in_span`.
- That a line the sanitizer's log names was skipped for the reason the log gives.
  That the log fits the file is shown only by the count and the token comparison
  passing.
- That `run_tokenizer` ran the tokenizer unchanged, or as a separate process.
- The byte offsets of the index. A document is located from the token counts of
  the index, sequence after sequence, and the `.bin` file must have exactly the
  size those counts give. A shard whose offsets say otherwise would be read at
  the wrong place; with a `tokenize` function the token comparison then fails.

## 5. Labels

The label follows from the mode and the result (plan 3.3). A tokenization link is
never "observed".

| Verification | Document | Label | The row names |
|---|---|---|---|
| passed | its tokens equal no other document's in its shard | "recovered" | one parent: the line |
| passed | an identical twin: its tokens equal another document's in the same shard | "candidates" | the lines of all its twins, itself included: two or more |
| failed, or not run | any | "unknown" | no record |

- The map's own label is "recovered" when its verification passed and "unknown"
  otherwise. The rows carry the label of each document.
- Twins in one shard are "candidates" in every mode. Swapping the two lines in the
  input of a recovery changes neither label nor candidate set.
- Twins in two shards are told apart by their shard: each is "recovered" with the
  line of its own shard's input.
- A displaced document whose content differs from its neighbour's is caught by the
  token comparison, and the position map is "unknown". Only a swap among twins can
  escape, and for them the label already says the link is not known.
- An "unknown" row names no record, so a map that failed says nothing about which
  line a document came from. It is still the shard's own record (D-24): the shard
  answers from "unknown" links and is not a shard with no map.
- A row's fingerprint is the fingerprint the store holds for its line. A
  "candidates" row carries one when all its lines share it.

## 6. What is written, and when nothing is

Everything goes through the store's own write methods, in one call.

1. All maps a call derived, and the drops, are one `store.write_maps` call and
   one transaction (plan 3.4). It is declared with the document count and the
   shard digest that were read from the shard's two files **before** the first
   token was read. The store reads the files again and refuses a count or a
   digest that does not match them. So a shard that changed while its map was
   derived gets no map.
2. Inside that transaction the store holds every map against what the shard
   already holds in the same namespace and derivation. An equal one is left
   alone. A verified map there with other contents refuses the whole call with
   `InvalidRecord`, and nothing is written: a map is never overwritten. A map
   that did not verify does not stand in the way (see below).
3. The drops are written in the same transaction, recorded for the shard, and
   only when one of the maps verified. A drop the store holds with another
   reason refuses the whole call.

A process killed at any point of the call leaves the maps with their drops, or
nothing. There is no state in which a map is stored and its drops are not.

A capture map goes to the namespace "capture", a recovery map to the namespace
"recovery" (D-20). A recovery never touches a capture map.

The same call again changes nothing.

**A run that did not verify can be corrected.** A map whose verification failed
or was not run is stored as the record of that run, with the label "unknown". It
does not take its namespace and derivation for good: a later run of the same
mode writes its map beside it, and a reader takes the verified one
(docs/store.md, "A map that did not verify does not take its place for good").
So a recovery with the wrong log or the wrong tokenizer, and a capture map
written with no `tokenize` function, do not shut out the run that gets it right.
Once a map of a namespace and derivation verified, that map is final there.

| Refused | Error | What is left |
|---|---|---|
| the shard's files differ from the registered digest | `ArtifactDigestMismatch` | no map, no drop |
| the input's file differs from the registered digest | `ArtifactDigestMismatch` | no map, no drop; the capture wrapper does not run the tokenizer |
| the shard changes while the map is derived | `DeclaredDigestMismatch` or `DeclaredCountMismatch`, from the store; `ArtifactDigestMismatch` when the number of documents changed under the reader | no map, no drop |
| the store holds a verified map of the same namespace and derivation with other contents | `InvalidRecord` | the held map, unchanged; none of this call's maps |
| the store holds one of the drops with another reason | `InvalidRecord` | no map, no drop |
| a sanitizer's log whose skips the store already holds under a registered file, or for a file that is the sanitizer's own output | `DropAlreadyRecorded` | nothing; nothing is derived |
| a file lies at the prefix before the tokenizer runs (`tokenize_shard`) | `InvalidRecord` | nothing; the tokenizer is not run |
| the tokenizer run leaves no pair | `InvalidRecord`, from the store | nothing |
| F1 said not to hold, or both maps asked for, and no `tokenize` function | `InvalidRecord` | nothing |
| a sanitizer's log that names a line the input does not have | `InvalidRecord` | nothing |
| a sanitizer that is not a registered normaliser, when a `tokenize` function is given | `UnknownNormaliser` | nothing |
| a shard whose index or `.bin` file cannot be read as the layout describes | `UnsupportedFormat` | no map |
| the store holds another number of records for the input than its file has lines | `MapCountMismatch` | nothing |
| the team's tokenizer, as a command or an entry point, fails | `TokenizerFailed` | nothing |

A shard that is registered and got no map has no record of its own. It answers
"not covered" until a map is written.

## 7. Recovery

For a run made without the wrapper, the shard is registered first (the store's
`register_shard`), then:

```python
skips = tokenmap.read_sanitizer_log("pipeline/sanitizer.log.jsonl", shard="shard-2")
result = tokenmap.recover_map(
    store,
    shard="E/shard-2",
    input="E/shard-2.chunk",                     # see the two cases below
    settings=tokenmap.TokenizerSettings(append_eod=True, eod_in_span=True, eod_token_id=50256),
    tokenize=team_tokenizer.tokenize,
    sanitizer=tokenmap.SanitizerRun(skips),
)
```

The input is a registered JSONL artifact, in one of two cases:

- **The file the tokenizer read** (`sanitizer=None`). Every line survived, and
  each line's text is tokenized again as it stands.
- **The file the sanitizer read** (`sanitizer=SanitizerRun(skips)`). The lines the
  sanitizer's log names are drops and take no ordinal, so the ordinals of the
  lines after them do not shift. Every other line survived, and its text is passed
  through the sanitizer again before it is tokenized again (plan F8). The
  sanitizer is the store's normaliser, or the one `SanitizerRun.normaliser` names
  in `waterwhite.normalise`. A drop is recorded with the reason the log gives,
  under the step name `SanitizerRun.step_name` ("sanitize" unless told
  otherwise), for the shard.

Position mode runs first. Token matching runs when position mode's verification
fails, when `f1_holds=False` (position mode is then not run), and when
`also_token_matching=True` asks for both maps so that they can be read and
compared. Every mode that ran leaves its map, in the namespace "recovery".

With `write=False` nothing is written and the result alone comes back, so a
result can be looked at first. (`waterwhite recover map --dry-run` is this.)

### A sanitizer's skip is recorded once

A line the sanitizer skipped, or an empty line, is recorded exactly once, as a
drop whose output is the first registered artifact that lacks the line
(docs/interfaces.md section 5.4). For this package that means:

- where the file the tokenizer read is registered, the map is recovered over it
  with `sanitizer=None`, and the skips are the business of
  `recovery.recover_sanitizer`, which records them with that file as their
  output;
- where only the file the sanitizer read is registered, the map is recovered
  over it with a `SanitizerRun`, and the skips are recorded with the shard as
  their output.

`recover_map` refuses a `SanitizerRun` with the named error "drop already
recorded" (`DropAlreadyRecorded`), before anything is derived or written, when:

- the input is itself the output of recorded sanitizer drops. It is the file the
  sanitizer wrote; the log numbers the lines of another file;
- a line the log names already has its drop under a registered JSONL file. That
  file lacks the line, and the map belongs over it, with no log;
- a line the log names already has a sanitizer's drop for this shard under
  another step name.

The same line dropped for another shard is no conflict: two tokenizer runs over
one sanitizer input have one drop each. A sanitizer's drop is a drop with one of
`vocabulary.SANITIZER_DROP_REASONS`; the step name is "sanitize"
(`vocabulary.STEP_SANITIZE`) unless `SanitizerRun.step_name` says otherwise, and
`recovery.sanitize_step` names the same step.

### The sanitizer's log

`read_sanitizer_log(path, shard=None, input=None)` reads JSON lines, one object
per skipped line, with at least:

- `line`: the line of the sanitizer's input file, counted from 1;
- `reason`: "sanitizer skip" or "empty line", spelled as the tool's drop reasons.
  Another word is refused, not guessed at.

A log that covers several files tells them apart by its fields `shard` and
`input`. Name one or both and only the rows with that value are read. With neither
named, every row is read, and a log whose rows name more than one shard or input
is refused, because its line numbers would be those of several files. A pipeline
with another log format builds the `SanitizerSkip(line, reason)` entries itself.

## 8. What a call returns

`ShardMaps`: the shard's and the input's instance ids, the document count, the
number of surviving lines, `maps` (one `MapOutcome` per mode that ran, in the
order they ran), `drops` and `notes`. `to_json()` gives the result as a JSON
object, the maps without their rows; it is what `waterwhite recover map` prints.

- `MapOutcome.record` is the `MapRecord` handed to the store, with every row.
  `verification` is the `Verification`: the result, whether the spans and the
  count held, how many documents were compared, the number of findings and the
  first five in words. `recovered` and `candidates` count the documents with each
  label. `stored` says whether the store holds the map now.
- `ShardMaps.map(derivation)` gives one mode's outcome; `verified` is true when
  one of the maps passed.
- `notes` says in words what was not done: position mode not run, token matching
  not run for want of a `tokenize` function, no drop recorded because nothing
  verified, nothing written because the call asked for the result only.

## 9. Memory

A shard can hold a hundred thousand documents. Neither a shard nor an input file
is ever held in memory whole: `ShardTokens` reads one document from its own byte
range in the `.bin` file, the input lines are read as a stream, and a sanitizer's
log is read as a stream too. What is held grows with the number of documents and
lines, not with their size:

| Held | Size | Where |
|---|---|---|
| the shard's index: one token count per document, one entry of the document list per line | 4 and 8 bytes each | `artifact_files.read_shard_index` |
| where each document starts in the token stream | 8 bytes per document | `ShardIndex.token_starts` |
| one digest of each document's tokens, to find twins | 16 bytes per document, as a dictionary entry | `DocumentClasses` in `_derive.py` |
| the line each ordinal was given | 8 bytes per document, per mode | `Assignment.line_of` |
| the fingerprint of each input line, from the store | 16 bytes per line | `_derive_and_write` in `_wrapper.py` |
| one map row per document, per map, as a Python object, until the store has written it | a few hundred bytes per document | `_map_record` in `_wrapper.py`, `Store.write_maps` |
| one skip per line a sanitizer's log names | a small object per skipped line | `read_sanitizer_log` in `_inputs.py` |

The map rows are the largest of these. They are handed to the store as one
record, because a map is written whole in one transaction (plan 3.4).

The index and the tokens are read by the store's reader
(`store.artifact_files`); this package holds no second reader of the shard
format.

Position mode reads every document twice (once for the twins, once for the
comparison) and tokenizes every line once. Token matching tokenizes every line
once more. The cost at the sizes of plan 5.6 is not measured here.

## 10. Limits

- A verified map is final for its namespace and derivation. A map that did not
  verify can be followed by another (section 6); it is never removed, so a shard
  that saw wrong runs keeps their maps beside the right one.
- Token matching needs the team's tokenizer reproduced exactly. For a run
  recovered long after the fact that can fail (plan Decision 4), and the map is
  then "unknown".
- A shard in which a sequence holds no tokens, or whose token type is a
  floating-point type, is refused as an unsupported format.
- The two refusals of a sanitizer's log rest on what the store holds. A
  sanitizer input that was sanitized twice, with one output registered as a file
  and the other not, cannot have the second run's shard recovered through the
  log: the store shows a registered file that lacks the line and cannot tell
  that the shard was made from another one.
- A tokenizer command is trusted to be the team's tokenizer, as a `tokenize`
  function is: the comparison of every document with its line is the evidence.
- The capture wrapper takes the file the tokenizer read. A pipeline whose
  sanitizer and tokenizer are one step, with only the sanitizer's input
  registered, uses `recover_map` with a `SanitizerRun`; its map then lies in the
  recovery namespace.
