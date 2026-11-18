# Recovery: links for steps nobody observed

A team that ran its pipeline without the capture wrappers (docs/capture.md) still
has the files its steps left behind: the source files, the chunk files, the
sanitizer's input and output, the sanitizer's log. Recovery reads those files and
the evidence store and derives, one step at a time, which record of an earlier
file each record of a later file was made from.

It covers two kinds of step:

- a **copy step** (shuffle, chunk, merge): the step passes a record's text on
  unchanged;
- the **sanitizer**, a deterministic rewrite: the step applies one fixed function
  to each text and skips some lines.

A generation step is not recovered. A parent of a generated record is never
inferred from similar text. Tokenization (which document of a shard is which line)
is recovered by another package, not this one.

Code: `src/waterwhite/recovery/` (`__init__.py` the three calls, `_match.py` the
matching rule, `_files.py` the reading of registered files, `_sanitizer_log.py`
the log reader, `_result.py` the result type). Tests: `tests/recovery/`. The
package is part of the tool: its errors are classes of `waterwhite.errors`, its
words constants of `waterwhite.vocabulary`, it reads a line through
`capture.line_text` and finds a file through `Store.disk_path`. The command line
reaches it as `waterwhite recover file`, `recover copy-step` and `recover
sanitizer` (docs/cli.md).

"Plan" is the build plan, cited by section. Sections served: 3.3 "Copy steps" and
"Deterministic rewrite", 3.1 "Fingerprint" and "Drop", 3.2 (the labels
"recovered" and "candidates"), 5.4 (the recovery points), part of Check 6 in 5.5,
limits L1, L3, L5 and L12, assumption F8, Decisions 7 and 11. `DECISIONS.md`
entries D-05 (a link derived afterwards never changes an instance), D-15 (a
recovered edge may name any registered record) and D-17 (one definition of the
normaliser) are relied on.

## 1. The calls

```python
from waterwhite import capture, recovery
from waterwhite.store import Store

store = Store.open("work/evidence.db")

# a file a step left behind, with no recorded producing step
recovery.register_found_file(store, "pipeline/sanitizer-input/shard-1.jsonl",
                             name="E/shard-1.sanitizer-input")
recovery.register_found_file(store, "pipeline/tokenizer-input/shard-1.jsonl",
                             name="E/shard-1.input")
recovery.register_found_file(store, "pipeline/R1.jsonl", name="R1",
                             role=capture.ROLE_RELEASE)

# the copy steps between the sources and the sanitizer's input
result = recovery.recover_copy_step(
    store, "E/shard-1.sanitizer-input", ["src/A", "src/B", "src/Z"],
    step=capture.step_ref("shuffle-chunk-merge", "not recorded"))

# the sanitizer
result = recovery.recover_sanitizer(
    store, "E/shard-1.input", "E/shard-1.sanitizer-input",
    step=recovery.sanitize_step("control-chars/1"),
    log_path="pipeline/sanitizer.log.jsonl",
    log_input="pipeline/partitioned/sanitizer-input/shard-1.jsonl")

result.recovered, result.candidates, result.no_edge, result.drops
result.wrote_nothing      # True on a second run of the same recovery
```

One call is one step and one later file. A step that wrote twenty chunks is twenty
calls. Every file is named by a registered name or by its instance id, and must be
a registered JSONL artifact. `step` is the step being recovered, built with
`capture.step_ref(name, version, params)`; it becomes the transform of every edge
the call writes and the step name of every drop. `recovery.sanitize_step(version,
params)` builds the sanitizer's step under the name "sanitize"
(`vocabulary.STEP_SANITIZE`), the name a map recovered through a sanitizer's log
gives the same drops. Every call takes `write=False`: nothing is written and the
result alone comes back, with `dry_run` true.

## 2. Registering a file found afterwards

`register_found_file(store, path, *, name, role="intermediate", text_field="text")`
registers a file that has no recorded producing step and returns its instance id.

`role` is the place the file has in the pipeline: "intermediate" (a chunk file, a
sanitizer's input, a tokenizer's input) or "release". "import" registers a
text-only import from outside the pipeline, as `capture.import_jsonl` does with
no file family. A source file is not found: it is ingested
(`capture.ingest_jsonl`).

The store has no role word of its own for "a pipeline file found afterwards",
and needs none. The role says what the file is in the pipeline, and the missing
producing step says that it was found and not captured. What the listing and the
answers then say:

- a listing names a found release among the releases it covers. Registered as an
  import, as before, a release found afterwards answered like any file and was
  missing from that list;
- an answer carries no role, so no answer differs by it;
- `show` prints the role and `"step": null`.

The file is read once, as every captured file is: one occurrence per line, the
fingerprint of each text after the store's normaliser, and the line count and
SHA-256 that the store checks against the file itself. The instance id of such a
file is taken from its content digest and the name it is registered under
(`ids.ingest_instance_id`), so the same bytes found under two names are two
instances.

What follows from it:

- The artifact has no step and no inputs.
- No source and no member is created. The name is stored as the artifact's member
  name, and that is only a part of its id.
- Until a recovery links them, the records of the file have no recorded origin.
  A query about the file answers "may contain" naming the first such record
  ("missing origin", plan 2.2(a), L12). It never answers "does not contain".
- The same file under the same name and role again is the same instance and
  writes nothing. Under the same name with another role it is refused.
- The store's normaliser must be registered in the process, because fingerprints
  are taken. Without it the call fails with "unknown normaliser".

## 3. The matching rule

Both recoveries use one rule (plan 3.1 "Fingerprint", 3.2).

1. **Candidates by fingerprint.** For a record of the later file, the candidates
   are the records of the named earlier files that have the same fingerprint. The
   fingerprints come from the store's occurrence records; no text is loaded to
   build the lookup. Only fingerprints the later file holds are kept.
2. **Direct comparison.** For each candidate the tool reads both records from
   their files and compares the two texts after the normaliser, character for
   character. Equal fingerprints over different texts are a hash collision, and
   that candidate is dropped (the result counts it in `collisions_rejected`).
3. **The edge.**

   | Candidates left | Written | Edge fields |
   |---|---|---|
   | exactly one | a "recovered" edge | one parent, parent mode "all", verification "passed" |
   | two or more | a "candidates" edge | every candidate as a parent, in the order the caller listed the files and then by line; parent mode "any" with no parent marked as selected; verification "passed" |
   | none | nothing | the record keeps no recorded origin |

   The derivation is "fingerprint" for a copy step and "function" for the
   sanitizer.
4. **The count: surplus copies.** A copy step and a sanitizer write a record at
   most once for each record they read. So the later file can hold a text at
   most as often as the named inputs hold it. Where it holds the text more
   often, at least one of the copies came through a file the caller did not
   name, and no rule can tell which copy. Then **no copy of that text gets an
   edge**. Each of them keeps no recorded origin, and the result counts them in
   `surplus` (and in `no_edge`).

   The other things a surplus copy could get, and what each would say:

   - a "recovered" or "candidates" edge to the parents that were found, as
     before. A "candidates" edge says the parent is one of those named; for a
     surplus copy that is not verified, and an answer for a source none of the
     candidates belongs to would count the link as a link to other sources and
     could be "does not contain" for the very source the unnamed file came from;
   - an edge labelled "unknown" that keeps the found parents as leads. The
     answer would be "may contain" as well, but a stored edge is not replaced,
     so the run that names the missing file could no longer write the right
     edges.

   With no edge, the answer is "may contain" naming "missing origin" for any
   source the file does not hold through another record (plan 2.2(a)), and a
   later run that names every input writes the edges. The count is taken inside
   one later file: a text one input holds once and two later files hold once
   each is not seen as surplus by either call.

A line that holds no text (a blank line, a line that does not parse, a text that
is empty after the normaliser) has no fingerprint and gets no edge. The result
counts it in `no_edge` and in `no_text`. The text of a line is read by
`capture.line_text`, the one definition in the tool.

**Only the named files are searched.** The caller says which registered files the
step can have read. The rest of the store is not looked at, and nothing is ever
matched by similarity (plan L5). So "exactly one candidate" means one among the
files named: a twin in a file the caller left out is not seen, and the edge is
then "recovered" where the full list would have given "candidates". Name every
file the step can have read.

**What the labels decide** is the proof's business (docs/coverage.md, plan 3.2),
not this package's:

- a "recovered" edge is a record-level link, walked like an observed one;
- a "candidates" edge gives "may contain" for each candidate's source, with the
  candidate sources listed in the answer (plan L3). Two cases are stronger: when
  every candidate belongs to the request the link shows "contains", and when none
  does it counts as a link to other sources. Identical texts inside one source
  are such a case: the link decides a request about that source and cannot decide
  a request narrowed to one of the twins;
- a record with no edge fails the proof at that record, for every source.

## 4. A copy step

`recover_copy_step(store, later, earlier, *, step, text_field="text")`

`later` is the file the step wrote and `earlier` the list of files it can have
read. The matching rule of section 3 is applied to every line of `later`; the
edges are handed to the store in one call.

A copy step is confirmed on normalised texts, so the store's normaliser must be
registered. When it is not, the call fails with "unknown normaliser" and writes
nothing. A copy step does not degrade.

## 5. The sanitizer

`recover_sanitizer(store, later, earlier, *, step, log_path=None, log_input=None,
text_field="text")`

`earlier` is the one file the sanitizer read and `later` the file it wrote.

**Re-running the function.** A sanitized text is not the text it was made from,
so the raw records cannot be compared. The recovery relies on the store's
normaliser being the pipeline's own sanitizer function (plan F8, D-17):

- every fingerprint in the store is taken after that function, so the lookup of
  step 1 already holds the function's result for each line of the earlier file;
- for the direct comparison of step 2 the function is run on both texts.

A record whose text holds control characters is therefore matched to its
sanitized line although the two raw texts differ.

**The log.** `log_path` is the sanitizer's own log of the lines it skipped, JSON
lines with one row per skipped line:

| Field | Meaning |
|---|---|
| `line` | the line of the sanitizer's input file, counted from 1 |
| `reason` | "empty line" or "sanitizer skip", spelled as the tool's drop reasons |
| `input` | which input file the row is about, when one log covers several files |

Other fields are not read. The pipeline replica writes this form
(docs/replica.md, section 2), with one log per layout and the path of the
sanitizer-input file in `input`.

- `log_input` is the value the rows carry in `input` for the earlier file. Rows
  about other files are passed over. With no `log_input` the log must be the log
  of one file whose rows carry no `input` field; a row that names a file is then
  refused. Which file a row is about is never guessed.
- Every row for the file becomes a **drop** of `step`, recorded for the later
  file (`store.add_drops`). A line logged twice with one reason is one drop. The
  later file is the first registered artifact that lacks the line, and the line
  is recorded once: a log whose lines the store already holds as drops for a
  shard is refused with "drop already recorded" (`DropAlreadyRecorded`), and so
  is a second step name for a line and later file that hold a sanitizer's drop
  already. A shard holds such a drop when `tokenmap.recover_map` read this log
  while only the sanitizer's input was registered. Run the recovery with no log
  to record the edges alone. A drop of the line for another JSONL file is no
  conflict: the sanitizer ran twice. docs/interfaces.md section 5.4 gives the
  rule for both packages.
- A logged line is **no candidate parent**: the step's own log says it wrote
  nothing for that line. Where the input holds one text on two lines and the log
  names one of them, the surviving line gets a "recovered" edge to the other.
- A row with another reason, a line that is no whole number from 1, a row that is
  not a JSON object, a line logged with two reasons, or a line past the end of
  the earlier file is refused. Nothing is written then.
- `read_sanitizer_log(path, input=None)` reads a log on its own and returns
  `LoggedSkip(line, reason)` entries in line order.

Without a log the call still matches records; it then records no drop.

**Accounting for the input.** `result.input_lines_unaccounted` counts the lines
of the earlier file that no edge of the run names and the log does not name
either. With a complete log and a sanitizer that writes one line per surviving
line it is 0. A skipped line with no log row shows up here.

The effect downstream: a line of the sanitizer's input that came from no source
(a line that does not parse) leaves that file at "may contain". The sanitizer
dropped it, so it is not in the sanitizer's output, and that output can be traced
whole.

## 6. When the normaliser is not available

The sanitizer can be re-run only when the function the store names is registered
in the process (`waterwhite.normalise`). When it is not, `recover_sanitizer`
matches no record and reports one of two named outcomes (plan 3.3, F8). It never
guesses a record-level link, for example from line positions.

| `result.outcome` | When | What is written |
|---|---|---|
| "record-level" | the normal case | edges and drops as above |
| "file-level" | the function is missing, and a file family or the step's own statement supports the link (below) | by the second way: one file-level link from the later file to each member (`store.add_source_link`, derivation "file"); and the log's drops |
| "unknown" | the function is missing, and neither supports it | the log's drops, nothing about the records |

A file-level link says that a file is known to derive from a source (plan 3.2).
That the later file was written from the earlier one is, in this call, the
caller's statement. The link is not written on that statement alone. It is kept
where one of two things supports it:

- **A file family.** The later file itself already holds a file-level link to a
  member (it was imported with a file family, `capture.import_jsonl`). That link
  is the answer; the run reports the member and writes nothing.
- **The step's own statement.** The sanitizer's log was given, the lines it
  names for the earlier file and the lines of the later file add up to the lines
  of the earlier file, and the store ties the earlier file as a whole to at
  least one member (a member whose part is the whole file, or a file-level link
  the earlier file holds). The later file then gets a file-level link to each
  such member. Its evidence text names the step, the earlier file's instance id,
  the log's count and the missing normaliser.

Otherwise the outcome is "unknown": with no log, with a log whose count does not
fit, or with an earlier file the store ties to no member. `result.reason` says
which. The count is a weak check: it shows that the log and the two files fit
each other, not that every line of the later file came from the earlier one.

- `result.reason` holds the error name "unknown normaliser", its detail and,
  for "unknown", what was missing.
- No record file is opened; `result.recovered` and `result.candidates` are 0 and
  every record counts in `result.no_edge`.
- A file-level link decides a request about the whole source and nothing else
  (plan 3.2, L1): the file answers "contains" for that source, a request narrowed
  to records answers "may contain", and so does a request about any other source.
- With the outcome "unknown" the records keep no recorded origin, and a query
  about the file answers "may contain" naming "missing origin". A later run with
  the normaliser available recovers the record-level links.

A file can be registered only while the normaliser is available (section 2), so
this case arises for files that were registered earlier, or on another machine.

## 7. The result

Every call returns a `RecoveryResult`.

| Field | Meaning |
|---|---|
| `kind` | "copy step" or "sanitizer" |
| `step` | the step named by the caller |
| `artifact`, `instance_id` | the later file |
| `outcome`, `reason` | section 6; "record-level" and `None` in the normal case |
| `records` | the lines of the later file |
| `recovered` | records that got a "recovered" edge |
| `candidates` | records that got a "candidates" edge |
| `no_edge` | records that got no edge; `no_text` of them hold no text, `surplus` of them are copies of a text the later file holds more often than the named inputs do (section 3) |
| `collisions_rejected` | candidate pairs with one fingerprint and different texts |
| `drops` | the lines the log names |
| `input_lines_unaccounted` | section 5; `None` for a copy step and for a degraded run |
| `edges_written`, `drops_written`, `links_written` | what this run added to the store |
| `members_linked` | the members a degraded run tied the file to |
| `files_read` | every file the run opened: registered files by a registered name, the log by its path |
| `wrote_nothing` | true when the three written counts are 0 |
| `dry_run` | true for a call with `write=False`: nothing was written, and the three written counts are 0 whatever the store holds |

`recovered + candidates + no_edge == records`. `to_json()` gives the same fields
as a JSON object.

## 8. Writing

A recovery of links (the edge recovery and the degraded one) writes through three
methods of the store and no other:
`add_recovered_edges` (all edges of the later file in one call),
`add_drops` (all drops in one call) and `add_source_link` (one call per member, a
degraded run only). Each call is one transaction. None of them changes an
instance id (D-05), and a recovered edge may name a record outside the inputs the
artifact was registered with (D-15).

Order: everything is read and checked first; then the edges are written, then the
drops. A degraded run writes the drops, then the links. With `write=False` none
of the three methods is called. A dry run does not see a refusal the store would
give at the write (an edge that would replace a stored one).

**A second run.** Running the same recovery again finds the same links. The store
passes over an edge, a drop or a link it already holds, so nothing is added:
`edges_written`, `drops_written` and `links_written` are 0 and `wrote_nothing` is
true. The counts `recovered`, `candidates`, `no_edge` and `drops` are the same as
in the first run, because they say what the run found, not what it added.

**An interrupted run.** A process that stops between two of the calls leaves the
earlier call stored whole and the later one not at all. Running the recovery
again completes it.

**Nothing is replaced.** A record that has an edge keeps it. A run that derives
another edge for such a record (the file was captured, or it was recovered before
against another list of files) is refused by the store, whole: no edge of that
call is written. The same holds for a log that gives a stored drop another reason;
that is found before anything is written.

## 9. How the files are read

A file can hold a hundred thousand lines. The recovery keeps no text.

- The fingerprint lookup is built from the store's occurrence records.
- The later file is read once, as a stream, line by line.
- A candidate file is opened only when one of its records has a fingerprint the
  later file holds. Its first use reads it once: the SHA-256 of the file, and the
  byte at which each line starts (one whole number per line). A candidate line is
  then read by a seek.
- Every file read is held against the digest its artifact is registered with. A
  file that changed after registration stops the run with "artifact digest
  mismatch" before anything is written.
- Every text read is fingerprinted again and held against the fingerprint the
  store holds for that line. A difference means the file was registered with
  another text field, or after another function; the run is refused.

What is held in memory: one fingerprint per line of the later file, the lookup
(a fingerprint and a place per matching line), one offset per line of each
candidate file that was opened, and the edges to write.

One measurement, on a shared laptop, of a merge of two files of 50,000 lines into
one shuffled file of 100,000: registering the later file 2.3 seconds, the first
recovery 12.1 seconds, the second (which writes nothing) 10.8 seconds, about
165 MB at the peak for the whole process, including the script that made the
files. It was taken once and is no benchmark.

## 10. What is refused

Every error is a class of `waterwhite.errors`. Nothing is written when a call is
refused, with the one exception named in section 8 (an interrupted run).

| Error | When |
|---|---|
| "invalid record" (`InvalidRecord`) | a file that is not registered or is not a JSONL artifact; no possible input, one named twice, the later file named as its own input, the inputs given as one text; no step; a role a found file cannot have; a text read that does not give the stored fingerprint; a log row that cannot be used, a log that does not fit the file, a log with no file to read; an edge or a drop that would replace a stored one |
| "drop already recorded" (`DropAlreadyRecorded`) | a sanitizer's log whose lines the store already holds as drops for a shard, or for the later file under another step name |
| "unknown normaliser" (`UnknownNormaliser`) | `register_found_file` and `recover_copy_step` when the store's normaliser is not registered |
| "artifact digest mismatch" (`ArtifactDigestMismatch`) | a file that no longer has the digest it was registered with |
| "map count mismatch" (`MapCountMismatch`) | a file whose line count differs from the store's records of it while its digest matches |
| "write refused" (`WriteRefused`) | a file that is not at its recorded place; a store opened read-only |

Everything `capture.import_jsonl` and the store refuse applies as well.

## 11. Limits

- **The list of inputs is the caller's.** A file left out is not searched, and a
  twin in it is not seen (section 3). The count of section 3 shows a missing
  file only where the later file holds a text more often than the named inputs.
- **Surplus is counted inside one later file.** A text that one input holds once
  and that two later files hold once each is not seen as surplus by either call.
- **Twins are not separated by position.** A sanitizer keeps the order of its
  lines, which could tell twins inside one file apart. This package matches by
  text only, so such twins stay "candidates".
- **One text field per call.** The later file and its inputs are read with the
  same `text_field`. Files registered with different fields are refused.
- **A found file's role is the caller's word.** The store keeps the role it is
  given; nothing checks that a file registered as a release is one.
- **A stored link is not improved.** A recovered edge or a drop cannot be
  replaced by a later, better one (docs/store.md, "Limits"). That is why a
  surplus copy gets no edge.
- **The degraded link rests on a count** when it rests on the step's log
  (section 6).
- **A dry run does not see the store's refusals** (section 8).
- **Speed at scale** beyond the one measurement of section 9 is not measured.
