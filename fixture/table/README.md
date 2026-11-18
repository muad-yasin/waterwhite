# The hand-written table

This folder holds the expected answers of the hand-checked fixture for Checks 1 to 9
(plan 5.5). Every row was written by hand from the plan's rules, before the tool
existed. No row comes from an output of the tool (plan W5). A test reads a row,
puts its question to the tool and compares.

Files: `check01.toml` to `check09.toml`, one per check. The fixture itself is
described in `fixture/spec/fixture_spec.toml` ("the spec").

Plan sections served: 5.1 (the fixture and where expected answers come from), 5.5
(the checks), W5 (the wall), and through the rows 2.1 to 2.4, 3.2 to 3.5, 4.1 to
4.4, Section 6 and Decisions 12, 13, 18, 20, 21, 22, 23, 25, 26 and 29.

## Core result

The core is what the tool answers with every extension switched off (plan 1.4). The
core checks' answers are the rows with `core = true`: every row of Checks 1 to 4, 6,
8 and 9, and the rows of Checks 5 and 7 that are not extension rows.

The extension rows carry `core = false` and name their extension. They are exactly
twelve: the four rows `c05.g.model-entry-removed.on.E/shard-3`, `.gen/P` and `.gen/T`
and `c05.withdraw-C.on.E/shard-3`, and the five rows `c07.R3.*` (extension
`generating-model`), and `c07.ban.R1`, `c07.ban.R2` and `c07.ban.scope` (extension
`content-ban`). An extension row is put only while its extension is on.
While it is off the row is not run and is not a failure (plan 5.5). What the tool
does while an extension is off is a core row: `c05.g.model-entry-removed.off.E/shard-3`
and `c07.ban.off`. One case of the generating-model extension is not a row of the table:
the stop of a query about R3 when a file that only a model's history reads has changed
on disk. It is pinned by the extension's own test (E2-W18) and is unchecked by
the table.

The core result is one file: for every core row, the answer the tool gave to the
row's question in a run with every extension off. It is committed at the last commit
of the core, before any extension is switched on. Later, after work on an extension,
a fresh run with every extension off is compared with that file, row by row. A
difference means the work changed a core answer.

The file is an output of the tool. It says what the core answered; it is never where
an expected answer comes from. The expected answers stay the rows of this table
(plan W5).

## 1. A row

```toml
[[row]]
id = "c08.withdraw-A.E@9"     # stable; a test quotes it
question = "..."              # the question in words
plan = "5.5 Check 8; 2.4"     # the plan passages that give the answer
core = true                   # false for an extension row; then `extension = "..."`
given = { ... }               # what is asked, of what, in which state
expect = { ... }              # the one expected outcome
source = "..."                # where the expected value comes from
```

A row may also carry `expect_under`, a table of fallback name to the `expect` that
holds when that fallback is in force (section 8), and `rule`, a sentence that says
which rule of the plan gives the answer when `plan` alone would not show it.

`source` starts with one of the plan's own categories:

| Start of `source` | Meaning |
|---|---|
| `hand-written table` | reasoned by hand from the plan and the spec; the value is in the row |
| `the oracle's log` | read from the oracle's log after the seed search; the row names the pending spec key |
| `the plant list` | counted from the generator's plant list by the counting script; the row names the script's output field in words |
| `the withheld files` | the cache files held back in Check 2 (i) |
| `a from-scratch reference` | the reference side of Check 9, built from clean sources |
| `the replica's records` | a manifest, a sanitizer log or a run record the replica wrote; outside the tool |

## 2. `given`

| Key | Meaning |
|---|---|
| `store` | which build is asked (section 4). Left out: `main` |
| `state` | `unmutated`, or the id of a fault, set-up or case from section 7. Left out: `unmutated` |
| `request` | a request name from section 3. Left out when no query is made |
| `artifact` | the artifact asked about, by its name (docs/interfaces.md 9.2, the spec). `(listing)` is a query with no artifact |
| `record` | a record handle (section 5) when the question is about one record |
| `run`, `segment`, `step`, `launch` | the part of a run the question is about |
| `set`, `layout` | a generated set (`10k`, `100k`) and its layout (`partitioned`, `round-robin`). `generated`, in Check 6, stands for each generated set the check is run on; the row's source names the output of the counting script that gives the value for that set |
| `map` | which tokenization map: `capture`, `position recovery`, `token-matching recovery`. One shard can hold all three in one store, side by side (DECISIONS.md D-20) |
| `extensions_on` | the extensions switched on for the row. Left out: none |
| other keys | set-up details of one row, in plain words |

## 3. Requests

| Name | The request (docs/interfaces.md 6) |
|---|---|
| `withdraw A` | `{"kind": "withdraw", "source": "A", "members": ["A/r1"], "records": []}` |
| `withdraw C` | the same for source C, member `C/r1` |
| `withdraw A, record X@A` | source A narrowed to the one record X@A of `src/A` (plan 2.3: a list of records). A record-level request |
| `withdraw B, record X@B` | source B narrowed to the one record X@B of `src/B`. A record-level request |
| `withdraw W` | source Z narrowed to the one record W, the shared-token document of search condition 12: an unrelated record of Z in shard 3, first read through one token at a step from 7 to 9 of run E. The spec names it under `positions.shared_token` (DECISIONS.md D-13 as amended by D-24) |
| `ban X` | a content ban on the text of X. An extension request |

"Affected" under `withdraw A` means A's records and everything derived from them:
X@A, SA, P, T and S (plan 5.1).

## 4. Stores

| `store` | What it holds |
|---|---|
| `main` | the fixture built by the replica with capture on: `src/A`, `src/B`, `src/C`, `src/Z`; `gen/P`, `gen/T`, `gen/S`; `R1`, `R2`; `E/shard-1` to `E/shard-3` with their capture maps; the mixtures `E/mix-1`, `E/mix-2`, `D/mix-1`, `Y/mix-1`, `K/mix-1`; runs E, D, Y and K with every step trace; the checkpoints `E@6`, `E@9`, `D@3`, `Y@6`, `K@3`, `K@8`, `K@40`. Run E is registered with the spec's step record, so step 8 is lowered. Every extension is off |
| `rr` | run E rebuilt in the round-robin layout: `E-rr/shard-1` to `E-rr/shard-3`, `E-rr/mix-1`, `E-rr/mix-2`, the two segments, and the one checkpoint `E-rr@6`, which the second segment resumes from (DECISIONS.md D-24). No row asks `E-rr@6` anything: the plan tabulates no checkpoint answer on this layout (4.5) |
| `swap` | the main store's sources and generation outputs with the release built again with dedup winner B and registered as `R1-swap`, beside `R1` |
| `no-capture` | the same files made with no wrapper (unseeded shuffles, ids dropped at tokenization) and recovered afterwards: sources ingested, copy steps recovered by fingerprint, shards recovered in position mode. The generation outputs are registered as plain files with no edge; they are not imported, so they are no origin. The table asks this store about files and shards only (section 9, reading 7) |
| `import` | the sources plus `import/T`, the text-only import of T with the file family `A/r1` |
| `scratch-500` | Check 4 only: a store with source A registered (any member), a 500-line file ingested under a source of its own, and that file's tokenized shard `c04/shard-500` of 500 documents |
| `scratch-notokens` | Check 3 only: a store with `src/Z` ingested and the shard `c03/shard-notokens` of the set-up `t.no-tokens-line` (section 7), with its capture map and its position-mode recovery map |

`successor of R1` and `successor of E/shard-3` are the artifacts the replica builds
from the rebuild report of `withdraw A` (plan 4.4); the replica names them.

Every artifact of docs/interfaces.md 9.2 has a row under `withdraw A`, unmutated:
the main store's and the `rr` store's in `check05.toml` (ids `c05.unmutated.*`),
`import/T` in `check06.toml`, `R1-swap` in `check07.toml`. There are three
exceptions. The checkpoint `E-rr@6` of the `rr` store: no answer is tabulated for
it. The tokenizer inputs of the shards (`E/shard-1.input` and so on), which
docs/interfaces.md 9.2 also names: no row asks them anything. Check 4's scratch
shard `c04/shard-500`: its rows ask it only in a state of Check 4 (section 7),
never unmutated.

## 5. Handles

| Handle | Record |
|---|---|
| `X@A`, `X@B` | the two copies of the passage X: `A-001` in `src/A`, `B-001` in `src/B` |
| `SA`, `SC` | the passages S is made from: `A-002` in `src/A`, `C-001` in `src/C` |
| `P`, `T`, `S` | the paraphrase of X@A, the translation of P, the summary of SA and SC |
| `X` | the row of a release that holds the passage X (in `R1`, `R1-swap` and their successors) |
| `U-<nnn>` | the unrelated record `Z-<nnn>` |
| `W` | the shared-token document: the unrelated record of Z in shard 3 that the spec names under `positions.shared_token` |
| `G6`, `G9`, `GU`, `GM` | the four records of release R3, which exists only with the generating-model extension on. Plan 5.1 describes R3; the spec names its four records and jobs (`[extension_releases.R3]`), so the `c07.R3.*` rows can be put while the extension is on |

Jobs are named as in the spec: `paraphrase` (makes P), `translation` (makes T),
`summary` (makes S).

A value written `"pending:<key>"` stands for the spec item `<key>`, filled from
the oracle after the seed search. A test resolves it from the spec. The spec's
`[pending]` list is empty since the seed search: every such key holds a value
there (`pending:positions.shared_token.record` is the record id of W). A test that
meets a pending item that is still empty reports the row as not checked, not as
passed.

## 6. `expect`

The answer fields have the names of docs/interfaces.md section 6. A test compares
only the fields a row holds.

| Key | Meaning and how it is compared |
|---|---|
| `answer` | one of the four answers, as `vocabulary.ANSWERS` spells them |
| `reason` | for "not covered": `record missing` or `not registered` |
| `names` | for "may contain": the named cause, a `vocabulary.NAMES_*` string. The extension rows use three causes of their own: "generating-model entry missing" (the log has no model entry), "generating model exposed" (the model is a checkpoint that answers "contains" or "may contain") and "generating model not registered" |
| `names_detail` | the reason beside "data-order trace missing", a `vocabulary.TRACE_MISSING_*` string. There are two (plan 3.5): "no recorded digest" and "digest mismatch" |
| `failing` | the link the answer names. Keys, each compared when present: `condition` (the letter of plan 2.2), `artifact`, `record` (a handle), `segment`, `step`, `job` |
| `candidates` | the candidate sources, compared as a set |
| `chain` | named hops of the evidence chain, from the artifact to the source record, written `artifact:handle` or `segment:step n`. The answer's chain must pass them in this order; hops through files this table does not name (the replica's intermediate files) may lie between them |
| `basis` | a `vocabulary.BASIS_*` string |
| `notes` | the full set of notes the answer carries, each a `vocabulary.NOTE_*` sentence, compared as a set. `[]` means the answer carries none. A row that leaves `notes` out says nothing about them |
| `untraced_steps` | `[segment, step]` pairs, compared as a set |
| `first_exposure` | `{segment, step}` |
| `scope` | for "does not contain". Keys, each compared when present: `source`, `members`, `records`, `steps` (a list of `[segment, first step, last step]`), `generation_edges` (the jobs whose established bound the proof rests on) |

Other keys, for rows that expect something else than an answer:

| Key | Meaning |
|---|---|
| `error` | the fixed name of a `waterwhite.errors` class; nothing is printed on standard output |
| `exit_code` | the exit code of the command line |
| `stdout_empty` | true: no answer is printed |
| `one_of`, `never` | the plan allows a set of outcomes; any of `one_of` passes, anything in `never` fails |
| `diff` | `empty`: the tool's rows equal the rows of the named source, row for row |
| `action` | a rebuild action of `vocabulary.ACTIONS`, or `unresolved` |
| `label`, `derivation`, `verification` | the stored fields of a link or a map |
| `counts`, `covers`, `not_covered` and plain-word keys | values named in the row's question |

## 7. Faults, set-ups and cases

Each fault is applied to a fresh copy of the store it names, one fault per copy.
"Outside the write API" means the SQLite file is edited directly. The answers are
then asked with the request the row gives.

### Check 5, under `withdraw A` unless the entry says otherwise

| Id | Target, in fixture terms | How it is applied | Why this target |
|---|---|---|---|
| `f.trace-deleted` | the step trace of step 3 of `E/seg-1` | outside the write API: the trace and its sample rows are deleted; the registered step stays | named by the plan |
| `f.trace-half` | the last sample (sample_pos 3) of the trace of step 4 of `E/seg-1` | outside the write API: its coverage mark is set to 16 of 33 tokens; the trace still counts 4 samples | the plan names the step; the last sample mirrors the lowered sample of step 8 |
| `f.edge-deleted` | the stated edge of P in `gen/P` (P made from X@A) | outside the write API: the edge and its input rows are deleted; P's occurrence and the harness log stay | it is the root of the two-step derivation X to P to T, so one missing link must be found by the shard, the mixtures and the checkpoints that hold or read T, while R1 keeps a second recorded chain to A and checkpoints that never read T keep their negative |
| `f.map-deleted` | the capture map of `E/shard-1` | outside the write API: the map's binding and rows are deleted; the shard's artifact row and the snapshots that name it stay. The store holds no recovery map for it | shard-1 holds no affected record, so the fault turns negatives into "may contain", the direction in which a tool that reads a gap as absence fails; mixtures that hold T keep "contains" (Decision 20) |
| `f.edge-file-level` | the stated edge of T in `gen/T` (T made from P), asked with `withdraw A, record X@A` | outside the write API: the edge's label is set to `file-level` and its derivation to `file`; nothing else changes | a translation whose file is known to derive from a source while its own parent is not known is the plan's own example of file-level evidence (L1) |
| `f.digest` | the declared digest of the capture map of `E/shard-3` | outside the write API: the first hex digit of the stored digest is replaced by another | Check 5 walks the proof conditions and this is condition (d) on a map; Check 4 injects the count mismatch and the three other loud faults, so the map-digest mismatch is injected here. Shard-3 holds T, so the stop is asked of artifacts that would answer "contains" |
| `f.label` | the label field of the capture map of `E/shard-2`, the shard with the skipped line (spec, `sanitizer_skip`) | outside the write API: the map's label is set to `observed`; its derivation (`position`) and verification (`passed`) stay; the labels of its rows stay | the plan asks for a position-derived map on a shard with one skipped line; the spec puts that line in shard-2 |
| `g.bound-field` | the generation edge of P, asked with `withdraw C` | outside the write API: its bound field is set to not established; the log stays | named by the plan |
| `g.log-outside-read` | the stored harness log of the job `paraphrase`, asked with `withdraw C` | outside the write API: one read is added that names X@B of `src/B`, a record outside the hand-off, not refused | named by the plan; the read is one the harness let through, because a refused attempt does not break the bound (DECISIONS.md D-07) |
| `g.log-removed` | the generation edge of P, asked with `withdraw C` | outside the write API: the edge's link to its harness log is removed and the log's rows are deleted; the bound field stays at established | the plan says "an edge"; P's edge is the one whose job the expected answer names |
| `g.model-entry-removed` | the stored harness log of the job `paraphrase`, asked with `withdraw C` | outside the write API: the model entry (`none`) is removed, so the log has no model entry | named by the plan; an extension row, and one core row for the off state |
| `s.in-scope-only` | the input SC of the edge of S in `gen/S`, asked with `withdraw C` | outside the write API: SC is marked as handed and not stated as used | named by the plan |
| `s.file-level-link-B` | the occurrence of P in `gen/P`, asked with `withdraw B, record X@B` | through the write API: `add_source_link` from P's occurrence to the member `B/r1`, label `file-level`, evidence "file family: paraphrases of passages that source B supplies". P keeps its observed edge to X@A | P is the one record with an observed link to A whose text B could also have supplied |
| `h.outside-read` | the job `paraphrase`, run again in the harness | the job is handed X@A and asks the harness for X@B of `src/B` | named by the plan |
| `q.unregistered` | the name `E@12` | a query names it; run E has no registered checkpoint after step 12 | a name no record of the store holds |

### Check 4

| Id | Target | How it is applied |
|---|---|---|
| `w.kill-after-100` | the map of `c04/shard-500` in the `scratch-500` store | the writer of the 500-row map is killed after 100 rows; the store is reopened |
| `w.retry` | the same map | after `w.kill-after-100`, the write is made again on the reopened store, declaring 500 rows and the shard's digest |
| `w.declare-499` | the same shard in a fresh `scratch-500` store | `write_map` is called declaring 499 rows |
| `l.count-499` | the stored count of the complete 500-row map | outside the write API: set to 499 |
| `l.edge-parent` | the parent reference of the stated edge of P in `gen/P` (it names X@A), in a copy of the main store | outside the write API: the reference is changed to an occurrence id that no occurrence has. No occurrence row is deleted, so no count or digest changes and only the edge-parent check can fire |
| `l.trace-count` | the trace of step 5 of `E/seg-1`, in a copy of the main store | outside the write API: its last sample and that sample's rows are deleted and its consumed-sample count is set to 3 |
| `l.artifact-digest` | the file of `R2`, in a copy of the main fixture folder | one byte of one line's text is changed on disk; the line count stays; the store is not edited |

### Check 3

| Id | Target | How it is applied |
|---|---|---|
| `t.no-tokens-line` | the shard `c03/shard-notokens` in the `scratch-notokens` store | the test writes a tokenizer input of five lines: lines 1, 2, 4 and 5 hold the sanitized texts of U-001 to U-004 in that order, and line 3 is a record whose text is the empty string. The line is put there past the sanitizer, which would have dropped it as an empty line; no pipeline of the fixture or of a generated set produces one. The input is tokenized by Megatron's preprocessing with the fixture's settings (the spec, `megatron` and `tokenizer`) and the shard is registered with its capture map; the position-mode recovery map is then recovered from the same input with no ids given |

| `t.twin-swap` | every planted pair of the 100k set whose two members lie in one shard | before the position-mode recovery is run, the pair's two lines change places in the shard's tokenizer input; nothing else changes |
| `t.verification-fails` | one shard of the 100k set, the first in name order | after tokenization and before the recovery is run, line 1 of the shard's tokenizer input is replaced by a record whose text is "This line was changed after the shard was tokenized." The line count stays. The position-mode recovery and the token-matching recovery are then run on the changed input |

### Checks 7 and 9

| Id | Target | How it is applied |
|---|---|---|
| `r.successor-registered` | the main store | the replica builds the repaired side from the rebuild report of `withdraw A` (plan 4.4) and the successors are registered in the same store beside the historical artifacts; the question is then put to the historical `R1` |
| `r.descendant-kept` | the rebuild report of `withdraw A` on the 10k set | before the replica builds the repaired side, the action of the first planted translation (`T-000001`) is changed from retire to keep; the reference side is built as always |
| `r.unrelated-paraphrase-retired` | the rebuild report of `withdraw A` on the 10k set | before the replica builds the repaired side, the action of the first paraphrase of an unrelated record (`UP-000001`, the first id of the counting script's list of the generated records that stay) is changed from keep to retire; the reference side is built as always |

### Check 2, each registered in a fresh store

In every case the mixture snapshots are kept and the operator's step record of the
spec is supplied, so step 8 of run E is lowered as in the main store.

| Id | Starting state |
|---|---|
| `c2.i` | the cache files of runs E, D and Y are withheld; settings, seeds and the recorded digests are kept |
| `c2.ii` | as `c2.i`, and the seed in the settings of runs E, D and Y is replaced by its value plus 1 |
| `c2.iii` | the cache files, the settings and the recorded digests of runs E, D and Y are withheld. The reason beside the named cause is "no recorded digest" |
| `c2.iv` | run K's cache files are kept and the shuffle digests are removed from its run record |

## 8. Fallbacks

| Name in `expect_under` | In force when |
|---|---|
| `F1` | Megatron's preprocessing does not write one document per line in input order: maps are built by token matching (plan F1, 3.3) |
| `F3` | a replay does not regenerate the cache arrays byte for byte (plan F3) |
| `F5` | the oracle does not run on the laptop: Check 1 is judged against the committed golden log (plan F5) |
| `F7` | no step comparison is possible: the exact-rerun restart is reported as the phase start (plan 4.3, F7) |
| `cyclic-wrapper-route` | the wrapper cannot record the shuffle digests of a cyclic run (plan 3.5): every checkpoint of run K answers "may contain" naming "cyclic order not recorded" |

A result obtained under a fallback is reported with the fallback (plan 5.5).

## 9. Readings the table follows

Where the plan's text leaves room, the table follows these readings, each stated
once here and used in every row. Readings 1 to 4 are chosen readings, stated in
DECISIONS.md D-22; each is given with the other reading the text allows and with a
row that would answer differently under it.

The rows also rest on these decisions of DECISIONS.md, which are choices the build
made and not readings of the plan's text:
- D-16: a shuffle digest of a cyclic run covers a whole epoch's order for one
  rank, whatever part of it a launch read (`c08.K.digests-match`).
- D-18: the map check compares a binding with the store's own records and the
  artifact check compares the registered digest with the file (`c04.count-499`,
  `c04.artifact-digest.R2`, the rows `c05.f.digest.*`). D-26 amends it: when the
  file's digest is the registered one, the declared count is also compared with
  the records the file holds.
- D-19: a release row that copies the withdrawn source's own record and has no
  equivalent copy is retired; "records removed" is a count (`check09.toml`).
- D-20: one shard can hold its capture map and one recovery map per derivation
  mode; a reader takes the capture map, else the verified position-mode map,
  else the verified token-matching map (`c03.map-fields`; the fault
  `f.map-deleted` is applied to a shard with no recovery map).
- D-21: the sole record a job is handed is a parent of its output; with two or
  more handed records only the stated ones are (`c05.s.in-scope-only.R1`, where
  the job `summary` is handed two).
- D-23: the 10k and 100k sets also hold paraphrases of unrelated records, which
  no request puts in scope and which stay on both sides of a repair
  (`c09.10k.wrong-report.unrelated-paraphrase-retired`).
- D-24: the failing link a checkpoint's "may contain" names is the earliest one
  of its history (`c05.f.edge-deleted.E@9`, `c05.f.edge-deleted.D@3`,
  `c05.f.edge-file-level.E@9`, `c05.f.edge-file-level.D@3`).
- D-26: an artifact's answer carries the process-evidence note when any chain
  that puts a record of the artifact in scope passes a generation edge, while the
  answer prints the first chain only (`c05.unmutated.R1`, `c07.withdraw-A.R1`).

The readings:

1. A checkpoint is answered from the records its steps consumed (plan 2.2(a),
   2.2(g), 2.4, 4.5). A fault on the chain of a record the checkpoint's mixture
   holds and its steps never read leaves its "does not contain" in place. The
   other reading: a checkpoint's closure holds everything its mixtures hold. Under
   it `c05.f.edge-deleted.Y@6` would be "may contain".
2. "Not covered" is only for the artifact whose own record is missing or that was
   never registered. Everything downstream answers "may contain" (plan 2.1,
   Decision 21), unless a recorded chain the fault does not touch shows
   "contains", which wins, for a mixture as for a checkpoint (plan 2.2 closing
   paragraph, Decision 20). The other reading: the recorded chain wins for a
   checkpoint only. Under it `c05.f.map-deleted.E/mix-2` would be "may contain".
3. A loud fault stops a query when the faulted record lies in the ancestor closure
   of the artifact asked about, and it stops the listing. A query about an
   artifact outside that closure prints its answer (docs/interfaces.md section 4).
   The other reading: a loud fault anywhere in the store stops every query. Under
   it `c05.f.digest.E@6` would be a stop.
4. A file-level link decides a request about the whole source and cannot decide a
   request narrowed to records (plan 3.2, L1): the file answers "contains"; each
   of its records answers "may contain" when the request is narrowed to it, and
   the rebuild report lists it as unresolved (plan 4.1). The other reading: a file-level link
   gives "may contain" for the file too. Under it `c06.import.withdraw-A` would be
   "may contain".
5. For a descendant checkpoint the rows compare the link the answer names
   (`failing`) and leave `names` open, because "the step is untraced" and "an
   ancestor answers may contain" are both true of it (plan 2.4).
6. The rows of run K under a fault of Check 5 carry the fallback
   `cyclic-wrapper-route` with the answer only, since under it two links fail.
7. Check 6 asks whether everything downstream of the unrecorded translation is
   "may contain". The table writes that for the files and the shard that hold T.
   It writes no row for a mixture or a checkpoint of a run made without capture:
   such a run has no mixture snapshot (plan L6), and whether a snapshot may be
   registered afterwards is open (plan 5.8, B-4).
8. The rows of the import `import/T` split by what states them. The answer states
   what the import answers to the request narrowed to the record X@A
   (`c06.import.record-request`, `c06.import.parent-unresolved`): "may contain", the
   failing link on the record T, and the candidate source A. The answer form has no
   field for a parent; "none recorded" is read from the failing link's record and
   from the candidates. The action "unresolved", the missing link in the report's
   words and the scope that needs review are fields of the rebuild report (plan
   4.1), stated in `c06.import.review-scope`; the test of the acceptance folder
   skips that row, by name, until the report exists.

## Check 10

`check10.toml` holds the reading rules for the baselines of plan 5.5 Check 10.
Checks 1 to 9 put their questions to the tool. Check 10 puts the same questions,
without the rows of an extension that is off, to six other subjects and records
one of three grades for each question and subject: right, wrong or cannot answer.

The rules that give a grade are written here before any baseline has run on the
replica's output, so that no rule is picked after an answer is seen (plan W5).
The file holds no expected result. The plan sets no pass bar and makes no claim
in advance that the design does better than an existing tool (plan 1.2, W7): the
per-question record is the result.

Two things the plan marks as unverified stay open and are not settled by this
file: whether the Dolma toolkit's tokenizer and mixer run on the replica's chunk
files, and whether OriginBlame's code runs on the replica's output. Both tools
were installed and run once on a small synthetic input before the build. That run
is written up in the baseline run report that plan 5.5 Check 10 cites; "report
section N" below is a section of it. Neither tool has run on the replica's output.

Plan sections served: 5.5 Check 10 and its opening paragraph (the pass condition),
1.2, W5, W7, Section 7 (the dependency on the two tools), Section 13 (existing
tools).

### The six subjects and their own records

A subject is read through its own records and nothing else. "Own records" are:

| Subject (`given.subject`) | What it is | Its own records |
|---|---|---|
| `manifest` | The file-level manifest. A reference point this project builds, not an existing tool | One row per artifact (a source file, a generation output, a release, an import, a chunk file, a shard, a mixture, a checkpoint): its name, its file paths relative to the build folder, and `built_from`, the names of the artifacts whose files went into it. A mixture's row names its shards. A checkpoint's row names the shards its segment was handed and its parent checkpoint. An import's row names the file of the member its file family gives. Nothing below a file: no record, line, ordinal, step, label, count or digest |
| `trace` | The positions-only step trace. The second project-built reference point: the exposure join's own position output with every source id removed. The tracer of the published pipeline is not reused (plan W4) | Per segment: its run, its steps, its shards in the trainer's order, its parent checkpoint, the checkpoints saved in it with their steps. Per step: the number of samples it counts. Per sample: its place in the step, its rank, its tokens and its covered tokens. Per row: shard, ordinal, token start, token end. No occurrence, record, text, fingerprint, file or source |
| `dolma` | The Dolma toolkit, an existing tool: its own tokenizer and mixer, run on the replica's chunk files | Report section 2.1. The tokenizer's companion rows, one per document, in a file with no header row: `start, end, id, src, loc`. `src` is the input path as it was given, `loc` the line number counted from 1, `id` the id of the input line, and `start` and `end` index the toolkit's own token file. The mixer's stamp on each document it writes, `metadata.provenance`: the base name of the input file and the line. It has no derived-from link and does not record which copy a dedup kept (plan Section 13, existing tools) |
| `originblame` | OriginBlame, an existing tool, run on the replica's output | Report section 3.1. Section records: a registered source file with its path, author ids, licence and year, named by a hash of these. Document-index records: packed file, line hash, sources, and which call wrote the record. Token-index entries, one per tracked document in the order they were tracked: `sources` (section hashes), `token_count`, `tokenizer`, `revoked`; no id, no offset, no line hash. The forget set: one bit per token-index entry, least significant bit first, 1 for revoked |
| `dolma+trace` | A combination: the Dolma toolkit's rows plus the positions-only trace | The records of both. The trace gives (step, shard, ordinal); the companion row of that document gives its line and its id |
| `originblame+trace` | A combination: OriginBlame's records plus the same trace | The records of both. The trace gives (step, shard, ordinal); the token-index entry of that document gives its sources |

### What a run hands a subject, and its conditions

Every row names the conditions of the run it describes in `given.conditions`:

| Condition | Meaning |
|---|---|
| `sandbox` | An existing tool is installed and run only inside the sandbox of report section 4: no network, a home folder that is an empty folder inside the work folder, only the work folder writable, a cleared environment. The Dolma toolkit's `setup.sh` and `make setup` are never run. Report section 5 is the set-up to repeat |
| `relative paths` | The Dolma toolkit writes the input path as given into `src`, and OriginBlame stores the paths it is handed (report sections 2.1 and 3.1). Both are started in the build folder and handed paths relative to it: `pipeline/<layout>/tokenizer-input/<shard>.jsonl` (the chunk file of a shard), `pipeline/<layout>/shards/<shard>.bin`, `sources/<KEY>.jsonl`, `gen/<NAME>.jsonl`. Every rule compares relative paths. No local path may reach a committed file |
| `row count checked` | After every tokenizer run the number of companion rows for each chunk file is compared with the number of its lines. The tokenizer can end with exit code 0 and write no document (report section 2.2). A run that writes no row for a chunk file that has lines is a failed attempt, not an answer |
| `one process in packed order` | Each shard has an OriginBlame store of its own. One process makes the native track calls for the shard, one per document in ordinal order, and `ob clean` is run before anything is read (report sections 3.1 and 5), so that entry k of the store is the document with ordinal k. The run states that it was done this way |
| `one revoke per fresh store` | Each request is put to a fresh copy of the tracked store, by the native by-sources call, once. A second revoke of the same target un-revokes it (report section 3.2) |
| `written while the build runs` | The manifest is a record made while a build runs, from what the build reads and writes. It is not taken from the tool's store. A build made with nothing recorded (the run made without capture of Check 6) has no manifest |
| `the join's own position output` | The trace is exported from the tool's exposure join. It is the tool's own output with the source ids removed, not an independent record |

What each existing tool is handed:

- **The Dolma toolkit.** Its tokenizer is run directly on the chunk files of a
  build, one layout at a time, with its id field read: the chunk files carry the
  replica's ids. To resolve a row's id is to find the file under `sources/` or
  `gen/` of the same build that holds a row with that id; a file under `sources/`
  is the file of a member. That lookup reads the replica's files. It is not a
  record of the toolkit, which records the id and nothing about where it came from.
  The mixer is run on the same chunk files with a filter that excludes nothing
  (report section 5). Its stamp keeps only the base name of a file and the chunk
  files of the two layouts share base names, so one mixer run takes the chunk files
  of one layout of one build.
- **OriginBlame.** Sections are registered through the public call, one per file
  under `sources/` and `gen/`. Each document of a shard is tracked with the section
  of the file that holds the record its chunk-file line carries, with the
  document's size in the shard's own index as its token count. The tool stores what
  its caller passes (report section 3.1) and the caller adds no lineage of its
  own: a generated record is tracked with the section of its generation output,
  not with the sources that record was made from.

Which route answers is named before any run:

- For the Dolma toolkit: the rows of the tokenizer run on the chunk files, read by
  `src` and `loc`. The route through the mixer's output (tokenize what the mixer
  wrote, then follow its stamp back to the chunk-file line) is not the route of any
  row.
- For OriginBlame, for a request about one source: the native by-sources call with
  the sections of the files of the members the request names, then `generate-set`.
  The command-line revoke by section sets no bit in the forget set, and the revoke
  by author covers every source of that author (report section 3.2); neither is a
  route of this table. No route is named for a request narrowed to records: a
  section is a file, and a revoke by line hash was not run in the baseline run
  (report section 6).

A run by another route than the one a row names is never that row's answer. It is
recorded beside the answer with its command.

### Readings

A rule turns a subject's records into one reading:

| Reading | When | It equals |
|---|---|---|
| `contains` | a record on the route names a member the request names | the answer "contains" |
| `does not contain` | the route reached a record for the artifact and none names such a member | the answer "does not contain" |
| no record | the subject keeps records for this kind of artifact and holds none for this one | the answer "not covered" |
| stopped | a command of the subject that reads the changed record ends with an error that reports it and gives no reading | a row that expects an `error` |
| cannot answer | the route cannot be walked: the subject has no record or field of the kind the question needs | nothing; it is its own grade |

The reader adds no check of its own. A gap inside a subject's records (a deleted
row, a step with no trace row, a name with no row behind it) is noticed only where
a command of the subject reports it. Otherwise the route reads what is there, and
the grade follows from the reading.

No rule names a reading that equals "may contain": no record of the six subjects
is known to carry a mark that says a link is unsure. If a run shows such a mark,
a reading rule for it is written into this table before it is read.

A combination reads the trace's rows for the steps of a checkpoint's history, ties
each row's document to the tool's record for it and reads that record. It reads
rows. It has no rule that turns a step without rows, or a sample with an uncovered
range, into doubt.

### Grades, and what is compared

- **right**: the reading gives the expected answer of the row asked.
- **wrong**: the reading gives another answer. A gap read as absence is wrong like
  any other answer.
- **cannot answer**: the subject ran, and its own records lack the record or field
  the question needs. The missing record or field is named as the evidence.

The grade is taken against the expected-answer source of the check the question
comes from: the `expect` of the row asked, in `check01.toml` to `check09.toml`,
whose `source` names the hand-written table, the oracle's log, the plant list
counted by the counting script, the withheld files, a from-scratch reference or
the replica's records. A row that carries `expect_under` is graded against it when
the fallback it names is in force. A row that does not apply under a fallback in
force is not put.

What is compared is the fact the question asks for: the answer; for a question
about rows, positions, counts, records or actions, those values. Each row's `right`
and `wrong` say which.

Not counted against a baseline, because they belong to this tool's answers (plan
5.5 Check 10): the process-evidence note of plan 2.1, the basis note of plan 2.4,
and the words "generating-model exposure not assessed".

Also not compared, as this table reads the plan: the other parts of an expected
row that are this tool's way of saying why. These are the named cause (`names`,
`names_detail`), the failing link, the chain, the scope, the reason beside "not
covered", the name of an error and its exit code, and a label word where the
question also asks for the record itself. A question whose whole answer is a label
or a candidate set is put whole, and a subject with no such field cannot answer it.

Two expected values are read off a subject's answer by the plan's own definitions:
an artifact's action (plan 4.1: an artifact that holds affected records is
rewritten, one with no ancestor in scope is kept) and the exposure-free restart
(plan 4.3: the last checkpoint whose history holds no exposure).

### Groups

A group is a set of rows of Checks 1 to 9 that ask the same question, most often
of many artifacts. A group is named `c<NN>.<slug>`: the check its rows come from
and a short name of the question. `check10.toml` holds one `[[group]]` entry per
group, with `id`, `check`, `question` (the question in words), `rows` (the ids of
the rows it covers) and `source` (where the expected answers of those rows come
from). Every core row of Checks 1 to 9 is in exactly one group. The nine rows of the
generating-model extension (`core = false`) are in groups of their own
(`c05.model-entry-removed-on`, `c05.withdraw-C-on`, `c07.R3-records` and
`c07.R3-release`; the owner's answer Q1 of 18 Nov 2026: the cases of an extension also
go into the comparison); the
three rows of the content-ban extension are in none. The six subjects have no extension:
each is asked what its own records hold of the case.

In a rule, "a listed row" is one of the rows a group lists. Where a group lists
artifacts of several kinds, the rule says how each kind is read.

| Group | Rows | The question |
|---|---|---|
| `c01.positions` | 11 | Do the rows (step, shard, ordinal, token start, token end) reported for runs E, E-rr and K equal the oracle's log, over each whole run and at each position the fixture names? |
| `c01.step-facts` | 7 | What does the join's output state about a step or a launch: the uncovered range of the last sample of step 8, its consumed-sample count, which steps are untraced, where each launch of run K starts, and the samples of each rank? |
| `c01.E@9` | 1 | Under withdraw A, does checkpoint 9 keep "contains" while step 8 is untraced? |
| `c02.arrays` | 4 | With the cache files withheld and settings and seed kept, are the index arrays replayed for each segment byte-identical to the withheld ones? |
| `c02.replay` | 2 | With the seed changed, is the replay rejected because its digests differ from the recorded ones; with settings and digests withheld, is no replay attempted? |
| `c02.checkpoints` | 15 | Under withdraw A, what does each checkpoint answer in each cache state: caches replayed (i), replay rejected (ii), nothing to replay from (iii), run K without its shuffle digests (iv)? |
| `c02.mixtures` | 9 | Under withdraw A, does each mixture's answer stand in the cache states (ii), (iii) and (iv)? |
| `c03.preconditions` | 2 | Before Check 3 runs: is the end-of-document convention recorded in the fixture spec, and does the plant list hold an identical pair of each kind? |
| `c03.spans` | 2 | On the 100k set, is the token span of every ordinal equal to that document's size in the shard's own index, and does the total equal the index total? |
| `c03.drops` | 2 | On the 100k set, does every skipped, empty or tokenless line appear as a drop without shifting an ordinal, and does each map record the append-eod setting? |
| `c03.drops.planted-line` | 1 | On a scratch input of five lines whose third line gives no tokens, does that line appear as a drop with the reason "no tokens", without shifting the ordinal of any later line? |
| `c03.records` | 3 | On the 100k set, is the record assigned to each ordinal the planted record whose tokenization equals the shard's tokens at that ordinal? |
| `c03.pairs.same-shard` | 2 | For each identical pair placed in one shard, is each of its two ordinals labelled "candidates" with exactly the pair's two occurrences, and does swapping the pair's lines change nothing? |
| `c03.pairs.split` | 1 | For each identical pair split across two shards, is each ordinal's record the occurrence in its own shard's chunk file? |
| `c03.labels` | 2 | Is a shard whose verification fails labelled "unknown", and does every map record carry its derivation mode, verification result and label? |
| `c04.interrupted-write` | 2 | After the writer of a 500-row map is stopped part way, is the map absent or complete; and after the write is made again, is it complete and bound to its shard, or refused? |
| `c04.declared-count` | 2 | Declaring 499 rows for a 500-document shard: is the write refused and no map left, and what does a query about the shard answer then? |
| `c04.count-499` | 1 | With the count of a complete 500-row map altered to 499, what happens to a query about the shard? |
| `c04.edge-parent` | 2 | With the parent of P's stated edge pointing at no occurrence, what happens to a query about R1 and about gen/P? |
| `c04.trace-count` | 2 | With the trace of step 5 one sample short of the batch size, what happens to a query about E@6 and about E@9? |
| `c04.artifact-digest` | 1 | With one byte of R2's file changed on disk, what happens to a query about R2? |
| `c04.not-a-stop` | 2 | Is a deleted trace, or a half-covered one, a loud stop, or does a query about E@6 still get an answer? |
| `c05.unmutated.files` | 9 | Under withdraw A, unmutated, what does each source file, generation output and release answer? |
| `c05.unmutated.shards` | 6 | Under withdraw A, unmutated, what does each shard of run E answer, in the partitioned and in the round-robin layout? |
| `c05.unmutated.mixtures` | 7 | Under withdraw A, unmutated, what does each mixture answer? |
| `c05.unmutated.checkpoints` | 7 | Under withdraw A, unmutated, what does each checkpoint answer? |
| `c05.f.trace-deleted` | 24 | After the fault f.trace-deleted (the trace of step 3 of E/seg-1 is deleted), what does each artifact of the main store answer under withdraw A? |
| `c05.f.trace-half` | 24 | After the fault f.trace-half (the last sample of step 4 of E/seg-1 is marked half covered), what does each artifact of the main store answer under withdraw A? |
| `c05.f.edge-deleted` | 24 | After the fault f.edge-deleted (the stated edge of P in gen/P is deleted), what does each artifact of the main store answer under withdraw A? |
| `c05.f.map-deleted` | 24 | After the fault f.map-deleted (the capture map of E/shard-1 is deleted), what does each artifact of the main store answer under withdraw A? |
| `c05.record-X@A.unmutated` | 10 | Unmutated, what does each artifact that holds or read T, and gen/S, answer for the record X@A alone? |
| `c05.f.edge-file-level` | 24 | After the fault f.edge-file-level (the stated edge of T in gen/T is downgraded to file-level), what does each artifact of the main store answer under the record-level request withdraw A, record X@A? |
| `c05.f.digest` | 25 | After the fault f.digest (the declared digest of the capture map of E/shard-3 is corrupted), what does each artifact of the main store answer under withdraw A, and to the listing? |
| `c05.f.label` | 24 | After the fault f.label (the label of the capture map of E/shard-2 is set to "observed" while its derivation stays position), what does each artifact of the main store answer under withdraw A? |
| `c05.record-X@B` | 2 | For the record X@B of source B alone, what does gen/P answer, before and after a file-level link to B is added to P beside its observed link to A? |
| `c05.withdraw-C.unmutated` | 2 | Under withdraw C, unmutated, what do shard 3 and R1 answer? |
| `c05.generation-edge-faults` | 5 | Under withdraw C, after one fault on a generation edge or its harness log (bound not established, an outside read in the log, the log removed, the model entry removed with the extension off, C's passage marked in scope only), what do shard 3 and R1 answer? |
| `c05.model-entry-removed-on` | 3 | With the generating-model rule on and the model entry removed from the stored log of P's job, what do shard 3 and the generation outputs P and T answer under withdraw C? |
| `c05.withdraw-C-on` | 1 | With the generating-model rule on, under withdraw C, unmutated, what does shard 3 answer? |
| `c05.harness` | 1 | When the stand-in job tries to read a record outside its hand-off, is the read refused and does the log show the attempt? |
| `c05.unregistered-checkpoint` | 1 | What does a query naming a checkpoint that was never registered (E@12) answer? |
| `c05.listing.map-deleted` | 1 | With the map of shard 1 deleted, does the listing under withdraw A name what it covers and list shard 1 as not covered instead of leaving it out? |
| `c06.documents-mapped` | 4 | On a run made without capture, is every tokenized document mapped to its occurrence or to its set of identical candidates? |
| `c06.drops` | 2 | On a run made without capture, is every skipped line, and every line that gives no tokens, accounted for as a drop? |
| `c06.candidates` | 2 | Where a copy step was unobserved and identical texts came from different origins, are those records reported as "candidates" with their sources named? |
| `c06.downstream` | 4 | On a run made without capture, under withdraw A, what does everything downstream of the unrecorded translation answer: the file that holds T, R1, shard 3, and their counterparts in the generated sets? |
| `c06.import.file-family` | 1 | For the text-only import of T, is the file family reported? |
| `c06.import.withdraw-A` | 1 | Under withdraw A, what does the import answer as a file? |
| `c06.import.record-level` | 3 | For the single record of the import: what does a request narrowed to the record X@A answer, is its individual parent left unresolved, and is the scope that needs review named? |
| `c07.release-answers` | 3 | Under withdraw A, what do the releases R1, R2 and R1-swap answer? |
| `c07.report-actions` | 9 | Under withdraw A, what action does the rebuild report give each row of R1 and of R1-swap, and what are the counts? |
| `c07.attribution` | 3 | Which copy is the row X attributed to: in the successor of R1, in R1's history after the successor is built, and in R1-swap? |
| `c07.instances` | 3 | Are two releases with the same bytes and different histories two instances, does swapping the kept copy change the instance id, and does registering the same history again keep it? |
| `c07.ban-off` | 1 | While the content-ban extension is off, what happens to the request "ban X"? |
| `c07.R3-records` | 4 | With the generating-model rule on, what does each record G6, G9, GU and GM of R3 answer under withdraw A? |
| `c07.R3-release` | 1 | With the generating-model rule on, what does R3 answer as a whole, and which records does the report list as unresolved? |
| `c08.checkpoints` | 7 | Under withdraw A, what do the checkpoints of run E, descendant D, branch Y and run K answer? |
| `c08.step-8-holds-no-T` | 1 | Does the answer of checkpoint 9 rest on step 7 alone, and not on the half-covered sample of step 8? |
| `c08.Y-mixture` | 1 | Does Y's mixture answer "contains" while Y's checkpoint answers "does not contain"? |
| `c08.K.digests-match` | 1 | Does the shuffle computed for run K match the digests in run K's run record, for every launch, epoch and rank? |
| `c08.shared-token` | 3 | For the record W alone, the document read only through a sample's shared last token, what do checkpoint 6, checkpoint 9 and descendant D answer? |
| `c09.repair-comparison` | 4 | On the 10k set, does the repaired side equal the reference built from sources that never held A (record multiset, per-shard digests in seeded mode), and does each of two deliberately wrong reports (one descendant of A kept; one paraphrase of an unrelated record retired) make the sides differ? |
| `c09.record-actions` | 3 | Are the report's action counts and its actions by kind of record the expected ones (X re-source, B's copy promote, P, T and S retire), on the 10k set and for the rows of the fixture's R1? |
| `c09.artifact-actions` | 8 | Which artifacts does the rebuild keep and which does it rewrite: R1, R2 and the three shards of the fixture, the kept shards of the 10k set in the partitioned layout, all 20 chunks in the round-robin layout? |
| `c09.mixture-successors` | 1 | On the fixture, which mixtures need a successor under withdraw A, and which are reused as they are? |
| `c09.successors` | 3 | Under withdraw A, what do the successor artifacts answer: the successor shards of the 10k set, the successor of shard 3 and the successor of R1? |
| `c09.history` | 2 | After the repair, does the historical R1 still answer "contains" for A, on the 10k set and on the fixture? |
| `c09.exposure-free-restart` | 1 | On run E after withdraw A, which checkpoint is the exposure-free restart? |
| `c09.exact-rerun-restart` | 1 | On run E, does the exact-rerun restart equal the first step at which the corrected run's order differs from the historical one? |
| `c09.stale-cache` | 2 | Are the cache files of E/seg-2 detected as stale and rebuilt, and those of E/seg-1 left alone? |

70 groups cover 379 listed rows (370 core rows and the nine rows of the generating-model extension); with six subjects each, `check10.toml` has 420 rows.

### A row of `check10.toml`

One row is one subject asked the questions of one group. Its id is
`c10.<group>.<subject>`. It has the keys of section 1, and:

| Key | Meaning |
|---|---|
| `given` | `subject`, `group`, and `conditions` (the table above) |
| `record` | which of the subject's own records answers, or `none` |
| `missing` | when `record` is `none`: the record or field the subject lacks. It is the evidence of the answer "cannot answer" |
| `route` | how the record is reached: which run, which call, which lookup |
| `rule` | how the record becomes a reading |
| `fault` | for a question of Check 4 or Check 5 that injects a fault: how the fault is put to the subject's own record |
| `expect` | `one_of`: the grades the rule can lead to. `right`, `wrong`, `cannot_answer`: what earns each. It holds no expected grade |
| `marks` | the marks every recorded answer of the row carries (below) |
| `excluded_from_tally` | true where the plan keeps the row out of any tally or comparison of the design against an existing tool |
| `repeats` | only on a row left out of the tally as a repeat of a base question for that subject: the id of its counterpart row of Checks 1 to 9 (see "Marks and the tally") |
| `needs_ordinal`, `ordinal` | for a Dolma row that needs an ordinal: the two readings (below) |
| `tie` | for an OriginBlame row that reads an entry by its place: the tie and when it is valid (below) |
| `source` | where the expected answers come from. It starts with the categories of section 1 that the rows of the group name |

A row whose `record` is `none` can only be graded cannot answer, and only after
the subject has run. If a run shows that the subject does hold such a record, the
row is not regraded on the spot: a reading rule for that record is written first.

### The Dolma tie to an ordinal: two readings from the start

The Dolma toolkit's tokenizer does not keep the order of its input, and its row
order changes from run to run (report section 2.2). A companion row is therefore
tied to a document of the replica's shard by its `src` and `loc`, never by its
place in the companion file. This replaces the tie "by its place in the input
order" of the plan as it was signed.

The tie rests on one fact that is not yet shown: that on the replica's chunk files,
under the replica's settings (one preprocessing partition, or
`--keep-sequential-samples`), the document with ordinal n of a shard, counted from
0, is line n + 1 of its chunk file, counted from 1. In the plan's words: document
i is line i.

Every row with `needs_ordinal = true` therefore carries both readings, under
`ordinal`, with the selector `document i is line i`:

| Selector | Reading | What it says |
|---|---|---|
| shown | `tie by src and loc` | the document with ordinal n is tied to the companion row whose `src` is the shard's chunk file and whose `loc` is n + 1 |
| not shown | `cannot answer` | the question is recorded as cannot answer, with the missing tie as the evidence |

The result that selects between them is recorded per chunk file, once, with the
run that shows it, before any Dolma row is read. It is shown for a chunk file when,
for every document of its shard, the shard's tokens at ordinal n equal the
tokenization of line n + 1. Where no result is recorded the reading is "not
shown". A question whose reading touches a chunk file for which it is not shown
takes the reading "cannot answer". So the rule is never picked after an answer is
seen.

Every answer that used the tie carries the mark `ordinal derived from src and loc
(owner's ruling after the debate)`, so that a comparison shows the departure from
the signed plan.

A row needs an ordinal when it reads one document of a shard: a combination's
checkpoint questions, and the per-document questions of Checks 3 and 6. A question
about a shard as a whole does not: its rows are selected by `src` alone.

### The OriginBlame tie by token range

A token-index entry carries no id and no offset (report section 3.1). The token
range of a document is the running sum of the entries' token counts. It is valid
only when one process tracked the documents in packed order: with two writer
processes the merged order is wrong (report section 3.2). A row that reads an entry
by its place carries `tie`, with `valid_when` and `otherwise`: where the run does
not show one process and packed order, the question is recorded as cannot answer,
and the missing field is an id or an offset on the entry.

### Marks and the tally

| Mark or flag | Where |
|---|---|
| `tool's own output; not an independent baseline` | every row of `trace` for a question of Check 1 or Check 2. These rows count toward the completeness of Check 10 for the trace |
| `step-to-ordinal part is the tool's own output` | every row of a combination. A comparison of the design against an existing tool shows the mark wherever it uses such a row |
| `excluded_from_tally = true` | the rows of `trace` and of both combinations for a question of Check 1 or Check 2, and a row of one subject that repeats a base question for that subject. A repeating row names its counterpart in `repeats`, and is left out only where, for that subject, the store and evidence, the selector, the artifact, the switch behaviour and the expected observable are the same as the counterpart's. A request that only Waterwhite can reduce to the counterpart's (a flagged file to a withdrawal and a record) never repeats: it stays in that subject's counts. A row that adds a warning, changes the failing job or grades a new request block is not a repeat. Every such row is recorded and shown, and is left out only of a tally or a comparison. Checks 1 and 2 still pass or fail on their own expected-answer sources |
| `ordinal derived from src and loc (owner's ruling after the debate)` | every recorded answer that used the Dolma tie (above) |

A tally gives each subject two counts, side by side and never merged: how many questions it could answer (every recorded answer that is not "cannot answer"; a failed run counts as no answer), and how many of the questions it answered are right. A question that `excluded_from_tally` leaves out is left out of both counts.

### The faults of Checks 4 and 5

The unmutated questions of Check 5 are put as they stand. Each injected fault of
Checks 4 and 5 is put to the subject's own record where the subject has the
corresponding record; the row's `fault` says how. Where it has none, the row's
`record` is `none`, `missing` names the record the fault would have changed, and
the questions of that fault are recorded as cannot answer.

| Fault | Put to |
|---|---|
| `f.trace-deleted`, `f.trace-half`, `l.trace-count` | the trace's rows of that step. Only the combinations read a source through them |
| `f.edge-deleted`, `l.edge-parent` | the manifest's `built_from` entry of `gen/P`, the file-level counterpart of the edge. No other subject has a derived-from link |
| `f.map-deleted` | the manifest's row of the shard; the Dolma toolkit's companion rows of the shard's chunk file; the token-index entries of the shard's OriginBlame store |
| `f.digest` | the Dolma toolkit's companion file and OriginBlame's token-index file, changed so that they no longer agree with the shard. Neither tool stores a digest of them |
| `l.count-499` | one companion row, or one token-index entry, deleted. Neither tool stores a count |
| `w.kill-after-100`, `w.retry` | the tool's own writer, stopped by its process id part way and run again |
| `l.artifact-digest` | the file itself, as the fault stands; only the manifest holds a row for a release |
| `w.declare-499`, `f.edge-file-level`, `f.label`, the faults on a generation edge or its log, the harness | no subject has the corresponding record |

### A failed attempt, the two routes after it, and when Check 10 is complete

- A failed attempt to run an existing tool is not an answer, alone or in a
  combination. It is recorded with its command and its failure output, never as
  "wrong" and never as "cannot answer", and it leaves Check 10 open.
- Two routes follow a failed attempt. First, the replica writes the tool's input
  in the tool's documented format from the same records. A file written this way
  keeps one line per line of the chunk file, in the same order, under a name that
  maps to the chunk file one to one, so that `src` and `loc` still name the
  chunk-file line. Second, a run on another machine, with the outputs committed.
- "Cannot answer" is for a tool that ran and whose own records lack what the
  question needs. A tool that did not run cannot earn it. On a store with no
  chunk file and no shard there is nothing for a tool to run on: a tool that ran
  in the same pass answers "cannot answer" there, citing that run, and a tool
  whose run failed reads "failed attempt". On every store a tool can read, the
  tool must run there, as before.
- Check 10 is complete when every question has a recorded answer (right, wrong, or
  cannot answer with its evidence) for every one of the six subjects: every listed
  row of every group, six times, except the rows of an extension that is off, which are
  not asked (plan :305). A question never
  attempted leaves it open. An
  existing tool that has not run by any of these routes leaves Check 10 open and
  the first build not done. There is no bar on what the answers are.
- A substitute existing tool, if the owner names one, gets its reading rules
  written into this table before it runs.

### Readings chosen here

The plan leaves these open. Each is one reading, used throughout the table. The
other reading is named beside it, so that a ruling can change it.

1. A subject is read through its own records only. A mixture folder's listing is
   not a record of the Dolma toolkit or of OriginBlame, so neither answers a
   question about a mixture. The other reading: a mixture's shards are taken from
   its folder.
2. The Dolma tokenizer reads the id field of the chunk files, and an id is resolved
   by a lookup in the replica's files. The other reading: the id field is switched
   off, as for a text-only file; no row then names a record, and every question
   about a source is cannot answer.
3. A document that resolves to a generation output makes a Dolma reading cannot
   answer, naming the missing derived-from link, unless another document already
   names the source. OriginBlame's forget set is the tool's own answer: a bit that
   is not set is read as the tool reads it. The other reading: treat both alike.
4. OriginBlame's caller passes no lineage of its own. The other reading: the caller
   passes a generated record's sources through, which plan Section 13 calls a
   reading of the tool's interface that its documentation does not show. A run
   made that way is another route and is recorded beside the answer with its
   command.
5. The manifest holds no digest and is written while a build runs, so a build made
   without capture has none. The other reading: the manifest is written from the
   build's configuration and exists for every build.
6. Check 9's questions about a repaired side are put only to a subject that has a
   rebuild report. No repaired side is built from OriginBlame's forget set. The
   other reading: its marked documents are left out and the result is compared
   with the reference.
7. A baseline is graded on the answer and on the values a question asks for, not
   on the words this tool uses to say why (the list under "Grades, and what is
   compared"). The plan names three notes that are not counted; the rest of that
   list is this table's reading. The other reading: every field of an expected row
   is compared, and a subject that cannot name a cause or show a chain is never
   right.
8. A combination reads the trace's rows and does not turn the trace's own mark of
   an untraced step into an answer. The other reading: a combination that sees an
   untraced step in a checkpoint's history gives no "does not contain"; it would
   then carry part of this tool's proof and say less about the existing tool.
