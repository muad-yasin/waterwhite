# The rebuild

What the tool does when a source is withdrawn and the artifacts downstream of it
have to be made again: the report it prints, what a repaired release must
reproduce, the two restart points of a training run, and how a repair is
verified. Plan sections served: 4.1 (the report), 4.2 (what a repaired release
reproduces), 4.3 (restart points), 4.4 (verification of a repair); the forms are in
`docs/interfaces.md` sections 7.1 to 7.4 and the commands in `docs/cli.md`
(`report`, `verify`).

## 1. What the rebuild is, and what the tool does not do

A withdrawal is a question (`query withdraw`) and then a piece of work. The question
is answered from the evidence store: which artifacts have the source behind them,
which are shown clean, which cannot be told. The work is the team's: rebuilding
releases, re-tokenizing shards, building new mixtures, training again. The tool does
three things around that work and executes none of it.

1. It prints a **report** (`report`): for every release, import, shard and mixture
   the one action to take, for every record of an artifact that is rewritten its own
   action and the reason, and for every run the restart points.
2. It is handed what the team's builder made, and it **verifies** it (`verify`): that
   the repaired side holds what a build without the withdrawn content holds, that
   what the report keeps is untouched, and what the new artifacts answer.
3. It answers questions about the new artifacts like any other registered artifact.
   A repaired file that is registered with capture on carries its edges and maps, and
   a successor is only an artifact with a name of the form `successor of <name>`.

The builder is the team's own. In this repository the replica stands in for it
(`python -m replica repair`, `docs/replica.md`); the tool never imports it and the
replica never imports the tool's rebuild code, which is why a report is a file.

The tool does not resolve an unresolved record (below), does not regenerate a retired
one (the team runs the generation job again), does not rewrite a kept artifact, and
does not open any path of an extension that is switched off: the generating model of
a record is not a link of the proof while that extension is off, and a successor is
never judged through it.

## 2. The report

The report is computed on demand from the store as it is, in one coverage session,
and nothing is written. A coverage session is valid for one state of the store; if
the store is written while the report is made, the command ends with "store written
during the query" and prints no report. The tool holds no report: a stored report
would be a derived fact that every write to the store would have to invalidate,
and `verify` is handed the report as a file, which is the document the repair was
driven by.

### 2.1 The action of a record

The verdict the proof of plan 2.2 gives a record for the request decides its action.

- **keep** only for the verdict "does not contain". "No chain to the source was
  found" is not enough: a record whose chain is broken is not shown clean.
- **unresolved** for the verdict "may contain", and for a record whose only evidence
  is a file-level link: the file derives from the source, no record of it is tied to a
  record of the source, and the link cannot say which of them the source reaches (plan
  3.2, L1). It has no action. The report names the missing link, the evidence it rests
  on and the scope a person has to review, which for a file-level link is every record
  of the file. The file-level evidence is a hop anywhere in the record's chain, not
  only the last one: an edge labelled file-level, a file-level link to a member, or a
  link to a member by a publisher's pointer. `verify` asks the report's own check for
  this word, and for a document of a shard also reads the document's own verdict, so
  it is never less strict than the report: a successor is never shown as holding no
  unresolved record while the report would count one of its records unresolved. The
  two do not always agree: on a store whose map-row label was edited outside the
  write path, `verify` counts a document unresolved that the report retires or
  keeps. A
  successor that holds an unresolved record answers "may contain" for the source until
  the team resolves it, and nothing in the tool resolves one.
- **retire** for a record that is a descendant of the source (plan 4.1, Decision 10):
  it leaves the successor when any parent of an "all" edge is in scope, or every
  parent of an "any" edge is. A summary of A and C with A withdrawn alone is retired.
  A release row that is the withdrawn source's own copy, through a single-parent copy
  edge with no equivalent copy to take its place, is retired as well. The report
  names the generation job that made the record and that job's step, name, version
  and parameter digest, so that the team can run it again; regenerating is the team's
  follow-on step. A copy of the source's own record has no generation edge and names
  none.
- **re-source** and **promote** are the two entries of one event (plan 4.1, Decision
  15). A release row made by an "any" edge whose selected parent is withdrawn, while
  an equivalent parent remains, keeps its place and takes the equivalent parent: that
  is the re-source, and the report gives both the parent it takes and the parent it
  leaves. The record that dedup dropped in favour of the withdrawn one, the dedup
  loser, returns to the release: that is the promote. The history keeps its original
  attribution.

A report says one action per record, and lists a record only for an artifact that is
rewritten or unresolved; an artifact that is kept has every record kept.

### 2.2 The action of an artifact

An artifact is judged by its own answer to the request: "does not contain" is keep,
"contains" is rewrite, and "may contain" or "not covered" is unresolved; an artifact
that "contains" only through file-level evidence, so that every record in scope is
unresolved, is unresolved too. A shard is kept when none of its documents is in scope
and it receives no promoted or re-sourced record that it does not already hold. A mixture
needs a successor exactly when its snapshot names a shard that is rewritten, and the report
prints the lists of the mixtures beside the counts: the mixtures that need a successor, the
mixtures that are kept, the unresolved ones and the ones that read the fourth word (below).
The artifact that holds an unresolved record but none shown in scope is not kept; it is
unresolved. Run segments and checkpoints have no action, only restart points.

**A record that returns.** A record the report promotes, or re-sources a row onto (its
`new_parent`), is a returning copy. Whether a shard that nothing of the source stands behind
can be changed by it depends on where the layout puts it, and the store records no layout.
So "keep" keeps its meaning, the shard stays byte-identical, and the report says it only
where it can:

- **Already held.** A shard holds a returning copy when a document of the shard that the
  report keeps as it is has a map parent from which the copy is reached by following only
  single-parent, record-level edges that are not generation edges: the copy itself, its
  parent, its parent's parent. No "any" edge is followed and no equivalent copy is counted;
  a row that names a set of candidates, a shard with no usable map and a document the report
  changes hold nothing. The count is made over one group at a time: the shards of one named
  layout, and the shards no layout names together (with no layout named, all shards are one
  group). In the fixture, shard 1 holds B's copy of X: a document of it reaches the copy in
  two such steps.
- **Placement stated.** Where the caller states the layout's placement (`report --placement`,
  docs/cli.md), a returning copy lands in the shard the placement names for its source, and
  only that shard can hold it. A shard that is named and does not hold the copy is rewritten
  and `receives` names it; a shard that is already rewritten takes the copy into its
  `receives`; a shard the store does not hold is named in `groups.placements[NAME].landing`,
  with no artifact entry. Where the placement states a shard's own re-sourced documents, it
  decides where they land too. The documents of the successor are the documents, less the
  ones that leave, plus the ones it receives.
- **Place not stated.** Where a copy has no stated place (no placement for the layout, or the
  placement names none for it) and no shard of the group holds it, every shard of the group
  whose own answer is the negative reads **"clean, place not stated"**: its own contents are
  clean, the copies it may receive are named (`returning`), and their place is not stated.
  The word is not "keep" and does not begin with it. A shard that is already rewritten or
  unresolved keeps its action. A mixture reads the word when a shard of its snapshot does and
  none is rewritten; a rewritten shard makes it rewrite, and an unresolved one leaves the
  rule above as it was. Such a mixture is not a mixture that needs no successor, so a run
  over it is never told that no exact rerun is needed: the report gives the phase start of the
  earliest segment over it, as an assumption.

### 2.2a The files upstream of the artifacts

Every affected file is in the report, and not every one has an action of section 2.2.
A registered JSONL file that has no entry in `artifacts` (a source file, a generation
output, an intermediate file ahead of a release or a shard) and whose answer is not
"does not contain" is listed in `upstream_files`, which follows `artifacts`, in the
order of registration, with its answer, an action where there is one, and the reason
(plan 4.1, BF-6):

- a revision named by the request takes no action, because the repair never
  edits a source revision; its reason says the request withdraws it, or, for a request
  narrowed to records, how many of its records are withdrawn and that the rest stay;
- a generation output takes no action: the repaired build reuses it as the same
  instance, and the records of it that the request reaches leave the release that
  copies them, so the output itself is left as it is. The store tells a generation
  output from a rebuilt intermediate file by the generation edges of its records;
- an intermediate file ahead of a release or a shard whose answer is "contains" is
  rewritten;
- a file whose answer is "may contain" is unresolved and carries the named cause and
  the failing link of the answer.

The list is kept out of `counts` and `bytes_to_rewrite`, and neither the replica nor
`verify` reads it: it is for the person who reads the report.

### 2.3 Counts

The report counts, per artifact and in total, the records of each action, the
records removed (the rows that leave the successor), the shards and the mixtures
kept and rewritten, and the bytes to rewrite. The size of a file is read from the
disk when the report is made; the store holds no size, and the report says which
files were not where they were registered. Shards are counted per run, by the runs
whose segments' mixtures hold them, and per layout where the caller names the
layouts: the store records no layout, so the tool cannot find one by itself.
A report made with no layout named carries `groups.layouts`, which tells the reader that
the count was not made and names `--layout`; with layouts named, the same key lists every
shard that no named prefix matches, so that no shard is silently missing from the
per-layout count (plan 4.1, BF-7).

The rows of the successor of a release are its kept, re-sourced and unresolved rows;
a promoted record serves the re-sourced row and adds none.

Under a layout partitioned by source a whole shard can be kept; under a layout that
deals records round-robin over chunks, a source whose records reach every chunk
rewrites every shard of the phase (a file with fewer records than chunks does not reach
every chunk). The report shows what the layout costs: the counts
are the point of it (plan 4.5).

## 3. What a repaired release reproduces

The retained records and their multiplicities equal a from-scratch run of the same
pipeline without the withdrawn content (plan 4.2, Decision 6). The successor shrinks:
removed tokens are not refilled. Bytes match only where every step of the build is
seeded; with an unseeded shuffle equality is of the record multiset over the whole
release. The equality is claimed only for a report with no unresolved record: with
one, the verification compares and lists differences but does not say the repair is
equal or different.

## 4. Restart points

Two restart points are given for a run (plan 4.3, Decision 23).

**The exposure-free restart** is the last checkpoint of the run whose consumed
history is completely traced, passes the proof of plan 2.2 and holds no exposure. The
report names it and states its basis. Where none qualifies, because the exposure is in
the first segment or an earlier step is untraced, the report does not name a
checkpoint. It says "no exposure-free restart is proven" and names the first step of
the lineage as the restart that is free of exposure by construction: a restart from
scratch on the corrected mixture. For a descendant run the lineage is the run it
descends from. The word is "proven" on purpose: the tool can show a restart clean
only from the traces it holds, and it does not call a restart clean that it cannot
show. The report also lists the checkpoints that carry exposure, with the answer and
the first exposure of each, and no count of reads (exposure counts belong to an
extension that is off). It also lists, in `checkpoints_not_covered`, every registered
checkpoint whose answer is "not covered", with the answer's reason, whether or not a
run reaches it: a checkpoint registered over a segment the store holds no row for is in
no run, and is named there and nowhere else. A registered checkpoint is therefore in
one of three places: negative and in no list, in `exposed_checkpoints`, or in
`checkpoints_not_covered`.

**The exact-rerun restart** is the first step at which the corrected mixture's data
order differs from the historical one, found by building the corrected order and
comparing the two. It needs the document lengths of the successor shards, which exist
only after the team's builder has made them from the report, so the step is stated by
`verify` and not computed by the report (plan 4.3, BF-5). The report states, for every
run that has a stale segment (one whose mixture needs a successor), which case the run is
in, by one check of its run records: a run is replayable when the record holds every
setting a replay needs (the sequence length, the random seed, the split, the
add-extra-token setting, and the train iterations or samples with the global batch
size). This is a property of the record and has nothing to do with `verify --mode`,
which is the build mode of plan 4.2 (the byte compare of shards).

- For a replayable run the report says, in the words of `RESTART_EXACT_BY_VERIFY`, that
  the restart is left to the verification, and names no step.
- For a run that is not replayable, the report itself gives the phase start, the first
  step of the earliest stale segment of the run, as an assumption, and says that no step
  comparison is possible (F7). `verify` states the same, without trying the replay.

Where a run has a stale segment and no registered checkpoint, the report still has an
entry for it. The two restarts are two keys of one entry (docs/interfaces.md 7.1).

## 5. Verification of a repair

The repaired side and the reference side are produced separately (plan 4.4). The
reference side is built by the same builder from source revisions that never held the
withdrawn content. The sides share the builder's pipeline steps and not the report:
the report decides which records leave, the source revisions decide what the clean
sources are, so a report that keeps a descendant, or drops a record that is not
affected, makes the two sides differ. That is what the comparison is for.

`verify` is handed the report and a sides file that names each side's file for each
artifact, and it does these:

- refuses a report for a store that does not hold an artifact the report names by its
  instance, and a report whose request the store no longer expands to the same
  members and records; the store a repair is verified against is the store the report
  was made for, or a faithful copy of it with the successors registered;
- compares, for every JSONL file named on both sides, the multiset of (fingerprint,
  multiplicity) of the lines, and, in seeded mode, the digests of every shard; a file
  that the report keeps or rewrites and that the sides do not name on both sides is
  named in the verification, and the comparison it belongs to is then "not compared"
  and never "equal"; the one exception is a rewritten file or shard that neither side
  names and that the report's own counts show is left no record (no row in the
  successor of a file, no document in that of a shard): nothing is there to compare, so
  it does not hold the comparison back, and the verification names it with the reason;
  a shard the report names only under a layout's landing is compared like a rewritten
  shard where the sides name it on both sides, and where both builds wrote it and the
  sides do not name it on both sides it is named and keeps the comparison from "equal";
- checks that every artifact the report keeps is byte-identical afterwards, against the
  content digest of the instance the report names and of nothing else (a store that
  does not hold that instance gives null and says why);
- puts the successors to the proof: a successor shard that passes the proof of plan
  2.2 and holds no unresolved record answers "does not contain" for the withdrawn
  source; one that holds an unresolved record answers "may contain", naming the link
  that failed; one with a missing map answers "not covered", and what is downstream
  of it "may contain";
- confirms that the historical release still answers "contains": history keeps its
  attribution, and the question that must find nothing is put to the successors only;
- detects stale index caches and states the corrected order's restart.

A successor that was rebuilt without capture, whose records include one that cannot
be told from a candidate of another source, answers "may contain" for the source
(plan 4.4). No check is made on such a successor in this repository: the sides are
built with capture on.

### Stale caches

Megatron's cache key holds the paths of the files and not their content, so a trainer
that finds index caches under the old paths uses them even when the shards behind
them have changed. The tool detects it with the **mixture digest**: the digest of the
content digests of the shards of a mixture's snapshot, in the trainer's order. It
covers what the index caches were built from, and nothing of the folder the shard
files were read from, so a mixture whose shards are unchanged has the same digest
wherever it was built. A segment's caches are stale when the digest of the mixture
the segment was registered with differs from the digest of the corrected mixture: the
mixture itself where the report keeps it, which cannot differ, and the registered
mixture called `successor of <mixture>` where the report rewrites it. A stale segment's corrected order
is built from an empty cache folder: the index arrays are derived again from the run
record and the document lengths of the corrected shards, and the kept cache files are
not read. A segment whose mixture is untouched is left alone.

## 6. Words and names

- The actions are "keep", "rewrite", "re-source", "promote" and "retire", and
  "unresolved" for what has no action (`vocabulary.ACTIONS`, `UNRESOLVED`). There is
  no action "remove": a release row that copies the source's own record is retired.
  "Records removed" is a count of rows.
- A successor is named `successor of <artifact>`. A repaired build that is
  registered with this prefix gives a file it left as it was the same instance as
  the historical one, under a second name; a rewritten file is a new instance. The
  instance id of a historical artifact never changes.
- The named cause "unresolved record" is a word of the vocabulary that no answer
  carries in this build. The answer of an artifact that holds an unresolved record
  names the failing link of the proof, as every "may contain" does.
- The two sides of a repair, the report and the verification are JSON documents with
  a format word in `form`.

## 7. What is not built

- The parts of the report and the verification that depend on exposure counts, on a content ban or on published models. With the generating-model switch on, the report lists a record whose generating model fails condition (h) as unresolved, names the model or the missing entry as the link, and no successor is clean while one remains; a record entry that rests on a temporal ancestry cycle reads "unresolved" and an artifact entry that rests on one stops the report with the named error.
- Resolving an unresolved record. The report lists it with the missing link and what
  to review.
- Registering a snapshot for a run made before the tool was installed (plan 5.8,
  B-4).
- A restart for a run that has no registered checkpoint: it has no entry in the
  report.

## 8. The incident, walked through

Plan 4.5 and 4.6 walk a team through a withdrawal on a layout like the one of a
published pipeline. They are a walk-through and not a build; this section says, for
each sentence, what the tool backs and by which row or check, and "no check" where
nothing does. A row is a row of the hand-written table (`fixture/table/`), a test is
a file of `tests/rebuild/`.

**What the tool backs.**

- "Every shard of the phase is rewritten where the withdrawn source's records reach
  every chunk" (plan 4.5): the report counts shards kept and rewritten per layout, and
  the layout is the caller's to name (`report withdraw ... --layout NAME=PREFIX`).
  Backed by the row `c09.10k.round-robin.all-rewrite` (all 20 chunks rewrite, none
  kept; it is a row of the 10k set, so it runs only on request), and by the test of
  the groups in `tests/rebuild/test_report_artifact_actions.py`.
- "A layout partitioned by source leaves whole shards reusable" (plan 4.5): backed by
  `c09.10k.partitioned.kept-shards` (the kept shards are exactly those no line of
  which is the source's or derived from it, and they are byte-identical afterwards)
  and, on the fixture, by `c09.fixture.shards.counts` (two kept, one rewritten).
- "Which records the successor keeps, re-sources or promotes" (plan 4.5): the actions
  of section 2.1, backed by the rows `c07.report.R1.*`, `c07.swap.report.*` and
  `c09.fixture.R1.counts`.
- "A generation edge with no established bound denies 'shown not to be affected'"
  (plan 4.5, 2.2(g)): backed by the Check 5 rows of the bound
  (`c05.g.bound-field.*`, `c05.g.log-removed.*`, `c05.g.log-outside-read.*`). The
  report reads the same answers: a record whose verdict is "may contain" is
  unresolved and never kept, and an artifact that holds one is not kept either
  (sections 2.1 and 2.2; the unit tests cover a missing origin, not a missing bound).
- "Which GPQA item each record came from cannot be known from text-only rows, so
  per-item answers stay 'may contain'" (plan 4.6, L1): backed by
  `c06.import.parent-unresolved` for the answer and by `c06.import.review-scope` for
  the report, which lists the record as unresolved with the whole file as the scope
  to review (no unit test builds a file-level case; the row is the only check).
- "Checkpoint answers depend on whether the run's caches were kept, or can be replayed
  and checked against digests recorded at training time" (plan 4.6, 3.5): backed by
  the Check 2 rows (`c02.i.*`, `c02.ii.*`, `c02.iii.*`), which belong to the
  registration of runs and not to the report; the report only reads the answers.
- "Every mixture whose snapshot holds a shard that received one of them contains":
  the rule is the mixture rule of the proof (a mixture contains where one of its
  shards does), and the report prints which mixtures need a successor
  (`c09.fixture.mixtures`, and the test of the mixtures in
  `tests/rebuild/test_report_artifact_actions.py`).

**What no check backs.**

- The answer a checkpoint gets on a round-robin layout (plan 4.5, the bracketed
  note): the plan tabulates none, so which checkpoints are shown not exposed on that
  layout stays a design statement. Only the number of shards rewritten has a row.
- Whether a team may register a snapshot for a run made before the tool was installed
  (plan 4.6, 5.8 B-4): the plan does not settle it, the tool promises nothing for
  such a run, and no operation of the tool registers a snapshot after the fact.
- How many chunks the real layout has, how large the withdrawn folders are and how the
  split treats each file (plan 4.5): the case the incident falls in is open. The tool
  gives the counts for a layout the team names and a fixture that stands in for one.
