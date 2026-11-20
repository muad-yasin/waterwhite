# The baselines run (plan 5.5 Check 10)

Check 10 puts the questions of the table to six subjects: the manifest, the trace, the Dolma
toolkit, OriginBlame, and the two combinations of a tool with the trace. This document says how
a run is started, what it checks before a tool starts, and what a run leaves behind. The table
is `fixture/table/check10.toml` and its README, section "Check 10"; the modules are in
`baselines/`.

## The command

    python -m baselines --subjects dolma originblame --step-cap 900 \
        --work WORK --run-record RECORD --lock-prefix COMMAND... --world WORLD.json \
        --program-sha256 PROGRAM_SHA256 --lock-wait-bound 7200 \
        --tool dolma ENV_DIR ENV_RECORD IDENTITY_SHA256 \
        --tool originblame ENV_DIR ENV_RECORD IDENTITY_SHA256 PROGRAM

- `--subjects` names the subjects to run (default: all six). The runner drives `dolma` and
  `originblame`; it refuses another subject.
- `--step-cap` is the cap of every tool step in seconds, a number above zero, and it is
  required: a step has no run without one. A step that reaches it is stopped by the id of its
  process and is a failed attempt that carries its seconds. The cap is an input, so it is set
  from measured seconds and not guessed in the code.
- `--work` is the work folder (the one place a tool may write), `--run-record` the new folder
  the run leaves (a folder that exists is refused), `--lock-prefix` the command every step is
  started behind, `--commit` the commit the run is on (default: the head of the repository).
- `--tool NAME ENV_DIR ENV_RECORD IDENTITY_SHA256 [PROGRAM]`: a tool's copied environment, the
  record the copy's setup left (`compare.json` and `identity.json`), the sha256 of that
  `identity.json` as the caller read it before the run, and for OriginBlame the program.
- `--program-sha256` is the sha256 the program of OriginBlame must have, given on the command
  line for this run: 64 hexadecimal digits. It is needed for a run of `originblame`. It is never
  read from `identity.json` and the code holds no such value: the runner hashes the program it is
  handed and stops, before any tool starts, when that is not the supplied value (stage `binary`).
- `--lock-wait-bound` is the number of seconds a step may wait for the lock behind the lock
  prefix (see "The lock prefix"). Without it no note is made and the prefix is a plain command.
- `--world` names the file of builds described below.
- `--extensions NAME...` names the extensions whose rows are asked (default: none). A row of an
  extension is put only while its extension is on (the table's README: "An extension row is put only
  while its extension is on", plan :305); a row of an extension that is off is not asked, is not a
  line and is not a none line, and is not counted. The names are written to `extensions.txt` in the
  run record beside `start.txt`; a re-grade reads that file (absent means none). A pass with an
  extension on is refused on a world that was not built with it (see "The world file": `extensions`):
  `main` exits with the message, as for the other wrong arguments, before the run record or any tool step
  exists (it is not a pre-run stop, and not exit 3).

The first line printed is the start line: the commit, the subjects, the step cap, the sha256
of each `identity.json`, where they were given the supplied program sha256 and the wait
bound, and the sha256 of the `WORLD.sha256` beside the world file (the word `none` where the
folder holds none). It is also kept in the run record as `start.txt`.

Exit status: 0 for a finished run, 3 for a stop before any tool command (see below), 4 for a stop
because the wait for the lock ran out before a step.

## The lock prefix

With `--lock-wait-bound`, every step of the run (the probe of the copy's python included) starts
behind the lock prefix, which takes the lock for that one step and holds it only for that step.
The runner makes a note file for the step in a folder of its own in the run record,
`locks/<step id>/note.txt`, and sets two environment values on the prefix's process:
`WATERWHITE_LOCK_NOTE` (that file) and `WATERWHITE_LOCK_WAIT_BOUND` (the bound, in seconds). The
prefix writes one line, `acquired <seconds waited>`, to the note once it holds the lock (the seconds as
digits with a decimal point, as `0.123`: the runner does not read a comma), and then
starts the command. If the bound runs out first it starts nothing and exits with status 75.
The runner hands the bound as a float (`7200.0`). A prefix that gives it to `flock -w` must make
sure `flock` can read it: in a locale whose decimal separator is a comma, as `de_DE`, `flock -w
7200.0` stops with "invalid timeout value", so such a prefix pins the numeric locale (`LC_ALL=C`)
or rounds the bound to whole seconds.

- The cap of a step counts from the acquire line and never before it: a wait longer than the cap
  is not a cap hit, and the seconds of a step do not hold the wait.
- The wait is a field of the step of its own, `lock_wait_seconds`, in `steps.jsonl`; the raw
  output holds only the tool's text. A step with no lock prefix has `null` there.
- That the bound ran out is read from the missing acquire line in the note, never from the exit
  status alone (a command may exit with 75 itself). The run then stops before that step: no step
  is kept for it, no line is written for its question, and the stop names the note. The lines and
  failed attempts made so far are written and sealed.
- A step stopped at the cap, and a writer stopped by the run, are ended by the id of their process
  group, which holds the prefix, the lock command, the sandbox and the tool, and the lock is free
  afterwards.

## Before a tool starts

Nothing is started until these have passed, in this order; a failure stops the whole run (exit
status 3), writes no answer line and names its stage.

0. The world folder is the one the world command wrote (stage `world-hash`): the `WORLD.sha256`
   beside the world file names no file that is changed (its sha256 is not the listed one) or
   missing, lists no path twice, has a row for `world.json` itself and for every path
   `world.json` names (the paths of the world command's own listing function), and a folder
   with no `WORLD.sha256` is refused, because the world command always writes it. The stop
   names the files by their paths in the folder. Every pre-run stop of every stage leaves
   `stopped.json` in the run record (the stop, no driver, no question), and the sealed record lists it.
1. Every faulted OriginBlame row has a putter (stage `putter`), and every request file handed to
   OriginBlame is a record file of the build of each store the request is put to, read from the same list the store
   registers from (stage `request-files`). A row is never dropped for a miss.
   A request word is put to the build of each
   OriginBlame question whose row carries it (the `request` of the row's `given`), the store or
   set the question is asked on; a word no question carries is checked against no build, and a
   question whose build the world does not name for OriginBlame makes its word count for every
   build.
1b. The work folder, when the pass runs the Dolma tool with the real starter, has a path of at most
   60 characters (stage `work-path`): the toolkit's tokenizer binds a unix socket at
   `<work>/dolma-tmp-N/pymp-XXXXXXXX/listener-XXXXXXXX`, a unix socket path has at most 107 characters, and
   a longer work path stops the tokenizer with `AF_UNIX path too long` (the first real pass lost three
   steps to it). The number 60 is computed from those names, and a test starts a real multiprocessing
   manager at the longest path and at one character more.
2. For each tool: `compare.json` in the setup's record says `equal` is true (stage `compare`);
   `identity.json` is there and its own sha256 is the one handed in (stage `identity`); the copy's
   python alone, started inside the sandbox through the built route, writes the path it resolves
   to and the sha256 of that binary into a file of the work folder, and that sha256 is the
   `interpreter_sha256` of `identity.json` (stage `probe`); for OriginBlame the program's sha256 is
   the sha256 supplied with `--program-sha256` and also the `ob_binary_sha256` of `identity.json`
   (stage `binary`). Only then does the gate of the tool open. The runner reads `identity.json`
   and never writes it, and its code holds no hash value.

`identity.json` is written by the setup, before the run: `python -m baselines.envcheck ENV_DIR
--original ORIGINAL --record RECORD --program FILE` hashes the program file and adds the one key
`ob_binary_sha256` to the record's `identity.json` (the setup script `tools/setup_baselines_env.sh`
takes the program as an optional fourth argument and passes it for the OriginBlame environment
only). The file is read, never started. Nothing else of the record changes, and nobody edits
`identity.json` by hand.

There is no mode that skips a part of this order.

## What a run leaves

The run record is one folder:

- `answers.jsonl`: one line for each question the run answered, each with its commit and the ids
  of the steps it rests on (`step_ids`).
- `attempts.jsonl`: each failed attempt, with the ids of the questions that cite it and of its steps.
- `steps.jsonl`: every command the runner started, as one step with an id (`step-1`, ...): the
  command, the exit code (none where the command did not end by itself), the raw output (standard
  output and error together), the seconds, `lock_wait_seconds` (the wait for the lock, apart from
  the seconds), the files kept with it and whether it hit the cap.
- `locks/<step id>/note.txt`: the note of the lock prefix for that step (with `--lock-wait-bound`).
- `kept/<step id>/<file>`: each file a reading parsed: the companion rows a tokenizer run
  wrote and `listing.json` (name, size and sha256 of everything in its destination, the token
  files included), and for OriginBlame the entry files of the store a step names and the forget
  set it was told to write.
- `unparseable.jsonl`: the lines of the builds that are not JSON, as one line for each file of each
  build (`build`, `file`, `lines`), in name order. A line of a record file that is not JSON holds no
  record: its id is not read, as the tool's own line reader gives such a line no text, and it is
  counted here, never dropped without a count. A generated set plants such lines in `sources/Z.jsonl`
  (docs/plant.md, section 4.5). The file is there when a subject reports the counts, which the
  subjects of `python -m baselines` do; a build is counted when a question first needs its ids.
- `withheld-step-<n>.txt`: the whole text of a step that holds a host or session name, which its
  line leaves out.
- `<tool>/compare.json`, `<tool>/identity.json` (copies of the setup's files, byte for byte) and
  `<tool>/interpreter.txt` (the two lines the probe wrote).
- `start.txt`, `summary.txt` and `sha256sums.txt`, which lists every other file of the folder with
  its sha256. A folder is copied back and checked against it before the machine is deleted.

A step is written with the folders of the run replaced by placeholders: `<work>`, an
environment copy, a build, the interpreter folders and the program. The lines of `answers.jsonl`,
`attempts.jsonl` and `steps.jsonl` go through the record writers, which refuse a local path, a
host name and a session name; a step that still holds a local path after the placeholders is kept
with each such path cut to its last name, and its output says so. `interpreter.txt` is the
probe's own two lines, kept as it wrote them: it names the interpreter path of the machine that
ran.

## Attempts and none lines

A subject that holds no record for a question reads "cannot answer" only after its own attempt on
the store or set the question is asked of. The runner records one attempt of a subject on each store or
set it ran on and writes the none lines from it; a store whose run failed gives the status of that
attempt (`no answer: failed attempt <id>`); a store the tool did not run on gives no line, and the
summary counts the question as not attempted. A store run twice (a writer stopped and then run
again on the same output) holds both attempts: the row `c04.retry` reads the second run, every
other row the first.

A reader that raises an error still gives a line. For the two rows the table gives no join for
(`c06.generated.downstream-of-translation` and `c09.10k.successor-shards`) the line is cannot
answer with the table's words; after a failed tokenizer run the line is the status of that
attempt; an expected value that could not be derived is cannot answer with the words "expected
value not derived". A row that names two layouts is read once for each layout and graded by both. For the two no-join rows
every subject that reads them (Dolma, OriginBlame and the combinations) puts the same line: the table reads a shard and gives no
fold of several shards into one answer, so no reading of the fold is made. In a row that names two layouts each layout that ran
reads its own cannot answer, a layout with no run reads not attempted, and the line follows the layouts' slots (wrong, then a
failed attempt, then not attempted, then right if all are right, else cannot answer).

**Layouts the world does not build.** A question whose listed row names a layout (`layout`, or the two of `layouts`) is put to a
subject only from a build in that layout. Before the first tool step the runner takes the layouts the world builds for each target
(the layout of its shard entries and the layout folder, `partitioned` or `round-robin`, of the chunk files of its tokenizer entries)
and leaves out every question that names a layout with no build: a row of one layout; a row of two layouts that records none; and
a routed row of two layouts of a subject that does not read per layout (plain Dolma, the manifest, the trace). A routed row of two
layouts of OriginBlame or a combination is not left out: its layout with no build reads None. The left-out questions stay not
attempted (never cannot answer, never a stop of the pass) and are named in `layouts-not-built.json` in the run record, beside
`start.txt`: for each its id, the layouts the row names, its target and the layouts built.

**One build per question.** A question reads exactly one build, picked by what its row names. The Dolma subject can hold a run for each layout of a target and reads the run of the layout the row names (the world builds one run per target until a world holds two layouts of a set); the world names the plant list's split pairs in the entry of each set and `c03.pairs.split` reads the list of its own set (a set with no list is an error); the files a request names are checked in the builds the questions read, for every subject. Beside the reading of `c03.pairs.split`, ungraded, the line of plain Dolma (not of the combination) carries a note on whether the tool's own row at each listed position names the listed member.

**The re-grade command.** `python -m baselines.regrade RUN_RECORD --out OUT` grades the kept readings again without running any
tool. It builds the graders from the expected-value sources the record kept (the counting documents, the shard indexes, the
oracle's logs and the request entries) and from the fixture spec, the table files and the acceptance modules the graders load (`expected.ACCEPTANCE_MODULES`) in the repository, which the record does not
copy: the record keeps their sha256 in `frozen-files.json` (written by the runner since 18 Nov 2026; the acceptance modules since the amendment of that date) and a re-grade refuses a
record when a frozen file differs or when the record holds no hash of a file the seal now names, naming it; a record with no such file is re-graded and its output says "frozen files not
recorded". It writes in the new folder `OUT` the lines with their new grade (`answers.regraded.jsonl`; a changed line names the
commit that graded it in `regraded_by_commit`, its own `commit` stays the pass's), the lines whose status or evidence changed
(`regrade-changes.tsv`), the counts (`regrade-summary.txt`) and `regrade.json` (the grading commit, the commit of the record, the
sha256 of the record's seal, whether the frozen files were checked, and whether the code tree had no change beside its commit), sealed. It refuses a record that holds `stopped.json` and
a record whose `sha256sums.txt` does not match its files (exit status 3). A line that rests on a failed attempt or holds no reading is kept as it is (a none line stands only with the
attempt of its subject, which the record does not keep as such, so it is never re-graded and never moves), and a row of two layouts is
re-graded only when both layouts have their raw reading. The summary is made again from the lines: the counts from the lines, the
questions open from the pass's own summary moved by the changed lines (an expected value that was not derived and is now graded leaves
the list; a grade that is now not derived joins it), and the completeness flags from that list. The record it reads is never changed.

## Faults

The faults of Checks 4 and 5 are put by the runner to the tool's own files, never through a call
of the tool. A Dolma fault is put to a copy of the companion rows of the chunk file, written back
in the file's own format (a gzip file stays gzip). An OriginBlame fault is put to the entry files
of a fresh copy of the store: `f.map-deleted` empties them (the section records stay),
`f.digest` changes the first digit of `token_count` of the first entry, `l.count-499` deletes the
last entry. The line of such a row records whether the binary index of the store existed and was
fresh (not older than any entry file) before the fault and after it, because the tool reads that
index when it is fresh.

## The world file

`--world` is a JSON file. Paths are absolute unless a key says relative to the build.

    tokenizer      the tokenizer file
    tie_record     the result of the tie show
    requests       request word -> {"files": [...], "record": id or absent}, files relative to the build
    extensions     the extensions the world was built for (absent: none); a pass that asks an extension
                   not in this list is refused
    split_pairs    the plant list's split pairs
    stores         for Dolma: [{"target": ["store", NAME] or ["set", SIZE], "build": ...,
                     "chunk_files": {artifact name: chunk file relative to the build},
                     "stop_and_run_again": true for the store whose writer is stopped,
                     "copy_of": {shard: the build's own path of the chunk file, for a set},
                     "layout": the layout of a build ADDED beside the target's own (the partitioned 10k set)}]
    shards         for OriginBlame: [{"name", "target", "layout", "run", "successor", "build",
                     "chunk_file", "idx_prefix"}]
    counted, counted_by_layout, index
                   size ("once" for a row asked once) -> counting document or shard index files;
                   counted_by_layout: layout -> counting document, for the one set the world builds in two
                   layouts (the 10k set, round-robin and partitioned; D-WORLD-12)
                   The grader puts that document in the place of a set's own only for the set built in two layouts: a row of
                   another size that names round-robin (Check 3 at small and 100k) keeps the document of its own size.
    subject_inputs.manifest also names, for the 10k set, the manifest of the repaired side of each build
                   (repaired-set-10k, repaired-set-10k-partitioned; D-WORLD-13), made by Check 9's report and
                   repair on a copy of that build's store
    shards         also holds, last of all, the successor shards of those repaired sides for the 10k set, in each layout
                   (flagged successor; their files are copies inside the historical build; D-WORLD-16)

A question is put to a subject only on a store or set the world names for it.

A run that stops on one question (an error a reader raises that no rule above turns into a line)
writes and seals the lines and attempts it has made so far, and the error names the question. A
pre-run stop seals the record with its steps and no answer line.

## The summary

`summary.txt` counts the questions by subject, size label (`once`, `small`, `10k`, `100k`) and
status word, counts the tally without the questions the table excludes, and says whether Check 10
is complete or open, for the run and for each label, with the open questions listed. The small set
is kept apart from the plan sizes. For each subject and label that answered anything it gives two counts side by
side, never merged: `answered` (the questions the subject could answer, a right or a wrong; a cannot answer, a failed
attempt and a question not attempted are no answer) and `right among answered`, in the whole summary and behind `tally`.
The text holds no sentence that sets one tool against another.

## Check 10 results

This section gives the counts of the questions of Check 10 at each size label, with their sources. The README holds the table of the questions asked once (`README.md`, "Check 10 results"). The counts are the tally as the documents define it (the section "The summary" above; `fixture/table/README.md`, "Marks and the tally"). Each row has two counts, side by side and never merged: "answered", the questions for which it gave a right or a wrong answer, and "right among answered", the questions among those that it answered right. A question that the table leaves out of the tally is in neither count, and "questions in the tally" is the number left. For the six subjects of the run, the documents' words for the three grades are: right, "the reading gives the expected answer of the row asked"; wrong, "the reading gives another answer"; cannot answer, "the subject ran, and its own records lack the record or field the question needs". A cannot answer is not an answer and is in neither count. For the Waterwhite rows, "answered" and "right" both mean that the record ended "passed" (the core result) or that the test passed (the test run): the row matched the fields the table pins. A question with no passed record is in the tally and in neither count.

| subject | size label | questions in the tally | answered | right among answered |
|---|---|---|---|---|
| Waterwhite (Checks 1 to 9 record, compared on tree c113fd44) | once | 362 | 353 | 353 |
| Waterwhite (Checks 1 to 9 record, compared on tree c113fd44) | small | 17 | 17 | 17 |
| Waterwhite (Checks 1 to 9 record, compared on tree c113fd44) | 10k | 4 | 4 | 4 |
| Waterwhite (Checks 1 to 9 record, compared on tree c113fd44) | 100k | 17 | 17 | 17 |
| Waterwhite, generating-model questions, switch on (test run on the tree of the first commit) | once | 9 | 9 | 9 |
| Dolma toolkit | once | 362 | 18 | 17 |
| Dolma toolkit | small | 17 | 7 | 7 |
| Dolma toolkit | 10k | 4 | 1 | 1 |
| Dolma toolkit | 100k | 17 | 7 | 7 |
| OriginBlame | once | 362 | 25 | 14 |
| OriginBlame | small | 17 | 2 | 2 |
| OriginBlame | 10k | 4 | 0 | 0 |
| OriginBlame | 100k | 17 | 2 | 2 |
| File-level manifest | once | 362 | 138 | 102 |
| File-level manifest | small | 17 | 0 | 0 |
| File-level manifest | 10k | 4 | 0 | 0 |
| File-level manifest | 100k | 17 | 0 | 0 |
| Positions-only step trace | once | 313 | 1 | 1 |
| Positions-only step trace | small | 17 | 0 | 0 |
| Positions-only step trace | 10k | 4 | 0 | 0 |
| Positions-only step trace | 100k | 17 | 0 | 0 |
| Dolma toolkit with the trace | once | 313 | 48 | 33 |
| Dolma toolkit with the trace | small | 17 | 7 | 7 |
| Dolma toolkit with the trace | 10k | 4 | 1 | 1 |
| Dolma toolkit with the trace | 100k | 17 | 7 | 7 |
| OriginBlame with the trace | once | 313 | 80 | 27 |
| OriginBlame with the trace | small | 17 | 2 | 2 |
| OriginBlame with the trace | 10k | 4 | 0 | 0 |
| OriginBlame with the trace | 100k | 17 | 2 | 2 |

A size label says at which size a question is put: `once` is a question asked once, and `small`, `10k` and `100k` are the sizes at which the other questions are asked; the small set is kept apart from the plan sizes (the section "The summary" above). Each row of the table is one subject, or one Waterwhite source, at one size label. The rows of the six subjects of the run agree with the `tally` lines of the run's `summary.txt` and are made from the run's answer file (sha256 `9f19fd752f44b57fcc10d62f6f01f323b597539bfb915d880917e6eb95be9477`), which is not in this repository. Over the four labels, the trace and the two combinations have 49 questions each that the table leaves out of the tally; the other three of the six have none. No question is left out of the Waterwhite rows. The row of Waterwhite in the README is the sum of the two Waterwhite rows for the questions asked once: 353 + 9 = 362.

- **Dolma toolkit** 1.2.1, run on Python 3.12.13: the toolkit's tokenizer is run on the chunk files, rows are selected by source, and the id of each row is resolved to the file that holds it.
- **OriginBlame** 0.2.2 (rust-originblame, commit 622e3513), run on Python 3.12.13: on a fresh copy of the store, the native by-sources call with the sections of the files of the members the request names, called once; then generate-set; bits read least significant first.
- **File-level manifest**: a file-level manifest exported from the store of a build that holds one.
- **Positions-only step trace**: a record of what a training step read: step, shard, ordinal and token range.
- **The two combinations**: the history is read from the trace (the checkpoint's segment up to its step, then the history of its parent checkpoint), each trace row is tied to the tool's record for its document, and that record is read.

The first Waterwhite row is from the committed core result (`tests/acceptance/core_result/`): its 391 core rows ended passed, each recorded on the tree `1c132e32` (2 of them on a later tree, named in the file's header) and compared equal on the tree `c113fd44`. That record was made with every extension switched off, so it holds no answer for the 9 generating-model questions: they count in the first row's tally and in neither of its counts. The second Waterwhite row is for those 9 questions, from the test run on the tree of this repository's first commit with the rule on (11 tests, all passed; junit file sha256 `ba32ddf042f5dd415213afe1832b7c8dd677b5058617c17daf53a19bd48ffb66`); that record is not in this repository, and neither of the trees `1c132e32` and `c113fd44` is in its history. The questions are Waterwhite's own. For the core rows, `tests/acceptance/README.md` says that the table was written by hand before the tool existed and that a row is never changed to fit an answer, and `fixture/table/README.md` says that no row comes from an output of the tool.

## The world builder (temporary section WORLD)

`python -m baselines.world` builds the world a run is put to, with the repository's own builders
called unchanged, and names and hashes every file it made. Nothing in it is typed by hand: each
input comes from a builder, from the frozen table (`fixture/table/check10.toml` and its README
handles), from the fixture spec or from the plant list, or is a named input of the command line.

    python -m baselines.world --out DIR --megatron-env DIR --tokenizer FILE --jobs N \
        [--sizes small [10k [100k]]] [--extensions NAME...]

- `--out` is a new folder; one that exists is refused before anything is built. The command
  writes nothing outside it.
- `--megatron-env` is the Megatron environment folder the builders tokenize with. `--tokenizer` is
  the tokenizer file a run hands to the toolkit; it is copied into the world and hashed.
- `--extensions` names the extensions whose rows the world is built for (default: none), and only an extension the
  builder has a build for (`BUILT_EXTENSIONS`: `generating-model`): any other name is refused with exit 2 before anything is
  built, since a flag is no build. With `generating-model` the main store is the replica's build `main+generating-model`
  (the main build, then the release R3 in the same store; it needs the confined harness, which the stores of the world use
  where a sandbox can be made, and the world refuses to build without one). The repaired side of main is made on a plain build of main made again in the folder of the repair's copy (`replica repair` does not know
  R3, an extension release, and the report of a store that holds it names it), so it is the one of the world without the extension. The files of the main build are the same bytes
  as in the world without the extension; the build adds the files of R3 and two rows, `gen/R3-models` and `R3`, to the
  build's `built-from.jsonl`. The routed rows of an extension are scanned for the successors, the stopped targets and the
  request words only while it is on, and `world.json` records the names in `extensions`.
- `--jobs` is the number of workers: builds that do not depend on each other run at the same time,
  each in a process of its own, and the number is handed to every builder that takes a job count.
  Inside a build the steps that read and write files of their own go on together too (in a store the
  tie show of its chunk files, the runs and the repair; in the runs the registration in the store and
  in each starting state of Check 2; in a set the counting and the tie show), and the tie show of a
  build uses one line tokenizer for each worker, at most one for each chunk file. With `--jobs 1`
  every step runs in order. A world meant for a run on a machine is built with `--jobs` set to the
  cores of that machine; the report says how many cores the command kept busy. The seeded files are
  the same, byte for byte, with one worker and with several. The files of a build made without
  capture (the store `no-capture`, README section 4) draw their shuffles from the operating system's
  randomness, on purpose, so its three chunk files hold the same lines in an order that differs from
  world to world. They are compared order-free: as the lines over the three shards together, between a
  world of one worker and a world of several; every other file of that build is compared by its
  sha256 like the seeded files. The worker counts compared are one worker against four (the count of the worlds of record) on a machine
  under 24 GB of memory or under eight cores, and one against four and against one for each core on
  a larger machine. Limit: the branch of the tests for a larger machine (worlds of one, four and one for each core, built together)
  ran on one rented server (16 cores, 30 GiB of memory) on the tree `f1bdaf4f`, where its tests passed, and has not run on a later tree; its memory was not measured there. What guards every file, the unseeded ones included, is its sha256 in
  `WORLD.sha256`: a run refuses the world before any tool starts (stage `world-hash`, see "Before a
  tool starts") when a file is not the listed one, a listed file is not there, a path is listed
  twice, a path `world.json` names has no row, `world.json` has no row or the folder has no
  `WORLD.sha256`, naming the file; the start line of the run names the sha256 of that list.
- `--sizes` names the generated sets (default: `small` only).

What it builds, all under `DIR/work/` (a build of Dolma lies inside the work folder of a run):

- the stores `main`, `rr`, `no-capture`, `scratch-500` and `scratch-notokens`, tokenized, each as a
  Dolma entry (`scratch-500` with `stop_and_run_again`, the store whose writer is stopped and run
  again); and as OriginBlame shards: `main` (`E/shard-1` to `E/shard-3`, layout `partitioned`, run
  `E`, and the successor of the third shard as the fourth and last shard of the world, see below), `rr` (`E-rr/shard-1` to `E-rr/shard-3`, layout `round-robin`, run `E-rr`), `no-capture` (`E/shard-1` to
  `E/shard-3`, layout `partitioned`, run `E`), `scratch-500` (`c04/shard-500`) and `scratch-notokens`
  (`c03/shard-notokens`). Every store of the list has OriginBlame shards of its own since 17 Nov 2026 (the table holds OriginBlame rows for
  `no-capture` and `scratch-notokens`). The 500-line file of `scratch-500` is the record
  file of the source the store ingests it under, so it is copied to `sources/c04.jsonl` of that
  build, where the readers of record files look;
- for each size: the generated set, its capture build in the round-robin layout, the counting
  document of the plant script on the chunk files of that build, the shard index files and the
  split pairs (the pairs the counting document places split), a Dolma entry `["set", SIZE]` and the
  OriginBlame shards of the set. The sets of all sizes deal their records over the same chunks, so
  their chunk files carry the same names; the smallest size names the path its build wrote
  (`pipeline/round-robin/tokenizer-input/shard-NN.jsonl`) and every other size names a copy of each
  chunk file under `set-SIZE/tokenizer-input/`, so that the tie record, which is keyed by the chunk
  file's path, holds one result for each;
- the successor of a shard: the report of the withdrawal the table's rows name, printed by the tool
  on a copy of the main store, and the replica's repair of that copy. The successor's chunk file is
  copied into the main build under a folder of its own (`successor/tokenizer-input/`) and is the
  chunk file named `successor of E/shard-3`, and the index and packed file of the repaired side's
  tokenized shard are copied beside it (`successor/shards/`), so the successor is the fourth and last
  OriginBlame shard of `main`, named `successor of E/shard-3` and flagged successor, with its three
  files inside the build of main; it comes last in `shards` and in `tracked_order`, so no other shard
  number moves;
- the runs of the stores that hold them (`main` and `rr`), built by the replica and registered
  through the tool, and the trace documents the tool's show command prints for their segments;
- the tie record: the tie show (`baselines.tie`) of every chunk file of every Dolma build, with the
  line tokenizer of the Megatron environment. A chunk file two builds name (the paths are relative to
  the build) is merged only when the two results are equal; a difference stops the command and
  names both builds;
- the requests: each request word a routed row carries, with the record files of its source and the
  one record it narrows to, read from the main build and the spec.

What it writes in `DIR`:

- `world.json`, in the format of "The world file", written after every file of `work/`;
- `WORLD.sha256`, one line for each file `world.json` names (and `world.json` itself): the sha256,
  two spaces, the path relative to `DIR`, in path order. Besides the files of each key it lists the
  record files and chunk files of every build and the request files each build holds;
- `world-report.txt`: the builds, the files and the hashes counted, the seconds of each step, the
  workers used, and the processor seconds of the command and of every process it started with the
  cores that kept busy on average (the summary the command prints says the same). It is not named by
  the world and not hashed.

`subject_inputs` is the key of the files the four subjects the runner does not drive read, so that
a later driver finds them and a pass is tied to them:

- `manifest`: for each build the `built-from.jsonl` at its root (path and sha256), or why a build
  has none; and the `built-from.jsonl` of the repaired side;
- `trace`: for each Dolma entry (`store-main`, `set-small`) the exported trace document
  (`{"segments": [...]}`, made from the tool's `show SEGMENT --trace` by `baselines.trace`), with
  the segments the tool refused by name in a starting state, with its words (`failed`); for
  `store-main` also `state_traces`, the trace written from each starting state of Check 2 (`c2.i`
  to `c2.iv`); or `{"none": reason}` for a store or set with no run registered. Any other end of
  the tool's show command, and a refusal in a store as built, stop the command;
- `megatron_tokenizer_files`: the vocabulary and merges files of the Megatron environment the line
  tokenizer reads, each by its name inside the environment folder and its sha256;
- `repository`: the frozen files of the repository the trace reads its places from, named by path
  relative to the repository and sha256;
- `records`: the record files of each build; `world_keys`: the keys of the world whose files the
  subjects read (their hashes are in `WORLD.sha256`); `tracked_order`: the order of `shards`, which
  maps `tracked-N` of OriginBlame to a shard.

## The subjects (temporary section SUBJECTS)

One pass drives any of the six subjects named by `--subjects` and leaves one sealed run record.
The drivers run in one order whatever the option says: `dolma`, `originblame`, then the other four in
the order of the table's list of subjects (`manifest`, `trace`, `dolma+trace`, `originblame+trace`). A combination needs its plain half in the same pass
(`dolma+trace` needs `dolma`, `originblame+trace` needs `originblame`); without it the run stops
before any command starts. A combination starts from the targets its plain half ran on and the
failed attempts of its runs, so no baseline tool runs again for a combination's line, and its none
lines stand on the same attempts. The plain subjects' lines, kept steps and failed attempts are the
same whether two or six subjects are driven; a step a later driver logs for a store belongs to that
driver's lines only.

`originblame+trace` reads the eight groups of the tool half (`c03.spans`, `c04.interrupted-write`,
`c04.count-499`, `c05.unmutated.shards`, `c05.withdraw-C.unmutated`, `c06.downstream`,
`c09.artifact-actions`, `c09.successors`) and the shard rows of `c05.f.map-deleted` and
`c05.f.digest` from the plain OriginBlame line of the same listed row and size. The raw reading is
the plain line's; the combination's own row grades it, and its line cites the steps of the plain
line. A row whose plain row holds no record has no twin and is read by the tool half. With the generating-model extension on, the row of the group `c05.withdraw-C-on` (the trace holds
no record about its artifact) is read the same way, from its twin.

A pass that stops (an error that leaves the run) seals its record with `stopped.json`: the text of
the error, the driver the pass was in and the question it was answering, none where the error came
while a driver was built. A stopped pass is not graded. `subject-facts.jsonl` holds the facts a
driver reports (each export with its outcome, each half's failure with its own id, each shard's
failure), sealed, also after a stop. The `driver_meta.json` of a store is kept with the entry files
of each step on it.

The manifest reads the walks of one artifact, the two mixture lists, the exposure-free restart and, since
17 Nov 2026, the artifact actions (at fixture size: one artifact's action, and the fixture's three shards of
a run listed by action), the successors (read in the manifest of the repaired side), the history (the rows of the
historical and the repaired build joined), the listing of the manifest after the fault `f.map-deleted` (every name
it holds by kind, and the names with no row of their own as `not_covered`; the manifest holds no run, so none is
listed) and the file family of an import (the files the `built_from` list of the row of `import/T` names). The
listing is graded by membership with no order and no reason, the file family by the files it names against the file of the member the listed row gives (the file comes from the world's own request entries and the fixture spec's member of each source).
The rows of the 10k set (since 18 Nov 2026): `c09.10k.partitioned.kept-shards` is the list of shards read keep and `c09.10k.round-robin.all-rewrite` the number of chunks read rewrite (the table's rule text), over the rows the manifest itself names (docs/replica.md section 16: a shard is a row whose files are its `.bin` and `.idx`, a chunk the row `<shard>.input`), each in the manifest of the layout the row names (a build added beside the target's own has the key `set-10k-partitioned`); `c09.10k.successor-shards` is the subject's cannot answer (the table names no fold of many shards into one answer); `c09.10k.R1-history` is not read until the repaired side of the set is built. The byte identity of the kept shards afterwards is not read (the rule text says so).
A question not read gives no line and a line in `subject-facts.jsonl`. `dolma+trace` and `originblame+trace` read the runs of
their plain half (the same objects, so no tool runs again) and the trace of each store or set.

The trace of a store with no registered run is `{"none": reason}`, under the key `<kind>-<name>` of the store (`store-swap`), in the trace driver's input: the
driver has run on that store, and its none lines are the table's own text.

## None rows on a store with no chunk file and no shard, and a refused track

A none row (a row whose record is none and whose rule is cannot answer) is put after the attempt of its subject on the store or
set its row names. A store the world names for the manifest or the trace that holds no Dolma entry and no shard of OriginBlame (the
table's stores `import` and `swap`) can be no target of a tool's own run: the attempts of the pass stand for it. Where the tool
completed a run in the pass, the none rows of the tool and of its combination read cannot answer with the row's own missing text; where
it completed none, the line is its failed attempt; with no run and no failure the line stays not attempted (`D-SUBJ-9`).

A line of a chunk file that gave no tokens was never packed: the shard's own index holds it as a document of no tokens (the
planted empty-text line of the store `scratch-notokens` is one), and the plan of a shard names the packed documents only, read
from that index and never from the plant list. The tool is told of the documents that were packed and runs on the shard; the
none rows of that store are then meant to read cannot answer with the row's own text ("holds nothing about a line that was skipped"; built, not yet shown on a pass). A packed
line whose id no record file holds is still refused (`ValueError`), and a track the tool itself refuses on a shard is a failed
attempt: the none rows of that store read "no answer: failed attempt <id>", and the writer rows of the shard give the same
failed attempt. (An earlier version of this section said a refused track reads as a failed attempt on this shard, and then that
the shard is left out; neither route happens.)

The oracle's log of each run the world builds (the source of the three whole-run diff rows `c01.E.diff`, `c01.E-rr.diff` and
`c01.K.diff`) is written at world build by `oracle/fixture_runs.py` through the acceptance helper, one file of lines for each run
(world key `oracle_logs`), hashed with the world; the runner is built to grade the three rows with `grade_whole_run` against it (not yet shown on a world or a pass), and a world
with no log gives the line "expected value not derived". The run record keeps, sealed beside the answers, the files the grader
read the expected values from (the counting documents, the shard indexes, the oracle's logs) in `expected-sources/`, with an
`index.json` of their sha256: a build made without capture cannot be built again (its shuffles are unseeded), so its expected
values exist only in the world that was built. The copies are kept, and `runner.world_from_record` reads them back (each checked against its sha256) in the shape of the world keys, so a re-grade builds its graders from the record (the counting documents, the indexes, the oracle's logs and the world's request entries are kept; a round-trip test shows the same grade for a file-family question and the same documents for a question of the set made without capture); the command that re-grades the kept readings is owed.

A line the grader cannot grade (an expected value not derived, or a row not gradable) keeps its line with the status cannot answer and
its words as the evidence, and the summary counts it under the word `expected value not derived` and leaves the question open.
