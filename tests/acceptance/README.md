# Acceptance tests

These tests put the rows of the hand-written table (`fixture/table/`) to the tool
and compare. The table was written by hand before the tool existed, so a row is
never changed to fit an answer, and neither is the tool changed here: this folder
holds test code only.

Plan sections served: 5.5 (the acceptance checks, here Checks 1 to 9), 5.1
(the fixture and the generated sets), W5 (expected answers come from outside the
tool).

## Files

| File | What it holds |
|---|---|
| `table.py` | reads a table file into rows, interprets a row's `given`, compares an outcome with the row's `expect` |
| `acceptance_world.py` | builds the table's stores with the replica, copies them, resolves record handles, runs the command line, puts each kind of question |
| `acceptance_faults.py` | the faults of the table's README section 7, each applied to a copy of a store |
| `acceptance_rows.py` | turns rows into tests and decides whether a row runs, is skipped or is an expected failure |
| `acceptance_probe.py` | what the machine and the tool can do today, found out by asking: whether `show` prints what Checks 3 and 6 need, whether the sandbox of the confined harness can be made, and which rows rest on a bound |
| `acceptance_sets.py`, `acceptance_maps.py` | the generated sets and the comparisons of Checks 3 and 6 (section "Checks 3 and 6" below) |
| `expected_failures.py` | rows that run and fail, each with its diagnosis |
| `conftest.py` | the session's stores and the count printed at the end of a run |
| `acceptance_runs.py`, `acceptance_join.py` | the run world (which segments a store registers, the starting states of Check 2, the commands that build and register runs, the oracle's own runs) and Check 1's comparison of the join's rows with the oracle's log, as pure functions (section "Checks 1, 2, 4 and 8" below) |
| `acceptance_trace.py` | what `show SEGMENT --trace`, `register-run` and `replay` print, read as pure functions: the rows of a trace in the shape of the oracle's log, the coverage mark, the ranks of a step, the launches, the step after which the epoch changes, the shuffle entries against the (launch, epoch, rank) cells, the kept cache file of each array and the comparison of a replayed array with it |
| `acceptance_scratch.py` | the writes of Check 4's scratch store: the map writer killed after 100 rows, the retry, the refused count of 499 and the edit of the stored count, and the table's sentences for what the store holds afterwards |
| `acceptance_report.py` | the rebuild report and the verification of a repair read as pure functions, in the form of docs/interfaces.md 7.1 and 7.3: the action of a record, the fields of an artifact's entry, the mixtures, the groups, the restart points, the `kept`, `caches` and `restarts` of a verification |
| `acceptance_repair.py` | a repair driven through the three documented command lines (`waterwhite report`, `python -m replica repair`, `waterwhite verify`) on a copy of a store, a wrong report made by changing one entry's `action`, and, for the 10k set, its build and repair in each layout (`SetRepairs`); the multiset and the shard digest read from the files the sides file names |
| `acceptance_check09.py` | Check 9: its rows read on their own (the table reader refuses most of them), what each waits for, the expected values from the plant list, the comparisons and `put_fixture` and `put_set` |
| `mini_store.py` | a small store written by hand with the fixture's names, for the tests of the faults on a map, a trace, an edge parent and a file, and a small stand-in for the scratch store of Check 4 |
| `test_check01.py`, `test_check02.py`, `test_check03.py`, `test_check04.py`, `test_check05.py`, `test_check06.py`, `test_check07.py`, `test_check08.py`, `test_check09.py` | one test per row (per size, for the rows of Checks 3 and 6 that are asked of a generated set) of `check01.toml` to `check09.toml`; `test_check07.py` and `test_check09.py` also put every reading of the report and of the verification to documents written by hand |
| `test_table.py`, `test_faults.py`, `test_faults_on_records.py`, `test_maps.py`, `test_probe.py`, `test_acceptance_world.py`, `test_harness_confined.py` | the reader, the faults, the comparisons, the probe, the world's reading of a store and the confined harness row tested on their own |
| `test_table_runs.py`, `test_runs.py`, `test_join.py` | how the rows of Checks 1, 2, 4 and 8 are read and what they wait for, the plan of a registration and the starting states of Check 2, and Check 1's diff shown to fail on rows written by hand |
| `test_trace.py`, `test_join_row_fields.py`, `test_launches_epochs_shuffle.py`, `test_registration.py`, `test_scratch.py` | the reading of a printed trace and of the arrays a replay wrote, the rows of Checks 1 and 8 that read the trace, the launches, the epochs, the skipped line and the shuffle, the rows of Check 2 that read a registration and a replay, and the scratch store's writes, each put to documents or a store written by hand so that it can be seen to fail |
| `jobs/` | the job script of the harness row, run in the sandbox, and the map writer of Check 4's scratch store, run as a process that is killed |

## How a row becomes a test

Each row is one test, named by the row's id:
`tests/acceptance/test_check07.py::test_check07[c07.withdraw-A.R2]`.

1. **The row is interpreted.** `table.interpret` reads `given`: the store (left
   out: `main`), the state or fault (left out: `unmutated`), the request, the
   artifact, the record. It raises `TableError` for anything it cannot read: an
   unknown key, store, state or request, a name that is no artifact name of
   docs/interfaces.md section 9.2, a fault asked with another request than the
   table's README fixes for it, an expected field that the way of asking does not
   give. That is an error of the test. A row is never skipped because it could not
   be read.
2. **Extension rows.** A row with `core = false` is skipped with the reason
   `extension off: <extension>`. What the tool does while an extension is off is a
   core row of the table (`c07.ban.off`, `c05.g.model-entry-removed.off.E/shard-3`)
   and runs like any other.
3. **Needs.** The row's needs follow from its `given`: a fault needs the record it
   is put on (a step trace, a tokenization map), a store needs what it holds, an
   artifact needs its kind to be registered (a shard, a mixture, a checkpoint, a
   successor), an expected action needs the rebuild report. A row with a need that
   `acceptance_world.NOT_BUILT` lists is skipped with the reason
   `not built yet: <needs>`. A skipped row is not a pass.
   **A row that rests on a bound.** Only a log written by the confined harness
   establishes a bound (Decision 13), and the confined harness needs a sandbox
   (bubblewrap). A row whose expected answer rests on a bound (a "does not contain"
   past a generation edge, an answer that is "may contain" because a bound is not
   established, a fault on a generation edge or its log) is, where the sandbox
   cannot be made, skipped with `not run: it rests on a bound ...` and the sandbox's
   own reason. It is not run, and it is not a pass.
   `acceptance_probe.rests_on_bound` counts these rows; the query of a row it did
   not count stops with an error when the tool's answer lists generation edges.
4. **Pending values.** `"pending:<key>"` in an `expect` is filled from the fixture
   spec. A row whose spec item is still empty is skipped as not checked.
5. **Expected failures.** A row listed in `expected_failures.py` is marked as an
   expected failure that carries its diagnosis (below).
6. **The question is put and compared.** Every expected field that differs is
   reported with what was expected and what the tool gave.

### A row with a field that cannot be compared yet

`c07.withdraw-A.R2` expects an answer, its notes and the rebuild action `keep`. The
answer and the notes can be compared today, the action cannot. Such a row gets two
tests: `...[c07.withdraw-A.R2]` compares the fields that can be compared, and
`...[c07.withdraw-A.R2 [not compared: action]]` is always skipped, with what the
field waits for as its reason. The row has passed only when it has no second test
any more.

It is the one such row of Checks 5 and 7. `c07.swap.instance-id-changes` expects the edge
digest of the two releases to differ; `show` prints an artifact's edge digest, so
that field is compared like the others and the row has one test. Rows of Check 2
hold fields that wait in the same way (section "Checks 1, 2, 4 and 8"). A row of Check 1
whose field is read from a form the command does not print yet is not split in two: it
is skipped by the name of the form, after every other field of it was compared
(section "Checks 1, 2, 4 and 8", "A form the command does not print").

## How a question is put

Every question goes through the command line as a caller would use it: a
subprocess of `python -m waterwhite.cli` with `src` on the Python path. The answer
is the JSON document on standard output. Where a row expects a refusal or a loud
stop, the exit code and the `error` field of the object on standard error are
compared, and that standard output is empty.

| Kind of row | Command |
|---|---|
| a request put to one artifact | `query withdraw SOURCE [--record ARTIFACT:LOCATOR] --artifact NAME` |
| a request put to `(listing)` | the same with no `--artifact`; `query ban TEXT` for the ban request |
| which copy a release row is attributed to | `show ARTIFACT --record LOCATOR`, the row's edge |
| two releases compared | `show` of each: content digest, instance id, producing step, inputs; `show --record` for the occurrence key of X |
| the same history registered again | `show` and `check` before and after the replica runs the store's build a second time into a copy |

One row does not go through the command line: `c05.harness.outside-read`. The
command line has no command that runs a generation job, so the test runs the job
script `jobs/asks_outside_record.py` in the confined harness
(`waterwhite.confine.run_confined_job` with `python_program`, docs/interfaces.md
section 5.3). The job reads the records it was handed, asks for one record outside
its hand-off, and emits "refused" or the text it was given; the test reads the log
file the harness wrote. Only a log of the confined harness establishes a bound
(Decision 13), so where bwrap is missing or cannot make its namespaces the row is
reported not run, with the sandbox's own reason, and never run in the process.
`test_harness_confined.py` reads back what shows the row ran confined: the log's
harness word, the refused read and the job's output.

How each field of `expect` is compared is in `fixture/table/README.md` section 6
and in `table.compare_answer`: notes as the full set, a chain as hops passed in
order with other hops allowed between them, a failing link key by key. Only the
fields a row holds are compared.

## Stores, handles and faults

- **Stores.** `replica.fixture.build_table_store(folder, name)` builds a store of
  the table's README section 4 (docs/replica.md section 5). Each store is built
  once per session; before a row is put to it the runs of a store are registered
  (section "The run world"), and rows only read it afterwards. Where the replica's function takes
  `generation_harness` and the sandbox can be made, the store is built with
  `generation_harness="confined"`, so that its generation edges can hold a bound
  (`acceptance_probe.generation_arguments`). Whether a store holds a shard or a
  mixture is told by the names its build registered, never by a list. A store
  that fails to build is reported to every row that asks it with the first cause.
- **Handles.** A row names a record by a handle (`X@A`, `P`, `R1:S`). Its line
  comes from the manifest the replica wrote beside the record's file, never from
  the tool, and its locator is `L<line>`. `X` in a release is the row whose
  manifest handle is `X@A` or `X@B`; `U-<nnn>` is `Z-<nnn>`.
- **Faults.** A fault is applied to a fresh copy of the store's folder, one fault
  per copy, by the route the table's README section 7 gives: a direct edit of the
  SQLite file for "outside the write API", the store's write method for the faults that
  go "through the write API" (and, for `l.count-499`, a write through the tool followed
  by an edit of the store file). Each edit checks that it changed
  exactly the rows the catalogue names. `test_faults.py` reads every faulted copy
  back through `show`.

## What is not built yet

`acceptance_world.NOT_BUILT` is not a list kept by hand. It holds what the probe found the replica
and the tool lack today: the forms of `show`, `replay`, and, for Checks 6, 7 and 9, the rebuild
report (`waterwhite report`, docs/interfaces.md 7.1), the verification (`waterwhite verify`, 7.3)
and the successors (the replica's `repair` word, 9.5, which needs a report to work from). The
probe asks each command for its help, once per run (`acceptance_probe.repair_gaps`), and a
command that exists leaves the dictionary, so the rows that wait for it start to run on the
commit that builds it, with no edit here. Rows that need one are skipped and counted. The scratch store of Check 4 is no entry any more: the replica builds it and
`acceptance_scratch.py` writes its set-ups. A tokenization map is no entry: the faults on one
are written (`acceptance_faults.py`) and its rows run wherever the store `main` is
tokenized, like the rows about a shard. Runs, their step traces, checkpoints and the record
W are no entry either: they are one need, `run capture`, answered by asking whether the two
commands that build and register runs exist (section "Checks 1, 2, 4 and 8").

When the commit that builds one of them lands, its entry leaves the dictionary on its own
and the rows that waited for it start to run; the way of asking that those rows need is
written already (the section "Checks 7 and 9, and the rebuild"). The dictionary cannot go
stale unnoticed: the tests stop with an error when the replica registers an artifact of a
kind it still calls missing.

What `show` prints is not a list kept here, because a list goes stale: the output
appears and the rows stay skipped. `acceptance_probe.show_gaps` registers a two-line
file in a scratch store through the command line and reads what `show` prints of
it, once per run. An output that `show` prints leaves `NOT_BUILT` on that run, so
the rows that wait for it start to run with no edit here. The probe's judgment is
tested on documents written by hand (`test_probe.py`). The same probe asks for two forms
of the runs: `show NAME --trace` of a name nothing holds (a parser that does not know the
flag ends with the usage error, a form that exists refuses the name as "request refused")
and `replay --help` (a parser that does not know the word names it as an invalid choice).
Any other ending is an error of the probe and never a skip. A row that waits for one is
skipped with the form as its reason: `not built yet: show --trace`, `not built yet:
waterwhite replay`. The printed `replay` object of `register-run` is not probed: the rows
that read it compare the route and the cause that `register-run` prints today and are
skipped, with `not built yet: register-run's replay object`, only when those agree and
the object is absent. The same holds for three more forms, none of them probed: the
`launches` of `show SEGMENT --trace` (`not built yet: show --trace launches`), the `epochs` of
its steps (`not built yet: show --trace epochs`) and the `shuffle` of `register-run`
(`not built yet: register-run's shuffle`). Each is looked for in what the command printed for
the row, a form that is present is compared, and a form that is wrong fails the row; the
skip comes only after everything else the row names was compared and agrees.

`c05.unregistered-checkpoint` asks about `E@12`, a name no record holds. It runs
today on a main store in which run E is not registered at all. It keeps running
unchanged when run E is.

## Expected failures and their diagnosis

When a row runs and fails, neither the row nor the tool is bent. Someone works out
which of three it is and writes it into `expected_failures.py` under the row's id:

- `tool`: the tool is wrong, with the rule of the plan it breaks;
- `row`: the row is wrong, with the rule it contradicts;
- `seam`: a name or signature between two parts that a revision in progress
  changes.

The entry names the expected fields that differ. The test is then an expected
failure (`XFAIL`) that prints the diagnosis. It is not a pass. If the row fails in
another field than the entry names, the test fails as a new failure. If the row
starts to pass, the test shows as `XPASS` and the entry is removed.

## Running

From the repository root, with an interpreter that has pytest and numpy:

```
python -m pytest tests/acceptance -q -rs -m "not slow and not megatron" \
    --basetemp=.work/pytest-acceptance
```

The stores are built under the base temp folder. The run needs no Megatron
environment and no network; the rows that rest on a bound need the sandbox of the
confined harness (bubblewrap) and are skipped without it. The store `main` is
tokenized only where the Megatron environment is set (`WATERWHITE_MEGATRON_ENV`):
there its rows about shards run, and without it they are skipped with that reason
while its other rows run. The tests that need Megatron or a generated set of the
plan's sizes carry the markers `megatron` and `slow`; "Running Checks 3 and 6"
below says how to run them. A set of the plan's sizes is a long build and runs
only when `WATERWHITE_CLEARED_LONG_RUNS` names its check and size, for example
`3:100k`, set for a run that has been cleared; the marker alone does not start it. It starts the command line many times, about a
third of a second each on a laptop, most of it the tool's own start. On a shared
machine run it through the machine's CPU lock.

## Reading the counts

pytest prints its own line, which counts tests, and `-rs` lists the skip reasons
with their numbers. Before that the run prints the count of rows by check, for
example (the numbers move as the parts the rows wait for are built):

```
acceptance rows by check (plan 5.5)
A skipped row is not a pass. An expected failure is not a pass.
Check 5: 221 rows; ran 35, skipped 186
  ran 35: passed 35, failed 0, expected failures 0, expected failures that passed 0
  skipped 29: not built yet: run capture
  ...
Check 7: 27 rows; ran 8, skipped 19
  ...
  1 of the rows that ran hold an expected field that was not compared; each has a skipped test of its own that names the field
```

- **rows** is the number of rows in the table file. **ran** is how many were put
  to the tool; the rest were **skipped**, each line giving a reason and how many
  rows it holds.
- A check is answered only when every core row ran and passed in full. The count
  above says how far that is.
- The tests of `test_table.py` and `test_faults.py` are not rows and are not in
  this count.

## The red run

A test of this folder must be able to fail. To see it, grade a scratch copy of the
table in which one expected answer is wrong:

```
mkdir -p .work/red-table
cp fixture/table/check05.toml fixture/table/check07.toml .work/red-table/
# in .work/red-table/check07.toml, row c07.withdraw-A.R2: answer = "contains"
WATERWHITE_ACCEPTANCE_TABLE=.work/red-table python -m pytest tests/acceptance -q \
    --basetemp=.work/pytest-red -k "withdraw-A.R2"
```

The variable names the folder the table files are read from. The test of the row
must fail with `answer: expected 'contains', got 'does not contain'`. The variable
is for this and for nothing else; the committed table is never edited.

## Adding a check

Add `test_check<NN>.py` beside the two that exist, with `table.load_check(NN)`.
Its rows will raise `TableError` for every state, store or expected field the
reader does not know yet. Teach `table.py` each one from the table's README, add
the probe of what it needs to `acceptance_probe` if that is missing, and add the way of asking to
`acceptance_world.py` when it is there.

## Checks 3 and 6

`test_check03.py` and `test_check06.py` grade the rows of `check03.toml`
(tokenization links, 15 rows) and `check06.toml` (missing evidence, 17 rows).
Plan sections served: 5.5 Checks 3 and 6, 3.3 (tokenization, copy steps), 3.1
(Drop), 3.2 (labels), 5.1 (the generated sets and where their expected answers
come from), F1, L1, L3, L5, W5.

| File | What it holds |
|---|---|
| `acceptance_sets.py` | the generated sets: generating one, building it through the replica, counting it, the team's tokenizer, and the set-ups of README section 7 for Check 3 |
| `acceptance_maps.py` | the comparisons: what `show` printed about maps, drops and links, against expected values from outside the tool |
| `test_maps.py` | those comparisons tested on documents written by hand |

### The row forms

Most rows of these two checks ask in plain words: `per_ordinal = "token_end -
token_start equals the document's size in the shard's index"`. A sentence cannot
be compared with an output, so `table.FORMS` lists every form the two files use:
the keys of `given`, the keys of `expect`, and each sentence. A sentence selects
the comparison written for it. Two rows with the same keys and another sentence
are two questions (`c03.spans.per-ordinal`, `c03.records.capture`). A row whose
sentence was changed matches no form and is an error of the test, because the
comparison written for the old sentence does not answer the new one. Every other
expected field is a value and is compared.

The rows of Check 6 whose `expect` holds an `answer` are put as the rows of Checks 5
and 7 are.
`never` lists answers a row must not get.

`expect_under = { F1 = { applies = false } }` takes a row out of its check while
that fallback is in force; the row is then skipped with the fallback. No fallback
is in force. The other expectations under F1 (`map_built_by`) are not read yet: if
F1 comes into force, their rows stop with a `TableError` until they are.

### One test per size

A row that names a generated set (`set = "100k"`, `set = "generated"`) is asked of
a set the test generates with `fixture/plant/generate.py`, builds through the
replica and counts with `fixture/plant/count.py` on the chunk files of that very
build. It gets one test per size, named `<row id> [<size>]`:

| Size | Records | Dedup | Asked by | Markers |
|---|---|---|---|---|
| `small` | 800 | off | Checks 3 and 6 | `megatron` (none for `c03.precondition.pair-kinds`) |
| `10k` | 10,000 | on | Check 6 | `slow`, `megatron` |
| `100k` | 100,000 | off | Checks 3 and 6 | `slow`, `megatron` |

`small` is this folder's own size. It is there so that the comparisons can be seen
to work in one run of a few minutes; the plan's answer to a check is the answer on the sizes
it names. Every set is built in the round-robin layout with 20 chunks. A row that
is asked of a store of the table, or of the spec, gets one test as before.

The sizes these rows have run on. On the tree of commit 5742d17, the 13 rows of
Check 3 asked of a generated set have run on the `small` and the `100k` sets, and
the 4 rows of Check 6 asked of a generated set on the `small` and the `10k` sets.
On the tree of commit 7960345, Check 6's four rows were run once on the 100k set
and stopped at the run's 50-minute cap without a result. Check 6 takes its answers
for the generated sets, which plan 5.1 gives as 10,000 and 100,000 records, so its
answer on the 100k set is still owed.

The summary counts a row once however many sizes it is asked of, and gives the
outcomes size by size:

```
Check 3: 15 rows
  asked once, of a store of the table or of the spec: 2 rows; ran 1, skipped 1
    ...
  asked of the generated set small (800 records): 13 rows; ran 1, skipped 12
    ...
```

When a run leaves a marker out, the rows whose tests carry it are not in the run.
The header then says how many of the check's rows were not selected.

### Where the expected values come from

Never from the tool (plan W5).

| Rows | Expected value | Read from |
|---|---|---|
| `c03.precondition.eod-convention` | the convention is recorded | the fixture spec, `tokenizer.eod_in_span`; the tool is not asked |
| `c03.precondition.pair-kinds` | pairs of each kind | `check3.pairs.counts` of the count; the tool is not asked |
| `c03.spans.*` | the size of each document | the shard's own `.idx` file, read with the replica's reader |
| `c03.drops.sanitizer`, `c06.generated.drops` | one drop per skipped line, its reason, its shard | `check3.drops` of the count (placed from the sanitizer-input files); the replica's sanitizer log must agree, and names the place of a dropped line |
| `c03.drops.no-tokens`, `c06.generated.drops` | no drop with the reason "no tokens" | the sets plant no such line (docs/plant.md section 8) |
| `c03.records.*`, `c06.generated.documents-mapped` | the planted record of each ordinal | `documents` of the count; for Check 3 every chunk-file line is also tokenized with Megatron's own tokenizer and held against the shard's tokens, read with Megatron's own reader (`oracle.megatron_data`) |
| `c03.pairs.*` | the two lines of each pair, same shard or split | `check3.pairs.list` of the count |
| `c03.verification-failed`, `c03.map-fields`, `c03.drops.no-tokens.planted-line`, the fixture rows of Check 6 | the values in the row | the hand-written table |
| `c06.generated.candidates` | the rows with candidates and their sources | `check6.candidates` of the count |
| `c06.generated.downstream-of-translation` | what holds a translation | `check6.translations` of the count |

A line of a registered file is named by the tool as `artifact:L<n>`. Which record
that line holds is read here, from the file at the path `show` prints for the
artifact: the id the replica wrote on the line. The tool is never given the ids.

### What show must print for Checks 3 and 6

Three things a row needs, and the forms the comparisons are written against. A row
that needs one that `show` does not print is skipped with the need as its reason
(`acceptance_probe.show_gaps` finds out, run by run; see "What is not built yet").
The forms:

1. **The rows of a named map.** `show SHARD --map NAMESPACE/DERIVATION` prints
   `{"artifact", "instance", "kind", "map": {"namespace", "derivation",
   "verification", "label", "declared_count", "append_eod", "eod_in_span"},
   "rows": [...]}`. Row n of `rows` is the document `D<n>`:
   `{"record": "D<n>", "label", "token_start", "token_end", "parent", "candidates"}`,
   with `parent` and each entry of `candidates` as `show --record D<n>` prints
   them today (`{"artifact", "instance", "record": "L<n>", ...}`). A map the shard
   does not hold is refused with a named error, and so is an artifact that is no
   shard ("usage" is the error of a flag the command does not know, and the probe
   takes it for the flag being absent). Before this form, `show --record D<n>`
   printed the row of the map a reader gets first, so the rows of a recovery map
   beside a capture map could not be read at all, and one document per command
   line is hours on the 100k set.
2. **The drops recorded under an artifact.** `show ARTIFACT` prints `drops`, a
   list with one entry per drop whose output is the artifact:
   `{"artifact", "instance", "record": "L<n>", "reason", "step"}`, where
   `artifact` and `record` name the dropped line. An artifact with none prints an
   empty list. Before this form a drop was printed only under the dropped record,
   so which artifact a drop was recorded under could not be read, and neither
   could "exactly once".
3. **The append-eod setting of a map.** Every entry of `bindings` in `show
   ARTIFACT` carries `append_eod` and `eod_in_span`: true or false for a
   tokenization map, null for a binding that is no map.

The three rows `c06.fixture.documents-mapped.*` do not wait for the first of
these. Each reads one shard of at most 64 documents, document by document, when
the map it names is the one a reader gets first.

**The import's rows.** The parent left unresolved and the missing link are fields
of the answer (plan 2.1, L1): `c06.import.parent-unresolved` asks the answer for
the request narrowed to the record, and compares `answer`, `failing` and
`candidates` as every answer row does. The action `unresolved` and the scope that
needs review are fields of the rebuild report (plan 4.1): `c06.import.review-scope`
is a row of the report and waits for it, with a skipped test of its own, as the
actions of Check 7 do. Check 6 is answered yes only when that row has run too.

### Checks 7 and 9, and the rebuild

A repair is three commands, run on the documented command lines and never through the Python
seam of docs/interfaces.md 7.4 (the replica's own Python names are not fixed):

1. `waterwhite report withdraw A --store HISTORICAL`, its output kept as a file;
2. `python -m replica repair --report FILE --historical DIR --out DIR --store COPY ...` on a
   COPY of the store's folder, so that the stores the other rows read never hold a successor
   (the listings of Check 5 must not meet one);
3. `waterwhite verify --store COPY --report FILE --sides FILE`.

A row about a successor, or the state `r.successor-registered`, is put to the copy
(`World.copy(name, "r.successor-registered")`); every other row is put to the store as the
replica built it. A report row is read from the report's `records` and `artifacts`: a kept
artifact lists no records, so a record of it is kept through its artifact's entry; a missing
entry, a missing key or a null is a difference and never a pass. The row X of a successor is
found in the successor's file by its text, because the docs fix no manifest of a successor.

The two wrong reports of Check 9 (`r.descendant-kept`, `r.unrelated-paraphrase-retired`) are
the report file with the `action` of one entry changed before the repair
(`acceptance_repair.edited_report`); a report that lacks the entry is an error of the run.

The multiset comparison of the 10k rows is made twice and each is held to the table: this
folder's own multiset (the SHA-256 of each line's text, never the tool's normaliser) of the
files the sides file names, and the tool's own word `multisets`. The per-shard digests are
the same: this folder's SHA-256 over a shard's two files, and the tool's `per_shard_digests`.

The rows of the 10k set (ten rows, 18 tests) are behind the long-run gate and are put through
`SetRepairs`; they run only when `WATERWHITE_CLEARED_LONG_RUNS` names `9:10k`. What the docs
leave to the replica (the name prefix of a generated set's shards in the partitioned layout,
how `repair --set` is handed the clean folder) is read from what the replica printed and
from the report's own entries, not assumed.

### What is told when a row is put

Two things are not in `NOT_BUILT`, because the replica says them itself (and
`show` is not in it either, because the probe says it):

- **A store the replica does not build.** `World.store` asks
  `replica.fixture.TABLE_STORES_NOT_BUILT`. A row of a store listed there is
  skipped with the replica's own reason. When the replica builds the store, the
  rows run with no change here. The stores `rr`, `no-capture`, `scratch-500` and
  `scratch-notokens` hold shards that Megatron tokenizes; they are built with
  `build_table_store(folder, name, tokenize=True, parallel=3)` and need the
  Megatron environment. `build_table_store` tokenizes nothing by default, so the
  store `main` is built without shards here; the shard rows of Check 5 that ask it
  stay skipped until a row's store is built with `tokenize=True`.
- **A store with no shard.** A row about a shard is skipped with `shard` when the
  build of its store registered none. For a generated set the reason names the
  replica's hook point. The store `main` is still built without tokenization, so
  its shard rows of Check 5 are skipped as before.

### The set-ups of Check 3

A set-up is not a question, so it need not go through the command line, which has
a `recover` command (docs/cli.md). Three set-ups of
README section 7 are applied by `acceptance_sets.py` to a copy of the capture
store of a set, through the public calls the command wraps
(`waterwhite.tokenmap.recover_map`, `waterwhite.recovery.register_found_file`).
One process then holds the team's tokenizer for every shard, and it is the same
tokenizer object that gives the expected values, so a set-up and the values it is
compared with cannot come from two tokenizers:

| State | What is done |
|---|---|
| the three maps side by side | `recover_map` on every shard from its registered `<shard>.input`, with `also_token_matching=True` |
| `t.twin-swap` | in every shard that holds both members of a pair, the pair's two lines change places; the changed file is registered as `<shard>.input.swapped` and the position-mode recovery is run on it |
| `t.verification-fails` | line 1 of the first shard's input is replaced; the changed file is registered as `<shard>.input.changed` and both recoveries are run on it |

A copy is the store file in a folder of its own with the build folder linked
beside it, so a 100k build is not copied.

The set made without capture is not recovered here. The recipe (which file a drop
is recorded under, which files are registered) is the replica's, so that a set and
the table's store `no-capture` follow one rule:
`replica.fixture.build_set_store(folder, set_folder, "no-capture", layout=...,
tokenizer_spec=..., parallel=...)`, returning what `build_table_store` returns.
Where the replica has no such function, the four generated rows of Check 6 are
skipped with that name.

### A line that gives no tokens

The generated sets and the fixture plant no surviving line without tokens, so the
rows `c03.drops.no-tokens`, `c06.generated.drops` and `c06.fixture.drops` expect
zero such drops. They fail for a tool that reports one where there is none. They
cannot fail for a tool that misses one.

The planted case is one row, `c03.drops.no-tokens.planted-line`, on the store
`scratch-notokens`. It runs as soon as the replica builds the store (it needs the
Megatron environment) and `show` prints drops; it is not skipped for a reason of
its own. The row does not take the planted line's place from the table: Megatron's
own tokenizer is run on every line of the input the replica registered, and the
shard that Megatron's preprocessing wrote from that input is read with the
replica's reader of its index. The line gives no tokens when the tokenizer's answer
is empty and the index holds no tokens at its place; every other line gives tokens,
and the index holds as many as the tokenizer gave, with the end-of-document token.
The row also counts the drops with the reason "no tokens" under the shard, and the
count must be the one the row names (one). It names the capture map and the
position-mode recovery map of one shard in a store built with capture. The store keeps one drop per dropped
line, step and output, so the two maps share one drop record: the recovery side
of the row can fail in its ordinals, its document count, its verification and its
labels, and not in the drop. No row puts such a line into a store made without
capture. Check 6's question about lines that give no tokens can therefore fail in
these tests only as a false report.

### Running Checks 3 and 6

```
python -m pytest tests/acceptance -q -rs -m "not slow and not megatron" \
    --basetemp=.work/pytest-acceptance
```

runs what needs no Megatron: Checks 5 and 7, the import rows of Check 6, and the
two preconditions of Check 3 (the row on the spec and the row on the plant list). With the Megatron environment, an interpreter that
has Megatron, and `WATERWHITE_MEGATRON_ENV`, `HF_HOME` and `HF_HUB_OFFLINE=1` set:

```
python -m pytest tests/acceptance/test_check03.py -q -rs -m "not slow" \
    --basetemp=.work/pytest-acceptance-03
python -m pytest tests/acceptance/test_check06.py -q -rs -m "not slow" \
    --basetemp=.work/pytest-acceptance-06
python -m pytest tests/acceptance/test_check05.py tests/acceptance/test_check07.py \
    tests/acceptance/test_faults.py -q -rs -m "not slow" --basetemp=.work/pytest-acceptance-57
python -m pytest tests/acceptance/test_check03.py tests/acceptance/test_check06.py \
    -q -rs -m slow --basetemp=.work/pytest-acceptance-sets
```

The first three add the small set and the tokenized stores of the table: on the
laptop the small set and its rows take about 150 seconds per check file, and each
file builds the set again in its own base folder, so one command per file keeps
each run short. The last is the 10k and the 100k set: each is generated once and
built and tokenized once per check that asks it. Run every one through the
machine's CPU lock.

A run with no `-m` selects every test of the folder, the 10k and 100k sets
among them. With `-m slow` or with no `-m`, a test of the 10k or the 100k set
still starts only when `WATERWHITE_CLEARED_LONG_RUNS` names its check and size;
otherwise it is skipped with that reason, and a skip is not a pass.

### The red run for Checks 3 and 6

The rows of Checks 3 and 6 that compare a value the table states are graded red on a
scratch copy of the table as above, one wrong expectation in each such row (nine of
Check 3 and nine of Check 6 when this was written; the others, such as
`c03.spans.per-ordinal`, `c03.records.capture`, `c03.records.position-recovery`
and `c03.pairs.same-shard.swap`, have no red run on the real path yet): for example, in `check03.toml`, `same_shard_pairs_at_least =
1000` for `c03.precondition.pair-kinds`, line 2 instead of line 3 for the planted
line, `"recovered"` instead of `"candidates"` for the labels of a pair in one
shard; in `check06.toml`, another `evidence` text for `c06.import.file-family` and
the source `"B"` instead of `"A"` as the candidate of the import. A wrong
expectation on a row whose target is not built is skipped, and a skip is not red:
for those rows the comparison itself is shown to fail on wrong documents in
`test_maps.py`. The probe and the gate on the sandbox are shown red in
`test_probe.py` (a command named `bwrap` that fails stands for a machine with no
sandbox).

## Checks 1, 2, 4 and 8

`test_check01.py`, `test_check02.py`, `test_check04.py` and `test_check08.py` grade
the rows of `check01.toml` (the exposure join, 19 rows), `check02.toml` (the cache
states of run E, 30 rows), `check04.toml` (store integrity, 12 rows) and
`check08.toml` (checkpoints with the evidence of run E, 13 rows). Plan sections
served: 5.5 Checks 1, 2, 4 and 8, 3.4 (the loud checks), 3.5 (the exposure join), 5.3
(the oracle). Where a row is of the same form as a row of Check 5 (a request put to an
artifact of a store in a state) it is put the same way, through the command line; the
rest are the forms below.

### The faults they add

| Fault | Target | Waits for |
|---|---|---|
| `f.map-deleted`, `f.digest`, `f.label` (Check 5) | the capture map of `E/shard-1`, `E/shard-3`, `E/shard-2`: binding and rows deleted; the first hex digit of the declared digest replaced; the label field set to `observed` | a store whose shards Megatron tokenized |
| `f.trace-deleted`, `f.trace-half` (Checks 4 and 5) | the trace of step 3 of `E/seg-1` deleted, the registered step kept; the last sample of step 4 marked 16 of 33 tokens read, the trace still counting four | the runs |
| `l.trace-count` | the last sample of the trace of step 5 deleted and its count set to 3 | the runs |
| `l.edge-parent` | the parent of the stated edge of P changed to an occurrence id no occurrence has | nothing |
| `l.artifact-digest` | one byte of one line's text of R2's file changed in the copy, never in the original (a path the store keeps that leaves the copy's folder is refused) | nothing |

A fault finds its target by the names the store keeps (`artifact_name`, the segment's
name), so the same function serves the stores the replica builds and the small store
written by hand (`mini_store.py`) that `test_faults_on_records.py` puts each fault on:
the test reads the SQLite file back and checks that exactly what the catalogue names
changed. The rows that ask a map fault run only where the store `main` is tokenized
(the Megatron environment); they skip with that reason elsewhere.

The faults of the scratch store are put on a copy of the store `scratch-500`, which the
replica builds with the shard `c04/shard-500` of 500 documents registered and no map
(`acceptance_scratch.py`, `jobs/write_scratch_map.py`):

| Set-up | What is done |
|---|---|
| `w.kill-after-100` | the shard's position-mode map is written by a process of its own that kills itself with SIGKILL as the row after the 100th of the map table goes in (the store's hook for a test that stops a writer in a transaction, `Store.set_row_hook`); a writer that was not killed is an error of the set-up. The row reads the store afterwards through `show` of the shard: no binding and no stored row, or one binding of 500 rows |
| `w.retry` | the kill above, then the same write again on the reopened store; the write is accepted and the shard then holds a binding of 500 rows with the shard's count and registered digest, or it is refused with a named mismatch and the shard holds no partial map |
| `w.declare-499` | `Store.write_map` is called in the test's process declaring 499 documents; the refusal must be "declared count mismatch", and the store holds no map and no row |
| `l.count-499` | the complete map is written first, by the writer above, then the stored count of its one binding is set to 499 in the store file; the edit must change exactly one binding |

The writer derives the map with no tokenizer, so the map is the one position mode gives
with no verification run: the rows of Check 4 ask what the store holds after the write,
not what the map says. The rows that read the store afterwards compare the table's own
sentences (`acceptance_scratch.SENTENCES`), and a sentence the comparison was not written
for stops the row with an error. `loud_stop = false` of the two rows `c04.not-a-stop.*`
means the question is answered: an exit code 3 is a mismatch of the field, whatever the
answer would have been.

### The run world

`acceptance_runs.py` builds the runs of a store of the table. It is the one place
that knows the commands:

1. the replica's run wrapper (`python -m replica build-runs`) writes one folder per
   run under a folder of its own: `<run>/<segment, slash as dash>.run.json`
   (docs/interfaces.md 8.1), `.steps.json` (8.2), `.cache/` (8.4) and the shard files;
2. `waterwhite register-run` registers one segment: the run record, its step record,
   its cache folder, its mixture and the checkpoint it starts from. The segments, the
   mixture each trains on and the order (a segment after the checkpoint it starts
   from) are the fixture spec's; `main` registers E, D, Y and K, `rr` registers E-rr.

`acceptance_runs.capture_missing` asks each command for its help once per run. Where
either does not exist the rows that need a step trace, a checkpoint or the record W of a
run are skipped with `not built yet: run capture`, and they start to
run on the commit that adds the second command, with no edit here. A command is
missing only where its parser does not know the word (a usage error naming it as an
invalid choice); a command that exists and fails any other way is an error of the
run, never a skip. The flags of the replica's run command (`--out`, `--fixture-build`,
`--spec`, `--runs`) are written in one function, `build_runs_arguments`, as docs/replica.md section 13
gives them. Both commands run with the Megatron environment's
interpreter: the wrapper imports torch, and a cyclic segment needs torch to compute
its shuffle.

The store `main` is copied before any run is registered (`before-runs.sqlite`, beside
the store, so that its relative paths hold). Check 2 registers the runs again in a
fresh copy for each starting state, never in a store that holds the traces:

| State | The run records and caches handed over |
|---|---|
| `c2.i` | no cache folder for E, D and Y; settings, seeds and digests kept |
| `c2.ii` | as `c2.i`, and the seed in the settings of E, D and Y plus 1 |
| `c2.iii` | no cache folder for E, D and Y, their replay settings (`random_seed`, `split`, `train_iters`, `train_samples`) emptied and their `cache_digests` emptied: one reading of "the settings and the recorded digests are withheld"; the batch sizes and the sampler stay, because the step record needs them |
| `c2.iv` | run K's shuffle digests removed from its run record, its cache folder kept |

A changed run record is written to a scratch folder with the shard folder linked
beside it; the replica's own file is never edited.

**The list of verified joins.** A store's list of verified joins is written when the
store is made, and nothing writes it later: by `init`, which lists the joins the tool
ships, or a subset of it that `--verified-join` names (a join outside it is refused;
docs/cli.md, "init"; D-51), and by the library with the joins it is given (a store made
so with its own list is outside what the tool verified). With an
empty list every segment registers with the route "none" and the cause "data-order trace
unverified for this version", whatever its run record holds. The replica creates the
stores that hold runs (`main`, `rr`) with the joins of those runs (docs/replica.md section
13). This folder reads the list (`acceptance_runs.verified_joins`) and never writes it: a
store that is to hold runs and lists no join is an error of the run, not a skip, because
the build that made it is at fault.

### The rows of Checks 1, 2 and 8 that are not an answer

- **A registration** (`c02.ii.replay-rejected`, `c02.iii.no-replay`): what
  `register-run` printed for each segment: a route `none` with the cause "data-order
  trace missing", the reason of the row, and the `replay` object (attempted, accepted,
  reason): attempted and not accepted for a replay whose digests differ, neither for a run
  with no recorded digest. Where the command does not print the object the row is
  skipped, by name, after everything it does print has been compared. `digests` of
  `c02.ii.replay-rejected` runs `replay` over the changed run record and asks that at least
  one array differ from its recorded digest and that `equal` say what the two digests say;
  until the command exists it is a field that waits (a skipped test of its own).
- **The replayed arrays** (the four rows `c02.i.arrays.*`): `replay --run-record PATH
  --out DIR` is run over the run record of the segment, whose cache folder the starting
  state withheld from `register-run` and which the replica kept; each array it wrote is
  compared with the kept cache file of the same kind and shard. The kept file of a
  (kind, shard) is found from the folder's own files: a dataset's `description.txt` names
  the dataset's path, whose last part is the prefix of a shard of the run record, and its
  arrays carry the same stem; the blend's arrays carry the blend's stem. An array is the
  same when its dtype, its shape and its bytes are: the plan states the comparison on the
  `.npy` index arrays, because the file names and `description.txt` carry the run's paths
  (plan 3.5, F3), and the digests of the run record are `ids.array_digest` values over
  dtype, shape and data (docs/interfaces.md 8.1), not digests of the file. The row names
  five kinds; a kind not written, an array written that the cache has no file for and a
  kept file that was not written are findings too.
- **The join's rows** (every row of Check 1 that names a run, and
  `c08.withdraw-A.E@9.step-8-holds-no-T`): `show SEGMENT --trace` of each segment the row
  names, flattened to the shape of the oracle's log (`acceptance_trace.flatten`) and
  diffed against it, which is the grading side. The oracle's log is read from a run of
  `oracle/fixture_runs.py` over the same build, started as a process of its own
  (`acceptance_runs.oracle_runs`). The comparison is `acceptance_join.diff`, a pure
  function tested on rows written by hand: a missing sample, a changed token end, a row
  the oracle has none of, the order of the rows of a sample, and the cut of the
  half-covered sample of step 8 by the spec's rule. It compares the fields the row names
  (step, shard position, ordinal, token start and end, and the place of the sample); the
  rank and the global index of a sample are printed by the trace and are graded by the
  rows that name them (`c01.K.ranks`), from the table's own values. Reading the tables of
  the store directly would grade the tool by its own file layout and is not done. The
  other fields of these rows are read from the same document: the rows of a position the
  spec records (`rows`), T whole as the second row of one sample and in no other row of run
  E, W as one row of one token in a sample of 33 tokens, the coverage mark and the
  uncovered range of the half-covered sample, the samples a step counts, the steps with an
  uncovered range or no trace, the samples and ranks of each step of run K, the steps a
  launch covers (the lines of the spec's positions are taken for the launch's steps), the
  document count of a shard (`show` of the shard) and whether `check` stops. Five fields are
  read from a form of their own (section "A form the command does not print", below):
  `consumed_at_start`, `first_step` and `last_step` of a launch from the `launches` of the trace,
  `boundary_after_step` from the `epochs` of its steps, and `skipped_line` from the `drops` that
  `show` prints. A field with no comparison at all stops the world with an error, which
  `test_join_row_fields.py` shows cannot happen for any row of the table.
  Until `show --trace` exists these rows are skipped with `not built yet: show --trace`.
- **The shuffle of run K** (`c08.K.digests-match`): reads the `shuffle` that `register-run`
  prints for `K/seg-1`: one entry for each (launch, epoch, rank) the launches read, in that
  order, each with `equal` true and its digest and count, and the digest and count it says it
  recorded, equal to the run record's own `shuffle_digests` (the replica's file, read here: a
  tool that prints its own digest as the recorded one finds no pass), and the epoch 0
  digests of launch 0 and launch 1 equal rank by rank. The cells are the spec's, not the tool's:
  the launches of the segment (launch 0 reads steps 1 to 3, launch 1 steps 4 to 40), an epoch of
  `positions.K.epoch_steps` steps and the data-parallel size give 8, the number the spec's
  `positions.K.shuffle_digests` states; the world stops with an error where the two disagree.
  Where `register-run` prints no `shuffle` the row is skipped with `not built yet: register-run's
  shuffle`. What grades the order, as against the digest, is Check 1's rows on run K
  (`c01.K.diff`, `c01.K.warm-start`, `c01.K.resume`, `c01.K.epoch-boundary`, `c01.K.ranks`): the
  join's shuffle and the replica's were written from one text of the interface, so they could
  share a wrong rule, and only the oracle's log tells them apart. The comment is beside the row's
  handling in `table.py`, because the table is not edited here.
- **The request withdraw W** (`c08.shared-token.*`): the source and the record are the
  spec's `positions.shared_token`; the row's own `record` must equal it.
- **A listing** (`c05.listing.map-deleted`): `covers` as sets of names, `not_covered`
  as a set of (artifact, reason); it names runs and checkpoints, so it needs the runs.

### A form the command does not print

Five fields of Check 1 and one row of Check 8 are read from forms of their own. `World`
compares each where the command prints it and skips the row by the form's name where it does
not (`Unavailable`, after everything else the row names was compared and agrees); a
mismatch found anywhere else in the row is a failure, never a skip.

| Field | Read from | Compared |
|---|---|---|
| `consumed_at_start`, `first_step`, `last_step` (`c01.K.warm-start`, `c01.K.resume`) | the `launches` of `show K/seg-1 --trace`: `{position, world_size, data_parallel_size, consumed_at_start, step_at_start, first_step, last_step}` in position order | the entry whose `position` is the row's `launch`; no entry at that position is a finding |
| `boundary_after_step` (`c01.K.epoch-boundary`) | the `epochs` of each step of the same trace (the epochs its samples are drawn from, in order) | the last step whose epoch is the first one; findings: a step that reads two epochs, a first epoch other than 0, an epoch that goes down or rises by more than one, a first step of the new epoch that is not read in the epoch the spec gives (`positions.K.epoch_of_that_step`) |
| `skipped_line` (`c01.E.skipped-line`) | the `drops` that `show` prints: each names the dropped line as `record` "L<n>" and gives its `reason` | exactly one drop of the row's reason ("sanitizer skip") among the drops of `E/shard-2` and of its registered inputs, and its record is `L6`, the row's `input_line` |
| `shuffle_digests` (`c08.K.digests-match`) | the `shuffle` of `register-run` for `K/seg-1` (section above) | one entry per (launch, epoch, rank) in order, each equal, its digest and count and the recorded ones equal to the run record's file, and the epoch 0 digests of the launches equal rank by rank |

`show E/shard-2` prints an empty `drops`: the sanitizer records its drop with the file it wrote as
the output, which is the shard's registered input `E/shard-2.input` (docs/cli.md, `show`: a JSONL
file carries the drops of the sanitizer that wrote it, a shard those of its tokenizer). The
row reads the shard's `inputs` and then the drops of each registered input; a drop that a later
build records under the shard itself is read too.

### What has run

Each of these is the state of this folder at the commit it was written, not a claim
about the tool beyond it. The rows that run without Megatron: `c04.edge-parent.R1`,
`c04.edge-parent.gen/P` and `c04.artifact-digest.R2` (three of Check 4's twelve rows)
pass on the store `main` built without tokenization. A row that needs a tokenized store
runs only where the Megatron environment is set (`WATERWHITE_MEGATRON_ENV`, `HF_HOME`,
`HF_HUB_OFFLINE=1`); through the machine's CPU lock:

```
python -m pytest tests/acceptance/test_check05.py tests/acceptance/test_check04.py \
    tests/acceptance/test_check08.py tests/acceptance/test_check02.py \
    tests/acceptance/test_check01.py -q -rs -m "not slow" --basetemp=.work/pytest-acceptance-run
```

On the tree these rows were first written against, with the Megatron environment set, each of
those files in one run of its own:

| Check | Rows | Ran and passed | Skipped, and why |
|---|---|---|---|
| 1 | 19 | 2 (`c01.E.step8.untraced`, `c01.E.E@9`, asked as answers) | 17: `not built yet: show --trace` |
| 2 | 30 | 24 | 4: `not built yet: waterwhite replay` (`c02.i.arrays.*`); 2: `not built yet: register-run's replay object` (`c02.ii.replay-rejected`, `c02.iii.no-replay`), after route, cause and reason agreed |
| 4 | 12 | 12, among them the five rows of the scratch store | none |
| 5 | 221 | 220 | 1: `extension off: generating-model` |
| 8 | 13 | 11 | 1: `not built yet: show --trace`; 1: `not built yet: the join's shuffle digests in the output of register-run` |

That tool had neither `show --trace` nor `replay` nor the replay object of `register-run`,
so the paths that read them (`_join_rows`, `_replayed_arrays`, `_replayed_digests` and the
replay object of `_registration`) had been put to documents written by hand only
(`test_trace.py`, `test_join_row_fields.py`, `test_registration.py`), never to a tool, and a
skipped row of Check 1 or 2 is not a pass.

On commit 987cae5, which has `show --trace`, `replay` and the replay object of `register-run`
but no `launches`, `epochs` or `shuffle`, Checks 1 and 8 were put to the real commands, each in one
run of its own with the Megatron environment set. Check 1, before the rows below were
changed to read the forms of the section above: 19 rows, 19 passed, the first time its trace
rows met the real command; its 4 skipped tests were the fields that then waited for good
(`consumed_at_start` of two rows, `skipped_line`, `boundary_after_step`). Check 8, with the
rows as they are now:

| Check | Rows | Ran and passed | Skipped, and why |
|---|---|---|---|
| 8 | 13 | 12, among them `c08.withdraw-A.E@9.step-8-holds-no-T` read from the trace | 1: `not built yet: register-run's shuffle` (`c08.K.digests-match`) |

The rows of Check 1 as they are now were put to documents written by hand
(`test_launches_epochs_shuffle.py`) and, for the skipped line, to the store the first run built:
`show E/shard-2.input` prints one drop, `L6` with the reason "sanitizer skip", which is what the
row reads. On 987cae5 `c01.K.warm-start` and `c01.K.resume` read no `launches` and
`c01.K.epoch-boundary` no `epochs` from the real trace (a read-only probe of that store), so they
skip by name there once everything else they name agrees. Checks 2, 4 and 5 were not run on that
commit for this change.

The oracle's runs were started once over the build of `main` and their logs read
(`acceptance_runs.oracle_runs`, `oracle_log`); the oracle is asked for the runs of the
store's own build only, because it refuses a build that holds one layout's shards when it is
asked for every run.
