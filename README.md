# Waterwhite

Waterwhite is a free, open-source tool for teams that train language models. When a source is withdrawn from a training corpus, it tells the team which mixtures need a successor, which records a successor can keep, which checkpoints have evidence of exposure, and what has to be rebuilt to remove the source and what is derived from it through the recorded steps, such as its translations. What "derived" reaches, and what it does not, is stated under "What the answers are exact about".

## What the answers are exact about

The answers are exact about the evidence that was recorded and about the trust assumptions printed beside each answer. They are not exact about the world. Three kinds of statement are kept apart:

- **Declared ancestry.** The parents a step states for an output, such as the records a generation job says it used. The tool stores such a list faithfully. It does not compare it with what the job read, and storing it is not a check of it.
- **Observed reads.** What a harness logged as the job read. The in-process harness logs only the reads made through it. On Linux, the confined harness uses bubblewrap to restrict which files the job can open. `docs/confine.md` describes the restrictions and the conclusions they support. A job is handed a text with each reference, and the store compares each text with the record the reference names when the job's output is registered, and refuses the registration when one is not that record. It makes that comparison for every text it is given, on the edge or on its log: those of the result a job run returns, registered in the same process, and those of an edge or a log built by hand that carries them; a record that holds no text takes only a text that is empty after the normaliser. A log registered without a text for every reference it handed (read back from its file, fetched from the store, or built by hand without its texts) is stored as not compared: it establishes no bound, so a "does not contain" past it reads "may contain" until the original texts are shown and compared (`waterwhite compare-hand-off`; `docs/capture.md` lists the routes). Stores made before this are migrated to schema version 4 and every log in them reads as not compared.
- **A proof of absence.** "Does not contain" is printed only when a coverage proof backs it, and the proof's scope (the source, the members, the steps and the logs it rests on) is printed with it.

Where evidence was not recorded, the tool says what it cannot know and what it answers instead.

The default withdrawal follows recorded ancestry: the inputs and parents of records, and the steps of a run. While the generating-model switch is off (the default), a generating model is neither an input, a parent nor an ancestor of what it generates, whatever model identifier a log holds, and a "does not contain" past a generation edge says "generating-model exposure not assessed". With the switch on, each such edge's model is read from its stored log and must be "none" or a registered checkpoint that itself answers "does not contain"; otherwise the answer is "may contain" naming the model. The off state is a decision about the core's scope. It does not say that every output of such a model carries the source.

The tool runs on your own machines and keeps its evidence store there. It makes no claim of legal compliance, and it makes no claim about cost or effect beyond what has been measured on a named workload.

The store protects against accidental edits and inconsistent imports: its loud checks compare what it records with other records and with the files it names, and its write methods never overwrite recorded history. Waterwhite does not defend against direct edits of its database: a store whose rows were edited outside the tool is read as stored. Its loud checks find some inconsistent edits, not all, and a check that finds nothing is not proof that the file was not edited.

## Status

**After the first commit (20 November 2026).** Later commits fix what an outside review of the first commit's tree found: a generation edge that left out a record its job was handed could support "does not contain"; `verify --require-held` could hold for a successor of the wrong kind, and (found while fixing that) for a report that stated a kind the store does not hold; a shard whose stored byte offsets disagreed with its lengths was accepted although Megatron reads at the stored offsets; and a number as a path in a sides file raised an uncaught error. They also make the tests run on a clean checkout: the test files no longer share a name, tests that need the Megatron environment, torch or transformers are deselected or skipped with a stated reason, and a workflow builds the wheel, installs it in a fresh environment and runs the quickstart. The Check 10 table, the measured figures and the core result were recorded on earlier trees, not on the trees of these changes.

This is a first build. The core is built and tested against a hand-checked fixture and a pipeline replica; its checks, their expected answers and the open items are in `docs/`, `DECISIONS.md` and `BUILT-LOG.md`.

- **Check 10 has been run.** It puts the same questions to six subjects and records every answer: two existing tools (the Dolma toolkit and OriginBlame), two reference points built here (a file-level manifest and a positions-only step trace), and the two combinations of an existing tool with that trace. On the tree of this repository's first commit, each of the six subjects received the same 400 questions (362 asked once and 38 asked at a size), including the questions about generating models. Check 10 recorded 2,400 responses. Each was marked right, wrong or "cannot answer". The answers are not in this repository; the table under "Check 10 results" below counts the questions asked once, and its row of Waterwhite is from Waterwhite's own records.
- Six extensions are designed. The generating-model rule is built and off by default: switched on in the store, it judges a negative past a generation edge by the edge's model, and a query whose answer depends on a temporal ancestry cycle through a model's own history is refused with "temporal ancestry cycle"; the store does not refuse the cycle when it is written (see Known limits). The build does not implement the content ban, flagged records, published models, exposure counts or adapters. A ban or flagged-record request is refused with "extension off" when its own switch is off. With the published-models switch off, a query naming a published model held in the store returns "not covered". Each of those requests is refused as not built when its own switch is on. No code reads the switches of exposure counts and adapters. Switching an extension on turns on only what is built. Section 1.4 of the plan (`docs/WATERWHITE_BUILD_PLAN.markdown`) says where the rule of each of the five that are not built is, and names its check or the check it would need.
- The tool's name and its first-build scope are the owner's decisions.

### Check 10 results

The table counts the 362 questions that were asked once; 38 more were asked at a size, and their counts, the grade words, the tool versions and the answer file are in `docs/baselines.md`, "Check 10 results". For the trace and the two combinations the table leaves 49 of the 362 out of the tally (`excluded_from_tally`), so their count is 313. "Questions" is the number of questions in the tally, "Answered" the number a subject gave a right or a wrong answer to, and "Right" how many of those are right. A question that a subject could not answer is in neither count.

| Subject | Questions | Answered | Right |
|---|---|---|---|
| Waterwhite | 362 | 362 | 362 |
| Dolma toolkit | 362 | 18 | 17 |
| OriginBlame | 362 | 25 | 14 |
| File-level manifest | 362 | 138 | 102 |
| Positions-only step trace | 313 | 1 | 1 |
| Dolma toolkit with the trace | 313 | 48 | 33 |
| OriginBlame with the trace | 313 | 80 | 27 |

The row of Waterwhite is from its own records, not from the Check 10 run: the 353 core rows are from the committed core result (recorded on the tree `1c132e32`, compared equal on the tree `c113fd44`), and the 9 generating-model rows are from the test run on the tree of this repository's first commit. Neither tree `1c132e32` nor `c113fd44` is in this repository's history. The other rows are from the Check 10 run. The questions are Waterwhite's own. For the core rows, `tests/acceptance/README.md` says that the table was written by hand before the tool existed and that a row is never changed to fit an answer.

## Measured costs

On one server, the replica built 20 workloads generated with one seed. In each capture-off run, it built the workload without creating an evidence store. In each capture-on run, it also recorded evidence in a Waterwhite store. The store, metadata, query, proof walk and repair figures come from the capture-on runs. The machine: type `CPX62-shared-vCPU` (the name the caller gave the runner: a cloud server with 16 shared virtual CPUs), 32,853,516,288 bytes of memory (about 32.9 GB), Ubuntu 24.04.4 LTS, role "server". The tables give two of the 20 workloads, one of 10,000 and one of 100,000 source records. Both tables show 2,000 characters per source record, one derivation, four sources and a partitioned layout. This was the only combination measured at both 10,000 and 100,000 source records. The larger set had a higher measured build rate in each capture mode at each worker setting. The timings do not identify the reason. `docs/measurements.md` gives all 20 workloads. The peaks come from a second pass of the same runs with the sampler on. Throughput is the number of source records in the workload per second of elapsed `build-set` time. A throughput, a query time and a proof walk time is the median of three runs; a peak memory is the greatest of three; a byte figure is the number the runs wrote, or the lowest and highest of the three when they differ.

There is one table for each worker setting. In the timed and memory passes, the runner launched one-worker builds in groups of six and four-worker builds in groups of three. The worker count and the number of builds launched together both changed. No workload ran alone as a reference. The tables are not a comparison of the two settings.

One worker, builds launched in groups of six:

| figure | 10,000 records | 100,000 records |
|---|---|---|
| throughput, capture off (source records/s) | 315.6 | 1,562.6 |
| throughput, capture on (source records/s) | 192.4 | 446.6 |
| peak memory, capture off (MB) | 1,412.2 | 1,989.1 |
| peak memory, capture on (MB) | 1,422.8 | 2,017.1 |
| build folder, capture off (bytes) | 76,969,988 | 786,182,847 to 786,182,849 |
| build folder, capture on (bytes) | 77,686,377 | 793,354,718 to 793,354,728 |
| store file (bytes) | 11,870,208 | 125,243,392 |
| metadata per record (bytes) | 77.33 | 81.72 |
| metadata per parent link (bytes) | 27.57 | 30.90 |
| metadata per stored step trace (bytes) | 268.43 | 286.03 |
| query time (s) | 1.90 | 101.27 |
| proof walk time (s) | 2.11 | 117.50 |
| repair bytes counted as read (bytes) | 89,556,585 | 918,598,110 to 918,598,120 |
| repair bytes counted as rewritten (bytes) | 70,628,518 to 70,628,520 | 724,030,273 to 724,030,277 |

Four workers, builds launched in groups of three:

| figure | 10,000 records | 100,000 records |
|---|---|---|
| throughput, capture off (source records/s) | 804.8 | 2,748.4 |
| throughput, capture on (source records/s) | 297.7 | 522.4 |
| peak memory, capture off (MB) | 5,323.1 | 5,912.1 |
| peak memory, capture on (MB) | 5,342.3 | 5,934.5 |
| build folder, capture off (bytes) | 76,969,984 to 76,969,987 | 786,182,835 to 786,182,843 |
| build folder, capture on (bytes) | 77,686,374 to 77,686,377 | 793,354,717 to 793,354,728 |
| store file (bytes) | 11,870,208 | 125,243,392 |
| metadata per record (bytes) | 77.33 | 81.72 |
| metadata per parent link (bytes) | 27.57 | 30.90 |
| metadata per stored step trace (bytes) | 268.43 | 286.03 |
| query time (s) | 1.84 | 101.46 |
| proof walk time (s) | 2.11 | 116.53 |
| repair bytes counted as read (bytes) | 89,556,582 to 89,556,585 | 918,598,109 to 918,598,120 |
| repair bytes counted as rewritten (bytes) | 70,628,517 to 70,628,521 | 724,030,276 to 724,030,281 |

The build-folder total excludes the store file. Each metadata figure is divided over the stored rows of its kind; for 10,000 source records the store holds 29,398 record rows. The replica runs the repair on a copy of the store. It counts the historical build size and the starting store size as bytes read. It counts the repaired build size and the increase in store size as bytes rewritten.

Each figure holds for its workload on this machine under the conditions listed in `docs/measurements.md`, and for nothing else: not for another machine, another corpus or another size.

## Known limits

What is not built or not checked, and what you get instead.

- **The store can hold a cycle through a model's own history.** With the generating-model switch on, a registered checkpoint whose own history reads a record that its own output feeds makes a cycle, which is not valid evidence. The store does not refuse such a record at write time, so any way of writing to it can leave one. With the generating-model switch on, a recorded "contains" for the same artifact takes precedence. If the artifact's answer depends on the cycle, the query exits 2 with "temporal ancestry cycle" and prints no answer. The listing and `report` stop at the first artifact whose answer depends on a cycle. On that refusal, `report` exits 2, writes the error to standard error and produces no report on standard output.
- **The generating-model rule is not paired with the content ban or flagged records, because neither is built.** The design requires that a ban's negative past a generation edge rest on the model answering "does not contain" for the same ban, and that an unmatched ban or an incomplete flagged request name its own link, never the model. Until they are built, the build checks neither.
- **A model that is itself a published model is not handled by a branch of its own.** With the published-models switch off, a query that names one answers "not covered". With the generating-model switch on, a generation edge whose model is a published model is treated like any model text that is not a registered checkpoint: the answer is "may contain", naming the model as not registered. A published model is never to be clean on its own, and that rule has no branch in the build.
- **One branch of the world builder's tests has not run on any machine.** The tests that compare worlds built with different worker counts build a world of one worker and one of four on a machine with less than 24 GB of memory or fewer than eight cores. On a larger machine they also build one world with a worker for each core. That branch has not been run anywhere. It belongs to the tests of the Check 10 world builder, not to the tool.
- **The plan is in this repository, and 822 of the 8,891 citations of it have no target.** The build plan is `docs/WATERWHITE_BUILD_PLAN.markdown`. It is the build copy of the Council's signed plan, with the changes made after the debate: each is marked `[BUILD COPY: <id>]` and listed in its ledger, and its header says that it is not the plan the Council signed. The files that its header and ledger name as sources (review notes, briefs and research notes) are not in this repository. `DECISIONS.md` (D-M6-1 to D-M6-3) records the open objections and notes handed over at the close-out and the reading of Sections 11 and 12 of the plan against the built tool. Code, tests and docs cite the plan by section number. `tools/check_plan_citations.py`, given the plan's path, lists every plan citation and every reference to a `DECISIONS.md` entry in the tracked files and checks that each target exists, in the plan or in `DECISIONS.md`; on this tree 822 of the 8,891 plan citations and 0 of the 793 `DECISIONS.md` references have no target, and the tool exits 1. It does not check that a target says what the citing line claims.
- **The trees named in the committed core result are not in this history.** The header of `tests/acceptance/core_result/core_result.jsonl.gz` and the page `tests/acceptance/core_result/README.md` name the trees that the file's records were made on. None of those trees is in this repository's history, so they cannot be fetched from it.

## Install

You need Python 3.12 or newer and numpy 2 or newer; `pip` installs numpy as the one required dependency. Waterwhite was built and tried on Fedora 44, where the tests used CPython 3.12 with the Megatron environment and, without it, CPython 3.14. Tests also ran on Ubuntu 24.04 servers. The measurement batch ran on Ubuntu 24.04.4 LTS. These are the operating systems the project has tried. No support for any other operating system is claimed. See Known limits for the test branch that uses one worker per core. From a checkout of this repository:

```
python -m pip install .
```

This line, and the wheel built from the checkout, were run in a fresh virtual environment on one laptop, on Python 3.12 and on 3.14, and the commands of `docs/quickstart.md` were run there against the installed wheel. Nothing is claimed for a hosted runner. A test run uses the checkout's own sources unless `WATERWHITE_EXPECT_INSTALLED` is set, as `.github/workflows/tests.yml` sets it.

Optional parts, each needed only for what it names:

- `python -m pip install ".[cyclic]"` adds torch. It is needed only to register a run that was read with the cyclic sampler, whose shuffle the join computes.
- `python -m pip install ".[test]"` adds pytest.
- The confined harness needs Linux and bubblewrap (`docs/confine.md`). The pipeline replica and the oracle need a Megatron environment built by `tools/setup_megatron_env.sh`. Neither is needed to record evidence with the capture calls or to ask a question of a store.

## Use

```
waterwhite <command> --store <file> [--format json|text] ...
```

The commands are `init`, `query`, `show`, `check`, `migrate`, `register-run`, `compare-hand-off`, `replay`, `recover`, `report` and `verify`. Every command prints one JSON document (or a few lines of text with `--format text`); an error prints one JSON object on standard error. `docs/cli.md` describes each command, its exit codes and the form of every error.

`docs/quickstart.md` is one small path from an empty folder to an answer: make a store, record a tiny pipeline with the capture calls, check the store, and ask what a withdrawn record reaches. The commands it shows are run by a test and their output is compared with what the page states.

A script that gates on `verify` reads `docs/cli.md`, under "verify", before it relies on the command's exit status. By default `verify` exits 0 whenever it made the verification, even when the two sides differ; `--require-held` makes it exit 5 when the conditions listed there are not all held, and the document is printed either way.

## Licence

MIT. See `LICENSE`.
