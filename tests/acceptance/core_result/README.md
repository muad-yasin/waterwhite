# The core result

`core_result.jsonl.gz` is the core result of `fixture/table/README.md` ("Core result"): for every core
row of Checks 1 to 9, what the tool gave to the row's question in a run with every extension switched off.
It is an output of the tool. It is never where an expected answer comes from; the expected answers are the
rows of the table (plan W5).

## What the file holds

- Gzip, written so that the same records give the same bytes (no file name, no time in the header).
  The sha256 of the text before compression and of the gzip file are in the commit message of the commit
  that added it.
- A header line (the format, the number of records, the tree it was recorded on, the machine, the sources
  and their sha256, the keys that were replaced, the twelve unseeded rows) and 400 records, one per row,
  part and size, in canonical JSON. 391 are rows of the core (`core = true`); 9 are rows of extensions
  (`core = false`) that were skipped because their extension is off: they are listed and never compared.
- One record per core row holds how the row ended, the reason it was skipped, and each program the helpers
  started while the row was put: the tool's command line with its whole standard output, and for any other
  program its exit code and the digest of its output. It holds what the tool gave, not the fields the row
  expects.
- The records were made by `core_record.py` (the recorder) on a test box, on an earlier state of the sources than this tree
  (the header names its tree under `tree`), except two: the records of `c01.E.diff` and `c01.E-rr.diff` are from
  the recording of a later tree (the header names it under `amended_records`), because the third call of each, `oracle/fixture_runs.py`, prints the
  sha256 of `fixture/spec/fixture_spec.toml`, which the R3 release added to the spec changed. The long-run
  rows (Check 3 on the 100k set, Check 6 on the 10k and 100k sets, Check 9 on the 10k set: 31 records) are
  skipped in a short run; they were recorded apart, in two further runs on the same tree, and the file holds
  those records in place of the skipped ones. `core_result_file.py combine` made the file, and refuses where
  it would have to choose: a skipped long-run record that no run ran, or that two ran; a record of a run
  that is not in the first file; a record that differs from the one kept.
- Twelve rows are asked of a store built unseeded (every shuffle drawn from the operating system): their
  ids, digests and spans differ with every build. They are compared by how the row ended, the reason of a
  skip, whether a program was started and the distinct exit codes of the programs; the number of programs
  is not compared. The other 379 core rows are compared in full.

## What the file lacks

`python3 -I tests/acceptance/core_result_file.py lacking core_result.jsonl.gz` names it: 21 core rows
started no program, so they hold the way they ended and no call. No core row is skipped in the file.

## How a later run is compared

    python3 -I tests/acceptance/core_result_file.py diff core_result.jsonl.gz LATER.jsonl

The comparison reads the core rows only; a core row missing on either side is a difference; the rows of
extensions are listed and never compared. Three verdicts:

- `FULL` (exit 0): no difference, and every core row ran. Only this verdict says that the later run gave the
  committed core result.
- `PARTIAL` (exit 3): no difference, but some of the 31 long-run rows were not run (a short run skips them
  by their own reason). It prints the rows compared and the rows not run, by id, and never the words of the
  full verdict. A key may be "not run here" only if the file's header names it as replaced and the later run
  skipped it with the long-run reason; every other skip, key or missing row is a difference.
- `DIFFERENCES` (exit 1): the keys that differ, and in which part of the record.

Plan line 370 says that an extension counts as built only when a run with every extension off gives exactly
the committed core result. A short run during the work is partial. An extension is claimed built only after a
FULL run, long rows included.

## What this tree is, against the tree of the records

The records were made on an earlier state of the sources: the header names its tree under `tree`, and two records come from a later tree (see above). This tree, the one this page is in, differs from the tree of the records in 84 files: 12 under `src/`, 4 under `replica/`, 15 under `baselines/`, 6 under `fixture/`, 35 under `tests/`, 10 under `docs/`, and `README.md` and `DECISIONS.md`.

A run of the core rows was compared with the file in this folder by `core_result_file.py diff` on a tree named `c113fd44`, an earlier state of the sources than this tree. The verdict was FULL: no difference in the core rows, 391 compared, 0 not run. That comparison was not repeated on this tree. This tree differs from `c113fd44` in 13 files (3 under `baselines/`, 4 under `tests/`, 4 under `docs/`, and `README.md` and `DECISIONS.md`). They are the code of `baselines/`, tests under `tests/baselines/`, pages of the documents and this page. None is under `src/`, `replica/`, `oracle/` or `fixture/`, and no Python file under `tests/acceptance/`, where the core rows are made, differs. The file in this folder is the same in both trees. Neither the tree of the records nor `c113fd44` is in this repository's history.

For a later run, compare the core rows with `core_result_file.py diff`. When the rows compared all match, the verdict is PARTIAL if any core rows were not run here and FULL if every core row ran.
