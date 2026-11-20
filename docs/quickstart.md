# Quickstart: from an empty folder to an answer

This page records a tiny pipeline and asks one question of it. A source file `A.jsonl` holds three
records. A step called `keep` copies the first and the third into a release file. Then one source record
is withdrawn, twice: once a record the release holds, once a record it does not.

The commands below are run by `tests/docs/test_quickstart_example.py` in a temporary folder, through the
real command line and the real capture calls. The test compares each output with the lines this page
states under the command. The install line is the one thing it does not run.

## What you need

- Python 3.12 or newer, and numpy 2 or newer (installed with the package).
- Nothing else for this page. The optional parts, the cyclic sampler's torch, the confined harness's
  bubblewrap and the Megatron environment of the replica, are listed in the README and are not used here.

Install from a checkout of the repository:

```sh
python -m pip install .
```

That gives the command `waterwhite`. `python -m waterwhite.cli` runs the same program.

## 1. Make a store

The store is one SQLite file. `init` makes it and writes the name of the normaliser, the function a
record's text passes before it is fingerprinted. There is no default normaliser. `init` prints the path
it made, the schema version, the normaliser and the joins the store lists as verified; this page does not
state that output.

```console
$ waterwhite init --store evidence.db --normaliser control-chars/1
```

## 2. Record the pipeline

There is no command that ingests a file: a pipeline calls the capture functions where it reads a source
file and where a step writes one (docs/capture.md, section 2). The script below stands in for such a
pipeline. It writes `A.jsonl`, ingests it as the source `A`, member `A/r1`, and registers
`release.jsonl` with one stated copy edge per line: line 1 of the release came from line 1 of `A.jsonl`,
line 2 from line 3.

```python
# build_evidence.py
import json

from waterwhite import capture
from waterwhite.store import Store

texts = ["first record", "second record", "third record"]
with open("A.jsonl", "w", encoding="utf-8") as out:
    for text in texts:
        out.write(json.dumps({"text": text}) + "\n")

with Store.open("evidence.db") as store:
    src_a = capture.ingest_jsonl(store, "A.jsonl", source="A", member="A/r1",
                                 revision="r1", name="src/A")

    step = capture.step_ref("keep", "1", {"keep": [1, 3]})
    with open("release.jsonl", "w", encoding="utf-8") as out:
        for text in (texts[0], texts[2]):
            out.write(json.dumps({"text": text}) + "\n")
    edges = [capture.copy_edge(1, capture.occurrence(src_a, 1), step),
             capture.copy_edge(2, capture.occurrence(src_a, 3), step)]
    capture.record_jsonl(store, "release.jsonl", name="release", role="release",
                         step=step, inputs=[src_a], edges=edges)
```

Run it in the folder that holds `evidence.db`. It prints nothing.

```console
$ python build_evidence.py
```

The store holds two files now, `src/A` and `release`, five records (three and two), and two edges. It
holds locators, fingerprints and links, never the text of a record.

## 3. Check the store

`check` runs the loud checks over the whole store: counts, digests, parents that must exist, and the
files on disk against their registered digests.

```console
$ waterwhite check --store evidence.db --format text
check: passed
artifacts: 2
missing_files: none
shards_not_passed: none
shard_records: unchanged
```

## 4. Ask what a withdrawn record reaches

`--record` narrows the withdrawal of source `A` to one record: a file name and a line. `--artifact` asks
about one artifact. Line 3 of `src/A` is in the release, as its line 2:

```console
$ waterwhite query withdraw A --record src/A:L3 --artifact release --store evidence.db --format text
release (jsonl) under withdraw A (members: A/r1), records: src/A:L3
  contains
  chain: release:L2 <- src/A:L3
```

The chain runs from the artifact to the source record. Line 2 of `src/A` is not in the release:

```console
$ waterwhite query withdraw A --record src/A:L2 --artifact release --store evidence.db --format text
release (jsonl) under withdraw A (members: A/r1), records: src/A:L2
  does not contain
  scope: source A, members A/r1
    records: src/A:L2
    records walked: 2
```

## What the two answers say

- "contains" rests on the recorded chain: a stated edge from a line of the release to the withdrawn
  line. It is a statement about the recorded evidence.
- "does not contain" is printed only with a proof, and the proof's scope is printed with it: the source
  and the members it covers, the records the request was narrowed to, and how many records of the
  release the proof visited. It says nothing about the same text arriving under another source, and
  nothing about what a model that generated data carries (docs/coverage.md, section 2).
- The parents of the copy edges here were stated by the step that wrote the file. The tool stores them
  and does not re-derive them from the text. For a generation job, the parent list is what the output
  states it used, and an answer that passes a generation edge says so (docs/capture.md, section 3).
- The store's loud checks catch accidental edits and inconsistent imports. They do not defeat someone who
  rewrites every commitment the store holds (docs/store.md).

## Where to go next

- `docs/cli.md`: every command, its exit codes and its errors. A script that gates on `verify` reads the
  section "verify" there before it relies on the command's exit status.
- `docs/capture.md`: the capture calls for ingest, copy steps, dedup, imports and generation jobs.
- `docs/store.md`: what the store records, how to open, back up and move one, and what each loud check
  compares.
