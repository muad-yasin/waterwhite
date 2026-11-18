# Waterwhite

Waterwhite is a free, open-source tool for teams that train language models. When a source is withdrawn from a training corpus, it tells the team which mixtures need a successor, which records a successor can keep, which checkpoints have evidence of exposure, and what has to be rebuilt to remove the source and what is derived from it through the recorded steps, such as its translations. What "derived" reaches, and what it does not, is stated under "What the answers are exact about".

## What the answers are exact about

The answers are exact about the evidence that was recorded and about the trust assumptions printed beside each answer. They are not exact about the world. Three kinds of statement are kept apart:

- **Declared ancestry.** The parents a step states for an output, such as the records a generation job says it used. The tool stores such a list faithfully. It does not compare it with what the job read, and storing it is not a check of it.
- **Observed reads.** What a harness logged as the job read. The in-process harness logs only the reads that pass through it; the confined harness (Linux, bubblewrap) runs the job with nothing else to open, and `docs/confine.md` says what that does and does not establish. A job is handed a text with each reference, and the store compares each text with the record the reference names when the job's output is registered, and refuses the registration when one is not that record. It makes that comparison for every text it is given, on the edge or on its log: those of the result a job run returns, registered in the same process, and those of an edge or a log built by hand that carries them; a record that holds no text takes only a text that is empty after the normaliser. A log registered without a text for every reference it handed (read back from its file, fetched from the store, or built by hand without its texts) is stored as not compared: it establishes no bound, so a "does not contain" past it reads "may contain" until the original texts are shown and compared (`waterwhite compare-hand-off`; `docs/capture.md` lists the routes). Stores made before this are migrated to schema version 4 and every log in them reads as not compared.
- **A proof of absence.** "Does not contain" is printed only when a coverage proof backs it, and the proof's scope (the source, the members, the steps and the logs it rests on) is printed with it.

Where evidence was not recorded, the tool says what it cannot know and what it answers instead.

The default withdrawal follows recorded ancestry: the inputs and parents of records, and the steps of a run. While the generating-model switch is off (the default), a generating model is neither an input, a parent nor an ancestor of what it generates, whatever model identifier a log holds, and a "does not contain" past a generation edge says "generating-model exposure not assessed". With the switch on, each such edge's model is read from its stored log and must be "none" or a registered checkpoint that itself answers "does not contain"; otherwise the answer is "may contain" naming the model. The off state is a decision about the core's scope. It does not say that every output of such a model carries the source.

The tool runs on your own machines and keeps its evidence store there. It makes no claim of legal compliance, and it makes no claim about cost or effect beyond what has been measured on a named workload.

The store protects against accidental edits and inconsistent imports: its loud checks compare what it records with other records and with the files it names, and its write methods never overwrite recorded history. Waterwhite does not defend against direct edits of its database: a store whose rows were edited outside the tool is read as stored. Its loud checks find some inconsistent edits, not all, and a check that finds nothing is not proof that the file was not edited.

## Status

This is a first build. The core is built and tested against a hand-checked fixture and a pipeline replica; its checks, their expected answers and the open items are in `docs/`, `DECISIONS.md` and `BUILT-LOG.md`.

- **Check 10 is open.** It puts the same questions to six subjects and records every answer: two existing tools (the Dolma toolkit and OriginBlame), two reference points built here (a file-level manifest and a positions-only step trace), and the two combinations of an existing tool with that trace. The existing tools have not yet been run on the replica's output, and the timed measurement runs have not been made, so no result is claimed here.
- Six extensions are designed. The generating-model rule is built and off by default: switched on in the store, it judges a negative past a generation edge by the edge's model, and a temporal ancestry cycle through a model's own history is refused. The content ban, flagged records, published models, exposure counts and adapters are not built: with the switch off, a request that needs the content ban or flagged records is refused with "extension off", and a query naming a published model the store holds answers "not covered"; with the switch on, all three are refused as not built. No code reads the switches of exposure counts and adapters. Switching an extension on turns on only what is built.
- The tool's name and its first-build scope are the owner's decisions.

## Install

You need Python 3.12 or newer and numpy 2 or newer; `pip` installs numpy as the one required dependency. The tool has been built and tried on one Linux machine (Fedora 44, CPython 3.12 for the tests); no other operating system has been tried, and no support for one is claimed. From a checkout of this repository:

```
python -m pip install .
```

This line has not been run from a clean environment in the tests of this repository. The tests run the commands of `docs/quickstart.md` through the checkout's own sources.

Optional parts, each needed only for what it names:

- `python -m pip install ".[cyclic]"` adds torch. It is needed only to register a run that was read with the cyclic sampler, whose shuffle the join computes.
- `python -m pip install ".[test]"` adds pytest.
- The confined harness needs Linux and bubblewrap (`docs/confine.md`). The pipeline replica and the oracle need a Megatron environment built by `tools/setup_megatron_env.sh`. Neither is needed to record evidence with the capture calls or to ask a question of a store.

## Use

```
waterwhite <command> --store <file> [--format json|text] ...
```

The commands are `init`, `query`, `show`, `check`, `migrate`, `register-run`, `replay`, `recover`, `report` and `verify`. Every command prints one JSON document (or a few lines of text with `--format text`); an error prints one JSON object on standard error. `docs/cli.md` describes each command, its exit codes and the form of every error.

`docs/quickstart.md` is one small path from an empty folder to an answer: make a store, record a tiny pipeline with the capture calls, check the store, and ask what a withdrawn record reaches. The commands it shows are run by a test and their output is compared with what the page states.

A script that gates on `verify` reads `docs/cli.md`, under "verify", before it relies on the command's exit status. By default `verify` exits 0 whenever it made the verification, even when the two sides differ; `--require-held` makes it exit 5 when the conditions listed there are not all held, and the document is printed either way.

## Licence

MIT. See `LICENSE`.
