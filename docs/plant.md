# The plant generator and the counting script

Two scripts in `fixture/plant/`:

- `generate.py` writes one generated set of records into a folder, with a list of
  everything it planted there (the plant list).
- `count.py` reads the plant list and, when asked, the chunk files a pipeline made
  from the set. It prints the answers the set must give when source A is withdrawn.

The hand-checked fixture is small enough to check by hand. The sets of 10,000 and
100,000 records are not. Their expected answers come from these two scripts. Both
are written before the tool exists and import nothing from it, from the replica or
from the oracle: the standard library only. So a check that compares the tool with
these answers can come out either way (plan W5).

Plan sections served: 5.1 (structure, sizes, where each set's expected answers come
from), 5.2 (A's records must reach every chunk), 5.5 Checks 3, 6 and 9, 5.6 (the
measured cells), 4.1 (actions), 4.4 (clean sources), 3.2 and L3 (candidates), W3.
DECISIONS.md D-19 fixes the outcome of a release row that copies A's own record.
DECISIONS.md D-23 adds the paraphrases of unrelated records (section 2.1).

## 1. Quick start

```
python fixture/plant/generate.py --out SET --records 10000 --seed 1
python fixture/plant/generate.py --out CLEAN --records 10000 --seed 1 --clean
python fixture/plant/count.py --set SET
python fixture/plant/count.py --set SET --chunks CHUNKS --layout round-robin
python fixture/plant/count.py --set SET --chunks CHUNKS --layout round-robin \
        --sanitizer-input RAW
```

`SET` and `CLEAN` must be new or empty folders. `CHUNKS` is a folder of
tokenizer-input files (section 5.2). `RAW` is a folder of sanitizer-input files
(section 5.3).

## 2. What a set holds

A set repeats the structure of the hand-checked fixture at scale (plan 5.1), with
one addition the fixture does not have: the paraphrases of unrelated records.

| Part | What it is |
|---|---|
| X pairs | Sources A and B each supply the same whole passage. Line k of A and line k of B hold the same text, for k from 1 to the number of X pairs. |
| Paraphrases P | Made from A's copy of an X passage. |
| Paraphrases of unrelated records UP | A few. Each is made from one unrelated record of Z, by the same job as P. They have nothing to do with A (section 2.1). |
| Translations T | German. Each is made from one paraphrase. |
| Summaries S | Each is made from one passage of A and one passage of C, parent mode "all". |
| Planted pairs | A fixed number of identical records across sources: one member in C, its twin in Z. They have nothing to do with A. |
| Sanitizer cases | Lines in Z that a sanitizer must skip, and records with control characters inside real text. |
| Unrelated records | Every other record. Source Z holds most of them. Nothing is derived from them, except that a few records of Z each have one paraphrase UP. |
| Release R1 | A's and B's X copies, every P, UP, T and S, the planted pairs, and the first half of each unrelated source. |
| Release R2 | All of B, and the second half of each unrelated source. Nothing of A. |

Texts are synthetic. Each is built from word lists written in `generate.py`. No
corpus text and no benchmark item is used (plan W3). P, T and S are fixed outputs
with no model call: each is a text of its own, keyed to its parents, and its first
sentence names the code words of the records it was made from.

### 2.1 The paraphrases of unrelated records

In the hand-checked fixture every generated record derives from A. If a generated
set only repeated that, a withdrawal of A would retire every generated record, and
a report that retired records because they are generated would give the same
counts and the same retained records as a report that retires them because of
where they come from. The rebuild check (plan Check 9) could not tell the two
apart. So both generated sets also hold a few generated records that do not derive
from A (plan 5.1 "Sizes"; DECISIONS.md D-23). The hand-checked fixture and its
table hold none.

- **What they are.** A paraphrase `UP-<n>` is a fixed output of the job
  `"paraphrase"`, handed one record and using it, with no model call: made exactly
  like P, so its parent link is established and nothing about it is unresolved.
- **Its parent** is an unrelated record of Z: a line of the first half of Z, which
  is an input of R1, that is no pair member and no sanitizer case. Each parent has
  one paraphrase. The parents are drawn with the seed.
- **Where they are.** Inside the total: each takes the place of one unrelated line
  of Z, so a set of 10,000 records still holds 10,000. In `generation.jsonl`, after
  A's paraphrases. In R1, as rows. In no other release.
- **Under "withdraw A"** none is in scope. Each stays a row of the successor
  (outcome `keep`), and so does its parent. The clean set holds them (section 4.4),
  so the reference side of a rebuild check holds them too.
- **Their ids** start with `UP`. No other id of a set starts with the letter U, so
  an id cannot be taken for one of A's paraphrases `P-<n>`, whether a reader
  compares the whole prefix or its first letter.
- **How many.** One for every thousand records, and never fewer than two: 10 in
  the 10k set, 100 in the 100k set. One would be enough to tell the two kinds of
  report apart. Two are the least with which a deliberately wrong report can retire
  one of them while another stays. One per thousand keeps their share the same in
  both sets: with one derivation, one generated record in 111. That is "a few" at
  both sizes, and enough for the round-robin layout to put them in several chunks.
  The number does not grow with `derivations`.
- **A set with `derivations` 0 holds none.** It has no generation job at all, so
  there is no generated record a report could retire wrongly.

A report that retires one of them makes the two sides of a rebuild check differ:
the reference side, built from the clean set, holds the record, and the repaired
side does not.

### Counts

`records` is the number of lines in all files together: every line of every source
file plus every row of `generation.jsonl`. It is also the number of rows in the
plant list. With N = `records`, d = `derivations`, K = `pairs` and C = `sanitizer_cases`:

| Thing | Count | Where |
|---|---|---|
| X pairs | N // 20 | A lines 1 to N//20, B lines 1 to N//20 |
| A's summary passages | N // 100 | A, after the X copies |
| B's other records | N // 20 | B, after the X copies; unrelated |
| C's summary passages | N // 100 | C lines 1 to N//100 |
| C's pair members | K | C, after the passages |
| C's other records | N // 100 | C, at the end; unrelated, in no release |
| Paraphrases P | d x (N // 20) | `generation.jsonl`, first |
| Paraphrases of unrelated records UP | max(2, N // 1000), or 0 when d is 0 | `generation.jsonl`, second |
| Translations | d x (N // 20) | `generation.jsonl`, third |
| Summaries | d x (N // 100) | `generation.jsonl`, last |
| Sanitizer cases | 3 x C | Z, at drawn lines in its first half |
| Z's pair members | K | Z, at drawn lines in its first half |
| Parents of the UP | as many as there are UP | Z, at drawn lines in its first half; unrelated records |
| Unrelated lines | the rest | Z, and with ten sources also Z2 to Z7 |

With ten sources the lines left after A, B, C and the generated records are split:
each of Z2 to Z7 takes a tenth of them and Z takes what remains.

`derivations` is the number of records made directly from each record of A that
things are made from. With d = 1 each X copy of A has one paraphrase, each
paraphrase has one translation, and each summary passage of A has one summary: the
fixture's own shape. With d = 2 each X copy has two paraphrases (each with one
translation) and each summary passage has two summaries, the second with the next
passage of C. With d = 0 `generation.jsonl` is empty. The paraphrases of
unrelated records are not derivations of A's records and are counted apart.

A record's text is about `length` characters: at least `length`, and less than
`length` + 130. A summary is about half as long.

For the 10k set with the defaults: A 600 lines, B 1,000, C 500, Z 6,790, generated
1,110 (500 P, 10 UP, 500 T, 100 S). For the 100k set with the defaults: A 6,000
lines, B 10,000, C 2,300, Z 70,600, generated 11,100 (5,000 P, 100 UP, 5,000 T,
1,000 S). Section 7.1 gives the counts `count.py` prints for the two sets the
checks use.

## 3. Parameters of `generate.py`

| Option | Default | Meaning |
|---|---|---|
| `--out` | required | The folder to write. It must be new or empty. |
| `--records` | required | N above. 10,000 and 100,000 are the sizes the checks use. |
| `--seed` | required | Zero or more. The same seed and parameters give the same bytes. |
| `--length` | 2000 | Characters per record, about. The measured cells use 500, 2000 and 8000. At least 120. |
| `--derivations` | 1 | 0, 1 or 2. With 0 the set holds no generated record at all. |
| `--sources` | 4 | 4 (A, B, C, Z) or 10 (also Z2 to Z7, which hold unrelated records only). |
| `--dedup` | on | `on` or `off`. Whether the releases are built with the exact-dedup step. Check 3's 100k set uses `off`, so both members of each pair reach a shard. |
| `--chunks` | 20 | The chunk count of the round-robin layout. |
| `--pairs` | 300 | The number of planted pairs. |
| `--sanitizer-cases` | 6 | Cases per group. There are three groups, so 18 lines by default. At least 3. |
| `--clean` | off | Write the same set without source A and without anything derived from it (section 4.4). The paraphrases of unrelated records stay. |

It prints one JSON document: the parameters, the row count per kind, and for each
file its size, line count and SHA-256, plus `set_digest`, one digest over all files.
Two runs with the same options print the same document.

It refuses, with a named error on standard error and nothing written:

- `ParameterError` (exit 2): the parameters cannot give a set with the guarantees
  of section 6. For example fewer X copies in A than chunks, or too few lines in Z
  for the pairs, the sanitizer cases and the parents of the paraphrases of
  unrelated records.
- `OutputFolderNotEmpty` (exit 2): the folder holds something. A set is never
  written over files.
- `SelfCheckFailed` (exit 3): the generator caught itself breaking a guarantee, for
  example two records with the same text outside a planted pair, or a clean set
  that would hold a record derived from A.

There is no clock in the script and no shared random state. Every text and every
placement is drawn from bytes that depend only on the seed and on a key that names
the thing being made.

## 4. Files of a set

```
SET/
  set.toml
  plant_list.jsonl
  generation.jsonl
  sources/A.jsonl  B.jsonl  C.jsonl  Z.jsonl  (Z2.jsonl to Z7.jsonl with ten sources)
```

All files are UTF-8 with one `\n` after each line.

### 4.1 Source files and ids

`sources/<NAME>.jsonl` has one row per line: `{"id": "<NAME>-<nnnnnn>", "text": "..."}`.

**An id names a line.** The number in an id is the line number of that row in its
file, counted from 1 and written with six digits. `Z-000037` is line 37 of
`sources/Z.jsonl`. This holds for every line, including a line that does not parse
as JSON: such a line has no `id` field of its own, and its id is its position. So a
range like `Z-000001..Z-003395` means lines 1 to 3,395 of that file, whatever they hold.

### 4.2 `generation.jsonl`

One row per generated record: `{"id", "job", "handed", "used", "text"}`.

- `id`: `P-<n>`, `UP-<n>`, `T-<n>` or `S-<n>`, six digits. A's paraphrases come
  first, then the paraphrases of unrelated records, then all translations, then
  all summaries.
- `job`: `"paraphrase"`, `"translation"` or `"summary"`. The job of a `UP` row is
  `"paraphrase"`, as for a `P` row.
- `handed`: the ids of the records the job was handed.
- `used`: the ids the output states it used. Here always equal to `handed`.
- `text`: the fixed output.

`P-<n>` is handed one X copy of A, read from the source file (before any dedup).
`UP-<n>` is handed one unrelated line of Z, read from the source file.
`T-<n>` is handed `P-<n>`. `S-<n>` is handed one passage of A and one of C.

An id of a generated record does not name a line: `UP-000001` is the row after the
last `P` row. The plant list gives each row's line.

### 4.3 `set.toml`

Every key:

| Key | Type | Meaning |
|---|---|---|
| `format` | string | `"waterwhite.plant-set/1"`. |
| `clean` | boolean | `true` for a set written with `--clean`. |
| `withdrawn` | string | The source the plant list's outcomes are about. Always `"A"`. |
| `parameters.records`, `.length`, `.derivations`, `.sources`, `.seed`, `.dedup`, `.chunks`, `.pairs`, `.sanitizer_cases` | | The options of section 3, as given. `dedup` is a boolean. |
| `normaliser.name` | string | `"control-chars/1"`: the normalisation the set's guarantees are stated for. It removes control characters and strips the ends. |
| `counts.<kind>` | integer | Rows per kind (section 4.5 lists the kinds). |
| `sources.<NAME>.file` | string | The source's file, relative to the set folder. |
| `sources.<NAME>.member` | string | The member name to register it under: `<NAME>/r1`. |
| `sources.<NAME>.revision` | string | `"r1"`. |
| `sources.<NAME>.lines` | integer | The number of lines in the file. |
| `sources.<NAME>.records` | list | The ids of all its lines, as one range. |
| `generation.file` | string | `"generation.jsonl"`. |
| `generation.model` | string | `"none"`: no job calls a model. |
| `generation.rows` | integer | The number of rows in the file. |
| `generation.jobs.paraphrase`, `.translation`, `.summary` | list | The ids each job made, as ranges, in the order of the job's rows in the file. Empty when there are none. The job `paraphrase` lists the `P` range and then the `UP` range. |
| `releases.<R>.inputs` | list | The records the release is built from, as ids and ranges, in order. |
| `releases.<R>.dedup` | boolean | Whether the release is built with the exact-dedup step. |
| `releases.<R>.dedup_priority` | list | Source names. Among inputs with the same normalised text, the copy from the source that comes first in this list is kept. |
| `releases.<R>.dedup_winner` | string | The first entry of `dedup_priority`. |
| `layouts."round-robin".chunks` | integer | The chunk count of the round-robin layout. |

The sources appear in the order A, B, C, Z, Z2 to Z7. The releases are `R1` and `R2`.

### 4.4 The clean set

`--clean` writes the sources as they would be if A had never existed (plan 4.4): no
`sources/A.jsonl`, no P, T or S (each derives from A), and no plant list. Its
`generation.jsonl` holds the paraphrases of unrelated records and nothing else,
each row byte for byte the row of the full set, and `generation.jobs.paraphrase`
lists the `UP` range alone. With `derivations` 0 the file is empty. Its `set.toml`
names the same releases with the withdrawn ids left out of their inputs, and a
`dedup_priority` without A. Every other source file is byte for byte the file of
the full set, so the ids of the two sets mean the same lines. The parents of the
`UP` rows are lines of Z, so the clean set holds them as well.

The reference side of a repair check is built from this folder. The retained
(text, multiplicity) reference itself comes from that build, not from these scripts.

### 4.5 `plant_list.jsonl`

One row per line of the set, in file order: A, B, C, Z, the further sources, then
`generation.jsonl`. Every row has exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `id` | string | The record's id. |
| `file` | string | The file it is in, relative to the set folder. |
| `line` | integer | Its line in that file, from 1. |
| `source` | string or null | The source name. Null for a generated record. |
| `kind` | string | One of the kinds below. |
| `job` | string or null | The generation job. Null for a source line. |
| `parents` | list | The ids it was made from. Empty for a source line. |
| `parent_mode` | string or null | `"all"` when it has parents (it needs every one of them), else null. |
| `pair` | string or null | The id of the identical pair it belongs to: `X-<n>` for an X pair, `PAIR-<n>` for a planted pair. Null otherwise. |
| `twin` | string or null | The id of the other member of its pair. |
| `withdraw_A.in_scope` | boolean | Whether "withdraw A" puts it in scope. |
| `withdraw_A.why` | string or null | `"own"`: it is A's own record. `"derived"`: it is derived from one. Null when out of scope. |
| `withdraw_A.roots` | list | For a derived record, the records of A its parents lead back to. Empty for a record out of scope, a paraphrase of an unrelated record included. |
| `releases.<R>.row` | string | Present only for a release it is an input of. `"present"`: it is a row of the release. `"dedup-loser"`: dedup dropped it. |
| `releases.<R>.withdraw_A` | string | What the rebuild does with it under "withdraw A": one of the five outcomes below. |
| `releases.<R>.resource_to` | string | Only with `"re-source"`: the copy the row is re-sourced to. |
| `releases.<R>.promoted_for` | string | Only with `"promote"`: the withdrawn row it stands in for. |
| `sanitizer` | table or null | For a sanitizer case (below). Null otherwise. |
| `text_sha256` | string or null | SHA-256 of the UTF-8 bytes of the record's text after normalisation: the text a chunk file must carry for it. Null for a line the sanitizer skips. |
| `chars` | integer or null | The length of that text. |

Kinds:

| `kind` | What |
|---|---|
| `x-copy` | A's or B's copy of an X passage. |
| `passage` | A passage of A or of C that summaries are made from. |
| `paraphrase`, `translation`, `summary` | A generated record made from A's records: a P, a T or an S. Always in scope. |
| `paraphrase-of-unrelated` | A generated record made from an unrelated record of Z: a UP (section 2.1). Never in scope. |
| `pair-member` | A member of a planted pair. |
| `sanitizer-case` | A planted sanitizer case. |
| `unrelated` | Any other record. |

Outcomes under "withdraw A" (plan 4.1 and 4.4). Every input of a release has
exactly one of five. Four are the plan's action words for a record; the fifth,
`none`, is for an input that is no row before and no row after.

| `withdraw_A` | Meaning |
|---|---|
| `keep` | The row stays. |
| `re-source` | The row stays in its place and is re-sourced to another source's identical copy (`resource_to`). It was A's copy, kept by dedup. |
| `promote` | This copy was dropped by dedup in favour of A's. It returns to the release (`promoted_for`). Re-source and promote are the two sides of one event, so their counts are equal. |
| `retire` | The row leaves the release. It is derived from A (a P, T or S), or it is a copy of A's own record with no equivalent copy among the release's inputs to take its place, which is every X row of A when dedup is off. |
| `none` | The record was dropped by dedup and stays dropped. It is in neither the release nor its successor. |

There is no outcome `remove` (DECISIONS.md D-19). A release row that copies A's own
record descends from that record through a single-parent copy edge, so when nothing
can stand in for it, it is retired like any other descendant. A plant list that
states another word than these five is refused by `count.py`. The number of records
removed from a release is still counted: it is the number of rows that leave the
successor (section 7).

With dedup on and A first in the priority (the 10k set): each X row of A is
`re-source`, each X copy of B is `promote`, every P, T and S is `retire`, Z's pair
members are `none`, everything else is `keep`. With dedup off (the 100k set of
Check 3): each X row of A is `retire`, B's copy, which is a row of its own, is
`keep`, and nothing is `re-source`, `promote` or `none`.

The row of a paraphrase of an unrelated record, in either set:

```
"kind": "paraphrase-of-unrelated", "job": "paraphrase",
"parents": ["Z-<line>"], "parent_mode": "all",
"withdraw_A": {"in_scope": false, "why": null, "roots": []},
"releases": {"R1": {"row": "present", "withdraw_A": "keep"}}
```

Its parent's row is an `unrelated` row of Z with the same entry for R1.

The `sanitizer` table:

| Key | Meaning |
|---|---|
| `case` | `"unparseable"`: the line is not JSON. `"empty"`: the text is `""`. `"control-only"`: the text is control characters only. `"control-and-space"`: control characters and spaces only. `"control-inside"`: control characters inside real text, at both ends and at four places within. |
| `skip` | `true` when the sanitizer must skip the line. `false` for `control-inside`: the record survives without its control characters. |
| `reason` | The drop reason the sanitizer must log: `"sanitizer skip"` for an unparseable line, `"empty line"` for a text that is empty after normalisation. Null when `skip` is false. |
| `raw_sha256` | SHA-256 of the line's bytes as they stand in the source file, without the line break. |

Cases run through the three groups in turn (unparseable, empty, control inside) and,
inside a group, through its variants. All of them sit in Z, in the part that is an
input of R1.

## 5. For whoever builds the pipeline that consumes a set

### 5.1 What the plant list assumes about the pipeline

The outcomes in the plant list are right only for a pipeline that does this:

1. **Release inputs.** A release is built from exactly the lines `releases.<R>.inputs`
   names, each once.
2. **Generation reads the sources.** A paraphrase is made from A's line, or for a
   `UP` from Z's line, in the source file, not from a release. Every row of
   `generation.jsonl` is built whatever it derives from; the releases decide which
   rows are selected.
3. **Dedup.** When `dedup` is true, inputs with the same normalised text form a
   group, and the copy whose source comes first in `dedup_priority` is the row;
   the others are dropped. A line that does not parse, or whose normalised text is
   empty, is never put in a group: it passes through and the sanitizer drops it
   later. No generated record is identical to another record.
4. **Sanitizer.** A line that is not JSON is skipped and logged with the reason
   "sanitizer skip". A line whose text is empty after normalisation is skipped and
   logged with the reason "empty line". Every other line survives with its
   normalised text.
5. **Normalisation** removes control characters and strips the ends
   (`control-chars/1`). The set is built so that every reading of that sentence
   agrees on its texts: section 6, guarantee 4.

`count.py` checks points 1, 3, 4 and 5 against the chunk files and stops when they
do not hold.

### 5.2 The chunk files `count.py` reads

A folder with one file per shard, named `<shard>.jsonl`. Each row is
`{"id": ..., "text": ...}`. The file holds only the lines that survived the
sanitizer, in order, with their sanitized text: line i of the file is document i of
the shard. The ids are the ids of this document. Files named `*.manifest.jsonl`
and files of other types in the folder are left alone.

Over all its files, the folder must hold exactly one line for each row of the
named releases that survives the sanitizer (`--release`, default `R1`; give it
twice for a folder built from both releases).

### 5.3 The sanitizer-input files `count.py` reads

A chunk file holds surviving lines only. Where a skipped line was read can
therefore not be taken from it. With `--sanitizer-input FOLDER` the script also
reads the chunks as the sanitizer read them: one file per shard, named
`<shard>.jsonl` like its chunk file, each line as its source file holds it, byte
for byte, readable or not (the replica writes them to
`pipeline/<layout>/sanitizer-input/`, docs/replica.md section 2).

A line there whose bytes have the digest the plant list states for a line the
sanitizer must skip (`sanitizer.raw_sha256`) is that line. Every other line must
be a row with an id, and its id must be the id of the next line of the shard's
chunk file: the surviving lines keep their order, so a skipped line shifts no
ordinal. The script stops when this does not hold (`SanitizerInputMismatch`). It
does not read the sanitizer's own log; a test compares the two.

## 6. Guarantees, and how each is checked

| # | Guarantee | How it is given | How it is checked |
|---|---|---|---|
| 1 | The same call gives the same bytes. | No clock, no shared random state; everything is drawn from bytes keyed by the seed. | `generate.py` prints a digest per file. The tests generate twice and compare. |
| 2 | Every text is unique after normalisation, except the two members of an X pair or of a planted pair. | The first sentence of each text holds a code word made from the id of the record that owns the text, and no two ids give the same word. A pair's second member copies its twin's text. | The generator stops if a text digest repeats outside a pair. `count.py` checks the digests in the plant list. The tests read the files and normalise them. |
| 3 | An id names a line. | Ids are written from line numbers. | `count.py` checks every row. The tests read the files. |
| 4 | Normalisation is not open to dispute on these texts. | The only control characters planted are U+0001 to U+0008, U+000B, U+000C and U+000E to U+001F. No text holds a tab, a line break, NUL, DEL or a C1 character. Outside the sanitizer cases no text has a control character or a space at either end. | The tests read every line. |
| 5 | A's records reach every round-robin chunk. | See below. | `count.py --layout round-robin` reads the chunk files and stops unless every chunk holds a record in scope. |
| 6 | Some planted pairs land in one shard and some are split. | See below. | `count.py` classifies each pair from the chunk-file ids and stops unless both kinds exist. |
| 7 | The plant list's outcomes follow from the set's structure. | The generator writes each outcome by construction. | `count.py` works every scope and outcome out again from sources, parents, text digests, release inputs and dedup priority, and stops on the first row that differs. |
| 8 | The clean set is the full set without the withdrawn content. | A record's text does not depend on what else is in the set. | The tests compare the files byte for byte and the ids with `clean_omits`. |
| 9 | Some generated records do not derive from A, and they stay. | The paraphrases of unrelated records (section 2.1): made from lines of Z, rows of R1, written into the clean set. | The generator stops if their number is not the layout's or if a clean set would hold a record derived from A. `count.py` works out each generated record's scope from its parents, stops on a row whose kind, scope or outcome says otherwise, and stops with `NoGeneratedRecordStays` on a release that retires generated records and keeps none. The tests walk the files on their own and compare. |

**Guarantee 5 in full.** `sources/A.jsonl` holds A's records and nothing else. Its
first N // 20 lines are inputs of R1, and all of them are rows of R1 when A comes
first in the dedup priority or dedup is off. The generator refuses parameters that
give fewer of them than there are chunks. Now suppose the pipeline deals the rows
that came from one file as one file: line j goes to chunk (start + j) modulo the
chunk count. Then a file of n rows puts at least n // chunks rows in every chunk,
wherever the dealing starts and however the file was shuffled first. So every chunk
gets at least (N // 20) // chunks of A's rows: 25 in the 10k set with 20 chunks.

This does not cover a pipeline that deals all rows of a release as one shuffled
file. There the place of A's rows in the dealt sequence decides, not their number
(plan 4.5). In the 10k set 1,600 of R1's 5,005 rows are in scope, so a chunk with
none is very unlikely, but it is not excluded. `count.py` assumes neither reading:
it reads the chunk files.

**Guarantee 6 in full.** Where a pair's members land is the pipeline's shuffle, not
the generator's choice. With 20 chunks, and each member landing in a chunk evenly
and independently of its twin, a pair shares a chunk with chance 1 in 20. The
chance that none of 300 pairs does is (19/20)^300, about 2 in 10 million. That no
pair is split is far less likely still. In a layout partitioned by source every
pair is split, since its members come from different sources. With dedup on, one
member of each pair is dropped before chunking, and no pair has two lines at all.

## 7. `count.py`

```
python fixture/plant/count.py --set SET
        [--chunks FOLDER --layout round-robin|partitioned]
        [--sanitizer-input FOLDER]
        [--release R1] [--release R2]
        [--require-pair-kinds auto|yes|no] [--expand-ids]
```

It reads `SET/set.toml` and `SET/plant_list.jsonl`, and the chunk folder when one
is given. It reads nothing the tool wrote. It prints one JSON document:

| Part | What it holds |
|---|---|
| `format` | `"waterwhite.plant-count/4"`. Version 1 had a sixth outcome, `remove`, in `actions` and `ids`. Version 2 had no part `generated`, no `releases.<R>.generated` and no `generated_kept` in a shard's row. Version 3 had no part `documents`, no `check3.drops` and no `check6.translations`; every part it had is unchanged. |
| `request` | `{"kind": "withdraw", "source": "A"}`. |
| `set` | The set's parameters. |
| `records` | `total`; `by_kind`; `by_source`; `in_scope` with `total`, `own` and `derived`; `removed` (records the clean sources omit); `retained` (the rest). |
| `clean_omits` | `count` and `ids`: the ids a clean source set omits. |
| `generated` | `total`: the generated records. `derived`: those the withdrawal reaches. `stay`: the generated records that stay, with `count`, `ids` and `list` (below). |
| `releases.<R>` | `inputs`; `dedup`; `rows_before`; `in_scope_rows_before` (the rows that are A's or derived from A); `dedup_losers`; `actions` (a count for each of keep, re-source, promote, retire, none); `retire_by_why` (the retired rows split into `own`, copies of A's own records, and `derived`); `removed` (the records removed: the rows that leave the successor, which is the retire count); `retained` (keep + re-source); `rows_after`; `rows_from_scratch` (the rows of the release built from scratch without the withdrawn content, counted by another road); `successor_needed`; `ids` (the ids with each of re-source, promote, retire); `generated` (the release's generated inputs: `inputs`, `out_of_scope`, `actions` with a count for each outcome, and `kept_ids`, the generated rows that stay). |
| `identities` | Each sum that must hold, with both sides and `holds`. |
| `actions.records` | The action counts over all releases. |
| `actions.shards` | `{"keep": n, "rewrite": n}` from the chunk files. Null without `--chunks`. |
| `check3.sanitizer_drops.<R>` | The lines of the release the sanitizer must log as drops: `id`, `source`, `line`, `case`, `reason`, `raw_sha256`. |
| `check3.pairs` | Null without `--chunks`. Else `required`, `counts` per kind and placement, and `list`. |
| `check3.drops` | Null without `--sanitizer-input`. Else `count`, `by_reason`, and a row per shard with `input_lines`, `surviving_lines` and its `drops` (below). |
| `check6.candidates` | Per release, each row whose identical text reached the release from more than one source. |
| `check6.translations` | The planted translations with the releases that receive each and, with `--chunks`, the shards (below). |
| `shards` | Null without `--chunks`. Else `layout`, `releases`, `lines`, a row per shard, `keep`, `rewrite`, `all_rewrite`, `actions`. |
| `documents` | Null without `--chunks`. Else `total` and a row per shard with the planted record of every chunk-file line (below). |

Id lists are written as ranges (`"A-000001..A-000500"`). With `--expand-ids` they
are written in full, and each release also lists `re_sourced`: which row goes to
which copy.

**The generated records that stay.** `generated.stay` lists every generated record
that "withdraw A" does not reach, so that a test can aim at one of them: retire it
in a deliberately wrong report and see the two sides of the rebuild check differ.
Which records are on the list is decided by scope, worked out from each record's
parents down to the source lines. Neither the kind nor the id decides. Each entry
of `list` gives `id`, `kind`, `job`, `parents`, `parent_mode`, `in_scope` (always
false), `releases` (the record's outcome in each release it is an input of) and
`lines`. `lines` is null without `--chunks`. With `--chunks` it holds, for each
line of the chunk files that carries the record, `shard`, `line` (from 1),
`ordinal` (from 0) and `shard_action`, the action of that shard.

In the sets the generator writes, the list is the `UP` records, each with the
outcome `keep` in R1.

**The sums in `identities`.** For the set: removed + retained = total; own +
derived = in scope. For each release: every input has exactly one outcome; keep +
re-source + retire = rows before; promote + none = dedup losers; re-source =
promote; removed = rows in scope before - re-source (a re-sourced row is in scope
and keeps its place); rows after = rows of a from-scratch build without the source;
generated rows kept = generated inputs the source does not reach (the left side is
counted from outcomes, the right side from scope). A sum that does not hold stops
the script. Two of these fail for a count that retires a generated record which
does not derive from A: the last one, and "rows after = rows of a from-scratch
build", since the from-scratch build holds that record.

**Shards (Check 9).** A shard is `keep` when none of its lines carries an id the
plant list marks as A's or derived from A, and it receives no record that is
promoted or re-sourced. Otherwise it is `rewrite`. A re-sourced row keeps its
place, so the shard that receives the promoted copy is the shard whose chunk file
carries the re-sourced row. Each shard's row gives `lines`, `in_scope_lines`,
`own`, `derived`, `receives`, `generated_kept` and `action`. `keep` is the
kept-shard list.

`generated_kept` is the number of the shard's lines that carry a generated record
that stays. It decides nothing: such a line is no reason to rewrite a shard, and a
shard that holds nothing else in scope is `keep`. Where the pipeline puts every
generation output into one shard, as the replica's partitioned layout does, the
`UP` records share a shard with P, T and S; that shard is `rewrite`, and its
successor must still hold them.

With `--layout round-robin` every chunk must be `rewrite`, and the number of shard
files must be the set's chunk count. With `--layout partitioned` at least one shard
must be `keep`.

**Pairs (Check 3).** Each identical pair is classified from the chunk-file ids:

| `placement` | Meaning |
|---|---|
| `same-shard` | Both members have a line, in one shard. |
| `split` | Both members have a line, in two shards. |
| `single` | One member has a line. Its twin was dropped by dedup. |
| `absent` | Neither has a line. |
| `multiple` | More than two lines carry the pair's text (a folder built from both releases with dedup off). |

Each member is listed with `id`, `source`, `shard`, `line` (from 1) and `ordinal`
(from 0, so `line` - 1), and `expected_label`: `"candidates"` when another line of
its pair is in the same shard, `"recovered"` when not (plan 3.3). `kind` is
`"pair-member"` for a planted pair and `"x-copy"` for an X pair. The planted pairs
must show at least one `same-shard` and one `split`. `--require-pair-kinds` is
`auto` by default: required when the set was generated with dedup off.

**Candidates (Check 6).** At a copy step that was not observed, a row whose text is
identical in two of the step's inputs has a set of candidate parents, and the answer
is "may contain" for each candidate's source (plan L3). Each entry gives `release`,
`pair`, `text_sha256`, `rows` (the members that are rows of the release), `label`
(`"candidates"`), `candidates` (each with `id` and `source`), `sources`,
`may_contain_for` and `withdrawn_source_among_candidates`. The X pairs name A and B.
The planted pairs name C and Z.

**Documents (Checks 3 and 6).** Line i of a chunk file, counted from 1, is
document i - 1 of its shard. `documents.shards` has one row per shard: `shard`,
`documents`, and `ids`, the id of each line in order, so that `ids[n]` is the
planted record of the document with ordinal n. An id stands for its planted
record here: the script has already refused a line whose id is no plant row, or
whose text is not that record's sanitized text. `identical` lists the documents
of the shard that hold the same text, each group with `pair`, `text_sha256`,
`ordinals`, `lines` and `ids`. No derivation can tell such documents apart, so
each is expected with the label "candidates" and its group as the candidate set,
and every other document with the label "recovered" (plan 3.3). The groups are
found from the text digests of the lines, not from the pair ids. `labels` counts
the documents of the shard with each expected label.

**Drops per shard (Check 3).** `check3.sanitizer_drops` lists the lines the
sanitizer must skip. `check3.drops` places them, from the sanitizer-input files
(section 5.3): per shard `input_lines`, `surviving_lines` and `drops`, each drop
with `id`, `source`, `line` (its line in the source file), `case`, `reason`,
`raw_sha256`, `shard`, `input_line` (its line in the shard's sanitizer input,
from 1) and `ordinal_of_next_line`, the number of surviving lines before it,
which is the ordinal the next surviving line of the shard gets. `count` and
`by_reason` are the totals. The sets plant no surviving line that gives no
tokens, so the expected number of drops with the reason "no tokens" is zero and
the script lists none.

**Translations (Check 6).** `check6.translations` holds `count`, `ids`, `file`
(the set's file of generated rows), `job` (`"translation"`), `releases` (per
release, how many translations are rows of it), and `list`: per translation
`id`, `parents`, `roots` (the records of A it leads back to), `releases` (the
releases it is a row of) and `lines`. `lines` is null without `--chunks`; with it,
each chunk-file line that carries the translation, as `shard`, `line` and
`ordinal`. `shards` lists the shards that receive at least one translation, each
with how many, and `shards_without` the shards that receive none; both are null
without `--chunks`. A translation whose making was not recorded has no recorded
origin (plan L5). Everything that holds one is what Check 6 asks about: the file
the translation job wrote (the replica registers it as `gen/translation`), the
releases with a count above zero, and the shards of `shards`.

### 7.1 The counts of the two sets the checks use

Both with seed 1 and every other option at its default. The counts follow from the
parameters and do not depend on the seed or on `--length`.

| | 10k, dedup on (Check 9) | 100k, dedup off (Check 3) |
|---|---|---|
| records in scope | 1,700 (600 own, 1,100 derived) | 17,000 (6,000 own, 11,000 derived) |
| records the clean sources keep | 8,300 | 83,000 |
| generated records | 1,110 (1,100 derived, 10 that stay) | 11,100 (11,000 derived, 100 that stay) |
| the generated records that stay | `UP-000001..UP-000010` | `UP-000001..UP-000100` |
| R1 inputs | 5,805 | 56,700 |
| R1 rows before, dedup losers | 5,005 and 800 | 56,700 and 0 |
| R1 `keep` | 3,405 | 40,700 |
| R1 `re-source` | 500 | 0 |
| R1 `promote` | 500 | 0 |
| R1 `retire` | 1,100 (all derived) | 16,000 (5,000 own, 11,000 derived) |
| R1 `none` | 300 | 0 |
| R1 records removed | 1,100 | 16,000 |
| R1 rows after | 3,905 | 40,700 |
| R1 generated rows: `retire`, `keep` | 1,100 and 10 | 11,000 and 100 |
| R2 | 4,395 inputs, all `keep` | 45,300 inputs, all `keep` |
| all releases: `keep`, `re-source`, `promote`, `retire`, `none` | 7,800, 500, 500, 1,100, 300 | 86,000, 0, 0, 16,000, 0 |
| lines the sanitizer drops in R1 | 12 (6 skips, 6 empty lines) | 12 (6 skips, 6 empty lines) |

In the 10k set the 3,405 kept rows of R1 are C's 300 pair members, 3,095 lines of
Z and the 10 paraphrases of unrelated records. In the 100k set the 16,000 retired
rows are A's 5,000 X rows, 5,000 P, 5,000 T and 1,000 S; the 40,700 kept rows are
B's 5,000 X copies, C's 300 pair members, 35,300 lines of Z and the 100 paraphrases
of unrelated records.

On a build of the 10k set by the replica (docs/replica.md section 4), the kept-shard
list of the partitioned layout is `shard-C` and `shard-Z`; `shard-A` and
`shard-gen` are `rewrite`, and `shard-gen` carries the 10 generated records that
stay. In the round-robin layout all 20 chunks are `rewrite`.

**What the paraphrases of unrelated records moved** (DECISIONS.md D-23). Each
takes the place of one unrelated line of Z, and Z's lines are split in half between
R1 and R2, so:

| | 10k before | 10k now | 100k before | 100k now |
|---|---|---|---|---|
| lines of Z | 6,800 | 6,790 | 70,700 | 70,600 |
| generated records | 1,100 | 1,110 | 11,000 | 11,100 |
| R1 inputs | 5,800 | 5,805 | 56,650 | 56,700 |
| R1 rows before | 5,000 | 5,005 | 56,650 | 56,700 |
| R1 `keep`, R1 rows after | 3,400 and 3,900 | 3,405 and 3,905 | 40,650 | 40,700 |
| R2 inputs, all `keep` | 4,400 | 4,395 | 45,350 | 45,300 |

R1 gains the new rows and loses half as many lines of Z; R2 loses the other half.
Nothing else moved: the records in scope, the records the clean sources keep,
`re-source`, `promote`, `retire`, `none`, the records removed, the sanitizer's
drops, and `keep` summed over both releases are what they were. The files are not:
every source file but Z's is byte for byte the same, while Z has fewer lines and
the lines planted in it are drawn again, so a set's `set_digest` differs from the
one the earlier generator printed.

The earlier change of DECISIONS.md D-19 renamed an outcome and moved no file and
no count of the 10k set: with dedup on no row ever had the outcome `remove`. In the
100k set the 5,000 X rows of A moved from `remove` to `retire`; records removed
stayed at 16,000.

### 7.2 Which row of the hand-written table reads which part

| Row | Part of the count | What the row takes from it |
|---|---|---|
| `c03.precondition.pair-kinds` | `check3.pairs.counts` | the number of planted pairs placed in one shard and split across two |
| `c03.drops.sanitizer` | `check3.drops`, `documents` | the lines the sanitizer must log as drops, each with its reason and its shard; per shard the number of surviving lines |
| `c03.drops.no-tokens` | `check3.drops`, `documents` | every surviving line is a document, so no drop has the reason "no tokens" |
| `c03.records.capture`, `c03.records.position-recovery`, `c03.records.token-matching` | `documents` | the planted record of each ordinal; the groups of identical documents |
| `c03.pairs.same-shard`, `c03.pairs.same-shard.swap` | `check3.pairs.list` | the pairs with the placement `same-shard`, each with its two chunk-file lines |
| `c03.pairs.split` | `check3.pairs.list` | the pairs with the placement `split`, each with its two chunk files and lines |
| `c06.generated.documents-mapped` | `documents` | per shard, the planted record of each surviving chunk-file line, and the groups of identical documents in one shard |
| `c06.generated.drops` | `check3.drops`, `documents` | the same as for `c03.drops.sanitizer`, on the build made without capture |
| `c06.generated.candidates` | `check6.candidates` | the rows with candidates and the sources of each pair's members |
| `c06.generated.downstream-of-translation` | `check6.translations` | the file, the releases and the shards that receive a translation |
| the rows of `check09.toml` on the 10k set | `releases`, `actions`, `shards`, `generated` | the action counts and the kept-shard list |

The rows `c03.spans.*`, `c03.verification-failed` and `c03.map-fields` read
nothing of the count: their values come from the shard's own index and from the
table. All of these parts are counted on the chunk files of the build the row is
asked of, so a build with unseeded shuffles is counted on its own chunk files.

### What `count.py` refuses

On a failure it prints `{"error": name, "detail": ...}` on standard error, nothing
on standard output, and exits nonzero.

| Error | Exit | When |
|---|---|---|
| `UsageError` | 2 | Bad arguments; a folder that is not a set; a clean set; `--sanitizer-input` without `--chunks`. |
| `PlantListInconsistent` | 3 | A row disagrees with the set's structure: a stated scope or outcome that the sources, parents, digests and dedup priority do not give (a paraphrase of an unrelated record stated as in scope or as retired, for one); a generated record whose kind says it was made from A's records while its parents do not lead to A, or the reverse; a paraphrase of an unrelated record whose parent is not one unrelated source line, or is not a row of the release that holds the paraphrase; an outcome word that is not one of the five (`remove` among them); a missing parent; a pair that is not two identical records from two sources; a repeated text outside a pair; an id that does not name its line; row counts that differ from `set.toml`; a sum that does not hold. |
| `NoGeneratedRecordStays` | 3 | A release retires generated records and keeps none. A report that retired every generated record would match such a set's counts, so it cannot stand behind the rebuild check. A set written before the paraphrases of unrelated records were added is refused this way; generate it again. A set with no generated record at all is counted. |
| `ChunkFilesMismatch` | 3 | The chunk files do not hold exactly the surviving rows: a missing or extra id, an unknown id, a line the sanitizer must skip, a text that is not the record's sanitized text, a line that is not a row. |
| `SanitizerInputMismatch` | 3 | The sanitizer-input files do not hold exactly the chunk-file lines, in order, and the lines the sanitizer must skip: a chunk file with no sanitizer input beside it, a line that is neither a planted skip nor a row with an id, a surviving line that is not the next line of the chunk file, a planted skip that is in no file. |
| `LayoutExpectationFailed` | 3 | Round-robin: a chunk with no record in scope, or a wrong number of shards. Partitioned: no shard can be kept. |
| `PairKindsMissing` | 3 | Pair kinds are required and the planted pairs show no `same-shard` or no `split`. |

## 8. Limits

- The sets hold no surviving line that gives no tokens. Every skipped line is
  dropped by the sanitizer.
- The generated records that stay are paraphrases only, each one step from a line
  of Z. No translation or summary of an unrelated record is planted, and nothing
  is made from a `UP`.
- In a pipeline that puts all generation outputs into one shard, no kept shard
  holds a generated record. That a report keeps the `UP` records is then shown by
  the retained records of the successor, not by the kept-shard list.
- A set with `derivations` 0 holds no generated record of either kind.
- `set.toml` is this generator's own format (`waterwhite.plant-set/1`). The
  replica reads the folder of a set as the generator wrote it, with
  `python -m replica build-set --set <folder>`, and nothing is converted. What
  `set.toml` does not say (a tokenizer, mixtures, a partitioned layout) the
  replica settles itself (docs/replica.md, section 4).
- The outcomes are written for the withdrawal of A only.
- A's copy always wins dedup in a generated set. `count.py` itself follows whatever
  `dedup_priority` says.
