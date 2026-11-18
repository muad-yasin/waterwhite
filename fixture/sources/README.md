# The fixture's texts

Every text in this folder was written for this project. Nothing is copied from a
corpus, a book or a benchmark, so the files can be published with the repository
(plan 1.2, W3, 5.1). The passages are short pieces on everyday subjects: a market,
a repair workshop, a town newsletter, a library, household advice.

The fixture spec (`fixture/spec/fixture_spec.toml`) names these files and says
which record plays which part. The pipeline replica (`docs/replica.md`) reads them.

## Files

One JSON object per line, UTF-8, a line feed after every line.

| File | Rows | What it holds |
|---|---|---|
| `A.jsonl` | `{"id", "text"}` | Source A. `A-001` is the shared passage X, whole, in English. `A-002` is SA, the passage of A that the summary is made from. |
| `B.jsonl` | `{"id", "text"}` | Source B. `B-001` holds the same text as `A-001`, byte for byte. `B-002` to `B-013` are B's own passages. |
| `C.jsonl` | `{"id", "text"}` | Source C. `C-001` is SC, the passage of C that the summary is made from. `C-002` to `C-004` are C's own passages. |
| `Z.jsonl` | `{"id", "text"}` | Source Z, `Z-001` to `Z-090`: unrelated passages. No request in the checks names Z and nothing is derived from it. |
| `generation.jsonl` | `{"id", "job", "handed", "used", "text"}` | The fixed outputs of the generation stand-in. No model wrote them at build time and none is called. |

## The generation outputs

| Id | Job | Handed and used | Text |
|---|---|---|---|
| `P` | `paraphrase` | `A-001` | an English paraphrase of X |
| `T` | `translation` | `P` | a German translation of P |
| `S` | `summary` | `A-002`, `C-001` | a summary of SA and SC |

`handed` lists the records the job is given. `used` lists the ones the output
states it used. In this fixture the two lists are equal for every output.

## X, P and T are short

X, its paraphrase P and the translation T are one sentence each. Their sizes,
counted with the fixture's tokenizer (GPT-2 BPE, DECISIONS.md D-09) and with the
end-of-document token that closes every document of a shard:

| Text | Record | Words | Tokens with the end-of-document token | Limit |
|---|---|---|---|---|
| X | `A-001`, `B-001` | 14 | 16 | 30 |
| P | `P` | 12 | 15 | 30 |
| T | `T` | 9 | 22 | 22 |

The limit on T is what the fixture needs (DECISIONS.md D-24). A packed training
sample of run E is 33 tokens. The plan has run E read T as the second document
of one sample that holds T (plan 5.1), so T must fit inside a sample together
with the end of the document before it and the start of the one after it. German
costs the tokenizer more tokens per word than English: T's nine words are 21
tokens.

To count again: load the tokenizer from the snapshot folder that the spec's
`tokenizer.vocab_file` lies in, encode the text and add one for the
end-of-document token. `tests/replica/test_replica_fixture_texts.py` does this
in its test marked `megatron`. `B-001` and `T` are documents of shard-1 and
shard-3, and their lengths in those shards' index files are the same 16 and 22.

## Things a reader should know

- Every line of every file here is a readable record with a text of its own. The
  fixture's one skipped sanitizer line is not in this folder: the spec writes it,
  under `[sanitizer_skip]`, into the file the sanitizer reads for shard-2, as
  line 6. It is a line that does not parse, it belongs to no source and it is
  never ingested. The sanitizer skips it and logs the skip with the reason
  "sanitizer skip", so shard-2 still holds 19 documents and no ordinal shifts.
- `Z-046` is an ordinary passage like its neighbours. In an earlier state of the
  fixture it held two control characters and stood in for the skipped line.
- No text holds a control character or starts or ends with white space. The
  sanitizer leaves each of them unchanged.
- Apart from `A-001` and `B-001`, no two texts are equal after normalisation.
- The passages of Z differ in length on purpose, from 13 to 61 words. Twenty-one
  of them have 20 words or fewer. Packed training samples of 32 tokens then start and
  end inside documents in many different ways.
- A change to any text changes the token counts of the shards, and with them every
  read position the fixture spec records. Do not edit a text without repeating the
  seed search (plan 5.1; docs/oracle.md, "The search that fixed the spec's seeds").
  The seeds and positions in the spec belong to these texts as they are.
