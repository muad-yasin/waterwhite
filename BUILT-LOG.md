# Built log

One line per commit: what it built and what it serves (the plan sections the commit
names, and the debate proposal id where the plan gives one).

- c30f0a4 Interfaces: answer vocabulary and named causes, named errors with exit codes, ids, record types, the store schema, the six extension switches, docs/interfaces.md, twelve entries of DECISIONS.md, the Megatron environment script (plan 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.4, 3.5, 4.1, 1.4, 5.3, 5.8, L7, F5, F6, Decisions 11 and 29; DEEPSEEKV41FLASH-1 for the store schema).
- acbd349 Fixture spec skeleton (plan 5.1) and DECISIONS.md D-13: Check 8 gets a row for a document read only through a sample's shared last token (plan 3.5, Decision 18).
- 397ae65 Hand-written table for Checks 1 to 9 with its README and fault catalogue, and the completed fixture spec (plan 5.1, 5.5, W5).
- ed80332 Plant generator and counting script for the generated sets, docs/plant.md, tests/plant (plan 5.1, 4.4, W5).
- 66a2e41 Test-time oracle and training stand-in: the index build, the consumer log for both samplers, the run record and step record, the replay tool of Check 2, the F1 probe (plan 3.5, 5.3, W5).
- 674a5f8 Evidence store: one SQLite file, transactional writes, the label rule, instance identity, typed read methods, the loud checks (plan 3.4, Decisions 2 and 29; DEEPSEEKV41FLASH-1).
- 44b433d Capture wrappers, the control-chars/1 normaliser and the in-process harness (plan 3.1, 3.3, 5.4, F8, F10).
- 7003e47 Fixture texts and the pipeline replica in both layouts (plan 5.1, 5.2, W3).
- 8941f0e DECISIONS.md D-14 to D-22, from the first integration of the modules.
- 1b16575 DECISIONS.md D-23: paraphrases of unrelated records in the generated sets (a marked addition to plan 5.1, ledger id BF-1).
- 43c59a2 Check 10 reading rules in fixture/table/check10.toml; the table, the spec and the plant scripts follow D-16 to D-22 (plan 5.5 Check 10, W5).
- 40df205 Oracle revision: whole-epoch shuffle digests (D-16), one command for every run of the fixture spec, the seed search (plan 5.1, 3.5, 5.3).
- fb76c81 Store and capture revision after D-14 to D-22; tests/integration runs the store, the capture wrappers and the harness together.
- 2fda03a Replica revision and the first captured fixture build in the real store (plan 5.4; D-17).
- e375144 Coverage proof, the four answers, withdraw requests and their refusals, listings, the commands init, query, show and check (plan 2.1 to 2.4; SONNET55-1).
- e80df6f Coverage tests: the parameter that named a removed reason is dropped (plan 3.5).
- 6e7df8a DECISIONS.md D-24: amendments after the second integration.
- 59a1178 Plant sets: the paraphrases of unrelated records that stay (D-23).
- 0357353 Replica tests: the counts of the 800-record set after D-23.
- 0334073 Citation check: a record locator is not a plan citation.
- 496441f Final fixture: short X, P and T, the seeds, every position filled in; the rows of Checks 1 and 8 follow; tests of the table's structure and of the spec's quoted log lines (D-24).
- afb7fc0 Store and proof seam: job names on edges, the reader's order of maps, full listings, six faults closed with a test each (plan 2.2; D-24).
- 4def68f Acceptance tests: the table rows of Checks 5 and 7 graded through the command line; the replica builds the table's stores main, swap and import.
- 4f1eb84 show prints an artifact's edge digest, so the instance row of Check 7 is compared through the command line (plan 3.1).
- 01b2ea8 DECISIONS.md D-25: amendments after the third integration; section 9 of docs/interfaces.md follows the final fixture.
- 2fb9bae Table: every core row in a Check 10 group, and the second wrong report of Check 9 (D-23).
- f9e78d5 Citation check: table plan fields, "plan Section N" and entries of DECISIONS.md.
- 6ec6b9f Integrity: a binding's declared count is compared with the file's records when the file's digest is the registered one; DECISIONS.md D-20 to D-26 (plan 2.2(d), 3.4).
- 2791e35 Docs and citations: anchors corrected, progress and built log current; the citation check reads more forms (plan 2.3, 5.8).
- d45d6ee Docs: the added refusals and the import marked in the files left over (plan 2.3, 5.4, 5.8).
- eb1743c LICENSE: the copyright holder is Sower Industries.
- 5609fdc Confined harness: a generation job run out of process under bubblewrap; DECISIONS.md D-27 (plan 3.3, 5.8, F10).
- a82dac7 Plant counts for Checks 3 and 6: documents, drops, translations (plan 5.1, 5.5).
- 72bec16 Tokenization maps: capture wrapper and recovery in two modes; DECISIONS.md D-28 (plan 3.3, Decision 4, F1).
- 5d67fe6 Recovery of copy steps and of the sanitizer without capture; DECISIONS.md D-29 (plan 3.3, 5.4, L3).
- 2a8e701 Tool seams: tokenization maps, recovery and the confined harness joined; schema version 2 and the migrate command; DECISIONS.md D-30 (plan 3.3, 3.4).
- 4d8ddec Decision D-26: a file that is gone leaves the answer standing.
- fd46a9c Confined harness: a job reaches no source file and no store file of its build (plan 3.3, F10, Decision 13).
- 8362e80 Table: Check 6's import rows split between the answer and the report; DECISIONS.md D-31 (plan 2.1, 3.2, 4.1, L1).
- 8aa149f show: a named map's rows, drops on every artifact, the eod settings (plan 3.3, 3.4, F1).
- f4e0cf7 Replica: capture of shards and mixtures, the run made without capture, table stores; DECISIONS.md D-32 (plan 5.4, 3.3, Checks 3 and 6).
- 2ad9ba8 Extensions off: a request of one is refused before any path of it runs (plan 1.4, 2.3, L7).
- 6498f6c Acceptance: Checks 3 and 6 graded through the command line (plan 5.5).
