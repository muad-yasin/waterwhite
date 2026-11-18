# Progress

- Milestone: M0 (fixture, oracle reading and the thinnest end-to-end path).
- Built (BUILT-LOG.md has one line per commit):
  - the interfaces: vocabulary, named errors, ids, record types, the store schema,
    the extension switches, docs/interfaces.md;
  - the hand-checked fixture: its texts, the completed spec with the seeds and every
    read position, the hand-written table for Checks 1 to 9 and the reading rules
    of Check 10;
  - the plant generator and the counting script for the generated sets;
  - the test-time oracle and the training stand-in, with the seed search;
  - the evidence store with its loud checks;
  - the capture wrappers for ingest, generation, selection and the text-only
    import, the normaliser and the harness;
  - the pipeline replica in both layouts, and the captured fixture build that
    registers the source files, the generation outputs, the releases and the
    import. Its hook points for tokenization maps, mixture snapshots and run
    registration hold no code yet;
  - the coverage proof, the four answers, withdraw requests and their refusals,
    listings, and the commands init, query, show and check;
  - the acceptance tests, which put the table rows of Checks 5 and 7 to the tool
    through the command line;
  - the citation check (tools/check_plan_citations.py).
- Last checks:
  - the fast suite: see the test run of the commit;
  - the Megatron-marked tests: see the test run of the commit;
  - the acceptance rows (tests/acceptance, described in its README): Check 5 has
    221 rows, of which 35 ran and passed and 186 are skipped; Check 7 has 27 rows,
    of which 8 ran and passed and 19 are skipped. No row is an expected failure.
    One row that ran, `c07.withdraw-A.R2`, holds a field that is not compared yet,
    the action. A skipped row is not a pass: its target is not built yet or its
    extension is off. pytest counts 80 tests passed and 206 skipped in
    tests/acceptance, and 13 passed in tests/table.
- Next step: the capture wrappers for tokenization, mixtures and runs; recovery
  without capture; Checks 3 and 6.
