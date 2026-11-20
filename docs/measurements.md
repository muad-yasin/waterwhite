# The measured figures of plan 5.6

This page holds the figures that plan 5.6 asks for: what recording evidence costs, measured on the pipeline replica's generated sets. They come from one batch of the cell runner (`docs/replica.md`, section 17), made on 19 November 2026: 20 workloads, each built with one worker and with four, with capture off and on, in three paired runs. That is 240 timed runs. The batch also holds 6 runs of the confined harness, and a second pass of the same 240 runs measured peak memory. Each figure is written beside its workload and the number of runs it comes from. None of them says what the tool costs on another machine, another corpus or another size.

## The machine

Every record of the batch names the same machine: type `CPX62-shared-vCPU` (the name the caller gave the runner: a cloud server whose 16 cores are shared virtual CPUs), 32,853,516,288 bytes of memory (about 32.9 GB), Ubuntu 24.04.4 LTS, role "server". The records name the code tree `db0489709de10f782f519ebb1b7306188cee88b0`, which is the tree of this repository's first commit.

## Conditions

- **Shared virtual CPUs.** The operating system showed 16 virtual CPUs, and the caller named them shared in the type string. Activity from other tenants on the physical host was unknown. The batch set no window file, and no workload was run alone to serve as a reference.
- **Runs at once.** In the timed and memory passes, the runner launched one-worker builds in groups of six and four-worker builds in groups of three. The worker count and the number of builds launched together both changed. No workload ran alone as a reference. With one worker, the six runs of a workload (three pairs) were launched together: each record shows at most 5 other runs beside it. With four workers, the six runs of a workload ran as two groups of three: at most 2 other runs beside each. A figure at one worker set against a figure at four is not a comparison of the settings. The separate confined runs used groups of three at both worker settings.
- **Peak memory** comes from a second pass of the same 240 runs with the sampler on, in a folder of its own, at the same levels. A run with the sampler is left out of the median throughput, so the timed runs carry no peak. The pair files of the two passes name the same input hashes, shard order and tokenizer hash for all 120 pairs.
- **How three runs become one figure.** Throughput is the median of the three runs (`median_throughput`), and the lowest and highest run are written beside it. Query time and proof walk time are the median of the three, with the lowest and highest beside it; the docs define no statistic for them, and this is the choice made here. Peak memory is the greatest of the three, with the lowest beside it. A byte or count figure is one number when the three runs wrote the same, and the lowest and highest when they did not. The tables report each worker setting and capture mode separately.
- **Bytes move a little between runs.** The size of the build folder differs between the runs of one setting by at most 8 bytes with capture off and at most 11 bytes with capture on (38 and 38 of 40 settings differ). The store file never differs between the runs of a setting (0 of 40 differ). The bytes a repair reads differ by at most 11 bytes and the bytes it rewrites by at most 7. The metadata bytes per record, per parent link and per stored step trace are the same in all runs of a setting.
- **Nothing failed.** All 240 timed runs, all 6 confined runs and all 240 memory runs exited 0. The runner recorded no retries. Its check passed for every pair of capture-off and capture-on runs (120 in each pass): the two runs of a pair had the same input file hashes, the same shard order and the same tokenizer hash. Across the timed and memory passes, the pair files listed the same input hashes, shard order and tokenizer hash.

## Reading the tables

A row is one workload at one worker setting. On one server, the replica built 20 workloads generated with one seed. In each capture-off run, it built the workload without creating an evidence store. In each capture-on run, it also recorded evidence in a Waterwhite store. The store, metadata, query, proof walk and repair figures come from the capture-on runs. The `records` column counts source records, and `characters` gives the length of each one. A derivation is a generated record made from source records: a paraphrase, a translation of a paraphrase, or a summary. The `derivations` setting is 0, 1 or 2: the number of paraphrases, translations and summaries the generator makes for each passage it plants. The `sources` column counts the sources in the set. A shard holds part of the dataset. The `layout` column says how the build arranges records in shards. In the `partitioned` layout each source file is a shard of its own, and when the set has generated records they fill one more shard. In the `round-robin` layout the files that release R1 reads (`docs/replica.md`, section 4) are shuffled and dealt in turn over 20 chunks, so a record can land in any chunk. A cell written "a (b to c)" is a figure a with the lowest and highest of the three runs, b to c. A megabyte (MB) is 1,000,000 bytes. "Runs off/on" is the number of capture-off and capture-on runs the row's figures come from.

## Throughput

Each timed run measures one complete `build-set` invocation. Its throughput is the number of source records in that workload per second of elapsed build time. The table gives the median of three run rates for each capture mode. The figures come from the 240 timed runs.

| records | characters | derivations | sources | layout | workers | throughput, capture off (records/s) | throughput, capture on (records/s) | runs off/on |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 500 | 0 | 4 | partitioned | 1 | 435.7 (434.4 to 437.6) | 319.1 (315.1 to 322.0) | 3/3 |
| 10,000 | 500 | 0 | 4 | partitioned | 4 | 910.5 (909.5 to 923.9) | 523.3 (523.1 to 525.1) | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 1 | 168.1 (166.4 to 168.3) | 147.6 (147.2 to 148.0) | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 4 | 436.0 (435.5 to 436.4) | 321.2 (319.2 to 321.6) | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 1 | 343.9 (335.7 to 345.7) | 218.3 (217.4 to 221.8) | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 4 | 914.6 (909.3 to 927.0) | 362.8 (361.5 to 372.5) | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 1 | 154.5 (152.8 to 154.5) | 122.3 (122.1 to 122.7) | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 4 | 435.3 (435.0 to 436.0) | 253.5 (253.0 to 253.6) | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 1 | 342.9 (341.7 to 344.1) | 186.2 (183.7 to 186.2) | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 4 | 923.8 (912.9 to 939.6) | 293.4 (289.5 to 294.4) | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 1 | 150.7 (149.8 to 151.2) | 107.7 (107.3 to 108.1) | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 4 | 421.0 (416.1 to 431.4) | 205.8 (205.3 to 216.7) | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 1 | 391.3 (389.8 to 394.2) | 272.7 (270.4 to 272.9) | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 4 | 792.1 (790.4 to 797.9) | 420.4 (415.4 to 420.6) | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 1 | 160.9 (160.8 to 161.2) | 135.4 (135.4 to 135.5) | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 4 | 415.4 (414.5 to 425.2) | 286.5 (277.8 to 289.0) | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 1 | 315.6 (311.4 to 316.7) | 192.4 (191.7 to 192.9) | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 4 | 804.8 (792.2 to 831.7) | 297.7 (295.2 to 314.9) | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 1 | 147.2 (145.8 to 147.8) | 111.4 (110.9 to 112.0) | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 4 | 415.6 (412.6 to 420.8) | 222.6 (220.9 to 227.2) | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 1 | 315.7 (315.2 to 315.8) | 162.3 (161.5 to 162.5) | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 4 | 845.6 (833.8 to 846.3) | 235.6 (234.1 to 244.4) | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 1 | 149.1 (149.0 to 149.6) | 102.1 (101.8 to 102.9) | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 4 | 407.3 (400.5 to 418.2) | 178.8 (176.5 to 189.7) | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 1 | 305.1 (301.9 to 306.8) | 184.9 (183.1 to 185.0) | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 4 | 531.4 (525.2 to 533.5) | 253.9 (248.1 to 254.4) | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 1 | 146.9 (146.4 to 147.7) | 112.1 (111.7 to 112.5) | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 4 | 384.8 (381.2 to 386.6) | 207.4 (207.1 to 209.3) | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 1 | 250.1 (248.1 to 250.6) | 135.4 (133.7 to 136.8) | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 4 | 546.6 (526.2 to 554.0) | 188.8 (186.4 to 192.4) | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 1 | 129.6 (129.3 to 129.9) | 89.7 (89.7 to 89.9) | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 4 | 354.9 (352.5 to 359.4) | 164.4 (162.1 to 164.6) | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 1 | 237.5 (237.3 to 238.4) | 115.0 (114.9 to 115.3) | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 4 | 569.4 (560.9 to 582.7) | 157.6 (154.9 to 167.1) | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 1 | 128.3 (127.0 to 128.4) | 80.1 (79.8 to 80.3) | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 4 | 327.2 (326.0 to 335.9) | 131.8 (131.5 to 133.2) | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 1 | 78.2 (78.0 to 78.6) | 66.9 (66.5 to 67.0) | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 4 | 267.3 (266.8 to 267.8) | 168.5 (167.9 to 169.9) | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 1 | 1,562.6 (1,545.5 to 1,570.6) | 446.6 (446.1 to 448.0) | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 4 | 2,748.4 (2,697.5 to 2,772.8) | 522.4 (519.5 to 524.5) | 3/3 |

## Peak memory

In the second pass (240 runs), the sampler recorded, every 0.1 s, the sum of resident memory across the processes of the `build-set` command of each run: the command and every process below it. Each peak-memory column gives the highest sampled sum across its three runs. It is for one build and does not add the builds that ran beside it. The "largest single process" column gives, for each capture mode, the highest resident memory sampled for any one process across the three runs. The final column gives the run counts.

| records | characters | derivations | sources | layout | workers | peak memory, capture off (MB) | peak memory, capture on (MB) | largest single process, off / on (MB) | runs off/on |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 500 | 0 | 4 | partitioned | 1 | 1,367.2 (1,361.9 to 1,367.2) | 1,378.6 (955.0 to 1,378.6) | 472.6 / 551.0 | 3/3 |
| 10,000 | 500 | 0 | 4 | partitioned | 4 | 3,979.5 (3,503.0 to 3,979.5) | 3,562.6 (3,009.2 to 3,562.6) | 472.5 / 549.9 | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 1 | 1,366.6 (1,363.6 to 1,366.6) | 1,381.7 (1,376.8 to 1,381.7) | 472.7 / 554.6 | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 4 | 4,839.3 (4,420.5 to 4,839.3) | 5,227.9 (4,835.4 to 5,227.9) | 472.7 / 549.6 | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 1 | 1,365.7 (1,365.2 to 1,365.7) | 1,384.4 (1,381.8 to 1,384.4) | 472.7 / 544.5 | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 4 | 5,267.5 (4,438.2 to 5,267.5) | 5,299.1 (5,284.9 to 5,299.1) | 472.7 / 549.8 | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 1 | 1,366.7 (1,364.3 to 1,366.7) | 1,390.1 (1,382.9 to 1,390.1) | 472.6 / 548.5 | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 4 | 5,275.5 (4,429.1 to 5,275.5) | 5,308.1 (4,456.5 to 5,308.1) | 472.6 / 551.0 | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 1 | 1,367.6 (1,365.2 to 1,367.6) | 1,390.6 (1,384.9 to 1,390.6) | 472.7 / 557.6 | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 4 | 5,277.5 (4,848.6 to 5,277.5) | 5,310.3 (4,467.3 to 5,310.3) | 472.6 / 555.8 | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 1 | 1,367.7 (1,364.7 to 1,367.7) | 1,389.3 (1,385.4 to 1,389.3) | 472.6 / 556.3 | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 4 | 5,269.5 (4,823.6 to 5,269.5) | 5,270.1 (4,874.2 to 5,270.1) | 472.6 / 549.9 | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 1 | 1,409.3 (1,405.6 to 1,409.3) | 1,420.0 (1,417.0 to 1,420.0) | 472.7 / 549.4 | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 4 | 4,020.6 (4,012.4 to 4,020.6) | 4,024.5 (3,592.3 to 4,024.5) | 472.5 / 550.0 | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 1 | 1,408.7 (1,402.0 to 1,408.7) | 1,425.4 (1,417.9 to 1,425.4) | 472.6 / 549.9 | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 4 | 5,319.0 (4,484.2 to 5,319.0) | 5,337.7 (4,897.2 to 5,337.7) | 472.6 / 551.5 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 1 | 1,412.2 (1,404.5 to 1,412.2) | 1,422.8 (1,420.0 to 1,422.8) | 472.7 / 545.4 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 4 | 5,323.1 (4,891.4 to 5,323.1) | 5,342.3 (5,332.6 to 5,342.3) | 472.5 / 551.5 | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 1 | 1,409.0 (1,403.5 to 1,409.0) | 1,426.7 (1,422.2 to 1,426.7) | 472.6 / 549.8 | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 4 | 5,315.4 (4,884.3 to 5,315.4) | 5,333.1 (4,905.9 to 5,333.1) | 472.6 / 551.0 | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 1 | 1,406.1 (1,401.7 to 1,406.1) | 1,428.1 (1,421.8 to 1,428.1) | 472.6 / 558.4 | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 4 | 5,316.8 (5,269.4 to 5,316.8) | 5,339.4 (5,327.3 to 5,339.4) | 472.6 / 550.0 | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 1 | 1,410.1 (1,405.4 to 1,410.1) | 1,428.4 (1,426.0 to 1,428.4) | 472.6 / 550.0 | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 4 | 5,308.9 (5,286.7 to 5,308.9) | 5,335.2 (5,295.5 to 5,335.2) | 472.6 / 549.9 | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 1 | 1,550.5 (1,548.5 to 1,550.5) | 1,564.8 (1,562.6 to 1,564.8) | 472.6 / 558.3 | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 4 | 4,163.6 (4,155.9 to 4,163.6) | 4,176.1 (4,173.2 to 4,176.1) | 472.5 / 550.2 | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 1 | 1,554.6 (1,554.4 to 1,554.6) | 1,570.9 (1,569.0 to 1,570.9) | 472.6 / 549.7 | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 4 | 5,473.0 (5,465.3 to 5,473.0) | 5,487.5 (5,484.2 to 5,487.5) | 472.6 / 549.8 | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 1 | 1,565.5 (1,563.4 to 1,565.5) | 1,577.9 (1,577.2 to 1,577.9) | 472.7 / 545.0 | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 4 | 5,485.0 (5,479.6 to 5,485.0) | 5,492.6 (5,486.9 to 5,492.6) | 472.6 / 551.1 | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 1 | 1,556.1 (1,554.6 to 1,556.1) | 1,576.6 (1,572.2 to 1,576.6) | 472.6 / 558.3 | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 4 | 5,472.6 (5,465.6 to 5,472.6) | 5,490.5 (5,488.4 to 5,490.5) | 472.7 / 549.9 | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 1 | 1,559.1 (1,557.2 to 1,559.1) | 1,582.6 (1,576.4 to 1,582.6) | 472.6 / 550.2 | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 4 | 5,475.7 (5,473.2 to 5,475.7) | 5,500.8 (5,493.1 to 5,500.8) | 472.6 / 549.6 | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 1 | 1,562.4 (1,559.0 to 1,562.4) | 1,582.1 (1,579.1 to 1,582.1) | 472.6 / 558.3 | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 4 | 5,470.8 (5,468.0 to 5,470.8) | 5,498.1 (5,496.2 to 5,498.1) | 472.6 / 551.1 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 1 | 1,403.9 (1,402.7 to 1,403.9) | 1,427.0 (1,421.7 to 1,427.0) | 472.7 / 549.9 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 4 | 5,312.1 (5,309.7 to 5,312.1) | 4,913.7 (4,430.6 to 4,913.7) | 472.7 / 558.3 | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 1 | 1,989.1 (1,987.1 to 1,989.1) | 2,017.1 (2,016.5 to 2,017.1) | 683.5 / 741.9 | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 4 | 5,912.1 (5,899.5 to 5,912.1) | 5,934.5 (5,924.7 to 5,934.5) | 683.5 / 742.0 | 3/3 |

## Disk and metadata

The table gives the build-folder size for each capture mode and the store-file size with capture on. Each metadata column uses the stored row count for that kind of metadata. For the workload of 10,000 source records, 2,000 characters, one derivation, four sources and the partitioned layout, the store contains 29,398 record rows. The build-folder total excludes the store file. The row counts are the same in every run of a setting. A capture-off run has no store, so those columns have no capture-off figure.

| records | characters | derivations | sources | layout | workers | build folder, off (bytes) | build folder, on (bytes) | store file, on (bytes) | per record (bytes) | per parent link (bytes) | per step trace (bytes) | rows: records / parent links / step traces | runs off/on |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 500 | 0 | 4 | partitioned | 1 | 20,981,097 to 20,981,099 | 20,984,777 to 20,984,780 | 7,311,360 | 77.61 | 27.91 | 289.19 | 28,288 / 19,076 / 3,980 | 3/3 |
| 10,000 | 500 | 0 | 4 | partitioned | 4 | 20,981,096 to 20,981,098 | 20,984,776 to 20,984,781 | 7,311,360 | 77.61 | 27.91 | 289.19 | 28,288 / 19,076 / 3,980 | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 1 | 21,026,096 to 21,026,100 | 21,032,011 to 21,032,014 | 7,311,360 | 77.18 | 27.91 | 290.51 | 28,288 / 19,076 / 3,976 | 3/3 |
| 10,000 | 500 | 0 | 10 | partitioned | 4 | 21,026,095 to 21,026,098 | 21,032,010 to 21,032,012 | 7,311,360 | 77.18 | 27.91 | 290.51 | 28,288 / 19,076 / 3,976 | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 1 | 22,214,698 to 22,214,701 | 22,931,085 to 22,931,093 | 8,273,920 | 77.75 | 27.57 | 287.74 | 29,398 / 21,396 / 4,783 | 3/3 |
| 10,000 | 500 | 1 | 4 | partitioned | 4 | 22,214,701 to 22,214,703 | 22,931,089 to 22,931,090 | 8,273,920 | 77.75 | 27.57 | 287.74 | 29,398 / 21,396 / 4,783 | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 1 | 22,250,143 to 22,250,145 | 22,968,754 to 22,968,761 | 8,261,632 | 77.34 | 27.77 | 285.47 | 29,392 / 21,390 / 4,778 | 3/3 |
| 10,000 | 500 | 1 | 10 | partitioned | 4 | 22,250,140 to 22,250,144 | 22,968,757 to 22,968,759 | 8,261,632 | 77.34 | 27.77 | 285.47 | 29,392 / 21,390 / 4,778 | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 1 | 23,433,328 to 23,433,332 | 24,856,296 to 24,856,299 | 9,232,384 | 78.03 | 28.00 | 285.49 | 30,498 / 23,696 / 5,581 | 3/3 |
| 10,000 | 500 | 2 | 4 | partitioned | 4 | 23,433,330 to 23,433,331 | 24,856,295 to 24,856,296 | 9,232,384 | 78.03 | 28.00 | 285.49 | 30,498 / 23,696 / 5,581 | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 1 | 23,461,572 to 23,461,575 | 24,886,744 to 24,886,747 | 9,203,712 | 76.97 | 27.84 | 285.12 | 30,492 / 23,690 / 5,574 | 3/3 |
| 10,000 | 500 | 2 | 10 | partitioned | 4 | 23,461,571 to 23,461,575 | 24,886,737 to 24,886,744 | 9,203,712 | 76.97 | 27.84 | 285.12 | 30,492 / 23,690 / 5,574 | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 1 | 72,817,455 to 72,817,457 | 72,821,136 to 72,821,139 | 10,313,728 | 77.47 | 27.91 | 268.24 | 28,288 / 19,076 / 14,980 | 3/3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 4 | 72,817,455 to 72,817,456 | 72,821,134 to 72,821,139 | 10,313,728 | 77.47 | 27.91 | 268.24 | 28,288 / 19,076 / 14,980 | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 1 | 72,853,910 to 72,853,912 | 72,859,823 to 72,859,824 | 10,305,536 | 77.32 | 27.91 | 267.23 | 28,288 / 19,076 / 14,975 | 3/3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 4 | 72,853,910 to 72,853,914 | 72,859,825 to 72,859,826 | 10,305,536 | 77.32 | 27.91 | 267.23 | 28,288 / 19,076 / 14,975 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 1 | 76,969,988 | 77,686,377 | 11,870,208 | 77.33 | 27.57 | 268.43 | 29,398 / 21,396 / 17,945 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 4 | 76,969,984 to 76,969,987 | 77,686,374 to 77,686,377 | 11,870,208 | 77.33 | 27.57 | 268.43 | 29,398 / 21,396 / 17,945 | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 1 | 76,981,079 to 76,981,080 | 77,699,693 to 77,699,698 | 11,833,344 | 76.51 | 27.77 | 267.25 | 29,392 / 21,390 / 17,932 | 3/3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 4 | 76,981,078 to 76,981,083 | 77,699,692 to 77,699,696 | 11,833,344 | 76.51 | 27.77 | 267.25 | 29,392 / 21,390 / 17,932 | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 1 | 81,090,468 | 82,513,428 to 82,513,432 | 13,406,208 | 77.09 | 28.00 | 267.89 | 30,498 / 23,696 / 20,901 | 3/3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 4 | 81,090,465 to 81,090,469 | 82,513,434 | 13,406,208 | 77.09 | 28.00 | 267.89 | 30,498 / 23,696 / 20,901 | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 1 | 81,097,799 to 81,097,803 | 82,522,972 to 82,522,975 | 13,385,728 | 76.43 | 27.84 | 267.67 | 30,492 / 23,690 / 20,888 | 3/3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 4 | 81,097,800 to 81,097,806 | 82,522,971 to 82,522,973 | 13,385,728 | 76.43 | 27.84 | 267.67 | 30,492 / 23,690 / 20,888 | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 1 | 280,245,176 to 280,245,178 | 280,249,057 to 280,249,061 | 22,900,736 | 77.61 | 27.91 | 271.46 | 28,288 / 19,076 / 58,997 | 3/3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 4 | 280,245,176 to 280,245,178 | 280,249,058 to 280,249,059 | 22,900,736 | 77.61 | 27.91 | 271.46 | 28,288 / 19,076 / 58,997 | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 1 | 280,286,708 to 280,286,713 | 280,292,825 to 280,292,829 | 22,798,336 | 77.32 | 27.91 | 269.73 | 28,288 / 19,076 / 58,996 | 3/3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 4 | 280,286,708 to 280,286,714 | 280,292,826 to 280,292,829 | 22,798,336 | 77.32 | 27.91 | 269.73 | 28,288 / 19,076 / 58,996 | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 1 | 296,174,296 to 296,174,299 | 296,890,886 to 296,890,892 | 27,029,504 | 77.47 | 27.57 | 272.69 | 29,398 / 21,396 / 70,658 | 3/3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 4 | 296,174,298 to 296,174,301 | 296,890,887 to 296,890,889 | 27,029,504 | 77.47 | 27.57 | 272.69 | 29,398 / 21,396 / 70,658 | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 1 | 296,134,330 to 296,134,333 | 296,853,143 to 296,853,146 | 26,959,872 | 76.79 | 27.77 | 271.97 | 29,392 / 21,390 / 70,619 | 3/3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 4 | 296,134,330 to 296,134,332 | 296,853,142 to 296,853,146 | 26,959,872 | 76.79 | 27.77 | 271.97 | 29,392 / 21,390 / 70,619 | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 1 | 311,982,409 to 311,982,410 | 313,405,569 to 313,405,572 | 31,256,576 | 77.63 | 28.00 | 274.71 | 30,498 / 23,696 / 82,260 | 3/3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 4 | 311,982,409 to 311,982,410 | 313,405,572 to 313,405,574 | 31,256,576 | 77.63 | 28.00 | 274.71 | 30,498 / 23,696 / 82,260 | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 1 | 311,934,193 to 311,934,196 | 313,359,562 to 313,359,567 | 31,109,120 | 76.70 | 27.84 | 273.30 | 30,492 / 23,690 / 82,219 | 3/3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 4 | 311,934,192 to 311,934,196 | 313,359,559 to 313,359,564 | 31,109,120 | 76.70 | 27.84 | 273.30 | 30,492 / 23,690 / 82,219 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 1 | 77,006,440 to 77,006,443 | 77,728,904 to 77,728,910 | 11,845,632 | 77.47 | 27.57 | 264.80 | 29,398 / 21,396 / 17,943 | 3/3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 4 | 77,006,435 to 77,006,442 | 77,728,905 to 77,728,907 | 11,845,632 | 77.47 | 27.57 | 264.80 | 29,398 / 21,396 / 17,943 | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 1 | 786,182,847 to 786,182,849 | 793,354,718 to 793,354,728 | 125,243,392 | 81.72 | 30.90 | 286.03 | 299,488 / 216,876 / 184,389 | 3/3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 4 | 786,182,835 to 786,182,843 | 793,354,717 to 793,354,728 | 125,243,392 | 81.72 | 30.90 | 286.03 | 299,488 / 216,876 / 184,389 | 3/3 |

## Query, proof walk and repair

The query times and proof walk times in this section are the first commit's figures. A later commit changed the statement of the store that finds the trace rows of one document; these times were not measured again with it.

Capture-on runs only: a capture-off run has no store to query or repair. The query times `waterwhite query withdraw A --artifact probe@N` for one artifact. `probe@N` is the checkpoint the probe run registers at its last step, and N is the number of steps that run reports, so it differs between workloads. The proof walk times `waterwhite query withdraw A` across the store. The replica runs the repair on a copy of the store. It counts the historical build size and the starting store size as bytes read. It counts the repaired build size and the increase in store size as bytes rewritten. These are sizes of files, not measured disk I/O.

| records | characters | derivations | sources | layout | workers | query time, on (s) | proof walk time, on (s) | repair reads (bytes) | repair rewrites (bytes) | runs on |
|---|---|---|---|---|---|---|---|---|
| 10,000 | 500 | 0 | 4 | partitioned | 1 | 0.90 (0.84 to 0.90) | 1.10 (1.03 to 1.10) | 28,296,137 to 28,296,140 | 24,777,049 to 24,777,054 | 3 |
| 10,000 | 500 | 0 | 4 | partitioned | 4 | 0.82 (0.81 to 0.90) | 1.10 (1.02 to 1.10) | 28,296,136 to 28,296,141 | 24,777,051 to 24,777,053 | 3 |
| 10,000 | 500 | 0 | 10 | partitioned | 1 | 0.87 (0.87 to 0.90) | 1.23 (1.20 to 1.27) | 28,343,371 to 28,343,374 | 24,804,367 to 24,804,370 | 3 |
| 10,000 | 500 | 0 | 10 | partitioned | 4 | 0.87 (0.84 to 0.87) | 1.20 (1.20 to 1.22) | 28,343,370 to 28,343,372 | 24,804,364 to 24,804,369 | 3 |
| 10,000 | 500 | 1 | 4 | partitioned | 1 | 1.37 (1.37 to 1.40) | 1.57 (1.57 to 1.60) | 31,205,005 to 31,205,013 | 23,266,656 to 23,266,657 | 3 |
| 10,000 | 500 | 1 | 4 | partitioned | 4 | 1.37 (1.31 to 1.40) | 1.57 (1.54 to 1.60) | 31,205,009 to 31,205,010 | 23,266,654 to 23,266,658 | 3 |
| 10,000 | 500 | 1 | 10 | partitioned | 1 | 1.40 (1.30 to 1.50) | 1.67 (1.64 to 1.71) | 31,230,386 to 31,230,393 | 23,292,009 to 23,292,010 | 3 |
| 10,000 | 500 | 1 | 10 | partitioned | 4 | 1.37 (1.31 to 1.54) | 1.67 (1.67 to 1.70) | 31,230,389 to 31,230,391 | 23,292,009 to 23,292,010 | 3 |
| 10,000 | 500 | 2 | 4 | partitioned | 1 | 1.97 (1.92 to 2.01) | 2.10 (2.10 to 2.21) | 34,088,680 to 34,088,683 | 21,710,907 to 21,710,910 | 3 |
| 10,000 | 500 | 2 | 4 | partitioned | 4 | 1.91 (1.90 to 2.00) | 2.17 (2.17 to 2.17) | 34,088,679 to 34,088,680 | 21,710,909 to 21,710,910 | 3 |
| 10,000 | 500 | 2 | 10 | partitioned | 1 | 2.07 (2.07 to 2.10) | 2.37 (2.37 to 2.38) | 34,090,456 to 34,090,459 | 21,770,331 to 21,770,333 | 3 |
| 10,000 | 500 | 2 | 10 | partitioned | 4 | 2.04 (2.00 to 2.11) | 2.37 (2.37 to 2.40) | 34,090,449 to 34,090,456 | 21,770,330 to 21,770,334 | 3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 1 | 1.12 (1.12 to 1.13) | 1.37 (1.33 to 1.41) | 83,134,864 to 83,134,867 | 76,621,601 to 76,621,602 | 3 |
| 10,000 | 2,000 | 0 | 4 | partitioned | 4 | 1.07 (1.07 to 1.10) | 1.40 (1.30 to 1.50) | 83,134,862 to 83,134,867 | 76,621,599 to 76,621,600 | 3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 1 | 1.17 (1.17 to 1.17) | 1.53 (1.50 to 1.54) | 83,165,359 to 83,165,360 | 76,636,277 to 76,636,279 | 3 |
| 10,000 | 2,000 | 0 | 10 | partitioned | 4 | 1.10 (1.07 to 1.22) | 1.60 (1.43 to 1.60) | 83,165,361 to 83,165,362 | 76,636,277 to 76,636,278 | 3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 1 | 1.90 (1.87 to 1.91) | 2.11 (2.07 to 2.14) | 89,556,585 | 70,628,518 to 70,628,520 | 3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 4 | 1.84 (1.81 to 2.10) | 2.11 (2.07 to 2.17) | 89,556,582 to 89,556,585 | 70,628,517 to 70,628,521 | 3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 1 | 1.90 (1.87 to 1.97) | 2.27 (2.27 to 2.37) | 89,533,037 to 89,533,042 | 70,658,196 to 70,658,198 | 3 |
| 10,000 | 2,000 | 1 | 10 | partitioned | 4 | 1.84 (1.84 to 1.92) | 2.21 (2.14 to 2.24) | 89,533,036 to 89,533,040 | 70,658,195 to 70,658,198 | 3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 1 | 2.77 (2.71 to 2.87) | 2.97 (2.92 to 3.21) | 95,919,636 to 95,919,640 | 64,625,234 to 64,625,239 | 3 |
| 10,000 | 2,000 | 2 | 4 | partitioned | 4 | 2.64 (2.62 to 2.92) | 2.91 (2.84 to 2.97) | 95,919,642 | 64,625,236 to 64,625,237 | 3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 1 | 2.80 (2.72 to 2.81) | 3.11 (3.07 to 3.61) | 95,908,700 to 95,908,703 | 64,655,556 to 64,655,559 | 3 |
| 10,000 | 2,000 | 2 | 10 | partitioned | 4 | 2.77 (2.71 to 2.82) | 3.07 (3.07 to 3.07) | 95,908,699 to 95,908,701 | 64,655,558 to 64,655,561 | 3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 1 | 1.87 (1.83 to 1.92) | 2.27 (2.27 to 2.30) | 303,149,793 to 303,149,797 | 284,033,138 to 284,033,139 | 3 |
| 10,000 | 8,000 | 0 | 4 | partitioned | 4 | 1.90 (1.84 to 1.91) | 2.27 (2.21 to 2.30) | 303,149,794 to 303,149,795 | 284,033,137 to 284,033,138 | 3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 1 | 1.87 (1.87 to 2.01) | 2.37 (2.37 to 2.44) | 303,091,161 to 303,091,165 | 284,097,950 | 3 |
| 10,000 | 8,000 | 0 | 10 | partitioned | 4 | 1.87 (1.87 to 1.90) | 2.47 (2.37 to 2.47) | 303,091,162 to 303,091,165 | 284,097,949 to 284,097,951 | 3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 1 | 2.84 (2.81 to 2.87) | 3.21 (3.17 to 3.21) | 323,920,390 to 323,920,396 | 260,190,506 to 260,190,510 | 3 |
| 10,000 | 8,000 | 1 | 4 | partitioned | 4 | 2.81 (2.71 to 2.87) | 3.31 (3.11 to 3.31) | 323,920,391 to 323,920,393 | 260,190,506 to 260,190,507 | 3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 1 | 2.81 (2.71 to 2.84) | 3.42 (3.41 to 3.54) | 323,813,015 to 323,813,018 | 260,140,448 to 260,140,451 | 3 |
| 10,000 | 8,000 | 1 | 10 | partitioned | 4 | 2.84 (2.84 to 2.87) | 3.27 (3.21 to 3.31) | 323,813,014 to 323,813,018 | 260,140,450 to 260,140,454 | 3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 1 | 4.18 (4.02 to 4.31) | 4.31 (4.27 to 4.31) | 344,662,145 to 344,662,148 | 236,169,073 to 236,169,075 | 3 |
| 10,000 | 8,000 | 2 | 4 | partitioned | 4 | 4.04 (3.97 to 4.04) | 4.41 (4.37 to 4.44) | 344,662,148 to 344,662,150 | 236,169,072 to 236,169,073 | 3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 1 | 4.24 (4.17 to 4.31) | 4.71 (4.61 to 4.71) | 344,468,682 to 344,468,687 | 236,123,366 to 236,123,368 | 3 |
| 10,000 | 8,000 | 2 | 10 | partitioned | 4 | 4.11 (4.04 to 4.27) | 4.64 (4.61 to 4.67) | 344,468,679 to 344,468,684 | 236,123,367 to 236,123,370 | 3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 1 | 1.91 (1.84 to 1.91) | 2.03 (2.00 to 2.17) | 89,574,536 to 89,574,542 | 70,684,164 to 70,684,167 | 3 |
| 10,000 | 2,000 | 1 | 4 | round-robin | 4 | 1.87 (1.84 to 1.92) | 1.91 (1.90 to 1.97) | 89,574,537 to 89,574,539 | 70,684,160 to 70,684,167 | 3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 1 | 101.27 (100.37 to 103.71) | 117.50 (116.83 to 118.33) | 918,598,110 to 918,598,120 | 724,030,273 to 724,030,277 | 3 |
| 100,000 | 2,000 | 1 | 4 | partitioned | 4 | 101.46 (100.92 to 102.28) | 116.53 (114.16 to 118.74) | 918,598,109 to 918,598,120 | 724,030,276 to 724,030,281 | 3 |

## The confined harness

The workload of 10,000 records, 2,000 characters, 1 derivation, 4 sources and the partitioned layout was also run under the confined harness, with capture on, in three runs per worker setting and three at once. It is a separate figure and not one of the 240. These runs were not sampled, so they have no peak memory.

| records | characters | derivations | sources | layout | workers | throughput (records/s) | build folder (bytes) | store file (bytes) | per record / parent link / step trace (bytes) | query time (s) | proof walk time (s) | repair reads / rewrites (bytes) | peak memory | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10,000 | 2,000 | 1 | 4 | partitioned | 1 | 78.8 (78.7 to 78.9) | 77,684,204 to 77,684,206 | 11,870,208 | 77.33 / 27.57 / 268.43 | 1.91 (1.77 to 2.01) | 2.01 (2.00 to 2.14) | 89,554,412 to 89,554,414 / 70,626,344 to 70,626,346 | not measured | 3 |
| 10,000 | 2,000 | 1 | 4 | partitioned | 4 | 95.3 (94.7 to 95.4) | 77,684,202 to 77,684,205 | 11,870,208 | 77.33 / 27.57 / 268.43 | 1.87 (1.81 to 1.91) | 2.07 (2.00 to 2.11) | 89,554,410 to 89,554,413 / 70,626,345 to 70,626,349 | not measured | 3 |

## Capture wiring

One static figure, not one per workload: capture is wired into one generation step by 33 lines of `StoreCapture` in `replica/capture_hooks.py` and by 2 calls in `replica/build.py`, at lines 449 and 465. The batch folders of the timed runs, the confined runs and the second pass hold the same figure.

## What has no figure

| figure | capture off | capture on |
|---|---|---|
| throughput | measured | measured |
| peak memory | measured, second pass | measured, second pass |
| build folder | measured | measured |
| store file | none: no store | measured |
| metadata bytes | none: no store | measured |
| query time | none: no store | measured |
| proof walk time | none: no store | measured |
| repair bytes | none: no store | measured |

The confined runs have no peak memory. Wall time is in every record and is not tabled: throughput is made from it.

## Where the figures come from

The run records are not in this repository, and neither is the script that computed these tables from them. The script wrote every number above; none was typed by hand. The 726 files consist of 240 timed grid run records, 6 confined run records, 240 run records from the memory pass and 240 pair files (120 from each pass). The SHA-256 digest of the sorted list of file hashes and paths, with a newline after each entry, is `7bb222b00a225bf587a6059e340d5794bb611d041c1a33f4572867d1fbd279c6`. An entry is the sha256 of a file, two spaces, and its name as `folder/records/name` or `folder/pairs/name`, the folders being `timed`, `confined` and `sampler`.
