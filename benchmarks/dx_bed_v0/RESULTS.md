# dx_bed_v0 sample results

Generated 2026-10-18 16:21 by `python -m dxbench.bench.make_results` from `outputs/bench/*/summary.json`. Per-episode tables: `outputs/bench/<experiment>/REPORT.md`. Experiment definitions: `benchmarks/dx_bed_v0/experiments/<experiment>.yaml`.

| experiment | policy | task | difficulty | episodes | success | score mean | stage completion | stage time (s) | tipped | task failures | errors | settings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| act_bed03 | ACT (`act_bed03`) via server | BED-03 | easy | 10 | 0/10 (0%) | 0.00 | cloth_0 0%, cloth_1 0% | - | 1 | 0 | 0 | assist radius 5 cm, towel E 1e6 (before D-U11 / D-28) |
| act_bed03_100k | ACT (`act_bed03_100k`) via server | BED-03 | easy | 10 | 0/10 (0%) | 0.00 | cloth_0 0%, cloth_1 0% | cloth_0 36 | 0 | 0 | 0 | assist radius 5 cm, towel E 1e6 (before D-U11 / D-28) |
| hold_bed03 | hold via server | BED-03 | easy | 10 | 0/10 (0%) | 0.00 | cloth_0 0%, cloth_1 0% | - | 0 | 0 | 0 | assist radius 5 cm, towel E 1e6 (before D-U11 / D-28) |
| oracle_bed03 | Bed03OraclePolicy | BED-03 | easy | 10 | 8/10 (80%) | 0.90 | cloth_0 100%, cloth_1 80% | cloth_0 34, cloth_1 38 | 0 | 0 | 0 | assist radius 2.5 cm, towel E 1e5, oracle grasp check (D-U11 / D-28 / D-29) |
| oracle_bed03_r25a | Bed03OraclePolicy | BED-03 | easy | 10 | 7/10 (70%) | 0.85 | cloth_0 100%, cloth_1 70% | cloth_0 33, cloth_1 38 | 0 | 0 | 0 | assist radius 2.5 cm, towel E 1e6, oracle before grasp check (2026-10-18) |
| oracle_bed03_r50 | Bed03OraclePolicy | BED-03 | easy | 10 | 8/10 (80%) | 0.90 | cloth_0 100%, cloth_1 80% | cloth_0 34, cloth_1 38 | 0 | 0 | 0 | assist radius 5 cm (before D-U11), stage 3 run |
| replay_bed03 | replay of demo 0 via server | BED-03 | easy | 10 | 0/10 (0%) | 0.00 | cloth_0 0%, cloth_1 0% | - | 0 | 1 | 0 | assist radius 5 cm, towel E 1e6 (before D-U11 / D-28) |
