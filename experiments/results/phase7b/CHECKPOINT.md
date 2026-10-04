# Phase 7B — Checkpoint

- **Phase**: 7B — High-Budget Fuzzing & Deep-State Discovery
- **Status**: PAUSED — WAITING FOR USAGE RESET
- **Total planned trials**: 35
- **Completed trials**: 35 / 35
- **Iteration budget**: 25,000 per trial (875,000 total executions)
- **RNG seeds**: 101, 202, 303, 404, 505
- **Executor**: persistent

## Configurations

| ID | Label | Seed Scheduler | Mutation Policy | Donor Selection |
|----|-------|---------------|-----------------|-----------------|
| A  | Random + Fixed | RandomScheduler | Fixed | — |
| B  | Heuristic + Fixed | HeuristicScheduler | Fixed | — |
| C  | LinearBandit + Fixed | LinearBanditScheduler | Fixed | — |
| D  | Random + UCB1 | RandomScheduler | UCB1 | — |
| E  | Random + Contextual | RandomScheduler | Contextual | — |
| F  | Random + Contextual + Compatible | RandomScheduler | Contextual | Compatible |
| G  | NeuroFuzz Full | LinearBanditScheduler | Contextual | Compatible |

## Output Files

| File | Status | Size |
|------|--------|------|
| `results.csv` | ✅ Complete (35 data rows) | 6,985 bytes |
| `discoveries.json` | ✅ Complete | 11,481 bytes |
| `summary.json` | ✅ Complete | 143,659 bytes |
| `trajectories/*.csv` | ✅ 35 trajectory files | All configs × all seeds |
| `discoveries/*/depth*.txt` | ✅ Discovery artifacts | Per-config depth breakthroughs |
| `plots/*.png` | ⚠️ Stale (from smoke test at 15:57) | 6 plot files — need regeneration |

**Total files on disk**: 113

## Last Observed Results (from results.csv)

| Config | Mean Coverage | Depth 15 Reached | Depth 16 Reached | Max Depth |
|--------|---------------|-------------------|-------------------|-----------|
| A: Random + Fixed | 62.2 | 3/5 | 2/5 | 16 |
| B: Heuristic + Fixed | 61.8 | 3/5 | 0/5 | 15 |
| C: LinearBandit + Fixed | 61.4 | 2/5 | 1/5 | 16 |
| D: Random + UCB1 | 62.0 | 3/5 | 0/5 | 15 |
| E: Random + Contextual | 61.6 | 3/5 | 0/5 | 15 |
| F: Random + Contextual + Compatible | 63.8 | 2/5 | 0/5 | 15 |
| G: NeuroFuzz Full | 63.2 | 2/5 | 1/5 | 16 |

## Launch Command

```
py -u experiments/phase7b_high_budget.py --iterations 25000 --seeds 101,202,303,404,505 --executor persistent
```

## Notes

- All 35 trials completed execution successfully.
- Task was canceled by the system during post-processing (plot generation / summary table printing), NOT during trial execution.
- The 6 plot PNG files under `plots/` are from the smoke test (1,000 iterations) and need to be regenerated from the full-experiment data.
- `results.csv`, `discoveries.json`, and `summary.json` contain the complete full-experiment data.
- No algorithms were modified during this experiment.
- Test suite: 242 tests passing at time of experiment launch.
