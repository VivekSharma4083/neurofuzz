# NeuroFuzz Experimental Artifact Index

This document provides the authoritative index of all major controlled empirical evaluations conducted during the NeuroFuzz research project.

---

## Phase 7B: High-Budget Evaluation & Discovery Plateau Analysis

- **Purpose**: Evaluate whether scaling execution budgets from 500 to 25,000 iterations allows conventional and bandit fuzzers to penetrate sequential protocol states.
- **Experiment Script**: `experiments/high_budget_experiment.py`
- **Result Directory**: `experiments/results/phase7b/`
- **Formal Analysis Report**: [`experiments/results/phase7b/PHASE7B_FINAL_ANALYSIS.md`](file:///c:/Users/HP/Desktop/neurofuzz/experiments/results/phase7b/PHASE7B_FINAL_ANALYSIS.md)
- **Experimental Scale**: 7 configurations × 5 random seeds = 35 trials (875,000 total executions)
- **Major Finding**: Execution scale alone cannot overcome sequential grammar barriers. All 7 configurations saturated around Depth 16 ($0\%$ reach rate for Depth 17–19). Proved that the sequential state wall was algorithmic rather than throughput-limited.

---

## Phase 7C: State-Sequence-Aware Frontier Exploration

- **Purpose**: Validate the `StateFrontier` mechanism, which targets terminal protocol delimiters with grammar tokens to cross sequential barriers.
- **Experiment Script**: `experiments/phase7c_experiment.py`
- **Result Directory**: `experiments/results/phase7c/`
- **Formal Analysis Report**: [`experiments/results/phase7c/PHASE7C_FINAL_ANALYSIS.md`](file:///c:/Users/HP/Desktop/neurofuzz/experiments/results/phase7c/PHASE7C_FINAL_ANALYSIS.md)
- **Experimental Scale**: 4 configurations × 5 random seeds = 20 trials (500,000 total executions)
- **Major Finding**: Enabling `StateFrontier` achieved a **100% reach rate for Depth 19** in under 350 iterations on all tested seeds, increasing maximum coverage from 64 to 72 units and breaking the sequential bottleneck.

---

## Phase 7D: Adaptive Frontier Allocation & Delimiter Generalization

- **Purpose**: Evaluate dynamic frontier rate decay after state saturation, and assess whether runtime delimiter inference eliminates manual configuration.
- **Experiment Script**: `experiments/phase7d_experiment.py`
- **Result Directory**: `experiments/results/phase7d/`
- **Formal Analysis Report**: [`experiments/results/phase7d/PHASE7D_FINAL_ANALYSIS.md`](file:///c:/Users/HP/Desktop/neurofuzz/experiments/results/phase7d/PHASE7D_FINAL_ANALYSIS.md)
- **Experimental Scale**: 4 configurations × 5 random seeds = 20 trials (500,000 total executions)
- **Major Finding**: Adaptive frontier allocation decayed mutation probability from 20% to the 2% floor after Depth 19 discovery, saving $4,480$ executions per trial without sacrificing coverage. Runtime delimiter discovery identified `b'|'` with 100% accuracy, achieving exact parity with static configuration.

---

## Phase 8: Rigorous Ablation & Scientific Evaluation

- **Purpose**: Definitive 11-configuration component ablation study to isolate the causal contributions of seed scheduling, contextual mutation, compatible donor selection, field splicing, and `StateFrontier`.
- **Experiment Script**: `experiments/phase8_ablation_experiment.py`
- **Result Directory**: `experiments/results/phase8/`
- **Formal Analysis Report**: [`experiments/results/phase8/PHASE8_FINAL_ANALYSIS.md`](file:///c:/Users/HP/Desktop/neurofuzz/experiments/results/phase8/PHASE8_FINAL_ANALYSIS.md)
- **Experimental Scale**: 11 configurations × 5 random seeds = 55 trials (1,375,000 total executions)
- **Key Artifacts**:
  - `experiments/results/phase8/results.csv` (Full 55-trial dataset, 34 telemetry metrics per trial)
  - `experiments/results/phase8/summary.json` (Aggregated statistics and transition counts)
  - `experiments/results/phase8/configuration.json` (Exact configuration parameters)
  - `experiments/results/phase8/trajectories/` (55 per-trial iteration-by-iteration trajectory logs)
  - `experiments/results/phase8/plots/` (13 publication-quality comparison figures)
- **Major Findings**:
  1. `StateFrontier` is the dominant factor driving deep state discovery ($+10.20$ coverage on random baseline, $100\%$ Depth 19 reach rate).
  2. `LinearBandit` seed scheduling adds $-34.3\%$ throughput overhead without improving coverage or depth in the presence of `StateFrontier`.
  3. Structured field splicing contributes $+2.40$ coverage; contextual mutation contributes $+1.40$; compatible donor selection contributes $+1.20$.
  4. Final recommended production architecture established as Configuration $F$ (`RandomScheduler` + `ContextualMutationPolicy` + `CompatibleDonor` + `FieldSplicer` + `AdaptiveStateFrontier` + `AutoDelimiter` + `PersistentHarness`).
