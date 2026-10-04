# NeuroFuzz Empirical Evaluation Results

**Authoritative Source**: `experiments/results/phase8/PHASE8_FINAL_ANALYSIS.md`  
**Dataset Scale**: 11 Configurations × 5 Deterministic Seeds = 55 Trials (1,375,000 Executions)  
**Target Program**: `targets/structured_target.exe` (Win64 MSVC binary)  
**Execution Budget**: 25,000 iterations per trial  

---

## 1. Experimental Setup & Configurations

All trials were executed under identical conditions using the persistent execution harness across random seeds `101`, `202`, `303`, `404`, and `505`.

| Config ID | Label | Seed Scheduler | Mutation Policy | Donor Selection | Field Splicing | StateFrontier | Delimiter |
|---|---|---|---|---|---|---|---|
| **A** | Random Baseline | Random | Fixed | Random | Disabled | OFF | Static (`\|`) |
| **B** | Pre-Frontier NeuroFuzz | LinearBandit | Contextual | Compatible | Enabled | OFF | Static (`\|`) |
| **C** | Full System (Fixed) | LinearBandit | Contextual | Compatible | Enabled | Fixed (20%) | Static (`\|`) |
| **D** | Full System (Adaptive)| LinearBandit | Contextual | Compatible | Enabled | Adaptive | Static (`\|`) |
| **E** | Random + Frontier | Random | Fixed | Random | Disabled | Fixed (20%) | Static (`\|`) |
| **F** | Ablate Seed Learning | Random | Contextual | Compatible | Enabled | Adaptive | Static (`\|`) |
| **G** | Ablate Contextual Mut | LinearBandit | Fixed | Compatible | Enabled | Adaptive | Static (`\|`) |
| **H** | Ablate Compatible Donor| LinearBandit | Contextual | Random | Enabled | Adaptive | Static (`\|`) |
| **I** | Ablate Field Splicing | LinearBandit | Contextual | Compatible | Disabled | Adaptive | Static (`\|`) |
| **J** | Fixed Frontier Rate | LinearBandit | Contextual | Compatible | Enabled | Fixed (20%) | Static (`\|`) |
| **K** | Auto Delimiter Discovery| LinearBandit | Contextual | Compatible | Enabled | Adaptive | Auto (`\|`) |

---

## 2. Summary Results Across All Configurations

Mean $\pm$ standard deviation across 5 trials per configuration:

| Configuration | Final Coverage | Coverage AUC | Max Depth | Depth AUC | D17 Reach Rate | Mean $T_{17}$ | Unique Crashes | Throughput (exec/s) |
|---|---|---|---|---|---|---|---|---|
| **A_random_baseline** | $61.8 \pm 1.30$ | $1,483,436 \pm 22,753$ | $14.8 \pm 1.10$ | $364,072 \pm 19,302$ | 0% | Never | $188.0 \pm 124.3$ | $1,431.2 \pm 238.9$ |
| **B_neurofuzz_no_frontier** | $64.8 \pm 2.05$ | $1,551,782 \pm 36,498$ | $15.2 \pm 0.84$ | $374,206 \pm 18,162$ | 0% | Never | $217.2 \pm 225.5$ | $700.1 \pm 168.5$ |
| **C_neurofuzz_full_fixed** | $72.0 \pm 1.58$ | $1,699,615 \pm 35,559$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $134.4 \pm 128.8$ | $552.8 \pm 106.2$ |
| **D_neurofuzz_full_adaptive**| $72.4 \pm 3.36$ | $1,705,683 \pm 68,347$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $130.4 \pm 83.9$ | $575.6 \pm 36.2$ |
| **E_random_frontier** | $72.0 \pm 0.00$ | $1,720,711 \pm 31,575$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | **291.8 ± 84.6** | $967.9 \pm 74.5$ |
| **F_no_seed_learning** (Recommended) | **72.6 ± 1.52** | **1,758,403 ± 34,398**| $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $189.8 \pm 109.0$ | $876.2 \pm 67.7$ |
| **G_no_contextual_mutation** | $71.0 \pm 1.73$ | $1,702,539 \pm 41,943$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $262.0 \pm 171.8$ | $576.7 \pm 56.3$ |
| **H_no_compatible_donor** | $71.2 \pm 2.95$ | $1,701,361 \pm 52,165$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $134.0 \pm 161.3$ | $653.0 \pm 49.4$ |
| **I_no_field_splicing** | $70.0 \pm 3.94$ | $1,675,979 \pm 74,524$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $175.6 \pm 125.7$ | $846.2 \pm 91.3$ |
| **J_no_adaptive_frontier** | $72.0 \pm 1.58$ | $1,699,615 \pm 35,559$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $134.4 \pm 128.8$ | $552.8 \pm 106.2$ |
| **K_no_auto_delimiter** | $72.4 \pm 3.36$ | $1,705,683 \pm 68,347$ | $19.0 \pm 0.00$ | $473,783 \pm 0$ | 100% | 261.0 iters | $130.4 \pm 83.9$ | $612.4 \pm 53.0$ |

---

## 3. Component Ablation Findings

Evaluating exact paired differences ($\Delta = \text{Metric}_{\text{Full } D} - \text{Metric}_{\text{Ablated}}$ across seeds `101`–`505`):

```
StateFrontier (E vs A)   [██████████████████████████████]  +10.20 Cov (5/5 pos, +4.20 Depth)
StateFrontier (D vs B)   [████████████████████]            +7.60 Cov (4/5 pos, +3.80 Depth)
Field Splicing (D vs I)  [██████]                          +2.40 Cov (3/5 pos, 1 tie)
Contextual Mut (D vs G)  [████]                            +1.40 Cov (4/5 pos)
Compatible Donor (D vs H)[███]                             +1.20 Cov (4/5 pos)
Adaptive Frontier(D vs J)[█]                               +0.40 Cov (+22.8 exec/s)
Auto Delimiter (D vs K)  [~]                                0.00 Cov (100% parity)
Seed Learning (D vs F)   [░░░░░░░░]                        -0.20 Cov (-300.6 exec/s penalty)
```

### Detailed Component Analysis
1. **`StateFrontier`**: The single most decisive mechanism in NeuroFuzz. Without it, depths 17–19 were unreachable ($0\%$ reach rate). With it, depths 17–19 were discovered in $100\%$ of trials in under 350 iterations.
2. **`Field Splicing`**: The strongest non-frontier mutation operator ($\Delta \text{Cov} = +2.40 \pm 5.59$). Recombines verified sub-tokens, preventing search entrapment.
3. **`Contextual Mutation`**: Modest, reliable coverage increase ($\Delta \text{Cov} = +1.40 \pm 3.91$, positive on 4/5 seeds) with negligible runtime overhead.
4. **`Compatible Donor Selection`**: Outperforms uniform random donor selection by $+1.20 \pm 5.36$ coverage units.
5. **`Adaptive Frontier Allocation`**: Decays allocation from 20% to 2% floor after state space saturation, saving $4,480$ executions per trial and modestly improving throughput (+22.8 exec/s).
6. **`Automatic Delimiter Discovery`**: Achieved 100% empirical parity with manual static configuration on all metrics.
7. **`ML Seed Learning (LinearBandit)`**: Imposed a **$-34.3\%$ throughput penalty** ($-300.6$ exec/s) and achieved lower coverage velocity than uniform random scheduling ($\text{AUC } 1.705\text{M}$ vs $1.758\text{M}$) once StateFrontier was enabled.

---

## 4. Key Figures

All figures are generated directly from the trial trajectories in `experiments/results/phase8/plots/`:

- **Coverage Discovery Velocity**: `coverage_vs_iterations.png` and `coverage_auc.png`
- **Protocol Depth Progression**: `depth_vs_iterations.png` and `maximum_depth.png`
- **State Transition Velocity ($T_{15}$–$T_{19}$)**: `time_to_depth.png` and `t15_t19_reach_rates.png`
- **Crash Discovery Tradeoffs**: `crash_comparison.png`
- **Throughput & Algorithmic Overhead**: `throughput_comparison.png`
- **Component Ablation Contributions**: `ablation_contribution.png`
