# Scientific Claims, Evidence & Limitations

This document provides a rigorous, audited mapping of every research claim made by the NeuroFuzz project to its empirical experimental evidence and scientific limitations.

---

### Claim 1: StateFrontier accelerates deep sequential-state discovery
- **CLAIM**: Targeting terminal protocol delimiters with grammar tokens (`StateFrontier`) provides an order-of-magnitude acceleration in discovering sequential state transitions that conventional point mutations fail to reach.
- **EVIDENCE**: 
  - In Phase 8 paired ablations, `StateFrontier` raised maximum reachable protocol depth from $14.8 \pm 1.10$ to $19.0 \pm 0.00$ on the random baseline ($E$ vs $A$, $+4.20$ depth levels, **positive across 100% of tested seeds**) and from $15.2 \pm 0.84$ to $19.0 \pm 0.00$ in the full system ($D$ vs $B$).
  - Depths 17, 18, and 19 were reached in **100% of trials** ($45/45$) with `StateFrontier` active in under 350 iterations, while unguided fuzzers reached Depth 17 in **0% of trials** ($0/10$).
- **LIMITATION**: Evaluated on the structured state-machine benchmark target (`structured_target.c`). Efficacy on protocols without clear delimiter structures or fixed grammar rules remains to be evaluated.

---

### Claim 2: Structured field splicing prevents search entrapment
- **CLAIM**: Crossover through structured field recombination prevents the fuzzer from becoming trapped in localized coverage plateaus.
- **EVIDENCE**:
  - Disabling field splicing ($D$ vs $I$) reduced mean coverage by $-2.40 \pm 5.59$ units and reduced Coverage AUC by $-29,703.8$.
  - In Seed 202, disabling splicing caused coverage to drop severely to 63 (compared to 73 in the full system).
- **LIMITATION**: Field splicing can only recombine tokens and fields already discovered in the corpus; it cannot synthesize novel atomic tokens.

---

### Claim 3: Contextual mutation provides modest coverage gains
- **CLAIM**: Conditioning mutation operator selection on seed context features (depth, command type, delimiter density) improves overall coverage discovery compared to a fixed static distribution.
- **EVIDENCE**:
  - Comparing full contextual selection to fixed mutation ($D$ vs $G$) demonstrated a $+1.40 \pm 3.91$ unit coverage advantage, showing positive improvement on **4 of 5 tested seeds**.
  - Operates with negligible runtime overhead ($\Delta \text{Throughput} \approx -1.0$ exec/s).
- **LIMITATION**: The effect size ($+1.40$ units) is modest compared to the dominant impact of `StateFrontier` ($+10.20$ units).

---

### Claim 4: Compatibility-aware donor selection improves crossover quality
- **CLAIM**: Prioritizing donor seeds that share command types and prefix structures improves the yield of field splicing compared to uniform random donor selection.
- **EVIDENCE**:
  - Compatibility-aware donor selection ($D$ vs $H$) increased mean coverage by $+1.20 \pm 5.36$ units, with positive gains on **4 of 5 seeds**.
- **LIMITATION**: Pairwise header and delimiter comparison introduces a slight runtime cost ($-77.4$ exec/s relative to random donor picking).

---

### Claim 5: Adaptive frontier allocation eliminates waste after state saturation
- **CLAIM**: Dynamically decaying frontier mutation probability upon consecutive stagnant attempts preserves execution budget without degrading depth discovery.
- **EVIDENCE**:
  - Adaptive allocation ($D$ vs $J$) achieved identical Depth 19 discovery at iteration 350 while decaying mutation rate to the 2% floor, saving **$4,480$ redundant frontier evaluations per trial** and improving throughput by $+22.8$ exec/s.
- **LIMITATION**: If an unexplored state branch requires thousands of attempts to unlock, aggressive rate decay could prematurely de-prioritize the frontier (mitigated in NeuroFuzz by rate replenishment on new branch discoveries).

---

### Claim 6: Delimiter discovery achieves empirical parity with manual configuration
- **CLAIM**: Automatic runtime inference of protocol delimiters eliminates hardcoded protocol assumptions without sacrificing fuzzing performance.
- **EVIDENCE**:
  - In Phase 8 ($K$ vs $D$), dynamic delimiter detection converged on `b'|'` with 100% confidence across all seeds, yielding **exact 100% parity** in final coverage ($\Delta = 0.00$), depth ($\Delta = 0.00$), and crashes ($\Delta = 0.00$).
- **LIMITATION**: Validated on character-delimited text/ASCII formats; binary length-value-type framing requires different inference heuristics.

---

### Claim 7: Online seed learning adds overhead without benefit in frontier fuzzing
- **CLAIM**: Machine learning seed scheduling (Linear Contextual Bandit) does not improve coverage or depth once state-frontier exploration is enabled, and imposes a substantial throughput penalty.
- **EVIDENCE**:
  - Comparing the bandit scheduler to uniform random scheduling ($D$ vs $F$) showed an inconclusive coverage difference ($\Delta \text{Coverage} = -0.20 \pm 4.02$, with random scheduling achieving higher mean coverage: $72.6$ vs $72.4$).
  - Random scheduling achieved higher coverage discovery velocity ($\text{AUC } 1,758,403.4$ vs $1,705,683.1$).
  - Online ridge regression feature extraction and updates imposed a **$-34.3\%$ throughput penalty** ($-300.6$ exec/s).
- **LIMITATION**: This finding applies to workloads where `StateFrontier` actively drives deep-state penetration; in targets without frontier guidance, learned seed scheduling may provide different utility.

---

### Claim 8: Syntax-preserving guided search trades shallow crashes for deep state coverage
- **CLAIM**: Unguided random search triggers more shallow parser crashes, while structured guided search reaches deeper program logic.
- **EVIDENCE**:
  - Pure random search with frontier (`E_random_frontier`) discovered **$291.8 \pm 84.6$ unique crashes**, compared to **$130.4 \pm 83.9$** for the full guided system ($D$).
  - Unguided mutations aggressively corrupt boundaries and field lengths, triggering shallow error handlers, while learned policies preserve syntax to reach deeper execution paths.
- **LIMITATION**: Crash discovery metrics depend heavily on whether target bugs reside in input validation parsers vs. deep business logic.
