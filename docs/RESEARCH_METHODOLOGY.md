# NeuroFuzz Research Methodology & Experimental Progression

**Project Timeline**: Phases 1 through 9  
**Core Research Focus**: Autonomous, AI-guided exploration of structured protocol state spaces  

---

## 1. Research Overview & Problem Formulation

Traditional coverage-guided fuzzers (e.g., AFL, libFuzzer) operate effectively on binary parsers and flat input structures by flipping random bits and substituting bytes. However, when applied to **structured multi-stage network protocols and stateful file formats**, conventional mutation struggles:

1. **Syntactic Fragility**: Random byte edits corrupt structural delimiters and magic headers, causing shallow parser rejections.
2. **Sequential State Barriers**: Moving from State $N$ to State $N+1$ requires preserving the exact prefix of valid states $1 \dots N$ while appending a syntactically valid transition token. The probability of random point mutations synthesizing multi-byte sequential tokens decreases exponentially ($P \propto (1/256)^k$).
3. **The Building Block Deficit**: Genetic crossover / splicing cannot invent new grammar tokens; it can only recombine tokens that already exist in the corpus.

NeuroFuzz was conceived to systematically test whether machine learning, contextual bandits, and structured state-frontier exploration could overcome these sequential barriers.

---

## 2. Experimental Progression

The research progressed through nine distinct phases, each driven by a specific hypothesis:

```
[Phase 1: Pure Mutation] ──> [Phase 2: Coverage Feedback] ──> [Phase 3: Seed Scheduling]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
▼
[Phase 4: Learned Bandits] ──> [Phase 5: Structured Target] ──> [Phase 6: Structured Mutators]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
▼
[Phase 7A: Persistent IPC] ──> [Phase 7B: 875k Benchmark] ──> [Phase 7C: StateFrontier]
                                                                        │
┌───────────────────────────────────────────────────────────────────────┘
▼
[Phase 7D: Delimiter & Adaptive] ──> [Phase 8: Rigorous Ablation] ──> [Phase 9: Frozen Package]
```

### Phase 1: Foundational Mutation Engine
- **Objective**: Construct a decoupled mutation engine with bit flips, byte replacements, insertions, and deletions.
- **Outcome**: Established baseline deterministic mutation operators and crash detection.

### Phase 2: Coverage Feedback Instrumentation
- **Objective**: Introduce coverage guidance without external compiler dependencies.
- **Implementation**: Structured stdout/stderr instrumentation emitting logical branch tokens (`__COV__:<PATH>`).
- **Outcome**: Demonstrated that coverage guidance increased branch discovery from 6 units (random) to 12 units on `vulnerable_target`.

### Phase 3: Intelligent Seed Scheduling
- **Objective**: Evaluate whether prioritizing high-yield seeds accelerates coverage accumulation.
- **Implementation**: Introduced `HeuristicScheduler` using Laplace-smoothed yield scoring with decay.
- **Outcome**: Established the concept of seed prioritization, improving early coverage discovery velocity.

### Phase 4: Online Learned Seed Policy (LinearBandit)
- **Objective**: Can an online machine learning model predict seed yield based on input features?
- **Implementation**: Deployed `LinearBanditScheduler`, an online ridge regression model updating weights ($w \in \mathbb{R}^8$) on execution feedback.
- **Outcome**: Successfully demonstrated weight convergence on features like `coverage_density` and `recency`.

### Phase 5: The Structured Benchmark Target
- **Objective**: Create a representative protocol benchmark target to expose the limits of flat fuzzers.
- **Implementation**: Built `targets/structured_target.c`, a multi-stage protocol parser featuring magic headers (`NF01`), command dispatch (`ECHO`, `CALC`, `DIAG`, `AUTH`), nested arithmetic, and a 5-step sequential state machine (`STEP=1` $\to \dots \to$ `STEP=5`, reaching Depth 19).
- **Outcome**: Flat mutation fuzzers plateaued at Depth 14, completely unable to discover `STEP=2`.

### Phase 6: Grammar Awareness & Structured Operators
- **Phase 6A (Dictionary)**: Injected valid protocol tokens via dictionary mutation.
- **Phase 6B (Boundary Awareness)**: Snapped dictionary insertions to structural delimiter boundaries (`|`).
- **Phase 6C (UCB1 Operator Selection)**: Learned a global multi-armed bandit policy over mutation operators.
- **Phase 6D (Contextual Mutation Bandit)**: Conditioned operator selection on seed context (depth, command type, delimiter density).
- **Phase 6E (Field Splicing)**: Introduced structured crossover between parent seeds.
- **Phase 6F (Compatibility-Aware Donor Selection)**: Prioritized donors sharing command types and prefix structures.
- **Key Finding**: While contextual mutation improved mean coverage to $56.8 \pm 3.3$, **none of the trials crossed Depth 14 $\to$ Depth 15**. Splicing could not invent `STEP=2` because `STEP=2` was not yet in the corpus.

### Phase 7A & 7B: Throughput Optimization & High-Budget Evaluation
- **Phase 7A (Persistent Harness)**: Developed an in-process persistent execution harness using length-prefixed IPC pipes, boosting throughput by **11.24×** (from $\sim 70$ exec/s to $>780$ exec/s).
- **Phase 7B (High-Budget Experiment)**: Executed a controlled 875,000-execution benchmark (7 configs × 5 seeds × 25,000 iterations).
- **Discovery**: Despite high execution budgets, **all configurations saturated at Depth 16**. The bottleneck was proven to be algorithmic: standard point mutations have vanishingly small probability of assembling multi-field state chains.

### Phase 7C: State-Sequence-Aware Frontier Mutation (`StateFrontier`)
- **Objective**: Formulate an explicit state-frontier exploration mechanism.
- **Mechanism**: `StateFrontier` tracks the deepest state reached per seed, identifies the terminal boundary, and directly synthesizes candidate transitions using protocol grammar.
- **Result**: **100% of trials reached Depth 19** in under 350 iterations (compared to 0% in all prior phases).

### Phase 7D: Adaptive Frontier & Delimiter Generalization
- **Objective**: Prevent waste after state saturation and remove hardcoded delimiter assumptions.
- **Implementation**:
  - `AdaptiveFrontierAllocator`: Dynamically decays frontier rate from 20% to 2% floor after saturation.
  - `DelimiterDetector`: Automatically infers delimiters (`|`) from corpus token frequencies with 100% accuracy.
- **Result**: Saved $4,480$ redundant frontier executions per trial while maintaining 100% Depth 19 reach rate.

### Phase 8: Rigorous Ablation & Scientific Evaluation
- **Objective**: Evaluate the complete 11-configuration matrix (55 trials, 1,375,000 executions) to determine which components actually matter.
- **Definitive Findings**:
  1. `StateFrontier` is the sole primary driver of deep-state discovery ($+10.20$ coverage, $+4.20$ depth).
  2. `LinearBanditScheduler` is ineffective when `StateFrontier` is active (inconclusive $\Delta \text{Cov} = -0.20$, but $-34.3\%$ throughput penalty).
  3. `StructuredFieldSplicing` is the strongest mutation operator ($+2.40$ coverage).
  4. Contextual mutation and compatible donor selection provide modest, consistent gains ($+1.40$ and $+1.20$).
  5. Random frontier fuzzing discovers over 2× more unique crashes than syntax-preserving guided search.

---

## 3. Scientific Integrity & Negative Results

A core principle of the NeuroFuzz methodology is **transparent reporting of negative findings**:

- **The Negative Seed-Learning Result**: The initial hypothesis that ML seed scheduling would optimize structured protocol exploration was **empirically disproven** by the ablation study once `StateFrontier` was active. Rather than concealing this result or forcing a positive spin, the LinearBandit scheduler was removed from the default configuration and documented as an overhead source.
- **The Crash Tradeoff**: While guided search achieved deeper protocol logic, it found fewer crashes than unguided random mutation because syntax preservation actively avoids destructive parser errors.
