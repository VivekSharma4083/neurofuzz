# NeuroFuzz Demonstration Script (3–5 Minute Walkthrough)

This document provides a structured, live demonstration sequence showcasing NeuroFuzz's architecture, state-frontier exploration, and empirical performance.

---

## 1. Problem Introduction (30 seconds)
- **Visual**: Show `targets/structured_target.c` lines 320–360 (the sequential state machine).
- **Explanation**: "Notice this target protocol requires sequential fields: `NF01|DIAG|STEP=1|STEP=2|...|STEP=5`. Reaching Depth 19 requires all five steps. A conventional fuzzer randomly corrupting bytes will almost certainly break the pipes or magic header, getting trapped at Depth 14."

---

## 2. Baseline Demonstration (45 seconds)
- **Command**:
  ```powershell
  python -m fuzzer.fuzzer --target targets/structured_target.exe --scheduler random --mutation-policy fixed --no-enable-splicing --no-frontier-extension --iterations 100 --seed 101
  ```
- **Observation**:
  - The baseline fuzzer runs.
  - Coverage plateaus around 41–43 units.
  - `Deepest State Reached: Depth 14`.
  - Notice it is completely incapable of finding `STEP=2` or crossing to Depth 15.

---

## 3. NeuroFuzz Live Run with StateFrontier (90 seconds)
- **Command**:
  ```powershell
  python -m fuzzer.fuzzer --target targets/structured_target.exe --corpus corpus_structured --dictionary dictionaries/structured_protocol.dict --dictionary-probability 0.3 --iterations 500 --seed 101
  ```
- **Observation & Live Commentary**:
  - **Auto-Discovery**: Note the banner: `Delimiter Discovery Mode: AUTO -> Active Delimiter: '|' (Confidence: 94.9%)`.
  - **Coverage Leap**: Within the first 350 iterations, notice the log lines:
    ```
    [+] NEW COVERAGE: Depth 15 reached! [state_frontier_extend]
    [+] NEW COVERAGE: Depth 16 reached! [state_frontier_extend]
    [+] NEW COVERAGE: Depth 17 reached! [state_frontier_extend]
    [+] NEW COVERAGE: Depth 18 reached! [state_frontier_extend]
    [+] NEW COVERAGE: Depth 19 reached! [state_frontier_extend]
    ```
  - **StateFrontier Telemetry**: Point to `Deepest State Reached: Depth 19`.
  - **Adaptive Rate**: Point to `Adaptive Frontier Rate: decayed to 2% floor` once depth saturated.
  - **Throughput**: Highlight execution speed (>850 exec/s under persistent harness).

---

## 4. Crash Handling Demonstration (30 seconds)
- **Inspection**:
  ```powershell
  Get-ChildItem -Path crashes -Filter "*.json" | Select-Object -First 3
  ```
- **Explanation**: "When an input triggers an access violation (`0xC0000005`) or divide-by-zero, the persistent executor captures the fault, deduplicates it by crash PC, and persists the raw input bytes and execution metadata to `crashes/`."

---

## 5. Research Findings & Ablation Summary (60 seconds)
- **Visual**: Open `experiments/results/phase8/plots/depth_vs_iterations.png` and `ablation_contribution.png`.
- **Takeaway**:
  - "In our 55-trial, 1.375-million execution ablation study:
    - `StateFrontier` alone was responsible for reaching Depth 19 in 100% of trials in under 350 iterations.
    - Splicing and contextual mutation added important coverage breadth.
    - Online linear bandit seed scheduling was removed because it added a 34% throughput penalty without coverage gain once StateFrontier was active."
