# Portfolio & LinkedIn Project Description

---

### Two-Line Elevator Summary
**NeuroFuzz** is an AI-guided structured fuzzing framework that combines contextual mutation bandits, compatibility-aware field splicing, and state-sequence frontier exploration to penetrate deep protocol states that conventional fuzzers fail to reach.

---

### Technical Stack
- **Languages**: Python 3.10+ (Standard Library, NumPy), C (Target Binary & Coverage Instrumentation)
- **Architecture**: Modular 4-tier pipeline (Corpus Management, Contextual Bandits, Structured Mutators, StateFrontier)
- **Harnessing**: Win64 persistent execution harness with anonymous pipes, NTSTATUS exception catching, and crash deduplication
- **Evaluation & Analysis**: Automated 55-trial paired ablation matrix (1.375M executions), Pandas, Matplotlib

---

### Core Research Question
> *"Can structured state-frontier exploration accelerate discovery of deep sequential program states that conventional mutation-based fuzzing fails to reach, and does machine learning seed scheduling remain beneficial once frontier guidance is active?"*

---

### Key Empirical Findings
1. **StateFrontier is decisive**: Reached maximum protocol depth (Depth 19) in 100% of trials in under 350 iterations (compared to 0% reach rate for conventional fuzzing up to 25,000 iterations), improving coverage by **+10.2 units**.
2. **Field splicing & contextual mutations provide breadth**: Crossover contributed **+2.4 units** of coverage, preventing search entrapment.
3. **Data-driven simplification**: An online ridge-regression seed scheduler was found to cause a **34% throughput penalty** with no coverage benefit in the presence of StateFrontier, and was removed from the default production architecture based on ablation data.

---

### Canonical GitHub Project Description
"NeuroFuzz is a structured coverage-guided fuzzing framework that combines contextual mutation selection, compatibility-aware field splicing, adaptive state-frontier exploration, runtime delimiter discovery, and persistent execution to accelerate exploration of deep sequential program states."

---

### Skills Demonstrated
- **Systems Programming & IPC**: Low-overhead persistent execution harness on Windows with sub-second timeout enforcement.
- **Vulnerability Research**: Crash deduplication, memory corruption classification, and grammar-aware protocol fuzzing.
- **Machine Learning & Bandits**: Contextual multi-armed bandits (LinUCB / Thompson-style) and online ridge regression.
- **Scientific Research Discipline**: Rigorous paired ablation methodology (1.375M executions), transparent reporting of negative findings, and publication-quality data visualization.
