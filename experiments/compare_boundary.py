"""Controlled experiment for Phase 6B: Boundary-Aware Dictionary Mutation.

Directly compares:
- Condition A: Random Dictionary Mutation (dict_prob=0.20, boundary_aware=False)
- Condition B: Boundary-Aware Dictionary Mutation (dict_prob=0.20, boundary_aware=True)

Target: targets/structured_target.exe
Initial Corpus: corpus_structured/ (unmodified Phase 5 seeds)
Scheduler: RandomScheduler (as specified to isolate mutation effects)
Trials: 5 independent RNG seeds (101, 202, 303, 404, 505)
Iterations: 2000 per trial
"""

import contextlib
import io
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fuzzer.dictionary import Dictionary
from fuzzer.fuzzer import Fuzzer, FuzzStats

TRIALS = [101, 202, 303, 404, 505]
ITERATIONS = 2000
TARGET = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus_structured"
DICTIONARY_FILE = PROJECT_ROOT / "dictionaries" / "structured_protocol.dict"
TIMEOUT = 1.0
CONDITIONS = ["random_dict", "boundary_aware_dict"]


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    boundary_aware = condition == "boundary_aware_dict"

    with tempfile.TemporaryDirectory() as tmp_corpus_dir, tempfile.TemporaryDirectory() as tmp_crashes_dir:
        tmp_corpus = Path(tmp_corpus_dir)
        tmp_crashes = Path(tmp_crashes_dir)

        # Copy pristine initial structured seed files
        for seed_file in INITIAL_CORPUS.glob("seed*.txt"):
            shutil.copy2(seed_file, tmp_corpus / seed_file.name)

        fuzzer = Fuzzer(
            target_path=TARGET,
            corpus_dir=tmp_corpus,
            crashes_dir=tmp_crashes,
            iterations=iterations,
            timeout=TIMEOUT,
            seed=rng_seed,
            stats_interval=99999,   # suppress periodic terminal logs
            calibrate=True,
            scheduler_type="random",
            dictionary_path=DICTIONARY_FILE,
            dictionary_probability=0.20,
            boundary_aware=boundary_aware,
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = fuzzer.run()

    return stats


def std_dev(values: List[float], mean_val: float) -> float:
    """Calculate sample standard deviation."""
    if len(values) <= 1:
        return 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def main() -> None:
    print("=" * 88)
    print(" NeuroFuzz Phase 6B - Boundary-Aware Dictionary Mutation Evaluation")
    print("=" * 88)
    print(f"Target:        {TARGET.name}")
    print(f"Initial Seeds: {INITIAL_CORPUS.name}/ ({len(list(INITIAL_CORPUS.glob('seed*.txt')))} seeds)")
    print(f"Dictionary:    {DICTIONARY_FILE.name} ({len(Dictionary(filepath=DICTIONARY_FILE))} tokens)")
    print(f"Scheduler:     RandomScheduler (isolated mutation comparison)")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Conditions:    Random Dict (Phase 6A) vs Boundary-Aware Dict (Phase 6B)")
    print(f"Environment:   Isolated temporary workspace per trial")
    print("=" * 88)

    results: Dict[str, List[FuzzStats]] = {c: [] for c in CONDITIONS}

    for rng_seed in TRIALS:
        print(f"\n--- Seed Trial: {rng_seed} ---")
        for cond in CONDITIONS:
            t0 = time.time()
            stats = run_single_trial(cond, rng_seed)
            elapsed = time.time() - t0
            results[cond].append(stats)
            trans_14_15 = stats.depth_transition_counts.get("14->15", 0)
            trans_15_16 = stats.depth_transition_counts.get("15->16", 0)
            trans_16_17 = stats.depth_transition_counts.get("16->17", 0)
            print(
                f"  {cond:<20s} | "
                f"cov={stats.total_coverage_units:2d} | "
                f"depth={stats.deepest_state_reached:2d} | "
                f"14->15={trans_14_15:2d} | "
                f"15->16={trans_15_16:2d} | "
                f"16->17={trans_16_17:2d} | "
                f"new_cov={stats.new_coverage_discoveries:2d} | "
                f"bound_cov={stats.boundary_new_coverage:2d} | "
                f"crashes={stats.unique_crashes:2d} | "
                f"exec/s={stats.exec_per_sec:5.1f} | "
                f"time={elapsed:.2f}s"
            )

    # Aggregate Analysis
    print("\n" + "=" * 88)
    print(" AGGREGATE RESULTS SUMMARY (Mean +/- Sample Std Dev across 5 trials)")
    print("=" * 88)
    header = (
        f"{'Condition':<20} | "
        f"{'Coverage Units':<16} | "
        f"{'Max Depth':<12} | "
        f"{'New Discov':<12} | "
        f"{'Corpus Size':<12} | "
        f"{'Crashes':<9} | "
        f"{'Bound Cov':<10} | "
        f"{'Exec/s':<8}"
    )
    print(header)
    print("-" * 88)

    for cond in CONDITIONS:
        runs = results[cond]
        n = len(runs)

        mean_cov = sum(r.total_coverage_units for r in runs) / n
        sd_cov = std_dev([float(r.total_coverage_units) for r in runs], mean_cov)

        mean_depth = sum(r.deepest_state_reached for r in runs) / n
        sd_depth = std_dev([float(r.deepest_state_reached) for r in runs], mean_depth)

        mean_new = sum(r.new_coverage_discoveries for r in runs) / n
        sd_new = std_dev([float(r.new_coverage_discoveries) for r in runs], mean_new)

        mean_corpus = sum(r.corpus_size for r in runs) / n
        sd_corpus = std_dev([float(r.corpus_size) for r in runs], mean_corpus)

        mean_crashes = sum(r.unique_crashes for r in runs) / n
        sd_crashes = std_dev([float(r.unique_crashes) for r in runs], mean_crashes)

        mean_bound_cov = sum(r.boundary_new_coverage for r in runs) / n
        sd_bound_cov = std_dev([float(r.boundary_new_coverage) for r in runs], mean_bound_cov)

        mean_exec_rate = sum(r.exec_per_sec for r in runs) / n

        print(
            f"{cond:<20} | "
            f"{mean_cov:5.1f} +/- {sd_cov:4.1f}    | "
            f"{mean_depth:4.1f} +/- {sd_depth:3.1f}  | "
            f"{mean_new:4.1f} +/- {sd_new:3.1f}  | "
            f"{mean_corpus:4.1f} +/- {sd_corpus:3.1f}  | "
            f"{mean_crashes:3.1f} +/- {sd_crashes:3.1f} | "
            f"{mean_bound_cov:4.1f} +/- {sd_bound_cov:3.1f}  | "
            f"{mean_exec_rate:6.1f}"
        )

    # Detailed Depth Transition Matrix
    print("\n" + "=" * 88)
    print(" DEPTH TRANSITION OCCURRENCES SUMMARY (Total occurrences across all 5 trials)")
    print("=" * 88)
    all_transitions = ["14->15", "15->16", "16->17", "17->18", "18->19"]
    trans_header = f"{'Condition':<20} | " + " | ".join(f"{t:^8}" for t in all_transitions) + " | Total Deep Transitions"
    print(trans_header)
    print("-" * 88)

    for cond in CONDITIONS:
        runs = results[cond]
        counts = {t: sum(r.depth_transition_counts.get(t, 0) for r in runs) for t in all_transitions}
        total_deep = sum(counts.values())
        row_str = f"{cond:<20} | " + " | ".join(f"{counts[t]:^8d}" for t in all_transitions) + f" | {total_deep:^23d}"
        print(row_str)

    print("=" * 88)


if __name__ == "__main__":
    main()
