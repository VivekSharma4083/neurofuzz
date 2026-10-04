"""Controlled 3-way comparison experiment on Phase 5 structured_target:
Random vs Heuristic vs LinearBandit.

Runs all three schedulers against the deep structured C target under identical
conditions across multiple independent RNG seeds.

Each trial runs in an isolated temporary directory containing a fresh copy of
the initial structured seeds (corpus_structured/), ensuring complete isolation.
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

from fuzzer.fuzzer import Fuzzer, FuzzStats

TRIALS = [101, 202, 303, 404, 505]
ITERATIONS = 500
TARGET = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus_structured"
TIMEOUT = 1.0
SCHEDULERS = ["random", "heuristic", "linear_bandit"]


def run_trial(scheduler_type: str, rng_seed: int, iterations: int = ITERATIONS) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
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
            stats_interval=99999,   # suppress periodic logs
            calibrate=True,
            scheduler_type=scheduler_type,
            epsilon=0.2,
            learning_rate=0.05,
            l2_reg=0.001,
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
    print("=" * 80)
    print(" NeuroFuzz Phase 5 - Benchmark Target 3-Way Policy Comparison Experiment")
    print("=" * 80)
    print(f"Target:        {TARGET.name}")
    print(f"Initial Seeds: {INITIAL_CORPUS.name}/ ({len(list(INITIAL_CORPUS.glob('seed*.txt')))} seeds)")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Policies:      {', '.join(SCHEDULERS)}")
    print(f"Hyperparams:   epsilon=0.20, learning_rate=0.05, l2_reg=0.001")
    print(f"Environment:   Isolated temporary workspace per trial")
    print("=" * 80)

    results: Dict[str, List[FuzzStats]] = {s: [] for s in SCHEDULERS}

    for rng_seed in TRIALS:
        print(f"\n--- Seed Trial: {rng_seed} ---")
        for sched_name in SCHEDULERS:
            t0 = time.time()
            stats = run_trial(sched_name, rng_seed)
            elapsed = time.time() - t0
            results[sched_name].append(stats)
            print(
                f"  {sched_name:14s} | "
                f"cov={stats.total_coverage_units:2d} | "
                f"depth={stats.deepest_state_reached:2d} | "
                f"new_inputs={stats.new_coverage_discoveries:2d} | "
                f"corpus={stats.corpus_size:2d} | "
                f"crashes={stats.unique_crashes:2d} | "
                f"reward={stats.total_reward:2d} | "
                f"exec/s={stats.exec_per_sec:5.1f} | "
                f"time={elapsed:.2f}s"
            )

    # Aggregate Analysis
    print("\n" + "=" * 80)
    print(" AGGREGATE RESULTS SUMMARY (Mean +/- Sample Std Dev across trials)")
    print("=" * 80)
    header = (
        f"{'Policy':<14} | "
        f"{'Coverage Units':<16} | "
        f"{'Max Depth':<11} | "
        f"{'New Discov':<12} | "
        f"{'Corpus Size':<12} | "
        f"{'Crashes':<9} | "
        f"{'Reward':<9} | "
        f"{'Exec/s':<8}"
    )
    print(header)
    print("-" * 80)

    for sched_name in SCHEDULERS:
        runs = results[sched_name]
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

        mean_reward = sum(r.total_reward for r in runs) / n
        sd_reward = std_dev([float(r.total_reward) for r in runs], mean_reward)

        mean_exec_rate = sum(r.exec_per_sec for r in runs) / n

        print(
            f"{sched_name:<14} | "
            f"{mean_cov:5.1f} +/- {sd_cov:4.1f}    | "
            f"{mean_depth:4.1f} +/- {sd_depth:3.1f} | "
            f"{mean_new:4.1f} +/- {sd_new:3.1f}  | "
            f"{mean_corpus:4.1f} +/- {sd_corpus:3.1f}  | "
            f"{mean_crashes:3.1f} +/- {sd_crashes:3.1f} | "
            f"{mean_reward:3.1f} +/- {sd_reward:3.1f} | "
            f"{mean_exec_rate:6.1f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
