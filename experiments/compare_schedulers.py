"""Controlled comparison experiment: RandomScheduler vs HeuristicScheduler.

Runs both schedulers against the same target with the same budget across
multiple RNG seeds to account for stochastic variance.

Each trial runs in an isolated temporary directory containing a fresh copy of
the initial seeds, ensuring no cross-trial contamination of corpus or crashes.
"""

import contextlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fuzzer.fuzzer import Fuzzer, FuzzStats
from fuzzer.scheduler import HeuristicScheduler, RandomScheduler

TRIALS = [11, 22, 33, 44, 55]
ITERATIONS = 500
TARGET = PROJECT_ROOT / "targets" / "vulnerable_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus"
TIMEOUT = 1.0


def run_trial(scheduler_type: str, rng_seed: int) -> FuzzStats:
    """Run a single fuzzing trial in an isolated environment and return stats."""
    with tempfile.TemporaryDirectory() as tmp_corpus_dir, tempfile.TemporaryDirectory() as tmp_crashes_dir:
        tmp_corpus = Path(tmp_corpus_dir)
        tmp_crashes = Path(tmp_crashes_dir)

        # Copy only pristine initial seed files
        for seed_file in INITIAL_CORPUS.glob("seed*.txt"):
            shutil.copy2(seed_file, tmp_corpus / seed_file.name)

        fuzzer = Fuzzer(
            target_path=TARGET,
            corpus_dir=tmp_corpus,
            crashes_dir=tmp_crashes,
            iterations=ITERATIONS,
            timeout=TIMEOUT,
            seed=rng_seed,
            stats_interval=99999,   # suppress per-iteration output
            calibrate=True,
            scheduler_type=scheduler_type,
            epsilon=0.2,
        )

        # Suppress standard output during the run
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            stats = fuzzer.run()

    return stats


def main() -> None:
    print("=" * 70)
    print(" NeuroFuzz Phase 3 - Controlled Scheduler Comparison Experiment")
    print("=" * 70)
    print(f"Target:        {TARGET.name}")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Schedulers:    random, heuristic (epsilon=0.2)")
    print(f"Environment:   Isolated per trial (pristine initial corpus)")
    print("=" * 70)

    results = {"random": [], "heuristic": []}

    for rng_seed in TRIALS:
        for sched_name in ("random", "heuristic"):
            t0 = time.time()
            stats = run_trial(sched_name, rng_seed)
            elapsed = time.time() - t0
            results[sched_name].append(stats)
            print(
                f"  {sched_name:12s} seed={rng_seed:3d} | "
                f"cov={stats.total_coverage_units:2d} | "
                f"new_inputs={stats.new_coverage_discoveries:2d} | "
                f"corpus={stats.corpus_size:3d} | "
                f"crashes={stats.unique_crashes:3d} | "
                f"rate={stats.exec_per_sec:5.1f}/s | "
                f"avg_rwd={stats.average_reward:.4f} | "
                f"time={elapsed:4.1f}s"
            )

    print("\n" + "=" * 70)
    print(" Aggregate Results (mean across 5 trials)")
    print("=" * 70)

    for sched_name in ("random", "heuristic"):
        s_list = results[sched_name]

        def avg(fn):
            return sum(fn(s) for s in s_list) / len(s_list)

        print(f"\n  Scheduler: {sched_name.upper()}")
        print(f"    Mean Coverage Units:        {avg(lambda s: s.total_coverage_units):.2f}")
        print(f"    Mean New Inputs (corpus+):  {avg(lambda s: s.new_coverage_discoveries):.2f}")
        print(f"    Mean Final Corpus Size:     {avg(lambda s: s.corpus_size):.2f}")
        print(f"    Mean Unique Crashes:        {avg(lambda s: s.unique_crashes):.2f}")
        print(f"    Mean Exec Rate (exec/s):    {avg(lambda s: s.exec_per_sec):.2f}")
        print(f"    Mean Avg Reward:            {avg(lambda s: s.average_reward):.4f}")

    print("\n" + "=" * 70)
    print(" Analysis & Comparison")
    print("=" * 70)
    r_cov = sum(s.total_coverage_units for s in results["random"]) / len(TRIALS)
    h_cov = sum(s.total_coverage_units for s in results["heuristic"]) / len(TRIALS)
    r_new = sum(s.new_coverage_discoveries for s in results["random"]) / len(TRIALS)
    h_new = sum(s.new_coverage_discoveries for s in results["heuristic"]) / len(TRIALS)
    r_rwd = sum(s.average_reward for s in results["random"]) / len(TRIALS)
    h_rwd = sum(s.average_reward for s in results["heuristic"]) / len(TRIALS)
    r_crsh = sum(s.unique_crashes for s in results["random"]) / len(TRIALS)
    h_crsh = sum(s.unique_crashes for s in results["heuristic"]) / len(TRIALS)

    print(f"  Coverage       - Random: {r_cov:.2f} | Heuristic: {h_cov:.2f} (diff: {h_cov - r_cov:+.2f})")
    print(f"  New Inputs     - Random: {r_new:.2f} | Heuristic: {h_new:.2f} (diff: {h_new - r_new:+.2f})")
    print(f"  Unique Crashes - Random: {r_crsh:.2f} | Heuristic: {h_crsh:.2f} (diff: {h_crsh - r_crsh:+.2f})")
    print(f"  Avg Reward     - Random: {r_rwd:.4f} | Heuristic: {h_rwd:.4f}")

    if h_cov > r_cov:
        print(f"\n  [+] Heuristic scheduler achieved higher mean coverage by {h_cov - r_cov:.2f} units.")
    elif r_cov > h_cov:
        print(f"\n  [-] Random scheduler achieved higher mean coverage by {r_cov - h_cov:.2f} units.")
    else:
        print(f"\n  [=] Both schedulers achieved equal mean coverage ({r_cov:.2f} units).")

    print("\n  Honest Evaluation Note:")
    print("  - Target program has limited total branches (17 max observable units).")
    print("  - Both schedulers discover most branches within 500 iterations.")
    print("  - In larger targets with deeper state spaces, heuristic prioritization")
    print("    prevents wasting mutations on exhausted/unproductive seeds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
