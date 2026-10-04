"""Controlled experiment for Phase 6C: UCB1 Learned Mutation-Operator Selection.

Directly compares:
- Condition A: Fixed Mutation Policy (Phase 6B baseline: dict_prob=0.20, boundary_aware=True, mutation_policy='fixed')
- Condition B: UCB1 Mutation Policy (Phase 6C learned: boundary_aware=True, mutation_policy='ucb1', ucb_c=1.0)

Target: targets/structured_target.exe
Initial Corpus: corpus_structured/ (6 pristine Phase 5 seeds)
Dictionary: dictionaries/structured_protocol.dict (85 tokens)
Scheduler: RandomScheduler (as specified to isolate mutation operator selection)
Trials: 5 independent RNG seeds (101, 202, 303, 404, 505)
Iterations: 2500 per trial
"""

import contextlib
import io
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fuzzer.dictionary import Dictionary
from fuzzer.fuzzer import Fuzzer, FuzzStats

TRIALS = [101, 202, 303, 404, 505]
ITERATIONS = 2500
TARGET = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus_structured"
DICTIONARY_FILE = PROJECT_ROOT / "dictionaries" / "structured_protocol.dict"
TIMEOUT = 1.0
CONDITIONS = ["fixed_policy", "ucb1_policy"]


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    is_ucb1 = condition == "ucb1_policy"
    mut_policy = "ucb1" if is_ucb1 else "fixed"

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
            boundary_aware=True,
            mutation_policy=mut_policy,
            ucb_c=1.0,
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
    print("=" * 92)
    print(" NeuroFuzz Phase 6C - UCB1 Learned Mutation-Operator Selection Evaluation")
    print("=" * 92)
    print(f"Target:        {TARGET.name}")
    print(f"Initial Seeds: {INITIAL_CORPUS.name}/ ({len(list(INITIAL_CORPUS.glob('seed*.txt')))} seeds)")
    print(f"Dictionary:    {DICTIONARY_FILE.name} ({len(Dictionary(filepath=DICTIONARY_FILE))} tokens)")
    print(f"Boundary Mode: ENABLED (snapped to delimiter boundaries)")
    print(f"Scheduler:     RandomScheduler (isolated mutation comparison)")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Conditions:    Fixed Policy (Phase 6B) vs UCB1 Policy (Phase 6C, c=1.0)")
    print(f"Environment:   Isolated temporary workspace per trial")
    print("=" * 92)

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
            print(
                f"  {cond:<15s} | "
                f"cov={stats.total_coverage_units:2d} | "
                f"depth={stats.deepest_state_reached:2d} | "
                f"14->15={trans_14_15:2d} | "
                f"15->16={trans_15_16:2d} | "
                f"new_cov={stats.new_coverage_discoveries:2d} | "
                f"crashes={stats.unique_crashes:2d} | "
                f"exec/s={stats.exec_per_sec:5.1f} | "
                f"time={elapsed:.2f}s"
            )

    # Aggregate Analysis
    print("\n" + "=" * 92)
    print(" AGGREGATE RESULTS SUMMARY (Mean +/- Sample Std Dev across 5 trials)")
    print("=" * 92)
    header = (
        f"{'Condition':<16} | "
        f"{'Coverage Units':<16} | "
        f"{'Max Depth':<12} | "
        f"{'New Discov':<12} | "
        f"{'Corpus Size':<12} | "
        f"{'Crashes':<9} | "
        f"{'Total Reward':<12} | "
        f"{'Exec/s':<8}"
    )
    print(header)
    print("-" * 92)

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

        mean_reward = sum(r.total_reward for r in runs) / n
        sd_reward = std_dev([float(r.total_reward) for r in runs], mean_reward)

        mean_exec_rate = sum(r.exec_per_sec for r in runs) / n

        print(
            f"{cond:<16} | "
            f"{mean_cov:5.1f} +/- {sd_cov:4.1f}    | "
            f"{mean_depth:4.1f} +/- {sd_depth:3.1f}  | "
            f"{mean_new:4.1f} +/- {sd_new:3.1f}  | "
            f"{mean_corpus:4.1f} +/- {sd_corpus:3.1f}  | "
            f"{mean_crashes:3.1f} +/- {sd_crashes:3.1f} | "
            f"{mean_reward:5.1f} +/- {sd_reward:4.1f}  | "
            f"{mean_exec_rate:6.1f}"
        )

    # Detailed Depth Transition Matrix
    print("\n" + "=" * 92)
    print(" DEPTH TRANSITION OCCURRENCES SUMMARY (Total occurrences across all 5 trials)")
    print("=" * 92)
    all_transitions = ["14->15", "15->16", "16->17", "17->18", "18->19"]
    trans_header = f"{'Condition':<16} | " + " | ".join(f"{t:^8}" for t in all_transitions) + " | Total Deep Transitions"
    print(trans_header)
    print("-" * 92)

    for cond in CONDITIONS:
        runs = results[cond]
        counts = {t: sum(r.depth_transition_counts.get(t, 0) for r in runs) for t in all_transitions}
        total_deep = sum(counts.values())
        row_str = f"{cond:<16} | " + " | ".join(f"{counts[t]:^8d}" for t in all_transitions) + f" | {total_deep:^23d}"
        print(row_str)

    # UCB1 Operator Selection & Reward Breakdown (Condition B)
    print("\n" + "=" * 92)
    print(" UCB1 OPERATOR BREAKDOWN ACROSS 5 TRIALS (Condition B: UCB1 Policy)")
    print("=" * 92)
    ucb_runs = results["ucb1_policy"]
    arms = [
        "flip_bit",
        "replace_byte",
        "insert_byte",
        "delete_byte",
        "dictionary_insert_boundary",
        "dictionary_replace",
    ]
    print(f"{'Operator Name':<28} {'Mean Selections':<16} {'Selection Share':<16} {'Mean Reward':<14} {'Mean Cov Discov':<16}")
    print("-" * 92)
    for arm in arms:
        arm_pulls = [r.mutation_bandit_telemetry.get("operators", {}).get(arm, {}).get("selections", 0) for r in ucb_runs]
        arm_means = [r.mutation_bandit_telemetry.get("operators", {}).get(arm, {}).get("mean_reward", 0.0) for r in ucb_runs]
        arm_covs = [r.mutation_bandit_telemetry.get("operators", {}).get(arm, {}).get("coverage_discoveries", 0) for r in ucb_runs]

        avg_pulls = sum(arm_pulls) / len(ucb_runs)
        sd_pulls = std_dev(arm_pulls, avg_pulls)
        avg_share = avg_pulls / ITERATIONS
        avg_mean_r = sum(arm_means) / len(ucb_runs)
        avg_covs = sum(arm_covs) / len(ucb_runs)

        print(
            f"{arm:<28} "
            f"{avg_pulls:6.1f} +/- {sd_pulls:4.1f}   "
            f"{avg_share:>7.1%}           "
            f"{avg_mean_r:>10.4f}    "
            f"{avg_covs:>6.1f}"
        )

    print("=" * 92)


if __name__ == "__main__":
    main()
