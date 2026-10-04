"""Controlled experiment for Phase 6D: Contextual Mutation Bandit Evaluation.

Directly compares 3 mutation policies:
1. Fixed Policy (Phase 6B baseline: fixed probabilities, dict_prob=0.20, boundary_aware=True)
2. UCB1 Policy (Phase 6C global bandit: UCB1 algorithm, c=1.0)
3. Contextual Policy (Phase 6D learned: OnlineLinearModel per arm, eps=0.20, eta=0.05, lambda=0.001)

Experimental Controls:
- Target: targets/structured_target.exe (19 depth levels, 6 deterministic bugs)
- Initial Corpus: corpus_structured/ (6 pristine seeds)
- Protocol Dictionary: dictionaries/structured_protocol.dict (85 tokens)
- Delimiter Boundary-Awareness: ENABLED across all 3 conditions
- Seed Scheduler: RandomScheduler (isolates mutation operator selection from seed scheduling)
- Trials: 5 independent RNG seeds (101, 202, 303, 404, 505)
- Budget: 2,500 iterations per trial
- Isolation: Fresh temporary directory per trial
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
CONDITIONS = ["fixed_policy", "ucb1_policy", "contextual_policy"]


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    if condition == "ucb1_policy":
        mut_policy = "ucb1"
    elif condition == "contextual_policy":
        mut_policy = "contextual"
    else:
        mut_policy = "fixed"

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
            mutation_epsilon=0.20,
            mutation_learning_rate=0.05,
            mutation_l2_reg=0.001,
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
    print("=" * 100)
    print(" NeuroFuzz Phase 6D - Contextual Mutation Bandit Evaluation")
    print("=" * 100)
    print(f"Target:        {TARGET.name}")
    print(f"Initial Seeds: {INITIAL_CORPUS.name}/ ({len(list(INITIAL_CORPUS.glob('seed*.txt')))} seeds)")
    print(f"Dictionary:    {DICTIONARY_FILE.name} ({len(Dictionary(filepath=DICTIONARY_FILE))} tokens)")
    print(f"Boundary Mode: ENABLED (snapped to delimiter boundaries)")
    print(f"Scheduler:     RandomScheduler (isolated mutation comparison)")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Conditions:    Fixed vs UCB1 (Phase 6C) vs Contextual (Phase 6D)")
    print(f"Environment:   Isolated temporary workspace per trial")
    print("=" * 100)

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
                f"  {cond:<18s} | "
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
    print("\n" + "=" * 100)
    print(" AGGREGATE RESULTS SUMMARY (Mean +/- Sample Std Dev across 5 trials)")
    print("=" * 100)
    header = (
        f"{'Condition':<18} | "
        f"{'Coverage Units':<16} | "
        f"{'Max Depth':<12} | "
        f"{'New Discov':<12} | "
        f"{'Corpus Size':<12} | "
        f"{'Crashes':<9} | "
        f"{'Total Reward':<12} | "
        f"{'Exec/s':<8}"
    )
    print(header)
    print("-" * len(header))

    for cond in CONDITIONS:
        trials = results[cond]
        n = len(trials)
        covs = [float(s.total_coverage_units) for s in trials]
        depths = [float(s.deepest_state_reached) for s in trials]
        discovs = [float(s.new_coverage_discoveries) for s in trials]
        corpus_sizes = [float(s.corpus_size) for s in trials]
        crashes = [float(s.unique_crashes) for s in trials]
        rewards = [float(s.total_reward) for s in trials]
        rates = [float(s.exec_per_sec) for s in trials]

        m_cov, s_cov = sum(covs) / n, std_dev(covs, sum(covs) / n)
        m_depth, s_depth = sum(depths) / n, std_dev(depths, sum(depths) / n)
        m_discov, s_discov = sum(discovs) / n, std_dev(discovs, sum(discovs) / n)
        m_corp, s_corp = sum(corpus_sizes) / n, std_dev(corpus_sizes, sum(corpus_sizes) / n)
        m_crash, s_crash = sum(crashes) / n, std_dev(crashes, sum(crashes) / n)
        m_rwd, s_rwd = sum(rewards) / n, std_dev(rewards, sum(rewards) / n)
        m_rate, s_rate = sum(rates) / n, std_dev(rates, sum(rates) / n)

        print(
            f"{cond:<18} | "
            f"{m_cov:4.1f} +/- {s_cov:4.1f}   | "
            f"{m_depth:4.1f} +/- {s_depth:3.1f}   | "
            f"{m_discov:4.1f} +/- {s_discov:3.1f}   | "
            f"{m_corp:4.1f} +/- {s_corp:3.1f}   | "
            f"{m_crash:3.1f} +/- {s_crash:3.1f} | "
            f"{m_rwd:4.1f} +/- {s_rwd:4.1f}   | "
            f"{m_rate:5.1f}"
        )

    # Depth Transitions Comparison
    print("\n" + "=" * 100)
    print(" CRITICAL FRONTIER DEPTH TRANSITIONS (Total occurrences across all 5 trials)")
    print("=" * 100)
    print(f"{'Condition':<18} | {'14->15':<10} | {'15->16':<10} | {'16->17':<10} | {'17->18':<10} | {'18->19':<10}")
    print("-" * 75)
    for cond in CONDITIONS:
        trials = results[cond]
        t14_15 = sum(s.depth_transition_counts.get("14->15", 0) for s in trials)
        t15_16 = sum(s.depth_transition_counts.get("15->16", 0) for s in trials)
        t16_17 = sum(s.depth_transition_counts.get("16->17", 0) for s in trials)
        t17_18 = sum(s.depth_transition_counts.get("17->18", 0) for s in trials)
        t18_19 = sum(s.depth_transition_counts.get("18->19", 0) for s in trials)
        print(f"{cond:<18} | {t14_15:<10} | {t15_16:<10} | {t16_17:<10} | {t17_18:<10} | {t18_19:<10}")

    # Contextual Mutation Bandit Telemetry Summary (if available)
    contextual_trials = results["contextual_policy"]
    if contextual_trials and contextual_trials[0].contextual_mutation_bandit_telemetry:
        print("\n" + "=" * 100)
        print(" CONTEXTUAL MUTATION OPERATOR TELEMETRY (Averaged across 5 trials)")
        print("=" * 100)
        # Aggregate operator pulls and rewards across trials
        op_agg: Dict[str, Dict[str, float]] = {}
        total_pulls_all = 0
        for s in contextual_trials:
            c_tel = s.contextual_mutation_bandit_telemetry
            ops = c_tel.get("operators", {})
            for op_name, op_data in ops.items():
                if op_name not in op_agg:
                    op_agg[op_name] = {
                        "selections": 0.0,
                        "total_reward": 0.0,
                        "nonzero": 0.0,
                        "cov_units": 0.0,
                        "mse": 0.0,
                    }
                op_agg[op_name]["selections"] += op_data["selections"]
                op_agg[op_name]["total_reward"] += op_data["total_reward"]
                op_agg[op_name]["nonzero"] += op_data["nonzero_rewards"]
                op_agg[op_name]["cov_units"] += op_data["coverage_discoveries"]
                op_agg[op_name]["mse"] += op_data.get("model_mse", 0.0)
                total_pulls_all += op_data["selections"]

        print(f"{'Operator Name':<28} {'Mean Pulls':<12} {'Share':<10} {'Mean Reward':<14} {'Nonzero':<10} {'Cov Found':<12} {'Mean MSE':<10}")
        print("-" * 100)
        num_trials = len(contextual_trials)
        sorted_agg = sorted(op_agg.items(), key=lambda kv: kv[1]["selections"], reverse=True)
        for op_name, data in sorted_agg:
            mean_pulls = data["selections"] / num_trials
            share = data["selections"] / max(1.0, total_pulls_all)
            mean_rew = data["total_reward"] / max(1.0, data["selections"])
            mean_nz = data["nonzero"] / num_trials
            mean_cov = data["cov_units"] / num_trials
            mean_mse = data["mse"] / num_trials
            print(
                f"{op_name:<28} "
                f"{mean_pulls:<12.1f} "
                f"{share:<10.1%} "
                f"{mean_rew:<14.4f} "
                f"{mean_nz:<10.1f} "
                f"{mean_cov:<12.1f} "
                f"{mean_mse:<10.4f}"
            )

        # Average weights for each operator
        print("\n--- Average Learned Weights per Operator across 5 Trials ---")
        first_ops = contextual_trials[0].contextual_mutation_bandit_telemetry.get("operators", {})
        if first_ops:
            op_sample = next(iter(first_ops.values()))
            feat_names = list(op_sample.get("weights", {}).keys())
            header = f"{'Operator':<26} " + " ".join(f"{fn:>10}" for fn in feat_names)
            print(header)
            print("-" * len(header))
            for op_name, _ in sorted_agg:
                avg_weights = {fn: 0.0 for fn in feat_names}
                for s in contextual_trials:
                    w = s.contextual_mutation_bandit_telemetry.get("operators", {}).get(op_name, {}).get("weights", {})
                    for fn in feat_names:
                        avg_weights[fn] += w.get(fn, 0.0)
                avg_str = " ".join(f"{avg_weights[fn] / num_trials:>+10.4f}" for fn in feat_names)
                print(f"{op_name:<26} {avg_str}")

    print("=" * 100)


if __name__ == "__main__":
    main()
