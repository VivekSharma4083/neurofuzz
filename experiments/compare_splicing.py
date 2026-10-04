"""Controlled experiment for Phase 6E: Structured Crossover & Field Splicing Evaluation.

Directly compares:
- Condition A (Baseline without Splicing):
    Phase 6B fixed policy (dict_prob=0.20, boundary_aware=True, enable_splicing=False)
- Condition B (With Structured Field Splicing):
    Phase 6E structured crossover (dict_prob=0.20, boundary_aware=True, enable_splicing=True, splice_prob=0.20)

Experimental Controls:
- Target: targets/structured_target.exe (19 depth levels, 6 deterministic bugs)
- Initial Corpus: corpus_structured/ (6 pristine seeds)
- Protocol Dictionary: dictionaries/structured_protocol.dict (85 tokens)
- Delimiter Boundary-Awareness: ENABLED across both conditions
- Seed Scheduler: RandomScheduler (isolates mutation/crossover effects from seed scheduling)
- Trials: 5 independent RNG seeds (101, 202, 303, 404, 505)
- Budget: 2,500 iterations per trial
- Isolation: Fresh temporary directory per trial
"""

import contextlib
import io
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time
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
CONDITIONS = ["baseline_no_splicing", "structured_field_splicing"]


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    enable_splicing = condition == "structured_field_splicing"
    splice_prob = 0.20 if enable_splicing else 0.0

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
            stats_interval=99999,  # suppress periodic terminal logs
            calibrate=True,
            scheduler_type="random",
            dictionary_path=DICTIONARY_FILE,
            dictionary_probability=0.20,
            boundary_aware=True,
            mutation_policy="fixed",
            enable_splicing=enable_splicing,
            splice_probability=splice_prob,
            max_splice_size=512,
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
    print(" NeuroFuzz Phase 6E - Structured Crossover & Field Splicing Evaluation")
    print("=" * 88)
    print(f"Target:        {TARGET.name}")
    print(f"Initial Seeds: {INITIAL_CORPUS.name}/ ({len(list(INITIAL_CORPUS.glob('seed*.txt')))} seeds)")
    print(f"Dictionary:    {DICTIONARY_FILE.name} ({len(Dictionary(filepath=DICTIONARY_FILE))} tokens)")
    print(f"Scheduler:     RandomScheduler (isolated mutation & crossover comparison)")
    print(f"Iterations:    {ITERATIONS} per trial")
    print(f"Trials:        {len(TRIALS)} (seeds: {TRIALS})")
    print(f"Conditions:    Baseline (No Splicing) vs Structured Field Splicing (Phase 6E)")
    print(f"Environment:   Isolated temporary workspace per trial")
    print("=" * 88)

    results: Dict[str, List[FuzzStats]] = {c: [] for c in CONDITIONS}
    overall_start = time.time()

    for cond in CONDITIONS:
        cond_label = "Condition A: Baseline (No Splicing)" if cond == "baseline_no_splicing" else "Condition B: Structured Field Splicing"
        print(f"\n>>> Running {cond_label} across {len(TRIALS)} seeds...")

        for idx, seed_val in enumerate(TRIALS, 1):
            t0 = time.time()
            print(f"  [Trial {idx}/{len(TRIALS)}] RNG Seed: {seed_val}... ", end="", flush=True)
            stats = run_single_trial(cond, seed_val, iterations=ITERATIONS)
            results[cond].append(stats)
            elapsed = time.time() - t0

            trans_14_15 = stats.depth_transition_counts.get("14->15", 0)
            splice_info = f" | Splices: {stats.splice_mutations} (Cov: +{stats.splice_new_coverage})" if stats.enable_splicing else ""
            print(
                f"Done ({elapsed:5.1f}s) | Cov: {stats.total_coverage_units:2d} | "
                f"MaxDepth: {stats.deepest_state_reached:2d} | 14->15: {trans_14_15} | "
                f"Corpus: {stats.corpus_size:2d} | Crashes: {stats.crashes_found:2d}{splice_info}"
            )

    total_time = time.time() - overall_start
    print("\n" + "=" * 88)
    print(f" EXPERIMENTAL RESULTS SUMMARY (Total Benchmark Time: {total_time:.1f}s)")
    print("=" * 88)

    # Detailed Comparison Table
    print(f"\n{'Metric':<32} {'Baseline (No Splicing)':<26} {'Field Splicing (Phase 6E)':<26}")
    print("-" * 88)

    def format_mean_std(values: List[float], fmt: str = "{:.1f}") -> str:
        mean_v = sum(values) / len(values)
        sd_v = std_dev(values, mean_v)
        return f"{fmt.format(mean_v)} +/- {fmt.format(sd_v)}"

    cov_base = [float(s.total_coverage_units) for s in results["baseline_no_splicing"]]
    cov_splc = [float(s.total_coverage_units) for s in results["structured_field_splicing"]]
    print(f"{'Final Coverage Units':<32} {format_mean_std(cov_base):<26} {format_mean_std(cov_splc):<26}")

    depth_base = [float(s.deepest_state_reached) for s in results["baseline_no_splicing"]]
    depth_splc = [float(s.deepest_state_reached) for s in results["structured_field_splicing"]]
    print(f"{'Max Depth Reached':<32} {format_mean_std(depth_base):<26} {format_mean_std(depth_splc):<26}")

    t14_15_base = [float(s.depth_transition_counts.get("14->15", 0)) for s in results["baseline_no_splicing"]]
    t14_15_splc = [float(s.depth_transition_counts.get("14->15", 0)) for s in results["structured_field_splicing"]]
    print(f"{'14->15 Transitions / Trial':<32} {format_mean_std(t14_15_base, '{:.2f}'):<26} {format_mean_std(t14_15_splc, '{:.2f}'):<26}")

    t15_16_base = [float(s.depth_transition_counts.get("15->16", 0)) for s in results["baseline_no_splicing"]]
    t15_16_splc = [float(s.depth_transition_counts.get("15->16", 0)) for s in results["structured_field_splicing"]]
    print(f"{'15->16 Transitions / Trial':<32} {format_mean_std(t15_16_base, '{:.2f}'):<26} {format_mean_std(t15_16_splc, '{:.2f}'):<26}")

    disc_base = [float(s.new_coverage_discoveries) for s in results["baseline_no_splicing"]]
    disc_splc = [float(s.new_coverage_discoveries) for s in results["structured_field_splicing"]]
    print(f"{'New Cov Discoveries / Trial':<32} {format_mean_std(disc_base):<26} {format_mean_std(disc_splc):<26}")

    corp_base = [float(s.corpus_size) for s in results["baseline_no_splicing"]]
    corp_splc = [float(s.corpus_size) for s in results["structured_field_splicing"]]
    print(f"{'Corpus Size':<32} {format_mean_std(corp_base):<26} {format_mean_std(corp_splc):<26}")

    crash_base = [float(s.crashes_found) for s in results["baseline_no_splicing"]]
    crash_splc = [float(s.crashes_found) for s in results["structured_field_splicing"]]
    print(f"{'Total Crashes Found':<32} {format_mean_std(crash_base):<26} {format_mean_std(crash_splc):<26}")

    ucrash_base = [float(s.unique_crashes) for s in results["baseline_no_splicing"]]
    ucrash_splc = [float(s.unique_crashes) for s in results["structured_field_splicing"]]
    print(f"{'Unique Crashes Found':<32} {format_mean_std(ucrash_base):<26} {format_mean_std(ucrash_splc):<26}")

    exec_base = [s.exec_per_sec for s in results["baseline_no_splicing"]]
    exec_splc = [s.exec_per_sec for s in results["structured_field_splicing"]]
    print(f"{'Throughput (exec/s)':<32} {format_mean_std(exec_base):<26} {format_mean_std(exec_splc):<26}")

    # Splicer Specific Telemetry
    print("\n--- Structured Field Splicing Breakdown (Condition B) ---")
    splicing_stats = results["structured_field_splicing"]
    tot_splice_muts = sum(s.splice_mutations for s in splicing_stats)
    tot_splice_cov = sum(s.splice_new_coverage for s in splicing_stats)
    tot_splice_crashes = sum(s.splice_crashes for s in splicing_stats)
    print(f"Total Splice Mutations:    {tot_splice_muts} ({tot_splice_muts / (ITERATIONS * len(TRIALS)):.1%} of all mutations)")
    print(f"Splice New Coverage Units: {tot_splice_cov} units ({tot_splice_cov / max(1, sum(disc_splc)):.1%} of discoveries)")
    print(f"Splice Crashes Discovered: {tot_splice_crashes}")

    # Aggregated Splicer Telemetry
    agg_attempts = sum(s.splice_telemetry.get("splice_attempts", 0) for s in splicing_stats)
    agg_success = sum(s.splice_telemetry.get("successful_splices", 0) for s in splicing_stats)
    agg_rejected = sum(s.splice_telemetry.get("rejected_splices", 0) for s in splicing_stats)
    agg_dups = sum(s.splice_telemetry.get("duplicate_prevention_events", 0) for s in splicing_stats)
    agg_14_15 = sum(s.splice_telemetry.get("transition_14_15_count", 0) for s in splicing_stats)

    print(f"Aggregated Splice Attempts:{agg_attempts}")
    print(f"Successful Splices:        {agg_success} ({agg_success / max(1, agg_attempts):.1%})")
    print(f"Rejected Splices:          {agg_rejected}")
    print(f"Duplicate Prevention Evts: {agg_dups}")
    print(f"14->15 Transitions (Splice):{agg_14_15}")

    agg_strats: Dict[str, int] = {}
    for s in splicing_stats:
        for strat_k, strat_v in s.splice_telemetry.get("strategy_counts", {}).items():
            agg_strats[strat_k] = agg_strats.get(strat_k, 0) + strat_v

    print("Strategy Selection Counts:")
    for strat_k, strat_v in sorted(agg_strats.items()):
        print(f"  - {strat_k:<24}: {strat_v} ({strat_v / max(1, agg_success):.1%})")

    # Per-Trial Breakdown Table
    print("\n--- Per-Trial Breakdown ---")
    print(f"{'Seed':<8} {'Cond A Cov':<12} {'Cond A Depth':<14} {'Cond B Cov':<12} {'Cond B Depth':<14} {'Cond B Splices':<16} {'Cond B 14->15':<14}")
    print("-" * 90)
    for i, seed_val in enumerate(TRIALS):
        sa = results["baseline_no_splicing"][i]
        sb = results["structured_field_splicing"][i]
        print(
            f"{seed_val:<8} "
            f"{sa.total_coverage_units:<12} "
            f"{sa.deepest_state_reached:<14} "
            f"{sb.total_coverage_units:<12} "
            f"{sb.deepest_state_reached:<14} "
            f"{sb.splice_mutations:<16} "
            f"{sb.depth_transition_counts.get('14->15', 0):<14}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
