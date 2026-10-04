"""Controlled experiment for Phase 6F: Compatibility-Aware Parent Selection Evaluation.

Directly compares:
- Condition A (Random Donor Selection - Phase 6E baseline):
    Structured field splicing with uniform random donor selection (donor_policy="random").
- Condition B (Compatibility-Aware Donor Selection - Phase 6F):
    Structured field splicing with deterministic structural compatibility donor selection (donor_policy="compatible").

Experimental Controls:
- Target: targets/structured_target.exe (19 depth levels, 6 deterministic bugs)
- Initial Corpus: corpus_structured/ (6 pristine seeds)
- Protocol Dictionary: dictionaries/structured_protocol.dict (85 tokens)
- Delimiter Boundary-Awareness: ENABLED across both conditions
- Splicing Probability: 0.20 strictly identical across both conditions
- Seed Scheduler: RandomScheduler (isolates donor selection effects from seed scheduling)
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

from fuzzer.fuzzer import Fuzzer, FuzzStats

TRIALS = [101, 202, 303, 404, 505]
ITERATIONS = 2500
TARGET = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus_structured"
DICTIONARY_FILE = PROJECT_ROOT / "dictionaries" / "structured_protocol.dict"
TIMEOUT = 1.0
CONDITIONS = ["random_donor", "compatible_donor"]


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    donor_policy = "compatible" if condition == "compatible_donor" else "random"

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
            enable_splicing=True,
            splice_probability=0.20,
            max_splice_size=512,
            donor_policy=donor_policy,
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
    print(" NeuroFuzz Phase 6F - Compatibility-Aware Parent Selection Benchmark")
    print("=" * 92)
    print(f"Target:              {TARGET}")
    print(f"Initial Corpus:      {INITIAL_CORPUS} (6 pristine seeds)")
    print(f"Dictionary:          {DICTIONARY_FILE} (85 tokens)")
    print(f"Budget per Trial:    {ITERATIONS} executions")
    print(f"Trials per Policy:   {len(TRIALS)} (Seeds: {TRIALS})")
    print(f"Conditions:          {CONDITIONS}")
    print(f"Splice Probability:  20.0% (strictly identical)")
    print("-" * 92)

    results: Dict[str, List[FuzzStats]] = {cond: [] for cond in CONDITIONS}
    start_all = time.time()

    for cond in CONDITIONS:
        cond_label = "Condition A (Random Donor)" if cond == "random_donor" else "Condition B (Compatible Donor)"
        print(f"\n>>> Running Benchmark for {cond_label}...")
        for trial_idx, seed_val in enumerate(TRIALS, start=1):
            t_start = time.time()
            stats = run_single_trial(cond, seed_val, iterations=ITERATIONS)
            t_elapsed = time.time() - t_start
            results[cond].append(stats)

            dt = stats.donor_telemetry
            print(
                f"  Trial {trial_idx}/{len(TRIALS)} (Seed {seed_val}): "
                f"Cov={stats.total_coverage_units:2d}, "
                f"MaxDepth={stats.deepest_state_reached:2d}, "
                f"14->15={stats.depth_transition_counts.get('14->15', 0)}, "
                f"Disc={stats.new_coverage_discoveries:2d}, "
                f"Corp={stats.corpus_size:2d}, "
                f"SpliceCov={stats.splice_new_coverage:2d}, "
                f"Crashes={stats.unique_crashes:2d}, "
                f"SameCmd={dt.get('same_command_pairing_rate', 0.0):.1%}, "
                f"Useful={dt.get('useful_donor_rate', 0.0):.1%}, "
                f"DIAG-DIAG={dt.get('diag_diag_pairings', 0)}, "
                f"Rate={stats.exec_per_sec:5.1f} exec/s ({t_elapsed:.1f}s)"
            )

    total_time = time.time() - start_all
    print("\n" + "=" * 92)
    print(" EXPERIMENT RESULTS & STATISTICAL COMPARISON")
    print("=" * 92)

    # Aggregation per condition
    summary_data: Dict[str, Dict[str, Any]] = {}
    for cond in CONDITIONS:
        stats_list = results[cond]
        n = len(stats_list)

        cov_list = [float(s.total_coverage_units) for s in stats_list]
        depth_list = [float(s.deepest_state_reached) for s in stats_list]
        disc_list = [float(s.new_coverage_discoveries) for s in stats_list]
        corp_list = [float(s.corpus_size) for s in stats_list]
        crash_list = [float(s.unique_crashes) for s in stats_list]
        rate_list = [float(s.exec_per_sec) for s in stats_list]
        splice_cov_list = [float(s.splice_new_coverage) for s in stats_list]

        t_14_15 = sum(s.depth_transition_counts.get("14->15", 0) for s in stats_list)
        t_15_16 = sum(s.depth_transition_counts.get("15->16", 0) for s in stats_list)
        t_16_17 = sum(s.depth_transition_counts.get("16->17", 0) for s in stats_list)
        t_17_18 = sum(s.depth_transition_counts.get("17->18", 0) for s in stats_list)
        t_18_19 = sum(s.depth_transition_counts.get("18->19", 0) for s in stats_list)

        same_cmd_rates = [float(s.donor_telemetry.get("same_command_pairing_rate", 0.0)) for s in stats_list]
        same_proto_rates = [float(s.donor_telemetry.get("same_protocol_pairing_rate", 0.0)) for s in stats_list]
        useful_rates = [float(s.donor_telemetry.get("useful_donor_rate", 0.0)) for s in stats_list]
        compat_scores = [float(s.donor_telemetry.get("average_compatibility_score", 0.0)) for s in stats_list]
        common_prefixes = [float(s.donor_telemetry.get("average_common_prefix_length", 0.0)) for s in stats_list]
        extra_fields = [float(s.donor_telemetry.get("average_donor_extra_fields", 0.0)) for s in stats_list]
        diag_diag = sum(s.donor_telemetry.get("diag_diag_pairings", 0) for s in stats_list)

        cov_m = sum(cov_list) / n
        depth_m = sum(depth_list) / n
        disc_m = sum(disc_list) / n
        corp_m = sum(corp_list) / n
        crash_m = sum(crash_list) / n
        rate_m = sum(rate_list) / n
        splice_cov_m = sum(splice_cov_list) / n

        same_cmd_m = sum(same_cmd_rates) / n
        same_proto_m = sum(same_proto_rates) / n
        useful_m = sum(useful_rates) / n
        compat_m = sum(compat_scores) / n
        prefix_m = sum(common_prefixes) / n
        extra_m = sum(extra_fields) / n

        summary_data[cond] = {
            "cov_mean": cov_m,
            "cov_std": std_dev(cov_list, cov_m),
            "depth_mean": depth_m,
            "depth_std": std_dev(depth_list, depth_m),
            "max_depth_seen": int(max(depth_list)),
            "disc_mean": disc_m,
            "disc_std": std_dev(disc_list, disc_m),
            "corp_mean": corp_m,
            "corp_std": std_dev(corp_list, corp_m),
            "crash_mean": crash_m,
            "crash_std": std_dev(crash_list, crash_m),
            "rate_mean": rate_m,
            "splice_cov_mean": splice_cov_m,
            "splice_cov_std": std_dev(splice_cov_list, splice_cov_m),
            "t_14_15": t_14_15,
            "t_15_16": t_15_16,
            "t_16_17": t_16_17,
            "t_17_18": t_17_18,
            "t_18_19": t_18_19,
            "same_cmd_mean": same_cmd_m,
            "same_proto_mean": same_proto_m,
            "useful_mean": useful_m,
            "compat_mean": compat_m,
            "prefix_mean": prefix_m,
            "extra_mean": extra_m,
            "diag_diag_total": diag_diag,
        }

    cond_a = summary_data["random_donor"]
    cond_b = summary_data["compatible_donor"]

    print(f"\n{'Metric':<34} {'Random Donor (Cond A)':<26} {'Compatible Donor (Cond B)':<26} {'Delta (B - A)':<14}")
    print("-" * 102)

    cov_delta = cond_b["cov_mean"] - cond_a["cov_mean"]
    print(
        f"{'Total Coverage Units':<34} "
        f"{cond_a['cov_mean']:5.1f} ± {cond_a['cov_std']:4.1f}               "
        f"{cond_b['cov_mean']:5.1f} ± {cond_b['cov_std']:4.1f}               "
        f"{cov_delta:>+6.1f} units"
    )

    depth_delta = cond_b["depth_mean"] - cond_a["depth_mean"]
    print(
        f"{'Max Protocol Depth Reached':<34} "
        f"{cond_a['depth_mean']:5.1f} ± {cond_a['depth_std']:4.1f}               "
        f"{cond_b['depth_mean']:5.1f} ± {cond_b['depth_std']:4.1f}               "
        f"{depth_delta:>+6.1f} levels"
    )

    print(
        f"{'Highest Single Depth Seen':<34} "
        f"Depth {cond_a['max_depth_seen']:<20} "
        f"Depth {cond_b['max_depth_seen']:<20} "
        f"{cond_b['max_depth_seen'] - cond_a['max_depth_seen']:>+6d} levels"
    )

    disc_delta = cond_b["disc_mean"] - cond_a["disc_mean"]
    print(
        f"{'Coverage Discoveries':<34} "
        f"{cond_a['disc_mean']:5.1f} ± {cond_a['disc_std']:4.1f}               "
        f"{cond_b['disc_mean']:5.1f} ± {cond_b['disc_std']:4.1f}               "
        f"{disc_delta:>+6.1f}"
    )

    corp_delta = cond_b["corp_mean"] - cond_a["corp_mean"]
    print(
        f"{'Final Corpus Size':<34} "
        f"{cond_a['corp_mean']:5.1f} ± {cond_a['corp_std']:4.1f}               "
        f"{cond_b['corp_mean']:5.1f} ± {cond_b['corp_std']:4.1f}               "
        f"{corp_delta:>+6.1f}"
    )

    crash_delta = cond_b["crash_mean"] - cond_a["crash_mean"]
    print(
        f"{'Unique Crashes Discovered':<34} "
        f"{cond_a['crash_mean']:5.1f} ± {cond_a['crash_std']:4.1f}               "
        f"{cond_b['crash_mean']:5.1f} ± {cond_b['crash_std']:4.1f}               "
        f"{crash_delta:>+6.1f}"
    )

    sp_cov_delta = cond_b["splice_cov_mean"] - cond_a["splice_cov_mean"]
    print(
        f"{'Coverage Discovered via Splicing':<34} "
        f"{cond_a['splice_cov_mean']:5.1f} ± {cond_a['splice_cov_std']:4.1f}               "
        f"{cond_b['splice_cov_mean']:5.1f} ± {cond_b['splice_cov_std']:4.1f}               "
        f"{sp_cov_delta:>+6.1f} units"
    )

    print(
        f"{'Throughput (exec/s)':<34} "
        f"{cond_a['rate_mean']:5.1f} /s                    "
        f"{cond_b['rate_mean']:5.1f} /s                    "
        f"{cond_b['rate_mean'] - cond_a['rate_mean']:>+6.1f} exec/s"
    )

    print("\n--- Pairing Quality & Structural Telemetry ---")
    print(
        f"{'Avg Compatibility Score':<34} "
        f"{cond_a['compat_mean']:5.3f}                       "
        f"{cond_b['compat_mean']:5.3f}                       "
        f"{cond_b['compat_mean'] - cond_a['compat_mean']:>+6.3f}"
    )
    print(
        f"{'Avg Common Prefix Length':<34} "
        f"{cond_a['prefix_mean']:5.2f} fields                "
        f"{cond_b['prefix_mean']:5.2f} fields                "
        f"{cond_b['prefix_mean'] - cond_a['prefix_mean']:>+6.2f} fields"
    )
    print(
        f"{'Same-Command Pairing Rate':<34} "
        f"{cond_a['same_cmd_mean']:5.1%}                       "
        f"{cond_b['same_cmd_mean']:5.1%}                       "
        f"{cond_b['same_cmd_mean'] - cond_a['same_cmd_mean']:>+6.1%}"
    )
    print(
        f"{'Same-Protocol Pairing Rate':<34} "
        f"{cond_a['same_proto_mean']:5.1%}                       "
        f"{cond_b['same_proto_mean']:5.1%}                       "
        f"{cond_b['same_proto_mean'] - cond_a['same_proto_mean']:>+6.1%}"
    )
    print(
        f"{'Useful Donor Rate (>0 extra)':<34} "
        f"{cond_a['useful_mean']:5.1%}                       "
        f"{cond_b['useful_mean']:5.1%}                       "
        f"{cond_b['useful_mean'] - cond_a['useful_mean']:>+6.1%}"
    )
    print(
        f"{'Avg Extra Fields Contributed':<34} "
        f"{cond_a['extra_mean']:5.2f} fields                "
        f"{cond_b['extra_mean']:5.2f} fields                "
        f"{cond_b['extra_mean'] - cond_a['extra_mean']:>+6.2f} fields"
    )
    print(
        f"{'Total DIAG-DIAG Pairings':<34} "
        f"{cond_a['diag_diag_total']:<26d} "
        f"{cond_b['diag_diag_total']:<26d} "
        f"{cond_b['diag_diag_total'] - cond_a['diag_diag_total']:>+6d}"
    )

    print("\n--- Protocol Depth Frontier Transitions Across All Trials ---")
    print(f"{'Transition':<16} {'Random Donor (Cond A)':<24} {'Compatible Donor (Cond B)':<24}")
    print("-" * 64)
    print(f"{'14->15':<16} {cond_a['t_14_15']:<24d} {cond_b['t_14_15']:<24d}")
    print(f"{'15->16':<16} {cond_a['t_15_16']:<24d} {cond_b['t_15_16']:<24d}")
    print(f"{'16->17':<16} {cond_a['t_16_17']:<24d} {cond_b['t_16_17']:<24d}")
    print(f"{'17->18':<16} {cond_a['t_17_18']:<24d} {cond_b['t_17_18']:<24d}")
    print(f"{'18->19':<16} {cond_a['t_18_19']:<24d} {cond_b['t_18_19']:<24d}")

    print(f"\nTotal Experiment Execution Time: {total_time:.1f} seconds")
    print("=" * 92)


if __name__ == "__main__":
    main()
