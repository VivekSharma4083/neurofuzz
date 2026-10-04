"""Controlled experiment for Phase 7A: Persistent Execution Harness Evaluation.

Directly compares:
- Condition A (Subprocess Executor - Baseline):
    One-shot subprocess.run execution launching a fresh target process per input.
- Condition B (Persistent Executor - Phase 7A):
    Long-lived persistent child process with framed pipe IPC, timeout protection,
    and automatic crash recovery.

Experimental Controls:
- Target: targets/structured_target.exe (19 depth levels, 6 deterministic bugs)
- Initial Corpus: corpus_structured/ (6 pristine seeds)
- Protocol Dictionary: dictionaries/structured_protocol.dict (85 tokens)
- Delimiter Boundary-Awareness: ENABLED across both conditions
- Splicing: ENABLED (compatible donor) across both conditions
- Seed Scheduler: LinearBanditScheduler
- Trials: 3 independent RNG seeds (101, 202, 303)
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

from fuzzer.executor import Executor
from fuzzer.fuzzer import Fuzzer, FuzzStats
from fuzzer.persistent_executor import PersistentExecutor

TRIALS = [101, 202, 303]
ITERATIONS = 2500
RAW_BENCHMARK_ITERATIONS = 2000
TARGET = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS = PROJECT_ROOT / "corpus_structured"
DICTIONARY_FILE = PROJECT_ROOT / "dictionaries" / "structured_protocol.dict"
TIMEOUT = 1.0
CONDITIONS = ["subprocess", "persistent"]


def run_raw_throughput_benchmark(target_path: Path, iterations: int = RAW_BENCHMARK_ITERATIONS) -> Dict[str, float]:
    """Measure raw harness execution throughput independent of mutators/schedulers."""
    print("\n" + "=" * 70)
    print(f" RAW HARNESS THROUGHPUT BENCHMARK ({iterations} executions)")
    print("=" * 70)

    test_inputs = [
        b"NF01|PING",
        b"NF01|INFO|VERBOSE",
        b"NF01|CALC|OP=ADD|NUM=100|DEN=5",
        b"NF01|DIAG|STEP=1",
        b"NF01|DIAG|STEP=1|STEP=2",
    ]

    results = {}

    # 1. Subprocess Executor
    print("Running Subprocess Executor raw benchmark...")
    sub_exec = Executor(target_path, timeout=TIMEOUT)
    t0 = time.perf_counter()
    for i in range(iterations):
        inp = test_inputs[i % len(test_inputs)]
        sub_exec.run(inp)
    t1 = time.perf_counter()
    sub_duration = t1 - t0
    sub_rate = iterations / sub_duration
    results["subprocess_rate"] = sub_rate
    results["subprocess_time"] = sub_duration
    print(f"  Subprocess: {iterations} execs in {sub_duration:.2f}s -> {sub_rate:.1f} exec/s")

    # 2. Persistent Executor
    print("Running Persistent Executor raw benchmark...")
    pers_exec = PersistentExecutor(target_path, timeout=TIMEOUT)
    t0 = time.perf_counter()
    for i in range(iterations):
        inp = test_inputs[i % len(test_inputs)]
        pers_exec.execute(inp)
    t1 = time.perf_counter()
    pers_exec.close()
    pers_duration = t1 - t0
    pers_rate = iterations / pers_duration
    results["persistent_rate"] = pers_rate
    results["persistent_time"] = pers_duration
    print(f"  Persistent: {iterations} execs in {pers_duration:.2f}s -> {pers_rate:.1f} exec/s")

    raw_speedup = pers_rate / max(1e-6, sub_rate)
    results["raw_speedup"] = raw_speedup
    print(f"  >>> RAW HARNESS SPEEDUP: {raw_speedup:.2f}x <<<\n")

    return results


def run_single_trial(
    condition: str,
    rng_seed: int,
    iterations: int = ITERATIONS,
) -> FuzzStats:
    """Run a single fuzzing trial in an isolated directory and return stats."""
    executor_type = condition

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
            stats_interval=1000,
            calibrate=True,
            scheduler_type="linear_bandit",
            epsilon=0.20,
            dictionary_path=DICTIONARY_FILE,
            dictionary_probability=0.30,
            boundary_aware=True,
            mutation_policy="contextual",
            enable_splicing=True,
            splice_probability=0.20,
            donor_policy="compatible",
            executor_type=executor_type,
        )

        # Silence stdout during trial runs
        with contextlib.redirect_stdout(io.StringIO()):
            stats = fuzzer.run()

        return stats


def compute_mean_std(values: List[float]) -> Dict[str, float]:
    """Compute arithmetic mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def main():
    print("=" * 70)
    print(" NeuroFuzz Phase 7A: Persistent Execution Harness Evaluation")
    print("=" * 70)
    print(f"Target:               {TARGET}")
    print(f"Initial Corpus:       {INITIAL_CORPUS}")
    print(f"Protocol Dictionary:  {DICTIONARY_FILE}")
    print(f"Fuzzing Trials:       {len(TRIALS)} (Seeds: {TRIALS})")
    print(f"Iterations per Trial: {ITERATIONS}")
    print(f"Conditions:           {CONDITIONS}")
    print("=" * 70)

    # Step 1: Raw harness benchmark
    raw_results = run_raw_throughput_benchmark(TARGET, iterations=RAW_BENCHMARK_ITERATIONS)

    # Step 2: Full end-to-end fuzzing trials
    trial_data: Dict[str, List[FuzzStats]] = {cond: [] for cond in CONDITIONS}

    for cond in CONDITIONS:
        print(f"\n[*] Evaluating Condition: '{cond.upper()}' ...")
        for trial_idx, seed in enumerate(TRIALS, 1):
            t_start = time.perf_counter()
            stats = run_single_trial(cond, rng_seed=seed, iterations=ITERATIONS)
            elapsed = time.perf_counter() - t_start

            trial_data[cond].append(stats)
            print(
                f"  Trial {trial_idx}/{len(TRIALS)} (Seed {seed:3d}): "
                f"Cov={stats.total_coverage_units:2d} | "
                f"MaxDepth={stats.deepest_state_reached:2d} | "
                f"Crashes={stats.unique_crashes:2d} | "
                f"Rate={stats.exec_per_sec:5.1f} exec/s | "
                f"WallClock={elapsed:5.2f}s"
            )

    # Step 3: Statistical summary
    print("\n" + "=" * 70)
    print(" STATISTICAL COMPARISON SUMMARY")
    print("=" * 70)

    metrics_to_summarize = [
        ("Throughput (exec/s)", lambda s: s.exec_per_sec, "{:.1f}"),
        ("Wall-Clock Time (s)", lambda s: s.elapsed_time, "{:.2f}"),
        ("Total Coverage Units", lambda s: float(s.total_coverage_units), "{:.1f}"),
        ("Max Depth Reached", lambda s: float(s.deepest_state_reached), "{:.1f}"),
        ("Unique Crashes Found", lambda s: float(s.unique_crashes), "{:.1f}"),
        ("Timeouts Encountered", lambda s: float(s.timeouts_found), "{:.1f}"),
        ("New Coverage Discoveries", lambda s: float(s.new_coverage_discoveries), "{:.1f}"),
    ]

    summary_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    for metric_name, extractor, _ in metrics_to_summarize:
        summary_stats[metric_name] = {}
        for cond in CONDITIONS:
            vals = [extractor(s) for s in trial_data[cond]]
            summary_stats[metric_name][cond] = compute_mean_std(vals)

    # Print Table
    header = f"{'Metric':<26} | {'Subprocess (A)':<18} | {'Persistent (B)':<18} | {'Delta / Speedup':<16}"
    print(header)
    print("-" * len(header))

    for metric_name, _, fmt in metrics_to_summarize:
        a_stats = summary_stats[metric_name]["subprocess"]
        b_stats = summary_stats[metric_name]["persistent"]

        a_str = f"{fmt.format(a_stats['mean'])} ± {fmt.format(a_stats['std'])}"
        b_str = f"{fmt.format(b_stats['mean'])} ± {fmt.format(b_stats['std'])}"

        if "Throughput" in metric_name:
            speedup = b_stats["mean"] / max(1e-6, a_stats["mean"])
            delta_str = f"{speedup:.2f}x faster"
        elif "Wall-Clock" in metric_name:
            speedup = a_stats["mean"] / max(1e-6, b_stats["mean"])
            delta_str = f"{speedup:.2f}x speedup"
        else:
            diff = b_stats["mean"] - a_stats["mean"]
            sign = "+" if diff >= 0 else ""
            delta_str = f"{sign}{diff:.2f}"

        print(f"{metric_name:<26} | {a_str:<18} | {b_str:<18} | {delta_str:<16}")

    print("=" * 70)
    print(" KEY SCIENTIFIC FINDINGS:")
    sub_tp = summary_stats["Throughput (exec/s)"]["subprocess"]["mean"]
    pers_tp = summary_stats["Throughput (exec/s)"]["persistent"]["mean"]
    fuzz_speedup = pers_tp / max(1e-6, sub_tp)

    print(f" 1. Raw Harness Speedup:     {raw_results['raw_speedup']:.2f}x ({raw_results['persistent_rate']:.1f} vs {raw_results['subprocess_rate']:.1f} exec/s)")
    print(f" 2. End-to-End Fuzz Speedup: {fuzz_speedup:.2f}x ({pers_tp:.1f} vs {sub_tp:.1f} exec/s)")
    print(f" 3. Subprocess 2500 Execs:   {summary_stats['Wall-Clock Time (s)']['subprocess']['mean']:.2f}s")
    print(f" 4. Persistent 2500 Execs:   {summary_stats['Wall-Clock Time (s)']['persistent']['mean']:.2f}s")
    cov_a = summary_stats["Total Coverage Units"]["subprocess"]["mean"]
    cov_b = summary_stats["Total Coverage Units"]["persistent"]["mean"]
    depth_a = summary_stats["Max Depth Reached"]["subprocess"]["mean"]
    depth_b = summary_stats["Max Depth Reached"]["persistent"]["mean"]
    print(f" 5. Coverage Consistency:    Subprocess = {cov_a:.1f}, Persistent = {cov_b:.1f}")
    print(f" 6. Max Depth Consistency:   Subprocess = {depth_a:.1f}, Persistent = {depth_b:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
