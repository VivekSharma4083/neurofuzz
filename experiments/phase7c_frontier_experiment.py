"""Phase 7C: State-Sequence-Aware Frontier Mutation Experiment.

Evaluates whether systematic frontier extension of deepest state sequences
can break through the Depth 16 ceiling and discover deeper protocol states
(Depth 17+) faster and more reliably than general-purpose mutation policies.

Comparison Arms:
  A. Baseline: Random scheduler + Fixed mutation (A_random_fixed)
  B. NeuroFuzz Full: LinearBandit + Contextual Bandit + Compatible Splicing (B_neurofuzz_full)
  C. Frontier Full: NeuroFuzz Full + State Frontier Extension (C_frontier_full)

RNG Seeds: [101, 202, 303, 404, 505]
Budget: 25,000 executions per trial (15 trials total = 375,000 executions)
Harness: Persistent executor
Corpus: corpus_structured (24 protocol seeds, baseline depth 14)
Dictionary: dictionaries/structured_protocol.dict (85 protocol tokens)
"""

import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fuzzer.coverage import extract_coverage_depth
from fuzzer.fuzzer import Fuzzer, FuzzStats

# Default paths
TARGET_PATH = PROJECT_ROOT / "targets" / "structured_target.exe"
INITIAL_CORPUS_DIR = PROJECT_ROOT / "corpus_structured"
DICTIONARY_PATH = PROJECT_ROOT / "dictionaries" / "structured_protocol.dict"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "phase7c"

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_ITERATIONS = 25000
TRAJECTORY_INTERVAL = 500

CONFIGURATIONS = {
    "A_random_fixed": {
        "label": "A: Random + Fixed Baseline",
        "scheduler_type": "random",
        "mutation_policy": "fixed",
        "enable_splicing": False,
        "donor_policy": "random",
        "enable_frontier_extension": False,
        "frontier_extension_rate": 0.0,
    },
    "B_neurofuzz_full": {
        "label": "B: NeuroFuzz Full (No Frontier)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "enable_splicing": True,
        "donor_policy": "compatible",
        "enable_frontier_extension": False,
        "frontier_extension_rate": 0.0,
    },
    "C_frontier_full": {
        "label": "C: NeuroFuzz Full + Frontier Extension",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "enable_splicing": True,
        "donor_policy": "compatible",
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
    },
}


def compute_auc(points: List[Tuple[int, float]]) -> float:
    """Compute Area Under Curve via trapezoidal rule for (x, y) points."""
    if len(points) < 2:
        return 0.0
    auc = 0.0
    for i in range(1, len(points)):
        x_prev, y_prev = points[i - 1]
        x_curr, y_curr = points[i]
        dx = x_curr - x_prev
        avg_y = (y_curr + y_prev) / 2.0
        auc += dx * avg_y
    return auc


class TrialTracker:
    """Tracks coverage, depth trajectory, and discovery artifacts during a trial."""

    def __init__(
        self,
        config_key: str,
        seed: int,
        trajectory_interval: int,
        discoveries_dir: Optional[Path] = None,
    ) -> None:
        self.config_key = config_key
        self.seed = seed
        self.trajectory_interval = trajectory_interval
        self.discoveries_dir = discoveries_dir

        self.time_to_depth: Dict[int, Optional[int]] = {d: None for d in range(14, 20)}
        self.time_to_depth[14] = 0  # Initial baseline corpus is Depth 14
        self.max_depth_seen: int = 14
        self.trajectory: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "coverage": 41,
                "max_depth": 14,
                "corpus_size": 24,
                "crashes": 0,
            }
        ]
        self.discoveries: List[Dict[str, Any]] = []

    def on_iteration(self, iteration: int, stats: FuzzStats, info: Dict[str, Any]) -> None:
        """Process callback from fuzzer loop."""
        child_depth = info.get("child_depth", 0)
        parent_depth = info.get("parent_depth", 0)
        op_used = info.get("op_used", "")
        mutated_bytes = info.get("mutated", b"")
        parent_record = info.get("selected_record")
        donor_bytes = info.get("donor")
        new_units = info.get("new_units", set())

        # Check time-to-depth metrics for levels 14..19
        for d in range(14, 20):
            if child_depth >= d and self.time_to_depth[d] is None:
                self.time_to_depth[d] = iteration

        # Check if a new maximum depth frontier (depth >= 15) was breached
        if child_depth > self.max_depth_seen:
            prev_max = self.max_depth_seen
            self.max_depth_seen = child_depth

            discovery_info = {
                "config": self.config_key,
                "seed": self.seed,
                "iteration": iteration,
                "depth": child_depth,
                "transition": f"{parent_depth}->{child_depth}",
                "previous_max_depth": prev_max,
                "operator": op_used,
                "parent_name": parent_record.name if parent_record else "unknown",
                "parent_payload": parent_record.data.decode("latin1", errors="replace") if parent_record else "",
                "mutated_payload": mutated_bytes.decode("latin1", errors="replace"),
                "donor_payload": donor_bytes.decode("latin1", errors="replace") if donor_bytes else None,
                "coverage_gained": len(new_units),
                "new_coverage_units": sorted(list(new_units)),
            }
            self.discoveries.append(discovery_info)

            # Save discovery artifacts to disk
            if self.discoveries_dir is not None:
                try:
                    cfg_disc_dir = self.discoveries_dir / f"{self.config_key}_seed{self.seed}"
                    cfg_disc_dir.mkdir(parents=True, exist_ok=True)
                    (cfg_disc_dir / f"depth{child_depth}.txt").write_bytes(mutated_bytes)

                    seed_disc_dir = self.discoveries_dir / f"seed{self.seed}"
                    seed_disc_dir.mkdir(parents=True, exist_ok=True)
                    (seed_disc_dir / f"depth{child_depth}.txt").write_bytes(mutated_bytes)
                except Exception as e:
                    print(f"Warning: could not write discovery artifact: {e}")

        # Periodic trajectory sampling
        if iteration % self.trajectory_interval == 0 or iteration == stats.total_executions:
            self.trajectory.append(
                {
                    "iteration": iteration,
                    "coverage": stats.total_coverage_units,
                    "max_depth": self.max_depth_seen,
                    "corpus_size": stats.corpus_size,
                    "crashes": stats.unique_crashes,
                }
            )


def run_single_trial(
    config_key: str,
    config_params: Dict[str, Any],
    seed: int,
    iterations: int,
    executor_type: str,
    output_dir: Path,
    silent: bool = True,
) -> Tuple[FuzzStats, TrialTracker]:
    """Execute a single fuzzing trial with full instrumentation and telemetry."""
    discoveries_dir = output_dir / "discoveries"
    tracker = TrialTracker(
        config_key=config_key,
        seed=seed,
        trajectory_interval=TRAJECTORY_INTERVAL,
        discoveries_dir=discoveries_dir,
    )

    with tempfile.TemporaryDirectory(prefix="neurofuzz_p7c_") as tmp_dir:
        tmp_corpus = Path(tmp_dir) / "corpus"
        tmp_corpus.mkdir(parents=True, exist_ok=True)
        for seed_file in INITIAL_CORPUS_DIR.glob("*.txt"):
            shutil.copy2(seed_file, tmp_corpus / seed_file.name)

        tmp_crashes = Path(tmp_dir) / "crashes"
        tmp_crashes.mkdir(parents=True, exist_ok=True)

        fuzzer = Fuzzer(
            target_path=TARGET_PATH,
            corpus_dir=str(tmp_corpus),
            iterations=iterations,
            timeout=1.0,
            seed=seed,
            stats_interval=5000,
            calibrate=True,
            scheduler_type=config_params["scheduler_type"],
            mutation_policy=config_params["mutation_policy"],
            dictionary_path=str(DICTIONARY_PATH),
            dictionary_probability=0.3,
            boundary_aware=True,
            enable_splicing=config_params["enable_splicing"],
            donor_policy=config_params["donor_policy"],
            enable_frontier_extension=config_params["enable_frontier_extension"],
            frontier_extension_rate=config_params["frontier_extension_rate"],
            crashes_dir=str(tmp_crashes),
            executor_type=executor_type,
            iteration_callback=tracker.on_iteration,
        )

        if silent:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                stats = fuzzer.run()
        else:
            stats = fuzzer.run()

    return stats, tracker


def run_experiment(
    iterations: int = DEFAULT_ITERATIONS,
    seeds: Optional[List[int]] = None,
    configs: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    executor_type: str = "persistent",
    smoke_test: bool = False,
) -> Path:
    """Execute the full Phase 7C controlled benchmark."""
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if configs is None:
        configs = list(CONFIGURATIONS.keys())
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if smoke_test:
        iterations = min(iterations, 500)
        print(f"[!] SMOKE TEST MODE: Reducing iterations to {iterations} per trial")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(parents=True, exist_ok=True)
    (output_dir / "discoveries").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    total_trials = len(configs) * len(seeds)
    print(f"\n{'='*70}")
    print(f" NeuroFuzz Phase 7C: State-Sequence-Aware Frontier Mutation Experiment")
    print(f"{'='*70}")
    print(f"Configurations ({len(configs)}):  {', '.join(configs)}")
    print(f"Seeds ({len(seeds)}):           {seeds}")
    print(f"Iterations / trial:  {iterations:,}")
    print(f"Total planned runs:  {total_trials} trials ({total_trials * iterations:,} executions)")
    print(f"Harness type:        {executor_type.upper()}")
    print(f"Output directory:    {output_dir}")
    print(f"{'='*70}\n")

    results_rows: List[Dict[str, Any]] = []
    all_discoveries: List[Dict[str, Any]] = []
    trial_count = 0
    start_all_time = time.time()

    for config_key in configs:
        cfg = CONFIGURATIONS[config_key]
        print(f"\n>>> Running Configuration: {config_key} ({cfg['label']})")

        for seed in seeds:
            trial_count += 1
            print(f"  [{trial_count:02d}/{total_trials:02d}] Config: {config_key:<20} Seed: {seed} ... ", end="", flush=True)

            t0 = time.time()
            stats, tracker = run_single_trial(
                config_key=config_key,
                config_params=cfg,
                seed=seed,
                iterations=iterations,
                executor_type=executor_type,
                output_dir=output_dir,
                silent=True,
            )
            elapsed = time.time() - t0
            exec_rate = stats.total_executions / elapsed if elapsed > 0 else 0.0

            # Compute AUCs
            cov_points = [(p["iteration"], float(p["coverage"])) for p in tracker.trajectory]
            depth_points = [(p["iteration"], float(p["max_depth"])) for p in tracker.trajectory]
            cov_auc = compute_auc(cov_points)
            depth_auc = compute_auc(depth_points)
            norm_cov_auc = cov_auc / (iterations * 84.0) if iterations > 0 else 0.0
            norm_depth_auc = depth_auc / (iterations * 19.0) if iterations > 0 else 0.0

            # State frontier stats
            sf_telemetry = stats.frontier_telemetry or {}

            row = {
                "config": config_key,
                "label": cfg["label"],
                "seed": seed,
                "iterations": stats.total_executions,
                "elapsed_sec": round(elapsed, 2),
                "exec_rate": round(exec_rate, 1),
                "final_coverage": stats.total_coverage_units,
                "new_coverage": stats.new_coverage_discoveries,
                "max_depth": tracker.max_depth_seen,
                "coverage_auc": round(cov_auc, 1),
                "depth_auc": round(depth_auc, 1),
                "norm_cov_auc": round(norm_cov_auc, 4),
                "norm_depth_auc": round(norm_depth_auc, 4),
                "t14": tracker.time_to_depth[14],
                "t15": tracker.time_to_depth[15],
                "t16": tracker.time_to_depth[16],
                "t17": tracker.time_to_depth[17],
                "t18": tracker.time_to_depth[18],
                "t19": tracker.time_to_depth[19],
                "unique_crashes": stats.unique_crashes,
                "total_crashes": stats.crashes_found,
                "final_corpus_size": stats.corpus_size,
                "dict_mutations": stats.dictionary_mutations,
                "boundary_mutations": stats.boundary_mutations,
                "splice_mutations": stats.splice_mutations,
                "frontier_mode": stats.enable_frontier_extension,
                "frontier_rate": cfg["frontier_extension_rate"],
                "frontier_attempts": stats.frontier_extension_attempts,
                "frontier_unique_candidates": sf_telemetry.get("unique_candidates_created", 0),
                "frontier_cov_discoveries": sf_telemetry.get("coverage_discoveries", 0),
                "frontier_depth_discoveries": sf_telemetry.get("depth_discoveries", 0),
                "frontier_success_rate": round(sf_telemetry.get("successful_candidate_rate", 0.0), 4),
                "frontier_avg_gen_cost_ms": round(sf_telemetry.get("avg_candidate_gen_cost_ms", 0.0), 3),
            }
            results_rows.append(row)
            all_discoveries.extend(tracker.discoveries)

            # Save individual trajectory
            traj_file = output_dir / "trajectories" / f"{config_key}_seed{seed}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(tracker.trajectory, f, indent=2)

            t17_str = f"T17={tracker.time_to_depth[17]}" if tracker.time_to_depth[17] is not None else "T17=None"
            print(
                f"Done in {elapsed:.1f}s ({exec_rate:.1f} exec/s) | "
                f"Cov: {stats.total_coverage_units} | Depth: {tracker.max_depth_seen} ({t17_str}) | "
                f"Crashes: {stats.unique_crashes}"
            )

    total_elapsed = time.time() - start_all_time
    print(f"\n[+] All {total_trials} trials completed in {total_elapsed:.1f}s ({total_trials * iterations / total_elapsed:.1f} exec/s avg)")

    # 1. Write results.csv
    csv_file = output_dir / "results.csv"
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"[+] Saved trial summary to {csv_file}")

    # 2. Write discoveries.json
    disc_file = output_dir / "discoveries.json"
    with open(disc_file, "w", encoding="utf-8") as f:
        json.dump(all_discoveries, f, indent=2)
    print(f"[+] Saved {len(all_discoveries)} depth frontier discoveries to {disc_file}")

    # 3. Aggregate summary stats
    summary: Dict[str, Any] = {
        "phase": "7C",
        "description": "State-Sequence-Aware Frontier Mutation Experiment",
        "iterations_per_trial": iterations,
        "executor_type": executor_type,
        "seeds": seeds,
        "total_executions": total_trials * iterations,
        "total_elapsed_sec": round(total_elapsed, 2),
        "configurations": {},
    }

    for config_key in configs:
        cfg_rows = [r for r in results_rows if r["config"] == config_key]
        if not cfg_rows:
            continue
        n = len(cfg_rows)

        def mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        def std(vals: List[float], m: float) -> float:
            return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0

        covs = [float(r["final_coverage"]) for r in cfg_rows]
        depths = [float(r["max_depth"]) for r in cfg_rows]
        cov_aucs = [float(r["coverage_auc"]) for r in cfg_rows]
        depth_aucs = [float(r["depth_auc"]) for r in cfg_rows]
        crashes = [float(r["unique_crashes"]) for r in cfg_rows]
        rates = [float(r["exec_rate"]) for r in cfg_rows]

        m_cov = mean(covs)
        s_cov = std(covs, m_cov)
        m_dep = mean(depths)
        s_dep = std(depths, m_dep)
        m_cauc = mean(cov_aucs)
        s_cauc = std(cov_aucs, m_cauc)
        m_dauc = mean(depth_aucs)
        s_dauc = std(depth_aucs, m_dauc)
        m_cra = mean(crashes)
        s_cra = std(crashes, m_cra)
        m_rate = mean(rates)
        s_rate = std(rates, m_rate)

        # Transition reach counts & mean arrival times
        transitions_summary = {}
        for d in range(15, 20):
            reached = [r[f"t{d}"] for r in cfg_rows if r[f"t{d}"] is not None]
            reach_count = len(reached)
            avg_time = mean([float(t) for t in reached]) if reach_count > 0 else None
            transitions_summary[f"reach_depth_{d}"] = {
                "count": reach_count,
                "rate": reach_count / n,
                "mean_iterations": round(avg_time, 1) if avg_time is not None else None,
                "times": reached,
            }

        # Frontier telemetry aggregation
        f_att = [float(r["frontier_attempts"]) for r in cfg_rows]
        f_cov = [float(r["frontier_cov_discoveries"]) for r in cfg_rows]
        f_dep = [float(r["frontier_depth_discoveries"]) for r in cfg_rows]

        summary["configurations"][config_key] = {
            "label": CONFIGURATIONS[config_key]["label"],
            "trials": n,
            "mean_coverage": round(m_cov, 2),
            "std_coverage": round(s_cov, 2),
            "mean_max_depth": round(m_dep, 2),
            "std_max_depth": round(s_dep, 2),
            "mean_coverage_auc": round(m_cauc, 1),
            "std_coverage_auc": round(s_cauc, 1),
            "mean_depth_auc": round(m_dauc, 1),
            "std_depth_auc": round(s_dauc, 1),
            "mean_unique_crashes": round(m_cra, 2),
            "std_unique_crashes": round(s_cra, 2),
            "mean_throughput": round(m_rate, 1),
            "std_throughput": round(s_rate, 1),
            "transitions": transitions_summary,
            "frontier_telemetry": {
                "mean_attempts": round(mean(f_att), 1),
                "mean_cov_discoveries": round(mean(f_cov), 2),
                "mean_depth_discoveries": round(mean(f_dep), 2),
            },
        }

    sum_file = output_dir / "summary.json"
    with open(sum_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved aggregated summary to {sum_file}")

    return output_dir


def generate_plots(output_dir: Path) -> None:
    """Generate high-resolution comparative publication plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib / numpy not available. Skipping plot generation.")
        return

    csv_file = output_dir / "results.csv"
    if not csv_file.exists():
        print(f"Warning: {csv_file} does not exist. Cannot plot.")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    configs = list(CONFIGURATIONS.keys())
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    colors = {
        "A_random_fixed": "#6c757d",      # Gray
        "B_neurofuzz_full": "#0d6efd",    # Blue
        "C_frontier_full": "#198754",     # Green
    }
    short_labels = {
        "A_random_fixed": "A: Baseline (Random+Fixed)",
        "B_neurofuzz_full": "B: NeuroFuzz Full",
        "C_frontier_full": "C: Full + Frontier Extension",
    }

    # Load trajectories
    trajectories: Dict[str, Dict[int, List[Dict[str, Any]]]] = {c: {} for c in configs}
    for c in configs:
        for r in rows:
            if r["config"] == c:
                seed = int(r["seed"])
                t_file = output_dir / "trajectories" / f"{c}_seed{seed}.json"
                if t_file.exists():
                    with open(t_file, "r", encoding="utf-8") as f:
                        trajectories[c][seed] = json.load(f)

    # 1. Coverage Growth Over Time
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for c in configs:
        seed_data = trajectories[c]
        if not seed_data:
            continue
        iters = [p["iteration"] for p in next(iter(seed_data.values()))]
        cov_matrix = np.array([[p["coverage"] for p in traj] for traj in seed_data.values()])
        mean_cov = np.mean(cov_matrix, axis=0)
        std_cov = np.std(cov_matrix, axis=0)

        ax.plot(iters, mean_cov, label=short_labels[c], color=colors[c], linewidth=2.5)
        ax.fill_between(iters, mean_cov - std_cov, mean_cov + std_cov, color=colors[c], alpha=0.15)

    ax.set_title("Phase 7C: Coverage Growth Over Execution Budget (Mean ± Std)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Unique Coverage Units", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_vs_iterations.png", dpi=300)
    plt.close()
    print("  [+] Generated coverage_vs_iterations.png")

    # 2. Maximum Depth Over Time
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for c in configs:
        seed_data = trajectories[c]
        if not seed_data:
            continue
        iters = [p["iteration"] for p in next(iter(seed_data.values()))]
        depth_matrix = np.array([[p["max_depth"] for p in traj] for traj in seed_data.values()])
        mean_depth = np.mean(depth_matrix, axis=0)

        # Draw individual seed paths in light color
        for traj in seed_data.values():
            ax.plot([p["iteration"] for p in traj], [p["max_depth"] for p in traj], color=colors[c], alpha=0.2, linewidth=1)

        ax.step(iters, mean_depth, where="post", label=short_labels[c], color=colors[c], linewidth=2.5)

    ax.axhline(14, color="#adb5bd", linestyle=":", label="Baseline Frontier (Depth 14)")
    ax.axhline(16, color="#dc3545", linestyle="--", label="Phase 7B Plateau (Depth 16)")
    ax.axhline(17, color="#20c997", linestyle="-.", label="Phase 7C Objective (Depth 17)")
    ax.set_title("Phase 7C: Maximum State Depth Discovery Trajectory", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Maximum Protocol Depth Level", fontsize=11)
    ax.set_yticks(range(13, 20))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "depth_vs_iterations.png", dpi=300)
    plt.close()
    print("  [+] Generated depth_vs_iterations.png")

    # 3. Time-to-Depth 17 Comparison (T17)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    t17_vals = {}
    for c in configs:
        t17_list = []
        for r in rows:
            if r["config"] == c:
                t17 = r.get("t17")
                t17_list.append(int(t17) if t17 and t17 != "" and t17 != "None" else None)
        t17_vals[c] = t17_list

    bar_width = 0.5
    x_pos = np.arange(len(configs))
    t17_means = []
    t17_reach_counts = []
    for c in configs:
        valid = [v for v in t17_vals[c] if v is not None]
        t17_reach_counts.append(len(valid))
        t17_means.append(np.mean(valid) if valid else 25000)

    bars = ax.bar(x_pos, t17_means, width=bar_width, color=[colors[c] for c in configs], alpha=0.85, edgecolor="black")
    for i, (bar, count) in enumerate(zip(bars, t17_reach_counts)):
        height = bar.get_height()
        label_text = f"Mean: {height:.0f} iter\n({count}/5 reached)" if count > 0 else f"NOT REACHED\n(0/5 reached)"
        ax.text(bar.get_x() + bar.get_width() / 2.0, height / 2.0, label_text, ha="center", va="center", color="white" if count > 0 else "black", fontweight="bold", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([short_labels[c] for c in configs], fontsize=9)
    ax.set_title("Phase 7C: Time-to-Depth 17 (T17) by Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Iterations to Reach Depth 17 (Lower is Better)", fontsize=10)
    ax.set_ylim(0, 27000)
    ax.axhline(25000, color="red", linestyle="--", alpha=0.7, label="Max Budget (25k)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(plots_dir / "t17_comparison.png", dpi=300)
    plt.close()
    print("  [+] Generated t17_comparison.png")

    # 4. Coverage AUC Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    auc_means = []
    auc_stds = []
    for c in configs:
        aucs = [float(r["norm_cov_auc"]) for r in rows if r["config"] == c]
        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs))

    bars = ax.bar(x_pos, auc_means, yerr=auc_stds, width=bar_width, color=[colors[c] for c in configs], capsize=5, alpha=0.85, edgecolor="black")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.015, f"{height:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([short_labels[c] for c in configs], fontsize=9)
    ax.set_title("Phase 7C: Normalized Coverage AUC (Discovery Velocity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Normalized Coverage AUC (Higher is Better)", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_auc_comparison.png", dpi=300)
    plt.close()
    print("  [+] Generated coverage_auc_comparison.png")

    # 5. Frontier Discovery Timeline & Depth Transition Counts
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    disc_file = output_dir / "discoveries.json"
    if disc_file.exists():
        with open(disc_file, "r", encoding="utf-8") as f:
            discs = json.load(f)

        transitions = ["14->15", "15->16", "16->17", "17->18", "18->19"]
        counts_by_cfg = {c: {t: 0 for t in transitions} for c in configs}
        for d in discs:
            c = d.get("config")
            t = d.get("transition")
            if c in counts_by_cfg and t in counts_by_cfg[c]:
                counts_by_cfg[c][t] += 1

        width = 0.25
        t_x = np.arange(len(transitions))
        for i, c in enumerate(configs):
            vals = [counts_by_cfg[c][t] for t in transitions]
            ax.bar(t_x + (i - 1) * width, vals, width=width, label=short_labels[c], color=colors[c], edgecolor="black")

        ax.set_xticks(t_x)
        ax.set_xticklabels([f"Transition\n{t}" for t in transitions], fontsize=10)
        ax.set_ylabel("Total Transition Discoveries Across 5 Trials", fontsize=10)
        ax.set_title("Phase 7C: Depth Frontier Transition Counts Across Benchmark Trials", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(plots_dir / "frontier_discovery_timeline.png", dpi=300)
        plt.close()
        print("  [+] Generated frontier_discovery_timeline.png")


def main() -> None:
    """CLI entrypoint for Phase 7C experiment."""
    parser = argparse.ArgumentParser(description="NeuroFuzz Phase 7C: State Frontier Mutation Experiment")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Executions per trial (default: 25000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="RNG seeds to evaluate")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGURATIONS.keys()), help="Configurations to evaluate")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Results output directory")
    parser.add_argument("--executor", choices=["subprocess", "persistent"], default="persistent", help="Execution harness")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test with 500 iterations")
    parser.add_argument("--plot-only", action="store_true", help="Generate plots from existing results without running trials")
    args = parser.parse_args()

    if args.plot_only:
        print(f"Generating plots from {args.output_dir}...")
        generate_plots(args.output_dir)
        return

    out_dir = run_experiment(
        iterations=args.iterations,
        seeds=args.seeds,
        configs=args.configs,
        output_dir=args.output_dir,
        executor_type=args.executor,
        smoke_test=args.smoke_test,
    )

    print("\nGenerating comparative plots...")
    generate_plots(out_dir)
    print(f"\n[+] Phase 7C Experiment Completed! All artifacts preserved under {out_dir}")


if __name__ == "__main__":
    main()
