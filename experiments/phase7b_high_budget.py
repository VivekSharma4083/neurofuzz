"""Phase 7B: High-Budget Fuzzing & Deep-State Discovery Experiment.

Evaluates whether NeuroFuzz's learned mutation and scheduling policies discover
deeper program states more effectively than conventional baselines when provided
with substantial execution budgets under the high-throughput persistent harness.

Configurations Evaluated:
  A. Random seed scheduler + fixed mutation policy
  B. Heuristic seed scheduler + fixed mutation policy
  C. Linear Bandit seed scheduler + fixed mutation policy
  D. Random seed scheduler + UCB1 mutation policy
  E. Random seed scheduler + contextual mutation policy
  F. Random seed scheduler + contextual mutation policy + compatible donor selection
  G. NeuroFuzz Full (Linear Bandit + contextual mutation policy + compatible donor selection)

Metrics:
  - Total executions & elapsed wall-clock time
  - Total unique coverage & new coverage discoveries
  - Maximum depth reached & Time-to-Depth (T14, T15, T16, T17, T18, T19)
  - Area Under Coverage Curve (Coverage AUC) & Depth AUC
  - Unique crashes & crash-triggering inputs
  - Corpus size & execution throughput (exec/s)
  - Periodic depth/coverage discovery trajectory tracking
  - Discovery artifact preservation for newly breached depth frontiers
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "phase7b"

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_ITERATIONS = 25000
TRAJECTORY_INTERVAL = 500  # snapshot coverage and max_depth every 500 executions

CONFIGURATIONS = {
    "A_random_fixed": {
        "label": "A: Random + Fixed",
        "scheduler_type": "random",
        "mutation_policy": "fixed",
        "enable_splicing": False,
        "donor_policy": "random",
    },
    "B_heuristic_fixed": {
        "label": "B: Heuristic + Fixed",
        "scheduler_type": "heuristic",
        "mutation_policy": "fixed",
        "enable_splicing": False,
        "donor_policy": "random",
    },
    "C_linear_bandit_fixed": {
        "label": "C: LinearBandit + Fixed",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "fixed",
        "enable_splicing": False,
        "donor_policy": "random",
    },
    "D_random_ucb1": {
        "label": "D: Random + UCB1",
        "scheduler_type": "random",
        "mutation_policy": "ucb1",
        "enable_splicing": False,
        "donor_policy": "random",
    },
    "E_random_contextual": {
        "label": "E: Random + Contextual",
        "scheduler_type": "random",
        "mutation_policy": "contextual",
        "enable_splicing": False,
        "donor_policy": "random",
    },
    "F_random_contextual_compatible": {
        "label": "F: Random + Contextual + Compatible",
        "scheduler_type": "random",
        "mutation_policy": "contextual",
        "enable_splicing": True,
        "donor_policy": "compatible",
    },
    "G_neurofuzz_full": {
        "label": "G: NeuroFuzz Full (Bandit + Contextual + Compatible)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "enable_splicing": True,
        "donor_policy": "compatible",
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
        auc += avg_y * dx
    return auc


def compute_mean_std(values: List[float]) -> Dict[str, float]:
    """Compute arithmetic mean, sample standard deviation, min, and max."""
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


class TrialTracker:
    """Tracks step-by-step discoveries, trajectories, and time-to-depth metrics for one trial."""

    def __init__(
        self,
        config_key: str,
        seed: int,
        trajectory_interval: int = TRAJECTORY_INTERVAL,
        discoveries_dir: Optional[Path] = None,
    ) -> None:
        self.config_key = config_key
        self.seed = seed
        self.trajectory_interval = trajectory_interval
        self.discoveries_dir = discoveries_dir

        self.time_to_depth: Dict[int, Optional[int]] = {d: None for d in range(14, 20)}
        self.time_to_depth[14] = 0  # Initial baseline corpus (seed6) is Depth 14
        self.max_depth_seen: int = 14
        self.trajectory: List[Dict[str, Any]] = [
            {
                "iteration": 0,
                "coverage": 41,
                "max_depth": 14,
                "corpus_size": 6,
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

        # Check if a new maximum depth frontier beyond baseline (depth >= 15) was breached
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

            # Save discovery artifacts to disk (Section 13)
            if self.discoveries_dir is not None:
                try:
                    # 1. Config and seed-specific directory
                    cfg_disc_dir = self.discoveries_dir / f"{self.config_key}_seed{self.seed}"
                    cfg_disc_dir.mkdir(parents=True, exist_ok=True)
                    (cfg_disc_dir / f"depth{child_depth}.txt").write_bytes(mutated_bytes)

                    # 2. Seed-specific directory (as requested in Section 13: discoveries/seed101/depth15.txt)
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


def run_single_high_budget_trial(
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

    with tempfile.TemporaryDirectory() as tmp_corpus_dir, tempfile.TemporaryDirectory() as tmp_crashes_dir:
        tmp_corpus = Path(tmp_corpus_dir)
        tmp_crashes = Path(tmp_crashes_dir)

        # Copy pristine structured seeds
        for seed_file in INITIAL_CORPUS_DIR.glob("seed*.txt"):
            shutil.copy2(seed_file, tmp_corpus / seed_file.name)

        fuzzer = Fuzzer(
            target_path=TARGET_PATH,
            corpus_dir=tmp_corpus,
            crashes_dir=tmp_crashes,
            iterations=iterations,
            timeout=1.0,
            seed=seed,
            stats_interval=5000,
            calibrate=True,
            scheduler_type=config_params["scheduler_type"],
            epsilon=0.20,
            learning_rate=0.05,
            l2_reg=0.001,
            dictionary_path=DICTIONARY_PATH,
            dictionary_probability=0.30,
            boundary_aware=True,
            mutation_policy=config_params["mutation_policy"],
            ucb_c=1.0,
            mutation_epsilon=0.20,
            mutation_learning_rate=0.05,
            mutation_l2_reg=0.001,
            enable_splicing=config_params["enable_splicing"],
            splice_probability=0.20 if config_params["enable_splicing"] else 0.0,
            donor_policy=config_params["donor_policy"],
            executor_type=executor_type,
            iteration_callback=tracker.on_iteration,
        )

        # Record post-calibration baseline
        if silent:
            with contextlib.redirect_stdout(io.StringIO()):
                stats = fuzzer.run()
        else:
            stats = fuzzer.run()

        # Ensure final point is in trajectory
        if not tracker.trajectory or tracker.trajectory[-1]["iteration"] != stats.total_executions:
            tracker.trajectory.append(
                {
                    "iteration": stats.total_executions,
                    "coverage": stats.total_coverage_units,
                    "max_depth": tracker.max_depth_seen,
                    "corpus_size": stats.corpus_size,
                    "crashes": stats.unique_crashes,
                }
            )

        return stats, tracker


def generate_plots(output_dir: Path, summary_data: Dict[str, Any], trajectory_data: Dict[str, Dict[int, List[Dict[str, Any]]]]):
    """Generate scientific comparison charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[!] Matplotlib not installed; skipping plot generation.")
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    configs = list(CONFIGURATIONS.keys())
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
    color_map = {cfg: colors[idx % len(colors)] for idx, cfg in enumerate(configs)}

    # Plot 1: Coverage vs Executions (Mean trajectory with shaded bounds)
    plt.figure(figsize=(10, 6))
    for cfg in configs:
        if cfg not in trajectory_data:
            continue
        seed_trajs = trajectory_data[cfg]
        all_iters = sorted(list({pt["iteration"] for s in seed_trajs for pt in seed_trajs[s]}))
        mean_covs = []
        std_covs = []
        for it in all_iters:
            covs_at_it = []
            for s in seed_trajs:
                pts = [pt["coverage"] for pt in seed_trajs[s] if pt["iteration"] == it]
                if pts:
                    covs_at_it.append(pts[0])
            if covs_at_it:
                stats = compute_mean_std(covs_at_it)
                mean_covs.append(stats["mean"])
                std_covs.append(stats["std"])
            else:
                mean_covs.append(mean_covs[-1] if mean_covs else 0.0)
                std_covs.append(0.0)

        plt.plot(all_iters, mean_covs, label=CONFIGURATIONS[cfg]["label"], color=color_map[cfg], linewidth=2.0)
        plt.fill_between(
            all_iters,
            [m - s for m, s in zip(mean_covs, std_covs)],
            [m + s for m, s in zip(mean_covs, std_covs)],
            color=color_map[cfg],
            alpha=0.15,
        )

    plt.title("Coverage Discovery vs Executions (High-Budget Phase 7B)", fontsize=14, fontweight="bold")
    plt.xlabel("Executions", fontsize=12)
    plt.ylabel("Cumulative Coverage Units", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_vs_executions.png", dpi=200)
    plt.close()

    # Plot 2: Maximum Depth vs Executions
    plt.figure(figsize=(10, 6))
    for cfg in configs:
        if cfg not in trajectory_data:
            continue
        seed_trajs = trajectory_data[cfg]
        all_iters = sorted(list({pt["iteration"] for s in seed_trajs for pt in seed_trajs[s]}))
        mean_depths = []
        for it in all_iters:
            depths_at_it = []
            for s in seed_trajs:
                pts = [pt["max_depth"] for pt in seed_trajs[s] if pt["iteration"] == it]
                if pts:
                    depths_at_it.append(pts[0])
            mean_depths.append(sum(depths_at_it) / len(depths_at_it) if depths_at_it else 0.0)

        plt.plot(all_iters, mean_depths, label=CONFIGURATIONS[cfg]["label"], color=color_map[cfg], linewidth=2.0)

    plt.title("Protocol Depth Frontier vs Executions", fontsize=14, fontweight="bold")
    plt.xlabel("Executions", fontsize=12)
    plt.ylabel("Maximum Depth Reached", fontsize=12)
    plt.yticks(range(0, 20, 2))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "max_depth_vs_executions.png", dpi=200)
    plt.close()

    # Plot 3: Final Coverage Comparison (Bar chart)
    plt.figure(figsize=(10, 5))
    labels = [CONFIGURATIONS[c]["label"] for c in configs if c in summary_data]
    means = [summary_data[c]["coverage"]["mean"] for c in configs if c in summary_data]
    stds = [summary_data[c]["coverage"]["std"] for c in configs if c in summary_data]
    bar_colors = [color_map[c] for c in configs if c in summary_data]

    bars = plt.bar(range(len(labels)), means, yerr=stds, capsize=5, color=bar_colors, alpha=0.85)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Final Coverage Units", fontsize=12)
    plt.title("Final Code Coverage Comparison Across Configurations", fontsize=14, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "final_coverage_comparison.png", dpi=200)
    plt.close()

    # Plot 4: Final Maximum Depth Comparison (Bar chart)
    plt.figure(figsize=(10, 5))
    means_d = [summary_data[c]["max_depth"]["mean"] for c in configs if c in summary_data]
    stds_d = [summary_data[c]["max_depth"]["std"] for c in configs if c in summary_data]

    bars = plt.bar(range(len(labels)), means_d, yerr=stds_d, capsize=5, color=bar_colors, alpha=0.85)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Maximum Depth Level", fontsize=12)
    plt.yticks(range(0, 20, 2))
    plt.title("Deepest Program State Discovered (Max Depth)", fontsize=14, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.2, f"{yval:.1f}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "final_depth_comparison.png", dpi=200)
    plt.close()

    # Plot 5: Time-to-Depth Comparison (T14 and T15)
    plt.figure(figsize=(10, 5))
    t14_means = []
    for c in configs:
        if c in summary_data:
            t14_vals = summary_data[c].get("t14_values", [])
            t14_means.append(sum(t14_vals) / len(t14_vals) if t14_vals else 0.0)

    plt.bar(range(len(labels)), t14_means, color=bar_colors, alpha=0.85)
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    plt.ylabel("Executions to First Reach Depth 14 (T14)", fontsize=12)
    plt.title("Search Efficiency: Time-to-Depth 14", fontsize=14, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plots_dir / "time_to_depth.png", dpi=200)
    plt.close()

    # Plot 6: Operator Selection Distribution (for UCB1, Contextual, NeuroFuzz)
    learned_cfgs = [c for c in ["D_random_ucb1", "E_random_contextual", "F_random_contextual_compatible", "G_neurofuzz_full"] if c in summary_data]
    if learned_cfgs:
        plt.figure(figsize=(11, 6))
        all_ops = ["flip_bit", "replace_byte", "insert_byte", "delete_byte", "dictionary_insert_boundary", "dictionary_replace", "field_splice"]
        op_colors = ["#4e79a7", "#a0cbe8", "#f28e2c", "#ffbe7d", "#59a14f", "#8cd17d", "#b07aa1"]

        bottoms = [0.0] * len(learned_cfgs)
        cfg_labels = [CONFIGURATIONS[c]["label"] for c in learned_cfgs]

        for op_idx, op_name in enumerate(all_ops):
            shares = []
            for c in learned_cfgs:
                dist = summary_data[c].get("operator_distribution", {})
                shares.append(dist.get(op_name, 0.0))
            plt.bar(range(len(learned_cfgs)), shares, bottom=bottoms, label=op_name, color=op_colors[op_idx % len(op_colors)], width=0.55)
            bottoms = [b + s for b, s in zip(bottoms, shares)]

        plt.xticks(range(len(learned_cfgs)), cfg_labels, rotation=25, ha="right", fontsize=9)
        plt.ylabel("Proportion of Operator Pulls", fontsize=12)
        plt.title("Learned Mutation Operator Selection Shares", fontsize=14, fontweight="bold")
        plt.ylim(0, 1.05)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=9)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(plots_dir / "operator_selection_distribution.png", dpi=200)
        plt.close()

    print(f"[+] Scientific plots saved to: {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="NeuroFuzz Phase 7B High-Budget Fuzzing Experiment")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help=f"Execution budget per trial (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)), help="Comma-separated RNG seeds")
    parser.add_argument("--executor", type=str, default="persistent", choices=["persistent", "subprocess"], help="Execution harness (default: persistent)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--configs", type=str, default="", help="Comma-separated subset of config keys to run (default: all)")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 1,000-iteration sanity smoke test")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    discoveries_dir = output_dir / "discoveries"
    discoveries_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    iterations = args.iterations

    if args.smoke_test:
        print("[*] SMOKE TEST MODE ACTIVATED: 1,000 iterations on first 2 seeds...")
        iterations = 1000
        seeds = seeds[:2]

    configs_to_run = list(CONFIGURATIONS.keys())
    if args.configs:
        selected = [c.strip() for c in args.configs.split(",") if c.strip()]
        configs_to_run = [c for c in configs_to_run if c in selected]

    total_runs = len(configs_to_run) * len(seeds)

    print("=" * 76)
    print(" NEUROFUZZ PHASE 7B — HIGH-BUDGET FUZZING & DEEP-STATE DISCOVERY")
    print("=" * 76)
    print(f"Target:               {TARGET_PATH}")
    print(f"Harness:              {args.executor.upper()}")
    print(f"Configurations ({len(configs_to_run)}):")
    for k in configs_to_run:
        print(f"  - {k}: {CONFIGURATIONS[k]['label']}")
    print(f"RNG Seeds ({len(seeds)}):        {seeds}")
    print(f"Budget per Trial:     {iterations:,} executions")
    print(f"Total Fuzzing Runs:   {total_runs} trials ({total_runs * iterations:,} total executions)")
    print(f"Output Directory:     {output_dir}")
    print("=" * 76)

    # Telemetry storage
    all_trial_rows: List[Dict[str, Any]] = []
    trajectory_data: Dict[str, Dict[int, List[Dict[str, Any]]]] = {c: {} for c in configs_to_run}
    all_discoveries: List[Dict[str, Any]] = []
    all_operator_distributions: Dict[str, Dict[str, float]] = {}
    config_telemetries: Dict[str, List[Dict[str, Any]]] = {}

    run_idx = 0
    t_suite_start = time.perf_counter()

    for config_key in configs_to_run:
        cfg = CONFIGURATIONS[config_key]
        print(f"\n{'#' * 76}")
        print(f" CONFIGURATION: {cfg['label']}")
        print(f"{'#' * 76}")

        config_operator_counts: Dict[str, int] = {}
        total_config_operator_pulls = 0

        for seed in seeds:
            run_idx += 1
            print(f"\n---> [Run {run_idx}/{total_runs}] Config: {config_key} | Seed: {seed} | Budget: {iterations:,} ...")
            t0 = time.perf_counter()

            stats, tracker = run_single_high_budget_trial(
                config_key=config_key,
                config_params=cfg,
                seed=seed,
                iterations=iterations,
                executor_type=args.executor,
                output_dir=output_dir,
                silent=True,
            )
            wall_clock = time.perf_counter() - t0

            # Store trajectories
            trajectory_data[config_key][seed] = tracker.trajectory
            traj_csv = trajectories_dir / f"{config_key}_seed{seed}.csv"
            with open(traj_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["iteration", "coverage", "max_depth", "corpus_size", "crashes"])
                writer.writeheader()
                writer.writerows(tracker.trajectory)

            # Collect discoveries
            all_discoveries.extend(tracker.discoveries)

            # Compute AUC metrics
            cov_pts = [(pt["iteration"], float(pt["coverage"])) for pt in tracker.trajectory]
            depth_pts = [(pt["iteration"], float(pt["max_depth"])) for pt in tracker.trajectory]
            cov_auc = compute_auc(cov_pts)
            depth_auc = compute_auc(depth_pts)
            norm_cov_auc = cov_auc / max(1.0, float(stats.total_executions))
            norm_depth_auc = depth_auc / max(1.0, float(stats.total_executions))

            # Operator distribution telemetry
            if stats.mutation_bandit_telemetry and "arm_statistics" in stats.mutation_bandit_telemetry:
                for arm, arm_stat in stats.mutation_bandit_telemetry["arm_statistics"].items():
                    pulls = arm_stat.get("pulls", 0)
                    config_operator_counts[arm] = config_operator_counts.get(arm, 0) + pulls
                    total_config_operator_pulls += pulls
            elif stats.contextual_mutation_bandit_telemetry and "arm_statistics" in stats.contextual_mutation_bandit_telemetry:
                for arm, arm_stat in stats.contextual_mutation_bandit_telemetry["arm_statistics"].items():
                    pulls = arm_stat.get("pulls", 0)
                    config_operator_counts[arm] = config_operator_counts.get(arm, 0) + pulls
                    total_config_operator_pulls += pulls

            # Compile trial record
            row = {
                "config": config_key,
                "label": cfg["label"],
                "seed": seed,
                "total_executions": stats.total_executions,
                "total_coverage": stats.total_coverage_units,
                "new_coverage_discoveries": stats.new_coverage_discoveries,
                "max_depth": tracker.max_depth_seen,
                "t14": tracker.time_to_depth[14],
                "t15": tracker.time_to_depth[15],
                "t16": tracker.time_to_depth[16],
                "t17": tracker.time_to_depth[17],
                "t18": tracker.time_to_depth[18],
                "t19": tracker.time_to_depth[19],
                "unique_crashes": stats.unique_crashes,
                "crash_inputs": stats.crashes_found,
                "corpus_size": stats.corpus_size,
                "exec_per_sec": stats.exec_per_sec,
                "elapsed_time": stats.elapsed_time,
                "coverage_auc": cov_auc,
                "coverage_auc_normalized": norm_cov_auc,
                "depth_auc": depth_auc,
                "depth_auc_normalized": norm_depth_auc,
                "dictionary_mutations": stats.dictionary_mutations,
                "boundary_mutations": stats.boundary_mutations,
                "splice_mutations": stats.splice_mutations,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            all_trial_rows.append(row)

            t15_str = str(tracker.time_to_depth[15]) if tracker.time_to_depth[15] is not None else "NOT REACHED"
            print(
                f"      Done in {wall_clock:5.2f}s ({stats.exec_per_sec:5.1f} exec/s) | "
                f"Cov={stats.total_coverage_units:2d} | "
                f"MaxDepth={tracker.max_depth_seen:2d} | "
                f"T14={tracker.time_to_depth[14]} | "
                f"T15={t15_str} | "
                f"Crashes={stats.unique_crashes:2d} | "
                f"CovAUC={norm_cov_auc:5.1f} | DepthAUC={norm_depth_auc:4.1f}"
            )

            if total_config_operator_pulls > 0:
                all_operator_distributions[config_key] = {
                    arm: count / total_config_operator_pulls for arm, count in config_operator_counts.items()
                }

            # Capture latest learning telemetry for summary
            if config_key not in config_telemetries:
                config_telemetries[config_key] = []
            config_telemetries[config_key].append({
                "seed": seed,
                "scheduler_telemetry": stats.scheduler_telemetry,
                "mutation_bandit_telemetry": stats.mutation_bandit_telemetry,
                "contextual_mutation_bandit_telemetry": stats.contextual_mutation_bandit_telemetry,
                "donor_telemetry": stats.donor_telemetry,
                "splice_telemetry": stats.splice_telemetry,
            })

    total_suite_elapsed = time.perf_counter() - t_suite_start

    # Save results.csv
    csv_file = output_dir / "results.csv"
    if all_trial_rows:
        fieldnames = list(all_trial_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trial_rows)
        print(f"\n[+] Raw trial results written to: {csv_file}")

    # Save discoveries.json
    discoveries_file = output_dir / "discoveries.json"
    with open(discoveries_file, "w", encoding="utf-8") as f:
        json.dump(all_discoveries, f, indent=2)
    print(f"[+] Deep-state discoveries metadata written to: {discoveries_file}")

    # Compute Summary Aggregates
    summary_data: Dict[str, Any] = {}
    for config_key in configs_to_run:
        c_rows = [r for r in all_trial_rows if r["config"] == config_key]
        if not c_rows:
            continue

        cov_vals = [r["total_coverage"] for r in c_rows]
        depth_vals = [r["max_depth"] for r in c_rows]
        t14_vals = [r["t14"] for r in c_rows if r["t14"] is not None]
        t15_vals = [r["t15"] for r in c_rows if r["t15"] is not None]
        crashes_vals = [r["unique_crashes"] for r in c_rows]
        corpus_vals = [r["corpus_size"] for r in c_rows]
        rate_vals = [r["exec_per_sec"] for r in c_rows]
        cov_auc_vals = [r["coverage_auc_normalized"] for r in c_rows]
        depth_auc_vals = [r["depth_auc_normalized"] for r in c_rows]

        summary_data[config_key] = {
            "label": CONFIGURATIONS[config_key]["label"],
            "trials_count": len(c_rows),
            "coverage": compute_mean_std(cov_vals),
            "max_depth": compute_mean_std(depth_vals),
            "t14_reached_count": len(t14_vals),
            "t14_mean": sum(t14_vals) / len(t14_vals) if t14_vals else None,
            "t14_values": t14_vals,
            "t15_reached_count": len(t15_vals),
            "t15_mean": sum(t15_vals) / len(t15_vals) if t15_vals else None,
            "t15_values": t15_vals,
            "coverage_auc_normalized": compute_mean_std(cov_auc_vals),
            "depth_auc_normalized": compute_mean_std(depth_auc_vals),
            "unique_crashes": compute_mean_std(crashes_vals),
            "corpus_size": compute_mean_std(corpus_vals),
            "throughput_exec_s": compute_mean_std(rate_vals),
            "operator_distribution": all_operator_distributions.get(config_key, {}),
            "learning_telemetry": config_telemetries.get(config_key, []),
        }

    # Save summary.json
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "target": str(TARGET_PATH),
                    "executor": args.executor,
                    "budget_per_trial": iterations,
                    "seeds": seeds,
                    "total_trials": len(all_trial_rows),
                    "total_wall_clock_s": total_suite_elapsed,
                },
                "summary": summary_data,
                "discoveries_count": len(all_discoveries),
            },
            f,
            indent=2,
        )
    print(f"[+] Statistical summary written to: {summary_file}")

    # Generate charts
    generate_plots(output_dir, summary_data, trajectory_data)

    # Print Final Comparison Table
    print("\n" + "=" * 105)
    print(" PHASE 7B CONFIGURATION COMPARISON TABLE")
    print("=" * 105)
    hdr = f"{'Configuration':<34} | {'Coverage':<14} | {'Max Depth':<11} | {'T14 (mean)':<11} | {'T15 (reach)':<13} | {'Cov AUC':<10} | {'Depth AUC':<10} | {'Throughput':<12}"
    print(hdr)
    print("-" * len(hdr))

    for config_key in configs_to_run:
        if config_key not in summary_data:
            continue
        sd = summary_data[config_key]
        lbl = sd["label"][:34]
        cov_str = f"{sd['coverage']['mean']:.1f} ± {sd['coverage']['std']:.1f}"
        depth_str = f"{sd['max_depth']['mean']:.1f} ± {sd['max_depth']['std']:.1f}"
        t14_str = f"{sd['t14_mean']:.0f}" if sd["t14_mean"] is not None else "N/A"
        t15_str = f"{sd['t15_reached_count']}/{sd['trials_count']}"
        if sd["t15_mean"] is not None:
            t15_str += f" ({sd['t15_mean']:.0f})"
        else:
            t15_str += " (NR)"
        cov_auc_str = f"{sd['coverage_auc_normalized']['mean']:.1f}"
        depth_auc_str = f"{sd['depth_auc_normalized']['mean']:.1f}"
        tp_str = f"{sd['throughput_exec_s']['mean']:.1f} exec/s"

        print(f"{lbl:<34} | {cov_str:<14} | {depth_str:<11} | {t14_str:<11} | {t15_str:<13} | {cov_auc_str:<10} | {depth_auc_str:<10} | {tp_str:<12}")

    print("=" * 105)
    print(f"Total Experiment Time: {total_suite_elapsed:.2f}s ({total_suite_elapsed / 60.0:.2f} minutes)")
    print("=" * 105)


if __name__ == "__main__":
    main()
