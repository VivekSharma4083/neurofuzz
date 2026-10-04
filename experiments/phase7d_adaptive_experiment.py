"""Phase 7D: Adaptive Frontier + Delimiter Generalization Experiment.

Evaluates:
1. Adaptive Frontier Rate Decay: dynamic allocation responding to stagnation
   and discoveries vs fixed rate allocation.
2. Automated Delimiter Discovery: multi-signal structural discovery vs static
   configured delimiter.
3. 2x2 Factorial Ablation across 4 arms:
   - Arm A: Phase 7C Baseline (Fixed Rate 0.20, Static Delimiter '|')
   - Arm B: Adaptive Frontier (Adaptive Rate [0.02, 0.30], Static Delimiter '|')
   - Arm C: Automated Delimiter (Fixed Rate 0.20, Auto-detected Delimiter)
   - Arm D: Full Phase 7D (Adaptive Rate + Auto-detected Delimiter)

Protocol:
- 5 RNG Seeds: [101, 202, 303, 404, 505]
- 25,000 iterations per trial (20 trials total = 500,000 executions)
- Harness: Persistent executor
- Target: targets/structured_target.exe
- Corpus: corpus_structured (24 protocol seeds)
- Dictionary: dictionaries/structured_protocol.dict (85 tokens)
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "phase7d"

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_ITERATIONS = 25000
TRAJECTORY_INTERVAL = 500

CONFIGURATIONS = {
    "A_baseline_p7c": {
        "label": "A: Phase 7C Baseline (Fixed 0.20, Static Delim)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "fixed",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
    },
    "B_adaptive_frontier": {
        "label": "B: Adaptive Frontier (Decay [0.02, 0.30], Static Delim)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "adaptive",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
    },
    "C_auto_delimiter": {
        "label": "C: Auto Delimiter (Fixed 0.20, Auto Delim)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "fixed",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "auto",
    },
    "D_full_phase7d": {
        "label": "D: Full Phase 7D (Adaptive Frontier + Auto Delim)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "adaptive",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "auto",
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
    """Tracks coverage, depth trajectory, rate dynamics, and discovery artifacts."""

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
                "frontier_rate": 0.20,
            }
        ]
        self.discoveries: List[Dict[str, Any]] = []
        self.current_frontier_rate: float = 0.20

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

        # Check if a new maximum depth frontier was breached
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

            # Save discovery artifacts
            if self.discoveries_dir is not None:
                try:
                    cfg_disc_dir = self.discoveries_dir / f"{self.config_key}_seed{self.seed}"
                    cfg_disc_dir.mkdir(parents=True, exist_ok=True)
                    (cfg_disc_dir / f"depth{child_depth}.txt").write_bytes(mutated_bytes)
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
                    "frontier_rate": round(self.current_frontier_rate, 4),
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
) -> Tuple[FuzzStats, TrialTracker, Dict[str, Any]]:
    """Execute a single fuzzing trial with full instrumentation and telemetry."""
    discoveries_dir = output_dir / "discoveries"
    tracker = TrialTracker(
        config_key=config_key,
        seed=seed,
        trajectory_interval=TRAJECTORY_INTERVAL,
        discoveries_dir=discoveries_dir,
    )

    with tempfile.TemporaryDirectory(prefix="neurofuzz_p7d_") as tmp_dir:
        tmp_corpus = Path(tmp_dir) / "corpus"
        tmp_corpus.mkdir(parents=True, exist_ok=True)
        for seed_file in INITIAL_CORPUS_DIR.glob("*.txt"):
            shutil.copy2(seed_file, tmp_corpus / seed_file.name)

        tmp_crashes = Path(tmp_dir) / "crashes"
        tmp_crashes.mkdir(parents=True, exist_ok=True)

        # Intercept tracker to sample live rate from fuzzer
        def iteration_wrapper(it: int, st: FuzzStats, inf: Dict[str, Any]) -> None:
            if hasattr(fuzzer, "state_frontier") and fuzzer.state_frontier is not None:
                tracker.current_frontier_rate = fuzzer.state_frontier.get_current_rate()
            tracker.on_iteration(it, st, inf)

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
            frontier_rate_mode=config_params["frontier_rate_mode"],
            min_frontier_rate=config_params["min_frontier_rate"],
            max_frontier_rate=config_params["max_frontier_rate"],
            delimiter=config_params["delimiter"],
            delimiter_mode=config_params["delimiter_mode"],
            crashes_dir=str(tmp_crashes),
            executor_type=executor_type,
            iteration_callback=iteration_wrapper,
        )

        if silent:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                stats = fuzzer.run()
        else:
            stats = fuzzer.run()

        # Capture final telemetry from fuzzer
        telemetry = {
            "active_delimiter": fuzzer.delimiter.decode("latin1", errors="replace"),
            "delimiter_confidence": round(fuzzer.delimiter_confidence, 4),
            "delimiter_reason": fuzzer.delimiter_selection_reason,
            "final_frontier_rate": round(fuzzer.state_frontier.get_current_rate(), 4),
            "frontier_telemetry": fuzzer.state_frontier.telemetry.to_dict(),
        }

    return stats, tracker, telemetry


def run_experiment(
    iterations: int = DEFAULT_ITERATIONS,
    seeds: Optional[List[int]] = None,
    configs: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    executor_type: str = "persistent",
    smoke_test: bool = False,
) -> Path:
    """Execute the full Phase 7D controlled benchmark."""
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
    print(f" NeuroFuzz Phase 7D: Adaptive Frontier + Delimiter Generalization Benchmark")
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
            print(f"  [{trial_count:02d}/{total_trials:02d}] Config: {config_key:<22} Seed: {seed} ... ", end="", flush=True)

            t0 = time.time()
            stats, tracker, telemetry = run_single_trial(
                config_key=config_key,
                config_params=cfg,
                seed=seed,
                iterations=iterations,
                executor_type=executor_type,
                output_dir=output_dir,
                silent=True,
            )
            wall_time = time.time() - t0

            # Calculate AUCs
            cov_pts = [(p["iteration"], float(p["coverage"])) for p in tracker.trajectory]
            dep_pts = [(p["iteration"], float(p["max_depth"])) for p in tracker.trajectory]
            cov_auc = compute_auc(cov_pts)
            depth_auc = compute_auc(dep_pts)

            f_telem = telemetry["frontier_telemetry"]

            row = {
                "config": config_key,
                "label": cfg["label"],
                "seed": seed,
                "iterations": stats.total_executions,
                "final_coverage": stats.total_coverage_units,
                "max_depth": tracker.max_depth_seen,
                "coverage_auc": round(cov_auc, 1),
                "depth_auc": round(depth_auc, 1),
                "unique_crashes": stats.unique_crashes,
                "total_crashes": stats.crashes_found,
                "final_corpus_size": stats.corpus_size,
                "elapsed_time_s": round(wall_time, 2),
                "exec_rate": round(stats.total_executions / max(0.001, wall_time), 1),
                "t15": tracker.time_to_depth[15],
                "t16": tracker.time_to_depth[16],
                "t17": tracker.time_to_depth[17],
                "t18": tracker.time_to_depth[18],
                "t19": tracker.time_to_depth[19],
                "frontier_rate_mode": cfg["frontier_rate_mode"],
                "initial_rate": cfg["frontier_extension_rate"],
                "final_frontier_rate": telemetry["final_frontier_rate"],
                "rate_changes_count": f_telem.get("rate_changes_count", 0),
                "delimiter_mode": cfg["delimiter_mode"],
                "active_delimiter": telemetry["active_delimiter"],
                "delimiter_confidence": telemetry["delimiter_confidence"],
                "delimiter_reason": telemetry["delimiter_reason"],
                "frontier_attempts": f_telem.get("frontier_extension_attempts", 0),
                "frontier_cov_discoveries": f_telem.get("frontier_discoveries", 0),
                "frontier_depth_discoveries": f_telem.get("frontier_depth_discoveries", 0),
            }
            results_rows.append(row)
            all_discoveries.extend(tracker.discoveries)

            # Save trajectory to disk
            traj_file = output_dir / "trajectories" / f"{config_key}_seed{seed}.json"
            with open(traj_file, "w", encoding="utf-8") as f:
                json.dump(tracker.trajectory, f, indent=2)

            t17_str = f"T17={tracker.time_to_depth[17]}" if tracker.time_to_depth[17] else "T17=None"
            print(
                f"Done ({wall_time:.1f}s) | Cov: {stats.total_coverage_units} | "
                f"Depth: {tracker.max_depth_seen} | {t17_str} | Rate: {row['exec_rate']:.1f}/s | "
                f"Delim: '{telemetry['active_delimiter']}' ({telemetry['delimiter_confidence']*100:.0f}%) | "
                f"RateMode: {cfg['frontier_rate_mode']} ({telemetry['final_frontier_rate']:.2f})"
            )

    total_wall_time = time.time() - start_all_time
    print(f"\n[+] Completed {trial_count} trials in {total_wall_time:.1f}s ({total_wall_time/60:.2f} min)")

    # Save results.csv
    csv_file = output_dir / "results.csv"
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"[+] Saved trial results to {csv_file}")

    # Save discoveries.json
    disc_file = output_dir / "discoveries.json"
    with open(disc_file, "w", encoding="utf-8") as f:
        json.dump(all_discoveries, f, indent=2)
    print(f"[+] Saved {len(all_discoveries)} state discoveries to {disc_file}")

    # Compute summary.json
    summary: Dict[str, Any] = {
        "phase": "7D",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_executions": trial_count * iterations,
        "total_trials": trial_count,
        "iterations_per_trial": iterations,
        "executor_type": executor_type,
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
        f_rates = [float(r["final_frontier_rate"]) for r in cfg_rows]

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

        # Delimiter telemetry summary
        delims = [r["active_delimiter"] for r in cfg_rows]
        confs = [float(r["delimiter_confidence"]) for r in cfg_rows]

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
            "mean_final_frontier_rate": round(mean(f_rates), 4),
            "transitions": transitions_summary,
            "delimiter_summary": {
                "active_delimiters": list(set(delims)),
                "mean_confidence": round(mean(confs), 4),
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
        "A_baseline_p7c": "#6c757d",       # Gray (Baseline)
        "B_adaptive_frontier": "#0d6efd",   # Blue (Adaptive Rate)
        "C_auto_delimiter": "#fd7e14",      # Orange (Auto Delimiter)
        "D_full_phase7d": "#198754",        # Green (Full 7D)
    }
    short_labels = {
        "A_baseline_p7c": "A: P7C Baseline (Fixed 0.20, Static '|')",
        "B_adaptive_frontier": "B: Adaptive Frontier (Decay, Static '|')",
        "C_auto_delimiter": "C: Auto Delim (Fixed 0.20, Auto '|')",
        "D_full_phase7d": "D: Full Phase 7D (Adaptive + Auto '|')",
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
        ax.fill_between(iters, mean_cov - std_cov, mean_cov + std_cov, color=colors[c], alpha=0.12)

    ax.set_title("Phase 7D: Coverage Growth Across Controlled Arms (Mean ± Std)", fontsize=13, fontweight="bold")
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

        ax.step(iters, mean_depth, where="post", label=short_labels[c], color=colors[c], linewidth=2.5)

    ax.axhline(14, color="#adb5bd", linestyle=":", label="Baseline Frontier (Depth 14)")
    ax.axhline(16, color="#dc3545", linestyle="--", label="Phase 7B Saturation (Depth 16)")
    ax.axhline(19, color="#6f42c1", linestyle="-.", label="Target Maximum (Depth 19)")
    ax.set_title("Phase 7D: Protocol Depth Discovery Trajectories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Protocol Depth Level", fontsize=11)
    ax.set_yticks(range(13, 20))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "depth_vs_iterations.png", dpi=300)
    plt.close()
    print("  [+] Generated depth_vs_iterations.png")

    # 3. Frontier Mutation Rate Dynamics Over Time
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    for c in configs:
        seed_data = trajectories[c]
        if not seed_data:
            continue
        iters = [p["iteration"] for p in next(iter(seed_data.values()))]
        rate_matrix = np.array([[p.get("frontier_rate", 0.20) for p in traj] for traj in seed_data.values()])
        mean_rate = np.mean(rate_matrix, axis=0)

        ax.plot(iters, mean_rate, label=short_labels[c], color=colors[c], linewidth=2.2)

    ax.axhline(0.20, color="#6c757d", linestyle=":", label="Nominal Rate (0.20)")
    ax.axhline(0.02, color="#dc3545", linestyle="--", label="Decay Floor (0.02)")
    ax.axhline(0.30, color="#198754", linestyle="-.", label="Replenish Ceiling (0.30)")
    ax.set_title("Phase 7D: Frontier Allocation Rate Dynamics Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Frontier Allocation Rate", fontsize=11)
    ax.set_ylim(-0.01, 0.35)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "frontier_rate_vs_iterations.png", dpi=300)
    plt.close()
    print("  [+] Generated frontier_rate_vs_iterations.png")

    # 4. Multi-metric Comparison Matrix (Final Coverage, Max Depth, Coverage AUC, Throughput)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    x_pos = np.arange(len(configs))
    labels = ["Arm A\nP7C Baseline", "Arm B\nAdaptive Rate", "Arm C\nAuto Delim", "Arm D\nFull 7D"]
    bar_colors = [colors[c] for c in configs]

    # Panel 1: Final Coverage
    cov_means = [np.mean([float(r["final_coverage"]) for r in rows if r["config"] == c]) for c in configs]
    cov_stds = [np.std([float(r["final_coverage"]) for r in rows if r["config"] == c]) for c in configs]
    axes[0, 0].bar(x_pos, cov_means, yerr=cov_stds, color=bar_colors, capsize=5, alpha=0.85)
    axes[0, 0].set_title("Final Coverage (Units)", fontweight="bold")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: Max Depth Reached
    depth_means = [np.mean([float(r["max_depth"]) for r in rows if r["config"] == c]) for c in configs]
    depth_stds = [np.std([float(r["max_depth"]) for r in rows if r["config"] == c]) for c in configs]
    axes[0, 1].bar(x_pos, depth_means, yerr=depth_stds, color=bar_colors, capsize=5, alpha=0.85)
    axes[0, 1].set_title("Max Protocol Depth Reached", fontweight="bold")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylim(14, 20)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 3: Coverage AUC
    cauc_means = [np.mean([float(r["coverage_auc"]) for r in rows if r["config"] == c]) for c in configs]
    cauc_stds = [np.std([float(r["coverage_auc"]) for r in rows if r["config"] == c]) for c in configs]
    axes[1, 0].bar(x_pos, cauc_means, yerr=cauc_stds, color=bar_colors, capsize=5, alpha=0.85)
    axes[1, 0].set_title("Coverage Discovery AUC (Velocity)", fontweight="bold")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 4: Throughput (exec/s)
    rate_means = [np.mean([float(r["exec_rate"]) for r in rows if r["config"] == c]) for c in configs]
    rate_stds = [np.std([float(r["exec_rate"]) for r in rows if r["config"] == c]) for c in configs]
    axes[1, 1].bar(x_pos, rate_means, yerr=rate_stds, color=bar_colors, capsize=5, alpha=0.85)
    axes[1, 1].set_title("Execution Throughput (exec/s)", fontweight="bold")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("Phase 7D: Comparative Performance Matrix Across 4 Controlled Arms", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "phase7d_comprehensive_matrix.png", dpi=300)
    plt.close()
    print("  [+] Generated phase7d_comprehensive_matrix.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroFuzz Phase 7D Controlled Experiment")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Iterations per trial (default: 25000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="RNG seeds to evaluate")
    parser.add_argument("--configs", type=str, nargs="+", default=None, choices=list(CONFIGURATIONS.keys()), help="Configs to run")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--executor-type", type=str, default="persistent", choices=["persistent", "subprocess"], help="Executor harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run 500 iterations smoke test")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    run_experiment(
        iterations=args.iterations,
        seeds=args.seeds,
        configs=args.configs,
        output_dir=out_path,
        executor_type=args.executor_type,
        smoke_test=args.smoke_test,
    )

    if not args.skip_plots:
        print("\n[*] Generating high-resolution publication plots...")
        generate_plots(out_path)
    print("\n[+] Phase 7D benchmark complete!")


if __name__ == "__main__":
    main()
