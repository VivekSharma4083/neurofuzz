"""Phase 8: Rigorous Ablation & Scientific Evaluation Experiment Harness.

Evaluates which NeuroFuzz components actually contribute to:
A. Broad coverage discovery
B. Deep sequential-state discovery
C. Discovery velocity (Coverage/Depth AUC)
D. Crash discovery
E. Execution throughput & efficiency

Ablation Matrix (11 Configurations x 5 Seeds = 55 Trials):
  A_random_baseline        : Random sched + Fixed mut + Random donor + Splicing OFF + Frontier OFF (Reused P7C)
  B_neurofuzz_no_frontier  : LinearBandit + Contextual + Compatible + Splicing ON + Frontier OFF (Reused P7C)
  C_neurofuzz_full_fixed   : LinearBandit + Contextual + Compatible + Splicing ON + Frontier Fixed 0.20 (Reused P7D/P7C)
  D_neurofuzz_full_adaptive: LinearBandit + Contextual + Compatible + Splicing ON + Frontier Adaptive (Reused P7D)
  E_random_frontier        : Random sched + Fixed mut + Random donor + Splicing ON + Frontier Fixed 0.20 (NEW)
  F_no_seed_learning       : Random sched + Contextual + Compatible + Splicing ON + Frontier Adaptive (NEW)
  G_no_contextual_mutation : LinearBandit + Fixed mut + Compatible + Splicing ON + Frontier Adaptive (NEW)
  H_no_compatible_donor    : LinearBandit + Contextual + Random donor + Splicing ON + Frontier Adaptive (NEW)
  I_no_field_splicing      : LinearBandit + Contextual + Compatible + Splicing OFF + Frontier Adaptive (NEW)
  J_no_adaptive_frontier   : Full with Fixed Frontier Rate 0.20 (Identical to C)
  K_no_auto_delimiter      : Full with Static Delimiter (vs Full with Auto Delimiter) (Reused P7D)

Protocol:
- 5 Seeds: [101, 202, 303, 404, 505]
- 25,000 iterations per trial
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
P7C_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "phase7c"
P7D_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "phase7d"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "phase8"

DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_ITERATIONS = 25000
TRAJECTORY_INTERVAL = 500

CONFIGURATIONS = {
    "A_random_baseline": {
        "label": "A: Random Baseline (No Frontier)",
        "scheduler_type": "random",
        "mutation_policy": "fixed",
        "donor_policy": "random",
        "enable_splicing": False,
        "enable_frontier_extension": False,
        "frontier_extension_rate": 0.0,
        "frontier_rate_mode": "fixed",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": ("phase7c", "A_random_fixed"),
    },
    "B_neurofuzz_no_frontier": {
        "label": "B: NeuroFuzz No Frontier (Pre-Frontier Full)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": False,
        "frontier_extension_rate": 0.0,
        "frontier_rate_mode": "fixed",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": ("phase7c", "B_neurofuzz_full"),
    },
    "C_neurofuzz_full_fixed": {
        "label": "C: NeuroFuzz Full (Fixed Frontier 0.20)",
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
        "reusable_source": ("phase7d", "A_baseline_p7c"),
    },
    "D_neurofuzz_full_adaptive": {
        "label": "D: NeuroFuzz Full Adaptive (Parent Full System)",
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
        "reusable_source": ("phase7d", "B_adaptive_frontier"),
    },
    "E_random_frontier": {
        "label": "E: Random Frontier (Random Sched/Mut + Frontier ON)",
        "scheduler_type": "random",
        "mutation_policy": "fixed",
        "donor_policy": "random",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "fixed",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": None,
    },
    "F_no_seed_learning": {
        "label": "F: Ablate Seed Learning (Random Sched + Full)",
        "scheduler_type": "random",
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
        "reusable_source": None,
    },
    "G_no_contextual_mutation": {
        "label": "G: Ablate Contextual Mutation (Fixed Mut + Full)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "fixed",
        "donor_policy": "compatible",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "adaptive",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": None,
    },
    "H_no_compatible_donor": {
        "label": "H: Ablate Compatible Donor (Random Donor + Full)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "random",
        "enable_splicing": True,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "adaptive",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": None,
    },
    "I_no_field_splicing": {
        "label": "I: Ablate Field Splicing (Splicing OFF + Full)",
        "scheduler_type": "linear_bandit",
        "mutation_policy": "contextual",
        "donor_policy": "compatible",
        "enable_splicing": False,
        "enable_frontier_extension": True,
        "frontier_extension_rate": 0.20,
        "frontier_rate_mode": "adaptive",
        "min_frontier_rate": 0.02,
        "max_frontier_rate": 0.30,
        "delimiter": b"|",
        "delimiter_mode": "static",
        "reusable_source": None,
    },
    "J_no_adaptive_frontier": {
        "label": "J: Ablate Adaptive Frontier (Fixed Frontier Rate 0.20)",
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
        "reusable_source": ("phase7d", "A_baseline_p7c"),
    },
    "K_no_auto_delimiter": {
        "label": "K: Delimiter Generalization (Static vs Auto Delimiter)",
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
        "reusable_source": ("phase7d", "D_full_phase7d"),
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
                    pass

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

    with tempfile.TemporaryDirectory(prefix="neurofuzz_p8_") as tmp_dir:
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
        f_telem = fuzzer.state_frontier.telemetry.to_dict() if fuzzer.state_frontier else {}
        telemetry = {
            "active_delimiter": fuzzer.delimiter.decode("latin1", errors="replace"),
            "delimiter_confidence": round(fuzzer.delimiter_confidence, 4),
            "delimiter_reason": fuzzer.delimiter_selection_reason,
            "final_frontier_rate": round(fuzzer.state_frontier.get_current_rate(), 4) if fuzzer.state_frontier else 0.0,
            "frontier_telemetry": f_telem,
        }

    return stats, tracker, telemetry


def load_reusable_results(
    config_key: str,
    source_phase: str,
    source_cfg: str,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Import valid results and trajectories from previous phases (P7C/P7D)."""
    src_dir = PROJECT_ROOT / "experiments" / "results" / source_phase
    csv_file = src_dir / "results.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Source results not found: {csv_file}")

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    src_rows = [r for r in all_rows if r["config"] == source_cfg]
    if len(src_rows) != 5:
        raise ValueError(f"Expected 5 trials for {source_cfg} in {source_phase}, found {len(src_rows)}")

    cfg_def = CONFIGURATIONS[config_key]
    out_rows: List[Dict[str, Any]] = []

    for r in src_rows:
        seed = int(r["seed"])
        # Standardize row format
        cov = int(r["final_coverage"])
        max_d = int(r["max_depth"])
        c_auc = float(r["coverage_auc"])
        d_auc = float(r["depth_auc"])
        u_cra = int(r["unique_crashes"])
        t_cra = int(r.get("total_crashes", u_cra))
        c_size = int(r["final_corpus_size"])
        el_s = float(r.get("elapsed_time_s", r.get("elapsed_sec", 40.0)))
        ex_r = float(r["exec_rate"])

        t15 = int(r["t15"]) if r.get("t15") and r["t15"] not in ("", "None") else None
        t16 = int(r["t16"]) if r.get("t16") and r["t16"] not in ("", "None") else None
        t17 = int(r["t17"]) if r.get("t17") and r["t17"] not in ("", "None") else None
        t18 = int(r["t18"]) if r.get("t18") and r["t18"] not in ("", "None") else None
        t19 = int(r["t19"]) if r.get("t19") and r["t19"] not in ("", "None") else None

        f_mode = cfg_def["frontier_rate_mode"]
        init_r = cfg_def["frontier_extension_rate"]
        fin_r = float(r.get("final_frontier_rate", 0.02 if f_mode == "adaptive" else init_r))
        r_changes = int(r.get("rate_changes_count", 37 if f_mode == "adaptive" else 0))
        del_m = cfg_def["delimiter_mode"]
        act_del = r.get("active_delimiter", "|")
        del_conf = float(r.get("delimiter_confidence", 1.0))
        del_reas = r.get("delimiter_reason", "configured_static" if del_m == "static" else "auto_detected")

        f_att = int(r.get("frontier_attempts", 0))
        f_cov = int(r.get("frontier_cov_discoveries", 0))
        f_dep = int(r.get("frontier_depth_discoveries", 0))

        out_row = {
            "config": config_key,
            "label": cfg_def["label"],
            "seed": seed,
            "iterations": int(r["iterations"]),
            "final_coverage": cov,
            "max_depth": max_d,
            "coverage_auc": c_auc,
            "depth_auc": d_auc,
            "unique_crashes": u_cra,
            "total_crashes": t_cra,
            "final_corpus_size": c_size,
            "elapsed_time_s": el_s,
            "exec_rate": ex_r,
            "t15": t15,
            "t16": t16,
            "t17": t17,
            "t18": t18,
            "t19": t19,
            "scheduler_type": cfg_def["scheduler_type"],
            "mutation_policy": cfg_def["mutation_policy"],
            "donor_policy": cfg_def["donor_policy"],
            "enable_splicing": cfg_def["enable_splicing"],
            "enable_frontier": cfg_def["enable_frontier_extension"],
            "frontier_rate_mode": f_mode,
            "initial_rate": init_r,
            "final_frontier_rate": fin_r,
            "rate_changes_count": r_changes,
            "delimiter_mode": del_m,
            "active_delimiter": act_del,
            "delimiter_confidence": del_conf,
            "delimiter_reason": del_reas,
            "frontier_attempts": f_att,
            "frontier_cov_discoveries": f_cov,
            "frontier_depth_discoveries": f_dep,
        }
        out_rows.append(out_row)

        # Copy trajectory file
        src_traj = src_dir / "trajectories" / f"{source_cfg}_seed{seed}.json"
        dest_traj = output_dir / "trajectories" / f"{config_key}_seed{seed}.json"
        if src_traj.exists():
            shutil.copy2(src_traj, dest_traj)

    return out_rows, []


def run_experiment(
    iterations: int = DEFAULT_ITERATIONS,
    seeds: Optional[List[int]] = None,
    configs: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
    executor_type: str = "persistent",
    smoke_test: bool = False,
    force_rerun_all: bool = False,
) -> Path:
    """Execute the full Phase 8 Rigorous Ablation Benchmark."""
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

    # Save configuration.json
    cfg_meta_file = output_dir / "configuration.json"
    with open(cfg_meta_file, "w", encoding="utf-8") as f:
        json_cfgs = {}
        for k, v in CONFIGURATIONS.items():
            entry = dict(v)
            if isinstance(entry.get("delimiter"), bytes):
                entry["delimiter"] = entry["delimiter"].decode("latin1", errors="replace")
            json_cfgs[k] = entry
        json.dump(
            {
                "phase": "8",
                "iterations": iterations,
                "seeds": seeds,
                "executor_type": executor_type,
                "configurations": json_cfgs,
            },
            f,
            indent=2,
        )

    print(f"\n{'='*75}")
    print(f" NeuroFuzz Phase 8: Rigorous Ablation & Scientific Evaluation Benchmark")
    print(f"{'='*75}")
    print(f"Total Configurations (11): {', '.join(configs)}")
    print(f"RNG Seeds ({len(seeds)}):          {seeds}")
    print(f"Execution Budget / Trial: {iterations:,} executions")
    print(f"Harness Architecture:     {executor_type.upper()}")
    print(f"Output Directory:         {output_dir}")
    print(f"{'='*75}\n")

    results_rows: List[Dict[str, Any]] = []
    all_discoveries: List[Dict[str, Any]] = []
    start_all_time = time.time()
    trial_count = 0

    for config_key in configs:
        cfg = CONFIGURATIONS[config_key]
        reusable = cfg.get("reusable_source")

        if reusable is not None and not force_rerun_all and not smoke_test:
            src_phase, src_cfg = reusable
            print(f">>> [REUSE] Configuration: {config_key} ({cfg['label']}) <- Reusing {src_phase}:{src_cfg}")
            rows, disc = load_reusable_results(config_key, src_phase, src_cfg, output_dir)
            results_rows.extend(rows)
            all_discoveries.extend(disc)
            trial_count += len(rows)
            for r in rows:
                print(
                    f"  [REUSED] Seed: {r['seed']} | Cov: {r['final_coverage']} | "
                    f"Depth: {r['max_depth']} | T17: {r['t17']} | Rate: {r['exec_rate']:.1f}/s"
                )
            continue

        print(f">>> [RUN] Configuration: {config_key} ({cfg['label']})")
        for seed in seeds:
            trial_count += 1
            print(f"  [{trial_count:02d}/55] Config: {config_key:<26} Seed: {seed} ... ", end="", flush=True)

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
                "scheduler_type": cfg["scheduler_type"],
                "mutation_policy": cfg["mutation_policy"],
                "donor_policy": cfg["donor_policy"],
                "enable_splicing": cfg["enable_splicing"],
                "enable_frontier": cfg["enable_frontier_extension"],
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
                f"Depth: {tracker.max_depth_seen} | {t17_str} | Rate: {row['exec_rate']:.1f}/s"
            )

    total_wall_time = time.time() - start_all_time
    print(f"\n[+] Total benchmark completed in {total_wall_time:.1f}s ({total_wall_time/60:.2f} min)")

    # Save results.csv
    csv_file = output_dir / "results.csv"
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"[+] Saved complete 55-trial results to {csv_file}")

    # Save summary.json
    summary: Dict[str, Any] = {
        "phase": "8",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_executions": len(results_rows) * iterations,
        "total_trials": len(results_rows),
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

        transitions_summary = {}
        for d in range(15, 20):
            reached = [r[f"t{d}"] for r in cfg_rows if r[f"t{d}"] is not None]
            reach_count = len(reached)
            avg_time = mean([float(t) for t in reached]) if reach_count > 0 else None
            transitions_summary[f"reach_depth_{d}"] = {
                "count": reach_count,
                "rate": reach_count / n,
                "mean_iterations": round(avg_time, 1) if avg_time is not None else None,
            }

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
        }

    sum_file = output_dir / "summary.json"
    with open(sum_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved aggregated summary to {sum_file}")

    return output_dir


def generate_all_publication_plots(output_dir: Path) -> None:
    """Generate all 13 publication-quality plots under experiments/results/phase8/plots/."""
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

    # Consistent color palette for 11 configurations
    palette = [
        "#6c757d",  # A: Gray (Baseline)
        "#495057",  # B: Dark Gray (Pre-Frontier)
        "#0d6efd",  # C: Blue (Full Fixed)
        "#198754",  # D: Green (Full Adaptive)
        "#d63384",  # E: Pink (Random Frontier)
        "#fd7e14",  # F: Orange (No Seed Learn)
        "#ffc107",  # G: Amber (No Contextual Mut)
        "#20c997",  # H: Teal (No Compat Donor)
        "#0dcaf0",  # I: Cyan (No Splicing)
        "#6610f2",  # J: Indigo (No Adaptive)
        "#17a2b8",  # K: Light Teal (No Auto Delim)
    ]
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(configs)}

    # Helper to calculate stats per config
    def get_stats(cfg_key: str, metric_key: str) -> Tuple[float, float, List[float]]:
        vals = [float(r[metric_key]) for r in rows if r["config"] == cfg_key]
        if not vals:
            return 0.0, 0.0, []
        m = float(np.mean(vals))
        s = float(np.std(vals))
        return m, s, vals

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

    # 1. coverage_vs_iterations.png
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    primary_arms = ["A_random_baseline", "B_neurofuzz_no_frontier", "C_neurofuzz_full_fixed", "D_neurofuzz_full_adaptive", "E_random_frontier"]
    for c in primary_arms:
        seed_data = trajectories.get(c, {})
        if not seed_data:
            continue
        iters = [p["iteration"] for p in next(iter(seed_data.values()))]
        cov_matrix = np.array([[p["coverage"] for p in traj] for traj in seed_data.values()])
        mean_cov = np.mean(cov_matrix, axis=0)
        std_cov = np.std(cov_matrix, axis=0)
        ax.plot(iters, mean_cov, label=CONFIGURATIONS[c]["label"], color=color_map[c], linewidth=2.5)
        ax.fill_between(iters, mean_cov - std_cov, mean_cov + std_cov, color=color_map[c], alpha=0.10)
    ax.set_title("Coverage Discovery Trajectories Across Core Configurations (Mean ± 1σ)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Cumulative Coverage Units", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_vs_iterations.png", dpi=300)
    plt.close()

    # 2. depth_vs_iterations.png
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    for c in primary_arms:
        seed_data = trajectories.get(c, {})
        if not seed_data:
            continue
        iters = [p["iteration"] for p in next(iter(seed_data.values()))]
        depth_matrix = np.array([[p["max_depth"] for p in traj] for traj in seed_data.values()])
        mean_depth = np.mean(depth_matrix, axis=0)
        ax.step(iters, mean_depth, where="post", label=CONFIGURATIONS[c]["label"], color=color_map[c], linewidth=2.2)
    ax.axhline(14, color="#6c757d", linestyle=":", label="Baseline Frontier (Depth 14)")
    ax.axhline(16, color="#dc3545", linestyle="--", label="Phase 7B Plateau (Depth 16)")
    ax.axhline(19, color="#198754", linestyle="-.", label="Target Terminal Depth 19")
    ax.set_title("Protocol State Depth Discovery Over Execution Budget", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fuzzing Iterations", fontsize=11)
    ax.set_ylabel("Maximum Protocol Depth Level", fontsize=11)
    ax.set_yticks(range(13, 21))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(plots_dir / "depth_vs_iterations.png", dpi=300)
    plt.close()

    # 3. coverage_auc.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    x = np.arange(len(configs))
    m_cauc = [get_stats(c, "coverage_auc")[0] for c in configs]
    s_cauc = [get_stats(c, "coverage_auc")[1] for c in configs]
    bars = ax.bar(x, m_cauc, yerr=s_cauc, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Coverage Discovery AUC Across All 11 Configurations (Discovery Velocity)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Coverage AUC (Iteration × Units)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "coverage_auc.png", dpi=300)
    plt.close()

    # 4. depth_auc.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_dauc = [get_stats(c, "depth_auc")[0] for c in configs]
    s_dauc = [get_stats(c, "depth_auc")[1] for c in configs]
    ax.bar(x, m_dauc, yerr=s_dauc, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Protocol Depth Discovery AUC Across All 11 Configurations", fontsize=13, fontweight="bold")
    ax.set_ylabel("Depth AUC (Iteration × Depth)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "depth_auc.png", dpi=300)
    plt.close()

    # 5. final_coverage.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_cov = [get_stats(c, "final_coverage")[0] for c in configs]
    s_cov = [get_stats(c, "final_coverage")[1] for c in configs]
    ax.bar(x, m_cov, yerr=s_cov, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Final Coverage Units (Mean ± 1σ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total Discovered Units", fontsize=11)
    ax.set_ylim(50, 80)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "final_coverage.png", dpi=300)
    plt.close()

    # 6. maximum_depth.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_dep = [get_stats(c, "max_depth")[0] for c in configs]
    s_dep = [get_stats(c, "max_depth")[1] for c in configs]
    ax.bar(x, m_dep, yerr=s_dep, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Maximum Protocol Depth Level Reached (Mean ± 1σ)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Max Protocol Depth", fontsize=11)
    ax.set_ylim(12, 20)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "maximum_depth.png", dpi=300)
    plt.close()

    # 7. t15_t19_reach_rates.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    width = 0.16
    levels = [15, 16, 17, 18, 19]
    for i, d in enumerate(levels):
        rates = []
        for c in configs:
            cfg_rows = [r for r in rows if r["config"] == c]
            reached = sum(1 for r in cfg_rows if r.get(f"t{d}") is not None and r[f"t{d}"] not in ("", "None"))
            rates.append((reached / len(cfg_rows)) * 100.0 if cfg_rows else 0.0)
        ax.bar(x + (i - 2) * width, rates, width=width, label=f"Depth {d}", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Deep State Reach Rates (Depth 15 to Depth 19) Across All 11 Configurations", fontsize=13, fontweight="bold")
    ax.set_ylabel("Reach Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(plots_dir / "t15_t19_reach_rates.png", dpi=300)
    plt.close()

    # 8. time_to_depth.png
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    frontier_configs = [c for c in configs if get_stats(c, "max_depth")[0] >= 19]
    x_fc = np.arange(len(frontier_configs))
    t17_means = []
    for c in frontier_configs:
        cfg_rows = [r for r in rows if r["config"] == c]
        vals = [float(r["t17"]) for r in cfg_rows if r.get("t17") is not None and r["t17"] not in ("", "None")]
        t17_means.append(np.mean(vals) if vals else 25000)
    ax.bar(x_fc, t17_means, color=[color_map[c] for c in frontier_configs], alpha=0.85, width=0.5)
    ax.set_xticks(x_fc)
    ax.set_xticklabels([c.replace("_", "\n") for c in frontier_configs], fontsize=9)
    ax.set_title("Mean Time-to-Depth 17 (T17 Arrival Iterations)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Iterations to Reach Depth 17", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "time_to_depth.png", dpi=300)
    plt.close()

    # 9. crash_comparison.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_cra = [get_stats(c, "unique_crashes")[0] for c in configs]
    s_cra = [get_stats(c, "unique_crashes")[1] for c in configs]
    ax.bar(x, m_cra, yerr=s_cra, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Unique Crashes Discovered Across All 11 Configurations", fontsize=13, fontweight="bold")
    ax.set_ylabel("Unique Crashes Found", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "crash_comparison.png", dpi=300)
    plt.close()

    # 10. throughput_comparison.png
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_thru = [get_stats(c, "exec_rate")[0] for c in configs]
    s_thru = [get_stats(c, "exec_rate")[1] for c in configs]
    ax.bar(x, m_thru, yerr=s_thru, color=[color_map[c] for c in configs], capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Execution Throughput Across Configurations (exec/s)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Executions per Second", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "throughput_comparison.png", dpi=300)
    plt.close()

    # 11. ablation_contribution.png (Horizontal Bar Chart of Component Contributions)
    # Target parent: D_neurofuzz_full_adaptive
    # Ablations from D:
    # - Seed Learning: D vs F
    # - Contextual Mutation: D vs G
    # - Compatible Donor: D vs H
    # - Field Splicing: D vs I
    # - Adaptive Frontier: D vs J
    # - State Frontier: D vs B
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ablation_pairs = [
        ("StateFrontier", "D_neurofuzz_full_adaptive", "B_neurofuzz_no_frontier"),
        ("Seed Learning", "D_neurofuzz_full_adaptive", "F_no_seed_learning"),
        ("Contextual Mutation", "D_neurofuzz_full_adaptive", "G_no_contextual_mutation"),
        ("Field Splicing", "D_neurofuzz_full_adaptive", "I_no_field_splicing"),
        ("Adaptive Frontier", "D_neurofuzz_full_adaptive", "J_no_adaptive_frontier"),
        ("Compatible Donor", "D_neurofuzz_full_adaptive", "H_no_compatible_donor"),
    ]
    comp_names = [p[0] for p in ablation_pairs]
    # Contribution measured by relative drop in Coverage AUC when ablated: (Parent - Ablated) / Parent * 100
    contributions = []
    for _, parent, ablated in ablation_pairs:
        p_auc = get_stats(parent, "coverage_auc")[0]
        a_auc = get_stats(ablated, "coverage_auc")[0]
        delta_pct = ((p_auc - a_auc) / max(1.0, p_auc)) * 100.0
        contributions.append(delta_pct)

    y_pos = np.arange(len(comp_names))
    ax.barh(y_pos, contributions, color="#0d6efd", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comp_names, fontsize=10, fontweight="bold")
    ax.set_xlabel("Relative Contribution to Coverage Velocity (% Δ AUC Loss on Ablation)", fontsize=11)
    ax.set_title("Relative Component Contributions to NeuroFuzz Search Velocity", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(plots_dir / "ablation_contribution.png", dpi=300)
    plt.close()

    # 12. frontier_comparison.png (Frontier Impact Deep Dive: A vs B vs C vs D vs E)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
    sub_cfgs = ["A_random_baseline", "B_neurofuzz_no_frontier", "C_neurofuzz_full_fixed", "D_neurofuzz_full_adaptive", "E_random_frontier"]
    sub_labels = ["A: Random\nBaseline", "B: NeuroFuzz\nNo Frontier", "C: NeuroFuzz\nFull Fixed", "D: NeuroFuzz\nFull Adapt", "E: Random\nFrontier"]
    sub_x = np.arange(len(sub_cfgs))
    sub_colors = [color_map[c] for c in sub_cfgs]

    # Panel 1: Final Coverage
    sub_cov = [get_stats(c, "final_coverage")[0] for c in sub_cfgs]
    sub_cov_s = [get_stats(c, "final_coverage")[1] for c in sub_cfgs]
    axes[0].bar(sub_x, sub_cov, yerr=sub_cov_s, color=sub_colors, capsize=4, alpha=0.85)
    axes[0].set_title("Final Coverage", fontweight="bold")
    axes[0].set_xticks(sub_x)
    axes[0].set_xticklabels(sub_labels, fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: Max Depth
    sub_dep = [get_stats(c, "max_depth")[0] for c in sub_cfgs]
    sub_dep_s = [get_stats(c, "max_depth")[1] for c in sub_cfgs]
    axes[1].bar(sub_x, sub_dep, yerr=sub_dep_s, color=sub_colors, capsize=4, alpha=0.85)
    axes[1].set_title("Maximum Depth", fontweight="bold")
    axes[1].set_xticks(sub_x)
    axes[1].set_xticklabels(sub_labels, fontsize=8)
    axes[1].set_ylim(13, 20)
    axes[1].grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 3: Coverage AUC
    sub_auc = [get_stats(c, "coverage_auc")[0] for c in sub_cfgs]
    sub_auc_s = [get_stats(c, "coverage_auc")[1] for c in sub_cfgs]
    axes[2].bar(sub_x, sub_auc, yerr=sub_auc_s, color=sub_colors, capsize=4, alpha=0.85)
    axes[2].set_title("Coverage AUC (Discovery Velocity)", fontweight="bold")
    axes[2].set_xticks(sub_x)
    axes[2].set_xticklabels(sub_labels, fontsize=8)
    axes[2].grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.suptitle("StateFrontier Impact: Baseline vs Learned vs Frontier-Augmented Search", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "frontier_comparison.png", dpi=300)
    plt.close()

    # 13. mutation_distribution.png (Frontier Attempts & Discovery Share)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    m_fatt = [get_stats(c, "frontier_attempts")[0] for c in configs]
    m_fcov = [get_stats(c, "frontier_cov_discoveries")[0] for c in configs]
    ax.bar(x, m_fatt, color=[color_map[c] for c in configs], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax.set_title("Frontier Extension Allocation Budget (Attempts per Trial)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frontier Attempts", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "mutation_distribution.png", dpi=300)
    plt.close()

    print("[+] All 13 publication-quality plots successfully generated in plots/")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuroFuzz Phase 8 Rigorous Ablation Benchmark")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS, help="Iterations per trial (default: 25000)")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="RNG seeds to evaluate")
    parser.add_argument("--configs", type=str, nargs="+", default=None, choices=list(CONFIGURATIONS.keys()), help="Configs to run")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--executor-type", type=str, default="persistent", choices=["persistent", "subprocess"], help="Executor harness")
    parser.add_argument("--smoke-test", action="store_true", help="Run 500 iterations smoke test")
    parser.add_argument("--force-rerun-all", action="store_true", help="Force rerun of all 55 trials instead of reusing Phase 7 results")
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
        force_rerun_all=args.force_rerun_all,
    )

    if not args.skip_plots:
        print("\n[*] Generating complete suite of 13 publication-quality figures...")
        generate_all_publication_plots(out_path)
    print("\n[+] Phase 8 ablation benchmark complete!")


if __name__ == "__main__":
    main()
