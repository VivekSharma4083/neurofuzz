"""
Phase 7B — Final Data Analysis & Plot Regeneration Script

Analyzes the completed 35 trials of the Phase 7B high-budget fuzzing experiment.
DOES NOT RERUN ANY FUZZING.
DOES NOT MODIFY ALGORITHMS OR RAW RESULTS.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("experiments/results/phase7b")
TRAJECTORIES_DIR = RESULTS_DIR / "trajectories"
PLOTS_DIR = RESULTS_DIR / "plots"
DISCOVERIES_FILE = RESULTS_DIR / "discoveries.json"
RESULTS_CSV = RESULTS_DIR / "results.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

CONFIG_LABELS = {
    "A_random_fixed": "A: Random + Fixed",
    "B_heuristic_fixed": "B: Heuristic + Fixed",
    "C_linear_bandit_fixed": "C: LinearBandit + Fixed",
    "D_random_ucb1": "D: Random + UCB1",
    "E_random_contextual": "E: Random + Contextual",
    "F_random_contextual_compatible": "F: Random + Contextual + Compatible",
    "G_neurofuzz_full": "G: NeuroFuzz Full (Bandit + Contextual + Compatible)",
}

CONFIG_SHORT_LABELS = {
    "A_random_fixed": "A: Rand+Fix",
    "B_heuristic_fixed": "B: Heur+Fix",
    "C_linear_bandit_fixed": "C: Band+Fix",
    "D_random_ucb1": "D: Rand+UCB1",
    "E_random_contextual": "E: Rand+Ctx",
    "F_random_contextual_compatible": "F: Rand+Ctx+Comp",
    "G_neurofuzz_full": "G: Full NeuroFuzz",
}

CONFIG_COLORS = {
    "A_random_fixed": "#4e79a7",
    "B_heuristic_fixed": "#f28e2c",
    "C_linear_bandit_fixed": "#e15759",
    "D_random_ucb1": "#76b7b2",
    "E_random_contextual": "#59a14f",
    "F_random_contextual_compatible": "#edc948",
    "G_neurofuzz_full": "#b07aa1",
}


def compute_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    n = len(values)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if n > 1 and std_val > 1e-9:
        ci = stats.t.interval(0.95, df=n - 1, loc=mean_val, scale=std_val / math.sqrt(n))
        ci_low, ci_high = float(ci[0]), float(ci[1])
    else:
        ci_low, ci_high = mean_val, mean_val
    return {
        "n": n,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Calculate Cohen's d effect size between two independent groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if s_pooled < 1e-9:
        return 0.0
    return float((m1 - m2) / s_pooled)


def load_data():
    print("[1] Verifying saved data...")
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {RESULTS_CSV}")
    
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"    Loaded {len(rows)} result rows from results.csv")
    configs = sorted(list({r["config"] for r in rows}))
    print(f"    Configurations found ({len(configs)}): {configs}")
    
    seeds_per_config = {}
    for r in rows:
        cfg = r["config"]
        s = int(r["seed"])
        seeds_per_config.setdefault(cfg, []).append(s)
        
    for cfg, s_list in seeds_per_config.items():
        print(f"    {cfg}: {len(s_list)} seeds -> {sorted(s_list)}")
        
    # Verify trajectory files
    trajectories = {}
    for cfg in configs:
        trajectories[cfg] = {}
        for s in [101, 202, 303, 404, 505]:
            traj_file = TRAJECTORIES_DIR / f"{cfg}_seed{s}.csv"
            if not traj_file.exists():
                print(f"[!] Warning: Missing trajectory {traj_file}")
                continue
            traj_pts = []
            with open(traj_file, "r", encoding="utf-8") as f:
                t_reader = csv.DictReader(f)
                for pt in t_reader:
                    traj_pts.append({
                        "iteration": int(pt["iteration"]),
                        "coverage": int(pt["coverage"]),
                        "max_depth": int(pt["max_depth"]),
                        "corpus_size": int(pt["corpus_size"]),
                        "crashes": int(pt["crashes"]),
                    })
            trajectories[cfg][s] = traj_pts
    print(f"    Loaded trajectory data for all {len(trajectories)} configs")
    
    # Load discoveries
    with open(DISCOVERIES_FILE, "r", encoding="utf-8") as f:
        discoveries = json.load(f)
    print(f"    Loaded {len(discoveries)} discoveries from discoveries.json")
    
    # Load summary
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        summary_obj = json.load(f)
    print(f"    Loaded summary.json successfully")
    
    return rows, trajectories, discoveries, summary_obj


def main():
    rows, trajectories, discoveries, summary_obj = load_data()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Organize rows by configuration
    by_config: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        cfg = r["config"]
        by_config.setdefault(cfg, []).append(r)
        
    print("\n[2] Computing exact statistics per configuration...")
    metrics_summary = {}
    for cfg, c_rows in by_config.items():
        cov_vals = [float(r["total_coverage"]) for r in c_rows]
        depth_vals = [float(r["max_depth"]) for r in c_rows]
        cov_auc_vals = [float(r["coverage_auc_normalized"]) for r in c_rows]
        depth_auc_vals = [float(r["depth_auc_normalized"]) for r in c_rows]
        crash_vals = [float(r["unique_crashes"]) for r in c_rows]
        speed_vals = [float(r["exec_per_sec"]) for r in c_rows]
        
        # Depth reach counts
        d15_count = sum(1 for r in c_rows if float(r["max_depth"]) >= 15)
        d16_count = sum(1 for r in c_rows if float(r["max_depth"]) >= 16)
        d17_count = sum(1 for r in c_rows if float(r["max_depth"]) >= 17)
        d18_count = sum(1 for r in c_rows if float(r["max_depth"]) >= 18)
        d19_count = sum(1 for r in c_rows if float(r["max_depth"]) >= 19)
        
        # Time-to-depth values (only for trials that reached)
        t15_vals = [float(r["t15"]) for r in c_rows if r["t15"] != "" and r["t15"] is not None]
        t16_vals = [float(r["t16"]) for r in c_rows if r["t16"] != "" and r["t16"] is not None]
        
        metrics_summary[cfg] = {
            "label": CONFIG_LABELS.get(cfg, cfg),
            "rows": c_rows,
            "coverage": compute_stats(cov_vals),
            "depth": compute_stats(depth_vals),
            "cov_auc": compute_stats(cov_auc_vals),
            "depth_auc": compute_stats(depth_auc_vals),
            "crashes": compute_stats(crash_vals),
            "speed": compute_stats(speed_vals),
            "reach": {
                "d15": d15_count,
                "d16": d16_count,
                "d17": d17_count,
                "d18": d18_count,
                "d19": d19_count,
            },
            "t15_vals": t15_vals,
            "t15_mean": float(np.mean(t15_vals)) if t15_vals else None,
            "t15_median": float(np.median(t15_vals)) if t15_vals else None,
            "t15_std": float(np.std(t15_vals, ddof=1)) if len(t15_vals) > 1 else (0.0 if t15_vals else None),
            "t16_vals": t16_vals,
            "t16_mean": float(np.mean(t16_vals)) if t16_vals else None,
            "t16_median": float(np.median(t16_vals)) if t16_vals else None,
            "t16_std": float(np.std(t16_vals, ddof=1)) if len(t16_vals) > 1 else (0.0 if t16_vals else None),
        }
        
    # Print Executive Summary Table
    print("\n" + "=" * 115)
    print(f"{'Configuration':<35} | {'Coverage':<12} | {'Depth':<9} | {'Cov AUC':<9} | {'Depth AUC':<10} | {'T15 (mean)':<12} | {'Crashes':<12} | {'Exec/s':<10}")
    print("=" * 115)
    for cfg in sorted(CONFIG_LABELS.keys()):
        m = metrics_summary[cfg]
        cov_s = f"{m['coverage']['mean']:.1f}±{m['coverage']['std']:.1f}"
        dep_s = f"{m['depth']['mean']:.1f}±{m['depth']['std']:.1f}"
        cauc_s = f"{m['cov_auc']['mean']:.1f}±{m['cov_auc']['std']:.1f}"
        dauc_s = f"{m['depth_auc']['mean']:.1f}±{m['depth_auc']['std']:.1f}"
        t15_s = f"{m['t15_mean']:.0f} (n={len(m['t15_vals'])})" if m['t15_mean'] is not None else "NR (0/5)"
        cra_s = f"{m['crashes']['mean']:.0f}±{m['crashes']['std']:.0f}"
        spd_s = f"{m['speed']['mean']:.0f}±{m['speed']['std']:.0f}"
        print(f"{m['label']:<35} | {cov_s:<12} | {dep_s:<9} | {cauc_s:<9} | {dauc_s:<10} | {t15_s:<12} | {cra_s:<12} | {spd_s:<10}")
    print("=" * 115)

    # Compute Cohen's d effect sizes
    print("\n[3] Computing Effect Sizes (Cohen's d) vs NeuroFuzz Full (G):")
    g_cov = [float(r["total_coverage"]) for r in by_config["G_neurofuzz_full"]]
    g_dep = [float(r["max_depth"]) for r in by_config["G_neurofuzz_full"]]
    g_cauc = [float(r["coverage_auc_normalized"]) for r in by_config["G_neurofuzz_full"]]
    g_dauc = [float(r["depth_auc_normalized"]) for r in by_config["G_neurofuzz_full"]]
    
    comparisons = [("G vs A", "A_random_fixed"), ("G vs D", "D_random_ucb1"), ("G vs E", "E_random_contextual")]
    for comp_name, comp_cfg in comparisons:
        comp_cov = [float(r["total_coverage"]) for r in by_config[comp_cfg]]
        comp_dep = [float(r["max_depth"]) for r in by_config[comp_cfg]]
        comp_cauc = [float(r["coverage_auc_normalized"]) for r in by_config[comp_cfg]]
        comp_dauc = [float(r["depth_auc_normalized"]) for r in by_config[comp_cfg]]
        print(f"  {comp_name}:")
        print(f"    Coverage Cohen's d:  {cohens_d(g_cov, comp_cov):+.3f}")
        print(f"    Depth Cohen's d:     {cohens_d(g_dep, comp_dep):+.3f}")
        print(f"    Cov AUC Cohen's d:   {cohens_d(g_cauc, comp_cauc):+.3f}")
        print(f"    Depth AUC Cohen's d: {cohens_d(g_dauc, comp_dauc):+.3f}")

    # Correlation between crashes and coverage / depth
    all_crashes = [float(r["unique_crashes"]) for r in rows]
    all_cov = [float(r["total_coverage"]) for r in rows]
    all_depth = [float(r["max_depth"]) for r in rows]
    corr_cov, p_cov = stats.pearsonr(all_crashes, all_cov)
    corr_dep, p_dep = stats.pearsonr(all_crashes, all_depth)
    print(f"\n[4] Crash Correlations across all 35 trials:")
    print(f"    Crashes vs Coverage: r = {corr_cov:.3f} (p = {p_cov:.4f})")
    print(f"    Crashes vs MaxDepth: r = {corr_dep:.3f} (p = {p_dep:.4f})")

    # Generate Plots
    print("\n[5] Regenerating scientific plots...")
    configs_ordered = list(CONFIG_LABELS.keys())
    
    # 1. coverage_vs_executions.png
    plt.figure(figsize=(11, 6))
    for cfg in configs_ordered:
        seed_trajs = trajectories[cfg]
        all_iters = sorted(list({pt["iteration"] for s in seed_trajs for pt in seed_trajs[s]}))
        # Downsample for plotting performance (every 100 iters or so)
        sample_iters = [it for it in all_iters if it % 100 == 0 or it == all_iters[-1]]
        mean_covs, std_covs = [], []
        for it in sample_iters:
            pts = [pt["coverage"] for s in seed_trajs for pt in seed_trajs[s] if pt["iteration"] == it]
            if pts:
                mean_covs.append(float(np.mean(pts)))
                std_covs.append(float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0)
            else:
                mean_covs.append(mean_covs[-1] if mean_covs else 0.0)
                std_covs.append(0.0)
        plt.plot(sample_iters, mean_covs, label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg], linewidth=2.0)
        plt.fill_between(
            sample_iters,
            [m - s for m, s in zip(mean_covs, std_covs)],
            [m + s for m, s in zip(mean_covs, std_covs)],
            color=CONFIG_COLORS[cfg],
            alpha=0.12,
        )
    plt.title("Cumulative Code Coverage vs Executions (Phase 7B, 25k Budget)", fontsize=13, fontweight="bold")
    plt.xlabel("Execution Count", fontsize=11)
    plt.ylabel("Total Coverage Units", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "coverage_vs_executions.png", dpi=200)
    plt.close()
    print("    -> coverage_vs_executions.png generated")

    # 2. depth_vs_executions.png & max_depth_vs_executions.png
    plt.figure(figsize=(11, 6))
    for cfg in configs_ordered:
        seed_trajs = trajectories[cfg]
        all_iters = sorted(list({pt["iteration"] for s in seed_trajs for pt in seed_trajs[s]}))
        sample_iters = [it for it in all_iters if it % 100 == 0 or it == all_iters[-1]]
        mean_depths = []
        for it in sample_iters:
            pts = [pt["max_depth"] for s in seed_trajs for pt in seed_trajs[s] if pt["iteration"] == it]
            mean_depths.append(float(np.mean(pts)) if pts else 0.0)
        plt.plot(sample_iters, mean_depths, label=CONFIG_LABELS[cfg], color=CONFIG_COLORS[cfg], linewidth=2.0)
    plt.title("Protocol Depth Frontier vs Executions (Phase 7B, 25k Budget)", fontsize=13, fontweight="bold")
    plt.xlabel("Execution Count", fontsize=11)
    plt.ylabel("Maximum Depth Level Reached", fontsize=11)
    plt.yticks(range(0, 20, 2))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "depth_vs_executions.png", dpi=200)
    plt.savefig(PLOTS_DIR / "max_depth_vs_executions.png", dpi=200)
    plt.close()
    print("    -> depth_vs_executions.png (and max_depth_vs_executions.png) generated")

    # 3. final_coverage_comparison.png
    plt.figure(figsize=(10, 5.5))
    labels = [CONFIG_SHORT_LABELS[c] for c in configs_ordered]
    means_cov = [metrics_summary[c]["coverage"]["mean"] for c in configs_ordered]
    stds_cov = [metrics_summary[c]["coverage"]["std"] for c in configs_ordered]
    colors = [CONFIG_COLORS[c] for c in configs_ordered]
    
    bars = plt.bar(range(len(labels)), means_cov, yerr=stds_cov, capsize=5, color=colors, alpha=0.85, edgecolor="#333333")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9.5)
    plt.ylabel("Final Coverage Units", fontsize=11)
    plt.title("Final Code Coverage Comparison (Mean ± Std Dev)", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.ylim(50, 70)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_coverage_comparison.png", dpi=200)
    plt.close()
    print("    -> final_coverage_comparison.png generated")

    # 4. final_depth_comparison.png
    plt.figure(figsize=(10, 5.5))
    means_dep = [metrics_summary[c]["depth"]["mean"] for c in configs_ordered]
    stds_dep = [metrics_summary[c]["depth"]["std"] for c in configs_ordered]
    
    bars = plt.bar(range(len(labels)), means_dep, yerr=stds_dep, capsize=5, color=colors, alpha=0.85, edgecolor="#333333")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9.5)
    plt.ylabel("Maximum Depth Level", fontsize=11)
    plt.yticks(range(0, 20, 2))
    plt.ylim(0, 18)
    plt.title("Maximum Protocol Depth Level Discovered (Mean ± Std Dev)", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_depth_comparison.png", dpi=200)
    plt.close()
    print("    -> final_depth_comparison.png generated")

    # 5. time_to_depth.png (Bar chart showing mean T15 for trials that reached it, with reach rate annotations)
    plt.figure(figsize=(10, 5.5))
    t15_means_plot = []
    t15_reach_labels = []
    for c in configs_ordered:
        vals = metrics_summary[c]["t15_vals"]
        if vals:
            t15_means_plot.append(float(np.mean(vals)))
            t15_reach_labels.append(f"{len(vals)}/5 reached\n(mean: {np.mean(vals):.0f})")
        else:
            t15_means_plot.append(0.0)
            t15_reach_labels.append("0/5 reached\n(NR)")
            
    bars = plt.bar(range(len(labels)), t15_means_plot, color=colors, alpha=0.85, edgecolor="#333333")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9.5)
    plt.ylabel("Mean Executions to Reach Depth 15 (T15)", fontsize=11)
    plt.title("Search Speed to Frontier State: Time-to-Depth 15 (T15)", fontsize=13, fontweight="bold")
    for idx, bar in enumerate(bars):
        yval = bar.get_height()
        annot = t15_reach_labels[idx]
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 400, annot, ha="center", va="bottom", fontsize=8.5)
    plt.ylim(0, max(t15_means_plot) * 1.25)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_to_depth.png", dpi=200)
    plt.close()
    print("    -> time_to_depth.png generated")

    # 6. crash_comparison.png
    plt.figure(figsize=(10, 5.5))
    means_cra = [metrics_summary[c]["crashes"]["mean"] for c in configs_ordered]
    stds_cra = [metrics_summary[c]["crashes"]["std"] for c in configs_ordered]
    
    bars = plt.bar(range(len(labels)), means_cra, yerr=stds_cra, capsize=5, color=colors, alpha=0.85, edgecolor="#333333")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9.5)
    plt.ylabel("Unique Crash Inputs Discovered", fontsize=11)
    plt.title("Unique Crash Discoveries Across Configurations", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "crash_comparison.png", dpi=200)
    plt.close()
    print("    -> crash_comparison.png generated")

    # Additional Plot 7: operator_selection_distribution.png
    learned_cfgs = [c for c in ["D_random_ucb1", "E_random_contextual", "F_random_contextual_compatible", "G_neurofuzz_full"] if c in summary_obj.get("summary", {})]
    if learned_cfgs:
        plt.figure(figsize=(11, 6))
        all_ops = ["flip_bit", "replace_byte", "insert_byte", "delete_byte", "dictionary_insert_boundary", "dictionary_replace", "field_splice"]
        op_colors = ["#4e79a7", "#a0cbe8", "#f28e2c", "#ffbe7d", "#59a14f", "#8cd17d", "#b07aa1"]
        bottoms = [0.0] * len(learned_cfgs)
        cfg_labels = [CONFIG_SHORT_LABELS[c] for c in learned_cfgs]
        
        # Calculate operator proportions from learning_telemetry across all seeds
        shares_by_cfg = {}
        for c in learned_cfgs:
            t_list = summary_obj["summary"][c].get("learning_telemetry", [])
            op_counts = {}
            tot_selections = 0
            for t in t_list:
                m = t["mutation_bandit_telemetry"] if t.get("mutation_bandit_telemetry") else t.get("contextual_mutation_bandit_telemetry")
                if not m or "operators" not in m:
                    continue
                for op, data in m["operators"].items():
                    op_counts[op] = op_counts.get(op, 0) + data.get("selections", 0)
                    tot_selections += data.get("selections", 0)
            shares_by_cfg[c] = {op: op_counts.get(op, 0) / tot_selections if tot_selections > 0 else 0.0 for op in all_ops}
            
        for op_idx, op_name in enumerate(all_ops):
            shares = [shares_by_cfg[c].get(op_name, 0.0) for c in learned_cfgs]
            plt.bar(range(len(learned_cfgs)), shares, bottom=bottoms, label=op_name, color=op_colors[op_idx % len(op_colors)], width=0.55)
            bottoms = [b + s for b, s in zip(bottoms, shares)]
            
        plt.xticks(range(len(learned_cfgs)), cfg_labels, rotation=15, ha="right", fontsize=9.5)
        plt.ylabel("Proportion of Operator Pulls", fontsize=11)
        plt.title("Learned Mutation Operator Selection Shares (125k Selections/Config)", fontsize=13, fontweight="bold")
        plt.ylim(0, 1.05)
        plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0), fontsize=8.5)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "operator_selection_distribution.png", dpi=200)
        plt.close()
        print("    -> operator_selection_distribution.png regenerated with full telemetry")

    # Additional Plot 8: throughput_comparison.png
    plt.figure(figsize=(10, 5.5))
    means_spd = [metrics_summary[c]["speed"]["mean"] for c in configs_ordered]
    stds_spd = [metrics_summary[c]["speed"]["std"] for c in configs_ordered]
    bars = plt.bar(range(len(labels)), means_spd, yerr=stds_spd, capsize=5, color=colors, alpha=0.85, edgecolor="#333333")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=9.5)
    plt.ylabel("Throughput (Executions / Second)", fontsize=11)
    plt.title("Harness Execution Throughput Across Configurations", fontsize=13, fontweight="bold")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "throughput_comparison.png", dpi=200)
    plt.close()
    print("    -> throughput_comparison.png generated")

    # Additional Plot 9: auc_comparison.png
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    width = 0.35
    means_cauc = [metrics_summary[c]["cov_auc"]["mean"] for c in configs_ordered]
    means_dauc = [metrics_summary[c]["depth_auc"]["mean"] for c in configs_ordered]
    rects1 = ax1.bar(x - width/2, means_cauc, width, label="Coverage AUC (Norm)", color="#4e79a7", alpha=0.85)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, means_dauc, width, label="Depth AUC (Norm)", color="#e15759", alpha=0.85)
    ax1.set_ylabel("Normalized Coverage AUC", color="#4e79a7", fontsize=11)
    ax2.set_ylabel("Normalized Depth AUC", color="#e15759", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9.5)
    ax1.set_ylim(50, 70)
    ax2.set_ylim(10, 18)
    plt.title("Normalized Area-Under-the-Curve (Coverage & Depth)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(PLOTS_DIR / "auc_comparison.png", dpi=200)
    plt.close()
    print("    -> auc_comparison.png generated")

    print("\n[6] All analysis and plot regeneration complete!")
    return metrics_summary


if __name__ == "__main__":
    main()
