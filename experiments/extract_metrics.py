import csv
import json
import math
import numpy as np
from scipy import stats

with open('experiments/results/phase7b/results.csv', 'r') as f:
    rows = list(csv.DictReader(f))

by_cfg = {}
for r in rows:
    by_cfg.setdefault(r['config'], []).append(r)

print("=== INDIVIDUAL TRIALS ===")
for cfg, r_list in by_cfg.items():
    print(f"\n--- {cfg} ---")
    for r in r_list:
        t15 = r['t15'] if r['t15'] else 'NOT REACHED'
        t16 = r['t16'] if r['t16'] else 'NOT REACHED'
        print(f"Seed {r['seed']}: Cov={r['total_coverage']}, Depth={r['max_depth']}, T15={t15}, T16={t16}, CovAUC={float(r['coverage_auc_normalized']):.2f}, DepthAUC={float(r['depth_auc_normalized']):.2f}, Crashes={r['unique_crashes']}, Exec/s={float(r['exec_per_sec']):.1f}")

print("\n=== METRIC SUMMARY TABLE ===")
for cfg, r_list in by_cfg.items():
    covs = [float(r['total_coverage']) for r in r_list]
    deps = [float(r['max_depth']) for r in r_list]
    c_aucs = [float(r['coverage_auc_normalized']) for r in r_list]
    d_aucs = [float(r['depth_auc_normalized']) for r in r_list]
    crashes = [float(r['unique_crashes']) for r in r_list]
    speeds = [float(r['exec_per_sec']) for r in r_list]
    t15s = [float(r['t15']) for r in r_list if r['t15']]
    t16s = [float(r['t16']) for r in r_list if r['t16']]
    
    ci_cov = stats.t.interval(0.95, df=4, loc=np.mean(covs), scale=np.std(covs, ddof=1)/math.sqrt(5))
    ci_dep = stats.t.interval(0.95, df=4, loc=np.mean(deps), scale=np.std(deps, ddof=1)/math.sqrt(5))
    ci_cauc = stats.t.interval(0.95, df=4, loc=np.mean(c_aucs), scale=np.std(c_aucs, ddof=1)/math.sqrt(5))
    ci_dauc = stats.t.interval(0.95, df=4, loc=np.mean(d_aucs), scale=np.std(d_aucs, ddof=1)/math.sqrt(5))
    ci_cra = stats.t.interval(0.95, df=4, loc=np.mean(crashes), scale=np.std(crashes, ddof=1)/math.sqrt(5))
    
    print(f"\nConfiguration: {cfg}")
    print(f"  Coverage: Mean={np.mean(covs):.2f}, Std={np.std(covs, ddof=1):.2f}, Min={min(covs)}, Max={max(covs)}, 95% CI=[{ci_cov[0]:.2f}, {ci_cov[1]:.2f}]")
    print(f"  Depth: Mean={np.mean(deps):.2f}, Std={np.std(deps, ddof=1):.2f}, Min={min(deps)}, Max={max(deps)}, 95% CI=[{ci_dep[0]:.2f}, {ci_dep[1]:.2f}]")
    print(f"  Coverage AUC: Mean={np.mean(c_aucs):.2f}, Std={np.std(c_aucs, ddof=1):.2f}, Min={min(c_aucs):.2f}, Max={max(c_aucs):.2f}, 95% CI=[{ci_cauc[0]:.2f}, {ci_cauc[1]:.2f}]")
    print(f"  Depth AUC: Mean={np.mean(d_aucs):.2f}, Std={np.std(d_aucs, ddof=1):.2f}, Min={min(d_aucs):.2f}, Max={max(d_aucs):.2f}, 95% CI=[{ci_dauc[0]:.2f}, {ci_dauc[1]:.2f}]")
    print(f"  Crashes: Mean={np.mean(crashes):.2f}, Std={np.std(crashes, ddof=1):.2f}, Min={min(crashes)}, Max={max(crashes)}, 95% CI=[{ci_cra[0]:.2f}, {ci_cra[1]:.2f}]")
    print(f"  Speed (exec/s): Mean={np.mean(speeds):.1f}, Std={np.std(speeds, ddof=1):.1f}, Min={min(speeds):.1f}, Max={max(speeds):.1f}")
    if t15s:
        std_t15 = np.std(t15s, ddof=1) if len(t15s) > 1 else 0.0
        print(f"  T15 (n={len(t15s)}): Mean={np.mean(t15s):.1f}, Median={np.median(t15s):.1f}, Std={std_t15:.1f}, Min={min(t15s)}, Max={max(t15s)}")
    else:
        print("  T15: NOT REACHED in any trial")
    if t16s:
        std_t16 = np.std(t16s, ddof=1) if len(t16s) > 1 else 0.0
        print(f"  T16 (n={len(t16s)}): Mean={np.mean(t16s):.1f}, Median={np.median(t16s):.1f}, Std={std_t16:.1f}, Min={min(t16s)}, Max={max(t16s)}")
    else:
        print("  T16: NOT REACHED in any trial")

print("\n=== DEPTH REACH RATES ===")
print("configuration | D15 | D16 | D17 | D18 | D19")
for cfg, r_list in by_cfg.items():
    deps = [float(r['max_depth']) for r in r_list]
    d15 = sum(1 for d in deps if d >= 15)
    d16 = sum(1 for d in deps if d >= 16)
    d17 = sum(1 for d in deps if d >= 17)
    d18 = sum(1 for d in deps if d >= 18)
    d19 = sum(1 for d in deps if d >= 19)
    print(f"{cfg} | {d15}/5 ({d15/5:.1%}) | {d16}/5 ({d16/5:.1%}) | {d17}/5 ({d17/5:.1%}) | {d18}/5 ({d18/5:.1%}) | {d19}/5 ({d19/5:.1%})")

print("\n=== DETAILED DISCOVERIES ===")
with open('experiments/results/phase7b/discoveries.json', 'r') as f:
    discs = json.load(f)

for d in discs:
    print(f"[{d['config']}] Seed {d['seed']}, Iter {d['iteration']}: Transition {d['transition']} (Depth {d['depth']}) via {d['operator']}")
    print(f"   Parent: {d['parent_name']}")
    print(f"   Mutated: {repr(d['mutated_payload'])}")
    print(f"   New coverage: {d['new_coverage_units']}")
