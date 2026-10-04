import csv
import json

with open('experiments/results/phase7b/results.csv', 'r') as f:
    rows = list(csv.DictReader(f))

print("| Config | Seed | Coverage | Max Depth | T14 | T15 | T16 | Cov AUC | Depth AUC | Crashes | Exec/s |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    t14 = r['t14'] if r['t14'] else "NR"
    t15 = r['t15'] if r['t15'] else "NR"
    t16 = r['t16'] if r['t16'] else "NR"
    print(f"| {r['config']} | {r['seed']} | {r['total_coverage']} | {r['max_depth']} | {t14} | {t15} | {t16} | {float(r['coverage_auc_normalized']):.2f} | {float(r['depth_auc_normalized']):.2f} | {r['unique_crashes']} | {float(r['exec_per_sec']):.1f} |")
