"""E26: Class weight sweep — run 3 configurations."""
import os, sys, json, subprocess, glob

BASE_DIR = os.path.expanduser("~/bitoguard-hackathon/bitoguard_core")
os.chdir(BASE_DIR)

def run_pipeline(env_vars, tag):
    env = os.environ.copy()
    env.update(env_vars)
    env["PYTHONPATH"] = "."
    env["SKIP_GNN"] = "1"
    print(f"\n{'='*60}", flush=True)
    print(f"Running {tag}... env: {env_vars}", flush=True)
    print(f"{'='*60}", flush=True)
    for f in glob.glob("artifacts/official_features/official_*.parquet"):
        os.remove(f)
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "official.pipeline"],
        env=env, capture_output=True, text=True, cwd=BASE_DIR
    )
    with open(f"/tmp/e_{tag}.log", "w") as f:
        f.write(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        print(f"  FAILED!", flush=True)
        for line in proc.stderr.strip().split("\n")[-10:]:
            print(f"  | {line}", flush=True)
        return None
    report_path = os.path.join(BASE_DIR, "artifacts/reports/official_validation_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            r = json.load(f)
        cal = r["calibrator"]
        sel = cal["selected_row"]
        sec = r.get("secondary_group_stress_metrics", {})
        result = {
            "tag": tag, "primary_f1": sel["f1"], "primary_ap": cal["average_precision"],
            "threshold": r["selected_threshold"],
            "secondary_f1": sec.get("f1", "N/A"), "secondary_ap": sec.get("average_precision", "N/A"),
            "tp": sel["tp"], "fp": sel["fp"], "fn": sel["fn"],
        }
        sec_f1 = f"{result['secondary_f1']:.4f}" if isinstance(result['secondary_f1'], float) else str(result['secondary_f1'])
        print(f"  Primary F1={result['primary_f1']:.4f}, AP={result['primary_ap']:.4f}, Secondary F1={sec_f1}", flush=True)
        return result
    print("  No report!", flush=True)
    return None

results = []
r = run_pipeline({"CB_MAX_CLASS_WEIGHT": "5", "XGB_SPW_CAP": "8"}, "E26a_low_cw")
if r: results.append(r)
r = run_pipeline({"CB_MAX_CLASS_WEIGHT": "20", "XGB_SPW_CAP": "25"}, "E26b_high_cw")
if r: results.append(r)
r = run_pipeline({"CB_MAX_CLASS_WEIGHT": "999", "XGB_SPW_CAP": "999"}, "E26c_uncapped")
if r: results.append(r)

print("\n" + "=" * 80, flush=True)
print("E26 RESULTS", flush=True)
print("=" * 80, flush=True)
print(f"{'Tag':<20} {'Pri F1':>8} {'Sec F1':>8} {'AP':>8} {'ΔF1':>8}", flush=True)
print("-" * 56, flush=True)
print(f"{'E15 baseline':<20} {'0.4418':>8} {'0.4304':>8} {'0.3842':>8} {'---':>8}", flush=True)
for r in results:
    delta = r["primary_f1"] - 0.4418
    sec_f1 = f"{r['secondary_f1']:.4f}" if isinstance(r["secondary_f1"], float) else str(r["secondary_f1"])
    print(f"{r['tag']:<20} {r['primary_f1']:>8.4f} {sec_f1:>8} {r['primary_ap']:>8.4f} {delta:>+8.4f}", flush=True)
with open("/tmp/e26_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
