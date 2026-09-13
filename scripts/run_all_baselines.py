"""
Script chạy tất cả 6 baseline models trên PeMS04 và PeMS08.
Kết quả được tổng hợp vào results/baseline_results.csv

Usage:
  # Smoke test (2 epochs, kiểm tra pipeline)
  python scripts/run_all_baselines.py --smoke-test

  # Full training (50 epochs)
  python scripts/run_all_baselines.py --epochs 50

  # Chỉ chạy 1 model
  python scripts/run_all_baselines.py --models pgcrn --datasets PEMS04
"""

import argparse
import sys
import os
import json
import csv
import subprocess
import time

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


MODELS = ["agcrn", "dcrnn", "stfgnn", "pdformer", "stdmae", "pgcrn"]
DATASETS = ["PEMS04", "PEMS08"]
HORIZONS = [15, 30, 60]  # minutes


def run_baseline(model: str, dataset: str, epochs: int, smoke_test: bool = False) -> dict:
    """
    Chạy một baseline model bằng subprocess để tránh memory leak.
    Trả về results dict hoặc None nếu lỗi.
    """
    cmd = [
        sys.executable, "-u", "-m", f"baselines.{model}.run",
        "--dataset", dataset,
        "--epochs", str(epochs),
    ]
    if smoke_test:
        cmd.append("--smoke-test")

    print(f"\n{'#'*60}")
    print(f"# Running: {model.upper()} on {dataset}")
    print(f"# Command: {' '.join(cmd)}")
    print(f"{'#'*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[ERROR] {model} on {dataset} failed with code {result.returncode}")
        return None

    results_path = os.path.join(ROOT, "results", f"{model}_{dataset.lower()}.json")
    if not os.path.exists(results_path):
        print(f"[ERROR] Results file not found: {results_path}")
        return None

    with open(results_path) as f:
        data = json.load(f)

    print(f"[OK] {model.upper()} on {dataset} done in {elapsed/60:.1f} min")
    return data.get("results", {})


def collect_results_from_json() -> list:
    rows = []
    results_dir = os.path.join(ROOT, "results")

    for model in MODELS:
        for dataset in DATASETS:
            path = os.path.join(results_dir, f"{model}_{dataset.lower()}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                data = json.load(f)
            results = data.get("results", {})
            for minutes_str, metrics in results.items():
                wmape_val = f"{metrics['wmape']:.2f}" if "wmape" in metrics else "N/A"
                rows.append({
                    "Model": model.upper().replace("STDMAE", "STD-MAE"),
                    "Dataset": dataset,
                    "Horizon_min": int(minutes_str),
                    "MAE": f"{metrics['mae']:.4f}",
                    "RMSE": f"{metrics['rmse']:.4f}",
                    "MAPE": f"{metrics['mape']:.2f}",
                    "WMAPE": wmape_val,
                })
    return rows


def save_csv(rows: list, path: str):
    if not rows:
        print("No results to save.")
        return

    fieldnames = ["Model", "Dataset", "Horizon_min", "MAE", "RMSE", "MAPE", "WMAPE"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Saved] {path}")


def print_table(rows: list):
    if not rows:
        return
    print(f"\n{'='*80}")
    print(f"{'BASELINE RESULTS':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Dataset':<8} {'Horizon':>8} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'WMAPE%':>8}")
    print(f"{'-'*80}")

    prev_model = None
    for row in sorted(rows, key=lambda r: (r["Model"], r["Dataset"], r["Horizon_min"])):
        model = row["Model"]
        if model != prev_model:
            if prev_model is not None:
                print(f"{'-'*80}")
            prev_model = model
        print(f"{model:<12} {row['Dataset']:<8} {row['Horizon_min']:>6}min "
              f"{row['MAE']:>8} {row['RMSE']:>8} {row['MAPE']:>7}% {row['WMAPE']:>7}%")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run all 6 baseline models")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        choices=MODELS, help="Models to run")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS, help="Datasets to use")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick smoke test (2 epochs, small batch)")
    parser.add_argument("--collect-only", action="store_true",
                        help="Chỉ thu thập kết quả từ JSON đã có, không train lại")
    args = parser.parse_args()

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    if args.collect_only:
        print("Collecting existing results...")
    else:
        failed = []
        for model in args.models:
            for dataset in args.datasets:
                result = run_baseline(model, dataset, args.epochs, args.smoke_test)
                if result is None:
                    failed.append(f"{model} on {dataset}")

        if failed:
            print(f"\n[WARNING] Failed models: {', '.join(failed)}")

    rows = collect_results_from_json()
    csv_path = os.path.join(results_dir, "baseline_results.csv")
    save_csv(rows, csv_path)
    print_table(rows)

    print(f"\nDone! Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
