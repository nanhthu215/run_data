# -*- coding: utf-8 -*-
"""
run_kfold_experiment.py — Unified Experiment Runner
====================================================
Chạy toàn bộ thực nghiệm K-fold × Multi-seed cho tất cả models.

Đảm bảo yêu cầu khoa học:
  1. Tất cả models chạy cùng môi trường, cùng split, cùng seed
  2. Checkpoint lưu riêng từng fold/seed để thầy kiểm tra
  3. Skip nếu kết quả đã tồn tại (tránh chạy lại tốn thời gian GPU)

Usage:
  # Chạy tất cả (4 baseline + model đề xuất) × 2 dataset × 4 fold × 3 seed
  python scripts/run_kfold_experiment.py

  # Chỉ chạy 1 dataset
  python scripts/run_kfold_experiment.py --datasets PEMS08

  # Chỉ chạy 1 model
  python scripts/run_kfold_experiment.py --models AGCRN

  # Chỉ chạy 1 fold 1 seed (debug)
  python scripts/run_kfold_experiment.py --folds 0 --seeds 42

  # Smoke test (kiểm tra pipeline trước khi chạy full)
  python scripts/run_kfold_experiment.py --smoke-test

  # Dry-run (chỉ in lệnh, không chạy)
  python scripts/run_kfold_experiment.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import json
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# Cấu hình thực nghiệm
# ============================================================

# Danh sách tất cả models cần chạy
# Mỗi entry: (model_key, module_path, extra_args)
MODEL_CONFIGS = {
    "DCRNN": {
        "module": "baselines.dcrnn.run",
        "args": [],
    },
    "AGCRN": {
        "module": "baselines.agcrn.run",
        "args": [],
    },
    "PGCRN": {
        "module": "baselines.pgcrn.run",
        "args": [],
    },
    "STFGNN": {
        "module": "baselines.stfgnn.run",
        "args": [],
    },
    "PDFormer": {
        "module": "baselines.pdformer.run",
        "args": [],
    },
    "STDMAE": {
        "module": "baselines.stdmae.run",
        "args": [],
    },
    "AdaptiveTGCN": {
        "module": "models.adaptive_tgcn.run",
        # Cấu hình tốt nhất từ ablation study
        "args": [
            "--graph-mode", "fused",
            "--fusion-type", "learnable",
            "--top-k", "5",
        ],
    },
}

DATASETS = ["PEMS04", "PEMS08"]
SEEDS = [42, 52, 62]
FOLDS = [0, 1, 2, 3]


def get_result_filename(model_key, dataset, seed, fold):
    """Dự đoán tên file kết quả để kiểm tra đã tồn tại chưa."""
    fold_tag = f"fold{fold}" if fold != "default" else "default"
    if model_key == "AdaptiveTGCN":
        return f"adaptive_tgcn_fused_learnable_k5_{dataset.lower()}_seed{seed}_{fold_tag}.json"
    else:
        return f"{model_key.lower()}_{dataset.lower()}_seed{seed}_{fold_tag}.json"


def build_command(model_key, config, dataset, seed, fold, smoke_test=False):
    """Xây dựng lệnh chạy cho 1 thực nghiệm."""
    cmd = [
        sys.executable, "-m", config["module"],
        "--dataset", dataset,
        "--seed", str(seed),
        "--fold", str(fold),
    ]
    cmd.extend(config["args"])
    if smoke_test:
        cmd.append("--smoke-test")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Unified K-Fold × Multi-Seed Experiment Runner"
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Smoke test (3 epoch, 10 batch)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chỉ in lệnh, không chạy")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Bỏ qua nếu kết quả JSON đã tồn tại")
    args = parser.parse_args()

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ---- Tính tổng số thực nghiệm ----
    experiments = []
    for model_key in args.models:
        for dataset in args.datasets:
            for fold in args.folds:
                for seed in args.seeds:
                    experiments.append((model_key, dataset, fold, seed))

    total = len(experiments)
    skipped = 0
    completed = 0
    failed = 0

    print(f"\n{'='*70}")
    print(f"  UNIFIED K-FOLD × MULTI-SEED EXPERIMENT RUNNER")
    print(f"  Models: {args.models}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Folds: {args.folds}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total experiments: {total}")
    if args.smoke_test:
        print(f"  MODE: SMOKE TEST (3 epochs, 10 batches)")
    if args.dry_run:
        print(f"  MODE: DRY RUN (print commands only)")
    print(f"{'='*70}\n")

    for i, (model_key, dataset, fold, seed) in enumerate(experiments, 1):
        # Check if already completed
        result_file = get_result_filename(model_key, dataset, seed, fold)
        result_path = os.path.join(results_dir, result_file)

        if args.skip_existing and os.path.exists(result_path) and not args.smoke_test:
            skipped += 1
            print(f"[{i}/{total}] SKIP {model_key} | {dataset} | fold{fold} | seed{seed} — result exists")
            continue

        config = MODEL_CONFIGS[model_key]
        cmd = build_command(model_key, config, dataset, seed, fold, args.smoke_test)

        print(f"\n[{i}/{total}] RUN {model_key} | {dataset} | fold{fold} | seed{seed}")
        print(f"  CMD: {' '.join(cmd)}")

        if args.dry_run:
            continue

        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, cwd=ROOT,
                capture_output=False,
                text=True,
            )
            elapsed = time.time() - t0

            if result.returncode == 0:
                completed += 1
                print(f"  [OK] Completed in {elapsed/60:.1f} min")
            else:
                failed += 1
                print(f"  [FAIL] Failed (exit code {result.returncode}) after {elapsed/60:.1f} min")

        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Completed {completed}/{total} experiments.")
            break
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {e}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"  Total: {total} | Completed: {completed} | Skipped: {skipped} | Failed: {failed}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
