# -*- coding: utf-8 -*-
"""
run_ablation_study.py — Trình Chạy Tự Động Toàn Bộ Bảng Ablation Study
=======================================================================
Tự động chạy lần lượt tất cả các biến thể kiến trúc của mô hình đề xuất
(Adaptive T-GCN) để phân tích đóng góp của từng thành phần:
  1. T-GCN-P        : Chỉ dùng đồ thị vật lý cố định
  2. T-GCN-A        : Chỉ dùng đồ thị thích nghi (tự học)
  3. T-GCN-PA fixed : Hợp nhất cố định alpha = 0.5
  4. T-GCN-PA learn : Hợp nhất có trọng số tự học (learnable alpha)
  5. + Top-k = 5    : Lọc top 5 cạnh mạnh nhất (CẤU HÌNH TỐI ƯU)
  6. + Top-k = 10   : Lọc top 10 cạnh mạnh nhất
  7. + Top-k = 20   : Lọc top 20 cạnh mạnh nhất
  8. + Graph Reg    : Top-k=10 kèm ràng buộc đồ thị (Graph Regularization)

Sau khi chạy xong, tự động tổng hợp ra bảng so sánh Ablation hoàn chỉnh
và lưu vào results/ablation_summary_{dataset}.md!

Usage:
  # Chạy toàn bộ 8 cấu hình trên PEMS08 (100 epoch, patience 20):
  python scripts/run_ablation_study.py --dataset PEMS08

  # Chạy trên PEMS04:
  python scripts/run_ablation_study.py --dataset PEMS04

  # Smoke test (kiểm tra toàn bộ pipeline, 3 epoch):
  python scripts/run_ablation_study.py --smoke-test

  # Chỉ chạy một số cấu hình cụ thể:
  python scripts/run_ablation_study.py --configs physical adaptive topk_5

  # Chạy trên 1 fold cụ thể (nếu muốn làm ablation trên fold):
  python scripts/run_ablation_study.py --fold 0
"""

import argparse
import os
import sys
import json
import subprocess
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# =========================================================================
# Danh sách 8 cấu hình Ablation Study
# =========================================================================
ABLATION_CONFIGS = {
    "physical": {
        "name": "T-GCN-P (Physical only)",
        "args": ["--graph-mode", "physical"],
        "out_tag": "physical",
    },
    "adaptive": {
        "name": "T-GCN-A (Adaptive only)",
        "args": ["--graph-mode", "adaptive"],
        "out_tag": "adaptive",
    },
    "fused_fixed": {
        "name": "T-GCN-PA (Fixed alpha=0.5)",
        "args": ["--graph-mode", "fused", "--fusion-type", "fixed", "--alpha", "0.5"],
        "out_tag": "fused_fixed_a0.5",
    },
    "fused_learnable": {
        "name": "T-GCN-PA (Learnable alpha)",
        "args": ["--graph-mode", "fused", "--fusion-type", "learnable"],
        "out_tag": "fused_learnable",
    },
    "topk_5": {
        "name": "T-GCN-PA (Learnable + Top-k=5)",
        "args": ["--graph-mode", "fused", "--fusion-type", "learnable", "--top-k", "5"],
        "out_tag": "fused_learnable_k5",
    },
    "topk_10": {
        "name": "T-GCN-PA (Learnable + Top-k=10)",
        "args": ["--graph-mode", "fused", "--fusion-type", "learnable", "--top-k", "10"],
        "out_tag": "fused_learnable_k10",
    },
    "topk_20": {
        "name": "T-GCN-PA (Learnable + Top-k=20)",
        "args": ["--graph-mode", "fused", "--fusion-type", "learnable", "--top-k", "20"],
        "out_tag": "fused_learnable_k20",
    },
    "topk_10_reg": {
        "name": "T-GCN-PA (Top-k=10 + Graph Reg)",
        "args": ["--graph-mode", "fused", "--fusion-type", "learnable", "--top-k", "10", "--use-graph-reg"],
        "out_tag": "fused_learnable_k10_reg",
    },
}


def get_expected_json_path(out_tag, dataset, seed, fold):
    results_dir = os.path.join(ROOT, "results")
    fold_tag = f"fold{fold}" if fold != "default" else "default"
    # Tên file theo quy chuẩn của models/adaptive_tgcn/run.py
    fname = f"adaptive_tgcn_{out_tag}_{dataset.lower()}_seed{seed}_{fold_tag}.json"
    p = os.path.join(results_dir, fname)
    if os.path.exists(p):
        return p
    # Fallback kiểm tra file tên cũ (nếu seed=42 và fold=default)
    old_fname = f"adaptive_tgcn_{out_tag}_{dataset.lower()}.json"
    old_p = os.path.join(results_dir, old_fname)
    if os.path.exists(old_p):
        return old_p
    # Kiểm tra trong results/old/
    old_dir_p = os.path.join(results_dir, "old", old_fname)
    if os.path.exists(old_dir_p):
        return old_dir_p
    return p


def print_ablation_table(dataset, seed, fold, selected_keys):
    """Đọc các file JSON và in ra bảng tổng hợp Ablation Study."""
    print(f"\n{'='*95}")
    print(f"  ABLATION STUDY SUMMARY TABLE — {dataset} (Seed: {seed}, Fold: {fold})")
    print(f"{'='*95}")
    print(f"| {'Configuration':<35} | {'MAE 15m':>8} | {'MAE 30m':>8} | {'MAE 60m':>8} | {'RMSE 60m':>8} | {'MAPE 60m':>9} |")
    print(f"|{'-'*37}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*11}|")

    summary_rows = []
    best_mae60 = float('inf')
    best_key = None

    for key in selected_keys:
        cfg = ABLATION_CONFIGS[key]
        json_path = get_expected_json_path(cfg["out_tag"], dataset, seed, fold)

        if not os.path.exists(json_path):
            print(f"| {cfg['name']:<35} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8} | {'N/A':>9} |")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            res = data.get("results", {})
            m15 = res.get("15", {}).get("mae", 0.0)
            m30 = res.get("30", {}).get("mae", 0.0)
            m60 = res.get("60", {}).get("mae", 0.0)
            r60 = res.get("60", {}).get("rmse", 0.0)
            p60 = res.get("60", {}).get("mape", 0.0)

            if m60 < best_mae60:
                best_mae60 = m60
                best_key = key

            row_str = f"| {cfg['name']:<35} | {m15:>8.2f} | {m30:>8.2f} | {m60:>8.2f} | {r60:>8.2f} | {p60:>8.2f}% |"
            print(row_str)
            summary_rows.append({
                "name": cfg["name"],
                "mae15": m15, "mae30": m30, "mae60": m60, "rmse60": r60, "mape60": p60,
            })
        except Exception as e:
            print(f"| {cfg['name']:<35} | Error reading file: {e}")

    print(f"{'='*95}")
    if best_key:
        print(f"  * BEST CONFIGURATION: {ABLATION_CONFIGS[best_key]['name']} (MAE 60m = {best_mae60:.2f})")
    print(f"{'='*95}\n")

    # Xuất ra file markdown tóm tắt
    md_path = os.path.join(ROOT, "results", f"ablation_summary_{dataset.lower()}.md")
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Ablation Study Results — {dataset}\n\n")
            f.write(f"| Configuration | MAE 15m | MAE 30m | MAE 60m | RMSE 60m | MAPE 60m |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            for r in summary_rows:
                f.write(f"| {r['name']} | {r['mae15']:.2f} | {r['mae30']:.2f} | {r['mae60']:.2f} | {r['rmse60']:.2f} | {r['mape60']:.2f}% |\n")
            if best_key:
                f.write(f"\n**Best configuration**: `{ABLATION_CONFIGS[best_key]['name']}` with MAE 60m = **{best_mae60:.2f}**.\n")
        print(f"  -> Summary table saved to: {md_path}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Automated Ablation Study Runner")
    parser.add_argument("--dataset", type=str, default="PEMS08", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) or 'default' for standard 60/20/20 split")
    parser.add_argument("--configs", nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Select specific configs to run")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip config if JSON result file already exists")
    parser.add_argument("--force", action="store_true",
                        help="Force re-running all configs without skipping")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick 3 epochs, 10 batches for pipeline verification")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.force and not args.smoke_test

    print(f"\n{'='*75}")
    print(f"  ABLATION STUDY AUTOMATION — {args.dataset}")
    print(f"  Configs: {len(args.configs)} | Epochs: {args.epochs} | Seed: {args.seed} | Fold: {args.fold}")
    if args.smoke_test:
        print(f"  MODE: SMOKE TEST (3 epochs, 10 batches)")
    print(f"{'='*75}\n")

    for i, key in enumerate(args.configs, 1):
        cfg = ABLATION_CONFIGS[key]
        expected_json = get_expected_json_path(cfg["out_tag"], args.dataset, args.seed, args.fold)

        if skip_existing and os.path.exists(expected_json):
            print(f"[{i}/{len(args.configs)}] SKIP: {cfg['name']} (Already exists: {os.path.basename(expected_json)})")
            continue

        cmd = [
            sys.executable, "-m", "models.adaptive_tgcn.run",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--seed", str(args.seed),
            "--fold", str(args.fold),
        ]
        cmd.extend(cfg["args"])
        if args.smoke_test:
            cmd.append("--smoke-test")

        print(f"\n[{i}/{len(args.configs)}] RUNNING: {cfg['name']}")
        print(f"  Command: {' '.join(cmd)}")

        if args.dry_run:
            continue

        t0 = time.time()
        res = subprocess.run(cmd, cwd=ROOT, text=True)
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"  [OK] Completed in {elapsed/60:.1f} min")
        else:
            print(f"  [FAIL] Failed (Exit code: {res.returncode})")

    # In bảng tổng hợp sau khi chạy xong
    if not args.dry_run:
        print_ablation_table(args.dataset, args.seed, args.fold, args.configs)


if __name__ == "__main__":
    main()
