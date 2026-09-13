# -*- coding: utf-8 -*-
"""
compute_statistics.py — Phân tích thống kê kết quả thực nghiệm
================================================================
Quét tất cả file JSON kết quả, tính Mean ± Std, chạy kiểm định thống kê.

Đáp ứng yêu cầu giảng viên:
  - Tính Mean ± Std cho tất cả metrics (MAE, RMSE, MAPE, WMAPE)
  - Kiểm định Paired t-test và Wilcoxon signed-rank test
  - Xuất bảng Markdown và LaTeX cho báo cáo
  - Đánh dấu cải thiện có ý nghĩa thống kê (p < 0.05)

Usage:
  python scripts/compute_statistics.py
  python scripts/compute_statistics.py --results-dir results/ --output-format markdown
  python scripts/compute_statistics.py --results-dir results/ --output-format latex
  python scripts/compute_statistics.py --proposed-model AdaptiveTGCN
"""

import argparse
import os
import sys
import json
import glob
import warnings
from collections import defaultdict

# Đảm bảo stdout hỗ trợ UTF-8 trên Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

# Optional: scipy cho kiểm định thống kê
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy không có sẵn — bỏ qua kiểm định thống kê. "
                  "Cài đặt: pip install scipy")


ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

HORIZONS = ["15", "30", "60"]
METRICS = ["mae", "rmse", "mape", "wmape"]


def load_all_results(results_dir):
    """
    Quét tất cả file JSON trong results_dir.
    Trả về dict: {model: {dataset: [result_dict, ...]}}

    LƯU Ý: File *_default.json (chạy fix-split 60/20/20, dùng cho ablation)
    được LOẠI KHỎI nhóm K-Fold chính để không làm nhiễu Mean ± Std.
    """
    all_results = defaultdict(lambda: defaultdict(list))
    default_results = defaultdict(lambda: defaultdict(list))
    n_skipped_default = 0

    for fpath in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        fname = os.path.basename(fpath)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        model = data.get("model", "")
        dataset = data.get("dataset", "")
        results = data.get("results", {})
        config = data.get("config", {})

        if not model or not dataset or not results:
            continue

        entry = {
            "file": fname,
            "seed": config.get("seed", "?"),
            "fold": config.get("fold", "?"),
            "results": results,
        }

        # Phân loại: file _default.json KHÔNG gộp vào K-Fold chính
        is_default = fname.endswith("_default.json") or str(config.get("fold", "")).lower() == "default"
        if is_default:
            default_results[model][dataset].append(entry)
            n_skipped_default += 1
        else:
            all_results[model][dataset].append(entry)

    if n_skipped_default > 0:
        print(f"\n[INFO] Đã loại {n_skipped_default} file *_default.json ra khỏi phép tính Mean±Std K-Fold.")
        print(f"       (Các file này chỉ dùng cho Ablation Study fix-split 60/20/20, không thuộc K-Fold.)")
        for model in sorted(default_results):
            for dataset in sorted(default_results[model]):
                files = [e["file"] for e in default_results[model][dataset]]
                print(f"       Loại: {model} × {dataset}: {', '.join(files)}")

    return all_results


def compute_mean_std(entries, horizon, metric):
    """
    Tính Mean ± Std cho 1 metric tại 1 horizon, qua tất cả entries (fold/seed).
    """
    values = []
    for e in entries:
        r = e["results"].get(str(horizon), {})
        v = r.get(metric)
        if v is not None:
            values.append(v)

    if not values:
        return None, None, []

    arr = np.array(values)
    return float(arr.mean()), float(arr.std()), values


def paired_test(values_a, values_b, test_type="ttest"):
    """
    Kiểm định thống kê paired (từng fold/seed phải khớp).
    Trả về (statistic, p_value) hoặc (None, None) nếu thiếu dữ liệu.
    """
    if not HAS_SCIPY:
        return None, None

    n = min(len(values_a), len(values_b))
    if n < 3:
        return None, None

    a = np.array(values_a[:n])
    b = np.array(values_b[:n])

    if test_type == "ttest":
        stat, pval = scipy_stats.ttest_rel(a, b)
    elif test_type == "wilcoxon":
        try:
            stat, pval = scipy_stats.wilcoxon(a, b)
        except ValueError:
            return None, None
    else:
        return None, None

    return float(stat), float(pval)


def format_mean_std(mean, std, decimal=2):
    """Format: 24.87 ± 0.47"""
    if mean is None:
        return "—"
    if std is None or std == 0:
        return f"{mean:.{decimal}f}"
    return f"{mean:.{decimal}f} ± {std:.{decimal}f}"


def print_summary_table(all_results, output_format="markdown"):
    """In bảng tổng hợp Mean ± Std cho tất cả models/datasets."""
    print(f"\n{'='*80}")
    print(f"  EXPERIMENTAL RESULTS SUMMARY (Mean ± Std across folds/seeds)")
    print(f"{'='*80}")

    for dataset in sorted({d for model_data in all_results.values() for d in model_data}):
        print(f"\n## Dataset: {dataset}\n")

        # Header
        if output_format == "markdown":
            header = "| Model | #Runs |"
            sep = "|---|---|"
            for h in HORIZONS:
                header += f" MAE {h}m |"
                sep += "---|"
            print(header)
            print(sep)

        for model in sorted(all_results.keys()):
            entries = all_results[model].get(dataset, [])
            if not entries:
                continue

            n_runs = len(entries)
            row = f"| {model} | {n_runs} |"

            for h in HORIZONS:
                mean, std, _ = compute_mean_std(entries, h, "mae")
                row += f" {format_mean_std(mean, std)} |"

            print(row)


def print_detailed_table(all_results, output_format="markdown"):
    """In bảng chi tiết tất cả metrics cho từng horizon."""
    for dataset in sorted({d for model_data in all_results.values() for d in model_data}):
        for h in HORIZONS:
            print(f"\n### {dataset} — Horizon {h} min\n")

            if output_format == "markdown":
                print("| Model | #Runs | MAE | RMSE | MAPE(%) | WMAPE(%) |")
                print("|---|---|---|---|---|---|")

            for model in sorted(all_results.keys()):
                entries = all_results[model].get(dataset, [])
                if not entries:
                    continue

                n_runs = len(entries)
                row = f"| {model} | {n_runs}"
                for metric in METRICS:
                    mean, std, _ = compute_mean_std(entries, h, metric)
                    row += f" | {format_mean_std(mean, std)}"
                row += " |"
                print(row)


def print_statistical_tests(all_results, proposed_model="AdaptiveTGCN"):
    """Kiểm định thống kê: model đề xuất vs từng baseline."""
    if not HAS_SCIPY:
        print("\n[WARNING] scipy is not installed — skipping hypothesis testing.")
        print("          Install via: pip install scipy")
        return

    print(f"\n{'='*80}")
    print(f"  STATISTICAL HYPOTHESIS TESTING: {proposed_model} vs Baselines")
    print(f"  (Paired t-test & Wilcoxon signed-rank test, p < 0.05 indicates significance)")
    print(f"{'='*80}")

    for dataset in sorted({d for model_data in all_results.values() for d in model_data}):
        proposed_entries = all_results.get(proposed_model, {}).get(dataset, [])
        if not proposed_entries:
            print(f"\n[WARNING] No results found for {proposed_model} on {dataset}")
            continue

        print(f"\n## {dataset}\n")
        print("| Baseline | Horizon | Metric | Proposed | Baseline | Diff | t-test p | Wilcoxon p | Sig? |")
        print("|---|---|---|---|---|---|---|---|---|")

        for baseline in sorted(all_results.keys()):
            if baseline == proposed_model:
                continue

            baseline_entries = all_results[baseline].get(dataset, [])
            if not baseline_entries:
                continue

            for h in ["60"]:  # Chỉ test horizon 60 phút (khó nhất)
                for metric in ["mae", "rmse"]:
                    _, _, vals_prop = compute_mean_std(proposed_entries, h, metric)
                    _, _, vals_base = compute_mean_std(baseline_entries, h, metric)

                    if len(vals_prop) < 3 or len(vals_base) < 3:
                        continue

                    mean_p = np.mean(vals_prop)
                    mean_b = np.mean(vals_base)
                    diff = mean_p - mean_b
                    diff_pct = diff / mean_b * 100

                    _, p_t = paired_test(vals_prop, vals_base, "ttest")
                    _, p_w = paired_test(vals_prop, vals_base, "wilcoxon")

                    sig = ""
                    if p_t is not None and p_t < 0.05:
                        sig = "✓" if diff < 0 else "✗"
                    if p_w is not None and p_w < 0.05:
                        sig += " (W✓)" if diff < 0 else " (W✗)"

                    p_t_str = f"{p_t:.4f}" if p_t is not None else "—"
                    p_w_str = f"{p_w:.4f}" if p_w is not None else "—"

                    print(f"| {baseline} | {h}p | {metric.upper()} | "
                          f"{mean_p:.2f} | {mean_b:.2f} | "
                          f"{diff:+.2f} ({diff_pct:+.1f}%) | "
                          f"{p_t_str} | {p_w_str} | {sig} |")


def print_per_run_detail(all_results):
    """In chi tiết từng lần chạy (cho thầy kiểm tra)."""
    print(f"\n{'='*80}")
    print(f"  DETAILED PER-RUN METRICS (Audit Log)")
    print(f"{'='*80}")

    for model in sorted(all_results.keys()):
        for dataset in sorted(all_results[model].keys()):
            entries = all_results[model][dataset]
            print(f"\n### {model} — {dataset} ({len(entries)} runs)\n")
            print("| File | Seed | Fold | MAE 60m | RMSE 60m |")
            print("|---|---|---|---|---|")

            for e in entries:
                r60 = e["results"].get("60", {})
                mae60 = r60.get("mae", 0)
                rmse60 = r60.get("rmse", 0)
                print(f"| {e['file']} | {e['seed']} | {e['fold']} | {mae60:.2f} | {rmse60:.2f} |")


def save_latex_table(all_results, output_path):
    """Xuất bảng LaTeX cho báo cáo."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Benchmark Performance (Mean $\\pm$ Std across folds and seeds)}")
    lines.append("\\label{tab:results}")
    lines.append("\\begin{tabular}{l" + "ccc" * 2 + "}")
    lines.append("\\toprule")

    header = "\\multirow{2}{*}{Model}"
    for dataset in ["PEMS04", "PEMS08"]:
        header += f" & \\multicolumn{{3}}{{c}}{{{dataset}}}"
    header += " \\\\"
    lines.append(header)

    sub_header = ""
    for _ in range(2):
        sub_header += " & MAE & RMSE & MAPE"
    sub_header += " \\\\"
    lines.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
    lines.append(sub_header)
    lines.append("\\midrule")

    for model in sorted(all_results.keys()):
        row = model
        for dataset in ["PEMS04", "PEMS08"]:
            entries = all_results[model].get(dataset, [])
            for metric in ["mae", "rmse", "mape"]:
                mean, std, _ = compute_mean_std(entries, "60", metric)
                if mean is not None:
                    if std and std > 0:
                        row += f" & ${mean:.2f} \\pm {std:.2f}$"
                    else:
                        row += f" & ${mean:.2f}$"
                else:
                    row += " & —"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nLaTeX table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis of Benchmark Results"
    )
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(ROOT, "results"))
    parser.add_argument("--proposed-model", type=str, default="AdaptiveTGCN")
    parser.add_argument("--output-format", choices=["markdown", "latex", "both"],
                        default="markdown")
    parser.add_argument("--latex-output", type=str,
                        default=os.path.join(ROOT, "results", "results_table.tex"))
    args = parser.parse_args()

    # ---- Load ----
    all_results = load_all_results(args.results_dir)

    if not all_results:
        print(f"No result files found in {args.results_dir}")
        return

    print(f"\nFound results for {len(all_results)} models:")
    expected_runs = 12  # 3 seeds × 4 folds
    incomplete_warns = []
    for model, datasets in sorted(all_results.items()):
        for dataset, entries in sorted(datasets.items()):
            n = len(entries)
            marker = ""
            if n < expected_runs:
                marker = f" ⚠️  INCOMPLETE (expected {expected_runs})"
                # Detail what's missing
                seeds_present = set(e["seed"] for e in entries)
                folds_present = set(str(e["fold"]) for e in entries)
                combos = set((e["seed"], str(e["fold"])) for e in entries)
                missing = []
                for s in sorted(seeds_present):
                    for f in ["0", "1", "2", "3"]:
                        if (s, f) not in combos:
                            missing.append(f"seed={s}/fold={f}")
                incomplete_warns.append((model, dataset, n, missing))
            print(f"  {model} × {dataset}: {n} runs{marker}")

    if incomplete_warns:
        print(f"\n{'!'*75}")
        print(f"  ⚠️  CẢNH BÁO: CÓ {len(incomplete_warns)} NHÓM (model, dataset) CHƯA ĐỦ DỮ LIỆU!")
        print(f"  Kỳ vọng mỗi nhóm cần {expected_runs} runs (3 seeds × 4 folds).")
        print(f"  Kết quả Mean ± Std của các nhóm dưới đây có thể KHÔNG ĐẠI DIỆN.")
        print(f"{'!'*75}")
        for model, dataset, n, missing in incomplete_warns:
            print(f"  • {model} × {dataset}: chỉ có {n}/{expected_runs} runs")
            if missing:
                print(f"    Thiếu: {', '.join(missing)}")
        print()

    # ---- Summary table ----
    print_summary_table(all_results, args.output_format)
    print_detailed_table(all_results, args.output_format)

    # ---- Kiểm định thống kê ----
    print_statistical_tests(all_results, args.proposed_model)

    # ---- Chi tiết từng run ----
    print_per_run_detail(all_results)

    # ---- LaTeX ----
    if args.output_format in ("latex", "both"):
        save_latex_table(all_results, args.latex_output)


if __name__ == "__main__":
    main()
