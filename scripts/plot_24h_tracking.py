# -*- coding: utf-8 -*-
"""
scripts/plot_24h_tracking.py — Vẽ Biểu Đồ Theo Dõi Dự Báo Liên Tục 24 Giờ / 48 Giờ
================================================================================
Script này chạy mô hình AdaptiveTGCN dự báo trượt liên tục trên 288 bước (24 giờ)
hoặc 576 bước (48 giờ) trên tập Test của PeMS08 / PeMS04.

Mục đích:
    Chứng minh khả năng bám sát chu kỳ ngày đêm (Day-Night Traffic Cycles),
    đỉnh cao điểm sáng (Morning Peak), cao điểm chiều (Evening Rush)
    và đáy ban đêm (Night Valley) của mô hình đề xuất.

Cách chạy:
    py -3.11 scripts/plot_24h_tracking.py --sensor 0 --horizon 30
    py -3.11 scripts/plot_24h_tracking.py --sensor 15 --hours 48
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.adaptive_tgcn.model import AdaptiveTGCN


def main():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ dự báo giao thông liên tục 24h/48h")
    parser.add_argument("--dataset", type=str, default="PEMS08", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--sensor", type=int, default=0, help="ID cảm biến (mặc định: 0)")
    parser.add_argument("--hours", type=int, default=24, help="Số giờ muốn vẽ (24 hoặc 48)")
    parser.add_argument("--horizon", type=int, default=30, choices=[15, 30, 45, 60], help="Tầm dự báo trước: 15, 30, 45, hoặc 60 phút")
    parser.add_argument("--ckpt", type=str, default=None, help="File .pt checkpoint")
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "data", args.dataset, "processed")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    adj_path = os.path.join(data_dir, "adj_matrix.npy")
    scaler_path = os.path.join(data_dir, "scaler.pkl")

    if not os.path.exists(x_test_path):
        print(f"[LỖI] Không tìm thấy {x_test_path}")
        return

    print(f"\n[1] Đang nạp dữ liệu {args.dataset}...")
    X_test = np.load(x_test_path, mmap_mode="r")
    y_test = np.load(y_test_path, mmap_mode="r")
    adj_matrix = np.load(adj_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    mean_flow, std_flow = scaler["mean"][0], scaler["std"][0]

    num_samples, in_steps, num_nodes, _ = X_test.shape
    sensor_id = max(0, min(args.sensor, num_nodes - 1))
    n_steps = min(int(args.hours * 12), num_samples)  # 12 bước = 1 giờ

    # Horizon index: 15m -> idx 2, 30m -> idx 5, 45m -> idx 8, 60m -> idx 11
    h_idx = (args.horizon // 5) - 1

    # Checkpoint
    if args.ckpt is None:
        args.ckpt = os.path.join(
            ROOT, "checkpoints", f"adaptive_tgcn_fused_learnable_k5_{args.dataset.lower()}_seed42_default", "best_model.pt"
        )
        if not os.path.exists(args.ckpt):
            args.ckpt = os.path.join(
                ROOT, "checkpoints", f"adaptive_tgcn_fused_learnable_k5_{args.dataset.lower()}_seed42_fold0", "best_model.pt"
            )

    print(f"[2] Nạp mô hình AdaptiveTGCN từ: {os.path.basename(os.path.dirname(args.ckpt))}")
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    model = AdaptiveTGCN(
        num_nodes=num_nodes,
        in_dim=3,
        hidden_dim=64,
        out_dim=1,
        num_layers=2,
        embed_dim=10,
        horizon=12,
        adj_matrix=adj_tensor,
        graph_mode="fused",
        fusion_type="learnable",
        top_k=5,
        use_graph_reg=False,
    )
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[3] Đang tính toán chuỗi dự báo liên tục {args.hours} giờ ({n_steps} mốc thời gian)...")
    bx = torch.tensor(X_test[:n_steps], dtype=torch.float32)
    with torch.no_grad():
        pred_out = model(bx)
        pred_arr = pred_out[0] if isinstance(pred_out, tuple) else pred_out
        preds_denorm = pred_arr.cpu().numpy() * std_flow + mean_flow

    pred_series = preds_denorm[:, h_idx, sensor_id, 0]
    true_series = y_test[:n_steps, h_idx, sensor_id, 0]
    mae = np.mean(np.abs(pred_series - true_series))
    mape = np.mean(np.abs(pred_series - true_series) / np.maximum(true_series, 1e-3)) * 100

    print(f"\n{'='*75}")
    print(f"  KẾT QUẢ THEO DÕI LIÊN TỤC {args.hours} GIỜ — CẢM BIẾN #{sensor_id} ({args.dataset})")
    print(f"  Tầm dự báo trước: +{args.horizon} phút | MAE: {mae:.2f} xe/h | MAPE: {mape:.2f}%")
    print(f"{'='*75}\n")

    # Vẽ biểu đồ
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    out_img = os.path.join(fig_dir, f"tracking_{args.hours}h_sensor_{sensor_id}_h{args.horizon}.png")

    plt.figure(figsize=(15, 6))
    time_hours = np.arange(n_steps) * 5 / 60.0

    plt.plot(time_hours, true_series, color="#16a34a", linewidth=2.2, label="Thực tế (Ground Truth)")
    plt.plot(
        time_hours,
        pred_series,
        color="#dc2626",
        linestyle="--",
        linewidth=2.2,
        label=f"Mô hình AdaptiveTGCN (+{args.horizon} phút) [MAE: {mae:.2f} xe/h]",
    )

    plt.title(
        f"Theo Dõi Lưu Lượng Giao Thông Liên Tục {args.hours} Giờ: Cảm biến #{sensor_id} ({args.dataset}) — Horizon +{args.horizon}m",
        fontsize=13,
        pad=12,
        fontweight="bold",
    )
    plt.xlabel("Thời gian trôi qua (Giờ)", fontsize=11)
    plt.ylabel("Lưu lượng giao thông (xe / giờ)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=11, framealpha=0.95)
    plt.tight_layout()

    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"[Đã lưu biểu đồ đẹp]: file:///{out_img.replace(chr(92), '/')}\n")


if __name__ == "__main__":
    main()
