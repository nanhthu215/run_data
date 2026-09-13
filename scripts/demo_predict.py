# -*- coding: utf-8 -*-
"""
scripts/demo_predict.py — Trực quan hóa và Kiểm chứng Dự báo Thực tế
===================================================================
Script này lấy 1 đoạn dữ liệu quá khứ thực tế (12 bước = 60 phút),
đưa qua mô hình AdaptiveTGCN đã huấn luyện, xuất ra các con số DỰ BÁO LƯU LƯỢNG
trong 60 phút tới (15m, 30m, 45m, 60m) và so sánh trực tiếp với thực tế diễn ra.

Cách chạy:
    py -3.11 scripts/demo_predict.py --sensor 10 --sample-idx 50
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.adaptive_tgcn.model import AdaptiveTGCN


def main():
    parser = argparse.ArgumentParser(description="Demo dự báo lưu lượng giao thông thực tế")
    parser.add_argument("--dataset", type=str, default="PEMS08", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--sensor", type=int, default=20, help="ID cảm biến muốn xem (0 đến 169 với PeMS08)")
    parser.add_argument("--sample-idx", type=int, default=194, help="Vị trí mẫu thời gian trong tập test (gợi ý mẫu dốc giờ cao điểm: 194, 221, 494)")
    parser.add_argument("--ckpt", type=str, default=None, help="Đường dẫn file best_model.pt")
    parser.add_argument("--plot", action="store_true", default=True, help="Vẽ đồ thị lưu lượng thực tế vs dự báo")
    parser.add_argument("--eval-all", action="store_true", default=False, help="Chạy đánh giá kiểm chứng trên toàn bộ tập test (ra MAE 24.87)")
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "data", args.dataset, "processed")
    x_test_path = os.path.join(data_dir, "X_test.npy")
    y_test_path = os.path.join(data_dir, "y_test.npy")
    adj_path = os.path.join(data_dir, "adj_matrix.npy")
    scaler_path = os.path.join(data_dir, "scaler.pkl")

    if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
        print(f"[LỖI] Không tìm thấy dữ liệu test tại {data_dir}. Vui lòng chạy tiền xử lý trước!")
        return

    # Load dữ liệu test
    print(f"\n[1] Đang nạp dữ liệu tập Test ({args.dataset})...")
    X_test = np.load(x_test_path, mmap_mode="r")  # shape: (N_samples, 12, N_nodes, 3)
    y_test = np.load(y_test_path, mmap_mode="r")  # shape: (N_samples, 12, N_nodes, 1)
    adj_matrix = np.load(adj_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    if isinstance(scaler, dict):
        mean_flow, std_flow = scaler["mean"][0], scaler["std"][0]
    else:
        mean_flow, std_flow = scaler.mean_[0], scaler.scale_[0]

    num_samples, in_steps, num_nodes, in_feats = X_test.shape
    sensor_id = max(0, min(args.sensor, num_nodes - 1))
    sample_idx = max(0, min(args.sample_idx, num_samples - 1))

    # Tìm checkpoint tốt nhất
    if args.ckpt is None:
        ckpt_candidate = os.path.join(
            ROOT, "checkpoints", f"adaptive_tgcn_fused_learnable_k5_{args.dataset.lower()}_seed42_default", "best_model.pt"
        )
        if not os.path.exists(ckpt_candidate):
            # Fallback sang  0
            ckpt_candidate = os.path.join(
                ROOT, "checkpoints", f"adaptive_tgcn_fused_learnable_k5_{args.dataset.lower()}_seed42_fold0", "best_model.pt"
            )
        args.ckpt = ckpt_candidate

    print(f"[2] Đang nạp mô hình đề xuất AdaptiveTGCN từ: {os.path.basename(os.path.dirname(args.ckpt))}")
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

    # Nếu người dùng muốn kiểm chứng toàn bộ tập Test (toàn bộ 3.568 mẫu)
    if args.eval_all:
        print(f"\n[3] Đang đánh giá trên TOÀN BỘ tập Test ({num_samples:,} mẫu x {num_nodes} nút x 12 bước)...")
        batch_size = 64
        all_preds = []
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                bx = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32)
                out = model(bx)
                pred = out[0] if isinstance(out, tuple) else out
                if pred.dim() == 3:
                    pred = pred.unsqueeze(-1)
                all_preds.append(pred.cpu().numpy())

        pred_arr = np.concatenate(all_preds, axis=0)  # (num_samples, 12, N, 1)
        pred_flow_all = pred_arr * std_flow + mean_flow
        true_flow_all = y_test  # (num_samples, 12, N, 1)

        from baselines.common.metrics import compute_horizon_metrics
        test_metrics = compute_horizon_metrics(pred_flow_all, true_flow_all, [3, 6, 12])

        print(f"\n{'='*75}")
        print(f"  KẾT QUẢ KIỂM CHỨNG TOÀN BỘ TẬP TEST ({args.dataset} — {num_samples:,} mẫu)")
        print(f"  Checkpoint: {os.path.basename(os.path.dirname(args.ckpt))}")
        print(f"{'='*75}")
        print(f"{'Horizon':<16} {'MAE':<12} {'RMSE':<12} {'MAPE (%)':<12} {'WMAPE (%)':<12}")
        print(f"{'-'*70}")
        for step, m in sorted(test_metrics.items()):
            mins = step
            print(f"{mins:2d} phút (bước {step//5:2d})   {m['mae']:<12.4f} {m['rmse']:<12.4f} {m['mape']:<12.2f} {m.get('wmape', 0.0):<12.2f}")
        print(f"{'='*75}\n")
        return

    # Lấy 1 mẫu dữ liệu cụ thể (12 bước quá khứ)
    # Shape: (1, 12, N, 3)
    sample_x = torch.tensor(X_test[sample_idx:sample_idx+1], dtype=torch.float32)
    # Ground truth tương lai (12 bước sau đó)
    ground_truth_y = y_test[sample_idx, :, sensor_id, 0]  # shape (12,)
    # Quá khứ 12 bước của cảm biến này (chuyển về giá trị thực xe/h)
    history_x = X_test[sample_idx, :, sensor_id, 0] * std_flow + mean_flow

    # TIẾN HÀNH DỰ BÁO
    print("[3] Đang thực hiện DỰ BÁO LƯU LƯỢNG tương lai...")
    with torch.no_grad():
        output = model(sample_x)
        pred = output[0] if isinstance(output, tuple) else output
        # pred shape: (1, 12, N, 1) -> chuyển về numpy và giải chuẩn hóa (denormalize)
        pred_norm = pred[0, :, sensor_id, 0].cpu().numpy()
        pred_flow = pred_norm * std_flow + mean_flow

    # IN KẾT QUẢ DỰ BÁO CHI TIẾT
    print(f"\n{'='*75}")
    print(f"   KẾT QUẢ DỰ BÁO LƯU LƯỢNG TẠI CẢM BIẾN #{sensor_id} ({args.dataset})")
    print(f"   Mẫu kiểm thử số: {sample_idx} / {num_samples}")
    print(f"{'='*75}")

    print(f"\n[Dữ liệu Quá khứ — 60 phút vừa qua (12 bước x 5 phút)]:")
    hist_str = " -> ".join([f"{val:.1f}" for val in history_x])
    print(f"  {hist_str} (xe/giờ)\n")

    print(f"{'Thời gian tới':<16} {'Dự báo (Pred)':<18} {'Thực tế (Real)':<18} {'Sai số tuyệt đối':<18}")
    print(f"{'-'*70}")

    for step in range(12):
        minutes = (step + 1) * 5
        pred_val = pred_flow[step]
        real_val = ground_truth_y[step]
        abs_err = abs(real_val - pred_val)
        rel_err = (abs_err / max(real_val, 1e-3)) * 100

        highlight = " <--" if minutes in [15, 30, 45, 60] else ""
        print(f"+{minutes:2d} phút ({step+1:2d}/12)    {pred_val:8.1f} xe/h       {real_val:8.1f} xe/h       {abs_err:6.1f} xe/h ({rel_err:4.1f}%){highlight}")

    print(f"{'-'*70}")
    mae_sample = np.mean(np.abs(pred_flow - ground_truth_y))
    print(f"==> MAE trung bình của lần dự báo này: {mae_sample:.2f} xe/giờ")
    print(f"{'='*75}\n")

    # Vẽ biểu đồ nếu yêu cầu
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig_dir = os.path.join(ROOT, "results", "figures")
            os.makedirs(fig_dir, exist_ok=True)
            out_img = os.path.join(fig_dir, f"forecast_demo_sensor_{sensor_id}_sample_{sample_idx}.png")

            plt.figure(figsize=(12, 6))
            # Trục thời gian: -55m đến 0m (quá khứ) và +5m đến +60m (tương lai)
            t_past = np.arange(-55, 5, 5)
            t_future = np.arange(5, 65, 5)

            # Vẽ quá khứ
            plt.plot(t_past, history_x, "o-", color="#2563eb", linewidth=2.5, label="Quá khứ quan sát được (1 giờ trước)")
            # Vẽ thực tế tương lai
            plt.plot(t_future, ground_truth_y, "s-", color="#16a34a", linewidth=2.5, label="Thực tế diễn ra (Ground Truth)")
            # Vẽ dự báo của mô hình
            plt.plot(t_future, pred_flow, "^--", color="#dc2626", linewidth=2.5, label="Mô hình AdaptiveTGCN Dự báo")

            # Đường phân cách hiện tại (t = 0)
            plt.axvline(0, color="gray", linestyle=":", linewidth=1.5, label="Thời điểm hiện tại (T_now)")

            plt.title(f"Dự báo Lưu lượng Giao thông: Cảm biến #{sensor_id} (PeMS08) — MAE: {mae_sample:.2f} xe/h", fontsize=13, pad=12)
            plt.xlabel("Mốc thời gian (phút)", fontsize=11)
            plt.ylabel("Lưu lượng giao thông (xe / giờ)", fontsize=11)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend(loc="best", fontsize=11)
            plt.tight_layout()

            plt.savefig(out_img, dpi=200)
            plt.close()
            print(f"[Đã vẽ biểu đồ dự báo]: file:///{out_img.replace(chr(92), '/')}")
            print("\n[Mẹo demo thuyết trình bảo vệ khóa luận]:")
            print("  * Xem mẫu leo dốc giờ cao điểm (sáng):  py -3.11 scripts/demo_predict.py --sensor 20 --sample-idx 194")
            print("  * Xem mẫu leo dốc trưa:                 py -3.11 scripts/demo_predict.py --sensor 15 --sample-idx 221")
            print("  * Xem biểu đồ chu kỳ 24H liên tục:      py -3.11 scripts/plot_24h_tracking.py --sensor 0 --horizon 30")
        except Exception as e:
            print(f"[Cảnh báo vẽ hình]: Không thể tạo file ảnh đồ thị ({e})")


if __name__ == "__main__":
    main()
