"""
Entry point cho Adaptive T-GCN (Physical-Adaptive Graph Fusion).

Usage:
  # Ablation: chỉ đồ thị vật lý
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode physical

  # Ablation: chỉ đồ thị thích nghi
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode adaptive

  # Fusion cố định alpha=0.5
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode fused --fusion-type fixed --alpha 0.5

  # Fusion learnable alpha
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode fused --fusion-type learnable

  # + Top-k sparsification
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode fused --fusion-type learnable --top-k 10

  # + Graph Regularization
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode fused --fusion-type learnable --top-k 10 --use-graph-reg

  # Smoke test (pipeline check, 3 epoch, 10 batch)
  python -m models.adaptive_tgcn.run --dataset PEMS08 --graph-mode fused --smoke-test
"""

import argparse
import sys
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import inspect
import time

# ----- Seed sẽ được set SAU khi parse args (để hỗ trợ --seed tùy chỉnh) -----
# (Không cố định SEED=42 ở đây nữa)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from baselines.common.dataloader import load_dataset
from baselines.common.metrics    import masked_mae
from baselines.common.trainer    import (
    EarlyStopping, eval_epoch, test_model,
    get_teacher_forcing_ratio, _get_scaler_params,
)
from models.adaptive_tgcn.model import AdaptiveTGCN


# =========================================================================
# Training loop tuỳ chỉnh — hỗ trợ graph_reg_loss từ model.forward()
# =========================================================================

def train_epoch_with_reg(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    teacher_forcing_ratio: float = 0.5,
    max_batches: int = None,
) -> float:
    """
    Training loop 1 epoch, cộng graph_reg_loss vào total loss nếu model trả về tuple.
    Hoàn toàn tương thích với trainer.py hiện có (eval_epoch, test_model không thay đổi).
    """
    model.train()
    total_loss   = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler) if scaler else (0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    for batch_idx, (X, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        X = X.to(device)
        y = y.to(device)

        y_norm = (y - mean_t) / std_t

        optimizer.zero_grad()
        output = model(X, y=y_norm, teacher_forcing_ratio=teacher_forcing_ratio)

        # model.forward() trả về (pred, graph_reg_loss)
        if isinstance(output, tuple):
            pred, reg_loss = output
        else:
            pred, reg_loss = output, torch.tensor(0.0, device=device)

        if pred.dim() == 3:
            pred = pred.unsqueeze(-1)

        loss = masked_mae(pred, y_norm) + reg_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss    += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / max(total_samples, 1)


def eval_epoch_adaptive(model, loader, device, scaler, max_batches=None):
    """
    Đánh giá 1 epoch — tương thích với model trả về tuple (pred, reg_loss).
    Teacher forcing tắt hoàn toàn khi eval.
    """
    model.eval()
    total_loss   = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler) if scaler else (0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            X = X.to(device)
            y = y.to(device)

            output = model(X)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output

            if pred.dim() == 3:
                pred = pred.unsqueeze(-1)

            y_norm = (y - mean_t) / std_t
            loss = masked_mae(pred, y_norm)
            total_loss    += loss.item() * X.size(0)
            total_samples += X.size(0)

    return total_loss / max(total_samples, 1)


def test_model_adaptive(model, loader, device, scaler, horizons=(3, 6, 12), max_batches: int = None):
    """
    Test — tương thích với model trả về tuple (pred, reg_loss).
    """
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            X = X.to(device)
            output = model(X)
            pred = output[0] if isinstance(output, tuple) else output
            if pred.dim() == 3:
                pred = pred.unsqueeze(-1)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.numpy())

    import numpy as np
    pred_arr = np.concatenate(all_pred, axis=0)
    true_arr = np.concatenate(all_true, axis=0)

    if isinstance(scaler, dict):
        mean, std = scaler['mean'][0], scaler['std'][0]
    elif hasattr(scaler, 'mean_'):
        mean, std = scaler.mean_[0], scaler.scale_[0]
    else:
        mean, std = 0.0, 1.0

    pred_inv = pred_arr * std + mean
    true_inv = true_arr

    from baselines.common.metrics import compute_horizon_metrics
    return compute_horizon_metrics(pred_inv, true_inv, list(horizons))


# =========================================================================
# Argument parser
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Adaptive T-GCN — Physical-Adaptive Graph Fusion")
    # Dataset
    parser.add_argument("--dataset",    type=str,   default="PEMS08", choices=["PEMS04", "PEMS08"])
    # Graph mode
    parser.add_argument("--graph-mode", type=str,   default="fused",
                        choices=["physical", "adaptive", "fused"])
    parser.add_argument("--fusion-type",type=str,   default="fixed",
                        choices=["fixed", "learnable"])
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Hệ số hợp nhất cố định (chỉ khi fusion-type=fixed)")
    parser.add_argument("--top-k",      type=int,   default=0,
                        help="Giữ top-k cạnh mỗi hàng (0=tắt)")
    parser.add_argument("--use-graph-reg", action="store_true",
                        help="Bật graph regularization loss")
    parser.add_argument("--graph-reg-weight", type=float, default=1e-4,
                        help="Trọng số của graph_reg_loss")
    # Bước 2a: Temperature scaling cho softmax của A_adapt
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature cho softmax của A_adapt (< 1.0 = sắc nét hơn, mặc định 1.0)")
    # Model architecture
    parser.add_argument("--hidden-dim", type=int,   default=64)
    # Bước 2b: embed_dim cho adaptive graph (mặc định 10)
    parser.add_argument("--embed-dim",  type=int,   default=10,
                        help="Embed dim của E1, E2 (mặc định 10)")
    parser.add_argument("--num-layers", type=int,   default=2)
    parser.add_argument("--gcn-depth",  type=int,   default=1, choices=[1, 2],
                        help="Độ sâu tích chập đồ thị: 1 (1-hop thuần T-GCN, mặc định) hoặc 2 (2-hop diffusion)")
    parser.add_argument("--predictor-type", type=str, default="autoregressive", choices=["direct", "autoregressive"],
                        help="Loại đầu ra: 'direct' (one-shot end_conv) hoặc 'autoregressive' (tuần tự qua decoder, mặc định)")
    # Training
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--tf-start",   type=float, default=0.8)
    parser.add_argument("--tf-end",     type=float, default=0.0)
    # Bước 4: --seed linh hoạt thay vì cố định SEED=42
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed (mặc định 42, thử 100/2024 cho đa seed eval)")
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) hoặc 'default' cho split chuẩn 60/20/20")
    # Misc
    parser.add_argument("--smoke-test", action="store_true",
                        help="Chạy nhanh 3 epoch, 10 batch để kiểm tra pipeline")
    return parser.parse_args()


# =========================================================================
# Main
# =========================================================================

def main():
    args       = get_args()

    # ----- Bước 4: Set seed SAU khi parse args (hỗ trợ --seed linh hoạt) -----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[Seed] {args.seed}")

    fold = int(args.fold) if args.fold.isdigit() else None

    smoke_max  = None
    if args.smoke_test:
        args.epochs = 3
        smoke_max   = 10
        print("[SMOKE TEST] epochs=3, max_batches=10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Fold: {args.fold}")

    # ----- Load data -----
    train_loader, val_loader, test_loader, scaler, adj_matrix = load_dataset(
        args.dataset, batch_size=args.batch_size, fold=fold
    )
    num_nodes = adj_matrix.shape[0]

    # Chuyển adj_matrix thành Tensor
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

    # ----- Xây dựng model -----
    # physical/fused cần adj_matrix; adaptive không cần
    adj_for_model = adj_tensor if args.graph_mode in ("physical", "fused") else None

    model = AdaptiveTGCN(
        num_nodes       = num_nodes,
        in_dim          = 3,
        hidden_dim      = args.hidden_dim,
        out_dim         = 1,
        num_layers      = args.num_layers,
        embed_dim       = args.embed_dim,
        horizon         = 12,
        adj_matrix      = adj_for_model,
        graph_mode      = args.graph_mode,
        fusion_type     = args.fusion_type,
        alpha           = args.alpha,
        top_k           = args.top_k,
        use_graph_reg   = args.use_graph_reg,
        graph_reg_weight= args.graph_reg_weight,
        temperature     = args.temperature,
        gcn_depth       = args.gcn_depth,
        predictor_type  = args.predictor_type,
    )

    # ----- Tên model cho log và checkpoint -----
    mode_tag = args.graph_mode
    if args.graph_mode == "fused":
        mode_tag += f"_{args.fusion_type}"
        if args.fusion_type == "fixed":
            mode_tag += f"_a{args.alpha}"
    if args.top_k > 0:
        mode_tag += f"_k{args.top_k}"
    if args.use_graph_reg:
        mode_tag += "_reg"
    if args.gcn_depth != 1:
        mode_tag += f"_hop{args.gcn_depth}"
    if args.predictor_type != "autoregressive":
        mode_tag += f"_{args.predictor_type}"
    if args.temperature != 1.0:
        mode_tag += f"_t{args.temperature}"
    if args.embed_dim != 10:
        mode_tag += f"_d{args.embed_dim}"

    fold_tag = f"fold{args.fold}" if args.fold != "default" else "default"
    model_name = f"AdaptiveTGCN_{mode_tag}_{args.dataset}"

    # ----- Training -----
    save_dir = os.path.join(ROOT, "checkpoints", f"adaptive_tgcn_{mode_tag}_{args.dataset.lower()}_seed{args.seed}_{fold_tag}")
    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    early_stop = EarlyStopping(patience=args.patience)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Epochs: {args.epochs} | LR: {args.lr} | Device: {device}")
    print(f"Graph mode: {args.graph_mode} | Fusion: {args.fusion_type} | "
          f"alpha={args.alpha} | top_k={args.top_k} | graph_reg={args.use_graph_reg} | "
          f"gcn_depth={args.gcn_depth} | predictor={args.predictor_type} | "
          f"temperature={args.temperature} | embed_dim={args.embed_dim} | seed={args.seed}")
    if args.predictor_type == "autoregressive":
        print(f"Scheduled Sampling: TF_start={args.tf_start:.2f} -> TF_end={args.tf_end:.2f}")
    else:
        print(f"Predictor: Direct Multi-Horizon Conv (Non-Autoregressive, Zero Exposure Bias)")
    print(f"Params: {total_params:,}")
    print(f"{'='*60}")

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep    = time.time()
        tf_ratio = get_teacher_forcing_ratio(epoch, args.epochs, args.tf_start, args.tf_end)

        train_loss = train_epoch_with_reg(
            model, train_loader, optimizer, device, scaler,
            teacher_forcing_ratio=tf_ratio,
            max_batches=smoke_max,
        )
        val_loss = eval_epoch_adaptive(model, val_loader, device, scaler, smoke_max)
        scheduler.step(val_loss)

        elapsed = time.time() - t_ep
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | TF_ratio: {tf_ratio:.2f} | "
                  f"Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | {elapsed:.1f}s", flush=True)

        if early_stop.step(val_loss, model):
            print(f"[EarlyStopping] Stopped at epoch {epoch} | Best Val MAE: {early_stop.best_loss:.4f}")
            break

    early_stop.restore_best(model)
    total_time = time.time() - t0
    print(f"\nTraining done in {total_time/60:.1f} min. Best Val MAE: {early_stop.best_loss:.4f}")

    # ----- Test -----
    results = test_model_adaptive(model, test_loader, device, scaler, max_batches=smoke_max)

    # In bảng kết quả
    print(f"\n[Test Results — {model_name}]")
    print(f"{'Horizon':>10} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>10} {'WMAPE(%)':>10}")
    print(f"{'-'*52}")
    for minutes, m in sorted(results.items()):
        print(f"{minutes:>8}min {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>10.2f} {m.get('wmape', 0.0):>10.2f}")

    # Nếu là smoke-test thì KHÔNG lưu checkpoint hay file kết quả (tránh làm hỏng số liệu thật)
    if args.smoke_test:
        print("\n[SMOKE TEST] Pipeline check passed.")
        return results

    # Lưu checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    # ----- Lưu kết quả JSON (tên file phân biệt rõ từng config ablation) -----
    results_dir  = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_fname    = f"adaptive_tgcn_{mode_tag}_{args.dataset.lower()}_seed{args.seed}_{fold_tag}.json"
    out_path     = os.path.join(results_dir, out_fname)

    # Ghi đầy đủ config vào JSON để dễ tra cứu khi so sánh ablation
    config_record = {
        "graph_mode"      : args.graph_mode,
        "fusion_type"     : args.fusion_type,
        "alpha"           : args.alpha,
        "top_k"           : args.top_k,
        "use_graph_reg"   : args.use_graph_reg,
        "graph_reg_weight": args.graph_reg_weight,
        "temperature"     : args.temperature,
        "gcn_depth"       : args.gcn_depth,
        "predictor_type"  : args.predictor_type,
        "hidden_dim"      : args.hidden_dim,
        "embed_dim"       : args.embed_dim,
        "num_layers"      : args.num_layers,
        "epochs"          : args.epochs,
        "lr"              : args.lr,
        "batch_size"      : args.batch_size,
        "seed"            : args.seed,
        "fold"            : args.fold,
    }
    out_data = {
        "model"  : "AdaptiveTGCN",
        "dataset": args.dataset,
        "config" : config_record,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # Đồng bộ lưu train_summary.json vào thư mục checkpoint
    summary_path = os.path.join(save_dir, "train_summary.json")
    summary_data = {
        "model_name": model_name,
        "checkpoint_dir": os.path.relpath(save_dir, ROOT).replace("\\", "/"),
        "checkpoint_file": os.path.relpath(os.path.join(save_dir, "best_model.pt"), ROOT).replace("\\", "/"),
        "best_val_mae": float(early_stop.best_loss),
        "epochs_trained": epoch,
        "training_time_seconds": round(total_time, 2),
        "config": config_record,
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Train summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    main()
