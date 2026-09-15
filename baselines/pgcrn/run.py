# -*- coding: utf-8 -*-
"""
Entry point cho PGCRN (Patch-based GCRN) baseline — đầy đủ theo bài báo.
Paper: Rao et al., GeoInformatica 2025.

Usage:
  python -m baselines.pgcrn.run --dataset PEMS08 --epochs 100
  python -m baselines.pgcrn.run --dataset PEMS08 --seed 42 --fold 0
  python -m baselines.pgcrn.run --dataset PEMS08 --smoke-test
  python -m baselines.pgcrn.run --dataset PEMS04 --patch-len 2
"""

import argparse
import sys
import os
import random
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from baselines.common.dataloader import load_dataset
from baselines.common.metrics import masked_mae
from baselines.common.trainer import EarlyStopping, test_model, _get_scaler_params
from baselines.pgcrn.model import PGCRN


def set_seed(seed):
    """Cố định seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="PGCRN Baseline (GeoInformatica 2025)")
    parser.add_argument("--dataset", type=str, default="PEMS04", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    # ── Patch parameters (bài báo: PEMS08 dùng patch_len=2) ──
    parser.add_argument("--patch-len", type=int, default=2,
                        help="Patch length (bài báo: 2 cho PEMS04/08)")
    # ── GCN parameters ──
    parser.add_argument("--cheb-k", type=int, default=3,
                        help="Chebyshev polynomial order")
    parser.add_argument("--hyper-dim", type=int, default=10,
                        help="Dimension of hypernet node embeddings")
    # ── Dynamic Graph + Contrastive Learning ──
    parser.add_argument("--dynamic", type=int, default=1,
                        help="1=enable dynamic graph learning, 0=disable")
    parser.add_argument("--contra", type=int, default=1,
                        help="1=enable contrastive learning, 0=disable")
    parser.add_argument("--contra-weight", type=float, default=0.1,
                        help="Weight of contrastive loss")
    parser.add_argument("--glu-layers", type=int, default=2,
                        help="Number of GLU layers for spectral augmentation")
    # ── Scheduled Sampling (exponential decay) ──
    parser.add_argument("--cl-decay-steps", type=int, default=560,
                        help="CL decay steps for exponential scheduled sampling (PEMS08=560)")
    # ── Training ──
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) hoặc 'default' cho split chuẩn 60/20/20")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Chạy nhanh 2-3 epoch kiểm tra pipeline")
    return parser.parse_args()


def train_epoch_pgcrn(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler,
    contra_weight: float = 0.1,
    max_batches: int = None,
) -> float:
    """1 epoch training cho PGCRN với Dynamic Graph + Contrastive Loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    for batch_idx, (X, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        X = X.to(device)
        y = y.to(device)

        y_norm = (y - mean_t) / std_t

        optimizer.zero_grad()

        # PGCRN returns (pred, contra_loss) during training
        result = model(X, y=y_norm)
        if isinstance(result, tuple):
            pred, contra_loss = result
        else:
            pred, contra_loss = result, None

        # Prediction loss
        loss = masked_mae(pred, y_norm)

        # Add contrastive loss
        if contra_loss is not None:
            loss = loss + contra_weight * contra_loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

        if max_batches is not None:
            print(f"  [Batch {batch_idx + 1}/{max_batches}] loss: {loss.item():.4f}"
                  + (f" (contra: {contra_loss.item():.4f})" if contra_loss is not None else ""), flush=True)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def eval_epoch_pgcrn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    max_batches: int = None,
) -> float:
    """1 epoch validation cho PGCRN (không dùng teacher forcing)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    for batch_idx, (X, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        X = X.to(device)
        y = y.to(device)

        y_norm = (y - mean_t) / std_t

        # Eval mode: model returns just pred (no contra_loss)
        pred = model(X)
        loss = masked_mae(pred, y_norm)
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / max(total_samples, 1)


def train_pgcrn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    scaler,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    contra_weight: float = 0.1,
    patience: int = 20,
    device: torch.device = torch.device("cpu"),
    model_name: str = "PGCRN",
    save_dir: str = "checkpoints",
    max_batches_per_epoch: int = None,
) -> dict:
    """Vòng lặp huấn luyện đầy đủ cho PGCRN."""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    early_stop = EarlyStopping(patience=patience)

    total_batches = len(train_loader)
    effective_batches = min(max_batches_per_epoch, total_batches) if max_batches_per_epoch else total_batches

    print(f"\n{'='*70}")
    print(f"Training: {model_name} | Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"Scheduled Sampling: exponential decay (cl_decay_steps={model.cl_decay_steps})")
    print(f"Dynamic Graph: {model.dynamic} | Contrastive: {model.contra} (weight={contra_weight})")
    print(f"Patch len: {model.patch_len} | Chebyshev K: {model.cheb_k}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if max_batches_per_epoch:
        print(f"[SMOKE] Max {effective_batches}/{total_batches} batches/epoch")
    print(f"{'='*70}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        train_loss = train_epoch_pgcrn(
            model, train_loader, optimizer, device, scaler,
            contra_weight=contra_weight,
            max_batches=max_batches_per_epoch,
        )
        val_loss = eval_epoch_pgcrn(
            model, val_loader, device, scaler,
            max_batches=max_batches_per_epoch,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t_ep
        if epoch % 5 == 0 or epoch == 1:
            tf_ratio = model.compute_sampling_threshold()
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"TF_ratio: {tf_ratio:.3f} | "
                f"Train MAE: {train_loss:.4f} | "
                f"Val MAE: {val_loss:.4f} | "
                f"batches_seen: {model.batches_seen} | "
                f"{elapsed:.1f}s"
            )

        if early_stop.step(val_loss, model):
            print(f"[EarlyStopping] Stopped at epoch {epoch} | Best Val MAE: {early_stop.best_loss:.4f}")
            break

    early_stop.restore_best(model)
    total_time = time.time() - t0
    print(f"\nTraining done in {total_time/60:.1f} min. Best Val MAE: {early_stop.best_loss:.4f}")

    # Lưu checkpoint
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Đánh giá trên tập test
    results = test_model(model, test_loader, device, scaler)

    # Lưu train_summary.json
    rel_save_dir = os.path.relpath(save_dir, ROOT).replace("\\", "/")
    summary = {
        "model_name": model_name,
        "checkpoint_dir": rel_save_dir,
        "checkpoint_file": f"{rel_save_dir}/best_model.pt",
        "best_val_mae": float(early_stop.best_loss),
        "epochs_trained": epoch,
        "training_time_seconds": round(total_time, 2),
        "results": results,
    }
    summary_path = os.path.join(save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Test Results — {model_name}]")
    print(f"{'Horizon':>10} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>10} {'WMAPE(%)':>10}")
    print(f"{'-'*52}")
    for minutes, m in sorted(results.items()):
        wmape_val = f"{m.get('wmape', 0.0):>10.2f}"
        print(f"{minutes:>8}min {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>10.2f} {wmape_val}")

    return results


def main():
    args = get_args()
    set_seed(args.seed)

    fold = int(args.fold) if args.fold.isdigit() else None

    smoke_max_batches = None
    if args.smoke_test:
        args.epochs = 2
        smoke_max_batches = 3
        print("[SMOKE TEST] epochs=2, batch=64, max_batches=3")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold}")

    # Load dữ liệu
    train_loader, val_loader, test_loader, scaler, adj_matrix = load_dataset(
        args.dataset, batch_size=args.batch_size, fold=fold
    )
    num_nodes = adj_matrix.shape[0]

    # Khởi tạo mô hình PGCRN đầy đủ
    model = PGCRN(
        num_nodes=num_nodes,
        adj_matrix=adj_matrix,
        in_dim=3,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        horizon=12,
        patch_len=args.patch_len,
        cheb_k=args.cheb_k,
        hyper_dim=args.hyper_dim,
        dynamic=bool(args.dynamic),
        contra=bool(args.contra),
        glu_layers=args.glu_layers,
        cl_decay_steps=args.cl_decay_steps,
    )

    # Checkpoint path
    fold_tag = f"fold{args.fold}" if args.fold != "default" else "default"
    ckpt_tag = f"pgcrn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}"
    save_dir = os.path.join(ROOT, "checkpoints", ckpt_tag)

    results = train_pgcrn(
        model, train_loader, val_loader, test_loader, scaler,
        epochs=args.epochs,
        lr=args.lr,
        contra_weight=args.contra_weight,
        patience=args.patience,
        device=device,
        model_name=f"PGCRN_{args.dataset}",
        save_dir=save_dir,
        max_batches_per_epoch=smoke_max_batches,
    )

    if args.smoke_test:
        print("\n[SMOKE TEST] Pipeline check passed.")
        return results

    # Lưu kết quả
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"pgcrn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}.json")
    out_data = {
        "model": "PGCRN",
        "dataset": args.dataset,
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "patch_len": args.patch_len,
            "cheb_k": args.cheb_k,
            "hyper_dim": args.hyper_dim,
            "dynamic": args.dynamic,
            "contra": args.contra,
            "contra_weight": args.contra_weight,
            "glu_layers": args.glu_layers,
            "cl_decay_steps": args.cl_decay_steps,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "fold": args.fold,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    main()
