"""
Entry point cho STD-MAE baseline.
2 giai đoạn: pre-train → fine-tune.
Usage:
  python -m baselines.stdmae.run --dataset PEMS04 --epochs-pretrain 50 --epochs-finetune 50
  python -m baselines.stdmae.run --dataset PEMS04 --smoke-test
  python -m baselines.stdmae.run --dataset PEMS08 --seed 42 --fold 0
"""

import argparse
import sys
import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from baselines.common.dataloader import load_dataset
from baselines.common.metrics import masked_mae
from baselines.common.trainer import test_model, EarlyStopping
from baselines.stdmae.model import STDMAEPretrain, STDMAEFinetune


def set_seed(seed):
    """Cố định seed cho reproducibility — gọi SAU parse_args."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description="STD-MAE Baseline")
    parser.add_argument("--dataset", type=str, default="PEMS04", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--epochs-pretrain", type=int, default=50)
    parser.add_argument("--epochs-finetune", type=int, default=50)
    parser.add_argument("--lr-pretrain", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--spatial-mask-ratio", type=float, default=0.25)
    parser.add_argument("--temporal-mask-ratio", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) hoặc 'default' cho split chuẩn 60/20/20")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def pretrain(model: STDMAEPretrain, train_loader: DataLoader,
             epochs: int, lr: float, device: torch.device, max_batches: int = None):
    """Pre-training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*60}")
    print(f"[STD-MAE] Pre-training | Epochs: {epochs} | LR: {lr}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        t0 = time.time()
        for batch_idx, (X, _) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            X = X.to(device)
            optimizer.zero_grad()
            loss = model.pretrain_loss(X)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / max(total_batches, 1)
        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Pretrain Loss: {avg_loss:.4f} | {elapsed:.1f}s")

    print("[STD-MAE] Pre-training done.")


def finetune(pretrained_model: STDMAEPretrain,
             train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
             scaler, epochs: int, lr: float, patience: int, device: torch.device,
             save_dir: str = "checkpoints", max_batches: int = None):
    """Fine-tuning loop."""
    os.makedirs(save_dir, exist_ok=True)
    model = STDMAEFinetune(
        num_nodes=pretrained_model.num_nodes,
        seq_len=pretrained_model.seq_len,
        d_model=pretrained_model.d_model,
        pretrained_model=pretrained_model,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    early_stop = EarlyStopping(patience=patience)

    # Lấy scaler params để normalize y
    if isinstance(scaler, dict):
        sc_mean = torch.tensor(scaler['mean'][0], dtype=torch.float32)
        sc_std  = torch.tensor(scaler['std'][0],  dtype=torch.float32)
    else:
        sc_mean = torch.tensor(0.0)
        sc_std  = torch.tensor(1.0)

    print(f"\n{'='*60}")
    print(f"[STD-MAE] Fine-tuning | Epochs: {epochs} | LR: {lr}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_idx, (X, y) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            # Normalize y để tính loss nhất quán
            y_norm = (y - sc_mean.to(device)) / sc_std.to(device)
            loss = masked_mae(pred, y_norm)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        train_loss = total_loss / max(total_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(val_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                X, y = X.to(device), y.to(device)
                pred = model(X)
                y_norm = (y - sc_mean.to(device)) / sc_std.to(device)
                loss = masked_mae(pred, y_norm)
                val_loss += loss.item() * X.size(0)
                val_samples += X.size(0)
        val_loss /= max(val_samples, 1)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")

        if early_stop.step(val_loss, model):
            print(f"[EarlyStopping] Stopped at epoch {epoch}")
            break

    early_stop.restore_best(model)
    # Lưu checkpoint
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)

    results = test_model(model, test_loader, device, scaler)

    print(f"\n[Test Results — STD-MAE]")
    print(f"{'Horizon':>10} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>10}")
    for minutes, m in sorted(results.items()):
        print(f"{minutes:>8}min {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>10.2f}")

    return results


def main():
    args = get_args()
    set_seed(args.seed)

    fold = int(args.fold) if args.fold.isdigit() else None

    smoke_max_batches = None
    if args.smoke_test:
        args.epochs_pretrain = 2
        args.epochs_finetune = 2
        args.batch_size = 8
        smoke_max_batches = 10
        print("[SMOKE TEST MODE] epochs_pretrain=2, epochs_finetune=2, batch=8, max_batches=10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold}")

    train_loader, val_loader, test_loader, scaler, adj_matrix = load_dataset(
        args.dataset, batch_size=args.batch_size, fold=fold
    )
    num_nodes = adj_matrix.shape[0]

    # Phase 1: Pre-training
    pretrain_model = STDMAEPretrain(
        num_nodes=num_nodes,
        seq_len=12,
        in_dim=3,
        d_model=args.d_model,
        decoder_dim=args.d_model // 2,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=2,
        nhead=args.nhead,
        spatial_mask_ratio=args.spatial_mask_ratio,
        temporal_mask_ratio=args.temporal_mask_ratio,
    )
    pretrain(pretrain_model, train_loader, args.epochs_pretrain, args.lr_pretrain, device, max_batches=smoke_max_batches)

    # Phase 2: Fine-tuning
    fold_tag = f"fold{args.fold}" if args.fold != "default" else "default"
    ckpt_tag = f"stdmae_{args.dataset.lower()}_seed{args.seed}_{fold_tag}"
    save_dir = os.path.join(ROOT, "checkpoints", ckpt_tag)

    results = finetune(
        pretrain_model, train_loader, val_loader, test_loader, scaler,
        args.epochs_finetune, args.lr_finetune, args.patience, device,
        save_dir=save_dir, max_batches=smoke_max_batches,
    )

    if args.smoke_test:
        print("\n[SMOKE TEST] Pipeline check passed.")
        return results

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"stdmae_{args.dataset.lower()}_seed{args.seed}_{fold_tag}.json")
    out_data = {
        "model": "STD-MAE",
        "dataset": args.dataset,
        "config": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "nhead": args.nhead,
            "spatial_mask_ratio": args.spatial_mask_ratio,
            "temporal_mask_ratio": args.temporal_mask_ratio,
            "epochs_pretrain": args.epochs_pretrain,
            "epochs_finetune": args.epochs_finetune,
            "lr_pretrain": args.lr_pretrain,
            "lr_finetune": args.lr_finetune,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "fold": args.fold,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    main()
