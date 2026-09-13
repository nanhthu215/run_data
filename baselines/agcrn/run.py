"""
Entry point cho AGCRN baseline.
Usage:
  python -m baselines.agcrn.run --dataset PEMS04 --epochs 50
  python -m baselines.agcrn.run --dataset PEMS04 --epochs 2 --smoke-test
  python -m baselines.agcrn.run --dataset PEMS08 --seed 42 --fold 0
"""

import argparse
import sys
import os
import json
import random
import numpy as np
import torch

# Seed sẽ được set SAU khi parse args (hỗ trợ --seed linh hoạt)

# Đảm bảo import từ project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# Thêm thư mục cha của baselines vào path
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from baselines.common.dataloader import load_dataset
from baselines.common.trainer import train
from baselines.agcrn.model import AGCRN


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
    parser = argparse.ArgumentParser(description="AGCRN Baseline")
    parser.add_argument("--dataset", type=str, default="PEMS04", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=10)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--tf-start", type=float, default=0.8, help="Scheduled sampling TF start ratio")
    parser.add_argument("--tf-end", type=float, default=0.0, help="Scheduled sampling TF end ratio")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) hoặc 'default' cho split chuẩn 60/20/20")
    parser.add_argument("--smoke-test", action="store_true", help="Chạy nhanh 2 epoch để kiểm tra pipeline")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    fold = int(args.fold) if args.fold.isdigit() else None

    smoke_max_batches = None
    if args.smoke_test:
        args.epochs = 3
        args.batch_size = 64
        smoke_max_batches = 10   # chỉ chạy 10 batch/epoch → ~30s/epoch
        print("[SMOKE TEST] epochs=3, batch=64, max_batches=10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold}")

    # Load data
    train_loader, val_loader, test_loader, scaler, adj_matrix = load_dataset(
        args.dataset, batch_size=args.batch_size, fold=fold
    )
    num_nodes = adj_matrix.shape[0]

    # Build model — KHÔNG thay đổi bất kỳ tham số kiến trúc nào
    model = AGCRN(
        num_nodes=num_nodes,
        in_dim=3,           # flow, speed, occupancy
        hidden_dim=args.hidden_dim,
        out_dim=1,          # forecast flow only
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        horizon=12,
    )

    # Checkpoint path: bao gồm seed + fold để tránh ghi đè
    fold_tag = f"fold{args.fold}" if args.fold != "default" else "default"
    ckpt_tag = f"agcrn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}"
    save_dir = os.path.join(ROOT, "checkpoints", ckpt_tag)

    # Train
    results = train(
        model, train_loader, val_loader, test_loader, scaler,
        epochs=args.epochs,
        lr=args.lr,
        tf_start=args.tf_start,
        tf_end=args.tf_end,
        patience=args.patience,
        device=device,
        model_name=f"AGCRN_{args.dataset}",
        save_dir=save_dir,
        max_batches_per_epoch=smoke_max_batches,
    )

    if args.smoke_test:
        print("\n[SMOKE TEST] Pipeline check passed.")
        return results

    # Save results — tên file bao gồm seed + fold
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"agcrn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}.json")
    out_data = {
        "model": "AGCRN",
        "dataset": args.dataset,
        "config": {
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
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
