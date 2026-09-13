"""
Entry point cho STFGNN baseline.
Usage:
  python -m baselines.stfgnn.run --dataset PEMS04 --epochs 50
  python -m baselines.stfgnn.run --dataset PEMS04 --smoke-test
  python -m baselines.stfgnn.run --dataset PEMS08 --seed 42 --fold 0
"""

import argparse
import sys
import os
import json
import random
import numpy as np
import torch

# Seed sẽ được set SAU khi parse args (hỗ trợ --seed linh hoạt)

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from baselines.common.dataloader import load_dataset
from baselines.common.trainer import train
from baselines.stfgnn.model import STFGNN
from scripts.generate_dtw_graph import get_or_generate_dtw_graph


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
    parser = argparse.ArgumentParser(description="STFGNN Baseline")
    parser.add_argument("--dataset", type=str, default="PEMS04", choices=["PEMS04", "PEMS08"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10, help="Top-k temporal neighbors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=str, default="default",
                        help="Fold index (0,1,2,3) hoặc 'default' cho split chuẩn 60/20/20")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    fold = int(args.fold) if args.fold.isdigit() else None

    smoke_max = None
    if args.smoke_test:
        args.epochs = 2
        smoke_max = 10
        print("[SMOKE TEST MODE] epochs=2, max_batches=10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed} | Fold: {args.fold}")

    train_loader, val_loader, test_loader, scaler, adj_matrix = load_dataset(
        args.dataset, batch_size=args.batch_size, fold=fold
    )
    num_nodes = adj_matrix.shape[0]

    # Nạp ma trận DTW (tự động load từ cache hoặc tính toán nếu chưa có)
    adj_temporal = get_or_generate_dtw_graph(args.dataset, top_k=args.top_k)
    print(f"Loaded DTW temporal graph: {adj_temporal.shape}, {adj_temporal.sum():.0f} edges")

    # Build model — KHÔNG thay đổi bất kỳ tham số kiến trúc nào
    model = STFGNN(
        num_nodes=num_nodes,
        adj_spatial=adj_matrix,
        adj_temporal=adj_temporal,
        in_dim=3,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        num_layers=args.num_layers,
        horizon=12,
        seq_len=12,
    )

    # Checkpoint path: bao gồm seed + fold
    fold_tag = f"fold{args.fold}" if args.fold != "default" else "default"
    ckpt_tag = f"stfgnn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}"
    save_dir = os.path.join(ROOT, "checkpoints", ckpt_tag)

    results = train(
        model, train_loader, val_loader, test_loader, scaler,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        model_name=f"STFGNN_{args.dataset}",
        save_dir=save_dir,
        max_batches_per_epoch=smoke_max,
    )

    if args.smoke_test:
        print("\n[SMOKE TEST] Pipeline check passed.")
        return results

    # Save results
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"stfgnn_{args.dataset.lower()}_seed{args.seed}_{fold_tag}.json")
    out_data = {
        "model": "STFGNN",
        "dataset": args.dataset,
        "config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "top_k": args.top_k,
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
    print(f"Results saved to: {out_path}")
    return results


if __name__ == "__main__":
    main()
