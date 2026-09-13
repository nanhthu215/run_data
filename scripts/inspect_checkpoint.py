"""
inspect_checkpoint.py — Công Cụ Kiểm Tra và Đọc File Checkpoint PyTorch (.pt)
====================================================================================
Hỗ trợ xem chi tiết tất cả các thành phần trong file checkpoint, bao gồm:
  - Thông tin file (.pt)
  - Danh sách các tham số (tên layer, shape, số lượng tham số)
  - Tổng số tham số toàn mô hình
  - Insight đặc biệt cho kiến trúc Adaptive T-GCN:
      • Trọng số alpha học được (learned alpha)
      • Hiệu quả fusion (physical + adaptive)
      • Kích thước embedding

Usage:
    # Kiểm tra checkpoint mới nhất trong thư mục checkpoints/
    python scripts/inspect_checkpoint.py

    # Kiểm tra checkpoint cụ thể
    python scripts/inspect_checkpoint.py <path/to/best_model.pt>

Ví dụ:
    # Kiểm tra T-GCN-PA (k=5) trên PEMS08
    python scripts/inspect_checkpoint.py checkpoints/adaptive_tgcn_fused_learnable_k5_pems08_seed42_default/best_model.pt
"""

import sys
import os
import argparse
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def find_latest_checkpoint(ckpt_dir):
    """Find the most recently modified best_model.pt in checkpoints directory."""
    if not os.path.exists(ckpt_dir):
        return None
    candidates = []
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt") or f.endswith(".pth"):
                full_path = os.path.join(root, f)
                candidates.append((os.path.getmtime(full_path), full_path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def inspect_checkpoint(ckpt_path, show_values=False):
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        return

    print(f"\n{'='*75}")
    print(f"  CHECKPOINT INSPECTION: {os.path.basename(ckpt_path)}")
    print(f"  File path: {ckpt_path}")
    file_size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"  File size: {file_size_kb:.1f} KB")
    print(f"{'='*75}\n")

    try:
        data = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return

    # Check if data is state_dict or full dict with metadata
    if isinstance(data, dict):
        if "state_dict" in data:
            state_dict = data["state_dict"]
            extra_keys = [k for k in data.keys() if k != "state_dict"]
            print(f"  Metadata fields: {extra_keys}")
        else:
            state_dict = data
    else:
        print(f"[INFO] Loaded object type: {type(data)}")
        return

    total_params = 0
    print(f"{'Layer Name':<45} {'Shape':<20} {'Params':>10}")
    print(f"{'-'*45} {'-'*20} {'-'*10}")

    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            n_elem = tensor.numel()
            total_params += n_elem
            shape_str = str(tuple(tensor.shape))
            print(f"{name:<45} {shape_str:<20} {n_elem:>10,d}")
        else:
            print(f"{name:<45} {str(type(tensor)):<20} {'N/A':>10}")

    print(f"{'-'*77}")
    print(f"{'TOTAL PARAMETERS':<66} {total_params:>10,d}\n")

    # Specific insights for AdaptiveTGCN
    if "graph_module._alpha_param" in state_dict:
        raw_alpha = state_dict["graph_module._alpha_param"].item()
        learned_alpha = torch.sigmoid(torch.tensor(raw_alpha)).item()
        print(f"  [AdaptiveTGCN Insights]")
        print(f"  - Learned Alpha Param: {raw_alpha:.4f}")
        print(f"  - Effective Alpha (sigmoid): {learned_alpha:.4f} "
              f"({learned_alpha*100:.1f}% physical + {(1-learned_alpha)*100:.1f}% adaptive)")

    if "graph_module.E1" in state_dict and "graph_module.E2" in state_dict:
        e1 = state_dict["graph_module.E1"]
        e2 = state_dict["graph_module.E2"]
        print(f"  - Node Embeddings: E1={tuple(e1.shape)}, E2={tuple(e2.shape)}")

    print(f"\n{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint (.pt) file")
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Path to .pt checkpoint file (defaults to latest in checkpoints/)")
    parser.add_argument("--show-values", action="store_true",
                        help="Display sample tensor values")
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = os.path.join(ROOT, "checkpoints")
        ckpt_path = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is None:
            print("[INFO] No checkpoint specified and no .pt file found in checkpoints/.")
            print("       Usage: python scripts/inspect_checkpoint.py <path/to/best_model.pt>")
            return
        print(f"[INFO] No path specified. Auto-detected latest checkpoint: {ckpt_path}")

    inspect_checkpoint(ckpt_path, show_values=args.show_values)


if __name__ == "__main__":
    main()
