# -*- coding: utf-8 -*-
"""
prepare_kfold_data.py — Expanding Window Cross-Validation cho Time Series
=========================================================================
Tạo K fold dữ liệu cho Expanding Window CV (biến thể K-Fold hợp lệ cho
Time Series, tránh temporal leakage — vi phạm chủ đạo trong Time Series CV).

Expanding Window CV:
  - Fold 0: Train [0 – 50%], Val [50 – 60%], Test [60 – 70%]
  - Fold 1: Train [0 – 60%], Val [60 – 70%], Test [70 – 80%]
  - Fold 2: Train [0 – 70%], Val [70 – 80%], Test [80 – 90%]
  - Fold 3: Train [0 – 80%], Val [80 – 90%], Test [90 – 100%]

Tại sao KHÔNG dùng Standard K-Fold (random shuffle)?
  → Trong Time Series, random K-Fold gây rò rỉ thời gian (temporal leakage):
    model "nhìn thấy tương lai" để dự đoán quá khứ, vi phạm nguyên tắc nhân quả.
  → Tất cả bài báo chuẩn mực (DCRNN, AGCRN, STGCN, GWNet) đều dùng
    chronological split. Expanding Window là phiên bản K-Fold đúng cho TS.

Mỗi fold lưu ra thư mục riêng:
  data/PEMS08/fold_0/  →  X_train.npy, y_train.npy, X_val.npy, ...
  data/PEMS08/fold_1/  →  ...
  ...

Scaler (Z-score) chỉ fit trên train set CỦA FOLD ĐÓ, tránh rò rỉ.

Usage:
  python scripts/prepare_kfold_data.py --dataset PEMS08 --n-folds 4
  python scripts/prepare_kfold_data.py --dataset PEMS04 --n-folds 4
"""

import argparse
import os
import sys
import pickle
import numpy as np

# Import BaseDataPreprocessor để kế thừa sliding window + normalize
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    from scripts.prepare_pems_data import GraphDataPreprocessor
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from prepare_pems_data import GraphDataPreprocessor


def create_kfold_splits(n_samples: int, n_folds: int = 4):
    """
    Tạo expanding window splits cho Time Series.

    Chia n_samples thành (n_folds + 3) segments bằng nhau.
    Fold k: train = [0, boundary_k], val = [boundary_k, boundary_k+1], test = [boundary_k+1, boundary_k+2]

    Với n_folds=4:
      segment_size = n_samples / 10  (chia 10 phần)
      Fold 0: train [0–5/10], val [5/10–6/10], test [6/10–7/10]  → train=50%, val=10%, test=10%
      Fold 1: train [0–6/10], val [6/10–7/10], test [7/10–8/10]  → train=60%, val=10%, test=10%
      Fold 2: train [0–7/10], val [7/10–8/10], test [8/10–9/10]  → train=70%, val=10%, test=10%
      Fold 3: train [0–8/10], val [8/10–9/10], test [9/10–10/10] → train=80%, val=10%, test=10%

    Returns:
        List of (train_idx, val_idx, test_idx) tuples
    """
    # Tổng cộng cần n_folds + 3 segments nhưng với 10-segment layout
    # thì n_folds=4 → 10 segments, 4 folds khớp hoàn hảo
    n_segments = n_folds + 3 + 3  # = 10 cho n_folds=4

    boundaries = [int(i * n_samples / n_segments) for i in range(n_segments + 1)]

    splits = []
    # Fold k: train [0, 5+k], val [5+k, 6+k], test [6+k, 7+k]
    for k in range(n_folds):
        train_end = boundaries[5 + k]         # expanding
        val_end   = boundaries[6 + k]
        test_end  = boundaries[7 + k]

        train_idx = np.arange(0, train_end)
        val_idx   = np.arange(train_end, val_end)
        test_idx  = np.arange(val_end, test_end)

        splits.append((train_idx, val_idx, test_idx))

    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Tạo Expanding Window K-Fold data cho Time Series CV"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["PEMS04", "PEMS08", "pems04", "pems08"],
    )
    parser.add_argument(
        "--data-root", type=str, default="data",
    )
    parser.add_argument(
        "--n-folds", type=int, default=4,
        help="Số fold (mặc định 4: Expanding Window 50→80%%)",
    )
    args = parser.parse_args()
    dataset = args.dataset.upper()

    print(f"\n{'='*60}")
    print(f"  EXPANDING WINDOW K-FOLD — {dataset}")
    print(f"  {args.n_folds} folds")
    print(f"{'='*60}")

    # ---- 1. Load raw data ----
    # Dùng GraphDataPreprocessor để có cả adjacency matrix
    preprocessor = GraphDataPreprocessor(
        dataset=dataset,
        data_root=args.data_root,
    )
    raw_data = preprocessor.load_data()
    T, N, F = raw_data.shape
    print(f"\n[Raw data] shape: T={T:,}, N={N}, F={F}")

    # ---- 2. Build sliding windows ----
    X, y = preprocessor.build_sliding_windows(raw_data)
    n_samples = len(X)
    print(f"[Sliding windows] {n_samples:,} samples (X: {X.shape}, y: {y.shape})")

    # ---- 3. Build adjacency matrix ----
    adj, edge_index, edge_weight = preprocessor.build_adjacency(N)

    # ---- 4. Tạo K-fold splits ----
    splits = create_kfold_splits(n_samples, args.n_folds)

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"\n{'-'*50}")
        print(f"  Fold {fold_idx}: train={len(train_idx):,} "
              f"({len(train_idx)/n_samples*100:.0f}%), "
              f"val={len(val_idx):,} ({len(val_idx)/n_samples*100:.0f}%), "
              f"test={len(test_idx):,} ({len(test_idx)/n_samples*100:.0f}%)")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # ---- Normalize: Z-score chỉ fit trên train set CỦA FOLD NÀY ----
        X_train_n, X_val_n, X_test_n, scaler = preprocessor.normalize(
            X_train, X_val, X_test, n_features=F
        )

        # ---- Lưu ra thư mục fold riêng ----
        fold_dir = os.path.join(args.data_root, dataset, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        np.save(os.path.join(fold_dir, "X_train.npy"), X_train_n)
        np.save(os.path.join(fold_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_dir, "X_val.npy"),   X_val_n)
        np.save(os.path.join(fold_dir, "y_val.npy"),   y_val)
        np.save(os.path.join(fold_dir, "X_test.npy"),  X_test_n)
        np.save(os.path.join(fold_dir, "y_test.npy"),  y_test)

        with open(os.path.join(fold_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        # Adjacency matrix giống cho mọi fold (cấu trúc đường không đổi)
        np.save(os.path.join(fold_dir, "adj_matrix.npy"), adj)
        np.save(os.path.join(fold_dir, "edge_index.npy"), edge_index)
        np.save(os.path.join(fold_dir, "edge_weight.npy"), edge_weight)

        print(f"  -> Saved to {fold_dir}")

        # Kiểm tra chống rò rỉ: time ordering
        assert train_idx[-1] < val_idx[0], f"Fold {fold_idx}: train overlaps val!"
        assert val_idx[-1] < test_idx[0], f"Fold {fold_idx}: val overlaps test!"

    print(f"\n{'='*60}")
    print(f"  DONE -- {args.n_folds} folds created")
    print(f"  Directory: {os.path.join(args.data_root, dataset, 'fold_*')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
