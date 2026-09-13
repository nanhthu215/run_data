# -*- coding: utf-8 -*-
"""
prepare_pems_data.py -- Chuan bi du lieu cho cac Baseline can do thi co dinh
Doc distance.csv -> xay do thi Gaussian kernel -> luu edge_index + edge_weight + adj_matrix.

Dung cho: DCRNN, STFGNN, AGCRN (cac mo hinh can ma tran ke vat ly)

"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

try:
    from scripts.prepare_pems_data_adaptive import BaseDataPreprocessor
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from prepare_pems_data_adaptive import BaseDataPreprocessor


SIGMA = 500.0          # Bandwidth Gaussian kernel (met)
DIST_THRESHOLD = 2000.0 # Loai canh neu khoang cach > 2000m


class GraphDataPreprocessor(BaseDataPreprocessor):
    """Pipeline tien xu ly co tao them ma tran ke do thi tu distance.csv."""

    def __init__(
        self,
        dataset,
        data_root="data",
        in_steps=12,
        out_steps=12,
        train_ratio=0.6,
        val_ratio=0.2,
        sigma=SIGMA,
        threshold=DIST_THRESHOLD,
    ):
        super().__init__(
            dataset=dataset,
            data_root=data_root,
            in_steps=in_steps,
            out_steps=out_steps,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        self.sigma = sigma
        self.threshold = threshold
        self.dist_path = os.path.join(self.data_root, self.dataset, "distance.csv")

    def build_adjacency(self, n_nodes):
        # Doc distance.csv -> tao adjacency matrix bang Gaussian kernel
        assert os.path.exists(self.dist_path), f"Khong tim thay: {self.dist_path}"

        df = pd.read_csv(self.dist_path, header=0)
        col_names = list(df.columns)
        from_col, to_col, cost_col = col_names[0], col_names[1], col_names[2]

        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        kept, total = 0, 0

        for _, row in df.iterrows():
            i = int(row[from_col])
            j = int(row[to_col])
            d = float(row[cost_col])
            total += 1

            if i >= n_nodes or j >= n_nodes:
                continue

            if d < self.threshold:
                w = np.exp(-(d ** 2) / (self.sigma ** 2))
                adj[i, j] = w
                adj[j, i] = w  # Do thi vo huong (symmetric)
                kept += 1

        print(f"  [Gaussian kernel] sigma={self.sigma}, threshold={self.threshold}")
        print(f"  Tong cap trong file: {total:,}")
        print(f"  So canh giu lai: {kept:,} ({kept / total * 100:.1f}%)")
        print(f"  Mat do do thi: {(adj > 0).sum() / (n_nodes * n_nodes) * 100:.2f}%")

        # COO format cho PyTorch Geometric
        sp = coo_matrix(adj)
        edge_index = np.vstack([sp.row, sp.col]).astype(np.int64)
        edge_weight = sp.data.astype(np.float32)

        return adj, edge_index, edge_weight

    def save_graph_outputs(self, adj, edge_index, edge_weight):
        os.makedirs(self.out_dir, exist_ok=True)
        np.save(os.path.join(self.out_dir, "edge_index.npy"), edge_index)
        np.save(os.path.join(self.out_dir, "edge_weight.npy"), edge_weight)
        np.save(os.path.join(self.out_dir, "adj_matrix.npy"), adj)

    def process(self):
        print(f"\n{'='*60}")
        print(f"  CHUAN BI DU LIEU (Baseline + Graph) -- {self.dataset}")
        print(f"{'='*60}")

        # 1. Load data
        data = self.load_data()
        T, N, F = data.shape
        print(f"\n[Load] Shape tho: T={T:,}, N={N}, F={F}")

        # 2. Sliding window
        print(f"\n[Sliding window] in={self.in_steps}, out={self.out_steps} buoc")
        X, y = self.build_sliding_windows(data)
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

        # 3. Chia train / val / test
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        print(f"\n[Chia] Train={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={test_ratio:.0%}")
        train_data, val_data, test_data = self.split_data(X, y)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        print(f"  Train: {X_train.shape[0]:,} | Val: {X_val.shape[0]:,} | Test: {X_test.shape[0]:,}")

        # 4. Z-score normalization
        print(f"\n[Chuan hoa] Z-score per feature (fit only on train):")
        X_train, X_val, X_test, scaler = self.normalize(X_train, X_val, X_test, n_features=F)

        # 5. Xay do thi Gaussian
        print(f"\n[Xay do thi] Tu {self.dist_path}")
        adj, edge_index, edge_weight = self.build_adjacency(N)

        # 6. Luu tat ca ket qua
        self.save_outputs(
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            scaler,
        )
        self.save_graph_outputs(adj, edge_index, edge_weight)

        # 7. Tom tat
        print(f"\n{'-'*60}")
        print(f"  TOM TAT OUTPUT")
        print(f"{'-'*60}")
        for split, X_s, y_s in [("Train", X_train, y_train),
                                ("Val", X_val, y_val),
                                ("Test", X_test, y_test)]:
            print(f"  {split:6s}  X: {str(X_s.shape):35s}  y: {y_s.shape}")
        print(f"  edge_index : {edge_index.shape}  (2 x num_edges)")
        print(f"  edge_weight: {edge_weight.shape}")
        print(f"  adj_matrix : {adj.shape}")
        print(f"\n  NaN trong X_train: {np.isnan(X_train).sum()}")
        print(f"\n[Luu thanh cong] {self.out_dir}")
        print(f"\n{'='*60}")
        print(f"  HOAN THANH -- San sang cho DCRNN, STFGNN, AGCRN")
        print(f"{'='*60}\n")


def main(dataset, data_root="data", sigma=SIGMA, threshold=DIST_THRESHOLD):
    preprocessor = GraphDataPreprocessor(
        dataset=dataset,
        data_root=data_root,
        sigma=sigma,
        threshold=threshold,
    )
    preprocessor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuan bi du lieu cho baseline can do thi co dinh")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PEMS04", "PEMS08", "pems04", "pems08"],
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument(
        "--sigma",
        type=float,
        default=SIGMA,
        help=f"Bandwidth Gaussian kernel (default: {SIGMA})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DIST_THRESHOLD,
        help=f"Nguong khoang cach toi da de tao canh (default: {DIST_THRESHOLD})",
    )
    args = parser.parse_args()
    main(args.dataset, args.data_root, args.sigma, args.threshold)
