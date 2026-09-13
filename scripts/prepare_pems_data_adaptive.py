# -*- coding: utf-8 -*-
"""
prepare_pems_data_adaptive.py -- Chuan bi du lieu cho T-GCN Adaptive
Tu hoc do thi qua node embeddings (khong can distance.csv).

"""

import argparse
import os
import pickle
import numpy as np


class BaseDataPreprocessor:
    """Tien xu ly du lieu chuoi thoi gian (sliding window, split, Z-score)."""

    def __init__(
        self,
        dataset,
        data_root="data",
        in_steps=12,
        out_steps=12,
        train_ratio=0.6,
        val_ratio=0.2,
        feature_names=None,
    ):
        self.dataset = dataset.upper()
        self.data_root = data_root
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.feature_names = feature_names or ["flow", "occupancy", "speed"]

        self.npz_path = os.path.join(self.data_root, self.dataset, f"{dataset.lower()}.npz")
        self.out_dir = os.path.join(self.data_root, self.dataset, "processed")

    def load_data(self):
        # Doc file .npz tho -> mang shape (T, N, F)
        assert os.path.exists(self.npz_path), f"Khong tim thay: {self.npz_path}"
        raw = np.load(self.npz_path, allow_pickle=True)
        return raw["data"].astype(np.float32)

    def build_sliding_windows(self, data):
        # Cat cua so truot: in_steps buoc qua khu -> out_steps buoc tuong lai
        T, N, F = data.shape
        n_samples = T - self.in_steps - self.out_steps + 1

        X = np.empty((n_samples, self.in_steps, N, F), dtype=np.float32)
        y = np.empty((n_samples, self.out_steps, N, 1), dtype=np.float32)

        for i in range(n_samples):
            X[i] = data[i : i + self.in_steps]
            y[i] = data[i + self.in_steps : i + self.in_steps + self.out_steps, :, 0:1]

        return X, y

    def split_data(self, X, y):
        # Chia train / val / test theo thu tu thoi gian (tranh data leakage)
        n = len(X)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train = (X[:n_train], y[:n_train])
        val = (X[n_train : n_train + n_val], y[n_train : n_train + n_val])
        test = (X[n_train + n_val :], y[n_train + n_val :])

        return train, val, test

    def normalize(self, X_train, X_val, X_test, n_features):
        # Chuan hoa Z-score per feature -- chi fit tren train set
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        means = []
        stds = []

        for f in range(n_features):
            mean_f = X_train[:, :, :, f].mean()
            std_f = X_train[:, :, :, f].std()

            if std_f < 1e-8:
                std_f = 1.0

            X_train[:, :, :, f] = (X_train[:, :, :, f] - mean_f) / std_f
            X_val[:, :, :, f] = (X_val[:, :, :, f] - mean_f) / std_f
            X_test[:, :, :, f] = (X_test[:, :, :, f] - mean_f) / std_f

            means.append(float(mean_f))
            stds.append(float(std_f))

            feat_label = self.feature_names[f] if f < len(self.feature_names) else f"F_{f}"
            print(f"  Feature {feat_label:10s}: mean={mean_f:.4f}, std={std_f:.4f}")

        scaler = {
            "mean": means,
            "std": stds,
            "feature_names": self.feature_names[:n_features],
        }
        return X_train, X_val, X_test, scaler

    def save_outputs(self, train, val, test, scaler):
        os.makedirs(self.out_dir, exist_ok=True)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

        np.save(os.path.join(self.out_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.out_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.out_dir, "X_val.npy"), X_val)
        np.save(os.path.join(self.out_dir, "y_val.npy"), y_val)
        np.save(os.path.join(self.out_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.out_dir, "y_test.npy"), y_test)

        with open(os.path.join(self.out_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        print(f"\nTat ca file da luu vao: {self.out_dir}")

    def process(self):
        raise NotImplementedError


class AdaptiveDataPreprocessor(BaseDataPreprocessor):
    """Pipeline tien xu ly cho mo hinh Adaptive (tu hoc do thi)."""

    def process(self):
        print(f"\n{'='*60}")
        print(f"  CHUAN BI DU LIEU (Adaptive) -- {self.dataset}")
        print(f"{'='*60}")

        # 1. Load data
        data = self.load_data()
        T, N, F = data.shape
        print(f"\n[Load] Shape tho: T={T:,}, N={N}, F={F}")

        # 2. Sliding window
        X, y = self.build_sliding_windows(data)
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")

        # 3. Chia train / val / test
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        print(f"\nTrain={self.train_ratio:.0%}, Val={self.val_ratio:.0%}, Test={test_ratio:.0%}")
        train_data, val_data, test_data = self.split_data(X, y)
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        print(f"  Train: {X_train.shape[0]:,} samples")
        print(f"  Val  : {X_val.shape[0]:,} samples")
        print(f"  Test : {X_test.shape[0]:,} samples")

        # 4. Z-score per feature
        print(f"\n[Chuan hoa] Z-score per feature -- fit chi tren train:")
        X_train, X_val, X_test, scaler = self.normalize(X_train, X_val, X_test, n_features=F)

        print(f"\n[Kiem tra] X_train sau chuan hoa:")
        for f in range(F):
            m = X_train[:, :, :, f].mean()
            s = X_train[:, :, :, f].std()
            ok_m = "OK" if abs(m) < 1e-4 else "CANH BAO"
            ok_s = "OK" if abs(s - 1.0) < 0.01 else "CANH BAO"
            feat_label = self.feature_names[f] if f < len(self.feature_names) else f"F_{f}"
            print(f"  {feat_label:10s}: mean={m:.6f} [{ok_m}], std={s:.6f} [{ok_s}]")

        # 5. Luu output
        self.save_outputs(
            (X_train, y_train),
            (X_val, y_val),
            (X_test, y_test),
            scaler,
        )

        for split, X_s, y_s in [("Train", X_train, y_train),
                                ("Val", X_val, y_val),
                                ("Test", X_test, y_test)]:
            print(f"  {split:6s}  X: {str(X_s.shape):30s}  y: {y_s.shape}")

        print(f"\n  NaN trong X_train: {np.isnan(X_train).sum()}")
        print(f"  NaN trong y_train: {np.isnan(y_train).sum()}")
        print(f"\n{'='*60}")
        print(f"  HOAN THANH -- San sang cho T-GCN Adaptive")
        print(f"{'='*60}\n")


def main(dataset, data_root="data"):
    preprocessor = AdaptiveDataPreprocessor(dataset=dataset, data_root=data_root)
    preprocessor.process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuan bi du lieu cho T-GCN Adaptive")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PEMS04", "PEMS08", "pems04", "pems08"],
    )
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    main(args.dataset, args.data_root)
