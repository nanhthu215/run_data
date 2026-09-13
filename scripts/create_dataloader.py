# -*- coding: utf-8 -*-
"""
create_dataloader.py -- Tao PyTorch Dataset va DataLoader cho T-GCN Adaptive

Mo ta:
  Wrap cac file X_train/val/test.npy va y_train/val/test.npy thanh
  Dataset chuan PyTorch, sau do tao DataLoader cho tung tap.

Chay kiem tra nhanh:
  python scripts/create_dataloader.py --dataset PEMS04
  python scripts/create_dataloader.py --dataset PEMS08

Output:
  In kich thuoc tung loader va kiem tra shape 1 batch.
  Cac ham get_dataloaders() duoc import boi model training script.
"""

import argparse
import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False



class PeMSDataset(Dataset):
    def __init__(self, X_path: str, y_path: str):
        assert os.path.exists(X_path), "Khong tim thay: " + X_path
        assert os.path.exists(y_path), "Khong tim thay: " + y_path

        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        assert len(self.X) == len(self.y), (
            "X va y phai co cung so samples: {} vs {}".format(
                len(self.X), len(self.y)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if TORCH_AVAILABLE:
            return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
        else:
            return self.X[idx], self.y[idx]

    @property
    def n_nodes(self):
        return self.X.shape[2]

    @property
    def n_features(self):
        return self.X.shape[3]

    @property
    def in_steps(self):
        return self.X.shape[1]

    @property
    def out_steps(self):
        return self.y.shape[1]


def get_dataloaders(dataset: str,
                    data_root: str = "data",
                    batch_size: int = 64,
                    num_workers: int = 0):
    """
    Tao va tra ve ba DataLoader cho train / val / test.

    Quy tac shuffle:
        - train_loader: shuffle=True  (ngau nhien thu tu batch trong tap train)
        - val_loader:   shuffle=False (giu thu tu de debug)
        - test_loader:  shuffle=False (BAT BUOC -- ket qua phai co thu tu xac dinh)

    shuffle=True TRONG loader khac voi viec shuffle truoc khi chia tap.
    Chia tap da duoc thuc hien theo thu tu thoi gian o buoc prepare -- khong doi.

    """
    dataset_upper = dataset.upper()
    proc_dir = os.path.join(data_root, dataset_upper, "processed")

    assert os.path.isdir(proc_dir), (
        "Chua co thu muc processed: {}. "
        "Hay chay prepare_pems_data_adaptive.py truoc.".format(proc_dir))

    train_ds = PeMSDataset(
        os.path.join(proc_dir, "X_train.npy"),
        os.path.join(proc_dir, "y_train.npy"))
    val_ds = PeMSDataset(
        os.path.join(proc_dir, "X_val.npy"),
        os.path.join(proc_dir, "y_val.npy"))
    test_ds = PeMSDataset(
        os.path.join(proc_dir, "X_test.npy"),
        os.path.join(proc_dir, "y_test.npy"))

    # Tao DataLoader
    if TORCH_AVAILABLE:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,          
            num_workers=num_workers,
            pin_memory=True,       
            drop_last=False)

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,         # KHONG shuffle test -- ket qua phai xac dinh
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False)
    else:
        # Fallback khi chua cai PyTorch: tra ve list of batches don gian
        train_loader = _simple_loader(train_ds, batch_size, shuffle=True)
        val_loader   = _simple_loader(val_ds,   batch_size, shuffle=False)
        test_loader  = _simple_loader(test_ds,  batch_size, shuffle=False)

    info = {
        "n_nodes":    train_ds.n_nodes,
        "n_features": train_ds.n_features,
        "in_steps":   train_ds.in_steps,
        "out_steps":  train_ds.out_steps,
        "n_train":    len(train_ds),
        "n_val":      len(val_ds),
        "n_test":     len(test_ds),
        "batch_size": batch_size,
    }
    return train_loader, val_loader, test_loader, info


def _simple_loader(dataset, batch_size, shuffle=False):
    """Fallback loader don gian khi chua co PyTorch."""
    indices = list(range(len(dataset)))
    if shuffle:
        import random
        random.shuffle(indices)
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i + batch_size]
        Xs = np.stack([dataset.X[j] for j in batch_idx])
        ys = np.stack([dataset.y[j] for j in batch_idx])
        batches.append((Xs, ys))
    return batches


def run_check(dataset: str, data_root: str = "data", batch_size: int = 64):
    print("\n" + "=" * 60)
    print("  KIEM TRA DATALOADER -- " + dataset.upper())
    print("=" * 60)

    if not TORCH_AVAILABLE:
        print("\n[CANH BAO] PyTorch chua duoc cai dat.")
        print("  Chay: pip install torch")
        print("  Dung fallback loader de kiem tra shape.\n")

    train_loader, val_loader, test_loader, info = get_dataloaders(
        dataset, data_root, batch_size)

    print("\n[Thong tin dataset]")
    print("  N_nodes   : {:,}".format(info["n_nodes"]))
    print("  N_features: {:,}".format(info["n_features"]))
    print("  In steps  : {:,}".format(info["in_steps"]))
    print("  Out steps : {:,}".format(info["out_steps"]))
    print("  Batch size: {:,}".format(info["batch_size"]))

    print("\n[Kich thuoc cac tap]")
    print("  Train     : {:,} samples".format(info["n_train"]))
    print("  Val       : {:,} samples".format(info["n_val"]))
    print("  Test      : {:,} samples".format(info["n_test"]))

    # Ky vong batch shape
    N = info["n_nodes"]
    F = info["n_features"]
    T_in  = info["in_steps"]
    T_out = info["out_steps"]
    B = batch_size
    expected_X_shape = (B, T_in,  N, F)
    expected_y_shape = (B, T_out, N, 1)

    print("\n[Ky vong shape 1 batch]")
    print("  batch_X : {}".format(expected_X_shape))
    print("  batch_y : {}".format(expected_y_shape))

    # Lay 1 batch dau tien tu train_loader de kiem tra
    print("\n[Kiem tra batch thuc te -- train_loader]")
    errors = 0

    if TORCH_AVAILABLE:
        batch_X, batch_y = next(iter(train_loader))
        actual_X = tuple(batch_X.shape)
        actual_y = tuple(batch_y.shape)
    else:
        batch_X, batch_y = train_loader[0]
        actual_X = batch_X.shape
        actual_y = batch_y.shape

    print("  batch_X shape: {}".format(actual_X))
    print("  batch_y shape: {}".format(actual_y))

    # Kiem tra shape (batch cuoi co the nho hon batch_size)
    ok_X = (actual_X[1] == T_in  and
            actual_X[2] == N     and
            actual_X[3] == F)
    ok_y = (actual_y[1] == T_out and
            actual_y[2] == N     and
            actual_y[3] == 1)

    if ok_X:
        print("  [PASS] batch_X shape dung (tru so batch)")
    else:
        print("  [FAIL] batch_X shape SAI -- kiem tra lai prepare script")
        errors += 1

    if ok_y:
        print("  [PASS] batch_y shape dung (tru so batch)")
    else:
        print("  [FAIL] batch_y shape SAI -- kiem tra lai prepare script")
        errors += 1

    # Kiem tra dtype
    if TORCH_AVAILABLE:
        if batch_X.dtype == torch.float32:
            print("  [PASS] dtype = torch.float32")
        else:
            print("  [FAIL] dtype = {} (ky vong float32)".format(batch_X.dtype))
            errors += 1

        # Kiem tra NaN trong batch
        if not torch.isnan(batch_X).any():
            print("  [PASS] Khong co NaN trong batch_X")
        else:
            print("  [FAIL] Co NaN trong batch_X!")
            errors += 1

    # Kiem tra so luong batch
    n_train_batches = len(train_loader)
    expected_batches = (info["n_train"] + batch_size - 1) // batch_size
    if n_train_batches == expected_batches:
        print("  [PASS] So batch train = {:,}".format(n_train_batches))
    else:
        print("  [WARN] So batch train = {:,} (ky vong {:,})".format(
              n_train_batches, expected_batches))

    # Kiem tra test_loader khong shuffle: 2 lan lay batch dau phai giong nhau
    if TORCH_AVAILABLE:
        bX1, _ = next(iter(test_loader))
        bX2, _ = next(iter(test_loader))
        if torch.equal(bX1, bX2):
            print("  [PASS] test_loader khong shuffle (ket qua xac dinh)")
        else:
            print("  [FAIL] test_loader co shuffle -- ket qua khong xac dinh!")
            errors += 1

    print("\n" + "=" * 60)
    if errors == 0:
        print("  OK  DataLoader san sang cho training!")
    else:
        print("  {} loi can sua truoc khi training!".format(errors))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tao va kiem tra DataLoader PeMS")
    parser.add_argument("--dataset",    type=str, required=True,
                        choices=["PEMS04", "PEMS08", "pems04", "pems08"])
    parser.add_argument("--data_root",  type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    run_check(args.dataset, args.data_root, args.batch_size)
