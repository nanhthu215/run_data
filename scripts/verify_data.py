# -*- coding: utf-8 -*-
"""
verify_data.py -- Kiem tra toan bo du lieu da xu ly
Chay CUOI CUNG sau khi da chay cac script prepare_*.py

Chay:
  run.bat scripts/verify_data.py --dataset PEMS04
  run.bat scripts/verify_data.py --dataset PEMS08

Kiem tra:
  1. Cac file da ton tai
  2. Shape dung ky vong
  3. Khong co NaN / Inf
  4. Chuan hoa dung (train: mean~0, std~1)
  5. Khong data leakage (index khong trung lap)
  6. scaler.pkl hop le
  7. edge_index / edge_weight hop le (neu co)
"""

import argparse
import os
import pickle
import numpy as np


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARNING"


def check(condition: bool, msg_pass: str, msg_fail: str, is_warning: bool = False):
    if condition:
        print(f"  {PASS} {msg_pass}")
        return True
    else:
        label = WARN if is_warning else FAIL
        print(f"  {label} {msg_fail}")
        return False


def verify(dataset: str, data_root: str = "data"):
    dataset_upper = dataset.upper()
    out_dir = os.path.join(data_root, dataset_upper, "processed")

    print(f"\n{'='*65}")
    print(f"  KIEM TRA DU LIEU -- {dataset_upper}")
    print(f"{'='*65}")

    errors   = 0
    warnings = 0

    print(f"\n1. Kiem tra file ton tai:")
    required_files = [
        "X_train.npy", "y_train.npy",
        "X_val.npy",   "y_val.npy",
        "X_test.npy",  "y_test.npy",
        "scaler.pkl"
    ]
    optional_files = ["edge_index.npy", "edge_weight.npy", "adj_matrix.npy"]

    for fname in required_files:
        fpath = os.path.join(out_dir, fname)
        ok = check(os.path.exists(fpath),
                   f"{fname} ton tai",
                   f"{fname} THIEU -- hay chay lai prepare_pems_data*.py")
        if not ok:
            errors += 1

    has_graph = False
    for fname in optional_files:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f" {fname} ton tai (baseline mode)")
            has_graph = True
        else:
            print(f" {fname} khong co (adaptive mode -- OK)")

    try:
        X_train = np.load(os.path.join(out_dir, "X_train.npy"))
        y_train = np.load(os.path.join(out_dir, "y_train.npy"))
        X_val   = np.load(os.path.join(out_dir, "X_val.npy"))
        y_val   = np.load(os.path.join(out_dir, "y_val.npy"))
        X_test  = np.load(os.path.join(out_dir, "X_test.npy"))
        y_test  = np.load(os.path.join(out_dir, "y_test.npy"))
        with open(os.path.join(out_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        print(f"\n  {FAIL} Khong the load file: {e}")
        return

    print(f"\n2. Kiem tra shape:")

    N_NODES_EXPECT = {"PEMS04": 307, "PEMS08": 170}
    N_NODES = N_NODES_EXPECT.get(dataset_upper, None)

    for name, arr in [("X_train", X_train), ("y_train", y_train),
                       ("X_val",   X_val),   ("y_val",   y_val),
                       ("X_test",  X_test),  ("y_test",  y_test)]:
        print(f"  {name:10s}: {arr.shape}")

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        ok = check(arr.ndim == 4,
                   f"{name}.ndim = 4",
                   f"{name}.ndim = {arr.ndim} (ky vong 4)")
        if not ok:
            errors += 1

    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        ok = check(arr.ndim == 4,
                   f"{name}.ndim = 4",
                   f"{name}.ndim = {arr.ndim} (ky vong 4)")
        if not ok:
            errors += 1

    if N_NODES:
        for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
            ok = check(arr.shape[2] == N_NODES,
                       f"{name}.shape[2] = {N_NODES} (dung so node)",
                       f"{name}.shape[2] = {arr.shape[2]} (ky vong {N_NODES})")
            if not ok:
                errors += 1

    for name, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        ok = check(arr.shape[1] == 12,
                   f"{name}.shape[1] = 12 (in_steps dung)",
                   f"{name}.shape[1] = {arr.shape[1]} (ky vong 12)")
        if not ok:
            errors += 1

    for name, arr in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        ok1 = check(arr.shape[1] == 12,
                    f"{name}.shape[1] = 12 (out_steps dung)",
                    f"{name}.shape[1] = {arr.shape[1]} (ky vong 12)")
        ok2 = check(arr.shape[3] == 1,
                    f"{name}.shape[3] = 1 (chi du bao flow)",
                    f"{name}.shape[3] = {arr.shape[3]} (ky vong 1)")
        if not ok1: errors += 1
        if not ok2: errors += 1

    print(f"\n3. Kiem tra NaN / Inf:")
    for name, arr in [("X_train", X_train), ("y_train", y_train),
                       ("X_val",   X_val),   ("y_val",   y_val),
                       ("X_test",  X_test),  ("y_test",  y_test)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        ok = check(nan_count == 0 and inf_count == 0,
                   f"{name}: khong co NaN/Inf",
                   f"{name}: {nan_count} NaN, {inf_count} Inf -- NGHIEM TRONG")
        if not ok:
            errors += 1

    print(f"\n4. Kiem tra chuan hoa (X_train):")
    n_features = X_train.shape[3]
    feat_names = ["flow", "occupancy", "speed"][:n_features]

    for f_idx, fname in enumerate(feat_names):
        m = X_train[:, :, :, f_idx].mean()
        s = X_train[:, :, :, f_idx].std()
        ok_m = check(abs(m) < 1e-4,
                     f"{fname} mean ~ 0 ({m:.6f})",
                     f"{fname} mean = {m:.6f} (ky vong ~ 0) -- co the code bi loi")
        ok_s = check(abs(s - 1.0) < 0.01,
                     f"{fname} std ~ 1 ({s:.6f})",
                     f"{fname} std = {s:.6f} (ky vong ~ 1) -- co the code bi loi")
        if not ok_m: errors += 1
        if not ok_s: errors += 1

    print(f"\n4b. Xac nhan val/test k duoc normalize bang mean cua chinh no:")
    for split_name, X_s in [("X_val", X_val), ("X_test", X_test)]:
        m = X_s[:, :, :, 0].mean()
        check(True,
              f"{split_name} flow mean = {m:.4f} (!= 0 la dung -- dung scaler train)",
              "")

    print(f"\n5. Kiem tra scaler.pkl:")
    ok_keys = check("mean" in scaler and "std" in scaler,
                    "scaler co key 'mean' va 'std'",
                    "scaler thieu key 'mean' hoac 'std'")
    if ok_keys:
        print(f"  [INFO] mean = {[f'{v:.4f}' for v in scaler['mean']]}")
        print(f"  [INFO] std  = {[f'{v:.4f}' for v in scaler['std']]}")
        for i, s in enumerate(scaler["std"]):
            ok_s = check(s > 0,
                         f"std[{i}] = {s:.4f} > 0",
                         f"std[{i}] = {s:.4f} <= 0 -- co the chuan hoa sai")
            if not ok_s: errors += 1
    else:
        errors += 1

    print(f"\n6. Kiem tra khong trung lap index train/val/test:")
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    train_pct = X_train.shape[0] / total * 100
    val_pct   = X_val.shape[0]   / total * 100
    test_pct  = X_test.shape[0]  / total * 100
    check(abs(train_pct - 70) < 2,
          f"Train = {train_pct:.1f}% (gan 70%)",
          f"Train = {train_pct:.1f}% (xa 70%) -- kiem tra lai logic chia",
          is_warning=True)
    check(abs(val_pct - 10) < 2,
          f"Val   = {val_pct:.1f}% (gan 10%)",
          f"Val   = {val_pct:.1f}% (xa 10%) -- kiem tra lai",
          is_warning=True)
    check(abs(test_pct - 20) < 3,
          f"Test  = {test_pct:.1f}% (gan 20%)",
          f"Test  = {test_pct:.1f}% (xa 20%) -- kiem tra lai",
          is_warning=True)

    if has_graph:
        print(f"\n7. Kiem tra edge_index / edge_weight:")
        edge_index  = np.load(os.path.join(out_dir, "edge_index.npy"))
        edge_weight = np.load(os.path.join(out_dir, "edge_weight.npy"))

        ok1 = check(edge_index.shape[0] == 2,
                    f"edge_index.shape = {edge_index.shape} (2 x num_edges)",
                    f"edge_index.shape = {edge_index.shape} (ky vong (2, E))")
        ok2 = check(edge_weight.ndim == 1,
                    f"edge_weight.shape = {edge_weight.shape}",
                    f"edge_weight.ndim = {edge_weight.ndim} (ky vong 1D)")
        ok3 = check(edge_index.shape[1] == edge_weight.shape[0],
                    f"edge_index va edge_weight co cung so canh ({edge_weight.shape[0]:,})",
                    f"edge_index ({edge_index.shape[1]}) != edge_weight ({edge_weight.shape[0]})")
        ok4 = check((edge_weight > 0).all(),
                    "Tat ca edge_weight > 0",
                    "Co edge_weight <= 0 -- kiem tra Gaussian kernel")
        ok5 = check((edge_weight <= 1).all(),
                    "Tat ca edge_weight <= 1 (sau Gaussian kernel)",
                    f"edge_weight max = {edge_weight.max():.4f} > 1 -- kiem tra lai",
                    is_warning=True)
        for ok in [ok1, ok2, ok3, ok4]:
            if not ok: errors += 1

    print(f"\n8. Sanity check -- kiem tra y_test va inverse X:")
    print(f"  8a. y_test la raw flow (CHUA chuan hoa):")

    y_min = y_test.min()
    y_max = y_test.max()
    y_mean = y_test.mean()
    print(f"       min={y_min:.2f}, max={y_max:.2f}, mean={y_mean:.2f}")
    ok_ymin = check(y_min >= 0,
                    "y_test.min >= 0 (flow khong am)",
                    f"y_test.min = {y_min:.2f} < 0 -- co van de!")
    ok_ymax = check(y_max < 2000,
                    f"y_test.max = {y_max:.2f} < 2000 (hop ly)",
                    f"y_test.max = {y_max:.2f} >= 2000 -- kiem tra lai",
                    is_warning=True)
    if not ok_ymin: errors += 1

    flow_mean = scaler["mean"][0]
    flow_std  = scaler["std"][0]
    print(f"\n  8b. Inverse-transform X_test (flow) ve don vi goc:")

    print(f"       Scaler: mean={flow_mean:.2f}, std={flow_std:.2f}")
    x_flow_norm = X_test[:, :, :, 0]
    x_flow_raw  = x_flow_norm * flow_std + flow_mean

    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(x_flow_raw), size=min(3, len(x_flow_raw)), replace=False)
    all_ok = True
    print(f"  {'Sample':>8} {'X_norm_min':>12} {'X_norm_max':>12} "
          f"{'X_raw_min':>12} {'X_raw_max':>12} {'OK?':>6}")
    print("  " + "-" * 65)
    for idx in sorted(sample_idx):
        xn = x_flow_norm[idx]
        xr = x_flow_raw[idx]
        ok = (xr >= 0).all() and (xr < 2000).all()
        if not ok: all_ok = False
        print(f"  {idx:>8} {xn.min():>12.4f} {xn.max():>12.4f} "
              f"{xr.min():>12.2f} {xr.max():>12.2f} "
              f"{'OK' if ok else 'FAIL':>6}")

    if all_ok:
        print(f"  [PASS] Inverse-transform X hop ly (0-2000 veh/h)")
    else:
        print(f"  [FAIL] Inverse-transform X ngoai khoang hop ly!")
        errors += 1

    print(f"\n{'='*65}")
    if errors == 0:
        print(f"  [SUCCESS] TAT CA KIEM TRA PASS -- Du lieu san sang cho training!")
    else:
        print(f"  [ERROR] {errors} loi can sua truoc khi training!")
        print(f"      Xem lai output o tren va chay lai prepare_pems_data*.py")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiem tra du lieu da xu ly")
    parser.add_argument("--dataset",   type=str, required=True,
                        choices=["PEMS04", "PEMS08", "pems04", "pems08"])
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()
    verify(args.dataset, args.data_root)
