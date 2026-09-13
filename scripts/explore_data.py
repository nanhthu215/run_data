# -*- coding: utf-8 -*-
"""
explore_data.py -- Kham pha va kiem tra du lieu tho PeMS04/08
Chay: python scripts/explore_data.py --dataset PEMS04
"""
import argparse
import os
import numpy as np
import pandas as pd


def explore(dataset: str, data_root: str = "data"):
    dataset_upper = dataset.upper()
    npz_path = os.path.join(data_root, dataset_upper, f"{dataset.lower()}.npz")
    dist_path = os.path.join(data_root, dataset_upper, "distance.csv")

    # 1. Load .npz 
    print("\nKHAM PHA DU LIEU: " + dataset_upper)

    assert os.path.exists(npz_path), "Khong tim thay: " + npz_path
    raw = np.load(npz_path, allow_pickle=True)

    data = raw["data"]
    T, N, F = data.shape
    print("\nShape: T = {:,} buoc | N = {} tram | F = {} features".format(T, N, F))
    print("Thoi gian: {:.0f} ngay ({:.0f} gio)".format(T * 5 / 60 / 24, T * 5 / 60))

    # 2. Thong ke tung feature 
    feat_names = ["flow", "occupancy", "speed"]
    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
          "Feature", "Min", "Max", "Mean", "Std", "Zeros(%)"))
    print("-" * 68)
    for i in range(F):
        feat = data[:, :, i]
        zeros_pct = (feat == 0).mean() * 100
        print("{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>9.1f}%".format(
              feat_names[i], feat.min(), feat.max(),
              feat.mean(), feat.std(), zeros_pct))

    # 3. Kiem tra NaN / Inf 
    print("\nTong NaN = {:,}".format(np.isnan(data).sum()))
    print("Tong Inf = {:,}".format(np.isinf(data).sum()))

    # 4. Kiem tra gia tri am 
    for i, name in enumerate(feat_names):
        neg = (data[:, :, i] < 0).sum()
        if neg > 0:
            print("{} co {:,} gia tri am".format(name, neg))
        else:
            print("{} - k co gia tri am".format(name))

    # 5. Phan tich zeros theo GIO TRONG NGAY
    # Moi ngay co 288 buoc (24h x 12 buoc/h)
    # Group theo tung gio (0-23) de xem zeros co tap trung ban dem khong
    STEPS_PER_HOUR = 12  

    print("\n[PHAN TICH ZEROS THEO GIO TRONG NGAY - flow]")
    print("(Neu zeros tap trung o 0h-5h -> zero that; neu ngau nhien -> missing value)")
    print("{:>6} {:>12} {:>12}".format("Gio", "Zeros(%)", "Mean_flow"))
    print("-" * 33)

    flow_raw = data[:, :, 0]
    for hour in range(24):
        start = hour * STEPS_PER_HOUR
        # Lay tat ca cac buoc thuoc gio nay tu tat ca cac ngay
        idx = np.arange(start, T, 288)  # 288 = 24h * 12 steps/h
        # Gop tat ca buoc trong gio nay tu tat ca cac ngay
        hour_steps = []
        for d_start in range(start, T, 288):
            for s in range(STEPS_PER_HOUR):
                t = d_start + s
                if t < T:
                    hour_steps.append(t)
        hour_steps = np.array(hour_steps)
        hour_data = flow_raw[hour_steps]
        z_pct = (hour_data == 0).mean() * 100
        m = hour_data.mean()
        print("{:>5}h {:>11.2f}% {:>12.2f}".format(hour, z_pct, m))

    # 6. Phan tich missing per sensor (Top 10 sensor nhieu zero nhat) 
    print("\n[MISSING PER SENSOR - flow (Top 10 cao nhat)]")
    zeros_per_sensor = (flow_raw == 0).mean(axis=0) * 100
    top10_idx = np.argsort(zeros_per_sensor)[::-1][:10]
    print("{:>8} {:>12}".format("SensorID", "Zeros(%)"))
    print("-" * 25)
    for idx in top10_idx:
        print("{:>8} {:>11.2f}%".format(idx, zeros_per_sensor[idx]))

    total_clean = (zeros_per_sensor == 0).sum()
    problem_sensors = (zeros_per_sensor > 5).sum()
    print("\nTong sensor sach (0% zero): {:,}".format(total_clean))
    print("Sensor co >5% zero (co van de): {:,} / {:,}".format(problem_sensors, N))

    # 7. So samples sau sliding window 
    in_steps, out_steps = 12, 12
    n_samples = T - in_steps - out_steps + 1
    n_train = int(n_samples * 0.7)
    n_val   = int(n_samples * 0.1)
    n_test  = n_samples - n_train - n_val

    print("\nSliding window: in={} steps, out={} steps".format(in_steps, out_steps))
    print("Tong samples : {:,}".format(n_samples))
    print("Train (70%)  : {:,}".format(n_train))
    print("Val   (10%)  : {:,}".format(n_val))
    print("Test  (20%)  : {:,}".format(n_test))

    # 8. Kiem tra distance.csv 
    if os.path.exists(dist_path):
        dist_df = pd.read_csv(dist_path)
        print("\n[distance.csv] Shape: {}".format(dist_df.shape))
        print("  Columns: {}".format(list(dist_df.columns)))
        print("  Khoang cach min: {:.1f}".format(dist_df.iloc[:, 2].min()))
        print("  Khoang cach max: {:.1f}".format(dist_df.iloc[:, 2].max()))
        print("  So canh (edges): {:,}".format(len(dist_df)))
    else:
        print("\nKhong tim thay distance.csv tai " + dist_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kham pha du lieu PeMS")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["PEMS04", "PEMS08", "pems04", "pems08"],
                        help="Ten dataset: PEMS04 hoac PEMS08")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Thu muc goc chua du lieu")
    args = parser.parse_args()
    explore(args.dataset, args.data_root)
