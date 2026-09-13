# -*- coding: utf-8 -*-
"""
Generate and cache Dynamic Time Warping (DTW) adjacency matrix for STFGNN baseline.
Paper: Li & Zhu, "Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting", AAAI 2021.

This script:
1. Loads traffic flow series from raw dataset (.npz).
2. Computes the average daily flow pattern (288 time steps per day) over the training split (60%).
3. Normalizes each sensor's daily curve (z-score).
4. Computes pairwise DTW distances using Sakoe-Chiba band (window=12) with multiprocessing.
5. Constructs a binary Top-k temporal graph A_T with self-loops.
6. Caches result to data/{dataset}/adj_dtw.npy.
"""

import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def fast_dtw_sakoe_chiba(s1: np.ndarray, s2: np.ndarray, w: int = 12) -> float:
    """
    Dynamic Time Warping with Sakoe-Chiba band constraint.
    s1, s2: 1D numpy arrays of length T (e.g. 288).
    w: window size (Sakoe-Chiba band width).
    """
    n, m = len(s1), len(s2)
    w = max(w, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m + 1, i + w + 1)
        for j in range(j_start, j_end):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m])


def _compute_row_dtw(args):
    """Worker function to compute DTW distances from node i to nodes j > i."""
    i, row_series, all_series, w = args
    N = len(all_series)
    row_results = []
    for j in range(i + 1, N):
        d = fast_dtw_sakoe_chiba(row_series, all_series[j], w=w)
        row_results.append((i, j, d))
    return row_results


def compute_dtw_distance_matrix(mean_patterns: np.ndarray, w: int = 12, max_workers: int = None) -> np.ndarray:
    """
    Compute full symmetric DTW distance matrix for N sensor daily patterns.
    mean_patterns: (N, 288) array of normalized mean daily curves.
    """
    N = mean_patterns.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float32)

    tasks = [(i, mean_patterns[i], mean_patterns, w) for i in range(N)]

    if max_workers == 1:
        # Single threaded
        for task in tasks:
            results = _compute_row_dtw(task)
            for i, j, d in results:
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
    else:
        # Multiprocessing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for results in executor.map(_compute_row_dtw, tasks):
                for i, j, d in results:
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

    return dist_matrix


def construct_topk_temporal_graph(dist_matrix: np.ndarray, top_k: int = 10) -> np.ndarray:
    """
    Construct binary adjacency matrix A_T from DTW distance matrix:
    - For each node i, select top-k closest nodes (smallest DTW distance).
    - Add self-loops (diagonal = 1).
    - Symmetrize: A_T = max(A_T, A_T.T).
    """
    N = dist_matrix.shape[0]
    A_T = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        # Sort distances ascending, exclude node itself
        dists = dist_matrix[i].copy()
        dists[i] = np.inf
        top_k_indices = np.argsort(dists)[:top_k]
        A_T[i, top_k_indices] = 1.0

    # Symmetrize
    A_T = np.maximum(A_T, A_T.T)
    # Add self loops
    np.fill_diagonal(A_T, 1.0)
    return A_T


def get_or_generate_dtw_graph(dataset: str, top_k: int = 10, w: int = 12, force_recompute: bool = False) -> np.ndarray:
    """
    Load cached DTW graph from disk or compute and cache it if not present.
    """
    dataset = dataset.upper()
    data_dir = os.path.join(ROOT, "data", dataset)
    cache_path = os.path.join(data_dir, f"adj_dtw_k{top_k}.npy")
    legacy_cache_path = os.path.join(data_dir, "adj_dtw.npy")

    if not force_recompute:
        if os.path.exists(cache_path):
            print(f"[DTW] Loading cached DTW adjacency matrix: {cache_path}")
            return np.load(cache_path)
        elif top_k == 10 and os.path.exists(legacy_cache_path):
            print(f"[DTW] Loading cached DTW adjacency matrix: {legacy_cache_path}")
            return np.load(legacy_cache_path)

    print(f"\n[DTW] Generating DTW adjacency matrix for {dataset} (top_k={top_k}, window={w})...")
    npz_path = os.path.join(data_dir, f"{dataset.lower()}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Raw data file not found: {npz_path}")

    raw_data = np.load(npz_path)["data"][:, :, 0]  # (Total_T, N)
    total_timesteps, num_nodes = raw_data.shape
    steps_per_day = 288
    total_days = total_timesteps // steps_per_day
    train_days = int(total_days * 0.6)

    print(f"[DTW] Dataset: {dataset} | Total days: {total_days} | Train days used for DTW: {train_days} | Nodes: {num_nodes}")

    train_data = raw_data[:train_days * steps_per_day].reshape(train_days, steps_per_day, num_nodes)
    mean_daily_pattern = train_data.mean(axis=0).T  # (num_nodes, 288)

    # Z-score normalization per sensor curve
    mean_norm = (mean_daily_pattern - mean_daily_pattern.mean(axis=1, keepdims=True)) / (
        mean_daily_pattern.std(axis=1, keepdims=True) + 1e-6
    )

    t0 = time.time()
    dist_matrix = compute_dtw_distance_matrix(mean_norm, w=w)
    elapsed = time.time() - t0
    print(f"[DTW] Pairwise DTW distances calculated in {elapsed:.2f}s.")

    # Save distance matrix for future use/analysis
    dist_cache_path = os.path.join(data_dir, "dtw_distances.npy")
    np.save(dist_cache_path, dist_matrix)

    # Construct Top-k binary adjacency matrix
    adj_dtw = construct_topk_temporal_graph(dist_matrix, top_k=top_k)
    np.save(cache_path, adj_dtw)
    if top_k == 10:
        np.save(legacy_cache_path, adj_dtw)

    n_edges = int(adj_dtw.sum())
    print(f"[DTW] Successfully generated and cached: {cache_path} ({n_edges} directed edges including self-loops).")
    return adj_dtw


def main():
    parser = argparse.ArgumentParser(description="Generate DTW Temporal Graph for STFGNN")
    parser.add_argument("--dataset", type=str, default="PEMS08", choices=["PEMS08", "PEMS04", "ALL"])
    parser.add_argument("--top-k", type=int, default=10, help="Top-k nearest neighbors in DTW space")
    parser.add_argument("--window", type=int, default=12, help="Sakoe-Chiba constraint window")
    parser.add_argument("--force", action="store_true", help="Force recomputation even if cache exists")
    args = parser.parse_args()

    datasets = ["PEMS08", "PEMS04"] if args.dataset == "ALL" else [args.dataset]
    for ds in datasets:
        adj = get_or_generate_dtw_graph(ds, top_k=args.top_k, w=args.window, force_recompute=args.force)
        print(f"Summary for {ds}: Shape {adj.shape}, Density: {adj.mean():.4f}")


if __name__ == "__main__":
    main()
