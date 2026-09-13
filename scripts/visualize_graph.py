# -*- coding: utf-8 -*-
"""
visualize_graph.py -- Truc quan hoa do thi mang luoi cam bien giao thong (PeMS04, PeMS08)

Chay:
  run.bat scripts/visualize_graph.py --dataset PEMS04
  run.bat scripts/visualize_graph.py --dataset PEMS08
  run.bat scripts/visualize_graph.py --all

Output:
  Bao gom 4 bieu do:
    1. Heatmap ma tran ke (Adjacency Matrix Heatmap)
    2. Mang luoi do thi khong gian (Graph Network Topology)
    3. Phan bo trong so canh Gaussian (Edge Weight Distribution)
    4. Phan bo bac cua cac nut (Node Degree Distribution)
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import networkx as nx


class GraphVisualizer:

    def __init__(self, dataset: str, data_root: str = "data", output_dir: str = "results/figures"):
        self.dataset = dataset.upper()
        self.data_root = data_root
        self.output_dir = output_dir

        self.processed_dir = os.path.join(self.data_root, self.dataset, "processed")
        self.adj_path = os.path.join(self.processed_dir, "adj_matrix.npy")
        self.edge_index_path = os.path.join(self.processed_dir, "edge_index.npy")
        self.edge_weight_path = os.path.join(self.processed_dir, "edge_weight.npy")

        os.makedirs(self.output_dir, exist_ok=True)

    def load_graph_data(self):
        assert os.path.exists(self.adj_path), (
            f"Khong tim thay {self.adj_path}. "
            "Hay chay prepare_pems_data.py truoc de tao do thi!"
        )
        adj = np.load(self.adj_path)
        edge_index = np.load(self.edge_index_path)
        edge_weight = np.load(self.edge_weight_path)
        return adj, edge_index, edge_weight

    def plot_and_save(self, dpi: int = 300):
        adj, edge_index, edge_weight = self.load_graph_data()
        n_nodes = adj.shape[0]
        n_edges = edge_weight.shape[0]
        density = (adj > 0).sum() / (n_nodes * n_nodes) * 100
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for idx in range(n_edges):
            u = int(edge_index[0, idx])
            v = int(edge_index[1, idx])
            w = float(edge_weight[idx])
            G.add_edge(u, v, weight=w)

        degrees = [d for _, d in G.degree()]
        avg_degree = np.mean(degrees)

        print(f"\n{'='*60}")
        print(f"  TRUC QUAN HOA DO THI -- {self.dataset}")
        print(f"{'='*60}")
        print(f"  So nut (Nodes)      : {n_nodes}")
        print(f"  So canh co trong so : {n_edges}")
        print(f"  Mat do do thi       : {density:.2f}%")
        print(f"  Bac trung binh      : {avg_degree:.2f}")

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        plt.subplots_adjust(hspace=0.28, wspace=0.25)
        fig.suptitle(
            f"Traffic Sensor Network Graph Analysis -- {self.dataset}\n"
            f"(Nodes: {n_nodes}, Edges: {n_edges}, Density: {density:.2f}%, Avg Degree: {avg_degree:.1f})",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # 1. Heatmap ma tran ke (Adjacency Matrix Heatmap)
        ax1 = axes[0, 0]
        im = ax1.imshow(adj, cmap="viridis", interpolation="nearest", aspect="auto")
        ax1.set_title("1. Adjacency Matrix Heatmap (Gaussian Kernel)", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Sensor Node Index", fontsize=11)
        ax1.set_ylabel("Sensor Node Index", fontsize=11)
        cbar1 = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Edge Weight (Connection Strength)", fontsize=10)

        # 2. Mang luoi do thi khong gian (Network Topology)
        ax2 = axes[0, 1]
        ax2.set_title(f"2. Sensor Network Topology (Spring Layout)", fontsize=13, fontweight="bold")
        pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
        node_colors = [d for d in degrees]
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax2,
            node_size=35,
            node_color=node_colors,
            cmap="plasma",
            alpha=0.85,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax2,
            alpha=0.2,
            edge_color="gray",
            width=0.8,
        )
        ax2.axis("off")

        # 3. Phan bo trong so canh (Edge Weight Distribution)
        ax3 = axes[1, 0]
        ax3.hist(edge_weight, bins=35, color="#2b5c8f", edgecolor="white", alpha=0.85)
        ax3.set_title("3. Gaussian Edge Weight Distribution", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Weight $w_{ij} = \exp(-d^2 / \sigma^2)$", fontsize=11)
        ax3.set_ylabel("Frequency (Edge Count)", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)

        stats_text = (
            f"Min weight: {edge_weight.min():.4f}\n"
            f"Max weight: {edge_weight.max():.4f}\n"
            f"Mean: {edge_weight.mean():.4f}\n"
            f"Std : {edge_weight.std():.4f}"
        )
        ax3.text(
            0.95,
            0.95,
            stats_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5),
        )

        # 4. Phan bo bac cua nut (Node Degree Distribution)
        ax4 = axes[1, 1]
        ax4.hist(degrees, bins=range(0, max(degrees) + 2), color="#e27d60", edgecolor="white", alpha=0.85)
        ax4.axvline(avg_degree, color="red", linestyle="dashed", linewidth=1.5, label=f"Mean: {avg_degree:.1f}")
        ax4.set_title("4. Node Degree Distribution (Neighbors <= 2km)", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Degree (Number of Connected Neighbors)", fontsize=11)
        ax4.set_ylabel("Number of Sensors", fontsize=11)
        ax4.legend(loc="upper right", fontsize=10)
        ax4.grid(True, linestyle="--", alpha=0.5)

        output_file = os.path.join(self.output_dir, f"graph_{self.dataset.lower()}.png")
        plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"  [SUCCESS] Da luu anh truc quan hoa tai: {output_file}")
        print(f"{'='*60}\n")
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Truc quan hoa do thi giao thong PeMS")
    parser.add_argument(
        "--dataset",
        type=str,
        default="PEMS04",
        choices=["PEMS04", "PEMS08", "pems04", "pems08"],
        help="Dataset can truc quan hoa (default: PEMS04)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ve va luu anh cho ca hai tap PEMS04 va PEMS08",
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--dpi", type=int, default=300, help="Do phan giai anh (default: 300)")

    args = parser.parse_args()

    if args.all:
        datasets = ["PEMS04", "PEMS08"]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        viz = GraphVisualizer(dataset=ds, data_root=args.data_root, output_dir=args.output_dir)
        viz.plot_and_save(dpi=args.dpi)


if __name__ == "__main__":
    main()
