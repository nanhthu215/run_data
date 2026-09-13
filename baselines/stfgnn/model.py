# -*- coding: utf-8 -*-
"""
STFGNN — Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting
Paper: Mengzhang Li & Zhanxing Zhu, AAAI 2021.
Official Paper: "Spatial-Temporal Fusion Graph Neural Networks for Traffic Flow Forecasting"

Architecture faithfully implementing the original paper:
1. Localized Spatial-Temporal Graph:
   - Construct (3N x 3N) super-graph fusing spatial topology (A_S) across consecutive time steps (t-1, t, t+1)
     with cross-time self-identity links (A_TC).
2. Dynamic Time Warping (DTW) Temporal Graph:
   - Construct (3N x 3N) super-graph fusing DTW-based temporal similarity graph (A_T)
     across consecutive time steps with cross-time identity links.
3. Spatial-Temporal Fusion Module (STFLayer):
   - Localized Spatial Graph Convolution on A_SG (3N x 3N)
   - Localized Temporal Graph Convolution on A_TG (3N x 3N)
   - Gated Fusion Mechanism: H_fused = H_S * sigmoid(H_T) + H_T * sigmoid(H_S)
   - Parallel Gated CNN (Dilated Temporal 1D Convolution with GLU)
   - Fusion of GNN and Gated CNN + Residual Connection + LayerNorm
4. Multi-layer Stacking + Output Generation Head to multi-step forecast (horizon=12).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    """
    Symmetric normalization D^{-1/2} A D^{-1/2}.
    """
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0.0).astype(np.float32)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return torch.tensor(normalized, dtype=torch.float32)


def construct_localized_st_graph(adj: np.ndarray, steps: int = 3) -> torch.Tensor:
    """
    Construct Localized Spatial-Temporal Graph of size (steps * N, steps * N).
    
    Structure for steps=3:
    [  A   I_N   0  ]
    [ I_N   A   I_N ]
    [  0   I_N   A  ]
    
    where A is the intra-timestep adjacency (A_S or A_T), and I_N represents
    inter-timestep self-connections across adjacent time slices.
    """
    N = adj.shape[0]
    A_local = np.zeros((N * steps, N * steps), dtype=np.float32)

    # 1. Diagonal blocks: intra-time connections
    for i in range(steps):
        A_local[i * N : (i + 1) * N, i * N : (i + 1) * N] = adj

    # 2. Off-diagonal adjacent blocks: inter-time identity connections (A_TC)
    for i in range(steps - 1):
        A_local[i * N : (i + 1) * N, (i + 1) * N : (i + 2) * N] = np.eye(N, dtype=np.float32)
        A_local[(i + 1) * N : (i + 2) * N, i * N : (i + 1) * N] = np.eye(N, dtype=np.float32)

    return normalize_adj(A_local)


class LocalizedGraphConv(nn.Module):
    """
    Graph convolution on a localized (3N x 3N) graph:
      H' = A_local * X * W + b
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x: torch.Tensor, A_local: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B_total, 3N, in_dim)
            A_local: (3N, 3N)
        Returns:
            out: (B_total, 3N, out_dim)
        """
        # A_local @ x -> (3N, 3N) x (B_total, 3N, in_dim) -> (B_total, 3N, in_dim)
        h = torch.einsum("mn,bni->bmi", A_local, x)
        return self.W(h)


class SpatialTemporalFusionLayer(nn.Module):
    """
    Core STFGNN Layer:
    - Splits temporal sequence into sliding windows of size 3 (steps=3).
    - Runs Localized Spatial GCN on A_SG (3N x 3N).
    - Runs Localized Temporal (DTW) GCN on A_TG (3N x 3N).
    - Gated Fusion: H_S * sigmoid(H_T) + H_T * sigmoid(H_S).
    - Extracts center time step (nodes N : 2N).
    - Parallel Gated CNN (Dilated Conv1D with GLU).
    - Residual connection + LayerNorm.
    """
    def __init__(self, hidden_dim: int, num_nodes: int, steps: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.steps = steps

        # GCN branches
        self.spatial_gcn = LocalizedGraphConv(hidden_dim, hidden_dim)
        self.temporal_gcn = LocalizedGraphConv(hidden_dim, hidden_dim)

        # Parallel Gated CNN along temporal dimension (kernel_size=3)
        self.gated_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=steps,
            padding=0,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, A_SG: torch.Tensor, A_TG: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, hidden_dim)
            A_SG: (3N, 3N)
            A_TG: (3N, 3N)
        Returns:
            out: (B, T - 2, N, hidden_dim)
        """
        B, T, N, C = x.shape
        steps = self.steps
        num_windows = T - steps + 1
        assert num_windows > 0, f"Time sequence length T={T} must be >= steps={steps}"

        # 1. Construct sliding windows across time:
        # For window w in [0 .. num_windows-1], window slice is x[:, w : w+steps, :, :]
        # Concatenate along node dimension -> (B, steps * N, C)
        windows = []
        for w in range(num_windows):
            w_slice = x[:, w : w + steps, :, :]  # (B, steps, N, C)
            w_cat = w_slice.permute(0, 1, 2, 3).reshape(B, steps * N, C)
            windows.append(w_cat)

        # Stack all windows: (B * num_windows, steps * N, C)
        X_local = torch.stack(windows, dim=1).reshape(B * num_windows, steps * N, C)

        # 2. Localized Graph Convolutions
        H_S = F.relu(self.spatial_gcn(X_local, A_SG))  # (B * num_windows, 3N, C)
        H_T = F.relu(self.temporal_gcn(X_local, A_TG))  # (B * num_windows, 3N, C)

        # 3. Gated Fusion Mechanism (from AAAI paper)
        # H_fused = H_S * sigmoid(H_T) + H_T * sigmoid(H_S)
        H_fused = H_S * torch.sigmoid(H_T) + H_T * torch.sigmoid(H_S)

        # Extract center time step (slice 1 out of 0, 1, 2)
        center_start = N
        center_end = 2 * N
        H_center = H_fused[:, center_start:center_end, :]  # (B * num_windows, N, C)
        H_center = H_center.reshape(B, num_windows, N, C)

        # 4. Parallel Gated CNN along time
        # Reshape to (B * N, C, T)
        x_perm = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        conv_out = self.gated_conv(x_perm)  # (B * N, 2*C, num_windows)
        p_val, p_gate = conv_out.chunk(2, dim=1)
        H_gated_cnn = p_val * torch.sigmoid(p_gate)  # (B * N, C, num_windows)
        H_gated_cnn = H_gated_cnn.reshape(B, N, C, num_windows).permute(0, 3, 1, 2)  # (B, num_windows, N, C)

        # 5. Residual connection with center slice of input
        # Input center slice: from index 1 to T-1
        X_center_res = x[:, 1 : 1 + num_windows, :, :]  # (B, num_windows, N, C)

        # Combine GNN fusion + Gated CNN + Residual + LayerNorm
        out = self.norm(X_center_res + H_center + H_gated_cnn)
        return out


class STFGNN(nn.Module):
    """
    Full STFGNN Model (AAAI 2021).
    """
    def __init__(
        self,
        num_nodes: int,
        adj_spatial: np.ndarray,
        adj_temporal: np.ndarray,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 3,
        horizon: int = 12,
        seq_len: int = 12,
        steps: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.seq_len = seq_len
        self.steps = steps
        self.num_layers = num_layers

        # Precompute and register localized (3N x 3N) graphs
        A_SG = construct_localized_st_graph(adj_spatial, steps=steps)
        A_TG = construct_localized_st_graph(adj_temporal, steps=steps)
        self.register_buffer("A_SG", A_SG)
        self.register_buffer("A_TG", A_TG)

        # Input feature projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Stack of Spatial-Temporal Fusion Layers
        self.layers = nn.ModuleList([
            SpatialTemporalFusionLayer(hidden_dim, num_nodes, steps=steps)
            for _ in range(num_layers)
        ])

        # Temporal length after num_layers with kernel steps=3:
        # Each layer reduces length by (steps - 1) = 2.
        # For seq_len=12, num_layers=3: 12 - 3*2 = 6.
        t_final = seq_len - num_layers * (steps - 1)
        if t_final <= 0:
            # Fallback if too many layers: pad internally
            t_final = 1
        self.t_final = t_final

        # Temporal aggregation head: maps remaining time steps to forecast horizon (12)
        self.temporal_proj = nn.Linear(t_final, horizon)

        # Output projection to target dimension (1)
        self.output_proj = nn.Linear(hidden_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.temporal_proj.weight)
        nn.init.zeros_(self.temporal_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, T_in=12, N, C_in=3)
        Returns:
            pred: (B, horizon=12, N, out_dim=1)
        """
        B, T, N, _ = X.shape

        # 1. Project input features
        h = self.input_proj(X)  # (B, T, N, hidden_dim)

        # 2. Pass through STFGNN layers
        for layer in self.layers:
            # If current length is smaller than steps, pad with replicate
            if h.shape[1] < self.steps:
                pad_len = self.steps - h.shape[1]
                h = F.pad(h.permute(0, 2, 3, 1), (0, pad_len), mode="replicate").permute(0, 3, 1, 2)
            h = layer(h, self.A_SG, self.A_TG)

        # 3. Temporal projection (t_final -> horizon)
        # h: (B, t_curr, N, hidden_dim)
        t_curr = h.shape[1]
        if t_curr != self.t_final:
            # Adjust if sequence length differs
            h_perm = h.permute(0, 2, 3, 1)  # (B, N, hidden, t_curr)
            h_proj = F.interpolate(h_perm, size=self.horizon, mode="linear", align_corners=False)
            h = h_proj.permute(0, 3, 1, 2)  # (B, horizon, N, hidden)
        else:
            h_perm = h.permute(0, 2, 3, 1)  # (B, N, hidden, t_final)
            h_proj = self.temporal_proj(h_perm)  # (B, N, hidden, horizon)
            h = h_proj.permute(0, 3, 1, 2)  # (B, horizon, N, hidden)

        # 4. Final output projection
        out = self.output_proj(h)  # (B, horizon, N, out_dim)
        return out


if __name__ == "__main__":
    N = 170
    adj_s = np.random.rand(N, N).astype(np.float32)
    adj_t = np.random.rand(N, N).astype(np.float32)
    model = STFGNN(num_nodes=N, adj_spatial=adj_s, adj_temporal=adj_t, hidden_dim=64, num_layers=3)
    x = torch.randn(4, 12, N, 3)
    y = model(x)
    print("STFGNN output shape:", y.shape)  # Expected: (4, 12, 170, 1)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"STFGNN parameters: {params:,}")
