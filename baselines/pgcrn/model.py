# -*- coding: utf-8 -*-
"""
PGCRN — Patch-based Graph Convolutional Recurrent Network
Paper: Rao, X., Shang, S., Jiang, R., Chen, L., Han, P. (2025). 
       "Traffic forecasting with patch-based graph convolutional recurrent network." 
       GeoInformatica, 29, 691-720.

Kiến trúc:
  1. Patch Embedding: Chia chuỗi 12 time step thành 12/patch_len = 4 patch
     (mỗi patch gộp patch_len time step liên tiếp thành vector đặc trưng)
  2. Graph Convolution (GraphConv): GCN dùng ma trận kề CỐ ĐỊNH chuẩn hóa D^{-1/2} A D^{-1/2}
  3. GCRCell (Graph Convolutional Recurrent Cell): GRU Cell tích hợp GraphConv
  4. Encoder-Decoder với Teacher Forcing cho các patch
  5. Un-patch / Output Projection: Chiếu các patch dự báo trở lại chuỗi 12 time step
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2}."""
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0.0).astype(np.float32)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return torch.tensor(normalized_adj, dtype=torch.float32)


class GraphConv(nn.Module):
    """
    Graph Convolution Layer dùng ma trận kề cố định:
      H' = A_norm · H · W + b
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            A: (N, N)
        Returns:
            out: (B, N, out_dim)
        """
        # (N, N) x (B, N, in_dim) -> (B, N, in_dim)
        h = torch.einsum("mn,bni->bmi", A, x)
        # (B, N, in_dim) x (in_dim, out_dim) -> (B, N, out_dim)
        out = torch.einsum("bni,io->bno", h, self.W) + self.b
        return out


class GCRCell(nn.Module):
    """
    GRU Cell với Graph Convolution cố định thay thế Linear.
    """
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Reset (r) và Update (z) gates: in_dim + hidden_dim -> 2 * hidden_dim
        self.gate_gcn = GraphConv(in_dim + hidden_dim, 2 * hidden_dim)
        # Candidate hidden: in_dim + hidden_dim -> hidden_dim
        self.cand_gcn = GraphConv(in_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            h: (B, N, hidden_dim)
            A: (N, N)
        Returns:
            h_new: (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)

        gates = torch.sigmoid(self.gate_gcn(xh, A))
        r, z = gates.chunk(2, dim=-1)

        xrh = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self.cand_gcn(xrh, A))

        h_new = (1 - z) * h + z * h_cand
        return h_new


class PGCRN(nn.Module):
    """
    Patch-based Graph Convolutional Recurrent Network.
    """
    def __init__(
        self,
        num_nodes: int,
        adj_matrix: np.ndarray,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        horizon: int = 12,
        patch_len: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.patch_len = patch_len

        assert horizon % patch_len == 0, f"Horizon {horizon} phải chia hết cho patch_len {patch_len}"
        self.num_patches_in = 12 // patch_len
        self.num_patches_out = horizon // patch_len

        # Ma trận kề cố định chuẩn hóa (register buffer để lưu cùng model và theo device)
        A_norm = normalize_adj(adj_matrix)
        self.register_buffer("A", A_norm)

        # Patch embedding: patch_len * in_dim -> hidden_dim
        self.enc_patch_proj = nn.Linear(patch_len * in_dim, hidden_dim)
        # Decoder patch embedding: patch_len * out_dim -> hidden_dim
        self.dec_patch_proj = nn.Linear(patch_len * out_dim, hidden_dim)

        # Encoder GCR cells
        self.encoder_cells = nn.ModuleList([
            GCRCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Decoder GCR cells
        self.decoder_cells = nn.ModuleList([
            GCRCell(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Un-patch / Output projection: hidden_dim -> patch_len * out_dim
        self.output_proj = nn.Linear(hidden_dim, patch_len * out_dim)

    def _init_hidden(self, batch_size: int, device: torch.device) -> list:
        return [
            torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def encode(self, X_patches: torch.Tensor) -> list:
        """
        Args:
            X_patches: (B, num_patches_in, N, hidden_dim)
        Returns:
            hidden_states: list of (B, N, hidden_dim)
        """
        B, P, N, _ = X_patches.shape
        h = self._init_hidden(B, X_patches.device)

        for p in range(P):
            x_p = X_patches[:, p, :, :]  # (B, N, hidden_dim)
            for layer_i, cell in enumerate(self.encoder_cells):
                h[layer_i] = cell(x_p, h[layer_i], self.A)
                x_p = h[layer_i]
        return h

    def decode(
        self,
        h: list,
        y_patches: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Decoder xử lý từng patch với teacher forcing.
        Args:
            h: list of encoder hidden states
            y_patches: (B, num_patches_out, N, patch_len * out_dim)
            teacher_forcing_ratio: xác suất dùng ground-truth patch khi training
        Returns:
            pred_patches: (B, num_patches_out, N, patch_len * out_dim)
        """
        B = h[0].shape[0]
        device = h[0].device

        # Patch đầu vào ban đầu cho
        #  decoder (GO token = zeros)
        dec_in = torch.zeros(B, self.num_nodes, self.patch_len * self.out_dim, device=device)
        pred_patches = []

        for p in range(self.num_patches_out):
            # Input dự báo patch p được chiếu từ dec_in (zeros ở p=0, hoặc patch p-1 trước đó)
            x_p = self.dec_patch_proj(dec_in)  # (B, N, hidden_dim)

            h_new = list(h)
            for layer_i, cell in enumerate(self.decoder_cells):
                h_new[layer_i] = cell(x_p, h[layer_i], self.A)
                x_p = h_new[layer_i]
            h = h_new

            pred_p = self.output_proj(x_p)  # (B, N, patch_len * out_dim)
            pred_patches.append(pred_p.unsqueeze(1))

            # Scheduled Sampling / Teacher Forcing cho patch TIẾP THEO (p + 1):
            # Tuyệt đối không rò rỉ target của patch hiện tại vào chính nó!
            if self.training and y_patches is not None and random.random() < teacher_forcing_ratio:
                dec_in = y_patches[:, p, :, :]
            else:
                dec_in = pred_p.detach()

        return torch.cat(pred_patches, dim=1)  # (B, num_patches_out, N, patch_len * out_dim)

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            X: (B, T_in=12, N, C=3)
            y: (B, T_out=12, N, 1) [optional target cho teacher forcing]
            teacher_forcing_ratio: float
        Returns:
            pred: (B, T_out=12, N, 1)
        """
        B, T_in, N, C_in = X.shape

        # 1. Patchify input: (B, T_in, N, C) -> (B, num_patches, N, patch_len * C)
        X_reshaped = X.view(B, self.num_patches_in, self.patch_len, N, C_in)
        X_reshaped = X_reshaped.permute(0, 1, 3, 2, 4).contiguous()
        X_patches_raw = X_reshaped.view(B, self.num_patches_in, N, self.patch_len * C_in)
        X_patches = self.enc_patch_proj(X_patches_raw)  # (B, P_in, N, hidden_dim)

        # 2. Patchify target y nếu có
        y_patches_raw = None
        if y is not None:
            B_y, T_out, N_y, C_out = y.shape
            y_reshaped = y.view(B_y, self.num_patches_out, self.patch_len, N_y, C_out)
            y_reshaped = y_reshaped.permute(0, 1, 3, 2, 4).contiguous()
            y_patches_raw = y_reshaped.view(B_y, self.num_patches_out, N_y, self.patch_len * C_out)

        # 3. Encode & Decode
        h = self.encode(X_patches)
        pred_patches = self.decode(h, y_patches_raw, teacher_forcing_ratio)

        # 4. Un-patch output: (B, num_patches_out, N, patch_len * out_dim) -> (B, T_out, N, out_dim)
        pred_unfolded = pred_patches.view(B, self.num_patches_out, N, self.patch_len, self.out_dim)
        pred_unfolded = pred_unfolded.permute(0, 1, 3, 2, 4).contiguous()
        pred = pred_unfolded.view(B, self.horizon, N, self.out_dim)

        return pred


if __name__ == "__main__":
    B, T, N, C = 4, 12, 170, 3
    dummy_adj = np.random.rand(N, N).astype(np.float32)
    dummy_adj = (dummy_adj > 0.8).astype(np.float32)

    model = PGCRN(num_nodes=N, adj_matrix=dummy_adj, in_dim=C, hidden_dim=64, out_dim=1, num_layers=2)
    x = torch.randn(B, T, N, C)
    y = torch.randn(B, T, N, 1)

    # Test inference mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"[Eval] PGCRN output shape: {out_eval.shape}")

    # Test training mode with teacher forcing
    model.train()
    out_train = model(x, y=y, teacher_forcing_ratio=0.5)
    print(f"[Train] PGCRN output shape: {out_train.shape}")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
