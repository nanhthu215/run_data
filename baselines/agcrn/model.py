# -*- coding: utf-8 -*-
"""
AGCRN — Adaptive Graph Convolutional Recurrent Network
Paper: Lei Bai et al., "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting", NeurIPS 2020.
Official GitHub: https://github.com/LeiBAI/AGCRN

Kiến trúc chuẩn theo NeurIPS 2020:
  1. Node Adaptive Parameter Learning (NAPL): node embeddings E ∈ R^{N × d} dùng chung toàn mạng
  2. Adaptive Graph Convolution (AGCN): A = softmax(ReLU(E · E^T)), W_node = E · W
  3. AGCRN Cell: thay thế Linear trong GRU bằng AGCN
  4. Non-Autoregressive Predictor: Chiếu trực tiếp trạng thái ẩn của Encoder sang toàn bộ 12 horizon (end_conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AGCN(nn.Module):
    """
    Adaptive Graph Convolution với node embedding (NAPL & AGCN).
    Công thức: H' = A_hat · (X · W_node) + b với A_hat = softmax(ReLU(E · E^T))
    """
    def __init__(self, in_dim: int, out_dim: int, num_nodes: int, embed_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Weight pool: (embed_dim, in_dim, out_dim) với scale 0.1 ổn định gradient
        self.W = nn.Parameter(torch.randn(embed_dim, in_dim, out_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(num_nodes, out_dim))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            node_embeddings: (N, d)
        Returns:
            out: (B, N, out_dim)
        """
        E = node_embeddings  # (N, d)
        # Tính adaptive adjacency matrix từ node embeddings dùng chung
        A = torch.softmax(F.relu(E @ E.T), dim=-1)  # (N, N)

        # Node-specific weight matrix: W_node[i] = E[i] · W (NAPL projection)
        W_node = torch.einsum("nd,dio->nio", E, self.W)  # (N, in_dim, out_dim)

        # Graph conv: H' = A · (x · W_node) + b
        support = torch.einsum("bni,nio->bno", x, W_node)   # (B, N, out_dim)
        output  = torch.einsum("mn,bno->bmo", A, support)    # (B, N, out_dim)
        output  = output + self.b                             # (B, N, out_dim)
        return output


class AGCRNCell(nn.Module):
    """
    GRU Cell với AGCN thay thế các phép nhân ma trận tuyến tính.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_nodes: int, embed_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # Reset gate (r) và Update gate (z): concatenate input + hidden → 2*hidden
        self.gate_agcn = AGCN(in_dim + hidden_dim, 2 * hidden_dim, num_nodes, embed_dim)
        # Candidate hidden state
        self.cand_agcn = AGCN(in_dim + hidden_dim, hidden_dim, num_nodes, embed_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            h: (B, N, hidden_dim)
            node_embeddings: (N, d)
        Returns:
            h_new: (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)  # (B, N, in_dim + hidden_dim)

        gates = torch.sigmoid(self.gate_agcn(xh, node_embeddings))  # (B, N, 2*hidden_dim)
        r, z = gates.chunk(2, dim=-1)                                # each (B, N, hidden_dim)

        # Candidate hidden với reset gate
        xrh = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self.cand_agcn(xrh, node_embeddings))   # (B, N, hidden_dim)

        h_new = (1 - z) * h + z * h_cand
        return h_new


class AGCRN(nn.Module):
    """
    Full AGCRN model (Chuẩn 100% theo Bai et al., NeurIPS 2020):
    - Shared Node Embeddings E ∈ R^{N × d}
    - Multi-layer AGCRNCell Encoder (xử lý chuỗi lịch sử 12 time step)
    - Direct Multi-Horizon Predictor: end_conv chiếu trực tiếp sang toàn bộ 12 horizon
    """
    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 3,       # số features (flow, speed, occ)
        hidden_dim: int = 64,
        out_dim: int = 1,      # dự báo flow
        num_layers: int = 2,
        embed_dim: int = 10,
        horizon: int = 12,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Node Embeddings dùng chung toàn mô hình (Shared Node Embeddings E)
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))

        # Encoder layers
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in_dim = in_dim if i == 0 else hidden_dim
            self.encoder_cells.append(AGCRNCell(cell_in_dim, hidden_dim, num_nodes, embed_dim))

        # Direct Multi-Horizon Predictor (Chuẩn bài báo NeurIPS 2020: end_conv)
        self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, hidden_dim), bias=True)

    def _init_hidden(self, batch_size: int, device: torch.device) -> list:
        return [
            torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            X: (B, T_in=12, N, C=3)
        Returns:
            pred: (B, T_out=12, N, 1)
        """
        B, T, N, C = X.shape
        device = X.device
        h = self._init_hidden(B, device)

        # 1. Chạy qua các bước thời gian của Encoder
        for t in range(T):
            x_t = X[:, t, :, :]  # (B, N, C)
            for layer_i, cell in enumerate(self.encoder_cells):
                h[layer_i] = cell(x_t, h[layer_i], self.node_embeddings)
                x_t = h[layer_i]

        # 2. Lấy hidden state tầng trên cùng ở bước cuối: (B, 1, N, hidden_dim)
        h_last = h[-1].unsqueeze(1)

        # 3. Chiếu trực tiếp sang toàn bộ 12 horizon dự báo (Non-autoregressive)
        out = self.end_conv(h_last).squeeze(-1)  # (B, horizon * out_dim, N)
        out = out.view(B, self.horizon, self.out_dim, N)
        out = out.permute(0, 1, 3, 2)            # (B, horizon, N, out_dim)

        return out


if __name__ == "__main__":
    # Quick test
    B, T, N, C = 4, 12, 307, 3
    model = AGCRN(num_nodes=N, in_dim=C, hidden_dim=64, embed_dim=10, num_layers=2)
    x = torch.randn(B, T, N, C)
    y = model(x)
    print(f"AGCRN output: {y.shape}")  # (4, 12, 307, 1)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
