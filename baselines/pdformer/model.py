"""
PDFormer — Propagation Delay-aware Dynamic Long-range Transformer
Paper: Jiawei Jiang et al., "PDFormer: Propagation Delay-Aware Dynamic 
       Long-Range Transformer for Traffic Flow Prediction", AAAI 2023.
GitHub: https://github.com/BUAABIGSCity/PDFormer

Kiến trúc:
  1. Spatial Self-Attention (SSA): attention giữa các node, 
     bias bởi short-range (hop distance) và long-range spatial dependency
  2. Temporal Self-Attention (TSA): standard multi-head attention qua thời gian
  3. Propagation Delay Bias: dùng shortest path distance để tính delay bias
  4. Stack các block + output layer

Lưu ý: Trong implementation này ta dùng shortest path distance từ adj matrix
  để tính spatial bias, thay vì exact propagation delay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


def compute_spatial_bias(adj: np.ndarray, max_dist: int = 10) -> torch.Tensor:
    """
    Tính ma trận shortest path distance từ adj matrix.
    Dùng làm spatial bias trong attention.
    
    Returns: (N, N) int tensor, clipped to [0, max_dist]
    """
    # Chuyển thành binary adj
    adj_binary = (adj > 0).astype(float)
    np.fill_diagonal(adj_binary, 0)
    
    # Shortest path
    dist = shortest_path(csr_matrix(adj_binary), method='D', directed=False)
    dist = np.clip(dist, 0, max_dist)
    dist = np.nan_to_num(dist, nan=max_dist)
    return torch.tensor(dist, dtype=torch.long)


class SpatialAttention(nn.Module):
    """
    Spatial Self-Attention với propagation delay bias.
    """
    def __init__(self, d_model: int, nhead: int, num_nodes: int, max_dist: int = 10):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Spatial bias embedding: distance → learnable bias
        self.spatial_bias_emb = nn.Embedding(max_dist + 1, nhead)

    def forward(self, x: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) — spatial attention trên từng time step
            dist_matrix: (N, N) long tensor of distances
        Returns:
            out: (B, N, d_model)
        """
        B, N, D = x.shape

        Q = self.q_proj(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)  # (B, H, N, d_h)
        K = self.k_proj(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, self.nhead, self.d_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)

        # Add spatial bias: (N, N, H) → (H, N, N)
        dist_matrix = dist_matrix.to(x.device)
        bias = self.spatial_bias_emb(dist_matrix)       # (N, N, H)
        bias = bias.permute(2, 0, 1).unsqueeze(0)       # (1, H, N, N)
        attn = attn + bias

        attn = torch.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class TemporalAttention(nn.Module):
    """
    Standard Multi-head Self-Attention theo chiều thời gian.
    """
    def __init__(self, d_model: int, nhead: int, seq_len: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        # Positional encoding cho temporal
        self.pos_emb = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — temporal attention trên từng node
        Returns:
            out: (B, T, d_model)
        """
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        out, _ = self.attn(x, x, x)
        return out


class PDFormerBlock(nn.Module):
    """
    PDFormer block: Spatial Attn → Temporal Attn → FFN
    """
    def __init__(self, d_model: int, nhead: int, num_nodes: int, seq_len: int, max_dist: int = 10):
        super().__init__()
        self.spatial_attn = SpatialAttention(d_model, nhead, num_nodes, max_dist)
        self.temporal_attn = TemporalAttention(d_model, nhead, seq_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, d_model)
            dist_matrix: (N, N)
        Returns:
            out: (B, T, N, d_model)
        """
        B, T, N, D = x.shape

        # Spatial attention: process each time step independently
        x_s = x.reshape(B * T, N, D)
        x_s = self.norm1(x_s + self.spatial_attn(x_s, dist_matrix))
        x_s = x_s.reshape(B, T, N, D)

        # Temporal attention: process each node independently
        x_t = x_s.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
        x_t = self.norm2(x_t + self.temporal_attn(x_t))
        x_t = x_t.reshape(B, N, T, D).permute(0, 2, 1, 3)   # (B, T, N, D)

        # FFN
        out = self.norm3(x_t + self.ffn(x_t))
        return out


class PDFormer(nn.Module):
    """
    Full PDFormer model.
    """
    def __init__(
        self,
        num_nodes: int,
        adj_matrix: np.ndarray,
        in_dim: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        out_dim: int = 1,
        horizon: int = 12,
        seq_len: int = 12,
        max_dist: int = 10,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon

        # Pre-compute spatial distance
        dist_matrix = compute_spatial_bias(adj_matrix, max_dist)
        self.register_buffer("dist_matrix", dist_matrix)

        # Input embedding
        self.input_emb = nn.Linear(in_dim, d_model)

        # PDFormer blocks
        self.blocks = nn.ModuleList([
            PDFormerBlock(d_model, nhead, num_nodes, seq_len, max_dist)
            for _ in range(num_layers)
        ])

        # Output: T_in → T_out
        self.temporal_proj = nn.Linear(seq_len, horizon)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, T_in=12, N, C=3)
        Returns:
            pred: (B, T_out=12, N, 1)
        """
        h = self.input_emb(X)  # (B, T, N, d_model)

        for block in self.blocks:
            h = block(h, self.dist_matrix)

        # Temporal projection
        h = h.permute(0, 2, 3, 1)          # (B, N, d_model, T_in)
        h = self.temporal_proj(h)           # (B, N, d_model, T_out)
        h = h.permute(0, 3, 1, 2)          # (B, T_out, N, d_model)

        return self.output_proj(h)           # (B, T_out, N, 1)


if __name__ == "__main__":
    N = 50  # small test
    adj = (np.random.rand(N, N) > 0.85).astype(float)
    np.fill_diagonal(adj, 0)
    model = PDFormer(num_nodes=N, adj_matrix=adj, in_dim=3, d_model=32, nhead=4, num_layers=2)
    x = torch.randn(2, 12, N, 3)
    y = model(x)
    print(f"PDFormer output: {y.shape}")  # (2, 12, 50, 1)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
