
"""
DCRNN — Diffusion Convolutional Recurrent Neural Network
Kiến trúc:
  1. Diffusion Convolution: mô phỏng quá trình khuếch tán trên đồ thị có hướng
     - Forward diffusion: D_O^{-1} · A (ma trận chuyển tiếp)
     - Backward diffusion: D_I^{-1} · A^T (ma trận ngược chiều)
     - K-hop: tổng K bước khuếch tán
  2. DCGRUCell: thay Linear trong GRU bằng DiffusionConv
  3. Seq2Seq với Scheduled Sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_diffusion_matrices(adj: np.ndarray, K: int = 2) -> list:
    """
    Tính K bước diffusion (forward + backward) từ adj matrix.
    
    Args:
        adj: (N, N) adjacency matrix (có thể có trọng số)
        K: số bước khuếch tán
    
    Returns:
        list of support matrices (PyTorch tensors) — gồm both directions
    """
    N = adj.shape[0]
    
    # Forward: D_O^{-1} · A
    D_out = adj.sum(axis=1)  # out-degree
    D_out_inv = np.where(D_out > 0, 1.0 / D_out, 0.0)
    A_fwd = np.diag(D_out_inv) @ adj  # row-normalized

    # Backward: D_I^{-1} · A^T
    D_in = adj.sum(axis=0)  # in-degree
    D_in_inv = np.where(D_in > 0, 1.0 / D_in, 0.0)
    A_bwd = np.diag(D_in_inv) @ adj.T

    supports = []
    # K-hop forward
    A_k = np.eye(N)
    for _ in range(K + 1):
        supports.append(torch.tensor(A_k, dtype=torch.float32))
        A_k = A_k @ A_fwd

    # K-hop backward
    A_k = np.eye(N)
    for _ in range(1, K + 1):
        A_k = A_k @ A_bwd
        supports.append(torch.tensor(A_k, dtype=torch.float32))

    return supports


class DiffusionConv(nn.Module):
    """
    Diffusion Convolution Layer.
    Tổng có trọng số của K-hop diffusion:
      H' = sum_{k=0}^{K} (A_fwd^k · H · W_fwd_k + A_bwd^k · H · W_bwd_k)
    """
    def __init__(self, in_dim: int, out_dim: int, num_supports: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Weight matrix: (num_supports, in_dim, out_dim)
        self.W = nn.Parameter(torch.empty(num_supports, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(out_dim))
        for i in range(num_supports):
            nn.init.xavier_uniform_(self.W[i])
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            supports: list of (N, N) tensors
        Returns:
            out: (B, N, out_dim)
        """
        device = x.device
        outputs = []
        for i, A in enumerate(supports):
            A = A.to(device)
            # (N, N) × (B, N, in_dim) → (B, N, in_dim)
            h = torch.einsum("mn,bni->bmi", A, x)
            # (B, N, in_dim) × (in_dim, out_dim) → (B, N, out_dim)
            h = torch.einsum("bni,io->bno", h, self.W[i])
            outputs.append(h)
        
        out = sum(outputs) + self.b
        return out


class DCGRUCell(nn.Module):
    """
    GRU cell với Diffusion Convolution thay thế Linear.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_supports: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Reset + Update gates (output dim = 2*hidden)
        self.gate_dc = DiffusionConv(in_dim + hidden_dim, 2 * hidden_dim, num_supports)
        # Candidate hidden
        self.cand_dc = DiffusionConv(in_dim + hidden_dim, hidden_dim, num_supports)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)
            h: (B, N, hidden_dim)
            supports: list of (N, N)
        Returns:
            h_new: (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)
        gates = torch.sigmoid(self.gate_dc(xh, supports))
        r, z = gates.chunk(2, dim=-1)

        xrh = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self.cand_dc(xrh, supports))

        h_new = (1 - z) * h + z * h_cand
        return h_new


class DCRNN(nn.Module):
    """
    Full DCRNN: Seq2Seq với DCGRUCell.
    """
    def __init__(
        self,
        num_nodes: int,
        adj_matrix: np.ndarray,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        K: int = 2,
        horizon: int = 12,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.out_dim = out_dim

        # Pre-compute diffusion matrices
        supports = compute_diffusion_matrices(adj_matrix, K)
        num_supports = len(supports)
        self.register_buffer("supports_stack", torch.stack(supports))  # (num_sup, N, N)
        self.num_supports = num_supports

        # Encoder
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = in_dim if i == 0 else hidden_dim
            self.encoder_cells.append(DCGRUCell(cell_in, hidden_dim, num_supports))

        # Decoder
        self.decoder_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = out_dim if i == 0 else hidden_dim
            self.decoder_cells.append(DCGRUCell(cell_in, hidden_dim, num_supports))

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def _get_supports(self):
        return [self.supports_stack[i] for i in range(self.num_supports)]

    def _init_hidden(self, B: int, device: torch.device) -> list:
        return [
            torch.zeros(B, self.num_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def encode(self, X: torch.Tensor) -> list:
        B, T, N, C = X.shape
        device = X.device
        h = self._init_hidden(B, device)
        supports = self._get_supports()

        for t in range(T):
            x_t = X[:, t, :, :]
            for i, cell in enumerate(self.encoder_cells):
                h[i] = cell(x_t, h[i], supports)
                x_t = h[i]
        return h

    def decode(
        self,
        h: list,
        y_true: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            h: list các hidden states từ encoder
            y_true: ground-truth target (B, horizon, N, out_dim) cho teacher forcing lúc training (optional)
            teacher_forcing_ratio: xác suất sử dụng ground-truth làm decoder input khi training
        Returns:
            output: (B, horizon, N, out_dim)
        """
        B = h[0].shape[0]
        device = h[0].device
        supports = self._get_supports()

        dec_in = torch.zeros(B, self.num_nodes, self.out_dim, device=device)
        outputs = []

        for t in range(self.horizon):
            x_t = dec_in
            h_new = list(h)
            for i, cell in enumerate(self.decoder_cells):
                h_new[i] = cell(x_t, h[i], supports)
                x_t = h_new[i]
            h = h_new
            pred_t = self.output_proj(x_t)
            outputs.append(pred_t.unsqueeze(1))
            if self.training and y_true is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_in = y_true[:, t, :, :]
            else:
                dec_in = pred_t.detach()

        return torch.cat(outputs, dim=1)  # (B, horizon, N, out_dim)

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
        h = self.encode(X)
        return self.decode(h, y_true=y, teacher_forcing_ratio=teacher_forcing_ratio)


if __name__ == "__main__":
    import numpy as np
    N = 307
    adj = np.random.rand(N, N)
    adj = (adj > 0.9).astype(float)  # sparse adj
    model = DCRNN(num_nodes=N, adj_matrix=adj, in_dim=3, hidden_dim=32, num_layers=1, K=2)
    x = torch.randn(2, 12, N, 3)
    y = model(x)
    print(f"DCRNN output: {y.shape}")  # (2, 12, 307, 1)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
