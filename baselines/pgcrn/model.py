# -*- coding: utf-8 -*-
"""
PGCRN — Patch-based Graph Convolutional Recurrent Network (Đầy đủ theo bài báo)
Paper: Rao, X., Shang, S., Jiang, R., Chen, L., Han, P. (2025).
       "Traffic forecasting with patch-based graph convolutional recurrent network."
       GeoInformatica, 29, 691-720.
Official code: https://github.com/kevin-xuan/PGCRN

Kiến trúc đầy đủ:
  1. Chebyshev GCN: K-hop convolution, hỗ trợ multi-support (static 2D + dynamic 3D)
  2. GCRCell: GRU Cell tích hợp Chebyshev GCN
  3. PGCRN_Encoder / PGCRN_Decoder: Encoder dùng static graph, Decoder dùng static + dynamic
  4. Time-Aware Dynamic Graph Learning: hypernet(h_t) → softmax(ReLU(E·E^T))
  5. Contrastive Self-Supervised Learning: FFT→GLU→IFFT augmentation + InfoNCE loss
  6. Scheduled Sampling: Exponential decay (cl_decay_steps)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: np.ndarray) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} (A + I) D^{-1/2}."""
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.where(row_sum > 0, 1.0 / np.sqrt(row_sum), 0.0).astype(np.float32)
    d_mat = np.diag(d_inv_sqrt)
    return torch.tensor(d_mat.dot(adj).dot(d_mat), dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# GLU (Gated Linear Unit) cho spectral augmentation
# ──────────────────────────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit: x_left * σ(x_right)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_left = nn.Linear(in_dim, out_dim)
        self.linear_right = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear_left(x) * torch.sigmoid(self.linear_right(x))


# ──────────────────────────────────────────────────────────────
# Chebyshev GCN — hỗ trợ cả static (N,N) và dynamic (B,N,N)
# ──────────────────────────────────────────────────────────────
class GCN(nn.Module):
    """
    Chebyshev Graph Convolution với multi-support.
    Mỗi support được mở rộng thành K Chebyshev polynomials T_0, T_1, ..., T_{K-1}.
    Hỗ trợ:
      - Static support: (N, N) — shared across batch
      - Dynamic support: (B, N, N) — per-sample adjacency
    """
    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, num_support: int):
        super().__init__()
        self.cheb_k = cheb_k
        self.num_support = num_support
        self.weights = nn.Parameter(torch.empty(num_support * cheb_k * dim_in, dim_out))
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x: torch.Tensor, supports: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim_in)
            supports: list of tensors — each (N, N) hoặc (B, N, N)
        Returns:
            out: (B, N, dim_out)
        """
        x_g = []
        for support in supports:
            if support.dim() == 2:
                # ── Static graph (N, N): Recurrence directly on features ──
                # T_0(A) x = x
                x_0 = x
                # T_1(A) x = A x
                x_1 = torch.matmul(support, x)
                x_g.extend([x_0, x_1])
                for _ in range(2, self.cheb_k):
                    x_2 = 2.0 * torch.matmul(support, x_1) - x_0
                    x_g.append(x_2)
                    x_0, x_1 = x_1, x_2
            else:
                # ── Dynamic graph (B, N, N): Batched recurrence ──
                # T_0(A) x = x
                x_0 = x
                # T_1(A) x = A x (via bmm)
                x_1 = torch.bmm(support, x)
                x_g.extend([x_0, x_1])
                for _ in range(2, self.cheb_k):
                    x_2 = 2.0 * torch.bmm(support, x_1) - x_0
                    x_g.append(x_2)
                    x_0, x_1 = x_1, x_2

        x_g = torch.cat(x_g, dim=-1)  # (B, N, num_support * cheb_k * dim_in)
        return torch.matmul(x_g, self.weights) + self.bias



# ──────────────────────────────────────────────────────────────
# GCRCell — GRU + Chebyshev GCN
# ──────────────────────────────────────────────────────────────
class GCRCell(nn.Module):
    """
    Graph Convolutional Recurrent Cell.
    GRU gates được thay thế bằng Chebyshev GCN.
    """
    def __init__(self, node_num: int, dim_in: int, dim_out: int,
                 cheb_k: int, num_support: int):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        gcn_dim = dim_in + dim_out
        self.gate = GCN(gcn_dim, 2 * dim_out, cheb_k, num_support)
        self.update = GCN(gcn_dim, dim_out, cheb_k, num_support)

    def forward(self, x: torch.Tensor, state: torch.Tensor,
                supports: list) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim_in)
            state: (B, N, hidden_dim)
            supports: list of graph tensors
        Returns:
            h_new: (B, N, hidden_dim)
        """
        state = state.to(x.device)
        xh = torch.cat([x, state], dim=-1)
        z_r = torch.sigmoid(self.gate(xh, supports))
        z, r = z_r.chunk(2, dim=-1)
        candidate = torch.cat([x, z * state], dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        return r * state + (1 - r) * hc

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


# ──────────────────────────────────────────────────────────────
# Encoder / Decoder
# ──────────────────────────────────────────────────────────────
class PGCRN_Encoder(nn.Module):
    """Encoder: xử lý patches tuần tự qua GCRCell stack."""
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.cells = nn.ModuleList()
        self.cells.append(GCRCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.cells.append(GCRCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, x, init_state, supports):
        """
        Args:
            x: (B, P, N, D) — P patches
            init_state: list of (B, N, D) per layer
            supports: list of graph tensors
        Returns:
            h_en: (B, P, N, D) — outputs of last layer
            output_hidden: list of (B, N, D) — last state per layer
        """
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.cells[i](current_inputs[:, t], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        return [cell.init_hidden_state(batch_size) for cell in self.cells]


class PGCRN_Decoder(nn.Module):
    """Decoder: xử lý từng patch một (autoregressive)."""
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super().__init__()
        self.rnn_layers = rnn_layers
        self.cells = nn.ModuleList()
        self.cells.append(GCRCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, rnn_layers):
            self.cells.append(GCRCell(node_num, dim_out, dim_out, cheb_k, num_support))

    def forward(self, xt, init_state, supports):
        """
        Args:
            xt: (B, N, D) — single patch input
            init_state: list of (B, N, D) per layer
        Returns:
            h_de: (B, N, D) — output of last layer
            output_hidden: list of (B, N, D) — updated states
        """
        current_inputs = xt
        output_hidden = []
        for i in range(self.rnn_layers):
            state = self.cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


# ──────────────────────────────────────────────────────────────
# PGCRN — Mô hình đầy đủ
# ──────────────────────────────────────────────────────────────
class PGCRN(nn.Module):
    """
    Patch-based Graph Convolutional Recurrent Network.

    Đóng góp chính so với GCRN cơ sở:
      1. Patch-based processing: gộp patch_len time steps → 1 token
      2. Dynamic Graph Learning: tính A(t) từ hidden state qua hypernet
      3. Contrastive Learning: FFT augmentation + InfoNCE loss
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
        patch_len: int = 2,
        cheb_k: int = 3,
        hyper_dim: int = 10,
        dynamic: bool = True,
        contra: bool = True,
        glu_layers: int = 2,
        cl_decay_steps: int = 560,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.patch_len = patch_len
        self.cheb_k = cheb_k
        self.dynamic = dynamic
        self.contra = contra
        self.cl_decay_steps = cl_decay_steps
        self.glu_layers = glu_layers
        self.batches_seen = 0  # Global batch counter cho scheduled sampling

        assert horizon % patch_len == 0, \
            f"Horizon {horizon} phải chia hết cho patch_len {patch_len}"
        self.num_patches = horizon // patch_len

        # ── Static adjacency matrix ──
        A_norm = normalize_adj(adj_matrix)
        self.register_buffer("A_static", A_norm)

        # ── Patch embedding ──
        self.enc_patch_proj = nn.Linear(patch_len * in_dim, hidden_dim)
        self.dec_patch_proj = nn.Linear(patch_len * out_dim, hidden_dim)

        # ── Encoder: chỉ dùng static support ──
        num_support_en = 1
        self.encoder = PGCRN_Encoder(
            num_nodes, hidden_dim, hidden_dim, cheb_k,
            num_layers, num_support_en
        )

        # ── Decoder: static + dynamic (+ contra dynamic) ──
        num_support_de = 1
        if dynamic:
            num_support_de += 1
        if dynamic and contra:
            num_support_de += 1
        self.decoder = PGCRN_Decoder(
            num_nodes, hidden_dim, hidden_dim, cheb_k + 1,
            num_layers, num_support_de
        )

        # ── Output projection ──
        self.out_proj = nn.Linear(hidden_dim, patch_len * out_dim)

        # ── Dynamic Graph Learning ──
        if dynamic:
            self.hypernet = nn.Linear(hidden_dim, hyper_dim)

        # ── Contrastive Learning ──
        if contra:
            assert dynamic, "Contrastive learning yêu cầu dynamic=True"
            # Fuse time + frequency representations
            self.decoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            # GLU layers cho FFT augmentation
            self.GLUs = nn.ModuleList()
            for _ in range(2 * glu_layers):
                self.GLUs.append(GLU(horizon, horizon))

    # ── Scheduled Sampling ──
    def compute_sampling_threshold(self) -> float:
        """Exponential decay scheduled sampling (bài báo gốc, Eq. in Section 4.3)."""
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(self.batches_seen / self.cl_decay_steps)
        )

    # ── Spectral Augmentation ──
    def spe_seq_cell(self, flow: torch.Tensor) -> torch.Tensor:
        """
        FFT → GLU filtering → IFFT tạo augmented view trong miền tần số.
        Args:
            flow: (B, N, T) — chuỗi flow gốc
        Returns:
            augmented: (B, N, T) — chuỗi flow đã augment
        """
        ffted = torch.fft.fft(flow, dim=-1)
        real = ffted.real
        imag = ffted.imag
        for i in range(self.glu_layers):
            real = self.GLUs[i * 2](real)
            imag = self.GLUs[i * 2 + 1](imag)
        return torch.fft.ifft(torch.complex(real, imag), dim=-1).real

    # ── Contrastive Loss ──
    def calculate_contra_loss(
        self, rep: torch.Tensor, aug_rep: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE-style contrastive loss giữa time-domain và frequency-domain.
        Args:
            rep: (B, N, D) — encoder output từ chuỗi gốc
            aug_rep: (B, N, D) — encoder output từ chuỗi augmented
        Returns:
            loss: scalar
        """
        # Aggregate over nodes → graph-level representation
        rep_agg = rep.sum(dim=1)       # (B, D)
        aug_agg = aug_rep.sum(dim=1)   # (B, D)
        # Cosine similarity matrix: (B, B)
        sim_matrix = F.cosine_similarity(
            rep_agg.unsqueeze(1), aug_agg.unsqueeze(0), dim=-1
        )
        sim_matrix = torch.exp(sim_matrix / 1.0)
        # Positive pairs: diagonal entries (sample i ↔ augmented i)
        B = rep_agg.size(0)
        pos_sim = sim_matrix[range(B), range(B)]
        # InfoNCE: -log(pos / sum_negatives)
        loss = -torch.log(pos_sim / sim_matrix.sum(dim=1).clamp(min=1e-8))
        return loss.mean()

    # ── Patchify / Unpatchify ──
    def _patchify(self, X: torch.Tensor) -> torch.Tensor:
        """(B, T, N, C) → (B, num_patches, N, patch_len * C)"""
        B, T, N, C = X.shape
        X = X.view(B, self.num_patches, self.patch_len, N, C)
        X = X.permute(0, 1, 3, 2, 4).contiguous()
        return X.view(B, self.num_patches, N, self.patch_len * C)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, N, patch_len * out_dim) → (B, T, N, out_dim)"""
        B, P, N, _ = patches.shape
        patches = patches.view(B, P, N, self.patch_len, self.out_dim)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()
        return patches.view(B, self.horizon, N, self.out_dim)

    # ── Forward ──
    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Args:
            X: (B, T_in=12, N, C=3)
            y: (B, T_out=12, N, 1) — optional target cho teacher forcing
            teacher_forcing_ratio: float — ignored, dùng internal exponential decay
        Returns:
            Nếu training hoặc y is not None:
                (pred, contra_loss) — pred: (B, T, N, 1), contra_loss: scalar hoặc None
            Nếu eval và y is None:
                pred — (B, T, N, 1)
        """
        B, T_in, N, C_in = X.shape
        supports_en = [self.A_static]

        # ═══════════════════════════════════════════════════════
        # 1. PATCHIFY + PROJECT INPUT
        # ═══════════════════════════════════════════════════════
        X_patches = self.enc_patch_proj(self._patchify(X))  # (B, P, N, hidden_dim)

        # ═══════════════════════════════════════════════════════
        # 2. ENCODE — TIME DOMAIN
        # ═══════════════════════════════════════════════════════
        init_state = self.encoder.init_hidden(B)
        h_en, _ = self.encoder(X_patches, init_state, supports_en)
        h_t = h_en[:, -1, :, :]  # (B, N, hidden_dim) — last patch hidden

        # ═══════════════════════════════════════════════════════
        # 3. CONTRASTIVE LEARNING — FREQUENCY DOMAIN
        # ═══════════════════════════════════════════════════════
        contra_loss = None
        h_t_fre = None

        if self.contra:
            # 3a. FFT augmentation trên flow channel
            flow = X[:, :, :, 0].permute(0, 2, 1)  # (B, N, T)
            flow_aug = self.spe_seq_cell(flow)       # (B, N, T) — augmented

            # 3b. Tạo augmented input: thay flow, giữ time features
            X_aug = X.clone()
            X_aug[:, :, :, 0] = flow_aug.permute(0, 2, 1)

            # 3c. Patchify + Encode augmented view
            X_aug_patches = self.enc_patch_proj(self._patchify(X_aug))
            init_state_fre = self.encoder.init_hidden(B)
            h_en_fre, _ = self.encoder(X_aug_patches, init_state_fre, supports_en)
            h_t_fre = h_en_fre[:, -1, :, :]  # (B, N, hidden_dim)

            # 3d. Contrastive loss (chỉ tính khi training)
            if self.training:
                contra_loss = self.calculate_contra_loss(h_t, h_t_fre)

        # ═══════════════════════════════════════════════════════
        # 4. DYNAMIC GRAPH LEARNING
        # ═══════════════════════════════════════════════════════
        supports_de = [self.A_static]

        if self.dynamic:
            # Dynamic graph từ time-domain hidden state
            node_emb = self.hypernet(h_t)  # (B, N, hyper_dim)
            A_dyn = F.softmax(
                F.relu(torch.einsum("bnc,bmc->bnm", node_emb, node_emb)),
                dim=-1
            )
            supports_de.append(A_dyn)

            if self.contra and h_t_fre is not None:
                # Dynamic graph từ frequency-domain hidden state
                node_emb_fre = self.hypernet(h_t_fre)
                A_dyn_fre = F.softmax(
                    F.relu(torch.einsum("bnc,bmc->bnm", node_emb_fre, node_emb_fre)),
                    dim=-1
                )
                supports_de.append(A_dyn_fre)

        # ═══════════════════════════════════════════════════════
        # 5. FUSE TIME + FREQUENCY REPRESENTATIONS
        # ═══════════════════════════════════════════════════════
        if self.contra and h_t_fre is not None:
            h_t = self.decoder_proj(torch.cat([h_t, h_t_fre], dim=-1))

        # ═══════════════════════════════════════════════════════
        # 6. DECODE — AUTOREGRESSIVE WITH SCHEDULED SAMPLING
        # ═══════════════════════════════════════════════════════
        ht_list = [h_t] * self.num_layers
        go = torch.zeros(B, N, self.patch_len * self.out_dim, device=X.device)

        # Patchify target nếu có (cho teacher forcing)
        y_patches = None
        if y is not None:
            y_patches = self._patchify(y)  # (B, P, N, patch_len * out_dim)

        pred_patches = []
        for t in range(self.num_patches):
            # Project decoder input
            dec_in = self.dec_patch_proj(go)  # (B, N, hidden_dim)

            # Decoder step
            h_de, ht_list = self.decoder(dec_in, ht_list, supports_de)

            # Output projection
            go = self.out_proj(h_de)  # (B, N, patch_len * out_dim)
            pred_patches.append(go.unsqueeze(1))

            # Scheduled Sampling (exponential decay)
            if self.training and y_patches is not None:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold():
                    go = y_patches[:, t, :, :]
                # else: dùng predicted go (autoregressive)

        # Increment batch counter (once per forward, not per patch)
        if self.training:
            self.batches_seen += 1

        pred_patches = torch.cat(pred_patches, dim=1)  # (B, P, N, patch_len*out_dim)
        pred = self._unpatchify(pred_patches)            # (B, T, N, out_dim)

        # Return format: tuple khi training, tensor khi inference
        if y is None and not self.training:
            return pred
        return pred, contra_loss


# ──────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, N, C = 4, 12, 170, 3
    dummy_adj = np.random.rand(N, N).astype(np.float32)
    dummy_adj = (dummy_adj > 0.8).astype(np.float32)

    model = PGCRN(
        num_nodes=N, adj_matrix=dummy_adj, in_dim=C,
        hidden_dim=64, out_dim=1, num_layers=2,
        patch_len=2, cheb_k=3, hyper_dim=10,
        dynamic=True, contra=True,
    )
    x = torch.randn(B, T, N, C)
    y = torch.randn(B, T, N, 1)

    # Test inference mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    print(f"[Eval] PGCRN output shape: {out_eval.shape}")

    # Test training mode
    model.train()
    out_train, contra_loss = model(x, y=y)
    print(f"[Train] PGCRN output shape: {out_train.shape}")
    print(f"[Train] Contra loss: {contra_loss.item():.4f}" if contra_loss is not None else "[Train] No contra loss")
    print(f"Total Params: {sum(p.numel() for p in model.parameters()):,}")
