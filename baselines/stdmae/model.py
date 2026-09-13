"""
STD-MAE — Spatial-Temporal Dual Masked Autoencoder for Traffic Forecasting
Paper: Haotian Gao et al., "Spatial-Temporal Dual Masked Autoencoder (STD-MAE)",
       ACM MM 2023.
GitHub: https://github.com/Jimmy-7664/STD-MAE

Kiến trúc 2 giai đoạn:
  Phase 1 — Pre-training:
    - Spatial MAE: mask ngẫu nhiên một số node, reconstruct từ visible nodes
    - Temporal MAE: mask ngẫu nhiên một số time step, reconstruct
  Phase 2 — Fine-tuning:
    - Dùng encoder đã pretrain, thêm prediction head để dự báo 12 bước

Token-based design (giống ViT MAE):
  - Chia (B, T, N, C) thành spatial tokens và temporal tokens
  - Encoder chỉ xử lý visible (unmasked) tokens
  - Decoder reconstruct masked tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class PatchEmbed(nn.Module):
    """Embed đầu vào thành tokens."""
    def __init__(self, in_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Standard Transformer block."""
    def __init__(self, d_model: int, nhead: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialMAEEncoder(nn.Module):
    """
    Spatial MAE Encoder: xử lý sequence of node tokens tại mỗi time step.
    Masked nodes bị loại, chỉ visible nodes qua encoder.
    """
    def __init__(self, num_nodes: int, in_dim: int, d_model: int, num_layers: int = 3, nhead: int = 4):
        super().__init__()
        self.d_model = d_model
        self.patch_emb = PatchEmbed(in_dim, d_model)
        self.pos_emb = nn.Embedding(num_nodes, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x: (B*T, N, in_dim)
            mask: (N,) boolean — True = masked (invisible)
        Returns:
            encoded: (B*T, N_visible, d_model)
            visible_idx: indices of visible tokens
        """
        N = x.shape[1]
        device = x.device
        visible_idx = (~mask).nonzero(as_tuple=True)[0]  # (N_visible,)

        # Embed all tokens then select visible
        tokens = self.patch_emb(x)  # (B*T, N, d_model)
        pos = self.pos_emb(torch.arange(N, device=device))  # (N, d_model)
        tokens = tokens + pos.unsqueeze(0)

        # Only process visible tokens
        tokens_vis = tokens[:, visible_idx, :]  # (B*T, N_vis, d_model)
        for blk in self.blocks:
            tokens_vis = blk(tokens_vis)
        return self.norm(tokens_vis), visible_idx


class SpatialMAEDecoder(nn.Module):
    """
    Spatial MAE Decoder: reconstruct masked tokens.
    """
    def __init__(self, num_nodes: int, d_model: int, decoder_dim: int, out_dim: int,
                 num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.proj = nn.Linear(d_model, decoder_dim)
        self.pos_emb = nn.Embedding(num_nodes, decoder_dim)
        self.blocks = nn.ModuleList([TransformerBlock(decoder_dim, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred_head = nn.Linear(decoder_dim, out_dim)

    def forward(self, encoded: torch.Tensor, visible_idx: torch.Tensor, num_nodes: int):
        """
        Args:
            encoded: (B*T, N_vis, d_model)
            visible_idx: (N_vis,)
            num_nodes: total N
        Returns:
            pred_masked: (B*T, N_masked, out_dim) — reconstruction of masked tokens
        """
        BT, N_vis, _ = encoded.shape
        device = encoded.device

        tokens = self.proj(encoded)  # (B*T, N_vis, decoder_dim)

        # Build full token sequence with mask tokens
        full_tokens = self.mask_token.expand(BT, num_nodes, -1).clone()
        full_tokens[:, visible_idx, :] = tokens

        # Add positional embedding
        pos = self.pos_emb(torch.arange(num_nodes, device=device))
        full_tokens = full_tokens + pos.unsqueeze(0)

        for blk in self.blocks:
            full_tokens = blk(full_tokens)
        full_tokens = self.norm(full_tokens)

        # Return only masked tokens
        masked_idx = torch.ones(num_nodes, dtype=torch.bool, device=device)
        masked_idx[visible_idx] = False
        return self.pred_head(full_tokens[:, masked_idx, :])


class STDMAEPretrain(nn.Module):
    """
    STD-MAE Pre-training model (Spatial + Temporal MAE combined).
    """
    def __init__(
        self,
        num_nodes: int,
        seq_len: int = 12,
        in_dim: int = 3,
        d_model: int = 64,
        decoder_dim: int = 32,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        nhead: int = 4,
        spatial_mask_ratio: float = 0.25,
        temporal_mask_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.spatial_mask_ratio = spatial_mask_ratio
        self.temporal_mask_ratio = temporal_mask_ratio
        self.d_model = d_model

        # Spatial MAE
        self.spatial_encoder = SpatialMAEEncoder(num_nodes, in_dim, d_model, num_encoder_layers, nhead)
        self.spatial_decoder = SpatialMAEDecoder(num_nodes, d_model, decoder_dim, in_dim,
                                                  num_decoder_layers, nhead)

        # Temporal MAE — treat each node's time series as sequence of tokens
        self.temporal_encoder = SpatialMAEEncoder(seq_len, in_dim, d_model, num_encoder_layers, nhead)
        self.temporal_decoder = SpatialMAEDecoder(seq_len, d_model, decoder_dim, in_dim,
                                                   num_decoder_layers, nhead)

    def _random_mask(self, N: int, ratio: float, device) -> torch.Tensor:
        """Tạo boolean mask — True = masked."""
        num_masked = int(N * ratio)
        idx = torch.randperm(N, device=device)[:num_masked]
        mask = torch.zeros(N, dtype=torch.bool, device=device)
        mask[idx] = True
        return mask

    def forward_spatial(self, X: torch.Tensor):
        """
        Spatial MAE forward pass.
        X: (B, T, N, C)
        Returns: loss
        """
        B, T, N, C = X.shape
        device = X.device
        
        mask = self._random_mask(N, self.spatial_mask_ratio, device)
        x_flat = X.reshape(B * T, N, C)  # (B*T, N, C)

        encoded, visible_idx = self.spatial_encoder(x_flat, mask)
        pred = self.spatial_decoder(encoded, visible_idx, N)  # (B*T, N_masked, C)

        # Target: masked patches
        masked_idx = (~(~mask)).clone()  # inverted mask for indexing
        masked_idx2 = mask
        target = x_flat[:, masked_idx2, :]  # (B*T, N_masked, C)
        loss = F.mse_loss(pred, target)
        return loss

    def forward_temporal(self, X: torch.Tensor):
        """
        Temporal MAE forward pass.
        X: (B, T, N, C) → for each node: (B*N, T, C)
        """
        B, T, N, C = X.shape
        device = X.device

        mask = self._random_mask(T, self.temporal_mask_ratio, device)
        x_flat = X.permute(0, 2, 1, 3).reshape(B * N, T, C)  # (B*N, T, C)

        encoded, visible_idx = self.temporal_encoder(x_flat, mask)
        pred = self.temporal_decoder(encoded, visible_idx, T)  # (B*N, T_masked, C)

        masked_idx2 = mask
        target = x_flat[:, masked_idx2, :]
        loss = F.mse_loss(pred, target)
        return loss

    def pretrain_loss(self, X: torch.Tensor) -> torch.Tensor:
        """Tổng loss của spatial + temporal MAE."""
        loss_s = self.forward_spatial(X)
        loss_t = self.forward_temporal(X)
        return loss_s + loss_t


class STDMAEFinetune(nn.Module):
    """
    STD-MAE Fine-tuning model.
    Dùng encoder đã pretrain + prediction head.
    """
    def __init__(
        self,
        num_nodes: int,
        seq_len: int = 12,
        in_dim: int = 3,
        d_model: int = 64,
        num_encoder_layers: int = 3,
        nhead: int = 4,
        out_dim: int = 1,
        horizon: int = 12,
        pretrained_model: "STDMAEPretrain" = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.d_model = d_model

        if pretrained_model is not None:
            # Transfer weights từ pretrained spatial encoder
            self.spatial_encoder = pretrained_model.spatial_encoder
        else:
            self.spatial_encoder = SpatialMAEEncoder(num_nodes, in_dim, d_model,
                                                      num_encoder_layers, nhead)

        # Temporal aggregation
        self.temporal_pool = nn.Linear(seq_len, horizon)

        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: (B, T_in=12, N, C=3)
        Returns:
            pred: (B, T_out=12, N, 1)
        """
        B, T, N, C = X.shape
        device = X.device

        # No masking at inference — use all nodes
        no_mask = torch.zeros(N, dtype=torch.bool, device=device)
        x_flat = X.reshape(B * T, N, C)

        encoded, _ = self.spatial_encoder(x_flat, no_mask)  # (B*T, N, d_model)
        encoded = encoded.reshape(B, T, N, self.d_model)     # (B, T, N, d_model)

        # Temporal projection: T_in → T_out
        h = encoded.permute(0, 2, 3, 1)   # (B, N, d_model, T)
        h = self.temporal_pool(h)          # (B, N, d_model, T_out)
        h = h.permute(0, 3, 1, 2)         # (B, T_out, N, d_model)

        return self.pred_head(h)           # (B, T_out, N, 1)


if __name__ == "__main__":
    N, T = 50, 12
    B = 2

    # Test pretrain
    pretrain = STDMAEPretrain(num_nodes=N, seq_len=T, in_dim=3, d_model=32, decoder_dim=16,
                               num_encoder_layers=2, num_decoder_layers=1, nhead=4)
    x = torch.randn(B, T, N, 3)
    loss = pretrain.pretrain_loss(x)
    print(f"STD-MAE pretrain loss: {loss.item():.4f}")

    # Test finetune
    finetune = STDMAEFinetune(num_nodes=N, seq_len=T, in_dim=3, d_model=32,
                               pretrained_model=pretrain)
    pred = finetune(x)
    print(f"STD-MAE finetune output: {pred.shape}")  # (2, 12, 50, 1)
    print(f"Params (pretrain): {sum(p.numel() for p in pretrain.parameters()):,}")
    print(f"Params (finetune): {sum(p.numel() for p in finetune.parameters()):,}")
