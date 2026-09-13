"""
Adaptive T-GCN — Adaptive Temporal Graph Convolutional Network
==============================================================
Paper tham khảo:
  - T-GCN: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for
    Traffic Prediction", IEEE TITS 2020.
  - AGCRN: Bai et al., "Adaptive Graph Convolutional Recurrent Network for
    Traffic Forecasting", NeurIPS 2020.

Đóng góp mới so với T-GCN gốc:
  1. Physical-Adaptive Graph Fusion (GraphFusionModule): kết hợp đồ thị
     khoảng cách vật lý (A_phys) và đồ thị tự học bất đối xứng (A_adapt)
     qua hệ số alpha cố định hoặc học được.
  2. Top-k Sparsification: cắt bỏ các cạnh yếu để giảm nhiễu đồ thị.
  3. Graph Regularization Loss: kiểm soát độ "gọn" của đồ thị học được.
  4. Thiết kế Encoder-Decoder đầy đủ với Scheduled Sampling.

Rút kinh nghiệm từ debug AGCRN / DCRNN / PGCRN:
  - Toàn bộ trọng số dùng Xavier init, KHÔNG nhân hệ số nhỏ tùy tiện.
  - Chỉ thay đổi MÔ-ĐUN ĐỒ THỊ, giữ nguyên GRU đơn giản của T-GCN để
    cô lập rõ đóng góp trong ablation study.
  - forward() trả về (pred, graph_reg_loss) để training loop cộng vào
    total loss mà không cần sửa trainer.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adaptive_tgcn.graph_module import GraphFusionModule


# =========================================================================
# TGCNCell — GRU Cell dùng Graph Convolution
# =========================================================================

class TGCNCell(nn.Module):
    """
    GRU Cell với Graph Convolution (hỗ trợ 1-hop thuần T-GCN và 2-hop Diffusion).

    Nhận ma trận đồ thị A làm tham số forward() để A có thể được tính động
    bởi GraphFusionModule mỗi forward pass — KHÔNG lưu A cố định trong cell.

    Kiến trúc:
        r = sigmoid(GCN([x, h], A) → 2*hidden → r)   # reset gate
        z = sigmoid(GCN([x, h], A) → 2*hidden → z)   # update gate
        h̃ = tanh(GCN([x, r⊙h], A) → hidden)          # candidate
        h_new = (1 - z) ⊙ h + z ⊙ h̃
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_nodes: int, gcn_depth: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.gcn_depth  = gcn_depth

        # Linear sau graph conv:
        # Nếu gcn_depth == 1: 1-hop (Ax) -> factor = 1
        # Nếu gcn_depth == 2: 2-hop diffusion [x, Ax, A(Ax)] -> factor = 3
        gcn_factor = 1 if gcn_depth == 1 else 3
        in_features = (in_dim + hidden_dim) * gcn_factor

        self.W_gate = nn.Linear(in_features, 2 * hidden_dim, bias=True)
        self.W_cand = nn.Linear(in_features, hidden_dim, bias=True)

        # Xavier init
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.xavier_uniform_(self.W_cand.weight)
        nn.init.zeros_(self.W_cand.bias)

    def _graph_conv(
        self, x: torch.Tensor, A: torch.Tensor, W: nn.Linear
    ) -> torch.Tensor:
        """
        Tích chập đồ thị:
        - gcn_depth == 1: H' = A · x, rồi chiếu tuyến tính.
        - gcn_depth == 2: H' = concat(x, A · x, A · (A · x)), rồi chiếu tuyến tính.

        Args:
            x : (B, N, C) — đặc trưng tại mỗi nút
            A : (N, N)    — ma trận kề (đã chuẩn hoá)
            W : Linear(C * factor, out_dim)
        Returns:
            (B, N, out_dim)
        """
        if self.gcn_depth == 1:
            Ax = torch.einsum("mn, bnd -> bmd", A, x)   # (B, N, C)
            return W(Ax)
        elif self.gcn_depth == 2:
            Ax = torch.einsum("mn, bnd -> bmd", A, x)    # 1-hop
            A2x = torch.einsum("mn, bnd -> bmd", A, Ax)  # 2-hop
            feats = torch.cat([x, Ax, A2x], dim=-1)      # (B, N, 3*C)
            return W(feats)
        else:
            raise ValueError(f"Unsupported gcn_depth: {self.gcn_depth}")

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x : (B, N, in_dim)
            h : (B, N, hidden_dim)
            A : (N, N) ma trận đồ thị từ GraphFusionModule
        Returns:
            h_new : (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)  # (B, N, in_dim + hidden_dim)

        gates = torch.sigmoid(self._graph_conv(xh, A, self.W_gate))  # (B, N, 2*hidden)
        r, z  = gates.chunk(2, dim=-1)                                # each (B, N, hidden)

        xrh   = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self._graph_conv(xrh, A, self.W_cand))   # (B, N, hidden)

        h_new  = (1 - z) * h + z * h_cand
        return h_new


# =========================================================================
# AdaptiveTGCN — Model chính (Encoder + Direct/Autoregressive Predictor)
# =========================================================================

class AdaptiveTGCN(nn.Module):
    """
    Adaptive T-GCN:
      - Physical-Adaptive Graph Fusion (GraphFusionModule)
      - 2-hop Spatial-Adaptive Diffusion Convolution
      - Direct Multi-Horizon Predictor Head (hoặc Autoregressive Decoder)

    Args:
        num_nodes        : Số nút N.
        in_dim           : Số chiều input (mặc định 3: flow, speed, occ).
        hidden_dim       : Chiều ẩn của TGCNCell.
        out_dim          : Số chiều output (mặc định 1: dự báo flow).
        num_layers       : Số lớp TGCNCell trong Encoder (và Decoder nếu có).
        embed_dim        : Chiều embedding E1, E2 của adaptive graph.
        horizon          : Số bước dự báo (mặc định 12).
        adj_matrix       : Ma trận kề vật lý (N, N). Bắt buộc nếu
                           graph_mode in {"physical", "fused"}.
        graph_mode       : "physical" | "adaptive" | "fused".
        fusion_type      : "fixed" | "learnable" (chỉ khi graph_mode="fused").
        alpha            : Hệ số hợp nhất cố định (chỉ khi fusion_type="fixed").
        top_k            : 0 = tắt sparsification; > 0 = giữ top-k cạnh/hàng.
        use_graph_reg    : Bật graph regularization loss.
        graph_reg_weight : Trọng số của graph_reg_loss trong total loss.
        temperature      : Nhiệt độ scaling cho softmax A_adapt.
        gcn_depth        : 1 (1-hop thuần) hoặc 2 (2-hop diffusion).
        predictor_type   : "direct" (one-shot end_conv) hoặc "autoregressive".
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int = 3,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        embed_dim: int = 10,
        horizon: int = 12,
        adj_matrix: torch.Tensor = None,
        graph_mode: str = "fused",
        fusion_type: str = "fixed",
        alpha: float = 0.5,
        top_k: int = 0,
        use_graph_reg: bool = False,
        graph_reg_weight: float = 1e-4,
        temperature: float = 1.0,
        gcn_depth: int = 1,
        predictor_type: str = "autoregressive",
    ):
        super().__init__()

        self.num_nodes        = num_nodes
        self.hidden_dim       = hidden_dim
        self.num_layers       = num_layers
        self.horizon          = horizon
        self.out_dim          = out_dim
        self.use_graph_reg    = use_graph_reg
        self.graph_reg_weight = graph_reg_weight
        self.gcn_depth        = gcn_depth
        self.predictor_type   = predictor_type

        # --- Mô-đun sinh ma trận đồ thị linh hoạt ---
        self.graph_module = GraphFusionModule(
            num_nodes=num_nodes,
            embed_dim=embed_dim,
            adj_matrix=adj_matrix,
            graph_mode=graph_mode,
            fusion_type=fusion_type,
            alpha=alpha,
            top_k=top_k,
            use_graph_reg=use_graph_reg,
            temperature=temperature,
        )

        # --- Encoder: num_layers TGCNCell xếp chồng ---
        self.encoder_cells = nn.ModuleList()
        for i in range(num_layers):
            cell_in = in_dim if i == 0 else hidden_dim
            self.encoder_cells.append(TGCNCell(cell_in, hidden_dim, num_nodes, gcn_depth=gcn_depth))

        # --- Predictor Head ---
        if predictor_type == "direct":
            # Direct Multi-Horizon Predictor (chuẩn AGCRN / STGCN): chiếu trực tiếp h_last sang toàn bộ horizon
            # Triệt tiêu exposure bias và tích luỹ sai số tự hồi quy
            self.end_conv = nn.Conv2d(1, horizon * out_dim, kernel_size=(1, hidden_dim), bias=True)
            nn.init.xavier_uniform_(self.end_conv.weight)
            nn.init.zeros_(self.end_conv.bias)
            self.decoder_cells = None
            self.output_proj = None
        elif predictor_type == "autoregressive":
            # Autoregressive Decoder tuần tự từng bước với Scheduled Sampling
            self.decoder_cells = nn.ModuleList()
            for i in range(num_layers):
                cell_in = out_dim if i == 0 else hidden_dim
                self.decoder_cells.append(TGCNCell(cell_in, hidden_dim, num_nodes, gcn_depth=gcn_depth))
            self.output_proj = nn.Linear(hidden_dim, out_dim)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            self.end_conv = None
        else:
            raise ValueError(f"Unsupported predictor_type: {predictor_type}. Must be 'direct' or 'autoregressive'.")

    def _init_hidden(self, batch_size: int, device: torch.device) -> list:
        return [
            torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def encode(self, X: torch.Tensor, A: torch.Tensor) -> list:
        """
        Encoder: chạy qua T bước thời gian.

        Args:
            X : (B, T, N, C)
            A : (N, N) — tính 1 lần trước vòng lặp, dùng chung mọi bước t
        Returns:
            h : list of (B, N, hidden_dim) — hidden states tầng cuối mỗi layer
        """
        B, T, N, C = X.shape
        h = self._init_hidden(B, X.device)

        for t in range(T):
            x_t = X[:, t, :, :]   # (B, N, C)
            for layer_i, cell in enumerate(self.encoder_cells):
                h[layer_i] = cell(x_t, h[layer_i], A)
                x_t = h[layer_i]

        return h

    def decode(
        self,
        h: list,
        A: torch.Tensor,
        y_true: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Decoder: sinh horizon bước dự báo, có Scheduled Sampling (chỉ dùng khi predictor_type='autoregressive').
        """
        B = h[0].shape[0]
        device = h[0].device

        dec_input = torch.zeros(B, self.num_nodes, self.out_dim, device=device)
        outputs = []

        for t in range(self.horizon):
            x_t   = dec_input
            h_new = list(h)
            for layer_i, cell in enumerate(self.decoder_cells):
                h_new[layer_i] = cell(x_t, h[layer_i], A)
                x_t = h_new[layer_i]
            h = h_new

            pred_t = self.output_proj(x_t)    # (B, N, out_dim)
            outputs.append(pred_t.unsqueeze(1))  # (B, 1, N, out_dim)

            if self.training and y_true is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y_true[:, t, :, :]  # ground-truth
            else:
                dec_input = pred_t.detach()      # tự hồi quy

        return torch.cat(outputs, dim=1)          # (B, horizon, N, out_dim)

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple:
        """
        Args:
            X                      : (B, T_in=12, N, C=3)
            y                      : (B, horizon=12, N, 1) target cho teacher forcing (chỉ dùng khi autoregressive)
            teacher_forcing_ratio  : float
        Returns:
            pred            : (B, horizon, N, out_dim)
            graph_reg_loss  : scalar tensor (0.0 nếu use_graph_reg=False)
        """
        # Tính A một lần trước toàn bộ forward pass
        A = self.graph_module()                    # (N, N)

        h = self.encode(X, A)

        if self.predictor_type == "direct":
            # Direct multi-horizon projection
            h_last = h[-1].unsqueeze(1)             # (B, 1, N, hidden_dim)
            out = self.end_conv(h_last).squeeze(-1) # (B, horizon * out_dim, N)
            out = out.view(X.shape[0], self.horizon, self.out_dim, self.num_nodes)
            pred = out.permute(0, 1, 3, 2)          # (B, horizon, N, out_dim)
        else:
            pred = self.decode(h, A, y_true=y, teacher_forcing_ratio=teacher_forcing_ratio)

        if self.use_graph_reg:
            graph_reg_loss = self.graph_reg_weight * self.graph_module.compute_graph_reg_loss()
        else:
            graph_reg_loss = torch.tensor(0.0, device=X.device)

        return pred, graph_reg_loss


# =========================================================================
# Quick smoke test
# =========================================================================

if __name__ == "__main__":
    import numpy as np

    N, B, T, C = 170, 4, 12, 3
    adj = torch.tensor(np.random.rand(N, N).astype("float32"))

    configs = [
        dict(graph_mode="physical",  fusion_type="fixed",     adj_matrix=adj, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="adaptive",  fusion_type="fixed",     adj_matrix=None, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="fixed",     adj_matrix=adj, alpha=0.5, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, top_k=10, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, top_k=10, use_graph_reg=True, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, predictor_type="autoregressive", gcn_depth=1),
    ]

    for cfg in configs:
        model = AdaptiveTGCN(num_nodes=N, in_dim=C, hidden_dim=64, **cfg)
        X = torch.randn(B, T, N, C)
        Y = torch.randn(B, T, N, 1)
        pred, reg_loss = model(X, y=Y, teacher_forcing_ratio=0.5)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        label = f"mode={cfg['graph_mode']:<8} pred={cfg['predictor_type']:<14} hop={cfg['gcn_depth']} reg={cfg.get('use_graph_reg',False)}"
        print(f"[{label}] out={pred.shape} reg_loss={reg_loss.item():.6f} params={params:,}")

