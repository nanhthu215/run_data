"""
Adaptive T-GCN — Adaptive Temporal Graph Convolutional Network (v2 Enhanced)
=============================================================================
Đóng góp cốt lõi:
  1. Physical-Adaptive Graph Fusion (GraphFusionModule): Hợp nhất ma trận kề
     vật lý khoảng cách (A_phys) và ma trận tương quan thích nghi có hướng (A_adapt)
     học từ node embeddings bất đối xứng E1, E2, kèm Top-k sparsification.
  2. Multi-Support Directed Diffusion Graph Convolution: Mô hình hóa dòng chảy
     giao thông xuôi dòng (forward A) và sóng xung kích tắc nghẽn dội ngược (backward A^T)
     kết hợp tự bảo toàn trạng thái (identity I).
  3. Input Feature & Spatial Embedding: Chiếu phi tuyến 3 kênh (flow, speed, occ)
     kết hợp spatial positional embedding riêng cho từng trạm đo.
  4. Multi-Scale Temporal Attention Predictor Head: Tổng hợp toàn bộ 12 bước lịch sử
     qua Gated Temporal Convolution (GLU) và Temporal Attention Pooling, chiếu trực tiếp
     sang 12 bước dự báo tương lai (Zero Exposure Bias, tốc độ song song cực nhanh).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adaptive_tgcn.graph_module import GraphFusionModule


# =========================================================================
# TGCNCell — Multi-Support Directed Diffusion Graph GRU Cell
# =========================================================================

class TGCNCell(nn.Module):
    """
    GRU Cell với Multi-Support Directed Diffusion Graph Convolution.

    Hỗ trợ 3 toán tử vật lý không gian:
      - S_0: Identity / Self-loop (X) — bảo toàn quán tính nội tại trạm đo
      - S_1: Forward transition (A @ X) — lan truyền xuôi dòng giao thông
      - S_2: Backward transition (A^T @ X) — sóng xung kích dội ngược lên thượng lưu

    Hỗ trợ cả ma trận kề 2D (N, N) tĩnh và 3D (B, N, N) động.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_nodes: int, gcn_depth: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.gcn_depth  = gcn_depth

        # gcn_depth=1: [X, AX] (factor=2)
        # gcn_depth=2: [X, AX, A^T X] (factor=3, multi-support directed diffusion)
        self.num_supports = 2 if gcn_depth == 1 else 3
        in_features = (in_dim + hidden_dim) * self.num_supports

        self.W_gate = nn.Linear(in_features, 2 * hidden_dim, bias=True)
        self.W_cand = nn.Linear(in_features, hidden_dim, bias=True)

        # Xavier initialization
        nn.init.xavier_uniform_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.xavier_uniform_(self.W_cand.weight)
        nn.init.zeros_(self.W_cand.bias)

    def _graph_conv(self, x: torch.Tensor, A: torch.Tensor, W: nn.Linear) -> torch.Tensor:
        """
        Tích chập đồ thị đa hướng:
        Args:
            x : (B, N, C)
            A : (N, N) hoặc (B, N, N)
            W : Linear layer
        Returns:
            (B, N, out_dim)
        """
        is_3d = (A.dim() == 3)

        # S0: Identity (Self-loop)
        s0 = x  # (B, N, C)

        # S1: Forward (A @ x)
        if is_3d:
            s1 = torch.bmm(A, x)
        else:
            s1 = torch.einsum("mn, bnd -> bmd", A, x)

        supports = [s0, s1]

        # S2: Backward (A^T @ x) nếu gcn_depth >= 2
        if self.num_supports >= 3:
            if is_3d:
                s2 = torch.bmm(A.transpose(1, 2), x)
            else:
                s2 = torch.einsum("nm, bnd -> bmd", A, x)
            supports.append(s2)

        feats = torch.cat(supports, dim=-1)  # (B, N, num_supports * C)
        return W(feats)

    def forward(self, x: torch.Tensor, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, in_dim)
            h : (B, N, hidden_dim)
            A : (N, N) hoặc (B, N, N)
        Returns:
            h_new : (B, N, hidden_dim)
        """
        xh = torch.cat([x, h], dim=-1)  # (B, N, in_dim + hidden_dim)

        gates = torch.sigmoid(self._graph_conv(xh, A, self.W_gate))  # (B, N, 2*hidden)
        r, z  = gates.chunk(2, dim=-1)

        xrh   = torch.cat([x, r * h], dim=-1)
        h_cand = torch.tanh(self._graph_conv(xrh, A, self.W_cand))

        h_new  = (1 - z) * h + z * h_cand
        return h_new


# =========================================================================
# TemporalAttentionPredictor — Đầu Dự Báo Không Gian - Thời Gian Đa Tầm
# =========================================================================

class TemporalAttentionPredictor(nn.Module):
    """
    Đầu dự báo đa quy mô không - thời gian:
      1. Gated Temporal Convolution (GLU) qua trục thời gian T=12 để nắm bắt
         gia tốc, xu thế và tính chu kỳ.
      2. Temporal Attention Pooling để tổng hợp có trọng số toàn bộ lịch sử.
      3. Chiếu trực tiếp đa bước (Direct Multi-Horizon Projection) sang 12 bước tương lai
         (Zero Exposure Bias, song song hoá 100% trên GPU).
    """

    def __init__(self, hidden_dim: int, horizon: int, out_dim: int = 1, num_nodes: int = 170):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon    = horizon
        self.out_dim    = out_dim

        # Temporal Conv1D với GLU: kernel=3, padding=1 giữ nguyên T=12
        self.tconv = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1)

        # Trọng số chú ý thời gian
        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

        # LayerNorm kết hợp
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Multi-Horizon Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, horizon * out_dim),
        )

        # Init
        nn.init.xavier_uniform_(self.tconv.weight)
        nn.init.zeros_(self.tconv.bias)

    def forward(self, H_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_seq : (B, T, N, D) — chuỗi biểu diễn ẩn từ Encoder qua T bước
        Returns:
            pred  : (B, horizon, N, out_dim)
        """
        B, T, N, D = H_seq.shape

        # Đổi trục để chạy Temporal Conv1D: (B*N, D, T)
        H_flat = H_seq.permute(0, 2, 3, 1).contiguous().view(B * N, D, T)
        conv_out = self.tconv(H_flat)                          # (B*N, 2*D, T)
        conv_val, conv_gate = conv_out.chunk(2, dim=1)         # each (B*N, D, T)
        t_feat = conv_val * torch.sigmoid(conv_gate)           # GLU: (B*N, D, T)

        # Đổi lại (B, T, N, D)
        t_feat = t_feat.view(B, N, D, T).permute(0, 3, 1, 2)  # (B, T, N, D)

        # Tính attention weights qua trục thời gian T
        attn_scores = self.attn_fc(t_feat)                     # (B, T, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)       # softmax theo T: (B, T, N, 1)

        # Tổng hợp có trọng số toàn bộ lịch sử
        h_pool = torch.sum(attn_weights * t_feat, dim=1)       # (B, N, D)

        # Residual với trạng thái ẩn bước cuối cùng
        h_last = H_seq[:, -1, :, :]                            # (B, N, D)
        h_combined = self.norm_out(h_pool + h_last)            # (B, N, D)

        # Chiếu trực tiếp ra horizon * out_dim
        out = self.proj_head(h_combined)                       # (B, N, horizon * out_dim)
        out = out.view(B, N, self.horizon, self.out_dim)       # (B, N, horizon, out_dim)
        pred = out.permute(0, 2, 1, 3)                         # (B, horizon, N, out_dim)
        return pred


# =========================================================================
# AdaptiveTGCN — Mô hình hoàn chỉnh (Adaptive T-GCN v2)
# =========================================================================

class AdaptiveTGCN(nn.Module):
    """
    Adaptive T-GCN:
      - Physical-Adaptive Graph Fusion (GraphFusionModule)
      - Multi-Support Directed Diffusion Spatial Convolution
      - Non-linear Feature Embedding + Learnable Spatial Positional Encoding
      - Multi-Layer Encoder với Residual Connection & Layer Normalization
      - Multi-Scale Temporal Attention Direct Multi-Horizon Predictor

    Args:
        num_nodes        : Số nút N (170 cho PEMS08, 307 cho PEMS04).
        in_dim           : Số chiều input (3: flow, speed, occ).
        hidden_dim       : Chiều ẩn của mạng (mặc định 64).
        out_dim          : Số chiều output (1: dự báo flow).
        num_layers       : Số lớp TGCNCell trong Encoder (mặc định 2).
        embed_dim        : Chiều embedding E1, E2 của adaptive graph (mặc định 10).
        horizon          : Số bước dự báo tương lai (mặc định 12 = 60 phút).
        adj_matrix       : Ma trận kề vật lý (N, N). Bắt buộc nếu graph_mode in {"physical", "fused"}.
        graph_mode       : "physical" | "adaptive" | "fused".
        fusion_type      : "fixed" | "learnable".
        alpha            : Hệ số hợp nhất cố định (khi fusion_type="fixed").
        top_k            : 0 = tắt; > 0 = giữ top-k cạnh/hàng.
        use_graph_reg    : Bật graph regularization loss.
        graph_reg_weight : Trọng số graph_reg_loss.
        temperature      : Nhiệt độ scaling cho softmax A_adapt.
        gcn_depth        : 1 (1-hop) hoặc 2 (multi-support directed diffusion, mặc định 2).
        predictor_type   : "direct" (Temporal Attention Multi-Horizon, mặc định) hoặc "autoregressive".
        use_dynamic      : Bật điều biến động theo ngữ cảnh thời gian thực (mặc định False).
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
        gcn_depth: int = 2,
        predictor_type: str = "direct",
        use_dynamic: bool = False,
        input_proj_type: str = "mlp",
        use_spatial_pos_emb: bool = True,
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
        self.input_proj_type  = input_proj_type
        self.use_spatial_pos_emb = use_spatial_pos_emb

        # --- 1. Mô-đun sinh ma trận đồ thị linh hoạt ---
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
            use_dynamic=use_dynamic,
        )

        # --- 2. Input Embedding & Spatial Positional Encoding ---
        if input_proj_type == "linear":
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if use_spatial_pos_emb:
            self.node_emb = nn.Parameter(torch.empty(num_nodes, hidden_dim))
            nn.init.xavier_uniform_(self.node_emb)
        else:
            self.node_emb = None

        # --- 3. Encoder: num_layers TGCNCell xếp chồng có Residual & LayerNorm ---
        self.encoder_cells = nn.ModuleList()
        self.layer_norms   = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_cells.append(TGCNCell(hidden_dim, hidden_dim, num_nodes, gcn_depth=gcn_depth))
            if i < num_layers - 1:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # --- 4. Predictor Head ---
        if predictor_type == "direct":
            # Multi-Scale Temporal Attention Direct Multi-Horizon Predictor (Tối ưu)
            self.predictor = TemporalAttentionPredictor(
                hidden_dim=hidden_dim,
                horizon=horizon,
                out_dim=out_dim,
                num_nodes=num_nodes,
            )
            self.decoder_cells = None
            self.output_proj   = None
        elif predictor_type == "autoregressive":
            # Fallback Autoregressive Decoder (cho tương thích backward nếu cần)
            self.decoder_cells = nn.ModuleList()
            for i in range(num_layers):
                cell_in = out_dim if i == 0 else hidden_dim
                self.decoder_cells.append(TGCNCell(cell_in, hidden_dim, num_nodes, gcn_depth=gcn_depth))
            self.output_proj = nn.Linear(hidden_dim, out_dim)
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
            self.predictor = None
        else:
            raise ValueError(f"Unsupported predictor_type: {predictor_type}. Must be 'direct' or 'autoregressive'.")

    def _init_hidden(self, batch_size: int, device: torch.device) -> list:
        return [
            torch.zeros(batch_size, self.num_nodes, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        ]

    def encode(self, X: torch.Tensor, A: torch.Tensor) -> tuple:
        """
        Encoder: chạy qua T bước thời gian.

        Args:
            X : (B, T, N, C)
            A : (N, N) hoặc (B, N, N)
        Returns:
            h      : list of (B, N, hidden_dim) — trạng thái ẩn tầng cuối
            all_h2 : (B, T, N, hidden_dim) — chuỗi trạng thái ẩn tầng trên cùng qua T bước
        """
        B, T, N, C = X.shape
        h = self._init_hidden(B, X.device)

        seq_states = []

        for t in range(T):
            # Chiếu đặc trưng đầu vào + cộng spatial node embedding (nếu có)
            x_raw = X[:, t, :, :]                        # (B, N, C)
            x_t   = self.input_proj(x_raw)
            if self.node_emb is not None:
                x_t = x_t + self.node_emb                # (B, N, hidden_dim)

            # Layer 0
            h[0] = self.encoder_cells[0](x_t, h[0], A)

            # Các layer tiếp theo kèm Residual Connection & LayerNorm
            x_prev = h[0]
            for layer_i in range(1, self.num_layers):
                cell = self.encoder_cells[layer_i]
                norm = self.layer_norms[layer_i - 1]
                x_in = norm(x_prev + x_t)  # residual connection
                h[layer_i] = cell(x_in, h[layer_i], A)
                x_prev = h[layer_i]

            seq_states.append(h[-1].unsqueeze(1))  # (B, 1, N, hidden_dim)

        all_h = torch.cat(seq_states, dim=1)  # (B, T, N, hidden_dim)
        return h, all_h

    def decode_autoregressive(
        self,
        h: list,
        A: torch.Tensor,
        y_true: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Decoder tự hồi quy (chỉ dùng khi predictor_type='autoregressive').
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

            pred_t = self.output_proj(x_t)
            outputs.append(pred_t.unsqueeze(1))

            if self.training and y_true is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = y_true[:, t, :, :]
            else:
                dec_input = pred_t.detach()

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple:
        """
        Args:
            X                      : (B, T_in=12, N, C=3)
            y                      : (B, horizon=12, N, 1)
            teacher_forcing_ratio  : float
        Returns:
            pred            : (B, horizon, N, out_dim)
            graph_reg_loss  : scalar tensor
        """
        # Sinh ma trận đồ thị từ GraphFusionModule
        A = self.graph_module()

        # Mã hóa chuỗi không - thời gian
        h_last_list, H_seq = self.encode(X, A)

        # Dự báo tương lai
        if self.predictor_type == "direct":
            pred = self.predictor(H_seq)
        else:
            pred = self.decode_autoregressive(h_last_list, A, y_true=y, teacher_forcing_ratio=teacher_forcing_ratio)

        # Graph regularization loss
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
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, top_k=5, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, top_k=5, use_graph_reg=True, predictor_type="direct", gcn_depth=2),
        dict(graph_mode="fused",     fusion_type="learnable", adj_matrix=adj, predictor_type="autoregressive", gcn_depth=1),
    ]

    print(f"{'='*80}")
    print("Testing AdaptiveTGCN Architectures:")
    print(f"{'='*80}")
    for cfg in configs:
        model = AdaptiveTGCN(num_nodes=N, in_dim=C, hidden_dim=64, **cfg)
        X = torch.randn(B, T, N, C)
        Y = torch.randn(B, T, N, 1)
        pred, reg_loss = model(X, y=Y, teacher_forcing_ratio=0.5)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        label = f"mode={cfg['graph_mode']:<8} pred={cfg['predictor_type']:<14} hop={cfg['gcn_depth']} reg={cfg.get('use_graph_reg',False)}"
        print(f"[{label}] out={list(pred.shape)} reg={reg_loss.item():.6f} params={params:,}")
    print(f"{'='*80}")
    print("All architecture configurations passed self-test successfully!")
