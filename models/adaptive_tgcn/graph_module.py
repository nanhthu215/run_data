"""
GraphFusionModule — Mô-đun sinh ma trận đồ thị linh hoạt
=========================================================
Hỗ trợ 3 chế độ đồ thị:
  - "physical"  : chỉ dùng A_phys cố định
  - "adaptive"  : học A_adapt từ E1, E2 bất đối xứng (vì giao thông có hướng)
  - "fused"     : kết hợp A_phys + A_adapt theo alpha (fixed hoặc learnable)

Tùy chọn bổ sung:
  - top_k sparsification   : mỗi hàng chỉ giữ top-k cạnh mạnh nhất
  - graph_reg_loss         : L2 norm của A_adapt làm loss phụ để kiểm soát
                             độ thưa và ổn định của đồ thị học được
  - temperature scaling    : softmax(logits / temperature) — temperature < 1
                             làm A_adapt sắc nét hơn, ít đồng đều hơn
                             (debug fix cho hiện tượng A_a suy biến uniform)

Rút kinh nghiệm từ debug PGCRN / AGCRN:
  - TUYỆT ĐỐI không nhân random init bằng hệ số nhỏ tùy tiện (như *0.01)
    vì sẽ triệt tiêu gradient qua các bước GRU. Luôn dùng Xavier / Kaiming.
  - Bước 2a (Debug adaptive graph): Thêm temperature scaling để tránh A_adapt
    suy biến thành phân phối uniform khi E1 @ E2.T cho ra giá trị gần bằng nhau.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphFusionModule(nn.Module):
    """
    Sinh ma trận đồ thị A (N, N) theo graph_mode đã chọn.

    Args:
        num_nodes    : Số nút đồ thị N.
        embed_dim    : Chiều embedding d của E1, E2 cho adaptive graph.
        adj_matrix   : Ma trận kề vật lý (N, N), float Tensor. Bắt buộc
                       khi graph_mode in {"physical", "fused"}.
        graph_mode   : "physical" | "adaptive" | "fused".
        fusion_type  : "fixed" | "learnable" — chỉ dùng khi graph_mode="fused".
        alpha        : Hệ số hợp nhất cố định (0–1), dùng khi fusion_type="fixed".
        top_k        : Nếu > 0: giữ top-k cạnh mạnh nhất mỗi hàng (0 = tắt).
        use_graph_reg: Nếu True: compute_graph_reg_loss() trả về L2 norm của A_adapt.
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 10,
        adj_matrix: torch.Tensor = None,
        graph_mode: str = "fused",
        fusion_type: str = "fixed",
        alpha: float = 0.5,
        top_k: int = 0,
        use_graph_reg: bool = False,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.num_nodes    = num_nodes
        self.embed_dim    = embed_dim
        self.graph_mode   = graph_mode
        self.fusion_type  = fusion_type
        self.top_k        = top_k
        self.use_graph_reg = use_graph_reg
        self.temperature  = temperature

        # ----- Ma trận vật lý cố định (không phải tham số học được) -----
        if graph_mode in ("physical", "fused"):
            if adj_matrix is None:
                raise ValueError("adj_matrix bắt buộc khi graph_mode là 'physical' hoặc 'fused'")
            if isinstance(adj_matrix, np.ndarray):
                A_p = torch.from_numpy(adj_matrix).float()
            else:
                A_p = adj_matrix.clone().detach().float()
            # Đảm bảo không có NaN / Inf
            A_p = torch.nan_to_num(A_p, nan=0.0, posinf=0.0, neginf=0.0)
            # Chuẩn hoá hàng để tổng mỗi hàng = 1
            row_sum = A_p.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            A_p = A_p / row_sum
            self.register_buffer("A_phys", A_p)  # (N, N)
        else:
            self.A_phys = None

        # ----- Node embeddings BẤT ĐỐI XỨNG để học đồ thị có hướng -----
        if graph_mode in ("adaptive", "fused"):
            self.E1 = nn.Parameter(torch.empty(num_nodes, embed_dim))
            self.E2 = nn.Parameter(torch.empty(num_nodes, embed_dim))
            nn.init.xavier_uniform_(self.E1)
            nn.init.xavier_uniform_(self.E2)
        else:
            self.E1 = None
            self.E2 = None

        # ----- Alpha hợp nhất -----
        if graph_mode == "fused":
            if fusion_type == "learnable":
                # Alpha là tham số học được, khởi tạo ~ 0.5 (sigmoid(0)=0.5)
                self._alpha_param = nn.Parameter(torch.zeros(1))
            else:
                self._alpha_fixed = alpha
        
    def _compute_adaptive_adj(self) -> torch.Tensor:
        """
        Tính A_adapt = softmax(ReLU(E1 @ E2.T) / temperature, dim=-1), shape (N, N).

        temperature < 1.0 làm phân phối "sắc nét" hơn (peaky), giảm hiện tượng
        A_adapt suy biến thành phân phối đồng đều (uniform) khi E1 @ E2.T cho
        ra giá trị gần bằng nhau ở mọi cặp nút (Bước 2a debug adaptive graph).

        Ghi chú toán học (Bước 3):
          - AGCRN dùng E @ E.T (1 embedding): ma trận kết quả luôn đối xứng và
            positive semi-definite (PSD). PSD đảm bảo eigenvalue >= 0, gradient
            lan truyền ổn định hơn.
          - AdaptiveTGCN dùng E1 @ E2.T (2 embedding): ma trận bất kỳ, không
            có ràng buộc PSD. Khởi tạo ngẫu nhiên Xavier cho E1, E2 khiến
            E1 @ E2.T có giá trị gần đồng đều (mean ≈ 0, variance nhỏ sau relu),
            dẫn đến softmax → uniform → GCN chỉ làm phép trung bình đơn giản.
          - Giải pháp ngắn hạn: temperature scaling (Bước 2a).
          - Giải pháp dài hạn: tăng embed_dim (Bước 2b) hoặc khởi tạo E2 = E1
            như AGCRN để đảm bảo cấu trúc PSD ban đầu.
        """
        logits  = F.relu(self.E1 @ self.E2.T)          # (N, N), bất đối xứng
        # Chia cho temperature trước softmax (temperature < 1 → sắc nét hơn)
        A_adapt = torch.softmax(logits / self.temperature, dim=-1)   # (N, N)
        return A_adapt

    def _topk_sparsify(self, A: torch.Tensor) -> torch.Tensor:
        """
        Giữ lại top-k giá trị lớn nhất mỗi hàng, đặt phần còn lại về 0,
        sau đó re-normalize để tổng mỗi hàng = 1.

        Args:
            A: (N, N) ma trận kề cần làm thưa.
        Returns:
            A_sparse: (N, N) đã thưa và chuẩn hóa lại.
        """
        k = min(self.top_k, self.num_nodes)
        # topk theo hàng
        topk_vals, topk_idx = A.topk(k, dim=-1)     # (N, k)
        # Tạo mask (N, N) chỉ giữ các cột top-k
        mask = torch.zeros_like(A)
        mask.scatter_(1, topk_idx, 1.0)
        A_sparse = A * mask
        # Re-normalize hàng
        row_sum = A_sparse.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return A_sparse / row_sum

    def compute_graph_reg_loss(self) -> torch.Tensor:
        """
        Trả về graph regularization loss = L2 norm của A_adapt (Frobenius norm).
        Dùng để ép đồ thị học được gọn hơn và tránh overfitting.
        Cộng vào total loss với trọng số nhỏ (ví dụ 1e-4) trong training loop.
        """
        if self.E1 is None:
            return torch.tensor(0.0)
        A_adapt = self._compute_adaptive_adj()
        return (A_adapt ** 2).sum()   # Frobenius norm bình phương

    def forward(self) -> torch.Tensor:
        """
        Trả về ma trận đồ thị cuối cùng A (N, N) theo graph_mode đã cấu hình.
        """
        if self.graph_mode == "physical":
            return self.A_phys

        # Tính adaptive
        A_adapt = self._compute_adaptive_adj()

        if self.graph_mode == "adaptive":
            if self.top_k > 0:
                A_adapt = self._topk_sparsify(A_adapt)
            return A_adapt

        # ---- graph_mode == "fused" ----
        if self.fusion_type == "learnable":
            alpha = torch.sigmoid(self._alpha_param)  # ép về (0, 1)
        else:
            alpha = self._alpha_fixed

        A_fused = alpha * self.A_phys + (1 - alpha) * A_adapt
        if self.top_k > 0:
            A_fused = self._topk_sparsify(A_fused)
        return A_fused
