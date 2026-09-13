# -*- coding: utf-8 -*-
"""
  - MAE   (Mean Absolute Error)
  - RMSE  (Root Mean Squared Error)
  - MAPE  (Mean Absolute Percentage Error) — mask giá trị ~0 để tránh chia-0
  - WMAPE (Weighted Mean Absolute Percentage Error) — tránh nhạy cảm khi traffic ~0
"""

import numpy as np
import torch


def masked_mae(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Tính độ lệch chuẩn tuyệt đối trung bình giữa giá trị thực tế và dự đoán, bỏ qua dữ liệu có giá trị xấp xỉ 0.
        Hàm tính lỗi chính, càng nhỏ càng tốt   """
    mask = true.abs() > 1e-4 if null_val == 0.0 else true != null_val
    mask = mask.float()
    loss = (pred - true).abs() * mask
    return loss.sum() / mask.sum().clamp(min=1)


def masked_rmse(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """ Tính căn bậc hai sai số bình phương trung bình
        Đánh giá độ lệch khi có những dự đoán sai lệch quá lớn, càng nhỏ càng tốt"""
    mask = true.abs() > 1e-4 if null_val == 0.0 else true != null_val
    mask = mask.float()
    loss = (pred - true).pow(2) * mask
    return (loss.sum() / mask.sum().clamp(min=1)).sqrt()


def masked_mape(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Tính phần trăm sai số tuyệt đối
    Đánh giá sai số dưới dạng phần trăm so với giá trị thực tế, càng nhỏ càng tốt"""
    mask = true.abs() > 1e-4 if null_val == 0.0 else true != null_val
    mask = mask.float()
    loss = ((pred - true).abs() / (true.abs().clamp(min=1e-4))) * mask
    return loss.sum() / mask.sum().clamp(min=1) * 100.0


def masked_wmape(pred: torch.Tensor, true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """Tính sai số phần trăm tuyệt đối trung bình có trọng số
       Đánh giá sai số dưới dạng phần trăm so với tổng lưu lượng giao thông trong toàn bộ tập dữ liệu
    """
    mask = true.abs() > 1e-4 if null_val == 0.0 else true != null_val
    mask = mask.float()
    loss = (pred - true).abs() * mask
    denom = true.abs() * mask
    return loss.sum() / denom.sum().clamp(min=1e-4) * 100.0


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    WMAPE: Weighted Mean Absolute Percentage Error (%).
    Công thức: sum(|y_true - y_pred|) / sum(|y_true|) * 100
    """
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_all_metrics(pred: torch.Tensor, true: torch.Tensor):
    """
    Tính MAE, RMSE, MAPE, WMAPE. Pred và true cùng shape.
    """
    return {
        "mae":   masked_mae(pred, true).item(),
        "rmse":  masked_rmse(pred, true).item(),
        "mape":  masked_mape(pred, true).item(),
        "wmape": masked_wmape(pred, true).item(),
    }

"""
Tính kết quả theo từng mốc thời gian cụ thể:
Bước 3 tương ứng 15 phút tới.
Bước 6 tương ứng 30 phút tới.
Bước 12 tương ứng 60 phút tới.
"""
def compute_horizon_metrics(
    pred: np.ndarray, true: np.ndarray,
    horizons: list = [3, 6, 12]
) -> dict:
    results = {}
    for h in horizons:
        idx = h - 1  
        p = torch.tensor(pred[:, idx, :, :], dtype=torch.float32)
        t = torch.tensor(true[:, idx, :, :], dtype=torch.float32)
        minutes = h * 5  
        results[minutes] = compute_all_metrics(p, t)
    return results
