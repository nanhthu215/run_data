import os
import time
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baselines.common.metrics import masked_mae, compute_all_metrics, compute_horizon_metrics

"""Tự động dừng sớm khi không cải thiện, để tránh overfitting"""
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Trả về True nếu nên dừng training."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        """Phục hồi weights tốt nhất"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _get_scaler_params(scaler):
    """Trích xuất mean/std từ scaler (dict hoặc sklearn)."""
    if isinstance(scaler, dict):
        return scaler['mean'][0], scaler['std'][0]
    elif hasattr(scaler, 'mean_'):
        return scaler.mean_[0], scaler.scale_[0]
    return 0.0, 1.0


def _forward_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor = None,
    teacher_forcing_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Gọi model.forward với các tham số phù hợp dựa trên signature của model.
    - Nếu model nhận y và teacher_forcing_ratio: truyền cả hai
    - Nếu model chỉ nhận y: truyền y
    - Nếu model không nhận y: chỉ truyền X
    """
    sig = inspect.signature(model.forward)
    params = sig.parameters
    if y is not None and "y" in params:
        if "teacher_forcing_ratio" in params:
            return model(X, y=y, teacher_forcing_ratio=teacher_forcing_ratio)
        return model(X, y=y)
    return model(X)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    max_batches: int = None,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """Một epoch training. Trả về average train loss (MAE)."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler) if scaler is not None else (0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    for batch_idx, (X, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        X = X.to(device)
        y = y.to(device)

        y_norm = (y - mean_t) / std_t

        optimizer.zero_grad()
        pred = _forward_model(model, X, y=y_norm, teacher_forcing_ratio=teacher_forcing_ratio)

        if pred.dim() == 3:
            pred = pred.unsqueeze(-1)

        loss = masked_mae(pred, y_norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler=None,
    max_batches: int = None,
) -> float:
    """Một epoch evaluation tắt hoàn toàn teacher forcing. Trả về average val loss (MAE, normalized scale)."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    mean, std = _get_scaler_params(scaler) if scaler is not None else (0.0, 1.0)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t  = torch.tensor(std,  dtype=torch.float32, device=device)

    for batch_idx, (X, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        if pred.dim() == 3:
            pred = pred.unsqueeze(-1)

        y_norm = (y - mean_t) / std_t
        loss = masked_mae(pred, y_norm)
        total_loss += loss.item() * X.size(0)
        total_samples += X.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler,
    horizons: list = [3, 6, 12],
) -> dict:
    """
    Chạy inference trên test set, tính metrics theo horizon.
    
    Lưu ý: y (target) đã ở dạng giá trị gốc (không normalize).
    Model được train để output giá trị gốc từ X đã normalize.

    """
    model.eval()
    all_pred = []
    all_true = []

    for X, y in loader:
        X = X.to(device)
        pred = model(X)
        if pred.dim() == 3:
            pred = pred.unsqueeze(-1)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y.numpy())

    pred_arr = np.concatenate(all_pred, axis=0)  # (N, 12, num_nodes, 1)
    true_arr = np.concatenate(all_true, axis=0)  # (N, 12, num_nodes, 1) — giá trị gốc

    # y đã ở giá trị gốc, pred cần inverse transform
    # Scaler là dict với keys 'mean', 'std' (từ prepare_pems_data.py)
    # Chỉ áp dụng cho feature flow (feature 0)
    if isinstance(scaler, dict):
        mean = scaler['mean'][0]
        std  = scaler['std'][0]
    elif hasattr(scaler, 'mean_'):
        mean = scaler.mean_[0]
        std  = scaler.scale_[0]
    else:
        mean = 0.0
        std  = 1.0
    pred_inv = pred_arr * std + mean  # denormalize pred
    true_inv = true_arr               # y đã là giá trị gốc

    return compute_horizon_metrics(pred_inv, true_inv, horizons)


def get_teacher_forcing_ratio(epoch: int, total_epochs: int, start: float = 0.8, end: float = 0.0) -> float:
    """
    Scheduled Sampling: Giảm tuyến tính teacher_forcing_ratio từ start về end qua các epoch.
    Giúp giải quyết Exposure Bias (khi model quen dựa vào ground truth lúc train nhưng phải tự hồi quy lúc val/test).
    """
    if total_epochs <= 0:
        return end
    progress = min(epoch / total_epochs, 1.0)
    return start + (end - start) * progress


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    scaler,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    tf_start: float = 0.8,
    tf_end: float = 0.0,
    teacher_forcing_ratio: float = None,
    patience: int = 10,
    device: torch.device = torch.device("cpu"),
    model_name: str = "model",
    save_dir: str = "checkpoints",
    verbose: bool = True,
    max_batches_per_epoch: int = None,  
) -> dict:
    """
    Training loop đầy đủ với early stopping và Scheduled Sampling (giảm dần teacher forcing).

    chạy vòng lặp: tính tỷ lệ TF giảm dần -> gọi train_epoch -> gọi eval_epoch -> in log -> ktra dừng sớm

    Khi xong hoặc dừng sớm: nạp lại checkpoint tốt nhất -> lưu file best_model.pt và train_summary.json.

    Chạy test_model và in bảng kết quả hoàn chỉnh ra màn hình.
        
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    early_stop = EarlyStopping(patience=patience)

    total_batches = len(train_loader)
    effective_batches = min(max_batches_per_epoch, total_batches) if max_batches_per_epoch else total_batches

    start_ratio = teacher_forcing_ratio if teacher_forcing_ratio is not None else tf_start

    print(f"\n{'='*60}")
    print(f"Training: {model_name} | Epochs: {epochs} | LR: {lr} | Device: {device}")
    print(f"Scheduled Sampling: TF_start={start_ratio:.2f} -> TF_end={tf_end:.2f}")
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    if max_batches_per_epoch:
        print(f"[SMOKE] Max {effective_batches}/{total_batches} batches/epoch")
    print(f"{'='*60}")

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        tf_ratio = get_teacher_forcing_ratio(epoch, epochs, start=start_ratio, end=tf_end)
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            max_batches=max_batches_per_epoch,
            teacher_forcing_ratio=tf_ratio,
        )
        val_loss   = eval_epoch(model, val_loader, device, scaler, max_batches_per_epoch)
        scheduler.step(val_loss)

        elapsed = time.time() - t_ep
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs} | TF_ratio: {tf_ratio:.2f} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f} | {elapsed:.1f}s")

        if early_stop.step(val_loss, model):
            print(f"[EarlyStopping] Stopped at epoch {epoch} | Best Val MAE: {early_stop.best_loss:.4f}")
            break

    # Restore best model
    early_stop.restore_best(model)
    total_time = time.time() - t0
    print(f"\nTraining done in {total_time/60:.1f} min. Best Val MAE: {early_stop.best_loss:.4f}")

    # Lưu checkpoint
    checkpoint_path = os.path.join(save_dir, "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)

    # Evaluate on test set (full test set)
    results = test_model(model, test_loader, device, scaler)

    # Lưu train_summary.json với đường dẫn tương đối (Relative Path)
    root_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    rel_save_dir = os.path.relpath(save_dir, root_dir).replace("\\", "/")
    summary = {
        "model_name": model_name,
        "checkpoint_dir": rel_save_dir,
        "checkpoint_file": f"{rel_save_dir}/best_model.pt",
        "best_val_mae": float(early_stop.best_loss),
        "epochs_trained": epoch,
        "training_time_seconds": round(total_time, 2),
        "results": results,
    }
    summary_path = os.path.join(save_dir, "train_summary.json")
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Test Results — {model_name}]")
    print(f"{'Horizon':>10} {'MAE':>8} {'RMSE':>8} {'MAPE(%)':>10} {'WMAPE(%)':>10}")
    print(f"{'-'*52}")
    for minutes, m in sorted(results.items()):
        wmape_val = f"{m.get('wmape', 0.0):>10.2f}"
        print(f"{minutes:>8}min {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['mape']:>10.2f} {wmape_val}")

    return results
