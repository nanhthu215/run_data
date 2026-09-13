import os
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader


# Mapping dataset name → path
DATASET_ROOTS = {
    "PEMS04": os.path.join(os.path.dirname(__file__), "..", "..", "data", "PEMS04", "processed"),
    "PEMS08": os.path.join(os.path.dirname(__file__), "..", "..", "data", "PEMS08", "processed"),
}


def load_dataset(dataset: str, batch_size: int = 64, num_workers: int = 0, fold=None):
    """
    Load dữ liệu và trả về (train_loader, val_loader, test_loader, scaler, adj_matrix).

    Args:
        dataset: "PEMS04" hoặc "PEMS08"
        batch_size: batch size cho DataLoader
        num_workers: số worker cho DataLoader
        fold: None hoặc "default" = load từ processed/ (hành vi cũ, backward compatible)
              0, 1, 2, 3 = load từ fold_0/, fold_1/, ... (Expanding Window CV)

    Returns:
        train_loader, val_loader, test_loader: DataLoader PyTorch
        scaler: dict (có 'mean', 'std')
        adj_matrix: np.ndarray (num_nodes, num_nodes)
    """
    base_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", dataset.upper()
    )
    base_root = os.path.normpath(base_root)

    if fold is not None and fold != "default":
        root = os.path.join(base_root, f"fold_{fold}")
    else:
        root = os.path.join(base_root, "processed")

    assert os.path.exists(root), f"Data directory not found: {root}"

    # Load numpy arrays
    X_train = np.load(os.path.join(root, "X_train.npy")) 
    X_val   = np.load(os.path.join(root, "X_val.npy"))
    X_test  = np.load(os.path.join(root, "X_test.npy"))
    y_train = np.load(os.path.join(root, "y_train.npy")) 
    y_val   = np.load(os.path.join(root, "y_val.npy"))
    y_test  = np.load(os.path.join(root, "y_test.npy"))

    # Load scaler
    with open(os.path.join(root, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    # Load adjacency matrix
    adj_matrix = np.load(os.path.join(root, "adj_matrix.npy"))  # (num_nodes, num_nodes)

    # Convert to tensors — dùng float32
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    fold_label = f"fold_{fold}" if fold is not None and fold != "default" else "default"
    print(f"[DataLoader] {dataset} ({fold_label}): train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"[DataLoader] adj_matrix: {adj_matrix.shape}, num_nodes={adj_matrix.shape[0]}")

    return train_loader, val_loader, test_loader, scaler, adj_matrix


def get_num_nodes(dataset: str) -> int:
    return {"PEMS04": 307, "PEMS08": 170}[dataset]
