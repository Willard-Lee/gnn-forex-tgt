"""
LSTM baseline for EUR/USD forecasting.

Simple 2-layer LSTM → FC → 3 heads (direction, return, volatility).
No graph structure — just raw indicator sequences.
Same multi-task loss and walk-forward protocol as TGT.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple

from configs.config import Config


class LSTMModel(nn.Module):
    """2-layer LSTM with 3 prediction heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.direction_head = nn.Linear(hidden_dim, 3)
        self.return_head = nn.Linear(hidden_dim, 1)
        self.volatility_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (B, T, input_dim) → predictions dict."""
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1])  # Last timestep
        h = self.shared(h)
        return {
            "direction": self.direction_head(h),
            "return": self.return_head(h),
            "volatility": self.volatility_head(h),
        }


class SequenceDataset(Dataset):
    """Simple sequence dataset for LSTM baseline."""

    def __init__(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        self.X = torch.from_numpy(X)
        self.y_dir = torch.from_numpy(y["direction"])
        self.y_ret = torch.from_numpy(y["return"])
        self.y_vol = torch.from_numpy(y["volatility"])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_dir[idx], self.y_ret[idx], self.y_vol[idx]


def train_lstm_fold(
    train_X: np.ndarray, train_y: Dict,
    val_X: np.ndarray, val_y: Dict,
    test_X: np.ndarray, test_y: Dict,
    cfg: Config, fold_id: int = 0,
) -> Dict:
    """Train LSTM on a single fold. Returns same format as TGT trainer."""
    device = torch.device(cfg.device)

    train_ds = SequenceDataset(train_X, train_y)
    val_ds = SequenceDataset(val_X, val_y)
    test_ds = SequenceDataset(test_X, test_y)

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size)

    input_dim = train_X.shape[2]
    model = LSTMModel(input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    ce = nn.CrossEntropyLoss()
    w_dir, w_ret, w_vol = cfg.train.loss_weight_direction, cfg.train.loss_weight_return, cfg.train.loss_weight_volatility

    best_val_loss = float("inf")
    best_state = None
    patience = 0
    best_epoch = 0

    for epoch in range(1, cfg.train.epochs + 1):
        # Train
        model.train()
        for X, yd, yr, yv in train_loader:
            X, yd, yr, yv = X.to(device), yd.to(device), yr.to(device), yv.to(device)
            pred = model(X)
            loss = (w_dir * ce(pred["direction"], yd) +
                    w_ret * F.mse_loss(pred["return"].squeeze(-1), yr) +
                    w_vol * F.l1_loss(pred["volatility"].squeeze(-1), yv))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
        scheduler.step()

        # Val
        model.eval()
        val_loss = 0
        n = 0
        with torch.no_grad():
            for X, yd, yr, yv in val_loader:
                X, yd, yr, yv = X.to(device), yd.to(device), yr.to(device), yv.to(device)
                pred = model(X)
                loss = (w_dir * ce(pred["direction"], yd) +
                        w_ret * F.mse_loss(pred["return"].squeeze(-1), yr) +
                        w_vol * F.l1_loss(pred["volatility"].squeeze(-1), yv))
                val_loss += loss.item()
                n += 1
        val_loss /= max(n, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            best_epoch = epoch
        else:
            patience += 1

        if epoch <= 3 or epoch % 10 == 0 or patience == 0:
            print(f"      Epoch {epoch:3d} | val={val_loss:.4f}{'  ⭐' if patience == 0 else ''}")

        if patience >= cfg.train.patience:
            print(f"      ⏹️  Early stop at epoch {epoch}")
            break

    # Test
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_dir, all_ret, all_vol = [], [], []
    tgt_dir, tgt_ret, tgt_vol = [], [], []
    with torch.no_grad():
        for X, yd, yr, yv in test_loader:
            pred = model(X.to(device))
            all_dir.append(pred["direction"].cpu().numpy())
            all_ret.append(pred["return"].squeeze(-1).cpu().numpy())
            all_vol.append(pred["volatility"].squeeze(-1).cpu().numpy())
            tgt_dir.append(yd.numpy())
            tgt_ret.append(yr.numpy())
            tgt_vol.append(yv.numpy())

    preds = {
        "direction": np.concatenate(all_dir),
        "return": np.concatenate(all_ret),
        "volatility": np.concatenate(all_vol),
    }
    targets = {
        "direction": np.concatenate(tgt_dir),
        "return": np.concatenate(tgt_ret),
        "volatility": np.concatenate(tgt_vol),
    }

    dir_acc = (preds["direction"].argmax(1) == targets["direction"]).mean()
    print(f"      📊 Test | dir_acc={dir_acc:.3f} | best_epoch={best_epoch}")

    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_losses": {"total": best_val_loss},
        "test_direction_accuracy": float(dir_acc),
        "test_predictions": preds,
        "test_targets": targets,
        "model_state": best_state,
    }
