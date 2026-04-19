"""
Training loop for Temporal Graph Transformer 

Handles:
  - Node feature augmentation (raw → [raw, z-score, 3d-slope, 5d-vol])
  - PyTorch Dataset bridging data pipeline + graph builder + model
  - Multi-task loss (CE ×0.2 + MSE ×0.4 + MAE ×0.4)
  - Walk-forward training with expanding window
  - Early stopping, cosine scheduler, gradient clipping
"""

import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from configs.config import Config
from models.temporal_graph_transformer import TemporalGraphTransformer
from utils.graph_builder import DynamicGraphBuilder


# ===========================================================================
# 1. Node Feature Augmentation
# ===========================================================================

def augment_node_features(df: pd.DataFrame, indicator_cols: List[str]) -> np.ndarray:
    """
    Augment each indicator from 1 feature to 4 features:
      [raw, z-score(20), 3d-slope, 5d-volatility]

    All computations are strictly causal (rolling lookback only).

    Args:
        df: DataFrame with indicator columns (already scaled).
        indicator_cols: list of 24 indicator column names.

    Returns:
        (T, N, 4) float32 array where T=len(df), N=len(indicator_cols).
    """
    T = len(df)
    N = len(indicator_cols)
    features = np.zeros((T, N, 4), dtype=np.float32)

    for j, col in enumerate(indicator_cols):
        vals = df[col].values.astype(np.float64)

        # Feature 0: raw (already scaled)
        features[:, j, 0] = vals

        # Feature 1: rolling z-score (20-day)
        s = pd.Series(vals)
        roll_mean = s.rolling(20, min_periods=5).mean()
        roll_std = s.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((s - roll_mean) / roll_std).fillna(0.0).values
        # Clip extreme z-scores
        features[:, j, 1] = np.clip(z, -3.0, 3.0)

        # Feature 2: 3-day slope (finite difference)
        slope = np.zeros(T)
        slope[2:] = (vals[2:] - vals[:-2]) / 2.0
        features[:, j, 2] = slope

        # Feature 3: 5-day volatility (rolling std)
        vol = pd.Series(vals).rolling(5, min_periods=2).std().fillna(0.0).values
        features[:, j, 3] = vol

    # Replace any remaining NaNs
    features = np.nan_to_num(features, nan=0.0)
    return features


# ===========================================================================
# 2. PyTorch Dataset
# ===========================================================================

class ForexGraphDataset(Dataset):
    """
    Dataset for TGT model. Each sample is a (seq_len, n_nodes, 4) tensor
    with associated graph snapshots and targets.

    Graphs are shared across the batch (same graph for same timestep).
    Graph selection: for sequence ending at time t, each internal timestep
    uses the graph computed at that point.
    """

    def __init__(
        self,
        node_features: np.ndarray,
        targets: Dict[str, np.ndarray],
        graph_snapshots: List[Dict],
        seq_len: int,
        start_offset: int = 0,
    ):
        """
        Args:
            node_features: (T_total, N, 4) full augmented features for this split.
            targets: dict with 'direction', 'return', 'volatility' arrays of length T_total.
            graph_snapshots: list of graph dicts, one per row in the split.
            seq_len: sequence length (30).
            start_offset: how many rows at the start of graph_snapshots to skip
                          (to align with node_features indexing).
        """
        self.node_features = node_features
        self.seq_len = seq_len
        self.start_offset = start_offset

        # Valid indices: need seq_len rows of history
        self.n_samples = len(node_features) - seq_len
        assert self.n_samples > 0, f"Not enough data: {len(node_features)} rows, need > {seq_len}"

        # Targets at the prediction point (end of each sequence)
        self.y_direction = targets["direction"][seq_len:]
        self.y_return = targets["return"][seq_len:]
        self.y_volatility = targets["volatility"][seq_len:]
        self.graph_snapshots = graph_snapshots

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict:
        # Sequence of node features: (seq_len, N, 4)
        x = self.node_features[idx:idx + self.seq_len]

        # Graph indices for this sequence
        graph_idx_start = self.start_offset + idx
        graph_indices = list(range(graph_idx_start, graph_idx_start + self.seq_len))

        return {
            "node_features": torch.from_numpy(x),
            "graph_indices": graph_indices,
            "y_direction": torch.tensor(self.y_direction[idx], dtype=torch.long),
            "y_return": torch.tensor(self.y_return[idx], dtype=torch.float32),
            "y_volatility": torch.tensor(self.y_volatility[idx], dtype=torch.float32),
        }


def collate_forex(batch: List[Dict], graph_snapshots: List[Dict], device: torch.device) -> Dict:
    """
    Custom collate that stacks node features and resolves graph snapshots.

    Returns a dict ready for model.forward():
      - node_features: (B, T, N, 4)
      - edge_indices: list of T tensors (2, E_t) on device
      - edge_weights: list of T tensors (E_t,) on device
      - y_direction: (B,)
      - y_return: (B,)
      - y_volatility: (B,)
    """
    B = len(batch)
    T = batch[0]["node_features"].size(0)

    node_features = torch.stack([b["node_features"] for b in batch]).to(device)

    # Resolve graphs for each timestep
    # All samples in batch share the same seq_len; graph may differ per sample
    # but for efficiency, use the graph from the FIRST sample in the batch
    # (graphs change only every 20 days, so within a batch they're nearly identical)
    ref_indices = batch[0]["graph_indices"]
    edge_indices = []
    edge_weights = []
    for t in range(T):
        g_idx = ref_indices[t]
        if 0 <= g_idx < len(graph_snapshots):
            g = graph_snapshots[g_idx]
            ei = torch.from_numpy(g["edge_index"]).to(device)
            ew = torch.from_numpy(g["edge_weight"]).to(device)
        else:
            # Fallback: empty graph
            ei = torch.zeros((2, 0), dtype=torch.long, device=device)
            ew = torch.zeros(0, dtype=torch.float32, device=device)
        edge_indices.append(ei)
        edge_weights.append(ew)

    y_direction = torch.stack([b["y_direction"] for b in batch]).to(device)
    y_return = torch.stack([b["y_return"] for b in batch]).to(device)
    y_volatility = torch.stack([b["y_volatility"] for b in batch]).to(device)

    return {
        "node_features": node_features,
        "edge_indices": edge_indices,
        "edge_weights": edge_weights,
        "y_direction": y_direction,
        "y_return": y_return,
        "y_volatility": y_volatility,
    }


# ===========================================================================
# 3. Multi-Task Loss
# ===========================================================================

class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss:
      L = w_dir * CE(direction) + w_ret * MSE(return) + w_vol * MAE(volatility)
    """

    def __init__(self, w_direction: float = 0.2, w_return: float = 0.4, w_volatility: float = 0.4):
        super().__init__()
        self.w_direction = w_direction
        self.w_return = w_return
        self.w_volatility = w_volatility
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        y_direction: torch.Tensor,
        y_return: torch.Tensor,
        y_volatility: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor for backprop
            loss_dict: individual losses for logging
        """
        l_dir = self.ce(pred["direction"], y_direction)
        l_ret = self.mse(pred["return"].squeeze(-1), y_return)
        l_vol = F.l1_loss(pred["volatility"].squeeze(-1), y_volatility)

        total = (
            self.w_direction * l_dir +
            self.w_return * l_ret +
            self.w_volatility * l_vol
        )

        return total, {
            "direction_ce": l_dir.item(),
            "return_mse": l_ret.item(),
            "volatility_mae": l_vol.item(),
            "total": total.item(),
        }


# ===========================================================================
# 4. Trainer
# ===========================================================================

class WalkForwardTrainer:
    """
    Walk-forward trainer for TGT.

    For each fold:
      1. Build augmented node features (fit z-score stats on train)
      2. Build graph sequences (strictly causal)
      3. Train with early stopping on validation loss
      4. Evaluate on test set
      5. Collect results across folds
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.results: List[Dict] = []

    def _build_fold_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[ForexGraphDataset, ForexGraphDataset, ForexGraphDataset, List[Dict]]:
        """Build datasets for a single fold."""
        seq_len = self.cfg.data.sequence_length

        # Concatenate for contiguous graph building
        # (graph at val/test time can reference trailing train data)
        full_df = pd.concat([train_df, val_df, test_df])
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)

        # 1. Augment node features on full timeline
        node_feats = augment_node_features(full_df, feature_cols)  # (T_total, N, 4)

        # 2. Build graph sequence over full timeline
        graph_builder = DynamicGraphBuilder(self.cfg.graph, feature_cols)
        all_graphs = graph_builder.build_graph_sequence(full_df, start_idx=0)

        # Pad graphs if build_graph_sequence started later
        if len(all_graphs) < len(full_df):
            # Pad front with empty graphs
            pad = len(full_df) - len(all_graphs)
            empty_graph = {
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_weight": np.zeros(0, dtype=np.float32),
                "adj_matrix": np.zeros((len(feature_cols), len(feature_cols))),
                "num_edges": 0,
                "computed_at": full_df.index[0],
            }
            all_graphs = [empty_graph] * pad + all_graphs

        # 3. Build targets
        def extract_targets(df):
            return {
                "direction": df["target_direction"].values.astype(np.int64),
                "return": df["target_return"].values.astype(np.float32),
                "volatility": df["target_volatility"].values.astype(np.float32),
            }

        # 4. Create datasets
        # Train: node_feats[0:n_train], graphs start at 0
        train_ds = ForexGraphDataset(
            node_feats[:n_train], extract_targets(train_df),
            all_graphs, seq_len, start_offset=0,
        )
        # Val: node_feats[n_train:n_train+n_val], graphs offset by n_train
        val_ds = ForexGraphDataset(
            node_feats[n_train:n_train + n_val], extract_targets(val_df),
            all_graphs, seq_len, start_offset=n_train,
        )
        # Test: node_feats[n_train+n_val:], graphs offset by n_train+n_val
        test_ds = ForexGraphDataset(
            node_feats[n_train + n_val:], extract_targets(test_df),
            all_graphs, seq_len, start_offset=n_train + n_val,
        )

        stats = graph_builder.get_stats(all_graphs)
        print(f"   📊 Graphs: {stats['unique_graphs']} unique, "
              f"{stats['edge_count_mean']:.0f} avg edges")
        print(f"   📊 Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

        return train_ds, val_ds, test_ds, all_graphs

    def _train_one_epoch(
        self,
        model: TemporalGraphTransformer,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: MultiTaskLoss,
        graph_snapshots: List[Dict],
    ) -> Dict[str, float]:
        """Train for one epoch. Returns average losses."""
        model.train()
        total_losses = {"direction_ce": 0, "return_mse": 0, "volatility_mae": 0, "total": 0}
        n_batches = 0

        for batch_raw in loader:
            batch = collate_forex(batch_raw, graph_snapshots, self.device)

            optimizer.zero_grad()
            pred = model(
                batch["node_features"],
                batch["edge_indices"],
                batch["edge_weights"],
            )

            loss, loss_dict = criterion(
                pred, batch["y_direction"], batch["y_return"], batch["y_volatility"],
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.max_grad_norm)
            optimizer.step()

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def _evaluate(
        self,
        model: TemporalGraphTransformer,
        loader: DataLoader,
        criterion: MultiTaskLoss,
        graph_snapshots: List[Dict],
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Evaluate on val/test set. Returns losses and predictions."""
        model.eval()
        total_losses = {"direction_ce": 0, "return_mse": 0, "volatility_mae": 0, "total": 0}
        n_batches = 0

        all_preds = {"direction": [], "return": [], "volatility": []}
        all_targets = {"direction": [], "return": [], "volatility": []}

        for batch_raw in loader:
            batch = collate_forex(batch_raw, graph_snapshots, self.device)

            pred = model(
                batch["node_features"],
                batch["edge_indices"],
                batch["edge_weights"],
            )

            _, loss_dict = criterion(
                pred, batch["y_direction"], batch["y_return"], batch["y_volatility"],
            )

            for k in total_losses:
                total_losses[k] += loss_dict[k]
            n_batches += 1

            all_preds["direction"].append(pred["direction"].cpu().numpy())
            all_preds["return"].append(pred["return"].squeeze(-1).cpu().numpy())
            all_preds["volatility"].append(pred["volatility"].squeeze(-1).cpu().numpy())
            all_targets["direction"].append(batch["y_direction"].cpu().numpy())
            all_targets["return"].append(batch["y_return"].cpu().numpy())
            all_targets["volatility"].append(batch["y_volatility"].cpu().numpy())

        avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
        preds = {k: np.concatenate(v) for k, v in all_preds.items()}
        targets = {k: np.concatenate(v) for k, v in all_targets.items()}

        return avg_losses, {"predictions": preds, "targets": targets}

    def _train_fold(
        self,
        fold_id: int,
        train_ds: ForexGraphDataset,
        val_ds: ForexGraphDataset,
        test_ds: ForexGraphDataset,
        graph_snapshots: List[Dict],
    ) -> Dict:
        """Train a single fold with early stopping."""
        cfg = self.cfg

        # DataLoaders (custom collate is handled inside _train_one_epoch)
        # We pass raw batch dicts and collate in the loop
        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.batch_size, shuffle=True,
            collate_fn=list,  # Return list of dicts
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.train.batch_size, shuffle=False,
            collate_fn=list,
        )
        test_loader = DataLoader(
            test_ds, batch_size=cfg.train.batch_size, shuffle=False,
            collate_fn=list,
        )

        # Fresh model per fold
        model = TemporalGraphTransformer(cfg.model, cfg.data).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=1e-6,
        )

        criterion = MultiTaskLoss(
            cfg.train.loss_weight_direction,
            cfg.train.loss_weight_return,
            cfg.train.loss_weight_volatility,
        )

        # Early stopping
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        best_epoch = 0

        print(f"\n   🔄 Training fold {fold_id} ({cfg.train.epochs} max epochs, "
              f"patience={cfg.train.patience})")

        for epoch in range(1, cfg.train.epochs + 1):
            t0 = time.time()

            train_losses = self._train_one_epoch(
                model, train_loader, optimizer, criterion, graph_snapshots,
            )
            val_losses, _ = self._evaluate(
                model, val_loader, criterion, graph_snapshots,
            )

            scheduler.step()
            elapsed = time.time() - t0

            # Early stopping check
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                best_epoch = epoch
                marker = " ⭐"
            else:
                patience_counter += 1
                marker = ""

            if epoch <= 5 or epoch % 10 == 0 or patience_counter == 0 or epoch == cfg.train.epochs:
                print(
                    f"      Epoch {epoch:3d} | "
                    f"train={train_losses['total']:.4f} | "
                    f"val={val_losses['total']:.4f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.1f}s{marker}"
                )

            if patience_counter >= cfg.train.patience:
                print(f"      ⏹️  Early stop at epoch {epoch} (best={best_epoch})")
                break

        # Restore best model and evaluate on test
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_losses, test_output = self._evaluate(
            model, test_loader, criterion, graph_snapshots,
        )

        # Direction accuracy
        dir_preds = test_output["predictions"]["direction"].argmax(axis=1)
        dir_targets = test_output["targets"]["direction"]
        dir_accuracy = (dir_preds == dir_targets).mean()

        print(f"      📊 Test | loss={test_losses['total']:.4f} | "
              f"dir_acc={dir_accuracy:.3f} | best_epoch={best_epoch}")

        return {
            "fold_id": fold_id,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "test_losses": test_losses,
            "test_direction_accuracy": dir_accuracy,
            "test_predictions": test_output["predictions"],
            "test_targets": test_output["targets"],
            "model_state": best_model_state,
        }

    def train_walk_forward(self, df: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
        """
        Run full walk-forward training.

        Args:
            df: Full DataFrame with indicators + targets (output of data pipeline).
            feature_cols: list of 24 indicator column names.

        Returns:
            List of fold results.
        """
        from utils.data_pipeline import walk_forward_splits, scale_features

        print("=" * 60)
        print("🚀 Walk-Forward Training")
        print("=" * 60)

        fold_results = []

        for fold_data in walk_forward_splits(
            df,
            initial_train_years=self.cfg.data.initial_train_years,
            val_years=self.cfg.data.validation_years,
            test_years=self.cfg.data.test_years,
            step_months=self.cfg.data.walk_forward_step_months,
            expanding=self.cfg.data.expanding_window,
        ):
            fold_id = fold_data["fold_id"]
            print(f"\n{'─' * 50}")
            print(f"📂 Fold {fold_id} | "
                  f"train→{fold_data['train_end'].date()} | "
                  f"val→{fold_data['val_end'].date()} | "
                  f"test→{fold_data['test_end'].date()}")

            # Scale features (fit on train only)
            train_scaled, val_scaled, test_scaled, scaler = scale_features(
                fold_data["train"], fold_data["val"], fold_data["test"], feature_cols,
            )

            # Build datasets with graphs
            train_ds, val_ds, test_ds, graphs = self._build_fold_data(
                train_scaled, val_scaled, test_scaled, feature_cols,
            )

            # Train this fold
            result = self._train_fold(fold_id, train_ds, val_ds, test_ds, graphs)
            fold_results.append(result)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"📊 Walk-Forward Summary ({len(fold_results)} folds)")
        print(f"{'=' * 60}")

        accs = [r["test_direction_accuracy"] for r in fold_results]
        losses = [r["test_losses"]["total"] for r in fold_results]
        print(f"   Direction accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
        print(f"   Test loss:          {np.mean(losses):.4f} ± {np.std(losses):.4f}")

        for r in fold_results:
            print(f"   Fold {r['fold_id']:2d}: acc={r['test_direction_accuracy']:.3f}, "
                  f"loss={r['test_losses']['total']:.4f}, best_ep={r['best_epoch']}")

        self.results = fold_results
        return fold_results

    def train_single_split(self, pipeline_output: Dict) -> Dict:
        """
        Train on a single chronological split (for quick testing).
        Uses the output of data_pipeline.run_pipeline().
        """
        cfg = self.cfg
        feature_cols = pipeline_output["feature_cols"]

        print("=" * 60)
        print("🚀 Single-Split Training (quick test mode)")
        print("=" * 60)

        train_ds, val_ds, test_ds, graphs = self._build_fold_data(
            pipeline_output["train_df"],
            pipeline_output["val_df"],
            pipeline_output["test_df"],
            feature_cols,
        )

        result = self._train_fold(0, train_ds, val_ds, test_ds, graphs)
        self.results = [result]

        print(f"\n✅ Done | dir_acc={result['test_direction_accuracy']:.3f} | "
              f"test_loss={result['test_losses']['total']:.4f}")
        return result


# ===========================================================================
# 5. Smoke Test
# ===========================================================================

def test_trainer():
    """Quick smoke test with synthetic data."""
    import warnings
    warnings.filterwarnings("ignore")

    cfg = Config()
    # Shrink for speed
    cfg.train.epochs = 3
    cfg.train.patience = 2
    cfg.train.batch_size = 8
    cfg.data.sequence_length = 10

    print("=" * 60)
    print("🧪 Trainer Smoke Test")
    print("=" * 60)

    # Synthetic data
    np.random.seed(cfg.train.seed)
    n_days = 150
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    feature_cols = cfg.data.feature_nodes

    data = {}
    base = np.cumsum(np.random.randn(n_days) * 0.01)
    for name in feature_cols:
        data[name] = base + np.random.randn(n_days) * 0.3

    # Add targets
    data["target_direction"] = np.random.randint(0, 3, n_days)
    data["target_return"] = np.random.randn(n_days) * 0.01
    data["target_volatility"] = np.abs(np.random.randn(n_days) * 0.005)

    df = pd.DataFrame(data, index=dates)

    # Split
    t_end = int(n_days * 0.6)
    v_end = int(n_days * 0.8)
    train_df = df.iloc[:t_end].copy()
    val_df = df.iloc[t_end:v_end].copy()
    test_df = df.iloc[v_end:].copy()

    pipeline_output = {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
    }

    # Test node feature augmentation
    print("\n1️⃣  Testing node feature augmentation...")
    feats = augment_node_features(train_df, feature_cols)
    print(f"   Shape: {feats.shape} (expected ({t_end}, {len(feature_cols)}, 4))")
    assert feats.shape == (t_end, len(feature_cols), 4)
    assert np.isfinite(feats).all(), "NaN/Inf in augmented features!"
    print("   ✅ Augmentation OK")

    # Test dataset
    print("\n2️⃣  Testing ForexGraphDataset...")
    graph_builder = DynamicGraphBuilder(cfg.graph, feature_cols)
    graphs = graph_builder.build_graph_sequence(train_df, start_idx=0)
    if len(graphs) < len(train_df):
        empty = {
            "edge_index": np.zeros((2, 0), dtype=np.int64),
            "edge_weight": np.zeros(0, dtype=np.float32),
            "adj_matrix": np.zeros((len(feature_cols), len(feature_cols))),
            "num_edges": 0,
            "computed_at": train_df.index[0],
        }
        graphs = [empty] * (len(train_df) - len(graphs)) + graphs

    targets = {
        "direction": train_df["target_direction"].values.astype(np.int64),
        "return": train_df["target_return"].values.astype(np.float32),
        "volatility": train_df["target_volatility"].values.astype(np.float32),
    }
    ds = ForexGraphDataset(feats, targets, graphs, cfg.data.sequence_length)
    print(f"   Samples: {len(ds)}")
    sample = ds[0]
    print(f"   Sample node_features: {sample['node_features'].shape}")
    print("   ✅ Dataset OK")

    # Test full training
    print("\n3️⃣  Testing single-split training...")
    trainer = WalkForwardTrainer(cfg)
    result = trainer.train_single_split(pipeline_output)
    assert "test_direction_accuracy" in result
    print(f"   ✅ Training OK (acc={result['test_direction_accuracy']:.3f})")

    print(f"\n✅ All trainer tests passed!")


if __name__ == "__main__":
    test_trainer()
