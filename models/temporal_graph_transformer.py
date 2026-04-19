"""
Temporal Graph Transformer (TGT) for EUR/USD Forex Forecasting v2.

Architecture:
  For each timestep t in [1..seq_len]:
    24 indicator nodes × 4 features → FC projection → GAT (2-layer) → mean pool → 128-dim snapshot
  Stack seq_len snapshots → [CLS] + positional encoding → Transformer encoder → [CLS] output →
    shared FC →
      Head 1: direction (3-class)
      Head 2: returns (scalar)
      Head 3: volatility (scalar)

Key design choice: GAT runs at EVERY timestep (not just the last one).
This lets the Transformer attend over a sequence of graph-enriched snapshots,
capturing both spatial (indicator relationships) and temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from models.layers import GATBlock, PositionalEncoding
from configs.config import ModelConfig, DataConfig


class TemporalGraphTransformer(nn.Module):
    """
    Full TGT model.

    Input per sample:
      - node_features: (seq_len, n_nodes, node_feature_dims) — 4 features per node per timestep
      - edge_index: (2, E) — graph structure (may vary per timestep)
      - edge_weight: (E,) — edge weights

    Output:
      - direction_logits: (batch, 3)
      - return_pred: (batch, 1)
      - volatility_pred: (batch, 1)
    """

    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.n_nodes = len(data_cfg.feature_nodes)
        self.node_feature_dims = data_cfg.node_feature_dims
        self.seq_len = data_cfg.sequence_length

        # --- Node feature projection ---
        # 4 raw features per node → gat_in_dim
        self.node_fc = nn.Sequential(
            nn.Linear(self.node_feature_dims, model_cfg.gat_in_dim),
            nn.LayerNorm(model_cfg.gat_in_dim),
            nn.ELU(),
        )

        # --- Spatial: GAT block (applied at each timestep) ---
        self.gat = GATBlock(
            in_dim=model_cfg.gat_in_dim,
            hidden=model_cfg.gat_hidden,
            out_dim=model_cfg.gat_out_dim,
            heads_l1=model_cfg.gat_heads_l1,
            heads_l2=model_cfg.gat_heads_l2,
            dropout=model_cfg.gat_dropout,
            edge_dropout=model_cfg.edge_dropout,
        )

        # --- Graph pooling: mean pool + project to snapshot dim ---
        self.pool_fc = nn.Sequential(
            nn.Linear(model_cfg.gat_out_dim, model_cfg.graph_snapshot_dim),
            nn.LayerNorm(model_cfg.graph_snapshot_dim),
            nn.ELU(),
        )

        # --- Temporal: Transformer encoder ---
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_cfg.transformer_d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(
            model_cfg.transformer_d_model,
            max_len=self.seq_len + 1,  # +1 for CLS
            dropout=model_cfg.transformer_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.transformer_d_model,
            nhead=model_cfg.transformer_nhead,
            dim_feedforward=model_cfg.transformer_dim_feedforward,
            dropout=model_cfg.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg.transformer_num_layers,
        )

        # --- Prediction heads ---
        self.shared_fc = nn.Sequential(
            nn.Linear(model_cfg.transformer_d_model, model_cfg.shared_dim),
            nn.LayerNorm(model_cfg.shared_dim) if model_cfg.use_layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Dropout(model_cfg.dropout),
        )

        self.direction_head = nn.Linear(model_cfg.shared_dim, model_cfg.num_direction_classes)
        self.return_head = nn.Linear(model_cfg.shared_dim, 1)
        self.volatility_head = nn.Sequential(
            nn.Linear(model_cfg.shared_dim, 1),
            nn.Softplus(),  # Volatility is non-negative
        )

    def _process_single_timestep(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process one timestep through node_fc + GAT + pooling.

        Args:
            node_feats: (B, N, F) — batch of node features for one timestep
            edge_index: (2, E) — shared graph structure
            edge_weight: (E,) — edge weights

        Returns:
            (B,  snapshot_dim) — pooled graph snapshot
        """
        B, N, F = node_feats.shape

        # Project node features
        h = self.node_fc(node_feats)  # (B, N, gat_in_dim)

        # GAT needs (N, in_dim) — process each sample in batch
        # For efficiency, batch all nodes together with offset edge indices
        h_flat = h.reshape(B * N, -1)  # (B*N, gat_in_dim)

        # Offset edge_index for batched graph processing
        if edge_index.size(1) > 0:
            offsets = torch.arange(B, device=h.device).unsqueeze(1) * N  # (B, 1)
            # Repeat edge_index for each batch item with offset
            batch_edge_index = []
            batch_edge_weight = []
            for b in range(B):
                batch_edge_index.append(edge_index + b * N)
                if edge_weight is not None:
                    batch_edge_weight.append(edge_weight)

            batch_edge_index = torch.cat(batch_edge_index, dim=1)  # (2, B*E)
            if edge_weight is not None:
                batch_edge_weight = torch.cat(batch_edge_weight)  # (B*E,)
            else:
                batch_edge_weight = None
        else:
            batch_edge_index = edge_index
            batch_edge_weight = edge_weight

        # GAT forward
        h_gat = self.gat(h_flat, batch_edge_index, batch_edge_weight)  # (B*N, gat_out_dim)

        # Reshape back and mean pool over nodes
        h_gat = h_gat.reshape(B, N, -1)  # (B, N, gat_out_dim)
        h_pool = h_gat.mean(dim=1)  # (B, gat_out_dim)

        # Project to snapshot dim
        snapshot = self.pool_fc(h_pool)  # (B, snapshot_dim)
        return snapshot

    def forward(
        self,
        node_features: torch.Tensor,
        edge_indices: List[torch.Tensor],
        edge_weights: List[Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            node_features: (B, T, N, F) — batch of sequences of node features
                B = batch size
                T = sequence length (30)
                N = number of nodes (24)
                F = node feature dims (4)
            edge_indices: list of T tensors, each (2, E_t) — graph per timestep
            edge_weights: list of T tensors, each (E_t,) — weights per timestep

        Returns:
            dict with:
                "direction": (B, 3) logits
                "return": (B, 1) predicted return
                "volatility": (B, 1) predicted volatility
        """
        B, T, N, F = node_features.shape

        # Process each timestep through GAT → snapshot
        snapshots = []
        for t in range(T):
            snapshot = self._process_single_timestep(
                node_features[:, t],        # (B, N, F)
                edge_indices[t],            # (2, E_t)
                edge_weights[t],            # (E_t,)
            )
            snapshots.append(snapshot)

        # Stack snapshots: (B, T, snapshot_dim)
        seq = torch.stack(snapshots, dim=1)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        seq = torch.cat([cls, seq], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        seq = self.pos_encoding(seq)

        # Transformer encoder
        seq = self.transformer(seq)  # (B, T+1, d_model)

        # Extract [CLS] output
        cls_out = seq[:, 0]  # (B, d_model)

        # Shared FC
        shared = self.shared_fc(cls_out)  # (B, shared_dim)

        # Prediction heads
        direction = self.direction_head(shared)     # (B, 3)
        ret = self.return_head(shared)              # (B, 1)
        vol = self.volatility_head(shared)          # (B, 1)

        return {
            "direction": direction,
            "return": ret,
            "volatility": vol,
        }

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        counts["node_fc"] = sum(p.numel() for p in self.node_fc.parameters())
        counts["gat"] = sum(p.numel() for p in self.gat.parameters())
        counts["pool_fc"] = sum(p.numel() for p in self.pool_fc.parameters())
        counts["cls_token"] = self.cls_token.numel()
        counts["pos_encoding"] = 0  # Buffer, not parameter
        counts["transformer"] = sum(p.numel() for p in self.transformer.parameters())
        counts["shared_fc"] = sum(p.numel() for p in self.shared_fc.parameters())
        counts["direction_head"] = sum(p.numel() for p in self.direction_head.parameters())
        counts["return_head"] = sum(p.numel() for p in self.return_head.parameters())
        counts["volatility_head"] = sum(p.numel() for p in self.volatility_head.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ===========================================================================
# Smoke Test
# ===========================================================================

def test_model():
    """Verify forward pass with synthetic data."""
    from configs.config import Config
    import numpy as np

    cfg = Config()

    print("=" * 60)
    print("🧪 TGT Model Smoke Test")
    print("=" * 60)

    model = TemporalGraphTransformer(cfg.model, cfg.data)

    # Print parameter counts
    counts = model.count_parameters()
    print(f"\n📊 Parameters:")
    for name, count in counts.items():
        print(f"   {name:20s}: {count:>10,}")

    # Synthetic input
    B = 4   # batch size
    T = cfg.data.sequence_length  # 30
    N = len(cfg.data.feature_nodes)  # 24
    F = cfg.data.node_feature_dims  # 4

    node_features = torch.randn(B, T, N, F)

    # Synthetic graph (same for all timesteps for simplicity)
    # Create a sparse random graph
    np.random.seed(42)
    edges_src, edges_tgt = [], []
    for i in range(N):
        neighbors = np.random.choice([j for j in range(N) if j != i], size=4, replace=False)
        for j in neighbors:
            edges_src.append(i)
            edges_tgt.append(j)
    edge_index = torch.tensor([edges_src, edges_tgt], dtype=torch.long)
    edge_weight = torch.rand(edge_index.size(1))

    edge_indices = [edge_index] * T
    edge_weights = [edge_weight] * T

    # Forward pass
    print(f"\n🔄 Forward pass: input ({B}, {T}, {N}, {F})")
    model.eval()
    with torch.no_grad():
        output = model(node_features, edge_indices, edge_weights)

    print(f"   ✅ direction:   {output['direction'].shape}  (logits)")
    print(f"   ✅ return:      {output['return'].shape}")
    print(f"   ✅ volatility:  {output['volatility'].shape}")

    # Verify shapes
    assert output["direction"].shape == (B, cfg.model.num_direction_classes)
    assert output["return"].shape == (B, 1)
    assert output["volatility"].shape == (B, 1)

    # Verify volatility is non-negative (Softplus)
    assert (output["volatility"] >= 0).all(), "Volatility should be non-negative!"
    print(f"   ✅ Volatility all non-negative")

    # Test gradient flow
    model.train()
    output = model(node_features, edge_indices, edge_weights)
    loss = output["direction"].sum() + output["return"].sum() + output["volatility"].sum()
    loss.backward()
    print(f"   ✅ Backward pass OK — gradients flow")

    # Check no dead gradients
    dead = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().max() == 0:
            dead.append(name)
    if dead:
        print(f"   ⚠️  Dead gradients in: {dead}")
    else:
        print(f"   ✅ No dead gradients")

    print(f"\n✅ All model tests passed!")
    return model


if __name__ == "__main__":
    test_model()
