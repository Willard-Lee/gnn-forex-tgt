"""
Custom layers for Temporal Graph Transformer v2.

Contains:
  - GATLayer: single-head graph attention (no PyTorch Geometric)
  - MultiHeadGAT: multi-head wrapper with concat or average aggregation
  - GATBlock: two-layer GAT stack (L1 concat, L2 average)
  - PositionalEncoding: sinusoidal positional encoding for Transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GATLayer(nn.Module):
    """
    Single-head Graph Attention layer (Velickovic et al., 2018).

    Manual implementation — no PyTorch Geometric dependency.
    Handles sparse graphs via edge_index.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, edge_dropout: float = 0.15):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.zeros(out_dim, 1))
        self.a_tgt = nn.Parameter(torch.zeros(out_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_tgt)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.edge_dropout = nn.Dropout(edge_dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) source-target pairs
            edge_weight: (E,) optional edge weights to scale attention

        Returns:
            (N, out_dim) updated node features
        """
        N = x.size(0)
        h = self.W(x)  # (N, out_dim)

        # Attention scores for all edges
        src, tgt = edge_index[0], edge_index[1]  # (E,)
        e_src = (h[src] @ self.a_src).squeeze(-1)  # (E,)
        e_tgt = (h[tgt] @ self.a_tgt).squeeze(-1)  # (E,)
        e = self.leaky_relu(e_src + e_tgt)  # (E,)

        # Scale by edge weight if provided
        if edge_weight is not None:
            e = e * edge_weight

        # Sparse softmax per target node
        e_max = torch.full((N,), float("-inf"), device=x.device)
        e_max = e_max.scatter_reduce(0, tgt, e, reduce="amax", include_self=True)
        e_exp = torch.exp(e - e_max[tgt])

        # Edge dropout (during training)
        e_exp = self.edge_dropout(e_exp)

        e_sum = torch.zeros(N, device=x.device)
        e_sum.scatter_add_(0, tgt, e_exp)
        alpha = e_exp / (e_sum[tgt] + 1e-10)  # (E,)

        # Aggregate
        out = torch.zeros(N, h.size(1), device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(1).expand(-1, h.size(1)), alpha.unsqueeze(1) * h[src])

        return self.dropout(out)


class MultiHeadGAT(nn.Module):
    """
    Multi-head GAT with concat or average aggregation.

    concat=True:  output dim = n_heads * out_dim
    concat=False: output dim = out_dim (average across heads)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int,
        concat: bool = True,
        dropout: float = 0.1,
        edge_dropout: float = 0.15,
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            GATLayer(in_dim, out_dim, dropout, edge_dropout)
            for _ in range(n_heads)
        ])
        self.concat = concat
        self.out_dim = out_dim * n_heads if concat else out_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        head_outs = [head(x, edge_index, edge_weight) for head in self.heads]

        if self.concat:
            return torch.cat(head_outs, dim=-1)  # (N, n_heads * out_dim)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)  # (N, out_dim)


class GATBlock(nn.Module):
    """
    Two-layer GAT stack as specified in the architecture:
      L1: in_dim → heads_l1 × hidden (concat) → ELU
      L2: (heads_l1 × hidden) → out_dim (average over heads_l2)

    Produces per-node embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        heads_l1: int = 8,
        heads_l2: int = 4,
        dropout: float = 0.1,
        edge_dropout: float = 0.15,
    ):
        super().__init__()
        self.gat_l1 = MultiHeadGAT(
            in_dim, hidden, heads_l1, concat=True,
            dropout=dropout, edge_dropout=edge_dropout,
        )
        self.norm1 = nn.LayerNorm(hidden * heads_l1)
        self.gat_l2 = MultiHeadGAT(
            hidden * heads_l1, out_dim, heads_l2, concat=False,
            dropout=dropout, edge_dropout=edge_dropout,
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E)
            edge_weight: (E,)

        Returns:
            (N, out_dim) node embeddings
        """
        h = self.gat_l1(x, edge_index, edge_weight)
        h = self.norm1(h)
        h = F.elu(h)
        h = self.gat_l2(h, edge_index, edge_weight)
        h = self.norm2(h)
        return h


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for the Transformer."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
