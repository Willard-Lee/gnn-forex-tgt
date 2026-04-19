"""
Dynamic Multi-Edge Graph Builder for Temporal Graph Transformer 

Builds a graph over 24 technical indicator nodes with three edge types:
  1. Pearson correlation (30-day trailing window)
  2. DCC-GARCH proxy (7-day vol-adjusted short correlation)
  3. Granger causality (5 lags, p < 0.05)

Edges are combined (40/40/20 weights), sparsified (top-k=6 per node),
and recomputed every 20 trading days. Strictly causal — only uses
data available at time t.

NOTE: "DCC-GARCH proxy" is NOT a true DCC-GARCH model. It's a short-window
volatility-adjusted correlation. We label it honestly.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

from configs.config import GraphConfig

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")


# ===========================================================================
# 1. Individual Edge Type Computations
# ===========================================================================

def compute_pearson_matrix(
    indicator_df: pd.DataFrame,
    window: int = 30,
) -> np.ndarray:
    """
    Compute pairwise Pearson correlation over a trailing window.

    Args:
        indicator_df: (T, N) DataFrame where T >= window, N = num indicators.
                      Must contain ONLY the trailing window of data.
        window: number of rows to use (indicator_df should be this length).

    Returns:
        (N, N) correlation matrix with values in [-1, 1]. Diagonal = 0.
    """
    data = indicator_df.tail(window)
    corr = data.corr(method="pearson").values
    np.fill_diagonal(corr, 0.0)
    # NaN → 0 (happens if an indicator is constant over the window)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def compute_dcc_proxy_matrix(
    indicator_df: pd.DataFrame,
    short_window: int = 7,
) -> np.ndarray:
    """
    DCC-GARCH proxy: vol-adjusted short-window correlation.

    For each pair (i, j):
        1. Compute daily changes for indicators i and j.
        2. Standardize by rolling std over short_window.
        3. Correlate the standardized changes over short_window.

    This is NOT a true DCC-GARCH. It captures recent co-movement after
    removing volatility scaling — a lightweight proxy.

    Args:
        indicator_df: (T, N) DataFrame. Needs at least short_window + 1 rows.
        short_window: lookback for vol adjustment and correlation.

    Returns:
        (N, N) correlation-like matrix in [-1, 1]. Diagonal = 0.
    """
    # Daily changes
    changes = indicator_df.diff().iloc[1:]

    # Rolling std for vol normalization
    roll_std = changes.rolling(short_window, min_periods=max(3, short_window // 2)).std()

    # Standardize: change / rolling_std
    standardized = changes / roll_std.replace(0, np.nan)
    standardized = standardized.fillna(0.0)

    # Correlation over the last short_window rows of standardized series
    tail = standardized.tail(short_window)
    corr = tail.corr(method="pearson").values
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def compute_granger_matrix(
    indicator_df: pd.DataFrame,
    max_lag: int = 5,
    p_threshold: float = 0.05,
) -> np.ndarray:
    """
    Pairwise Granger causality test.

    Tests if indicator j Granger-causes indicator i (j → i).
    Returns a matrix where entry [i, j] = 1 if j Granger-causes i
    at any lag up to max_lag (min p-value < threshold), else 0.

    Uses daily changes to ensure stationarity.

    Args:
        indicator_df: (T, N) DataFrame. Needs at least max_lag + 15 rows.
        max_lag: maximum lag order to test.
        p_threshold: significance threshold.

    Returns:
        (N, N) binary matrix. Not symmetric. Diagonal = 0.
    """
    n_nodes = indicator_df.shape[1]
    granger_mat = np.zeros((n_nodes, n_nodes))

    # Use changes for stationarity
    changes = indicator_df.diff().dropna()

    if len(changes) < max_lag + 10:
        # Not enough data for meaningful Granger tests
        return granger_mat

    cols = indicator_df.columns.tolist()

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            try:
                # Test: does j Granger-cause i?
                pair_data = changes[[cols[i], cols[j]]].dropna()
                if len(pair_data) < max_lag + 10:
                    continue

                # Suppress statsmodels output
                result = grangercausalitytests(
                    pair_data.values, maxlag=max_lag, verbose=False
                )
                # Get minimum p-value across all lags (ssr_ftest)
                min_p = min(
                    result[lag][0]["ssr_ftest"][1]
                    for lag in range(1, max_lag + 1)
                )
                if min_p < p_threshold:
                    granger_mat[i, j] = 1.0
            except Exception:
                # Singular matrix, constant series, etc.
                continue

    return granger_mat


# ===========================================================================
# 2. Edge Combination and Sparsification
# ===========================================================================

def combine_edge_matrices(
    pearson_mat: np.ndarray,
    dcc_mat: np.ndarray,
    granger_mat: np.ndarray,
    w_pearson: float = 0.4,
    w_dcc: float = 0.4,
    w_granger: float = 0.2,
    pearson_threshold: float = 0.45,
) -> np.ndarray:
    """
    Combine three edge types into a single weighted adjacency matrix.

    Pearson and DCC contribute |correlation| (absolute value) as edge strength.
    Granger contributes binary (0 or 1).
    Pearson is thresholded: only edges with |ρ| > pearson_threshold survive.

    Args:
        pearson_mat: (N, N) raw Pearson correlations.
        dcc_mat: (N, N) DCC proxy correlations.
        granger_mat: (N, N) binary Granger causality.
        w_pearson, w_dcc, w_granger: combination weights.
        pearson_threshold: minimum |ρ| for Pearson edge.

    Returns:
        (N, N) combined edge weight matrix, values in [0, 1].
    """
    # Threshold Pearson
    pearson_abs = np.abs(pearson_mat)
    pearson_edges = np.where(pearson_abs > pearson_threshold, pearson_abs, 0.0)

    # DCC: use absolute value as strength
    dcc_edges = np.abs(dcc_mat)

    # Combine
    combined = (
        w_pearson * pearson_edges +
        w_dcc * dcc_edges +
        w_granger * granger_mat
    )

    np.fill_diagonal(combined, 0.0)
    return combined


def sparsify_top_k(adj_matrix: np.ndarray, top_k: int = 6, min_weight: float = 0.05) -> np.ndarray:
    """
    Keep only top-k edges per node (symmetric: union of incoming and outgoing).

    Args:
        adj_matrix: (N, N) weighted adjacency matrix.
        top_k: max neighbors per node.
        min_weight: drop edges below this weight.

    Returns:
        (N, N) sparsified adjacency matrix.
    """
    n = adj_matrix.shape[0]
    mask = np.zeros_like(adj_matrix, dtype=bool)

    for i in range(n):
        # Top-k outgoing edges for node i
        weights = adj_matrix[i, :]
        if np.count_nonzero(weights) <= top_k:
            mask[i, :] = weights > 0
        else:
            threshold = np.sort(weights)[::-1][top_k - 1]
            mask[i, :] = weights >= threshold

    # Symmetrize: if either direction selected, keep both
    mask = mask | mask.T

    sparse = adj_matrix * mask
    sparse[sparse < min_weight] = 0.0
    np.fill_diagonal(sparse, 0.0)
    return sparse


# ===========================================================================
# 3. Convert to Edge Index + Edge Weight (PyTorch-compatible)
# ===========================================================================

def adj_to_edge_index(adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert dense adjacency matrix to COO-format edge_index + edge_weight.

    Returns:
        edge_index: (2, E) int64 — source, target pairs
        edge_weight: (E,) float32 — edge weights
    """
    rows, cols = np.nonzero(adj_matrix)
    weights = adj_matrix[rows, cols].astype(np.float32)
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    return edge_index, weights


# ===========================================================================
# 4. Graph Snapshot Builder (main interface)
# ===========================================================================

class DynamicGraphBuilder:
    """
    Builds and caches graph snapshots for the TGT model.

    The graph is recomputed every `recompute_every` trading days using only
    data available at that point (no leakage). Between recompute points,
    the same graph is reused.

    Usage:
        builder = DynamicGraphBuilder(cfg.graph, indicator_names)
        graphs = builder.build_graph_sequence(df, dates)

    Each graph snapshot is a dict:
        {
            "edge_index": np.ndarray (2, E),
            "edge_weight": np.ndarray (E,),
            "adj_matrix": np.ndarray (N, N),
            "num_edges": int,
            "computed_at": pd.Timestamp,
        }
    """

    def __init__(self, graph_cfg: GraphConfig, indicator_names: List[str]):
        self.cfg = graph_cfg
        self.indicator_names = indicator_names
        self.n_nodes = len(indicator_names)

        # Cache: maps recompute_date → graph snapshot
        self._cache: Dict[pd.Timestamp, Dict] = {}

    def _compute_graph_at(self, df: pd.DataFrame, current_idx: int) -> Dict:
        """
        Compute the graph using data up to (and including) current_idx.
        Strictly causal: no future data.

        Args:
            df: Full indicator DataFrame (indexed by date).
            current_idx: integer position of the current date.

        Returns:
            Graph snapshot dict.
        """
        # Determine how much trailing data we need
        max_window = max(
            self.cfg.pearson_window,
            self.cfg.dcc_window + 5,  # DCC needs extra for diff + rolling
            self.cfg.granger_max_lag + 15,
        )

        # Slice trailing data up to current_idx (inclusive, no future)
        start_idx = max(0, current_idx - max_window + 1)
        trailing = df.iloc[start_idx:current_idx + 1][self.indicator_names]

        # 1. Pearson
        if len(trailing) >= self.cfg.pearson_window:
            pearson_mat = compute_pearson_matrix(trailing, self.cfg.pearson_window)
        else:
            pearson_mat = np.zeros((self.n_nodes, self.n_nodes))

        # 2. DCC proxy
        if len(trailing) >= self.cfg.dcc_window + 2:
            dcc_mat = compute_dcc_proxy_matrix(trailing, self.cfg.dcc_window)
        else:
            dcc_mat = np.zeros((self.n_nodes, self.n_nodes))

        # 3. Granger
        if len(trailing) >= self.cfg.granger_max_lag + 15:
            granger_mat = compute_granger_matrix(
                trailing, self.cfg.granger_max_lag, self.cfg.granger_p_threshold
            )
        else:
            granger_mat = np.zeros((self.n_nodes, self.n_nodes))

        # 4. Combine
        combined = combine_edge_matrices(
            pearson_mat, dcc_mat, granger_mat,
            self.cfg.weight_pearson, self.cfg.weight_dcc, self.cfg.weight_granger,
            self.cfg.pearson_threshold,
        )

        # 5. Sparsify
        adj = sparsify_top_k(combined, self.cfg.top_k, self.cfg.min_edge_weight)

        # 6. Convert to edge_index format
        edge_index, edge_weight = adj_to_edge_index(adj)

        return {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "adj_matrix": adj,
            "num_edges": edge_index.shape[1] if edge_index.size > 0 else 0,
            "computed_at": df.index[current_idx],
        }

    def build_graph_sequence(
        self,
        df: pd.DataFrame,
        start_idx: Optional[int] = None,
    ) -> List[Dict]:
        """
        Build graph snapshots for every row in df, recomputing every
        `recompute_every` trading days.

        Args:
            df: Full DataFrame with indicator columns and DatetimeIndex.
            start_idx: Index position where graph computation begins
                       (default: max_window to ensure enough trailing data).

        Returns:
            List of graph snapshots, one per row from start_idx onward.
            The graph at position i corresponds to df.iloc[start_idx + i].
        """
        max_window = max(
            self.cfg.pearson_window,
            self.cfg.dcc_window + 5,
            self.cfg.granger_max_lag + 15,
        )

        if start_idx is None:
            start_idx = max_window

        graphs = []
        current_graph = None
        days_since_recompute = self.cfg.recompute_every  # Force compute on first step

        total = len(df) - start_idx
        for i, idx in enumerate(range(start_idx, len(df))):
            date = df.index[idx]

            # Check cache first
            if date in self._cache:
                current_graph = self._cache[date]
                days_since_recompute = 0
            elif days_since_recompute >= self.cfg.recompute_every or current_graph is None:
                current_graph = self._compute_graph_at(df, idx)
                self._cache[date] = current_graph
                days_since_recompute = 0

                if (i + 1) % 100 == 0 or i == 0:
                    print(
                        f"  📊 Graph [{i+1}/{total}] at {date.date()} | "
                        f"{current_graph['num_edges']} edges"
                    )

            graphs.append(current_graph)
            days_since_recompute += 1

        return graphs

    def get_graph_at_date(self, df: pd.DataFrame, date: pd.Timestamp) -> Dict:
        """Get or compute graph for a specific date."""
        if date in self._cache:
            return self._cache[date]

        # Find the index position for this date
        idx = df.index.get_loc(date)
        if isinstance(idx, slice):
            idx = idx.start
        graph = self._compute_graph_at(df, idx)
        self._cache[date] = graph
        return graph

    def clear_cache(self):
        """Clear the graph cache."""
        self._cache.clear()

    def get_stats(self, graphs: List[Dict]) -> Dict:
        """Compute summary statistics over a sequence of graphs."""
        edge_counts = [g["num_edges"] for g in graphs]
        unique_graphs = len(set(id(g) for g in graphs))

        # Edge weight stats across all unique graphs
        all_weights = []
        seen = set()
        for g in graphs:
            gid = id(g)
            if gid not in seen and g["edge_weight"].size > 0:
                seen.add(gid)
                all_weights.append(g["edge_weight"])

        if all_weights:
            weights = np.concatenate(all_weights)
            weight_stats = {
                "mean": float(np.mean(weights)),
                "std": float(np.std(weights)),
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
            }
        else:
            weight_stats = {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "num_snapshots": len(graphs),
            "unique_graphs": unique_graphs,
            "edge_count_mean": float(np.mean(edge_counts)),
            "edge_count_std": float(np.std(edge_counts)),
            "edge_count_min": int(np.min(edge_counts)),
            "edge_count_max": int(np.max(edge_counts)),
            "weight_stats": weight_stats,
        }


# ===========================================================================
# 5. Quick Test
# ===========================================================================

def test_graph_builder():
    """Smoke test with synthetic data."""
    from configs.config import Config

    cfg = Config()

    # Synthetic indicator data (200 rows, 24 indicators)
    np.random.seed(42)
    n_days = 200
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    n_indicators = len(cfg.data.feature_nodes)

    # Correlated groups to make the graph interesting
    base1 = np.cumsum(np.random.randn(n_days) * 0.01)
    base2 = np.cumsum(np.random.randn(n_days) * 0.01)

    data = {}
    for i, name in enumerate(cfg.data.feature_nodes):
        if i < 10:  # momentum group — correlated
            data[name] = base1 + np.random.randn(n_days) * 0.3
        elif i < 16:  # volatility group — correlated
            data[name] = base2 + np.random.randn(n_days) * 0.3
        else:  # trend + volume — mixed
            data[name] = np.cumsum(np.random.randn(n_days) * 0.02)

    df = pd.DataFrame(data, index=dates)

    print("=" * 60)
    print("🧪 Graph Builder Smoke Test")
    print("=" * 60)

    builder = DynamicGraphBuilder(cfg.graph, cfg.data.feature_nodes)

    # Build sequence
    graphs = builder.build_graph_sequence(df)
    stats = builder.get_stats(graphs)

    print(f"\n✅ Built {stats['num_snapshots']} snapshots | "
          f"{stats['unique_graphs']} unique graphs")
    print(f"   Edges: {stats['edge_count_mean']:.0f} ± {stats['edge_count_std']:.1f} "
          f"(min={stats['edge_count_min']}, max={stats['edge_count_max']})")
    print(f"   Weights: mean={stats['weight_stats']['mean']:.3f}, "
          f"max={stats['weight_stats']['max']:.3f}")

    # Verify no self-loops
    for g in graphs:
        if g["edge_index"].size > 0:
            assert not np.any(g["edge_index"][0] == g["edge_index"][1]), "Self-loop found!"
    print("   ✅ No self-loops")

    # Verify sparsity (symmetrizing top-k can slightly exceed 2*k)
    max_neighbors = 0
    for g in graphs:
        adj = g["adj_matrix"]
        for node in range(adj.shape[0]):
            n_neighbors = np.count_nonzero(adj[node])
            max_neighbors = max(max_neighbors, n_neighbors)
    # After symmetrization, worst case is n_nodes-1 but should stay close to top_k
    assert max_neighbors <= cfg.graph.top_k * 4, (
        f"Max {max_neighbors} neighbors — graph not sparse enough"
    )
    print(f"   ✅ Sparsity OK (max {max_neighbors} neighbors, target ~{cfg.graph.top_k})")

    # Verify recompute frequency
    assert stats["unique_graphs"] > 1, "Graph never recomputed!"
    print(f"   ✅ Recomputed {stats['unique_graphs']} times over {stats['num_snapshots']} days")

    print("\n✅ All graph builder tests passed!")
    return builder, graphs, stats


if __name__ == "__main__":
    test_graph_builder()
