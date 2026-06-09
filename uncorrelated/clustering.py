"""Correlation-based hierarchical clustering of the asset universe.

Assets that move together land in the same cluster; the point of the whole
exercise is then to weight *across* clusters so the portfolio is built from
genuinely independent return streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class ClusterResult:
    labels: dict[str, int]          # ticker -> cluster id
    k: int
    linkage_matrix: np.ndarray
    order: list[str]                # leaf order (for heatmaps/dendrograms)
    silhouette_by_k: dict[int, float]


def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert a correlation matrix to a proper distance: d = sqrt(0.5(1-ρ))."""
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr.values), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return (dist + dist.T) / 2.0  # enforce exact symmetry


def _silhouette(dist: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette score from a precomputed distance matrix."""
    n = len(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= n:
        return -1.0
    scores = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        a = dist[i, same].mean() if same.any() else 0.0
        b = min(
            dist[i, labels == c].mean()
            for c in uniq if c != labels[i]
        )
        scores[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(scores.mean())


def cluster_assets(returns: pd.DataFrame, k_range: range | None = None) -> ClusterResult:
    corr = returns.corr()
    dist = correlation_distance(corr)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    tickers = list(corr.columns)
    k_range = k_range or range(2, min(len(tickers), 8))

    # Pick k by best silhouette on the correlation distance.
    sil: dict[int, float] = {}
    best_k, best_score = 2, -np.inf
    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        s = _silhouette(dist, labels)
        sil[k] = round(s, 4)
        if s > best_score:
            best_k, best_score = k, s

    labels = fcluster(Z, t=best_k, criterion="maxclust")
    from scipy.cluster.hierarchy import leaves_list
    order = [tickers[i] for i in leaves_list(Z)]

    return ClusterResult(
        labels={t: int(l) for t, l in zip(tickers, labels)},
        k=best_k,
        linkage_matrix=Z,
        order=order,
        silhouette_by_k=sil,
    )
