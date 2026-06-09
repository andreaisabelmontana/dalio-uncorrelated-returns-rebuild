"""Portfolio weighting: inverse-vol within clusters, risk parity across them.

Two-level scheme:
  1. Within each cluster, assets get inverse-volatility weights (cheap, robust).
  2. Each cluster becomes a single synthetic return stream; those streams are
     combined by RISK PARITY so every cluster contributes equally to total
     portfolio risk. This is what operationalizes "uncorrelated streams,
     balanced risk".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .returns import covariance


def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """w_i proportional to 1/sigma_i, normalized to sum to 1."""
    vol = returns.std(ddof=1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w = inv / inv.sum()
    return w.fillna(0.0)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio variance."""
    port_var = float(weights @ cov @ weights)
    if port_var <= 0:
        return np.zeros_like(weights)
    marginal = cov @ weights
    return weights * marginal / port_var


def risk_parity_weights(cov: pd.DataFrame) -> pd.Series:
    """Long-only weights equalizing risk contributions (sum to 1).

    Solved as a constrained least-squares on the risk-contribution vector;
    SLSQP is reliable for the small cluster-level problems we hand it.
    """
    C = cov.values
    n = C.shape[0]
    target = 1.0 / n

    def objective(w):
        rc = risk_contributions(w, C)
        return float(np.sum((rc - target) ** 2))

    w0 = np.full(n, 1.0 / n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(1e-4, 1.0)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return pd.Series(w, index=cov.index)


def build_weights(returns: pd.DataFrame, labels: dict[str, int]) -> dict:
    """Combine intra-cluster inverse-vol with cross-cluster risk parity.

    Returns a dict with final per-asset weights plus the intermediate pieces
    for transparency / plotting.
    """
    clusters = sorted(set(labels.values()))

    # 1) Intra-cluster inverse-vol weights and a synthetic return per cluster.
    intra: dict[str, float] = {}
    cluster_returns = pd.DataFrame(index=returns.index)
    for c in clusters:
        members = [t for t, lab in labels.items() if lab == c]
        w = inverse_vol_weights(returns[members])
        for t in members:
            intra[t] = float(w[t])
        cluster_returns[f"cluster_{c}"] = returns[members] @ w

    # 2) Cross-cluster risk parity on the synthetic cluster streams.
    cov = covariance(cluster_returns)
    cluster_w = risk_parity_weights(cov)

    # 3) Final asset weight = cluster weight * intra-cluster weight.
    final: dict[str, float] = {}
    for c in clusters:
        members = [t for t, lab in labels.items() if lab == c]
        for t in members:
            final[t] = cluster_w[f"cluster_{c}"] * intra[t]

    total = sum(final.values())
    final = {t: w / total for t, w in final.items()}

    return {
        "weights": pd.Series(final).reindex(returns.columns),
        "cluster_weights": cluster_w,
        "intra_weights": pd.Series(intra),
        "cluster_returns": cluster_returns,
    }
