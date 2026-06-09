"""Visualization: correlation heatmap, weight bars, equity curve."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _ensure(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def correlation_heatmap(returns: pd.DataFrame, order: list[str], out: str | Path) -> Path:
    corr = returns.corr().loc[order, order]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_title("Asset correlation (clustered order)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = _ensure(out)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def weight_bars(weights: pd.Series, labels: dict[str, int], out: str | Path) -> Path:
    w = weights.sort_values(ascending=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap((labels.get(t, 0)) % 10) for t in w.index]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(w.index, w.values * 100, color=colors)
    ax.set_xlabel("weight (%)")
    ax.set_title("Recommended portfolio weights (colored by cluster)")
    for i, v in enumerate(w.values):
        ax.text(v * 100, i, f" {v * 100:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    out = _ensure(out)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def equity_curve(test_returns: pd.Series, benchmark: pd.Series | None, out: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot((1 + test_returns).cumprod(), label="Uncorrelated portfolio", lw=2)
    if benchmark is not None:
        ax.plot((1 + benchmark).cumprod(), label="Equal weight", lw=1.5, ls="--", alpha=0.8)
    ax.set_title("Out-of-sample growth of $1")
    ax.set_ylabel("growth")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = _ensure(out)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
