"""Portfolio evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .returns import TRADING_DAYS, annualized_return, annualized_vol


@dataclass
class Performance:
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio return series for a fixed weight vector."""
    w = weights.reindex(returns.columns).fillna(0.0)
    return returns @ w


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline of the cumulative growth curve."""
    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    return float((curve / peak - 1.0).min())


def evaluate(returns: pd.Series, rf: float = 0.0) -> Performance:
    ann_ret = float(annualized_return(returns))
    ann_vol = float(annualized_vol(returns))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(returns)
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    return Performance(ann_ret, ann_vol, sharpe, mdd, calmar)


def diversification_ratio(returns: pd.DataFrame, weights: pd.Series) -> float:
    """Weighted-avg vol / portfolio vol. >1 means diversification is working."""
    w = weights.reindex(returns.columns).fillna(0.0).values
    vols = (returns.std(ddof=1) * np.sqrt(TRADING_DAYS)).values
    cov = returns.cov().values * TRADING_DAYS
    port_vol = np.sqrt(w @ cov @ w)
    return float((w @ vols) / port_vol) if port_vol > 0 else 0.0
