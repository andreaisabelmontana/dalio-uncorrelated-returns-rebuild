"""Return computation and annualization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns from a price panel."""
    return prices.pct_change().dropna(how="all").fillna(0.0)


def annualized_return(returns: pd.Series | pd.DataFrame):
    """Geometric annualized return."""
    growth = (1.0 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return growth ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series | pd.DataFrame):
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized covariance matrix."""
    return returns.cov() * TRADING_DAYS
