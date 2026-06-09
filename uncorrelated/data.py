"""Price data loading.

Offline by default: a deterministic synthetic generator builds a realistic
multi-asset universe whose returns are driven by a few latent factors, so
genuine correlation *clusters* emerge (equities move together, bonds together,
crypto on its own, etc.). This lets the whole pipeline — and the test suite —
run with no network and reproducible numbers.

Pass ``source="live"`` (and install the ``live`` extra) to pull real prices
from Yahoo Finance instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    category: str  # the *true* group, used only to validate clustering


# A liquid, multi-asset universe spanning the buckets Dalio-style
# diversification cares about. The `factor loadings` below decide co-movement.
UNIVERSE: list[Asset] = [
    Asset("USEQ", "US Large Cap", "equity"),
    Asset("EUEQ", "Europe Equity", "equity"),
    Asset("EMEQ", "EM Equity", "equity"),
    Asset("SMCAP", "US Small Cap", "equity"),
    Asset("LTGOV", "Long Govt Bonds", "rates"),
    Asset("ITGOV", "Interm Govt Bonds", "rates"),
    Asset("IGCRD", "IG Credit", "rates"),
    Asset("GOLD", "Gold", "commodity"),
    Asset("BCOM", "Broad Commodities", "commodity"),
    Asset("TIPS", "Inflation Linked", "inflation"),
    Asset("BTC", "Bitcoin", "crypto"),
    Asset("ETH", "Ethereum", "crypto"),
]

# Latent factors: market, rates, commodity, inflation, crypto.
_FACTORS = ["mkt", "rates", "cmdty", "infl", "crypto"]

# Each asset's loading on each factor (rows align with UNIVERSE).
_LOADINGS = np.array([
    # mkt   rates  cmdty  infl   crypto
    [0.95, -0.10, 0.05, 0.00, 0.05],   # USEQ
    [0.90, -0.10, 0.05, 0.00, 0.05],   # EUEQ
    [0.85, -0.05, 0.15, 0.05, 0.10],   # EMEQ
    [0.95, -0.10, 0.05, 0.00, 0.10],   # SMCAP
    [-0.15, 0.95, 0.00, 0.10, 0.00],   # LTGOV
    [-0.10, 0.90, 0.00, 0.10, 0.00],   # ITGOV
    [0.25, 0.70, 0.05, 0.05, 0.00],    # IGCRD
    [0.00, 0.05, 0.85, 0.35, 0.05],    # GOLD
    [0.10, -0.05, 0.95, 0.30, 0.05],   # BCOM
    [-0.05, 0.45, 0.20, 0.80, 0.00],   # TIPS
    [0.20, 0.00, 0.05, 0.05, 0.95],    # BTC
    [0.20, 0.00, 0.05, 0.05, 0.93],    # ETH
])

# Annualized vol and drift per asset (rough, plausible).
_VOL = np.array([0.16, 0.18, 0.22, 0.20, 0.12, 0.06, 0.08, 0.15, 0.17, 0.07, 0.65, 0.75])
_DRIFT = np.array([0.08, 0.07, 0.08, 0.09, 0.03, 0.02, 0.04, 0.05, 0.04, 0.03, 0.30, 0.28])


def synthetic_prices(days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Generate a deterministic daily price panel for the universe."""
    rng = np.random.default_rng(seed)
    n_assets = len(UNIVERSE)
    dt = 1.0 / 252.0

    # Daily factor shocks + small idiosyncratic noise per asset.
    factor_shocks = rng.standard_normal((days, len(_FACTORS)))
    idio = rng.standard_normal((days, n_assets))

    # Scale loadings so each asset hits its target vol.
    raw = factor_shocks @ _LOADINGS.T + 0.35 * idio
    raw_std = raw.std(axis=0, ddof=1)
    scaled = raw / raw_std * (_VOL * np.sqrt(dt))

    daily_returns = _DRIFT * dt + scaled
    prices = 100.0 * np.cumprod(1.0 + daily_returns, axis=0)

    idx = pd.bdate_range(end="2026-01-31", periods=days)
    return pd.DataFrame(prices, index=idx, columns=[a.ticker for a in UNIVERSE])


def live_prices(tickers: dict[str, str], period: str = "5y") -> pd.DataFrame:
    """Fetch real adjusted-close prices via yfinance (optional extra)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover - optional dep
        raise RuntimeError("Live data needs: pip install 'uncorrelated[live]'") from e
    data = yf.download(list(tickers.values()), period=period, auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data else data
    return close.dropna(how="all").ffill().dropna()


def load_prices(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    if source == "synthetic":
        return synthetic_prices(**kwargs)
    if source == "live":
        mapping = {a.ticker: a.ticker for a in UNIVERSE}
        return live_prices(mapping, **kwargs)
    raise ValueError(f"Unknown data source {source!r}")
