"""Tests for the uncorrelated-portfolio pipeline (deterministic synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from uncorrelated import (
    PortfolioCreator,
    cluster_assets,
    inverse_vol_weights,
    risk_contributions,
    risk_parity_weights,
)
from uncorrelated.data import UNIVERSE, synthetic_prices
from uncorrelated.returns import covariance, to_returns


@pytest.fixture(scope="module")
def returns():
    return to_returns(synthetic_prices(days=1200, seed=7))


def test_synthetic_universe_shape(returns):
    assert returns.shape[1] == len(UNIVERSE)
    assert len(returns) > 1000


def test_inverse_vol_weights_sum_to_one_and_favor_low_vol(returns):
    w = inverse_vol_weights(returns)
    assert abs(w.sum() - 1.0) < 1e-9
    # The lowest-vol asset should get more weight than the highest-vol one.
    vols = returns.std()
    assert w[vols.idxmin()] > w[vols.idxmax()]


def test_risk_parity_equalizes_risk_contributions(returns):
    cov = covariance(returns)
    w = risk_parity_weights(cov)
    assert abs(w.sum() - 1.0) < 1e-6
    rc = risk_contributions(w.values, cov.values)
    # All contributions should be close to 1/n.
    assert np.std(rc) < 0.02


def test_clustering_separates_crypto_from_bonds(returns):
    result = cluster_assets(returns)
    assert 2 <= result.k <= 7
    # BTC and ETH are highly correlated -> same cluster.
    assert result.labels["BTC"] == result.labels["ETH"]
    # Crypto and long govt bonds are nearly uncorrelated -> different clusters.
    assert result.labels["BTC"] != result.labels["LTGOV"]


def test_pipeline_runs_and_diversifies(returns):
    result = PortfolioCreator(source="synthetic", test_fraction=0.3).run()
    w = result.weights
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all()  # long-only
    # Diversification ratio > 1 means the mix lowers vol vs weighted-average vol.
    assert result.diversification_ratio > 1.0


def test_portfolio_lowers_vol_vs_equal_weight():
    result = PortfolioCreator(source="synthetic", test_fraction=0.3).run()
    # A risk-balanced portfolio should not have wildly higher vol than equal weight;
    # typically it is lower. Assert it is at most the equal-weight vol (with margin).
    assert result.test_perf.ann_vol <= result.equal_weight_test_perf.ann_vol + 0.02
