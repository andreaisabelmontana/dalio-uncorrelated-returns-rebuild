"""PortfolioCreator — the end-to-end pipeline.

universe -> prices -> returns -> cluster -> weight (inverse-vol + risk parity)
-> evaluate out-of-sample.

Weights are learned on a training window and then evaluated on a held-out test
window, so the reported performance is genuinely out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .clustering import ClusterResult, cluster_assets
from .data import load_prices
from .metrics import Performance, diversification_ratio, evaluate, portfolio_returns
from .returns import to_returns
from .weights import build_weights


@dataclass
class PortfolioResult:
    weights: pd.Series
    clusters: ClusterResult
    train_perf: Performance
    test_perf: Performance
    equal_weight_test_perf: Performance
    diversification_ratio: float
    test_returns: pd.Series = field(repr=False)
    cluster_weights: pd.Series = field(repr=False)


class PortfolioCreator:
    def __init__(self, source: str = "synthetic", test_fraction: float = 0.3, **load_kwargs):
        self.source = source
        self.test_fraction = test_fraction
        self.load_kwargs = load_kwargs

    def run(self) -> PortfolioResult:
        prices = load_prices(self.source, **self.load_kwargs)
        returns = to_returns(prices)

        split = int(len(returns) * (1 - self.test_fraction))
        train, test = returns.iloc[:split], returns.iloc[split:]

        # Learn structure + weights on the training window only.
        clusters = cluster_assets(train)
        built = build_weights(train, clusters.labels)
        weights = built["weights"]

        train_ret = portfolio_returns(train, weights)
        test_ret = portfolio_returns(test, weights)

        # Naive equal-weight benchmark on the same test window.
        eq = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        eq_test_ret = portfolio_returns(test, eq)

        return PortfolioResult(
            weights=weights.sort_values(ascending=False),
            clusters=clusters,
            train_perf=evaluate(train_ret),
            test_perf=evaluate(test_ret),
            equal_weight_test_perf=evaluate(eq_test_ret),
            diversification_ratio=diversification_ratio(test, weights),
            test_returns=test_ret,
            cluster_weights=built["cluster_weights"],
        )
