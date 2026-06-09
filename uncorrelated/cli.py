"""CLI: run the full pipeline and print weights + out-of-sample performance."""

from __future__ import annotations

import argparse
import json
import sys

from .data import UNIVERSE, load_prices
from .portfolio import PortfolioCreator
from .returns import to_returns


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uncorrelated",
                                 description="Build an uncorrelated-return portfolio.")
    ap.add_argument("--source", choices=["synthetic", "live"], default="synthetic")
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--plots", metavar="DIR", help="Write heatmap/weights/equity PNGs here.")
    args = ap.parse_args(argv)

    creator = PortfolioCreator(source=args.source, test_fraction=args.test_fraction)
    result = creator.run()

    names = {a.ticker: a.name for a in UNIVERSE}
    print(f"\nClusters found: k={result.clusters.k}  "
          f"(silhouette by k: {result.clusters.silhouette_by_k})")
    print("\nRecommended weights:")
    for ticker, w in result.weights.items():
        c = result.clusters.labels.get(ticker, "?")
        print(f"  [{c}] {ticker:6s} {names.get(ticker, ''):20s} {w * 100:6.2f}%")

    print("\nOut-of-sample performance:")
    print("  portfolio  :", json.dumps(result.test_perf.as_dict()))
    print("  equal-weight:", json.dumps(result.equal_weight_test_perf.as_dict()))
    print(f"  diversification ratio: {result.diversification_ratio:.2f}")

    if args.plots:
        from . import plots
        prices = load_prices(args.source)
        rets = to_returns(prices)
        import pandas as pd
        eq = pd.Series(1.0 / rets.shape[1], index=rets.columns)
        bench = (rets.iloc[len(rets) - len(result.test_returns):] @ eq)
        plots.correlation_heatmap(rets, result.clusters.order, f"{args.plots}/correlation.png")
        plots.weight_bars(result.weights, result.clusters.labels, f"{args.plots}/weights.png")
        plots.equity_curve(result.test_returns, bench, f"{args.plots}/equity.png")
        print(f"\nPlots written to {args.plots}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
