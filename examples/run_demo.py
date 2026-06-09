"""Run the full pipeline and print a summary.

    python examples/run_demo.py
"""

from uncorrelated import PortfolioCreator
from uncorrelated.data import UNIVERSE

names = {a.ticker: a.name for a in UNIVERSE}
result = PortfolioCreator(source="synthetic", test_fraction=0.3).run()

print(f"Clusters: k={result.clusters.k}\n")
print("Weights:")
for t, w in result.weights.items():
    print(f"  {t:6s} {names[t]:20s} {w*100:6.2f}%  (cluster {result.clusters.labels[t]})")

print("\nOut-of-sample:")
print("  portfolio   :", result.test_perf.as_dict())
print("  equal-weight:", result.equal_weight_test_perf.as_dict())
print(f"  diversification ratio: {result.diversification_ratio:.2f}")
