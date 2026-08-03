import pandas as pd, numpy as np

s = pd.read_csv(
    "../../data/closeness_ranking.csv"
)  # two-sided; repeat for one-sided
for p in ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]:
    print(f"{p:9s}  corr(dist): {s[p].corr(s['distance']):+.3f}")
print("distance range:", s["distance"].min(), "..", s["distance"].max())

from sklearn.tree import DecisionTreeRegressor, plot_tree

X, y = s[["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]], s["distance"]
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20).fit(X, y)
print(dict(zip(X.columns, tree.feature_importances_.round(3))))
print(
    "tree R^2 (how much of closeness the params explain):",
    round(tree.score(X, y), 3),
)
