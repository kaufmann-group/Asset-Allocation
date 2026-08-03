import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import ks_2samp

PARAMS = ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]
METRICS = ["risk", "return", "sharpe"]
ALPHA = 0.75  # weight on noisiness vs. mean closeness
TOP_FRACTION = 0.10  # top decile

df = pd.read_parquet("data/sweep/all_runs_long.parquet")

# ---- 1. per-row differences and health flags --------------------------------
for m in METRICS:
    df[f"d_{m}"] = df[f"caa_{m}"] - df[f"aa_{m}"]
df["degenerate"] = (
    df[[f"caa_{m}" for m in METRICS] + [f"aa_{m}" for m in METRICS]]
    .isna()
    .any(axis=1)
)

# ---- 2. global scales (pooled classical IQR, one per metric) ----------------
# Robust to outliers; do NOT scale per combo or noisy combos get flattered.
scales = {
    m: np.subtract(*np.nanpercentile(df[f"aa_{m}"], [75, 25])) for m in METRICS
}
print("global scales:", scales)


# ---- 3. per-combo aggregation and score -------------------------------------
def combo_stats(g):
    out = {"n_degenerate": int(g["degenerate"].sum())}
    gg = g[~g["degenerate"]]
    score = 0.0
    for m in METRICS:
        mad = gg[f"d_{m}"].abs().mean()
        sd = gg[f"d_{m}"].std()
        out[f"mad_{m}"], out[f"sd_{m}"] = mad, sd
        score += (mad + ALPHA * sd) / scales[m]
    out["score"] = (
        score if len(gg) >= 30 else np.inf
    )  # too many bad runs -> disqualify
    for p in PARAMS:
        out[p] = g[p].iloc[0]
    return pd.Series(out)


combos = df.groupby("combo_id").apply(combo_stats).reset_index()
combos = combos.sort_values("score")
combos.to_csv("data/sweep/combo_scores.csv", index=False)
print(combos.head(15)[["combo_id", "score", "n_degenerate"] + PARAMS])

# ---- 4. top-decile distribution shift ---------------------------------------
valid = combos[np.isfinite(combos["score"])]
top = valid.head(max(10, int(TOP_FRACTION * len(valid))))

fig, axes = plt.subplots(1, 5, figsize=(20, 3.5))
for ax, p in zip(axes, PARAMS):
    ax.hist(valid[p], bins=25, density=True, alpha=0.4, label="all")
    ax.hist(top[p], bins=25, density=True, alpha=0.6, label="top decile")
    ks = ks_2samp(valid[p], top[p])
    ax.set_title(f"{p}  (KS p={ks.pvalue:.3f})")
    ax.legend()
fig.suptitle("Parameters where the top decile concentrates define your region")
plt.tight_layout()
plt.savefig("data/sweep/top_decile_shift.png", dpi=200)

# ---- 5. surrogate: which parameters matter, and where -----------------------
X = valid[PARAMS].copy()
# log-space for parameters spanning orders of magnitude
for p in ["lambda_1", "lambda_2", "lambda_3"]:
    X[p] = np.log10(X[p])
y = np.log10(valid["score"])  # log target: scores are ratio-like

gbm = GradientBoostingRegressor(
    n_estimators=600,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    random_state=0,
).fit(X, y)
imp = pd.Series(gbm.feature_importances_, index=PARAMS).sort_values(
    ascending=False
)
print("\nfeature importances:\n", imp)

fig, ax = plt.subplots(figsize=(16, 3.5))
PartialDependenceDisplay.from_estimator(gbm, X, PARAMS, ax=ax)
plt.tight_layout()
plt.savefig("data/sweep/partial_dependence.png", dpi=200)

# ---- 6. recommended zoom box: bounding box of the top decile, padded --------
print("\nzoom box for the refinement sweep:")
for p in PARAMS:
    lo, hi = top[p].quantile(0.05), top[p].quantile(0.95)
    pad = 0.15 * (hi - lo)
    print(f"  {p:9s}: [{max(lo - pad, valid[p].min()):.3g}, {hi + pad:.3g}]")
