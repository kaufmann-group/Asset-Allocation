import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import ks_2samp

PARAMS = ["lambda_1", "lambda_2", "lambda_3"]
METRICS = ["risk", "return", "sharpe"]
ALPHA = 0.75  # weight on noisiness vs. mean closeness
TOP_FRACTION = 0.20  # top decile

df = pd.read_parquet("../../data/sweep_refine/all_runs_long.parquet")

# ---- 1. per-row differences and health flags --------------------------------
for m in METRICS:
    df[f"d_{m}"] = df[f"caa_{m}"] - df[f"aa_{m}"]
df["degenerate"] = (
    df[[f"caa_{m}" for m in METRICS] + [f"aa_{m}" for m in METRICS]]
    .isna()
    .any(axis=1)
)
# not NaN, in risk/return, so the NaN check alone misses it on those axes.
df["degenerate"] |= ((df["caa_risk"] == 0) & (df["caa_return"] == 0)) | (
    (df["aa_risk"] == 0) & (df["aa_return"] == 0)
)


# ---- 2. global scales (pooled classical IQR, one per metric) ----------------
# Robust to outliers; do NOT scale per combo or noisy combos get flattered.
scales = {
    m: np.subtract(*np.nanpercentile(df[f"aa_{m}"], [75, 25])) for m in METRICS
}
print("global scales:", scales)
# NOTE: these scales are computed from THIS sweep's pooled classical values, so
# scores are not directly comparable to the coarse sweep's scores. Since the
# refinement box excludes the degenerate low-lambda_3 regions, the classical
# spread here is tighter and scores will look inflated relative to round 1.
# Compare RANKINGS and regions across sweeps, not raw score values. (If you want
# cross-sweep comparable scores, hardcode the coarse sweep's printed scales here.)


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
    out["score"] = score if len(gg) >= 30 else np.inf
    out["aa_raw_sum_mean"] = gg["aa_raw_sum"].mean()
    out["caa_upper_raw_sum_mean"] = gg["caa_upper_raw_sum"].mean()
    out["caa_inner_min_mean"] = gg["caa_inner_raw_sum_min"].mean()
    out["worst_raw_dev"] = (
        np.nanmax(
            np.abs(
                np.concatenate(
                    [
                        gg["aa_raw_sum"].to_numpy(),
                        gg["caa_upper_raw_sum"].to_numpy(),
                        gg["caa_inner_raw_sum_min"].to_numpy(),
                        gg["caa_inner_raw_sum_max"].to_numpy(),
                    ]
                )
                - 1.0
            )
        )
        if len(gg)
        else np.nan
    )
    out["frac_merged_communities"] = (g["caa_n_communities"] < 5).mean()
    for p in PARAMS:
        out[p] = g[p].iloc[0]
    return pd.Series(out)


combos = df.groupby("combo_id").apply(combo_stats).reset_index()
combos = combos.sort_values("score")
combos.to_csv("../../data/sweep_refine/combo_scores.csv", index=False)
print(
    combos.head(15)[
        [
            "combo_id",
            "score",
            "n_degenerate",
            "worst_raw_dev",
            "frac_merged_communities",
        ]
        + PARAMS
    ]
)

# ---- 4. top-quintile distribution shift ---------------------------------------
valid = combos[np.isfinite(combos["score"])]
top = valid.head(max(10, int(TOP_FRACTION * len(valid))))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, p in zip(axes, PARAMS):
    bins = np.logspace(np.log10(valid[p].min()), np.log10(valid[p].max()), 20)
    ax.hist(valid[p], bins=bins, density=True, alpha=0.4, label="all")
    ax.hist(top[p], bins=bins, density=True, alpha=0.6, label="top 20%")
    ax.set_xscale("log")
    ks = ks_2samp(valid[p], top[p])
    ax.set_title(f"{p}  (KS p={ks.pvalue:.3f})")
    ax.legend()
fig.suptitle("Refinement sweep: where the top quintile concentrates")
plt.tight_layout()
plt.savefig("../../data/sweep_refine/top_quintile_shift.png", dpi=200)


# ---- 5. surrogate: which parameters matter, and where -----------------------
X = valid[PARAMS].copy()
# log-space for parameters spanning orders of magnitude
for p in PARAMS:
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

fig, ax = plt.subplots(figsize=(12, 3.5))
PartialDependenceDisplay.from_estimator(gbm, X, PARAMS, ax=ax)
plt.tight_layout()
plt.savefig("../../data/sweep_refine/partial_dependence.png", dpi=200)

# ---- 6. recommended zoom box: bounding box of the top decile, padded --------
print("\nfinal region (top-quintile 5th-95th percentile, padded):")
for p in PARAMS:
    lo, hi = top[p].quantile(0.05), top[p].quantile(0.95)
    pad = 0.15 * (hi - lo)
    print(f"  {p:9s}: [{max(lo - pad, valid[p].min()):.3g}, {hi + pad:.3g}]")

# ---- 7. NEW: the artifact check - raw weight sums vs lambda_2 ---------------
# Answers whether the low-lambda_2 closeness region is genuine or an artifact of
# renormalization hiding budget violations. Healthy = raw sums near 1 everywhere;
# artifact = deviation blowing up exactly where the score is best (low lambda_2).
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.scatter(valid["lambda_2"], valid["aa_raw_sum_mean"], s=14, label="classical")
ax.scatter(
    valid["lambda_2"],
    valid["caa_upper_raw_sum_mean"],
    s=14,
    label="community (upper)",
)
ax.scatter(
    valid["lambda_2"],
    valid["caa_inner_min_mean"],
    s=14,
    label="community (worst inner)",
)
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("lambda_2")
ax.set_ylabel("mean raw weight sum")
ax.legend()
ax.set_title("Budget adherence vs constraint strength")

ax = axes[1]
sc = ax.scatter(
    valid["lambda_2"],
    valid["score"],
    c=valid["worst_raw_dev"],
    cmap="viridis",
    s=18,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("lambda_2")
ax.set_ylabel("score (lower = closer)")
plt.colorbar(sc, ax=ax, label="worst |raw sum - 1|")
ax.set_title("Is closeness bought with budget violation?")

plt.tight_layout()
plt.savefig("../../data/sweep_refine/raw_sum_check.png", dpi=200)
print(
    "\nwrote raw_sum_check.png - if the best (lowest) scores in the left of the "
    "right panel are also the brightest points, the low-lambda_2 region is suspect."
)
