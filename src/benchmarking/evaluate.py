"""
This was first considered, and then dropped in favor of Post_FirstGeneration_Evaluate.py and Post_SecondGeneration_Evaluate.py
Run those files instead
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

"""
STAGE 2 - CLOSENESS EVALUATION (two-sided)

Goal: find the parameter combinations whose community portfolio behaves as close as
possible to the classical portfolio, measured on the ACTUAL portfolio return and risk
(not the Sharpe ratio).

Score = normalized Euclidean distance between (caa_return, caa_risk) and (aa_return, aa_risk),
each axis scaled by its sweep-wide spread so return and risk contribute comparably, and with
per-axis measurement uncertainty folded in so noisy configs cannot masquerade as close.
Lower score = closer / better.

TWO-SIDED vs ONE-SIDED:
    This scores two-sided closeness (deviating above or below classical counts equally).
    To make an axis one-sided-from-below (only penalize the community portfolio for coming
    in LOWER than classical), replace  abs(m)  with  max(0.0, -m)  in _axis_distance below.
"""

# ------------------------------------------------------------------ config
DATA = "../../data/sweep/all_runs_long.parquet"
OUT = "../../data/two_sided_closeness_ranking.csv"
CONF = 0.95  # one-sided confidence level for the per-axis uncertainty term
TOP_N = 20
PARAMS = ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]
# -------------------------------------------------------------------------


def _axis_distance(diffs, scale):
    """Confidence-aware distance from zero for one axis (return or risk), in normalized units.

    diffs : per-run differences (caa - aa) for this combination, on one axis.
    scale : sweep-wide std of that axis's differences, used to normalize across axes.
    Returns (|mean| + k*SE) / scale.
    (For one-sided-from-below on this axis, swap abs(m) for max(0.0, -m).)
    """
    n = len(diffs)
    m = float(np.mean(diffs))
    s = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    k = stats.t.ppf(CONF, max(n - 1, 1))
    penalized = (
        abs(m) + k * se
    )  # <-- two-sided. one-sided: max(0.0, -m) + k * se
    return penalized / scale if scale > 0 else penalized


def main():
    df = pd.read_parquet(DATA)
    df["d_return"] = df["caa_return"] - df["aa_return"]
    df["d_risk"] = df["caa_risk"] - df["aa_risk"]

    return_scale = df["d_return"].std(ddof=1)
    risk_scale = df["d_risk"].std(ddof=1)

    records = []
    for combo_id, g in df.groupby("combo_id"):
        dist_return = _axis_distance(g["d_return"].to_numpy(), return_scale)
        dist_risk = _axis_distance(g["d_risk"].to_numpy(), risk_scale)
        distance = np.sqrt(dist_return**2 + dist_risk**2)

        rec = {
            "combo_id": combo_id,
            "distance": distance,
            "dist_return": dist_return,
            "dist_risk": dist_risk,
            "mean_d_return": float(g["d_return"].mean()),
            "mean_d_risk": float(g["d_risk"].mean()),
            "caa_return_mean": float(g["caa_return"].mean()),
            "aa_return_mean": float(g["aa_return"].mean()),
            "caa_risk_mean": float(g["caa_risk"].mean()),
            "aa_risk_mean": float(g["aa_risk"].mean()),
            "caa_sharpe_mean": float(g["caa_sharpe"].mean()),  # reference only
            "aa_sharpe_mean": float(g["aa_sharpe"].mean()),
            "n": len(g),
        }
        for p in PARAMS:
            rec[p] = g[p].iloc[0]
        records.append(rec)

    summary = (
        pd.DataFrame(records).sort_values("distance").reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    summary.to_csv(OUT, index=False)

    show = ["combo_id", "distance", "mean_d_return", "mean_d_risk"] + PARAMS
    pd.set_option("display.width", 160, "display.max_columns", 30)
    print(
        f"Closest {TOP_N} configurations (lower distance = behaves more like classical):\n"
    )
    print(summary[show].head(TOP_N).to_string(index=False))
    print(f"\nFull ranking of {len(summary)} configurations written to {OUT}")


if __name__ == "__main__":
    main()

"""
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_parquet("../../data/sweep/all_runs_long.parquet")
df["delta"] = df["caa_sharpe"] - df["aa_sharpe"]


def summarise(g):
    n = len(g)
    delta = g["caa_sharpe"] - g["aa_sharpe"]
    m, s = delta.mean(), delta.std(ddof=1)
    se = s / np.sqrt(n)

    # if the paired differences are effectively constant, the t-test is
    # meaningless (and numerically unstable). Decide by the sign of the edge.
    if s < 1e-9:
        t = np.inf if m > 0 else (-np.inf if m < 0 else 0.0)
        p = (
            0.0 if m > 0 else 1.0
        )  # one-sided: a stable positive edge is "certain"
    else:
        t, p = stats.ttest_rel(g["caa_sharpe"], g["aa_sharpe"])

    return pd.Series(
        {
            "n": n,
            "mean_delta": m,
            "std_delta": s,
            "lcb": m - stats.t.ppf(0.99, n - 1) * se,
            "t_stat": t,
            "p_value": p,
            "caa_sharpe_mean": g["caa_sharpe"].mean(),
            "aa_sharpe_mean": g["aa_sharpe"].mean(),
        }
    )


params = ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]
summary = (
    df.groupby("combo_id")
    .apply(lambda g: pd.concat([g[params].iloc[0], summarise(g)]))
    .reset_index(drop=True)
)

# multiple-comparison control via FDR on the paired-test p-values
from scipy.stats import false_discovery_control

summary["p_fdr"] = false_discovery_control(summary["p_value"].values)
winners = summary[(summary["lcb"] > 0) & (summary["p_fdr"] < 0.01)].sort_values(
    "lcb", ascending=False
)
winners.to_csv("../../data/winners.csv", index=False)

top = pd.read_csv("../../data/winners.csv").head(63)
for p in ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]:
    print(f"{p:8s} range in top20: {top[p].min():7.3f} .. {top[p].max():7.3f}")

"""
#
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

n = 40
dof = n - 1
confidence = 0.99  # one-sided
k = stats.t.ppf(confidence, dof)
print(k)  # 1.6849...

lcb_array = []
mean_delta = []
std_delta = []
combo_id = []
gamma_arr = []
beta_arr = []
l1_arr = []
l2_arr = []
l3_arr = []


def per_file(path):
    df = pd.read_csv(path)
    df["Delta"] = df["caa_sharpe"] - df["aa_sharpe"]
    mean_of_diff = np.mean(df["Delta"])
    std_of_diff = np.std(df["Delta"], ddof=1)
    lcb = mean_of_diff - (k) * ((std_of_diff) / np.sqrt(n))
    id = df["combo_id"].iloc[0]
    gamma = df["gamma"].iloc[0]
    beta = df["beta"].iloc[0]
    lambda_1 = df["lambda_1"].iloc[0]
    lambda_2 = df["lambda_2"].iloc[0]
    lambda_3 = df["lambda_3"].iloc[0]

    return [
        id,
        lcb,
        mean_of_diff,
        std_of_diff,
        gamma,
        beta,
        lambda_1,
        lambda_2,
        lambda_3,
    ]


# -----------------
# loop_over_files
# -----------------

for i in range(512):
    s = f"{ i:04d}"
    fil = "../../data/sweep/combo_" + s + ".csv"
    path = Path(fil)
    # for fil in Path("../../data/sweep").glob("combo_*.csv"):
    arr = per_file(fil)

    combo_id.append(arr[0])
    lcb_array.append(arr[1])
    mean_delta.append(arr[2])
    std_delta.append(arr[3])
    gamma_arr.append(arr[4])
    beta_arr.append(arr[5])
    l1_arr.append(arr[6])
    l2_arr.append(arr[7])
    l3_arr.append(arr[8])

# -----------------
# Compile into summary dataframe
# -----------------

summary_df = pd.DataFrame(
    {
        "Combo_id": combo_id,
        "Score": lcb_array,
        "Mean_delta": mean_delta,
        "StDev_delta": std_delta,
        "Gamma": gamma_arr,
        "Beta": beta_arr,
        "Lambda1": l1_arr,
        "Lambda2": l2_arr,
        "Lambda3": l3_arr,
    }
)

# -----------------
# Compile into winners
# -----------------

winners = (summary_df[summary_df["Score"] > 0]).sort_values(
    "Score", ascending=False
)
print(len(winners))
print()
print(winners)
print()
winners.to_csv("../../data/winners.csv", index=False)
"""
