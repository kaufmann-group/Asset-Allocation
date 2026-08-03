import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import random
import numpy as np
import pandas as pd

import git_root

sys.path.append("..")

from modules import *
from main import community_asset_allocation

"""
FIDELITY PROFILER (run BEFORE generate_data.py)

Holds ONE fixed parameter set and varies num_reads to find the "knee" - the read count
past which caa_sharpe / aa_sharpe stop improving. This does NOT use the Sobol design and
does NOT write per-combination files. It prints a small table and writes ONE summary CSV.
"""

# ---- ONE fixed parameter set to profile (keep these constant across all read levels) ----
NUMBER_COMMUNITIES = 5
GAMMA, BETA = 8.0, 1.0
LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1.0, 50.0, 10.0

NUM_READS = 400  # held fixed while we vary sweeps
SWEEPS_TO_TRY = [400]
REPEATS = 15  # runs per read level; we average these
BASE_SEED = 999

if __name__ == "__main__":
    # data + deterministic quantities, computed once
    price_cache = f"{git_root.git_root()}/data/prices_cache.parquet"
    if os.path.exists(price_cache):
        daily_returns = pd.read_parquet(price_cache)
    else:
        assets_dict = parse_assets_file("benchmarking_assets.txt")
        rng = np.random.default_rng(BASE_SEED)
        pick = lambda area, n: rng.choice(
            assets_dict[area], n, replace=False
        ).tolist()
        assets = (
            pick("Technology & Semiconductors", 5)
            + pick("Communication Services & Media", 5)
            + pick("Consumer Discretionary & Retail", 5)
            + pick("Industrials, Energy & Utilities", 5)
        )
        daily_returns = closing_prices(assets=assets)
        daily_returns.to_parquet(price_cache)

    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )
    graph_adjacency = get_correlation(
        daily_returns=daily_returns, zero_diagonal=True
    )

    summary = []
    for sweeps in SWEEPS_TO_TRY:
        caa_sharpes, aa_sharpes = [], []
        for r in range(REPEATS):
            seed = int(
                np.random.SeedSequence([BASE_SEED, sweeps, r]).generate_state(
                    1
                )[0]
            )
            np.random.seed(seed)
            random.seed(seed)

            caa = community_asset_allocation(
                daily_returns=daily_returns,
                number_communities=NUMBER_COMMUNITIES,
                gamma=GAMMA,
                beta=BETA,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                lambda_3=LAMBDA_3,
                num_reads=NUM_READS,
                num_sweeps=sweeps,
                write_report=False,
                annual_returns=annual_returns,
                covariance_matrix=covariance_matrix,
                graph_adjacency=graph_adjacency,
            )
            aa = AssetAllocation(
                returns=annual_returns.to_numpy(),
                covariance=covariance_matrix.to_numpy(),
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
                lambda_3=LAMBDA_3,
            ).run(
                "SIMULATED",
                num_reads=NUM_READS,
                num_sweeps=sweeps,
                write_report=False,
            )

            caa_sharpes.append(
                get_sharpe_ratio(
                    allocations=caa,
                    returns=annual_returns,
                    covariance=covariance_matrix,
                )
            )
            aa_sharpes.append(
                get_sharpe_ratio(
                    allocations=aa,
                    returns=annual_returns,
                    covariance=covariance_matrix,
                )
            )

        row = {
            "num_sweeps": sweeps,
            "caa_sharpe_mean": float(np.nanmean(caa_sharpes)),
            "caa_sharpe_std": float(np.nanstd(caa_sharpes)),
            "aa_sharpe_mean": float(np.nanmean(aa_sharpes)),
            "aa_sharpe_std": float(np.nanstd(aa_sharpes)),
        }
        summary.append(row)
        print(
            f"sweeps={sweeps:>5}  caa={row['caa_sharpe_mean']:.4f} (+/-{row['caa_sharpe_std']:.4f})  "
            f"aa={row['aa_sharpe_mean']:.4f} (+/-{row['aa_sharpe_std']:.4f})"
        )

    out = f"{git_root.git_root()}/data/fidelity_profile.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(summary).to_csv(out, index=False)
    print(
        f"\nWrote {out}. Pick the smallest num_sweeps where the means stop moving, "
        f"then set NUM_READS/NUM_SWEEPS in generate_data.py to that."
    )
