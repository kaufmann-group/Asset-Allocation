import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import numpy as np
import pandas as pd

import git_root

sys.path.append("..")

from modules import *  # noqa: F401,F403  (closing_prices, get_covariance, get_correlation, etc.)

# ---- settings shared by both diagnostics ----
NUMBER_COMMUNITIES = 5
LAMBDA_1, LAMBDA_2, LAMBDA_3 = 1.0, 50.0, 10.0
NUM_READS = 400
NUM_SWEEPS = 300
BASE_SEED = 4242


def load_market_data():
    """Reuse the cached prices so diagnostics match the sweep's asset universe."""
    price_cache = f"{git_root.git_root()}/data/prices_cache.parquet"
    if os.path.exists(price_cache):
        daily_returns = pd.read_parquet(price_cache)
    else:
        assets_dict = parse_assets_file("benchmarking_assets.txt")
        rng = np.random.default_rng(
            20240101
        )  # matches generate_data.py's default universe
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

    asset_names = list(daily_returns.columns)
    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )
    graph_adjacency = get_correlation(
        daily_returns=daily_returns, zero_diagonal=True
    )
    return (
        daily_returns,
        asset_names,
        annual_returns,
        covariance_matrix,
        graph_adjacency,
    )


def partition_from_labels(labels, asset_names):
    """Turn a community-label vector into a list of asset-name groups."""
    return [
        [asset_names[i] for i in range(len(labels)) if labels[i] == c]
        for c in sorted(set(labels))
    ]
