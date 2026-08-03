import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
import git_root

sys.path.append("..")

from modules import *
from main import community_asset_allocation

"""
test to see how the returns and risk of the community based asset allocation compares with that 
of the standard asset allocation algorithm. 
"""

if __name__ == "__main__":

    number_runs = 250
    number_communities = 5

    assets_dict = parse_assets_file("benchmarking_assets.txt")
    choose = lambda area, n: np.random.choice(
        assets_dict[area], n, replace=False
    ).tolist()

    assets = (
        choose("Technology & Semiconductors", 5)
        + choose("Communication Services & Media", 5)
        + choose("Consumer Discretionary & Retail", 5)
        + choose("Industrials, Energy & Utilities", 5)
    )

    """
    benchmarking
    """

    daily_returns = closing_prices(assets=assets)

    returns = daily_returns.mean() * 252  # returns
    cov_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )  # covariance matrix

    caa_rar = []  # community asset allocation risk and returns

    aa_rar = []  # asset allocation risk and returns
    # random_rar = [] # random asset allocation risk and returns

    sharpe_ratios = []  # sharpe ratios

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(
            daily_returns=daily_returns,
            number_communities=number_communities,
            adjacency=get_correlation,
        )  # community asset allocations
        caa_rar.append(
            (
                get_risk(covariance=cov_matrix, allocations=caa),
                get_returns(allocations=caa, returns=returns),
            )
        )

        """
        classical asset allocation
        """
        aa = AssetAllocation(
            returns=returns.to_numpy(), covariance=cov_matrix.to_numpy()
        )  # asset allocation object
        aas = aa.run(solver_type="SIMULATED")  # asset allocations
        aa_rar.append(
            (
                get_risk(covariance=cov_matrix, allocations=aas),
                get_returns(allocations=aas, returns=returns),
            )
        )

        """
        random allocations
        """
        # random_aa = np.random.dirichlet(np.ones(len(assets)))
        # random_rar.append((get_risk(covariance=cov_matrix, allocations=random_aa), get_returns(allocations=random_aa, returns=returns)))

        sharpe_ratios.append(
            (
                get_sharpe_ratio(
                    allocations=caa, returns=returns, covariance=cov_matrix
                ),
                get_sharpe_ratio(
                    allocations=aas, returns=returns, covariance=cov_matrix
                ),
            )
        )
        # sharpe_ratios.append((get_sharpe_ratio(allocations=caa, returns=returns, covariance=cov_matrix), get_sharpe_ratio(allocations=random_aa, returns=returns, covariance=cov_matrix)))

    """
    plotting
    """

    figure_1, axes = plt.subplots(1, 2, figsize=(12, 5))

    figure_1.suptitle(
        "Asset Allocation vs Community Based Asset Allocation",
        fontsize=14,
        y=1.00,
    )
    # figure_1.suptitle("Random Asset Allocation vs Community Based Asset Allocation", fontsize=14, y=1.00)

    axes[0].plot(*zip(*aa_rar), "bs", markersize=5, label="Asset Allocation")
    # axes[0].plot(*zip(*random_rar), "bs", markersize=5, label="Random Asset Allocation")

    axes[0].plot(
        *zip(*caa_rar), "r^", markersize=6, label="Community Asset Allocation"
    )

    axes[0].set_xlabel("Risk", fontsize=11)
    axes[0].set_ylabel("Return", fontsize=11)
    axes[0].set_title("Risk vs Returns", fontsize=12)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(*zip(*sharpe_ratios), "r*", label="Sharpe Comparison")
    axes[1].axline(
        (0, 0),
        (1, 1),
        color="k",
        linestyle=":",
        linewidth=1,
        transform=axes[1].transAxes,
    )

    axes[1].set_ylim([0, 5])
    axes[1].set_xlim([0, 5])

    axes[1].set_xlabel("Sharpe Ratio: Community Asset Allocation", fontsize=11)
    axes[1].set_ylabel("Sharpe Ratio: Asset Allocation", fontsize=11)
    # axes[1].set_ylabel("Sharpe Ratio: Random Asset Allocation", fontsize=11)

    axes[1].set_title("Sharpe Ratios Comparison", fontsize=12)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        f"{git_root.git_root()}/data/risk_and_returns_benchmark.png", dpi=300
    )
    plt.show()

    """
    save simulation run data to csv
    """
    # unzip tuple lists into individual arrays
    caa_risk, caa_return = zip(*caa_rar)
    aa_risk, aa_return = zip(*aa_rar)
    caa_sharpe, aa_sharpe = zip(*sharpe_ratios)

    data = {
        "Run Number": np.arange(1, number_runs + 1),
        "Community Asset Allocation Risk": caa_risk,
        "Community Asset Allocation Returns": caa_return,
        "Community Asset Allocation Sharpe": caa_sharpe,
        "Classical Asset Allocation Risk": aa_risk,
        "Classical Asset Allocation Returns": aa_return,
        "Classical Asset Allocation Sharpe": aa_sharpe,
    }

    df = pd.DataFrame(data)
    df.to_csv(
        f"{git_root.git_root()}/data/risk_and_returns_benchmark.csv",
        index=False,
    )
