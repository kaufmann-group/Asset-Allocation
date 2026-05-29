"""
benchmarking community biased asset allocation.
"""

from tqdm import tqdm

from modules import *
from main import community_asset_allocation

"""
this test runs asset allocation on 20 assets against community
detection based asset allocation on 20 assets.
"""

def benchmark_1(assets, number_runs=50, solver_type="HYBRID", number_communities=5):
    daily_returns = closing_prices(assets=assets)

    returns = daily_returns.mean() * 252 # returns 
    cov_matrix = get_covariance(daily_returns=daily_returns) # covariance matrix

    caa_rar = [] # community asset allocation risk and returns
    aa_rar = [] # asset allocation risk and returns

    for _ in tqdm(range(number_runs)):
        caa = community_asset_allocation(daily_returns=daily_returns, number_communities=number_communities, solver_type=solver_type) # community asset allocations

        caa_rar.append((getRisk(covariance=cov_matrix, allocations=caa), getReturns(allocations=caa, returns=returns)))

        aa = AssetAllocation(returns=returns.to_numpy(), covariance=cov_matrix.to_numpy()) # asset allocation object 
        aas = aa.run(solver_type=solver_type) # asset allocations 

        aa_rar.append((getRisk(covariance=cov_matrix, allocations=aas), getReturns(allocations=aas, returns=returns)))

    return aa_rar, caa_rar

if __name__ == "__main__":
    with open("benchmarking_assets.txt", "r") as file:
        total_assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        assets = total_assets[:20]

        """
        first benchmarking test
        """

        asset_allocations, community_asset_allocations = benchmark_1(assets=assets)

        plt.plot(*zip(*asset_allocations), "bo", label="asset allocations")
        plt.plot(*zip(*community_asset_allocations), "ro", label="community asset allocations")

        plt.xlabel("Risk") 
        plt.ylabel("Return")
        plt.title("Regular Asset Allocations vs Community Based Asset Allocation")
        plt.legend()

        plt.savefig("../figures/benchmark_1.png", dpi=300)
        plt.show()

        """
        second benchmarking test
        """


