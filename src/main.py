"""
program runs community biased asset allocation on predefined stocks.
"""

import numpy as np

from modules import *

def community_asset_allocation(daily_returns, number_communities, min_community_size,solver_type="QPU", adjacency=get_correlation):
    annual_returns = daily_returns.mean() * 252

    covariance_matrix = get_covariance(daily_returns=daily_returns, annualize=True)
    graph_adjacency = adjacency(daily_returns=daily_returns, zero_diagonal=True)

    community_detection = CommunityDetection(adjacency_matrix=graph_adjacency, number_communities=number_communities,min_community_size=min_community_size)
    community_labels,qubo = community_detection.run(solver_type=solver_type)

    partitions = [np.where(community_labels == community)[0] for community in np.unique(community_labels)]
    print("Returning partitions")

    pd.DataFrame(qubo).to_csv(
        "community_detection_qubo.csv",
        index=False
    )
    return partitions, community_labels
    # lower_allocations = []
    # community_daily_returns = []

    # # optimize assets within each community first
    # for cluster in partitions:
    #     cluster_returns = annual_returns.iloc[cluster].to_numpy()
    #     cluster_covariance = (covariance_matrix.iloc[cluster, cluster].to_numpy())

    #     inner_optimizer = AssetAllocation(returns=cluster_returns, covariance=cluster_covariance)
    #     inner_weights = inner_optimizer.run(solver_type=solver_type)

    #     lower_allocations.append(inner_weights)

    #     # return history of the actual optimized community portfolio
    #     cluster_return_history = (daily_returns.iloc[:, cluster].to_numpy() @ inner_weights)
    #     community_daily_returns.append(cluster_return_history) 

    # # stats of the optimized community portfolios
    # community_daily_returns = np.column_stack(community_daily_returns)
    # community_annual_returns = (community_daily_returns.mean(axis=0) * 252)
    # community_covariance = (np.cov(community_daily_returns, rowvar=False) * 252)

    # # optimize allocation between communities.
    # upper_optimizer = AssetAllocation(returns=community_annual_returns, covariance=community_covariance)
    # upper_weights = upper_optimizer.run(solver_type=solver_type)

    # # combine the community weights with the inner community weights.
    # allocations = np.zeros(daily_returns.shape[1])

    # for community_weight, cluster, inner_weights in zip(upper_weights, partitions, lower_allocations):
    #     allocations[cluster] = (community_weight * inner_weights)

    return allocations
if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        daily_returns = closing_prices(assets=assets)
        partitions, community_labels = community_asset_allocation(daily_returns=daily_returns, number_communities=4,min_community_size=3)
        print(f"Partitions: {partitions}")
        # for asset, allocation in zip(assets, allocations):
        #     print(f"{asset}: {allocation}")
        