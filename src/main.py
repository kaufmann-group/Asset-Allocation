"""
program runs community biased asset allocation on predefined stocks.
"""

import numpy as np

from modules import *


def community_asset_allocation( daily_returns, number_communities, solver_type="SIMULATED", adjacency=get_correlation, gamma=8.0, beta=1.0, lambda_1=1.0, lambda_2=50.0, lambda_3=10.0, bits_per_asset=6, num_reads=3000, num_sweeps=1000, annual_returns=None, covariance_matrix=None, graph_adjacency=None):

    if annual_returns is None:
        annual_returns = daily_returns.mean() * 252

    if covariance_matrix is None:
        covariance_matrix = get_covariance(daily_returns=daily_returns, annualize=True)

    if graph_adjacency is None:
        graph_adjacency = adjacency(daily_returns=daily_returns, zero_diagonal=True)

    community_detection = CommunityDetection(adjacency_matrix=graph_adjacency, number_communities=number_communities, gamma=gamma, beta=beta)
    community_labels = community_detection.run(solver_type=solver_type, num_reads=num_reads, num_sweeps=num_sweeps) # num_reads and num_sweeps only for 'SIMULATED'

    partitions = [np.where(community_labels == community)[0] for community in np.unique(community_labels)]

    lower_allocations = []
    community_daily_returns = []

    # optimize assets within each community first
    for cluster in partitions:
        cluster_returns = annual_returns.iloc[cluster].to_numpy()
        cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

        inner_optimizer = AssetAllocation(returns=cluster_returns, covariance=cluster_covariance, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, bits_per_asset=bits_per_asset)
        inner_weights = inner_optimizer.run(solver_type=solver_type, num_reads=num_reads, num_sweeps=num_sweeps) # num_reads and num_sweeps only for 'SIMULATED'

        lower_allocations.append(inner_weights)

        # return history of the actual optimized community portfolio
        cluster_return_history = (daily_returns.iloc[:, cluster].to_numpy() @ inner_weights)
        community_daily_returns.append(cluster_return_history) 

    # stats of the optimized community portfolios
    community_daily_returns = np.column_stack(community_daily_returns)
    community_annual_returns = (community_daily_returns.mean(axis=0) * 252)
    community_covariance = (np.cov(community_daily_returns, rowvar=False) * 252)

    # optimize allocation between communities.
    upper_optimizer = AssetAllocation(returns=community_annual_returns, covariance=community_covariance, lambda_1=lambda_1, lambda_2=lambda_2, lambda_3=lambda_3, bits_per_asset=bits_per_asset)
    upper_weights = upper_optimizer.run(solver_type=solver_type, num_reads=num_reads, num_sweeps=num_sweeps)

    # combine the community weights with the inner community weights.
    allocations = np.zeros(daily_returns.shape[1])

    for community_weight, cluster, inner_weights in zip(upper_weights, partitions, lower_allocations):
        allocations[cluster] = (community_weight * inner_weights)

    return allocations

if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        daily_returns = closing_prices(assets=assets)
        allocations = community_asset_allocation(daily_returns=daily_returns, number_communities=4)

        for asset, allocation in zip(assets, allocations):
            print(f"{asset}: {allocation}")
        