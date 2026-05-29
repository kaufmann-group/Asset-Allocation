"""
program runs community biased asset allocation on predefined stocks.
"""

import numpy as np
import networkx as nx

from modules import *

def community_asset_allocation(daily_returns, number_communities, solver_type="QPU", gamma=8.0, beta=1.0):
    returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(daily_returns=daily_returns) # should we replace this with something else? 

    Graph = nx.from_numpy_array(covariance_matrix.to_numpy())
    #draw_graph(Graph=Graph, name=".graph.png") # comment out 

    community_detection = CommunityDetection(adjacency_matrix=covariance_matrix, number_communities=number_communities)
    communities = community_detection.run(solver_type=solver_type)

    comms = {}
    nodes = list(Graph.nodes())
    for node, c in zip(nodes, communities):
        comms.setdefault(int(c), set()).add(node)
    partitions = [group for group in comms.values() if group]

    #draw_graph(Graph=Graph, name=".graph_with_communities.png", labels=communities) # comment this out

    group_average_returns = {}
    group_daily_returns = np.zeros((len(daily_returns), number_communities))

    for group_index, asset_group in enumerate(partitions):
        asset_group = list(asset_group)

        group_average_returns[group_index] = returns.iloc[asset_group].mean()
        group_daily_returns[:, group_index] = daily_returns.iloc[:, asset_group].mean(axis=1).to_numpy()

    partition_covariance_matrix = np.cov(group_daily_returns, rowvar=False)

    upper_allocation = AssetAllocation(returns=list(group_average_returns.values()), covariance=partition_covariance_matrix)
    upper_allocations = upper_allocation.run(solver_type=solver_type)

    lower_allocations = []

    for i, cluster in enumerate(partitions):
        cluster = list(cluster)

        cluster_returns = [returns.iloc[asset] for asset in cluster]
        cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

        inner_allocation = AssetAllocation(returns=cluster_returns, covariance=cluster_covariance) # we use covariance in asset allocation? is that nessisary?  
        lower_allocations.append(inner_allocation.run(solver_type=solver_type))

    allocations = np.array([x * y for x, group in zip(upper_allocations, lower_allocations) for y in group])

    return allocations

if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [ticker for line in file if (ticker := line.split("#")[0].split("-")[0].split(" ")[0].strip())]

        daily_returns = closing_prices(assets=assets)
        allocations = community_asset_allocation(daily_returns=daily_returns, assets=assets, number_communities=2)

        for asset, allocation in zip(assets, allocations):
            print(f"{asset}: {allocation}")