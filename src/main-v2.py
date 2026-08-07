"""
program runs community biased asset allocation on predefined stocks and prints graph metadata.
"""

import numpy as np
from modules import *
import pandas as pd

def get_dcca_matrix(daily_returns, scale=16, zero_diagonal=True):
    """
    Computes the Detrended Cross-Correlation Analysis (DCCA) coefficient matrix 
    for multivariate daily returns at a time scale 's' (window size).
    """
    data = daily_returns.to_numpy() if isinstance(daily_returns, (pd.DataFrame, pd.Series)) else np.asarray(daily_returns)
    T, M = data.shape  # T = time steps, M = number of assets

    # 1. Compute cumulative sum of deviations from mean (integrated profile)
    profiles = np.cumsum(data - np.mean(data, axis=0, keepdims=True), axis=0)

    # 2. Divide into non-overlapping segments of length 'scale'
    num_segments = T // scale
    if num_segments == 0:
        raise ValueError(f"Time series length ({T}) is smaller than scale size ({scale}).")

    truncated_profiles = profiles[:num_segments * scale]
    segments = truncated_profiles.reshape(num_segments, scale, M)

    # 3. Vectorized linear detrending across all segments and assets
    t = np.arange(scale)
    A = np.column_stack([np.ones(scale), t])  # Design matrix [1, t]
    pinv_A = np.linalg.pinv(A)                # Pseudo-inverse for regression

    detrended = np.zeros_like(segments)
    for k in range(num_segments):
        # Fit linear trend per segment: beta shape (2, M)
        params = pinv_A @ segments[k]
        trend = A @ params                     # Trend shape (scale, M)
        detrended[k] = segments[k] - trend

    # Flatten detrended segments: (N_samples, M)
    detrended_flat = detrended.reshape(-1, M)

    # 4. Detrended cross-covariance matrix F_DCCA^2(s)
    f_dcca = (detrended_flat.T @ detrended_flat) / (num_segments * scale)

    # 5. Normalize by DFA variances: rho_DCCA(s) = F_DCCA^2 / (F_DFA_x * F_DFA_y)
    variances = np.diag(f_dcca)
    std_devs = np.sqrt(np.maximum(variances, 1e-12))
    denom = np.outer(std_devs, std_devs)

    dcca_matrix = np.clip(f_dcca / denom, -1.0, 1.0)

    if zero_diagonal:
        np.fill_diagonal(dcca_matrix, 0.0)

    if isinstance(daily_returns, pd.DataFrame):
        return pd.DataFrame(dcca_matrix, index=daily_returns.columns, columns=daily_returns.columns)

    return dcca_matrix


"""
Program runs community-biased asset allocation using Detrended Cross-Correlation Analysis (DCCA)
to build the market graph.
"""

import numpy as np
import networkx as nx
from modules import *


def analyze_qubo_graph(optimizer, name="Optimizer"):
    """Extracts logical QUBO graph stats and queries minor embedding details."""
    Q = optimizer.build_qubo()
    num_vars = len(Q)

    interaction_edges = [
        (u, v)
        for u in range(num_vars)
        for v in range(u + 1, num_vars)
        if Q[u, v] != 0
    ]
    num_edges = len(interaction_edges)

    print(f"\n--- {name} Graph Info ---")
    print(f"  • Logical Variables (Qubits): {num_vars}")
    print(f"  • Logical Interaction Edges:  {num_edges}")

    if hasattr(optimizer, "get_minor_embedding"):
        try:
            embedding = optimizer.get_minor_embedding()
            if embedding:
                chain_lengths = [len(chain) for chain in embedding.values()]
                total_physical_qubits = sum(chain_lengths)
                max_chain = max(chain_lengths)
                avg_chain = np.mean(chain_lengths)

                print(f"  • Embedded Physical Qubits:   {total_physical_qubits}")
                print(f"  • Max Chain Length:          {max_chain} physical qubits")
                print(f"  • Avg Chain Length:          {avg_chain:.2f} physical qubits")
            else:
                print("  • Embedding: Failed to find valid mapping.")
        except Exception as e:
            print(f"  • Embedding Query Error: {e}")

def community_asset_allocation(
    daily_returns,
    number_communities,
    solver_type="QPU",
    scale=16,
    modularity_filename="modularity_matrix.csv",
):
    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(daily_returns=daily_returns, annualize=True)

    # -------------------------------------------------------------
    # DCCA Adjacency Matrix replacing Pearson correlation
    # -------------------------------------------------------------
    graph_adjacency = get_dcca_matrix(
        daily_returns=daily_returns, 
        scale=scale, 
        zero_diagonal=True
    )

    if hasattr(graph_adjacency, "to_numpy"):
        adjacency_values = graph_adjacency.to_numpy()
    else:
        adjacency_values = np.array(graph_adjacency)

    # -------------------------------------------------------------
    # Calculate and Save Modularity Matrix B = A - (k_i * k_j) / 2m
    # -------------------------------------------------------------
    k = np.sum(adjacency_values, axis=1)  # Node strengths (degrees)
    two_m = np.sum(k)                     # Total degree sum (2m)

    if two_m != 0:
        modularity_matrix = adjacency_values - (np.outer(k, k) / two_m)
    else:
        modularity_matrix = np.zeros_like(adjacency_values)

    modularity_df = pd.DataFrame(
        modularity_matrix,
        index=daily_returns.columns,
        columns=daily_returns.columns,
    )
    modularity_df.to_csv(modularity_filename)
    print(f"Modularity matrix saved to '{modularity_filename}'")

    # 1. Market Graph Overview
    market_graph = nx.from_numpy_array(np.abs(adjacency_values))
    print("\n" + "=" * 50)
    print(" 1. DCCA MARKET GRAPH OVERVIEW")
    print("=" * 50)
    print(f"Time Scale (s):              {scale} trading days")
    print(f"Total Market Assets (Nodes): {market_graph.number_of_nodes()}")
    print(f"Correlation Edges:           {market_graph.number_of_edges()}")
    print(f"Graph Density:               {nx.density(market_graph):.4f}")

    # Community detection using DCCA graph
    community_detection = CommunityDetection(
        adjacency_matrix=graph_adjacency,
        number_communities=number_communities,
    )
    community_labels = community_detection.run(solver_type=solver_type)

    partitions = [
        np.where(community_labels == community)[0]
        for community in np.unique(community_labels)
    ]

    print("\n" + "=" * 50)
    print(" 2. COMMUNITY PARTITIONS & INNER OPTIMIZERS")
    print("=" * 50)

    lower_allocations = []
    community_daily_returns = []

    # 2. Inner Community Optimization
    for idx, cluster in enumerate(partitions):
        cluster_assets = daily_returns.columns[cluster].tolist()
        print(f"\n[Community {idx + 1}] ({len(cluster)} assets)")
        print(f"  Assets: {', '.join(cluster_assets)}")

        cluster_returns = annual_returns.iloc[cluster].to_numpy()
        cluster_covariance = covariance_matrix.iloc[cluster, cluster].to_numpy()

        inner_optimizer = AssetAllocation(
            returns=cluster_returns, covariance=cluster_covariance
        )

        analyze_qubo_graph(inner_optimizer, name=f"Inner Community {idx + 1}")

        inner_weights = inner_optimizer.run(solver_type=solver_type)
        lower_allocations.append(inner_weights)

        cluster_return_history = (
            daily_returns.iloc[:, cluster].to_numpy() @ inner_weights
        )
        community_daily_returns.append(cluster_return_history)

    # Stats of the optimized community portfolios
    community_daily_returns = np.column_stack(community_daily_returns)
    community_annual_returns = community_daily_returns.mean(axis=0) * 252
    community_covariance = np.cov(community_daily_returns, rowvar=False) * 252

    # 3. Inter-Community (Upper Level) Optimization
    print("\n" + "=" * 50)
    print(" 3. INTER-COMMUNITY (UPPER LEVEL) OPTIMIZER")
    print("=" * 50)

    upper_optimizer = AssetAllocation(
        returns=community_annual_returns, covariance=community_covariance
    )
    Q_upper = upper_optimizer.build_qubo()
    pd.DataFrame(Q_upper).to_csv(
        "qubo_asset_allocation_upper.csv",
        index=False
    )
    Q_inner = inner_optimizer.build_qubo()
    pd.DataFrame(Q_inner).to_csv(
        f"qubo_asset_allocation_inner_{idx+1}.csv",
        index=False
    )
    Q_inner = inner_optimizer.build_qubo()
    pd.DataFrame(Q_inner).to_csv(
        f"qubo_asset_allocation_inner_{idx+1}.csv",
        index=False
    )
    analyze_qubo_graph(upper_optimizer, name="Upper Level Meta-Asset")

    upper_weights = upper_optimizer.run(solver_type=solver_type)

    # 4. Final Combination
    allocations = np.zeros(daily_returns.shape[1])
    for community_weight, cluster, inner_weights in zip(
        upper_weights, partitions, lower_allocations
    ):
        allocations[cluster] = community_weight * inner_weights

    return allocations


if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [
            ticker
            for line in file
            if (
                ticker := line.split("#")[0]
                .split("-")[0]
                .split(" ")[0]
                .strip()
            )
        ]

    daily_returns = closing_prices(assets=assets)
    
    # Run allocation with scale s=16 (~3 weeks of trading)
    allocations = community_asset_allocation(
        daily_returns=daily_returns, 
        number_communities=4,
        scale=16
    )

    print("\n" + "=" * 50)
    print(" 4. FINAL PORTFOLIO ALLOCATIONS")
    print("=" * 50)
    for asset, allocation in zip(assets, allocations):
        print(f"{asset:10s}: {allocation:.4f}")

if __name__ == "__main__":
    with open("assets.txt", "r") as file:
        assets = [
            ticker
            for line in file
            if (
                ticker := line.split("#")[0]
                .split("-")[0]
                .split(" ")[0]
                .strip()
            )
        ]

    daily_returns = closing_prices(assets=assets)
    
    # Run allocation with scale s=16 (~3 weeks of trading)
    allocations = community_asset_allocation(
        daily_returns=daily_returns, 
        number_communities=4,
        scale=16
    )

    print("\n" + "=" * 50)
    print(" 4. FINAL PORTFOLIO ALLOCATIONS")
    print("=" * 50)
    for asset, allocation in zip(assets, allocations):
        print(f"{asset:10s}: {allocation:.4f}")