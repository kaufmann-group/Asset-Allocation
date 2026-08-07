# # Purdue Experimental Math Lab: Kaufmann Group

import numpy as np
import pandas as pd
import networkx as nx

import os
import numpy as np
from dotenv import load_dotenv
from src import *

load_dotenv()

token = "uyoo-7caa187a8128d495d7c78170ec88b2c94310fcf0"

num_partitions = 3
gamma=8.0
beta=1.0

with open('assets.txt', 'r') as file:
    assets = [line.strip() for line in file]

print(assets)

daily_returns = closing_prices(assets=assets, start="2020-01-01")
returns = daily_returns.mean() * 252

covariance_matrix = get_covarience(daily_returns=daily_returns)

import random
def generate_subsets(assets):
    result = []
    for k in range(5, 15):  # k = 5 to 20
        subset = random.sample(assets, k)
        result.append(subset)
    return result

subsets = generate_subsets(assets)

for k, subset in zip(range(5, 15), subsets):
    print(f"k={k}: {subset}")
# subset_allocations = []
# for subset in subsets:
#     subset_indices = [assets.index(asset) for asset in subset]
#     subset_returns = returns.iloc[subset_indices]
#     subset_covariance = covariance_matrix.iloc[subset_indices, subset_indices].to_numpy()
#     Q_subset = build_aa_qubo(n=len(subset), mu=subset_returns, C=subset_covariance)
#     best_sample_subset, _, _ = solve_qubo(Q=Q_subset, token=token)
#     subset_allocations.append(binary_to_float(bits=best_sample_subset, split=len(subset)))   


results = []
subset_allocations = []

for subset in subsets:
    subset_indices = [assets.index(asset) for asset in subset]

    subset_returns = returns.iloc[subset_indices]
    subset_covariance = covariance_matrix.iloc[
        subset_indices, subset_indices
    ].to_numpy()

    Q_subset = build_aa_qubo(
        n=len(subset),
        mu=subset_returns,
        C=subset_covariance
    )

    # 🔴 capture timing here
    best_sample_subset, _, sampleset, timing_info = solve_qubo(
        Q=Q_subset,
        token=token
    )

    subset_allocations.append(
        binary_to_float(bits=best_sample_subset, split=len(subset))
    )

    if timing_info:
        access_times = [t["qpu_access_time"] for t in timing_info if "qpu_access_time" in t]
        avg_time = np.mean(access_times) if access_times else np.nan
    else:
        avg_time = np.nan

    results.append({
        "subset_size": len(subset),
        "qpu_access_time_us": avg_time
    })

# 🔴 create dataframe
df = pd.DataFrame(results)
# df.to_csv("timing_results.csv", index=False)
df = pd.read_csv("timing_results.csv")
df['qpu_access_time_ms'] = df['qpu_access_time_us']/1000
plt.plot(df["subset_size"], df["qpu_access_time_ms"], marker='o')
plt.xlabel("Subset Size")
plt.ylabel("QPU Access Time (ms)")
plt.title("Asset Allocation Complexity")
plt.show()

