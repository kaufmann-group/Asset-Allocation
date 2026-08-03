import random
import numpy as np
import pandas as pd

from _diagnostic_common import (
    load_market_data,
    NUMBER_COMMUNITIES,
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_3,
    NUM_READS,
    NUM_SWEEPS,
    BASE_SEED,
)
import sys

sys.path.append("..")
from modules import AssetAllocation
from main import community_asset_allocation

"""
TEST 2 - COMMUNITY PORTFOLIO vs CLASSICAL PORTFOLIO

Compares the community model's weight vector (caa) against the classical one (aas) for a few
settings. Answers: does grouping actually change the portfolio, or has the model collapsed to
classical?

  L1 weight distance ~ 0    -> collapsed to classical (grouping changes nothing)
  L1 weight distance  > noise floor -> genuinely different portfolios (grouping matters)

It also measures a NOISE FLOOR: classical run twice on different seeds. Any community-vs-classical
distance below that floor is just annealer noise, not a real grouping effect.
"""

# a few (gamma, beta) settings to probe; grouping side only
SETTINGS = [(1.0, 1.0), (8.0, 1.0), (20.0, 1.0), (8.0, 0.1), (8.0, 5.0)]


def classical_weights(annual_returns, covariance_matrix, seed):
    np.random.seed(seed)
    random.seed(seed)
    aa = AssetAllocation(
        returns=annual_returns.to_numpy(),
        covariance=covariance_matrix.to_numpy(),
        lambda_1=LAMBDA_1,
        lambda_2=LAMBDA_2,
        lambda_3=LAMBDA_3,
    )
    return aa.run(
        solver_type="SIMULATED",
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        write_report=False,
    )


def main():
    (
        daily_returns,
        asset_names,
        annual_returns,
        covariance_matrix,
        graph_adjacency,
    ) = load_market_data()

    # ---- noise floor: classical vs itself on two different seeds ----
    aas_a = classical_weights(annual_returns, covariance_matrix, BASE_SEED)
    aas_b = classical_weights(annual_returns, covariance_matrix, BASE_SEED + 1)
    noise_floor = float(np.abs(aas_a - aas_b).sum())
    print("=" * 78)
    print(
        f"NOISE FLOOR (classical vs classical, different seeds): L1 = {noise_floor:.4f}"
    )
    print(
        "Community-vs-classical distances must clearly EXCEED this to count as a real effect."
    )
    print("=" * 78)
    print()

    classical = pd.Series(np.round(aas_a, 4), index=asset_names)

    rows = []
    for gamma, beta in SETTINGS:
        seed = int(
            np.random.SeedSequence(
                [BASE_SEED, int(gamma * 100), int(beta * 100)]
            ).generate_state(1)[0]
        )
        np.random.seed(seed)
        random.seed(seed)
        caa = community_asset_allocation(
            daily_returns=daily_returns,
            number_communities=NUMBER_COMMUNITIES,
            gamma=gamma,
            beta=beta,
            lambda_1=LAMBDA_1,
            lambda_2=LAMBDA_2,
            lambda_3=LAMBDA_3,
            num_reads=NUM_READS,
            num_sweeps=NUM_SWEEPS,
            write_report=False,
            annual_returns=annual_returns,
            covariance_matrix=covariance_matrix,
            graph_adjacency=graph_adjacency,
        )

        l1 = float(np.abs(caa - aas_a).sum())
        verdict = (
            "DIFFERENT"
            if l1 > 2 * noise_floor
            else ("borderline" if l1 > noise_floor else "COLLAPSED")
        )
        rows.append((gamma, beta, l1, verdict))

        print("-" * 78)
        print(
            f"gamma = {gamma:<5}  beta = {beta:<5}   L1 weight distance = {l1:.4f}   [{verdict}]"
        )
        community = pd.Series(np.round(caa, 4), index=asset_names)
        cmp = pd.DataFrame(
            {"community(caa)": community, "classical(aas)": classical}
        )
        cmp = cmp[
            (cmp["community(caa)"] > 1e-6) | (cmp["classical(aas)"] > 1e-6)
        ]
        print(cmp.to_string())
        print()

    print("=" * 78)
    print("SUMMARY (L1 weight distance vs classical)")
    print("=" * 78)
    print(f"noise floor = {noise_floor:.4f}")
    for gamma, beta, l1, verdict in rows:
        print(f"  gamma={gamma:<5} beta={beta:<5}  L1={l1:.4f}  {verdict}")
    print()
    print("COLLAPSED  -> grouping changed nothing (model == classical)")
    print("DIFFERENT  -> grouping produced a genuinely different portfolio")


if __name__ == "__main__":
    main()
