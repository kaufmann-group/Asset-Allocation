import random
import numpy as np

from diagnostic_common import (
    load_market_data,
    partition_from_labels,
    NUMBER_COMMUNITIES,
    NUM_READS,
    NUM_SWEEPS,
    BASE_SEED,
)
import sys

sys.path.append("..")
from modules import CommunityDetection

"""
TEST 1 - PARTITION STRUCTURE

Runs ONLY the community-detection step across a gamma/beta grid and prints the resulting
asset groupings. Answers: are the groups meaningful, and do they change as gamma/beta vary?

  distinct partitions == 1  -> grouping is STABLE across the range; the earlier zero
                               correlation means 'robust partition', not 'meaningless'.
  distinct partitions  > 1  -> grouping DOES move with gamma/beta.
Repeats each setting to separate a genuinely stable partition from annealer noise.
"""

GAMMA_VALUES = [1.0, 5.0, 8.0, 14.0, 20.0]
BETA_VALUES = [0.1, 1.0, 3.0, 5.0]
REPEATS = 3


def main():
    _, asset_names, _, _, graph_adjacency = load_market_data()
    distinct = {}

    for beta in BETA_VALUES:
        for gamma in GAMMA_VALUES:
            print("-" * 78)
            print(f"gamma = {gamma:<5}  beta = {beta:<5}")
            print("-" * 78)

            signatures = []
            first_groups = None
            for rep in range(REPEATS):
                seed = int(
                    np.random.SeedSequence(
                        [BASE_SEED, int(gamma * 100), int(beta * 100), rep]
                    ).generate_state(1)[0]
                )
                np.random.seed(seed)
                random.seed(seed)

                cd = CommunityDetection(
                    adjacency_matrix=graph_adjacency,
                    number_communities=NUMBER_COMMUNITIES,
                    gamma=gamma,
                    beta=beta,
                )
                labels = cd.run(
                    solver_type="SIMULATED",
                    num_reads=NUM_READS,
                    num_sweeps=NUM_SWEEPS,
                    write_report=False,
                )

                groups = partition_from_labels(labels, asset_names)
                signature = tuple(sorted(tuple(sorted(g)) for g in groups))
                signatures.append(signature)
                distinct.setdefault(signature, (gamma, beta))
                if first_groups is None:
                    first_groups = groups

            n_unique = len(set(signatures))
            print(
                f"  across {REPEATS} repeats: "
                f"{'STABLE' if n_unique == 1 else f'VARIES ({n_unique} distinct partitions)'}"
            )
            sizes = [len(g) for g in first_groups]
            for gi, grp in enumerate(first_groups):
                print(f"    group {gi} ({len(grp)}): {grp}")
            # degeneracy flags
            if len(first_groups) == 1:
                print("    !! all assets in ONE group - degenerate")
            if all(s == 1 for s in sizes):
                print("    !! every asset in its own group - degenerate")
            print()

    print("=" * 78)
    print(
        f"SUMMARY: {len(distinct)} distinct partition(s) across "
        f"{len(GAMMA_VALUES) * len(BETA_VALUES)} gamma/beta settings"
    )
    print("=" * 78)
    if len(distinct) == 1:
        print("Grouping NEVER changes across the swept gamma/beta range.")
        print(
            "-> zero correlation = STABLE/ROBUST partition, not 'meaningless'."
        )
        print(
            "-> widening gamma/beta only matters if you want to probe OUTSIDE this range."
        )
    else:
        print(
            "Grouping changes with gamma/beta. Distinct partitions (first seen at):"
        )
        for i, (sig, where) in enumerate(distinct.items()):
            print(
                f"  partition {i}: gamma={where[0]}, beta={where[1]}  |  "
                f"{len(sig)} groups, sizes {sorted(len(g) for g in sig)}"
            )


if __name__ == "__main__":
    main()
