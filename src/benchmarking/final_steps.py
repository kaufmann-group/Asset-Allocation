# confirm_candidate.py
# ------------------------------------------------------------------
# STAGE 3 - CONFIRMATION RUN
#
# Re-evaluates the winning configuration (combo 100 from the refinement
# sweep) under fresh annealer noise, with full weight vectors recorded,
# crossed with a small gamma grid to settle the community-merging question.
#
# Outputs, per (gamma, run):
#   * risk / return / sharpe for both models      (fresh-noise closeness)
#   * the full 20-asset weight vectors            (for the L1 weight check)
#   * n_communities, raw weight sums              (health diagnostics)
# ------------------------------------------------------------------
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import random
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

import git_root

sys.path.append("..")

from modules import *
from main import community_asset_allocation

# ---------------- configuration ----------------
CANDIDATE = dict(lambda_1=6.79, lambda_2=20.5, lambda_3=29.95)
BETA_FIXED = 1.0
GAMMAS = [
    8.0,
    12.0,
    16.0,
    20.0,
]  # gamma experiment folded in; [8.0] for candidate only
RUNS_PER_GAMMA = 200

NUMBER_COMMUNITIES = 5
NUM_READS = 400  # same fidelity as both sweeps
NUM_SWEEPS = 300

BASE_SEED = 20260801  # fresh again - independent of both sweeps
WORKERS = os.cpu_count()
OUT_DIR = f"{git_root.git_root()}/data/confirm"
# ------------------------------------------------

_SHARED = {}


def _init_worker(
    daily_returns, annual_returns, covariance_matrix, graph_adjacency
):
    _SHARED["daily_returns"] = daily_returns
    _SHARED["annual_returns"] = annual_returns
    _SHARED["covariance_matrix"] = covariance_matrix
    _SHARED["graph_adjacency"] = graph_adjacency


def _one_task(job):
    gamma, run_idx = job
    seed = int(
        np.random.SeedSequence(
            [BASE_SEED, int(gamma * 1000), run_idx]
        ).generate_state(1)[0]
    )
    np.random.seed(seed)
    random.seed(seed)

    daily_returns = _SHARED["daily_returns"]
    annual_returns = _SHARED["annual_returns"]
    covariance_matrix = _SHARED["covariance_matrix"]
    graph_adjacency = _SHARED["graph_adjacency"]

    caa, caa_diag = community_asset_allocation(
        daily_returns=daily_returns,
        number_communities=NUMBER_COMMUNITIES,
        gamma=gamma,
        beta=BETA_FIXED,
        **CANDIDATE,
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        write_report=False,
        annual_returns=annual_returns,
        covariance_matrix=covariance_matrix,
        graph_adjacency=graph_adjacency,
        collect_diagnostics=True,
    )

    aa = AssetAllocation(
        returns=annual_returns.to_numpy(),
        covariance=covariance_matrix.to_numpy(),
        **CANDIDATE,
    )
    aas = aa.run(
        solver_type="SIMULATED",
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        write_report=False,
    )

    row = {
        "gamma": gamma,
        "run": run_idx,
        "caa_risk": float(
            get_risk(covariance=covariance_matrix, allocations=caa)
        ),
        "caa_return": float(
            get_returns(allocations=caa, returns=annual_returns)
        ),
        "caa_sharpe": get_sharpe_ratio(
            allocations=caa,
            returns=annual_returns,
            covariance=covariance_matrix,
        ),
        "aa_risk": float(
            get_risk(covariance=covariance_matrix, allocations=aas)
        ),
        "aa_return": float(
            get_returns(allocations=aas, returns=annual_returns)
        ),
        "aa_sharpe": get_sharpe_ratio(
            allocations=aas,
            returns=annual_returns,
            covariance=covariance_matrix,
        ),
        "l1_weight_distance": float(np.abs(caa - aas).sum()),
        "n_communities": caa_diag["n_communities_found"],
        "aa_raw_sum": aa.raw_weight_sum,
        "caa_upper_raw_sum": caa_diag["upper_raw_sum"],
        "caa_inner_raw_sum_min": float(
            np.asarray(caa_diag["inner_raw_sums"]).min()
        ),
    }
    # full weight vectors, one column per asset
    for i, w in enumerate(caa):
        row[f"caa_w_{i}"] = float(w)
    for i, w in enumerate(aas):
        row[f"aa_w_{i}"] = float(w)
    return row


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    price_cache = f"{git_root.git_root()}/data/prices_cache.parquet"
    daily_returns = pd.read_parquet(
        price_cache
    )  # must exist: same universe as sweeps
    print(f"Loaded cached prices from {price_cache}")

    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )
    graph_adjacency = get_correlation(
        daily_returns=daily_returns, zero_diagonal=True
    )

    tasks = [(g, r) for g in GAMMAS for r in range(RUNS_PER_GAMMA)]
    chunksize = max(1, len(tasks) // (WORKERS * 8))
    print(
        f"Configurations: {len(GAMMAS)} gammas x {RUNS_PER_GAMMA} runs = {len(tasks)} tasks"
    )

    start = perf_counter()
    rows = []
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=_init_worker,
        initargs=(
            daily_returns,
            annual_returns,
            covariance_matrix,
            graph_adjacency,
        ),
    ) as executor:
        for row in tqdm(
            executor.map(_one_task, tasks, chunksize=chunksize),
            total=len(tasks),
        ):
            rows.append(row)

    full = pd.DataFrame(rows)
    full.to_parquet(f"{OUT_DIR}/confirmation_runs.parquet")
    full.to_csv(f"{OUT_DIR}/confirmation_runs.csv", index=False)
    print(
        f"\nDone in {(perf_counter()-start)/60:.1f} min. Wrote {len(full)} rows to {OUT_DIR}"
    )

    # ---------------- quick summary ----------------
    asset_names = list(daily_returns.columns)
    print("\n=== per-gamma summary ===")
    for g, sub in full.groupby("gamma"):
        print(f"\ngamma = {g}")
        print(
            f"  merged-run fraction : {(sub['n_communities'] < 5).mean():.3f}"
        )
        print(
            f"  |d_return| mean     : {(sub['caa_return'] - sub['aa_return']).abs().mean():.4f}"
        )
        print(
            f"  |d_risk|   mean     : {(sub['caa_risk']   - sub['aa_risk']).abs().mean():.4f}"
        )
        print(
            f"  L1 weight distance  : mean {sub['l1_weight_distance'].mean():.3f}, "
            f"median {sub['l1_weight_distance'].median():.3f}, "
            f"90th pct {sub['l1_weight_distance'].quantile(0.9):.3f}   (max possible = 2)"
        )

    # classical-vs-classical baseline: how much do two INDEPENDENT classical runs
    # differ from each other? This is the noise floor the community model should be
    # judged against - matching classical more tightly than classical matches itself
    # is impossible.
    g0 = full[full["gamma"] == GAMMAS[0]]
    aa_w = g0[[f"aa_w_{i}" for i in range(len(asset_names))]].to_numpy()
    idx = np.random.default_rng(0).permutation(len(aa_w))
    self_l1 = np.abs(aa_w - aa_w[idx]).sum(axis=1)
    print(
        f"\nclassical self-distance (noise floor): mean {self_l1.mean():.3f}, "
        f"median {np.median(self_l1):.3f}"
    )
