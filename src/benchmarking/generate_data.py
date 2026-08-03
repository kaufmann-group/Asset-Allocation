# Keep BLAS single-threaded per worker BEFORE numpy is imported, so that running
# many worker processes does not oversubscribe the cores with nested threads.
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
from scipy.stats.qmc import Sobol

import git_root

sys.path.append("..")

from modules import *
from main import community_asset_allocation

"""
STAGE 1 - DATA GENERATION

Sweeps the five model parameters (gamma, beta, lambda_1, lambda_2, lambda_3) over a
space-filling (Sobol) design and, for every combination, runs RUNS_PER_COMBO repeats of
both the community model and the classical baseline. It writes ONE CSV per combination,
each holding the raw per-run rows (community and classical side by side) so that stage 2
can apply any scoring formula later without regenerating anything.

All work is spread across every CPU core as a single flat pool of (combination x run) tasks.

STAGE 1b - REFINEMENT SWEEP

Second-pass data generation inside the region identified by the coarse sweep:

  * gamma and beta are FIXED (coarse sweep showed them irrelevant to closeness:
    ~4% GBM importance each, flat partial dependence, no top-decile concentration).
  * Only the three lambdas are swept, over the zoomed/extended bounds, so the same
    budget covers a 3D box far more densely than the 5D coarse pass.
  * Lambdas are sampled in LOG10 space - penalty strengths act multiplicatively,
    and the coarse-pass PDPs were natural in log units.
  * Raw (pre-normalization) weight sums are recorded for every solve, to test
    whether the low-lambda_2 closeness region is genuine or an artifact of
    renormalization masking large budget violations.
  * Fresh BASE_SEED so the region is confirmed with annealer noise independent
    of the coarse sweep (this pass doubles as a validation set).

Requires: AssetAllocation.decode_solution setting self.raw_weight_sum, and
community_asset_allocation(collect_diagnostics=True) returning (weights, diag).
"""

# =====================================================================================
# CONFIGURATION - the three cost knobs plus the design size
# =====================================================================================
NUMBER_COMMUNITIES = 5

# Fixed community-detection parameters (see docstring for evidence).
GAMMA_FIXED = 8.0
BETA_FIXED = 1.0

# Runs per combination. Was 250 for the plot; ~20-40 is plenty to estimate per-combo stats.
RUNS_PER_COMBO = 40

# Annealing fidelity. THESE VALUES ARE BAKED INTO THE GENERATED DATA - choose deliberately.
# Profile the reads-vs-quality knee on one combination before a large run.
NUM_READS = 400
NUM_SWEEPS = 300

# Sobol design size. Number of parameter combinations = 2 ** DESIGN_POWER.
#   DESIGN_POWER = 7 -> 128 combinations
#   DESIGN_POWER = 8 -> 256 combinations   (default)
#   DESIGN_POWER = 9 -> 512 combinations
# Each combination -> one CSV file. Sobol is only perfectly balanced at powers of two.
DESIGN_POWER = 7

# Classical baseline uses the SAME lambdas as the community model, so a win reflects the
# community structure rather than lambda tuning. Set False to hold the baseline at defaults.
TIE_CLASSICAL_LAMBDAS = True

BASE_SEED = 20260731
WORKERS = os.cpu_count()
OUT_DIR = f"{git_root.git_root()}/data/sweep_refine"

# Five-parameter search space:   gamma        beta        lambda_1      lambda_2       lambda_3
# PARAM_NAMES = ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]
PARAM_NAMES = ["lambda_1", "lambda_2", "lambda_3"]
PARAM_BOUNDS = np.array(
    [
        #        [1.0, 20.0],  # gamma    - one-hot softness in community detection
        #        [0.1, 5.0],  # beta     - modularity weight in community detection
        [2.5, 30.0],  # lambda_1 - return reward strength
        [1.0, 80.0],  # lambda_2 - budget constraint strength
        [15.0, 100.0],  # lambda_3 - risk penalty strength
    ]
)
LOG_BOUNDS = np.log10(PARAM_BOUNDS)


def build_design():
    """2**DESIGN_POWER space-filling points, scaled from the unit cube into the ranges."""
    unit = Sobol(
        d=len(PARAM_NAMES), scramble=True, seed=BASE_SEED
    ).random_base2(m=DESIGN_POWER)
    log_scaled = LOG_BOUNDS[:, 0] + unit * (LOG_BOUNDS[:, 1] - LOG_BOUNDS[:, 0])
    scaled = 10.0**log_scaled
    return [dict(zip(PARAM_NAMES, row)) for row in scaled]


# =====================================================================================
# WORKER SIDE - top-level so child processes can import/pickle it (Windows uses spawn)
# =====================================================================================
_SHARED = {}


def _init_worker(
    daily_returns, annual_returns, covariance_matrix, graph_adjacency
):
    """Runs once per worker process. The read-only, deterministic inputs are transferred
    a single time per core instead of being re-pickled for every task."""
    _SHARED["daily_returns"] = daily_returns
    _SHARED["annual_returns"] = annual_returns
    _SHARED["covariance_matrix"] = covariance_matrix
    _SHARED["graph_adjacency"] = graph_adjacency


def _one_task(job):
    """One (combination, run) evaluation: community model + classical baseline on one seed."""
    combo_id, params, run_idx = job

    # Independent, reproducible RNG per (combo, run). Seed both numpy's legacy global RNG
    # and stdlib random, since the downstream annealing routines rely on the globals.
    seed = int(
        np.random.SeedSequence([BASE_SEED, combo_id, run_idx]).generate_state(
            1
        )[0]
    )
    np.random.seed(seed)
    random.seed(seed)

    daily_returns = _SHARED["daily_returns"]
    annual_returns = _SHARED["annual_returns"]
    covariance_matrix = _SHARED["covariance_matrix"]
    graph_adjacency = _SHARED["graph_adjacency"]

    # ---- community-based asset allocation (our model) ----
    caa, caa_diag = community_asset_allocation(
        daily_returns=daily_returns,
        number_communities=NUMBER_COMMUNITIES,
        gamma=GAMMA_FIXED,
        beta=BETA_FIXED,
        lambda_1=params["lambda_1"],
        lambda_2=params["lambda_2"],
        lambda_3=params["lambda_3"],
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        write_report=False,
        annual_returns=annual_returns,
        covariance_matrix=covariance_matrix,
        graph_adjacency=graph_adjacency,
        collect_diagnostics=True,
    )

    # ---- classical asset allocation (baseline) ----
    lam = (
        dict(
            lambda_1=params["lambda_1"],
            lambda_2=params["lambda_2"],
            lambda_3=params["lambda_3"],
        )
        if TIE_CLASSICAL_LAMBDAS
        else {}
    )
    aa = AssetAllocation(
        returns=annual_returns.to_numpy(),
        covariance=covariance_matrix.to_numpy(),
        **lam,
    )
    aas = aa.run(
        solver_type="SIMULATED",
        num_reads=NUM_READS,
        num_sweeps=NUM_SWEEPS,
        write_report=False,
    )

    inner_sums = np.asarray(caa_diag["inner_raw_sums"], dtype=float)

    row = {
        "combo_id": combo_id,
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
        # ---- budget-violation diagnostics (pre-normalization weight sums) ----
        # A healthy solve has raw sums near 1.0; large deviations mean the
        # renormalization in decode_solution is doing the work, and closeness
        # in that regime should be treated with suspicion.
        "aa_raw_sum": aa.raw_weight_sum,
        "caa_upper_raw_sum": caa_diag["upper_raw_sum"],
        "caa_inner_raw_sum_min": float(inner_sums.min()),
        "caa_inner_raw_sum_mean": float(inner_sums.mean()),
        "caa_inner_raw_sum_max": float(inner_sums.max()),
        "caa_n_communities": caa_diag["n_communities_found"],
    }
    row.update(params)
    return row


# =====================================================================================
# MAIN
# =====================================================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- fixed asset universe + price data: fetched ONCE, then cached to disk ----
    price_cache = f"{git_root.git_root()}/data/prices_cache.parquet"
    if os.path.exists(price_cache):
        daily_returns = pd.read_parquet(price_cache)
        print(f"Loaded cached prices from {price_cache}")
    else:
        assets_dict = parse_assets_file("benchmarking_assets.txt")
        rng = np.random.default_rng(
            20240101  # deliberately the OLD seed: keep the identical universe
        )  # fixed => same universe every run
        pick = lambda area, n: rng.choice(
            assets_dict[area], n, replace=False
        ).tolist()
        assets = (
            pick("Technology & Semiconductors", 5)
            + pick("Communication Services & Media", 5)
            + pick("Consumer Discretionary & Retail", 5)
            + pick("Industrials, Energy & Utilities", 5)
        )
        daily_returns = closing_prices(assets=assets)
        daily_returns.to_parquet(price_cache)
        print(f"Fetched prices and cached to {price_cache}")

    # deterministic quantities, computed once and shared with every worker
    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )
    graph_adjacency = get_correlation(
        daily_returns=daily_returns, zero_diagonal=True
    )

    # ---- build the space-filling design and record it ----
    design = build_design()

    combos_index = pd.DataFrame(
        [{"combo_id": i, **p} for i, p in enumerate(design)]
    )
    combos_index["gamma"] = GAMMA_FIXED
    combos_index["beta"] = BETA_FIXED
    combos_index.to_csv(f"{OUT_DIR}/combos_index.csv", index=False)

    tasks = [
        (cid, params, run_idx)
        for cid, params in enumerate(design)
        for run_idx in range(RUNS_PER_COMBO)
    ]
    chunksize = max(1, len(tasks) // (WORKERS * 8))

    print(
        f"Combinations: {len(design)}  |  runs/combo: {RUNS_PER_COMBO}  |  total tasks: {len(tasks)}"
    )
    print(
        f"Fixed: gamma={GAMMA_FIXED}, beta={BETA_FIXED}  |   "
        f"Fidelity: num_reads={NUM_READS}, num_sweeps={NUM_SWEEPS}  |  workers: {WORKERS}"
    )

    # ---- run everything across all cores as one flat pool ----
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
    elapsed = perf_counter() - start

    # ---- write ONE CSV per parameter combination (raw per-run paired rows) ----
    full = pd.DataFrame(rows)
    column_order = (
        ["combo_id", "run"]
        + PARAM_NAMES
        + [
            "caa_risk",
            "caa_return",
            "caa_sharpe",
            "aa_risk",
            "aa_return",
            "aa_sharpe",
            "aa_raw_sum",
            "caa_upper_raw_sum",
            "caa_inner_raw_sum_min",
            "caa_inner_raw_sum_mean",
            "caa_inner_raw_sum_max",
            "caa_n_communities",
        ]
    )
    full = full[column_order]

    for cid, sub in full.groupby("combo_id"):
        sub.sort_values("run").to_csv(
            f"{OUT_DIR}/combo_{cid:04d}.csv", index=False
        )

    # convenience: everything in one long-format file too (handy for stage 2)
    full.to_parquet(f"{OUT_DIR}/all_runs_long.parquet")

    print(f"\nDone in {elapsed/60:.1f} min.")
    print(
        f"Wrote {full['combo_id'].nunique()} combination CSVs + combos_index.csv "
        f"+ all_runs_long.parquet to {OUT_DIR}"
    )


"""
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

import sys
import git_root

sys.path.append("..")

from modules import *
from main import community_asset_allocation


# ---------------------------------------------------------------------------
# worker-side code: must be top-level so child processes can import/pickle it
# ---------------------------------------------------------------------------

# read-only data shared by every run, populated once per process by _init_worker
_SHARED = {}


def _init_worker(
    daily_returns, returns, cov_matrix, number_communities, base_seed
):

    _SHARED["daily_returns"] = daily_returns
    _SHARED["returns"] = returns
    _SHARED["cov_matrix"] = cov_matrix
    _SHARED["number_communities"] = number_communities
    _SHARED["base_seed"] = base_seed


def _single_run(run_index, beta, gamma, lambda_1, lambda_2, lambda_3):
    # Independent + reproducible RNG per run. Seed BOTH numpy's legacy global
    # RNG and stdlib random, since the downstream routines rely on the globals.
    seed = int(
        np.random.SeedSequence(
            [_SHARED["base_seed"], run_index]
        ).generate_state(1)[0]
    )
    np.random.seed(seed)
    random.seed(seed)

    daily_returns = _SHARED["daily_returns"]
    returns = _SHARED["returns"]
    cov_matrix = _SHARED["cov_matrix"]
    number_communities = _SHARED["number_communities"]

    # community asset allocation
    caa = community_asset_allocation(
        daily_returns=daily_returns,
        number_communities=number_communities,
        adjacency=get_correlation,
        gam=gamma,
        bet=beta,
        lam_1=lambda_1,
        lam_2=lambda_2,
        lam_3=lambda_3,
    )
    caa_risk = get_risk(covariance=cov_matrix, allocations=caa)
    caa_return = get_returns(allocations=caa, returns=returns)

    # classical asset allocation (simulated-annealing solver)
    aa = AssetAllocation(
        returns=returns.to_numpy(), covariance=cov_matrix.to_numpy()
    )
    aas = aa.run(solver_type="SIMULATED")
    aa_risk = get_risk(covariance=cov_matrix, allocations=aas)
    aa_return = get_returns(allocations=aas, returns=returns)

    # sharpe ratios
    caa_sharpe = get_sharpe_ratio(
        allocations=caa, returns=returns, covariance=cov_matrix
    )
    aa_sharpe = get_sharpe_ratio(
        allocations=aas, returns=returns, covariance=cov_matrix
    )

    

    return caa_risk, caa_return, aa_risk, aa_return, caa_sharpe, aa_sharpe


if __name__ == "__main__":

    number_runs = 50
    number_communities = 5
    workers = os.cpu_count()  # use every logical core; tune down if desired
    base_seed = 12345  # fixed => reproducible; change for fresh draws

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

    gams = [4.0, 8.0, 16.0]
    bets = [0.5, 1.0, 2.0]
    lams1 = [0.5, 1.0, 2.0]
    lams2 = [35.0, 50.0, 100.0]
    lams3 = [5.0, 10.0, 20.0]


    daily_returns = closing_prices(assets=assets)
    returns = daily_returns.mean() * 252  # returns
    cov_matrix = get_covariance(
        daily_returns=daily_returns, annualize=True
    )  # covariance matrix



    tasks = []
    for g in gams:
        for b in bets:
            for l1 in lam1:
                for l2 in lam2:
                    for l3 in lam3:
                        for run_index in range(number_runs):
                            # order MUST match single_run's signature
                            tasks.append((run_index, beta, gam, lam1, lam2, lam3))

    results = [None] * len(tasks)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(daily_returns, returns, cov_matrix, number_communities, base_seed),
    ) as executor:
        futures = [executor.submit(_single_run, *args) for args in tasks]
        for i, fut in enumerate(tqdm(futures, total=len(futures))):
            results[i] = fut.result()


    # unpack the flat result tuples back into the original lists
    caa_risk, caa_return, aa_risk, aa_return, caa_sharpe, aa_sharpe = map(
        list, zip(*results)
    )

    caa_rar = list(zip(caa_risk, caa_return))  # community risk/return pairs
    aa_rar = list(zip(aa_risk, aa_return))  # classical risk/return pairs
    sharpe_ratios = list(
        zip(caa_sharpe, aa_sharpe)
    )  # (community, classical) sharpe pairs


    figure_1, axes = plt.subplots(1, 2, figsize=(12, 5))
    figure_1.suptitle(
        "Asset Allocation vs Community Based Asset Allocation",
        fontsize=14,
        y=1.00,
    )

    axes[0].plot(*zip(*aa_rar), "bs", markersize=5, label="Asset Allocation")
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
    axes[1].set_title("Sharpe Ratios Comparison", fontsize=12)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(
        f"{git_root.git_root()}/data/risk_and_returns_benchmark-{run_count}.png", dpi=300
    )
    plt.close()
    #print("Graph done, going to writing now")


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
        f"{git_root.git_root()}/data/risk_and_returns_benchmark-{run_count}.csv",
        index=False,
    )

    run_count = 0
    for g in gams:
        for b in bets:
            for l1 in lams1:
                for l2 in lams2:
                    for l3 in lams3:
                        run_count = run_count + 1
                            # ---- run all 50 iterations across every core -------------------------
                            results = [None] * number_runs
                            with ProcessPoolExecutor(
                                max_workers=workers,
                                initializer=_init_worker,
                                initargs=(
                                    daily_returns,
                                    returns,
                                    cov_matrix,
                                    number_communities,
                                    base_seed,
                                ),
                            ) as executor:
                                # executor.map preserves input order, so results stay aligned to run index
                                for i, res in enumerate(
                                    tqdm(
                                        executor.map(_single_run, range(number_runs)), total=number_runs
                                    )
                                ):
                                    results[i] = res

                            # unpack the flat result tuples back into the original lists
                            caa_risk, caa_return, aa_risk, aa_return, caa_sharpe, aa_sharpe = map(
                                list, zip(*results)
                            )

                            caa_rar = list(zip(caa_risk, caa_return))  # community risk/return pairs
                            aa_rar = list(zip(aa_risk, aa_return))  # classical risk/return pairs
                            sharpe_ratios = list(
                                zip(caa_sharpe, aa_sharpe)
                            )  # (community, classical) sharpe pairs


                            figure_1, axes = plt.subplots(1, 2, figsize=(12, 5))
                            figure_1.suptitle(
                                "Asset Allocation vs Community Based Asset Allocation",
                                fontsize=14,
                                y=1.00,
                            )

                            axes[0].plot(*zip(*aa_rar), "bs", markersize=5, label="Asset Allocation")
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
                            axes[1].set_title("Sharpe Ratios Comparison", fontsize=12)
                            axes[1].legend(frameon=True, facecolor="white", edgecolor="none")
                            axes[1].grid(True, linestyle="--", alpha=0.6)

                            plt.tight_layout()
                            plt.savefig(
                                f"{git_root.git_root()}/data/risk_and_returns_benchmark-{run_count}.png", dpi=300
                            )
                            plt.close()
                            #print("Graph done, going to writing now")


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
                                f"{git_root.git_root()}/data/risk_and_returns_benchmark-{run_count}.csv",
                                index=False,
                            )

"""
