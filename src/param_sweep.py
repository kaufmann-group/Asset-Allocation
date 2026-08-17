import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol

from main import community_asset_allocation
from modules import *

NUMBER_COMMUNITIES = 5
RUNS_PER_COMBO = 40
NUM_READS = 400
NUM_SWEEPS = 300
BITS_PER_ASSET = 6
DESIGN_POWER_STAGE_1 = 7
DESIGN_POWER_STAGE_2 = 7
BASE_SEED_STAGE_1 = 20260731
BASE_SEED_STAGE_2 = 20260732
NUM_CORES = os.cpu_count() or 1

ALPHA = 0.75
TOP_FRACTION = 0.10

STAGE_1_NAMES = ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]
STAGE_1_BOUNDS = np.array([[1.0, 20.0], [0.1, 5.0], [2.5, 30.0], [1.0, 80.0], [15.0, 100.0]])

GAMMA_STAGE_2 = 8.0
BETA_STAGE_2 = 1.0
STAGE_2_NAMES = ["lambda_1", "lambda_2", "lambda_3"]

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"
ASSET_FILE = ROOT / "benchmarking" / "benchmarking_assets.txt"
PRICE_CACHE = DATA_DIR / "param_sweep_prices.csv"
STAGE_1_FILE = DATA_DIR / "param_sweep_stage1.csv"
STAGE_1_SCORES_FILE = DATA_DIR / "param_sweep_stage1_scores.csv"
STAGE_2_FILE = DATA_DIR / "param_sweep_stage2.csv"
STAGE_2_SCORES_FILE = DATA_DIR / "param_sweep_stage2_scores.csv"

_SHARED = {}


def build_design(names, bounds, design_power, seed):
    unit_points = Sobol(d=len(names), scramble=True, seed=seed).random_base2(m=design_power)
    log_bounds = np.log10(bounds)
    points = 10 ** (log_bounds[:, 0] + unit_points * (log_bounds[:, 1] - log_bounds[:, 0]))
    return [dict(zip(names, point)) for point in points]


def choose_assets():
    asset_groups = parse_assets_file(ASSET_FILE)
    rng = np.random.default_rng(20240101)
    pick = lambda group: rng.choice(asset_groups[group], 5, replace=False).tolist()
    return pick("Technology & Semiconductors") + pick("Communication Services & Media") + pick("Consumer Discretionary & Retail") + pick("Industrials, Energy & Utilities")


def load_returns(assets):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if PRICE_CACHE.exists():
        daily_returns = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
        if list(daily_returns.columns) == assets:
            print(f"Using cached price data from {PRICE_CACHE}")
            return daily_returns

    daily_returns = closing_prices(assets=assets)
    daily_returns.to_csv(PRICE_CACHE)
    print(f"Saved price data to {PRICE_CACHE}")
    return daily_returns


def _init_worker(daily_returns, annual_returns, covariance_matrix, graph_adjacency, base_seed, gamma_fixed, beta_fixed):
    _SHARED["daily_returns"] = daily_returns
    _SHARED["annual_returns"] = annual_returns
    _SHARED["covariance_matrix"] = covariance_matrix
    _SHARED["graph_adjacency"] = graph_adjacency
    _SHARED["base_seed"] = base_seed
    _SHARED["gamma_fixed"] = gamma_fixed
    _SHARED["beta_fixed"] = beta_fixed
    Qubo.save_benchmark_report = lambda *args, **kwargs: None


def _run_one(job):
    combo_id, params, run = job
    seed = int(np.random.SeedSequence([_SHARED["base_seed"], combo_id, run]).generate_state(1)[0])
    np.random.seed(seed)
    random.seed(seed)

    daily_returns = _SHARED["daily_returns"]
    annual_returns = _SHARED["annual_returns"]
    covariance_matrix = _SHARED["covariance_matrix"]
    graph_adjacency = _SHARED["graph_adjacency"]

    gamma = params["gamma"] if "gamma" in params else _SHARED["gamma_fixed"]
    beta = params["beta"] if "beta" in params else _SHARED["beta_fixed"]

    caa = community_asset_allocation(daily_returns=daily_returns, number_communities=NUMBER_COMMUNITIES, gamma=gamma, beta=beta, lambda_1=params["lambda_1"], lambda_2=params["lambda_2"], lambda_3=params["lambda_3"], bits_per_asset=BITS_PER_ASSET, num_reads=NUM_READS, num_sweeps=NUM_SWEEPS, annual_returns=annual_returns, covariance_matrix=covariance_matrix, graph_adjacency=graph_adjacency)

    aa = AssetAllocation(returns=annual_returns.to_numpy(), covariance=covariance_matrix.to_numpy(), lambda_1=params["lambda_1"], lambda_2=params["lambda_2"], lambda_3=params["lambda_3"], bits_per_asset=BITS_PER_ASSET)
    aa_weights = aa.run(solver_type="SIMULATED", num_reads=NUM_READS, num_sweeps=NUM_SWEEPS)

    return {"combo_id": combo_id, "run": run, "gamma": gamma, "beta": beta, "lambda_1": params["lambda_1"], "lambda_2": params["lambda_2"], "lambda_3": params["lambda_3"], "caa_risk": float(get_risk(covariance_matrix, caa)), "caa_return": float(get_returns(caa, annual_returns)), "caa_sharpe": float(get_sharpe_ratio(caa, annual_returns, covariance_matrix)), "aa_risk": float(get_risk(covariance_matrix, aa_weights)), "aa_return": float(get_returns(aa_weights, annual_returns)), "aa_sharpe": float(get_sharpe_ratio(aa_weights, annual_returns, covariance_matrix))}


def run_sweep(name, design, output_file, daily_returns, annual_returns, covariance_matrix, graph_adjacency, base_seed, gamma_fixed=None, beta_fixed=None):
    tasks = [(combo_id, params, run) for combo_id, params in enumerate(design) for run in range(RUNS_PER_COMBO)]
    chunksize = max(1, len(tasks) // (NUM_CORES * 8))

    print(f"\n{name}")
    print(f"Combinations: {len(design)}")
    print(f"Runs per combination: {RUNS_PER_COMBO}")
    print(f"Total runs: {len(tasks)}")
    print(f"Cores: {NUM_CORES}")

    start = perf_counter()
    rows = []

    with ProcessPoolExecutor(max_workers=NUM_CORES, initializer=_init_worker, initargs=(daily_returns, annual_returns, covariance_matrix, graph_adjacency, base_seed, gamma_fixed, beta_fixed)) as executor:
        for i, row in enumerate(executor.map(_run_one, tasks, chunksize=chunksize), start=1):
            rows.append(row)
            if i % 100 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} runs finished")

    results = pd.DataFrame(rows)
    results.to_csv(output_file, index=False)
    print(f"{name} finished in {(perf_counter() - start) / 60:.1f} minutes")
    print(f"Results written to {output_file}")
    return results


def score_combinations(results):
    results = results.copy()

    for metric in ["risk", "return", "sharpe"]:
        results[f"d_{metric}"] = results[f"caa_{metric}"] - results[f"aa_{metric}"]

    metric_columns = ["caa_risk", "caa_return", "caa_sharpe", "aa_risk", "aa_return", "aa_sharpe"]
    results["degenerate"] = results[metric_columns].isna().any(axis=1)
    results["degenerate"] |= ((results["caa_risk"] == 0) & (results["caa_return"] == 0)) | ((results["aa_risk"] == 0) & (results["aa_return"] == 0))

    scales = {}
    for metric in ["risk", "return", "sharpe"]:
        q75, q25 = np.nanpercentile(results[f"aa_{metric}"], [75, 25])
        scales[metric] = q75 - q25
        if scales[metric] == 0 or not np.isfinite(scales[metric]):
            scales[metric] = 1.0

    rows = []
    for combo_id, group in results.groupby("combo_id"):
        good = group[~group["degenerate"]]
        row = {"combo_id": combo_id, "n_degenerate": int(group["degenerate"].sum())}
        score = 0.0

        for metric in ["risk", "return", "sharpe"]:
            mad = good[f"d_{metric}"].abs().mean()
            sd = good[f"d_{metric}"].std()
            row[f"mad_{metric}"] = mad
            row[f"sd_{metric}"] = sd
            score += (mad + ALPHA * sd) / scales[metric]

        row["score"] = score if len(good) >= 30 else np.inf

        for parameter in ["gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]:
            row[parameter] = group[parameter].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def refinement_bounds(scores):
    valid = scores[np.isfinite(scores["score"])]
    top_count = max(10, int(TOP_FRACTION * len(valid)))
    top = valid.head(top_count)

    bounds = []
    print("\nStage 2 lambda bounds:")

    for parameter in STAGE_2_NAMES:
        lo, hi = top[parameter].quantile(0.05), top[parameter].quantile(0.95)
        pad = 0.15 * (hi - lo)
        lower = max(lo - pad, valid[parameter].min())
        upper = min(hi + pad, valid[parameter].max())

        if lower >= upper:
            lower = valid[parameter].min()
            upper = valid[parameter].max()

        bounds.append([lower, upper])
        print(f"{parameter}: [{lower:.4g}, {upper:.4g}]")

    return np.array(bounds)


if __name__ == "__main__":
    assets = choose_assets()
    daily_returns = load_returns(assets)
    annual_returns = daily_returns.mean() * 252
    covariance_matrix = get_covariance(daily_returns=daily_returns, annualize=True)
    graph_adjacency = get_correlation(daily_returns=daily_returns, zero_diagonal=True)

    print(f"Assets: {len(assets)}")
    print(f"num_reads={NUM_READS}, num_sweeps={NUM_SWEEPS}")

    stage_1_design = build_design(STAGE_1_NAMES, STAGE_1_BOUNDS, DESIGN_POWER_STAGE_1, BASE_SEED_STAGE_1)
    stage_1_results = run_sweep("STAGE 1: coarse five-parameter sweep", stage_1_design, STAGE_1_FILE, daily_returns, annual_returns, covariance_matrix, graph_adjacency, BASE_SEED_STAGE_1)

    stage_1_scores = score_combinations(stage_1_results)
    stage_1_scores.to_csv(STAGE_1_SCORES_FILE, index=False)

    print("\nBest Stage 1 combinations:")
    print(stage_1_scores.head(10)[["combo_id", "score", "gamma", "beta", "lambda_1", "lambda_2", "lambda_3"]].to_string(index=False))

    stage_2_bounds = refinement_bounds(stage_1_scores)
    stage_2_design = build_design(STAGE_2_NAMES, stage_2_bounds, DESIGN_POWER_STAGE_2, BASE_SEED_STAGE_2)
    stage_2_results = run_sweep("STAGE 2: refined lambda sweep", stage_2_design, STAGE_2_FILE, daily_returns, annual_returns, covariance_matrix, graph_adjacency, BASE_SEED_STAGE_2, GAMMA_STAGE_2, BETA_STAGE_2)

    stage_2_scores = score_combinations(stage_2_results)
    stage_2_scores.to_csv(STAGE_2_SCORES_FILE, index=False)

    print("\nBest Stage 2 combinations:")
    print(stage_2_scores.head(10)[["combo_id", "score", "lambda_1", "lambda_2", "lambda_3"]].to_string(index=False))
    print(f"\nStage 2 used gamma={GAMMA_STAGE_2} and beta={BETA_STAGE_2}")
    print(f"Best final parameters: lambda_1={stage_2_scores.iloc[0]['lambda_1']:.6g}, lambda_2={stage_2_scores.iloc[0]['lambda_2']:.6g}, lambda_3={stage_2_scores.iloc[0]['lambda_3']:.6g}")
