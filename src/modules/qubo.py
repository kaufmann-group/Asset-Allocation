from abc import ABC, abstractmethod

import numpy as np
from time import perf_counter

import dimod
import neal

from dwave.cloud import Client
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

import os
import git_root
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("API_TOKEN")

endpoint = "https://cloud.dwavesys.com/sapi"


class Qubo(ABC):
    solvers_available = None
    sampler_cache = {}

    def __init__(self):
        self.Q = None
        self.solution = None
        self.energy = None
        self.sampleset = None

        self.benchmark_metrics = {}

    @abstractmethod
    def build_qubo(self):
        pass

    @abstractmethod
    def decode_solution(self, x):
        pass

    @classmethod
    def get_solvers(cls):
        if cls.solvers_available is not None:
            return cls.solvers_available

        with Client.from_config(token=token) as client:
            solvers = client.get_solvers()

        cls.solvers_available = [
            (s.name, s.properties.get("category", "unknown").upper())
            for s in solvers
        ]

        return cls.solvers_available

    @classmethod
    def get_sampler(cls, solver_type):
        solver_type = solver_type.upper()

        if solver_type in cls.sampler_cache:
            return cls.sampler_cache[solver_type]

        if solver_type == "EXACT":
            sampler = dimod.ExactSolver()

        elif solver_type == "SIMULATED":
            sampler = neal.SimulatedAnnealingSampler()

        elif solver_type in {"QPU", "HYBRID"}:
            solvers = cls.get_solvers()
            solver_name = next(
                (name for name, category in solvers if category == solver_type),
                None,
            )

            if solver_name is None:
                raise ValueError(
                    f"No solver found for solver_type='{solver_type}'."
                )

            if solver_type == "QPU":
                sampler = EmbeddingComposite(
                    DWaveSampler(
                        solver=solver_name, token=token, endpoint=endpoint
                    )
                )
            else:
                sampler = LeapHybridSampler(
                    solver=solver_name, token=token, endpoint=endpoint
                )

        else:
            raise ValueError(
                "solver_type must be 'EXACT', 'SIMULATED', 'QPU', or 'HYBRID'."
            )

        cls.sampler_cache[solver_type] = sampler
        return sampler

    """
    calls dwave api and gets solution of qubo (both upper triangular and symmetric)
    objective function.
    """

    def solve(
        self, solver_type, num_reads=3000, write_report=True, **sample_kwargs
    ):
        solver_type = solver_type.upper()
        sampler = self.get_sampler(solver_type)

        n = self.Q.shape[0]
        Q_dict = {}

        for i in range(n):
            if self.Q[i, i] != 0:
                Q_dict[(i, i)] = float(self.Q[i, i])

            for j in range(i + 1, n):
                coefficient = self.Q[i, j] + self.Q[j, i]

                if coefficient != 0:
                    Q_dict[(i, j)] = float(coefficient)

        build_start = perf_counter()
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)
        bqm_build_time = perf_counter() - build_start

        possible_edges = bqm.num_variables * (bqm.num_variables - 1) / 2

        metrics = {
            "problem": {
                "logical_variables": bqm.num_variables,
                "logical_interactions": bqm.num_interactions,
                "logical_graph_density": (
                    bqm.num_interactions / possible_edges
                    if possible_edges > 0
                    else 0.0
                ),
            },
            "solver": {
                "solver_type": solver_type,
                "num_reads_requested": (
                    num_reads if solver_type in {"SIMULATED", "QPU"} else None
                ),
                "sample_kwargs": dict(sample_kwargs),
            },
            "timing": {"bqm_build_time_seconds": bqm_build_time},
            "embedding": {},
            "solution": {},
        }

        sample_start = perf_counter()

        if solver_type == "EXACT":
            sampleset = sampler.sample(bqm)

        elif solver_type == "SIMULATED":
            sampleset = sampler.sample(
                bqm, num_reads=num_reads, **sample_kwargs
            )

        elif solver_type == "QPU":
            sampleset = sampler.sample(
                bqm,
                num_reads=num_reads,
                return_embedding=True,
                chain_break_fraction=True,
                **sample_kwargs,
            )

        elif solver_type == "HYBRID":
            sampleset = sampler.sample(bqm, **sample_kwargs)

        else:
            raise ValueError(
                "solver_type must be 'EXACT', 'SIMULATED', 'QPU', or 'HYBRID'"
            )

        metrics["timing"]["sampling_wall_time_seconds"] = (
            perf_counter() - sample_start
        )

        # raw info before aggregation
        raw_info = dict(sampleset.info)
        occurrences = np.asarray(sampleset.record.num_occurrences, dtype=int)
        energies = np.asarray(sampleset.record.energy, dtype=float)

        minimum_energy = float(energies.min())
        best_mask = np.isclose(energies, minimum_energy, rtol=0, atol=1e-10)

        total_reads = int(occurrences.sum())
        weighted_mean_energy = float(np.average(energies, weights=occurrences))

        metrics["solution"].update(
            {
                "total_reads_returned": total_reads,
                "unique_solutions_before_aggregation": len(sampleset),
                "best_energy": minimum_energy,
                "mean_energy": weighted_mean_energy,
                "energy_std": float(
                    np.sqrt(
                        np.average(
                            (energies - weighted_mean_energy) ** 2,
                            weights=occurrences,
                        )
                    )
                ),
                "worst_energy": float(energies.max()),
                "best_solution_probability": float(
                    occurrences[best_mask].sum() / total_reads
                ),
            }
        )

        metrics["timing"]["solver_reported"] = raw_info.get("timing", {})

        if solver_type == "QPU":
            context = raw_info.get("embedding_context", {})
            embedding = context.get("embedding", {})
            lengths = np.asarray(
                [len(chain) for chain in embedding.values()], dtype=float
            )

            metrics["embedding"].update(
                {
                    "physical_qubits": len(
                        {
                            qubit
                            for chain in embedding.values()
                            for qubit in chain
                        }
                    ),
                    "mean_chain_length": (
                        float(lengths.mean()) if lengths.size else None
                    ),
                    "std_chain_length": (
                        float(lengths.std()) if lengths.size else None
                    ),
                    "min_chain_length": (
                        int(lengths.min()) if lengths.size else None
                    ),
                    "max_chain_length": (
                        int(lengths.max()) if lengths.size else None
                    ),
                    "chain_strength": context.get("chain_strength"),
                    "embedding_timing": context.get("timing", {}),
                }
            )

            if "chain_break_fraction" in sampleset.record.dtype.names:
                break_fractions = np.asarray(
                    sampleset.record.chain_break_fraction, dtype=float
                )

                metrics["embedding"].update(
                    {
                        "mean_chain_break_fraction": float(
                            np.average(break_fractions, weights=occurrences)
                        ),
                        "max_chain_break_fraction": float(
                            break_fractions.max()
                        ),
                        "samples_with_any_break_fraction": float(
                            occurrences[break_fractions > 0].sum() / total_reads
                        ),
                        "samples_over_10_percent_breaks": float(
                            occurrences[break_fractions > 0.10].sum()
                            / total_reads
                        ),
                    }
                )

        aggregated = sampleset.aggregate()
        lowest = aggregated.lowest(rtol=0, atol=1e-10).aggregate()

        best = max(
            lowest.data(["sample", "energy", "num_occurrences"]),
            key=lambda row: row.num_occurrences,
        )

        x = np.asarray(
            [best.sample[variable] for variable in range(n)], dtype=int
        )

        self.solution = x
        self.energy = float(best.energy)
        self.sampleset = aggregated
        self.benchmark_metrics = metrics

        self.save_benchmark_report()

        if write_report:
            self.save_benchmark_report()

        return x

    """
    get solution via annealer to qubo
    """

    def run(self, solver_type, **sample_kwargs):
        self.Q = self.build_qubo()
        x = self.solve(solver_type=solver_type, **sample_kwargs)
        return self.decode_solution(x)

    """
    Save benchmark metrics from the most recent run.
    """

    def save_benchmark_report(
        self,
        filepath=f"{git_root.git_root()}/src/benchmark-reports/benchmark_report.txt",
    ):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as file:
            file.write("QUBO BENCHMARK REPORT\n")
            file.write("=" * 50 + "\n\n")

            for section, metrics in self.benchmark_metrics.items():
                if not metrics:
                    continue

                file.write(section.upper() + "\n")
                file.write("-" * len(section) + "\n")

                for name, value in metrics.items():
                    label = name.replace("_", " ").title()

                    if isinstance(value, dict):
                        file.write(f"{label}:\n")

                        for sub_name, sub_value in value.items():
                            sub_label = sub_name.replace("_", " ").title()
                            file.write(f"  {sub_label}: {sub_value}\n")
                    else:
                        file.write(f"{label}: {value}\n")

                file.write("\n")
