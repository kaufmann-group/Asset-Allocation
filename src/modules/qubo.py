from abc import ABC, abstractmethod

import numpy as np

import dimod

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

from get_solvers import get_solvers

import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("API_TOKEN")

solver = "Advantage2_system1"
endpoint = "https://cloud.dwavesys.com/sapi"

class Qubo(ABC):
    def __init__(self):
        self.Q = None
        self.solution = None
        self.energy = None
        self.sampleset = None

    @abstractmethod
    def build_qubo(self):
        pass

    @abstractmethod
    def decode_solution(self, x):
        pass

    """
    calls dwave api and gets solution of qubo (both upper triangular and symmetric)
    objective function.
    """

    def solve(self, solver_type, num_reads=3000, **sample_kwargs):
        solver_type = solver_type.upper()

        n = self.Q.shape[0]

        Q_dict = {}

        for i in range(n):
            if self.Q[i, i] != 0:
                Q_dict[(i, i)] = float(self.Q[i, i])

            for j in range(i + 1, n):
                coeff = self.Q[i, j] + self.Q[j, i]
                if coeff != 0:
                    Q_dict[(i, j)] = float(coeff)

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q_dict)

        if solver_type == "EXACT":
            sampler = dimod.ExactSolver()
            sampleset = sampler.sample(bqm)

        elif solver_type == "SIMULATED":
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)

        elif solver_type in {"QPU", "HYBRID"}:
            solvers = get_solvers()

            solver_name = next((name for name, category in solvers if category == solver_type), None)

            if solver_name is None:
                raise ValueError(f"No solver found for solver_type='{solver_type}'.")

            if solver_type == "QPU":
                sampler = EmbeddingComposite(DWaveSampler(solver=solver_name, token=token, endpoint=endpoint))

                sampleset = sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)

            else:
                sampler = LeapHybridSampler(solver=solver_name, token=token, endpoint=endpoint)

                sampleset = sampler.sample(bqm,**sample_kwargs)
        else:
            raise ValueError("solver_type must be one of: 'SIMULATED', 'QPU', or 'HYBRID'.")

        sampleset = sampleset.aggregate()
        lowest = sampleset.lowest(rtol=0, atol=0).aggregate()

        best = max(lowest.data(["sample", "energy", "num_occurrences"]), key=lambda row: row.num_occurrences)

        x = np.array([best.sample[i] for i in range(n)], dtype=int)

        self.solution = x
        self.energy = best.energy
        self.sampleset = sampleset

        return x

    """
    get solution via annealer to qubo
    """
    def run(self, solver_type):
        self.Q = self.build_qubo()
        return self.decode_solution(self.solve(solver_type=solver_type))