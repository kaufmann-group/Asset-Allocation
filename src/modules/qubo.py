from abc import ABC, abstractmethod

import numpy as np

import dimod
import neal

from dwave.cloud import Client
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler

import os
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

        cls.solvers_available = [(s.name, s.properties.get("category", "unknown").upper()) for s in solvers]

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
            solver_name = next((name for name, category in solvers if category == solver_type), None)

            if solver_name is None:
                raise ValueError(f"No solver found for solver_type='{solver_type}'.")

            if solver_type == "QPU":
                sampler = EmbeddingComposite(DWaveSampler(solver=solver_name, token=token, endpoint=endpoint))
            else:
                sampler = LeapHybridSampler(solver=solver_name, token=token, endpoint=endpoint)

        else:
            raise ValueError("solver_type must be 'EXACT', 'SIMULATED', 'QPU', or 'HYBRID'.")

        cls.sampler_cache[solver_type] = sampler
        return sampler

    """
    calls dwave api and gets solution of qubo (both upper triangular and symmetric)
    objective function.
    """

    def solve(self, solver_type, num_reads=3000, **sample_kwargs):
        solver_type = solver_type.upper()
        sampler = self.get_sampler(solver_type)

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
            sampleset = sampler.sample(bqm)

        elif solver_type == "SIMULATED":
            sampleset = sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)
        
        elif solver_type == "QPU":
            sampleset = sampler.sample(bqm, num_reads=num_reads, **sample_kwargs)
        
        elif solver_type == "HYBRID":
            sampleset = sampler.sample(bqm,**sample_kwargs)

        else:
            raise ValueError("solver_type must be 'EXACT', 'SIMULATED', 'QPU', or 'HYBRID'.")

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
    def run(self, solver_type, **sample_kwargs):
        self.Q = self.build_qubo()
        x = self.solve(solver_type=solver_type, **sample_kwargs)
        return self.decode_solution(x)