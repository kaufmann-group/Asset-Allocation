import numpy as np
import dwave.graphs as dg
from minorminer import busclique
from .qubo import *

class AssetAllocation(Qubo):
    """
    Takes returns and covariance to build a quadratic unconstrained binary optimization (QUBO)
    matrix for mean-variance portfolio allocation.
    """
    def __init__(self, returns, covariance, lambda_1=1.0, lambda_2=50.0, lambda_3=10.0, bits_per_asset=6):
        super().__init__()
        self.returns = np.array(returns)
        self.covariance = np.array(covariance)
        self.lambda_1 = lambda_1  # Return reward strength
        self.lambda_2 = lambda_2  # Budget constraint strength
        self.lambda_3 = lambda_3  # Risk penalty strength
        self.bits_per_asset = bits_per_asset

    def build_qubo(self):  
        mu = self.returns
        C = self.covariance

        n = len(mu)
        k = self.bits_per_asset
        size = n * k

        Q = np.zeros((size, size))

        asset_idx = np.zeros(size, dtype=int)
        bit_weights = np.zeros(size)

        for u in range(size):
            asset_idx[u] = u // k
            bit_power = (u % k) + 1
            bit_weights[u] = 2.0 ** (-bit_power)

        for u in range(size):
            i = asset_idx[u]
            a_u = bit_weights[u]

            for v in range(u, size):
                j = asset_idx[v]
                a_v = bit_weights[v]

                variance_quad = self.lambda_3 * C[i, j] * a_u * a_v
                budget_quad = self.lambda_2 * a_u * a_v

                coeff = variance_quad + budget_quad

                if u == v:
                    return_lin = -self.lambda_1 * mu[i] * a_u
                    budget_lin = -2.0 * self.lambda_2 * a_u
                    Q[u, u] = coeff + return_lin + budget_lin
                else:
                    Q[u, v] = 2.0 * coeff

        return Q
    
    def get_minor_embedding(self, target_graph=None):
        """
        Calculates a deterministic clique embedding (K_N) for the fully connected QUBO graph.
        """
        Q = self.build_qubo()
        size = len(Q)  # Total logical variables in the clique K_N

        # Default to Pegasus graph (P16) topology if no target graph is provided
        if target_graph is None:
            target_graph = dg.pegasus_graph(16)

        # Polynomial-time clique embedding via busclique
        embedding = busclique.find_clique_embedding(size, target_graph)

        if not embedding:
            print(f"Failed to find a clique embedding for K_{size} on the target graph.")
            return None

        return embedding

    def decode_solution(self, x):
        n = len(self.returns)
        allocations = []

        for i in range(n):
            bits = x[i * self.bits_per_asset : (i + 1) * self.bits_per_asset]
            value = sum(bit * 2 ** (-j) for j, bit in enumerate(bits, start=1))
            allocations.append(value)

        allocations = np.array(allocations)

        if allocations.sum() != 0:
            allocations = allocations / allocations.sum()

        return allocations