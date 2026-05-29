import numpy as np
from .qubo import *

class AssetAllocation(Qubo):
    def __init__(self, returns, covariance, p=0.1, lambda_3=10, bits_per_asset=6):
        super().__init__()
        self.returns = np.array(returns)
        self.covariance = np.array(covariance)
        self.p = p
        self.lambda_3 = lambda_3
        self.lambda_1 = self.p ** -2 # convention in paper
        self.lambda_2 = 1.0 # convention in paper
        self.bits_per_asset = bits_per_asset

    """
    builds asset allocation upper triangular qubo
    """
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

                return_quad = self.lambda_1 * mu[i] * mu[j] * a_u * a_v
                budget_quad = self.lambda_2 * a_u * a_v
                variance_quad = self.lambda_3 * C[i, j] * a_u * a_v

                coeff = return_quad + budget_quad + variance_quad

                if u == v:
                    return_lin = -2.0 * self.lambda_1 * self.p * mu[i] * a_u
                    budget_lin = -2.0 * self.lambda_2 * a_u

                    Q[u, u] = coeff + return_lin + budget_lin
                else:
                    Q[u, v] = 2.0 * coeff

        return Q
    
    """
    decodes solution with one hot scheme
    """
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