import numpy as np
from .qubo import *


class AssetAllocation(Qubo):
    """
    takes returns and covariance
    """

    def __init__(
        self,
        returns,
        covariance,
        lambda_1=1.0,
        lambda_2=50.0,
        lambda_3=10.0,
        bits_per_asset=6,
    ):
        super().__init__()
        self.returns = np.array(returns)
        self.covariance = np.array(covariance)
        self.lambda_1 = lambda_1  # return reward strength - makes the optimizer chase return more aggressively
        self.lambda_2 = lambda_2  # budget constraint strength - forces the raw decoded weights to sum close to 1
        self.lambda_3 = lambda_3  # risk penalty strength - makes the optimizer avoid risky or high variance/covariance allocations
        self.bits_per_asset = bits_per_asset
        self.raw_weight_sum = (
            None  # set by decode_solution; None = not yet solved
        )

    """
    builds asset allocation upper triangular qubo
    """

    def build_qubo(self):
        mu = self.returns
        C = self.covariance

        n = len(mu)
        k = self.bits_per_asset
        size = n * k

        # Vectorized construction. Exactly equivalent to the original double loop,
        # but built with array operations instead of a size x size Python loop.
        asset_idx = np.repeat(np.arange(n), k)  # bit -> which asset
        bit_weights = 2.0 ** -(
            np.tile(np.arange(1, k + 1), n)
        )  # fixed-point weight per bit

        # C[asset(u), asset(v)] expanded to the full bit-by-bit grid
        C_expanded = C[np.ix_(asset_idx, asset_idx)]
        outer = np.outer(bit_weights, bit_weights)

        # symmetric quadratic coefficient: risk penalty (lambda_3) + budget quadratic (lambda_2)
        coeff = (self.lambda_3 * C_expanded + self.lambda_2) * outer

        # off-diagonal entries: doubled and kept strictly upper-triangular (matches original)
        Q = 2.0 * np.triu(coeff, k=1)

        # diagonal: quadratic self-term + return reward (linear) + budget linear term
        diagonal = (
            np.diag(coeff)
            - self.lambda_1 * mu[asset_idx] * bit_weights
            - 2.0 * self.lambda_2 * bit_weights
        )
        np.fill_diagonal(Q, diagonal)

        return Q

    """
    decodes solution with fixed point binary encoding
    """

    def decode_solution(self, x):
        n = len(self.returns)
        allocations = []

        for i in range(n):
            bits = x[i * self.bits_per_asset : (i + 1) * self.bits_per_asset]

            value = sum(bit * 2 ** (-j) for j, bit in enumerate(bits, start=1))

            allocations.append(value)

        allocations = np.array(allocations)

        self.raw_weight_sum = float(
            allocations.sum()
        )  # <-- ADD: pre-normalization budget

        if allocations.sum() != 0:
            allocations = allocations / allocations.sum()

        return allocations
