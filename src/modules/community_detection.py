import numpy as np
from .qubo import *

class CommunityDetection(Qubo):
    def __init__(self, adjacency_matrix, number_communities, gamma=8.0, beta=1.0):
        super().__init__()
        self.adjacency_matrix = np.asarray(adjacency_matrix)
        self.number_communities = number_communities 
        self.gamma = gamma # parameter defining how soft the one hot encoding terms is
        self.beta = beta # parameter scaling the weight of the modularity term

    """
    creates modularity matrix - m
    """
    def modularity_matrix(self):
        A = self.adjacency_matrix
        g = A.sum(axis=1)
        m = g.sum() / 2.0

        B = A - np.outer(g, g) / (2.0 * m)

        return B

    """
    builds symmetric community detection qubo
    """
    def build_qubo(self):
        B = self.modularity_matrix()
        n = B.shape[0]
        N = n * self.number_communities

        Q = np.zeros((N, N), dtype=float)

        # modularity block term
        for c in range(self.number_communities):
            sl = slice(c * n, (c + 1) * n)
            Q[sl, sl] += -self.beta * B

        # one hot (soft) constraints
        for i in range(n):
            # community variable indices corresponding to node i
            idxs = [c * n + i for c in range(self.number_communities)]

            for idx in idxs:
                Q[idx, idx] += -self.gamma

            for a in range(self.number_communities):
                for b in range(a + 1, self.number_communities):
                    ia, ib = idxs[a], idxs[b]
                    
                    # split the 2*gamma penalty since its symmetric
                    Q[ia, ib] += self.gamma
                    Q[ib, ia] += self.gamma
                    
        return Q

    """
    outputs communities of each asset  
    """
    def decode_solution(self, x):
        n = self.adjacency_matrix.shape[0]
        labels = np.zeros(n, dtype=int)

        for i in range(n):
            vals = np.array([x[c * n + i] for c in range(self.number_communities)])

            labels[i] = int(np.argmax(vals))

        return labels