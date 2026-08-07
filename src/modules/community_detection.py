# # import numpy as np
# # from .qubo import *

# # class CommunityDetection(Qubo):
# #     def __init__(self, adjacency_matrix, number_communities, gamma=8.0, beta=1.0, csv_filename="modularity_matrix.csv"):
# #         super().__init__()
# #         self.adjacency_matrix = np.asarray(adjacency_matrix)
# #         self.number_communities = number_communities 
# #         self.gamma = gamma  # parameter defining how soft the one hot encoding terms is
# #         self.beta = beta    # parameter scaling the weight of the modularity term
# #         self.csv_filename = csv_filename

# #         # Automatically export the modularity matrix to CSV on initialization
# #         if self.csv_filename:
# #             self.save_modularity_matrix_csv(self.csv_filename)

# #     """
# #     creates modularity matrix - m
# #     """
# #     def modularity_matrix(self): # note modularity can break if adjacency has negative weights or zero total edge mass
# #         A = self.adjacency_matrix
# #         g = A.sum(axis=1)
# #         m = g.sum() / 2.0

# #         B = A - np.outer(g, g) / (2.0 * m)

# #         return B

# #     """
# #     saves the modularity matrix B directly to a CSV file
# #     """
# #     def save_modularity_matrix_csv(self, filename="modularity_matrix.csv"):
# #         B = self.modularity_matrix()
# #         # Saves matrix as comma-separated values with float formatting
# #         np.savetxt(filename, B, delimiter=",", fmt="%.6f")
# #         return filename

# #     """
# #     builds symmetric community detection qubo
# #     """
# #     def build_qubo(self):
# #         B = self.modularity_matrix()
# #         n = B.shape[0]
# #         N = n * self.number_communities

# #         Q = np.zeros((N, N), dtype=float)

# #         # modularity block term
# #         for c in range(self.number_communities):
# #             sl = slice(c * n, (c + 1) * n)
# #             Q[sl, sl] += -self.beta * B

# #         # one hot (soft) constraints
# #         for i in range(n):
# #             # community variable indices corresponding to node i
# #             idxs = [c * n + i for c in range(self.number_communities)]

# #             for idx in idxs:
# #                 Q[idx, idx] += -self.gamma

# #             for a in range(self.number_communities):
# #                 for b in range(a + 1, self.number_communities):
# #                     ia, ib = idxs[a], idxs[b]
                    
# #                     # split the 2*gamma penalty since its symmetric
# #                     Q[ia, ib] += self.gamma
# #                     Q[ib, ia] += self.gamma
                    
# #         return Q

# #     """
# #     decodes solution with one hot encoding   
# #     """
# #     def decode_solution(self, x):
# #         n = self.adjacency_matrix.shape[0]
# #         labels = np.zeros(n, dtype=int)

# #         for i in range(n):
# #             vals = np.array([x[c * n + i] for c in range(self.number_communities)])

# #             labels[i] = int(np.argmax(vals))

# #         return labels

# import numpy as np
# from .qubo import *

# class CommunityDetection(Qubo):
#     def __init__(self, adjacency_matrix, number_communities, gamma=8.0, beta=1.0):
#         super().__init__()
#         self.adjacency_matrix = np.asarray(adjacency_matrix)
#         self.number_communities = number_communities 
#         self.gamma = gamma # parameter defining how soft the one hot encoding terms is
#         self.beta = beta # parameter scaling the weight of the modularity term

#     """
#     creates modularity matrix - m
#     """
#     def modularity_matrix(self): # note modularity can break if adjacency has negative weights or zero total edge mass
#         A = self.adjacency_matrix
#         g = A.sum(axis=1)
#         m = g.sum() / 2.0

#         B = A - np.outer(g, g) / (2.0 * m)

#         return B

#     """
#     builds symmetric community detection qubo
#     """
#     def build_qubo(self):
#         B = self.modularity_matrix()
#         n = B.shape[0]
#         N = n * self.number_communities

#         Q = np.zeros((N, N), dtype=float)

#         # modularity block term
#         for c in range(self.number_communities):
#             sl = slice(c * n, (c + 1) * n)
#             Q[sl, sl] += -self.beta * B

#         # one hot (soft) constraints
#         for i in range(n):
#             # community variable indices corresponding to node i
#             idxs = [c * n + i for c in range(self.number_communities)]

#             for idx in idxs:
#                 Q[idx, idx] += -self.gamma

#             for a in range(self.number_communities):
#                 for b in range(a + 1, self.number_communities):
#                     ia, ib = idxs[a], idxs[b]
                    
#                     # split the 2*gamma penalty since its symmetric
#                     Q[ia, ib] += self.gamma
#                     Q[ib, ia] += self.gamma
                    
#         return Q

#     """
#     decodes solution with one hot encoding  
#     """
#     def decode_solution(self, x):
#         n = self.adjacency_matrix.shape[0]
#         labels = np.zeros(n, dtype=int)

#         for i in range(n):
#             vals = np.array([x[c * n + i] for c in range(self.number_communities)])

#             labels[i] = int(np.argmax(vals))

#         return labels
import numpy as np
from .qubo import *

class CommunityDetection(Qubo):
    def __init__(
        self, 
        adjacency_matrix, 
        number_communities, 
        gamma=8.0, 
        beta=1.0, 
        min_community_size=None, 
        max_community_size=None, 
        lambda_size=50.0
    ):
        super().__init__()
        self.adjacency_matrix = np.asarray(adjacency_matrix)
        self.number_communities = number_communities 
        self.gamma = gamma  # Parameter defining how soft the one-hot encoding terms are
        self.beta = beta    # Parameter scaling the weight of the modularity term
        
        # Size constraint parameters: a <= sum_i y_{i,j} <= b
        self.min_community_size = min_community_size  # 'a'
        self.max_community_size = max_community_size  # 'b'
        self.lambda_size = lambda_size                # Penalty multiplier for size constraint

    def modularity_matrix(self):
        """Creates modularity matrix B."""
        A = self.adjacency_matrix
        g = A.sum(axis=1)
        m = g.sum() / 2.0

        if m == 0:
            raise ValueError("Total edge weight in adjacency matrix is zero.")

        B = A - np.outer(g, g) / (2.0 * m)
        return B

    def _get_slack_weights(self, K):
        """
        Generates weights for bounded logarithmic slack encoding:
        s = sum_{l=0}^{m-1} 2^l * s_l + (K - 2^m + 1) * s_m
        """
        if K <= 0:
            return []
        
        m = int(np.floor(np.log2(K)))
        weights = [2**l for l in range(m)]
        weights.append(K - (2**m) + 1)
        return weights

    def build_qubo(self):
        """Builds symmetric community detection QUBO matrix Q."""
        B = self.modularity_matrix()
        n = B.shape[0]
        k = self.number_communities

        # Determine if size constraints are active
        a = self.min_community_size
        b = self.max_community_size
        has_size_constraint = (a is not None) or (b is not None)

        if has_size_constraint:
            if a is None:
                a = 0
            if b is None:
                b = n
            
            K = b - a
            if K < 0:
                raise ValueError("min_community_size (a) cannot be greater than max_community_size (b).")
            
            slack_weights = self._get_slack_weights(K)
            n_slack = len(slack_weights)
        else:
            n_slack = 0

        N_nodes = n * k
        N_total = N_nodes + k * n_slack
        Q = np.zeros((N_total, N_total), dtype=float)

        # 1. Modularity term (-beta * B on block diagonal)
        for c in range(k):
            sl = slice(c * n, (c + 1) * n)
            Q[sl, sl] += -self.beta * B

        # 2. One-hot soft constraints per node i: gamma * (sum_c y_{i,c} - 1)^2
        for i in range(n):
            idxs = [c * n + i for c in range(k)]

            for idx in idxs:
                Q[idx, idx] += -self.gamma

            for a_idx in range(k):
                for b_idx in range(a_idx + 1, k):
                    ia, ib = idxs[a_idx], idxs[b_idx]
                    Q[ia, ib] += self.gamma
                    Q[ib, ia] += self.gamma

        # 3. Single-slack bounded size constraints per community c
        if has_size_constraint:
            lam = self.lambda_size
            slack_offset = N_nodes

            for c in range(k):
                # Build list of terms inside the constraint expression before squaring:
                # Expr = sum_{i=0}^{n-1} (+1)*y_{i,c} + sum_{l=0}^{m} (-w_l)*s_{c,l} - a
                terms = []
                
                # Add node variables y_{i,c} with weight v_p = +1
                for i in range(n):
                    terms.append((c * n + i, 1.0))
                
                # Add slack variables s_{c,l} with weight v_p = -w_l
                for l, w in enumerate(slack_weights):
                    s_idx = slack_offset + c * n_slack + l
                    terms.append((s_idx, -float(w)))

                # Expand: lam * (sum_p v_p * x_p - a)^2
                #       = lam * [ sum_p (v_p^2 - 2*a*v_p)*x_p + 2 * sum_{p<q} v_p*v_q*x_p*x_q ]
                for p_idx, (idx_p, v_p) in enumerate(terms):
                    # Linear (diagonal) contribution
                    Q[idx_p, idx_p] += lam * (v_p**2 - 2.0 * a * v_p)

                    # Quadratic interaction (off-diagonal) contributions
                    for q_idx in range(p_idx + 1, len(terms)):
                        idx_q, v_q = terms[q_idx]
                        coeff = lam * v_p * v_q
                        Q[idx_p, idx_q] += coeff
                        Q[idx_q, idx_p] += coeff

        return Q

    def decode_solution(self, x):
        """Decodes binary solution vector into community labels per node."""
        n = self.adjacency_matrix.shape[0]
        labels = np.zeros(n, dtype=int)

        # Reads node variables (ignoring any appended slack variables)
        for i in range(n):
            vals = np.array([x[c * n + i] for c in range(self.number_communities)])
            labels[i] = int(np.argmax(vals))

        return labels