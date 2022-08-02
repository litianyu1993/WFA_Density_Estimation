import numpy as np
from scipy import linalg as la
N = 5

q = np.arange(N, dtype=np.float64)
col, row = np.meshgrid(q, q)
r = 2 * q + 1
M = -(np.where(row >= col, r, 0) - np.diag(q))
T = np.sqrt(np.diag(2 * q + 1))
A = T @ M @ np.linalg.inv(T)
B = np.diag(T)[:, None]

for t in range(1, 10):
    At = A / t
    Bt = B / t
    tmpA = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
    tmpB = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
    print(tmpA)
    print(tmpB)

# print(A)
# print(B)