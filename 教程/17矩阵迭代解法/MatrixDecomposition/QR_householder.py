import numpy as np
# https://github.com/danbar/qr_decomposition/blob/master/qr_decomposition/qr_decomposition.py
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]],dtype = float)

(num_rows, num_cols) = np.shape(A)

# Initialize orthogonal matrix Q and upper triangular matrix R.
Q = np.eye(num_rows)
R = np.copy(A)

# Iterative over column sub-vector and
# compute Householder matrix to zero-out lower triangular matrix entries.
for cnt in range(num_rows - 1):
    x = R[cnt:, cnt]

    e = np.zeros_like(x)
    e[0] = - np.sign(A[cnt, cnt]) * np.linalg.norm(x)
    u = x + e
    v = u / np.linalg.norm(u)

    Q_cnt = np.eye(num_rows)
    Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)

    R = np.dot(Q_cnt, R)
    Q = np.dot(Q, Q_cnt.T)


