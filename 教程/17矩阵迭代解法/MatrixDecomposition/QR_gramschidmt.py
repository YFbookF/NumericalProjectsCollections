import numpy as np
# https://github.com/danbar/qr_decomposition/blob/master/qr_decomposition/qr_decomposition.py
A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]],dtype = float)

(num_rows, num_cols) = np.shape(A)

# Initialize empty orthogonal matrix Q.
Q = np.empty([num_rows, num_rows])
cnt = 0

# Compute orthogonal matrix Q.
for a in A.T:
    u = np.copy(a)
    for i in range(0, cnt):
        proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        u -= proj

    e = u / np.linalg.norm(u)
    Q[:, cnt] = e

    cnt += 1  # Increase columns counter.

# Compute upper triangular matrix R.
R = np.dot(Q.T, A)

