import numpy as np
A = np.array([[4,2,4],[2,13,23],[4,23,77]])
n = 3
lower = np.zeros((n,n))
upper = np.zeros((n,n))
v = np.zeros((n))
for j in range(0,n):
    for i in range(j,n):
        v[i] = A[i,j]
        for k in range(0,j):
            # A 矩阵的形式是上三角
            v[i] -= lower[j,k]*upper[k,i]
        lower[i,j] = v[i] / np.sqrt(v[j])
        upper[j,i] = lower[i,j]
        