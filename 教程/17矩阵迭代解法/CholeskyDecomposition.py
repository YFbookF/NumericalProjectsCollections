from scipy import array, linalg, dot
# Cholesky 分解
# 此为完全Cholesky 分解，相当准确也相当耗时
# 因此实际会使用不完全Cholesky 分解，不怎么准确但凑合能用，而且非常节省时间
import numpy as np
A = np.array([[1,2,4],[2,13,23],[4,23,77]])
n = 3
L = np.zeros((n,n))# 修正
d = np.zeros((n))

# A = {L}{L^T}
v = np.zeros((n))
for j in range(0,n):
    for i in range(j,n):
        v[i] = A[i,j]
        for k in range(0,j):
            v[i] -= L[j,k]*L[i,k]
        L[i,j] = v[i] / np.sqrt(v[j])
        
        
L0 = linalg.cholesky(A, lower=True) # 库函数的结果