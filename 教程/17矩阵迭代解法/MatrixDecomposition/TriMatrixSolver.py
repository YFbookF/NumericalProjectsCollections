import numpy as np
n = 3
L = np.zeros((3,3))
cnt = 1
for j in range(n):
    for i in range(j,n):
        L[i,j] = cnt
        cnt += 1

rhs = np.ones((n))

# 下三角矩阵求解
# L11 * res1 = rhs1
# L21 * res1 + L22 * res2 = rhs 2
result = np.zeros((n))
for i in range(n):
    result[i] = rhs[i] / L[i,i]
    for j in range(i):
        result[i] -= L[i,j] / L[i,i] * result[j]
        
# 上三角矩阵求解
# L33 * res3 = rhs3
# L22 * res2 + L23 * res3 = rhs2
resultT = np.zeros((n))
for i in range(n-1,-1,-1):
    resultT[i] = rhs[i] / L[i,i]
    for j in range(i+1,n):
        resultT[i] -= L[j,i] / L[i,i] * resultT[j]