import numpy as np
import sys
# 1.6.5 Lanczos Process
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

def computeNorm(xVec):
    res = 0
    n = len(xVec)
    for i in range(n):
        res += xVec[i]**2
    return np.sqrt(res)

A = np.array([[1,-1,1],[1,0,1],[1,1,2]],dtype = float)
n = 3

v = np.zeros((n,n+1))
v[0,0] = 1
beta = 0
vtemp = np.zeros((n))
for i in range(n):
    w = np.dot(A,v[:,i]) - beta * vtemp
    alpha = np.dot(np.transpose(w),v[:,i])
    w = w - alpha * v[:,i]
    beta = computeNorm(w)
    if abs(beta) < 1e-10:
        break
    v[:,i+1] = w / beta