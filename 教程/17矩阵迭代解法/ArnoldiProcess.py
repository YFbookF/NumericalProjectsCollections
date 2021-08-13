import numpy as np
import sys
# 1.6.4 Arnoldi Process
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

m = 10
V = np.zeros((n,m))
H = np.zeros((m+1,m))

V[0,0] = 1 # 最简单的范数为1的向量了，实际应该是归一化的b - Ax
for j in range(0,m):
    w = np.dot(A,V[:,j])
    for i in range(j+1):
        H[i,j] = np.dot(np.transpose(w),V[:,i])
        # 下面的是GradSchmidt，但上面这一行看不懂
        w = w - H[i,j] * V[:,i]
    H[j+1,j] = computeNorm(w)
    if abs(H[j+1,j] < 1e-10):
        break
    V[:,j+1] = w / H[j+1,j]