import numpy as np
import sys
# 1.6.6 Lanczos Bi Normalization Process
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

def computeNorm(xVec):
    res = 0
    n = len(xVec)
    for i in range(n):
        res += xVec[i]**2
    return np.sqrt(res)

def computeSqrt(vec0,vec1):
    res = 0
    n = len(vec0)
    for i in range(n):
        res += vec0[i]*vec1[i]
    return np.sqrt(res)

A = np.array([[1,-1,1],[1,0,1],[1,1,2]],dtype = float)
n = 3

v = np.zeros((n,n+1))
w = np.zeros((n,n+1))
v[0,0] = 1
w[0,0] = 1
beta = 0
delta = 0
vminus = np.zeros((n))
wminus = np.zeros((n))
for i in range(n):
    alpha = np.dot(np.transpose(w[:,i]),np.dot(A,v[:,i]))
    vhat = np.dot(A,v[:,i]) - alpha * v[:,i] - beta * vminus
    what = np.dot(np.transpose(A),w[:,i]) - alpha * w[:,i] - delta * wminus
    delta = computeSqrt(vhat, what)
    if abs(delta) < 1e-10:
        break
    beta = delta # 这步似乎有问题？
    w[:,i+1] = what / beta
    v[:,i+1] = vhat / alpha
    