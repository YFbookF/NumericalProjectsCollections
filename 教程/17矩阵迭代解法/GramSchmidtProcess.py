import numpy as np
import sys
# 1.6.1 Gram-Schmidt Process
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# https://math.hmc.edu/calculus/hmc-mathematics-calculus-online-tutorials/linear-algebra/gram-schmidt-method/
# 个人水平有限，程序可能有错，仅供参考

def computeNorm(xVec):
    res = 0
    n = len(xVec)
    for i in range(n):
        res += xVec[i]**2
    return np.sqrt(res)
        
def dot(vec0,vec1):
    res = 0
    n = len(vec0)
    for i in range(n):
        res += vec0[i] * vec1[i]
    return res

def computeNorm2(xVec):
    res = 0
    n = len(xVec)
    for i in range(n):
        res += xVec[i]**2
    return res

x = np.array([[1,-1,1],[1,0,1],[1,1,2]],dtype = float)
n = 3
r = np.zeros((n,n))
r[0,0] = computeNorm(x[0,:])
if r[0,0] == 0:
    sys.exit()
norm = computeNorm(x[0,:])
x[0,:] /= norm
for j in range(1,n):
    for i in range(j):
        r[i,j] = dot(x[j,:],x[i,:])/computeNorm2(x[j-1,:])
    for i in range(j):
        x[j,:] -= r[i,j] * x[i,:] 
    norm = computeNorm(x[j,:])
    x[j,:] /= norm
    