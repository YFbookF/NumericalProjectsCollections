import numpy as np
# 1.5.3 极小残差
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

A = np.array([[3,0,1],[0,4,2],[1,2,3]])
b = np.array([1,0,0])
n = 3

r = np.zeros((3)) # 残差
x = np.zeros((3))

error = 1
eps = 1e-10
ite_max = 100
for ite in range(ite_max):
    r = b - np.dot(A,x)
    error = np.dot(np.transpose(r),r)
    if(error < eps): # 此时收敛
        break
    Ar = np.dot(A,r)
    a0 = np.dot(np.transpose(r),Ar)
    a1 = np.dot(np.transpose(Ar),Ar) # 怀疑书上公式写错
    a = a0 / a1
    x = x + a * r
    
    
