import numpy as np
# 6.3.1 Newton 牛顿法解非线性方程
# 参考 《迭代方法和预处理技术》谷同祥 等 编著 科技出版社
# 个人水平有限，程序可能有错，仅供参考

def Fun(x,y,z):
    return np.array([2*x + 4*y + z - 13,
                     2*x**2 + 4*y**2 + z**2 - 27,
                     2*x*y + 4*y*z + z*x - 31],dtype =float)

def Jacobi(x,y,z):
    return np.array([[2,    4,      1],
                    [4*x,   8*y,    2*z],
                    [2*y,   4*z,    x]],dtype =float)


xx = np.zeros((3))
for i in range(3):
    xx[i] = i * 0.1
ite_max = 1000
for ite in range(ite_max):
    Jmat = Jacobi(xx[0],xx[1],xx[2])
    F_x_inv = np.linalg.inv(Jmat)
    Fx = Fun(xx[0],xx[1],xx[2])
    sk = - np.dot(F_x_inv, Fx)
    xx = xx + sk
    
res = Fun(xx[0],xx[1],xx[2])