import numpy as np

n = 4
steps = 32
eps = 1e-5

# 矩阵A已经被我吃掉了
b = np.zeros((n*n))
b[5] = 1
b[6] = -1
x = np.zeros((n*n)) # 待求解的变量
d = np.zeros((n*n)) # 方向
r = np.zeros((n*n)) # 残差

def c(i,j,field):
    if i < 0 or i > n - 1 or j < 0 or j > n - 1:
        return 0
    return field[j*n+i]

def A(i,j,field):
    return field[j*n+i] * 4 - (c(i-1,j,field) + c(i+1,j,field)
                    + c(i,j-1,field) + c(i,j+1,field))

def init():
    for i in range(n):
        for j in range(n):
            idx = j * n + i
            d[idx] = b[idx] - A(i,j,x)
            r[idx] = d[idx]
           
def substep():
    global b
    global x
    global d
    global r
    dAd = eps
    for i in range(n):
        for j in range(n):
            idx = j * n + i
            dAd += d[idx] * A(i,j,d)
    alpha = 0
    for i in range(n*n):
        alpha += r[i] * r[i] / dAd
    beta = 0
    for i in range(n):
        for j in range(n):
            idx = j * n + i
            x[idx] = x[idx] + alpha * d[idx]
            r[idx] = r[idx] - alpha * A(i,j,d)
            beta += r[idx] * r[idx] / ((alpha + eps) * dAd)
    for i in range(n*n):
        d[i] = r[i] + beta * d[i]
 
init()
for ite in range(10):
    substep()
    
p = np.zeros((n,n)) # 压力
for i in range(n):
    for j in range(n):
        idx = j * n + i
        p[i,j] = x[idx]