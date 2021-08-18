import numpy as np

# https://www.continuummechanics.org/deformationgradient.html

# 比如一个三角形，它的初始位置

x0_rest = np.array([0,0])
x1_rest = np.array([1,0])
x2_rest = np.array([0,1])

# 首先计算它的逆

m = np.array([x1_rest - x0_rest,x2_rest - x0_rest])

def computeInvM(m):
    det = m[0,0]*m[1,1] - m[1,0]*m[0,1]
    
    assert(det > 0)
    minv = np.zeros((2,2))
    minv[0,0] = m[1,1] / det
    minv[1,1] = m[0,0] / det
    minv[1,0] = - m[1,0] / det
    minv[0,1] = - m[0,1] / det
    return m
    
minv = computeInvM(m)
# 经过某些时间，三角形来到了新的位置
x0 = np.array([0,0])
x1 = np.array([0,2])
x2 = np.array([-2,0])

# 计算它的deformation Gradient
def computeDeformationGradient(x,invM):
    e1 = x[1,:] - x[0,:]
    e2 = x[2,:] - x[0,:]
    m = np.array([e1,e2])
    f = np.dot(m,invM) # ?? 此处有问题，应该是矩阵乘法
    # f = x * invM
    return f

f = computeDeformationGradient(np.array([x0,x1,x2]), minv)

def QRdecomposition2d(f):
    div = np.sqrt(f[0,0] + f[1,0])
    Q = np.zeros((2,2))
    R = np.zeros((2,2))
    Q[0,0] = f[0,0] / div
    Q[0,1] = - f[1,0] / div
    Q[1,0] = - f[1,0] / div 
    Q[1,1] = f[0,0] / div
    R[0,0] = div
    R[0,1] = (f[0,0]*f[0,1] + f[1,0]*f[1,1]) / div
    R[1,1] = (f[0,0]*f[1,1] - f[0,1]*f[1,0]) / div
    return Q    

q = QRdecomposition2d(f)

# Cauchy`s linear strain tensor
f0 = np.dot(np.transpose(q),f)
e = (f0 + np.transpose(f0)) / 2 - np.identity(2)

# compute plastic strain using FrobeniusNorm
ef = 0
for i in range(2):
    for j in range(2):
        ef += e[i,j] * e[i,j]
ef = np.sqrt(ef)
mEp = 0
yieldForce = 0
creep = 0
epmax = 0
dt = 0
if ef > yieldForce:
    mEp += dt * creep * e
    
    
if ef > epmax:
    mEp *= epmax / ef

# adjust strain
e -= mEp

# lame constants
lameLambda = 0
lameMu = 0

# isotropic hookean stress tensor
# https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
trace_e = e[0,0] + e[1,1]
s = lameLambda * trace_e * np.identity(2) + lameMu * 2 * e

# CalcCauchyStrainTensorDt
# calculate time derivative of Cauchy's strain tensor
v0 = np.array([0,0])
v1 = np.array([1,0])
v2 = np.array([0,1]) # 三角形的速度
dfdt = computeDeformationGradient(np.array([v0,v1,v2]), minv)


dfdt = np.dot(np.transpose(q),dfdt)
dedt = (np.transpose(dfdt) + dfdt) / 2

damp = 0
trace_dedt = dedt[0,0] + dedt[1,1]
dsdt = damp * trace_dedt * np.identity(2) + damp * 2 * dedt

p = s + dsdt

# 然后获取 p 的特征值

factor_a = 1
factor_b = - (p[0,0] + p[1,1])
factor_c = p[0,0] * p[1,1] - p[0,1] * p[1,0]

# ax^2 + bx + d 的根就是特征值
e1 = 0
e2 = 0