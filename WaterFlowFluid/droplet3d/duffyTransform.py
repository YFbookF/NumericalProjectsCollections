import numpy as np
# 2d Generalized Duffy Quadrature in 2 D
# 将三角形变换为正方形
# Generalized Duffy Transformation for Integrating Vertex Singularities
coord = np.array([[0,0],[1,0],[0,1]],dtype = float)

a = coord[0,0]
b = coord[0,1]

#shift 
sh_coord = np.array([[coord[0,0] - a,coord[0,1] - b],
                     [coord[1,0] - a,coord[1,1] - b],
                     [coord[2,0] - a,coord[2,1] - b]])

#affine mapping
a11 = sh_coord[1,0]
a12 = sh_coord[2,0] - sh_coord[1,0]
a21 = sh_coord[1,1]
a22 = sh_coord[2,1] - sh_coord[1,1]
A = np.array([[a11,a12],[a21,a22]])
detA = np.linalg.det(A) # detA 是 三角形面积的两倍
gauss_points = np.array([-0.5377,0.5377])
gauss_weight = np.array([1.0,1.0])

nsp = 2
x0 = gauss_points / 2 + 0.5
w0 = gauss_weight / 2

beta = 1 # Duffy parameter

Nx = nsp * nsp
Z1x = np.zeros((Nx))
Z1y = np.zeros((Nx))
Z1w = np.zeros((Nx))
Z2x = np.zeros((Nx))
Z2y = np.zeros((Nx))
Z2w = np.zeros((Nx))
for j in range(nsp):
    for i in range(nsp):
        idx = j * nsp + i
        Z1x[idx] = x0[i]**beta
        Z1y[idx] = x0[i]**beta*x0[j]
        Z1w[idx] = w0[i]*w0[j]*beta*x0[i]**(2*beta-1)
        
        Z2x[idx] = a11 * Z1x[idx] + a12 * Z1y[idx] + a
        Z2y[idx] = a21 * Z1x[idx] + a22 * Z1y[idx] + b
        Z2w[idx] = detA * Z1w[idx]