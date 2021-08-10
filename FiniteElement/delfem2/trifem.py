import numpy as np

pos0 = np.array([0,0,0])
pos1 = np.array([0.04,0,0])
pos2 = np.array([0,0.04,0])

C = np.array([pos0,pos1,pos2])

def cross3D(vec0,vec1,vec2):
    res = np.zeros((3))
    res[0] = vec0[0]*(vec1[1]*vec2[2] - vec1[2]*vec2[1])
    res[1] = vec0[1]*(vec1[2]*vec2[0] - vec1[0]*vec2[2])
    res[2] = vec0[2]*(vec1[0]*vec2[1] - vec1[1]*vec2[0])
    return res

def cross2D(vec1,vec2):
    res = np.zeros((3))
    res[0] = (vec1[1]*vec2[2] - vec1[2]*vec2[1])
    res[1] = (vec1[2]*vec2[0] - vec1[0]*vec2[2])
    res[2] = (vec1[0]*vec2[1] - vec1[1]*vec2[0])
    return res

def dot(vec0,vec1):
    n = len(vec0)
    res = 0
    for i in range(n):
        res += vec0[i]*vec1[i]
    return res


Gd = np.zeros((3,3)) # undeformaed edge vector
Gd[0,0] = C[1,0] - C[0,0]
Gd[0,1] = C[1,1] - C[0,1]
Gd[0,2] = C[1,2] - C[0,2]
Gd[1,0] = C[2,0] - C[0,0]
Gd[1,1] = C[2,1] - C[0,1]
Gd[1,2] = C[2,2] - C[0,2]

# 计算面积 开始

Gd[2,0] = (C[1,1] - C[0,1])*(C[2,2] - C[0,2]) - (C[2,1] - C[0,1])*(C[1,2] - C[0,2])
Gd[2,1] = (C[1,2] - C[0,2])*(C[2,0] - C[0,0]) - (C[2,2] - C[0,2])*(C[1,0] - C[0,0])
Gd[2,2] = (C[1,0] - C[0,0])*(C[2,1] - C[0,1]) - (C[2,0] - C[0,0])*(C[1,1] - C[0,1])

area = np.sqrt(Gd[2,0]**2 + Gd[2,1]**2  +Gd[2,2]**2) / 2

Gd[2,0] = Gd[2,0] / 2 / area
Gd[2,1] = Gd[2,1] / 2 / area
Gd[2,2] = Gd[2,2] / 2 / area

# 计算面积 结束

Gu = np.zeros((2,3)) # inverse of Gd
Gu[0,:] = cross2D(Gd[1,:], Gd[2,:])
invtmp1 = 1 / dot(Gu[0,:],Gd[0,:])
Gu[0,:] *= invtmp1

Gu[1,:] = cross2D(Gd[2,:], Gd[0,:])
invtmp1 = 1 / dot(Gu[1,:],Gd[1,:])
Gu[1,:] *= invtmp1

c = C.copy()

# deformed edge vector
gd = np.array([[c[1,0]-c[0,0],c[1,1]-c[0,1],c[1,2]-c[0,2]],
               [c[2,0]-c[0,0],c[2,1]-c[0,1],c[2,2]-c[0,2]]])

#  // green lagrange strain (with engineer's notation) 完全看不出来啊？？？
E2 = np.array([0.5*(dot(gd[0,:],gd[0,:]) - dot(Gd[0,:],Gd[0,:])),
               0.5*(dot(gd[1,:],gd[1,:]) - dot(Gd[1,:],Gd[1,:])),
               (dot(gd[0,:],gd[1,:]) - dot(Gd[0,:],Gd[1,:]))])

GuGu2= np.array([dot(Gu[0,:],Gu[0,:]),dot(Gu[1,:],Gu[1,:]),dot(Gu[0,:],Gu[1,:])])

Cons = np.zeros((3,3)) # consitutive tensor
lam = 1
nu = 4
Cons[0,0] = lam * GuGu2[0] * GuGu2[0] + 2 * nu * GuGu2[0] * GuGu2[0]
Cons[0,1] = lam * GuGu2[0] * GuGu2[1] + 2 * nu * GuGu2[2] * GuGu2[2]
Cons[0,2] = lam * GuGu2[0] * GuGu2[2] + 2 * nu * GuGu2[0] * GuGu2[2]
Cons[1,0] = lam * GuGu2[1] * GuGu2[0] + 2 * nu * GuGu2[2] * GuGu2[2]
Cons[1,1] = lam * GuGu2[1] * GuGu2[1] + 2 * nu * GuGu2[1] * GuGu2[1]
Cons[1,2] = lam * GuGu2[1] * GuGu2[2] + 2 * nu * GuGu2[2] * GuGu2[1]
Cons[2,0] = lam * GuGu2[2] * GuGu2[0] + 2 * nu * GuGu2[0] * GuGu2[2]
Cons[2,1] = lam * GuGu2[2] * GuGu2[1] + 2 * nu * GuGu2[2] * GuGu2[1]
Cons[2,2] = lam * GuGu2[2] * GuGu2[2] + 1 * nu * GuGu2[0] * GuGu2[1]

S2 = np.zeros((3))
S2[0] = Cons[0,0] * E2[0] + Cons[0,1] * E2[1] + Cons[0,2] * E2[2]
S2[1] = Cons[1,0] * E2[0] + Cons[1,1] * E2[1] + Cons[1,2] * E2[2]
S2[2] = Cons[2,0] * E2[0] + Cons[2,1] * E2[1] + Cons[2,2] * E2[2]

# compute energy
w = 0.5 * area * (E2[0] * S2[0] + E2[1] * S2[1] + E2[2] *S2[2])

dW = np.zeros((3,3))
dNdr = np.array([[-1,-1],[1,0],[0,1]],dtype = float)
for i in range(3):
    for j in range(3):
        dW[i,j] = area * (S2[0] * gd[0,j] * dNdr[i,0] + 
                          S2[2] * gd[0,j] * dNdr[i,1] + 
                          S2[2] * gd[1,j] * dNdr[i,0] + 
                          S2[1] * gd[1,j] * dNdr[i,1])
        
S3 = S2.copy()

# 计算正定矩阵
b = (S2[0] + S2[1]) / 2
d = (S2[0] - S2[1]) * (S2[0] - S2[1]) /4 + S2[2] * S2[2]
e = np.sqrt(d)
if b - e > 1e-20:
    S3 = S2.copy()
elif b + e < 0:
    S3[:] = 0
else:
    l = b + e 
    t0 = np.array([S2[0] -l,S2[2]])
    t1 = np.array([S2[2],S2[1] - l])
    sqlent0 = t0[0]**2 + t0[1]**2
    sqlent1 = t1[0]**2 + t1[1]**2
    if sqlent1 > sqlent0:
        if sqlent0 < 1e-20:
            S3[:] = 0
        else:
            t0 /= np.sqrt(sqlent0)
            t1 /= np.sqrt(sqlent0)
            S3[0] = l * t0[0] * t0[0]
            S3[1] = l * t0[1] * t0[1]
            S3[2] = l * t0[0] * t0[1]
    else:
        if sqlent1 < 1e-20:
            S3[:] = 0
        else:
            t0 /= np.sqrt(sqlent1)
            t1 /= np.sqrt(sqlent1)
            S3[0] = l * t1[0] * t1[0]
            S3[1] = l * t1[1] * t1[1]
            S3[2] = l * t1[0] * t1[1]
            
ddW = np.zeros((9,9))
for ino in range(3):
    for jno in range(3):
        for idim in range(3):
            for jdim in range(3):
                dtmp0 = 0
                dtmp0 += gd[0][idim] * dNdr[ino][0] * Cons[0][0] * gd[0][jdim] * dNdr[jno][0]
                dtmp0 += gd[0][idim] * dNdr[ino][0] * Cons[0][1] * gd[1][jdim] * dNdr[jno][1]
                dtmp0 += gd[0][idim] * dNdr[ino][0] * Cons[0][2] * gd[0][jdim] * dNdr[jno][1]
                dtmp0 += gd[0][idim] * dNdr[ino][0] * Cons[0][2] * gd[1][jdim] * dNdr[jno][0]
                dtmp0 += gd[1][idim] * dNdr[ino][1] * Cons[1][0] * gd[0][jdim] * dNdr[jno][0]
                dtmp0 += gd[1][idim] * dNdr[ino][1] * Cons[1][1] * gd[1][jdim] * dNdr[jno][1]
                dtmp0 += gd[1][idim] * dNdr[ino][1] * Cons[1][2] * gd[0][jdim] * dNdr[jno][1]
                dtmp0 += gd[1][idim] * dNdr[ino][1] * Cons[1][2] * gd[1][jdim] * dNdr[jno][0]
                dtmp0 += gd[0][idim] * dNdr[ino][1] * Cons[2][0] * gd[0][jdim] * dNdr[jno][0]
                dtmp0 += gd[0][idim] * dNdr[ino][1] * Cons[2][1] * gd[1][jdim] * dNdr[jno][1]
                dtmp0 += gd[0][idim] * dNdr[ino][1] * Cons[2][2] * gd[0][jdim] * dNdr[jno][1]
                dtmp0 += gd[0][idim] * dNdr[ino][1] * Cons[2][2] * gd[1][jdim] * dNdr[jno][0]
                dtmp0 += gd[1][idim] * dNdr[ino][0] * Cons[2][0] * gd[0][jdim] * dNdr[jno][0]
                dtmp0 += gd[1][idim] * dNdr[ino][0] * Cons[2][1] * gd[1][jdim] * dNdr[jno][1]
                dtmp0 += gd[1][idim] * dNdr[ino][0] * Cons[2][2] * gd[0][jdim] * dNdr[jno][1]
                dtmp0 += gd[1][idim] * dNdr[ino][0] * Cons[2][2] * gd[1][jdim] * dNdr[jno][0]
                idx_no = jno * 3 + ino
                idx_dim = jdim * 3 + idim
                ddW[idx_no,idx_dim] = dtmp0 * area
                
                dtmp1 = area * (S3[0] * dNdr[ino,0] * dNdr[jno,0] 
                                + S3[2] * dNdr[ino,0] * dNdr[jno,1]
                                + S3[2] * dNdr[ino,1] * dNdr[jno,0]
                                + S3[1] * dNdr[ino,1] * dNdr[jno,1])
                
                ddW[idx_no,0] += dtmp1
                ddW[idx_no,4] += dtmp1
                ddW[idx_no,8] += dtmp1
                
# 此时能量W，一阶导dW,二阶导ddw就计算完了