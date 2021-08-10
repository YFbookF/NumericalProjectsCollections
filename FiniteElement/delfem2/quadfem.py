import numpy as np

pos0 = np.array([0,0,0])
pos1 = np.array([0.04,0.04,0])
pos2 = np.array([0.04,0,0])
pos3 = np.array([0,0.04,0])

C = np.array([pos0,pos1,pos2,pos3])

def triArea3d(vec1,vec2,vec3):
    x = (vec2[1] - vec1[1]) * (vec3[2] - vec1[2]) - (vec3[1] - vec1[1]) * (vec2[2] - vec1[2])
    y = (vec2[2] - vec1[2]) * (vec3[0] - vec1[0]) - (vec3[2] - vec1[2]) * (vec2[0] - vec1[0])
    z = (vec2[0] - vec1[0]) * (vec3[1] - vec1[1]) - (vec3[0] - vec1[0]) * (vec2[1] - vec1[1])
    return np.sqrt(x*x + y*y + z*z) / 2

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

def distance(vec0,vec1):
    n = len(vec0)
    res = 0
    for i in range(n):
        res += (vec0[i] - vec1[i])**2
    return np.sqrt(res)

A0 = triArea3d(C[0,:], C[2,:], C[3,:])
A1 = triArea3d(C[1,:], C[3,:], C[2,:])
L0 = distance(C[2,:],C[3,:])
H0 = A0 * 2 / L0
H1 = A1 * 2 / L0

e23 = np.zeros((3))
e02 = np.zeros((3))
e03 = np.zeros((3))
e12 = np.zeros((3))
e13 = np.zeros((3))
for i in range(3):
    e23[i] = C[3,i] - C[2,i]
    e02[i] = C[2,i] - C[0,i]
    e03[i] = C[3,i] - C[0,i]
    e12[i] = C[2,i] - C[1,i]
    e13[i] = C[3,i] - C[1,i]

cot023 = - dot(e02, e23) / H0
cot032 = dot(e03,e23) / H0

cot123 = - dot(e12, e23) / H1
cot132 = dot(e13,e23) / H1

stiffness = 0.001
tmp0 = stiffness / ((A0 + A1) * L0 * L0)
K = np.array([-cot023 - cot032,-cot123 - cot132,
              cot032 + cot132,cot023 + cot123])

ddW = np.zeros((16,9))
for i in range(4):
    for j in range(4):
        temp = K[i] * K[j] * tmp0
        idx = j * 4 + i
        ddW[idx,0] = temp
        ddW[idx,4] = temp
        ddW[idx,8] = temp
        
W = 0
dW = np.zeros((4,3))
for ino in range(4):
    for idim in range(3):
        for jno in range(4):
            for jdim in range(3):
                idx_no = jno * 4 + ino
                idx_dim = jdim * 3 + idim
                dW[ino,idim] += ddW[idx_no,idx_dim] * C[jno,jdim]
        W += dW[ino,idim] * C[ino,idim]