import numpy as np

nl = 16
nr = 16
length = 1
radius = 1
dl = 1 / nl # 圆柱侧边一段的长
dr = 2 * np.pi / nr # 圆柱底面一段的角度

num_coord = (nr * (nl + 1) + 2) * 3
coord = np.zeros((num_coord,3))
coord[0,:] = np.array([0,-length / 2 - radius,0])
cnt = 1
for i in range(5):
    t0 = np.pi / 2 * (nr - 1 - i) / nr
    y0 = - length / 2 - radius * np.sin(t0)
    c0 = radius * np.cos(t0)
    for j in range(nr):
        x0 = c0 * np.cos(2 * np.pi * j / nr)
        z0 = c0 * np.sin(2 * np.pi * j / nr)
        coord[cnt,:] = np.array([x0,y0,z0])
        cnt += 1        
        
for i in range(7):
    y0 = - length / 2 + (i + 1) * length / 8
    for j in range(nr):
        x0 = radius * np.cos(2 * np.pi * j / nr)
        z0 = radius * np.sin(2 * np.pi * j / nr)
        coord[cnt,:] = np.array([x0,y0,z0])
        cnt += 1      
        
for i in range(5):
    t0 = np.pi / 2 * i / nr
    y0 = length / 2 + radius * np.sin(t0)
    c0 = radius * np.cos(t0)
    for j in range(nr):
        x0 = c0 * np.cos(2 * np.pi * j / nr)
        z0 = c0 * np.sin(2 * np.pi * j / nr)
        coord[cnt,:] = np.array([x0,y0,z0])
        cnt += 1      

coord[cnt,:] = np.array([0,length / 2 + radius,0])


num_elements = nr * nl * 2 + nr *2
elements = np.zeros((num_elements,3))
cnt_ele = 0
for i in range(nr):
    elements[cnt_ele,:] = np.array([0,int(i%nr)+1,int((i+1)%nr)+1])
    cnt_ele += 1
for i in range(nl):
    for j in range(nr):
        idx1 = (i + 0) * nr + 1 + (j + 0) % nr
        idx2 = (i + 0) * nr + 1 + (j + 1) % nr
        idx3 = (i + 1) * nr + 1 + (j + 1) % nr
        idx4 = (i + 1) * nr + 1 + (j + 0) % nr
        elements[cnt_ele,:] = np.array([idx3,idx2,idx1])
        cnt_ele += 1
        elements[cnt_ele,:] = np.array([idx4,idx3,idx1])
        cnt_ele += 1
for i in range(nr):
    idx0 = nr * (nl + 1) + 1
    idx1 = nl * nr + 1 + int((i + 1) % nr)
    idx2 = nl * nr + 1 + int((i + 0) % nr)
    elements[cnt_ele,:] = np.array([idx0,idx1,idx2])
    cnt_ele += 1
    
num_bone = 2
bone_parent = np.zeros((num_bone))
bone_invBindMat = np.zeros((num_bone,4,4))
bone_transRelative = np.zeros((num_bone,4,4))
bone_affine = np.zeros((num_bone,4,4))
bone_parent[0] = -1
for i in range(4):
    bone_invBindMat[:,i,i] = 1
bone_invBindMat[0,1,3] = 0.5
bone_transRelative[0,0,1] = - 0.1
bone_transRelative[1,0,1] = 0.1
aw = np.zeros((num_coord * num_bone)) # 似乎是权重
for i in range(num_coord):
    pos = coord[i,:]
    w_tot = 0
    for j in range(num_bone):
        pb = np.array([bone_invBindMat[j,3],bone_invBindMat[j,7],bone_invBindMat[j,11]])
        # 模式上每个节点 离 骨骼的距离
        bone_len = np.sqrt((pos[0] - pb[0])**2 + (pos[1] - pb[1])**2 + (pos[2] - pb[2])**2)
        # 的倒数
        wb = 1 / (bone_len + 1e-10)
        idx = int(i * num_bone + j)
        # 就是权重
        aw[idx] = wb
        w_tot = wb
        # 加权平均一下
    for j in range(num_bone):
        idx = int(i * num_bone + j)
        aw[idx] /= w_tot

def QuatMul(p,q):
    res = np.zeros((4))
    res[0] = p[3] * q[0] + p[0] * q[3] + p[1] * q[2] - p[2] * q[1]
    res[1] = p[3] * q[1] + p[0] * q[2] + p[1] * q[3] + p[2] * q[0]
    res[2] = p[3] * q[2] + p[0] * q[1] - p[1] * q[0] + p[2] * q[3]
    res[3] = p[3] * q[3] - p[0] * q[0] - p[1] * q[1] - p[2] * q[2]
    return res


        
# 将矩阵转换为四元数


def MulMat(mat,vec):
    n = len(vec)
    res = np.zeros((n))
    for i in range(n):
        res[i] = np.dot(mat[i,:],vec)
    return res

coord_rest = coord.copy()
time = 0
timeFinal = 1
while(time < timeFinal):
    rx = 0
    ry = 0
    rz = np.sin(time / 100) /  2
    dqx = np.array([np.sin(rx / 2), 0,  0,  np.cos(rx / 2)])
    dqy = np.array([0,  np.sin(ry / 2),  0,  np.cos(ry / 2)])
    dqz = np.array([0,  0,  np.sin(rz / 2),  np.cos(rz / 2)])
    dtemp_yx = QuatMul(dqy,dqx)
    q = QuatMul(dqz,dtemp_yx)
    
    for i in range(num_bone):
        m01 = bone_transRelative[i,:,:]
        m01 = Quat()
        bone_affine[i,:,:] = m01[:,:]
    
    for i in range(num_coord):
        p0 = coord_rest[i,:]
        p1 = np.zeros((3,3))
        for j in range(num_bone):
            p2 = np.zeros((3))
            p0a = np.array([p0[0],p0[1],p0[2],1])
            p1a = MulMat(bone_invBindMat[j,:,:], p0a)
            p2a = MulMat(bone_affine[j,:,:], p0a)
            p2[:] = p2a[:-1]
            idx = int(i * num_bone + j)
            p1 += aw[idx] * p2