import numpy as np

# 参考自 opencloth corotation 

def cross(vec0,vec1):
    res = np.zeros((3))
    res[0] = vec0[1]*vec1[2] - vec0[2]*vec1[1]
    res[1] = vec0[2]*vec1[0] - vec0[0]*vec1[2]
    res[2] = vec0[0]*vec1[1] - vec0[1]*vec1[0]
    return res

def dot(vec0,vec1):
    res = 0
    for i in range(len(vec0)):
        res += vec0[i]*vec1[i]
    return res

nu = 0.33
Young = 5e5
d = Young / (1 + nu) / (1 - 2 * nu)
d0 = (1 - nu) * d
d1 = nu * d
d2 = Young / (1 + nu) / 2
# p351 eq.(10.72)
Dmat = np.array([[d0,d1,d1,0,0,0],
                 [d1,d0,d1,0,0,0],
                 [d1,d1,d0,0,0,0],
                 [0,0,0,d2,0,0],
                 [0,0,0,0,d2,0],
                 [0,0,0,0,0,d2]])
dt = 0.1
creep = 2
strain_yield = 1

# 初始化节点
Nx = 10
Ny = 3
Nz = 3
dx = 1 / Nx
node_num = (Nx+1)*(Ny+1)*(Nz+1)
node_pos = np.zeros((node_num,3))
node_pos_init = np.zeros((node_num,3))
node_fixed = np.zeros((node_num),dtype = bool)
node_mass = np.zeros((node_num))
node_force = np.zeros((node_num,3))

f_u = np.zeros((node_num,3))
f_ext = np.zeros((node_num,3))
'''

每个 K 是 7 * 9 的 矩阵的原因是
这是三维的，每个点前后左右上下加上自己一共 7 个点，每个点都需要

'''
K_sparse = np.zeros((node_num,7,9))
A_sparse = np.zeros((node_num,3,3))
b_sparse = np.zeros((node_num,3))

cnt = 0
for k in range(Nz+1):
    for j in range(Ny+1):
        for i in range(Nx+1):
            node_pos[cnt,:] = np.array([i*dx,j*dx,k*dx])
            node_pos_init[cnt,:] = node_pos[cnt,:]
            if node_pos[cnt,0] < 0.01:
                node_fixed[cnt] = True
            else:
                node_fixed[cnt] = False
            node_pos[cnt,1] += 0.5
            node_pos[cnt,2] -= 1.5 * dx
            cnt += 1

# 初始化元素
element_num = 5 * Nx * Ny * Nz
element_volume = np.zeros((element_num))
element = np.zeros((element_num,4),dtype = int)
element_undeformed = np.zeros((element_num,3,3))
element_Re = np.zeros((element_num,3,3))
element_Ke = np.zeros((element_num,16,9))
element_B = np.zeros((element_num,4,3))
element_plastic = np.zeros((element_num,6))
cnt = 0

# 添加元素
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            p0 = k*(Ny+1)*(Nx+1) + j*(Nx+1) + i
            p1 = p0 + 1
            p3 = (k+1)*(Ny+1)*(Nx+1) + j*(Nx+1) + i
            p2 = p3 + 1
            p7 = (k+1)*(Ny+1)*(Nx+1) + (j+1)*(Nx+1) + i
            p6 = p7 + 1
            p4 = k*(Ny+1)*(Nx+1) + (j+1)*(Nx+1) + i
            p5 = p4 + 1
            '''
            前面
            p3 --- p2
            |      |
            p0 ---p1
            后面
            p7 --- p6
            |      |
            p4 --- p5
            
            
            '''
            
            if (i+j+k)%2 == 1:
                element[cnt+0,:] = np.array([p1,p2,p6,p3])
                element[cnt+1,:] = np.array([p3,p6,p4,p7])
                element[cnt+2,:] = np.array([p1,p4,p6,p5])
                element[cnt+3,:] = np.array([p1,p3,p4,p0])
                element[cnt+4,:] = np.array([p1,p6,p4,p3])
            else:
                element[cnt+0,:] = np.array([p2,p0,p5,p1])
                element[cnt+1,:] = np.array([p2,p7,p0,p3])
                element[cnt+2,:] = np.array([p2,p5,p7,p6])
                element[cnt+3,:] = np.array([p0,p7,p5,p4])
                element[cnt+4,:] = np.array([p2,p0,p7,p5])
            cnt += 5
  
for ie in range(element_num):
    x0 = node_pos[element[ie,0],:]
    x1 = node_pos[element[ie,1],:]
    x2 = node_pos[element[ie,2],:]
    x3 = node_pos[element[ie,3],:]
    
    e10 = x1 - x0
    e20 = x2 - x0
    e30 = x3 - x0
    
    element_volume[ie] = dot(e10,cross(e20, e30))/6
    
    E = np.array([e10,e20,e30])
    invDetE = 1 / np.linalg.det(E)
    
    # 手动逆矩阵，牛逼
    invE10 = (e20[2]*e30[1] - e20[1]*e30[2]) * invDetE
    invE20 = (e30[2]*e10[1] - e30[1]*e10[2]) * invDetE
    invE30 = (e10[2]*e20[1] - e10[1]*e20[2]) * invDetE
    invE00 = - invE10 - invE20 - invE30
    
    invE11 = (e20[0]*e30[2] - e20[2]*e30[0]) * invDetE
    invE21 = (e30[0]*e10[2] - e30[2]*e10[0]) * invDetE
    invE31 = (e10[0]*e20[2] - e10[2]*e20[0]) * invDetE
    invE01 = - invE11 - invE21 - invE31
    
    invE12 = (e20[1]*e30[0] - e20[0]*e30[1]) * invDetE
    invE22 = (e30[1]*e10[0] - e30[0]*e10[1]) * invDetE
    invE32 = (e10[1]*e20[0] - e10[0]*e20[1]) * invDetE
    invE02 = - invE12 - invE22 - invE32
    
    element[ie,0,:] = np.array([invE00,invE01,invE02])
    element[ie,1,:] = np.array([invE10,invE11,invE12])
    element[ie,2,:] = np.array([invE20,invE21,invE22])
    element[ie,3,:] = np.array([invE30,invE31,invE32])
    
density = 1000
def calculateMass():
    for i in range(node_num):
        if node_fixed[i] == True:
            # 一个很大，很奢华至尊的数字
            node_mass[i] = 8848
        else:
            node_mass[i] = 1 / node_num
    for ie in range(element_num):
        term = density * element_volume[ie] * 0.25
        node_mass[element[ie,0]] += term
        node_mass[element[ie,1]] += term
        node_mass[element[ie,2]] += term
        node_mass[element[ie,3]] += term

def ortho(A):
    row0 = A[0,:]
    row1 = A[0,:]
    row2 = A[0,:]
    L0 = np.linalg.norm(row0)
    if L0 != 0:
        row0 /= L0
    row1 -= row0 * dot(row0,row1)
    L1 = np.linalg.norm(row1)
    if L1 != 0:
        row1 /= L1
    row2 = cross(row0,row1)
    A[0,:] = row0
    A[1,:] = row1
    A[2,:] = row2

def updateOrientation():
    for ie in range(element_num):
        div = 1 / element_volume[ie] * 6
        p0 = node_pos[element[ie,0],:]
        p1 = node_pos[element[ie,1],:]
        p2 = node_pos[element[ie,2],:]
        p3 = node_pos[element[ie,3],:]
        
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        
        # Cramer`s rule
        n1 = cross(e2, e3) * div
        n2 = cross(e3, e1) * div
        n3 = cross(e1, e2) * div
        
        e1 = element_undeformed[ie,0,:]
        e2 = element_undeformed[ie,1,:]
        e3 = element_undeformed[ie,2,:]
        
        element_Re[ie,0,0] = e1[0]*n1[0] + e2[0]*n2[0] + e3[0]*n3[0]
        element_Re[ie,0,1] = e1[0]*n1[1] + e2[0]*n2[1] + e3[0]*n3[1]
        element_Re[ie,0,2] = e1[0]*n1[2] + e2[0]*n2[2] + e3[0]*n3[2]
        
        element_Re[ie,1,0] = e1[1]*n1[0] + e2[1]*n2[0] + e3[1]*n3[0]
        element_Re[ie,1,1] = e1[1]*n1[1] + e2[1]*n2[1] + e3[1]*n3[1]
        element_Re[ie,1,2] = e1[1]*n1[2] + e2[1]*n2[2] + e3[1]*n3[2]
        
        element_Re[ie,2,0] = e1[2]*n1[0] + e2[2]*n2[0] + e3[2]*n3[0]
        element_Re[ie,2,1] = e1[2]*n1[1] + e2[2]*n2[1] + e3[2]*n3[1]
        element_Re[ie,2,2] = e1[2]*n1[2] + e2[2]*n2[2] + e3[2]*n3[2]
        

def computeForce():
    gravity = 0
    for i in range(node_num):
        node_force[i,1] = node_mass[i] * gravity
        node_force[i,:] = 0
        
def idxToSparseIdx(idx0,idx1):
    res = 0
    if idx0 == idx1:
        res = 0
    elif idx0 - 1 == idx1:
        res = 1
    elif idx0 + 1 == idx1:
        res = 2
    elif idx0 - Nx == idx1:
        res = 3
    elif idx0 + Nx == idx1:
        res = 4
    elif idx0 - Nx * Ny == idx1:
        res = 5
    elif idx0 + Nx * Ny == idx1:
        res = 6
    return res
        
def AssemblyStiffness():
    for ie in range(element_num):
        Bmat = element_B[ie,:,:]
        BmatT = Bmat.T
        
        for j in range(4):
            f = np.zeros((3))
            for i in range(4):
                Ke = element_Ke[j*4+i]
                x0 = node_pos_init[element[ie,i]]
                prod = np.array([[Ke[0]*x0[0]+Ke[1]*x0[1]+Ke[2]*x0[2]],
                                 [Ke[3]*x0[0]+Ke[4]*x0[1]+Ke[5]*x0[2]],
                                 [Ke[6]*x0[0]+Ke[7]*x0[1]+Ke[7]*x0[2]]])
                f += prod
                if j >= i:
                    temp = np.dot(np.dot(Bmat,Ke),BmatT)
                    idx0 = element[ie,j]
                    idx1 = idxToSparseIdx(idx0,element[ie,i])
                    for j0 in range(3):
                        for i0 in range(3):
                            K_sparse[idx0,idx1] += temp
                    if j > i:
                        idx0 = element[ie,i]
                        idx1 = idxToSparseIdx(idx0,element[ie,j])
                        K_sparse[idx0,idx1] += temp.T
            
            f_ext[element[ie,j]] -= Bmat * f
                        
def computePlastic():
    plastic_matrix = np.zeros((3,6))
    for ie in range(element_num):
        strain_total = np.zeros((6)) # plastic strain
        # compute total strain
        for i in range(4):
            pos = node_pos[element[ie,i],:]
            pos_init = node_pos_init[element[ie,i],:]
            ReT = element_Re[ie,:,:]
            temp = np.dot(ReT,pos) - pos_init
            
            bn = element_B[ie,i,0]
            cn = element_B[ie,i,0]
            dn = element_B[ie,i,0]
            Bmat = np.array([[bn,0,0],
                             [0,cn,0],
                             [0,0,dn],
                             [cn,bn,0],
                             [dn,0,bn],
                             [0,dn,cn]])
            
            strain_total += np.dot(Bmat,temp)
            
        # compute elastic strain
        strain_elastic = strain_total - element_plastic[ie,:]
        elastic_norm = np.linalg.norm(strain_elastic)
        strain_yield = 1
        if elastic_norm > strain_yield:
            amount = dt * min(creep,1/dt)
            element_plastic[ie,:] += amount * strain_elastic
        
        for j in range(4):
            bn = element_B[ie,j,0]
            cn = element_B[ie,j,0]
            dn = element_B[ie,j,0]
            Bmat = np.array([[bn,0,0],
                             [0,cn,0],
                             [0,0,dn],
                             [cn,bn,0],
                             [dn,0,bn],
                             [0,dn,cn]])
            BtD = np.dot(Bmat,Dmat)
            # P 是 3 x 1 矩阵
            P = element_volume[ie] * np.dot(BtD,e_plastic)
            f = np.dot(element_Re[ie,:,:],P)
            f_u[element[ie,j]] += f
            
def prepareSolve():
    # A = M + dt C + dt dt K
    # b = M v - dt (K x + f_u - f_ext)
    for i in range(node_num):
        b_sparse[i,:] = 0
        for j in range(4):
            
            Kmat = np.zeros((3,3))
            b_sparse[i] -= np.dot(Kmat,node_pos[i])
            
            A_sparse[i,:] = K_sparce[i,:] * dt * dt
            
            if i == j:
                A_sparse[i,0] += node_mass[i] + dt * mass_damping + node_mass[i]
                A_sparse[i,4] += node_mass[i] + dt * mass_damping + node_mass[i]
                A_sparse[i,8] += node_mass[i] + dt * mass_damping + node_mass[i]
        b_sparse[i] = node_mass[i] * vel[i] - dt *  (b_sparse[i] + f_u[i] - f_ext[i])
        
        