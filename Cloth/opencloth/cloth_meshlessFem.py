import numpy as np

# 参考自 opencloth meshless fem
# Point Based Animation of Elastic, Plastic and Melting Objects

def cross(vec0,vec1):
    res = np.zeros((3))
    res[0] = vec0[1]*vec1[2] - vec0[2]*vec1[1]
    res[1] = vec0[2]*vec1[0] - vec0[0]*vec1[2]
    res[2] = vec0[0]*vec1[1] - vec0[1]*vec1[0]
    return res

def outterProduct(vec0,vec1):
    res = np.zeros((3,3))
    for i in range(len(vec0)):
        res[i] = vec0[i]*vec1[:]
    return res

def dot(vec0,vec1):
    res = 0
    for i in range(len(vec0)):
        res += vec0[i]*vec1[i]
    return res

nu = 0.33
Young = 5e5
d = Young / (1 + nu) / (1 - 2 * nu)
d_0 = (1 - nu) * d
d_1 = nu * d
d_2 = Young / (1 + nu) / 2
scaleFactor = 0
density = 1e4
damping = 500

# 初始化节点
Nx = 10
Ny = 5
Nz = 5
dx = 1 / Nx
node_num = Nx * Ny * Nz # 节点数量
node_pos = np.zeros((node_num,3)) # 节点位置
node_pos_init = np.zeros((node_num,3)) # 节点初始位置
node_fixed = np.zeros((node_num),dtype = bool) # 节点是否固定
node_mass = np.zeros((node_num)) # 节点质量
node_density = np.zeros((node_num)) # 节点密度
node_volume = np.zeros((node_num)) # 节点体积
node_force = np.zeros((node_num,3)) # 节点上的力
node_vel = np.zeros((node_num,3)) # 节点的速度
node_di = np.zeros((node_num,3)) 
node_acc = np.zeros((node_num,3)) # 节点的加速度
node_minv = np.zeros((node_num,3,3)) # 节点的逆动量矩阵，用来算defomration gradient
node_J = np.zeros((node_num,3,3))
node_strain = np.zeros((node_num,3,3)) # 节点的应变
node_stress = np.zeros((node_num,3,3)) # 节点的应力

# meshless 就是不使用那些四边形单元，三角形单元，四面体单元的那些
# 但是还是处理邻近节点，所以，似乎，大概，没什么区别？
neighbor_num = 10
neighbor_idx = np.zeros((node_num,neighbor_num),dtype = int)
neighbor_dis = np.zeros((node_num,neighbor_num))
neighbor_rdis = np.zeros((node_num,neighbor_num,3))
neighbor_weight = np.zeros((node_num,neighbor_num))
neighbor_dj = np.zeros((node_num,neighbor_num,3))
node_radius = np.zeros((node_num))
node_support_radii = np.zeros((node_num))
node_displacement = np.zeros((node_num,3))
disarr = np.zeros((node_num))
idxarr = np.zeros((node_num),dtype = int)

cnt = 0
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            sizeX = 4
            hsizeY = 0.5
            hsizeZ = 0.5
            ypos = 4
            node_pos[cnt,:] = np.array([i/(Nx-1)*sizeX,
                   ((j/(Ny-1))*2-1)*hsizeY+ypos,
                   ((k/(Nz-1))*2-1)*hsizeY])
            if node_pos[cnt,0] == 0:
                node_fixed[cnt] = True
            else:
                node_fixed[cnt] = False
            cnt += 1
            
def quickSort(left,right):
    if left >= right:
        return 
    key = disarr[left]
    keyidx = idxarr[left]
    i = left
    j = right
    while i < j:
        while i < j and disarr[j] >= key:
            j -= 1
        if i < j:
            disarr[i] = disarr[j]
            idxarr[i] = idxarr[j]
            i += 1
        while i < j and disarr[i] < key:
            i += 1
        if i < j:
            disarr[j] = disarr[i]
            idxarr[j] = idxarr[i]
            j -= 1
    disarr[i] = key
    idxarr[i] = keyidx
    quickSort(left, i-1)
    quickSort(i+1, right)
    
            
def kNearest():
    for i in range(node_num):
        for j in range(node_num):
            d = node_pos[i] - node_pos[j]
            disarr[j] = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            idxarr[j] = j
        quickSort(0, node_num-1)
        avg = 0
        for j in range(neighbor_num):
            neighbor_dis[i,j] = disarr[j+1]
            neighbor_idx[i,j] = idxarr[j+1]
            neighbor_rdis[i,j,:] = node_pos[idxarr[j+1],:] - node_pos[i,:]
            avg += disarr[j+1]
        node_radius[i] = avg / 10
        node_support_radii[i] = avg * 3 / 10
        # For this read section 3.2 Initialization on page 4 Based on Eq. 9
        mul_factor = 315 / (64 * np.pi * pow(node_support_radii[i],9))
        h2 = node_support_radii[i]**2
        for j in range(neighbor_num):
            r2 = neighbor_dis[i,j]**2
            neighbor_weight[i,j] = mul_factor * pow(h2 - r2,3)
            
def computeFactor():
    global scaleFactor
    for i in range(node_num):
        summ = 0
        for j in range(neighbor_num):
            radius = node_radius[neighbor_idx[i,j]]
            weight = neighbor_weight[i,j]
            summ += pow(radius,3) * weight
        scaleFactor += 1 / summ
    scaleFactor /= node_num
    
    # 计算质量
    for i in range(node_num):
        node_mass[i] = scaleFactor * pow(node_radius[i],3) * density
    
    # 计算密度和体积
    for i in range(node_num):
        for j in range(neighbor_num):
            neighbor_mass = node_mass[neighbor_idx[i,j]]
            node_density[i] += neighbor_mass * neighbor_weight[i,j]
        node_volume[i] = node_mass[i] / node_density[i]
        
    # 计算动量矩阵的逆，为计算deformation gradient作准备
    for i in range(node_num):
        Amat = np.zeros((3,3))
        for j in range(neighbor_num):
            rdist = neighbor_rdis[i,j,:]
            weight = neighbor_weight[i,j]
            Asum = outterProduct(rdist,rdist * weight)
            Amat += Asum
        node_minv[i,:,:] = np.linalg.inv(Amat)
        
    # Eq.20 and Eq.21
    for i in range(node_num):
        for j in range(neighbor_num):
            rdist = neighbor_rdis[i,j,:]
            weight = neighbor_weight[i,j]
            neighbor_dj[i,j,:] = np.dot(node_minv[i,:,:],rdist * weight)
            node_di[i,:] -= neighbor_dj[i,j,:]
        
def computeJacobians():
    for i in range(node_num):
        for j in range(neighbor_num):
            idx = neighbor_idx[i,j]
            node_displacement[idx] = node_pos[idx] - node_pos[idx]
        Bmat = np.zeros((3,3))
        for j in range(neighbor_num):
            idx = neighbor_idx[i,j]
            # 记住，是 别人 减 自己
            dj = node_displacement[idx] - node_displacement[i]
            rdist = neighbor_rdis[i,j] 
            bj = outterProduct(dj,rdist) * neighbor_weight[i,j]
            Bmat += bj
        Bmat = Bmat.T
        du = np.dot(node_minv[i,:,:],Bmat)# deformation gradient 3 x 3
        node_J[i,:,:] = np.identity(3) + du.T
        
def computeStrainAndStress():
 
    for i in range(node_num):
        
        jac = node_J[i,:,:]
        node_strain[i,:,:] = np.dot(jac.T,jac) - np.identity(3)
        e = node_strain[i,:,:]
        
        node_stress[i,0,0] = d_0 * e[0,0] + d_1 * e[1,1] + d_1 * e[2,2]
        node_stress[i,1,1] = d_1 * e[0,0] + d_0 * e[1,1] + d_1 * e[2,2]
        node_stress[i,2,2] = d_1 * e[0,0] + d_1 * e[1,1] + d_0 * e[2,2]
        
        node_stress[i,0,1] = node_stress[i,1,0] = d_2 * e[0,1]
        node_stress[i,1,2] = node_stress[i,2,1] = d_2 * e[1,2]
        node_stress[i,2,0] = node_stress[i,0,2] = d_2 * e[2,0]
        
        
def updateForce():
    
    for i in range(node_num):
        node_force[i,:] = 0
        gravity = 0
        if node_fixed[i] == False:
            node_force[i,1] = gravity
        node_force[i,:] -= node_vel[i,:] * damping
    
    computeJacobians()
    
    computeStrainAndStress()
    
    for i in range(node_num):
        F_e = np.zeros((3,3))
        F_v = np.zeros((3,3))
        F_e = - 2 * node_volume[i] * np.dot(node_J[i],node_stress[i])
        
        row0 = cross(node_J[i,1,:], node_J[i,2,:])
        row1 = cross(node_J[i,2,:], node_J[i,0,:])
        row2 = cross(node_J[i,0,:], node_J[i,1,:])
        
        mt = np.array([row0,row1,row2]).T
        
        detJ = np.linalg.det(node_J[i,:,:])
        kv = 0
        F_v = - node_volume[i] * kv * (detJ - 1) * mt
        for j in range(neighbor_num):
            node_force[i] += np.dot(F_e + F_v,neighbor_dj[i,j])
        node_force[i,:] += np.dot(F_e + F_v,node_di[i,:])

def explicitIntegrate():
    for i in range(node_num):
        if node_fixed[i] == False:
            node_acc[i] = node_force[i] / node_mass[i]
            node_pos[i] += dt * node_vel[i] + (node_acc[i] * dt * dt * 0.5)
        if node_pos[i,1] < 0:
            node_pos[i,1] = 0
    updateForce()
    for i in range(node_num):
        if node_fixed[i] == False:
            newacc = node_force[i] / node_mass[i]
            node_vel[i] += (newacc + node_acc[i] * dt * dt * 0.5)
dt = 1 
time = 0
timeFinal = 1

kNearest()
computeFactor()
while(time < timeFinal):
    time += dt
    updateForce()
    explicitIntegrate()
        