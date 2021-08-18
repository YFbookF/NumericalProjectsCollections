import numpy as np

f = open("armadillo_4k.ele","r")   #设置文件对象
line = f.readline()
num_ele = 3717
element4 = np.zeros((num_ele,4),dtype = int)
cnt = 0
while line:   
   line = f.readline()
   st = line.split(' ')
   idx = -1
   for i in range(len(st)):
       if len(st[i]) > 0:
           if idx >= 0:
               element4[cnt,idx] = int(st[i])
           idx += 1
   cnt += 1
   if cnt >= num_ele:
       break
   
f = open("armadillo_4k.node","r")   #设置文件对象
line = f.readline()
num_nodes = 1180
node_pos = np.zeros((num_nodes,3))
cnt = 0
while line:   
   line = f.readline()
   st = line.split(' ')
   idx = -1
   for i in range(len(st)):
       if len(st[i]) > 0:
           if idx >= 0:
               node_pos[cnt,idx] = float(st[i])
           idx += 1
   cnt += 1
   if cnt >= num_nodes:
       break
   
f.close()

density = 1000
E = 1e6
nu = 0.33
mu = E / (2 * (1 + nu))
Lambda = (E * nu) / ((1 + nu) * (1 - 2 * nu))

angle = 180
anchors = np.array([811,0.02])

rotationMatrix = np.array([[-1,0,0],[0,1,0],[0,0,-1]],dtype = float)
for i in range(num_nodes):
    scale = 0.1
    node_pos[i,:] = np.dot(rotationMatrix,node_pos[i,:]) * scale
    
triplets_D = np.zeros((num_ele*9*4))
triplets_D_row = np.zeros((num_ele*9*4),dtype = int)
triplets_D_col = np.zeros((num_ele*9*4),dtype = int)
triplets_D_cnt = 0

triplets_K = np.zeros((num_ele*9))
triplets_K_row = np.zeros((num_ele*9),dtype = int)
triplets_K_col = np.zeros((num_ele*9),dtype = int)
triplets_K_cnt = 0

triplets_M = np.zeros((num_ele*4))
triplets_M_row = np.zeros((num_ele*4),dtype = int)
triplets_M_col = np.zeros((num_ele*4),dtype = int)
triplets_M_cnt = 0

Kreal = np.zeros((num_ele))
Dt = np.zeros((num_ele,4,3))
Dm_inv = np.zeros((num_ele,3,3))
rest_volume = np.zeros((num_ele))
invMass = np.zeros((num_ele))

dt = 0.01
for tet in range(num_ele):
    it = element4[tet,:]
    
    # 为了在之后计算Deformation Gradient
    Dm = np.zeros((3,3))
    Dm[0,:] = node_pos[it[1],:] - node_pos[it[0],:]
    Dm[1,:] = node_pos[it[2],:] - node_pos[it[0],:]
    Dm[2,:] = node_pos[it[3],:] - node_pos[it[0],:]
    
    rest_volume[tet] = 1 / 6 * np.linalg.det(Dm)
    assert(rest_volume[tet] >= 0)
    Dm_inv[tet,:,:] = np.linalg.inv(Dm)
    
    Kreal[tet] = 2 * dt * dt * mu * rest_volume[tet]
    
    for i in range(9):
        idx = 9 * tet + i
        triplets_K[triplets_K_cnt] = Kreal[tet]
        triplets_K_row[triplets_K_cnt] = idx
        triplets_K_col[triplets_K_cnt] = idx
        triplets_K_cnt += 1
        
    for i in range(4):
        idx = it[i]
        invMass[idx] += density / 4 * rest_volume[tet]
        triplets_M[triplets_M_cnt] = invMass[idx]
        triplets_M_row[triplets_M_cnt] = idx
        triplets_M_col[triplets_M_cnt] = idx
        triplets_M_cnt += 1
        
    # 注意转置
    for i in range(3):
        Dt[tet,0,i] = - Dm_inv[tet,i,0] - Dm_inv[tet,i,1] - Dm_inv[tet,i,2]
        
    for j in range(1,4):
        for k in range(3):
            Dt[tet,j,k] = Dm_inv[tet,k,j-1]
    
    for i in range(4):
        for j in range(3):
            triplets_D[triplets_D_cnt] = Dt[tet,i,j]
            triplets_D_col[triplets_D_cnt] = 9 * tet + 3 * j
            triplets_D_row[triplets_D_cnt] = it[i]
            triplets_D_cnt += 1
            
vecSize = int(num_ele / 8) + 1
quat = np.zeros((vecSize,4,8))
for i in range(vecSize):
    quat[i,3,:] = 1 # w



nFixedVertices = 20
for i in range(num_ele):
    if i > nFixedVertices and abs(invMass[i]) > 1e-10:
        invMass[i] = 1 / invMass[i]
    else:
        invMass[i] = 0
            
# ConstrainGraphColoring

volume_constraint_phase = np.zeros((60,200),dtype = int)
particleColors = np.zeros((60,num_nodes),dtype = bool)
colorSize = 0

# 下面这段，就是把四面体划分为60大块，每块之中的四面体都不能共点，共一个点也不可以

for i in range(num_ele):
    newColor = True
    
    for j in range(colorSize):
        addToThisColor = True
        
        for k in range(4):
            if particleColors[j,element4[i,k]] == True:
                addToThisColor = False
                break
        
        if addToThisColor == True:
            idx = volume_constraint_phase[j,199]
            volume_constraint_phase[j,idx] = i
            volume_constraint_phase[j,199] += 1
            
            for k in range(4):
                particleColors[j,element4[i,k]] = True
            newColor = False
            break
        
    if newColor == True:
        particleColors[colorSize,:] = False
        idx = volume_constraint_phase[colorSize,199]
        volume_constraint_phase[colorSize,idx] = i
        volume_constraint_phase[colorSize,199] += 1
        for k in range(4):
            particleColors[colorSize,element4[i,k]] = True
        colorSize += 1            
            
# initializeVolumeConstraints
inv_mass_phase = np.zeros((colorSize,200,32))
rest_volume_phase = np.zeros((colorSize,200,8))
kappa_phase = np.zeros((colorSize,200,8))
alpha_phase = np.zeros((colorSize,200,8))
for phase in range(colorSize):
    
    phase_size = volume_constraint_phase[phase,199]
    for c in range(phase_size):
        
        c8 = np.zeros((8),dtype = int)
        for k in range(8):
            if c + k < phase_size:
                c8[k] = volume_constraint_phase[phase,c+k]
        
        w0 = np.zeros((8))
        w1 = np.zeros((8))
        w2 = np.zeros((8))
        w3 = np.zeros((8))
        vol = np.zeros((8))
        alpha = np.zeros((8))
        
        for k in range(8):
            if c + k < phase_size:
                w0[k] = invMass[element4[c8[k],0]]
                w1[k] = invMass[element4[c8[k],1]]
                w2[k] = invMass[element4[c8[k],2]]
                w3[k] = invMass[element4[c8[k],3]]
                
                vol[k] = rest_volume[c8[k]]
                alpha[k] = 1 / (Lambda + vol[k] * dt * dt)
            else:
                vol[k] = 1
                alpha[k] = 0
                
                w0[k] = invMass[element4[c8[k],0]]
                w1[k] = invMass[element4[c8[k],1]]
                w2[k] = invMass[element4[c8[k],2]]
                w3[k] = invMass[element4[c8[k],3]]
    
        pos = int(inv_mass_phase[phase,199,0])
        inv_mass_phase[phase,pos,0:8] = w0[:]
        inv_mass_phase[phase,pos,8:16] = w1[:]
        inv_mass_phase[phase,pos,16:24] = w2[:]
        inv_mass_phase[phase,pos,24:32] = w3[:]
        inv_mass_phase[phase,199,0] += 1
        
        pos = int(rest_volume_phase[phase,199,0])
        rest_volume_phase[phase,pos,:] = vol[:]
        rest_volume_phase[phase,199,0] += 1
        
        pos = int(kappa_phase[phase,199,0])
        kappa_phase[phase,pos,:] = 0
        kappa_phase[phase,199,0] += 1
        
        pos = int(alpha_phase[phase,199,0])
        alpha_phase[phase,pos,:] = alpha[:]
        alpha_phase[phase,199,0] += 1
        
        
def computeDeformationGradient(i):
    # 一次性计算 8 个四面体的deformation Gradient
        regluarPart = int(num_ele / 8) * 8
        i8 = i * 8
        vertices = np.zeros((4,3,8))
        if i8 < regluarPart:
            for j in range(4):
                p0 = node_pos[element4[i8 + 0,j],:]
                p1 = node_pos[element4[i8 + 1,j],:]
                p2 = node_pos[element4[i8 + 2,j],:]
                p3 = node_pos[element4[i8 + 3,j],:]
                p4 = node_pos[element4[i8 + 4,j],:]
                p5 = node_pos[element4[i8 + 5,j],:]
                p6 = node_pos[element4[i8 + 6,j],:]
                p7 = node_pos[element4[i8 + 7,j],:]
                
                vertices[j,0,:] = np.array([p0[0],p1[0],p2[0],p3[0],p4[0],p5[0],p6[0],p7[0]])
                vertices[j,1,:] = np.array([p0[1],p1[1],p2[1],p3[1],p4[1],p5[1],p6[1],p7[1]])
                vertices[j,2,:] = np.array([p0[2],p1[2],p2[2],p3[2],p4[2],p5[2],p6[2],p7[2]])
                
        else:
            
            for j in range(4):
                p0 = np.zeros((3,8))
                for k in (regluarPart,regluarPart+8):
                    if k < num_ele:
                        p0[k - regluarPart,:] = node_pos[element4[k,j],:]
                    else:
                        p0[k - regluarPart,:] = node_pos[element4[num_ele-1,j]]
                        
                vertices[j,:,:] = p0[:,:]
                
        F1 = np.zeros((3,8))
        F2 = np.zeros((3,8))
        F3 = np.zeros((3,8))
        for j in range(4):
            for k in range(8):
                F1[0,k] += vertices[j,0,k] * Dt[i*8+k,j,0]
                F1[1,k] += vertices[j,1,k] * Dt[i*8+k,j,0]
                F1[2,k] += vertices[j,2,k] * Dt[i*8+k,j,0]
                
                F2[0,k] += vertices[j,0,k] * Dt[i*8+k,j,1]
                F2[1,k] += vertices[j,1,k] * Dt[i*8+k,j,1]
                F2[2,k] += vertices[j,2,k] * Dt[i*8+k,j,1]
                
                F3[0,k] += vertices[j,0,k] * Dt[i*8+k,j,2]
                F3[1,k] += vertices[j,1,k] * Dt[i*8+k,j,2]
                F3[2,k] += vertices[j,2,k] * Dt[i*8+k,j,2]
        return F1,F2,F3
        
def QuatMultiply(quat0,quat1):
    # 4 x scaler 8
    res = np.zeros((4,8))
    res[0,:] = quat0[3,:] * quat1[0,:] + quat0[0,:] * quat1[3,:] + quat0[1,:] * quat1[2,:] - quat0[2,:] * quat1[1,:]
    res[1,:] = quat0[3,:] * quat1[1,:] - quat0[0,:] * quat1[2,:] + quat0[1,:] * quat1[3,:] + quat0[2,:] * quat1[0,:]
    res[2,:] = quat0[3,:] * quat1[2,:] + quat0[0,:] * quat1[1,:] - quat0[1,:] * quat1[0,:] + quat0[2,:] * quat1[3,:]
    res[3,:] = quat0[3,:] * quat1[3,:] - quat0[0,:] * quat1[0,:] - quat0[1,:] * quat1[1,:] - quat0[2,:] * quat1[2,:]
    return res
    
def QuatToRotationMatrix(q):
    tx = 2 * q[0,:]
    ty = 2 * q[1,:]
    tz = 2 * q[2,:]
    twx = tx * q[3,:]
    twy = ty * q[3,:]
    twz = tz * q[3,:]
    txx = tx * q[0,:]
    txy = ty * q[0,:]
    txz = tz * q[0,:]
    tyy = tx * q[1,:]
    tyz = ty * q[1,:]
    tzz = tz * q[2,:]
    
    R = np.zeros((3,3,8))
    R[0,0,:] = 1 - (tyy + tzz)
    R[0,1,:] = txz - twz
    R[0,2,:] = twz + twy
    R[1,0,:] = txy + twz
    R[1,1,:] = 1 - (txx + tzz)
    R[1,2,:] = tyz - twx
    R[2,0,:] = txz - twy
    R[2,1,:] = tyz + twx
    R[2,2,:] = 1 - (txx + tyy)
    
    return R

def SolveOptimizationProblem(pos):
    x = pos.copy()
    rhs_size = num_nodes - nFixedVertices
    rhs = np.zeros((rhs_size,8))
    
    for i in range(vecSize):
        
        F1,F2,F3 = computeDeformationGradient(i)
        
        for ite in range(1):
            
            Ro = QuatToRotationMatrix(quat[i,:,:])
                 
            # F1 是 3 x 8
            # R 是 3 x 3 x 8
            # B 是 left cauchy tensor
            B0 = np.zeros((3,8))
            B1 = np.zeros((3,8))
            B2 = np.zeros((3,8))
            for k in range(8):
                rt = np.transpose(Ro[:,:,k])
                B0[:,k] = np.dot(rt,F1[:,k])
                B1[:,k] = np.dot(rt,F2[:,k])
                B2[:,k] = np.dot(rt,F3[:,k])
            
            gradient = np.array([B2[1] - B1[2],B0[2] - B2[0],B1[0] - B0[1]])
            
            # 计算对称的 Hessian 矩阵
            h00 = B1[1] + B2[2]
            h11 = B0[0] + B2[2]
            h22 = B0[0] + B1[1]
            h01 = (B1[0] + B0[1]) * (-0.5)
            h02 = (B2[0] + B0[2]) * (-0.5)
            h12 = (B2[1] + B1[2]) * (-0.5)
            
            detH =  - h02 * h02 * h11 + 2 * h01 * h02 * h12 - h00 * h12 * h12 - h01 * h01 * h22 + h00 * h11 * h22
            
            omega = np.zeros((3,8))
            factor = -0.25 / detH
            omega[0] = (h11*h22-h12*h12) * gradient[0] + (h02*h12-h01*h22) * gradient[1] + (h01*h12-h02*h11) * gradient[2]
            omega[1] = (h02*h12-h01*h22) * gradient[0] + (h00*h22-h02*h02) * gradient[1] + (h01*h02-h00*h12) * gradient[2]
            omega[2] = (h01*h12-h02*h11) * gradient[0] + (h01*h02-h00*h12) * gradient[1] + (h00*h11-h01*h01) * gradient[2]
            omega[0] *= factor
            omega[1] *= factor
            omega[2] *= factor
            
            gradient = (1 - omega) * ( -gradient ) + omega * gradient
            
            useGd = np.ones((8))
            # 此处有一段没看懂
            l_omega2 = np.zeros((8))
            for k in range(8):
                l_omega2[k] = np.sqrt(omega[0,k]**2 + omega[1,k]**2 + omega[2,k]**2)
            
            vec = np.zeros((4,8))
            for k in range(3):
                vec[k,:] = omega[k,:] * (2 / (1 + l_omega2[:]))
                
            vec[3,:] = (1 - l_omega2) / (1 + l_omega2)
            
            q = QuatMultiply(quat[i,:,:],vec)
            
                                 
                
            test = 1
    
            
        
time = 0
dt = 0.01
timeFinal = 0.001
node_vel = np.zeros((num_nodes,3))
node_pos_old = np.zeros((num_nodes,3))
while(time < timeFinal):
    time = time + dt
    
    for  i in range(nFixedVertices):
        node_vel[i,2] -= dt * 9.81
        node_pos_old[i,:] = node_pos[i,:]
        node_pos[i,:] += dt * node_vel[i,:]
    
    # Solve Optimzation
    SolveOptimizationProblem(node_pos)