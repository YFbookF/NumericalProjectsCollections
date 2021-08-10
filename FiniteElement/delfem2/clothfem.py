import numpy as np

num_axis = 25
elem_length = 1 / num_axis
Nx = num_axis + 1
num_nodes = Nx * Nx
coord = np.zeros((num_nodes,3))
for i in range(Nx):
    for j in range(Nx):
        idx = j * Nx + i
        coord[idx,0] = i * elem_length
        coord[idx,1] = j * elem_length
        
element3 = np.zeros((num_axis*num_axis*2,3),dtype = int)
for j in range(num_axis):
    for i in range(num_axis):
        idx = (j * num_axis + i) * 2
        element3[idx,0] = j * Nx + i
        element3[idx,1] = (j + 1) * Nx + i
        element3[idx,2] = j * Nx + i + 1
        
        element3[idx+1,0] = (j + 1) * Nx + i + 1
        element3[idx+1,1] = j * Nx + i + 1
        element3[idx+1,2] = (j + 1) * Nx + i
        
element4 = np.zeros((num_axis*num_axis + num_axis*(num_axis-1)*2,4))
idx = 0
for j in range(num_axis):
    for i in range(num_axis):
       element4[idx,0] = j * Nx + i
       element4[idx,1] = (j + 1) * Nx + i + 1
       element4[idx,2] = (j + 1) * Nx + i 
       element4[idx,3] = j * Nx + i + 1
       idx += 1
for j in range(num_axis):
    for i in range(num_axis-1):
       element4[idx,0] = (j + 1) * Nx + i
       element4[idx,1] = j * Nx + i + 2
       element4[idx,2] = (j + 1) * Nx + i + 1 
       element4[idx,3] = j * Nx + i + 1
       idx += 1
for j in range(num_axis-1):
    for i in range(num_axis):
       element4[idx,0] = j * Nx + i + 1
       element4[idx,1] = (j + 2) * Nx + i
       element4[idx,2] = (j + 1) * Nx + i 
       element4[idx,3] = (j + 1) * Nx + i + 1
       idx += 1

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

def triArea3d(vec1,vec2,vec3):
    x = (vec2[1] - vec1[1]) * (vec3[2] - vec1[2]) - (vec3[1] - vec1[1]) * (vec2[2] - vec1[2])
    y = (vec2[2] - vec1[2]) * (vec3[0] - vec1[0]) - (vec3[2] - vec1[2]) * (vec2[0] - vec1[0])
    z = (vec2[0] - vec1[0]) * (vec3[1] - vec1[1]) - (vec3[0] - vec1[0]) * (vec2[1] - vec1[1])
    return np.sqrt(x*x + y*y + z*z) / 2

def wdwddw_cst(C,c):
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
    return e,dW,ddW

def WdWddW_Bend(C,c):
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
            
    return W,dW,ddW
clothsize = 1
mass_point = clothsize * clothsize / num_nodes
coord_init = coord.copy()
vel = np.zeros((num_nodes,0))

time = 0
timeFinal = 1
while(time < timeFinal):
    time = 2
    C = np.zeros((3,3))
    c = np.zeros((3,3))
    energy = 0
    d_energy = np.zeros((num_nodes,3))
    dd_energy = np.zeros((num_nodes,num_nodes,9))
    
    for ie in range(element3.shape[0]):
        ind = element3[ie,:]
        for i in range(3):
            C[i,:] = coord_init[ind[i],:]
            c[i,:] = coord[ind[i],:]
        e,de,dde = wdwddw_cst(C, c)
        energy += e
        for i in range(3):
            d_energy[ind[i],:] += de[i,:]
        for i in range(3):
            for j in range(3):
                dd_energy[ind[i],ind[j],:] += dde[j*3+i:]
    
    C = np.zero((4,3))
    c = np.zeros((4,3))
    for ie in range(element4.shape[0]):
        ind = element4[ie,:]
        for i in range(4):
            C[i,:] = coord_init[ind[i],:]
            c[i,:] = coord[ind[i],:]
        e,de,dde = WdWddW_Bend(C, c)
        energy += e
        for i in range(3):
            d_energy[ind[i],:] += de[i,:]
        for i in range(4):
            for j in range(4):
                dd_energy[ind[i],ind[j],:] += dde[j*4+i,:]
    
    for ie in range(num_nodes):
        stiff_contact = 1000
        contact_tolerance = 0.02
        pd = contact_tolerance - 100
        if pd > 0:
            pd = 0
            # do something
            energy += stiff_contact * pd * pd / 2
            # and something
            
    for ie in range(num_nodes):
        pos = coord[ie,:]
        gravity = np.array([0,-1,0])
        energy -= mass_point * dot(pos,gravity)
        d_energy[ie,:] -= mass_point * gravity
                
    dt = 0.02
    for ie in range(num_nodes):
        d_energy[i,:] = - d_energy[i,:] + mass_point * vel[i,:] / dt
        
    for i in range(num_nodes): # 索引怎么只有易班
        dd_energy[i,0] += mass_point / dt / dt
        dd_energy[i,4] += mass_point / dt / dt
        dd_energy[i,8] += mass_point / dt / dt
            
    # 然后共轭梯度算什么东西