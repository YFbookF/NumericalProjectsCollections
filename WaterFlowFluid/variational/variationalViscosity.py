import numpy as np
import random
# https://github.com/christopherbatty/VariationalViscosity2D
# SCA 2008 paper "Accurate Viscous Free Surfaces[...]" by Batty & Bridson
Nx = 32
Ny = 32
dx = 1 / Nx

u = np.zeros((Nx+1,Ny))
utemp = np.zeros((Nx+1,Ny))
v = np.zeros((Nx+1,Ny))
vtemp = np.zeros((Nx+1,Ny))

solidphi = np.zeros((Nx+1,Ny+1))
liquidphi = np.zeros((Nx+1,Ny+1))

particleRadius = dx / np.sqrt(2) / 10
viscosity = np.ones((Nx,Ny))

circle = np.array([[0.3,0.3],[0.75,0.75]])
rad = np.array([0.1,0.1])

particleCount = 1
particlePos = np.array([0.5,0.5])

def lerp(a,b,x):
    return (1 - x) * a + x * b

def bilerp(a,b,c,d,x,y):
    return lerp(lerp(a, b, x),lerp(c, d, x),y)

def getVelocity():
    ix = int(pos[0] / dx)
    iy = int(pos[1] / dx)
    fx = pos[0] - ix
    fy = pos[1] - iy
    v00 = u[ix,iy]
    v10 = u[ix+1,iy]
    v01 = u[ix,iy+1]
    v11 = u[ix+1,iy+1]
    uval = bilerp(v00,v10,v01,v11,fx, fy)
    v00 = v[ix,iy]
    v10 = v[ix+1,iy]
    v01 = v[ix,iy+1]
    v11 = v[ix+1,iy+1]
    vval = bilerp(v00,v10,v01,v11,fx, fy)
    return np.array([uval,vval])
    
def Interpolate(pos,field):
    ix = int(pos[0] / dx)
    iy = int(pos[1] / dx)
    fx = pos[0] - ix
    fy = pos[1] - iy
    v00 = field[ix,iy]
    v10 = field[ix+1,iy]
    v01 = field[ix,iy+1]
    v11 = field[ix+1,iy+1]
    return bilerp(v00,v10,v01,v11,fx, fy)
    

def computeVolumeFraction(levelset,fraction,origin,subdivision):
    
    subdx = 1 / subdivision
    samplemax = subdivision * subdivision
    for j in range(Ny):
        for i in range(Nx):
            startx = origin[0] + i
            starty = origin[1] + j
            incount = 0
            for subj in range(subdivision):
                for subi in range(subdivision):
                    pos = np.array([startx + (subi + 0.5) * subdx,
                            starty + (subi + 0.5) * subdx])
                    phival = Interpolate(pos, levelset)
                    if phival < 0:
                        incount += 1
            fraction[i,j] = incount / samplemax
                    
def fractionInside2(phiLeft,phiRight):
    if (phiLeft < 0) & (phiRight < 0):
        return 1
    if (phiLeft < 0) & (phiRight >= 0):
        return phiLeft / (phiLeft - phiRight)
    if (phiLeft >= 0) & (phiRight < 0):
        return phiRight / (phiRight - phiLeft)
    return 0

def clamp(x):
    if x < 0:
        return 0
    elif x > 1:
        return 1
    else:
        return x

for i in range(Nx):
    for j in range(Ny):
        phi = 1000
        for k in range(circle.shape[0]):
            phi = min(phi,np.sqrt((i*dx - circle[k,0])**2 + (j*dx - circle[k,1])**2) - rad[k])
        solidphi[i,j] = phi
        
for i in range(Nx//2,Nx):
    for j in range(Ny//4,Ny//3):
        jitterx = random.random() * dx
        jittery = random.random() * dx
        if solidphi[i,j] > 0:
            newPos = np.array([i*dx + jitterx,j*dx + jittery])
            particlePos = np.vstack((particlePos,newPos))
            particleCount += 1
            
particleVelocity = np.array((particleCount,2))
time = 0
timeFinal = 1
while(time < timeFinal):
    
    # 对流粒子
    for pidx in range(particleCount):
        # Advection Velocity
        
        
        # puts Particles back in fluid domain
        ppos = particlePos[pidx,:]
        ix = int(ppos[0] / dx)
        iy = int(ppos[1] / dx)
        fx = ppos[0] - ix
        fy = ppos[1] - iy
        v00 = solidphi[ix,iy]
        v10 = solidphi[ix+1,iy]
        v01 = solidphi[ix,iy+1]
        v11 = solidphi[ix+1,iy+1]
        phi = bilerp(v00,v10,v01,v11,fx, fy)
        if phi < 0:
            ddy0 = v01 - v00
            ddy1 = v11 - v10
            ddx0 = v10 - v00
            ddx1 = v11 - v01
            normal = np.array([lerp(ddx0, ddx1, fx),lerp(ddy0, ddy1, fy)])
            norm = np.sqrt(normal[0]**2 + normal[1]**2)
            particlePos[pidx,:] -= phi * normal / norm
            
    # 计算phi
    for pidx in range(particleCount):
        ppos = particlePos[pidx,:]
        ix = int(ppos[0]/dx - 0.5)
        iy = int(ppos[1]/dx - 0.5)
        
        for i in range(ix-2,ix+3):
            for j in range(iy-2,iy+3):
                if (i < 0) | (i >= Nx):
                    continue
                if (j < 0) | (j >= Ny):
                    continue
                gridpos = np.array([(i+0.5)*dx,(j+0.5)*dx])
                phi = np.sqrt((gridpos[0] - ppos[0])**2 + (gridpos[1] - ppos[1])**2) - 1.02 * particleRadius
                liquidphi[i,j] = min(liquidphi[i,j],phi)
    
    # 如果是固体边界
    for j in range(Ny):
        for i in range(Nx):
            if liquidphi[i,j] < dx / 2:
                solidphival = (solidphi[i,j] + solidphi[i+1,j] + solidphi[i,j+1] + solidphi[i+1,j+1]) / 4
                if solidphival < 0:
                    liquidphi[i,j] = - dx / 2
                    
    # 对流速度，半隐式Lagrange
    for j in range(Ny):
        for i in range(Nx):
            dt = 1
            pos = np.array([i*dx,(j+0.5)*dx])
            pos -= dt * getVelocity(pos)
            utemp[i,j] = getVelocity(pos)
            pos = np.array([(i+0.5)*dx,*dx])
            pos -= dt * getVelocity(pos)
            vtemp[i,j] = getVelocity(pos)
            
    u = utemp.copy()
    v = vtemp.copy()
    # 应用重力
    v[:,:] -= 1
    cvol = np.zeros((Nx,Ny))
    nvol = np.zeros((Nx,Ny))
    uvol = np.zeros((Nx,Ny))
    vvol = np.zeros((Nx,Ny))
    # 添加可变粘性，首先计算权重
    computeVolumeFraction(liquidphi, cvol, [-0.5,-0.5], 2)
    computeVolumeFraction(liquidphi, nvol, [-1,-1], 2)
    computeVolumeFraction(liquidphi, uvol, [-1,-0.5], 2)
    computeVolumeFraction(liquidphi, vvol, [-0.5,-1], 2)
    
    # 计算粘性
    
    # 速度状态，1是固体，0是液体
    ustate = np.zeros((Nx+1,Ny))
    vstate = np.zeros((Nx,Ny+1))
    for j in range(Ny):
        for i in range(Nx+1):
            if (i == 0) | (i == Nx) | ((solidphi[i,j+1] + solidphi[i,j])/2 <= 0):
                ustate[i,j] = 1
    for j in range(Ny+1):
        for i in range(Nx):
            if (j == 0) | (j == Nx) | ((solidphi[i+1,j] + solidphi[i,j])/2 <= 0):
                vstate[i,j] = 1
                
    # 计算矩阵
    elts = (Nx + 1) * Ny + Nx * (Nx + 1)
    vrhs = np.zeros((elts))
    velocities = np.zeros((elts))
    vmatrix = np.zeros((elts,elts))
    factor = dt / np.sqrt(dx)
    
    for j in range(1,Nx - 1):
        for i in range(1,Nx - 1):
            if ustate[i,j] == 0:
                idx = j * (Nx + 1) + i
                vrhs[idx] = uvol[i,j] * u[i,j]
                vmatrix[idx,idx] = uvol[i,j]
                
                visc_right = viscosity[i,j]
                visc_left = viscosity[i-1,j]
                vol_right = cvol[i,j]
                vol_left = cvol[i-1,j]
                
                vmatrix[idx,idx] += 2*factor*visc_right*vol_right
                if ustate[i+1,j] == 0:
                    vmatrix[idx+1,idx] -= 2*factor*visc_right*vol_right
                else:
                    vrhs[idx] += 2*factor*visc_right*vol_right
            
    
    # 计算压力
    u_weight = np.zeros((Nx,Ny))
    v_weight = np.zeros((Nx,Ny))
    for j in range(Ny):
        for i in range(Nx):
            temp = 1 - fractionInside2(solidphi[i,j+1],solidphi[i,j])
            u_weight[i,j] = clamp(temp)
            temp = 1 - fractionInside2(solidphi[i+1,j],solidphi[i,j])
            v_weight[i,j] = clamp(temp)
            
    rhs = np.zeros((Nx*Ny))
    pressure = np.zeros((Nx*Ny))
    matrix = np.zeros((Nx*Ny,Nx*Ny))
    
    for j in range(1,Ny-1):
        for i in range(1,Nx-1):
            idx = Nx * j + i
            rhs[idx] = 0
            pressure[idx] = 0
            centrephi = liquidphi[i,j]
            if centrephi < 0:
                
                # Right neighbour
                term = u_weight[i+1,j] * dt / np.sqrt(dx)
                phi_right = liquidphi[i+1,j]
                if phi_right < 0:
                    matrix[idx,idx] += term
                    matrix[idx,idx+1] += -term
                else:
                    theta = fractionInside2(centrephi, phi_right)
                    if theta < 0.01:
                        theta = 0.01
                    matrix[idx,idx] += term / theta
                rhs[idx] -= u_weight[i+1,j]*u[i+1,j] / dx
                
                # Left neighbour
                
                # Top neighbour
                
                # Bottom neighbour