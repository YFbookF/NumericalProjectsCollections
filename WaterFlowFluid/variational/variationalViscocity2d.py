import numpy as np
import random
# https://github.com/christopherbatty/VariationalViscosity2D
# SCA 2008 paper "Accurate Viscous Free Surfaces[...]" by Batty & Bridson
# Fluid Ver.
# 目前的代码解压力泊松方程之前有一点问题
Nx = 32
Ny = 32
dx = 1 / Nx

u = np.zeros((Nx+1,Ny))
u_weights = np.zeros((Nx+1,Ny))
u_valid = np.zeros((Nx+1,Ny),dtype = bool)
v = np.zeros((Nx,Ny+1))
v_weights = np.zeros((Nx,Ny+1))
v_valid = np.zeros((Nx,Ny+1),dtype = bool)

solidphi = np.zeros((Nx+1,Ny+1))
liquidphi = np.zeros((Nx+1,Ny+1))


Amat = np.zeros((Nx * Nx,5)) # 自身，i-1,i+1,j-1,j+1
rhs = np.zeros((Nx * Nx))
pressure = np.zeros((Nx * Nx))

particleRadius = dx / np.sqrt(2)


circle = np.array([[0.5,0.5],[0.7,0.5],[0.3,0.35],[0.5,0.7]])
rad = np.array([0.4,0.1,0.1,0.1])

viscosity = np.ones((Nx,Ny))
c_vol = np.zeros((Nx,Ny))
n_vol = np.zeros((Nx,Ny))
u_vol = np.zeros((Nx,Ny))
v_vol = np.zeros((Nx,Ny))

particleCount = 8
particlePos = np.array([[0.801373184,0.419566065],
                        [0.727583468,0.721774817],
                        [0.562456191,0.628744423],
                        [0.664881229,0.472821087],
                        [0.551453352,0.835115790],
                        [0.753610969,0.485379040],
                        [0.575982809,0.225956142],
                        [0.646220088,0.148354426]])

def lerp(a,b,x):
    return (1 - x) * a + x * b

def bilerp(a,b,c,d,x,y):
    return lerp(lerp(a, b, x),lerp(c, d, x),y)

def getFieldInDomain(x,y,field):
    nx = field.shape[0]
    ny = field.shape[1]
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x > nx - 1:
        x = nx - 1
    if y > ny - 1:
        y = ny - 1
    return field[x,y]

def getVelocity(pos):
    gx = pos[0] / dx - 0.5
    gy = pos[1] / dx
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    v00 = getFieldInDomain(ix,iy,u)
    v10 = getFieldInDomain(ix+1,iy,u)
    v01 = getFieldInDomain(ix,iy+1,u)
    v11 = getFieldInDomain(ix+1,iy+1,u)
    uval = bilerp(v00,v10,v01,v11,fx, fy)
    gx = pos[0] / dx 
    gy = pos[1] / dx - 0.5
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    v00 = getFieldInDomain(ix,iy,v)
    v10 = getFieldInDomain(ix+1,iy,v)
    v01 = getFieldInDomain(ix,iy+1,v)
    v11 = getFieldInDomain(ix+1,iy+1,v)
    vval = bilerp(v00,v10,v01,v11,fx, fy)
    return np.array([uval,vval])
    
def Interpolate(pos,field):
    gx = pos[0] / dx 
    gy = pos[1] / dx
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    v00 = field[ix,iy]
    v10 = field[ix+1,iy]
    v01 = field[ix,iy+1]
    v11 = field[ix+1,iy+1]
    return bilerp(v00,v10,v01,v11,fx, fy)

def Interpolate_Gradient(pos,field):
    gx = pos[0] / dx 
    gy = pos[1] / dx
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    v00 = getFieldInDomain(ix,iy,field)
    v10 = getFieldInDomain(ix+1,iy,field)
    v01 = getFieldInDomain(ix,iy+1,field)
    v11 = getFieldInDomain(ix+1,iy+1,field)
    ddy0 = v01 - v00
    ddy1 = v11 - v10
    ddx0 = v10 - v00
    ddx1 = v11 - v01
    return np.array([lerp(ddx0, ddx1, fx),lerp(ddy0, ddy1, fy)])
                    
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
    
def cfl():
    maxvel = 1e-10
    for i in range(Nx+1):
        for j in range(Nx):
            maxvel = max(maxvel,abs(u[i,j]))
            maxvel = max(maxvel,abs(v[j,i]))
    return dx / maxvel

def advect_particles(substep):
    for pidx in range(particleCount):
        
        ppos = particlePos[pidx,:]
        vel0 = getVelocity(ppos)
        ppos1 = ppos + vel0 * substep
        vel1 = getVelocity(ppos1)
        ppos2 = ppos1 + vel1 * substep
        particlePos[pidx,:] = (ppos + ppos2) / 2
        
        phi = Interpolate(particlePos[pidx,:], solidphi)
        
        if phi < 0:
            
            gx = particlePos[pidx,0] / dx 
            gy = particlePos[pidx,1] / dx 
            ix = int(gx)
            iy = int(gy)
            fx = gx - ix
            fy = gy - iy
            
            v00 = solidphi[ix,iy]
            v01 = solidphi[ix,iy+1]
            v10 = solidphi[ix+1,iy]
            v11 = solidphi[ix+1,iy+1]
            
            ddy0 = v01 - v00
            ddy1 = v11 - v10
            ddx0 = v10 - v00
            ddx1 = v11 - v01
            
            normal = np.array([lerp(ddx0, ddx1, fx),lerp(ddy0, ddy1, fy)])
            norm = np.sqrt(normal[0]**2 + normal[1]**2)
            particlePos[pidx,:] -= phi * normal / norm
            
def advect_grid_velocity(substep):
    global u
    global v
    temp_u = u.copy()
    temp_v = v.copy()
    for i in range(Nx+1):
        for j in range(Ny):
            pos = np.array([i*dx,(j+0.5)*dx])
            vel0 = getVelocity(pos)
            pos1 = pos - vel0 * substep
            vel1 = getVelocity(pos1)
            pos2 = pos1 - vel1 * substep
            vel2 = getVelocity(pos2)
            temp_u[i,j] = ((vel0 + vel2)/2)[0]
    for i in range(Nx):
        for j in range(Ny+1):
            pos = np.array([(i+0.5)*dx,j*dx])
            vel0 = getVelocity(pos)
            pos1 = pos - vel0 * substep
            vel1 = getVelocity(pos1)
            pos2 = pos1 - vel1 * substep
            vel2 = getVelocity(pos2)
            temp_v[i,j] = ((vel0 + vel2)/2)[1]
    u = temp_u.copy()
    v = temp_v.copy()

def in_domain(x,y):
    if x < 0 or x >= Nx or y < 0 or y >= Nx:
        return False
    return True

def compute_liquidphi():
    
    liquidphi[:,:] = 3 * dx
    
    for pidx in range(particleCount):
        ppos = particlePos[pidx,:]
        ix = int(ppos[0] / dx - 0.5)
        iy = int(ppos[1] / dx - 0.5)
        for j in range(iy-2,iy+3):
            for i in range(ix-2,ix+3):
                if in_domain(i, j):
                    pos = np.array([(i + 0.5)*dx,(j+0.5)*dx])
                    phi_temp = np.sqrt((pos[0] - ppos[0])**2 + (pos[1] - ppos[1])**2) - 1.02*particleRadius
                    liquidphi[i,j] = min(liquidphi[i,j],phi_temp)
     
    for j in range(Nx):
        for i in range(Ny):
            if liquidphi[i,j] < 0.5*dx:
                solidval = (solidphi[i,j] + solidphi[i+1,j] + solidphi[i,j+1] + solidphi[i+1,j+1])/4
                if solidval < 0:
                    liquidphi[i,j] = -0.5 * dx

def compute_weights():
    
    for i in range(Nx+1):
        for j in range(Nx):
            u_weights[i,j] = clamp(1 - fractionInside2(solidphi[i,j+1], solidphi[i,j]))
            
    for i in range(Nx):
        for j in range(Nx+1):
            v_weights[i,j] = clamp(1 - fractionInside2(solidphi[i+1,j], solidphi[i,j]))
    
def assemble_matrix(substep):
    global Amat
    global rhs
    
    Amat[:,:] = 0
    rhs[:] = 0
    
    for j in range(1,Nx-1):
        for i in range(1,Nx-1):
            idx = j * Nx + i
            centre_phi = liquidphi[i,j]
            if centre_phi < 0:
                
                # right
                term = u_weights[i+1,j] * substep / dx / dx
                right_phi = liquidphi[i+1,j]
                if right_phi < 0: # 在液体区域里
                    Amat[idx,0] += term
                    Amat[idx,2] -= term
                else:
                    theta = fractionInside2(centre_phi, right_phi)
                    if theta < 0.01:
                        theta = 0.01
                    Amat[idx,0] += term / theta
                rhs[idx] -= u_weights[i+1,j] * u[i+1,j] / dx
                
                # left
                term = u_weights[i,j] * dt / dx / dx
                left_phi = liquidphi[i-1,j]
                if left_phi < 0: # 在液体区域里
                    Amat[idx,0] += term
                    Amat[idx,1] -= term
                else:
                    theta = fractionInside2(centre_phi, left_phi)
                    if theta < 0.01:
                        theta = 0.01
                    Amat[idx,0] += term / theta
                rhs[idx] += u_weights[i,j] * u[i,j] / dx
                
                # top
                term = v_weights[i,j+1] * dt / dx / dx
                top_phi = liquidphi[i,j+1]
                if top_phi < 0: # 在液体区域里
                    Amat[idx,0] += term
                    Amat[idx,4] -= term
                else:
                    theta = fractionInside2(centre_phi, top_phi)
                    if theta < 0.01:
                        theta = 0.01
                    Amat[idx,0] += term / theta
                rhs[idx] -= v_weights[i,j+1] * v[i,j+1] / dx
                
                # bottom
                term = v_weights[i,j] * dt / dx / dx
                bottom_phi = liquidphi[i,j-1]
                if bottom_phi < 0: # 在液体区域里
                    Amat[idx,0] += term
                    Amat[idx,3] -= term
                else:
                    theta = fractionInside2(centre_phi, bottom_phi)
                    if theta < 0.01:
                        theta = 0.01
                    Amat[idx,0] += term / theta
                rhs[idx] += v_weights[i,j] * v[i,j] / dx
    
def computeA(i,j,field):
    idx = j * Nx + i
    term = 0
    if i > 0:
        term += field[idx-1] * Amat[idx,1]
    if i < Nx - 1:
        term += field[idx+1] * Amat[idx,2]
    if j > 0:
        term += field[idx-Nx] * Amat[idx,3]
    if j < Nx - 1:
        term += field[idx+Nx] * Amat[idx,4]
    return field[idx] * Amat[idx,0] + term
    
def cg_solver(x):
    direction = np.zeros((Nx * Nx))
    residual = np.zeros((Nx * Nx))
    for j in range(Nx):
        for i in range(Nx):
            idx = j * Nx + i
            direction[idx] = rhs[idx] - computeA(i, j,x)
            residual[idx] = direction[idx]
    
    eps = 1e-5
    for ite in range(10):
        dAd = eps
        for j in range(Nx):
            for i in range(Nx):
                idx = j * Nx + i
                dAd += direction[idx] * computeA(i, j, direction)
        alpha = 0
        for i in range(Nx*Nx):
            alpha += residual[i] * residual[i] / dAd
        beta = 0
        for j in range(Nx):
            for i in range(Nx):
                idx = j * Nx + i
                x[idx] = x[idx] + alpha * direction[idx]
                residual[idx] = residual[idx] - alpha * computeA(i, j, direction)
                beta += residual[idx] * residual[idx] / ((alpha + eps) * dAd)
        for i in range(Nx * Nx):
            direction[i] = residual[i] + beta * direction[i]
        
def substract_gradient():
    
    u_valid[:,:] = False
    v_valid[:,:] = False
    
    for j in range(Nx):
        for i in range(1,Nx):
            idx = j * Nx + i
            if u_weights[i,j] > 0 and (liquidphi[i,j] < 0 or liquidphi[i-1,j] < 0):
                theta = 1
                if liquidphi[i,j] >= 0 or liquidphi[i-1,j] >= 0:
                    theta = fractionInside2(liquidphi[i-1,j],liquidphi[i,j])
                    if theta < 0.01:
                        theta = 0.01
                    u[i,j] -= dt * (pressure[idx] - pressure[idx-1]) / dx / theta
                    u_valid[i,j] = True
            else:
                u[i,j] = 0
    
    for j in range(1,Nx):
        for i in range(Nx):
            idx = j * Nx + i
            if v_weights[i,j] > 0 and (liquidphi[i,j] < 0 or liquidphi[i,j-1] < 0):
                theta = 1
                if liquidphi[i,j] >= 0 or liquidphi[i,j-1] >= 0:
                    theta = fractionInside2(liquidphi[i,j-1],liquidphi[i,j])
                    if theta < 0.01:
                        theta = 0.01
                    v[i,j] -= dt * (pressure[idx] - pressure[idx-Nx]) / dx / theta
                    v_valid[i,j] = True
            else:
                v[i,j] = 0
                
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
            
def compute_viscosity_weights():
    computeVolumeFraction(liquidphi, c_vol, np.array([-0.5,-0.5]), 2)
    computeVolumeFraction(liquidphi, n_vol, np.array([-1,-1]), 2)
    computeVolumeFraction(liquidphi, u_vol, np.array([-1,-0.5]), 2)
    computeVolumeFraction(liquidphi, v_vol, np.array([-0.5,-1]), 2)
    
def assemble_viscosity_matrix():
    
    # True 则是固体，False 则是液体
    u_state = np.zeros((Nx+1,Ny),dtype = bool)
    v_state = np.zerso((Nx,Ny+1),dtype = bool)
    
    for j in range(Nx):
        for i in range(Nx+1):
            if i - 1 < 0 or i >= Nx or (solidphi[i,j+1] + solidphi[i,j]) <= 0:
                u_state[i,j] = True
            else:
                u_state[i,j] = False
                
    for j in range(Nx+1):
        for i in range(Nx):
            if j - 1 < 0 or j >= Nx or (solidphi[i+1,j] + solidphi[i,j]) <= 0:
                v_state[i,j] = True
            else:
                v_state[i,j] = False
                
    Amat[:,:] = 0
    rhs[:] = 0
    factor = dt / dx / dx
    for j in range(1,Nx-1):
        for i in range(1,Nx-1):
            if u_state[i,j] == True:
                continue
            idx = j * Nx + 1
            rhs[idx] = u_vol[i,j] * u[i,j]
            Amat[idx,0] = u_vol[i,j]
            
            visc_right = v
                
    
    
                                 
def extrapolate(field,weight):
    
    nx = field.shape[0]
    ny = field.shape[1]
    valid = np.zeros((nx,ny),dtype = bool)
    for j in range(ny):
        for i in range(nx):
            valid[i,j] = False
            if weight[i,j] > 0:
                valid[i,j] = True
    
    for layer in range(10):
        valid_old = valid.copy()
        for j in range(1,ny-1):
            for i in range(1,nx-1):
                summ = 0
                count = 0
                if valid_old[i,j] == False:
                    
                    if valid_old[i+1,j] == True:
                        summ += field[i+1,j]
                        count += 1
                    if valid_old[i-1,j] == True:
                        summ += field[i-1,j]
                        count += 1
                    if valid_old[i,j+1] == True:
                        summ += field[i,j+1]
                        count += 1
                    if valid_old[i,j-1] == True:
                        summ += field[i,j-1]
                        count += 1
                        
                    if count > 0:
                        field[i,j] = summ / count
                        valid[i,j] == True
        
def constrain_velocity():
    global u
    global v
    temp_u = u.copy()
    temp_v = v.copy()
    
    for j in range(Nx):
        for i in range(Nx+1):
            pos = np.array([i*dx,(j+0.5)*dx])
            vel = getVelocity(pos)
            normal = Interpolate_Gradient(pos,solidphi)
            normal /= np.linalg.norm(normal)
            perp_component = normal[0]*vel[0] + normal[1]*vel[1]
            vel -= perp_component * normal
            temp_u[i,j] = vel[0]
            
    for j in range(Nx):
        for i in range(Nx+1):
            pos = np.array([(i+0.5)*dx,j*dx])
            vel = getVelocity(pos)
            normal = Interpolate_Gradient(pos,solidphi)
            normal /= np.linalg.norm(normal)
            perp_component = normal[0]*vel[0] + normal[1]*vel[1]
            vel -= perp_component * normal
            temp_v[i,j] = vel[1]
            
    u = temp_u.copy()
    v = temp_v.copy()
    
for i in range(Nx+1):
    for j in range(Ny+1):
        pos = np.array([i*dx,j*dx])
        phi0 = -(np.sqrt((pos[0] - circle[0,0])**2 + (pos[1] -circle[0,1])**2) - rad[0])
        phi1 = np.sqrt((pos[0] - circle[1,0])**2 + (pos[1] -circle[1,1])**2) - rad[1]
        phi2 = np.sqrt((pos[0] - circle[2,0])**2 + (pos[1] -circle[2,1])**2) - rad[2]
        phi3 = np.sqrt((pos[0] - circle[3,0])**2 + (pos[1] -circle[3,1])**2) - rad[3]
        solidphi[i,j] = phi0
            
particleVelocity = np.array((particleCount,2))
time = 0
timeFinal = 1
dt = 0.002
while(time < timeFinal):
    t = 0
    while(t < dt):
        substep = cfl()
        if t + substep > dt:
            substep = dt - t
            
        advect_particles(substep)
        
        compute_liquidphi()     
        
        advect_grid_velocity(substep)

        v[:,:] -= 0.1 # 重力的影响

        compute_viscosity_weights()
        
        assemble_viscosity_matrix()
        
        compute_weights()
        
        assemble_matrix(substep)
        
        cg_solver(pressure)
        
        pr = np.zeros((Nx,Nx))
        for j in range(Nx):
            for i in range(Nx):
                idx = j * Nx + i
                pr[i,j] = rhs[idx]
                
        substract_gradient()
        
        extrapolate(u, u_weights)
        
        extrapolate(v, v_weights)
        
        constrain_velocity()
                
        t += substep