import numpy as np
import random
# https://github.com/christopherbatty/VariationalViscosity2D
# SCA 2008 paper "Accurate Viscous Free Surfaces[...]" by Batty & Bridson
# Fluid Ver.
Nx = 32
Ny = 32
dx = 1 / Nx

u = np.zeros((Nx+1,Ny))
u_weights = np.zeros((Nx+1,Ny))
v = np.zeros((Nx,Ny+1))
v_weights = np.zeros((Nx+1,Ny))

solidphi = np.zeros((Nx+1,Ny+1))
liquidphi = np.zeros((Nx+1,Ny+1))


Amat = np.zeros((Nx * Nx,5)) # 自身，i-1,i+1,j-1,j+1
rhs = np.zeros((Nx * Nx))
pressure = np.zeros((Nx * Nx))

particleRadius = dx / np.sqrt(2)
viscosity = np.ones((Nx,Ny))

circle = np.array([[0.5,0.5],[0.7,0.5],[0.3,0.35],[0.5,0.7]])
rad = np.array([0.4,0.1,0.1,0.1])

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

def getVelocityInDomain(x,y,field):
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
    v00 = getVelocityInDomain(ix,iy,u)
    v10 = getVelocityInDomain(ix+1,iy,u)
    v01 = getVelocityInDomain(ix,iy+1,u)
    v11 = getVelocityInDomain(ix+1,iy+1,u)
    uval = bilerp(v00,v10,v01,v11,fx, fy)
    gx = pos[0] / dx 
    gy = pos[1] / dx - 0.5
    ix = int(gx)
    iy = int(gy)
    fx = gx - ix
    fy = gy - iy
    v00 = getVelocityInDomain(ix,iy,v)
    v10 = getVelocityInDomain(ix+1,iy,v)
    v01 = getVelocityInDomain(ix,iy+1,v)
    v11 = getVelocityInDomain(ix+1,iy+1,v)
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
            gy = particlePos[pidx,1] / dx - 0.5
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
    for j in range(1,Nx-1):
        for i in range(1,Nx-1):
            idx = j * Nx + 1
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
                rhs[idx] -= u_weights[i,j] * u[i,j] / dx
                
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
                
                # left
                term = u_weights[i,j] * dt / dx / dx
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
    return field[idx] * Amat[idx,0] - term
    
def cg_solver():
    direction = np.zeros((Nx * Nx))
    residual = np.zeros((Nx * Nx))


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
        
        advect_grid_velocity(substep)

        v[:,:] -= 0.1 # 重力的影响

        compute_liquidphi()        
        
        compute_weights()
        
        assemble_matrix()
        
        cg_solver()
        
                
        