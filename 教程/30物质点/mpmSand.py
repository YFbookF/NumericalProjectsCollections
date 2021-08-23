import numpy as np

dt = 1e-4
frameRate = 60
stepsPerFrame = int(1 / (dt / (1 / frameRate)))
alpha = 0.95

gravity = np.array([0.0,-10.0,0.0])
density = 1.0
mass = 10.0
volume = mass / density

filestr = "small.obj"
f = open(filestr,"r")   #设置文件对象
line = f.readline()
num = 0
while line:   
   num += 1
   line = f.readline()
   
particle_num = num
particle_pos = np.zeros((particle_num,3))   
f = open(filestr,"r")   #设置文件对象
line = f.readline()
st = line.split(' ')
particle_pos[0,0] = float(st[1])
particle_pos[0,1] = float(st[2])
particle_pos[0,2] = float(st[3])
for cnt in range(1,particle_num):
   line = f.readline()
   st = line.split(' ')
   particle_pos[cnt,0] = float(st[1])
   particle_pos[cnt,1] = float(st[2])
   particle_pos[cnt,2] = float(st[3])

particle_vel = np.zeros((particle_num,3))
particle_mass = np.zeros((particle_num))
particle_volume = np.zeros((particle_num,3))
particle_BP = np.zeros((particle_num,3,3))
particle_F = np.zeros((particle_num,3,3))
particle_Fe = np.zeros((particle_num,3,3))
particle_Fp = np.zeros((particle_num,3,3))
for i in range(particle_num):
    particle_mass[i] = mass
    particle_volume[i] = volume
    particle_F[i,:,:] = np.identity(3)
    particle_Fe[i,:,:] = np.identity(3)
    particle_Fp[i,:,:] = np.identity(3)
    
    
dx = 0.1
grid_nx = int(1 / dx) + 1
grid_ny = int(1 / dx) + 1
grid_nz = int(1 / dx) + 1
grid_num = grid_nx * grid_ny * grid_nz
grid_massG = np.zeros((grid_num))
grid_force = np.zeros((grid_num,3))
grid_velG = np.zeros((grid_num,3))
grid_velGn = np.zeros((grid_num,3))
grid_xi = np.zeros((grid_num,3))
grid_BP = np.zeros((grid_num))
cnt = 0
for k in range(grid_nz):
    for j in range(grid_ny):
        for i in range(grid_nx):
            grid_xi[cnt] = np.array([i,j,k])
            cnt += 1

useAPIC = False
energyDensityFunction = 1

def Reinitialize():
    cnt = 0
    for k in range(grid_nz):
        for j in range(grid_ny):
            for i in range(grid_nx):
                grid_massG[cnt] = 0
                grid_force[cnt,:] = 0
                grid_velG[cnt,:] = 0
                grid_velGn[cnt,:] = 0
                cnt += 1

def getBase(pos):
    basex = int(np.floor(pos[0]))
    basey = int(np.floor(pos[1]))
    basez = int(np.floor(pos[2]))
    return np.array([basex,basey,basez])

def computeWeightsQuadratic(pos):
    base = getBase(pos - 0.5)
    f = pos - base
    wp = np.zeros((3,3))
    wp[:,0] = 0.5 * (1.5 - f) * (1.5 - f)
    wp[:,1] = 0.75 - (f - 1) * (f - 1)
    wp[:,2] = 0.5 * (f - 0.5) * (f - 0.5)
    return wp

def computeWeightsCubic(pos):
    base = getBase(pos - 1)
    f = pos - base
    wp = np.zeros((3,4))
    wp[:,0] = (2 - f)*(2 - f)*(2 - f)*0.166666
    wp[:,1] = 0.5*(f-1)*(f-1)*(f-1) - (f-1)*(f-1) + 0.666666
    wp[:,1] = 0.5*(1-f)*(1-f)*(1-f) - (1-f)*(1-f) + 0.666666
    wp[:,2] = (f + 1)*(f + 1)*(f + 1)*0.166666
    return wp

def computeGradQuadratic(pos):
    base = getBase(pos)
    f = pos - base
    dwp = np.zeros((3,3))
    dwp[:,0] = - (1.5 - f)
    dwp[:,1] = -2*(f - 1)
    dwp[:,2] = f - 0.5
    return dwp

def computeGradCubic(pos):
    base = getBase(pos - 1)
    f = pos - base
    dwp = np.zeros((3,4))
    dwp[:,0] = 0.5 * (2 - f) * (2 - f)
    dwp[:,1] = 1.5 * (f - 1) * (f - 1) - 2 * (f - 1)
    dwp[:,2] = 1.5 * (2 - f) * (2 - f) - 2 * (2 - f)
    dwp[:,3] = 0.5 * (3 - f) * (3 - f)
    return dwp
    
def particleToGrid():
    wp = np.zeros((3,3))
    dwp = np.zeros((3,3))
    for pidx in range(particle_num):
        pos = particle_pos[pidx,:] / dx
        wp = computeWeightsQuadratic(pos)
        base = getBase(pos - 0.5)
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    w = wp[0,i] * wp[1,j] * wp[2,k]
                    idx = (base[2]+k)*grid_nx*grid_ny + (base[1]+j)*grid_nx + (base[0]+i)
                    grid_massG[idx] += particle_mass[pidx]
                    term = np.zeros((3))
                    if useAPIC == True:
                        term = 4 * np.dot(particle_BP[pidx,:,:],np.array([i,j,k]))
                    grid_velGn[idx,:] += w * particle_mass[pidx] * (particle_vel[pidx,:] + term)
    for k in range(grid_nz):
        for j in range(grid_ny):
            for i in range(grid_nx):
                idx = k*grid_nx*grid_ny + j*grid_nx + i
                if grid_massG[idx] >= 1e-16:
                    print(idx)
                    grid_velGn[idx,:] /= grid_massG[idx]
                else:
                    grid_velGn[idx,:] = 0
               
def addGravity():
    for i in range(grid_num):
        grid_force[i,:] += grid_massG[i] * gravity
        
def svd3d(defGrad):
    U,s,V = np.linalg.svd(defGrad)
    sigma = np.zeros((3,3))
    sigma[0,0] = s[0]
    sigma[1,1] = s[1]
    sigma[2,2] = s[2]
    if np.linalg.det(U) < 0:
        U[:,2] *= -1
        sigma[2,2] *= -1
    if np.linalg.det(V) < 0:
        V[:,2] *= -1
        sigma[2,2] *= -1
    if sigma[0,0] < sigma[1,1]:
        temp = sigma[0,0]
        sigma[0,0] = sigma[1,1]
        sigma[1,1] = temp
    return U,sigma,V
        
def corotatePiola(defGrad):
    piola = np.zeros((3,3))
    E = 50
    nu = 0.3
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    U,sigma,V = svd3d(defGrad)
    R = np.dot(U,V.T)
    J = np.linalg.det(defGrad)
    piola = 2 * mu * (defGrad - R) + lamb * (J - 1) * J * np.linalg.inv(defGrad.T)
    return piola
    
def neoHookeanPiola(defGrad):
    piola = np.zeros((3,3))
    E = 50
    nu = 0.3
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    U,sigma,V = svd3d(defGrad)
    R = np.dot(U,V.T)
    J = np.linalg.det(defGrad)
    piola = mu * (defGrad - np.linalg.inv(defGrad.T)) + (lamb + np.log(J) * np.linalg.inv(defGrad.T))
    fTf = np.dot(defGrad.T,defGrad)
    fTfTrace = fTf[0,0] + fTf[1,1] + fTf[2,2]
    energy = (mu / 2) * (fTfTrace - 3) - mu * np.log(J) + lamb / 2 * np.log(J) * np.log(J)
    return piola
    
def stVernantPiola(defGrad):
    piola = np.zeros((3,3))
    E = 50
    nu = 0.3
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    U,sigma,V = svd3d(defGrad)
    R = np.dot(U,V.T)
    J = np.linalg.det(defGrad)
    logSigma = np.log(sigma)
    logSigmaTrace = logSigma[0,0] + logSigma[1,1] + logSigma[2,2]
    piolaSingluar = 2 * mu * np.dot(logSigma,np.linalg.inv(sigma)) + lamb * logSigmaTrace * sigma.T
    piola = np.dot(np.dot(U,piolaSingluar),V.T)
    energy = mu * (logSigma[0,0]**2 + logSigma[1,1]**2 + logSigma[2,2]**2) + lamb / 2 * logSigmaTrace**2
    return piola

def snowPiola(defGrad,Fp,Fe):
    piola = np.zeros((3,3))
    E = 50
    nu = 0.3
    lamb = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    hc = 10
    Jp = np.linalg.det(Fp)
    Je = np.linalg.det(Fe)
    muFp = mu * np.exp(10 * (1 - Jp))
    lambdaFp = lamb * np.exp(10 * (1 - Jp))
    U,sigma,V = svd3d(defGrad)
    Re = np.dot(U,V.T)
    piola = 2 * muFp * (Fe - Re) + lambdaFp * (Je - 1) * Je * np.linalg.inv(Fe.T)
    return piola
        
def addGridForce():
    for pidx in range(particle_num):
        volume = particle_volume[pidx]
        defGrad = particle_F[pidx]
        Fp = particle_Fp[pidx]
        Fe = particle_Fe[pidx]
        
        piola = np.zeros((3,3))
        if energyDensityFunction == 1:
            piola = corotatePiola(defGrad)
        elif energyDensityFunction == 2:
            piola = neoHookeanPiola(defGrad)
        elif energyDensityFunction == 3:
            piola = stVernantPiola(defGrad)
        elif energyDensityFunction == 4:
            piola = snowPiola(defGrad)
        pos = particle_pos[pidx,:] / dx
        wp = computeWeightsQuadratic(pos)
        dwp = computeGradQuadratic(pos)
        base = np.array([int(pos[0] - 1),int(pos[1] - 1),int(pos[2] - 1)])
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    gradWip = np.zeros((3))
                    gradWip[0] = dwp[0,i] * wp[1,j] * wp[2,k] / dx
                    gradWip[1] = wp[0,i] * dwp[1,j] * wp[2,k] / dx
                    gradWip[2] = wp[0,i] * wp[1,j] * dwp[2,k] / dx
                    fi = - volume * np.dot(np.dot(piola,Fe.T),gradWip)
                    if energyDensityFunction != 3:
                        fi = - volume * np.dot(np.dot(piola,defGrad.T),gradWip)
                    idx = (base[2]+k)*grid_nx*grid_ny + (base[1]+j)*grid_nx + (base[0]+i)
                    grid_force[idx] += fi

def updateGridVelocity():
    bound = 2
    for k in range(grid_nz):
        for j in range(grid_ny):
            for i in range(grid_nx):
                idx = k * grid_ny * grid_nx + j * grid_nx + i
                if grid_massG[idx] < 1e-16:
                    continue
                grid_velG[idx] = grid_velGn[idx] + dt * grid_force[idx] / grid_massG[idx]
                if i < bound and grid_velG[idx,0] < 0:
                    grid_velG[idx,0] = 0
                if i > grid_nx - bound and grid_velG[idx,0] > 0:
                    grid_velG[idx,0] = 0
                if j < bound and grid_velG[idx,1] < 0:
                    grid_velG[idx,1] = 0
                if j < grid_ny - bound and grid_velG[idx,1] > 0:
                    grid_velG[idx,1] = 0
                if k < bound and grid_velG[idx,2] < 0:
                    grid_velG[idx,2] = 0
                if k < grid_nz - bound and grid_velG[idx,2] > 0:
                    grid_velG[idx,2] = 0
                    
def sphereGridCollision():
    levelset = np.zeros((grid_num))
    for k in range(grid_nz):
        for j in range(grid_ny):
            for i in range(grid_nx):
                idx = k * grid_ny * grid_nx + j * grid_nx + i
                if grid_massG[idx] < 1e-16:
                    continue
                squareNorm = np.sqrt((i*dx-0.5)**2 + j**2 + (k*dx-0.5)**2)
                levelset[idx] = squareNorm - 0.2
                if levelset[idx] < 0:
                    # grid_velG[idx] = 0
                    norml = np.zeros((3))
                    if squareNorm > 1e-7:
                        norml = np.array([i*dx-0.5,j**2,k*dx-0.5**2]) / squareNorm
                    dotResult = norml[0]*grid_velG[idx,0] + norml[1]*grid_velG[idx,1] + norml[2]*grid_velG[idx,2]
                    if dotResult < 0:
                        grid_velG[idx,:] -= dotResult * norml
                        friction = 0
                        if friction != 0:
                            squareNorm = np.sqrt(grid_velG[idx,0]**2 + grid_velG[idx,1]**2 + grid_velG[idx,2]**2)
                            if -dotResult * friction < squareNorm:
                                grid_velG[idx,:] += dotResult * friction * grid_velG[idx,:] / squareNorm
                            else:
                                dotResult[idx,:] = np.zeros((3))
        
def updateDeformationGradient():
    for pidx in range(particle_num):
        pos = particle_pos[pidx,:] / dx
        wp = computeWeightsQuadratic(pos)
        dwp = computeGradQuadratic(pos)
        base = np.array([int(pos[0] - 1),int(pos[1] - 1),int(pos[2] - 1)])
        defGrad = particle_F[pidx,:]
        grad_vp = np.zeros((3,3))
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    idx = (base[2]+k)*grid_nx*grid_ny + (base[1]+j)*grid_nx + base[0]+i
                    gradWip = np.zeros((3,3))
                    gradWip[0] = dwp[0,i] * wp[1,j] * wp[2,k] / dx
                    gradWip[1] = wp[0,i] * dwp[1,j] * wp[2,k] / dx
                    gradWip[2] = wp[0,i] * wp[1,j] * dwp[2,k] / dx
                    grad_vp += np.dot(gradWip,np.transpose(grid_velG[idx,:]))
        newdefGrad = defGrad + dt * np.dot(grad_vp,defGrad)
        if True == True:
            thetaC = 0.025
            thetaS = 0.0055
            Fe = particle_Fe[pidx,:,:]
            newFe = Fe * dt * np.dot(grad_vp,Fe)
            U,sigma,V = svd3d(newFe)
            if i in range(3):
                sigma[i,i] = max(1-thetaC,min(sigma[i,i],1+thetaS))
            particle_Fe[pidx] = np.dot(np.dot(U,sigma),V.T)
            particle_Fp[pidx] = np.dot(np.dot(np.dot(V,np.linalg.inv(sigma)),U.T),newdefGrad)
        particle_F[pidx] = newdefGrad

def GridToParticle():
    useAPIC = True
    wp = np.zeros((3,3))
    for pidx in range(particle_num):
        pos = particle_pos[pidx,:] / dx
        wp = computeWeightsQuadratic(pos)
        base = np.array([int(pos[0] - 1),int(pos[1] - 1),int(pos[2] - 1)])
        vpic = np.zeros((3))
        vflip = np.zeros((3))
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    w = wp[0,i] * wp[1,j] * wp[2,k]
                    idx = (base[2]+k)*grid_nx*grid_ny + (base[1]+j)*grid_nx + (base[0]+i)
                    if useAPIC == True:
                        particle_BP[pidx,:,:] += w * np.dot(grid_velG[idx,:],np.array([i,j,k]))
                    vpic += w * grid_velG[idx]
                    vflip += w * (grid_velG[idx,:] - grid_velGn[idx,:])
        if useAPIC == True:
            particle_vel[pidx] = vpic
        else:
            alpha = 0.95
            particle_vel[pidx] = (1 - alpha) * vpic + alpha * vflip
        particle_pos[pidx] += dt * vpic
        
def sphereParticleCollision():
    for pidx in range(particle_num):
        pos = particle_pos[pidx,:]
        squareNorm = np.sqrt((i*dx-0.5)**2 + j**2 + (k*dx-0.5)**2)
        levelset = squareNorm - 0.2
        if levelset < 0:
            # particle_vel[pidx,:] = np.zeros((3))
            normal = np.zeros((3))
            if squareNorm > 1e-7:
                normal = np.array([i*dx-0.5,j**2,k*dx-0.5**2]) / squareNorm
            dotResult = normal[0]*particle_vel[pidx,0] + normal[1]*particle_vel[pidx,1] + normal[2]*particle_vel[pidx,2]
            if dotResult < 0:
                particle_vel[pidx] -= dotResult * normal
                friction = 0
                if friction != 0:
                    squareNorm = np.sqrt(particle_vel[pidx,0]**2 + particle_vel[pidx,1]**2 + particle_vel[pidx,2]**2)
                    if -dotResult * friction < squareNorm:
                        particle_vel[pidx,:] += dotResult * friction * particle_vel[pidx,:] / squareNorm
                    else:
                        particle_vel[pidx,:] = np.zeros((3))
        
time = 0
timeFinal = 10
while(time < timeFinal):
    time += 1
    Reinitialize()
    particleToGrid()
    addGravity()
    addGridForce()
    updateGridVelocity()
    sphereGridCollision()
    updateDeformationGradient()
    GridToParticle()
    sphereGridCollision()