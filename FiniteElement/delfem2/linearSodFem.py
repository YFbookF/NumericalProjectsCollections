import numpy as np
f = open('linear_coord.txt')  # 打开txt文件，以‘utf-8’编码读取
line = f.readline()   # 以行的形式进行读取文件
coord = np.zeros((1885,2))
idx = 0
while line:
    a = line.split()
    coord[idx,0] = a[0]
    line = f.readline()
    a = line.split()
    coord[idx,1] = a[0]
    line = f.readline()
    idx += 1
f.close()

f = open('linear_tri.txt')  # 打开txt文件，以‘utf-8’编码读取
line = f.readline()   # 以行的形式进行读取文件
element3 = np.zeros((3430,3),dtype = int)
idx = 0
while line:
    a = line.split()
    element3[idx,0] = a[0]
    line = f.readline()
    a = line.split()
    element3[idx,1] = a[0]
    line = f.readline()
    a = line.split()
    element3[idx,2] = a[0]
    line = f.readline()
    idx += 1
f.close()

def triArea2D(p0,p1,p2):
    return ((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))/2

num_nodes = 1885

# 第一次见到这样的矩阵

aVal = np.zeros((1885,2))
Amat = np.zeros((num_nodes,num_nodes,4))
bvec = np.zeros((num_nodes,2))
for ie in range(element3.shape[0]):
    eres = np.zeros((3,2))
    emat = np.zeros((9,4))
    dldx = np.zeros((3,2))
    p0 = coord[element3[ie,0],:]
    p1 = coord[element3[ie,1],:]
    p2 = coord[element3[ie,2],:]
    disp = np.array([p0,p1,p2])
    
    area = triArea2D(p0, p1, p2)
    const_term = np.zeros((3))
    temp = 1 / 2 / area
    
    const_term[0] = temp * (p1[0]*p2[1] - p2[0]*p1[1])
    const_term[1] = temp * (p2[0]*p0[1] - p0[0]*p2[1])
    const_term[2] = temp * (p0[0]*p1[1] - p1[0]*p0[1])
    
    dldx[0,0] = temp * (p1[1] - p2[1])
    dldx[1,0] = temp * (p2[1] - p0[1])
    dldx[2,0] = temp * (p0[1] - p1[1])
    dldx[0,1] = temp * (p2[0] - p1[0])
    dldx[1,1] = temp * (p0[0] - p2[0])
    dldx[2,1] = temp * (p1[0] - p0[0])
    
    lam = 10
    mu = 10
    
    for j in range(3):
        for i in range(3):
            idx = j * 3 + i
            emat[idx,0] = area * (lam + mu)*dldx[i,0]*dldx[j,0]
            emat[idx,1] = area * (lam * dldx[i,0] * dldx[j,1] + mu * dldx[j,0] * dldx[i,1])
            emat[idx,2] = area * (lam * dldx[i,1] * dldx[j,0] + mu * dldx[j,1] * dldx[i,0])
            emat[idx,3] = area * (lam + mu) * dldx[i,1] * dldx[j,1]
            dtemp1 = area * mu *(dldx[i,1] * dldx[j,1] + dldx[i,0] * dldx[j,0])
            emat[idx,0] += dtemp1
            emat[idx,3] += dtemp1
            
    rho = 1
    gx = 0
    gy = -3
    for i in range(3):
        eres[i,0] = area * rho * gx / 3
        eres[i,1] = area * rho * gy / 3
        
    for i in range(3):
        for j in range(3):
            idx = j * 3 + i
            eres[i,0] -= emat[idx,0] * disp[j,0] + emat[idx,1] * disp[j,1]
            eres[i,1] -= emat[idx,2] * disp[j,0] + emat[idx,3] * disp[j,1] 
    
    for i in range(3):
        idx = element3[ie,i]
        bvec[idx,0] += eres[i,0]
        bvec[idx,1] += eres[i,1]
        
    for i in range(3):
        for j in range(3):
            idxi = element3[ie,i]
            idxj = element3[ie,j]
            idx = j * 3 + i
            Amat[idxi,idxj,:] += emat[idx,:]
    