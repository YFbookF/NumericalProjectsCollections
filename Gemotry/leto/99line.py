import numpy as np

nelx = 10
nely = 10
volfrac = 1
penal = 1
rmin = 0.01
E = 1
nu = 0.3
k = np.array([0.5-nu/6,0.125+nu/8,
              -0.25-nu/12,-0.125+3*nu/8,
              -0.25+nu/12,-0.125-nu/8,
              nu/6,0.125-3*nu/8])

Ke = E / (1 - nu**2)*np.array([
    [k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],
    [k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],
    [k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],
    [k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],
    [k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],
    [k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],
    [k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],
    [k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])

def FEanalysis(x):
    size = 2 *(nelx + 1) * (nely + 1)
    K = np.zeros((size,size))
    F = np.zeros((size))
    U = np.zeros((size))
    for elx in range(nelx):
        for ely in range(nely):
            n1= (nelx + 1)*ely + elx
            n2 = (nelx + 1)*(ely + 1) + elx
            edof = np.array([2*n1,2*n1+1,
                             2*n2,2*n2+1,
                             2*n2+2,2*n2+3,
                             2*n1+2,2*n1+3],dtype = int)
            for i in range(8):
                for j in range(8):
                    idxi = edof[i]
                    idxj = edof[j]
                    K[idxi,idxj] += x[elx,ely]**penal*Ke[i,j]
            test = 1
            
    F[1] = -1
    fixed = np.array([0,2,4,6,8,10,12,14,16,18,20,241])
    for i in range(len(fixed)):
        K[fixed[i],:] = 0
        K[fixed[i],fixed[i]] = 1
    U = np.dot(np.linalg.inv(K),F)
    for i in range(len(fixed)):
        U[fixed[i]] = 0
    return U
    

x = np.ones((nelx,nely)) *volfrac
loop = 0
change = 1
c = 0
while change > 0.01:
    loop = loop + 1
    xold = x
    dc = np.zeros((nelx,nely))
    U = FEanalysis(x)
    for elx in range(nely):
        for ely in range(nelx):
            n1= (nelx + 1)*ely + elx
            n2 = (nelx + 1)*(ely + 1) + elx
            edof = np.array([2*n1,2*n1+1,
                             2*n2,2*n2+1,
                             2*n2+2,2*n2+3,
                             2*n1+2,2*n1+3],dtype = int)
            Ue = np.zeros((8))
            for i in range(8):
                Ue[i] = U[edof[i]]
            factor = np.dot(np.dot(np.transpose(Ue),Ke),Ue)
            c = c + x[elx,ely]**penal * factor
            dc[elx,ely] = -penal * x[elx,ely]**(penal-1)*factor
    
    dcn = np.zeros((nelx,nely))
    for i in range(nelx):
         for j in range(nely):
             summ = 0
             kstart = max(i-np.floor(rmin)-1,0)
             kend = min(i+np.floor(rmin),nelx)
             lstart = max(j-np.floor(rmin)-1,0)
             lend = min(j+np.floor(rmin),nely)
             for k in range(kstart,kend):
                 for l in range(lstart,lend):
                     fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                     summ = summ + max(0,fac)
                     dcn[i,j] = dcn[i,j] + max(0,fac)*x[k,l]*dc[k,l]
             dcn[i,j] = dcn[i,j] / (x[i,j] * summ)
             
    l1 = 0
    l2 = 100000
    move = 0.2
    while(l2 - l1 > 1e-4):
        lmid= (l2 + l1) * 0.5
        xnew = max(0.001,max(x-move,min(1,min(x+move,x*np.sqrt(-dcn / lmid)))))
        if sum(sum(xnew)) - volfrac*nelx*nely >0:
            l1 = lmid
        else:
            l2 = lmid
        