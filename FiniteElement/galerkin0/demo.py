import numpy as np
nmax = 20

def fullweight(n,z,p,xx):
    nminus = n - 1
    m = 1
    c1 = 1
    c4 = z[0] - xx
        
    c = np.zeros((n,m+1))
    c[0,0] = 1
        
    for i in range(n-1):
        mn = min((i+1),m)
        c2 = 1
        c5 = c4
        c4 = z[i+1] - xx
        for j in range(i+1):
            c3 = z[i+1] - z[j]
            c2 = c2 * c3
            for k in range(mn,0,-1):
                c[i+1,k] = c1*(k*c[i,k-1] - c5*c[i,k])/c2
            c[i+1,0] = -c1*c5*c[i,0] / c2
            for k in range(mn,0,-1):
                c[j,k] = (c4*c[j,k] - k*c[j,k-1]) / c3
            c[j,0] = c4*c[j,0]/c3
        c1 = c2
    return c
# compute N + 1 Gauss-Lobatto-Legendre nodes
def SEMhat(N):
    n = N + 1
    z = np.zeros((n))
    weight = np.zeros((n))
    z[0] = - 1
    z[n-1] = 1
    
    if n > 1:
        if n == 2:
            z[1] = 0
        else:
            M = np.zeros((N-1,N-1))
            for i in range(N-2):
                M[i,i+1] = np.sqrt((i+1)*(i+3)/(i+1.5)/(i+2.5))/2
                M[i+1,i] = M[i,i+1]
            eigenvalue,eigenvector = np.linalg.eig(M)
            z[1:N] = eigenvalue[:]
            
    weight[N] = weight[0] = 2 / N / n
    
    for i in range(1,n):
        x = z[i]
        z0 = 1
        z1 = x
        for j in range(0,N-1):
            z2 = x*z1*(2*j+3) / (j+2) - z0*(j+1)/(j+2)
            z0 = z1
            z1 = z2
        weight[i] = 2 / (N*n*z2*z2)
        
    Bh = np.zeros((n,n))
    for i in range(n):
        Bh[i,i] = weight[i]
        
    
    Dh = np.zeros((n,n))
    for p in range(n):

        c = fullweight(n, z, p,z[p])
        Dh[p,:] = c[:,1]
        
    Ah = np.dot(np.dot(np.transpose(Dh),Bh),Dh)
    Ch = np.dot(Bh,Dh)
    return Ah,Bh,Ch,Dh,z,weight

Ah,Bh,Ch,Dh,z,w = SEMhat(nmax)
No = int(nmax*3/2) # Overintgration for Convection
Ao,Bo,Co,Do,zo,wo = SEMhat(No)
bo = np.zeros((No + 1))
for i in range(No+1):
    bo[i] = Bo[i,i]
for i in range(No+1):
    Bo[i,:] = bo[i] * bo[:]

Ih = np.zeros((No,nmax))
for p in range(No):
    temp = fullweight(nmax, z, p,zo[p])
    Ih[p,:] = temp[:,0]



