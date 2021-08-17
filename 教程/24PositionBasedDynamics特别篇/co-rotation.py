import numpy as np

def jacobiRotate(A,R,p,q):
    # p 和 q 是索引
    # A 和 R 是 3x3 矩阵
    if A[p,q] == 0:
        return R
    d = (A[p,p] - A[q,q]) / (2 * A[p,q])
    t = 1 / (abs(d) + np.sqrt(d*d + 1))
    if d < 0:
        t = -t
    c = 1 / np.sqrt(t * t + 1)
    s = t * c
    A[p,p] += t * A[p,q]
    A[q,q] -= t * A[p,q]
    A[p,q] = A[q,p] = 0
    
    for k in range(3):
        if k != p and k != q:
            Akp = c * A[k,p] + s * A[k,q]
            Akq = - s * A[k,p] + c * A[k,q]
            A[k,p] = A[p,k] = Akp
            A[k,q] = A[q,k] = Akq
    
    for k in range(3):
        Rkp = c * R[k,p] + s * R[k,q]
        Rkq = - s * R[k,p] + c * R[k,q]
        R[k,p] = Rkp
        R[k,q] = Rkq
        
def eigenDecomposition(A,eigenVec,eigenVal):
    numIterations = 10
    eps = 1e-10
    
    D = A.copy() # 3 x 3 matrix
    eigenVec = np.eye(3)
    for ite in range(numIterations):
        maxd = abs(D[0,1])
        p = 0 
        q = 1
        a = abs(D[0,2])
        if a > maxd:
            p = 0
            q = 2
            maxd = a
        a = abs(D[1,2])
        if a > maxd:
            p = 1
            q = 2
            maxd = a
        if maxd < eps:
            break
        eigenVec = jacobiRotate(D, eigenVec, p, q)
    eigenVal[0] = D[0,0]
    eigenVal[1] = D[1,1]
    eigenVal[2] = D[2,2]
    