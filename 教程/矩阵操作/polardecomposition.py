import numpy as np

# Implicit-shifted Symmetric QR Singular Value Decomposition of 3 × 3 Matrices
# 论文很通俗易懂，代码也很易懂

def G2_12(c,s):
    given_rotation = np.array([[c,s],[-s,c]])
    return given_rotation

def G3_12(c,s):
    given_rotation = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    return given_rotation

def G3_23(c,s):
    given_rotation = np.array([[1,0,0],[0,c,s],[0,-s,c]])
    return given_rotation

def G3_13(c,s):
    given_rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return given_rotation

# 二维极分解
def polarDecomposition(A):
    x = A[0,0] + A[1,1]
    y = A[1,0] - A[0,1]
    d = np.sqrt(x*x + y*y)
    c = 1
    s = 0
    if d != 0:
        c = x / d
        s = - y / d
    R = np.array([[c,s],[-s,c]])
    S = np.dot(R.T,A)
    return R,S

def svd2d(A):
    R,S = polarDecomposition(A)
    cosine = 0
    sine = 0
    x = S[0,0]
    y = S[0,1]
    z = S[1,1]
    sigma = np.zeros((2))
    if y == 0:
        cosine = 1
        sine = 0
        sigma[0] = x
        sigma[1] = z
    else:
        tau = 0.5 * (x - z)
        w = np.sqrt(tau * tau + y * y)
        t = 0
        if tau > 0:
            t = y / (tau + w)
        else:
            t = y / (tau - w)
        cosine = 1 / np.sqrt(t*t + 1)
        sine = - t * cosine
        c2 = cosine * cosine
        csy = 2 * cosine * sine * y
        s2 = sine * sine
        sigma[0] = c2 * x - csy + s2 * z
        sigma[1] = s2 * x - csy + c2 * z
    vc = cosine
    vs = sine
    if sigma[0] < sigma[1]:
        temp = sigma[0]
        sigma[0] = sigma[1]
        sigma[1] = temp
        vc = - sine
        vs = cosine
    V = np.array([[vc,vs],[-vs,vc]])
    U = np.dot(R,V)
    return U,sigma,V
    
def wilkinsonShift(a1,b1,a2):
    d = 0.5*(a1 - a2)
    bs = b1 * b1
    mu = a2 - bs / (d + np.sign(d) * np.sqrt(d*d + bs))
    return mu
    
def zeroChasing(U,A,V):
    G = G3_23(A[0,1], A[0,2])
    A = np.dot(A,G)
    U = np.dot(G.T,U)
    G = G3_23(A[0,1], A[0,2])
    A = np.dot(G.T,A)
    V = np.dot(G.T,V)
    G = G3_23(A[1,1],A[2,1])
    A = np.dot(G.T,A)
    U = np.dot(U,G)
    return U,A,V
    
def bidiagonalize(U,A,V):
    G = G3_23(A[1,0],A[2,0])
    A = np.dot(G.T,A)
    U = np.dot(U,G)
    U,A,V = zeroChasing(U, A, V)
    return U,A,V

def solveReducedTopLeft(B,U,sigma,V):
    s3 = B[2,2]
    u = G2_12(0, 1)
    v = G2_12(0, 1)
    u,s,v = svd2d(B[0:2,0:2])
    u = G3_12(u[0,0], u[0,1])
    v = G3_12(v[0,0], v[0,1])
    U = np.dot(U,u)
    V = np.dot(V,v)
    sigma[0,0] = s[0]
    sigma[1,1] = s[1]
    sigma[2,2] = s3
    return U,sigma,V

def solveReducedBotRight(B,U,sigma,V):
    s1 = B[1,1]
    u = G2_12(0, 1)
    v = G2_12(0, 1)
    u,s,v = svd2d(B[1:3,1:3])
    u = G3_23(u[0,0], u[0,1])
    v = G3_23(v[0,0], v[0,1])
    U = np.dot(U,u)
    V = np.dot(V,v)
    sigma[0,0] = s1
    sigma[1,1] = s[0]
    sigma[2,2] = s[1]
    return U,sigma,V

def sortWithTopLeftSub(U,sigma,V):
    if abs(sigma[1,1]) > abs(sigma[2,2]):
        if sigma[1,1] < 0:
            sigma[1,1] = - sigma[1,1]
            sigma[2,2] = - sigma[2,2]
            U[1,:] = - U[1,:]
            U[2,:] = - U[2,:]
            return U,sigma,V
    if sigma[2,2] < 0:
        sigma[1,1] = - sigma[1,1]
        sigma[2,2] = - sigma[2,2]
        U[1,:] = - U[1,:]
        U[2,:] = - U[2,:]

def svd3d(A):
    tol = 1e-12
    B = A.copy()
    U = np.identity(3)
    V = np.identity(3)
    U,B,V = bidiagonalize(U, B, V)
    alpha_1 = B[0,0]
    alpha_2 = B[1,1]
    alpha_3 = B[2,2]
    beta_1 = B[0,1]
    beta_2 = B[1,2]
    gamma_1 = alpha_1 * beta_1
    gamma_2 = alpha_2 * beta_2
    tol = tol * max(1,0.5*(alpha_1*alpha_1 + alpha_2*alpha_2 + alpha_3*alpha_3 + beta_1*beta_1 + beta_2*beta_2))
    
    while abs(beta_2) > tol and abs(beta_1) > tol and abs(alpha_1) > 0 and abs(alpha_2) > tol and abs(alpha_3) > tol:
        mu = wilkinsonShift(alpha_2*alpha_2+beta_1*beta_1, gamma_2, alpha_3*alpha_3 + beta_2*beta_2)
        G = G3_12(alpha_1*alpha_1 - mu, gamma_1)
        B = np.dot(B,G)
        V = np.dot(V,G)
        U,B,V = zeroChasing(U, B, V)
        alpha_1 = B[0,0]
        alpha_2 = B[1,1]
        alpha_3 = B[2,2]
        beta_1 = B[0,1]
        beta_2 = B[1,2]
        gamma_1 = alpha_1 * beta_1
        gamma_2 = alpha_2 * beta_2
    sigma = np.zeros((3,3))
    if abs(beta_2) < tol:
        solveReducedTopLeft(B,U,sigma,V)
        sortWithTopLeftSub(U,sigma,V)