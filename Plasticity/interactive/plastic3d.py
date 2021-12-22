import numpy as np

node_num = 4
node_pos = np.array([[0,0,0],
                     [1,0,0],
                     [0,1,0],
                     [0,0,1]],dtype = float)
elem_num = 1
elem_idx = np.array([[0,1,2,3]])

def realDot(vec0,vec1):
    length = len(vec0)
    res = 0
    for i in range(length):
        res += vec0[i] * vec1[i]
    return res

r1 = node_pos[elem_idx[0,1]] - node_pos[elem_idx[0,0]]
r2 = node_pos[elem_idx[0,2]] - node_pos[elem_idx[0,0]]
r3 = node_pos[elem_idx[0,3]] - node_pos[elem_idx[0,0]]
J = realDot(np.cross(r1, r2),r3)
mVolume = abs(J) * 0.1666
div6V = 1.0 / mVolume * 6.0
n1x = (r2[1] * r3[2] - r3[1] * r2[2]) * div6V
n1y = (r3[0] * r2[2] - r2[0] * r3[2]) * div6V
n1z = (r2[0] * r3[1] - r3[0] * r2[1]) * div6V
n2x = (r1[2] * r3[1] - r1[1] * r3[2]) * div6V
n2y = (r1[0] * r3[2] - r1[2] * r3[0]) * div6V
n2z = (r1[1] * r3[0] - r1[0] * r3[1]) * div6V
n3x = (r1[1] * r2[2] - r1[2] * r2[1]) * div6V
n3y = (r1[2] * r2[0] - r1[0] * r2[2]) * div6V
n3z = (r1[0] * r2[1] - r1[1] * r2[0]) * div6V

time = 0
timeFinal = 100
while time < timeFinal:
    r1 = node_pos[elem_idx[0,1]] - node_pos[elem_idx[0,0]]
    r2 = node_pos[elem_idx[0,2]] - node_pos[elem_idx[0,0]]
    r3 = node_pos[elem_idx[0,3]] - node_pos[elem_idx[0,0]]
    
    J = realDot(np.cross(r1, r2),r3)
    Jinv = 1.0 / J
    mVolume = abs(J) * 0.1666
    mBti = np.zeros((4,3))
    
    b1 = (r2[2] * r3[1] - r2[1] * r3[2]) * Jinv
    b2 = (r1[1] * r3[2] - r1[2] * r3[1]) * Jinv
    b3 = (r1[2] * r2[1] - r1[1] * r2[2]) * Jinv
    b0 = - b1 - b2 - b3
    
    c1 = (r2[0] * r3[2] - r2[2] * r3[0]) * Jinv
    c2 = (r1[2] * r3[0] - r1[0] * r3[2]) * Jinv
    c3 = (r1[0] * r2[2] - r1[2] * r2[0]) * Jinv
    c0 = - c1 - c2 - c3
    
    d1 = (r2[1] * r3[0] - r2[0] * r3[1]) * Jinv
    d2 = (r1[0] * r3[1] - r1[1] * r3[0]) * Jinv
    d3 = (r1[1] * r2[0] - r1[0] * r2[1]) * Jinv
    d0 = - d1 - d2 - d3
    Bmat = np.array([[b0,0,0,b1,0,0,b2,0,0,b3,0,0],
                     [0,c0,0,0,c1,0,0,c2,0,0,c3,0],
                     [0,0,d0,0,0,d1,0,0,d2,0,0,d3],
                     [c0,b0,0,c1,b1,0,c2,b2,0,c3,b3,0],
                     [0,d0,c0,0,d1,c1,0,d2,c2,0,d3,c3],
                     [d0,0,b0,d1,0,b1,d2,0,b2,d3,0,b3]])
    
    nu = 10
    young = 10
    Dmat = np.array([[1 - nu,nu,nu,0,0,0],
                     [nu,1 - nu,nu,0,0,0],
                     [nu,nu,1 - nu,0,0,0],
                     [0,0,0,(1 - 2*nu)*0.5,0,0,],
                     [0,0,0,0,(1-2*nu)*0.5,0],
                     [0,0,0,0,0,(1-2*nu)*0.5]])
    Dmat *= young / (1 + nu) / (1 - 2 * nu)
    Kmat = mVolume * np.dot(np.dot(Bmat.T,Dmat),Bmat)
    Re = np.zeros((3,3))
    Re[0,0] = r1[0] * n1x + r2[0] * n2x + r3[0] * n3x
    Re[0,1] = r1[0] * n1y + r2[0] * n2y + r3[0] * n3y
    Re[0,2] = r1[0] * n1z + r2[0] * n2z + r3[0] * n3z
    Re[1,0] = r1[1] * n1x + r2[1] * n2x + r3[1] * n3x
    Re[1,1] = r1[1] * n1y + r2[1] * n2y + r3[1] * n3y
    Re[1,2] = r1[1] * n1z + r2[1] * n2z + r3[1] * n3z
    Re[2,0] = r1[2] * n1x + r2[2] * n2x + r3[2] * n3x
    Re[2,1] = r1[2] * n1y + r2[2] * n2y + r3[2] * n3y
    Re[2,2] = r1[2] * n1z + r2[2] * n2z + r3[2] * n3z
    
    
    