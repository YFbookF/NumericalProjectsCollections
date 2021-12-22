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

time = 0
timeFinal = 100
while time < timeFinal:
    r1 = node_pos[elem_idx[time,1]] - node_pos[elem_idx[time,0]]
    r2 = node_pos[elem_idx[time,2]] - node_pos[elem_idx[time,0]]
    r3 = node_pos[elem_idx[time,3]] - node_pos[elem_idx[time,0]]
    
    d1 = r1[0]
    d5 = r1[1]
    d9 = r1[2]
    d2 = r2[0]
    d6 = r2[1]
    d10 = r2[2]
    d3 = r3[0]
    d7 = r3[1]
    d11 = r3[2]
    d12 = (d1 * d6 * d11 + d2 * d7 * d9 + d3 * d5 * d10) - d1 * d7 * d10 - d2 * d5 * d11 - d3 * d6 * d9
    d13 = 1.0 / d12
    d14 = abs(d12) / 6
    B = np.zeros((4,4))
    B[1,0] = (d10 * d7 - d6 * d11) * d13;
    B[2,0] = (d5 * d11 - d9 * d7) * d13;
    B[3,0] = (d9 * d6 - d5 * d10) * d13;
    B[0,0] = -B[1,0] - B[2,0] - B[3,0];
    B[1,1] = (d2 * d11 - d10 * d3) * d13;
    B[2,1] = (d9 * d3 - d1 * d11) * d13;
    B[3,1] = (d1 * d10 - d9 * d2) * d13;
    B[0,1] = -B[1,1] - B[2,1] - B[3,1];
    B[1,2] = (d6 * d3 - d2 * d7) * d13;
    B[2,2] = (d1 * d7 - d5 * d3) * d13;
    B[3,2] = (d5 * d2 - d1 * d6) * d13;
    B[0,2] = -B[1,2] - B[2,2] - B[3,2];
    nu = 10
    young = 10
    d15 = young / (1 + nu) / (1 - 2 * nu)
    d16 = (1 - nu) * d15
    d17 = nu * d15
    d18 = young / 2 / (1 + nu)
    Kemat = np.zeros((12,12))
    for i in range(4):
        for j in range(4):
            idx = i * 3
            idy = j * 3
            d19 = B[i,0]
            d20 = B[i,1]
            d21 = B[i,2]
            d22 = B[j,0]
            d23 = B[j,1]
            d24 = B[j,2]
            Kemat[idx+0,idy+0] = d16 * d19 * d22 + d18 * (d20 * d23 + d21 * d24)
            Kemat[idx+0,idy+1] = d17 * d19 * d23 + d18 * (d20 * d22)
            Kemat[idx+0,idy+2] = d17 * d19 * d24 + d18 * (d21 * d22)
            Kemat[idx+1,idy+0] = d17 * d20 * d22 + d18 * (d19 * d23)
            Kemat[idx+1,idy+1] = d16 * d20 * d23 + d18 * (d19 * d22 + d21 * d24)
            Kemat[idx+1,idy+2] = d17 * d20 * d24 + d18 * (d21 * d23)
            Kemat[idx+2,idy+0] = d17 * d21 * d22 + d18 * (d19 * d24)
            Kemat[idx+2,idy+1] = d17 * d21 * d23 + d18 * (d20 * d24)
            Kemat[idx+2,idy+2] = d16 * d21 * d24 + d18 * (d20 * d23 + d19 * d22)
    Kemat *= d14
            
    
    time += 1