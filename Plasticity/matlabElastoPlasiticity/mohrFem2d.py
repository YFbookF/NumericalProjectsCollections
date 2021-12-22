import numpy as np
E = 10e3
nu = 0.25
D = 2
Lenght = 1
phi = 30
tsi = 30
cohesion = 1e6
sig_v = 0
sig_h = 0
C = E / (1 + nu) / (1 - 2*nu)*np.array([[1-nu,nu,0,nu],[nu,1-nu,0,nu],
                                        [0,0,0.5-nu,0],[nu,nu,0,1-nu]])
dt = 1
node_row_num = 5
dx = 1.0 / (node_row_num - 1)
node_num = node_row_num * node_row_num
element_row_num = int((node_row_num - 1)/2)
element_num = element_row_num * element_row_num * 2

element_idx = np.zeros((element_num,6),dtype = int)

node_pos = np.zeros((node_num,2),dtype = float)
node_vel = np.zeros((node_num,2),dtype = float)
node_force = np.zeros((node_num,2),dtype = float)

for j in range(node_row_num):
    for i in range(node_row_num):
        pos_x = float(i) * dx 
        pos_y = float(i) * dx * 2 - 1
        node_pos[j*node_row_num+i] = np.array([pos_x,pos_y])
cnt = 0
for j in range(element_row_num):
    for i in range(element_row_num):
        idx = j * node_row_num * 2 + i * 2
        element_idx[cnt,0] = idx
        element_idx[cnt,1] = idx + 2
        element_idx[cnt,2] = idx + node_row_num * 2
        element_idx[cnt,3] = idx + 1
        element_idx[cnt,4] = idx + 1 + node_row_num
        element_idx[cnt,5] = idx + node_row_num
        cnt += 1
        element_idx[cnt,0] = idx + 2
        element_idx[cnt,1] = idx + node_row_num * 2 + 2
        element_idx[cnt,2] = idx + node_row_num * 2
        element_idx[cnt,3] = idx + 2 + node_row_num
        element_idx[cnt,4] = idx + node_row_num * 2 + 1
        element_idx[cnt,5] = idx + node_row_num + 1
        cnt += 1