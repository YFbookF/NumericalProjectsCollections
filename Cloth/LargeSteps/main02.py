import numpy as np
from shearCondition import *

node_num = 3
node_pos = np.zeros((node_num,3))
node_vel = np.zeros((node_num,3))

node_pos = np.array([[0,0,0],
                     [1,0.5,0],
                     [0.5,1,0]],dtype = float)

element_num = 1
element = np.zeros((element_num,3))
element_uv = np.zeros((element_num,3,2))
element_mass = np.ones((element_num))
element = np.array([[0,1,2]],dtype = int)
element_uv[0,:,:] = np.array([[0,0],[1,0],[0,1]],dtype = float)

def computeW(uv,pos):
    duv21 = uv[1,:] - uv[0,:]
    duv31 = uv[2,:] - uv[0,:]
    dpos21 = pos[1,:] - pos[0,:]
    dpos31 = pos[2,:] - pos[0,:]
    
    det =  duv21[0]*duv31[1] - duv31[0]*duv21[1]
    # area = 0.5 * abs(det)
    
    wu = (dpos21 * duv31[1] - dpos31 * duv21[1]) / det
    wv = (-dpos21 * duv31[0] + dpos31 * duv21[0])/ det
    
    dwudx1_scalar = (duv21[1] - duv31[1]) / det
    dwudx2_scalar = duv31[1] / det
    dwudx3_scalar = - duv21[1] / det
    
    dwvdx1_scalar = (duv31[0] - duv21[0]) / det
    dwvdx2_scalar = - duv31[0] / det
    dwvdx3_scalar = duv21[0] / det
    
    dw = np.array([dwudx1_scalar,
                   dwudx2_scalar,
                   dwudx3_scalar,
                   dwvdx1_scalar,
                   dwvdx2_scalar,
                   dwvdx3_scalar])
    
    return wu,wv,dw
    

con = shearCondition()

time = 0
timeFinal = 1
dt = 1
while(time < timeFinal):
    time = 5
    
    dfdx = np.zeros((node_num*3,node_num*3))
    dfdv = np.zeros((node_num*3,node_num*3))
    force = np.zeros((node_num*3))
    vel = np.zeros((node_num*3))
    
    for ie in range(element_num):
        
        uv = element_uv[ie,:,:]
        pos = node_pos[element[ie,:],:]
        wu,wv,dw = computeW(uv, pos)
        con.computeShear(wu,wv,dw,pos)
        con.computeShearForce(force,dfdx,dfdv,element[ie,:])
        
    for i in range(node_num):
        vel[i*3] = node_vel[i,0]
        vel[i*3+1] = node_vel[i,1]
        vel[i*3+2] = node_vel[i,2]
        
    Amat = np.identity(node_num*3) - dfdx * dt * dt - dfdv * dt
    bvec = force * dt + np.dot(dfdx,vel) * dt * dt
        
    dv = np.dot(np.linalg.inv(Amat),bvec)
    node_dvel = np.zeros((node_num,3))
    for i in range(node_num):
        node_dvel[i,0] = dv[i*3]
        node_dvel[i,1] = dv[i*3+1]
        node_dvel[i,2] = dv[i*3+2]
        
        