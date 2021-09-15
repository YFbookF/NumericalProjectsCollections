import numpy as np
# 初始化三角形初始位置

node_num = 3

node_pos = np.array([[0,0],[1,0],[0,1]],dtype = float)
node_dx = np.zeros((node_num,2))

element_num = 1

element = np.array([[0,1,2]])

element_minv = np.zeros((element_num,2,2))

element_area = np.zeros((element_num))

kmat = np.zeros((node_num*2,node_num*2))

for i in range(element_num):
    p0 = node_pos[element[i,0],:]
    p1 = node_pos[element[i,1],:]
    p2 = node_pos[element[i,2],:]
    
    Ds = np.array([[p1[0] - p0[0],p2[0] - p0[0]],[p1[1] - p0[1],p2[1] - p0[1]]])
    
    element_minv[i,:,:] = np.linalg.inv(Ds)

node_pos[0,0] = -0.1

# https://www.continuummechanics.org/tensornotationbasic.html
def doubleDotProduct(A,B):
    return A[0,0]*B[0,0] + A[1,0]*B[1,0] + A[0,1]*B[0,1] + A[1,1]*B[1,1]

def trace(A):
    return A[0,0] + A[1,1]

time = 0
timeFinal = 100
while(time < timeFinal):
    time += 1
    node_dx[:,:] = 0
    kmat[:,:] = 0
    for i in range(element_num):
        p0 = node_pos[element[i,0],:]
        p1 = node_pos[element[i,1],:]
        p2 = node_pos[element[i,2],:]
        
        Ds = np.array([[p1[0] - p0[0],p2[0] - p0[0]],[p1[1] - p0[1],p2[1] - p0[1]]])
        # 形变梯度
        F = np.dot(Ds,element_minv[i,:,:])
        # lame常数
        mu = 1
        # lame常数
        la = 1
        
        pJpF = np.array([[F[1,1],-F[1,0]],[-F[0,1],F[0,0]]])
        J = max(0.01,np.linalg.det(F))
        logJ = np.log(J)
        I_c = F[0,0]**2 + F[1,1]**2
        energy = 0.5 * mu * (I_c - 2) - mu * logJ + 0.5 * la * logJ**2
        piola = mu * (F - 1.0 / J * pJpF) + la * logJ / J * pJpF
        # Jminus = np.linalg.det(F) - 1.0 -  mu / la
        # piola = mu * F + la * Jminus * pJpF
        # 三角形面积
        area = 0.5
        # 计算力
        mm = np.linalg.inv(element_minv[i,:,:])
        H = area * np.dot(piola,mm.transpose())
        
        gradC0 = np.array([H[0,0],H[1,0]])
        gradC1 = np.array([H[0,1],H[1,1]])
        gradC2 = np.array([-H[0,0]-H[0,1],-H[1,0]-H[1,1]])
        
        node_force = np.zeros((3,2))
        #第一个顶点
        node_force[0,:] = gradC0
        #第二个顶点
        node_force[1,:] = gradC1
        #第三个顶点
        node_force[2,:] = gradC2
        
        invMass = 1
        
        dt = 0.1
        sumGradC = invMass * (gradC0[0]**2 + gradC0[1]**2)
        sumGradC += invMass * (gradC1[0]**2 + gradC1[1]**2)
        sumGradC += invMass * (gradC2[0]**2 + gradC2[1]**2)
        
        if sumGradC < 1e-10:
            break
        energy = 1
        # node_dx[0,:] += dt * gradC0
        # node_dx[1,:] += dt * gradC1
        # node_dx[2,:] += dt * gradC2
        
        node_dx[element[i,0],:] += energy / sumGradC * invMass * gradC0
        node_dx[element[i,1],:] += energy / sumGradC * invMass * gradC1
        node_dx[element[i,2],:] += energy / sumGradC * invMass * gradC2
        
        element_area[i] = np.cross(node_pos[1,:] - node_pos[0,:],node_pos[2,:] - node_pos[0,:])*0.5
    

    for ie in range(element_num):
        dD1dx = np.array([[1,0],[0,0]])
        dD1dy = np.array([[0,0],[1,0]])
        dD2dx = np.array([[0,1],[0,0]])
        dD2dy = np.array([[0,0],[0,1]])
        dD0dx = - dD1dx - dD2dx
        dD0dy = - dD1dy - dD2dy
        
        p0 = node_pos[element[i,0],:]
        p1 = node_pos[element[i,1],:]
        p2 = node_pos[element[i,2],:]
        Ds = np.array([[p1[0] - p0[0],p2[0] - p0[0]],[p1[1] - p0[1],p2[1] - p0[1]]])
        minv = element_minv[i,:,:]
        F = np.dot(Ds,minv)
        dF0dx = np.dot(dD0dx,minv)
        dF0dy = np.dot(dD0dy,minv)
        dF1dx = np.dot(dD1dx,minv)
        dF1dy = np.dot(dD1dy,minv)
        dF2dx = np.dot(dD2dx,minv)
        dF2dy = np.dot(dD2dy,minv)
        F_inv = np.linalg.inv(F)
        F_inv_T = F_inv.T
        logJ = np.log(max(np.linalg.det(F),0.01))
        dP0dx = mu * dF0dx + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF0dx.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF0dx)) * F_inv_T
        dP0dy = mu * dF0dy + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF0dy.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF0dy)) * F_inv_T
        dP1dx = mu * dF1dx + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF1dx.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF1dx)) * F_inv_T
        dP1dy = mu * dF1dy + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF1dy.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF1dy)) * F_inv_T
        dP2dx = mu * dF2dx + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF2dx.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF2dx)) * F_inv_T
        dP2dy = mu * dF2dy + (mu - la * logJ) * np.dot(np.dot(F_inv.T,dF2dy.T),F_inv.T
                        ) + la * trace(np.dot(F_inv,dF2dy)) * F_inv_T
        area = 0.5
        tm = np.transpose(minv)
        # dH0dx = np.dot(np.dot(dP0dx,dF0dx),tm) * area
        # dH0dy = np.dot(np.dot(dP0dy,dF0dy),tm) * area
        # dH1dx = np.dot(np.dot(dP1dx,dF1dx),tm) * area
        # dH1dy = np.dot(np.dot(dP1dy,dF1dy),tm) * area
        # dH2dx = np.dot(np.dot(dP2dx,dF2dx),tm) * area
        # dH2dy = np.dot(np.dot(dP2dy,dF2dy),tm) * area
        
        dH0dx = np.dot(dP0dx,tm) * area
        dH0dy = np.dot(dP0dy,tm) * area
        dH1dx = np.dot(dP1dx,tm) * area
        dH1dy = np.dot(dP1dy,tm) * area
        dH2dx = np.dot(dP2dx,tm) * area
        dH2dy = np.dot(dP2dy,tm) * area
        idx0 = element[i,0]
        idx1 = element[i,1]
        idx2 = element[i,2]
        
        idxx = idx0 * 2 + 0
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH0dx[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH0dx[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH0dx[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH0dx[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH0dx[0,0] - dH0dx[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH0dx[1,0] - dH0dx[1,1]
        
        idxx = idx0 * 2 + 1
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH0dy[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH0dy[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH0dy[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH0dy[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH0dy[0,0] - dH0dy[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH0dy[1,0] - dH0dy[1,1]
        
        idxx = idx1 * 2 + 0
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH1dx[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH1dx[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH1dx[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH1dx[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH1dx[0,0] - dH1dx[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH1dx[1,0] - dH1dx[1,1]
        
        idxx = idx1 * 2 + 1
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH1dy[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH1dy[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH1dy[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH1dy[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH1dy[0,0] - dH1dy[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH1dy[1,0] - dH1dy[1,1]
        
        idxx = idx2 * 2 + 0
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH2dx[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH2dx[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH2dx[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH2dx[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH2dx[0,0] - dH2dx[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH2dx[1,0] - dH2dx[1,1]
        
        idxx = idx2 * 2 + 1
        idxy = idx1 * 2 + 0
        kmat[idxx,idxy] += dH2dy[0,0]
        idxy = idx1 * 2 + 1
        kmat[idxx,idxy] += dH2dy[1,0]
        idxy = idx2 * 2 + 0
        kmat[idxx,idxy] += dH2dy[0,1]
        idxy = idx2 * 2 + 1
        kmat[idxx,idxy] += dH2dy[1,1]
        idxy = idx0 * 2 + 0
        kmat[idxx,idxy] += - dH2dy[0,0] - dH2dy[0,1]
        idxy = idx0 * 2 + 1
        kmat[idxx,idxy] += - dH2dy[1,0] - dH2dy[1,1]
    
    node_vel_flatten = np.zeros((node_num*2))
    for i in range(node_num):
        node_vel_flatten[i*2 + 0] = node_dx[i,0]
        node_vel_flatten[i*2 + 1] = node_dx[i,1]
    kkmat = np.identity(node_num*2) + kmat*0.2
    dx = np.dot(np.linalg.inv(kkmat),node_vel_flatten) 
    for i in range(node_num):
        node_pos[i,0] += dx[i * 2 + 0] * 0.5
        node_pos[i,1] += dx[i * 2 + 1] * 0.5
    t = 1
