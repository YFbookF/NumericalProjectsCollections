import numpy as np
# 初始化三角形初始位置

node_num = 4

node_pos = np.array([[0,0],[1,0],[0,1],[1,1]],dtype = float)

node_dx = np.zeros((node_num,2))

element_num = 2

element = np.array([[0,1,2],[2,1,3]])

element_minv = np.zeros((element_num,2,2))

element_area = np.zeros((element_num))

for i in range(element_num):
    p0 = node_pos[element[i,0],:]
    p1 = node_pos[element[i,1],:]
    p2 = node_pos[element[i,2],:]
    
    Ds = np.array([[p1[0] - p0[0],p2[0] - p0[0]],[p1[1] - p0[1],p2[1] - p0[1]]])
    
    element_minv[i,:,:] = np.linalg.inv(Ds)

node_pos[1,0] = 0.2

# https://www.continuummechanics.org/tensornotationbasic.html
def doubleDotProduct(A,B):
    return A[0,0]*B[0,0] + A[1,0]*B[1,0] + A[0,1]*B[0,1] + A[1,1]*B[1,1]

time = 0
timeFinal = 3000
while(time < timeFinal):
    time += 1
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
        
        node_dx[0,:] += dt * gradC0
        node_dx[1,:] += dt * gradC1
        node_dx[2,:] += dt * gradC2
        
        # node_dx[element[i,0],:] += energy / sumGradC * invMass * gradC0
        # node_dx[element[i,1],:] += energy / sumGradC * invMass * gradC1
        # node_dx[element[i,2],:] += energy / sumGradC * invMass * gradC2
        
        element_area[i] = np.cross(node_pos[1,:] - node_pos[0,:],node_pos[2,:] - node_pos[0,:])*0.5
    
    node_pos += dt * node_dx
    node_dx[:,:] = 0
    
