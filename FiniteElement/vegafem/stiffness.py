import numpy as np

# corotational linear fem

element_num = 1
node_num = 4
coord = np.array([[0,0,0],
                  [2,0,0],
                  [1,1,-1],
                  [1,1,1]],dtype = float)
element = np.zeros((element_num,4))
element[0,:] = np.array([0,1,2,3])

for ie in range(element_num):
    m = np.ones((4,4))
    m[0:3,0] = coord[element[ie,0],:]
    m[0:3,1] = coord[element[ie,1],:]
    m[0:3,2] = coord[element[ie,2],:]
    m[0:3,3] = coord[element[ie,3],:]
    '''
         [v0_x  v1_x  v2_x  v3_x]
    M =  [v0_y  v1_y  v2_y  v3_y]
         [v0_z  v1_z  v2_z  v3_z]
         [1      1      1     1 ]
    '''
    minv = np.linalg.inv(m)
    
    # 是不是反了？3，7，11，15不见了
    B = np.zeros((6,12))
    B = np.array([[minv[0,0],0,0,            minv[1,0],0,0,             minv[2,2],0,0,             minv[3,1],0,0],
                  [0,minv[0,1],0,            0,minv[1,1],0,             0,minv[2,3],0,             0,minv[3,2],0],
                  [0,0,minv[0,2],            0,0,minv[1,2],             0,0,minv[2,4],             0,0,minv[3,3]],
                  [minv[0,1],minv[0,0],0,    minv[1,1],minv[1,0],0,     minv[2,1],minv[2,0],0,     minv[3,1],minv[3,0],0],
                  [0,minv[0,2],minv[0,1],    minv[1,2],minv[1,1],0,     minv[2,2],minv[2,1],0,     minv[3,2],minv[3,1],0],
                  [minv[0,2],0,minv[0,0],    minv[1,2],0,minv[1,0],     minv[2,2],minv[2,0],0,     minv[3,2],minv[3,0]]])
    
    lamb = 1
    mu = 1
    E = np.zeros((6,6))
    E = np.array([[lamb + 2*mu,lamb,lamb,0,0,0],
                  [lamb,lamb + 2*mu,lamb,0,0,0],
                  [lamb,lamb,lamb + 2*mu,0,0,0],
                  [0,0,0,mu,0,0],
                  [0,0,0,0,mu,0],
                  [0,0,0,0,0,mu]])
    
    K = np.zeros((12,12))
    
    # 刚度矩阵
    K = np.dot(np.dot(B.T,E),B)
    