import numpy as np

class Condition:
    
    def __init__(self):
        self.alpha = 1
    
    def computeShear(self,wu,wv,dw,pos):
        dwudx1_scalar = dw[0]
        dwudx2_scalar = dw[1]
        dwudx3_scalar = dw[2]
    
        dwvdx1_scalar = dw[3]
        dwvdx2_scalar = dw[4]
        dwvdx3_scalar = dw[5]
        
        self.c = self.alpha * (wu[0]*wv[0] + wu[1]*wv[1])
        
        # Shear Condition 的 一阶导，3x1 矩阵
        self.dcdx1 = self.alpha * (dwudx1_scalar * wv + dwvdx1_scalar * wu)
        self.dcdx2 = self.alpha * (dwudx2_scalar * wv + dwvdx2_scalar * wu)
        self.dcdx3 = self.alpha * (dwudx3_scalar * wv + dwvdx3_scalar * wu)
        
        # Shear Condition 的 二阶导，3x3 矩阵
        self.d2cdx1x1 = self.alpha * 2 * dwudx1_scalar * dwvdx1_scalar * np.identity(3) 
        self.d2cdx1x2 = self.alpha * (dwudx1_scalar * dwvdx2_scalar 
                                            + dwvdx1_scalar * dwudx2_scalar) * np.identity(3)
        self.d2cdx1x3 = self.alpha * (dwudx1_scalar * dwvdx3_scalar 
                                            + dwvdx1_scalar * dwudx3_scalar) * np.identity(3)
        
        self.d2cdx2x1 = self.d2cdx1x2.copy()
        self.d2cdx2x2 = self.alpha * 2 * dwudx2_scalar * dwvdx2_scalar * np.identity(3) 
        self.d2cdx2x3 = self.alpha * (dwudx2_scalar * dwvdx3_scalar 
                                            + dwvdx2_scalar * dwudx3_scalar) * np.identity(3)
        
        self.d2cdx3x1 = self.d2cdx1x3.copy()
        self.d2cdx3x2 = self.d2cdx2x3.copy()
        self.d2cdx2x2 = self.alpha * 2 * dwudx3_scalar * dwvdx3_scalar * np.identity(3) 
        
        # shear Condition 对时间求一阶导，是一个标量
        term1 = self.dcdx1[0]*pos[0,0] + self.dcdx1[1]*pos[1,0] + self.dcdx1[2]*pos[2,0]
        term2 = self.dcdx2[0]*pos[0,1] + self.dcdx2[1]*pos[1,1] + self.dcdx2[2]*pos[2,1]
        term3 = self.dcdx3[0]*pos[0,2] + self.dcdx3[1]*pos[1,2] + self.dcdx3[2]*pos[2,2]
        
        self.dcdt = term1 + term2 + term3
        
    def computeShearForce(self,force,dfdx,dfdv,idx):
        force[3*idx[0]:3*idx[0]+3] += - self.k * self.c * self.dcdx1
        force[3*idx[1]:3*idx[1]+3] += - self.k * self.c * self.dcdx2
        force[3*idx[2]:3*idx[2]+3] += - self.k * self.c * self.dcdx3
        
        # 3 x 3 矩阵
        term = np.dot(self.dcdx1,np.transpose(self.dcdx1))
        df1dx1 = - self.k * (term + self.c * self.d2cdx1x1)
        df1dv1 = - self.d * term
        term = np.dot(self.dcdx1,np.transpose(self.dcdx2))
        df1dx2 = - self.k * (term + self.c * self.d2cdx1x2)
        df1dv2 = - self.d * term
        term = np.dot(self.dcdx1,np.transpose(self.dcdx3))
        df1dx3 = - self.k * (term + self.c * self.d2cdx1x3)
        df1dv3 = - self.d * term
        
        term = np.dot(self.dcdx2,np.transpose(self.dcdx1))
        df2dx1 = - self.k * (term + self.c * self.d2cdx2x1)
        df2dv1 = - self.d * term
        term = np.dot(self.dcdx2,np.transpose(self.dcdx2))
        df2dx2 = - self.k * (term + self.c * self.d2cdx2x2)
        df2dv2 = - self.d * term
        term = np.dot(self.dcdx2,np.transpose(self.dcdx3))
        df2dx3 = - self.k * (term + self.c * self.d2cdx2x3)
        df2dv3 = - self.d * term
        
        term = np.dot(self.dcdx3,np.transpose(self.dcdx1))
        df3dx1 = - self.k * (term + self.c * self.d2cdx3x1)
        df3dv1 = - self.d * term
        term = np.dot(self.dcdx3,np.transpose(self.dcdx2))
        df3dx2 = - self.k * (term + self.c * self.d2cdx3x2)
        df3dv2 = - self.d * term
        term = np.dot(self.dcdx3,np.transpose(self.dcdx3))
        df3dx3 = - self.k * (term + self.c * self.d2cdx3x3)
        df3dv3 = - self.d * term
        
        for i in range(3):
            for j in range(3):
                dfdx[3*idx[0]+i,3*idx[0]+j] += df1dx1[i,j]
                dfdx[3*idx[0]+i,3*idx[1]+j] += df1dx2[i,j]  
                dfdx[3*idx[0]+i,3*idx[2]+j] += df1dx3[i,j]
                
                dfdx[3*idx[1]+i,3*idx[0]+j] += df2dx1[i,j]  
                dfdx[3*idx[1]+i,3*idx[1]+j] += df2dx2[i,j]  
                dfdx[3*idx[1]+i,3*idx[2]+j] += df2dx3[i,j]  
                
                dfdx[3*idx[2]+i,3*idx[0]+j] += df3dx1[i,j]  
                dfdx[3*idx[2]+i,3*idx[1]+j] += df3dx2[i,j]  
                dfdx[3*idx[2]+i,3*idx[2]+j] += df3dx3[i,j] 
                
        for i in range(3):
            for j in range(3):
                dfdv[3*idx[0]+i,3*idx[0]+j] += df1dv1[i,j]
                dfdv[3*idx[0]+i,3*idx[1]+j] += df1dv2[i,j]  
                dfdv[3*idx[0]+i,3*idx[2]+j] += df1dv3[i,j]
                
                dfdv[3*idx[1]+i,3*idx[0]+j] += df2dv1[i,j]  
                dfdv[3*idx[1]+i,3*idx[1]+j] += df2dv2[i,j]  
                dfdv[3*idx[1]+i,3*idx[2]+j] += df2dv3[i,j]  
                
                dfdv[3*idx[2]+i,3*idx[0]+j] += df3dv1[i,j]  
                dfdv[3*idx[2]+i,3*idx[1]+j] += df3dv2[i,j]  
                dfdv[3*idx[2]+i,3*idx[2]+j] += df3dv3[i,j] 
                
        # 3 x 3 矩阵
        