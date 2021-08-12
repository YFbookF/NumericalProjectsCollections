import numpy as np
import sys
# 专门实现 please_Compute_Porous_Slip_Velocity
# 已完成 2021-8-12
# 我真心觉得这种把大项目拆解成小项目的模式更加容易学习
# 毕竟很多研究都报告，愚蠢的人类只擅长处理模块化的任务

xPos = np.array([0.7,0.6998,0.6990,0.6978,0.6962,0.6940,0.6914,0.6883]) # 多孔介质点的位置
yPos = np.array([1,1.0196,1.0392,1.0587,1.0780,1.0972,1.1161,1.1348])  # 多孔介质点的位置
c = np.array([-2,-1,0,0,0,0,1,2]) # 有限差分方向
Fx = np.array([-4818,-4812,-4795,-4766,-4725,-4673,-4610,-4536])
Fy = np.array([0,-472,-944,-1413,-1880,-2341,-2797,-3246,-3246])

ds = 0.0156

Np = 8
xL_s = np.zeros((Np))
yL_s = np.zeros((Np))

# give_Me_Lagrangian_Derivatives
# https://zh.wikipedia.org/wiki/%E6%9C%89%E9%99%90%E5%B7%AE%E5%88%86%E4%BF%82%E6%95%B8
# 就是有限差分，不过阶数很高，平常难以见到所以不熟悉
for i in range(Np):
    if c[i] == -2:
        xL_s[i] = (-25.0/12*xPos[i] + 4*xPos[i+1] - 3*xPos[i+2] + 4/3*xPos[i+3] - 1/4*xPos[i+4]) / ds
        yL_s[i] = (-25.0/12*yPos[i] + 4*yPos[i+1] - 3*yPos[i+2] + 4/3*yPos[i+3] - 1/4*yPos[i+4]) / ds
    elif c[i] == -1:
        xL_s[i] = (-0.25*xPos[i-1] - 5/6*xPos[i] + 1.5*xPos[i+1] -0.5*xPos[i+2] + 1/12*xPos[i+3]) / ds
        yL_s[i] = (-0.25*yPos[i-1] - 5/6*yPos[i] + 1.5*yPos[i+1] -0.5*yPos[i+2] + 1/12*yPos[i+3]) / ds
    elif c[i] == 0:
        xL_s[i] = (xPos[i-2]/12 - 2/3*xPos[i-1] + 2/3*xPos[i+1] - xPos[i+2]/12) / ds
        yL_s[i] = (yPos[i-2]/12 - 2/3*yPos[i-1] + 2/3*yPos[i+1] - yPos[i+2]/12) / ds
    elif c[i] == 1:
        xL_s[i] = (-xPos[i-3]/12 + xPos[i-2]/2 - xPos[i-1]*3/2 + xPos[i]*5/6 + xPos[i+1]/4) / ds
        yL_s[i] = (-yPos[i-3]/12 + yPos[i-2]/2 - yPos[i-1]*3/2 + yPos[i]*5/6 + yPos[i+1]/4) / ds
    elif c[i] == 2:
        xL_s[i] = (xPos[i-4]/4 - 4*xPos[i-3]/3 + xPos[i-2]*3 - 4*xPos[i-1] + xPos[i]*25/12) / ds
        yL_s[i] = (yPos[i-4]/4 - 4*yPos[i-3]/3 + yPos[i-2]*3 - 4*yPos[i-1] + yPos[i]*25/12) / ds
    else:
        print("error")
        sys.exit()
        
# 计算法向量
Nx = np.zeros((Np))
Ny = np.zeros((Np))
sqrtN = np.zeros((Np))
for i in range(Np):
    sqrtN[i] = np.sqrt(xL_s[i]**2 + yL_s[i]**2)
    Nx[i] = yL_s[i] / sqrtN[i]
    Ny[i] = - xL_s[i] / sqrtN[i]
    
# Darcy`s Law
porous = 1e-4 # 多孔介质系数，也就是那个Alpha
Ux = np.zeros((Np))
Uy = np.zeros((Np))
for i in range(Np):
    Ux[i] = - porous  * Nx[i] * Fx[i] / sqrtN[i]
    Uy[i] = - porous  * Ny[i] * Fy[i] / sqrtN[i]