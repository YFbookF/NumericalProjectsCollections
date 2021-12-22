# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 12:35:11 2021
https://github.com/matlabfem/matlab_fem_elastoplasticity
plasicity_dp_2d
@author: acer
"""
import numpy as np

node_row_num = 3
dx = 1.0 / (node_row_num - 1)
node_num = node_row_num * node_row_num
element_row_num = node_row_num - 1
element_num = element_row_num * element_row_num * 2

element_idx = np.zeros((element_num,3),dtype = int)
element_minv = np.zeros((element_num,2,2))

node_pos = np.zeros((node_num,2),dtype = float)
node_vel = np.zeros((node_num,2),dtype = float)
node_force = np.zeros((node_num,2),dtype = float)

for j in range(node_row_num):
    for i in range(node_row_num):
        node_pos[j*node_row_num+i] = np.array([i,j]) * dx
cnt = 0
for j in range(element_row_num):
    for i in range(element_row_num):
        idx = j * node_row_num + i
        element_idx[cnt,0] = idx
        element_idx[cnt,1] = idx + 1
        element_idx[cnt,2] = idx + node_row_num
        cnt += 1
        element_idx[cnt,0] = idx + 1
        element_idx[cnt,1] = idx + node_row_num + 1
        element_idx[cnt,2] = idx + node_row_num
        cnt += 1
        

Xi = np.array([[1.0/3],[1.0/3]])
HatP = np.array([1 - Xi[0,:] - Xi[1,:],Xi[0,:],Xi[1,:]])
DHatP1 = np.array([-1.0,1.0,0.0])
DHatP2 = np.array([-1.0,0.0,1.0])
Wf = 0.5
    
# 每个三角形第0个元素和第一个元素的坐标系
Dphi1 = np.zeros((3,element_num))
Dphi2 = np.zeros((3,element_num))
for ie in range(element_num):
    J_00 = 0
    J_01 = 0
    J_10 = 0
    J_11 = 0
    for it in range(3):
        valx = node_pos[element_idx[ie,it],0]
        valy = node_pos[element_idx[ie,it],1]
        J_00 += valx * DHatP1[it]
        J_01 += valy * DHatP1[it]
        J_10 += valx * DHatP2[it]
        J_11 += valy * DHatP2[it]
    det = J_00 * J_11 - J_01 * J_10
    Jinv_00 = J_11 / det
    Jinv_11 = J_00 / det
    Jinv_01 = - J_01 / det
    Jinv_10 = J_10 / det
    Dphi1[:,ie] = Jinv_00 * DHatP1[:] + Jinv_01 * DHatP2[:]
    Dphi2[:,ie] = Jinv_10 * DHatP1[:] + Jinv_11 * DHatP2[:]
    
node_per_element = 3
n_b = 6 * node_per_element
Bmat = np.zeros((n_b,8))
for j in range(8):
    for i in range(0,node_per_element):
        Bmat[6*i+0,j] = Dphi1[i,j]
        Bmat[6*i+5,j] = Dphi1[i,j]
        Bmat[6*i+4,j] = Dphi2[i,j]
        Bmat[6*i+2,j] = Dphi2[i,j]
    
young = 1e7 # young`s module
poisson = 0.48
shear = young / (2 * (1 + poisson))
bulk = young / ( 3 * (1 - 2 * poisson))
vol = np.array([[1.0,1.0,0],[1.0,1.0,0],[0,0,0]])
dev = np.array([[1.0,0,0],[0,1.0,0],[0,0,0.5]]) - vol * 0.3333
elast = 2 * dev * shear + vol * bulk
  
Dmat = np.zeros((9,element_num))
for j in range(3):
    for i in range(3):
        for k in range(element_num):
            Dmat[j*3+i,k] = elast[i,j] * Wf
        
Kmat = np.dot(Bmat.T,np.dot(Dmat,Bmat))

c0 = 450
phi = np.pi / 9
eta = 3*np.tan(phi)/np.sqrt(9+12*(np.tan(phi))**2)

zeta_old = 0 # load factor
d_zeta = 1e-3 # load increment
pressure_old = 0

while True:
    zeta = zeta_old + d_zeta
    
    it_max = 25
    it = 0
        