# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:55:14 2021
1D_ElastoPlasticity-master
@author: acer
"""
import numpy as np
level = 1
Ec = 206900
H_m = 10000
sigma_Y = 450 * np.sqrt(2.0 / 3)
traction_force = 1000
size_x = 1

node_num = 3
elem_num = node_num - 1
node_pos = np.zeros((node_num))
elem_idx = np.zeros((elem_num,2),dtype = int)
for i in range(node_num):
    node_pos[i] = float(i) / (node_num - 1)
for i in range(elem_num):
    elem_idx[i,0] = i
    elem_idx[i,1] = i + 1
xi = 0
weight = 2
HatP = np.array([[1-xi],[1+xi]])*0.5
dHatP = np.array([-0.5,0.5])
dPhi = np.zeros((elem_num,2))
det = 0
for i in range(elem_num):
    pos0 = node_pos[elem_idx[i,0]]
    pos1 = node_pos[elem_idx[i,1]]
    J = pos0 * dHatP[0] + pos1 * dHatP[1]
    det = J
    Jinv = 1.0 / J 
    dPhi[i,:] = Jinv * dHatP
    
Bmat = np.zeros((elem_num,elem_num + 1))
Dmat = np.zeros((elem_num,elem_num))
for i in range(elem_num):
    Bmat[i,i] = dPhi[i,0]
    Bmat[i,i+1] = dPhi[i,1]
    Dmat[i,i] = abs(det) * weight * Ec
    
Kmat = np.dot(Bmat.T,np.dot(Dmat,Bmat))

zeta = np.zeros((45))
for i in range(1,45):
    if i < 11:
        zeta[i] = zeta[i - 1] + 0.1
    elif i < 33:
        zeta[i] = zeta[i - 1] - 0.1
    else:
        zeta[i] = zeta[i - 1] + 0.1
        
int_num = elem_num
U = np.zeros((node_num))
U_zeros = np.zeros((node_num))
dU = np.zeros((node_num))
U_old = np.zeros((node_num))
F = np.zeros((node_num)) # internal forces
E = np.zeros((int_num)) # strain tensors at integration point
S_old = np.zeros((int_num)) # previous stress tensors at intergration point
S_new = np.zeros((int_num)) # previois stress tensors at intergration point
Hard_old = np.zeros((int_num)) # hardening tensors at intergarion point
Hard_new = np.zeros((int_num)) # hardening tensors at intergration point 

it0 = 1
it0_max = 45

while(it0 < it0_max):
    it0 += 1
    f_t = traction_force / weight * (zeta[it0] - zeta[it0 - 1])
    S_old = S_new.copy()
    Hard_old = Hard_new.copy()
    U_it = U_zeros.copy()
    
    it1 = 0
    it1_max = 50
    while(it1 < it1_max):
        it1 += 1
        E = np.dot(Bmat,U_it)
        hard_part = np.ones((int_num)) * sigma_Y + H_m * Hard_old
        S_el = Ec * E
        S_tr = S_old + S_el
        Ds_tr = (Ec * H_m) / (Ec + H_m)
        Ds = np.ones((int_num)) * Ec
        tau_eps = 1e-6
        S = np.zeros((int_num))
        denom = Ec + H_m
        phi1 = - S_tr - hard_part
        phi2 = S_tr - hard_part
        for i in range(int_num):
            S[i] = S_el[i]
            if S_tr[i] + hard_part[i] < - tau_eps:
                S[i] = S_el[i] + Ec / denom * phi1[i]
                Ds[i] = Ds_tr
            if S_tr[i] - hard_part[i] > tau_eps:
                S[i] = S_el[i] - Ec / denom * phi2[i]
                Ds[i] = Ds_tr
        D_p = np.zeros((int_num,int_num))
        for i in range(int_num):
            D_p[i,i] = Ds[i] * 0.5
        K_tangent = Kmat + np.dot(Bmat.T,np.dot(D_p - Dmat,Bmat))
        F = np.dot(Bmat.T,S * 0.5)
        for i in range(1,node_num-1):
            dU[i] = (f_t - F[i]) / K_tangent[i,i]
        U_new = U_it + dU
        q1sqr = np.dot(dU.T,np.dot(Kmat,dU))
        q2sqr = np.dot(U_it.T,np.dot(Kmat,U_it))
        q3sqr = np.dot(U_new.T,np.dot(Kmat,U_new))
        criterion = np.sqrt(q1sqr / (q2sqr + q3sqr))
        U_it = U_new.copy()
        if criterion < 1e-12:
            break
    U = U_old + U_it
    E = np.dot(Bmat,U_it)
    hard_part = np.ones((int_num)) * sigma_Y + H_m * Hard_old
    S_el = Ec * E
    S_tr = S_old + S_el
    Ds_tr = (Ec * H_m) / (Ec + H_m)
    Ds = np.ones((int_num)) * Ec
    tau_eps = 1e-6
    S = np.zeros((int_num))
    denom = Ec + H_m
    phi1 = - S_tr - hard_part
    phi2 = S_tr - hard_part
    for i in range(int_num):
        S[i] = S_el[i]
        if S_tr[i] + hard_part[i] < - tau_eps:
            S[i] = S_el[i] + Ec / denom * phi1[i]
        if S_tr[i] - hard_part[i] > tau_eps:
            S[i] = S_el[i] - Ec / denom * phi2[i]
    S_new = S_old + S
    U_old = U.copy()
    Hard_new = Hard_old + Hard_new
    check = 1
        
        
        