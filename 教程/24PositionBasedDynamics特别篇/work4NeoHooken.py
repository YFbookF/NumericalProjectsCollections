import numpy as np

fg = np.array([[1.1,0.1,0],
               [0.05,0.9,0.15],
               [0.2,0,1.2]])

# polar decomposition
C = np.dot(np.transpose(fg),fg)

Cval,Cvec = np.linalg.eig(C)

Cval3x3 = np.zeros((3,3))
for i in range(3):
    Cval3x3[i,i] = Cval[i]

# Right Stretch Tensor
temp = np.dot(Cvec,np.sqrt(Cval3x3))
U = np.dot(temp,np.linalg.inv(Cvec))

# Rotation in undef config
R = np.dot(fg,np.linalg.inv(U))

# Material properties for the neo-Hookean constitutive model
props1 = 0.2   # C10 (MPa)
props2 = 2    # D1 (MPa^-1)

#NeoHookean

# Calculate the isotropic stress
# right Deformation tensor
B = np.dot(fg,np.transpose(fg))
J = np.linalg.det(fg)
J23 = J**(2/3)
I1 = B[0,0] + B[1,1] + B[2,2] # 黎曼第一不变量

# The volumetric part of the isotropic stress
kirch = 2 * (J - 1) * J / props2 * np.eye(3)
 


kirch = kirch + 2 * props1 * np.transpose(B) / J23 

kirch = kirch - 2 * props1 * I1 / 3 / J23 * np.eye(3) 

cstress = kirch / J

# Calculate pressure stress
pGlob = -(cstress[0,0] + cstress[1,1] + cstress[2,2])/3;