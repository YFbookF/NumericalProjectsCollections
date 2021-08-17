import numpy as np

# basic deformation
F = np.array([[1,0,0],[0,1.1,0],[0,0,1/1.1]])

# rotation
t = -45 # degree
t = t / 360 * 2 * np.pi
T = np.array([[np.cos(t),-np.sin(t),0],
              [np.sin(t),np.cos(t),0],
              [0,0,1]])

F = np.dot(T,F)

# polar decomposition
# Right Stretch tensor
C = np.dot(np.transpose(F),F)

Cval,Cvec = np.linalg.eig(C)

Cval3x3 = np.zeros((3,3))
for i in range(3):
    Cval3x3[i,i] = Cval[i]

# Right Stretch Tensor
temp = np.dot(Cvec,np.sqrt(Cval3x3))
U = np.dot(temp,np.linalg.inv(Cvec))

# Rotation in undef config
R = np.dot(F,np.linalg.inv(U))

# # left cauchy green tensor
# B = np.dot(F,np.transpose(F))
# Bval,Bvec = np.linalg.eig(B)
# Bval3x3 = np.zeros((3,3))
# for i in range(3):
#     Bval3x3[i,i] = Bval[i]

# # Left Stretch Tensor
# temp = np.dot(Cvec,np.sqrt(Bval3x3))
# V = np.dot(temp,np.linalg.inv(Bvec))
# R = np.dot(F,np.linalg.inv(V))

# "Abaqus Local" deformation gradient
Fal = np.dot(np.dot(R.T,F),R)

# classic deformation
Fd = np.dot(R.T,F)