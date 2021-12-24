import numpy as np

theta = 0 / 180 * np.pi
Re = np.array([[np.cos(theta),-np.sin(theta)],
               [np.sin(theta),np.cos(theta)]])
x0 = np.array([0,1],dtype = float)
x1 = np.dot(Re,x0)
Ke = np.array([[1,-1],[-1,1]],dtype = float)
Reinv = np.linalg.inv(Re)
Kmat1 = np.dot(Re,Ke)
Kmat = np.dot(np.dot(Re,Ke),np.linalg.inv(Re))
f0e = - np.dot(np.dot(Re,Ke),x0)


A = np.identity(2) + Kmat
bvec = - (np.dot(Kmat, x0) + f0e)
v = np.dot(np.linalg.inv(A),bvec)