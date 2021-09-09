import numpy as np
#TriDiagnoal Solver
a = np.array([0,1,1],dtype = float)
b = np.array([2,2,2],dtype = float)
c = np.array([1,1,0],dtype = float)

d = np.array([3,4,3],dtype = float)

c[0] /= b[0]
d[0] /= b[0]
n = 2
for i in range(1,n):
    c[i] = c[i] / (b[i] - a[i] * c[i - 1])
    d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1]);

d[n] = (d[n] - a[n] * d[n - 1]) / (b[n] - a[n] * c[n - 1]);

# 结果
for i in range(n-1,-1,-1):
    d[i] -= c[i] * d[i + 1]