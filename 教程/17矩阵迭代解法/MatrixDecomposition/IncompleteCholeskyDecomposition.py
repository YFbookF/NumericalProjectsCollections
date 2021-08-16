import numpy as np
# 参考自维基百科
a = np.array([[1,2,4],[2,13,23],[4,23,77]])
n = 3
for k in range(n):
    a[k,k] = np.sqrt(a[k,k])
    for i in range(k+1,n):
        if a[i,k] != 0:
            a[i,k] = a[i,k] / a[k,k]
    for j in range(k+1,n):
        for i in range(j,n):
            if a[i,j] != 0:
                a[i,j] = a[i,j] - a[i,k] * a[j,k]
for i in range(n):
    for j in range(i+1,n):
        a[i,j] = 0