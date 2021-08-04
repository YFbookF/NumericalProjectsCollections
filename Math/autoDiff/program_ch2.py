import numpy as np

L = np.array([[1,0],[1,1]]) # L 是 单位下三角矩阵
D = np.array([[1,0],[0,1]]) # D 是 单位上三角矩阵
what = np.dot(np.dot(L,D),np.transpose(L))

inv = np.linalg.inv(L)
'''

L D L^T = [1 1]
          [1 2]
          
L D L^T x= [1 1][4] = d = [9]
           [1 2][5]       [14]
           
根据书上所说，x既可以是变量，也可以是dx。
我觉得这是个废话，因为这是个线性方程组
并且下面的程序与自动微分也没什么关系
          
'''
n = 2 # 矩阵的长度
v = np.zeros((n))
x = np.array([9,14]) 
for i in range(n):
    v[i] = x[i]
    for j in range(i):
        v[i] -= L[i,j] * x[j]
w = np.zeros((n))
for i in range(n):
    w[i] = v[i] / D[i,i]
y = np.zeros((n))
for i in range(n-1,-1,-1):
    y[i] = w[i]
    for j in range(i+1,n):
        y[i] -= L[j,i] * w[j]