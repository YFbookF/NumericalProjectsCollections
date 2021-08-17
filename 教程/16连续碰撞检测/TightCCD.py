import numpy as np

def cross(vec0,vec1):
    res = np.zeros((3))
    res[0] = vec0[1]*vec1[2] - vec0[2]*vec1[1]
    res[1] = vec0[2]*vec1[0] - vec0[0]*vec1[2]
    res[2] = vec0[0]*vec1[1] - vec0[1]*vec1[0]
    return res

def norm(vec0,vec1,vec2):
    return cross(vec1 - vec0,vec2 - vec0)

def dot(vec0,vec1):
    res = 0
    for i in range(len(vec0)):
        res += vec0[i]*vec1[i]
    return res

pos0 = np.array([0.75,0.75,1]) # 点P的位置 
pos1 = np.array([1,0,0]) # 三角形顶点 1 的位置
pos2 = np.array([0,1,0]) # 三角形顶点 2 的位置
pos3 = np.array([1,1,0]) # 三角形顶点 3 的位置

vel0 = np.array([0,0,-0.5]) # 点 P 的速度
vel1 = np.array([0,0,0]) # 三角形顶点 1 的速度
vel2 = np.array([0,0,0]) # 三角形顶点 2 的速度
vel3 = np.array([0,0,0]) # 三角形顶点 3 的速度

v01 = vel0 - vel1
v21 = vel2 - vel1
v31 = vel3 - vel1

x01 = pos0 - pos1
x21 = pos2 - pos1
x31 = pos3 - pos1

n0 = norm(pos1,pos2,pos3)
n1 = norm(pos1 + vel1,pos2 + vel2,pos3 + vel3)
delta = norm(vel1,vel2,vel3)
nX = (n0 + n1 - delta) / 2

pa0 = pos0 - pos1
pa1 = pos0 + vel0 - pos1 - vel1

A = dot(n0, pa0)
B = dot(n1, pa1)
C = dot(nX, pa0)
D = dot(nX, pa1)
E = dot(n1, pa0)
F = dot(n0, pa1)

p0 = A
p1 = C * 2 + F
p2 = D * 2 + E
p3 = B 

# 这玩意的解应该是绮罗罗·伯恩斯坦多项式
# x = 0.5 ,
# b03 = 1/8 b13 = 3/8 b23 = 3/8 b33 = 1/8