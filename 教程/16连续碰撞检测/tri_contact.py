import numpy as np

def cross(vec0,vec1):
    res = np.zeros((3))
    res[0] = vec0[1]*vec1[2] - vec0[2]*vec1[1]
    res[1] = vec0[2]*vec1[0] - vec0[0]*vec1[2]
    res[2] = vec0[0]*vec1[1] - vec0[1]*vec1[0]
    return res

def dot(vec0,vec1):
    res = 0
    for i in range(len(vec0)):
        res += vec0[i]*vec1[i]
    return res

def project(ax):
    global p1,p2,p3,p4,p5,p6
    q1 = dot(ax, p1)
    q2 = dot(ax, p2)
    q3 = dot(ax, p3)
    q4 = dot(ax, p4)
    q5 = dot(ax, p5)
    q6 = dot(ax, p6)
    
    mx1 = max(max(q1,q2),q3)
    mn1 = min(min(q1,q2),q3)
    mx2 = max(max(q4,q5),q6)
    mn2 = min(min(q4,q5),q6)
    
    if mn1 > mx2 or mn2 > mx1:
        return True
    return False

# 一个三角形
# 顶点位置
pos1 = np.array([0,0,1],dtype = float)
pos2 = np.array([2,0,1],dtype = float)
pos3 = np.array([1,1,1],dtype = float)

p1 = pos1 - pos1
p2 = pos2 - pos1
p3 = pos3 - pos1
# 边
e1 = p2 - p1
e2 = p3 - p2
e3 = p1 - p3

# 三角形法向量
n1 = cross(e1, e2)
# 每条边的外法向量
g1 = cross(e1,n1)
g2 = cross(e2,n1)
g3 = cross(e3,n1)

# 另一个三角形
pos4 = np.array([0,0,1],dtype = float)
pos5 = np.array([2,0,1],dtype = float)
pos6 = np.array([1,1,1],dtype = float)

p4 = pos4 - pos1
p5 = pos5 - pos1
p6 = pos6 - pos1

e4 = p5 - p4
e5 = p6 - p5
e6 = p4 - p6

n2 = cross(e4, e5)

g4 = cross(e4,n2)
g5 = cross(e5,n2)
g6 = cross(e6,n2)

e14 = cross(e1,e4)
e15 = cross(e1,e5)
e16 = cross(e1,e6)
e24 = cross(e2,e4)
e25 = cross(e2,e5)
e26 = cross(e2,e6)
e34 = cross(e3,e4)
e35 = cross(e3,e5)
e36 = cross(e3,e6)

collision = False
if project(n1) == True:
    collision = True
if project(n2) == True:
    collision = True
    
if project(e14) == True:
    collision = True
if project(e15) == True:
    collision = True
if project(e16) == True:
    collision = True
if project(e24) == True:
    collision = True
if project(e25) == True:
    collision = True
if project(e26) == True:
    collision = True
if project(e34) == True:
    collision = True
if project(e35) == True:
    collision = True
if project(e36) == True:
    collision = True
    
if project(g1) == True:
    collision = True
if project(g2) == True:
    collision = True
if project(g2) == True:
    collision = True
if project(g3) == True:
    collision = True
if project(g4) == True:
    collision = True
if project(g5) == True:
    collision = True
if project(g6) == True:
    collision = True