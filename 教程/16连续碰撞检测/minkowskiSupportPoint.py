import numpy as np

# 计算三角形的Minkowski support point

def ScaleDot(vec0,vec1):
        res = 0
        for i in range(len(vec0)):
            res += vec0[i] * vec1[i]
        return res

def Triangle():
        
    # 方向
    direction = np.array([1,0,0])
    
    # 三角形三个点
    a = np.array([0,0,0])
    b = np.array([2,0,1])
    c = np.array([1,1,2])
    

    
    dota = ScaleDot(direction, a)
    dotb = ScaleDot(direction, b)
    dotc = ScaleDot(direction, c)
    
    minkowski = np.zeros((3))
    if dota > dotb:
        if dotc > dota:
            minkowski = c
        else:
            minkowski = a
    else:
        if dotc > dotb:
            minkowski = c
        else:
            minkowski = b
            
def Ellipsoid():
        
    # 方向
    direction = np.array([1,0,0])
    
    rad = np.array([1,2,3],dtype = float) # 椭球三个方向的半径
    
    a2 = rad[0] * rad[0]
    b2 = rad[1] * rad[1]
    c2 = rad[2] * rad[2]
    v = np.array([a2*direction[0],b2*direction[1],c2*direction[2]])
    d = np.sqrt(ScaleDot(v,direction))
    
    minkowski = v / d
    
def Convex():
        
    # 方向
    direction = np.array([1,0,0])
    
    vertex = np.array([[0,0,0],
                       [2,0,1],
                       [1,1,2]])
    
    minkowski = np.zeros((3))
    
    maxdot = 0
    for i in range(vertex.shape[0]):
        dotres = ScaleDot(vertex[i,:], direction)
        if dotres > maxdot:
            maxdot = dotres
            minkowski = vertex[i,:]
            