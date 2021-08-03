import numpy as np

edge0_vertex0_start = np.array([0.1,0.1,0.1])
edge0_vertex1_start = np.array([0,0,1])
edge0_vertex0_end = np.array([1,1,1])
edge0_vertex1_end = np.array([0,1,1])

edge1_vertex0_start = np.array([0.1,0.1,0.1])
edge1_vertex1_start = np.array([0,0,0])
edge1_vertex0_end = np.array([0,1,0])
edge1_vertex1_end = np.array([1,0,0])

err = np.array([-1,-1,-1])
ms = 1e-8
tolerance = 1e-6
t_max = 1
max_itr = 1e6

def vec_mindist(vec0,vec1):
    res = 0
    for i in range(3):
        if res < abs(vec0[i] - vec1[i]):
            res = abs(vec0[i] - vec1[i])
    return res
            
def max_linf_4(p1,p2,p3,p4,p1e,p2e,p3e,p4e):
    t1 = vec_mindist(p1,p1e)
    t2 = vec_mindist(p2,p2e)
    t3 = vec_mindist(p3,p3e)
    t4 = vec_mindist(p4,p4e)
    r = max(max(t1,t2),max(t3,t4))
    return r
    

p000 = edge0_vertex0_start - edge1_vertex0_start
p001 = edge0_vertex0_start - edge1_vertex1_start
p011 = edge0_vertex1_start - edge1_vertex1_start
p010 = edge0_vertex1_start - edge1_vertex0_start
p100 = edge0_vertex0_end - edge1_vertex0_end
p101 = edge0_vertex0_end - edge1_vertex1_end
p111 = edge0_vertex1_end - edge1_vertex1_end
p110 = edge0_vertex1_end - edge1_vertex0_end

dl = 3 * max_linf_4(p000, p001, p011, p010, p100, p101, p111, p110)
edge0_length = 3 * max_linf_4(p000, p100, p101, p001, p010, p110, p111, p011)
edge1_length = 3 * max_linf_4(p000, p100, p110, p010, p001, p101, p111, p011)

a = np.array([dl / tolerance, edge0_length / tolerance,edge1_length / tolerance])

ccd_type = 0 
# 0 is normal ccd, 
# 1 is ccd with input time interval upper bound, using real tolerance, max_itr and horizontal tree.


# interval_root_finder_double_horizontal_tree      1 ccd
t_up = np.array([0,0,0,0,1,1,1,1],dtype = float)
u_up = np.array([0,0,1,1,0,0,1,1],dtype = float)
v_up = np.array([0,1,0,1,0,1,0,1],dtype = float)
t_dw = np.ones((8),dtype = float)
u_dw = np.ones((8),dtype = float)
v_dw = np.ones((8),dtype = float)

eps = 7.1054273576010019e-15
ms = 1.0000000000000000e-08
bbox_in = np.zeros((3),dtype = bool)
for d in range(3):
    
    rst = np.zeros((8))
    for i in range(8):
        edge0_vertex0 = (edge0_vertex0_end[d] - edge0_vertex0_start[d]) * t_up[i] / t_dw[i] + edge0_vertex0_start[d]
        edge0_vertex1 = (edge0_vertex1_end[d] - edge0_vertex1_start[d]) * t_up[i] / t_dw[i] + edge0_vertex1_start[d]
        edge1_vertex0 = (edge1_vertex0_end[d] - edge1_vertex0_start[d]) * t_up[i] / t_dw[i] + edge1_vertex0_start[d]
        edge1_vertex1 = (edge1_vertex1_end[d] - edge1_vertex1_start[d]) * t_up[i] / t_dw[i] + edge1_vertex1_start[d]
        
        edge0_vertex = (edge0_vertex1 - edge0_vertex0) * u_up[i] / u_dw[i] + edge0_vertex0
        edge1_vertex = (edge1_vertex1 - edge1_vertex0) * v_up[i] / v_dw[i] + edge1_vertex0
        
        rst[i] = edge0_vertex - edge1_vertex
        
    minv = rst[0]
    maxv = rst[0]
    for i in range(8):
        if minv > rst[i]:
            minv = rst[i]
        if maxv < rst[i]:
            maxv = rst[i]
            
    tol = maxv - minv # real tolerance
    result = True
    bbox_in_eps = False
    if minv - ms > eps or maxv + ms < -eps:
        result = False
    if minv + ms >= eps and maxv - ms <= eps:
        bbox_in[d] = True
    else:
        bbox_in[d] = False
        
box_in_eps = False
if bbox_in[0] == True and bbox_in[1] == True and bbox_in[2] == True:
    box_in_eps = True

    

