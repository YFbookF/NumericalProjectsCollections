import numpy as np

width = 3
height = 3
depth = 3



alpha = 0.75
facturing = True
fractureDistanceTolerance = 999
fractureRotationTolerance = 0.6
kRegionDamping = 0.25

particle_num = width * height * depth
particle_pos = np.zeros((particle_num,3),dtype = float)
particle_pos_rest = np.zeros((particle_num,3),dtype = float)
particle_parentRegion_size = np.zeros((particle_num),dtype = int)
particle_parentRegion_mass = np.zeros((particle_num))
particle_v = np.zeros((particle_num,3))
particle_M = np.zeros((particle_num,3,3))
particle_mass = 1

bar_num = width * height * depth + (width - 1) * height * depth
plate_num = 2 * width * height * depth # 每层两套网格，自己层用一套，别的层一套
region_num = particle_num

bar_v = np.zeros((bar_num))
bar_M = np.zeros((bar_num,3,3))
bar_idx = np.zeros((bar_num,3))# w = 1 的时候，就是一个线段两个点
plate_v = np.zeros((plate_num))
plate_M = np.zeros((plate_num,3,3))
plate_idx = np.zeors((plate_num,7))
region_v = np.zeros((region_num))
region_M = np.zeros((region_num,3,3))
region_idx = np.zeros((region_num,4))
region_mass = np.zeros((region_num,3))
region_c0 = np.zeros((region_num))

cnt = 0
for i in range(width):
    for j in range(height):
        for k in range(depth):
            particle_pos[cnt,:] = np.array([i,j,k])
            edge = ((i == 0 or i == width - 1) 
                    + (j == 0 or j == height - 1) 
                    + (k == 0 or k == depth - 1))
            if edge == 0:
                particle_parentRegion_size[cnt] = 26
            elif edge == 1:
                particle_parentRegion_size[cnt] = 17
            elif edge == 2:
                particle_parentRegion_size[cnt] = 11
            elif edge == 3:
                particle_parentRegion_size[cnt] = 7
            particle_parentRegion_mass[cnt] = particle_mass / particle_parentRegion_size[cnt]
            particle_v[cnt,:] = particle_parentRegion_mass[cnt] * particle_pos[cnt,:]
            particle_pos_rest[cnt,:] = particle_pos[cnt,:]
            particle_pos[cnt,1] += 5
            cnt += 1

for i in range(bar_num):
    if i < particle_num:
        bar_idx[i,0] = i
        bar_idx[i,2] = 1
    else:
        idx = (i - particle_num)
        bar_idx[i,2] = 2
        bar_idx[i,0] = idx
        bar_idx[i,1] = idx + height * depth
    
# 点·边·面·体
# 索引真的要这么复杂吗？编完所有人都升华了？放心，这个复杂程度还不及adpative refinement mesh的千分之一    
cnt = 0
for k in range(depth):
    for j in range(height):
        for i in range(width):
            if i == 0:
                if j == 0:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt + depth * height
                    plate_idx[cnt,1] = cnt + depth + particle_num
                elif j == height - 1:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth * height
                    plate_idx[cnt,1] = cnt - depth * height + depth + particle_num
                else:
                    plate_idx[cnt,6] = 3
                    plate_idx[cnt,0] = cnt - depth + particle_num
                    plate_idx[cnt,1] = cnt
                    plate_idx[cnt,2] = cnt + depth + particle_num
            elif i == width - 1:
                if j == 0:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth * height
                    plate_idx[cnt,1] = cnt - depth * height + depth + particle_num
                if j == height - 1:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth * height - depth + particle_num
                    plate_idx[cnt,1] = cnt - depth * height
                else:
                    plate_idx[cnt,6] = 3
                    plate_idx[cnt,0] = cnt - depth * height - depth + particle_num
                    plate_idx[cnt,1] = cnt - depth * height
                    plate_idx[cnt,2] = cnt - depth * height + depth + particle_num
            else:
                if j == 0:
                    plate_idx[cnt,6] = 4
                    plate_idx[cnt,0] = cnt - depth * height 
                    plate_idx[cnt,1] = cnt - depth * height  + depth
                    plate_idx[cnt,2] = cnt + depth + particle_num
                    plate_idx[cnt,3] = cnt + depth * height
                elif j == height - 1:
                    plate_idx[cnt,6] = 4
                    plate_idx[cnt,0] = cnt - depth * height 
                    plate_idx[cnt,1] = cnt - depth * height  - depth
                    plate_idx[cnt,2] = cnt - depth + particle_num
                    plate_idx[cnt,3] = cnt + depth * height
                else:
                    plate_idx[cnt,6] = 6
                    plate_idx[cnt,0] = cnt - depth * height - depth
                    plate_idx[cnt,1] = cnt - depth * height
                    plate_idx[cnt,2] = cnt - depth * height + depth
                    plate_idx[cnt,3] = cnt - depth + particle_num
                    plate_idx[cnt,4] = cnt + depth + particle_num
                    plate_idx[cnt,5] = cnt + depth * height
            cnt += 1
                    
for k in range(depth):
    for j in range(height):
        for i in range(width):
            if i == 0:
                if j == 0:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt
                    plate_idx[cnt,1] = cnt + depth
                elif j == height - 1:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth
                    plate_idx[cnt,1] = cnt
                else:
                    plate_idx[cnt,6] = 3
                    plate_idx[cnt,0] = cnt - depth
                    plate_idx[cnt,1] = cnt
                    plate_idx[cnt,2] = cnt + depth
            elif i == width - 1:
                if j == 0:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth * height
                    plate_idx[cnt,1] = cnt - depth * height + depth
                if j == height - 1:
                    plate_idx[cnt,6] = 2
                    plate_idx[cnt,0] = cnt - depth * height - depth
                    plate_idx[cnt,1] = cnt - depth * height
                else:
                    plate_idx[cnt,6] = 3
                    plate_idx[cnt,0] = cnt - depth * height - depth
                    plate_idx[cnt,1] = cnt - depth * height
                    plate_idx[cnt,2] = cnt - depth * height + depth
            else:
                if j == 0:
                    plate_idx[cnt,6] = 4
                    plate_idx[cnt,0] = cnt - particle_num
                    plate_idx[cnt,1] = cnt + depth - particle_num
                    plate_idx[cnt,2] = cnt
                    plate_idx[cnt,3] = cnt + depth
                elif j == height - 1:
                    plate_idx[cnt,6] = 4
                    plate_idx[cnt,0] = cnt - depth - particle_num
                    plate_idx[cnt,1] = cnt - particle_num
                    plate_idx[cnt,2] = cnt - depth
                    plate_idx[cnt,3] = cnt
                else:
                    plate_idx[cnt,6] = 6
                    plate_idx[cnt,0] = cnt - depth - particle_num
                    plate_idx[cnt,1] = cnt - particle_num
                    plate_idx[cnt,2] = cnt + depth - particle_num    
                    plate_idx[cnt,3] = cnt - depth
                    plate_idx[cnt,4] = cnt
                    plate_idx[cnt,5] = cnt + depth
            cnt += 1
                    
# 编完恶心的plate_num后，region就好很多了
cnt = 0
for k in range(depth):
    for j in range(height):
        for i in range(width):
            if j == 0 or j == height - 1:
                region_idx[cnt,3] = 2
                region_idx[cnt,0] = cnt
                region_idx[cnt,1] = cnt + region_num + height * width
            else:
                region_idx[cnt,3] = 3
                region_idx[cnt,0] = cnt
                region_idx[cnt,1] = cnt + region_num + height * width
                region_idx[cnt,1] = cnt + region_num - height * width
            cnt += 1
        
ParticleToRegion()
for ridx in range(region_num):
    region_mass[ridx]  = region_M[ridx,0,0]
    region_c0[ridx] = region_v[ridx] / region_mass[ridx]
        
def vecMulvec(vec0,vec1):
    n = len(vec0)
    res = np.zeros((n,n))
    for i in range(n):
        res[i,:] = vec0[i] * vec1[:]
    return  res
            
def ParticleToRegion():
    for i in range(bar_num):
        size = bar_idx[i,2]
        for j in range(size):
            bar_v[i] += particle_v[bar_idx[j]]
            bar_M[i] += particle_M[bar_idx[j]]
    for i in range(plate_num):
        size = plate_idx[i,6]
        for j in range(size):
            plate_v[i] += bar_v[plate_idx[j]]
            plate_M[i] += bar_M[plate_idx[j]]
    for i in range(region_num):
        size = region_idx[i,6]
        for j in range(size):
            region_v[i] += plate_v[region_idx[j]]
            region_M[i] += plate_M[region_idx[j]]
            
# 难道没权重吗？这么草率？
def RegionToParticle():
    for i in range(region_num):
        size = region_idx[i,6]
        for j in range(size):
            plate_v[region_idx[j]] += region_v[i]
            plate_M[region_idx[j]] += region_M[i]

def shapeMatch():
    for i in range(particle_num):
        particle_v[i,:] = particle_parentRegion_mass[i] * particle_pos[i,:]
        particle_M[i,:,:] = particle_parentRegion_mass[i] * vecMulvec(particle_pos[i,:],particle_pos_rest[i,:])
    
    ParticleToRegion()
    
    for ridx in range(region_num):
        
        Fmixi = region_v[ridx]
        Fmixi0T = region_M[ridx]
        
        rc = 1 / region_mass[ridx] * Fmixi
        
        rA = Fmixi0T - region_mass[ridx] * vecMulvec(region_c0[ridx],rc) # 3 x 3 矩阵
        
        S = np.dot(rA.T,rA)
        eigenVector = np.linalg.eig(r) # 扯淡，r 根本没有特征向量
        S = np.dot(np.transpose(eigenVector),S,eigenVector)
        
        
            
time = 0
timeFinal = 1
dt = 0.1
while(time < timeFinal):
    
    shapeMatch()