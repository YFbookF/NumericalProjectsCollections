import numpy as np

# https://github.com/brainexcerpts/Dual-Quaternion-Skinning-Sample-Codes/blob/master/quat_cu.hpp
# 将四元数转换成旋转矩阵形式 Q54
def Quat(q):
    x2 = q[0] * q[0] * 2
    y2 = q[1] * q[1] * 2
    z2 = q[2] * q[2] * 2
    xy = q[0] * q[1] * 2
    yz = q[1] * q[2] * 2
    zx = q[2] * q[0] * 2
    xw = q[0] * q[3] * 2
    yw = q[1] * q[3] * 2
    zw = q[2] * q[3] * 2
    m = np.zeros((4,4))
    m[0,0] = 1 - y2 - z2
    m[0,1] = xy - zw
    m[0,2] = zx + yw
    m[1,0] = xy + zw
    m[1,1] = 1 - z2 - x2
    m[1,2] = yz - xw
    m[2,0] = zx - yw
    m[2,1] = yz + xw
    m[2,2] = 1 - x2 - y2
    m[3,3] = 1
    return m

# 将旋转矩阵转换为四元数形式
def RotationMatrixToQuaterion():
    mat = np.zeros((4,4))
    T = 1 + mat[0,0] + mat[1,1] + mat[2,2] 
    # T = 4 - 4x^2 + 4y^2 + 4z^2
    if (T < 1e-10):
        # error
        return 
    S = np.sqrt(T) * 2
    X = (mat[2,1] - mat[1,2]) / 2
    Y = (mat[0,2] - mat[2,0]) / 2
    Z = (mat[1,0] - mat[0,1]) / 2
    W = S / 4
    Quaterion = np.array([X,Y,Z,W])