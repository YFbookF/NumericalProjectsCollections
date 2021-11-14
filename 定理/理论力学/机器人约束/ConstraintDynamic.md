===========Physically Based Modeling Constrained Dynamics  

解决的问题，现在有一个点在圆上运动，它的一开始的位置和速度都是零，我们需要找到一个constraint force，来让它的加速度也是零。

The idea of constrained particle dynamics is that our description of the system includes not only particles and forces, but restrictions on the way the particles are permitted to move. For example, we might constrain a particle to move along a specified curve, or require two particles to remain a specified distance apart. The problem of constrained dynamics is to make the particles obey Newton’s laws, and at the same time obey the geometric constraints.  

我们可以使用能能力函数。

![image-20211113150701325](E:\mycode\collection\定理\连续介质力学\image-20211113150701325.png)

legal 是必须遵守的，那么速度，加速度分别是
$$
\dot C = \bold x \cdot \bold x = 0 \qquad \ddot{ C} = \ddot{\bold x} \cdot \bold x + \dot{\bold x} \cdot \dot{\bold x}
$$
接下来f 是 the given applied force，并且f is the as yet unknown constraint force
$$
\ddot{\bold x} = \frac{\bold f + \hat{\bold f}}{m}
$$
最后得到了
$$
\ddot C = \frac{\bold f + \hat{\bold f}}{m} \cdot \bold x + \dot{\bold x} \cdot \dot{\bold x} = 0 \qquad \hat{\bold f} \cdot \bold x = - \bold f \cdot \bold x - m \dot{\bold x} \cdot \dot{\bold x}
$$
这时候我们仅仅只有一个方程和两个未知数，也就是unknown constraint force。动能如下
$$
T = \frac{m}{2}\dot{\bold x} \cdot \dot{\bold x}
$$
动能的导数就是里做的功
$$
\dot T = m\ddot{\bold x} \cdot \dot{\bold x} = m\bold f \cdot \dot{\bold x} + m\hat{\bold f} \cdot \dot{\bold x}
$$
which is the work done by f and hat f。由于是虚功，所以第二项肯定是零，也就是hat f 肯定和x同方向
$$
\hat {\bold f} = \lambda \bold x
$$
然后算 lambda
$$
\lambda = \frac{-\bold f \cdot \bold x - m \dot{\bold x} \cdot \dot{\bold x}}{\bold x \cdot \bold x}
$$
when the system is at rest , the constraint force given in equation 7 reduces to

现在求导C，
$$
\dot{\bold C} = \frac{\partial \bold C}{\partial \bold q} \dot{\bold q} \qquad \ddot{\bold C} = \dot{\bold J}\dot{\bold q} + \bold J \ddot{\bold q} \qquad \dot{\bold J} = \frac{\partial \bold J}{\partial \bold q }\dot{\bold q} \qquad \bold J = \frac{\partial \bold C}{\partial \bold q}
$$
C 是一个标量，C求导是个三维向量，也就是顶点上的力。求二次导是个矩阵，？？？

J 应该是个矩阵，求导就是三维向量，或者rank 3 tensor

然后继续算，Q是力
$$
\ddot{\bold C} = \dot{\bold J}\dot{\bold q} + \bold J \bold M^{-1}(\bold Q + \hat{\bold Q})
$$
上面的式子是零
$$
\bold J \bold M^{-1}\hat{\bold Q} = -\dot{\bold J}\dot{\bold q} - \bold J \bold M^{-1}\bold Q
$$
both C and q are vector。To ensure that the constraint foece does not work, we therefore require that
$$
\hat{\bold Q} \cdot \dot{\bold x} = 0 \qquad \hat{\bold Q} = \bold J^T \lambda
$$
===================Intro to physics based animation

![image-20211113172950865](E:\mycode\collection\定理\连续介质力学\image-20211113172950865.png)

![image-20211113170332393](E:\mycode\collection\定理\连续介质力学\image-20211113170332393.png)

===================Physically Based Modeling Constrained Dynamics  

注意上面这个式子，可以这么写
$$
\bold J \bold M^{-1}\bold J ^T \lambda = - \dot{\bold J}\dot{\bold q} - \bold J \bold M^{-1}\bold Q - k_s \bold C - k_d \dot{\bold C}
$$
ks is spring constants, kd is damping constraint 或者下面这么写
$$
\bold J \bold M^{-1} \bold J^T \lambda = - \dot{\bold J}\dot{\bold q} - \bold J \bold M^{-1}\bold Q
$$
===================RedMax

![image-20211114215934920](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114215934920.png)

![image-20211114215953318](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114215953318.png)

![image-20211114220114825](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114220114825.png)
