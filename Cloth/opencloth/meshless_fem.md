Several rotation-independent deformation tensors are used in mechanics. Since a pure rotation should not induce any strains in a deformable body, it is often convenient to use rotation-independent measures of deformation.

比如right cauchy Green deformation tensor
$$
\bold C = \bold F^T\bold F
$$
left cauchy green deformation tensor
$$
\bold B = \bold F\bold F^T
$$
cauhcy deformation tensor or piola tensor
$$
\bold c = \bold B^{-1} = \bold F^{-T}\bold F^{-1}
$$
The concept of *strain* is used to evaluate how much a given displacement differs locally from a rigid body displacement.

Lagrangian finite strain tensor,or Green-Lagrange strain tensor,or Green St-Venant strain tensor
$$
\bold E = \frac{1}{2}(\bold F^T\bold F - \bold I)
$$
linear strain tensor,logarithmic strain tensor,green lagrange strain tensor

euler almanasi strain tensor
$$
\bold E = \lambda - 1 \qquad \bold E = \ln (\lambda) \\
\bold E = \frac{1}{2}(\lambda^2 - 1) \qquad \bold E = \frac{1}{2}(1 - \frac{1}{\lambda^2})
$$
当物体没有拉伸的时候，lambda等于一。我们希望找到一个关系，也就是f(lambda) = 0当lambda = 1。



defomation gradient只提供了有关形变和刚体旋转的信息。如果使用deformation tensor可以提供一些有关translation的信息。Whereas the left and right Cauchy-Green tensors give information about the change in angle between line elements and the stretch of line elements,  

拉伸信息可通过如下看出来
$$
\lambda = \frac{|d\bold x|}{|d \bold X|}
$$
也就是对deformation tensor对角线上的元素操作。请看

http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_III/Chapter_2_Kinematics/Kinematics_of_CM_02_Deformation_Strain.pdf

### 本篇

$$
\bold J = \bold I + \nabla \bold u^T = \begin{bmatrix} u_x +1 & u_y & u_z \\ v_x & v_y + 1 & v_z \\ w_x & w_y & w_z + 1 \end{bmatrix}
$$

GreenSaintVenant strain tensor
$$
\varepsilon = \bold J^T \bold J - \bold I = \nabla \bold u + \nabla \bold u^T + \nabla \bold u \nabla \bold u^T
$$
J是不是就是那个deformation gradient啊。注意符号https://www.comsol.com/multiphysics/analysis-of-deformation。大X是未变形的位置，小x是变形后的位置，u是位移，也就是变形前后位置的差。
$$
\bold F = \frac{\partial \bold x}{\partial \bold X} = \bold I + \frac{\partial \bold u}{\partial \bold X} \\
= \begin{bmatrix} \partial x/\partial X & \partial x/\partial Y & \partial x/\partial Z\\\partial y/\partial X & \partial y/\partial Y & \partial y/\partial Z\\\partial z/\partial X & \partial z/\partial Y & \partial z/\partial Z\end{bmatrix} = \begin{bmatrix}1+ \partial u/\partial X & \partial u/\partial Y & \partial u/\partial Z\\\partial v/\partial X & 1 + \partial v/\partial Y & \partial v/\partial Z\\\partial w/\partial X & \partial w/\partial Y & 1 +\partial w/\partial Z\end{bmatrix}
$$

```
def computeJacobians():
    for i in range(node_num):
        for j in range(neighbor_num):
            idx = neighbor_idx[i,j]
            node_displacement[idx] = node_pos[idx] - node_pos[idx]
        Bmat = np.zeros((3,3))
        for j in range(neighbor_num):
            idx = neighbor_idx[i,j]
            # 记住，是 别人 减 自己
            dj = node_displacement[idx] - node_displacement[i]
            rdist = neighbor_rdis[i,j] 
            bj = outterProduct(dj,rdist) * neighbor_weight[i,j]
            Bmat += bj
        Bmat = Bmat.T
        du = np.dot(node_minv[i,:,:],Bmat)# deformation gradient 3 x 3
        node_J[i,:,:] = np.identity(3) + du.T
```

不过不理解的是，为什么把别的点也扯进来了？不过如果是标准的单元法的话，只要自己就可以了。

不同模型

这是胡克模型，应力和应变的关系是线性的
$$
\sigma = \bold C \varepsilon
$$

《计算结构力学》section 2.3

K是对称矩阵，可由弹性力学中的功的互等定理得到证明。主对角线元素是恒定值，因为其中每一个元素均为刚度系数。其物理意义是，当节点发送单位位移分量时，所需施加的节点力的分量。而主对角线元素是正值，说明节点位移的方向与施加节点力的方向是一致的。

K是奇异矩阵，此即单元刚度矩阵每一行元素之和为零，单元刚度矩阵的元素所组成的行列式为零，其物理意义是，在无约束的条件下单元可由做刚体运动，其位移是不定的