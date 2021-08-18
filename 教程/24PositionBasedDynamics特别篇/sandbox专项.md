请配合daynamicsDeformation这篇pdf观看，相关代码在下面可找



https://web.cse.ohio-state.edu/~wang.3602/Wang-2016-DME/Wang-2016-DME.zip

还有SVD算法偶

**Computing the Singular Value Decomposition of** 3 *×* 3 **matrices with minimal**

**branching and elementary flfloating point operations**

3X3 和 2X2

defomation Gradient 计算如下
$$
\bold F = \frac{\partial x}{\partial \bold X}
$$
小x是当前左边。大X是初始坐标。形变梯度可polar decomposition为roation part 和 stretching part
$$
\bold F = \bold R \cdot \bold U = \bold V \cdot \bold R
$$
可以计算right Cauchy-Green defomation tensor
$$
\bold C = \bold F^T \bold F
$$
开平方后就是U，即rihgt stretch tensor。以及Left Cauchy Green defomration tensor
$$
\bold B = \bold F \bold F^T
$$
开平方后就是C

sandbox 已经挖掘完毕，除了还有一些破裂理论

FEM Simulation of 3D Deformable Solids: A practitioner’s
guide to theory, discretization and model reduction  