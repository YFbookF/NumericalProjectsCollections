Direct Position-Based Solver for Stiff Rods  
$$
\bold M \ddot {\bold x} = -\nabla U(x) + \bold f_{ext}
$$
差分
$$
\bold M(\frac{\bold x(t+\Delta t) - 2\bold x(t) + \bold x(t - \Delta t)}{\Delta t^2}) = -\nabla U(\bold x(t + \Delta t)) + \bold f_{ext}
$$
elastic potential Ux has the form
$$
U(\bold x) = \frac{1}{2}\bold C(\bold x)^T\vec \alpha^{-1}\bold C(\bold x)
$$
C x is a vector of constraint function, alpha is the diagonal compliance matrix describing the inverse stiffness. 比如对于Cosserat 的bending 和Torsion来说，C就是dOmega，alpha就是K-1

看看
$$
E_b = \frac{1}{2}\int \Delta \Omega^T K\Delta \Omega ds \qquad K = \begin{bmatrix} EI_1 & 0 & 0 \\ 0 & EI_2 & 0 \\ 0 & 0 & GJ\end{bmatrix}
$$
算力的时候
$$
\bold f_{el} = -\nabla U(\bold x) = -\bold J^T \vec \alpha ^{-1} \bold C = \Delta t^2 \bold J^T \vec \lambda \qquad \bold J = \nabla \bold C
$$
lambda 就是 Lagrane multiplier

![image-20211030131740751](D:\定理\积分技巧\image-20211030131740751.png)