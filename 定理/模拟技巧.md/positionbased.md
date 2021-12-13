=======strain based dynamics

![image-20211030135004871](D:\定理\模拟技巧.md\image-20211030135004871.png)

那么

![image-20211030135107222](D:\定理\模拟技巧.md\image-20211030135107222.png)

Hierarchical Position Based Dynamics  

一阶的constraint如下
$$
C(\bold p + \Delta \bold p) = C(\bold p) + \nabla_{\bold P}C(\bold p) \cdot \Delta \bold p + O(|\Delta \bold p|^2) = 0
$$
Delta p is global correction vector

我们可以使用拉格朗日乘子法
$$
\Delta \bold p = \lambda \nabla_\bold p C(\bold p) = -sw_i \nabla \bold p_i C(\bold p)
$$
其中
$$
w_i = 1/m_i \qquad s = \frac{C(\bold p)}{\sum_j w_j|\nabla_{\bold p_j} C(\bold p)|}
$$
A classical PBD solver [Muller et al ¨ . 2007] performs three steps.
In the first step, the positions are initialized by an explicit Euler
step, ignoring internal forces. In the second step, the positions are
updated by projecting the current configuration consecutively on
each constraint set respecting the mass weighting. In the last step,
the velocities are updated as vn+1 = (qn+1 - qn)=h.  

==================A Survey on Position Based Dynamics, 2017  

https://matthias-research.github.io/pages/tenMinutePhysics/index.html

http://blog.mmacklin.com/
