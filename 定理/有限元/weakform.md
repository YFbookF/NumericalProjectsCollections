Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org)

![image-20211025152642853](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025152642853.png)

=======Application of the Element Free Galerkin Method to Elastic Rods by Daniel Dreyer  

differential form
$$
\frac{d}{dx}(EA\frac{du}{dx}) + f^b = 0
$$
以下是几个看不懂的公式，C是四阶向量，S是第一PK，E是infinitesimal strain tensor
$$
\bold S = C \bold E  \qquad \bold E = \frac{1}{2}(\nabla \bold u + \nabla \bold u^T)
$$
In the linearized theory it is assumed that S approx T, that mean cauchy stress tensor is a sufficiently accurate representation of the first piolaKirchhoff stress tensor.The Cauchy stress tensor and the infinitesimal strain tensor E are sym.

第一pk也可用下面的式子表示
$$
\bold S = 2\mu \bold E + \lambda(tr \bold E) \bold I \qquad \lambda = \frac{E\nu}{(1+\nu)(1-\nu)} \qquad \mu = G = \frac{E}{2(1+\nu)}
$$
the differential equation of equilibrium enforces equilibrium of the external forces / tractions and internal forces / stress. and 
$$
Div \bold S + \bold b_0 = 0 \qquad \mu \Delta \bold u + (\lambda + \nu)\nabla Div \bold u + \bold b_0 = \bold 0
$$
反向
$$
\bold E = \frac{1}{2\mu}(\bold S - \frac{\lambda}{2\mu + 3\lambda}(tr\bold S)\bold I)
$$
![image-20211026192918383](E:\mycode\collection\定理\有限元\image-20211026192918383.png)

![image-20211026192954297](E:\mycode\collection\定理\有限元\image-20211026192954297.png)