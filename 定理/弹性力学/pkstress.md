Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org)

比如有个平面方形，它的一边长(2,0,0)，另一边长(0,1,0)，那么它的面积S = 2，也就是reference configurations
$$
d\bold S = \bold N dS  = d\bold r_1 \times  d\bold r_2 = (2,0,0)\times(0,1,0) = (0,0,1) d2 = (0,0,2)
$$
对于current configuration
$$
d\bold s = \bold n ds \qquad dV = d\bold x \cdot d \bold S = \frac{1}{J} dv = \frac{1}{J}d\bold r \cdot d\bold s = \frac{1}{J} d\bold x \cdot \bold n ds
$$
上面就是面积变化，图找不到了
$$
d\bold r = \bold J d \bold x
$$
也就是
$$
J d\bold x \cdot \bold N dS = d \bold r \cdot \bold n ds = (\bold Jd\bold x)\cdot \bold n ds
$$
那么最终得到
$$
\bold N = \frac{1}{J}\bold J^T \bold n \frac{ds}{dS} \tag{2.88}
$$
dx is arbitrary, N is a unit vector , this equation is called Nanson`s formula。经过以下变换
$$
\bold n ds = J(\bold J^T)^{-1}\bold N dS \qquad \bold f = \vec \sigma_n ds = \vec \sigma \bold n ds = (J\sigma\bold J^{-T})\bold NdS 
$$
那么就得到了first piola-Kirchhoff力
$$
\vec \sigma_{P1} = J \sigma (\bold J^T)^{-1}
$$

```
defgrad = np.array([[2,0,0],[0,2,0],[0,0,1]])
sigma = np.array([[0,0,0],[0,0,0],[0,0,2]])
pk1 = np.linalg.det(defgrad) * np.dot(sigma,np.linalg.inv(defgrad.T))
pk1 = np.array([[0,0,0],[0,0,0],[0,0,8]])
```

意思是在现在每施加2的影响力，那么原来就施加8的影响力。

=================Stable Neo-Hookean Flesh Simulation  

对于线弹性来说，pk1 stress 如下
$$
\bold P(\bold F) = 2\mu \varepsilon + \lambda \trace(\varepsilon)\bold I
$$
对于new stable neo hookean来说
$$
\bold P(\bold F) = \mu(1 - \frac{1}{I_C + 1})\bold F + \lambda(J - \alpha)\frac{\partial J}{\partial \bold F}
$$
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

=================Robust Quasistatic Finite Elements and Flesh Simulation  

psi strain energy
$$
\vec f = -\frac{\partial \psi}{\partial \vec x}
$$
first pk stress which is the gradient of the strain energy with respect to the formation gradient
$$
\bold P = \frac{\partial \psi}{\partial \bold F}
$$
The force on a node i due to tetrahedron incident to it is
$$
\bold g_i = -\bold P (A_1 \bold N_1 + A_2 \bold N_2 + A_3 \bold N_3)/3 \qquad \bold g_0 = -(\bold g_1 + \bold g_2 + \bold g_3)
$$
AN is the area weighted normals of the faces.

也可下面这么计算
$$
\bold G = \bold P \bold B \qquad \bold G = (\bold g_1,\bold g_2, \bold g_3) \qquad \bold B_m = (\bold b_1,\bold b_2,\bold b_3) = -V\bold D_m^{-T}
$$
Dm就是reference configuration，用于F= DsDm^{-1}的那个

The first Piola-Kirchhoff stress is invariant under rotations of either material or world space for isotropic materials.

Furthermore, the defermation gradient can be transformed into a diagonal matrix hat F, with an application of a material and a world space rotation
$$
\bold F = \bold U \hat{\bold F}\bold V^T
$$
Yied diagonal for isotropic
$$
\bold P(\bold F) = \bold U\bold P (\bold U^T \bold F \bold V)\bold V^T = \bold U\bold P(\hat{\bold F})\
$$
=====================Position-Based Simulation of Continuous Materials  

![image-20211029135455817](D:\定理\弹性力学\image-20211029135455817.png)

=====================连续介质力学基础

![image-20211025202113775](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025202113775.png)

Applied Mechanics of Solids by Allan F. Bower (z-lib.org)

section 2.2.4

Kirchhoff stress
$$
\bold \tau = J \sigma \qquad \tau_{ij}  = J \sigma_{ij}
$$
Nominal stress first Piola Kirchhoff
$$
\bold S = J \bold F^{-1}\cdot \bold \sigma \qquad S_{ij} = J F_{ik}^{-1}\sigma_{kj}
$$
Material second PioKirchhoff
$$
\sum = J \bold F^{-1} \cdot \sigma \cdot \bold F^{-T} \qquad \sum_{ij} = J F^{-1}_{ik}\sigma_{kl}F^{-1}_{jl}
$$
René De Borst, Mike Crisfield, Joris Remmers and Clemens Verhoosel - Nonlinear Finite Element Analysis of Solids and Structures, 2nd edition

since the Cauchy stress tensor is referred to the current, unknown, configuration, the use of an auxiliary stress measure, which refers to a reference configuration, is needed.

tau is the second pk stress tensor, related to the cauchy tensor sigma through
$$
\vec \sigma = \frac{\rho}{\rho_0}\vec F \cdot \vec \tau \cdot \vec F^T \qquad \sigma_{ij} = \frac{\rho}{\rho_0}F_{ik}\tau_{kl}F_{jl}
$$
The second Piola Kirchhoff stress tensor has no direct physical relevance. When the stresses must be determined in an analysis in addition to the displacements, Cauchy stressed have to be computed from the second pk stresses.

For small displacement gradient rho approx rho0, F approx I, this two is coincide.

![image-20211023113506727](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211023113506727.png)

请看平衡方程.md
$$
\delta W_s = - \int_v \vec \sigma:(\delta \bold J)\bold J^{-1}dv = - \int_VJ \vec \sigma:(\delta \bold J)\bold J^{-1}dV
$$
![image-20211025170526696](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025170526696.png)

section 3.6

For the most part, particularly in the analysis of large deformation used in this book, the
second Piola–Kirchhoff stress tensor will be used because this is the tensor associated with the
Green–Lagrange strain tensor. In some plasticity and viscoelasticity formulations that require
the use of true stress measures, the Cauchy stress or the Kirchhoff stress tensors are used.
Furthermore, being able to identify the effect of the hydrostatic pressure is not only important
in many large-deformation formulations, but it is also important in improving the numerical
performance of the finite elements in some applications. Although in some nonlinear formulations such as the J2 plasticity formulations, the constitutive equations are formulated in terms
of the deviatoric stresses because the hydrostatic pressure is of less significance; in some other
linear and nonlinear applications, the performance of the finite elements can significantly
deteriorate owing to locking problems associated with the volumetric changes, as in the case
of incompressible or nearly incompressible materials. By understanding the contributions
of the hydrostatic pressure to the elastic forces, one can propose solutions to these locking
problems. Some of these locking problems are discussed in more detail in Chapter 5.  

Recall that the second Piola–Kirchhoff stress tensor does not change under an arbitrary
rigid-body rotation. I  

连续介质力学基础

![image-20211025202802563](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025202802563.png)

![image-20211025203150185](C:\Users\acer\AppData\Roaming\Typora\typora-user-images\image-20211025203150185.png)

========A FINITE STRAIN BEAM FORMULATION. THE THREE-DIMENSIONAL  

![image-20211026142917603](D:\定理\弹性力学\image-20211026142917603.png)

