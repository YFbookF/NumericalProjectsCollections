Long, slender elastic bodies, rods in short, can be found in many places in the natural world,  

=============René De Borst, Mike Crisfield, Joris Remmers and Clemens Verhoosel - Nonlinear Finite Element Analysis of Solids and Structures, 2nd edition

section 9.1.2 Including Shear Deformation: Timoshenko beam

As a consequence, theta, rotation of the normal to the centreline, becomes an independent variable, and the curvature is given by:
$$
\chi = \frac{d\theta}{dx} \qquad Q = \int_{-h/2}^{+h/2}b(z_l)\tau dz_l
$$
第二个是shear force, 那么下面分别是shear strain 以及 virtual work contribution
$$
\int_{l_0}Q\delta \gamma dx \qquad \gamma = \theta  + \frac{dw}{dx}
$$
那么总方程变为
$$
\int_{l_0}(N\delta \overline \varepsilon + M\delta \chi + Q\delta \gamma)dx = \delta \bold u^T \bold f_{ext}
$$
The interpolation now formally follows from
$$
u_l = \bold h_u^T \bold a \qquad w = \bold h_w^T \bold w \qquad \theta = \bold h_{\theta}^T \bold \theta
$$
and the spatial derivatives follow by straightforward differentiation
$$
\frac{d u_l}{dx} = \bold b_{u}^T \bold a \qquad \frac{dw}{dx} = \bold b_w^T \bold w \qquad \frac{d\theta}{dx} = \bold b_{\theta}^T \bold \theta
$$
and
$$
\bold a^T = (a_1,a_2,\Delta a_c) \qquad \bold w^T = (w_1,w_2,\Delta w_c) \qquad \theta^T = (\theta_1,\theta_2,\Delta \theta_c)
$$
section 9.3.2
$$
\overline M = EI \overline \chi = -\frac{EI}{l_0}(\overline \theta_2 - \overline \theta_1) = -\frac{EI}{l_0}(\theta_2 - \theta_1)
$$
Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org)

Shear locking, which is also a source of numerical problems in beam and plate problems,
is the result of excessive shear stresses. For thin elements, the cross section is expected to
remain perpendicular to the element centerline or midsurface of the element. This is the basic
assumption used in Euler–Bernoulli beam theory. Elements that are based on this theory do not
allow for shear deformation, and therefore, such elements do not, in general, suffer from the
shear locking problem. Examples of these elements are the two-dimensional Euler–Bernoulli
beam element and the three-dimensional cable element discussed previously in this chapter.
These elements, as demonstrated in the literature, are efficient in thin-beam applications. Shear
deformable elements, on the other hand, can suffer from locking problems if they are used in
thin structure applications. When these elements are used, the cross section does not remain
perpendicular to the element centerline, leading to shear forces. For thin structures, the resulting shear stresses can be very high leading to numerical problems. This problem can be circumvented by using the elastic line or midsurface approaches, the mixed variational principles,
or reduced integration methods.  

A Primer on the Kinematics of Discrete Elastic Rods by M. Khalid Jawed, Alyssa Novelia, Oliver M. OReilly (z-lib.org)

Euler beam Theory

This
classic theory, which dates to 1744, assumes that the bending moment in the
rod is linearly proportional to the curvature of the centerline of the rod, that the
deformation of the rod is planar, and that cross-sections that were normal to the
centerline in an undeformed configuration remain normal to the centerline as the
rod is deformed.   

cantilevered rod   

Kirchhoff’s Rod Theory  

The classic model for this problem dates to Kirchhoff in 1858. In his theory [31],
the centerline of the rod is not restricted to be planar and the cross-sections of the
rod are free to twist about the centerline. The cross-sections are assumed to remain  

normal to the centerline and their orientation can be described using a rotation tensor
RD.  

Finite element of slender beams in finite transformations  

![image-20211026133903982](D:\定理\有限元\image-20211026133903982.png)

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

==================Direct Position-Based Solver for Stiff Rods  

Cosserat Model

The Cosserat theory models a rod as a smooth curve s, called centerline. To describe the bending and twisting degrees of freedom, an orthonormal frame with basis d1, d2, d3 is attached to each point of centerline.

![image-20211030111858153](D:\定理\有限元\image-20211030111858153.png)

the cosserat theory defines the strain measure Gamma determines stretch and shear as
$$
\Gamma(s) = \frac{\partial }{\partial s}\bold r(s) - \bold d_3(s)
$$
bold r is a unit-speed parametrization for the rod`s rest pose. It follows that the tangent of centerline drds has unit length 。在我看在d3是一个类似Tangent的东西

Darboux Omega和angular velocity omega差不多，下面是它们的区别
$$
\frac{\partial }{\partial s}\bold d_i(s) = \Omega(s) \times \bold d_i(s) \qquad \frac{\partial}{\partial t}\bold d_i(s) = \omega(s) \times \bold d_i(s)
$$
Darboux vecor可以用下面表示
$$
\Omega(s) = 2\overline q(s)\frac{\partial}{\partial s}q(s)
$$
The first and second component W1 and W2 measure local bending or curvature in d1 and d2
direction and the third component measures torsion around d3.  

![image-20211030112919086](D:\定理\有限元\image-20211030112919086.png)

===Lagrangian field theory in space-time for geometrically exact Cosserat rods  

quaternion
$$
p = p_0 + \hat p = (p_0,p_1,p_2,p_3)
$$
the three columes of R are the so called directors - or the moving base vectors. The first two of them are spanning the rigid cross section of the rod. 
$$
d_1(p) = \begin{bmatrix} p_0^2 + p_1^2 - p_2^2 - p_3^2 \\ 2(p_1p_2 + p_0p_3) \\ 2(p_1p_3 - p_0p_2)\end{bmatrix} \qquad d_2(p) = \begin{bmatrix} 2(p_1p_2 - p_0p_3) \\ p_0^2 - p_1^2 + p_2^2 - p_3 ^2 \\ 2(p_2p_3 + p_0p_1)\end{bmatrix}
$$
The third one
$$
d_2(p) = \begin{bmatrix} 2(p_1p_3 + p_0p_2) \\ 2(p_2p_3 - p_0p_1) \\ p_0^2 - p_1^2 - p_2^2 + p_3^3\end{bmatrix}
$$
is kept close to the tangent of centerline by shearing force

![image-20211030155400931](D:\定理\有限元\image-20211030155400931.png)

![image-20211030155436120](D:\定理\有限元\image-20211030155436120.png)

![image-20211030155625208](D:\定理\有限元\image-20211030155625208.png)

![image-20211030155723842](D:\定理\有限元\image-20211030155723842.png)

![image-20211030161909637](D:\定理\有限元\image-20211030161909637.png)

================matlab有限元结构动力学

![image-20211102143226096](D:\定理\有限元\image-20211102143226096.png)

![image-20211102143331957](D:\定理\有限元\image-20211102143331957.png)
