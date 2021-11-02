A Primer on the Kinematics of Discrete Elastic Rods by M. Khalid Jawed, Alyssa Novelia, Oliver M. OReilly (z-lib.org)
$$
ax(\bold C)\times \bold b = \bold C \bold b
$$
2.3
$$
\vec v = ax(\dot {\bold F}\bold F^T) \qquad \vec \omega = ax(\bold {F}'\bold F^T)
$$
w 是darboux vector。v is known as an angular velocity vector.

https://en.wikipedia.org/wiki/Darboux_vector

在frenet 中，Darboux vector w 有如下性质
$$
\vec \omega = \tau \bold T + \kappa \bold B \qquad \vec w \times \begin{bmatrix} \bold T\\ \bold N \\ \bold B\end{bmatrix} = \begin{bmatrix} \bold T'\\ \bold N' \\ \bold B'\end{bmatrix}
$$
![image-20211026105051344](D:\定理\几何\image-20211026105051344.png)

The Frenet–Serret axis system, moving with the point, has an angular velocity. Dividing this by the
(signed) point speed, that is, taking the derivative of the angular position of the axis system with respect to
the path position, gives the Darboux vector

![image-20211026105605035](D:\定理\几何\image-20211026105605035.png)

A FINITE STRAIN BEAM FORMULATION. THE THREE-DIMENSIONAL  

time derivatives of the moving frame
$$
\dot {\bold t}_1(S,t) = \bold W(S,t)\bold t_1(S,t) 
$$
W = -W^T is a spatial skew-symmetric tensor which defines the spin of the moving frame. The associated axial vector w(S,t), which satisfies W(S,t)w(S,t) = 0, gives the vorticity of the moving frame. In terms of the vorticity vector. (2.2) may be written as
$$
\dot {\bold t_1}(S,t) = \bold w(S,t)\times \bold t_1(S,t)
$$
linear momentum per unit of reference arc length
$$
\bold L_t = \int \rho_0(\xi,S)\dot {\vec \varphi}(\xi,S,t)d\xi = A\dot \varphi_0(S,t)
$$
angular momentum per unit of reference arc length
$$
\bold H_t = \int _A \rho_0(\xi,S)[\bold x - \vec \varphi_0(S,t)]\times \dot{\vec \varphi}(\xi,S,t)d\xi
$$
又因为
$$
\dot{\vec \varphi} - \dot{\vec \varphi}_0 = \sum_{\Gamma = 1}^2\xi_r\dot {\bold t_r} = \bold w \times (\vec \varphi - \vec \varphi_0)
$$
那么

![image-20211026142432965](D:\定理\几何\image-20211026142432965.png)

![image-20211026142856930](D:\定理\几何\image-20211026142856930.png)

Basile Audoly, Yves Pomeau - Elasticity and Geometry_ From hair curls to the nonlinear response of shells-Oxford University Press (2010)
$$
\bold d'_1(s) = \bold \Omega(s) \times \bold d_1(s) \qquad \bold d_2'(s) = \bold \Omega(s) \times \bold d_2(s) \qquad \bold d_3'(s) = \bold \Omega(s) \times \bold d_3(s)
$$
he interpretation of equation (3.2) is that the material frame rotates with a rotation
‘velocity’ Ω(s) when the centre line is followed at unit speed—the vector Ω(s) is in fact
a rate of rotation per unit length along the centre line.   

where we have introduced the Darboux vector
$$
\bold \Omega(s) = \kappa^{(1)}\bold d_1(s) + \kappa^{(2)}\bold d_2(s)  +\tau(s)\bold d_3(s)
$$
From equation (3.3), the numbers
κ(1) and κ(2) express by how much the material frame rotates around the directions d1 and
d2 of the cross-section. They are called the material curvatures. The number τ expresses
how much the material frame rotates around the tangent d3, and is called the material
twist of the rod, or the twist rate.  

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

![image-20211030113357261](D:\定理\几何\image-20211030113357261.png)

Position and Orientation Based Cosserat Rods  

strain measure for shear and stretch
$$
\Gamma(s) = \partial _s \bold r(s) - \bold d_3(s) \qquad \tilde \Gamma = \bold R(q)^T \partial_s \bold r - \bold e_3
$$
Further deformations of the rod are bending and twisting. To define a strain measure for
those we use the Darboux vector W. It is defined as  
$$
\vec \Omega = \frac{1}{2}\sum_{k=1}\bold d_k \times \bold d_k'
$$
An important feature of
the material frame strain measures is that they are invariant under
rigid body transformations. When the entire rod gets translated or
rotated the strain measures report the same deformation.  

![image-20211030140814316](D:\定理\几何\image-20211030140814316.png)