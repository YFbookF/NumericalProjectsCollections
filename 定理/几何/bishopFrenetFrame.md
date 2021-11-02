Super-Helices for Predicting the Dynamics of Natural Hair  

The centerline, r(s,t), is the
curve passing through the center of mass of every cross section.
This curve describes the shape of the rod at a particular time t but
it does not tell how much the rod twists around its centerline. In
order to keep track of twist, the Cosserat model introduces a material frame ni(s,t) at every point of the centerline2. By material,
we mean that the frame ‘flows’ along with the surrounding material
upon deformation.   

![image-20211027100906988](D:\定理\几何\image-20211027100906988.png)

Basile Audoly, Yves Pomeau - Elasticity and Geometry_ From hair curls to the nonlinear response of shells-Oxford University Press (2010)

There are two important symmetries for a uniformly bent rod. The first is an invariance
by mirror symmetry with respect to any cross-section. The second is the invariance by
translation along the centre line. Upon mirror symmetry with respect to the plane of a
cross-section, the material vectors d1 and d2 are conserved, while the material tangent
d3 is inverted. As a result, the invariance requires d1(s) · d3(s) = 0 and d2(s) · d3(s) = 0
for all s. These scalar products are zero, as in the reference configuration, and so the
strain components 13 = 0 and 23 = 0 (recall that the off-diagonal stress components
measure the change in the scalar product between material vectors). In other words,
the material tangent d3 remains everywhere exactly10 perpendicular to the plane of the
cross-section.  

Therefore, the derivative of any unitary
vector (in fact, of any vector of constant norm) with respect to some parameter yields a
vector that is perpendicular to it.   

法向量
$$
\bold n(s) = \frac{\bold t'(s)}{|\bold t'(s)|}
$$
Improving_Frenets_Frame_Using_Bishops_Frame
$$
\kappa = ||\frac{dT}{ds}|| \qquad T(t) = \frac{\vec s'(t)}{||\vec s'(t)||}
$$
### Frenet Frame

$$
\dot T = \kappa^f N^f \qquad \dot N^f = -\kappa^fT + \tau^f B^f \qquad \dot B^f = -\tau N^f
$$
Basile Audoly, Yves Pomeau - Elasticity and Geometry_ From hair curls to the nonlinear response of shells-Oxford University Press (2010)

注意
$$
\bold n' = (\bold n' \cdot \bold t)\bold t + (\bold n' \cdot \bold b)\bold b + (\bold n' \cdot \bold n)\bold n
$$
那么
$$
\bold n' \cdot \bold t = \frac{d(\bold n \cdot \bold t)}{ds} - \bold n \cdot \bold t' = -\bold n \cdot \bold t' = -k
$$
Darboux vector 如下，g 是torsion, k 是curvature
$$
\vec \Omega(s) = g(s)\bold t(s) + k(s)\bold b(s)
$$
By these relations, Ω(s) can be interpreted as the local rate of rotation of the Serret–Fr´enet
frame.6 Note that, by construction, Ω(s) · n(s) = 0: the Serret–Fr´enet does not rotate about
n(s), as indicated by the cross in Fig. 1.2.  

The Serret–Fr´enet frame arises naturally when one deals with mathematical curves, but
it is not well suited to the mechanics of rods. One of the problems associated with it is that
it is not always well defined, even when the curve is smooth: this happens for a straight
line, for instance, for which t(s) vanishes everywhere and so n(s) is ill-defined. Even worse,
the Serret–Fr´enet might not be continuous even though the curve is C∞ (case of a planar
curve with an inflexion point), and k(s) and/or g(s) may not go to zero even though the
underlying curve converges uniformly to a straight line (the case of a helix with constant
pitch and infinitesimal radius).  

### bishop frame

$$
\dot T = \kappa_1 M_1 + \kappa M_2 \qquad \dot M_1 = -\kappa_1 T \qquad \dot M_2 = -\kappa_2T
$$
两者之间的联系是
$$
\kappa^f(s) = \sqrt{\kappa_1^2(s) + \kappa_2^2(s)}
$$
并且
$$
\bold N = \cos\theta \bold M_1 + \sin\theta M_2 \qquad B = -\sin\theta M_1 + \cos\theta M_2 \\
\kappa_1 = \kappa \cos \theta \qquad \kappa_2 = \kappa \sin\theta
$$
A Primer on the Kinematics of Discrete Elastic Rods by M. Khalid Jawed, Alyssa Novelia, Oliver M. OReilly (z-lib.org)

注意，R是旋转，E1E2E3是笛卡尔坐标系
$$
\bold e_t = \bold R_B \bold E_3 \qquad \bold u = \bold R_B \bold E_1 \qquad \bold v = \bold R_B E_2
$$
旋转还有以下几种表示方式，Rd是triads，Rb是bishop，Rsf是Frenet
$$
\bold R_D = \bold R(\phi,\bold e_t)\bold R_{SF} = \bold R(\phi + \varphi,\bold e_t) \bold R_B
$$
triads好像是随便选了俩
$$
\bold R(\theta,\bold r) = \cos(\theta)(\bold I - \bold r \otimes \bold r) + \sin (\theta)skewt(\bold r) + \bold r \otimes \bold r
$$
r是任何一个向量

Basile Audoly, Yves Pomeau - Elasticity and Geometry_ From hair curls to the nonlinear response of shells-Oxford University Press (2010)

the elementary
$$
d\bold M = [(x - x_0)\bold d_1(s) + (y - y_0)\bold d_2(s)] \times \sigma_{iz}(x,y)\bold d_i(s)dxdy
$$
the elementary contact force applied by the downstream part is
$$
d\bold F = \sigma_{iz}(x,y)\bold d_i(s)dxdy
$$
The arm of this moment with respect to the intersection of the centre line and the cross-section s, with coordinates (x0, y0, s)
$$
(x - x_0)\bold d_1(s) + (y - y_0)\bold d_2(s)
$$
and
$$
\bold M = EI^{(1)}\kappa^{(1)}\bold d_1 + EI^{(2)}\kappa^{(2)}\bold d_2
$$
==================Direct Position-Based Solver for Stiff Rods  

Cosserat Model

The Cosserat theory models a rod as a smooth curve s, called centerline. To describe the bending and twisting degrees of freedom, an orthonormal frame with basis d1, d2, d3 is attached to each point of centerline.

![image-20211030111858153](D:\定理\几何\image-20211030111858153.png)
