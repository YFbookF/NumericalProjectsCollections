Classical Mechanics An Introduction by Dieter Strauch (z-lib.org)

section 8.6.2
$$
\bold N = \dot{\bold L} = \sum_i \bold r_i \times \bold F_i = \bold R \times \bold F\qquad \bold L = \Theta \cdot \omega = I \dot {\bold \omega}
$$
然后
$$
bold \, \omega = \omega \bold e_w \qquad \bold N = N e_{w} \qquad I = e_w \cdot \bold \Theta \cdot e_w
$$


N is torque
$$
N = \bold e_w \cdot \bold N = e_w \cdot \bold R \times \bold F = e_w \cdot(-RMg\sin \varphi \bold e_w) = -RMg\sin\varphi
$$
此时varphi是R和F之间的夹角，此时角动量是
$$
L_z = \Theta_{zz}\dot \varphi = I \dot \varphi
$$
对于纯旋转来说
$$
I \ddot\varphi = -RMg\sin \varphi \qquad \ddot \varphi = -\frac{RMg}{I}\sin\varphi
$$
section 2.2.3

对于旋转物体，如果角度的变化为零，也就是角度为常数
$$
\bold r = r \bold e_r \qquad \bold v = \dot {\bold r} = \dot r \bold e_r \qquad \bold \alpha = \ddot {\bold r} = \ddot r \bold e_r \qquad \bold p = m\dot{\bold r} = m\dot{r}\bold e_r \\
l = \bold r \times \bold p = 0 \qquad T = \frac{1}{2}m \dot{\bold r}^2 = \frac{1}{2}m\dot r^2
$$
如果径向距离的变化为零，也就是r是常数
$$
\bold r = r \bold e_r \qquad \bold v = \dot{\bold r} = r\dot \varphi \bold e_\varphi \qquad \alpha = \ddot{\bold r} = -r\dot \varphi^2 \bold e_r + r\ddot \varphi \bold e_{\varphi}\\
\bold p = m\dot{\bold r} = mr\dot \varphi \bold e_\varphi = mv_\varphi \bold e_\varphi \qquad l = \bold r \times \bold p = mr^2 \dot \varphi \bold e_w = mr^2 \bold \omega\\ T = \frac{1}{2}m\dot {\bold r} = \frac{1}{2}mr^2 \dot \varphi^2
$$
原始，动量
$$
\bold p = m \dot{\bold r} = m(\dot r \bold e_r + r\dot \varphi \bold e_\varphi) \\
\bold L = \bold r \times \bold p = m(\bold r \times \bold v) = mr\bold e_r \times (\dot r \bold e_r + r \dot \varphi \bold e_\varphi) \\
= m(r\dot r \bold e_r \times  \bold e_r + r^2 \dot \varphi \bold e_r \times \bold e_\varphi) = mr^2 \dot \varphi \bold e_w = mr^2 \bold w  = \theta \bold w
$$
以及一些基础理论
$$
\begin{bmatrix} \bold e_r \\ \bold e_\varphi\end{bmatrix} = \begin{bmatrix} \cos \varphi & \sin\varphi \\ -\sin \varphi & \cos \varphi \end{bmatrix}\begin{bmatrix} \bold e_x \\ \bold e_y\end{bmatrix}
$$
那么求导
$$
\begin{bmatrix} \dot{\bold e_r} \\ \bold {\dot e_\varphi}\end{bmatrix} = -\dot \varphi \begin{bmatrix} -\sin \varphi & \cos\varphi \\ -\cos \varphi & -\sin \varphi \end{bmatrix}\begin{bmatrix} \bold e_x \\ \bold e_y\end{bmatrix}  \
$$

$$
 = -\dot \varphi \begin{bmatrix} -\sin \varphi & \cos\varphi \\ -\cos \varphi & -\sin \varphi \end{bmatrix}\begin{bmatrix} \cos\varphi & -\sin \varphi \\ \sin \varphi & \cos \varphi\end{bmatrix}\begin{bmatrix} \bold e_r \\ \bold e_\varphi\end{bmatrix}\\
 = \dot \varphi \begin{bmatrix} 0 & 1\\ -1 & 0\end{bmatrix}\begin{bmatrix} \bold e_r \\ \bold e_\varphi\end{bmatrix} = \dot \varphi\begin{bmatrix}  \bold e_\varphi \\ -\bold e_ r\end{bmatrix}
$$

section 2.2.3
$$
\bold r = r\bold e_r \qquad \dot {\bold r} = \dot r \bold e_r + r \bold {\dot e}_r = \dot r \bold e_r + r \dot \varphi \bold e_\varphi = v_r\bold e_r + v_\varphi \bold e_\varphi
$$
那么kinetic 
$$
T = \frac{1}{2}m\dot {\bold r}^2 = \frac{1}{2}m(v_r^2 + v_\varphi^2) = \frac{1}{2}m(\dot r^2 + r^2\dot \varphi^2)
$$
那么加速度
$$
\ddot{\bold r} = (\ddot r\bold e_r + \dot r \dot{\bold e}_r) + (\dot r \dot \varphi \bold e_\varphi + r\ddot \varphi \bold e_\varphi + r\dot \varphi \dot {\bold e}_\varphi) \\
=\ddot r\bold e_r + \dot r \dot \varphi \bold e_\varphi + \dot r \dot \varphi \bold e_\varphi + r\ddot \varphi \bold e_\varphi - r\dot \varphi \dot \varphi \bold e_r \\
= (\ddot r - r\dot \varphi^2)\bold e_r + (r\ddot \varphi + 2\dot r \dot \varphi)\bold e_\varphi \\
-\frac{1}{m}\nabla V = -\frac{1}{m}(\frac{\partial V}{\partial r}\bold e_r + \frac{1}{r}\frac{\partial V}{\partial \varphi}\bold e_\varphi)
$$
其中V是势能
$$
V(x) = -\int
$$
Computational Continuum Mechanics by Ahmed A. Shabana (z-lib.org)

![image-20211025194431539](E:\mycode\collection\定理\理论力学\动力学\image-20211025194431539.png)

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

![image-20211026142432965](E:\mycode\collection\定理\理论力学\动力学\image-20211026142432965.png)

===========An Introduction to Physically Based Modeling: Rigid Body Simulation I—Unconstrained Rigid Body Dynamics  

the quantity w(t) is called the angular velocity, the magnitude |w(t)| tells how fast the body is spinning.

For relating rotating matrix R and vector w, consider a vector r in world space. r is a direction which is independent of translational effects, so dot r(t) is independent of v(t) which is the rigid body velocity.

Then we decompose vec r into vec a and vec b, vec a is parallel to w, and vec b is perpendicular to w. So the radius of the circle is |b|, velociry is |w||b|，so
$$
|w(t) \times b| = |w(t)||b| \qquad \dot r(t) = w(t) \times (b)  \qquad 0 = w(t) \times (a)
$$
也就是
$$
\dot r(t) = w(t) \times (b) = w(t) \times (b) + w(t) \times (a) = \omega(t) \times r(t)
$$
注意
$$
R(t) = \begin{bmatrix} r_{xx} & r_{yx} & r_{zx} \\ r_{xy} & r_{yy} & r_{zy} \\ r_{xz} & r_{yz} & r_{zz}\end{bmatrix}
$$
那么

Let’s put all this together now. At time t, we know that the direction of the x axis of the rigid
body in world space is the first column of R.t/, which is  
$$
R(t) \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} r_{xx} \\ r_{xy} \\ r_{xz} \end{bmatrix}
$$
At time t, the derivative of the first column of R.t/ is just the rate of change of this vector: using the cross product rule we just discovered, this change is  
$$
\dot R(t) = \begin{bmatrix}w(t) \times \begin{bmatrix} r_{xx}  \\ r_{xy}  \\ r_{xz} \end{bmatrix} & w(t) \times \begin{bmatrix} r_{yx}  \\ r_{yy}  \\ r_{yz} \end{bmatrix} & w(t) \times \begin{bmatrix} r_{zx}  \\ r_{zy}  \\ r_{zz} \end{bmatrix}\end{bmatrix}
$$
===========An Introduction to Physically Based Modeling: Rigid Body Simulation I—Unconstrained Rigid Body Dynamics  

if a and b are 3 -vectors then a \times b is
$$
a^*b = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0\end{bmatrix}\begin{bmatrix} b_x \\ b_y \\ b_z\end{bmatrix} = \begin{bmatrix} a_yb_z - b_ya_z \\ a_zb_x - a_xb_z \\ a_xb_y - b_xa_y\end{bmatrix} = a\times b
$$
同样rotation 也可以将叉乘化为矩阵乘法

===========An Introduction to Physically Based Modeling: Rigid Body Simulation I—Unconstrained Rigid Body Dynamics  

torque
$$
\tau_i(t) = (r_i(t) - x(t)) \times F(t)
$$
total linear momentum
$$
P(t) = \sum m_i \dot r_i(t) = \sum (m_i v(t) + m_i \omega(t) \times (r_i(t) - x(t)) \\= \sum m_i v(t) + \omega(t) \times \sum m_i(r_i(t) - x(t))
$$
这是因为
$$
\dot r_i(t) = v(t) + \omega(t) \times (r_i(t) - x(t))
$$
这又是因为
$$
\dot r_i(t) = \omega^*R(t) r_{0i} + v(t) = \omega(t)^*(R(t) r_{0i} + x(t) -x(t)) + v(t) \\
= \omega(t)^*(r_i(t) - x(t)) + v(t)
$$
![image-20211026211144686](E:\mycode\collection\定理\理论力学\动力学\image-20211026211144686.png)

之后还有代码，好notes

An Introduction to Physically Based Modeling Rigid Body Simulation II Nonpenetration Constraints  

线加速度，角角速度之类的
$$
\dot L = \tau = \sum r_i' \times F_i = \dot I (t)\omega + I(t)\dot \omega = \frac{d}{dt}(I(t)\omega)
$$
============physically-based models with rigid and deformable components

![image-20211102160941069](E:\mycode\collection\定理\理论力学\动力学\image-20211102160941069-16358406252211.png)

![image-20211102161041685](E:\mycode\collection\定理\理论力学\动力学\image-20211102161041685.png)



![image-20211102161112935](E:\mycode\collection\定理\理论力学\动力学\image-20211102161112935.png)



![image-20211102161123705](E:\mycode\collection\定理\理论力学\动力学\image-20211102161123705.png)



