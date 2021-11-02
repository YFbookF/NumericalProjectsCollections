Position-Based Simulation of Continuous Material  

Energy 的定义方法如下
$$
E_s= \int_\Omega\Psi d\bold X = V \Psi_s(\bold F)
$$
求导
$$
\nabla_\bold x E_s = \int_\Omega \frac{\partial \Psi}{\partial \bold x}d\bold X = \int_\Omega \frac{\partial \Psi_s}{\partial \bold F}\frac{\partial \bold F}{\partial \bold x} d\bold X = \int_\Omega \bold P(\bold F)\frac{\partial \bold F}{\partial \bold x}d\bold X
$$
其中应变能密度和pk如下
$$
\qquad \Psi_s = \frac{1}{2}\varepsilon:S = \frac{1}{2}\varepsilon :C\varepsilon \qquad \bold P(\bold F) = \bold F \bold C \vec \varepsilon
$$
is the first piola-kirchhoff stress tensor of saint venant-kirchhoff model

如果是neohookean model，P如下
$$
\bold P(\bold F) = \mu \bold F - \mu \bold F^{-T} + \frac{\lambda \log(I_3)}{2}\bold F^{-T} \qquad \Psi_s = \frac{\mu}{2}(I_1 - \log(I_3)-3) + \frac{\lambda}{8}\log^2 (I_3)
$$
能量求导如下
$$
\begin{bmatrix} \frac{\partial E_s}{\partial \bold x_1} & \frac{\partial E_s}{\partial \bold x_2}  & \frac{\partial E_s}{\partial \bold x_3} \end{bmatrix} = V \bold P(\bold F)\bold D_m^T \qquad \bold F = \bold D_s \bold D_m^{-1}
$$
注意Stress 与Strain的关系，以及strain energy的表达式
$$
\bold S = \bold C \varepsilon
$$
============Analytic Eigensystems for Isotropic Distortion Energies  

应变能的梯度和Hess都很重要. The gradient of Psi with respect to the flattened deformation gradient f = vec(F) yields
$$
\frac{\partial \Psi}{\partial \bold f} = \sum_i \frac{\partial \Psi}{\partial I_i} \frac{\partial I_i}{\partial f}
$$
![image-20211030195848764](D:\定理\连续介质力学\image-20211030195848764.png)

```
  //symmetric ARAP Hessian
  DRDF =        (2 / (s0 + s1)) * (q0 * q0');
  DRDF = DRDF + (2 / (s1 + s2)) * (q1 * q1');
  DRDF = DRDF + (2 / (s0 + s2)) * (q2 * q2');
```

3d hessian of I1 is then
$$
\frac{\partial^2 I_1}{\partial \bold f^2} = \frac{2}{\sigma_2 + \sigma_3}\bold t_1 \bold t_1^T + \frac{2}{\sigma_3 + \sigma_1}\bold t_2 \bold t_2^T + \frac{2}{\sigma_1 + \sigma_3}\bold t_3 \bold t^T
$$
=============Point Based Animation of Elastic, Plastic and Melting Objects  

we compute the elastic body forces via the strain energy density
$$
U = \frac{1}{2}(\varepsilon \cdot \sigma) = \frac{1}{2}(\sum_{i=1}^j\sum_{i=1}^j \varepsilon_{ij}\sigma_{ij})
$$
the elastic force per unit volume at a point xi is the negative gradient of the strain energy density with respect to the point displament ui. For a Hookean, this expression can be written as
$$
\bold f_i = -\nabla_{\bold u_i}U = -\sigma \frac{\vec \varepsilon}{\bold u_i}
$$
Green’s strain tensor given in Eqn. (3) measures linear elongation (normal strain) and alteration of angles (shear strain)
but is zero for a volume inverting displacement field. Thus,
volume inversion does not cause any restoring elastic body
forces. To solve this problem, we add another energy term  
$$
U_v = \frac{1}{2}k_v(|\bold J| - 1)^2
$$
that penalizes deviations of the determinant of the Jacobian from positive unity, i.e. deviations from a right handed volume conserving transformation. The corresponding body forces are  
$$
\bold f_i = -\nabla_{\bold u_i}U = -k_v(|\bold J| - 1)\nabla_{\bold u_i}|\bold J|
$$
we estimate the strain energy stored around phyxel i as
$$
U_i = v_i \frac{1}{2}(\varepsilon_i \cdot \sigma_i)
$$
yields the forces acting at phyxel i and all its neighbors j
$$
\bold f_j = -v_i \sigma_i \nabla_{\bold u_j}\varepsilon_i
$$
![image-20211101212105690](D:\定理\连续介质力学\image-20211101212105690.png)
