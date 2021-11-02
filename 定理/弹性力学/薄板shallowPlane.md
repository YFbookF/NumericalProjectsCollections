René De Borst, Mike Crisfield, Joris Remmers and Clemens Verhoosel - Nonlinear Finite Element Analysis of Solids and Structures, 2nd edition

section 10.1

The curvature are given by:
$$
\chi = \begin{bmatrix} \chi_x \\ \chi_y \\ \chi_{xy}\end{bmatrix} = \begin{bmatrix} \frac{\partial \theta_x}{\partial x} \\ \frac{\partial \theta_y}{\partial y} \\ \frac{\partial \theta_x}{\partial y} + \frac{\partial \theta_y}{\partial x} \end{bmatrix}
$$
the out-of-plane shear strains follow a standard format
$$
\gamma = \begin{bmatrix} \gamma_{xz} \\ \gamma_{yz}\end{bmatrix} =  \begin{bmatrix} \theta_x \\ \theta_y\end{bmatrix} +  \begin{bmatrix} \frac{\partial w}{\partial x} \\ \frac{\partial w}{\partial y}\end{bmatrix}
$$
In a shell we have five non-vanishing stress components, the in-plane normal stresses sigmaxx, sigmayy, the in-plane shear stress sigmaxy, and the out-of-plane shear stresses sigmaxz and sigmayz

normal forces
$$
\bold N = \begin{bmatrix} N_x \\ N_y \\ N_{xy} \end{bmatrix} = \int_{-h/2}^{+h/2}\begin{bmatrix} \sigma_{xx}(z_l) \\ \sigma_{yy}(z_l) \\ \sigma_{xy}(z_l) \end{bmatrix}dz_l
$$
the bending moments
$$
\bold M = \begin{bmatrix} M_x \\ M_y \\ M_{xy} \end{bmatrix} = \int_{-h/2}^{+h/2}\begin{bmatrix} \sigma_{xx}(z_l) \\ \sigma_{yy}(z_l) \\ \sigma_{xy}(z_l) \end{bmatrix}z_ldz_l
$$
and the out-of-plane shear forces
$$
\bold Q = \begin{bmatrix} Q_x \\ Q_y \end{bmatrix} = \int_{-h/2}^{+h/2}\begin{bmatrix} \sigma_{xz}(z_l) \\ \sigma_{yz}(z_l) \end{bmatrix}dz_l
$$
================matlab有限元结构动力学

![image-20211102151714139](D:\定理\弹性力学\image-20211102151714139.png)
