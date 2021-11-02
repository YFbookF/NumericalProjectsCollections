================A Finite Element Method for Animating Large Viscoplastic Flow  

![image-20211029131647789](D:\定理\有限元\image-20211029131647789.png)

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
===============Nonlinear Material Design Using Principal Stretches  

G 是tetrahedron上的点，P is the first piola kirchhoff stress, b = an , AN is the area weighted material normals

The gradient of the elastic force of a tetrahedron is, F is deformation gradient
$$
\frac{\partial \bold G}{\partial \bold u} = \frac{\partial \bold G}{\partial \bold F}\frac{\partial \bold F}{\partial \bold u} = (\frac{\partial \bold P}{\partial \bold F}\bold B_m)\frac{\partial \bold F}{\partial \bold u} \in \R^{9\times12}
$$
G 是3x3矩阵。u 是12的向量，也就是顶点的displacement. dFdu 和b 是模拟期间的常数矩阵。the force gradient of the remaining vertex is
$$
\frac{\partial \bold g_0}{\partial \bold u} = -(\frac{\partial \bold g_1}{\partial \bold u} + \frac{\partial \bold g_2}{\partial \bold u} + \frac{\partial \bold g_3}{\partial \bold u})
$$
那么
$$
\frac{\partial \bold P}{\partial \bold F_{ij}} = \frac{\partial \bold U}{\partial \bold F_{ij}}\bold P(\hat{\bold F})\bold V^T + \bold U\frac{\partial \bold P(\hat{\bold F})}{\partial \bold F_{ij}}\bold V^T + \bold U\bold P(\hat{\bold F})\frac{\partial \bold V^T}{\partial \bold F_{ij}}
$$
![image-20211030094627933](D:\定理\有限元\image-20211030094627933.png)

![image-20211030094816105](D:\定理\有限元\image-20211030094816105.png)

话说lambda是个什么？，应该是the strain energy may be expressed in terms of the invariants

![image-20211030095321449](D:\定理\有限元\image-20211030095321449.png)
$$
I_c = \trace(\bold C) = \lambda_1^2 + \lambda_2^2 + \lambda_3^2 \qquad III_C = \det(\bold C) = \lambda_1^2\lambda_2^2\lambda_3^2 \\
II_C = \bold C :\bold C = \lambda_1^4 + \lambda_2^4 + \lambda_3^4
$$
![image-20211030095030173](D:\定理\有限元\image-20211030095030173.png)
$$
\bold P(\hat{\bold F}) = diag(\frac{\partial \Psi}{\partial \lambda_1},\frac{\partial \Psi}{\partial \lambda_2},\frac{\partial \Psi}{\partial \lambda_3})
$$
because we simplify the design by assuming that psi takes the form
$$
\Psi(\lambda_1,\lambda_2,\lambda_3) = f(\lambda_1) + f(\lambda_2) + f(\lambda_3) + g(\lambda_1\lambda_2) + g(\lambda_2\lambda_3) + g(\lambda_3\lambda_1) + h(\lambda_1\lambda_2\lambda_3)
$$
Generally, a hyperelastic material model should satisfy the Hill’s stability criterion (also called Drucker’s condition), which requires a monotonic increase of strain energy density with increase in strain   

![image-20211030100509375](D:\定理\有限元\image-20211030100509375.png)

