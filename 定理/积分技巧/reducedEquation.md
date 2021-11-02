Real-Time Subspace Integration for St.Venant-Kirchhoff Deformable Models  
$$
M \ddot u + D(u,\dot u) + R(u) = f
$$
其中
$$
D(u,\dot u) = (\alpha M + \beta K(u))\dot u
$$
in model reduction of solid mechanics, the displacement vector is expressed as u = Uq, U is some displacement basis matrix, q is the vector of reduced coordinates
$$
\ddot q + \tilde D(q,\dot q) + \tilde R(q) = \tilde f \qquad \tilde D = U^TD(Uq,U\dot q) \\
\tilde R(q) = U^TR(Uq) \qquad \tilde f = U^T f
$$
![image-20211026213817154](D:\定理\积分技巧\image-20211026213817154.png)