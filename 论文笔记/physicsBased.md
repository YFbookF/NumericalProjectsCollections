若物体的重心为x，空间位置r，旋转矩阵为R
$$
\bold p(t) = \bold x(t) + \bold R(t)\bold r_0
$$
但是，旋转矩阵对时间求导
$$
\dot{\bold R}(t)\bold r_0 = \bold \omega(t)\times \bold r(t)
$$
这样速度如下
$$
\bold v(t) = \dot{\bold x}(t) + \omega(t) \times \bold r(t)
$$
w则是角动量

线性动量
$$
\bold P(t) = \sum_{i=1}^N m_i \bold v_i(t)
$$
角动量
$$
\bold L(t) = \sum_{i=1}^N \bold r_i(t) \times m_i \bold v_i(t)
$$
那么合起来就算
$$
L(t) = \sum_{i=1}^N m_i\bold r_i(t) \times(\dot{\bold x}(t) + \omega(t) \times \bold r_i(t)) = \\\sum_{i=1}^N m_i\bold r_i \times \dot{\bold x}(t) + \sum_{i=1}^Nm_i\bold r_i(t)\times \omega(t)\times \bold r_i(t)
$$
牛顿第二定律
$$
\frac{d}{dt}\begin{bmatrix}\bold P(t) \\ \bold L(t) \end{bmatrix} = \begin{bmatrix}\bold f(t) \\ \bold \tau(t) \end{bmatrix}
$$
f是net force，tau是net torque

![preview](https://pic2.zhimg.com/v2-d4a238facfcb39193e245cb064aa0650_r.jpg)

力矩
$$
\tau = \bold r\times \bold F
$$
r是位置，F是力的大小。比如跷跷板，x = 1m的位置处放y = 1n的物体，那么力矩就是z = 1m*n。

angular momentum for an arbitray

旋转角动量如下
$$
L = \sum m_a \bold r_a \times \bold v_a = \sum m_a \bold r_a \times(\omega \times \bold r_a)
$$
当然可以这么写
$$
\bold r \times(\omega \times \bold r) = \\\begin{bmatrix} (y^2 +x^2)\omega_x & -xy\omega_y & -xz\omega_z \\-yx\omega_x & (z^2 + x^2)\omega_y & -yz\omega_z \\ -zx\omega_x & -yz\omega_y & (x^2 + y^2)\omega_z \end{bmatrix}
$$
于是我们可以写出tensor of interia
$$
\bold I = \sum m_a \begin{bmatrix}y^2 + x^2 & -xy & -xz \\ -yx & z^2 + x^2 & -yz \\ -zx & -yz & x^2+ y^2 \end{bmatrix}
$$
也就是
$$
\bold L = \bold I \omega
$$
习题

三个点的位置为[a,0,0],[0,a,2a],[0,2a,a]，那么principal moments of inertia如下
$$
\bold I = \begin{bmatrix} 10 ma^2 & 0 & 0 \\ 0 & 6ma^2 & -4ma^2 \\ 0  & -4ma^2 & 6ma^2\end{bmatrix}
$$
那么特征值为2ma^2，10ma^2，10ma^2，对应的特征向量为
$$
\frac{1}{\sqrt{2}}\begin{bmatrix} 0 \\ 1 \\ 1\end{bmatrix} \qquad \begin{bmatrix} 1 \\ 0 \\ 0\end{bmatrix} \qquad \frac{1}{\sqrt{2}}\begin{bmatrix} 0 \\ 1 \\ -1\end{bmatrix}
$$
也就是principle axes如下
$$
\hat u_1 = \frac{1}{\sqrt{2}}(\hat j + \hat k) \qquad \hat u_2 = \hat i \qquad \hat u_3 = \frac{1}{\sqrt{2}}(\hat j - \hat k)
$$
http://home.iitb.ac.in/~shukla/class_mech.html

弹性势能密度是应变和应力的积
$$
\eta = \frac{1}{2}\sigma_{ij}\varepsilon_{ij} = \frac{1}{2}\sigma :\varepsilon
$$
上面这个公式你肯定记不住，干脆看看

```
float eta = 0;
for(int i = 0;i < 3;i++)
	for(int j = 0;j < 3;j++)
		eta += stress[i,j]*strain[i,j]/2
```

Elastic forces will seek to reduce the energy of the system,thus they will be in the direction of negative gradient of energy.That is the force at a point will be
$$
\bold f_i = -\frac{\partial \eta}{\partial \bold x_i}
$$
Another important quantity is the traction,or force per unit area
$$
\tau = \sigma \bold n
$$
Forces can also be defined by integrating tractions over the boundary of a region R
$$
\bold f = \oint \sigma \bold ndS
$$
The definition of traction makes clear that stress maps from normals to forces

如果法向量和力都在世界空间，那么那么这就是Cauchy stress,sigma。如果法向量和力都在材料空间，那么这就是second Piola-Kirchhoff力S。如果法向量在材料空间，力在世界空格键，那么就first Piola-Kirchhoff力P

由于它们描述的都是同一个事情，仅仅是在不同的坐标系下，所以可以
$$
\bold P = J \sigma \bold F^{-T} = \bold F \bold S\qquad J = det(\bold F)
$$
We could breaking the deformation gradient F into two parts,an elastic part and a plastic part
$$
\bold F = \bold F_e \bold F_p
$$
eqn.145

probably the most common strain model in computer graphics,is the co-rotated model where we compute the polar decompostiion
$$
\bold F = \bold Q \widetilde {\bold F}
$$
the compute the strain as
$$
\varepsilon = \frac{1}{2}(\widetilde{\bold F} + \widetilde{\bold F}^T) - \bold I
$$
then compute stress as 
$$
\sigma = \lambda Tr(\varepsilon) \bold I + 2\mu\varepsilon
$$
and finally forces are given by
$$
\bold f = \bold Q \sigma \bold n_i
$$

```
        # 注意Eigen库和numpy算乘法的方式不一样，不过怎么会不一样呢？
        F = np.dot(basis[ie],X)
        
        # scipy 的 svd 解法的精度问题
        for i in range(3):
            for j in range(3):
                if F[i,j] < 1e-10:
                    F[i,j] = 0
        Q = np.zeros((3,3)) # 旋转矩阵
        # 注意scipy 算出来的是V的转置，所以是V_transpose
        U,sigma,Vt = scipy.linalg.svd(F)
        
        Q = np.dot(Vt,U)
        Ftilde = np.dot(F,np.transpose(Q))
        
        iden = np.zeros((3,3))
        iden[0,0] = iden[1,1] = iden[2,2] = 1
        strain = (Ftilde + np.transpose(Ftilde)) / 2 - iden
        tr = strain[0,0] + strain[1,1] + strain[2,2]
        stress = lam * tr * iden + 2 * mu * strain

        qs = np.dot(stress,Q)
        
        elementQ[ie,:,:] = Q[:,:]
        
        force[elements[ie,0],:] += scaledot(qs,normal[ie,0,:]) / 6
        force[elements[ie,1],:] += scaledot(qs,normal[ie,1,:]) / 6
        force[elements[ie,2],:] += scaledot(qs,normal[ie,2,:]) / 6
        force[elements[ie,3],:] += scaledot(qs,normal[ie,3,:]) / 6
```

![image-20211003221552355](D:\图形学书籍\系列流体文章\gif\image-20211003221552355.png)

隐式方法一
$$
\bold v(t+\Delta t) = \bold v(t) + \Delta t \cdot \bold M^{-1}\bold f(\bold x(t+\Delta t),t+\Delta t) \\
\bold x(t+\Delta t) = \bold x(t) + \Delta t \cdot\bold (t + \Delta t)
$$
最后由于
$$
\bold K\bold x - \bold K\bold x_0+ \bold D\dot{\bold x} + \bold M\ddot{\bold x} = \bold f_{ext}
$$
也就是
$$
\bold f = \bold f_{ext} - \bold K\bold x + \bold K\bold x_0- \bold D\dot{\bold x}
$$
then
$$
\bold v(t+\Delta t) = \bold v(t) + \\\Delta t \cdot \bold M^{-1}[-\bold K(\bold x(t) + \Delta t\bold v(t+\Delta t)) + \bold K\bold x_0 - \bold D\bold v(t+\Delta t) + \bold f_{ext}]
$$
Rearranging
$$
(\bold M + \Delta t^2 \bold K + \Delta t\bold D)\bold v(t+\Delta t) = \bold M \bold v(t) + \Delta t(-\bold K(\bold x(t) - \bold x_0) + \bold f_{ext})
$$
The integrator is often referred to as Linearly Implicit Euler and can be also be derived form a first-order Taylor expansion

旋转矩阵
$$
\begin{bmatrix} \cos \phi & -\sin\phi & 0 & 0 \\ \sin\phi & \cos \phi & 0 & 0 \\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1\end{bmatrix}\begin{bmatrix} \cos \theta & 0 & \sin\theta & 0 \\ 0 & 1 & 0 & 0 \\ -\sin\theta & 0 & \cos\theta & 0\\ 0 & 0 & 0 & 1\end{bmatrix}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos \psi & -\sin\psi & 0 \\0 & \sin\psi & \cos\psi & 0\\ 0 & 0 & 0 & 1\end{bmatrix} \\
= \begin{bmatrix} c\phi c\theta & c\phi s\theta s\psi - s\phi c\psi & c\phi s\theta c\psi + s\phi s\psi & 0 \\ s\phi c\theta  & s\phi s\theta s\psi + c\phi c\psi  & s\phi s\theta c\psi - c\phi s\psi & 0 \\ -s\theta & c\theta s\psi & c\theta s\psi & 0 \\ 0 & 0 & 0 & 1\end{bmatrix} \\
$$
![image-20211003225908690](D:\图形学书籍\系列流体文章\gif\image-20211003225908690.png)

![image-20211003225931366](D:\图形学书籍\系列流体文章\gif\image-20211003225931366.png)

这是嘛玩意

![image-20211003231722819](D:\图形学书籍\系列流体文章\gif\image-20211003231722819.png)

![image-20211003231703855](D:\图形学书籍\系列流体文章\gif\image-20211003231703855.png)