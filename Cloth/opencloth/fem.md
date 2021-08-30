$$
\int _{\Omega^e}W_\sigma d = W_q + \int_{\delta \Omega^e}W_tdA
$$

上式左边是体积力，右边是外力加表面力。这就是功平衡方程(work balance equation)。功的表达式是位移乘上力，那么每单位体积的功可由下面的式子表示
$$
W_{\sigma} = (\delta \varepsilon)^T\sigma
$$
外力由结点力组成qe组成，可由下式组成
$$
W_q = (\delta u^e)^Tq^e
$$
ue是位移。那么就是下面的方程
$$
\int_{\Omega^e}(\delta \varepsilon)^T\sigma dV = (\delta \bold u^e)^T \bold q^e + \int_{\delta \Omega^e}(\delta \bold u)^T \bold t dA
$$
这个t是表面力。而且位移被换成位移的变化率了。上面的经过替换就是
$$
(\delta \bold u^e)^T\int_{\Omega^e}\bold B^T \sigma dV = (\delta \bold u^e)^T(\bold q^e + \int_{\delta \Omega^e} \bold N^T \bold t dA)
$$
这已经是变分后了的，可以将ue消掉
$$
\bold q^e + \int_{\delta \Omega^e}\bold N^T \bold t dA = \int_{\Omega^e}\bold B^T\sigma dV = \int_{\Omega^e}\bold B^T\bold D \varepsilon dV= \int_{\Omega^e}\bold B^T\bold D \bold B \bold u^e dV
$$
那么
$$
\bold K^e = \bold B^T \bold D\bold BV^e = \int_{\Omega^e}\bold B^T\bold D \bold B  dV
$$

$$
\bold K^e \bold u^e = \bold q^e + \bold f^e \qquad \bold f^e = \int_{\delta \Omega^e}\bold N^t\bold tdA
$$

当然，我知道你肯定没看懂这玩意，我最初看的时候也没理解。

我们知道的是作用在节点上的力，我们想知道的是节点的位移。他俩之间的关系可以由功平衡关系式求出。功平衡关系式就是说

(体积 乘以 单位体积上所作的功) 等于 (外部做功 加上 表面积 乘以 单位表面积上的所作的功)

这个关系式是对于一个单元来说的，这个单元可能是由三个节点的平面三角形，可能是有四个节点的平面四边形，也可能是有四个节点的空间四面体。

对于平面三角形来说，D就是3x3矩阵。对于单独的一个节点来说，B是3x2矩阵，因为每个节点有x和y两个方向。对于三角形的三个节点来说，就是3x6矩阵。

对于平面四面体来说,B就是3x8矩阵了，这很合理。不过涉及到等参单元，为什么要用Nxi去乘位置呢？

对于空间四面体来说，B就是6x12矩阵了，6行是因为三维的，12列是因为每个节点要占3列。此时D也是6x6的。

在谢祚水，王自力，吴剑国主编的《计算结构力学》的第3.3.1节，

### 平面应力

就是以下几个应力为零
$$
\sigma_z = \sigma_{xz} = \sigma_{yz} = 0
$$
stress strain下的isotropic elasticity matrix
$$
\begin{bmatrix} \sigma_x \\ \sigma_y \\ \tau_{xy} \end{bmatrix} = \frac{E}{1-\nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & (1-\nu)/2 \end{bmatrix}\begin{bmatrix} \varepsilon_x \\ \varepsilon_y \\ \gamma_{xy} \end{bmatrix}
$$
strain-stress
$$
\begin{bmatrix} \varepsilon_x \\ \varepsilon_y \\ \gamma_{xy} \end{bmatrix}= \frac{1}{E} \begin{bmatrix} 1 & -\nu & 0 \\ -\nu & 1 & 0 \\ 0 & 0 & 2(1+\nu) \end{bmatrix}\begin{bmatrix} \sigma_x \\ \sigma_y \\ \tau_{xy} \end{bmatrix} 
$$


剪切模量为
$$
G = \frac{E}{2(1+\nu)}
$$

### 平面应变

$$
\varepsilon_z = \gamma_{xz} = \gamma_{yz} = 0 \qquad \tau_{xz} = \tau_{yz} = 0
$$

stress-strain
$$
\begin{bmatrix} \sigma_x \\ \sigma_y \\ \tau_{xy} \end{bmatrix} = \frac{E}{(1+\nu)(1-2\nu)} \begin{bmatrix} 1 - \nu & \nu & 0 \\ \nu & 1-\nu & 0 \\ 0 & 0 & (1-2\nu)/2 \end{bmatrix}\begin{bmatrix} \varepsilon_x \\ \varepsilon_y \\ \gamma_{xy} \end{bmatrix}
$$
并且
$$
\sigma_z = \frac{E}{1+\nu}(\frac{\nu}{1-2\nu}(\varepsilon_x + \varepsilon_y))
$$






### 三角形

应变与位移的关系如下
$$
\varepsilon = \begin{bmatrix} \varepsilon_x \\ \varepsilon_y \\ \gamma_{xy} \end{bmatrix} = \begin{bmatrix} \partial /\partial x & 0 \\ 0 &\partial /\partial y  \\ \partial /\partial y & \partial /\partial x\end{bmatrix}\begin{bmatrix} u \\ v \end{bmatrix} = \bold B\bold d
$$
这是对于单个节点的。对于三角形上的三个节点，B矩阵形式如下
$$
\bold B = \frac{1}{2\Delta }\begin{bmatrix} b_i & 0 & b_j &0 & b_m & 0 \\ 0 & c_i & 0   & c_j & 0 & c_m \\ c_i & b_i & c_j & b_j & c_m & b_m \end{bmatrix}
$$
那个向上的三角形是三角形的面积。应变和应力的关系是
$$
\sigma = \bold D \bold B \bold d - \bold D \varepsilon_0 = \bold S \bold d - \bold D \varepsilon_0
$$
其中对于平面应力问题，每个节点的S矩阵如下
$$
\bold S = \frac{E}{2(1-\mu^2)\Delta}\begin{bmatrix} b_i & \mu c_i \\ \mu b_i & c_i \\ (1-\mu)c_i/2 & (1-\mu)b_i/2\end{bmatrix}
$$
如果是平面应变问题
$$
\bold S = \frac{E(1-\nu)}{2(1+\nu)(1-2\nu)\Delta}\begin{bmatrix} b_i & \nu c_i/(1-\nu) \\ \nu b_i/(1-\nu) & c_i \\ (1-2\nu)c_i/2(1-\nu) & (1-\nu)b_i/2(1-\nu)\end{bmatrix}
$$


单位体积应变能
$$
\bold U_0 = \frac{1}{2}\varepsilon^T\bold D\varepsilon + \varepsilon ^T\sigma_0
$$
上面这个玩意怎么求出来的？不清楚，只知道它非常像共轭梯度要求的那个东西。最后F是体积力，p是表面力，那么整个单元总势能
$$
\Pi_{pe} = \iint_A(\frac{1}{2}\varepsilon^T\bold D\varepsilon + \varepsilon ^T\sigma_0)dA - \iint_A f^TFdA - \int_S f^TpdS
$$
这里的f其实就是下面那个d。f是每个节点的位移，但是每个节点的位移最终还是由笛卡尔坐标系上三个分量表示，而d就是那个三个分量。由于vareps = Bd，也就是
$$
\Pi_{pe} = \frac{1}{2}\bold d^t\iint_A(\bold B^T\bold D\bold BdA)\bold d + \bold d^T\iint_A \bold B ^T\sigma_0dA \\-\bold d^T\iint_A \bold N^T\bold FdA - \bold d^T\int_S\bold N^T\bold pdS
$$


根据变分原理，单元总势能泛函式的一阶变分等于零，即
$$
\frac{\partial \Pi_{pe}}{\partial d_i} = 0 \qquad \bold K^e\bold d = \bold P^e
$$
也就是可以想象一个开口向上的二次函数，x轴代表某个位移状态，y轴就是它的势能。为了让势能最低，位移状态肯定处在斜率为零的位置。

那么可由将位移变量消去，得到
$$
\bold K^e = \iint_A \bold B^T\bold D\bold BdA
$$

$$
\bold P^e = -\iint_A \bold B ^T\sigma_0dA +\iint_A \bold N^T\bold FdA + \int_S\bold N^T\bold pdS
$$



### 四面体

对四面体单元的结构刚度矩阵时，位移函数应该取线性形式
$$
\begin{cases} u = a_1 + a_2x + a_3y + a_4z \\v = a_5 + a_6x + a_7y + a_8z\\w = a_9 + a_{10}x + a_{11} + a_{12}z\end{cases}
$$
或者
$$
d = \phi_0\alpha
$$
其中
$$
d = \begin{bmatrix} u & v & w\end{bmatrix} \quad \phi = \begin{bmatrix} \phi_p & 0 & 0 \\ 0 & \phi_p & 0 \\ 0 & 0  & \phi_p \end{bmatrix} \quad \phi_p = \begin{bmatrix}1 \\ x \\ y \\ z \end{bmatrix}^T \quad \alpha = \begin{bmatrix}a_1 \\ a_2 \\ .. \\ a_{12} \end{bmatrix}^T
$$
这里d就是节点所有方向上的位移，phi就是个坐标轴上单位向量，alpha就是个系数矩阵。反过来写
$$
\alpha = \phi^{-1}d
$$

$$
\phi^{-1} = \frac{1}{6V}\begin{bmatrix} \phi_e & 0 & 0 \\ 0 & \phi_e & 0 \\ 0 & 0 & \phi^e\end{bmatrix}
$$

其中体积求法如下
$$
Volume = \frac{1}{6}\begin{bmatrix} 1 & x_1 & y_1 & z_1 \\ 1 & x_2 & y_2 & z_2 \\ 1&x_3 & y_3 & z_3 \\ 1 & x_4 & y_4 & z_4\end{bmatrix}
$$
strain-displacement
$$
\begin{bmatrix} \varepsilon_{xx} \\  \varepsilon_{yy}\\ \varepsilon_{zz}\\ \gamma_{xy}\\ \gamma_{xz} \\ \gamma_{yz}\end{bmatrix} = \begin{bmatrix} \partial /\partial x & 0 & 0  \\ 0 & \partial /\partial y & 0 \\ 0 & 0 & \partial /\partial z \\ \partial /\partial y & \partial /\partial x & 0 \\ \partial /\partial z & 0 & \partial /\partial x \\ 0 & \partial /\partial z & \partial /\partial y  \end{bmatrix}\begin{bmatrix} u \\ v \\ w\end{bmatrix} = \bold S \bold u
$$
注意黑体u是位移。与之前结构力学中的d对应。

注意三角形是线性的。四边形不是线性的？不过
$$
det\begin{bmatrix} 1 & x_0 & y_0 \\ 1 & x_1 & y_1 \\ 1 & x_2 & y_2\end{bmatrix} = det\begin{bmatrix} x_2 - x_0  & y_2 - y_0 \\ x_3 - x_0 & y_3 - y_0\end{bmatrix}
$$
那么
$$
\bold P^{-1} = \begin{bmatrix} . & ... \\ : & \bold E^{-1} \end{bmatrix}
$$
也就是
$$
\frac{dN_n}{dx} = \bold E^{-1}_{0,n-1} = 1 - \bold E^{-1}_{00} - \bold E^{-1}_{01} - \bold E^{-1}_{02}\\
\frac{dN_n}{dy} = \bold E^{-1}_{1,n-1}= 1 - \bold E^{-1}_{10} - \bold E^{-1}_{11} - \bold E^{-1}_{12}\\
\frac{dN_n}{dz} = \bold E^{-1}_{2,n-1}= 1 - \bold E^{-1}_{20} - \bold E^{-1}_{21} - \bold E^{-1}_{22}
$$
我发现同样的公式，python写起来更好看。matlab和julia虽然也很简介，但没python好看。c++组装矩阵要写几行。glm库eigen库都是如此。

```
    element_volume[ie] = dot(e10,cross(e20, e30))/6
    E = np.array([e10,e20,e30])
    invDetE = 1 / np.linalg.det(E)
    
    # 手动逆矩阵，牛逼
    invE10 = (e20[2]*e30[1] - e20[1]*e30[2]) * invDetE
    invE20 = (e30[2]*e10[1] - e30[1]*e10[2]) * invDetE
    invE30 = (e10[2]*e20[1] - e10[1]*e20[2]) * invDetE
    invE00 = - invE10 - invE20 - invE30
    
    invE11 = (e20[0]*e30[2] - e20[2]*e30[0]) * invDetE
    invE21 = (e30[0]*e10[2] - e30[2]*e10[0]) * invDetE
    invE31 = (e10[0]*e20[2] - e10[2]*e20[0]) * invDetE
    invE01 = - invE11 - invE21 - invE31
    
    invE12 = (e20[1]*e30[0] - e20[0]*e30[1]) * invDetE
    invE22 = (e30[1]*e10[0] - e30[0]*e10[1]) * invDetE
    invE32 = (e10[1]*e20[0] - e10[0]*e20[1]) * invDetE
    invE02 = - invE12 - invE22 - invE32
    
    element_B[ie,0,:] = np.array([invE00,invE01,invE02])
    element_B[ie,1,:] = np.array([invE10,invE11,invE12])
    element_B[ie,2,:] = np.array([invE20,invE21,invE22])
    element_B[ie,3,:] = np.array([invE30,invE31,invE32])
```

