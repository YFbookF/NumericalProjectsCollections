Step1 仅有剪切力，无阻尼，显式求解

Step2 仅有剪切力，有阻尼，显式求解

Step4 剪切 加 拉伸，有阻尼，隐式求解，步长5.0也可以。如果显式的话，这个步长已经炸了。 

假设一个一个三角形三个顶点的uv为[0,0],[1,0],[0,1]，可以看出是它初始顶点相对位置。此时我们已经可以计算出
$$
\begin{bmatrix} \bold w_u  &\bold w_v \end{bmatrix} = \begin{bmatrix} \bold \Delta x_1  &\bold \Delta x_2 \end{bmatrix}\begin{bmatrix} \Delta u_1 & \Delta u_2\\ \Delta v_1 & \Delta v_2\end{bmatrix}^{-1} = \begin{bmatrix} \bold \Delta x_1  &\bold \Delta x_2 \end{bmatrix}
$$
如果这个三角形三个顶点三维空间坐标为[0,0,0],[1,0,0],[0,1,0]，那么显然这个三角形的位置相对初始位置不会变动，那么我们也不应该添加新的力上去，这里所说的力包括剪切力，拉伸力，以及弯曲力。

如果顶点坐标分别为[0,0,0],[0,1,0],[-1,0,0]，那么这个三角形也仅仅是旋转了，并没有变形。我们也不应该添加任何力上去。此时我们的w矩阵长下面这个样子
$$
\begin{bmatrix} \bold w_u  &\bold w_v \end{bmatrix} = \begin{bmatrix} 0  &-1 \\ 1 & 0 \\ 0 & 0 \end{bmatrix}
$$
这点非常重要，物体纯旋转和纯位移，物体的面积或者体积不会变化，与形变没有半毛钱关系。弹性力学中天天嚷嚷着要对形变梯度要QR分解啦，要SVD分解啦，要算形变张量啦，都是为了把旋转分量揪出来。

真正能让物体体积改变的，就是剪切形变和拉伸形变。然而剪切也分纯剪切和带旋转的剪切，后者我们当然是不要的。至于前者，可能导致三角形的顶点位置变为[0,0,0],[1,0.5,0],[0.5,1,0]，那么W矩阵当然就变成了下面这样。
$$
\begin{bmatrix} \bold w_u  &\bold w_v \end{bmatrix} = \begin{bmatrix} 1  & 0.5 \\ 0.5 & 1 \\ 0 & 0 \end{bmatrix}
$$
由于三角形边产生了纯剪切形变，所以剪切力的Condition不再为零
$$
\bold C_{shear}(\bold x) = \bold w_u^T \bold w_v = 1 \qquad
$$
由于两条边也变长，所以拉伸力的Condition也不为零
$$
\bold C_{stretch}(\bold x) =  \begin{cases} ||\bold w_u(\bold x) ||-\bold b_u = \sqrt{1^2+(0.5)^2 + 0^2}-1\\||\bold w_v(\bold x) ||-\bold b_v = \sqrt{1^2+(0.5)^2 + 0^2}-1\end{cases}
$$
算出Condition后，想要算力的话，就要算出能量函数
$$
E_{\bold C}(\bold x) = \frac{k}{2}\bold C(\bold x)^T\bold C(\bold x)
$$
然后算力
$$
\bold f_i = \frac{\partial E_{\bold C}}{\partial \bold x_i} = -k\frac{\partial \bold C(\bold x)}{\partial \bold x_i}\bold C(\bold x)
$$
由于每个Condition的构造方法不同，所以对x求导的结果也不同。所以力要分开计算。不过共同点是，由于C是由w构成，所以要算C的导数，就要先算w对三个顶点的导数
$$
\frac{\partial \bold w_u}{\partial \bold x_0} = (\Delta v_1 - \Delta v_2)/det = -1\\
\frac{\partial \bold w_u}{\partial \bold x_1} = ( v_2)/det = 1\\
\frac{\partial \bold w_u}{\partial \bold x_2} = (-\Delta v_1)/det = 0\\
\frac{\partial \bold w_v}{\partial \bold x_0} = (\Delta u_2 - \Delta u_1)/det = -1\\
\frac{\partial \bold w_v}{\partial \bold x_1} = (- \Delta u_2)/det = 0\\
\frac{\partial \bold w_v}{\partial \bold x_2} = (\Delta v_1)/det = 1\\
$$
那么也很容易算出C对每个顶点求导了，它的结果是一个3x1的矩阵，有一列是因为只针对一个顶点。有三行是因为三维空间有三个坐标轴。比如
$$
\frac{\partial \bold C_{shear}(\bold x)}{\partial \bold x_0} = \begin{bmatrix} -1.5 \\ -1.5 \\ 0 \end{bmatrix} \qquad \frac{\partial \bold C_{shear}(\bold x)}{\partial \bold x_1} = \begin{bmatrix} 0.5 \\ 1 \\ 0 \end{bmatrix} \\
\frac{\partial \bold C_{shear}(\bold x)}{\partial \bold x_2} = \begin{bmatrix} 1 \\ 0.5 \\ 0 \end{bmatrix}
$$
上面三个变量需要给出三个顶点的坐标后才能算出来。那么作用在点x0上的力就是
$$
\bold f_{x0} = k\begin{bmatrix} 1.5 \\ 1.5 \\ 0 \end{bmatrix}
$$
也就是这个力会把x0往[1.5,1.5,0]这个方向推。剩下两个力也很容易算出来。

为了验证这个结果是否正确，我们把三角形质量设为1，刚度设为1，时间设为dt = 0.2，那么三个顶点在dt时间后，新坐标为
$$
\bold x_0 = \begin{bmatrix} 0.06 \\ 0.06 \\ 0\end{bmatrix} \qquad \bold x_0 = \begin{bmatrix} 0.98 \\ 0.46 \\ 0\end{bmatrix} \qquad \bold x_0 = \begin{bmatrix} 0.46 \\ 0.98 \\ 0\end{bmatrix}
$$
那么新的w矩阵就变成了如下样子
$$
\begin{bmatrix} \bold w_u  &\bold w_v \end{bmatrix} = \begin{bmatrix} 0.92  & 0.4 \\ 0.4 & 0.92 \\ 0 & 0 \end{bmatrix}
$$
然后再算一遍剪切的Condition，它只剩下0.736，比之前的1要小。这说明我们操控三角形的目的已经到达了。在这样继续下去，这个三角形的剪切Condition变成零。

当然，你会发现这个它的边越来越短啦，再短下去就变成个点了。于是我们可以用最熟悉的弹簧质点模型，也就是本篇论文中提到的第二个力，Strecth力做出约束了。和剪切力同样的思路。

除此之外，我们也希望布料很平坦，也就是相邻三角形网格之间的二面角为零，于是要引入第三种力，也就是弯曲力，它的Condition就是角度，范围是0~pi
$$
\bold C_{bend}(\bold x) = \theta
$$
但是定义一时爽，求导火葬场。要求出角度的对每个顶点的导，我们需要binormal

https://cg.informatik.uni-freiburg.de/course_notes/sim_03_cloth1.pdf
$$
\sin\theta = (\bold n_0 \times \bold n_1) \cdot \bold e\\
\cos \theta = \bold n_0 \cdot \bold n_1
$$
那么以下是几个关键的计算步骤
$$
\frac{\partial \cos \theta}{\partial \bold x} = -\sin\theta \frac{\partial \theta}{\partial \bold x} = \frac{\partial \bold n_0 \times \bold n_1}{\partial \bold x} \\
\frac{\partial \theta}{\partial \bold x} = \frac{-1}{\sin \theta}\frac{\partial \bold n_0 \cdot \bold n_1}{\partial \bold x} = \frac{-1}{(\bold n_0 \times \bold n_1)}\frac{\partial \bold n_0 \cdot \bold n_1}{\partial \bold x}
$$
而且
$$
\bold n = \begin{bmatrix} n_x \\ n_y \\  n_z\end{bmatrix} \qquad \widetilde n = \begin{bmatrix} 0 & -n_z & n_y \\ n_z & 0 & -n_x \\ -n_y & n_x & 0\end{bmatrix}
$$
那么将叉乘转换为点乘
$$
\bold n_0 \times \bold n_1 = \bold {\widetilde n_0} \cdot \bold n_1 =\begin{bmatrix} n_{0y}n_{1z} - n_{1y}n_{0x} \\ n_y \\  n_z\end{bmatrix} 
$$

$$
n_0 = \begin{bmatrix} (\bold x_{2} - \bold x_1)_y(\bold x_{0} - \bold x_1)_z  -(\bold x_{2} - \bold x_1)_z(\bold x_{0} - \bold x_1)_y \\  (\bold x_{2} - \bold x_1)_z(\bold x_{0} - \bold x_1)_x  -(\bold x_{2} - \bold x_1)_x(\bold x_{0} - \bold x_1)_z \\  (\bold x_{2} - \bold x_1)_x(\bold x_{0} - \bold x_1)_y  -(\bold x_{2} - \bold x_1)_y(\bold x_{0} - \bold x_1)_x\end{bmatrix}
$$

那么

然后对这玩意求导，很简单，因式分解就行了。先看第一行
$$
(\bold x_{2y} - \bold x_{0y})(\bold x_{1z} - \bold x_{0z}) - (\bold x_{2z} - \bold x_{0z})(\bold x_{1y} - \bold x_{0y}) \\= 
\bold x_{2y}\bold x_{1z} +(-\bold x_{1z} + \bold x_{2z})\bold x_{0y }+ (-\bold x_{2y} + \bold x_{1y})\bold x_{0z} -\bold x_{2z}\bold x_{1y}
$$


sim_03_cloth1

这玩意是个反对称矩阵，所以clothsim只算了一半？
$$
\frac{\partial \bold n_0}{\partial \bold x_0} = \begin{bmatrix} \partial \bold n_{0x}/\partial \bold x_{0x} & \partial\bold n_{0x}/\partial \bold x_{0y} & \partial\bold n_{0z}/\partial \bold x_{0x} \\ \partial\bold n_{0y}/\partial \bold x_{0x} & \partial\bold n_{0y}/\partial \bold x_{0y} & \partial\bold n_{0y}/\partial \bold x_{0z} \\ \partial\bold n_{0z}/\partial \bold x_{0x} & \partial\bold n_{0z}/\partial \bold x_{0y}& \partial\bold n_{0z}/\partial \bold x_{0z}\end{bmatrix} = \\
\begin{bmatrix}0 & -\bold x_{2z} + \bold x_{1z} & \bold x_{2y} - \bold x_{1y} \\ \bold x_{2z} - \bold x_{1z} & 0 & -\bold x_{2x} + \bold x_{1x} \\ -\bold x_{2y} + \bold x_{1y} & \bold x_{2x} - \bold x_{1x} & 0 \end{bmatrix} =\begin{bmatrix}0 & 0 & 1 \\ 0 & 0 & 1 \\ -1 & -1 & 0  \end{bmatrix}
$$
mplementing Baraff & Witkin’s Cloth Simulation
David Pritchard

强烈推荐先看这个，  

Derivation of discrete bending forces and
their gradients  
$$
\nabla_{\bold x_0}\theta = -\frac{1}{h_{01}}\hat {\bold n}_0^T\\
\nabla_{\bold x_1}\theta = \frac{\cos \alpha_{02}}{h_{01}}\hat {\bold n}_0^T + \frac{\cos \alpha_{12}}{h_{11}}\hat {\bold n}_1^T\\
\nabla_{\bold x_2}\theta =\frac{\cos \alpha_{01}}{h_{02}}\hat {\bold n}_0^T + \frac{\cos \alpha_{11}}{h_{12}}\hat {\bold n}_1^T\\
\nabla_{\bold x_3}\theta = -\frac{1}{h_{13}}\hat {\bold n}_1^T
$$


```
	dThetadP0 = -n0 / d00;
	dThetadP1 = c02 * n0 / d01 + c12 * n1 / d11;
	dThetadP2 = c01 * n0 / d02 + c11 * n1 / d12;
	dThetadP3 = -n1 / d13;
```

先看第一个 2A1 = e0 * h01
$$
\nabla_{\bold x_0}\theta = \frac{-1}{||\bold e_0||}(\cot \alpha_{11} + \cot \alpha_{12})\hat {\bold n_2}^T =-\frac{||\bold e_0||}{2A_1}\hat {\bold n}_0^T  =-\frac{1}{h_{01}}\hat {\bold n}_0^T
$$
注意
$$
\bold n_0 \times \bold n_1 = \bold e_{12}\sin \theta
$$
那么n应该是单位向量
$$
(\hat {\bold n}_0 \times \hat {\bold n}_1)\cdot \bold e_{01} = (\hat {\bold e}_{12} \cdot \bold e_{01})\sin\theta = -||\bold e_{01}||\cos \alpha_{01}\sin \theta
$$
alpha01是指在零个三角形中，顶点1的角度。theta是二面角。e01是顶点0到顶点1的距离。因为e12是单位向量，所以结果为1，最后被省略掉了。
$$
\begin{bmatrix} 0 \\ 1 \\ 0\end{bmatrix} \cdot \begin{bmatrix} 1 \\ 1 \\ 0\end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 0\end{bmatrix} \cos (45^o) = 1
$$
角度对顶点的导数推导完毕。

余弦求导，为什么有两个binormal?因为b00.dot(v01) 就是h01
$$
\nabla_{\bold x0} \cos \alpha_{01} = -\frac{h_{01}}{||\bold e_{12}||||\bold e_{10}||}\hat {\bold m}_3^T
$$

```
dc01dP0 = -b02 * b00.dot(v01) / v01.dot(v01);
```

法向量求导就和binormal有关
$$
\nabla_{\bold x_0}\hat {\bold n}_0 = \frac{\bold m_{00}}{h_{00}}\hat {\bold n}_0^T
$$
那个m就是binormal，见式子22

```
dn0dP0 = b00 * n0.transpose() / d00;
```

那么
$$
\nabla _{x0}(\frac{||\bold e_0||}{||\bold n_0||}) = \frac{\bold e_{12} \cdot \bold e_{10}}{||\bold n_0||^2}\hat{\bold m}^T_{01} = -\frac{\cos \alpha_{01}}{h_{01}h_{3}}\hat{\bold m}^T_{01}
$$
那么式子27
$$
\nabla (\frac{||\bold e_0||}{||\bold n_0||}) = \nabla(\frac{1}{h_{01}}) = -\frac{\Delta h_{01}}{h_{01}^2}
$$

求导
$$
\nabla (\bold u \cdot \bold v) = \bold u^T \nabla \bold v + \bold v^T \nabla
$$

bending energy is a function of the square of the mean curvature
$$
\nabla ||\bold u|| = \hat{\bold u}^T\nabla \bold u
$$
投影
$$
\nabla \hat {\bold u} = (I - \hat {\bold u}\hat {\bold u}^T)\frac{\nabla \bold u}{||\bold u||}
$$
叉积
$$
\nabla (\bold u \times \bold v) = \bold u \times \nabla \bold v - \bold v \times \nabla \bold u
$$

$$
\nabla \bold e_{10} = \bold x_1 - \bold x_0 = (-I,I,0,0)\\
\nabla \bold e_{20} = \bold x_2 - \bold x_0 = (-I,0,I,0)\\
\nabla \bold e_{12} = \bold x_1 - \bold x_2 = (0,I,-I,0)\\
\nabla \bold e_{13} = \bold x_1 - \bold x_3 = (0,I,0,I)\\
\nabla \bold e_{23} = \bold x_1 - \bold x_3 = (0,0,I,I)\\
$$
The gradient of the normal with
respect to a vertex is the outer product of the opposing edge normal with the
normal itself and scaled with one over the height to the opposing edge.  
$$

$$
![image-20210906120252024](D:\图形学书籍\系列流体文章\gif\image-20210906120252024.png)

先来对一个余弦求导，符号下标以ClothSim中的为准，和t的文章中不同。

首先对于
$$
\nabla \cos \alpha_{01} = \nabla (\hat {\bold e}_{10} \cdot \hat{\bold e}_{12}) = \hat {\bold e}_{10}\hat{\bold e}_{12}^T + \hat{\bold e}_{12}\hat {\bold e}_{10}^T = \\
\hat {\bold e}_{10}(1-\hat {\bold e}_{12}\hat {\bold e}_{12}^T)\frac{\nabla {\bold e}_{12}}{|| {\bold e}_{12}||} + \hat {\bold e}_{12}(1-\hat {\bold e}_{10}\hat {\bold e}^T_{10})\frac{\nabla {\bold e}_{10}}{|| {\bold e}_{10}||}
$$
还记得单位向量怎么求导吗？再推一遍
$$
{\bold e}_{10} = \bold x_1 - \bold x_0 \qquad \frac{\partial {\bold e}_{10}}{\partial \bold x_1} = \bold I \qquad\frac{\partial {\bold e}_{10}}{\partial \bold x_0} = -\bold I\\
{\bold e}_{12} = \bold x_1 - \bold x_2 \qquad \frac{\partial {\bold e}_{12}}{\partial \bold x_1} = \bold I \qquad\frac{\partial {\bold e}_{12}}{\partial \bold x_2} = -\bold I
$$
上面的e是3x1矩阵，I是3x3单位矩阵。那么余弦对各个点求导结果也很好写了
$$
\nabla_{\bold x0} \cos \alpha_{01} = - \frac{\hat {\bold e}^T_{12}(1-\hat {\bold e}_{10}\hat {\bold e}_{10}^T)}{||{\bold e}_{10}||}
$$




```
d2ThetadP0dP0 = -dn0dP0 / d00 + n0 * dd00dP0.transpose() / (d00 * d00);
```

$$
\nabla_{\bold x_0}(\nabla_{\bold x_0}\theta)^T = -\frac{\hat {\bold m}_{00}\hat {\bold n}_0^T + \hat {\bold n}_0\hat {\bold m}_{00}^T}{d_{00}^2}
$$

对于那篇文章的公式36，下标不同请注意

同样，对对面角求导，那么结果是零。所以本篇公式下面对应参考[1]的公式37
$$
\nabla_{\bold x_3}(\nabla_{\bold x_0}\theta)^T = 0
$$

```
d2ThetadP0dP3 = -dn0dP3 / d00 + n0 * dd00dP3.transpose() / (d00 * d00) = 0
```

那么对于第
$$
\nabla_{\bold x_0}(\nabla_{\bold x_2}\theta)^T = -\frac{1}{d_{00}d_{01}}(\hat {\bold m}_{00}\hat {\bold n}_0^T - \cos \alpha_{02}\hat {\bold n}_0\hat {\bold m}_{00}^T)
$$

```
d2ThetadP0dP1 = -dn0dP1 / d00 + n0 * dd00dP1.transpose() / (d00 * d00);
```



对点到对面直线距离求导，就是参考[2]的21页上面那几个公式
$$
\nabla_{\bold x_0}h_{00} = -\hat {\bold m}_{00}^T
$$

$$
\nabla_{\bold x_1}d_{00} = \frac{d_{00}}{d_{01}}\cos\alpha_{02}\hat {\bold m}_{00}^T
$$

算了，就当它是这样吧，公式是对的，看不懂代码

```
dd00dP0 = -b00;
dd00dP1 = b00 * -v21.dot(v02) / v21.dot(v21);
```

能这么写的原因是
$$
\frac{||\bold v_{02}||}{||\bold v_{21}||} = \frac{d_{00}}{d_{01}} \qquad ||\bold v_{12}|| ||\bold v_{02}||\cos\alpha_{02} = \bold v_{12} \cdot \bold v_{02}
$$
推完这个，剩下的就很轻松了