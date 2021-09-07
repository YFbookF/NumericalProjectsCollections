写完这一节最大的收获，就是对“法向量对顶点求导”，“余弦对顶点求导”，“角度对顶点求导”的认识更加多了。首先我们的三角形如下。其中三角形021是第0个三角形，三角形231是第一个三角形

```
2---3
| \ |
0---1
```

然后再复习一遍三角形余弦公式
$$
\bold u \cdot \bold v = ||\bold u||||\bold v||\cos \alpha_{uv}
$$
毫无压力。接下来我们会以各种各样的姿势见到这个公式，比如，如果u是单位向量，那么等式直接变成
$$
\hat{\bold u} \cdot \bold v = ||\bold v||\cos \alpha_{uv}
$$
所以我们会看到某个项推导着推导着就突然没了，其实是单位向量的长度为1就自动消去了。好了，开始吧。

第一小步，算出边的向量v，以及对应的单位向量e。p是顶点位置。代码写出来很简单，就不写公式了。其中normalized()是Eigen库的函数，这是c++中很欢迎的库。

```
v10 = p1 - p0;
v20 = p2 - p0;
v23 = p2 - p3;
v13 = p1 - p3;
v12 = p1 - p2;

e10 = v10.normalized();
e20 = v20.normalized();
e23 = v23.normalized();
e13 = v13.normalized();
e12 = v12.normalized();
```

第二小步，算余弦和法向量。也是一刀秒掉。

```
c00 = e10.dot(e20);//c00是指第0个三角中，顶点0的余弦
c01 = e10.dot(e12);
c02 = -e20.dot(e12);

c13 = e13.dot(e23);
c11 = e12.dot(e13);
c12 = -e23.dot(e12);

n0 = (e12.cross(e10)).normalized();
n1 = (e23.cross(e12)).normalized();
```

第四步，计算单位化的binormal，也就是和法向量以及切向量都正交的那个家伙。这里的binormal是指在一个三角形里，由三角形某个顶点指向对边的方向。代码可如下写

```
b00 = (e01 - e21 * (e21.dot(e01))).normalized();
b01 = (-e01 - e02 * (e02.dot(-e01))).normalized();
b02 = (-e02 - e01 * (e01.dot(-e02))).normalized();

b13 = (e32 - e21 * (e21.dot(e32))).normalized();
b12 = (-e32 - e31 * (e31.dot(-e32))).normalized();
b11 = (-e31 - e32 * (e32.dot(-e31))).normalized();
```

或者直接叉积也可以，注意方向。要算的结果是由三角形某个顶点指向对边的方向，因此法向量需要在边向量的左边。

```
b00 = -e12.cross(n0);
b01 = -e20.cross(n0);
b02 = e10.cross(n0);

b13 = e12.cross(n1);
b12 = -e13.cross(n1);
b11 = e23.cross(n1);
```

第五步，计算顶点到到对边的距离，结果为标量。原理为很简单的向量点积，binormal是单位向量哦。
$$
\hat {\bold m}_{00}\cdot \bold v_{10} = ||\bold v_{10}||\cos\alpha_{00}
$$

```
d00 = b00.dot(v10);//d00是指第0个三角形中，顶点0到对边的距离
d01 = b01.dot(-v12);
d02 = b02.dot(-v20);

d11 = b11.dot(-v13);
d12 = b12.dot(v12);
d13 = b13.dot(v13);
```

第六步，计算弯曲的Condition，也就是角度
$$
\sin \theta = (\bold n_0 \times \bold n_1)\cdot \bold e_{12}\qquad
\cos \theta = \bold n_0 \cdot \bold n_1
$$

```
Real sinTheta = n1.dot(b00);
Real cosTheta = n0.dot(n1);
theta = atan2(sinTheta, cosTheta);
```

第八步，计算法向量对顶点的导数。法向量为单位向量。下面公式的计算结果都是3x3的矩阵。
$$
\nabla \hat{\bold n} = \nabla (\frac{\bold n}{||\bold n||}) = \bold n\nabla(\frac{1}{||\bold n||}) + \frac{1}{||\bold n||}\nabla \bold n
$$
而且
$$
\nabla(\frac{1}{||\bold n||}) = -\frac{1}{||\bold n||^2}\nabla(||\bold n||)=\frac{-1}{||\bold n||^2}(\hat{\bold n}^T\nabla \bold n)
$$
用上面两个式子化简可得
$$
\nabla \hat{\bold n} = (1 - \hat{\bold n}\hat{\bold n}^T)\frac{\nabla \bold n}{||\bold n||}
$$
这也使参考[2]第8页第一个公式。这就是将单位法向量求导转换为了对法向量求导。又
$$
\nabla(\bold n) = \nabla(\bold u \times \bold v) = \frac{\bold u \times \nabla \bold v - \bold v \times \nabla \bold u}{||\bold u \times \bold v||}
$$
那么参考[2]的第九页中间那个公式我看不懂，暂且跳过。总之经过一顿操作可以得到下面的式子
$$
\nabla \hat{\bold n} = \frac{(\bold u \times \hat{\bold n})\hat{\bold n}^T \nabla \bold v - (\bold v \times \hat{\bold n})\hat{\bold n}^T \nabla \bold u}{||\bold n||}
$$
对于第0个三角形的法向量，它的导数就应该是
$$
\nabla \hat{\bold n}_0 = \frac{(\bold e_{12} \times \hat{\bold n}_0)\hat{\bold n}_0^T \nabla \bold e_{10} - (\bold e_{10} \times \hat{\bold n}_0)\hat{\bold n}_0^T \nabla \bold e_{12}}{||\bold n_0||}
$$
首先是边的单位向量乘单位法向量再除以法向量长度，可以转换为单位binormal除以距离，如下
$$
\frac{(\bold e_{12} \times \hat{\bold n}_0)}{||\bold n_0||} = \frac{||\bold e_0||(\hat{\bold e}_{12} \times \hat{\bold n}_0)}{||\bold n_0||} = -\frac{||\bold e_{12}||\hat {\bold m}_{00}}{||\bold n_0||} = -\frac{\hat {\bold m}_{00}}{h_{00}}
$$
这也是参考[2]第17页第一个公式。那么对于第0个三角形的法向量，导数为
$$
\nabla \hat{\bold n}_0 =-\frac{\hat {\bold m}_{00}}{h_{00}}\hat{\bold n}_0^T \nabla \bold e_{10} - \frac{\hat {\bold m}_{02}}{h_{01}}\hat{\bold n}_0^T \nabla \bold e_{12}
$$
接下来让边的单位向量，对各个顶点求导
$$
\nabla \bold e_{10} = \bold x_1 - \bold x_0 = (-I,I,0,0)\\
\nabla \bold e_{20} = \bold x_2 - \bold x_0 = (-I,0,I,0)\\
\nabla \bold e_{12} = \bold x_1 - \bold x_2 = (0,I,-I,0)\\
\nabla \bold e_{13} = \bold x_1 - \bold x_3 = (0,I,0,-I)\\
\nabla \bold e_{23} = \bold x_1 - \bold x_3 = (0,0,I,-I)\\
$$
那么第0个法向量对第0个顶点求导，e10求导完是-1，e12求导完是0，结果是一个3x3矩阵。
$$
\nabla_{\bold x_0}\hat{\bold n}_0 = \frac{\hat{\bold m}_{00}}{d_{00}}\hat{\bold n}_{0}^T
$$
第0个法向量对第1个顶点求导，e10求导完是1，e12求导完是1，结果如下
$$
\nabla_{\bold x_1}\hat{\bold n}_0 = -\frac{(\hat{\bold m}_{00} + \hat{\bold m}_{02})}{d_{01}}\hat{\bold n}_{0}^T= \frac{\hat{\bold m}_{01}}{d_{01}}\hat{\bold n}_{0}^T
$$
对第二个顶点，e10求导完是0，e12求导完是-1，结果如下
$$
\nabla_{\bold x_2}\hat{\bold n}_0 = \frac{\hat{\bold m}_{02}}{d_{02}}\hat{\bold n}_{0}^T
$$
对于第三个顶点，e10和e12求导完结果都是0。第1个三角形的法向量求导同理。写成代码如下

```
dn0dP0 = b00 * n0.transpose() / d00;
dn0dP1 = b01 * n0.transpose() / d01;
dn0dP2 = b02 * n0.transpose() / d02;
dn0dP3 = Matrix3::Zero();

dn1dP0 = Matrix3::Zero();
dn1dP1 = b11 * n1.transpose() / d11;
dn1dP2 = b12 * n1.transpose() / d12;
dn1dP3 = b13 * n1.transpose() / d13;
```

第八步，计算角度对顶点的导数，也就是参考[2]第16页公式(18)~(21)。角度求导公式为
$$
\nabla \theta = -\frac{\nabla (\hat{\bold n}_0 \cdot\hat{\bold n}_1)}{\sin \theta}=  -\frac{\hat{\bold n}_0^T \nabla \hat{\bold n}_1 + \hat{\bold n}_1^T\nabla \hat{\bold n}_0}{\sin \theta}
$$
法向量求导之前已经推导过了，但是现在我们需要将叉积换个位置，也就是使用公式
$$
(\bold u \times \bold v)\cdot \bold w = (\bold w \times \bold u)\cdot \bold v
$$
那么就是下面这个
$$
-(\hat {\bold n}_0 \times \bold e_{12}) \cdot \hat {\bold n}_1 = (\hat {\bold n}_0 \times \hat {\bold n}_1) \cdot \bold e_{12} = (\hat {\bold e}_{12} \cdot \bold e_{12}) \sin \theta = ||\bold e_{12}||\sin\theta
$$
上面这个将sin消去的变换详细请看参考[2]第12~13页的公式。
$$
\frac{\hat{\bold n}_0^T \nabla \hat{\bold n}_1 }{\sin \theta} = \frac{||\bold e_{12}||\hat{\bold n}_1^T\nabla \bold e_{32} -(\hat{\bold e}_{12} \cdot \bold e_{32})\hat{\bold n}_1^T \nabla \bold e_{12}}{||\bold n_1||}
$$
以及
$$
\frac{\hat{\bold n}_1^T \nabla \hat{\bold n}_0 }{\sin \theta} = \frac{-||\bold e_{12}||\hat{\bold n}_0^T\nabla \bold e_{20} +(\hat{\bold e}_{12} \cdot \bold e_{20})\hat{\bold n}_0^T \nabla \bold e_{12}}{||\bold n_0||}
$$
由
$$
2A_0 = ||\bold n_0|| = ||\bold e_{12}||d_{00}
$$
再结合边的单位向量的求导公式。
$$
\nabla_{\bold x_0}\theta = -\frac{1}{d_{00}}\hat {\bold n}_0^T \qquad \nabla_{\bold x_3}\theta = -\frac{1}{d_{13}}\hat {\bold n}_1^T
$$

再结合参考[2]第14~16页的公式，就得到了下面的公式。这玩意推了5页多，实在太多了，所以细节还是请看参考[2]吧。本篇只打算将大概流程整理一遍。
$$
\nabla_{\bold x_1}\theta = \frac{\cos \alpha_{02}}{d_{01}}\hat {\bold n}_0^T + \frac{\cos \alpha_{12}}{d_{11}}\hat {\bold n}_1^T
$$

$$
\nabla_{\bold x_2}\theta =\frac{\cos \alpha_{01}}{d_{02}}\hat {\bold n}_0^T + \frac{\cos \alpha_{11}}{d_{12}}\hat {\bold n}_1^T
$$

```
dThetadP0 = -n0 / d00;
dThetadP1 = c02 * n0 / d01 + c12 * n1 / d11;
dThetadP2 = c01 * n0 / d02 + c11 * n1 / d12;
dThetadP3 = -n1 / d13;
```

第九步，距离对顶点求导

由于我们算的binormal是顶点到对边的单位向量，所以如果顶点每向沿着法向量的方向移动一个量，必然导致这个点到对边的距离减少这个量点积刚才算出来的binormal
$$
\nabla_{\bold x_0}d_{00} = -\hat {\bold m}_{00}^T 
$$
举个例子，一个二维三角形三个顶点分部是p0 = [0,0],p1 = [2,0],p2 = [0,2]。那点p0的binormal很明显就是a = [0.707,0.707]。如果这个点移动了b= [0.4,0]，那么这个点离对边的距离，减少了向量b的长度，也就是0.2828。但是巧了，a点积b结果的也是这个值。这种情况就是因为一开始提到的三角形余弦公式。

因此可以说某个点到对边的距离，对这个点求导，结果是这个点的binormal乘上-1。

那么就有人说，如果不移动p0，偏要移动p1，那么随着p1的移动，p0到对边的距离会如何变化？

假如p1移动了一个向量a，我们先让这个这个向量点积点p0的binormal，毕竟如果往别的方向移动的话，是不会增加d00的长度的。

![image-20210906234113096](D:\图形学书籍\系列流体文章\gif\image-20210906234113096.png)

现在已经求出了蓝色虚线的长度，也就是向量a点积binormal的结果，那么与蓝色虚线平行的橙色虚线的长度是多少？那么这样求相似三角形就可以了。很显然，蓝色虚线的长度比橙色虚线的长度，等于d01比图中红色实线的长度，而后者可通过d00乘以cos(角021)求出来，那么就得到了下面这个式子
$$
\nabla_{\bold x_1}d_{00} = \frac{d_{00}}{d_{01}}\cos\alpha_{02}\hat {\bold m}_{00}^T
$$
剩下移动p2导致的d00的改变也用相似的秋风可求得。最后，无论我们如何移动p3，p0到对边的距离都不会有任何改变。所以对其求导结果是0。
$$
\qquad \nabla_{\bold x_2}d_{00} = \frac{d_{00}}{d_{02}}\cos\alpha_{01}\hat {\bold m}_{00}^T \qquad \nabla_{\bold x_3}d_{00} = 0
$$
这些公式就是参考[2]第21页前4个公式。代码可以这么写

```
dd00dP0 = -b00;
dd00dP1 = b00 * -v12.dot(v20) / v12.dot(v12);
dd00dP2 = b00 * v12.dot(v10) / v12.dot(v12);
dd00dP3 = Vector3::Zero();
```

这是因为
$$
\frac{-\bold v_{12} \cdot \bold v_{20}}{||\bold v_{12}||^2} =\frac{ \cos \alpha_{02}||\bold v_{12}||||\bold v_{20}||}{||\bold v_{12}||^2}  =\frac{ \cos \alpha_{02}||\bold v_{20}||}{||\bold v_{12}||} 
$$
再用一遍一遍三角形余弦定理即可推出上式。

第十步，余弦对顶点求导

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

不过我并没有看懂第23页的推导，也许和张量分析有关？不过我试着翻阅了基本有关张量分析的书，还是不知道最后的公式是怎么推出来的。而且我也查阅了一些有关微分几何的书，对于如何求余弦/角度/法向量对顶点的导数也是只字未提。我觉得去纯抄那些自己无法理解的公式毫无作用，所以暂时先放掉这一段。

总之，依据参考[2]第23页推导，可以得到如下的代码

```
	dc01dP0 = -b02 * b00.dot(v10) / v10.dot(v10);
	dc01dP2 = -b00 * b02.dot(v12) / v12.dot(v12);
	dc01dP1 = -dc01dP0 - dc01dP2;
	dc01dP3 = Vector3::Zero();

	dc02dP0 = -b01 * b00.dot(v20) / v02.dot(v20);
	dc02dP1 = b00 * b01.dot(v12) / v21.dot(v12);
	dc02dP2 = -dc02dP0 - dc02dP1;
	dc02dP3 = Vector3::Zero();

	dc11dP0 = Vector3::Zero();
	dc11dP2 = -b13 * b12.dot(v12) / v21.dot(v12);
	dc11dP3 = -b12 * b13.dot(v13) / v31.dot(v13);
	dc11dP1 = -dc11dP2 - dc11dP3;

	dc12dP0 = Vector3::Zero();
	dc12dP1 = b13 * b11.dot(v12) / v21.dot(v12);
	dc12dP3 = -b11 * b13.dot(v23) / v32.dot(v23);
	dc12dP2 = -dc12dP1 - dc12dP3;
```

第十一步，求角度对顶点的二阶导。这一步倒很简单。角度对顶点的一次导结果如下：

```
dThetadP0 = -n0 / d00;
dThetadP1 = c02 * n0 / d01 + c12 * n1 / d11;
dThetadP2 = c01 * n0 / d02 + c11 * n1 / d12;
dThetadP3 = -n1 / d13;
```

这里面的每个因子都能对顶点再求一遍导，因此我们只需要把这些分部求导就行了，部分代码如下

```
d2ThetadP0dP0 = -dn0dP0 / d00 + n0 * dd00dP0.transpose() / (d00 * d00);
d2ThetadP0dP1 = -dn0dP1 / d00 + n0 * dd00dP1.transpose() / (d00 * d00);
d2ThetadP0dP2 = -dn0dP2 / d00 + n0 * dd00dP2.transpose() / (d00 * d00);
d2ThetadP0dP3 = -dn0dP3 / d00 + n0 * dd00dP3.transpose() / (d00 * d00);

```

之后组装矩阵就像之前那个样子就像了。

