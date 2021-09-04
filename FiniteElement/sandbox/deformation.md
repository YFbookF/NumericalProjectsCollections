$$
F = \frac{\partial \bold x}{\partial \bold X}
$$

相比于硅基生物，我们作为碳基生物其实比较擅长的是模块化的任务。想要算出上面的Deformation Gradient，那么我们先把它下部分算出来，也就是
$$
\frac{1}{\partial \bold X}
$$

```
import numpy as np
pos0 = np.array([0,0])
pos1 = np.array([1,0])
pos2 = np.array([0,2])
e10 = pos1 - pos0
e20 = pos2 - pos0
d_X = np.array([e10,e20])
minv = np.linalg.inv(d_X)
```

这应该就是动量的逆，在很多代码这个变量都叫做monunment ，取名叫minv。

接下来，我们来一个例子一个例子验证，不放过任何一个细节。毕竟如果随缘给物理参数的话，那么算出来的结果也就很随缘了。

比如下面的这个变换如何？算了一下确实是这样。

```
pos0_new = np.array([0,0])
pos1_new = np.array([0.707,0.707])
pos2_new = np.array([0.707,-0.707])
e10_new = pos1_new - pos0_new
e20_new = pos2_new - pos0_new
d_x = np.array([e10_new,e20_new])
defGrad = np.dot(minv,d_x)
init = np.dot(defGrad,pos2 - pos1)
```

http://www.continuummechanics.org/deformationgradient.html
$$
\bold F = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}
$$

$$
x = \bold X \cos \theta - \bold Y \sin \theta \\
y = \bold X \sin \theta + \bold Y \cos \theta
$$

拉伸，没什么

Shear 可以分为带旋转的Shear和纯Shear。后者是对称的。
$$
\bold F_{rotation} = \begin{bmatrix} 1 & 0 \\ 0.5  & 1\end{bmatrix} \qquad \bold F = \begin{bmatrix} 1 & 0.5 \\ 0.5  & 1\end{bmatrix}
$$
顺带一道思考题，如果要用一个标量值，来表明剪切力度的大小，你会怎么做呢？
$$
\begin{bmatrix} \end{bmatrix}
$$
顺便一提，Large Steps in Cloth Simulation 里面的w，我认为这玩意就tm是Deformation Gradient，所谓的duv矩阵，就是逆动量矩阵，也就是分母。而[dx]就是分子。而左侧的就是Deformation Gradient。

那么defGrad对x1求导的意义就是，当x1在某个方向上移动某个距离，defGrad对应的在相应位置的变化率。不论x1往x轴还是往y轴移动都是一样的。记得，

dwudx1 = dwudx_x = dwudx_y

而且wu和wv，或者说是defGrad的第一列和第二列都会被影响。如果是x轴，就影响第一行，如果是y轴，就影响第二行。

那么二阶导的意义也很容易理解了。

它虽然是这样的
$$
\begin{bmatrix}\Delta \bold x_{21x}  & \Delta \bold x_{31x} \\ \Delta \bold x_{21y}  & \Delta \bold x_{31y}\end{bmatrix} = \begin{bmatrix}\bold w_{ux} & \bold w_{vx} \\  \bold w_{uy}  &  \bold w_{vy}\end{bmatrix}\begin{bmatrix}\Delta u_1  & \Delta u_2 \\ \Delta v_1  & \Delta v_2\end{bmatrix}
$$
w矩阵和deformationGradient虽然不是同一个东西，但是非常相似，甚至可以写成下面这样
$$
(\partial \bold x)^T = \bold w (\partial \bold X)^T \qquad (\partial \bold x) = \bold F (\partial \bold X)
$$
所以很多有关defomrationGradient的知识可以直接套在w矩阵。

比如，如果我们想知道某个物体的不包括旋转部分的形变，也就是拉伸加纯剪切形变，可以用Right Cauchy Green Tensor
$$
\bold C = \bold F^T \bold F
$$
巧了，Large Steps In Cloth Simulation用的是一模一样的方法。不过注意Right Cauchy Green Tensor是个张量，而下面这个是个标量
$$
\bold C(\bold x) = \alpha \bold W_u^T\bold W_v
$$


如果你对带旋转的剪切和纯剪切不熟悉，可以看下面的链接

http://www.continuummechanics.org/deformationgradient.html

剩下的Stretch 和Bend就很理解了

这些讲义，说得非常棒，对于细节的解释也很好，但这仅仅限于在你写出代码后。如果你完全不知道怎么写代码的话，你也会发现你根本不知道这个公式在讲什么