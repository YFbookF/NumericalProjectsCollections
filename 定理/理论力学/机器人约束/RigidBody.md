===============redMax

The configuration of a rigid body is represented by the usual 4 × 4 transformation matrix consisting of rotational and translational components:  
$$
E = \begin{bmatrix}  R & \bold p \\ 0 & 1\end{bmatrix}
$$
p is the position of the frame`s origin expressed in Given a local position xi on a rigid body, its world position is x0
$$
\bold x_0 = \bold E \bold x_i
$$
旋转矩阵很重要的形式是
$$
RR^T = R^T R = 1
$$
所以
$$
\begin{bmatrix}  R & \bold p \\ 0 & 1\end{bmatrix}^{-1} = \begin{bmatrix}  R^T & -R^T \bold p \\ 0 & 1\end{bmatrix}
$$
The spatial velocity , also called a "twist" is composed of the angular component wi and the linear component vi , both expressed in body cooordinataes
$$
\phi = \begin{bmatrix} \bold w \\ \bold v\end{bmatrix}
$$
上面是6x1向量，但是我们可以将它们表示成下面的矩阵形式
$$
[\phi] = \begin{bmatrix} [w] & v \\ 0 & 0 \end{bmatrix}
$$
其中 [a]b = a x b，并且
$$
[\bold a] = \begin{bmatrix} 0 & -a_z & a_y \\ a_z & 0 & -a_x \\ -a_y & a_x & 0\end{bmatrix}
$$
我们可以这么写
$$
R = \exp([w]) \qquad [w] = \log(R) \qquad E = \exp([\phi]) \qquad [\phi] = \log(E)
$$
我们可以这么写
$$
\dot E \approx \frac{E(t + \Delta t) - E(t)}{\Delta t} \qquad E(t + \Delta t) = E(t)\exp(\Delta t \phi(t))
$$
![image-20211114211411380](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114211411380.png)

![image-20211114211421617](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114211421617.png)

Adjoint
$$
A = \begin{bmatrix} R & 0 \\ [\bold p] R & R\end{bmatrix}
$$
![image-20211114215246200](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114215246200.png)

![image-20211114215315996](E:\mycode\collection\定理\理论力学\机器人约束\image-20211114215315996.png)

