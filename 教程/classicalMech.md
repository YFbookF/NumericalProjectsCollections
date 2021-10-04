linear momentum
$$
\bold p = m \bold v
$$
angular monentum
$$
\bold L = \bold r \times \bold p
$$
r is radius vector from O to the particle. We then define the moment of force or torque about O as
$$
\bold N = \bold r \times \bold F
$$
the equation analogous to 1.3 
$$
\bold r \times \bold F = \bold N = \bold r \times \frac{d}{dt}(m\bold v)
$$
equation 1.9 can be wriiten in a different form by using the vector identity
$$
\frac{d}{dt}(\bold r \times m \bold v) = \bold v \times m \bold v + \bold r \times \frac{d}{dt}(m\bold v)
$$
where the first term the right obviously vanished. In consequence
$$
\bold N = \frac{d}{dt}(\bold r\times m\bold v) = \dot {\bold L}
$$
if the total torque N is zero,then dLdt = 0,and angular momentum L is conserved.

the total angular momentum about O is
$$
\bold L = \bold R \times M \bold v + \sum_{i} \bold r'_i \times \bold p_i^{\prime}
$$
R是重心的世界坐标，M是质量，v是重心的世界速度。r是重心到粒子的向量，而v'是粒子的速度
$$
\bold p_i^{'} = mv_i^{\prime}
$$
the total kinetic energy of the system is
$$
T = \frac{1}{2}\sum_{i}m_i \bold v_i^2 = \frac{1}{2}\sum_i m_i(\bold v + \bold v_i^{\prime})(\bold v + \bold v_i^{\prime}) \\ = \frac{1}{2}\sum_i m_i v^2 + \frac{1}{2}\sum_i m_i v_i^{\prime 2} + \bold v \cdot \frac{d}{dt}(\sum_{i}m_i \bold r_i^{\prime})
$$
equ1.57
$$
\frac{d}{dt}(\frac{\partial L}{\partial \dot{q_j}}) - \frac{\partial L}{\partial q_j} = 0
$$
expression referred to as "Lagrange`s equations",where
$$
L = T - V \qquad F = -\nabla V
$$
T is the kinetic energy,V is the potential energy

for monogenic systems, Hamilton`s principle can be stated as
$$
\bold I = \int_{t_1}^{t_2}L dt
$$
变分法原理

https://www.zhihu.com/search?type=content&q=%E5%8F%98%E5%88%86%E6%B3%95

如下形式的定积分
$$
\bold I = \int_a^b f(y,y')dx
$$
如果y变化了dy，那么I变化为
$$
\delta I = \int_a^b [\frac{\partial f}{\partial y}\delta y + \frac{\partial f}{\partial y'}\delta y']dx
$$
第二项改写为如下
$$
\int_a^b\frac{\partial f}{\partial y'} =\int_a^b \frac{\partial f}{\partial y'}\delta y' dx  = \int_a^b \frac{\partial f}{\partial y'}d(\delta y) =\\ \frac{\partial f}{\partial y'}\delta y'|_b^a - \int_a^b \delta y \frac{d}{dx}(\frac{\partial f}{\partial y'})dx
$$
因此上式可简化为
$$
\delta I= \int_a^b [\frac{\partial f}{\partial y}- \frac{d}{dx}(\frac{\partial f}{\partial y'})]\delta y(x)dx
$$
如果I有极值，对任意满足边界条件的dy都有dI  = 0，也就是
$$
\frac{\partial f}{\partial y} - \frac{d}{dx}(\frac{\partial f}{\partial y'}) = 0
$$


也就是Euler Lagrange方程

例如最速降曲线，要求解的方程如下
$$
T = \frac{1}{\sqrt{2g}}\int_0^a \sqrt{\frac{1+y^2}{y}}dx
$$
那么取
$$
f(y,y') = \sqrt{\frac{1+y'^2}{y}}
$$
那么真正的Euler Lagrange就算
$$
\frac{\partial f}{\partial y} = -\frac{1}{2}\sqrt{\frac{1+y'^2}{y}} \qquad \frac{\partial f}{\partial y'} = \frac{y'}{\sqrt{y(1+y'^2)}}
$$
也就是最后要求这一堆东西
$$
\frac{1}{2}\sqrt{\frac{1+y^{'2}}{2}} + \frac{d}{dx}(\frac{y'}{\sqrt{y(1+y'^2)}}) = 0
$$
https://www.zhihu.com/search?type=content&q=%E5%8F%98%E5%88%86%E6%B3%95

mininum surface of revolution

the total strip area of the surface
$$
2\pi\int_1^2 x\sqrt{1+\dot y^2}dx
$$
令
$$
f = x\sqrt{1 + \dot {y}^2}
$$
那么
$$
\frac{\partial f}{\partial y} = 0 \qquad \frac{\partial f}{\partial y'} = \frac{xy'}{\sqrt{1+y'^2}}
$$
带回容易得到
$$
\frac{d}{dx}(\frac{xy'}{\sqrt{1+y'^2}}) = 0
$$
也就是
$$
\frac{xy'}{\sqrt{1+y'^2}} = const
$$
也就是
$$
x^2y'^2 = c^2 + c^2y'^2
$$
也就是
$$
y' = \frac{c}{x^2 - c^2}
$$
也就是
$$
y = a\int\frac{1}{\sqrt{x^2 - a^2}}dx = a arc \cosh \frac{x}{a} + b 
$$
![image-20211002210101541](D:\图形学书籍\系列流体文章\gif\image-20211002210101541.png)

又比如

![image-20211002212045513](D:\图形学书籍\系列流体文章\gif\image-20211002212045513.png)

The kinetic energy
$$
T = \frac{1}{2}M\dot{x}^2 + \frac{1}{2}Mr^2 \dot{\theta}^2
$$
The potential energy is
$$
V = Mg (l - x)\sin \phi
$$
where l is length of the inclined plane and the Lagrangian is
$$
L = T - V 
$$
实际问题2，

a hoop rolls down the incline with only one-half the acceleration it would have slipping down a frectionless plane

一个箍沿着斜坡滚下，加速度只有它在无摩擦平面上滑下的加速度的一半
$$
L= T - V = \frac{M \dot{x}^2}{2} + \frac{Mr^2\dot{\theta}^2}{2} - Mg(l-x)\sin\theta
$$
限制为
$$
dx - rd\theta = 0
$$
那么The two Lagrange equations therefore are
$$
M \ddot{x} - Mg\sin \phi + \lambda = 0 \qquad Mr^2\ddot{\theta} - \lambda r = 0
$$
又由于
$$
r\ddot{\theta} = \ddot{x} \qquad M\ddot{x} = \lambda
$$
那么
$$
\ddot{x} = \frac{g\sin\phi}{2}
$$
