本篇提示是关于Introduction to Classical Mechanics With Problems and Solutions by David Morin 的

section 6.1

变分法和传统牛顿法都能得到一样的结果，选择哪种全凭个人喜好。但是当问题中变量多于一个的时，写变分法更加容易，而用传统牛顿法都需要写下所有力的关系，而且容易搞混力的方向。而变分法只需要闭眼求导就行了。

section 6.2

*The path of a particle is the one that yields a stationary value of the action*

action 
$$
S = \int_{t_1}^{t_2}L(x,\dot x,t)dt
$$
法则6.5

如果
$$
L = \int_{x_1}^{x_2}f(y)\sqrt{1+y'^2}dx
$$
那么可得下面的偏微分方程
$$
1 + y'^2 = Bf(y)^2
$$
然后
$$
\frac{d}{dx}(f \cdot y' \cdot \frac{1}{\sqrt{1 + y'^2}}) = f'\sqrt{1 +y'^2}
$$
然后
$$
\frac{f'y'^2}{\sqrt{1 + y'^2}} + \frac{fy''}{\sqrt{1 + y'^2}} - \frac{fy'^2 y^{\prime\prime}}{\sqrt{1+y'^2}^{3/2}} = f'\sqrt{1 + y'^2}
$$
然后得到
$$
fy^{\prime\prime} = f'(1+y'^2) \qquad \frac{1}{2}\ln(1+y'^2)= \ln (f) + C \qquad 1+y'^2 = Bf(y)^2
$$
 section 7.2

假设有个极坐标上点，那么Lagrangian量
$$
\mathcal{L} = \frac{1}{2}m(\dot r^2 + r^2 \dot \theta^2) - V(r)
$$
分别微分
$$
m \ddot {r} = mr\dot \theta - V'(r) \qquad \frac{d}{dt}(mr^2\dot \theta) = 0
$$
那么可以得到
$$
L = mr^2 \dot \theta \qquad m\ddot r = \frac{L^2}{mr^3} - V'(r)
$$
那么得到energy，是积分得到的
$$
\frac{1}{2}m\dot r^2 + (\frac{L^2}{2mr^2} + V(r)) = E
$$
