Position-Free Monte Carlo Simulation for Arbitrary Layered BSDFs  

BSDF分为几个步骤，

第一步，采样

蒙特卡洛积分的基本形式
$$
\int_a^b f(x)dx \approx \frac{1}{N}\sum_{i=1}^N\frac{f(X_i)}{pdf(X_i)}
$$
路径追踪中的积分问题
$$
L_o(p,\omega_o) \approx \frac{1}{N}\sum_{i=1}^N\frac{L_i(p,w_i)f_r(p,w_i,w_o)(n\cdot w_i)}{p(\omega_i)}
$$
bsdf samplig，是根据概率密度分布函数，由ingoing方向绘制outgoing