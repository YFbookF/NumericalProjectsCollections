https://www.youtube.com/watch?v=kEcDbl7eS0w

1823 菲涅尔等式
$$
F(\theta) = \frac{F_s(\theta) + F_p(\theta)}{2}
$$
其中
$$
F_s(\theta) = \frac{a^2+b^2-2a\cos\theta+\cos^2\theta}{a^2+b^2+2\cos \theta + \cos^2\theta}
$$
而且
$$
F_p(\theta) = F_s(\theta)\frac{a^2 +b^2 - 2a\sin\theta \tan\theta + \sin^2\theta \tan^2\theta}{a^2 +b^2 + 2a\sin\theta \tan\theta + \sin^2\theta \tan^2\theta}
$$
Fresnel 在计算机图形学中
$$
\frac{D(\theta_h)G(\theta_i,\theta_o,\theta_h)F(\theta_h)}{\cos\theta_o}
$$
Schlick 近似
$$
F(\theta) \approx F_0 + (1-F_0)\frac{max(0,F'(\theta) - F'(0^o))}{1-F`(0^o)}
$$
变为
$$
F(\theta) \approx F_0 + (1 - F_0)(1 - \cos\theta)^5
$$
Lazanyi 近似
$$
F(\theta) \approx F_0 + (1 - F_0)(1 - \cos\theta)^5 - a\cos\theta(1 - \cos\theta)^\alpha
$$
请看视频教程，有曲线图更好理解

Artist Friendly metallic Fresnel
$$
\eta(F_0,g) = g\frac{1 - F_0}{1+F_0} + (1 - g)\frac{1+\sqrt{F_0}}{1+\sqrt{F_0}}
$$

$$
\kappa(F_0,\eta) = \sqrt{\frac{1}{1 - F_0}(F_0(\eta + 1)^2 - (\eta - 1)^2)}
$$

