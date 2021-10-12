Physcially-based BRDFs

Helmholtz reciprocity
$$
f(\omega_i,\omega_o) = f(\omega_o,\omega_i)
$$
Positivity
$$
f(\omega_i,\omega_o) >= 0
$$
Energy Conservation
$$
\int_{\Omega}f(\omega_i,\omega_o)\cos\theta_od\omega_o \le 1
$$
per wavelength. the brdf f is not a scalar value.
$$
f^d(\omega_i,\omega_o) = \frac{k_d}{\pi} \qquad f^s(\omega_i,\omega_o) = k_s(r\cdot \omega_o)^s \qquad r = 2n(\omega_i \cdot n) - \omega_i
$$
Fresnel term
$$
F_{schlick}(h,w_o,F_0) = F_0 + ((1,1,1) - F_0)(1 - (\bold h \cdot \bold w_o))^5
$$
Normal Distribution Function
$$
NDF = \frac{\alpha^2}{\pi((\bold n \cdot \bold h)(\alpha^2-  1) + 1)^2}
$$
Geometry Function
$$
G = \frac{\bold n \cdot \bold v}{(\bold n \cdot \bold v)(1- k) + k}
$$

Heaviside piecewise function
$$
H(x) := \begin{cases} 1 , x>0 \\ 0 ,x \le 0\end{cases}
$$
Practical Multiple Scattering for Rough Surfaces  

This results in a masking 
$$
G_1(\bold o,\bold h) = H(\frac{\bold o \cdot \bold h}{\bold o \cdot \bold n})\min(1,2\frac{|\bold h \cdot \bold n||\bold o \cdot \bold n|}{|\bold o \cdot \bold n|})
$$
This geometry imposes a strong correlation between facets,so that a facet can only be shadow-masked by its adjacent-facing facet. This results in a masking term given by

where H(.) is the Heaviside function ensuring that backfacing microfacets are discarded. The G1 function models the masked visivle area of V-groove.

然后总能量损失by 遮罩和阴影at directions i and o can be computed from the geometric term
$$
G(\bold i ,\bold o,\bold h) = min(G_1(\bold i,\bold h),G_1(\bold o,\bold h))
$$
beckmann NDF
$$
P_B(\theta_r) = \frac{\alpha}{2\sqrt{\pi}}D_B(\bold s_r)\sin^2\theta_r
$$
GGX NDF
$$
P_G(\theta_r) = \frac{\alpha}{4}D_G(\bold s_r)\sin^2\theta_r
$$
phong NDF
$$
P_p(\theta_r) = \frac{\alpha_p + 2}{8 \sqrt{\pi}} \frac{\gamma((\alpha_p+1)/2)\gamma((\alpha_p+3)/2)}{\gamma((\alpha_p + 4)/2)}D_p(\bold s_r)\sin^2\theta_r
$$
Unfortunately,none of these distributions have an analytical form for the inverted CDF.

5.理论推断

- 每个facet 对应V-groove cavity的某一面
- V-groove cavity平行于物体的表面
- 光线的masking和Shadowing仅仅发生在一个cavity中，不会出现多个cavities中
- V-groove中的光线传输是完全specular中

那么与通常的微表面理论有两个关键的不同，传统的微表面理论如下

- 光线在离开V-groove前可多次反射
- V-groove不必沿着表面法线对称

Number of internal reflections

Moreover,an incident ray can only go through k - 1 or k reflections,depending on the incident angle theta_i,也就是
$$
k = [(\pi + 2\theta_i)/(\pi - 2|\theta_s|)] + 1
$$
outgoing direction
$$
\theta_{\bold o} = \begin{cases} (-1)^{k-1}(\theta_i + \pi - (k-1)(\pi - 2|\theta_s|)),(k-1)reflections \\ (-1)^k(\theta_i + \pi - k(\pi - 2|\theta_s|)),(k)reflections\end{cases}
$$
given th defintion of the brdf
$$
\rho(\bold i,\bold o) = \frac{dL_o(\bold o)}{L_i(i)|\bold i \cdot \bold n|d\bold i}
$$
and the projection factor
$$
|\frac{d \bold s}{d \bold o}| = \frac{\sin \theta_s}{k\sin \theta_h}\frac{1}{4\cos\theta_d}
$$
theta_h is the angle between h and n, and theta_d is the angle between h and i.

We can express the differential outgoing radiance dLo(o) reflected by a given facet in terms of the geometric attenuation term G as follows
$$
dL_o(\bold o) = \frac{d\Phi_{o,s}}{dA^{\perp}d\bold o} = \frac{d \bold s}{d\bold o}\frac{\Pi ^{k}_{j=1}FDG}{|\bold i \cdot \bold n||\bold o \cdot \bold n|}L_i(\bold i)(\bold i \cdot \bold n) d\bold i
$$
