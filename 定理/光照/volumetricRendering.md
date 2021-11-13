==========BSSRDF Explorer: A rendering framework for the BSSRDF  

Volumetric Scattering  

吸收，吸收的radiance会被转换为热量，但我们可以直接让它消失即可。x is the current position along the ray, w is the direction ray, and radiance ray
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = -\sigma_a L(\vec x,\vec w)
$$
Emission，介质自己会发光。
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = Q(\vec x,\vec w)
$$
OutScattering. At each step, the radiance may also be reduced due to light be scattered into other directions than w
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = -\sigma_s L(\vec x,\vec w)
$$
InScattering 其它方向的光也有可能照射到这个方向上来
$$
(\vec w \cdot \nabla)L(\vec x,\vec w) = \int_{\Omega}p(\vec w',\vec w)L(\vec x,\vec w')d\vec w'
$$
p is phase function. The phase function describes the angular distribution of light intensity being scattering.

A simple, but naive and inefficient implementation of the trace function is given as follows. It
is a straightforward application of Monte Carlo to the RTE:  

\1. Start with a position ~x and direction ~!
\2. Pick a random distance s to travel before the next scattering event
\3. Advance by s along ~! and pick a new random direction ~!0 on the sphere
\4. Modify the path weight by extinction e-σts, scattering coefficient σs and phase function
p(~! · ~!0)
\5. Repeat steps 2-4 until the surface is hit
\6. Modify the path weight by the transmission term ft  

但这种方法并不是很好，尤其是对高吸收率的材料的时候，需要超大样本数才能收敛。

=============Volumetric Path Tracing   Steve Marschner  Cornell University  CS 6630 Fall 2015, 29 October  

用蒙特卡洛法好是好，但是对于inhomogeneous 或者高散射的物体来说，就好了。

The intergral form of the volume rendering volume

![image-20211111201459891](E:\mycode\collection\定理\光照\image-20211111201459891.png)

![image-20211111201423560](E:\mycode\collection\定理\光照\image-20211111201423560.png)

如果只有Emission，那么tau = 1，那么sigmas 和 sigmaa = 0
$$
L(x,w) = \int_{x}^y \varepsilon(x',w)dx'
$$
If we use uniform sampling with respect to distance along the ray, we have the estimator
$$
g = \varepsilon(x',w)||x - y||
$$
如果只有Homogeneous 的 absorption，那么

vareps = sigmas = 0，并且
$$
L(\bold x,w) = \tau(\bold x,\bold y)L_e(\bold y,w)
$$
没有积分，也无需蒙特卡洛。

只有Emission和homogeneous absorption，那么我们暂且还能积分
$$
L(x,w) = \int_x^y \tau(x,x')\varepsilon(x',w)dx' + \tau(x,y)L_e(y,w)
$$
for uniform sampling，
$$
g = ||x - y||exp(-\sigma_n||x - x'||)\varepsilon(x',w)
$$
但是如果有衰减的话，我们就不能统一采样。但是我们可以重要性采样衰减。homogeneous介质的衰减是
$$
\tau(s) = \exp(-\sigma_t s)
$$
where s is arc length along the ary。我们需要找到一个pdf，与tau(s)。我们只需要归一化就行了，就得到了pdf
$$
p(s) = \frac{\tau(s)}{\int_0^{\infin}\tau(t)dt} = \frac{\exp(-\sigma_t s)}{-\exp(-\sigma_t t/\sigma_t)}_{t=0} = \sigma_t \exp(-\sigma_t s)
$$
然后算cumulative distribution function cdf
$$
P(s) = \int_0^s p(s')ds' = (-\exp(-\sigma_t s')_0^s) = 1 - \exp(-\sigma_t s)
$$
然后不知道在算什么东西
$$
\xi = 1 - \exp(-\sigma_ts) \qquad s = -\frac{\ln(1-\xi)}{\sigma_t} \qquad s = -\frac{\ln s}{\sigma_t}
$$

```
function radiance_estimator(x,w)
	xi = rand();
	s = - ln(xi)/sigma_t
	if s < s_max
		return vareps * (x - s * w) / sigma_t
	return L
```

======================pbrt

要算的积分式子
$$
L_i(p,w) = T_r(p_0 \rightarrow p)L_0(p_0,-w) + \int_0^t T_r(p+tw \rightarrow p)L_s(p+ tw,-w)dt
$$
p = p + tmax w 就是撞击点。这时候有两种情况，首先如果不采样[0,tmax]直接，那么我们只需要推算第一项。

我们假设p(t)是光线前进一个单位距离时碰到的粒子程度，
$$
p_{surf} = 1 - \int_0^{tmax} p_t(t)dt
$$
同时sigma_t 是衰减因子，随波长变化。
$$
p_{surf} = \frac{1}{n}\sum_{i=1}^n \exp(-\sigma_t^i t_{max})
$$

```
// Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);

```

采样完后是一个数字，最终被作为因子，乘上采样灯光的系数上去了

```
// Sample the participating medium, if present
        MediumInteraction mi;
        if (ray.medium) beta *= ray.medium->Sample(ray, sampler, arena, &mi);
        if (beta.IsBlack()) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (bounces >= maxDepth) break;

            ++volumeInteractions;
            // Handle scattering at point in medium for volumetric path tracer
            const Distribution1D *lightDistrib =
                lightDistribution->Lookup(mi.p);
            L += beta * UniformSampleOneLight(mi, scene, arena, sampler, true,
                                              lightDistrib);

            Vector3f wo = -ray.d, wi;
            mi.phase->Sample_p(wo, &wi, sampler.Get2D());
            ray = mi.SpawnRay(wi);
            specularBounce = false;
        } else {
```

注意sigma t是在一个特殊的spectral channel 里面。而sigma_a 是吸收 absorb，而sigma_s 是发射 scattering，它俩之和sigma t也就是transmittance 是固定的，注意Transmittance 是透光率

就比如Grid Distance

```
while (true) {
        t -= std::log(1 - sampler.Get1D()) * invMaxDensity / sigma_t;
        if (t >= tMax) break;
        if (Density(ray(t)) * invMaxDensity > sampler.Get1D()) {
            // Populate _mi_ with medium interaction information and return
            PhaseFunction *phase = ARENA_ALLOC(arena, HenyeyGreenstein)(g);
            *mi = MediumInteraction(rWorld(t), -rWorld.d, rWorld.time, this,
                                    phase);
            return sigma_s / sigma_t;
        }
    }
```

$$
t_i = t_{i-1} - \frac{\ln(1 - \xi)}{\sigma_t}
$$

当t > tmax，说明光已经离开了介质，此时返回的透光率是1。Density 用的是三线性插值，也就是本地粒子的密度。Get1D是个随机。

注意光线强度随指数衰减，那么，那么直接算个1/sigma_T就行了吗？可能它不是无限远，仅仅是从0积分到t

而且sigma_t 以及density 是数组，需要把各个波长的衰减都考虑进去
$$
t = -\frac{\ln(1 - \xi)}{\sigma_t} \qquad p_t(t) = \sigma_te^{-\sigma_t t}
$$


```
 bool sampledMedium = t < ray.tMax;
    if (sampledMedium)
        *mi = MediumInteraction(ray(t), -ray.d, ray.time, this,
                                ARENA_ALLOC(arena, HenyeyGreenstein)(g));

    // Compute the transmittance and sampling density
    Spectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * ray.d.Length());

    // Return weighting factor for scattering from homogeneous medium
    Spectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
    // 注意这是个float，意味着我们平均了各个波长
    Float pdf = 0;
    for (int i = 0; i < Spectrum::nSamples; ++i) pdf += density[i];
    pdf *= 1 / (Float)Spectrum::nSamples;
    if (pdf == 0) {
        CHECK(Tr.IsBlack());
        pdf = 1;
    }
    return sampledMedium ? (Tr * sigma_s / pdf) : (Tr / pdf);
```

