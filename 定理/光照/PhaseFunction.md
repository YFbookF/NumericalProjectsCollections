Inverse Volume Rendering with Material Dictionaries  

p 是 phase function determing the amount of light that gets scattered towards each direction psi relative to the incident direction w. The phase function is often assumed to be invariant to rotations of the incident direction and cylindrically symmetric; therefore, it is a function of only θ = arccos (! · ) satisfying the normalization constraint  
$$
\theta = \arccos(\omega \cdot \psi) \qquad 2\pi \int_{\theta=0}^{\pi}p(\theta)\sin(\theta) d\theta = 1
$$
而由某个方向w进入的光线，最终散射到另一个方向w'的概率是P(w,w')
$$
P(\omega,\omega')
$$
The phase function Pr;l(q) characterizing the scattering
caused by a water droplet of size r at wavelength l is obtained by the Mie theory [BH98]. It is quite expensive to  compute, highly dependent on r and l, and its intensity oscillates strongly when q varies. This is what makes the Mie
function so impractical, and probably why it is avoided in

CG cloud simulations.

真正的相函数，取决于雨滴的尺寸等因素，很难计算。

======LIGHT REFLECTION FUNCTIONS FOR SIMULATION OF CLOUDS AND DUSTY SURFACES  

due to the symmetry of the particles this net brightness can be assumed to vary only as a function of the lighting direction from the point of view of the observer.   

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

===============pbrt

how light is scattered at a point in space, it is the volumetric analog to theBSDF  

描述了光线在空间点上是如何散射的  , 可以在空间上模拟BRDF

In a slightly confusing overloading of terminology, phase functions themselves can be
isotropic or anisotropic as well. Thus, we might have an anisotropic phase function in an
isotropic medium. An isotropic phase function describes equal scattering in all directions
and is thus independent of either of the two directions. Because phase functions are
normalized, there is only one such function:  

各项同性相函数只有下面一种
$$
p(w_o,w_i) = \frac{1}{4\pi}
$$
phase function have a normalization constraint,for all w, the condition
$$
\int p(w,w')dw' = 1
$$
henyey GreenStein
$$
p_{HG}(\cos \theta) = \frac{1}{4\pi}\frac{1 - g^2}{(1 + g^2 + 2g(\cos \theta))^{3/2}}
$$
g is parameter called asymmetry

算出啊的用于

```
Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta, Float r) {
    // Compute material parameters and minimum $t$ below the critical angle
    Float sigma_t = sigma_a + sigma_s, rho = sigma_s / sigma_t;
    Float tCrit = r * SafeSqrt(Sqr(eta) - 1);

    Float Ess = 0;
    const int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        // Evaluate single-scattering integrand and add to _Ess_
        Float ti = tCrit + SampleExponential((i + 0.5f) / nSamples, sigma_t);
        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        Float d = std::sqrt(Sqr(r) + Sqr(ti));
        Float cosTheta_o = ti / d;

        // Add contribution of single scattering at depth $t$
        Ess += rho * FastExp(-sigma_t * (d + tCrit)) / Sqr(d) *
               HenyeyGreenstein(cosTheta_o, g) * (1 - FrDielectric(-cosTheta_o, eta)) *
               std::abs(cosTheta_o);
    }
    return Ess / nSamples;
}

```

这个会被用于生成查找表，然后查找表在SubsurfaceFromDiffuse(用于传给invertCatmullRom中算rho，最终用于算Sr。Sr是什么？

