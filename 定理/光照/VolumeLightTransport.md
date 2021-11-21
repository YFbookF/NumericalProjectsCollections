Inverse Volume Rendering with Material Dictionaries  

Scattering occurs as light propagates through a medium and interacts with material structures. There are many volume events that
cause absorption or change of propagation direction. This process
has been modeled by the radiative transfer equation (RTE) [Chandrasekhar 1960; Ishimaru 1978]:  
$$
(\omega^T \nabla)L(x,w) = Q(x,w) - \sigma_tL(x,w) + \sigma_s\int_{S^2}p(w,\psi)L(x,\psi)d\mu(\psi)
$$
x is a point in the interior or boundary of the scattering medium

w,psi are points in the sphere of direction and mu is the usual spherical measure

Q accounts for emission from light source

L is the resulting light field radiance at every spatial location and orientation  

p 是 phase function determing the amount of light that gets scattered towards each direction psi relative to the incident direction w. The phase function is often assumed to be invariant to rotations of the incident direction and cylindrically symmetric; therefore, it is a function of only θ = arccos (! · ) satisfying the normalization constraint  
$$
\theta = \arccos(\omega \cdot \psi) \qquad 2\pi \int_{\theta=0}^{\pi}p(\theta)\sin(\theta) d\theta = 1
$$


![image-20211030154132749](D:\定理\光照\image-20211030154132749.png)

