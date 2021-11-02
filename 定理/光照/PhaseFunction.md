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
