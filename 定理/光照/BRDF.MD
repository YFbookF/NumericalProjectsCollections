Modeling Anisotropic Surface Reflectance with Example-Based Microfacet Synthesis  

![image-20211030150950714](D:\定理\光照\image-20211030150950714.png)

===========Crash Course in BRDF Implementation  

The BRDF describes reflectance of the surface for given combination of incoming and outgoing
light direction. In other words, it determines how much light is reflected in given direction when certain
amount of light is incident from another direction, depending on properties of the surface.  

对于粗糙的表面，BRDF几乎不反射，对于镜子，BRDF几乎全反射到一个方向。且BRDF不区分直接光和不直接光，也就是本地光和全局光。

Helmholtz reciprocity – the incoming and outgoing directions can be swapped (hence the name bidirectional), and the resulting reflectance will be the same. Note that some algorithms trace light rays originating from the camera (path tracing), other from the light sources (photon mapping) or both (bidirectional path tracing). Helmholtz reciprocity ensures consistent results for all types of algorithms  

Lambertian is one of the simplest BRDF functions used for diffuse term which assumes that incident light is scattered in all possible directions equally (within the hemisphere around the surface normal), but it is still a good approximation for behavior of many real-world materials and is very fast to evaluate.   

```
brdfLambertian = (diffuseReflectance / PI) * dot(N,L);
```



phong

```
brdfPhong = specular_reflectance * pow(dot(R, V), shininess) * dot(N, L);
```



 D Term – A microfacet distribution function – tells us what fraction of microfacets are oriented
in direction 𝐻 so that light incoming from direction L will be reflected in direction 𝑉.
 F Term – Fresnel term, evaluates how much light is reflected off the surface under given angle
of incidence.
 G Term – Geometric attenuation term (also masking and shadowing term denoted 𝐺ଶ later in
this text), accounts for mutual shadowing (and masking) of microfacets, sometimes also used
for normalization of BRDF.
 Denominator – comes from derivation of the microfacet model using perfect mirrors as
microfacets. Note that original paper by Cook and Torrance [30] uses the constant π instead
of 4 in denominator. Walter points out [32] this is due to different normalization of 𝐷 term,
and most recent literature agrees on using the constant 4.  

=============Notes on the Ward BRDF  

Another useful BRDF operation is bounding its maximum value over a set of directions. While this is a much less common operation than sampling, it is a powerful operation that is required by some rendering algorithms [Walter et al. 2005].   

The second component is a gaussian anisotropic gloss lobe defined by three parameters, ρs, αx, and αy, as:  
$$
f_r(\bold i,\bold o) = \frac{\rho_s}{4\pi \alpha_x\alpha_y\sqrt{(\bold i \cdot \bold n)(\bold o \cdot \bold n)}}exp(-\frac{((\bold h \cdot \bold x)/\alpha_x)^2 + ((\bold h \cdot \bold y) / \alpha_y)^2}{(\bold h \cdot \bold n)^2})
$$
rho_s controls the magnitude of the lobe, ax an ay control the width of the lobe in the two principal directions of anisotropy.

![image-20211031094159304](D:\定理\光照\image-20211031094159304.png)