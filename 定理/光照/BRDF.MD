Modeling Anisotropic Surface Reflectance with Example-Based Microfacet Synthesis  

![image-20211030150950714](E:\mycode\collection\定理\光照\image-20211030150950714.png)

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

![image-20211031094159304](E:\mycode\collection\定理\光照\image-20211031094159304.png)

================pbrt

Torrance Sparrow BRDF
$$
f_r(w_o,w_i) = \frac{D(w_h)G(w_o,w_i)F_r(w_o)}{4\cos \theta_o \cos \theta_i}
$$
one of the nice things about the Torrance–Sparrow model is that the derivation doesn’t depend on the particular microfacet distribution being used. Furthermore, it doesn’t depend on a particular Fresnel function, so it can be used for both conductors and dielectrics.   

注意前一步是
$$
L(w_o) = \frac{F_r(w_o)L_i(w_i)D(w_h)}{4\cos \theta_o}dw_i
$$
很多BRDF 模型都没搞清楚的一点四，Fresnel 反射会导致故昂能到达物体底部的光线会变少。大概就是木板上的玻璃画，当从另一个角度看时，就看到完全不同的景色。因此衍生出 Fresnel Blend BRDF
$$
f_r(p,w_i,w_o) = \frac{28R_d}{23\pi}(1 - R_s)(1 - (1 - \frac{\bold n \cdot w_i}{2})^5)(1-(1-\frac{\bold n \cdot w_o}{2})^5)
$$
Examples include layered materials like metals with smooth or rough coatings or fabrics,
which are often partially retro-reflective.  可以存一个超大的4D查找表，但显然这并不是一个好主意。

====================Crash Course

```
    // compute V by reflecting L across H
    vec3 V = 2*dot(H,L)*H - L;

    vec3 b = BRDF( L, V, normal, tangent, bitangent );

    // apply N . L
    if (useNDotL != 0) b *= clamp(L[2],0,1);

    if (showChroma != 0) {
        float norm = max(b[0],max(b[1],b[2]));
        if (norm > 0) b /= norm;
    }

    // brightness
    b *= brightness;

    // exposure
    b *= pow( 2.0, exposure );

    // gamma
    b = pow( b, vec3( 1.0 / gamma ) );

    fragColor = vec4( clamp( b, vec3(0.0), vec3(1.0) ), 1.0 );
```

 eval evaluates the BRDF function for known incoming and outgoing directions
 sample samples BRDF to return a new (outgoing) ray direction
 pdf returns PDF of sampling given outgoing direction
 evalIndirect combines these functions in an optimized version  

BRDF大概就是算，从光源照到物体上的光线，有多大部分进入人的眼睛

====================LearnOpenGL

渲染方程是这么写的
$$
L_o(p,\omega_o) = \int\limits_{\Omega} 
    	(k_d\frac{c}{\pi} + \frac{DFG}{4(\omega_o \cdot n)(\omega_i \cdot n)})
    	L_i(p,\omega_i) n \cdot \omega_i  d\omega_i
$$
代码是这么写的

```
	for(int i = 0; i < 4; ++i) // 4就是灯光的数量
    	vec3 numerator    = NDF * G * F; 
        float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
        vec3 specular = numerator / denominator;
        // add to outgoing radiance Lo
        vec3 radiance = lightColors[i] * attenuation; // 灯光的颜色
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;  
    }   
    // ambient 应该就是自身的颜色
    vec3 ambient = vec3(0.03) * albedo * ao;

    vec3 color = ambient + Lo;

    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    FragColor = vec4(color, 1.0);
    

```

