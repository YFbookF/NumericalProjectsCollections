==============Directional Dipole Model for Subsurface Scattering  

BSSRDF模型定义为emergent radiance dL and an element of incident flux
$$
S(x_i,w_i,x_o,w_o) = \frac{dL_r(x_o,w_o)}{d\Phi(x_i,w_i)}
$$
BSSRDF一般分为三个项，reduced intensity (direct transmission)，单散射，以及多散射。但是单散射模拟并不准确，尤其是对translucent物体。

另一种方法是基于delta Eddington的单散射，此时the part of single scattering that continues along the refracted ray is moved to the reduced intensity term Se. The full BSSRDF in this approximation becomes
$$
S = T_{12}(S_d + S_{\delta E})T_{21}
$$
where T12 and T21 are the Fresnel transmittance terms at the locations where the radiance enters and exits the medium, respectively.  

==============Efficient Simulation and Rendering of Sub-surface Scattering  

Many different phase functions have been developed in the literature. The most commonly
used in computer graphics are the isotropic, the Henyey-Greenstein, the Schlick, the Rayleigh
and the Lorenz-Mie phase function. Because most materials contain distributions of particles of
many different sizes simple particle phase functions are not applicable. Thus, the material phase
function is described by an empirical formula introduced by Henyey and Greenstein (1941).   

Boundary The mirroring is performed above an extrapolated boundary above the surface where the fluence is zero, at the linear-extrapolated distance, zb = 2AD. This offset
takes into account the index of refraction mismatch at the boundary through the reflection
parameter:  
$$
A = \frac{1 + F_{dr}}{1- F_{dr}}
$$

```
 /* Average diffuse reflectance due to mismatched indices of refraction */
        m_Fdr = fresnelDiffuseReflectance(1 / m_eta);

        /* Dipole boundary condition distance term */
        Float A = (1 + m_Fdr) / (1 - m_Fdr);
```

其中Fdr，上面来源于Mitbusa

```
Float fresnelDiffuseReflectance(Float eta, bool fast) {
    if (fast) {
        /* Fast mode: the following code approximates the
         * diffuse Frensel reflectance for the eta<1 and
         * eta>1 cases. An evalution of the accuracy led
         * to the following scheme, which cherry-picks
         * fits from two papers where they are best.
         */
        if (eta < 1) {
            /* Fit by Egan and Hilgeman (1973). Works
               reasonably well for "normal" IOR values (<2).

               Max rel. error in 1.0 - 1.5 : 0.1%
               Max rel. error in 1.5 - 2   : 0.6%
               Max rel. error in 2.0 - 5   : 9.5%
            */
            return -1.4399f * (eta * eta)
                  + 0.7099f * eta
                  + 0.6681f
                  + 0.0636f / eta;
        } else {
            /* Fit by d'Eon and Irving (2011)
             *
             * Maintains a good accuracy even for
             * unrealistic IOR values.
             *
             * Max rel. error in 1.0 - 2.0   : 0.1%
             * Max rel. error in 2.0 - 10.0  : 0.2%
             */
            Float invEta = 1.0f / eta,
                  invEta2 = invEta*invEta,
                  invEta3 = invEta2*invEta,
                  invEta4 = invEta3*invEta,
                  invEta5 = invEta4*invEta;

            return 0.919317f - 3.4793f * invEta
                 + 6.75335f * invEta2
                 - 7.80989f * invEta3
                 + 4.98554f * invEta4
                 - 1.36881f * invEta5;
        }
    } else {
        GaussLobattoIntegrator quad(1024, 0, 1e-5f);
        return quad.integrate(
            boost::bind(&fresnelDiffuseIntegrand, eta, _1), 0, 1);
    }

    return 0.0f;
}

```

The Diffuse Reflectance Profike:

渲染表面需要计算光离开不同点，Fick Lwa which states that for isotropic sources, the vector flix E is the gradient of the fluence phi
$$
\vec E(r) = - D \vec{\nabla}\phi(r)
$$
In classical diffusion theory and due to ficks law the diffuse reflectance profile Rdr depends only on the surface flux (the gradient of the fluence along the surface normal) and not on the fulence iat selft.

这样就导出了radiant exitance 在边界上是vector flux和surface birnak
$$
R_d(r) = \vec{E}(r) \cdot n = - D (\vec {\nabla} \cdot \vec n) \phi(r)
$$
The directional derivative nabla n of the fluence in the direction of the nornal is needed to be evaluated. This gives
$$
R_d(r) = \frac{\alpha}{4\pi}(\frac{z_r(1 + \sigma_{tr}d_r e^{\sigma_{tr}d_r}}{d_r^3} + \frac{z_v(1 + \sigma_{tr}d_v e^{\sigma_{tr}d_v}}{d_v^3})
$$

```
   //mitbusa
   inline void operator()(const IrradianceSample &sample) {
        Spectrum rSqr = Spectrum((p - sample.p).lengthSquared());

        /* Distance to the real source */
        Spectrum dr = (rSqr + zr*zr).sqrt();

        /* Distance to the image point source */
        Spectrum dv = (rSqr + zv*zv).sqrt();

        Spectrum C1 = zr * (sigmaTr + Spectrum(1.0f) / dr);
        Spectrum C2 = zv * (sigmaTr + Spectrum(1.0f) / dv);

        /* Do not include the reduced albedo - will be canceled out later */
        Spectrum dMo = Spectrum(INV_FOURPI) *
             (C1 * ((-sigmaTr * dr).exp()) / (dr * dr)
            + C2 * ((-sigmaTr * dv).exp()) / (dv * dv));

        result += dMo * sample.E * sample.area;
    }

```

===================A BSSRDF Model for Efficient Rendering of Fur with Global Illumination  

while the bssrdf is often difficult to calculate, propesed a dipole method to multiple scattering part of it. 

The dipole method assumes local flatness putting a real point source beneath th incident xi and a virtual point source above it. 

![image-20211111145656538](E:\mycode\collection\定理\光照\image-20211111145656538.png)
$$
z_r = \frac{1}{\sigma_t'} \qquad z_v = -z_r(1 + 4A/3)
$$
are positive and negative z-coordinates of the rael and virtual point sources.
$$
d_r = \sqrt{r^2 + z_r^2} \qquad d_v = \sqrt{r^2 + z_v^2}
$$
are distance from xo to the sources
$$
\sigma_{tr} = \sqrt{3\sigma_a \sigma_{t}'}
$$
.

is the effective extinction coefficient。下面的代码来自mitsuba

```
m_sigmaSPrime = m_sigmaS * (Spectrum(1.0f) - m_g);
        m_sigmaTPrime = m_sigmaSPrime + m_sigmaA;

        /* Find the smallest mean-free path over all wavelengths */
        Spectrum mfp = Spectrum(1.0f) / m_sigmaTPrime;
        
                m_zr = mfp;
        m_zv = mfp * (1.0f + 4.0f/3.0f * A);
        m_sigmaTr = (m_sigmaA * m_sigmaTPrime * 3.0f).sqrt();
        /* Distance to the real source */
        Spectrum dr = (rSqr + zr*zr).sqrt();

        /* Distance to the image point source */
        Spectrum dv = (rSqr + zv*zv).sqrt();
```

================Directional Dipole for Subsurface Scattering in Translucent Materials  

it is common to split bssrdf into a single scattering term s1 and mutiple scattering term sd, such that
$$
S = T_{12}(S_d + S^{(1)})T_{21}
$$
====================directional sensor response function

```
//BSSRDF Explorer: A rendering framework for the BSSRDF
analytic radial no_wi no_wo

vec3 lu;
vec3 alphaPrime;
vec3 SigmaTr;
float A;
void init()
{
	float Fdr = -1.44/(IOR * IOR) + 0.71/IOR + 0.668 + 0.0636*IOR;
	A = (1.0 + Fdr) / (1 - Fdr);
	vec3 SigmaSPrime = SigmaS *(1.0 - G);
	vec3 SigmaTPrime = SigmaSPrime + SigmaA;
	SigmaTr = sqrt(3 * SigmaA * SigmaTPrime);
	Lu = 1.0 / SigmaTPrime;
}

vec3 DiffuseReflectance(vec3 AlphaPrime)
{
	vec3 T = sqrt(3.0 * (1.0 - AlphaPrime));
	return AlphaPrime * 0.5 * (1.0 - exp(-4/3*A*T)) * exp(-T);
}

vec3 Bssrdf(vec3 Xi, vec Wi, vec3 Xo,vec3 Wo)
{
	vec3 R = vec3(dot(Xi - Xo,Xi - Xo));
	vec3 Zr = Lu;
	vec3 Zv = Lu * (1.0 + 4.0 / 3.0 * A);
	vec3 Dr = sqrt(R + zr * zr);
	vec3 Dv = sqrt(R + zv * zv);
	vec3 C1 = Zr * (SigmaTr + 1.0 / Dr);
	vec3 C2 = Zv * (SigmaTr + 1.0 / Dv);
	vec3 FluenceR = C1 * exp(-SigmaTr * Dr) / (Dr * Dr);
	vec3 FluenceV = C2 * exp(-SigmaTr * Dv) / (Dv * Dv);
	return Multiplier / (4.0 * Pi) * AlphaPrime * (FluenceR + FluenceV);
}
```

==========BSSRDF Explorer: A rendering framework for the BSSRDF  

The distribution of phtotons exiting through the surface gives rise to a radially symmetric reflectance called reduced reflectance
$$
R_d(\vec x) =\int_{\Omega}L_r(\vec x,\vec w')f_t(\vec x,\vec w')d\vec w'
$$
ft is the bidirectional transmittance distrbution function BTDF.

In BSSRDF Sd cannot be constructed exactly from Rd, many methods in graphics approximate Sd using the reflectance distribution profile and a directionally dependent Fresnel transmission term
$$
S_d = \frac{1}{\pi}F_t(\vec x_i,\vec w_i) R_d(\vec x_o - \vec x_i)\frac{F_t(\vec x_o,\vec w_o)}{4C_{\phi}(1/\eta)}
$$
4 C phi is an approximate normalization factor

Diffusion Theory

In this setting, a useful simplificaton is to only consider angular integrals nth moments of the radiance. The first two moments of the radiance are denoted as fluence phi and flux E
$$
\phi(\vec x) = \int_{\Omega}L(\vec x,\vec w)d\vec w \qquad E(\vec x) = \int_{\Omega}L(\vec x,\vec w)\vec wd\vec w 
$$
The radiance is first expaned in spherical harmonics. The 1 order approximation of the radiance in terms of phi and E
$$
L(\vec x,\vec w) \approx \frac{1}{4\pi}\phi(\vec x) + \frac{3}{4\pi}\vec E(\vec x)\cdot \vec w
$$
将其替换到第一部分的radiative transport equation，并且在所有方向上积分
$$
-D \nabla^2 \phi(\vec x) + \sigma_\alpha \phi)(\vec x) = Q(\vec x)
$$
First Pass: Irradiance Sampling  

Given the set P of surface samples, the irradiance for each has to be computed. BRDF或者光子映射都可以。

Second Pass: BSSRDF Evaluation  

Given the set irradiance samples and an evaluation point x, we now need an efficient method for evaluating the radiant exitance at x0. 

====================PBRT-V4

bssrdf 的公式如下
$$
L_o(p_o,w_o) = \iint S(p_o,w_o,p_i,w_o)L_i(p_i,w_i)|\cos \theta_i|dw_i dA
$$


```
            // Account for attenuated subsurface scattering, if applicable
            if (isect.bssrdf && (flags & BSDF_TRANSMISSION)) {
                // Importance sample the BSSRDF
                SurfaceInteraction pi;
                Spectrum S = isect.bssrdf->Sample_S(
                    scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
                DCHECK(std::isinf(beta.y()) == false);
                if (S.IsBlack() || pdf == 0) break;
                beta *= S / pdf;

                // Account for the attenuated direct subsurface scattering
                // component
                L += beta *
                     UniformSampleOneLight(pi, scene, arena, sampler, true,
                                           lightDistribution->Lookup(pi.p));

                // Account for the indirect subsurface scattering component
                Spectrum f = pi.bsdf->Sample_f(pi.wo, &wi, sampler.Get2D(),
                                               &pdf, BSDF_ALL, &flags);
                if (f.IsBlack() || pdf == 0) break;
                beta *= f * AbsDot(wi, pi.shading.n) / pdf;
                DCHECK(std::isinf(beta.y()) == false);
                specularBounce = (flags & BSDF_SPECULAR) != 0;
                ray = pi.SpawnRay(wi);
            }
```

SeparableBSSRDF interface casts the BSSRDF into a separaable form with three independent component
$$
S(p_o,w_o,p_i,w_i) \approx (1- F_r(\cos \theta_o))S_pS_w
$$
=============Realistaic Image Synthesis

section 2.4.1

当光照到物体上的时候，它通常会进入物体，然后散射一下，然后离开物体。这种再marble和skin上很明显，因此称为transulent物体，但是对于非金属材质的话也有不同的次表面散射。

