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

