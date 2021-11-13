============pbrt section 12.1.1

Plank`s law gives the radiance emitted by a blackbody as a function of wavelength lambda and Temperature T measured in Kelvins
$$
L_e(\lambda,T) = \frac{2hc^2}{\lambda^5(\exp(hc/\lambda k_b T) - 1)}
$$

```
void Blackbody(const Float *lambda, int n, Float T, Float *Le) {
    if (T <= 0) {
        for (int i = 0; i < n; ++i) Le[i] = 0.f;
        return;
    }
    const Float c = 299792458;
    const Float h = 6.62606957e-34;
    const Float kb = 1.3806488e-23;
    for (int i = 0; i < n; ++i) {
        // Compute emitted radiance for blackbody at wavelength _lambda[i]_
        Float l = lambda[i] * 1e-9;
        Float lambda5 = (l * l) * (l * l) * l;
        Le[i] = (2 * h * c * c) /
                (lambda5 * (std::exp((h * c) / (l * kb * T)) - 1));
        CHECK(!std::isnan(Le[i]));
    }
}
```

而黑体的话，则有kirchoff law描述，它在说，任何频率下的radiance distribution，于黑体在那个频率下的radiance 乘上被物体吸收的incident radiance 那部分的比例。被物体系统的比例等于1 减去被反射的，也就是
$$
L'e(T,w,\lambda) = L_e(T,\lambda)(1 - \rho_{hd}(w))
$$
rho hd is the 半球方向的reflectance

hemispherocal - directional reflectance 给出的在给定方向上的所有reflection，这些reflection 是由于半球上的常数光照导致的，或者是由于光从指定方向上照射导致的

再来复习一下rho是什么

```
Spectrum BxDF::rho(const Vector3f &w, int nSamples, const Point2f *u) const {
    Spectrum r(0.);
    for (int i = 0; i < nSamples; ++i) {
        // Estimate one term of $\rho_\roman{hd}$
        Vector3f wi;
        Float pdf = 0;
        Spectrum f = Sample_f(w, &wi, u[i], &pdf);
        if (pdf > 0) r += f * AbsCosTheta(wi) / pdf;
    }
    return r / nSamples;
}
```

