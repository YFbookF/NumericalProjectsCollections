=============pbrt

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

==================Realistic Image Synthesis

The quantify the amount of light reflected by a surface we can use the ratio of reflected to incident，也就是打到表面的光线，被反射的比例，公式2.21
$$
\rho(x) = \frac{d\Phi_r(x)}{d\Phi_i(x)} = \frac{\int_{\Omega}\int_{\Omega}f_r(x,\vec w',\vec w')L_i(x,\vec w')d\vec w' d\vec w}{\int_{\Omega}L_i(x,\vec w')d\vec w'}
$$
