https://github.com/ZackMisso/TransmittanceEstimation

To obtain an expression for transmittance T(a,b) = L(a)  / L(b)，可以计算
$$
-\frac{L(x)}{dx} = -\mu(x)L(x)
$$
获得下面的式子
$$
T(a,b) = \exp(-\int_{a}^b \mu(x)dx) = \exp(-\tau(a,b))
$$
对于homo来说很容易，但是对于heterogeneous media很难，

```
Float Expected::T(TransmittanceQuaryRecord& rec, Sampler* sampler) const {
    Float od = rec.extFunc->calculateExtinctionIntegral(rec.a, rec.b);
    rec.transmittance = exp(-od);
    return exp(-od);
}
```

optical thickness intergral tau 无法算出解析解，但是可以用经典的ray march方法算，

Track Length 方法很简单，如果在离开介质前撞到粒子，那么透光率就是零，否则就是1。

```
Float TrackLength::T(TransmittanceQuaryRecord& rec, Sampler* sampler) const {
    // Choose a constant k such that k >= ext(x) for all x[a,b]
    Float k = rec.extFunc->calculateMajorant((rec.b + rec.a) / 2.0);
    Float x = rec.a;
    x = 0.0;

    do {
        // Sample a random value y with pdf(y) = k * exp(-ky) set x = a + y
        x += sampleExpFF(rec, sampler, k);

        // If x > b stop and return 1
        if (x >= rec.b) {
            rec.transmittance = 1.0;
            return 1.0;
        }
        // Sample a uniform random number g[0,1]
        Float g = sampler->next1D();

        // If g <= ext(x) / k; stop and return 0
        if (g <= rec.extFunc->calculateExtinction(x, rec.extCalls) / k) {
            rec.transmittance = 0.0;
            return 0.0;
        }
    } while(true);
    // this should never be reached
    return 0.0;
}
```

很好

```
Float Estimator::sampleExpFF(const TransmittanceQuaryRecord& rec, Sampler* sampler, Float maj) const {
    return -log(1.0 - sampler->next1D()) / maj;
}
```

