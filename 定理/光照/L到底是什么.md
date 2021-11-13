=================pbrt-v3-master.code

L 主要用于确定像素的颜色

```
pixel.contribSum += L * sampleWeight * filterWeight;
```

L 由Li 决定

```
L = Li(ray, scene, *tileSampler, arena);
```

在Path.cpp中

```
PathIntegrator::Li(const RayDifferential &r, const Scene &scene,
```

有很多bounce。返回值是L。L首先是0。

第一步：如果是第一次bounce，那么首先加上Le

```
// Possibly add emitted light at intersection
        if (bounces == 0 || specularBounce) {
            // Add emitted light at path vertex or from the environment
            if (foundIntersection) {
                L += beta * isect.Le(-ray.d);
                VLOG(2) << "Added Le -> L = " << L;
            } else {
                for (const auto &light : scene.infiniteLights)
                    L += beta * light->Le(ray);
                VLOG(2) << "Added infinite area lights -> L = " << L;
            }
        }
```

Le 又分为两种，一种是SurfaceInteraction，另一种直接返回零，



```

Spectrum SurfaceInteraction::Le(const Vector3f &w) const {
    const AreaLight *area = primitive->GetAreaLight();
    return area ? area->L(*this, w) : Spectrum(0.f);
}

Spectrum InfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Spectrum(Lmap->Lookup(st), SpectrumType::Illuminant);
}

Spectrum Light::Le(const RayDifferential &ray) const { return Spectrum(0.f); }
```

那物体自身的颜色呢？注意了，物体呈现蓝色，是因为其它颜色光都被吸收了，就蓝色被反弹了啊啊啊。光越强，蓝色越明显。如果光是红色的，

似乎是直接乘上去？那么万一(1,0,0) x (0,0,1)应该等于多少？

```
accumulatedColor += segment.color * material.color * material.emittance;
```

第二步，算单光源1影响，

```
Spectrum Ld = beta * UniformSampleOneLight(isect, scene, arena,
                                                       sampler, false, distrib);
```

第三步，如果是bssrdf，继续算光源影响

```
Spectrum S = isect.bssrdf->Sample_S(
                scene, sampler.Get1D(), sampler.Get2D(), arena, &pi, &pdf);
            DCHECK(!std::isinf(beta.y()));
            if (S.IsBlack() || pdf == 0) break;
            beta *= S / pdf;

            // Account for the direct subsurface scattering component
            L += beta * UniformSampleOneLight(pi, scene, arena, sampler, false,
                                              lightDistribution->Lookup(pi.p));
```

UniformSampleOneLight 又调用 EstimateDirect，EstimateDirect 又调用 Sample_Li

Sample_Li 在干啥？不知道，书上说传入一个物体的点在空间中的坐标，那么Sample_Li 将会返回一个由于自己的灯光对那个点造成的影响。

```
Spectrum DiffuseAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                     Vector3f *wi, Float *pdf,
                                     VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    Interaction pShape = shape->Sample(ref, u, pdf);
    pShape.mediumInterface = mediumInterface;
    if (*pdf == 0 || (pShape.p - ref.p).LengthSquared() == 0) {
        *pdf = 0;
        return 0.f;
    }
    *wi = Normalize(pShape.p - ref.p);
    *vis = VisibilityTester(ref, pShape);
    return L(pShape, -*wi);
}
```

上面的L 用于算光线释是否遮挡

```
    Spectrum L(const Interaction &intr, const Vector3f &w) const {
        return (twoSided || Dot(intr.n, w) > 0) ? Lemit : Spectrum(0.f);
    }
```

Sample_Le 主要用于产生randomwalk光线

### SampleF

主要用于产生一个随机光线，然后算f和相应的pdf。