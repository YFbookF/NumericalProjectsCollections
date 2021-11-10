//https://github.com/mmp/pbrt-v4
SampledSpectrum SimplePathIntegrator::Li(RayDifferential ray, SampledWavelengths &lambda,
                                         Sampler sampler, ScratchBuffer &scratchBuffer,
                                         VisibleSurface *) const {
    // Estimate radiance along ray using simple path tracing
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = true;
    int depth = 0;
    while (beta) {
        // Find next _SimplePathIntegrator_ vertex and accumulate contribution
        // Intersect _ray_ with scene
        pstd::optional<ShapeIntersection> si = Intersect(ray);

        // Account for infinite lights if ray has no intersection
        if (!si) {
            if (!sampleLights || specularBounce)
                for (const auto &light : infiniteLights)
                    L += beta * light.Le(ray, lambda);
            break;
        }

        // Account for emissive surface if light was not sampled
        SurfaceInteraction &isect = si->intr;
        if (!sampleLights || specularBounce)
            L += beta * isect.Le(-ray.d, lambda);

        // End path if maximum depth reached
        if (depth++ == maxDepth)
            break;

        // Get BSDF and skip over medium boundaries
        BSDF bsdf = isect.GetBSDF(ray, lambda, camera, scratchBuffer, sampler);
        if (!bsdf) {
            isect.SkipIntersection(&ray, si->tHit);
            continue;
        }

        // Sample direct illumination if _sampleLights_ is true
        Vector3f wo = -ray.d;
        if (sampleLights) {
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(sampler.Get1D());
            if (sampledLight) {
                // Sample point on _sampledLight_ to estimate direct illumination
                Point2f uLight = sampler.Get2D();
                pstd::optional<LightLiSample> ls =
                    sampledLight->light.SampleLi(isect, uLight, lambda);
                if (ls && ls->L && ls->pdf > 0) {
                    // Evaluate BSDF for light and possibly add scattered radiance
                    Vector3f wi = ls->wi;
                    SampledSpectrum f = bsdf.f(wo, wi) * AbsDot(wi, isect.shading.n);
                    if (f && Unoccluded(isect, ls->pLight))
                        L += beta * f * ls->L / (sampledLight->p * ls->pdf);
                }
            }
        }

        // Sample outgoing direction at intersection to continue path
        if (sampleBSDF) {
            // Sample BSDF for new path direction
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = bsdf.Sample_f(wo, u, sampler.Get2D());
            if (!bs)
                break;
            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
            specularBounce = bs->IsSpecular();
            ray = isect.SpawnRay(bs->wi);

        } else {
            // Uniformly sample sphere or hemisphere to get new path direction
            Float pdf;
            Vector3f wi;
            BxDFFlags flags = bsdf.Flags();
            if (IsReflective(flags) && IsTransmissive(flags)) {
                wi = SampleUniformSphere(sampler.Get2D());
                pdf = UniformSpherePDF();
            } else {
                wi = SampleUniformHemisphere(sampler.Get2D());
                pdf = UniformHemispherePDF();
                if (IsReflective(flags) && Dot(wo, isect.n) * Dot(wi, isect.n) < 0)
                    wi = -wi;
                else if (IsTransmissive(flags) && Dot(wo, isect.n) * Dot(wi, isect.n) > 0)
                    wi = -wi;
            }
            beta *= bsdf.f(wo, wi) * AbsDot(wi, isect.shading.n) / pdf;
            specularBounce = false;
            ray = isect.SpawnRay(wi);
        }

        CHECK_GE(beta.y(lambda), 0.f);
        DCHECK(!IsInf(beta.y(lambda)));
    }
    return L;
}