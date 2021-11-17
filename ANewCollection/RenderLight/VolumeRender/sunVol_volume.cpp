https://github.com/sunwj/SunVolumeRender
__host__ __device__ glm::vec3 GetRadiance() const {return 500.f * color * intensity * float(M_1_PI) / disk.GetArea();}

__inline__ __device__ glm::vec3 sample_light(const cudaAreaLight& light, const glm::vec3& volSamplePos, curandState& rng, glm::vec3* lightPos, glm::vec3* wi, float* pdf)
{
    // uniform sample a point on the light
    glm::vec2 localPos = uniform_sample_disk(rng, light.GetRadius());
    auto lightNormal = light.GetNormal(glm::vec3(glm::uninitialize));
    cudaONB localONB(lightNormal);

    glm::vec3 lightCenter = light.GetCenter();
    *lightPos = lightCenter + localONB.u * localPos.x + localONB.v * localPos.y;

    glm::vec3 shadowVec = *lightPos - volSamplePos;
    *wi = glm::normalize(shadowVec);

    float cosTerm = glm::dot(lightNormal, -(*wi));
    *pdf = glm::dot(shadowVec, shadowVec) / (fabsf(cosTerm) * light.GetArea());

    return cosTerm > 0.f ? light.GetRadiance() : glm::vec3(0.f);
}


__inline__ __device__ bool terminate_with_raussian_roulette(glm::vec3* troughput, curandState& rng)
{
    float illum = 0.2126f * troughput->x + 0.7152f * troughput->y + 0.0722 * troughput->z;
    if(curand_uniform(&rng) > illum) return true;
    *troughput /= illum;

    return false;
}

enum ShadingType{SHANDING_TYPE_ISOTROPIC, SHANDING_TYPE_BRDF};
__inline__ __device__ glm::vec3 bsdf(const VolumeSample& vs, const glm::vec3& wi, ShadingType st)
{
    glm::vec3 diffuseColor = glm::vec3(vs.color_opacity.x, vs.color_opacity.y, vs.color_opacity.z);

    glm::vec3 L;
    if(st == SHANDING_TYPE_ISOTROPIC)
    {
         L = diffuseColor * hg_phase_f(vs.wo, wi);
    }
    else if(st == SHANDING_TYPE_BRDF)
    {
        auto normal = glm::normalize(vs.gradient);
        normal = glm::dot(vs.wo, normal) < 0.f ? -normal : normal;

        float cosTerm = fmaxf(0.f, glm::dot(wi, normal));
        float ks = schlick_fresnel(1.0f, IOR, cosTerm);
        float kd = 1.f - ks;

        auto diffuse = diffuseColor * lambert_brdf_f(wi, vs.wo);
        auto specular = glm::vec3(1.f) * microfacet_brdf_f(wi, vs.wo, normal, IOR, ALPHA);

        L = (kd * diffuse + ks * specular) * cosTerm;
    }

    return L;
}

__inline__ __device__ glm::vec3 sample_bsdf(const VolumeSample& vs, glm::vec3* wi, float* pdf, curandState& rng, ShadingType st)
{
    if(st == SHANDING_TYPE_ISOTROPIC)
    {
        hg_phase_sample_f(PHASE_FUNC_G, vs.wo, wi, pdf, rng);
        return glm::vec3(vs.color_opacity) * hg_phase_f(vs.wo, *wi);
    }
    else if(st == SHANDING_TYPE_BRDF)
    {
        auto normal = glm::normalize(vs.gradient);
        auto cosTerm = glm::dot(vs.wo, normal);
        if(cosTerm < 0.f)
        {
            cosTerm = -cosTerm;
            normal = -normal;
        }

        auto ks = schlick_fresnel(1.f, IOR, cosTerm);
        auto kd = 1.f - ks;
        auto p = 0.25f + 0.5f * ks;

        if(curand_uniform(&rng) < p)
        {
            microfacet_brdf_sample_f(vs.wo, normal, ALPHA, wi, pdf, rng);
            auto f = microfacet_brdf_f(*wi, vs.wo, normal, IOR, ALPHA);
            return glm::vec3(1.f) * f * ks / p;
        }
        else
        {
            lambert_brdf_sample_f(vs.wo, normal, wi, pdf, rng);
            auto f = lambert_brdf_f(*wi, vs.wo);
            return glm::vec3(vs.color_opacity.x, vs.color_opacity.y, vs.color_opacity.z) * f * kd / (1.f - p);
        }
    }

    return glm::vec3(0.f);
}

__inline__ __device__ float sample_distance(const cudaRay& ray, const cudaVolume& volume, const cudaTransferFunction& tf, curandState& rng)
{
    float tNear, tFar;
    if(volume.Intersect(ray, &tNear, &tFar))
    {
        ray.tMin = tNear < 0.f ? 1e-6 : tNear;
        ray.tMax = tFar;
        auto t = ray.tMin;

        float sigmaMax = tf.GetMaxOpacity();
        float invSigmaMax = 1.f / sigmaMax;
        float invSigmaMaxSampleInterval = 1.f / (sigmaMax * BASE_SAMPLE_STEP_SIZE);
        while(true)
        {
            t += -logf(1.f - curand_uniform(&rng)) * invSigmaMaxSampleInterval;
            if(t > ray.tMax)
                return -FLT_MAX;

            auto ptInWorld = ray.PointOnRay(t);
            auto intensity = volume(ptInWorld);
            auto color_opacity = tf(intensity);
            auto sigma_t = color_opacity.w;

            if(curand_uniform(&rng) < sigma_t * invSigmaMax || t > ray.tMax)
                break;
        }

        return t;
    }

    return -FLT_MAX;
}

__inline__ __device__ float transmittance(const glm::vec3& start, const glm::vec3& end, const cudaVolume& volume, const cudaTransferFunction& tf, curandState& rng)
{
    cudaRay ray(start, glm::normalize(end - start));

    auto t = sample_distance(ray, volume, tf, rng);
    auto flag = (t > ray.tMin) && (t < ray.tMax);
    return flag ? 0.f : 1.f;
}

__inline__ __device__ glm::vec3 estimate_direct_light(const VolumeSample& vs, curandState& rng, ShadingType st)
{
    glm::vec3 Li = glm::vec3(0.f);

    if(num_areaLights == 0)
        return Li;

    // randomly choose a single light
    int lightId = num_areaLights * curand_uniform(&rng);
    lightId = lightId < num_areaLights ? lightId : num_areaLights - 1;
    const cudaAreaLight& light = areaLights[lightId];

    // sample light
    glm::vec3 lightPos;
    glm::vec3 wi;
    float pdf;
    Li = sample_light(light, vs.ptInWorld, rng, &lightPos, &wi, &pdf);

    if(pdf > 0.f && fmaxf(Li.x, fmaxf(Li.y, Li.z)) > 0.f)
    {
        auto Tr = transmittance(vs.ptInWorld, lightPos, volume, transferFunction, rng);
        Li = Tr * num_areaLights * bsdf(vs, wi, st) * Li / pdf;
    }
    else
        Li = glm::vec3(0.f);

    return Li;
}

__global__ void kernel_pathtracer(const RenderParams renderParams, uint32_t hashedFrameNo)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * WIDTH + idx;
    curandState rng;
    curand_init(hashedFrameNo + offset, 0, 0, &rng);

    glm::vec3 L = glm::vec3(0.f);
    glm::vec3 T = glm::vec3(1.f);

    cudaRay ray;
    camera.GenerateRay(idx, idy, rng, &ray);

    LightSample ls;
    bool hitLight = get_nearest_light_sample(ray, areaLights, num_areaLights, &ls);
    for(auto k = 0; k < renderParams.traceDepth; ++k)
    {
        auto t = sample_distance(ray, volume, transferFunction, rng);

        if((k == 0) && hitLight)
        {
            t = t < 0.f ? FLT_MAX : t;
            if(ls.t < t)
            {
                auto cosTerm = glm::dot(ls.normal, -ray.dir);
                L += T * ls.radiance * (cosTerm <= 0.f ? 0.f : 1.f);
                break;
            }
        }

        if(t < 0.f)
        {
            //L += T * envLight.GetEnvRadiance(ray.dir);
            break;
        }

        VolumeSample vs;

        vs.wo = -ray.dir;
        vs.ptInWorld = ray.PointOnRay(t);
        vs.intensity = volume(vs.ptInWorld);
        vs.color_opacity = transferFunction(vs.intensity);
        vs.gradient = volume.Gradient_CentralDiff(vs.ptInWorld);
        vs.gradientMagnitude = sqrtf(glm::dot(vs.gradient, vs.gradient));

        glm::vec3 wi;
        float pdf = 0.f;
        ShadingType st;

        auto gradientFactor = volume.GetGradientFactor();
        auto Pbrdf = vs.color_opacity.a * (1.f - expf(-25.f * gradientFactor * gradientFactor * gradientFactor * vs.gradientMagnitude * 65535.f * volume.GetInvMaxMagnitude()));
        if(curand_uniform(&rng) < Pbrdf)
            st = SHANDING_TYPE_BRDF;
        else
            st = SHANDING_TYPE_ISOTROPIC;

        L += T * estimate_direct_light(vs, rng, st);

        auto f = sample_bsdf(vs, &wi, &pdf, rng, st);
        float cosTerm = fabsf(glm::dot(glm::normalize(vs.gradient), wi));
        if(fmaxf(f.x, fmaxf(f.y, f.z)) > 0.f && pdf > 0.f)
        {
            if(st == SHANDING_TYPE_ISOTROPIC)
                T *= f / (pdf * (1.f - Pbrdf));
            else
                T *= f * cosTerm / (pdf * Pbrdf);
        }

        ray.orig = vs.ptInWorld;
        ray.dir = wi;

        if(k >= 3)
        {
            if(terminate_with_raussian_roulette(&T, rng))
                break;
        }
    }

    running_estimate(renderParams.hdrBuffer[offset], L, renderParams.frameNo);
}
