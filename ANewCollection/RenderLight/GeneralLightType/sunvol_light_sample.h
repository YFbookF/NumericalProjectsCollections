https://github.com/sunwj/SunVolumeRender//
// Created by 孙万捷 on 16/6/4.
//

#ifndef SUNVOLUMERENDER_LIGHT_SAMPLE_H
#define SUNVOLUMERENDER_LIGHT_SAMPLE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "../sampling.h"

struct LightSample
{
    float t = -1.f;
    glm::vec3 normal;
    glm::vec3 radiance;
};

__inline__ __device__ bool get_nearest_light_sample(const cudaRay& ray, cudaAreaLight* lights, const uint32_t n, LightSample* ls)
{
    auto tNear = FLT_MAX;
    auto t = FLT_MAX;
    int id = -1.f;

    for(auto i = 0; i < n; ++i)
    {
        if(lights[i].Intersect(ray, &t) && (t < tNear))
        {
            tNear = t;
            id = i;
        }
    }

    if(id != -1)
    {
        ls->t = tNear;
        ls->normal = lights[id].GetNormal(glm::vec3(glm::uninitialize));
        ls->radiance = lights[id].GetRadiance();

        return true;
    }

    ls->t = -FLT_MAX;
    return false;
}

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

#endif //SUNVOLUMERENDER_LIGHT_SAMPLE_H
