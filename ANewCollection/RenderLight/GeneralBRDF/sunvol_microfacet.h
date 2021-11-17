// https://github.com/sunwj/SunVolumeRender
// Created by 孙万捷 on 16/6/26.
//

#ifndef SUNVOLUMERENDER_MICROFACET_H
#define SUNVOLUMERENDER_MICROFACET_H

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include <curand_kernel.h>

#include "fresnel.h"
#include "../cuda_onb.h"

#define DISTRIBUTION_BECKMANN

__inline__ __device__ float beckmann_distribution(const glm::vec3& normal, const glm::vec3& wh, float alpha)
{
    auto cosTerm2 = glm::dot(normal, wh);
    cosTerm2 *= cosTerm2;

    auto pdf = expf((cosTerm2 - 1.f) / (alpha * alpha * cosTerm2)) / (float(M_PI) * alpha * alpha * cosTerm2 * cosTerm2);
    return pdf;
}

__inline__ __device__ float chiGGX(float v)
{
    return v > 0.f ? 1.f : 0.f;
}

__inline__ __device__ float GGX_distribution(const glm::vec3& normal, const glm::vec3& wh, float alpha)
{
    auto cosTerm = glm::dot(normal, wh);
    auto cosTerm2 = cosTerm * cosTerm;
    auto alpha2 = alpha * alpha;
    auto den = cosTerm2 * alpha2 + (1.f - cosTerm2);

    return alpha2 * chiGGX(cosTerm) / (float(M_PI) * den * den);
}

__inline__ __device__ float geometry_cook_torrance(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, const glm::vec3& wh)
{
    auto cosO = glm::dot(wo, wh);
    auto cosTerm = glm::dot(normal, wh);
    auto g1 = 2.f * cosTerm * glm::dot(normal, wo) / cosO;
    auto g2 = 2.f * cosTerm * glm::dot(normal, wi) / cosO;

    return fminf(1.f, fminf(g1, g2));
}

__inline__ __device__ float microfacet_brdf_f(const glm::vec3& wi, const glm::vec3& wo, const glm::vec3& normal, float ior, float alpha)
{
    // in different hemisphere
    if(glm::dot(wi, normal) * glm::dot(wo, normal) < 0.f) return 0.f;

    //else
    auto wh = glm::normalize(wi + wo);
    auto fresnelTerm = schlick_fresnel(1.f, ior, fabsf(glm::dot(wh, wo)));
    auto geometryTerm = geometry_cook_torrance(wi, wo, normal, wh);
#ifdef DISTRIBUTION_BECKMANN
    auto D = beckmann_distribution(normal, wh, alpha);
#else
    auto D = GGX_distribution(normal, wh, alpha);
#endif

    return fresnelTerm * geometryTerm * D / (4.f * fabsf(glm::dot(normal, wi)) * fabsf(glm::dot(normal, wo)));
}

__inline__ __device__ glm::vec3 sample_beckmann(const glm::vec3& normal, float alpha, curandState& rng)
{
    cudaONB onb(normal);
    float phi = 2.f * float(M_PI) * curand_uniform(&rng);

    float cosTheta = 1.f / (1.f - alpha * alpha * log(1.f - curand_uniform(&rng)));
    float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));

    return glm::normalize(sinTheta * cosf(phi) * onb.u + sinTheta * sinf(phi) * onb.v + cosTheta * onb.w);
}

__inline__ __device__ glm::vec3 sample_GGX(const glm::vec3& normal, float alpha, curandState& rng)
{
    cudaONB onb(normal);
    float phi = 2.f * float(M_PI) * curand_uniform(&rng);

    float u = curand_uniform(&rng);
    float tanTheta = alpha * sqrtf(u) / sqrtf(1.f - u);
    float theta = atanf(tanTheta);
    float sinTheta = sinf(theta);
    float cosTheta = sqrtf(fmaxf(0.f, 1.f - sinTheta * sinTheta));

    return glm::normalize(sinTheta * cosf(phi) * onb.u + sinTheta * sinf(phi) * onb.v + cosTheta * onb.w);
}

__inline__ __device__ void microfacet_brdf_sample_f(const glm::vec3& wo, const glm::vec3& normal, float alpha, glm::vec3* wi, float* pdf, curandState& rng)
{
#ifdef DISTRIBUTION_BECKMANN
    auto wh = sample_beckmann(normal, alpha, rng);
#else
    auto wh = sample_GGX(normal, alpha, rng);
#endif
    wh = glm::dot(wo, wh) >= 0.f ? wh : -wh;

    *wi = glm::reflect(-wo, wh);

#ifdef DISTRIBUTION_BECKMANN
    *pdf = beckmann_distribution(normal, wh, alpha) / (4.f * fabsf(glm::dot(wo, wh)));
#else
    *pdf = GGX_distribution(normal, wh, alpha) / (4.f * fabsf(glm::dot(wo, wh)));
#endif
}

#endif //SUNVOLUMERENDER_MICROFACET_H
